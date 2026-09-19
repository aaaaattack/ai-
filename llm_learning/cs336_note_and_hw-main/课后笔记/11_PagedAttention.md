# 第11课：PagedAttention + Prefix Caching

> 学习路线：阶段3 推理引擎核心（nano-vllm engine/）
> 对应文件：`nano-vllm/nanovllm/engine/block_manager.py`

---

## 一、传统 KV Cache 的浪费

### 1.1 连续内存分配

传统方式：每个 seq 预分配一段连续显存，容量 = max_seq_len × KV 大小。

```
seq1 (实际 2000 tokens): ████████████████████████░░░░░░░░  预分配 2048，浪费 48
seq2 (实际 1000 tokens): ██████████░░░░░░░░░░░░░░░░░░░░░░  预分配 2048，浪费 1048
seq3 (实际  500 tokens): █████░░░░░░░░░░░░░░░░░░░░░░░░░░  预分配 2048，浪费 1548

总利用率 ≈ 57%，浪费 ≈ 43% 显存
```

### 1.2 分页化的 KV Cache

受操作系统**虚拟内存分页**启发，把 KV Cache 切成固定大小的 block：

```
seq1 (2000 tokens): [B0][B1][B2][B3][B4][B5][B6][B7]    每 block=256 tokens
seq2 (1000 tokens): [B8][B9][B10][B11]                   用多少分多少
seq3 ( 500 tokens): [B12][B13]

零浪费，block 用完就回收给 free pool
```

**类比**：

```
操作系统页表:       虚拟地址 → 物理页框        (MMU 管理)
PagedAttention:    逻辑 token → 物理 KV Cache block  (BlockManager 管理)
```

这就是 PagedAttention 的名字由来。

---

## 二、Block — 一个 KV Cache 页

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id        # 物理 block 编号 (0, 1, 2, ...)
        self.ref_count = 0              # 引用计数：几个 seq 正在用这个 block
        self.hash = -1                  # 内容的 xxhash64（prefix caching 用）
        self.token_ids = []             # 这个 block 包含的 token IDs
```

**关键方法**：

```python
    def update(self, hash, token_ids):
        self.hash = hash
        self.token_ids = token_ids      # prefill 完成后更新 hash 和 content

    def reset(self):
        self.ref_count = 1              # 新分配时初始 ref_count=1
        self.hash = -1
        self.token_ids = []
```

**`ref_count` 是什么？** 一个 block 可能被多个 seq 共享（prefix caching）。每多一个 seq 引用，ref_count +1。归零才回收。

---

## 三、BlockManager — 分页管理器

### 3.1 数据结构

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size                              # 每 block 的 token 数
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有 block

        self.hash_to_block_id: dict[int, int] = {}   # hash → block_id 映射表
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲列表
        self.used_block_ids: set[int] = set()                       # 已使用集合
```

**三种管理结构**：

```
blocks:            [Block0, Block1, ..., BlockN-1]   (固定数组)
hash_to_block_id:  {hash_0: 0, hash_1: 5, ...}       (快速查找缓存)
free_block_ids:    deque([7, 8, 9, ...])              (O(1) 取空闲 block)
used_block_ids:    set({0, 1, 2, 3, 4, 5, 6})        (O(1) 检查是否在使用)
```

### 3.2 分配：`allocate()`

```python
def allocate(self, seq, num_cached_blocks):
    assert not seq.block_table              # 只能分配一次
    h = -1

    # Phase 1: 已缓存的 block（prefix cache 命中）
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]

        if block_id in self.used_block_ids:
            block.ref_count += 1            # 已经在用 → 共享，引用+1
        else:
            block.ref_count = 1             # 第一次用 → 引用=1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)

        seq.block_table.append(block_id)

    # Phase 2: 新 block（从 free pool 分配）
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())

    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

**`_allocate_block()`**：

```python
def _allocate_block(self):
    block_id = self.free_block_ids.popleft()  # 从左侧取（FIFO）
    block = self.blocks[block_id]
    assert block.ref_count == 0               # 必须没人用

    if block.hash != -1:
        del self.hash_to_block_id[block.hash] # 如果以前被缓存过，清理旧 hash

    block.reset()                             # 重置为初始状态
    self.used_block_ids.add(block_id)
    return block_id
```

### 3.3 释放：`deallocate()`

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):    # 从后往前释放
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:                  # 没人引用了 → 回收
            self._deallocate_block(block_id)

    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**`_deallocate_block()`**：

```python
def _deallocate_block(self, block_id):
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)    # 归还到 free pool（放右侧）
```

### 3.4 Decode 追加：`can_append()` / `may_append()`

```python
def can_append(self, seq) -> bool:
    # 只在"刚好填满一个 block"时检查
    # len(seq) % block_size == 1: 当前 token 是新 block 的第一个 token
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq):
    if len(seq) % self.block_size == 1:     # 需要新 block
        seq.block_table.append(self._allocate_block())
    # 否则：当前 block 还有空间，直接追加（block_table 不变）
```

**Decode 的 block 增长过程**：

```
seq 有 510 tokens, block_size=256:
  block_table = [B0, B1]         (B0=B1 都满了，256+254)
  生成 token 511 → len=511, 511%256=255 → 不需要新 block
  生成 token 512 → len=512, 512%256=0   → 不需要（仍在 B1 末尾）
  生成 token 513 → len=513, 513%256=1   → 触发 may_append → block_table = [B0, B1, B2]
```

---

## 四、Prefix Caching（前缀缓存）

### 4.1 问题

多个请求可能有相同前缀：

```
seq1: "请翻译：Hello world"       → tokens: [0,1,2, 3,4,5,6,7,8]
seq2: "请翻译：Good morning"      → tokens: [0,1,2, 9,10,11,12,13]
seq3: "请翻译：How are you"       → tokens: [0,1,2, 14,15,16,17,18]
                                        ↑ 前 3 个 token 完全一样
```

传统做法：每个 seq 从零开始 prefill → 浪费 3 个 token 的 KV Cache × N 个请求 → O(N) 浪费。

Prefix Caching：共享相同前缀的 KV Cache block。

### 4.2 链式哈希

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))    # 链上前一个 block 的 hash
    h.update(np.array(token_ids).tobytes())         # 本 block 的 token
    return h.intdigest()
```

**为什么是链式的？**

```
普通 hash:       H(token_ids_B0)                   → 独立
链式 hash:       H(-1, token_ids_B0)               → h0
                 H(h0, token_ids_B1)               → h1     ← 依赖 B0
                 H(h1, token_ids_B2)               → h2     ← 依赖 B0, B1

前缀匹配要求从第一个 block 开始连续匹配。
如果 B0 不匹配 → B1 和 B2 的链式 hash 一定不同（因为 h0 不同）。
→ 一次不匹配，后面都不需要检查了。
```

### 4.3 Cache 命中检查：`can_allocate()`

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks

    for i in range(seq.num_blocks - 1):    # 最后一个 block 不缓存（不完整）
        token_ids = seq.block(i)           # seq 第 i 个 block 的 tokens
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)

        if block_id == -1:                                  # hash 不存在
            break
        if self.blocks[block_id].token_ids != token_ids:   # hash 碰撞
            break

        num_cached_blocks += 1
        if block_id in self.used_block_ids:     # 已在其他 seq 使用
            num_new_blocks -= 1                 # 不需要新分配

    if len(self.free_block_ids) < num_new_blocks:
        return -1                              # 空闲 block 不够
    return num_cached_blocks
```

**为什么最后一个 block 不缓存？**

```
block_size=256, prompt=600 tokens:
  B0: 256 tokens (完整, 固定)   → 可以缓存
  B1: 256 tokens (完整, 固定)   → 可以缓存
  B2: 88  tokens (不完整)       → 不缓存（不同 seq 的最后一个 block 长度可能不同）
```

### 4.4 Prefill 完成后更新缓存

```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size

    if start == end: return    # 没有一个完整的 block → 不更新

    h = self.blocks[seq.block_table[start-1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)      # 链式哈希
        block.update(h, token_ids)                # 更新 block 的 hash 和 token_ids
        self.hash_to_block_id[h] = block.block_id  # 登记到全局 hash 表
```

**调用时机**：Scheduler 的 `postprocess()` 中，每次 step 后调用。

---

## 五、完整数据流

```
新请求 seq 到达
  │
  ▼
can_allocate(seq) → 计算 num_cached_blocks
  检查 prefix cache 命中数 + 空闲 block 是否够
  │
  ▼
allocate(seq, num_cached_blocks)
  Phase 1: 命中的 block → ref_count += 1 (多个 seq 共享)
  Phase 2: 新 block → _allocate_block() (从 free pool 取)
  │
  ▼ seq.block_table = [0, 1, 7, 8, 9]
  (B0, B1 是 prefix cache 命中, B7-B9 是新分配的)
  │
  ▼ prefill 完成 → hash_blocks(seq)
  把新 block 的 token_ids 和 hash 登记到全局 hash_to_block_id
  │
  ▼ deckle 循环 → may_append(seq)
  新 token 填满当前 block → 分配下一个 block
  │
  ▼ seq 结束
  deallocate(seq)
  每个 block 的 ref_count -= 1
  ref_count=0 → 回收给 free pool
```

---

## 六、白话总结

```
PagedAttention:
  KV Cache 不是一条大鱼 → 是切成 256 token 一份的"寿司卷"
  每人按需取用，用完还给公共盘子

  seq.block_table = [0, 3, 7]  ← 物理 block ID
  就像页表: 虚拟页 0,1,2 → 物理页 0,3,7

Prefix Caching:
  前缀相同的请求共享 KV Cache block
  "请翻译：" 这个前缀被 1000 个请求共享 → 只存一份 KV Cache
  省显存 + 省 prefill 时间（不需要重复计算相同的 K 和 V）

引用计数:
  多少人用 → ref_count
  人走茶凉 (ref_count=0) → 回收碗筷
```

---

## 七、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **PagedAttention** | 你们的 TCIM Runtime 固定 shape → KV Cache 是连续内存 → 不支持分页。如果要支持，需要 HAL 支持 block_table 的非连续内存访问 |
| **Prefix Caching** | 基于 token 内容的哈希匹配 → 需要驱动支持"根据 token 内容查找已缓存的 KV block" → 软件层可做 |
| **推理工具链** | nano-vllm 的 block_manager 是纯 Python/CPU → 不影响 NPU 的计算。但 block_table 传递给 model_runner 后，必须翻译成硬件能理解的非连续内存描述符 |
| **量化** | KV Cache 量化后，block 的内容是 INT8 → 哈希需要基于量化后的值还是量化前的 token_ids？nano-vllm 用 token_ids 做 hash，绕过这个问题 |

---

## 八、思考题

1. **为什么 prefix caching 的哈希是链式的，而不是独立的？**
   - 提示：前缀必须从头开始连续匹配，链式哈希天然保证这个约束

2. **`deallocate` 的循环为什么从后往前（`reversed`）？**
   - 提示：从前往后释放可能导致中间的 block 已被后端的其他 seq 引用，ref_count 减错

3. **如果多个 seq 共享 block，其中一个 seq 的 ref_count 还没归零，block 的 KV Cache 内容是否安全？**
   - 提示：共享 block 的 KV Cache 内容是只读的（prefill 后就不再修改），所以安全

4. **你们的 TCIM Runtime 如果要支持 PagedAttention，需要硬件/驱动层做哪些改动？**
   - 提示：block_table 翻译、非连续内存访问、decode 时的 scatter/gather

---

## 附录A：Houmo TCIM 的 KV Cache 实现 — 与 PagedAttention 对比

### A.1 Houmo 的方式：编译期固定 KV Cache

在 `build.py` 中从 ONNX 模型自动检测 KV Cache 输入：

```python
def get_n_blocks(model_path):
    model = onnx.load(model_path)
    total_inputs = model.graph.input
    n_kvcaches = 0
    for inp in total_inputs:
        if "kcache" in inp.name or "vcache" in inp.name:
            n_kvcaches += 1
    n_blocks = n_kvcaches // 2
    return n_blocks
```

ONNX 模型里有多少个 `*kcache_input` / `*vcache_input` → 就有多少层 Block。

### A.2 KV Cache 传递方式

从 `qwen_base.py`：

```python
# Prefill 计算 KV → Decode 接收 KV
for i in range(self.nblocks):
    cache = self.prefill.get_dev_input(f"model_layers_{i}_self_attn_kcache_input")
    self.decode.set_input(f"model_layers_{i}_self_attn_kcache_input", cache)
    # vcache 同理
```

KV Cache 是**每个 Block 一对固定 shape 的 tensor**，prefill 完成后直接传给 decode。

### A.3 与 PagedAttention 的对比

| | nano-vllm PagedAttention | Houmo TCIM |
|---|---|---|
| **KV Cache 形状** | 256-token block，动态分配 | 编译时固定 shape（如 `(1, 4096, 4, 128)`） |
| **多请求共享** | 引用计数共享 block | 每个请求独立 KV Cache tensor |
| **Prefix Caching** | 有（xxhash + 链式哈希） | 无 |
| **显存利用率** | 接近 100% | < 50%（短 prompt pad 到 max_seq_len） |
| **实现位置** | 纯 Python (BlockManager) | 编译器 + Runtime（`.hmm` 固定 tensor） |

**核心原因**：`.hmm` 是静态编译模型 → KV Cache shape 编译时就确定 → 不支持运行时动态分配。

---

## 附录B：静态图 vs 动态图

### B.1 核心差异

```
静态图 (Houmo .hmm):
  编译时: 确定了所有 tensor 的 shape、所有算子的执行顺序
  运行时: 只能按编译好的图执行，shape 不能变

动态图 (PyTorch/vLLM):
  编译时: 不确定 shape
  运行时: 每次 forward 可以有不同的 batch size、seq_len
```

| 场景 | 动态图 (nano-vllm) | 静态图 (Houmo .hmm) |
|---|---|---|
| **batch size 变化** | `model(tensor(bs=3))` → `model(tensor(bs=5))` 都能跑 | 编译时 `batch=1` → 只能跑 batch=1 |
| **seq_len 变化** | 短 prompt 和长 prompt 同一个模型 | 必须 pad 到固定长度 |
| **KV Cache** | 动态分配 block，用完回收 | 固定大小 tensor，编译就定好了 |
| **新请求加入** | Scheduler 随时 add_request | 只能等上一个完全结束 |

### B.2 不是所有编译模型都是静态 shape

| 框架 | 图结构 | Shape | 例子 |
|---|---|---|---|
| **静态图 + 静态 shape** | 编译期固定 | 编译期固定 | Houmo `.hmm`、CUDA Graph |
| **静态图 + 动态 shape** | 编译期固定 | 运行时可变（有范围） | TensorRT（min/opt/max）、ONNX dynamic_axes、`torch.compile(dynamic=True)` |
| **动态图** | 运行时构建 | 每次都可变 | PyTorch eager、vLLM 整体 |

### B.3 ONNX Dynamic Axes 示例

```python
# ONNX 导出时标注动态维度
torch.onnx.export(
    model, x, "model.onnx",
    dynamic_axes={
        "input": {0: "batch", 1: "seq_len"},    # 这两维都可变
        "output": {0: "batch", 1: "seq_len"}
    }
)

# 同一份编译产物，不同输入 shape
session.run(["output"], {"input": x_batch3})   # batch=3 ✅
session.run(["output"], {"input": x_batch5})   # batch=5 ✅
```

### B.4 nano-vllm 的两层混合

```
Scheduler (动态图): Python 灵活调度，动态管理 Sequence 和 block_table
  ↓
ModelRunner.run_model():
  ├── prefill → eager mode (动态 shape)       ← seq_len 每次可能不同
  └── decode  → CUDA Graph (静态 shape)       ← 每个 batch_size 一个静态图
```

---

## 附录C：Houmo 是否实现了 Continuous Batching？

**没有实现。** 证据：

1. **batch=1 硬编码**：`get_model.py:114` 中 `"batch": 1` 编译时写死
2. **串行推理**：一次 prompt → prefill → decode → 输出 → 再等下一个输入
3. **无调度器**：没有 waiting/running 队列、Scheduler、BlockManager 等结构

---

## 附录D：Houmo 实现 PagedAttention + Continuous Batching 的路线

### D.1 核心障碍

```
当前 Houmo:
  .hmm 编译时 shape 写死 → KV Cache 是固定 tensor → 无法动态分页
  batch=1 硬编码 → 无法并行处理多个请求
```

### D.2 四步改动计划

#### 第1步：编译器 — 支持 Dynamic Shape

```
当前: .hmm 中每个 tensor 的 shape 是 compile-time constant

需要:
  标注 dynamic dim: [batch, seq_len, ...]
  batch: 1~64，seq_len: 1~4096
  不是每个 shape 都编译一个 kernel → 按 shape 范围做代码生成
```

参考：ONNX `dynamic_axes`、TensorRT `min/opt/max profile`。

#### 第2步：Runtime — Tensor Shape 运行时可变

```
当前: tcim::Tensor shape 构造时固定

需要: Module::SetInputShape(actual_shape)
  或在 Module::Run 时传入实际 shape
```

#### 第3步：KV Cache — 从固定 Tensor 到 Block Pool

```
方式A (软件层，过渡方案):
  Runtime 分配一大块连续显存 → 软件层分页管理
  prefill/decode 时，根据 block_table 组装连续内存
  DMA 到 NPU → NPU 计算（NPU 看到的是连续内存）

方式B (硬件层，最终目标):
  NPU 原生支持 Scatter/Gather → 直接接受 block_table
  类似 GPU Flash Attention 的 block_table 参数
```

#### 第4步：调度器 — Continuous Batching

```
新增 Scheduler（纯 CPU/Python，复用 nano-vllm 逻辑）:
  waiting/running deque
  schedule() → prefill + decode 两阶段
  preempt() → 显存不够时踢人
```

### D.3 分阶段落地

| 阶段 | 改动范围 | 效果 |
|---|---|---|
| **Phase 1** | 编译器 + Runtime 支持 dynamic batch | batch>1 能跑，多个独立 prompt 合并 prefill |
| **Phase 2** | 软件层 Block Pool + 调度器 | 连续批处理，能同时服务 N 个请求 |
| **Phase 3** | 硬件层 Scatter/Gather | 无需软件 DMA 拼装，性能最优 |

### D.4 和各团队讨论的重点

| 团队 | 改动 | 关键技术点 |
|---|---|---|
| **编译器** | dynamic shape 支持 | shape range profiling、kernel 代码生成 |
| **Runtime** | SetInputShape + 动态显存分配 | Tensor shape 运行时可变 |
| **HAL/驱动** | Block Pool 显存管理 / 非连续内存 DMA | Scatter-Gather 或软件拼装 |
| **推理工具链** | Scheduler + BlockManager（纯软件）| nano-vllm 逻辑可直接复用 |
