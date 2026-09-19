# 第12课：ModelRunner — 从调度到执行

> 学习路线：阶段3 推理引擎核心（nano-vllm engine/model_runner.py）
> 对应文件：`nano-vllm/nanovllm/engine/model_runner.py`（257 行）

---

## 一、ModelRunner 是引擎的"最后一环"

```
Scheduler 决定 → ModelRunner 执行 → Sampler 采样
      ↑                  ↑               ↑
   "处理谁"          "怎么算"         "选哪个token"
```

**6 个核心方法概览**：

| 方法 | 作用 | 行号 |
|---|---|---|
| `allocate_kv_cache()` | 根据剩余显存动态计算 KV Cache 容量 | 103-121 |
| `prepare_prefill(seqs)` | 拼装变长序列输入 + slot_mapping | 129-170 |
| `prepare_decode(seqs)` | 拼装单 token 输入 + block_tables | 172-188 |
| `run_model()` | prefill → eager, decode → CUDA Graph replay | 196-212 |
| `capture_cudagraph()` | 为不同 batch size 预捕获计算图 | 223-257 |
| `run()` | 总入口: prepare → run_model → sample | 214-220 |

---

## 二、初始化流程

```python
def __init__(self, config, rank, event):
    # 1. TP 通信初始化
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size, rank)
    torch.cuda.set_device(rank)

    # 2. 加载模型
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)

    # 3. 关键初始化顺序
    self.warmup_model()          # 先跑一次 forward → 分配 cuDNN buffer
    self.allocate_kv_cache()     # 基于 peak memory 计算 KV Cache 容量
    self.capture_cudagraph()     # 预录制 decode 计算图

    # 4. TP 多进程通信（rank > 0 进入事件循环）
    if self.world_size > 1:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True)
        else:
            self.shm = SharedMemory(name="nanovllm")
            self.loop()    # rank > 0 永久等待 rank 0 的指令
```

**为什么 warmup 在 KV Cache 分配之前？** warmup 会触发 cuDNN/cuBLAS 的首次 kernel 编译和 buffer 分配，显存会有一个峰值。先 warmup 拿到真实的 peak memory，再算可用 KV Cache 容量，才不会超分配。

---

## 三、allocate_kv_cache — 动态计算 KV Cache 容量

```python
def allocate_kv_cache(self):
    # Step 1: 查询 GPU 显存状态
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]     # 历史峰值
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前分配

    num_kv_heads = hf_config.num_key_value_heads // self.world_size   # TP 切分
    head_dim = ...  # 128

    # Step 2: 计算单个 KV Cache block 的字节数
    block_bytes = 2(K,V) × num_layers × block_size × num_kv_heads × head_dim × dtype_size

    # Step 3: 动态算出能分配多少 block
    available = total × gpu_memory_utilization − used − peak + current
    num_blocks = available // block_bytes

    # Step 4: 一次性分配整块 KV Cache 显存
    self.kv_cache = torch.empty(
        2,           # K 和 V 两个缓存
        num_layers,  # 每层独立
        num_blocks,  # block 池
        block_size,  # 每 block 的 token 数
        num_kv_heads,
        head_dim
    )

    # Step 5: 把 KV Cache 子视图绑定到每层的 Attention 模块
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]  # ← 引用，非拷贝！
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### 显存预算公式详解

```
物理显存:  16GB
gpu_memory_utilization: 0.9  →  可用 14.4GB

peak: warmup 时的峰值 (模型权重 + optimizer state + 临时 buffer)
current: warmup 后释放了临时 buffer，当前实际占用

available = 14.4GB - 已用 - (peak - current)
          = 14.4GB - 已用 - 临时buffer已释放部分的差值
          ≈ 14.4GB - 模型权重占用

num_blocks = available / block_bytes
```

**为什么用 `peak - current`？** warmup 后临时 buffer 已经释放了，但 `used` 还没更新（PyTorch 的 caching allocator 可能保留着）。`peak - current` 把已释放的 buffer 加回来。

### 为什么逐层绑定引用而非拷贝？

```python
module.k_cache = self.kv_cache[0, layer_id]  # 引用
# 不是: module.k_cache = self.kv_cache[0, layer_id].clone()  ← 拷贝

# 好处：
# 1. 零额外显存（28 层共享同一块 KV Cache pool）
# 2. store_kvcache() 写入后，BlockManager 直接通过 block_id 索引即可访问
```

---

## 四、prepare_prefill — 拼装变长序列输入

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []        # 所有 seq 的 token 串联
    positions = []        # 每个 token 的绝对位置
    cu_seqlens_q = [0]   # query 的累积长度（变长序列格式）
    cu_seqlens_k = [0]   # key 的累积长度
    slot_mapping = []     # 每个 token 的 KV 写入槽位

    for seq in seqs:
        start = seq.num_cached_tokens              # 已缓存的 token 数
        seqlen_q = seq.num_scheduled_tokens        # 本次要处理的 token 数
        end = start + seqlen_q

        # 收集 token 和位置
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))

        # 变长序列格式
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + end)

        # 计算 KV 写入槽位
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * self.block_size + offset
            slot_mapping.extend(range(slot_start, slot_end))

    # prefix cache: query shorter than key → 需要 block_tables
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    # 全部 pin_memory → non_blocking 异步传输
    input_ids = tensor(input_ids, pin_memory=True).cuda(non_blocking=True)
    ...

    # 设置全局 context（供 attention kernel 读取）
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)
    return input_ids, positions
```

### cu_seqlens — 变长序列格式

Flash Attention 的 `varlen_func` 要求变长序列的累积长度格式：

```python
# 三个 seq: prompt 长度分别为 300, 500, 200
cu_seqlens = [0, 300, 800, 1000]
# 含义:
#   seq_0: input_ids[0:300]
#   seq_1: input_ids[300:800]
#   seq_2: input_ids[800:1000]

# 所有 token 串联在一起，节省了 padding 开销
```

### slot_mapping — KV 写入位置

每个 token 的 K/V 应写入 KV Cache 的哪个 slot：

```python
seq: block_table = [0, 3, 7], block_size = 256

# chunked prefill: num_cached_tokens=200, scheduled=312 (共 512 tokens)
# start=200, end=512

start_block = 200 // 256 = 0    # B0（部分已有缓存）
end_block   = (512+255)//256 = 2  # B3, B7

B0: slot_start=0×256 + 200%256=200, slot_end=0×256+256=256
    → slot_mapping = [200, 201, ..., 255]   ← B0 的后 56 个位置

B3: slot_start=3×256, slot_end=3×256+256=768
    → slot_mapping = [768, 769, ..., 1023]  ← B3 的全部 256 个位置

B7: slot_start=7×256, slot_end=7×256+(512-2×256)=7×256
    → slot_mapping = [1792, 1793, ...]      ← B7 的前面 token 位置
```

### prefix cache 的 block_tables

```python
# 当 Key 比 Query 长时 (prefix cache 命中):
# cu_seqlens_k[-1] > cu_seqlens_q[-1]
# → 有些 K/V 不在当前 batch 的 input_ids 中 → 需要 block_tables 去 KV Cache 中查找

if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

---

## 五、prepare_decode — 单 token 输入

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = [seq.last_token for seq in seqs]              # (batch_size,)
    positions = [len(seq) - 1 for seq in seqs]                # 绝对位置
    slot_mapping = [
        seq.block_table[-1] * self.block_size +               # 最后一个 block 的基地址
        seq.last_block_num_tokens - 1                         # + block 内偏移
        for seq in seqs
    ]
    context_lens = [len(seq) for seq in seqs]                 # 每个 seq 的总长度
    block_tables = self.prepare_block_tables(seqs)

    set_context(False, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

**为什么 decode 这么简单？** 每次 1 token × batch_size，不需要变长序列的复杂拼接。

---

## 六、run_model — Eager vs CUDA Graph

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # Prefill / 强制 eager / batch > 512:
        # 正常 eager forward → 动态图
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Decode: CUDA Graph replay → 静态图
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        # 找到 ≥ bs 的最小预录制 batch size

        # 填充静态图的输入 buffer
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs] = context.block_tables

        graph.replay()   # ← 一次提交所有 kernel，零 launch 开销
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 为什么 prefill 不能用 CUDA Graph？

Prefill 的输入 `(batch, seq_len, d_model)` 每次 seq_len 不同 → 无法预录制。Decode 的 `(batch, 1, d_model)` batch 只有有限个值 → 可以预录制。

### `next(x for x in graph_bs if x >= bs)` 详解

```python
graph_bs = [1, 2, 4, 8, 16, 32, 48, ..., 512]

bs=3  → next(x >= 3)  → 4    (用 batch=4 的图)
bs=16 → next(x >= 16) → 16   (精确匹配)
bs=9  → next(x >= 9)  → 16   (向上取整到最近的预录 batch)

# 仅填充前 bs 个 slot → 其余 slot 保持不变 (被 mask 掉 → 不参与计算)
```

---

## 七、capture_cudagraph — 预录制

```python
@torch.inference_mode()
def capture_cudagraph(self):
    max_bs = min(max_num_seqs, 512)
    max_num_blocks = ceil(max_model_len / block_size)

    # 预分配静态 buffer（所有 batch size 共享）
    input_ids = torch.zeros(max_bs, dtype=torch.int64)      # ↙ 填充时会覆写
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hidden_size)

    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}

    # 从大到小录制 → 共享 graph pool → 省显存
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # warmup: 先跑一次，分配 kernel 需要的内部 buffer
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # capture: 录制 GPU 命令序列
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        if self.graph_pool is None:
            self.graph_pool = graph.pool()    # 第一个(最大)的 pool 被后续图共享
        self.graphs[bs] = graph
```

### 录制 vs 执行的对比

```
正常执行 (eager):
  CPU: launch linear_1 → wait → launch norm → wait → launch attention → ...
  GPU:  [linear_1][norm][attention]...
  每次 kernel launch: 5-15μs CPU↔GPU 开销

Graph Replay:
  CPU: graph.replay()
  GPU: [linear_1][norm][attention]...  (一口气执行)
  一次 launch → 零额外开销
```

### 为什么 reversed 录制？

```
先录最大的 (bs=512) → graph_pool = mem_pool_512
再录 bs=480      → 复用 mem_pool_512 (子集)
再录 bs=464      → 复用 mem_pool_512 (子集)
...
最后录 bs=1      → 复用 mem_pool_512

所有图共享一个内存池 → 显存高效
```

---

## 八、Tensor Parallel 多进程通信

nano-vllm 的 TP 通过 Python multiprocessing + SharedMemory 实现：

```python
# LLMEngine.__init__:
for i in range(1, tensor_parallel_size):   # 启动额外进程
    event = mp.Event()
    process = mp.Process(target=ModelRunner, args=(config, i, event))
    process.start()

# rank > 0 的进程进入事件循环:
def loop(self):
    while True:
        method_name, args = self.read_shm()   # 等待 rank 0 发指令
        self.call(method_name, *args)
        if method_name == "exit": break

# rank 0 发送指令:
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)    # pickle 序列化 → 共享内存
        for event in self.events:
            event.set()                        # 唤醒所有 worker 进程
    return getattr(self, method_name)(*args)   # 所有 rank 执行同一个方法
```

**流程**：

```
Step 1: LLMEngine.step() → self.model_runner.call("run", seqs, is_prefill)
Step 2: rank 0 把 ("run", seqs, is_prefill) pickle → SharedMemory
Step 3: rank 0 对所有 event.set() → 唤醒 rank 1,2,3...
Step 4: 所有 rank 执行 self.run(seqs, is_prefill)
          → 模型内部 RowParallelLinear 自动 dist.all_reduce()
Step 5: rank 0 返回结果
```

---

## 九、run() — 总入口

```python
def run(self, seqs, is_prefill):
    # 1. 准备输入
    input_ids, positions = (
        self.prepare_prefill(seqs) if is_prefill
        else self.prepare_decode(seqs)
    )

    # 2. 准备采样参数（仅 rank 0）
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

    # 3. 模型推理
    logits = self.run_model(input_ids, positions, is_prefill)

    # 4. 采样（仅 rank 0）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    # 5. 清理全局 context
    reset_context()
    return token_ids
```

---

## 十、与 Houmo 的对应

| nano-vllm ModelRunner | Houmo HLIEvLLM 等效 |
|---|---|
| `allocate_kv_cache` → 固定 tensor | vLLM KVCacheManager + helion 显存管理 |
| `prepare_prefill` → cu_seqlens + slot_mapping | vLLM GPUModelRunner._prepare_inputs |
| `prepare_decode` → block_tables | vLLM BlockTable.commit_block_table |
| `run_model(eager)` → CUDA forward | vllm_houmo forward → helion execute |
| `run_model(CUDA Graph)` → graph.replay() | 无 NPU 等效（静态图本身就是"预编译"的） |
| SharedMemory TP | NCCL all_reduce（vLLM 标准） |

---

## 十一、思考题

1. **为什么 warmup 必须在 allocate_kv_cache 之前？**
   - 提示：warmup 分配 cuDNN buffer → peak memory ↑ → 需要先测量 peak

2. **slot_mapping 和 block_table 的区别是什么？**
   - 提示：slot_mapping = 每个 token 写入位置（1D），block_table = 逻辑块→物理块（2D）

3. **为什么 CUDA Graph 的 bs 列表是 [1,2,4,8,16,32,...] 而不是连续的？**
   - 提示：每个图都要占显存，太多图会浪费

4. **你们的 NPU 能否类比 CUDA Graph 做"预录制"？还是 .hmm 本身就是"预编译"的静态图？**
