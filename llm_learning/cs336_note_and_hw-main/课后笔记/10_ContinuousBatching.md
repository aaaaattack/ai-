# 第10课：引擎层 — Continuous Batching 调度

> 学习路线：阶段3 推理引擎核心（nano-vllm engine/）
> 对应文件：
> - `nano-vllm/nanovllm/engine/sequence.py`（Sequence 状态机）
> - `nano-vllm/nanovllm/engine/scheduler.py`（调度器）
> - `nano-vllm/nanovllm/engine/llm_engine.py`（引擎入口）
> - `nano-vllm/nanovllm/engine/block_manager.py`（PagedAttention — 下节课详讲）

---

## 一、问题：N 个请求不能排队等

### 1.1 传统串行推理

```
请求1: [token_A → token_B → token_C → token_D → 结束]
请求2:                                               [token_X → token_Y → 结束]
请求3:                                                                       [token_P → ...]
        ↑ 一个完全结束后才开始下一个
        ↑ decode 阶段每个 token 计算量极小（访存密集），GPU 空转严重
```

### 1.2 Continuous Batching 的直觉

```
请求1: [A][B][C]──[C1][C2][C3]──结束
请求2:       [X][Y]──[Y1]──[Y2]──结束
请求3:             [P][Q]──[Q1]──[Q2]──[Q3]──结束
         ↑  prefill       ↑  decode
         prefill 错开做      decode 合并到一个 batch
```

**核心收益**：

- 多个请求的 decode 合并计算 → GPU 利用率提升
- 新请求可以随时加入 → 不用等待前面的请求结束
- 短请求不会被长请求阻塞

---

## 二、Sequence — 一个请求的状态机

源码：`nano-vllm/nanovllm/engine/sequence.py`

### 2.1 三种状态

```python
class SequenceStatus(Enum):
    WAITING  = auto()   # 刚提交，prompt token 还没被 prefill 处理
    RUNNING  = auto()   # prompt 已被 prefill 处理完毕，正在 decode
    FINISHED = auto()   # 遇到 EOS 或达到 max_tokens
```

状态迁移：

```
用户提交 prompt
  │
  ▼
WAITING  ──→ prefill 完成 ──→  RUNNING  ──→ EOS/max_tokens ──→  FINISHED
   ↑                               │
   └───────── preempt ─────────────┘  (显存不够，被踢回 waiting)
```

### 2.2 关键属性

```python
class Sequence:
    block_size = 256                        # 每个 KV Cache block 的大小

    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(counter)         # 全局唯一 ID
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)    # 所有 token (prompt + 生成的)
        self.last_token = token_ids[-1]     # 最后一个 token（decode 时用）
        self.num_tokens = len(token_ids)    # 当前总 token 数
        self.num_prompt_tokens = len(token_ids)  # prompt 部分 token 数
        self.num_cached_tokens = 0          # 已被 prefill 处理过的 token 数
        self.num_scheduled_tokens = 0       # 本次 step 排了多少个 token
        self.is_prefill = True              # 当前处于 prefill 阶段
        self.block_table = []               # KV Cache 物理 block 列表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

**属性关系**：

```
num_tokens = len(token_ids) = num_prompt_tokens + num_completion_tokens

num_cached_tokens:  模型已经看过多少 token（<= num_tokens）
                           每次 step 后 + num_scheduled_tokens

num_blocks = ceil(num_tokens / block_size)
            KV Cache 需要的 block 数 = 总 token 数 ÷ 每 block 容量
```

### 2.3 辅助属性和方法

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens

@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size

def block(self, i):
    """返回第 i 个 block 应该包含的 token_ids"""
    return self.token_ids[i*self.block_size : (i+1)*self.block_size]

def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

---

## 三、Scheduler — 核心调度逻辑

源码：`nano-vllm/nanovllm/engine/scheduler.py`

### 3.1 数据结构

```python
class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs                # 最大并行请求数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 每 step 最大 token 处理量
        self.eos = config.eos
        self.block_size = config.kvcache_block_size            # KV Cache block 大小

        self.block_manager = BlockManager(...)  # KV Cache 管理器（下节课详讲）
        self.waiting: deque[Sequence] = deque()  # 待 prefill 的请求队列
        self.running: deque[Sequence] = deque()  # 正在 decode 的请求队列
```

**两个队列**：

```
waiting (deque): [seq_A, seq_B, seq_C]  ← 新请求，prompt 还没处理完
running (deque): [seq_X, seq_Y, seq_Z]  ← 正在 decode，每个 step 生成 1 token
```

### 3.2 schedule() — 一次调度决策

完整代码逻辑：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_batched_tokens = 0

    # ═══════════════════════════════════
    # Phase 1: Prefill
    # ═══════════════════════════════════
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查 1: token 预算
        remaining = self.max_num_batched_tokens - num_batched_tokens
        if remaining == 0:
            break

        # 检查 2: KV Cache 显存
        if not seq.block_table:
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break          # 显存不够，等下个 step

            # 需要处理的 token 数 = 总 token - 已缓存的 token
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
        else:
            num_tokens = seq.num_tokens - seq.num_cached_tokens

        # Chunked Prefill: 如果剩余预算不够处理所有 token
        # 且已有人被调度 → 等下个 step（只能第一个 seq 做 chunked）
        if remaining < num_tokens and scheduled_seqs:
            break

        # 分配 KV Cache
        if not seq.block_table:
            self.block_manager.allocate(seq, num_cached_blocks)

        # 实际处理多少 token
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        num_batched_tokens += seq.num_scheduled_tokens

        # prefill 完成 → 转到 running
        if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

        scheduled_seqs.append(seq)

    # 如果有 prefill 任务 → 直接返回
    if scheduled_seqs:
        return scheduled_seqs, True     # is_prefill = True

    # ═══════════════════════════════════
    # Phase 2: Decode
    # ═══════════════════════════════════
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查：当前 block 写满了没？需要新 block 吗？
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())  # 显存不够 → 踢一个出去
            else:
                self.preempt(seq)                 # 只剩自己 → 踢自己
                break
        else:
            seq.num_scheduled_tokens = 1               # decode: 每次 1 token
            seq.is_prefill = False
            self.block_manager.may_append(seq)          # 需要时分配新 block
            scheduled_seqs.append(seq)

    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))  # 保持 deque 顺序
    return scheduled_seqs, False                        # is_prefill = False
```

### 3.3 Token 预算示例

```
max_num_batched_tokens = 16384

seq1: prompt=4000 tokens → 预算: 4000
seq2: prompt=8000 tokens → 预算: 4000+8000 = 12000
seq3: prompt=5000 tokens → 预算: 12000+5000 = 17000 > 16384
                             → seq3 拒绝，留在 waiting

结果: 这个 step prefill seq1 和 seq2，下个 step 再 prefill seq3
```

### 3.4 Chunked Prefill

大 prompt 一次 prefill 不完 → 分块处理：

```
seq_X: prompt=20000 tokens, block_size=256

Step 1: num_batched_tokens=0, budget=16384
  → num_scheduled_tokens = min(20000, 16384) = 16384
  → num_cached_tokens = 16384 < num_tokens=20000
  → 留在 waiting，状态不变

Step 2: num_cached_tokens=16384, 剩余 3616
  → num_scheduled_tokens = min(3616, budget) = 3616
  → num_cached_tokens = 20000 == num_tokens
  → 转到 RUNNING
```

**为什么只有第一个 seq 做 chunked？**

```
if remaining < num_tokens and scheduled_seqs:
    break
# 逻辑: 如果已经有其他人被调度了，就不做 chunked
# 原因: chunked prefill 会把这个 seq 留在 waiting
#       如果同时有多个 chunked seq，调度变复杂
```

### 3.5 抢占机制 Preempt

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)   # 释放 KV Cache
    self.waiting.appendleft(seq)          # 优先级最高：插队到队首
```

**触发条件**：decode 时显存不够分配 KV Cache block → 从 running 队尾踢人。

**为什么从队尾踢？** 队尾的是最早的请求（最接近结束），释放的 KV Cache 最多。而且已经生成了很多 token → 重新 prefill 代价大 → 不优先踢队尾 → 但从队尾踢能释放更多显存。

**实际逻辑**：代码中 `self.running.pop()` 从右侧 pop → 队尾被踢。

---

## 四、LLMEngine — 引擎入口

源码：`nano-vllm/nanovllm/engine/llm_engine.py`

### 4.1 架构

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.scheduler = Scheduler(config)        # 调度器
        self.model_runner = ModelRunner(config)   # 模型执行器（下下节课）
        # 多卡时启动额外进程（TP）
```

### 4.2 add_request — 提交请求

```python
def add_request(self, prompt, sampling_params):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)   # tokenize
    seq = Sequence(prompt, sampling_params)       # 创建 Sequence
    self.scheduler.add(seq)                       # 加入 waiting 队列
```

### 4.3 step — 一次推理步

```python
def step(self):
    # 1. 调度：决定这个 step 处理哪些 seq
    seqs, is_prefill = self.scheduler.schedule()

    # 2. 执行：模型 forward
    token_ids = self.model_runner.call("run", seqs, is_prefill)

    # 3. 后处理：更新 Sequence 状态
    self.scheduler.postprocess(seqs, token_ids, is_prefill)

    # 4. 返回已完成的 seq 的 token
    outputs = [(seq.seq_id, seq.completion_token_ids)
               for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

### 4.4 generate — 完整的生成循环

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    # 提交所有请求
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    outputs = {}
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()

        # 统计 throughput
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)

        # 收集完成的输出
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            pbar.update(1)

    # 按 seq_id 排序返回
    outputs = [outputs[i] for i in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(ids), "token_ids": ids}
               for ids in outputs]
    return outputs
```

**`num_tokens > 0` 判断 prefill vs decode**：

```python
# schedule() 返回时:
# prefill: num_tokens = sum(num_scheduled_tokens)  → 正值
# decode:  num_tokens = -len(seqs)                 → 负值

# 用正负号区分阶段：prefill 统计 token throughput，decode 统计 seq throughput
```

---

## 五、白话总结

```
Scheduler = 一个餐厅的排号系统

waiting = 门口排队（prompt 还没处理完）
running = 正在用餐的桌子（正在 decode）

每个 step:
  1. 从 waiting 里叫号 prefill
     → 要算 token 预算 = 厨房每轮最多备多少菜
     → 大菜（长 prompt）可能分两次做（chunked prefill）
     → prefill 完的客人移到 running

  2. waiting 空了 → 给每桌上一个菜（decode 1 token）
     → 每桌只要一个菜 → 可以同时上很多桌（batch 合并）
     → 桌子不够 → 把离吃完最远的客人请回门口（preempt）
     → 吃完了 → 收桌子（deallocate）
```

---

## 六、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **推理工具链** | Continuous batching 是 vLLM 相比传统推理框架的核心创新。你们的定制版是否实现了调度器？ |
| **TCIM Runtime** | 你们目前 `.hmm` 的 prefill/decode 分开编译 + 固定 shape → 不支持 dynamic batch → 无法做 continuous batching |
| **HAL** | PagedAttention 需要 KV Cache 分页 → BlockManager 管理 → HAL 需要支持非连续的内存访问（block_table 映射） |

---

## 七、思考题

1. **为什么 prefill 优先于 decode？如果反过来会怎样？**
   - 提示：prefill 是新请求的"入场券"，不做 prefill 就没法 decode

2. **Chunked prefill 的 `first seq only` 限制为什么存在？如果允许多个 seq 同时 chunked，会有什么问题？**

3. **preempt 为什么从 running 队尾踢人，而不是队首？**
   - 提示：队首是最新加入的 seq，prompt tokens 少 → KV Cache 少 → 抢占比小

4. **你们的 TCIM Runtime 如果要支持 continuous batching，需要哪些改动？**
   - 提示：动态 shape、KV Cache 分页、prefill/decode 动态切换
