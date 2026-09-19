# 生产功能 07：Pipeline Parallel (PP)

> 学习路线：P0 必学 — Houmo 已在用 (prefill_pp_decode_pp/tp.py)
> 对应源码：Houmo `qwen3_pipeline/build.py` + `prefill_pp_decode_pp.py` + `prefill_pp_decode_tp.py`；HLIEvLLM `vllm/distributed/parallel_state.py` + `vllm/v1/worker/gpu/pp_utils.py`

---

## 一、问题：单芯片放不下整个模型

### 1.1 回忆 Tensor Parallel（TP）的局限

之前学习的 TP（第12课）是在**同一层内部**把权重矩阵按列/行切分到多张卡：

```
TP:  同一层的 W 切到卡0和卡1
     卡0: 算 W[:half]@X，卡1: 算 W[half:]@X → all-reduce 合并

问题：TP 的 all-reduce 通信量大（每层都要同步），多卡间通信带宽成为瓶颈。
      4卡 TP 通信开销 ≈ 3卡 TP 的 2倍 → 扩展性受限。
```

### 1.2 Pipeline Parallel 的思路

**按层切分模型**，不同卡负责不同层的计算：

```
DP (Data Parallel):        模型复制
TP (Tensor Parallel):      层内切权重
PP (Pipeline Parallel):    层间切模型 ← 这次学的

PP:  卡0: Layer 0~15 (Embedding + 前半 Transformer Blocks)
     卡1: Layer 16~31 (后半 Transformer Blocks + LM Head)

     数据流: 卡0 算完 → hidden states 传给卡1 → 卡1 继续算
```

**PP 的核心优势：**
- 通信量极小（只有层边界处的 hidden states 传输，比 TP 的每层 all-reduce 少得多）
- 每块卡只需要存储自己的那部分权重 → 可以把大模型塞进多块小芯片
- 天然支持 Prefill/Decode 分离到不同芯片

---

## 二、PP 的核心概念

### 2.1 Bubble（气泡）— PP 的关键代价

```
时间 →

Stage 0:  [F0]    [F1]    [F2]    [F3]    [空]
Stage 1:  [空]    [F0]    [F1]    [F2]    [F3]
Stage 2:  [空]    [空]    [F0]    [F1]    [F2]
Stage 3:  [空]    [空]    [空]    [F0]    [F1]
                                  ↑
                              Bubble（空闲时间）
```

**Bubble 利用率 = 有用计算时间 / (有用计算时间 + 空闲时间)**

```
如果每次只处理 1 个请求:
  Bubble 利用率 = stages / (2 × stages - 1) → 4 stages ≈ 57%

解决方案：微批次 (micro-batch)
  把 1 个 batch 拆成 M 个 micro-batch，流水线上同时有 M 个 micro-batch 在跑
  当 M 很大时 → Bubble 利用率 → 1
```

### 2.2 关键通信原语

```
Stage i-1 → Stage i:   send_tensor_dict( hidden_states + residuals )
Stage i   → Stage i+1: recv_tensor_dict( src=i-1 )

每次传输的数据量 = batch_size × hidden_size × dtype_bytes
  例如: bs=1, hidden=4096, fp16 → 8KB  ← 非常小，几乎无通信瓶颈
```

### 2.3 PP vs TP：通信量对比

| | PP (Pipeline) | TP (Tensor) |
|---|---|---|
| **切分维度** | 按层 (depth) | 按权重矩阵 (width) |
| **通信时机** | 只在层边界 (1~3 次) | 每层都要同步 (N 次) |
| **单次通信量** | 小 (hidden states) | 大 (激活值 all-reduce) |
| **总通信量** | O(batch × hidden) × num_stages | O(batch × hidden × num_layers × tp_size) |
| **Bubble** | 有（微批次缓解） | 无 |
| **适用场景** | 模型太大单卡放不下 | 单层计算太慢需加速 |

---

## 三、vLLM 的 PP 实现

### 3.1 PP Group 初始化

[`vllm/distributed/parallel_state.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/parallel_state.py)

vLLM 将所有并行维度组织为：`ExternalDP × DP × PP × PCP × TP`

```python
# parallel_state.py — PP Group 初始化
group_ranks = (
    all_ranks.transpose(2, 4)
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
_PP = init_model_parallel_group(group_ranks, ...)

# 全局访问
def get_pp_group():
    return _PP
```

**GroupCoordinator 的关键属性：**

```python
@property
def first_rank(self):
    return self.ranks[0]          # PP 组的第一张卡

@property
def last_rank(self):
    return self.ranks[-1]         # PP 组的最后一张卡

@property
def is_first_rank(self):
    return self.rank == self.first_rank   # 我是第一张卡吗？

@property
def is_last_rank(self):
    return self.rank == self.last_rank    # 我是最后一张卡吗？

@property
def next_rank(self):
    return self.ranks[(self.rank_in_group + 1) % self.world_size]

@property
def prev_rank(self):
    return self.ranks[(self.rank_in_group - 1) % self.world_size]
```

### 3.2 前向传播中的 PP 数据流

**第一 stage（`is_first_rank=True`）：**
- 正常接收 token IDs/embeddings
- 前向传播到自己负责的最后一层
- 输出 `IntermediateTensors` → 传给下一 stage

**中间 stage（既不是 first 也不是 last）：**
- 接收上一 stage 的 `IntermediateTensors`
- 继续前向传播
- 输出传给下一 stage

**最后 stage（`is_last_rank=True`）：**
- 接收上一 stage 的数据
- 完成最后一层的计算
- 输出 logits → 采样 token
- 将采样结果 **广播** 给所有其他 stages

### 3.3 GPU Worker 中的 PP 编排

vLLM V1 worker 用异步通信实现非阻塞 pipeline：

```python
# gpu_worker.py — execute_model
def execute_model(self, scheduler_output):
    # 确保上一次的非阻塞 send 已完成
    if self._pp_send_work:
        for handle in self._pp_send_work:
            handle.wait()

    # 非第一 rank：异步接收 intermediate tensors
    if forward_pass and not get_pp_group().is_first_rank:
        tensor_dict, comm_handles, _ = get_pp_group().irecv_tensor_dict(...)
        intermediate_tensors = AsyncIntermediateTensors(...)

    # 执行模型前向
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

    # 非最后 rank：异步发送 intermediate tensors
    self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
    return None   # 非最后 stage 不需要返回结果给 scheduler
```

### 3.4 采样 Token 广播

[`vllm/v1/worker/gpu/pp_utils.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/worker/gpu/pp_utils.py)

最后 stage 采样完 token 后，需要广播给前面所有 stages（用于更新 KV Cache）：

```python
# pp_utils.py — 从 last rank 广播采样结果
def pp_broadcast(sampled_token_ids, num_sampled, num_rejected):
    pp = get_pp_group()
    assert pp.is_last_rank
    torch.distributed.broadcast(sampled_token_ids.contiguous(), 
                                src=pp.last_rank, group=pp.device_group)
    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)

# 非 last rank 接收
def pp_receive(num_reqs, max_sample_len=1):
    pp = get_pp_group()
    assert not pp.is_last_rank
    sampled_tokens = torch.empty(num_reqs, max_sample_len, ...)
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)
    ...
    return sampled_tokens, num_sampled, num_rejected
```

### 3.5 Scheduler 的 PP 适配

```python
# scheduler.py — PP 时需要更多并发 batch 来填满 pipeline
pp_size = self.parallel_config.pipeline_parallel_size
concurrent_batches = 2 if pp_size <= 1 else pp_size
# PP 需要 pp_size 个并发 micro-batch 来填充 pipeline 各 stage
```

---

## 四、Houmo 的 Pipeline Parallel 实现

### 4.1 整体架构

Houmo 的 PP 是为**多芯片 NPU 系统**设计的，核心思路和 vLLM 一致（按层切分），但实现方式完全不同——用 `multiprocessing.Queue` 做进程间通信，天然匹配 NPU 静态图的特点。

```
┌──────────────────────────────────────────────────────────────┐
│                      多进程流水线架构                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Chip 0 (Producer)          Chip 1 (Consumer)                │
│  ┌──────────────────┐      ┌──────────────────┐              │
│  │ Prefill Part 0   │ ───→ │ Prefill Part 1   │ ───→ ...     │
│  │ (Layer 0~8)      │Queue │ (Layer 9~17)     │              │
│  └──────────────────┘      └──────────────────┘              │
│  ┌──────────────────┐      ┌──────────────────┐              │
│  │ Decode Part 0    │ ←── │ Decode Part 1    │ ←── ...     │
│  │ (Layer 0~8)      │Queue │ (Layer 9~17)     │              │
│  └──────────────────┘      └──────────────────┘              │
│                                                              │
│  Chip N-1 (result_shower / decode_runner)                    │
│  ┌──────────────────────────────┐                            │
│  │ Prefill Part N-1 (最后几层)   │                            │
│  │ + 采样 获取第一个 token       │                            │
│  │ + Decode Part N-1 (最后几层)  │                            │
│  │ + 每步采样 + token 解码       │                            │
│  └──────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 编译构建：按层切分 ONNX 模型

[`apis/converts/qwen3_pipeline/build.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/houmo-examples-xh2/apis/converts/qwen3_pipeline/build.py)

**Step 1: 识别 Transformer Block 数量**（第170-186行）

```python
def get_n_blocks(model_path):
    model = onnx.load(model_path, load_external_data=False)
    total_inputs = model.graph.input
    n_kvcaches = 0
    for inp in total_inputs:
        if "kcache" in inp.name or "vcache" in inp.name:
            n_kvcaches += 1
    n_blocks = n_kvcaches // 2    # 每层有 K cache + V cache
    return n_blocks, cache_start_idx
```

**Step 2: 均匀切分**（第189-259行）

```python
def clip(raw_path, split_paths, ndevice, model_name):
    n_blocks, cache_start_idx = get_n_blocks(raw_path)
    n_kvcache_per_stage = n_blocks // ndevice  # 每个 stage 分到多少层
    
    # 计算切分点：在 add_{2*k-1} 节点处切分
    mid_layer_names = []
    for i in range(ndevice):
        if i == ndevice - 1:
            mid_layer_names.append(model.graph.output[0].name)  # 最后一段：输出
        else:
            # 切分点 = add_{2*(层数*stage_idx) - 1}
            mid_layer_names.append(f"add_{2 * n_kvcache_per_stage * (i + 1) - 1}")
    
    # 每个 stage 的输入/输出
    for i in range(ndevice):
        inputs = []
        if i == 0:
            # 第一段：原始输入 + 属于它的 KV Cache
            inputs = [model.graph.input[j].name for j in range(cache_start_idx)]
            # 加上 Stage 0 的 KV Cache 输入
            for j in range(n_kvcache_per_stage):
                inputs.append(f"model_layers_{j}_self_attn_kcache_input")
                inputs.append(f"model_layers_{j}_self_attn_vcache_input")
        else:
            # 后续段：前一段的输出 + 属于它的 KV Cache
            inputs.append(stages_outputs[i-1][0])  # ← 关键：上一段输出作为本段输入
            for j in range(n_kvcache_per_stage):
                inputs.append(
                    f"model_layers_{j + i * n_kvcache_per_stage}_self_attn_kcache_input"
                )
                inputs.append(
                    f"model_layers_{j + i * n_kvcache_per_stage}_self_attn_vcache_input"
                )
        
        # 用 onnx.utils.extract_model 提取子图
        onnx.utils.extract_model(
            raw_path, split_paths[i],
            input_names=inputs, output_names=outputs
        )
```

**可视化：Qwen3-8B (32层) 切到 4 芯片：**

```
原始 ONNX: [Embedding] → [Block 0] → [Block 1] → ... → [Block 31] → [LM Head]
                        ↑ K0,V0     ↑ K1,V1              ↑ K31,V31

ndevice=4 → n_kvcache_per_stage = 32/4 = 8

Part 0: [Embedding] → [Block 0..7]   → output: add_15
         ↑ K0..K7,V0..V7
         输入: tokens + K0..K7,V0..V7

Part 1: 输入: add_15(Part0输出) + K8..K15,V8..V15
         → [Block 8..15] → output: add_31

Part 2: 输入: add_31 + K16..K23,V16..V23
         → [Block 16..23] → output: add_47

Part 3: 输入: add_47 + K24..K31,V24..V31
         → [Block 24..31] → [LM Head] → logits
```

**Step 3: 编译每个 part**（第333-381行）

```python
# 编译 Prefill 子模型 (ndevice=1 — 每个 part 单独编译到一个芯片)
for i in range(ndevice):
    Xh2Exec.build_from_hmonnx(
        is_prefill=True,
        hmonnx=f"hmquant_{model_name}_part{i}_with_act.onnx",
        hmm_name=f"{model_name}-{model_size}_prefill_part{i}",
        ndevice=1,          # ← 每个 part 跑在一个芯片上
        ...
    )

# 编译 Decode 子模型
for i in range(ndevice):
    Xh2Exec.build_from_hmonnx(
        hmonnx=f"hmquant_{model_name}_part{i}_with_act.onnx",  # 同一个切分 ONNX
        hmm_name=f"{model_name}-{model_size}_decode_part{i}",
        ndevice=1,          # ← decode 也是单芯片
        ...
    )

# 额外编译一个完整的 decode.hmms（ndevice=全芯片数）给 TP 模式用
Xh2Exec.build_from_hmonnx(
    hmonnx=decode_raw_path,   # 原始未切分的 ONNX
    hmm_name=f"{model_name}-{model_size}_decode",
    ndevice=ndevice,          # ← 多芯片 TP
    ...
)
```

**最终产物：**

```
output/xh2/
├── qwen3-pipeline-8b_prefill_part0.hmm   ← 芯片0的prefill子模型
├── qwen3-pipeline-8b_prefill_part1.hmm   ← 芯片1的prefill子模型
├── qwen3-pipeline-8b_prefill_part2.hmm
├── qwen3-pipeline-8b_prefill_part3.hmm
├── qwen3-pipeline-8b_decode_part0.hmm    ← 芯片0的decode子模型
├── qwen3-pipeline-8b_decode_part1.hmm    ← 芯片1的decode子模型
├── qwen3-pipeline-8b_decode_part2.hmm
├── qwen3-pipeline-8b_decode_part3.hmm
└── qwen3-pipeline-8b_decode.hmms         ← 完整decode(多芯片TP模式用)
```

---

## 五、两种运行时模式

### 5.1 模式 A：纯 PP（prefill_pp_decode_pp.py）

**Prefill 和 Decode 都用流水线并行。**

[`apis/inferences/qwen3_pipeline/prefill_pp_decode_pp.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/houmo-examples-xh2/apis/inferences/qwen3_pipeline/prefill_pp_decode_pp.py)

#### Producer (Chip 0) — Prefill 阶段（第319-458行）

```python
def producer(model_params, out_queue, in_queue, question_len, barrier):
    # 加载 Prefill Part 0
    prefill_model = tcim.runtime.load(model_params["prefill_path"], option_prefill)
    
    # 加载 Decode Part 0（用 dummy_tensors 跳过 KV Cache 初始化）
    option_decode.set_dummy_tensors(dummy_tensor_names)
    decode_model = tcim.runtime.load(model_params["decode_path"], option_decode)
    
    barrier.wait()  # 所有进程同步启动
    
    # === Prefill ===
    for round in range(prefill_loop_round):
        # Embedding → Prefill Model
        inputs_embeds = F.embedding(input_ids, embedding_weight)
        prefill_model.set_input(input_name, inputs_embeds)
        prefill_model.run()
        prefill_model.sync()
        
        output_data = prefill_model.get_output(output_name)  # hidden states
        out_queue.put((output_data, ...))                     # → 发给 Chip 1
    
    out_queue.put("END")
    
    # === KV Cache 拷贝：Prefill → Decode ===
    for i in range(total_layer_num):
        layer_name = prefill_model.get_input_name(i)
        if "cache" in layer_name:
            cache = prefill_model.get_dev_input(layer_name).to_host(True).numpy()
            decode_model.set_input(layer_name, cache)  # 写入 Decode 模型
    
    # === Decode ===
    while True:
        data = in_queue.get()                              # ← 从结果端收到的新 token
        if data == "END": break
        decode_model.set_input(input_name, output_data)
        decode_model.run()
        out_queue.put(output_data)                         # → 发给 Chip 1
```

#### Consumer (Chip 1..N-2) — Prefill + Decode（第508-605行）

```python
def consumer(model_params, in_queue, out_queue, barrier):
    # 同样加载 Prefill + Decode 两个模型
    
    # === Prefill ===
    while True:
        data = in_queue.get()                              # ← 接收 Chip i-1 的输出
        if data == "END": break
        prefill_model.set_input(input_name, output_data)   # 直接作为本 chip 输入
        prefill_model.run()
        prefill_model.sync()
        output_data = prefill_model.get_output(...)
        out_queue.put(output_data)                         # → 发给 Chip i+1
    
    # KV Cache 拷贝
    for layer_name in kvcache_layers:
        cache = prefill_model.get_dev_input(layer_name).to_host(True).numpy()
        decode_model.set_input(layer_name, cache)
    
    # === Decode ===
    while True:
        data = in_queue.get()                              # ← 接收新 token
        if data == "END": break
        decode_model.set_input(input_name, output_data)
        decode_model.run()
        out_queue.put(output_data)                         # → 发给 Chip i+1
```

#### result_shower (Chip N-1) — 采样 + Token 解码（第627-720行）

```python
def result_shower(model_params, in_queue, out_queue, barrier):
    # 不加载模型 — 只做采样和 token 解码
    samplingmanager = SamplingManager(temperature, top_k, top_p, ...)
    
    # 接收 Prefill 最终输出 → 采样第一个 token
    pre_data = ...  # 从 in_queue 接收 Chip N-2 的输出
    next_id = pre_data[0].argmax(-1)[0]
    
    # === Decode 循环（纯 CPU） ===
    while True:
        # 把当前 token embedding 发给 Producer → 流经整个 pipeline
        input_data = F.embedding(next_id, embedding_weight)
        out_queue.put((input_data.numpy(), ...))           # → Producer → Consumer链
        
        # 等待 pipeline 传回最后阶段的 hidden states
        decode_output = in_queue.get()                      # ← 经过整条 pipeline
        decode_next_id = samplingmanager.sample(decode_output)
        
        # 打印 token
        print(tokenizer.decode(decode_next_id))
```

**PP 模式下 Decode 的完整数据流：**

```
result_shower (Chip 3)
    │
    │ 1. embed(next_token)  → out_queue → ──────────────────┐
    │                                                        ▼
    │                                            Producer (Chip 0)
    │                                            decode_model.run()
    │                                                 │
    │                                            out_queue → Consumer (Chip 1)
    │                                                        decode_model.run()
    │                                                             │
    │                                                        out_queue → Consumer (Chip 2)
    │                                                                    decode_model.run()
    │                                                                         │
    │ 2. in_queue ← ─────────────────────────────────────────────────────────┘
    │    sample(output) → next_token
    │
    └── 循环直到 EOS
```

**性能数据：**

| 配置 | 30k input prefill | decode per token |
|---|---|---|
| 1 chip | 41.56s | baseline |
| 4 chips PP | 12.23s (**3.39x**) | 121.8 ms/token |
| 4 chips TP | 18.23s (2.28x) | 56.4 ms/token |

### 5.2 模式 B：Prefill PP + Decode TP（prefill_pp_decode_tp.py）

**Prefill 用 PP，Decode 切换为 TP。**

[`apis/inferences/qwen3_pipeline/prefill_pp_decode_tp.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/houmo-examples-xh2/apis/inferences/qwen3_pipeline/prefill_pp_decode_tp.py)

```python
def decode_runner(model_params, decode_in_queue, barrier, kvcache):
    """Decode 用多芯片 TP"""
    # 关键：用 DevManager 统一管理多设备
    dev_manager = tcim.runtime.DevManager(model_params["device_id"], "Xh2HalBackend")
    weight_manager = tcim.runtime.WeightManager(dev_manager)
    option_decode = tcim.runtime.Option(weight_manager)
    
    # 加载 decode.hmms（多设备 TP 模型，不是 per-part！）
    decode_model = tcim.runtime.load(model_params["decode_path"], option_decode)
    # model_params["decode_path"] = "qwen3-pipeline-8b_decode.hmms"
    
    # 从 kvcache_queue 接收所有芯片的 Prefill KV Cache
    while True:
        kvcache_dict = kvcache.get(timeout=1)
        if isinstance(kvcache_dict, dict):
            for key, value in kvcache_dict.items():
                decode_model.set_input(key, value)  # 写入 TP decode 模型
    
    # === Decode 循环 ===
    while True:
        # embed → TP decode (多芯片并行)
        input_data = F.embedding(next_id, embedding_weight)
        decode_model.set_input(input_name, input_data.numpy())
        decode_model.run()
        decode_model.sync()
        next_id = samplingmanager.sample(output_data)
```

**TP vs PP 在 Decode 阶段的区别：**

```
PP (prefill_pp_decode_pp.py):
  Chip 0 → Queue → Chip 1 → Queue → Chip 2 → Queue → Chip 3
  串行延迟累加 → 121.8 ms/token

TP (prefill_pp_decode_tp.py):
  Chip 0 ─┐
  Chip 1 ─┤  DevManager 统一调度
  Chip 2 ─┤  并行计算 → 56.4 ms/token
  Chip 3 ─┘
  并行 → 延迟更低
```

**为什么 Prefill 不用 TP 而用 PP？** Prefill 是 compute-bound（一次处理大量 token），TP 的通信开销会拖累。PP 的通信只有一次（hidden states 传输），prefill 效率更高。

---

## 六、Houmo PP vs vLLM PP 的区别

| | vLLM PP (GPU) | Houmo PP (NPU) |
|---|---|---|
| **通信方式** | NCCL P2P (GPU Direct RDMA) | `multiprocessing.Queue` (CPU 中转) |
| **Hidden States 传输** | GPU tensor → NCCL send/recv（~ns 级延迟） | GPU→CPU→Queue→CPU→GPU（~ms 级延迟） |
| **模型格式** | 单份 HuggingFace 权重，按层加载 | 多个独立 .hmm 编译产物，每个芯片独立 |
| **Prefill/Decode 关系** | 同一份权重，按 forward 模式切换 | 两套独立 .hmm（prefill_part.hmm + decode_part.hmm） |
| **Bubble 处理** | 微批次流水线（PP_size 个 micro-batch） | 单请求串行（bubble 利用率低） |
| **KV Cache** | 同一张卡上的同一个 tensor | Prefill→Decode 拷贝（CPU 中转 `to_host().numpy()`） |

**核心差异：** vLLM 的 PP 是**细粒度**的（每个 micro-batch 在 GPU 间直接传输），Houmo 的 PP 是**粗粒度**的（通过 CPU Queue 中转，prefill 和 decode 是两套独立编译产物）。这是因为 NPU 的静态图限制：prefill 和 decode 必须分开编译，不同芯片之间没有直接的 NPU-to-NPU 通信。

---

## 七、KV Cache 跨 chip 传递

PP 中 KV Cache 的分布和传递是一个重要细节：

```
Qwen3-8B, 32 层, 4 chip PP:

Chip 0: KV Cache for Block 0~7    ← 8 层的 K/V
Chip 1: KV Cache for Block 8~15   ← 8 层的 K/V
Chip 2: KV Cache for Block 16~23  ← 8 层的 K/V
Chip 3: KV Cache for Block 24~31  ← 8 层的 K/V

每个 chip 只管理自己那部分层的 KV Cache — 不需要跨 chip 共享。
因为 Attention 计算时：
  - Block 5 的 Attention 只需要 Chip 0 上的 K(0..5), V(0..5) 
  - Block 20 的 Attention 需要 Chip 2 上的所有 K(0..20), V(0..20)
    → 但是！PP 切分是按层顺序的，Block 20 属于 Chip 2，
    Chip 2 只知道自己的 KV Cache...
```

**等等，那 Attention 怎么从之前 chip 拿到 K/V？**

关键在于：**PP 是按层纵向切的，不是横向。** 每个 chip 上的 Attention 只需要自己的 K/V。因为：

- Chip 0 上 Layer 5 的 Attention 只需要 `K(previous_layers_on_chip0)` — 都在 Chip 0 上
- Chip 2 上 Layer 20 的 Attention 也只需要 `K(previous_layers_on_chip2)` — 也在 Chip 2 上

**PP 里的 Attention 不需要跨 chip 访问 KV Cache。** 因为 hidden states 已经经过前一层，包含了所有需要的信息。

---

## 八、PP vs TP vs DP 总结对比

```
                     DP                    TP                    PP
切分什么          数据(batch)          权重矩阵(列/行)       模型层(depth)
每个设备        完整模型副本          部分权重              部分模型层
通信量          梯度all-reduce      每层all-reduce         边界hidden states
               (训练才需要)         (推理也需要)            (非常少)
通信频率        1次/step            每层1次               1~3次/step
显存节省        无                  大(按TP度线性)        大(按PP度线性)
延迟影响        无                  小(增加了通信)         大(Bubble)
最佳场景        高吞吐              单层太大放不下         整个模型太大放不下
```

---

## 九、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **编译器团队** | PP 需要在编译期把 ONNX 切分成多个子模型 → `onnx.utils.extract_model` + 逐 part 编译；切分点的选取直接影响各 stage 的负载均衡 |
| **量化团队** | 量化在切分之前完成（`hmquant/` 下的量化 ONNX 作为切分输入），量化策略对所有 part 一致，但每个 part 的 calibration 可独立优化 |
| **Runtime/驱动** | PP 依赖进程间 Queue 通信 → CPU 中转模式。升级方向：NPU 间直接 DMA → 消除 CPU 中转延迟；`DevManager`(多设备协调) 是 TP decoupling 的关键 API |
| **性能优化** | Prefill PP + Decode TP 是当前最优组合（取各自的优势）；未来可以探索 Decode PP 的微批次流水线来减少 Bubble |

---

## 十、白话总结

```
Pipeline Parallel = 工厂流水线

单芯片 = 一个工人从原料做到成品（慢，但所有工序一个人会）
PP     = 流水线分 4 个工位：
          工位1: 粗加工 (Embedding + 前8层)
          工位2: 精加工 (中8层)   
          工位3: 超精加工 (后8层)
          工位4: 包装质检 (最后8层 + LM Head + 采样)

优点:
  - 每个工位只管理自己的工具(权重)，内存省 4x
  - 工位间只传递一个"半成品"(hidden states)，通信极少

缺点:
  - 流水线启动时有"空转"时间 (bubble)
  - 一个工位卡了，整条线都等它 (木桶效应)

Houmo 的做法:
  - Prefill: 4 芯片流水线 → 3.4x 加速
  - Decode:  4 芯片流水线 → 121 ms/token（Bubble 拖累）
            或 4 芯片 TP    → 56 ms/token （无 Bubble）
```

---

## 十一、思考题

1. **为什么 Houmo PP 模式下 Decode 的延迟（121 ms/token）比 TP 模式（56 ms/token）差这么多？**
   - 提示：Bubble + Queue 通信延迟 vs 并行计算

2. **PP 切分 ONNX 时，切分点为什么选在 `add_{2*k-1}` 而不是任意节点？**
   - 提示：ONNX DAG 结构，add 节点通常是残差连接后的汇合点

3. **为什么 Prefill 用 PP 比 TP 好，但 Decode 恰恰相反？**
   - 提示：Prefill 是 compute-bound（token 多），Decode 是 memory-bound（token 少）。TP 的通信开销在 Decode 时占比更小

4. **如果 Transformer Block 数量不能被 ndevice 整除怎么办？**
   - 提示：`build.py` 里有 `assert n_blocks % 2 == 0`，Houmo 直接 assert 要求整除。如果不整除，需要怎么做？

5. **Houmo PP 中 Queue 的 maxsize 设置为 5，如果改成 1 或 100 会有什么影响？**
   - 提示：太小 → 流水线频繁停顿；太大 → 内存浪费，且对 pipeline 平衡无帮助
