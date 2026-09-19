# CS336 课程 + nano-vllm 源码结合学习路线

> 目标：从"理解单个算子"到"理解完整推理引擎"，覆盖算子层 → 模型组装 → 引擎调度 → 性能优化全链路
> 适用角色：推理工具链 / 编译器 / 量化方向工程师

---

## 一、两个项目的关系

```
CS336 课程（教学版）                    nano-vllm（工程版）
───────────────────                   ───────────────────
chapter1/hw3/RMSnorm.py          ←→   nanovllm/layers/layernorm.py
chapter1/hw3/rope.py             ←→   nanovllm/layers/rotary_embedding.py
chapter1/hw3/SwiGLU.py           ←→   nanovllm/layers/activation.py
chapter1/hw3/scaled_dot_product_attention.py  ←→  nanovllm/layers/attention.py
chapter1/hw3/causal_multi_head_attention.py   ←→  nanovllm/layers/attention.py
chapter1/hw3/transformer_block.py  ←→  nanovllm/models/qwen3.py (Qwen3DecoderLayer)
chapter1/hw7/transformermodule.py  ←→  nanovllm/models/qwen3.py (Qwen3ForCausalLM)
chapter1/hw6/inference.py (top-p)  ←→  nanovllm/layers/sampler.py
chapter1/hw1/pair_all_bpe_tokenzier.py  ←→  (nano-vllm 用 HF AutoTokenizer)
chapter2/hw1/triton_causal_forawrdflash_attention.py  ←→  nanovllm/layers/attention.py (flash_attn + store_kvcache_kernel)
```

**核心差异**：CS336 教你"算子怎么写"，nano-vllm 教你"算子怎么被引擎调用"。

---

## 二、学习路线总览

```
阶段1: 算子层对比（CS336 hw3 ↔ nano-vllm layers/）
  ↓
阶段2: 模型组装对比（CS336 hw7 ↔ nano-vllm models/qwen3.py）
  ↓
阶段3: 推理引擎核心（nano-vllm engine/，CS336 未覆盖）
  ↓
阶段4: Kernel 与性能优化（CS336 hw2 ↔ nano-vllm attention.py + model_runner.py）
  ↓
阶段5: 端到端跑通（nano-vllm example.py + bench.py）
```

---

## 三、阶段1：算子层对比（CS336 hw3 ↔ nano-vllm layers/）

### 1.1 RMSNorm

| 对比项 | CS336 [RMSnorm.py](chapter1/hw3/RMSnorm.py) | nano-vllm [layernorm.py](nano-vllm/nanovllm/layers/layernorm.py) |
|---|---|---|
| 数学公式 | `x / sqrt(mean(x²) + eps) * weight` | 相同 |
| 精度处理 | 转 float32 计算后转回 | 相同 |
| **残差融合** | 无 | `add_rms_forward(x, residual)` 把残差加法和 norm 融合在一起 |
| torch.compile | 无 | 有 `@torch.compile` 装饰器 |
| **学习重点** | 理解 RMSNorm 的数学定义 | 理解推理引擎中为什么需要残差融合（减少一次 kernel launch） |

**阅读顺序**：
1. 读 CS336 `RMSnorm.py`，理解 `variance = x.pow(2).mean(-1)` + `rsqrt` + `weight * x`
2. 读 nano-vllm `layernorm.py`，对比 `rms_forward` 和 `add_rms_forward` 的差异
3. 思考：量化场景下，`rsqrt(var + eps)` 的精度如何保证？

### 1.2 RoPE 旋转位置编码

| 对比项 | CS336 [rope.py](chapter1/hw3/rope.py) | nano-vllm [rotary_embedding.py](nano-vllm/nanovllm/layers/rotary_embedding.py) |
|---|---|---|
| 频率计算 | `1.0 / (theta ** (arange(0, d_k, 2) / d_k))` | 相同 |
| cos/sin cache | `register_buffer("cos_cache")`, `register_buffer("sin_cache")` | 合并为一个 `cos_sin_cache`（cat 拼接） |
| **旋转应用方式** | 复数乘法实现（`x1 * cos - x2 * sin`） | 相同，但用 `chunk` 分割 + `cat` 拼接 |
| **调用方式** | `forward(x, token_positions)` | `forward(positions, query, key)` 同时处理 Q 和 K |
| torch.compile | 无 | 有 |
| **学习重点** | 理解旋转位置编码的数学推导 | 理解推理引擎中 RoPE 如何被批量调用 |

**阅读顺序**：
1. 读 CS336 `rope.py`，理解 `sinusoids = outer(positions, freqs)` 和旋转公式
2. 读 nano-vllm `rotary_embedding.py`，注意 `apply_rotary_emb` 的 chunk/cat 实现
3. 思考：cos/sin cache 在推理时如何被索引？量化时 cos/sin 是否需要高精度？

### 1.3 SwiGLU 激活

| 对比项 | CS336 [SwiGLU.py](chapter1/hw3/SwiGLU.py) | nano-vllm [activation.py](nano-vllm/nanovllm/layers/activation.py) |
|---|---|---|
| 公式 | `w2(silu(w1(x)) * w3(x))` | `silu(x) * y`（输入已包含 gate 和 up） |
| **权重布局** | 三个独立 Linear：w1, w2, w3 | gate_up_proj 合并成一个矩阵，`chunk(2, -1)` 拆分 |
| **学习重点** | 理解 SwiGLU 门控机制 | 理解为什么推理引擎要把 gate 和 up 融合（减少一次 matmul kernel） |

**阅读顺序**：
1. 读 CS336 `SwiGLU.py`，理解 `silu(x) = x * sigmoid(x)` 和三个线性变换
2. 读 nano-vllm `activation.py`，注意输入是 `chunk(2, -1)` 拆分后的 gate 和 up
3. 在 `qwen3.py` 中看 `Qwen3MLP`：`gate_up_proj` → `SiluAndMul` → `down_proj`

### 1.4 Attention（重点）

| 对比项 | CS336 [causal_multi_head_attention.py](chapter1/hw3/causal_multi_head_attention.py) | nano-vllm [attention.py](nano-vllm/nanovllm/layers/attention.py) |
|---|---|---|
| Attention 实现 | 手写 `matmul(Q, K.T) / sqrt(d_k)` + softmax + matmul V | 调用 `flash_attn_varlen_func` / `flash_attn_with_kvcache` |
| **KV Cache** | 无 | 有：`store_kvcache` Triton kernel 把 K/V 写入 paged cache |
| **Prefill vs Decode** | 不区分 | 区分：prefill 用 `varlen_func`，decode 用 `with_kvcache` |
| **Prefix Caching** | 无 | 有：检查 `block_tables` 是否非空 |
| GQA | 无 | 有：`num_kv_heads` 可小于 `num_heads` |
| **学习重点** | 理解 Attention 的数学定义和 mask 机制 | 理解推理引擎中 Attention 如何与 KV Cache 交互 |

**阅读顺序**：
1. 读 CS336 `scaled_dot_product_attention.py`，理解 `QK^T / sqrt(d_k)` + mask + softmax + V
2. 读 CS336 `causal_multi_head_attention.py`，理解 QKV 投影 + 分头/合头
3. 读 CS336 `causal_multi_head_attention_no_weight.py`（hw7），理解权重内置版本
4. 读 nano-vllm `attention.py`：
   - `store_kvcache_kernel`：Triton kernel，把 K/V 按 slot_mapping 写入 paged cache
   - prefill 路径：`flash_attn_varlen_func`（变长序列批量 prefill）
   - decode 路径：`flash_attn_with_kvcache`（从 paged cache 读取 K/V）
5. 思考：slot_mapping 是怎么计算的？为什么需要它？（答案在 model_runner.py 的 `prepare_prefill`/`prepare_decode`）

### 1.5 Softmax & Sampler

| 对比项 | CS336 [softmax.py](chapter1/hw3/softmax.py) + [inference.py](chapter1/hw6/inference.py) | nano-vllm [sampler.py](nano-vllm/nanovllm/layers/sampler.py) |
|---|---|---|
| Softmax 实现 | `exp(x - max) / sum(exp(x - max))` | `torch.softmax(logits / temperature)` |
| 采样方式 | top-p sampling（排序 + 累积概率 + 截断） | **Gumbel 采样**：`probs / exponential_(1).argmax()`（更高效） |
| **学习重点** | 理解数值稳定 softmax 和 top-p | 理解推理引擎用的高效采样策略 |

**阅读顺序**：
1. 读 CS336 `softmax.py`，理解 max-subtract trick
2. 读 CS336 `inference.py`，理解 temperature scaling 和 top-p sampling
3. 读 nano-vllm `sampler.py`，理解 Gumbel-max 采样：`probs.div_(exponential_(1)).argmax()`

---

## 四、阶段2：模型组装对比（CS336 hw7 ↔ nano-vllm models/qwen3.py）

| 对比项 | CS336 [transformermodule.py](chapter1/hw7/transformermodule.py) | nano-vllm [qwen3.py](nano-vllm/nanovllm/models/qwen3.py) |
|---|---|---|
| 结构 | Embedding → N×Block → Linear | VocabParallelEmbedding → N×DecoderLayer → ParallelLMHead |
| **残差连接** | `x = x + attn(norm(x))` | `hidden_states, residual = layernorm(x, residual)` 融合残差 |
| **Tensor Parallel** | 无 | QKVParallelLinear / RowParallelLinear（all-reduce 通信） |
| **GQA** | 无 | `num_kv_heads != num_heads`，Q/KV split 大小不同 |
| **Weight Tying** | 无 | `lm_head.weight = embed_tokens.weight`（tie_word_embeddings） |
| **QK Norm** | 无 | `q_norm` / `k_norm`（Qwen3 特有） |
| **学习重点** | 理解基础 Transformer 堆叠 | 理解工程版的多卡并行、GQA、weight tying |

**阅读顺序**：
1. 读 CS336 `transformer_no_weight_block.py`，理解 pre-norm 残差结构
2. 读 CS336 `transformermodule.py`，理解 Embedding → Blocks → LM Head
3. 读 nano-vllm `qwen3.py`：
   - `Qwen3DecoderLayer.forward`：注意 `residual` 的传递方式
   - `Qwen3Attention`：QKV 合并投影 + GQA + RoPE + QK Norm
   - `Qwen3ForCausalLM`：weight tying
4. 读 nano-vllm [linear.py](nano-vllm/nanovllm/layers/linear.py)：理解 ColumnParallel / RowParallel / QKVParallel 的权重切分和通信

---

## 五、阶段3：推理引擎核心（nano-vllm engine/，CS336 未覆盖）

这是 nano-vllm 相对 CS336 的**核心增量**——课程只教算子，没教引擎调度。

### 3.1 请求生命周期

```
用户 prompt
  │
  ▼
llm_engine.py: add_request() → tokenize → 创建 Sequence → 加入 Scheduler.waiting
  │
  ▼
llm_engine.py: step() → scheduler.schedule() → 返回 (seqs, is_prefill)
  │                                           ↓
  │              ┌─── is_prefill=True ──→ model_runner.prepare_prefill()
  │              │                         (拼装 input_ids, positions, cu_seqlens, slot_mapping)
  │              └─── is_prefill=False ─→ model_runner.prepare_decode()
  │                                        (拼装 last_token, block_tables, context_lens)
  ▼
model_runner.py: run() → run_model() → model.forward() → sampler → token_ids
  │
  ▼
scheduler.py: postprocess() → 更新 KV cache hash → append_token → 检查 EOS/max_tokens
  │
  ▼
循环直到所有 Sequence 状态变为 FINISHED
```

### 3.2 核心文件阅读

#### [llm_engine.py](nano-vllm/nanovllm/engine/llm_engine.py) — 引擎入口

**重点看**：
- `__init__`：Engine = Scheduler + ModelRunner + Tokenizer，TP 多进程用 SharedMemory 通信
- `step()`：一个推理 step = schedule + run + postprocess
- `generate()`：批量提交 → 循环 step → 收集输出，带 prefill/decode 吞吐统计

#### [scheduler.py](nano-vllm/nanovllm/engine/scheduler.py) — 调度器

**重点看**：
- `schedule()`：
  - **Prefill 阶段**：从 waiting 队列取 seq，检查 `max_num_batched_tokens` 预算，支持 chunked prefill
  - **Decode 阶段**：从 running 队列取 seq，每个 seq 生成 1 个 token
  - **抢占机制**：`preempt()` — KV cache 不够时把 running seq 退回 waiting，释放 block
- `postprocess()`：更新 block hash、append token、检查终止条件

**思考**：
- 为什么 prefill 优先于 decode？（prefill 是计算密集，decode 是访存密集，混跑会降低 GPU 利用率）
- chunked prefill 的条件是什么？（`remaining < num_tokens and scheduled_seqs` 为空时允许）

#### [block_manager.py](nano-vllm/nanovllm/engine/block_manager.py) — PagedAttention

**重点看**：
- `Block`：每个 block 有 `block_id`, `ref_count`, `hash`, `token_ids`
- `can_allocate()`：检查是否有足够空闲 block，同时计算 prefix cache 命中数
- `allocate()`：分配 block，已缓存的 block 引用计数 +1，新 block 从 free 列表取
- `can_append()` / `may_append()`：decode 时检查是否需要新 block（当前 block 写满时）
- `hash_blocks()`：用 xxhash 计算 block 内容哈希，用于 prefix caching
- `preempt()` → `deallocate()`：释放 block，ref_count 归零时归还 free 列表

**思考**：
- prefix caching 的哈希为什么是链式的（`compute_hash(token_ids, prefix)`）？（因为前缀必须连续匹配）
- ref_count 的作用是什么？（多个 seq 共享同一个 prefix block）

#### [model_runner.py](nano-vllm/nanovllm/engine/model_runner.py) — 模型执行器

**重点看**：
- `allocate_kv_cache()`：根据剩余显存计算可分配的 KV cache block 数量
- `prepare_prefill()`：拼装 `cu_seqlens_q/k`（变长序列累积长度）、`slot_mapping`（KV cache 写入位置）
- `prepare_decode()`：拼装 `context_lens`、`block_tables`
- `capture_cudagraph()`：为不同 batch size 预捕获 CUDA Graph，避免 kernel launch 开销
- `run_model()`：prefill 或 >512 token 用 eager，decode 用 CUDA Graph replay

**思考**：
- `slot_mapping` 是怎么映射到 KV cache 物理位置的？（`block_table[i] * block_size + offset`）
- CUDA Graph 为什么只用于 decode？（prefill 序列长度变化大，难以静态捕获）

#### [sequence.py](nano-vllm/nanovllm/engine/sequence.py) — 序列状态

**重点看**：
- `Sequence` 状态机：`WAITING → RUNNING → FINISHED`
- `block_table`：记录这个 seq 用了哪些 KV cache block
- `__getstate__` / `__setstate__`：pickle 序列化，用于 TP 多进程通信

#### [context.py](nano-vllm/nanovllm/utils/context.py) — 全局上下文

用全局变量传递 prefill/decode 上下文信息（`cu_seqlens`, `slot_mapping`, `block_tables`），避免修改模型 forward 签名。

---

## 六、阶段4：Kernel 与性能优化

### 4.1 Triton Kernel 对比

| 对比项 | CS336 [triton_causal_forawrdflash_attention.py](chapter2/hw1/triton_causal_forawrdflash_attention.py) | nano-vllm [attention.py](nano-vllm/nanovllm/layers/attention.py) |
|---|---|---|
| Kernel 功能 | Flash Attention 前向（Q×K^T 分块 + online softmax + ×V） | `store_kvcache_kernel`（把 K/V 写入 paged cache） |
| 复杂度 | 高（完整的 Flash Attention 算法） | 低（简单的散写操作） |
| **学习重点** | tiling、online softmax、block pointer | slot_mapping 如何索引 KV cache |

**阅读顺序**：
1. 读 CS336 Triton kernel，理解 `tl.make_block_ptr` + 循环遍历 K/V tile + online softmax rescaling
2. 读 nano-vllm `store_kvcache_kernel`，理解 `slot = load(slot_mapping_ptr + idx)` → `store(k_cache_ptr + slot * D, key)`
3. 思考：为什么 nano-vllm 不自己写 Flash Attention kernel？（用 flash_attn 库的 `flash_attn_varlen_func`）

### 4.2 CUDA Graph

nano-vllm [model_runner.py](nano-vllm/nanovllm/engine/model_runner.py) 的 `capture_cudagraph()`：

```python
# 为不同 batch size 预捕获计算图
for bs in reversed(self.graph_bs):   # [1, 2, 4, 8, 16, 32, ...]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    self.graphs[bs] = graph

# decode 时 replay
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
graph.replay()  # 零 kernel launch 开销
```

**为什么重要**：decode 阶段每个 seq 只生成 1 个 token，计算量极小，kernel launch 开销占比高。CUDA Graph 把整条计算图录下来，一次 replay 搞定。

### 4.3 性能 profiling 工具

参见之前整理的 profiling 工具表（nsys / ncu / torch.profiler / NVTX / vLLM 内置统计 / timeit）。

---

## 七、阶段5：端到端跑通

### 5.1 运行 example.py

[nano-vllm/example.py](nano-vllm/example.py)：

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)
```

**建议**：
1. 先用 `enforce_eager=True` 跑通（关闭 CUDA Graph）
2. 在关键位置加 print / NVTX 标记，观察 prefill 和 decode 的交替
3. 用 `nsys profile` 采集一次，对照时间线理解调度流程

### 5.2 运行 bench.py

[nano-vllm/bench.py](nano-vllm/bench.py) 测吞吐，对比 eager vs CUDA Graph 的性能差异。

---

## 八、每周学习计划

| 周次 | CS336 课程 | nano-vllm 源码 | 产出 |
|---|---|---|---|
| **第1周** | hw3: RMSNorm + Softmax | layers/layernorm.py + layers/sampler.py | 理解 norm 和采样的教学版 vs 工程版差异 |
| **第2周** | hw3: RoPE + SwiGLU | layers/rotary_embedding.py + layers/activation.py | 理解位置编码和门控激活的工程优化 |
| **第3周** | hw3: Attention（全部） | layers/attention.py + models/qwen3.py (Attention部分) | 理解 KV Cache + Flash Attention + GQA |
| **第4周** | hw7: Transformer 组装 | models/qwen3.py + layers/linear.py | 理解模型组装 + Tensor Parallel |
| **第5周** | hw6: 推理 | layers/sampler.py | 理解采样策略差异 |
| **第6周** | hw2: Flash Attention Triton | layers/attention.py (store_kvcache_kernel) | 理解 Triton kernel 在引擎中的角色 |
| **第7周** | — | engine/scheduler.py + engine/sequence.py | 理解 continuous batching + 调度 |
| **第8周** | — | engine/block_manager.py | 理解 PagedAttention + prefix caching |
| **第9周** | — | engine/model_runner.py + engine/llm_engine.py | 理解 CUDA Graph + 显存管理 + 端到端 |
| **第10周** | — | example.py + bench.py + nsys profiling | 跑通 + 性能分析 |

---

## 九、关键问题清单

学完每个阶段后，确认能回答以下问题：

### 算子层
- [ ] RMSNorm 为什么先转 float32？量化时 scale 怎么融合？
- [ ] RoPE 的 cos/sin cache 在推理时如何被索引？量化时需要高精度吗？
- [ ] SwiGLU 的 gate 和 up 为什么在推理时要融合成一个矩阵？
- [ ] Attention 的 `1/sqrt(d_k)` 缩放在量化时为什么精度敏感？

### 模型层
- [ ] GQA 的 KV head 数少于 Q head 数时，权重怎么切分？
- [ ] Weight Tying 对量化有什么影响？（embedding 和 lm_head 共享权重）
- [ ] Tensor Parallel 的 ColumnParallel 和 RowParallel 通信方式有什么区别？

### 引擎层
- [ ] PagedAttention 的 block_size 怎么选？太大太小各有什么问题？
- [ ] Continuous batching 中 prefill 和 decode 为什么分开调度？
- [ ] 抢占机制什么时候触发？被抢占的 seq 的 KV cache 怎么处理？
- [ ] Prefix caching 的哈希为什么是链式的？
- [ ] CUDA Graph 为什么只用于 decode 而不用于 prefill？
- [ ] slot_mapping 是怎么计算的？它解决什么问题？

### 性能层
- [ ] decode 阶段的瓶颈是计算还是访存？为什么？
- [ ] nsys 时间线上 prefill 和 decode 的 kernel 密度有什么区别？
- [ ] 如何用 torch.profiler 定位最耗时的算子？
