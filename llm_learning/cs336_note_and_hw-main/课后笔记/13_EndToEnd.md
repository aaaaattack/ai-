# 第13课：端到端 — 使用与性能验证

> 学习路线：阶段5 端到端跑通
> 对应文件：`nano-vllm/example.py` + `nano-vllm/bench.py`

---

## 一、example.py — 最小使用示例

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 1. 加载模型
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 2. 定义采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 3. 构造 prompt（用 chat_template 格式化）
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # 4. 生成
    outputs = llm.generate(prompts, sampling_params)

    # 5. 打印结果
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
```

### 一行 `llm.generate()` 背后的完整链路

```
用户调用 llm.generate(prompts, sampling_params)
    │
    ├─ [LLMEngine.generate]  --------- 循环直到所有请求完成
    │   ├─ tokenize → Sequence → Scheduler.add()
    │   │
    │   └─ while not done:
    │       ├─ [Scheduler.schedule]       → 决定本轮 prefill/decode 哪些 seq
    │       ├─ [ModelRunner.run]
    │       │   ├─ prepare_prefill/decode → 拼装 input_ids, slot_mapping, block_tables
    │       │   ├─ run_model              → prefill=eager, decode=CUDA Graph replay
    │       │   └─ Sampler                → temperature + softmax + Gumbel-max
    │       └─ [Scheduler.postprocess]    → 更新 KV Cache hash, check EOS
    │
    └─ detokenize → 返回 {"text": "...", "token_ids": [...]}
```

### `output['text']` 和 `output['token_ids']` 的来源

```python
# LLMEngine.generate() 的返回 (llm_engine.py:89-90):
return [{"text": self.tokenizer.decode(token_ids),
         "token_ids": token_ids}
        for token_ids in outputs]
```

---

## 二、bench.py — 性能基准测试

```python
import os, time
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    seed(0)

    # 测试参数
    num_seqs = 256            # 256 个请求
    max_input_len = 1024      # 输入 100-1024 tokens
    max_ouput_len = 1024      # 输出 100-1024 tokens

    # 加载模型（开启 CUDA Graph）
    llm = LLM("~/huggingface/Qwen3-0.6B/",
              enforce_eager=False,    # ← CUDA Graph ON
              max_model_len=4096)

    # 生成随机 prompt（直接用 token IDs，跳过 tokenize）
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                      max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]

    # Warmup（第一次运行有 kernel 编译开销）
    llm.generate(["Benchmark: "], SamplingParams())

    # 正式测试
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, "
          f"Throughput: {throughput:.2f}tok/s")
```

### 基准结果（官方）

| 推理引擎 | 输出 Tokens | 耗时 | 吞吐量 |
|---|---|---|---|
| vLLM | 133,966 | 98.37s | 1361.84 tok/s |
| **nano-vllm** | 133,966 | 93.41s | **1434.13 tok/s** |

nano-vllm 以 1200 行代码达到甚至超越了标准 vLLM 的吞吐。CUDA Graph 在 decode 阶段的 kernel launch 消除是关键优化。

### 测试参数分析

```
num_seqs = 256          → Scheduler 的 waiting queue 初始有 256 个请求
max_input_len = 1024    → prompt 长度 100-1024 随机
max_output_len = 1024   → 生成长度 100-1024 随机
ignore_eos = True       → 不提前终止（保证 output 长度可控）

enforce_eager = False   → CUDA Graph ON
                        = Decode 阶段 batch 1~512 用预录制图 → 零 launch 开销
```

### Warmup 为什么必要？

```python
llm.generate(["Benchmark: "], SamplingParams())
# 第一次运行：
#   1. 加载模型权重到 GPU
#   2. cuDNN/cuBLAS 首次编译 kernel
#   3. allocate_kv_cache（显存分配）
#   4. capture_cudagraph（预录制计算图）
#
# 后续运行：
#   以上一次性开销已消除 → 实测的是纯推理吞吐
```

---

## 三、全链路学习回顾

```
你从零学到的 13 门课，组成一条完整的推理引擎链路：

用户 prompt
  ↓ tokenize (BPE)
  ↓ Sequence WAITING
  ↓
Scheduler.schedule()     ← 第10课 Continuous Batching
  ↓ 决定 prefill/decode
  ↓
ModelRunner.run()
  ├─ prepare_prefill     ← 第12课 cu_seqlens + slot_mapping
  │   input_ids → Embedding (查表)
  │     ↓
  │   N × TransformerBlock    ← 第6课 RMSNorm + Attention + SwiGLU
  │     ├─ RMSNorm        ← 第1课
  │     ├─ QKV Proj → RoPE → Attention    ← 第2,5课
  │     │    └─ Flash Attention / PagedAttention   ← 第8,11课
  │     └─ SwiGLU         ← 第3课
  │     ↓
  │   LM Head → logits
  │     ↓
  │   Sampler             ← 第4,7课 Softmax + Gumbel-max
  │     ↓ next_token
  │
  ├─ prepare_decode       ← 第12课 last_token + block_tables
  │    ↓
  │   CUDA Graph replay   ← 第12课 零 launch 开销
  │    ↓
  │   ...同上 Block 链路...
  │    ↓
  │   Sampler → next_token
  │
  └─ Scheduler.postprocess()
      ├─ hash_blocks (prefix cache)    ← 第11课
      ├─ append_token
      └─ check EOS / max_tokens
  ↓
循环直到 FINISHED → detokenize → 输出文本
```

```
性能观测 (第9课):
  nsys / torch.profiler / timeit / NVTX
  逐层下钻: 端到端 → 阶段 → 算子 → kernel

Houmo 产线落地 (附录):
  TCIM Runtime → HLIEvLLM → helion NPU kernel
  静态图 vs 动态图 → Software DMA Assembly → 借壳 vLLM
```
