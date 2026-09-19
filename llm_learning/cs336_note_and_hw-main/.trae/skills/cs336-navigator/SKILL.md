---
name: "cs336-navigator"
description: "Navigates the CS336 course + nano-vllm codebase for LLM inference engineers. Invoke when user asks about Transformer operators, BPE tokenizer, Flash Attention, vLLM engine internals, PagedAttention, KV Cache, or wants to locate/explain code in this project."
---

# CS336 Navigator

This skill helps navigate the CS336 course notes/homework and nano-vllm source code, with a focus on **inference toolchain / compiler / quantization** perspectives.

## Project Structure

```
cs336_note_and_hw-main/
├── chapter1/          # Transformer basics (operators, tokenizer, inference)
│   ├── hw1/           # BPE Tokenizer training
│   ├── hw2/           # Tokenizer encode
│   ├── hw3/           # Core operators: RMSNorm, Softmax, Attention, RoPE, SwiGLU
│   ├── hw4/           # Training: AdamW, CrossEntropy, GradientClip, LRSchedule (SKIP for inference)
│   ├── hw5/           # Checkpoint, DataLoader (SKIP for inference)
│   ├── hw6/           # Inference: top-p sampling, temperature, decode loop
│   └── hw7/           # Full model assembly, training script, inference
├── chapter2/          # Systems optimization
│   └── hw1/           # Flash Attention Triton kernel (forward/backward), NVTX, timeit
├── chapter3/          # Scaling Law (isoflop curves)
├── chapter4/          # Data pipeline (SKIP for inference)
├── chapter5/          # LLM alignment (SFT, GRPO, vLLM usage)
├── chapter5-supplement/ # DPO (SKIP for inference)
├── nano-vllm/         # Lightweight vLLM reimplementation (~1200 lines)
│   └── nanovllm/
│       ├── engine/    # Scheduler, BlockManager, ModelRunner, Sequence
│       ├── layers/    # Attention, RMSNorm, RoPE, SiluAndMul, Sampler, Linear
│       └── models/    # Qwen3 model implementation
├── 笔记/              # Course notes (PDF, Chinese)
└── 原始官方讲义/       # Official assignment PDFs
```

## Learning Path Files

- `针对推理工具链工程师的学习路径.md` — Priority-based learning path for inference engineers
- `CS336与nano-vllm源码结合学习路线.md` — Combined CS336 + nano-vllm learning path with weekly plan

## Key Code Mappings (CS336 → nano-vllm)

| Concept | CS336 file | nano-vllm file |
|---------|-----------|----------------|
| RMSNorm | chapter1/hw3/RMSnorm.py | nanovllm/layers/layernorm.py |
| RoPE | chapter1/hw3/rope.py | nanovllm/layers/rotary_embedding.py |
| SwiGLU | chapter1/hw3/SwiGLU.py | nanovllm/layers/activation.py |
| Attention | chapter1/hw3/causal_multi_head_attention.py | nanovllm/layers/attention.py |
| Model assembly | chapter1/hw7/transformermodule.py | nanovllm/models/qwen3.py |
| Sampling | chapter1/hw6/inference.py | nanovllm/layers/sampler.py |
| Flash Attention | chapter2/hw1/triton_causal_forawrdflash_attention.py | nanovllm/layers/attention.py |
| BPE Tokenizer | chapter1/hw1/pair_all_bpe_tokenzier.py | (uses HF AutoTokenizer) |

## nano-vllm Engine Architecture

```
LLM (llm.py) → LLMEngine (engine/llm_engine.py)
  ├── Scheduler (engine/scheduler.py)     # continuous batching, prefill/decode scheduling
  │    └── BlockManager (engine/block_manager.py)  # PagedAttention, prefix caching
  ├── ModelRunner (engine/model_runner.py) # CUDA Graph, KV cache allocation, TP
  │    └── Qwen3ForCausalLM (models/qwen3.py)
  │         ├── layers: attention, rotary_embedding, activation, layernorm, linear, sampler
  │         └── utils: context (global context passing), loader (weight loading)
  └── Sequence (engine/sequence.py)        # request state machine
```

## Inference Request Flow

```
prompt → tokenize → Sequence → Scheduler.waiting
  → schedule() [prefill: batch tokens / decode: 1 token per seq]
  → ModelRunner.prepare_prefill/decode() [build input_ids, positions, slot_mapping, block_tables]
  → ModelRunner.run_model() [model.forward() or CUDA Graph replay]
  → Sampler [temperature + softmax + sampling]
  → Scheduler.postprocess() [update KV cache hash, append token, check EOS]
  → loop until FINISHED
```

## Key Concepts to Explain

- **PagedAttention**: KV cache split into fixed-size blocks, managed by BlockManager with ref counting
- **Continuous Batching**: prefill and decode scheduled separately; prefill is compute-bound, decode is memory-bound
- **Prefix Caching**: block content hashed (xxhash, chained) for prefix reuse across requests
- **CUDA Graph**: pre-capture decode computation graph for different batch sizes, replay with zero kernel launch overhead
- **slot_mapping**: maps logical token positions to physical KV cache slots (`block_id * block_size + offset`)
- **GQA (Grouped Query Attention)**: num_kv_heads < num_heads, K/V shared across query head groups
- **Tensor Parallel**: ColumnParallel (split output, no comm) + RowParallel (split input, all-reduce)
