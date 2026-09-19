# nano-vllm vs HLIEvLLM 对比分析

> nano-vllm: 教学版 vLLM 核心，~1200 行 Python  
> HLIEvLLM: 后摩智能生产级 vLLM fork，数万行 + 多个私有包

---

## 一、定位差异

```
nano-vllm:  "如果我来写一个最简 vLLM，需要多少行代码？"
            → 1200 行讲清楚 Scheduler + PagedAttention + CUDA Graph

HLIEvLLM:   "如何在生产级 vLLM 上跑后摩 NPU？"
            → 借壳标准 vLLM，替换算子执行为 NPU
```

---

## 二、功能对比

### 2.1 引擎核心（两者共有）

| 功能 | nano-vllm | HLIEvLLM | 备注 |
|---|---|---|---|
| Scheduler (continuous batching) | 200 行 Python | vLLM v1/sched (~千行 C++) | nano-vllm 是教学简化版 |
| PagedAttention (BlockManager) | 120 行 Python | vLLM KVCacheManager (~数百行) | 两者核心逻辑一致 |
| Prefix Caching | xxhash 链式哈希 | 同，但支持更多缓存策略 | nano-vllm 的实现已足够理解 |
| CUDA Graph | capture + replay | capture + replay | 完全相同 |
| ModelRunner | prepare_prefill/decode + run_model | GPUModelRunner (多层优化) | nano-vllm 更易读 |

### 2.2 HLIEvLLM 额外拥有的

| 功能 | 说明 | nano-vllm 有没有 |
|---|---|---|
| **多硬件后端** | NVIDIA GPU / AMD GPU / Intel GPU / CPU / TPU | ❌ 仅 CUDA |
| **Houmo NPU 后端** | helion + vllm_houmo + tcim_runtime | ❌ |
| **GGUF 模型格式** | 支持 .gguf 文件（内嵌编译产物） | ❌ |
| **vLLM 平台插件系统** | entry_points 注册自定义平台 | ❌ |
| **LoRA** | 低秩适配微调，运行时热加载 | ❌ |
| **Speculative Decoding** | 草稿模型 + 验证，加速解码 | ❌ |
| **多模态** | 文本 + 图像/音频/视频 | ❌ |
| **Structured Outputs** | JSON Schema / Regex 约束生成 | ❌ |
| **KV Cache Offloading** | GPU → CPU → Disk 多级缓存 | ❌ |
| **Tensor Parallel** | NCCL all_reduce (生产级) | ❌ (仅 SharedMemory demo) |
| **Pipeline Parallel** | 多卡流水线 | ❌ |
| **量化推理** | GPTQ/AWQ/FP8/W8A8 等 | ❌ |
| **HTTP Server** | OpenAI 兼容 API | ❌ |
| **生产级 Profiling** | torch.profiler + nsys + ncu 全链路 | ❌ (仅 timeit) |
| **Auto-Tune** | Helion kernel 自动调优 | ❌ |

---

## 三、代码规模对比

| 维度 | nano-vllm | HLIEvLLM |
|---|---|---|
| 总代码行数 | ~1,200 行 | ~数十万行 |
| 引擎核心文件 | 6 个 .py | ~50+ 个 .py + C++/CUDA |
| 支持模型 | 1 个（Qwen3-0.6B） | 几乎所有 HuggingFace 模型 |
| 外部依赖 | torch, flash_attn, transformers | + helion, vllm_houmo, NCCL, cuBLAS, ... |

---

## 四、你学到的内容如何映射到 HLIEvLLM

```
你学的 nano-vllm                →  HLIEvLLM 对应部分

Scheduler.schedule()            →  vllm/v1/core/sched/scheduler.py
  prefill/decode 两阶段             prefill/decode 两阶段 (完全一致)
  token budget 管理                 token budget 管理
  chunked prefill                   chunked prefill
  preempt 抢占                     preempt 抢占

BlockManager                     →  vllm/v1/core/kv_cache_manager.py
  Block Pool + 引用计数            Block Pool + 引用计数 (完全一致)
  xxhash 链式哈希                  hash-based prefix cache
  can_allocate / allocate / free   allocate_slots / free

ModelRunner                      →  vllm/v1/worker/gpu_model_runner.py
  prepare_prefill/decode           _prepare_inputs (更复杂但逻辑一致)
  run_model (eager vs graph)       execute_model (CUDA Graph 同样)
  capture_cudagraph                capture_model (同样逻辑)

nano-vllm的 CUDA kernel          →  Houmo的 Helion NPU kernel
  flash_attn_varlen_func            helion.silu_mul_fp8
  flash_attn_with_kvcache           helion.rms_norm_fp8
  store_kvcache_kernel              ... 等 NPU 原生 kernel
```

---

## 五、HLIEvLLM 的"魔改"到底在哪

```
HLIEvLLM 目录结构:
├── vllm/                     ← 标准 vLLM 代码 (微小改动)
│   ├── engine/arg_utils.py   ← GGUF 检测 (新增 30 行)
│   ├── kernels/helion/       ← Helion kernel 配置 (全新)
│   └── ...其余全是标准 vLLM
│
├── 未包含在本 repo 的私有包:
│   ├── vllm_houmo/           ← GGUF 解析 + NPU 执行调度
│   ├── helion==1.0.0         ← NPU kernel 后端
│   └── vllm-houmo-plugin     ← vLLM 平台插件 (entry_points 注册)
│
└── houmo_examples/           ← 使用示例
```

**结论**：HLIEvLLM 的 95% 代码是标准 vLLM。真正"魔改"的部分不超过 500 行业务代码 + 3 个私有包。

---

## 六、一句话总结

```
nano-vllm 教你"vLLM 为什么这么设计"（1200 行讲清楚核心思想）
HLIEvLLM 告诉你"这些设计在 Houmo NPU 上怎么落地"（借壳 + 替换后端）

学完 nano-vllm = 理解 vLLM 的全部核心概念
看懂 HLIEvLLM = 知道这些概念在生产中怎么用 + 如何适配自家 NPU
```
