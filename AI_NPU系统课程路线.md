# AI/NPU 系统课程路线

## 学习方式

每个主题按以下闭环学习：

```text
直观解释
→ 正式定义与公式
→ 带做例题
→ 课后习题
→ 批改与错因分类
→ 针对性补练
→ 工程场景应用
```

只有当本节核心题正确率达到约 80%，并且能够解释计算过程时，才进入下一节。若低于 50%，会拆小知识点并增加例题；50%～80% 时维持难度并针对错误补练。

## L0：数值与 Tensor 基础

- L0.1 二进制、二进制小数和科学计数法
- L0.2 FP32、FP16、BF16 的符号位、指数位、尾数位和 bias
- L0.3 overflow、underflow、舍入误差与特殊值
- L0.4 Tensor shape、stride、layout 与 contiguous
- L0.5 view、reshape、transpose、permute 与 broadcast
- L0.6 NCHW、NHWC、Attention Tensor shape
- L0.7 FLOPs、TOPS、带宽、延迟、吞吐和利用率

**当前状态：** 正在学习 L0.1～L0.2。

## L1：PyTorch、Transformers 与 ONNX

- Tensor、`nn.Module`、forward、hook
- `state_dict`、`named_modules`、推理模式
- 常见基础算子
- Tokenizer、Processor、Model 与 generate
- ONNX Graph、Node、Tensor、Initializer、opset
- 静态 shape、动态 shape 与模型导出

**当前状态：** 待学习；已有一定概念基础。

## L2：模型结构

- CNN、Residual、Depthwise Conv
- Transformer 与 Q/K/V
- MHA、MQA、GQA、RoPE、RMSNorm、SwiGLU
- Decoder-only LLM、MoE 与长上下文
- ViT、VLM、VLA、Action Token 与 Diffusion Policy

**当前状态：** 待学习；Q/K/V shape 与 GQA 较薄弱。

## L3：量化

- 量化与反量化公式
- scale、zero-point、clipping、rounding、saturation
- 对称/非对称、per-tensor/per-channel/per-group
- PTQ、QAT、校准、outlier、敏感层与 mixed precision
- W8A8、W8A16、W4A16、W4A8
- SmoothQuant、AWQ、GPTQ

**当前状态：** 待学习；属于最高优先级薄弱项。

## L4：GPU、ASIC 与 NPU

- 计算核心、DDR/HBM、DMA、SRAM/Cache
- SIMD、SIMT、Tensor Core、Systolic Array
- MAC、Ops、TOPS 与峰值算力
- Arithmetic Intensity、Compute-bound、Memory-bound
- Roofline Model

**当前状态：** 待学习；属于最高优先级薄弱项。

## L5：Compiler 与 Runtime

- Graph、IR、Pass、Fusion、Lowering、Scheduling、Codegen
- Unsupported OP、Graph Break、CPU Fallback
- Layout Conversion、Memory Planning、Memcpy、同步
- 编译日志和 Runtime 日志分析

**当前状态：** 待学习；已有日志现象识别能力。

## L6：AI Infra 和推理优化

- Online Serving 与 Offline Inference
- Batch、Continuous Batching、Scheduler
- PagedAttention、Prefix Cache、Chunked Prefill
- TP、PP、DP 与 Speculative Decoding
- TTFT、TPOT、Prefill/Decode 吞吐

**当前状态：** 待学习；已理解 TTFT/TPOT 基本概念。

## L7：精度 Debug

- First Bad Stage 与 First Bad Tensor
- cosine、MAE、MSE、RMSE、Max Error
- 统计分布、percentile、saturation ratio
- 逐层 dump、敏感层与 mixed precision
- 根因假设、单变量实验和证据闭环

**当前状态：** 待学习；具备初步定位意识。

## L8：性能 Debug

- E2E Breakdown
- Graph、Subgraph、Operator、Kernel、Hardware 分层
- Operator Profiling、Roofline 和瓶颈判断
- DDR、DMA、SRAM、fallback、layout、同步和 launch overhead
- LLM Prefill、Decode、TTFT、TPOT、KV Cache

**当前状态：** 待学习；会做基础延迟拆分。

## L9：综合解决方案

- 部署可行性判断
- 性能预估与验收指标
- 量化方案和精度风险
- 客户信息采集与最小复现
- 问题分层、研发转交和证据闭环
- VLM、VLA、LLM 等真实客户场景

**当前状态：** 最后阶段综合训练。

## 当前学习顺序

```text
L0 数值基础
→ L3 量化基础
→ L4 硬件与 Roofline
→ L2 模型结构
→ L1 模型与 ONNX
→ L5 Compiler/Runtime
→ L6 AI Infra
→ L7 精度 Debug
→ L8 性能 Debug
→ L9 综合解决方案
```

主题编号保持 L0～L9 不变，但教学顺序根据摸底结果调整，使量化和性能计算尽早补齐。
