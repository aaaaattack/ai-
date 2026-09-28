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

## 固定章节评估与夯实规则

从第 01 课开始，每完成一个章节，先执行以下步骤，再决定是否学习新内容：

```text
收集作答
→ 判断独立正确率、错误类型与关键概念解释能力
→ 写入本章学习评估
→ 达标：进入下一章
→ 未达标：在原章节 Markdown 继续追加夯实内容与练习
```

- 章节默认门槛：核心题独立正确率约 80%，且能说明关键公式或判断依据。
- 未达标时不引入新主题，优先在原文件补充最小必要的讲解、例题和针对性练习。
- 夯实完成后重新评估；达标后才更新课程路线中的“当前状态”。

## L0：数值与 Tensor 基础

- L0.1 二进制、二进制小数和科学计数法
- L0.2 FP32、FP16、BF16 的符号位、指数位、尾数位和 bias
- L0.3 overflow、underflow、舍入误差与特殊值
- L0.4 Tensor shape、stride、layout 与 contiguous
- L0.5 view、reshape、transpose、permute 与 broadcast
- L0.6 NCHW、NHWC、Attention Tensor shape
- L0.7 FLOPs、TOPS、带宽、延迟、吞吐和利用率

**当前状态：** L0.1～L0.3 已通过；L3 与 L4 作为自学模块保留；L2.1 Q/K/V 与 Attention Tensor shape 已通过，当前进入 L2.2 MHA、MQA、GQA 与 KV Cache。后续保留 bias、特殊值和 subnormal 的间隔复习。

**代码结合学习：** 已建立 `AI_NPU代码结合学习地图.md`。课程按“CS336 笔记恢复概念 → nano-vLLM 追最小实现 → ModelZoo 对照真实部署”的顺序连接；当前 L2.2 阅读 nano-vLLM 中 Qwen3 Attention、KV Cache、block table 与 slot mapping 的真实实现，不修改第三方源码。

### 近期学习安排

1. 第 02 课｜L0.3：overflow、underflow、舍入误差与特殊值；连接 FP16/BF16 的位结构与精度 Debug。
2. 第 03 课｜L3.1：量化的目的、scale、对称 INT8 量化与反量化。
3. 第 04 课｜L3.2：非对称量化、zero-point、clipping 与 rounding。
4. 第 05 课｜L4.1：MAC、Ops、TOPS 与单位换算。

每节完成后先执行章节评估；未达标时只在该节 Markdown 中追加夯实内容，不提前学习后续主题。

### 当前安排调整

L3 的非对称量化、校准、粒度、PTQ/QAT、混合精度与常见方法已补入第 03 课作为自学材料。本阶段不要求逐题考核；完成阅读后，下一段交互式学习进入 L4.1：MAC、Ops、TOPS 与单位换算。

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

**当前状态：** L2.1 Q/K/V 与 Attention Tensor shape 已通过；L2.2 已完成 MHA/MQA/GQA、KV Cache、block table 与 slot mapping 的概念学习，容量与映射计算留作间隔复习。按当前学习安排进入 L2.3 RoPE：理解如何向 Q/K 注入位置与相对位置信息，再连接 nano-vLLM 的 Qwen3 实现。

## L3：量化

- 量化与反量化公式
- scale、zero-point、clipping、rounding、saturation
- 对称/非对称、per-tensor/per-channel/per-group
- PTQ、QAT、校准、outlier、敏感层与 mixed precision
- W8A8、W8A16、W4A16、W4A8
- SmoothQuant、AWQ、GPTQ
 - LoRA、QLoRA、量化基座与参数高效微调

 **补充主题位置：** LoRA 先连接 L2 的 Transformer Linear，再放在 L3 量化之后学习 QLoRA、merge/unmerged、重新量化与部署验收；相关误差进入 L7 精度 Debug。

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
- 专题案例：`AI_NPU系统课程_L5_Compiler_Runtime与LUT职责边界.md`
  - `PyTorch → Q 图 → IR → Lowering → Runtime → NPU` 全链路
  - 量化前端、编译器、Runtime 与硬件的职责边界
  - LUT/FakeQuant/归一化/设备执行误差的分层定位
  - 根据 First Bad Stage 决定下一步实验与对接团队

**当前状态：** 已补充未编号的 LUT 职责边界专题；Compiler/Runtime 主干课程仍待系统学习。

## L6：AI Infra 和推理优化

- Online Serving 与 Offline Inference
- Batch、Continuous Batching、Scheduler
- PagedAttention、Prefix Cache、Chunked Prefill
- TP、PP、DP 与 Speculative Decoding
- TTFT、TPOT、Prefill/Decode 吞吐
**投机解码专题位置：** 放在 L6 的 Prefill/Decode、KV Cache 和 TTFT/TPOT 之后，先讲 draft/verify、接受率和候选树，再连接 L8 的端到端性能拆解。DFlash 2 作为不占正式课号的工程补充案例，覆盖 block-parallel draft、local dynamic convolution、candidate selector 和目标硬件验收。

**当前状态：** 待学习；已理解 TTFT/TPOT 基本概念。

## L7：精度 Debug

- First Bad Stage 与 First Bad Tensor
- cosine、MAE、MSE、RMSE、Max Error
- 统计分布、percentile、saturation ratio
- 逐层 dump、敏感层与 mixed precision
- 根因假设、单变量实验和证据闭环

**当前状态：** 待正式学习；具备初步定位意识。已保留一个不占用课号的《GR00T Visual 精度 Debug》L7 补充实战案例：涵盖四路对比、伪线索排除、ONNX initializer、加载后重初始化与重新 PTQ 验收。完成 L1 模型加载/ONNX 与 L3 量化基础后，按该案例进行正式交互式训练。

**双案例实战安排：** Visual 侧增加 V1～V5 参数与加载取证实验；新增不编号的 [Prefill 案例](AI_NPU系统课程_L7_GR00T_Prefill精度Debug实战.md)，设置 P1～P5 FX/PTQ 消融、校准对照、混精生效检查与部署验收实验。先读 [资源清单](AI_NPU_GR00T双案例_实战方案与资源清单.md)，按离线证据、GPU 复现、设备验收逐步实践。当前 RoPE 正式课进度保持。

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
