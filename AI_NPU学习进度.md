# AI/NPU 学习进度

## 第 01 课｜L0 二进制与 FP16：章节评估

**最终结论：** 已通过夯实评估，可进入 L0.3；保留 bias、FP16 编码/解码的间隔复习。

- 已具备：二进制位权与小数展开、规格化、FP16 bias 正反算、基础 FP16 编码/解码，以及 FP16/BF16 的范围和精度取舍。
- 仍需巩固：将 bias 的编码值与真实指数清晰区分，并把位结构连接到 overflow、underflow 与实际精度 Debug。
- 第 02 课｜L0.3：已通过，进入 L3.1；保留特殊值、subnormal 和舍入误差的间隔复习。
- 已具备：FP16 overflow 范围判断、underflow 与舍入的基本区分、`inf`/`NaN` 的特殊编码、First Bad Tensor 定位和单变量实验原则。
- 仍需巩固：subnormal 与特殊指数编码目前主要在讲解后能够复述，需要在后续课程中重新提取。
- 间隔复习：第 03 课中再次将“指数全 1、尾数非 0”答为 `inf`；该规则需在后续课前继续短测。
- 当前学习：第 03 课｜L3 量化，自学模式。已补齐对称/非对称量化、zero-point、粒度、校准、PTQ/QAT、异常值、混合精度和常见方法的参考内容。
- 考核安排：本阶段按自学处理，不要求完成所有手算题；遇到实际量化任务时再按“First Bad Tensor → 校准/粒度 → 单层混合精度”路径深入。
- L4 自学完成：已阅读峰值算力、阵列利用率、shape/batch、频率、稀疏峰值、带宽、Arithmetic Intensity、Roofline、融合和 layout conversion。
- 当前交互式学习：第 06 课｜L2.1 Transformer、Q/K/V 与 Attention Tensor shape。
- 本节目标：能从 `batch_size、sequence_length、hidden_size、num_heads、head_dim` 推导 Q/K/V、attention score 和输出 Tensor shape，为后续 MHA/MQA/GQA 与 KV Cache 做准备。
- 代码学习支架：已建立 `AI_NPU代码结合学习地图.md`。本课完成原有 shape 练习后，追加一张 nano-vLLM Qwen3 Attention 的代码观察卡；暂不运行模型或 ModelZoo 性能工具。
- 第 06 课首次评估：暂不通过。核心题独立完成 2/4；第 1、4 题正确，第 2 题漏掉拆分/转置步骤，第 3 题混淆输入 hidden shape 与 score shape。
- 第 06 课第一次复测：1/4，仅能识别 Decode 使用 `flash_attn_with_kvcache`；`head_dim=hidden_size/num_heads` 计算、`[batch_size, num_heads, sequence_length, head_dim]` 和 `[batch_size, num_heads, sequence_length, sequence_length]` 的维度顺序仍未掌握。
- 当前处理：继续在第 06 课原文夯实，不进入下一节。已追加只含两个 shape 模板的第二轮复测，三题全部正确后再进入 MHA、MQA、GQA 与 KV Cache。
- 第 06 课第二轮复测：`head_dim=32`、Q/K/V `[3,4,10,32]`、score `[3,4,10,10]` 均正确；但未回答 score 最后两个维度的语义，当前为 2/3 完整正确。只需补答 query 位置与 key 位置的含义，不重做其他题。

## 综合摸底试卷第 1 套

**评估结果：** 39/100  
**总体判断：** 工程直觉好于底层计算基础。能够识别部分部署现象并进行基础延迟计算，但在浮点编码、Transformer Tensor shape、量化、峰值算力和 Roofline 上存在明显知识断层，尚不能独立完成完整的精度与性能证据闭环。

| 主题 | 得分 | 当前状态 | 后续训练重点 |
|---|---:|---|---|
| L0 数值与 Tensor | 7/10 | 部分掌握 | 二进制科学计数法、FP16 bias 与编码 |
| L1 PyTorch/ONNX | 8/10 | 基本掌握 | API 实践、hook、动态 shape 与 shape bucket |
| L2 模型结构 | 2/10 | 薄弱 | Q/K/V shape、MHA/MQA/GQA、KV Cache |
| L3 量化 | 0/10 | 尚未掌握 | scale、zero-point、量化/反量化、per-channel、outlier |
| L4 GPU/ASIC/NPU | 0/10 | 尚未掌握 | MAC/Ops/TOPS、带宽、AI、Roofline |
| L5 Compiler/Runtime | 5/10 | 部分掌握 | graph break 成本、fallback 计时、对照实验 |
| L6 AI Infra | 5.5/10 | 部分掌握 | PagedAttention、Prefix Cache、KV Cache 优化 |
| L7 精度 Debug | 3.5/10 | 初步理解 | Stage/Tensor 区分、统计指标、单变量实验 |
| L8 性能 Debug | 3.5/10 | 初步理解 | 利用率解释、bound 判断、最小实验 |
| L9 综合方案 | 4.5/10 | 初步理解 | 真实轨迹评估、验收条件、证据闭环 |

## 已表现较好的能力

- 能计算 Tensor 元素数、存储量和 E2E 延迟比例；
- 理解 `eval()`、`no_grad()`、动态轴和 forward hook 的基本用途；
- 能从日志中识别 fusion、unsupported op、layout conversion 和 CPU fallback；
- 能正确理解 TTFT、TPOT 的基本含义；
- 能发现 NPU FP16 已先于 INT8 出现精度异常；
- 能计算 VLA FP16/W8A8 的端到端延迟及理论频率。

## 当前优先补强顺序

1. L0：二进制科学计数法、FP16 指数与 bias；
2. L3：INT8 对称量化的完整手算；
3. L4：峰值算力、单位换算、Arithmetic Intensity 与 Roofline；
4. L2：Transformer Q/K/V shape 与 KV Cache；
5. L5/L7/L8：用单变量实验建立精度和性能证据闭环。

## 下一轮难度

降低一个台阶，以中等偏基础计算为主，但保留 3 道简单工程应用题。建议仍为 10 题：

- 2 题浮点数与二进制；
- 3 题量化计算；
- 2 题峰值算力与 Roofline；
- 1 题 Transformer shape；
- 1 题精度 Debug；
- 1 题性能 Debug。

当 L0、L3、L4 的计算题正确率达到 80% 后，再恢复综合客户场景训练。
