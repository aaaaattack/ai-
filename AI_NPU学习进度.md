# AI/NPU 学习进度

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
