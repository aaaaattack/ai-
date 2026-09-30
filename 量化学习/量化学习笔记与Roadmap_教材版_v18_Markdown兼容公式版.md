# 量化学习笔记与 Roadmap（教材版）

> **公式预览说明**：本版本使用 Markdown 常见的 `$$...$$`（独立公式）和 `$...$`（行内公式）语法。若仍显示原始符号，请使用支持 MathJax/KaTeX 的预览器，例如 VS Code 的 Markdown Preview Enhanced、Typora、Obsidian、Quarto 或 MkDocs Material；普通纯文本编辑器不会渲染数学公式。


> 目标：建立从 **数值表示 → PTQ/QAT → W8A8/W4A16 → Transformer 量化 → SmoothQuant/GPTQ/AWQ/QuaRot/DuQuant → 编译器/NPU → VLA 量化** 的系统性理解。
>
> 本文不是论文列表，而是一份面向模型部署 / NPU 应用 / 量化问题定位的长期学习笔记。

> **阅读说明（v14 修订）**：本文同时包含通用量化原理、工程经验和当前 Spirit/XH2 工程映射。凡是没有明确标注工具版本、硬件型号或实测结果的内容，不应视为 `xhquant` 的能力承诺。公式和代码中有一部分是教学简化版，实际 runtime 可能采用不同的 scale、rounding、layout 或 kernel 实现。

### 证据等级

本文使用以下约定：

- **原理**：适用于一般量化系统的数学或架构事实；
- **工程经验**：常见实现方式，但依赖框架、编译器和硬件；
- **当前工程映射**：已在本项目源码中看到的调用关系；
- **待实测**：需要在具体 `xhquant` / XH2a 版本上通过 ONNX、HMONNX、HMM 和 NPU 运行确认。

### 当前工程边界

本文的 XH2/VLA 相关判断以当前 Spirit-v1.5 工程为上下文。不要把 Spirit 专用脚本中的支持情况，直接推广为所有 VLA 或所有模型的支持情况；新模型必须经过“导出 → 算子检查 → HMONNX 转换 → HMM 编译 → 精度/性能验证”闭环。

---

## 目录

1. [总览：量化到底在做什么](#1-总览量化到底在做什么)
2. [数值表示基础](#2-数值表示基础)
3. [线性量化数学基础](#3-线性量化数学基础)
4. [一层 Linear 的 W8A8 完整执行链](#4-一层-linear-的-w8a8-完整执行链)
5. [两层量化 Linear 如何连接：Q/DQ、Requant、Scale Propagation](#5-两层量化-linear-如何连接qdqrequantscale-propagation)
6. [PTQ / Calibration / Observer / Hook](#6-ptq--calibration--observer--hook)
7. [Transformer Block 为什么难量化](#7-transformer-block-为什么难量化)
8. [Activation Outlier](#8-activation-outlier)
9. [SmoothQuant](#9-smoothquant)
10. [SmoothQuant 的 Calibration 与 PyTorch 实现](#10-smoothquant-的-calibration-与-pytorch-实现)
11. [SmoothQuant 接入真正 W8A8](#11-smoothquant-接入真正-w8a8)
12. [GPTQ：二阶信息与误差补偿](#12-gptq二阶信息与误差补偿)
13. [GPTQ 手推：Sequential Quantization](#13-gptq-手推sequential-quantization)
14. [GPTQ Group-wise INT4 与 W4A16 Runtime](#14-gptq-group-wise-int4-与-w4a16-runtime)
15. [AWQ：Activation-aware Weight Quantization](#15-awqactivation-aware-weight-quantization)
16. [QuaRot：Rotation Quantization](#16-quarotrotation-quantization)
17. [DuQuant：Rotation + Permutation](#17-duquantrotation--permutation)
18. [算法横向总结](#18-算法横向总结)
19. [编译器量化与 NPU 落地](#19-编译器量化与-npu-落地)
20. [VLA 量化：QuantVLA / QVLA 的位置](#20-vla-量化quantvla--qvla-的位置)
21. [学习 Roadmap](#21-学习-roadmap)
22. [阶段性自测](#22-阶段性自测)
23. [Sensitivity Analysis + Mixed Precision](#23-sensitivity-analysis--mixed-precision)
24. [Quantization Error Debugging](#24-quantization-error-debugging真实量化问题定位)
25. [Quantization Debug Harness](#25-quantization-debug-harness把排障方法代码化)
26. [Calibration 与 Observer](#26-calibration-策略与-observer-设计)
27. [Quantization Granularity](#27-quantization-granularity-系统比较)
28. [Q/DQ Graph 与 Quant IR](#28-qdq-graph-与-quantization-compiler-ir)
29. [INT8 GEMM 与 NPU 执行](#29-int8-gemm-在-npu-上-的真实执行)
30. [Roofline](#30-rooflinememory-bound-vs-compute-bound-的定量分析)
31. [NPU Performance Profiling](#31-npu-performance-profiling-实战框架)
32. [完整量化部署流水线](#32-从-pytorch-到-npu-binary-完整量化部署流水线)
33. [MLP W8A8 Case Study](#33-case-study-1两层-mlp-的完整-w8a8-量化)
34. [Quantization Debug Toolkit](#34-case-study-2自动化-quantization-debug-toolkit)
35. [Transformer Block Case Study](#35-case-study-3transformer-block-quantization)
36. [Attention Quantization](#36-attention-quantizationqk-双边误差softmax-敏感性与-temperature-drift)
37. [Residual / LayerNorm / Scale Alignment](#37-residuallayernormscale-alignment-专题)
38. [当前 xhquant / Spirit 工程映射](#38-当前-xhquant--spirit-工程映射与验证闭环)
39. [NPU 解决方案工程师进阶学习清单](#39-npu-解决方案工程师进阶学习清单)

---

# 1. 总览：量化到底在做什么

量化不是简单地把：

```text
FP16 → INT8
```

而是：

```text
浮点模型
  ↓
确定量化方案
  ↓
Calibration / 统计运行时数据
  ↓
确定 scale / zero-point / clipping
  ↓
量化 weight / activation
  ↓
图改写 / QDQ / Requant / Mixed Precision
  ↓
编译到 INT8 / INT4 Kernel
  ↓
验证精度 + 性能
```

从工程角度，量化最核心的矛盾是：

> **有限 bit 数量化格点 vs. Tensor 的动态范围。**

如果一个 Tensor 中：

```text
[0.1, 0.2, 0.3, 20.0]
```

少数 outlier 把动态范围撑大，那么大多数普通值只能落在少量整数格点中，量化误差就会明显上升。

---

# 2. 数值表示基础

## 2.1 FP 与 INT 的根本区别

FP16 可以写成：

$$
(-1)^s \times 2^e \times (1+m)
$$

指数位让浮点数具有较大的动态范围。

> 这是教学级简化表达。完整 IEEE 754 FP16 还包含 subnormal、Inf、NaN 和 bias exponent 等特殊情况；本节重点只是说明指数位带来动态范围。

INT8 则只有固定整数范围：

$$
[-128,127]
$$

因此，如果要用 INT8 表示浮点数，就必须额外提供一个“尺度”：

$$
x \approx q \cdot s
$$

其中：

- $x$：原始浮点值
- $q$：INT8 整数值
- $s$：scale

---

# 3. 线性量化数学基础

## 3.1 对称量化

最简单的 symmetric INT8：

$$
s=\frac{\max |x|}{127}
$$

量化：

$$
q=\operatorname{round}(x/s)
$$

再做饱和：

$$
q=\operatorname{clip}(q,-128,127)
$$

反量化：

$$
\hat{x}=q\cdot s
$$

### 手算例子

假设：

$$
x=[-1.0,\ 0.5,\ 2.0]
$$

则：

$$
s=\frac{2}{127}\approx0.01575
$$

量化：

$$
[-1.0,0.5,2.0]
\rightarrow
[-64,32,127]
$$

反量化：

$$
[-64,32,127]\times0.01575
\approx
[-1.008,0.504,2.0]
$$

误差来源主要有：

- rounding error
- clipping error

## 3.2 非对称量化

一般形式：

$$
q=\operatorname{round}(x/s)+z
$$

$$
\hat{x}=s(q-z)
$$

其中 $z$ 是 zero-point。

---

## 3.3 Per-Tensor / Per-Channel / Per-Group

### Per-Tensor

整个 tensor 共用一个 scale：

```text
Tensor
└── scale = 0.02
```

优点：简单、硬件友好。

缺点：容易被 outlier 撑大。

### Per-Channel

每个 channel 单独 scale：

```text
channel 0 → s0
channel 1 → s1
channel 2 → s2
```

精度更高，但需要额外 scale metadata。

### Per-Group

每 N 个 weight 共用一个 scale：

```text
group_size = 128
```

常见于 INT4 LLM quantization。

### 3.4 量化误差的来源与 scale 选择

**系统讲解**：对称线性量化可以看成把连续区间映射到有限整数格点。设量化范围为 $[-Q,Q]$，scale 为 (s)，则每个相邻整数格点在实数域的间隔为 (s)。scale 越大，能够覆盖的动态范围越大，但量化分辨率越低；scale 越小，分辨率越高，但更容易发生 clipping。

**公式**：对称量化的误差可以分成 rounding error 与 clipping error：

$$
e(x)=x-s\cdot \operatorname{clip}\left(\operatorname{round}(x/s),-Q,Q\right)
$$

若 (|x|\le Qs)，误差主要来自 rounding，理论上满足：

$$
|e(x)|\le \frac{s}{2}
$$

若 (|x|>Qs)，误差还包括饱和造成的 clipping error。因此 observer 的本质是选择一个合适的 (s)，在覆盖范围和分辨率之间折中。

**代码实现**：对应通用 quantizer 的最小实现如下；实际 `xhquant` 还会结合 axis、dtype、目标设备和算子配置处理 scale。

```python
def symmetric_quantize(x, qmax=127, clip=None):
    # calibration 可选择 percentile/MSE clipping，而不是永远使用 max
    bound = x.abs().max() if clip is None else clip
    scale = bound.clamp_min(1e-12) / qmax
    q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale
```

---

# 4. 一层 Linear 的 W8A8 完整执行链

普通 Linear：

$$
Y=XW^T+b
$$

假设做 W8A8：

$$
X\approx s_xX_q
$$

$$
W\approx s_wW_q
$$

代入：

$$
XW^T
\approx
(s_xX_q)(s_wW_q)^T
$$

所以：

$$
XW^T
\approx
s_xs_w(X_qW_q^T)
$$

因此：

$$
s_{acc}=s_xs_w
$$

---

## 4.1 为什么 accumulator 是 INT32

INT8 最大值：

$$
127
$$

单次乘法：

$$
127\times127=16129
$$

如果 `in_features = 4096`：

$$
16129\times4096
$$

远超 INT8 范围，因此通常：

```text
INT8 × INT8
    ↓
INT32 accumulate
```

---

## 4.2 Bias 为什么常见 INT32

accumulator 的真实 scale：

$$
s_{acc}=s_xs_w
$$

为了直接在 INT32 accumulator 中加 bias：

$$
b_q=\operatorname{round}
\left(
\frac{b}{s_xs_w}
\right)
$$

于是：

```text
INT32 accumulator
+
INT32 bias
```

> 适用条件：bias 直接加入 INT32 accumulator 的 integer-only 路径。部分 runtime 会保留 FP32 bias，或在 dequant/fusion 阶段处理 bias，因此“bias 常见 INT32”不是所有硬件实现的强制规则。

---

## 4.3 PyTorch 最小实现

```python
import torch

def quantize_int8(x):
    scale = x.abs().max() / 127.0
    q = torch.round(x / scale)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8), scale


X = torch.tensor([[1.0, -0.5]])

W = torch.tensor([
    [0.8, -0.2],
    [0.1,  0.6]
])

b = torch.tensor([0.1, -0.2])


# FP baseline
y_fp = X @ W.T + b


# quantize
X_q, s_x = quantize_int8(X)
W_q, s_w = quantize_int8(W)


# INT8 × INT8 -> INT32
acc = (
    X_q.to(torch.int32)
    @ W_q.to(torch.int32).T
)


# bias quantization
b_q = torch.round(
    b / (s_x * s_w)
).to(torch.int32)

acc = acc + b_q


# dequant
y_q = acc.float() * (s_x * s_w)

print("FP output:", y_fp)
print("Quantized output:", y_q)
print("Error:", y_q - y_fp)
```

**系统讲解**：W8A8 的关键不是把两个输入都转换成 INT8 就结束，而是维护三套尺度：activation scale (s_x)、weight scale (s_w) 和输出 scale (s_y)。整数矩阵乘法产生的 accumulator 表示的是实数 (s_xs_wA_qB_q)，因此输出若要继续作为 INT8 输入，必须执行 requant：

**公式**：

$$
Y_q=\operatorname{round}\left(\frac{s_xs_w}{s_y}A_qB_q\right)
$$

这里的 (A_qB_q) 通常用 INT32 累加。bias 若在 accumulator 域相加，则使用 (b_q=\operatorname{round}(b/(s_xs_w)))；若 runtime 在 dequant 后相加，则 bias 可以保留 FP32。这是“数学等价关系”，不是对所有硬件 kernel 的固定实现要求。

**代码实现**：

```python
def w8a8_linear(x_q, w_q, sx, sw, sy, bias=None):
    # INT8 × INT8，必须扩大到 INT32 累加
    acc = x_q.to(torch.int32) @ w_q.to(torch.int32).T
    if bias is not None:
        bq = (bias / (sx * sw)).round().to(torch.int32)
        acc = acc + bq
    # accumulator scale -> output activation scale
    yq = (acc.float() * (sx * sw) / sy).round()
    return yq.clamp(-128, 127).to(torch.int8)
```

---

# 5. 两层量化 Linear 如何连接：Q/DQ、Requant、Scale Propagation

假设：

```text
X
 ↓
Linear1
 ↓
Y1
 ↓
Linear2
 ↓
Y2
```

第一层 accumulator 的 scale：

$$
s_{acc1}=s_xs_{w1}
$$

第二层输入可能要求自己的 activation scale：

$$
s_{y1}
$$

因此：

$$
Y_{1q}
=
\operatorname{round}
\left(
acc_1
\frac{s_xs_{w1}}{s_{y1}}
\right)
$$

这一步就是：

> **Requantization**

---

## 5.1 三个概念不要混

### Quantize

```text
FP → INT
```

### Dequantize

```text
INT → FP
```

### Requantize

```text
INT(scale A)
   ↓
INT(scale B)
```

公式：

$$
q_B=
\operatorname{round}
\left(
q_A\frac{s_A}{s_B}
\right)
$$

---

## 5.2 Requant 代码

```python
def requantize_int32_to_int8(
    acc,
    acc_scale,
    out_scale
):
    q = torch.round(
        acc.float()
        * acc_scale
        / out_scale
    )
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)
```

**系统讲解**：两层量化 Linear 连接时，第一层输出的整数值不能直接作为第二层输入。第一层输出的真实 scale 是 (s_{y1})，第二层需要的输入 scale 可能是 (s_{x2})。如果二者不同，必须在整数域执行一次 scale conversion；这就是 requant，而不是重新做一次浮点量化。

**公式**：

$$
q_{x2}=\operatorname{round}\left(q_{y1}\frac{s_{y1}}{s_{x2}}\right)
$$

工程实现通常把比例 (s_{y1}/s_{x2}) 近似成定点 multiplier 和 shift，以便在 NPU 上执行：

```text
INT32 accumulator
  → fixed-point multiplier
  → right shift / rounding
  → saturation
  → INT8 output
```

**代码实现**：

```python
def requantize(q, src_scale, dst_scale):
    # q 可以是 INT32 accumulator，也可以是中间 INT8
    ratio = src_scale / dst_scale
    return (q.float() * ratio).round().clamp(-128, 127).to(torch.int8)
```

当前项目中的 `clone_inputs_as()`、`prepare_hmonnx_llm_args()` 主要负责 dtype 和输入容器对齐；真正的算子级 requant 由 `xhquant` 图转换和后端 kernel 负责，不能把这两个 Python 辅助函数误认为是完整量化实现。

---

## 5.3 为什么编译器想消掉 Q/DQ

低效：

```text
INT8 Linear1
 ↓
DQ
 ↓
FP16
 ↓
Q
 ↓
INT8 Linear2
```

理想：

```text
INT8 Linear1
 ↓ requant
INT8 Linear2
```

这样减少：

- 数据搬运
- Q/DQ kernel
- bandwidth
- latency

---

# 6. PTQ / Calibration / Observer / Hook

## 6.1 Calibration 是什么

Calibration 的核心不是拿最终预测结果，而是：

> **让真实数据流过模型，采集运行时 Tensor 分布。**

例如 VLA：

```text
image + instruction + state
        ↓
Vision Encoder
        ↓     ← collect activation
VLM
        ↓     ← collect activation
Action Head
        ↓     ← collect activation
action
```

通常不会真的驱动机器人。

---

## 6.2 Calibration 和真实推理的关系

内部 forward 路径应尽量一致：

- preprocessing 一致
- tensor shape 一致
- attention / MLP 路径一致
- diffusion timestep 尽量覆盖真实范围

但 calibration 会额外插入：

- observer
- hook
- histogram/stat collector

---

## 6.3 Hook 示例

```python
def hook_fn(module, inputs, output):
    x = inputs[0]
    print(x.shape)

handle = model.fc.register_forward_hook(
    hook_fn
)

model(x)

handle.remove()
```

**系统讲解**：Calibration 的目标是估计部署时真实激活分布，而不是验证最终任务准确率。一个 observer 至少需要明确四件事：统计哪个 tensor、沿哪个 axis 聚合、覆盖哪些样本、最终把统计量转换成哪种量化参数。对 VLA 来说，图片、文本、机器人 state、历史帧和不同 action/timestep 都可能改变分布。

**公式**：以 per-channel symmetric observer 为例，若收集到 (N) 个样本的第 (c) 个 channel 最大绝对值，则：

$$
m_c=\max_{n=1}^{N}|x_{n,c}|,
\qquad
s_c=\frac{m_c}{Q_{max}}
$$

如果使用 percentile clipping，则 (m_c) 不再是最大值，而是经验分位点，从而减少单个 outlier 对 scale 的影响。

**代码实现**：

```python
class ChannelMaxObserver:
    def __init__(self):
        self.max_abs = None

    def update(self, x):
        x = x.detach().float().reshape(-1, x.shape[-1])
        cur = x.abs().amax(dim=0)
        self.max_abs = cur if self.max_abs is None else torch.maximum(self.max_abs, cur)

    def scale(self, qmax=127):
        return self.max_abs.clamp_min(1e-12) / qmax
```

当前工程对应关系：`build_processor_inputs()` 构造视觉/文本输入，`build_converter_dummies()` 组织 prefill、decode、vision、DiT 的输入；它们是 converter dummy，不等价于覆盖真实工作负载的完整 calibration dataset。

---

# 7. Transformer Block 为什么难量化

典型结构：

```text
Input
  │
LayerNorm
  ↓
Q/K/V Linear
  ↓
QKᵀ / √d
  ↓
Softmax
  ↓
Attention × V
  ↓
O Projection
  ↓
Residual Add
  ↓
LayerNorm
  ↓
FC1
  ↓
GELU / SiLU
  ↓
FC2
  ↓
Residual Add
```

真正难的不是 Linear，而是：

1. nonlinear
2. residual
3. activation distribution

---

## 7.1 Attention logits 为什么敏感

$$
S=\frac{QK^T}{\sqrt d}
$$

$$
A=softmax(S)
$$

如果：

$$
Q'=Q+\Delta Q,\qquad
K'=K+\Delta K
$$

则：

$$
Q'K'^T
=
QK^T
+
\Delta QK^T
+
Q\Delta K^T
+
\Delta Q\Delta K^T
$$

误差会进入 Softmax。

Softmax 对 logits 差异敏感，因此很小的量化误差可能导致 attention distribution 明显变化。

---

## 7.2 Residual 为什么要 Scale Alignment

假设：

```text
branch A:
INT8 @ scale 0.02

branch B:
INT8 @ scale 0.05
```

相同整数 10：

$$
10\times0.02=0.2
$$

$$
10\times0.05=0.5
$$

所以不能直接整数相加。

必须先对齐 scale。

**系统讲解**：Residual 是量化 Transformer 中的结构性难点。shortcut 分支保留 (X)，transform 分支输出 (F(X))。即使两条分支各自的 cosine 都很高，只要其中一条分支的能量被 scale 或 saturation 系统性压低，Add 后的 hidden state 分布仍可能漂移，并在后续层累积。

**公式**：若

$$
X=s_xX_q,\qquad F=s_fF_q,\qquad Y=s_yY_q
$$

则正确的整数加法为：

$$
Y_q=\operatorname{round}\left(\frac{s_x}{s_y}X_q+\frac{s_f}{s_y}F_q\right)
$$

这说明 Residual Add 的核心不是“两个 int8 相加”，而是先把两个 branch 映射到同一 output scale。

**代码实现**：

```python
def quantized_add(xq, fq, sx, sf, sy):
    # 两个 branch 先对齐到 output scale，再相加和饱和
    y = (xq.float() * sx / sy + fq.float() * sf / sy).round()
    return y.clamp(-128, 127).to(torch.int8)
```

---

# 8. Activation Outlier

典型：

```text
[0.2, 0.3, 0.5, 17.5]
```

per-tensor scale：

$$
s=\frac{17.5}{127}
$$

此时：

```text
0.2
0.3
0.5
```

只能占很少几个整数格点。

核心：

> 少数 outlier 把量化动态范围撑大，导致多数普通值量化分辨率下降。

**系统讲解**：outlier 不应只用肉眼看最大值判断。工程上要同时记录最大值、P99/P99.9、RMS、饱和比例和量化后的 MSE。若最大值很大但 P99 正常，通常说明少量异常点主导了 per-tensor scale；此时 percentile/MSE clipping、per-channel 或 SmoothQuant 可能比盲目提高 bit 更有效。

**公式**：对称 INT8 的饱和比例可以定义为：

$$
r_{sat}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{|x_i|>127s\}
$$

**代码实现**：

```python
def activation_stats(x, scale):
    flat = x.detach().float().reshape(-1)
    return {
        "max_abs": flat.abs().max().item(),
        "rms": flat.square().mean().sqrt().item(),
        "p99": torch.quantile(flat.abs(), 0.99).item(),
        "saturation_ratio": (flat.abs() > 127 * scale).float().mean().item(),
    }
```

---

# 9. SmoothQuant

SmoothQuant 核心公式：

$$
XW=(XS^{-1})(SW)
$$

它不改变 FP 模型函数，只改变数值分布。

---

## 9.1 核心思想

```text
Activation 难量
Weight 相对好量
      ↓
把一部分 Activation 动态范围
迁移到 Weight
```

如果某 activation channel：

```text
15
```

取：

$$
S=5
$$

则：

$$
15/5=3
$$

同时对应 weight：

$$
W\rightarrow5W
$$

最终乘积不变。

---

## 9.2 Smooth Scale

常见形式：

$$
s_j=
\frac{
\max(|X_j|)^\alpha
}{
\max(|W_j|)^{1-\alpha}
}
$$

$\alpha$ 控制：

```text
Activation 量化难度
↕
Weight 量化难度
```

> 工程注意：scale 的方向、channel axis、LayerNorm/Linear 的相对位置必须与具体实现一致。上式是常见 SmoothQuant 形式，不应脱离模型图直接套用。

---

# 10. SmoothQuant 的 Calibration 与 PyTorch 实现

## 10.1 收集 Linear 输入 channel max

```python
class ActivationCollector:
    def __init__(self):
        self.max_abs = None

    def __call__(self, module, inputs, output):
        x = inputs[0].detach()

        # [B, ..., C] -> [N, C]
        x = x.reshape(-1, x.shape[-1])

        current_max = (
            x.abs()
            .max(dim=0)
            .values
        )

        if self.max_abs is None:
            self.max_abs = current_max
        else:
            self.max_abs = torch.maximum(
                self.max_abs,
                current_max
            )
```

注册：

```python
collector = ActivationCollector()

handle = model.fc.register_forward_hook(
    collector
)

with torch.no_grad():
    for x in calibration_data:
        model(x)

handle.remove()
```

---

## 10.2 Weight channel max

PyTorch Linear：

```text
weight shape:
[out_features, in_features]
```

对应 input channel 的 weight max：

```python
weight_max = (
    model.fc.weight
    .detach()
    .abs()
    .max(dim=0)
    .values
)
```

---

## 10.3 计算 Smooth Scale

```python
alpha = 0.5
eps = 1e-5

smooth_scale = (
    act_max.clamp(min=eps).pow(alpha)
    /
    weight_max.clamp(min=eps).pow(
        1 - alpha
    )
)
```

---

## 10.4 Scale Folding 到 LayerNorm

LayerNorm：

$$
Y=\hat X\gamma+\beta
$$

希望：

$$
Y'=\frac{Y}{S}
$$

因此：

$$
\gamma'=\gamma/S
$$

$$
\beta'=\beta/S
$$

Linear：

$$
W'=WS
$$

> 该 folding 需要满足相应的 LayerNorm/Linear 拓扑和 broadcasting 约定。若一个输出同时 fan-out 到 Q/K/V 或多个消费者，必须保证所有消费者同步变换，否则会破坏数学等价。

代码：

```python
def smooth_ln_linear(
    ln,
    linear,
    scales
):
    ln.weight.data.div_(scales)

    if ln.bias is not None:
        ln.bias.data.div_(scales)

    linear.weight.data.mul_(
        scales.unsqueeze(0)
    )
```

---

# 11. SmoothQuant 接入真正 W8A8

## 11.1 直接 W8A8

```python
x_q, sx = quantize_int8_per_tensor(x)
w_q, sw = quantize_int8_per_tensor(w)

acc = (
    x_q.to(torch.int32)
    @
    w_q.to(torch.int32).T
)

y_int8 = acc.float() * sx * sw
```

---

## 11.2 Smooth 后再 W8A8

```python
x_smooth = x / smooth_scale

w_smooth = (
    w * smooth_scale.unsqueeze(0)
)

x_s_q, sx_s = quantize_int8_per_tensor(
    x_smooth
)

w_s_q, sw_s = quantize_int8_per_tensor(
    w_smooth
)

acc_s = (
    x_s_q.to(torch.int32)
    @
    w_s_q.to(torch.int32).T
)

y_smooth_int8 = (
    acc_s.float()
    * sx_s
    * sw_s
)
```

然后比较：

$$
MSE(Y_{fp},Y_{int8})
$$

和：

$$
MSE(Y_{fp},Y_{smooth,int8})
$$

SmoothQuant 的目标是让后者更小。

**系统讲解**：SmoothQuant 的“等价变换”成立于浮点数学层面，但部署时还要考虑量化 granularity、clipping、LayerNorm 位置、矩阵布局和 compiler folding。判断它是否有效，不能只比较 weight 的范围，必须用代表性 calibration 数据同时比较 FP 输出、普通 W8A8 输出和 Smooth 后 W8A8 输出。

**代码实现**：

```python
def evaluate_quant_error(y_ref, y_quant):
    err = (y_ref.float() - y_quant.float()).reshape(-1)
    return {
        "mse": err.square().mean().item(),
        "max_abs": err.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            y_ref.float().reshape(1, -1),
            y_quant.float().reshape(1, -1),
        ).item(),
    }
```

当前 `ptq_dump.py` 中的 `cosine_similarity()`、`mse()`、`max_abs_diff()` 正是这一类输出对比的工程化版本；它们还额外处理了 numpy/tensor 转换、shape 对齐和 missing output。

---

# 12. GPTQ：二阶信息与误差补偿

GPTQ 最经典场景：

```text
W4A16
```

目标不是单纯最小化：

$$
\|W-\hat W\|^2
$$

而是：

$$
\|XW-X\hat W\|^2
$$

即：

$$
\|X(W-\hat W)\|^2
$$

因此，真实 activation $X$ 会决定：

> 哪些 weight error 更重要。

---

## 12.1 Hessian 直觉

展开：

$$
L
=
\Delta W^T
X^TX
\Delta W
$$

所以：

$$
H\approx X^TX
$$

> 这里是 GPTQ 的二阶近似直觉。真实实现通常还包含阻尼、分块、Cholesky/逆 Hessian 处理、group-wise quantization 和特定的权重布局；本节手推不能替代真实 GPTQ 实现。

其中：

- 对角线：channel 自己的能量
- 非对角线：channel 之间相关性

---

# 13. GPTQ 手推：Sequential Quantization

假设：

$$
w_1=0.53,\qquad
w_2=0.47
$$

calibration 数据：

```text
x1 = [1,2,3]
x2 = [1,1,1]
```

量化：

$$
w_1:0.53\rightarrow0.50
$$

误差：

$$
\Delta w_1=-0.03
$$

希望调整 $w_2$ 使：

$$
\|x_1\Delta w_1+x_2\Delta w_2\|^2
$$

最小。

可得：

$$
\Delta w_2
=
-
\frac{x_2^Tx_1}
{x_2^Tx_2}
\Delta w_1
$$

这里：

$$
x_2^Tx_1=6
$$

$$
x_2^Tx_2=3
$$

所以：

$$
\Delta w_2
=
-2(-0.03)
=
0.06
$$

即：

$$
w_2:0.47\rightarrow0.53
$$

---

## 13.1 Python 验证

```python
import torch

X = torch.tensor([
    [1., 1.],
    [2., 1.],
    [3., 1.]
])

w = torch.tensor([
    0.53,
    0.47
])

y_fp = X @ w

q1 = torch.tensor(0.50)
delta_w1 = q1 - w[0]

x1 = X[:, 0]
x2 = X[:, 1]

delta_w2 = -(
    torch.dot(x2, x1)
    /
    torch.dot(x2, x2)
) * delta_w1

w_comp = torch.tensor([
    q1,
    w[1] + delta_w2
])

y_comp = X @ w_comp

print("baseline:", y_fp)
print("compensated:", y_comp)
```

---

## 13.2 Sequential Quantization

真实 GPTQ：

```text
量化 w1
↓
产生 error
↓
利用 H / H^-1
调整 w2,w3,...

量化 w2
↓
继续补偿 w3,w4,...

...
```

这就是：

> Sequential Quantization + Error Compensation

---

# 14. GPTQ Group-wise INT4 与 W4A16 Runtime

## 14.1 Group-wise

如果：

```text
group_size = 128
```

则每 128 个 weight 共用：

- scale
- optional zero-point

对于 4096 个 input features：

$$
4096/128=32
$$

每个 output channel 会有 32 个 group。

---

## 14.2 INT4 Packing

两个 INT4 可以塞进一个 byte。

8 个 INT4 可以塞进一个 int32 container。

因此 GPTQ checkpoint 常见：

```text
qweight: int32 tensor
```

但逻辑上里面是：

```text
packed INT4 weight
```

---

## 14.3 常见字段

```text
qweight
scales
qzeros
g_idx
bias
```

### qweight

packed INT4。

### scales

group-wise scale。

### qzeros

asymmetric zero-point。

### g_idx

input channel 到 quantization group 的映射。

---

## 14.4 W4A16 推理

逻辑上：

```text
FP16 activation
    ×
packed INT4 weight
    ↓
unpack / dequant
    ↓
GEMM
```

高效 kernel 会做：

```text
load packed INT4
↓
unpack in register/tile
↓
apply scale
↓
fused GEMM
```

而不是完整解压成 FP16 再写回内存。

---

# 15. AWQ：Activation-aware Weight Quantization

AWQ 的核心不是：

> 大 Weight 更重要。

而是：

> 对应 activation 大的 weight/channel 更值得保护。

假设：

```text
w1 = 0.5,  x1 = 0.1
w2 = 0.05, x2 = 20
```

输出贡献：

$$
0.5\times0.1=0.05
$$

$$
0.05\times20=1
$$

所以虽然 $w_2$ 更小，它更重要。

---

## 15.1 AWQ Scaling

仍然利用：

$$
XW=(XS^{-1})(SW)
$$

但目的不是 SmoothQuant 的 activation smoothing，而是：

> 提高 salient weight 的有效量化精度。

---

## 15.2 Calibration

可用：

```python
act_stat = X.abs().mean(dim=0)
```

表示不同 input channel 的重要程度。

---

## 15.3 Scale Search

简化：

$$
s_j=a_j^\alpha
$$

然后搜索不同 $\alpha$：

```python
best_error = float("inf")

for alpha in alphas:

    scale = act_stat.pow(alpha)

    X_s = X / scale
    W_s = W * scale

    W_q = quantize_int4(W_s)

    Y_q = X_s @ dequant(W_q).T

    error = mse(Y_ref, Y_q)

    if error < best_error:
        best_error = error
        best_alpha = alpha
```

---

# 16. QuaRot：Rotation Quantization

核心公式：

$$
XW=(XR)(R^TW)
$$

其中 $R$ 是正交矩阵：

$$
R^TR=I
$$

Rotation 的目标是：

> 把集中在少数 channel 上的 outlier 能量摊到更多 channel。

---

## 16.1 二维例子

原 activation：

$$
x=[10,0]
$$

Hadamard rotation：

$$
R=
\frac{1}{\sqrt2}
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}
$$

得到：

$$
xR=[7.07,7.07]
$$

最大值：

```text
10 → 7.07
```

---

## 16.2 PyTorch Layout

PyTorch Linear：

```text
W.shape = [out_features, in_features]
```

若：

$$
X'=XR
$$

则：

$$
W'=WR
$$

因为：

$$
(XR)(WR)^T
=
XRR^TW^T
=
XW^T
$$

---

## 16.3 Fast Hadamard Transform

普通 dense rotation：

$$
O(n^2)
$$

Fast Hadamard：

$$
O(n\log n)
$$

适合 runtime rotation。

---

# 17. DuQuant：Rotation + Permutation

DuQuant 重点处理：

- normal outlier
- massive outlier
- block-wise imbalance

核心流程：

```text
Calibration
↓
找到 outlier dimensions
↓
Outlier-aware block rotation
↓
Zigzag permutation
↓
Second rotation
↓
W4A4 quantization
```

---

## 17.1 Block-wise Rotation

目的：

> 先在每个 block 内把 massive outlier 摊开。

---

## 17.2 Zigzag Permutation

假设：

```text
[10,9,8,7,1,1,1,1]
```

直接分 block：

```text
[10,9,8,7]
[1,1,1,1]
```

不均匀。

Permutation 后希望类似：

```text
[10,1,8,1]
[9,1,7,1]
```

让各 block dynamic range 更接近。

---

## 17.3 第二次 Rotation

Permutation 后再 rotation：

```text
block 内局部差异
↓
进一步摊平
```

---

# 18. 算法横向总结

| 方法 | 关键词 | 典型目标 | 核心思想 |
|---|---|---|---|
| SmoothQuant | 搬 | W8A8 | 把 activation 动态范围迁移到 weight |
| GPTQ | 补 | W4A16 | Hessian + error compensation |
| AWQ | 护 | W4A16 | 用 activation 保护重要 weight |
| QuaRot | 摊 | W4A4/W4A8 | 正交旋转摊平 outlier |
| DuQuant | 摊+排 | W4A4 | rotation + permutation + rotation |

可以压缩成：

```text
SmoothQuant → 搬
GPTQ        → 补
AWQ         → 护
QuaRot      → 摊
DuQuant     → 摊 + 排
```

**系统讲解**：这些算法不是互相替代的“量化开关”，而是在误差来源不同的情况下改变优化目标：SmoothQuant 主要处理 activation outlier，GPTQ/AWQ 主要处理 weight-only 误差，QuaRot/DuQuant 改变表示空间以降低 outlier。选择算法前应先通过 sensitivity 和 calibration 实验确定主要误差来自 activation、weight、attention、residual 还是 runtime。

**公式**：可以用统一的任务损失表示选择过程：

$$
\min_{\hat W,\hat X}\; \mathcal{L}\big(f_{\hat W,\hat X}(D), f_{W,X}(D)\big)
$$

其中 (D) 是 calibration/评估数据，(mathcal{L}) 可以是 tensor MSE、cosine loss、logit loss 或 VLA 的 action/task loss。不同算法主要区别在于约束集合、可调整变量和近似目标不同。

**代码实现**：

```python
def choose_candidate(y_ref, candidates):
    # candidates: {"w8a8": y1, "smooth_w8a8": y2, ...}
    scores = {name: evaluate_quant_error(y_ref, y)["mse"]
              for name, y in candidates.items()}
    return min(scores, key=scores.get), scores
```

---

# 19. 编译器量化与 NPU 落地

算法量化回答：

> 怎么量才能少掉精度？

编译器量化回答：

> 怎么把量化意图变成硬件真正可执行的低 bit graph？

主要包括：

```text
Q/DQ
Requant
Scale Propagation
Scale Folding
Operator Fusion
Graph Rewrite
Mixed Precision
Backend Lowering
INT8/INT4 Kernel
```

---

## 19.1 编译器眼里的 Tensor

```text
Tensor A
dtype = int8
scale = 0.02

Tensor B
dtype = int8
scale = 0.05
```

如果做 Add：

```text
scale mismatch
↓
插入 requant
```

---

## 19.2 Quantization Accuracy ≠ Quantization Speedup

例如：

```text
模型体积缩小 4×
```

不代表：

```text
Latency 缩短 4×
```

因为还有：

- dequant overhead
- memory movement
- unsupported op fallback
- kernel efficiency
- tiling
- layout
- bandwidth
- compute utilization

**系统讲解**：算法量化和编译器量化解决的是两个不同问题。算法决定 scale、clipping、granularity、bit allocation 和 reparameterization；compiler 决定这些信息能否被表示成目标 IR，是否能完成 fusion/lowering，最终是否能落到真正的低比特 kernel。一个模型可能在 Python FakeQuant 中精度很好，但因为 unsupported op 或 fallback，部署性能并没有提升。

**代码实现**：当前工程中的底层转换入口可以抽象为：

```python
from xhquant.api import DeviceType, QuantScheme
from xhquant.api import create_quant_config, convert_onnx_to_hmonnx

scheme = QuantScheme(
    target_device=DeviceType.XH2a,
    quant_type="w8a8h1_sefp",
)
config = create_quant_config(scheme)

convert_onnx_to_hmonnx(
    onnx_path,
    representative_inputs,
    out_hmonnx_file=output_path,
    device_type="XH2A",
    quant_config=config,
)
```

在 Spirit 工程中，`ptq.py:quantize_qwen3_vl()` 使用 `LLMConverter.from_pretrained()` 处理 Qwen3-VL；`ptq.py:quantize_dit()` 先用 `torch.onnx.export()` 导出 DiT，再调用 `convert_onnx_to_hmonnx()`。这些脚本是模型专用封装，不能直接等同于任意模型的通用入口。

---

# 20. VLA 量化：QuantVLA / QVLA 的位置

VLA：

```text
Image
↓
Vision Encoder
↓
VLM / LLM
↓
Action Head / DiT
↓
Action
↓
Robot
```

**系统讲解**：VLA 的量化评价必须分成三层：数值层、分布层和控制层。数值层比较 tensor 的 MSE/cosine；分布层检查 attention temperature、residual branch energy 和 action head 输出范围；控制层检查 action L2、轨迹偏差、碰撞率和任务成功率。前两层通过不能推出第三层通过，因为 action 会反馈到下一时刻的 observation。

**公式**：闭环误差可抽象为：

$$
s_{t+1}=T(s_t,a_t),\qquad a_t=\pi(o_t),\qquad o_t=O(s_t)
$$

量化策略 (hat\pi) 引入的动作扰动为：

$$
\delta a_t=\hat\pi(o_t)-\pi(o_t)
$$

即使单步 (|\delta a_t|) 很小，经过状态转移 (T) 后也可能影响后续观测和动作，因此 VLA 需要 trajectory/closed-loop 评估。

**代码实现**：当前 Spirit 工程的四段拆分可以用以下接口关系理解：

```python
# 只展示数据流，不代表通用 VLA API
image = preprocess_observation(obs.image)
vision_tokens = visual_hmm(image)
prefill_out = prefill_hmm(vision_tokens, text_ids, state, kv_cache)
action_cond = host_projection(prefill_out, obs.state)
action = dit_hmm(action_cond, timestep)
action = host_postprocess(action)
```

`ptq_dump.py` 中对应的工程验证组件是 `vision`、`prefill`、`decode`、`dit`；完整任务成功率仍需在策略级 runtime 中单独验证。

与普通 LLM 不同的是：

> 量化误差最终进入控制闭环。

因此：

```text
第 t 步误差
↓
机器人状态变化
↓
第 t+1 步输入变化
↓
误差可能继续积累
```

---

## 20.1 QuantVLA

重点：

```text
DuQuant-style reparameterization
+
Selective Quantization
+
ATM
+
OHB
```

它关注：

- attention temperature drift
- residual energy drift
- DiT sensitivity

---

## 20.2 QVLA

重点：

```text
Action-centric channel sensitivity
```

也就是：

> 某个 channel 量化误差最终对 robot action 的影响有多大？

---

# 21. 学习 Roadmap

## 阶段 1：数值表示

- FP32 / FP16 / BF16
- INT8 / INT4
- exponent / mantissa
- dynamic range
- overflow / underflow

## 阶段 2：基础 Quantizer

- scale
- zero-point
- symmetric / asymmetric
- per-tensor / per-channel / per-group
- clipping / saturation

## 阶段 3：整数推理

- W8A8 Linear
- INT32 accumulator
- bias quant
- requant
- Q/DQ
- scale propagation

## 阶段 4：Transformer Quant

- QKV
- Softmax
- LayerNorm
- Residual
- activation outlier

## 阶段 5：主流 PTQ 算法

- SmoothQuant
- GPTQ
- AWQ
- QuaRot
- DuQuant

## 阶段 6：Sensitivity + Mixed Precision

- layer-wise sensitivity
- activation / weight attribution
- selective fallback
- INT8/INT4/FP16 混合精度

## 阶段 7：Compiler / NPU

- quant IR
- QDQ lowering
- fusion
- kernel mapping
- tiling
- SRAM / DDR
- bandwidth
- NPU utilization

## 阶段 8：VLM / VLA

- VLM quantization
- DiT quantization
- QuantVLA
- QVLA
- temporal error accumulation
- closed-loop success rate

---

# 22. 阶段性自测

## 基础

1. 为什么 INT8 需要 scale，而 FP16 不需要外部 scale？
2. 为什么 INT8 × INT8 通常用 INT32 accumulator？
3. 为什么 bias 常常是 INT32？
4. Quantize / Dequantize / Requantize 有什么区别？

## Transformer

5. 为什么 Residual Add 前要对齐 scale？
6. 为什么 Softmax / LayerNorm 比 Linear 更难整数化？
7. 为什么 activation outlier 会让 per-tensor quantization 失效？

## SmoothQuant

8. 为什么：

$$
(XS^{-1})(SW)=XW
$$

9. $\alpha$ 太大会发生什么？
10. 为什么 SmoothQuant 需要 Calibration？

## GPTQ

11. 为什么 GPTQ 优化的是：

$$
\|X(W-\hat W)\|^2
$$

而不是：

$$
\|W-\hat W\|^2
$$

12. $X^TX$ 的对角线和非对角线分别表示什么？
13. 什么是 Sequential Quantization？

## AWQ

14. 为什么小 Weight 不一定不重要？
15. 为什么 AWQ 主要看 activation magnitude？

## QuaRot / DuQuant

16. 为什么正交旋转不丢信息？
17. Hadamard 为什么适合 runtime rotation？
18. DuQuant 为什么需要 permutation？

## 部署

19. 为什么 W4A16 与 W8A8 的 runtime 路径不同？
20. 为什么模型压缩 4× 不代表 latency 4×？
21. 什么情况下算法量化精度很好，但 NPU 性能反而不好？

---

# 最终目标

真正应该形成的不是“认识很多算法名”，而是下面这条排障链：

```text
模型量化后精度下降
        ↓
哪一层先开始漂？
        ↓
Weight 还是 Activation？
        ↓
是不是 outlier？
        ↓
是不是 calibration 数据不代表真实 workload？
        ↓
scale / clipping / granularity 是否合理？
        ↓
per-channel / per-group 是否能解决？
        ↓
是否需要 Smooth / Rotation / Compensation？
        ↓
是否存在 sensitive layer？
        ↓
Mixed Precision / FP16 fallback
        ↓
精度恢复
        ↓
再看：
QDQ / Kernel / Bandwidth / NPU Utilization
```

最终能力目标：

> **看到一个真实模型量化失败，不只是知道“可以换个算法”，而是能定位误差来源、判断该修改 quantizer、数据分布、bit allocation、mixed precision，还是编译器/Kernel。**


---

# 23. Sensitivity Analysis + Mixed Precision

这一部分开始从“理解量化算法”转向“真实模型量化失败时如何定位问题”。

**系统讲解**：Sensitivity analysis 的目标是估计“把某个模块从 FP 改成量化后，系统损失增加多少”。它不是对模块本身做静态打分，而是一个带上下文的干预实验：其他模块、输入数据、随机种子和评价指标都应保持一致。单层 sensitivity 只能近似衡量局部影响，多个敏感层同时量化时还可能出现非线性交互，因此最终仍需要全模型验证。

**公式**：对模块 (i) 的任务指标 sensitivity 可写为：

$$
S_i=M(f_{FP},D)-M(f_{Q_i},D)
$$

若使用 tensor-level reference，则可以定义：

$$
S_i^{tensor}=\operatorname{MSE}(y_i^{FP},y_i^{Q})
\quad\text{或}\quad
1-\operatorname{CosSim}(y_i^{FP},y_i^{Q})
$$

**代码实现**：

```python
def layer_sensitivity(fp_metric, quantized_metric):
    # 对 accuracy 是下降量；对 loss 可反向定义
    return fp_metric - quantized_metric

for name, module in model.named_modules():
    if not is_candidate(name, module):
        continue
    replace_with_quantized(module)
    metric = evaluate(model, calibration_or_eval_data)
    print(name, layer_sensitivity(fp_metric, metric))
    restore_fp(module)
```

## 23.1 为什么要做 Sensitivity Analysis

假设一个 Transformer 有 32 层。

直接做：

```text
全模型 W8A8
```

结果：

```text
FP16 accuracy = 85%
INT8 accuracy = 71%
```

真正应该先问的不是“换哪个量化算法”，而是：

> 到底是哪几层导致了主要精度损失？

现实里常见情况是：

```text
Layer 0   → -0.1%
Layer 1   → -0.2%
Layer 2   → -0.1%
...
Layer 17  → -6.8%   ← sensitive
...
Layer 29  → -4.2%   ← sensitive
```

这时最简单的解法可能只是：

```text
Layer 17 → FP16
Layer 29 → FP16
其他层   → INT8
```

这就是 Mixed Precision。

---

## 23.2 Sensitivity 的定义

可简单理解为：

> 某个模块被量化后，对最终模型指标造成多大影响。

例如：

$$
Sensitivity_i
=
Metric_{FP}
-
Metric_{QuantizeLayer_i}
$$

如果：

```text
FP16 accuracy = 90%
```

只量化 Layer 5 后：

```text
accuracy = 89.9%
```

则 Layer 5 sensitivity 很低。

如果只量化 Layer 12 后：

```text
accuracy = 80%
```

则 Layer 12 sensitivity 很高。

---

## 23.3 最朴素的 Layer-wise Sensitivity Analysis

```text
FP baseline
↓
只量化 Layer0
↓
Evaluate
↓
恢复 FP
↓
只量化 Layer1
↓
Evaluate
...
```

最终得到：

```text
Layer      Accuracy Drop
layer0     0.03%
layer1     0.12%
layer2     0.05%
layer3     4.80%   ← sensitive
layer4     0.20%
layer5     7.20%   ← very sensitive
```

概念代码：

```python
import copy
import torch

baseline_score = evaluate(model)

results = []

for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue

    test_model = copy.deepcopy(model)

    target = dict(
        test_model.named_modules()
    )[name]

    quantize_linear_inplace(target)

    score = evaluate(test_model)

    drop = baseline_score - score
    results.append((name, drop))

results.sort(
    key=lambda x: x[1],
    reverse=True
)
```

真实大模型不会直接 deepcopy 多次，这里只是说明逻辑。

---

## 23.4 为什么不同层敏感度差很多

常见原因包括：

### Activation Distribution 不同

```text
Layer A:
[-1, 1]

Layer B:
[-2, 1, 0.5, 37]
```

Layer B 有明显 outlier，更难量化。

### 信息瓶颈

例如：

```text
Vision Encoder
↓
Projector
```

或：

```text
Transformer
↓
Action Head
```

这类位置一点误差可能就会被放大。

### 误差传播路径不同

某些误差会进入：

```text
Softmax
Residual
DiT iterative denoising
```

从而被进一步放大。

---

## 23.5 区分 Weight Sensitivity 和 Activation Sensitivity

发现 Layer 12 很敏感后，不要立刻整层回退 FP16。

做两个实验：

### Weight-only

```text
W8A16
```

### Activation-only

```text
W16A8
```

假设：

```text
W8A16 accuracy = 89.8%
W16A8 accuracy = 75.0%
```

则主要问题来自 Activation。

优先考虑：

```text
calibration
clipping
SmoothQuant
rotation
```

如果反过来：

```text
W4A16 掉得明显
W16A8 正常
```

则主要是 Weight 问题。

优先考虑：

```text
per-channel
per-group
GPTQ
AWQ
higher bit
```

---

## 23.6 Sensitivity 决策树

```text
某层敏感
   ↓
拆开测试
   │
   ├─ Weight-only 掉？
   │       ↓
   │    Weight 问题
   │       ↓
   │    GPTQ / AWQ
   │    per-channel
   │    higher bit
   │
   └─ Activation-only 掉？
           ↓
       Activation 问题
           ↓
       calibration
       clipping
       SmoothQuant
       QuaRot / DuQuant
```

---

## 23.7 Tensor-level Error

任务级 evaluation 成本很高，因此常先比较中间 tensor：

### MSE

$$
MSE
=
\frac1N
\sum_i
(y_i-\hat y_i)^2
$$

```python
error = torch.mean(
    (output_fp - output_q) ** 2
)
```

### Cosine Similarity

$$
cos(x,y)
=
\frac{x\cdot y}
{\|x\|\|y\|}
$$

```python
import torch.nn.functional as F

sim = F.cosine_similarity(
    y_fp.flatten(),
    y_q.flatten(),
    dim=0
)
```

---

## 23.8 SQNR

Signal-to-Quantization-Noise Ratio：

$$
SQNR
=
10\log_{10}
\left(
\frac{\|x\|^2}
{\|x-\hat x\|^2}
\right)
$$

其中：

```text
x     = FP tensor
x_hat = quantized/dequantized tensor
```

SQNR 越高，quantization noise 越小。

---

## 23.9 为什么单层 MSE 不等于 Task Sensitivity

假设：

```text
Layer A MSE = 0.1
Layer B MSE = 0.01
```

不能直接认为 Layer A 更重要。

Layer B 可能位于：

```text
Attention logits
Action Head
Final classifier
```

少量误差就可能改变最终结果。

所以 sensitivity 最好分三层：

```text
Level 1:
Tensor Error
MSE / Cosine / SQNR

Level 2:
Model Output Error
logits / KL / PPL

Level 3:
Task Metric
Accuracy / Success Rate
```

---

## 23.10 Mixed Precision

Sensitivity 分析后：

```text
Layer0 → low
Layer1 → low
Layer2 → high
Layer3 → low
```

可以配置：

```text
Layer0 → INT8
Layer1 → INT8
Layer2 → FP16
Layer3 → INT8
```

更进一步：

```text
Layer0 → INT4
Layer1 → INT8
Layer2 → FP16
Layer3 → INT4
```

---

## 23.11 Mixed Precision 是约束优化

目标：

$$
\min Cost(b_1,b_2,\dots,b_n)
$$

subject to：

$$
Accuracy(b_1,b_2,\dots,b_n)
\ge A_{min}
$$

其中 $b_i$ 可以是：

```text
INT4
INT8
FP16
```

---

## 23.12 Greedy Mixed Precision

例如：

```text
FP baseline = 90%
All INT8    = 80%
Target      = 88%
```

Sensitivity 排名：

```text
Layer12
Layer7
Layer29
...
```

依次恢复：

```text
Layer12 → FP16
accuracy = 84%

Layer7 → FP16
accuracy = 87%

Layer29 → FP16
accuracy = 88.5%
```

得到：

```text
29 层 INT8
3 层 FP16
```

---

## 23.13 NPU 上的 Mixed Precision 代价

如果：

```text
INT8
↓
FP16
↓
INT8
```

通常意味着：

```text
INT8
↓ DQ
FP16
↓ kernel
FP16
↓ Q
INT8
```

所以敏感层过于碎片化会引入：

- Q/DQ overhead
- layout conversion
- memory movement
- fusion 被打断

因此：

> 精度最优的 Mixed Precision，不一定是性能最优的 Mixed Precision。

---

## 23.14 硬件感知 Mixed Precision

除了 sensitivity，还应同时考虑：

```text
kernel support
Q/DQ overhead
layer latency
memory bandwidth
fusion
```

如果某个 sensitive layer：

```text
FP16 latency = 0.05 ms
```

保留 FP16 代价可能很低。

如果某个巨大 MLP：

```text
FP16 fallback → latency +40%
```

就值得优先用：

```text
SmoothQuant
QuaRot
DuQuant
```

尝试把它救回低 bit。

---

## 23.15 实际排障示例

假设：

```text
FP16 accuracy = 92%
INT8 accuracy = 79%
```

### Step 1：Layer-wise compare

发现：

```text
Layer14 cosine = 0.83
其他层大多 > 0.98
```

### Step 2：分离 Weight / Activation

```text
W8A16 → 91.5%
W16A8 → 81%
```

说明主要是 Activation。

### Step 3：检查分布

```text
activation max = 38
99.9 percentile = 4.2
```

明显 outlier。

### Step 4：尝试 SmoothQuant

```text
INT8 accuracy → 90.8%
```

如果仍无法恢复，则：

```text
Layer14 → FP16
```

然后重新评估 Accuracy + Latency。

---

## 23.16 VLA 中的 Sensitivity

VLA 不能只看：

```text
Layer MSE
```

还要看：

```text
Action Error
Temporal Stability
Task Success Rate
```

因此需要：

> Action-aware / Task-aware Sensitivity

这也是 QVLA 一类方法的动机。

---

## 23.17 这一阶段的核心排障树

```text
精度下降
   ↓
定位 first bad layer
   ↓
Layer-wise sensitivity
   ↓
Weight-only / Activation-only 分离
   │
   ├─ Weight 问题
   │    ├─ per-channel
   │    ├─ group size
   │    ├─ GPTQ
   │    └─ AWQ
   │
   └─ Activation 问题
        ├─ calibration
        ├─ clipping
        ├─ SmoothQuant
        ├─ QuaRot
        └─ DuQuant
   ↓
仍无法恢复
   ↓
Mixed Precision
   ↓
重新测 Accuracy + Latency
```

核心结论：

1. Sensitivity Analysis 是为了找到哪些位置不适合低 bit。
2. Weight 和 Activation sensitivity 要尽量分开判断。
3. Tensor MSE 不等于 Task Sensitivity。
4. Mixed Precision 是 accuracy 与 cost 的约束优化。
5. NPU 上 FP16 fallback 的代价包含 Q/DQ、访存和 fusion 损失。


---

# 24. Quantization Error Debugging：真实量化问题定位

**系统讲解**：First Bad Layer 的定义必须依赖一个明确阈值和一致的比较协议。建议同时记录输入误差与输出误差：如果某层输入已经坏，问题来自上游；如果输入正常而输出突然坏，问题更可能在该层的 weight、activation、scale、layout、kernel 或 fusion。只看最终 output 会丢失定位信息。

**公式**：给定参考输出 (y_{fp}) 和量化输出 (y_q)，可定义相对误差：

$$
e_{rel}=\frac{\|y_{fp}-y_q\|_2}{\|y_{fp}\|_2+\epsilon}
$$

以及 cosine drop：

$$
d_{cos}=1-\frac{y_{fp}\cdot y_q}{\|y_{fp}\|_2\|y_q\|_2+\epsilon}
$$

**代码实现**：

```python
def first_bad_layer(rows, rel_threshold=0.05, cosine_threshold=0.99):
    for row in rows:  # rows 按执行顺序排列
        if row["rel_error"] > rel_threshold or row["cosine"] < cosine_threshold:
            return row["name"]
    return None
```

这一部分关注真实工程问题：

> FP16 模型正常，但 INT8 / INT4 模型精度明显下降时，如何系统定位问题。

---

## 24.1 First Bad Layer

最终输出误差最大的位置，不一定是问题源头。

例如：

```text
Layer0 cosine = 0.9999
Layer1 cosine = 0.9987
Layer2 cosine = 0.9979
Layer3 cosine = 0.82
Layer4 cosine = 0.74
```

这里应该优先检查：

```text
Layer3
```

因为 Layer4 很可能只是继承并放大了 Layer3 的误差。

因此：

> First bad layer 比 worst layer 更重要。

---

## 24.2 FP / Quant 模型逐层 Hook

```python
class OutputCollector:
    def __init__(self):
        self.outputs = {}

    def hook(self, name):
        def fn(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]

            self.outputs[name] = (
                output.detach()
                .float()
                .cpu()
            )

        return fn
```

注册 FP 模型：

```python
fp_collector = OutputCollector()
fp_handles = []

for name, module in fp_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        handle = module.register_forward_hook(
            fp_collector.hook(name)
        )
        fp_handles.append(handle)
```

Quant 模型同理。

同一份输入：

```python
with torch.no_grad():
    fp_model(inputs)
    quant_model(inputs)
```

然后比较同名节点输出。

---

## 24.3 三个基础误差指标

### MSE

$$
MSE
=
\frac1N
\sum_i
(x_i-\hat{x}_i)^2
$$

```python
mse = torch.mean(
    (fp_out - q_out) ** 2
)
```

### Cosine Similarity

$$
cos(x,\hat{x})
=
\frac{x\cdot\hat{x}}
{\|x\|\|\hat{x}\|}
$$

```python
import torch.nn.functional as F

cosine = F.cosine_similarity(
    fp_out.flatten(),
    q_out.flatten(),
    dim=0
)
```

### SQNR

$$
SQNR
=
10\log_{10}
\left(
\frac{\|x\|^2}
{\|x-\hat{x}\|^2}
\right)
$$

```python
signal = torch.sum(fp_out ** 2)

noise = torch.sum(
    (fp_out - q_out) ** 2
)

sqnr = 10 * torch.log10(
    signal / noise
)
```

---

## 24.4 统一比较函数

```python
def compare_tensor(fp, q, eps=1e-12):
    fp = fp.float()
    q = q.float()

    mse = torch.mean(
        (fp - q) ** 2
    ).item()

    cos = torch.nn.functional.cosine_similarity(
        fp.flatten(),
        q.flatten(),
        dim=0
    ).item()

    signal = torch.sum(fp ** 2)

    noise = torch.sum(
        (fp - q) ** 2
    )

    sqnr = 10 * torch.log10(
        (signal + eps)
        /
        (noise + eps)
    )

    return {
        "mse": mse,
        "cosine": cos,
        "sqnr_db": sqnr.item()
    }
```

逐层执行：

```python
for name in fp_collector.outputs:

    if name not in q_collector.outputs:
        continue

    metrics = compare_tensor(
        fp_collector.outputs[name],
        q_collector.outputs[name]
    )

    print(name, metrics)
```

---

## 24.5 Input 正常、Output 异常

如果某层 output 很差，先检查该层 input。

### 情况 A

```text
input cosine  = 0.80
output cosine = 0.75
```

说明当前层可能只是继承了上游误差。

### 情况 B

```text
input cosine  = 0.9999
output cosine = 0.75
```

这才强烈说明：

> 当前 Layer / Operator 自己有问题。

---

## 24.6 First Bad Operator

还可以从 Layer 继续细分：

```text
LayerNorm
↓
Q/K/V
↓
Attention Logits
↓
Softmax
↓
O Projection
```

例如：

```text
LayerNorm output  cos = 0.999
Q output          cos = 0.998
K output          cos = 0.997
Attention logits  cos = 0.81
```

则 first bad operator 很可能是：

```text
Attention logits
```

---

## 24.7 Weight / Activation 分离实验

对于某个异常 Linear：

```text
W16A16 baseline
W8A16  weight-only
W16A8  activation-only
```

例如：

```text
W16A16 cos = 1.000
W8A16  cos = 0.998
W16A8  cos = 0.76
```

说明 Activation 是主要问题。

---

## 24.8 Activation 分布分析

```python
x = activation.detach().float()

print("min:", x.min())
print("max:", x.max())
print("mean:", x.mean())
print("std:", x.std())

abs_x = x.abs().flatten()

print(
    torch.quantile(
        abs_x,
        torch.tensor([
            0.90,
            0.99,
            0.999,
            1.0
        ])
    )
)
```

例如：

```text
90%   = 1.2
99%   = 2.8
99.9% = 4.7
max   = 38.5
```

说明极少数 outlier 把 scale 撑大。

可能考虑：

```text
percentile clipping
SmoothQuant
QuaRot
DuQuant
```

---

## 24.9 Calibration Distribution Shift

如果 calibration：

```text
activation max = 5
```

真实 runtime：

```text
activation max = 17
```

则固定 scale 很可能导致 saturation。

因此需要比较：

```text
Calibration Distribution
vs
Runtime Distribution
```

---

## 24.10 Saturation Ratio

```python
def saturation_ratio(x, scale):
    q = torch.round(x / scale)

    saturated = (
        (q > 127) |
        (q < -128)
    )

    return saturated.float().mean()
```

高 saturation ratio 往往说明：

- calibration 不匹配
- scale 太小
- clipping 策略不合理

但 saturation 很低不代表量化一定好，因为 scale 过大仍会造成 rounding resolution 不足。

---

## 24.11 Weight Reconstruction Error

```python
W_fp = module.weight.float()

W_hat = dequantize(
    quantize(W_fp)
)

metrics = compare_tensor(
    W_fp,
    W_hat
)
```

还可以看 per-channel：

```python
channel_error = torch.mean(
    (W_fp - W_hat) ** 2,
    dim=1
)
```

如果少数 channel 特别差，可以考虑：

```text
per-channel
per-group
GPTQ
AWQ
```

---

## 24.12 三个 Reference Baseline

真实 NPU 精度排障建议建立：

```text
Baseline 1:
FP Reference

Baseline 2:
Software Quant Reference

Target:
NPU Quant Result
```

### 情况 A

```text
FP        accuracy 92%
FakeQuant accuracy 80%
NPU       accuracy 79.5%
```

说明主要是量化算法 / 参数问题。

### 情况 B

```text
FP        accuracy 92%
FakeQuant accuracy 91.5%
NPU       accuracy 75%
```

高度怀疑：

```text
compiler
kernel
Q/DQ
requant
scale
zero-point
layout
overflow
```

---

## 24.13 Software Quant 正常、NPU 异常时优先排查

1. scale
2. zero-point
3. quant axis
4. weight layout
5. bias scale
6. requant multiplier
7. shift
8. rounding mode
9. saturation behavior
10. accumulator overflow
11. operator fusion
12. tensor layout transform

---

## 24.14 Rounding Mode

例如：

```text
2.5
```

不同实现可能得到：

```text
round-half-up       → 3
round-to-nearest-even → 2
```

因此 reference quantizer 应尽量模拟硬件真实 rounding。

---

## 24.15 Bias Scale

如果：

$$
s_{acc}=s_xs_w
$$

则：

$$
s_b=s_xs_w
$$

若 weight 是 per-channel：

$$
s_{w,j}
$$

则：

$$
s_{b,j}=s_xs_{w,j}
$$

Bias scale 使用错误会造成明显 channel 偏移。

---

## 24.16 Requant Debug

理论：

$$
q_y=
round
\left(
acc
\frac{s_xs_w}{s_y}
\right)
$$

硬件通常实现成：

```text
integer multiplier
+
shift
+
round
+
saturate
```

排障时建议同时 dump：

```text
sx
sw
sy
multiplier
shift
```

---

## 24.17 Accumulator Overflow

粗略 worst-case：

$$
acc_{max}
\approx
K\times127\times127
$$

例如：

$$
K=4096
$$

则约：

$$
66M
$$

相比 INT32：

$$
2^{31}-1\approx2.147B
$$

仍有较大余量。

但具体硬件可能使用特殊 accumulator 或分段 accumulate，因此仍要结合 backend。

---

## 24.18 Layout 问题

理论 weight：

```text
[out, in]
```

实际 NPU kernel 可能使用：

```text
blocked layout
tile layout
packed layout
```

因此：

> 数值量化正确，不代表 layout mapping 一定正确。

---

## 24.19 Operator Fusion

例如：

```text
Conv
+
Bias
+
ReLU
```

被 fuse 后，如果输出异常，可能需要：

```text
关闭 fusion
dump fusion 前后 IR
逐算子 reference compare
```

才能判断具体哪一步有问题。

---

## 24.20 推荐排障顺序

```text
Step 1
确认 FP baseline

Step 2
建立 Software Quant baseline

Step 3
比较 NPU Quant

Step 4
逐层 hook / dump

Step 5
找 first bad layer

Step 6
找 first bad operator

Step 7
比较 op input / output

Step 8
拆 Weight / Activation

Step 9
检查：
scale
zero-point
clipping
outlier
distribution shift

Step 10
Software Quant 好、NPU 坏：
查 compiler / kernel / layout / requant

Step 11
Software Quant 自己坏：
查 calibration / algorithm / mixed precision
```

---

## 24.21 Debug 决策树

```text
NPU INT8 精度差
        ↓
FP 正常吗？
        │
        └─ 否 → 不是量化问题
        ↓ 是
Software Quant 正常吗？
        │
        ├─ 否
        │   ↓
        │  Quant Algorithm 问题
        │   ↓
        │  calibration
        │  scale
        │  clipping
        │  outlier
        │  sensitivity
        │  mixed precision
        │
        └─ 是
            ↓
          NPU Execution 问题
            ↓
          Q/DQ
          requant
          rounding
          bias scale
          layout
          fusion
          kernel
```

---

## 24.22 VLA Debug

VLA 可以沿着：

```text
Vision feature
↓
Projector output
↓
VLM hidden state
↓
Action latent
↓
Action output
↓
Task success
```

逐级比较。

最终目标是找到：

> 是前端 representation 先漂，还是 action head 对少量误差特别敏感。

---

## 24.23 核心原则

1. 先找 first bad layer，而不是只找最终误差最大的层。
2. 先区分 Software Quant 问题还是 NPU Execution 问题。
3. Input 正常、Output 异常，才说明当前算子自身值得重点查。
4. 量化排障必须同时看数值误差、任务敏感度和硬件执行语义。


---

# 25. Quantization Debug Harness：把排障方法代码化

**系统讲解**：debug harness 应把一次排障实验固定成可复现的数据记录，而不是只在终端打印一个 cosine。最小记录单元应包含 module name、输入/输出 shape、dtype、scale、min/max、RMS、saturation ratio、MSE 和 cosine，并保存 FP、FakeQuant、NPU 三路结果的来源。

**代码实现**：

```python
def compare_tensor(name, ref, test):
    a, b = ref.detach().float(), test.detach().float()
    diff = a - b
    return {
        "name": name,
        "shape": tuple(a.shape),
        "mse": diff.square().mean().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1)
        ).item(),
    }
```

当前 `ptq_dump.py` 的 `compare_named_arrays()`、`write_cosine_report()` 和 `write_pair_report()` 可以视为这一思想的离线 golden 版本；它们面向 ONNX IO 名称，而通用 debug harness 通常面向 PyTorch module 名称。

这一部分把前面的排障方法做成一个可复用的小型工具框架。

目标：

```text
1. 给 FP / Quant 模型挂 Hook
2. 抓同名模块输入输出
3. 自动计算 MSE / MAE / Cosine / SQNR
4. 按误差排序
5. 找 first bad layer / operator
6. 扩展到 FP / FakeQuant / NPU 三路对比
```

---

## 25.1 基础数据结构

```python
class TensorRecord:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
```

Collector：

```python
class ModelCollector:
    def __init__(self):
        self.records = TensorRecord()
        self.handles = []
        self.execution_order = []
```

---

## 25.2 Input Hook

```python
def make_input_hook(self, name):

    def hook(module, inputs):

        if not inputs:
            return

        x = inputs[0]

        if torch.is_tensor(x):

            self.records.inputs[name] = (
                x.detach()
                .float()
                .cpu()
            )

    return hook
```

Input 很重要，因为：

```text
Input 正常
Output 异常
```

才更能说明当前算子自己存在问题。

---

## 25.3 Output Hook

```python
def make_output_hook(self, name):

    def hook(module, inputs, output):

        if name not in self.execution_order:
            self.execution_order.append(name)

        if isinstance(output, tuple):
            output = output[0]

        if torch.is_tensor(output):

            self.records.outputs[name] = (
                output.detach()
                .float()
                .cpu()
            )

    return hook
```

---

## 25.4 完整 Collector

```python
import torch
import torch.nn as nn

class TensorRecord:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}


class ModelCollector:

    def __init__(self):
        self.records = TensorRecord()
        self.handles = []
        self.execution_order = []

    def make_input_hook(self, name):

        def hook(module, inputs):

            if not inputs:
                return

            x = inputs[0]

            if torch.is_tensor(x):

                self.records.inputs[name] = (
                    x.detach()
                    .float()
                    .cpu()
                )

        return hook

    def make_output_hook(self, name):

        def hook(module, inputs, output):

            if name not in self.execution_order:
                self.execution_order.append(name)

            if isinstance(output, tuple):
                output = output[0]

            if torch.is_tensor(output):

                self.records.outputs[name] = (
                    output.detach()
                    .float()
                    .cpu()
                )

        return hook

    def register(
        self,
        model,
        module_types=(nn.Linear,)
    ):

        for name, module in model.named_modules():

            if isinstance(module, module_types):

                h1 = module.register_forward_pre_hook(
                    self.make_input_hook(name)
                )

                h2 = module.register_forward_hook(
                    self.make_output_hook(name)
                )

                self.handles.extend([
                    h1,
                    h2
                ])

    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles.clear()
```

---

## 25.5 误差指标函数

```python
import torch.nn.functional as F

def tensor_metrics(
    ref,
    test,
    eps=1e-12
):

    ref = ref.float()
    test = test.float()

    diff = ref - test

    mse = torch.mean(
        diff ** 2
    ).item()

    mae = torch.mean(
        diff.abs()
    ).item()

    max_error = diff.abs().max().item()

    cosine = F.cosine_similarity(
        ref.flatten(),
        test.flatten(),
        dim=0,
        eps=eps
    ).item()

    signal = torch.sum(
        ref ** 2
    )

    noise = torch.sum(
        diff ** 2
    )

    sqnr = (
        10
        * torch.log10(
            (signal + eps)
            /
            (noise + eps)
        )
    ).item()

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "cosine": cosine,
        "sqnr_db": sqnr,
    }
```

---

## 25.6 自动比较两个模型

```python
def compare_collectors(
    ref_collector,
    test_collector
):

    results = []

    ref_outputs = (
        ref_collector.records.outputs
    )

    test_outputs = (
        test_collector.records.outputs
    )

    for name, ref_out in ref_outputs.items():

        if name not in test_outputs:
            continue

        test_out = test_outputs[name]

        if ref_out.shape != test_out.shape:
            continue

        metrics = tensor_metrics(
            ref_out,
            test_out
        )

        metrics["name"] = name

        results.append(metrics)

    return results
```

按 cosine 从差到好排序：

```python
results = compare_collectors(
    fp_collector,
    q_collector
)

results = sorted(
    results,
    key=lambda x: x["cosine"]
)

for item in results[:10]:
    print(item)
```

---

## 25.7 First Bad Layer

固定阈值：

```python
def find_first_bad_layer(
    ref_collector,
    test_collector,
    cosine_threshold=0.98
):

    for name in ref_collector.execution_order:

        if (
            name not in
            test_collector.records.outputs
        ):
            continue

        ref = (
            ref_collector
            .records
            .outputs[name]
        )

        test = (
            test_collector
            .records
            .outputs[name]
        )

        metrics = tensor_metrics(
            ref,
            test
        )

        if (
            metrics["cosine"]
            < cosine_threshold
        ):
            return name, metrics

    return None, None
```

注意：

> 0.98 只是筛查阈值，不是行业统一真理。

---

## 25.8 最大 Cosine Drop

比固定阈值更实用的方法是找误差突变。

定义：

$$
\Delta_i
=
cos_{i-1}
-
cos_i
$$

实现：

```python
def find_largest_cosine_drop(
    ref_collector,
    test_collector
):

    history = []

    for name in ref_collector.execution_order:

        if (
            name not in
            test_collector.records.outputs
        ):
            continue

        ref = (
            ref_collector
            .records.outputs[name]
        )

        test = (
            test_collector
            .records.outputs[name]
        )

        metrics = tensor_metrics(
            ref,
            test
        )

        history.append(
            (
                name,
                metrics["cosine"]
            )
        )

    best = None

    for i in range(
        1,
        len(history)
    ):

        prev_name, prev_cos = history[i - 1]
        name, cos = history[i]

        drop = prev_cos - cos

        if (
            best is None
            or drop > best["drop"]
        ):
            best = {
                "previous": prev_name,
                "name": name,
                "drop": drop,
                "cosine": cos
            }

    return best
```

---

## 25.9 Input / Output 联合判断

```python
def compare_module_io(
    name,
    ref_collector,
    test_collector
):

    result = {}

    ref_inputs = (
        ref_collector.records.inputs
    )

    test_inputs = (
        test_collector.records.inputs
    )

    ref_outputs = (
        ref_collector.records.outputs
    )

    test_outputs = (
        test_collector.records.outputs
    )

    if (
        name in ref_inputs
        and name in test_inputs
    ):
        result["input"] = tensor_metrics(
            ref_inputs[name],
            test_inputs[name]
        )

    if (
        name in ref_outputs
        and name in test_outputs
    ):
        result["output"] = tensor_metrics(
            ref_outputs[name],
            test_outputs[name]
        )

    return result
```

判断：

```text
input cosine ≈ 1
output cosine 很低
```

强烈指向当前算子自身。

---

## 25.10 Activation Distribution Statistics

```python
def tensor_statistics(x):

    x = x.float()

    abs_x = x.abs().flatten()

    quantiles = torch.quantile(
        abs_x,
        torch.tensor([
            0.5,
            0.9,
            0.99,
            0.999,
            1.0
        ])
    )

    return {
        "min": x.min().item(),
        "max": x.max().item(),
        "mean": x.mean().item(),
        "std": x.std().item(),

        "abs_p50": quantiles[0].item(),
        "abs_p90": quantiles[1].item(),
        "abs_p99": quantiles[2].item(),
        "abs_p999": quantiles[3].item(),
        "abs_max": quantiles[4].item(),
    }
```

例如：

```text
p99   = 1.8
p99.9 = 3.0
max   = 42
```

说明存在明显尖锐 outlier。

---

## 25.11 Outlier Ratio

可定义一个调试 heuristic：

$$
R
=
\frac{\max |x|}
{P_{99.9}(|x|)}
$$

```python
def outlier_ratio(x):

    abs_x = (
        x.float()
        .abs()
        .flatten()
    )

    p999 = torch.quantile(
        abs_x,
        0.999
    )

    max_v = abs_x.max()

    return (
        max_v
        /
        (p999 + 1e-12)
    ).item()
```

该指标不是行业统一标准，但适合快速筛查极端 outlier。

---

## 25.12 Saturation Detection

```python
def quant_saturation_stats(
    x,
    scale,
    qmin=-128,
    qmax=127
):

    q_float = x / scale

    low = q_float < qmin
    high = q_float > qmax

    saturated = low | high

    return {
        "low_ratio":
            low.float().mean().item(),

        "high_ratio":
            high.float().mean().item(),

        "total_ratio":
            saturated.float()
            .mean()
            .item()
    }
```

可以区分：

```text
scale 太小导致 clipping
```

和：

```text
scale 太大导致 resolution 不足
```

---

## 25.13 FP / FakeQuant / NPU 三路比较

推荐建立：

```text
FP Reference
↓
FakeQuant Reference
↓
NPU Quant Result
```

对每层分别比较：

```text
FP vs FakeQuant
FakeQuant vs NPU
```

### 情况 A

```text
FP vs FakeQuant cosine = 0.75
FakeQuant vs NPU cosine = 0.999
```

说明：

> NPU 正确执行了一个本身就不够准的量化模型。

属于量化算法 / calibration 问题。

### 情况 B

```text
FP vs FakeQuant cosine = 0.998
FakeQuant vs NPU cosine = 0.71
```

说明：

> 理论量化没问题，NPU 执行路径有问题。

应重点排查：

```text
Q/DQ
requant
rounding
layout
kernel
fusion
```

---

## 25.14 NPU Dump 对齐问题

真实工程中最难的通常不是 MSE，而是：

```text
PyTorch module
↕
ONNX node
↕
Compiler IR op
↕
NPU kernel
```

的 mapping。

例如：

```text
PyTorch:
layers.12.mlp.fc1

Compiler IR:
quant_gemm_427

Runtime:
tensor_813
```

成熟的 debug 工具需要维护这套映射。

---

## 25.15 Shape 与 Layout 对齐

比较前必须检查：

```python
if ref.shape != test.shape:
    print(
        "Shape mismatch:",
        ref.shape,
        test.shape
    )
```

例如：

```text
NPU:
[N,H,W,C]

PyTorch:
[N,C,H,W]
```

必须先：

```python
npu = npu.permute(
    0, 3, 1, 2
)
```

再进行数值比较。

---

## 25.16 Packed INT4 的 Debug

GPTQ / AWQ 场景下：

```text
packed qweight
↓
unpack
↓
apply zero-point
↓
apply scale
↓
dequantized weight
↓
compare FP
```

不能直接拿 packed int32 container 和 FP weight 做数值比较。

---

## 25.17 Debug Harness 的最终结构

```text
QuantDebugHarness
│
├─ Capture
│   ├─ PyTorch Hook
│   ├─ FakeQuant Hook
│   └─ NPU Dump Loader
│
├─ Alignment
│   ├─ module mapping
│   ├─ shape
│   └─ layout
│
├─ Metrics
│   ├─ MSE
│   ├─ MAE
│   ├─ Cosine
│   ├─ SQNR
│   ├─ Saturation
│   └─ Distribution
│
└─ Diagnosis
    ├─ first bad layer
    ├─ first bad op
    ├─ weight vs activation
    └─ software vs NPU
```

---

## 25.18 当前阶段最值得自己动手的实验

实现一个 Tiny Debug Harness：

```text
两个 PyTorch 模型
↓
自动 Hook 所有 Linear
↓
跑相同输入
↓
输出表格：

name
input_cos
output_cos
mse
sqnr
```

然后故意把某一层 scale 改坏，看工具能否自动找到 first bad layer。

这个实验能把：

```text
理论量化
→ 数值调试
→ 工程排障
```

真正连起来。


---

# 26. Calibration 策略与 Observer 设计

**系统讲解**：observer 不是量化器本身，而是把运行时样本分布压缩成量化参数的统计器。observer 的设计要与量化粒度、目标 dtype 和后端 kernel 约束同时确定；例如 per-channel weight observer 与 per-token activation observer 的 axis 完全不同，不能共用同一套 reduction 逻辑。

**公式**：对 (N) 个样本、channel (c) 的 percentile observer：

$$
m_c=\operatorname{Percentile}_{p}\left(\{|x_{n,c}|\}_{n=1}^{N}\right),
\qquad s_c=\frac{m_c}{Q_{max}}
$$

**代码实现**：

```python
def percentile_scale(x, percentile=0.999, qmax=127):
    flat = x.detach().abs().reshape(-1)
    bound = torch.quantile(flat, percentile)
    return bound.clamp_min(1e-12) / qmax
```

对 VLA calibration，还应按 image/text/state/timestep 分桶统计，避免一种模态或单一 diffusion timestep 主导全部 scale。

这一部分解决的问题是：

> Calibration 收集完分布后，如何决定 scale / zero-point / clipping threshold？

很多量化精度问题，本质上并不是“算法不够高级”，而是：

```text
Observer 选择不合适
Calibration 数据不代表真实 workload
Clipping threshold 不合理
```

## 26.1 Observer 是什么

Observer 可以理解成：

> 观察 Tensor 分布，并根据统计结果生成量化参数的模块。

```text
Activation Tensor
      ↓
Observer
      ↓
统计：
min / max / histogram / percentile / ...
      ↓
生成：
scale / zero-point / clipping threshold
```

Calibration 是整个数据采集过程，Observer 是统计和量化参数决策方式。

## 26.2 MinMax Observer

对于 symmetric INT8：

$$
q \in [-128,127]
$$

取：

$$
a=\max(|x_{min}|,|x_{max}|)
$$

则：

$$
s=\frac{a}{127}
$$

```python
class MinMaxObserver:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def update(self, x):
        cur_min = x.min().item()
        cur_max = x.max().item()

        if self.min_val is None:
            self.min_val = cur_min
            self.max_val = cur_max
        else:
            self.min_val = min(self.min_val, cur_min)
            self.max_val = max(self.max_val, cur_max)
```

MinMax 最大问题是：过度保护少数 outlier，导致普通值的量化分辨率变差。

## 26.3 Clipping Error 与 Rounding Error

$$
Total Error
=
Clipping Error
+
Rounding Error
$$

Range 太小：

```text
clipping ↑
resolution ↑
```

Range 太大：

```text
clipping ↓
resolution ↓
```

## 26.4 Percentile Observer

例如：

```text
p99.9 = 4.5
max   = 25
```

则：

$$
s=\frac{4.5}{127}
$$

而不是：

$$
s=\frac{25}{127}
$$

```python
def percentile_scale(
    x,
    percentile=0.999,
    qmax=127
):
    abs_x = (
        x.detach()
        .float()
        .abs()
        .flatten()
    )

    threshold = torch.quantile(
        abs_x,
        percentile
    )

    return threshold / qmax
```

Percentile 越低，scale 越细，但 clipping 越严重。

## 26.5 MSE-based Calibration

目标：

$$
t^*
=
\arg\min_t
\|X-Q_t(X)\|^2
$$

```python
def fake_quant_symmetric(
    x,
    threshold,
    qmax=127
):
    scale = threshold / qmax
    scale = max(float(scale), 1e-12)

    q = torch.round(x / scale)
    q = torch.clamp(
        q,
        -qmax - 1,
        qmax
    )

    return q * scale
```

搜索：

```python
def search_best_threshold(
    x,
    candidates
):
    best_threshold = None
    best_mse = float("inf")

    for threshold in candidates:

        x_hat = fake_quant_symmetric(
            x,
            threshold
        )

        mse = torch.mean(
            (x - x_hat) ** 2
        ).item()

        if mse < best_mse:
            best_mse = mse
            best_threshold = threshold

    return best_threshold, best_mse
```

## 26.6 Histogram / KL Observer

Histogram：

```python
hist = torch.histc(
    x.float(),
    bins=2048,
    min=min_val,
    max=max_val
)
```

KL divergence：

$$
D_{KL}(P\|Q)
=
\sum_i
P(i)\log\frac{P(i)}{Q(i)}
$$

通过搜索 threshold，使原始 distribution 与量化后重构 distribution 的 KL divergence 尽可能小。

## 26.7 Static vs Dynamic Quantization

Static：

```text
Calibration
↓
提前确定 scale
↓
Runtime 固定使用
```

优点：

```text
runtime 简单
易整数 kernel
易 fusion
```

缺点：

```text
依赖 calibration representative
distribution shift 时容易出问题
```

Dynamic：

```text
Runtime Tensor
↓
现场统计
↓
动态算 scale
↓
quantize
```

例如：

$$
s_x=\frac{\max|X|}{127}
$$

它用 runtime 统计开销换取更强的分布适应能力。

## 26.8 Per-Token Dynamic Quantization

对 activation：

```text
[B,T,C]
```

每个 token 使用：

$$
s_{b,t}
=
\frac{
\max_c |X_{b,t,c}|
}{
127
}
$$

这样不同 token 的动态范围不会互相影响。

## 26.9 Calibration Dataset 选择

LLM 要覆盖：

```text
短 prompt
长 prompt
不同 domain
不同 token pattern
```

VLM 要覆盖：

```text
不同图像
不同文本长度
不同 image-text combination
```

VLA 要覆盖：

```text
不同 observation
不同 instruction
不同 robot state
不同 action phase
不同 diffusion timestep
```

样本数量不是无限越多越好，更重要的是覆盖真实 distribution。

## 26.10 Observer 选择的实用逻辑

```text
Activation 很稳定
      ↓
MinMax

少数明显 outlier
      ↓
Percentile / MSE

Distribution tail 复杂
      ↓
Histogram / KL

Runtime 变化很大
      ↓
Dynamic / Per-token

某些 channel 极端异常
      ↓
SmoothQuant / Rotation
```

## 26.11 Activation Quantization 排障升级路线

```text
Level 1
换 Observer / Scale
MinMax → Percentile → MSE

Level 2
换 Granularity
per-tensor → per-token / per-channel

Level 3
改 Distribution
SmoothQuant / QuaRot / DuQuant

Level 4
Mixed Precision
FP16 fallback
```

## 26.12 核心结论

1. Calibration 是采集数据，Observer 是根据数据决定量化参数。
2. MinMax 最大问题是过度保护 outlier。
3. Clipping threshold 是 clipping error 与 rounding error 的权衡。
4. Percentile / MSE / KL 都是在寻找更合适的 quantization range。
5. Static 依赖 calibration representative，Dynamic 用 runtime 开销换适应性。
6. 好的 calibration dataset 往往比单纯增加样本数量更重要。


---

# 27. Quantization Granularity 系统比较

**系统讲解**：granularity 决定 scale 的自由度与 runtime 元数据开销。per-tensor 只存一个 scale，硬件和内存访问最简单；per-channel 能适应不同输出 channel 的范围；per-group 是精度、scale 开销和 packed weight kernel 之间的折中。选择粒度时要同时考虑误差和硬件是否有对应的 native kernel，否则可能因为 dequant/fallback 抵消精度收益。

**公式**：若 weight 被按 group size (G) 划分，第 (g) 个 group 的 scale 为：

$$
s_g=\frac{\max_{j\in g}|W_j|}{Q_{max}}
$$

反量化为：

$$
\hat W_j=s_{g(j)}Q_j
$$

**代码实现**：

```python
def per_group_scale(w, group_size=128, qmax=7):
    # 假设最后一维是 input channel，长度可被 group_size 整除
    groups = w.reshape(*w.shape[:-1], -1, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    return scale
```

所谓 granularity，本质上回答：

> 多少个数共享一套 scale / zero-point？

它直接决定：

- 精度
- metadata 数量
- kernel 复杂度
- requant 方式
- 硬件友好度

## 27.1 Per-Tensor

假设：

$$
W\in\mathbb{R}^{4\times4}
$$

整个 tensor 共用一个 scale：

$$
s_W
=
\frac{\max |W|}{127}
$$

优点：

```text
实现简单
metadata 最少
kernel 友好
```

缺点：

```text
容易被局部 outlier 撑大
小值量化分辨率差
```

## 27.2 Per-Channel Weight Quantization

PyTorch Linear：

```text
weight.shape = [out_features, in_features]
```

通常每个 output channel 单独一个 scale：

$$
s_j
=
\frac{
\max_i|W_{j,i}|
}{
127
}
$$

代码：

```python
def quantize_weight_per_channel(
    W,
    qmax=127
):
    scale = (
        W.abs()
        .max(dim=1)
        .values
        / qmax
    ).clamp(min=1e-8)

    q = torch.round(
        W /
        scale.unsqueeze(1)
    )

    q = torch.clamp(
        q,
        -128,
        127
    )

    return (
        q.to(torch.int8),
        scale
    )
```

这里：

```text
scale.shape = [out_features]
```

## 27.3 Quant Axis

Quant axis 表示：

> scale 对应 tensor 的哪个维度。

PyTorch Linear weight：

```text
[out_features, in_features]
```

如果每个 output channel 一个 scale：

```text
axis = 0
```

## 27.4 为什么 Weight 特别适合 Per-Channel

因为 Weight 是静态的，部署前即可离线计算 scale。

更重要的是：

> Weight per-channel scale 位于 output 维度，不在 GEMM reduction 维度内。

所以硬件容易在 accumulation 后处理。

## 27.5 Activation Per-Channel 为什么难

假设：

$$
X_i \approx s_{x,i}X_{q,i}
$$

$$
W_{j,i}\approx s_{w,j}W_{q,j,i}
$$

则：

$$
Y_j
\approx
\sum_i
s_{x,i}s_{w,j}
X_{q,i}W_{q,j,i}
$$

由于：

$$
s_{x,i}
$$

沿 reduction 维度 $i$ 变化，无法简单写成：

$$
scale \times INT32\ accumulator
$$

因此 activation per-channel 对纯整数 GEMM 很不友好。

## 27.6 Per-Group Quantization

例如：

```text
group_size = 128
```

表示每连续 128 个 weight 共用一个 scale。

如果：

```text
W.shape = [4096,4096]
```

沿 input 维度分组：

$$
4096/128=32
$$

则：

```text
scale.shape ≈ [4096,32]
```

特别适合 INT4 Weight。

代码：

```python
def quantize_int4_groupwise(
    W,
    group_size=128
):
    out_dim, in_dim = W.shape

    assert in_dim % group_size == 0

    num_groups = (
        in_dim // group_size
    )

    groups = W.view(
        out_dim,
        num_groups,
        group_size
    )

    scale = (
        groups.abs()
        .amax(dim=-1)
        / 7.0
    ).clamp(min=1e-8)

    q = torch.round(
        groups /
        scale.unsqueeze(-1)
    )

    q = torch.clamp(
        q,
        -8,
        7
    )

    return q, scale
```

## 27.7 Group Size 的 Trade-off

group_size 大：

```text
scale 少
metadata 少
kernel 简单
精度较低
```

group_size 小：

```text
scale 多
metadata 多
runtime bookkeeping 多
精度通常更高
```

因此它是：

```text
Accuracy
↕
Runtime Complexity
```

的 trade-off。

## 27.8 Per-Token Quantization

Activation：

```text
X.shape = [B,T,C]
```

每个 token hidden vector `[C]` 共用一个 scale：

$$
s_{b,t}
=
\frac{
\max_c|X_{b,t,c}|
}{
127
}
$$

代码：

```python
def quantize_activation_per_token(
    X,
    qmax=127
):
    scale = (
        X.abs()
        .amax(
            dim=-1,
            keepdim=True
        )
        / qmax
    ).clamp(min=1e-8)

    q = torch.round(
        X / scale
    )

    q = torch.clamp(
        q,
        -128,
        127
    )

    return (
        q.to(torch.int8),
        scale
    )
```

这里：

```text
scale.shape = [B,T,1]
```

## 27.9 为什么 Per-Token 对 LLM 友好

对于 token $t$：

$$
X_t=s_tX_{q,t}
$$

Weight per-channel：

$$
W_j=s_{w,j}W_{q,j}
$$

则：

$$
Y_{t,j}
=
s_ts_{w,j}
\sum_i
X_{q,t,i}W_{q,j,i}
$$

由于：

$$
s_t
$$

在 reduction 维度上不变，因此仍然可以：

```text
INT8 dot product
↓
INT32 accumulator
↓
最后乘 s_t * s_wj
```

这就是 per-token 比 activation per-channel 更适合硬件的根本原因。

## 27.10 常见 Scale Shape

假设：

```text
Weight W: [out,in]
Activation X: [B,T,in]
```

### Weight Per-Tensor

```text
scale.shape = []
```

### Weight Per-Channel

```text
scale.shape = [out]
```

### Weight Per-Group

```text
scale.shape = [out, in/group_size]
```

### Activation Per-Token

```text
scale.shape = [B,T,1]
```

## 27.11 Scale Broadcasting

Weight per-channel：

```text
W.shape     = [out,in]
scale.shape = [out]
```

量化时：

```python
W / scale.unsqueeze(1)
```

Per-group：

```text
groups.shape = [out,num_groups,group_size]
scale.shape  = [out,num_groups]
```

需要：

```python
scale.unsqueeze(-1)
```

## 27.12 从 GEMM 看 Scale

### A Per-Tensor + W Per-Tensor

$$
A=s_aA_q
$$

$$
W=s_wW_q
$$

所以：

$$
Y=s_as_w(A_qW_q)
$$

### A Per-Tensor + W Per-Channel

$$
Y_j
=
s_as_{w,j}
\sum_i
A_{q,i}W_{q,j,i}
$$

Accumulator scale 为：

```text
[out]
```

### A Per-Token + W Per-Channel

$$
Y_{t,j}
=
s_{a,t}s_{w,j}
Acc_{t,j}
$$

逻辑 scale 是：

$$
S_{token}\otimes S_{weight}
$$

### A Per-Channel + W Per-Channel

$$
Y_j
=
\sum_i
s_{a,i}s_{w,j}
A_{q,i}W_{q,j,i}
$$

由于 activation scale 位于 reduction 维度内，对整数 GEMM 不友好。

## 27.13 为什么 W4A16 更能接受 Per-Group

W4A16：

```text
Activation FP16
Weight INT4 per-group
```

kernel 可以：

```text
load packed INT4
↓
unpack
↓
apply group scale
↓
与 FP16 activation 计算
```

因此 per-group weight 的 runtime 成本可在 tile 内局部处理。

## 27.14 Granularity 与 Observer 是两个独立维度

Observer 回答：

> scale 的值怎么定？

Granularity 回答：

> 多少数据共享这个 scale？

例如都合法：

```text
Per-Tensor + MinMax
Per-Tensor + Percentile
Per-Token + MinMax
Per-Channel + MSE
```

## 27.15 完整 Quantization Configuration

一个量化方案至少可以描述成：

```text
bit-width
+
granularity
+
observer
+
symmetric/asymmetric
+
static/dynamic
```

例如：

```text
Activation:
INT8
per-token
dynamic MinMax
symmetric

Weight:
INT4
per-group 128
asymmetric
```

## 27.16 从编译器 IR 看 Quantized Tensor

一个 quantized tensor 往往至少需要描述：

```text
dtype
scale
zero_point
axis
group_size
symmetric/asymmetric
```

整数 tensor 本身没有完整语义，必须结合这些 metadata 才能还原真实数值。

## 27.17 Granularity 配错的后果

例如 Weight 本应：

```text
per-output-channel
axis = 0
```

编译器却按：

```text
axis = 1
```

解释。

即使整数 weight 和 scale 数值都没变，每个 scale 对应错 channel，输出也会严重错误。

## 27.18 核心结论

1. Granularity 决定“多少数共享一个 scale”。
2. Weight per-channel 很自然，因为 scale 位于 output 维度，不破坏 reduction。
3. Activation per-channel 难，是因为 scale 位于 GEMM reduction 维度内部。
4. Per-token 对 LLM activation 友好，因为一个 token 内的 scale 在 reduction 维度上保持不变。
5. Granularity、Observer、bit-width、symmetric/asymmetric 是彼此独立但组合使用的设计维度。


---

# 28. Q/DQ Graph 与 Quantization Compiler IR

**系统讲解**：Q/DQ 图把量化语义显式写进计算图。`QuantizeLinear` 表示从浮点到整数，`DequantizeLinear` 表示从整数恢复到浮点；编译器可以识别连续的 Q/DQ 模式，将其融合成 quantized operator。真正的部署结果取决于 Q/DQ 的 scale、zero-point、axis、dtype 和相邻算子是否满足 legalization 规则。

**公式**：非对称 Q/DQ 通常为：

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s)+z,q_{min},q_{max}\right)
$$

$$
x_{dq}=s(q-z)
$$

**代码实现**：

```python
def fake_qdq(x, scale, zero_point=0, qmin=-128, qmax=127):
    q = (x / scale + zero_point).round().clamp(qmin, qmax)
    return (q - zero_point) * scale
```

工程排查时要分别检查：图中是否存在 Q/DQ、compiler 是否消除了冗余 Q/DQ、是否发生 unsupported fallback、最后 kernel 使用的 scale/axis 是否与图中一致。

这一部分开始进入“量化算法结果如何变成编译器和 NPU 可执行图”。

## 28.1 QuantizeLinear

一般形式：

$$
q=
\operatorname{round}
\left(
\frac{x}{s}
\right)
+z
$$

再做 clipping：

$$
q=
clip(q,q_{min},q_{max})
$$

因此：

```text
FP Tensor
↓
QuantizeLinear
↓
INT Tensor
```

量化语义至少包括：

```text
scale
zero_point
axis
dtype
```

## 28.2 DequantizeLinear

$$
\hat{x}
=
s(q-z)
$$

它恢复的是量化后的近似浮点值：

$$
\hat{x}\approx x
$$

而不是原始精确值。

## 28.3 QDQ Graph 的意义

典型图：

```text
FP Input
↓
Q
↓
INT8
↓
DQ
↓
FP Approx
↓
MatMul
```

Weight 也可以有相同 Q/DQ。

QDQ Graph 的目的不是要求 runtime 真的执行：

```text
INT8 → FP → FP MatMul
```

而是：

> 显式描述 tensor 的量化语义，供 compiler 后续识别并 lower 成真正的 INT8/INT4 kernel。

## 28.4 Fake Quant 与 QDQ

FakeQuant 本质近似：

```text
FP
↓
Quantize
↓
Dequantize
↓
FP
```

Tensor dtype 可能仍然是 FP，但数值已经受量化格点限制。

因此 FakeQuant 与 Q+D Q 在概念上非常接近。

## 28.5 Per-Channel Weight QDQ

PyTorch Linear Weight：

```text
[out_features, in_features]
```

如果 Weight per-channel：

```text
scale.shape = [out_features]
axis = 0
```

则：

$$
W_{q,j,i}
=
round
\left(
\frac{W_{j,i}}{s_j}
\right)
$$

Compiler 必须知道 axis，否则 scale 语义会错。

## 28.6 Compiler 识别 QDQ Pattern

例如：

```text
DQ(X_q)
  ↓
MatMul
  ↑
DQ(W_q)
```

如果 compiler 知道：

```text
X_q = INT8
W_q = INT8
```

则可识别为：

```text
QuantizedMatMul
```

并 lower 成：

```text
INT8 GEMM
```

## 28.7 为什么要消掉 DQ

如果真实执行：

```text
INT8
↓
DQ
↓
FP16
↓
FP16 GEMM
```

低比特算力收益会大幅损失。

因此 compiler 会把：

```text
DQ + MatMul + Q
```

融合成：

```text
INT8 GEMM + Requant
```

## 28.8 数学对应

输入：

$$
X=s_xX_q
$$

Weight：

$$
W=s_wW_q
$$

则：

$$
Y
\approx
s_xs_w
(X_qW_q)
$$

令：

$$
Acc=X_qW_q
$$

则：

$$
Y=s_{acc}Acc
$$

其中：

$$
s_{acc}=s_xs_w
$$

如果输出：

$$
Y=s_yY_q
$$

则：

$$
Y_q
=
round
\left(
Acc
\frac{s_xs_w}{s_y}
\right)
$$

这就是 Requantization。

## 28.9 Quantization Lowering

可以理解为：

```text
High-Level MatMul
↓
QuantizedMatMul
↓
Backend INT8 GEMM
↓
NPU Matrix Instruction
```

这个逐级变换过程就是 lowering。

## 28.10 Quant IR 需要的信息

一个 QuantizedMatMul 逻辑上至少需要：

```text
Input dtype
Input scale
Input zero-point

Weight dtype
Weight scale
Weight zero-point
Weight axis

Accumulator dtype

Output scale
Output zero-point
```

## 28.11 Per-Channel Weight 的 Requant

如果：

$$
s_w=s_{w,j}
$$

则：

$$
s_{acc,j}
=
s_xs_{w,j}
$$

输出：

$$
Y_{q,j}
=
round
\left(
Acc_j
\frac{s_xs_{w,j}}{s_y}
\right)
$$

所以 requant multiplier 往往也是 per-output-channel。

## 28.12 Integer Multiplier + Shift

硬件通常不直接做浮点：

$$
Acc \times
\frac{s_xs_w}{s_y}
$$

而会近似为：

$$
Acc \times M \gg n
$$

即：

```text
integer multiplier
+
right shift
+
round
+
saturate
```

例如：

$$
0.3
\approx
\frac{77}{256}
$$

可以实现为：

```text
Acc * 77
↓
right shift 8
```

## 28.13 Scale Propagation

假设：

```text
Linear1
↓
ReLU
↓
Linear2
```

如果 Linear1 输出 scale 与 Linear2 输入 scale 可以共享，就不必插入额外 requant。

因此 compiler 会尽量：

```text
传播 / 统一 scale
↓
减少 requant
```

这叫 Scale Propagation。

## 28.14 Scale Folding

某些显式 scale 可以被吸收入：

```text
Weight
LayerNorm parameter
Linear parameter
```

从而消除 runtime scale op。

SmoothQuant 本身就是一个典型 reparameterization / scale folding 思路。

## 28.15 Q/DQ Elimination

如果：

```text
INT8(scale=0.02)
↓
DQ
↓
Q(scale=0.02)
```

那么 Q/DQ 可以直接消掉。

如果 scale 不同：

```text
0.02
↓
0.05
```

则可替换成：

```text
Requant
```

即：

$$
q_2
=
round
\left(
q_1
\frac{0.02}{0.05}
\right)
$$

## 28.16 Residual Add 的 Scale Alignment

假设：

$$
A=s_AA_q
$$

$$
B=s_BB_q
$$

希望：

$$
Y=s_YY_q
$$

则：

$$
Y_q
=
round
\left(
A_q\frac{s_A}{s_Y}
+
B_q\frac{s_B}{s_Y}
\right)
$$

因此 QuantizedAdd 需要对两个 branch 分别做 scale alignment。

Residual 是 Quant Compiler 的典型难点。

## 28.17 LayerNorm / Softmax 常保 FP16

因为包含：

```text
mean
variance
sqrt
exp
division
```

整数化复杂，所以 graph 常出现：

```text
INT8 Linear
↓
DQ
↓
FP16 LayerNorm / Softmax
↓
Q
↓
INT8 Linear
```

这是真实 Mixed Precision Runtime Cost。

## 28.18 Quant IR 与普通 IR

普通 IR：

```text
MatMul
Add
ReLU
```

Quant IR：

```text
QuantizedMatMul
QuantizedAdd
Requantize
Quantize
Dequantize
```

而 tensor 还带：

```text
dtype
scale
zero_point
axis
```

因此 Quant IR 是：

> 带数值解释规则的计算图。

## 28.19 典型 Quant Compiler Pipeline

```text
FP Graph
↓
Quantization Annotation
↓
Insert Q/DQ
↓
QDQ Pattern Recognition
↓
Scale Propagation
↓
QDQ Elimination
↓
Requant Insertion
↓
Operator Fusion
↓
Backend Lowering
↓
INT8 / INT4 Kernel
```

## 28.20 Fake Quant Graph vs Integer Graph

FakeQuant Graph：

```text
FP Tensor
↓
模拟 Q/DQ
↓
FP Kernel
```

用于：

```text
精度验证
QAT
reference
```

Integer Graph：

```text
INT Tensor
↓
INT Kernel
↓
INT Accumulator
↓
Requant
```

用于真实部署。

因此：

> FakeQuant 正常，不代表 Integer Graph 一定正确。

## 28.21 Compiler Quantization Debug Checklist

如果：

```text
FakeQuant 正常
NPU 错
```

优先检查：

```text
Q/DQ 参数
quant axis
per-channel scale 顺序
QDQ elimination
requant ratio
multiplier / shift
residual scale alignment
fusion
layout
rounding / saturation
```

## 28.22 整体链路

```text
Calibration
↓
Observer
↓
Scale / Zero-point
↓
Granularity
↓
QDQ Graph
↓
Quant IR
↓
Scale Propagation
↓
Requant
↓
Fusion
↓
Backend Lowering
↓
INT8 / INT4 Kernel
↓
NPU
```

## 28.23 核心结论

1. Q/DQ Graph 是量化语义表示，不等于 runtime 必须执行 Q→DQ。
2. Compiler 会把 QDQ pattern lower 成 INT8/INT4 kernel。
3. Requant 是不同整数 scale 域之间的转换。
4. Scale Propagation / QDQ Elimination 用来减少 runtime 转换。
5. Residual Add 是 scale alignment 的典型难点。
6. FakeQuant 正常、NPU 异常时，应重点查 Quant IR、requant、layout、fusion 和 kernel。


---

# 29. INT8 GEMM 在 NPU 上的真实执行

**系统讲解**：INT8 GEMM 的理论计算量下降并不自动转化为端到端加速。NPU 实际执行还包含 weight/input 搬运、tile 裁剪、scale/requant、layout transform、DMA 和同步。特别是 decode 的 (M\approx1) 场景，矩阵阵列利用率可能很低，性能瓶颈常常从计算转为权重带宽。

**公式**：

$$
C_{M\times N}=A_{M\times K}B_{K\times N},
\qquad
Ops\approx 2MKN
$$

**代码实现**：kernel 伪代码需要同时体现累加和 requant：

```cpp
for (tile_m, tile_n, tile_k) {
    int32_t acc = dot_int8(a_tile, w_tile); // INT8×INT8→INT32
    float y = acc * input_scale * weight_scale / output_scale;
    out_q[tile_m][tile_n] = saturate_int8(round(y));
}
```

## 29.1 Roofline：Memory-bound vs Compute-bound 的定量分析（系统补充）

**系统讲解**：Roofline 用算术强度把性能问题分成计算受限和带宽受限。对同一模型，prefill 的 (M=B\times T) 较大，通常更接近 GEMM/compute-bound；decode 的 (M\approx1)，往往更接近 GEMV/memory-bound。实际判断必须使用 profiler 的带宽和阵列利用率，而不能只看理论 TOPS。

**公式**：

$$
AI=\frac{Ops}{Bytes}
$$

$$
P_{attainable}=\min(P_{peak}, BW\times AI)
$$

**代码实现**：

```python
def roofline_bound(ops, bytes_moved, peak_ops, peak_bandwidth):
    arithmetic_intensity = ops / max(bytes_moved, 1)
    memory_roof = peak_bandwidth * arithmetic_intensity
    return min(peak_ops, memory_roof), arithmetic_intensity

这一部分把量化和硬件执行真正连起来：Tile、SRAM、MAC Array、DDR、带宽、Data Reuse、Fusion、Layout。

## 29.1 GEMM 基础

\[
C=AB
\]

其中：

\[
A\in\mathbb{R}^{M\times K}
\]

\[
B\in\mathbb{R}^{K\times N}
\]

输出：

\[
C\in\mathbb{R}^{M\times N}
\]

每个元素：

\[
C_{m,n}
=
\sum_{k=0}^{K-1}
A_{m,k}B_{k,n}
\]

INT8 GEMM 中：

```text
A_q = INT8
B_q = INT8
Accumulator = INT32
```

## 29.2 为什么需要 Tiling

大矩阵无法整体放进有限的片上 SRAM，因此要切成小块：

```text
A_tile [tile_M, tile_K]
B_tile [tile_K, tile_N]
↓
C_tile [tile_M, tile_N]
```

沿 K 分块：

\[
C_{tile}
=
A_0B_0
+
A_1B_1
+
\cdots
\]

因此 INT32 accumulator 要保存 partial sum 直到 K 维全部完成。

## 29.3 NPU 存储层级

可以抽象为：

```text
DDR / HBM
  ↓
Global / Shared SRAM
  ↓
Local Buffer / Scratchpad
  ↓
Register / MAC Array
```

越靠近计算单元：

```text
容量越小
速度越快
能耗越低
```

优化核心是：

> 尽可能提高片上数据复用，减少高层存储访问。

## 29.4 量化为什么可能提升性能

FP16：

```text
2 bytes / element
```

INT8：

```text
1 byte / element
```

INT4：

```text
0.5 byte / element
```

在相同 SRAM 容量下，低 bit 可以：

```text
放更大 tile
保存更多并行数据
更容易 double buffer
降低 DDR bandwidth
```

因此量化收益不只是“整数 MAC 更快”。

## 29.5 MAC Array

MAC 单元执行：

\[
a\times b+acc
\]

NPU 通常有大量并行 MAC。

阵列可能偏好固定 tile 形状，例如：

```text
64 × 64
32 × 32
```

因此矩阵 shape 是否与硬件 tile 对齐会显著影响利用率。

## 29.6 Tail Tile

如果硬件最适合：

```text
N multiple of 64
```

但实际：

```text
N = 4100
```

则最后剩余：

```text
4 columns
```

会形成尾块。

大量 MAC 单元空闲，导致利用率下降。

## 29.7 Utilization

如果 MAC array 是：

```text
64 × 64
```

但 tile 实际只有：

```text
64 × 10
```

有效利用率约：

\[
\frac{64\times10}
{64\times64}
=
15.6\%
\]

因此 bit 更低不代表阵列利用率一定更高。

## 29.8 Batch Size 的影响

LLM Decode：

```text
batch = 1
```

常见 GEMM：

\[
[1,K]\times[K,N]
\]

更接近 GEMV。

M 维很小，MAC array 很难吃满，因此常常偏：

```text
memory-bound
```

LLM Prefill：

```text
一次处理多个 token
```

例如：

\[
M=512
\]

矩阵乘更大，阵列更容易吃满，更可能偏：

```text
compute-bound
```

## 29.9 Arithmetic Intensity

定义：

\[
AI
=
\frac{
Operations
}{
Bytes\ Moved
}
\]

AI 低：

```text
数据搬运多
计算少
→ memory-bound
```

AI 高：

```text
数据复用强
计算多
→ compute-bound
```

INT4 减少 Weight Bytes，因此可显著提高 AI，尤其对 memory-bound workload 有帮助。

## 29.10 为什么 4× Weight Compression 不等于 4× Speedup

因为还有：

```text
unpack
dequant
scale load
activation load
output write
kernel launch
attention
KV cache
```

以及实际 bandwidth / kernel efficiency 的限制。

因此：

\[
4\times Compression
\neq
4\times Speedup
\]

## 29.11 Data Reuse

例如：

```text
A_tile × B0
A_tile × B1
A_tile × B2
```

如果 A_tile 留在 SRAM，只需搬一次，就能提高 reuse。

GEMM 优化本质上就是：

> 在有限 SRAM 中最大化 Input / Weight / Output 的数据复用。

## 29.12 常见 Dataflow

### Weight Stationary

```text
Weight 留在计算阵列附近
Activation 流过去
```

### Output Stationary

```text
Partial Sum 留在本地
不断累加
```

### Input Stationary

```text
Activation 尽量留在本地复用
```

不同 NPU 会使用不同 dataflow 或其组合。

## 29.13 INT32 Accumulator 的 SRAM 成本

即使 Input / Weight 是 INT8：

```text
Accumulator = INT32
```

例如 output tile：

```text
128 × 128
```

INT32 accumulator 占：

\[
128\times128\times4
=
65536\ bytes
\]

约 64 KB。

因此 Tiling 时必须同时考虑：

```text
A tile
B tile
C accumulator
Scale
Bias
Double buffer
```

## 29.14 Tile Search 是约束优化

要求：

\[
Memory(A_{tile})
+
Memory(B_{tile})
+
Memory(C_{acc})
+\cdots
\le SRAM
\]

Compiler / kernel 需要选择：

```text
tile_M
tile_N
tile_K
```

以平衡：

```text
reuse
utilization
SRAM capacity
DMA cost
```

## 29.15 Double Buffer

普通：

```text
load
↓
compute
↓
load next
```

会让 compute 等待 memory。

Double Buffer：

```text
Buffer A:
当前 tile compute

Buffer B:
同时 load next tile
```

这样可以重叠：

```text
DMA
+
Compute
```

低 bit tensor 更小，因此更容易在固定 SRAM 中实现双缓冲。

## 29.16 GEMM Epilogue

INT8 GEMM：

```text
INT8 A
×
INT8 W
↓
INT32 Acc
↓
Bias
↓
Requant
↓
INT8 Output
```

理想情况下：

```text
Bias
ReLU
Requant
```

直接 fuse 到 GEMM epilogue。

## 29.17 为什么 Fusion 重要

不 fuse：

```text
GEMM
↓
写 INT32 tensor
↓
读回来
↓
Requant kernel
```

fuse：

```text
GEMM
↓
Bias / ReLU / Requant
↓
直接输出 INT8
```

主要收益往往来自减少中间 tensor 搬运。

## 29.18 Mixed Precision 为什么会破坏 Fusion

例如：

```text
INT8 GEMM
↓
FP16 LayerNorm
↓
INT8 GEMM
```

通常需要：

```text
DQ
FP16 kernel
Q
```

所以 FP16 fallback 成本不只是单层计算，还包含：

```text
Q/DQ
memory movement
fusion break
```

## 29.19 Weight Layout

数学上 Weight：

```text
[out,in]
```

硬件可能要求 blocked layout：

```text
[out/32, in/32, 32, 32]
```

以匹配 MAC array 和 DMA。

因此 compiler 会做：

```text
Weight Packing
Layout Transform
```

## 29.20 INT4 Layout

INT4 还需要 bit packing：

```text
2 × INT4 → 1 byte
8 × INT4 → int32 container
```

同时可能有：

```text
group scale
zero point
```

因此实际 kernel 读取的是：

```text
packed weight tiles
+
scale tiles
+
zero-point tiles
```

## 29.21 GPTQ / AWQ Checkpoint 为什么不能直接适配所有 NPU

Checkpoint packing 通常针对特定 GPU / runtime kernel。

NPU 可能要求不同：

```text
blocked packing format
group layout
scale layout
```

所以部署时通常需要：

```text
offline repack
```

## 29.22 Linear 到 NPU Instruction 的链路

```text
nn.Linear
↓
MatMul + Bias
↓
QuantizedMatMul
↓
Choose Tile
↓
Choose Layout
↓
Pack Weight
↓
DMA Schedule
↓
NPU GEMM Instruction
↓
Epilogue
```

## 29.23 量化后反而不快的常见原因

### Kernel 不支持目标 bit

```text
W4
↓
dequant FP16
↓
FP16 GEMM
```

### 模型太小

```text
Q/DQ / launch overhead
```

占比过高。

### Shape 不友好

```text
tail tile
padding
```

导致利用率低。

### Mixed Precision 过碎

大量：

```text
Q/DQ
layout conversion
```

### 原本就不是 memory-bound

Weight 压缩对 compute-bound workload 收益有限。

## 29.24 性能排障框架

```text
目标 bit kernel 真支持吗？
↓
实际是 INT8/INT4 GEMM 吗？
↓
是否 fallback FP16？
↓
Shape 能吃满阵列吗？
↓
Tile 是否合适？
↓
Bandwidth 是瓶颈吗？
↓
Q/DQ / Requant / Layout 开销大吗？
↓
Fusion 是否被破坏？
```

## 29.25 Accuracy Debug vs Performance Debug

Accuracy Debug：

```text
FP vs Quant
Cosine
SQNR
First Bad Layer
Observer
Scale
```

Performance Debug：

```text
Kernel
Tile
Bandwidth
Utilization
Fusion
Layout
```

两类问题要分开分析。

## 29.26 完整链路

```text
FP Model
↓
Calibration
↓
Scale / Quant Policy
↓
QDQ Graph
↓
Quant IR
↓
INT8 / INT4 Kernel
↓
Tile
↓
SRAM
↓
MAC Array
↓
Requant / Epilogue
↓
Output
```

性能由：

```text
Compute
+
Memory
+
Data Reuse
+
Tile Shape
+
Fusion
+
Layout
```

共同决定。

## 29.27 核心结论

1. Tile 是为了让大矩阵分块进入有限 SRAM。
2. INT8/INT4 性能收益很大一部分来自更少的数据搬运。
3. INT32 accumulator 也会显著占用片上存储。
4. Batch1 LLM Decode 常偏 memory-bound，因此 Weight-only INT4 很有价值。
5. Fusion 的主要收益常常是减少中间 Tensor 读写。
6. 量化后不快，要查 kernel、tiling、layout、bandwidth 和 utilization，而不能只看 bit-width。


---

# 30. Roofline：Memory-bound vs Compute-bound 的定量分析

这一部分用定量方法判断某个 workload 为什么“降 bit 有用”或“几乎没用”。

## 30.1 Roofline 核心公式

定义 Arithmetic Intensity：

\[
AI
=
\frac{\text{Operations}}
{\text{Bytes Moved}}
\]

芯片理论可达性能：

\[
P
=
\min
\left(
P_{peak},
BW\times AI
\right)
\]

其中：

- \(P_{peak}\)：峰值计算吞吐
- \(BW\)：内存带宽
- \(AI\)：Arithmetic Intensity

## 30.2 Memory-bound

如果：

\[
BW\times AI
<
P_{peak}
\]

则性能主要受带宽限制：

\[
P\approx BW\times AI
\]

此时更有效的优化是：

```text
减少 Bytes Moved
提高 Data Reuse
压缩 Weight / Activation
```

## 30.3 Compute-bound

如果：

\[
BW\times AI
>
P_{peak}
\]

则性能主要受计算吞吐限制：

\[
P\approx P_{peak}
\]

此时更应该关注：

```text
MAC throughput
array utilization
tiling
kernel efficiency
```

## 30.4 Ridge Point

分界点：

\[
AI_{ridge}
=
\frac{P_{peak}}{BW}
\]

例如：

\[
P_{peak}=100\ TOPS
\]

\[
BW=500\ GB/s
\]

则：

\[
AI_{ridge}\approx200\ OPS/Byte
\]

所以：

```text
AI < 200  → 更偏 memory-bound
AI > 200  → 更偏 compute-bound
```

## 30.5 GEMM 的 Operations

对于：

\[
[M,K]\times[K,N]
\]

若乘法和加法各算一个 op：

\[
Ops
\approx
2MKN
\]

## 30.6 GEMM 的数据量近似

设 element byte 数分别为：

\[
b_A,b_B,b_C
\]

则：

\[
Bytes
\approx
MKb_A
+
KNb_B
+
MNb_C
\]

因此：

\[
AI
\approx
\frac{
2MKN
}{
MKb_A+KNb_B+MNb_C
}
\]

## 30.7 LLM Decode 示例

设：

\[
M=1,\quad
K=N=4096
\]

则：

\[
Ops
=
2\times1\times4096\times4096
\approx33.6M
\]

FP16 Weight：

\[
4096^2\times2
\approx33.6MB
\]

Activation 和 Output 相比 Weight 很小。

所以：

\[
AI
\approx
\frac{33.6M}{33.6MB}
\approx1\ OP/Byte
\]

如果 Ridge Point 是 200 OP/Byte，则：

```text
1 << 200
```

属于极强 memory-bound。

## 30.8 Weight Bit-width 对 Decode AI 的影响

Batch1 decode 下，Weight traffic 占主导：

\[
AI
\approx
\frac{2}{b_W}
\]

因此：

### FP16 Weight

\[
b_W=2
\]

\[
AI\approx1
\]

### INT8 Weight

\[
b_W=1
\]

\[
AI\approx2
\]

### INT4 Weight

\[
b_W=0.5
\]

\[
AI\approx4
\]

所以 batch1 decode 对 Weight Compression 极其敏感。

## 30.9 LLM Prefill

如果：

\[
M=512
\]

而：

\[
K=N=4096
\]

Ops：

\[
2\times512\times4096\times4096
\]

大约：

\[
17.2B
\]

Weight 仍然只有一份，因此在理想复用下 AI 会大幅提高。

粗略可达到：

```text
数百 OP/Byte
```

因此更可能从 memory-bound 转向 compute-bound。

## 30.10 Prefill vs Decode

同一个 Linear：

```text
Decode:
M = 1
AI ≈ 1
→ memory-bound

Prefill:
M = 512
AI ≫ 1
→ 可能 compute-bound
```

所以：

> 不能只看模型结构判断性能瓶颈，还必须看实际 workload shape。

## 30.11 Weight-only vs Full Integer Quantization

### W4A16

主要收益：

```text
Weight memory ↓
Weight bandwidth ↓
```

非常适合 memory-bound decode。

### W8A8

除了 Weight bandwidth，还可能带来：

```text
Activation bandwidth ↓
INT8 compute throughput ↑
SRAM efficiency ↑
```

因此在 compute-heavy workload 中更有价值。

## 30.12 为什么 W4A16 很流行

LLM decode 常常强 memory-bound。

因此即使 Activation 仍然 FP16，仅压缩 Weight 也能解决主要瓶颈，同时精度风险比 W4A4 更低。

这也是 GPTQ / AWQ 工程上非常常见的重要原因。

## 30.13 理论 AI 与实际 AI

实际还有：

```text
scale
zero-point
KV cache
activation
temporary buffer
layout transform
repeated memory access
```

所以理论 AI 只是近似或上界。

## 30.14 实际 Compute 利用率

真实 kernel 往往只能达到：

\[
\eta P_{peak}
\]

其中：

\[
0<\eta<1
\]

原因包括：

```text
tail tile
small batch
bad layout
pipeline bubble
```

因此更现实：

\[
P
\approx
\min(
\eta P_{peak},
\eta_{bw}BW\times AI
)
\]

## 30.15 Achieved TOPS 与 Achieved Bandwidth

已知：

```text
Ops
Bytes
Latency
```

可算：

\[
Achieved\ Performance
=
\frac{Ops}{Latency}
\]

\[
Achieved\ Bandwidth
=
\frac{Bytes}{Latency}
\]

然后与：

```text
Peak Compute
Peak Bandwidth
```

比较。

## 30.16 Memory-bound 示例

某 Linear：

```text
Ops = 34M
Weight traffic = 8.5MB
Latency = 30 us
```

则：

\[
Achieved\ Compute
\approx1.13\ TOPS
\]

\[
Achieved\ BW
\approx283\ GB/s
\]

若芯片：

```text
Peak Compute = 100 TOPS
Peak BW      = 300 GB/s
```

则：

```text
Compute utilization ≈ 1%
Bandwidth utilization ≈ 94%
```

非常明显是 memory-bound。

## 30.17 Compute-bound 示例

假设：

```text
Ops = 17B
Bytes = 45MB
Latency = 250 us
```

则：

\[
Achieved\ Compute
\approx68\ TOPS
\]

\[
Achieved\ BW
\approx180\ GB/s
\]

若峰值：

```text
100 TOPS
300 GB/s
```

则更偏 compute-bound。

## 30.18 性能 Debug 流程

```text
算 Ops
↓
算 Bytes
↓
算 AI
↓
和 Ridge Point 比
↓
判断理论瓶颈
↓
测 Latency
↓
算 Achieved TOPS / GB/s
↓
确认实际瓶颈
```

如果理论 compute-bound，但 TOPS 利用率很低：

```text
查 tile_M / tile_N / tile_K
tail tile
MAC occupancy
pipeline
```

如果理论 memory-bound，但 bandwidth 只跑到峰值很小比例：

```text
查 DMA
layout
prefetch
double buffer
memory access pattern
```

## 30.19 Roofline 与量化算法的联系

GPTQ / AWQ：

```text
W4A16
→ 主要解决 Weight Bandwidth
→ 特别适合 Decode
```

SmoothQuant：

```text
W8A8
→ 同时压缩 Weight / Activation
→ 还可能利用 INT8 Compute
```

因此不同量化算法对应的性能收益，要结合 workload bottleneck 判断。

## 30.20 VLA 中不同模块的瓶颈可能不同

例如：

```text
Vision Encoder
→ 大矩阵 / spatial compute
→ 可能 compute-heavy

LLM Decode
→ batch1
→ memory-heavy

DiT Action Head
→ 多 timestep 重复计算
→ compute + memory 都重要
```

所以一个 VLA 模型内，不同模块可能适合不同 quantization strategy。

## 30.21 核心结论

1. Roofline 核心公式是 \(P=\min(P_{peak},BW\times AI)\)。
2. Arithmetic Intensity 决定 workload 更偏 memory-bound 还是 compute-bound。
3. Batch1 Decode 的 AI 很低，所以 Weight bit-width 对性能特别重要。
4. Prefill 中 Weight 能跨多个 token 复用，因此 AI 显著提高。
5. 理论 bottleneck 和实际 bottleneck 要通过 achieved TOPS / GB/s 验证。
6. Roofline 最大价值是告诉你下一步该查 compute、memory、tiling 还是 layout。


---

# 31. NPU Performance Profiling 实战框架

**系统讲解**：性能 profiling 应把端到端 latency 拆成 Host preprocessing、H2D/D2H、kernel、同步和 scheduler 等区间。只有把时间归因到具体阶段，才能判断量化收益被算子执行、数据搬运还是系统固定开销吞掉。

**代码实现**：建议每次 benchmark 至少保存一条结构化记录：

```python
record = {
    "batch": batch,
    "seq_len": seq_len,
    "dtype": dtype,
    "latency_ms": latency_ms,
    "h2d_ms": h2d_ms,
    "kernel_ms": kernel_ms,
    "d2h_ms": d2h_ms,
    "bandwidth_gbps": bandwidth_gbps,
    "npu_utilization": npu_utilization,
    "fallback_ops": fallback_ops,
}
json.dump(record, open("profile.json", "w"), indent=2)
```

这一部分目标是：看到 profiler 数据后，能够系统判断真正的性能瓶颈。

## 31.1 第一层：End-to-End Latency

总延迟应先拆成：

```text
Preprocess
Host-to-Device
Model Execution
Device-to-Host
Postprocess
```

只有确认瓶颈落在 Model Execution 后，才值得继续深入 NPU kernel。

## 31.2 第二层：Top Latency Operators

重点看：

```text
Operator latency
Call count
Total time = per-call latency × call count
```

优先抓耗时大头，而不是先优化很小的算子。

特别是 Diffusion / DiT：

```text
小算子 × 多 timestep
```

累计可能变成主要开销。

## 31.3 确认实际 Kernel DType

必须确认真实执行的是：

```text
INT8
INT4
FP16
Fallback
```

因为“模型标称 INT8”不代表“Kernel 就一定是 INT8”。

可能发生：

```text
INT8 Tensor
↓
DQ
↓
FP16 Kernel
```

或：

```text
W4 storage
↓
dequant FP16
↓
FP16 GEMM
```

因此要区分 Storage Precision 和 Compute Precision。

## 31.4 Compute Utilization

\[
Achieved\ Compute
=
\frac{Ops}{Latency}
\]

再算：

\[
U_{compute}
=
\frac{Achieved\ Compute}
{Peak\ Compute}
\]

但 Compute Utilization 低不一定代表 kernel 差，需要结合 Roofline 判断是否 memory-bound。

## 31.5 Bandwidth Utilization

\[
Achieved\ BW
=
\frac{Bytes}{Latency}
\]

\[
U_{BW}
=
\frac{Achieved\ BW}
{Peak\ BW}
\]

例如：

```text
Compute Util = 20%
Bandwidth Util = 91%
```

更像 memory-bound。

## 31.6 一个简单判断矩阵

| Compute Util | BW Util | 可能情况 |
|---:|---:|---|
| 高 | 高 | 接近硬件上限 |
| 低 | 高 | Memory-bound |
| 高 | 低 | Compute-bound 或 reuse 很好 |
| 低 | 低 | 值得重点排查 |

最值得警惕的是：

```text
Compute Low
Bandwidth Low
```

通常可能意味着：

```text
small shape
tail tile
pipeline bubble
DMA waiting
launch overhead
layout conversion
dependency
```

## 31.7 MAC / Array Utilization

如果 workload 理论上偏 compute-bound，但 MAC utilization 很低，需要排查：

```text
M/N/K 太小
tail tile
padding
tile shape
阵列对齐
```

## 31.8 Padding Overhead

如果：

```text
K = 4100
```

为了 tile 对齐 padding 到：

```text
K = 4160
```

则多算出来的部分会造成：

```text
Hardware Ops > Model Theoretical Ops
```

这种额外成本叫 Padding Overhead。

## 31.9 DMA

DMA 用来在不同层级间搬数据，例如：

```text
DDR
↓
SRAM
```

如果：

```text
Compute = 0.3 ms
DMA = 0.8 ms
```

并且不能充分 overlap，则说明 memory movement 是主要瓶颈。

## 31.10 DMA 与 Compute Overlap

理想双缓冲：

```text
Buffer A:
Compute current tile

Buffer B:
Load next tile
```

总时间更接近：

\[
\max(T_{DMA},T_{Compute})
\]

而不是：

\[
T_{DMA}+T_{Compute}
\]

## 31.11 Pipeline Bubble

如果数据或依赖没有及时准备好，会出现：

```text
Compute
[Idle]
Compute
```

中间空档即 pipeline bubble。

常见 profiler stall：

```text
Memory Stall
Dependency Stall
Sync Stall
Pipeline Stall
```

## 31.12 Multi-Core Utilization

多核 NPU 还要看：

```text
用了几个 core
每个 core 是否均衡
跨核通信成本
```

Batch1 场景通常难沿 M 维切分，因此多核扩展可能受限。

## 31.13 Kernel Gap

不要只看 kernel 本身。

例如：

```text
Kernel A = 100 us
Gap      = 80 us
Kernel B = 100 us
```

总时间实际：

```text
280 us
```

Gap 可能来自：

```text
host launch
synchronization
DMA
layout conversion
```

## 31.14 小算子与 Launch Overhead

假设：

```text
kernel compute = 3 us
launch overhead = 5 us
```

优化 kernel 本身收益有限。

这类场景更应该做 Fusion。

## 31.15 Operator Fusion

例如：

```text
GEMM
↓
Bias
↓
ReLU
↓
Requant
```

融合成：

```text
GEMM + Bias + ReLU + Requant
```

可减少：

```text
kernel launch
中间 tensor
memory traffic
```

## 31.16 Layout Conversion

需要单独关注：

```text
transpose
reorder
memcpy
NCHW ↔ NHWC
normal ↔ blocked
```

特别是 Mixed Precision 场景：

```text
INT8 op
↓
FP16 op
↓
INT8 op
```

可能同时引入：

```text
DQ
layout conversion
FP16 compute
layout conversion
Q
```

## 31.17 完整 Profiler 排查顺序

```text
Step 1
End-to-End Latency

Step 2
拆 Model / Preprocess / Transfer / Postprocess

Step 3
找 Top Latency Operators

Step 4
确认实际 Kernel DType

Step 5
算理论 Ops / Bytes / AI

Step 6
判断 Roofline Bottleneck

Step 7
看 Achieved TOPS / BW

Step 8
看 MAC / Core Utilization

Step 9
看 DMA / Stall / Bubble

Step 10
看 Tile / Tail / Padding

Step 11
看 Fusion / Kernel Count

Step 12
看 Layout Conversion / QDQ
```

## 31.18 INT8 Linear 加速不明显示例

如果：

```text
FP16 = 1.0 ms
INT8 = 0.85 ms
```

先确认不是 fallback。

若 Roofline 显示该层 compute-bound，而：

```text
FP16 compute util = 75%
INT8 compute util = 30%
```

则应优先怀疑：

```text
INT8 tiling
array utilization
kernel implementation
```

而不是 bandwidth。

## 31.19 W4A16 示例

如果：

```text
FP16 Decode = 2.0 ms
W4A16       = 1.1 ms
```

且：

```text
Bandwidth Util 高
Compute Util 低
```

说明这是典型 memory-bound，W4 主要通过减少 Weight traffic 获益。

如果 W4 只快 1.2×，但 Weight traffic 理论降 4×，则应排查：

```text
unpack
dequant
scale load
layout conversion
packing format
```

## 31.20 高质量性能分析的输出形式

不要只说“这个模型性能不好”，而应形成证据链：

```text
模型总 latency = ...
Top op = ...
该 op 实际 kernel = ...
理论 AI = ...
理论 bottleneck = ...
Achieved TOPS / BW = ...
MAC util = ...
主要原因 = ...
优化方向 = ...
```

## 31.21 Profiler 指标四层结构

```text
Level 1：Model
End-to-End Latency
Throughput

Level 2：Operator
Latency
Call Count
Kernel Type

Level 3：Hardware
TOPS Utilization
Bandwidth
Core / MAC Utilization
DMA / Stall

Level 4：Compiler / Kernel
Tile
Layout
Fusion
Padding
Requant
Packing
```

从上往下逐层缩小问题范围。

## 31.22 VLA Profiler

VLA 应按模块拆：

```text
Vision Encoder
VLM
Action Head / DiT
```

尤其 DiT：

```text
Single-step latency × Denoising Steps
```

才是完整成本。

## 31.23 两套核心 Debug 框架

Accuracy Debug：

```text
FP
↓
FakeQuant
↓
NPU
↓
First Bad Layer
↓
Scale / Quant / Kernel Correctness
```

Performance Debug：

```text
Latency
↓
Top Operators
↓
Roofline
↓
TOPS / BW
↓
Tile / DMA / Fusion / Layout
```

## 31.24 核心结论

1. Profiler 先看总时间和 Top Operators。
2. 必须确认实际 kernel dtype。
3. TOPS 利用率低不一定差，要结合 Roofline。
4. Compute 与 Bandwidth 都低时，重点查 tiling、stall、DMA、launch、layout。
5. 小算子通常更适合通过 fusion 优化。
6. 高质量性能分析必须给出：理论瓶颈 → 实测指标 → 原因 → 优化方向。


---

# 32. 从 PyTorch 到 NPU Binary：完整量化部署流水线

**系统讲解**：完整流水线至少包含两个闭环。模型闭环保证计算图和权重正确，部署闭环保证目标硬件真的执行了量化 kernel。任意中间产物成功都不能替代最后的 NPU 精度与性能验证。

**代码实现**：当前 Spirit 工程的核心顺序可以简化为：

```python
# ptq.py
quantize_qwen3_vl(args)  # vision/prefill/decode → HMONNX
quantize_dit(args)       # DiT → ONNX → HMONNX

# build.py（概念流程）
compile_hmonnx_to_hmm(output_dir)
run_subgraph_golden_tests(output_dir)

# ptq_dump.py
dump_and_stage_golden(args)
write_cosine_report(output_dir, rows, "all")
```

**公式**：部署验收应同时满足：

\[
\text{ExportOK}\land\text{ConvertOK}\land\text{CompileOK}
\land\text{RuntimeOK}\land\text{AccuracyOK}\land\text{PerformanceOK}
\]

否则只能称为“流程某一步成功”，不能称为模型已部署。

这一部分把此前学习过的 Calibration、Observer、Q/DQ、Quant IR、Requant、Kernel、Tiling、Profiler 串成一条完整工程链路。

## 32.1 总体流程

```text
PyTorch Model
    ↓
Graph Capture / Export
    ↓
ONNX / FX / Exported Graph
    ↓
Graph Rewrite / Canonicalization
    ↓
Calibration
    ↓
Quantization Parameters
    ↓
Insert Q/DQ / Quant Annotation
    ↓
Quantized Graph
    ↓
Compiler IR
    ↓
Quantization Optimization
    ↓
Backend Lowering
    ↓
Kernel Selection
    ↓
Tiling / Memory Scheduling
    ↓
Weight Packing
    ↓
NPU Binary
    ↓
Runtime
    ↓
Profiler / Accuracy Debug
```

解决方案工程中最重要的问题之一是：

> 先判断问题发生在这条链的哪一层。

## 32.2 PyTorch Model

PyTorch 模型处于 Framework Level。

开发者看到的是：

```text
Module
Tensor
Python Forward
```

NPU 编译器不能直接把 Python forward 当成硬件执行计划，因此需要 Graph Capture。

## 32.3 Graph Capture

Python 执行逻辑会被转换成显式计算图，例如：

```text
x
↓
Linear
↓
SiLU
↓
Linear
↓
output
```

进一步可能拆成：

```text
MatMul
↓
Add
↓
SiLU
↓
MatMul
↓
Add
```

## 32.4 ONNX 的位置

ONNX 可以理解为跨框架的通用 Graph IR。

它描述：

```text
Operator
Tensor
Shape
Constant Weight
Graph Connectivity
```

PyTorch Linear 可能导出为：

```text
MatMul + Add
```

或：

```text
Gemm
```

因此：

```text
PyTorch
↓ Export
ONNX
```

本质是 Framework IR 到通用 Graph IR 的转换。

## 32.5 厂商 Internal IR

ONNX 太通用，厂商通常会转换到自己的 Internal IR：

```text
ONNX
↓
Vendor Internal IR
```

同时补充：

```text
shape
layout
dtype
quant metadata
hardware constraints
```

## 32.6 Graph Canonicalization

同一个数学操作可能有多种写法：

```text
MatMul + Add
Gemm
Transpose + MatMul + Add
```

Compiler 通常会统一成 Canonical Linear，避免后续 Pass 处理大量等价形式。

## 32.7 参数 Folding

例如：

```text
Conv
↓
BatchNorm
```

BatchNorm 参数可提前 fold 到 Conv Weight / Bias。

SmoothQuant 中的 LayerNorm/Linear scale folding 也是类似思路。

原则：

> 能离线吸收到参数里的计算，尽量不要留到 runtime。

## 32.8 Calibration 阶段

PTQ 中：

```text
FP Graph
+
Observer
↓
Calibration Data Forward
```

收集：

```text
min/max
histogram
percentile
channel statistics
```

此时通常还不是高效 INT8 kernel。

## 32.9 Calibration 的输出

Calibration 真正产生的是 Quantization Parameters：

```text
tensor_17:
scale = 0.031
zero_point = 0
dtype = int8

fc1.weight:
scale = [...]
axis = 0
dtype = int8
```

可以理解为：

```text
Calibration Data
↓
Observer
↓
Quant Params Database
```

## 32.10 Quantization Policy

Compiler 还需要知道：

```text
哪些 Tensor INT8？
哪些 Weight INT4？
哪些 Op FP16？
```

例如：

```text
fc1       → W8A8
softmax   → FP16
fc2       → W8A8
layernorm → FP16
```

这就是 Mixed Precision Policy。

## 32.11 插入 Q/DQ

Q/DQ 很多并不代表 runtime 一定逐个执行，它们首先是量化语义标记。

## 32.12 Quant IR

例如：

```text
DQ(X_q)
↓
MatMul
↑
DQ(W_q)
↓
Q
```

可以被识别为：

```text
QuantizedMatMul
```

IR 中直接携带：

```text
input_scale
weight_scale
output_scale
weight_axis
acc_dtype
```

Quant IR 是连接模型语义与硬件语义的关键层。

## 32.13 Quantization Optimization

典型优化包括：

```text
QDQ Elimination
Requant Insertion
Scale Propagation
Scale Folding
Mixed Precision Boundary Optimization
```

Residual Add 还需要处理多个 branch 的 Scale Alignment。

## 32.14 Backend Lowering

QuantizedMatMul 仍然不是具体硬件指令。

Backend 要决定：

```text
INT8 GEMM Kernel A
INT8 GEMM Kernel B
W4A16 Kernel
FP16 Fallback
```

这就是 Kernel Selection。

## 32.15 Kernel Selection 的输入

常见考虑因素：

```text
M / N / K
dtype
layout
group_size
per-channel / per-tensor
SRAM capacity
core count
dynamic shape
```

同一个 Linear，在 Prefill 和 Decode 中可能选择完全不同的 kernel。

## 32.16 Tiling

确定 kernel 后，需要选择：

```text
tile_M
tile_N
tile_K
```

满足：

\[
Memory_{tile}\le SRAM
\]

同时追求：

```text
MAC utilization 高
Data reuse 高
DMA cost 低
```

## 32.17 Memory Scheduling

Compiler 还需要决定：

```text
什么时候 load
什么时候 compute
什么时候 store
```

并安排：

```text
DMA
Double Buffer
Prefetch
```

以尽量重叠 Memory 与 Compute。

## 32.18 Buffer Lifetime / Memory Planning

Tensor 用完后，其 SRAM Buffer 可以复用。

因此 compiler 会做：

```text
Buffer Allocation
Lifetime Analysis
Memory Reuse
```

以降低 SRAM 峰值。

## 32.19 Weight Packing

部署前 Weight 往往经历：

```text
Quantize
↓
Reorder
↓
Tile
↓
Pack
```

例如 INT4：

```text
FP16 Weight [out,in]
↓
INT4 Group Quant
↓
Blocked Layout
↓
2 × INT4 / Byte
↓
NPU-specific Format
```

因此最终 binary 内的 Weight 往往已经不像 PyTorch 二维矩阵。

## 32.20 NPU Binary

编译最终可生成包含：

```text
Instructions
Constants
Packed Weights
Memory Layout
Kernel Parameters
DMA Descriptors
```

的 NPU Binary。

它更接近“可执行程序”，而不是单纯参数文件。

## 32.21 Runtime

Runtime 负责真正加载和执行：

```text
Load Binary
↓
Allocate Device Memory
↓
Copy Input
↓
Launch NPU Program
↓
Sync
↓
Copy Output
```

Dynamic Shape 场景还可能进行：

```text
Shape Selection
Kernel Variant Selection
Workspace Allocation
```

## 32.22 Compile-time vs Runtime

Compile-time：

```text
Graph Optimization
Quantization
Kernel Selection
Tiling
Weight Packing
Memory Planning
```

Runtime：

```text
Input
Memory Transfer
Kernel Launch
Synchronization
Output
```

因此出现性能问题时，需要先判断是执行计划本身不好，还是 Runtime 调度不好。

## 32.23 W8A8 Linear 完整例子

PyTorch：

```python
y = linear(x)
```

Export：

```text
MatMul + Bias
```

Calibration：

```text
sx = 0.03
sw = per-channel vector
sy = 0.05
```

Quant IR：

```text
INT8 GEMM
INT32 Accumulator
Per-channel Weight Scale
INT8 Output
```

Requant：

\[
r_j
=
\frac{s_xs_{w,j}}{s_y}
\]

转换为：

```text
multiplier_j
shift_j
```

随后：

```text
Kernel Selection
↓
Tiling
↓
Weight Pack
↓
Binary
↓
Runtime
```

## 32.24 W4A16 链路

GPTQ / AWQ Weight：

```text
INT4 per-group
```

Activation：

```text
FP16
```

Backend 需要原生：

```text
W4A16 Kernel
```

Kernel 内：

```text
load packed W4
↓
unpack
↓
apply group scale
↓
GEMM with FP16 Activation
```

如果没有原生 kernel，而是：

```text
W4
↓
dequant FP16
↓
FP16 GEMM
```

实际速度收益会显著受限。

## 32.25 Quant Algorithm 与 Compiler 的责任边界

GPTQ / AWQ 主要解决：

> Weight 怎么量才能少掉精度。

SmoothQuant 主要解决：

> 如何重构 Activation / Weight 分布，使 W8A8 更友好。

而：

```text
Weight Layout
Kernel
Tiling
DMA
Runtime
```

属于 Compiler / Backend / Kernel 层。

## 32.26 反向 Debug

Accuracy Bad：

```text
Runtime Output
↓
NPU Kernel
↓
Backend IR
↓
Quant IR
↓
QDQ Graph
↓
Calibration Params
↓
FP Graph
```

如果：

```text
FakeQuant Good
NPU Bad
```

优先往 Compiler / Kernel 层查，而不是继续怀疑 Calibration。

## 32.27 性能 Debug 也要按层反推

```text
Profiler
↓
Kernel
↓
Tiling
↓
Layout
↓
Quant IR
↓
Fusion
```

不要把 kernel 性能问题错误归因到 SmoothQuant / GPTQ 参数。

## 32.28 五层问题归类

### Model / Algorithm

```text
模型本身是否正常？
```

### Quantization Algorithm

```text
Scale / Outlier / Calibration 是否合理？
```

### Graph / Compiler

```text
QDQ / Fusion / Requant 是否正确？
```

### Kernel

```text
INT8 / INT4 Kernel 是否正确高效？
```

### Runtime

```text
调度 / Memory Transfer / Sync 是否合理？
```

## 32.29 责任边界示例

如果：

```text
FP Accuracy        = 91%
FakeQuant Accuracy = 90.8%
NPU Accuracy       = 73%
```

说明算法侧大概率不是主要问题。

若进一步发现：

```text
Input matches
Weight Dequant matches
Requant output differs
```

则应重点交给：

```text
Compiler / Kernel Team
```

## 32.30 核心结论

1. PyTorch 到 NPU Binary 中间是一系列 IR 与优化阶段。
2. Calibration 输出的核心是 Quant Parameters。
3. QDQ 是量化语义，Quant IR 是编译器内部表达。
4. Backend Lowering 决定 Kernel、Tile、Layout 和 Weight Packing。
5. NPU Binary 更像已经编译好的执行程序。
6. 解决方案工程师必须先判断问题属于算法、Quant、Compiler、Kernel 还是 Runtime。


---

# 33. Case Study 1：两层 MLP 的完整 W8A8 量化

这一节用一个最小模型把此前学过的 Calibration、W8A8、Per-Channel Weight、INT32 Accumulate、Requant、误差传播、Sensitivity 和 Mixed Precision 串起来。

## 33.1 FP 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            4,
            8
        )

        self.fc2 = nn.Linear(
            8,
            3
        )

    def forward(self, x):

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        return x
```

初始化：

```python
torch.manual_seed(0)

model = TinyMLP()
model.eval()
```

## 33.2 Calibration 与 Test Data 分离

```python
calib_data = torch.randn(
    256,
    4
)

test_data = torch.randn(
    32,
    4
)
```

原则：

```text
Calibration Set
≠
Evaluation Set
```

Calibration 用来确定量化参数，Evaluation 用来验证量化精度。

## 33.3 FP Baseline

```python
with torch.no_grad():
    y_fp = model(test_data)
```

后面所有误差都与 FP reference 比较。

## 33.4 Activation Scale

最简单 symmetric INT8：

```python
def get_int8_scale(x):

    amax = x.abs().max()

    scale = (
        amax / 127.0
    ).clamp(min=1e-8)

    return scale
```

第一层输入：

```python
sx1 = get_int8_scale(
    calib_data
)
```

## 33.5 Weight Per-Channel INT8

PyTorch Linear Weight：

```text
[out_features, in_features]
```

每个 output channel 一个 scale：

```python
def quantize_weight_pc_int8(W):

    scale = (
        W.abs()
        .amax(dim=1)
        / 127.0
    ).clamp(min=1e-8)

    q = torch.round(
        W /
        scale.unsqueeze(1)
    )

    q = torch.clamp(
        q,
        -128,
        127
    )

    return (
        q.to(torch.int8),
        scale
    )
```

第一层：

```python
w1_q, sw1 = (
    quantize_weight_pc_int8(
        model.fc1.weight.data
    )
)
```

此时：

```text
w1_q.shape = [8,4]
sw1.shape  = [8]
```

## 33.6 模拟真正的 INT8 Linear

```python
def int8_linear(
    x_fp,
    sx,
    w_q,
    sw,
    bias_fp=None,
    sy=None
):

    x_q = torch.round(
        x_fp / sx
    )

    x_q = torch.clamp(
        x_q,
        -128,
        127
    ).to(torch.int8)

    acc = (
        x_q.to(torch.int32)
        @
        w_q.to(torch.int32).T
    )

    if bias_fp is not None:

        bias_q = torch.round(
            bias_fp /
            (sx * sw)
        ).to(torch.int32)

        acc = (
            acc
            +
            bias_q.unsqueeze(0)
        )

    acc_scale = (
        sx * sw
    )

    if sy is None:

        y = (
            acc.float()
            *
            acc_scale.unsqueeze(0)
        )

        return y

    else:

        multiplier = (
            acc_scale / sy
        )

        y_q = torch.round(
            acc.float()
            *
            multiplier.unsqueeze(0)
        )

        y_q = torch.clamp(
            y_q,
            -128,
            127
        )

        return (
            y_q.to(torch.int8),
            sy
        )
```

这段函数包含：

```text
Activation Quant
Weight Per-Channel
INT8 × INT8
INT32 Accumulator
Bias Quantization
Per-Channel Acc Scale
Requant
```

## 33.7 单层误差比较

```python
with torch.no_grad():

    fc1_fp = model.fc1(
        test_data
    )

    fc1_q_sim = int8_linear(
        test_data,
        sx1,
        w1_q,
        sw1,
        model.fc1.bias,
        sy=None
    )
```

比较函数：

```python
def compare(
    ref,
    test,
    eps=1e-12
):

    mse = torch.mean(
        (ref - test) ** 2
    )

    cosine = F.cosine_similarity(
        ref.flatten(),
        test.flatten(),
        dim=0
    )

    signal = torch.sum(
        ref ** 2
    )

    noise = torch.sum(
        (ref - test) ** 2
    )

    sqnr = (
        10 *
        torch.log10(
            (signal + eps)
            /
            (noise + eps)
        )
    )

    return {
        "mse": mse.item(),
        "cosine": cosine.item(),
        "sqnr": sqnr.item()
    }
```

## 33.8 第二层 Activation Calibration

第二层输入不是原始 Input，而是：

\[
X_2=ReLU(fc1(X))
\]

因此：

```python
with torch.no_grad():

    fc1_calib = model.fc1(
        calib_data
    )

    x2_calib = F.relu(
        fc1_calib
    )

sx2 = get_int8_scale(
    x2_calib
)
```

不同 Layer 的 Activation Scale 来自不同中间 Tensor。

## 33.9 第二层 Weight

```python
w2_q, sw2 = (
    quantize_weight_pc_int8(
        model.fc2.weight.data
    )
)
```

此时：

```text
w2_q.shape = [3,8]
sw2.shape  = [3]
```

## 33.10 第一层 Requant

```python
fc1_int8, sx2_used = int8_linear(
    test_data,
    sx1,
    w1_q,
    sw1,
    model.fc1.bias,
    sy=sx2
)
```

第一层 output 直接变成：

```text
INT8 @ scale sx2
```

## 33.11 Integer ReLU

Symmetric zero-point = 0 时：

\[
ReLU(q)=\max(q,0)
\]

所以：

```python
relu_q = torch.clamp(
    fc1_int8,
    min=0
)
```

Linear + ReLU 因此非常容易做 integer fusion。

## 33.12 第二层直接接受 INT8 Input

```python
def int8_linear_from_int8(
    x_q,
    sx,
    w_q,
    sw,
    bias_fp=None
):

    acc = (
        x_q.to(torch.int32)
        @
        w_q.to(torch.int32).T
    )

    if bias_fp is not None:

        bias_q = torch.round(
            bias_fp /
            (sx * sw)
        ).to(torch.int32)

        acc = (
            acc
            +
            bias_q.unsqueeze(0)
        )

    y = (
        acc.float()
        *
        (sx * sw).unsqueeze(0)
    )

    return y
```

运行：

```python
y_int8 = int8_linear_from_int8(
    relu_q,
    sx2,
    w2_q,
    sw2,
    model.fc2.bias
)
```

完整整数链：

```text
FP Input
↓ Q
INT8 fc1
↓
INT32 Acc
↓ Requant
INT8
↓ ReLU
INT8
↓ INT8 fc2
↓
INT32 Acc
↓ DQ
FP Output
```

## 33.13 误差传播

如果：

```text
fc1 cosine = 0.999
final cosine = 0.985
```

误差可能来自：

```text
fc1 quant error
↓
requant error
↓
ReLU boundary effect
↓
fc2 quant error
↓
final error
```

多层量化是逐层误差传播，而不是只看单层误差。

## 33.14 制造 Activation Outlier

```python
bad_calib = torch.randn(
    256,
    4
)

bad_calib[:, 2] *= 30
```

重新计算：

```python
sx1_bad = get_int8_scale(
    bad_calib
)
```

由于第三个 channel 被放大，per-tensor scale 会显著增大，普通 channel 的量化分辨率下降。

## 33.15 Percentile Observer

```python
def percentile_threshold(
    x,
    p=0.999
):
    return torch.quantile(
        x.abs().flatten(),
        p
    )
```

然后：

```python
threshold = percentile_threshold(
    bad_calib,
    0.999
)

sx1_percentile = (
    threshold / 127
)
```

可直接比较：

```text
MinMax
vs
Percentile
```

观察是否因为 clipping 少量 outlier，反而降低总体量化误差。

## 33.16 Weight Per-Tensor 对比

```python
def quantize_weight_pt_int8(W):

    scale = (
        W.abs().max()
        / 127
    ).clamp(min=1e-8)

    q = torch.round(
        W / scale
    )

    q = torch.clamp(
        q,
        -128,
        127
    )

    return (
        q.to(torch.int8),
        scale
    )
```

将其与 Per-Channel Weight 对比，可观察 granularity 对 Weight Reconstruction Error 的影响。

## 33.17 制造 Sensitive Layer

例如：

```python
with torch.no_grad():
    model.fc2.weight[0] *= 20
```

重新 calibration + quantization。

如果 fc2 的量化误差明显放大，则 sensitivity analysis 应把 fc2 排到更前面。

## 33.18 Mixed Precision 恢复

例如：

```text
fc1 → INT8
ReLU → INT8
fc2 → FP16
```

执行：

```python
fc1_q, _ = int8_linear(
    test_data,
    sx1,
    w1_q,
    sw1,
    model.fc1.bias,
    sy=sx2
)

relu_q = torch.clamp(
    fc1_q,
    min=0
)

relu_fp = (
    relu_q.float()
    * sx2
)

y_mixed = model.fc2(
    relu_fp
)
```

如果最终误差显著恢复，说明 fc2 是高敏感层，FP16 fallback 有效。

但代价是：

```text
INT8
↓
DQ
↓
FP16 fc2
```

Mixed Precision 会引入额外 Q/DQ 与 FP16 kernel。

## 33.19 Bias Quantization

每个 output channel：

\[
s_{acc,j}
=
s_xs_{w,j}
\]

Bias：

\[
b_{q,j}
=
round
\left(
\frac{b_j}
{s_xs_{w,j}}
\right)
\]

所以 Bias Scale 本质也是 per-output-channel。

## 33.20 Requant 的来源

第一层：

\[
Real_j
=
Acc_j
s_xs_{w,j}
\]

下一层希望：

\[
Real_j=q_js_{x2}
\]

因此：

\[
q_j
=
round
\left(
Acc_j
\frac{s_xs_{w,j}}{s_{x2}}
\right)
\]

这就是 Compiler 中 per-channel multiplier / shift 的数学来源。

## 33.21 映射到 Transformer

Toy MLP：

```text
fc1
↓
ReLU
↓
fc2
```

对应 Transformer MLP：

```text
up_proj / gate_proj
↓
SiLU / GELU
↓
down_proj
```

Transformer 只是在此基础上增加：

```text
LayerNorm
Residual
Attention
```

底层量化逻辑完全相通。

## 33.22 本 Case Study 串联的知识

```text
Calibration
MinMax Observer
Percentile
Activation Per-Tensor
Weight Per-Channel
INT8 Quantize
INT32 Accumulate
Bias Quantization
Requant
Integer ReLU
Error Propagation
Sensitivity
Mixed Precision
```

这是一条从量化数学到工程实现的完整最小链路。


---

# 34. Case Study 2：自动化 Quantization Debug Toolkit

这一节把上一章的手工 W8A8 实验升级为自动化调试工具。

目标：

```text
1. 抓 FP / Quant 中间输出
2. 自动计算 MSE / MAE / Cosine / SQNR
3. 自动找 first bad layer
4. 比较 MinMax / Percentile Observer
5. 做 Layer-wise Sensitivity Ranking
6. 自动尝试 Mixed Precision Fallback
```

## 34.1 统一 Tensor Metrics

```python
def tensor_metrics(
    ref,
    test,
    eps=1e-12
):
    ref = ref.float()
    test = test.float()

    diff = ref - test

    mse = torch.mean(
        diff ** 2
    ).item()

    mae = torch.mean(
        diff.abs()
    ).item()

    cosine = F.cosine_similarity(
        ref.flatten(),
        test.flatten(),
        dim=0,
        eps=eps
    ).item()

    signal = torch.sum(
        ref ** 2
    )

    noise = torch.sum(
        diff ** 2
    )

    sqnr = (
        10 *
        torch.log10(
            (signal + eps) /
            (noise + eps)
        )
    ).item()

    return {
        "mse": mse,
        "mae": mae,
        "cosine": cosine,
        "sqnr_db": sqnr,
    }
```

以后所有层统一使用同一套指标。

## 34.2 FP Collector

```python
class Collector:

    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.execution_order = []
        self.handles = []

    def input_hook(self, name):

        def hook(module, inputs):

            if not inputs:
                return

            x = inputs[0]

            if torch.is_tensor(x):
                self.inputs[name] = (
                    x.detach()
                    .float()
                    .cpu()
                )

        return hook

    def output_hook(self, name):

        def hook(module, inputs, output):

            if name not in self.execution_order:
                self.execution_order.append(name)

            if torch.is_tensor(output):
                self.outputs[name] = (
                    output.detach()
                    .float()
                    .cpu()
                )

        return hook

    def register(self, model):

        for name, module in model.named_modules():

            if isinstance(
                module,
                nn.Linear
            ):

                self.handles.append(
                    module.register_forward_pre_hook(
                        self.input_hook(name)
                    )
                )

                self.handles.append(
                    module.register_forward_hook(
                        self.output_hook(name)
                    )
                )

    def remove(self):

        for h in self.handles:
            h.remove()

        self.handles.clear()
```

## 34.3 Quantized Path 记录中间 Tensor

```python
def run_quantized_mlp(
    model,
    x,
    sx1,
    sx2,
    w1_q,
    sw1,
    w2_q,
    sw2
):

    records = {}

    records["fc1_input"] = x.float()

    fc1_q, _ = int8_linear(
        x,
        sx1,
        w1_q,
        sw1,
        model.fc1.bias,
        sy=sx2
    )

    records["fc1_output_int8"] = (
        fc1_q
    )

    records["fc1_output_dq"] = (
        fc1_q.float() * sx2
    )

    relu_q = torch.clamp(
        fc1_q,
        min=0
    )

    records["fc2_input_int8"] = (
        relu_q
    )

    records["fc2_input_dq"] = (
        relu_q.float() * sx2
    )

    y = int8_linear_from_int8(
        relu_q,
        sx2,
        w2_q,
        sw2,
        model.fc2.bias
    )

    records["fc2_output"] = y

    return y, records
```

## 34.4 为什么比较前要 Dequant

INT8：

```text
[10,20,30]
scale = 0.1
```

其真实值是：

```text
[1,2,3]
```

因此不能直接把整数 Tensor 与 FP Tensor 比 MSE。

必须先：

\[
x_{real}=q\times scale
\]

## 34.5 自动 Layer Report

```python
def build_report(
    fp_collector,
    q_records
):

    report = []

    mapping = {
        "fc1":
            "fc1_output_dq",

        "fc2":
            "fc2_output",
    }

    for fp_name, q_name in mapping.items():

        fp_tensor = (
            fp_collector
            .outputs[fp_name]
        )

        q_tensor = (
            q_records[q_name]
            .cpu()
        )

        metrics = tensor_metrics(
            fp_tensor,
            q_tensor
        )

        metrics["layer"] = fp_name

        report.append(metrics)

    return report
```

典型输出：

```text
fc1:
cosine = 0.9995
sqnr   = 31 dB

fc2:
cosine = 0.982
sqnr   = 18 dB
```

## 34.6 First Bad Layer

```python
def find_first_bad(
    report,
    cosine_threshold=0.98
):

    for item in report:

        if (
            item["cosine"]
            <
            cosine_threshold
        ):
            return item

    return None
```

固定阈值只是筛查工具，不是统一行业标准。

## 34.7 最大 Cosine Drop

```python
def largest_cosine_drop(report):

    best = None

    for i in range(
        1,
        len(report)
    ):

        prev = report[i - 1]
        cur = report[i]

        drop = (
            prev["cosine"]
            -
            cur["cosine"]
        )

        if (
            best is None
            or drop > best["drop"]
        ):
            best = {
                "from": prev["layer"],
                "to": cur["layer"],
                "drop": drop
            }

    return best
```

相比“最差层”，误差突变位置更接近 first bad layer。

## 34.8 Activation Statistics

```python
def activation_stats(x):

    x = x.float()

    abs_x = (
        x.abs()
        .flatten()
    )

    qs = torch.quantile(
        abs_x,
        torch.tensor([
            0.9,
            0.99,
            0.999,
            1.0
        ])
    )

    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "p90": qs[0].item(),
        "p99": qs[1].item(),
        "p999": qs[2].item(),
        "max": qs[3].item(),
    }
```

## 34.9 Outlier Ratio

可定义：

\[
R=
\frac{max}{p99.9}
\]

```python
def outlier_ratio(x):

    stats = activation_stats(x)

    return (
        stats["max"]
        /
        (
            stats["p999"]
            + 1e-12
        )
    )
```

这是调试 heuristic，不是行业统一指标。

## 34.10 自动比较 MinMax / Percentile

```python
def minmax_scale(x):

    return (
        x.abs().max()
        / 127
    ).clamp(min=1e-8)


def percentile_scale(
    x,
    p=0.999
):

    threshold = torch.quantile(
        x.abs().flatten(),
        p
    )

    return (
        threshold / 127
    ).clamp(min=1e-8)
```

真正应该比较的是：

```text
Tensor MSE
Layer Cosine
Final Output Error
Task Metric
```

而不是简单比较哪个 scale 更小。

## 34.11 Saturation Ratio

```python
def saturation_ratio(
    x,
    scale,
    qmin=-128,
    qmax=127
):

    q_float = (
        x / scale
    )

    sat = (
        (q_float < qmin)
        |
        (q_float > qmax)
    )

    return (
        sat.float()
        .mean()
        .item()
    )
```

这样可以同时观察：

```text
MinMax:
saturation ≈ 0
但 resolution 可能差

Percentile:
允许少量 saturation
但总体误差可能更低
```

## 34.12 Layer-wise Sensitivity

只量 fc1：

```text
fc1 INT8
fc2 FP
```

只量 fc2：

```text
fc1 FP
fc2 INT8
```

然后比较最终输出误差。

可定义：

\[
S_i
=
MSE(
Y_{fp},
Y_{\text{only quant layer }i}
)
\]

或：

\[
S_i
=
1-Cosine
\]

## 34.13 Sensitivity Score

```python
def sensitivity_score(
    y_fp,
    y_test
):

    m = tensor_metrics(
        y_fp,
        y_test
    )

    return {
        "mse": m["mse"],
        "cosine_drop":
            1 - m["cosine"]
    }
```

然后排序：

```python
ranking = sorted(
    scores.items(),
    key=lambda item:
        item[1]["cosine_drop"],
    reverse=True
)
```

## 34.14 自动 Mixed Precision Search

如果：

```text
All INT8
```

无法达到目标精度，就按照 sensitivity ranking：

```text
最敏感层
↓
逐层恢复 FP16
```

直到满足目标。

Toy Model 中可能得到：

```text
fc1 = INT8
fc2 = FP16
```

## 34.15 Mixed Precision 与 Bit Allocation

真实模型中候选可能是：

```text
INT4
INT8
FP16
```

因此 Mixed Precision Search 本质上已经是 Bit Allocation 的雏形。

## 34.16 加入性能 Cost

例如：

```python
latency_cost = {
    "fc1": {
        "INT8": 0.10,
        "FP16": 0.18
    },

    "fc2": {
        "INT8": 0.05,
        "FP16": 0.07
    }
}
```

那么精度恢复还应结合 latency cost。

可以定义一个简单思想：

\[
Benefit_i
=
\frac{
Accuracy\ Recovery_i
}{
Performance\ Cost_i
}
\]

这就是 Hardware-aware Mixed Precision 的基本方向。

## 34.17 理想 Toolkit Report

```text
=== Quantization Debug Report ===

Layer: fc1
Input outlier ratio: 1.4
Output cosine: 0.9987
SQNR: 28.2 dB
Sensitivity rank: #2

Layer: fc2
Input outlier ratio: 6.8
Output cosine: 0.912
SQNR: 11.3 dB
Sensitivity rank: #1

Observer Test:
MinMax:
  final cosine = 0.912
  saturation = 0%

P99.9:
  final cosine = 0.974
  saturation = 0.08%

Mixed Precision:
fc1 INT8 + fc2 FP16
  final cosine = 0.995
  estimated latency +0.02 ms
```

## 34.18 与真实 NPU 工具链连接

当前 Toolkit 使用：

```text
PyTorch Quant Simulation
```

未来只需增加：

```text
NPU Tensor Dump Loader
```

就能形成：

```text
FP
↓
FakeQuant
↓
NPU
```

三路对比。

## 34.19 真实工具还需要 Mapping

最终需要建立：

```text
PyTorch Module
↕
ONNX Node
↕
Compiler IR Op
↕
NPU Tensor / Kernel
```

一旦 Mapping 建立，现有的：

```text
MSE
Cosine
SQNR
First Bad
Sensitivity Ranking
```

都可以继续复用。

## 34.20 当前阶段的能力变化

最开始关注的是：

```text
INT8 如何表示 FP 数
```

现在关注已经升级为：

```text
哪个 Layer 出问题？
为什么？
Observer？
Outlier？
Granularity？
Requant？
Kernel？
是否应该 Mixed Precision？
性能代价多少？
```

这标志着从“量化知识”进入“量化工程能力”。


---

# 35. Case Study 3：Transformer Block Quantization

**系统讲解**：Transformer Block 的量化实验必须拆开 Q/K/V、attention score、softmax、O projection、MLP 和 residual，而不是只比较 block 最终输出。这样才能区分“Linear 量化误差”与“Softmax 放大误差”以及“Residual scale mismatch”。

**代码实现**：

```python
with torch.no_grad():
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn = score.softmax(dim=-1)
    out = o_proj(attn @ v)
    y = x + out
```

对每个中间 tensor 分别保存 FP、FakeQuant 和 NPU dump，才能定位 first bad operator。

这一节从 TinyMLP 升级到 Transformer Block，重点理解：

```text
LayerNorm
Q / K / V
QK^T
Softmax
Residual
MLP
```

这些结构为什么比普通 Linear 更难量化。

## 35.1 Tiny Attention Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyAttentionBlock(nn.Module):

    def __init__(self, dim=16):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        self.ln2 = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(
            dim,
            dim * 4
        )

        self.fc2 = nn.Linear(
            dim * 4,
            dim
        )

    def forward(self, x):

        residual = x

        h = self.ln1(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        logits = (
            q @ k.transpose(-1, -2)
        ) / math.sqrt(q.shape[-1])

        attn = torch.softmax(
            logits,
            dim=-1
        )

        context = attn @ v

        out = self.o_proj(
            context
        )

        x = residual + out

        residual = x

        h = self.ln2(x)

        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)

        x = residual + h

        return x
```

## 35.2 第一版 Quantization Strategy

不要一开始全量化。

第一版更合理：

```text
LN1       FP
Q/K/V     W8A8
QK^T      FP
Softmax   FP
A×V       FP
O_proj    W8A8

Residual  FP

LN2       FP
FC1       W8A8
GELU      FP
FC2       W8A8
Residual  FP
```

主要 GEMM 使用 W8A8，而敏感/非线性算子保高精度。

## 35.3 为什么先量 Linear

Transformer 主要参数与计算集中在：

```text
Q Projection
K Projection
V Projection
O Projection
FC1
FC2
```

这些本质都是：

\[
Y=XW^T+b
\]

因此前面 TinyMLP 的 INT8 Linear 实现可以直接复用。

## 35.4 为什么 LayerNorm 常保 FP

LayerNorm：

\[
\mu=
\frac1D\sum_i x_i
\]

\[
\sigma^2=
\frac1D\sum_i(x_i-\mu)^2
\]

\[
y_i=
\gamma_i
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
+
\beta_i
\]

包含：

```text
mean
variance
sqrt
division
```

整数实现复杂，而且会直接影响后续 Q/K/V 的 distribution。

## 35.5 LayerNorm 与 Quantization 的连接

虽然 LayerNorm 本身常保 FP，但其输出直接进入：

```text
Q / K / V Linear
```

因此：

```text
LayerNorm Output
```

是 QKV Activation Calibration 的关键 Tensor。

SmoothQuant 也常作用在：

```text
LayerNorm
↓
Linear
```

这个边界。

## 35.6 Q/K/V Weight Per-Channel INT8

```python
q_w, q_sw = quantize_weight_pc_int8(
    model.q_proj.weight
)

k_w, k_sw = quantize_weight_pc_int8(
    model.k_proj.weight
)

v_w, v_sw = quantize_weight_pc_int8(
    model.v_proj.weight
)
```

三者 Weight Scale：

```text
shape = [out_dim]
```

## 35.7 Q/K/V 共享 Activation Scale

Q/K/V 输入都是：

```text
h = LayerNorm(x)
```

因此可共享：

\[
s_h
\]

而不是分别产生三套 activation scale。

这对 fan-out Tensor 更自然，也更有利于 Compiler / Runtime。

## 35.8 SmoothQuant 对 Q/K/V 的一致性

若：

\[
h'=h/S
\]

则必须同步：

\[
W_Q'=W_QS
\]

\[
W_K'=W_KS
\]

\[
W_V'=W_VS
\]

不能只修改 Q 分支，否则数学等价关系被破坏。

## 35.9 QK^T 的双边误差

量化后：

\[
Q\rightarrow Q+\Delta Q
\]

\[
K\rightarrow K+\Delta K
\]

则：

\[
\hat L
=
(Q+\Delta Q)
(K+\Delta K)^T
\]

展开：

\[
\hat L
=
QK^T
+
\Delta QK^T
+
Q\Delta K^T
+
\Delta Q\Delta K^T
\]

所以：

\[
\Delta L
=
\Delta QK^T
+
Q\Delta K^T
+
\Delta Q\Delta K^T
\]

Attention 与普通 Linear 的区别之一是：

```text
Q 是动态 Activation
K 也是动态 Activation
```

两边量化误差会共同传播到 logits。

## 35.10 Q/K Cosine 高，不代表 Logits 一定高

可能出现：

```text
Q cosine      = 0.998
K cosine      = 0.998
Logits cosine = 0.93
```

因此 Transformer Debug 不应只检查 Q/K/V Linear 输出，还要检查：

```text
Attention Logits
```

## 35.11 Softmax 敏感性

Softmax：

\[
p_i
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

它对 logits 的相对差异敏感。

例如：

```text
[5.0, 4.8, 1.0]
```

变为：

```text
[4.8, 5.0, 1.0]
```

虽然误差很小，但最大 attention 概率的位置会交换。

因此：

```text
Attention Logits
Softmax
```

往往比普通 hidden state 更敏感。

## 35.12 Attention × V

\[
Context=AV
\]

如果 A 保高精度而 V 量化：

\[
\hat V=V+\Delta V
\]

则：

\[
A\hat V
=
AV+A\Delta V
\]

相比 QK 双边量化，误差结构更简单。

## 35.13 Residual Scale Alignment

Residual：

\[
Y=X+O
\]

若：

\[
X=s_xX_q
\]

\[
O=s_oO_q
\]

不能直接：

```text
X_q + O_q
```

必须对齐到 output scale：

\[
Y_q
=
round
\left(
X_q\frac{s_x}{s_y}
+
O_q\frac{s_o}{s_y}
\right)
\]

这就是 Residual Scale Alignment。

## 35.14 Residual Dynamic Range Mismatch

如果：

```text
Residual magnitude ≈ 10
Attention output magnitude ≈ 0.1
```

为了统一 scale，大 residual 可能使小 branch 的量化分辨率变差。

因此 residual 不只是 scale alignment 问题，还可能有 branch energy mismatch。

## 35.15 Transformer Debug Checkpoints

建议记录：

```text
block_input
ln1_output

q_output
k_output
v_output

attention_logits
softmax_output
context

o_proj_output
residual1_output

ln2_output
fc1_output
activation_output
fc2_output

block_output
```

这样才能定位真正的 First Bad Operator。

## 35.16 Debuggable Forward

```python
def forward_debug(self, x):

    r = {}

    r["block_input"] = x

    residual = x

    h = self.ln1(x)
    r["ln1"] = h

    q = self.q_proj(h)
    k = self.k_proj(h)
    v = self.v_proj(h)

    r["q"] = q
    r["k"] = k
    r["v"] = v

    logits = (
        q @ k.transpose(-1, -2)
    ) / math.sqrt(q.shape[-1])

    r["logits"] = logits

    attn = torch.softmax(
        logits,
        dim=-1
    )

    r["attn"] = attn

    context = attn @ v
    r["context"] = context

    out = self.o_proj(context)
    r["o_proj"] = out

    x = residual + out
    r["residual1"] = x

    return x, r
```

真实研究代码里可显式保存 Tensor；工具链中则更适合 FX / Graph IR 插桩。

## 35.17 Activation Calibration Points

至少要收集：

```text
LN1 Output      → Q/K/V Input
Context         → O Projection Input
LN2 Output      → FC1 Input
GELU Output     → FC2 Input
```

所以一个 block 内会存在多套 activation scale：

```text
s_qkv_in
s_o_in
s_fc1_in
s_fc2_in
```

## 35.18 Q/K/V 量化实验设计

可做：

```text
Experiment 1
Only Q Quant

Experiment 2
Only K Quant

Experiment 3
Q + K Quant

Experiment 4
Q + K + V Quant
```

记录：

```text
Q Cos
K Cos
Logits Cos
Attn Cos
Output Cos
```

用来观察双边误差如何经 QK^T 和 Softmax 放大。

## 35.19 制造 Q/K Outlier

可以人为：

```python
h[..., 3] *= 20
```

再观察：

```text
Per-Tensor INT8
↓
Q/K cosine
↓
Logits cosine
↓
Softmax cosine
```

然后依次尝试：

```text
Percentile
SmoothQuant
QuaRot
DuQuant
```

## 35.20 SmoothQuant 为什么适合 Transformer

LayerNorm output 的 hidden channel outlier 会直接进入 Q/K/V。

通过：

\[
h'=h/S
\]

\[
W_Q'=W_QS
\]

\[
W_K'=W_KS
\]

\[
W_V'=W_VS
\]

可以把 activation difficulty 迁移到更容易 per-channel quantize 的 Weight。

## 35.21 Transformer First Bad Operator

例如：

```text
LN1           cosine = 0.9999
Q             cosine = 0.998
K             cosine = 0.998
V             cosine = 0.999
Logits        cosine = 0.91
Softmax       cosine = 0.80
Context       cosine = 0.78
O Projection  cosine = 0.76
```

不能因为 O Projection 最差就认为它是源头。

真正 first bad operator 是：

```text
Attention Logits
```

优先检查：

```text
Q/K Quantization
Scale
Outlier
QK Compute Precision
```

## 35.22 两种误差传播机制

普通 Linear Chain：

```text
Quant Error
↓
Layer-by-Layer Propagation
```

Attention：

```text
Q Error + K Error
↓
QK^T 双边组合
↓
Softmax 非线性放大
```

Attention 的误差传播机制明显更复杂。


---

# 36. Attention Quantization：Q/K 双边误差、Softmax 敏感性与 Temperature Drift

**系统讲解**：Attention 的误差具有双边传播特征：Q/K 量化误差共同影响 (QK^T)，score 的小变化再经过 Softmax 变成概率分布变化。若量化使 score 方差变小，attention 可能过于平坦；若方差变大，attention 可能过于尖锐，这可以用 temperature drift 描述。

**公式**：设量化前后 score 的标准差分别为 (sigma_{fp})、(sigma_q)，可定义近似 temperature ratio：

\[
r_T=\frac{\sigma_q}{\sigma_{fp}+\epsilon}
\]

如果 (r_T\ll1)，分布可能变平；如果 (r_T\gg1)，分布可能变尖。这个指标不能替代任务评估，但有助于区分 Q/K 误差和普通输出 MSE。

**代码实现**：

```python
def attention_temperature_ratio(score_fp, score_q):
    return score_q.float().std() / (score_fp.float().std() + 1e-6)
```

这一节从最小手算例子出发，理解为什么 Q/K 的小量化误差会被 \(QK^T\) 和 Softmax 放大，并进一步连接到 Attention Temperature Drift 与 Residual Energy Drift。

## 36.1 最小 Attention 手算

设：

\[
Q=
\begin{bmatrix}
1 & 1\\
1 & 0
\end{bmatrix}
\]

\[
K=
\begin{bmatrix}
1 & 1\\
0.8 & 1
\end{bmatrix}
\]

暂时忽略：

\[
1/\sqrt d
\]

则：

\[
L=QK^T
\]

得到：

\[
L=
\begin{bmatrix}
2 & 1.8\\
1 & 0.8
\end{bmatrix}
\]

第一行 Softmax：

\[
softmax([2,1.8])
\approx
[0.550,0.450]
\]

说明两个 key 的注意力权重很接近，因此容易受到小误差扰动。

## 36.2 Q/K 小误差如何改变 Logits

设量化后：

\[
\hat Q=
\begin{bmatrix}
0.95 & 1\\
1 & 0
\end{bmatrix}
\]

\[
\hat K=
\begin{bmatrix}
1 & 1\\
0.9 & 1
\end{bmatrix}
\]

第一行 logits：

\[
[1.95,\ 1.855]
\]

原始间距：

\[
2-1.8=0.2
\]

量化后：

\[
1.95-1.855=0.095
\]

说明 Q/K 元素误差并不大，但 logits 相对间距几乎缩小一半。

## 36.3 Softmax 后的变化

原：

\[
softmax([2,1.8])
\approx
[0.550,0.450]
\]

量化后：

\[
softmax([1.95,1.855])
\approx
[0.524,0.476]
\]

关键在于：

> Softmax 的行为主要由 logits 的相对差异决定。

当 logits 原本很接近时，小误差很容易改变 attention ranking。

## 36.4 Q/K 双边误差展开

若：

\[
\hat Q=Q+\Delta Q
\]

\[
\hat K=K+\Delta K
\]

则：

\[
\hat L
=
(Q+\Delta Q)
(K+\Delta K)^T
\]

展开：

\[
\hat L
=
QK^T
+
\Delta QK^T
+
Q\Delta K^T
+
\Delta Q\Delta K^T
\]

所以：

\[
\Delta L
=
\Delta QK^T
+
Q\Delta K^T
+
\Delta Q\Delta K^T
\]

Attention 与普通 Linear 的一个关键区别是：

```text
Q 是动态 Tensor
K 也是动态 Tensor
```

两边误差会共同进入 logits。

## 36.5 为什么 Q/K Cosine 高仍可能有问题

可能出现：

```text
Q cosine      = 0.998
K cosine      = 0.998
Logits cosine = 0.93
```

甚至：

```text
Logits cosine ≈ 1
```

但 Softmax 分布依旧明显变化。

所以 Attention Debug 不能只看 Q/K 的 cosine。

## 36.6 Attention Temperature

更一般地：

\[
A=
softmax
\left(
\frac{L}{T}
\right)
\]

其中 \(T\) 是 temperature。

例如：

```text
T 大 → 分布更平
T 小 → 分布更尖
```

如果量化后：

\[
\hat L\approx\alpha L
\]

则等效 temperature：

\[
T_{effective}
=
\frac1\alpha
\]

若：

\[
\alpha>1
\]

Attention 变尖；

若：

\[
\alpha<1
\]

Attention 变平。

## 36.7 Temperature Drift

量化不一定只是产生 element-wise error，还可能系统性改变整个 attention head 的 logits dispersion。

这类问题可以表现为：

```text
Logits direction 基本正确
但 logits amplitude / std 漂移
```

因此称为：

```text
Attention Temperature Drift
```

## 36.8 为什么只看 Cosine 会漏掉 Temperature Drift

如果：

\[
\hat L=2L
\]

则：

\[
cos(L,\hat L)=1
\]

但：

\[
softmax(L)
\neq
softmax(2L)
\]

因此 Attention Debug 还需要看：

```text
Logits Std
Logits RMS
Logits Range
Softmax Entropy
```

## 36.9 Attention Logits Statistics

```python
def attention_stats(logits):

    return {
        "mean":
            logits.mean().item(),

        "std":
            logits.std().item(),

        "rms":
            torch.sqrt(
                torch.mean(
                    logits ** 2
                )
            ).item(),

        "max":
            logits.max().item(),

        "min":
            logits.min().item(),
    }
```

如果：

```text
FP std    = 1.20
Quant std = 1.75
```

说明 quantized attention 变得更尖锐。

## 36.10 Softmax Entropy

定义：

\[
H(p)
=
-\sum_i p_i\log p_i
\]

```python
def attention_entropy(
    attn,
    eps=1e-12
):

    return (
        -(
            attn
            *
            torch.log(
                attn + eps
            )
        )
        .sum(dim=-1)
        .mean()
        .item()
    )
```

如果：

```text
FP entropy    = 2.1
Quant entropy = 1.4
```

说明量化 Attention 明显变尖。

## 36.11 Attention Temperature Matching 的直觉

如果：

```text
FP logits std    = 1.2
Quant logits std = 1.8
```

可估一个校准因子：

\[
\alpha
=
\frac{1.2}{1.8}
\approx0.667
\]

然后：

\[
L_{corrected}
=
0.667L_{quant}
\]

使 quantized logits dispersion 更接近 FP。

教学版实现：

```python
def estimate_temperature_scale(
    fp_logits,
    q_logits,
    eps=1e-8
):

    fp_std = fp_logits.std(
        dim=(-2, -1),
        keepdim=True
    )

    q_std = q_logits.std(
        dim=(-2, -1),
        keepdim=True
    )

    alpha = (
        fp_std /
        (q_std + eps)
    )

    return alpha
```

然后：

```python
q_logits_corrected = (
    q_logits * alpha
)
```

## 36.12 为什么适合 Per-Head

不同 attention head 的：

```text
Q/K Distribution
Logits Variance
Attention Sharpness
```

差异很大，因此：

```text
Per-Head Calibration
```

比全层共享一个校准因子更自然。

## 36.13 ATM 修的不是 Cosine

ATM 主要修：

```text
Logits Scale / Distribution
```

而不是单纯修 element-wise error。

典型现象：

```text
Q/K cosine 很高
Logits cosine 很高
但 Logits Std Ratio 异常
Softmax Entropy 明显漂移
```

这时问题更像 Temperature Drift，而不是方向误差。

## 36.14 Attention Debug 指标链

建议依次观察：

```text
Q cosine / SQNR / Outlier
K cosine / SQNR / Outlier
↓
Logits cosine
Logits std / RMS ratio
↓
Softmax cosine
Softmax entropy
↓
Context cosine / magnitude
```

First Bad Phenomenon 不一定表现为 cosine collapse，也可能是 scale / temperature drift。

## 36.15 Residual Energy Drift

Attention 输出最终进入：

\[
Y=X+F(X)
\]

如果量化改变了：

```text
F(X) 的幅值
```

则 residual 中：

```text
旧信息 X
+
新信息 F(X)
```

的相对能量关系也会改变。

多层堆叠后可能产生：

```text
Residual-Stream Energy Drift
```

## 36.16 VLA 中为什么更敏感

VLA 可能经历：

```text
VLM Hidden State
↓
Action Head / DiT
↓
Multiple Denoising Steps
↓
Continuous Action
```

前部的 attention / residual scale drift 可能在：

```text
Residual Stack
+
Diffusion Steps
```

中持续传播。

最终可能影响：

```text
Action Magnitude
Action Direction
Temporal Consistency
```

## 36.17 Attention Quant Debug 表

| Checkpoint | 重点指标 |
|---|---|
| Q | Cosine / SQNR / Outlier |
| K | Cosine / SQNR / Outlier |
| Logits | Cosine + Std/RMS Ratio |
| Softmax | Cosine + Entropy |
| Context | Cosine / Magnitude |
| O_proj | Cosine / Energy |
| Residual | Branch Energy Ratio |

Attention Debug 不应只依赖一个 Cosine 指标。


---

# 37. Residual / LayerNorm / Scale Alignment 专题

**系统讲解**：Pre-Norm Transformer 中，LayerNorm 主要稳定 transform branch 的输入，而 shortcut branch 仍然保留原始 hidden state。因此 LayerNorm 不能自动修复两个 branch 的量化能量不匹配。排障时应同时记录 `shortcut RMS`、`transform RMS`、`branch ratio` 和 Add 后输出误差。

**公式**：

\[
r_{branch}=\frac{RMS(F(X))}{RMS(X)+\epsilon}
\]

比较 (r_{branch}^{fp}) 与 (r_{branch}^{q})，往往比只看最终 Add 的 cosine 更容易发现 transform branch 被 scale 压扁或 saturation 的问题。

**代码实现**：

```python
def branch_energy(x, f):
    rms = lambda t: t.float().square().mean().sqrt()
    return {
        "shortcut_rms": rms(x).item(),
        "transform_rms": rms(f).item(),
        "branch_ratio": (rms(f) / (rms(x) + 1e-6)).item(),
    }

这一节重点理解 Transformer Residual 中不同量化尺度如何对齐、为什么会产生 branch dynamic range mismatch 与 residual energy drift，以及 LayerNorm / SmoothQuant folding 在这里如何连接。

## 37.1 Residual Add 的基本形式

Transformer 中常见：

$$
Y=X+F(X)
$$

例如：

```text
X ─────────────────────┐
                       Add → Y
LN → Attention → O ────┘
```

或：

```text
X ──────────────────┐
                    Add → Y
LN → MLP ───────────┘
```

## 37.2 不同 Scale 的 INT8 Tensor 不能直接相加

设：

$$
X=s_xX_q
$$

$$
F=s_fF_q
$$

如果：

$$
s_x\neq s_f
$$

则：

```text
X_q + F_q
```

没有正确的实数意义。

Quantized Tensor 应理解为：

```text
整数值
+
量化单位 scale
```

不同 scale 就像不同单位，必须先对齐。

## 37.3 Quantized Residual Add

希望输出：

$$
Y=s_yY_q
$$

则：

$$
s_yY_q
=
s_xX_q+s_fF_q
$$

所以：

$$
Y_q
=
round
\left(
\frac{s_x}{s_y}X_q+
\frac{s_f}{s_y}F_q
\right)
$$

这就是 Quantized Residual Add。

## 37.4 手算 Requant

例如：

$$
s_x=0.1
$$

$$
s_f=0.01
$$

选择：

$$
s_y=0.1
$$

则：

$$
Y_q
=
X_q+0.1F_q
$$

本质上对 F branch 做：

$$
F_q'
=
round
\left(
F_q\frac{s_f}{s_y}
\right)
$$

即 Requant。

## 37.5 Compiler 的三种对齐策略

### B 对齐 A

$$
s_y=s_A
$$

### A 对齐 B

$$
s_y=s_B
$$

### 两支都对齐到新的 Output Scale

$$
s_y
$$

由输出范围、精度和性能共同决定。

## 37.6 Output Scale 的 Trade-off

如果：

$$
s_y
$$

太大：

```text
dynamic range 大
quantization resolution 差
```

如果太小：

```text
resolution 好
但更容易 saturation
```

Residual Add 本身也需要 Quantization Policy。

## 37.7 Branch Dynamic Range Mismatch

例如：

```text
Residual range:
[-10,10]

Attention output:
[-0.1,0.1]
```

若 output scale 主要由 residual 决定：

$$
s_y
\approx
10/127
\approx0.0787
$$

那么 Attention branch：

$$
0.1/0.0787
\approx1.27
$$

只有极少数整数等级可表示。

这可能导致 transform branch 的细节被严重压缩。

## 37.8 为什么这比普通 MSE 更危险

如果：

$$
X=10
$$

$$
F=0.08
$$

真实：

$$
Y=10.08
$$

若 F 被量化成 0：

$$
\hat Y=10
$$

绝对误差只有 0.08，但结构上：

```text
Residual branch 保留
Transform branch 消失
```

网络语义可能受到更大影响。

## 37.9 Branch Energy

可用 RMS 衡量：

$$
RMS(X)
=
\sqrt{
\frac1N
\sum_iX_i^2
}
$$

代码：

```python
def rms(x):
    return torch.sqrt(
        torch.mean(
            x.float() ** 2
        )
    )
```

定义 transform / residual ratio：

$$
R=
\frac{
RMS(F(X))
}{
RMS(X)
}
$$

## 37.10 FP 与 Quant Branch Ratio

例如：

```text
FP:
Residual RMS  = 4.0
Attention RMS = 1.0
Ratio         = 0.25

Quant:
Residual RMS  = 4.0
Attention RMS = 0.4
Ratio         = 0.10
```

即使 residual output cosine 很高，也可能已经存在明显 branch energy drift。

## 37.11 Residual Output Cosine 会掩盖 Branch Error

如果：

$$
|X|\gg|F|
$$

即使：

$$
F
$$

误差很大，

最终：

$$
Y=X+F
$$

仍主要由 X 决定。

因此：

```text
Residual Output Cosine ≈ 1
```

并不能证明 transform branch 正常。

Residual Debug 必须同时看：

```text
Transform Branch
Residual Branch
Add Output
```

## 37.12 Residual Energy Drift 的累积

每层：

$$
X_{l+1}
=
X_l+F_l(X_l)
$$

量化后如果：

$$
\hat F_l
=
\alpha_lF_l
$$

即使每层只有小幅系统性偏差，经过多层 residual stack 后，也可能改变整体信息注入轨迹。

这不是简单的：

$$
0.9^{32}
$$

因为：

$$
F_l
$$

依赖当前 hidden state，但系统性 branch ratio 偏差仍会累积。

## 37.13 简单 Energy Calibration

若 calibration 数据上：

$$
RMS(F_{fp})=1.2
$$

$$
RMS(F_q)=0.8
$$

可估：

$$
\beta
=
\frac{1.2}{0.8}
=
1.5
$$

然后：

$$
F_{corrected}
=
1.5F_q
$$

再进入 residual：

$$
Y=X+1.5F_q
$$

这是一种朴素的 Output Branch Magnitude Calibration。

## 37.14 为什么 Energy Scale 可以较便宜地部署

若：

$$
O_{fp}
=
O_qs_o
$$

还要：

$$
O'=\beta O_{fp}
$$

则：

$$
O'
=
O_q(\beta s_o)
$$

所以：

$$
\beta
$$

可吸收到 dequant / requant scale 中，而不一定新增 runtime Mul。

## 37.15 与 Attention Temperature Calibration 的联系

ATM：

$$
L_{corrected}
=
\alpha L_q
$$

Residual Energy Calibration：

$$
F_{corrected}
=
\beta F_q
$$

二者本质相同：

> 量化后的方向关系大致正确，但幅值统计发生 drift，用轻量 scale 做 distribution calibration。

区别在于校准对象不同。

## 37.16 Pre-Norm Transformer

Pre-Norm：

$$
X_{l+1}
=
X_l+
F(LN(X_l))
$$

Residual shortcut：

```text
X_l
```

Transform branch input：

```text
LN(X_l)
```

LayerNorm 可以一定程度稳定 transform branch 输入分布。

## 37.17 LayerNorm 不能自动修复 Residual Energy Drift

虽然：

```text
LN(X_l)
```

被重新归一化，

但 shortcut：

```text
X_l
```

不会经过 LayerNorm 再进入 Add。

所以 LayerNorm 能稳定 transform branch 输入，但不能自动恢复：

```text
Residual Branch
vs
Transform Branch
```

的相对能量。

## 37.18 LayerNorm Output 为什么适合作为量化边界

LayerNorm 理论上使 hidden vector：

```text
mean ≈ 0
variance ≈ 1
```

因此其输出往往比 residual stream 更规整。

所以：

```text
LN
↓
QKV Linear
```

是自然的 activation quantization boundary。

## 37.19 LayerNorm 后仍可能有 Channel Outlier

LayerNorm 是 token-wise normalization。

它不保证：

```text
每个 hidden channel
```

分布相同。

所以仍然可能出现长期 channel-wise outlier。

这也是 SmoothQuant 主要针对的问题之一。

## 37.20 LayerNorm + Linear Scale Folding

设：

$$
h=LN(x)
$$

希望：

$$
h'=h/S
$$

则：

$$
h'
=
(\gamma/S)\odot\hat x
+
(\beta/S)
$$

因此：

$$
\gamma'
=
\gamma/S
$$

$$
\beta'
=
\beta/S
$$

而 Linear Weight：

$$
W'=WS
$$

可保持整体 FP 数学等价。

## 37.21 PyTorch Folding

```python
def fold_smooth_scale_into_ln_linear(
    ln,
    linear,
    scale
):
    ln.weight.data.div_(scale)

    if ln.bias is not None:
        ln.bias.data.div_(scale)

    linear.weight.data.mul_(
        scale.unsqueeze(0)
    )
```

Q/K/V fan-out 时：

```python
ln.weight.data.div_(scale)
ln.bias.data.div_(scale)

q_proj.weight.data.mul_(scale.unsqueeze(0))
k_proj.weight.data.mul_(scale.unsqueeze(0))
v_proj.weight.data.mul_(scale.unsqueeze(0))
```

必须同步修改所有消费者。

## 37.22 Fan-out 为什么容易出 Compiler Bug

如果同一个 Tensor：

```text
H
├→ Q
├→ K
└→ V
```

Compiler 只在其中一支做了 scale folding，其他分支仍使用原 Tensor，则会破坏数学等价。

因此：

```text
branch
fan-out
residual
concat
```

都是 Scale Propagation / Folding 的重点风险区域。

## 37.23 Residual Quant IR

概念上：

```text
%x_q : int8, scale=sx
%f_q : int8, scale=sf

%x_aligned = requant %x_q
    {from=sx, to=sy}

%f_aligned = requant %f_q
    {from=sf, to=sy}

%y_q = add_int8
    %x_aligned,
    %f_aligned
```

某些 backend 也可能有原生：

```text
quantized_add
```

直接处理多套 scale。

## 37.24 Asymmetric Residual Add

如果：

$$
X=s_x(X_q-z_x)
$$

$$
F=s_f(F_q-z_f)
$$

则：

$$
Y_q
=
round
\left(
\frac{s_x}{s_y}(X_q-z_x)
+
\frac{s_f}{s_y}(F_q-z_f)
\right)
+z_y
$$

硬件复杂度明显高于 symmetric quantization。

因此内部 GEMM / residual path 常偏好 symmetric INT8。

## 37.25 Residual Debug Checklist

FakeQuant 正常但 NPU residual 后突然坏时，优先检查：

```text
1. Branch A scale
2. Branch B scale
3. Output scale
4. Zero-point
5. Requant multiplier
6. Requant shift
7. Rounding mode
8. Saturation
9. Add 顺序
10. Layout / channel mapping
```

## 37.26 典型 Requant Ratio Bug

正确：

$$
F'_q
=
round
\left(
F_q\frac{s_f}{s_y}
\right)
$$

若错误写成：

$$
F'_q
=
round
\left(
F_q\frac{s_y}{s_f}
\right)
$$

可能导致输出大面积 saturation。

这种错误非常适合通过：

```text
FakeQuant vs NPU
```

逐层比对定位。

## 37.27 Residual Debug 应记录的指标

```text
Residual Input RMS
Transform Output RMS

FP Branch Ratio
Quant Branch Ratio

Add Output Cosine
Add Output RMS
```

例如：

```text
FP:
X RMS = 4.0
F RMS = 1.0
Ratio = 0.25

Quant:
X RMS = 4.0
F RMS = 0.45
Ratio = 0.1125
```

即使：

```text
Output cosine = 0.995
```

也应警惕 branch energy drift。

## 37.28 面向 VLA 的评价维度

VLA 不应只看：

```text
Layer Cosine
```

还应加入：

```text
Attention Temperature Ratio
Residual Branch Energy Ratio
Action Head Output Scale
Temporal Error
```

形成：

```text
Numerical Similarity
+
Distribution Similarity
+
Structural Energy
+
Task Metric
```

四层评价框架。

## 37.29 核心结论

1. 不同 scale 的 INT8 Tensor 不能直接相加。
2. Residual Add 本质需要 Scale Alignment / Requant。
3. 大 Residual Branch 可能让小 Transform Branch 失去量化分辨率。
4. Residual Output Cosine 很高，也可能掩盖 Transform Branch 已经严重失真。
5. 深层网络要关注 Branch Energy Ratio 的系统性 Drift。
6. LayerNorm 能稳定 Transform Branch 输入，但不能自动修复 Residual Energy Mismatch。
7. SmoothQuant 的 scale 可以 fold 到 LayerNorm + QKV Weight，避免新增 runtime op。

---

# 38. 当前 xhquant / Spirit 工程映射与验证闭环

本节把前面的通用概念映射到当前 `/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt` 工程。这里的映射属于当前工程事实；是否能推广到其他 VLA，必须重新验证。

| 教材概念 | 当前工程位置 | 验证重点 |
|---|---|---|
| Calibration dummy | `ptq_dump.py:build_converter_dummies()` | 输入 shape、dtype、视觉 token 对齐 |
| 像素归一化与 patch 重排 | `hm_pixel_to_native_patches()` | `[0,255]` 与 native 输入范围、layout |
| KV cache | `make_empty_caches()`、`CacheTensor` | cache axis、层数、past length |
| 输入 dtype 对齐 | `clone_inputs_as()`、`prepare_hmonnx_llm_args()` | fp16/int32 转换是否符合 HMONNX 契约 |
| ONNX IO 对齐 | `onnx_graph_io_names()` | initializer 排除、输入名与 cache 识别 |
| HMONNX golden | `dump_hmonnx_golden()` | 输入 wrapping、输出保存、执行设备 |
| Native 对照 | `run_native_outputs()` | native 与 HMONNX 的输入是否真正等价 |
| FX 量化 | `run_fx_outputs()` | 仅对支持的组件尝试，失败要记录原因 |
| 数值比较 | `compare_named_arrays()` | shape、cosine、MSE、max_abs 同时判断 |

推荐的实验闭环：

```text
FP/PyTorch baseline
  → wrap baseline
  → FakeQuant 或 FX quant
  → ONNX
  → HMONNX
  → HMM 编译
  → NPU dump / runtime
  → First Bad Operator
  → action / closed-loop metric
```

必须区分四个结论：

1. **导出成功**：模型可以表示为 ONNX；
2. **转换成功**：ONNX 可以转换为 HMONNX；
3. **编译成功**：HMONNX 可以编译成目标 HMM；
4. **部署成功**：NPU 可以执行且精度、延迟、任务指标达标。

前一个结论成立，不代表后一个结论自动成立。对新 VLA，应先分别验证 vision、prefill、decode、action head，再验证完整闭环。

## 38.1 新模型支持判定表

**代码实现**：`ptq_dump.py` 的组件调度可以抽象为以下最小结构：

```python
components = ("vision", "prefill", "decode", "dit")
for component in components:
    onnx_file = find_component_onnx(output_dir, component)
    if onnx_file is None:
        continue
    inputs = dummies[component]["hmonnx"]
    post_dir, post_raw = dump_hmonnx_golden(
        component, onnx_file, device, inputs, work_dir
    )
    post_outputs = named_outputs(onnx_file, post_raw)
    save_build_named_npy(post_dir, prefix, post_outputs, "output")
```

这里的 `find_component_onnx()`、`dump_hmonnx_golden()`、`named_outputs()` 和 `save_build_named_npy()` 分别对应“发现子图、执行 HMONNX、按 ONNX 输出名整理结果、保存 golden”四个职责。新模型迁移时，优先保持这四个职责分离，避免把模型加载、输入构造、转换和指标计算写进一个大函数。

迁移其他 VLA 或其他模型时，按以下顺序检查：

```text
能否稳定导出 ONNX？
  → 算子是否在当前 xhquant/XH2a 支持范围？
  → shape、dtype、layout 是否满足编译约束？
  → 是否有代表性 calibration 数据？
  → HMONNX 是否能转换？
  → HMM 是否能编译？
  → NPU 输出是否与 reference 对齐？
  → action/任务指标是否达标？
```

当前脚本中的 Qwen3-VL、Spirit policy、DiT 和 KV cache 逻辑是业务封装，不应被理解为通用量化 API。底层 `QuantScheme`、`create_quant_config()` 和 `convert_onnx_to_hmonnx()` 可以复用，但输入构造、模型导出、cache、子图拆分和精度验证通常需要重写。

## 38.2 参考来源与版本记录

后续复盘时，建议为每个外部结论记录：论文/官方文档、访问日期、框架版本、工具链版本、目标硬件和实测命令。至少应补充以下来源类别：

- SmoothQuant、GPTQ、AWQ、QuaRot、DuQuant 原论文；
- PyTorch FX / `torch.export` / ONNX 导出文档；
- ONNX Quantization 文档；
- 当前 `xhquant`、`xh_model_zoo`、HMONNX、XH2a 编译器版本说明；
- 当前 Spirit-v1.5 工程的 README、`ptq.py`、`ptq_dump.py` 和 build 日志。

可直接复盘的公开入口：

- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [QuaRot](https://arxiv.org/abs/2404.00456)
- [DuQuant](https://arxiv.org/abs/2406.01721)
- [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html)
- [PyTorch `torch.export`](https://docs.pytorch.org/docs/stable/export.html)
- [ONNX Runtime quantization overview](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

没有实测或版本信息的陈述，应标注为“待实测”，不要写成确定的硬件支持结论。

# 39. NPU 解决方案工程师进阶学习清单

这一章是在前面量化、Compiler、Kernel、Profiler、VLM/VLA 学习基础上的进一步扩展路线。目标不是“每个方向都懂一点”，而是形成能够跨模型、编译器、Kernel、Runtime 与系统层定位问题的能力。

整体能力链路：

```text
模型结构
Transformer / VLM / DiT
        ↓
模型导出
PyTorch / FX / ONNX
        ↓
Compiler IR
Quant IR / Pass / Lowering
        ↓
Kernel
GEMM / Attention / Layout / Tile
        ↓
NPU Hardware
SRAM / DMA / MAC / Core
        ↓
Runtime
Prefill / Decode / KV Cache / Batching
        ↓
System
PCIe / Host / Device / Driver
        ↓
Profiler
Accuracy + Performance
        ↓
问题定位与解决
```

真正的技术壁垒不是在某一层比对应专家更深，而是：

> 能跨层判断问题究竟出在哪一层，并把证据链讲清楚。

## 39.1 S 级：Compiler IR / Graph Debug

### IR 基础
掌握 Graph、Node、Edge、Value、Tensor、SSA、Producer / Consumer、Topological Order、Static / Dynamic Shape、Constant / Parameter / Runtime Tensor、Operator Attribute、dtype、shape、stride、layout。

目标：看到 `%3 = matmul %1, %2`，能够判断输入来源、输出消费者、shape/dtype，以及是否带 quant metadata。

### torch.fx
掌握 GraphModule、Node、placeholder、call_module、call_function、call_method、get_attr、output，并至少亲自使用一次：

```python
symbolic_trace(model)
```

需要能回答 nn.Linear、F.relu、nn.ReLU、Parameter 在 FX Graph 中分别如何表示。

### torch.export
理解 Python Model → ExportedProgram → 更严格 Graph Representation 的关系。重点掌握 Dynamic Shape、Graph Constraint、Control Flow、Functionalization、Decomposition，不必现在深入 Dynamo 内部实现。

### ONNX
熟练阅读 Graph、Node、Input、Output、Initializer、Attribute、Opset、Domain。重点算子：

```text
MatMul / Gemm / Conv / Add / Mul / Div
Reshape / Transpose / Concat / Slice / Gather
ReduceMean / Softmax / LayerNormalization
Cast / QuantizeLinear / DequantizeLinear
```

### PyTorch → ONNX Mapping
理解 nn.Linear 可能导出为 MatMul + Add 或 Gemm，LayerNorm 可能是单算子也可能被分解。目标是看到 ONNX Graph 后能反推大致 PyTorch 结构。

### Graph Rewrite
理解 Pattern Matching、Rewrite、Canonicalization、Constant Folding、Dead Code Elimination、Common Subexpression。

### Fusion
重点理解 GEMM+Bias+Activation、Conv+BN+ReLU、DQ+MatMul+Bias+Requant。会分析 Fusion 失败原因：Shape、Layout、DType、Multiple Consumers、Dynamic Shape、Quant Scale、Unsupported Activation。

### Quant IR
熟练理解 Q/DQ、FakeQuant、Quantized Operator、Scale Propagation、Scale Folding、Requant、Mixed Precision Boundary。看到 qgemm、input_scale、weight_scale、output_scale、axis、group_size 能直接解释语义。

### Legalization
理解 Generic MatMul → NPU QuantizedGemm 需要满足 dtype、shape、axis、group_size、layout 等目标硬件约束，否则可能 fallback。

### Lowering
理解：

```text
PyTorch
↓
ONNX
↓
High-level IR
↓
Quant IR
↓
Hardware IR
↓
Kernel / Instruction
```

越往下，模型语义越少，硬件细节越多。

### Layout
掌握 NCHW、NHWC、Row Major、Column Major、Blocked Layout、Packed Layout，重点区分 Logical Shape 与 Physical Layout。

### 实战任务
1. TinyMLP 导出 ONNX。
2. Transformer Block 导出 ONNX。
3. 对比 PyTorch Module ↔ ONNX Node。
4. 人为加入 Unsupported Op。
5. 对比 Fusion 前后 Graph。
6. 阅读一次 Quant Graph 的 Q/DQ。

### 学完标准
模型转换失败时，能查哪个 Op/Shape 不支持、哪一步 decomposition 变复杂；INT8 没跑起来时，能查 QDQ、fallback、fusion、scale axis。

## 39.2 S 级：NPU Kernel / Performance

### GEMM
熟练：
$$
C_{M\times N}=A_{M\times K}B_{K\times N}
$$
$$
Ops\approx2MKN
$$

看到 X[B,T,K] 和 W[N,K]，能立即得到 M=B×T、K=hidden、N=output。

### GEMV vs GEMM
Decode：M≈1，更像 GEMV；Prefill：M≫1，大 GEMM。

### Tiling
深入理解 tile_M / tile_N / tile_K，以及 SRAM、MAC Array、Data Reuse、DMA 对 tile 的约束。

### SRAM / Buffer
能估算 A Tile + B Tile + INT32 Accumulator + Output + Scale + Bias + Double Buffer 是否能放入 SRAM。

### Dataflow
理解 Weight Stationary、Output Stationary、Input Stationary，本质是“谁留在片上复用，谁不断搬运”。

### MAC Array
理解 Matrix Engine、MAC Array、Vector Unit、Load/Store Unit、DMA Engine，并重点观察 Array Utilization。

### Tail Tile / Padding
理解尾块利用率和 Padding Overhead。

### DMA
掌握 DDR↔SRAM 搬运，以及 DMA Engine 与 Compute Engine 是否能 overlap。

### Double Buffer
理解理想时间：
$$
T\approx\max(T_{DMA},T_{Compute})
$$
而不是二者相加。

### Stall
区分 Memory Stall、Compute Stall、Dependency Stall、Sync Stall、Pipeline Bubble。

### Operator Fusion
理解 GEMM + Bias + Activation + Requant 融合为何能减少 Kernel Launch、Intermediate Tensor、Memory Read/Write 和 Synchronization。

### Quant Kernel
W8A8：INT8×INT8→INT32→Requant。
W4A16：Packed INT4 Weight→Unpack/Dequant→FP16 Activation × Weight。
W4A8：W4 Unpack + A8 + Group Scale + 低比特/混合累加。

### Weight Packing
理解 Logical Weight→Reorder→Tile→Pack；INT4 还涉及 2 weights/byte、scale、zero-point、group metadata。

### Roofline
熟练：
$$
AI=\frac{Ops}{Bytes}
$$
$$
P=\min(P_{peak},BW\times AI)
$$

给定 M/N/K、dtype、latency，能判断更像 memory-bound 还是 compute-bound。

### Kernel Benchmark
建议测试 M=1/4/16/64/256/512，观察 Latency、TOPS、Bandwidth、Utilization，建立 Decode→Prefill 性能直觉。

## 39.3 S/A 级：LLM Runtime / KV Cache

### Prefill / Decode
Prefill：整个 Prompt，一般 M 大、GEMM、compute-heavy。
Decode：每次生成一个 Token，一般 M≈1、GEMV-like、memory-heavy。

### KV Cache
理解历史 K/V 缓存，以及 Decode 中 new Q × all cached K。

### KV Cache Memory
能推导：
$$
Memory\approx2\times Layers\times SeqLen\times KVHeads\times HeadDim\times Bytes
$$
并估算 FP16 / INT8 / INT4 KV 在不同 context 下占用。

### MHA / MQA / GQA
理解它们的 KV Head 数量关系，以及对 KV Cache 大小和带宽的直接影响。

### Paged Attention
理解 Page/Block Allocation、Fragmentation、Page Table、KV Block Management。

### Continuous Batching
理解完成一个 Request 后立即插入新 Request，提高 Device Utilization。

### Request Scheduling
理解 Prefill 与 Decode 在 Throughput、Latency、Fairness、Memory 上的调度权衡。

### Chunked Prefill
理解长 Prefill 拆成 Chunk，与 Decode 交错执行以降低尾延迟。

### Prefix Cache
理解复用相同前缀已有 KV，从而减少重复 Prefill。

### Speculative Decoding
理解 Draft Model 预测多个 Token，Target Model 验证，从而减少 Target Model 串行 Decode 次数。

### KV Cache Quantization
重点研究 FP16 / INT8 / INT4 KV、Per-token / Per-head Scale，以及 Memory、Bandwidth、Attention Error、Runtime Cost。

### Runtime Debug
两个 Request 同时运行卡死时，考虑 Scheduler、Batch Shape、KV Allocation、Workspace、Stream Synchronization、Runtime Thread、NPU Queue。

## 39.4 A 级：Transformer / VLM / DiT 结构

### Transformer Attention
熟练 Q/K/V、Softmax(QK^T/√d)、AV。

### Multi-Head Attention
理解 Hidden→Split Heads→Per-head Attention→Concat→O Projection。

### RoPE
理解其是对 Q/K 做位置相关旋转，并知道它会影响 Q/K Layout、KV Cache、Long Context。

### RMSNorm vs LayerNorm
理解 LayerNorm 减 Mean 再除 Std，RMSNorm 不减 Mean，只按 RMS 归一化。

### SwiGLU
掌握 gate_proj、up_proj、SiLU(gate)*up、down_proj。

### VLM
理解 Vision Encoder→Projector→Vision Tokens + Text Tokens→LLM，以及 Cross-modal Distribution 与 Projector 的作用。

### ViT
掌握 Patch Embedding、Patch Token、Position Embedding、Vision Attention、MLP。

### DiT
理解 Noisy Action/Latent + Timestep + Condition→Transformer→Noise/Velocity/Flow Prediction。

### Diffusion / Flow Matching
理解 Diffusion 的加噪/去噪，以及 Flow Matching 学习 Velocity / Vector Field 的基本概念。

### Action Chunk
理解一次预测未来 N 步动作可降低 inference frequency、提高 temporal consistency，同时也可能带来 error accumulation。

## 39.5 A 级：VLA Quantization

固定拆成 Vision Encoder、Projector、VLM Backbone、Action Head 四层。

### Vision Calibration
覆盖 Camera、Lighting、Object Distribution、Viewpoint、Motion Stage。

### Multimodal Calibration
同时覆盖 Image、Instruction、Robot State、History。

### Modality Distribution Shift
比较 Vision/Text/State Token 的 Max、P99、RMS、Outlier Ratio。

### Projector Sensitivity
单独测试 FP16 / INT8 / INT4，并观察 Vision Feature→VLM Hidden 的误差。

### VLM Backbone Quant
结合 QKV、O、MLP、LN、Softmax、Residual，重点关注 Attention Temperature Drift 和 Residual Energy Drift。

### DiT Timestep Calibration
必须覆盖 Early / Middle / Late 多个 Timestep。

### Temporal Error Accumulation
比较 Single Denoise Step 与 Full Denoising Trajectory。

### Action Metrics
加入 Action L2、Position Error、Rotation Error、Gripper Accuracy。

### Closed-loop Evaluation
最终评价 Success Rate、Collision Rate、Completion Time、Trajectory Deviation。

### VLA Quantization 方法地图
```text
Activation Outlier
→ SmoothQuant / Rotation / DuQuant

Weight Error
→ GPTQ / AWQ

Attention Drift
→ Temperature Calibration

Residual Drift
→ Energy Calibration

Sensitive Layer
→ Mixed Precision

Temporal Error
→ Timestep-aware Calibration
```

### 最终报告
至少记录 Module、Operator、DType、Granularity、Observer、Tensor Cosine、Temperature Ratio、Residual Energy Ratio、Action Error、Task Success、Latency。

## 39.6 A 级：PCIe / 系统性能

### PCIe 基础
掌握 Root Complex、Endpoint、Switch、Lane、Link。

### Generation / Lane Width
理解 Gen3/4/5 与 x1/x4/x8/x16，并能判断实际 H2D/D2H 是否匹配链路能力。

### Link Training
至少理解 LTSSM、Negotiated Speed、Negotiated Width。

### BAR / MMIO
理解 Host 如何映射 Device 地址空间。

### DMA
理解 Host Memory→PCIe→Device Memory 的 DMA 搬运。

### H2D / D2H
区分 Host-to-Device 与 Device-to-Host 的带宽和延迟。

### Pinned Memory
理解 Page-locked Memory 为什么更适合 DMA。

### IOMMU
理解 Device DMA Address ↔ Physical Memory 的基本映射关系。

### Interrupt
至少知道 MSI / MSI-X 在设备完成任务通知 Host 中的作用。

### Driver / Runtime
理解：
```text
Application
↓
Runtime API
↓
Driver
↓
PCIe
↓
Device Firmware
↓
NPU
```

### PCIe Bandwidth Test
会测 H2D、D2H、Bidirectional Bandwidth、Small Transfer Latency、Large Transfer Bandwidth。

### Small vs Large Transfer
理解小传输 Latency-bound，大传输 Bandwidth-bound。

### Batch 与 PCIe
理解频繁 small copy + launch + copy back 会让固定开销占比很高，Batch 增大可以摊薄固定成本。

### Zero-copy / Shared Memory
理解 Zero-copy、Shared Memory、Unified Memory 的基本思想。

### PCIe Debug Checklist
```text
1. Link Speed
2. Link Width
3. H2D Bandwidth
4. D2H Bandwidth
5. Transfer Size
6. Pinned Memory
7. DMA Concurrency
8. Compute / Transfer Overlap
9. Synchronization
10. NUMA / CPU Placement
```

## 39.7 推荐学习顺序

```text
① Compiler IR / Graph Debug
        ↓
② NPU Kernel / Performance
        ↓
③ LLM Runtime / KV Cache
        ↓
④ Transformer / VLM / DiT
        ↓
⑤ VLA Quantization
        ↓
⑥ PCIe / System
```

更实际的组织：

```text
主线：
Compiler → Kernel → Runtime

辅线：
Transformer / VLM / DiT

专项：
VLA Quant

系统补充：
PCIe
```

## 39.8 四阶段学习计划

### 第一阶段：Compiler + Kernel
目标：能从模型 Graph 一路追到 NPU Kernel。

```text
FX
ONNX
IR
Pass
Fusion
Quant IR
Lowering
Layout

GEMM
Tile
SRAM
DMA
Roofline
Profiler
```

### 第二阶段：LLM Runtime
目标：能解释一个 LLM Request 到底如何在设备上执行。

```text
Prefill
Decode
KV Cache
GQA
Paged Attention
Continuous Batching
Scheduler
```

### 第三阶段：模型结构 + VLA
目标：能理解为什么某些模块特别敏感。

```text
Transformer
ViT
VLM
DiT
Flow Matching
Action Chunk
VLA Quantization
```

### 第四阶段：System / PCIe
目标：把性能分析从 NPU 芯片内部扩展到整机。

```text
Host
PCIe
DMA
Driver
Runtime
NPU
```

## 39.9 最终毕业标准

假设真实问题：

```text
某 7B 模型
W8A8 FakeQuant 精度正常
部署到 M.2 NPU 后精度下降 8%
Decode 比 FP16 只快 20%
```

应该拆成两条独立调查链。

### Accuracy Chain
```text
PyTorch FP
↓
ONNX FP
↓
FakeQuant
↓
Compiler IR
↓
NPU Dump
↓
First Bad Operator
↓
Scale / Axis / Requant / Layout / Kernel
```

### Performance Chain
```text
End-to-End
↓
PCIe or Model?
↓
Top Operator
↓
Actual Kernel
↓
M=1 Decode
↓
Roofline
↓
Weight Bandwidth
↓
W8 Kernel Utilization
↓
Tile / DMA / Packing
```

高质量结论示例：

```text
精度问题：
首次出现在第 17 层 Residual Add。
FakeQuant 正常，而 NPU Branch-B 出现明显 Saturation。
进一步检查发现 Per-Channel Requant Scale 顺序与 Weight Blocked Layout 不一致。

性能问题：
Decode GEMM 为 M=1，理论上 Memory-bound。
实际 DDR Bandwidth 已接近峰值，因此 W8A8 的主要收益来自 Weight Traffic 减少，
而不是 INT8 Peak Compute。
当前加速受 Memory Path 限制，应继续评估 W4A16 Native Kernel，
而不是继续优化 INT8 MAC 峰值。
```

如果能稳定做出这种分析，就已经具备较强的 NPU 解决方案 / 大模型部署技术壁垒。

---

# 40. 工程代码映射索引（技术增强附录）

**系统讲解**：本附录把全文最常出现的抽象概念收敛到当前 Spirit 工程的函数名，便于从教材回到源码。下表只表示当前代码中的映射，不表示其他模型必须采用相同实现。

| 抽象概念 | 当前代码 | 核心数据/结果 |
|---|---|---|
| 参数解析与静态校验 | `ptq_dump.py:parse_args()`、`validate_vision_grid()` | batch、context、image grid、device |
| 模型加载 | `load_wrap_graph_model()`、`load_spirit_policy()` | Qwen3-VL、Spirit policy |
| 视觉预处理 | `build_processor_inputs()` | `input_ids`、`pixel_values`、`image_grid_thw` |
| 视觉 embedding | `visual.forward_ori()` | `image_embeds`、deepstack embeddings |
| 多模态序列整理 | `Qwen3_VLDataPreprocess` | `inputs_embeds`、三维 position ids |
| KV cache 初始化 | `make_empty_caches()` | `CacheTensor` key/value lists |
| prefill/decode dummy | `build_converter_dummies()` | 分块序列/单 token 输入 |
| native 对照 | `run_native_outputs()` | PyTorch reference outputs |
| wrap 对照 | `run_module_outputs()` | wrap graph outputs |
| FX quant | `run_fx_outputs()` | FX converted model outputs |
| HMONNX golden | `dump_hmonnx_golden()` | saved input/output npy |
| 输出命名 | `onnx_graph_io_names()`、`named_outputs()` | ONNX IO name → numpy |
| 数值指标 | `cosine_similarity()`、`mse()`、`max_abs_diff()` | cosine/MSE/max_abs |
| 报告输出 | `write_cosine_report()`、`write_pair_report()` | TSV reports |
| Qwen-VL PTQ | `ptq.py:quantize_qwen3_vl()` | vision/prefill/decode HMONNX |
| DiT PTQ | `ptq.py:quantize_dit()` | ONNX → HMONNX DiT |

## 40.1 代码块来源：绝对路径与行号

**代码实现**：下面是本教材中工程映射代码块对应的当前容器源码位置。行号是本次读取容器时的源码行号；如果远端代码发生修改，应重新用 `nl -ba` 或 IDE 行号校验。

### `ptq_dump.py`

```text
/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:37
  parse_args()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:121
  validate_vision_grid()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:348
  cosine_similarity()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:360
  mse()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:369
  max_abs_diff()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:636
  build_quant_scheme()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:736
  build_dit_dummies()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:770
  hm_pixel_to_native_patches()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:835
  make_empty_caches()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:853
  build_processor_inputs()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:888
  build_converter_dummies()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1009
  wrap_vision_model()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1023
  wrap_llm_model_for_graph()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1038
  run_module_outputs()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1049
  run_native_outputs()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1103
  run_fx_outputs()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1119
  dump_hmonnx_golden()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1139
  dump_and_stage_golden()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq_dump.py:1346
  main()
```

### `ptq.py`

```text
/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:49-59
  HOUMO_TARGET / MODEL_FOLDER

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:189
  quantize_qwen3_vl()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:238-252
  MatMul QuantScheme 与 XH2a 配置

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:272-278
  xhquant_init() 与 LLMConverter.from_pretrained()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:312
  quantize_dit()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:354-368
  torch.onnx.export() 导出 DiT

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:380-394
  DiT QuantScheme 与 convert_onnx_to_hmonnx()

/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt/ptq.py:414
  main()
```

### `xhquant` 安装包

```text
/opt/venv/dadao/lib/python3.12/site-packages/xhquant/__init__.py
  包入口

/opt/venv/dadao/lib/python3.12/site-packages/xhquant/api/quant_type.py:51
  class QuantScheme

/opt/venv/dadao/lib/python3.12/site-packages/xhquant/api/quant_type.py:210
  create_quant_config()

/opt/venv/dadao/lib/python3.12/site-packages/xhquant/api/ptq_export_hmonnx.py:302
  convert_onnx_to_hmonnx()

/opt/venv/dadao/lib/python3.12/site-packages/xhquant/xhonnxruntime/hmonnx_inference.py:1012
  class HMONNXGoldenInference

/opt/venv/dadao/lib/python3.12/site-packages/xhquant/core/cache_tensor.py:8
  CacheTensor
```

**代码实现**：容器内可以用下面的命令重新确认 `xhquant` 的真实路径和行号：

```bash
python3 -c "import inspect, xhquant; from xhquant.api import QuantScheme, create_quant_config, convert_onnx_to_hmonnx; print(xhquant.__file__); print(inspect.getsourcefile(QuantScheme), inspect.getsourcelines(QuantScheme)[1]); print(inspect.getsourcefile(create_quant_config), inspect.getsourcelines(create_quant_config)[1]); print(inspect.getsourcefile(convert_onnx_to_hmonnx), inspect.getsourcelines(convert_onnx_to_hmonnx)[1])"
```

**代码实现**：阅读任意一个新模型时，可以先建立同样的最小接口：

```python
class ModelQuantAdapter:
    def build_inputs(self, sample): ...       # representative inputs
    def export_onnx(self, model, inputs): ... # graph contract
    def build_quant_config(self): ...         # dtype/granularity/device
    def run_reference(self, inputs): ...      # FP/wrap baseline
    def run_hmonnx(self, inputs): ...         # post-quant output
    def compare(self, ref, test): ...         # shape + numeric metrics
```

**复盘原则**：如果一个量化问题无法定位，应把它拆成三条独立证据链：

```text
数学链：scale / clipping / rounding / requant 是否正确？
图链：ONNX / QDQ / HMONNX / layout 是否保持语义？
硬件链：kernel / DMA / fallback / runtime 是否按预期执行？
```

只有三条链都闭合，才能把结论从“理论上应该支持”提升为“当前版本、当前模型、当前硬件上已经验证支持”。






