# 生产功能第3课：SmoothQuant — W8A8 的数学等价变换

> 学习路线：P0 必学 — 量化推理全链路 > 1.4 SmoothQuant
> 前置：量化基础 (生产_01)，GPTQ (生产_02)，SwiGLU (第3课)，RMSNorm (第1课)
> 核心论文：SmoothQuant (Xiao et al., ICML 2023)

---

## 一、问题：激活值的 Outlier 通道

### 1.1 什么是 Outlier

在第1课 RMSNorm 中学过，RMSNorm 把每个 token 的 RMS 归一化到 1。但它不约束单个通道的最大值。

```python
# 激活值 X: (batch, seq_len, d_model) = (1, 10, 4096)
# 某个 token 的 4096 维向量:
[0.2, 0.5, -0.3, 1.2, 0.1, ..., 58.7, 0.4, ..., -32.1, 0.8, ...]
                          ↑ channel_j       ↑ channel_k

# RMS ≈ 1.0 ✓ (RMSNorm 保证了这一点)
# 但 channel_j = 58.7 和 channel_k = -32.1 是 outlier!
```

### 1.2 Outlier 的来源

SwiGLU 学过的 `silu(gate) * up`：

```python
# 第3课 SwiGLU.forward:
gate_up = down_proj(x)       # (bs, seq, 2*intermediate)
gate, up = gate_up.chunk(2)
out = silu(gate) * up

# silu 函数: x * sigmoid(x)
# 对于大的正 x: sigmoid(x)≈1 → silu(x)≈x（线性放大）
# → 如果某个 gate 通道本身就大，silu 会把它进一步放大 → outlier
```

### 1.3 为什么 Outlier 让 W8A8 困难

```python
# W8A8 需要对激活值做 per-tensor 量化（计算效率最高）
# 但 outlier 的存在让 per-tensor scale 变得极小:

X = [0.2, 0.5, 58.7, -0.3, 1.2, -32.1, ...]
max_abs = 58.7
scale = 58.7 / 127 = 0.46

INT8 = round([0.2, 0.5, 58.7, ...] / 0.46)
     = [0, 1, 127, -1, 3, -70, ...]
# 0.2 → 0 (被吞掉了!)
# 大部分值在这个 scale 下精度极低
```

**GPTQ 不需要解决这个问题**（GPTQ 只量化权重，激活保持 FP16）。

---

## 二、SmoothQuant 的核心 Trick：数学等价变换

### 2.1 插入对角矩阵

```
原始: Y = X @ W

插入 diag(s) @ diag(s⁻¹) = I (单位矩阵):
Y = X @ diag(s) @ diag(s⁻¹) @ W

重写:
Y = (X @ diag(s)) @ (diag(s⁻¹) @ W)
  = X_smooth   @   W_smooth

数学上完全等价 → Y 不变
但量化的难度变了!
```

### 2.2 具体数值示例

```python
# 假设 d_model=4, W=(4,4)
X = [[0.2, 0.5, 58.7, -0.3]]    # channel_2 是 outlier
W = [[1.0, 0.5, 1.5, 0.8],      # W[0,:]
     [2.0, 1.0, 0.5, 1.2],      # W[1,:]
     [0.5, 1.5, 0.8, 1.0],      # W[2,:]
     [1.2, 0.3, 2.0, 0.5]]      # W[3,:]

# 直接量化 → X 有 outlier → 精度差

# SmoothQuant 变换:
s = [0.5, 0.8, 8.0, 0.6]        # s[2]=8.0，因为 channel_2 是 outlier

X_smooth = X / s = [[0.4, 0.625, 7.34, -0.5]]
# channel_2: 58.7/8.0 = 7.34  ← outlier 被"平滑"了

W_smooth = W * s.unsqueeze(-1)
# W_smooth[2,:] = [0.5, 1.5, 0.8, 1.0] * 8.0 = [4.0, 12.0, 6.4, 8.0]
# channel_2 的权重被"放大"了 ← 但权重可以做 per-channel scale!

# 验证: X_smooth @ W_smooth = X @ W ✓
```

### 2.3 为什么把困难转移给权重就没事？

```
激活值量化:
  每层 forward 时动态产生 → 必须 per-token scale（每次计算 scale）
  outlier 让 per-tensor scale 不准 → 需要 per-token，但矩阵乘不方便

权重量化:
  编译时固定 → 可以做 per-channel scale（离线算好，写入 .hmm）
  per-channel scale 天然能处理每行尺度不同的情况
```

---

## 三、平滑因子 s 的计算

### 3.1 公式

```python
# s[j] = max(|X[:, j]|)^α / max(|W[j, :]|)^(1-α)
# 其中:
#   X[:, j]: 激活值第 j 列的绝对值最大值（统计自校准数据）
#   W[j, :]: 权重第 j 行的绝对值最大值
#   α: 迁移强度参数

def compute_smooth_factor(X_abs_max, W_abs_max, alpha=0.5):
    """
    X_abs_max: (in_features,)  每列激活值的最大绝对值
    W_abs_max: (in_features,)  每行权重的最大绝对值
    """
    s = (X_abs_max ** alpha) / (W_abs_max ** (1 - alpha))
    return s
```

### 3.2 α 的含义

```python
α = 0: s = 1 / W_abs_max
       → X 不变，W 完全按 W_abs_max 缩放
       → 所有迁移给权重

α = 1: s = X_abs_max
       → X 完全按 X_abs_max 缩放，W 不变
       → 所有迁移给激活

α = 0.5:  √(X_abs_max / W_abs_max)
         → 均分迁移

# 论文推荐 α ≈ 0.5 → 效果好且稳定
```

### 3.3 完整算法

```python
def smoothquant(model, X_calib, alpha=0.5):
    """
    model:  待量化的模型
    X_calib: 校准激活值 (n_samples, seq_len, in_features)
    alpha:  平滑迁移强度
    """

    # 1. 统计激活值范围
    X_abs_max = X_calib.abs().amax(dim=(0, 1))[:]  # (in_features,)

    # 2. 对每个 Linear 层做 SmoothQuant
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # 统计这层权重的范围
        W_abs_max = module.weight.abs().amax(dim=1)  # (in_features,)

        # 计算平滑因子
        s = (X_abs_max ** alpha) / (W_abs_max ** (1 - alpha) + 1e-8)

        # 变换: 权重乘 s，激活除 s
        module.weight.data = module.weight.data * s.unsqueeze(1)

        # 对于下一个 Linear 层，X 已经变成了 X/s
        # 所以下一个层对应的 X_abs_max 需要更新:
        X_abs_max = X_abs_max / s

    return model
```

**`X_abs_max = X_abs_max / s` 这行很关键**：

```python
# 第一层 Linear 的输入是原始 X
# 变换后 X 变成了 X/s → 下一层 Linear 的输入是 X/s
# → 下一层 Linear 的 X_abs_max 需要用 X_abs_max/s 来计算

# 这是 SmoothQuant 的"迁移传播"：每层溢出一点到权重，激活逐步变温和
```

---

## 四、SmoothQuant vs GPTQ 对比

| | GPTQ | SmoothQuant |
|---|---|---|
| **量化对象** | 权重 (W4A16) | 权重+激活 (W8A8) |
| **核心手段** | Hessian 矩阵补偿 | 对角矩阵变换 |
| **数学等价性** | 否（近似）| 是（严格等价） |
| **需要校准数据** | 是（用于算 Hessian） | 是（用于算 X_abs_max） |
| **矩阵乘效率** | INT4 @ FP16 → FP16 | INT8 @ INT8 → INT32 (快 2×) |
| **适用算子** | Linear | Linear, Attention (QKV Proj) |
| **实现复杂度** | 高 (Cholesky + Schur 补) | 低 (逐元素乘除) |

---

## 五、SmoothQuant 对已学算子的适配

### 5.1 Linear — 直接适用

```python
# 对每个 nn.Linear 层:
# QKV Proj:    W_qkv → W_qkv * s.T → Y = (X/s) @ (W*s)  不变
# FFN gate_up: W_gate_up → W_gate_up * s.T
# FFN down:    W_down → W_down * s.T
# Output Proj: W_o → W_o * s.T
```

### 5.2 Attention — Q·K^T 需要特殊处理

```python
# Q = X @ W_q, K = X @ W_k, V = X @ W_v

# SmoothQuant 变换后:
# Q_s = X/s @ (W_q * s.T) = Q   ← Q 不变! 因为 (X/s)@(W_q*s)=X@W_q
# K_s = K                       ← K 也不变
# V_s = V                       ← V 也不变

# Q·K^T 的 scale = 1/sqrt(d_k) 不变
# → Attention 的输出也不变
# → 不需要特殊处理! SmoothQuant 是严格的数学等价变换
```

### 5.3 RMSNorm — 不参与量化

```python
# RMSNorm 在 SmoothQuant 变换中不变
# RMSNorm 的计算用 FP16/FP32 → 不受量化影响
# SmoothQuant 只改变 Linear 层的输入输出分布
```

### 5.4 SwiGLU — gate/up 的融合处理

```python
# nano-vllm Qwen3MLP (第3课):
gate_up = gate_up_proj(x)        # MergedColumnParallelLinear
x = SiluAndMul(gate_up)
out = down_proj(x)

# SmoothQuant:
# gate_up_proj: W_gate_up → W_gate_up * s.T
# down_proj: W_down → W_down * s.T  
# SiluAndMul (silu(gate) * up): 逐元素操作，不涉及权重 → 不受影响
```

---

## 六、W8A8 在推理时的实际计算

```python
# 反量化流程:
# 1. 输入 X 已经过 SmoothQuant 变换 + 量化
X_int8 = quantize(X_smooth)            # per-token scale_x
W_int8 = quantize(W_smooth)            # per-channel scale_w

# 2. INT8 矩阵乘
Y_int32 = X_int8 @ W_int8.T            # INT8@INT8 → INT32 累加器

# 3. 反量化
Y_fp = Y_int32.float() * scale_x.unsqueeze(-1) * scale_w

# 4. 下一个 SmoothQuant 层的输入已是变换后的 X_smooth
# → 链式传播，不需要额外的 de-smooth 步骤
```

---

## 七、Houmo 的对应

```python
# Helion kernel 支持 W8A8:
# silu_mul_fp8  → SwiGLU 的 W8A8 融合算子
# rms_norm_fp8  → RMSNorm (norm 本身用 FP32, weight 是 FP8)

# Houmo PTQ 流程 (ptq.py):
# 1. 收集校准数据 → 统计每层 X_abs_max
# 2. 计算 smooth factor → 转变换后的权重
# 3. 做 W8A8 量化 (per-channel weight + per-token activation)
# 4. 编译为 Helion FP8 kernel → .hmm
```

---

## 八、思考题

1. **α=0 和 α=1 分别意味着什么？在实际应用中哪个更可行？**
   - 提示：α=0 → 全部推给权重（GPTQ 已证明可行）。α=1 → 全部推给激活（但 activation outlier 就是原始问题）

2. **SmoothQuant 和 GPTQ 能组合使用吗？**
   - 提示：理论上可以 — 先 SmoothQuant 变换、再对变换后的权重做 GPTQ 量化（W4A8?）。但工程上很少有人这么组合

3. **如果模型只用 RMSNorm + Linear + SwiGLU（没有 BatchNorm），SmoothQuant 的变换是否需要考虑 RMSNorm 的 weight？**
   - 提示：RMSNorm 的 weight 是 (d_model,) 向量 → 相当于 per-channel scale → 已经自带平滑效果。但它的作用范围和 SmoothQuant 的 s 不同
