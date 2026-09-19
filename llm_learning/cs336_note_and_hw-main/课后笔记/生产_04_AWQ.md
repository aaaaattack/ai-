# 生产功能第4课：AWQ — 激活感知的权重量化

> 学习路线：P0 必学 — 量化推理全链路 > 1.3 AWQ
> 前置：量化基础 (生产_01)，GPTQ (生产_02)，SmoothQuant (生产_03)
> 核心论文：AWQ (Lin et al., MLSys 2024)

---

## 一、核心洞察：只有 1% 的权重通道是"显著的"

### 1.1 AWQ 的观察

对 LLaMA 模型的每一层做了实验统计：

```
权重矩阵 W: (in_features, out_features) = (4096, 4096)

直接 per-channel 量化 (group_size=-1):
  精度损失: 很大（perplexity 升高 ~0.5）

逐通道观察:
  ~99% 的通道: 量化后对输出影响 < 1%
  ~1% 的通道（"salient channels"）: 量化后输出变化明显

  这些显著通道恰好对应激活值的 outlier 通道
```

### 1.2 这和 SmoothQuant 的观察一致

```
SmoothQuant: 激活值有一些 outlier 通道 → 量化困难
AWQ:         权重有一些显著通道 → 量化精度损失的主要来源

两者说的是同一个物理现象:
  - 某些 feature dimension 的激活值天然大
  - 对应的线性层权重列对输出贡献大
  - 这些列的权重量化误差会被放大
```

### 1.3 AWQ 的解法 vs GPTQ 的解法

```
GPTQ:
  "所有列都可能产生误差 → 逐列补偿，用 Hessian 分配误差"
  复杂度: 高 (需要逐列计算 + Schur 补更新)
  精度: 极高

AWQ:
  "只有 1% 的列是关键 → 找到它们，给它们更好的 scale"
  复杂度: 低 (只需要一次 activation 统计 + scale 搜索)
  精度: 接近 GPTQ，但实现简单得多
```

---

## 二、核心思路：重要通道用更大的量化 scale

### 2.1 基本公式

```python
# 普通 per-channel 量化:
scale_j = abs(W[:, j]).max() / 7           # 第 j 列一个 scale
W_int4_j = round(W[:, j] / scale_j) * scale_j

# AWQ: 重要通道的 scale 乘以放大因子
s_j = f(activation_importance_j)            # 重要通道: s_j > 1
scale_j = abs(W[:, j]).max() / 7 * s_j     # scale 放大
W_int4_j = round(W[:, j] / scale_j) * scale_j
```

**为什么放大 scale 能提高精度？**

```python
# 假设 W[:, j] 的值域为 [0.1, 0.5, 0.8]:
普通 scale = 0.8/7 = 0.114
INT4: [1, 4, 7] * 0.114 = [0.114, 0.456, 0.798]
误差: [0.014, -0.044, -0.002]  ← 对重要通道来说误差偏大

AWQ s_j = 2:
scale = 0.114 * 2 = 0.228
INT4: [0, 2, 3] * 0.228 = [0, 0.456, 0.684]
误差: [-0.1, -0.044, -0.116]  ← 但这是不重要通道可以接受

# 直觉: 缩小了 INT4 的表示范围 (0-3 vs 1-7)
#       → 牺牲了不重要通道的精度
#       → 换来了重要通道的更好精度
```

### 2.2 为什么"牺牲不重要通道"可以接受？

```
权重矩阵中，不重要通道的输出贡献本来就小:
  Y = Σⱼ X[:, j] * W[:, j]

  如果 X[:, j] 小（不重要通道），W[:, j] 的量化误差对 Y 的影响也小
  如果 X[:, j] 大（重要通道），W[:, j] 的量化误差对 Y 的影响大

→ 保护重要通道，牺牲不重要通道 → 总输出误差降低
```

---

## 三、算法流程

### 3.1 Step 1: 统计激活值的重要程度

```python
def compute_activation_importance(X_calib):
    """
    X_calib: 校准数据 (n_samples, seq_len, in_features)
    返回每列的激活值平均幅度
    """
    # 第一步: 取绝对值 → 沿 sample 和 seq 维求平均
    X_abs_mean = X_calib.abs().mean(dim=(0, 1))   # (in_features,)

    # 可选: 对不同量化组独立统计
    # (实际实现中，每个 group_size 的组有自己的统计)
    return X_abs_mean
```

### 3.2 Step 2: 搜索最优 scale 放大因子

```python
def search_scale(X_abs_mean, alpha=0.5):
    """
    alpha 控制放大强度:
      0:  不放大（退化为普通 per-channel）
      0.5: 温和放大
      1:   最大放大
    """
    # 归一化: 平均幅度 / 全局平均 → 相对重要程度
    s_candidate = (X_abs_mean / X_abs_mean.mean()) ** alpha

    # 可选: 对 s_candidate 做网格搜索，找到最优的离散 scale
    # 或者直接用连续值
    return s_candidate
```

### 3.3 Step 3: 量化权重

```python
def awq_quantize(W, s, group_size=128):
    """
    W: 权重 (in_features, out_features)
    s: 放大因子 (in_features,)
    """
    W_q = torch.zeros_like(W)

    for g_start in range(0, W.shape[0], group_size):
        g_end = min(g_start + group_size, W.shape[0])

        for j in range(W.shape[1]):
            group = W[g_start:g_end, j]
            s_j = s[g_start:g_end]                  # 这组内的放大因子

            # per-group scale
            scale = group.abs().max() / 7            # INT4 max = 7

            # 重要通道: scale × s_j (放大了)
            # 不重要通道: scale × s_j (s_j≈1，不放大)
            effective_scale = scale * s_j

            W_q[g_start:g_end, j] = (
                torch.round(group / effective_scale)
                .clamp(-8, 7)                        # INT4 范围
                * effective_scale                     # 反量化
            )

    return W_q
```

---

## 四、AWQ vs GPTQ vs SmoothQuant 三合一对比

| | GPTQ | AWQ | SmoothQuant |
|---|---|---|---|
| **量化目标** | W4A16 | W4A16 | W8A8 |
| **核心 trick** | Hessian 矩阵补偿 | 重要通道 scale 放大 | 对角矩阵数学等价变换 |
| **数学等价性** | 近似 | 近似 | **严格等价** |
| **需要校准数据** | 是（算 Hessian） | 是（统计激活值） | 是（统计 X_abs_max） |
| **校准数据量** | 128-256 条 | 128-256 条 | 128-256 条 |
| **计算复杂度** | 高（O(n³) 或 O(n²) 优化后） | 低（一次 activation 统计 + scale 搜索） | 极低（逐元素乘除） |
| **耗时 (8B模型)** | 1-4 小时 | **几秒到几分钟** | **< 1 秒** |
| **精度** | 极高 | 接近 GPTQ | W8A8 精度好 |
| **实现难度** | 高 (Cholesky + Schur) | 低 (统计 + 搜索) | 极低 (逐元素) |
| **扩展性** | 难（需要 Hessian，大矩阵时慢） | 易（scale 搜索可并行） | 易 |
| **Houmo 采用** | GPTQModel ✅ | 未知 | Helion FP8 kernel 隐含 |

### 为什么 AWQ 这么简单但效果这么好？

```
1. 精准定位了问题 (只 1% 通道是关键)
2. 解法直接 (放大重要通道的 scale)
3. 不需要 Hessian (GPTQ 的核心复杂性来源)
4. 不需要改动模型结构 (SmoothQuant 需要修改权重值)

类似: 
  GPTQ 是用精密手术刀做逐列修复
  AWQ 是用"哪个通道重要"这个简单规则做粗调
  SmoothQuant是用数学魔术直接消除问题
```

---

## 五、对已学算子的影响

### 5.1 Linear — 直接适用

```python
# 和 GPTQ 的适用方式相同:
# 对每个 nn.Linear 层做 AWQ 量化
for layer in [q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj]:
    W_int4 = awq_quantize(layer.weight, activation_importance, group_size=128)
```

### 5.2 SwiGLU — 合并权重后还是分开量化？

```python
# gate_up_proj = MergedColumnParallelLinear (第3课)
# gate 和 up 合并为一个矩阵 → AWQ 的激活统计应该分开处理吗?

# 推荐: 分开处理
gate_importance = compute_activation_importance(gate_act)
up_importance   = compute_activation_importance(up_act)
# 分别量化 gate 和 up 部分 → 精度更好

# 简化: 合并处理（用 gate_up 的激活值一起统计）
# 实现更简单但精度略差
```

### 5.3 RMSNorm / RoPE — 不适用

```
AWQ 只量化权重 → 对 RMSNorm 和 RoPE 的 cos/sin cache 不适用
这些用 FP16 保持精度即可
```

---

## 六、Houmo 的 AWQ 实践

```python
# hmodel/gptqmodel/ 中有 AWQ 的引用:
# tests/test_awq.py → AWQ 测试

# 推断: Houmo 同时支持 GPTQ 和 AWQ 两种 W4 量化方案
# 两者产出的量化权重可以在编译阶段互换

# 量化流水线:
# PyTorch model → AWQ scale search → 量化权重
#               → Helion INT4 kernel 编译 → .hmm
#               → GGUF 打包 → 推理
```

---

## 七、思考题

1. **AWQ 和 SmoothQuant 都是"转移量化困难"，但方向不同。分别是什么方向？**
   - 提示：AWQ 把困难从重要通道转移到不重要通道（列→列）。SmoothQuant 把困难从激活转移到权重（激活→权重）

2. **如果 AWQ 的 scale 放大因子 s_j 太大，会导致什么？**
   - 提示：s_j 太大 → INT4 表示范围被压缩到很小 (如只有 0-3) → 这列的表示精度反而变差

3. **AWQ alpha=0 和 alpha=1 分别退化成什么？**
   - 提示：alpha=0 → s 全为 1 → per-channel 量化。alpha=1 → s ≈ X_abs_mean → 完全按激活重要程度分配
