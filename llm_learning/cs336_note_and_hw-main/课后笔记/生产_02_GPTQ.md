# 生产功能第2课：GPTQ — 逐列量化的 Hessian 补偿

> 学习路线：P0 必学 — 量化推理全链路 > 1.2 GPTQ
> 前置：量化基础 (生产_01)，矩阵乘法基础 (第5课)
> 关联工作：Houmo GPTQModel (`hmodel/gptqmodel/`)

---

## 一、问题：为什么不能直接四舍五入？

### 1.1 最简单的量化

```python
# 最粗暴的方法：全矩阵统一 scale，直接 round
scale = W.abs().max() / 7       # INT4: [-8, +7] → 7 是 max
W_int4 = torch.round(W / scale)
# 精度损失: 非常大，模型基本不可用
```

### 1.2 为什么误差会累积

```python
Y = X @ W
# W: (in_features, out_features) = (4096, 4096)

# 如果直接量化 W 的所有列:
Y_quant = X @ W_quant
error = Y - Y_quant
# 4096 列量化误差全部叠加到 Y 上 → 误差巨大

# 如果有办法让后续列"吸收"前面列产生的误差...
```

### 1.3 GPTQ 的直觉

```
不是"一次性量化全部列"
而是"量化第 0 列 → 马上把误差补偿到第 1..4095 列"
     "量化第 1 列 → 马上把误差补偿到第 2..4095 列"
     ...

这样，前面列产生的误差被后面的列逐步吸收掉。
最后一个输出 Y 的误差就会小很多。
```

---

## 二、Hessian 矩阵：误差分配的依据

### 2.1 什么是 Hessian？

```python
# 模型的损失函数 L(W) = ||Y - X@W||²

# Hessian = 损失函数对权重的二阶导数
# H[i,j] = ∂²L / ∂W[i] ∂W[j]

# 对于线性层: H = 2 × X^T @ X
# 来源: L(W) = ||Y - XW||² → 对 W 求两次导 → 2 × X^T X

H = 2 * X.T @ X           # X: (calib_samples, in_features)
# H: (in_features, in_features) ← 和输入维度平方成正比！
```

### 2.2 Hessian 的物理意义

```
H[i,i] 大 → 第 i 个输入维度对损失贡献大 → 重要维度
H[i,i] 小 → 第 i 个输入维度对损失贡献小 → 不重要维度

量化误差补偿的原则:
  - 重要维度的权重 → 少让它们吸收误差（保持精度）
  - 不重要维度的权重 → 多让它们吸收误差
```

### 2.3 为什么不用单位矩阵来补偿？

```python
# 如果 H = I (单位矩阵):
W[:, j+1:] += err                            # 所有剩余列均分误差
# 没区分"重要维度"和"不重要维度"

# 如果用 H⁻¹:
W[:, j+1:] += (err / H⁻¹[j,j]) * H⁻¹[j, j+1:]   
# 按重要性加权分配误差 → 误差集中在不重要列
```

---

## 三、GPTQ 算法逐步解析

### 3.1 准备工作

```python
def gptq(W, X_calib, group_size=128):
    """
    W:       权重矩阵 (in_features, out_features) = (4096, 4096)
    X_calib: 校准数据 (n_samples, in_features) → 从训练集随机取 128 条
    group_size: per-group 量化的组大小
    """
    n_columns = W.shape[1]                         # out_features = 4096

    # Step 1: 计算 Hessian 矩阵
    # L(W) = ||Y - XW||² → H = 2 * X^T @ X
    X = X_calib.float()                            # FP32 精度
    H = 2 * X.T @ X                                # (4096, 4096)

    # Step 2: Cholesky 分解（比直接求逆更稳定）
    # H⁻¹ 在后续循环中会逐列更新
    H_inv = torch.linalg.cholesky(H)               # 初始化为下三角 Cholesky 因子
    H_inv = torch.cholesky_inverse(H_inv)          # 转成 H⁻¹
```

**为什么是 `2 * X^T @ X`？**

```python
# L(W) = ||Y - XW||²_F = Σ_i ||Y_i - X_i @ W||²
# ∂L/∂W = -2 X^T (Y - XW)
# ∂²L/∂W² = 2 X^T X     ← Hessian 是常数！不依赖 W！

# 因此对线性层来说，Hessian = 2 * X^T X，非常简洁
# 对非线性层（如 Attention），这种简化不成立 → GPTQ 主要用于 Linear 层
```

### 3.2 主循环：逐列量化 + 补偿

```python
    # Step 3: 逐列量化主循环
    W_q = W.clone()          # 拷贝一份，后续会直接修改
    Q = torch.zeros_like(W)  # 量化后的权重

    for j in range(n_columns):
        # 3a. 量化第 j 列（用 per-group 量化）
        for g_start in range(0, W.shape[0], group_size):
            g_end = min(g_start + group_size, W.shape[0])
            group = W_q[g_start:g_end, j]
            scale = group.abs().max() / 7           # INT4 max = 7
            Q[g_start:g_end, j] = torch.round(group / scale).clamp(-8, 7) * scale

        # 3b. 计算第 j 列的量化误差
        误差 = Q[:, j] - W[:, j]                     # (in_features,)
        # 为什么 Q - W 而不是 W - Q?
        # 后面用 H⁻¹ 补偿时方向是: W += H⁻¹ * err
        # W[j+1:] += H⁻¹[j+1:, j] * (err / H⁻¹[j,j])
        # 等价于: 把量化导致的损失"推给"后续列

        # 3c. 用 Hessian 逆矩阵补偿后续列
        补偿因子 = 误差 / H_inv[j, j]                 # 标量归一化
        W_q[:, j+1:] -= H_inv[j+1:, j].unsqueeze(0) * 补偿因子

        # 3d. 更新 H⁻¹: 删除第 j 行 j 列后的 Schur 补
        # (数学上等价于固定第 j 列不变，重新求剩余列的 H⁻¹)
        H_inv = H_inv[1:, 1:] - H_inv[1:, 0:1] @ H_inv[0:1, 1:] / H_inv[0, 0]
```

### 3.3 Schur 补更新的数学含义

```python
# H⁻¹ 的逐列更新:
# 量完了第 j 列 → 这列不再改变 → 从 H⁻¹ 中移除
# 剩余 (n-j-1) × (n-j-1) 的 H⁻¹ 可以用公式更新，不需要重算

# 直观理解:
#   量完一列后，后续列的"重要程度"会变化
#   (因为第 j 列的误差已经推给它们了 → 它们的权重变了 → Hessian 变了)
#   Schur 补正好公式化地描述了这种变化
```

---

## 四、完整算法伪代码（简化版）

```python
def gptq_simplified(W, X_calib, group_size=128):
    """
    简化版 GPTQ，忽略 Cholesky 和 Schur 补的数值细节
    """
    n_columns = W.shape[1]
    W_q = W.clone()
    H = 2 * X_calib.T @ X_calib                    # Hessian
    H_inv = torch.inverse(H)                         # 初始 H⁻¹

    result = torch.zeros_like(W)

    for j in range(n_columns):
        # 1. 量化第 j 列 (per-group, group_size=128)
        result[:, j] = quantize_per_group(W_q[:, j], group_size)

        # 2. 计算误差
        err = (result[:, j] - W[:, j]) / H_inv[j, j]

        # 3. 补偿后续列
        for k in range(j+1, n_columns):
            W_q[:, k] -= err * H_inv[j, k]

        # 4. 更新 H⁻¹ (Schur 补)
        # ... 省略数值细节 ...

    return result
```

---

## 五、GPTQ vs 其他量化方案对比

| 方案 | 量化对象 | 精度策略 | 复杂度 | 代表实现 |
|---|---|---|---|---|
| **Round-to-nearest** | 权重 | per-channel scale | O(1) | 基线 |
| **GPTQ** | 权重 | 逐列 + Hessian 补偿 | O(n³) 或 O(n²) 优化后 | HuggingFace `auto-gptq` |
| **AWQ** | 权重 | 激活重要性 scale 搜索 | O(n) | `llm-awq` |
| **SmoothQuant** | 权重+激活 | 数学等价变换 | O(n) | 见下节课 |

---

## 六、工程实现细节

### 6.1 Hessian 计算优化

```python
# 直接算 H = 2 * X^T @ X
# X: (128 样本, 4096 维度) → H: (4096, 4096) = 64MB
# H⁻¹: 4096³ ≈ 68B FLOPs → GPU 上 0.1 秒，可接受

# 但如果是更大的模型: d_model=8192, H: 256MB, H⁻¹: ~500B FLOPs
# → 但仍只做一次，校准阶段可承受
```

### 6.2 Cholesky 为什么比直接求逆好？

```python
# torch.inverse(H):
#   数值不稳定 (H 可能接近奇异, d_model 大时行列式接近 0)
#   → inverse 可能产生极大的值 → 补偿时数值爆炸

# torch.cholesky(H):
#   Cholesky = 下三角分解 H = L@L^T
#   只对三角矩阵操作 → 数值稳定得多
#   且只需要 L (下三角), 内存省一半
```

### 6.3 校准数据的选择

```python
# 原则: 从训练集中随机取 128-256 条
# 太少 (8条):   H 不能代表真实数据分布 → 补偿不准
# 太多 (2048条): H 计算开销大 → 且收益递减

# Houmo 的做法 (从 ptq.py 推断):
# 用 128-256 条典型 prompt 做一次 forward → 收集每层的 X
# → 用于 GPTQ 校准
```

### 6.4 分组大小 trade-off

```python
# group_size = 128:   粒度中等，精度高，scale 存储开销 = 1/128
# group_size = 64:    粒度小，精度更高，scale 开销 = 1/64
# group_size = -1:    per-channel，精度低，scale 开销极低

# GPTQ 论文推荐: group_size=128 (精度和开销的甜点)
```

---

## 七、Houmo 的 GPTQModel

### 7.1 代码位置

```
HLIEvLLM-1.4.0rc0/hmodel/gptqmodel/
├── chat/chat.py              ← 量化模型使用示例
├── dev/test_gptq.py          ← GPTQ 测试
├── xh2/                      ← XH2 平台特定代码
├── pyproject.toml            ← 包配置
└── README.md
```

### 7.2 量化流程

```python
# 推断的 Houmo GPTQ 使用流程:
# 1. 加载 HF 模型
# 2. 校准: 跑 128-256 条 prompt → 收集每层激活值
# 3. GPTQ 逐层量化:
#    for layer in model.layers:
#        H = 2 * X[校准批次]^T @ X[校准批次]
#        for column in layer.weight:
#            量化 → 补偿后续列 → 更新 H⁻¹
# 4. 保存量化权重 → 编译为 Helion FP4/INT4 kernel → .hmm
# 5. 打包为 GGUF → 推理部署
```

### 7.3 和量化团队的讨论话题

```
1. "Houmo 的 GPTQ 实现是 per-channel 还是 per-group? group_size 选多少?"
   → 看 build.py 的 config.yaml 或 ptq.py 的量化参数

2. "Helion 的 INT4 kernel 是怎么接收量化权重的? 
   是 INT4 compressed (2个值装1字节), 还是 uncompress 后传给 NPU?"

3. "校准数据的质量对量化精度影响多大? 
   是否需要高质量的标注数据?"

4. "量化后的精度验证: 用哪些 benchmark? 
   (perplexity / MMLU / GSM8K / ...)"
```

---

## 八、思考题

1. **GPTQ 逐列量化时，为什么"先量化完一列再补偿后续列"？如果直接量完所有列再整体补偿会怎样？**
   - 提示：整体补偿需要等全部列都量化完才能算总误差 → 那时候已经没法改之前的列了

2. **Hessian 矩阵大小 = d_model²。d_model=4096 时 64MB。d_model=8192 时 256MB。实战中怎么优化内存？**
   - 提示：只保留上/下三角（Cholesky 分解）；分 batch 处理 long sequence

3. **GPTQ 逐列处理 vs AWQ 的"一次性 scale 搜索"，哪种更适合 Houmo NPU 的离线编译流水线？**
   - 提示：GPTQ 耗时更长（逐列 + Hessian）但精度理论保证更强。AWQ 更快（只需跑一次 forward 看激活），对 NPU 离线编译可能更友好
