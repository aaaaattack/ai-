# 第3课：SwiGLU 激活函数 — 教学版 vs 工程版

> 学习路线：阶段1 算子层对比（CS336 hw3 ↔ nano-vllm layers/）
> 对应文件：
> - CS336 教学版：`chapter1/hw3/SwiGLU.py`
> - nano-vllm 工程版：`nano-vllm/nanovllm/layers/activation.py`
> - 模型调用处：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3MLP)

---

## 一、SwiGLU 是什么？

SwiGLU = **Swi**sh + **GLU**（Gated Linear Unit），是 LLM 中 FFN（前馈网络）的标准激活函数。

### 1.1 传统 FFN vs SwiGLU FFN

```
传统 FFN (原始 Transformer):
  out = Linear₂(ReLU(Linear₁(x)))
  2 个权重矩阵
  ReLU 负值全截断为 0

SwiGLU FFN (LLaMA / Qwen / ChatGLM):
  out = Linear₂(SiLU(Linear_gate(x)) ⊙ Linear_up(x))
  3 个权重矩阵，多了一个"门控"通道
  SiLU 负值端允许小负值通过
```

### 1.2 SiLU 激活函数

```
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

形状：
```
        │        ╱
        │      ╱
        │    ╱
   ─────┼──╱──────→
     ╱  │
   ╱    │
```

特点：
- x→+∞：sigmoid→1，SiLU≈x（近似线性）
- x→0：sigmoid→0.5，SiLU≈0（光滑通过）
- x→-∞：sigmoid→0，SiLU→0⁻（允许小负值，不直接截断）
- 处处**光滑可导**（不像 ReLU 在 x=0 处有突变）

这些特性使 SwiGLU 在训练时梯度流动更稳定，推理时精度更高，已取代 ReLU 成为 LLM 标配。

---

## 二、CS336 教学版逐行解析

源码：`chapter1/hw3/SwiGLU.py`

### 2.1 初始化：三个独立 Linear

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate 通道权重
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # 输出投影（降维回 d_model）
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up 通道权重

    def silu(self, x):
        return x * torch.sigmoid(x)
```

**`d_ff`（中间维度）**：通常是 d_model 的 2-4 倍。d_model=512 时 d_ff=2048（膨胀比 4）。

### 2.2 前向传播

```python
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
```

计算图拆解：

```
x: (batch, seq, d_model)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  w1(x)     w3(x)
  (gate)    (up)
    │         │
    ▼         │
  SiLU        │
    │         │
    └────⊙────┘   逐元素乘
         │
         ▼
       w2(x)
  (batch, seq, d_model)
```

**参数数量**：

```
w1: d_model × d_ff = 512 × 2048 = 1,048,576 参数
w2: d_ff × d_model = 2048 × 512 = 1,048,576 参数
w3: d_model × d_ff = 512 × 2048 = 1,048,576 参数
总计: 3,145,728 参数
```

对比传统 FFN（只有 w1 和 w2）：2,097,152 参数。SwiGLU 多了 50% 的 FFN 参数，这也解释了为什么同等参数量下 SwiGLU 效果更好。

### 2.3 `nn.Linear` 的公式

```
nn.Linear(in_features, out_features):
  output = input @ W.T + b

对于 w1: input @ w1.T = (batch, seq, d_model) @ (d_model, d_ff)
         = (batch, seq, d_ff)
```

CS336 的注释特别说明了"注意讲义上是 Wx 这种列向量的形式出现，简单写法 self.w1(x) 就是按照行向量了"，因为 PyTorch 默认用行向量输入。

---

## 三、nano-vllm 工程版逐行解析

### 3.1 优化1：gate 和 up 的权重合并

CS336 教学版 gate 和 up 是两个独立的 `nn.Linear`，但它们的**输入相同**（都是 x），**输出形状相同**（都是 d_ff），只是输出用途不同。这是完美的融合机会。

```python
# nano-vllm 的 Qwen3MLP
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,                    # d_model = 576
    [intermediate_size] * 2,        # [1024, 1024] → 输出维度 = 2048
    bias=False,
)
```

**`MergedColumnParallelLinear`** 内部是一个合并的权重矩阵：

```
W_gate_up = [W_gate; W_up]
            ↑     ↑
          上半   下半

W_gate_up 形状: (2×intermediate_size, hidden_size) = (2048, 576)

gate_up_proj(x) → matmul(x, W_gate_up^T) → (batch, seq, 2048)
# 前 1024 维是 gate，后 1024 维是 up
```

**融合的收益**：

```
分离方式:
  w1(x)   → launch matmul kernel → 读 X 读 w1 → 写 gate
  w3(x)   → launch matmul kernel → 读 X 读 w3 → 写 up
  共 2 次 matmul，X 被读了 2 次

融合方式:
  gate_up_proj(x) → launch matmul kernel → 读 X 读 [w1;w3] → 写 [gate;up]
  共 1 次 matmul，X 只读 1 次！
```

在 decode 阶段（每个 token 独立计算），每省 1 次 matmul kernel 都有意义。

### 3.2 优化2：SiluAndMul —— 拆分 + 激活 + 乘法融合

```python
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)    # 沿最后一维对半分 → gate 和 up
        return F.silu(x) * y     # SiLU(gate) * up
```

**`chunk(2, dim=-1)`**：把 `(batch, seq, 2048)` → `(batch, seq, 1024)` 和 `(batch, seq, 1024)`

```python
# 等价于
gate = x[..., :1024]     # 前一半
up   = x[..., 1024:]     # 后一半
result = F.silu(gate) * up
```

**`@torch.compile`** 把 chunk → silu → mul 三步融合成一个 CUDA kernel：

```
无 compile: 3 个 kernel (chunk/split, silu, mul)
有 compile: 1 个 kernel (fused_chunk_silu_mul)
```

### 3.3 完整调用流程

源码：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3MLP)

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)   # (bs,seq, 2*1024) = (bs,seq, 2048)
        x = self.act_fn(gate_up)          # chunk → silu → mul → (bs,seq, 1024)
        x = self.down_proj(x)             # (bs,seq, hidden_size)
        return x
```

**与 CS336 的对应**：

| CS336 | nano-vllm | 说明 |
|---|---|---|
| `w1(x)` → SiLU | `gate_up_proj(x)` → chunk → `F.silu(x)` | gate 通道 |
| `w3(x)` | `gate_up_proj(x)` → chunk → `y` | up 通道 |
| `SiLU(gate) * up` | `SiluAndMul` 融合 | 逐元素乘 |
| `w2(result)` | `down_proj(x)` | 投影回 d_model |

### 3.4 详解：gate_up_proj 和 down_proj 的 Tensor Parallel 设计

nano-vllm 的 SwiGLU 用了两种不同的并行 Linear，设计原理值得展开：

#### gate_up_proj：MergedColumnParallelLinear（按列并行）

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size, [intermediate_size] * 2, bias=False
)
```

继承自 `ColumnParallelLinear`，按**输出维度（列）**切分权重。

```
原始权重: W = (2×intermediate_size, hidden_size) = (2048, 576)

TP=2 时，每张卡持有:
  卡0: W₀ = (1024, 576)   ← 输出维度 / 2，输入维度不变
  卡1: W₁ = (1024, 576)

forward:
  卡0: y₀ = x @ W₀.T  → (bs, seq, 1024)   ← 各自计算一半输出
  卡1: y₁ = x @ W₁.T  → (bs, seq, 1024)

  结果: y = [y₀, y₁]   ← 自然拼接，无需通信！
```

**关键**：ColumnParallel 的输出是切分后的局部结果，但因为 gate_up_proj 的输出后续只做逐元素操作（chunk → silu → mul），不做跨卡计算，所以**不需要 all-reduce 通信**。

#### down_proj：RowParallelLinear（按行并行）

```python
self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
```

按**输入维度（行）**切分权重——和 ColumnParallel 刚好相反。

```
原始权重: W = (hidden_size, intermediate_size) = (576, 1024)

TP=2 时，每张卡持有:
  卡0: W₀ = (576, 512)   ← 输入维度 / 2，输出维度不变
  卡1: W₁ = (576, 512)

forward:
  卡0: y₀ = x₀ @ W₀.T  → (bs, seq, 576)  ← 各自用局部输入算出完整输出
  卡1: y₁ = x₁ @ W₁.T  → (bs, seq, 576)

  结果: y = y₀ + y₁    ← 通过 all_reduce 求和 → 所有卡得到相同结果
```

**关键**：RowParallel 的输出是**部分和**，必须通过 `dist.all_reduce(y)` 把所有卡的结果加起来才是正确输出。

#### 为什么这样搭配？

```
SwiGLU 的数据流:

  x (576)
    │
    ▼  ← 输入完整
  gate_up_proj (ColumnParallel)
    │  按输出列切分，无通信
    ▼
  [gate | up] → 卡0: 前1024  卡1: 后1024
    │  ↑ 各自独立
    ▼  ← chunk → silu → mul
  中间结果 (1024)   ← 每卡的输出是局部的
    │
    ▼  ← RowParallel: 按输入切分
  down_proj
    │  ← 每卡的局部输入 × 局部权重，输出求和
    ▼  ← all_reduce
  out (576)   ← 所有卡结果一致
```

**设计原则**：
1. 输入的张量在卡间是**完整复制**的 → 用 ColumnParallel（各算各的输出，无通信）
2. 中间结果在卡间是**按维度切分**的 → 用 RowParallel（局部计算 + all-reduce 汇总）
3. 每层 Block 输入和输出都是完整复制的 → 首尾都是 ColumnParallel，中间是 RowParallel，形成一个**无额外通信的闭环**

#### RowParallelLinear 源码

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 按行切分：output_size 不变，input_size / tp_size
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)   # ← 求和所有卡的结果
        return y
```

**`self.tp_rank == 0` 时才加 bias**：bias 加在所有卡上会导致重复加（每次 all_reduce 后再各自加 bias 就加多了），所以只有 rank 0 加，然后 all_reduce 广播给其他卡。`dist.all_reduce(y)` 默认操作是 `SUM`，所以：

```
卡0: y₀ + bias
卡1: y₁ + (无 bias)
all_reduce(SUM): 结果 = (y₀ + bias) + y₁ = y₀ + y₁ + bias  ← 正确！bias 只加了一次
```

#### 补充：如何区分输入维度和输出维度

```python
# nn.Linear 签名: in_features → 输入, out_features → 输出
nn.Linear(in_features, out_features)

# weight 矩阵形状: (out_features, in_features)
weight.shape = (out_features, in_features)

# forward:
y = x @ weight.T
# x: (..., in_features)  → y: (..., out_features)
```

**在 Tensor Parallel 中的对应**：

```python
# ColumnParallel: 切 output 维度（权重矩阵的第 0 维）
ColumnParallelLinear(in=576, out=1024)   # 原 weight (1024, 576)
TP=2 → 每卡 weight (512, 576)            # out 减半，in 不变

# RowParallel: 切 input 维度（权重矩阵的第 1 维）
RowParallelLinear(in=1024, out=576)      # 原 weight (576, 1024)
TP=2 → 每卡 weight (576, 512)            # in 减半，out 不变
```

**记法**：

```
Linear(in, out) → weight = (out, in)

你想切哪个维度变化？out 还是 in？
  out 变小 → ColumnParallel  (按列切)
  in  变小 → RowParallel     (按行切)
```

**SwiGLU 中**：

```
gate_up_proj: 576 → 2048   (out 维度膨胀 2×)  → ColumnParallel 切 out
down_proj:    1024 → 576   (in 维度从 1024 来) → RowParallel 切 in
```

---

## 四、教学版 vs 工程版差异总结

| 差异点 | CS336 教学版 | nano-vllm 工程版 | 意义 |
|---|---|---|---|
| **gate/up 投影** | 两个独立 `nn.Linear` | 一个 `MergedColumnParallelLinear` | 省 1 次 matmul kernel launch + 1 次 X 显存读取 |
| **激活 + 乘法** | `silu` + `*` 两步分开 | `SiluAndMul` 把 chunk + silu + mul 融合 | `@torch.compile` 融合成一个 kernel |
| **总 kernel 数** | 3 matmul + silu + mul = 5 个 kernel | 2 matmul + 1 融合 kernel = 3 个 kernel | 减少约 40% kernel launch |
| **X 读取次数** | 2 次（gate 和 up 各自读一次） | 1 次（合并矩阵一次读完） | decode 阶段省 1 次显存读取 |
| **命名** | `SwiGLU`（强调算法） | `SiluAndMul`（描述操作） | 工程命名更关注"做了什么" |

---

## 五、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **编译器** | `MergedColumnParallelLinear` 的 gate+up 合并是编译器 IR fusion 的典型目标。你们编译器是否支持自动识别同类输入、同形状输出的 Linear 并合并？ |
| **量化** | gate 和 up 合并后共享同一个权重矩阵 → 是否共享同一个 scale？如果分开量化（各自 scale）精度更好，但需要改 `MergedColumnParallelLinear` 的 weight_loader |
| **推理工具链** | SwiGLU 在 FFN 中的计算量占整个 forward 的约 60%（d_ff 远大于 d_model），decode 阶段第二大耗时项（仅次于 Attention） |
| **TCIM Runtime** | `.hmm` 编译后，SwiGLU 是三个独立算子还是已融合？chunk + silu + mul 是否被编译器 fused？ |
| **性能分析** | Prefill: SwiGLU 的两次 matmul 都是大 GEMM → 计算密集 → Tensor Core 利用率关键 |
|  | Decode: 矩阵很小 → 访存密集 → gate/up 融合收益大 |

---

## 六、思考题

1. **为什么 nano-vllm 把 gate 和 up 合并成一个矩阵，而不是像 CS336 那样分开？**
   - 提示：两个 Linear 输入相同、输出形状相同，是一次 matmul 能搞定的天然融合机会。decode 阶段省 1 次 kernel launch。

2. **`MergedColumnParallelLinear` 在 Tensor Parallel 时怎么切分权重？gate 和 up 各分多少？**
   - 提示：把 `[intermediate_size] * 2` 的 total 按 TP 大小切分，gate 和 up 同比例削减。看 `linear.py` 的 `weight_loader` 中的 `shard_offset` 和 `shard_size`。

3. **decode 阶段，SwiGLU 的瓶颈是计算（矩阵乘）还是访存？算术强度大约多少？**
   - 提示：FFN 的两个 matmul（gate_up_proj 和 down_proj）各涉及多大的权重矩阵？对一个 token 的 FLOPs 和 Bytes 分别是多少？

4. **为什么 SwiGLU 的三个矩阵（w1/w2/w3）都用 bias=False？**
   - 提示：RMSNorm 已经做了归一化，bias 在 SwiGLU 门控机制中可能引入不必要的偏移。LLaMA/Qwen 的代码也都是 False。

---

## 七、CS336 课后作业

### 作业要求（根据讲义和代码还原）

实现一个 SwiGLU 模块：

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model (int): 输入/输出维度
            d_ff (int): 中间膨胀维度（通常 = d_model * 2~4）
        """

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU 激活函数: x * sigmoid(x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
```

### 实现要点

1. 三个 `nn.Linear(d_model, d_ff)` 的无偏置线性层（gate/up/down）
2. SiLU 实现：`x * torch.sigmoid(x)`
3. 前向逻辑：`down(silu(gate(x)) * up(x))`

### 验证方法

```python
import torch

d_model, d_ff = 512, 2048
swiglu = SwiGLU(d_model, d_ff)
x = torch.randn(2, 10, d_model)
out = swiglu(x)

assert out.shape == (2, 10, d_model)
assert out.dtype == x.dtype

# 梯度验证
loss = out.sum()
loss.backward()
for name, param in swiglu.named_parameters():
    assert param.grad is not None, f"{name} has no gradient"
```

### 扩展思考（结合 nano-vllm 工程版）

尝试模仿 nano-vllm 的方式，将 gate 和 up 权重合并，并实现 `SiluAndMul` 融合算子，Benchmark 对比：

```python
# 三个独立 Linear（CS336 教学版）
# 合并 gate+up + SiluAndMul（nano-vllm 工程版）

# Benchmark 代码
import timeit

x = torch.randn(1, 1, d_model)  # 模拟 decode

t1 = timeit.timeit(lambda: swiglu_naive(x), number=1000)
t2 = timeit.timeit(lambda: swiglu_fused(x), number=1000)

print(f"Naive (3 matmuls):  {t1:.3f}s")
print(f"Fused (2 matmuls):  {t2:.3f}s")
print(f"Speedup: {t1/t2:.2f}x")
```
