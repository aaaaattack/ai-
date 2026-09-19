# 第1课：RMSNorm — 教学版 vs 工程版

> 学习路线：阶段1 算子层对比（CS336 hw3 ↔ nano-vllm layers/）
> 对应文件：
> - CS336 教学版：`chapter1/hw3/RMSnorm.py`
> - nano-vllm 工程版：`nano-vllm/nanovllm/layers/layernorm.py`
> - 模型调用处：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3DecoderLayer)

---

## 一、数学公式

RMSNorm（Root Mean Square Normalization）是 LayerNorm 的简化版本，去掉了均值计算和偏移项：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

拆解为 4 步：

```
Step 1: variance = mean(x²)              ← 计算均方值（不是方差，因为没有减均值）
Step 2: inv_rms = rsqrt(variance + eps)  ← 取倒数平方根
Step 3: x_norm = x * inv_rms             ← 归一化
Step 4: output = x_norm * weight         ← 逐元素缩放（可学习参数）
```

### RMSNorm vs LayerNorm 对比

| | LayerNorm | RMSNorm |
|---|---|---|
| 均值计算 | `mean = x.mean(dim=-1)` | 不计算 |
| 方差计算 | `var = ((x - mean)²).mean()` | `var = x².mean()` |
| 偏移项 | `output = x_norm * weight + bias` | `output = x_norm * weight` |
| 计算量 | 多一次 mean 和减法 | 更少 |
| 适合 LLM | 训练稳定但开销大 | 开销小，LLaMA 系列标配 |

---

## 二、CS336 教学版逐行解析

源码：`chapter1/hw3/RMSnorm.py`

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        # weight 是可学习参数 gamma，初始化为全 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 保存原始精度，转成 float32 计算
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # 2. 计算均方值
        variance = x.pow(2).mean(-1, keepdim=True)
        # 详细见下方 "语法详解：x.pow(2).mean(-1, keepdim=True)"

        # 3. 归一化：x / sqrt(var + eps)
        x = x * torch.rsqrt(variance + self.eps)
        # rsqrt = 1/sqrt，比先 sqrt 再取倒数更快
        # eps 防止除以零

        # 4. 缩放并转回原始精度
        return (self.weight * x).to(input_dtype)
```

### 关键点

1. **精度处理**：先转 float32 计算，再转回原精度（bf16/fp16）
   - 原因：bf16 只有 7 位有效数字，`x.pow(2)` 可能溢出，`mean` 和 `rsqrt` 需要高精度
2. **`rsqrt` 代替 `1/sqrt`**：硬件指令直接支持，减少一次除法
3. **`keepdim=True`**：保持 `(batch, seq_len, 1)` 形状，便于广播乘法

### 语法详解：`x.pow(2).mean(-1, keepdim=True)`

这一行链式调用了两个操作，等价于：

```python
# Step 1: 逐元素平方
squared = x.pow(2)         # 等价于 x ** 2，每个元素自己平方

# Step 2: 沿最后一个维度求平均
variance = squared.mean(-1, keepdim=True)
```

#### `x.pow(2)` — 逐元素平方

```python
# x 形状: (batch_size, seq_len, d_model)
# 例如: (2, 10, 512)

x.pow(2)   # 每个元素平方，等价于 x ** 2
# 形状不变: (2, 10, 512)
```

具体例子：
```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
x.pow(2)
# tensor([[ 1.,  4.,  9.],
#         [16., 25., 36.]])
```

#### `.mean(-1, keepdim=True)` — 沿最后一维求平均

**参数 `-1`**：指定沿哪个维度求平均。`-1` 表示最后一个维度。

```python
# x.pow(2) 形状: (batch_size, seq_len, d_model)
#                        0           1        2
# -1 就是 dim=2，即 d_model 维度

# 沿 d_model 维度求平均：每个 token 的 d_model 个值取平均
```

**参数 `keepdim`**：是否保留被求平均的维度。

```python
squared = torch.tensor([[ 1.,  4.,  9.],
                        [16., 25., 36.]])

squared.mean(-1)                    # keepdim=False (默认)
# tensor([4.6667, 25.6667])        # 形状: (2,)    ← 维度被压缩消失

squared.mean(-1, keepdim=True)
# tensor([[4.6667],                # 形状: (2, 1)  ← 维度保留为 1
#         [25.6667]])
```

**为什么必须 `keepdim=True`？** 后续要做广播乘法：

```python
# x 形状:               (2, 10, 512)
# variance 形状:
#   keepdim=True:       (2, 10, 1)    ← 可以直接广播乘法 ✅
#   keepdim=False:      (2, 10)       ← 形状不匹配，广播出错 ❌

x * torch.rsqrt(variance + eps)
# (2, 10, 512) * (2, 10, 1) → 广播：最后一维 1 → 512 ✅
```

#### 数值意义

对于每个 token 的 d_model 维向量 `x = [x₁, x₂, ..., x_d]`：

```
mean(x²) = (x₁² + x₂² + ... + x_d²) / d

这就是该 token 向量的"平均能量"（mean square）
```

后续 `rsqrt(variance + eps)` 得到 `1/sqrt(mean(x²))`，乘以 x 后每个 token 的 RMS 归一化为 1：

```
RMS(x_norm) = sqrt(mean(x_norm²)) ≈ 1.0
```

---

## 三、nano-vllm 工程版逐行解析

源码：`nano-vllm/nanovllm/layers/layernorm.py`

与教学版数学逻辑完全相同，但有三项关键工程优化，每一项都有明确的设计意图。

### 3.1 基础版本 `rms_forward`

```python
@torch.compile                                          # ← 差异1: 算子融合
def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))                 # ← 差异2: in-place 操作
    x = x.to(orig_dtype).mul_(self.weight)              # ← 差异2: in-place 操作
    return x
```

#### 差异1: `@torch.compile` — 算子融合

**语法作用**：Python 装饰器，对函数做 JIT（Just-In-Time）编译。

**原理**：PyTorch 的 `torch.compile` 会把函数内的多个算子（`pow` → `mean` → `add(eps)` → `rsqrt` → `mul` → `mul(weight)`）**追踪成一张计算图**，然后交给后端编译器（如 Triton / Inductor）生成一个融合的 CUDA kernel。

```
无 torch.compile:  6 个独立 kernel launch
  GPU:  [launch pow][launch mean][launch add][launch rsqrt][launch mul][launch mul(weight)]
         ↑每次 launch 有 5-15μs 的 CPU→GPU 调度开销

有 torch.compile:  1 个融合 kernel launch
  GPU:  [launch fused_kernel]
        ↑ kernel launch 开销减少为 1/6
```

**为什么 decode 阶段特别重要**：decode 时每个 seq 只算 1 个 token，每个 kernel 的计算量极小（几百个元素），kernel launch 开销甚至超过计算本身。融合成 1 个 kernel 能大幅降低这种开销。

**与编译器团队的关联**：你们编译器的 IR fusion（算子融合 pass）和 `torch.compile` 的思路完全相同——在编译期合并相邻算子，减少 runtime 的 kernel launch 次数。

---

#### 差异2: in-place 操作 (`mul_`)

**语法对比**：

```python
# 教学版：out-of-place（创建新 tensor）
x = x * torch.rsqrt(var + self.eps)     # "x * y" 分配新内存，x 指向新 tensor
                                         # 旧的 x 等待 GC 回收

# 工程版：in-place（原地修改）
x.mul_(torch.rsqrt(var + self.eps))     # "mul_" 在 x 的原内存上直接修改
                                         # 零额外内存分配
```

**下划线约定**：PyTorch 中所有带 `_` 后缀的方法都是 in-place 操作（`add_`, `mul_`, `div_`, `sub_` 等）。

**意义**：每次 out-of-place 操作都需要：
1. 分配新显存块
2. 把旧数据拷贝到新位置（或计算得到新数据写入新位置）
3. 旧块标记为可回收

在推理时（特别是 decode 阶段），tensor 很小但操作频繁，频繁的内存分配/释放会导致：
- **显存碎片**：多次分配和释放产生碎片，可能让大块连续内存分配失败
- **时间开销**：`cudaMalloc` 一次约 10-50μs，积少成多
- **不友好于 CUDA Graph**：CUDA Graph 要求显存地址固定，out-of-place 产生新地址会导致图录制失效

---

### 3.2 残差融合版本 `add_rms_forward`（关键优化）

```python
@torch.compile
def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())               # ← 差异3: 残差融合
    residual = x.to(orig_dtype)                        # ← 保存残差给下一层
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

#### 差异3: 残差融合 — 本次课程的重点优化

**回顾 Transformer Block 的计算**：

```
传统方式（CS336 教学版）：
  norm_out  =  RMSNorm(hidden)           # kernel 1: pow → mean → rsqrt → mul
  attn_out  =  Attention(norm_out)       # kernel 2: flash_attn
  out       =  attn_out + hidden          # kernel 3: 独立的残差加法 ← 这个被融合了！

融合方式（nano-vllm）：
  norm_out, residual  =  RMSNorm(hidden, residual)   # kernel 1: add → pow → mean → rsqrt → mul
  attn_out            =  Attention(norm_out)          # kernel 2: flash_attn
  # 不需要 kernel 3！残差已经在 kernel 1 内部完成了
```

**为什么叫"融合"**：

| | 分离写法 | 融合写法 |
|---|---|---|
| 残差加法 | `out = attn(x) + residual`（独立 add kernel） | 在 norm 内部做 `x.add_(residual)` |
| norm 归一化 | `norm(x)`（独立 norm kernel） | 同上一个 kernel 继续做 |
| 结果 | **2 次 kernel launch** | **1 次 kernel launch** |

**为什么 decode 阶段收益大**：

- Prefill 阶段（如 512 tokens 一批）：kernel 计算量大（512 次成吨的 FLOPs），launch 开销占比 < 1%，融合收益不明显
- Decode 阶段（1 token）：kernel 计算量极小（几百个 FLOPs），launch 开销占 30-50%，每省一次 launch 都立竿见影

```
28 层的 Transformer Block，每层有 2 次残差加法：
  无融合: decode 每步额外 launch 56 个 add kernel（56 × 10μs = 0.56ms）
  有融合: 0 个额外 launch
  假设一次 decode 耗时 2ms → 加速约 28%
```

**与驱动/HAL 层的关联**：HAL 层在实现 `add_rms_forward` 时需要确保：残差加法和 norm 归一化走同一条 DMA 路径，避免中间结果写回显存（在片上 SRAM 内完成）。

---

### 3.3 统一入口 `forward`

```python
def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    if residual is None:
        return self.rms_forward(x)                     # 第一层，无残差
    else:
        return self.add_rms_forward(x, residual)       # 后续层，残差融合
```

**`residual: torch.Tensor | None = None`** — Python 3.10+ 的类型联合语法，表示 `residual` 可以是 `Tensor` 或 `None`，默认为 `None`。

**分支逻辑**：第一层 Transformer Block 没有上一层的残差输入，所以直接走 `rms_forward`；后续层都有残差，走融合版本。

---

### 3.4 在模型中的调用

源码：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3DecoderLayer)

```python
def forward(self, positions, hidden_states, residual):
    if residual is None:
        # 第一层 Block: norm 和残差分离
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        #                       └─ norm  ─┘               └─ 残差初始化为 hidden_states ─┘
    else:
        # 后续层 Block: 残差融合进 norm
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        #                       └─ 残差加法 + norm 一步完成 ─┘
    
    hidden_states = self.self_attn(positions, hidden_states)
    
    # Post-Attention Norm 也用残差融合（同上）
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

**残差传递方式的变化**：

```
CS336 教学版：残差在 Block 外部管理
  hidden → norm → attn → +hidden → norm → ffn → +result
  
nano-vllm：残差作为参数在层间显式传递
  residual 在每层间流动，相当于 "残差通道" 独立在 Block 外
  Qwen3DecoderLayer.forward(hidden_states, residual) → (hidden_states, residual)
```

这种设计把残差管理从 Block 内部提升到外部，使得 `RMSNorm` 作为一个无状态的算子被复用，不需要 `nn.Module` 内部存储残差引用。Cleaner and more composable。

---

### 三种差异的意义优先级

| 优先级 | 差异 | 对推理性能的影响 | 与你工作的关联 |
|---|---|---|---|
| **高** | 残差融合 | decode 阶段减少 2×层数个 kernel launch | HAL/驱动：需要支持 fused kernel |
| **高** | @torch.compile | 把 6 个子算子融合为 1 个 kernel | 编译器：IR fusion pass 的核心价值 |
| **中** | in-place 操作 | 减少显存分配/碎片，支持 CUDA Graph 录制 | 量化：减少中间 tensor 精度损失累积 |

---

## 四、教学版 vs 工程版差异总结

| 差异点 | CS336 教学版 | nano-vllm 工程版 | 为什么 |
|---|---|---|---|
| **in-place 操作** | `x = x * rsqrt(...)` 创建新 tensor | `x.mul_(rsqrt(...))` 原地修改 | 省显存分配，减少内存碎片 |
| **残差融合** | 无（残差在 Block 中单独加） | `add_rms_forward` 把残差加法融进 norm | 少一次 kernel launch，decode 阶段收益大 |
| **torch.compile** | 无 | `@torch.compile` | 把 pow→mean→rsqrt→mul 融合成一个 CUDA kernel |
| **eps 默认值** | 1e-5 | 1e-6 | 不同模型的超参数选择 |
| **参数命名** | `d_model` | `hidden_size` | 同一概念的不同命名 |
| **精度转换** | `.to(torch.float32)` | `.float()` | 等价，简写 |

---

## 五、残差融合的原理图解

### CS336 教学版（残差不融合）

```
输入 in_features
  │
  ├──→ rms_norm1(in_features) ──→ attention ──→ x1
  │                                    │
  └──────────────────────────────────→ x1 + in_features  ← 独立的 add kernel
                                              │
                                   ├──→ rms_norm2 ──→ swiglu ──→ x2
                                   │                      │
                                   └─────────────────→ x2 + x1  ← 又一个独立 add kernel
```

每个 Transformer Block 有 **2 次独立的残差加法 kernel**。

### nano-vllm 工程版（残差融合进 norm）

```
输入 hidden_states, residual
  │
  ├──→ input_layernorm(hidden_states, residual)  ← 残差加法 + norm 一步完成
  │         │
  │         ├──→ norm_output (给 attention)
  │         └──→ residual (更新后的残差，给下一层)
  │
  ├──→ attention(norm_output)
  │
  ├──→ post_attention_layernorm(attn_output, residual)  ← 又一次残差融合
  │         │
  │         ├──→ norm_output (给 mlp)
  │         └──→ residual
  │
  └──→ mlp(norm_output)
```

每个 Transformer Block **少了 2 次独立的 add kernel**，直接融合进 norm。

---

## 六、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **量化** | RMSNorm 的 `rsqrt(var + eps)` 对精度敏感，量化时通常**保持 norm 层在 fp16/fp32**，只量化 weight。weight 量化时要注意 scale 融合 |
| **编译器** | `@torch.compile` 的算子融合和编译器的 IR fusion 是同一个思路：把 pow→mean→rsqrt→mul 合并成一个 kernel |
| **推理工具链** | 残差融合减少 kernel launch，在 decode 阶段（访存瓶颈）每个 kernel launch 都是开销 |
| **驱动/HAL** | `mul_` 这种 in-place 操作减少显存分配，HAL 层需要支持高效的 in-place 算子 |
| **llama.cpp** | llama.cpp 的 `llama_vec_norm_rms` 函数实现的就是这个 RMSNorm，量化时 norm 层用 fp16 计算 |

---

## 七、思考题

1. **为什么 RMSNorm 比 LayerNorm 更适合 LLM？**
   - 提示：少了 mean 的计算和 bias，计算量更小；LLM 规模大，每一点节省乘以层数和序列长度都是可观的

2. **`add_rms_forward` 融合了哪两个操作？为什么这对 decode 性能重要？**
   - 提示：残差加法 + 归一化；decode 阶段是访存瓶颈，少一次 kernel launch = 少一次显存读写

3. **量化场景下，RMSNorm 的 `weight` 参数应该量化吗？`variance` 计算应该用什么精度？**
   - 提示：weight 是逐通道缩放，量化误差会影响每个 token 的输出幅度；variance 涉及平方和开方，精度要求更高

4. **`torch.compile` 具体融合了哪几个算子？如果不融合，会 launch 几个 kernel？**
   - 提示：pow, mean, add(eps), rsqrt, mul, mul(weight) = 至少 6 个 kernel 融合成 1 个

5. **在 nano-vllm 的 `Qwen3DecoderLayer` 中，第一层为什么不用残差融合？**
   - 提示：第一层没有前一层传来的 residual，`residual is None` 时走 `rms_forward` 而非 `add_rms_forward`

---

## 八、CS336 课后作业

### 作业来源

- 课程：Stanford CS336 Spring 2025, Assignment 1
- 讲义：`原始官方讲义/cs336_spring2025_assignment1_basics.pdf`
- 实现：`chapter1/hw3/RMSnorm.py`

### 作业要求（根据讲义和代码还原）

实现一个 RMSNorm 模块，满足以下规格：

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model (int): 嵌入维度，即每个 token 的特征维度
            eps (float): 数值稳定性的小常数
            device: 计算设备
            dtype: 数据类型
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) 输入张量
        Returns:
            (batch_size, seq_len, d_model) 归一化后的张量
        """
```

### 实现要点

1. **可学习参数**：`weight`（gamma），形状 `(d_model,)`，初始化为全 1
2. **精度处理**：输入先转 `float32` 计算，输出转回原始 dtype
3. **公式实现**：`x / sqrt(mean(x²) + eps) * weight`
4. **不允许使用** `torch.nn.RMSNorm`（PyTorch 内置版本），必须从零实现

### 验证方法

```python
import torch

# 基础验证
d_model = 512
rms = RMSNorm(d_model)
x = torch.randn(2, 10, d_model, dtype=torch.bfloat16)
out = rms(x)
assert out.shape == x.shape
assert out.dtype == x.dtype

# 数值验证：输出应该近似单位方差
x = torch.randn(1000, d_model)
out = rms(x)
# 不带 weight 时，输出的 RMS 应该接近 1
# 带 weight（全 1）时，同理
print(f"Output RMS: {out.pow(2).mean().sqrt():.4f}")  # 应接近 1.0

# 梯度验证
x = torch.randn(2, 10, d_model, requires_grad=True)
out = rms(x)
loss = out.sum()
loss.backward()
assert rms.weight.grad is not None
assert rms.weight.grad.shape == (d_model,)
```

### 扩展思考（结合 nano-vllm 工程版）

尝试在你的教学版实现上添加以下工程优化，并 benchmark 性能差异：

```python
# 优化1：in-place 操作
# 优化2：残差融合 add_rms_forward
# 优化3：@torch.compile

# Benchmark 代码模板
import timeit

d_model = 4096
batch_size = 1
seq_len = 1  # 模拟 decode

rms_naive = RMSNorm(d_model)
rms_inplace = RMSNormInplace(d_model)
rms_fused = RMSNormFused(d_model)

x = torch.randn(batch_size, seq_len, d_model)
residual = torch.randn(batch_size, seq_len, d_model)

t1 = timeit.timeit(lambda: rms_naive(x), number=1000)
t2 = timeit.timeit(lambda: rms_inplace(x), number=1000)
t3 = timeit.timeit(lambda: rms_fused(x, residual), number=1000)

print(f"Naive:     {t1:.3f}s")
print(f"In-place:  {t2:.3f}s")
print(f"Fused:     {t3:.3f}s")
```
