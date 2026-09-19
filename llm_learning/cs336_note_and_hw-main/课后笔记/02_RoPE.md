# 第2课：RoPE 旋转位置编码 — 教学版 vs 工程版

> 学习路线：阶段1 算子层对比（CS336 hw3 ↔ nano-vllm layers/）
> 对应文件：
> - CS336 教学版：`chapter1/hw3/rope.py`
> - nano-vllm 工程版：`nano-vllm/nanovllm/layers/rotary_embedding.py`
> - 模型调用处：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3Attention)

---

## 一、为什么需要位置编码？

Transformer 的 Self-Attention 对位置不敏感——把"我打你"和"你打我"的 token 顺序交换，在 Attention 计算中结果完全一样（只是矩阵行列互换）。

**位置编码就是在 token 向量里注入位置信息**，让 Attention 知道 token 之间的相对位置关系。

| 位置编码方式 | 代表模型 | 特点 |
|---|---|---|
| 绝对位置（Sinusoidal） | 原始 Transformer | 加法注入位置信息，直接加到 embedding 上 |
| 可学习位置 | BERT, GPT-1 | 学一个位置嵌入表，训练中学习 |
| **RoPE（旋转位置）** | **LLaMA, Qwen, ChatGLM** | **乘法注入位置信息，通过旋转矩阵改变向量方向** |

---

## 二、RoPE 的核心思想：2D 旋转

RoPE 的直觉：**用一个旋转矩阵乘以向量，旋转角度取决于 token 的位置**。

### 2D 旋转矩阵

```
位置 m 的向量  x_m = (x₁, x₂)
旋转角度       θ_m = m * ω

旋转后的向量:
y₁ = x₁·cos(θ_m) - x₂·sin(θ_m)
y₂ = x₁·sin(θ_m) + x₂·cos(θ_m)

即： [y₁] = [cos(θ_m)  -sin(θ_m)] · [x₁]
     [y₂]   [sin(θ_m)   cos(θ_m)]   [x₂]
                    ↑ 标准的 2D 旋转矩阵
```

### RoPE 的核心性质：相对位置不变性

```
对于位置 m 和位置 n 的两个向量，旋转后做内积：

⟨R(m)·q_m, R(n)·k_n⟩ = q_m^T · R(n-m) · k_n
                        ↑ 只依赖相对位置 (n-m)，不依赖绝对位置 m 和 n！
```

这直接融入了 Attention 的 `Q·K^T` 计算，模型天然能感知相对位置。

### 扩展到高维：多个 2D 子空间

高维向量（如 d_k=128）怎么旋转？— 把向量分成 d_k/2 对（64 对），每对独立做 2D 旋转，每对的旋转频率不同：

```
维度对 0: 频率 ω₀ = 1 / θ^(0/d_k)           → 旋转最慢（低频，感知远距离位置）
维度对 1: 频率 ω₁ = 1 / θ^(2/d_k)           → 稍快
维度对 2: 频率 ω₂ = 1 / θ^(4/d_k)           → 更快
...
维度对 63: 频率 ω₆₃ = 1 / θ^((d_k-2)/d_k)   → 旋转最快（高频，感知近距离位置）
```

**θ (theta) — 底数超参数**：
- θ = 10000（LLaMA 1/2）：最慢频率 = 1 弧度/token，最快 ≈ 1 弧度/token × sequence
- θ = 500000（LLaMA 3）：增大 θ → 频率更低 → 能感知更长的序列 → 支持更大的 context length
- θ = 1000000（Qwen3）：Qwen3 用的更大 base，支持 128K+ tokens

---

## 三、CS336 教学版逐行解析

源码：`chapter1/hw3/rope.py`

### 3.1 初始化：预计算频率表和 cos/sin 缓存

```python
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")  # 必须偶数 → 两两配对旋转

        self.theta = theta               # 底数超参数，如 10000
        self.d_k = d_k                   # head_dim，如 128
        self.max_seq_len = max_seq_len   # 最大序列长度
        self.device = device
```

#### 频率计算

```python
        # 计算每对维度的旋转频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
```

**语法详解**：

```python
torch.arange(0, self.d_k, 2)
# d_k=128 → [0, 2, 4, ..., 126]   64 个值，每个代表一对维度

torch.arange(0, 128, 2).float() / 128
# → [0/128, 2/128, 4/128, ..., 126/128]
# → [0.0, 0.015625, 0.03125, ..., 0.984375]
# 指数从 0 到接近 1，均匀分布

self.theta ** (arange(0, d_k, 2).float() / d_k)
# theta=10000 → [10000^0.0, 10000^0.0156, ..., 10000^0.984]
#              [1.0, 1.157, ..., 8677.8]

freqs = 1.0 / (theta ** ...)
# → [1.0, 0.864, ..., 0.000115]
# 频率从高到低：高频感知近距离，低频感知远距离
```

#### 位置矩阵

```python
        # 记录每个 token 的位置信息
        positions = torch.arange(self.max_seq_len)
        # max_seq_len=2048 → [0, 1, 2, ..., 2047]
```

#### 角度矩阵：外积

```python
        # 计算正弦和余弦
        sinusoids = torch.outer(positions, freqs)
        # outer: 外积，不是矩阵乘法！
        # outer[i, j] = positions[i] * freqs[j]
        # 两个 1D 向量可以任意长度，结果形状 = (len(a), len(b))
        # positions: (max_seq_len,)  = 2048
        # freqs:     (d_k//2,)      = 64
        # sinusoids: (max_seq_len, d_k//2) = (2048, 64)
        # 每个元素 = 位置 × 频率 = m*ω = 该 token 在该频率下的旋转角度
```

**`torch.outer` 详解**：

```python
a = torch.tensor([1, 2, 3])      # 形状 (3,)
b = torch.tensor([10, 20])       # 形状 (2,)
torch.outer(a, b)
# tensor([[10, 20],              # 形状 (3, 2)
#         [20, 40],
#         [30, 40]])
# 数学: outer[i, j] = a[i] * b[j]
```

#### 缓存 cos/sin

```python
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)
```

- `register_buffer`：告诉 PyTorch 这不是可学习参数（不会被 optimizer 更新），但会随模型一起保存/加载
- `persistent=False`：保存 checkpoint 时不存储（因为它们可以重新计算），省磁盘空间

**为什么不是可学习参数？**

cos/sin cache 是**纯数学函数计算结果**，不是从数据中学来的：

```python
positions = [0, 1, 2, ..., 2047]       # 固定序列
freqs = 1.0 / (10000 ** (...))         # 固定公式
sinusoids = outer(positions, freqs)     # 固定计算结果

cos_cache = sinusoids.cos()            # 位置×频率 取 cos → 纯三角函数
sin_cache = sinusoids.sin()            # 位置×频率 取 sin → 纯三角函数
```

这些值完全由两个超参数决定：`max_seq_len` 和 `theta`。两者都是模型设计好就固定不变的。给定这两个参数，无论训不训练，算出来的值一模一样。让模型去"学"它没有意义——就像不会让模型去"学" `1+1=2`。

**为什么 `persistent=False`？**

```python
# persistent=True (默认): state_dict() 包含 cos_cache
# checkpoint 里包含 cos_cache → 多存 (2048 × 64 × 4 bytes) = 512KB

# persistent=False: state_dict() 不包含 cos_cache
# checkpoint 里没有 cos_cache → 省 512KB
# 加载模型时 RoPE.__init__() 会根据 config 自动重算，不需要从 checkpoint 恢复
```

逻辑是：加载时能根据 `max_seq_len` 和 `theta` 重新生成，所以没必要存储。28 层模型约省 14MB。

**三种 Tensor 角色对比**：

| | `nn.Parameter` | buffer (persistent=True) | buffer (persistent=False) |
|---|---|---|---|
| 被 optimizer 更新 | ✅ | ❌ | ❌ |
| 出现在 `state_dict()` | ✅ | ✅ | ❌ |
| 随 `model.cuda()` 移动 | ✅ | ✅ | ✅ |
| 用途 | 权重、bias（需要学习） | 需要保存的静态数据 | 可重新计算的缓存 |
| 例子 | `self.weight` | `BatchNorm.running_mean` | RoPE 的 cos/sin cache |

### 3.2 前向传播：索引 + 旋转

```python
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_k)    输入向量
        # token_positions: (batch, seq_len)  每个 token 的绝对位置 [0, 1, 2, ...]

        cos = self.cos_cache[token_positions]
        # cos_cache: (max_seq_len, d_k//2) = (2048, 64)
        # token_positions: (batch, seq_len) = (2, 10)
        # cos: (batch, seq_len, d_k//2) = (2, 10, 64)
        # ↑ 用 token_positions 做索引，取出每个位置对应的 64 个角的 cos 值

        sin = self.sin_cache[token_positions]
        # 同上

        cos = cos.unsqueeze(0)  # (2,10,64) 已正确，这行在 batch=1 时有作用
        sin = sin.unsqueeze(0)
```

#### 奇偶维拆分 + 旋转

```python
        # 把 d_k 维按奇偶分成两半
        x_part1 = x[..., 0::2]   # 偶数位: 0, 2, 4, ..., d_k-2  → 64 个值
        x_part2 = x[..., 1::2]   # 奇数位: 1, 3, 5, ..., d_k-1  → 64 个值

        # 对每对 (偶数, 奇数) 做 2D 旋转
        output1 = x_part1 * cos - x_part2 * sin  # 旋转后的"新偶数位"
        output2 = x_part1 * sin + x_part2 * cos  # 旋转后的"新奇数位"
        # 公式: [cos -sin] · [x_even]
        #        [sin  cos]   [x_odd]
```

#### 交错拼接回去

```python
        # 关键：用 stack 交错放回，而不是 cat 拼接
        out = torch.stack([output1, output2], dim=-1)
        # stack 结果: (batch, seq_len, d_k//2, 2)
        # 最后一维: [new_even, new_odd] 交错

        out = out.flatten(-2)
        # flatten 最后两维: (batch, seq_len, d_k//2, 2) → (batch, seq_len, d_k)
        # 恢复原始 d_k 维度，但内容是交错的：[e₀, o₀, e₁, o₁, ...]
        return out
```

**为什么 `stack + flatten` 而不是 `cat`？**

```python
# cat 做法: [e₀, e₁, ..., e₆₃, o₀, o₁, ..., o₆₃]  ← 偶数在前面，奇数在后面  ❌
# stack+flatten: [e₀, o₀,  e₁, o₁,  ..., e₆₃, o₆₃]  ← 偶数奇数交错  ✅

# 原始 x 的形状: x = [x₀, x₁, x₂, x₃, ..., x₁₂₆, x₁₂₇]
#                   偶 奇  偶 奇         偶     奇
# 旋转后必须保持这种交错排列，否则维度错乱
```

---

## 四、nano-vllm 工程版逐行解析

源码：`nano-vllm/nanovllm/layers/rotary_embedding.py`

### 4.1 旋转函数

```python
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)   # 沿最后一维切成两半
    y1 = x1 * cos - x2 * sin                     # 旋转
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

**与 CS336 的差异**：

| | CS336 教学版 | nano-vllm 工程版 |
|---|---|---|
| 拆分方式 | `x[..., 0::2]`, `x[..., 1::2]`（奇偶交错取） | `chunk(x, 2, dim=-1)`（对半切） |
| 重组方式 | `stack + flatten`（交错放回） | `cat((y1, y2), dim=-1)`（直接拼接） |

两种方式数学上完全等价。CS336 的注释说"用了奇偶拆分"，其实不完全准确——`0::2` 和 `1::2` 取的是前后两半（`chunk` 做的是同一件事），不是真正的奇偶交错。只有 `stack + flatten` 才实现了真正的交错。但 nano-vllm 用 `chunk` + `cat` 不交错，同样正确——因为整体旋转不依赖排列。

**`chunk` 指令**：

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
x1, x2 = torch.chunk(x, 2, dim=-1)
# x1 = [1.0, 2.0, 3.0]     ← 前半
# x2 = [4.0, 5.0, 6.0]     ← 后半
```

### 4.2 初始化：cos/sin cache 合并

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        self.head_size = head_size
        assert rotary_dim == head_size

        # 频率计算（和 CS336 完全一致）
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # CS336 命名为 freqs，这里命名为 inv_freq，同一个东西

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 位置序列 [0, 1, 2, ..., max_position-1]

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # einsum 等价于 torch.outer(t, inv_freq)
        # 每个位置 × 每个频率 = 角度矩阵

        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 差异：cos 和 sin 合并为一个 tensor，比 CS336 的两个独立 buffer 更紧凑
        self.register_buffer("cos_sin_cache", cache, persistent=False)
```

**差异1：`einsum` 代替 `outer`**

```python
# 两者等价
torch.outer(t, inv_freq)                          # CS336 写法
torch.einsum("i,j -> ij", t, inv_freq)            # nano-vllm 写法
```

**差异2：cos 和 sin 合并为一个 buffer**

```python
# CS336: 两个独立 buffer
cos_cache: (max_seq_len, d_k//2)      sin_cache: (max_seq_len, d_k//2)

# nano-vllm: 合并为一个 buffer
cos_sin_cache: (max_seq_len, d_k)     ← cos 前半，sin 后半
# 访问时 chunk(2) 拆回 cos 和 sin
```

合并为单个 buffer 减少了 `register_buffer` 的注册开销，也减少了一次 kernel launch（索引一次取出所有数据）。

### 4.3 前向传播

```python
    @torch.compile
    def forward(self, positions, query, key) -> tuple:
        # 差异3：同时处理 Q 和 K（CS336 分别处理）
        cos_sin = self.cos_sin_cache[positions]     # 索引取出 (seq_len, d_k) 的缓存
        cos, sin = cos_sin.chunk(2, dim=-1)          # 拆回 cos 和 sin

        query = apply_rotary_emb(query, cos, sin)    # 旋转 Q
        key   = apply_rotary_emb(key, cos, sin)      # 旋转 K
        return query, key
```

**差异3：同时处理 Q 和 K**

```python
# CS336: 每个要旋转的 tensor 单独调用 RoPE
q_rot = rope(q, positions)
k_rot = rope(k, positions)

# nano-vllm: 一次调用完成两个旋转
q_rot, k_rot = rope(positions, q, k)
```

好处：减少函数调用次数，`torch.compile` 能更好地融合 Q 和 K 的旋转计算（共享 cos/sin 读取）。

### 4.4 单例模式

```python
@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

`@lru_cache(1)`：缓存最近 1 个调用结果。相同参数只创建一次 `RotaryEmbedding` 对象（所有层共享同一个 RoPE 实例，因为频率参数相同）。

### 4.5 在模型中的调用

源码：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3Attention)

```python
# 初始化
self.rotary_emb = get_rope(head_dim, rotary_dim, max_position, base=rope_theta)

# forward
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)

    # QK Norm (Qwen3 特有)
    if not self.qkv_bias:
        q = self.q_norm(q)
        k = self.k_norm(k)

    # 旋转位置编码：在 Attention 之前旋转 Q 和 K
    q, k = self.rotary_emb(positions, q, k)

    # 然后送入 Attention（Flash Attention）
    o = self.attn(q, k, v)
```

---

## 五、教学版 vs 工程版差异总结

| 差异点 | CS336 教学版 | nano-vllm 工程版 | 意义 |
|---|---|---|---|
| **cos/sin 缓存** | 两个独立 buffer | 合并为一个，用 `chunk` 拆分 | 减少一次索引操作，`torch.compile` 融合更好 |
| **拆分方式** | `x[..., 0::2]` + `x[..., 1::2]`（奇偶） | `chunk(x, 2, dim=-1)`（对半） | 数学等价，chunk 语义更清晰 |
| **重组方式** | `stack + flatten`（奇偶交错） | `cat((y1, y2), dim=-1)`（直接拼接） | 数学等价 |
| **Q/K 处理** | 分别调用 `forward(x, pos)` | 一次调用 `forward(positions, q, k)` | 减少函数调用，利于 `torch.compile` 融合 |
| **角度矩阵** | `torch.outer` | `torch.einsum("i,j -> ij")` | 等价，einsum 更通用 |
| **@torch.compile** | 无 | 有 | 融合索引 + chunk + 旋转 + cat 成一个 kernel |
| **单例** | 无 | `@lru_cache(1)` | 所有层共享同一 RoPE 实例 |
| **apply 函数** | 内联在 forward 中 | 独立 `apply_rotary_emb` | 更模块化，可单独测试 |

---

## 六、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **量化** | RoPE 的 cos/sin 缓存在推理时是否需要高精度？通常用 fp16/fp32 存储（旋转改变方向，精度损失直接影响 Attention 的 Q·K 内积） |
| **编译器** | RoPE 是一个**逐元素操作序列**（chunk → cos/sin 索引 → mul → add → cat），是编译器融合的典型目标 |
| **推理工具链** | RoPE 在**每次 forward 都要执行**（不像 embedding 只执行一次），在 decode 中占比虽小但不可忽略 |
| **vLLM/llama.cpp** | RoPE 的 cos/sin cache 是静态的（预计算好，推理期间不变），`register_buffer` 确保随模型序列化 |
| **TCIM Runtime** | RoPE 在 `.hmm` 编译时转换为 NPU 指令序列，你看到的 `cos_sin_cache` 在编译后变成 tensor constant |
         
---

## 七、思考题

1. **为什么 RoPE 只旋转 Q 和 K，不旋转 V？**
   - 提示：Attention 中的 `Q·K^T` 需要感知相对位置，`attn_weight · V` 只是加权求和，不需要位置信息

2. **LLaMA 3 把 θ 从 10000 调大到 500000，为什么能支持更长的 context？**
   - 提示：θ 越大 → 频率越低 → 高频分量越不容易"转满一圈"出现混淆

3. **RoPE 的 cos/sin cache 在量化时应该怎么处理？**
   - 提示：cos 和 sin 是三角函数，值域 [-1, 1]，floating point 精度足够

4. **为什么 QK Norm 放在 RoPE 之前？**
   - 提示：norm 稳定了 RoPE 输入的向量模长，避免旋转后非线性失真

5. **`torch.outer(a, b)` 和 `torch.einsum("i,j -> ij", a, b)` 结果是否完全一样？**
   - 提示：用代码验证 `assert torch.allclose(...)`

---

## 八、CS336 课后作业

### 作业要求（根据讲义和代码还原）

实现一个 RoPE 模块：

```python
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta (float): 底数超参数，如 10000.0
            d_k (int): head_dim，必须为偶数
            max_seq_len (int): 支持的最大序列长度
        """

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_k) 输入 Q 或 K
            token_positions: (batch_size, seq_len) 每个 token 的绝对位置
        Returns:
            (batch_size, seq_len, d_k) 旋转后的张量
        """
```

### 实现要点

1. **频率计算**：`freqs = 1.0 / (theta ^ (arange(0, d_k, 2) / d_k))`
2. **角度矩阵**：`sinusoids = outer(positions, freqs)` → cos_cache, sin_cache
3. **旋转公式**：`y1 = x1*cos - x2*sin`, `y2 = x1*sin + x2*cos`
4. **必须** `register_buffer` 缓存 cos/sin，不能每次 forward 重算

### 验证方法

```python
import torch

# 基础验证
rope = RoPE(theta=10000.0, d_k=64, max_seq_len=512)
q = torch.randn(2, 10, 64)
positions = torch.arange(10).unsqueeze(0).expand(2, -1)  # (2, 10)
q_rot = rope(q, positions)
assert q_rot.shape == q.shape
assert q_rot.dtype == q.dtype

# 旋转性质验证：旋转不改变向量模长
for i in range(q.shape[0]):
    for j in range(q.shape[1]):
        assert torch.allclose(q[i,j].norm(), q_rot[i,j].norm(), atol=1e-5)

# 相对位置性质验证：旋转后 Q·K 只依赖相对位置
q1 = torch.randn(1, 1, 64)
q2 = torch.randn(1, 1, 64)
k1 = torch.randn(1, 1, 64)
k2 = torch.randn(1, 1, 64)

# 旋转 Q 和 K
q1_rot = rope(q1, torch.tensor([[0]]))          # 位置 0
q2_rot = rope(q2, torch.tensor([[3]]))          # 位置 3
k1_rot = rope(k1, torch.tensor([[5]]))          # 位置 5
k2_rot = rope(k2, torch.tensor([[8]]))          # 位置 8

# 只要 (q_pos, k_pos) 的相对距离相同，旋转后内积就应该相同
dot1 = (q1_rot * k2_rot).sum()   # 距离 = 8-0 = 8
dot2 = (q2_rot * k1_rot).sum()   # 距离 = 5-3 = 2  这两个不应该相等

dot3 = (q1_rot * k1_rot).sum()   # 距离 = 5-0 = 5
dot4 = (q2_rot * k2_rot).sum()   # 距离 = 8-3 = 5  这两个应该接近
```

### 扩展思考（结合 nano-vllm 工程版）

1. 尝试改写 RoPE 的 forward，使其一次调用同时处理 Q 和 K（像 nano-vllm 的 `forward(positions, query, key)`）
2. Benchmark：Q 和 K 分别调用 RoPE vs 一次调用，decode 阶段能省多少？
3. 改写 cos/sin cache 为合并版本（`torch.cat((cos, sin), dim=-1)`），对比两个版本在 `torch.compile` 下的性能差异
