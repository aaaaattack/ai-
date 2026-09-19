# 第4课：Softmax — 数值稳定的归一化

> 学习路线：阶段1 算子层对比（CS336 hw3）
> 对应文件：
> - CS336 教学版：`chapter1/hw3/softmax.py`
> - nano-vllm 工程版：无独立文件（用 `torch.softmax`），Sampler 中有 softmax 用法
> - nano-vllm Sampler：`nano-vllm/nanovllm/layers/sampler.py`

---

## 一、Softmax 公式

```python
softmax(x_i) = exp(x_i) / Σⱼ exp(xⱼ)
```

把任意实数向量转换为概率分布：所有输出在 (0,1)，和为 1。

---

## 二、CS336 教学版逐行解析

源码：`chapter1/hw3/softmax.py`

```python
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Step 1: 找指定维度的最大值
    x_max = x.max(dim=dim, keepdim=True)[0]

    # Step 2: 所有值减去最大值，再取 exp
    x_exp = torch.exp(x - x_max)

    # Step 3: 除以总和，归一化
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
```

### 2.1 逐行解析

#### `x.max(dim=dim, keepdim=True)[0]`

```python
x.max(dim=-1)
# 返回值是一个 tuple: (values, indices)
# [0] 取出 values（最大值），[1] 是最大值的索引

# keepdim=True: 保持维度，如 (batch, seq_len, 1)，便于后续广播
```

#### `torch.exp(x - x_max)` — 数值稳定性的核心 trick

**问题**：直接 `exp(x)` 当 x=1000 时溢出为 `+inf`：

```python
x = [1000, 1, 2]
exp(x) = [inf, e, e²]  → softmax = [nan, nan, nan]  ❌
```

**解决**：减掉最大值，数学上等价：

```
exp(x_i) / Σexp(xⱼ) = exp(x_i - max) / Σexp(x_j - max)
```

因为分子分母同除以 `exp(max)`，比值不变。

```python
x - max = [0, -999, -998]
exp([0, -999, -998]) = [1.0, 0.0, 0.0]  → softmax = [1.0, 0.0, 0.0]  ✅
```

**关键洞察**：减了最大值后，最大的 exp = 1，其余 < 1，永不上溢。最小值可能下溢到 0，但 softmax 中接近 0 的值不影响结果。

#### `x_exp.sum(dim=dim, keepdim=True)`

```python
# keepdim=True: (batch, seq_len, 1)，保持维度便于广播除法
# keepdim=False: (batch, seq_len)，形状丢失，除法需要手动 unsqueeze
```

和 RMSNorm 中 `mean(-1, keepdim=True)` 的原因完全相同：保持维度以便 `x_exp / sum` 可以正确广播。

### 2.2 Softmax 的数值稳定性对比

```python
# 不稳定版本（教学不建议）
def softmax_naive(x, dim):
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

# 稳定版本（CS336 实现）
def softmax_stable(x, dim):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

# 验证：两者数学等价
x = torch.tensor([10.0, 20.0, 30.0])  # 不会溢出
assert torch.allclose(softmax_naive(x, -1), softmax_stable(x, -1))

x = torch.tensor([1000.0, 1.0, 2.0])  # naive 版本溢出
print(softmax_stable(x, -1))  # tensor([1., 0., 0.])  正常
# naive 版本: inf → nan
```

---

## 三、Softmax 在 LLM 中的两个位置

| 位置 | dim | 输入 shape | 作用 |
|---|---|---|---|
| **Attention weights** | `dim=-1` | `(bs, n_heads, seq_q, seq_k)` | 对 key 维度归一化，得到该 query token 对每个 key token 的注意力权重 |
| **LM Head 输出 (Sampler)** | `dim=-1` | `(bs, vocab_size)` | 对词表维度归一化，得到下一个 token 的概率分布 |

**注意**：Attention 中的 softmax 配合 `mask` 使用（CS336 的 `scores.masked_fill(mask, -1e9)`），被 mask 的位置在取 exp 后趋近于 0。

---

## 四、nano-vllm Sampler：Softmax + 温度缩放 + 采样

源码：`nano-vllm/nanovllm/layers/sampler.py`

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 2. Softmax → 概率分布
        probs = torch.softmax(logits, dim=-1)

        # 3. Gumbel-max 采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
```

### 4.1 温度缩放

#### 语法详解

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
```

这行包含四个操作，链式调用：

**Step 1: `.float()` — 转 float32 精度**

```python
logits.float()
# 模型输出的 logits 可能是 bf16/fp16
# 后续 softmax 的 exp 对精度敏感 → 必须转 float32
# 这个转换是临时操作（in-place），不影响模型本身的权重大小
```

**Step 2: `.unsqueeze(dim=1)` — 插入维度用于广播**

```python
# temperatures 来自 prepare_sample()（model_runner.py L190-192）：
temperatures = [seq.temperature for seq in seqs]
# shape: (batch_size,)  e.g. (3,) = [0.6, 0.6, 1.0]

temperatures.unsqueeze(dim=1)
# (3,) → (3, 1)
# 在 dim=1 处插入大小为 1 的新维度
```

**为什么需要 `unsqueeze`？**

```python
# logits shape:    (batch_size, vocab_size)  e.g. (3, 151936)
# temperatures:    (batch_size,)             e.g. (3,)

# 直接相除：
logits / temperatures   # (3, 151936) / (3,)
# PyTorch 广播规则：从右向左对齐，最后两维 (151936) vs (3)  → 部分匹配，可能出错

# unsqueeze 后：
logits / temperatures.unsqueeze(1)   # (3, 151936) / (3, 1)
# 广播: (3,1) → expand → (3, 151936)，每个 seq 的所有 vocab 用同一个 temperature
# 意图明确：每行（每个 seq）有自己的 temperature
```

**Step 3: `.div_(...)` — in-place 除法**

```python
.div_(temperatures.unsqueeze(1))
# 等价于 logits = logits / temperatures.unsqueeze(1)
# 但 in-place 版本在原地修改，不分配新 tensor
# 省一次显存分配（decode 阶段 tensor 很小但操作频繁）
```

**完整数据流**：

```python
logits: (bs, vocab_size)          e.g. (3, 151936)
        ↓ .float()                # bf16 → float32，softmax 需要高精度
        ↓ .div_(
temperatures: (bs,) → unsqueeze(1) → (bs, 1)
# 广播除法: 每个 seq 的 vocab logits 除以该 seq 的 temperature
        )
        ↓
scaled_logits: (bs, vocab_size)   # float32
        ↓ torch.softmax(dim=-1)
probs: (bs, vocab_size)           # 概率分布
```

**为什么每个 seq 有不同的 temperature？**

Continuous batching 时，同一个 batch 里可能有多个请求，各自的 `SamplingParams` 不同：

```python
seq_0: temperature=0.6  (确定性生成，代码补全)
seq_1: temperature=0.6  (同上)
seq_2: temperature=1.2  (创造性生成，写作)
```

Sampler 一次 forward 处理整个 batch，`unsqueeze(1)` 让温度按行独立生效。

#### 温度效果

| temperature | 效果 | 适用场景 |
|---|---|---|
| < 1.0 (如 0.6) | 分布更尖锐，高概率 token 更突出 | 确定性任务（代码生成） |
| = 1.0 | 不做缩放，原始概率 | 默认 |
| > 1.0 (如 1.5) | 分布更平滑，低概率 token 被放大 | 创造性任务（写作） |

### 4.2 Gumbel-max 采样

nano-vllm 用 Gumbel-max 采样，比 CS336 hw6 的 top-p 采样更简洁高效。

#### 问题：如何从概率分布中采样？

```python
probs = [0.7, 0.2, 0.1]   # 三个 token 的概率分布

# 期望：以 70% 概率选中 token_0，20% 概率选中 token_1，10% 概率选中 token_2
```

**传统方法**：

| 方法 | 算法 | 复杂度 |
|---|---|---|
| `torch.multinomial(probs)` | 计算累积分布 (CDF) → 二分查找 | O(n) 建 CDF + O(log n) 采样 |
| top-p (CS336 hw6) | **排序** → 累积和 → 找到阈值位置 → 截断 → multinomial | O(n log n) — 排序是瓶颈 |
| **Gumbel-max** | `argmax(probs / exp_noise)` | **O(n)** — 一次 argmax 搞定 |

#### 数学原理

**定理**：从 Gumbel(0,1) 分布取独立噪声加到每个 log_prob 上，取最大值的索引等价于从原始类别分布中采样。

```
argmax_i (log(prob_i) + G_i)
其中 G_i ~ Gumbel(0,1) 独立同分布
等价于
从 Categorical(prob) 中采样
```

Gumbel(0,1) 分布的 CDF：`P(G ≤ x) = exp(-exp(-x))`

**证明直觉**（非严格）：

```
P(argmax 是 i) = P(log(p_i) + G_i > log(p_j) + G_j, for all j ≠ i)

固定所有 j ≠ i，已知 G_j 的具体值：
条件: G_i > log(p_j) - log(p_i) + G_j, for all j
     = G_i > max_{j≠i}[log(p_j/p_i) + G_j]

G_i ~ Gumbel(0,1):
P(G_i > t) = 1 - GumbelCDF(t) = 1 - exp(-exp(-t))

可以证明最终结果 = p_i
```

**nano-vllm 的等价变换**：

```
标准 Gumbel-max:
  argmax(log(p_i) + G_i)   其中 G_i ~ Gumbel(0,1)

等价变换（nano-vllm 的写法）:
  G_i = -log(E_i)  其中 E_i ~ Exponential(1)
  argmax(log(p_i) - log(E_i))
= argmax(log(p_i / E_i))
= argmax(p_i / E_i)    ← 这就是 nano-vllm 的 probs.div_(exponential_noise)
```

#### 代码逐行

```python
# nano-vllm sampler.py L10-11
probs = torch.softmax(logits, dim=-1)    # (bs, vocab_size)

# 一步采样
sample_tokens = probs.div_(
    torch.empty_like(probs)
    .exponential_(1)                     # ← 指数分布噪声
    .clamp_min_(1e-10)                   # ← 防止除以 0
).argmax(dim=-1)                         # ← 取最大值索引 = 采样结果
```

**Step 1: `torch.empty_like(probs).exponential_(1)`**

```python
# 创建一个和 probs 同 shape 的 tensor
# 每个元素从 λ=1 的指数分布中独立抽样
# PDF: f(x) = e^(-x), x ≥ 0
# 几乎所有值 > 0（因为 x≥0 且 x=0 概率密度为 0）

# 实际数值示例：
tensor([[0.34, 1.52, 0.08, ..., 0.91],   # bs=1, vocab=4 示意
        [0.67, 0.23, 1.89, ..., 0.45]])
```

**Step 2: `.clamp_min_(1e-10)`**

```python
# 指数分布理论上可能抽到 0（概率密度为 0，但浮点误差可能导致）
# 如果某个噪声恰好为 0 → probs / 0 = +inf
# clamp_min(1e-10) 把 < 1e-10 的值设为 1e-10 → 安全
```

**Step 3: `probs.div_(noise)`**

```python
# 等价于 log_prob + Gumbel(0,1) 噪声后的 argmax
# in-place 除法，省显存

probs / noise:
  (bs, vocab_size) / (bs, vocab_size) → (bs, vocab_size)
```

**Step 4: `.argmax(dim=-1)`**

```python
# 沿 vocab_size 维度取最大值索引
# 输出: (bs,) — 每个 seq 选出的 token_id
```

#### 具体数值示例

```python
# 假设 vocab_size=5, bs=1
probs = torch.tensor([[0.5, 0.3, 0.1, 0.07, 0.03]])

# 指数噪声
noise = torch.tensor([[2.1, 0.4, 0.15, 3.2, 0.02]])
#  clamp_min 后:  [[2.1, 0.4, 0.15, 3.2, 0.02]]

probs / noise:
# [[0.5/2.1, 0.3/0.4, 0.1/0.15, 0.07/3.2, 0.03/0.02]]
# = [[0.238, 0.75, 0.667, 0.022, 1.5]]
#                               ↑  argmax = 4 (token_4)

# 为什么 token_4 被选中？概率只有 3%！
# 因为它的噪声 (0.02) 特别小 → 除以小噪声 → 值被放大
# 这就是采样：低概率事件偶尔也会被选中
```

#### 与 CS336 hw6 top-p 采样对比

| | top-p (CS336 hw6) | Gumbel-max (nano-vllm) |
|---|---|---|
| 步骤 | sort → cumsum → 截断 → multinomial | 一次 argmax |
| 复杂度 | O(n log n) | O(n) |
| 额外内存 | 排序需要保存索引 | 只需要噪声 tensor |
| 是否可并行 | 排序步骤串行 | 完全可并行 |
| @torch.compile | 控制流多，难融合 | 纯粹的逐元素 + reduction，完美融合 |

#### 为什么推理时用 Gumbel-max 更好？

```
decode 阶段: 每个 token 都要采样
vocab_size = 151936 (Qwen3)

top-p (CS336):
  sort 151936 个元素 → O(n log n) ≈ 2.6M 次比较
  cumsum + 截断
  每一次 decode 都做一遍

Gumbel-max (nano-vllm):
  一次 exp 抽样 151936 个 → 一次 element-wise div → 一次 argmax
  全部在 GPU 上并行完成
  配合 @torch.compile 融合成单个 kernel
```

#### 实践对照：Houmo TCIM Qwen3 Demo 的采样参数

在你们公司的 `houmo-examples-xh2/apis/inferences/qwen3/demo.py` 中，可以看到完全对应的采样参数：

```python
# demo.py 的命令行参数
--temperature 1.0          # 温度缩放，和 nano-vllm Sampler 的 temperature 完全对应
                           # logits / temperature → softmax → 概率分布

--topk None                # top-k 采样：只保留概率最高的 k 个 token，其余置 0
                           # k=50 → 只从 top 50 中采样

--topp 1.0                 # top-p 采样：保留概率累积到 p 的最小集合
                           # p=0.9 → 排序后累加概率到 90%，截断后面的 token

--repetition_penalty 1.0   # 重复惩罚：对已生成的 token 降权
                           # >1.0 → 惩罚重复，倾向于多样化输出
                           # <1.0 → 鼓励重复
```

**对比 nano-vllm 和 Houmo 的采样差异**：

| | nano-vllm Sampler | Houmo Qwen3 demo |
|---|---|---|
| 温度 | ✅ `temperatures`（支持每 seq 不同温度） | ✅ `--temperature`（全局统一） |
| top-p | ❌ 用 Gumbel-max 替代 | ✅ `--topp`（传统排序+截断） |
| top-k | ❌ 不支持 | ✅ `--topk` |
| 重复惩罚 | ❌ 不支持 | ✅ `--repetition_penalty` |
| 采样方式 | Gumbel-max（GPU 并行，一步 argmax） | 传统 multinomial（CPU 侧运算） |
| 设计哲学 | 极致性能，最小 kernel | 功能丰富，工程灵活 |

**为什么 Houmo 用传统采样？**

Houmo 的推理链路是：

```
HmmQwenBase (CPU/Python)
  │  1. logits = decode_model(...)           ← NPU 上执行 .hmm
  │  2. logits 拷贝回 CPU
  │  3. SamplingManager 在 CPU 上做 top-p/top-k → 采样
  │  4. 下一个 token_id
  ▼
下一次 decode_model(token_id, ...)            ← 重新调用 .hmm
```

采样在 CPU 上做，不受 GPU kernel 融合的限制，所以可以用更复杂的功能（top-p、top-k、重复惩罚）。nano-vllm 的采样必须在 GPU 上做（配合 CUDA Graph），所以追求极简的 Gumbel-max。

**完整的采样控制链路**：

```
logits (NPU 输出)
  │
  ├── / temperature          ← 温度缩放
  │
  ├── repetition_penalty     ← 抑制重复 token（降权已出现过的）
  │
  ├── top-k 过滤             ← 只保留概率最高的 k 个
  │
  ├── top-p 过滤             ← 只保留累积概率达到 p 的最小集合
  │
  ├── softmax                ← 归一化为概率
  │
  └── multinomial 采样       ← 从截断后的分布中按概率抽样
```

---

## 五、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **量化** | softmax 涉及 exp + sum + div，精度敏感。`x - max` 减法需要高精度，bf16 下 max 找不准会导致软错误。Sampler 中 `logits.float()` 正是为了精度 |
| **Flash Attention** | online softmax 是真核心——分块处理时每个块保留局部 max 和 sum，最后 rescale。和 CS336 的 `x-max` trick 是同一个思想，只是分块后需要多次 rescale |
| **编译器** | softmax 包含 reduction（max, sum）和 element-wise（sub, exp, div），是融合优化的经典模式。`@torch.compile` 会把它们融合 |
| **TCIM Runtime** | `.hmm` 编译后 softmax 被编译成 NPU 指令序列。Attention 中的 online softmax 在编译时是否自动 tiling？ |

---

## 六、思考题

1. **为什么 `keepdim=True`？如果改成 `keepdim=False`，`x_exp / x_exp.sum(...)` 会怎样？**
   - 提示：广播维度不匹配会发生什么？参考 RMSNorm 中的解释

2. **Flash Attention 的 online softmax 和 CS336 的 `x-max` 有什么异同？**
   - 提示：Flash Attention 分块处理 K/V，每个块的局部 max 和 sum 需要 rescale 合并

3. **量化时 softmax 的精度损失主要在哪个步骤？为什么 Sampler 中先把 logits 转 float？**
   - 提示：exp 函数对输入极敏感，bf16 只有 7 位有效数字

4. **为什么 Gumbel-max 采样等价于从概率分布中采样？**
   - 提示：`argmax(log_prob + Gumbel(0,1)) ∼ Categorical(prob)`

---

## 七、CS336 课后作业

### 作业要求

实现一个数值稳定的 softmax 函数：

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        x: 任意形状的输入张量
        dim: 沿哪个维度做 softmax 归一化
    Returns:
        和 x 相同形状的概率分布张量
    """
```

### 实现要点

1. 使用 `x.max(dim, keepdim=True)[0]` 找最大值
2. `exp(x - max)` 防止上溢
3. 除以 `sum(exp(x - max))` 归一化

### 验证方法

```python
import torch
from softmax import softmax

# 基础验证
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
out = softmax(x, dim=-1)
assert out.shape == x.shape
assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)
assert torch.all(out >= 0)

# 每个值都 > 0，总和 = 1
print(f"Softmax result: {out}")
# tensor([0.0321, 0.0871, 0.2369, 0.6439])

# 批量验证
x = torch.randn(4, 5, 6)
out = softmax(x, dim=-1)
assert out.shape == x.shape
assert torch.allclose(out.sum(dim=-1), torch.ones(4, 5), atol=1e-6)

# 和 PyTorch 内置版本对比
assert torch.allclose(softmax(x, dim=-1), torch.softmax(x, dim=-1), atol=1e-6)

# 大值不溢出验证
x = torch.tensor([1000.0, 1.0, 2.0, 3.0])
out = softmax(x, dim=-1)
assert not torch.isnan(out).any()
assert not torch.isinf(out).any()
print(f"Large value test passed: {out}")
# tensor([1.0000, 0.0000, 0.0000, 0.0000])
```

### 扩展思考

1. 实现一个多模态的 softmax（batch × seq × dim，沿 dim=-1 做），验证和 `torch.softmax` 一致
2. 尝试实现 Flash Attention 中的 online softmax：给定一个 tiled Q@K^T 的结果，逐步更新 running max 和 running sum，最终 rescale。对比 naive softmax 和 online softmax 的中间值差异
