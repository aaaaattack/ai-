# 第5课：Attention — 缩放点积 + 因果多头注意力

> 学习路线：阶段1 算子层对比（CS336 hw3 ↔ nano-vllm layers/）
> 对应文件：
> - CS336 教学版：`chapter1/hw3/scaled_dot_product_attention.py`
> - CS336 教学版：`chapter1/hw3/causal_multi_head_attention.py`
> - nano-vllm 工程版：`nano-vllm/nanovllm/layers/attention.py`
> - nano-vllm 调用处：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3Attention)

---

## 一、Attention 的直觉

Attention 回答一个问题：**当前 token 应该关注过去哪些 token，各关注多少？**

```
输入: "我 爱 北京 天安门"

处理到 "天安门" 时:
  "天安门" 对 "我"     的关注度: 0.05   ← 无关
  "天安门" 对 "爱"     的关注度: 0.10   ← 弱相关
  "天安门" 对 "北京"   的关注度: 0.70   ← 强相关
  "天安门" 对 "天安门" 的关注度: 0.15   ← 自己

这些权重就是 Attention 的输出。
```

---

## 二、Scaled Dot-Product Attention 核心公式

```python
out = softmax(Q @ K^T / sqrt(d_k)) @ V
```

四个步骤的数学拆解：

```
d_k = Q.shape[-1]                        # 缩放因子

S = Q @ K^T                              # Step 1: 计算相关性分数矩阵
S_scaled = S / sqrt(d_k)                 # Step 2: 缩放，防止梯度消失
A = softmax(S_scaled, dim=-1)            # Step 3: 转成概率分布（每行和=1）
out = A @ V                              # Step 4: 用注意力权重加权求和 V
```

### Step 1: `Q @ K^T` — 相关性分数矩阵

```
Q: (seq_len, d_k)      每行 = 一个 token 的 Query（"我在找什么"）
K: (seq_len, d_k)      每行 = 一个 token 的 Key（"我有什么"）

S = Q @ K^T  → (seq_len, seq_len)
     S[i, j] = Q_i 和 K_j 的内积
     = token_i "认为" token_j 有多重要

例子 (seq_len=4):
        K₀  K₁  K₂  K₃
  Q₀  [ 5   2   1   0  ]  ← token_0 对 token_0 最关注
  Q₁  [ 2   8   3   1  ]  ← token_1 对 token_1 最关注
  Q₂  [ 1   4   7   2  ]
  Q₃  [ 0   2   4   9  ]
```

### Step 2: `/ sqrt(d_k)` — 为什么需要缩放？

**问题**：d_k 越大 → Q·K 内积的方差越大 → softmax 输出趋近 one-hot → 梯度接近 0。

```python
# d_k=64  vs  d_k=1024 的对比
d_k=64:   Q·K 的典型值范围: [-8, +8]   → softmax 输出较平滑 → 梯度正常
d_k=1024: Q·K 的典型值范围: [-32, +32] → softmax 输出非常尖锐 → 梯度≈0
```

**数学原理**：假设 Q 和 K 的元素独立同分布 N(0,1)，则 `Q_i · K_j = Σ(q_k · k_k)`，方差 = d_k。除以 `sqrt(d_k)` 后方差归一化 ≈ 1，softmax 输出保持在合理范围。

**CS336 代码**：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
#                     ↑ Q@K^T是(batch,seq,seq)           ↑ 标量 sqrt(d_k)
```

### Step 3: `softmax(S, dim=-1)` — 转成概率

对每行做 softmax，输出每行的注意力百分比：

```python
attn_weights = torch.softmax(scores, dim=-1)
# dim=-1 表示沿着最后一维（key 维度）做归一化
# 结果每行 sum = 1.0
```

**配合 mask**：Causal attention 中需要阻止看到未来的 token：

```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
    # mask 中为 0 的位置 = 未来位置 → score 设为 -1e9
    # softmax(-1e9) ≈ 0 → 不关注未来
```

**为什么填 `-1e9` 不填 `0`？**

```python
# 填 0 的做法（错误）:
scores.masked_fill(mask==0, 0)
softmax([5, 0, 0]) → [0.95, 0.025, 0.025]  ← 仍然有 2.5% 注意力！❌

# 填 -1e9 的做法（正确）:
scores.masked_fill(mask==0, -1e9)
softmax([5, -1e9, -1e9]) → [1.0, 0.0, 0.0]  ← exp(-1e9)≈0 ✅
```

### Step 4: `A @ V` — 加权求和

```python
out = torch.matmul(attn_weights, V)
# attn_weights: (..., seq, seq)   注意力权重矩阵
# V:            (..., seq, d_v)   每个 token 的 Value
# out:          (..., seq, d_v)   加权后的上下文表示
```

```
out[i] = Σⱼ A[i,j] * V[j]

意思是: token_i 的输出 = 所有 token_j 的 value 按关注度加权求和
```

---

## 三、CS336 Scaled Dot-Product Attention 源码逐行解析

源码：`chapter1/hw3/scaled_dot_product_attention.py`

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 无参数 — 纯计算层

    def forward(self, Q, K, V, mask=None):
        # 1. 获取 d_k 作为缩放因子
        d_k = Q.shape[-1]

        # 2. Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))

        # 3. 应用因果 mask（如果提供了）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. softmax → 注意力权重
        attn_weights = torch.softmax(scores, dim=-1)

        # 5. 加权求和 V
        return torch.matmul(attn_weights, V)
```

---

## 四、CS336 Causal Multi-Head Attention 源码逐行解析

源码：`chapter1/hw3/causal_multi_head_attention.py`

### 4.1 初始化

```python
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.d_model = d_model    # 总维度，如 512
        self.n_heads = n_heads    # 头数，如 8
        self.head_dim = d_model // n_heads  # 每个头的维度 = 64
```

**为什么用 `head_dim` 而不是 `d_k`？**

```
d_model = n_heads × head_dim
        = 8       × 64
        = 512

每个头拿到的是 1/8 的维度，独立做 Attention，最后拼回来。
```

### 4.2 forward：QKV 投影 + 切分多头 + Attention + 拼接

```python
    def forward(self, x, wq, wk, wv, wo):
        batch_size, seq_len, d_model = x.shape
```

#### QKV 投影

```python
        q = x @ wq.T  # (bs, seq, 512) @ (512, 512) → (bs, seq, 512)
        k = x @ wk.T
        v = x @ wv.T
```

注意：CS336 教学版把权重当作外部参数传入，而不是 `nn.Linear` 成员变量。这是为了让学生看清计算流程。

#### 切分成多头

```python
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (bs, seq, 512) → (bs, seq, 8, 64)

        q = q.transpose(1, 2)
        # (bs, 8, seq, 64)
        # 把 n_heads 维度提到前面，便于并行计算
```

**`view + transpose` 详解**：

```python
# view 所做的事情：把最后一维 (d_model=512) 拆成 (n_heads=8, head_dim=64)
# (bs, seq, 512) → view → (bs, seq, 8, 64)
# 因为是连续内存，view 只是"换个视角看"，不移动数据
# 等价于把原始 [a₁a₂...a₆₄ | b₁b₂...b₆₄ | ... | h₁h₂...h₆₄]
# 重新解释为 [head₀, head₁, ..., head₇]

# transpose(1, 2): 交换 seq 和 n_heads 维度
# (bs, seq, 8, 64) → (bs, 8, seq, 64)
# 这样可以用 batch matmul 一次性对所有头算 Attention
```

**为什么 transpose 后能并行？**

```python
# transpose 后形状: (bs, 8, seq, 64)
# PyTorch 的 matmul 支持 batch 维度：torch.matmul(Q, K.transpose(-2,-1))
# 自动对 (bs, 8) 维度上的所有头并行计算
```

#### 创建 Causal Mask

```python
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        # triu(..., diagonal=1): 保留对角线以上（不含对角线）的元素
        #
        # seq_len=4:
        # [[F, T, T, T],    ← token_0 不能看 1,2,3（未来）
        #  [F, F, T, T],    ← token_1 能看 0 和 1，不能看 2,3
        #  [F, F, F, T],    ← token_2 能看 0,1,2，不能看 3
        #  [F, F, F, F]]    ← token_3 都能看

        mask = mask.unsqueeze(0).unsqueeze(0)
        # (seq, seq) → (1, 1, seq, seq)
        # 方便广播到 (batch, heads, seq, seq) 形状
```

**`torch.triu` 语法**：

```python
torch.triu(input, diagonal=0)
# 保留上三角（含对角线及以上）

diagonal=0:  保留对角线及以上
diagonal=1:  保留对角线上方（不含对角线）
diagonal=-1: 保留对角线上方（含对角线上一行）
```

#### Attention + 拼接 + 输出投影

```python
        out = self.attention(q, k, v, mask)
        # 每个头独立做 Attention → (bs, 8, seq, 64)

        out = out.transpose(1, 2)
        # (bs, 8, seq, 64) → (bs, seq, 8, 64)

        out = out.contiguous().view(batch_size, seq_len, d_model)
        # .contiguous() 确保内存连续（transpose 后内存可能不连续）
        # .view(...) → (bs, seq, 512)
        # 把 8×64 重新合并回 512

        out = out @ wo.T
        # 输出投影: (bs, seq, 512) @ (512, 512) → (bs, seq, 512)
        return out
```

**`.contiguous()` 为什么需要？**

```python
# transpose 只交换了 strides（步长），不移动数据
# 内存实际上还是 (bs, 8, seq, 64) 的布局
# view 要求内存连续 → 必须先 .contiguous() 复制一份连续内存
```

---

## 四-A、CS336 Causal Multi-Head Attention 的三个变体

CS336 有三个 Attention 实现，渐进式演进：

```
v1: CausalMultiHeadAttention          (hw3)  无 RoPE，权重外置
v2: CausalMultiHeadAttentionWithRoPE  (hw3) + RoPE，权重外置
v3: CausalMultiHeadAttentionNoWeight  (hw7) + RoPE，权重内置 (nn.Linear)
```

核心差异只有两点：**RoPE 的有无** 和 **权重的存放方式**。

### 4A.1 v1：CausalMultiHeadAttention（无 RoPE，权重外置）

源码：`chapter1/hw3/causal_multi_head_attention.py`

已在前面第四节完整解析。forward 签名：

```python
def forward(self, x, wq, wk, wv, wo) -> torch.Tensor:
    # x: (bs, seq, d_model)
    # wq,wk,wv,wo: (d_model, d_model)  ← 外部传入
    ...
    out = self.attention(q, k, v, mask)    # 无 RoPE
    ...
    return out @ wo.T
```

### 4A.2 v2：CausalMultiHeadAttentionWithRoPE（+ RoPE，权重外置）

源码：`chapter1/hw3/causal_multi_head_attention_with_rope.py`

**变化**：forward 签名多了 `token_positions`，切分多头后对 Q 和 K 做 RoPE。

```python
class CausalMultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, theta, device=None):
        self.head_dim = d_model // n_heads
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device)
        # ⚠️ 注意：RoPE 用的维度是 head_dim，不是 d_model！
        # 因为 RoPE 在"切分多头之后"应用，每个头独立旋转
```

**forward**：

```python
    def forward(self, x, wq, wk, wv, wo, token_positions) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Step 1: QKV 投影（同 v1）
        q = x @ wq.T   # (bs, seq, d_model)
        k = x @ wk.T
        v = x @ wv.T

        # Step 2: 切分多头（同 v1）
        q = q.view(bs, seq, self.n_heads, self.head_dim)
        k = k.view(bs, seq, self.n_heads, self.head_dim)
        v = v.view(bs, seq, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # (bs, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ═══════ Step 2.5: RoPE ═══════ ← 唯一新增的步骤！
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        # 对每个头的 Q 和 K 分别做旋转位置编码
        # Q 形状: (bs, n_heads, seq, head_dim)
        # RoPE 在每个头的 head_dim 维度上做旋转

        # Step 3: Causal Mask（同 v1）
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Step 4: Attention + 拼接 + 输出（同 v1）
        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(bs, seq, d_model)
        return out @ wo.T
```

**与 v1 的唯一差异**：

```
v1:  QKV → view → transpose → attention → transpose → view → output
v2:  QKV → view → transpose → [RoPE Q, K] → attention → transpose → view → output
                                  ↑ 只多了这两行
```

**为什么 RoPE 在 view+transpose 之后？**

```python
# 切分前: (bs, seq, d_model=512)  ← 不能直接做 RoPE
# 切分后: (bs, n_heads=8, seq, head_dim=64)  ← 每个头独立旋转

# RoPE 的 theta=10000, d_k=head_dim=64
# 每个头有自己的频率空间，8 个头独立做旋转
# 这正是 nano-vllm 的做法（qwen3.py L85: q,k = self.rotary_emb(positions, q, k)）
```

**RoPE 参数为什么用 `head_dim` 而不是 `d_model`？**

```python
# d_model = 512, n_heads = 8, head_dim = 64

# 如果用 d_model=512 → 频率分布: f_i = 1/10000^(i/256)  (256 对)
# 如果只对前 64 维做 RoPE（实际上整个 d_model 被切成 8 份）
# → 每个头只有 64/2=32 对 → 应该对应 32 个频率

# 正确做法: RoPE(theta, head_dim=64, ...)
# → 频率分布: f_i = 1/10000^(i/32)  (每个头 32 对，互不干扰)
```

### 4A.3 v3：CausalMultiHeadAttentionNoWeight（+ RoPE，权重内置）

源码：`chapter1/hw7/causal_multi_head_attention_no_weight.py`

**变化**：`wq, wk, wv, wo` 不再外部传入，改为 `nn.Linear` 成员变量。

```python
class CausalMultiHeadAttentionNoWeight(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, theta, device=None):
        self.head_dim = d_model // n_heads
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device)

        # 四个投影层改为 nn.Linear（权重内置，自动管理）
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, token_positions) -> torch.Tensor:
        # 直接调用 self.wq(x)，不再传入外部权重
        q = self.wq(x)  # nn.Linear 内部做 x @ wq.weight.T
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bs, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        mask = torch.triu(...)
        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(bs, seq, d_model)
        return self.wo(out)
```

**与 v2 的区别**：forward 签名简化（`(x, token_positions)` 代替 `(x, wq, wk, wv, wo, token_positions)`），更接近 nano-vllm 的工程实践。

### 4A.4 三版对照表

| | v1 (无 RoPE) | v2 (+ RoPE) | v3 (+ RoPE 内置权重) |
|---|---|---|---|
| **文件** | `hw3/causal_multi_head_attention.py` | `hw3/causal_multi_head_attention_with_rope.py` | `hw7/causal_multi_head_attention_no_weight.py` |
| **RoPE** | ❌ | ✅ | ✅ |
| **权重方式** | 外部传入 4 个参数 | 外部传入 4 个参数 | `nn.Linear` 内置 |
| **forward 签名** | `(x, wq, wk, wv, wo)` | `(x, wq, wk, wv, wo, token_positions)` | `(x, token_positions)` |
| **RoPE 调用** | 无 | `self.rope(q, pos); self.rope(k, pos)` | 同 v2 |
| **RoPE 参数** | — | `head_dim = d_model // n_heads` | 同 v2 |
| **教学目的** | 最小 Attention 实现 | 演示 RoPE 怎么嵌入 Attention | 演示工程化封装方式 |

### 4A.5 为什么 RoPE 对 Attention 如此重要？

```
无 RoPE 的 Attention:
  Q·K^T 内积只取决于 token 的内容，不感知 token 的位置
  "我打你" 和 "你打我" 的 Attention 分数矩阵只是做了转置，语义完全不同但模型感受不到

有 RoPE 的 Attention:
  Q 和 K 先按各自位置做旋转 → Q·K^T 的内积隐藏了相对位置信息
  token_3 (Q) · token_0 (K) 和 token_8 (Q) · token_5 (K)
  如果相对距离都是 3，旋转后的内积就包含相同的相对位置信息
```

这就是为什么所有现代 LLM（LLaMA、Qwen、ChatGLM）都标配 RoPE。

---

## 五、nano-vllm 工程版：直接用 Flash Attention

源码：`nano-vllm/nanovllm/layers/attention.py`

nano-vllm 不手写 matmul + softmax + matmul，而是直接调 `flash_attn` 库：

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads          # Q 头数
        self.head_dim = head_dim
        self.scale = scale                  # 1/sqrt(head_dim)
        self.num_kv_heads = num_kv_heads    # KV 头数（GQA: 可小于 Q 头数）
        self.k_cache = self.v_cache = torch.tensor([])  # KV Cache 引用

    def forward(self, q, k, v):
        # 1. 存储 KV 到 Cache（如果 cache 已分配）
        if k_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. 分 prefill 和 decode 两路
        if context.is_prefill:
            # Prefill: 用 var-len Flash Attention（支持变长序列）
            o = flash_attn_varlen_func(q, k, v, ...)
        else:
            # Decode: 用 KV Cache 的 Flash Attention（只算当前 1 个 token）
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, ...)

        return o
```

**为什么不用手写 Attention？**

手写 Attention 的 `Q@K^T` 产生 `(seq, seq)` 的中间矩阵，需要 O(n²) 显存：

```
seq_len=1024: Q@K^T = (1024, 1024) × 2 bytes = 2MB   ← 还行
seq_len=4096: Q@K^T = (4096, 4096) × 2 bytes = 32MB  ← 8 个头 × 32MB = 256MB
seq_len=32K:  Q@K^T = 2GB ← 单一层就爆显存
```

Flash Attention 通过 tiling（分块）把中间矩阵留在 SRAM 里，不写回 HBM。这是 Chapter 2 的核心内容。

**在模型中的调用**（Qwen3Attention）：

```python
# nano-vllm/models/qwen3.py
qkv = self.qkv_proj(hidden_states)       # 合并的 QKV 投影
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)  # RoPE 旋转
o = self.attn(q, k, v)                   # Flash Attention
output = self.o_proj(o.flatten(1, -1))   # 输出投影
```

---

## 六、CS336 vs nano-vllm 差异总结

| 差异点 | CS336 教学版 | nano-vllm 工程版 | 意义 |
|---|---|---|---|
| **Attention 计算** | 手写 matmul + div + softmax + matmul | 调 `flash_attn_varlen_func` / `flash_attn_with_kvcache` | O(n²)→O(n) 显存 |
| **QKV 投影** | 三个独立权重参数，外部传入 | `QKVParallelLinear` 合并投影 + TP | 少 kernel launch |
| **头数** | `n_heads` 固定，Q=K=V 头数相同 | `num_heads` ≠ `num_kv_heads`（GQA） | Q 多头，KV 少头，省显存 |
| **KV Cache** | 无 | `store_kvcache` Triton kernel 写入 paged cache | 支持长序列推理 |
| **权重绑定** | 外部传入 `wq, wk, wv, wo` | `nn.Linear` 成员变量 + `weight_loader` | 自动从 HF checkpoint 加载 |
| **Prefill/Decode** | 不区分（统一用同一个 Attention） | 分成两路：`varlen_func` vs `with_kvcache` | 推理性能优化 |

---

## 七、白话总结

```
Attention 就三步：
  1. 拿我的"查询 Q"去匹配所有人的"钥匙 K"，算出相似度分数
  2. 分数经过缩放+mask+softmax，转成注意力百分比
  3. 用百分比加权所有人的"内容 V"，得到我的注意力输出

Multi-Head = 分成 8 份，每份独立做一遍 Attention，拼回来

Causal = 不能看未来，用 mask 把未来位置的分数置为 -inf → softmax 后 = 0
```

---

## 八、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **量化** | `1/sqrt(d_k)` 缩放因子对精度敏感，量化时 `softmax(QK^T/scale + bias)` 需要高精度计算。QK 的 scale 和 bias 在 W8A8 中通常是单独处理的 calibration 参数 |
| **编译器** | Attention 的 `Q@K^T → scale → mask → softmax → @V` 是编译器融合的最重要目标之一。Flash Attention 本身就是一个极致融合的例子 |
| **推理工具链** | Prefill 的 `flash_attn_varlen_func` 和 Decode 的 `flash_attn_with_kvcache` 是两个完全不同特性的 kernel（计算密集 vs 访存密集），在你们的 `.hmm` 中需要分开编译 |
| **驱动/HAL** | Flash Attention 的 tiling 策略依赖 SRAM 大小，HAL 层需要提供准确的 SRAM 容量给编译器做 tiling size 决策 |
| **TCIM Runtime** | 你们 Qwen3 demo 的 prefill.hmm 和 decode.hmm 分开编译，正是因为 prefill 和 decode 的 Attention 计算模式完全不同 |

---

## 九、思考题

1. **为什么 `masked_fill(..., -1e9)` 而不是 `0`？**
   - 提示：`softmax([5, 0, 0])` 和 `softmax([5, -inf, -inf])` 结果一样吗？

2. **如果不缩放（去掉 `/sqrt(d_k)`），d_k=1024 时会有什么问题？**
   - 提示：Q·K 内积的方差 ≈ d_k，方差大了 softmax 会怎样？

3. **Multi-Head 时，8 个头算出来的 (64,) 值是独立互不干扰的吗？**
   - 提示：QKV 投影矩阵把 d_model 映射到 d_model，再切分成 8 份。每个头的输入是否包含其他头的信息？

4. **Flash Attention 的 tiling 为什么能省显存？省了多少？**
   - 提示：对比 O(n²) 和 O(n) 的中间变量大小

---

## 十、CS336 课后作业

### 作业 1：Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_k)
        mask: (batch, seq_len, seq_len) or None

        Returns: (batch, seq_len, d_k)
        """
```

### 作业 2：Causal Multi-Head Attention

```python
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        ...

    def forward(self, x, wq, wk, wv, wo):
        """
        x: (batch, seq_len, d_model)
        wq, wk, wv: (d_model, d_model)
        wo: (d_model, d_model)

        Returns: (batch, seq_len, d_model)
        """
```

### 验证方法

```python
# Scaled Dot-Product Attention 验证
d_k = 64
sdpa = ScaledDotProductAttention()
Q = torch.randn(2, 10, d_k)
K = torch.randn(2, 10, d_k)
V = torch.randn(2, 10, d_k)

out = sdpa(Q, K, V)
assert out.shape == (2, 10, d_k)
assert not torch.isnan(out).any()

# 验证每行的注意力权重和为 1（mask=None 时）
mask = None
out = sdpa(Q, K, V, mask)

# Causal Multi-Head Attention 验证
d_model, n_heads = 512, 8
mha = CausalMultiHeadAttention(d_model, n_heads)
wq = torch.randn(d_model, d_model)
wk = torch.randn(d_model, d_model)
wv = torch.randn(d_model, d_model)
wo = torch.randn(d_model, d_model)
x = torch.randn(2, 10, d_model)

out = mha(x, wq, wk, wv, wo)
assert out.shape == x.shape

# 梯度验证
loss = out.sum()
loss.backward()
assert wq.grad is not None
```

### 扩展思考

1. 尝试修改 Multi-Head Attention，把 QKV 投影权重改为 `nn.Linear` 成员变量（nano-vllm 的做法），对比两种实现
2. 用一个简单序列（如 `[1,2,3,4]`）手算一遍 Attention，验证 causal mask 的效果：每个 token 只能关注自己和它之前的 token
3. 思考：GQA（num_kv_heads < num_heads）时，K 和 V 怎么从 4 个头扩展到 8 个 Q 头？（答案：复制/插值）
