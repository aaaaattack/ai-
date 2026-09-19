# 第6课：Transformer Block + 完整模型组装

> 学习路线：阶段2 模型组装对比（CS336 hw7 ↔ nano-vllm models/qwen3.py）
> 对应文件：
> - CS336 hw3（权重外置版）：`chapter1/hw3/transformer_block.py`
> - CS336 hw7（权重内置版）：`chapter1/hw7/transformer_no_weight_block.py`
> - CS336 完整模型：`chapter1/hw7/transformermodule.py`
> - nano-vllm 工程版：`nano-vllm/nanovllm/models/qwen3.py`（Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM）

---

## 一、前面所有算子的拼图

前面学过的 5 个算子：

```
RMSNorm ✓  RoPE ✓  SwiGLU ✓  Softmax ✓  Attention ✓
```

现在把它们拼成一个个模块，串成一条流水线。一个 Transformer Block 就是：

```
TransformerBlock = PreNorm → Attention → +残差 → PreNorm → FFN → +残差
```

---

## 二、Transformer Block 的结构

```
输入 x: (batch, seq, d_model)
  │
  ├── RMSNorm(x) ──→ MultiHeadAttention ──→ +x
  │    Pre-Norm         子层1              残差1
  │                                           │
  │                                           ▼
  │                                          x1
  │
  └──→ RMSNorm(x1) ──→ SwiGLU(FFN) ──→ +x1 ──→ 输出
       Pre-Norm         子层2          残差2
```

**Pre-Norm 模式**：先归一化再计算，残差加在子层输出上。

```
Post-Norm (原始 Transformer):  x + Norm(SubLayer(x))
Pre-Norm (LLaMA/Qwen):        x + SubLayer(Norm(x))
                                 ↑ 先归一化，梯度更稳定
```

---

## 三、CS336 两个版本对比

### 3.1 hw3 权重外置版

源码：`chapter1/hw3/transformer_block.py`

```python
class TransformerBlock(nn.Module):
    def __init__(self, ...,
        attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight, attn_o_proj_weight,
        ln1_weight, ln2_weight,
        ffn_w1_weight, ffn_w2_weight, ffn_w3_weight, ...):

        # 把外部传入的权重赋给成员变量
        self.attn_q_proj_weight = attn_q_proj_weight
        ...

        # 创建算子实例，加载外部权重
        self.rms_norm1 = RMSNorm(d_model, eps=1e-5)
        self.rms_norm1.load_state_dict({"weight": ln1_weight})

        self.swiglu = SwiGLU(d_model, d_ff)
        self.swiglu.load_state_dict({
            "w1.weight": ffn_w1_weight, "w2.weight": ffn_w2_weight,
            "w3.weight": ffn_w3_weight
        })

    def forward(self, in_features):
        token_positions = torch.arange(in_features.shape[1])
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1, ..., token_positions)
        x1 = x1 + in_features                        # ← 残差
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        return x2 + x1                               # ← 残差
```

**特点**：11 个权重参数作为外部参数传入，手动 `load_state_dict`。教学目的：让你看清楚每个算子的所有权重。

### 3.2 hw7 权重内置版

源码：`chapter1/hw7/transformer_no_weight_block.py`

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, theta, device=None):
        self.rms_norm1 = RMSNorm(d_model, eps=1e-5)       # weight 自动创建为全 1
        self.rms_norm2 = RMSNorm(d_model, eps=1e-5)
        self.swiglu = SwiGLU(d_model, d_ff)               # w1,w2,w3 自动创建
        self.causal_multi_head_attention = \
            CausalMultiHeadAttentionNoWeight(d_model, n_heads, max_seq_len, theta)
            # 这个版本的 Attention 内部自带 wq, wk, wv, wo

    def forward(self, in_features):
        token_positions = torch.arange(in_features.shape[1])
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1, token_positions)
        x1 = x1 + in_features                            # ← 残差
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        return x2 + x1                                   # ← 残差
```

**特点**：权重都藏在 `nn.Linear` 里，不需要外部传入。和 nano-vllm 的工程实践对齐。唯一区别是残差没融合。

---

## 四、nano-vllm 工程版：Qwen3DecoderLayer

源码：`nano-vllm/nanovllm/models/qwen3.py`

### 4.1 完整代码

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(...)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            # 第一层：无残差输入 → norm 不带残差融合
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：残差融合进 norm
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # Post-Attention Norm + 残差（同上逻辑）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

### 4.2 与 CS336 的三个关键差异

#### 差异1：残差融合进 RMSNorm

不再是独立的 `x1 = x1 + in_features`。残差加法和 norm 合二为一：

```python
# CS336:
x1 = RMSNorm(in_features)
x1 = Attention(x1)
x1 = x1 + in_features       # 3 个 kernel

# nano-vllm:
x1, residual = RMSNorm(in_features, residual)  # 残差加法 + norm 一步完成
x1 = Attention(x1)
# 不需要 x1 + residual!  2 个 kernel
```

（第1课 RMSNorm 详细讲过，原理参考 [01_RMSNorm.md](01_RMSNorm.md)）

#### 差异2：残差外提 — 作为参数在层间流动

```
CS336: 残差在 Block 内部，"谁用谁加"
  forward(x):
    x1 = norm(x) → attn → +x
    x2 = norm(x1) → ffn → +x1
    return x2

nano-vllm: 残差在外，"层间传递的光缆"
  forward(positions, hidden_states, residual):
    hidden_states, residual = norm(hidden_states, residual)
    hidden_states = attn(hidden_states)
    hidden_states, residual = norm(hidden_states, residual)
    hidden_states = ffn(hidden_states)
    return hidden_states, residual
                  ↑              ↑
              计算流            残差流（独立通道）
```

**好处**：RMSNorm 成为一个无状态的纯函数，不依赖 Block 内部状态，更干净、更可复用。

#### 差异3：QK Norm（Qwen3 特有）

```python
# Qwen3 在 Q 和 K 投影后、RoPE 之前各加一个 RMSNorm
if not self.qkv_bias:
    q = self.q_norm(q)   # RMSNorm(head_dim)
    k = self.k_norm(k)   # RMSNorm(head_dim)
```

**为什么需要 QK Norm？**

```
Q·K 内积对 Q 和 K 的模长很敏感。
如果 Q 或 K 的模长大 → softmax 分布极尖锐 → 梯度过小。
QK Norm 把 Q 和 K 分别归一化到单位方差，稳定训练梯度。
```

LLaMA 没有这个设计，这是 Qwen3 的创新。

---

## 五、完整模型组装

### 5.1 CS336 教学版

源码：`chapter1/hw7/transformermodule.py`

```python
class TransformerModule(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, theta,
                 n_layers, vocab_size, device=None):
        # N 个 Transformer Block
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, theta, device)
            for _ in range(n_layers)
        ])
        self.embedding_module = EmbeddingModule(vocab_size, d_model, device)
        self.linear_module = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding_module(x)           # token IDs → 稠密向量
        for block in self.transformer_blocks:   # 逐层变换
            x = block(x)
        x = self.linear_module(x)              # 投影到词表大小 → logits
        return x
```

**`nn.ModuleList`**：PyTorch 的列表容器，让列表中的子模块能被 `model.parameters()` 和 `model.to(device)` 自动管理。

数据流：

```
input_ids:        [101, 2023, 2512, ...]     (batch, seq)
  │  Embedding
  ▼
embeddings:       (batch, seq, d_model)
  │  Block_0
  ▼
hidden:           (batch, seq, d_model)
  │  Block_1 ... Block_N-1
  ▼
last_hidden:      (batch, seq, d_model)
  │  LM Head (Linear)
  ▼
logits:           (batch, seq, vocab_size)
```

### 5.2 nano-vllm 工程版

源码：`nano-vllm/nanovllm/models/qwen3.py` (Qwen3Model + Qwen3ForCausalLM)

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # Weight Tying: Embedding 和 LM Head 共享权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

**Weight Tying 详解**：

```
Embedding 权重: (vocab_size, d_model) = (151936, 576) ≈ 87M 参数
LM Head 权重:   (d_model, vocab_size) = (576, 151936) ≈ 87M 参数

不共享: 174M 参数 → 348MB (bf16)
共享:    87M 参数 → 174MB (bf16) — 省了一半
```

直觉上合理："每个 token 的向量表示"和"预测这个 token 的分数"，用的可以是一组权重。

**nano-vllm 的额外工程特性**：

| 特性 | CS336 | nano-vllm |
|---|---|---|
| Embedding | `nn.Embedding` | `VocabParallelEmbedding`（支持 TP） |
| LM Head | `nn.Linear` | `ParallelLMHead`（支持 TP） |
| 最后的 Norm | Block 内部自己处理 | 独立 `self.norm` 在所有 Block 之后 |
| Weight Tying | 无 | `tie_word_embeddings=True` |
| 残差传递 | Block 内部 `+in_features` | 层间显式传递 `residual` 参数 |
| Prefill/Decode | 同一个 forward | 同一个模型，prefill 和 decode 分开调度 |

---

## 六、CS336 多个 Block 版本对照

| 文件 | 特点 | 权重位置 |
|---|---|---|
| `hw3/transformer_block.py` | 11 个外部权重参数，手动 load_state_dict | 外部 |
| `hw7/transformer_no_weight_block.py` | 权重内置在 `nn.Linear` 里 | 内部 |
| `hw7/transformer_block_without_rmsnorm.py` | 去掉 RMSNorm 的变体（实验对比用）| 内部 |

---

## 七、白话总结

```
一个 TransformerBlock = 两个 "Pre-Norm + 子层 + 残差" 串联
  子层1: Attention — "计算我该关注哪些 token"
  子层2: FFN (SwiGLU) — "从关注结果中提取有用的特征"

堆 N 个 Block = 反复交替 "理解上下文" 和 "特征变换"

前面加上 Embedding = 把 token 编号变成高维向量
后面加上 LM Head = 把最终向量变回词表概率
最后取最后一个 token 的 logits → softmax → 采样 → 下一个 token
```

---

## 八、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **量化** | Weight Tying 让 Embedding 和 LM Head 共享权重 → 量化方案需要兼顾查表和 matmul 两种操作。Residual 流的精度需要和主计算流匹配 |
| **编译器** | TransformerBlock 是编译器的**基本优化单元**。28 层的相同结构 = 同一个 Block 被调用 28 次 → 编译器可以针对 Block 做一次优化，复用到所有层 |
| **推理工具链** | Prefill 和 Decode 的 Block 执行路径相同（都是这个结构），但输入 shape 不同 → 需要分开编译（你们 Houmo 的 `prefill.hmm` 和 `decode.hmm`） |
| **驱动/HAL** | Residual 传递需要保持高精度（避免 fp16 下小值被吞掉），HAL 层的 residual add 可能需要在更高精度下执行 |

---

## 九、思考题

1. **为什么是 Pre-Norm（先归一化再计算）而不是 Post-Norm（先计算再归一化）？**
   - 提示：Post-Norm 在原始 Transformer 中训练不稳定，梯度在低层容易消失。Pre-Norm 让梯度从最后一层直通第一层

2. **QK Norm 放在 RoPE 之前还是之后？为什么？**
   - 提示：RoPE 是旋转（保持模长），放在 Norm 之后 → Q 和 K 的模长先被归一化，再做旋转，效果稳定

3. **Weight Tying 对量化有什么挑战？**
   - 提示：Embedding 是**查表操作**（读一行），LM Head 是**矩阵乘**（全矩阵）。相同权重用于两种不同操作 → 量化参数（scale/zero_point）需要同时适用于两者

4. **TransformerBlock 一共做了多少次矩阵乘？**（包括 Attention 和 FFN）
   - 提示：QKV 投影 + Output 投影 = 4 次 + FFN 的 gate_up + down = 2 次 → 6 次 matmul

---

## 十、CS336 课后作业

### 作业：组装完整的 Transformer 语言模型

```python
class TransformerModule(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, theta,
                 n_layers, vocab_size, device=None):
        """组装 Embedding → N×Block → LM Head"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) token IDs
        Returns: (batch, seq_len, vocab_size) logits
        """
```

### 验证方法

```python
# 一个小型模型
model = TransformerModule(
    d_model=512, n_heads=8, d_ff=2048,
    max_seq_len=256, theta=10000.0,
    n_layers=4, vocab_size=10000
)

x = torch.randint(0, 10000, (2, 32))  # batch=2, seq=32
logits = model(x)

assert logits.shape == (2, 32, 10000)
assert not torch.isnan(logits).any()

# 检查参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 逐个 Block 检查输出不会爆炸
x_emb = model.embedding_module(x)
for i, block in enumerate(model.transformer_blocks):
    x_emb = block(x_emb)
    print(f"Block {i}: mean={x_emb.mean():.4f}, std={x_emb.std():.4f}")
    # 如果 std 逐渐增大或减小，说明残差设计有问题
```

### 扩展思考

1. 对比 `forward` 中每个 Block 的输入和输出，验证残差连接是否工作（输入和输出不应该太"远"）
2. 尝试去掉 RMSNorm（用 `TransformerBlockWithoutRMSNorm`），观察输出 std 是否会爆炸
3. 思考：如果 `n_layers=28`，一次 forward 多少个 kernel launch？（手算或 estimate）
