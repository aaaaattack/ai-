
## 阶段 4 · 第 2 讲：EAGLE 的 Hidden Feature 与 Token Embedding 融合

上一讲介绍了 EAGLE 的核心思想：

```text
Target hidden state
        ↓
EAGLE Draft Network
        ↓
预测下一个 hidden feature
        ↓
映射为 token
```

本节重点是：EAGLE 如何同时利用：

```text
Target hidden feature
Token embedding
```

---

### 1. 为什么需要两类输入？

假设当前已经生成了 token：

```text
x_t
```

Target Model 提供：

```text
h_t
```

其中：

- `x_t` 表示明确的离散 token；
- `h_t` 表示上下文语义、语法和预测状态。

如果只使用 `h_t`：

```text
h_t → h_{t+1}
```

Draft Network 可能缺少当前 token 的明确身份。

如果只使用 token embedding：

```text
embedding(x_t) → h_{t+1}
```

又会丢失 Target Model 已经计算出的上下文信息。

因此 EAGLE 将两者结合：

```text
embedding(x_t) + Target hidden feature h_t
        ↓
Draft Network
        ↓
预测下一位置 feature
```

---

### 2. Token Embedding 的来源

EAGLE Draft Network 中定义了自己的 embedding：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:486
```

核心代码：

```python
self.embed_tokens = nn.Embedding(
    config.vocab_size,
    config.hidden_size,
    self.padding_idx
)
```

在 Draft Network forward 中：

```python
inputs_embeds = self.embed_tokens(input_ids)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:600-610
```

这一步将离散 token 转换为连续向量：

```text
token id
   ↓
embedding lookup
   ↓
token embedding
```

---

### 3. Hidden Feature 的维度转换

Target Model 和 EAGLE Draft Network 的 hidden size 不一定相同。

源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:531-535
```

```python
if hasattr(config, "target_hidden_size"):
    self.fc = nn.Linear(
        config.target_hidden_size * 3,
        self.hidden_size,
        bias=False
    )
else:
    self.fc = nn.Linear(
        config.hidden_size * 3,
        self.hidden_size,
        bias=False
    )
```

实际 forward 中：

```python
if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
    hidden_states = self.fc(hidden_states)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:637-640
```

作用是：

```text
Target feature dimension
        ↓ Linear projection
EAGLE Draft hidden dimension
```

可以写成：

```text
h'_t = W_f h_t
```

其中：

- `h_t`：Target hidden feature；
- `W_f`：维度映射层；
- `h'_t`：Draft Network 使用的 feature。

---

### 4. EAGLE 不是简单相加

EAGLE 的一个重要实现细节是：

```text
token embedding 和 hidden feature 不是直接相加
```

在 Draft Decoder Layer 中，源码执行：

```python
hidden_states = self.hidden_norm(hidden_states)

hidden_states = torch.cat(
    (input_emb, hidden_states),
    dim=-1
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:401-452
```

核心逻辑：

```python
hidden_states = self.hidden_norm(hidden_states)

hidden_states = torch.cat(
    (input_emb, hidden_states),
    dim=-1
)
```

因此输入结构是：

```text
[input embedding ; Target hidden feature]
```

而不是：

```text
input embedding + Target hidden feature
```

拼接后的维度大致是：

```text
2 × hidden_size
```

所以后续 Attention 的输入投影层也需要接收更大的维度。

---

### 5. 为什么使用拼接？

如果直接相加：

```text
z = e_t + h_t
```

那么两类信息会混合到同一个向量空间中，模型无法明确区分：

```text
哪些维度来自 token identity
哪些维度来自 Target context
```

使用拼接：

```text
z = [e_t ; h_t]
```

可以保留两种信息的独立结构：

```text
z = [
    token identity information,
    target contextual information
]
```

Draft Network 后续可以自行学习：

```text
如何使用 token 信息
如何使用 Target hidden 信息
二者应该如何交互
```

---

### 6. Draft Decoder Layer 的输入结构

EAGLE 中的 Draft Decoder Layer 使用：

```text
input_emb
hidden_states
```

源码中的 Attention 投影层：

```python
self.q_proj = nn.Linear(
    self.hidden_size * 2,
    self.num_heads * self.head_dim,
    bias=False
)

self.k_proj = nn.Linear(
    self.hidden_size * 2,
    self.num_key_value_heads * self.head_dim,
    bias=False
)

self.v_proj = nn.Linear(
    self.hidden_size * 2,
    self.num_key_value_heads * self.head_dim,
    bias=False
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:191-212
```

这说明 Draft Attention 明确接收拼接后的双倍维度输入。

结构可以画成：

```text
token embedding ─────────────┐
                             ├── concat ── Attention ── MLP
Target hidden feature ───────┘
```

---

### 7. EAGLE 的单步 Draft 过程

假设当前已有：

```text
token x_t
Target feature h_t
```

一轮 Draft 计算可以抽象为：

```python
input_emb = embed_tokens(x_t)

h_t = normalize(h_t)

z_t = concat(input_emb, h_t)

draft_hidden = DraftDecoder(z_t)

draft_logits = lm_head(norm(draft_hidden))

x_hat_next = topk(draft_logits)
```

对应关系：

```text
input_ids
    ↓
embed_tokens()
    ↓
input_emb

Target hidden_states
    ↓
fc()
    ↓
hidden_states

input_emb + hidden_states
    ↓
concat
    ↓
Draft Decoder
    ↓
lm_head
    ↓
candidate token
```

---

### 8. EAGLE 的自回归 Draft

EAGLE 并不是一次性从一个 hidden state 直接生成所有 token。

它会重复执行：

```text
预测 feature
    ↓
预测 token
    ↓
token 转 embedding
    ↓
与新 feature 融合
    ↓
预测下一个 feature
```

即：

```text
h_t, x_t
    ↓
ĥ_{t+1}, x̂_{t+1}

ĥ_{t+1}, x̂_{t+1}
    ↓
ĥ_{t+2}, x̂_{t+2}

ĥ_{t+2}, x̂_{t+2}
    ↓
ĥ_{t+3}, x̂_{t+3}
```

这也是 EAGLE 与 Medusa 的重要区别：

```text
Medusa：
多个 head 直接对同一个 Target hidden state 预测

EAGLE：
使用前一步预测的 feature 和 token，继续进行 Draft 自回归
```

---

### 9. `topK_genrate()` 中的融合过程

EAGLE 的候选树生成函数：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:670-825
```

第一次 Draft 计算：

```python
out_hidden, past_key_values = self(
    hidden_states,
    input_ids=input_ids,
    use_cache=True,
)
```

取最后一个 hidden feature：

```python
last_hidden = out_hidden[:, -1]
```

映射为 token logits：

```python
last_headout = self.lm_head(
    self.norm(last_hidden)
)
```

获得 top-k token：

```python
top = torch.topk(
    last_p,
    top_k,
    dim=-1
)
```

下一层继续使用：

```python
input_hidden = out_hidden[:, out_ids]
input_ids = topk_index.view(-1)[topk_cs_index][None]
```

这两个变量分别表示：

```text
input_hidden：
下一轮 Draft 的 hidden feature

input_ids：
下一轮 Draft 的 token 输入
```

因此 EAGLE 每层都会同时维护：

```text
candidate token
candidate hidden feature
candidate score
candidate parent
```

---

### 10. EAGLE-3 的多层 Hidden Feature

在 EAGLE-3 分支中，代码会拼接多个 Target hidden state：

```python
hidden_states = torch.cat(
    outputs["hidden_states"],
    dim=-1
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:246-253
```

其逻辑是：

```text
Target layer 1 feature
Target layer 2 feature
Target layer 3 feature
        ↓
concat
        ↓
EAGLE Draft Network
```

与只使用单层 feature 相比，多层 feature 可以提供不同抽象层次的信息：

```text
浅层：
局部 token、词法和短程信息

中层：
句法和局部语义

深层：
全局上下文和任务相关信息
```

这也是 EAGLE-3 中 feature 输入维度更大的原因。

---

### 11. 与 Medusa 的直接对比

#### Medusa

```text
Target hidden state
        ↓
Head 1 → token logits
Head 2 → token logits
Head 3 → token logits
        ↓
候选树
```

特点：

```text
预测结构简单
多个 head 并行
主要输出 token logits
```

#### EAGLE

```text
Target hidden state
        +
token embedding
        ↓
Draft Decoder
        ↓
预测 hidden feature
        ↓
lm_head
        ↓
candidate token
        ↓
继续下一轮 Draft
```

特点：

```text
Draft 过程更深
存在 hidden feature 自回归
存在 Draft KV Cache
可以通过 feature prediction 提高候选质量
```

---

### 12. 本节核心结论

需要记住：

1. EAGLE 同时使用 token embedding 和 Target hidden feature；
2. hidden feature 会先经过维度映射；
3. 两类信息通过 concat 融合，而不是简单相加；
4. Draft Network 预测 feature，再通过 `lm_head` 得到 token；
5. 上一步生成的 feature 和 token 会共同参与下一步预测；
6. EAGLE-3 会拼接多个 Target 层的 hidden feature；
7. Medusa 主要是多头 token 预测，EAGLE 更像一个 feature-level Draft Model。

下一部分是：

**EAGLE-1、EAGLE-2、EAGLE-3 的演进，以及它们如何改进候选树和 acceptance rate。**