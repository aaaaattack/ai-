# 第 08 课｜L2.3 RoPE：让 Attention 知道 token 顺序

## 一、为什么 Q/K/V 还不够

上一课的 Attention 会根据 Q 与 K 的相似度分配注意力，但单靠 Q/K/V 并不知道 token 的绝对位置或相对距离。

例如把两个 token 交换位置，若没有任何位置信息，Attention 只能看到两组内容特征换了顺序，无法天然理解“谁在前、谁在后、相隔多远”。语言、代码和动作序列都依赖这种顺序。

因此，位置编码的目标是把 token 位置融入 Attention 分数：

```text
attention score
= 位置处理后的 Q 与 位置处理后的 K 的相似度
```

## 二、RoPE 的直觉：按位置旋转 Q/K 的二维小对

RoPE 是 Rotary Position Embedding（旋转位置编码）。它把 Q/K 的最后一维两两配对；每对可看作二维平面中的一个小向量：

```text
(x0, x1), (x2, x3), (x4, x5) ...
```

对不同 token 位置，RoPE 让每个二维小向量旋转不同角度。位置越靠后，旋转角度越大；不同维度对的旋转速度也不同。

```text
token position 0：Q/K 不旋转或旋转 0 度
token position 1：Q/K 旋转一小段角度
token position 2：Q/K 再多旋转一段角度
```

二维旋转的形式是：

```text
[x0']   [ cos(θ)  -sin(θ) ] [x0]
[x1'] = [ sin(θ)   cos(θ) ] [x1]
```

这里不要求手算三角函数。需要抓住两点：

1. RoPE 只处理 Q 和 K，不处理 V；
2. Q/K 的旋转角度依赖 token 位置，所以 Q·K 的结果会包含位置信息。

## 三、为什么它能表达相对位置

假设 Query 在位置 `query_position`，Key 在位置 `key_position`。RoPE 分别给它们旋转对应角度后，点积会与二者的角度差有关；这个差由二者的位置差决定：

```text
relative_position = query_position - key_position
```

所以模型在计算“当前 token 应该关注哪个历史 token”时，既利用内容相似度，也能利用两者相隔多远、先后关系如何。

## 四、和 nano-vLLM Qwen3 的对应

在 `llm_learning/nano-vllm/nanovllm/models/qwen3.py` 中，Q/K 投影并 reshape 后，会调用 RoPE：

```python
q, k = self.rotary_emb(positions, q, k)
```

随后才进入 Attention：

```python
o = self.attn(q, k, v)
```

这和上面的概念严格对应：V 没有经过 RoPE；旋转后的 Q/K 决定 attention score。

### 逐行注释版：nano-vLLM 的 RoPE 最小实现

来源：`llm_learning/nano-vllm/nanovllm/layers/rotary_embedding.py`。

```python
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))  # 每个二维子空间的角频率
t = torch.arange(max_position_embeddings, dtype=torch.float)                    # 预先建立所有位置的整数序列
freqs = torch.einsum("i,j -> ij", t, inv_freq)                                  # 位置 × 频率得到旋转角度
cos = freqs.cos()                                                                # 预计算 cos 查找表
sin = freqs.sin()                                                                # 预计算 sin 查找表
cos_sin = torch.cat((cos, sin), dim=-1).unsqueeze_(1)                             # 拼成 [position, 1, rotary_dim]
cos, sin = cos_sin[positions].chunk(2, dim=-1)                                  # 按本轮 position id 取出角度
x1, x2 = torch.chunk(x.float(), 2, dim=-1)                                      # 转 FP32 后拆成二维旋转的两半
y1 = x1 * cos - x2 * sin                                                        # 二维旋转后的第一半
y2 = x2 * cos + x1 * sin                                                        # 二维旋转后的第二半
return torch.cat((y1, y2), dim=-1).to(x.dtype)                                  # 合并并恢复输入 dtype
```

工程上不必每次手推三角恒等式，但要能核对三件事：`positions` 是否连续、cos/sin 是否与 `head_dim` 对齐、V 是否没有经过这段函数。

## 五、第一段检查题

只回答两点：

1. RoPE 处理 Q、K、V 中的哪些 Tensor？
Q K
2. 它要解决 Attention 的什么缺失能力？
无法理解位置信息

**批改：两题均正确。** 更完整地表述：RoPE 只作用于 Q 和 K，不作用于 V；它为 Attention 注入 token 的顺序、绝对位置变化以及 Query 与 Key 之间的相对位置信息，使模型能够区分“谁在前、谁在后、相隔多远”。

第 08 课第一段通过。下一段学习二维旋转公式中 `cos`、`sin` 与 `query_position/key_position` 的关系。

## 六、第二段：二维旋转到底做了什么

先只看最后一维中的一对数，不看完整 Tensor。设一个二维小向量为：

```text
x = (x0, x1)
```

在 token 位置 `token_position`，RoPE 给它一个角度 `θ(token_position)`，然后应用旋转矩阵：

```text
x0' = x0 × cos(θ) - x1 × sin(θ)
x1' = x0 × sin(θ) + x1 × cos(θ)
```

这一步不会改变二维向量的长度，只会改变它在平面中的方向。不同位置使用不同角度，因此同一个内容方向在不同位置会变成不同方向。

### 6.1 一个最小例子

设：

```text
x = (1, 0)
```

若位置角度为 `0°`：

```text
(1, 0) → (1, 0)
```

若位置角度为 `90°`，因为 `cos(90°)=0、sin(90°)=1`：

```text
x0' = 1×0 - 0×1 = 0
x1' = 1×1 + 0×0 = 1

(1, 0) → (0, 1)
```

所以可以把 RoPE 想成：位置不直接拼接到 token 向量后面，而是改变 Q/K 向量内部各二维分量的方向。

### 6.2 为什么 Q/K 的点积包含相对位置

假设 Query 在 `query_position`，Key 在 `key_position`。它们分别旋转后再做点积：

```text
rotate(Q, θ_query) · rotate(K, θ_key)
```

二维旋转有一个重要性质：两个向量旋转后的夹角，等于原始夹角加上两个旋转角度之差。因此点积会依赖：

```text
θ_query - θ_key
```

而角度由位置决定，所以这个差进一步对应：

```text
query_position - key_position
```

这就是 RoPE 能表达相对位置的原因。它不是把“位置编号”作为一个新维度硬拼进去，而是把位置差编码进 Q/K 的相似度计算。

### 6.3 回到完整 Q/K Tensor

如果：

```text
Q = [batch_size, num_heads, sequence_length, head_dim]
K = [batch_size, num_kv_heads, sequence_length, head_dim]
```

RoPE 沿着最后的 `head_dim` 两两配对并旋转。它不改变 Tensor shape：

```text
Q after RoPE   : [batch_size, num_heads,    sequence_length, head_dim]
K after RoPE   : [batch_size, num_kv_heads, sequence_length, head_dim]
V              : unchanged
```

### 第二段检查题

只回答两点：

1. 向量 `(1,0)` 经过 `90°` 旋转后变成什么？
2. Query 在位置 5、Key 在位置 2 时，RoPE 让 attention score 感知到的相对位置差是多少？

**我的回答：**

```text
不会。
```

**参考答案与讲解：**

1. `(1,0)` 经过 `90°` 旋转后是 `(0,1)`。

因为：

```text
cos(90°)=0，sin(90°)=1
x0' = 1×0 - 0×1 = 0
x1' = 1×1 + 0×0 = 1
```

2. 相对位置差是：

```text
query_position - key_position = 5 - 2 = 3
```

这不表示 RoPE 直接把整数 `3` 拼进 Tensor，而是 Query 和 Key 的旋转角度之差对应了位置差 3，进而影响它们的点积。

## 七、工程实践中需要推导到什么程度

写推理代码、排查精度和性能问题时，通常**不需要每次手撕完整三角函数推导**。但是只把 RoPE 当成完全不透明的黑盒也不够。建议分成三层掌握：

### 第一层：必须掌握的工程不变量

这些内容足以覆盖大多数模型接入、导图和精度排查：

- RoPE 作用于 Q/K，不作用于 V；
- RoPE 不改变 Q/K 的 shape；
- 位置索引会影响旋转角度；
- Query 与 Key 的位置差会影响 attention score；
- Decode 时，新 token 的 position id 必须连续且与 KV Cache 中历史位置一致；
- GQA 中，Q 的 `num_heads` 与 K 的 `num_kv_heads` 可以不同，但 RoPE 仍分别沿最后的 `head_dim` 处理 Q/K。

### 第二层：遇到问题时必须会检查

当出现长上下文、Prefill/Decode 不一致或 NPU 对齐问题，应检查：

```text
position_ids 的起点和偏移是否正确
Q/K 是否使用了同一套 rotary frequency 配置
head_dim 是否与 RoPE 实际作用的维度一致
cos/sin 的 shape、dtype、device 是否正确
Prefill 与 Decode 是否使用同样的 position 规则
```

很多所谓“Attention 精度问题”其实是 position id、cache offset 或 RoPE scaling 配置不一致。

### 第三层：需要改算子或做数值定位时再深入推导

只有在以下场景，才值得手推到公式或逐元素对照：

- 自己实现 RoPE kernel 或替换 rotary embedding；
- 对比不同实现的 `rotate_half`、交错维度与半维度布局；
- 排查 `float32`、`float16`、`bfloat16` 下 sin/cos 误差；
- 支持 YaRN、NTK scaling、linear scaling 等长上下文变体；
- 怀疑 Q/K 的旋转顺序、频率表或导出图发生错误。

因此本课程把 RoPE 作为“可解释的半黑盒”：平时使用时不要求重复推导，但要能说明它改变了什么、保持了什么、哪些输入必须一致；遇到异常时再展开公式和实现细节。

### 工程判断题

如果 Prefill 与 Decode 的 Q/K shape 都正确，但 Decode 从第一个新 token 开始 cosine 明显下降，优先检查什么？

**参考答案：** 优先检查 position id 与 KV Cache offset，其次检查 Prefill/Decode 是否使用了相同的 RoPE base、scaling、cos/sin dtype 和 head_dim 配置。shape 正确只能说明 Tensor 排列没有明显错误，不能证明每个 token 使用了正确的位置角度。

## 八、第三段：Prefill/Decode 的 position id 连续性

假设 prompt 有 4 个 token：

```text
Prefill position id = [0, 1, 2, 3]
```

Prefill 完成后，KV Cache 中已经保存了这 4 个位置的 K/V。下一次 Decode 生成的新 token 应使用：

```text
Decode position id = 4
```

再下一个 token 使用 5，以此类推。Decode 不能因为每次输入 shape 是 `[batch_size, 1]` 就把 position id 重新设为 0；那会让新 Query/Key 使用错误的旋转角度。

### 8.1 三种长度不要混淆

```text
prompt_length       ：Prefill 已处理的 token 数
generated_length    ：已经接受并写入 Cache 的新 token 数
current_position    ：本次新 token 的位置
```

通常：

```text
current_position = prompt_length + generated_length
```

如果 prompt 长度为 4，已经生成 2 个 token，下一次 Decode 的新 token 位置就是 6。当前 Q 会读取位置 0 到 6 的 K/V；但只有位置 6 的新 K/V 在本步追加写入 Cache。

### 8.2 RoPE Cache 与 KV Cache 是两件事

RoPE cos/sin cache 是按位置准备的旋转系数；KV Cache 保存旋转之后的 K/V 激活。二者不能混为一个 Cache：

- RoPE cache 不保存模型历史 token 的 K/V；
- KV Cache 不负责生成新的 cos/sin 角度；
- Decode 必须用当前 position id 从正确位置取 RoPE 系数，再把新 K/V 写到正确的 KV Cache slot。

### 8.3 常见错误表现

```text
position id 重置为 0     → Decode 首 token 就开始明显偏离
RoPE base/scaling 不同   → 短上下文可能接近，长上下文逐渐恶化
cos/sin dtype 不一致     → 误差通常较小，但长序列可能累积
KV slot 偏移错误         → 输出可能错乱，且不一定表现为单调 cosine 下降
```

排查时应同时 dump：当前 token 的 position id、cos/sin 选取范围、旋转前后的 Q/K、KV Cache 写入 slot，以及 target/reference 的对应 Tensor。

### 第三段检查题

某请求的 prompt 长度为 13，已经接受并写入 KV Cache 的生成 token 有 5 个。下一次 Decode：

1. 当前新 token 的 position id 是多少？
2. 它的 Q 需要关注位置 0 到哪个位置的 K/V？
