
## 阶段 6 · 第 2 讲：DFlash 的 Block-level Draft

本节重点学习 DFlash 如何一次生成整个 Draft block，并对应当前工程中的：

```text
candidate_ids
first_scores
transition_scores
noise token padding
anchor
block attention mask
```

DFlash 原论文的核心是：使用轻量 Block Diffusion Draft，在一次 forward 中并行预测一段候选 token，同时将 Target Model 的上下文 hidden feature 注入 Draft Model，提高候选质量。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

---

## 1. 从逐 Token Draft 到 Block Draft

传统 MTP 的 Draft 是：

```text
h_t
 ↓
d1
 ↓
d2
 ↓
d3
 ↓
d4
```

每一步都依赖上一步。

DFlash 改成：

```text
h_t + anchor
        ↓
┌───────────────┐
│ d1 d2 d3 d4   │
└───────────────┘
```

四个 Draft 位置同时计算。

从计算图看：

```text
MTP：

h_t → d1 → h_{t+1} → d2 → h_{t+2} → d3

DFlash：

h_t
 ├── d1
 ├── d2
 ├── d3
 └── d4
```

DFlash 不是让 Draft Model 逐个生成 token，而是让它处理一个固定长度的 token block。

---

## 2. Block 的基本组成

假设：

```text
block_size = 5
```

一个 DFlash block 可以表示为：

```text
[a, MASK, MASK, MASK, MASK]
```

其中：

```text
a：
anchor token

MASK：
待预测位置
```

Draft Model 输出：

```text
[a, d1, d2, d3, d4]
```

如果 anchor 只是已知上下文，则真正的 speculative token 是：

```text
[d1, d2, d3, d4]
```

所以：

```text
物理 block 长度 = 5
逻辑 Draft token 数量 = 4
```

当前 `speculators` 配置明确区分了这两个概念：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\config.py:44-48
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\config.py:73-80
```

---

## 3. Anchor 的来源

DFlash 的 anchor 通常来自已经被 Target 确认的 token。

假设上一轮结果是：

```text
[prefix, x1, x2, x3]
```

那么下一轮可以使用：

```text
anchor = x3
```

然后构造：

```text
[x3, MASK, MASK, MASK, MASK]
```

DFlash Draft 预测：

```text
[d1, d2, d3, d4]
```

Target 再验证：

```text
[prefix, x3, d1, d2, d3, d4]
```

因此 anchor 的本质是：

```text
连接上一轮正式上下文与下一轮 Draft block 的边界 token
```

它不是普通的随机输入，而是：

```text
已经经过 Target 确认的真实 token
```

---

## 4. 为什么不能全部使用 MASK？

如果输入是：

```text
[MASK, MASK, MASK, MASK, MASK]
```

Draft Model 不知道当前 block 从哪个真实上下文开始。

它只能根据 Target hidden feature 猜测：

```text
这个 block 应该接在什么位置？
```

而 anchor 提供了确定边界：

```text
当前已知 token 是 a
请预测 a 后面的内容
```

数学上，DFlash 需要建模的是：

```text
p(d_1, d_2, ..., d_k | prefix, a)
```

而不是：

```text
p(d_1, d_2, ..., d_k)
```

这使得 Draft block 与真实自回归生成过程保持一致。

---

## 5. Noise Token Padding 是什么？

在实际工程中，通常不会直接把 Python 的 `None` 或空值送进模型，而是使用一个固定的占位 token：

```text
noise_token_id
```

例如：

```text
[a, noise, noise, noise, noise]
```

这里的 `noise` 不是最终输出 token，它只是：

```text
固定形状占位符
```

它的作用是：

1. 保持 block shape 固定；
2. 让 embedding 层可以正常查表；
3. 让 Draft Transformer 能够一次接收整个 block；
4. 通过 Attention Mask 控制哪些位置可以互相通信。

当前 `speculators` 实现对应代码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:406-416
```

核心逻辑可以概括为：

```python
mask_token_ids = torch.full(
    (1, num_anchors * block_size),
    mask_token_id,
)

mask_token_ids[:, ::block_size] = (
    input_ids[:, anchor_positions]
)

noise_embedding = embed_tokens(
    mask_token_ids
)
```

含义是：

```text
先把所有 block 位置填成 mask/noise token，
再把每个 block 的第一个位置替换成 anchor。
```

最终：

```text
[a1, noise, noise, noise, noise,
 a2, noise, noise, noise, noise,
 ...]
```

---

## 6. 为什么要固定 Block Shape？

固定 block shape 对 GPU Runtime 很重要。

假设每轮 Draft 长度都变化：

```text
第 1 轮：5
第 2 轮：7
第 3 轮：3
第 4 轮：8
```

会导致：

```text
动态 Tensor Shape
动态 Attention Mask
动态 Kernel 调度
CUDA Graph 难以复用
```

DFlash 更倾向于：

```text
物理 block size 固定为 K
逻辑接受长度动态变化
```

例如：

```text
block_size = 5
```

每轮都计算：

```text
[a, noise, noise, noise, noise]
```

但 Target 验证后可能只接受：

```text
d1
```

或者：

```text
d1, d2, d3
```

因此区分：

```text
物理计算宽度：
block_size = 5

逻辑有效长度：
accept_length ∈ [0, 4]
```

---

## 7. Draft Block 的张量形状

假设：

```text
batch_size = B
block_size = K
hidden_size = H
vocab_size = V
```

输入 embedding 形状：

```text
[B, K, H]
```

Draft Model 输出 hidden：

```text
[B, K, H]
```

经过 lm_head 后得到 logits：

```text
[B, K, V]
```

即：

```text
noise_embedding
    [B, K, H]
        ↓
DFlash Transformer
        ↓
draft_hidden
    [B, K, H]
        ↓
lm_head
        ↓
draft_logits
    [B, K, V]
```

与 MTP 不同的是：

```text
MTP：
通常逐步取得 [B, 1, V]

DFlash：
一次得到 [B, K, V]
```

这就是 block-level draft 的基本计算形式。

---

## 8. Target Hidden Feature 如何注入？

DFlash 不仅输入 noise embedding，还输入 Target hidden feature。

当前实现中，Target 的多个辅助层 hidden state 会先经过投影：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:116-124
```

逻辑可以写成：

```python
target_feature = concat(
    target_hidden_layer_1,
    target_hidden_layer_2,
    target_hidden_layer_3,
)

target_feature = fc(
    target_feature
)

target_feature = hidden_norm(
    target_feature
)
```

得到：

```text
target_feature.shape = [B, context_len, H]
```

然后这个 feature 会提供给 DFlash Draft layers。

---

## 9. Target Feature 为什么比单纯 Token 输入更有效？

一个小 Draft Model 如果只看到：

```text
[prefix token ids]
```

就需要自己完成：

```text
语义理解
句法分析
任务识别
未来 token 预测
```

而 Target Model 已经完成了前面的复杂计算。

DFlash 直接复用：

```text
Target hidden feature
```

因此 Draft Model 主要负责：

```text
把 Target 的上下文表示转化为未来 block 的 token 预测
```

这相当于：

```text
Target：
负责理解上下文

DFlash：
负责快速展开未来 block
```

论文的关键观点正是：Target 的 hidden feature 中已经包含未来 token 的信息，DFlash 使用这些 feature 作为条件来获得高质量并行 Draft。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

---

## 10. Hidden Feature 如何进入每个 Draft Layer？

DFlash 并不是只在第一层注入 Target feature。

在当前实现中，每个 Draft layer 都会接收到：

```text
当前 block hidden
Target hidden feature
Attention Mask
Position Embedding
```

对应代码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:450-466
```

逻辑可以抽象为：

```python
for layer in draft_layers:
    noise_embedding = layer(
        hidden_states=noise_embedding,
        target_hidden=target_feature,
        attention_mask=block_mask,
        position_ids=position_ids,
    )
```

也就是说：

```text
每一层 Draft Transformer
都可以访问 Target Context Feature。
```

这样比只在输入层拼接一次更充分。

---

## 11. Hidden Feature 注入到 K/V

在 Attention 内部，Target hidden feature 通常用于生成上下文 K/V。

当前代码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\model_definitions.py:117-124
```

逻辑类似：

```python
k_ctx = project_context_key(
    target_hidden
)

v_ctx = project_context_value(
    target_hidden
)

k_noise = project_noise_key(
    draft_hidden
)

v_noise = project_noise_value(
    draft_hidden
)

k = concat(
    k_ctx,
    k_noise,
)

v = concat(
    v_ctx,
    v_noise,
)
```

抽象结构：

```text
Target Feature
    ├── K_context
    └── V_context

Draft Block Hidden
    ├── K_block
    └── V_block
```

然后 Attention 同时访问：

```text
Target context KV
Draft block KV
```

这就是 DFlash 与普通小 Draft Model 的重要区别：

```text
普通 Draft：
主要依赖自己的历史状态

DFlash：
Draft block 直接读取 Target 提供的上下文 KV
```

---

## 12. Block Attention Mask

DFlash 的 block Attention 需要满足三个条件：

### 条件一：可以看历史上下文

每个 Draft block 位置可以看 anchor 之前的正式上下文：

```text
prefix before anchor
```

### 条件二：可以看自己的 block

同一个 block 内的 Draft 位置可以互相交互：

```text
d1 ↔ d2 ↔ d3 ↔ d4
```

这使得 block 内预测可以并行协作。

### 条件三：不能看其他 block

如果同时训练多个 anchor：

```text
anchor_1 → block_1
anchor_2 → block_2
```

则：

```text
block_1 不能看 block_2
block_2 不能看 block_1
```

当前 mask 定义：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\attention.py:7-100
```

核心逻辑是：

```text
每个 query block 只属于一个 anchor，
只能看对应 anchor 之前的 base token，
以及自己所属的 synthetic block。
```

---

## 13. 一个具体的 Attention 例子

假设：

```text
prefix = [p1, p2, p3]
anchor = a
block = [d1, d2, d3]
```

DFlash 的输入布局：

```text
[p1, p2, p3, a, d1, d2, d3]
```

允许的可见关系：

```text
d1 可以看：
p1, p2, p3, a, d1, d2, d3

d2 可以看：
p1, p2, p3, a, d1, d2, d3

d3 可以看：
p1, p2, p3, a, d1, d2, d3
```

在 block 内部是非因果的，因此：

```text
d1 可以看到 d2、d3 的 block-level 表示
```

但这不代表最终生成顺序被改变，因为：

```text
Target Verify 仍然决定最终合法 token 序列。
```

---

## 14. `candidate_ids` 的作用

在当前 Qwen DFlash 工程中，可以把：

```text
candidate_ids
```

理解为：

```text
本轮 Draft block 中真正拿来进行 Target Verify 的 token IDs。
```

假设 block 为：

```text
[a, d1, d2, d3, d4]
```

如果 anchor 不作为 speculative token，则：

```text
candidate_ids = [d1, d2, d3, d4]
```

如果 anchor 也参与采样，则可能是：

```text
candidate_ids = [a, d1, d2, d3, d4]
```

所以看到 `candidate_ids` 时，需要先确认：

```text
anchor 是否包含在 candidate_ids 中？
```

不能只根据 tensor 长度猜测语义。

---

## 15. `first_scores` 的作用

`first_scores` 可以理解为：

```text
Draft block 起始预测位置的候选分数。
```

它通常来自：

```text
anchor 对应位置的 logits
```

如果 Draft Model 需要选择第一个预测 token：

```python
first_logits = draft_logits[:, 0, :]
first_scores = softmax(first_logits)
```

然后根据：

```text
greedy
sampling
top-k
top-p
```

得到第一个候选 token。

如果 anchor 只作为上下文，则：

```text
first_scores
```

对应的是 block 第一个可预测位置。

如果 anchor 也参与采样，则：

```text
first_scores
```

可能直接对应 anchor position 的 sampling distribution。

因此：

```text
first_scores 的语义取决于 sample_from_anchor 配置。
```

---

## 16. `transition_scores` 的作用

`transition_scores` 表示 block 中后续位置的 Draft 分数或条件转移分数。

例如：

```text
transition_scores[:, 0]：
d1 的分数

transition_scores[:, 1]：
d2 的分数

transition_scores[:, 2]：
d3 的分数
```

逻辑上可以写为：

```python
transition_scores = gather(
    draft_logits,
    candidate_ids
)
```

它们可用于：

```text
选择 Draft token
估计 Draft 置信度
计算 rejection sampling 所需的 q 分布
统计每个位置的 acceptance
```

需要区分：

```text
Draft score：
DFlash 对候选的判断

Target score：
Target 对同一 token 的判断
```

验证阶段最终比较的是：

```text
p_target
```

而不是只看：

```text
transition_scores
```

---

## 17. `first_scores` 与 `transition_scores` 的关系

可以将它们理解为：

```text
first_scores：
block 起点的 token distribution

transition_scores：
block 内各后续位置的 token distribution
```

一个 block：

```text
[a, d1, d2, d3]
```

可能对应：

```text
first_scores：
P(d1 | prefix, a)

transition_scores：
P(d2 | block context)
P(d3 | block context)
```

DFlash 的特殊点是：

```text
这些位置可以在一次 block forward 中同时计算，
并不要求 d2 的计算必须等待 d1 完全生成。
```

---

## 18. Noise Padding 与 Candidate IDs 的区别

这两个概念容易混淆。

### Noise Padding

```text
用于输入 Draft Model 的占位 token
```

例如：

```text
[a, noise, noise, noise]
```

它是模型输入。

### Candidate IDs

```text
从 Draft logits 中采样得到的候选 token
```

例如：

```text
[d1, d2, d3]
```

它是 Target Verify 的输入。

过程是：

```text
noise padding
    ↓
DFlash Draft forward
    ↓
draft logits
    ↓
sampling/argmax
    ↓
candidate_ids
    ↓
Target Verify
```

所以：

```text
noise token 不是最终生成 token
candidate_ids 才是需要验证的 token
```

---

## 19. DFlash 一次 Draft 的伪代码

下面是逻辑化简版本：

```python
# 1. 构造固定大小的 Draft 输入
draft_input = full(
    [anchor, noise, noise, noise, noise]
)

# 2. 获得 Target auxiliary hidden states
target_feature = collect_target_hidden_states(
    target_model
)

# 3. 投影 Target feature
target_feature = project(
    target_feature
)

# 4. Block-level Draft forward
draft_hidden = dflash_model(
    draft_input,
    target_feature=target_feature,
    attention_mask=block_mask,
)

# 5. 得到每个位置的 Draft logits
draft_logits = lm_head(
    draft_hidden
)

# 6. 从 block positions 采样候选
candidate_ids = sample(
    draft_logits
)

# 7. 提取分数
first_scores = get_first_scores(
    draft_logits
)

transition_scores = get_transition_scores(
    draft_logits,
    candidate_ids
)
```

对应源码入口：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:436-455
```

统一实现的 Draft block 逻辑：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:387-486
```

---

## 20. DFlash 与 EAGLE 的计算方式对比

### EAGLE

```python
for step in range(k):
    hidden = eagle_draft(
        hidden,
        previous_token,
    )
    token = sample(hidden)
```

Draft 计算次数：

```text
与 k 近似相关
```

### DFlash

```python
hidden = dflash_draft(
    block_input,
    target_feature,
    block_mask,
)

tokens = sample(
    hidden
)
```

Draft forward 次数：

```text
每个 block 主要一次
```

因此 DFlash 的目标是让：

```text
Draft latency ≈ 与 block size 弱相关
```

但实际耗时仍会受到：

```text
block size
draft layers
attention kernel
memory bandwidth
```

影响。

---

## 21. 为什么 DFlash 不能简单把整个 block 当成普通 batch？

因为 batch 中的样本彼此独立：

```text
sample 1 不看 sample 2
```

但 DFlash block 需要：

```text
共享 prefix
共享 anchor
block 内允许特定交互
不同 anchor block 互相隔离
```

所以它不是普通的：

```text
[B, K]
```

batch，而是带有结构化 Attention 的：

```text
base context + synthetic blocks
```

必须额外构造：

```text
anchor_positions
block_indices
position_ids
attention_mask
```

当前统一实现的 block mask 入口：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:320-375
```

---

## 22. 多 Anchor 训练布局

训练时可以在一个序列中选择多个 anchor：

```text
序列：
x0 x1 x2 x3 x4 x5 x6 x7 x8 x9
```

随机选择：

```text
anchor_1 = x2
anchor_2 = x5
```

构造：

```text
block_1 = [x2, MASK, MASK, MASK]
block_2 = [x5, MASK, MASK, MASK]
```

然后拼接成：

```text
[原始序列, block_1, block_2]
```

Draft Model 一次处理多个 synthetic block。

源码中：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\utils.py:21-67
```

负责选择 anchor：

```python
select_anchors(...)
```

而：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\utils.py:6-18
```

负责根据 anchor 位置生成 block 的基础索引。

---

## 23. 为什么训练时需要多个 Anchor？

多个 anchor 可以增加训练样本密度。

如果每条序列只取一个 anchor：

```text
一条序列只产生一个 block 样本
```

如果取多个 anchor：

```text
一条序列产生多个 block 样本
```

同时，模型能够学习：

```text
不同上下文位置的 block continuation
```

但多个 block 之间必须通过 mask 隔离，否则会出现：

```text
未来 block 信息泄漏
```

因此训练过程必须同时完成：

```text
anchor 采样
block 构造
mask 构造
target 对齐
loss mask 构造
```

---

## 24. Loss Mask 与 Anchor

当前实现中：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:468-486
```

会构造：

```python
aligned_loss_mask
```

如果：

```python
sample_from_anchor = False
```

则 block 第一个位置是 anchor，不应该作为预测目标：

```python
aligned_loss_mask[:, ::block_size] = 0
```

即：

```text
anchor：
不计算 Draft loss

mask positions：
计算 Draft loss
```

这与推理阶段的语义一致：

```text
anchor 是已知 token
真正需要预测的是后续 masked positions
```

---

## 25. 与当前 Qwen DFlash 的对应关系

当前工程中的 Draft 过程可以按下面的逻辑理解：

```text
1. 从正式上下文中取 anchor
2. 构造固定长度 block
3. 使用 noise token padding
4. 读取 Target auxiliary hidden states
5. 运行 DFlash Draft Module
6. 得到 block logits
7. 生成 candidate_ids
8. 保存 first_scores / transition_scores
9. 交给 Target Verify
```

对应源码入口：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_engine.py:176-191
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_process.py:204-226
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:436-455
```

---

## 26. 本节最重要的理解

DFlash 的核心不是：

```text
把 MTP Head 数量增加
```

而是改变 Draft 计算依赖：

```text
MTP：
后一个 token 依赖前一个 token

DFlash：
一整个 token block 共享 anchor 和 Target context，
block 内部并行预测
```

对比：

```text
MTP：

d1 → d2 → d3 → d4


DFlash：

┌──── d1 ────┐
│            │
anchor → d2  │
│            │
└──── d3 ────┘
       d4
```

---

## 27. 本节核心结论

需要记住：

1. DFlash 的 Draft 单位是 block，而不是单个 token；
2. 每个 block 通常由一个 anchor 和多个 mask/noise position 组成；
3. noise token 只是输入占位符，不是最终输出；
4. Target hidden feature 为 Draft 提供高质量上下文；
5. DFlash Draft Model 一次输出 `[B, block_size, vocab_size]` 的 logits；
6. `candidate_ids` 是从 logits 中得到的待验证 token；
7. `first_scores` 描述 block 起始位置的 Draft 分数；
8. `transition_scores` 描述后续位置的 Draft 分数；
9. Attention Mask 保证不同 block 之间不发生信息泄漏；
10. `block_size` 是固定物理宽度，`accept_length` 是动态逻辑长度；
11. DFlash 的并行 Draft 不改变 Target Verify 的正确性保证。

下一部分将继续分析：

**DFlash 的 Verify、accepted prefix、fallback，以及 Target KV Cache 如何回滚和提交。**