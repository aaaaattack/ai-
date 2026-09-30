
## 阶段 4 · 第 1 讲：EAGLE 为什么使用 Hidden State Draft

前面学习了：

- `llama.cpp`：独立 Draft Model 生成候选；
- Medusa：多个 decoding head 直接生成候选树；
- 当前 Qwen 工程：MTP、DFlash 生成候选并由 Target 验证。

EAGLE 的核心变化是：

> Draft 阶段不只预测 token，而是预测 Target model 的 hidden feature。

---

### 1. 传统 Draft Model 的问题

经典投机解码使用一个独立的小模型：

```text
context
   ↓
Draft Model
   ↓
draft tokens
   ↓
Target Model 验证
```

Draft Model 只能看到：

```text
token embedding
历史 token
自身 hidden state
```

但它无法直接使用 Target Model 的内部表示。

Target Model 在每个位置产生的 hidden state 通常包含更丰富的语义信息：

```text
h_t = TargetModel(x_1, x_2, ..., x_t)
```

如果 Draft Model 只使用 token，而看不到 `h_t`，就需要用一个较大的模型容量去近似 Target 的判断。

---

### 2. EAGLE 的核心思路

EAGLE 将生成过程改为：

```text
Target Model 计算 hidden state
        ↓
EAGLE Draft Network 读取 hidden feature
        ↓
预测下一位置的 hidden feature
        ↓
通过 lm_head 映射成 token
        ↓
构造候选树
        ↓
Target Model 验证
```

可以抽象为：

```text
Target:
h_t = F(x_≤t)

EAGLE Draft:
ĥ_{t+1} = G(h_t, embedding(x_t))

Token:
x̂_{t+1} = argmax(lm_head(ĥ_{t+1}))
```

其中：

- `F`：Target Model；
- `G`：EAGLE 的轻量 Draft Network；
- `h_t`：Target Model 的 hidden feature；
- `ĥ_{t+1}`：EAGLE 预测的 feature；
- `x̂_{t+1}`：Draft token。

---

### 3. Target hidden state 从哪里来？

EAGLE 的基础模型 forward 会返回 Target 的 hidden state。

源码位置：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\ea_model.py:172-196
```

核心代码：

```python
outputs = self.base_model.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    position_ids=position_ids,
)

hidden_states = outputs[0]
```

如果需要 Target 原始 logits：

```python
orig = self.base_model.lm_head(outputs[0])
```

因此 EAGLE 的第一步不是重新运行一个完整 Draft Model，而是直接复用 Target Model 已经计算出的 hidden state。

---

### 4. EAGLE Draft Network 的结构

EAGLE 的 Draft Network 位于：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:478-535
```

主要组件包括：

```python
self.embed_tokens = nn.Embedding(...)
self.lm_head = nn.Linear(...)
self.midlayer = LlamaDecoderLayeremb(config)
self.fc = nn.Linear(...)
self.norm = LlamaRMSNorm(...)
```

可以理解为：

```text
token embedding
        +
Target hidden feature
        ↓
feature projection
        ↓
轻量 Transformer layer
        ↓
预测 feature
        ↓
lm_head
        ↓
draft token
```

其中：

```python
self.fc = nn.Linear(
    config.target_hidden_size * 3,
    self.hidden_size,
    bias=False
)
```

表示某些 EAGLE 版本会将多个 Target hidden feature 拼接后，再投影到 Draft Network 的 hidden dimension。

---

### 5. 为什么不是直接预测 token？

如果直接预测 token：

```text
h_t → token_{t+1}
```

Draft Model 需要学习完整的词表分类边界。

EAGLE 采用：

```text
h_t → ĥ_{t+1} → token_{t+1}
```

这样 Draft Network 学习的是：

```text
Target hidden representation 的演化规律
```

而不是完全从 token 序列重新恢复 Target Model 的全部决策逻辑。

这种方式的优势是：

1. 可以使用 Target Model 已经提取的语义特征；
2. Draft Network 参数量较小；
3. 更容易预测后续 token 的表示；
4. 可以继续使用 Target 的 `lm_head` 进行 token 映射。

---

### 6. EAGLE 如何生成第一个候选树？

初始化流程位于：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:232-254
```

首先运行 Target Model：

```python
outputs, orig, hidden_states = model(
    input_ids,
    past_key_values=past_key_values,
    output_orig=True,
)
```

然后使用 Target 的最后一个输出预测第一个 token：

```python
token = torch.argmax(orig[:, -1])
token = token[None, None]
```

接着把这个 token 接到输入后面：

```python
input_ids = torch.cat(
    (input_ids, token.to(input_ids.device)),
    dim=1
)
```

最后把 Target hidden state 交给 EAGLE Draft Network：

```python
draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
    model.ea_layer.topK_genrate(
        hidden_states,
        input_ids,
        model.base_model.lm_head,
        logits_processor,
    )
)
```

这里的 `topK_genrate()` 会构造候选树。

---

### 7. EAGLE 的候选树生成

核心函数：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:670-825
```

主要流程：

```text
输入 Target hidden state
        ↓
EAGLE Draft Network 预测 feature
        ↓
lm_head 得到 token logits
        ↓
取 top-k token
        ↓
对每个 token 继续预测下一层 feature
        ↓
继续取 top-k
        ↓
形成候选树
```

代码中的第一层 top-k 预测：

```python
last_headout = self.lm_head(
    self.norm(last_hidden)
)

last_p = self.logsoftmax(last_headout)

top = torch.topk(
    last_p,
    top_k,
    dim=-1
)
```

这会得到：

```text
top-k token
top-k 分数
```

然后将每个候选继续送回 Draft Network：

```python
out_hidden, past_key_values = self(
    input_hidden,
    input_ids=input_ids,
    past_key_values=past_key_values,
    position_ids=position_ids,
    use_cache=True,
)
```

这说明 EAGLE 的 Draft 阶段仍然是自回归的，但自回归对象主要是：

```text
hidden feature
```

而不仅仅是 token。

---

### 8. EAGLE 与 Medusa 的区别

| 对比项 | Medusa | EAGLE |
|---|---|---|
| Draft 结构 | 多个并行 decoding heads | 轻量 autoregressive Draft Network |
| 主要输入 | Target hidden state | Target hidden state + token embedding |
| 预测对象 | token logits | hidden feature，再映射成 token |
| 候选生成 | 多个 head 直接生成 | Draft Network 逐层生成 |
| 是否有 Draft KV Cache | 通常较少 | 有 Draft Network 的 KV Cache |
| 候选树 | 有 | 有 |
| 目标 | 低成本多头预测 | 更准确地拟合 Target feature 演化 |

可以用一句话区分：

```text
Medusa：在 Target hidden state 上增加多个预测头。

EAGLE：用 Target hidden state 初始化一个小型自回归 Draft Network。
```

---

### 9. EAGLE 的 Target 验证

EAGLE 生成候选后，Target Model 通过 `tree_decoding()` 一次验证整棵树。

源码位置：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:306-331
```

核心代码：

```python
outputs, tree_logits, hidden_state = model(
    tree_candidates,
    output_orig=True,
    past_key_values=past_key_values,
    position_ids=position_ids,
)
```

然后根据 `retrieve_indices` 把树结构重新映射成候选路径：

```python
logits = tree_logits[0, retrieve_indices]
```

后续再调用：

```python
evaluate_posterior(
    logits,
    candidates,
    logits_processor
)
```

选择最佳候选路径。

整体流程是：

```text
Target hidden state
        ↓
EAGLE Draft Network
        ↓
candidate tree
        ↓
Target tree decoding
        ↓
posterior evaluation
        ↓
接受最长正确前缀
```

---

### 10. EAGLE 一轮完整流程

对应主循环：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\ea_model.py:199-295
```

简化后：

```python
# 1. 初始化候选树
draft_tokens, retrieve_indices, tree_mask, tree_position_ids, \
logits, hidden_state, sample_token = initialize_tree(...)

for _ in range(max_length):

    # 2. Target 一次计算整棵树
    logits, hidden_state_new, outputs = tree_decoding(...)

    # 3. 将树结构转换为候选路径
    candidates = draft_tokens[0, retrieve_indices]

    # 4. 验证并选择最佳候选
    best_candidate, accept_length, sample_p = evaluate_posterior(
        logits,
        candidates,
        logits_processor,
    )

    # 5. 提交接受路径，更新 Draft 状态
    input_ids, draft_tokens, retrieve_indices, tree_mask, \
    tree_position_ids, new_token, hidden_state, sample_token = \
        update_inference_inputs(...)
```

注意 EAGLE 的 `update_inference_inputs()` 不仅更新 token 和 Target KV Cache，还要更新：

```text
EAGLE Draft Network 的 hidden state
EAGLE Draft Network 的 KV Cache
下一轮候选树的结构
```

---

### 11. EAGLE 与当前 Qwen 工程的联系

当前 Qwen DFlash 的主要状态包括：

```text
Target KV Cache
Draft KV Cache
hidden state
conv state
recurrent state
context length
```

EAGLE 的核心状态更偏向：

```text
Target KV Cache
EAGLE Draft KV Cache
Target hidden feature
Draft hidden feature
tree mask
retrieve indices
```

两者共同点：

```text
候选生成
    ↓
Target 验证
    ↓
选择接受前缀
    ↓
提交接受状态
    ↓
丢弃或覆盖拒绝状态
```

主要区别：

```text
DFlash：
重点是 block-level draft 和状态回滚。

EAGLE：
重点是 hidden feature prediction 和 tree-based draft。
```

---

### 12. 本节核心结论

需要记住：

1. EAGLE 直接使用 Target Model 的 hidden feature；
2. EAGLE Draft Network 预测后续 hidden feature；
3. 预测出来的 feature 再通过 `lm_head` 转成 token；
4. EAGLE 仍然使用候选树和 Target 一次验证；
5. 与 Medusa 相比，EAGLE 有更明显的 Draft 自回归过程；
6. EAGLE 的优势是 Draft 不必从 token 序列重新学习全部语义表示。

下一部分是：

**EAGLE 的 hidden feature 如何与 token embedding 融合，以及 EAGLE-1、EAGLE-2、EAGLE-3 的演进。**