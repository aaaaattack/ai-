
## 阶段 4 · 第 4 讲：EAGLE 候选树生成与 Target Verification

本节把 EAGLE 的完整推理链路串起来：

```text
Hidden feature
    ↓
EAGLE Draft Network
    ↓
候选树
    ↓
Tree Attention
    ↓
Target 一次验证
    ↓
选择最佳路径
    ↓
更新 KV Cache 和 hidden state
```

---

### 1. 为什么需要候选树？

如果 EAGLE 只生成一条 Draft 链：

```text
A → B → C → D
```

一旦中间某个 token 错误，后面的 token 也可能全部失效。

候选树可以同时保留多条可能路径：

```text
             root
          ┌───┼───┐
          A   B   C
        ┌─┼─┐ │   │
       A1 A2 A3 B1 C1
```

如果 Target 不接受：

```text
root → A → A1
```

仍可能接受：

```text
root → B → B1
```

因此候选树提高了 Draft 覆盖正确路径的概率。

---

### 2. EAGLE 如何生成候选树？

核心函数：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:670-825
```

函数名称：

```python
topK_genrate(...)
```

虽然函数名中有拼写错误，但它负责：

```text
生成 top-k 候选
维护父节点
累计路径分数
构造树结构
生成 tree mask
生成 position ids
```

---

### 3. 第一层候选

EAGLE 首先使用当前 hidden feature 得到 logits：

```python
last_headout = self.lm_head(
    self.norm(last_hidden)
)

last_p = self.logsoftmax(last_headout)
```

然后选择 top-k token：

```python
top = torch.topk(
    last_p,
    top_k,
    dim=-1
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:704-713
```

例如：

```text
top-k = 3

A: 0.60
B: 0.25
C: 0.10
```

此时第一层候选树为：

```text
root
├── A
├── B
└── C
```

---

### 4. 继续扩展每个候选

EAGLE 不会只扩展最高分的 `A`，而是继续对多个候选进行扩展。

下一轮输入包括：

```python
input_hidden
input_ids
past_key_values
```

源码：

```python
out_hidden, past_key_values = self(
    input_hidden,
    input_ids=input_ids,
    past_key_values=past_key_values,
    position_ids=position_ids,
    use_cache=True,
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:720-727
```

于是每个分支都可以继续生成：

```text
root
├── A
│   ├── A1
│   ├── A2
│   └── A3
├── B
│   ├── B1
│   ├── B2
│   └── B3
└── C
    ├── C1
    ├── C2
    └── C3
```

---

### 5. 父节点与路径分数

EAGLE 使用：

```python
parents_list = []
scores_list = []
```

记录：

```text
每个节点的父节点
每条路径的累计分数
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:676-680
```

当前 token 分数与父节点分数相加：

```python
cu_scores = topk_p + scores[:, None]
```

源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:735
```

例如：

```text
score(A) = -0.5
score(A1) = -0.2
```

则：

```text
score(A → A1) = -0.7
```

EAGLE 通过累计分数衡量一条路径的整体可信度。

---

### 6. 为什么需要父节点索引？

候选树展开后通常不是嵌套结构，而是扁平数组：

```text
[root, A, B, C, A1, A2, B1, C1]
```

为了知道：

```text
A1 的父节点是 A
B1 的父节点是 B
```

需要保存父节点信息。

源码：

```python
parents = (
    topk_cs_index
    + bias
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:728-734
```

父节点信息之后会用于构造：

```text
tree_mask
tree_indices
retrieve_indices
```

---

### 7. 候选树的扁平化

最终，EAGLE 会从所有候选节点中选择分数最高的 `total_tokens` 个：

```python
top_scores = torch.topk(
    scores_list,
    total_tokens,
    dim=-1
)
```

然后获得最终 Draft token：

```python
draft_tokens = ss_token_list[top_scores_index]
```

源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:776-789
```

最终结构可能被表示为：

```python
draft_tokens = [
    root,
    A,
    B,
    A1,
    A2,
    B1,
]
```

虽然它看起来是一维数组，但父节点关系仍由额外索引保存。

---

### 8. Tree Mask 的作用

普通 causal attention 是：

```text
当前位置只能看自己和之前的位置
```

候选树不是普通线性序列。

例如：

```text
root
├── A
│   └── A1
└── B
    └── B1
```

正确的可见关系是：

```text
A  可以看 root
A1 可以看 root、A
B  可以看 root
B1 可以看 root、B
```

但：

```text
A  不应该看 B
A1 不应该看 B、B1
B  不应该看 A
```

因此需要 Tree Attention Mask。

EAGLE 构造 mask 的代码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:800-815
```

核心逻辑：

```python
tree_mask = torch.eye(
    total_tokens + 1
).bool()

tree_mask[:, 0] = True
```

然后根据每个节点的父节点补充可见位置：

```python
tree_mask[i + 1].add_(
    tree_mask[mask_index_list[i]]
)
```

这表示：

```text
当前节点可以看到父节点可以看到的所有节点
```

因此祖先关系会逐层传递。

---

### 9. Tree Mask 示例

假设节点顺序为：

```text
0: root
1: A
2: B
3: A1
4: B1
```

树结构：

```text
root
├── A
│   └── A1
└── B
    └── B1
```

可见关系可以表示为：

```text
root: root
A:    root, A
B:    root, B
A1:   root, A, A1
B1:   root, B, B1
```

矩阵中：

```text
A1 行不能看到 B
B1 行不能看到 A
```

这保证 Target Model 对每条候选路径分别进行因果计算。

---

### 10. Tree Position IDs

树中的不同节点虽然位于同一个扁平数组中，但它们的真实位置取决于深度。

例如：

```text
root: position 0
A:    position 1
B:    position 1
A1:   position 2
B1:   position 2
```

EAGLE 通过：

```python
tree_position_ids = torch.sum(
    tree_mask,
    dim=1
) - 1
```

计算位置：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:818-820
```

在 Target 验证阶段，再加上原始输入长度：

```python
position_ids = (
    tree_position_ids
    + input_ids.shape[1]
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:314
```

如果原始上下文长度是：

```text
input_ids.shape[1] = 100
```

那么：

```text
root: 100
A/B:  101
A1/B1: 102
```

---

### 11. Target 如何一次验证整棵树？

EAGLE 的 Target 验证函数：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:306-331
```

核心调用：

```python
outputs, tree_logits, hidden_state = model(
    tree_candidates,
    output_orig=True,
    past_key_values=past_key_values,
    position_ids=position_ids,
)
```

这里传入：

```text
tree_candidates
tree_position_ids
tree attention mask
Target past_key_values
```

Target 并不是分别执行：

```text
root → A → A1
root → B → B1
```

而是将整个候选树一次性输入，并使用 Tree Attention Mask 限制路径之间的信息流动。

---

### 12. 为什么一次 Forward 可以验证多条路径？

因为不同分支之间不需要互相通信，只需要共享祖先：

```text
A1 共享 root、A
B1 共享 root、B
```

Tree Attention 可以实现：

```text
共享公共前缀
隔离不同分支
```

因此 Target 可以在一次 forward 中同时得到：

```text
root 的 logits
A 的 logits
B 的 logits
A1 的 logits
B1 的 logits
```

计算结构从：

```text
多次串行 forward
```

变成：

```text
一次带 Tree Mask 的 forward
```

---

### 13. `retrieve_indices` 的作用

Target 输出仍然按照树的扁平顺序排列：

```text
tree_logits:
[root, A, B, A1, B1]
```

但后续验证需要按候选路径排列：

```text
candidate 0: [root, A, A1]
candidate 1: [root, B, B1]
```

因此需要：

```python
logits = tree_logits[
    0,
    retrieve_indices
]
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:330
```

`retrieve_indices` 完成：

```text
Tree Layout
    ↓
Candidate Path Layout
```

它与 Medusa 中的作用类似。

---

### 14. Candidate 构造

EAGLE 会先得到扁平树 token：

```python
tree_candidates = candidates[tree_indices]
```

然后使用 `retrieve_indices` 生成候选路径：

```python
cart_candidates = tree_candidates_ext[
    retrieve_indices
]
```

源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:284-303
```

最后得到类似：

```text
[
    [root, A, A1],
    [root, B, B1],
]
```

然后交给：

```python
evaluate_posterior(...)
```

进行验证。

---

### 15. EAGLE 的完整一轮

主循环位于：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\ea_model.py:246-295
```

可以简化为：

```python
# 1. EAGLE Draft 生成候选树
draft_tokens = ...

# 2. Target 使用 Tree Attention 验证
logits, hidden_state_new, outputs = tree_decoding(
    self,
    draft_tokens,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
)

# 3. 转换成候选路径
candidates = draft_tokens[0, retrieve_indices]

# 4. 选择最佳路径
best_candidate, accept_length, sample_p = evaluate_posterior(
    logits,
    candidates,
    logits_processor,
)

# 5. 提交接受状态
input_ids, draft_tokens, ... = update_inference_inputs(...)
```

整个过程是：

```text
Draft tree generation
        ↓
Tree layout encoding
        ↓
Target tree decoding
        ↓
Candidate path reconstruction
        ↓
Posterior evaluation
        ↓
Cache/state update
```

---

### 16. EAGLE 与 Medusa Tree Attention 的区别

两者都使用候选树，但候选树来源不同：

```text
Medusa：
多个 decoding head 直接生成候选 token

EAGLE：
Draft Network 自回归生成 hidden feature 和 token
```

两者共同使用：

```text
tree_mask
tree_position_ids
tree_indices
retrieve_indices
```

因此可以把 Medusa 和 EAGLE 的 Target 阶段看成相似结构：

```text
候选树
    ↓
Tree Attention
    ↓
一次 Target Forward
    ↓
路径验证
```

差异主要发生在 Draft 阶段。

---

### 17. 本节核心结论

需要记住：

1. EAGLE 将候选树扁平化存储；
2. `parents_list` 保存父节点关系；
3. `tree_mask` 保证不同分支互不泄漏；
4. `tree_position_ids` 表示树中的真实深度；
5. `tree_indices` 将候选 token 映射到树节点；
6. `retrieve_indices` 将树输出恢复为候选路径；
7. Target 通过一次 Tree Attention Forward 验证多条路径；
8. 最终由 `evaluate_posterior()` 选择最佳路径。

至此，EAGLE 阶段完成：

```text
Hidden Feature
→ Feature Fusion
→ EAGLE Evolution
→ Candidate Tree
→ Tree Verification
```

下一阶段按照路线进入：

**阶段 5：MTP（Multi-Token Prediction）**。