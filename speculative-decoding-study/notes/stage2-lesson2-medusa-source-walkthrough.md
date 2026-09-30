# 阶段 2 · 第 2 讲：Medusa 源码主链路

源码目录：

```
D:\speculative-decoding-study\repos\Medusa\medusa\model
```

本节按真实调用顺序阅读：

```
medusa_model_new.py
    ↓
utils.py
    ↓
candidate tree
    ↓
tree attention
    ↓
posterior evaluation
    ↓
KV cache update
```

------

## 1. 创建 Medusa Heads

文件：

```
medusa_model_new.py
```

核心结构：

```
self.medusa_head = nn.ModuleList(...)
```

推理时：

```
for i in range(self.medusa):
    medusa_logits.append(
        self.medusa_head[i](hidden_states)
    )
```

得到的结果可以理解为：

```
medusa_logits[0] → 第一个未来位置
medusa_logits[1] → 第二个未来位置
medusa_logits[2] → 第三个未来位置
...
```

这些 Head 共享同一个 Target Backbone 的 hidden state。

------

## 2. 生成 Tree Attention 所需的 Buffer

文件：

```
utils.py
```

函数：

```
generate_medusa_buffers(medusa_choices)
```

它主要生成四类数据：

```
{
    "medusa_attn_mask": ...,
    "tree_indices": ...,
    "medusa_position_ids": ...,
    "retrieve_indices": ...,
}
```

### `medusa_attn_mask`

决定每个树节点可以看到哪些祖先。

例如：

```
root
 ├── A
 │   └── A1
 └── B
     └── B1
```

注意力关系：

```
A  可以看 root
B  可以看 root
A1 可以看 root、A
B1 可以看 root、B
```

### `tree_indices`

把树结构映射成扁平 Tensor 索引。

### `medusa_position_ids`

为每个节点设置树深度对应的位置编号。

```
root -> position 0
A/B  -> position 1
A1/B1 -> position 2
```

### `retrieve_indices`

记录如何从扁平化结果中恢复每一条完整路径。

------

## 3. 生成候选树

函数：

```
generate_candidates(...)
```

第一步，生成普通 Target 候选：

```
candidates_logit = torch.argmax(logits[:, -1])
```

如果是采样模式，则使用 typical 或 nucleus sampling。

第二步，从每个 Medusa Head 中取 Top-K：

```
candidates_medusa_logits = torch.topk(
    medusa_logits[:, 0, -1],
    TOPK,
    dim=-1,
).indices
```

第三步，把候选映射到树结构：

```
tree_candidates = candidates[tree_indices]
```

最终得到的不是单条序列，而是类似：

```
root
 ├── A
 │   ├── A1
 │   └── A2
 └── B
     ├── B1
     └── B2
```

------

## 4. 为什么要同时有 tree candidates 和 cartesian candidates？

源码中会返回：

```
return cart_candidates, tree_candidates
```

二者作用不同。

### `tree_candidates`

用于真正送入 Target 模型执行 Tree Attention。

结构类似：

```
[root, A, B, A1, A2, B1, B2]
```

### `cart_candidates`

用于后续恢复完整路径：

```
[root, A, A1]
[root, A, A2]
[root, B, B1]
[root, B, B2]
```

可以理解为：

```
tree_candidates：适合模型计算
cart_candidates：适合候选路径比较
```

------

## 5. Tree Decode

函数：

```
tree_decoding(...)
```

核心调用：

```
tree_medusa_logits, outputs, tree_logits = model(
    tree_candidates,
    output_orig=True,
    past_key_values=past_key_values,
    position_ids=position_ids,
    medusa_forward=True,
)
```

这里发生了一次 Target forward。

输入是整棵候选树，但通过：

```
medusa_attn_mask
```

保证每个节点只关注自己的祖先。

然后使用：

```
retrieve_indices
```

重新排列 logits：

```
logits = tree_logits[0, retrieve_indices]
medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]
```

这样就能把树状计算结果重新组织成多条候选路径。

------

## 6. Greedy 模式下如何选择路径

函数：

```
evaluate_posterior(...)
```

当：

```
temperature == 0
```

时，使用 greedy 逻辑：

```
posterior_mask = (
    candidates[:, 1:]
    == torch.argmax(logits[:, :-1], dim=-1)
).int()
```

这一步是在逐位置比较：

```
候选 token == Target argmax
```

然后：

```
candidates_accept_length = (
    torch.cumprod(posterior_mask, dim=1)
).sum(dim=1)
```

这里的 `cumprod` 非常关键。

例如：

```
posterior_mask = [1, 1, 0, 1]
```

经过累乘：

```
cumprod = [1, 1, 0, 0]
```

最终接受长度：

```
accept_length = 2
```

即：

```
只能接受最长连续正确前缀
```

这和经典 speculative decoding 的规则完全一致。

------

## 7. Medusa 的路径选择

如果有多条候选：

```
路径 1：[A, A1, A2]
路径 2：[A, A1, B2]
路径 3：[B, B1, B2]
```

每条路径都会得到一个接受长度：

```
路径 1：2
路径 2：1
路径 3：3
```

源码：

```
accept_length = candidates_accept_length.max()
best_candidate = torch.argmax(
    candidates_accept_length
)
```

最终选择：

```
路径 3
```

因为它拥有最长的 Target 一致前缀。

------

## 8. 与 DFlash 接受逻辑的对比

### DFlash

```
for index, token in enumerate(draft):
    if target_argmax[index] != token:
        break
    accepted += 1
```

DFlash 只有一条 Draft 路径。

### Medusa

```
posterior_mask = candidates == target_argmax
candidates_accept_length = cumprod(mask).sum(...)
best_candidate = argmax(accept_length)
```

Medusa 同时评估多条候选路径，然后选接受长度最长的一条。

本质区别：

```
DFlash：
    单路径 + block verify

Medusa：
    多路径 + tree verify
```

------

## 9. 更新输入和 KV Cache

函数：

```
update_inference_inputs(...)
```

它负责：

- 把最佳候选路径追加到 `input_ids`
- 更新 logits
- 更新 Medusa logits
- 更新 KV cache
- 更新当前序列长度
- 准备下一轮推理

对应 DFlash 中的：

```
commit_draft_context(...)
advance(...)
```

Medusa 也面临同一个核心问题：

```
模型暂时验证了很多候选
但最终只能提交一条路径的 accepted prefix
```

因此候选树的临时状态必须被正确映射回正式 KV cache。

------

# 本节最重要的源码结论

Medusa 的完整链路是：

```
Medusa Heads
    ↓
generate_candidates()
    ↓
tree_candidates
    ↓
tree_decoding()
    ↓
Target 一次验证
    ↓
evaluate_posterior()
    ↓
cumprod 找最长连续接受前缀
    ↓
update_inference_inputs()
    ↓
更新 input_ids 和 KV cache
```

其中最值得记住的代码模式是：

```
posterior_mask = candidate == target_argmax
accepted_prefix = torch.cumprod(
    posterior_mask,
    dim=1,
)
```

这就是“最长连续接受前缀”的向量化实现。

下一步学习重点是：Medusa 的 Tree Attention 如何把树结构转换成普通 Transformer 可以执行的 Attention Mask。
