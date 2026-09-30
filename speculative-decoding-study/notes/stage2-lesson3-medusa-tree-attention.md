# 阶段 2 · 第 3 讲：Medusa 的 Tree Attention

上一讲已经看到了：

```
多头预测
    ↓
候选树
    ↓
一次 Target forward
```

本讲重点解释：树结构如何变成 Transformer 能执行的 Attention Mask。

------

## 1. 一个具体候选树

假设 Candidate Tree 是：

```
root
 ├── A
 │   ├── A1
 │   └── A2
 └── B
     └── B1
```

对应路径：

```
[A]
[B]
[A, A1]
[A, A2]
[B, B1]
```

在 Medusa 中，这些路径通常用整数索引表示：

```
medusa_choices = [
    [0],
    [1],
    [0, 0],
    [0, 1],
    [1, 0],
]
```

含义是：

- `0`：父节点下的第一个候选
- `1`：父节点下的第二个候选

------

## 2. 树被打平为序列

Transformer 输入通常不是树，而是一维 Tensor：

```
[root, A, B, A1, A2, B1]
```

每个节点需要知道：

```
它能关注哪些节点？
```

规则是：

```
节点可以看：
1. root
2. 自己的祖先
3. 自己

节点不能看：
1. 其他分支
2. 兄弟节点
3. 其他路径的后代
```

------

## 3. Attention Mask

对应源码：

```
Medusa/medusa/model/utils.py:32-59
```

最简代码：

```
sorted_choices = sorted(
    medusa_choices,
    key=lambda x: (len(x), x),
)

medusa_len = len(sorted_choices) + 1
medusa_attn_mask = torch.eye(medusa_len, medusa_len)
medusa_attn_mask[:, 0] = 1
```

先得到一个单位矩阵：

```
每个节点只能看自己
```

然后：

```
ancestor_idx = ...
medusa_attn_mask[row, ancestor_idx] = 1
```

把祖先节点加入可见范围。

------

## 4. 具体 Mask 示例

节点顺序：

```
0: root
1: A
2: B
3: A1
4: A2
5: B1
```

对应允许注意的位置：

```
root -> root

A    -> root, A
B    -> root, B

A1   -> root, A, A1
A2   -> root, A, A2

B1   -> root, B, B1
```

Mask 可以表示为：

```
       root A B A1 A2 B1
root    1  0 0  0  0  0
A       1  1 0  0  0  0
B       1  0 1  0  0  0
A1      1  1 0  1  0  0
A2      1  1 0  0  1  0
B1      1  0 1  0  0  1
```

例如 `A1`：

```
A1 可以看 root
A1 可以看 A
A1 可以看自己
A1 不能看 B
A1 不能看 A2
A1 不能看 B1
```

这样虽然输入被打平成序列，但模型计算仍然保持树结构的因果关系。

------

## 5. Position IDs

源码中的：

```
medusa_position_ids
```

用于表示节点在树中的深度：

```
root -> 0
A/B  -> 1
A1/A2/B1 -> 2
```

示例：

```
position_ids = [0, 1, 1, 2, 2, 2]
```

注意：

```
同一深度的不同分支可以拥有相同的位置编号
```

因为它们都表示相同的相对生成深度。

对应源码：

```
Medusa/medusa/model/utils.py:85-92
```

------

## 6. Tree Indices

Medusa Head 的输出不是普通的一维 logits，而是按：

```
不同 Head
不同 Top-K 候选
```

排列的。

因此需要：

```
tree_indices
```

把树节点映射到 Head 输出中的具体候选槽位。

源码：

```
Medusa/medusa/model/utils.py:74-83
```

简化理解：

```
tree_index = parent_candidate + TOPK * depth + 1
```

例如：

```
root       -> 原始 Target logits
深度 1 节点 -> Head 1 的候选
深度 2 节点 -> Head 2 的候选
```

所以 `tree_indices` 的作用是：

```
把各个 Head 的候选结果
重新排布成树节点顺序
```

------

## 7. Retrieve Indices

Tree Attention 计算完成后，输出仍然按照打平树的顺序排列。

但后续要比较完整路径：

```
[root, A, A1]
[root, A, A2]
[root, B, B1]
```

所以还需要：

```
retrieve_indices
```

它记录：

```
每条路径对应哪些扁平节点
```

例如：

```
路径 [A, A1] -> [root_index, A_index, A1_index]
路径 [A, A2] -> [root_index, A_index, A2_index]
路径 [B, B1] -> [root_index, B_index, B1_index]
```

对应源码：

```
Medusa/medusa/model/utils.py:94-106
```

------

## 8. Tree Decode

对应源码：

```
Medusa/medusa/model/utils.py:309-340
```

核心代码：

```
position_ids = (
    medusa_position_ids
    + input_ids.shape[1]
)

tree_medusa_logits, outputs, tree_logits = model(
    tree_candidates,
    output_orig=True,
    past_key_values=past_key_values,
    position_ids=position_ids,
    medusa_forward=True,
)
```

这一步将整棵候选树一次送入 Target Model。

然后根据路径索引恢复：

```
logits = tree_logits[0, retrieve_indices]
medusa_logits = tree_medusa_logits[
    :, 0, retrieve_indices
]
```

此时 `logits` 已经可以按完整候选路径进行比较。

------

## 9. 为什么 Tree Attention 能加速？

如果不用 Tree Attention，可能需要分别执行：

```
Target(root, A, A1)
Target(root, A, A2)
Target(root, B, B1)
```

这会重复计算：

```
root
A
B
```

使用 Tree Attention 后：

```
一次 forward 同时计算：
root、A、B、A1、A2、B1
```

共享部分只计算一次，分支部分并行计算。

这就是它的主要收益：

```
共享公共前缀
减少重复 Target forward
提高 GPU 并行度
```

------

## 10. 与 DFlash 的区别

### DFlash

```
一条 block Draft 路径
    ↓
Target 验证
```

### Medusa

```
多条候选路径
    ↓
Tree Attention 一次验证
    ↓
选择接受长度最长的路径
```

因此：

```
DFlash 的核心是 block Draft
Medusa 的核心是 tree-shaped verification
```

------

## 11. 本节源码索引

```
Medusa/medusa/model/utils.py:32-125
    生成 Tree Attention buffers

Medusa/medusa/model/utils.py:258-306
    生成 tree candidates

Medusa/medusa/model/utils.py:309-340
    一次 Tree Decode

Medusa/medusa/model/utils.py:436-489
    计算各路径的 accepted prefix

Medusa/medusa/model/utils.py:531-580
    更新 input_ids 和 KV cache
```

------

## 本节总结

Medusa 把一棵树转换成 Transformer 可执行的形式，需要四类信息：

```
medusa_attn_mask
    控制祖先可见关系

medusa_position_ids
    表示树深度

tree_indices
    把 Head 输出映射到树节点

retrieve_indices
    把扁平输出恢复成候选路径
```

最终流程是：

```
树结构
  ↓
扁平 Tensor
  ↓
Tree Attention Mask
  ↓
一次 Target forward
  ↓
恢复候选路径
  ↓
选择最长接受前缀
```

下一步学习 Medusa 的 `evaluate_posterior()`，重点比较它的 greedy 接受和典型采样接受与经典 rejection sampling 的区别。
