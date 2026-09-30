# 阶段 2 · 第 1 讲：Medusa 的核心思想

## 1. Medusa 解决什么问题

经典投机解码需要：

```text
小 Draft Model 预测
大 Target Model 验证
```

Medusa 改成：

```text
一个 Target Model
    ├── 原始 LM Head
    ├── Medusa Head 1
    ├── Medusa Head 2
    ├── Medusa Head 3
    └── ...
```

因此不需要额外加载一个完整 Draft Model。

## 2. Medusa Head

当前上下文最后一个 hidden state 为 `h_t`：

```text
原始 LM Head      -> 预测 x(t+1)
Medusa Head 1     -> 预测未来 token
Medusa Head 2     -> 预测更远 token
Medusa Head 3     -> 预测更远 token
```

源码位置：

```text
repos/Medusa/medusa/model/medusa_model_new.py
```

重点代码：

对应本地工程：

```text
Medusa/medusa/model/medusa_model_new.py:73-79,171-176
```

```python
self.medusa_head = nn.ModuleList(...)

for i in range(self.medusa):
    medusa_logits.append(self.medusa_head[i](hidden_states))
```

多个 Head 共享 Target Backbone 的 hidden state。

## 3. Candidate Tree

如果每个 Head 只保留一个候选：

```text
Head 1 -> A
Head 2 -> B
Head 3 -> C
```

未来 token 的条件依赖没有被充分表达。因此 Medusa 通常保留每个位置的多个候选，形成树：

```text
root
 ├── A
 │   ├── A1
 │   └── A2
 └── B
     ├── B1
     └── B2
```

候选路径为：

```text
[A, A1]
[A, A2]
[B, B1]
[B, B2]
```

## 4. Tree Attention

Medusa 不逐条验证候选路径，而是把树打平，一次 Target forward，并使用 Tree Attention 限制可见范围。

节点可见关系：

```text
A  -> root
B  -> root
A1 -> root, A
A2 -> root, A
B1 -> root, B
B2 -> root, B
```

不同分支不能互相关注：

```text
A1 不能看 B
B1 不能看 A
A1 不能看 A2
```

## 5. 相关源码

源码目录：

```text
D:/speculative-decoding-study/repos/Medusa/medusa/model
```

重点文件：

| 文件 | 作用 |
|---|---|
| `medusa_model_new.py` | Medusa Heads 和生成主循环 |
| `utils.py` | 候选树、attention mask、候选路径和验证逻辑 |
| `medusa_choices.py` | 预定义候选树结构 |
| `kv_cache.py` | KV cache 初始化与管理 |
| `medusa_model.py` | 模型封装与兼容逻辑 |

## 6. Medusa 推理流程

```text
Target Backbone hidden state
        ↓
Medusa Heads 输出
        ↓
generate_candidates()
        ↓
构造 candidate tree
        ↓
tree_decoding()
        ↓
Target 一次验证整棵候选树
        ↓
evaluate_posterior()
        ↓
选择最佳路径
        ↓
update_inference_inputs()
```

主要函数：

```python
initialize_medusa(...)
generate_candidates(...)
tree_decoding(...)
evaluate_posterior(...)
update_inference_inputs(...)
```

## 7. 与其他方法的区别

### 经典 Draft Model

```text
Target Model + 独立小 Draft Model
```

需要额外模型、显存和模型切换成本。

### Medusa

```text
Target Backbone + 多个轻量级 Head
```

不需要完整 Draft Model，但需要训练额外 Heads，并维护候选树。

### MTP

MTP 更偏向模型训练目标或原生结构；Medusa 更偏向在已有模型上增加多预测头和树解码框架。

### DFlash

DFlash 重点是 block-level Draft 图；Medusa 重点是 multi-head prediction + candidate tree + Tree Attention。

## 8. 本讲结论

Medusa 的核心不是单纯增加几个 Linear Head，而是：

```text
Multi-head prediction
        +
Candidate tree
        +
Tree Attention
        +
Path selection
```

## 关联工程的最简实现：Medusa

本阶段使用：

```text
repos/Medusa/medusa/model/medusa_model_new.py
repos/Medusa/medusa/model/utils.py
```

Head 构造与输出：

```python
# medusa/model/medusa_model_new.py:73-79, 171-176
self.medusa_head = nn.ModuleList(...)
medusa_logits = []
for i in range(self.medusa):
    medusa_logits.append(self.medusa_head[i](hidden_states))
return torch.stack(medusa_logits, dim=0)
```

树结构和 attention mask：

```python
# medusa/model/utils.py:32-59
def generate_medusa_buffers(medusa_choices, device="cuda"):
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_attn_mask = torch.eye(len(sorted_choices) + 1)
    medusa_attn_mask[:, 0] = 1
```

Greedy 模式中用 `cumprod` 只接受连续前缀：

```python
# medusa/model/utils.py:436-458
posterior_mask = (
    candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
).int()
candidates_accept_length = torch.cumprod(
    posterior_mask, dim=1
).sum(dim=1)
```
