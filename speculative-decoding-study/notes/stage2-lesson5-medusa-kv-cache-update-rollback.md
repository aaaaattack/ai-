
## 阶段 2 · 第 5 讲：Medusa 的 KV Cache 更新与回滚

本节核心：Medusa 验证完候选路径后，必须把选中路径对应的 KV Cache 保留下来，并丢弃其他分支。

### 1. 为什么需要更新 KV Cache？

Medusa 一次会计算整棵候选树：

```text
root
├── A
│   ├── A1
│   └── A2
└── B
    └── B1
```

但最终只会选择一条路径，例如：

```text
root → A → A1
```

因此需要：

```text
保留 root、A、A1 的 KV Cache
丢弃 A2、B、B1 的 KV Cache
```

否则下一轮推理会把错误分支也当作历史上下文。

### 2. 源码入口

Medusa 的缓存更新函数：

```text
D:\speculative-decoding-study\repos\Medusa\medusa\model\utils.py:531-600
```

核心函数：

```python
update_inference_inputs(...)
```

它主要完成四件事：

1. 找到最佳候选路径；
2. 将接受的 token 追加到输入序列；
3. 按最佳路径重排 KV Cache；
4. 更新当前有效长度和下一轮 logits。

### 3. 找到最佳路径

源码首先计算当前输入长度：

```python
prev_input_len = input_ids.shape[1]
```

然后利用 `retrieve_indices` 找到最佳路径在树结构中的位置：

```python
select_indices = (
    retrieve_indices[best_candidate, : accept_length + 1]
    + prev_input_len
)
```

这里的关键是：

```text
retrieve_indices
```

它负责把候选树中的路径映射回实际缓存位置。

例如：

```text
树节点编号：
root = 0
A    = 1
A1   = 3
A2   = 4
B    = 2
B1   = 5
```

如果最佳路径是：

```text
root → A → A1
```

那么 `retrieve_indices` 可能得到：

```text
[0, 1, 3]
```

加上原始输入长度后，就可以定位到 KV Cache 中对应的位置。

### 4. 将接受的 token 追加到输入

源码：

```python
input_ids = torch.cat(
    [
        input_ids,
        candidates[
            None,
            best_candidate,
            : accept_length + 1
        ]
    ],
    dim=-1
)
```

注意这里是：

```python
accept_length + 1
```

而不是单纯的 `accept_length`。

原因是一次验证完成后，除了接受的候选 token，还需要保留一个新的 target token，作为下一轮生成的起点。

可以理解为：

```text
accept_length：
已经确认正确的候选 token 数量

accept_length + 1：
接受的候选 token + 下一轮自回归起点
```

### 5. KV Cache 的重排

源码首先从原始 Cache 中取出最佳路径：

```python
tgt = past_key_values_data[
    ...,
    select_indices,
    :
]
```

然后确定目标位置：

```python
dst = past_key_values_data[
    ...,
    prev_input_len : prev_input_len + tgt.shape[-2],
    :
]
```

最后进行原地复制：

```python
dst.copy_(tgt, non_blocking=True)
```

这个过程可以抽象为：

```text
原始 Tree KV Cache
        ↓
按照最佳路径读取
        ↓
覆盖到连续 Cache 区域
        ↓
下一轮只使用最佳路径
```

Medusa 并不需要重新分配一份完整 KV Cache，而是直接在已有缓存中完成重排。

### 6. 为什么需要 `retrieve_indices`？

候选树的存储顺序与最佳路径的顺序不一定相同。

例如，树结构可能被展开为：

```text
[root, A, B, A1, A2, B1]
```

最佳路径是：

```text
[root, A, A1]
```

它们对应的索引是：

```text
[0, 1, 3]
```

如果直接按照连续位置 `[0, 1, 2]` 读取，就会错误地把 `B` 当成 `A1`。

因此：

```text
tree_indices：
树结构中的节点位置

retrieve_indices：
某条候选路径对应的实际读取顺序
```

### 7. 更新当前长度

源码：

```python
current_length_data.fill_(
    prev_input_len + tgt.shape[-2]
)
```

它记录下一轮推理应该看到的有效序列长度。

例如：

```text
原始上下文长度 = 100
本轮接受路径长度 = 3
```

那么：

```text
下一轮有效长度 = 103
```

后续模型只应该从位置 103 继续生成，而不是重新处理整个候选树。

### 8. 更新 logits

源码：

```python
logits = logits[
    None,
    best_candidate,
    accept_length : accept_length + 1
]

medusa_logits = medusa_logits[
    :,
    None,
    best_candidate,
    accept_length : accept_length + 1
]
```

这一步保留最佳路径最后一个位置的输出，用作下一轮生成的起点。

因此，下一轮流程是：

```text
保留最佳路径最后状态
        ↓
使用对应 logits
        ↓
继续生成下一棵候选树
```

### 9. Medusa 中的“回滚”是什么？

Medusa 通常不是把错误分支逐个删除，而是通过重新排列 Cache 来实现逻辑回滚：

```text
错误分支的 Cache 不再被 current_length_data 覆盖
```

也就是说：

```text
物理上：Cache 中可能仍有旧数据
逻辑上：current_length 之后的数据全部无效
```

这种方式比逐层释放和重新申请显存更高效。

### 10. 与 DFlash 的对比

两者都需要处理接受和拒绝的 token，但结构不同：

```text
DFlash：
线性候选序列
    ↓
验证前缀
    ↓
回滚未接受尾部

Medusa：
树形候选序列
    ↓
选择最佳分支
    ↓
按照 retrieve_indices 重排 Cache
```

因此 Medusa 的缓存管理重点是：

```text
Tree Cache → Best Path Cache
```

而 DFlash 的缓存管理重点是：

```text
Tentative Cache → Accepted Prefix Cache
```

### 11. 本节核心结论

需要记住：

1. `best_candidate` 决定保留哪条候选路径；
2. `retrieve_indices` 将树路径映射到 Cache 位置；
3. `copy_` 将最佳路径 Cache 压缩到连续区域；
4. `current_length_data` 决定下一轮有效上下文长度；
5. Medusa 通过 Cache 重排实现逻辑回滚。
