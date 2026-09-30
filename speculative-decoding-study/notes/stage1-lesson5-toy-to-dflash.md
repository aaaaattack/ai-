# 阶段 1 · 第 4 讲：从 Toy 模拟器到 DFlash 实现

前面的 Toy 模拟器只有三个步骤：

```
Draft
  ↓
Verify
  ↓
Accept
```

当前 DFlash 工程在此基础上增加了：

```
Prefill
模型图调用
KV cache
Draft context
状态回滚
性能统计
```

------

## 1. Toy 模拟器与 DFlash 的对应关系

| Toy 模拟器        | DFlash 工程                           |
| ----------------- | ------------------------------------- |
| `draft`           | `module.draft_decode()`               |
| `verify_greedy()` | `process.verify_speculative()`        |
| `target_rows`     | `module.verify()` 返回的 logits       |
| `accepted`        | 接受的 Draft token 数                 |
| `next_token`      | Target 的 fallback/bonus token        |
| `emitted`         | 本轮真实提交 token 数                 |
| 无状态            | Target/Draft KV cache 和 hidden state |

DFlash 的主循环位于：

```
qwen_dflash_engine.py
```

------

## 2. DFlash 的完整主循环

对应当前 Qwen3.5 工程：

```text
qwen_dflash_engine.py:176-191
```

核心逻辑可以简化为：

```
draft_result = self.module.draft_decode(
    current,
    past_seq_len,
    self.process.build_noise_embedding(current),
)

draft = self.process.select_draft(
    draft_result,
    self.module.num_draft_tokens,
)

verify, target_hidden = self.module.verify(
    current,
    draft,
    past_seq_len,
    self.process.build_verify_block(current, draft),
)

accepted, next_token = self.process.verify_speculative(
    verify,
    draft,
    verify_length=self.module.block_size,
)

self.module.rollback_linear_cache(accepted + 1)

emitted = accepted + 1

self.module.commit_draft_context(
    target_hidden,
    past_seq_len,
    emitted,
)

self.module.advance(emitted)
```

可以拆成五个阶段。

------

# 3. 阶段一：Draft

```
draft_result = self.module.draft_decode(...)
```

DFlash Draft 图输出：

```
candidate_ids
first_scores
transition_scores
```

然后：

```
draft = self.process.select_draft(...)
```

得到：

```
draft = [d1, d2, ..., d7]
```

与 Toy 模拟器不同的是，DFlash 不是直接返回一个简单 token 列表，而是先返回候选结构和分数，再通过 `select_draft()` 选择一条 Draft 路径。

------

## 为什么有 transition scores？

第一个 Draft token 可能有多个候选：

```
d1 ∈ {A, B, C}
```

选出 `d1` 后，第二个 token 的候选概率会依赖 `d1`：

```
d2 ∈ candidates(d1)
```

所以后续 token 不是完全独立选择，而是沿着候选转移路径选择：

```
first_scores
    ↓
选择第一个 token
    ↓
transition_scores
    ↓
选择后续 token
```

这就是 DFlash 与简单逐 token Argmax Draft 的区别之一。

------

# 4. 阶段二：构造 Verify Block

DFlash 默认：

```
num_draft_tokens = 7
verify_length = 8
```

工程会构造：

```
[current, d1, d2, d3, d4, d5, d6, d7]
```

对应：

```
self.process.build_verify_block(current, draft)
```

如果 Draft token 不足固定 block 长度，就使用 `noise_token_id` 补齐：

```
[current, d1, d2, ..., noise, noise]
```

但补齐位置会通过 mask 屏蔽，不参与有效验证。

------

# 5. 阶段三：Target Verify

```
verify, target_hidden = self.module.verify(...)
```

Target 一次性运行整个 block。

输出包括：

```
verify.logits
target_hidden
```

其中：

```
verify.logits
```

用于判断：

```
d1 是否正确
d2 是否正确
...
d7 是否正确
```

而：

```
target_hidden
```

用于后续更新 Draft context 和状态。

------

## DFlash 中的 logits 行对应关系

假设 block 是：

```
[current, d1, d2, d3, d4, d5, d6, d7]
```

工程会把 logits reshape 成：

```
rows = values.reshape(verify_length, -1)
```

然后：

```
rows[0] → 判断 d1
rows[1] → 判断 d2
rows[2] → 判断 d3
...
rows[6] → 判断 d7
rows[7] → bonus token
```

------

# 6. 阶段四：计算接受长度

代码逻辑：

```
accepted = 0

for index, token in enumerate(draft):
    if int(np.argmax(rows[index])) != token:
        break
    accepted += 1

next_token = int(np.argmax(rows[accepted]))
```

假设：

```
Draft：
[d1, d2, d3, d4, d5, d6, d7]

Target：
[d1, d2, x3, ...]
```

则：

```
accepted = 2
next_token = x3
```

本轮输出：

```
[d1, d2, x3]
```

这和 Toy 模拟器完全一致。

------

# 7. 阶段五：为什么要回滚 cache？

Verify 阶段计算了整个 block：

```
current, d1, d2, d3, d4, d5, d6, d7
```

但实际只接受：

```
d1, d2, x3
```

因此不能直接把整个 Verify 的状态都保留下来。

工程执行：

```
self.module.rollback_linear_cache(accepted + 1)
```

这里：

```
accepted + 1 = 3
```

表示真正提交了：

```
d1、d2、x3
```

而不是：

```
accepted = 2
```

因为 `accepted` 只统计 Draft token，不包括 Target 的 `next_token`。

------

# 8. 为什么还要提交 Draft context？

Target 和 Draft 不能各自保留不同的历史。

工程执行：

```
self.module.commit_draft_context(
    target_hidden,
    past_seq_len,
    emitted,
)
```

其中：

```
emitted = accepted + 1
```

作用是：

```
把 Target 已确认的 hidden state
同步给 Draft context
```

下一轮 Draft 必须基于：

```
原始上下文 + 本轮真正接受的 token
```

而不能基于被拒绝的 Draft token。

------

# 9. 更新上下文长度

最后：

```
self.module.advance(emitted)
```

如果本轮接受两个 Draft token，再加一个 Target token：

```
emitted = 3
```

则：

```
context_length += 3
```

不能写成：

```
context_length += 8
```

因为 Verify block 长度是 8，但真正确认的 token 只有 3 个。

------

# 10. 为什么 `current` 会变成 `next_token`

本轮提交后：

```
current = next_token
```

下一轮 Draft 以这个最新确认的 token 作为 anchor：

```
上一轮：
[d1, d2, x3]

下一轮：
以 x3 为起点继续 Draft
```

这保证了 Draft 始终从真实确认的上下文继续，而不是从错误候选继续。

------

# 11. DFlash 的状态流转

可以画成：

```
Prefill
  │
  ├── Target KV cache
  ├── Draft context
  └── current token
        │
        ▼
Draft block
        │
        └── draft tokens
              │
              ▼
Target Verify
        │
        ├── verify logits
        └── target hidden
              │
              ▼
Accept prefix
        │
        ├── accepted draft tokens
        └── next token
              │
              ▼
Rollback unused state
              │
              ▼
Commit accepted hidden/cache
              │
              ▼
Advance context length
              │
              ▼
下一轮
```

------

# 12. DFlash 与 Toy 模拟器的本质区别

Toy 模拟器解决的是：

```
算法正确性
```

DFlash 还要解决：

```
硬件执行
HMM 图输入输出
KV cache 复用
conv/recurrent state
固定 block shape
attention mask
设备 Tensor 绑定
性能统计
```

因此 DFlash 的复杂度主要不在接受规则，而在：

```
如何让“暂时计算的 block”
最终只提交“真正接受的前缀”
```

------

## 本节总结

Toy 模拟器中的：

```
emitted = accepted + 1
```

在 DFlash 中对应：

```
rollback_linear_cache(accepted + 1)
commit_draft_context(..., accepted + 1)
advance(accepted + 1)
```

这是理解当前工程最关键的映射。

## 关联工程的最简实现：Qwen3.5 DFlash

本节对应当前工程：

```text
qwen_dflash_engine.py:176-191
```

```python
draft_result = self.module.draft_decode(current, past_seq_len, noise)
draft = self.process.select_draft(draft_result, self.module.num_draft_tokens)
verify, target_hidden = self.module.verify(current, draft, past_seq_len, block)
accepted, next_token = self.process.verify_speculative(verify, draft, verify_length=8)

emitted = accepted + 1
self.module.rollback_linear_cache(emitted)
self.module.commit_draft_context(target_hidden, past_seq_len, emitted)
self.module.advance(emitted)
```

接受规则的真实实现位于：

```text
qwen_dflash_process.py:228-245
```

```python
accepted = 0
for index, token in enumerate(draft):
    if int(np.argmax(rows[index])) != token:
        break
    accepted += 1
next_token = int(np.argmax(rows[accepted]))
```

cache 状态推进位于：

```text
qwen_dflash_module.py:503-540
```

下一步进入阶段 2：学习 Medusa，理解它如何通过多个 decoding heads 和 candidate tree 降低 Draft 阶段的串行开销。
