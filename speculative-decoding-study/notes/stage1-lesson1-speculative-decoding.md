# 阶段 1 · 第 1 讲：经典 Speculative Decoding

学习材料：

论文的核心思想可以概括为：

> 让便宜的 Draft Model 先猜，让昂贵的 Target Model 批量验证；猜对的连续 token 直接复用，猜错的由 Target 修正。

------

## 1. 普通自回归解码的问题

假设当前上下文是：

```
x1, x2, x3
```

普通解码需要：

```
Target(x1,x2,x3)       → x4
Target(x1,x2,x3,x4)     → x5
Target(x1,x2,x3,x4,x5) → x6
```

每次只产生一个 token，Target Model 被连续调用多次。

投机解码改成：

```
Draft(x1,x2,x3) → [x4', x5', x6', x7']
Target(x1,x2,x3,[x4',x5',x6',x7']) → 一次验证
```

因为 Target 可以在一次 forward 中并行计算多个位置，所以有机会用一次 Target forward 替代多次 Target decode。

对应的经典工程实现位于：

```text
llama.cpp/examples/speculative-simple/speculative-simple.cpp:187-258
```

```cpp
// llama.cpp/examples/speculative-simple/speculative-simple.cpp:221-258
common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
for (size_t i = 0; i < draft.size(); ++i) {
    common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
}
llama_decode(ctx_tgt, batch_tgt);
auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
```

------

# 2. Draft Model 和 Target Model

## Target Model

Target Model 是最终可信的模型，通常是：

- 参数量更大
- 计算更慢
- 质量更高
- 决定最终输出

记它在某个位置上的概率分布为：

```
p(x)
```

## Draft Model

Draft Model 是快速预测器，通常是：

- 更小的模型
- MTP head
- EAGLE/DFlash speculator
- N-gram 或其他启发式方法

记它的概率分布为：

```
q(x)
```

Draft Model 不需要完全正确，只需要：

```
尽量便宜
尽量接近 Target
```

接近程度越高，接受的 draft token 越多。

------

# 3. Greedy Accept-if-Equal

这是当前工程最容易理解、也最直接使用的模式。

假设 Draft 生成：

```
d1, d2, d3, d4
```

Target 对对应位置的 greedy 结果是：

```
t1, t2, t3, t4
```

逐个比较：

```
accepted = 0

for i, draft_token in enumerate(draft_tokens):
    target_token = argmax(target_logits[i])

    if target_token != draft_token:
        break

    accepted += 1
```

例如：

```
Draft： [A, B, C, D]
Target：[A, B, X, ...]
```

那么：

```
A 接受
B 接受
C 拒绝
D 丢弃
```

最终输出：

```
[A, B, X]
```

这里的 `X` 是 Target 在第一个不匹配位置生成的 token。

------

## 为什么后续 token 必须全部丢弃？

因为 `D` 是基于错误的 `C` 继续预测出来的：

```
A → B → C → D
        ↑
       错误
```

一旦 `C` 错了，`D` 的条件上下文就已经不成立，因此只能接受最长连续前缀。

这个前缀叫：

```
accepted prefix
```

------

# 4. Bonus Token

假设 Draft 预测 4 个 token：

```
d1, d2, d3, d4
```

Target 验证后：

```
d1, d2, d3, d4, d5
```

如果全部匹配，那么：

```
accepted draft tokens = 4
bonus token = d5
```

本轮实际提交：

```
d1, d2, d3, d4, d5
```

因此一轮最多提交：

```
draft_length + 1
```

这就是为什么你当前 DFlash 工程中：

```
7 个 draft token
+ 1 个 Target bonus token
= 8 个 verify positions
```

对应代码：

```
accepted, next_token = self.process.verify_speculative(...)
```

其中：

- `accepted`：接受了多少个 Draft token
- `next_token`：第一个不匹配位置，或 bonus 位置的 Target token

------

# 5. Rejection Sampling

Greedy 模式只适用于确定性 Argmax。

如果使用：

```
temperature > 0
top-p
top-k
```

就不能简单地使用：

```
argmax(target_logits) == draft_token
```

因为 Draft 和 Target 可能从不同概率分布中采样。

论文使用 rejection sampling 保证最终分布仍然等于 Target 分布。

设：

```
p(x)：Target 对 token x 的概率
q(x)：Draft 对 token x 的概率
```

Draft 采样得到 token `x` 后，以概率：

```
min(1, p(x) / q(x))
```

接受它。

## 两种情况

### 情况一：Target 概率更高

如果：

```
p(x) >= q(x)
```

那么：

```
p(x) / q(x) >= 1
```

接受概率为：

```
1
```

Draft token 一定接受。

### 情况二：Target 概率更低

如果：

```
p(x) < q(x)
```

接受概率为：

```
p(x) / q(x)
```

这时 Draft token 只有一定概率被接受。

------

## 拒绝之后如何重新采样？

如果 Draft token 被拒绝，不能简单重新从 Target 原始分布采样，否则会改变整体分布。

需要从残差分布中采样：

```
p'(x) = max(0, p(x) - q(x))
```

然后归一化：

```
p'(x) =
max(0, p(x)-q(x))
-----------------
Σy max(0,p(y)-q(y))
```

这部分是实现“无损随机采样”的关键。

------

# 6. 什么叫“无损”？

这里有两个层次，必须区分。

## Greedy 场景

无损通常表示：

```
投机解码输出
=
Target 单独 greedy 解码输出
```

只要每次：

```
Target argmax == Draft token
```

就接受；不匹配就使用 Target token。

当前 DFlash/MTP 工程基本属于这一类，因为代码中使用：

```
np.argmax(...)
```

## Sampling 场景

无损表示：

```
投机解码最终生成分布
=
Target 单独采样时的分布
```

这需要使用上面的 rejection sampling，而不是简单的 argmax 比较。

因此要特别注意：

> 当前工程中的 Argmax 投机解码可以保持 Target 的 greedy 结果，但不等于已经实现了完整的随机采样版 speculative sampling。

------

# 7. Acceptance Length

设一轮 Draft 产生 `K` 个 token。

定义随机变量：

```
A = 本轮被连续接受的 draft token 数
```

则：

```
0 <= A <= K
```

如果 Draft 产生：

```
[d1, d2, d3, d4]
```

Target 结果是：

```
[d1, d2, x3, ...]
```

则：

```
A = 2
```

平均接受长度为：

```
E[A]
```

如果每个位置独立地以概率 `α` 被接受，那么：

```
P(A >= i) = α^i
```

因此：

```
E[A] = α + α² + α³ + ... + αᴷ
```

等价于：

```
E[A] = α(1 - αᴷ) / (1 - α)
```

每轮通常还会产生一个 Target token，所以平均每轮输出量近似为：

```
1 + E[A]
```

这解释了为什么工程会统计：

```
accepted_draft_tokens
accepted_per_round
acceptance_rate
```

------

# 8. Acceptance Rate 和 Acceptance Length 的区别

这两个概念很容易混淆。

## Acceptance Rate

```
accepted draft tokens
/
proposed draft tokens
```

例如：

```
Draft 总共提出 700 个 token
接受了 490 个
```

则：

```
acceptance_rate = 490 / 700 = 70%
```

## Acceptance Length

每一轮实际平均接受多少个连续 Draft token：

```
accepted_draft_tokens / speculative_rounds
```

例如：

```
490 个接受 token
100 轮
```

则：

```
accepted_per_round = 4.9
```

但每轮还包含 bonus token，所以平均实际输出量可能接近：

```
5.9 token / round
```

------

# 9. 加速比模型

设：

```
C_t(1) = Target 单 token decode 的成本
C_t(K) = Target 一次验证 K 个 token 的成本
C_d(K) = Draft 产生 K 个 token 的成本
```

普通解码每生成一个 token，大致需要：

```
C_t(1)
```

投机解码每轮生成约：

```
1 + E[A]
```

个 token，每轮成本为：

```
C_d(K) + C_t(K)
```

因此粗略加速比：

```
Speedup ≈
(1 + E[A]) × C_t(1)
--------------------
C_d(K) + C_t(K)
```

这说明加速不只取决于接受率，还取决于：

- Draft Model 是否足够快
- Target Verify 是否真正高效
- Verify block 是否增加大量计算
- KV cache 是否需要频繁复制和回滚
- CPU/GPU 同步开销
- batch size 和并发量

所以：

```
高 acceptance rate ≠ 一定高 speedup
```

如果 Draft 很慢，或者 cache 回滚成本很高，即使接受率不错，也可能没有明显收益。

------

# 10. 与当前 DFlash 工程对应

当前工程中的对应关系是：

| 理论概念              | 工程实现                  |
| --------------------- | ------------------------- |
| Draft proposal        | `draft_decode()`          |
| Draft token selection | `select_draft()`          |
| Target verification   | `verify()`                |
| Greedy acceptance     | `verify_speculative()`    |
| Accepted prefix       | `accepted`                |
| Bonus token           | `next_token`              |
| Target state rollback | `rollback_linear_cache()` |
| Draft context commit  | `commit_draft_context()`  |
| Acceptance statistics | `stats()`                 |

你可以把 DFlash 的主循环简化为：

```
draft = make_draft(current)

target_logits = verify(
    current,
    draft,
)

accepted, next_token = accept_prefix(
    target_logits,
    draft,
)

commit(accepted_draft_tokens)
commit(next_token)
rollback_rejected_state()
```

------

# 本阶段必须掌握的结论

1. Draft Model 只负责提出候选，不决定最终输出。
2. Target Model 通过一次 forward 验证多个 Draft token。
3. 只能接受最长连续匹配前缀。
4. 第一个不匹配位置使用 Target token。
5. Greedy 模式使用 accept-if-equal。
6. 随机采样模式需要 rejection sampling。
7. 平均接受长度决定每轮能提交多少 token。
8. 最终加速比同时取决于接受长度、Draft 成本、Verify 成本和 cache 开销。
9. 当前 DFlash/MTP 工程主要实现的是 Argmax/greedy 版本。

## 课后练习

给定：

```
Draft：
[A, B, C, D, E]

Target：
[A, B, X, Y, Z]
```

回答：

```
accepted = ?
next_token = ?
本轮最终提交哪些 token？
D 和 E 为什么不能提交？
```

下一步应继续学习论文中的 rejection sampling 公式，并用一个 4-token 小词表手算一次完整的接受概率。

## 关联工程的最简实现：llama.cpp

本阶段对应 `llama.cpp` 的经典 Draft/Verify 实现，而不是 DFlash 工程。

```cpp
// examples/speculative-simple/speculative-simple.cpp:187-196
// 1. 让 Draft 生成一个候选 block。
common_speculative_get_draft_params(spec, seq_id) = {
    .drafting = true,
    .n_max    = n_draft_max,
    .pos0     = n_past,
    .id_last  = id_last,
    .result   = &draft,
};
common_speculative_draft(spec);
```

```cpp
// examples/speculative-simple/speculative-simple.cpp:221-258
// 2. Target 一次计算 [last_token, draft...]
common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
for (size_t i = 0; i < draft.size(); ++i) {
    common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
}
llama_decode(ctx_tgt, batch_tgt);

// 3. 根据 Target sampler 接受 Draft，并返回 bonus/fallback token。
auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
```

这三步就是经典投机解码的最小工程闭环：Draft、批量 Verify、接受前缀。
