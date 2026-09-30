# 第 2 讲：Rejection Sampling

## 1. 为什么 Greedy 规则不够

Greedy 模式只比较：

```
argmax(target_logits) == draft_token
```

但如果使用随机采样：

```
temperature > 0
top-k
top-p
```

Target 和 Draft 可能分别采样出不同 token，即使 Draft token 不是 Target 的最大概率 token，也可能仍然应该被接受。

因此需要根据概率分布决定是否接受。

------

## 2. 基本符号

设词表中有四个 token：

```
[a, b, c, d]
```

Target 分布：

```
p = [0.5, 0.2, 0.2, 0.1]
```

Draft 分布：

```
q = [0.4, 0.3, 0.1, 0.2]
```

其中：

```
p(x)：Target 概率
q(x)：Draft 概率
```

Draft 先从 `q` 中采样一个 token。

------

## 3. Draft token 的接受概率

接受概率是：

```
min(1, p(x) / q(x))
```

对每个 token 计算：

| token | p(x) | q(x) | 接受概率 |
| ----- | ---- | ---- | -------- |
| a     | 0.5  | 0.4  | 1        |
| b     | 0.2  | 0.3  | 2/3      |
| c     | 0.2  | 0.1  | 1        |
| d     | 0.1  | 0.2  | 1/2      |

解释：

- 如果 `p(x) >= q(x)`，Target 认为这个 token 的概率不低于 Draft，直接接受。
- 如果 `p(x) < q(x)`，按比例降低接受概率。

例如 Draft 采样到 `b`：

```
accept_prob = 0.2 / 0.3 = 2/3
```

以 `2/3` 概率接受，以 `1/3` 概率拒绝。

------

## 4. 拒绝后不能直接从 Target 分布采样

这是最关键的地方。

如果 Draft 采样到 `b`，然后被拒绝，不能简单地执行：

```
sample_from(p)
```

因为 Draft 已经贡献了一部分概率质量，直接从完整的 `p` 重新采样会重复计算概率，导致最终分布改变。

正确做法是从残差分布采样。

------

## 5. 残差分布

先计算：

```
max(0, p(x) - q(x))
```

本例中：

```
p - q = [0.1, -0.1, 0.1, -0.1]
```

取正值：

```
residual = [0.1, 0, 0.1, 0]
```

归一化：

```
residual_sum = 0.2
```

所以拒绝后的重新采样分布是：

```
p'(x) = [0.5, 0, 0.5, 0]
```

也就是：

```
50% 采样 a
50% 采样 c
```

------

## 6. 为什么这样能保持 Target 分布

对于 token `x`：

### Draft 被接受的概率质量

```
q(x) × min(1, p(x)/q(x))
= min(p(x), q(x))
```

### 被拒绝后补充的概率质量

```
max(0, p(x) - q(x))
```

两者相加：

```
min(p(x), q(x)) + max(0, p(x)-q(x))
= p(x)
```

因此最终输出概率正好是：

```
p(x)
```

这就是 rejection sampling 的核心正确性。

------

## 7. 多 token Draft 时的流程

假设 Draft 产生：

```
[d1, d2, d3, d4]
```

执行过程：

```
1. 根据 q(x1) 采样 d1
2. 按 min(1, p(x1)/q(x1)) 接受或拒绝 d1
3. 如果接受，继续处理 d2
4. 如果拒绝，从残差分布采样替代 token，并结束本轮
5. 如果 d1~d4 全部接受，再从 Target 分布采样 bonus token
```

伪代码：

```
for i in range(num_draft_tokens):
    token = draft_tokens[i]

    accept_prob = min(
        1.0,
        target_prob[i, token] / draft_prob[i, token],
    )

    if random() < accept_prob:
        emit(token)
    else:
        emit(sample_residual(target_prob[i], draft_prob[i]))
        stop_round()
        break
else:
    emit(sample_target(target_prob[-1]))
```

注意：

> 一旦某个 Draft token 被拒绝，后面的 Draft token 全部丢弃。

原因仍然是：后续 token 是基于已经被拒绝的前缀产生的。

对应 `llama.cpp` 的真实实现：

```text
llama.cpp/examples/speculative/speculative.cpp:296-386
```

其中 `p_tgt/p_dft` 用于接受判断，拒绝后把 `max(0, p_tgt-p_dft)` 写回 Target 分布并重新采样。

------

## 8. 与当前工程的关系

当前 DFlash/MTP 工程中使用的是：

```
np.argmax(...)
```

因此当前实现是：

```
greedy speculative decoding
```

而不是完整的：

```
probabilistic speculative sampling
```

工程中的：

```
verify_speculative()
```

主要完成：

```
Target argmax 与 Draft token 的逐位置比较
```

它没有使用：

```
p(x) / q(x)
```

也没有构造：

```
max(0, p(x)-q(x))
```

因此当前代码的接受规则更简单，但适用范围主要是 Argmax/greedy。

------

## 9. 本节必须掌握的结论

1. Draft token 的接受概率是：

```
min(1, p(x)/q(x))
```

1. 被拒绝后从残差分布采样：

```
max(0,p(x)-q(x))
```

1. 接受质量和残差质量相加，正好恢复 Target 分布。
2. 拒绝后后续 Draft token 全部丢弃。
3. 全部 Draft token 接受后，还要从 Target 采样 bonus token。
4. 当前 DFlash 工程实现的是 greedy 版，而不是 rejection sampling 版。

## 练习

使用：

```
p = [0.5, 0.2, 0.2, 0.1]
q = [0.4, 0.3, 0.1, 0.2]
```

回答：

1. Draft 采样到 `b` 时，接受概率是多少？
2. 如果 `b` 被拒绝，残差分布是什么？
3. 如果 Draft 采样到 `c`，接受概率是多少？
4. 为什么最终分布仍然等于 `p`？

## 10. 关联工程的最简实现：llama.cpp

本讲对应 `llama.cpp` 的随机采样分支。它在接受 Draft token 时读取 Target/Draft 概率：

```cpp
// examples/speculative/speculative.cpp:296-320
// p_tgt 和 p_dft 是 Target/Draft 对当前 Draft token 的概率。
if (r <= p_tgt / p_dft) {
    accept = true;
    token_id = drafts[s].tokens[i_dft];
} else {
    accept = false;
    drafts[s].active = false;
}
```

拒绝后，源码构造残差分布并归一化：

```cpp
// examples/speculative/speculative.cpp:323-354
for (size_t i = 0; i < dist_tgt.size; i++) {
    dist_tgt.data[i].p = std::max(
        0.0f,
        dist_tgt.data[i].p - dist_dft.data[i].p
    );
    sum_probs += dist_tgt.data[i].p;
}
for (size_t i = 0; i < dist_tgt.size; i++) {
    dist_tgt.data[i].p /= sum_probs;
}
```

如果 Draft 全部被拒绝，则从残差分布采样：

```cpp
// examples/speculative/speculative.cpp:372-386
std::vector<float> probs(dist_tgt.size);
for (size_t i = 0; i < dist_tgt.size; ++i) {
    probs[i] = dist_tgt.data[i].p;
}
std::discrete_distribution<> dist(probs.begin(), probs.end());
token_id = dist(rng);
```

这正对应本讲的：接受概率 `min(1,p/q)`、残差 `max(0,p-q)` 和拒绝后重新采样。

## 1. Draft 采样到 `b` 时，接受概率是多少？

已知：

```
p(b) = 0.2
q(b) = 0.3
```

接受概率：

```
min(1, p(b) / q(b))
= min(1, 0.2 / 0.3)
= 2/3
```

所以：

```
接受概率 = 66.7%
拒绝概率 = 33.3%
```

原因是 Draft 对 `b` 的估计概率高于 Target，说明 Draft 有些“高估”了 `b`。

------

## 2. 如果 `b` 被拒绝，残差分布是什么？

原始分布：

```
p = [0.5, 0.2, 0.2, 0.1]
q = [0.4, 0.3, 0.1, 0.2]
```

计算：

```
max(0, p-q)
= [0.1, 0, 0.1, 0]
```

残差总和：

```
0.1 + 0.1 = 0.2
```

归一化后：

```
p_residual = [0.5, 0, 0.5, 0]
```

所以拒绝 `b` 后：

```
50% 选择 a
50% 选择 c
```

不会选择 `b` 或 `d`。

原因是：

- `b` 已经被 Draft 高估，没有剩余概率需要补充
- `d` 的 Target 概率低于 Draft，也没有剩余概率
- `a` 和 `c` 的 Target 概率高于 Draft，需要补回差额

------

## 3. 如果 Draft 采样到 `c`，接受概率是多少？

已知：

```
p(c) = 0.2
q(c) = 0.1
```

接受概率：

```
min(1, 0.2 / 0.1)
= min(1, 2)
= 1
```

因此：

```
c 一定接受
```

原因是 Draft 没有高估 `c`，反而低估了它。Target 认为 `c` 的概率更高，所以不需要拒绝。

------

## 4. 为什么最终分布仍然等于 `p`？

对任意 token `x`，它最终获得的概率质量由两部分组成。

### 第一部分：Draft token 被接受

Draft 采样到 `x` 的概率是：

```
q(x)
```

接受概率是：

```
min(1, p(x)/q(x))
```

两者相乘：

```
q(x) × min(1, p(x)/q(x))
= min(p(x), q(x))
```

### 第二部分：Draft 被拒绝后的残差补偿

残差概率是：

```
max(0, p(x)-q(x))
```

### 两部分相加

```
min(p(x), q(x)) + max(0, p(x)-q(x))
= p(x)
```

因此每一个 token 最终获得的概率都正好等于 Target 分布中的概率。

这就是 rejection sampling 的无损性来源。

------

## 用本例验证

以 token `a` 为例：

```
p(a) = 0.5
q(a) = 0.4
```

Draft 接受部分贡献：

```
0.4 × 1 = 0.4
```

残差补偿：

```
0.5 - 0.4 = 0.1
```

最终：

```
0.4 + 0.1 = 0.5 = p(a)
```

以 token `b` 为例：

```
p(b) = 0.2
q(b) = 0.3
```

Draft 接受部分贡献：

```
0.3 × (2/3) = 0.2
```

残差补偿：

```
max(0, 0.2-0.3) = 0
```

最终：

```
0.2 + 0 = 0.2 = p(b)
```

因此，Draft 只是改变了“如何更快得到样本”，没有改变最终 Target 分布。

## 最核心的一句话

```
Draft 接受部分提供 min(p,q)
拒绝采样部分补充 max(p-q,0)
两者正好组成 p
```

这就是为什么 rejection sampling 能在加速的同时保持 Target 模型的采样分布。
