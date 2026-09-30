
## 阶段 6 · 第 3 讲：DFlash Verify、Accepted Prefix 与多状态回滚

DFlash 的 Draft 可以并行生成，但最终结果仍然由 Target Model 决定。

原论文的核心结构是：

```text
DFlash Draft：
一次生成一个 block

Target Model：
一次验证这个 block

最终输出：
只提交 Target 确认的部分
```

因此 DFlash 的正确性不来自 Draft，而来自 Target Verify。原论文也将 DFlash 描述为“并行 Draft + Target 验证”的投机解码框架。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

---

## 1. DFlash 的一轮完整流程

假设当前正式序列是：

```text
[prefix, anchor]
```

DFlash Draft 生成：

```text
candidate_ids = [d1, d2, d3, d4]
```

Target 一次验证后得到：

```text
target_tokens = [d1, d2, x3, ...]
```

则：

```text
d1：接受
d2：接受
d3：拒绝
d4：不再接受
```

最终提交：

```text
[prefix, anchor, d1, d2, x3]
```

这一轮的核心结果是：

```text
accepted_length = 2
fallback_token = x3
```

---

## 2. DFlash Verify 的源码位置

当前 Qwen 工程中，验证逻辑位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_process.py:228-245
```

Engine 调度位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_engine.py:176-191
```

Module 层的 Target 验证位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:473-501
```

状态提交与回滚位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:503-540
```

职责划分为：

```text
qwen_dflash_engine.py：
组织整轮流程

qwen_dflash_module.py：
执行 Draft/Verify 计算和 Cache 操作

qwen_dflash_process.py：
计算接受长度和输出 token
```

---

## 3. Target Verify 输入是什么？

DFlash Draft 产生：

```text
candidate_ids
```

例如：

```text
[d1, d2, d3, d4]
```

Target Verify 需要看到：

```text
[prefix, anchor, d1, d2, d3, d4]
```

但需要注意：

```text
Target 不会把 d1、d2、d3、d4 当作已经确认的历史 token。
```

它们只是待验证输入。

逻辑上可以表示为：

```python
verify_input = concat(
    committed_context,
    candidate_ids,
)
```

Target 使用已有的正式 KV Cache：

```text
Target Cache(prefix, anchor)
```

然后对 Draft block 做一次批量 forward。

---

## 4. 为什么 Target 可以一次验证整个 Block？

假设 Draft block 为：

```text
[d1, d2, d3, d4]
```

Target 可以通过一次 forward 得到多个位置的 logits：

```text
logits_1 = Target(prefix, anchor)
logits_2 = Target(prefix, anchor, d1)
logits_3 = Target(prefix, anchor, d1, d2)
logits_4 = Target(prefix, anchor, d1, d2, d3)
```

在实现上，这些位置通常通过批量输入和 KV Cache 复用来完成。

因此验证的结果可以排列为：

```text
target prediction:
[t1, t2, t3, t4]
```

再和：

```text
draft:
[d1, d2, d3, d4]
```

进行比较。

---

## 5. Greedy Verify

在 greedy 模式下，Target 每个位置取最大概率 token：

```python
target_token_i = argmax(
    target_logits_i
)
```

接受规则：

```python
accepted_length = 0

for draft_token, target_token in zip(
    draft_tokens,
    target_tokens,
):
    if draft_token == target_token:
        accepted_length += 1
    else:
        break
```

例如：

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, x4]
```

结果：

```text
accepted_length = 2
```

注意：

```text
x4 即使后续看起来是正确 token，也不能接受。
```

因为 d3 已经错了，d4 的上下文条件也已经不成立。

---

## 6. 为什么是最长连续前缀？

Draft token 的生成条件是：

```text
d1：
基于 prefix

d2：
基于 prefix + d1

d3：
基于 prefix + d1 + d2

d4：
基于 prefix + d1 + d2 + d3
```

如果 d3 被 Target 拒绝：

```text
真实上下文不再是 prefix + d1 + d2 + d3
```

而是：

```text
prefix + d1 + d2 + x3
```

所以 d4 的 Draft 条件分布已经不成立。

因此只能保留：

```text
d1, d2
```

并使用：

```text
x3
```

作为新的真实 token。

---

## 7. Fallback Token

第一个拒绝位置的 Target token 称为：

```text
fallback token
```

例子：

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, ...]
```

第一个拒绝位置是：

```text
position = 2
```

因此：

```text
fallback_token = x3
```

最终输出：

```text
[d1, d2, x3]
```

Fallback 的作用是让生成过程继续前进：

```text
如果没有 fallback：
[prefix, d1, d2]

有 fallback：
[prefix, d1, d2, x3]
```

Target 已经计算出了 x3，因此不需要再额外执行一次完整 Target forward。

---

## 8. 三个容易混淆的长度

DFlash 中至少存在三个长度概念：

### Block Size

```text
block_size
```

表示物理 Draft block 宽度。

例如：

```text
block_size = 5
```

表示每轮固定准备 5 个位置。

### Candidate Length

```text
candidate_length
```

表示真正参与验证的候选 token 数量。

如果第一个位置是 anchor：

```text
candidate_length = block_size - 1
```

### Accepted Length

```text
accepted_length
```

表示 Target 实际接受的连续 Draft token 数量。

例如：

```text
block_size = 5
candidate_length = 4
accepted_length = 2
```

这三个值不能混用：

```text
block_size：
物理计算宽度

candidate_length：
待验证 token 数量

accepted_length：
真正提交的 Draft token 数量
```

---

## 9. Anchor、Candidate 与 Bonus 的关系

假设：

```text
block = [anchor, d1, d2, d3, d4]
```

在常见设置下：

```text
anchor：
上一轮已经确认的 token

d1、d2、d3、d4：
本轮 Draft token
```

如果 Target 全部接受：

```text
d1、d2、d3、d4
```

还可能有一个 bonus token：

```text
x5
```

于是最终新增 token 可能是：

```text
[d1, d2, d3, d4, x5]
```

如果中途拒绝：

```text
[d1, d2, x3]
```

因此工程中要明确：

```text
anchor：
本轮 Draft 的条件输入

accepted candidates：
被 Target 接受的 Draft token

fallback：
第一个拒绝位置的 Target token

bonus：
整个 Draft block 都被接受后，Target 额外产生的 token
```

---

## 10. Sampling Verify 与 Greedy Verify

### Greedy

只比较：

```text
draft_token == argmax(target_logits)
```

### Sampling

需要比较 Draft 分布和 Target 分布：

```text
p_draft(x)
p_target(x)
```

经典无损接受概率：

```text
α(x)
=
min(
    1,
    p_target(x) / p_draft(x)
)
```

如果接受：

```text
继续检查下一个 Draft token
```

如果拒绝：

```text
从 residual distribution 采样
```

DFlash 的并行 Draft 并不会改变这个原则：

```text
DFlash：
改变 q 分布的生成方式

Rejection Sampling：
决定如何保持 Target 的采样分布
```

所以：

```text
DFlash block diffusion ≠ sampling correctness

Target Verify + 正确的 rejection rule
才决定无损采样。
```

---

## 11. `first_scores` 与 `transition_scores` 在 Verify 中的意义

Draft 阶段会产生：

```text
first_scores
transition_scores
```

它们属于 Draft 分布信息。

可以理解为：

```text
first_scores：
第一个可预测位置的 q 分布

transition_scores：
后续位置的 q 分布
```

Target Verify 会得到：

```text
target_scores
```

于是 sampling 验证需要比较：

```text
p_target(candidate)
p_draft(candidate)
```

逻辑上：

```python
accept_prob = min(
    1.0,
    target_prob / draft_prob
)
```

注意：

```text
first_scores 和 transition_scores
不能直接当成 Target 结果。
```

它们只是用于：

```text
记录 Draft 概率
执行 rejection sampling
分析每个位置的接受率
```

---

## 12. Verify 后的状态

假设：

```text
old_context = [prefix, anchor]
draft = [d1, d2, d3, d4]
target = [d1, d2, x3, ...]
accepted_length = 2
fallback = x3
```

验证之前：

```text
正式状态：
[prefix, anchor]

临时状态：
[d1, d2, d3, d4]
```

验证之后：

```text
正式状态：
[prefix, anchor, d1, d2, x3]

无效状态：
d3, d4
```

需要更新：

```text
input_ids
Target KV Cache
DFlash Draft Cache
hidden state
conv state
recurrent state
context length
```

---

## 13. Target KV Cache 的回滚

假设 Target 在 Draft 阶段暂时处理了：

```text
[prefix, anchor, d1, d2, d3, d4]
```

但最终只接受：

```text
[prefix, anchor, d1, d2, x3]
```

Target Cache 不能继续使用：

```text
d3, d4
```

因此需要：

```text
保留 prefix、anchor、d1、d2
丢弃 d3、d4
加入 x3
```

逻辑边界：

```text
old_length = L
accepted_length = A
fallback = 1 token
```

最终有效长度：

```text
new_length = L + A + 1
```

例如：

```text
L = 100
A = 2
```

则：

```text
有效位置：
100：d1
101：d2
102：x3

new_length = 103
```

---

## 14. DFlash 为什么需要多状态回滚？

普通 Transformer 主要回滚：

```text
KV Cache
```

但当前 Qwen DFlash 工程还可能包含：

```text
Target KV Cache
Draft KV Cache
hidden state
conv state
recurrent state
context length
```

因为 DFlash Draft Module 可能使用：

```text
卷积状态
递归状态
Draft hidden state
```

如果只回滚 token 和 KV Cache，而不回滚这些状态：

```text
下一轮 Draft 会从错误状态继续
```

例如：

```text
input_ids 已经回滚到 d2
conv state 仍然包含 d3、d4
```

下一轮计算实际上变成：

```text
token context：
[prefix, d1, d2]

conv context：
[prefix, d1, d2, d3, d4]
```

这会产生状态不一致。

因此 DFlash 的回滚本质是：

```text
所有依赖序列长度的状态，
都必须回滚到同一个边界。
```

---

## 15. Qwen DFlash 的 Verify 逻辑分层

当前代码可以按三层理解。

### Process 层

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_process.py:228-245
```

负责：

```text
读取 Target 结果
比较 candidate_ids
计算 accepted_length
选择 fallback token
```

### Module 层 Verify

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:473-501
```

负责：

```text
执行 Target/Draft 的模型计算
生成 logits
准备验证所需状态
```

### Module 层 Commit/Rollback

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:503-540
```

负责：

```text
根据 accepted_length 选择保留状态
回滚拒绝状态
更新 context length
提交 fallback/bonus
```

这三个层次不要混淆：

```text
谁接受 token：
Process

谁计算 logits：
Module

谁修改 Cache：
Module 的 commit/rollback
```

---

## 16. 逻辑回滚与物理回滚

DFlash 可能使用两种回滚方式。

### 逻辑回滚

只更新有效长度：

```python
current_length = new_length
```

物理 Cache 中可能还残留：

```text
d3, d4
```

但因为它们超过有效长度，下一轮不会被读取。

优点：

```text
不需要移动大量数据
速度快
```

### 物理回滚

把正确状态复制到目标位置：

```python
cache[:new_length] = selected_cache[:new_length]
```

或者：

```python
destination.copy_(
    source
)
```

优点：

```text
状态内容清晰
支持复杂的非连续状态
```

代价：

```text
增加显存带宽开销
```

实际工程通常会根据状态类型分别处理：

```text
KV Cache：
可以通过长度截断

conv/recurrent state：
可能需要显式 copy 或 restore

hidden state：
可能直接选择 accepted position
```

---

## 17. DFlash 的提交示意

逻辑化简：

```python
def commit_dflash_round(
    state,
    candidate_ids,
    accepted_length,
    fallback_token,
):
    old_len = state.context_length

    # 1. 接受 Draft 前缀
    accepted_ids = candidate_ids[
        :accepted_length
    ]

    # 2. 回滚所有临时状态
    state.rollback_to(
        old_len + accepted_length
    )

    # 3. 提交 accepted Draft tokens
    state.append(
        accepted_ids
    )

    # 4. 提交 Target fallback
    state.append(
        fallback_token
    )

    # 5. 更新正式长度
    state.context_length = (
        old_len
        + accepted_length
        + 1
    )
```

真实实现可能不会按照这个顺序逐 token append，而会：

```text
预分配 block
原地复制
更新指针
一次性提交
```

但逻辑等价。

---

## 18. 三种验证结果对应的状态变化

### 全部拒绝

```text
Draft：
[d1, d2, d3, d4]

Target：
[x1, ...]
```

```text
accepted_length = 0
fallback = x1
```

提交：

```text
[prefix, x1]
```

回滚：

```text
丢弃全部 Draft 状态
```

### 部分接受

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, ...]
```

```text
accepted_length = 2
fallback = x3
```

提交：

```text
[prefix, d1, d2, x3]
```

回滚：

```text
丢弃 d3、d4 的所有状态
```

### 全部接受

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, d3, d4]
```

提交：

```text
[prefix, d1, d2, d3, d4]
```

如果有 bonus token：

```text
[prefix, d1, d2, d3, d4, bonus]
```

此时没有普通意义上的拒绝位置，需要按照工程接口处理 bonus 状态。

---

## 19. 为什么 DFlash 需要固定 Verify Block？

固定 block 有利于：

```text
固定 GPU Tensor shape
复用 Attention Kernel
使用 CUDA Graph
减少动态内存管理
```

但逻辑提交长度仍然动态：

```text
block_size = 8
accepted_length = 0~7
```

因此 DFlash 的 Runtime 同时维护：

```text
固定物理 block：
用于计算

动态逻辑 prefix：
用于提交和回滚
```

这也是 DFlash 工程比普通线性 Draft 更复杂的原因：

```text
计算张量是固定的
有效序列是动态的
```

---

## 20. DFlash Verify 的性能公式

设：

```text
K：
候选 Draft token 数量

A：
平均 accepted_length

T_d：
DFlash Draft 时间

T_v：
Target Verify 时间

T_c：
Commit/Rollback 时间
```

每轮平均生成：

```text
A + 1
```

个新 token。

吞吐近似：

```text
throughput
=
(A + 1)
/
(T_d + T_v + T_c)
```

DFlash 的目标是：

```text
通过 block 并行降低 T_d
通过 Target Feature 提高 A
通过固定 shape 降低 T_c
```

如果只降低 `T_d`，但 `A` 很低：

```text
最终速度仍然可能不理想。
```

---

## 21. 与普通 Draft Verify 的区别

普通线性 Draft：

```text
Draft：
[d1, d2, d3, d4]

Target：
逐位置验证

Cache：
线性前缀回滚
```

DFlash：

```text
Draft：
固定宽度 block
可能含 anchor/noise position

Target：
处理 block 对齐的验证输入

Cache：
同时回滚：
KV
hidden
conv
recurrent
context length
```

所以 DFlash 的主要难点不是接受规则本身，而是：

```text
如何让复杂 Block 状态在接受/拒绝之后保持一致。
```

---

## 22. 本节核心结论

需要记住：

1. DFlash Draft 只负责提出候选，Target 决定最终结果；
2. `accepted_length` 是最长连续接受前缀长度；
3. 第一个拒绝位置必须使用 Target fallback；
4. `block_size` 是固定物理宽度，`accepted_length` 是动态逻辑长度；
5. 拒绝 token 后面的所有状态都必须回滚；
6. 回滚对象不只是 Target KV Cache，还可能包括：

```text
Draft KV Cache
hidden state
conv state
recurrent state
context length
```

7. DFlash 的正确性仍然来自：

```text
Target Verify
+
正确的 greedy/rejection sampling 规则
```

8. 当前 Qwen 工程的关键源码链是：

```text
qwen_dflash_process.py:228-245
        ↓
qwen_dflash_module.py:473-501
        ↓
qwen_dflash_module.py:503-540
```

下一部分将学习：

**DFlash2 的状态设计、与 DFlash 的差异，以及 block draft 的性能优化。**