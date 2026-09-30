
## 阶段 5 · 第 2 讲：MTP 的逐 Token Draft Chain

本节重点分析 MTP 的真实推理链：

```text
上一轮已提交状态
        ↓
MTP Draft Chain
        ↓
生成多个 draft token
        ↓
Target 批量验证
        ↓
计算 accepted prefix
        ↓
提交 Cache
        ↓
进入下一轮
```

对应当前 Qwen 工程：

```text
/workspace/imodelzoo/models/llm/qwen3.5
```

主要文件：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_process.py
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py
```

---

## 1. 先看一轮 MTP 的状态

假设当前已经提交的序列是：

```text
[prefix]
```

当前有效长度为：

```text
L
```

MTP 计划生成 4 个 Draft token：

```text
d1, d2, d3, d4
```

完整状态变化如下：

```text
初始：
[prefix]

Draft：
[prefix, d1, d2, d3, d4]

Target 验证：
Target 预测为 [d1, d2, x3, ...]

最终提交：
[prefix, d1, d2, x3]
```

其中：

```text
d1、d2：
MTP Draft 生成并被接受

x3：
Target 在第一个拒绝位置生成的 fallback token

d3、d4：
被拒绝，不能继续使用
```

因此一轮实际新增：

```text
accepted_length + 1
```

如果接受长度为 2，则新增：

```text
2 个 Draft token + 1 个 Target fallback token
```

---

## 2. MTP 的调用层次

可以把调用链分成三层。

### Engine 层

负责一轮推理调度：

```text
qwen_mtp_engine.py
```

主要工作：

```text
调用 Draft
调用 Verify
读取接受长度
触发 Cache Commit
准备下一轮输入
```

关键位置：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:145-172
```

### Process 层

负责处理 token 级别的逻辑：

```text
qwen_mtp_process.py
```

主要工作：

```text
比较 Draft token 与 Target token
计算 accepted prefix
识别第一个不匹配位置
生成 fallback token
```

关键位置：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_process.py:148-169
```

### Module 层

负责 MTP 模块的计算和 Cache：

```text
qwen_mtp_module.py
```

主要工作：

```text
运行 MTP Head
维护 MTP hidden state
维护 MTP KV Cache
提交接受状态
回滚拒绝状态
```

关键位置：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py:387-417
```

可以记成：

```text
Engine：决定什么时候做什么

Process：决定 token 接受多少

Module：真正计算并管理状态
```

---

## 3. Draft Chain 的输入是什么？

MTP Draft 并不是每一轮都从完整输入重新开始。

它的输入通常包括：

```text
input_ids
Target 最后 hidden state
Target KV Cache
MTP KV Cache
当前 context length
```

上一轮提交完成后，系统会保留：

```text
[prefix]
```

以及对应状态：

```text
Target Cache(prefix)
MTP Cache(prefix)
hidden_state(prefix)
```

下一轮 Draft 从这个已经提交的位置继续：

```text
prefix → d1 → d2 → d3 → d4
```

而不是重新计算：

```text
整个历史序列
```

这正是 KV Cache 能够降低重复计算的原因。

---

## 4. 第一个 MTP Head 如何生成 token？

假设 Target 主干刚刚处理完当前上下文，得到：

```text
h_L
```

第一个 MTP Head 使用：

```text
h_L
```

预测下一个 token：

```text
d1 = MTPHead_1(h_L)
```

逻辑上可以表示为：

```python
hidden = target_hidden

draft_token_1 = mtp_head_1(
    hidden
)
```

如果使用 greedy：

```python
d1 = argmax(logits_1)
```

如果使用 sampling：

```python
d1 = sample(logits_1)
```

当前工程中这一阶段由：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
```

中的 `_draft()` 调度。

---

## 5. 第二个 MTP Head 如何接力？

第二个 Head 通常不能简单地再次使用原始的 `h_L`。

它需要看到第一个 Draft token：

```text
d1
```

因此逻辑变成：

```text
h_L + d1
    ↓
MTP Head 2
    ↓
d2
```

抽象代码：

```python
hidden = update_hidden(
    hidden,
    draft_token_1
)

draft_token_2 = mtp_head_2(
    hidden
)
```

然后继续：

```text
h_L + d1 + d2
    ↓
MTP Head 3
    ↓
d3
```

完整链路：

```text
h_L
  ↓
Head 1 → d1
  ↓
Head 2 → d2
  ↓
Head 3 → d3
  ↓
Head 4 → d4
```

这就是 MTP 的 Draft Chain。

---

## 6. 为什么 MTP Draft 不是完全并行？

因为后一个预测依赖前一个 token：

```text
d2 = f(h_L, d1)
d3 = f(h_L, d1, d2)
d4 = f(h_L, d1, d2, d3)
```

如果 `d1` 没有产生，就无法准确计算 `d2`。

因此 MTP 通常存在串行依赖：

```text
Head 1 → Head 2 → Head 3 → Head 4
```

而 Medusa 更接近：

```text
同一个 hidden state
    ├── Head 1
    ├── Head 2
    └── Head 3
```

两者的核心区别是：

```text
MTP：
后一个预测依赖前一个预测

Medusa：
多个 head 可以从同一个状态直接产生候选
```

---

## 7. MTP Draft Cache 的作用

MTP Draft Chain 中，每个 Draft token 可能都会写入临时状态：

```text
MTP Cache position L     ← d1
MTP Cache position L + 1 ← d2
MTP Cache position L + 2 ← d3
MTP Cache position L + 3 ← d4
```

此时这些状态还不是最终提交状态。

它们属于：

```text
tentative cache
```

也就是：

```text
暂存 Cache
```

Target 验证之前，系统不能认为这些 token 一定正确。

所以需要区分：

```text
Committed Cache：
已经确认属于正式序列

Tentative Cache：
本轮 Draft 生成但尚未验证
```

状态图：

```text
Committed Cache
[prefix]
     ↓
Tentative Cache
[d1, d2, d3, d4]
```

---

## 8. Target 如何批量验证 Draft Block？

MTP 生成：

```text
[d1, d2, d3, d4]
```

Target 不需要单独运行四次：

```text
Target(d1)
Target(d2)
Target(d3)
Target(d4)
```

而是将 Draft block 放入一次验证过程：

```text
Target Verify(
    [prefix, d1, d2, d3, d4]
)
```

Target 会输出每个位置的 logits：

```text
target_logits_1
target_logits_2
target_logits_3
target_logits_4
```

然后分别得到 Target 的预测：

```text
[t1, t2, t3, t4]
```

比较：

```text
Draft：
[d1, d2, d3, d4]

Target：
[t1, t2, t3, t4]
```

在 greedy 情况下：

```python
accepted = (
    draft_tokens == target_tokens
)
```

---

## 9. 接受规则的具体例子

### 情况一：全部接受

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, d3, d4]
```

则：

```text
accept_length = 4
```

最终至少提交：

```text
[d1, d2, d3, d4]
```

如果还存在 bonus token，则可能额外提交一个 Target token。

---

### 情况二：中间拒绝

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, ...]
```

则：

```text
d1 接受
d2 接受
d3 拒绝
d4 不再检查
```

因此：

```text
accept_length = 2
```

最终结果：

```text
[d1, d2, x3]
```

其中：

```text
x3：
Target 在拒绝位置生成的 token
```

---

### 情况三：第一个 token 就拒绝

```text
Draft：
[d1, d2, d3, d4]

Target：
[x1, ...]
```

则：

```text
accept_length = 0
```

最终结果：

```text
[x1]
```

此时 MTP 这一轮没有贡献任何有效 Draft token，但仍然通过 Target fallback 向前推进了一个 token。

---

## 10. Process 层如何计算接受长度？

当前源码位置：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_process.py:148-169
```

逻辑可以化简为：

```python
accept_length = 0

for draft_token, target_token in zip(
    draft_tokens,
    target_tokens,
):
    if draft_token == target_token:
        accept_length += 1
    else:
        break
```

关键点是：

```text
一旦遇到第一个不匹配，后续 token 全部失效。
```

原因是后续 Draft token 建立在错误 token 的条件上：

```text
d3 是在 d2 正确的前提下生成的
```

如果 `d2` 已经错误，那么：

```text
d3、d4 的上下文条件也不再成立
```

---

## 11. 为什么不能跳过错误 token 继续接受后面 token？

假设：

```text
Draft：
[A, B, C, D]

Target：
[A, X, C, D]
```

虽然 `C` 和 `D` 的 token 可能看起来与 Target 一致，但不能直接接受：

```text
[A, C, D]
```

因为 Draft 生成 `C` 时看到的上下文是：

```text
[A, B]
```

而正确上下文是：

```text
[A, X]
```

因此后续 token 的条件分布不同：

```text
P(C | A, B)
不等于
P(C | A, X)
```

所以投机解码只能接受：

```text
最长连续正确前缀
```

而不能接受离散的正确位置。

---

## 12. Engine 层如何串起 Draft 和 Verify？

主流程可以抽象为：

```python
while not finished:

    draft_tokens = self._draft(
        input_ids,
        state,
    )

    verify_result = self._verify(
        input_ids,
        draft_tokens,
        state,
    )

    accept_length = verify_result.accept_length
    fallback_token = verify_result.fallback_token

    self._commit(
        accept_length,
        fallback_token,
        state,
    )
```

对应当前工程：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:145-172
```

Engine 层不应该负责具体的矩阵计算，它主要负责：

```text
何时调用 Draft
何时调用 Verify
何时提交 Cache
何时判断 EOS
何时进入下一轮
```

---

## 13. Commit 阶段发生什么？

假设：

```text
prefix length = L
draft tokens = [d1, d2, d3, d4]
accept_length = 2
fallback = x3
```

那么提交后的正式序列是：

```text
[prefix, d1, d2, x3]
```

Cache 也要从：

```text
[prefix] + [d1, d2, d3, d4]
```

变成：

```text
[prefix] + [d1, d2, x3]
```

可以表示为：

```text
提交前：
Committed: [prefix]
Tentative: [d1, d2, d3, d4]

提交后：
Committed: [prefix, d1, d2, x3]
Tentative: []
```

---

## 14. `commit_verify_cache()` 的职责

当前源码：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py:387-417
```

它负责把 Verify 结果应用到内部状态。

逻辑上包含：

```text
保留接受 token 的 Cache
丢弃拒绝 token 的 Cache
提交 fallback token 对应状态
更新 current length
更新 MTP 下一轮输入状态
```

可以抽象为：

```python
accepted_end = prefix_len + accept_length

# 只保留 accepted prefix
cache = cache[..., :accepted_end, :]

# 写入 Target fallback
cache = append(
    cache,
    fallback_state
)

current_length = accepted_end + 1
```

真实实现可能使用：

```text
原地 copy
预分配 Cache
指针更新
长度更新
```

而不是重新申请张量。

---

## 15. MTP 与 Target Cache 的关系

需要区分两种 Cache：

```text
Target KV Cache
MTP Draft Cache
```

### Target KV Cache

保存主模型正式上下文：

```text
[prefix, accepted tokens, fallback]
```

### MTP Draft Cache

保存 MTP Draft 的中间状态：

```text
可能包含本轮 d1、d2、d3、d4
```

验证后：

```text
只保留与正式序列一致的部分
```

如果只回滚 Target Cache，却没有回滚 MTP Cache：

```text
下一轮 MTP 会从错误状态继续
```

如果只回滚 MTP Cache，却没有回滚 Target Cache：

```text
Target 和 Draft 的上下文长度会不一致
```

因此提交必须保持：

```text
Target Cache
MTP Cache
input_ids
context length
hidden state
```

的一致性。

---

## 16. MTP 一轮状态转换图

```text
┌──────────────────────┐
│ Committed Prefix     │
│ Target Cache         │
│ MTP Cache            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ MTP Draft Chain      │
│ d1 → d2 → d3 → d4    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Tentative Cache      │
│ d1, d2, d3, d4       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Target Verify        │
│ t1, t2, t3, t4       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ accepted prefix      │
│ + fallback token     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Commit / Rollback    │
└──────────────────────┘
```

---

## 17. MTP 的主要性能瓶颈

MTP 不一定总是更快，原因包括：

### Draft 串行成本

```text
Head 1 → Head 2 → Head 3 → Head 4
```

Head 越多，Draft 链越长。

### Verify 成本

Draft block 越长：

```text
Target 验证计算量越大
```

### Cache 管理成本

每轮都需要：

```text
提交
回滚
复制
更新长度
```

### 接受率不足

如果平均只能接受一个 token：

```text
Draft 计算成本可能无法被摊薄
```

因此重要指标不是：

```text
MTP Head 数量
```

而是：

```text
accepted tokens per round
```

---

## 18. MTP 的性能公式

设：

```text
A = 每轮接受的 Draft token 数
T_draft = MTP Draft 时间
T_verify = Target Verify 时间
T_commit = Cache 提交时间
```

则一轮平均吞吐近似为：

```text
tokens_per_round = A + 1
```

单轮耗时：

```text
T_round =
T_draft
+ T_verify
+ T_commit
```

平均速度：

```text
throughput =
(A + 1) / T_round
```

如果不使用 MTP，Target 每次只生成一个 token：

```text
throughput_baseline =
1 / T_target_step
```

MTP 只有在：

```text
(A + 1) / T_round
>
1 / T_target_step
```

时才真正加速。

---

## 19. 本节核心结论

需要记住：

1. MTP Draft 是一个逐 token 的接力链；
2. 后一个 MTP Head 依赖前一个 Head 的输出；
3. Draft 期间产生的是 tentative state；
4. Target 只接受最长连续正确前缀；
5. 第一个错误位置使用 Target fallback token；
6. `commit_verify_cache()` 同时影响 Target 和 MTP 状态；
7. Cache、hidden state、input length 必须保持一致；
8. MTP 性能取决于：

```text
accepted tokens per round
/
Draft + Verify + Commit 总成本
```

下一部分是：

**MTP 的 `commit_verify_cache()`：详细分析接受、回滚、fallback 和 Cache 长度更新。**