
## 阶段 5 · 第 3 讲：MTP 的 `commit_verify_cache()`

本节是 MTP 工程实现的关键。重点不是“如何生成 token”，而是：

> Target 验证之后，哪些状态可以正式提交，哪些状态必须回滚？

对应源码：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py:387-417
```

关联调度代码：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:145-172
```

---

## 1. 为什么需要 Commit？

MTP Draft 阶段会提前生成多个 token：

```text
d1, d2, d3, d4
```

同时，MTP 可能已经计算并写入了这些 token 对应的状态：

```text
MTP hidden state
MTP KV Cache
Target 临时 Cache
current length
```

但在 Target 验证之前，这些 token 都只是“候选”：

```text
[prefix] + [d1, d2, d3, d4]
                         ↑
                    尚未确认
```

Target 验证后可能得到：

```text
d1 正确
d2 正确
d3 错误
d4 不再有效
```

因此最终正式序列是：

```text
[prefix, d1, d2, x3]
```

其中：

```text
d1、d2：
接受的 Draft token

x3：
Target fallback token

d3、d4：
回滚
```

Commit 的任务就是把临时状态转换成正式状态。

---

## 2. 三类 Cache 状态

理解 `commit_verify_cache()`，首先要区分三类状态。

### 已提交状态

```text
Committed State
```

它对应正式序列：

```text
[prefix, d1, d2, x3]
```

下一轮 Target 和 MTP 都必须从这里继续。

### 验证临时状态

```text
Verify/Tentative State
```

它对应本轮 Draft：

```text
[d1, d2, d3, d4]
```

这些状态还没有被确认。

### 回滚边界

```text
rollback boundary
```

它表示：

```text
从哪个位置开始，后面的状态都无效
```

例如：

```text
prefix_len = 100
accept_length = 2
```

那么：

```text
d1 → 位置 100
d2 → 位置 101
d3 → 位置 102
d4 → 位置 103
```

第一个拒绝位置是：

```text
position = 100 + 2 = 102
```

因此提交前缀的边界是：

```text
[0, 102)
```

fallback token 位于：

```text
position 102
```

---

## 3. `accept_length` 的含义

假设：

```text
draft_tokens = [d1, d2, d3, d4]
target_tokens = [d1, d2, x3, ...]
```

则：

```text
accept_length = 2
```

它表示：

```text
有 2 个 Draft token 被接受
```

不是：

```text
最终新增 token 数量
```

最终新增数量通常是：

```text
accept_length + 1
```

因为第一个错误位置还要加入 Target fallback：

```text
2 个 accepted Draft token
+ 1 个 Target fallback token
= 3 个新 token
```

这是实现中最容易发生 off-by-one 的地方。

---

## 4. 提交位置计算

设：

```text
old_length = L
accept_length = A
```

那么：

```text
accepted Draft 区间：
[L, L + A)
```

fallback 位置：

```text
L + A
```

提交后的新长度：

```text
new_length = L + A + 1
```

例如：

```text
old_length = 100
accept_length = 2
```

则：

```text
accepted Draft：
[100, 101]

fallback：
102

new_length：
103
```

对应序列：

```text
position 100 → d1
position 101 → d2
position 102 → x3
```

---

## 5. Commit 的逻辑化简

下面是 `commit_verify_cache()` 的逻辑化简，不是原文件逐字代码：

```python
def commit_verify_cache(
    old_length,
    accept_length,
    fallback_token,
):
    accepted_end = old_length + accept_length

    # 保留 accepted Draft 的状态
    keep_cache_until(accepted_end)

    # 写入 Target fallback 的状态
    append_fallback_state(fallback_token)

    # 更新正式长度
    current_length = accepted_end + 1

    return current_length
```

对应真实工程：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py:387-417
```

真实实现可能使用预分配张量和原地复制，而不是上述形式的切片操作。

---

## 6. 为什么不能简单清空整个 Draft Cache？

假设：

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, ...]
```

如果验证后直接清空整个 MTP Cache：

```text
MTP Cache = empty
```

虽然逻辑上不会错，但会浪费：

```text
d1、d2 对应的已确认状态
```

下一轮可能需要重新计算这些状态。

更合理的做法是：

```text
保留 d1、d2
删除 d3、d4
加入 x3
```

即：

```text
旧状态：
[prefix, d1, d2, d3, d4]

提交后：
[prefix, d1, d2, x3]
```

这就是“部分提交”。

---

## 7. Cache 回滚的两种方式

### 方式一：逻辑长度回滚

Cache 的物理张量不一定真的删除，只更新有效长度：

```python
current_length = old_length + accept_length + 1
```

后面的数据虽然仍在显存中：

```text
[d3, d4]
```

但由于超过 `current_length`，下一轮不会读取。

优点：

```text
不需要搬运大量数据
实现简单
速度快
```

### 方式二：物理复制或重排

如果 Cache 不是简单的连续前缀，或者需要处理分支状态，就需要实际复制：

```python
cache[:new_length] = selected_cache[:new_length]
```

或者：

```python
destination.copy_(source)
```

优点：

```text
Cache 内容更加干净
支持复杂状态重排
```

代价：

```text
需要额外显存带宽
```

当前工程的实现重点是：

```text
预分配 Cache
根据接受长度更新有效范围
必要时进行状态复制
```

---

## 8. Target Cache 与 MTP Cache 必须同步

一轮提交至少涉及：

```text
input_ids
Target KV Cache
MTP KV Cache
MTP hidden state
current length
```

它们必须满足：

```text
序列长度一致
上下文内容一致
下一个计算位置一致
```

错误示例：

```text
input_ids 长度 = 103
Target Cache 长度 = 103
MTP Cache 长度 = 105
```

这意味着：

```text
MTP 认为 d3、d4 已经属于上下文
Target 却认为它们不存在
```

下一轮会发生状态错位：

```text
Draft 与 Target 看到的上下文不同
```

因此提交逻辑不能只更新 token 序列，还要同步更新所有状态。

---

## 9. Fallback token 如何进入 Cache？

假设：

```text
Draft：
[d1, d2, d3]

Target：
[d1, d2, x3]
```

Target 在位置 3 产生：

```text
x3
```

最终序列：

```text
[prefix, d1, d2, x3]
```

注意：

```text
x3 不是下一轮 Draft 才生成的普通 token
```

它是当前 Verify 阶段已经由 Target 计算出来的结果，因此可以直接作为下一轮起点。

状态流程：

```text
Target logits at rejected position
        ↓
sample/argmax x3
        ↓
写入 input_ids
        ↓
提交 x3 对应的状态
        ↓
下一轮从 x3 继续
```

如果忽略 fallback：

```text
[prefix, d1, d2]
```

下一轮会重复计算第一个错误位置，造成额外开销。

---

## 10. 全部接受时怎么办？

假设：

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

这里需要特别注意 bonus token。

有些实现会让 Target 同时计算 Draft block 之后的下一个 token：

```text
x5
```

那么最终可能提交：

```text
[d1, d2, d3, d4, x5]
```

有些实现只提交：

```text
[d1, d2, d3, d4]
```

下一轮再生成：

```text
x5
```

所以必须区分：

```text
accepted Draft tokens
bonus token
fallback token
```

它们在数学含义上相近，但在具体工程接口中可能不同。

---

## 11. 三种验证结果

### 情况一：部分接受

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, ...]
```

提交：

```text
[d1, d2, x3]
```

Cache：

```text
保留 d1、d2
加入 x3
删除 d3、d4
```

### 情况二：全部接受

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, d3, d4]
```

提交：

```text
[d1, d2, d3, d4]
```

或者：

```text
[d1, d2, d3, d4, bonus]
```

取决于接口设计。

### 情况三：全部拒绝

```text
Draft：
[d1, d2, d3, d4]

Target：
[x1, ...]
```

提交：

```text
[x1]
```

Cache：

```text
丢弃全部 Draft 状态
只保留 Target fallback
```

---

## 12. 为什么 MTP 状态比普通 KV Cache 更复杂？

普通独立 Draft Model 通常主要需要：

```text
Draft KV Cache
Target KV Cache
```

而 MTP 可能还需要：

```text
MTP hidden state
每个 Head 的中间状态
head-specific Cache
context length
verify length
draft length
```

因为 MTP Head 之间可能存在接力关系：

```text
Head 1 状态
    ↓
Head 2 状态
    ↓
Head 3 状态
```

如果第 2 个 token 被拒绝：

```text
Head 2 的状态可能保留
Head 3 及之后的状态必须回滚
```

因此回滚边界不仅作用于 token，还作用于 Head Chain。

---

## 13. MTP Cache 提交流程

可以抽象为：

```text
1. 读取 accept_length
2. 确定第一个拒绝位置
3. 截断 Draft 临时状态
4. 保留 accepted prefix
5. 加入 fallback/bonus token
6. 更新 input_ids
7. 更新 Target Cache 长度
8. 更新 MTP Cache 长度
9. 更新下一轮 hidden state
10. 清理临时 verify 状态
```

伪代码：

```python
def commit_round(state, verify_result):
    old_len = state.context_length
    accepted = verify_result.accept_length
    fallback = verify_result.fallback_token

    # 1. accepted prefix
    accepted_end = old_len + accepted

    # 2. 回滚 Draft 临时状态
    state.mtp_cache.truncate(accepted_end)

    # 3. 提交 fallback
    state.input_ids.append(fallback)
    state.target_cache.append(fallback)
    state.mtp_cache.append(fallback)

    # 4. 更新有效长度
    state.context_length = accepted_end + 1

    # 5. 清理本轮临时变量
    state.clear_verify_buffer()
```

---

## 14. Engine 为什么必须等待 Commit 完成？

错误顺序：

```text
Draft 下一轮
    ↓
上一轮 Cache 还没提交
```

这会造成：

```text
下一轮 Draft 读到错误状态
```

正确顺序：

```text
Draft
  ↓
Verify
  ↓
Compute accept_length
  ↓
Commit/Rollback
  ↓
Update context length
  ↓
下一轮 Draft
```

因此 Engine 层的调度必须保证：

```text
Commit 完成后，才能开始下一轮 Draft
```

相关流程位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:145-172
```

---

## 15. 一轮完整数值例子

假设：

```text
旧序列长度：
L = 10

Draft：
[d1, d2, d3, d4]

Target：
[d1, d2, x3, ...]
```

### Draft 前

```text
正式序列长度 = 10
Target Cache 有效长度 = 10
MTP Cache 有效长度 = 10
```

### Draft 后

```text
临时 Draft 区间：
位置 10 → d1
位置 11 → d2
位置 12 → d3
位置 13 → d4
```

### Verify 后

```text
accept_length = 2
fallback = x3
```

### Commit 后

```text
位置 10 → d1
位置 11 → d2
位置 12 → x3
```

新长度：

```text
new_length = 10 + 2 + 1 = 13
```

最终：

```text
Target Cache 有效长度 = 13
MTP Cache 有效长度 = 13
input_ids 长度 = 13
```

位置 13 之后的内容：

```text
无效
```

---

## 16. 与 DFlash 状态提交的区别

MTP 的提交重点：

```text
MTP Head Chain
MTP hidden state
MTP KV Cache
```

DFlash 的提交重点：

```text
Draft context
Target KV Cache
conv state
recurrent state
noise padding
block rollback
```

MTP 更像：

```text
线性 Draft Chain 的前缀提交
```

DFlash 更像：

```text
复杂 Block State 的统一提交与回滚
```

但两者遵循同一原则：

```text
只提交 Target 已确认的状态
拒绝状态不能泄漏到下一轮
```

---

## 17. 本节核心结论

需要记住：

1. `accept_length` 表示接受的 Draft token 数量；
2. fallback 位于第一个拒绝位置；
3. 新序列长度通常是：

```text
old_length + accept_length + 1
```

4. Commit 必须同步更新：

```text
input_ids
Target Cache
MTP Cache
hidden state
context length
```

5. 拒绝 token 的状态必须回滚或变为无效；
6. 全部接受时要注意 bonus token 语义；
7. `commit_verify_cache()` 是保证 Draft 与 Target 状态一致的核心；
8. 下一轮 Draft 必须在 Commit 完成后开始。

下一部分是：

**原生 MTP 与外部 Draft Model 的工程区别，以及为什么 MTP 不等于普通 Speculative Decoding。**