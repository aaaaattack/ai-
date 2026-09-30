# 阶段 1 · 第 3 讲：Acceptance Length 与加速比

前两讲解决了：

```
Draft 如何提出候选？
Target 如何接受或拒绝？
```

这一讲解决：

```
一轮平均能提交多少 token？
为什么有时投机解码反而不加速？
```

------

## 1. Acceptance Length

设一轮 Draft 产生 `K` 个 token：

```
[d1, d2, ..., dK]
```

定义：

```
A = 连续被 Target 接受的 Draft token 数
```

例如：

```
Draft ：[A, B, C, D, E]
Target：[A, B, X, ...]
```

则：

```
A = 2
```

因为 `A、B` 被接受，`C` 在第三个位置被拒绝。

如果本轮有 bonus token，则本轮实际提交：

```
A + 1
```

本例实际提交：

```
[A, B, X]
```

也就是：

```
2 个 accepted draft token + 1 个 Target token
```

------

## 2. Acceptance Rate

Acceptance rate 通常定义为：

```
accepted_draft_tokens
/
proposed_draft_tokens
```

例如运行 100 轮，每轮 Draft 5 个 token：

```
总提出 token = 100 × 5 = 500
总接受 token = 325
```

则：

```
acceptance_rate = 325 / 500 = 65%
```

但平均接受长度是：

```
accepted_per_round = 325 / 100 = 3.25
```

如果每轮都有 bonus token，平均每轮实际输出约为：

```
3.25 + 1 = 4.25 tokens
```

这两个指标必须区分：

```
Acceptance rate：整体比例
Accepted per round：每轮连续接受多少
```

`llama.cpp` 中对应的最小统计代码位于：

```text
llama.cpp/examples/speculative-simple/speculative-simple.cpp:248-298
```

```cpp
const size_t n_draft = draft.size();
auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
n_drafted += n_draft;
n_accept  += ids.size() - 1;
n_predict += ids.size();
```

------

## 3. 从接受率估算接受长度

假设每个 Draft token 被接受的概率都近似为 `α`，并且暂时假设位置之间独立。

要让第一个 token 被接受：

```
P(A >= 1) = α
```

要让前两个 token 都被接受：

```
P(A >= 2) = α²
```

要让前三个都被接受：

```
P(A >= 3) = α³
```

因此：

```
E[A] = α + α² + α³ + ... + αᴷ
```

等价于：

```
E[A] = α(1 - αᴷ) / (1 - α)
```

其中：

- `K`：每轮 Draft token 数
- `α`：单个位置的平均接受概率
- `E[A]`：平均接受 Draft token 数

------

## 4. 示例：Draft 长度为 4，接受率为 0.8

设：

```
K = 4
α = 0.8
```

那么：

```
E[A]
= 0.8 + 0.8² + 0.8³ + 0.8⁴
= 0.8 + 0.64 + 0.512 + 0.4096
= 2.3616
```

所以平均每轮：

```
accepted draft tokens ≈ 2.36
```

如果有 bonus token：

```
平均输出量 ≈ 2.36 + 1 = 3.36 tokens/round
```

这意味着原本需要大约 3.36 次 Target 单 token decode 的工作，现在可能由一轮 Draft/Verify 完成。

------

## 5. 为什么接受率不是越高越好？

接受率很重要，但不是唯一因素。

假设两种方案：

### 方案 A

```
Draft：很快
接受率：60%
```

### 方案 B

```
Draft：很慢
接受率：95%
```

方案 B 看起来接受率更高，但如果 Draft 本身耗时太大，最终可能比方案 A 更慢。

因此真正需要优化的是：

```
每轮产生的 token 数
/
Draft 时间 + Verify 时间 + cache 管理时间
```

而不是单独优化 acceptance rate。

------

## 6. 加速比模型

定义：

```
C_t(1)
```

表示 Target 生成一个 token 的成本。

普通解码每生成一个 token，大致成本是：

```
C_t(1)
```

投机解码一轮的成本包括：

```
C_d(K)：Draft 产生 K 个候选的成本
C_v(K)：Target 验证 K 个候选的成本
C_cache：cache 提交、回滚、同步成本
```

一轮总成本：

```
C_round = C_d(K) + C_v(K) + C_cache
```

一轮平均输出：

```
L_round = 1 + E[A]
```

因此投机解码吞吐近似为：

```
Throughput_spec =
(1 + E[A])
------------------------------
C_d(K) + C_v(K) + C_cache
```

普通解码吞吐近似为：

```
Throughput_normal =
1 / C_t(1)
```

加速比：

```
Speedup =
(1 + E[A]) × C_t(1)
--------------------------------
C_d(K) + C_v(K) + C_cache
```

------

## 7. 一个数值例子

假设：

```
普通 Target 单 token decode：
C_t(1) = 10 ms

Draft 生成 4 个 token：
C_d(4) = 2 ms

Target 验证 4 个 token：
C_v(4) = 12 ms

cache 管理：
C_cache = 1 ms

平均每轮输出：
L_round = 3.36 tokens
```

投机一轮成本：

```
C_round = 2 + 12 + 1 = 15 ms
```

投机吞吐：

```
3.36 / 15 ≈ 0.224 tokens/ms
```

普通吞吐：

```
1 / 10 = 0.1 tokens/ms
```

加速比：

```
0.224 / 0.1 ≈ 2.24x
```

------

## 8. 为什么 Verify 不是完全免费？

很多人会误以为：

```
Target 一次验证 K 个 token
```

就等于只付一次单 token decode 成本。

实际并不是。

Verify 需要：

- 计算多个位置的 logits
- 写入多个 KV cache 位置
- 处理 attention mask
- 处理 recurrent/conv state
- 最后进行 cache 回滚或提交

因此：

```
C_v(K) > C_t(1)
```

但通常：

```
C_v(K) < K × C_t(1)
```

投机解码能否加速，取决于这个差值是否足够大。

------

## 9. Draft 长度 K 的权衡

增大 `K` 有两个相反影响。

### 优点

```
可能一次提交更多 token
```

### 缺点

```
Draft 计算增加
Verify 计算增加
后面位置的接受概率通常下降
拒绝 token 可能浪费
```

因此不能无限增加 Draft 长度。

典型现象：

```
K 太小：
    Draft 不够积极，收益有限

K 太大：
    后部 token 接受率下降，计算浪费

K 合适：
    接受长度和 Draft/Verify 成本达到平衡
```

这就是实际系统需要调节：

```
num_draft_tokens
verify_length
```

的原因。

------

## 10. 对当前 DFlash 工程的对应

当前 DFlash 默认：

```
num_draft_tokens = 7
verify_length = 8
```

工程统计：

```
state.proposed_tokens += len(draft)
state.accepted_tokens += emitted
state.accepted_draft_tokens += accepted
```

最终指标包括：

```
draft_tokens
accepted_draft_tokens
speculative_rounds
acceptance_rate
accepted_per_round
verify_tokens
```

其中：

```
acceptance_rate
=
accepted_draft_tokens / draft_tokens
```

而：

```
accepted_per_round
=
accepted_draft_tokens / speculative_rounds
```

注意：

```
accepted_per_round
```

不包含 bonus token；实际每轮提交量还要再加上 Target token。

------

## 11. 本节结论

1. `Acceptance Length` 决定每轮实际能提交多少 token。
2. `Acceptance Rate` 和 `Accepted Per Round` 不是同一个指标。
3. 接受率越高通常越好，但 Draft 成本也必须足够低。
4. Verify 虽然一次处理多个 token，但并不是零成本。
5. Draft 长度存在最优值，不能无限增大。
6. 真正的优化目标是：

```
每轮提交 token 数
/
每轮 Draft + Verify + cache 总耗时
```

下一步可以进入阶段 1 的最后一部分：把上述公式整理成一个小型 CPU 模拟器，用代码验证 acceptance length、bonus token 和加速比。

## 关联工程的最简实现：llama.cpp

`llama.cpp` 用返回 token 数量直接统计接受长度：

```cpp
// examples/speculative-simple/speculative-simple.cpp:248-258
const size_t n_draft = draft.size();
auto ids = common_sampler_sample_and_accept_n(
    smpl.get(), ctx_tgt, draft
);
// ids 至少包含一个 Target token；ids.size()-1 是接受的 Draft 数。
```

提交和统计逻辑：

```cpp
// examples/speculative-simple/speculative-simple.cpp:292-298
common_speculative_accept(spec, seq_id, ids.size() - 1);
n_past    += ids.size() - 1;
n_drafted += n_draft;
n_accept  += ids.size() - 1;
n_predict += ids.size();
```

这里的 `ids.size()` 对应本讲的：

```text
accepted draft tokens + 1 个 Target token
```
