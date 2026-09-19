# 第 06 课｜L2.1：Transformer、Q/K/V 与 Attention Tensor Shape

## 一、学习目标

完成本课后，你应能：

1. 说明 Transformer Attention 中 Q、K、V 的作用；
2. 使用 `B、S、H、A、D` 表示常见 Tensor shape；
3. 推导 Q/K/V、attention score 和 attention output 的 shape；
4. 理解为什么 Attention 的 score 矩阵会随序列长度平方增长；
5. 为后续 MHA、MQA、GQA、KV Cache 和长上下文优化打基础。

---

## 二、五个必须记住的符号

| 符号 | 含义 | 示例 |
|---|---|---|
| B | batch size，一次处理多少条序列 | 2 |
| S | sequence length，序列中 token 数 | 128 |
| H | hidden size，总隐藏维度 | 768 |
| A | attention head 数 | 12 |
| D | 每个 head 的维度 | 64 |

通常：

```text
H = A × D
```

例如 `768 = 12 × 64`。

---

## 三、Q、K、V 是什么

输入 hidden state 的 shape 通常是：

```text
X: [B, S, H]
```

经过三组线性投影得到：

```text
Q = X × Wq
K = X × Wk
V = X × Wv
```

投影后初始 shape 仍可写成：

```text
Q, K, V: [B, S, H]
```

再拆成多个 attention head：

```text
[B, S, H]
→ [B, S, A, D]
→ transpose
→ [B, A, S, D]
```

直观上：

- Q（Query）表示当前 token 想查询什么；
- K（Key）表示每个 token 可被匹配的特征；
- V（Value）表示被关注后实际取回的信息。

---

## 四、Attention 的 shape 推导

每个 head 内计算：

```text
scores = Q × Kᵀ / sqrt(D)
```

Q 是 `[B, A, S, D]`，K 转置后最后两维是 `[D, S]`，因此：

```text
scores: [B, A, S, S]
```

最后两个 `S` 的含义分别是：

```text
query token 位置 × key token 位置
```

对 score 做 softmax 后，shape 不变：

```text
P = softmax(scores): [B, A, S, S]
```

再与 V 相乘：

```text
output = P × V
[B, A, S, S] × [B, A, S, D]
→ [B, A, S, D]
```

将 head 合并后：

```text
[B, A, S, D]
→ transpose → [B, S, A, D]
→ reshape → [B, S, H]
```

---

## 五、为什么长序列 Attention 很贵

score 的 shape 是 `[B, A, S, S]`。当序列长度从 `S` 变成 `2S` 时，score 元素数量从 `S²` 变为 `(2S)²=4S²`。

这也是长上下文训练和 Prefill 阶段的主要计算/显存挑战之一。后续会学习 KV Cache、GQA 和 PagedAttention 如何缓解其中一部分成本。

---

## 六、带做例题

设：

```text
B = 2
S = 128
H = 768
A = 12
D = 64
```

则：

```text
X:      [2, 128, 768]
Q/K/V:  [2, 128, 768]
split:  [2, 128, 12, 64]
Q/K/V:  [2, 12, 128, 64]
scores: [2, 12, 128, 128]
output: [2, 12, 128, 64]
merge:  [2, 128, 768]
```

---

## 七、常见 shape 错误

| 现象 | 常见原因 |
|---|---|
| `H` 无法拆为 `A×D` | head 数与 head dim 配置不匹配 |
| Q 与 K 矩阵乘法报错 | 转置维度错误，最后两维不是 `[S,D] × [D,S]` |
| attention mask 广播失败 | mask 的 `[B,1,S,S]` 与 score 的 `[B,A,S,S]` 不兼容 |
| 显存突然增大 | score `[B,A,S,S]` 随 S 平方增长 |

---

## 八、本课练习

### 第 1 题｜维度关系

若 `H=1024`、`A=16`，每个 head 的维度 `D` 是多少？

**我的回答：**


### 第 2 题｜Q/K/V shape

`B=4`、`S=64`、`H=512`、`A=8`。拆分并转置后，Q 的 shape 是什么？

**我的回答：**


### 第 3 题｜score shape

沿用第 2 题，attention score 的 shape 是什么？

**我的回答：**


### 第 4 题｜长序列

若 S 从 256 增至 512，score 矩阵元素数量变为原来的几倍？为什么？

**我的回答：**


## 九、通过标准

- 能从 `H=A×D` 推导 head dim；
- 能写出 Q/K/V 和 score 的 shape；
- 能解释 score 随 S² 增长的原因。

完成后先做章节评估；未达标时只在本文追加夯实内容。下一课学习 MHA、MQA、GQA 与 KV Cache。

---

## 十、代码结合练习（不运行模型）

本节的代码路线和后续工具映射见 `AI_NPU代码结合学习地图.md`。完成前面四道 shape 题后，再用下面的观察卡把公式连接到实现；只需阅读，不修改或执行任何外部项目。

阅读顺序：

1. CS336 笔记：`E:\cs336_note_and_hw-main\cs336_note_and_hw-main\课后笔记\05_Attention.md`；
2. nano-vLLM：`E:\cs336_note_and_hw-main\nano-vllm\nanovllm\models\qwen3.py` 与 `layers\attention.py`；
3. ModelZoo：`E:\cs336_note_and_hw-main\houmo-examples-xh2\apis\inferences\qwen3\README.MD`。

### L2.1 代码观察卡

```markdown
- QKV 在 nano-vLLM 中的切分位置：
- RoPE 应用于：
- Prefill 与 decode 的 Attention 路径分别是：
- `scores [B,A,S,S]` 在工程实现中为什么不显式长期保存在 HBM：
```

提示：先找 `qkv.split(...)`、RoPE 调用和 `context.is_prefill` 分支。最后一题的关键词是 Flash Attention 的分块计算与避免把完整 score 矩阵写回高带宽内存。
