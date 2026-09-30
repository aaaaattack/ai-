
## 阶段 6 · 第 1 讲：DFlash 总体架构——Block Diffusion Draft

本节开始学习 DFlash。DFlash 的原论文是：

[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

论文的核心目标是解决：

```text
EAGLE / MTP：
Draft token 仍然逐个生成，存在串行瓶颈

DFlash：
一次性生成一个 Draft token block
```

论文提出的核心方法是轻量级 Block Diffusion Draft，并利用 Target Model 的 hidden feature 提高候选质量。论文摘要明确强调：DFlash 在单次 forward 中生成 Draft block，同时使用 Target 的上下文特征进行条件化。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

---

## 1. 为什么还需要 DFlash？

前面学习的 Draft 方法大致分为三类。

### 经典 Draft Model

```text
Draft Model：
d1 → d2 → d3 → d4
```

每个 token 都依赖前一个 token。

### MTP

```text
MTP Head 1 → d1
MTP Head 2 → d2
MTP Head 3 → d3
```

虽然每个 Head 较轻，但通常仍存在接力依赖。

### EAGLE

```text
Target hidden feature
        ↓
EAGLE Draft Network
        ↓
逐步预测 feature
        ↓
生成候选树
```

EAGLE 的候选树可以提升接受率，但 Draft Network 本身仍有自回归过程。

论文指出，现有方法虽然可以让 Target 并行验证多个 token，但 Draft 阶段通常仍然是自回归的，因此实际加速会受到串行 Draft 延迟限制。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

---

## 2. DFlash 的核心思想

DFlash 把 Draft 过程从：

```text
逐 token 生成
```

改成：

```text
逐 block 生成
```

例如 block size 为 5：

```text
anchor, mask, mask, mask, mask
```

Draft Model 一次 forward 后直接产生：

```text
anchor, d1, d2, d3, d4
```

可以抽象为：

```text
Target Context Feature
        +
Anchor Token
        +
Masked Draft Block
        ↓
DFlash Draft Model
        ↓
[d1, d2, d3, d4]
```

与 MTP 的区别：

```text
MTP：
d1 → d2 → d3 → d4

DFlash：
[d1, d2, d3, d4] 一次并行预测
```

---

## 3. 什么是 Block Diffusion？

普通自回归模型的 Attention 结构是：

```text
第 1 个位置只能看历史
第 2 个位置只能看历史和第 1 个位置
第 3 个位置只能看历史和前两个位置
```

即：

```text
causal attention
```

DFlash 的 Draft block 内部则允许多个 masked 位置共同参与计算：

```text
d1 可以看 block context
d2 可以看 block context
d3 可以看 block context
d4 可以看 block context
```

也就是说，Draft block 内部可以使用非因果或 block-level attention：

```text
block 内部并行交互
block 外部仍然受到 anchor 和上下文约束
```

这使得整个 block 可以在一次 forward 中生成。

需要注意：

```text
DFlash 的 Draft Model 可以非因果，
但 Target Verify 仍然按照 Target 的自回归分布进行验证。
```

因此 DFlash 的并行只发生在：

```text
Draft 预测阶段
```

而不是改变 Target Model 的自回归语义。

---

## 4. Anchor 是什么？

DFlash 的 block 通常包含一个 anchor token。

假设当前正式序列是：

```text
[prefix]
```

上一轮 Target 已经确认的最后一个 token 是：

```text
a
```

DFlash 构造：

```text
[a, MASK, MASK, MASK, MASK]
```

其中：

```text
a：
真实的、已经由 Target 确认的 anchor token

MASK：
待预测位置
```

Draft Model 的目标是：

```text
预测 anchor 后面的 token
```

输出：

```text
[a, d1, d2, d3, d4]
```

如果 anchor 本身只是上下文，而不是 Draft token，那么真正参与 speculative verify 的是：

```text
[d1, d2, d3, d4]
```

当前 `speculators` 实现明确区分了这种语义：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\config.py:73-80
```

逻辑是：

```text
sample_from_anchor = False：
anchor 是 bonus token，只预测 block_size - 1 个 speculative token

sample_from_anchor = True：
anchor 也参与采样，总共预测 block_size 个 speculative token
```

这就是为什么工程中经常出现：

```text
block_size
```

和：

```text
speculative_tokens = block_size - 1
```

之间的差异。

---

## 5. 为什么必须使用 Anchor？

如果整个 block 都是随机 mask：

```text
[MASK, MASK, MASK, MASK]
```

Draft Model 缺少一个明确的自回归起点。

而 anchor 提供：

```text
当前正式序列的最后状态
```

因此 Draft Model 实际解决的是：

```text
给定已经确认的 token a，
同时预测 a 后面的多个 token。
```

数学上可以写成：

```text
p(d_1, d_2, ..., d_k | prefix, a)
```

而不是：

```text
p(d_1, d_2, ..., d_k)
```

Anchor 的作用类似于：

```text
已知边界条件
```

它让 block diffusion draft 与当前真实生成上下文连接起来。

---

## 6. DFlash 的“Target Knows Best”

DFlash 原论文中一个重要观点是：

> Target Model 已经知道很多未来信息。

Target 在处理上下文时，内部 hidden feature 已经包含：

```text
语义信息
句法信息
任务信息
未来 token 的预测信息
```

因此 DFlash 不希望一个小 Draft Model 完全从 token 重新理解上下文，而是把 Target 的 hidden feature 提供给 Draft。

论文将这种思想概括为：

```text
Target knows best
```

论文说明 DFlash 使用 Target 提取的上下文特征作为条件，从而提高 Draft 质量和 acceptance rate。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

---

## 7. Target Hidden Feature 如何进入 DFlash？

DFlash Draft Model 一般不是只接收 token embedding，还会接收 Target 的辅助 hidden states：

```text
Target layer 1 hidden
Target layer 2 hidden
Target layer 3 hidden
...
        ↓
拼接或投影
        ↓
DFlash Draft layers
```

当前 `speculators` 工程的配置中：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\config.py:51-58
```

使用：

```python
aux_hidden_state_layer_ids
```

指定从 Target 的哪些层提取辅助 hidden state。

DFlash Draft Model 会通过一个线性投影将多层 Target feature 映射到 Draft hidden dimension。

源码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:116-124
```

核心结构可以抽象为：

```python
target_features = concat(
    target_hidden_layer_1,
    target_hidden_layer_2,
    target_hidden_layer_3,
)

draft_hidden = fc(
    target_features
)
```

这与 EAGLE 类似，都使用 Target feature；但 DFlash 的目标不是逐步预测下一 hidden state，而是条件化整个 Draft block。

---

## 8. DFlash 与 EAGLE 的关键区别

### EAGLE

```text
h_t
  ↓
预测 ĥ_{t+1}
  ↓
预测 d1
  ↓
使用 ĥ_{t+1}, d1
  ↓
预测 ĥ_{t+2}
  ↓
预测 d2
```

特点：

```text
feature autoregression
```

### DFlash

```text
Target hidden features
        +
anchor
        +
masked block
        ↓
一次 forward
        ↓
[d1, d2, d3, d4]
```

特点：

```text
block-level parallel drafting
```

可以总结为：

```text
EAGLE：
预测一条更聪明的 Draft 链

DFlash：
一次预测整段 Draft block
```

---

## 9. DFlash 的训练方式

DFlash 的训练必须模拟推理时的输入形式。

论文描述的训练过程是：

1. 使用 Target Model 处理干净序列；
2. 从 response 中随机选择 anchor；
3. 将 anchor 后面的 block 位置 mask 掉；
4. 注入 Target hidden feature；
5. 让 DFlash Draft Model 并行预测 masked token。

论文中这种训练方式直接对应推理时的 block 构造：anchor 来自上一轮 Target 已确认的 token，后续位置需要由 Draft 预测。[DFlash 原论文](https://arxiv.org/abs/2602.06036)

训练样本可以表示为：

```text
原始序列：

[prefix, a, x1, x2, x3, x4]
```

构造 Draft 输入：

```text
[anchor=a, MASK, MASK, MASK, MASK]
```

监督目标：

```text
[x1, x2, x3, x4]
```

训练目标：

```text
L_DFlash
=
CE(x1, p1)
+
CE(x2, p2)
+
CE(x3, p3)
+
CE(x4, p4)
```

关键是：

```text
四个位置可以同时计算 loss
```

不需要像 MTP 那样严格按照：

```text
x1 → x2 → x3 → x4
```

串行训练。

---

## 10. DFlash 的 Attention Mask

DFlash 的 block 内部需要特殊 Attention Mask。

假设：

```text
base sequence：
[prefix, anchor]

draft block：
[d1, d2, d3, d4]
```

Draft block 中每个位置应该能够看到：

```text
prefix
anchor
自己的 Draft block
```

但不同 anchor 的 block 之间不能互相看到。

当前实现中的 Attention Mask 生成函数：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\attention.py:7-100
```

该函数名为：

```python
create_anchor_block_mask_mod(...)
```

其设计逻辑是：

```text
每个 query block 绑定一个 anchor
```

对于属于第 `j` 个 block 的 query：

```text
可以看到：
1. 同一文档中 anchor 之前的 base token
2. 自己所属的 synthetic block

不能看到：
1. 其他 synthetic block
2. anchor 之后的 base token
3. 其他文档的 token
```

抽象成：

```text
Query block j
    ├── base prefix before anchor_j
    └── synthetic block j
```

而不是：

```text
Query block j
    └── all blocks
```

---

## 11. 为什么不同 Block 之间不能互相看？

假设训练时有两个 anchor：

```text
anchor_1 = position 10
anchor_2 = position 20
```

对应两个 Draft block：

```text
block_1 = [d11, d12, d13]
block_2 = [d21, d22, d23]
```

如果 `block_1` 能看到 `block_2`：

```text
d11 → d21
```

那么训练时就会产生错误的信息泄漏：

```text
block_1 使用了未来 block_2 的信息
```

推理时不存在这种信息，因此会产生 train/inference mismatch。

所以 Attention Mask 必须保证：

```text
block_1 只看 block_1
block_2 只看 block_2
```

同时二者都可以看到自己的 anchor 之前的上下文。

---

## 12. DFlash 的 block size

假设：

```text
block_size = 8
```

如果 anchor 不参与 speculative sampling：

```text
anchor + 7 个 masked positions
```

真正的 Draft token 数量为：

```text
7
```

如果 anchor 也参与 sampling：

```text
8 个 token 都可以作为 speculative token
```

当前实现中的配置：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\config.py:44-48
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash\core.py:256-259
```

逻辑化简：

```python
if sample_from_anchor:
    speculative_tokens = block_size
else:
    speculative_tokens = block_size - 1
```

这会直接影响：

```text
每轮最多 Draft token 数量
Verify block 宽度
接受长度统计
bonus token 语义
```

---

## 13. 当前 Qwen DFlash 的工程主链

当前 Qwen 工程的 DFlash 主循环位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_engine.py:176-191
```

可以抽象为：

```python
while not finished:

    draft_state = select_draft(...)
    draft_output = draft_decode(...)

    verify_output = verify(...)

    commit_or_rollback(...)
```

各文件职责：

### Engine

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_engine.py:176-191
```

负责：

```text
组织一轮 Draft → Verify → Commit
```

### Process

Draft 选择：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_process.py:204-226
```

Verify 接受：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_process.py:228-245
```

负责：

```text
候选 token 选择
Target token 对比
accept_length 计算
fallback 处理
```

### Module

Draft 计算：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:436-455
```

Verify 计算：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:473-501
```

状态回滚与提交：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:503-540
```

---

## 14. 一轮 DFlash 推理示例

假设当前已确认序列为：

```text
[prefix, a]
```

其中：

```text
a：
上一轮 Target 生成的 bonus token
```

DFlash 构造：

```text
[a, MASK, MASK, MASK, MASK]
```

Draft 一次 forward：

```text
[a, d1, d2, d3, d4]
```

Target Verify：

```text
Target：
[d1, d2, x3, ...]
```

则：

```text
d1 接受
d2 接受
d3 拒绝
d4 无效
```

最终提交：

```text
[prefix, a, d1, d2, x3]
```

下一轮新的 anchor 是：

```text
x3
```

下一轮 DFlash 输入：

```text
[x3, MASK, MASK, MASK, MASK]
```

因此 DFlash 的 anchor 会随着 Target 验证结果不断向前移动：

```text
a → x3 → x_next → ...
```

---

## 15. 为什么称为“固定 Verify Block”？

DFlash 每轮通常使用固定的 block size：

```text
block_size = K
```

因此 Target 每轮验证固定宽度的 Draft block：

```text
[d1, d2, ..., dK]
```

这对 Runtime 很重要，因为固定宽度更容易：

```text
预分配 Tensor
固定 CUDA Graph shape
编译 Attention Kernel
减少动态 shape 开销
```

但接受长度仍然是动态的：

```text
accept_length ∈ [0, K]
```

因此：

```text
Verify 的计算形状可以固定
Commit 的有效长度动态变化
```

这也是工程实现中常见的：

```text
固定物理 block
动态逻辑 prefix
```

---

## 16. DFlash 为什么比 MTP 更适合并行 Draft？

MTP：

```text
d1 → d2 → d3 → d4
```

DFlash：

```text
[d1, d2, d3, d4]
```

MTP 的计算依赖链长度为：

```text
O(K)
```

DFlash 的 Draft block 可以近似一次完成：

```text
O(1) 次 Draft forward
```

这里的 `O(1)` 指：

```text
相对于 block 内 token 数量，不再逐 token 调用完整 Draft step
```

但实际计算量仍然会随着：

```text
block size
draft layers
attention width
```

增加。

所以更准确的说法是：

```text
DFlash 将 token-level serial dependency
转换成 block-level parallel computation。
```

---

## 17. DFlash 的速度来源

一轮速度大致为：

```text
T_round
=
T_dflash_draft
+
T_target_verify
+
T_state_commit
```

新增 token 数量约为：

```text
accepted_length + 1
```

平均吞吐：

```text
throughput
≈
(accepted_length + 1)
/
T_round
```

DFlash 的关键是同时优化：

```text
T_dflash_draft 尽量低
accepted_length 尽量高
```

如果只追求并行，而 Draft 质量下降：

```text
T_draft 下降
acceptance length 也下降
```

最终未必加速。

因此 DFlash 必须结合：

```text
Block Diffusion
+
Target Hidden Conditioning
+
高效 Verify
```

---

## 18. DFlash 的损失无关性来自哪里？

DFlash Draft 本身只是一个近似模型：

```text
DFlash token 不一定等于 Target token
```

它的并行生成不会自动保证正确性。

正确性来自：

```text
Target Verify
```

在 greedy 模式下：

```text
只有和 Target 预测一致的连续 prefix 才会被提交
```

在 sampling 模式下：

```text
需要使用正确的 rejection sampling 规则
```

因此：

```text
DFlash 的并行 Draft：
影响速度

Target Verify：
决定正确性

Rejection Sampling：
决定采样分布是否严格保持
```

这和前面学习的 MTP、EAGLE 完全一致。

---

## 19. DFlash 与 EAGLE、MTP 的最终对比

| 方法 | Draft 方式 | 是否逐 token | 是否使用 Target Feature | 候选结构 |
|---|---|---:|---:|---|
| MTP | Head Chain | 通常是 | 是 | 线性 block |
| EAGLE | Feature autoregression | 是 | 是 | Tree |
| Medusa | 多头并行 | 否 | 是 | Tree |
| DFlash | Block Diffusion | 否，按 block | 是 | 固定 block |

核心差异：

```text
MTP：
深度方向扩展

Medusa：
宽度方向扩展

EAGLE：
Feature 方向扩展

DFlash：
Block 并行方向扩展
```

---

## 20. 本节核心结论

需要记住：

1. DFlash 用 Block Diffusion 替代逐 token Draft；
2. 一个 block 通常由 anchor 和多个 masked position 构成；
3. Draft Model 在一次 forward 中并行预测整个 block；
4. Target hidden feature 为 DFlash 提供高质量上下文；
5. Attention Mask 保证每个 block 只能看到自己的 anchor 和合法上下文；
6. `block_size` 是物理验证宽度，`accept_length` 是动态有效长度；
7. DFlash 的正确性仍然来自 Target Verify；
8. DFlash 的加速来自：

```text
并行 Draft
+
高质量 Target Feature Conditioning
+
固定形状 Verify
```

9. 当前工程的主链是：

```text
qwen_dflash_engine.py
    ↓
qwen_dflash_process.py
    ↓
qwen_dflash_module.py
```

下一部分将详细分析：

**DFlash 的 Block-level Draft：anchor、noise token padding、candidate_ids、first_scores、transition_scores，以及一次 forward 如何生成整个 Draft block。**