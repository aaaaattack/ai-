
## 阶段 6 · 第 4 讲：DFlash2 的状态设计与性能优化

DFlash2 不是把 DFlash 改成自回归 Draft，而是继续保持：

```text
一次 forward
并行生成整个 Draft block
Target 批量验证
```

它主要解决 DFlash 的两个问题：

1. 每个位置只取 top-1，候选之间不一定连贯；
2. block 越往后，候选质量越容易下降，出现 suffix decay。

DFlash2 的公开技术说明显示，它在 DFlash 的并行 block Draft 上加入了轻量级 path selector 和局部动态卷积，同时保持单次并行 Draft。[DFlash2 官方技术说明](https://inco.ai/blog/dflash2/)

---

# 一、DFlash 的两个主要问题

## 1. 每个位置独立选择，可能导致 block 不连贯

假设 DFlash 对四个位置的 top-1 预测是：

```text
position 1: A
position 2: A
position 3: B
position 4: B
```

最终 block：

```text
[A, A, B, B]
```

每个位置单独看都可能是高概率 token，但整体可能产生：

```text
重复
语义跳跃
局部不衔接
```

原因是基础 DFlash 的每个 block position 主要独立产生 unary prediction：

```text
position 1 → top-1
position 2 → top-1
position 3 → top-1
position 4 → top-1
```

它没有显式回答：

```text
position 2 的 token 是否适合接在 position 1 后面？
```

---

## 2. Block 后半段的接受率下降

DFlash 是并行预测：

```text
d1, d2, d3, d4, d5, d6
```

但这些位置的条件依赖并没有像标准自回归模型一样逐步传播。

因此越靠后的 Draft position，越容易出现：

```text
前面的 token 条件没有被充分传递
预测误差积累
accepted prefix 在后面被截断
```

这叫：

```text
suffix decay
```

例如：

```text
position:
1     2     3     4     5     6

准确率：
90%   88%   85%   81%   76%   70%
```

即使第 6 个位置单独预测并不差，但由于前面任何位置错误都会截断 accepted prefix，最终有效接受长度会明显下降。

---

# 二、DFlash2 的整体结构

DFlash2 在 DFlash 上增加两个组件：

```text
DFlash Base Backbone
    ├── Local Dynamic Convolution
    └── Candidate Path Selector
```

完整流程：

```text
Target hidden feature
        +
anchor + noise block
        ↓
DFlash2 backbone
        ↓
每个位置得到 unary logits
        ↓
每个位置保留 top-k candidate
        ↓
Path Selector 重新评分相邻 token
        ↓
选出一条连贯路径
        ↓
Target Verify
```

对应本地 `speculators` 源码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\core.py:30-64
```

核心类：

```python
class DFlash2DraftModel(DFlashDraftModel):
```

DFlash2 继承 DFlash 的基础 Draft Model，并增加：

```python
self.candidate_selector = CandidateSelector(...)
```

因此：

```text
DFlash2 不是重新设计完整 Draft 架构，
而是复用 DFlash 并增加两个轻量模块。
```

---

# 三、改进一：Candidate Path Selector

## 1. DFlash 只保留 top-1

基础 DFlash 的逻辑大致是：

```python
unary_logits = draft_model(...)
draft_token = unary_logits.argmax(dim=-1)
```

对于每个位置，只留下一个候选：

```text
position 1 → A
position 2 → B
position 3 → C
```

问题是：

```text
如果 position 2 的 top-1 不适合接在 A 后面，
即使 position 2 的 top-2 才是正确 token，
它也已经被丢弃。
```

---

## 2. DFlash2 保留 top-k 候选

DFlash2 首先不直接选 top-1，而是保留每个位置的 top-k：

```python
candidate_ids = unary_logits.topk(
    top_k,
    dim=-1
).indices
```

源码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:308-316
```

假设：

```text
block_size = 4
top_k = 3
```

得到：

```text
position 1: [A1, A2, A3]
position 2: [B1, B2, B3]
position 3: [C1, C2, C3]
position 4: [D1, D2, D3]
```

张量形状可以表示为：

```text
candidate_ids:
[block_size, top_k]
=
[4, 3]
```

此时正确 token 即使不是某个位置的 top-1，也仍然可能留在 top-k 候选集合中。

---

## 3. Unary Score

DFlash2 首先保留 DFlash 原始 logits：

```text
unary score
```

可以理解为：

```text
U_t(b)
```

它表示：

```text
当前位置 t 对候选 token b 的单点偏好。
```

如果只使用 unary score：

```text
score(b) = U_t(b)
```

就会退化为：

```text
每个位置单独选择 top-1
```

DFlash2 在此基础上增加相邻 token 的 transition score。

---

# 四、Transition Score：让候选 token 连起来

## 1. 核心公式

DFlash2 的 path selector 对相邻 token 进行重新评分：

```text
S_t(a, b)
=
U_t(b)
+
transition_t(a, b)
```

其中：

```text
a：
前一个 token

b：
当前位置候选 token

U_t(b)：
DFlash unary logit

transition_t(a, b)：
a → b 的相邻转移分数
```

官方说明给出的形式是一个低秩双线性打分：

```text
S_t(a,b)
=
U_t(b)
+
⟨A(a) ⊙ H(h_t), B(b)⟩
```

其中：

```text
A(a)：
前驱 token 的 embedding

B(b)：
当前候选 token 的 embedding

H(h_t)：
当前 hidden state 产生的 context gate

⊙：
逐元素乘法
```

其含义是：

```text
当前候选 b 本身是否可能
+
b 是否适合接在前一个 token a 后面
```

---

## 2. 本地实现对应

DFlash2 的 selector 位于：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:244-316
```

核心模块：

```python
class CandidateSelector(nn.Module):
```

它包含：

```python
self.predecessor_codebook
self.successor_codebook
self.hidden_projection
```

对应三类信息：

```text
predecessor_codebook：
前驱 token 表示

successor_codebook：
候选 token 表示

hidden_projection：
当前 hidden state 的上下文投影
```

---

## 3. Context Gate

源码：

```python
def context(
    self,
    hidden_states,
    predecessor_ids,
):
    predecessor = self.predecessor_codebook[
        predecessor_ids.long()
    ]

    projected_hidden = self.hidden_projection(
        hidden_states
    )

    return predecessor * projected_hidden
```

对应：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:260-274
```

可以写成：

```text
context(a, h_t)
=
A(a) ⊙ H(h_t)
```

这个 context 表示：

```text
在当前上下文 h_t 下，
前驱 token a 对后续 token 选择的影响。
```

---

## 4. Transition Score

源码：

```python
def transition_scores(
    self,
    hidden_states,
    predecessor_ids,
    candidate_ids,
):
    context = self.context(
        hidden_states,
        predecessor_ids
    )

    successors = self.successor_codebook[
        candidate_ids.long()
    ]

    return (
        context.unsqueeze(-2)
        * successors
    ).sum(dim=-1)
```

对应：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:276-285
```

公式对应：

```text
transition(a,b)
=
⟨A(a) ⊙ H(h_t), B(b)⟩
```

这个分数并不是完整的 vocabulary logits，而是只对保留下来的 top-k candidate 进行重新排序。

---

# 五、为什么 Selector 不会破坏并行性？

这是 DFlash2 的关键设计。

一种错误实现是：

```text
选择 d1
    ↓
重新运行 Draft 得到 d2
    ↓
重新运行 Draft 得到 d3
```

这会变回自回归 Draft。

DFlash2 不是这样做的。

它执行：

```text
一次 DFlash forward
    ↓
所有位置得到 unary logits
    ↓
所有位置保留 top-k
    ↓
并行计算所有相邻 candidate pair 的 transition score
    ↓
最后进行轻量路径选择
```

因此：

```text
神经网络计算仍然是并行的
```

只有最后的路径遍历是轻量级串行操作：

```text
从 anchor 开始
选择 position 1 的候选
再选择 position 2 的候选
继续到 block 末尾
```

这里的串行部分只是在已有分数上做选择，不再执行新的 Transformer forward。

---

# 六、Path Selection 的具体例子

假设：

```text
anchor = A
```

各位置的候选为：

```text
position 1: [B1, B2, B3]
position 2: [C1, C2, C3]
position 3: [D1, D2, D3]
```

假设 unary score：

```text
position 1:
B1 = 0.90
B2 = 0.80
B3 = 0.60

position 2:
C1 = 0.95
C2 = 0.85
C3 = 0.50

position 3:
D1 = 0.90
D2 = 0.80
D3 = 0.70
```

单独取 top-1：

```text
[B1, C1, D1]
```

但 transition score 可能是：

```text
A → B1 = +0.10
A → B2 = +0.30

B1 → C1 = -0.30
B1 → C2 = +0.25

B2 → C1 = +0.05
B2 → C2 = +0.10

C2 → D2 = +0.25
```

于是更好的路径可能是：

```text
A → B2 → C2 → D2
```

虽然：

```text
B2 不是 position 1 的 top-1
C2 不是 position 2 的 top-1
D2 不是 position 3 的 top-1
```

但它们组合起来更加连贯。

这正是 DFlash2 的价值：

```text
从“每个位置独立最优”
变成“整条 block 路径更优”。
```

---

# 七、Path Selector 的运行时流程

可以抽象为：

```python
# 1. 一次 DFlash forward
unary_logits = dflash_forward(...)

# 2. 每个位置保留 top-k
candidate_ids = unary_logits.topk(
    top_k,
    dim=-1,
).indices

# 3. 计算相邻候选的 transition score
candidate_scores = selector.score_candidates(
    unary_logits,
    hidden_states,
    predecessor_ids,
    candidate_ids,
)

# 4. 从 anchor 开始选择一条路径
selected_path = select_path(
    candidate_ids,
    candidate_scores,
    anchor_token,
)

# 5. 交给 Target Verify
target_verify(selected_path)
```

源码入口：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:287-316
```

---

# 八、Selector 的训练

DFlash2 的 selector 不能只训练：

```text
每个位置的 token 分类
```

还需要训练：

```text
在 top-k 候选中，
哪个 token 是 Target 真正会接受的 token。
```

因此训练大致包含两部分：

```text
DFlash unary loss
+
Candidate selector loss
```

源码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\core.py:101-114
```

---

## 1. Unary Loss

Unary loss 训练基础 DFlash：

```text
block position → target token
```

对应：

```text
DFlash 原始 Draft 能力
```

## 2. Selector Loss

Selector loss 训练：

```text
前驱 token
+
当前 hidden state
+
top-k candidates
        ↓
选择 Target token
```

源码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\metrics.py:76-159
```

最终损失大致为：

```text
L_total
=
L_unary
+
α · L_selector
```

其中：

```text
L_unary：
基础 DFlash token prediction loss

L_selector：
候选路径选择 loss

α：
selector loss 权重
```

---

## 3. 如果 Target token 不在 top-k 中怎么办？

训练时可能出现：

```text
Target token 不在 unary top-k candidates 中
```

如果直接计算 selector loss：

```text
正确答案不在候选集合中
```

就没有有效的分类目标。

当前实现会在训练候选集合中处理这种情况：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\metrics.py:31-62
```

逻辑是：

```text
如果 Target token 已经在 top-k 中：
    保留原候选集合

如果 Target token 不在 top-k 中：
    用 Target token 替换候选集合中的一个位置
```

这样 selector 每个位置都有明确的训练目标。

但要注意：

```text
训练时可以注入 Target token，
推理时不能凭空注入 Target token。
```

训练注入只是为了构造可学习的监督目标。

---

# 九、改进二：Local Dynamic Convolution

Path selector 解决：

```text
候选选择不连贯
```

但它不能改变候选集合本身。

如果 block 后半段已经没有高质量候选：

```text
top-k 中都没有正确 token
```

selector 也无法恢复。

因此 DFlash2 增加第二个组件：

```text
local dynamic convolution
```

它解决：

```text
block 内局部依赖传播不足
```

---

## 1. DFlash 原始 block 的问题

DFlash 的 block 内部虽然可以并行计算，但不同位置之间的局部依赖传播有限。

例如：

```text
d1 影响 d2
d2 影响 d3
d3 影响 d4
```

标准自回归模型会逐步传播：

```text
d1 → d2 → d3 → d4
```

DFlash 为了保持并行，没有完全采用这种串行计算。

于是需要一种：

```text
不破坏并行
但能传播邻居信息
```

的机制。

---

## 2. 两 Tap Convolution

DFlash2 使用局部两 Tap 卷积：

```text
Conv(x_t)
=
k_{t,0} ⊙ x_t
+
k_{t,1} ⊙ x_{t-1}
```

含义：

```text
当前位置表示
+
前一个位置表示
```

对于第一个 block 位置：

```text
x_{t-1} = anchor representation
```

对于后续位置：

```text
x_{t-1} = block 内前一个位置
```

官方技术说明将其描述为：每个位置混合自身表示和前驱位置表示，同时所有位置仍然可以并行计算。[DFlash2 官方技术说明](https://inco.ai/blog/dflash2/)

---

## 3. 为什么卷积仍然是并行的？

表面上：

```text
position t 读取 position t-1
```

似乎有串行依赖。

但卷积不是：

```text
先计算 x1，再计算 x2，再计算 x3
```

而是使用固定的 shift/gather 操作：

```text
shifted_x = [anchor, x1, x2, x3]
```

然后整体计算：

```text
y = k0 ⊙ x + k1 ⊙ shifted_x
```

所以整个 block 仍然是一次 Tensor 运算。

关键区别：

```text
自回归：
前一个位置必须先完成神经网络计算

局部卷积：
前一个位置只需作为固定张量偏移参与计算
```

因此 DFlash2 保留了 block-level 并行性。

---

# 十、DFlash2 卷积的源码

配置定义：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\config.py:23-44
```

默认参数包括：

```python
conv_kernel_size = 2
conv_group_size = 16
```

卷积实现：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:105-174
```

核心类：

```python
class GroupedDynamicCausalConv(nn.Module):
```

它包含：

```python
self.base_kernel
self.kernel_projection
```

其中：

```text
base_kernel：
基础卷积权重

kernel_projection：
根据当前 hidden state 动态生成卷积修正
```

---

## 1. Dynamic Kernel

代码逻辑：

```python
kernels = self.kernel_projection(
    hidden_states
)
```

也就是说，卷积核不是完全固定的：

```text
kernel_t = f(hidden_state_t)
```

因此不同 token、不同上下文可以使用不同的局部混合权重。

---

## 2. Grouped Convolution

hidden dimension 被分成多个 group：

```text
hidden_size / group_size
```

例如：

```text
hidden_size = 4096
group_size = 16
```

则：

```text
num_groups = 256
```

每个 group 使用自己的动态修正，避免为每个通道都学习一个完全独立的卷积核。

这样可以降低参数量和计算量。

---

# 十一、卷积放在哪里？

DFlash2 不只在输入处放一次卷积，而是在每个 Draft Decoder Layer 中围绕 Attention 和 MLP 使用。

源码：

```text
D:\speculative-decoding-study\repos\speculators\src\speculators\models\dflash2\model_definitions.py:175-240
```

结构大致是：

```text
Input LayerNorm
    ↓
Attention Conv.prepare
    ↓
Self Attention
    ↓
Attention Conv.finish
    ↓
Residual
    ↓
Post Attention LayerNorm
    ↓
MLP Conv.prepare
    ↓
MLP
    ↓
MLP Conv.finish
    ↓
Residual
```

伪代码：

```python
hidden = input_layernorm(hidden)

hidden, attn_kernel = attention_conv.prepare(
    hidden
)

hidden = self_attn(
    hidden,
    target_hidden=target_hidden,
)

hidden = attention_conv.finish(
    hidden,
    attn_kernel
)

hidden = residual + hidden

hidden = post_attention_layernorm(
    hidden
)

hidden, mlp_kernel = mlp_conv.prepare(
    hidden
)

hidden = mlp(hidden)

hidden = mlp_conv.finish(
    hidden,
    mlp_kernel
)

hidden = residual + hidden
```

---

# 十二、为什么 Attention 和 MLP 都需要卷积？

Attention 主要负责：

```text
读取 Target context
建模长距离关系
```

MLP 主要负责：

```text
非线性变换
特征提炼
```

而 block 内局部依赖主要表现为：

```text
当前位置与前一个 Draft position 的关系
```

DFlash2 在两类子层前后都加入局部卷积，使局部信息可以持续传播：

```text
Attention：
理解上下文

Local Conv：
传递邻接 token 信息

MLP：
完成非线性变换
```

---

# 十三、DFlash2 的状态管理变化

DFlash2 增加了：

```text
local convolution
candidate selector
```

但没有改变 Target Verify 的基本状态机：

```text
Draft
    ↓
Target Verify
    ↓
accepted prefix
    ↓
Commit/Rollback
```

---

## 1. Convolution State

DFlash2 的卷积是 block-local 的。

它只在当前 Draft block 内传播：

```text
anchor → d1 → d2 → d3
```

下一轮开始时：

```text
使用新的 anchor
重新构造新的 block
重新进行局部卷积
```

因此它不是普通自回归模型那种长期 recurrent state。

---

## 2. 当前 block 被拒绝时

假设：

```text
Draft：
[d1, d2, d3, d4]

Target：
[d1, x2, ...]
```

则：

```text
d1 接受
d2、d3、d4 拒绝
```

卷积产生的：

```text
d2、d3、d4 局部状态
```

不能泄漏到下一轮。

下一轮必须从：

```text
[prefix, d1, x2]
```

对应的新 anchor 重新建立 block。

因此 DFlash2 的卷积状态必须满足：

```text
每轮 block 结束后可重建
不能把拒绝 suffix 作为下一轮历史
```

---

## 3. Selector 状态

Candidate Selector 通常不是一个需要长期保存的 KV Cache，而是：

```text
当前 block 的 candidate IDs
当前 block 的 selector scores
当前 block 的 predecessor IDs
```

一轮结束后：

```text
accepted path：
提交为正式 token

rejected candidates：
直接丢弃
```

因此 Selector 的回滚比 Target/Draft KV Cache 简单。

---

# 十四、DFlash2 的完整 Draft 流程

```text
1. 读取上一轮正式状态
        ↓
2. 取最后一个确认 token 作为 anchor
        ↓
3. 构造 [anchor, noise, ..., noise]
        ↓
4. 读取 Target hidden features
        ↓
5. DFlash2 block forward
        ↓
6. 生成每个位置的 unary logits
        ↓
7. 每个位置保留 top-k candidates
        ↓
8. Candidate Selector 计算 pairwise scores
        ↓
9. 选择一条 coherent path
        ↓
10. 交给 Target Verify
        ↓
11. 提交 accepted prefix
        ↓
12. 回滚 rejected suffix
```

---

# 十五、DFlash2 的性能构成

设：

```text
T_backbone：
DFlash2 Draft Backbone 时间

T_selector：
Path Selector 时间

T_verify：
Target Verify 时间

T_commit：
状态提交和回滚时间

A：
平均 accepted tokens
```

一轮总时间：

```text
T_round
=
T_backbone
+
T_selector
+
T_verify
+
T_commit
```

平均吞吐：

```text
throughput
≈
(A + 1)
/
T_round
```

DFlash2 的优化目标是：

```text
提高 A
同时让 T_selector 和新增卷积成本很小
```

官方技术说明报告，在其测试中 path selector 和 convolution 带来的额外 cycle latency 较小，同时显著提高平均 acceptance length；具体数值属于其模型和 benchmark 条件下的结果。[DFlash2 官方技术说明](https://inco.ai/blog/dflash2/)

---

# 十六、为什么不能只扩大 DFlash Draft Model？

一个直接方案是：

```text
DFlash 5 layers
    ↓
DFlash 15 layers
```

这样可以提升 Draft 能力，但代价是：

```text
参数量增加
Draft 时间增加
显存增加
Target Verify 的收益被 Draft 成本抵消
```

DFlash2 采用更有针对性的方案：

```text
局部卷积：
修复 block 内相邻依赖

Path selector：
修复候选路径选择
```

相较于简单增加层数：

```text
只在真正的瓶颈位置增加能力
```

---

# 十七、DFlash2 与 DSpark 的区别

从结构上看：

```text
DFlash2：
并行 block
+
局部卷积
+
candidate path selector
```

DSpark 更偏向：

```text
DFlash block
+
Markov-style sequential correction
+
confidence head
```

关键区别：

```text
DFlash2 尽量保持 one-pass parallel drafting

DSpark 使用额外的结构对 block 内 token 关系进行修正
```

因此 DFlash2 的设计目标是：

```text
不牺牲并行 Draft 的主要优势，
用很小的模块恢复局部连贯性。
```

---

# 十八、DFlash2 对当前 Qwen 工程的启示

当前 Qwen DFlash 工程的状态回滚重点仍然在：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_dflash_module.py:503-540
```

DFlash2 如果接入同样的 Runtime，仍然需要保证：

```text
Target KV Cache
Draft KV Cache
hidden state
conv state
current length
candidate path
```

在接受/拒绝后保持一致。

需要特别注意：

```text
candidate selector 选出的是一条 Draft path，
并不代表 Target 已经接受它。
```

流程仍然是：

```text
selector path
    ↓
Target Verify
    ↓
accepted prefix
    ↓
commit/rollback
```

因此不能把：

```text
selector score
```

误认为：

```text
Target acceptance
```

---

# 十九、DFlash、DFlash2、EAGLE、MTP 对比

| 方法 | Draft 方式 | 主要问题 | 解决方案 |
|---|---|---|---|
| MTP | 串行 Head Chain | Draft 逐 token | 共享 Target 主干 |
| EAGLE | Feature autoregression | Draft 仍有串行链 | 使用 hidden feature |
| DFlash | 并行 block | 位置之间不够连贯 | Target hidden conditioning |
| DFlash2 | 并行 block | 候选选择和 suffix decay | Selector + Local Conv |

可以把 DFlash2 理解为：

```text
DFlash 的并行 Draft
+
EAGLE 关注的 feature quality
+
轻量局部依赖建模
+
路径级候选选择
```

---

# 二十、本节核心结论

需要记住：

1. DFlash2 仍然保持 block-level parallel drafting；
2. 它不通过自回归 correction 恢复连贯性；
3. 它通过两个轻量模块进行改进：

```text
Local Dynamic Convolution
Candidate Path Selector
```

4. Path Selector：

```text
每个位置保留 top-k
计算相邻 candidate transition score
选择一条更连贯的路径
```

5. Local Dynamic Convolution：

```text
当前位置混合自身表示和前驱位置表示
```

6. 两者都不改变 Target Verify 的最终权威性；
7. DFlash2 的状态提交仍然遵循：

```text
只提交 Target 接受的 prefix
拒绝 suffix 全部回滚
```

8. 性能目标是：

```text
提高 accepted tokens per round
同时控制 Draft 和 Selector 开销
```

DFlash 阶段至此完成：

```text
DFlash Block Draft
    ↓
Target Verify
    ↓
Accepted Prefix
    ↓
Multi-state Rollback
    ↓
DFlash2 Selector + Local Conv
```

下一阶段进入生产级 Runtime：

**Speculators：如何把 EAGLE、MTP、DFlash 统一成可训练、可转换、可部署的工程框架。**