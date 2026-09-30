
## 阶段 4 · 第 3 讲：EAGLE-1、EAGLE-2、EAGLE-3 的演进

EAGLE 系列的演进主线可以概括为：

```text
EAGLE-1：预测 Target 的 hidden feature
        ↓
EAGLE-2：根据置信度动态调整候选树
        ↓
EAGLE-3：训练阶段模拟推理，并融合多层 feature
```

---

### 1. EAGLE-1：Hidden Feature Extrapolation

EAGLE-1 的核心思想是：

> 使用 Target Model 的中间 hidden feature，预测后续位置的 feature。

论文和源码介绍中强调的是 Target 的 second-top-layer contextual feature。

本地说明位置：

```text
D:\speculative-decoding-study\repos\EAGLE\README.md:34
```

基本流程：

```text
Target Model
    ↓
提取 hidden feature h_t
    ↓
EAGLE Draft Network
    ↓
预测 ĥ_{t+1}
    ↓
lm_head
    ↓
预测 token x̂_{t+1}
```

数学形式：

```text
h_t = Target(x_≤t)

ĥ_{t+1} = G(h_t, embedding(x_t))

x̂_{t+1} = argmax(lm_head(ĥ_{t+1}))
```

EAGLE-1 相比独立 Draft Model 的优势是：

```text
独立 Draft Model：
从 token 序列重新理解上下文

EAGLE-1：
直接使用 Target 已经提取的上下文 feature
```

---

### 2. EAGLE-1 的主要限制

EAGLE-1 通常依赖一个预先设计好的候选树结构：

```text
固定 top-k
固定树深度
固定候选节点数量
```

例如：

```text
depth = 5
top_k = 8
total_tokens = 63
```

候选树可能类似：

```text
             root
       ┌──────┼──────┐
       A      B      C
     ┌─┼─┐  ┌─┼─┐  ┌─┼─┐
    A1 A2 A3 B1 B2 B3 C1 C2
```

问题是：

- 某些输入很容易预测，不需要这么多分支；
- 某些输入很困难，需要更多分支；
- 固定树无法适应不同上下文；
- 候选数量增加后，Target 验证和 Attention mask 的成本会上升。

因此 EAGLE-1 的主要矛盾是：

```text
候选树太小：可能漏掉正确路径

候选树太大：验证成本和显存开销增加
```

---

### 3. EAGLE-2：Dynamic Draft Tree

EAGLE-2 的主要改进是：

> 根据 Draft Model 的置信度，动态决定候选树的结构。

本地 README 的说明：

```text
D:\speculative-decoding-study\repos\EAGLE\README.md:46-50
```

EAGLE-2 不再简单使用固定树，而是根据每个候选的分数决定：

```text
哪些节点值得继续扩展
哪些节点应该被裁剪
```

---

### 4. Draft Score 如何体现置信度？

在 EAGLE 的候选生成代码中，会维护：

```python
scores_list = []
parents_list = []
ss_token = []
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:676-680
```

每个 token 都有一个 log probability：

```python
last_p = self.logsoftmax(last_headout)
```

然后取 top-k：

```python
top = torch.topk(
    last_p,
    top_k,
    dim=-1
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:704-713
```

下一层会将父节点分数和当前 token 分数相加：

```python
cu_scores = topk_p + scores[:, None]
```

这相当于计算一条路径的累计分数：

```text
score(path)
=
score(token_1)
+
score(token_2)
+
...
+
score(token_n)
```

源码位置：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:730-742
```

---

### 5. 为什么累计分数可以用于选树？

假设有三条候选路径：

```text
path A: p(A1) = 0.90, p(A2) = 0.85
path B: p(B1) = 0.80, p(B2) = 0.20
path C: p(C1) = 0.55, p(C2) = 0.50
```

直观上：

```text
A：每一步都比较可靠
B：第二步明显不可靠
C：整体置信度偏低
```

如果使用 log probability 累计：

```text
score(A) = log(0.90) + log(0.85)
score(B) = log(0.80) + log(0.20)
score(C) = log(0.55) + log(0.50)
```

那么：

```text
A 会被优先保留
B 的低置信度分支会被抑制
```

这比只看某一层的局部 top-k 更合理，因为它考虑了整条路径的质量。

---

### 6. EAGLE-2 的动态树构造

在生成多层候选之后，EAGLE 会对所有候选路径进行排序：

```python
scores_list = torch.cat(scores_list, dim=0).view(-1)

top_scores = torch.topk(
    scores_list,
    total_tokens,
    dim=-1
)

top_scores_index = top_scores.indices
top_scores_index = torch.sort(
    top_scores_index
).values
```

源码位置：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:776-786
```

最终只保留分数最高的候选节点：

```python
draft_tokens = ss_token_list[top_scores_index]
```

这一步实现了：

```text
生成较多候选
        ↓
计算候选分数
        ↓
选择高置信度节点
        ↓
形成最终稀疏候选树
```

所以 EAGLE-2 的“动态”主要体现在：

```text
候选节点不是简单按照固定位置保留，
而是根据候选路径分数进行筛选。
```

---

### 7. EAGLE-1 与 EAGLE-2 对比

| 对比项 | EAGLE-1 | EAGLE-2 |
|---|---|---|
| Draft 表示 | Hidden feature | Hidden feature |
| 候选树 | 相对固定 | 动态调整 |
| 节点选择 | 固定 top-k / 固定结构 | 根据置信度筛选 |
| 主要优化点 | Feature extrapolation | Draft tree search |
| 适应不同上下文 | 较弱 | 更强 |
| 目标 | 生成候选 | 生成更有价值的候选 |

可以理解为：

```text
EAGLE-1 解决“如何预测 feature”

EAGLE-2 解决“应该保留哪些 feature 路径”
```

---

### 8. EAGLE-3：Training-Time Test

EAGLE-3 的核心变化是：

> 在训练阶段模拟真实推理过程，减少训练和推理之间的差异。

本地 README：

```text
D:\speculative-decoding-study\repos\EAGLE\README.md:52-57
```

---

### 9. 什么是训练-推理不一致？

训练阶段通常使用真实的 Target feature：

```text
真实 h_t
    ↓
预测 h_{t+1}
```

但是推理阶段，下一步输入往往是模型自己预测出来的 feature：

```text
真实 h_t
    ↓
预测 ĥ_{t+1}

ĥ_{t+1}
    ↓
继续预测 ĥ_{t+2}
```

于是出现误差累积：

```text
第 1 步：预测误差较小
第 2 步：使用带误差的 feature
第 3 步：误差进一步扩大
第 4 步：候选 token 质量下降
```

这类似于序列模型中的 exposure bias：

```text
训练时：使用 ground-truth 状态

推理时：使用模型自己的预测状态
```

---

### 10. EAGLE-3 的 Training-Time Test

EAGLE-3 在训练过程中主动模拟推理：

```text
使用真实 feature 预测第一步
        ↓
使用预测 feature 预测第二步
        ↓
继续使用预测 feature
        ↓
训练模型适应自身误差
```

示意：

```text
训练输入：
h_t → ĥ_{t+1} → ĥ_{t+2} → ĥ_{t+3}
```

而不是：

```text
h_t → h_{t+1} → h_{t+2} → h_{t+3}
```

这样 Draft Network 在训练时就会见到：

```text
不完美的历史 feature
```

因此推理阶段更稳定。

---

### 11. EAGLE-3 的多层 Feature Fusion

EAGLE-3 认为只使用高层 feature 可能存在限制：

```text
高层 feature 更偏向最终 next-token prediction
```

因此 EAGLE-3 将多个层级的 feature 融合：

```text
低层 feature
中层 feature
高层 feature
        ↓
concat
        ↓
EAGLE-3 Draft Network
```

源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\utils.py:246-253
```

核心代码：

```python
if model.use_eagle3:
    ea_device = model.ea_layer.lm_head.weight.device

    if outputs["hidden_states"][0].device != ea_device:
        outputs["hidden_states"] = [
            x.to(ea_device)
            for x in outputs["hidden_states"]
        ]

    hidden_states = torch.cat(
        outputs["hidden_states"],
        dim=-1
    )
```

随后在 Draft Network 中进行维度投影：

```python
self.fc = nn.Linear(
    config.target_hidden_size * 3,
    self.hidden_size,
    bias=False
)
```

对应源码：

```text
D:\speculative-decoding-study\repos\EAGLE\eagle\model\cnets.py:531-535
```

这里的 `* 3` 可以理解为多个层级 feature 拼接后的输入维度。

---

### 12. 多层 Feature 的作用

不同层级的 hidden feature 通常包含不同信息：

```text
低层：
局部 token、词法和短程模式

中层：
句法、实体和局部语义

高层：
上下文语义、任务意图和 next-token 信息
```

融合之后：

```text
低层 feature ─┐
中层 feature ─┼─ concat ─ Draft Network
高层 feature ─┘
```

Draft Network 可以根据任务学习：

```text
什么时候依赖局部信息
什么时候依赖全局信息
什么时候依赖最终语义表示
```

---

### 13. EAGLE-3 与 EAGLE-1 的区别

| 对比项 | EAGLE-1 | EAGLE-3 |
|---|---|---|
| Feature 来源 | 主要使用指定层 feature | 融合低、中、高层 feature |
| 训练方式 | 常规 feature prediction | Training-Time Test |
| 推理误差适应 | 相对有限 | 更强 |
| Draft 输入 | 单层或较少 feature | 多层 feature concat |
| 主要目标 | 预测后续 feature | 提高长链 Draft 稳定性 |

可以这样记：

```text
EAGLE-1：预测 feature

EAGLE-2：选择更好的树

EAGLE-3：让 feature prediction 更接近真实推理
```

---

### 14. 三代 EAGLE 的完整演进

```text
EAGLE-1
  ├── Target hidden feature
  ├── Feature extrapolation
  └── 生成候选树

EAGLE-2
  ├── 继承 EAGLE-1 的 feature draft
  ├── 计算路径置信度
  └── 动态选择候选树

EAGLE-3
  ├── 训练时模拟推理误差
  ├── 融合多层 hidden feature
  └── 提高长链候选的稳定性
```

---

### 15. Acceptance Rate 与候选树的关系

候选树越大，不一定越快。

设：

```text
N = 候选节点数
L = 平均接受 token 数
T_draft = Draft 生成时间
T_verify = Target 验证时间
```

一轮有效速度大致取决于：

```text
speed ≈ (L + 1) / (T_draft + T_verify)
```

如果增加候选节点：

```text
L 可能增加
T_verify 也会增加
```

因此真正需要优化的是：

```text
accepted tokens per round
```

而不是单独追求：

```text
candidate count
```

EAGLE-2 的动态树思想就是：

```text
优先保留更可能被 Target 接受的节点，
减少低价值候选带来的验证开销。
```

---

### 16. 本节核心结论

需要记住：

1. EAGLE-1 解决 hidden feature extrapolation；
2. EAGLE-2 根据 Draft confidence 动态调整候选树；
3. EAGLE-3 用 training-time test 缓解训练和推理不一致；
4. EAGLE-3 融合低层、中层和高层 Target feature；
5. 候选树优化的目标不是树越大越好，而是提高 accepted tokens per round；
6. EAGLE 的演进主线是：

```text
Feature quality
    ↓
Tree quality
    ↓
Inference robustness
```

下一部分是：

**EAGLE 候选树的具体生成、Tree Attention 和 Target Verification。**