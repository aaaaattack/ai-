
## 阶段 2 · 第 4 讲：Medusa 的 Posterior Evaluation

本节目标：理解 Medusa 如何在候选树中选择“最长正确前缀”。

### 1. 为什么需要 Posterior Evaluation？

Medusa 会一次生成多条候选路径：

```text
root
├── A
│   ├── A1
│   └── A2
└── B
    └── B1
```

候选路径可能是：

```text
root → A → A1
root → A → A2
root → B → B1
```

Target model 计算后，需要判断：

1. 哪条候选路径最可信；
2. 这条路径连续接受了多少个 token；
3. 从哪个 token 开始回退并重新生成。

Medusa 的核心逻辑位于：

`C:\Users\wei.xie\Documents\Codex\2026-09-28\data-xiewei-imodelzoo-models-llm-qwen3\work\..\repos\Medusa\medusa\model\utils.py`

源码中的主要函数是：

```python
evaluate_posterior()
```

### 2. Greedy Posterior Evaluation

Medusa 的基本判断方式是：

```python
posterior_mask = (
    candidates[:, 1:]
    == torch.argmax(logits[:, :-1], dim=-1)
).int()

candidates_accept_length = (
    torch.cumprod(posterior_mask, dim=1).sum(dim=1)
)

accept_length = candidates_accept_length.max()
best_candidate = torch.argmax(
    candidates_accept_length
).to(torch.long)
```

对应源码位置：

`C:\Users\wei.xie\Documents\Codex\2026-09-28\data-xiewei-imodelzoo-models-llm-qwen3\repos\Medusa\medusa\model\utils.py:436-489`

### 3. 逐行理解

假设候选序列为：

```text
candidate 0: [root, A, A1]
candidate 1: [root, A, A2]
candidate 2: [root, B, B1]
```

Target model 的贪心预测结果为：

```text
candidate 0: [A, A1]
candidate 1: [A, B]
candidate 2: [B, B2]
```

逐 token 比较：

```text
candidate 0: A  == A
             A1 == A1
             接受 2 个 token

candidate 1: A  == A
             A2 != B
             接受 1 个 token

candidate 2: B  == B
             B1 != B2
             接受 1 个 token
```

因此：

```text
candidates_accept_length = [2, 1, 1]
best_candidate = 0
accept_length = 2
```

最终接受：

```text
root → A → A1
```

### 4. `cumprod` 的作用

`posterior_mask` 可能是：

```text
[1, 1, 0, 1]
```

含义是：

```text
第 1 个 token 正确
第 2 个 token 正确
第 3 个 token 错误
第 4 个 token 即使正确，也不能继续接受
```

`torch.cumprod` 会把它转换成：

```text
[1, 1, 0, 0]
```

因此最终只接受最长连续前缀：

```text
accept_length = 2
```

这体现了自回归生成的依赖关系：后续 token 建立在前面的 token 正确的基础上。

### 5. 与经典投机解码的区别

经典投机解码通常是：

```text
draft token
    ↓
target 分布比较
    ↓
接受 / 拒绝
```

它可以使用：

```text
min(1, p_target / p_draft)
```

进行 rejection sampling，从而保持 target model 的输出分布。

Medusa 的基础 greedy posterior evaluation 更接近：

```text
candidate token
    ↓
是否等于 target 的 argmax
    ↓
接受最长连续前缀
```

两者区别如下：

| 对比项 | 经典投机解码 | Medusa Greedy Evaluation |
|---|---|---|
| 候选来源 | 独立 Draft Model | Target Model 内部的多个 Head |
| 判断依据 | 概率比或 token 相等 | Target logits 的 argmax |
| 候选结构 | 通常是一条线性序列 | 多分支候选树 |
| 接受方式 | 逐 token 接受/拒绝 | 选择最长正确候选前缀 |
| 是否天然保持采样分布 | Rejection Sampling 可以 | Greedy 模式主要针对贪心解码 |
| 主要目标 | 无损加速 | 用多头预测提高候选覆盖率 |

### 6. Medusa 的候选选择过程

完整过程可以概括为：

```text
Medusa Heads
    ↓
生成多分支候选
    ↓
Tree Attention 一次计算
    ↓
Target logits
    ↓
evaluate_posterior()
    ↓
选择 best_candidate
    ↓
接受 accept_length 个 token
    ↓
更新 KV Cache
```

候选树的生成位于：

`C:\Users\wei.xie\Documents\Codex\2026-09-28\data-xiewei-imodelzoo-models-llm-qwen3\repos\Medusa\medusa\model\utils.py:258-340`

候选状态更新位于：

`C:\Users\wei.xie\Documents\Codex\2026-09-28\data-xiewei-imodelzoo-models-llm-qwen3\repos\Medusa\medusa\model\utils.py:531-580`

### 7. 与 DFlash 的联系

DFlash 中也存在类似的“验证—接受—回滚”流程：

- DFlash 生成候选 token；
- Target model 验证候选；
- 计算接受长度；
- 提交接受的 token；
- 回滚未接受的 KV Cache。

相关代码：

`C:\Users\wei.xie\Documents\Codex\2026-09-28\data-xiewei-imodelzoo-models-llm-qwen3\repos\qwen3.5\qwen_dflash_process.py:228-245`

Medusa 的差异在于：

```text
DFlash：通常是一段连续候选
Medusa：是一棵候选树
```

因此 Medusa 的关键难点不是单纯的 token 验证，而是：

```text
如何构造候选树
如何高效执行 Tree Attention
如何从多个分支中选择最佳路径
```

### 8. 本节核心结论

需要记住四点：

1. `posterior_mask` 判断候选 token 是否等于 Target model 的预测；
2. `cumprod` 保证只接受连续正确的前缀；
3. `best_candidate` 选择接受长度最大的候选路径；
4. Medusa 的 posterior evaluation 是“候选树验证”，不是普通线性 Draft 验证。

下一节可以继续学习：

**Medusa 的 KV Cache 更新与回滚机制**。