
## 阶段 5 · 第 1 讲：MTP（Multi-Token Prediction）总体架构

前面学习了 EAGLE：

```text
Target hidden feature
    ↓
EAGLE Draft Network
    ↓
候选树
    ↓
Target 验证
```

MTP 的核心思路不同：

> 在 Target Model 内部增加多个 Multi-Token Prediction Head，让模型能够连续预测未来多个 token。

---

### 1. MTP 解决什么问题？

普通自回归模型一次只预测一个 token：

```text
x_t
  ↓
Target Model
  ↓
x_{t+1}
```

生成下一个 token 后，才能继续预测：

```text
x_{t+1}
  ↓
Target Model
  ↓
x_{t+2}
```

MTP 希望利用额外的预测模块：

```text
当前 hidden state
    ↓
MTP Head 1 → 预测 x_{t+1}
    ↓
MTP Head 2 → 预测 x_{t+2}
    ↓
MTP Head 3 → 预测 x_{t+3}
```

然后由 Target 主干统一验证这些预测。

---

### 2. MTP 的基本结构

可以把 MTP 看成：

```text
Target Backbone
    ├── 原始 LM Head
    ├── MTP Head 1
    ├── MTP Head 2
    ├── MTP Head 3
    └── ...
```

其中：

```text
原始 LM Head：
生成当前 token 的正式预测

MTP Head：
预测未来 token，作为 speculative draft
```

推理时：

```text
Target Backbone
    ↓
hidden state h_t
    ↓
MTP Head 1
    ↓
draft token x̂_{t+1}
    ↓
MTP Head 2
    ↓
draft token x̂_{t+2}
    ↓
MTP Head 3
    ↓
draft token x̂_{t+3}
```

---

### 3. MTP 与独立 Draft Model 的区别

经典投机解码：

```text
小 Draft Model
    ↓
draft tokens
    ↓
大 Target Model 验证
```

MTP：

```text
Target Model + MTP Heads
    ↓
draft tokens
    ↓
Target Model 验证
```

区别如下：

| 对比项 | 独立 Draft Model | MTP |
|---|---|---|
| Draft 来源 | 独立小模型 | Target 内部额外模块 |
| 参数关系 | 两个独立模型 | 共享 Target 主干 |
| Draft 成本 | 需要运行另一个模型 | 运行轻量 MTP Head |
| 预测方式 | 小模型自回归 | 多个 MTP Head 串联 |
| 工程重点 | 两套模型 Cache | 主干 Cache + MTP Cache |
| 典型问题 | 模型不一致 | MTP head 预测误差累积 |

---

### 4. MTP 为什么通常仍然是串行 Draft？

MTP Head 之间通常存在依赖关系：

```text
MTP Head 1 预测 x̂_{t+1}
        ↓
MTP Head 2 使用 x̂_{t+1}
        ↓
预测 x̂_{t+2}
        ↓
MTP Head 3 使用前面结果
```

因此不是所有 MTP Head 都能完全并行计算。

典型流程：

```text
h_t
 ↓
Head 1 → x̂_{t+1}
 ↓
Head 2 → x̂_{t+2}
 ↓
Head 3 → x̂_{t+3}
```

这与 Medusa 的并行多头不同：

```text
Medusa：
同一个 hidden state
    ├── Head 1
    ├── Head 2
    └── Head 3
```

可以简单记忆：

```text
Medusa：多头并行预测

MTP：多头串行接力
```

---

### 5. 当前 Qwen 工程中的 MTP 入口

当前工程中的 MTP 调度入口：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
```

该部分主要负责：

```text
启动 MTP Draft
组织预测结果
调用 Verify
管理一轮推理状态
```

从工程视角看，MTP Engine 主要负责调度：

```text
_prefill
    ↓
_draft
    ↓
_verify
    ↓
_commit
```

---

### 6. MTP 的 Draft 阶段

当前工程的 MTP Draft 逻辑位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
```

可以抽象为：

```python
def _draft(...):
    hidden_state = target_forward(...)
    draft_tokens = []

    for mtp_head in mtp_heads:
        token = mtp_head(hidden_state)
        draft_tokens.append(token)
        hidden_state = update_state(
            hidden_state,
            token
        )

    return draft_tokens
```

其本质是：

```text
主干模型产生初始状态
        ↓
MTP Head 逐步预测
        ↓
得到一段 draft block
```

例如：

```text
输入：
The capital of France is

MTP Draft：
[Paris, and, it]
```

之后 Target Model 会验证：

```text
Paris
and
it
```

---

### 7. MTP 的 Verify 阶段

MTP Verify 主要位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
```

处理逻辑可以抽象为：

```text
MTP Draft：
[d1, d2, d3, d4]

Target 验证结果：
[d1, d2, x3, ...]
```

则：

```text
接受 d1
接受 d2
拒绝 d3
使用 Target token x3 作为 fallback
```

最终输出：

```text
[d1, d2, x3]
```

这仍然是：

```text
accepted prefix + fallback token
```

与经典 Speculative Decoding 的结构相同。

---

### 8. 接受规则

MTP 的 greedy 验证可以写成：

```python
for i in range(num_draft_tokens):
    if draft[i] == target[i]:
        accept_length += 1
    else:
        break
```

如果：

```text
draft  = [A, B, C, D]
target = [A, B, X, ...]
```

那么：

```text
accept_length = 2
```

最终保留：

```text
[A, B, X]
```

这里：

```text
A、B：
来自 MTP Draft

X：
来自 Target fallback
```

当前工程中与验证接受相关的逻辑位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_process.py:148-169
```

---

### 9. 为什么需要 fallback token？

如果第一个不匹配 token 被直接丢弃，Target 就没有向序列追加新 token：

```text
Draft：
[A, B, C]

Target：
[A, B, X]
```

如果只接受：

```text
[A, B]
```

生成过程就会停在 `B`。

因此需要使用 Target 在第一个拒绝位置的结果：

```text
fallback = X
```

最终：

```text
accepted draft prefix + target fallback
```

即：

```text
[A, B] + [X]
```

这保证生成过程可以继续向前推进。

---

### 10. MTP Cache 的组成

MTP 推理通常至少涉及两类状态：

```text
Target Model Cache
MTP Module Cache
```

更具体地说：

```text
Target KV Cache
MTP KV Cache
MTP hidden state
accepted length
context length
```

一轮推理中可能发生：

```text
Draft 阶段：
写入临时 MTP Cache

Verify 阶段：
Target 验证 draft token

Commit 阶段：
只提交接受前缀

Rollback 阶段：
丢弃拒绝 token 的状态
```

---

### 11. `commit_verify_cache()`

当前工程中 MTP Cache 提交逻辑位于：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py:387-417
```

它的核心任务是：

```text
根据 accept_length 保留有效 Cache
清理或覆盖拒绝部分
更新当前序列长度
让下一轮 MTP 从正确状态继续
```

可以抽象为：

```python
accepted_len = verify_result.accept_length

mtp_cache = mtp_cache[
    ...,
    :prefix_len + accepted_len,
    :
]

context_length = prefix_len + accepted_len
```

如果还有 Target fallback token，则需要把 fallback 对应状态一并提交。

---

### 12. MTP 的完整一轮

完整流程可以表示为：

```text
已有上下文
    ↓
Target 主干计算 hidden state
    ↓
MTP Head 1 预测未来 token
    ↓
MTP Head 2 继续预测
    ↓
MTP Head 3 继续预测
    ↓
得到 draft block
    ↓
Target 批量验证
    ↓
计算 accepted prefix
    ↓
提交接受的 Cache
    ↓
回滚拒绝状态
    ↓
进入下一轮
```

伪代码：

```python
draft_tokens, draft_state = mtp_draft(
    input_ids,
    target_state,
)

target_logits = target_verify(
    draft_tokens,
    target_cache,
)

accept_length, fallback = verify(
    draft_tokens,
    target_logits,
)

commit_verify_cache(
    accept_length=accept_length,
    fallback=fallback,
)
```

---

### 13. MTP 与 Medusa 的区别

```text
Medusa：
一个 Target hidden state
    ├── Head 1
    ├── Head 2
    └── Head 3

MTP：
Target hidden state
    ↓
Head 1
    ↓
Head 2
    ↓
Head 3
```

| 对比项 | Medusa | MTP |
|---|---|---|
| Head 关系 | 并行 | 通常串行 |
| 预测输入 | 主要是同一个 hidden state | 前一个 head 的结果 |
| 候选结构 | 树形 | 通常线性 draft block |
| Draft Cache | 较简单 | 需要维护 MTP 状态 |
| 主要优势 | 并行候选覆盖 | 与 Target 共享主干 |
| 主要风险 | 多头质量差异 | 误差逐步累积 |

---

### 14. MTP 与 EAGLE 的区别

```text
EAGLE：
Target hidden feature
    ↓
专门的 Draft Network
    ↓
预测 feature
    ↓
token
```

```text
MTP：
Target hidden state
    ↓
MTP Head 1
    ↓
token
    ↓
MTP Head 2
    ↓
token
```

主要差异：

| 对比项 | EAGLE | MTP |
|---|---|---|
| Draft 模块 | 独立轻量 Draft Network | 多个 MTP Head |
| 预测对象 | Hidden feature + token | 未来 token |
| 生成结构 | 候选树 | 通常线性链 |
| 自回归状态 | Draft feature state | MTP head state |
| 工程复杂度 | Tree mask、tree cache | Commit/rollback cache |

---

### 15. 为什么 MTP 仍然可能获得加速？

虽然 MTP Draft 具有串行依赖，但每个预测模块通常比完整 Target Model 轻量得多：

```text
完整 Target forward：
计算所有 Transformer 层

MTP forward：
只计算额外的轻量预测模块
```

因此总体耗时可能变成：

```text
T_round
=
T_mtp_draft
+
T_target_verify
```

只要：

```text
一次接受的 token 数
>
单轮 Draft 和 Verify 带来的额外开销
```

就可以获得加速。

近似速度：

```text
speedup
≈
(accepted tokens per round + 1)
/
(T_mtp_draft + T_verify)
```

---

### 16. 本节核心结论

需要记住：

1. MTP 是 Target Model 内部的多 token 预测模块；
2. MTP Head 通常形成串行 Draft 链；
3. Draft token 由 MTP Head 预测，Target 负责最终验证；
4. 第一个不匹配位置需要使用 Target fallback token；
5. MTP 需要管理 Target Cache 和 MTP Cache；
6. `commit_verify_cache()` 负责提交接受状态并回滚拒绝状态；
7. Medusa 偏向并行多头，MTP 偏向串行多步预测；
8. MTP 的真正收益取决于：

```text
accepted tokens per round
/
draft、verify、同步和 cache 管理成本
```

下一部分是：

**MTP 的逐 token Draft Chain：逐步分析 `qwen_mtp_engine.py`、`qwen_mtp_process.py` 和 `qwen_mtp_module.py` 的调用关系。**