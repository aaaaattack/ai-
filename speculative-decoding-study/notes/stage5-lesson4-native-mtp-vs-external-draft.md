
## 阶段 5 · 第 4 讲：原生 MTP 与外部 Draft Model 的工程区别

这一讲回答一个核心问题：

> MTP 是不是普通 Speculative Decoding？

答案是：

```text
算法目标相同：
Draft → Verify → Accept/Rollback

模型组织方式不同：
MTP 是 Target 内部的预测模块，
外部 Draft 是独立模型。
```

---

## 1. 两种架构对比

### 外部 Draft Model

```text
┌─────────────────┐
│ Draft Model     │
│ 小模型          │
└────────┬────────┘
         │ draft tokens
         ▼
┌─────────────────┐
│ Target Model    │
│ 大模型          │
└─────────────────┘
```

流程：

```text
输入上下文
    ↓
Draft Model 生成多个 token
    ↓
Target Model 批量验证
    ↓
接受前缀
    ↓
回滚 Draft/Target 状态
```

经典 `llama.cpp` 的实现属于这种模式。

源码入口：

```text
D:\speculative-decoding-study\repos\llama.cpp\examples\speculative-simple\speculative-simple.cpp:187-298
```

---

### 原生 MTP

```text
┌────────────────────────────┐
│ Target Model               │
│                            │
│  Backbone                  │
│     ├── Original LM Head   │
│     ├── MTP Head 1         │
│     ├── MTP Head 2         │
│     └── MTP Head 3         │
└──────────────┬─────────────┘
               │ draft tokens
               ▼
        Target Verify
```

流程：

```text
Target Backbone 计算 hidden state
    ↓
MTP Head 链生成 Draft token
    ↓
Target Backbone 验证
    ↓
提交接受前缀
    ↓
回滚 MTP 临时状态
```

当前 Qwen 工程属于这种模式：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_process.py
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py
```

---

## 2. 最大区别：参数是否共享

### 外部 Draft Model

Draft 和 Target 是两个独立模型：

```text
Draft parameters：
θ_d

Target parameters：
θ_t
```

通常：

```text
θ_d ≠ θ_t
```

它们可能有不同的：

```text
层数
hidden size
词表
Tokenizer
KV Cache
设备
量化配置
```

例如：

```text
Draft：
1B 模型

Target：
7B 模型
```

### MTP

MTP 通常共享 Target 的主干：

```text
Target Backbone：
θ_t

MTP Heads：
φ_1, φ_2, φ_3
```

模型变成：

```text
Target = Backbone(θ_t) + MTP Modules(φ)
```

因此 Draft 不需要重新运行一个完整的小模型。

---

## 3. Draft 生成路径不同

### 外部 Draft Model

```python
draft_logits = draft_model(
    input_ids,
    past_key_values=draft_cache,
)

draft_token = sample(draft_logits)
```

下一步继续：

```python
draft_logits = draft_model(
    input_ids + [draft_token],
    past_key_values=draft_cache,
)
```

Draft 的 token 生成完全依赖：

```text
Draft Model 自己的参数和 Cache
```

### MTP

MTP 通常是：

```python
hidden = target_backbone(...)
draft_token_1 = mtp_head_1(hidden)

hidden = update_with_token(
    hidden,
    draft_token_1,
)
draft_token_2 = mtp_head_2(hidden)
```

它使用：

```text
Target hidden state
MTP Head
MTP 自己的中间状态
```

当前 Qwen 的 Draft 调度入口：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_engine.py:81-106
```

---

## 4. 为什么原生 MTP 通常 Draft 更“贴近” Target？

外部 Draft Model 需要近似 Target：

```text
Draft Model ≈ Target Model
```

即使两个模型训练目标相似，也可能出现：

```text
隐藏状态不同
词表概率不同
长文本行为不同
指令跟随行为不同
```

MTP 直接使用 Target Backbone 的 hidden state：

```text
Target hidden
    ↓
MTP Head
```

因此 MTP 不需要重新学习：

```text
完整语义表示
上下文建模
长距离依赖
```

它主要学习：

```text
如何从 Target hidden state 预测未来 token
```

所以 MTP 的 Draft token 往往更容易与 Target 保持一致。

---

## 5. MTP 为什么仍然会产生误差？

共享 Backbone 不代表 MTP 预测完全正确。

原因包括：

### 预测距离增加

```text
Head 1：
预测 t+1

Head 2：
预测 t+2

Head 3：
预测 t+3
```

越远的 token，预测难度越高。

### 前面预测会影响后面

```text
d1 错误
    ↓
Head 2 读取错误条件
    ↓
d2 更可能错误
```

这与普通 Draft Model 的误差累积类似。

### MTP Head 能力有限

MTP Head 通常比完整 Target Backbone 小得多：

```text
MTP Head 的表达能力
<
Target Backbone 的表达能力
```

因此仍然必须使用 Target Verify。

---

## 6. Cache 结构不同

### 外部 Draft Model

需要维护两套完整 Cache：

```text
Draft KV Cache
Target KV Cache
```

状态如下：

```text
Draft Model：
[prefix, d1, d2, d3]

Target Model：
[prefix]
```

Verify 后：

```text
Target：
[prefix, accepted tokens, fallback]

Draft：
回滚到相同前缀
```

优点：

```text
结构清晰
两个模型彼此独立
```

缺点：

```text
显存开销更大
两套 Cache 都要管理
两个模型之间需要同步
```

### MTP

主要维护：

```text
Target Backbone Cache
MTP Module Cache
MTP hidden state
```

它们共享同一个 Target 上下文，但 MTP 有额外临时状态：

```text
Target Cache：
正式序列状态

MTP Cache：
Draft Chain 状态
```

因此 MTP 的 Cache 数量可能较少，但状态依赖更紧密。

相关提交代码：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_module.py:387-417
```

---

## 7. Verify 阶段本质上相同

虽然 Draft 来源不同，但 Verify 的思想相同：

```text
Draft：
[d1, d2, d3, d4]

Target：
[t1, t2, t3, t4]
```

逐位置比较：

```text
d1 == t1
d2 == t2
d3 == t3
...
```

只接受最长连续前缀：

```python
accept_length = 0

for d, t in zip(draft, target):
    if d == t:
        accept_length += 1
    else:
        break
```

Qwen MTP 验证逻辑：

```text
/workspace/imodelzoo/models/llm/qwen3.5/qwen_mtp_process.py:148-169
```

经典 `llama.cpp` 线性 Draft 的验证逻辑：

```text
D:\speculative-decoding-study\repos\llama.cpp\examples\speculative-simple\speculative-simple.cpp:221-298
```

因此：

```text
MTP 改变的是 Draft 生成方式，
不是 Verify 的基本原则。
```

---

## 8. 原生 MTP 不自动保证采样无损

这是一个很重要的概念。

MTP 只说明：

```text
它是一种 Draft 生成机制
```

并不意味着：

```text
它天然满足无损采样
```

### Greedy 解码

如果使用：

```text
Target argmax
```

并且只接受：

```text
draft_token == target_argmax
```

那么可以保持 greedy 输出一致。

### Sampling 解码

如果 Target 使用概率采样，仅仅比较：

```text
draft_token == target_token
```

并不能保证严格保持 Target 分布。

要实现无损 sampling，需要：

```text
rejection sampling
```

即使用：

```text
min(
    1,
    p_target(x) / p_draft(x)
)
```

并在拒绝后从 residual distribution 采样。

因此：

```text
MTP 是 Draft 结构，
Rejection Sampling 是分布校正方法。
```

两者是两个不同层次的问题。

---

## 9. 原生 MTP 的训练要求

### 外部 Draft Model

通常可以独立训练：

```text
训练 Draft Model
    ↓
部署 Draft + Target
```

只要 Draft 和 Target 的输出空间兼容，就可以使用。

### MTP

MTP Head 必须针对 Target 训练：

```text
Target Backbone
    +
MTP Head
```

训练数据通常需要构造未来 token 的监督目标：

```text
位置 t：
预测 x_{t+1}

位置 t+1：
预测 x_{t+2}

位置 t+2：
预测 x_{t+3}
```

可以表示为：

```text
L_MTP
=
L_1(x_{t+1}, p_1)
+
L_2(x_{t+2}, p_2)
+
L_3(x_{t+3}, p_3)
```

其中：

```text
p_i：
第 i 个 MTP Head 的预测分布
```

MTP Head 与 Target Backbone 的匹配程度直接影响：

```text
acceptance rate
accepted tokens per round
```

---

## 10. 为什么外部 Draft 更灵活？

外部 Draft Model 的优势是组件解耦。

可以自由组合：

```text
Draft A + Target B
Draft B + Target C
Draft 量化版本 + Target 全精度版本
```

还可以把 Draft 放在：

```text
另一张 GPU
CPU
边缘设备
独立服务进程
```

适合：

```text
快速替换 Draft
实验不同模型组合
保持 Target 模型不变
```

但代价是：

```text
通信开销
显存开销
Cache 同步
Tokenizer 对齐
调度复杂度
```

---

## 11. 为什么原生 MTP 更适合模型厂商？

MTP 与 Target 一起训练、一起发布，部署时更紧凑：

```text
一个 Target checkpoint
    +
MTP 参数
```

相比：

```text
Target checkpoint
    +
Draft checkpoint
    +
两套运行时
    +
两套 Cache
```

原生 MTP 的优势：

```text
模型协同优化
hidden state 直接共享
不需要额外模型通信
更容易做 kernel 融合
调度链路更短
```

缺点：

```text
只能服务于对应 Target
MTP Head 需要额外训练
模型结构耦合
难以直接替换 Draft
```

---

## 12. MTP 与 Medusa 的工程区别

### Medusa

```text
Target hidden state
    ├── Head 1
    ├── Head 2
    └── Head 3
```

特点：

```text
多头并行
候选树天然明显
头之间相对独立
需要 Tree Attention
```

### MTP

```text
Target hidden state
    ↓
MTP Head 1
    ↓
MTP Head 2
    ↓
MTP Head 3
```

特点：

```text
串行接力
通常生成线性 Draft block
需要维护 Head Chain 状态
重点是 Cache 提交
```

一句话：

```text
Medusa 主要增加“宽度”。

MTP 主要增加“预测深度”。
```

---

## 13. MTP 与 EAGLE 的工程区别

### EAGLE

```text
Target hidden feature
    ↓
EAGLE Draft Network
    ↓
hidden feature prediction
    ↓
candidate tree
```

EAGLE 通常有：

```text
Draft Network
Draft KV Cache
Tree Mask
Tree Position IDs
Candidate Path
```

### MTP

```text
Target hidden state
    ↓
MTP Head Chain
    ↓
linear draft block
```

MTP 通常更依赖：

```text
MTP Module Cache
Head-specific state
Commit Verify Cache
```

可以这样理解：

```text
EAGLE：
通过一个小 Draft Network 预测 feature

MTP：
通过多个专门 Head 预测未来 token
```

---

## 14. 调度复杂度对比

### 外部 Draft

```text
调度对象：
Draft Model
Target Model

需要同步：
Draft input
Target input
Draft Cache
Target Cache
```

### 原生 MTP

```text
调度对象：
Target Backbone
MTP Modules

需要同步：
Target Cache
MTP Cache
MTP hidden state
context length
```

外部 Draft 的复杂度主要来自：

```text
两个模型之间的协调
```

MTP 的复杂度主要来自：

```text
同一模型内部多个状态的提交与回滚
```

---

## 15. 失败模式对比

### 外部 Draft 常见问题

```text
Draft 与 Target tokenizer 不一致
Draft 和 Target 词表不一致
Draft Cache 与 Target Cache 长度不同
Draft 运行时间太长
Draft acceptance rate 太低
```

### MTP 常见问题

```text
MTP Head 之间状态错位
accept_length off-by-one
fallback 没有写入 MTP 状态
Target Cache 与 MTP Cache 长度不一致
训练时 teacher forcing，推理时误差累积
```

当前 Qwen 工程中，尤其需要检查：

```text
qwen_mtp_engine.py
qwen_mtp_process.py
qwen_mtp_module.py
```

三者对下面变量的理解是否一致：

```text
accept_length
context length
draft length
verify length
fallback token
```

---

## 16. 选择哪一种更合适？

### 适合外部 Draft Model 的情况

```text
已有成熟 Draft Model
需要灵活替换 Draft
Target 不方便重新训练
想快速做实验
希望 Draft 与 Target 独立部署
```

### 适合原生 MTP 的情况

```text
可以修改并训练 Target
希望降低 Draft 运行成本
希望共享 Target hidden state
追求更紧密的硬件优化
模型厂商能够同时发布 MTP 权重
```

### 适合 Medusa 的情况

```text
希望并行生成多分支候选
希望减少 Draft 串行依赖
可以接受 Tree Attention 工程复杂度
```

### 适合 EAGLE 的情况

```text
希望利用 hidden feature
愿意训练轻量 Draft Network
希望提升长链候选质量
可以管理 Draft Tree 和 Feature Cache
```

---

## 17. 四种方法放在一张图中

```text
经典 Draft Model：

小 Draft Model
    ↓
线性 draft tokens
    ↓
Target Verify


MTP：

Target Backbone
    ↓
MTP Head 1 → Head 2 → Head 3
    ↓
线性 draft tokens
    ↓
Target Verify


Medusa：

Target hidden state
    ├── Head 1
    ├── Head 2
    └── Head 3
    ↓
候选树
    ↓
Target Tree Verify


EAGLE：

Target hidden feature
    ↓
EAGLE Draft Network
    ↓
Feature autoregression
    ↓
候选树
    ↓
Target Tree Verify
```

---

## 18. 最终结论

需要记住：

1. MTP 和普通 Speculative Decoding 共享同一个基本框架：

```text
Draft → Verify → Commit/Rollback
```

2. 最大区别是 Draft 来源：

```text
普通 Speculative Decoding：
独立 Draft Model

MTP：
Target 内部的预测模块
```

3. MTP 共享 Target Backbone，因此 Draft 更贴近 Target；
4. 外部 Draft 更灵活，但需要管理两套模型和两套 Cache；
5. MTP 通常具有串行 Draft Chain；
6. Medusa 更偏向并行多头；
7. EAGLE 更偏向 hidden feature autoregression；
8. MTP 本身不等于 rejection sampling，也不自动保证采样分布无损；
9. 工程上最重要的是保证：

```text
input_ids
Target Cache
MTP Cache
hidden state
context length
```

在每轮 Commit 后保持完全一致。

至此，MTP 阶段的四个核心问题已经覆盖：

```text
MTP Head 如何预测
MTP Draft Chain 如何接力
Verify Cache 如何提交与回滚
原生 MTP 与外部 Draft 如何区分
```

下一阶段按照路线进入：

**阶段 6：DFlash / DFlash2——Block-level Draft、固定 Verify Block 与多状态回滚。**