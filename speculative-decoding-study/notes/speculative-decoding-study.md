# 投机解码系统学习路线

## 目标

围绕当前 `qwen3.5` 工程，系统掌握：

1. Speculative Decoding 的理论与无损接受规则
2. Draft Model、MTP、Medusa、EAGLE、DFlash 的架构差异
3. Target verify、accepted prefix、bonus token 的实现
4. KV cache、hidden state、recurrent/conv state 的提交与回滚
5. vLLM、SGLang、TensorRT-LLM 中的生产级调度和硬件优化

## 推荐学习顺序

### 阶段 0：先读当前工程

工程目录：`/workspace/imodelzoo/models/llm/qwen3.5`

建议顺序：

1. `qwen_dflash_engine.py`：看一轮 draft → verify → commit 的调度
2. `qwen_dflash_process.py`：看 draft 选择和 verify 接受规则
3. `qwen_dflash_module.py`：看 HMM 图、KV cache 和状态回滚
4. `qwen_mtp_engine.py`：对比 MTP 的逐 token draft 流程
5. `python/demo_dflash.py`、`python/demo_mtp.py`：看模型路径和运行参数

重点回答四个问题：

- Draft token 是怎么产生的？
- Target 如何一次验证多个位置？
- 第一个不匹配位置如何处理？
- 被拒绝 token 的 cache/state 如何回滚？

### 阶段 1：理论基础

阅读：`papers/01_leviathan_speculative_decoding.pdf`

重点：

- Draft model / Target model
- greedy accept-if-equal
- rejection sampling
- 无损解码的条件
- acceptance length 与加速比

配套源码：`repos/llama.cpp/docs/speculative.md`

### 阶段 2：基础工程实现

学习 `llama.cpp` 的传统 draft-model speculative decoding：

- 先生成 draft block
- Target 批量验证
- 接受最长连续前缀
- 处理拒绝后的 fallback
- 更新 KV cache

### 阶段 3：多头和候选树

阅读：`papers/02_medusa.pdf`

源码：`repos/Medusa`

重点：

- 多个 decoding heads
- candidate tree
- tree attention
- longest accepted prefix
- Medusa 与独立 Draft Model 的差异

### 阶段 4：Hidden-state Draft

源码：`repos/EAGLE`

重点：

- 为什么使用 Target hidden feature
- EAGLE-1、EAGLE-2、EAGLE-3 的演进
- tree-based candidate generation
- 如何提高 acceptance rate
- EAGLE 与 MTP、DFlash 的关系

### 阶段 5：MTP

对照当前工程：

- `qwen_mtp_engine.py`
- `qwen_mtp_module.py`
- `qwen_mtp_process.py`

重点：

- MTP head 如何预测未来 token
- 为什么 MTP 通常仍有 draft 串行链
- `commit_verify_cache()` 如何提交接受结果
- 原生 MTP 与外部 Draft Model 的工程区别

### 阶段 6：DFlash / DFlash2

对照当前工程和 `repos/speculators`：

- block-level draft
- candidate_ids / first_scores / transition_scores
- 固定 verify block
- noise token padding 与 attention mask
- target cache rollback
- draft context commit
- acceptance rate 与 accepted-per-round

### 阶段 7：生产级 Runtime

依次阅读：

1. `repos/speculators`：训练、保存、评测 speculator 的统一框架
2. `repos/sglang`：多请求、scheduler、overlap、adaptive speculative decoding
3. TensorRT-LLM 官方文档：编译图、CUDA Graph、kernel 和 runtime 优化；源码仓库体量较大，按需单独拉取
4. `repos/vllm`：paged KV cache、batch speculative decoding 和服务化

## 建议的源码阅读任务

### 任务 A：手工验证接受规则

用一个小词表构造：

```text
draft  = [d1, d2, d3, d4]
target = [d1, d2, x3, ...]
```

确认最终输出是：

```text
[d1, d2, x3]
```

### 任务 B：画 cache 状态图

标记以下状态何时写入、何时提交、何时回滚：

- Target KV cache
- Draft KV cache
- hidden state
- conv cache
- recurrent state
- context length

### 任务 C：做 acceptance-rate 实验

分别记录：

- speculative rounds
- proposed/draft tokens
- accepted draft tokens
- acceptance rate
- accepted tokens per round
- draft time
- verify time
- end-to-end tokens/s

不要只看 acceptance rate；最终速度还取决于 Draft、Verify、同步和 cache 管理开销。

## 论文与源码索引

| 编号 | 主题 | 本地位置 |
|---|---|---|
| 01 | 基础 Speculative Decoding | `papers/01_leviathan_speculative_decoding.pdf` |
| 02 | Medusa 多头和候选树 | `papers/02_medusa.pdf` |
| 03 | EAGLE 系列 | `repos/EAGLE`，论文链接见 `papers/README.md` |
| 04 | MTP | 当前工程 `qwen_mtp_*`，以及 vLLM/SGLang 文档 |
| 05 | DFlash/DFlash2 | 当前工程 `qwen_dflash_*`，以及 `repos/speculators` |
| 06 | 生产 Runtime | `repos/sglang`；vLLM/TensorRT-LLM 参考官方文档 |

## 推荐最终产出

完成学习后，建议自己实现一个最小版本：

```text
small draft model
    ↓
draft N tokens
    ↓
target batch verify
    ↓
accept longest prefix
    ↓
rollback rejected state
```

然后再把它扩展成：

1. greedy speculative decoding
2. rejection sampling
3. tree-based candidates
4. MTP-style draft
5. block-parallel DFlash-style draft

## 阶段与工程代码对应表

后续笔记中的代码示例应优先对应下面的工程；路径均为对应仓库的相对路径，并在代码块中注明行号。

| 学习阶段 | 对应工程 | 最小代码入口 |
|---|---|---|
| 理论与经典基础 | `repos/llama.cpp` | `examples/speculative-simple/speculative-simple.cpp:187-258` |
| Rejection Sampling | `repos/llama.cpp` | `examples/speculative/speculative.cpp:296-386` |
| Acceptance Length / 统计 | `repos/llama.cpp` | `examples/speculative-simple/speculative-simple.cpp:248-298` |
| Toy 模拟器 | 当前 `work` | `toy_speculative_decoder.py:25-67` |
| MTP | 当前 Qwen3.5 工程 | `qwen_mtp_engine.py:81-106`、`qwen_mtp_process.py:148-169` |
| DFlash | 当前 Qwen3.5 工程 | `qwen_dflash_engine.py:176-191`、`qwen_dflash_process.py:228-245` |
| 多头与候选树 | `repos/Medusa` | `medusa/model/utils.py:32-59`、`258-340`、`436-489` |
| Hidden-state Draft | `repos/EAGLE` | 以对应 EAGLE draft model 与 tree verify 代码为准 |
| 生产级 Speculator | `repos/speculators` | `src/speculators/models/` 与 `docs/user_guide/algorithms/` |
