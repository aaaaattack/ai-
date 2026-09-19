# AI/NPU 代码结合学习地图

## 目标

把已有材料变成同一条由浅入深的学习链路，而不是同时阅读三个项目：

```text
CS336 课后笔记（先恢复概念和公式）
→ nano-vLLM（用约 1200 行的实现看概念怎样落到代码）
→ Houmo ModelZoo（确认量化、编译、运行、测量在真实 NPU 产品中的位置）
```

三者不是替代关系：CS336 回答“为什么”；nano-vLLM 回答“最小实现怎样做”；ModelZoo 回答“产品链路里在哪里测、如何验证”。因此学习时优先从左到右，不直接从大型工程倒推概念。

## 材料边界与使用原则

本地图只引用下列本地材料，不修改它们：

- `E:\cs336_note_and_hw-main\cs336_note_and_hw-main\课后笔记`：个人 CS336 学习笔记；用于复习、找回术语和已学过的推导。
- `E:\cs336_note_and_hw-main\nano-vllm`：极简 LLM 推理引擎；用于逐函数追踪。
- `E:\cs336_note_and_hw-main\houmo-examples-xh2`：ModelZoo；用于观察真实部署的输入、模型、性能数据和验证工具。

每个学习单元只产出一张小的“证据卡”，而不是尝试读完整个仓库。证据卡固定回答四件事：

```text
1. 这段代码/日志解决了什么问题？
2. 输入和输出的 shape、dtype 或指标是什么？
3. 它对应哪一个课程概念？
4. 若结果异常，第一项可验证的假设是什么？
```

第一轮以阅读、检索和手工追踪为主；不要为了学习而立即下载模型、编译或运行设备基准。涉及模型下载、芯片、公司运行环境或大规模编译时，先确定环境、数据和设备均可安全使用后再执行。

## 课程模块到代码的映射

| 当前课程模块 | CS336 笔记：概念锚点 | nano-vLLM：最小实现锚点 | ModelZoo：真实链路锚点 | 本节完成证据 |
|---|---|---|---|---|
| L2.1 Q/K/V、Attention shape | `05_Attention.md`、`02_RoPE.md`、`03_SwiGLU.md` | `models/qwen3.py`、`layers/attention.py`、`layers/rotary_embedding.py` | `apis/inferences/qwen3/README.MD` | 写出一次 `X → QKV → RoPE → Attention → o_proj` 的 shape 流，并标出 Q 头与 KV 头可不同 |
| L2.2 MHA/MQA/GQA、KV Cache | `05_Attention.md`、`07_Inference.md` | `layers/attention.py`、`engine/model_runner.py` | Qwen3 推理样例与 prefix-cache demo | 说明 GQA 为什么减少 KV 存储；指出一次 token 写入 cache 的位置 |
| L3 量化 | `生产_01_量化基础.md` 到 `生产_05_KVCache量化.md` | 先不强行从 nano-vLLM 找量化实现；它主要用作未量化推理的性能基线 | `hmodel` 下的量化/导出示例、Qwen 模型配置与精度评测工具 | 从一种量化配置中辨认权重/激活精度、校准数据和精度验收指标 |
| L4 TOPS、带宽、Roofline | `08_FlashAttention.md`、`09_Profiling.md` | `bench.py`、`layers/attention.py`、`engine/model_runner.py` | `tools/computing_perf`、`tools/bandwidth_perf`、`tools/hm_check` | 用实测算力、读/写带宽画出自己的简化 Roofline，并判断一个瓶颈假设 |
| L5 Compiler / Runtime | `TCIM_Runtime_Tensor_Buffer动态形状分析.md`、`推理性能优化系统学习路线.md` | 观察引擎如何把 Python 侧 token/shape 准备为模型调用 | `hmatc`、`hmodel`、Qwen3 样例的模型准备、C++ 推理入口 | 画出“权重/模型准备 → 编译产物 → runtime 调用 → 输出”的链路，记录每一步的输入输出 |
| L6 推理引擎与调度 | `07_Inference.md`、`10_ContinuousBatching.md`、`11_PagedAttention.md`、`12_ModelRunner.md`、`13_EndToEnd.md` | `engine/llm_engine.py`、`scheduler.py`、`block_manager.py`、`sequence.py` | `apis/inferences/qwen3` 的 prefix cache / 多 batch / pipeline / speculative 示例，`tools/llm_perf` | 追踪一个请求从 waiting 到 finished；比较 prefill 与 decode 的调度单位 |
| L7 精度 Debug | 量化笔记、`09_Profiling.md` 中的观测方法 | 用未量化 PyTorch 输出作为概念上的参考点 | `tools/hmeval` 与模型评测/量化产物 | 为“量化后回答变差”列出 First Bad Stage、指标和单变量实验 |
| L8 性能 Debug | `09_Profiling.md`、`08_FlashAttention.md` | `bench.py`、engine 与 attention 中 prefill/decode 分支 | `tools/llm_perf`、`tools/tcim_perf`、硬件性能工具 | 将 TTFT、TPOT、prefill 吞吐、decode 吞吐和 E2E 分层；给出下一步测量 |
| L9 综合方案 | `对比_nano-vllm_vs_HLIEvLLM.md`、`附录_HLIEvLLM引擎层实现分析.md` | 以极简引擎作为设计参照 | 以一个 Qwen3 样例作为落地对象 | 写一页部署方案：目标、约束、量化、性能预算、精度验收和风险 |

## 当前正在学习的 L2.1：第一条代码闭环

这一课不需要运行模型。完成下面三个短任务即可把抽象的 shape 变成可检验的代码理解。

### 步骤 1：用自己的笔记恢复公式（15 分钟）

阅读 `课后笔记\05_Attention.md` 的“核心公式、切分多头、RoPE、工程版 Attention”部分。只抄下并理解：

```text
X [B,S,H]
→ Q/K/V [B,S,H]
→ split + transpose [B,A,S,D]
→ scores [B,A,S,S]
→ output [B,A,S,D]
→ merge [B,S,H]
```

### 步骤 2：在 nano-vLLM 中定位同一条链（20 分钟）

依次看：

1. `nanovllm/models/qwen3.py` 的 `Qwen3Attention`：定位合并的 QKV 投影、Q/K/V 切分、RoPE、Attention 调用和输出投影。
2. `nanovllm/layers/attention.py`：观察它没有手写 `QK^T → softmax → V`，而是在 prefill 与 decode 选择不同的 Flash Attention 路径。
3. 把“合并 QKV 投影”和“Q/K/V 头数可能不同”记为后续 GQA 的问题，不要求这一课立即掌握实现细节。

### 步骤 3：把它对应到真实推理样例（10 分钟）

查看 `houmo-examples-xh2\apis\inferences\qwen3\README.MD` 中 Qwen3 的 prefill、decode、TTFT、TPOT 与 prefix cache 演示。此处只确认两件事：

- prefill 和 decode 被当作不同阶段计量，不能只看总耗时；
- prefix cache 主要减少重复 prefill，并不等价于让每个 decode token 都更快。

本课答完原有四道 shape 题后，再补充下面这张代码观察卡即可，不新增考试负担：

```markdown
### L2.1 代码观察卡

- QKV 在 nano-vLLM 中的切分位置：
- RoPE 应用于：
- Prefill 与 decode 的 Attention 路径分别是：
- `scores [B,A,S,S]` 在工程实现中为什么不显式长期保存在 HBM：
```

## 四阶段学习节奏

### 阶段 A：恢复模型直觉（现在至 L2 完成）

以 CS336 笔记为主，nano-vLLM 为代码验证。每节只追一条数据流，不运行。重点是 shape、RoPE、RMSNorm、SwiGLU、GQA 与 KV Cache；这是后续量化和性能判断的共同语言。

### 阶段 B：把数字概念连接到模型（L3/L4 复盘）

将已自学的量化、TOPS、带宽和 Roofline 落到 ModelZoo 的量化配置、`computing_perf` 与 `bandwidth_perf`。特别要区分“理论峰值”“工具实测峰值”“真实模型吞吐”三个不同量，不能互相替代。

### 阶段 C：理解部署和引擎（L5/L6）

用 nano-vLLM 的 `LLMEngine → Scheduler → ModelRunner → Attention` 建立小而完整的心智模型，再用 ModelZoo 的 Qwen3 推理、性能工具和 C++ 样例识别产品中对应的边界。此阶段重点是 KV block、prefix cache、continuous batching、输入准备、模型调用与性能统计。

### 阶段 D：带着问题做真实观测（L7/L8/L9）

只在有可用设备与可运行环境时执行最小实验：先记录基线，再一次只改变一个变量（量化精度、batch、上下文长度或是否启用 cache），以精度和性能两组证据给出结论。运行结果写回学习进度，不写入公司代码仓库。

## 推荐的第一个小项目：Qwen3 推理路径解剖

目标不是改代码，而是在 60～90 分钟内完成一张从模型结构到部署指标的对照图。

```text
CS336 Attention 公式
  → nano-vLLM Qwen3Attention（QKV / RoPE / Flash Attention）
  → nano-vLLM ModelRunner（prefill / decode 的输入准备）
  → nano-vLLM Scheduler + BlockManager（请求与 KV block）
  → ModelZoo Qwen3 README（TTFT / TPOT / prefix cache 结果）
```

完成标准不是复述源码，而是能独立回答：

1. 为什么 `H=A×D` 是 Attention 代码的形状约束？
2. 为什么 prefill 会受 `S²` Attention 影响，而 decode 的典型瓶颈更接近 KV Cache 读写？
3. 为什么 prefix cache 主要改善重复上下文的 TTFT？
4. 如果 TPOT 异常，应该先看调度、KV Cache、模型执行还是采样？为什么？

## 何时开始运行工具

| 目的 | 首选材料 | 可以运行的最小对象 | 前置条件 |
|---|---|---|---|
| 理解逻辑 | CS336 + nano-vLLM | 不运行，仅读代码 | 无 |
| 熟悉量化/编译入口 | ModelZoo | 查看配置、帮助信息或已有日志 | 不下载模型、不写共享目录 |
| 测硬件上限 | `computing_perf` / `bandwidth_perf` | 单项性能工具 | 已确认 Linux、设备可用、测试不影响他人 |
| 测真实 LLM | `llm_perf` / Qwen3 样例 | 小模型、固定 prompt、少量输出 token | 已确认模型、runtime、设备和输出目录 |

完成每次实际运行后，必须记录命令、模型/精度、输入 shape、环境、预热方式、指标和对照组；否则数值不能用来解释性能差异。

## 后续如何进入课程

- 当前先完成第 06 课的四道 shape 题和“L2.1 代码观察卡”。
- 通过后学习 MHA、MQA、GQA 与 KV Cache；届时直接追 `nano-vllm` 的 `attention.py`、`model_runner.py`、`block_manager.py`，而不是新找材料。
- L3、L4 的理论已自学，等进入 L7/L8 时再用 ModelZoo 工具做有目的的验证；避免过早把时间花在环境与模型准备上。
