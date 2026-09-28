# L6 补充主题｜投机解码与 DFlash 2

本主题放在 L6「AI Infra 和推理优化」中，接在 Prefill/Decode、TTFT/TPOT、KV Cache 和 Continuous Batching 之后；再与 L8 的性能分析、L9 的部署验收连接。本文件不占正式课号。

## 一、它解决什么问题

普通自回归 Decode 每次由大模型生成一个 token。生成 100 个 token 往往需要大模型串行运行约 100 次。

投机解码加入一个更快的 proposer/drafter：它先猜一小段 token，再让 target model 一次并行验证这段候选。被接受的连续 token 越多，大模型串行步数越少。

普通 Decode：Target → 1 token → Target → 1 token

投机 Decode：Draft k tokens → Target 一次验证 → 接受 r tokens

标准 draft/verify 方案通过拒绝采样可以保持 target 的输出分布；greedy 场景则通常比较 target 与 draft 的 token 是否一致。加速与输出等价必须分别验证。

一个工程近似是：每轮有效产出约等于平均接受 token 数；每轮耗时约等于 draft_time + target_verify_time + sampling/管理开销。投机解码有效的条件是 target 验证多个 token 的成本接近验证一个 token，同时 drafter 足够快且接受率足够高。

## 二、发展里程碑

### 1. 并行验证的早期思想

2018 年 Blockwise Parallel Decoding 探索一次提出多个未来位置、再并行计算的方向，提供了把串行依赖变成短块并行验证的早期系统视角。

### 2. 2022/2023：Draft & Verify 成为通用范式

Leviathan、Kalman 和 Matias 的 Fast Inference from Transformers via Speculative Decoding 将小模型草拟、大模型验证和保持输出分布的采样方法组合起来；论文报告在 T5-XXL/T5X 上约 2–3 倍加速，并强调无需改动目标模型或重新训练目标模型。

Google 的 Accelerating Large Language Model Decoding with Speculative Sampling 随后在 Chinchilla 70B 的分布式场景中报告约 2–2.5 倍加速。

这阶段的关键贡献是证明候选可以批量验证，拒绝采样可以让最终分布保持一致。

### 3. 2023：树状候选和服务系统

SpecInfer 将候选组织成 token tree，用树状并行验证减少大模型的串行步骤，并把投机推理带入服务系统、调度和批处理视角。

Self-speculative 的 Draft & Verify 通过跳过目标模型的部分中间层来快速草拟，再用完整模型验证，减少独立 drafter 的部署成本。

### 4. 2024：减少或移除独立 draft model

Medusa 在目标模型上添加多个 decoding heads，一次预测多个未来位置，并用 tree attention 验证候选。Medusa-1 冻结 backbone 训练新增 heads；Medusa-2 联合训练 backbone 与 heads。

EAGLE 将草拟从 token 层推进到 feature 层，利用 target 中间特征训练轻量 drafter；EAGLE-2 根据上下文置信度动态选择 draft tree，而不是固定树结构。

### 5. 2025：可扩展训练与上下文适应

EAGLE-3 放弃原有 feature prediction 约束，使用多层特征融合和 training-time test，使 drafter 在训练时接近自己的推理状态，并继续提高训练数据规模带来的收益。

工程重点逐渐转向接受率稳定性、动态树调度、batch 场景和 draft/verify 设备流水线。

### 6. 2026：DFlash 与 DFlash 2

DFlash 将 drafter 改成小型 block diffusion 模型：以 anchor 和 mask token 为输入，在一次非因果 block 前向中并行预测候选块；target 再验证并接受最长有效前缀。

DFlash 2 在 DFlash 骨干上加入 local dynamic convolution 和 candidate selector。前者让邻近 draft 位置交换局部信息，但不跨 anchor block 边界；后者先取得每个位置的 unary top-k 候选，再用前驱 token 条件化的低秩转移分数选择候选路径。

公开 Speculators 文档列出的关键参数包括 conv_kernel_size、conv_group_size、selector_rank、selector_top_k、selector_loss_alpha，以及 draft layers、block size、anchor 采样方式和损失配置。不同实现的默认值必须显式记录。

## 三、方法关系

| 方法 | 草拟来源 | 是否逐 token 草拟 | 主要控制量 |
|---|---|---:|---|
| Draft model | 独立小模型 | 通常是 | draft 成本、接受率 |
| Self-speculative | target 的部分层 | 是 | 跳层策略、验证成本 |
| Medusa | target 上的多头 | 各 head 并行 | head 训练、树宽 |
| EAGLE 系列 | learned drafter / feature | 取决于实现 | 特征融合、动态树、训练方式 |
| DFlash | block diffusion drafter | block 内并行 | block size、anchor、验证率 |
| DFlash 2 | DFlash + 局部卷积 + selector | block 内并行 | kernel/group、selector top-k/rank |

## 四、必须测的指标

至少记录 draft_time、target_verify_time、acceptance_rate、mean_accepted_length、tokens_per_speculation_round、rejected_token_ratio、额外 KV cache/workspace、TTFT、TPOT、E2E latency，以及不同 batch 和上下文长度下的吞吐。

常见误判：接受率高但 drafter 过慢；单请求变快但 batch 吞吐下降；只比较输出文本而没有比较分布；只报平均 TPS 而没有 p50/p95；只验证 draft 输出而没有验证 target 接受路径；漏掉 tree、block 和 selector 的显存。

## 五、课程位置与学习顺序

1. L2 前置：Q/K/V、Causal Attention、KV Cache、Prefill/Decode。
2. L6 基础：TTFT、TPOT、Decode memory-bound、Continuous Batching。
3. 本主题：draft/verify、接受规则、候选树、MTP/Medusa/EAGLE/DFlash。
4. L8 实验：接受率与 TPOT 分解、draft/verify 时间线、batch 与上下文长度对吞吐的影响。
5. L9 交付：质量等价、端到端收益、内存、稳定性和回退策略。

## 六、DFlash 2 实战任务

### S1：建立 vanilla baseline
固定 target、tokenizer、采样参数、prompt 集和输出长度，测普通 Decode 的 TTFT、TPOT、E2E、tokens/s、显存，并记录单请求和目标并发。

### S2：画出一轮数据流
标明 anchor、mask block、target hidden states、DFlash 2 draft、candidate selector、target verify、longest accepted prefix 和 KV Cache 更新点。

### S3：拆 selector 的收益
固定 DFlash 骨干，只比较 unary candidates 与 selector 路径，记录 selector_top_k、平均接受长度、draft latency 和 verify latency。

### S4：消融 block size 和上下文长度
在相同 prompt 集上比较 block size 以及短/长上下文，记录接受长度、target verify 时间、额外 workspace 和 TPS。

### S5：正确性验收
分别测试 greedy 一致性和采样质量，报告接受 token、首个 mismatch 位置、回退 token 和输出分布指标。

### S6：目标硬件验收
检查编译、kernel、同步、KV Cache、并发、p50/p95、温度和长时间稳定性；在 NPU/RK 平台上分别核对 drafter 与 target 的算子支持、内存搬运和 fallback。

## 七、参考来源

- https://arxiv.org/abs/2211.17192
- https://arxiv.org/abs/2302.01318
- https://arxiv.org/abs/2305.09781
- https://arxiv.org/abs/2401.10774
- https://arxiv.org/abs/2401.15077
- https://arxiv.org/abs/2406.16858
- https://arxiv.org/abs/2503.01840
- https://arxiv.org/abs/2602.06036
- https://docs.vllm.ai/projects/speculators/en/stable/user_guide/algorithms/dflash2/

## 八、工程备注

公开资料足以支撑概念、结构和实验指标教学；公司适配仍需要 target/drafter 模型、硬件后端、server 分支、checkpoint、编译日志和基线测量。公开文档明确指出实现仍在发展中，部分硬件配置尚未验证。公开 speedup 只能作背景，不能替代公司测量。
