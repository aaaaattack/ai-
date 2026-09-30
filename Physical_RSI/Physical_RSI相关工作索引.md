# Physical RSI 相关工作索引

更新时间：2026-09-23

说明：Physical RSI（Physical Recursive Self-Improvement）指机器人在真实物理环境中，通过自动复位、执行、验证、失败分析、策略/代码改进和再次验证形成持续改进闭环。本索引区分“前置工作”“核心闭环”“评测基础设施”和“最新进展”，不把一次自动调参或单次 Demo 等同于强 RSI。

## 一、前置工作：AI 自动改进与 World Model

### 1. Eureka：Human-Level Reward Design with Large Language Models
- 类型：论文 / 仿真中的自动奖励设计
- 论文：https://arxiv.org/abs/2310.12931
- PDF：`01_前置工作/Eureka_2310.12931.pdf`
- 贡献：LLM 根据任务和环境接口生成、评估并迭代强化学习 reward function。
- 与 Physical RSI 的关系：建立“自然语言任务 → 训练目标 → 评估 → 修改”的早期闭环，但主要在仿真环境，尚未解决真机复位、安全和物理反馈。
- 建议重点：奖励函数生成、自动评估循环、reward hacking 风险。

### 2. RISE：Self-Improving Robot Policy with Compositional World Model
- 类型：论文 / World Model / imagined rollout
- 论文：https://arxiv.org/abs/2602.11075
- PDF：`01_前置工作/RISE_2602.11075.pdf`
- 贡献：学习可组合 World Model，在想象轨迹中估计优势并改进 robot policy，再用真实任务验证。
- 结果摘要：论文报告动态砖块分拣、背包收纳、盒子关闭任务分别超过 +35%、+45%、+35% 的绝对性能提升。
- 与 Physical RSI 的关系：用低成本 imagined rollouts 减少真实机器人试错，但仍依赖真实反馈校正 World Model。
- 建议重点：compositional World Model、imagined rollout、real-world transfer。

### 3. Voyager：An Open-Ended Embodied Agent with Large Language Models
- 类型：论文 / 技能自动生成与技能库
- 论文：https://arxiv.org/abs/2305.16291
- 项目：https://github.com/MineDojo/Voyager
- 贡献：在 Minecraft 中通过自动课程、技能代码和技能库实现持续探索。
- 与 Physical RSI 的关系：不是物理机器人，但提供“技能生成—执行—反馈—技能积累”的重要软件前身。
- 建议重点：automatic curriculum、skill library、iterative prompting。

### 4. Code as Policies
- 类型：论文 / Code-as-Policy
- 论文：https://arxiv.org/abs/2209.07753
- 贡献：让语言模型生成可调用机器人 API 的控制代码。
- 与 Physical RSI 的关系：为 Agent 修改 robot control program 提供程序化接口。
- 建议重点：API grounding、代码执行安全、策略程序可验证性。

## 二、核心 Physical RSI 闭环

### 5. ENPIRE：Agentic Robot Policy Self-Improvement in the Real World
- 类型：论文 / 真实机器人闭环 / coding-agent research harness
- 论文：https://arxiv.org/abs/2606.19980
- 官方项目：https://research.nvidia.com/labs/gear/enpire/
- PDF：`02_核心Physical_RSI/ENPIRE_2606.19980.pdf`
- 四个模块：Environment（自动复位和验证）、Policy Improvement（策略/训练改进）、Rollout（单机或机器人集群执行）、Evolution（日志分析、文献检索、分支比较和代码演化）。
- 结果摘要：官方项目页报告真实 PushT、插针、扎带等任务最高约 99% pass@8；同时提出 MRU 和 MTU 衡量机器人及 Agent token 的利用效率。
- 工程重点：自动复位、成功判定、视频/状态/动作日志、策略版本、真实硬件安全边界。

### 6. ASPIRE：Agentic Skills Discovery for Robotics
- 类型：论文 / 技能发现 / 持续学习
- 论文：https://arxiv.org/abs/2607.00272
- 官方项目：https://research.nvidia.com/labs/gear/aspire/
- PDF：`02_核心Physical_RSI/ASPIRE_2607.00272.pdf`
- 三个核心组件：闭环执行引擎、持续扩展技能库、Evolutionary Search。
- 结果摘要：论文报告 LIBERO-Pro、Robosuite、BEHAVIOR-1K 任务上的显著提升，并展示仿真技能向不同本体和真实机器人迁移的初步证据。
- 工程重点：失败诊断、修复合成、技能抽象、跨任务复用、sim-to-real。

### 7. RoboRSI
- 类型：项目 / 多 Agent / 移动机器人自进化
- 项目：https://robo-rsi.com/blog/2-roborsi/
- 贡献：在真实移动机器人上尝试自动迭代改进，任务链包括目标搜索、移动接近、抓取、搬运和放置。
- 与 ENPIRE 的差异：更关注长程移动操作、多 Agent 协作和技能层级，而非单一桌面操作。
- 注意：目前公开材料仍属于项目展示和早期工程验证，应区分公开结果、实验条件和可复现程度。

## 三、评测与基础设施

### 8. RoboDojo：Unified Sim-and-Real Benchmark
- 类型：论文 / 仿真与真机统一评测
- 论文：https://arxiv.org/abs/2607.04434
- 官网：https://robodojo-benchmark.com/
- GitHub：https://github.com/RoboDojo-Benchmark/RoboDojo
- PDF：`03_评测与基础设施/RoboDojo_2607.04434.pdf`
- 规模：42 项仿真任务、18 项真实任务，覆盖泛化、记忆、精密操作、长程执行和开放指令。
- 作用：为 Physical RSI 提供固定任务、统一硬件/复位/协议和公共排行榜，防止 Agent 只在固定场景上过拟合。
- 工程重点：held-out 场景、统一 reset、真实任务复现、评测结果审计。

### 9. XPolicyLab
- 类型：开源基础设施 / 策略服务接口
- 项目：https://github.com/XPolicyLab/XPolicyLab
- 作用：为不同策略模型提供统一部署和评测接口，使策略可以接入 RoboDojo 仿真和真机评估。
- 与 Physical RSI 的关系：降低策略替换和跨模型比较成本。

### 10. RoboTwin 2.0
- 类型：论文 / 仿真数据生成 / 双臂操作
- 论文：https://arxiv.org/abs/2506.18932
- 项目：https://github.com/RoboTwin-Platform/RoboTwin
- 作用：通过大规模随机化生成双臂操作数据和评测任务，为 RSI 提供低成本实验环境与数据源。

## 四、最新进展

### 11. ENPIRE 的机器人集群与自动科研
- 项目：https://enpire-research.github.io/
- 进展：从单机器人试验扩展到多机器人并行探索；Agent 可以读取日志、修改控制/训练代码并比较不同实验分支。
- 观察指标：机器人利用率、有效实验数、单位成功率提升成本、失败回归率。

### 12. Physical RSI 的技能库化
- 代表：ASPIRE
- 进展：从“当前任务修复”转向“将修复抽象为可复用 skill”，并尝试跨任务、跨本体和仿真到真实迁移。
- 观察指标：skill reuse rate、跨任务成功率、跨本体迁移成功率。

### 13. World Model + 真机闭环
- 代表：RISE
- 进展：将大量低成本试错放在 imagined environment，真实机器人只验证筛选后的策略。
- 观察指标：World Model 预测误差、imagined-to-real ranking correlation、真实试验节省比例。

### 14. RoboDojo 作为统一考场
- 官网：https://robodojo-benchmark.com/
- 进展：公开仿真和真机评测，减少不同团队自定义场景和精选 Demo 造成的不可比性。
- 观察指标：新场景成功率、随机化鲁棒性、真机/仿真差距、长程任务失败位置。

## 五、建议学习顺序

1. Code as Policies：理解策略如何转化为可执行程序；
2. Eureka：理解 LLM 自动设计训练目标；
3. Voyager：理解技能库和开放式迭代；
4. RISE：理解 World Model 如何降低真实试错；
5. ASPIRE：理解技能发现、修复和复用；
6. ENPIRE：理解真实机器人自动研究闭环；
7. RoboDojo/XPolicyLab：理解统一评测和部署基础设施；
8. RoboRSI：观察长程移动机器人和多 Agent 自进化。

## 六、对 NPU/Runtime 方案的工程启示

Physical RSI 的端侧瓶颈不仅是 TOPS，还包括：

- VLA policy 的稳定时延和动作周期；
- 多摄像头输入与 DMA；
- 失败检测模型和策略模型并行；
- 长时间运行的 Runtime/Driver 稳定性；
- 视频、动作、状态日志的压缩与传输；
- 模型/编译版本回滚；
- 温度、功耗、内存和错误码审计；
- 多机器人并发推理与调度。

建议为 M50/XH2A 建立 Physical RSI 专用指标：policy latency、action cycle jitter、reset success rate、failure-diagnosis latency、tokens per experiment、robot utilization、NPU power per successful trial。

## 七、重要辨析

Physical RSI 目前更准确地称为“受约束的物理世界自动研究闭环”，不等于已经实现无限递归、自主修改目标和跨任务稳定进化。评估时必须区分：

- 任务成功率 vs. 改进过程效率；
- 固定 benchmark 提升 vs. held-out 场景泛化；
- 单次策略修复 vs. 可复用技能；
- 仿真自我改进 vs. 真机自我改进；
- Agent 修改代码 vs. Agent 修改改进机制本身。

## 八、2026-09-24 之后增量整理

详见：[增量整理_2026-09-24至2026-09-29.md](04_最新进展/增量整理_2026-09-24至2026-09-29.md)

- **Physical RSI / 机器人 Agent**：CognitiveReality、RegenHarness。
- **投机解码 / Runtime**：Programmable KV Cache、sampling-mask logprob 对齐、NebulaSD、DPara、H-Spec。
- **边缘 NPU / 系统**：Arm 的计算-存储协同、Qualcomm 端侧 MoE、Fraunhofer RISC-V Secure Element。
- **工程关注点**：KV 生命周期、host/device 状态一致性、Agent trace、secure boot、DMA isolation、模型授权与回滚。

## 九、VLA 投机解码专门资料库

已单独建立：[VLA投机解码](../VLA投机解码/README.md)

- 已开源代码：Realtime-VLA FLASH、KERV、Spec-VLA、SpecPrune-VLA；
- 暂未确认官方完整代码：WA-SpecDec、A3、A2C2；
- π0.5 优先参考：Realtime-VLA FLASH，其连续 action-chunk speculative inference 与 π0.5 的 Flow Matching 路径最接近。

