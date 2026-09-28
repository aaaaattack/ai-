# AI/NPU 学习进度

## GR00T 双案例实战材料（暂存，不占正式课号）

- 已新增 Prefill 补充案例与 P1～P5 实验；Visual 补充 V1～V5 实验。配套 `AI_NPU_GR00T双案例_实战方案与资源清单.md` 记录现有脚本和待补的工程、输入、日志、环境。
- Prefill 历史证据：禁用 PTQ 约 0.999987，基线 PTQ 约 0.930479，方案 B 约 0.989426；新混精导出、单算子收益归因、真实任务和性能验收仍待完成。方案 A 为合成 embedding，且同时改变校准/评测输入，需重新做受控对照。

## LoRA / QLoRA 补充主题（已纳入课程，暂不改变当前学习位置）

- 已确定课程位置：L3 量化基础之后，连接 L2 的 Transformer Linear、L7 精度 Debug、L8 性能和 L9 交付。
- 已新增 [LoRA 与 QLoRA](AI_NPU系统课程_L3_LoRA与QLoRA.md)，覆盖低秩增量、target modules、merge/unmerged、QLoRA dtype、重新量化和部署验收。
- 当前仍是课程材料，尚未绑定具体公司模型或运行环境；正式实战需要 adapter、base checkpoint、target module 配置和四路输出。

## 投机解码专题（已纳入课程，暂不改变当前学习位置）

- 已确定课程位置：L6，在 Prefill/Decode、KV Cache、TTFT/TPOT 之后；与 L8 的性能拆解和 L9 的交付验收连接。
- 已新增 [L6 投机解码与 DFlash 2](AI_NPU系统课程_L6_投机解码与DFlash2.md)，包含从 Blockwise Parallel Decoding、Draft/Verify、SpecInfer、Medusa、EAGLE 系列到 DFlash/DFlash 2 的里程碑，以及 DFlash 2 的 S1～S6 实战任务。
- DFlash 2 的公开工程结构已记录为：block-parallel diffusion drafter + target hidden/KV conditioning + local dynamic convolution + candidate selector；公开 speedup 仅作背景，正式结论需在公司 target、硬件、server 和 checkpoint 上复测。

## 第 01 课｜L0 二进制与 FP16：章节评估

**最终结论：** 已通过夯实评估，可进入 L0.3；保留 bias、FP16 编码/解码的间隔复习。

- 已具备：二进制位权与小数展开、规格化、FP16 bias 正反算、基础 FP16 编码/解码，以及 FP16/BF16 的范围和精度取舍。
- 仍需巩固：将 bias 的编码值与真实指数清晰区分，并把位结构连接到 overflow、underflow 与实际精度 Debug。
- 第 02 课｜L0.3：已通过，进入 L3.1；保留特殊值、subnormal 和舍入误差的间隔复习。
- 已具备：FP16 overflow 范围判断、underflow 与舍入的基本区分、`inf`/`NaN` 的特殊编码、First Bad Tensor 定位和单变量实验原则。
- 仍需巩固：subnormal 与特殊指数编码目前主要在讲解后能够复述，需要在后续课程中重新提取。
- 间隔复习：第 03 课中再次将“指数全 1、尾数非 0”答为 `inf`；该规则需在后续课前继续短测。
- 当前学习：第 03 课｜L3 量化，自学模式。已补齐对称/非对称量化、zero-point、粒度、校准、PTQ/QAT、异常值、混合精度和常见方法的参考内容。
- 考核安排：本阶段按自学处理，不要求完成所有手算题；遇到实际量化任务时再按“First Bad Tensor → 校准/粒度 → 单层混合精度”路径深入。
- L4 自学完成：已阅读峰值算力、阵列利用率、shape/batch、频率、稀疏峰值、带宽、Arithmetic Intensity、Roofline、融合和 layout conversion。
- 当前交互式学习：第 07 课｜L2.2 MHA、MQA、GQA 与 KV Cache。
- 本节目标：能从 `batch_size、sequence_length、hidden_size、num_heads、head_dim` 推导 Q/K/V、attention score 和输出 Tensor shape，为后续 MHA/MQA/GQA 与 KV Cache 做准备。
- 代码学习支架：已建立 `AI_NPU代码结合学习地图.md`。本课完成原有 shape 练习后，追加一张 nano-vLLM Qwen3 Attention 的代码观察卡；暂不运行模型或 ModelZoo 性能工具。
- 第 06 课首次评估：暂不通过。核心题独立完成 2/4；第 1、4 题正确，第 2 题漏掉拆分/转置步骤，第 3 题混淆输入 hidden shape 与 score shape。
- 第 06 课第一次复测：1/4，仅能识别 Decode 使用 `flash_attn_with_kvcache`；`head_dim=hidden_size/num_heads` 计算、`[batch_size, num_heads, sequence_length, head_dim]` 和 `[batch_size, num_heads, sequence_length, sequence_length]` 的维度顺序仍未掌握。
- 第 06 课第二轮复测：3/3 完整正确。已掌握 `head_dim=32`、Q/K/V `[3,4,10,32]`、score `[3,4,10,10]`，并能区分 score 的 query 位置与 key 位置。
- 第 06 课最终结论：已通过。当前进入第 07 课 L2.2：MHA、MQA、GQA 与 KV Cache；先学习三种 Attention 的 head 数关系与分组共享，不一次展开整章。
- 第 07 课第一段检查：已正确判断 `num_heads=32、num_kv_heads=8` 为 GQA，并计算出每4个 Query heads 共享一组 Key/Value。当前学习 GQA 下 Q、K、V 的不同 head 数与 Tensor shape。
- 第 07 课第二段：已通过。已能说明 Q shape `[1,8,5,8]` 中第二维是 Query head 数、最后一维是每个 head 的 `head_dim`；已澄清最后一维是 token 激活特征而非模型参数。当前进入 KV Cache：先理解为何只缓存 K/V，以及 GQA 的 cache head 数为何为 `num_kv_heads`。
- 第 07 课第三段基础检查：已通过。能判断 KV Cache 只保存 K/V，并由“8 个 Query heads 分成 4 组”推导出仅需 2 个 KV heads。已开始结合 nano-vLLM 的 block 化实际存储 layout，下一步确认单层 K/V cache 的逻辑 shape 与容量。
- 第 07 课第三段容量计算：尚未通过。尚不能由 `num_kv_heads、context_length、head_dim` 直接写出单层 K/V cache 的 shape 与容量；已追加公式展开和更小的复测题，暂不进入 Prefill/Decode。
- 第 07 课第三段容量计算：参考答案已提供，待用户结合课程文档自行复习；按用户要求继续进入 Prefill/Decode。第四段聚焦 cache 的写入与读取时机，以及 nano-vLLM 中 `prepare_prefill()`、`prepare_decode()` 和 Attention kernel 的对应关系。
- 第 07 课第四段第一次检查：部分通过。能判断生成阶段是 Decode；尚未掌握当前 Q 会读取“prompt + 已生成 + 当前”全部 K/V，已给出 `100 + 2 + 1 = 103` 的展开与最小复测。
- 第 07 课第四段案例：按用户要求已直接提供并写入完整案例。`prompt_length=4`、已生成1个 token 时，下一步是 Decode，当前 Q 读取 `4 + 1 + 1 = 6` 个 token 的 K/V；文档中已补充从 Prefill 到连续 Decode 的 P1–P4、G1、G2 演进。
- 当前学习：第 07 课第五段｜nano-vLLM 的 block 化 KV Cache。已引入 `block_table`（逻辑 block 到物理 block 的读取目录）与 `slot_mapping`（本次新 token 的写入位置），下一步完成位置10的映射练习。
- 第 07 课第五段第一次检查：尚未通过。已逐步展开 token位置10的映射：`10 // 4 = 2`、`10 % 4 = 2`、`block_table[2] = 14`，即物理 block 14 的 slot 2；下一步用位置5做最小复测。
- 第 07 课第五段补充：已按用户要求直接给出 token位置5的参考答案：逻辑 block 1、slot 1，对应物理 block 3 的 slot 1。第07课的 block 目录/写入概念已完成；容量与映射计算转入后续间隔复习。
- 当前交互式学习：第 08 课｜L2.3 RoPE 位置编码。先理解为什么 Attention 需要位置信息、RoPE 为何只作用于 Q/K，以及 nano-vLLM Qwen3 的对应调用。
- 第 08 课 RoPE 第一段检查：已通过。用户正确回答 RoPE 处理 Q/K、不处理 V，并能说明其用于补充 Attention 的位置信息；下一段进入二维旋转公式与相对位置。
- 第 08 课 RoPE 第二段：已补充二维旋转公式、`(1,0)` 旋转示例、Q/K 点积为何包含相对位置，以及 RoPE 不改变 Tensor shape；等待第二段检查题。
- 第 08 课 RoPE 第二段检查：用户暂不会计算，已记录参考答案 `(1,0) → (0,1)` 与相对位置差 `5-2=3`。同步补充工程掌握层次：日常不要求手撕完整三角函数，但必须掌握 Q/K、shape、position id、KV Cache offset 与 Prefill/Decode 一致性；下一题检查 Decode 首 token 精度下降时的排查优先级。
- 第 08 课 RoPE 第三段：已补充 Prefill/Decode 的 position id 连续性、RoPE cache 与 KV Cache 的区别，以及 position reset、scaling、dtype、slot offset 的典型错误表现。下一题：prompt 长度13、已生成5个 token 时的 current position 与 Q 的 K/V 读取范围。
- L7 实战案例已入库：保留为不占用课号的《GR00T Visual 精度 Debug》补充案例。案例以本次真实排查为主线，强调“首个可观测分叉不等于根因阶段”；现阶段不打断当前第 07 课的 L2 KV Cache 学习。
- L7 补充案例已加入工具链基础：先解释 native、wrap、FX、converter、ONNX/HMONNX 的职责与四路对比的每一跳，再进入 GR00T 案例，避免将工具名当作前置知识。

## 综合摸底试卷第 1 套

**评估结果：** 39/100  
**总体判断：** 工程直觉好于底层计算基础。能够识别部分部署现象并进行基础延迟计算，但在浮点编码、Transformer Tensor shape、量化、峰值算力和 Roofline 上存在明显知识断层，尚不能独立完成完整的精度与性能证据闭环。

| 主题 | 得分 | 当前状态 | 后续训练重点 |
|---|---:|---|---|
| L0 数值与 Tensor | 7/10 | 部分掌握 | 二进制科学计数法、FP16 bias 与编码 |
| L1 PyTorch/ONNX | 8/10 | 基本掌握 | API 实践、hook、动态 shape 与 shape bucket |
| L2 模型结构 | 2/10 | 薄弱 | Q/K/V shape、MHA/MQA/GQA、KV Cache |
| L3 量化 | 0/10 | 尚未掌握 | scale、zero-point、量化/反量化、per-channel、outlier |
| L4 GPU/ASIC/NPU | 0/10 | 尚未掌握 | MAC/Ops/TOPS、带宽、AI、Roofline |
| L5 Compiler/Runtime | 5/10 | 部分掌握 | graph break 成本、fallback 计时、对照实验 |
| L6 AI Infra | 5.5/10 | 部分掌握 | PagedAttention、Prefix Cache、KV Cache 优化 |
| L7 精度 Debug | 3.5/10 | 初步理解 | Stage/Tensor 区分、统计指标、单变量实验 |
| L8 性能 Debug | 3.5/10 | 初步理解 | 利用率解释、bound 判断、最小实验 |
| L9 综合方案 | 4.5/10 | 初步理解 | 真实轨迹评估、验收条件、证据闭环 |

## 已表现较好的能力

- 能计算 Tensor 元素数、存储量和 E2E 延迟比例；
- 理解 `eval()`、`no_grad()`、动态轴和 forward hook 的基本用途；
- 能从日志中识别 fusion、unsupported op、layout conversion 和 CPU fallback；
- 能正确理解 TTFT、TPOT 的基本含义；
- 能发现 NPU FP16 已先于 INT8 出现精度异常；
- 能计算 VLA FP16/W8A8 的端到端延迟及理论频率。

## 当前优先补强顺序

1. L0：二进制科学计数法、FP16 指数与 bias；
2. L3：INT8 对称量化的完整手算；
3. L4：峰值算力、单位换算、Arithmetic Intensity 与 Roofline；
4. L2：Transformer Q/K/V shape 与 KV Cache；
5. L5/L7/L8：用单变量实验建立精度和性能证据闭环。

## 下一轮难度

降低一个台阶，以中等偏基础计算为主，但保留 3 道简单工程应用题。建议仍为 10 题：

- 2 题浮点数与二进制；
- 3 题量化计算；
- 2 题峰值算力与 Roofline；
- 1 题 Transformer shape；
- 1 题精度 Debug；
- 1 题性能 Debug。

当 L0、L3、L4 的计算题正确率达到 80% 后，再恢复综合客户场景训练。
