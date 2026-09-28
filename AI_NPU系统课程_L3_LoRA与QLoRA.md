# L3 补充主题｜LoRA、QLoRA 与量化模型适配

本主题不占正式课号，放在 L3 量化基础之后、真实部署方案之前。它连接 L2 的 Transformer Linear、L3 的量化与混合精度、L7 的精度 Debug，以及 L9 的客户适配交付。

## 一、LoRA 解决什么问题

完整微调大模型需要更新全部权重，显存、优化器状态和训练通信成本都很高。LoRA 冻结原始权重，只训练低秩增量，从而降低可训练参数和优化器状态。

对一个 Linear 权重 W，LoRA 使用：

W' = W + scale × B × A

其中 W 是冻结的基座权重；A 和 B 是可训练的低秩矩阵；rank r 远小于输入/输出维度；通常 scale = lora_alpha / r。

训练时只更新 A、B。常见初始化是 A 使用随机初始化、B 初始化为零，使训练开始时增量接近零，模型先保持基座行为。

## 二、一个形状例子

假设 Linear 的输入维度为 4096，输出维度为 11008，LoRA rank 为 16：

原始权重参数量 = 11008 × 4096
LoRA 参数量 = 16 × 4096 + 11008 × 16

LoRA 训练的参数量远小于完整权重。A、B 的矩阵乘法方向必须跟项目 Linear 的 weight shape 对齐，不能只凭变量名判断；要检查 forward 和 state_dict 的真实 shape。

## 三、LoRA 应该插在哪些模块

常见 target modules 包括 Attention 的 q_proj、k_proj、v_proj、o_proj，以及 MLP 的 gate_proj、up_proj、down_proj。不同任务的敏感模块不同：

- 只给 q_proj/v_proj 加 LoRA：参数更少，适合作为起点；
- 给 q/k/v/o 加 LoRA：注意力适配能力更强；
- 加入 MLP 投影：可能提高任务适配能力，但参数和训练开销增加；
- 不要把 LayerNorm、Embedding 或 lm_head 是否加入当成固定规则，要根据任务、框架和验收结果决定。

LoRA 的插入位置必须和量化、导出、权重加载约定一致。模块名字匹配成功不代表实际 forward 已使用 adapter，应通过输出变化或模块 hook 验证。

## 四、训练后如何部署

LoRA 有两种常见推理形态：

1. Merge：把 B×A 乘出来加回 W，得到一个新的 W'，再按普通模型推理。无额外 LoRA 矩阵乘法，但需要重新量化或重新导出。
2. Unmerged：保留冻结 W 和 A/B，推理时额外计算 LoRA 分支。可以热插拔多个 adapter，但增加计算、访存和调度开销。

量化模型如果先 merge 再量化，与保持量化基座并在运行时叠加 LoRA，数值路径不同，必须分别测试。

## 五、QLoRA 的核心

QLoRA 通常冻结一个低比特量化的基座模型，在计算时将基座权重解量化到计算 dtype，再训练低秩 LoRA 参数。LoRA 参数通常保留在 BF16 或 FP16 等较高精度。

要区分：

- QLoRA 不是把 LoRA 参数也简单量化到同一个低比特格式；
- 基座权重的存储 dtype、计算 dtype、LoRA dtype 和梯度 dtype 是四个需要分别记录的概念；
- QLoRA 训练精度好，不等于部署后的 merged INT8/INT4 模型一定对齐；merge、重新量化和导出仍可能引入误差。

经典 QLoRA 论文讨论了 NF4、double quantization 和分页优化器等方法。具体框架是否支持这些选项，以实际版本实现为准。

## 六、LoRA 与量化精度 Debug 的连接

出现适配后精度下降时，要把问题拆成：

1. 基座模型原始输出；
2. 基座量化模型输出；
3. 加载 LoRA 后、未 merge 的输出；
4. merge 后高精度输出；
5. merge 后重新量化/导出的输出。

如果 3 对齐而 4 不对齐，优先检查 merge 公式、矩阵转置、scale 和 dtype；如果 4 对齐而 5 不对齐，优先检查重新量化、校准和导出；如果从 1 到 2 就掉点，应先处理基座量化误差。

至少记录 base weight、adapter weight、merged weight 的 dtype、shape、范数、最大绝对值，以及 adapter 被命中的模块数量。

## 七、LoRA 与服务性能

LoRA 参数量小，不等于推理开销一定为零：

- merge 后通常没有额外 LoRA kernel，但需要新的权重存储或重新编译；
- unmerged 多一个低秩分支，可能增加小矩阵 GEMM、访存和同步；
- 多 adapter、多租户热切换会增加权重管理、cache 和调度复杂度；
- 在 NPU 上要确认 LoRA 分支算子支持、融合能力、layout 和 fallback。

性能验收要分别测 base、merged、unmerged，以及单 adapter、多 adapter 和目标并发。

## 八、课程实战任务

### L3-LORA-01：手算参数和 shape

给定 Linear 输入 4096、输出 11008、rank 16、alpha 32，计算 A/B 参数量、scale，并说明增量矩阵的 shape。

### L3-LORA-02：读 state_dict 验证 adapter

统计实际命中的 target modules，检查 A/B shape、dtype 和是否加载成功；不能只看 missing_keys=0。

### L3-LORA-03：四路精度对照

固定输入，对比基座、LoRA unmerged、LoRA merged、merged 后量化四种输出，逐层和逐 token 统计 cosine、MSE、max_abs。

### L3-LORA-04：QLoRA 方案选择

根据显存、目标硬件和是否需要热插拔，选择 merge 或 unmerged；说明基座量化、LoRA dtype、重新量化与导出的风险。

### L3-LORA-05：客户交付验收

报告任务指标、模型大小、加载时间、TTFT、TPOT、显存、并发、adapter 切换时间和回退方案。

## 九、进入其他课程的位置

- L2：只在 Transformer Linear 和 Attention 结构处建立 LoRA 插入点的基础认识；
- L3：正式学习 LoRA、QLoRA、量化基座与混合精度；
- L7：把 merge、量化、导出误差放进 First Bad Stage 排查；
- L8：比较 merged/unmerged 的 kernel 和内存开销；
- L9：形成面向客户的适配、版本、验收和回退方案。
