# L7 补充实战案例｜GR00T Visual 精度 Debug

本案例不编号，留待后续实战学习。配套：[双案例实战方案与资源清单](AI_NPU_GR00T双案例_实战方案与资源清单.md)。文中历史指标来自用户排查记录，课程整理时未重新运行原环境；实际练习见末尾 V1～V5。

## 一、案例目标

本课使用一次真实的 GR00T visual 子图量化精度问题，学习如何从业务现象建立可证伪的证据链，而不是看到“量化后精度掉了”就把原因归给量化框架或 NPU。

最终要掌握的闭环：

```text
业务现象
→ 拆分执行阶段
→ 找到首个可观测分叉
→ 排除伪线索
→ 对比中间 Tensor 与模型参数
→ 将根因前移到最早出错阶段
→ 最小修复
→ 重新导出并验收
```

**本案例的最终结论：** visual 精度下降的根因不是 xhquant 或 HMONNX 本身，而是 transformers 5.3 加载模型后触发了二次初始化；项目的 SigLIP 初始化代码用 `.data.fill_()` 覆盖了已从 checkpoint 加载的权重。xhquant 只是忠实导出了已经错误的模型。

## 二、先补齐工具链概念：native、wrap、FX、converter 与 ONNX

本案例会反复出现五个词。先不用记实现细节，只要把它们放在同一条流水线上：

```text
原始 PyTorch 模型
      │ native：按原始模型的 forward 直接运行
      ▼
wrap 后的模型
      │ 为部署/导图适配输入、输出和部分算子表达
      ▼
FX 图 / FX 量化图
      │ PyTorch 可执行的“前向计算图”表示
      ▼
ONNX / HMONNX
      │ 可交给后端推理器执行的导出图
      ▼
硬件或 HMONNX 推理
```

### 2.1 native：原始 PyTorch 模型

`native` 只是“未经部署包装的原模型”。它直接执行模型源码里的 `forward()`，使用 checkpoint 加载后的 PyTorch 参数。

在本案例里，native 是最初的数值参照，但它不是绝对真理：如果 checkpoint 加载后已被代码覆盖，native 本身也会错。因此还需要把 native 参数与 checkpoint 文件直接比较。

### 2.2 wrap：给部署流程套一个适配层

`wrap` 可理解成给原模型套上一个“适配器”。它通常把原本复杂或不固定的调用方式，改成更适合 tracer、量化和导出的形式，例如：

- 固定或明确输入输出 Tensor；
- 将模型内部的特殊调用包装成可识别模块；
- 补充部署所需配置，如 batch、token 长度或 cache 相关开关。

它**不是**量化本身，也不等于 ONNX。由于某些项目中的 wrap 会原地修改模块，实验中 native 必须先于 wrap 运行，避免比较基准被污染。

### 2.3 FX：PyTorch 的计算图表示

FX（Torch FX）可以把 PyTorch 前向过程记录为一张图：节点代表模块调用或算子，边代表 Tensor 流动。它仍然可以在 PyTorch 中运行。

```text
PyTorch 模块 + 输入
→ FX trace
→ graph（例如 LayerNorm → Linear → Add）
→ FX GraphModule，仍可用 PyTorch 执行
```

**FX 量化图**是在这张图中加入量化模块、量化参数或目标设备算子后的可执行图。它是检查“量化本身是否引入误差”的关键节点，不等于已导出的 ONNX 文件。

### 2.4 converter：一段转换流程，不是一种图格式

`converter` 在这个工程里是把模型推向部署表示的一组流程/代码。它负责准备与实际推理一致的 dummy 输入，调用 wrap、把 FX 模型转为量化 FX 图，并把量化图导出为 HMONNX。

因此下列关系要分清：

```text
converter：做转换的流程
FX：转换过程中的 PyTorch 图表示
ONNX/HMONNX：转换后的导出图格式/产物
```

### 2.5 ONNX 与 HMONNX：导出后的图和参数

ONNX 是一种跨框架模型表示，包含：

- **node：** 计算节点，如 LayerNorm、MatMul；
- **initializer：** 图中常量参数，如 Linear 权重和 LayerNorm 的 γ/β；
- **输入/输出：** 图的 Tensor 接口。

本工程的 HMONNX 是面向后端推理的 ONNX 导出产物。它不再调用原始 PyTorch `forward()`；推理器按图中的 node 和 initializer 执行。故而只要 initializer 里的 LayerNorm γ/β 已是 `1/0`，即便导出器工作完全正确，推理结果仍会错误。

### 2.6 本案例里每一跳各回答什么问题

| 对比 | 要验证的事情 |
|---|---|
| checkpoint → native 参数 | 权重是否加载正确、是否被加载后代码覆盖 |
| native → wrap | 适配层是否改变语义 |
| wrap → FX | 图转换与量化是否引入误差 |
| FX → HMONNX | 导出图和后端推理是否保持 FX 的结果 |

后文的四路 cosine 不是“比较四个陌生名词”，而是在对这四个问题逐一做实验。

## 三、问题现象与约束

917 的 GR00T visual 子图量化后，与量化前结果的 cosine 约为 `0.47`，业务仿真无法对齐浮点模型。

这类问题至少可能来自：

- 模型加载或权重映射错误；
- wrap 改变前向语义；
- FX 图或量化引入误差；
- ONNX/HMONNX 导出错误；
- dummy 输入与 converter 不一致；
- dump、输出解包、dtype、layout 或 Tensor 配对错误。

因此，第一原则是：**不要由最终 cosine 直接推断根因。**

## 四、第一步：四路对比，定位首个可观测分叉

对同一份 converter dummy，在四个节点记录输出：

```text
native PyTorch → wrap → FX quant graph → HMONNX inference
```

实际结果：

| 子图 / 输出 | native→wrap | wrap→FX | FX→HMONNX | wrap→HMONNX |
|---|---:|---:|---:|---:|
| visual `output_latent` | 0.99997 | 0.99973 | **0.497** | **0.497** |
| prefill `hidden_states` | 0.99969 | 0.930 | **1.000** | 0.930 |
| head_pre | 1.000 | 1.000 | 1.000 | 1.000 |
| head `act_pred` | 0.99998 | 1.000 | 1.000 | 1.000 |

读法：

- visual 的异常**首次被观测到**在 `FX→HMONNX`；
- prefill 的 `wrap→FX=0.930` 是量化误差，但导出结果与 FX 一致，和 visual 问题不同；
- 这一步只能说明应优先检查导出产物与其上游输入，**不能证明根因一定在 exporter**。

### 检查题 1

为什么 visual 的 `FX→HMONNX≈0.497` 只能称为“首个可观测分叉”，而不能立即写成“xhquant exporter 是根因”？

## 五、第二步：排除会污染比较的伪线索

本案例先后排除了以下问题：

1. **强制 dtype 转换：** visual wrap 图实际使用 BF16；dump 若强制 FP16，基准已经变了，但数值差远不足以解释 cosine `0.47`。
2. **dummy 不同：** 必须使用 converter 原样的输入构造方式；OOD dummy 会夸大差异。
3. **QTensor 污染：** converter 会原地改写模块；测量前应删除 policy 并重新加载干净 FP 模型。
4. **四路顺序错误：** wrap 会原地修改模块，所以必须先跑所有 native，再跑 wrap/FX。
5. **输出没有剥到 Tensor：** HF `ModelOutput`、dict、tuple 都可能嵌套，比较前需要递归取真正输出 Tensor。
6. **layout / token 重排猜测：** 三路输出 shape 都是 `(1,324,1152)`；逐 token 重排不能恢复 cosine。

进一步的形态证据是：按通道拟合缩放后 cosine 从 `0.497` 升到 `0.975`。这更像通道仿射参数错误，而不是 padding、pack 或 token 顺序问题。

## 六、第三步：中间 golden 与 ONNX initializer 取证

HMONNX 的最后输出与 `identity LayerNorm(add_53)` 的 cosine 为 `0.99999`。这说明输出近似使用了：

```text
γ = 1，β = 0
```

而 checkpoint 中 visual LayerNorm 的仿射参数本应明显非 identity，例如：

- `post_layernorm.weight` 相对 1 的 MAE 约 `0.246`；
- 第 0 层 `layer_norm1.weight` 的均值约 `0.47`；
- 第 0 层 `layer_norm2.weight` 最大可到 `14.6`。

导出的 917 visual ONNX 中，55 个 LayerNorm 的 γ/β 却全为 `1/0`；对照工程在相同命名下保留了真实权重。

这里得到强证据：**HMONNX 的 LayerNorm 参数确实错误，但还要继续追问“这些参数最早何时变坏”。**

## 七、第四步：把根因前移到模型加载

加载探针直接比较 safetensors checkpoint 与 `AutoModel.from_pretrained()` 后的 `named_parameters()`：

```text
missing = 0，unexpected = 0，mismatched = 0
917 loaded visual LayerNorm：54/54 为 identity
checkpoint LayerNorm vs loaded LayerNorm cosine ≈ 0.518
checkpoint FC1 / patch embedding vs loaded 也不一致
```

加载后的参数与 checkpoint 不一致是直接证据。`missing=0` 只表示没有报告缺失键，单独不能证明参数数值正确，也不能证明覆盖的发生时刻。结合初始化代码路径与修前/修后对照，用户记录将根因定位到加载收尾阶段的覆盖行为。

历史四路表还有一项必须补齐的来源检查：若 native/FX 已使用同一份错误权重，正确导出它们通常应继续保持一致。表中的 FX→HMONNX 大差异需要由各轮加载实例、旧产物或执行路径的差别进一步解释。实战中记录每路权重、输入和产物的版本，不能只以“首个可观测分叉”掩盖这个未解释点。

## 八、第五步：代码级根因与最小修复

在 transformers 5.3 中，加载收尾会进入初始化路径。GR00T 的 `is_remote_code()` 为 false，visual 子树没有被正确排除。SigLIP 的 `_init_weights` 包含：

```python
module.bias.data.zero_()
module.weight.data.fill_(1.0)
```

`.data.*` 会绕过 Transformers 对“已加载参数不得重新初始化”的保护，因此真实 LayerNorm γ/β 被写成 identity；同类初始化还会影响 Linear 和 patch embedding。

修复原则不是改 LayerNorm 公式、网络结构或量化配置，而是把初始化改为遵守已加载参数保护：

```python
from transformers import initialization as init

if module.bias is not None:
    init.zeros_(module.bias)
if module.weight is not None:
    init.ones_(module.weight)
```

同样把 `normal_`、`xavier_uniform_`、自定义 `lecun_normal_` 等 `.data` 或裸 Tensor 初始化替换为受保护的 `init.*` 实现。这样，缺失权重仍能初始化；已经从 checkpoint 加载的权重会保持不变。

## 九、第六步：重新导出与验收

修复代码后，旧 HMONNX 不会自动变好，必须重新 PTQ 并导出 visual 子图。验收应同时满足：

| 验收点 | 修复后目标 / 实际结果 |
|---|---|
| checkpoint vs loaded visual 权重 | 一致 |
| visual ONNX LayerNorm γ/β | 非 identity，且与 checkpoint 对应 |
| visual FX→HMONNX | `1.000` |
| visual wrap→HMONNX | 约 `0.995`，仅保留量化误差 |
| 仿真业务结果 | 验收目标：与浮点重新对齐；当前修后记录未给出对应任务成功率，待补日志 |

注意：`wrap→FX≈0.995` 属于已知量化误差；它和本案例修复的“导出产物错误”是两类问题，不能混为一谈。

## 十、证据脚本地图

以下脚本是本案例的证据材料，不修改其内容：

| 目的 | 脚本 |
|---|---|
| 检查加载后权重是否仍等于 checkpoint | `example_gr00t/_tmp_probe_visual_ln_load.py` |
| 对齐 checkpoint、ONNX initializer 与中间 golden | `example_gr00t/_tmp_visual_ln_vs_ckpt.py` |
| 在 native、FX、export graph、ONNX 间追 LayerNorm 参数 | `example_gr00t/_tmp_probe_visual_ln_export.py` |
| 判断通道仿射异常还是 layout 异常 | `example_gr00t/_tmp_visual_fx_hmonnx_probe.py` |
| 排除 prefill 自身的 FX/PTQ 误差 | `example_gr00t/_tmp_prefill_fx_ptq_ablation.py` |

## 十一、综合练习：写一份可交付的根因结论

请用不超过 8 句话向三个对象说明本案：

1. 给模型仓库维护者：说明代码根因和最小修复；
2. 给量化框架维护者：说明为何 xhquant 不是根因、仍应保留什么诊断能力；
3. 给客户：说明修复范围、为何必须重新 PTQ、以及如何验收。

通过标准：能区分“首个可观测分叉”与“根因阶段”，并至少列出两个独立证据支撑结论。

## 十二、实战训练：从读源码到验证修复

每次推进一个实验。先补实验需要的概念，再读关键函数，预测输出、执行并解释。完整环境不足时先用配对 Tensor 和参数切片做离线练习；完整执行要求见资源清单。

| 任务 | 概念与操作 | 提交物与通过要求 |
|---|---|---|
| V1 / L7-VIS-01 | 理解 checkpoint、参数和加载信息；读 `_tmp_probe_visual_ln_load.py` 的 `load_ckpt_tensor`、`compare_key`；对比 LN、FC1、patch embedding | 逐参数键名/shape/dtype、cosine、MAE、max_abs；解释 missing=0 为何不足以证明正确 |
| V2 / L7-VIS-02 | 理解 initializer 与 external data；读 `_tmp_visual_ln_vs_ckpt.py`；比较 checkpoint 与 ONNX 的 LN 仿射参数 | 列出异常参数并检查是否有融合补偿；解释 γ=1、β=0 只代表仿射部分为恒等，LN 本身仍做归一化 |
| V3 / L7-VIS-03 | 理解运行实例与产物来源；读 `_tmp_probe_visual_ln_export.py`；记录加载、wrap、FX、导出前后的参数摘要 | 建立每路输入/权重/版本表，定位最早变化处；解释 FX→HMONNX 分叉或明确缺少哪项证据 |
| V4 / L7-VIS-04 | 理解初始化与保护标志；审查修前/修后 `_init_weights`，使用干净加载过程做对照 | 证明已加载参数保留、真正缺失参数仍被合理初始化；同时检查 LN、Linear/Embedding，不能只验证 LN |
| V5 / L7-VIS-05 | 理解重新导出与业务验收；在独立实验输出目录生成新产物并对齐 | 新产物的四路指标、参数检查、任务结果和版本；未有业务数据时保留“任务验收待完成” |

V1 只需参数切片也可起步；V3/V4 需要匹配的模型代码与实际加载环境；V5 需要量化/导出依赖和业务评测环境。一次只改变初始化修复这一因素；环境版本对照另开实验，避免同时换环境和补丁。

### 实验记录与批改

每项任务在此追加：假设、唯一改变因素、命令、输入/权重/代码/产物标识、结果路径、结论、尚未验证项。由导师在同一任务下批改；已有历史答案不等于已独立复现。
