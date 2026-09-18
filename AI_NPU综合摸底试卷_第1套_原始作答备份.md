# AI/NPU 综合摸底试卷（第 1 套）

## 试卷说明

- 覆盖范围：L0～L9，共 10 个主题。
- 题量与分值：共 10 题，每题 10 分，总分 100 分。
- 难度顺序：基础计算 → 模型与部署分析 → 精度/性能 Debug → 综合客户方案。
- 建议作答时间：120～180 分钟。
- 请写出计算过程、单位和判断依据；只写最终结论可能无法获得全部分数。
- 暂时不会的小问可以写“不确定”，并说明你能想到的排查方向。
- 本试卷不附答案。完成后交给我，我会逐题批改并判断知识掌握情况。

---

## 第 1 题｜L0：数值与 Tensor 基础（10 分）

已知一个 Tensor：

```text
x.shape = [1, 3, 224, 224]
x.dtype = FP16
layout = NCHW
```

请回答：

1. FP16 的符号位、指数位和尾数位各有多少 bit？指数 bias 是多少？（2 分）
答:1 . FP16 的符号位为 1 bit，指数位为 5 bit，尾数位为 10 bit。指数 bias 为 15。但我不太理解bias是什么，能否帮我解释一下？
2. 将十进制数 `-6.5` 转换成规格化二进制科学计数法，并写出它的 FP16 符号位、指数位和尾数位。（3 分）
答：不会转
3. 该 Tensor 一共有多少个元素？仅计算原始数据时占用多少 Byte、多少 KiB？请写清单位换算。（2 分）
答：该 Tensor 的 shape 为 [1, 3, 224, 224]，所以元素总数为 1 * 3 * 224 * 224 = 150528 个元素。每个 FP16 元素占用 2 Byte，因此总占用为 150528 * 2 = 301056 Byte。换算为 KiB：301056 Byte / 1024 = 294 KiB。
4. 将它执行 `permute(0, 2, 3, 1)` 后，shape 是什么？这一步是否必然复制底层数据？为什么后续某些 NPU 算子可能要求再执行 contiguous/layout conversion？（3 分）
答：执行 `permute(0, 2, 3, 1)` 后，shape 变为 [1, 224, 224, 3]。这一步不必然复制底层数据，因为 permute 只是改变了 tensor 的视图（view），并没有改变数据的存储顺序。然而，某些 NPU 算子可能要求数据是连续的（contiguous）或者特定的布局（layout conversion），因为它们在硬件上对数据访问有特定的要求，如果数据不是连续存储的，可能会导致性能下降或无法正确执行。

## 第 2 题｜L1：PyTorch、Transformers 与 ONNX（10 分）

某团队准备将文本分类模型从 PyTorch 导出到 ONNX。模型含 Dropout，输入序列长度在 16～512 之间变化。工程师使用如下流程：

```python
model.train()
output = model(input_ids)
```

导出时固定使用 `input_ids.shape = [1, 128]`，且没有声明 dynamic axes。客户随后输入 `[4, 256]`，Runtime 报 shape 不匹配。

请回答：

1. `model.train()` 在这里有什么风险？推理前通常应如何设置模型和梯度环境？（2 分）
答：`model.train()` 会启用训练模式，这意味着模型中的 Dropout 层会在前向传播时随机丢弃一部分神经元，从而引入随机性。这在推理阶段是不希望的，因为我们希望模型的输出是确定性的。推理前通常应将模型设置为评估模式，即调用 `model.eval()`，并且关闭梯度计算，可以使用 `torch.no_grad()` 上下文管理器来避免不必要的梯度计算，从而提高推理效率和减少内存占用。
2. `state_dict()` 与 `named_modules()` 分别用于查看什么？如果要捕获某个 Linear 层的中间输出，更适合使用什么机制？（2 分）
答：`state_dict()` 用于查看模型的参数和缓冲区的状态，包括权重和偏置等信息，它返回一个字典，其中键是参数的名称，值是对应的张量。但我不知道具体怎么使用这个api，批改过程中可以直接在这里插入示例代码。
`named_modules()` 用于查看模型中所有子模块的名称和实例，它返回一个生成器，生成每个子模块的名称和模块对象。如果要捕获某个 Linear 层的中间输出，更适合使用 forward hook 机制，可以通过 `register_forward_hook()` 方法在该层注册一个钩子函数，在前向传播时捕获该层的输出。这个我也不会用，需要具体示例，批改过程中可以直接在这里插入示例代码
3. 为什么导出的 ONNX 可能只能接受 `[1, 128]`？如何让 batch 和 sequence length 支持动态变化？（3 分）
答：导出的 ONNX 可能只能接受 `[1, 128]`，是因为在导出时没有声明动态轴（dynamic axes），导致模型在 ONNX 中被固定为特定的输入形状。如果要让 batch 和 sequence length 支持动态变化，需要在导出时使用 `torch.onnx.export()` 函数的 `dynamic_axes` 参数，指定哪些维度是动态的。例如，可以设置 `dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}}`，这样导出的 ONNX 模型就可以接受不同的 batch size 和 sequence length。但我不知道具体代码，批改的时候在这里加上具体代码示例，另外我挺好奇底层怎么实现的。
4. 即使 ONNX 声明了动态 shape，为什么 NPU 编译器仍可能要求若干固定 shape 或 shape bucket？请给出两个工程原因。（3 分）
答： 硬件优化：NPU 的硬件架构可能对特定的输入形状进行了优化，例如针对特定的 batch size 或 sequence length 进行了内存布局和计算路径的优化。为了充分利用这些优化，编译器可能要求输入形
状在一定范围内固定，从而生成高效的执行计划。
## 第 3 题｜L2：模型结构（10 分）

某 Decoder-only Transformer 的配置如下：

```text
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
batch = 2
sequence_length = 128
```

请回答：

1. 每个 attention head 的维度是多少？该模型使用的是 MHA、MQA 还是 GQA？为什么？（2 分）
答：说不清
2. 写出 Q、K、V 经过拆分 head 后的 shape。统一采用 `[B, heads, S, head_dim]`。（3 分）
答：不会
3. 与 32 个 KV heads 的标准 MHA 相比，本配置的 K/V Cache 元素数量约缩小为几分之一？暂不考虑层数和数据类型。（2 分）
答：四分之一
4. 为什么这种变化通常更有利于 decode，而对 prefill 计算量的改善未必同样明显？请从计算和内存访问两个角度解释。（3 分）
答：不会
## 第 4 题｜L3：量化（10 分）

某 Linear 层有两个输出通道，其权重范围分别为：

```text
Channel 0: [-1.27, 0.80]
Channel 1: [-0.10, 0.12]
```

采用 signed INT8 对称 per-channel 量化，整数范围为 `[-127, 127]`。

请回答：

1. 分别计算两个通道的 scale。（2 分）
答：不会
2. 分别量化 Channel 0 中的 `0.50` 和 Channel 1 中的 `0.05`，再反量化，并计算各自的绝对误差。（3 分）
答：不会
3. 如果改用 per-tensor 量化，统一 scale 应是多少？它为什么可能明显伤害 Channel 1 的精度？（2 分）
答：不会，什么是 per-tensor 量化，scale 是怎么计算的都不知道
4. 某模型 W8A8 后精度下降。你发现某层 activation 的绝对最大值为 40，但 99.9% 的数值落在 `[-2, 2]`。说明 outlier 会怎样影响 scale 和大多数普通值，并设计一个单变量实验验证 clipping 或 mixed precision 是否有效。（3 分）
答：不会，什么是 outlier，scale 是怎么计算的都不知道
## 第 5 题｜L4：GPU、ASIC 与 NPU（10 分）

某 NPU 有 20 个计算核心，每个核心每周期可完成 256 个 FP16 MAC，频率为 1.2 GHz。厂商采用 `1 MAC = 2 Ops` 的统计口径。某算子完成一次推理需要 `120 GFLOPs`，并从 DDR 读写合计 `30 GB`。芯片可持续带宽为 `600 GB/s`。

请回答：

1. 计算该 NPU 的理论 FP16 峰值算力，单位为 TOPS。（3 分）
2. 计算该算子的 Arithmetic Intensity，单位为 FLOPs/Byte。（2 分）
3. 根据 Roofline，计算带宽侧性能上限，并判断该算子更可能是 compute-bound 还是 memory-bound。（3 分）
4. 忽略其他开销，估算该算子的理论最短执行时间。若实测为 80 ms，你会优先怀疑哪些额外开销？列出两项。（2 分）
全不会，不理解关键词及相关概念
## 第 6 题｜L5：Compiler 与 Runtime（10 分）

编译和运行日志摘录如下：

```text
[Compiler] Conv_12 + Relu_13 fused successfully
[Compiler] Unsupported operator: GridSample_27
[Compiler] Graph partitioned into 3 subgraphs
[Compiler] Insert Transpose NCHW->NHWC before Conv_31
[Compiler] Insert Transpose NHWC->NCHW after Conv_31
[Runtime] Execute subgraph_0 on NPU: 2.1 ms
[Runtime] Execute GridSample_27 on CPU: 5.8 ms
[Runtime] Memcpy D2H: 1.2 ms
[Runtime] Memcpy H2D: 1.0 ms
[Runtime] Execute subgraph_2 on NPU: 2.4 ms
```

请回答：

1. 日志中有哪些现象分别属于 fusion、unsupported op、graph break、layout conversion 和 CPU fallback？（3 分）
答：fusion: Conv_12 + Relu_13 fused successfully; unsupported op: GridSample_27; graph break: Graph partitioned into 3 subgraphs（猜的，不理解）; layout conversion: Insert Transpose NCHW->NHWC before Conv_31 and Insert Transpose NHWC->NCHW after Conv_31; CPU fallback: Execute GridSample_27 on CPU
2. 仅按日志相加，模型主体执行链路至少是多少 ms？其中 CPU fallback 相关段落至少占多少 ms？（2 分）
答：模型主体执行链路至少是 2.1 + 2.4 = 4.5 ms；CPU fallback 相关段落至少占 5.8 ms。
3. 为什么一个只有 5.8 ms 的 CPU 算子可能导致超过 8 ms 的额外端到端开销？（2 分）
答：因为 CPU fallback 涉及数据搬运、同步等额外开销，这些开销可能远超算子本身的计算时间。
4. 设计两个独立实验，用来区分主要瓶颈究竟来自 GridSample 本身，还是来自 graph break 导致的数据搬运与同步。（3 分）
答：不会

## 第 7 题｜L6：AI Infra 和推理优化（10 分）

某 7B LLM 在线服务的测试结果如下：

| 配置 | TTFT | TPOT | 并发吞吐 |
|---|---:|---:|---:|
| Batch 1 | 120 ms | 18 ms/token | 45 token/s |
| Continuous Batching | 180 ms | 20 ms/token | 240 token/s |
| Continuous Batching + Prefix Cache | 70 ms | 20 ms/token | 260 token/s |

测试请求大量共享相同的 1,000-token system prompt。

请回答：

1. TTFT 和 TPOT 分别描述什么？它们主要对应 prefill 还是 decode？（2 分）
答：TTFT（Time To First Token）描述从请求开始到生成第一个 token 的时间，主要对应 prefill 阶段；TPOT（Time Per Output Token）描述生成每个后续 token 所需的平均时间，主要对应 decode 阶段。
2. 为什么 continuous batching 提高了总吞吐，却可能让单请求 TTFT 和 TPOT 变差？（2 分）
答：Continuous batching 通过批处理多个请求来提高整体吞吐量，但可能导致单个请求的等待时间增加，从而影响 TTFT 和 TPOT。
3. 为什么 prefix cache 显著改善 TTFT，但 TPOT 几乎不变？（2 分）
答：不理解什么是prefix cache，猜测是缓存了前缀的计算结果，从而减少了 prefill 阶段的计算时间，因此显著改善了 TTFT；而 TPOT 主要受 decode 阶段影响，prefix cache 对 decode 的影响较小，因此 TPOT 几乎不变。
4. 如果显存主要被 KV Cache 占满，你会考虑 PagedAttention、降低最大上下文长度、降低 KV Cache 精度还是 Tensor Parallel？请选择两个优先方向，并解释它们的作用及风险。（4 分）
答：不理解什么是PagedAttention，Tensor Parallel。
## 第 8 题｜L7：精度 Debug（10 分）

某图像分类模型的验证结果如下：

| 阶段 | Top-1 | 与上一阶段输出 cosine |
|---|---:|---:|
| PyTorch FP32 | 84.2% | — |
| PyTorch FP16 | 84.1% | 0.9998 |
| ONNX FP16 | 84.0% | 0.9997 |
| NPU FP16 | 76.3% | 0.9120 |
| NPU INT8 | 75.8% | 0.9870（相对 NPU FP16） |

逐层比较 ONNX FP16 与 NPU FP16 后发现：

```text
Layer 0～21 cosine > 0.999
Layer 22 cosine = 0.998
Layer 23 cosine = 0.71
Layer 24 cosine = 0.69
```

Layer 23 是一个 Softmax，ONNX 输入范围约为 `[-8, 7]`，NPU dump 输入范围约为 `[-80, 70]`。

请回答：

1. First Bad Stage 和 First Bad Tensor 分别最可能在哪里？（2 分）
答：First Bad Stage 最可能在 Layer 23，因为从 Layer 22 到 Layer 23 的 cosine 相似度下降显著，从 0.998 降到 0.71，说明在 Layer 23 出现了较大的数值偏差。First Bad Tensor 也最可能在 Layer 23 的输入张量，因为该层的输入范围在 NPU 上被放大了约 10 倍（[-80, 70]），这可能导致 Softmax 的输出与预期差异较大，从而影响后续层的计算。
2. 为什么不能先把主要责任归因于 INT8 量化？（2 分）
答：因为 NPU FP16 的输出已经出现了显著的精度下降（Top-1 从 84.0% 降到 76.3%，cosine 从 0.9997 降到 0.9120），而 INT8 量化后的输出与 NPU FP16 的 cosine 相似度为 0.9870，说明 INT8 量化对精度的影响相对较小。因此，主要责任更可能在于 NPU FP16 的计算或数据处理，而不是 INT8 量化。
3. 针对 Layer 23 前输入范围放大 10 倍，提出至少三个根因假设。（3 分）
答：不会。
4. 为每个假设设计一个尽量单变量、可形成证据闭环的验证实验。（3 分）
答：不会。
## 第 9 题｜L8：性能 Debug（10 分）

某视觉模型在 GPU 上延迟为 8 ms，在 NPU 上端到端延迟为 20 ms。NPU 数据如下：

| 阶段 | 延迟 |
|---|---:|
| CPU 预处理 | 3.0 ms |
| H2D | 1.0 ms |
| NPU 图执行 | 11.0 ms |
| D2H | 1.0 ms |
| CPU 后处理 | 4.0 ms |

NPU 图内 Profile：

| 类别 | 延迟 |
|---|---:|
| 2 个大算子 | 4.0 ms |
| 120 个小算子 | 3.6 ms |
| Layout Conversion | 2.0 ms |
| 同步等待 | 1.4 ms |

补充数据：大算子的计算单元利用率为 82%，小算子平均利用率为 12%，DDR 带宽峰值利用率为 35%。

请回答：

1. 分别计算 NPU 图执行和 CPU 前后处理占端到端延迟的比例。（2 分）
答：NPU 图执行占端到端延迟的比例为 11.0 ms / 20 ms = 55%；CPU 前后处理占端到端延迟的比例为 (3.0 ms + 4.0 ms) / 20 ms = 7.0 ms / 20 ms = 35%。
2. 为什么不能仅根据 12% 的小算子计算利用率，就断定 NPU kernel 实现很差？（2 分）
答：不懂
3. 根据现有证据，按优先级给出三个优化方向，并说明每项主要改善 E2E、Graph、Operator、Kernel 还是 Hardware 层。（3 分）
答：前后处理放到npu或者使用simd实现，改善e2e；做算子融合，改善graph；优化小算子kernel实现，改善kernel
4. 设计一个最小实验，判断 120 个小算子的主要问题是 launch overhead、shape 太小，还是 memory-bound。说明需要对比的变量和观察指标。（3 分）
答：不会
## 第 10 题｜L9：综合解决方案（10 分）

一家机器人客户希望把 VLA 模型部署到 NPU，控制频率目标为 20 Hz，即每个控制周期最多 50 ms。当前原型数据如下：

```text
相机预处理：6 ms
Vision Encoder（NPU FP16）：18 ms
Projector（NPU FP16）：3 ms
Language/Action Model（NPU FP16）：32 ms
动作后处理与控制发送：5 ms
```

客户希望改成 W8A8，把端到端延迟降到 40 ms。初次量化后测得：

```text
Vision Encoder：12 ms
Projector：2 ms
Language/Action Model：22 ms
端到端：47 ms
离线 action-token accuracy：下降 1.2%
机器人任务成功率：从 91% 降到 78%
```

请以解决方案工程师的身份回答：

1. FP16 原型是否达到 20 Hz？请计算端到端延迟和理论频率。（2 分）
答：FP16 原型的端到端延迟为 6 ms + 18 ms + 3 ms + 32 ms + 5 ms = 64 ms。理论频率为 1 / (64 ms) = 15.625 Hz。因此，FP16 原型未达到 20 Hz 的目标。
2. W8A8 后各 NPU 模块合计节省多少 ms？为什么端到端仍未达到 40 ms？（2 分）
答：W8A8 后各 NPU 模块的节省时间为 (18 ms - 12 ms) + (3 ms - 2 ms) + (32 ms - 22 ms) = 6 ms + 1 ms + 10 ms = 17 ms。端到端延迟为 6 ms + 12 ms + 2 ms + 22 ms + 5 ms = 47 ms，仍未达到 40 ms 的目标，主要原因可能是 CPU 前处理和后处理的时间（6 ms + 5 ms = 11 ms）占据了较大比例，且 W8A8 的优化主要集中在 NPU 模块上，而 CPU 部分未得到优化。
3. 离线 accuracy 只下降 1.2%，但业务成功率下降 13 个百分点。列出至少三个可能原因，并说明为什么不能只看离线 token accuracy。（2 分）
答：实时性要求不同，实际执行过程中会累计误差，只能想到这么多  帮我补充
4. 给出一个按优先级排列的最小验证计划，必须同时包含：性能 breakdown、First Bad Stage/Tensor、敏感模块 mixed precision、真实轨迹或动作序列评估。（2 分）
答：不会
5. 写一段不超过 150 字的客户结论，明确回答：目前能否交付、主要风险、下一步需要客户提供什么、什么条件满足后可以进入试点。（2 分）
答：不会
---

## 建议答题格式

```text
第 1 题
1. ...
2. ...
3. ...
4. ...

第 2 题
1. ...
...
```

如果一次写不完，可以分批提交，例如先回答第 1～3 题。我会等整套试卷完成后统一评分；如果你希望边做边改，也可以明确告诉我“逐题批改”。
