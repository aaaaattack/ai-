# AI/NPU 综合摸底试卷（第 1 套）—批改与修订版

## 批改说明

- 原始得分：**39 / 100**。
- 本文已将错误或空缺答案直接修正为标准答案，并将正确答案补充为更系统的工程表达。
- 原始作答保存在 `AI_NPU综合摸底试卷_第1套_原始作答备份.md`。
- 分数反映的是本次作答时的掌握状态，不代表阅读本文后的水平。

---

## 第 1 题｜L0：数值与 Tensor 基础（原始得分：7/10）

已知：`x.shape = [1, 3, 224, 224]`、`x.dtype = FP16`、`layout = NCHW`。

### 1. FP16 的组成和 bias

**批改：正确，补充 bias 的含义。**

FP16 共 16 bit：符号位 1 bit、指数位 5 bit、尾数位 10 bit，指数 bias 为 15。

对规格化数，真实指数 `E` 与指数位中存储的无符号整数 `e` 满足：

```text
e = E + bias
E = e - bias
```

bias 的直观作用是把可能为负数的真实指数平移为无符号整数，从而不必再给指数单独设置符号位。例如 `E=-3` 时，`e=-3+15=12=01100₂`。FP16 的指数位 `00000` 和 `11111` 留给 0、非规格化数、无穷大和 NaN。

### 2. 将 `-6.5` 转成 FP16

**批改：原答案空缺，已补全。**

```text
6 = 110₂
0.5 = 0.1₂
-6.5 = -110.1₂ = -1.101₂ × 2²
```

- 符号位：负数，所以 `sign=1`；
- 真实指数：`E=2`；
- 存储指数：`e=2+15=17=10001₂`；
- 尾数：规格化表示最前面的 `1` 是隐藏位，只保存小数点后的 `101`，补足 10 bit 得 `1010000000`。

最终 FP16 位模式：

```text
1 | 10001 | 1010000000
```

### 3. Tensor 元素数量与存储量

**批改：正确。**

```text
元素数 = 1 × 3 × 224 × 224 = 150,528
存储量 = 150,528 × 2 Byte = 301,056 Byte
       = 301,056 ÷ 1,024 KiB
       = 294 KiB
```

`KB` 常按 `1,000 Byte`，`KiB` 明确按 `1,024 Byte`；本题答案为 **294 KiB**。

### 4. permute、stride 与 contiguous

**批改：正确，补充 stride 和 NPU 访问原因。**

`x.permute(0, 2, 3, 1)` 后 shape 为 `[1, 224, 224, 3]`。`permute` 通常只改变 shape 和 stride 元数据，不复制底层存储。虽然逻辑次序已经是 NHWC，但内存里的元素仍按原 NCHW 方式排列，所以它通常不是目标布局下的 contiguous Tensor。

NPU kernel 往往要求连续或硬件友好的布局，以便使用连续 DMA、向量化读取、矩阵分块和静态 SRAM 规划。因此编译器或 Runtime 可能插入 `contiguous`、Transpose 或 Layout Conversion。这不会改变数学结果，但会增加内存流量和延迟。

---

## 第 2 题｜L1：PyTorch、Transformers 与 ONNX（原始得分：8/10）

### 1. 推理模式

**批改：正确，补充 `inference_mode`。**

`model.train()` 会启用训练行为，Dropout 会随机丢弃元素，BatchNorm 也可能使用并更新批次统计量，导致推理输出不稳定或与导出模型不一致。

```python
model.eval()

with torch.inference_mode():
    output = model(input_ids)
```

`torch.no_grad()` 也能关闭梯度记录；`torch.inference_mode()` 进一步关闭部分 autograd 元数据维护。`eval()` 和关闭梯度解决的是不同问题，推理时通常都要使用。

### 2. `state_dict()`、`named_modules()` 与 hook

**批改：正确，按要求补充代码。**

`state_dict()` 返回参数和持久化 buffer：

```python
state = model.state_dict()
for name, tensor in state.items():
    print(name, tensor.shape, tensor.dtype)

torch.save(state, "model_state.pt")
model.load_state_dict(torch.load("model_state.pt", weights_only=True))
```

`named_modules()` 用于遍历模块结构：

```python
for name, module in model.named_modules():
    print(name, type(module).__name__)
```

捕获某个 Linear 层输出可使用 forward hook：

```python
captured = {}

def save_output(name):
    def hook(module, inputs, output):
        captured[name] = output.detach().cpu()
    return hook

target = dict(model.named_modules())["classifier"]
handle = target.register_forward_hook(save_output("classifier"))

with torch.inference_mode():
    _ = model(input_ids)

print(captured["classifier"].shape)
handle.remove()
```

在精度 Debug 中，可用同一批输入分别 dump PyTorch、ONNX 和 NPU 的中间 Tensor，再逐层比较。

### 3. ONNX 动态 shape

**批改：正确，补充导出代码和底层含义。**

未声明动态轴时，示例输入 `[1,128]` 会把 batch 和 sequence length 固化为常量维度，因此 `[4,256]` 无法通过 shape 校验。

```python
dummy_input = torch.ones((1, 128), dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_input,),
    "classifier.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
)
```

底层是把 ONNX Tensor 某些维度由固定整数改成符号维度。Runtime 收到实际输入后再传播实际 shape。但这只表达“模型图允许变化”，不保证每个后端都能对任意 shape 高效编译和执行。

### 4. 为什么 NPU 仍可能要求固定 shape 或 shape bucket

**批改：部分正确；原答案只覆盖了一个原因。**

1. **Kernel 专用化与调度优化**：不同 shape 可能需要不同 tiling、并行度、向量宽度和 SRAM 分块。
2. **静态内存规划**：编译器需要提前确定中间 Tensor 大小、内存复用和 workspace。
3. **Fusion 条件依赖 shape**：某些融合只在特定维度或对齐方式下成立。
4. **硬件约束**：矩阵维度可能需要按 8、16、32 等粒度对齐。

Shape bucket 是折中方案：例如只编译序列长度 128、256、512 三个版本，运行时把请求路由到合适的 bucket。

---

## 第 3 题｜L2：模型结构（原始得分：2/10）

配置：`hidden_size=4096`、`num_attention_heads=32`、`num_key_value_heads=8`、`B=2`、`S=128`。

### 1. Head dimension 与 Attention 类型

**批改：原答案空缺，已补全。**

```text
head_dim = hidden_size ÷ num_attention_heads
         = 4096 ÷ 32
         = 128
```

这是 **GQA（Grouped-Query Attention）**：Q 有 32 个 head，但 K/V 只有 8 个 head，每 4 个 Q heads 共享 1 组 K/V。MHA 的 Q/K/V head 数相同；MQA 只有 1 个 KV head；GQA 的 KV head 数介于二者之间。

### 2. Q、K、V shape

**批改：原答案空缺，已补全。**

统一采用 `[B, heads, S, head_dim]`：

```text
Q: [2, 32, 128, 128]
K: [2,  8, 128, 128]
V: [2,  8, 128, 128]
```

### 3. KV Cache 缩减比例

**批改：正确。**

```text
8 ÷ 32 = 1/4
```

因此 K/V Cache 元素数约缩小到标准 MHA 的 **四分之一**。

### 4. 为什么 GQA 对 decode 更有利

**批改：原答案空缺，已补全。**

Prefill 一次处理整个 prompt，Attention 的 QKᵀ 和概率矩阵乘 V 计算量大致随 `S²` 增长。减少 KV heads 会降低 K/V 投影和存储开销，但不会让所有 Query heads 的 Attention 主计算同比缩小到四分之一。

Decode 每一步通常只有一个新 Query token，却要读取此前所有 token 的 KV Cache，因而更容易受内存容量和带宽限制。GQA 将 KV Cache 缩小到约四分之一，减少每步读取量，支持更长上下文或更大并发，因此通常对 TPOT 的改善更明显。

---

## 第 4 题｜L3：量化（原始得分：0/10）

### 1. Per-channel scale

**批改：原答案空缺，已补全。**

signed INT8 对称量化使用 `[-127,127]`，zero-point 为 0：

```text
scale = max(|x_min|, |x_max|) ÷ 127
```

```text
Channel 0: scale₀ = 1.27 ÷ 127 = 0.01
Channel 1: scale₁ = 0.12 ÷ 127 ≈ 0.000944882
```

scale 表示一个整数刻度对应多少浮点数值。

### 2. 量化、反量化和误差

```text
q = clip(round(x / scale), -127, 127)
x̂ = q × scale
绝对误差 = |x - x̂|
```

Channel 0 的 `x=0.50`：

```text
q = round(0.50 / 0.01) = 50
x̂ = 50 × 0.01 = 0.50
绝对误差 = 0
```

Channel 1 的 `x=0.05`：

```text
q = round(0.05 / 0.000944882) = round(52.9167) = 53
x̂ = 53 × 0.000944882 ≈ 0.0500787
绝对误差 ≈ 0.0000787
```

### 3. Per-tensor 量化

**批改：原答案空缺，已补全。**

Per-tensor 表示整个 Tensor 共用一个 scale；per-channel 表示每个输出通道分别使用自己的 scale。整个权重 Tensor 的最大绝对值为 1.27：

```text
scale_tensor = 1.27 ÷ 127 = 0.01
```

Channel 1 独立量化时步长约为 `0.000944882`，改用 per-tensor 后步长变成 `0.01`，约粗 `10.58` 倍。大量不同浮点值会被舍入到相同整数，细节丢失更严重。

### 4. Outlier、clipping 与 mixed precision

**批改：原答案空缺，已补全。**

Outlier 是数量很少但绝对值显著大于大多数数据的异常大值。本题中 ±40 属于 outlier。若用最大值确定 scale：

```text
scale = 40 ÷ 127 ≈ 0.315
```

`[-2,2]` 内的大多数普通值只使用大约 `-6～6` 的少量整数档位，量化非常粗糙。

单变量实验：

1. **Clipping**：模型和数据不变，只把该层阈值从 ±40 改为 ±2、±3、±4；记录业务指标、cosine、MAE 和 saturation ratio。若普通值误差下降且业务精度恢复，说明 outlier 拉大 scale 是重要根因。
2. **Mixed precision**：其余层保持 INT8，只把该层 activation 恢复为 FP16。若精度显著恢复，可证明该层对 activation 量化敏感。

Clipping 会让阈值外数据饱和；mixed precision 会牺牲部分性能和带宽收益。

---

## 第 5 题｜L4：GPU、ASIC 与 NPU（原始得分：0/10）

### 1. 理论峰值算力

**批改：原答案空缺，已补全。**

```text
峰值 Ops/s
= 核心数 × 每核心每周期 MAC 数 × 频率 × 每 MAC 的 Ops
= 20 × 256 × 1.2×10⁹ × 2
= 12.288×10¹² Ops/s
= 12.288 TOPS
```

### 2. Arithmetic Intensity

Arithmetic Intensity 表示每搬运 1 Byte 数据完成多少计算：

```text
AI = 120 GFLOPs ÷ 30 GB = 4 FLOPs/Byte
```

### 3. Roofline 与瓶颈类型

```text
P_bandwidth = AI × Bandwidth
            = 4 FLOPs/Byte × 600 GB/s
            = 2,400 GFLOPs/s
            = 2.4 TFLOPs

P_roof = min(12.288, 2.4) TFLOPs = 2.4 TFLOPs
```

带宽侧上限更低，所以该算子更可能是 **memory-bound**。

### 4. 理论最短时间与额外开销

```text
计算侧时间 = 120 GFLOPs ÷ 12,288 GFLOPs/s ≈ 9.77 ms
带宽侧时间 = 30 GB ÷ 600 GB/s = 50 ms
理论最短时间 = max(9.77, 50) ms = 50 ms
```

若实测为 80 ms，优先排查实际有效带宽不足、Layout Conversion、额外 Memcpy、kernel launch、同步、图切分和 CPU fallback。

---

## 第 6 题｜L5：Compiler 与 Runtime（原始得分：5/10）

### 1. 日志分类

**批改：正确，补充 graph break 含义。**

- Fusion：`Conv_12 + Relu_13 fused successfully`；
- Unsupported OP：`Unsupported operator: GridSample_27`；
- Graph Break：计算图被 GridSample 打断并切成 3 个 subgraphs；
- Layout Conversion：在 Conv_31 前后插入 NCHW↔NHWC Transpose；
- CPU Fallback：`GridSample_27` 从 NPU 回退到 CPU。

Graph break 会破坏跨算子融合，并可能引入设备间搬运和同步。

### 2. 执行链路时间

**批改：原答案漏加 CPU 算子和两次 Memcpy。**

```text
模型主体链路 = 2.1 + 5.8 + 1.2 + 1.0 + 2.4 = 12.5 ms
fallback 相关 = 1.2 + 5.8 + 1.0 = 8.0 ms
```

如果日志没有单列同步和队列等待，真实开销还可能更高。

### 3. 为什么 5.8 ms 会带来超过 8 ms 的代价

**批改：正确，补充完整链路。**

```text
等待前序 NPU → D2H → layout/dtype 转换 → CPU 执行
→ H2D → 等待后续 NPU 恢复
```

总损失包含搬运、同步、队列切换、缓存失效和融合机会丢失，而不只是 CPU kernel 的 5.8 ms。

### 4. 区分算子计算和 graph break 开销

**批改：原答案空缺，已补全。**

实验一：保留 NPU→CPU→NPU 的分区和搬运，只把 CPU GridSample 临时替换成等 shape、低计算量的 CPU identity。若总时间仍接近 8 ms，说明主要损失来自 graph break；若明显下降约 5.8 ms，则 GridSample 计算本身占主要部分。

实验二：保留数学功能和输入，只用受支持的 NPU 等价实现或插件 kernel 替换 GridSample。比较分区数、D2H/H2D 和总延迟。若 subgraph 合并且约 8 ms 开销消失，即形成 fallback 根因的证据闭环。

---

## 第 7 题｜L6：AI Infra 和推理优化（原始得分：5.5/10）

### 1. TTFT 与 TPOT

**批改：正确。**

- TTFT：请求到达至首个输出 token 可用的时间，主要受排队、调度和 prefill 影响；
- TPOT：首 token 后每生成一个 token 的平均时间，主要反映 decode 性能。

### 2. Continuous Batching 的吞吐和延迟权衡

**批改：基本正确，补充调度机制。**

Continuous batching 动态加入新请求并移除完成请求，使 NPU 保持较高 batch 和利用率，因此总吞吐提高。但单请求可能等待调度 slot，与更多请求共享一次 decode step，并与 prefill 争抢计算和带宽，所以 TTFT、TPOT 可能略差。

### 3. Prefix Cache

**批改：判断正确，补充正式定义。**

Prefix Cache 保存共享前缀已经计算出的 K/V Cache。本题可复用 1,000-token system prompt 的 prefill 结果，所以 TTFT 显著下降。Decode 仍需为每个新 token 前向计算并读取上下文 KV，因此 TPOT 基本不变。

### 4. KV Cache 内存优化方向

**批改：原答案空缺，已补全。**

优先选择 **PagedAttention + 降低 KV Cache 精度**：

1. PagedAttention 对 KV Cache 分页管理，减少连续内存预留、碎片和未使用空间；风险是页表管理开销并依赖 Runtime 支持。
2. KV Cache 从 FP16/BF16 降到 FP8/INT8，可直接降低容量和带宽需求；风险是长上下文精度下降和 kernel 支持限制。

降低最大上下文长度会改变产品能力；Tensor Parallel 会增加多卡通信和部署成本，通常不是消除单卡内存浪费的第一步。

---

## 第 8 题｜L7：精度 Debug（原始得分：3.5/10）

### 1. First Bad Stage 与 First Bad Tensor

**批改：部分正确；原答案把 Stage 和 Layer 混在了一起。**

First Bad Stage 是 `ONNX FP16 → NPU FP16`，因为业务指标首次从 84.0% 明显降到 76.3%。

First Bad Tensor 应定位在 **Layer 23 的输入或产生该输入的 Layer 22 输出**。Layer 23 输出 cosine 首次骤降到 0.71，而且 Softmax 输入在 NPU 上整体放大约 10 倍。

cosine 对整体倍数缩放不敏感。若 NPU Tensor 近似等于 ONNX Tensor 的 10 倍，cosine 仍可能接近 1，因此 Layer 22 的 0.998 不能排除 scale 错误；必须同时比较 min/max、MAE、范数比和逐元素比值。

### 2. 为什么不能先归因于 INT8

**批改：正确。**

NPU FP16 已经产生主要精度下降；NPU INT8 相对 NPU FP16 只进一步下降 0.5 个百分点，且 cosine 为 0.987。因此主要问题在 INT8 之前已经出现。

### 3. 根因假设

**批改：原答案空缺，已补全。**

1. 缩放常量错误或遗漏，例如 Attention 中应除以 `sqrt(head_dim)`，但编译后漏除或常量错误；
2. 广播、shape 或 layout 错误，导致 scale Tensor 沿错误维度广播；
3. MatMul、Add、Scale、Softmax 融合/Lowering 后执行语义改变；
4. dump 的 dtype、layout、量化 scale 或字节序被错误解析。

Softmax 数值近似也要验证，但由于其输入已经异常，优先追查输入生成链路。

### 4. 单变量验证实验

**批改：原答案空缺，已补全。**

1. 检查 ONNX Graph 和 Compiler IR 中 Layer 23 前的 Div/Mul 常量；只把 scale 显式改写为独立算子。若输入范围和精度恢复，缩放假设成立。
2. 用各维数值模式不同的小型合成 Tensor，分别在 ONNX 与 NPU 运行；只关闭相关 Transpose 或固定 shape。根据错位模式判断广播/layout 问题。
3. 只关闭 Layer 22～23 的 fusion，或将该小段强制在参考后端执行。未融合版本对齐而融合版本失配，可收敛到编译 Pass 或融合 kernel。
4. 用已知序列验证 dump→解析链路，并按候选 dtype/layout 重新解码原始二进制，排除观测工具问题。

每个实验记录业务指标、min/max、MAE、cosine、范数比以及唯一改变的变量。

---

## 第 9 题｜L8：性能 Debug（原始得分：3.5/10）

### 1. 延迟比例

**批改：正确。**

```text
NPU 图执行占比 = 11 ÷ 20 = 55%
CPU 前后处理占比 = (3 + 4) ÷ 20 = 35%
H2D+D2H 占比 = 2 ÷ 20 = 10%
```

### 2. 为什么低利用率不等于 kernel 很差

**批改：原答案空缺，已补全。**

小算子工作量不足时，可能没有足够并行任务填满 NPU；固定 launch、调度和同步开销也会占据较大比例。低利用率还可能来自 shape/batch 太小、算术强度低、内存等待、图过碎或数据依赖。12% 是现象，不是根因，必须结合 shape、执行时长、AI、带宽和 launch 次数判断。

### 3. 优化方向与层级

**批改：原答案方向基本合理，但 kernel 优化的优先级过早。**

1. 融合 120 个小算子并消除不必要的 Layout Conversion：主要改善 Graph/Operator 层；
2. 减少同步并尝试异步流水或传输重叠：主要改善 Runtime/Graph 层；
3. 优化或下沉 CPU 前后处理：主要改善 E2E 层，可尝试 SIMD、多线程、零拷贝或移入 NPU 图。

大算子利用率已有 82%，暂不是首要对象。小算子 kernel 是否需要重写，应在排除 shape、launch 和 memory-bound 后再决定。

### 4. 区分 launch、shape 和 memory-bound

**批改：原答案空缺，已补全。**

选择一个代表性小算子，保持算子类型、dtype 和软件栈不变，只将 batch 或 Tensor 尺寸按 `1×、2×、4×、8×` 放大。记录单次 kernel 延迟、吞吐、计算利用率、实际 DDR 带宽、launch 时间和 Arithmetic Intensity。

- shape 增大但初期延迟几乎不变：固定 launch overhead 主导；
- shape 增大后吞吐和计算利用率显著上升：原 shape 太小；
- 延迟随数据字节数近似线性增长，计算利用率低而有效带宽接近该访问模式上限：memory-bound。

再做 fused/unfused 对照；融合后若延迟显著下降，可证明 launch/图碎片是主要因素。

---

## 第 10 题｜L9：综合解决方案（原始得分：4.5/10）

### 1. FP16 是否达到 20 Hz

**批改：正确。**

```text
E2E = 6 + 18 + 3 + 32 + 5 = 64 ms
频率 = 1,000 ms/s ÷ 64 ms ≈ 15.625 Hz
```

FP16 原型未达到 20 Hz 所要求的 50 ms 周期。

### 2. W8A8 的性能收益

**批改：正确。**

```text
FP16 NPU 模块 = 18 + 3 + 32 = 53 ms
W8A8 NPU 模块 = 12 + 2 + 22 = 36 ms
节省 = 53 - 36 = 17 ms
```

预处理和后处理仍占 11 ms，所以 E2E 为 47 ms。它满足 20 Hz 的 50 ms 周期，但距离客户的 40 ms 目标仍差 7 ms。

### 3. 离线指标和真实成功率为何不一致

**批改：原答案只指出了误差累积，已补充。**

1. 闭环误差累积：单步偏差改变下一时刻状态，使模型逐渐偏离离线数据分布；
2. 平均指标掩盖关键动作：少数抓取、避障、终止 token 出错即可导致整条任务失败；
3. 离散 token 不等于控制误差：相邻 token 也可能造成明显位置、角度或速度偏差；
4. 量化可能破坏动作序列的平滑性、时序一致性或 action chunk 边界；
5. 校准集可能未覆盖真实光照、视角、运动模糊和机器人状态；
6. P99 延迟抖动和过期动作也可能损害闭环成功率。

因此必须同时看中间 Tensor、动作序列误差、轨迹偏差和真实闭环成功率。

### 4. 最小验证计划

**批改：原答案空缺，已补全。**

1. 建立可重复基线，采集 FP16/W8A8 的平均、P95、P99 延迟，拆分预处理、各 NPU 模块、传输、后处理和同步；
2. 依次比较 Vision Encoder、Projector、Language/Action Model 和后处理输出，定位 First Bad Stage；
3. 在首个异常模块逐层 dump，以 cosine、MAE、min/max、percentile 和 saturation ratio 定位 First Bad Tensor；
4. 其余模块保持 W8A8，每次只把一个可疑模块或层恢复 FP16，比较精度恢复量和延迟回退量；
5. 先用录制轨迹做 open-loop 动作序列对比，再做安全的 closed-loop A/B 测试，统计成功率、轨迹偏差、动作抖动和失败类型。

### 5. 客户结论（150 字以内）

**批改：原答案空缺，已补全。**

当前版本可继续技术验证，但不建议直接量产交付：47 ms满足20 Hz，却未达40 ms目标，且任务成功率由91%降至78%。请提供代表性校准集、失败轨迹、逐模块输出及完整Profiler。若混合精度将成功率恢复至验收线且P99延迟不超过40 ms，可进入小规模试点。

---

## 本轮评分与掌握情况

| 主题 | 原始得分 | 掌握判断 | 主要表现 |
|---|---:|---|---|
| L0 数值与 Tensor | 7/10 | 部分掌握 | Tensor 容量、permute 理解较好；浮点编码不会独立计算 |
| L1 PyTorch/ONNX | 8/10 | 基本掌握 | 概念和排查方向正确；缺 API 实践与动态编译理解 |
| L2 模型结构 | 2/10 | 薄弱 | 知道 KV Cache 比例，但 Q/K/V shape 和 GQA 原理不牢 |
| L3 量化 | 0/10 | 尚未掌握 | scale、量化/反量化、per-channel、outlier 需从头训练 |
| L4 GPU/ASIC/NPU | 0/10 | 尚未掌握 | 峰值算力、MAC/Ops、AI、Roofline 需从头训练 |
| L5 Compiler/Runtime | 5/10 | 部分掌握 | 能识别日志现象；不会完整计算 fallback 成本和设计实验 |
| L6 AI Infra | 5.5/10 | 部分掌握 | TTFT/TPOT 理解较好；KV Cache 优化手段不足 |
| L7 精度 Debug | 3.5/10 | 初步理解 | 能排除 INT8 主责；Stage/Tensor 区分和假设实验不足 |
| L8 性能 Debug | 3.5/10 | 初步理解 | 会做 E2E 占比；bound 判断和实验设计不足 |
| L9 综合方案 | 4.5/10 | 初步理解 | 性能计算正确；精度风险和交付闭环不完整 |
| **合计** | **39/100** | **基础不均衡** | **工程直觉好于底层计算基础** |

## 下一阶段建议

下一轮不宜继续做同难度综合卷，建议按以下顺序补基础：

1. 二进制科学计数法、FP16 指数与 bias；
2. 对称 INT8 的 scale、量化、反量化和误差；
3. MAC、Ops、TOPS、带宽和单位换算；
4. Arithmetic Intensity 与 Roofline；
5. Transformer 的 Q/K/V shape 与 KV Cache；
6. 再回到 First Bad Tensor 和单变量实验设计。

建议下一组采用 10 道中等偏基础题：前 7 题集中训练 L0、L3、L4，后 3 题用简单部署案例连接 L5、L7、L8。
