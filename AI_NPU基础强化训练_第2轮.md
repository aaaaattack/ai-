# AI/NPU 基础强化训练（第 2 轮）

## 训练说明

- 共 10 题，总分 100 分。
- 本轮重点：L0 浮点数、L3 量化、L4 峰值算力与 Roofline。
- 辅助复习：L2 Transformer shape、L7 精度 Debug、L8 性能 Debug。
- 难度：中等偏基础，按“公式理解 → 手算 → 简单应用”递进。
- 请写出公式、代入过程和单位，不要只写最终结果。
- 暂不附答案。你完成后，我会直接在本文中批改和修订。

---

## 第 1 题｜L0：二进制科学计数法（10 分）

将十进制数 `10.625` 转换为二进制，并写成规格化形式：

```text
1.fraction × 2^E
```

请依次写出：

1. 十进制整数部分 `10` 对应的二进制；
2. 十进制小数部分 `0.625` 对应的二进制；
3. `10.625` 的完整二进制表示；
4. 规格化二进制科学计数法；
5. 真实指数 `E`。

### 我的回答

1. 
2. 
3. 
4. 
5. 

## 第 2 题｜L0：FP16 编码与 bias（10 分）

将十进制数 `-0.75` 编码为 FP16。已知 FP16 的指数 bias 为 15。

请计算：

1. `0.75` 的二进制表示；
2. 规格化后的 `-1.fraction × 2^E`；
3. 符号位；
4. 真实指数 `E`；
5. 存储指数 `e=E+bias` 的十进制值和 5 bit 二进制值；
6. 10 bit 尾数；
7. 最终的 16 bit FP16 位模式。

### 我的回答

1. 
2. 
3. 
4. 
5. 
6. 
7. 

## 第 3 题｜L3：INT8 对称量化（10 分）

某 Tensor 的数值范围为 `[-2.54, 1.80]`，采用 signed INT8 对称量化，整数范围为 `[-127,127]`，zero-point 为 0。

使用公式：

```text
scale = max(|x_min|, |x_max|) / 127
q = clip(round(x / scale), -127, 127)
x_hat = q × scale
```

请计算：

1. scale；
2. 浮点数 `x=0.72` 的量化整数 q；
3. q 反量化后的 `x_hat`；
4. 绝对误差 `|x-x_hat|`；
5. 浮点数 `x=3.00` 会量化成什么整数？为什么？

### 我的回答

1. 
2. 
3. 
4. 
5. 

## 第 4 题｜L3：INT8 非对称量化（10 分）

某 activation 的真实范围为 `[-1.0, 4.1]`，量化到 unsigned INT8 `[0,255]`。

使用公式：

```text
scale = (x_max - x_min) / (q_max - q_min)
zero_point = round(q_min - x_min / scale)
q = clip(round(x / scale) + zero_point, 0, 255)
x_hat = (q - zero_point) × scale
```

请计算：

1. scale；
2. zero-point；
3. `x=1.4` 的量化整数 q；
4. 反量化值；
5. 为什么非对称量化通常需要 zero-point，而对称量化常把 zero-point 设为 0？

### 我的回答

1. 
2. 
3. 
4. 
5. 

## 第 5 题｜L3：Per-tensor 与 Per-channel（10 分）

某两通道权重范围为：

```text
Channel 0: [-1.27, 1.00]
Channel 1: [-0.127, 0.100]
```

采用 signed INT8 对称量化 `[-127,127]`。

请回答：

1. Per-channel 量化时，两个通道的 scale 分别是多少？
2. Per-tensor 量化时，共用的 scale 是多少？
3. 对 Channel 1 中的 `x=0.05`，分别使用 per-channel 和 per-tensor 量化，计算 q、反量化值及绝对误差；
4. 即使本题中的 `0.05` 恰好都能准确表示，为什么 per-channel 对 Channel 1 的其他数值通常仍更有利？请从量化步长解释。

### 我的回答

1. 
2. 
3. 
4. 

## 第 6 题｜L4：峰值算力（10 分）

某 NPU 有 8 个计算核心，每个核心每周期可完成 128 个 INT8 MAC，频率为 `800 MHz`。厂商采用 `1 MAC=2 Ops` 的统计口径。

请计算：

1. `800 MHz` 等于每秒多少个周期？
2. 整颗芯片每周期可完成多少个 MAC？
3. 每秒可完成多少个 MAC？
4. 理论 INT8 峰值是多少 TOPS？
5. 如果实测为 `1.024 TOPS`，计算利用率是多少？

### 我的回答

1. 
2. 
3. 
4. 
5. 

## 第 7 题｜L4：Arithmetic Intensity 与 Roofline（10 分）

某算子一次推理需要 `80 GFLOPs`，从 DDR 读写合计 `40 GB`。芯片计算峰值为 `16 TFLOPs`，可持续 DDR 带宽为 `400 GB/s`。

请计算并判断：

1. Arithmetic Intensity，单位为 FLOPs/Byte；
2. 带宽侧性能上限，单位为 TFLOPs；
3. Roofline 理论性能上限；
4. 该算子是 compute-bound 还是 memory-bound；
5. 计算侧理想时间、带宽侧理想时间和最终理论最短时间；
6. 如果把计算峰值从 16 TFLOPs 提升到 32 TFLOPs，但数据量和带宽不变，理论最短时间是否会明显改善？为什么？

### 我的回答

1. 
2. 
3. 
4. 
5. 
6. 

## 第 8 题｜L2：GQA Tensor Shape（10 分）

某 Transformer 配置如下：

```text
hidden_size = 2048
num_attention_heads = 16
num_key_value_heads = 4
batch = 1
sequence_length = 64
```

统一采用 `[B, heads, S, head_dim]`。

请回答：

1. `head_dim` 是多少？
2. 这是 MHA、MQA 还是 GQA？
3. Q、K、V 的 shape 分别是什么？
4. 每几个 Q heads 共享一组 K/V？
5. 与 16 个 KV heads 的标准 MHA 相比，KV Cache 元素数量约变为原来的多少？

### 我的回答

1. 
2. 
3. 
4. 
5. 

## 第 9 题｜L7：First Bad Stage 与量化误差（10 分）

某模型的验证数据如下：

| 阶段 | Top-1 |
|---|---:|
| PyTorch FP32 | 80.0% |
| PyTorch FP16 | 79.9% |
| ONNX FP16 | 79.8% |
| NPU FP16 | 79.7% |
| NPU INT8 | 71.0% |

进一步发现，NPU INT8 某层 activation 的校准范围为 `[-32,32]`，但 99.8% 的实际数据位于 `[-1,1]`，该层 saturation ratio 接近 0。

请回答：

1. First Bad Stage 是哪一段？依据是什么？
2. saturation ratio 接近 0，是否能证明该层量化没有问题？为什么？
3. outlier 可能如何影响该层 scale 和普通值的量化精度？
4. 设计两个单变量实验验证该层是否为主要敏感层；每个实验写出唯一改变的变量和需要观察的指标。

### 我的回答

1. 
2. 
3. 
4. 

## 第 10 题｜L8：简单性能 Debug（10 分）

某模型端到端延迟为 `25 ms`：

| 阶段 | 延迟 |
|---|---:|
| CPU 预处理 | 4 ms |
| H2D | 1 ms |
| NPU 图执行 | 15 ms |
| D2H | 1 ms |
| CPU 后处理 | 4 ms |

NPU 图执行的 15 ms 中：

```text
大算子：7 ms，计算利用率 85%
80 个小算子：4 ms，计算利用率 10%
Layout Conversion：3 ms
同步等待：1 ms
```

请回答：

1. NPU 图执行、CPU 前后处理、H2D+D2H 分别占 E2E 的百分比；
2. 当前是否应优先优化 85% 利用率的大算子？为什么？
3. 按优先级给出三个优化方向，并标明主要作用于 E2E、Graph、Runtime、Operator 或 Kernel 哪一层；
4. 设计一个最小对照实验，判断 80 个小算子的 4 ms 是否主要来自 launch overhead。

### 我的回答

1. 
2. 
3. 
4. 

---

## 提交说明

完成后直接保存本文件并告诉我“第2轮已完成”。我会：

- 逐题判定正确、部分正确或错误；
- 直接修订错误答案并系统化润色正确答案；
- 更新 L0～L9 掌握度；
- 判断是否可以提高难度，或是否需要继续针对 L0、L3、L4 训练。
