# 第 03 课｜L3.1：对称 INT8 量化、scale 与量化误差
https://qcnlbq68okz6.feishu.cn/wiki/KAFzwJPX1igaeOkBQIIcODXjnfd
## 一、学习目标

学完本课后，你应能：

1. 解释量化为什么能节省模型存储与计算带宽；
2. 用对称 INT8 量化计算 scale；
3. 将浮点数量化为 INT8，再反量化回近似浮点数；
4. 计算并解释量化误差、clipping 与 saturation；
5. 区分“浮点舍入误差”和“量化误差”。

---

## 二、间隔复习

### 复习 1：特殊值

FP16 中，指数位全为 1、尾数非 0 表示什么？

**我的回答：**

`inf`。

**批改：错误。**

```text
指数全 1，尾数全 0 → ±inf
指数全 1，尾数非 0 → NaN
```

尾数是否为 0 是这里的决定条件。下一次复习会再次检查这一点。

### 复习 2：subnormal

FP16 的指数位全为 0、尾数非 0 表示什么？它比最小正常数更大还是更小？

**我的回答：**


---

## 三、为什么需要量化

模型常以 FP32 或 FP16 保存权重和 activation。量化把它们映射到更少的整数 bit，例如 INT8。

```text
FP32：每个值 4 Byte
FP16：每个值 2 Byte
INT8：每个值 1 Byte
```

量化通常能减少模型大小、内存带宽和传输开销，也能让支持 INT8 MAC 的硬件获得更高吞吐。但整数只能表示有限个离散值，因此会引入误差。

---

## 四、对称 INT8 量化

对称量化使用以 0 为中心的整数范围：

```text
q ∈ [-127, 127]
```

给定浮点范围 `[x_min, x_max]`，先取：

```text
max_abs = max(|x_min|, |x_max|)
scale = max_abs / 127
```

量化和反量化公式：

```text
q = clip(round(x / scale), -127, 127)
x_hat = q × scale
```

这里：

- `q` 是 INT8 整数；
- `x_hat` 是反量化后的近似值；
- `clip` 防止整数超出 INT8 范围；
- 对称量化的 zero-point 固定为 0。

---

## 五、带做例题

某层权重范围为 `[-1.0, 0.8]`，采用 signed INT8 对称量化。

```text
max_abs = max(1.0, 0.8) = 1.0
scale = 1.0 / 127 ≈ 0.007874
```

量化 `x=0.5`：

```text
q = round(0.5 / 0.007874)
  = round(63.5)
  = 64
```

反量化：

```text
x_hat = 64 × 0.007874 ≈ 0.50394
量化误差 = x_hat - x ≈ +0.00394
```

这不是 overflow；它来自把连续浮点数映射到离散整数格点时的 rounding。

---

## 六、clipping 与 saturation

若某个数超出校准得到的浮点范围，它对应的 `q` 可能大于 127 或小于 -127。`clip` 会把它截断到边界：

```text
q > 127  →  127
q < -127 → -127
```

这种“撞到边界”的现象叫 saturation。它通常比普通舍入带来更大的误差，因此校准范围与 outlier 是量化中的重点。

---

## 六点五、nano-vLLM 对照：量化前的 Linear 基线

nano-vLLM 本身没有实现 INT8 权重量化；它的 `layers/linear.py` 提供的是未量化基线。先看清基线，才能判断量化后到底改变了什么。

来源：`llm_learning/nano-vllm/nanovllm/layers/linear.py`。

```python
class ReplicatedLinear(LinearBase):                         # 每个 rank 都保存完整权重的线性层
    def forward(self, x: torch.Tensor) -> torch.Tensor:     # 输入 x 的最后一维是 input_size
        return F.linear(x, self.weight, self.bias)          # y = xW^T + b，权重仍是浮点格式
```

量化版本通常在这条基线上增加反量化或整数乘法：

```python
q_weight = torch.round(weight / scale).clamp(-127, 127)     # 用 scale 把浮点权重映射到 INT8，并截断饱和
dequant_weight = q_weight * scale                            # 需要浮点计算时恢复近似权重
y = F.linear(x, dequant_weight, bias)                       # 与未量化基线比较输出误差
```

因此本课的最小验证闭环是：同一输入分别经过浮点 `F.linear` 和量化-反量化后的 `F.linear`，比较最大绝对误差与余弦相似度。不要把 nano-vLLM 的 `Linear` 误认为已经完成 INT8 推理。

---

## 六点六、真实案例：`w8a8h1_sefp` 中的 mantissa 和共享指数

这个格式来自后续 GR00T Prefill 精度排查案例。它不是普通 INT8，也不是 IEEE FP8，而是框架提供的 Shared-Exponent Floating Point（SEFP，共享指数浮点）配置。

| 字段 | 在当前配置中的含义 |
|---|---|
| `w8` | 权重使用 8 bit mantissa（尾数/有效数字字段） |
| `a8` | 激活使用 8 bit mantissa |
| `h1` | 使用 hidden bit；最高有效位可隐含，相当于增加有效精度 |
| `sefp` | 同一分组共享 exponent，每个元素保存自己的 mantissa |

### 6.6.1 指数在哪里

`w8` 和 `a8` 只说明每个元素的 mantissa 精度；指数不是没有，而是由一个 block/group 中的多个元素共享：

```text
                 ┌── mantissa_0 → value_0
shared exponent ─┼── mantissa_1 → value_1
                 ├── mantissa_2 → value_2
                 └── mantissa_n → value_n
```

概念上可以近似写为：

```text
value_i ≈ sign_i × significand_i × 2^E_shared
```

若该格式启用了 hidden bit，可把 `significand_i` 直观理解为包含隐含最高有效位的 `1.mantissa_i`。但实际符号位是否计入 `w8/a8`、共享指数有几 bit、采用什么 bias、每组包含多少元素，都属于后端格式定义，**不能只从 `w8a8h1_sefp` 名称推断**。

### 6.6.2 它和对称 INT8 有什么区别

| 对比项 | 对称 INT8 | SEFP |
|---|---|---|
| 每个元素保存 | 8 bit 整数 `q` | 自己的 mantissa（以及格式定义的符号信息） |
| 一组数共享 | `scale` | exponent |
| 近似恢复 | `x_hat = q × scale` | `x_hat ≈ significand × 2^E_shared` |
| 主要误差来源 | rounding、clipping、scale 选择 | mantissa 舍入、共享指数选择、组内动态范围差异 |

可以把二者都理解成“共享一部分缩放信息”，但它们的编码和硬件计算路径不同，不能因为都出现 `8` 就把 `w8a8h1_sefp` 直接叫作 INT8。

### 6.6.3 为什么组内离群值可能伤害精度

假设同一组里既有较大的数，也有非常小的数。共享指数通常需要覆盖较大值；指数确定后，小数值能够使用的 mantissa 格点可能变得很粗：

```text
同组数值：8.0、7.5、0.02、0.01
                    ↑
大值决定共享指数后，小值可能被严重舍入，甚至接近 0
```

所以排查 SEFP 精度问题时，要额外观察：

1. 分组大小和分组轴；
2. 每组最大值与最小非零值的比例；
3. outlier 是否迫使共享指数变大；
4. 将可疑 Tensor 提升为 `a16h1_sefp` 后，First Bad Tensor 是否恢复；
5. 改变精度时是否只改变 FakeQuant，还是也改变了实际 kernel/LUT 路径。

这个概念会在 `AI_NPU系统课程_L7_GR00T_Prefill精度Debug实战.md` 中再次出现；届时重点不再是格式定义，而是如何通过单变量混精实验验证问题是否来自 mantissa 精度或共享指数。

## 六点七、LUT 基础概念卡：它不是量化格式

LUT（Look-Up Table，查找表）是一种函数近似方法，不是 INT8、FP8 或 SEFP 这样的数据格式。它可以用有限表项近似 `exp`、`sigmoid`、`tanh` 等函数。

数值稳定的 Softmax 通常先减去最大值：

```text
m = max(x)
softmax(x)_i = exp(x_i-m) / Σ exp(x_j-m)
```

假设一个教学用 LUT 只保存 `0、-1、-2、-3` 四个输入点。对于：

```text
x = [0, -0.8, -2.2]
```

精确参考 Softmax 约为：

```text
[0.6410, 0.2880, 0.0710]
```

若用最近点查表，`-0.8→-1`、`-2.2→-2`，近似结果约为：

```text
[0.6652, 0.2447, 0.0900]
```

这里尚未加入量化，误差来自 LUT 网格和查值方式。若 score 进入 LUT 前又经过 a8 FakeQuant，还会先产生输入量化误差；LUT 输出再次量化，则会增加输出量化误差。因此 `a8→a16` 只说明 LUT 两侧的格点更细，不能说明 LUT 已被关闭。

底层如何布局表项、计算索引、插值和向量化属于 Compiler/Runtime 专题。详见 `AI_NPU系统课程_L5_Compiler_Runtime与LUT职责边界.md`。

---

## 七、本课练习

### 第 1 题｜量化目的

为什么 INT8 量化通常能减少模型存储与内存带宽？代价是什么？

**我的回答：**


### 第 2 题｜scale

浮点范围为 `[-2.0, 1.5]`，采用 signed INT8 对称量化。请计算 `max_abs` 和 scale。

**我的回答：**


### 第 3 题｜量化与反量化

沿用第 2 题的 scale，量化 `x=1.0`，再反量化为 `x_hat`。写出误差 `x_hat-x`。

**我的回答：**


### 第 4 题｜saturation

若校准范围是 `[-1.0, 1.0]`，但实际输入出现 `x=1.5`，在对称 INT8 量化中会发生什么？

**我的回答：**


### 第 5 题｜工程判断

INT8 模型精度下降时，为什么不能直接认定 NPU 的量化算子有 bug？请写出一个先验证的单变量实验。

**我的回答：**


### 第 6 题｜SEFP 格式辨析

看到配置 `w8a8h1_sefp` 时，请回答：

1. `w8/a8` 描述的是每个元素的什么部分？
2. exponent 是每个元素独立保存，还是一个分组共享？
3. 为什么不能只根据这个名称断言 exponent 的位宽？
4. 为什么组内 outlier 可能让小数值精度变差？

**我的回答：**


### 第 7 题｜LUT 与 FakeQuant

某量化 Softmax 将输入从 `a8h1_sefp` 改为 `a16h1_sefp` 后 cosine 提升。请回答：

1. 哪一类误差最可能被减小？
2. 这能否证明 LUT 被关闭？
3. 要隔离 LUT 本身的误差，还需要什么对照路径？

**我的回答：**


## 八、通过标准

- 能独立写出对称 INT8 的 scale、量化与反量化公式；
- 能完成一次基本手算；
- 能区分普通 rounding 与 saturation；
- 能区分对称 INT8 与 SEFP，并解释 mantissa、hidden bit 和 shared exponent 的关系；
- 能区分 LUT 近似误差与 LUT 前后 FakeQuant 误差；
- 能用单变量实验判断量化问题。

完成后先做章节评估；未达标时只在本文追加夯实内容。

---

## 九、自学补充：量化常用知识全景

本节作为参考材料。日常做 NPU 部署、查看量化日志或和算法同事沟通时，知道这些概念及其关系即可；需要排查具体精度问题时，再回到相应小节深入。

### 1. 非对称量化与 zero-point

当数据范围明显不围绕 0，例如 activation 范围是 `[2.0, 6.0]`，对称量化会浪费一半整数范围。非对称量化用 `zero_point` 把真实的 0 映射到一个整数位置：

```text
scale = (x_max - x_min) / (q_max - q_min)
q = clip(round(x / scale) + zero_point, q_min, q_max)
x_hat = (q - zero_point) × scale
```

常见 unsigned INT8 范围是 `[0, 255]`。权重常用对称量化；activation 可能使用对称或非对称量化，取决于框架和硬件支持。

### 2. 量化粒度

| 粒度 | 含义 | 典型取舍 |
|---|---|---|
| per-tensor | 整个 Tensor 共用一个 scale | 实现简单、开销低，但误差可能较大 |
| per-channel | 每个输出通道各有 scale | 常用于卷积/线性层权重，精度通常更好 |
| per-group | 将通道或元素分组，每组一个 scale | 精度与元数据、计算开销之间折中 |

粒度越细，越能适应不同通道或分组的数值范围，但需要保存更多 scale，也可能增加运行时处理成本。

### 3. 校准、clipping 与 outlier

PTQ（训练后量化）需要用代表性数据收集 activation 的范围，这一步叫 calibration。若范围太大，scale 变大、普通值的整数刻度变粗；若范围太小，异常大值会被 clip 到 `-127/127` 或 `0/255`。

```text
范围过宽 → 普通值量化误差增大
范围过窄 → saturation / clipping 增多
```

极少数特别大的 activation 称为 outlier。它们常会迫使整个 Tensor 使用很大的 scale，是 INT8 精度下降的常见原因。

### 4. PTQ 与 QAT

| 方法 | 含义 | 适用场景 |
|---|---|---|
| PTQ | 训练完成后直接量化和校准 | 上线快、成本低，是部署首选起点 |
| QAT | 训练时模拟量化误差，再微调模型 | PTQ 精度不达标且可重新训练时使用 |

PTQ 精度下降后，不应直接判定硬件算子有问题。先比较 FP32/FP16 基线、不同校准集、per-tensor/per-channel、以及可疑层保留高精度后的结果。

### 5. 常见权重/激活精度组合

| 组合 | 含义 | 常见特点 |
|---|---|---|
| W8A8 | 权重 INT8、activation INT8 | 速度和带宽收益通常最大，精度风险较高 |
| W8A16 | 权重 INT8、activation FP16/BF16 | 对 activation 更稳，常见于 LLM 推理 |
| W4A16 | 权重 INT4、activation FP16/BF16 | 显著降低权重显存，需更谨慎处理误差 |
| W4A8 | 权重 INT4、activation INT8 | 压缩强，精度和硬件支持要求更高 |

### 6. 混合精度

并非所有层都必须用同一种精度。对量化敏感的层，例如输入/输出层、softmax、norm、异常值明显的线性层，可以保留 FP16、BF16 或 FP32；其余层使用 INT8。这就是 mixed precision。

一个实用的定位方法是：先找 First Bad Tensor，再一次只将一个可疑层恢复高精度，观察精度是否恢复。由此判断该层是否敏感。

### 7. 常见方法名只需认识

- **SmoothQuant**：在 activation 与权重之间迁移一部分量化难度，缓解 activation outlier。
- **AWQ**：优先保护对模型输出更重要的权重，常用于权重量化。
- **GPTQ**：使用二阶近似等方法减少权重量化误差，常用于 LLM 权重量化。

目前不要求手推这些算法；知道它们主要用来降低量化误差即可。

### 8. 量化精度问题的排查顺序

```text
确认 FP32/FP16 基线
→ 确认第一个出现显著误差的阶段或 Tensor
→ 更换校准集或校准范围
→ 比较 per-tensor 与 per-channel
→ 对一个可疑层执行混合精度实验
→ 再判断是量化策略、模型特性、编译器还是硬件问题
```

---

## 十、自学阅读建议

建议先完整阅读“对称 INT8 量化”“校准、clipping 与 outlier”“混合精度”三节；其他内容在遇到实际模型或日志时再回看。完成阅读后无需手算全部练习，可直接进入 L4 的算力和带宽基础。
