# 第 05 课｜L4.2：内存带宽、Arithmetic Intensity 与 Roofline

## 一、学习目标

完成本课后，你应能：

1. 区分峰值算力与内存带宽；
2. 计算 Arithmetic Intensity（算术强度）；
3. 使用 Roofline 判断算子是 compute-bound 还是 memory-bound；
4. 根据理论上限估计最短执行时间；
5. 把性能判断转化为可执行的优化方向。

---

## 二、计算快不等于端到端快

一个算子要完成两件事：

```text
从 DDR/HBM 搬运输入、权重和输出
→ 在计算单元上执行 MAC/FLOPs
```

若数据搬运跟不上，计算单元会等待数据；若计算太重，带宽可能足够但计算单元成为瓶颈。Roofline 用来比较这两种上限。

---

## 三、内存带宽

带宽表示单位时间最多搬运多少数据：

```text
600 GB/s = 每秒最多搬运约 600 GB 数据
```

这是持续可用带宽时的理想上限。实际还会受访问是否连续、DMA、缓存命中、并发和读写竞争影响。

---

## 四、Arithmetic Intensity

Arithmetic Intensity（AI）表示每搬运 1 Byte 数据，完成多少次浮点运算：

```text
AI = FLOPs / Bytes
```

AI 越低，说明一个算子相对更依赖数据搬运；AI 越高，说明每份数据被重复利用得更多，通常更可能受计算能力限制。

---

## 五、Roofline 公式

设：

```text
P_peak = 峰值计算性能（FLOP/s）
BW = 可持续内存带宽（Byte/s）
AI = FLOPs / Byte
```

则理论性能上限为：

```text
P_roof = min(P_peak, BW × AI)
```

其中：

```text
BW × AI = 带宽允许达到的计算性能上限
```

若 `BW × AI < P_peak`，是 **memory-bound**；若 `BW × AI ≥ P_peak`，是 **compute-bound**。

关键分界点叫 ridge point：

```text
AI_ridge = P_peak / BW
```

算子的 AI 低于它时更偏带宽受限，高于它时更偏计算受限。

---

## 六、带做例题

某算子需要 `240 GFLOPs`，外部 DDR 读写总量 `60 GB`。芯片峰值计算性能 `120 TFLOP/s`，持续带宽 `600 GB/s`。

```text
AI = 240 GFLOPs / 60 GB
   = 4 FLOPs/Byte
```

带宽上限：

```text
P_bw = 600 GB/s × 4 FLOPs/Byte
     = 2400 GFLOP/s
     = 2.4 TFLOP/s
```

理论计算上限：

```text
P_roof = min(120, 2.4) TFLOP/s
       = 2.4 TFLOP/s
```

因此是 memory-bound。忽略其他开销时：

```text
time = 240 GFLOPs / 2.4 TFLOP/s
     = 0.1 s
     = 100 ms
```

---

## 七、优化方向

| 判断 | 优先方向 |
|---|---|
| memory-bound | 减少外部读写、提高缓存复用、融合算子、减少 layout conversion、使用更低 bit 数据 |
| compute-bound | 提高计算阵列利用率、使用更快 precision、优化矩阵 shape、提高并行度、减少不必要计算 |

优化 memory-bound 算子时，单纯提高 TOPS 往往没有明显作用；应先减少数据搬运或提高每个 Byte 的计算复用。

---

## 八、本课练习

### 第 1 题｜算术强度

某算子执行 `120 GFLOPs`，外部读写 `30 GB`。它的 AI 是多少 FLOPs/Byte？

**我的回答：**


### 第 2 题｜Roofline 判断

芯片峰值 `80 TFLOP/s`、带宽 `400 GB/s`。若算子 AI 是 `5 FLOPs/Byte`，带宽上限是多少？该算子是 memory-bound 还是 compute-bound？

**我的回答：**


### 第 3 题｜理论时间

沿用第 2 题，算子总计算量为 `100 GFLOPs`。忽略其他开销时，理论最短时间是多少？

**我的回答：**


### 第 4 题｜优化选择

某模型 profiler 显示 DDR 带宽接近上限，Transpose/Layout Conversion 占用明显。请给出两个优先优化方向，并说明理由。

**我的回答：**


## 九、通过标准

- 能独立计算 AI、带宽上限和 Roofline 上限；
- 能判断 memory-bound 或 compute-bound；
- 能根据判断提出对应的优化方向。

完成后先做章节评估；未达标时只在本文追加夯实内容。后续进入 L2：Transformer 结构与 Q/K/V Tensor shape。

---

## 十、自学补充：Roofline 在真实模型中的用法

### 1. 带宽不只来自 DDR

真实芯片往往有 DDR/HBM、片上 SRAM、Cache 和寄存器等多级存储。Roofline 中的带宽通常指当前瓶颈层级能持续提供的带宽。若数据能被有效复用在片上存储，外部 DDR 流量会下降，实际 AI 会提高。

### 2. 算子融合为什么重要

未融合的算子可能在每一步都把中间 Tensor 写回外存，再读回下一步：

```text
算子 A 写回中间结果
→ 算子 B 再读回中间结果
```

融合后可将中间结果留在寄存器、Cache 或 SRAM，减少外部读写，尤其有利于 memory-bound 场景。

### 3. Layout Conversion 的代价

Transpose、NCHW/NHWC 转换和 Attention Tensor 的维度重排常带来额外读写，但本身计算量很小，因此 AI 往往低。这类算子容易 memory-bound，优化重点是减少转换次数、统一图内 layout 或与相邻算子融合。

### 4. LLM 的 Prefill 与 Decode

Prefill 一次处理较长 prompt，矩阵乘法通常更大，算术强度更高；Decode 每步只生成少量 token，频繁读取 KV Cache，小矩阵和访存占比更高。因此同一模型的 Prefill 与 Decode 可能受完全不同的瓶颈限制。

### 5. Roofline 是上限，不是完整解释

即使 AI 显示某算子应接近计算受限，真实性能仍可能受小算子、同步、shape bucket、CPU fallback、DMA 或 kernel launch 开销影响。性能定位应结合：

```text
E2E breakdown
→ Graph / Subgraph 时间
→ Operator profile
→ 内存流量与阵列利用率
→ Roofline 判断
```

### 6. 自学结论

看到“TOPS 很高但速度不快”时，先检查数据搬运、layout conversion、算子融合和 CPU fallback；再用 AI 与 Roofline 判断提高算力是否真的有用。
