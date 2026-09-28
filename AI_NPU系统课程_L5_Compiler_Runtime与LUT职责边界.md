# L5 专题课｜Compiler、Runtime 与 LUT 职责边界

本专题暂不占正式课号。它连接 L3 量化基础、L5 Compiler/Runtime 与 L7 精度 Debug，并以 GR00T Prefill 中的量化 Softmax 为贯穿案例。

## 一、学习目标

完成后应能够：

1. 区分量化前端、编译器后端、Runtime 与 NPU 硬件的职责；
2. 读懂 `QSoftmax → IR Softmax → NPU kernel` 的基本链路；
3. 判断 LUT、FakeQuant、lowering 与设备执行分别可能引入什么误差；
4. 根据“误差最早出现的位置”决定应该找量化、编译器还是 Runtime 团队；
5. 设计跨层对照实验，不把所有 NPU 精度问题都归因于底层硬件。

## 二、从 PyTorch Softmax 到 NPU 执行

一条典型链路可以抽象为：

```text
PyTorch 模型
→ FX / 前端计算图
→ QSoftmax 或 QMaskedSoftmax
→ 携带 LUT 与量化参数的 IR 算子
→ Compiler lowering
→ Kernel / 指令序列
→ Runtime 准备输入并调度
→ NPU 执行
→ Runtime 取回输出
```

Python 中若看到类似调用：

```python
out = ir_softmax_xh2a_*(                 # 选择面向 XH2 的 Softmax IR 算子
    x,                                   # 输入 Attention score
    dim,                                 # Softmax 的归一化维度
    lut_cut_points,                      # LUT 的分段点或索引边界
    lut_values,                          # 查找表中的近似值
    lut_scale,                           # 表内数据和真实数值之间的缩放
)
out = o_quantizer(out)                   # 对 IR Softmax 输出做 FakeQuant
```

它说明前端已经选择了某类 LUT Softmax，并把必要参数放进图中；它没有完整展示 LUT 在设备上如何布局、查找、插值和向量化。

## 三、四层职责边界

| 层次 | 主要职责 | 本案例中的典型对象 |
|---|---|---|
| 量化框架/图前端 | wrap 模块、插入 FakeQuant、选择量化规格、生成或选择 LUT 参数、导出 IR | `QSoftmax`、`QMaskedSoftmax`、`w8a8h1_sefp`、`a16h1_sefp` |
| 编译器后端 | IR 合法化、lowering、算子融合、布局转换、tiling、内存规划、codegen | 将 `ir_softmax_xh2a_*` 变成目标 kernel 或指令序列 |
| Runtime/驱动 | 加载编译产物、准备 Tensor、分配 Buffer、Memcpy、同步、提交 kernel、取回输出 | set input、run、get output、事件同步 |
| NPU 硬件 | 按指令执行查表、算术、归约与数据搬运 | 片上存储访问、向量计算、DMA |

边界并非绝对：某些 LUT 由前端生成，某些由编译器内置；某些归一化由专用 kernel 完成，某些会被拆成多条指令。因此最终判断必须依据当前版本的图、IR、编译日志和设备 dump。

## 四、LUT 底层实现属于谁

“如何在设备上真正查表”主要属于编译器后端、算子库和硬件实现，包括：

- LUT 放在片上 SRAM、常量区还是其他存储；
- 输入如何变成表索引；
- 使用最近点、分段线性还是其他插值；
- 超出 LUT 定义域时如何裁剪；
- 一次向量化处理多少元素；
- reduction、倒数与缩放采用什么中间精度；
- 是否与 mask、scale 或后续算子融合；
- 长序列如何切块以及块间如何同步。

解决方案或模型侧工程师通常不需要一开始就手撕这些实现。首先要通过分层对照，证明误差是在编译前还是编译后出现。

## 五、Softmax 的完整误差栈

量化 Softmax 的概念路径是：

```text
Attention score
→ 输入 FakeQuant
→ LUT 定义域转换与裁剪
→ exp/LUT 近似
→ 求和、倒数和归一化
→ 输出 FakeQuant
→ Attention probability
```

因此至少要区分四类误差：

1. **输入 FakeQuant 误差**：score 在进入 LUT 前已被舍入或饱和；
2. **LUT 近似误差**：有限表项、分段或插值不能完全等于参考函数；
3. **归约与归一化误差**：定点求和、倒数或缩放产生额外误差；
4. **输出 FakeQuant 误差**：Softmax 概率再次落到有限格点。

设备结果还可能叠加第五类误差：lowering、布局、融合、Runtime 或 kernel 实现与前端仿真的不一致。

## 六、三个常见开关不能混为一谈

| 操作 | 能确认改变的内容 | 不能直接推出的结论 |
|---|---|---|
| `force_fp32` | 使用该配置的算子或中间计算精度 | Softmax 若不读取该开关，就不会绕过 LUT |
| `a8h1_sefp → a16h1_sefp` | FakeQuant 的 mantissa 精度、格点或可表示范围 | LUT 不一定关闭，kernel 也不一定换成原生 FP16 Softmax |
| `disable_quant()` | 取决于框架具体实现 | 不能只凭函数名断言“仅关闭 FakeQuant” |
| `is_quanting=False` | 若代码进入 `super().forward()`，通常回到 PyTorch 路径 | 该结果不能代表 LUT-only 精度 |

关键审查点：如果 `disable_quant()` 同时令 `is_quanting=False`，那么 `cosine≈1` 只能说明 Q 模块包装没有破坏结果，不能证明 LUT 几乎无损。

## 七、用 First Bad Stage 决定找谁

```text
浮点 PyTorch 已经错误
→ 模型、输入、权重或基准问题

浮点正确，Q 图/FakeQuant 仿真开始错误
→ 量化配置、校准、FakeQuant、LUT 参数或前端算子

Q 图仿真正确，导出 IR/HMONNX 错误
→ 导出、图改写、常量或 layout 问题

IR/模拟器正确，编译后模型错误
→ compiler lowering、fusion、codegen 或算子库

编译器模拟正确，只有真实 NPU 错误
→ Runtime、驱动、Buffer、同步或设备 kernel
```

“输出在 NPU 上错了”只说明最终观测位置，不说明根因一定在 NPU。要追踪第一个发生分歧的阶段。

## 八、跨层证据表

每次排查都建议保存下面的表：

| 阶段 | 产物/执行体 | 输入标识 | shape/dtype | cosine | max_abs | 是否首个异常 |
|---|---|---|---|---:|---:|---|
| A | PyTorch reference | 固定输入 | 记录 | 1.0 | 0 | 否 |
| B | Q wrapper 非量化路径 | 同上 | 记录 |  |  |  |
| C | LUT-only 前端仿真 | 同上 | 记录 |  |  |  |
| D | 完整 FakeQuant 前端仿真 | 同上 | 记录 |  |  |  |
| E | 导出图推理 | 同上 | 记录 |  |  |  |
| F | 编译器模拟器 | 同上 | 记录 |  |  |  |
| G | 真实 NPU | 同上 | 记录 |  |  |  |

只有固定输入、权重、mask、positions、预处理和比较边界，阶段间结果才可归因。

## 九、LUT Softmax 的七路消融

| 路径 | LUT | 输入 FakeQuant | 输出 FakeQuant | 目的 |
|---|---:|---:|---:|---|
| A：PyTorch reference | 否 | 否 | 否 | 建立参考结果 |
| B：Q wrapper 非量化分支 | 否 | 否 | 否 | 验证模块替换本身 |
| C：LUT-only | 是 | 否 | 否 | 隔离 LUT/归一化近似 |
| D：LUT + 输入 FQ | 是 | 是 | 否 | 测输入量化增量误差 |
| E：LUT + 输出 FQ | 是 | 否 | 是 | 测输出量化增量误差 |
| F：完整 a8 | 是 | 是 | 是 | 部署量化基线 |
| G：完整 a16 | 是 | 是 | 是 | 测混精收益与代价 |

若框架不能分别关闭输入和输出 FakeQuant，应记录这个工具限制，不能用不等价的路径冒充某一路实验。

## 十、编译器阶段还要检查什么

当 First Bad Stage 已进入编译后，检查顺序建议为：

1. IR 中算子类型、属性、常量 LUT 和量化参数是否完整；
2. lowering 前后 shape、dtype、axis、mask 语义是否一致；
3. 是否发生算子融合、layout conversion 或 CPU fallback；
4. 静态/动态 shape 是否选择了不同 kernel；
5. 中间 Buffer 是否被复用、覆盖或错误对齐；
6. Runtime 的 set input、run、get output 和同步边界是否正确；
7. 编译器模拟器与真实设备是否使用同一份产物和输入。

## 十一、本专题练习

### 练习 1：职责判断

前端 Q 图和导出图结果一致，但编译器模拟器首次出现误差。应优先调查哪一层？列出三个检查点。

### 练习 2：证据审查

`disable_quant()` 后 cosine 从 0.93 恢复到 1.0。为什么还不能断言误差全部来自 FakeQuant？还需要检查什么状态？

### 练习 3：设计 LUT-only 实验

写出如何保持 LUT 分支开启，同时旁路输入和输出 FakeQuant。若当前框架做不到，应如何表述结论？

### 练习 4：团队分流

为以下三种情况分别选择主要对接团队，并说明证据：

1. FakeQuant 后立即出现 First Bad Tensor；
2. Q 图正确，lowering 后错误；
3. 编译器模拟正确，只有真实设备错误。

## 十二、通过标准

- 能画出 `PyTorch → Q 图 → IR → Compiler → Runtime → NPU` 链路；
- 能将 Softmax 误差拆成输入 FQ、LUT、归一化、输出 FQ 与设备差异；
- 能解释为什么 `a8→a16` 不等于关闭 LUT；
- 能解释为什么 `disable_quant()` 的结果取决于 `is_quanting` 等实际状态；
- 能根据 First Bad Stage 决定下一步实验和主要对接团队。

下一步回到 `AI_NPU系统课程_L7_GR00T_Prefill精度Debug实战.md`，将本课的七路消融应用到真实 Prefill 子图。
