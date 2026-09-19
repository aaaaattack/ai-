# 第9课：NVTX 和 timeit — 推理性能观测

> 学习路线：阶段4 Kernel 与性能优化（CS336 Chapter 2 hw1）
> 对应文件：
> - CS336：`chapter2/hw1/train_nvtx.py`（NVTX 标记）
> - CS336：`chapter2/hw1/train_timeit.py`（timeit 计时）
> - 之前讨论过的 profiling 工具：`课后笔记/推理性能优化系统学习路线.md`

---

## 一、NVTX — 在时间线上打标记

### 1.1 是什么

NVTX（NVIDIA Tools Extension）允许你在代码中插入命名的范围标记，配合 `nsys` 查看时，这些标记会以彩色色块出现在 GPU 时间线上。

```python
import torch.cuda.nvtx as nvtx

with nvtx.range("forward"):
    logits = model(x)
    loss = loss_fn(logits, y)

with nvtx.range("backward"):
    loss.backward()

with nvtx.range("optimizer_step"):
    optimizer.step()
```

### 1.2 在 nsys 时间线上看到什么

运行：
```bash
nsys profile --trace=cuda,osrt,nvtx -o report.qdrep python train_nvtx.py
nsys-ui report.qdrep
```

时间线示意：
```
GPU Stream:
  [▇████ forward ████▇][▇████ backward ████▇][▉ optimizer ▉]
   ↑ NVTX "forward" 标记的 kernel 区间
   ↑ 所有在这个 with 块中 launch 的 CUDA kernel 都归入这个色块
```

### 1.3 `torch.cuda.synchronize()` 的作用

```python
with nvtx.range("forward"):
    logits = model(x)           # launch 了很多 kernel，但异步执行
    if device.type == 'cuda':
        torch.cuda.synchronize()  # 等待所有 kernel 完成
    # 这样 NVTX 的 forward 色块才准确反映 GPU 实际执行时间
```

**不加 synchronize 的后果**：CPU 侧的 `with nvtx.range` 可能在 GPU 还在跑的时候就结束了，时间线上 forward 色块很短，但实际 GPU 执行更长 → 误导。

### 1.4 NVTX 的不同级别

```python
# 粗粒度：阶段级
with nvtx.range("epoch_0"):
    for step in range(steps):
        ...

# 细粒度：算子级
with nvtx.range("attention"):
    attn_out = attention(q, k, v)
with nvtx.range("ffn"):
    ffn_out = swiglu(x)
with nvtx.range("norm"):
    norm_out = rms_norm(x)
```

级别越细，越能定位瓶颈在哪个算子。

---

## 二、timeit — Python 级快速计时

### 2.1 `timeit.default_timer`

```python
import timeit

timer_func = timeit.default_timer  # 平台最优的计时器
start = timer_func()
output = model(x)
elapsed = timer_func() - start
print(f"forward: {elapsed*1000:.2f}ms")
```

`timeit.default_timer` 自动选择最精确的计时器（Windows 上 `perf_counter`，Linux 上 `time.perf_counter`）。

### 2.2 统计多次运行

```python
# timeit.timeit: 自动多次运行求平均
t = timeit.timeit(
    lambda: model(x),
    number=100,           # 跑 100 次
    setup="model.eval()"
)
print(f"avg forward: {t/100*1000:.2f}ms")
```

**`number=100`**：跑 100 次，除以 100 得平均值。比单次测量更稳定，排除噪声。

### 2.3 和 wandb 集成

CS336 把 timeit 结果记录到 wandb：

```python
# train_timeit.py
if (epoch+1) >= 10:           # 前 10 个 epoch 预热，不统计
    timer_func = timeit.default_timer
    start_time = timer_func()  # 第 10 个 epoch 开始计时

# epoch 结束后：
total_time = timer_func() - start_time
wandb.log({"epoch_time": total_time})
```

**为什么前 10 个 epoch 不统计？** GPU 有 warmup（第一次启动 kernel 有额外初始化开销），前几个 epoch 的计时包含 cold start 噪声。

---

## 三、在你们的 NPU 上怎么用？

### 3.1 工具对照

| NVIDIA 工具 | 你们的等效方式 | 适用场景 |
|---|---|---|
| `nvtx.range("name")` | 手动 timeit 打标 + JSON 日志 | 函数/算子级耗时划分 |
| `nsys profile` | 问驱动/编译器同事要 NPU profiler | 全链路时间线 |
| `nsys-ui` | 厂商 profiler 的 GUI | 可视化分析 |
| `torch.cuda.synchronize()` | 你们 NPU 的同步 API | 确保 GPU 侧执行完成 |

### 3.2 通用方案：自制计时器

如果你的 NPU 还没有成熟的 profiler，最简单的方案：

```python
import time
import json

class TimelineProfiler:
    def __init__(self):
        self.events = []

    def mark(self, name, start=True):
        self.events.append({
            "name": name,
            "ts": time.perf_counter(),
            "type": "start" if start else "end"
        })

    def report(self):
        """打印各阶段耗时"""
        pairs = {}
        for i in range(0, len(self.events), 2):
            name = self.events[i]["name"]
            elapsed = (self.events[i+1]["ts"] - self.events[i]["ts"]) * 1000
            pairs[name] = elapsed

        for name, ms in sorted(pairs.items(), key=lambda x: -x[1]):
            print(f"  {name:20s}: {ms:8.2f}ms")
        return pairs

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.events, f)


# 使用
prof = TimelineProfiler()

prof.mark("prefill")
prefill_out = npu_model(prefill_ids)
npu_synchronize()
prof.mark("prefill", False)

for i in range(100):
    prof.mark(f"decode_{i}")
    token = npu_model(token_id)
    npu_synchronize()
    prof.mark(f"decode_{i}", False)

prof.report()
#  prefill              :    18.34ms
#  decode_avg           :     2.15ms (取平均)
#  decode_total         :   215.0 ms
```

### 3.3 逐层计时

如果想定位到具体哪一层 Transformer Block 最慢：

```python
# 如果是 PyTorch 模型（有 hooks）
for name, layer in model.named_modules():
    layer.register_forward_hook(
        lambda m, inp, out, name=name:
            print(f"{name}: {(time.perf_counter()-start)*1000:.2f}ms")
    )

# 如果是编译后的 .hmm（Houmo），需要编译器/Runtime 支持逐层标记
# 问编译器同事：.hmm 能否输出 per-layer 耗时？
```

---

## 四、性能分析的三层递进

```
层级1: timeit → 知道"整个 forward 花了多久"
  └─ prefill: 18ms, decode: 2ms/token

层级2: NVTX/手动打标 → 知道"哪个阶段花了多久"
  └─ Attention: 12ms, FFN: 5ms, Norm: 0.3ms

层级3: Kernel profiler (ncu/厂商) → 知道"哪个 kernel 为什么慢"
  └─ matmul 的 Tensor Core 利用率只有 60%
  └─ 显存带宽利用率只有 30%（memory-bound）
```

从粗到细逐层下钻，先确定瓶颈大类，再精确定位。

---

## 五、思考题

1. **为什么 NVTX 需要 `torch.cuda.synchronize()` 才能准确反映 GPU 耗时？**
   - 提示：CUDA kernel 是异步执行的，CPU 侧 `with nvtx.range` 结束后 GPU 可能还在跑

2. **timeit 的 `number=100` 比单次测量好在哪？**
   - 提示：GPU 有时钟抖动、OS 调度干扰 → 多次平均降低方差

3. **你们的 NPU profiler 能提供什么级别的信息？等同于 Nsight 的哪一层？**

---

## 六、CS336 作业要点

### NVTX 作业

在训练循环中加 NVTX 标记，用 `nsys profile` 采集并查看时间线：

```python
for step in range(steps):
    with nvtx.range("data_loading"):
        x, y = data_loader.get_batch()

    with nvtx.range("forward"):
        logits = model(x)
        loss = loss_fn(logits, y)
        torch.cuda.synchronize()

    with nvtx.range("backward"):
        loss.backward()
        torch.cuda.synchronize()

    with nvtx.range("optimizer"):
        optimizer.step()
        torch.cuda.synchronize()
```

```bash
nsys profile --trace=cuda,osrt,nvtx -o report.qdrep python train_nvtx.py
# 打开 nsys-ui → 观察 forward/backward/optimizer 各占多少
```

### timeit 作业

用 timeit 统计训练各 step 的耗时，排除前 10 个 epoch 的 warmup：

```python
import timeit
start = timeit.default_timer()
# ... 训练若干 epoch
elapsed = timeit.default_timer() - start
print(f"avg per step: {elapsed / total_steps * 1000:.2f}ms")
```
