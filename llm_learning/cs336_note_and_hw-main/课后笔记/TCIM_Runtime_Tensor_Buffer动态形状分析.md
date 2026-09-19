# TCIM Runtime Tensor/Buffer 动态形状分析

> 分析对象：`houmo_tcim_runtime_xh2_linux_x86_64-1.4.0`
> 结论：**不支持动态形状**。Buffer 大小固定不可 Resize，Tensor shape 不可变，Module 输入输出 shape 在 `.hmm` 编译期写死。

---

## 一、Buffer：固定大小，不可 Resize

Buffer 的构造函数只有一种模式 — 一次性分配固定大小：

```cpp
static Buffer CreateDeviceBuffer(size_t size, ...);  // 分配 size 字节，之后不可变
static Buffer CreateHostBuffer(size_t size, ...);     // 同上
size_t Size() const;                                  // 只能查询，不能修改
```

**没有** `Resize()`、`Realloc()`、`SetSize()` 等方法。分配后大小锁定。

子 Buffer 只是原 Buffer 的视图（共享内存），不改变大小：

```cpp
Buffer GetSubBuffer(size_t size, size_t offset = 0) const;  // 视图，不是新分配
Buffer Clone(bool auto_copy = true) const;                   // 复制，大小不变
```

---

## 二、Tensor：形状不可变

Tensor 由 `TensorInfo + Buffer` 构造：

```cpp
Tensor(const TensorInfo& info, const Buffer& buffer);
static Tensor CreateDeviceTensor(const TensorInfo& info, size_t mem_size, ...);
```

TensorInfo 存储 shape/dtype/strides 等信息，创建后即固定：

```cpp
// TensorInfo 只有查询方法，没有修改方法
std::vector<int64_t> Shape() const;     // 只读
int64_t Rank() const;                   // 只读
size_t Size() const;                    // 只读
DataType GetDataType() const;           // 只读
size_t DataTypeSize() const;            // 只读：单元素字节数
size_t MemSize() const;                 // 只读：显存总占用
DataFmt Format() const;                 // 只读：数据格式
```

**没有** `Reshape()`、`SetShape()`、`Slice()` 等动态修改形状的 API。

**能做的变换**（都不改变 shape）：

```cpp
Tensor AsType(DataType dtype, bool auto_cast = true) const;   // 改 dtype，shape 不变
Tensor Clone(bool auto_copy = true) const;                     // 复制，shape 不变
Status CastTo(TensorInfo info);                                // 按另一个 info 重排数据，不是动态 reshape
```

---

## 三、Module 输入：编译期固定

模型的输入/输出 shape 在 `.hmm` 编译时写死，Runtime 只能查询不能修改：

```cpp
// Module API: 只能查询
TensorInfo GetInputInfo(int64_t i) const;      // 查询编译期 shape
int64_t GetInputNum() const;                   // 输入个数
std::string GetInputName(int64_t i) const;     // 输入名称
// 没有 SetInputShape() 或类似的修改接口
```

### Qwen3 demo 的实际验证

从 `qwen_base.py` 可以看到所有 shape 都是编译期常量：

```python
# 从编译好的 .hmm 文件中读取固定 shape
self.prefill_length = self.prefill.get_input_info(0).shape[1]       # 编译时写死
self.embedding_len   = self.prefill.get_input_info(0).shape[2]      # 编译时写死
self.context_max_length = self.decode.get_input_info(3).shape[2]    # 编译时写死
self.batch = self.decode.get_input_info(0).shape[0]                 # 编译时写死
```

Preill 和 Decode 甚至需要**分开编译**成两个 `.hmm` 文件，正是因为它们有完全不同的静态 shape：

```python
self.prefill = tcim.runtime.load("qwen3-8b_prefill.hmm")   # 大矩阵 fixed shape
self.decode  = tcim.runtime.load("qwen3-8b_decode.hmm")    # 1 token fixed shape
```

---

## 四、变长输入的处理：Padding + Valid Length

Runtime 不支持动态形状，变长输入通过 Padding + Valid Length 机制处理：

```
实际 prompt: ["你好，世界"]         实际长度 = 5 tokens
编译期固定:  shape = (1, 512, d_model)  ← 最大长度 512

处理方式:
  input_ids = pad([101, 102, 103, 104, 105, 0, 0, ..., 0], length=512)
  valid_length = 5   ← 告诉模型实际有效长度是 5

模型内部的 attention 用 valid_length 做 mask，忽略 padding 部分
```

Qwen3 demo 中对应的两个额外输入：

```python
self.prefill_valid_length_name = self.prefill.get_input_name(1)   # "实际有几个 token"
self.prefill_current_length_name = self.prefill.get_input_name(2) # "当前总长度"
```

---

## 五、对推理引擎的影响

| 影响 | 说明 |
|---|---|
| **Prefill 有显存浪费** | 短 prompt 也必须 pad 到最大长度，浪费计算和显存 |
| **Decode 的 batch size 固定** | 不能动态添加新请求，无 continuous batching 基础 |
| **KV Cache shape 固定** | 所有请求的 KV Cache 大小一样，不支持 PagedAttention 的分页机制 |
| **不支持 dynamic batching** | 无法在 decode 过程中加入新的请求 |

---

## 六、如果要支持动态形状

若想让 Runtime 支持动态 batch / 动态序列长度，需要各层配合改动：

| 层 | 需要改什么 |
|---|---|
| **编译器** | 编译时标注 dynamic dim，支持 shape range（min/max/opt），生成带动态描述符的 `.hmm` |
| **Runtime (tcim_runtime)** | `TensorInfo` 支持 dynamic shape（min/max/opt），`Module::Run` 前可 `SetInputShape(actual_shape)` |
| **HAL** | NPU 上的 DMA 描述符支持可变长度，不接受预分配固定大小的 buffer |
| **驱动** | 每次 `Run` 时根据实际 shape 重新计算 device buffer 分配 |

**本质**：这是从**静态图**（shape 编译时确定）升级到**动态图**（shape 运行时可变）。

---

## 七、与 nano-vllm 的对比

| | TCIM Runtime | nano-vllm |
|---|---|---|
| **模型表示** | 编译后的静态 `.hmm` 文件 | PyTorch `nn.Module`（动态图） |
| **输入 shape** | 编译时固定，Padding 处理 | 每次 forward 可不同 shape |
| **Batch 机制** | 固定 batch size，prefill/decode 分两个模型 | Continuous Batching，prefill/decode 同一模型 |
| **KV Cache** | 固定 shape 的命名 tensor | PagedAttention + BlockManager 动态管理 |
| **动态形状** | 不支持 | PyTorch 原生支持 |

这也解释了为什么 nano-vllm 的 `scheduler.py` 和 `block_manager.py` 如此关键 — 它们恰好在**调度层面**解决了硬件层面静态形状带来的灵活性不足问题。
