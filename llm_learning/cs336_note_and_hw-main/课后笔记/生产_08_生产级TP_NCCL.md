# 生产功能 08：生产级 Tensor Parallel + NCCL

> 学习路线：P0 必学 — 理解 nano-vllm SharedMemory demo → 生产级 NCCL 的全部差异
> 对应源码：HLIEvLLM `vllm/distributed/` (parallel_state, cuda_communicator, custom_all_reduce, pynccl) + nano-vllm `linear.py` (SharedMemory TP)

---

## 一、回顾：nano-vllm 的 TP（教学版）

在第3课我们学过 nano-vllm 的 TP 实现。它是教学级别的，核心机制是 **SharedMemory**：

[/nano-vllm/nanovllm/layers/linear.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/nano-vllm/nanovllm/layers/linear.py)

```python
# nano-vllm 的 TP — 简化版
class ColumnParallelLinear(LinearBase):
    def forward(self, x):
        # 每个 rank 只算自己那份 W_i · X
        return F.linear(x, self.weight, self.bias)  # 不通信，输出是部分的
        # ↑ 局部输出 Y_i，需要后续 RowParallel 来 reduce

class RowParallelLinear(LinearBase):
    def forward(self, x):
        # 每个 rank 的输入 x 已经是 ColumnParallel 的部分输出
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # ← 直接用 torch.distributed
        return y
```

nano-vllm 的 `dist.all_reduce` 背后用的是 PyTorch 内置的 NCCL ProcessGroup，但它只是一个 **black-box 调用**。在生产环境中，这远远不够。

---

## 二、nano-vllm TP → 生产级 TP：五个关键差异

| 维度 | nano-vllm (教学) | vLLM (生产) |
|---|---|---|
| **通信机制** | `dist.all_reduce()` → 依赖 PyTorch ProcessGroup | 多层回退链：Custom AR → FlashInfer AR → PyNccl → torch.distributed |
| **CUDA Graph 兼容** | 无 CUDA Graph | Custom AR + PyNCCL 在 graph capture 内安全运行 |
| **通信-计算重叠** | 无 | FP8 kernel (silu_mul_fp8) 内融合 all-reduce |
| **量化集成** | 无 | ColumnParallel/RowParallel 内建 `quant_method` 抽象 |
| **多节点** | 仅单节点 SharedMemory | 单节点用 Custom AR (IPC P2P)，多节点退回到 NCCL |

**核心问题：为什么 `dist.all_reduce()` 不够用？**

1. **CUDA Graph 不兼容**：`torch.distributed.all_reduce` 内部调用了额外的 CUDA API，在 graph capture 期间会报错
2. **性能不最优**：`torch.distributed` 走 NCCL 的通用路径，没有针对单节点 NVLink 全互联拓扑做优化
3. **无法量化融合**：量化场景下希望 all-reduce 能和量化操作的 kernel 融合

---

## 三、vLLM 的生产级 TP 通信栈

### 3.1 整体架构

```
模型层
  │
  ├── ColumnParallelLinear.forward()
  │     └── tensor_model_parallel_all_gather()         ← 拼接各 rank 的输出
  │
  └── RowParallelLinear.forward()
        └── tensor_model_parallel_all_reduce()         ← 求和各 rank 的部分积


                  ↓ 全部委托给 GroupCoordinator


GroupCoordinator.all_reduce(input_)
  │
  └── device_communicator.all_reduce(input_)
        │
        └── CudaCommunicator.all_reduce(input_)       ← 统一调度器
              │
              ├─ [P1] NCCL Symmetric Memory
              ├─ [P2] QuickAllReduce  (AMD MI300 only)
              ├─ [P3] FlashInferAllReduce
              ├─ [P4] CustomAllreduce ⭐ 核心：CUDA IPC P2P
              ├─ [P5] SymmMemCommunicator (Torch对称内存)
              ├─ [P6] PyNcclCommunicator (纯Python NCCL封装)
              └─ [P7] torch.distributed.all_reduce
```

### 3.2 统一入口：`communication_op.py`

[/vllm/distributed/communication_op.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/communication_op.py)

所有模型层都通过这四个函数发起 TP 通信，不直接碰 NCCL：

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce 跨 TP 组"""
    return get_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1):
    """All-gather 跨 TP 组 — ColumnParallel 收尾用"""
    return get_tp_group().all_gather(input_, dim)

def tensor_model_parallel_reduce_scatter(input_: torch.Tensor, dim: int = -1):
    """Reduce-scatter 跨 TP 组"""
    return get_tp_group().reduce_scatter(input_, dim)

def tensor_model_parallel_gather(input_: torch.Tensor, dst: int = 0, dim: int = -1):
    """Gather 到指定 rank — LogitsProcessor 用"""
    return get_tp_group().gather(input_, dst, dim)
```

### 3.3 生产级 `ColumnParallelLinear`

[/vllm/model_executor/layers/linear.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/model_executor/layers/linear.py#L582-L600)

```python
class ColumnParallelLinear(LinearBase):
    """
    Y = XA + b, A 沿第二维切分: A = [A_1, ..., A_p]
    """
    def __init__(self, input_size, output_size, bias=True,
                 gather_output=False, quant_config=None, ...):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, self.tp_size)

        # 量化感知：通过 quant_method 创建权重
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            ...
        )

    def forward(self, input_):
        # 量化框架下的矩阵乘法（可能是 FP8 kernel）
        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)  # 拼接
        else:
            output = output_parallel
        return output
```

**对比 nano-vllm：**
- nano-vllm：`F.linear(x, self.weight)` — 硬编码 FP16/FP32 GEMM
- vLLM：`self.quant_method.apply()` — 支持 INT8/FP8/INT4 等多种量化 GEMM

### 3.4 生产级 `RowParallelLinear`

[/vllm/model_executor/layers/linear.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/model_executor/layers/linear.py#L1544-L1570)

```python
class RowParallelLinear(LinearBase):
    """
    Y = XA + b, A 沿第一维切分
              | A_1 |
          A = | ... |
              | A_p |
    """
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=True,      # 输入是否已经按列切分
                 reduce_results=True,          # 是否做 all-reduce
                 quant_config=None, ...):
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.quant_method.create_weights(...)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_   # 输入已切分（来自上一层的 ColumnParallel）
        else:
            # 需要先按最后一维切分
            split_input = split_tensor_along_last_dim(input_, self.tp_size)
            input_parallel = split_input[self.tp_rank].contiguous()

        # 量化 GEMM
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)  # all-reduce
        else:
            output = output_parallel
        return output
```

**SwiGLU 中的 TP 模式（回顾第3课）：**

```
ColumnParallel (gate_proj + up_proj):
    每个 rank: W_gate_i ∈ R^(hidden/tp, intermediate/tp)
    每个 rank: W_up_i   ∈ R^(hidden/tp, intermediate/tp)

    → gate_i = SiLU(X_i @ W_gate_i)
    → up_i = X_i @ W_up_i
    → intermediate_i = gate_i ⊙ up_i                 ← 第 i 块部分积

RowParallel (down_proj):
    W_down_i ∈ R^(intermediate/tp, hidden)

    → output_i = intermediate_i @ W_down_i
    → output = all_reduce([output_0, ..., output_{p-1}])    ← 这里 all-reduce
```

SwishGLU 的中间激活不需要 all-gather！因为 `gate ⊙ up` 已经在每个 rank 本地完成，只需在 `down_proj` 后做一次 all-reduce。

---

## 四、Custom AllReduce — 单节点性能之王

Custom AllReduce 是 vLLM 最核心的 TP 优化：**绕过 NCCL，直接用 CUDA IPC 实现 P2P all-reduce**。

### 4.1 原理：为什么比 NCCL 快？

```
NCCL all-reduce (Ring):
  Rank0 → Rank1 → Rank2 → Rank3 → Rank0
  4 次 P2P 传输，总延迟 = 4 × hop_delay
  
Custom AR (1-stage, NVLink 全互联):
  Rank0 ─→ Rank1, Rank2, Rank3   (同时写)
  Rank1 ─→ Rank0, Rank2, Rank3   (同时写)
  ... 
  1 次 P2P 写，每个 block 直接从所有 rank 读
  总延迟 = 1 × 读延迟（因为 NVLink 全互联，所有 rank 可直接访问彼此显存）
```

### 4.2 Python 端：`CustomAllreduce` 类

[/vllm/distributed/device_communicators/custom_all_reduce.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/device_communicators/custom_all_reduce.py)

```python
class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]   # ← 只支持这些配置

    def __init__(self, group, device, max_size=8192*1024):
        # 1. 分配共享缓冲区（CUDA IPC 可访问）
        self._ptr = ops.allocate_shared_buffer_and_handle(
            group, device, max_size
        )
        # 2. 通过 CPU group (gloo) 交换各 rank 的 IPC 句柄
        # 3. 每个 rank 获取所有其他 rank 的显存指针
        #    → 可以直接读写其他 GPU 的显存！

    def should_custom_ar(self, inp):
        """判断是否应该用 custom allreduce"""
        # - 必须在 TP 组内
        # - 张量大小 % 16 == 0
        # - 张量是弱连续的
        # - 大小不超过 max_size
        # - 所有 GPU 之间支持 P2P (NVLink/PCIe P2P)
        return ...

    def custom_all_reduce(self, inp):
        """调用 C++/CUDA kernel"""
        out = torch.empty_like(inp)
        ops.all_reduce(self._ptr, inp, out, ...)  # → custom_all_reduce.cu
        return out
```

### 4.3 CUDA Kernel：两种算法

[/csrc/custom_all_reduce.cuh](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/csrc/custom_all_reduce.cuh)

**算法选择逻辑**（第588-600行）：

```cuda
if (world_size_ == 2) {
    cross_device_reduce_1stage;    // 2 GPU 永远用 1-stage
} else if (fully_connected_) {
    if (world_size_ <= 4 && bytes < 512KB)
        cross_device_reduce_1stage;    // 小数据 → 1-stage
    else
        cross_device_reduce_2stage;    // 大数据 → 2-stage
}
// 可通过 VLLM_CUSTOM_ALLREDUCE_ALGO=1stage|2stage 强制选择
```

**1-stage（小数据，全互联）：**

```cuda
template <typename T, int ngpus>
__global__ void cross_device_reduce_1stage(
    RankData* _dp,       // 所有 rank 的显存指针
    RankSignals sg,      // 同步信号
    Signal* self_sg,
    T* result, int rank, int size
) {
    auto dp = *_dp;
    barrier_at_start<ngpus>(sg, self_sg, rank);  // ← P2P 原子同步
    
    // 每个线程读所有 rank 的数据并求和
    for (int idx = ...; idx < size; idx += stride) {
        result[idx] = packed_reduce<P, ngpus, A>(
            (const P**)&dp.ptrs[0], idx
        );  // sum over all ranks
    }
    
    barrier_at_end<ngpus, true>(sg, self_sg, rank);
}
```

**2-stage（大数据）：**

```cuda
// Stage 1: Reduce-Scatter
//   每个 rank 只归约总数据量的 1/ngpus（分片归约）
//   Rank i: 归约所有 rank 的第 i 个分片 → 写到自己缓冲区
for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
}

barrier_at_end<ngpus>(sg, self_sg, rank);

// Stage 2: All-Gather
//   每个 rank 从所有 rank 读它们归约好的分片
for (int i = 0; i < ngpus; i++) {
    ((P*)result)[idx + i * part] = tmps[i][idx];
}
```

**P2P 原子同步机制**（第260-284行）：

```cuda
// barrier: 所有 GPU 的 block 0 达到同一进度
// 使用 GPU 间的原子操作实现（不是 CPU 端同步！）
__scoped_atomic_store_n(
    &sg.signals[threadIdx.x]->end[blockIdx.x][rank],
    flag, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM
);
// 等待其他 rank
while (
    __scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                           __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < flag
);
```

**关键：** 这个 barrier 是 **GPU 端 block 级别的细粒度同步**，不是 CPU 端全局 barrier。多个 SM 上的 block 可以流水线执行，不需要等所有 block 完成才开始下一阶段。

### 4.4 限制条件

- **只支持单节点**（需要 NVLink / PCIe P2P 全互联）
- **只支持 2/4/6/8 GPU**（模板特化，编译时确定通道数）
- **只支持 TP 组**（非 DP/PP/EP 组不用 custom AR）
- **数据量有上限**：H100 上 4 way → 最大 4MB，8 way → 最大 2MB

---

## 五、CudaCommunicator — 智能调度器

[/vllm/distributed/device_communicators/cuda_communicator.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/device_communicators/cuda_communicator.py#L25-L231)

```python
class CudaCommunicator(DeviceCommunicatorBase):
    def __init__(self, cpu_group, device, device_group, unique_name, ...):
        # 只对 TP 组启用 custom allreduce
        if "tp" not in unique_name:
            use_custom_allreduce = False
        else:
            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE

        # 初始化所有后端（按优先级）
        self.pynccl_comm = PyNcclCommunicator(...)      # [P6] 总是初始化
        self.ca_comm = CustomAllreduce(...)              # [P4] TP 组才初始化
        self.fi_ar_comm = FlashInferAllReduce(...)       # [P3] 可选
        self.symm_mem_comm = SymmMemCommunicator(...)    # [P5] 可选
        self.qr_comm = QuickAllReduce(...)               # [P2] AMD only

    def all_reduce(self, input_):
        # 按优先级尝试每个后端，第一个成功的返回
        
        # [P1] NCCL 对称内存 all-reduce
        if should_nccl_symm_mem_allreduce(world_size, input_):
            out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)
            if out is not None: return out
        
        # [P2] AMD MI300 Quick AllReduce (量化)
        if qr_comm and qr_comm.should_quick_allreduce(input_):
            return qr_comm.quick_all_reduce(input_)
        
        # [P3] FlashInfer AllReduce
        if fi_ar_comm and fi_ar_comm.should_use_fi_ar(input_):
            return fi_ar_comm.all_reduce(input_)
        
        # [P4] Custom AllReduce ⭐
        if ca_comm and ca_comm.should_custom_ar(input_):
            return ca_comm.custom_all_reduce(input_)
        
        # [P5] Torch 对称内存
        if symm_mem_comm and symm_mem_comm.should_use_symm_mem(input_):
            return symm_mem_comm.all_reduce(input_)
        
        # [P6] PyNCCL — 纯 Python NCCL 封装
        if pynccl_comm and not pynccl_comm.disabled:
            return pynccl_comm.all_reduce(input_)
        
        # [P7] 最终回退 → torch.distributed.all_reduce
        out = input_.clone()
        torch.distributed.all_reduce(out, group=self.device_group)
        return out
```

**这个优先级链的精妙之处：**

1. Custom AR 最快（绕过 NCCL 开销，P2P 直接读写），但有限制（大小、拓扑、单节点）
2. 当 custom AR 不可用时，自动降级到 PyNccl（CUDA Graph 兼容的 NCCL）
3. 最后才用 `torch.distributed.all_reduce`（最慢但最通用）

---

## 六、PyNCCL — CUDA Graph 兼容的 NCCL

### 6.1 为什么需要自己封装 NCCL？

[/vllm/distributed/device_communicators/pynccl_wrapper.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/device_communicators/pynccl_wrapper.py)

```python
class NCCLLibrary:
    """通过 ctypes 直接加载 libnccl.so，绕过 PyTorch"""
    
    def __init__(self):
        self.lib = ctypes.CDLL(find_nccl_library())  # libnccl.so.2
        
        # 直接绑定 C API
        self.ncclAllReduce = self.lib.ncclAllReduce
        self.ncclAllGather = self.lib.ncclAllGather
        self.ncclReduceScatter = self.lib.ncclReduceScatter
        self.ncclSend = self.lib.ncclSend
        self.ncclRecv = self.lib.ncclRecv
        self.ncclBroadcast = self.lib.ncclBroadcast
        # ...
```

**为什么不用 `torch.distributed.all_reduce`？**

```
torch.distributed.all_reduce 的问题：
  1. 内部包含 cudaStreamSynchronize 等 CUDA API 调用
  2. CUDA Graph capture 期间不允许这些 API → graph capture 会失败
  3. PyTorch ProcessGroup 有自己的 stream 管理，可能与 vLLM 的 stream 冲突

PyNCCL 的解决方案：
  1. 直接用 ctypes 调 libnccl.so → 零额外 PyTorch 开销
  2. 所有 NCCL 操作都在用户指定的 stream 上
  3. 在 CUDA Graph 内安全运行（只需确保 communicator 在 capture 前已初始化）
```

### 6.2 PyNcclCommunicator

[/vllm/distributed/device_communicators/pynccl.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/device_communicators/pynccl.py)

```python
class PyNcclCommunicator:
    def all_reduce(self, tensor, op=ReduceOp.SUM):
        """直接调 NCCL C API"""
        self.nccl.ncclAllReduce(
            tensor.data_ptr(),       # input
            out.data_ptr(),          # output (in-place 时相同)
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,               # ncclComm_t
            cuda_stream,             # CUDA stream
        )
```

---

## 七、TP 在 SwiGLU 中的实际数据流

回顾第3课学到的 SwiGLU TP 模式，现在用生产级视角重新看：

```
输入 X: [batch, seq, hidden]
   
    TP 切分: X → [X_0, X_1]  (hidden 维切分)

Rank 0:                                    Rank 1:
  X_0 [bs, hidden/2]                       X_1 [bs, hidden/2]
    │                                         │
    ▼                                         ▼
┌─────────────────┐                    ┌─────────────────┐
│ gate_proj (CP)  │                    │ gate_proj (CP)  │
│ W_gate_0        │                    │ W_gate_1        │
│ [hidden/2,      │                    │ [hidden/2,      │
│  inter/2]       │                    │  inter/2]       │
└────────┬────────┘                    └────────┬────────┘
         │ gate_0                              │ gate_1
         ▼                                      ▼
    SiLU(gate_0)                           SiLU(gate_1)
         │                                      │
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│ up_proj (CP)    │                    │ up_proj (CP)    │
│ W_up_0          │                    │ W_up_1          │
└────────┬────────┘                    └────────┬────────┘
         │ up_0                                │ up_1
         │                                      │
         ▼                                      ▼
    gate_0 ⊙ up_0                         gate_1 ⊙ up_1
    = inter_0                              = inter_1
         │                                      │
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│ down_proj (RP)  │                    │ down_proj (RP)  │
│ W_down_0        │                    │ W_down_1        │
│ [inter/2,       │                    │ [inter/2,       │
│  hidden]        │                    │  hidden]        │
└────────┬────────┘                    └────────┬────────┘
         │ out_0                                │ out_1
         │                                      │
         └────────── all_reduce ────────────────┘
                        │
                        ▼
                   output [bs, hidden]
```

**通信量分析**（Qwen3-8B 单层，bs=1, hidden=4096, fp16, tp=4）：

```
ColumnParallel (gate + up):
  中间激活 (inter): bs × inter_dim/tp × 2 = 1 × (4096×8/3 / 4) × 2 ≈ 5.5K elements
  不需要通信！（只在本地做 gate ⊙ up）

RowParallel (down):
  all_reduce: bs × hidden = 1 × 4096 = 4K elements = 8KB

单层总通信量: 8KB
32 层总通信量: 256KB
```

**对比没有 custom AR 时，256KB 意味着什么：**

- 用 `torch.distributed.all_reduce`：约 30-50μs/次 × 32 = 1.0-1.6ms
- 用 Custom AR 1-stage：约 10-15μs/次 × 32 = 0.3-0.5ms

每步 decode 省 ~1ms，每天百万 token 省数十秒。

---

## 八、通信-计算重叠：Helion FP8 Kernel

[/HLIEvLLM-1.4.0rc0/vllm/kernels/helion/](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/kernels/helion/)

Houmo 的 Helion FP8 kernel（`silu_mul_fp8` 等）是通信-计算重叠的实际例子：

```python
# 概念性代码：量化 kernel 内融合 all-reduce
# 传统的两步走:
output_fp16 = F.linear(x, weight)  # FP16 GEMM
output_fp16 = all_reduce(output_fp16)
output_fp8 = quantize(output_fp16)  # FP16 → FP8

# Helion 融合后:
# all-reduce 和 quantize 在 CUDA graph 内同一个 kernel 完成
output_fp8 = silu_mul_fp8_quant(x, weight)  # 内建 all-reduce
```

虽然 Houmo NPU 用的是静态图而非 CUDA Graph，但这个设计思路是相同的：**把通信和计算放在同一条指令流中，避免 kernel launch 开销。**

---

## 九、Houmo NPU 的 TP 实现

在上一课的 [prefill_pp_decode_tp.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/houmo-examples-xh2/apis/inferences/qwen3_pipeline/prefill_pp_decode_tp.py#L549-L553) 中，Houmo 的 Decode TP 模式使用了 `DevManager`：

```python
# Houmo TP: DevManager 统一管理多 NPU
dev_manager = tcim.runtime.DevManager(model_params["device_id"], "Xh2HalBackend")
weight_manager = tcim.runtime.WeightManager(dev_manager)
option_decode = tcim.runtime.Option(weight_manager)

# 加载 decode.hmms（ndevice=全芯片数，NPU 内部处理 TP 切分和通信）
decode_model = tcim.runtime.load(model_params["decode_path"], option_decode)
```

**Houmo TP vs GPU TP 的区别：**

| | GPU TP (vLLM) | NPU TP (Houmo) |
|---|---|---|
| **权重切分** | Python 层 `ColumnParallel/RowParallel` | 编译时 `.hmms` 内建切分 |
| **通信机制** | NCCL / Custom AR（显式调用） | DevManager 封装（runtime 内部） |
| **CUDA Graph** | 有，decode 阶段用 | 无，但静态图天然等效 |
| **Python 层** | 可见 `all_reduce` 调用 | 不可见，`decode_model.run()` 内部处理 |
| **灵活性** | 高，可自由组合 TP+PP+EP | 低，编译期决定，运行时不可变 |

**核心差异：** GPU TP 是在 Python 层显式管理通信，Houmo TP 是在 `.hmms` 编译时和 DevManager 中隐式处理的。对外部使用者来说，Houmo TP 就是一个 "multidevice 黑盒"。

---

## 十、TP 调优参数

### 10.1 环境变量

| 环境变量 | 作用 |
|---|---|
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | 强制选算法：`1stage`/`oneshot` 或 `2stage`/`twoshot` |
| `VLLM_SKIP_P2P_CHECK` | 跳过 P2P 可用性检测（信任驱动报告） |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | 启用 Torch 对称内存 all-reduce |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | 启用 FlashInfer 融合 all-reduce |
| `VLLM_NCCL_SO_PATH` | 指定 libnccl.so 路径 |

### 10.2 Custom AR 大小阈值

[/vllm/distributed/device_communicators/all_reduce_utils.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/distributed/device_communicators/all_reduce_utils.py)（第31-49行）

```python
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    2: {  # world_size=2
        (9, 0):  8 * 1024 * 1024,  # H100: 8MB
        (10, 0): 8 * 1024 * 1024,  # B200: 8MB
        (10, 3): 8 * 1024 * 1024,  # GB200: 8MB
    },
    4: {
        (9, 0):  4 * 1024 * 1024,  # H100: 4MB
        (10, 0): 4 * 1024 * 1024,
        (10, 3): 4 * 1024 * 1024,
    },
    # ...
}
```

超过这个大小的 all-reduce 会自动回退到 PyNCCL。

---

## 十一、白话总结

```
nano-vllm TP = 教学版：
  "dist.all_reduce() 调用一下就行"
  用 SharedMemory pickle 传参数

vLLM TP = 生产版：
  Custom AR:       直接 P2P 读隔壁 GPU 的显存，不要 NCCL 在中间"翻译"
  PyNCCL:          自己封装 NCCL C API，让 CUDA Graph 能正常工作
  CudaCommunicator: 7 层回退链，"哪个能用、哪个最快就用哪个"
  量化集成:         all-reduce 在 FP8 kernel 里顺手做了，不等 GEMM 结束
  
就像:
  nano-vllm = 快递（dist.all_reduce）每个包裹送到中转站再分发
  vLLM     = 直达专线（Custom AR）直接从邻居家取，不经过中转站
             快递不方便？（PyNCCL）那还是用快递
             快递也不行？（torch.distributed）那用最慢但保证能到的
```

---

## 十二、思考题

1. **Custom AR 的 1-stage 和 2-stage 分别适合什么场景？为什么 2 GPU 总是用 1-stage？**
   - 提示：1-stage 要求所有 rank 都能直接读彼此的显存（全互联），2-stage 不要求

2. **为什么 Custom AR 只支持 2/4/6/8 GPU，不支持 3/5/7？这和 NVLink 拓扑有什么关系？**
   - 提示：NVLink 全互联拓扑通常是 2 的幂（2/4/8），6 是特殊的 NVSwitch 拓扑

3. **`CudaCommunicator` 的优先级链如果在 multi-node 环境下会发生什么？Custom AR 为什么被自动跳过？**
   - 提示：跨节点 GPU 之间没有 NVLink，不支持 CUDA IPC

4. **Houmo 的 `DevManager` TP 和 vLLM 的 Python 层 TP 各有什么优缺点？**
   - 提示：编译期 vs 运行时、灵活性 vs 性能、可调试性

5. **如果 TP size=8 时 RowParallel 的 all-reduce 数据量是 16KB，Custom AR 和 NCCL 哪个更快？为什么？**
   - 提示：参考 CUSTOM_ALL_REDUCE_MAX_SIZES 和算法选择逻辑
