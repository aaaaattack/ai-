# 生产_11：LoRA 推理原理与实现

> P2 了解专题 1/4 | 源码参考：HLIEvLLM-1.4.0rc0 `vllm/lora/`、`vllm/config/lora.py`

---

## 一、LoRA 原理回顾

### 1.1 核心公式

**LoRA (Low-Rank Adaptation, Hu et al. 2021)** 的核心思想：预训练权重冻结，增量更新用两个低秩矩阵的乘积表示。

```
Y = W₀ @ X  +  (α/r) · B @ A @ X
  └─基座输出─┘  └── LoRA 增量 ──┘
```

其中：
- `W₀ ∈ R^(d_out × d_in)`：冻结的预训练权重
- `A ∈ R^(r × d_in)`：低秩矩阵 A（"shrink"，降维）
- `B ∈ R^(d_out × r)`：低秩矩阵 B（"expand"，升维）
- `r ≪ min(d_in, d_out)`：秩，通常 1~64
- `α`：缩放因子（通常 = r 的倍数），实际缩放 = `α / r`

**参数量对比**：全量微调需要 `d_in × d_out` 个参数，LoRA 只需要 `r × (d_in + d_out)`。例如 `r=16, d_in=4096, d_out=4096`：全量 16.7M 参数 vs LoRA 131K 参数（~130x 减少）。

### 1.2 应用于 Attention 权重

LoRA 最常见的应用目标是对 Transformer 的 Q、K、V、O 投影矩阵做微调：

```
Q = W_Q @ X + B_Q @ A_Q @ X
K = W_K @ X + B_K @ A_K @ X
V = W_V @ X + B_V @ A_V @ X
O = W_O @ X + B_O @ A_O @ X
```

一个典型的 LoRA adapter 包含约 4 组 `(A, B)` 矩阵对，每层都有。例如 Qwen3-7B 有 28 层，每层 4 组 `r=16` 的 LoRA → 约 28×4×16×(4096+4096) ≈ 14.7M 参数 ≈ 约 60MB（bfloat16）。

---

## 二、推理时 LoRA 的核心挑战

| 挑战 | 传统方案 | vLLM 的 Punica 方案 |
|------|----------|---------------------|
| **多 adapter 并发** | 每个 batch 只用 1 个 adapter | 同一 batch 支持 N 个不同 adapter |
| **Adapter 切换开销** | 重新加载所有权重 | 所有权重预分配在 GPU 堆叠缓冲区，按 slot 索引切换 |
| **计算效率** | `B@A@X` 逐 adapter 串行 | **SGMV (Segmented Gather Matrix-Vector)**：按 lora_id 分组并行 |
| **显存** | 每个 adapter 独立分配显存 | 固定大小的预分配 `lora_a_stacked` / `lora_b_stacked` + LRU 淘汰 |

**Punica 论文** (Chen et al., 2023): "Punica: Multi-Tenant LoRA Serving" 提出的核心创新就是 SGMV kernel。

---

## 三、vLLM LoRA 架构总览

```
请求到达 (带 lora_name)
    │
    ▼
┌──────────────────────────────────────────────────┐
│               LoRAModelRunnerMixin               │
│  • _load_adapter(lora_name) → LoRAModel          │
│  • set_active_loras(input_batch)                 │
│      → prompt_lora_mapping / token_lora_mapping  │
└──────────────────────┬───────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────┐
    │           LoRAModelManager (Engine Core)      │
    │  • _registered_adapters: LRU cache (CPU)      │
    │  • _active_adapters: LRU cache (GPU slots)    │
    │  • modules: dict[str, BaseLayerWithLoRA]      │
    │  • activate_adapter(id) → set_lora(slot, A,B)│
    │  • set_adapter_mapping(mapping)               │
    └──────────────────┬───────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────┐
    │          PunicaWrapperGPU (Triton 内核)        │
    │  • LoRAKernelMeta: token→lora_id 分组排序     │
    │  • add_shrink():  X @ A^T  (SGMV)            │
    │  • add_expand():  (X@A) @ B^T  (SGMV)        │
    │  • add_lora_linear(): shrink + expand 组合    │
    └──────────────────┬───────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────┐
    │         BaseLayerWithLoRA (各层 forward)       │
    │  output = W@X + bias                         │
    │  output += lora_shrink + lora_expand          │
    │  可选：双 CUDA Stream 并行 (base ∥ lora)      │
    └──────────────────────────────────────────────┘
```

---

## 四、配置系统：LoRAConfig

[`vllm/config/lora.py#L31-L73`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/config/lora.py#L31-L73)

```python
class LoRAConfig:
    max_lora_rank: int = 16       # 最大秩 (1/8/16/32/64/128/256/320/512)
    max_loras: int = 1            # 单 batch 最大 active LoRA 数量 = GPU slot 数
    max_cpu_loras: int | None     # CPU 内存中缓存的 LoRA 数量，默认 = max_loras
    fully_sharded_loras: bool     # 是否对 LoRA 做全 TP 分片 (S-LoRA)
    lora_dtype: str = "auto"      # LoRA 权重数据类型
    target_modules: list[str] | None  # 限制 LoRA 作用的目标模块后缀
    specialize_active_lora: bool  # 按 active LoRA 数 (2的幂) 独立捕获 CUDA Graph
```

**关键配置含义**：

- `max_loras`：决定 GPU 上预分配的 `lora_a_stacked` / `lora_b_stacked` 的 slot 数量。例如设为 4 意味着最多同时服务 4 个不同的 LoRA adapter。
- `max_cpu_loras`：CPU 内存中缓存的 LoRA 数量（LRU cache 容量）。通常远大于 `max_loras`。
- `specialize_active_lora`：设为 True 时，vLLM 会为不同数量的 active LoRA（1、2、4、8...）各自捕获一个 CUDA Graph。好处是 kernel grid 更紧凑，代价是内存和启动时间增加。

---

## 五、权重存储与加载

### 5.1 LoRALayerWeights — 单层 LoRA 权重

[`vllm/lora/lora_weights.py#L13-L42`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/lora_weights.py#L13-L42)

```python
class LoRALayerWeights:
    module_name: str
    rank: int
    lora_alpha: int
    lora_a: torch.Tensor    # shape: (rank, input_dim)
    lora_b: torch.Tensor    # shape: (output_dim, rank)
    scaling: float          # = lora_alpha / rank

    def optimize(self):
        """将 scaling 因子融合进 lora_b，运行时只需 scaling=1"""
        if self.scaling == 1:
            return self
        self.lora_b *= self.scaling    # B = B * (α/r)
        self.scaling = 1
        return self
```

**`optimize()` 的作用**：将运行时乘法 `(α/r) * B @ A @ X` 变为 `B @ A @ X`（因为 B 已经乘以了 α/r），避免每个 forward 都做一次 scale 乘法。

### 5.2 PackedLoRALayerWeights — 合并层 LoRA 权重

[`vllm/lora/lora_weights.py#L99-L152`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/lora_weights.py#L99-L152)

对于 **QKV 合并投影**（如 `qkv_proj` 包含 Q、K、V 三个子投影）：

```python
class PackedLoRALayerWeights(LoRALayerWeights):
    lora_a: list[torch.Tensor | None]  # [lora_a_q, lora_a_k, lora_a_v]
    lora_b: list[torch.Tensor | None]  # [lora_b_q, lora_b_k, lora_b_v]
    scaling: list[float]

    @classmethod
    def pack(cls, loras):
        """将分散的 LoRA 合并为一个 PackedLoRA"""
        first_lora = next(lora for lora in loras if lora is not None)
        # 合并 lora_a 列表、lora_b 列表
        obj = cls(module_name, rank,
            [lora.lora_alpha if lora else None for lora in loras],
            [lora.lora_a if lora else None for lora in loras],
            [lora.lora_b if lora else None for lora in loras],
            scaling=[1 if lora else None for lora in loras])
        return obj
```

### 5.3 MoE LoRA 权重堆叠

[`vllm/lora/lora_weights.py#L155-L228`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/lora_weights.py#L155-L228)

对于 Mixture-of-Experts 模型（如 Qwen3-MoE），每个 expert 都有独立的 LoRA：

```python
@classmethod
def pack_moe(cls, loras, module_name, is_non_gated_moe=False):
    """将 per-expert LoRA 堆叠为 3D tensor"""
    for eid in range(len(loras) // 3):
        w1_lora_a_lst.append(w1_lora.lora_a)
        w2_lora_a_lst.append(w2_lora.lora_a)
        w3_lora_a_lst.append(w3_lora.lora_a)
        # ... 同理 lora_b

    w1_lora_a = torch.stack(w1_lora_a_lst, dim=0)  # (E, rank, input)
    w2_lora_a = torch.stack(w2_lora_a_lst, dim=0)
    w1_lora_b = torch.stack(w1_lora_b_lst, dim=0)  # (E, output, rank)
    w2_lora_b = torch.stack(w2_lora_b_lst, dim=0)
```

### 5.4 加载全流程

```
1. LoRARequest 到达（lora_name, lora_path）
   │
2. PEFTHelper.from_local_dir()
   → 解析 adapter_config.json (r, lora_alpha, target_modules, ...)
   │
3. LoRAModel.from_local_checkpoint(path)
   → 读取 adapter_model.safetensors
   → parse_fine_tuned_lora_name() 解析 tensor 名
       "model.layers.0.self_attn.q_proj.lora_A.weight"
       → module_name = "model.layers.0.self_attn.q_proj"
       → is_lora_a = True
   → 构建 LoRALayerWeights (lora_a, lora_b)
   │
4. LoRAModelManager._add_adapter()
   → _create_merged_loras_inplace()
       → PackedLoRALayerWeights.pack() 合并 QKV
       → PackedLoRALayerWeights.pack_moe() 堆叠 MoE
       → optimize() 融合 scaling
   → 存入 _registered_adapters (LRU cache, CPU)
   │
5. activate_adapter(lora_id)
   → 分配空闲 GPU slot index
   → 遍历所有 modules:
       module.set_lora(index, lora_a, lora_b)
         → lora_a_stacked[index].copy_(lora_a)
         → lora_b_stacked[index].copy_(lora_b)
```

---

## 六、GPU 显存布局：堆叠缓冲区

### 6.1 预分配结构

[`vllm/lora/layers/base_linear.py#L106-L157`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/layers/base_linear.py#L106-L157)

```python
def create_lora_weights(self, max_loras, lora_config, model_config):
    """为每个 LoRA adapter 预分配 GPU 缓冲区"""
    # lora_a_stacked: (max_loras, 1, rank, input_dim)
    self.lora_a_stacked = tuple(
        torch.zeros(max_loras, 1, lora_a_out_size, self.input_size,
                    dtype=lora_config.lora_dtype, device=self.device)
        for _ in range(self.n_slices)
    )
    # lora_b_stacked: (max_loras, 1, output_dim, rank)
    self.lora_b_stacked = tuple(
        torch.zeros(max_loras, 1, lora_b_out_size, lora_config.max_lora_rank,
                    dtype=lora_config.lora_dtype, device=self.device)
        for _ in range(self.n_slices)
    )
```

**维度的含义**：
- `max_loras`（第0维）：GPU slot 数量，每个 slot 存一个 adapter
- `1`（第1维）：预留维度（实际是 `layer_idx`，当前固定为 1）
- `rank` 或 `output_dim`（第2维）：LoRA 的秩或输出维度
- `input_dim` 或 `rank`（第3维）：输入维度或秩

### 6.2 激活/销毁 Adapter

```python
def set_lora(self, index, lora_a, lora_b):
    """将 CPU 上的权重 copy 到 GPU slot[index]"""
    self.reset_lora(index)  # 清零旧权重
    self.lora_a_stacked[0][index, 0, :r, :d].copy_(lora_a, non_blocking=True)
    self.lora_b_stacked[0][index, 0, :d, :r].copy_(lora_b, non_blocking=True)

def reset_lora(self, index):
    """释放 GPU slot"""
    self.lora_a_stacked[0][index] = 0
    self.lora_b_stacked[0][index] = 0
```

`non_blocking=True` 意味着 copy 操作不阻塞 CPU 线程，权重传输可以和后续计算 overlap。

### 6.3 LRU 缓存管理

- `_registered_adapters`：LRU cache，容量 = `max_cpu_loras`（CPU 上的完整 LoRA 权重）
- `_active_adapters`：LRU cache，容量 = `max_loras`（GPU slot 占用状态）

当 GPU slots 满时，最久未使用的 adapter 被驱逐：
1. `remove_adapter(id)` → LRU 淘汰
2. 对应的 GPU slot 被 `reset_lora()` 清零
3. `lora_index_to_id[slot] = None` 标记为空闲

---

## 七、Forward 计算：Y = W@X + B@A@X

### 7.1 两步分解

[`vllm/lora/punica_wrapper/punica_gpu.py#L203-L264`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/punica_wrapper/punica_gpu.py#L203-L264)

```python
def add_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale, output_slices):
    """两步 LoRA 计算"""
    r = lora_b_stacked[0].size(-1)  # rank
    # Step 1: buffer = X @ A^T  → buffer shape: (n_slices, n_tokens, rank)
    buffer = torch.empty((len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device)
    self.add_shrink(buffer, x, lora_a_stacked, scale)

    # Step 2: output += buffer @ B^T  → output shape: (n_tokens, output_dim)
    self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True)
```

**为什么 buffer 用 float32**：
Triton 内核中 float16 的 partial sum 累积精度不够，可能导致 LoRA 输出出现数值误差。使用 float32 作为中间 buffer 保证精度，最终输出仍保持原始 dtype。

### 7.2 SHRINK 阶段：`X @ A^T`

[`vllm/lora/punica_wrapper/punica_gpu.py#L90-L121`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/punica_wrapper/punica_gpu.py#L90-L121)

```python
def add_shrink(self, y, x, lora_a_stacked, scale):
    """
    Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
    """
    x = x.view(-1, x.shape[-1])
    lora_shrink(x, lora_a_stacked, y, *self.token_mapping_meta.meta_args(...), scale)
```

Triton kernel `lora_shrink` 的核心特征：
- **3D grid**: `(SPLIT_K × ceil(M/BLOCK_M) × ceil(N/BLOCK_N), n_slices, n_active_loras)`
- 第三维按 `lora_id` 分组：同一个 thread block 只处理属于一个 LoRA adapter 的 token
- 通过 `token_indices_sorted_by_lora_ids` 和 `lora_token_start_loc` 定位 token 范围

### 7.3 EXPAND 阶段：`buffer @ B^T`

[`vllm/lora/punica_wrapper/punica_gpu.py#L123-L169`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/punica_wrapper/punica_gpu.py#L123-L169)

```python
def add_expand(self, y, x, lora_b_stacked, output_slices, offset_start=0, add_inputs=True):
    """
    Semantics:
        for i in range(len(lora_b_stacked)):
            slice = output_slices[i]
            y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
            offset += slice
    """
    lora_expand(x, lora_b_stacked, y, *self.token_mapping_meta.meta_args(...),
                offset_start=offset_start, add_inputs=add_inputs)
```

`add_inputs=True`：将 `y` 的原有值读入并累加 LoRA 输出（不是覆盖）。

### 7.4 BaseLayerWithLoRA.apply() — 完整前向

[`vllm/lora/layers/base_linear.py#L192-L236`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/layers/base_linear.py#L192-L236)

```python
def apply(self, x, bias=None):
    if self._enable_aux_cuda_stream:
        return torch.ops.vllm.lora_linear_async(self.layer_name, output_size, x, bias)
    else:
        return self._apply_sync(x, bias)

def _apply_sync(self, x, bias=None):
    # 1. 基座层 forward
    output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

    # 2. LoRA forward → output += B@A@x
    return self._apply_lora_to_output(x, output)

def _apply_lora_to_output(self, x, output):
    lora_output = self.punica_wrapper.add_lora_linear(
        output, x, self.lora_a_stacked, self.lora_b_stacked,
        1.0, self.output_slices
    )
    return output  # add_lora_linear 是 in-place 修改 output
```

---

## 八、SGMV Triton 内核设计

### 8.1 LoRAKernelMeta — 元数据准备

[`vllm/lora/ops/triton_ops/lora_kernel_metadata.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/ops/triton_ops/lora_kernel_metadata.py)

```python
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor       # (n_tokens,) → lora_id
    token_indices_sorted_by_lora_ids: torch.Tensor  # 按 lora_id 排序后的 token 索引
    active_lora_ids: torch.Tensor          # 活跃的 lora id 列表
    num_tokens_per_lora: torch.Tensor      # 每个 lora 的 token 数量
    lora_token_start_loc: torch.Tensor     # 每个 lora 的起始位置 (cumsum)
    num_active_loras_cpu: torch.Tensor     # 活跃 LoRA 数 (CPU tensor, compile 友好)
    no_lora_flag_cpu: torch.Tensor         # 是否无 LoRA active

    def prepare_tensors(self, token_lora_indices):
        # 检查是否所有 token 的 lora_id == -1
        self.no_lora_flag_cpu = (token_lora_indices == -1).all().cpu()

        # torch.sort → 按 lora_id 排序
        self.token_lora_mapping, self.token_indices_sorted_by_lora_ids = (
            token_lora_indices.sort()
        )
        # torch.unique_consecutive → 找到活跃 lora_ids
        self.active_lora_ids, self.num_tokens_per_lora = (
            self.token_lora_mapping.unique_consecutive(return_counts=True)
        )
        # cumsum → 起始位置
        self.lora_token_start_loc = self.num_tokens_per_lora.cumsum(0)
```

核心思想：**将按 lora_id 乱序的 token 排序分组**，这样 Triton kernel 可以逐组处理，每组内的 token 共享同一组 LoRA 权重。

### 8.2 SGMV 网格布局

```
Grid: (num_slices, n_active_loras, n_m_blocks × n_n_blocks × SPLIT_K)

dim 0 (num_slices):      处理打包层的多个 slice (如 QKV 的 Q/K/V)
dim 1 (n_active_loras):  按 lora_id 分组
dim 2 (space):           矩阵乘块的分布
```

**关键优势**：
- 同一个 warp/block 只访问一组 LoRA 权重 → 权重被缓存在 shared memory 或寄存器中
- 不同 lora 的 token 完全并行处理
- `no_lora_flag_cpu=True` 时整个 LoRA 路径被跳过

---

## 九、双 CUDA Stream 优化

### 9.1 问题

单 stream 下：基座层 `W@X` 和 LoRA `B@A@X` 是串行的。两者之间没有数据依赖（LoRA 的输入是 X 而非 W@X 的输出），可以并行。

### 9.2 实现

[`vllm/lora/layers/base_linear.py#L238-L302`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/lora/layers/base_linear.py#L238-L302)

```python
def _apply_async_impl(self, x, bias=None):
    """基座和 LoRA 在不同 CUDA Stream 上并行执行"""

    def base_fn():
        # default stream 上跑基座层
        return self.base_layer.quant_method.apply(self.base_layer, x, bias)

    def lora_fn():
        # aux stream 上跑 LoRA
        lora_output = torch.zeros((num_tokens, output_size), device=self.device, dtype=x.dtype)
        self.punica_wrapper.add_lora_linear(
            lora_output, x_2d, self.lora_a_stacked, self.lora_b_stacked,
            1.0, self.output_slices, add_inputs=False  # 不累加到 input
        )
        return lora_output

    # 并行执行：base_fn 在 default stream, lora_fn 在 aux stream
    output, lora_result = maybe_execute_in_parallel(
        base_fn, lora_fn, self._events[0], self._events[1], self._lora_stream
    )

    # 等待两个 stream 都完成，合并结果
    output.add_(lora_result)
    return output
```

**关键细节**：
- `add_inputs=False`：LoRA 输出不累加到 base output，而是写入独立的 zero tensor
- CUDA Event 同步：`base_fn` 完成后设置 event[0]，`lora_fn` 完成后设置 event[1]，`output.add_()` 等待两个 event
- 通过 `VLLM_LORA_ENABLE_DUAL_STREAM` 环境变量控制

---

## 十、nano-vllm 对比

nano-vllm **完全没有 LoRA 支持**。这是教学项目和生产的天然分界线：

| 维度 | nano-vllm | HLIEvLLM |
|------|-----------|----------|
| LoRA 推理 | 不支持 | 完整支持 |
| 多 adapter 并发 | 无 | 同一 batch 多 adapter |
| 权重存储 | 无 | GPU 堆叠缓冲区 + CPU LRU cache |
| LoRA kernel | 无 | Triton SGMV (shrink + expand) |
| 双 stream | 无 | 可选，基座∥LoRA 并行 |
| MoE + LoRA | 无 | 融合 Triton kernel |
| 代码量 | 0 行 | ~5000+ 行 |

---

## 十一、总结

### 核心设计模式

| 设计 | 应用 | 意义 |
|------|------|------|
| **低秩分解** | `B @ A` 代替 `ΔW` | 参数量减少 100x+，一个 adapter 仅 ~60MB |
| **预分配 GPU 堆叠** | `lora_a_stacked[max_loras, ...]` | 避免动态分配 + 支持 slot 热交换 |
| **SGMV 内核** | Triton 3D grid 按 lora_id 分组 | O(N_adapter × tokens) → O(tokens) |
| **LRU 缓存** | CPU/GPU 两级缓存 | 自动管理有限显存，驱逐最不常用 adapter |
| **双 stream** | CUDA Stream 并行 | base 和 lora 重叠，减少延迟 |
| **Scaling 融合** | `optimize()` → `B *= α/r` | 运行时免去 scale 乘法 |

### 与你工作的关联

1. **客户场景**：LoRA 是多租户部署的核心——同一个基座模型，每个客户加载自己的 adapter，无需额外 GPU
2. **Houmo NPU 的 LoRA 支持**：如果 Houmo 未来要支持 LoRA 推理，核心挑战是实现 SGMV 等价算子（因为 Triton 不一定能直接在 NPU 上跑）
3. **量化 + LoRA**：LoRA 权重通常保持 fp16/bf16，而基座权重可能被量化为 INT4/INT8，两者组合是常见需求

---

## 思考题

1. `lora_a_stacked` 的 shape 是 `(max_loras, 1, rank, input_dim)`。第1维（`layer_idx`）当前固定为 1，为什么保留这个维度？

2. SGMV kernel 和普通 `bmm` 的区别在哪里？为什么不能直接对所有 adapter 做 `bmm`？

3. 双 CUDA Stream 优化中，`lora_fn` 使用 `add_inputs=False` 写入独立的 zero tensor。为什么不直接 in-place 写 output？

4. `max_loras=4` 时，如果第 5 个客户请求到达，vLLM 如何处理？LRU 驱逐策略会逐出哪个 adapter？

5. MoE + LoRA 的 `pack_moe` 将 `(num_experts, rank, input)` 堆叠为 3D tensor。这和普通 dense 模型的 `lora_a_stacked` 有什么根本区别？
