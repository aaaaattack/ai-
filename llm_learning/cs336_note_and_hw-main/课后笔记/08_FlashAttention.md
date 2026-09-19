# 第8课：Flash Attention Triton Kernel

> 学习路线：阶段4 Kernel 与性能优化（CS336 Chapter 2 hw1）
> 对应文件：
> - CS336：`chapter2/hw1/triton_causal_forawrdflash_attention.py`（前向 kernel）
> - CS336：`chapter2/hw1/triton_backward_success.py` + `triton_backward_failure.py`（反向 kernel）
> - CS336：`chapter2/hw1/flashattention_autograd_function_pytorch.py`（PyTorch 封装）
> - nano-vllm：`nano-vllm/nanovllm/layers/attention.py`（工程调用 `flash_attn_varlen_func`）
> - nano-vllm：`nano-vllm/nanovllm/layers/attention.py`（`store_kvcache_kernel` — 另一个 Triton kernel）

---

## 一、问题：标准 Attention 为什么不行？

### 1.1 标准 Attention 的显存瓶颈

```python
# 标准 Attention（第5课学的版本）
S = Q @ K^T                   # (batch, n_heads, seq, seq) — O(n²) 中间矩阵
P = softmax(S, dim=-1)        # 从 HBM 读 S，写回 P — 又 O(n²)
O = P @ V                     # 从 HBM 读 P — 又 O(n²)

# HBM 总读写量: 3×O(n²)×dtype_size
# seq=4096, fp16, 8 heads: 3 × 4096² × 2B × 8 = 768MB（单层！）
```

**问题根源**：`S = Q@K^T` 产生了一个巨大的中间矩阵，必须写回 HBM（高带宽显存），然后下一步再从 HBM 读出来做 softmax。

### 1.2 Flash Attention 的核心思路

```
把"写回 HBM → 再从 HBM 读"这两步省掉。
中间结果留在 SRAM（片上缓存）里，只把最终结果写回 HBM。
```

SRAM 比 HBM 快 10-20 倍，但容量极小（~200KB per SM）。所以必须**分块**计算。

```
Flash Attention 的两类 tiling:

1. Q 分块 (outer loop):   把 Q 的行切分成多个 tile
                          每个 Q_tile 独立计算自己对应位置的输出

2. K/V 遍历 (inner loop): 对每个 Q_tile，遍历所有 K/V tiles
                          每个 K/V tile 出一个局部 softmax，online 合并
```

---

## 二、Tiling + Online Softmax 算法图解

```
Q = | Q₀ |    (tile_size, d)         K = | K₀ K₁ K₂ |   各 (tile_size, d)
    | Q₁ |                              | K₀ K₁ K₂ |
    | Q₂ |
    | ... |

对 Q₀ 的处理:
  Q₀ × K₀ᵀ → 局部 S_00 → online softmax(O₀, M₀, L₀) → 初始值
  Q₀ × K₁ᵀ → 局部 S_01 → online 更新 (O₀, M₀, L₀)    → 发现更大 max，rescale
  Q₀ × K₂ᵀ → 局部 S_02 → online 更新 → 最终 O₀ = O_acc / L_acc

对 Q₁ 的处理: 同上，重新遍历 K₀, K₁, K₂
```

**Online Softmax 的三个状态变量**：

| 变量 | 含义 | 初始值 |
|---|---|---|
| `M_acc` | 当前见过的最大 Q·K 分数 | `-inf` |
| `L_acc` | 累积的 softmax 分母（rescale 后的 sum(exp)）| `0` |
| `O_acc` | 累积的加权 V（rescale 后的 sum(P @ V)）| `0` |

**当新 K/V tile 到达时**：

```
1. 计算局部 S_ij = Q_i @ K_j^T / sqrt(d_k)

2. 找局部最大值 M_ij = max(S_ij)

3. 更新全局最大值 M_new = max(M_old, M_ij)

4. Rescale（如果 M_new > M_old）:
   - 旧结果乘 exp(M_old - M_new)  ← 旧 tile 的相对重要性降低了
   - 新结果乘 exp(M_ij  - M_new)  ← 新 tile 按新 max 缩放
   - 两者相加

5. 重复直到所有 K/V tile 处理完

6. 最终 O = O_acc / L_acc
```

### 为什么 Rescale 是合法的？

```
数学上:
  softmax(x₁,x₂) = [exp(x₁), exp(x₂)] / [exp(x₁)+exp(x₂)]

分块后 (先算 x₁, 后来 x₂ 更大):
  第一遍: M₁=x₁,  O₁=exp(x₁-x₁)*V₁=V₁, L₁=exp(x₁-x₁)=1
  第二遍: M₂=x₂(x₂>x₁), M_new=x₂
          O₂ = exp(x₁-x₂)*O₁ + exp(x₂-x₂)*V₂
             = exp(x₁-x₂)*V₁ + V₂     ← V₁ 被"缩小"了
          L₂ = exp(x₁-x₂) + 1

  O₂/L₂ = (exp(x₁-x₂)*V₁ + V₂) / (exp(x₁-x₂) + 1)
        = (exp(x₁)*V₁ + exp(x₂)*V₂) / (exp(x₁) + exp(x₂))  ← 和直接 softmax 一致！
```

---

## 三、Triton Kernel 代码逐段解析

源码：`chapter2/hw1/triton_causal_forawrdflash_attention.py`

### 3.1 kernel 入口 — 确定当前处理的 Q tile

```python
@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
                     stride_qb, stride_qq, stride_qd, ...,):
    # Triton 并行方式: 每个 program 处理 1 个 Q_tile × 1 个 batch
    i = tl.program_id(0)        # 第几个 Q tile（0, 1, 2, ...）
    batch_index = tl.program_id(1)  # 第几个 batch

    # 所有 program 同时运行，各处理各自的 Q tile
```

**`tl.program_id`**：Triton 把 kernel 启动为 grid，每个 grid cell 运行一个 program instance。`program_id(0)` 告诉你"你在处理第几个Q分块"。

### 3.2 Block Pointer — 描述要加载的数据块

```python
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,  # 跳过前面的 batch
        shape=(N_QUERIES, D),                  # 完整张量的形状
        strides=(stride_qq, stride_qd),        # 各维度的步长
        offsets=(i * Q_TILE_SIZE, 0),          # 当前块的起始偏移
        block_shape=(Q_TILE_SIZE, D),          # 要加载的块大小
        order=(1, 0),                          # 列优先访问（针对 memory coalescing）
    )
```

**`tl.make_block_ptr` 是 Triton 最核心的抽象**：描述一个二维数组的子区域，而不需要手动算偏移量。

```
完整 Q:  shape=(N_QUERIES, D)
offsets=(i*TILE, 0):          从第 i*TILE 行、第 0 列开始
block_shape=(TILE, D):        加载 TILE 行 × D 列

结果: tl.load(Q_block_ptr) → 从 HBM 加载 (TILE, D) 的块到 SRAM
```

**K/V 的 block_ptr 初始偏移都是 0**：

```python
    K_block_ptr = tl.make_block_ptr(..., offsets=(0, 0), ...)
    V_block_ptr = tl.make_block_ptr(..., offsets=(0, 0), ...)
    # K 和 V 的起始都是 0，它们会在内层循环中被 advance() 移动
```

### 3.3 加载 Q tile + 初始化 Online Softmax 状态

```python
    Q_i = tl.load(Q_block_ptr)                              # (Q_TILE_SIZE, D)
    O_i_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # 加权 V 累积
    L_i_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32) # softmax 分母累积
    M_i_acc = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype=tl.float32) # 最大分数
```

**为什么用 `tl.float32`？** Softmax 的 exp 对精度极其敏感，用 float16 会有数值问题。这和 CS336 的 `x.to(torch.float32)` 是同一个道理。

### 3.4 内层循环 — 遍历所有 K/V tiles

```python
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):  # K/V 分块数
        K_j = tl.load(K_block_ptr)                   # 当前 K tile: (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr)                   # 当前 V tile: (K_TILE_SIZE, D)
```

### 3.5 局部 Attention Scores

```python
        S_ij = tl.dot(Q_i, K_j.T) * scale
        # Q_i: (Q_TILE_SIZE, D)
        # K_j^T: (D, K_TILE_SIZE)
        # S_ij: (Q_TILE_SIZE, K_TILE_SIZE)
        # scale = 1 / sqrt(d_k)
```

**`tl.dot`**：Triton 的矩阵乘法指令，在 Tensor Core 上执行。

### 3.6 Causal Mask（分块版本）

```python
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            # Q tile 中每个 query token 的全局位置
            # i=2, Q_TILE_SIZE=32 → [64, 65, ..., 95]

            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            # K tile 中每个 key token 的全局位置
            # j=0, K_TILE_SIZE=32 → [0, 1, ..., 31]

            causal_mask = q_idx >= k_idx
            # (Q_TILE_SIZE, K_TILE_SIZE) — 每个 query 只能看到 <= 其位置的 key

            S_ij = tl.where(causal_mask, S_ij, -1e6)
            # 被 mask 的 position → score = -1e6 → exp(-1e6) ≈ 0
```

**为什么分块时需要全局索引？** 因为每个 tile 只知道自己的块内序号，不知道在整个序列中的绝对位置。`i * Q_TILE_SIZE + 块内偏移` 把块内序号映射到全局位置。

### 3.7 Online Softmax 更新

```python
        # Step A: 计算局部 softmax
        M_ij = tl.max(S_ij, axis=1, keep_dims=True)     # 当前块每行的 max
        P_ij = tl.exp(S_ij - M_ij)                       # 局部 softmax 分子

        # Step B: 更新全局 max
        M_i_new = tl.maximum(M_i_acc, M_ij)              # 全局 max

        # Step C: Rescale + 累加
        # L_i_new = exp(M_old - M_new) * L_old + exp(M_ij - M_new) * sum(P_ij)
        L_i_new = tl.exp(M_i_acc - M_i_new) * L_i_acc \
                + tl.exp(M_ij - M_i_new) * tl.sum(P_ij, axis=1, keep_dims=True)

        # O_i_new = exp(M_old - M_new) * O_old + exp(M_ij - M_new) * (P_ij @ V_j)
        P_ij_cast = P_ij.to(V_block_ptr.type.element_ty)  # 精度对齐
        O_i_new = tl.exp(M_i_acc - M_i_new) * O_i_acc \
                + tl.exp(M_ij - M_i_new) * tl.dot(P_ij_cast, V_j)

        # 更新状态
        M_i_acc = M_i_new
        O_i_acc = O_i_new
        L_i_acc = L_i_new
```

**Rescale 的数值示例**：

```
第一块: S_00 = [5, 2, 1], max=5
  P_00 = [exp(0), exp(-3), exp(-4)] = [1.0, 0.05, 0.018]
  O_acc = P_00 @ V_0  (某个值)
  L_acc = sum(P_00) = 1.068
  M_acc = 5

第二块: S_01 = [8, 4, 3], max=8 > 5 ← 新 max 更大！
  M_new = 8
  Rescale 因子 = exp(5-8) = exp(-3) = 0.05
  O_new = 0.05 * O_acc + exp(8-8) * (P_01 @ V_1)
        = 0.05 * O_acc + 1.0 * (P_01 @ V_1)
        ↑ 旧结果被缩小到 5%
  L_new = 0.05 * L_acc + 1.0 * sum(P_01)
```

### 3.8 前进 K/V block pointer

```python
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        # 向前移动 K_TILE_SIZE 行，准备加载下一个 K/V tile
```

**`advance`** 在原地修改 block pointer 的偏移量，不重新分配。下一轮循环的 `tl.load(K_block_ptr)` 会自动加载下一块。

### 3.9 最终归一化

```python
    # 循环结束后:
    O_i = O_i_acc / L_i_acc          # 加权 V 和 / softmax 分母 = Attention输出
    L_i = M_i_acc + tl.log(L_i_acc)   # log-sum-exp（反向传播需要）
    tl.store(O_block_ptr, O_i)        # 写回 HBM
    tl.store(L_block_ptr, L_i)
```

---

## 四、Kernel 启动配置

```python
# kernel 调用（在 FlashAttentionAutogradFunctionTriton 中）
grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)
# grid[0]: Q tile 数量 — 每个 Q tile 一个 program
# grid[1]: batch 数量 — 每个 batch 独立

flash_fwd_kernel[grid](
    q, k, v, o, l,
    ..., N_QUERIES, N_KEYS, scale, D,
    Q_TILE_SIZE=32, K_TILE_SIZE=32, is_causal=True
)
```

---

## 五、HBM 读写分析：为什么从 O(n²) 降到 O(n)

```
标准 Attention HBM 读写:
  S = Q@K^T: 读 Q(n×d) + 读 K(n×d) + 写 S(n×n) = O(n²)
  P = softmax(S): 读 S(n×n) + 写 P(n×n) = O(n²)
  O = P@V: 读 P(n×n) + 读 V(n×d) + 写 O(n×d) = O(n²)
  总计: O(n²)

Flash Attention HBM 读写:
  对每个 Q tile (n/Q_TILE 个):
    读 Q_i: (TILE, d)
    遍历 K/V tiles (n/K_TILE 个):
      读 K_j: (TILE, d)
      读 V_j: (TILE, d)
      无中间写回！(S_ij, P_ij 留在 SRAM)
    写 O_i: (TILE, d)
  总计: O(n×d) = O(n)  ← 省了一个数量级
```

**核心**：`S_ij` 和 `P_ij` 从不写回 HBM，始终留在 SRAM 中。只有最终 `O_i` 才写回。

---

## 六、反向 kernel 简介

反向传播也需要 tiling（`triton_backward_success.py`），思路类似但更复杂：

- 需要前向留下的 `L` (log-sum-exp) 和 `O` (最终输出)
- 对每个 Q tile，重新计算该块的 `S_ij` 和 `P_ij`（因为前向没保存它们）
- 计算 `dQ`, `dK`, `dV` 的梯度

这是一个"recompute"策略 — 用计算换显存。

---

## 七、与 nano-vllm 的对接

nano-vllm 不自己实现 Flash Attention，而是调用 `flash_attn` 库：

```python
# nano-vllm layers/attention.py
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# Prefill: 变长序列批量 Attention
o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=..., max_seqlen_q=...)

# Decode: 带 KV Cache 的 Attention
o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=...)
```

但 nano-vllm 也有自己的 Triton kernel — `store_kvcache_kernel`，用于把当前 token 的 K/V 写入 paged KV Cache。

---

## 八、与编译器/硬件的关联（本课最重要）

| 方面 | 具体关联 |
|---|---|
| **Tiling size** | `Q_TILE_SIZE` / `K_TILE_SIZE` 受 SRAM 大小限制。2×(TILE×D) + (TILE×TILE) 必须 < SRAM。你们 HAL 需要暴露 SRAM 大小给编译器 |
| **Block pointer** | `tl.make_block_ptr` → `tl.load` → `tl.advance` 是 Triton 的抽象。你们的编译器如果从 Triton IR lower，需要把 block_ptr → DMA 描述符 → 实际地址计算 |
| **Online softmax** | `exp(M_old - M_new)` 涉及 fp32 的 exp 和乘加。在 NPU 上是否有硬件 exp 指令？如果用查表近似，精度是否够？ |
| **`tl.dot`** | Triton 的矩阵乘降级到 Tensor Core 的 MMA 指令。你们的 NPU 提供什么矩阵乘指令？ |
| **HBM → SRAM** | `tl.load` → DMA 读取。data layout（行优先 vs 列优先）直接影响 DMA 效率。`order=(1,0)` 的选择影响 memory coalescing |
| **Causal mask** | `q_idx >= k_idx` 是逐元素比较，在 SIMD/NPU 上能否向量化？ |

---

## 九、思考题

1. **为什么 Flash Attention 能把 HBM 读写从 O(n²) 降到 O(n)？具体省了什么？**
   - 提示：S 和 P 从来不写回 HBM

2. **Online softmax 的 rescale 因子为什么是 `exp(M_old - M_new)`？**
   - 提示：数学上 `exp(x_i - M_new) = exp(x_i - M_old) × exp(M_old - M_new)`

3. **`Q_TILE_SIZE` 和 `K_TILE_SIZE` 的选取受什么限制？**
   - 提示：每个 SM 的 SRAM 容量 ~200KB，三个 tile 的显存和不能超过

4. **反向 kernel 为什么不直接保存前向的 S_ij 和 P_ij？**
   - 提示：如果保存了，就退回了 O(n²) 显存 → Flash Attention 的核心优势没了

---

## 十、CS336 课后作业

### 作业核心

理解并完成 `triton_causal_forawrdflash_attention.py` 的 kernel 实现，并封装为 PyTorch autograd Function。

### 关键理解点

1. `tl.make_block_ptr` 的 shape/strides/offsets/block_shape/order 参数含义
2. Online softmax 的三个状态变量（M_acc, L_acc, O_acc）的更新逻辑
3. Causal mask 在分块场景下的全局索引计算
4. `advance()` 的作用

### 验证方法

```python
# 和 PyTorch 标准 Attention 对比数值
q = torch.randn(2, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 512, 64, device='cuda', dtype=torch.float16)

# 标准 Attention
scale = 1.0 / math.sqrt(64)
scores = q @ k.transpose(-2, -1) * scale
mask = torch.triu(torch.ones(512, 512, dtype=torch.bool), diagonal=1).cuda()
scores.masked_fill_(mask, float('-inf'))
attn = torch.softmax(scores, dim=-1)
ref_out = attn @ v

# Flash Attention
flash_out = flash_attention_fn(q, k, v, is_causal=True)

assert torch.allclose(ref_out, flash_out, atol=1e-2)
```
