# 生产功能第5课：KV Cache 量化

> 学习路线：P0 必学 — 量化推理全链路 > 1.5 KV Cache 量化
> 前置：量化基础 (生产_01)，PagedAttention (第11课)，Roofline 模型 (推理性能优化路线)
> 关联工作：Helion FP8 kernel、decode 阶段性能优化

---

## 一、为什么 KV Cache 量化是 decode 的最强优化

### 1.1 回顾：decode 是访存瓶颈

```
decode 阶段: 每次只算 1 个 token
  计算量:  矩阵乘 (1, d_model) × (d_model, d_ff) ≈ 小
  访存量: 需要读取所有历史 token 的完整 KV Cache
  算术强度 ≈ 3 FLOPs/Byte（严重访存瓶颈）

KV Cache 是 decode 阶段的主要访存来源:
  ↓ KV Cache 大小 = ↓ 显存带宽压力 → 近似线性加速 decode
```

### 1.2 量化收益量化

```
Qwen3-0.6B (28 层, 4 KV heads, head_dim=128):

seq_len=4096 的 KV Cache:
  FP16: 2 × 28 × 4096 × 4 × 128 × 2 bytes = 224 MB
  FP8:  2 × 28 × 4096 × 4 × 128 × 1 byte  = 112 MB
  INT8: 112 MB + scale 开销

seq_len=32768 的 KV Cache:
  FP16: 2 × 28 × 32768 × 4 × 128 × 2 = 1.8 GB
  FP8:  0.9 GB

如果能节省 50%:
  → 相同显存可支持 2× 并发请求
  → 或 2× context length
  → 或 decode 吞吐提升 ~40-50%（访存减半，计算不变）
```

### 1.3 和 Weight 量化 + Activation 量化的组合

```
W4A16 (GPTQ/AWQ)  +  KV Cache FP8
↓ 权重显存      +  ↓ KV 显存
= 全面显存优化

W8A8 (SmoothQuant) +  KV Cache FP8
↓ 矩阵乘加速     +  ↓ KV 显存
= 吞吐极致优化
```

---

## 二、KV Cache 的量化粒度

### 2.1 四种粒度对比

| 粒度 | 每 token 几个 scale | 精度 | 适用场景 |
|---|---|---|---|
| per-tensor | 1 (整层 KV) | 低 | 基本不用 |
| per-channel | num_kv_heads 个 scale | 中 | 简单场景 |
| per-token | 1 (每个 token) | 高 | **业界常用** |
| per-group | group_size 个 scale/group | 极高 | H100 FP8 无需 group |

### 2.2 Per-token 量化（推荐方案）

```python
# K: (seq_len, num_kv_heads, head_dim)
# 每个 token 的 KV 向量独立量化:
#   scale_t = max(|K[t, :, :]|) / 127

for t in range(seq_len):
    k_token = K[t]                                    # (num_kv_heads, head_dim)
    scale = k_token.abs().max() / 127                 # 标量
    K_int8[t] = round(k_token / scale).clamp(-128, 127)
    K_scales[t] = scale                               # 存 per-token scale

# scale 存储开销:
# FP16 scale: 2 bytes per token
# seq_len=4096: 4096 × 2 = 8KB per layer ← 基本可忽略
# 28 layers: 224KB total ← 对比 224MB KV 完全可忽略
```

**为什么 per-token 比 per-channel 好？**

```python
# 同一 token 的 KV：不同 head 的数值范围相近（都被 RMSNorm 约束）
per-token: 1 个 scale → 所有 head 共享 → scale 开销小

# 同一个 head 的 KV：不同 token 的数值范围差异大
per-channel: 1 个 scale/head → 但 token 间差异被忽略 → 精度低
```

### 2.3 Per-channel 的局限

```python
# 示例: head_0 的 KV，seq_len=4
K[0:4, 0, :] = [
    [0.5, 0.3, 0.8, 0.2],      # token_0
    [-12.3, 5.7, 0.1, 0.4],    # token_1 ← outlier
    [0.2, 0.1, -0.3, 0.5],     # token_2
    [0.6, 0.4, 0.9, 0.1],      # token_3
]

per-channel (head_0):
  scale = max_abs(K[:, 0, :]) / 127 = 12.3 / 127 = 0.097
  token_0: [0.5, 0.3, 0.8, 0.2] / 0.097 = [5, 3, 8, 2]     ← 精度足够
  token_1: [-12.3, 5.7, ...] / 0.097 = [-127, 59, ...]       ← 精度足够
  token_2: [0.2, 0.1, -0.3, 0.5] / 0.097 = [2, 1, -3, 5]   ← 精度足够?
  # token_2 的 0.2→2, 0.1→1，量化间隔 0.097 ≈ 0.1，还行但不算好

per-token (token_2):
  scale = max_abs(K[2, 0, :]) / 127 = 0.5 / 127 = 0.0039
  token_2: [0.2, 0.1, -0.3, 0.5] / 0.0039 = [51, 26, -76, 128]
  # 量化间隔 0.0039 → 精度远好于 per-channel
```

---

## 三、FP8 vs INT8：哪个更适合 KV Cache？

### 3.1 数值格式对比

| | FP8 (E4M3) | INT8 |
|---|---|---|
| 指数位 | 4 位 | 0 位 |
| 尾数位 | 3 位 | 8 位（隐含在 scale 中） |
| 表示范围 | ~[2^-6, 448] | [-127×scale, 127×scale] |
| 动态范围 | 2^8 = 256 倍 | 由 scale 决定 |
| 精度 | 12.5% 相对误差 | scale 决定绝对误差 |
| 需要 scale？ | 否（范围覆盖 KV 值域） | 是（至少 per-tensor） |
| 硬件支持 | H100/Helion NPU | 所有 GPU |

### 3.2 FP8 对 KV Cache 的天然优势

```python
# FP8 E4M3 格式:
# 1 位符号 + 4 位指数 + 3 位尾数
# 能表示的值: sign × 2^(exponent-7) × (1 + mantissa/8)

# 举例:
# 0_1000_010 = 1 × 2^(8-7) × (1 + 2/8) = 2 × 1.25 = 2.5
# 0_1100_000 = 1 × 2^(12-7) × (1 + 0) = 32 × 1 = 32
# 1_1110_111 = -1 × 2^(14-7) × (1 + 7/8) = -128 × 1.875 = -240

# KV Cache 值在 Prefill 后范围约为 [-50, 50] (被 RMSNorm 约束)
# → FP8 的 E4M3 范围 [2^-6=0.016, 448] 完全覆盖
# → 不需要 scale!（不需要 calibration!）
```

### 3.3 INT8 需要 scale 带来的麻烦

```python
# INT8 KV Cache: 需要存每个 token 的 scale
# Attention 反量化:
def decode_attention_with_int8_kv(q, K_int8, V_int8, K_scales, V_scales):
    for blk in block_table:
        K_fp = K_int8[blk] * K_scales[blk]    # ← 每个 block 都要反量化
        V_fp = V_int8[blk] * V_scales[blk]    # ← 反量化到 FP16 再计算
        # Attention: q @ K_fp^T / sqrt(d_k)
        # softmax @ V_fp

# 问题: 反量化本身消耗了 FP8 省下的部分带宽
# 如果 scale 是 per-token: 反量化 = 1 次 multiply per token → 可接受
# 如果 scale 是 per-channel: 反量化 = 4 次 multiply per token (4 KV heads) → 仍可接受
```

---

## 四、PagedAttention + KV Cache 量化的组合

### 4.1 两者的协同

```
PagedAttention (第11课):
  KV Cache 分页 → 省掉 50% 显存浪费（短 prompt 场景）
  软件层: Block Pool + block_table

KV Cache FP8 (本课):
  KV Cache 压缩 → 省掉 50% 显存（所有场景）
  硬件层: FP8 格式直接存储

组合:
  同样 24GB 显存:
    原始: 支持 ~100 seq × 4K context
    +PagedAttention: 支持 ~200 seq
    +PagedAttention + KV FP8: 支持 ~400 seq
```

### 4.2 在 block_manager 中的存储

```python
# nano-vllm BlockManager (第11课):
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0

# 扩展: Block 支持量化存储
class Block:
    def __init__(self, block_id, block_size, num_kv_heads, head_dim, kv_dtype):
        # 如果 kv_dtype == torch.float8, kv_data 存 FP8
        # 如果 kv_dtype == torch.float16, kv_data 存 FP16
        self.kv_data = torch.empty(2, block_size, num_kv_heads, head_dim, dtype=kv_dtype)

    def read_kv(self):
        # 反量化（如果 KV 是 INT8）
        if self.kv_dtype == torch.int8:
            return self.kv_data.float() * self.scales
        return self.kv_data
```

### 4.3 在 Flash Attention 中的使用

```python
# nano-vllm Attention (第5课/第8课):
# decode 路径: flash_attn_with_kvcache(q, k_cache, v_cache, ...)

# 如果 k_cache/v_cache 是 FP8:
# flash_attn 需要支持 FP8 KV Cache
# → Flash Attention v3 已支持 FP8 K/V
# → q: FP16, k: FP8, v: FP8 → 混合精度 Attention
# → For Houmo Helion: FP8 KV 直接传给 NPU，不反量化
```

---

## 五、Houmo Helion 的 FP8 路线

### 5.1 为什么 Helion 全线 FP8

从 Helion kernel 命名推断：

```python
# HLIEvLLM-1.4.0rc0/vllm/kernels/helion/ops/silu_mul_fp8.py
# HLIEvLLM-1.4.0rc0/scripts/autotune_helion_kernels.py:15
#   --kernels silu_mul_fp8 rms_norm_fp8

# 全部带 _fp8 后缀 → Helion 后端原生 FP8
# 不是 INT8 → 不需要 scale/zero_point 逻辑
```

**FP8 vs INT8 对 NPU 设计的影响**：

```
INT8 NPU: 需要 multiply-accumulate → INT32 累加 → 反量化
FP8 NPU:   需要 FP8 multiply-accumulate → FP32 累加
          → FP8 和 FP16 之间直接转换（只需要调整指数和尾数，不需要 scale）

对 Kv Cache 的好处:
  FP8 KV Cache → Attention 中 Q(FP16)·K(FP8) 直接乘 → 不需要反量化
  这比 INT8 KV Cache 少一步反量化 → 更快
```

### 5.2 Houmo 的 KV Cache 量化推断

```python
# 从 HLIEvLLM 的存储模型推断:
# 1. KV Cache 由 vLLM KVCacheManager 管理 (第11课附录)
# 2. 物理存储: tcim_runtime 管理的 NPU 显存
# 3. 量化决策: Helion FP8 kernel 在 .hmm 编译时决定
# 4. 反量化: NPU 硬件支持 FP8@FP16 混合精度 Attention

# 流程:
# Prefill:   计算 K, V (FP16) → Helion kernel 压缩为 FP8 → 写回 Block Pool
# Decode:    q (FP16) + K_cache (FP8) + V_cache (FP8) → Helion Attention kernel
#            → 硬件层面 FP8·FP16 混合精度点积 → 输出 FP16
```

---

## 六、量化全链路总结

```
量化推理的完整脉络:

W4A16 (GPTQ / AWQ)
  ↓ 权重 INT4, 激活 FP16
  ↓ 省权重大小，模型文件 16GB→4GB
  
W8A8 (SmoothQuant)
  ↓ 权重 INT8, 激活 INT8
  ↓ 矩阵乘加速 2×, 需要数学变换平滑 outlier

KV Cache FP8
  ↓ KV 存储 FP8
  ↓ decode 访存瓶颈直接减半
  
三者可叠加:
  W4A16 + KV FP8:  全面显存优化 (推荐)
  W8A8 + KV FP8:  吞吐极致优化 (H100/Helion)
```

---

## 七、思考题

1. **为什么 KV Cache 量化对 decode 加速最有效，但对 prefill 帮助不大？**
   - 提示：prefill 是计算密集（大 GEMM），KV Cache 在 prefill 中只写不读

2. **FP8 的 4 位指数提供 [2^-6, 448] 的动态范围。如果 KV 值的实际范围是 [-200, 200]，FP8 能否精确表示 0.1？**
   - 提示：FP8 在 2^-6=0.016 附近的精度 ≈ 0.016/8=0.002 → 0.1 可以表示

3. **per-token scale 在 Attention 计算中怎么用？**
   - 提示：Q·K^T 时，K 反量化后做点积。scale 可以作为 K 的一部分被预先吸收（类似 SmoothQuant 的变换）
