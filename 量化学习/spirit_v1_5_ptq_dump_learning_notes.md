# Spirit-v1.5_gpt PTQ Dump 学习与复盘笔记

远端工程：`/workspace/921_embodied_ai/embodied_ai/spirit-v1.5_gpt`

## 1. 系统总览

Spirit-v1.5 在 XH2 后端拆成 Host 与多个 HMM 子图：

```text
observation
  → Host 图像预处理、tokenize、归一化
  → visual.hmm
  → Host token embedding、多模态序列拼接
  → prefill.hmm
  → decode.hmm（KV cache）
  → Host projection
  → dit.hmm（flow matching）
  → Host 动作后处理
```

主要文件：

| 文件 | 作用 |
|---|---|
| `ptq_dump.py` | 复用 HMONNX，生成 golden，做多后端比较 |
| `ptq.py` | Qwen3-VL、LLM、DiT 导出和 PTQ |
| `build.py` | HMONNX 编译 HMM，执行子图测试 |
| `export_host_weights.py` | 导出 Host 侧小权重 |
| `spirit_model/model/modeling_spirit_vla.py` | Spirit VLA 策略模型 |

## 2. 入口调用链

`ptq_dump.py:1346` 的 `main()`：

```text
parse_args()
  ├─ --dump-only → dump_and_stage_golden()
  └─ 普通模式
       → ptq.quantize_qwen3_vl()
       → ptq.quantize_dit()
       → dump_and_stage_golden()
```

如果没有显式指定 `--dump-golden` 或 `--four-way-compare`，脚本会自动打开两者。

`--dump-only` 适合学习和复盘：它跳过重新 PTQ，只读取已有 `output/xh2/hmquant` 产物。

## 3. 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `batch` | 1 | batch size |
| `context_length` | 512 | 最大上下文 |
| `max_pe_length` | 32768 | 位置编码最大长度 |
| `quant_type` | `w8a8h1_sefp` | 量化模式 |
| `image_max_height` | 224 | 图像高度 |
| `image_max_weight` | 320 | 图像宽度，参数名确实是 weight |
| `image_max_temporal` | 2 | 时间维度 |
| `patch_size` | 16 | 空间 patch |
| `temporal_patch_size` | 2 | 时间 patch |
| `dit_encoder_length` | 300 | DiT 固定条件序列长度 |
| `input-sequence-length` | 256 | prefill 分块长度 |

`validate_vision_grid()` 要求图像高宽是 `patch_size * 2 = 32` 的正整数倍，因为 Qwen3-VL 使用 2x2 spatial merge。

## 4. `build_converter_dummies()` 的职责

位置：`ptq_dump.py:888`。

它不是简单的随机输入生成器，而是把真实图片经过 Qwen3-VL processor 后，组织成可供 vision、prefill、decode、DiT 以及不同 runtime 使用的多套输入。

输入：

```python
build_converter_dummies(args, wrap_graph, device, dit_policy=None)
```

- `args`：图像、patch、序列长度等静态参数；
- `wrap_graph`：加载了权重的 Qwen3-VL + Spirit 外层模型；
- `device`：通常为 `cuda:0`；
- `dit_policy`：用于读取 DiT 配置，可选。

## 5. 函数内部的数据流

### 5.1 Processor 输入

函数先调用：

```python
inputs = build_processor_inputs(args, wrap_graph)
```

该函数构造一条“图片 + Describe this image.”消息，并使用 `Qwen3VLProcessor` 生成：

```text
input_ids
pixel_values
image_grid_thw
hm_pixel_values
```

这里的关键点是：输入来自真实 processor，能保持 Qwen3-VL 的 token、视觉 patch 和 grid 约定。

### 5.2 视觉前向

```python
image_embeds, deepstack_image_embeds = visual.forward_ori(
    inputs["pixel_values"],
    grid_thw=inputs["image_grid_thw"],
)
```

输出分为：

- `image_embeds`：主视觉 embedding；
- `deepstack_image_embeds`：供 LLM 多层视觉注入的 embedding。

随后视觉模块和结果被移回 CPU，控制显存占用。

### 5.3 多模态预处理

函数把以下内容交给 `Qwen3_VLDataPreprocess`：

```text
input_ids
image_embeds
deepstack_image_embeds
past_seq_length = 0
image_grid_thw
```

预处理后得到：

```text
inputs_embeds
time_position_ids
height_position_ids
width_position_ids
past_seq_length
current_seq_length
deepstack_image_embed_0/1/2
```

核心转换是：

```text
input_ids + image_embeds
  → inputs_embeds
  → 三维位置编码
  → 多层 deepstack 视觉 embedding
```

### 5.4 KV cache

`make_empty_caches()` 位于 `ptq_dump.py:835`。

每个 Transformer layer 都创建 key/value cache：

```text
[batch, num_key_value_heads, context_length, head_dim]
```

cache 被包装为 `CacheTensor`，不能简单当普通 tensor 展平、复制或传给 HMONNX。

### 5.5 Prefill 输入

prefill 输入包含：

```text
inputs_embeds[:input_sequence_length]
三组 position ids
past_seq_length
current_seq_length
三组 deepstack image embedding
past_key_caches
past_value_caches
```

它表示“首次处理一段序列”。由于只取前 `input_sequence_length` 个 token，所以支持分块执行。

### 5.6 Decode 输入

decode 只取一个 token：

```python
inputs_embeds[:, :1, :]
```

它使用单 token position ids、新的 K/V cache，并用零 tensor 占位 deepstack embedding。对比关系：

| 阶段 | 序列 | 视觉 embedding | cache |
|---|---:|---|---|
| prefill | 一段 | 真实 deepstack | 初始 cache |
| decode | 1 token | 零占位 | 增量 cache |

## 6. Native aligned 输入

`hm_pixel_to_native_patches()` 位于 `ptq_dump.py:770`。

它将 HMONNX 像素格式转换到 native 视觉模块格式：

```python
pixel = (pixel - 127.5) / 127.5
```

随后按时间 patch、空间 patch、2x2 merge 重新排列，返回展平 patch 和 `grid_thw`。

`native_aligned` 的意义是区分两类问题：

```text
native 与 wrap 不一致
  → 可能是模型差异
  → 也可能是输入归一化/patch 排布差异
```

因此代码额外保留 aligned 路径来定位输入契约问题。

## 7. 返回的数据结构

大致结构：

```python
{
    "vision": {"wrap", "native", "native_aligned", "hmonnx"},
    "prefill": {"wrap", "native", "hmonnx", "num_layers"},
    "decode": {"wrap", "native", "hmonnx", "num_layers"},
    "dit": {"wrap", "native", "hmonnx"},
}
```

`dit` 仅在传入 `dit_policy` 时加入。

## 8. DiT 输入契约

`build_dit_dummies()` 位于 `ptq_dump.py:736`，形状由 policy 配置决定：

```text
hidden_states:          [1, 1 + n_action_steps, hidden_size]
encoder_hidden_states:  [1, dit_encoder_length, encoder_hidden_size]
timestep:               [1]
encoder_attention_mask: [1, dit_encoder_length]
```

当前 README 记录的 checkpoint 为 `n_action_steps=50`，所以第一维序列长度为 51。

`timestep` 使用连续浮点数 `0.5`，不能转成 int，否则 flow-matching 时间会被截断。

## 9. 四路比较

| 路径 | 实现 |
|---|---|
| native | 原生 PyTorch vision / LLM / DiT |
| wrap | 工具链 wrap 模型 |
| FX | `convert_fx_model_to_quanted_model`，当前主要用于 LLM |
| HMONNX | `HMONNXGoldenInference` |

比较指标：

- cosine：方向相似度；
- MSE：平均平方误差；
- max_abs：最大绝对误差。

判断顺序应是：

```text
先看 shape
  → 再看 missing/skipped
  → 再看 cosine
  → 最后结合 MSE 和 max_abs 判断误差类型
```

## 10. 当前环境记录

已确认：

- 容器为 `xw-1.5.0-build`；
- `output/xh2/hmquant` 已有四类组件产物和 cosine 报告；
- 必须设置：

```bash
export HOUMO_TARGET=xh2
export PYTHONPATH="$PWD/spirit_model:/workspace/921_embodied_ai/embodied_ai/xh2modelzoo"
```

- 未设置 `PYTHONPATH` 时会报 `No module named 'xh_model_zoo'`；
- torch 为 `2.8.0+cu128`；
- torchao 的 C++ extension 有兼容性提示，但该提示不等同于 PTQ 失败。

## 11. 下一步学习顺序

1. 阅读 `Qwen3_VLDataPreprocess.__call__`，确认 token 和视觉 embedding 如何拼接；
2. 阅读视觉模块 `forward_ori()`，确认 `pixel_values` 与 `grid_thw` 的真实 shape；
3. 阅读 `CacheTensor`，理解 KV cache 的特殊语义；
4. 阅读 `run_native_outputs()`，对照 native LLM 的 `position_ids` 和 deepstack 输入；
5. 阅读 `dump_hmonnx_golden()`，理解 ONNX 输入名如何触发 cache wrapping；
6. 最后回到 `ptq.py:189`，理解 Qwen3-VL 的导出和量化。

## 12. 复盘问题

每次排错先回答：

1. 当前组件是 vision、prefill、decode 还是 dit？
2. 当前 tensor 是 native、wrap 还是 HMONNX 格式？
3. 是否包含 `CacheTensor`？
4. 当前是完整序列、prefill chunk 还是单 token？
5. 视觉输入是否正确归一化和重排？
6. shape 是否一致后再比较数值？
7. 问题属于模型、输入契约、量化配置还是 runtime？
