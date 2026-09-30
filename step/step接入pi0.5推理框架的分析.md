结论：这个思路可以嫁接到现有 Pi0.5 工程，而且比之前的 `realtime-vla-flash` draft 更适合当前场景。原因是 STEP 不要求替换或蒸馏 Pi0.5 主模型，而是在 Pi0.5 的 flow 推理前增加一个轻量 warm-start predictor。

但不能直接把 STEP 官方代码拷贝进 Pi0.5。需要针对 Pi0.5 的输入特征、flow 时间方向、action normalization 和 HMM 接口重新实现。

参考材料：

- 本地训练方案：[step_train.md](D:/Project/ai-/step/step_train.md)
- 本地论文：`:codex-file-citation{path="D:__codex_directive_quoted_backslash__Project__codex_directive_quoted_backslash__ai-__codex_directive_quoted_backslash__step__codex_directive_quoted_backslash__VLA推理速度优化-STEP.pdf" purpose="source"}`
- [STEP 官方代码](https://github.com/Kimho666/STEP)
- [STEP 论文](https://arxiv.org/abs/2602.08245)

## 1. STEP 的核心思想

STEP 不是重新训练一个完整 VLA，也不是用 draft 完全替代主模型，而是：

```
当前观测 o_t
历史动作块 A_cache
        │
        ▼
轻量 STEP Predictor
        │
        ▼
预测动作块 Â_t
        │
        ▼
构造 Pi0.5 flow 中间状态 x_t
        │
        ▼
Pi0.5 主模型执行 1～2 次 flow refinement
        │
        ▼
最终动作块 A_t
        │
        └── 缓存给下一次 predictor
```

它同时利用：

- 当前视觉/语言条件，保证 spatial consistency；
- 上一轮动作块，保证 temporal consistency；
- 原始 Pi0.5 主模型，负责最终修正 predictor 的误差。

STEP 官方实现使用历史动作和当前观测，通过轻量 Transformer 预测动作序列，再将预测结果作为扩散过程的 warm-start 输入。官方 README 中明确包含 action predictor、combined inference 和 warm-start diffusion 三部分。[STEP 官方实现](https://github.com/Kimho666/STEP)

## 2. 论文方法和 Pi0.5 的差异

论文主体主要基于 DDPM/DDIM 形式推导，初始化形式为：

```
A_K' = σ · Â_t + σ_t · ε
```

而 Pi0.5 使用 flow matching，当前工程的采样逻辑是：

```
x_t = noise

for step in range(num_steps):
    t = 1.0 - step / num_steps
    v_t = denoise_step(x_t, t)
    x_t = x_t - v_t / num_steps
```

对应的时间方向是：

```
t = 1       纯噪声
t = 0       干净动作
```

所以 Pi0.5 不能简单使用：

```
x_t = sigma * predicted_action + sigma_t * noise
```

应该采用与 Pi0.5 flow 定义一致的插值：

```
x_t = (1 - t_start) * predicted_action + t_start * noise
```

例如：

```
t_start = 0.5
refine_times = [0.5, 0.25, 0.0]
```

然后执行：

```
for t_i, t_next in [(0.5, 0.25), (0.25, 0.0)]:
    v_t = pi05_denoise(x_t, t_i)
    x_t = x_t - (t_i - t_next) * v_t
```

这才是适配 Pi0.5 的 STEP warm-start。

论文补充实验中报告了 Pi0.5 的结果：

```
Pi0.5 原始 10 step：成功率平均 0.972
1 step + STEP：成功率平均 0.963
```

但论文没有公开 Pi0.5 专用 predictor 权重，因此需要针对你的 Pi0.5 checkpoint 重新训练 predictor。

## 3. 当前 Pi0.5 工程的实际推理链

当前主要代码位于：

[modeling_pi05_hm.py](D:/Project/embodied_ai-main/embodied_ai-main/pi0.5/modeling_pi05_hm.py)

推理流程是：

```
HMPI05Policy.predict_action_chunk
        │
        ├── _preprocess_images
        │
        └── PI05PolicyXH2a.sample_actions
                │
                ├── embed_prefix
                │      ├── SigLIP HMM
                │      └── token embedding
                │
                ├── Gemma 2B prefill HMM
                │
                └── denoise loop
                       ├── action_in_proj HMM
                       ├── time_mlp HMM
                       ├── Gemma expert decode HMM
                       └── action_out_proj HMM
```

关键位置：

- Prefix 构造：`modeling_pi05_hm.py:496`
- Action suffix 构造：`modeling_pi05_hm.py:546`
- Flow 推理循环：`modeling_pi05_hm.py:562`
- 单步 denoise：`modeling_pi05_hm.py:653`
- HMM 推理入口：`modeling_pi05_hm.py:756`

当前 `sample_actions()` 每次都从随机噪声开始：

```
noise = self.sample_noise(actions_shape, device)
x_t = noise.float()
```

因此 STEP 的主要接入点就是这里。

## 4. 当前工程中可以复用的 Pi0.5 特征

当前 Pi0.5 已经计算出了 predictor 所需的大部分条件。

### 图像特征

当前有两个图像输入：

```
image_0: [B, 3, H, W]
image_1: [B, 3, H, W]
```

每路 SigLIP 输出通常类似：

```
[B, 256, 2048]
```

拼接后：

```
[B, 512, 2048]
```

### 语言特征

语言 token：

```
[B, L]
```

经过 embedding：

```
[B, L, 2048]
```

当前 bench 中 `L=200` 时，Prefix 长度为：

```
2 × 256 + 200 = 712
```

这和当前 HMM 的 prefix 长度契约一致。

### Prefix 特征汇总

第一版 STEP predictor 不建议重新运行视觉模型，也不建议重新设计完整 VLM。直接复用已有 Prefix embedding：

```
image_summary = image_embeds.mean(dim=1)
language_summary = masked_mean(language_embeds, language_mask)

condition = torch.cat(
    [image_summary, language_summary],
    dim=-1,
)
```

形状：

```
image_summary:    [B, 2048]
language_summary: [B, 2048]
condition:        [B, 4096]
```

再通过一个投影层：

```
[B, 4096] → [B, 128]
```

需要注意：当前代码中的两个 image slot 是两个摄像头，不是论文中的两帧时间观测。不能把当前 Pi0.5 的两个摄像头直接理解成 `n_obs=2` 的时间序列。

## 5. 建议的 Pi0.5 STEP Predictor 结构

第一版建议使用：

```
输入：
    image_summary       [B, 2048]
    language_summary    [B, 2048]
    previous_action     [B, H, A]

条件投影：
    image/language     → 128
    previous_action    → 128

网络：
    2 层 Transformer/Cross-Attention
    hidden_dim = 128
    num_heads = 4 或 8

输出：
    predicted_action   [B, T, max_action_dim]
```

其中：

```
H = 历史动作长度，例如 16
T = Pi0.5 config.chunk_size
A = Pi0.5 config.max_action_dim
```

为了和现有 `action_in_proj` 完全兼容，建议 predictor 输出内部维度：

```
max_action_dim
```

而不是只输出真实机械臂维度。多余维度使用零填充，最终仍由：

```
actions[:, :, :original_action_dim]
```

裁剪。

## 6. 训练目标

### 第一阶段：直接预测归一化动作块

训练数据：

```
obs_t
language
current image features
previous action chunk
target action chunk
```

目标：

```
target_action = normalized_expert_action_chunk
pred_action = predictor(condition, previous_action)
loss = mse(pred_action, target_action)
```

这是最稳妥的第一版，因为 Pi0.5 现有后处理已经定义了动作归一化规则。

训练目标必须和 Pi0.5 使用同一套：

- action normalization；
- action horizon；
- action dimension；
- action chunk 对齐方式；
- 数据采样时间点。

### 第二阶段：加入 flow warm-start 训练

在 direct action predictor 收敛后，再随机采样 `t_start`：

```
noise = torch.randn_like(target_action)
x_target = (1 - t_start) * target_action + t_start * noise
```

增加：

```
pred_action = predictor(...)
x_pred = (1 - t_start) * pred_action + t_start * noise
loss_warm = mse(x_pred, x_target)
```

或者直接保留：

```
loss = mse(pred_action, target_action)
```

然后在推理阶段构造 flow 中间状态。

第一阶段不建议直接训练复杂的 `flow_residual`，因为当前 Pi0.5 的 flow 方向和时间调度还需要先通过 golden 数据确认。

## 7. 历史动作如何维护

当前 `HMPI05Policy` 只有：

```
self._action_queue
```

它用于执行当前动作块，不等价于 STEP 需要的历史动作缓存。

需要额外维护：

```
self._prev_action_chunk
```

推荐逻辑：

```
episode reset:
    prev_action_chunk = None

第一次推理:
    没有历史动作，使用原始 Pi0.5 多步采样

后续推理:
    predictor(obs, prev_action_chunk)
    warm-start + 少量 Pi0.5 flow refinement
    保存本轮完整 action chunk
```

缓存的应该是完整动作块，而不是只缓存已经执行的前 `n_action_steps` 个动作。

如果每次只执行 `n_action_steps`，需要在下一轮进行 chunk 对齐：

```
上一轮完整 chunk:
[A0, A1, ..., A(T-1)]

执行:
[A0, ..., A(n-1)]

下一轮历史:
[A(n), ..., A(T-1)] + 尾部 padding
```

否则 predictor 看到的历史动作会发生时间错位。

## 8. 速度感知扰动应该放到最后

论文中的 velocity-aware perturbation 用于解决真实机器人动作停滞，不应该在第一版就加入。

建议先完成：

```
predictor
→ flow warm-start
→ 2-step refinement
→ 成功率验证
```

再加入：

```
delta_exec = norm(current_executed_action - previous_executed_action)

if delta_exec < stall_threshold:
    x_t += bounded_perturbation
```

注意这里的扰动应该：

- 在 Pi0.5 归一化动作空间中进行；
- 有最大幅度；
- 经过 action clipping；
- 只在连续多帧停滞时触发；
- 不应直接把大比例高斯噪声加到最终机械臂动作上。

论文给出的真实机器人经验范围是 `sigma_stall=1.0~1.4`，但这个数值不能直接复制到 Pi0.5，需要根据动作归一化统计重新标定。

## 9. 现有 `step_train.md` 中需要修正的地方

当前文档中的整体方向是对的，但有几处不能直接按代码执行。

### `pi05.encode_prefix()` 尚不存在

文档伪代码中有：

```
cond = pi05.encode_prefix(...)
```

现有工程只有：

```
embed_prefix(...)
```

且它返回的是：

```
prefix_embs, valid_prefix_length
```

需要新增一个用于 predictor 的特征汇总接口，而不是假设 `encode_prefix()` 已经存在。

### `flow_interpolate()` 尚不存在

文档中使用：

```
target_x_t = pi05.flow_interpolate(...)
```

当前工程没有这个函数，需要按照 Pi0.5 当前时间方向实现：

```
x_t = (1 - t) * action + t * noise
```

### `refine_from_warm_start()` 尚不存在

当前 `sample_actions()` 只支持：

```
x_t = noise
```

需要增加：

```
warm_start_action
warm_start_time
refine_steps
```

并让 flow loop 使用：

```
x_t = x_t - dt * v_t
```

### 文档中的两帧图像和当前两个摄像头不同

`step_train.md` 中：

```
images: [o_{t-1}, o_t]
```

表示时间上的两帧观测。

当前 Pi0.5 的：

```
HMONNX_SELECTED_IMAGE_INDICES = (0, 1)
```

表示两个摄像头，不是两帧图像。

第一版应该使用：

```
当前两个摄像头特征 + 历史动作
```

而不是为了凑 `n_obs=2` 再重复运行两次视觉 HMM。

## 10. 推荐的实施顺序

### 阶段一：GPU/PyTorch golden

不改 HMM，先实现一个纯 PyTorch 参考版本：

```
Pi0.5 Prefix embedding
        ↓
feature pooling
        ↓
STEP predictor
        ↓
flow-consistent warm-start
        ↓
原始 Pi0.5 10-step / 4-step / 2-step
```

验证：

```
predictor action MSE
predictor action cosine
warm-start 后的 action MSE
warm-start 后的 action cosine
LIBERO success rate
```

### 阶段二：只减少主模型步数

固定 predictor，比较：

| 模式     | Predictor | Pi0.5 flow steps |
| -------- | --------- | ---------------- |
| Baseline | 无        | 10               |
| Low-step | 无        | 2                |
| STEP-4   | 有        | 4                |
| STEP-2   | 有        | 2                |

只有 STEP-2 或 STEP-4 的任务成功率接近 Baseline，才继续量化 predictor。

### 阶段三：导出 predictor HMM

新增一个小模型导出链：

```
PyTorch STEP Predictor
        ↓
ONNX
        ↓
HMONNX
        ↓
XH2 HMM
```

推荐固定输入：

```
condition:       [1, 4096] 或 [1, 128]
prev_action:     [1, H, max_action_dim]
```

输出：

```
pred_action:     [1, chunk_size, max_action_dim]
```

第一版 predictor 可以放在 host 上运行，确认精度后再编译到 XH2。

### 阶段四：接入 HMM Pi0.5

最终推理链：

```
_preprocess_images
        ↓
embed_prefix
        ├── image_summary
        ├── language_summary
        └── predictor HMM
                 ↓
             predicted_action
        ↓
Gemma prefill
        ↓
flow warm-start
        ↓
2～4 次 Gemma expert decode
        ↓
action_out_proj
        ↓
postprocess
```

## 11. 最终判断

这个方案是可行的，但落地点不是“把 realtime-vla-flash 的 draft 接到 Pi0.5 上”，而是：

```
以当前 Pi0.5 为 teacher
重新训练一个 Pi0.5-specific STEP predictor
使用 Pi0.5 原生 flow 语义构造 warm-start
保留 Pi0.5 action expert 做最终 refinement
```

它和之前失败的 draft 方案的本质区别是：

```
之前：
Pi0 draft → 强行作为 Pi0.5 draft
现在：
Pi0.5 teacher → 训练 Pi0.5 专用 warm-start predictor
```

因此不会再出现之前那种因为 draft action/flow 空间不一致而导致 verifier 全部拒绝的问题。

我目前只完成了论文、训练文档和现有 Pi0.5 工程的分析，没有修改代码。