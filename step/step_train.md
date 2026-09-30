下面给出一个“以 π0.5 为基座、训练 STEP warm-start 模块”的可落地方案。核心原则是：先冻结 π0.5，只训练轻量 predictor；验证有效后，再考虑 LoRA 或局部联合微调。

## 一、总体架构

```
图像 + 语言 + 状态
        ↓
冻结的 π0.5 Prefix/VLM
        ↓
视觉-语言条件特征 z_t
        +
历史动作 A_{t-H:t-1}
        ↓
STEP Spatiotemporal Predictor
        ↓
warm-start action/noise x_{t0}
        ↓
π0.5 Action Expert
        ↓
2～4 步 Flow Matching
        ↓
动作后处理与执行
```

π0.5 主干负责理解环境，STEP predictor 负责预测一个接近目标动作分布的初始状态。

## 二、训练数据

每个训练样本建议组织成：

```
{
    images: [o_{t-1}, o_t],
    language: instruction,
    state: proprioception,
    history_actions: [a_{t-H}, ..., a_{t-1}],
    target_actions: [a_t, ..., a_{t+T-1}]
}
```

推荐初始配置：

```
观测帧数 n_obs = 2
动作窗口 T = 16
历史动作长度 H = 16
动作维度 = π0.5 checkpoint 的 action_dim
```

训练数据可以来自：

- π0.5 原有机器人数据；
- 目标机器人自己的示范数据；
- LIBERO、DROID、ALOHA 等数据；
- 后摩已有 pi0.5 工程采集的真实轨迹。

重要的是，动作定义、归一化统计和 π0.5 checkpoint 必须完全一致。

## 三、阶段 0：建立 π0.5 基线

先不要训练 STEP，测量：

```
π0.5 原始 flow steps = 10 / 20
成功率
单次推理延迟
动作 chunk 质量
控制频率
执行停滞率
```

建议至少保存：

```
baseline_success_rate
baseline_latency
baseline_action_chunks
baseline_flow_trajectory
```

如果原始 π0.5 在目标任务上不能稳定完成，STEP 不会从根本上解决任务能力问题。

## 四、阶段 1：冻结 π0.5，训练 STEP Predictor

### 1. Predictor 结构

可以从论文同等规模开始：

```
输入：
    π0.5 visual/language feature
    历史 action
    当前 state

网络：
    action projection
    2 层 Transformer/Cross-Attention
    hidden_dim = 128 或 256

输出：
    [T, action_dim]
```

建议第一版只训练：

```
STEP predictor
```

冻结：

```
π0.5 vision encoder
π0.5 language encoder
π0.5 action expert
```

### 2. 训练目标

不要直接把 predictor 输出当最终动作，而是让它预测 flow sampler 在某个中间时间点的状态。

设：

```
a0：真实动作序列
ε：高斯噪声
t0：warm-start 时间点
```

使用 π0.5 原生 flow matching 的插值定义生成目标：

```
x_t0 = FlowInterpolate(ε, a0, t0)
```

然后训练：

```
L_warm = MSE(predictor(z_t, history), x_t0)
```

其中 `t0` 必须和 π0.5 的 flow sampler 时间方向保持一致，不能自行假设正向或反向定义。

### 3. 推荐损失

第一版可以使用：

```
L = L_warm
  + λ1 * L_action
  + λ2 * L_temporal
  + λ3 * L_velocity
```

含义：

```
L_warm：
    预测中间 flow 状态与教师状态的误差

L_action：
    将预测结果经过少量 flow refinement 后，与真实动作的误差

L_temporal：
    约束动作序列相邻时间步平滑

L_velocity：
    约束动作速度和加速度不过大
```

初始权重可以设置为：

```
λ1 = 0.5
λ2 = 0.05
λ3 = 0.01
```

不要一开始把平滑损失设得过大，否则会导致动作过于保守。

### 4. Predictor 训练伪代码

```
for batch in loader:
    images = batch["images"].cuda()
    language = batch["language"]
    state = batch["state"].cuda()
    history = batch["history_actions"].cuda()
    target_action = batch["target_actions"].cuda()

    # 冻结 π0.5，仅提取条件特征
    with torch.no_grad():
        cond = pi05.encode_prefix(
            images=images,
            language=language,
            state=state,
        )

    # 使用 π0.5 的 flow schedule 构造 warm-start target
    noise = torch.randn_like(target_action)
    t0 = sample_warm_start_time(batch_size=target_action.shape[0])

    target_x_t = pi05.flow_interpolate(
        noise=noise,
        action=target_action,
        time=t0,
    )

    pred_x_t = step_predictor(
        cond=cond,
        history_actions=history,
        time=t0,
    )

    loss_warm = F.mse_loss(pred_x_t, target_x_t)

    # 可选：少量 flow refinement 后再计算动作误差
    refined_action = pi05.refine_from_warm_start(
        cond=cond,
        x_t=pred_x_t,
        steps=2,
    )

    loss_action = F.mse_loss(refined_action, target_action)

    velocity = refined_action[:, 1:] - refined_action[:, :-1]
    loss_temporal = velocity.square().mean()

    loss = (
        loss_warm
        + 0.5 * loss_action
        + 0.05 * loss_temporal
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(step_predictor.parameters(), 1.0)
    optimizer.step()
```

## 五、阶段 2：联合验证 2-step / 4-step

训练完 predictor 后，固定 π0.5 和 predictor，比较：

```
10-step π0.5
4-step + STEP
2-step + STEP
2-step 无 STEP
```

重点比较：

| 方案         | 目的                   |
| ------------ | ---------------------- |
| 原始 10-step | 质量上限               |
| 原始 2-step  | 证明少步采样本身的问题 |
| STEP 4-step  | 稳定性优先             |
| STEP 2-step  | 延迟优先               |

只有当 STEP 2-step 的任务成功率接近原始 10-step，才值得进一步做硬件优化。

## 六、阶段 3：可选的联合微调

如果 predictor 已经有效，可以只对 π0.5 的 action expert 做 LoRA 或小学习率微调：

```
冻结：
    PaliGemma / vision / language backbone

训练：
    STEP predictor
    action expert LoRA
    action projection
```

推荐顺序：

```
先只训练 predictor
    ↓
再训练 predictor + action expert LoRA
    ↓
最后才考虑全量微调
```

联合微调的损失：

```
L_joint =
    L_flow
  + λ1 * L_warm
  + λ2 * L_action
  + λ3 * L_temporal
```

学习率建议：

```
STEP predictor：1e-4 ～ 3e-4
Action Expert LoRA：1e-5 ～ 5e-5
π0.5 主干：先冻结
```

## 七、阶段 4：加入速度感知扰动

速度感知扰动建议最后加入，不要和 predictor 同时从零开始训练。

运行时判断：

```
delta = torch.norm(
    predicted_action[:, 1:] - predicted_action[:, :-1],
    dim=-1,
).mean()

if delta < stall_threshold:
    x_t = x_t + sigma_stall * torch.randn_like(x_t)
else:
    x_t = sigma_scale * x_t
```

建议初始参数：

```
仿真：
    sigma_stall = 0.1

真实机器人：
    sigma_stall = 1.0～1.4
```

真实部署时必须逐步增大，否则可能产生动作抖动。

## 八、显存和训练配置

### 推荐第一阶段配置

```
π0.5：BF16
π0.5：冻结
STEP predictor：BF16
batch size：1～4
gradient accumulation：8～32
image resolution：224 或模型原生分辨率
```

显存估算：

```
离线缓存 π0.5 特征：8～16 GB
在线冻结 π0.5 + predictor：16～24 GB
LoRA 联合微调：24～40 GB
全量微调：70～80 GB 以上
```

因此建议：

```
24 GB：可以开始验证
48 GB：适合正式训练
80 GB：用于联合/全量微调
```

## 九、评估指标

不要只看 predictor 的 MSE，至少要测：

```
1. Task Success Rate
2. 2-step / 4-step 推理延迟
3. 控制频率 Hz
4. Action Chunk MSE
5. Flow refinement correction norm
6. 动作平滑度
7. Execution Stall Rate
8. 碰撞率/失败率
9. 不同初始状态下的鲁棒性
```

最重要的判断标准是：

```
STEP 2-step 成功率 ≥ 原始 π0.5 10-step 的 95%
同时延迟下降 ≥ 2 倍
```

## 十、迁移到 M50/XH2A 的落地路径

推荐部署顺序：

```
GPU PyTorch 训练
    ↓
保存 STEP predictor checkpoint
    ↓
导出 ONNX/TorchScript
    ↓
在 M50/XH2A 上部署 predictor
    ↓
π0.5 Action Expert 做 2～4 步 flow inference
    ↓
动作后处理和执行
```

硬件侧优先优化：

```
1. 缓存 π0.5 Prefix/VLM 特征
2. predictor 使用 BF16/INT8
3. 固定 action horizon
4. 固定 flow steps
5. 预分配 noise/action buffer
6. 避免每一步 PCIe 往返
7. 使用静态 shape 和 graph capture
```

最终建议是先完成这一条最小闭环：

```
冻结 π0.5
+ 训练一个 2-layer STEP predictor
+ 2-step flow refinement
+ LIBERO/自有数据验证
```

确认成功率和延迟收益后，再投入 LoRA 微调和 NPU 算子优化。
