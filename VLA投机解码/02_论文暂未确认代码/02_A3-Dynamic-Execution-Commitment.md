# A3：Dynamic Execution Commitment of VLA Models

- **论文**：[arXiv 2605.11567](https://arxiv.org/abs/2605.11567)
- **代码**：截至本次整理，未确认作者公开的官方完整代码仓库。
- **类型**：self-speculative prefix verification / adaptive action horizon

## 核心方法

- 通过多次采样计算 trajectory-wise consensus；
- 选择代表性 draft；
- 验证低一致性动作；
- 只执行最长、连续、通过验证的 action prefix；
- 根据状态动态决定 action horizon，避免固定 chunk 长度。

## 对 π0.5 的意义

可以作为 π0.5 action chunk 提交策略的上层控制器：空旷场景执行更长 chunk，接触/抓取场景自动缩短 horizon。
