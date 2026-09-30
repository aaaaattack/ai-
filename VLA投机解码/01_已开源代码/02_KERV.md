# KERV：Kinematic-Rectified Speculative Decoding

- **论文**：[arXiv 2603.01581](https://arxiv.org/abs/2603.01581)
- **官方代码**：[zhengzihaoPKU/KERV](https://github.com/zhengzihaoPKU/KERV)
- **类型**：离散 action-token VLA + 运动学补偿
- **会议信息**：DAC 2026（仓库说明）

## 核心方法

- draft model 生成 action token；
- target VLA 并行验证；
- 被拒绝 token 不立即触发完整重推理；
- 使用 Kalman Filter 补偿剩余动作；
- 根据实时运动学变化动态调整 acceptance threshold。

## 工程实现

仓库包含 CPU-GPU collaborative execution、static tree-based verification、CUDA Graph replay、persistent cache、hardware-aware operator fusion 和 drafter training pipeline。

## 公开结果

论文报告约 27%–37% 加速，Success Rate 基本无明显损失。默认配置面向 batch-1 BF16；模型权重、数据集和机器人资产不随代码发布。

## 后摩迁移关注点

- CPU/NPU 协同可改为 CPU/NPU 或 host/M50 协同；
- Kalman compensation 是否能部署在 NPU vector/scalar 单元；
- 动态阈值需要读取实时状态和控制周期；
- tree verification 与 NPU memory layout 的匹配。
