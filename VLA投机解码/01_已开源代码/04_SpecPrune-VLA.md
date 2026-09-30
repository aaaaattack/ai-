# SpecPrune-VLA

- **论文**：[arXiv 2509.05614](https://arxiv.org/abs/2509.05614)
- **代码仓库**：[alexwhz-sjtu/SpecPrune-VLA](https://github.com/alexwhz-sjtu/SpecPrune-VLA)
- **类型**：action-aware self-speculative pruning

## 核心方法

- action-level static visual-token pruning；
- layer-level dynamic pruning；
- action-aware controller 根据粗粒度/精细动作调整剪枝强度；
- 利用相邻机器人观察的时空一致性。

## 公开结果

- OpenVLA-OFT / LIBERO 最高约 1.57×；
- Dexbotic-OFT / SimplerEnv 约 1.44×；
- 与 uniform feature caching 组合时约 1.42×；
- 真实 Flexiv Rizon4 任务约 1.70×。

## 对后摩 NPU 的意义

它不是经典 draft/target 解码，但可以和 FLASH/KERV 叠加：先减少视觉 token 和 VLM backbone 计算，再做 action speculation。
