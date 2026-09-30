# Spec-VLA

- **论文**：[ACL Anthology / EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1367/)
- **PDF**：[论文 PDF](https://aclanthology.org/2025.emnlp-main.1367.pdf)
- **代码仓库**：[PineTreeWss/SpecVLA](https://github.com/PineTreeWss/SpecVLA)
- **类型**：离散 action-token VLA speculative decoding

## 核心方法

compact VLA drafter 生成候选 action tokens，full VLA target 并行验证，并使用 action-token 相对距离进行 relaxed acceptance。

## 公开结果

- acceptance length 提升约 44%；
- 相对 OpenVLA 约 1.42× 加速；
- 不降低任务成功率。

## 与 π0.5 的关系

主要适合 OpenVLA 这类离散 action-token 模型。π0.5 低层主要是 Flow Matching，因此不能直接套用；可用于 π0.5 的高层文本/子任务路径，或作为 FAST/discrete action 分支的参考。
