# WA-SpecDec

- **论文**：[arXiv 2608.08725](https://arxiv.org/abs/2608.08725)
- **代码**：截至本次整理，未确认公开的官方完整代码仓库。
- **类型**：world-aware speculative decoding for VLA

## 核心方法

在 VLA prefill 阶段注入 world-model-derived physical scene awareness，使 draft proposal 和 target verification 共享 world-aware prefill state。

## 公开结果

- 相同成功率 operating point 下约 1.5× matched-success speedup；
- near-contact failure 平均下降 18.6%。

## 学习价值

它代表 acceptance 从静态 action/token 距离走向物理场景感知，适合在 FLASH 或 π0.5 action-chunk verifier 上继续研究。
