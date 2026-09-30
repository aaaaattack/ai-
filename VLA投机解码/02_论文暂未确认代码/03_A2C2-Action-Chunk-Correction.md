# A2C2：Leave No Observation Behind

- **论文**：[arXiv 2509.23224](https://arxiv.org/abs/2509.23224)
- **代码**：截至本次整理，未确认官方完整代码仓库。
- **类型**：异步 action-chunk correction（相关工作，非严格 speculative decoding）

## 核心方法

使用轻量 correction head 在每个控制周期修正已经生成的 action chunk，减少完整 VLA 重算和延迟导致的 stale action。

## 学习价值

适合与 FLASH 的 speculative round、π0.5 的 Flow Matching action expert 结合，作为被拒绝 chunk 或旧 chunk 的低成本修正路径。
