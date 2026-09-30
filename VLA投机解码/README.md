# VLA 投机解码资料库

本目录整理 VLA speculative decoding / speculative inference 的论文、开源代码和工程落地信息。

## 目录

- `01_已开源代码/`：有公开代码，可优先复现；
- `02_论文暂未确认代码/`：论文或项目页面已公开，但暂未找到可确认的官方完整代码；
- `03_复现与后摩NPU迁移记录/`：用于记录 π0.5、OpenVLA 和 M50/XH2A 的复现实验。

## 推荐阅读顺序

1. **Realtime-VLA FLASH**：最接近 π0/π0.5 Flow Matching / Diffusion action expert；
2. **KERV**：学习运动学补偿、动态 acceptance threshold 和 Runtime 协同；
3. **Spec-VLA**：理解离散 action-token speculative decoding；
4. **SpecPrune-VLA**：理解视觉 token 的 self-speculative pruning；
5. **WA-SpecDec / A3**：学习 world-aware verification 和动态 action horizon。

## 与 π0.5 的关系

π0.5 低层控制主要使用 Flow Matching 连续动作专家，因此优先参考 FLASH；Spec-VLA、KERV 更适合离散 action-token VLA。π0.5 的高层文本/子任务路径可以直接复用传统 speculative decoding，但低层 Flow Matching 需要改成 action-chunk / trajectory-level verification。

## 后摩 NPU 迁移重点

- draft / target action expert 的异步并行；
- action chunk acceptance 与最长可执行前缀；
- action distance、运动学约束和碰撞安全检查；
- visual feature / VLM prefix / KV cache 复用；
- NPU stream、DMA、rollback buffer 和控制周期抖动；
- `accepted_horizon`、`action_deviation`、`task_success`、`p95 latency`、`tokens/J` 等指标。
