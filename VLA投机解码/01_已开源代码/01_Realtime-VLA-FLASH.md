# Realtime-VLA FLASH

- **论文**：[arXiv 2605.13778](https://arxiv.org/abs/2605.13778)
- **代码仓库**：[dexmal/realtime-vla-flash](https://github.com/dexmal/realtime-vla-flash)
- **项目主页**：[Realtime-VLA FLASH](https://dexmal.github.io/realtime-vla-flash/)
- **类型**：Diffusion/Flow-based VLA speculative inference
- **与 π0.5 的关系**：最高优先级参考，面向连续 action chunk。

## 核心方法

```text
Full Path：完整图像编码 + VLM + Action Expert
Flash Path：轻量 draft + 主 Action Expert 验证
                         ↓
              执行最长一致 action prefix
                         ↓
              不一致时回退 Full Path
```

## 公开结果

- 完整推理轮次约 58.0 ms；
- speculative round 最快约 7.8 ms；
- LIBERO 任务级平均推理延迟约 19.1 ms；
- 报告约 3.04× 平均任务级加速；
- 展示真实传送带分拣任务。

## 代码入口

仓库 README 给出的典型流程：`enc_cache.py` 构建 prefix cache，`spec_draft_train.py` 训练 draft head，`spec_serve_policy.py` 启动 serving，`spec_client_libero.py` 做 LIBERO 评测，`run_sweep.py` 做参数 sweep。

## 后摩迁移关注点

- prefix cache 是否能映射到 M50/XH2A SRAM/DRAM；
- draft 与 target 是否能放在不同 stream；
- verification 和 action safety check 是否可融合；
- rollback 是否可以不经过 host/PCIe；
- speculative round 中 NPU、DMA 和同步各占多少。
