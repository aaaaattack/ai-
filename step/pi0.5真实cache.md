是的，真实 cache 中的 `condition_features` 必须由真实的 Pi0.5 HMM 推理产生；仅用随机/零特征只能验证训练代码能否运行，不能代表真实数据分布。

我已在 102 机器的 `xw-dadao-xh2` 容器内完成验证：

- HMM 模型目录：

  `/workspace/921_embodied_ai/embodied_ai/pi0.5/output/xh2`

- 新增真实 cache 导出脚本：

  `/workspace/921_embodied_ai/embodied_ai/pi0.5/dump_pi05_step_cache.py`

- 已生成真实 cache：

  `/workspace/921_embodied_ai/embodied_ai/pi0.5/output/step_cache/libero_real_ep0_4.npz`

- 已复制到 STEP 工程：

  `/workspace/my_git_project/STEP/data/pi05_step_cache/libero_real_ep0_4.npz`

实际生成过程调用了：

```
policy.predict_action_chunk(processed)
```

因此会真实执行：

```
图像预处理
→ SigLIP HMM
→ Gemma prefill HMM
→ prefix_embs
→ STEP cache
```

生成结果：

```
condition_features: (4, 2, 2048)
prev_action:         (4, 16, 32)
target_action:       (4, 50, 32)
```

其中 `prefix_embs` 原始形状为：

```
(1, 712, 2048)
```

STEP 导出器将其压缩为两个条件 token：

```
image token:    图像 prefix 的平均特征
language token: 有效语言 prefix 的平均特征
```

在 STEP 工程中已经验证可以加载：

```
REAL_CACHE_LOAD_OK
dataset length = 4
```

并完成了一轮真实 cache 训练 smoke test：

```
train_mse = 1.2363
val_mse   = 0.9676
```

这说明：

1. 102 上的 HMM 推理链路可用；
2. cache 确实来自真实 HMM 输出；
3. cache 能被 STEP 数据集正确加载；
4. STEP 模型可以正常训练。

但目前只有 4 个样本，只能证明流程打通，不能用于有效训练。下一步应扩大 cache，例如：

```
cd /workspace/921_embodied_ai/embodied_ai/pi0.5

python dump_pi05_step_cache.py \
  --output output/step_cache/libero_real_large.npz \
  --episodes 20 \
  --max-samples 2000 \
  --stride 5 \
  --device cuda
```

然后将生成的 `.npz` 复制到：

```
/workspace/my_git_project/STEP/data/pi05_step_cache/
```

再正式训练 STEP predictor。已开始扩充 5000 个真实样本。

当前任务：

```
PID: 4128618
进度: 约 147 / 5000
输出:
 /workspace/921_embodied_ai/embodied_ai/pi0.5/output/step_cache/libero_real_5000.npz
日志:
 /tmp/pi05_real_5000.log
```

运行命令：

```
cd /workspace/921_embodied_ai/embodied_ai/pi0.5

nohup python dump_pi05_step_cache.py \
  --output output/step_cache/libero_real_5000.npz \
  --episodes 20 \
  --max-samples 5000 \
  --stride 5 \
  --device cpu \
  > /tmp/pi05_real_5000.log 2>&1 &
```

当前容器中 CUDA 不可用，因此使用了 `--device cpu`。这只影响 PyTorch/LeRobot 预处理器，实际 HMM 推理仍然通过：

```
TCIM_Xh2HalBackend
```

执行。

查看进度：

```
ps -p 4128618 -o pid=,etime=,stat=,cmd=
grep -c "sample=" /tmp/pi05_real_5000.log
tail -n 20 /tmp/pi05_real_5000.log
```

当前速度约为每秒 1.5～2 个样本，预计还需要约 40～55 分钟。生成完成后我会检查：

```
condition_features.shape
prev_action.shape
target_action.shape
valid_prefix_length
NaN/Inf
数值范围
```

然后将最终 cache 复制到：

```
/workspace/my_git_project/STEP/data/pi05_step_cache/
```