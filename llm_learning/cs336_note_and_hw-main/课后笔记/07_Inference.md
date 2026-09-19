# 第7课：推理采样与自回归生成

> 学习路线：阶段1 推理链路（CS336 hw6 + hw7）
> 对应文件：
> - CS336 hw6：`chapter1/hw6/inference.py`
> - CS336 hw7（修正版）：`chapter1/hw7/inference.py`
> - nano-vllm Sampler：`nano-vllm/nanovllm/layers/sampler.py`（第4课已讲）
> - Houmo demo：`houmo-examples-xh2/apis/inferences/qwen3/demo.py`

---

## 一、自回归 Decode 循环

模型生成文本不是一次性输出全部 token，而是一个 token 一个 token 地"长"出来：

```
输入: "你好"
  → model("你好") → logits → 采样 → "世"
  → model("你好世") → logits → 采样 → "界"
  → model("你好世界") → logits → 采样 → "！"
  → ...直到 max_tokens 或 <eos>
```

**为什么叫"自回归"**：每一步都把自己的输出喂回给下一步。

### 1.1 decode_token 主函数

```python
def decode_token(input_tokens, model, max_tokens_to_generate, top_p=0.9, temperature=1.0):
    model.eval()
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)   # (seq,) → (1, seq)

    with torch.no_grad():   # 推理不需要梯度，省显存
        for _ in range(max_tokens_to_generate):
            logits = model(input_tokens)                     # (1, seq, vocab_size)
            probabilities = temperature_scaling(logits, temperature)
            next_token_idx = top_p_sampling(probabilities, top_p)
            input_tokens = torch.cat([input_tokens, next_token_idx], dim=-1)

    return input_tokens
```

### 1.2 每个关键语句

| 语句 | 作用 |
|---|---|
| `model.eval()` | 关掉 dropout 等训练专用层 |
| `torch.no_grad()` | 不构建计算图，不存中间激活值 → 省 50%+ 显存 |
| `model(input_tokens)` | forward 一次，输出整个序列每个位置的 logits |
| `torch.cat([...])` | 把新 token 拼到序列末尾，下次 forward 用更长的输入 |

---

## 二、Temperature 温度缩放

```python
def temperature_scaling(logits, temperature=1.0):
    probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    return probabilities
```

### 2.1 `logits[:, -1, :]` — 为什么只取最后一个位置？

```
模型的输出 logits: (1, seq_len, vocab_size)
                     batch 0
                     位置0: [各token分数]
                     位置1: [各token分数]
                     ...
                     位置N-1: [各token分数]  ← 最后一个位置

我们只需要"下一个token" → 最后一个位置的 logits
前面位置的 logits 是历史预测，不需要了
```

### 2.2 `/ temperature` — 温度的效果

回顾第4课 Softmax 的讲解：

```
temperature = 0.6: logits / 0.6 → 高概率 token 更突出（写代码）
temperature = 1.0: 不做缩放
temperature = 1.5: logits / 1.5 → 分布更平滑（写诗）
```

### 2.3 与 nano-vllm 的对比

```python
# CS336: 先 / temperature，然后 softmax
probs = softmax(logits[:, -1, :] / temperature)

# nano-vllm: 先 / temperature，然后 softmax，然后 Gumbel 采样
logits = logits.float().div_(temperatures.unsqueeze(1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(exp_noise).argmax(dim=-1)  # Gumbel-max
```

---

## 三、Top-p (Nucleus) 采样

### 3.1 核心思想

不是从全部词表中采样，而是从"累积概率达到 p 的最小集合"中采样：

```
所有 token 按概率排序: [A:0.5, B:0.3, C:0.1, D:0.07, E:0.03]

累积和: [0.5, 0.8, 0.9, 0.97, 1.0]
               ↑ top_p=0.9  → 保留 A, B, C，丢弃 D, E

重新归一化: [0.56, 0.33, 0.11]
然后 multinomial 采样 → 结果只能是 A, B, C 之一
```

### 3.2 hw6 版本（有 bug 的版本）

```python
def top_p_sampling(probabilities, top_p=0.9):
    sort_probabilities, idx = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sort_probabilities, dim=-1)

    mask = cumulative_probabilities > top_p   # 超过阈值的标为 True
    sort_probabilities[mask] = 0              # 这些概率置 0

    sort_probabilities.div_(sort_probabilities.sum(dim=-1, keepdim=True))
    next_token_idx = torch.multinomial(sort_probabilities, 1)
    next_token_idx = torch.gather(idx, dim=-1, index=next_token_idx)
    return next_token_idx
```

**Bug 在哪？**

```python
# 极端情况: 所有 token 都超过阈值
probs = [0.5, 0.3, 0.2]     # 排序降序
cumsum = [0.5, 0.8, 1.0]    # 全部 > top_p=0.0？不，top_p 通常 0.9

# 但考虑 top_p=0.4:
cumsum = [0.5, 0.8, 1.0]
mask = cumsum > 0.4 → [True, True, True]  # 全部被 mask！
# softmax 全 0 → 除以 0 → nan
```

### 3.3 hw7 版本（修正版）

```python
def top_p_sampling(probabilities, top_p=0.9):
    sort_probabilities, idx = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sort_probabilities, dim=-1)

    mask = cumulative_probabilities > top_p

    # ═══════ 修正 ═══════
    mask[..., 1:] = mask[..., :-1].clone()  # 右移一位：第 i 个继承 i-1 的标记
    mask[..., 0] = 0                         # 第一个 token 永不 mask
    # 保证: 至少保留 1 个 token

    sort_probabilities[mask] = 0
    sort_probabilities.div_(sort_probabilities.sum(dim=-1, keepdim=True))

    next_token_idx = torch.multinomial(sort_probabilities, 1)
    next_token_idx = torch.gather(idx, dim=-1, index=next_token_idx)
    return next_token_idx
```

**`mask[..., 1:] = mask[..., :-1].clone()` 详解**：

```python
# 原始 mask（cumsum > 0.9）:
cumsum = [0.5, 0.8, 0.95, 0.99, 1.0]
mask   = [F,   F,   T,    T,    T  ]
# 问题: 第三个 token (0.95 > 0.9) 也要被筛掉 → 只剩 2 个

# mask[1:] = mask[:-1].clone()
mask_before = [F, F, T, T, T]
shifted     = [F, F, F, T, T]   # 右移一位
# 第 0 位固定为 F（不被筛）

# 最终 mask = [F, F, F, T, T]
# → 保留前 3 个 token ✅
```

**`torch.multinomial`**：从概率分布中按权重随机采样。

```python
# multinomial 内部: 用 CDF + 随机数实现加权采样
probs = [0.56, 0.33, 0.11]
cdf   = [0.56, 0.89, 1.00]
rand  = random.uniform(0, 1)   # e.g. 0.67
# 0.67 > 0.56, 0.67 < 0.89 → 选中 idx=1
```

**`torch.gather(idx, dim=-1, index=chosen)`**：排序后的索引映射回原始 token_id。

```python
# 排序后: sorted_probs 的第 k 个 → 原始 idx[k]
idx = [3, 1, 0, 2]       # 原来第 0,1,2,3 个 token 重排
sorted_probs 已改变 → multinomial 选中 sorted_probs 的第 1 个
gather(idx, index=1) → idx[1] = 1  ← 原始 token_id
```

---

## 四、完整推理链路

```
用户 prompt: "你好"
  │
  ▼ tokenize
[101, 204, 305]   (3 个 token IDs)
  │
  ▼ decode_token 主循环:
  │
  ├─[iter 1] model([101,204,305]) → logits(1,3,10K) → 取 [-1] → probs → 采样 → 42
  │           input: [101,204,305,42]
  │
  ├─[iter 2] model([101,204,305,42]) → logits(1,4,10K) → 取 [-1] → probs → 采样 → 87
  │           input: [101,204,305,42,87]
  │
  ├─[iter 3] ... → 采样 → eos_token → break
  │
  ▼
output: [101,204,305,42,87]
  │
  ▼ detokenize
"你好世界"
```

---

## 五、与 nano-vllm 和 Houmo 的对比

| | CS336 (hw6/hw7) | nano-vllm Sampler | Houmo Qwen3 demo |
|---|---|---|---|
| **温度** | `softmax(logits[:,-1,:] / T)` | `logits.float().div_(temperatures.unsqueeze(1))` | `--temperature` |
| **采样方式** | Top-p + multinomial | Gumbel-max | Top-p + Top-k + multinomial |
| **重复惩罚** | ❌ | ❌ | `--repetition_penalty` |
| **停止条件** | `<endoftext>` 或 max_tokens | Scheduler 检查 FINISHED 状态 | 同 |
| **执行位置** | CPU/Python（代码简单） | GPU 上融合 kernel（性能优先） | CPU/Python （功能丰富） |
| **batch 支持** | batch=1 | Continuous batching（多请求并行）| batch=1 |

---

## 六、推理时的两个性能问题

### 6.1 为什么 Decode 比 Prefill 慢那么多？

```
Prefill: "你好，世界" → 4 个 token → 一次 forward → 4 次矩阵乘并行
Decode:  生成 100 个 token → 100 次 forward → 每次 1 个 token 的矩阵乘

Prefill 计算密集（大 GEMM），GPU 忙。
Decode 访存密集（KV Cache 大，计算量小），GPU 等显存。
```

### 6.2 为什么每次要重新输入整个序列？

```python
# naive: 每次只输入最后 1 个 token → 模型不知道历史上下文 → 输出乱码
# 正确: 输入完整序列 → 模型看到所有历史

# KV Cache 优化: 不重复计算历史 token 的 K 和 V
# 只算新 token 的 Q、K、V → 新 Q 和所有历史的 K、V 做 Attention
# 这是 nano-vllm block_manager 和 flash_attn_with_kvcache 的核心价值
```

---

## 七、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **推理工具链** | decode 循环是你的引擎最核心的调度逻辑。每步 forward 后如何处理 logits、如何采样、如何管理 token 序列 → vLLM/llama.cpp 的核心 |
| **量化** | Sampler 中的 `logits[:, -1, :] / temperature` 在 W8A8 时 logits 是 INT8 → 温度缩放要转回 FP 再做 |
| **编译器** | top-p 的 sort + cumsum + gather 是复杂的控制流操作，编译器很难融合。Gumbel-max（nano-vllm）恰好避开了这个问题 |
| **HAL/Runtime** | 每次 decode 需要从 NPU 取回 1 个 token 的 logits → CPU 做采样 → 下一个 token_id 再传给 NPU。这个来回延迟在你们的 TCIM Runtime 中占多少？ |

---

## 八、思考题

1. **为什么 `logits[:, -1, :]` 只取最后一个位置的 logits？前面位置的 logits 能不能用？**
   - 提示：生成任务只需要"下一个 token"，前面的 logits 是对应之前位置的输出

2. **hw7 的 mask 右移修正为什么能保证至少保留一个 token？**
   - 提示：`mask[0] = 0` 确保第一个（概率最高的）永不被筛

3. **自回归 `torch.cat` 不断拼接输入，序列越来越长，显存会怎么变化？**
   - 提示：KV Cache 随序列长度线性增长 → PagedAttention 和 KV Cache 量化的必要性

4. **如果 batch_size > 1，不同 seq 生成的 token 数量不同，decode 循环怎么处理？**
   - 提示：这是 continuous batching 的核心问题 → nano-vllm scheduler 用 FINISHED 状态管理

---

## 九、CS336 课后作业

### 作业：实现完整的推理 pipeline

```python
# 实现三个函数
def temperature_scaling(logits, temperature=1.0):
    """返回最后一个 token 的概率分布"""

def top_p_sampling(probabilities, top_p=0.9):
    """从概率分布中按 top-p 策略采样"""

def decode_token(input_tokens, model, max_tokens, top_p=0.9, temperature=1.0):
    """自回归生成循环"""
```

### 验证方法

```python
# 用最简单的场景测试

# 模拟: 一个"模型"总是输出 logits，第一个 token 最大
class DummyModel:
    def eval(self): pass
    def __call__(self, x):
        bs, seq = x.shape
        logits = torch.zeros(1, seq, 10)
        logits[:, -1, 0] = 10.0  # token_0 永远是最高分的
        return logits

model = DummyModel()
output = decode_token([1, 2, 3], model, max_tokens=5, temperature=1.0)
print(output)  # 应该是 [1,2,3,0,0,0,0,0] — 每次采样都选 token_0

# Top-p 测试: 极端的 top_p=0.01 → 应该只保留最高概率 token
probs = torch.tensor([0.6, 0.3, 0.1])
token = top_p_sampling(probs, top_p=0.01)
# 结果应该永远是 0（因为 0.6 > 0.01）
```

### 扩展思考

1. 修改 decode 循环，加一个 KV Cache 机制：每次只输入最后一个 token，自己手动管理 past_key_values
2. Benchmark: 100 次 decode 的 wall-clock 时间，对比 model.forward 和 sampling 各占多少
3. 实现 repetition penalty：对已出现的 token，logit 乘以一个惩罚因子 < 1.0
