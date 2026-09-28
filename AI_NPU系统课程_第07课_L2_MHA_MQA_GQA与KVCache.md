# 第 07 课｜L2.2 MHA、MQA、GQA 与 KV Cache

## 一、本课学习顺序

本课不会一次讲完，按以下顺序逐段学习并检查：

1. MHA、MQA、GQA 的 head 数关系；
2. GQA 中 Query head 如何分组共享 K/V；
3. Q、K、V 的不同 shape；
4. KV Cache 为什么主要改善 Decode；
5. nano-vLLM 中 KV Cache 的实际存储、block table 和 slot mapping；
6. Prefill 与 Decode 如何使用 KV Cache。

本课结合以下实际源码：

- `llm_learning/nano-vllm/nanovllm/models/qwen3.py`
- `llm_learning/nano-vllm/nanovllm/layers/attention.py`
- `llm_learning/nano-vllm/nanovllm/engine/model_runner.py`
- `llm_learning/nano-vllm/nanovllm/engine/block_manager.py`

---

## 二、第一段：MHA、MQA、GQA 的区别

先只看两个数量：

```text
num_heads     ：Query head 的数量
num_kv_heads  ：Key head 和 Value head 的数量
```

### 2.1 MHA

Multi-Head Attention 中：

```text
num_kv_heads = num_heads
```

每个 Query head 都有自己对应的一组 Key head 和 Value head。

示例：

```text
num_heads = 8
num_kv_heads = 8
```

### 2.2 MQA

Multi-Query Attention 中：

```text
num_kv_heads = 1
```

所有 Query heads 共同使用同一组 Key/Value。KV Cache 最省，但共享最强，模型能力和实现取舍需要额外考虑。

示例：

```text
num_heads = 8
num_kv_heads = 1
```

### 2.3 GQA

Grouped-Query Attention 位于二者之间：

```text
1 < num_kv_heads < num_heads
```

多个 Query heads 被分成若干组，每组共享一组 Key/Value：

```text
每组 Query head 数 = num_heads / num_kv_heads
```

示例：

```text
num_heads = 8
num_kv_heads = 2

每组 Query head 数 = 8 / 2 = 4
```

也就是每 4 个 Query heads 共享 1 个 Key head 和 1 个 Value head。

### 2.4 为什么要减少 KV heads

Decode 时，每生成一个新 token，都要读取历史 token 的 K/V。减少 `num_kv_heads` 可以直接减少：

- KV Cache 占用；
- 每个 Decode step 的 K/V 读取量；
- 内存带宽压力；
- 长上下文和高并发时的显存压力。

GQA 在“保留多个 KV heads”与“降低 KV Cache 成本”之间做折中。

---

## 三、nano-vLLM 中的真实实现

文件：`llm_learning/nano-vllm/nanovllm/models/qwen3.py`

`Qwen3Attention` 同时接收两个不同参数：

```python
num_heads=config.num_attention_heads
num_kv_heads=config.num_key_value_heads
```

源码分别计算 Query 与 K/V 的大小：

```python
self.head_dim = head_dim or hidden_size // self.total_num_heads
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
```

投影后也按不同大小切分和 reshape：

```python
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

这说明在 GQA 中，Q 的 head 数与 K/V 的 head 数确实不同，不是只在概念上共享。

Flash Attention kernel 可以按照分组关系处理它们，通常不需要先在 HBM 中把 K/V 物理复制成与 Q 相同的 head 数。

### 逐行注释版：GQA 的最小切分

```python
self.q_size = self.num_heads * self.head_dim                 # Q 的总宽度
self.kv_size = self.num_kv_heads * self.head_dim              # K/V 的总宽度，GQA 下通常更小
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 一次投影后按宽度切分
q = q.view(-1, self.num_heads, self.head_dim)                 # Q 使用更多 heads
k = k.view(-1, self.num_kv_heads, self.head_dim)             # K 使用较少 heads
v = v.view(-1, self.num_kv_heads, self.head_dim)             # V 与 K 共享 KV head 数
```

例如 `num_heads=16、num_kv_heads=4` 时，每 4 个 Q heads 共享 1 组 K/V；KV Cache 的 head 维度约为 MHA 的四分之一。

---

## 四、第一段检查题

某模型配置：

```text
num_heads = 32
num_kv_heads = 8
```

只回答两个问题：

1. 这是 MHA、MQA 还是 GQA？
2. 每几个 Query heads 共享一组 Key/Value？

**我的回答：**

1. GQA。
2. 4。

**批改：完全正确。**

```text
1 < num_kv_heads(8) < num_heads(32)，所以是 GQA。
每组 Query heads = num_heads / num_kv_heads = 32 / 8 = 4。
```

也就是说，Query heads 以4个为一组，共享同一组 Key/Value。

---

## 五、第二段：GQA 下 Q、K、V 的不同 shape

上一课中，MHA 的 Q、K、V 都可写为：

```text
[batch_size, num_heads, sequence_length, head_dim]
```

但 GQA 不同：Query 使用 `num_heads`，Key/Value 使用更少的 `num_kv_heads`：

```text
Q : [batch_size, num_heads,    sequence_length, head_dim]
K : [batch_size, num_kv_heads, sequence_length, head_dim]
V : [batch_size, num_kv_heads, sequence_length, head_dim]
```

这与 nano-vLLM 的实际代码一致：`q.view(..., self.num_heads, self.head_dim)`，而 `k/v.view(..., self.num_kv_heads, self.head_dim)`。

## 六、第二段检查题

继续使用：

```text
batch_size = 2
sequence_length = 16
hidden_size = 1024
num_heads = 16
num_kv_heads = 4
```

已知：

```text
head_dim = hidden_size / num_heads
```

请写出：

1. `head_dim`；
2. Q 的 shape；
3. K 和 V 的 shape。

**我的回答：**

1. `head_dim=hidden_size/num_heads=64/8=8`。
2. Q/K/V shape 还不清楚。
3. Q/K/V shape 还不清楚。

**批改：** `head_dim=8` 正确。Q/K/V shape 暂不要求继续作答；先用一个最小例子建立“拆头”直觉。

### 先理解：hidden_size 为什么会消失

`hidden_size=64` 的意思是：**每个 token 在拆头前有一个长度为64的向量。**

当 `num_heads=8` 时，不是把 64 丢掉，而是把一个长度为64的向量均分成8段：

```text
64 = 8 × 8
     ↑   ↑
     8个 head，每个 head 内有8个数
```

因此：

```text
hidden_size = num_heads × head_dim
64          = 8         × 8
```

拆头后不再用一个 `hidden_size=64` 维度表示它，而是改用两个维度 `num_heads=8` 和 `head_dim=8` 表示同一批数。**元素总数没有改变。**

### 最小例子：只看 Q

给定：

```text
batch_size=1
sequence_length=5
hidden_size=64
num_heads=8
head_dim=8
```

输入有1条序列、5个 token、每个 token 64个数：

```text
X: [1, 5, 64]
```

Q 投影后，仍是每个 token 一个64维向量：

```text
Q projection output: [1, 5, 64]
```

现在只对最后的64维拆头：

```text
[1, 5, 64]
→ reshape
[1, 5, 8, 8]
          ↑  ↑
       8个head，每个head有8个数
→ transpose
[1, 8, 5, 8]
```

最后的 `[1,8,5,8]` 读法是：

```text
1条序列
→ 8个 Query heads
→ 每个 head 有5个 token 位置
→ 每个 token 在该 head 内有8个数
```

所以 Q 的正确 shape 是：

```text
Q = [batch_size, num_heads, sequence_length, head_dim]
  = [1, 8, 5, 8]
```

### GQA 中 K/V 为什么不同

GQA 的 `num_kv_heads=2`，表示 K/V 只有2个 heads，而不是8个。每个 K/V head 的内部宽度仍是 `head_dim=8`：

```text
K/V projection output: [1, 5, 16]
                           ↑
                    16 = 2 × 8

view + transpose: [1, 2, 5, 8]
```

因此：

```text
Q   = [1, 8, 5, 8]
K/V = [1, 2, 5, 8]
```

在本例中，8个 Query heads 分为4组，每组4个 Query heads 共享1个 K/V head。nano-vLLM 的 Qwen3 实现正是分别用 `self.num_heads` reshape Q、用 `self.num_kv_heads` reshape K/V。

### 修正原题答案

原题参数是 `batch_size=2、sequence_length=16、hidden_size=1024、num_heads=16、num_kv_heads=4`：

```text
head_dim = 1024 / 16 = 64

Q   : [batch_size, num_heads,    sequence_length, head_dim]
    = [2, 16, 16, 64]

K/V : [batch_size, num_kv_heads, sequence_length, head_dim]
    = [2,  4, 16, 64]
```

本题的关键不是记住一串数字，而是区分字段职责：

```text
hidden_size：拆头前的总宽度，不再出现在拆头后的 shape 中；
num_heads：只出现在 Q 的第二维；
num_kv_heads：只出现在 K/V 的第二维；
head_dim：最后一维，且 head_dim = hidden_size / num_heads。
```

### 第二段最小确认

暂不要求你重新计算完整 Q/K/V。只回答下面两句：

```text
在 Q=[1,8,5,8] 中，第二个 8 表示什么？
在 Q=[1,8,5,8] 中，最后一个 8 表示什么？
```

本题通过后，再学习 KV Cache 的容量为什么随 `num_kv_heads` 改变。

**我的回答：**

```text
第一个 8 是 Q 头数；第二个 8 是每个 token 在这个头里有 8 个参数。
```

**批改：正确。** 更严谨地说，最后一个 `8` 是 `head_dim`：每个 token 在**每一个 Query head** 内的 8 个特征分量。这里的“参数”容易和模型权重混淆；它们是本次前向计算中 token 的激活值。

因此：

```text
Q = [batch_size, num_heads, sequence_length, head_dim]
  = [1,          8,         5,               8]
```

第二段通过。现在进入 KV Cache。

---

## 七、第三段：KV Cache 为什么只缓存 K/V

在自回归 Decode 中，模型每次只新生成一个 token。新 token 的 Query 要和所有历史 token 的 Key 做注意力计算，并使用对应的 Value 汇总信息。

历史 token 的 K/V 不会变化；若每一步重新计算它们，会重复做大量相同工作。因此把每层历史 K 与 V 保存下来：这就是 KV Cache。当前 token 的 Q 只在当前这一步需要，下一步会产生新的 Q，所以通常不缓存 Q。

对于逻辑上的一层 GQA KV Cache：

```text
K cache: [batch_size, num_kv_heads, context_length, head_dim]
V cache: [batch_size, num_kv_heads, context_length, head_dim]
```

注意第二维是 `num_kv_heads`，不是 `num_heads`；这正是 GQA 减少缓存量的原因。

### 第三段检查题

继续使用：

```text
num_heads = 8
num_kv_heads = 2
```

只回答：KV Cache 要保存 Q、K、V 中的哪些？它的 head 数为什么是 2 而不是 8？

**我的回答：**

```text
KV，分成4组了。
```

**批改：正确。** KV Cache 保存 K 和 V，不保存 Q。`8 / 2 = 4` 表示每个 KV head 被 4 个 Query heads 共享；因此只有 2 个不同的 K/V heads 需要存入 Cache，而不是为 8 个 Query heads 各存一份。

### nano-vLLM 的实际 Cache layout

上一节是方便理解的逻辑 shape；nano-vLLM 为了让不同序列共享固定大小的内存块，实际分配的是：

```text
[2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim]
 ↑
K/V 两份缓存
```

这来自 `engine/model_runner.py`：第一维的 `2` 区分 K cache 与 V cache，`block_size` 表示一个缓存块能容纳多少个 token。某一层、某一序列实际使用的 K cache，仍可以理解为由若干 block 拼出的：

```text
[num_kv_heads, context_length, head_dim]
```

例如在 GQA `num_kv_heads=2、head_dim=8` 中，历史长度为 5 时，该层的 K 与 V 各有：

```text
[2, 5, 8]
```

它表示 2 个 KV heads、5 个历史 token、每个 token 在每个 KV head 内 8 个特征分量。

### 第三段进阶确认

若 GQA 使用：

```text
num_heads = 16
num_kv_heads = 4
head_dim = 64
context_length = 100
```

忽略 `batch_size` 和 layer 数。请写出**单层、仅 K cache**的逻辑 shape；再说明 K 加 V 一共含有多少个数值。

**我的回答：**

```text
不知道。
```

**讲解与订正：** 单层、仅 K cache 的逻辑 shape 直接套用：

```text
[num_kv_heads, context_length, head_dim]
= [4,            100,            64]
```

K cache 的数值个数是各维相乘：

```text
4 × 100 × 64 = 25,600
```

V cache 的 shape 和数值个数完全相同，因此 K 加 V：

```text
2 × 25,600 = 51,200 个数值
```

这里没有 `num_heads=16`，因为存的是 K/V；GQA 的 K/V head 数是 `num_kv_heads=4`。`context_length=100` 表示这100个历史 token 的 K/V 都要留下，故缓存容量会随上下文长度线性增长。

### 最小复测

不做大数相乘。若：

```text
num_kv_heads = 2
context_length = 3
head_dim = 4
```

请只写：单层 K cache 的 shape，以及 K 加 V 的数值总个数。

**参考答案：**

```text
K cache shape = [2, 3, 4]
K 的数值数 = 2 × 3 × 4 = 24
K + V 的数值总数 = 24 + 24 = 48
```

你选择先查看参考答案；后续可结合本段内容自行复习。继续学习不依赖于此题的独立作答。

---

## 八、第四段：Prefill 与 Decode 如何使用 KV Cache

一次生成可以分为两个阶段：

```text
用户输入的 prompt（很多 token）
        │
        ▼
Prefill：计算整段 prompt 的 K/V，并写入 KV Cache
        │
        ▼
Decode：每次生成 1 个新 token；写入新 token 的 K/V，读取全部历史 K/V
        │
        ▼
下一个 Decode step
```

### 8.1 Prefill：第一次建缓存

假设 prompt 有 100 个 token。Prefill 一次处理这 100 个 token，算出它们所有的 Q/K/V；其中 K/V 会写入 cache，供后续生成使用。

这一步的 Attention 中，token 之间还需要互相计算，所以计算量通常较大；但一次并行处理多个 token，通常更偏向计算密集。

### 8.2 Decode：反复使用缓存

假设接着要生成第 101 个 token：

1. 仅为新 token 算它的 Q/K/V；
2. 把新 token 的 K/V 追加到 cache；
3. 用新 token 的 Q 读取 cache 中第 1 到 101 个 token 的 K/V；
4. 得到第 101 个 token 的输出。

生成第 102 个 token 时，同样只算新 token，但这次读取长度为 102 的历史 K/V。也就是说 Decode 的单步计算很小，读 cache 的量却随着上下文变长而增加，常常更受显存带宽限制。

### 8.3 nano-vLLM 对应位置

在 `llm_learning/nano-vllm/nanovllm/engine/model_runner.py`，`run()` 依据 `is_prefill` 选择 `prepare_prefill()` 或 `prepare_decode()`，分别准备不同的 `slot_mapping`、`block_tables` 和 context 信息。

在 `llm_learning/nano-vllm/nanovllm/layers/attention.py`：

- 两阶段都会调用 `store_kvcache(...)` 写入本次新得到的 K/V；
- Prefill 使用 `flash_attn_varlen_func(...)` 处理整段 token；
- Decode 使用 `flash_attn_with_kvcache(...)`，将当前 Q 与已有 K/V cache 一起用于注意力计算。

### 第四段检查题

某请求的 prompt 长度为 100，随后已经生成了 2 个 token。现在要生成第 3 个新 token。

只回答两句：

1. 此时属于 Prefill 还是 Decode？
2. 当前 token 的 Q 要读取 cache 中多少个 token 的 K/V？

**我的回答：**

```text
1. Decode。
2. 不知道。
```

**批改：** 第 1 题正确。第 2 题答案是 **103 个 token 的 K/V**。

计算顺序：

```text
prompt 中已有              100 个 token 的 K/V
此前已生成                    2 个 token 的 K/V
本次第 3 个新 token 先写入     1 个 token 的 K/V
---------------------------------------------
当前 Q 可读取的 K/V 总数      103 个 token
```

这里“当前 token 也能关注自己”，所以总数要加上本次新 token。简记为：

```text
当前 Decode step 的 attention 长度
= prompt_length + 已生成 token 数 + 1
```

### 第四段最小复测

prompt 长度为 4，已经生成了 1 个 token；现在要生成下一个 token。它属于哪个阶段？当前 Q 要读取多少个 token 的 K/V？

**参考答案：**

```text
阶段：Decode

当前 Q 读取的 K/V 数量：
4（prompt）+ 1（此前已生成）+ 1（当前新 token）= 6 个 token 的 K/V
```

### 完整案例：从 Prefill 到连续 Decode

假设用户输入的 prompt 是 4 个 token：

```text
[P1, P2, P3, P4]
```

#### Step 1：Prefill

一次性处理四个 prompt token，计算并写入它们的 K/V：

```text
Cache = [P1, P2, P3, P4]
cache 长度 = 4
```

#### Step 2：生成第 1 个新 token `G1`

计算 `G1` 的 Q/K/V，先把 `G1` 的 K/V 写入 cache；`G1` 的 Q 读取 P1 到 G1 的 K/V：

```text
Cache = [P1, P2, P3, P4, G1]
attention 长度 = 4 + 0 + 1 = 5
```

#### Step 3：生成第 2 个新 token `G2`

```text
Cache = [P1, P2, P3, P4, G1, G2]
attention 长度 = 4 + 1 + 1 = 6
```

通用规律：

```text
第 n 个新 token 的 Decode attention 长度
= prompt_length + (n - 1) + 1
= prompt_length + n
```

因此，随着连续生成，KV Cache 和每个 Decode step 需要读取的历史 K/V 都线性增长。

---

## 九、第五段：为什么需要 block table 与 slot mapping

前面把一条序列的 cache 想成连续的：

```text
[num_kv_heads, context_length, head_dim]
```

真实服务里会同时处理很多请求，它们的长度不同、结束时间不同，也可能共享相同的 prompt 前缀。若每条请求都要求一段连续显存，扩容与回收会造成大量碎片。

nano-vLLM 因此把 KV Cache 切成固定大小的 **block**。可以把它想成停车场：

```text
逻辑 token 位置：  0   1   2   3 | 4   5   6   7 | 8 ...
逻辑 block：       0               | 1               | 2 ...

物理 block 编号：  9               | 3               | 14 ...
```

逻辑 block 的顺序不变，但它们在显存中的物理编号可以不连续。

### 9.1 block table：读 cache 时的“地址目录”

对某条序列，`block_table` 记录：每个逻辑 block 对应哪个物理 block。

例如 block size 为 4，某条序列有 10 个 token，且：

```text
block_table = [9, 3, 14]
```

则：

```text
token 0–3   → 物理 block 9
token 4–7   → 物理 block 3
token 8–9   → 物理 block 14
```

Attention 在读取历史 K/V 时，依赖这个目录把逻辑 token 位置翻译到正确的物理 block。`engine/model_runner.py` 在准备 Prefill/Decode context 时会传入 `block_tables`；`layers/attention.py` 的 Decode 路径把它交给 `flash_attn_with_kvcache(...)`。

### 9.2 slot mapping：写入 cache 时的“具体车位”

`slot_mapping` 标出**本次新算出的每个 token 的 K/V 应写入物理 cache 的哪个 slot**。

仍取 block size 为 4。若一个新 token 应追加到逻辑位置 10，而这条序列的第三个逻辑 block 映射到物理 block 14：

```text
逻辑位置 10
→ 逻辑 block = 10 // 4 = 2
→ block 内偏移 = 10 % 4 = 2
→ 物理位置 = 物理 block 14 的第 2 个 slot
```

在 `layers/attention.py` 中，`store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)` 使用这个映射，把新得到的 K/V 写到已分配好的物理位置。

### 9.3 两者不要混淆

```text
block_table  ：一条序列的所有逻辑 block → 物理 block；主要帮助读取历史 K/V。
slot_mapping ：本次新 token → 物理 cache 的具体 slot；主要帮助写入新 K/V。
```

### 第五段检查题

block size 为 4，某条序列的：

```text
block_table = [9, 3, 14]
```

现要把一个新 token 写到逻辑位置 10。

请写出三个答案：它属于第几个逻辑 block？block 内偏移是多少？最终写到哪个物理 block 的哪个 slot？

**我的回答：**

```text
不知道。
```

**逐步讲解：**

`block_size=4` 表示一个 block 能放 4 个 token。先把逻辑 token 位置按每4个一组排开：

```text
逻辑 block 0：token 0, 1, 2, 3
逻辑 block 1：token 4, 5, 6, 7
逻辑 block 2：token 8, 9, 10, 11
```

所以逻辑位置 10 落在逻辑 block 2，且是这个 block 内从0开始数的第2个 slot：

```text
逻辑 block 编号 = 10 // 4 = 2
block 内偏移    = 10 % 4 = 2
```

然后查目录 `block_table=[9,3,14]`：

```text
逻辑 block 0 → 物理 block 9
逻辑 block 1 → 物理 block 3
逻辑 block 2 → 物理 block 14
```

因此最终位置是：

```text
物理 block 14 的 slot 2
```

完整链路：

```text
逻辑 token 10
→ 逻辑 block 2，block 内 slot 2
→ 查 block_table[2] = 14
→ 写入物理 block 14 的 slot 2
```

### 最小复测

仍使用 `block_size=4`、`block_table=[9,3,14]`。逻辑 token 位置 5 应写到哪个物理 block 的哪个 slot？

**参考答案：**

```text
逻辑 block = 5 // 4 = 1
block 内 slot = 5 % 4 = 1
物理 block = block_table[1] = 3

最终写入物理 block 3 的 slot 1。
```

本节至此不再补充新的分页 KV Cache 概念。block 的分配、释放、共享和 Prefix Cache 属于后续 L6 推理优化；本课只要求能区分 `block_table` 的读取目录职责与 `slot_mapping` 的新 K/V 写入职责。
