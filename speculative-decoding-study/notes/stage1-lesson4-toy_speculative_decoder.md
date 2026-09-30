下一步的 Toy 模拟器已保存：

```
D:\speculative-decoding-study\notes\toy_speculative_decoder.py
```

它包含：

- Draft token 序列
- Target block 验证
- longest accepted prefix
- bonus token
- acceptance length 估算
- speedup 粗略估算

## 1. 当前示例的执行结果

代码中：

```
Draft：
[A, B, C, D, E]
```

Target 的 greedy 结果：

```
[A, B, X, Y, Z, Q]
```

逐位置比较：

```
A == A  接受
B == B  接受
C != X  拒绝
```

因此：

```
accepted = 2
next_token = X
emitted = [A, B, X]
```

`D` 和 `E` 不会被提交，因为它们依赖已经被拒绝的 `C`。

------

## 2. 模拟器中的核心函数

对应最简代码文件：

```text
toy_speculative_decoder.py:25-45
```

```python
accepted = 0
for index, draft_token in enumerate(draft_tokens):
    target_token = argmax(target_rows[index])
    if target_token != draft_token:
        break
    accepted += 1
next_token = argmax(target_rows[accepted])
emitted = draft_tokens[:accepted] + [next_token]
```

### `verify_greedy()`

对应当前工程中的：

```
verify_speculative()
```

执行：

```
Draft block
    ↓
逐位置比较 Target argmax
    ↓
遇到第一个不匹配就停止
    ↓
取该位置 Target token
    ↓
返回 accepted + bonus token
```

### `expected_acceptance_length()`

对应公式：

```
E[A] = α + α² + ... + αᴷ
```

例如：

```
alpha = 0.8
draft_length = 5
```

则：

```
E[A]
= 0.8 + 0.64 + 0.512 + 0.4096 + 0.32768
= 2.68928
```

有 bonus token 时，平均每轮输出：

```
1 + E[A] = 3.68928 tokens
```

------

## 3. 速度估算

示例参数：

```
普通 Target 单 token：10 ms
Draft block：2 ms
Target Verify：12 ms
cache 管理：1 ms
```

投机一轮成本：

```
2 + 12 + 1 = 15 ms
```

普通解码吞吐：

```
1 / 10 = 0.1 tokens/ms
```

投机解码吞吐：

```
3.68928 / 15 ≈ 0.246 tokens/ms
```

估算加速比：

```
0.246 / 0.1 ≈ 2.46x
```

这只是理论估算。真实工程还会受到以下因素影响：

- GPU kernel 效率
- KV cache 写入和回滚
- CPU/GPU 同步
- tokenizer 解码
- batch size
- 请求长度
- Draft 和 Target 的实际接受率

------

## 4. 建议你现在修改三个地方

### 练习一：全部接受

把 Target 前五行改成：

```
[A, B, C, D, E]
```

预期：

```
accepted = 5
emitted = [A, B, C, D, E, bonus]
```

### 练习二：第一个 token 就拒绝

把第一行 Target 最大概率 token 改成：

```
X
```

预期：

```
accepted = 0
emitted = [X]
```

### 练习三：比较不同 Draft 长度

分别设置：

```
K = 2
K = 4
K = 8
```

观察：

```
E[A]
estimated speedup
```

你会看到：

> Draft 长度增加不一定带来更高加速，因为后面位置的接受概率会逐渐下降，而 Draft/Verify 成本会增加。

完成这三个实验后，下一步进入阶段 1 的总结：推导为什么 speculative decoding 能减少 Target 调用次数，并把 Toy 模拟器映射到 DFlash 的真实代码。

## 关联工程的最简实现：Toy 代码

本节的可运行最小实现位于：

```text
toy_speculative_decoder.py:25-45
```

```python
def verify_greedy(draft_tokens, target_rows):
    accepted = 0
    for index, draft_token in enumerate(draft_tokens):
        target_token = argmax(target_rows[index])
        if target_token != draft_token:
            break
        accepted += 1

    next_token = argmax(target_rows[accepted])
    emitted = draft_tokens[:accepted] + [next_token]
    return accepted, next_token, emitted
```

这段代码是 `llama.cpp` 的 `sample_and_accept_n()` 和当前 Qwen 工程 `verify_speculative()` 的 Python 最小等价物。
