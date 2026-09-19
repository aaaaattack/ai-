# 生产_13：Structured Outputs 原理与实现

> P2 了解专题 3/4 | 源码参考：HLIEvLLM-1.4.0rc0 `vllm/v1/structured_output/`、`vllm/config/structured_outputs.py`

---

## 一、什么是 Structured Outputs

### 1.1 问题场景

LLM 直接采样可能产生"格式上不合法"的输出：

```
用户要求："以 JSON 格式输出电影信息"
模型输出：{"title": "星际穿越", "year": 2014, "director": "诺兰"  # 缺少 }
```

或更致命的：
```
用户要求："输出一个合法 JSON 对象"
模型输出：I think the answer is {"name": "Alice"} maybe   # 前后有废话
```

**Structured Outputs (约束解码/Guided Decoding)** 在采样时保证输出严格遵守指定的格式：

- **JSON Schema**：输出是合法的 JSON，且字段类型、必选字段等完全满足 schema
- **Regex**：输出匹配指定的正则表达式
- **Choice**：输出必须是给定选项之一
- **EBNF Grammar**：输出符合自定义的上下文无关文法
- **Structural Tag**：输出遵循结构化标签格式（如 `{ "reasoning": "...", "answer": "..." }`）

### 1.2 核心原理：FSM + Logits Masking

```
每个 token 生成前：
  1. FSM (有限状态机) 记录当前处于语法规则的哪个位置
  2. 从 FSM 获取当前状态下所有合法的 next token
  3. 构建 bitmask: 合法 token → 1，非法 token → 0
  4. 将非法 token 的 logits 设为 -inf
  5. 采样 → 永远只输出合法 token
  6. 推进 FSM 到新状态
```

---

## 二、整体架构

```
用户请求 (API)
    │
    ▼
┌────────────────────────────────────────────────────┐
│  StructuredOutputsParams                           │
│  json | regex | choice | grammar | json_object     │
│  structural_tag                                     │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  _validate_structured_outputs()                    │
│  - 后端选择: auto → xgrammar → guidance → outlines │
│  - 约束合法性预检                                     │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  StructuredOutputManager (Engine Core)              │
│  - grammar_init(): 异步编译 Grammar + FSM          │
│  - grammar_bitmask(): 为 batch 中所有约束请求       │
│    生成 bitmask tensor                             │
│  - should_advance(): 决定是否推进 FSM               │
└──────────────────┬─────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ XGrammar │ │ Guidance │ │ Outlines │  ← 4 种后端，统一接口
│ (默认)   │ │ (llgid.) │ │ (regex)  │
└──────────┘ └──────────┘ └──────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│  StructuredOutputsWorker (GPU)                     │
│  Triton Kernel: bitmask → -inf logits              │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  Sampler → topk_topp_sampler → 采样 → 合法 token    │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  Scheduler.update_from_output()                    │
│  grammar.accept_tokens(new_token_ids) → 推进 FSM   │
└────────────────────────────────────────────────────┘
```

---

## 三、用户侧的约束指定

### 3.1 StructuredOutputsParams

[`vllm/sampling_params.py#L41-L57`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/sampling_params.py#L41-L57)

```python
class StructuredOutputsParams:
    json: str | dict | None = None         # JSON Schema 字符串
    regex: str | None = None               # 正则表达式
    choice: list[str] | None = None        # 多选一
    grammar: str | None = None             # EBNF 语法
    json_object: bool | None = None        # 任意合法 JSON 对象
    structural_tag: str | None = None      # 结构化标签 (reasoning)
    disable_any_whitespace: bool = False   # JSON 禁用空格
    disable_additional_properties: bool    # 禁用额外属性
    whitespace_pattern: str | None = None  # 自定义空白符正则

    _backend: str | None           # 内部: 选择的后端 ("xgrammar"/"guidance"/...)
    _backend_was_auto: bool        # 内部: 是否自动选择
```

**互斥性**：`json`, `regex`, `choice`, `grammar`, `json_object`, `structural_tag` 六者互斥，每次只能指定一种。

### 3.2 使用示例

```python
# JSON Schema
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}'
    )
)

# Regex
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(regex=r'\d{3}-\d{4}-\d{4}')
)

# Choice
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(choice=["positive", "negative", "neutral"])
)
```

### 3.3 OpenAI API 兼容

[`vllm/entrypoints/openai/chat_completion/protocol.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/openai/chat_completion/protocol.py)

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "movie",
      "schema": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "year": {"type": "integer"}
        },
        "required": ["title", "year"]
      }
    }
  }
}
```

或 `"type": "json_object"` 强制输出任意合法 JSON。

---

## 四、配置系统：StructuredOutputsConfig

[`vllm/config/structured_outputs.py#L17-L43`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/config/structured_outputs.py#L17-L43)

```python
class StructuredOutputsConfig:
    backend: str = "auto"              # "auto"/"xgrammar"/"guidance"/"outlines"/"lm-format-enforcer"
    disable_any_whitespace: bool       # JSON 输出压缩 (无空格)
    disable_additional_properties: bool  # Guidance 后端禁用 additionalProperties
    reasoning_parser: str = ""         # Reasoning 模型解析器
    enable_in_reasoning: bool = False  # reasoning 阶段是否启用约束
```

---

## 五、四种后端统一接口

### 5.1 抽象基类

[`vllm/v1/structured_output/backend_types.py#L31-L96`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/structured_output/backend_types.py#L31-L96)

```python
class StructuredOutputGrammar(ABC):
    """请求级别的结构化输出后端"""
    def accept_tokens(self, request_id, tokens) -> bool: ...
        """接受 token 并推进 FSM，返回是否接受"""

    def validate_tokens(self, tokens) -> list[int]: ...
        """验证 token 序列（不回退 FSM），返回合法前缀"""

    def rollback(self, num_tokens) -> None: ...
        """回退 FSM 指定步数"""

    def fill_bitmask(self, bitmask, batch_index) -> None: ...
        """填充当前状态下合法 token 的 bitmask"""

    def is_terminated(self) -> bool: ...
        """FSM 是否到达终止状态"""

    def reset(self): ...
        """重置 FSM"""

class StructuredOutputBackend(ABC):
    """引擎级别的后端"""
    def compile_grammar(self, request_type, grammar_spec) -> StructuredOutputGrammar: ...
        """编译语法规格 → Grammar + FSM"""

    def allocate_token_bitmask(self, max_num_seqs) -> torch.Tensor: ...
        """分配 bitmask tensor"""

    def destroy(self): ...
        """后端销毁"""
```

### 5.2 四种后端对比

| 后端 | 库 | FSM 引擎 | 主要特点 |
|------|-----|----------|----------|
| **XGrammar** (默认) | `xgrammar` (MLC-AI) | 位打包 DFA | 最快、支持最全面的 JSON Schema、内置 LRU 缓存 (512MB) |
| **Guidance** | `llguidance` | 字节级解析器 | 支持复杂的 JSON Schema 特性 (`patternProperties` 等) |
| **Outlines** | `outlines_core` | regex-automata | 纯正则引擎、tokenizer 无关 |
| **LM Format Enforcer** | `lmformatenforcer` | 字符级解析器 | 最轻量、验证而非生成 |

**auto 模式选择逻辑**：
1. 先尝试 XGrammar → 检查 JSON Schema 是否有不兼容特性
2. 若不兼容且有 `patternProperties` → Guidance
3. 若不兼容且 tokenizer 是 Mistral (Tekken) → Outlines
4. 否则 → Guidance (fallback)

XGrammar **不兼容的 JSON Schema 特性** (`has_xgrammar_unsupported_json_features()`):
- `multipleOf`（数值）
- `uniqueItems`, `contains`, `minContains`, `maxContains`（数组）
- 非标准 `format`（字符串，支持 email/date/time/datetime/uri 等标准格式）
- `patternProperties`, `propertyNames`（对象）

---

## 六、XGrammar 后端详解

### 6.1 初始化

[`vllm/v1/structured_output/backend_xgrammar.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/structured_output/backend_xgrammar.py)

```python
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self):
        # 1. 构建 tokenizer 信息
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size
        )

        # 2. 创建 Grammar 编译器 (支持多线程 + LRU 缓存)
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            max_cache_size_bytes=512 * 1024 * 1024,  # 512MB
        )
```

### 6.2 Grammar 编译

根据约束类型分发：

```python
def compile_grammar(self, request_type, grammar_spec):
    if request_type == JSON:
        compiled = self.compiler.compile_json_schema(
            grammar_spec,
            any_whitespace=not disable_any_whitespace
        )
    elif request_type == JSON_OBJECT:
        compiled = self.compiler.compile_json_schema('{"type": "object"}')
    elif request_type == GRAMMAR:
        compiled = self.compiler.compile_grammar(grammar_spec)
    elif request_type == REGEX:
        compiled = self.compiler.compile_regex(grammar_spec)
    elif request_type == STRUCTURAL_TAG:
        compiled = self.compiler.compile_structural_tag(
            grammar_spec, structural_tag_format_new=True
        )

    matcher = xgr.GrammarMatcher(compiled, max_rollback_tokens=...)
    return XgrammarGrammar(matcher, compiled, self.vocab_size)
```

### 6.3 FSM 核心操作

```python
class XgrammarGrammar(StructuredOutputGrammar):
    def accept_tokens(self, request_id, tokens) -> bool:
        for token in tokens:
            if not self.matcher.accept_token(token):
                return False
        return self.matcher.is_terminated()

    def validate_tokens(self, tokens) -> list[int]:
        accepted = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted.append(token)
            else:
                break
        self.matcher.rollback(len(accepted))  # 回退到原始状态
        return accepted

    def rollback(self, num_tokens):
        self.matcher.rollback(num_tokens)

    def fill_bitmask(self, bitmask, batch_index):
        self.matcher.fill_next_token_bitmask(bitmask, batch_index)
        # bitmask[i] 的每一位对应 vocab 中的一个 token
        # 1 → 允许, 0 → 禁止

    def is_terminated(self):
        return self.matcher.is_terminated()

    def reset(self):
        self.matcher.reset()
```

---

## 七、StructuredOutputManager — 引擎级管理

[`vllm/v1/structured_output/__init__.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/structured_output/__init__.py)

### 7.1 异步 Grammar 编译

```python
def grammar_init(self, request):
    # 首次请求时初始化后端
    if self.backend is None:
        backend = request.sampling_params.structured_outputs._backend
        if backend == "xgrammar":
            self.backend = XgrammarBackend(vllm_config, tokenizer, vocab_size)
        # ...

    # 异步提交编译任务到 ThreadPoolExecutor
    if self._use_async_grammar_compilation:
        grammar = self.executor.submit(self._create_grammar, request)
    else:
        grammar = self._create_grammar(request)

    request.structured_output_request.grammar = grammar  # Future 或编译结果
```

**设计要点**：
- Grammar 编译在后台线程进行，不阻塞调度循环
- 编译完成前请求保持 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 状态
- `external_launcher` 模式下禁用异步（避免多 TP rank 间步调不一致）

### 7.2 批量 Bitmask 生成

```python
def grammar_bitmask(self, requests, structured_output_request_ids, scheduled_spec_decode_tokens):
    """为 batch 中所有结构化输出请求生成 bitmask"""

    # 1. 分配 bitmask tensor: (max_batch_size * (1 + spec_tokens), ceil(vocab_size/32))
    if self._grammar_bitmask is None:
        self._grammar_bitmask = self.backend.allocate_token_bitmask(
            max_batch_size * (1 + max_num_spec_tokens)
        )

    # 2. 对每个结构化输出请求填充 bitmask
    for req_id in structured_output_request_ids:
        grammar = requests[req_id].structured_output_request.grammar
        apply_bitmask = self.should_fill_bitmask(request)

        # 推测解码：为每个 bonus token 位置单独填充 bitmask
        req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
        for token in itertools.chain(req_tokens, (-1,)):
            self._fill_bitmasks(((grammar, idx, apply_bitmask),))
            if apply_bitmask and not grammar.is_terminated():
                grammar.accept_tokens(req_id, [token])  # 推进 FSM
            idx += 1
        grammar.rollback(state_advancements)  # 回退到填充前的状态

    # 3. 返回 numpy 数组 (比 torch.Tensor 序列化更高效)
    return bitmask_tensor.numpy()
```

**加速策略**：当 batch 中结构化输出请求数 > 128 且无推测解码时，使用 `ThreadPoolExecutor`（最多 8 worker）并行填充 bitmask。

### 7.3 Reasoning/Thinking 模式支持

```python
def should_fill_bitmask(self, request):
    """决定是否对当前 step 应用 bitmask"""
    reasoner = self._get_reasoner(request)
    if reasoner is None:
        return True  # 无 reasoning → 始终应用

    if self.enable_in_reasoning:
        return True  # reasoning 阶段也应用

    # reasoning 结束后才开始应用 bitmask
    return request.structured_output_request.reasoning_ended

def should_advance(self, request):
    """决定是否在 sched 中推进 FSM"""
    # thinking 阶段不推进 FSM，只在 reasoning 结束后推进
    ...
```

---

## 八、GPU 上的 Bitmask 应用

[`vllm/v1/worker/gpu/structured_outputs.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/worker/gpu/structured_outputs.py)

### 8.1 完整流程

```python
class StructuredOutputsWorker:
    def apply_grammar_bitmask(self, logits, input_batch, grammar_req_ids, grammar_bitmask):
        # 1. 异步拷贝 bitmask 到 GPU (独立 CUDA stream)
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(grammar_bitmask)

        # 2. 构建 mapping: grammar_req_id → logits 行索引
        #    每个 grammar 请求可能有多行 logits (推测解码)
        mapping = []
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        # 3. 异步拷贝 mapping 到 GPU
        logits_indices = tensor(mapping, device="cuda")

        # 4. 同步 stream 后启动 Triton kernel
        current_stream.wait_stream(self.copy_stream)
        _apply_grammar_bitmask_kernel[grid](logits, logits_indices, bitmask, vocab_size)

        # 5. copy_stream 等待 kernel 完成
        self.copy_stream.wait_stream(current_stream)
```

### 8.2 Triton Kernel 核心逻辑

[`vllm/v1/worker/gpu/structured_outputs.py#L86-L115`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/worker/gpu/structured_outputs.py#L86-L115)

```python
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr, logits_stride,
    logits_indices_ptr,
    bitmask_ptr, bitmask_stride,
    vocab_size, BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)  # 哪个 grammar 请求
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)  # 对应的 logits 行

    # 从 bitmask 解包
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset)
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0

    # 将非法 token 的 logits 设为 -inf
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
```

**关键设计**：
- Bitmask 是位打包的 int32：vocab_size=128000 需要 4000 个 int32 = 16KB
- 每个 block 处理 8192 个 token，减少内存访问
- 通过 `logits_indices` 间接寻址，支持稀疏 grammar（非约束请求不受影响）

---

## 九、与推测解码的交互

Structured Outputs + Speculative Decoding 需要精细的交互逻辑：

### 9.1 Bitmask 分配

每个结构化输出请求在 bitmask 中分配 `(1 + num_speculative_tokens)` 行：
- 第 0 行：非推测 token / bonus token 的 bitmask
- 第 1~N 行：N 个推测位置的 bitmask

### 9.2 Draft Token 验证

[`scheduler.py#L1629-L1661`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/core/sched/scheduler.py)

```python
# 用 grammar.validate_tokens() 过滤不符合语法的 draft tokens
validated_tokens = grammar.validate_tokens(draft_tokens)
# 只保留合法前缀，后面的 token 被丢弃
# 被过滤的 token 替换为 -1 (padding)
```

### 9.3 FSM 回退

推测解码可能拒绝部分 draft token→需要回退 FSM→`max_rollback_tokens` 预设足够大的回退空间。

---

## 十、nano-vllm 对比

nano-vllm **完全没有 Structured Outputs 支持**。

| 维度 | nano-vllm | HLIEvLLM |
|------|-----------|----------|
| 约束解码 | 不支持 | 6 种约束类型，4 种后端 |
| FSM | 无 | GrammarMatcher 位打包 DFA |
| Bitmask | 无 | Triton kernel + -inf logits |
| 多请求并发 | 不适用 | 同一 batch 多 grammar 独立 FSM |
| 推测解码兼容 | 不适用 | rollback + 多位置 bitmask + draft 验证 |
| Reasoning | 不适用 | thinking 阶段延迟 FSM 推进 |

---

## 十一、总结

### 核心设计模式

| 设计 | 应用 | 意义 |
|------|------|------|
| **FSM + Logits Masking** | grammar bitmask → -inf logits | 采样前一次性约束，不侵入采样器 |
| **策略模式** | 4 种后端统一 `StructuredOutputGrammar` 接口 | 不同场景选最优后端 |
| **异步编译** | ThreadPoolExecutor 后台编译 grammar | 不阻塞调度循环 |
| **CPU 编译 + GPU 执行** | Grammar/FSM 在 CPU，bitmask 在 GPU Triton | 利用 CPU 做 DFA，GPU 只做 -inf masking |
| **两级验证** | bitmask 理论保证 + `accept_tokens` 每次实际验证 | 防御性编程，保底 |
| **Per-request FSM** | 每个请求独立 FSM 实例 | 完全隔离，同一 batch 不同约束 |

### 与你工作的关联

1. **API 兼容性**：如果 Houmo 提供 OpenAI 兼容 API，structured output 是客户高频需求
2. **FSM 性能**：`fill_bitmask` 在 CPU 上运行，不占 NPU 算力，但大 batch 下的 CPU 开销需要注意
3. **XGrammar 依赖**：XGrammar 当前仅支持 CUDA/ROCm/CPU Python 环境，在 NPU 上运行时需要注意依赖兼容性

---

## 思考题

1. 为什么 bitmask 在位域（int32 打包）而非直接用 bool tensor？这和 vocab_size=128000 下的内存开销有什么关系？

2. `accept_tokens()` 和 `fill_bitmask()` 的关系是什么？为什么 fill_bitmask 之后还需要 accept_tokens？

3. 推测解码中，为什么需要 `validate_tokens()`（不回退 FSM 的验证）而不是直接用 `accept_tokens()`？

4. 如果 JSON Schema 使用了 XGrammar 不支持的特性（如 `patternProperties`），vLLM 如何处理？

5. nano-vllm 如果要支持 structured output，最简单的实现方案是什么？（提示：不需要完整的 FSM 引擎）
