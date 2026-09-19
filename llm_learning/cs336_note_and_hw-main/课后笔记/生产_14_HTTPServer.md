# 生产_14：HTTP Server / OpenAI API 原理与实现

> P2 了解专题 4/4（最后一课）| 源码参考：HLIEvLLM-1.4.0rc0 `vllm/entrypoints/`

---

## 一、总览

vLLM 提供两种使用方式：

| 方式 | 入口 | 适用场景 |
|------|------|----------|
| **在线服务** `vllm serve` | `vllm/entrypoints/openai/api_server.py` | OpenAI 兼容 REST API + SSE 流式 |
| **离线推理** `LLM(...)` | `vllm/entrypoints/llm.py` | 批量处理、评测、脚本 |

本笔记聚焦在线服务的完整架构。

---

## 二、目录结构

```
vllm/entrypoints/
├── launcher.py                  # Uvicorn HTTP 服务层 + 引擎看门狗
├── llm.py                       # 离线推理 LLM 类 (~1900 行)
├── openai/
│   ├── api_server.py            # 核心：FastAPI 构建、路由注册、中间件
│   ├── server_utils.py          # ASGI 中间件 (Auth/X-Request-Id)、异常处理器
│   ├── chat_completion/         # /v1/chat/completions 端点 + 流式处理
│   ├── completion/              # /v1/completions 端点 + 流式处理
│   ├── engine/protocol.py       # ErrorResponse/UsageInfo/DeltaMessage 等协议类型
│   ├── models/                  # /v1/models 端点
│   ├── responses/               # OpenAI Responses API (新版)
│   ├── realtime/                # WebSocket 实时推理
│   ├── speech_to_text/          # 转录/翻译
│   └── pooling/                 # /v1/embeddings 等池化端点
├── serve/                       # 内部管理路由
│   ├── instrumentator/          # /health, /metrics (Prometheus), /version
│   ├── lora/                    # LoRA adapter 管理
│   ├── tokenize/                # 分词/去分词
│   ├── render/                  # 聊天模板渲染
│   ├── elastic_ep/              # 弹性端点扩展 (ScalingMiddleware)
│   ├── sleep/                   # 引擎休眠/唤醒
│   ├── profile/                 # 性能分析
│   ├── cache/                   # 前缀缓存重置
│   └── disagg/                  # 分离式 prefill-decode 协议
├── sagemaker/                   # AWS Sagemaker 兼容端点
└── chat_utils.py                # 聊天模板加载
```

---

## 三、服务器启动流程

### 3.1 从 CLI 到就绪状态

```
vllm serve --model Qwen/Qwen3-7B --port 8000
  │
  ├─ 1. cli_env_setup() → 设置 multiprocessing 为 'spawn'
  ├─ 2. setup_server(args) → 绑定 TCP socket（在引擎启动前）
  ├─ 3. decorate_logs("APIServer") → stdout/stderr 加进程前缀
  └─ 4. uvloop.run(run_server(args))
       │
       ├─ build_async_engine_client(args)
       │   └─ AsyncLLM.from_vllm_config(vllm_config)
       │       → 创建 V1 异步引擎 (独立进程)
       │
       ├─ engine_client.get_supported_tasks()
       │   → 查询模型能力: ("generate", "embed", ...)
       │
       ├─ build_app(args, supported_tasks, model_config)
       │   ├─ 创建 FastAPI app
       │   ├─ 注册管理路由 (health/metrics/version/lora/...)
       │   ├─ 注册 OpenAI 路由 (chat/completions/embeddings/models)
       │   ├─ 注册中间件 (CORS → Auth → X-Request-Id → Scaling)
       │   └─ 注册异常处理器
       │
       ├─ init_app_state(engine_client, state, args)
       │   └─ app.state.engine_client = engine_client
       │   └─ app.state.openai_serving_chat = OpenAIServingChat(...)
       │   └─ ...
       │
       └─ serve_http(app, sock, **uvicorn_kwargs)
           └─ uvicorn.Server(uvicorn.Config(app)).serve(sockets=[sock])
```

### 3.2 引擎生命周期

[`vllm/entrypoints/openai/api_server.py#L77-L105`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/openai/api_server.py#L77-L105)

```python
@asynccontextmanager
async def build_async_engine_client(args):
    engine_args = AsyncEngineArgs.from_cli_args(args)
    async with build_async_engine_client_from_engine_args(engine_args) as engine:
        yield engine
    # 退出时自动 shutdown

async def build_async_engine_client_from_engine_args(engine_args):
    vllm_config = engine_args.create_engine_config()
    async_llm = AsyncLLM.from_vllm_config(vllm_config=vllm_config)
    yield async_llm
    # finally: async_llm.shutdown()
```

**关键设计**：
- 引擎是独立进程（`AsyncLLM` 内部管理推理核心进程）
- API Server 通过异步生成器与引擎通信
- `async with` 确保退出时正确清理

### 3.3 引擎健康监控

[`vllm/entrypoints/launcher.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/launcher.py)

```python
async def watchdog_loop(engine_client, server):
    while not server.should_exit:
        await asyncio.sleep(5)
        if engine_client.errored and not engine_client.is_running:
            if envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH:
                logger.error("Engine dead, keeping server alive")
            else:
                logger.error("Engine dead, shutting down server")
                server.should_exit = True
```

---

## 四、App State — 共享对象依赖图

[`vllm/entrypoints/openai/api_server.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/openai/api_server.py)

```python
app.state
  ├── engine_client: AsyncLLM          # 共享的异步引擎（所有请求共用）
  │
  ├── openai_serving_models            # 模型列表 + LoRA 查找
  ├── openai_serving_render            # 聊天模板渲染 → EngineInputs
  ├── openai_serving_tokenization      # 分词服务
  ├── openai_serving_chat             # 聊天补全处理器
  ├── openai_serving_completion       # 文本补全处理器
  ├── serving_embedding               # 嵌入处理器
  │
  ├── server_load_metrics: int        # 原子计数器：并发请求数
  └── enable_server_load_tracking: bool
```

所有处理器共享同一个 `engine_client`。每个请求到来时，路由处理器从 `app.state` 获取对应的 handler → 调用 handler 的方法 → handler 内部调用 `engine_client.generate()`。

---

## 五、中间件管道

[`vllm/entrypoints/openai/server_utils.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/openai/server_utils.py)

请求到达时的执行顺序（从外到内）：

```
┌─────────────────────────────────────────┐
│ [1] CORSMiddleware                      │ ← 最外层：跨域头
├─────────────────────────────────────────┤
│ [2] AuthenticationMiddleware (可选)      │ ← Bearer token 验证 (/v1/* 路径)
├─────────────────────────────────────────┤
│ [3] XRequestIdMiddleware (可选)         │ ← X-Request-Id 注入
├─────────────────────────────────────────┤
│ [4] ScalingMiddleware                   │ ← 弹性扩展：扩展时返回 503
├─────────────────────────────────────────┤
│ [5] 用户自定义中间件                      │ ← args.middleware 动态加载
├─────────────────────────────────────────┤
│ [6] FastAPI 路由匹配 + Handler           │
└─────────────────────────────────────────┘
```

### 5.1 AuthenticationMiddleware

```python
class AuthenticationMiddleware:
    """纯 ASGI 中间件。仅拦截 /v1 开头的路径。"""
    def __init__(self, app, tokens):
        self.api_tokens = [hashlib.sha256(t.encode()).digest() for t in tokens]

    def verify_token(self, headers):
        value = headers.get("Authorization")
        scheme, _, param = value.partition(" ")
        if scheme.lower() != "bearer":
            return False
        param_hash = hashlib.sha256(param.encode()).digest()
        # 常量时间比较，防止时序攻击
        return any(secrets.compare_digest(param_hash, th) for th in self.api_tokens)
```

### 5.2 XRequestIdMiddleware

```python
class XRequestIdMiddleware:
    """如果请求头中没有 X-Request-Id，自动生成 uuid4 注入响应头"""
    async def send_with_request_id(self, message):
        if message["type"] == "http.response.start":
            request_id = request_headers.get("X-Request-Id", uuid.uuid4().hex)
            response_headers.append("X-Request-Id", request_id)
```

### 5.3 ScalingMiddleware

弹性端点扩展专用：当模型正在扩展时，所有请求立即返回 `503 Service Unavailable`。

---

## 六、请求生命周期：`/v1/chat/completions`

### 6.1 完整链路

```
HTTP POST /v1/chat/completions
  Body: {"model": "Qwen3-7B", "messages": [...], "stream": true}
  │
  ▼
[中间件管道] CORS → Auth → X-Request-Id → Scaling
  │
  ▼
[FastAPI 路由]
  ● validate_json_request: 检查 Content-Type: application/json
  ● Pydantic: JSON → ChatCompletionRequest
  ● with_cancellation: 监听客户端断开
  ● load_aware_call: server_load_metrics++
  │
  ▼
[create_chat_completion(request, raw_request)]
  │
  ├─ 1. OpenAIServingChat._create_chat_completion()
  │     ├─ render_chat_request(request)
  │     │   → 聊天模板 → tokenize → EngineInputs
  │     ├─ request.to_sampling_params(max_tokens)
  │     │   → temperature, top_p, max_tokens, ...
  │     └─ engine_client.generate(engine_input, sampling_params, request_id)
  │         → AsyncGenerator[RequestOutput]
  │
  ├─ 2. 如果 stream=True:
  │     chat_completion_stream_generator()
  │     → AsyncGenerator[str]  (SSE 格式)
  │     → StreamingResponse(media_type="text/event-stream")
  │
  └─ 3. 如果 stream=False:
        chat_completion_full_generator()
        → ChatCompletionResponse (Pydantic 模型)
        → JSONResponse
```

### 6.2 流式输出 (SSE) 格式

每个 token 作为一个 SSE 事件发送：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你好"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"！"},"finish_reason":"stop"}]}

data: [DONE]
```

**工具调用场景**：tool parser 在流式输出中解析 tool call JSON，增量发送 `delta.tool_calls`：

```
data: {..., "delta":{"tool_calls":[{"index":0,"id":"call_xxx","function":{"name":"get_weather","arguments":""}}]}}

data: {..., "delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}

data: {..., "delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"北京\"}"}}]}}
```

### 6.3 取消处理

[`vllm/entrypoints/utils.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/utils.py)

```python
@with_cancellation
async def create_chat_completion(request, raw_request):
    ...
```

`with_cancellation` 装饰器的逻辑：
1. 竞速：处理器 coroutine vs `listen_for_disconnect()` (监听 `http.disconnect`)
2. 客户端断开 → 取消处理器任务 → `CancelledError`
3. 非流式：返回 499 错误
4. 流式：生成器被中断，SSE 流自然关闭

---

## 七、异常处理体系

[`vllm/entrypoints/openai/server_utils.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/openai/server_utils.py)

| 异常类型 | 处理器 | HTTP 状态码 |
|----------|--------|------------|
| `EngineGenerateError` | `engine_error_handler` | 500 + 引擎健康检查 |
| `EngineDeadError` | `engine_error_handler` | 500 → 可能关闭服务器 |
| `GenerationError` | `generation_error_handler` | 500（不记录堆栈） |
| `HTTPException` | `http_exception_handler` | 原状转发 |
| `RequestValidationError` | `validation_exception_handler` | 400 + OpenAI 格式错误 |
| `Exception` | `exception_handler` | 500（通用兜底） |

**`create_error_response()` 辅助函数**的智能分类：

```python
def create_error_response(status_code, message, exc_type):
    if isinstance(exc, VLLMValidationError):    → 400 BadRequestError
    elif isinstance(exc, VLLMNotFoundError):     → 404 NotFoundError
    elif isinstance(exc, ValueError/TypeError):  → 400 BadRequestError
    elif isinstance(exc, NotImplementedError):   → 501 NotImplementedError
    else:                                         → 500 InternalServerError
```

**流式处理中错误的特殊处理**：
- HTTP 200 已发送，无法通过状态码报错
- 改为在 SSE 流中发送错误块：`data: {"error": {"message":"...", "code":500}}\n\n`
- 然后发送 `data: [DONE]\n\n` 关闭流

---

## 八、并发模型

### 8.1 单线程事件循环

整个服务器运行在 **单个 uvloop 事件循环** 上：
- 所有请求并发是协作式的（asyncio）
- 没有多线程竞争（除了后台任务线程池用于 grammar 编译、bitmask 填充等 CPU 密集操作）
- 引擎（`AsyncLLM`）在独立进程中运行，API Server 通过异步生成器与之通信

### 8.2 引擎共享

```python
# 所有路由处理器通过 FastAPI 依赖注入获取共享引擎
def chat(request: Request) -> OpenAIServingChat:
    return request.app.state.openai_serving_chat
    # 内部 self.engine_client = 同一个 AsyncLLM 实例
```

所有并发请求共享同一个引擎，调度器自动 batching 处理。

### 8.3 负载感知

```python
@load_aware_call
async def create_chat_completion(...):
    # 进入时 server_load_metrics++
    # 退出时 server_load_metrics--
```

`server_load_metrics` 是原子计数器，用于 `/metrics` 端点暴露并发请求数。

---

## 九、OpenAI API 兼容性

| 端点 | 路径 | 状态 |
|------|------|------|
| Chat Completions | `POST /v1/chat/completions` | 完全兼容 |
| Completions | `POST /v1/completions` | 完全兼容（Legacy） |
| Embeddings | `POST /v1/embeddings` | 完全兼容 |
| Models | `GET /v1/models` | 完全兼容 |
| Responses | `POST /v1/responses` | 新版 API，实验性 |
| Realtime | WebSocket `/v1/realtime` | WebSocket 实时推理 |
| Speech-to-Text | `POST /v1/audio/transcriptions` | 转录/翻译 |

**关键兼容性特性**：
- `tool_choice` + `tools`：自动 function calling，通过 `ToolParserManager` 提取
- `response_format: {"type": "json_object"}` / `json_schema`：→ Structured Outputs
- `stream_options.include_usage`：流式输出末尾附 usage
- `logprobs` / `top_logprobs`：内部 Logprob → OpenAI `ChatCompletionLogProbs`
- Pydantic `extra="allow"`：未知字段自动忽略，保证前向兼容

---

## 十、离线推理：LLM 类

[`vllm/entrypoints/llm.py`](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/entrypoints/llm.py)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-7B")

# 基础推理
outputs = llm.generate(["Hello!", "What is AI?"], SamplingParams(max_tokens=100))

# 聊天
outputs = llm.chat([
    [{"role": "user", "content": "你是谁？"}]
])

# 嵌入
embeddings = llm.embed(["text1", "text2"])

# Beam Search
outputs = llm.beam_search(["prompt"], BeamSearchParams(beam_width=5, max_tokens=50))
```

**内部实现**：
- 同步风格 API，内部包装 `LLMEngine`（V1 引擎）
- `generate()` → `engine.add_request()` → `engine.step()` 循环 → 收集完成结果
- 通过 `tqdm` 显示进度条
- 输出按 `request_id` 排序

---

## 十一、nano-vllm 对比

nano-vllm **没有 HTTP Server 支持**。它只有一个简单的 Python 脚本调用示例：

```python
# nano-vllm 的使用方式
llm = LLM("Qwen/Qwen3-0.6B")
output = llm.generate("Hello!")
```

| 维度 | nano-vllm | HLIEvLLM |
|------|-----------|----------|
| HTTP Server | 无 | FastAPI + Uvicorn + uvloop |
| OpenAI 兼容 | 无 | `/v1/chat/completions` 等 6+ 端点 |
| 流式输出 (SSE) | 无 | 完整支持 |
| 中间件 | 无 | CORS/Auth/X-Request-Id/Scaling |
| 异常处理 | 无 | 7 种异常处理器 |
| 并发 | 单请求 | asyncio 协作式并发 + 共享引擎 |
| 监控 | 无 | Prometheus /health /metrics |
| LoRA 管理 | 无 | `/v1/loras` REST API |
| 工具调用 | 无 | ToolParser 自动解析 |
| 代码量 | 0 行 | ~10000+ 行 |

---

## 十二、总结

### 核心设计模式

| 设计 | 应用 | 意义 |
|------|------|------|
| **ASGI + FastAPI** | 异步 HTTP 框架 | 原生支持 asyncio + SSE 流式 |
| **单例引擎共享** | `app.state.engine_client` | 所有请求共享同一引擎，调度器自动 batching |
| **生成器链式流式** | `Engine async gen → Serving async gen → StreamingResponse` | 无缓冲，逐 token 直推 |
| **ASGI 中间件管道** | CORS → Auth → X-Request-Id → Scaling | 关注点分离，插件化扩展 |
| **Pydantic 协议层** | 所有请求/响应都是 Pydantic 模型 | 自动验证 + 序列化 + OpenAI 格式兼容 |
| **请求取消** | `with_cancellation` + `listen_for_disconnect` | 客户端断开时立即释放引擎资源 |
| **引擎健康监控** | `watchdog_loop` 每 5 秒检查 | 引擎崩溃时优雅关闭 |

### 与你工作的关联

1. **API 兼容性测试**：OpenAI 兼容意味着可以直接用 OpenAI SDK 测试 Houmo 的服务
2. **流式性能**：SSE 逐 token 推送对延迟要求高，NPU 场景下的 token 生成延迟直接影响用户体验
3. **自定义中间件**：如果需要接入公司的认证/限流/审计系统，可以通过 `args.middleware` 注入

---

## 思考题

1. vLLM 为什么使用 `asyncio`（单线程事件循环）而不是传统的多线程模型？这与引擎的架构有什么关系？

2. 流式输出时 HTTP 200 已经发送，如果后续生成出错（如 KV Cache OOM），vLLM 如何处理？

3. `app.state` 的设计有什么优点？为什么不用全局变量或者每个请求 new 一个新的 engine？

4. OpenAI Chat Completion API 的 `tool_choice` 参数如何影响 vLLM 内部的 `SamplingParams` 和 `StructuredOutputsParams`？

5. nano-vllm 如果要实现 `vllm serve`，最核心的改动是什么？为什么不是简单套一个 Flask/FastAPI？

---

> 🎉 **恭喜！10 个生产功能专题全部完成！**  
> P0 (量化/GPTQ/SmoothQuant/AWQ/KV Cache量化/Pipeline/TP-NCCL) + P1 (投机解码/KV Cache Offloading/多硬件后端) + P2 (LoRA/多模态/Structured Outputs/HTTP Server)
