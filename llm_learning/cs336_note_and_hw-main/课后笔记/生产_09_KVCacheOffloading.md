# 生产功能 09：KV Cache Offloading

> 学习路线：P1 重点 — 显存管理高级手段，直接影响最大 context length
> 对应源码：HLIEvLLM `vllm/v1/kv_offload/` (新版) + `vllm/v1/simple_kv_offload/` (简化版) + `vllm/distributed/kv_transfer/` (分布式框架)

---

## 一、问题：GPU 显存是硬约束

### 1.1 回放已知知识

从第11课（PagedAttention）我们知道：

```
GPU 显存 = 模型权重 + KV Cache + 激活值 + CUDA 开销

KV Cache 大小 ≈ 2 × num_layers × num_kv_heads × head_dim × max_tokens × dtype
                × batch_size

例: Qwen3-8B, bs=32, 16K context, FP16:
    ≈ 2 × 32 × 8 × 128 × 16384 × 2 bytes × 32
    ≈ 68 GB ← 远超单张 H100 (80GB) 的可用空间！
```

### 1.2 Offloading 的核心思路

**不是所有 KV Cache block 都需要一直留在 GPU 显存里。**

```
Request A: 生成了 4096 个 token，卡在 waiting 队列等 scheduler 调度
  → KV Cache 在 GPU 上，但 90% 的时间不在被使用
  → 挪到 CPU 上暂存，腾出 GPU 显存给 active 的请求

Request B: 正在被 active decode，每次 forward 都需要访问完整的 KV Cache
  → 留在 GPU 上
```

**三级缓存架构：**

```
GPU HBM (80GB)     ← 最快，最贵，active KV Cache
    ↕ swap_blocks_batch (DMA, ~50GB/s)
CPU DRAM (512GB)   ← 中等，inactive KV Cache
    ↕ mmap / 文件系统 (SSD, ~7GB/s) 
SSD (2TB+)         ← 最慢，最便宜，归档 KV Cache (FlexGen 风格)
```

---

## 二、vLLM 的 KV Cache Offloading 架构

### 2.1 两条路径

HLIEvLLM 提供了两套 KV Offloading 实现，通过环境变量切换：

```
VLLM_USE_SIMPLE_KV_OFFLOAD=0 (默认)  → 新版 Native Offloading (kv_offload/)
VLLM_USE_SIMPLE_KV_OFFLOAD=1         → 简化版 Simple Offloading (simple_kv_offload/)
```

| | 新版本 (Native) | 简化版 (Simple) |
|---|---|---|
| **内存分配** | mmap 共享内存 (`/dev/shm`) | `torch.zeros` pinned memory |
| **传输机制** | `swap_blocks_batch` 自定义算子 | `cuMemcpyBatchAsync` |
| **缓存策略** | 可插拔 (LRU / ARC) | 固定 LRU (复用 GPU BlockPool) |
| **存储模式** | 多 worker 共享 mmap 区域 | Lazy / Eager 两种模式 |
| **适用场景** | 多 worker 进程共享 CPU 内存 | 简单快速、单 worker 场景 |

### 2.2 配置入口

[/vllm/config/cache.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/config/cache.py#L167-L176)

```python
# CacheConfig — KV Cache 卸载配置
kv_offloading_size: float | None = None
"""CPU 卸载缓冲区大小 (GiB)。TP > 1 时是所有 rank 的总和。
设为 None 表示不启用 KV offloading。"""

kv_offloading_backend: KVOffloadingBackend = "native"
"""卸载后端: 'native' (vLLM 原生) 或 'lmcache' (LMCache)"""

# 使用方式:
llm = LLM(
    model="Qwen3-8B",
    kv_offloading_size=10,         # 10GB CPU 缓冲区
    kv_offloading_backend="native" # 使用原生后端
)
```

### 2.3 整体架构

```
Scheduler 层:
  ┌─────────────────────────────────────────┐
  │ CPUOffloadingManager / SimpleCPUOffloadScheduler
  │  - lookup(key) → 块在 CPU 上吗？         │
  │  - prepare_load(keys) → 准备从 CPU 加载  │
  │  - prepare_store(keys) → 准备存到 CPU    │
  │  - complete_load/store → 标记完成         │
  │  - 内部: BlockPool + LRU/ARC 缓存策略     │
  └──────────────┬──────────────────────────┘
                 │ KVConnectorMetadata
                 ▼
Worker 层:
  ┌─────────────────────────────────────────┐
  │ SingleDirectionOffloadingHandler / DmaCopyBackend
  │  - GPU → CPU: 等 compute stream 完成后 DMA │
  │  - CPU → GPU: 异步 prefetch               │
  │  - swap_blocks_batch / cuMemcpyBatchAsync  │
  └──────────────┬──────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────┐
  │ SharedOffloadRegion (mmap) / torch.zeros │
  │  - CPU 端物理存储                          │
  └─────────────────────────────────────────┘
```

---

## 三、新版 Native Offloading 详解

### 3.1 核心抽象：`OffloadingManager`

[/vllm/v1/kv_offload/base.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/kv_offload/base.py)

```python
# OffloadKey — 块标识符: block_hash + group_idx
OffloadKey = NewType("OffloadKey", bytes)

def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    """将 block hash 和 KV cache group index 打包"""
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big"))


# OffloadingManager 提供 5 个原语:
class OffloadingManager(ABC):
    def lookup(key, req_context) -> bool | None:
        """检查块是否在 CPU 上且可读
        True  = 已存储，可读
        False = 不在 CPU
        None  = 正在写入 (caller 应该等会儿再试)
        """

    def prepare_load(keys, req_context) -> LoadStoreSpec:
        """准备加载: ref_cnt++，保护不被驱逐"""

    def touch(keys, req_context):
        """标记为最近使用（更新 LRU 位置）"""

    def complete_load(keys, req_context):
        """加载完成: ref_cnt--"""

    def prepare_store(keys, req_context) -> PrepareStoreOutput | None:
        """准备存储: 过滤低于阈值的块、驱逐 LRU 块、分配 CPU 块"""

    def complete_store(keys, req_context, success=True):
        """存储完成: 标记块可用 或 回滚"""
```

### 3.2 CPUOffloadingManager — 可插拔缓存策略

[/vllm/v1/kv_offload/cpu/manager.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/kv_offload/cpu/manager.py)

```python
class CPUOffloadingManager(OffloadingManager):
    """
    带可插拔 CachePolicy (LRU 或 ARC) 的 OffloadingManager。

    职责:
    - 引用计数管理
    - Block pool 管理 (allocate/free)
    - 存储阈值控制 (store_threshold)
    - 驱逐决策 (委托给 CachePolicy)
    """

    def __init__(self, num_blocks, cache_policy="lru",
                 store_threshold=1, max_tracker_size=64000):
        # 块池管理
        self._num_blocks = num_blocks
        self._num_allocated_blocks = 0
        self._free_list = []          # 空闲块 ID 列表

        # 可插拔缓存策略
        policy_cls = _CACHE_POLICIES[cache_policy]  # "lru" 或 "arc"
        self._policy = policy_cls(cache_capacity=num_blocks)

        # 访问频率追踪 (用于 store_threshold)
        self.counts = OrderedDict()    # key → 命中次数
        self.store_threshold = store_threshold   # 达到多少次才允许卸载
        self.max_tracker_size = max_tracker_size # counts 最大条目数

    # === 完整的 prepare_store 流程 ===
    def prepare_store(self, keys, req_context) -> PrepareStoreOutput | None:
        # 1. 过滤: 只有命中次数 >= store_threshold 的块才卸载
        if self.counts is not None:
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]

        # 2. 去重: 已经在 CPU 上的跳过
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        # 3. 驱逐: 如果空间不够，LRU 驱逐旧块
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()
        if num_blocks_to_evict > 0:
            protected = set(keys)  # 本次要存的块不能被驱逐
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None  # 驱逐不了 → 放弃本次 store
            for key, block in evicted:
                self._free_block(block)

        # 4. 分配 CPU 块 + 插入缓存
        blocks = self._allocate_blocks(keys_to_store)
        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=self._get_load_store_spec(keys_to_store, blocks),
            evicted_keys=to_evict,
        )
```

**`store_threshold` 的设计意义：**

```
store_threshold=1:  所有块都卸载 (激进，带宽压力大)
store_threshold=2:  只有被 lookup() 2 次以上的块才卸载
                     → 过滤掉"再也不被访问"的块 (如一次性 prompt)
store_threshold=3:  更保守，只卸载频繁共享的前缀块
```

### 3.3 LRU 驱逐策略

[/vllm/v1/kv_offload/cpu/policies/lru.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/kv_offload/cpu/policies/lru.py)

```python
class LRUCachePolicy(CachePolicy):
    """LRU 缓存策略 — 单个 OrderedDict"""

    def __init__(self, cache_capacity):
        self.blocks = OrderedDict()  # key → BlockStatus

    def get(self, key) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key, block):
        self.blocks[key] = block

    def touch(self, keys):
        """更新最近访问时间 — 移到 OrderedDict 末尾"""
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(self, n, protected):
        """驱逐 n 个 ref_cnt=0 且不受保护的块"""
        candidates = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
                if len(candidates) == n:
                    break
        if len(candidates) < n:
            return None  # 驱逐不够 → 放弃
        for key, _ in candidates:
            del self.blocks[key]
        return candidates
```

### 3.4 mmap 共享内存

[/vllm/v1/kv_offload/cpu/shared_offload_region.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/kv_offload/cpu/shared_offload_region.py)

```python
class SharedOffloadRegion:
    """
    多个 worker 进程通过 /dev/shm 共享 CPU KV Cache 内存。

    交错布局 (interleaved layout):
      worker0_block0 | worker1_block0 | ... | workerN_block0
      worker0_block1 | worker1_block1 | ... | workerN_block1
      ...
    
    每个 worker 只负责自己那一列，避免跨 worker 同步。
    """

    def __init__(self, instance_id, total_size_bytes, num_blocks, rank, num_workers):
        self.mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
        self._row_stride = cpu_page_size * num_workers  # 一行 = 所有 worker 的数据

        # 第一个 worker 创建文件并 ftruncate
        try:
            self.fd = os.open(self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(self.fd, total_size_bytes)  # 预分配物理空间
            self._creator = True
        except FileExistsError:
            # 后续 worker 打开已有文件
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, total_size_bytes)  # 等 creator 写完

        self.mmap_obj = mmap.mmap(self.fd, total_size_bytes,
                                   flags=mmap.MAP_SHARED,
                                   prot=mmap.PROT_READ | mmap.PROT_WRITE)

        # MADV_POPULATE_WRITE: 预分配物理页，避免按需分页的延迟尖峰
        _MADV_POPULATE_WRITE = 23  # Linux 5.14+
        for block in range(num_blocks):
            raw_offset = block * self._row_stride + worker_offset
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, aligned_offset, aligned_length)
```

**为什么用 mmap 而不是 `torch.zeros`？**

```
torch.zeros (pinned memory):
  - 每个 worker 独立分配 → N 个 worker = N 份内存
  - 浪费 N 倍内存

mmap MAP_SHARED:
  - 所有 worker 共享同一份物理内存
  - 内存使用 = 1 份（总量）
  - /dev/shm 上的文件不占磁盘空间 (tmpfs)
```

### 3.5 DMA 传输 Handler

[/vllm/v1/kv_offload/cpu/gpu_worker.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/kv_offload/cpu/gpu_worker.py#L111-L340)

```python
class SingleDirectionOffloadingHandler(OffloadingHandler):
    """
    处理 GPU→CPU 或 CPU→GPU 的单向传输。
    使用 swap_blocks_batch 自定义 CUDA 算子执行 DMA。
    """

    def __init__(self, gpu_tensors, cpu_tensors, block_size_factor,
                 kv_cache_groups_data_refs, gpu_to_cpu):
        # GPU tensors: (num_gpu_blocks, gpu_page_size_bytes) int8
        # CPU tensors: (num_cpu_blocks, cpu_page_size_bytes) int8
        self.src_tensors = gpu_tensors if gpu_to_cpu else cpu_tensors
        self.dst_tensors = cpu_tensors if gpu_to_cpu else gpu_tensors
        self.gpu_to_cpu = gpu_to_cpu

        # 独立 CUDA stream 池 → 传输不阻塞计算
        self._stream_pool = []  # 可复用的 streams

    def transfer_async(self, job_id, transfer_spec):
        # 构建批量 DMA 描述符
        batch_src = [...]   # 源块指针数组
        batch_dst = [...]   # 目标块指针数组
        batch_sizes = [...] # 块大小数组

        # 分配/复用 stream
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()

        # 保证顺序执行
        if self.gpu_to_cpu:
            stream.wait_stream(torch.cuda.current_stream())  # 等计算完成
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)  # 等上一个传输完成

        with torch.cuda.stream(stream):
            start_event.record(stream)
            if num_copy_ops > 0:
                # ← 自定义 CUDA 算子：批量块拷贝
                ops.swap_blocks_batch(
                    batch_src, batch_dst, batch_sizes,
                    is_src_access_order_any=(not self.gpu_to_cpu),  # CPU→GPU 优化
                )
            end_event.record(stream)

        self._transfers.append(Transfer(job_id, stream, start_event, end_event))

    def get_finished(self):
        """轮询已完成的传输 → 返回完成的 job_ids"""
        finished = []
        while self._transfers:
            if self._transfers[0].end_event.query():  # CUDA event 已完成
                finished.append(self._transfers.popleft().job_id)
            else:
                break
        return finished
```

---

## 四、简化版 Simple Offloading

### 4.1 SimpleCPUOffloadScheduler

[/vllm/v1/simple_kv_offload/manager.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/simple_kv_offload/manager.py)

简化版的最大特点：**复用 GPU 的 BlockPool 和 prefix caching 基础设施作为 CPU 端缓存。**

```
GPU BlockPool (80GB):   活跃块的 KV Cache
CPU BlockPool (10GB):   不活跃块的 KV Cache  — 用和 GPU 完全一样的数据结构
                            ↓
             复用 KVCacheCoordinator + prefix caching hash 匹配
```

```python
class SimpleCPUOffloadScheduler:
    def __init__(self, cpu_bytes_to_use, block_size, ...):
        # 在 CPU 端创建一个独立的 BlockPool
        self.cpu_block_pool = BlockPool(num_cpu_blocks, ...)
        self.cpu_kv_cache_coordinator = KVCacheCoordinator(...)

    def bind_gpu_block_pool(self, gpu_block_pool):
        """绑定 GPU BlockPool，用于 touch() 防止过早释放"""
        self.gpu_block_pool = gpu_block_pool

    def _prepare_eager_store_specs(self, scheduler_output):
        """Eager 模式: 每个 step 结束后立即卸载已计算的块"""
        for block in scheduler_output.computed_blocks:
            # 只卸载 GPU 缓存中已经有 prefix cache 命中的块
            # → 这些块被多个请求共享，值得保留
            ...

    def _prepare_lazy_store_specs(self):
        """Lazy 模式: 游标扫描 GPU 空闲队列，选择性卸载"""
        for block in self._scan_gpu_idle_queue():
            if self._should_store(block):
                # 只有当 CPU 端没有这个块的内容时才卸载
                ...

    def update_connector_output(self, outputs):
        """收集各 worker 的完成状态 → complete_store"""
        for worker_output in outputs:
            for job_id in worker_output.finished:
                self.complete_store(job_id)
```

### 4.2 DmaCopyBackend

[/vllm/v1/simple_kv_offload/copy_backend.py](file:///e:/cs336_note_and_hw-main/cs336_note_and_hw-main/HLIEvLLM-1.4.0rc0/vllm/v1/simple_kv_offload/copy_backend.py)

```python
class DmaCopyBackend:
    """使用 cuMemcpyBatchAsync 做 GPU↔CPU DMA 传输"""

    def __init__(self):
        self.load_stream = torch.cuda.Stream()   # CPU→GPU 独立流
        self.store_stream = torch.cuda.Stream()  # GPU→CPU 独立流

    def launch_copy(self, src_ptrs, dst_ptrs, sizes, direction):
        """异步启动批量 DMA 拷贝"""
        if direction == "gpu_to_cpu":
            with torch.cuda.stream(self.store_stream):
                cuMemcpyBatchAsync(dst_ptrs, src_ptrs, sizes)
        else:
            with torch.cuda.stream(self.load_stream):
                cuMemcpyBatchAsync(dst_ptrs, src_ptrs, sizes)
```

---

## 五、完整数据流

### 5.1 存储流程（GPU → CPU）

```
1. Scheduler 每个 step 结束后:
   ┌──────────────────────────────────────────────────────┐
   │ _build_kv_connector_meta(connector, scheduler_output) │
   │                                                      │
   │ connector.scheduler.update_state_after_alloc()        │
   │   → 找到本轮新计算的 GPU 块                            │
   │   → CPUOffloadingManager.prepare_store(keys)          │
   │     → 过滤 (store_threshold)                          │
   │     → 驱逐 (LRU eviction)                              │
   │     → 分配 CPU 块 + 生成 store_spec                  │
   │                                                      │
   │ scheduler_output.kv_connector_metadata = {            │
   │   "store_jobs": [(job_id, block_ids, ...)],           │
   │   "load_jobs":  [...]                                 │
   │ }                                                    │
   └──────────────────┬───────────────────────────────────┘
                      │
2. Worker 收到 scheduler_output:
   ┌──────────────────▼───────────────────────────────────┐
   │ ActiveKVConnector.post_forward()                      │
   │                                                      │
   │ for job in store_jobs:                                │
   │   handler.transfer_async(job_id, transfer_spec)       │
   │     → 独立 CUDA stream                                │
   │     → stream.wait_stream(compute_stream)  # 等计算完成 │
   │     → swap_blocks_batch(src=gpu, dst=mmap, sizes)     │
   │                                                      │
   │ 返回 finished_sending = [完成的 job_ids]               │
   └──────────────────┬───────────────────────────────────┘
                      │
3. Scheduler 下一个 step:
   ┌──────────────────▼───────────────────────────────────┐
   │ connector.scheduler.update_connector_output(outputs)  │
   │                                                      │
   │ for job_id in finished:                               │
   │   CPUOffloadingManager.complete_store(keys)            │
   │     → block.ref_cnt = 0  # 标记可读                   │
   │     → 触发事件 (OffloadingEvent)                      │
   └──────────────────────────────────────────────────────┘
```

### 5.2 加载流程（CPU → GPU）

```
1. Scheduler 做 prefix caching 检查时:
   ┌──────────────────────────────────────────────────┐
   │ can_allocate(seq)                                 │
   │                                                   │
   │ 1. 先查 GPU BlockPool（同第11课）                   │
   │ 2. GPU 未命中 → 查 KVConnector (CPU offload)       │
   │    connector.get_num_new_matched_tokens(seq)        │
   │      → CPUOffloadingManager.lookup(key)             │
   │        → 命中！返回 True                            │
   │                                                   │
   │ 3. prepare_load(keys)                              │
   │    → ref_cnt++，保护不被驱逐                        │
   │    → 生成 load_spec (src=cpu_block_ids)            │
   └──────────────────┬───────────────────────────────┘
                      │
2. Worker pre_forward():
   ┌──────────────────▼───────────────────────────────┐
   │ ActiveKVConnector.start_load_kv()                 │
   │                                                   │
   │ for job in load_jobs:                             │
   │   handler.transfer_async(job_id, load_spec)        │
   │     → swap_blocks_batch(src=mmap, dst=gpu, sizes) │
   │     → is_src_access_order_any=True (CPU 不会被并发写)│
   └──────────────────┬───────────────────────────────┘
                      │
3. Worker post_forward():
   ┌──────────────────▼───────────────────────────────┐
   │ 返回 finished_recving = [完成的 load job_ids]      │
   │                                                   │
   │ Scheduler: complete_load(keys) → ref_cnt--         │
   └──────────────────────────────────────────────────┘
```

---

## 六、多 Worker TP 场景下的 mmap 交错布局

```
TP=4, 每个 worker 管理独立的 GPU KV Cache

CPU mmap 文件: /dev/shm/vllm_offload_xxx.mmap

Layout (交错):
  Row 0 (block 0): [Worker0_data] [Worker1_data] [Worker2_data] [Worker3_data]
  Row 1 (block 1): [Worker0_data] [Worker1_data] [Worker2_data] [Worker3_data]
  ...

每个 worker 只读写自己的列:
  Worker 0: offset = 0 * cpu_page_size
  Worker 1: offset = 1 * cpu_page_size
  Worker 2: offset = 2 * cpu_page_size
  Worker 3: offset = 3 * cpu_page_size
```

**为什么是交错的而不是连续的？**

```
连续布局:
  [W0_block0, W0_block1, ..., W0_blockN, W1_block0, ...]
  问题: 不同 worker 的块可能大小不同 → 需要额外偏移表

交错布局:
  [W0_b0, W1_b0, W2_b0, W3_b0, W0_b1, W1_b1, W2_b1, W3_b1, ...]
  优势: 每个 worker 的 stride = cpu_page_size × num_workers
        块大小统一、寻址简单、cache-friendly
```

---

## 七、性能分析

### 7.1 延迟 vs 容量

```
                    HBM          DRAM         SSD
延迟 (ns):         ~200         ~100,000     ~10,000,000
带宽 (GB/s):      2000         50-100       3-7
容量 (TB):         0.08        0.5-2        10+

Offloading trade-off:
  保持 100% 在 GPU:      最低延迟，最小 capacity
  10% offload 到 CPU:     +2-5ms/step，capacity ×2
  全部 offload 到 SSD:    +50-100ms/step，capacity ×100
```

### 7.2 何时 offloading 有收益？

```
场景 A: 长 context + 多请求共享前缀
  GPU 存不下所有 KV Cache → 必须 offload
  收益: 支持更大 batch / 更长 context (功能需求)

场景 B: GPU 足够，offload 作预取缓存
  CPU 作为二级缓存存储 prefix 块
  收益: 新请求 prefill 跳过 prefix → 延迟降低 30-50%

场景 C: GPU 勉强够用，offload 降级
  驱逐不活跃块到 CPU，active 块独占 GPU
  收益: 避免 OOM，througput 可能反而提升 (减少 GPU 碎片)
```

---

## 八、与推理工具链工作的关联

| 关联方向 | 具体场景 |
|---|---|
| **Runtime/驱动** | `swap_blocks_batch` 是自定义 CUDA 算子 → 需要 HAL 层的 DMA 支持；Houmo NPU 如果要做 offloading，需要 NPU→CPU DMA 通道 |
| **编译器团队** | Houmo .hmm 静态图 → KV Cache 形状固定 → offloading 需要把 `block_table` 翻译成 CPU 端 DMA 描述符（类似 PagedAttention 的 Software DMA Assembly） |
| **量化团队** | CPU 端存储可以用更小的 dtype（如 int8）→ 减少 mmap 大小 → `gpu_page_size != cpu_page_size` → 需要 `block_size_factor` 参数 |
| **客户方案** | "支持 XXX K context" → GPU 显存不够 → KV offloading 是核心卖点 |

---

## 九、白话总结

```
KV Cache Offloading = 显存管理的"挪车"策略

Imagine GPU 显存是一个停车场 (HBM, 80 个车位):
  - 正在被 active 使用的车 → 停在地上 (GPU)
  - 等红灯的车 (waiting 队列) → 停到地库 (CPU DRAM)
  - 几个月才用一次的车 → 停到郊区 (SSD)

PagedAttention = 把停车场分成"车位" (blocks)
Offloading    = 决定哪个车位停地上、哪个停地下

关键设计:
  - LRU/ARC 淘汰策略: 把最不常用的车挪到地库
  - mmap 共享内存: 地上地下用同一个"车位编号系统"
  - 独立 DMA stream: 挪车不影响正在开车的 (compute stream)
  - store_threshold: 只挪"被找过多次"的车 (共享前缀)，一次性的不管
```

---

## 十、思考题

1. **为什么 `store_threshold > 1` 能减少无意义的卸载？哪些块的访问频率最高？**
   - 提示：prefix caching → 被多个请求共享的前缀块

2. **mmap 交错布局在 TP 多 worker 场景下有什么优势？如果改成连续布局会有什么问题？**
   - 提示：块大小统一 vs 需要偏移表、cache line 利用率

3. **GPU→CPU 传输时为什么 `wait_stream(compute_stream)` 而 CPU→GPU 时不需要？**
   - 提示：GPU KV Cache 正在被 compute kernel 写入 → 必须等完成才能读

4. **Houmo NPU 要实现 KV Cache offloading，需要 Runtime/驱动层做什么？和 PagedAttention 的 Software DMA Assembly 有何异同？**
   - 提示：.hmm 静态 KV Cache shape vs 动态 block 分配、BSP 同步 vs 异步 DMA

5. **如果 CPU→GPU 带宽是 50 GB/s，加载一个 4096 token 序列的全部 KV Cache (约 1.5GB) 需要多久？这段时间能被打断吗？**
   - 提示：1.5GB / 50GB/s ≈ 30ms，是异步 DMA 可以同时做其他事
