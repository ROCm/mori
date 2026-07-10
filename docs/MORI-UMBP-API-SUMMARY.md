# MORI UMBP API 总结报告

本文档基于 MORI 仓库中的 UMBP 文档和代码整理，包括：

- `docs/api/umbp.rst`
- `src/umbp/doc/design-master-control-plane.md`
- `src/umbp/doc/runtime-env-vars.md`
- `docs/MORI-UMBP-SINGLE-NODE-GUIDE.md`
- `docs/MORI-UMBP-PD-BENCHMARK.md`
- `src/pybind/pybind_umbp.cpp`
- `src/umbp/include/umbp/umbp_client.h`

## 1. UMBP 是什么

UMBP 是 MORI 中面向 LLM KV cache 场景的分层 KV block 存储和分布式路由模块。它主要服务 SGLang HiCache 这类系统，用 key/hash 管理 KV cache block，并支持把数据放在本地或远端的 HBM、DRAM、SSD 等层级中。

UMBP 的核心能力包括：

- 本地或分布式 KV block `Put` / `Get`
- 批量 KV block 写入、读取和存在性检查
- DRAM / SSD 分层存储
- 分布式 master 路由和 peer-to-peer 数据传输
- 外部 KV cache block 的上报、撤销和匹配查询
- 基于 external KV match 的热度计数
- RDMA zero-copy 所需的内存注册
- 可选 Prometheus metrics 和 SGLang HiCache 部署集成

## 2. 必要背景

UMBP 当前采用 `master-as-advisor` 设计。

这个设计的关键点是：

- Master 是路由建议者，不是真实 allocator。
- Peer 是真实数据 owner，负责本地 HBM / DRAM / SSD 上的 page 和 block 状态。
- Master 上的 `GlobalBlockIndex` 是通过 peer heartbeat 异步投影出来的索引。
- 数据面传输不经过 master，而是在 client 和 peer 之间通过 MORI-IO / RDMA / peer service 完成。
- Master 索引可能滞后一个 heartbeat；如果索引和 peer 真实状态冲突，以 peer 为准。

简化结构如下：

```text
UMBPClient / DistributedClient
        |
        v
PoolClient
        |
        +--> MasterClient  ---- heartbeat / route / external KV RPC ----> UMBP Master
        |
        +--> PeerServiceServer <---- Allocate / Resolve / SSD read ----> other peers
        |
        +--> PeerDramAllocator / PeerSsdManager
```

## 3. 对外 API 分类

UMBP 对外接口主要分为四类：

| 类别 | 代表对象 | 用途 |
| --- | --- | --- |
| 数据面客户端 | `UMBPClient` / `IUMBPClient` | KV block put/get、batch 操作、内存注册、external KV 查询 |
| 控制面客户端 | `UMBPMasterClient` | master 注册、external KV report/revoke/match、hit count 查询 |
| 配置对象 | `UMBPConfig` 等 | 控制 DRAM/SSD/distributed/eviction/copy pipeline 配置 |
| 辅助工具 | `UMBPHostMemAllocator` | 分配 host buffer，支持 hugepage、NUMA、prefault 等选项 |

Python 中主要通过两个入口使用：

```python
from mori.umbp import UMBPClient, UMBPConfig
from mori.cpp import UMBPMasterClient, UMBPTierType
```

## 4. `UMBPClient` 数据面 API

`UMBPClient` 是 UMBP 最主要的数据面入口。它是 `IUMBPClient` 的 Python binding，根据 `UMBPConfig` 自动创建 standalone 或 distributed 实现。

### 4.1 构造

```python
from mori.umbp import UMBPClient, UMBPConfig

cfg = UMBPConfig()
client = UMBPClient(cfg)
```

当 `cfg.distributed` 为空时，创建本地 standalone client；当 `cfg.distributed` 被设置时，创建 distributed client。

### 4.2 单 key 操作

| API | 作用 | 返回值 |
| --- | --- | --- |
| `put_from_ptr(key, src, size)` | 从用户指针 `src` 写入 `size` 字节到 `key` | `bool` |
| `get_into_ptr(key, dst, size)` | 从 `key` 读取数据到用户指针 `dst` | `bool` |
| `exists(key)` | 检查 `key` 是否存在 | `bool` |

示例：

```python
import ctypes
from mori.umbp import UMBPClient, UMBPConfig

client = UMBPClient(UMBPConfig())

src = (ctypes.c_ubyte * 256)(*([ord("A")] * 256))
dst = (ctypes.c_ubyte * 256)()

assert client.put_from_ptr("block-0", ctypes.addressof(src), 256)
assert client.exists("block-0")
assert client.get_into_ptr("block-0", ctypes.addressof(dst), 256)
```

### 4.3 批量操作

| API | 作用 |
| --- | --- |
| `batch_put_from_ptr(keys, ptrs, sizes)` | 批量写入多个 key |
| `batch_put_from_ptr_with_depth(keys, ptrs, sizes, depths)` | 带 radix-tree depth 信息的批量写入，用于 eviction priority |
| `batch_get_into_ptr(keys, ptrs, sizes)` | 批量读取多个 key |
| `batch_exists(keys)` | 批量存在性检查 |
| `batch_exists_consecutive(keys)` | 从第一个 key 开始连续命中数量，遇到第一个 miss 停止 |

`batch_exists_consecutive()` 适合 prefix cache 场景。例如给定一串 prefix block hash，它可以快速返回从开头连续命中的 block 数量。

### 4.4 生命周期 API

| API | 作用 |
| --- | --- |
| `clear()` | 清空当前 client 管理的内容 |
| `flush()` | 持久化 pending write-back 数据 |
| `close()` | 关闭 client，停止后台线程并释放资源 |
| `is_distributed()` | 返回当前 client 是否为 distributed 模式 |

`close()` 是幂等的，重复调用应当安全。具体实现的析构函数也会做清理，但推荐显式调用。

### 4.5 内存注册 API

| API | 作用 |
| --- | --- |
| `register_memory(ptr, size)` | 注册 host buffer，用于 distributed RDMA zero-copy |
| `deregister_memory(ptr)` | 注销已注册 buffer |

Standalone 模式下这通常是 no-op；distributed 模式下会 pin/export buffer，使远端传输可以走更直接的数据路径。

### 4.6 External KV API

`UMBPClient` 也暴露 external KV 相关接口。它们用于上报和查询不由 UMBP allocator 直接管理的 KV cache block，例如 SGLang HiCache 自己持有的 L1/L2/L3 cache block。

| API | 作用 |
| --- | --- |
| `report_external_kv_blocks(hashes, tier)` | 上报本节点在某 tier 持有这些 hashes |
| `revoke_external_kv_blocks(hashes, tier)` | 从某 tier 撤销这些 hashes |
| `revoke_all_external_kv_blocks_at_tier(tier)` | 撤销本节点某 tier 上所有 external KV |
| `match_external_kv(hashes, count_as_hit=False)` | 查询哪些节点持有这些 hashes |
| `get_external_kv_hit_counts(hashes)` | 查询 external KV 命中计数 |

注意：在 distributed data-plane client 中，external KV 查询失败可能返回空列表，而不是抛异常。如果调用方需要区分“真的没有匹配”和“RPC 失败”，应使用 `UMBPMasterClient`。

## 5. `UMBPMasterClient` 控制面 API

`UMBPMasterClient` 是轻量 master control-plane 客户端。文档明确指出它不是完整数据面客户端，不注册 peer service，不启动 heartbeat thread，适合 scheduler、sidecar、dashboard、控制脚本使用。

### 5.1 构造

```python
from mori.cpp import UMBPMasterClient

client = UMBPMasterClient(
    "127.0.0.1:15558",
    node_id="worker-0",
    node_address="worker-0:8080",
)
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `master_address` | UMBP master 地址，例如 `127.0.0.1:15558` |
| `node_id` | 当前节点 ID，注册时必需 |
| `node_address` | 当前节点地址，供 peer 或 scheduler 使用 |

构造本身是非阻塞的，gRPC channel 是 lazy 创建的。即使 master 暂时不可达，构造也不会立即失败。

### 5.2 节点注册 API

| API | 作用 |
| --- | --- |
| `register_self(tier_capacities)` | 注册当前节点及各 tier 容量 |
| `unregister_self()` | 注销当前节点 |
| `is_registered()` | 查询本地对象是否认为自己已注册 |

容量格式：

```python
from mori.cpp import UMBPTierType

_1GB = 1024 * 1024 * 1024
client.register_self({
    UMBPTierType.DRAM: (_1GB, _1GB),
})
```

tuple 含义是：

```text
(total_capacity_bytes, available_capacity_bytes)
```

### 5.3 External KV 上报和撤销

| API | 作用 |
| --- | --- |
| `report_external_kv_blocks(node_id, hashes, tier)` | 上报 `node_id` 在 `tier` 上持有这些 hashes |
| `revoke_external_kv_blocks(node_id, hashes, tier)` | 撤销 `node_id` 在单个 tier 上的这些 hashes |
| `revoke_all_external_kv_blocks_at_tier(node_id, tier)` | 撤销 `node_id` 某个 tier 上的所有 external KV |

语义要点：

- `report_external_kv_blocks()` 是 additive 的。
- 同一个 hash 可以同时存在于同一个 node 的多个 tier。
- 重复 report 同一 tier 是 no-op。
- report 到新 tier 会增加新的 tier bucket，不会删除旧 tier。
- revoke 单个 tier 不影响其它 tier。
- bulk revoke 只清理指定 tier，不影响其它 tier。

示例：

```python
from mori.cpp import UMBPMasterClient, UMBPTierType

master = "127.0.0.1:15558"
node = UMBPMasterClient(master, node_id="node-a", node_address="node-a:8080")

hashes = ["sha256-abc", "sha256-def"]
node.report_external_kv_blocks("node-a", hashes, UMBPTierType.DRAM)
node.report_external_kv_blocks("node-a", hashes, UMBPTierType.HBM)

# 只撤销 HBM bucket，DRAM bucket 仍保留。
node.revoke_external_kv_blocks("node-a", hashes, UMBPTierType.HBM)
```

### 5.4 External KV 查询

| API | 作用 |
| --- | --- |
| `match_external_kv(hashes, count_as_hit=False)` | 查询哪些节点持有这些 hashes |
| `get_external_kv_hit_counts(hashes)` | 查询 hashes 的累计命中计数 |

`match_external_kv()` 返回 `list[UMBPExternalKvNodeMatch]`。每个 match 表示一个 node。

`UMBPExternalKvNodeMatch` 字段：

| 字段 / 方法 | 说明 |
| --- | --- |
| `node_id` | 持有匹配 hash 的节点 |
| `peer_address` | 该节点 peer service 地址，可能为空 |
| `hashes_by_tier` | `dict[UMBPTierType, list[str]]`，按 tier 分组的匹配 hashes |
| `matched_hash_count()` | 去重后的匹配 hash 数量 |

重要语义：

- 返回结果按 node 分组，再按 tier 分组。
- 同一个 hash 在同一 node 的多个 tier 中出现时，`matched_hash_count()` 只计一次。
- 不要简单把各 tier bucket 长度相加作为命中数量。
- `count_as_hit=True` 只应在真实请求路由路径使用。

示例：

```python
query = UMBPMasterClient("127.0.0.1:15558")
matches = query.match_external_kv(["h0", "h1", "h2"])

for match in matches:
    print(match.node_id, match.matched_hash_count())
    for tier, hashes in match.hashes_by_tier.items():
        print("  ", tier, hashes)
```

## 6. External KV Hit Count 语义

UMBP master 维护一个 external KV hit-count index，用于记录某些 hash 被真实路由查询命中的次数。

核心语义：

- 只有 `match_external_kv(hashes, count_as_hit=True)` 会增加计数。
- 每次调用中，每个 unique hash 最多增加 1。
- hash 不匹配任何 external placement 时不计数。
- 同一个 hash 被多个 node 或多个 tier 持有，也不会在一次 query 中重复增加。
- counter 是 lifetime cumulative，不是 QPS，也不是滑动窗口。
- master 重启后计数丢失。
- 某个 hash 超过 `UMBP_HIT_INDEX_TTL_SEC` 未被 counted match，会被 GC。
- `get_external_kv_hit_counts()` 是 sparse lookup，未出现的 hash 不会返回 `0`，而是直接缺失。

典型用法：

```python
router = UMBPMasterClient("127.0.0.1:15558")

# 真实请求路由路径，才设置 count_as_hit=True。
matches = router.match_external_kv(prefix_hashes, count_as_hit=True)

monitor = UMBPMasterClient("127.0.0.1:15558")
entries = monitor.get_external_kv_hit_counts(prefix_hashes)
hotness = {entry.hash: entry.hit_count_total for entry in entries}
```

UMBP 不会根据 hotness 自动迁移、复制、pin 或改变 eviction 策略。它只暴露计数 primitive，后续策略由上层 scheduler 或 sidecar 实现。

## 7. `UMBPTierType` 和 tier 语义

| Tier | 说明 |
| --- | --- |
| `UMBPTierType.Unknown` | 未知或未指定 |
| `UMBPTierType.HBM` | GPU HBM |
| `UMBPTierType.DRAM` | Host DRAM |
| `UMBPTierType.SSD` | SSD |

需要区分两类 SSD：

1. **UMBP-owned SSD tier**
   - 数据由 `PeerSsdManager` 管理。
   - 位置存储在 `GlobalBlockIndex`。
   - 通过 heartbeat 上报 `KvEvent{tier=SSD}`。
   - 可以通过 UMBP get path 读取。

2. **External KV 的 SSD tier**
   - 只是外部系统上报的调度 metadata。
   - 存储在 `ExternalKvBlockIndex`。
   - UMBP 不拥有数据，也不能通过 `PrepareSsdRead` 读取它。

这两者名字相同但服务路径不同，不能混淆。

## 8. 配置 API

配置 API 的核心对象是 `UMBPConfig`。调用方通过它描述要创建什么类型的 `UMBPClient`，以及本地 DRAM/SSD、分布式 master、peer service、copy pipeline 等能力是否启用。

配置对象大致分为几类：

| 配置对象 | API 层面的作用 |
| --- | --- |
| `UMBPConfig` | 顶层配置，传给 `UMBPClient(config)`，决定创建 standalone 还是 distributed client |
| `UMBPDramConfig` | 配置本地 DRAM tier 是否可用、容量策略和 host memory 行为 |
| `UMBPSsdConfig` | 配置 SSD tier 是否启用，以及使用 file、SPDK 或 SPDK proxy 作为后端 |
| `UMBPEvictionConfig` | 配置本地或分布式路径中的 eviction 策略参数 |
| `UMBPCopyPipelineConfig` | 配置 DRAM/HBM 到 SSD 的异步 copy pipeline |
| `UMBPDistributedConfig` | 配置 distributed 模式，包括 master 地址、本节点标识、IO engine 和 peer service |
| `UMBPMasterClientConfig` | 配置 master control-plane client 的连接信息和节点身份 |
| `UMBPIoConfig` / `UMBPIoEngineConfig` | 配置本地或分布式数据传输使用的 I/O 后端和 endpoint |

最重要的语义是：`UMBPConfig.distributed` 为空时，`UMBPClient` 是本地 standalone client；设置了 `UMBPDistributedConfig` 后，`UMBPClient` 会启用 master/peer/RDMA 相关的 distributed path。

构造 standalone client：

```python
cfg = UMBPConfig()
# 根据需要开启或关闭本地 DRAM/SSD tier。
client = UMBPClient(cfg)
```

构造 distributed client：

```python
cfg = UMBPConfig()
dist = UMBPDistributedConfig()

# 填入 master、本节点身份、IO engine 和 peer service 等部署信息。
cfg.distributed = dist

client = UMBPClient(cfg)
```

也可以用 `UMBPConfig.from_environment()` 从 `UMBP_*` 环境变量生成配置。适合 launcher 或集成层根据部署环境自动构造 client。

## 9. Host Memory Allocator API

UMBP 提供 `UMBPHostMemAllocator`，用于分配 host buffer。

入口：

```python
from mori.umbp import UMBPHostMemAllocator, UMBPHostBufferBacking
```

API：

```python
allocator = UMBPHostMemAllocator()

handle = allocator.alloc(
    size,
    backing=UMBPHostBufferBacking.Anonymous,
    hugepage_size=2 * 1024 * 1024,
    numa_node=-1,
    prefault=True,
)

allocator.free(handle)
```

`UMBPHostBufferHandle` 字段：

| 字段 | 说明 |
| --- | --- |
| `ptr` | 分配出来的地址 |
| `requested_size` | 用户请求大小 |
| `mapped_size` | 实际 mmap 大小 |
| `actual_backing` | 实际使用的 backing |
| `actual_alignment` | 实际对齐 |
| `bool(handle)` | 是否有效 |

`AnonymousHugetlb` 失败时会 fallback 到 `Anonymous`，返回的 handle 中 `actual_backing` 会反映真实 backing。

## 10. Master Server 启动接口

UMBP master 是独立 binary。

```bash
umbp_master [listen_address] [metrics_port]
```

默认：

```text
listen_address = 0.0.0.0:50051
metrics_port = 9091
```

构建：

```bash
mkdir -p build
cd build
cmake .. -DUMBP=ON
make -j$(nproc) umbp_master
```

示例：

```bash
./build/src/umbp/umbp_master
./build/src/umbp/umbp_master 127.0.0.1:15558
./build/src/umbp/umbp_master 127.0.0.1:15558 9099
MORI_UMBP_LOG_LEVEL=DEBUG ./build/src/umbp/umbp_master 127.0.0.1:15558
```

Python `mori.umbp` 包会尝试自动发现 wheel 中打包的 `umbp_master`，并设置 `UMBP_MASTER_BIN`。用户也可以显式设置该环境变量。

## 11. Runtime 环境变量

UMBP 文档把环境变量分为几类。

### 11.1 Master / registry

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UMBP_HEARTBEAT_TTL_SEC` | `10` | heartbeat TTL |
| `UMBP_REAPER_INTERVAL_SEC` | `5` | reaper 周期 |
| `UMBP_MAX_MISSED_HEARTBEATS` | `3` | 连续 missed heartbeat 数量 |
| `UMBP_LEASE_DURATION_SEC` | `10` | master read lease |
| `UMBP_HIT_INDEX_TTL_SEC` | `7200` | hit count entry TTL |
| `UMBP_HIT_INDEX_GC_INTERVAL_SEC` | `60` | hit index GC 周期 |
| `UMBP_HIT_QUERY_MAX_BATCH` | `4096` | hit count 查询最大 batch |
| `UMBP_ROUTE_PUT_SELECT_ALGO` | `most_available` | put 路由算法 |
| `UMBP_ROUTE_PUT_NODE_AFFINITY` | `none` | put node affinity 策略 |

### 11.2 Pool client / peer

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UMBP_RPC_SHUTDOWN_TIMEOUT_MS` | `3000` | shutdown RPC timeout |
| `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` | `3` | gRPC server shutdown deadline |
| `UMBP_METRICS_REPORT_INTERVAL_MS` | `1000` | metrics 上报周期 |
| `UMBP_SSD_GET_MAX_ATTEMPTS` | `1` | remote SSD get 尝试次数 |
| `UMBP_SSD_GET_RETRY_BACKOFF_MS` | `2` | SSD get retry backoff |
| `UMBP_RELEASE_LEASE_TIMEOUT_MS` | `1000` | release SSD lease timeout |
| `UMBP_SSD_PREPARE_TIMEOUT_MS` | `0` | PrepareSsdRead deadline |

### 11.3 Client 配置 overlay

| 环境变量 | 说明 |
| --- | --- |
| `UMBP_DRAM_CAPACITY` | DRAM capacity |
| `UMBP_DRAM_HIGH_WM` / `UMBP_DRAM_LOW_WM` | DRAM eviction 水位线 |
| `UMBP_SSD_ENABLED` | 是否启用 SSD |
| `UMBP_SSD_DIR` | SSD file backend 目录 |
| `UMBP_SSD_CAPACITY` | SSD capacity |
| `UMBP_SSD_BACKEND` | `file` / `spdk` / `spdk_proxy` |
| `UMBP_EVICTION_POLICY` | eviction policy |
| `UMBP_ROLE` | `leader` / `follower` / `standalone` |
| `UMBP_SPDK_*` | SPDK 和 SPDK proxy 配置 |

### 11.4 部署脚本变量

这些通常由 SGLang / HiCache wrapper 或 UMBP 脚本使用：

| 环境变量 | 说明 |
| --- | --- |
| `UMBP_MASTER_ADDRESS` | master 地址 |
| `UMBP_MASTER_LISTEN` | master listen 地址 |
| `UMBP_MASTER_AUTO_START` | 是否自动启动 master |
| `UMBP_MASTER_BIN` | master binary 路径 |
| `UMBP_NODE_ADDRESS` | 当前节点对外地址 |
| `UMBP_IO_ENGINE_HOST` | IO engine host |
| `UMBP_IO_ENGINE_PORT` / `UMBP_IO_ENGINE_PORTS` | IO engine port |
| `UMBP_PEER_SERVICE_PORT` | peer service port |
| `UMBP_CACHE_REMOTE_FETCHES` | 是否缓存远程 fetch 回来的 block |

环境变量解析通常在首次使用时缓存。修改 env 后通常需要重启进程。

## 12. API 使用建议

### 12.1 数据面优先使用 `UMBPClient`

如果调用方需要真正读写 KV bytes，应使用：

```python
UMBPClient.put_from_ptr(...)
UMBPClient.get_into_ptr(...)
UMBPClient.batch_put_from_ptr(...)
UMBPClient.batch_get_into_ptr(...)
```

不要用 `UMBPMasterClient` 搬数据。它只做控制面。

### 12.2 调度器和 dashboard 使用 `UMBPMasterClient`

如果只需要：

- 注册节点
- 查询 external KV placement
- 查询 hit count
- 从 sidecar 代替 worker report/revoke external KV

则使用 `UMBPMasterClient`。

### 12.3 `count_as_hit=True` 只能用于真实路由路径

不要在 debug、health check、dashboard、probe 中设置 `count_as_hit=True`，否则会污染 hotness counter。

### 12.4 不要混淆 external KV 和 UMBP-owned KV

External KV 是调度 metadata，不一定由 UMBP 存储 bytes。UMBP-owned KV 是 UMBP 自己 Put/Get 管理的数据，可以通过 UMBP 数据面读取。

### 12.5 远端 SSD staging 需要按模型 KV 大小配置

文档中提到 remote SSD read 必须让单个 key 的完整 value 放进一个 staging slot：

```text
per_slot = ssd_staging_buffer_size / ssd_staging_buffer_slots
```

如果 slot 太小，会出现 `size_too_large`，请求通常不会报错，但会退回 recompute，导致 hit rate 和吞吐下降。修复方式是增大 `UMBP_SSD_STAGING_BYTES` 或减少 slot 数。

## 13. 总结

UMBP 对外 API 可以用一句话概括：

```text
UMBPClient 负责数据面 KV block 读写；
UMBPMasterClient 负责 master 控制面和 external KV 查询；
UMBPConfig 系列负责本地/分布式/DRAM/SSD/SPDK 等配置；
UMBPHostMemAllocator 提供 host buffer 分配辅助能力。
```

从架构上看，UMBP 不是强一致的全局 KV 数据库，而是面向 LLM KV cache 的高吞吐分层存储和路由系统。它把真实状态放在 peer 上，把 master 设计成 advisor，并通过 heartbeat 异步投影全局索引。这种设计降低了 master 对热路径的影响，适合 SGLang HiCache、prefill-decode disaggregation 和跨节点 KV-cache-aware scheduling 等场景。
