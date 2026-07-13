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

UMBP 的核心能力可以概括为：

| 能力 | API 相关入口 |
| --- | --- |
| 本地或分布式 KV block `Put` / `Get` | `UMBPClient` |
| 批量 KV block 写入、读取和存在性检查 | `UMBPClient` batch API |
| DRAM / SSD 分层存储 | `UMBPConfig` 系列配置对象 |
| 分布式 master 路由和 peer-to-peer 数据传输 | `UMBPClient` distributed 模式 |
| 外部 KV cache block 的上报、撤销和匹配查询 | `UMBPClient` / `UMBPMasterClient` external KV API |
| 基于 external KV match 的热度计数 | `match_external_kv(..., count_as_hit=True)` / `get_external_kv_hit_counts()` |
| RDMA zero-copy 所需的内存注册 | `register_memory()` / `deregister_memory()` |
| host buffer 分配 | `UMBPHostMemAllocator` |

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

API 总览：

| 类别 | API | 作用 | 使用建议 |
| --- | --- | --- | --- |
| 构造 | `UMBPClient(config)` | 创建 standalone 或 distributed data-plane client | 需要读写 KV bytes 时优先使用 |
| 单 key 写入 | `put_from_ptr(key, src, size)` | 从用户指针写入一个 KV block | 返回 `bool` 表示是否成功 |
| 单 key 读取 | `get_into_ptr(key, dst, size)` | 读取一个 KV block 到用户指针 | 由调用方保证 `dst` buffer 足够大 |
| 单 key 查询 | `exists(key)` | 检查 key 是否存在 | 只判断 UMBP-owned KV |
| 批量写入 | `batch_put_from_ptr(keys, ptrs, sizes)` | 批量写入多个 KV block | 适合批量 cache block 写入 |
| 批量写入 | `batch_put_from_ptr_with_depth(keys, ptrs, sizes, depths)` | 批量写入并携带 radix-tree depth | depth 可用于 eviction priority |
| 批量读取 | `batch_get_into_ptr(keys, ptrs, sizes)` | 批量读取多个 KV block | 适合 prefix/cache block 批量读取 |
| 批量查询 | `batch_exists(keys)` | 批量检查 key 是否存在 | 返回每个 key 的命中状态 |
| 连续命中查询 | `batch_exists_consecutive(keys)` | 返回从第一个 key 开始连续命中的数量 | 适合 prefix cache 场景 |
| 生命周期 | `clear()` / `flush()` / `close()` | 清空、持久化 pending 数据、关闭 client | 推荐显式调用 `close()` |
| 模式查询 | `is_distributed()` | 判断当前 client 是否为 distributed 模式 | 用于分支处理本地/分布式路径 |
| 内存注册 | `register_memory(ptr, size)` / `deregister_memory(ptr)` | 注册或注销 host buffer | distributed RDMA zero-copy 场景使用 |
| External KV | `report_external_kv_blocks(hashes, tier)` | 上报本节点持有 external KV hashes | external KV 是调度 metadata，不代表 UMBP 拥有数据 |
| External KV | `revoke_external_kv_blocks(hashes, tier)` | 撤销本节点某 tier 上的 external KV hashes | 只影响指定 tier |
| External KV | `revoke_all_external_kv_blocks_at_tier(tier)` | 撤销本节点某 tier 的全部 external KV | 用于 tier 级别清理 |
| External KV | `match_external_kv(hashes, count_as_hit=False)` | 查询哪些节点持有这些 hashes | data-plane client 中查询失败可能返回空列表 |
| External KV | `get_external_kv_hit_counts(hashes)` | 查询 external KV 累计命中计数 | 监控或调度器使用 |

### 4.1 构造示例

```python
from mori.umbp import UMBPClient, UMBPConfig

cfg = UMBPConfig()
client = UMBPClient(cfg)
```

当 `cfg.distributed` 为空时，创建本地 standalone client；当 `cfg.distributed` 被设置时，创建 distributed client。

### 4.2 单 key 操作示例

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

`batch_exists_consecutive()` 适合 prefix cache 场景。例如给定一串 prefix block hash，它可以快速返回从开头连续命中的 block 数量。

`close()` 是幂等的，重复调用应当安全。具体实现的析构函数也会做清理，但推荐显式调用。

Standalone 模式下这通常是 no-op；distributed 模式下会 pin/export buffer，使远端传输可以走更直接的数据路径。

注意：在 distributed data-plane client 中，external KV 查询失败可能返回空列表，而不是抛异常。如果调用方需要区分“真的没有匹配”和“RPC 失败”，应使用 `UMBPMasterClient`。

## 5. `UMBPMasterClient` 控制面 API

`UMBPMasterClient` 是轻量 master control-plane 客户端。文档明确指出它不是完整数据面客户端，不注册 peer service，不启动 heartbeat thread，适合 scheduler、sidecar、dashboard、控制脚本使用。

API 总览：

| 类别 | API | 作用 | 使用建议 |
| --- | --- | --- | --- |
| 构造 | `UMBPMasterClient(master_address, node_id=None, node_address=None)` | 创建 master control-plane client | 构造非阻塞，gRPC channel lazy 创建 |
| 节点注册 | `register_self(tier_capacities)` | 注册当前节点及各 tier 容量 | peer 或 worker 节点使用 |
| 节点注册 | `unregister_self()` | 注销当前节点 | 节点退出时使用 |
| 节点注册 | `is_registered()` | 查询本地对象是否认为已注册 | 本地状态判断，不等于全局强一致状态 |
| External KV 上报 | `report_external_kv_blocks(node_id, hashes, tier)` | 上报某节点在某 tier 持有 hashes | additive，重复上报同一 tier 是 no-op |
| External KV 撤销 | `revoke_external_kv_blocks(node_id, hashes, tier)` | 撤销某节点单个 tier 上的 hashes | 不影响其它 tier |
| External KV 撤销 | `revoke_all_external_kv_blocks_at_tier(node_id, tier)` | 撤销某节点某 tier 上全部 external KV | 用于 worker/tier 级别清理 |
| External KV 查询 | `match_external_kv(hashes, count_as_hit=False)` | 查询哪些节点持有这些 hashes | scheduler、sidecar、dashboard 使用 |
| Hit count 查询 | `get_external_kv_hit_counts(hashes)` | 查询 hashes 的累计命中计数 | 返回 sparse 结果，未出现的 hash 不返回 `0` |

### 5.1 构造示例

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

### 5.2 节点注册示例

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

### 5.3 External KV 上报和撤销语义

| 语义 | 说明 |
| --- | --- |
| additive report | `report_external_kv_blocks()` 只增加 placement，不删除旧 tier |
| 多 tier 共存 | 同一个 hash 可以同时存在于同一个 node 的多个 tier |
| 重复上报 | 重复 report 同一 tier 是 no-op |
| 单 tier 撤销 | revoke 单个 tier 不影响其它 tier |
| tier 级撤销 | bulk revoke 只清理指定 tier，不影响其它 tier |

External KV 是调度 metadata，不一定由 UMBP 存储 bytes。UMBP-owned KV 是 UMBP 自己 Put/Get 管理的数据，可以通过 UMBP 数据面读取。

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

### 5.4 External KV 查询和 hit count

`match_external_kv()` 返回 `list[UMBPExternalKvNodeMatch]`。每个 match 表示一个 node，结果按 node 分组，再按 tier 分组。

`UMBPExternalKvNodeMatch` 字段：

| 字段 / 方法 | 说明 |
| --- | --- |
| `node_id` | 持有匹配 hash 的节点 |
| `peer_address` | 该节点 peer service 地址，可能为空 |
| `hashes_by_tier` | `dict[UMBPTierType, list[str]]`，按 tier 分组的匹配 hashes |
| `matched_hash_count()` | 去重后的匹配 hash 数量 |

Hit count 语义：

| 规则 | 说明 |
| --- | --- |
| 计数入口 | 只有 `match_external_kv(hashes, count_as_hit=True)` 会增加计数 |
| 去重规则 | 每次调用中，每个 unique hash 最多增加 1 |
| miss 不计数 | hash 不匹配任何 external placement 时不计数 |
| 多副本不重复计数 | 同一个 hash 被多个 node 或多个 tier 持有，也不会在一次 query 中重复增加 |
| 计数性质 | counter 是 lifetime cumulative，不是 QPS，也不是滑动窗口 |
| 查询结果 | `get_external_kv_hit_counts()` 是 sparse lookup，未出现的 hash 不返回 `0` |
| 使用边界 | `count_as_hit=True` 只应在真实请求路由路径使用，debug、health check、dashboard、probe 不应设置 |

UMBP 不会根据 hotness 自动迁移、复制、pin 或改变 eviction 策略。它只暴露计数 primitive，后续策略由上层 scheduler 或 sidecar 实现。

示例：

```python
query = UMBPMasterClient("127.0.0.1:15558")
matches = query.match_external_kv(["h0", "h1", "h2"], count_as_hit=True)

for match in matches:
    print(match.node_id, match.matched_hash_count())
    for tier, hashes in match.hashes_by_tier.items():
        print("  ", tier, hashes)

entries = query.get_external_kv_hit_counts(["h0", "h1", "h2"])
hotness = {entry.hash: entry.hit_count_total for entry in entries}
```

### 5.5 `UMBPTierType` 枚举

| Tier | 说明 |
| --- | --- |
| `UMBPTierType.Unknown` | 未知或未指定 |
| `UMBPTierType.HBM` | GPU HBM |
| `UMBPTierType.DRAM` | Host DRAM |
| `UMBPTierType.SSD` | SSD |

需要区分两类 SSD：

| SSD 语义 | 数据归属 | 索引位置 | 是否能通过 UMBP get path 读取 |
| --- | --- | --- | --- |
| UMBP-owned SSD tier | UMBP / `PeerSsdManager` 管理 | `GlobalBlockIndex` | 可以 |
| External KV 的 SSD tier | 外部系统持有，UMBP 只记录 placement metadata | `ExternalKvBlockIndex` | 不可以 |

## 6. 配置 API

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

## 7. Host Memory Allocator API

UMBP 提供 `UMBPHostMemAllocator`，用于分配 host buffer。

入口：

```python
from mori.umbp import UMBPHostMemAllocator, UMBPHostBufferBacking
```

API 总览：

| API / 对象 | 作用 | 说明 |
| --- | --- | --- |
| `UMBPHostMemAllocator()` | 创建 host memory allocator | 用于分配 UMBP 相关 host buffer |
| `allocator.alloc(size, backing, hugepage_size, numa_node, prefault)` | 分配 host buffer | 返回 `UMBPHostBufferHandle` |
| `allocator.free(handle)` | 释放 host buffer | 释放 `alloc()` 返回的 handle |
| `UMBPHostBufferBacking.Anonymous` | 普通 anonymous mmap backing | 通用 fallback |
| `UMBPHostBufferBacking.AnonymousHugetlb` | hugetlb backing | 失败时会 fallback 到 `Anonymous` |

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

`UMBPHostBufferHandle` API：

| 字段 / 方法 | 说明 |
| --- | --- |
| `ptr` | 分配出来的地址 |
| `requested_size` | 用户请求大小 |
| `mapped_size` | 实际 mmap 大小 |
| `actual_backing` | 实际使用的 backing |
| `actual_alignment` | 实际对齐 |
| `bool(handle)` | 是否有效 |

`AnonymousHugetlb` 失败时会 fallback 到 `Anonymous`，返回的 handle 中 `actual_backing` 会反映真实 backing。

## 8. UMBP Feature 总结

| Feature | 相关组件 / API | 简明说明 |
| --- | --- | --- |
| 本地 `Put` / `Get` | `UMBPClient.put_from_ptr()` / `get_into_ptr()`、`LocalStorageManager` | 默认先写入或读取本地 DRAM/SSD tier；同一个 key 视为 content-addressed block，重复写入会 dedup。 |
| 分布式 `Get` | `PoolClient.GetRemote()`、`RouteGet`、`PeerService` | 本地 miss 后可向 master 查询 remote placement，再从远端 DRAM 直接 RDMA read，或从远端 SSD staging slot 读取。 |
| 分布式 `Put` | `PoolClient.PutRemote()`、`RoutePut`、`FinalizeAllocation` | 可通过 master 选择目标节点和 tier，远端 DRAM 走 RDMA write，远端 SSD 走 write staging slot + commit。 |
| 批量操作 | `batch_put_from_ptr()` / `batch_get_into_ptr()` / `batch_exists()` | 面向 KV block 批量写入、读取和存在性检查，适合 prefix cache / block cache 场景。 |
| 分层存储 | `LocalStorageManager`、`DRAMTier`、`SSDTier`、`SpdkSsdTier`、`SpdkProxyTier` | 本地存储按最快到最慢 tier 管理，支持 DRAM、POSIX/file SSD、SPDK SSD、SPDK proxy 等后端。 |
| Eviction / Demotion | `LocalStorageManager.Evict()` / `Demote()` / `SelectVictim()` | DRAM 空间不足时可按策略选择 victim 并 demote 到更慢 tier；SSD tier 也有各自的 eviction 实现。 |
| Read Promotion | `ReadIntoPtr()`、`MaybeAutoPromote()`、`Promote()` | 读取慢 tier 命中后可按配置自动 promote 到更快 tier，提高后续访问性能。 |
| Tier-change 发布 | `SetOnTierChange()`、`PublishLocalBlock()`、`UnregisterFromMaster()` | block 在 DRAM/SSD 间迁移或被完全淘汰时，回调会更新 master 上的 placement，避免 global index 长期指向旧位置。 |
| 异步 DRAM→SSD copy | `CopyPipeline`、`MaybeCopyToSharedSSD()` | 写入 DRAM 后可异步复制到 shared SSD，用于 shared-SSD leader/follower 或低成本持久化路径。 |
| Global Block Index | `GlobalBlockIndex` | master 侧维护 `key -> locations`，每个 key 可对应多个 node/tier/location，并记录访问指标。 |
| Local Block Index | `LocalBlockIndex` | client 本地维护 `key -> tier/size/location` hint，用于快速判断本地命中和清理 stale hint。 |
| Global Routing | `Router`、`RouteGetStrategy`、`RoutePutStrategy` | `RouteGet` 从多个 replica 中选择读源；`RoutePut` 从 alive clients 中选择写入节点和 tier。 |
| `Put` / `Get` / Eviction 扩展点 | `RoutePutStrategy`、`RouteGetStrategy`、`UMBPEvictionConfig`、`TierBackend` | 写入放置、读取 replica 选择、淘汰 victim 选择和存储 tier 后端都有独立扩展点；当前主要在构造期或配置期替换。 |
| 可插拔路由策略 | `RouteGetStrategy` / `RoutePutStrategy` | 读写路由策略通过接口注入，默认读策略是随机 replica，默认写策略是 fastest-tier-first + most-available。当前主要是构造期替换。 |
| 可插拔存储后端 | `TierBackend` | DRAM、file SSD、SPDK、SPDK proxy 等都通过 tier backend 接口接入，便于扩展新的存储实现。 |
| 可配置 eviction 策略 | `UMBPEvictionConfig`、`SelectVictim()` | eviction policy 可配置，例如普通 LRU 或 prefix-aware LRU；当前主要是配置期选择。 |
| Remote SSD staging | `PeerService`、`PrepareSsdRead`、`AllocateWriteSlot`、`CommitSsdWrite` | 远端 SSD 读写通过固定 staging slot 和 lease 协议完成，避免多个 RDMA/SSD I/O 并发覆盖同一 buffer。 |
| RDMA zero-copy | `register_memory()` / `deregister_memory()`、`PoolClient` | 调用方可注册 host buffer，分布式路径在匹配已注册内存时走更直接的数据传输。 |
| External KV metadata | `report_external_kv_blocks()` / `match_external_kv()` | 支持上报外部系统持有的 KV cache placement，用于调度查询；UMBP 不拥有 external KV bytes。 |
| Hit count primitive | `match_external_kv(..., count_as_hit=True)` / `get_external_kv_hit_counts()` | master 记录真实路由查询命中的累计计数，供 scheduler 或 sidecar 做热度策略。 |
| 节点生命周期 | `UMBPMasterClient.register_self()` / `unregister_self()`、heartbeat/reaper | 节点注册、注销和心跳用于维护 alive clients，master 索引会随节点失效清理。 |

### 8.1 未来规划：Agent Hints

目标：让 scheduler 根据 session 生命周期和 semantic spans，提前决定 sticky routing、UMBP admission 和 session-scoped metadata 清理。

| Hint | 首期用途 |
| --- | --- |
| `session_id` | sticky session routing，提高本地 HiCache L1/L2 命中率 |
| `session_action` | 管理 `bind` / `open` / `close` 生命周期边界 |
| `session_timeout_ms` | inactivity cleanup，处理漏发 `close` 的场景 |
| semantic spans | 首期识别 `SYSTEM_PROMPT`、`USER_QUERY`、`RESPONSE` |
| admission action | 收敛为 `skip` / `report` / `put` |

| Semantic phase | Admission | 策略 |
| --- | --- | --- |
| `SYSTEM_PROMPT` | `report`，后续可开放 `put` | 优先 L1/L2 命中；未命中可从 UMBP prefetch；close 后仍可保留 |
| `USER_QUERY` | 默认 `skip` UMBP | 只走本地 HiCache / engine prefix cache；close 时清理 session-scoped metadata |
| `RESPONSE` | 默认 `skip` UMBP | 避免污染外部缓存；close 时标记为可淘汰 |

| 组件 | 首期职责 |
| --- | --- |
| Scheduler | 解析 session hints 和 semantic spans；执行 sticky routing、lifecycle handling 和 semantic admission |
| HiCache | 执行 worker 内部 L1/L2 分层缓存，包括本地 prefix match、load back 和 selective write-through |
| UMBP | 作为共享 KV backend 或 external metadata path，提供跨 worker KV 发现、`report` / `revoke` 和可选 bytes `put` / `get` |

## 9. 总结

UMBP 对外 API 可以总结为：

| API 类别 | 主要对象 | 职责 |
| --- | --- | --- |
| 数据面 | `UMBPClient` | KV block 读写、批量操作、内存注册、external KV 查询 |
| 控制面 | `UMBPMasterClient` | master 注册、external KV 上报/撤销/查询、hit count 查询 |
| 配置 | `UMBPConfig` 系列 | 控制本地/分布式/DRAM/SSD/SPDK 等能力 |
| 辅助工具 | `UMBPHostMemAllocator` | 分配 host buffer，并暴露实际 backing、大小和对齐信息 |
