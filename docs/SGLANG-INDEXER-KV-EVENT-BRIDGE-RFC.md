# RFC: SGLang KV Indexer

## 目标

1. Indexer 需要感知和收集 KV 位置索引，因此设计 Bridge 将 SGLang KV event 转换成 indexer 的 KV 位置索引操作。

   Bridge 订阅 SGLang ZMQ `KVEventBatch`，将 `BlockStored` / `BlockRemoved` / `AllBlocksCleared` 转换成 `report` / `revoke` / `revoke-all`，用于维护 `hash -> node -> tier` 形式的 KV 位置索引。

2. Indexer 提供跨 worker 的 KV 位置索引查询。

   Scheduler 或查询方可以通过 indexer 查询某组 KV hashes 当前位于哪些 worker、哪些 tier 上，并可结合 hit count 统计做后续调度决策。该查询只返回位置元数据，不搬运 KV bytes。

## 宏观框架

```text
+------------------+        +----------------------------+
|      SGLang      |        |           Bridge           |
|------------------|        |----------------------------|
| HiCache / KV     | ZMQ    | SglangIndexKvEventBridge   |
| Event Publisher  +------->| - subscribe KVEventBatch   |
| BlockStored      | events | - decode events            |
| BlockRemoved     |        | - map hash / medium / tier |
| AllBlocksCleared |        | - batch report / revoke    |
+------------------+        +-------------+--------------+
                                           |
                                           | External KV API
                                           | report / revoke / revoke-all
                                           v
+------------------------+  +-------------+--------------+
|     Metadata Store     |<-|          Indexer           |
|------------------------|  |----------------------------|
| hash -> node           |  | External KV placement      |
| node -> tier           |  | match_external_kv          |
| hit counts             |  | get_external_kv_hit_counts |
|------------------------|  |                            |
| backends:              |  |                            |
| - in-memory            |  |                            |
| - Redis                |  |                            |
| - MySQL / PostgreSQL   |  |                            |
| - RocksDB              |  |                            |
+------------------------+  +----------------------------+
```

整体链路由四个组件组成：`SGLang` 负责产生 KV cache 生命周期事件，`Bridge` 负责订阅、解码和转换事件，`Indexer` 负责提供 External KV metadata 的写入和查询接口，`Metadata Store` 负责保存 placement 与 hit-count 等状态。Bridge 不参与 KV bytes 的读写，只把 SGLang 侧的 block 变化同步为 indexer 可查询的 metadata。

数据流是单向写入为主：SGLang 通过 ZMQ 推送 `BlockStored`、`BlockRemoved`、`AllBlocksCleared`，Bridge 将其转换成 report / revoke / revoke-all 请求写入 Indexer；调度器或查询方后续通过 Indexer 查询哪些节点持有某些 hashes，以及这些 hashes 的累计命中情况。

## SGLang - Indexer Bridge 设计

`SglangIndexKvEventBridge` 是 SGLang KV event stream 和 indexer External KV metadata API 之间的适配层。它部署在 SGLang worker 旁边，订阅 worker 发布的 KV cache 事件，并把事件转换成 indexer 可理解的 report / revoke 操作。

### 部署形态

Bridge 可以有三种实现 / 部署方式：

| 方式 | 位置 | 优点 | 不足 |
| --- | --- | --- | --- |
| 内嵌 SGLang | SGLang worker 进程内 | 事件获取最直接，本地上下文最完整 | 需要改 SGLang，和 SGLang 版本耦合 |
| 内嵌 indexer | indexer 进程内 | indexer 统一管理订阅、转换和写入 | 需要连接所有 SGLang workers，部署拓扑更复杂 |
| 独立 sidecar | SGLang worker 旁边独立进程 | 不侵入 SGLang，失败隔离好，可独立升级 | 多一个部署和监控组件 |

推荐首期采用 **独立 sidecar**。这种方式不需要修改 SGLang，也能保持 Bridge 靠近 event source；后续如果需要减少组件数量，可以再评估内嵌到 indexer。

### 功能

| 功能 | 说明 |
| --- | --- |
| 事件订阅 | 通过 ZMQ SUB 订阅 SGLang `KVEventBatch` |
| 事件解码 | 解码 `BlockStored`、`BlockRemoved`、`AllBlocksCleared` |
| tier 映射 | 将 SGLang `StorageMedium` 映射到 indexer tier |
| hash 规范化 | 将 SGLang block hash 转成 indexer 使用的 string hash |
| 批量上报 | 按 action 和 tier 聚合 hashes，减少 RPC 次数 |
| 多源支持 | 支持多个 worker / DP rank event source |
| 基础观测 | 记录 decode error、sequence gap、report / revoke 失败等计数 |

### 工作原理

Bridge 从 SGLang `/server_info` 自动发现 event publisher 的 endpoint、topic、DP rank 数和 block size；也可以通过配置手动指定 ZMQ endpoint。启动后，Bridge 为每个 event source 建立 SUB socket，按 batch 拉取事件并执行以下流程：

```text
ZMQ message
  -> decode KVEventBatch
  -> validate topic / sequence / dp_rank
  -> extract event type, block_hashes, medium
  -> normalize hash and map medium to tier
  -> batch by action + tier
  -> call indexer External KV API
```

### Event 映射

| SGLang event | Bridge 动作 |
| --- | --- |
| `BlockStored(block_hashes, medium)` | `report_external_kv_blocks(hashes, mapped_tier)` |
| `BlockRemoved(block_hashes, medium)` | `revoke_external_kv_blocks(hashes, mapped_tier)` |
| `AllBlocksCleared()` | 对所有已配置 tier 执行 `revoke_all_external_kv_blocks_at_tier(tier)` |

默认 medium 映射：

| SGLang medium | Indexer tier |
| --- | --- |
| `GPU` | `HBM` |
| `CPU_PINNED` | `DRAM` |
| `DISK` | `SSD` |
| `EXTERNAL` | 首期跳过，后续扩展 |

### 多 Worker / DP Rank

多 worker 场景下，Bridge 使用 `sources[]` 管理多个 event source。每个 source 独立绑定 `node_id`、`server_info_url` 或 ZMQ endpoint。DP attention 场景下，SGLang 通常每个 DP rank 一个 publisher，端口可按 `endpoint_port_base + dp_rank` 展开；Bridge 为每个 rank 建立独立 SUB socket，并分别维护 `last_seq`、`dropped_seq_count` 和 `last_event_ts`。

### 错误处理

Bridge 对事件流采用 best-effort 同步模型。ZMQ 暂时无消息时不视为错误；decode 失败、unknown medium、sequence gap 和 indexer RPC 失败都会进入统计。

## Indexer RPC API 设计

`SglangIndexKvEventBridge` 调用的是 indexer 的 External KV metadata API。这组 API 通过 protobuf / gRPC 暴露在 indexer service 中；在本 RFC 中可以把这组接口理解为 indexer RPC API。

### API 概览

| API | 作用 | 关键语义 |
| --- | --- | --- |
| `report_external_kv_blocks(hashes, tier)` | 上报本节点持有 external KV hashes | 写入 `(hash, node_id, tier)` metadata；重复上报同一 tuple 是 no-op |
| `revoke_external_kv_blocks(hashes, tier)` | 撤销本节点某 tier 上的 external KV hashes | 只删除指定 tier，不影响同一 hash 在其它 tier 的副本 |
| `revoke_all_external_kv_blocks_at_tier(tier)` | 撤销本节点某 tier 的全部 external KV | 用于 `AllBlocksCleared` 或 tier 级别清理 |
| `match_external_kv(hashes, count_as_hit=False)` | 查询哪些节点持有这些 hashes | 返回按 node 分组、按 tier 分桶的 placement |
| `get_external_kv_hit_counts(hashes)` | 查询 external KV 累计命中计数 | 返回 sparse hit-count entries，供监控或调度使用 |

用户侧 API 隐含使用当前节点的 `node_id`。RPC request 里会显式携带 `node_id`，因为 indexer 需要知道是谁在 report / revoke。

### Proto 形态

Proto 将 External KV 分成 mutation RPC 和 query RPC。Mutation request 主要携带 `node_id`、`hashes`、`tier`；query request 携带 `hashes` 和可选的 `count_as_hit`。

| RPC | Request 关键字段 | Response 关键字段 |
| --- | --- | --- |
| `ReportExternalKvBlocks` | `node_id`, `hashes`, `tier` | 空 |
| `RevokeExternalKvBlocks` | `node_id`, `hashes`, `tier` | 空 |
| `RevokeAllExternalKvBlocksAtTier` | `node_id`, `tier` | 空 |
| `MatchExternalKv` | `hashes`, `count_as_hit` | `matches[]`，每项包含 `node_id`, `peer_address`, `hashes_by_tier[]` |
| `GetExternalKvHitCounts` | `hashes` | `entries[]`，每项包含 `hash`, `hit_count_total` |

Indexer 还可以提供 `RevokeAllExternalKvBlocksForNode(node_id)`，用于节点注销或整节点清理。本 RFC 的 bridge 首期只需要 tier 级别 revoke-all。

### Indexer 实现模型

External KV placement index 可以抽象为 `hash -> node_id -> set<tier>`。因此一个 hash 可以同时存在于多个 node，也可以在同一个 node 的多个 tier 上存在副本。`match_external_kv` 返回时按 node 聚合，并把每个 node 命中的 hashes 再按 tier 分桶。

Hit count 单独维护为 `hash -> hit_count_total`。只有 `match_external_kv(..., count_as_hit=True)` 且实际命中的 hash 才会增加计数；未命中的 hash 不计数，查询时也不会返回空记录。

### 实现边界

External KV RPC 只更新和查询 indexer 的 metadata index，不进入 data-plane 路径。`report` / `revoke` 更新 placement，`match` 查询 placement，`get_external_kv_hit_counts` 查询命中统计。

## Indexer 编程实现语言选择

Indexer 位于 metadata 控制面核心路径，语言选择需要兼顾 RPC 并发、低延迟查询、后台任务和长期维护成本。

| 语言 | 优点 | 不足 |
| --- | --- | --- |
| Python | 开发快，适合原型和测试工具 | 高并发、低延迟服务受 GIL 和运行时开销影响 |
| C++ | 性能强，生态成熟 | 内存安全和并发正确性维护成本高 |
| Rust | 性能接近 C++，内存和线程安全更好，async / gRPC / 存储生态成熟 | 学习曲线更高 |

推荐使用 **Rust** 实现 indexer。它能提供接近 C++ 的性能，同时通过类型系统降低 shared metadata map / counter 并发读写中的内存和数据竞争风险。首期可以用 Rust 实现 in-memory store，并保留 store trait，后续扩展 Redis、MySQL / PostgreSQL 或 RocksDB 后端。

## Redis backend 实现现状与性能

上文的 in-memory store 是首期形态。UMBP 侧已经落地了一个 RESP/Redis metadata backend（ROCm/mori PR #468），把 master 变成 stateless：store 里的状态是心跳投影出的可重建软状态，因此磁盘持久化是可选项，和副本 HA 相互独立。indexer 后续接入 external store 时可以直接复用这套实现和结论。

**选型速览**（详细数字与推理见下方「性能对比」与「关键结论」）：

| 后端 | 读 / 写吞吐（store 直连峰值） | 横向扩展 | HA | 定位 |
|---|---|---|---|---|
| in-memory | 5.5M / ~128k | `index_shards` | 无 | 性能上限参照，无持久 / HA |
| redis 多实例 | 24.5k / 18k | 加实例 | 无 | 吞吐 / 核最高，但无 HA / 容错 |
| **redis cluster** | ≈ 多实例 85–90% / ~15k | 加 master | 跨机自动 failover | **高可用首选**，性能约多实例的 85–90% |
| dragonfly 单实例 | 读随 proactor 扩展 / 写塌陷 ~0.4k | `proactor_threads` | 无 | 只适合读极重、写极轻 |

### 实现现状

后端在 client-agnostic 的 `IRespClient` seam 之下，通过 `UMBP_METADATA_BACKEND=redis` + `UMBP_REDIS_URI` 选择三种部署：

| 部署 | 选择方式 | 客户端 | 说明 |
| --- | --- | --- | --- |
| 单节点 Redis / Dragonfly | `UMBP_REDIS_URI` | hiredis `RespClient` | 单端口，读靠分片喂满 Dragonfly proactor |
| 多实例 | `UMBP_REDIS_SHARD_URIS`（逗号分隔，一实例一 shard） | hiredis `RespClient` | 每个 block shard 落在独立 server 进程上 |
| Redis Cluster | `UMBP_REDIS_CLUSTER=1` + seed 列表 | redis-plus-plus `RespClusterClient` | 副本 + 自动 failover，跨机 HA |

横向扩展靠三个旋钮：

- **分片**：block keyspace 按 hash 切成多个 shard（`UMBP_REDIS_BLOCK_SHARDS`）；
- **多端点**：每个 shard 落到独立 server（`UMBP_REDIS_SHARD_URIS`），突破单实例单线程上限；
- **并发读**：worker pool 把一次 `BatchLookup` 向各实例并发下发（pipeline），只花一趟 round trip。

External-KV 全套已落地（report / revoke / revoke-all、`match`（含 `count_as_hit` 命中计数）、get_hit_counts、hit GC、eviction 候选枚举），extkv / hit 已按 hash 分片、不再挤在单一 control slot，tier-set 用 per-node bitmask。原子性靠 Lua 脚本（EVALSHA），时间戳由调用方传入保证跨后端确定性，每个 store op 的耗时经 `mori_umbp_store_op_latency_seconds` 打点。

### 高可用与容错

- **Redis Cluster**：redis-plus-plus `RedisCluster`，副本 + 自动 failover，`MOVED` / `ASK` / reshard 全交客户端；启动时按 `CLUSTER SLOTS` 做 balanced per-master shard 放置。已实测「kill master → 副本升主」「live reshard 容忍」后一致性 / bench 仍过。
- **读容错**：某 shard 实例挂 → 该 key 记 miss、不拖垮整批读；单端点模式保持严格，让整体宕机可见而非伪装成全 miss。
- **写容错**：某 shard 挂 → `ApplyHeartbeat` 返回 `SEQ_GAP` → master 转 full_sync → peer 重发快照自愈（block ADD/REMOVE 幂等）。
- **启动就绪**：factory 启动时 Ping 每个 endpoint，`UMBP_REDIS_REQUIRED`（默认 true）控制 fail-fast 或降级启动。

### 性能对比

数字均取自 store 直连 microbench（`bench_umbp_master_metadata_store` / `_extkv`，无 gRPC/RDMA）与全链路 8 进程 proxy；batch=32、keys=50000，单机多核独占、loopback，glibc redis 7.2.5 / dragonfly v1.23.2。单元格为 `ops/s (p50 ms)`，同列 client 线程数 `t`（施加负载）一致，行间差距即后端本身。

**主元数据 · store 直连 · RouteGet（读热路径，含 lease 写记账）**

| 模式 | 内部配置 | t8 | t16 | t32 | t64 | t128 | t256 |
|---|---|---|---|---|---|---|---|
| in-memory | `index_shards=256` | 620,399 (0.01) | 1,082,374 (0.01) | 1,958,402 (0.02) | 3,271,864 (0.02) | 4,316,796 (0.03) | 5,106,055 (0.04) |
| redis 多实例 | 8 实例, pool=128 | 25,356 (0.3) | 26,445 (0.6) | 27,194 (1.2) | 26,958 (2.4) | 26,892 (4.7) | 26,727 (9.6) |
| redis cluster | 8主 + 8从, balanced | 20,909 (0.4) | 21,790 (0.7) | 21,174 (1.6) | 22,681 (2.8) | 21,048 (6.1) | 21,638 (11.7) |
| dragonfly 单实例 | `proactor_threads=32, block_shards=64`, pool=256 | 5,430 (1.4) | 10,391 (1.5) | 18,914 (1.6) | 28,645 (2.1) | 38,252 (3.2) | 43,894 (5.2) |

**主元数据 · store 直连 · Heartbeat（写热路径）**

| 模式 | 内部配置 | t8 | t16 | t32 | t64 | t128 |
|---|---|---|---|---|---|---|
| in-memory | `index_shards=256` | 88,375 (0.05) | 121,136 (0.06) | 147,339 (0.15) | 119,426 (0.4) | 99,950 (1.0) |
| redis 多实例 | 8 实例 | 14,139 (0.5) | 18,634 (0.8) | 19,332 (1.5) | 20,948 (2.9) | 20,644 (6.1) |
| redis cluster | 8主 + 8从 | 11,212 (0.7) | 13,197 (1.1) | 12,456 (2.3) | 15,423 (4.1) | 15,195 (8.4) |
| dragonfly 单实例 | proactor=32 | 403 (19.8) | 400 (40.0) | 402 (79.5) | 392 (163) | 402 (318) |

**主元数据 · 全链路 8 进程 proxy**（`umbp_master` + Router/gRPC，8 客户端进程打同一 master）：`GETMODE=exists` 单元格 = `get QPS / BatchLookup p50 (ms)`；`GETMODE=both`（真读热路径 `route_get_batch` + 一次 RDMA 取数）单元格 = `get QPS / BatchRouteGet p50 (ms)`（both 每轮计 lookup+fetch ≈ 2 次，只在本表内比）。

| 模式 | exists 8c | exists 32c | exists 64c | both 8c | both 32c |
|---|---|---|---|---|---|
| in-memory | 24,969 / 0.50 | 62,986 / 0.50 | 81,320 / 0.51 | 40,827 / 0.50 | 66,377 / 0.50 |
| redis 多实例 | 9,444 / 0.51 | 4,455 / 1.67 | 1,388 / 30.5 | 13,206 / 0.50 | 8,843 / 1.81 |
| redis cluster | 9,381 / 0.51 | 3,829 / 1.90 | 1,344 / 18.7 | 11,664 / 0.50 | 7,464 / 2.54 |
| dragonfly 单实例 | 863 / 3.06 | 653 / 17.5 | 393 / 67.6 | 1,740 / 2.73 | 1,474 / 13.17 |

**External-KV · store 直连**（`bench_umbp_master_metadata_store_extkv`，batch=32 / keys=50000 / nodes=8）。`match` = `MatchExternalKv(count_as_hit=true)`（读 + 每-hash 命中计数写），`match_nohit` 隔离纯读，`report` = `RegisterExternalKvIfAlive`（写）。cluster = 8主+8从（16 进程，8 个服务）。

`match`（读 + 命中计数写）：

| 模式 | t16 | t32 | t64 |
|---|--:|--:|--:|
| in-memory | 54,684 (0.02) | 54,652 (0.02) | 54,355 (0.02) |
| redis 多实例 | 33,116 (0.5) | 29,814 (1.1) | 31,839 (2.0) |
| redis cluster | 26,901 (0.6) | 27,966 (1.1) | 27,344 (2.3) |
| redis 单实例 | 8,216 (2.3) | 8,289 (3.2) | 8,098 (7.2) |
| dragonfly | 3,422 (4.3) | 5,559 (5.3) | 10,275 (5.9) |

dragonfly 过 t64 仍在爬：t128 9,313 (12.6) → t256 11,134 (15.9)。

`match_nohit`（纯读，不写命中）：

| 模式 | t16 | t32 | t64 |
|---|--:|--:|--:|
| in-memory | 1,673,248 (0.01) | 3,269,436 (0.01) | 3,909,725 (0.01) |
| redis 多实例 | 46,754 (0.3) | 49,644 (0.6) | 48,737 (1.3) |
| redis cluster | 44,761 (0.3) | 44,220 (0.7) | 44,514 (1.4) |
| redis 单实例 | 13,763 (1.4) | 13,800 (2.0) | 13,691 (4.3) |
| dragonfly | 14,296 (1.1) | 24,227 (1.2) | 40,641 (1.5) |

dragonfly：t128 54,933 (2.1) → t256 63,113 (3.8)。

`report`（写）：

| 模式 | t16 | t32 | t64 |
|---|--:|--:|--:|
| in-memory | 27,469 (0.03) | 27,228 (0.03) | 26,092 (0.03) |
| redis 多实例 | 29,769 (0.5) | 28,869 (1.0) | 26,327 (2.2) |
| redis cluster | 21,834 (0.7) | 23,442 (1.3) | 25,149 (2.6) |
| redis 单实例 | 8,614 (1.8) | 8,660 (2.9) | 10,063 (6.2) |
| dragonfly | 4,389 (3.4) | 8,208 (3.6) | 11,265 (5.4) |

dragonfly：t128 13,968 (8.9) → t256 11,667 (20.7)（已过拐点）。

**性能要点与根因**（选型建议见节首「选型速览」，此处只讲各后端的性能特征与原因）：

- **in-memory**：纯读差距最大（`match_nohit` 随核数爬到 3.9M，比 redis 高 45~280×），但一旦带写，全局写锁把 `match`/`report` 在 t16 就压到和 redis 同一个几万量级（读 5.5M / 写 ~128k）；无 stateless master / HA / 持久，只作上限参照。
- **redis 多实例**：吞吐/核最高（8 进程 ~24.5k ≈ 3k/核），读写都稳（读 24.5k / 写 18k），t64 即饱和；但自身无副本、无 failover（无 HA），需客户端 fan-out。
- **redis cluster**：随 master 数近线性扩展（读 3→9.7k / 6→18.6k / 8→22k ≈ 同节点数多实例的 85–90%；8 主写 ~15k），并自带副本 + 跨机自动 failover——有高可用要求时的形态（代价是 per-command slot 校验 + redis++ MOVED/ASK 路由，evalsha ~73µs vs ~26µs）。
- **dragonfly 单实例**：单端口读随 proactor 扩展（P32 43.7k → P48 ~51k → P64 ~53k，收益递减），但心跳写走 store-wide 全局锁 → 塌陷到 ~0.4k。只适合读极重、写/心跳极轻。
- redis 相对 in-memory 慢的根因 = 每 op 一次网络 RTT + RESP 序列化 + Lua 脚本 + 单 slot 单线程执行；而 in-memory 是进程内内存访问、且读可并发（`shared_lock`）。

> 复现命令见来源文档（占位符 `CTR=umbp-app`、`DEV=/path/to/workdir`），此处不重复。主元数据详表见 `dev/perf_cmp/PERF-COMPARISON-{zh,en}.md`，External-KV 详解见 `dev/perf_cmp/EXTKV-PERF.md`。

## 附录：UMBP Feature Proposal

UMBP 已经具备一组可复用的 KV backend、分层存储、路由和 metadata 能力。SGLang Indexer Bridge 主要复用其中的 External KV metadata / hit count 思路，并将其抽取为独立 indexer 组件。

| Feature | 相关组件 / API | 简明说明 |
| --- | --- | --- |
| 本地 `Put` / `Get` | `put_from_ptr()` / `get_into_ptr()`、local storage manager | 默认在本地 tier 内完成 KV block 写入和读取；同一个 key 可作为 content-addressed block 做 dedup。 |
| 分布式 `Put` / `Get` | route get / route put、peer service | 本地 miss 后可查询 remote placement，并从远端节点读取或写入 KV block；这是 bytes path，不属于本 RFC 的 bridge 范围。 |
| 批量操作 | batch put / get / exists | 面向 prefix cache / block cache 的批量访问模式，减少逐 block RPC 或查询开销。 |
| 分层存储 | HBM / DRAM / SSD tier | KV block 可以分布在不同速度和容量的 tier 上；调度时可优先选择更快 tier。 |
| Global / Local Block Index | global placement index、local hint index | 全局 index 维护跨节点 placement，本地 index 维护快速命中 hint。 |
| Global Routing | route get / route put strategy | 根据 placement、节点状态和 tier 信息选择读源或写入目标；External KV 的 `match` 结果可作为调度输入。 |
| External KV metadata | `report` / `revoke` / `match` | 允许外部系统上报自己持有的 KV placement；indexer 只记录 metadata，用于跨 worker KV 发现。 |
| Hit count primitive | `match(..., count_as_hit=True)`、hit-count query | 将真实路由查询命中累计成 per-hash 热度信号，供 scheduler 或 bridge 后续策略使用。 |
| Agent Hints | `session_id`、semantic spans、admission action | 后续可把 session 生命周期和语义阶段传给 scheduler / indexer，用于 sticky routing、admission 和 metadata 清理。 |
