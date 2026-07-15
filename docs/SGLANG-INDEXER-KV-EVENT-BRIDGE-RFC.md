# RFC: SGLang Indexer KV Event Bridge

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

## 附录：UMBP Feature Proposal

UMBP 已经具备一组可复用的 KV backend、分层存储、路由和 metadata 能力。SGLang Indexer Bridge 主要复用其中的 External KV metadata / hit count 思路，并将其抽取为独立 indexer 组件。

| Feature | 相关组件 / API | 简明说明 |
| --- | --- | --- |
| 本地 `Put` / `Get` | `put_from_ptr()` / `get_into_ptr()`、local storage manager | 默认在本地 tier 内完成 KV block 写入和读取；同一个 key 可作为 content-addressed block 做 dedup。 |
| 分布式 `Put` / `Get` | route get / route put、peer service | 本地 miss 后可查询 remote placement，并从远端节点读取或写入 KV block；这是 bytes path，不属于本 RFC 的 bridge 范围。 |
| 批量操作 | batch put / get / exists | 面向 prefix cache / block cache 的批量访问模式，减少逐 block RPC 或查询开销。 |
| 分层存储 | HBM / DRAM / SSD tier | KV block 可以分布在不同速度和容量的 tier 上；调度时可优先选择更快 tier。 |
| Global / Local Block Index | global placement index、local hint index | 全局 index 维护跨节点 placement，本地 index 维护快速命中 hint；独立 indexer 会复用类似的 metadata 建模方式。 |
| Global Routing | route get / route put strategy | 根据 placement、节点状态和 tier 信息选择读源或写入目标；External KV 的 `match` 结果可作为调度输入。 |
| External KV metadata | `report` / `revoke` / `match` | 允许外部系统上报自己持有的 KV placement；indexer 只记录 metadata，用于跨 worker KV 发现。 |
| Hit count primitive | `match(..., count_as_hit=True)`、hit-count query | 将真实路由查询命中累计成 per-hash 热度信号，供 scheduler 或 bridge 后续策略使用。 |
| Agent Hints | `session_id`、semantic spans、admission action | 后续可把 session 生命周期和语义阶段传给 scheduler / indexer，用于 sticky routing、admission 和 metadata 清理。 |
