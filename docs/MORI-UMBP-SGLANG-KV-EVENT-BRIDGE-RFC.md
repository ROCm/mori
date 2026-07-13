# RFC: SGLang Index KV Event Bridge

## 背景

SGLang / HiCache 会通过 ZMQ 发布 KV cache 事件，例如 block 写入、删除和整体清空。Indexer 需要感知这些外部 KV cache 的 placement metadata，以便后续进行跨 worker 的 KV 查询、调度和命中统计。

`SglangIndexKvEventBridge` 是一个独立 bridge / sidecar。它只同步 metadata，不搬运 KV bytes，也不把 external KV 纳入 indexer-owned 数据面。

## 目标

- 订阅 SGLang ZMQ `KVEventBatch`。
- 将 `BlockStored` / `BlockRemoved` / `AllBlocksCleared` 转换成 indexer External KV API 调用。
- 支持单 worker 和多 worker / DP rank 事件源。
- 维护基础观测信息，例如 sequence gap、decode error、report / revoke 失败计数。

## 非目标

- 不实现 KV bytes 的 put / get。
- 不调用 indexer-owned `Put`、`PublishLocalBlock` 或 data-plane resolve 路径。
- 不在首期处理 SGLang `EXTERNAL` medium 的跨系统语义。

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
|     Metadata Store     |<-|            Index           |
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

整体链路由四个组件组成：`SGLang` 负责产生 KV cache 生命周期事件，`Bridge` 负责订阅、解码和转换事件，`Index` 负责提供 External KV metadata 的写入和查询接口，`Metadata Store` 负责保存 placement 与 hit-count 等状态。Bridge 不参与 KV bytes 的读写，只把 SGLang 侧的 block 变化同步为 indexer 可查询的 metadata。

数据流是单向写入为主：SGLang 通过 ZMQ 推送 `BlockStored`、`BlockRemoved`、`AllBlocksCleared`，Bridge 将其转换成 report / revoke / revoke-all 请求写入 Index；调度器或查询方后续通过 Index 查询哪些节点持有某些 hashes，以及这些 hashes 的累计命中情况。

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

Bridge 对事件流采用 best-effort 同步模型。ZMQ 暂时无消息时不视为错误；decode 失败、unknown medium、sequence gap 和 indexer RPC 失败都会进入统计。`AllBlocksCleared` 需要尽量收敛 metadata，因此应对 revoke-all 做有限重试，失败后保留错误计数，交给上层监控或后续补偿流程处理。

## Indexer RPC API 设计

`SglangIndexKvEventBridge` 调用的是 indexer 的 External KV metadata API。这组 API 通过 protobuf / gRPC 暴露在 indexer service 中；在本 RFC 中可以把这组接口理解为 indexer RPC API。它们只维护 placement metadata，不代表 indexer 拥有 KV bytes。

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

### Index 实现模型

External KV placement index 可以抽象为 `hash -> node_id -> set<tier>`。因此一个 hash 可以同时存在于多个 node，也可以在同一个 node 的多个 tier 上存在副本。`match_external_kv` 返回时按 node 聚合，并把每个 node 命中的 hashes 再按 tier 分桶。

Hit count 单独维护为 `hash -> hit_count_total`。只有 `match_external_kv(..., count_as_hit=True)` 且实际命中的 hash 才会增加计数；未命中的 hash 不计数，查询时也不会返回空记录。

### Server 行为

| RPC | 行为摘要 |
| --- | --- |
| `ReportExternalKvBlocks` | 校验 `node_id` 和 `hashes`，写入 External KV placement index |
| `RevokeExternalKvBlocks` | 删除指定 `(hash, node_id, tier)` |
| `RevokeAllExternalKvBlocksAtTier` | 删除该 node 在指定 tier 的全部 external KV placement |
| `MatchExternalKv` | 查询 placement index，补充 alive peer 的 `peer_address`，必要时累加 hit count |
| `GetExternalKvHitCounts` | 查询 hit index，并限制单次查询 batch 大小 |

这组 RPC 与 indexer-owned data-plane API 分离。`PublishLocalBlock`、`Put`、`RouteGet`、`ResolveKey` 表示 indexer-owned KV 数据路径；External KV RPC 只表示外部系统持有 KV bytes，indexer 记录可用于调度的 metadata。

## Indexer 编程实现语言选择

Indexer 位于 metadata 控制面核心路径，语言选择需要兼顾 RPC 并发、低延迟查询、后台任务和长期维护成本。

| 语言 | 优点 | 不足 |
| --- | --- | --- |
| Python | 开发快，适合原型和测试工具 | 高并发、低延迟服务受 GIL 和运行时开销影响 |
| C++ | 性能强，生态成熟 | 内存安全和并发正确性维护成本高 |
| Rust | 性能接近 C++，内存和线程安全更好，async / gRPC / 存储生态成熟 | 学习曲线更高 |

推荐使用 **Rust** 实现 indexer。它能提供接近 C++ 的性能，同时通过类型系统降低 shared metadata map / counter 并发读写中的内存和数据竞争风险。首期可以用 Rust 实现 in-memory store，并保留 store trait，后续扩展 Redis、MySQL / PostgreSQL 或 RocksDB 后端。
