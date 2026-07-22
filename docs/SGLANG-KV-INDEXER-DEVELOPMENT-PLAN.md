# SGLang KV Indexer 开发计划

## 目标

MVP 采用 out-of-tree Rust 项目实现，先完成 block-level KV placement metadata 的收集、更新和查询能力。

Indexer 只回答某些 KV block hash 位于哪些 worker 和 tier。MVP 不实现 SGLang 侧 longest-prefix helper，也不在 Indexer 内部构建 radix tree。

## 阶段 1：定义并实现 gRPC API

把 RFC 里的 API 设计写成正式 `.proto` 文件。`.proto` 文件相当于服务端和客户端共同遵守的接口合同，会明确有哪些 RPC、每个 RPC 的请求字段、返回字段、字段类型和字段编号。

Rust 侧可以使用 `tonic` 读取这个 `.proto` 文件，并自动生成 gRPC 服务端和客户端的基础代码，包括请求 / 响应结构、客户端调用 stub、服务端 trait，以及序列化 / 反序列化逻辑。开发时只需要在生成的服务端框架里填写业务逻辑。

- `ApplyExternalKvBatch`
- `MatchExternalKv`
- `GetExternalKvHitCounts`

- 定义 `TierType`、`ExternalKvAction`、各 RPC request / response、`ExternalKvNodeMatch`、`TierHashes` 和 `HitCountEntry`。
- `ApplyExternalKvBatch` 是唯一写入口，按 SGLang 事件原始顺序应用 report、revoke 和 clear-at-tier action。
- `MatchExternalKvRequest.hashes` 表示查询的 hash 列表。MVP 阶段 Indexer 只做 block-level placement match，不计算 longest prefix。
- `MatchExternalKvResponse` 按 worker 和 tier 返回实际匹配到的 hashes。
- `count_as_hit=true` 时，只统计实际匹配到的 hash；未匹配 hash 不计数。
- API 不传输 KV cache bytes，只传输 metadata。

gRPC handler 和内部 metadata store 的关系：

```text
ApplyExternalKvBatch
  -> ordered [
       PlacementIndex::register,
       PlacementIndex::unregister,
       PlacementIndex::unregister_by_worker_at_tier,
       ...
     ]

MatchExternalKv
  -> PlacementIndex::match_hashes

GetExternalKvHitCounts
  -> HitCountIndex::lookup
```

实际开发时，可以先完成 `.proto` 和 gRPC server 框架；等阶段 2 的 metadata store 完成后，再把 handler 接到 `PlacementIndex` 和 `HitCountIndex`。

验收标准：

- proto 可以被 Rust `tonic` 正常生成代码。
- server、client、integration test 都基于同一份 proto。
- 注释明确说明 MVP 不计算 longest prefix。
- gRPC server 可以启动和关闭。
- 所有 MVP RPC 可以被本地 client 调通。
- `MatchExternalKv` 返回每个 worker / tier 实际匹配到的 hashes，不计算 longest prefix。
- `count_as_hit` 行为与 UMBP 一致。

## 阶段 2：实现 metadata store

实现 Indexer 最核心的内存 metadata store。它包含两部分：`PlacementIndex` 记录 hash 在哪里，`HitCountIndex` 记录哪些 hash 被查询命中过。

`PlacementIndex` 对齐 UMBP External KV index：

```text
hash -> worker_id -> set<tier>
```

`PlacementIndex` 核心操作：

- `register(worker_id, hashes, tier) -> mutated_count`
- `unregister(worker_id, hashes, tier) -> mutated_count`
- `unregister_by_worker_at_tier(worker_id, tier) -> mutated_count`
- `unregister_by_worker(worker_id) -> mutated_count`
- `match_hashes(hashes) -> Vec<NodeMatch>`

`HitCountIndex` 核心操作：

- `increment_hits(hashes)`
- `lookup_hit_counts(hashes) -> Vec<HitCountEntry>`

要求：

- 重复上报同一个 `(hash, worker_id, tier)` 不重复计数。
- 同一个 hash 可以存在于多个 worker 和多个 tier。
- revoke 只删除指定 tier 的 placement。
- 当某个 hash 不再属于任何 worker / tier 时，应从 index 中清理。
- hit count 只统计实际匹配到的 hash，未匹配的 hash 不计数。
- revoke 不删除 hit count；hit count 是历史统计。

验收标准：

- 单元测试覆盖 duplicate report、multi-worker、multi-tier、partial revoke、revoke all at tier。
- 单元测试覆盖 hit count increment、重复查询去重和 lookup。
- `PlacementIndex` 和 `HitCountIndex` 可独立于 gRPC 测试。

## 阶段 3：实现本地 test client / integration test

在没有 SGLang Bridge 的情况下，先验证 Indexer 自身闭环。

最小测试流程：

```text
report worker-a HBM h1,h2,h3
report worker-b DRAM h1,h2
match h1,h2,h3,h4
revoke worker-a HBM h2
match h1,h2,h3
get-hit-counts h1,h2,h3
```

验收标准：

- 可以自动化验证 `report -> match -> revoke -> match`。
- gRPC 序列化和返回结构正确。
- 覆盖多 worker、多 tier、部分 revoke 和 hit count。

## 阶段 4：实现 SGLang Bridge sidecar

让 Indexer 的 metadata update 来自真实或模拟的 SGLang KV events。

- 订阅 SGLang ZMQ `KVEventBatch`。
- 解码 `BlockStored`、`BlockRemoved`、`AllBlocksCleared`。
- 将 SGLang medium 映射到 Indexer tier。
- 按事件原始顺序构造 action，并通过一次 `ApplyExternalKvBatch` 调用提交完整 batch。
- 记录 sequence gap、decode error、RPC failure。
- RPC 失败时保留未确认的 batch，重连后先重试该 batch。
- 如果配置了 SGLang replay endpoint，检测到 sequence gap 时先 replay 缺失 batch，再处理当前 batch。

事件映射：

```text
BlockStored(block_hashes, medium)
  -> ACTION_REPORT(hashes, tier)

BlockRemoved(block_hashes, medium)
  -> ACTION_REVOKE(hashes, tier)

AllBlocksCleared()
  -> ACTION_CLEAR_ALL_AT_TIER(tier)

所有 action 保持顺序装入同一个 ApplyExternalKvBatch。
```

验收标准：

- Bridge 可以将 store/remove/clear 事件按序转换为一个 `ApplyExternalKvBatch`。
- 支持至少单 worker 的真实或模拟事件回放测试。
- 出现 sequence gap 或 RPC 失败时有可观测日志，并能在 replay buffer 覆盖范围内恢复。

## 阶段 5：端到端验证

确认 SGLang 事件能够正确更新 Indexer metadata，并能通过 match 查询到结果。

```text
1. 启动 Rust Indexer。
2. 启动 SGLang worker 或事件模拟器。
3. 启动 Bridge sidecar。
4. 触发 KV block store 事件。
5. Bridge report 到 Indexer。
6. 使用 test client 调用 MatchExternalKv。
7. 触发 KV block remove / clear 事件。
8. 再次调用 MatchExternalKv 验证 metadata 被撤销。
```

验收标准：

- 真实或模拟 SGLang KV events 可以驱动 Indexer metadata 更新。
- match 查询结果与事件输入一致。
- revoke / clear 后 stale metadata 被清理。
- 端到端 demo 可以稳定复现。

## 阶段 6：可靠性和性能测试

验证 MVP 是否足以支撑后续集成，并为是否引入 radix tree、Redis 或分片提供依据。

可靠性测试：

- 重复 report / revoke。
- revoke 不存在的 hash。
- event 乱序、丢失或 sequence gap。
- Bridge / Indexer 重启。
- worker crash。
- gRPC 失败和重试。

性能测试：

- 单次 match 的 hash 数量、worker 数量、总 hash 数量。
- report / revoke QPS。
- match P50 / P95 / P99 latency。
- 内存占用。

验收标准：

- 给出 MVP 的延迟、吞吐和内存数据。
- 明确下一阶段是否需要优化 index 数据结构、持久化后端或 radix tree。

## 推荐执行顺序

1. 定义并实现 gRPC API，包括 `kv_indexer.proto`、`tonic` 生成代码和 service handlers。
2. 实现 metadata store，包括 `PlacementIndex` 和 `HitCountIndex`。
3. 写本地 test client / integration test，验证 `report -> match -> revoke -> match`。
4. 写 SGLang Bridge sidecar，把 KV events 转成 Indexer RPC。
5. 做端到端验证，确认 SGLang 事件能正确更新 metadata。
6. 做可靠性和性能测试。

## MVP 完成标准

- Rust Indexer 可以独立运行。
- Indexer 支持 report / revoke / match / hit-count API。
- Indexer 内部维护 UMBP-style block placement metadata。
- 本地 integration test 可以验证完整 metadata 操作闭环。
- Bridge 可以从 SGLang KV events 更新 Indexer metadata。
- 端到端验证可以证明事件驱动的 metadata 更新和查询是正确的。

MVP 不要求：

- SGLang 已经完成 longest-prefix helper。
- Scheduler 已经完成基于 prefix length 的 routing。
- Indexer 内部已经实现 radix tree。
- Metadata 已经持久化到 Redis、RocksDB 或 SQL。
