# PoolClient — 数据面集成设计文档

**作者:** Dev3
**状态:** 草稿
**范围:** PoolClient（控制面 + 数据面）、PeerService（SSD 协调）、PoolAllocator
**依赖:** `design-master-control-plane.md`（MasterClient、MasterServer、BlockIndex、Router、ClientRegistry）

---

## 1. 概述

`PoolClient` 是构建在 `MasterClient` 之上的高层客户端类。它将 Master 的路由决策（控制面）与
MORI IO Engine 的数据传输（数据面）相结合，向上层调用者提供简洁的 Put/Get API，将数据存储到
集群中任意节点注册的 DRAM 或 SSD 空间。

**核心组件：**

- **MasterClient** — 控制面：RoutePut/RouteGet 路由决策，Register/Unregister 元数据管理，
  Heartbeat 心跳保活。
- **MORI IO Engine** — 数据面：RDMA（生产环境）或 TCP（测试/开发环境）实际块传输。
- **PeerService** — 每个提供存储的节点上运行的轻量 gRPC 服务，负责 IO Engine 握手和 SSD 协调。

**四种存储路径**（由 RoutePut 返回的 `node_id` 和 `tier` 决定）：

- **本地 DRAM**：目标是本节点，tier=DRAM。直接 `memcpy` 到本地 DRAM 缓冲区，无需 RDMA。
- **本地 SSD**：目标是本节点，tier=SSD。直接 PosixFile 写入本地 SSD，无需 RDMA。
- **远端 DRAM**：目标是其他节点，tier=DRAM。纯 RDMA 单边读写。Master 集中管理
  DRAM 偏移量分配，RoutePut 原子地选择节点并分配偏移量。
- **远端 SSD**：目标是其他节点，tier=SSD。双阶段传输——先 RDMA 写入远端 DRAM staging 区，
  再通过 PeerService RPC 让远端用 PosixFile 持久化到 SSD。

---

## 2. 架构

```
                                       ┌─────────────────────────────────┐
                                       │       UMBP Master Server        │
  ┌────────────────────┐               │  (纯控制面)                      │
  │    本地节点          │   gRPC        │                                 │
  │                    │               │  ClientRegistry                 │
  │  ┌──────────────┐  │  RoutePut     │    + peer_address               │
  │  │  PoolClient  │──────────────────►    + engine_desc                │
  │  │              │  │  RouteGet     │    + dram_memory_desc           │
  │  │  ┌─────────┐ │◄─────────────────    + PoolAllocator per node/tier │
  │  │  │ Master  │ │  │  Register     │                                 │
  │  │  │ Client  │ │  │  Unregister   │  BlockIndex                     │
  │  │  └─────────┘ │  │               │  Router                        │
  │  │              │  │               └─────────────────────────────────┘
  │  │  ┌─────────┐ │  │
  │  │  │IO Engine│ │  │  RDMA          ┌─────────────────────────────────┐
  │  │  │         │─────────────────────►  远端节点 B                     │
  │  │  └─────────┘ │  │               │                                 │
  │  │              │  │  gRPC          │  ┌───────────────────────────┐  │
  │  │  PeerService ├──────────────────►│  │ PeerService               │  │
  │  │  stub        │  │  (SSD 操作    │  │  GetPeerInfo              │  │
  │  └──────────────┘  │   + 握手)     │  │  CommitSsdWrite           │  │
  │                    │               │  │  PrepareSsdRead            │  │
  └────────────────────┘               │  └───────────────────────────┘  │
                                       │                                 │
                                       │  RDMA 注册的 DRAM 缓冲区         │
                                       │  SSD 存储 (PosixFile)           │
                                       └─────────────────────────────────┘
```

---

## 3. 关键设计决策

### 3.1 Master 通过 PoolAllocator 集中管理各 Tier 分配

Master 集中管理所有节点的**所有 tier**（DRAM 和 SSD）的容量与分配。当 `RoutePut` 选中目标节点时，
Master 原子地从该节点对应 tier 的分配器中扣减容量。这保证了：

- **一致性**：Master 对每个 tier 的可用容量视图始终准确，不会因心跳间隔导致信息滞后。
- **原子性**：节点选择和容量预留在单次 RPC 中完成，无竞态。
- **最少 RPC**：DRAM Put 只需 2 次 gRPC（RoutePut + Register）加 1 次 RDMA 传输。

统一的 `PoolAllocator` 用于两种 tier，行为差异如下：

- **DRAM**：管理**偏移量分配 + 容量跟踪**。分配的偏移量在 `RoutePut` 中返回，用于 RDMA 寻址。
- **SSD**：仅管理**容量跟踪**。SSD 寻址基于文件（`{key}.bin`），由 PeerService 处理。
  分配器返回的偏移量在外部不使用。

```cpp
struct PoolAllocator {
    uint64_t total_size = 0;
    uint64_t used_size = 0;

    // 偏移量管理（DRAM 使用；SSD 设为 nullopt）
    struct OffsetTracker {
        uint64_t bump = 0;
        std::vector<std::pair<uint64_t, uint64_t>> free_list;  // {offset, size}
    };
    std::optional<OffsetTracker> offset_tracker;

    // DRAM: 从 offset_tracker 分配偏移量 + 扣减 used_size。
    // SSD: 仅扣减 used_size（返回 0）。
    // 容量不足时返回 nullopt。
    std::optional<uint64_t> Allocate(uint64_t size);

    // DRAM: 归还偏移量到 free_list + 加回 used_size。
    // SSD: 仅加回 used_size（offset 参数被忽略）。
    // 相邻空闲块可合并（仅 DRAM）。
    void Deallocate(uint64_t offset, uint64_t size);

    uint64_t AvailableBytes() const { return total_size - used_size; }
};
```

每个节点在 `ClientRecord` 中有一个 `std::map<TierType, PoolAllocator>`。
当块被 `Unregister` 时，Master 在对应 tier 的分配器上调用 `Deallocate` 回收容量
（DRAM 还会回收偏移量）。

### 3.2 PeerService 负责 SSD 操作和 IO Engine 握手

每个提供存储的节点运行一个轻量 gRPC `PeerService`，有两个职责：

1. **IO Engine 握手**（`GetPeerInfo`）：首次连接时，`PoolClient` 调用 `GetPeerInfo` 获取远端
   节点的 `EngineDesc` 和 `MemoryDesc`。用于注册远端 IO Engine 并建立 RDMA 连接。
   每个 peer 一次性成本。

2. **SSD 协调**：SSD 不在 RDMA 可访问范围内。写入远端 SSD 需要先通过 RDMA 将数据写入远端节点
   的 DRAM staging 区，然后请求远端 PeerService 通过 PosixFile 将数据持久化到 SSD。从远端 SSD
   读取则相反：请求 PeerService 将 SSD 数据加载到 DRAM staging，再通过 RDMA 读回。

Master **不**管理 SSD 偏移量分配。SSD 写入产生的 `location_id`（如文件路径）由 PeerService
生成，Master 不透明地存储。

### 3.3 MORI IO Engine 作为数据面

`PoolClient` 使用 `mori::io::IOEngine` 进行所有 RDMA 传输。IO Engine 同时支持 RDMA（生产）
和 TCP（测试/开发）后端，API 完全一致。

每个节点分配并注册一个本地 staging 缓冲区用于 RDMA 传输：
- 发送前暂存数据（调用者的源指针可能不在 RDMA 注册内存中）。
- 接收后暂存数据，再拷贝到调用者的目标地址。

---

## 4. Proto 扩展

### 4.1 Master Proto（`umbp.proto`）

支持 PoolClient 所需的最小扩展：

```protobuf
message RegisterClientRequest {
  string node_id                      = 1;
  string node_address                 = 2;
  repeated TierCapacity tier_capacities = 3;
  // --- 新增字段 ---
  string peer_address                 = 4;  // PeerService gRPC 地址
  bytes  engine_desc                  = 5;  // 打包的 EngineDesc
  bytes  dram_memory_desc             = 6;  // 打包的 MemoryDesc（RDMA 注册的 DRAM）
}

message RoutePutResponse {
  bool     found            = 1;
  string   node_id          = 2;
  string   node_address     = 3;
  TierType tier             = 4;
  // --- 新增字段 ---
  string   peer_address     = 5;  // 目标节点的 PeerService 地址
  bytes    engine_desc      = 6;  // 目标节点的 EngineDesc
  bytes    dram_memory_desc = 7;  // 目标节点的 MemoryDesc（DRAM）
  uint64   allocated_offset = 8;  // Master 分配的偏移量（仅 DRAM tier）
}

message RouteGetResponse {
  bool     found            = 1;
  Location source           = 2;
  // --- 新增字段 ---
  string   peer_address     = 3;  // 源节点的 PeerService 地址
  bytes    engine_desc      = 4;  // 源节点的 EngineDesc
  bytes    dram_memory_desc = 5;  // 源节点的 MemoryDesc（DRAM）
}
```

### 4.2 PeerService Proto（`umbp_peer.proto`）

```protobuf
syntax = "proto3";
package umbp;

service UMBPPeer {
  // IO Engine 握手（每个 peer 一次）
  rpc GetPeerInfo(GetPeerInfoRequest) returns (GetPeerInfoResponse);

  // SSD 协调
  rpc CommitSsdWrite(CommitSsdWriteRequest) returns (CommitSsdWriteResponse);
  rpc PrepareSsdRead(PrepareSsdReadRequest) returns (PrepareSsdReadResponse);
}

// --- 握手 ---
message GetPeerInfoRequest {}
message GetPeerInfoResponse {
  bytes  engine_desc         = 1;  // 打包的 EngineDesc
  bytes  dram_memory_desc    = 2;  // 打包的 MemoryDesc
  uint64 ssd_capacity        = 3;
  uint64 ssd_available       = 4;
  uint64 staging_base_offset = 5;  // SSD staging 区在 DRAM buffer 中的起始 offset
}

// --- SSD 写入：RDMA 写入 staging → 持久化到 SSD ---
message CommitSsdWriteRequest {
  string key              = 1;
  uint64 staging_offset   = 2;  // DRAM staging 区内的偏移量
  uint64 size             = 3;
}
message CommitSsdWriteResponse {
  bool   success          = 1;
  string ssd_location_id  = 2;  // SSD 上的不透明标识
}

// --- SSD 读取：从 SSD 加载到 staging → 调用方 RDMA 读取 ---
message PrepareSsdReadRequest {
  string key              = 1;
  string ssd_location_id  = 2;
  uint64 size             = 3;
}
message PrepareSsdReadResponse {
  bool   success          = 1;
  uint64 staging_offset   = 2;  // 数据已加载到的 DRAM staging 偏移量
}
```

---

## 5. ClientRecord 扩展

```cpp
struct ClientRecord {
    std::string node_id;
    std::string node_address;
    ClientStatus status = ClientStatus::UNKNOWN;
    std::chrono::steady_clock::time_point last_heartbeat;
    std::chrono::steady_clock::time_point registered_at;
    std::map<TierType, TierCapacity> tier_capacities;

    // --- PoolClient 集成新增字段 ---
    std::string peer_address;                     // PeerService gRPC 地址
    std::vector<uint8_t> engine_desc_bytes;       // 打包的 EngineDesc
    std::vector<uint8_t> dram_memory_desc_bytes;  // 打包的 MemoryDesc（DRAM）
    std::map<TierType, PoolAllocator> allocators; // per-tier 分配
    // DRAM: PoolAllocator（含 offset_tracker，管理偏移量 + 容量）
    // SSD:  PoolAllocator（无 offset_tracker，仅管理容量）
};
```

---

## 6. RoutePutStrategy 扩展

`RoutePutResult` 新增数据面信息：

```cpp
struct RoutePutResult {
    std::string node_id;
    std::string node_address;
    TierType tier;

    // --- 新增 ---
    std::string peer_address;
    std::vector<uint8_t> engine_desc_bytes;
    std::vector<uint8_t> dram_memory_desc_bytes;
    uint64_t allocated_offset = 0;  // 仅 DRAM tier 有效
};
```

`Router::RoutePut` 将策略选择和分配合并为一次调用，内部自动重试：

```
Router::RoutePut(key, node_id, block_size):
    candidates = registry_.GetAliveClients()
    loop:
        result = strategy.Select(candidates, block_size)
        if !result: return nullopt              // 所有候选已耗尽

        alloc = registry_.AllocateForPut(result.node_id, result.tier, block_size)
        if alloc:
            // 将分配信息合入 result（peer_address, engine_desc 等）
            return result

        // 分配失败 — 从候选中移除该 (node, tier)，重试
        candidates[result.node_id].tier_capacities.erase(result.tier)
```

`MasterServer` 的 `RoutePut` handler 只需调用 `router_.RoutePut()` 并填充响应，
无需在 handler 层面处理重试逻辑。

**容量一致性**：`AllocateForPut` 和 `DeallocateForUnregister` 在每次分配/回收后，
将 `tier_capacities[tier].available_bytes` 与 `PoolAllocator::AvailableBytes()` 同步。
这确保路由策略始终看到准确的剩余容量，而非心跳上报的过时值。

SSD tier 的 `allocated_offset` 为 0（仅容量模式）——SSD 寻址通过
PeerService 使用文件路径处理。

---

## 7. PoolClient C++ 接口

### 7.1 配置

```cpp
namespace mori::umbp {

struct PoolClientConfig {
    MasterClientConfig master_config;

    // MORI IO Engine
    mori::io::IOEngineConfig io_config;
    mori::io::BackendType backend_type = mori::io::BackendType::RDMA;
    std::unique_ptr<mori::io::BackendConfig> backend_config;

    // 本地 staging 缓冲区（RDMA 传输中转）
    size_t staging_buffer_size = 64ULL * 1024 * 1024;  // 64 MB

    // 暴露给集群的 DRAM 缓冲区（可选；使本节点成为存储提供方）
    void*  exportable_dram_buffer = nullptr;
    size_t exportable_dram_buffer_size = 0;

    // 暴露给集群的 SSD 存储（可选）
    std::string exportable_ssd_dir;
    size_t exportable_ssd_capacity = 0;

    // 向 Master 上报的 tier 容量
    std::map<TierType, TierCapacity> tier_capacities;

    // PeerService 监听端口（0 = 不启动，纯消费者模式）
    uint16_t peer_service_port = 0;
};

}  // namespace mori::umbp
```

### 7.2 类定义

```cpp
namespace mori::umbp {

class PoolClient {
 public:
  explicit PoolClient(PoolClientConfig config);
  ~PoolClient();

  PoolClient(const PoolClient&) = delete;
  PoolClient& operator=(const PoolClient&) = delete;

  // --- 生命周期 ---
  bool Init();
  void Shutdown();

  // --- 核心 API ---
  bool Put(const std::string& key, const void* src, size_t size);
  bool Get(const std::string& key, void* dst, size_t size);
  bool Remove(const std::string& key);

  // --- 访问内部组件 ---
  MasterClient& Master();
  bool IsInitialized() const;

 private:
  PoolClientConfig config_;
  bool initialized_ = false;

  // 控制面
  std::unique_ptr<MasterClient> master_client_;

  // 数据面 —— IO Engine
  std::unique_ptr<mori::io::IOEngine> io_engine_;
  mori::io::MemoryDesc staging_mem_;
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  // 数据面 —— PeerService 服务端（本节点作为存储提供方时运行）
  std::unique_ptr<PeerServiceServer> peer_service_;

  // 本地缓存：key → Location（Put 时写入，Remove 时查询，免去 Lookup RPC）
  std::mutex cache_mutex_;
  std::unordered_map<std::string, Location> location_cache_;

  // 缓存的 peer 连接（懒初始化）
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    mori::io::MemoryDesc dram_memory;
    uint64_t staging_base_offset = 0;  // SSD staging 区在 DRAM buffer 中的起始 offset
    bool engine_registered = false;
    // 远端 PeerService 的 gRPC stub
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;  // 序列化对该 peer 的 SSD 操作，防止 staging 竞态
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, PeerConnection> peers_;

  PeerConnection& GetOrConnectPeer(const std::string& node_id,
                                   const std::string& peer_address,
                                   const std::vector<uint8_t>& engine_desc_bytes,
                                   const std::vector<uint8_t>& dram_memory_desc_bytes);

  // DRAM 路径：纯 RDMA
  bool DramWrite(PeerConnection& peer, const void* src, size_t size, uint64_t offset);
  bool DramRead(PeerConnection& peer, void* dst, size_t size, uint64_t offset);

  // SSD 路径：RDMA + PeerService 协调
  bool SsdWrite(PeerConnection& peer, const std::string& key,
                const void* src, size_t size);
  bool SsdRead(PeerConnection& peer, const std::string& key,
               void* dst, size_t size);
};

}  // namespace mori::umbp
```

---

## 8. 数据流

### 8.1 Init

```
  PoolClient                IOEngine              PeerServiceServer    Master
      │                         │                       │                │
      │  IOEngine(key, config)  │                       │                │
      │────────────────────────►│                       │                │
      │  CreateBackend(RDMA)    │                       │                │
      │────────────────────────►│                       │                │
      │                         │                       │                │
      │  分配 staging_buffer    │                       │                │
      │  RegisterMemory(buf)    │                       │                │
      │────────────────────────►│  → staging_mem_       │                │
      │                         │                       │                │
      │  [如果是存储提供方:]     │                       │                │
      │  RegisterMemory(        │                       │                │
      │    exportable_dram)     │                       │                │
      │────────────────────────►│  → export_mem_        │                │
      │                         │                       │                │
      │  [如果 peer_port > 0:]  │                       │                │
      │  启动 PeerService ──────────────────────────────►  监听中         │
      │                         │                       │                │
      │  打包 engine_desc +     │                       │                │
      │  dram_memory_desc       │                       │                │
      │  RegisterSelf(caps,     │                       │                │
      │    peer_addr,           │                       │                │
      │    engine_desc,         │                       │                │
      │    dram_memory_desc)    │                       │                │
      │──────────────────────────────────────────────────────────────────►
      │                         │                       │  存入           │
      │                         │                       │  ClientRecord  │
      │  heartbeat_interval ◄────────────────────────────────────────────│
      │  StartHeartbeat()       │                       │                │
```

### 8.2 Put/Get 分发逻辑

RoutePut 返回的目标节点可能是本节点或远端节点，tier 可能是 DRAM 或 SSD。
`PoolClient` 根据 `is_local = (result.node_id == self.node_id)` 和 `result.tier`
做 4-way 分发：

```
Put(key, src, size):
    result = Master::RoutePut(key, size)
    is_local = (result.node_id == config_.master_config.node_id)

    if is_local && tier == DRAM:
        memcpy(local_dram + offset, src, size)       // 直接本地拷贝
    elif is_local && tier == SSD:
        PosixFile write {ssd_dir}/{key}.bin           // 直接本地写文件
    elif remote && tier == DRAM:
        RDMA write → remote DRAM at offset            // 见 8.3
    elif remote && tier == SSD:
        RDMA write → remote staging + CommitSsdWrite  // 见 8.4

    Master::Register(key, location)
    location_cache_[key] = location
```

```
Get(key, dst, size):
    location = Master::RouteGet(key)  // 或从 location_cache_ 查
    is_local = (location.node_id == config_.master_config.node_id)

    if is_local && tier == DRAM:
        memcpy(dst, local_dram + offset, size)        // 直接本地拷贝
    elif is_local && tier == SSD:
        PosixFile read {ssd_dir}/{key}.bin             // 直接本地读文件
    elif remote && tier == DRAM:
        RDMA read ← remote DRAM at offset             // 见 8.5
    elif remote && tier == SSD:
        PrepareSsdRead + RDMA read ← remote staging   // 见 8.6
```

本地路径不需要 RDMA 和 PeerService，仅涉及 memcpy 或 PosixFile I/O。
以下各节详述远端路径的完整时序。

> **Zero-copy 模式**：已实现。详见 Section 8.8。

### 8.3 Put — 远端 DRAM（2 次 RPC + 1 次 RDMA）

```
  调用者         PoolClient          Master              IOEngine     远端节点 B
    │                 │                 │                    │              │
    │  Put(key,src,sz)│                 │                    │              │
    │────────────────►│                 │                    │              │
    │                 │  RoutePut(key,  │                    │              │
    │                 │    node_id, sz) │                    │              │
    │                 │────────────────►│                    │              │
    │                 │                 │  选择节点 B         │              │
    │                 │                 │  分配 offset=4096  │              │
    │                 │                 │  扣减容量           │              │
    │                 │  {node_B, DRAM, │                    │              │
    │                 │   offset=4096,  │                    │              │
    │                 │   engine_desc,  │                    │              │
    │                 │   memory_desc}  │                    │              │
    │                 │◄────────────────│                    │              │
    │                 │                 │                    │              │
    │                 │  [首次: GetOrConnectPeer              │              │
    │                 │   → RegisterRemoteEngine]            │              │
    │                 │                 │                    │              │
    │                 │  memcpy(staging, src, sz)            │              │
    │                 │  Write(staging, 0,                   │              │
    │                 │    remote_dram, 4096, sz)            │              │
    │                 │────────────────────────────────────►│              │
    │                 │                 │                    │  RDMA WRITE  │
    │                 │                 │                    │─────────────►│
    │                 │                 │                    │  完成         │
    │                 │                 │                    │◄─────────────│
    │                 │  status OK ◄─────────────────────────│              │
    │                 │                 │                    │              │
    │                 │  loc = Location{│                    │              │
    │                 │    node_B,      │                    │              │
    │                 │    "4096",      │                    │              │
    │                 │    sz, DRAM}    │                    │              │
    │                 │  Register(key,  │                    │              │
    │                 │    loc)         │                    │              │
    │                 │────────────────►│                    │              │
    │                 │  OK ◄───────────│                    │              │
    │                 │                 │                    │              │
    │                 │  location_cache_│                    │              │
    │                 │    [key] = loc  │                    │              │
    │  true           │                 │                    │              │
    │◄────────────────│                 │                    │              │
```

### 8.4 Put — 远端 SSD（3+ 次 RPC + 1 次 RDMA）

PoolClient 在整个 SSD 操作期间持有 `peer.ssd_op_mutex`，确保同一远端节点的
staging 区不被并发写入覆盖。staging 区是固定复用的 buffer，无需分配/释放。

```
  调用者    PoolClient        Master         PeerService_B    IOEngine    远端 B
    │           │                │                │              │           │
    │  Put(k,s) │                │                │              │           │
    │──────────►│                │                │              │           │
    │           │  RoutePut ─────►                │              │           │
    │           │                │  选择 B, SSD   │              │           │
    │           │  {B, SSD,      │                │              │           │
    │           │   peer_addr,   │                │              │           │
    │           │   engine_desc} │                │              │           │
    │           │◄───────────────│                │              │           │
    │           │                │                │              │           │
    │           │  lock(peer_B.ssd_op_mutex)      │              │           │
    │           │  memcpy(staging, src, sz)       │              │           │
    │           │                │                │              │           │
    │           │  --- 阶段 1: RDMA 写入远端 DRAM staging ---               │
    │           │  Write(staging, 0, remote_dram, │              │           │
    │           │    staging_base_offset, sz)     │              │           │
    │           │──────────────────────────────────────────────►│           │
    │           │                │                │              │  RDMA 写  │
    │           │                │                │              │──────────►│
    │           │  OK ◄─────────────────────────────────────────│           │
    │           │                │                │              │           │
    │           │  --- 阶段 2: RPC 持久化到 SSD (PosixFile) ---             │
    │           │  CommitSsdWrite(key,            │              │           │
    │           │    staging_base_offset, sz) ───►              │           │
    │           │                │                │  写 SSD      │           │
    │           │                │                │  (PosixFile) │           │
    │           │  {ssd_loc_id} ◄─────────────────│              │           │
    │           │                │                │              │           │
    │           │  loc = Location│                │              │           │
    │           │    {B, ssd_loc,│                │              │           │
    │           │     sz, SSD}   │                │              │           │
    │           │  Register(key, │                │              │           │
    │           │    loc) ───────►                │              │           │
    │           │  OK ◄──────────│                │              │           │
    │           │                │                │              │           │
    │           │  location_cache│                │              │           │
    │           │    [key] = loc │                │              │           │
    │           │  unlock(peer_B.ssd_op_mutex)   │              │           │
    │  true     │                │                │              │           │
    │◄──────────│                │                │              │           │
```

### 8.5 Get — 远端 DRAM（1 次 RPC + 1 次 RDMA）

```
  调用者         PoolClient          Master              IOEngine     远端节点 B
    │                 │                 │                    │              │
    │  Get(key,dst,sz)│                 │                    │              │
    │────────────────►│                 │                    │              │
    │                 │  RouteGet(key)  │                    │              │
    │                 │────────────────►│                    │              │
    │                 │  Location{B,    │                    │              │
    │                 │    "4096", sz,  │                    │              │
    │                 │    DRAM} +      │                    │              │
    │                 │   engine_desc,  │                    │              │
    │                 │   memory_desc   │                    │              │
    │                 │◄────────────────│                    │              │
    │                 │                 │                    │              │
    │                 │  Read(staging, 0,                    │              │
    │                 │    remote_dram, 4096, sz)            │              │
    │                 │────────────────────────────────────►│              │
    │                 │                 │                    │  RDMA READ   │
    │                 │                 │                    │─────────────►│
    │                 │                 │                    │  数据         │
    │                 │                 │                    │◄─────────────│
    │                 │  status OK ◄─────────────────────────│              │
    │                 │  memcpy(dst, staging, sz)            │              │
    │  true           │                 │                    │              │
    │◄────────────────│                 │                    │              │
```

### 8.6 Get — 远端 SSD（2 次 RPC + 1 次 RDMA）

同样持有 `peer.ssd_op_mutex`，确保 staging 区不被并发操作覆盖。

```
  调用者    PoolClient        Master         PeerService_B    IOEngine    远端 B
    │           │                │                │              │           │
    │  Get(k,d) │                │                │              │           │
    │──────────►│                │                │              │           │
    │           │  RouteGet ─────►                │              │           │
    │           │  Location{B,   │                │              │           │
    │           │   ssd_loc, sz, │                │              │           │
    │           │   SSD} +       │                │              │           │
    │           │   peer_addr... │                │              │           │
    │           │◄───────────────│                │              │           │
    │           │                │                │              │           │
    │           │  lock(peer_B.ssd_op_mutex)      │              │           │
    │           │  --- 阶段 1: RPC 加载 SSD 数据到 DRAM staging ---         │
    │           │  PrepareSsdRead(key,            │              │           │
    │           │    ssd_loc, sz) ────────────────►              │           │
    │           │                │                │  读 SSD      │           │
    │           │                │                │  (PosixFile) │           │
    │           │                │                │  → staging   │           │
    │           │  {staging_off} ◄─────────────────│              │           │
    │           │                │                │              │           │
    │           │  --- 阶段 2: RDMA 从远端 DRAM staging 读回 ---            │
    │           │  Read(staging, 0, remote_dram,  │              │           │
    │           │    staging_off, sz)             │              │           │
    │           │──────────────────────────────────────────────►│           │
    │           │                │                │              │  RDMA 读  │
    │           │                │                │              │──────────►│
    │           │  OK ◄─────────────────────────────────────────│           │
    │           │  memcpy(dst, staging, sz)       │              │           │
    │           │  unlock(peer_B.ssd_op_mutex)    │              │           │
    │  true     │                │                │              │           │
    │◄──────────│                │                │              │           │
```

### 8.7 Remove

`PoolClient` 在每次 Put 成功后将 `Location` 缓存到本地
`unordered_map<string, Location>` 中。Remove 时直接查本地缓存，不需要
Lookup RPC。

```
  调用者         PoolClient          Master
    │                 │                 │
    │  Remove(key)    │                 │
    │────────────────►│                 │
    │                 │  location =     │
    │                 │  location_cache_│
    │                 │    [key]        │
    │                 │                 │
    │                 │  Unregister(key,│
    │                 │    location)    │
    │                 │────────────────►│
    │                 │                 │  从 BlockIndex 移除
    │                 │                 │  allocators[tier].Deallocate(
    │                 │                 │    offset, size)
    │                 │                 │  // DRAM: 回收偏移量 + 容量
    │                 │                 │  // SSD: 仅回收容量
    │                 │  OK ◄───────────│
    │                 │                 │
    │                 │  从 location_   │
    │                 │  cache_ 删除    │
    │  true           │                 │
    │◄────────────────│                 │
```

---

### 8.8 Zero-copy 模式

远端路径默认通过 staging buffer 中转（`memcpy` → RDMA），引入一次额外拷贝。
Zero-copy 模式允许调用者预先注册内存，RDMA 直接操作注册内存，跳过 staging。

#### 接口

```cpp
// 注册：调用者保证 ptr 在 RegisterMemory 到 DeregisterMemory 期间有效
bool RegisterMemory(void* ptr, size_t size);
void DeregisterMemory(void* ptr);

// Put/Get 新增 zero_copy 参数（默认 true）
bool Put(const std::string& key, const void* src, size_t size, bool zero_copy = true);
bool Get(const std::string& key, void* dst, size_t size, bool zero_copy = true);
```

#### 行为

| `zero_copy` | 指针已注册 | 行为 |
|-------------|-----------|------|
| `true` | 是 | 直接用注册内存做 RDMA，不持 `staging_mutex_` |
| `true` | 否 | log warning，fallback 到 staging 路径 |
| `false` | 不关心 | 始终走 staging 路径 |

本地路径（本地 DRAM/SSD）不受 `zero_copy` 影响——本身就直接操作传入指针。

#### 内部实现

```cpp
struct RegisteredRegion {
    void* base;
    size_t size;
    mori::io::MemoryDesc mem_desc;
};
std::mutex registered_mem_mutex_;
std::vector<RegisteredRegion> registered_regions_;

// 查找 ptr 是否在已注册区域内，返回 {MemoryDesc, offset_within_region}
std::optional<std::pair<mori::io::MemoryDesc, size_t>>
FindRegisteredMemory(const void* ptr, size_t size);
```

`RegisterMemory` 调用 `IOEngine::RegisterMemory` 获取 `MemoryDesc` 后存入列表。
`FindRegisteredMemory` 对每个注册区域做范围检查：`ptr >= base && ptr + size <= base + region_size`。
支持子区间匹配——注册一块大 buffer 后，可以用其中任意子区间做 zero-copy。

#### 使用示例

```cpp
auto buf = std::make_unique<char[]>(1024 * 1024);
client.RegisterMemory(buf.get(), 1024 * 1024);

// 同一注册区域内的子区间均可 zero-copy
client.Put("key1", buf.get(), 4096);               // zero-copy
client.Put("key2", buf.get() + 4096, 4096);         // zero-copy
client.Put("key3", unregistered_ptr, 4096);          // fallback + warning

client.DeregisterMemory(buf.get());
```

#### 并发优势

zero-copy 路径不持有 `staging_mutex_`，多个 Put/Get 可以并行 RDMA，
避免了 staging buffer 的串行瓶颈。

#### 未来扩展：共享内存注册

当前 `RegisterMemory` 与 PoolClient 内部的 IOEngine 绑定。未来如果需要
一块内存被多个 PoolClient 共享（如多个 TP rank 共用一块 host memory），
可以新增 `ImportMemory(ptr, size, MemoryDesc)` 方法，接受外部已注册的
`MemoryDesc`，跳过 IOEngine 注册步骤。改动量很小。

---

## 9. PeerService 实现

### 9.1 PeerServiceServer

```cpp
namespace mori::umbp {

class PeerServiceServer {
 public:
  PeerServiceServer(void* dram_buffer, size_t dram_size,
                    const mori::io::MemoryDesc& dram_mem_desc,
                    const mori::io::EngineDesc& engine_desc,
                    const std::string& ssd_dir, size_t ssd_capacity);

  void Start(uint16_t port);
  void Stop();

 private:
  // SSD staging 缓冲区：RDMA 注册的 DRAM 内的一块简单固定区域。
  // 不由 Master 的 PoolAllocator 管理——仅本地 mutex 保护。
  // SSD 操作由 ssd_mutex_ 序列化，单个 staging slot 即可。
  void* ssd_staging_base_;
  size_t ssd_staging_size_;

  // SSD：PosixFile I/O
  std::string ssd_dir_;
  size_t ssd_capacity_;
  size_t ssd_used_ = 0;
  std::mutex ssd_mutex_;  // 序列化所有 SSD 操作 + staging 访问

  // IO Engine 描述符（GetPeerInfo 返回给调用方）
  mori::io::EngineDesc engine_desc_;
  mori::io::MemoryDesc dram_mem_desc_;
};

}  // namespace mori::umbp
```

SSD 操作使用标准 POSIX 文件 I/O：
- 写入：`open(O_WRONLY | O_CREAT | O_TRUNC)` → `write` → `fsync` → `close`
- 读取：`open(O_RDONLY)` → `pread` → `close`
- 文件布局：每个 block 一个文件，路径为 `{ssd_dir}/{key}.bin`

### 9.2 DRAM 布局

节点的 DRAM 缓冲区逻辑上分为两个区域：

```
  ┌──────────────────────────────────────────────────────┐
  │              RDMA 注册的 DRAM 缓冲区                    │
  │                                                      │
  │  ┌────────────────────┐  ┌────────────────────────┐  │
  │  │  主区域             │  │  SSD staging 区域       │  │
  │  │  (Master 管理       │  │  (PeerService 管理      │  │
  │  │   偏移量分配)        │  │   SSD 读写中转)         │  │
  │  └────────────────────┘  └────────────────────────┘  │
  └──────────────────────────────────────────────────────┘
```

- **主区域**：由 Master 的 `PoolAllocator`（DRAM 模式）管理。用于 DRAM tier 块存储。地址在 `RoutePut` 中返回。
- **SSD staging 区域**：由 PeerService 本地管理（`ssd_mutex_` 保护的固定 buffer）。
  SSD 读写操作的临时中转区，固定复用，无需分配/释放。

两个区域属于同一个 RDMA 注册的 `MemoryDesc`，但占用不重叠的 offset 范围。

---

## 10. 需要新增/修改的文件

**新增文件：**
- `include/umbp/pool_client.h` — PoolClient 头文件
- `src/pool_client.cpp` — PoolClient 实现
- `include/umbp/peer_service.h` — PeerServiceServer 头文件
- `src/peer_service.cpp` — PeerServiceServer 实现
- `proto/umbp_peer.proto` — PeerService proto 定义
- `include/umbp/pool_allocator.h` — PoolAllocator 头文件

**修改文件：**
- `proto/umbp.proto` — 新增 `peer_address`、`engine_desc`、`dram_memory_desc`、`allocated_offset` 字段
- `include/umbp/types.h` — 扩展 `ClientRecord`（peer 信息 + per-tier `PoolAllocator`）
- `include/umbp/client.h` — `MasterClient::RegisterSelf` 新增 peer/engine 参数
- `src/client.cpp` — 实现扩展的 `RegisterSelf`，解析新响应字段
- `src/client_registry.cpp` — 存储新字段到 `ClientRecord`
- `src/master_server.cpp` — 填充新响应字段；RoutePut 中调用 `PoolAllocator`；Unregister 中对 DRAM 和 SSD tier 均调用 `Deallocate`
- `include/umbp/route_put_strategy.h` — 扩展 `RoutePutResult`
- `CMakeLists.txt` — 新增源文件、新 proto、链接 `mori::io`

---

## 11. 并发设计

### 11.1 PoolAllocator 线程安全

`PoolAllocator` 存在于 Master 的 `ClientRecord` 中。RoutePut（写 `Allocate`）和
Unregister（写 `Deallocate`）都会修改分配器状态。这些操作来自多个 gRPC handler 线程。

**方案**：`PoolAllocator` 内部**不加锁**。它受 `ClientRegistry` 的 `shared_mutex` 保护：

- **现有问题**：当前 `RoutePut` 路径只调用 `GetAliveClients()`（读操作，加读锁），
  不修改 `ClientRecord`。新增 `PoolAllocator::Allocate` 后变成写操作。
- **解决方案**：`Router::RoutePut` 内部分两步执行：
  1. 读锁获取 alive clients（调用 `GetAliveClients()`）
  2. 策略选中目标节点后，通过 `registry_.AllocateForPut()` **加写锁**调用
     `allocators[tier].Allocate(block_size)` 并同步 `tier_capacities`
  3. 如果分配失败，从本地候选列表中移除该 (node, tier)，重试——
     全部在 `Router::RoutePut` 内部完成

伪代码：

```
Router::RoutePut(key, node_id, block_size):
    candidates = registry_.GetAliveClients()         // 读锁
    loop:
        result = strategy.Select(candidates, block_size)
        if !result: return nullopt

        alloc = registry_.AllocateForPut(...)        // 写锁
        if alloc:
            // 合入分配结果，返回
        candidates[result.node_id].tier_capacities.erase(result.tier)  // 重试
```

- `AllocateForPut` 和 `DeallocateForUnregister` 在每次操作后均将
  `tier_capacities[tier].available_bytes` 与 `PoolAllocator::AvailableBytes()` 同步，
  保证路由策略看到准确的容量。
- `Unregister` 已经在写锁下执行（现有设计），直接在其中调用 `Deallocate` 即可。
- `PoolAllocator` 本身无锁，避免嵌套锁问题。

### 11.2 PoolClient 内部线程安全

| 资源 | 保护方式 |
|------|----------|
| `staging_buffer_` / `staging_mem_` | `staging_mutex_`，串行化 RDMA 传输 |
| `location_cache_` | `cache_mutex_`，读写锁均可 |
| `peers_` | `peers_mutex_`，懒初始化 + 缓存查询 |
| `MasterClient` | 内部已线程安全（gRPC channel 线程安全） |
| `PeerServiceServer` | gRPC 服务端自身线程安全；`ssd_mutex_` 序列化 SSD 操作 |

---

## 12. MasterClient 接口变更

### 12.1 MasterClientConfig 扩展

无变化。`peer_address` 等信息由 `PoolClient` 在调用 `RegisterSelf` 时传入，
不需要固化到 config 中。

### 12.2 RegisterSelf 签名变更

```cpp
// 现有签名
grpc::Status RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities);

// 新签名：增加 PoolClient 数据面信息
grpc::Status RegisterSelf(
    const std::map<TierType, TierCapacity>& tier_capacities,
    const std::string& peer_address,                    // PeerService gRPC 地址
    const std::vector<uint8_t>& engine_desc_bytes,      // packed EngineDesc
    const std::vector<uint8_t>& dram_memory_desc_bytes  // packed MemoryDesc (DRAM)
);
```

实现变更：将新参数填入 `RegisterClientRequest` 的 field 4-6。

### 12.3 RoutePut 返回值扩展

`RoutePutResult` 已在 Section 6 中定义。`MasterClient::RoutePut` 的签名不变：

```cpp
grpc::Status RoutePut(const std::string& key, uint64_t block_size,
                      std::optional<RoutePutResult>* out_result);
```

实现变更：解析 `RoutePutResponse` 的新字段 (field 5-8)，填入 `RoutePutResult` 的
`peer_address`、`engine_desc_bytes`、`dram_memory_desc_bytes`、`allocated_offset`。

### 12.4 RouteGet 返回值扩展

需要新增一个扩展结构体来携带数据面信息：

```cpp
struct RouteGetResult {
    Location location;                                // 现有：block 位置
    std::string peer_address;                         // 新增
    std::vector<uint8_t> engine_desc_bytes;           // 新增
    std::vector<uint8_t> dram_memory_desc_bytes;      // 新增
};
```

签名变更：

```cpp
// 现有签名
grpc::Status RouteGet(const std::string& key,
                      std::optional<Location>* out_location);

// 新签名
grpc::Status RouteGet(const std::string& key,
                      std::optional<RouteGetResult>* out_result);
```

实现变更：解析 `RouteGetResponse` 的新字段 (field 3-5)。

### 12.5 Unregister 变更

签名不变。实现变更：`MasterServer` 端在 `Unregister` handler 中，
从 `Location` 解析 tier 和 offset，调用
`allocators[tier].Deallocate(offset, size)` 回收容量。

---

## 13. 测试策略

### 13.1 PoolAllocator 单元测试

- Allocate 基本分配：分配后 `AvailableBytes` 正确扣减
- Allocate 超过容量：返回 `nullopt`
- Deallocate 回收：释放后 `AvailableBytes` 恢复
- Allocate + Deallocate + 再 Allocate：free list 复用已释放的块
- 碎片化场景：分配 A、B、C → 释放 A 和 C → 分配 D（D 大于 A 或 C 单独大小）→ 失败
- SSD 模式（无 offset_tracker）：`Allocate` 返回 0，仅扣减容量
- 并发 Allocate/Deallocate（压力测试）

### 13.2 PeerService 单元测试

- GetPeerInfo：返回正确的 EngineDesc 和 MemoryDesc
- CommitSsdWrite：RDMA staging 数据正确写入 SSD 文件 `{ssd_dir}/{key}.bin`
- PrepareSsdRead：SSD 文件正确读入 staging，返回 staging_offset
- CommitSsdWrite 容量超限：返回 `success=false`
- PrepareSsdRead 文件不存在：返回 `success=false`

### 13.3 PoolClient 四路径集成测试

每种路径测试 Put → Get → 验证数据一致 → Remove → 验证清除：

**13.3.1 本地 DRAM**

```
1. 配置 RoutePut 策略使其返回本节点 + tier=DRAM
2. Put(key, src, size) → 成功
3. Get(key, dst, size) → 成功，dst 内容 == src
4. Remove(key) → 成功
5. Get(key, ...) → 失败（key 不存在）
```

**13.3.2 本地 SSD**

```
1. 配置 RoutePut 策略使其返回本节点 + tier=SSD
2. Put(key, src, size) → 成功
3. 验证 {ssd_dir}/{key}.bin 文件存在
4. Get(key, dst, size) → 成功，dst 内容 == src
5. Remove(key) → 成功
```

**13.3.3 远端 DRAM**

```
1. 启动 MasterServer + 两个 PoolClient（节点 A 和节点 B）
2. 两节点均 RegisterSelf，上报 DRAM 容量
3. 节点 A 调用 Put(key, src, size)
   → RoutePut 返回节点 B + DRAM + offset
   → RDMA 写入节点 B
   → Register 注册元数据
4. 节点 A 调用 Get(key, dst, size)
   → RouteGet 返回节点 B 的 Location
   → RDMA 读取节点 B
   → dst 内容 == src
5. 节点 A 调用 Remove(key)
   → Unregister + Deallocate 回收 offset
```

**13.3.4 远端 SSD**

```
1. 启动 MasterServer + 两个 PoolClient（节点 A 和节点 B，均配置 peer_service_port > 0）
2. 节点 A 调用 Put(key, src, size)
   → RoutePut 返回节点 B + SSD
   → RDMA 写入节点 B 的 DRAM staging
   → CommitSsdWrite → 节点 B 持久化到 SSD 文件
   → Register 注册元数据
3. 节点 A 调用 Get(key, dst, size)
   → RouteGet 返回节点 B + SSD
   → PrepareSsdRead → 节点 B 加载 SSD 到 staging
   → RDMA 读取 staging
   → dst 内容 == src
5. 节点 A 调用 Remove(key)
```

### 13.4 MasterClient 扩展测试

- RegisterSelf 携带 peer_address / engine_desc / dram_memory_desc
- RoutePut 返回扩展字段（peer_address, engine_desc, memory_desc, allocated_offset）
- RouteGet 返回扩展字段
- RoutePut 后 PoolAllocator 容量正确扣减
- Unregister 后 PoolAllocator 容量正确恢复
- 多个 RoutePut 分配不同 offset，无重叠
- 节点满后 RoutePut 路由到其他节点

### 13.5 边界与异常测试

- Put 时所有节点均无可用空间 → RoutePut 返回 `found=false` → Put 返回 false
- RDMA 传输失败 → Put 返回 false，已分配的 offset 需要回滚（或由后续 GC 清理）
- PeerService 不可达 → SSD 路径 Put/Get 返回 false
- PoolClient Init 失败（无法连接 Master）→ Init 返回 false
- Remove 不存在的 key → location_cache_ 查不到 → 返回 false
- 并发 Put/Get/Remove 压力测试

---

## 14. 开放问题

- **Staging 缓冲区并发**：多个并发 Put/Get 操作争用本地 staging 缓冲区。初版使用 mutex
  串行化；后续可升级为 ring buffer 或 arena allocator。
- **大块传输**：如果 `block_size > staging_buffer_size`，需要分片传输。初版可拒绝超大块。
- **DRAM 碎片化**：bump + free list 分配器可能随时间碎片化。`Deallocate` 时应合并相邻空闲块。
  如碎片化严重，可升级为 slab allocator。
- **SSD staging 并发扩展**：当前 per-peer `ssd_op_mutex` 序列化 SSD 操作。如需更高并发，
  可将 staging 区分成多个 slot，支持同时多个 SSD 操作到同一远端节点。
- **PeerService 连接池**：高并发场景下可能需要 gRPC 连接池或流式 RPC 减少连接开销。

---

## 15. 实现状态与待办

### 15.1 已完成

| 功能 | 文件 | 测试 |
|------|------|------|
| PoolAllocator（DRAM offset + SSD 容量） | `pool_allocator.h` | 11 个单元测试全部通过 |
| Master Proto 扩展 | `umbp.proto` | 编译通过 |
| PeerService Proto + 实现 | `umbp_peer.proto`, `peer_service.h/cpp` | 5 个单元测试全部通过 |
| Server 端扩展（ClientRecord, RoutePut 分配, Unregister 回收） | `types.h`, `client_registry.h/cpp`, `master_server.cpp`, `route_put_strategy.h` | 现有测试无回归 |
| MasterClient 接口扩展 | `client.h/cpp` | 现有测试无回归 |
| PoolClient 本地 DRAM Put/Get/Remove | `pool_client.h/cpp` | 单元测试通过 + demo 验证 |
| PoolClient 本地 SSD Put/Get/Remove | `pool_client.h/cpp` | 单元测试通过 |
| PoolClient 远端 DRAM Put/Get（IOEngine RDMA） | `pool_client.cpp` | 代码完成，待 RDMA 端到端验证 |
| PoolClient 远端 SSD Put/Get（RDMA + PeerService） | `pool_client.cpp` | 代码完成，待 RDMA 端到端验证 |
| location_cache_ 本地缓存 | `pool_client.cpp` | 已测试 |
| Demo 可执行文件（provider/consumer/dram-only/ssd-only） | `pool_client_main.cpp` | 本地路径 demo 通过 |

### 15.2 待做

| 优先级 | 功能 | 说明 |
|--------|------|------|
| **P0** | 远端 RDMA 端到端验证 | 需要在两个节点手动跑 provider + consumer，验证远端 DRAM 和 SSD 路径数据正确性 |
| **P1** | staging buffer 大块保护 | `block_size > staging_buffer_size` 时应拒绝或分片，当前无检查 |
| **P2** | PoolAllocator 碎片整理 | Deallocate 已实现相邻块合并，但长期运行可能仍有碎片；可升级为 slab allocator |
| **P2** | SSD staging 并发扩展 | 当前 per-peer `ssd_op_mutex` 串行化；如需更高并发可分成多个 slot |
| **P2** | PeerService 连接池 | 高并发场景下 gRPC channel 复用 |
| **P3** | Zero-copy RDMA 优化 | 调用者源指针已在 RDMA 注册内存中时，跳过 staging memcpy |
| **P3** | ~~心跳容量实时更新~~ | **已完成**：`AllocateForPut`/`DeallocateForUnregister` 在每次操作后同步 `tier_capacities.available_bytes`，路由策略始终看到准确容量。RoutePut handler 增加 exclude+retry 机制应对竞态。 |

### 15.3 远端路径验证命令

测试远端 DRAM（provider 只注册 DRAM，用 `--tier dram`）：

```bash
# Terminal 1: Master
./build_umbp/src/umbp/umbp_master 0.0.0.0:50051

# Terminal 2: Provider（仅 DRAM）
./build_umbp/src/umbp/umbp_pool_client localhost:50051 node-1 localhost:8080 \
  --provider --tier dram --io-host 10.67.77.61 --io-port 18080 --peer-port 19080

# Terminal 3: Consumer（路由到 node-1 的 DRAM）
./build_umbp/src/umbp/umbp_pool_client localhost:50051 node-2 localhost:8081 \
  --consumer --io-host 10.67.77.61 --io-port 18081
```

测试远端 SSD（provider 只注册 SSD，用 `--tier ssd`）：

```bash
# Terminal 2: Provider（仅 SSD）
./build_umbp/src/umbp/umbp_pool_client localhost:50051 node-1 localhost:8080 \
  --provider --tier ssd --ssd-dir /mnt/nvme0/umbp_ssd \
  --io-host 10.67.77.61 --io-port 18080 --peer-port 19080

# Terminal 3: Consumer（路由到 node-1 的 SSD）
./build_umbp/src/umbp/umbp_pool_client localhost:50051 node-2 localhost:8081 \
  --consumer --io-host 10.67.77.61 --io-port 18081
```
