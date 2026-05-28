# CCO 工作交接（完整版）

你是接受这个任务的 agent。mori 代码库在 `/home/jiahzhou/workspace/mori`。你的任务是实现 CCO 模块（host 端初始化，device API 骨架）。下面包含你需要的所有信息。

---

## 背景

mori 是一个 GPU 通信库，有 `mori-SHMEM` 模块提供 GPU-initiated P2P / RDMA / SDMA 传输。当前 SHMEM 用 Meyer's singleton（`ShmemStatesSingleton`）管理全局状态，一个进程只能有一套通信上下文。

**CCO 目标**：实现类似 NCCL LSA（P2P）+ GIN（RDMA）+ SDMA 的功能，用显式 comm 句柄替代 singleton，支持单进程内多个独立 comm 实例。

核心特性：
1. **无 singleton**：每个 `CcoComm` 独立堆分配，多线程可并发使用
2. **三段式初始化**：`CommCreate` → `WindowRegister` → `DevCommCreate`
3. **三条显式传输路径**：P2P（直接 GPU store）、RDMA（ibgda）、SDMA（DMA 引擎），用户在 kernel 里显式选择，不做自动 dispatch
4. **统一 VMM 内存**：window 底层用 `hipMemCreate`，一次注册同时具备三条路径能力

---

## 命名约定（Naming Convention）

CCO 包含两层 API，**风格刻意分开**：

### Host 端（CCO comm/window 生命周期 + 资源管理）

沿用 mori 通用风格（Google C++ Style 变体），与 `mori_application` / `mori_shmem` / `Rdma*` / `Sdma*` 兄弟模块对齐：

| 元素 | 规则 | 例子 |
|------|------|------|
| 函数（free function） | `CcoPascalCase`（带前缀） | `CcoCommCreate`, `CcoMemAlloc`, `CcoWindowRegister`, `CcoDevCommCreate`, `CcoBarrierAll` |
| Struct / Class | `CcoPascalCase`（带前缀） | `CcoComm`, `CcoWindowHost`, `CcoDevComm` |
| Handle typedef | `CcoPascalCase_t`（带前缀 + `_t` 后缀） | `CcoWindow_t`, `CcoDevComm_t` |
| 字段（struct member） | `camelCase` | `worldSize`, `flatBase`, `nextOffset`, `numQpPerPe` |
| 常量 / 宏 | `CCO_UPPER_SNAKE_CASE`（带前缀） | `CCO_WINDOW_TABLE_SIZE` |
| 文件 | `cco_xxx.hpp` / `cco_xxx.cpp` | `cco_api.hpp`, `cco_init.cpp`, `cco_memory.cpp` |
| 命名空间 | `mori::cco` | — |

### Device 端（CcoLsa / CcoGda / CcoSdma session + 内部辅助）

借鉴 **NCCL device API 风格**（`nccl_device/gin.h`, `nccl_device/lsa_barrier.h`），让从 NCCL/NVSHMEM 迁移过来的用户有熟悉感：

| 元素 | 规则 | 例子 |
|------|------|------|
| Session class | `CcoPascalCase`（带前缀） | `CcoGda`, `CcoLsa`, `CcoSdma`, `CcoLsaBarrierSession` |
| Session 成员函数 | `camelCase`（首字母小写） | `put`, `get`, `signal`, `flush`, `flushAsync`, `wait`, `readSignal`, `waitSignal`, `resetSignal` |
| Tag 类型（模板 dispatch） | `CcoModule_TagName`（**下划线**分隔模块名和 Tag 名） | `CcoGda_None`, `CcoGda_NoSignal`, `CcoGda_SignalInc`, `CcoGda_SignalAdd`, `CcoGda_CounterInc`, `CcoGda_SegmentDevice` |
| Handle typedef | `CcoPascalCase_t`（带前缀 + `_t` 后缀） | `CcoGdaSignal_t`, `CcoGdaCounter_t`, `CcoGdaRequest_t`, `CcoLsaBarrierHandle_t` |
| Enum 值 | `CcoModuleAction`（PascalCase，比 NCCL `SCREAMING_SNAKE_CASE` 短） | `CcoGdaSignalInc`, `CcoGdaSignalAdd` |
| 内部 / private 成员 | `_camelCase`（下划线前缀） | `_gdaHandle`, `_signalShadows` |
| Namespace 内 free function | `camelCase`（**不带** `Cco` 前缀，namespace 已经 disambiguate） | `mori::cco::findWindow`, `mori::cco::getPeerPtr`, `mori::cco::gda::put`, `mori::cco::gda::flush` |
| 字段（struct member） | `camelCase` | `signalId`, `counterId`, `contextId`, `winBase`, `stride4G` |
| 文件 | `xxx_device_common.hpp` + `xxx_device_api.hpp` | `gda_device_common.hpp`, `gda_device_api.hpp` |
| 命名空间 | `mori::cco::<backend>` | `mori::cco::gda`, `mori::cco::lsa`, `mori::cco::sdma` |

### 共同约定

- **缩写当作普通单词**（与 mori 现有代码一致）：`Cco` / `Rdma` / `Sdma` / `Gda` / `Lsa` / `Shmem` / `Nic` —— **不**写成 `CCO` / `RDMA` / `LSA`
- **类型 vs 函数的判别准则**：能用 `using` 在调用点免去 `mori::cco::` 前缀的（类型/handle）保留 `Cco` 前缀；不会单独被 `using` 出去的（namespace 内 free function）不加前缀
- **公共 API 入口（含 `__device__`）一律加 `Cco` 前缀**，方便用户 grep；内部 helper 不加

### 现有代码的对照

```cpp
// Host (mori style)
int CcoCommCreate(application::BootstrapNetwork* bootNet, ...);   // PascalCase
struct CcoComm { int rank; int worldSize; void* flatBase; };       // camelCase fields
static constexpr int CCO_WINDOW_TABLE_SIZE = 32;                   // SCREAMING_SNAKE

// Device GDA backend (NCCL style)
namespace mori::cco::gda {
  struct CcoGda_NoSignal {};                                       // Tag with `_`
  struct CcoGda_SignalInc { CcoGdaSignal_t signalId; };
  typedef uint32_t CcoGdaSignal_t;                                 // _t suffix

  struct CcoGda {
    void* _gdaHandle;                                              // internal `_` prefix
    __device__ void put(int peer, ...);                            // camelCase method
    __device__ void flushAsync(int peer, ...);
  };

  __device__ inline static void put(CcoGdaCtx ctx, ...);           // namespace-internal, no prefix
}
```

---

## 关键参考文件

| 角色 | 路径 |
|------|------|
| SHMEM 内部状态/结构体 | `include/mori/shmem/internal.hpp` |
| SHMEM 初始化逻辑 | `src/shmem/init.cpp` |
| VMM heap 完整实现 | `src/application/memory/symmetric_memory.cpp` |
| Context（RDMA 端点、传输类型） | `include/mori/application/context/context.hpp` |
| Device 类型定义 | `include/mori/application/application_device_types.hpp` |
| SHMEM Device API | `include/mori/shmem/shmem_device_api.hpp` |
| SDMA kernel | `include/mori/shmem/shmem_sdma_kernels.hpp` |
| 示例 | `examples/shmem/put_thread_allgather.cpp` |

---

## 关键设计判断（必读）

### 为什么必须用 VMM 内存（hipMemCreate）

hipMalloc 内存只能通过 `hipIpcOpenMemHandle` 在 peer 端打开，返回**固定 VA**，无法 remap 到指定地址。要实现 `flatBase + pe * perRankSize + offset` 这样的连续平坦地址空间，必须用 `hipMemCreate` 生成 `hipMemGenericAllocationHandle_t`，再通过 `hipMemMap` 映射到指定 VA。

### 双地址表设计（peerPtrs + p2pPeerPtrs）

与现有 `SymmMemObj` 一致，`CcoWindowDevice` 维护两套地址表：

- **`p2pPeerPtrs[pe]`**（P2P / SDMA 用）：本地 flat VA，`= flatBase + pe*perRankSize + slotOffset`。仅同节点 P2P 可达 peer 有值，远程 peer 为 0
- **`peerPtrs[pe]`**（RDMA 用）：iova=0 时全部为 0；iova=VA fallback 时存远端 PE 的 localPtr

三条路径的寻址：

- **P2P**：`remote = p2pPeerPtrs[pe] + dstOff`（仅同节点可达）
- **RDMA**：`raddr = peerPtrs[pe] + dstOff`（iova=0 时 = dstOff；iova=VA 时 = 远端VA + dstOff。两种模式同一份 kernel 代码）
- **SDMA**：`dstPtr = p2pPeerPtrs[pe] + dstOff`（与 P2P 共用，仅同节点可达）

一次 `CcoWindowRegister` 同时拥有三条路径的能力。

### iova=0 RDMA 机制

调用 `ibv_reg_dmabuf_mr(pd, offset=0, size, iova=0, fd, access)`，使该 MR 的 IOVA 地址空间从 0 开始。kernel 里填 `raddr = dstOff`（window 内偏移），NIC 通过 rkey 找到对端 MR 对应的物理内存。不需要知道对端的绝对 VA。

**为什么用 `ibv_reg_dmabuf_mr` 而不是 `ibv_reg_mr_iova2`**：
- `ibv_reg_mr_iova2` 走内核 `get_user_pages()` pin 页面路径，AMD GPU 上 `hipMemCreate`（VMM）分配的物理句柄不在用户空间页表中，注册会 ENOMEM
- `ibv_reg_dmabuf_mr` 走内核 dma-buf 子系统直接获取物理地址，绕过 `get_user_pages()`，VMM 内存可用
- 在 mlx5 + NVIDIA 环境中两者均可工作（`ibv_reg_mr_iova2` 靠 `nvidia-peermem` 模块），但 dmabuf 路径在 AMD/AINIC 和 NVIDIA/mlx5 上都可用，更通用
- 已在 MI355X + AINIC (vendor 0x1dd8) 上实测验证：`ibv_reg_dmabuf_mr(iova=0)` PASS，`ibv_reg_mr_iova2(iova=0)` ENOMEM

**Fallback（iova=VA 模式）**：
若 `ibv_reg_dmabuf_mr(iova=0)` 在某些 NIC 上不可用，可回退到 `ibv_reg_dmabuf_mr(pd, 0, size, iova=ptr, fd, access)`（当前 mori 已有 `RegisterRdmaMemoryRegionDmabuf` 实现），此时 `raddr = peerPtrs[pe] + dstOff`，需要 Allgather 交换各 PE 的 localPtr 作为 IOVA。

### MemAlloc 和 WindowRegister 分离（参考 ncclMemAlloc + ncclCommWindowRegister）

- `CcoMemAlloc`：VMM 分配 + P2P flat space 映射，**不做** RDMA MR 注册
- `CcoWindowRegister(comm, ptr, size, win)`：接受 MemAlloc 的 ptr，做 RDMA MR 注册 + SDMA signal setup + 构建 GPU device 结构
- `CcoWindowRegister(comm, size, win, &ptr)`：便捷重载，内部 = MemAlloc + WindowRegister(ptr)

### DevComm requirements + Connection + Team（参考 ncclDevCommRequirements）

CCO 学 NCCL 把 device 端的资源描述集中到一个 `CcoDevCommRequirements` 结构，由用户在 `CcoDevCommCreate` 时显式传入。三个核心维度：

**1. Connection type（GDA QP 分配策略）**

| 类型 | QP 数 (per rank) | 涵盖哪些 peer | 何时使用 |
|------|------------------|--------------|---------|
| `CCO_GDA_CONNECTION_NONE` | 0 | 不建 NIC QP | 纯 intra-node 应用，省 NIC 资源 |
| `CCO_GDA_CONNECTION_FULL` | `worldSize - 1` | 所有 peer（含同节点） | uniform addressing，便利但浪费 intra-node QP |
| `CCO_GDA_CONNECTION_CROSSNODE` ⭐ | `worldSize - lsaSize` | 所有跨节点 peer（跳过同节点） | **CCO 新增**：搭配 explicit `lsa.put`/`gda.put` 模型，省同节点 QP |
| `CCO_GDA_CONNECTION_RAIL` | `nNodes - 1` | 仅同 rail（同 NIC slot index）跨节点 peer | 分层算法（节点内 LSA + 节点间 rail-aware GDA） |

`CROSSNODE` 是 CCO 相对 NCCL 的延伸：NCCL 的 explicit team model 偏 uniform，缺少这个档；CCO 的 explicit backend selection 让 "GDA 永不发同节点" 成为合理设计点，刚好填上 FULL 和 RAIL 之间的空白。

**2. Team（kernel 端的 peer 寻址空间）**

`CcoTeam` 是 3-int 的逻辑 rank 子集描述符（与 ncclTeam 同构）：

```cpp
struct CcoTeam {
  int nRanks;   // 子集大小
  int rank;     // 我在子集中的 index
  int stride;   // 在 world rank 空间的步长
};
```

内置 team（device-side inline 函数）：

| Team | 用途 | 公式 |
|------|------|------|
| `CcoTeamWorld(devComm)` | 全部 ranks | `{worldSize, rank, 1}` |
| `CcoTeamLsa(devComm)` | 同节点 | `{lsaSize, lsaRank, 1}` |
| `CcoTeamCrossNode(devComm)` ⭐ | 跨节点（跳过自己节点）| `{worldSize - lsaSize, ?, 1}` |
| `CcoTeamRail(devComm)` | 跨节点同 rail | `{worldSize/lsaSize, rank/lsaSize, lsaSize}` |

转换公式：`worldRank = comm.rank + (teamRank - team.rank) * team.stride`

**3. Connection × Team 的兼容契约**

`gda.put(team, peer, ...)` 内部把 team rank 转成 QP 数组下标。两者必须匹配：

| Connection 设置 | 可用的 team（gda.put 接受） |
|-----------------|------------------------------|
| `NONE` | （无） |
| `FULL` | World / CrossNode / Rail / LSA 任意 |
| `CROSSNODE` | CrossNode / Rail（前提是 rail ⊆ crossnode；通常成立） |
| `RAIL` | Rail（或 Rail 的子集） |

注：LSA backend 不接受 team 参数（peer 永远是 intra-node local rank），同理 SDMA。Team 只用于 GDA。

**4. Requirements struct（用户显式填）**

```cpp
struct CcoDevCommRequirements {
  // forward-compat 三件套（必须由 INITIALIZER 宏填充）
  size_t   size;
  uint32_t magic;
  uint32_t version;

  // 资源链表（per-backend session 通过 CreateRequirement 往里加 buffer slot）
  CcoDevResourceRequirements* resourceRequirementsList;

  // ── GDA (RDMA) ──
  CcoGdaConnectionType gdaConnectionType;   // 默认 NONE
  int                  gdaContextCount;     // 独立 QP set 数量（hint，对应 numQpPerPe），默认 4
  int                  gdaSignalCount;      // signal slot 数（从 id=0 起），默认 16
  int                  gdaCounterCount;     // counter slot 数（从 id=0 起），默认 16
  int                  gdaQueueDepth;       // 0 = use provider default
  int                  gdaTrafficClass;     // -1 = use MORI_RDMA_TC env

  // ── LSA (P2P) ──
  int lsaBarrierCount;                      // CcoLsaBarrierSession 数量

  // ── SDMA ──
  int sdmaQueueCount;                       // 每 peer SDMA 队列数

  // ── Hybrid barrier ──
  int barrierCount;                         // LSA + GDA-Rail 二段式 barrier
};

#define CCO_DEV_COMM_REQUIREMENTS_INITIALIZER { \
  sizeof(CcoDevCommRequirements), CCO_API_MAGIC, CCO_API_VERSION, \
  /* resourceRequirementsList */ nullptr, \
  /* gda */ CCO_GDA_CONNECTION_NONE, 4, 16, 16, 0, -1, \
  /* lsa */ 0, \
  /* sdma */ 0, \
  /* hybrid barrier */ 0, \
}

struct CcoDevResourceRequirements {
  CcoDevResourceRequirements* next;
  size_t   bufferSize;
  size_t   bufferAlign;
  uint32_t* outBufferHandle;     // 创建后回填，是 comm 内部 buffer 的 offset (>>7)
  int      gdaSignalCount;
  int      gdaCounterCount;
  uint32_t* outGdaSignalStart;   // 回填：分配到的 signal id 起点
  uint32_t* outGdaCounterStart;
};
```

**5. 使用范式（kernel 端）**

```cpp
// host
CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
reqs.gdaSignalCount = CTA_COUNT;
reqs.lsaBarrierCount = CTA_COUNT;
CcoDevCommCreate(comm, &reqs, &devComm);

// kernel
__global__ void hybrid_alltoall(CcoDevComm* comm, CcoWindow_t win) {
  CcoLsa lsa;
  CcoGda gda(*comm, /*contextIdx=*/0);

  // intra-node 走 LSA（peer 是 lsa local rank）
  for (int p = 0; p < comm->lsaSize; p++) {
    if (p == comm->lsaRank) continue;
    lsa.put(p, win, dstOff, win, srcOff, bytes);
  }

  // 跨节点走 GDA + CrossNode team
  CcoTeam xnode = CcoTeamCrossNode(*comm);
  for (int p = 0; p < xnode.nRanks; p++) {
    gda.put(xnode, p, win, dstOff, win, srcOff, bytes,
            CcoGda_SignalInc{sigId});
  }
  gda.flush();
}
```

**6. 实现要点**

- **lsa topology 探测**：`CcoCommCreate` 时通过 `hipDeviceCanAccessPeer` + `LocalBootstrapNetwork` 确定哪些 rank 在同节点；存 `lsaSize`、`lsaRank` 到 `CcoComm` 和 `CcoDevComm`
- **CROSSNODE QP 分配**：host 端构造 peer endpoint 列表时，跳过 `[myNodeStart, myNodeStart + lsaSize)` 的 rank
- **device 端 rank→QP index 映射**：
  ```cpp
  __device__ int teamRankToGdaRank(CcoDevComm const& c, CcoTeam tm, int teamRank) {
    int wr = c.rank + (teamRank - tm.rank) * tm.stride;
    switch (c.gdaConnType) {
      case CCO_GDA_CONNECTION_FULL:      return wr;
      case CCO_GDA_CONNECTION_CROSSNODE: {
        int myNodeStart = (c.rank / c.lsaSize) * c.lsaSize;
        return wr < myNodeStart ? wr : wr - c.lsaSize;
      }
      case CCO_GDA_CONNECTION_RAIL:      return wr / c.lsaSize;
    }
  }
  ```
- **forward compat**：`CcoDevCommCreate` 入口检查 `reqs->size == sizeof(*reqs) && magic == CCO_API_MAGIC`，否则报错
- **degenerate case**：单节点跑 `CROSSNODE` 时 `nGdaRanks = 0`，host 自动降级为 NONE 并 warn

---

## 数据结构定义

### CcoComm（host 端，堆分配）

```cpp
struct CcoComm {
    int rank, worldSize;
    application::BootstrapNetwork* bootNet;
    application::Context*          ctx;       // RDMA 端点、传输类型协商

    // ── Topology (intra-node detection) ──
    int lsaSize;           // # of ranks on my node
    int lsaRank;           // my index within node [0..lsaSize)
    int myNodeStart;       // (rank / lsaSize) * lsaSize, world-rank of node[0]
    // 假设所有节点 lsaSize 相同（典型部署：8 GPU/节点）。
    // CcoCommCreate 时通过 hipDeviceCanAccessPeer + Allgather 探测。

    // VMM flat address space
    void*  flatBase;       // hipMemAddressReserve 返回的连续 VA 基址
    size_t perRankSize;    // 每 rank 的 VA slot 大小（用户指定，>= 所有 window 总大小）
    size_t nextOffset;     // slot 内下一个可用偏移

    // SDMA (per-comm，所有 window 共享，CcoCommCreate 时初始化)
    anvil::SdmaQueueDeviceHandle** sdmaDevHandles;
    int    sdmaNumQueue;   // 默认值，可被 DevCommRequirements.sdmaQueueCount 覆盖

    // 内存分配元数据（MemAlloc 时存入，WindowRegister(ptr) 时查询）
    struct AllocMeta {
        hipMemGenericAllocationHandle_t physHandle;
        int    shareFd;      // dma-buf FD，供 WindowRegister 时 RDMA MR 注册复用
        size_t slotOffset;   // 在 per-rank slot 内的起始偏移
        size_t size;
    };
    std::unordered_map<void*, AllocMeta> allocTable; // key = localPtr

    std::vector<CcoWindowHost*> windows; // 供 Destroy 时清理

    // ── DevComm 端点池 ──
    // RDMA endpoints 不再写死在 CcoComm 里，改为 CcoDevCommCreate 时按 reqs
    // 创建 ctx->CreateAdditionalEndpoints；CcoComm 只持有 Context 和默认参数。
};
```

### CcoDevComm（GPU 显存，kernel 接收此指针）

```cpp
struct CcoDevComm {
    // ── World / topology ──
    int rank, worldSize;
    int lsaSize, lsaRank;          // 从 CcoComm copy

    // ── GDA backend ──
    CcoGdaConnectionType gdaConnType;
    int                  gdaNumQpPerPe;        // = reqs.gdaContextCount
    int                  gdaNGdaRanks;         // 视 connType 而定
    ShmemRdmaEndpoint*   gdaEndpoints;         // GPU buf，长度 gdaNGdaRanks * gdaNumQpPerPe
    CcoIbgdaContext      ibgda;                // signal/counter resources

    // ── LSA / SDMA ──
    int       sdmaNumQueue;

    // ── Window lookup table ──
    void*  flatBase;
    size_t perRankSize;
    CcoWindowTableNode* windowTable;
};
typedef CcoDevComm* CcoDevComm_t;
```

### CcoWindowDevice（GPU 显存，kernel 接收此指针）

```cpp
struct CcoWindowDevice {
    // ── P2P / SDMA（同节点，本地 flat VA）──
    uintptr_t* p2pPeerPtrs; // [worldSize]，p2pPeerPtrs[pe] = flatBase + pe*perRankSize + slotOffset
                             // 仅同节点 P2P 可达 peer 有值，远程 peer 为 0
    // remote_va = p2pPeerPtrs[pe] + dstOff
    // local_va  = p2pPeerPtrs[rank] + srcOff（= localPtr + srcOff）

    // ── RDMA（跨节点 / 通用）──
    void*      localPtr;    // = flatBase + rank*perRankSize + slotOffset
    uintptr_t* peerPtrs;    // [worldSize]，RDMA 用地址
                             // iova=0 时：全部为 0，raddr = 0 + dstOff = dstOff
                             // iova=VA 时：peerPtrs[pe] = 远端 PE 的 localPtr（Allgather 交换）
    uint32_t*  peerRkeys;   // [worldSize]，Allgather 交换
    uint32_t   lkey;
    // 统一计算：raddr = peerPtrs[pe] + dstOff（两种 iova 模式代码一致）
    anvil::SdmaQueueDeviceHandle** deviceHandles_d; // 来自 comm->sdmaDevHandles，per-comm 共享
    HSAuint64* signalPtrs;       // [worldSize * sdmaNumQueue]
    HSAuint64* expectSignalsPtr; // [worldSize * sdmaNumQueue]
    HSAuint64** peerSignalPtrs;  // [worldSize]，各 pe 的 signal 地址
    uint32_t   sdmaNumQueue;
};
typedef CcoWindowDevice* CcoWindow_t;
```

### CcoWindowHost（host 端记录，供 Deregister 清理）

```cpp
struct CcoWindowHost {
    void*     localPtr;
    size_t    size;
    // RDMA MR 句柄（供 Deregister 时 deregister）
    uint32_t  lkey;
    // SDMA signal 数组（供 Deregister 时 hipFree）
    HSAuint64* signalPtrs;
    HSAuint64* expectSignalsPtr;
    HSAuint64** peerSignalPtrs;
    // GPU device 结构（供 Deregister 时 hipFree）
    CcoWindowDevice* devPtr;
    // GPU buf（供 Deregister 时 hipFree）
    uintptr_t* p2pPeerPtrs_gpu;
    uintptr_t* peerPtrs_gpu;
    uint32_t*  peerRkeys_gpu;
    HSAuint64** peerSignalPtrs_gpu;
};
```

---

## Host API

```cpp
// ── 阶段一：comm 初始化 ──
ncclResult_t CcoCommCreate(application::BootstrapNetwork* bootNet,
                               size_t perRankVmmSize,
                               CcoComm** comm);
ncclResult_t CcoCommDestroy(CcoComm* comm);

// ── 阶段 1.5（可选）：VMM 内存分配 + P2P flat space 映射 ──
// 不做 RDMA MR 注册；可在 WindowRegister 之前独立调用
ncclResult_t CcoMemAlloc(CcoComm* comm, size_t size, void** ptr);
ncclResult_t CcoMemFree(CcoComm* comm, void* ptr);

// ── 阶段二：window 注册（两个重载，三路传输同时就绪）──
// 重载 A：内部分配（= CcoMemAlloc + CcoWindowRegister(ptr)）
ncclResult_t CcoWindowRegister(CcoComm* comm, size_t size,
                                  CcoWindow_t* win, void** localPtr);
// 重载 B：接受 CcoMemAlloc 返回的 ptr
ncclResult_t CcoWindowRegister(CcoComm* comm, void* ptr, size_t size,
                                  CcoWindow_t* win);
ncclResult_t CcoWindowDeregister(CcoComm* comm, CcoWindow_t win);

// ── 阶段三：固化 GPU 端 comm 结构（带 requirements） ──
int CcoDevCommCreate(CcoComm* comm,
                     const CcoDevCommRequirements* reqs,
                     CcoDevComm** devComm);
int CcoDevCommDestroy(CcoDevComm* devComm);

// Host barrier
int CcoBarrierAll(CcoComm* comm);  // bootNet->Barrier()
```

**典型调用顺序（带 reqs）：**

```cpp
CcoCommCreate(bootNet, perRankVmmSize, &comm);

void *buf_a, *buf_b;
CcoWindowRegister(comm, size_a, &win_a, &buf_a);
CcoWindowRegister(comm, size_b, &win_b, &buf_b);

// ── 配置 DevComm 资源 ──
CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
reqs.gdaSignalCount    = CTA_COUNT;
reqs.gdaCounterCount   = CTA_COUNT;
reqs.lsaBarrierCount   = CTA_COUNT;

CcoDevCommCreate(comm, &reqs, &devComm);
my_kernel<<<grid, block>>>(devComm, win_a, win_b, ...);

CcoDevCommDestroy(devComm);
CcoWindowDeregister(comm, win_a);
CcoWindowDeregister(comm, win_b);
CcoCommDestroy(comm);
```

**MemAlloc + WindowRegister 分离形式同样兼容**，仅 `CcoDevCommCreate` 签名变化。

---

## 初始化流程（详细步骤）

### CcoCommCreate

```
1. new CcoComm
2. bootNet->Initialize()          → rank/worldSize 发现
3. new Context(*bootNet)          → RDMA 端点建立、传输类型协商、numQpPerPe
4. hipMemAddressReserve(&flatBase, worldSize * perRankVmmSize)
   → 预留连续 VA，slot[rank] = flatBase + rank * perRankVmmSize
   → 此时 slot 内没有物理内存映射（VA 仅保留）
5. InitSdmaContext()
   → 初始化 anvil SDMA 队列（参考 src/shmem/init.cpp 中 SDMA 初始化）
   → 存入 comm->sdmaDevHandles, comm->sdmaNumQueue
6. 从 ctx->GetRdmaEndpoints() 取 rdmaEndpoints，存入 comm->rdmaEndpoints
   从 ctx->GetNumQpPerPe() 取 numQpPerPe
```

### CcoMemAlloc（VMM 分配 + P2P flat space 映射）

```
CcoMemAlloc(comm, size, &ptr):
1. slotOffset = comm->nextOffset
2. 构建 hipMemAllocationProp allocProp：
     .type = hipMemAllocationTypePinned (或 Uncached)
     .requestedHandleType = hipMemHandleTypePosixFileDescriptor  ← 必须，否则无法 export FD
     .location = {hipMemLocationTypeDevice, currentDev}
   hipMemCreate(&physHandle, size, &allocProp, 0)
3. hipMemMap(flatBase + rank*perRankSize + slotOffset, size, physHandle)
   hipMemSetAccess(...)                         → 本 rank 可读写
4. hipMemExportToShareableHandle(physHandle,
       hipMemHandleTypePosixFileDescriptor)     → fd（dma-buf，同时供 P2P + RDMA 用）
5. ExchangeFileDescriptors(fd) → peerFDs[]
   仅在同节点 P2P 可达的 peer 之间交换（via LocalBootstrapNetwork）
   （参考 symmetric_memory.cpp 中 RegisterP2PPeerMemory() 的 FD exchange 逻辑）
6. for each peer pe where CanUseP2P(pe):
     hipMemImportFromShareableHandle(peerFDs[pe]) → importedHandle
     hipMemMap(flatBase + pe*perRankSize + slotOffset, size, importedHandle)
     hipMemSetAccess(...)
   → 结果：flatBase + pe*perRankSize + slotOffset 指向 pe 的物理内存
   注意：远程节点的 peer slot 保持无物理映射（P2P/SDMA 不可达，仅走 RDMA）
7. comm->nextOffset += alignUp(size, vmmGranularity)
8. localPtr = flatBase + rank*perRankSize + slotOffset
9. comm->allocTable[localPtr] = {physHandle, fd, slotOffset, size}
10. *ptr = localPtr
```

### CcoWindowRegister（重载 B：接受 ptr）

```
CcoWindowRegister(comm, ptr, size, &win):
0. meta = comm->allocTable[ptr]
   slotOffset = meta.slotOffset
   fd         = meta.shareFd
   localPtr   = ptr

── RDMA MR 注册（复用同一 dma-buf FD）──
1. ibv_reg_dmabuf_mr(pd, offset=0, size, iova=0, fd, access) → lkey, rkey
   （参考 VMMAllocChunk() 中 RegisterRdmaChunks()，
    底层调用 RegisterRdmaMemoryRegionDmabuf，需新增 iova=0 版本）
   Fallback：若 iova=0 不可用 → ibv_reg_dmabuf_mr(pd, 0, size, iova=localPtr, fd, access)
2. Allgather(rkey → peerRkeys[worldSize])
   Fallback iova=VA 时额外：Allgather(localPtr → peerPtrs[worldSize])
   ← iova=0 时 raddr = dstOff（不需要 peerPtrs）
   ← iova=VA 时 raddr = peerPtrs[pe] + dstOff

── SDMA signal 数组（per-window，不需要 MR）──
3. hipMalloc signalPtrs[worldSize * sdmaNumQueue]，hipMemset 0
4. hipMalloc expectSignalsPtr[worldSize * sdmaNumQueue]，hipMemset 0
5. Allgather(signalPtrs → peerSignalPtrs_host[worldSize])
   hipMalloc peerSignalPtrs_gpu[worldSize] + hipMemcpy H2D

── 构建 P2P 地址表 ──
6. 构建 p2pPeerPtrs_host[worldSize]：
     对每个 pe：p2pPeerPtrs_host[pe] = (CanUseP2P(pe) || pe==rank)
               ? flatBase + pe*perRankSize + slotOffset : 0
   hipMalloc p2pPeerPtrs_gpu + hipMemcpy H2D

── 构建 RDMA 地址表 ──
7. 构建 peerPtrs_host[worldSize]：
     iova=0 模式：全部填 0
     iova=VA 模式：Allgather 交换各 PE 的 localPtr
   hipMalloc peerPtrs_gpu + hipMemcpy H2D
   hipMalloc peerRkeys_gpu + hipMemcpy H2D

── 构建 GPU 端 CcoWindowDevice ──
8. 填 CcoWindowDevice shadow：
   .localPtr         = localPtr
   .p2pPeerPtrs      = p2pPeerPtrs_gpu
   .peerPtrs         = peerPtrs_gpu
   .peerRkeys        = peerRkeys_gpu
   .lkey             = lkey
   .deviceHandles_d  = comm->sdmaDevHandles   ← per-comm 共享，直接填指针
   .signalPtrs       = signalPtrs
   .expectSignalsPtr = expectSignalsPtr
   .peerSignalPtrs   = peerSignalPtrs_gpu
   .sdmaNumQueue     = comm->sdmaNumQueue
9. hipMalloc CcoWindowDevice（GPU 显存）+ hipMemcpy H2D → devPtr
10. new CcoWindowHost{...}，push_back 到 comm->windows
11. *win = devPtr
```

### CcoWindowRegister（重载 A：内部分配）

```
CcoWindowRegister(comm, size, &win, &localPtr):
→ CcoMemAlloc(comm, size, &ptr)
→ CcoWindowRegister(comm, ptr, size, win)
→ *localPtr = ptr
```

### CcoDevCommCreate

```
CcoDevCommCreate(comm, &devComm):
1. 填 CcoDevComm host shadow：
   .rank            = comm->rank
   .worldSize       = comm->worldSize
   .numQpPerPe      = comm->numQpPerPe
   .rdmaEndpoints   → hipMalloc[worldSize*numQpPerPe] + hipMemcpy H2D
   .flatBase        = comm->flatBase
   .perRankSize     = comm->perRankSize
2. hipMalloc CcoDevComm（GPU 显存）+ hipMemcpy H2D
3. *devComm = GPU 指针（直接作为 kernel 参数传入）
```

---

## Device API（NCCL 风格 session class，参见"命名约定"）

CCO device API 分两层：

1. **通用辅助**（`include/mori/cco/cco_device_api.hpp`）
   - `findWindow(comm, ptr)` — 在 windowTable 里查 window
   - `getPeerPtr(win, pe, off)` / `getLocalPtr(win, off)` — 计算 flat VA 地址（P2P / SDMA 用）
   - 在 `mori::cco` namespace 内，**不带** `Cco` 前缀，camelCase

2. **per-backend session class**（每个 backend 一个子目录）

```cpp
// ── GDA backend (RDMA via NIC GPU-direct, ncclGin 同款) ──
// include/mori/cco/gda/gda_device_api.hpp
namespace mori::cco::gda {

// Tag types (template dispatch)
struct CcoGda_NoSignal {};
struct CcoGda_NoCounter {};
struct CcoGda_SignalInc { CcoGdaSignal_t signalId; };
struct CcoGda_SignalAdd { CcoGdaSignal_t signalId; uint64_t value; };
struct CcoGda_CounterInc { CcoGdaCounter_t counterId; };

// Handles
typedef uint32_t CcoGdaSignal_t;
typedef uint32_t CcoGdaCounter_t;
typedef void*    CcoGdaRequest_t;

// Session
struct CcoGda {
  CcoDevComm const& comm;
  uint32_t contextId;
  CcoGdaCtx ctx;
  void* _gdaHandle;

  __device__ CcoGda(CcoDevComm const&, int contextIndex);

  template <typename RemoteAction = CcoGda_NoSignal,
            typename LocalAction  = CcoGda_NoCounter>
  __device__ void put(int peer, CcoWindow_t dstWin, size_t dstOff,
                      CcoWindow_t srcWin, size_t srcOff, size_t bytes,
                      RemoteAction = CcoGda_NoSignal{},
                      LocalAction  = CcoGda_NoCounter{});

  template <typename T, typename RemoteAction = CcoGda_NoSignal>
  __device__ void putValue(int peer, CcoWindow_t dstWin, size_t dstOff,
                           T value, RemoteAction = CcoGda_NoSignal{});

  __device__ void get(int peer, CcoWindow_t remoteWin, size_t remoteOff,
                      CcoWindow_t localWin, size_t localOff, size_t bytes);
  template <typename RemoteAction>
  __device__ void signal(int peer, RemoteAction);

  __device__ uint64_t readSignal (CcoGdaSignal_t,  int bits = 64);
  __device__ void     waitSignal (CcoGdaSignal_t,  uint64_t least, int bits = 64);
  __device__ void     resetSignal(CcoGdaSignal_t);
  __device__ uint64_t readCounter (CcoGdaCounter_t, int bits = 56);
  __device__ void     waitCounter (CcoGdaCounter_t, uint64_t least, int bits = 56);
  __device__ void     resetCounter(CcoGdaCounter_t);

  __device__ void flush();
  __device__ void flushAsync(int peer, CcoGdaRequest_t* outRequest);
  __device__ void wait(CcoGdaRequest_t& request);
};

}  // namespace mori::cco::gda

// ── LSA backend (intra-node P2P direct store)：Phase 2 ──
// include/mori/cco/lsa/lsa_device_api.hpp
namespace mori::cco::lsa {
struct CcoLsa {
  __device__ void put(int peer, CcoWindow_t dst, size_t dstOff,
                      CcoWindow_t src, size_t srcOff, size_t bytes);
  template <typename T>
  __device__ void putValue(int peer, CcoWindow_t dst, size_t dstOff, T value);
};

struct CcoLsaBarrierSession {
  template <typename Coop>
  __device__ void arrive(Coop, cuda::memory_order);
  template <typename Coop>
  __device__ void wait  (Coop, cuda::memory_order);
  template <typename Coop>
  __device__ void sync  (Coop, cuda::memory_order);
};
}  // namespace mori::cco::lsa

// ── SDMA backend (intra-node SDMA copy engine)：Phase 2 ──
// include/mori/cco/sdma/sdma_device_api.hpp
namespace mori::cco::sdma {
struct CcoSdma {
  __device__ void put(int peer, CcoWindow_t dst, size_t dstOff,
                      CcoWindow_t src, size_t srcOff, size_t bytes, int queueId = 0);
  __device__ void quiet(int peer, int queueId = 0);
};
}  // namespace mori::cco::sdma
```

**用户使用范式**（per-CTA 实例化 session 后调方法）：

```cpp
__global__ void my_kernel(CcoDevComm* comm,
                          CcoWindow_t dst, CcoWindow_t src) {
  // RDMA put with remote signal
  mori::cco::gda::CcoGda gda(*comm, /*contextIndex=*/0);
  gda.put(peer, dst, dstOff, src, srcOff, bytes,
          mori::cco::gda::CcoGda_SignalInc{sigId});
  gda.flush();

  // P2P put (intra-node)
  mori::cco::lsa::CcoLsa lsa;
  lsa.put(peer, dst, dstOff, src, srcOff, bytes);

  // SDMA put (intra-node)
  mori::cco::sdma::CcoSdma sdma;
  sdma.put(peer, dst, dstOff, src, srcOff, bytes);
  sdma.quiet(peer);
}
```

**字段依赖一览**：

| backend | session class | 来自 `CcoDevComm` | 来自 `CcoWindowDevice` |
|---------|---------------|------------------|------------------------|
| GDA (RDMA) | `CcoGda` | `ibgda` (QP endpoints + signal/counter) | `ibgdaWin` (rkeys + lkey) |
| LSA (P2P)  | `CcoLsa`  | — | `winBase`, `stride4G`, `rank` (flat VA addressing) |
| SDMA       | `CcoSdma` | — | `deviceHandles_d`, `signalPtrs`, `expectSignalsPtr`, `peerSignalPtrs` |

---

## 文件结构

```
include/mori/cco/
├── cco_types.hpp           ← Host/device 共享类型：CcoComm, CcoDevComm,
│                              CcoWindowDevice, CcoWindowHost, CcoIbgdaContext
├── cco_api.hpp             ← Host API 声明（mori 风格）
├── cco_device.hpp          ← Device API umbrella，include 下面所有
├── cco_device_api.hpp      ← 通用 device 辅助：findWindow, getPeerPtr, getLocalPtr
└── gda/                    ← GDA (RDMA) backend (NCCL 风格 session)
    ├── gda_device_common.hpp  ← CcoGda struct 声明 + tag 类型 + handle typedef
    └── gda_device_api.hpp     ← CcoGda 成员函数实现 + namespace 内 free function

(后续阶段)
└── lsa/                    ← LSA (P2P direct store) backend，TODO
    ├── lsa_device_common.hpp
    └── lsa_device_api.hpp
└── sdma/                   ← SDMA backend，TODO
    ├── sdma_device_common.hpp
    └── sdma_device_api.hpp

src/cco/
├── cco_init.cpp        ← CommCreate/Destroy, DevCommCreate/Destroy, MemAlloc/Free, BarrierAll
└── cco_memory.cpp      ← WindowRegister/Deregister
```

CMakeLists.txt 新增 `mori_cco` target，链接 `mori_application`（Context 等）。

---

## 可直接复用的现有代码

| 需要做的事 | 参考位置 |
|-----------|---------|
| VMM：hipMemAddressReserve 预留连续 VA | `symmetric_memory.cpp`：`InitializeVMMHeap()` |
| VMM：hipMemCreate + hipMemMap 本 rank | `symmetric_memory.cpp`：`VMMAllocChunk()` |
| VMM：FD export + ExchangeFileDescriptors | `symmetric_memory.cpp`：`VMMAllocChunk()` |
| VMM：peer hipMemImport + hipMemMap | `symmetric_memory.cpp`：`VMMAllocChunk()` P2P import 段 |
| RDMA：dma-buf FD → RDMA MR（iova=0 via ibv_reg_dmabuf_mr） | `symmetric_memory.cpp`：`RegisterRdmaChunks()`，需修改 `RegisterRdmaMemoryRegionDmabuf()` 支持 iova=0 参数 |
| Allgather peerPtrs / peerRkeys | `symmetric_memory.cpp`：`RegisterSymmMemObj()` |
| SDMA：signal 数组分配 + Allgather | `symmetric_memory.cpp`：`RegisterSymmMemObj()` SDMA 段 |
| SDMA：anvil context 初始化 | `src/shmem/init.cpp` SDMA 初始化 |
| ShmemRdmaEndpoint 结构体 | `include/mori/shmem/internal.hpp` |
| Device barrier | `ShmemInternalBarrierBlock`（`shmem_device_api.hpp`）|
| P2P put kernel | p2p provider（`shmem_device_api.hpp`）|
| RDMA put kernel | ibgda provider（`shmem_device_api.hpp`）|
| SDMA put kernel | `core::SdmaPutThread`（`shmem_sdma_kernels.hpp`）|

**不要用**：`SymmMemManager`（内部 hipMalloc，无法支持 flat VA）、`DISPATCH_TRANSPORT_TYPE` 宏、`ShmemStatesSingleton`。

---

## 第一阶段 Scope（只做 host 端）—— ✅ 已完成

Device API 骨架（声明 + 注释）也要写，实现留后续迭代：

1. ✅ `CcoCommCreate` / `CcoCommDestroy`（含 lsa detection、HeapVAManager、auto-deregister straggler windows）
2. ✅ `CcoMemAlloc` / `CcoMemFree`（HeapVAManager 管理 flat VA 槽，按 lsaRank 切分）
3. ✅ `CcoWindowRegister`（两个重载）/ `CcoWindowDeregister`（含 VMM + P2P import + RDMA MR + Allgather rkeys + leak hardening）
4. ✅ `CcoDevCommCreate` / `CcoDevCommDestroy`（含 connType FULL/CROSSNODE/RAIL/NONE + resource window pool + inline 优化）
5. ✅ `CcoBarrierAll`
6. ✅ 头文件骨架（types, api, device_api 声明）

### Phase 1 → Phase 2 后续 TODO

- [ ] `CcoLsaBarrierSession` / `CcoGdaSession` / `CcoSdmaSession` device class
- [ ] `resourceRequirementsList` 接通（把 session 描述的 buffer 都 sub-allocate 进 resource window）
- [ ] `gdaQueueDepth` / `gdaTrafficClass` 透传到 `RdmaEndpointConfig`
- [ ] SDMA reqs (`sdmaQueueCount`) — 等 device 端 SDMA session 落地
- [ ] 用户 buffer 注册 API（接受任意指针，不要求来自 `CcoMemAlloc`）
- [ ] 每 window 的 backend 选择 flag（RDMA / P2P / SDMA 按需开关）
- [ ] `CcoWindowRegister` 异常安全的 scope guard（替代手写 rollback）
- [ ] `CcoComm` 跟踪 live DevComm 列表，`CcoCommDestroy` 自动清理

---

## reqs 字段进度

`CcoDevCommRequirements` 各字段当前生效情况（截至 `fa92cca0`）：

| 字段 | 状态 | 说明 |
|------|------|------|
| `size` / `magic` / `version` | ✅ 已生效 | `CcoDevCommCreate` 入口校验 |
| `gdaConnectionType = NONE` | ✅ 已生效 | 完全跳过 QP 创建；空 peerMask |
| `gdaConnectionType = CROSSNODE` | ✅ 已生效 | `cap.canRDMA && !cap.sameHost`；单节点自动 collapse 到 NONE |
| `gdaConnectionType = FULL` | ✅ 已生效 | `cap.canRDMA` 全部 peer（除 self） |
| `gdaConnectionType = RAIL` | ✅ 已生效 | `cap.canRDMA && !cap.sameHost && peer%lsaSize == myLsaRank`；2-node 验证 QP=`(nNodes-1)*qpsPerPe` |
| `gdaContextCount` | ✅ 已生效 | numQpPerPe |
| `gdaSignalCount` | ✅ 已生效 | IBGDA signalBuf 大小（resource window 内 offset 0） |
| `gdaCounterCount` | ✅ 已生效 | IBGDA counterBuf 大小（resource window 内） |
| `gdaQueueDepth` | ❌ TODO | 透传给 `RdmaEndpointConfig`，~15 行 |
| `gdaTrafficClass` | ❌ TODO | 透传给 RDMA endpoint（目前用 `MORI_RDMA_TC` env），~20 行 |
| `sdmaQueueCount` | ❌ TODO | 等 device 端 SDMA session 落地后再做（现在用 anvil 默认） |
| `lsaBarrierCount` | ✅ host 已生效 | resource window 内 sub-allocate `(3N + N*lsaSize)*4` 字节，handle 存 `devComm.lsaBarrier = {bufOffset, nBarriers}`；device session class 未实装 |
| `railGdaBarrierCount` | ✅ host 已生效 | 复用 IBGDA signal pool，handle 存 `devComm.railGdaBarrier = {signal0, nBarriers}`；nNodes==1 或 connType==NONE 时自动 collapse 为 disabled |
| `barrierCount` | ✅ host 已生效 | 同时驱动 `hybridLsaBarrier`（resource window 内）+ `hybridRailGdaBarrier`（IBGDA signal pool）一对 handle，构成两阶段 world barrier |
| `resourceRequirementsList` | ❌ TODO | 等 device 端 `CcoLsa` / `CcoSdma` session 落地（resource window 已经支持 sub-allocation 的底座） |

`MORI_CCO_LOG_TRANSPORT=1` 可在 `CcoDevCommCreate` 之后打印 per-rank
transport 矩阵（CAP=硬件能力 / ACT=本 DevComm 实例化的），用于验证
connType 的实际行为。

---

## Phase 2 进展：Resource Window + Inline

CCO host API 当前已经对齐 NCCL 的 resource-window 模型：

**1. IBGDA 资源池 → 单一 symmetric window**（`cdf00b13`）

`CcoDevCommCreate` 不再为 signalBuf / signalShadows / counterBuf 各
分配一块，而是一次 `CcoMemAlloc + CcoWindowRegister` 出一个 "resource
window"，三个 buffer 作 sub-pointer 落进去。

> **底层不走 `hipMalloc`**：`CcoMemAlloc` 用的是 VMM API
> （`hipMemCreate` 分配物理 handle + `hipMemMap` 映射到 LSA flat VA
> 里的预留槽），这样才能既被映射到 symmetric 的固定 VA、又能 export
> 成 dma-buf FD 注册 RDMA MR + P2P import 给 peer。`hipMalloc` 在 CCO
> 里只用于不需要 peer 访问的小 staging buffer（如 epsGpu / windowTable
> nodes / sdmaDevHandles）。

这个 window 是完整的 CCO symmetric window：

- 位于 LSA flat VA，**peer 可 P2P-load/store 访问**（`CcoLsaBarrier`
  直接走这条）
- 有 RDMA MR，**peer 可 RDMA-write**（IBGDA signal atomic add 走这条）
- rkey 自动经 `CcoWindowRegister` 内部 Allgather 分发，存
  `resourceWindow->ibgdaWin.peerRkeys`

`signalBuf` 在 resource window 内 offset 0，让 device-side RDMA raddr
仍是 `slot_id * sizeof(uint64)`，跟旧寻址兼容。

**1b. 四个 barrier handle 同步落地（对齐 NCCL `ncclDevComm`）**

NCCL `ncclDevComm` 公开的 4 个 barrier handle 全部加上（除了
`lsaMultimem`——NV-only），由两类 handle 组合：

| Handle 字段（CcoDevComm） | 类型 | 大小 | 资源宿主 | 驱动 reqs |
|---|---|---|---|---|
| `lsaBarrier` | `CcoLsaBarrierHandle` | 8B | resource window 内 sub-allocate `(3N+N*lsaSize)*4` 字节 | `lsaBarrierCount` |
| `hybridLsaBarrier` | `CcoLsaBarrierHandle` | 8B | resource window 内 sub-allocate（同上公式，N=barrierCount） | `barrierCount` |
| `railGdaBarrier` | `CcoGdaBarrierHandle` | 8B | IBGDA signal pool 占 `N*nNodes` 个 slot | `railGdaBarrierCount` |
| `hybridRailGdaBarrier` | `CcoGdaBarrierHandle` | 8B | IBGDA signal pool 占 `N*nNodes` 个 slot | `barrierCount` |

`CcoLsaBarrierHandle = {uint32_t bufOffset, int nBarriers}`，对应 NCCL
`ncclLsaBarrierHandle`。
`CcoGdaBarrierHandle = {uint32_t signal0, int nBarriers}`，对应 NCCL
`ncclGinBarrierHandle`（signal id 是 uint32，slot 值是 uint64）。

```
resource window
├─ [offset 0]            ibgda.signalBuf       (RDMA atomic add 目标 / GDA barrier slots)
│   ├─ [0..gdaSignalCount)              user 显式申请的 signal
│   ├─ [.., +N*nNodes)                  railGdaBarrier   ← signal0 起点
│   └─ [.., +M*nNodes)                  hybridRailGdaBarrier
├─ [offset signal]       ibgda.signalShadows
├─ [offset counter]      ibgda.counterBuf
├─ [offset lsaBarrier]   LSA barrier slab     (lsaBarrier.bufOffset)
└─ [offset hybLsa]       hybrid LSA slab      (hybridLsaBarrier.bufOffset)
```

实测 (`lsaSize=8`, `gdaSignalCount=16`, `gdaCounterCount=16`,
`lsaBarrierCount=4`, `barrierCount=3`, `railGdaBarrierCount=2`)：

| connType | totalSize | lsaBarOff | hybLsaBarOff | signals | railGdaSig0 | hybRailGdaSig0 |
|---|---|---|---|---|---|---|
| NONE (1 node) | 388 | 0x0 | 0x100 | 0 | 0 (collapsed) | 0 (collapsed) |
| FULL (1 node) | 772 | 0x180 | 0x280 | 16 | 16 (collapsed) | 16 (collapsed) |

> nNodes==1 时所有 rail GDA handle 的 `nBarriers` 强制清零（disabled），
> 因为没有跨节点 peer 可寻址；`signal0` 值仍然在累加位置上，方便
> 未来 2-node 测试自然激活。

**Collapse 规则**：

- `connType == NONE` 或 `nNodes == 1` → `railGdaBarrier`、`hybridRailGdaBarrier` 自动 disable（`nBarriers=0`）
- 用户传 `count=0` 的字段（默认 initializer 全 0）→ 对应 handle 完全跳过分配
- `lsaBarrier` / `hybridLsaBarrier` 单节点也能工作（intra-node P2P），不 collapse

Device session class 暂未实装；4 个 handle 已经能让 kernel 直接寻址：

- LSA：`winBase + peerLsa*stride4G<<32 + bufOffset + barrierIdx*lsaSize*4 + myLsa*4`
- GDA：RDMA atomic_add 到 `raddr = (signal0 + barrierIdx*nNodes + myNodeIdx) * 8`

**2. resource window 内嵌 `CcoDevComm`**（`fa92cca0`）

跟 NCCL 同款：

```cpp
struct CcoDevComm {
  ...
  CcoWindowDevice* resourceWindow;          // GPU pointer (身份)
  CcoWindowDevice  resourceWindow_inlined;  // 32B 内嵌拷贝 (数据)
  ...
};
```

Kernel 读 `winBase` / `stride4G` / `ibgdaWin.{lkey,peerRkeys}` 直接走
cmem，不必通过 `resourceWindow` 指针访问 GPU global。`CcoDevComm`
从 152B → 184B，仍远在 4KB kernel param 上限内。

**3. DevCommCreate 分配次数下降**

分清"symmetric 内存"和"staging 内存"两类：

|  | 旧 (Phase 1) | 现在 |
|---|---|---|
| Symmetric IBGDA pool（VMM + dma-buf + RDMA MR + P2P import） | 3 块独立分配，分别注册 | **1 块**（resource window）一次 alloc + 一次 register，覆盖 P2P + RDMA 双通道 |
| `signalBuf` MR 注册 | 仅 signalBuf 一段 | 整个 resource window 全段都可 P2P/RDMA |
| Host staging hipMalloc（`epsGpu` / windowTable nodes / sdmaDevHandles / devCommGpu） | ~6 | ~6（未变；这些不需要 peer 访问） |

剩下的 staging buffer（epsGpu / windowTable nodes / sdma.* / devCommGpu）
都用 `hipMalloc` 即可，不需要并入 resource window。Phase 2.3 接通
`resourceRequirementsList` 后，所有需要 peer 访问的 session buffer
（LSA barrier、LLA2A staging、用户自定义 session）都会自动沉淀进
resource window 这一块 VMM 分配里。

---

## SPMT (单进程多线程) 支持

CCO **天然 SPMT-friendly**，无 process-global singleton 漏出：

| Backend | SPMT 处理 |
|---|---|
| Bootstrap (SocketBootstrap) | 跨线程通过 TCP loopback 自动 work |
| RDMA Context / QP | 每线程独立 `Context`，独立 NIC QP set |
| anvil SDMA queue | 单例已经 `(srcDev,dstDev)` keyed + mutex（PR #308） |
| SDMA signal pool | `Context::SameProcessP2P(peer)` 区分 → 同进程用 raw VA + `hipDeviceEnablePeerAccess`；跨进程用 `hipIpcOpenMemHandle` |
| Resource window FD exchange | `LocalBootstrapNetwork` 的 SCM_RIGHTS 对同进程也 work（kernel 不区分），仅启动稍慢 |

用户契约（满足这两条即可）：

1. **每个 thread 一个 `CcoComm`**（不要跨线程共享 comm 句柄）
2. **每个 thread 在 `CcoCommCreate` 之前 `hipSetDevice(...)`** —— comm
   会 cache `cudaDev`，后续 API 调用必须保持线程绑在同一 device

测试覆盖：`test_cco_host`（8 thread SPMT）+ `test_cco_gda_modes`
（同样 SPMT, 4 个 connType × 8 thread）均通过。

---

## 验证

1. 两线程各自 `CcoCommCreate` + `CcoWindowRegister`，互不干扰（验证无 singleton）
2. 改写 `examples/shmem/put_thread_allgather.cpp` 使用 CCO API
3. 同进程两个 comm 并发 put + barrier，验证资源互不污染
