# XSHMEM 工作交接（完整版）

你是接受这个任务的 agent。mori 代码库在 `/home/jiahzhou/workspace/mori`。你的任务是实现 XSHMEM 模块（host 端初始化，device API 骨架）。下面包含你需要的所有信息。

---

## 背景

mori 是一个 GPU 通信库，有 `mori-SHMEM` 模块提供 GPU-initiated P2P / RDMA / SDMA 传输。当前 SHMEM 用 Meyer's singleton（`ShmemStatesSingleton`）管理全局状态，一个进程只能有一套通信上下文。

**XSHMEM 目标**：实现类似 NCCL LSA（P2P）+ GIN（RDMA）+ SDMA 的功能，用显式 comm 句柄替代 singleton，支持单进程内多个独立 comm 实例。

核心特性：
1. **无 singleton**：每个 `XshmemComm` 独立堆分配，多线程可并发使用
2. **三段式初始化**：`CommCreate` → `WindowRegister` → `DevCommCreate`
3. **三条显式传输路径**：P2P（直接 GPU store）、RDMA（ibgda）、SDMA（DMA 引擎），用户在 kernel 里显式选择，不做自动 dispatch
4. **统一 VMM 内存**：window 底层用 `hipMemCreate`，一次注册同时具备三条路径能力

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

与现有 `SymmMemObj` 一致，`XshmemWindowDevice` 维护两套地址表：

- **`p2pPeerPtrs[pe]`**（P2P / SDMA 用）：本地 flat VA，`= flatBase + pe*perRankSize + slotOffset`。仅同节点 P2P 可达 peer 有值，远程 peer 为 0
- **`peerPtrs[pe]`**（RDMA 用）：iova=0 时全部为 0；iova=VA fallback 时存远端 PE 的 localPtr

三条路径的寻址：

- **P2P**：`remote = p2pPeerPtrs[pe] + dstOff`（仅同节点可达）
- **RDMA**：`raddr = peerPtrs[pe] + dstOff`（iova=0 时 = dstOff；iova=VA 时 = 远端VA + dstOff。两种模式同一份 kernel 代码）
- **SDMA**：`dstPtr = p2pPeerPtrs[pe] + dstOff`（与 P2P 共用，仅同节点可达）

一次 `XshmemWindowRegister` 同时拥有三条路径的能力。

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

- `XshmemMemAlloc`：VMM 分配 + P2P flat space 映射，**不做** RDMA MR 注册
- `XshmemWindowRegister(comm, ptr, size, win)`：接受 MemAlloc 的 ptr，做 RDMA MR 注册 + SDMA signal setup + 构建 GPU device 结构
- `XshmemWindowRegister(comm, size, win, &ptr)`：便捷重载，内部 = MemAlloc + WindowRegister(ptr)

---

## 数据结构定义

### XshmemComm（host 端，堆分配）

```cpp
struct XshmemComm {
    int rank, worldSize;
    application::BootstrapNetwork* bootNet;
    application::Context*          ctx;       // RDMA 端点、传输类型协商

    // VMM flat address space
    void*  flatBase;       // hipMemAddressReserve 返回的连续 VA 基址
    size_t perRankSize;    // 每 rank 的 VA slot 大小（用户指定，>= 所有 window 总大小）
    size_t nextOffset;     // slot 内下一个可用偏移

    // RDMA
    std::vector<ShmemRdmaEndpoint> rdmaEndpoints; // DevCommCreate 时 H2D copy
    int    numQpPerPe;

    // SDMA（per-comm，所有 window 共享）
    anvil::SdmaQueueDeviceHandle** sdmaDevHandles;
    int    sdmaNumQueue;

    // Barrier
    uint64_t* internalSyncGpuPtr; // GPU 显存，128×uint64_t

    // 内存分配元数据（MemAlloc 时存入，WindowRegister(ptr) 时查询）
    struct AllocMeta {
        hipMemGenericAllocationHandle_t physHandle;
        int    shareFd;      // dma-buf FD，供 WindowRegister 时 RDMA MR 注册复用
        size_t slotOffset;   // 在 per-rank slot 内的起始偏移
        size_t size;
    };
    std::unordered_map<void*, AllocMeta> allocTable; // key = localPtr

    std::vector<XshmemWindowHost*> windows; // 供 Destroy 时清理
};
```

### XshmemDevComm（GPU 显存，kernel 接收此指针）

```cpp
struct XshmemDevComm {
    int rank, worldSize, numQpPerPe;
    ShmemRdmaEndpoint* rdmaEndpoints;  // GPU buf，长度 worldSize*numQpPerPe
    uint64_t*          internalSyncPtr; // GPU buf，128×uint64_t
    void*  flatBase;
    size_t perRankSize;
};
typedef XshmemDevComm* XshmemDevComm_t;
```

### XshmemWindowDevice（GPU 显存，kernel 接收此指针）

```cpp
struct XshmemWindowDevice {
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
typedef XshmemWindowDevice* XshmemWindow_t;
```

### XshmemWindowHost（host 端记录，供 Deregister 清理）

```cpp
struct XshmemWindowHost {
    void*     localPtr;
    size_t    size;
    // RDMA MR 句柄（供 Deregister 时 deregister）
    uint32_t  lkey;
    // SDMA signal 数组（供 Deregister 时 hipFree）
    HSAuint64* signalPtrs;
    HSAuint64* expectSignalsPtr;
    HSAuint64** peerSignalPtrs;
    // GPU device 结构（供 Deregister 时 hipFree）
    XshmemWindowDevice* devPtr;
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
ncclResult_t XshmemCommCreate(application::BootstrapNetwork* bootNet,
                               size_t perRankVmmSize,
                               XshmemComm** comm);
ncclResult_t XshmemCommDestroy(XshmemComm* comm);

// ── 阶段 1.5（可选）：VMM 内存分配 + P2P flat space 映射 ──
// 不做 RDMA MR 注册；可在 WindowRegister 之前独立调用
ncclResult_t XshmemMemAlloc(XshmemComm* comm, size_t size, void** ptr);
ncclResult_t XshmemMemFree(XshmemComm* comm, void* ptr);

// ── 阶段二：window 注册（两个重载，三路传输同时就绪）──
// 重载 A：内部分配（= XshmemMemAlloc + XshmemWindowRegister(ptr)）
ncclResult_t XshmemWindowRegister(XshmemComm* comm, size_t size,
                                  XshmemWindow_t* win, void** localPtr);
// 重载 B：接受 XshmemMemAlloc 返回的 ptr
ncclResult_t XshmemWindowRegister(XshmemComm* comm, void* ptr, size_t size,
                                  XshmemWindow_t* win);
ncclResult_t XshmemWindowDeregister(XshmemComm* comm, XshmemWindow_t win);

// ── 阶段三：固化 GPU 端 comm 结构 ──
ncclResult_t XshmemDevCommCreate(XshmemComm* comm, XshmemDevComm** devComm);
ncclResult_t XshmemDevCommDestroy(XshmemDevComm* devComm);

// Host barrier
ncclResult_t XshmemBarrierAll(XshmemComm* comm);  // bootNet->Barrier()
```

**典型调用顺序 A（全自动）：**

```cpp
XshmemCommCreate(bootNet, perRankVmmSize, &comm);

void *buf_a, *buf_b;
XshmemWindowRegister(comm, size_a, &win_a, &buf_a);
XshmemWindowRegister(comm, size_b, &win_b, &buf_b);

XshmemDevCommCreate(comm, &devComm);
my_kernel<<<grid, block>>>(devComm, win_a, win_b, ...);

XshmemDevCommDestroy(devComm);
XshmemWindowDeregister(comm, win_a);
XshmemWindowDeregister(comm, win_b);
XshmemCommDestroy(comm);
```

**典型调用顺序 B（分离，类比 ncclMemAlloc + ncclCommWindowRegister）：**

```cpp
XshmemCommCreate(bootNet, perRankVmmSize, &comm);

void *buf_a, *buf_b;
XshmemMemAlloc(comm, size_a, &buf_a);
XshmemMemAlloc(comm, size_b, &buf_b);
// 此时 buf 已可 P2P 访问，可做其他事

XshmemWindowRegister(comm, buf_a, size_a, &win_a);  // 仅 RDMA MR + SDMA 注册
XshmemWindowRegister(comm, buf_b, size_b, &win_b);

XshmemDevCommCreate(comm, &devComm);
my_kernel<<<grid, block>>>(devComm, win_a, win_b, ...);

XshmemDevCommDestroy(devComm);
XshmemWindowDeregister(comm, win_a);
XshmemWindowDeregister(comm, win_b);
XshmemMemFree(comm, buf_a);
XshmemMemFree(comm, buf_b);
XshmemCommDestroy(comm);
```

---

## 初始化流程（详细步骤）

### XshmemCommCreate

```
1. new XshmemComm
2. bootNet->Initialize()          → rank/worldSize 发现
3. new Context(*bootNet)          → RDMA 端点建立、传输类型协商、numQpPerPe
4. hipMemAddressReserve(&flatBase, worldSize * perRankVmmSize)
   → 预留连续 VA，slot[rank] = flatBase + rank * perRankVmmSize
   → 此时 slot 内没有物理内存映射（VA 仅保留）
5. InitSdmaContext()
   → 初始化 anvil SDMA 队列（参考 src/shmem/init.cpp 中 SDMA 初始化）
   → 存入 comm->sdmaDevHandles, comm->sdmaNumQueue
6. AllocateInternalSync()
   → hipMalloc 128×uint64_t → comm->internalSyncGpuPtr
   → Allgather（各 rank 交换 GPU 地址，使 barrier 可跨 rank 原子操作）
7. 从 ctx->GetRdmaEndpoints() 取 rdmaEndpoints，存入 comm->rdmaEndpoints
   从 ctx->GetNumQpPerPe() 取 numQpPerPe
```

### XshmemMemAlloc（VMM 分配 + P2P flat space 映射）

```
XshmemMemAlloc(comm, size, &ptr):
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

### XshmemWindowRegister（重载 B：接受 ptr）

```
XshmemWindowRegister(comm, ptr, size, &win):
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

── 构建 GPU 端 XshmemWindowDevice ──
8. 填 XshmemWindowDevice shadow：
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
9. hipMalloc XshmemWindowDevice（GPU 显存）+ hipMemcpy H2D → devPtr
10. new XshmemWindowHost{...}，push_back 到 comm->windows
11. *win = devPtr
```

### XshmemWindowRegister（重载 A：内部分配）

```
XshmemWindowRegister(comm, size, &win, &localPtr):
→ XshmemMemAlloc(comm, size, &ptr)
→ XshmemWindowRegister(comm, ptr, size, win)
→ *localPtr = ptr
```

### XshmemDevCommCreate

```
XshmemDevCommCreate(comm, &devComm):
1. 填 XshmemDevComm host shadow：
   .rank            = comm->rank
   .worldSize       = comm->worldSize
   .numQpPerPe      = comm->numQpPerPe
   .rdmaEndpoints   → hipMalloc[worldSize*numQpPerPe] + hipMemcpy H2D
   .internalSyncPtr = comm->internalSyncGpuPtr（已在 GPU，直接填）
   .flatBase        = comm->flatBase
   .perRankSize     = comm->perRankSize
2. hipMalloc XshmemDevComm（GPU 显存）+ hipMemcpy H2D
3. *devComm = GPU 指针（直接作为 kernel 参数传入）
```

---

## Device API（`include/mori/xshmem/xshmem_device_api.hpp`）

```cpp
// ── P2P：直接 GPU store，同机 xGMI ──
__device__ inline void XshmemP2pPutThread(
    XshmemWindow_t dst, size_t dstOff,
    XshmemWindow_t src, size_t srcOff,
    size_t bytes, int pe) {
    void* remote = (void*)(dst->p2pPeerPtrs[pe] + dstOff);
    void* local  = (void*)((uintptr_t)src->localPtr + srcOff);
    // 复用 mori p2p provider 的 put 函数（参考 shmem_device_api.hpp P2P 路径）
    p2pPutThread(local, remote, bytes);
}

// ── RDMA：ibgda RDMA Write，跨机 ──
__device__ void XshmemRdmaPutThread(
    XshmemDevComm* comm,
    XshmemWindow_t dst, size_t dstOff,
    XshmemWindow_t src, size_t srcOff,
    size_t bytes, int pe, int qpId = 0);
// raddr = dst->peerPtrs[pe] + dstOff, rkey = dst->peerRkeys[pe]
// laddr = src->peerPtrs[rank] + srcOff, lkey = src->lkey
// iova=0 时 peerPtrs[pe]=0 → raddr=dstOff；iova=VA 时 peerPtrs[pe]=远端VA → raddr=VA+dstOff
// 两种模式同一份 kernel 代码
// 复用 ibgda provider（参考 shmem_device_api.hpp RDMA 路径）

__device__ void XshmemRdmaPutSignalThread(
    XshmemDevComm* comm,
    XshmemWindow_t dst, size_t dstOff,
    XshmemWindow_t src, size_t srcOff, size_t bytes,
    XshmemWindow_t sig, size_t sigOff, uint64_t sigVal, atomicType sigOp,
    int pe, int qpId = 0);

__device__ void XshmemRdmaQuietThread(XshmemDevComm* comm, int pe, int qpId = 0);

// ── SDMA：DMA 引擎 packet queue，同机 ──
__device__ void XshmemSdmaPutThread(
    XshmemWindow_t dst, size_t dstOff,
    XshmemWindow_t src, size_t srcOff,
    size_t bytes, int pe, int qpId = 0);
// dstPtr = dst->p2pPeerPtrs[pe] + dstOff（本地 flat VA，与 P2P 共用）
// srcPtr = src->localPtr + srcOff
// 复用 core::SdmaPutThread（参考 shmem_sdma_kernels.hpp）

__device__ void XshmemSdmaQuietThread(XshmemWindow_t win, int pe, int qpId = 0);

// ── Barrier ──
__device__ void XshmemBarrierAllBlock(XshmemDevComm* comm);
// 复用 ShmemInternalBarrierBlock 逻辑，传入 comm->internalSyncPtr
// 参考 shmem_device_api.hpp 中 barrier 实现
```

**字段依赖一览**：

| 路径 | 来自 XshmemDevComm | 来自 XshmemWindowDevice |
|------|-------------------|------------------------|
| P2P  | — | `p2pPeerPtrs`, `localPtr` |
| RDMA | `rdmaEndpoints` (QP handles) | `peerPtrs`, `peerRkeys`, `lkey`, `localPtr` |
| SDMA | — | `p2pPeerPtrs`, `localPtr`, `deviceHandles_d`, `signalPtrs`, `expectSignalsPtr`, `peerSignalPtrs` |

---

## 文件结构

```
include/mori/xshmem/
├── xshmem_types.hpp       ← XshmemComm, XshmemDevComm, XshmemWindowDevice, XshmemWindowHost
├── xshmem_api.hpp         ← Host API 声明
└── xshmem_device_api.hpp  ← Device inline 函数实现

src/xshmem/
├── xshmem_init.cpp        ← CommCreate/Destroy, DevCommCreate/Destroy, MemAlloc/Free, BarrierAll
└── xshmem_memory.cpp      ← WindowRegister/Deregister
```

CMakeLists.txt 新增 `mori_xshmem` target，链接 `mori_application`（Context 等）。

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

## 第一阶段 Scope（只做 host 端）

Device API 骨架（声明 + 注释）也要写，实现留后续迭代：

1. `XshmemCommCreate` / `XshmemCommDestroy`
2. `XshmemMemAlloc` / `XshmemMemFree`
3. `XshmemWindowRegister`（两个重载）/ `XshmemWindowDeregister`
4. `XshmemDevCommCreate` / `XshmemDevCommDestroy`
5. `XshmemBarrierAll`
6. 头文件骨架（types, api, device_api 声明）

---

## 验证

1. 两线程各自 `XshmemCommCreate` + `XshmemWindowRegister`，互不干扰（验证无 singleton）
2. 改写 `examples/shmem/put_thread_allgather.cpp` 使用 XSHMEM API
3. 同进程两个 comm 并发 put + barrier，验证 `internalSyncPtr` 互不污染
