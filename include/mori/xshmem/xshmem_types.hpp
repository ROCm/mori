// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mori/application/transport/sdma/anvil_device.hpp"
#include "mori/hip_compat.hpp"
#include "mori/shmem/internal.hpp"

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <unordered_map>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"
#endif

namespace mori {
namespace xshmem {

/* ────────────────────────────────────────────────────────────────────────────
 *  GPU-side structures (device-safe, no STL)
 * ──────────────────────────────────────────────────────────────────────────── */

struct XshmemWindowDevice;

static constexpr int XSHMEM_WINDOW_TABLE_SIZE = 32;

struct XshmemWindowTableNode {
  struct Entry {
    uintptr_t base;       // localPtr as uintptr_t
    uintptr_t size;
    XshmemWindowDevice* window;
  } entries[XSHMEM_WINDOW_TABLE_SIZE];
  XshmemWindowTableNode* next;
};

// IBGDA context: QP endpoints + signal/counter resources bundled together.
// Analogous to NCCL's ncclGinGdakiGPUContext.
// One context per comm (single NIC). Future multi-NIC: array of contexts.
struct XshmemIbgdaContext {
  // QP endpoints: indexed by [pe * numQpPerPe + qpId]
  shmem::ShmemRdmaEndpoint* endpoints;  // GPU buf, length = worldSize * numQpPerPe
  int numQpPerPe;

  // Signal: remote peers RDMA-atomic to our signalBuf after put completes
  int signalCount;
  uint64_t* signalBuf;         // GPU buf [signalCount], remote write target
  uint64_t* signalShadows;     // GPU buf [signalCount], local sent-signal tracking
  uint32_t* peerSignalRkeys;   // GPU buf [worldSize], each peer's signalBuf rkey
  uint32_t signalLkey;         // local signalBuf MR lkey

  // Counter: NIC loopback writes here after source data fully transmitted
  int counterCount;
  uint64_t* counterBuf;        // GPU buf [counterCount]
};

struct XshmemDevComm {
  int rank;
  int worldSize;
  uint64_t* internalSyncPtr;                // GPU buf, 128 × uint64_t
  void* flatBase;
  size_t perRankSize;
  XshmemWindowTableNode* windowTable;       // GPU, linked list of registered windows

  // IBGDA context (QP + signal + counter)
  XshmemIbgdaContext ibgda;
};
typedef XshmemDevComm* XshmemDevComm_t;

// Per-window RDMA context (analogous to NCCL's ncclGinWindow_t ginWins[])
// One MR per window, shared by all QPs. peerRkeys indexed by [pe].
struct XshmemIbgdaWin {
  uint32_t* peerRkeys;     // [worldSize], Allgather-exchanged
  uint32_t lkey;            // local MR key for this window
};

struct XshmemWindowDevice {
  // ── flat VA addressing (P2P / SDMA / general) ──
  // Intentionally duplicated from DevComm so window is self-contained.
  // winBase = flatBase + slotOffset (pre-computed, like NCCL's lsaFlatBase + bigOffset)
  char* winBase;           // = flatBase + slotOffset (rank 0's window start in flat VA)
  uint32_t stride4G;       // = perRankSize >> 32 (4GB-aligned stride, like NCCL)
  int rank;                // = comm->rank
  int worldSize;           // = comm->worldSize

  // P2P/SDMA: winBase + ((uint64_t)pe * stride4G << 32) + offset
  // local:    winBase + ((uint64_t)rank * stride4G << 32) + offset

  // ── RDMA / IBGDA (iova=0, offset-based) ──
  // QP endpoints: DevComm->ibgda.endpoints[pe * ibgda.numQpPerPe + qpId]
  // MR keys are per-window (same MR for all QPs):
  XshmemIbgdaWin ibgdaWin;
  // raddr = dstOff (iova=0)
  // laddr = srcOff (iova=0)
  // rkey  = ibgdaWin.peerRkeys[pe]
  // lkey  = ibgdaWin.lkey

  // ── SDMA signals ──
  anvil::SdmaQueueDeviceHandle** deviceHandles_d;  // per-comm shared
  HSAuint64* signalPtrs;                            // [worldSize * sdmaNumQueue]
  HSAuint64* expectSignalsPtr;                      // [worldSize * sdmaNumQueue]
  HSAuint64** peerSignalPtrs;                       // [worldSize]
  uint32_t sdmaNumQueue;
};
typedef XshmemWindowDevice* XshmemWindow_t;

/* ────────────────────────────────────────────────────────────────────────────
 *  Host-only structures
 * ──────────────────────────────────────────────────────────────────────────── */

#if !defined(__HIPCC__) && !defined(__CUDACC__)

struct XshmemWindowHost {
  void* localPtr;
  size_t size;
  // SDMA signals (for Deregister cleanup)
  HSAuint64* signalPtrs;
  HSAuint64* expectSignalsPtr;
  HSAuint64** peerSignalPtrs;
  // GPU device struct (for Deregister cleanup)
  XshmemWindowDevice* devPtr;
  // GPU arrays (for Deregister cleanup)
  uint32_t* peerRkeys_gpu;
  HSAuint64** peerSignalPtrs_gpu;
};

struct XshmemComm {
  int rank{0};
  int worldSize{0};
  application::BootstrapNetwork* bootNet{nullptr};
  application::Context* ctx{nullptr};

  // Group ID: rank 0's pid, shared via Allgather. Used to derive unique
  // LocalBootstrapNetwork socket paths across independent comm groups.
  int64_t groupId{0};

  // VMM flat address space
  void* flatBase{nullptr};
  size_t perRankSize{0};
  size_t nextOffset{0};
  size_t vmmGranularity{0};

  // RDMA
  std::vector<shmem::ShmemRdmaEndpoint> rdmaEndpoints;
  int numQpPerPe{4};
  bool iovaZeroMode{true};

  // Signal / Counter requirements (set before DevCommCreate)
  int signalCount{16};         // default: 16 signal slots (one per CTA)
  int counterCount{16};        // default: 16 counter slots

  // SDMA (per-comm, shared across all windows)
  anvil::SdmaQueueDeviceHandle** sdmaDevHandles{nullptr};
  int sdmaNumQueue{0};

  // Barrier
  uint64_t* internalSyncGpuPtr{nullptr};

  // Allocation metadata (populated by MemAlloc, queried by WindowRegister)
  struct AllocMeta {
    hipMemGenericAllocationHandle_t physHandle;
    int shareFd{-1};
    size_t slotOffset{0};
    size_t size{0};
  };
  std::unordered_map<void*, AllocMeta> allocTable;

  std::vector<XshmemWindowHost*> windows;

  // Window table: host shadow of GPU-side linked list (for DevCommCreate to build)
  struct WindowTableEntry {
    uintptr_t base;
    uintptr_t size;
    XshmemWindowDevice* devPtr;
  };
  std::vector<WindowTableEntry> windowTableEntries;
};

#endif  // !defined(__HIPCC__) && !defined(__CUDACC__)

}  // namespace xshmem
}  // namespace mori
