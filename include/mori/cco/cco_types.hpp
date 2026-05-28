// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mori/application/transport/sdma/anvil_device.hpp"
#include "mori/hip_compat.hpp"
#include "mori/shmem/internal.hpp"

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/application/memory/va_manager.hpp"
#endif

namespace mori {
namespace cco {

// CcoDevCommRequirements carries {size, magic, version} so we can grow the
// struct ABI-compatibly. CcoDevCommCreate validates these on entry; older
// binaries pass a smaller `size` and the runtime fills the missing tail
// with INITIALIZER defaults.
static constexpr uint32_t CCO_API_MAGIC = 0x0CC0AAAA;
static constexpr uint32_t CCO_API_VERSION = 1;

// GDA backend QP allocation strategy.
enum CcoGdaConnectionType {
  CCO_GDA_CONNECTION_NONE = 0,  // no GDA QPs
  CCO_GDA_CONNECTION_FULL = 1,  // QPs to every peer (incl. intra-node) — TODO: not yet enforced
  CCO_GDA_CONNECTION_CROSSNODE = 2,  // QPs only to cross-node peers (default)
  CCO_GDA_CONNECTION_RAIL = 3,  // QPs only to same-rail cross-node peers — TODO: not yet enforced
};

// 3-int rank subset descriptor.
//   worldRank = commRank + (teamRank - team.rank) * team.stride
// Built-in teams live in cco_team.hpp: ccoTeamWorld / Lsa / CrossNode / Rail.
struct CcoTeam {
  int nRanks;
  int rank;
  int stride;
};
typedef CcoTeam CcoTeam_t;

/* ────────────────────────────────────────────────────────────────────────────
 *  GPU-side structures (device-safe, no STL)
 * ──────────────────────────────────────────────────────────────────────────── */

struct CcoWindowDevice;

static constexpr int CCO_WINDOW_TABLE_SIZE = 32;

struct CcoWindowTableNode {
  struct Entry {
    uintptr_t base;
    uintptr_t size;
    CcoWindowDevice* window;
  } entries[CCO_WINDOW_TABLE_SIZE];
  CcoWindowTableNode* next;
};

// Per-window RDMA context: one MR shared by all QPs of one window.
// peerRkeys is worldSize-sized — GDA targets any peer including FULL-mode
// intra-node loopback.
struct CcoIbgdaWin {
  uint32_t* peerRkeys;  // [worldSize]
  uint32_t lkey;
};

struct CcoWindowDevice {
  // LSA flat-VA addressing (intra-node only). winBase is the window's slot in
  // the LSA-sized flat VA reservation. Peer addressing uses LSA rank, not
  // world rank. Cross-node access goes through ibgdaWin (iova=0 + offset).
  //   winBase = flatBase + slotOffset
  //   peer_va = winBase + ((uint64_t)peerLsaRank * stride4G << 32) + offset
  //   local   = winBase + ((uint64_t)lsaRank     * stride4G << 32) + offset
  char* winBase;
  uint32_t stride4G;  // perRankSize >> 32 (perRankSize is 4GB-aligned)
  int lsaRank;        // caller's index in the LSA team

  // GDA / IBGDA (iova=0). raddr=dstOff, laddr=srcOff, rkey=peerRkeys[worldRank].
  CcoIbgdaWin ibgdaWin;

  // SDMA signal pool lives on CcoDevComm::sdma (per-DevComm, not per-window).
  // Kernels consume signals via devComm->sdma.signalBuf indexed by (lsaPeer, queueId).
};
typedef CcoWindowDevice* CcoWindow_t;

// IBGDA context: QP endpoints + signal/counter resources for one DevComm.
// One context per comm today (single NIC). Future multi-NIC may use an array.
//
// signalBuf / signalShadows / counterBuf are sub-pointers into the DevComm's
// resourceWindow (a regular CCO symmetric window). For RDMA atomic add to a
// peer's signalBuf, kernels use:
//   lkey  = devComm->resourceWindow_inlined.ibgdaWin.lkey
//   rkey  = devComm->resourceWindow_inlined.ibgdaWin.peerRkeys[peerWorldRank]
//   raddr = signal_slot_id * sizeof(uint64)   (signalBuf is at offset 0
//                                               within the resource window)
struct CcoIbgdaContext {
  shmem::ShmemRdmaEndpoint* endpoints;  // [worldSize * numQpPerPe]
  int numQpPerPe;

  // Signal: remote peers atomic +1 here after put completes.
  int signalCount;
  uint64_t* signalBuf;      // [signalCount]  — sub-ptr into resourceWindow
  uint64_t* signalShadows;  // [signalCount]  — sub-ptr into resourceWindow

  // Counter: NIC loopback writes here after source data fully transmitted.
  int counterCount;
  uint64_t* counterBuf;  // [counterCount] — sub-ptr into resourceWindow
};

// LSA barrier handle: a {byte-offset, count} pair pointing into the DevComm's
// resourceWindow. The barrier inbox/state buffer lives inside the resource
// window so it inherits LSA peer P2P addressing for free (no separate
// hipMalloc / Allgather).
//
// Layout in window (NCCL-style, lsa team):
//   uint32_t state[3*nBarriers]                          ← local epoch / arrive counters
//   uint32_t inbox[nBarriers * lsaSize]                  ← per-rank slots, peers store-add here
//
// Sizing convention:
//   bufferBytes = (3*N + N*lsaSize) * sizeof(uint32_t)
//
// Device side: barrier session computes the peer slot address as
//   winBase + peerLsa*stride4G<<32 + bufOffset + barrierIdx*lsaSize*4 + myLsa*4
struct CcoLsaBarrierHandle {
  uint32_t bufOffset;  // byte offset within resourceWindow; 0 == disabled
  int nBarriers;       // 0 == disabled
};

// GDA barrier handle: barriers via IBGDA signal pool. NCCL-style — each
// barrier consumes `team.nRanks` signal slots; peers do RDMA atomic-add to
// `signalBuf[signal0 + barrierIdx * team.nRanks + mySrcIdx]` and poll/reset.
// Team is determined by which DevComm field this handle lives in:
//   * railGdaBarrier         → same-lsaRank cross-node (rail) team, size = nNodes
//   * hybridRailGdaBarrier   → same rail team (paired with hybridLsaBarrier
//                              for two-stage world-spanning barrier)
//
// Sizing convention:
//   ginSignalCount = nBarriers * teamSize    (no buffer bytes consumed)
//
// Device side: barrier session uses signal0 + barrierIdx*teamSize + srcRailIdx
// to compute the slot id, then RDMA atomic-add via IBGDA window.
struct CcoGdaBarrierHandle {
  uint32_t signal0;  // starting slot id in ibgda.signalBuf; 0 + nBarriers==0 == disabled
  int nBarriers;     // 0 == disabled
};

// SDMA context: per-DevComm signal pool + IPC-mapped peer pointers.
// Empty when SDMA is not used by this DevComm.
struct CcoSdmaContext {
  uint32_t sdmaNumQueue;                         // 0 when SDMA disabled
  anvil::SdmaQueueDeviceHandle** deviceHandles;  // [lsaSize * sdmaNumQueue], shared from comm
  HSAuint64* signalBuf;                          // [lsaSize * sdmaNumQueue], local pool
  HSAuint64* expectSignals;                      // [lsaSize * sdmaNumQueue], local
  HSAuint64** peerSignalPtrs;                    // [lsaSize], peer signalBuf via IPC
};

struct CcoDevComm {
  // World / topology
  int rank;
  int worldSize;
  int lsaSize;      // # of ranks on my node
  int lsaRank;      // my index in lsa team [0..lsaSize)
  int myNodeStart;  // world rank of node[0]

  // GDA backend
  CcoGdaConnectionType gdaConnType;

  // Common
  void* flatBase;
  size_t perRankSize;
  CcoWindowTableNode* windowTable;  // GPU linked list of registered windows

  // Resource window: a CCO-internal symmetric window backing all per-
  // DevComm session state (today: IBGDA signal/shadows/counter pool).
  // Lives in the LSA flat VA so peers can either P2P-load/store into it
  // (intra-node) or RDMA-write to it (cross-node) using the standard
  // window addressing formula:
  //   peer_va = winBase + peerLsa * stride4G<<32 + offset
  //   raddr   = offset, rkey = peerRkeys[peer]
  //
  // Two fields, matching NCCL's `resourceWindow` + `resourceWindow_inlined`:
  //   * resourceWindow         : GPU pointer to the window struct. Used
  //                              by host-side bookkeeping (DevCommDestroy
  //                              looks up the matching CcoWindowHost via
  //                              this pointer); also lets `findWindow`
  //                              from device kernels return a stable
  //                              handle.
  //   * resourceWindow_inlined : 32-byte CcoWindowDevice copy embedded
  //                              right here in the kernel parameter
  //                              space. Kernels read winBase / stride4G /
  //                              ibgdaWin.{lkey,peerRkeys} directly out
  //                              of cmem with no GPU-memory dereference.
  CcoWindowDevice* resourceWindow;         // pointer into windowTable
  CcoWindowDevice resourceWindow_inlined;  // host-side snapshot

  // IBGDA context (QP + signal + counter); empty when gdaConnType==NONE.
  CcoIbgdaContext ibgda;
  // Standalone barriers (mirroring NCCL `ncclDevComm`):
  //   * lsaBarrier            — intra-node, driven by reqs.lsaBarrierCount
  //   * railGdaBarrier        — same-rail cross-node, driven by reqs.railGdaBarrierCount
  // Hybrid barrier pair (two-stage LSA+Rail world barrier):
  //   * hybridLsaBarrier      — intra-node half
  //   * hybridRailGdaBarrier  — inter-node half (same rail team)
  // All four are driven from CcoDevCommRequirements counts; any field with
  // nBarriers==0 is disabled.
  CcoLsaBarrierHandle lsaBarrier;
  CcoGdaBarrierHandle railGdaBarrier;
  CcoLsaBarrierHandle hybridLsaBarrier;
  CcoGdaBarrierHandle hybridRailGdaBarrier;
  // SDMA context (signal pool); empty when SDMA not materialized.
  CcoSdmaContext sdma;
};
typedef CcoDevComm* CcoDevComm_t;

/* ────────────────────────────────────────────────────────────────────────────
 *  DevComm requirements
 *
 *    CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
 *    reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
 *    reqs.gdaSignalCount = numCTAs;
 *    CcoDevCommCreate(comm, &reqs, &devComm);
 * ──────────────────────────────────────────────────────────────────────────── */

// Per-backend resource buffer reservation node. Currently declared as a Phase 2
// scaffold; will be consumed when CcoLsa / CcoSdma / CcoLsaBarrierSession land.
struct CcoDevResourceRequirements {
  CcoDevResourceRequirements* next;
  size_t bufferSize;
  size_t bufferAlign;
  uint32_t* outBufferHandle;  // populated on success: offset in comm buf
  int gdaSignalCount;
  int gdaCounterCount;
  uint32_t* outGdaSignalStart;
  uint32_t* outGdaCounterStart;
};

struct CcoDevCommRequirements {
  // Forward-compat triplet (set by INITIALIZER, do not touch).
  size_t size;
  uint32_t magic;
  uint32_t version;

  // Resource buffer linked list (Phase 2 scaffold).
  CcoDevResourceRequirements* resourceRequirementsList;

  // GDA (RDMA).
  CcoGdaConnectionType gdaConnectionType;
  int gdaContextCount;  // # of independent QP sets per peer
  int gdaSignalCount;
  int gdaCounterCount;
  int gdaQueueDepth;    // 0 = provider default
  int gdaTrafficClass;  // -1 = MORI_RDMA_TC env

  // LSA (intra-node P2P).
  int lsaBarrierCount;

  // GDA-Rail (same-lsaRank cross-node) standalone barrier.
  int railGdaBarrierCount;

  // SDMA.
  int sdmaQueueCount;  // 0 = anvil default

  // Hybrid barrier (LSA + GDA-Rail two-stage). Drives BOTH
  // hybridLsaBarrier and hybridRailGdaBarrier with the same N.
  int barrierCount;
};

#define CCO_DEV_COMM_REQUIREMENTS_INITIALIZER                                   \
  {                                                                             \
      sizeof(::mori::cco::CcoDevCommRequirements),                              \
      ::mori::cco::CCO_API_MAGIC,                                               \
      ::mori::cco::CCO_API_VERSION,                                             \
      nullptr,                                   /* resourceRequirementsList */ \
      ::mori::cco::CCO_GDA_CONNECTION_CROSSNODE, /* gdaConnectionType */        \
      4,                                         /* gdaContextCount    */       \
      16,                                        /* gdaSignalCount     */       \
      16,                                        /* gdaCounterCount    */       \
      0,                                         /* gdaQueueDepth      */       \
      -1,                                        /* gdaTrafficClass    */       \
      0,                                         /* lsaBarrierCount    */       \
      0,                                         /* railGdaBarrierCount*/       \
      0,                                         /* sdmaQueueCount     */       \
      0,                                         /* barrierCount       */       \
  }

/* ────────────────────────────────────────────────────────────────────────────
 *  Host-only structures
 * ──────────────────────────────────────────────────────────────────────────── */

#if !defined(__HIPCC__) && !defined(__CUDACC__)

struct CcoWindowHost {
  void* localPtr;
  size_t size;
  CcoWindowDevice* devPtr;
  uint32_t* peerRkeys_gpu;
  // Peer's dma-buf imported handles. WindowRegister inserts one per P2P-
  // mapped peer; WindowDeregister hipMemUnmap's the peer VA then
  // hipMemRelease's the handle to drop the cross-process refcount.
  std::vector<hipMemGenericAllocationHandle_t> peerImportedHandles;
};

struct CcoComm {
  int rank{0};
  int worldSize{0};
  application::BootstrapNetwork* bootNet{nullptr};
  application::Context* ctx{nullptr};

  // rank 0's pid, gathered via Allgather. Disambiguates LocalBootstrap socket
  // paths across independent comm groups in the same process tree.
  int64_t groupId{0};

  // Local HIP device this comm is bound to. Cached at CommCreate so we
  // don't call hipGetDevice() on the hot path (per-MemAlloc / per-Window).
  // Callers MUST keep the calling thread bound to this device for the
  // lifetime of any CCO API call on this comm.
  int cudaDev{-1};

  // Intra-node topology (populated at CommCreate).
  int lsaSize{1};
  int lsaRank{0};
  int myNodeStart{0};

  // VMM flat address space (sized lsaSize * perRankSize).
  void* flatBase{nullptr};
  size_t perRankSize{0};
  size_t vmmGranularity{0};

  // Per-rank slot allocator within [0, perRankSize). Reuses the application
  // HeapVAManager (first-fit + O(log n) coalescing, already used by shmem).
  // baseAddr=0 so Allocate() returns the offset directly.
  std::unique_ptr<application::HeapVAManager> vaManager;

  // Default # of QPs per peer (from Context). Per-DevComm may override via reqs.
  int defaultNumQpPerPe{4};
  bool iovaZeroMode{true};

  // SDMA queue handles (per-comm, sized lsaSize * sdmaNumQueue, indexed by lsaRank).
  anvil::SdmaQueueDeviceHandle** sdmaDevHandles{nullptr};
  int sdmaNumQueue{0};

  struct AllocMeta {
    hipMemGenericAllocationHandle_t physHandle;
    int shareFd{-1};
    size_t slotOffset{0};
    size_t size{0};
  };
  std::unordered_map<void*, AllocMeta> allocTable;

  // Protects allocTable, windows, windowTableEntries against concurrent
  // MemAlloc / MemFree / WindowRegister / WindowDeregister from multiple
  // threads sharing the same CcoComm. The vaManager has its own mutex.
  mutable std::mutex allocMutex;

  std::vector<CcoWindowHost*> windows;

  struct WindowTableEntry {
    uintptr_t base;
    uintptr_t size;
    CcoWindowDevice* devPtr;
  };
  std::vector<WindowTableEntry> windowTableEntries;
};

#endif  // !defined(__HIPCC__) && !defined(__CUDACC__)

}  // namespace cco
}  // namespace mori