// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mori/application/transport/sdma/anvil_device.hpp"
#include "mori/hip_compat.hpp"
#include "mori/shmem/internal.hpp"

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <mutex>
#include <unordered_map>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"
#endif

namespace mori {
namespace cco {

/* ────────────────────────────────────────────────────────────────────────────
 *  API forward-compat constants
 * ──────────────────────────────────────────────────────────────────────────── */

// Magic + version are baked into CcoDevCommRequirements via the
// CCO_DEV_COMM_REQUIREMENTS_INITIALIZER macro. CcoDevCommCreate validates them
// on entry so we can safely add new fields to the struct in the future without
// breaking existing user binaries (older binaries pass a smaller struct; the
// runtime sees the smaller `size` and uses defaults for the missing tail).
static constexpr uint32_t CCO_API_MAGIC   = 0x0CC0AAAA;
static constexpr uint32_t CCO_API_VERSION = 1;

/* ────────────────────────────────────────────────────────────────────────────
 *  GDA backend connection topology
 * ──────────────────────────────────────────────────────────────────────────── */

enum CcoGdaConnectionType {
  CCO_GDA_CONNECTION_NONE      = 0,   // No GDA QPs at all.
  CCO_GDA_CONNECTION_FULL      = 1,   // QPs to every peer (incl. intra-node).
                                       // TODO: not yet enforced — currently
                                       // falls back to CROSSNODE because
                                       // Context skips P2P-reachable peers.
  CCO_GDA_CONNECTION_CROSSNODE = 2,   // QPs only to cross-node peers (default).
                                       // CCO addition not present in NCCL.
  CCO_GDA_CONNECTION_RAIL      = 3,   // QPs only to same-rail cross-node peers.
                                       // TODO: not yet enforced — falls back
                                       // to CROSSNODE.
};

/* ────────────────────────────────────────────────────────────────────────────
 *  Team — 3-int rank subset descriptor, mirrors ncclTeam
 *
 *  Conversion to world rank:
 *    worldRank = commRank + (teamRank - team.rank) * team.stride
 *
 *  Built-in teams (declared in cco_team.hpp as __device__ inline funcs):
 *    CcoTeamWorld    : all ranks
 *    CcoTeamLsa      : ranks on the same node
 *    CcoTeamCrossNode: cross-node ranks (skipping my node)
 *    CcoTeamRail     : same NIC-rail-index cross-node ranks
 * ──────────────────────────────────────────────────────────────────────────── */

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
    uintptr_t base;       // localPtr as uintptr_t
    uintptr_t size;
    CcoWindowDevice* window;
  } entries[CCO_WINDOW_TABLE_SIZE];
  CcoWindowTableNode* next;
};

// IBGDA context: QP endpoints + signal/counter resources bundled together.
// Analogous to NCCL's ncclGinGdakiGPUContext.
// One context per comm (single NIC). Future multi-NIC: array of contexts.
struct CcoIbgdaContext {
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

struct CcoDevComm {
  // ── World / topology ──
  int rank;
  int worldSize;
  int lsaSize;        // # of ranks on my node (intra-node group size)
  int lsaRank;        // my index in lsa team [0..lsaSize)
  int myNodeStart;    // = (rank / lsaSize) * lsaSize, world rank of node[0]

  // ── GDA backend ──
  CcoGdaConnectionType gdaConnType;

  // ── Common ──
  uint64_t* internalSyncPtr;                // GPU buf, 128 × uint64_t
  void* flatBase;
  size_t perRankSize;
  CcoWindowTableNode* windowTable;       // GPU, linked list of registered windows

  // IBGDA context (QP + signal + counter); empty when gdaConnType==NONE
  CcoIbgdaContext ibgda;
};
typedef CcoDevComm* CcoDevComm_t;

// Per-window RDMA context (analogous to NCCL's ncclGinWindow_t ginWins[])
// One MR per window, shared by all QPs. peerRkeys indexed by [pe].
struct CcoIbgdaWin {
  uint32_t* peerRkeys;     // [worldSize], Allgather-exchanged
  uint32_t lkey;            // local MR key for this window
};

struct CcoWindowDevice {
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
  CcoIbgdaWin ibgdaWin;
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
typedef CcoWindowDevice* CcoWindow_t;

/* ────────────────────────────────────────────────────────────────────────────
 *  DevComm requirements (host-input, drives DevCommCreate resource allocation)
 *
 *  Always initialize via CCO_DEV_COMM_REQUIREMENTS_INITIALIZER:
 *
 *    CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
 *    reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
 *    reqs.gdaSignalCount = numCTAs;
 *    CcoDevCommCreate(comm, &reqs, &devComm);
 *
 *  Forward-compat: CcoDevCommCreate checks {size, magic, version}. Adding
 *  new tail fields is ABI-safe — old binaries pass a smaller `size` and the
 *  runtime fills the missing tail with INITIALIZER defaults.
 * ──────────────────────────────────────────────────────────────────────────── */

// Linked-list node for per-backend resource buffer reservations
// (mirrors ncclDevResourceRequirements). Phase 1 scope: declared but unused;
// will be wired up when CcoLsa / CcoSdma / CcoLsaBarrierSession land.
struct CcoDevResourceRequirements {
  CcoDevResourceRequirements* next;
  size_t   bufferSize;
  size_t   bufferAlign;
  uint32_t* outBufferHandle;     // populated on success: offset in comm buf
  int      gdaSignalCount;
  int      gdaCounterCount;
  uint32_t* outGdaSignalStart;   // populated: signal id range start
  uint32_t* outGdaCounterStart;  // populated: counter id range start
};

struct CcoDevCommRequirements {
  // ── forward-compat triplet (do not touch — set by INITIALIZER) ──
  size_t   size;
  uint32_t magic;
  uint32_t version;

  // ── resource buffer linked list (Phase 2 — currently informational only) ──
  CcoDevResourceRequirements* resourceRequirementsList;

  // ── GDA (RDMA) ──
  CcoGdaConnectionType gdaConnectionType;   // default CROSSNODE
  int                  gdaContextCount;     // # of independent QP sets (numQpPerPe)
  int                  gdaSignalCount;      // remote-write signal slots
  int                  gdaCounterCount;     // local NIC-loopback counter slots
  int                  gdaQueueDepth;       // 0 = provider default
  int                  gdaTrafficClass;     // -1 = MORI_RDMA_TC env

  // ── LSA (intra-node P2P) ──
  int lsaBarrierCount;

  // ── SDMA ──
  int sdmaQueueCount;                       // 0 = use anvil default

  // ── Hybrid barrier (LSA + GDA-Rail two-stage) ──
  int barrierCount;
};

// Default values mirror the previously hardcoded constants in cco_init.cpp.
// gdaConnectionType defaults to CROSSNODE because that matches what mori's
// Context naturally produces (P2P/SDMA-reachable peers get no NIC QP).
#define CCO_DEV_COMM_REQUIREMENTS_INITIALIZER {                                \
    sizeof(::mori::cco::CcoDevCommRequirements),                               \
    ::mori::cco::CCO_API_MAGIC,                                                \
    ::mori::cco::CCO_API_VERSION,                                              \
    nullptr,                                  /* resourceRequirementsList */   \
    ::mori::cco::CCO_GDA_CONNECTION_CROSSNODE,/* gdaConnectionType */          \
    4,                                        /* gdaContextCount    */         \
    16,                                       /* gdaSignalCount     */         \
    16,                                       /* gdaCounterCount    */         \
    0,                                        /* gdaQueueDepth      */         \
    -1,                                       /* gdaTrafficClass    */         \
    0,                                        /* lsaBarrierCount    */         \
    0,                                        /* sdmaQueueCount     */         \
    0,                                        /* barrierCount       */         \
}

/* ────────────────────────────────────────────────────────────────────────────
 *  Host-only structures
 * ──────────────────────────────────────────────────────────────────────────── */

#if !defined(__HIPCC__) && !defined(__CUDACC__)

struct CcoWindowHost {
  void* localPtr;
  size_t size;
  // SDMA signals (for Deregister cleanup)
  HSAuint64* signalPtrs;
  HSAuint64* expectSignalsPtr;
  HSAuint64** peerSignalPtrs;
  // GPU device struct (for Deregister cleanup)
  CcoWindowDevice* devPtr;
  // GPU arrays (for Deregister cleanup)
  uint32_t* peerRkeys_gpu;
  HSAuint64** peerSignalPtrs_gpu;
};

struct CcoComm {
  int rank{0};
  int worldSize{0};
  application::BootstrapNetwork* bootNet{nullptr};
  application::Context* ctx{nullptr};

  // Group ID: rank 0's pid, shared via Allgather. Used to derive unique
  // LocalBootstrapNetwork socket paths across independent comm groups.
  int64_t groupId{0};

  // ── Topology (intra-node detection, populated at CommCreate) ──
  // Assumes all nodes have the same lsaSize (typical: 8 GPUs/node).
  // Probed via hipDeviceCanAccessPeer + Allgather.
  int lsaSize{1};
  int lsaRank{0};
  int myNodeStart{0};

  // VMM flat address space
  void* flatBase{nullptr};
  size_t perRankSize{0};
  size_t nextOffset{0};
  size_t vmmGranularity{0};

  // Default # of QPs per peer, read from Context. Per-DevComm may override
  // via CcoDevCommRequirements::gdaContextCount.
  int defaultNumQpPerPe{4};
  bool iovaZeroMode{true};

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

  // Protects nextOffset (slot bump pointer) + allocTable + windows + windowTableEntries
  // against concurrent MemAlloc/MemFree/WindowRegister/WindowDeregister from
  // multiple threads sharing the same CcoComm. (Per-thread CcoComm in
  // test_cco_host doesn't need this — each thread has its own comm — but we
  // make CcoComm itself thread-safe so a future multi-thread-per-comm
  // workload doesn't silently corrupt internal data structures.)
  mutable std::mutex allocMutex;

  std::vector<CcoWindowHost*> windows;

  // Window table: host shadow of GPU-side linked list (for DevCommCreate to build)
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
