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

// CcoDevCommRequirements carries {size, magic, version} so we can grow the
// struct ABI-compatibly. CcoDevCommCreate validates these on entry; older
// binaries pass a smaller `size` and the runtime fills the missing tail
// with INITIALIZER defaults.
static constexpr uint32_t CCO_API_MAGIC   = 0x0CC0AAAA;
static constexpr uint32_t CCO_API_VERSION = 1;

// GDA backend QP allocation strategy.
enum CcoGdaConnectionType {
  CCO_GDA_CONNECTION_NONE      = 0,   // no GDA QPs
  CCO_GDA_CONNECTION_FULL      = 1,   // QPs to every peer (incl. intra-node) — TODO: not yet enforced
  CCO_GDA_CONNECTION_CROSSNODE = 2,   // QPs only to cross-node peers (default)
  CCO_GDA_CONNECTION_RAIL      = 3,   // QPs only to same-rail cross-node peers — TODO: not yet enforced
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

// IBGDA context: QP endpoints + signal/counter resources for one DevComm.
// One context per comm today (single NIC). Future multi-NIC may use an array.
struct CcoIbgdaContext {
  shmem::ShmemRdmaEndpoint* endpoints;  // [worldSize * numQpPerPe]
  int numQpPerPe;

  // Signal: remote peers atomic +1 here after put completes.
  int signalCount;
  uint64_t* signalBuf;         // [signalCount]
  uint64_t* signalShadows;     // [signalCount], local sent-signal tracking
  uint32_t* peerSignalRkeys;   // [worldSize], each peer's signalBuf rkey
  uint32_t signalLkey;         // signalBuf MR lkey

  // Counter: NIC loopback writes here after source data fully transmitted.
  int counterCount;
  uint64_t* counterBuf;        // [counterCount]
};

struct CcoDevComm {
  // World / topology
  int rank;
  int worldSize;
  int lsaSize;        // # of ranks on my node
  int lsaRank;        // my index in lsa team [0..lsaSize)
  int myNodeStart;    // world rank of node[0]

  // GDA backend
  CcoGdaConnectionType gdaConnType;

  // Common
  uint64_t* internalSyncPtr;             // [128] for device barriers
  void* flatBase;
  size_t perRankSize;
  CcoWindowTableNode* windowTable;       // GPU linked list of registered windows

  // IBGDA context (QP + signal + counter); empty when gdaConnType==NONE.
  CcoIbgdaContext ibgda;
};
typedef CcoDevComm* CcoDevComm_t;

// Per-window RDMA context: one MR shared by all QPs of one window.
// peerRkeys is worldSize-sized — GDA targets any peer including FULL-mode
// intra-node loopback.
struct CcoIbgdaWin {
  uint32_t* peerRkeys;     // [worldSize]
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
  uint32_t stride4G;       // perRankSize >> 32 (perRankSize is 4GB-aligned)
  int lsaRank;             // caller's index in the LSA team

  // GDA / IBGDA (iova=0). raddr=dstOff, laddr=srcOff, rkey=peerRkeys[worldRank].
  CcoIbgdaWin ibgdaWin;

  // SDMA signals (intra-node only, indexed by LSA rank).
  anvil::SdmaQueueDeviceHandle** deviceHandles_d;   // per-comm shared
  HSAuint64* signalPtrs;                             // [lsaSize * sdmaNumQueue]
  HSAuint64* expectSignalsPtr;                       // [lsaSize * sdmaNumQueue]
  HSAuint64** peerSignalPtrs;                        // [lsaSize]
  uint32_t sdmaNumQueue;
};
typedef CcoWindowDevice* CcoWindow_t;

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
  size_t   bufferSize;
  size_t   bufferAlign;
  uint32_t* outBufferHandle;     // populated on success: offset in comm buf
  int      gdaSignalCount;
  int      gdaCounterCount;
  uint32_t* outGdaSignalStart;
  uint32_t* outGdaCounterStart;
};

struct CcoDevCommRequirements {
  // Forward-compat triplet (set by INITIALIZER, do not touch).
  size_t   size;
  uint32_t magic;
  uint32_t version;

  // Resource buffer linked list (Phase 2 scaffold).
  CcoDevResourceRequirements* resourceRequirementsList;

  // GDA (RDMA).
  CcoGdaConnectionType gdaConnectionType;
  int                  gdaContextCount;     // # of independent QP sets per peer
  int                  gdaSignalCount;
  int                  gdaCounterCount;
  int                  gdaQueueDepth;       // 0 = provider default
  int                  gdaTrafficClass;     // -1 = MORI_RDMA_TC env

  // LSA (intra-node P2P).
  int lsaBarrierCount;

  // SDMA.
  int sdmaQueueCount;                       // 0 = anvil default

  // Hybrid barrier (LSA + GDA-Rail two-stage).
  int barrierCount;
};

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
  HSAuint64* signalPtrs;
  HSAuint64* expectSignalsPtr;
  HSAuint64** peerSignalPtrs;
  CcoWindowDevice* devPtr;
  uint32_t* peerRkeys_gpu;
  HSAuint64** peerSignalPtrs_gpu;
};

struct CcoComm {
  int rank{0};
  int worldSize{0};
  application::BootstrapNetwork* bootNet{nullptr};
  application::Context* ctx{nullptr};

  // rank 0's pid, gathered via Allgather. Disambiguates LocalBootstrap socket
  // paths across independent comm groups in the same process tree.
  int64_t groupId{0};

  // Intra-node topology (populated at CommCreate).
  int lsaSize{1};
  int lsaRank{0};
  int myNodeStart{0};

  // VMM flat address space (sized lsaSize * perRankSize).
  void* flatBase{nullptr};
  size_t perRankSize{0};
  size_t vmmGranularity{0};

  // Per-rank slot allocator within [0, perRankSize). Tracks allocated
  // intervals in a sorted "cuts" array; segments alternate empty<->full.
  // Lazy first-fit on alloc; coalescing free.
  struct AllocSpace {
    std::vector<int64_t> cuts;
  };
  AllocSpace allocSpace;

  // Default # of QPs per peer (from Context). Per-DevComm may override via reqs.
  int defaultNumQpPerPe{4};
  bool iovaZeroMode{true};

  // SDMA queue handles (per-comm, sized lsaSize * sdmaNumQueue, indexed by lsaRank).
  anvil::SdmaQueueDeviceHandle** sdmaDevHandles{nullptr};
  int sdmaNumQueue{0};

  uint64_t* internalSyncGpuPtr{nullptr};

  struct AllocMeta {
    hipMemGenericAllocationHandle_t physHandle;
    int shareFd{-1};
    size_t slotOffset{0};
    size_t size{0};
  };
  std::unordered_map<void*, AllocMeta> allocTable;

  // Protects allocSpace, allocTable, windows, windowTableEntries against
  // concurrent MemAlloc / MemFree / WindowRegister / WindowDeregister from
  // multiple threads sharing the same CcoComm.
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
