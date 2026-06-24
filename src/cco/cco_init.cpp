// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "mori/application/application.hpp"  // Context, BootstrapNetwork
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/memory/va_manager.hpp"  // HeapVAManager
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/cco/cco.hpp"  // public, self-contained (opaque ccoComm fwd-decl)
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace cco {

// ccoProviderType is cco's self-contained copy of core::ProviderType; the cast
// below relies on a 1:1 mapping, so guard it (this TU sees both enums).
static_assert(static_cast<int>(CCO_PROVIDER_UNKNOWN) ==
                  static_cast<int>(core::ProviderType::Unknown),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_MLX5) == static_cast<int>(core::ProviderType::MLX5),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_BNXT) == static_cast<int>(core::ProviderType::BNXT),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_PSD) == static_cast<int>(core::ProviderType::PSD),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_IBVERBS) ==
                  static_cast<int>(core::ProviderType::IBVERBS),
              "ccoProviderType drifted from core::ProviderType");

// Out-of-line dtor for the unique_ptr<HeapVAManager> member: ccoComm is defined
// in cco.hpp with HeapVAManager only forward-declared, so its destruction must
// be emitted here where HeapVAManager (va_manager.hpp) is complete.
ccoComm::~ccoComm() = default;

static size_t AlignUp(size_t x, size_t align) { return (x + align - 1) & ~(align - 1); }

// Local slot base = the VA where this rank's slice of the flat VA starts.
// Used as HeapVAManager's baseAddr so Allocate() returns dereferenceable
// localVa directly. Guaranteed non-zero because flatBase comes from
// hipMemAddressReserve.
static uintptr_t LocalSlotBase(const ccoComm* comm) {
  return reinterpret_cast<uintptr_t>(comm->flatBase) +
         static_cast<uintptr_t>(comm->lsaRank) * comm->perRankSize;
}

/* ========================================================================== */
/*                              ccoCommCreate                              */
/* ========================================================================== */

int ccoGetUniqueId(ccoUniqueId* uniqueId) {
  if (!uniqueId) return -1;
  static_assert(sizeof(application::UniqueId) <= sizeof(ccoUniqueId),
                "ccoUniqueId must be large enough to hold application::UniqueId");
  // Encode rank 0's socket rendezvous endpoint into the id; non-root ranks
  // connect here during ccoCommCreate, so the address+port must be concrete.
  // Pick a free port by random probe-bind (zero-config, no fixed port to
  // collide) — same scheme as shmem's ShmemGetUniqueId. Interface from
  // MORI_SOCKET_IFNAME, else the first non-loopback NIC. Caller broadcasts the
  // POD id to every rank out-of-band.
  const char* ifname = std::getenv("MORI_SOCKET_IFNAME");
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> portDis(10000, 60000);
  constexpr int kMaxPortRetries = 20;

  try {
    for (int attempt = 0; attempt < kMaxPortRetries; attempt++) {
      int port = portDis(gen);
      int probeFd = socket(AF_INET, SOCK_STREAM, 0);
      if (probeFd < 0) continue;
      int opt = 1;
      setsockopt(probeFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      struct sockaddr_in probeAddr{};
      probeAddr.sin_family = AF_INET;
      probeAddr.sin_port = htons(static_cast<uint16_t>(port));
      probeAddr.sin_addr.s_addr = htonl(INADDR_ANY);
      if (bind(probeFd, reinterpret_cast<struct sockaddr*>(&probeAddr), sizeof(probeAddr)) == 0) {
        close(probeFd);
        application::UniqueId appUid =
            ifname
                ? application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface(ifname, port)
                : application::SocketBootstrapNetwork::GenerateUniqueIdWithLocalAddr(port);
        std::memset(uniqueId, 0, sizeof(*uniqueId));
        std::memcpy(uniqueId, &appUid, sizeof(appUid));
        return 0;
      }
      close(probeFd);
    }
  } catch (const std::exception& e) {
    MORI_SHMEM_ERROR("ccoGetUniqueId failed: {} (set MORI_SOCKET_IFNAME=<iface>)", e.what());
    return -1;
  }
  MORI_SHMEM_ERROR("ccoGetUniqueId: no free port after {} attempts", kMaxPortRetries);
  return -1;
}

// Internal bootstrap helper: caller-provided transport (ownership transferred to
// the comm; Finalize()d + deleted in ccoCommDestroy). Not part of the public API
// — the ccoUniqueId overload below builds the built-in socket bootstrap and
// delegates here.
static int ccoCommCreateImpl(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                             ccoComm** outComm);

// Self-contained overload: build cco's built-in socket bootstrap from the id and
// delegate to the internal helper, which takes ownership (the socket bootstrap is
// Finalize()d + deleted in ccoCommDestroy).
int ccoCommCreate(const ccoUniqueId& uniqueId, int nRanks, int rank, size_t perRankVmmSize,
                  ccoComm** outComm) {
  if (!outComm || nRanks <= 0 || rank < 0 || rank >= nRanks) return -1;
  application::UniqueId appUid;
  std::memcpy(&appUid, &uniqueId, sizeof(appUid));
  auto* boot = new application::SocketBootstrapNetwork(appUid, rank, nRanks);
  return ccoCommCreateImpl(boot, perRankVmmSize, outComm);
}

static int ccoCommCreateImpl(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                             ccoComm** outComm) {
  auto* comm = new ccoComm();
  *outComm = comm;

  // Step 1: bootstrap
  comm->bootNet = bootNet;
  comm->bootNet->Initialize();
  comm->rank = comm->bootNet->GetLocalRank();
  comm->worldSize = comm->bootNet->GetWorldSize();

  // Derive a shared group ID (rank 0's pid) for unique LocalBootstrap socket paths
  int64_t myPid = static_cast<int64_t>(getpid());
  std::vector<int64_t> allPids(comm->worldSize);
  comm->bootNet->Allgather(&myPid, allPids.data(), sizeof(int64_t));
  comm->groupId = allPids[0];

  MORI_SHMEM_TRACE("ccoCommCreate: rank={} worldSize={} groupId={}", comm->rank, comm->worldSize,
                   comm->groupId);

  // Step 2: context (RDMA endpoints + transport-type negotiation).
  comm->ctx = new application::Context(*comm->bootNet);
  comm->defaultNumQpPerPe = comm->ctx->GetNumQpPerPe();

  // Step 2.5: detect intra-node topology (LSA = Local Symmetric Access).
  // Use Context's capability discovery (PeerCapabilities.sameHost) rather
  // than the chosen transport — LSA membership is a hardware fact and must
  // not flip even if policy routes intra-node traffic via RDMA.
  //
  // HARD CONTRACT — violations are fatal:
  //   (a) node-major contiguous ranks (same-host peers form a single block)
  //   (b) every rank observes the same lsaSize
  // Both are required by the flat-VA formula `lsaFlatBase + lsaRank * stride`.
  {
    int lsaCount = 0;
    int firstSameNode = comm->rank;
    int lastSameNode = comm->rank;
    for (int pe = 0; pe < comm->worldSize; pe++) {
      const auto& cap = comm->ctx->GetPeerCapabilities(pe);
      const bool sameNode = (pe == comm->rank) || cap.sameHost;
      if (sameNode) {
        if (pe < firstSameNode) firstSameNode = pe;
        if (pe > lastSameNode) lastSameNode = pe;
        lsaCount++;
      }
    }

    if (lastSameNode - firstSameNode + 1 != lsaCount) {
      MORI_SHMEM_ERROR(
          "ccoCommCreate: non-contiguous lsa membership "
          "(rank {}: first={} last={} count={}). CCO requires "
          "node-major contiguous rank layout. Reorder ranks in your "
          "launch (mpirun -host A:N,B:N or equivalent).",
          comm->rank, firstSameNode, lastSameNode, lsaCount);
      delete comm->ctx;
      comm->bootNet->Finalize();
      delete comm;
      *outComm = nullptr;
      return -1;
    }

    std::vector<int> allLsaSizes(comm->worldSize);
    comm->bootNet->Allgather(&lsaCount, allLsaSizes.data(), sizeof(int));
    for (int r = 0; r < comm->worldSize; r++) {
      if (allLsaSizes[r] != lsaCount) {
        MORI_SHMEM_ERROR(
            "ccoCommCreate: heterogeneous lsa sizes detected "
            "(my rank {} sees lsaSize={}, rank {} sees lsaSize={}). "
            "CCO requires uniform GPUs-per-node across all nodes.",
            comm->rank, lsaCount, r, allLsaSizes[r]);
        delete comm->ctx;
        comm->bootNet->Finalize();
        delete comm;
        *outComm = nullptr;
        return -1;
      }
    }

    comm->lsaSize = lsaCount;
    comm->myNodeStart = firstSameNode;
    comm->lsaRank = comm->rank - firstSameNode;

    MORI_SHMEM_INFO("ccoCommCreate: lsa topology rank={} lsaSize={} lsaRank={} myNodeStart={}",
                    comm->rank, comm->lsaSize, comm->lsaRank, comm->myNodeStart);
  }

  // Step 3: reserve flat VA. Always 4GB-aligned so stride4G = perRankSize >> 32
  // is lossless. perRankVmmSize == 0 defaults to GPU total memory.
  if (perRankVmmSize == 0) {
    size_t freeMem = 0, totalMem = 0;
    HIP_RUNTIME_CHECK(hipMemGetInfo(&freeMem, &totalMem));
    perRankVmmSize = totalMem;
  }
  perRankVmmSize = AlignUp(perRankVmmSize, 1ULL << 32);
  comm->perRankSize = perRankVmmSize;

  // Cache the device once — subsequent API calls (ccoMemAlloc, ccoWindow-
  // Register) reuse this without re-querying hipGetDevice. Callers MUST keep
  // the calling thread bound to this device for any later CCO API on this comm.
  HIP_RUNTIME_CHECK(hipGetDevice(&comm->hipDev));

  // Query granularity with the SAME allocProp MemAlloc will use — granularity
  // can shift when requestedHandleType (FD export) is enabled.
  hipMemAllocationProp allocProp = {};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = comm->hipDev;

  // RECOMMENDED granularity (typically 2 MiB on modern GPUs) trades a small
  // amount of internal fragmentation for fewer page-table entries, matching
  // CCO's "few large buffers" usage pattern.
  size_t granularity = 0;
  HIP_RUNTIME_CHECK(hipMemGetAllocationGranularity(&granularity, &allocProp,
                                                   hipMemAllocationGranularityRecommended));
  comm->vmmGranularity = granularity;

  // Flat VA covers the LSA team only. Cross-node peers don't use VA — RDMA
  // goes through iova=0 + offset.
  size_t totalVaSize = static_cast<size_t>(comm->lsaSize) * perRankVmmSize;
  HIP_RUNTIME_CHECK(hipMemAddressReserve(&comm->flatBase, totalVaSize, granularity, nullptr, 0));
  MORI_SHMEM_TRACE(
      "ccoCommCreate: flatBase={} totalVA={} (lsaSize={} x perRankSize={}) granularity={}",
      comm->flatBase, totalVaSize, comm->lsaSize, perRankVmmSize, granularity);

  // Per-rank slot allocator. baseAddr is THIS rank's slot in the flat VA,
  // so vaManager->Allocate() returns a dereferenceable localVa directly.
  // flatBase + lsaRank*perRankSize is granularity-aligned (perRankSize is
  // 4 GiB-aligned) and non-zero (kernel-allocated VA), satisfying
  // HeapVAManager's invariants.
  comm->vaManager.reset(new application::HeapVAManager(LocalSlotBase(comm), perRankVmmSize, 0));

  // Step 4: SDMA queue setup. Materialize only if the user opted in
  // (MORI_ENABLE_SDMA) AND at least one peer has SDMA-capable hardware.
  bool anySdmaCapable = false;
  for (int pe = 0; pe < comm->worldSize; pe++) {
    if (comm->ctx->GetPeerCapabilities(pe).canSDMA) {
      anySdmaCapable = true;
      break;
    }
  }
  if (comm->ctx->IsSdmaEnabled() && anySdmaCapable) {
    comm->sdmaNumQueue = anvil::GetSdmaNumChannels();
    comm->ctx->EnsureSdmaTransport();

    // sdmaDevHandles is lsaSize × sdmaNumQueue, indexed by lsaRank. Assumes
    // ranks bind 1:1 to GPUs within a node (rank lsa ⇒ GPU lsa).
    int srcDeviceId = comm->hipDev;
    size_t numSlots = static_cast<size_t>(comm->lsaSize) * comm->sdmaNumQueue;
    HIP_RUNTIME_CHECK(
        hipMalloc(&comm->sdmaDevHandles, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));
    HIP_RUNTIME_CHECK(
        hipMemset(comm->sdmaDevHandles, 0, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));

    for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
      int pe = comm->myNodeStart + lsa;
      if (!comm->ctx->GetPeerCapabilities(pe).canSDMA) continue;
      int dstDeviceId = lsa;
      for (int q = 0; q < comm->sdmaNumQueue; q++) {
        auto* handle = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();
        HIP_RUNTIME_CHECK(hipMemcpy(&comm->sdmaDevHandles[lsa * comm->sdmaNumQueue + q], &handle,
                                    sizeof(handle), hipMemcpyHostToDevice));
      }
    }
  } else {
    comm->sdmaNumQueue = 0;
  }

  // RDMA QP endpoints are NOT pre-allocated here. ccoDevCommCreate builds
  // a fresh QP set per DevComm via ctx->CreateAdditionalEndpoints, sized by
  // reqs.gdaContextCount, so multiple DevComms can coexist with independent
  // QP state.

  MORI_SHMEM_INFO(
      "ccoCommCreate: rank={}/{} groupId={} flatBase={} perRankSize={} "
      "granularity={} defaultNumQpPerPe={} sdmaNumQueue={} rdma={}",
      comm->rank, comm->worldSize, comm->groupId, comm->flatBase, comm->perRankSize,
      comm->vmmGranularity, comm->defaultNumQpPerPe, comm->sdmaNumQueue,
      comm->ctx->RdmaTransportEnabled());
  return 0;
}

/* ========================================================================== */
/*                             ccoCommDestroy                              */
/* ========================================================================== */

int ccoCommDestroy(ccoComm* comm) {
  if (!comm) return 0;

  MORI_SHMEM_TRACE("ccoCommDestroy: rank={}", comm->rank);

  // Safety net for callers that didn't pair every WindowRegister with a
  // matching Deregister: walk and properly deregister each straggler so
  // peer-imported handles, peer VA mappings, RDMA MRs, and GPU shadow
  // structs all get released. Each Deregister removes from comm->windows,
  // so iterate via .back() until empty.
  while (!comm->windows.empty()) {
    ccoWindowHost* wh = comm->windows.back();
    if (!wh || !wh->devPtr) {
      delete wh;
      comm->windows.pop_back();
      continue;
    }
    MORI_SHMEM_WARN(
        "ccoCommDestroy: window {} not deregistered by caller; "
        "auto-deregistering",
        wh->localPtr);
    (void)ccoWindowDeregister(comm, wh->devPtr);
  }

  for (auto& [ptr, meta] : comm->allocTable) {
    if (meta.shareFd >= 0) close(meta.shareFd);
  }
  comm->allocTable.clear();

  if (comm->sdmaDevHandles) HIP_RUNTIME_CHECK(hipFree(comm->sdmaDevHandles));

  // Release flat VA — sized to match the reservation in ccoCommCreate.
  if (comm->flatBase) {
    size_t totalVaSize = static_cast<size_t>(comm->lsaSize) * comm->perRankSize;
    HIP_RUNTIME_CHECK(hipMemAddressFree(comm->flatBase, totalVaSize));
  }

  delete comm->ctx;
  comm->bootNet->Finalize();
  delete comm->bootNet;

  delete comm;
  return 0;
}

/* ========================================================================== */
/*                              ccoMemAlloc                                */
/* ========================================================================== */

int ccoMemAlloc(ccoComm* comm, size_t size, void** outPtr) {
  if (outPtr == nullptr) {
    MORI_SHMEM_ERROR("ccoMemAlloc: outPtr is NULL");
    return -1;
  }
  if (size == 0) {
    *outPtr = nullptr;
    return 0;
  }

  size_t alignedSize = AlignUp(size, comm->vmmGranularity);

  // Reserve a slot via first-fit in the per-rank HeapVAManager. The returned
  // address IS the local VA for this rank's slot — directly dereferenceable.
  // 0 is the failure sentinel; baseAddr was set to flatBase + lsaRank*perRankSize
  // which is non-zero, so 0 unambiguously means failure.
  uintptr_t slotAddr = comm->vaManager->Allocate(alignedSize, comm->vmmGranularity);
  if (slotAddr == 0) {
    MORI_SHMEM_ERROR(
        "ccoMemAlloc: slot exhausted (no contiguous {} bytes free in perRankSize={}). "
        "Increase perRankVmmSize at ccoCommCreate or free unused allocations.",
        alignedSize, comm->perRankSize);
    return -1;
  }
  // slotOffset is the offset within the rank's perRankSize slot; needed for
  // peer-VA computation (peer's localVa = flatBase + peerLsaRank*stride + slotOffset).
  size_t slotOffset = static_cast<size_t>(slotAddr - LocalSlotBase(comm));

  MORI_SHMEM_TRACE("ccoMemAlloc: rank={} size={} alignedSize={} slotOffset={}", comm->rank, size,
                   alignedSize, slotOffset);

  // Return the reserved slot to the vaManager on any failure after this point.
  auto rollbackSlot = [&]() { (void)comm->vaManager->Free(slotAddr); };

  hipMemAllocationProp allocProp = {};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = comm->hipDev;

  hipMemGenericAllocationHandle_t physHandle = 0;
  hipError_t err = hipMemCreate(&physHandle, alignedSize, &allocProp, 0);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("ccoMemAlloc: hipMemCreate failed: {} ({})", static_cast<int>(err),
                     hipGetErrorString(err));
    rollbackSlot();
    return -1;
  }

  // Map only the local slot. Peer slots are mapped lazily in WindowRegister.
  // slotAddr already equals flatBase + lsaRank*perRankSize + slotOffset because
  // vaManager's baseAddr was set to LocalSlotBase(comm).
  void* localVa = reinterpret_cast<void*>(slotAddr);
  err = hipMemMap(localVa, alignedSize, 0, physHandle, 0);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("ccoMemAlloc: hipMemMap failed: {} ({})", static_cast<int>(err),
                     hipGetErrorString(err));
    (void)hipMemRelease(physHandle);
    rollbackSlot();
    return -1;
  }

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = comm->hipDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  err = hipMemSetAccess(localVa, alignedSize, &accessDesc, 1);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("ccoMemAlloc: hipMemSetAccess failed: {} ({})", static_cast<int>(err),
                     hipGetErrorString(err));
    (void)hipMemUnmap(localVa, alignedSize);
    (void)hipMemRelease(physHandle);
    rollbackSlot();
    return -1;
  }

  // dma-buf FD is stashed for WindowRegister to share (P2P FD exchange + RDMA MR).
  int shareFd = -1;
  err = hipMemExportToShareableHandle(reinterpret_cast<void*>(&shareFd), physHandle,
                                      hipMemHandleTypePosixFileDescriptor, 0);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("ccoMemAlloc: hipMemExportToShareableHandle failed: {} ({})",
                     static_cast<int>(err), hipGetErrorString(err));
    (void)hipMemUnmap(localVa, alignedSize);
    (void)hipMemRelease(physHandle);
    rollbackSlot();
    return -1;
  }

  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    ccoComm::AllocMeta meta;
    meta.physHandle = physHandle;
    meta.shareFd = shareFd;
    meta.slotOffset = slotOffset;
    meta.size = alignedSize;
    comm->allocTable[localVa] = meta;
  }

  *outPtr = localVa;
  MORI_SHMEM_TRACE("ccoMemAlloc: done, localPtr={}", localVa);
  return 0;
}

/* ========================================================================== */
/*                              ccoMemFree                                 */
/* ========================================================================== */

int ccoMemFree(ccoComm* comm, void* ptr) {
  if (ptr == nullptr) return 0;

  // Snapshot meta + return the slot to vaManager, then drop the cco mutex
  // before the (potentially slow) hipMem* calls so concurrent MemAlloc
  // isn't blocked. vaManager->Free takes its own mutex internally.
  ccoComm::AllocMeta meta;
  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    auto it = comm->allocTable.find(ptr);
    if (it == comm->allocTable.end()) {
      MORI_SHMEM_WARN("ccoMemFree: ptr {} not found", ptr);
      return -1;
    }
    meta = it->second;
    comm->allocTable.erase(it);
  }
  // ptr == LocalSlotBase(comm) + meta.slotOffset == the address vaManager handed out.
  (void)comm->vaManager->Free(reinterpret_cast<uintptr_t>(ptr));

  size_t alignedSize = meta.size;

  MORI_SHMEM_TRACE("ccoMemFree: rank={} ptr={} size={}", comm->rank, ptr, alignedSize);

  hipError_t err = hipMemUnmap(ptr, alignedSize);
  if (err != hipSuccess) {
    MORI_SHMEM_WARN("ccoMemFree: local hipMemUnmap failed: {} ({})", static_cast<int>(err),
                    hipGetErrorString(err));
  }
  err = hipMemRelease(meta.physHandle);
  if (err != hipSuccess) {
    MORI_SHMEM_WARN("ccoMemFree: hipMemRelease failed: {} ({})", static_cast<int>(err),
                    hipGetErrorString(err));
  }

  if (meta.shareFd >= 0) close(meta.shareFd);

  return 0;
}

/* ========================================================================== */
/*                         ccoWindowRegister (ptr)                         */
/* ========================================================================== */

int ccoWindowRegister(ccoComm* comm, void* ptr, size_t size, ccoWindow_t* outWin) {
  auto it = comm->allocTable.find(ptr);
  if (it == comm->allocTable.end()) {
    MORI_SHMEM_ERROR("ccoWindowRegister: ptr {} not in allocTable", ptr);
    return -1;
  }

  auto& meta = it->second;
  size_t slotOffset = meta.slotOffset;
  int shareFd = meta.shareFd;
  void* localPtr = ptr;
  int worldSize = comm->worldSize;
  int rank = comm->rank;

  size_t alignedSize = meta.size;

  MORI_SHMEM_TRACE("ccoWindowRegister: rank={} ptr={} size={} slotOffset={}", rank, ptr, size,
                   slotOffset);

  // P2P imported handles — collected during the FD-exchange loop below,
  // ownership later transferred to ccoWindowHost so Deregister can release.
  std::vector<hipMemGenericAllocationHandle_t> p2pImportedHandles;

  // P2P: exchange dma-buf FDs with same-node peers and map their slots into
  // the LSA flat VA.
  std::vector<int> p2pPeers;
  for (int pe = 0; pe < worldSize; pe++) {
    if (comm->ctx->CanUseP2P(pe)) {
      p2pPeers.push_back(pe);
    }
  }

  if (!p2pPeers.empty()) {
    std::vector<int> sortedGroup = p2pPeers;
    sortedGroup.push_back(rank);
    std::sort(sortedGroup.begin(), sortedGroup.end());

    int myPeerRank = 0;
    for (int i = 0; i < static_cast<int>(sortedGroup.size()); i++) {
      if (sortedGroup[i] == rank) {
        myPeerRank = i;
        break;
      }
    }
    int p2pWorldSize = static_cast<int>(sortedGroup.size());

    // Socket path must agree across the group but be unique per (group, window).
    // groupId = rank 0's pid; slotOffset identifies the window.
    std::string socketPath =
        "/tmp/mori_cco_" + std::to_string(comm->groupId) + "_" + std::to_string(slotOffset) + "_";

    // Best-effort cleanup of stale sockets from crashed runs.
    if (myPeerRank == 0) {
      for (int i = 0; i < p2pWorldSize; i++) {
        for (int j = 0; j < p2pWorldSize; j++) {
          std::string stale = socketPath + std::to_string(i) + "_" + std::to_string(j);
          unlink(stale.c_str());
        }
        unlink((socketPath + "barrier_arrive_" + std::to_string(i)).c_str());
        unlink((socketPath + "barrier_depart_" + std::to_string(i)).c_str());
      }
    }

    application::LocalBootstrapNetwork localBoot(myPeerRank, p2pWorldSize, socketPath);
    localBoot.Initialize();

    std::vector<int> myFds = {shareFd};
    std::vector<std::vector<int>> allFds;
    if (!localBoot.ExchangeFileDescriptors(myFds, allFds)) {
      MORI_SHMEM_ERROR("ccoWindowRegister: P2P FD exchange failed");
      localBoot.Finalize();
      return -1;
    }

    // All peer-supplied fds (everything in allFds except our own slot) must
    // be close()'d exactly once — closing them at the very end on every
    // exit path (success and bail) is the easiest invariant to enforce.
    // hipMemImportFromShareableHandle already dup's the underlying dma-buf
    // reference internally, so it's safe to delay the close to here.
    auto closePeerFds = [&]() {
      for (int i = 0; i < static_cast<int>(allFds.size()); i++) {
        if (i == myPeerRank) continue;  // our own shareFd is owned by meta
        for (int fd : allFds[i]) {
          if (fd >= 0) close(fd);
        }
      }
      allFds.clear();
    };

    std::vector<int> globalToPeer(worldSize, -1);
    for (int i = 0; i < p2pWorldSize; i++) {
      globalToPeer[sortedGroup[i]] = i;
    }

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = comm->hipDev;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    // Track already-mapped peers so we can roll back if any later peer
    // fails — partial success would leave the window with missing P2P
    // links and silently segfault when a kernel touches a missing peer.
    struct MappedPeer {
      hipMemGenericAllocationHandle_t handle;
      void* peerVa;
    };
    std::vector<MappedPeer> mappedPeers;
    mappedPeers.reserve(p2pPeers.size());

    auto rollbackMappedPeers = [&]() {
      for (auto& mp : mappedPeers) {
        (void)hipMemUnmap(mp.peerVa, alignedSize);
        (void)hipMemRelease(mp.handle);
      }
      mappedPeers.clear();
    };

    auto bail = [&]() {
      rollbackMappedPeers();
      closePeerFds();
      localBoot.Finalize();
    };

    for (int pe : p2pPeers) {
      int pr = globalToPeer[pe];
      if (pr < 0 || pr >= static_cast<int>(allFds.size())) {
        MORI_SHMEM_ERROR("ccoWindowRegister: PE {} missing in FD exchange result", pe);
        bail();
        return -1;
      }
      int peerFd = allFds[pr][0];
      if (peerFd < 0) {
        MORI_SHMEM_ERROR("ccoWindowRegister: PE {} delivered invalid FD ({})", pe, peerFd);
        bail();
        return -1;
      }

      hipMemGenericAllocationHandle_t importedHandle;
      hipError_t err = hipMemImportFromShareableHandleCompat(&importedHandle, peerFd,
                                                             hipMemHandleTypePosixFileDescriptor);
      if (err != hipSuccess) {
        MORI_SHMEM_ERROR("ccoWindowRegister: import from PE {} failed: {}", pe,
                         static_cast<int>(err));
        bail();
        return -1;
      }

      int peerLsaRank = pe - comm->myNodeStart;
      void* peerVa = static_cast<char*>(comm->flatBase) +
                     static_cast<size_t>(peerLsaRank) * comm->perRankSize + slotOffset;
      hipError_t mapErr = hipMemMap(peerVa, alignedSize, 0, importedHandle, 0);
      if (mapErr != hipSuccess) {
        MORI_SHMEM_ERROR("ccoWindowRegister: hipMemMap PE {} failed: {}", pe,
                         static_cast<int>(mapErr));
        (void)hipMemRelease(importedHandle);
        bail();
        return -1;
      }

      // hipMemSetAccess can transiently fail under concurrent VMM operations.
      hipError_t setErr = hipSuccess;
      for (int retry = 0; retry < 5; retry++) {
        setErr = hipMemSetAccess(peerVa, alignedSize, &accessDesc, 1);
        if (setErr == hipSuccess) break;
        usleep(1000 * (1 << retry));
      }
      if (setErr != hipSuccess) {
        MORI_SHMEM_ERROR("ccoWindowRegister: hipMemSetAccess PE {} failed after retries: {}", pe,
                         static_cast<int>(setErr));
        (void)hipMemUnmap(peerVa, alignedSize);
        (void)hipMemRelease(importedHandle);
        bail();
        return -1;
      }

      mappedPeers.push_back({importedHandle, peerVa});
    }

    // Stash handles on the WindowHost so Deregister can release them.
    p2pImportedHandles.reserve(mappedPeers.size());
    for (auto& mp : mappedPeers) p2pImportedHandles.push_back(mp.handle);

    closePeerFds();
    localBoot.Finalize();
  }

  // RDMA MR registration + rkey Allgather.
  uint32_t lkey = 0;
  uint32_t localRkey = 0;

  application::RdmaDeviceContext* rdmaDevCtx = comm->ctx->GetRdmaDeviceContext();
  if (rdmaDevCtx && shareFd >= 0) {
    application::RdmaMemoryRegion mr;
    if (comm->iovaZeroMode) {
      mr = rdmaDevCtx->RegisterRdmaMemoryRegionDmabufIova0(localPtr, size, shareFd);
    } else {
      mr = rdmaDevCtx->RegisterRdmaMemoryRegionDmabuf(localPtr, size, shareFd);
    }
    lkey = mr.lkey;
    localRkey = mr.rkey;
  }

  // Allgather rkeys into a std::vector so an exception in Allgather doesn't
  // leak the host buffer (HIP_RUNTIME_CHECKs below abort the process anyway,
  // but bootNet->Allgather is throwing).
  std::vector<uint32_t> peerRkeys_host(worldSize, 0);
  peerRkeys_host[rank] = localRkey;
  comm->bootNet->Allgather(&localRkey, peerRkeys_host.data(), sizeof(uint32_t));

  // SDMA signal pool is per-DevComm, materialized by ccoDevCommCreate.
  // WindowRegister no longer allocates SDMA state — kernels look up signals
  // via devComm->sdma.

  uint32_t* peerRkeys_gpu = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&peerRkeys_gpu, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(peerRkeys_gpu, peerRkeys_host.data(), sizeof(uint32_t) * worldSize,
                              hipMemcpyHostToDevice));

  ccoWindowDevice hostShadow = {};
  hostShadow.winBase = static_cast<char*>(comm->flatBase) + slotOffset;
  hostShadow.stride4G = static_cast<uint32_t>(comm->perRankSize >> 32);
  hostShadow.lsaRank = comm->lsaRank;
  hostShadow.ibgdaWin.peerRkeys = peerRkeys_gpu;
  hostShadow.ibgdaWin.lkey = lkey;

  ccoWindowDevice* devPtr = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&devPtr, sizeof(ccoWindowDevice)));
  HIP_RUNTIME_CHECK(hipMemcpy(devPtr, &hostShadow, sizeof(ccoWindowDevice), hipMemcpyHostToDevice));

  // Publish into the per-comm window table (drives findWindow lookups).
  ccoComm::WindowTableEntry tableEntry;
  tableEntry.base = reinterpret_cast<uintptr_t>(localPtr);
  tableEntry.size = static_cast<uintptr_t>(size);
  tableEntry.devPtr = devPtr;
  comm->windowTableEntries.push_back(tableEntry);

  auto* wh = new ccoWindowHost();
  wh->localPtr = localPtr;
  wh->size = size;
  wh->devPtr = devPtr;
  wh->peerRkeys_gpu = peerRkeys_gpu;
  wh->peerImportedHandles = std::move(p2pImportedHandles);
  comm->windows.push_back(wh);

  *outWin = devPtr;

  char* winBase = static_cast<char*>(comm->flatBase) + slotOffset;
  MORI_SHMEM_INFO("ccoWindowRegister: rank={} win={} winBase={} size={} slotOffset={} lkey={}",
                  rank, (void*)devPtr, (void*)winBase, size, slotOffset, lkey);
  for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
    int pe = comm->myNodeStart + lsa;
    void* peerVa = winBase + static_cast<size_t>(lsa) * comm->perRankSize;
    MORI_SHMEM_INFO("  LSA[{}] (PE {}): flatVA={} rkey={}", lsa, pe, peerVa, peerRkeys_host[pe]);
  }
  for (int pe = 0; pe < worldSize; pe++) {
    if (pe >= comm->myNodeStart && pe < comm->myNodeStart + comm->lsaSize) continue;
    MORI_SHMEM_INFO("  XNODE PE {}: rkey={} (RDMA via iova=0)", pe, peerRkeys_host[pe]);
  }
  // peerRkeys_host is std::vector — destructs cleanly at scope exit.

  return 0;
}

/* ========================================================================== */
/*                      ccoWindowRegister (convenience)                    */
/* ========================================================================== */

int ccoWindowRegister(ccoComm* comm, size_t size, ccoWindow_t* outWin, void** localPtr) {
  void* ptr = nullptr;
  int ret = ccoMemAlloc(comm, size, &ptr);
  if (ret != 0) return ret;

  ret = ccoWindowRegister(comm, ptr, size, outWin);
  if (ret != 0) {
    ccoMemFree(comm, ptr);
    return ret;
  }

  *localPtr = ptr;
  return 0;
}

/* ========================================================================== */
/*                          ccoWindowDeregister                            */
/* ========================================================================== */

int ccoWindowDeregister(ccoComm* comm, ccoWindow_t win) {
  ccoWindowHost* wh = nullptr;
  size_t idx = 0;
  for (size_t i = 0; i < comm->windows.size(); i++) {
    if (comm->windows[i]->devPtr == win) {
      wh = comm->windows[i];
      idx = i;
      break;
    }
  }
  if (!wh) {
    MORI_SHMEM_WARN("ccoWindowDeregister: win {} not found", (void*)win);
    return -1;
  }

  MORI_SHMEM_TRACE("ccoWindowDeregister: rank={} ptr={}", comm->rank, wh->localPtr);

  // Unmap the P2P peer slots that WindowRegister mapped (ENOMAP is fine).
  auto allocIt = comm->allocTable.find(wh->localPtr);
  if (allocIt != comm->allocTable.end()) {
    size_t slotOff = allocIt->second.slotOffset;
    size_t allocSize = allocIt->second.size;
    for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
      if (lsa == comm->lsaRank) continue;
      int pe = comm->myNodeStart + lsa;
      if (!comm->ctx->CanUseP2P(pe)) continue;
      void* peerVa = static_cast<char*>(comm->flatBase) +
                     static_cast<size_t>(lsa) * comm->perRankSize + slotOff;
      (void)hipMemUnmap(peerVa, allocSize);
    }
  }

  // Drop refcount on each peer's imported handle. hipMemUnmap above
  // detaches VA mappings but doesn't release the handle itself.
  for (auto handle : wh->peerImportedHandles) {
    (void)hipMemRelease(handle);
  }
  wh->peerImportedHandles.clear();

  auto& entries = comm->windowTableEntries;
  entries.erase(
      std::remove_if(entries.begin(), entries.end(),
                     [win](const ccoComm::WindowTableEntry& e) { return e.devPtr == win; }),
      entries.end());

  application::RdmaDeviceContext* rdmaDevCtx = comm->ctx->GetRdmaDeviceContext();
  if (rdmaDevCtx) rdmaDevCtx->DeregisterRdmaMemoryRegion(wh->localPtr);

  if (wh->peerRkeys_gpu) HIP_RUNTIME_CHECK(hipFree(wh->peerRkeys_gpu));
  if (wh->devPtr) HIP_RUNTIME_CHECK(hipFree(wh->devPtr));

  comm->windows.erase(comm->windows.begin() + idx);
  delete wh;
  return 0;
}

/* ========================================================================== */
/*                            ccoDevCommCreate                             */
/* ========================================================================== */

int ccoDevCommCreate(ccoComm* comm, const ccoDevCommRequirements* reqs, ccoDevComm* outDevComm) {
  MORI_SHMEM_TRACE("ccoDevCommCreate: rank={}", comm->rank);

  // Forward-compat: validate {magic, version}.
  if (reqs == nullptr) {
    MORI_SHMEM_ERROR(
        "ccoDevCommCreate: reqs is NULL — must initialize via "
        "CCO_DEV_COMM_REQUIREMENTS_INITIALIZER");
    return -1;
  }
  if (reqs->magic != CCO_API_MAGIC) {
    MORI_SHMEM_ERROR(
        "ccoDevCommCreate: reqs->magic mismatch (got {:#x}, expect {:#x}) — "
        "must initialize via CCO_DEV_COMM_REQUIREMENTS_INITIALIZER",
        reqs->magic, CCO_API_MAGIC);
    return -1;
  }
  if (reqs->version > CCO_API_VERSION) {
    MORI_SHMEM_WARN("ccoDevCommCreate: reqs->version={} > runtime CCO_API_VERSION={}",
                    reqs->version, CCO_API_VERSION);
  }

  // Resolve connection type. CROSSNODE collapses to NONE on single-node
  // deployments (no cross-node peers exist). RAIL collapses to NONE if it
  // ends up with zero peers (single-node, or self-rail only).
  ccoGdaConnectionType connType = reqs->gdaConnectionType;
  if (connType == CCO_GDA_CONNECTION_CROSSNODE && comm->lsaSize == comm->worldSize) {
    MORI_SHMEM_TRACE("ccoDevCommCreate: single-node, CROSSNODE -> NONE");
    connType = CCO_GDA_CONNECTION_NONE;
  }

  ccoDevComm hostShadow = {};
  hostShadow.rank = comm->rank;
  hostShadow.worldSize = comm->worldSize;
  hostShadow.lsaSize = comm->lsaSize;
  hostShadow.lsaRank = comm->lsaRank;
  hostShadow.myNodeStart = comm->myNodeStart;
  hostShadow.gdaConnType = connType;
  hostShadow.flatBase = comm->flatBase;
  hostShadow.perRankSize = comm->perRankSize;

  // Fresh QP set per DevComm.
  ccoIbgdaContext& ibgda = hostShadow.ibgda;
  int numQpPerPe = reqs->gdaContextCount > 0 ? reqs->gdaContextCount : comm->defaultNumQpPerPe;
  ibgda.numQpPerPe = numQpPerPe;

  size_t numEps = static_cast<size_t>(comm->worldSize) * numQpPerPe;
  core::RdmaEndpointDevice* epsGpu = nullptr;

  // Build the peer mask once based on connType. Context::CreateAdditional /
  // ConnectAdditional take the same mask. Empty mask if NONE.
  //
  // Layout assumption (HARD CONTRACT enforced at CommCreate): ranks are
  // node-major contiguous and lsaSize is uniform across nodes, so each
  // peer's lsaRank is `peer % comm->lsaSize`.
  std::vector<bool> peerMask;
  if (connType != CCO_GDA_CONNECTION_NONE) {
    peerMask.assign(comm->worldSize, false);
    for (int peer = 0; peer < comm->worldSize; peer++) {
      if (peer == comm->rank) continue;
      const auto& cap = comm->ctx->GetPeerCapabilities(peer);
      switch (connType) {
        case CCO_GDA_CONNECTION_FULL:
          peerMask[peer] = cap.canRDMA;
          break;
        case CCO_GDA_CONNECTION_CROSSNODE:
          peerMask[peer] = cap.canRDMA && !cap.sameHost;
          break;
        case CCO_GDA_CONNECTION_RAIL: {
          const int myLsaRank = comm->lsaRank;
          const int peerLsaRank = peer % comm->lsaSize;
          peerMask[peer] = cap.canRDMA && !cap.sameHost && (peerLsaRank == myLsaRank);
          break;
        }
        default:
          break;
      }
    }
    // Collapse to NONE if the resolved mask is empty (e.g. RAIL on single
    // node, or CROSSNODE that lost all peers).
    if (std::none_of(peerMask.begin(), peerMask.end(), [](bool b) { return b; })) {
      MORI_SHMEM_TRACE(
          "ccoDevCommCreate: resolved peer mask is empty, "
          "downgrading connType {} -> NONE",
          static_cast<int>(connType));
      connType = CCO_GDA_CONNECTION_NONE;
      peerMask.clear();
    }
    hostShadow.gdaConnType = connType;  // may have been collapsed above
  }

  if (connType != CCO_GDA_CONNECTION_NONE && comm->ctx->RdmaTransportEnabled()) {
    // Collective: every rank must call CreateAdditionalEndpoints together.
    auto newEps = comm->ctx->CreateAdditionalEndpoints(numQpPerPe, peerMask);
    comm->ctx->ConnectAdditionalEndpoints(newEps, numQpPerPe, peerMask);

    // Note: post-Connect RTS verification via ibv_query_qp doesn't work for
    // direct-verbs providers (bnxt, mlx5), which keep QPs in their own
    // containers and leave ibvHandle.qp null. Provider-side QueryQpState is
    // a TODO; for now, rely on modify_qp's internal check + the transport
    // map dump (MORI_CCO_LOG_TRANSPORT) for visibility.

    std::vector<core::RdmaEndpointDevice> epsHost(numEps);
    for (size_t i = 0; i < numEps; i++) {
      epsHost[i].vendorId = newEps[i].vendorId;
      epsHost[i].qpn = newEps[i].handle.qpn;
      epsHost[i].wqHandle = newEps[i].wqHandle;
      epsHost[i].cqHandle = newEps[i].cqHandle;
      epsHost[i].atomicIbuf = newEps[i].atomicIbuf;
      // Cache the GDA provider from the first connected endpoint (empty peer
      // slots keep vendorId==Unknown) as an informational parameter on the comm.
      if (comm->providerType == CCO_PROVIDER_UNKNOWN) {
        core::ProviderType p = epsHost[i].GetProviderType();
        if (p != core::ProviderType::Unknown) comm->providerType = static_cast<ccoProviderType>(p);
      }
    }

    HIP_RUNTIME_CHECK(hipMalloc(&epsGpu, numEps * sizeof(core::RdmaEndpointDevice)));
    HIP_RUNTIME_CHECK(hipMemcpy(epsGpu, epsHost.data(), numEps * sizeof(core::RdmaEndpointDevice),
                                hipMemcpyHostToDevice));
  }
  ibgda.endpoints = epsGpu;

  // Resource window: a CCO symmetric window backing this DevComm's session
  // state. Lives in the LSA flat VA + has an RDMA MR, so each block inside
  // is simultaneously P2P-load/store-addressable by intra-node peers AND
  // RDMA-write-target-addressable by cross-node peers — every per-session
  // sub-allocation gets the full transport matrix "for free".
  //
  // Current residents:
  //   * IBGDA signal / shadows / counter pool (gdaConnType != NONE)
  //   * LSA barrier inbox+state buffer        (lsaBarrierCount > 0)
  //
  // Layout pins signalBufOffset == 0 so a peer's RDMA atomic add still uses
  // raddr = signal_slot_id * 8 (no per-rank offset shift needed).
  // counterBuf is software-incremented (via CQ-polling + GPU store) —
  // placed in the pool for uniformity even though peers never write to it.
  //
  // Allocated BEFORE the windowTable build below so the GPU windowTable
  // includes it (a kernel can findWindow(devComm.resourceWindow) too).
  // Rail team size = # of nodes (one peer per node at this lsaRank slot).
  // GDA-Rail barriers only make sense when there are cross-node peers AND
  // we actually have RDMA QPs to talk to them; otherwise collapse to 0.
  int nNodes = comm->worldSize / comm->lsaSize;
  bool gdaRailUsable = (connType != CCO_GDA_CONNECTION_NONE) && (nNodes > 1);

  int signalCountUser = (connType == CCO_GDA_CONNECTION_NONE) ? 0 : reqs->gdaSignalCount;
  int counterCount = (connType == CCO_GDA_CONNECTION_NONE) ? 0 : reqs->gdaCounterCount;
  int lsaBarrierCount = reqs->lsaBarrierCount;
  int railGdaBarrierCount = gdaRailUsable ? reqs->railGdaBarrierCount : 0;
  int hybridBarrierCount = reqs->barrierCount;
  // hybrid Rail half is only active when we have cross-rail peers + RDMA.
  int hybridRailBarrierCount = gdaRailUsable ? hybridBarrierCount : 0;

  // Signal slot assignment:
  //   [0 .. signalCountUser)                 — user-visible signal slots
  //   [signalCountUser .. +A)                — railGdaBarrier (A = N*nNodes)
  //   [.. +B)                                — hybridRailGdaBarrier (B = N*nNodes)
  uint32_t railGdaBarrierSignal0 = static_cast<uint32_t>(signalCountUser);
  int railGdaBarrierSignals = railGdaBarrierCount * nNodes;
  uint32_t hybridRailBarrierSignal0 =
      railGdaBarrierSignal0 + static_cast<uint32_t>(railGdaBarrierSignals);
  int hybridRailBarrierSignals = hybridRailBarrierCount * nNodes;
  int signalCount = signalCountUser + railGdaBarrierSignals + hybridRailBarrierSignals;
  ibgda.signalCount = signalCount;
  ibgda.counterCount = counterCount;

  auto alignTo = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
  auto lsaBarBytes = [&](int n) -> size_t {
    // Multimem epoch/inbox omitted; add when hardware support lands.
    return static_cast<size_t>(n + n * comm->lsaSize) * sizeof(uint32_t);
  };

  struct ResourceWindowLayout {
    size_t signalBufOffset = 0;
    size_t signalShadowsOffset = 0;
    size_t counterBufOffset = 0;
    size_t lsaBarrierOffset = 0;
    size_t lsaBarrierBytes = 0;
    size_t hybridLsaBarrierOffset = 0;
    size_t hybridLsaBarrierBytes = 0;
    size_t totalSize = 0;
  } layout;
  if (lsaBarrierCount > 0) layout.lsaBarrierBytes = lsaBarBytes(lsaBarrierCount);
  if (hybridBarrierCount > 0) layout.hybridLsaBarrierBytes = lsaBarBytes(hybridBarrierCount);

  bool needWindow =
      signalCount > 0 || counterCount > 0 || lsaBarrierCount > 0 || hybridBarrierCount > 0;
  if (needWindow) {
    size_t off = 0;
    layout.signalBufOffset = off;  // pinned at 0
    off += static_cast<size_t>(signalCount) * sizeof(uint64_t);
    off = alignTo(off, 8);
    layout.signalShadowsOffset = off;
    off += static_cast<size_t>(signalCount) * sizeof(uint64_t);
    off = alignTo(off, 8);
    layout.counterBufOffset = off;
    off += static_cast<size_t>(counterCount) * sizeof(uint64_t);
    // LSA barrier slabs: 128B align so peers' P2P stores hit a cache-line-
    // isolated region.
    if (lsaBarrierCount > 0) {
      off = alignTo(off, 128);
      layout.lsaBarrierOffset = off;
      off += layout.lsaBarrierBytes;
    }
    if (hybridBarrierCount > 0) {
      off = alignTo(off, 128);
      layout.hybridLsaBarrierOffset = off;
      off += layout.hybridLsaBarrierBytes;
    }
    layout.totalSize = off;
  }

  void* resourceWindowPtr = nullptr;
  ccoWindow_t resourceWindow = nullptr;
  if (layout.totalSize > 0) {
    if (ccoMemAlloc(comm, layout.totalSize, &resourceWindowPtr) != 0) {
      MORI_SHMEM_ERROR("ccoDevCommCreate: resource window MemAlloc failed");
      if (epsGpu) HIP_RUNTIME_CHECK(hipFree(epsGpu));
      return -1;
    }
    HIP_RUNTIME_CHECK(hipMemset(resourceWindowPtr, 0, layout.totalSize));
    if (ccoWindowRegister(comm, resourceWindowPtr, layout.totalSize, &resourceWindow) != 0) {
      MORI_SHMEM_ERROR("ccoDevCommCreate: resource window Register failed");
      (void)ccoMemFree(comm, resourceWindowPtr);
      if (epsGpu) HIP_RUNTIME_CHECK(hipFree(epsGpu));
      return -1;
    }
    auto* base = static_cast<uint8_t*>(resourceWindowPtr);
    if (signalCount > 0) {
      ibgda.signalBuf = reinterpret_cast<uint64_t*>(base + layout.signalBufOffset);
      ibgda.signalShadows = reinterpret_cast<uint64_t*>(base + layout.signalShadowsOffset);
    }
    if (counterCount > 0) {
      ibgda.counterBuf = reinterpret_cast<uint64_t*>(base + layout.counterBufOffset);
    }
    if (lsaBarrierCount > 0) {
      hostShadow.lsaBarrier.bufOffset = static_cast<uint32_t>(layout.lsaBarrierOffset);
      hostShadow.lsaBarrier.nBarriers = lsaBarrierCount;
    }
    if (hybridBarrierCount > 0) {
      hostShadow.hybridLsaBarrier.bufOffset = static_cast<uint32_t>(layout.hybridLsaBarrierOffset);
      hostShadow.hybridLsaBarrier.nBarriers = hybridBarrierCount;
    }

    // Snapshot the GPU resource-window struct into the DevComm so kernels
    // can read winBase/stride4G/ibgdaWin.{lkey,peerRkeys} straight out of
    // kernel cmem (no extra GPU memory load through the pointer).
    HIP_RUNTIME_CHECK(hipMemcpy(&hostShadow.resourceWindow_inlined, resourceWindow,
                                sizeof(ccoWindowDevice), hipMemcpyDeviceToHost));
  }
  hostShadow.resourceWindow = resourceWindow;

  // GDA barrier handles point into ibgda.signalBuf; no resource-window bytes
  // consumed. Disabled handles stay {0,0}.
  if (railGdaBarrierCount > 0) {
    hostShadow.railGdaBarrier.signal0 = railGdaBarrierSignal0;
    hostShadow.railGdaBarrier.nBarriers = railGdaBarrierCount;
  }
  if (hybridRailBarrierCount > 0) {
    hostShadow.hybridRailGdaBarrier.signal0 = hybridRailBarrierSignal0;
    hostShadow.hybridRailGdaBarrier.nBarriers = hybridRailBarrierCount;
  }

  MORI_SHMEM_TRACE(
      "ccoDevCommCreate: resourceWindow={} ptr={} totalSize={} signals={} "
      "counters={} lsaBar={} lsaBarOff={:#x} hybLsaBar={} hybLsaBarOff={:#x} "
      "railGdaBar={} railGdaSig0={} hybRailGdaBar={} hybRailGdaSig0={}",
      (void*)resourceWindow, resourceWindowPtr, layout.totalSize, signalCount, counterCount,
      lsaBarrierCount, layout.lsaBarrierOffset, hybridBarrierCount, layout.hybridLsaBarrierOffset,
      railGdaBarrierCount, railGdaBarrierSignal0, hybridRailBarrierCount, hybridRailBarrierSignal0);

  // Build window-table linked list on GPU.
  const auto& tableEntries = comm->windowTableEntries;
  size_t numWindows = tableEntries.size();
  size_t numNodes = (numWindows + CCO_WINDOW_TABLE_SIZE - 1) / CCO_WINDOW_TABLE_SIZE;
  if (numNodes == 0) numNodes = 1;

  std::vector<ccoWindowTableNode*> gpuNodes(numNodes, nullptr);
  for (size_t n = 0; n < numNodes; n++) {
    HIP_RUNTIME_CHECK(hipMalloc(&gpuNodes[n], sizeof(ccoWindowTableNode)));
    HIP_RUNTIME_CHECK(hipMemset(gpuNodes[n], 0, sizeof(ccoWindowTableNode)));
  }

  for (size_t n = 0; n < numNodes; n++) {
    ccoWindowTableNode nodeHost = {};
    size_t base = n * CCO_WINDOW_TABLE_SIZE;
    for (int i = 0; i < CCO_WINDOW_TABLE_SIZE; i++) {
      size_t idx = base + i;
      if (idx < numWindows) {
        nodeHost.entries[i].base = tableEntries[idx].base;
        nodeHost.entries[i].size = tableEntries[idx].size;
        nodeHost.entries[i].window = tableEntries[idx].devPtr;
      }
    }
    nodeHost.next = (n + 1 < numNodes) ? gpuNodes[n + 1] : nullptr;
    HIP_RUNTIME_CHECK(
        hipMemcpy(gpuNodes[n], &nodeHost, sizeof(ccoWindowTableNode), hipMemcpyHostToDevice));
  }
  hostShadow.windowTable = gpuNodes[0];

  MORI_SHMEM_TRACE("ccoDevCommCreate: windowTable with {} windows in {} nodes", numWindows,
                   numNodes);

  // SDMA signal pool (per-DevComm). Materialized only if comm-level SDMA
  // queues are up. Pool: [lsaSize × sdmaNumQueue × uint64], shared by all
  // windows. Kernels index via devComm->sdma.signalBuf[lsaPeer * sdmaNumQueue + qId].
  //
  // SPMT-safe peer-pointer exchange: hipIpcOpenMemHandle fails when the
  // handle was exported by the same process, so for SPMT we Allgather raw
  // VAs alongside IPC handles and pick per-peer based on SameProcessP2P.
  // (See shmem's SymmMemManager::Register for the same pattern.)
  ccoSdmaContext& sdma = hostShadow.sdma;
  sdma.sdmaNumQueue = static_cast<uint32_t>(comm->sdmaNumQueue);
  if (comm->sdmaNumQueue > 0) {
    size_t poolBytes = static_cast<size_t>(comm->lsaSize) * comm->sdmaNumQueue * sizeof(HSAuint64);
    HIP_RUNTIME_CHECK(hipMalloc(&sdma.signalBuf, poolBytes));
    HIP_RUNTIME_CHECK(hipMemset(sdma.signalBuf, 0, poolBytes));
    HIP_RUNTIME_CHECK(hipMalloc(&sdma.expectSignals, poolBytes));
    HIP_RUNTIME_CHECK(hipMemset(sdma.expectSignals, 0, poolBytes));

    // Use std::vector for host scratch so any exception thrown by
    // bootNet->Allgather (cross-rank comm) doesn't leak heap.
    hipIpcMemHandle_t myHandle;
    HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&myHandle, sdma.signalBuf));
    std::vector<hipIpcMemHandle_t> handles(comm->worldSize);
    comm->bootNet->Allgather(&myHandle, handles.data(), sizeof(hipIpcMemHandle_t));

    // Also Allgather raw VAs — used for same-process peers where IPC fails.
    HSAuint64* myRawVa = sdma.signalBuf;
    std::vector<HSAuint64*> rawVas(comm->worldSize, nullptr);
    comm->bootNet->Allgather(&myRawVa, rawVas.data(), sizeof(HSAuint64*));

    std::vector<HSAuint64*> peerPtrs_host(comm->lsaSize, nullptr);
    peerPtrs_host[comm->lsaRank] = sdma.signalBuf;
    for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
      if (lsa == comm->lsaRank) continue;
      int pe = comm->myNodeStart + lsa;
      if (!comm->ctx->GetPeerCapabilities(pe).canSDMA) continue;

      if (comm->ctx->SameProcessP2P(pe)) {
        // Same process (SPMT): use peer's raw VA, defensively enable peer
        // access for its device. hipIpcMemLazyEnablePeerAccess doesn't run
        // here because we're not opening an IPC handle.
        peerPtrs_host[lsa] = rawVas[pe];
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, rawVas[pe]) == hipSuccess &&
            attr.device != hipInvalidDeviceId) {
          hipError_t peerErr = hipDeviceEnablePeerAccess(attr.device, 0);
          (void)hipGetLastError();
          if (peerErr != hipSuccess && peerErr != hipErrorPeerAccessAlreadyEnabled) {
            MORI_SHMEM_WARN(
                "ccoDevCommCreate: hipDeviceEnablePeerAccess(peer={}, "
                "device={}) failed: {}",
                pe, attr.device, hipGetErrorString(peerErr));
          }
        } else {
          (void)hipGetLastError();
        }
      } else {
        // Cross-process: standard IPC open.
        void* mapped = nullptr;
        HIP_RUNTIME_CHECK(hipIpcOpenMemHandle(&mapped, handles[pe], hipIpcMemLazyEnablePeerAccess));
        peerPtrs_host[lsa] = reinterpret_cast<HSAuint64*>(mapped);
      }
    }
    HIP_RUNTIME_CHECK(hipMalloc(&sdma.peerSignalPtrs, sizeof(HSAuint64*) * comm->lsaSize));
    HIP_RUNTIME_CHECK(hipMemcpy(sdma.peerSignalPtrs, peerPtrs_host.data(),
                                sizeof(HSAuint64*) * comm->lsaSize, hipMemcpyHostToDevice));
    // handles / rawVas / peerPtrs_host destructed by std::vector RAII.

    sdma.deviceHandles = comm->sdmaDevHandles;
    MORI_SHMEM_TRACE(
        "ccoDevCommCreate: SDMA pool signalBuf={} expectSignals={} "
        "peerSignalPtrs={} numQueue={}",
        (void*)sdma.signalBuf, (void*)sdma.expectSignals, (void*)sdma.peerSignalPtrs,
        sdma.sdmaNumQueue);
  }

  // Fill the caller-provided host struct in place — no device allocation. It
  // holds device pointers (windowTable, endpoints, resource pools) but lives on
  // the host; kernels take it by value.
  *outDevComm = hostShadow;
  MORI_SHMEM_INFO("ccoDevCommCreate: rank={} windows={} signals={} counters={} resourceWindow={}",
                  comm->rank, numWindows, signalCount, counterCount, (void*)resourceWindow);

  // Optional transport map dump, gated on MORI_CCO_LOG_TRANSPORT. Shows each
  // peer's hardware capability (canP2P/canSDMA/canRDMA) alongside whether
  // this DevComm has materialized resources for that transport — useful for
  // verifying gdaConnectionType behavior end-to-end.
  if (const char* env = std::getenv("MORI_CCO_LOG_TRANSPORT")) {
    if (env[0] != '0') {
      const char* connTypeStr = "?";
      switch (connType) {
        case CCO_GDA_CONNECTION_NONE:
          connTypeStr = "NONE";
          break;
        case CCO_GDA_CONNECTION_FULL:
          connTypeStr = "FULL";
          break;
        case CCO_GDA_CONNECTION_CROSSNODE:
          connTypeStr = "CROSSNODE";
          break;
        case CCO_GDA_CONNECTION_RAIL:
          connTypeStr = "RAIL";
          break;
      }
      const bool sdmaPoolActive =
          (hostShadow.sdma.sdmaNumQueue > 0 && hostShadow.sdma.signalBuf != nullptr);

      // Build the entire table into one string and emit atomically — avoids
      // interleaving when ranks fork-write to the same stderr concurrently.
      std::string buf;
      buf.reserve(256 + 64 * comm->worldSize);
      char line[160];
      snprintf(line, sizeof(line),
               "[cco] DevComm rank=%d/%d connType=%s — transport map "
               "(CAP=hardware capability, ACT=materialized by this DevComm):\n",
               comm->rank, comm->worldSize, connTypeStr);
      buf += line;
      buf += "  peer  | cap                | active\n";
      buf += "  ------+--------------------+--------------------\n";
      const bool rdmaEnabled = comm->ctx->RdmaTransportEnabled();
      for (int peer = 0; peer < comm->worldSize; peer++) {
        if (peer == comm->rank) {
          snprintf(line, sizeof(line), "  %4d* | SELF               | SELF\n", peer);
          buf += line;
          continue;
        }
        const auto& cap = comm->ctx->GetPeerCapabilities(peer);

        // Active = "has resources / connectivity right now":
        //   P2P  — sameHost peer with capability (LSA flat-VA covers
        //          intra-node; window-level FD exchange happens in WindowRegister).
        //   SDMA — this DevComm allocated an SDMA signal pool AND peer canSDMA.
        //   RDMA — this DevComm allocated a QP for peer (depends on connType).
        const bool actP2P = cap.canP2P && cap.sameHost;
        const bool actSDMA = sdmaPoolActive && cap.canSDMA;
        bool actRDMA = false;
        if (connType != CCO_GDA_CONNECTION_NONE && rdmaEnabled &&
            peer < static_cast<int>(peerMask.size())) {
          actRDMA = peerMask[peer];
        }

        auto fmt = [](bool p2p, bool sdma, bool rdma, char* out, size_t n) {
          out[0] = '\0';
          if (p2p) snprintf(out + strlen(out), n - strlen(out), "P2P ");
          if (sdma) snprintf(out + strlen(out), n - strlen(out), "SDMA ");
          if (rdma) snprintf(out + strlen(out), n - strlen(out), "RDMA ");
          if (out[0] == '\0') snprintf(out, n, "(none)");
        };
        char capStr[32], actStr[32];
        fmt(cap.canP2P, cap.canSDMA, cap.canRDMA, capStr, sizeof(capStr));
        fmt(actP2P, actSDMA, actRDMA, actStr, sizeof(actStr));
        snprintf(line, sizeof(line), "  %4d  | %-18s | %-18s%s\n", peer, capStr, actStr,
                 cap.sameHost ? " (intra-node)" : "");
        buf += line;
      }
      fwrite(buf.data(), 1, buf.size(), stderr);
      fflush(stderr);
    }
  }

  return 0;
}

/* ========================================================================== */
/*                           ccoDevCommDestroy                             */
/* ========================================================================== */

int ccoDevCommDestroy(ccoComm* comm, ccoDevComm* devComm) {
  if (!devComm) return 0;

  // devComm is the caller's host struct (filled by ccoDevCommCreate); read its
  // device-pointer fields directly to release the resources they reference.
  ccoDevComm& hostShadow = *devComm;

  auto& ibgda = hostShadow.ibgda;

  // Resource window: undoes ccoMemAlloc + ccoWindowRegister done in
  // DevCommCreate. WindowDeregister handles MR deregister, peer-VA unmap,
  // imported handle release, and frees the GPU ccoWindowDevice. MemFree
  // then releases the physical pages and returns the slot to vaManager.
  // Look up the wh->localPtr before Deregister erases the entry.
  if (hostShadow.resourceWindow && comm) {
    void* resourceWindowLocalPtr = nullptr;
    for (auto* wh : comm->windows) {
      if (wh && wh->devPtr == hostShadow.resourceWindow) {
        resourceWindowLocalPtr = wh->localPtr;
        break;
      }
    }
    (void)ccoWindowDeregister(comm, hostShadow.resourceWindow);
    if (resourceWindowLocalPtr) (void)ccoMemFree(comm, resourceWindowLocalPtr);
  }

  // QP endpoints array. signalBuf/Shadows/counterBuf are sub-pointers into
  // the resource window — they were freed above by ccoWindowDeregister +
  // ccoMemFree, so no separate hipFree needed.
  if (ibgda.endpoints) HIP_RUNTIME_CHECK(hipFree(ibgda.endpoints));

  // SDMA pool cleanup. peerSignalPtrs is a GPU array of host-mapped peer
  // pointers — only the cross-process entries came from hipIpcOpenMemHandle
  // and need a matching close; same-process entries are raw VAs into a peer
  // thread's signalBuf and must NOT be passed to hipIpcCloseMemHandle.
  auto& sdma = hostShadow.sdma;
  if (sdma.peerSignalPtrs) {
    std::vector<HSAuint64*> peerPtrs_host(hostShadow.lsaSize, nullptr);
    HIP_RUNTIME_CHECK(hipMemcpy(peerPtrs_host.data(), sdma.peerSignalPtrs,
                                sizeof(HSAuint64*) * hostShadow.lsaSize, hipMemcpyDeviceToHost));
    for (int lsa = 0; lsa < hostShadow.lsaSize; lsa++) {
      if (lsa == hostShadow.lsaRank) continue;
      if (!peerPtrs_host[lsa]) continue;
      int pe = hostShadow.myNodeStart + lsa;
      if (comm && comm->ctx && comm->ctx->SameProcessP2P(pe)) continue;
      (void)hipIpcCloseMemHandle(peerPtrs_host[lsa]);
    }
    HIP_RUNTIME_CHECK(hipFree(sdma.peerSignalPtrs));
  }
  if (sdma.signalBuf) HIP_RUNTIME_CHECK(hipFree(sdma.signalBuf));
  if (sdma.expectSignals) HIP_RUNTIME_CHECK(hipFree(sdma.expectSignals));

  ccoWindowTableNode* node = hostShadow.windowTable;
  while (node) {
    ccoWindowTableNode nodeHost;
    HIP_RUNTIME_CHECK(
        hipMemcpy(&nodeHost, node, sizeof(ccoWindowTableNode), hipMemcpyDeviceToHost));
    HIP_RUNTIME_CHECK(hipFree(node));
    node = nodeHost.next;
  }

  return 0;
}

/* ========================================================================== */
/*                             ccoBarrierAll                               */
/* ========================================================================== */

int ccoBarrierAll(ccoComm* comm) {
  comm->bootNet->Barrier();
  return 0;
}

ccoDevComm* ccoDevCommCopyToDevice(const ccoDevComm* host) {
  ccoDevComm* device = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&device, sizeof(ccoDevComm)));
  HIP_RUNTIME_CHECK(hipMemcpy(device, host, sizeof(ccoDevComm), hipMemcpyHostToDevice));
  return device;
}

void ccoDevCommFreeDeviceCopy(ccoDevComm* devicePtr) {
  if (devicePtr) HIP_RUNTIME_CHECK(hipFree(devicePtr));
}

}  // namespace cco
}  // namespace mori
