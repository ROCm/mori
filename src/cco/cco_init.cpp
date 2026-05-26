// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#include "mori/cco/cco_api.hpp"

#include <cstring>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace cco {

static constexpr size_t INTERNAL_SYNC_COUNT = 128;
static constexpr size_t INTERNAL_SYNC_BYTES = INTERNAL_SYNC_COUNT * sizeof(uint64_t);

static size_t AlignUp(size_t x, size_t align) {
  return (x + align - 1) & ~(align - 1);
}

/* ========================================================================== */
/*                              CcoCommCreate                              */
/* ========================================================================== */

int CcoCommCreate(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                     CcoComm** outComm) {
  auto* comm = new CcoComm();
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

  MORI_SHMEM_TRACE("CcoCommCreate: rank={} worldSize={} groupId={}", comm->rank,
                   comm->worldSize, comm->groupId);

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
          "CcoCommCreate: non-contiguous lsa membership "
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
            "CcoCommCreate: heterogeneous lsa sizes detected "
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

    MORI_SHMEM_INFO(
        "CcoCommCreate: lsa topology rank={} lsaSize={} lsaRank={} myNodeStart={}",
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

  int currentDev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDev));

  // Query granularity with the SAME allocProp MemAlloc will use — granularity
  // can shift when requestedHandleType (FD export) is enabled.
  hipMemAllocationProp allocProp = {};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = currentDev;

  size_t granularity = 0;
  HIP_RUNTIME_CHECK(
      hipMemGetAllocationGranularity(&granularity, &allocProp, hipMemAllocationGranularityMinimum));
  comm->vmmGranularity = granularity;

  // Flat VA covers the LSA team only. Cross-node peers don't use VA — RDMA
  // goes through iova=0 + offset.
  size_t totalVaSize = static_cast<size_t>(comm->lsaSize) * perRankVmmSize;
  HIP_RUNTIME_CHECK(hipMemAddressReserve(&comm->flatBase, totalVaSize, granularity, nullptr, 0));
  MORI_SHMEM_TRACE("CcoCommCreate: flatBase={} totalVA={} (lsaSize={} x perRankSize={}) granularity={}",
                   comm->flatBase, totalVaSize, comm->lsaSize, perRankVmmSize, granularity);

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
    int srcDeviceId = currentDev;
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
        HIP_RUNTIME_CHECK(hipMemcpy(
            &comm->sdmaDevHandles[lsa * comm->sdmaNumQueue + q],
            &handle, sizeof(handle), hipMemcpyHostToDevice));
      }
    }
  } else {
    comm->sdmaNumQueue = 0;
  }

  // Step 5: device-barrier scratch.
  HIP_RUNTIME_CHECK(hipMalloc(&comm->internalSyncGpuPtr, INTERNAL_SYNC_BYTES));
  HIP_RUNTIME_CHECK(hipMemset(comm->internalSyncGpuPtr, 0, INTERNAL_SYNC_BYTES));

  // RDMA QP endpoints are NOT pre-allocated here. CcoDevCommCreate builds
  // a fresh QP set per DevComm via ctx->CreateAdditionalEndpoints, sized by
  // reqs.gdaContextCount, so multiple DevComms can coexist with independent
  // QP state.

  MORI_SHMEM_INFO("CcoCommCreate: rank={}/{} groupId={} flatBase={} perRankSize={} "
                  "granularity={} defaultNumQpPerPe={} sdmaNumQueue={} rdma={}",
                  comm->rank, comm->worldSize, comm->groupId, comm->flatBase, comm->perRankSize,
                  comm->vmmGranularity, comm->defaultNumQpPerPe, comm->sdmaNumQueue,
                  comm->ctx->RdmaTransportEnabled());
  return 0;
}

/* ========================================================================== */
/*                             CcoCommDestroy                              */
/* ========================================================================== */

int CcoCommDestroy(CcoComm* comm) {
  if (!comm) return 0;

  MORI_SHMEM_TRACE("CcoCommDestroy: rank={}", comm->rank);

  for (auto* wh : comm->windows) delete wh;
  comm->windows.clear();

  for (auto& [ptr, meta] : comm->allocTable) {
    if (meta.shareFd >= 0) close(meta.shareFd);
  }
  comm->allocTable.clear();

  if (comm->sdmaDevHandles) HIP_RUNTIME_CHECK(hipFree(comm->sdmaDevHandles));
  if (comm->internalSyncGpuPtr) HIP_RUNTIME_CHECK(hipFree(comm->internalSyncGpuPtr));

  // Release flat VA — sized to match the reservation in CcoCommCreate.
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
/*                              CcoMemAlloc                                */
/* ========================================================================== */

int CcoMemAlloc(CcoComm* comm, size_t size, void** outPtr) {
  int currentDev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDev));

  size_t alignedSize = AlignUp(size, comm->vmmGranularity);

  // Reserve a slot via bump pointer; rolled back below if any later step fails.
  size_t slotOffset;
  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    slotOffset = comm->nextOffset;
    if (slotOffset + alignedSize > comm->perRankSize) {
      MORI_SHMEM_ERROR(
          "CcoMemAlloc: slot exhausted (offset {} + size {} > perRankSize {}). "
          "Increase perRankVmmSize at CcoCommCreate or reduce window count.",
          slotOffset, alignedSize, comm->perRankSize);
      return -1;
    }
    comm->nextOffset += alignedSize;
  }

  MORI_SHMEM_TRACE("CcoMemAlloc: rank={} size={} alignedSize={} slotOffset={}", comm->rank,
                   size, alignedSize, slotOffset);

  // Best-effort slot rollback: only undoes the bump if no other thread has
  // moved past us. Holes are leaked until comm destruction (no freelist yet).
  auto rollbackSlot = [&]() {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    if (comm->nextOffset == slotOffset + alignedSize) {
      comm->nextOffset = slotOffset;
    }
  };

  hipMemAllocationProp allocProp = {};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = currentDev;

  hipMemGenericAllocationHandle_t physHandle = 0;
  hipError_t err = hipMemCreate(&physHandle, alignedSize, &allocProp, 0);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("CcoMemAlloc: hipMemCreate failed: {} ({})", static_cast<int>(err),
                     hipGetErrorString(err));
    rollbackSlot();
    return -1;
  }

  // Map only the local slot. Peer slots are mapped lazily in WindowRegister.
  void* localVa =
      static_cast<char*>(comm->flatBase) + static_cast<size_t>(comm->lsaRank) * comm->perRankSize + slotOffset;
  err = hipMemMap(localVa, alignedSize, 0, physHandle, 0);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("CcoMemAlloc: hipMemMap failed: {} ({})", static_cast<int>(err),
                     hipGetErrorString(err));
    (void)hipMemRelease(physHandle);
    rollbackSlot();
    return -1;
  }

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = currentDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  err = hipMemSetAccess(localVa, alignedSize, &accessDesc, 1);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("CcoMemAlloc: hipMemSetAccess failed: {} ({})", static_cast<int>(err),
                     hipGetErrorString(err));
    (void)hipMemUnmap(localVa, alignedSize);
    (void)hipMemRelease(physHandle);
    rollbackSlot();
    return -1;
  }

  // dma-buf FD is stashed for WindowRegister to share (P2P FD exchange + RDMA MR).
  int shareFd = -1;
  err = hipMemExportToShareableHandle(
      reinterpret_cast<void*>(&shareFd), physHandle, hipMemHandleTypePosixFileDescriptor, 0);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("CcoMemAlloc: hipMemExportToShareableHandle failed: {} ({})",
                     static_cast<int>(err), hipGetErrorString(err));
    (void)hipMemUnmap(localVa, alignedSize);
    (void)hipMemRelease(physHandle);
    rollbackSlot();
    return -1;
  }

  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    CcoComm::AllocMeta meta;
    meta.physHandle = physHandle;
    meta.shareFd = shareFd;
    meta.slotOffset = slotOffset;
    meta.size = alignedSize;
    comm->allocTable[localVa] = meta;
  }

  *outPtr = localVa;
  MORI_SHMEM_TRACE("CcoMemAlloc: done, localPtr={}", localVa);
  return 0;
}

/* ========================================================================== */
/*                              CcoMemFree                                 */
/* ========================================================================== */

int CcoMemFree(CcoComm* comm, void* ptr) {
  // Snapshot meta under the lock, then drop it before the (potentially slow)
  // hipMem* calls so concurrent MemAlloc isn't blocked.
  CcoComm::AllocMeta meta;
  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    auto it = comm->allocTable.find(ptr);
    if (it == comm->allocTable.end()) {
      MORI_SHMEM_WARN("CcoMemFree: ptr {} not found", ptr);
      return -1;
    }
    meta = it->second;
    comm->allocTable.erase(it);
  }

  size_t alignedSize = meta.size;
  size_t slotOffset = meta.slotOffset;

  MORI_SHMEM_TRACE("CcoMemFree: rank={} ptr={} size={}", comm->rank, ptr, alignedSize);

  // Unmap peer slots that WindowRegister mapped. ENOMAP for never-registered
  // windows is expected and ignored.
  for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
    if (lsa == comm->lsaRank) continue;
    int pe = comm->myNodeStart + lsa;
    if (!comm->ctx->CanUseP2P(pe)) continue;

    void* peerVa = static_cast<char*>(comm->flatBase) +
                   static_cast<size_t>(lsa) * comm->perRankSize + slotOffset;
    hipError_t err = hipMemUnmap(peerVa, alignedSize);
    if (err != hipSuccess) {
      MORI_SHMEM_WARN("CcoMemFree: unmap PE {} (lsa={}) failed: {}",
                      pe, lsa, static_cast<int>(err));
    }
  }

  hipError_t err = hipMemUnmap(ptr, alignedSize);
  if (err != hipSuccess) {
    MORI_SHMEM_WARN("CcoMemFree: local hipMemUnmap failed: {} ({})", static_cast<int>(err),
                    hipGetErrorString(err));
  }
  err = hipMemRelease(meta.physHandle);
  if (err != hipSuccess) {
    MORI_SHMEM_WARN("CcoMemFree: hipMemRelease failed: {} ({})", static_cast<int>(err),
                    hipGetErrorString(err));
  }

  if (meta.shareFd >= 0) close(meta.shareFd);

  // slotOffset is not returned to nextOffset — bump pointer only. Long-running
  // alloc/free churn will eventually exhaust perRankSize; a segmented
  // allocator is future work.
  return 0;
}

/* ========================================================================== */
/*                            CcoDevCommCreate                             */
/* ========================================================================== */

int CcoDevCommCreate(CcoComm* comm,
                     const CcoDevCommRequirements* reqs,
                     CcoDevComm** outDevComm) {
  MORI_SHMEM_TRACE("CcoDevCommCreate: rank={}", comm->rank);

  // Forward-compat: validate {magic, version}.
  if (reqs == nullptr) {
    MORI_SHMEM_ERROR("CcoDevCommCreate: reqs is NULL — must initialize via "
                     "CCO_DEV_COMM_REQUIREMENTS_INITIALIZER");
    return -1;
  }
  if (reqs->magic != CCO_API_MAGIC) {
    MORI_SHMEM_ERROR("CcoDevCommCreate: reqs->magic mismatch (got {:#x}, expect {:#x}) — "
                     "must initialize via CCO_DEV_COMM_REQUIREMENTS_INITIALIZER",
                     reqs->magic, CCO_API_MAGIC);
    return -1;
  }
  if (reqs->version > CCO_API_VERSION) {
    MORI_SHMEM_WARN("CcoDevCommCreate: reqs->version={} > runtime CCO_API_VERSION={}",
                    reqs->version, CCO_API_VERSION);
  }

  // Resolve connection type. FULL/RAIL fall back to CROSSNODE (Context already
  // skips P2P-reachable peers). CROSSNODE collapses to NONE on single-node.
  CcoGdaConnectionType connType = reqs->gdaConnectionType;
  if (connType == CCO_GDA_CONNECTION_FULL ||
      connType == CCO_GDA_CONNECTION_RAIL) {
    MORI_SHMEM_WARN("CcoDevCommCreate: gdaConnectionType={} (FULL/RAIL) not "
                    "yet implemented; falling back to CROSSNODE.",
                    static_cast<int>(connType));
    connType = CCO_GDA_CONNECTION_CROSSNODE;
  }
  if (connType == CCO_GDA_CONNECTION_CROSSNODE && comm->lsaSize == comm->worldSize) {
    MORI_SHMEM_TRACE("CcoDevCommCreate: single-node, CROSSNODE -> NONE");
    connType = CCO_GDA_CONNECTION_NONE;
  }

  CcoDevComm hostShadow = {};
  hostShadow.rank = comm->rank;
  hostShadow.worldSize = comm->worldSize;
  hostShadow.lsaSize = comm->lsaSize;
  hostShadow.lsaRank = comm->lsaRank;
  hostShadow.myNodeStart = comm->myNodeStart;
  hostShadow.gdaConnType = connType;
  hostShadow.flatBase = comm->flatBase;
  hostShadow.perRankSize = comm->perRankSize;
  hostShadow.internalSyncPtr = comm->internalSyncGpuPtr;

  // Fresh QP set per DevComm.
  CcoIbgdaContext& ibgda = hostShadow.ibgda;
  int numQpPerPe = reqs->gdaContextCount > 0 ? reqs->gdaContextCount
                                              : comm->defaultNumQpPerPe;
  ibgda.numQpPerPe = numQpPerPe;

  size_t numEps = static_cast<size_t>(comm->worldSize) * numQpPerPe;
  shmem::ShmemRdmaEndpoint* epsGpu = nullptr;

  if (connType != CCO_GDA_CONNECTION_NONE && comm->ctx->RdmaTransportEnabled()) {
    // Collective: every rank must call CreateAdditionalEndpoints together.
    // Context skips QPs for non-RDMA peers, so intra-node slots stay empty.
    auto newEps = comm->ctx->CreateAdditionalEndpoints(numQpPerPe);
    comm->ctx->ConnectAdditionalEndpoints(newEps, numQpPerPe);

    std::vector<shmem::ShmemRdmaEndpoint> shmemEps(numEps);
    for (size_t i = 0; i < numEps; i++) {
      shmemEps[i].vendorId = newEps[i].vendorId;
      shmemEps[i].qpn = newEps[i].handle.qpn;
      shmemEps[i].wqHandle = newEps[i].wqHandle;
      shmemEps[i].cqHandle = newEps[i].cqHandle;
      shmemEps[i].atomicIbuf = newEps[i].atomicIbuf;
    }

    HIP_RUNTIME_CHECK(hipMalloc(&epsGpu, numEps * sizeof(shmem::ShmemRdmaEndpoint)));
    HIP_RUNTIME_CHECK(hipMemcpy(epsGpu, shmemEps.data(),
                                numEps * sizeof(shmem::ShmemRdmaEndpoint), hipMemcpyHostToDevice));
  }
  ibgda.endpoints = epsGpu;

  // Build window-table linked list on GPU.
  const auto& tableEntries = comm->windowTableEntries;
  size_t numWindows = tableEntries.size();
  size_t numNodes =
      (numWindows + CCO_WINDOW_TABLE_SIZE - 1) / CCO_WINDOW_TABLE_SIZE;
  if (numNodes == 0) numNodes = 1;

  std::vector<CcoWindowTableNode*> gpuNodes(numNodes, nullptr);
  for (size_t n = 0; n < numNodes; n++) {
    HIP_RUNTIME_CHECK(hipMalloc(&gpuNodes[n], sizeof(CcoWindowTableNode)));
    HIP_RUNTIME_CHECK(hipMemset(gpuNodes[n], 0, sizeof(CcoWindowTableNode)));
  }

  for (size_t n = 0; n < numNodes; n++) {
    CcoWindowTableNode nodeHost = {};
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
        hipMemcpy(gpuNodes[n], &nodeHost, sizeof(CcoWindowTableNode), hipMemcpyHostToDevice));
  }
  hostShadow.windowTable = gpuNodes[0];

  MORI_SHMEM_TRACE("CcoDevCommCreate: windowTable with {} windows in {} nodes", numWindows,
                   numNodes);

  // Signal + counter buffers (skipped when GDA is disabled).
  int signalCount = (connType == CCO_GDA_CONNECTION_NONE) ? 0 : reqs->gdaSignalCount;
  int counterCount = (connType == CCO_GDA_CONNECTION_NONE) ? 0 : reqs->gdaCounterCount;
  ibgda.signalCount = signalCount;
  ibgda.counterCount = counterCount;

  uint64_t* signalBufGpu = nullptr;
  uint64_t* signalShadowsGpu = nullptr;
  if (signalCount > 0) {
    size_t sigBytes = signalCount * sizeof(uint64_t);
    HIP_RUNTIME_CHECK(hipMalloc(&signalBufGpu, sigBytes));
    HIP_RUNTIME_CHECK(hipMemset(signalBufGpu, 0, sigBytes));
    HIP_RUNTIME_CHECK(hipMalloc(&signalShadowsGpu, sigBytes));
    HIP_RUNTIME_CHECK(hipMemset(signalShadowsGpu, 0, sigBytes));
  }
  ibgda.signalBuf = signalBufGpu;
  ibgda.signalShadows = signalShadowsGpu;

  uint64_t* counterBufGpu = nullptr;
  if (counterCount > 0) {
    size_t ctrBytes = counterCount * sizeof(uint64_t);
    HIP_RUNTIME_CHECK(hipMalloc(&counterBufGpu, ctrBytes));
    HIP_RUNTIME_CHECK(hipMemset(counterBufGpu, 0, ctrBytes));
  }
  ibgda.counterBuf = counterBufGpu;

  // Register signalBuf as an RDMA MR and Allgather rkeys.
  uint32_t signalLkey = 0;
  uint32_t localSignalRkey = 0;
  application::RdmaDeviceContext* rdmaDevCtx = comm->ctx->GetRdmaDeviceContext();
  if (rdmaDevCtx && signalBufGpu && signalCount > 0) {
    application::RdmaMemoryRegion mr =
        rdmaDevCtx->RegisterRdmaMemoryRegion(signalBufGpu, signalCount * sizeof(uint64_t));
    signalLkey = mr.lkey;
    localSignalRkey = mr.rkey;
  }
  ibgda.signalLkey = signalLkey;

  auto* peerSignalRkeys_host = static_cast<uint32_t*>(calloc(comm->worldSize, sizeof(uint32_t)));
  peerSignalRkeys_host[comm->rank] = localSignalRkey;
  comm->bootNet->Allgather(&localSignalRkey, peerSignalRkeys_host, sizeof(uint32_t));

  uint32_t* peerSignalRkeysGpu = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&peerSignalRkeysGpu, sizeof(uint32_t) * comm->worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(peerSignalRkeysGpu, peerSignalRkeys_host,
                              sizeof(uint32_t) * comm->worldSize, hipMemcpyHostToDevice));
  ibgda.peerSignalRkeys = peerSignalRkeysGpu;
  free(peerSignalRkeys_host);

  MORI_SHMEM_TRACE("CcoDevCommCreate: signals={} counters={} signalLkey={}", signalCount,
                   counterCount, signalLkey);

  CcoDevComm* devCommGpu = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&devCommGpu, sizeof(CcoDevComm)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(devCommGpu, &hostShadow, sizeof(CcoDevComm), hipMemcpyHostToDevice));

  *outDevComm = devCommGpu;
  MORI_SHMEM_INFO("CcoDevCommCreate: rank={} devComm={} windows={} signals={} counters={} "
                  "signalBuf={} counterBuf={} signalLkey={}",
                  comm->rank, (void*)devCommGpu, numWindows, signalCount, counterCount,
                  (void*)signalBufGpu, (void*)counterBufGpu, signalLkey);
  return 0;
}

/* ========================================================================== */
/*                           CcoDevCommDestroy                             */
/* ========================================================================== */

int CcoDevCommDestroy(CcoDevComm* devComm) {
  if (!devComm) return 0;

  CcoDevComm hostShadow;
  HIP_RUNTIME_CHECK(
      hipMemcpy(&hostShadow, devComm, sizeof(CcoDevComm), hipMemcpyDeviceToHost));

  auto& ibgda = hostShadow.ibgda;
  if (ibgda.endpoints) HIP_RUNTIME_CHECK(hipFree(ibgda.endpoints));
  if (ibgda.signalBuf) HIP_RUNTIME_CHECK(hipFree(ibgda.signalBuf));
  if (ibgda.signalShadows) HIP_RUNTIME_CHECK(hipFree(ibgda.signalShadows));
  if (ibgda.counterBuf) HIP_RUNTIME_CHECK(hipFree(ibgda.counterBuf));
  if (ibgda.peerSignalRkeys) HIP_RUNTIME_CHECK(hipFree(ibgda.peerSignalRkeys));

  CcoWindowTableNode* node = hostShadow.windowTable;
  while (node) {
    CcoWindowTableNode nodeHost;
    HIP_RUNTIME_CHECK(
        hipMemcpy(&nodeHost, node, sizeof(CcoWindowTableNode), hipMemcpyDeviceToHost));
    HIP_RUNTIME_CHECK(hipFree(node));
    node = nodeHost.next;
  }

  HIP_RUNTIME_CHECK(hipFree(devComm));
  return 0;
}

/* ========================================================================== */
/*                             CcoBarrierAll                               */
/* ========================================================================== */

int CcoBarrierAll(CcoComm* comm) {
  comm->bootNet->Barrier();
  return 0;
}

}  // namespace cco
}  // namespace mori
