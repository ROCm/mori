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

  // Step 2: context (RDMA endpoints, transport type negotiation)
  comm->ctx = new application::Context(*comm->bootNet);
  comm->defaultNumQpPerPe = comm->ctx->GetNumQpPerPe();

  // Step 2.5: detect intra-node topology (lsa = "local symmetric access").
  //
  // Consult Context's capability discovery directly (`PeerCapabilities.sameHost`)
  // rather than inferring sameHost from Context's chosen transport. This way
  // CCO stays correct even if env vars / policy decide that an intra-node peer
  // should use RDMA (e.g. CCO_GDA_CONNECTION_FULL): the peer is still on the
  // same node, lsa membership doesn't change.
  //
  // ASSUMPTION (matches mori shmem conventions): rank layout is contiguous
  // within each node — i.e. ranks 0..lsaSize-1 are on node 0,
  // lsaSize..2*lsaSize-1 are on node 1, etc. Validated below.
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
    comm->lsaSize = lsaCount;
    comm->myNodeStart = firstSameNode;
    comm->lsaRank = comm->rank - firstSameNode;

    // Sanity: contiguous block of [firstSameNode, lastSameNode]
    if (lastSameNode - firstSameNode + 1 != lsaCount) {
      MORI_SHMEM_WARN("CcoCommCreate: non-contiguous lsa membership detected "
                      "(first={} last={} count={}); team math assumes contiguous "
                      "node-major rank layout",
                      firstSameNode, lastSameNode, lsaCount);
    }
    MORI_SHMEM_INFO("CcoCommCreate: lsa topology rank={} lsaSize={} lsaRank={} "
                    "myNodeStart={}",
                    comm->rank, comm->lsaSize, comm->lsaRank, comm->myNodeStart);
  }

  // Step 3: reserve contiguous VA for flat address space
  // If user passes 0, default to GPU total memory.
  // Always align to 4GB so stride4G = perRankSize >> 32 is lossless (like NCCL).
  if (perRankVmmSize == 0) {
    size_t freeMem = 0, totalMem = 0;
    HIP_RUNTIME_CHECK(hipMemGetInfo(&freeMem, &totalMem));
    perRankVmmSize = totalMem;
  }
  perRankVmmSize = AlignUp(perRankVmmSize, 1ULL << 32);  // force 4GB alignment
  comm->perRankSize = perRankVmmSize;

  int currentDev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDev));

  // Query granularity with the SAME allocProp that MemAlloc will use,
  // including requestedHandleType — granularity may differ with FD export enabled
  hipMemAllocationProp allocProp = {};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = currentDev;

  size_t granularity = 0;
  HIP_RUNTIME_CHECK(
      hipMemGetAllocationGranularity(&granularity, &allocProp, hipMemAllocationGranularityMinimum));
  comm->vmmGranularity = granularity;

  size_t totalVaSize = static_cast<size_t>(comm->worldSize) * perRankVmmSize;
  HIP_RUNTIME_CHECK(hipMemAddressReserve(&comm->flatBase, totalVaSize, granularity, nullptr, 0));
  MORI_SHMEM_TRACE("CcoCommCreate: flatBase={} totalVA={} granularity={}", comm->flatBase,
                   totalVaSize, granularity);

  // Step 4: SDMA device handles (per-comm, shared across windows).
  //
  // capability AND policy: only materialize SDMA queues if the user opted in
  // via MORI_ENABLE_SDMA (policy) and at least one peer has the SDMA hardware
  // available (capability). Pure P2P deployments (the default) skip this
  // entirely, leaving comm->sdmaNumQueue == 0.
  bool anySdmaCapable = false;
  for (int pe = 0; pe < comm->worldSize; pe++) {
    if (comm->ctx->GetPeerCapabilities(pe).canSDMA) {
      anySdmaCapable = true;
      break;
    }
  }
  if (comm->ctx->IsSdmaEnabled() && anySdmaCapable) {
    comm->sdmaNumQueue = anvil::GetSdmaNumChannels();
    comm->ctx->EnsureSdmaTransport();  // lazy anvil.init + connect + EnablePeerAccess

    int srcDeviceId = currentDev;
    size_t numSlots = static_cast<size_t>(comm->worldSize) * comm->sdmaNumQueue;
    HIP_RUNTIME_CHECK(
        hipMalloc(&comm->sdmaDevHandles, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));
    HIP_RUNTIME_CHECK(
        hipMemset(comm->sdmaDevHandles, 0, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));

    for (int pe = 0; pe < comm->worldSize; pe++) {
      if (!comm->ctx->GetPeerCapabilities(pe).canSDMA) continue;
      int dstDeviceId = pe % 8;
      for (int q = 0; q < comm->sdmaNumQueue; q++) {
        auto* handle = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();
        HIP_RUNTIME_CHECK(hipMemcpy(
            &comm->sdmaDevHandles[dstDeviceId * comm->sdmaNumQueue + q],
            &handle, sizeof(handle), hipMemcpyHostToDevice));
      }
    }
  } else {
    comm->sdmaNumQueue = 0;  // canonicalize: SDMA not enabled ⇒ no queues
  }

  // Step 5: internal sync for device barriers
  HIP_RUNTIME_CHECK(hipMalloc(&comm->internalSyncGpuPtr, INTERNAL_SYNC_BYTES));
  HIP_RUNTIME_CHECK(hipMemset(comm->internalSyncGpuPtr, 0, INTERNAL_SYNC_BYTES));

  // Note: per-DevComm RDMA QP endpoints are no longer pre-allocated here.
  // They are created lazily in CcoDevCommCreate via ctx->CreateAdditionalEndpoints,
  // sized by CcoDevCommRequirements::gdaContextCount. This lets multiple
  // independent DevComms coexist on the same Comm with different QP counts.

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

  // Free remaining windows
  for (auto* wh : comm->windows) {
    // Caller should have called WindowDeregister; clean up stragglers
    delete wh;
  }
  comm->windows.clear();

  // Free remaining allocations
  for (auto& [ptr, meta] : comm->allocTable) {
    if (meta.shareFd >= 0) {
      close(meta.shareFd);
    }
  }
  comm->allocTable.clear();

  // SDMA device handles
  if (comm->sdmaDevHandles) {
    HIP_RUNTIME_CHECK(hipFree(comm->sdmaDevHandles));
  }

  // Internal sync
  if (comm->internalSyncGpuPtr) {
    HIP_RUNTIME_CHECK(hipFree(comm->internalSyncGpuPtr));
  }

  // Release VA space
  if (comm->flatBase) {
    size_t totalVaSize = static_cast<size_t>(comm->worldSize) * comm->perRankSize;
    HIP_RUNTIME_CHECK(hipMemAddressFree(comm->flatBase, totalVaSize));
  }

  // Context + bootstrap
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
  size_t slotOffset = comm->nextOffset;

  MORI_SHMEM_TRACE("CcoMemAlloc: rank={} size={} alignedSize={} slotOffset={}", comm->rank,
                   size, alignedSize, slotOffset);

  // Step 1: create physical memory
  hipMemAllocationProp allocProp = {};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = currentDev;

  hipMemGenericAllocationHandle_t physHandle;
  HIP_RUNTIME_CHECK(hipMemCreate(&physHandle, alignedSize, &allocProp, 0));

  // Step 2: map to local slot only (no cross-rank communication)
  void* localVa =
      static_cast<char*>(comm->flatBase) + static_cast<size_t>(comm->rank) * comm->perRankSize + slotOffset;
  HIP_RUNTIME_CHECK(hipMemMap(localVa, alignedSize, 0, physHandle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = currentDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_RUNTIME_CHECK(hipMemSetAccess(localVa, alignedSize, &accessDesc, 1));

  // Step 3: export dma-buf FD (for later use by WindowRegister: P2P + RDMA MR)
  int shareFd = -1;
  HIP_RUNTIME_CHECK(hipMemExportToShareableHandle(
      reinterpret_cast<void*>(&shareFd), physHandle, hipMemHandleTypePosixFileDescriptor, 0));

  // Step 4: advance offset and record metadata
  comm->nextOffset += alignedSize;

  CcoComm::AllocMeta meta;
  meta.physHandle = physHandle;
  meta.shareFd = shareFd;
  meta.slotOffset = slotOffset;
  meta.size = alignedSize;
  comm->allocTable[localVa] = meta;

  *outPtr = localVa;
  MORI_SHMEM_TRACE("CcoMemAlloc: done, localPtr={} (local only, P2P mapping deferred to WindowRegister)",
                   localVa);
  return 0;
}

/* ========================================================================== */
/*                              CcoMemFree                                 */
/* ========================================================================== */

int CcoMemFree(CcoComm* comm, void* ptr) {
  auto it = comm->allocTable.find(ptr);
  if (it == comm->allocTable.end()) {
    MORI_SHMEM_WARN("CcoMemFree: ptr {} not found", ptr);
    return -1;
  }

  auto& meta = it->second;
  size_t alignedSize = meta.size;
  size_t slotOffset = meta.slotOffset;

  MORI_SHMEM_TRACE("CcoMemFree: rank={} ptr={} size={}", comm->rank, ptr, alignedSize);

  int currentDev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDev));

  // Unmap peer slots
  for (int pe = 0; pe < comm->worldSize; pe++) {
    if (pe == comm->rank) continue;
    if (!comm->ctx->CanUseP2P(pe)) continue;

    void* peerVa = static_cast<char*>(comm->flatBase) +
                   static_cast<size_t>(pe) * comm->perRankSize + slotOffset;
    hipError_t err = hipMemUnmap(peerVa, alignedSize);
    if (err != hipSuccess) {
      MORI_SHMEM_WARN("CcoMemFree: unmap PE {} failed: {}", pe, err);
    }
  }

  // Unmap local slot
  HIP_RUNTIME_CHECK(hipMemUnmap(ptr, alignedSize));
  HIP_RUNTIME_CHECK(hipMemRelease(meta.physHandle));

  if (meta.shareFd >= 0) {
    close(meta.shareFd);
  }

  comm->allocTable.erase(it);
  return 0;
}

/* ========================================================================== */
/*                            CcoDevCommCreate                             */
/* ========================================================================== */

int CcoDevCommCreate(CcoComm* comm,
                     const CcoDevCommRequirements* reqs,
                     CcoDevComm** outDevComm) {
  MORI_SHMEM_TRACE("CcoDevCommCreate: rank={}", comm->rank);

  // ── forward-compat: validate reqs ──
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

  // Sanitize / log connection type
  CcoGdaConnectionType connType = reqs->gdaConnectionType;
  if (connType == CCO_GDA_CONNECTION_FULL ||
      connType == CCO_GDA_CONNECTION_RAIL) {
    MORI_SHMEM_WARN("CcoDevCommCreate: gdaConnectionType={} (FULL/RAIL) not yet "
                    "implemented; falling back to CROSSNODE semantics. "
                    "QP allocation is driven by Context::CreateAdditionalEndpoints "
                    "which already skips P2P-reachable peers.",
                    static_cast<int>(connType));
    connType = CCO_GDA_CONNECTION_CROSSNODE;
  }
  // Auto-downgrade CROSSNODE to NONE on single-node deployments
  if (connType == CCO_GDA_CONNECTION_CROSSNODE && comm->lsaSize == comm->worldSize) {
    MORI_SHMEM_TRACE("CcoDevCommCreate: single-node deployment, downgrading "
                     "CROSSNODE -> NONE (no cross-node peers)");
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

  // ── IBGDA Context: create fresh QP set per DevComm ──
  CcoIbgdaContext& ibgda = hostShadow.ibgda;
  int numQpPerPe = reqs->gdaContextCount > 0 ? reqs->gdaContextCount
                                              : comm->defaultNumQpPerPe;
  ibgda.numQpPerPe = numQpPerPe;

  size_t numEps = static_cast<size_t>(comm->worldSize) * numQpPerPe;
  shmem::ShmemRdmaEndpoint* epsGpu = nullptr;

  if (connType != CCO_GDA_CONNECTION_NONE && comm->ctx->RdmaTransportEnabled()) {
    // Create and connect fresh QPs (collective: all ranks must call together).
    // Context internally only allocates QPs for transportType==RDMA peers,
    // which corresponds to cross-node — this is the CROSSNODE behavior.
    auto newEps = comm->ctx->CreateAdditionalEndpoints(numQpPerPe);
    comm->ctx->ConnectAdditionalEndpoints(newEps, numQpPerPe);

    // Convert to ShmemRdmaEndpoint format
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

  // Build window table linked list on GPU
  const auto& tableEntries = comm->windowTableEntries;
  size_t numWindows = tableEntries.size();
  size_t numNodes =
      (numWindows + CCO_WINDOW_TABLE_SIZE - 1) / CCO_WINDOW_TABLE_SIZE;
  if (numNodes == 0) numNodes = 1;

  // Allocate all nodes on GPU, build from host
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

  // ── IBGDA Context: Signal / Counter buffers ──
  // Only allocate when GDA is enabled (NONE => skip).
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

  // Register signalBuf as RDMA MR and exchange rkeys
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

  // Copy struct to GPU
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

  // Free IBGDA context resources
  auto& ibgda = hostShadow.ibgda;
  if (ibgda.endpoints) HIP_RUNTIME_CHECK(hipFree(ibgda.endpoints));
  if (ibgda.signalBuf) HIP_RUNTIME_CHECK(hipFree(ibgda.signalBuf));
  if (ibgda.signalShadows) HIP_RUNTIME_CHECK(hipFree(ibgda.signalShadows));
  if (ibgda.counterBuf) HIP_RUNTIME_CHECK(hipFree(ibgda.counterBuf));
  if (ibgda.peerSignalRkeys) HIP_RUNTIME_CHECK(hipFree(ibgda.peerSignalRkeys));

  // Free window table linked list
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
