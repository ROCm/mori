// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#include "mori/xshmem/xshmem_api.hpp"

#include <cstring>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace xshmem {

static constexpr size_t INTERNAL_SYNC_COUNT = 128;
static constexpr size_t INTERNAL_SYNC_BYTES = INTERNAL_SYNC_COUNT * sizeof(uint64_t);

static size_t AlignUp(size_t x, size_t align) {
  return (x + align - 1) & ~(align - 1);
}

/* ========================================================================== */
/*                              XshmemCommCreate                              */
/* ========================================================================== */

int XshmemCommCreate(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                     XshmemComm** outComm) {
  auto* comm = new XshmemComm();
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

  MORI_SHMEM_TRACE("XshmemCommCreate: rank={} worldSize={} groupId={}", comm->rank,
                   comm->worldSize, comm->groupId);

  // Step 2: context (RDMA endpoints, transport type negotiation)
  comm->ctx = new application::Context(*comm->bootNet);
  comm->numQpPerPe = comm->ctx->GetNumQpPerPe();

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
  MORI_SHMEM_TRACE("XshmemCommCreate: flatBase={} totalVA={} granularity={}", comm->flatBase,
                   totalVaSize, granularity);

  // Step 4: SDMA device handles (per-comm, shared across windows)
  comm->sdmaNumQueue = anvil::GetSdmaNumChannels();
  if (comm->sdmaNumQueue > 0) {
    int srcDeviceId = currentDev;
    size_t numSlots = static_cast<size_t>(comm->worldSize) * comm->sdmaNumQueue;
    HIP_RUNTIME_CHECK(
        hipMalloc(&comm->sdmaDevHandles, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));
    HIP_RUNTIME_CHECK(
        hipMemset(comm->sdmaDevHandles, 0, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));

    for (int pe = 0; pe < comm->worldSize; pe++) {
      if (comm->ctx->GetTransportType(pe) != application::TransportType::SDMA) continue;
      int dstDeviceId = pe % 8;
      for (int q = 0; q < comm->sdmaNumQueue; q++) {
        auto* handle = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();
        HIP_RUNTIME_CHECK(hipMemcpy(
            &comm->sdmaDevHandles[dstDeviceId * comm->sdmaNumQueue + q],
            &handle, sizeof(handle), hipMemcpyHostToDevice));
      }
    }
  }

  // Step 5: internal sync for device barriers
  HIP_RUNTIME_CHECK(hipMalloc(&comm->internalSyncGpuPtr, INTERNAL_SYNC_BYTES));
  HIP_RUNTIME_CHECK(hipMemset(comm->internalSyncGpuPtr, 0, INTERNAL_SYNC_BYTES));

  // Step 6: RDMA endpoints
  if (comm->ctx->RdmaTransportEnabled()) {
    const auto& hostEps = comm->ctx->GetRdmaEndpoints();
    size_t numEps = static_cast<size_t>(comm->worldSize) * comm->numQpPerPe;
    comm->rdmaEndpoints.resize(numEps);
    for (size_t i = 0; i < numEps; i++) {
      comm->rdmaEndpoints[i].vendorId = hostEps[i].vendorId;
      comm->rdmaEndpoints[i].qpn = hostEps[i].handle.qpn;
      comm->rdmaEndpoints[i].wqHandle = hostEps[i].wqHandle;
      comm->rdmaEndpoints[i].cqHandle = hostEps[i].cqHandle;
      comm->rdmaEndpoints[i].atomicIbuf = hostEps[i].atomicIbuf;
    }
  }

  MORI_SHMEM_INFO("XshmemCommCreate: rank={}/{} groupId={} flatBase={} perRankSize={} "
                  "granularity={} numQpPerPe={} sdmaNumQueue={} rdma={}",
                  comm->rank, comm->worldSize, comm->groupId, comm->flatBase, comm->perRankSize,
                  comm->vmmGranularity, comm->numQpPerPe, comm->sdmaNumQueue,
                  comm->ctx->RdmaTransportEnabled());
  if (!comm->rdmaEndpoints.empty()) {
    for (int pe = 0; pe < comm->worldSize; pe++) {
      for (int qp = 0; qp < comm->numQpPerPe; qp++) {
        auto& ep = comm->rdmaEndpoints[pe * comm->numQpPerPe + qp];
        MORI_SHMEM_INFO("  QP[pe={},qp={}]: vendor={:#x} qpn={}", pe, qp,
                        static_cast<uint32_t>(ep.vendorId), ep.qpn);
      }
    }
  }
  return 0;
}

/* ========================================================================== */
/*                             XshmemCommDestroy                              */
/* ========================================================================== */

int XshmemCommDestroy(XshmemComm* comm) {
  if (!comm) return 0;

  MORI_SHMEM_TRACE("XshmemCommDestroy: rank={}", comm->rank);

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
/*                              XshmemMemAlloc                                */
/* ========================================================================== */

int XshmemMemAlloc(XshmemComm* comm, size_t size, void** outPtr) {
  int currentDev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDev));

  size_t alignedSize = AlignUp(size, comm->vmmGranularity);
  size_t slotOffset = comm->nextOffset;

  MORI_SHMEM_TRACE("XshmemMemAlloc: rank={} size={} alignedSize={} slotOffset={}", comm->rank,
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

  XshmemComm::AllocMeta meta;
  meta.physHandle = physHandle;
  meta.shareFd = shareFd;
  meta.slotOffset = slotOffset;
  meta.size = alignedSize;
  comm->allocTable[localVa] = meta;

  *outPtr = localVa;
  MORI_SHMEM_TRACE("XshmemMemAlloc: done, localPtr={} (local only, P2P mapping deferred to WindowRegister)",
                   localVa);
  return 0;
}

/* ========================================================================== */
/*                              XshmemMemFree                                 */
/* ========================================================================== */

int XshmemMemFree(XshmemComm* comm, void* ptr) {
  auto it = comm->allocTable.find(ptr);
  if (it == comm->allocTable.end()) {
    MORI_SHMEM_WARN("XshmemMemFree: ptr {} not found", ptr);
    return -1;
  }

  auto& meta = it->second;
  size_t alignedSize = meta.size;
  size_t slotOffset = meta.slotOffset;

  MORI_SHMEM_TRACE("XshmemMemFree: rank={} ptr={} size={}", comm->rank, ptr, alignedSize);

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
      MORI_SHMEM_WARN("XshmemMemFree: unmap PE {} failed: {}", pe, err);
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
/*                            XshmemDevCommCreate                             */
/* ========================================================================== */

int XshmemDevCommCreate(XshmemComm* comm, XshmemDevComm** outDevComm) {
  MORI_SHMEM_TRACE("XshmemDevCommCreate: rank={}", comm->rank);

  XshmemDevComm hostShadow = {};
  hostShadow.rank = comm->rank;
  hostShadow.worldSize = comm->worldSize;
  hostShadow.flatBase = comm->flatBase;
  hostShadow.perRankSize = comm->perRankSize;
  hostShadow.internalSyncPtr = comm->internalSyncGpuPtr;

  // ── IBGDA Context: create fresh QP set (independent from previous DevComms) ──
  XshmemIbgdaContext& ibgda = hostShadow.ibgda;
  ibgda.numQpPerPe = comm->numQpPerPe;

  size_t numEps = static_cast<size_t>(comm->worldSize) * comm->numQpPerPe;
  shmem::ShmemRdmaEndpoint* epsGpu = nullptr;

  if (comm->ctx->RdmaTransportEnabled()) {
    // Create and connect fresh QPs (collective: all ranks must call together)
    auto newEps = comm->ctx->CreateAdditionalEndpoints(comm->numQpPerPe);
    comm->ctx->ConnectAdditionalEndpoints(newEps, comm->numQpPerPe);

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
      (numWindows + XSHMEM_WINDOW_TABLE_SIZE - 1) / XSHMEM_WINDOW_TABLE_SIZE;
  if (numNodes == 0) numNodes = 1;

  // Allocate all nodes on GPU, build from host
  std::vector<XshmemWindowTableNode*> gpuNodes(numNodes, nullptr);
  for (size_t n = 0; n < numNodes; n++) {
    HIP_RUNTIME_CHECK(hipMalloc(&gpuNodes[n], sizeof(XshmemWindowTableNode)));
    HIP_RUNTIME_CHECK(hipMemset(gpuNodes[n], 0, sizeof(XshmemWindowTableNode)));
  }

  for (size_t n = 0; n < numNodes; n++) {
    XshmemWindowTableNode nodeHost = {};
    size_t base = n * XSHMEM_WINDOW_TABLE_SIZE;
    for (int i = 0; i < XSHMEM_WINDOW_TABLE_SIZE; i++) {
      size_t idx = base + i;
      if (idx < numWindows) {
        nodeHost.entries[i].base = tableEntries[idx].base;
        nodeHost.entries[i].size = tableEntries[idx].size;
        nodeHost.entries[i].window = tableEntries[idx].devPtr;
      }
    }
    nodeHost.next = (n + 1 < numNodes) ? gpuNodes[n + 1] : nullptr;
    HIP_RUNTIME_CHECK(
        hipMemcpy(gpuNodes[n], &nodeHost, sizeof(XshmemWindowTableNode), hipMemcpyHostToDevice));
  }
  hostShadow.windowTable = gpuNodes[0];

  MORI_SHMEM_TRACE("XshmemDevCommCreate: windowTable with {} windows in {} nodes", numWindows,
                   numNodes);

  // ── IBGDA Context: Signal / Counter buffers ──
  int signalCount = comm->signalCount;
  int counterCount = comm->counterCount;
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

  MORI_SHMEM_TRACE("XshmemDevCommCreate: signals={} counters={} signalLkey={}", signalCount,
                   counterCount, signalLkey);

  // Copy struct to GPU
  XshmemDevComm* devCommGpu = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&devCommGpu, sizeof(XshmemDevComm)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(devCommGpu, &hostShadow, sizeof(XshmemDevComm), hipMemcpyHostToDevice));

  *outDevComm = devCommGpu;
  MORI_SHMEM_INFO("XshmemDevCommCreate: rank={} devComm={} windows={} signals={} counters={} "
                  "signalBuf={} counterBuf={} signalLkey={}",
                  comm->rank, (void*)devCommGpu, numWindows, signalCount, counterCount,
                  (void*)signalBufGpu, (void*)counterBufGpu, signalLkey);
  return 0;
}

/* ========================================================================== */
/*                           XshmemDevCommDestroy                             */
/* ========================================================================== */

int XshmemDevCommDestroy(XshmemDevComm* devComm) {
  if (!devComm) return 0;

  XshmemDevComm hostShadow;
  HIP_RUNTIME_CHECK(
      hipMemcpy(&hostShadow, devComm, sizeof(XshmemDevComm), hipMemcpyDeviceToHost));

  // Free IBGDA context resources
  auto& ibgda = hostShadow.ibgda;
  if (ibgda.endpoints) HIP_RUNTIME_CHECK(hipFree(ibgda.endpoints));
  if (ibgda.signalBuf) HIP_RUNTIME_CHECK(hipFree(ibgda.signalBuf));
  if (ibgda.signalShadows) HIP_RUNTIME_CHECK(hipFree(ibgda.signalShadows));
  if (ibgda.counterBuf) HIP_RUNTIME_CHECK(hipFree(ibgda.counterBuf));
  if (ibgda.peerSignalRkeys) HIP_RUNTIME_CHECK(hipFree(ibgda.peerSignalRkeys));

  // Free window table linked list
  XshmemWindowTableNode* node = hostShadow.windowTable;
  while (node) {
    XshmemWindowTableNode nodeHost;
    HIP_RUNTIME_CHECK(
        hipMemcpy(&nodeHost, node, sizeof(XshmemWindowTableNode), hipMemcpyDeviceToHost));
    HIP_RUNTIME_CHECK(hipFree(node));
    node = nodeHost.next;
  }

  HIP_RUNTIME_CHECK(hipFree(devComm));
  return 0;
}

/* ========================================================================== */
/*                             XshmemBarrierAll                               */
/* ========================================================================== */

int XshmemBarrierAll(XshmemComm* comm) {
  comm->bootNet->Barrier();
  return 0;
}

}  // namespace xshmem
}  // namespace mori
