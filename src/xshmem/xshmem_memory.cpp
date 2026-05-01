// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#include "mori/xshmem/xshmem_api.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace xshmem {

/* ========================================================================== */
/*                         XshmemWindowRegister (ptr)                         */
/* ========================================================================== */

int XshmemWindowRegister(XshmemComm* comm, void* ptr, size_t size, XshmemWindow_t* outWin) {
  auto it = comm->allocTable.find(ptr);
  if (it == comm->allocTable.end()) {
    MORI_SHMEM_ERROR("XshmemWindowRegister: ptr {} not in allocTable", ptr);
    return -1;
  }

  auto& meta = it->second;
  size_t slotOffset = meta.slotOffset;
  int shareFd = meta.shareFd;
  void* localPtr = ptr;
  int worldSize = comm->worldSize;
  int rank = comm->rank;

  size_t alignedSize = meta.size;

  MORI_SHMEM_TRACE("XshmemWindowRegister: rank={} ptr={} size={} slotOffset={}", rank, ptr, size,
                   slotOffset);

  int currentDev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDev));

  // ── P2P: exchange FDs with same-node peers and map their memory into flat VA ──
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
      if (sortedGroup[i] == rank) { myPeerRank = i; break; }
    }
    int p2pWorldSize = static_cast<int>(sortedGroup.size());

    // Socket path must be SAME across all ranks in this comm group,
    // but UNIQUE per window and per group to avoid collision.
    // groupId (rank 0's pid) identifies the group; slotOffset identifies the window.
    std::string socketPath = "/tmp/mori_xshmem_" + std::to_string(comm->groupId) + "_" +
                             std::to_string(slotOffset) + "_";

    // Clean up stale socket files from previous crashed runs (rank 0 only to avoid race)
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
      MORI_SHMEM_ERROR("XshmemWindowRegister: P2P FD exchange failed");
      localBoot.Finalize();
      return -1;
    }

    std::vector<int> globalToPeer(worldSize, -1);
    for (int i = 0; i < p2pWorldSize; i++) {
      globalToPeer[sortedGroup[i]] = i;
    }

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = currentDev;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    for (int pe : p2pPeers) {
      int pr = globalToPeer[pe];
      if (pr < 0 || pr >= static_cast<int>(allFds.size())) continue;
      int peerFd = allFds[pr][0];
      if (peerFd < 0) continue;

      hipMemGenericAllocationHandle_t importedHandle;
      hipError_t err = hipMemImportFromShareableHandleCompat(
          &importedHandle, peerFd, hipMemHandleTypePosixFileDescriptor);
      if (err != hipSuccess) {
        MORI_SHMEM_WARN("XshmemWindowRegister: import from PE {} failed: {}", pe, err);
        continue;
      }

      void* peerVa = static_cast<char*>(comm->flatBase) +
                     static_cast<size_t>(pe) * comm->perRankSize + slotOffset;
      HIP_RUNTIME_CHECK(hipMemMap(peerVa, alignedSize, 0, importedHandle, 0));

      // hipMemSetAccess can transiently fail under concurrent VMM operations (multi-thread)
      for (int retry = 0;; retry++) {
        hipError_t setErr = hipMemSetAccess(peerVa, alignedSize, &accessDesc, 1);
        if (setErr == hipSuccess) break;
        if (retry >= 5) { HIP_RUNTIME_CHECK(setErr); }
        usleep(1000 * (1 << retry));
      }
    }

    localBoot.Finalize();
  }

  // ── RDMA MR registration ──
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

  // Exchange rkeys (Allgather is implicitly synchronizing)
  auto* peerRkeys_host = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  peerRkeys_host[rank] = localRkey;
  comm->bootNet->Allgather(&localRkey, peerRkeys_host, sizeof(uint32_t));

  // ── SDMA signals ──
  int sdmaNumQueue = comm->sdmaNumQueue;
  size_t signalArraySize = static_cast<size_t>(worldSize) * sdmaNumQueue * sizeof(HSAuint64);

  HSAuint64* signalPtrs = nullptr;
  HSAuint64* expectSignalsPtr = nullptr;
  HSAuint64** peerSignalPtrs_host_arr = nullptr;
  HSAuint64** peerSignalPtrs_gpu = nullptr;

  // Only allocate SDMA signals if there are SDMA peers
  bool hasSdmaPeers = false;
  for (int pe = 0; pe < worldSize; pe++) {
    if (comm->ctx->GetTransportType(pe) == application::TransportType::SDMA) {
      hasSdmaPeers = true;
      break;
    }
  }

  if (hasSdmaPeers && sdmaNumQueue > 0) {
    HIP_RUNTIME_CHECK(hipMalloc(&signalPtrs, signalArraySize));
    HIP_RUNTIME_CHECK(hipMemset(signalPtrs, 0, signalArraySize));
    HIP_RUNTIME_CHECK(hipMalloc(&expectSignalsPtr, signalArraySize));
    HIP_RUNTIME_CHECK(hipMemset(expectSignalsPtr, 0, signalArraySize));

    // Exchange signal pointers via IPC
    hipIpcMemHandle_t signalHandle;
    HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&signalHandle, signalPtrs));

    auto* signalHandles =
        static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
    comm->bootNet->Allgather(&signalHandle, signalHandles, sizeof(hipIpcMemHandle_t));

    peerSignalPtrs_host_arr = static_cast<HSAuint64**>(calloc(worldSize, sizeof(HSAuint64*)));
    peerSignalPtrs_host_arr[rank] = signalPtrs;
    for (int pe = 0; pe < worldSize; pe++) {
      if (comm->ctx->GetTransportType(pe) != application::TransportType::SDMA) continue;
      if (pe == rank) continue;
      void* mapped = nullptr;
      HIP_RUNTIME_CHECK(
          hipIpcOpenMemHandle(&mapped, signalHandles[pe], hipIpcMemLazyEnablePeerAccess));
      peerSignalPtrs_host_arr[pe] = reinterpret_cast<HSAuint64*>(mapped);
    }

    HIP_RUNTIME_CHECK(hipMalloc(&peerSignalPtrs_gpu, sizeof(HSAuint64*) * worldSize));
    HIP_RUNTIME_CHECK(hipMemcpy(peerSignalPtrs_gpu, peerSignalPtrs_host_arr,
                                sizeof(HSAuint64*) * worldSize, hipMemcpyHostToDevice));
    free(signalHandles);
  }

  // ── Copy arrays to GPU ──
  uint32_t* peerRkeys_gpu = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&peerRkeys_gpu, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(peerRkeys_gpu, peerRkeys_host, sizeof(uint32_t) * worldSize,
                              hipMemcpyHostToDevice));

  // ── Build GPU-side XshmemWindowDevice ──
  XshmemWindowDevice hostShadow = {};
  hostShadow.winBase = static_cast<char*>(comm->flatBase) + slotOffset;
  hostShadow.stride4G = static_cast<uint32_t>(comm->perRankSize >> 32);
  hostShadow.rank = rank;
  hostShadow.worldSize = worldSize;
  hostShadow.ibgdaWin.peerRkeys = peerRkeys_gpu;
  hostShadow.ibgdaWin.lkey = lkey;
  hostShadow.deviceHandles_d = comm->sdmaDevHandles;
  hostShadow.signalPtrs = signalPtrs;
  hostShadow.expectSignalsPtr = expectSignalsPtr;
  hostShadow.peerSignalPtrs = peerSignalPtrs_gpu;
  hostShadow.sdmaNumQueue = static_cast<uint32_t>(sdmaNumQueue);

  XshmemWindowDevice* devPtr = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&devPtr, sizeof(XshmemWindowDevice)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(devPtr, &hostShadow, sizeof(XshmemWindowDevice), hipMemcpyHostToDevice));

  // ── Register in window table (for ncclFindWindow-style lookup) ──
  XshmemComm::WindowTableEntry tableEntry;
  tableEntry.base = reinterpret_cast<uintptr_t>(localPtr);
  tableEntry.size = static_cast<uintptr_t>(size);
  tableEntry.devPtr = devPtr;
  comm->windowTableEntries.push_back(tableEntry);

  // ── Record host-side metadata ──
  auto* wh = new XshmemWindowHost();
  wh->localPtr = localPtr;
  wh->size = size;
  wh->signalPtrs = signalPtrs;
  wh->expectSignalsPtr = expectSignalsPtr;
  wh->peerSignalPtrs = peerSignalPtrs_host_arr;
  wh->devPtr = devPtr;
  wh->peerRkeys_gpu = peerRkeys_gpu;
  wh->peerSignalPtrs_gpu = peerSignalPtrs_gpu;
  comm->windows.push_back(wh);

  *outWin = devPtr;

  // Print window info
  char* winBase = static_cast<char*>(comm->flatBase) + slotOffset;
  MORI_SHMEM_INFO("XshmemWindowRegister: rank={} win={} winBase={} size={} slotOffset={} lkey={}",
                  rank, (void*)devPtr, (void*)winBase, size, slotOffset, lkey);
  for (int pe = 0; pe < worldSize; pe++) {
    void* peerVa = winBase + static_cast<size_t>(pe) * comm->perRankSize;
    MORI_SHMEM_INFO("  PE {}: flatVA={} rkey={}", pe, peerVa, peerRkeys_host[pe]);
  }
  if (signalPtrs) {
    MORI_SHMEM_INFO("  SDMA: signalPtrs={} expectSignals={} numQueue={}",
                    (void*)signalPtrs, (void*)expectSignalsPtr, sdmaNumQueue);
  }
  MORI_SHMEM_INFO("  deviceHandles_d={}", (void*)comm->sdmaDevHandles);

  free(peerRkeys_host);

  return 0;
}

/* ========================================================================== */
/*                      XshmemWindowRegister (convenience)                    */
/* ========================================================================== */

int XshmemWindowRegister(XshmemComm* comm, size_t size, XshmemWindow_t* outWin, void** localPtr) {
  void* ptr = nullptr;
  int ret = XshmemMemAlloc(comm, size, &ptr);
  if (ret != 0) return ret;

  ret = XshmemWindowRegister(comm, ptr, size, outWin);
  if (ret != 0) {
    XshmemMemFree(comm, ptr);
    return ret;
  }

  *localPtr = ptr;
  return 0;
}

/* ========================================================================== */
/*                          XshmemWindowDeregister                            */
/* ========================================================================== */

int XshmemWindowDeregister(XshmemComm* comm, XshmemWindow_t win) {
  // Find matching XshmemWindowHost
  XshmemWindowHost* wh = nullptr;
  size_t idx = 0;
  for (size_t i = 0; i < comm->windows.size(); i++) {
    if (comm->windows[i]->devPtr == win) {
      wh = comm->windows[i];
      idx = i;
      break;
    }
  }
  if (!wh) {
    MORI_SHMEM_WARN("XshmemWindowDeregister: win {} not found", (void*)win);
    return -1;
  }

  MORI_SHMEM_TRACE("XshmemWindowDeregister: rank={} ptr={}", comm->rank, wh->localPtr);

  // Unmap P2P peer slots (mapped during WindowRegister)
  auto allocIt = comm->allocTable.find(wh->localPtr);
  if (allocIt != comm->allocTable.end()) {
    size_t slotOff = allocIt->second.slotOffset;
    size_t allocSize = allocIt->second.size;
    for (int pe = 0; pe < comm->worldSize; pe++) {
      if (pe == comm->rank) continue;
      if (!comm->ctx->CanUseP2P(pe)) continue;
      void* peerVa = static_cast<char*>(comm->flatBase) +
                     static_cast<size_t>(pe) * comm->perRankSize + slotOff;
      hipMemUnmap(peerVa, allocSize);
    }
  }

  // Remove from window table
  auto& entries = comm->windowTableEntries;
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [win](const XshmemComm::WindowTableEntry& e) {
                                 return e.devPtr == win;
                               }),
                entries.end());

  // Deregister RDMA MR
  application::RdmaDeviceContext* rdmaDevCtx = comm->ctx->GetRdmaDeviceContext();
  if (rdmaDevCtx) {
    rdmaDevCtx->DeregisterRdmaMemoryRegion(wh->localPtr);
  }

  // Free GPU arrays
  if (wh->peerRkeys_gpu) HIP_RUNTIME_CHECK(hipFree(wh->peerRkeys_gpu));
  if (wh->signalPtrs) HIP_RUNTIME_CHECK(hipFree(wh->signalPtrs));
  if (wh->expectSignalsPtr) HIP_RUNTIME_CHECK(hipFree(wh->expectSignalsPtr));
  if (wh->peerSignalPtrs_gpu) HIP_RUNTIME_CHECK(hipFree(wh->peerSignalPtrs_gpu));
  if (wh->devPtr) HIP_RUNTIME_CHECK(hipFree(wh->devPtr));

  // Free host-side signal pointer array
  free(wh->peerSignalPtrs);

  // Remove from list
  comm->windows.erase(comm->windows.begin() + idx);
  delete wh;
  return 0;
}

}  // namespace xshmem
}  // namespace mori
