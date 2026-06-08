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
#pragma once

// clang-format off
#include "mori/cco/gda/gda_device_types.hpp"
#include "mori/cco/gda/gda_device_primitives.hpp"
// clang-format on

namespace mori {
namespace cco {
namespace gda {

// translate a GDA team-local peer index to a global rank.
// FULL:      identity
// RAIL:      teamPeer is node_id; global = teamPeer * lsaSize + lsaRank
// CROSSNODE: team = [0,nodeStart) ∪ {self at nodeStart} ∪ [nodeStart+lsaSize,worldSize)
// NONE:      returns -1
__device__ inline int GdaPeerToWorld(ccoDevComm const& comm, int teamPeer) {
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      return teamPeer;
    case CCO_GDA_CONNECTION_RAIL:
      return teamPeer * comm.lsaSize + comm.lsaRank;
    case CCO_GDA_CONNECTION_CROSSNODE: {
      int nodeStart = (comm.rank / comm.lsaSize) * comm.lsaSize;
      if (teamPeer < nodeStart) return teamPeer;
      if (teamPeer == nodeStart) return comm.rank;
      return teamPeer + comm.lsaSize - 1;
    }
    default:
      return -1;
  }
}

// translate a global rank to a GDA team-local peer index (inverse of GdaPeerToWorld).
// FULL:      identity
// RAIL:      teamPeer = globalPeer / lsaSize (node_id of globalPeer)
// CROSSNODE: reverse the team layout described above
// NONE:      returns -1
__device__ inline int WorldPeerToGda(ccoDevComm const& comm, int globalPeer) {
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      return globalPeer;
    case CCO_GDA_CONNECTION_RAIL:
      return globalPeer / comm.lsaSize;
    case CCO_GDA_CONNECTION_CROSSNODE: {
      int nodeStart = (comm.rank / comm.lsaSize) * comm.lsaSize;
      if (globalPeer < nodeStart) return globalPeer;
      if (globalPeer == comm.rank) return nodeStart;
      return globalPeer - comm.lsaSize + 1;
    }
    default:
      return -1;
  }
}

template <core::ProviderType PrvdType>
__device__ inline ccoGda<PrvdType>::ccoGda(ccoDevComm const& comm_, int contextIndex)
    : comm(comm_), contextId(contextIndex) {
  this->_gdaHandle = (void*)&comm.ibgda;
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      this->rank = comm.rank;
      this->nRanks = comm.worldSize;
      break;
    case CCO_GDA_CONNECTION_RAIL:
      this->rank = comm.rank / comm.lsaSize;
      this->nRanks = comm.worldSize / comm.lsaSize;
      break;
    case CCO_GDA_CONNECTION_CROSSNODE: {
      int nodeStart = (comm.rank / comm.lsaSize) * comm.lsaSize;
      this->rank = nodeStart;
      this->nRanks = comm.worldSize - comm.lsaSize + 1;
      break;
    }
    default:  // CCO_GDA_CONNECTION_NONE
      this->rank = 0;
      this->nRanks = 0;
      break;
  }
}

// put: RDMA write with optional signal/counter
template <core::ProviderType PrvdType>
template <typename RemoteAction, typename LocalAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                             ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                             RemoteAction remoteAction, LocalAction localAction,
                                             Coop coop, uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int teamPeer = WorldPeerToGda(comm, peer);

    // step 1: parse windows to extract lkey/rkey
    ccoWindowDevice* dstWinDev = reinterpret_cast<ccoWindowDevice*>(dstWin);
    ccoWindowDevice* srcWinDev = reinterpret_cast<ccoWindowDevice*>(srcWin);

    uint32_t srcLkey = srcWinDev->ibgdaWin.lkey;
    uint32_t dstRkey = dstWinDev->ibgdaWin.peerRkeys[teamPeer];

    uintptr_t localAddr = srcOffset;
    uintptr_t remoteAddr = dstOffset;

    // step 2: select endpoint (based on team peer + contextId)
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // step 3: parse RemoteAction -> signal parameters
    constexpr bool hasSignal = !std::is_same_v<RemoteAction, ccoGda_NoSignal>;
    uintptr_t signalRaddr = 0;
    uint32_t signalRkey = 0;
    ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
    uint64_t signalOpArg = 0;

    if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[teamPeer];
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[teamPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // step 4: parse LocalAction -> counter parameters
    constexpr bool hasCounter = !std::is_same_v<LocalAction, ccoGda_NoCounter>;
    uintptr_t counterRaddr = 0;
    uint32_t counterRkey = 0;

    if constexpr (std::is_same_v<LocalAction, ccoGda_CounterInc>) {
      uintptr_t counterBaseAddr = reinterpret_cast<uintptr_t>(ibgda->counterBuf);
      counterRaddr = counterBaseAddr + localAction.counterId * sizeof(uint64_t);
      counterRkey = comm.resourceWindow_inlined.ibgdaWin.lkey;
    }

    // step 5: call primitive API (PrvdType is compile-time determined)
    putImpl<PrvdType>(ep, qpn,
                      bytes > 0,            // hasData
                      localAddr, srcLkey,   // local
                      remoteAddr, dstRkey,  // remote
                      bytes, hasSignal, signalRaddr, signalRkey, signalOp, signalOpArg, hasCounter,
                      counterRaddr, counterRkey, optFlags);
  }
  coop.sync();
}

// putValue: write immediate value (≤8 bytes)
template <core::ProviderType PrvdType>
template <typename T, typename RemoteAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::putValue(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                                  T value, RemoteAction remoteAction, Coop coop,
                                                  uint32_t optFlags) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");

  coop.sync();
  if (coop.thread_rank() == 0) {
    int teamPeer = WorldPeerToGda(comm, peer);

    // step 1: parse window to extract rkey
    ccoWindowDevice* dstWinDev = reinterpret_cast<ccoWindowDevice*>(dstWin);
    uint32_t dstRkey = dstWinDev->ibgdaWin.peerRkeys[teamPeer];
    uintptr_t remoteAddr = dstOffset;

    // step 2: select endpoint
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // step 3: parse RemoteAction
    constexpr bool hasSignal = !std::is_same_v<RemoteAction, ccoGda_NoSignal>;
    uintptr_t signalRaddr = 0;
    uint32_t signalRkey = 0;
    ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
    uint64_t signalOpArg = 0;

    if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[teamPeer];
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[teamPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // step 4: call primitive API
    putValueImpl<PrvdType, T>(ep, qpn, remoteAddr, dstRkey, value, hasSignal, signalRaddr,
                              signalRkey, signalOp, signalOpArg, optFlags);
  }
  coop.sync();
}

// get: RDMA read
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                                             Coop coop, uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int teamPeer = WorldPeerToGda(comm, peer);

    // step 1: parse windows
    ccoWindowDevice* remoteWinDev = reinterpret_cast<ccoWindowDevice*>(remoteWin);
    ccoWindowDevice* localWinDev = reinterpret_cast<ccoWindowDevice*>(localWin);

    uint32_t remoteRkey = remoteWinDev->ibgdaWin.peerRkeys[teamPeer];
    uint32_t localLkey = localWinDev->ibgdaWin.lkey;

    uintptr_t remoteAddr = remoteOffset;
    uintptr_t localAddr = localOffset;

    // step 2: select endpoint
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // step 3: call primitive API
    getImpl<PrvdType>(ep, qpn, localAddr, localLkey, remoteAddr, remoteRkey, bytes, optFlags);
  }
  coop.sync();
}

// signal: send to remote peer
template <core::ProviderType PrvdType>
template <typename RemoteAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::signal(int peer, RemoteAction remoteAction, Coop coop) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int teamPeer = WorldPeerToGda(comm, peer);

    // select endpoint first to get ibgda context
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // parse RemoteAction
    ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
    uint64_t signalOpArg = 0;
    uintptr_t signalRaddr = 0;
    uint32_t signalRkey = 0;

    if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[teamPeer];
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[teamPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // call primitive signal
    signalImpl<PrvdType>(ep, qpn, signalRaddr, signalRkey, signalOp, signalOpArg);
  }
  coop.sync();
}

// flush = flushAsync + wait per peer.
// flushAsync rings the doorbell if any WQEs are pending (skips if already rung),
// then wait polls CQ until all submitted WQEs complete.

// flush all peers: distribute peers across the Coop group (default: warp).
// all threads in the group must call flush together.
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::flush(Coop coop) {
  coop.sync();
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  for (int teamPeer = coop.thread_rank(); teamPeer < this->nRanks; teamPeer += coop.size()) {
    if (teamPeer == this->rank) continue;
    int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
    uint32_t postIdx = 0;
    flushAsyncImpl<PrvdType>(ep, ep->qpn, &postIdx);
    waitImpl<PrvdType>(ep, postIdx);
  }
  coop.sync();
}

// flush single peer: ring doorbell if needed, then poll CQ until complete.
template <core::ProviderType PrvdType>
__device__ inline void ccoGda<PrvdType>::flush(int peer) {
  int teamPeer = WorldPeerToGda(comm, peer);
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  uint32_t postIdx = 0;
  flushAsyncImpl<PrvdType>(ep, ep->qpn, &postIdx);
  waitImpl<PrvdType>(ep, postIdx);
}

// flushAsync: ring doorbell for peer, return a request handle for wait().
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::flushAsync(int peer, ccoGdaRequest_t* outRequest,
                                                    Coop coop) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int teamPeer = WorldPeerToGda(comm, peer);
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = teamPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];

    uint32_t postIdx = 0;
    flushAsyncImpl<PrvdType>(ep, ep->qpn, &postIdx);

    outRequest->qpIdx = qpIdx;
    outRequest->postIdx = static_cast<uint64_t>(postIdx);
  }
  coop.sync();
}

// wait: block until a request returned by flushAsync completes.
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::wait(ccoGdaRequest_t& request, Coop coop) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    waitImpl<PrvdType>(&ibgda->endpoints[request.qpIdx], static_cast<uint32_t>(request.postIdx));
  }
  coop.sync();
}

// readSignal: read local signal value
template <core::ProviderType PrvdType>
__device__ inline uint64_t ccoGda<PrvdType>::readSignal(ccoGdaSignal_t signalId, int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  return readSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, signalId, bits);
}

// waitSignal: wait until local signal reaches specified value
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::waitSignal(ccoGdaSignal_t signalId, uint64_t least,
                                                    Coop coop, int bits) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    waitSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, signalId, least, bits);
  }
  coop.sync();
}

// resetSignal: reset local signal to zero
template <core::ProviderType PrvdType>
__device__ inline void ccoGda<PrvdType>::resetSignal(ccoGdaSignal_t signalId) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  resetSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, signalId);
}

// readCounter: read local counter value
template <core::ProviderType PrvdType>
__device__ inline uint64_t ccoGda<PrvdType>::readCounter(ccoGdaCounter_t counterId, int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  return readCounterImpl<PrvdType>(ibgda->counterBuf, counterId, bits);
}

// waitCounter: wait until local counter reaches specified value
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::waitCounter(ccoGdaCounter_t counterId, uint64_t least,
                                                     Coop coop, int bits) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    waitCounterImpl<PrvdType>(ibgda->counterBuf, counterId, least, bits);
  }
  coop.sync();
}

// resetCounter: reset local counter to zero
template <core::ProviderType PrvdType>
__device__ inline void ccoGda<PrvdType>::resetCounter(ccoGdaCounter_t counterId) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  resetCounterImpl<PrvdType>(ibgda->counterBuf, counterId);
}

}  // namespace gda
}  // namespace cco
}  // namespace mori
