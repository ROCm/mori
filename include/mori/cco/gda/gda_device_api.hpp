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
#pragma once

#include "mori/cco/gda/gda_device_common.hpp"

namespace mori {
namespace cco {
namespace gda {

__device__ inline ccoGda::ccoGda(ccoDevComm const& comm_, int contextIndex)
    : comm(comm_), contextId(contextIndex) {
  this->ctx.rank = comm.rank;
  this->ctx.worldSize = comm.worldSize;
  this->ctx.contextId = contextIndex;
  this->ctx.handle = (void*)&comm.ibgda;
  this->_gdaHandle = (void*)&comm.ibgda;
}

// Put: RDMA write with optional signal/counter
template <typename RemoteAction = ccoGda_NoSignal, typename LocalAction = ccoGda_NoCounter>
__device__ inline void ccoGda::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                   ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                   RemoteAction remoteAction, LocalAction localAction) {
  bool isSignal = false;
  ccoGdaSignal_t signalId = 0;
  ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
  uint64_t signalOpArg = 0;

  if constexpr (!std::is_same<RemoteAction, ccoGda_NoSignal>::value) {
    isSignal = true;
    if constexpr (std::is_same<RemoteAction, ccoGda_SignalInc>::value) {
      signalId = remoteAction.signalId;
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same<RemoteAction, ccoGda_SignalAdd>::value) {
      signalId = remoteAction.signalId;
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }
  }

  bool isCounter = false;
  ccoGdaCounter_t counterId = 0;

  if constexpr (!std::is_same<LocalAction, ccoGda_NoCounter>::value) {
    isCounter = true;
    if constexpr (std::is_same<LocalAction, ccoGda_CounterInc>::value) {
      counterId = localAction.counterId;
    }
  }
  gda::put(this->ctx, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, isSignal, signalId,
           signalOp, signalOpArg, isCounter, counterId);
}

// PutValue: write immediate value (≤8 bytes)
template <typename T, typename RemoteAction = ccoGda_NoSignal>
__device__ inline void ccoGda::putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                        RemoteAction remoteAction) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");

  bool isSignal = false;
  ccoGdaSignal_t signalId = 0;
  ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
  uint64_t signalOpArg = 0;

  if constexpr (!std::is_same<RemoteAction, ccoGda_NoSignal>::value) {
    isSignal = true;
    if constexpr (std::is_same<RemoteAction, ccoGda_SignalInc>::value) {
      signalId = remoteAction.signalId;
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same<RemoteAction, ccoGda_SignalAdd>::value) {
      signalId = remoteAction.signalId;
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }
  }
  gda::putValue(this->ctx, peer, dstWin, dstOffset, value, isSignal, signalId, signalOp,
                signalOpArg);
}

// Get: RDMA read
__device__ inline void ccoGda::get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                                   ccoWindow_t localWin, size_t localOffset, size_t bytes) {
  gda::get(this->ctx, peer, remoteWin, remoteOffset, localWin, localOffset, bytes);
}

// Signal: send to remote peer
template <typename RemoteAction>
__device__ inline void ccoGda::signal(int peer, RemoteAction remoteAction) {
  ccoGdaSignal_t signalId = 0;
  ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
  uint64_t signalOpArg = 0;

  if constexpr (std::is_same<RemoteAction, ccoGda_SignalInc>::value) {
    signalId = remoteAction.signalId;
    signalOp = ccoGdaSignalInc;
    signalOpArg = 1;
  } else if constexpr (std::is_same<RemoteAction, ccoGda_SignalAdd>::value) {
    signalId = remoteAction.signalId;
    signalOp = ccoGdaSignalAdd;
    signalOpArg = remoteAction.value;
  }

  gda::signal(this->ctx, peer, signalId, signalOp, signalOpArg);
}

// Flush: ensure all operations complete
__device__ inline void ccoGda::flush() {
  for (int peer = 0; peer < this->ctx.worldSize; peer++) {
    if (peer != this->ctx.rank) {
      gda::flush(this->ctx, peer);
    }
  }
}

// FlushAsync: async flush for peer
__device__ inline void ccoGda::flushAsync(int peer, ccoGdaRequest_t* outRequest) {
  gda::flushAsync(this->ctx, peer, outRequest);
}

// Wait: wait for async request
__device__ inline void ccoGda::wait(ccoGdaRequest_t& request) {
  // TODO: poll completion queue
}
__device__ inline uint64_t ccoGda::readSignal(ccoGdaSignal_t signalId, int bits) {
  return gda::readSignal(this->ctx, signalId, bits);
}

// WaitSignal: wait until local signal reaches specified value
__device__ inline void ccoGda::waitSignal(ccoGdaSignal_t signalId, uint64_t least, int bits) {
  gda::waitSignal(this->ctx, signalId, least, bits);
}

__device__ inline void ccoGda::resetSignal(ccoGdaSignal_t signalId) {
  gda::resetSignal(this->ctx, signalId);
}
__device__ inline uint64_t ccoGda::readCounter(ccoGdaCounter_t counterId, int bits) {
  return gda::readCounter(this->ctx, counterId, bits);
}

// WaitCounter: wait until local counter reaches specified value
__device__ inline void ccoGda::waitCounter(ccoGdaCounter_t counterId, uint64_t least, int bits) {
  gda::waitCounter(this->ctx, counterId, least, bits);
}

// ResetCounter: reset local counter to zero
__device__ inline void ccoGda::resetCounter(ccoGdaCounter_t counterId) {
  gda::resetCounter(this->ctx, counterId);
}

// Low-level GDA API (to be implemented with actual GDA hardware API)
__device__ inline static void put(ccoGdaCtx ctx, int peer, ccoWindow_t dstWin, size_t dstOffset,
                                  ccoWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal,
                                  ccoGdaSignal_t signalId, ccoGdaSignalOp_t signalOp,
                                  uint64_t signalOpArg, bool isCounter, ccoGdaCounter_t counterId) {
}

template <typename T>
__device__ inline static void putValue(ccoGdaCtx ctx, int peer, ccoWindow_t dstWin,
                                       size_t dstOffset, T value, bool isSignal,
                                       ccoGdaSignal_t signalId, ccoGdaSignalOp_t signalOp,
                                       uint64_t signalOpArg) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");
  // TODO: Implement with actual GDA hardware API
}
__device__ inline static void get(ccoGdaCtx ctx, int peer, ccoWindow_t remoteWin,
                                  size_t remoteOffset, ccoWindow_t localWin, size_t localOffset,
                                  size_t bytes) {}

__device__ inline static void flush(ccoGdaCtx ctx, int peer) {}
__device__ inline static void flushAsync(ccoGdaCtx ctx, int peer, ccoGdaRequest_t* outRequest) {}
__device__ inline static void signal(ccoGdaCtx ctx, int peer, ccoGdaSignal_t signalId,
                                     ccoGdaSignalOp_t signalOp, uint64_t signalOpArg) {}

__device__ inline static void resetSignal(ccoGdaCtx ctx, ccoGdaSignal_t signalId) {}
__device__ inline static uint64_t readSignal(ccoGdaCtx ctx, ccoGdaSignal_t signalId, int bits) {
  return 0;
}

__device__ inline static void waitSignal(ccoGdaCtx ctx, ccoGdaSignal_t signalId, uint64_t least,
                                         int bits) {}
__device__ inline static uint64_t readCounter(ccoGdaCtx ctx, ccoGdaCounter_t counterId, int bits) {
  return 0;
}

__device__ inline static void resetCounter(ccoGdaCtx ctx, ccoGdaCounter_t counterId) {}
__device__ inline static void waitCounter(ccoGdaCtx ctx, ccoGdaCounter_t counterId, uint64_t least,
                                          int bits) {}

}  // namespace gda
}  // namespace cco
}  // namespace mori
