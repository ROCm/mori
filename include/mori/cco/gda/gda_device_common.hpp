// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {
namespace gda {

struct CcoGda_NoSignal {};
struct CcoGda_NoCounter {};

struct CcoGda_SignalInc {
  CcoGdaSignal_t signalId;
  __device__ inline CcoGda_SignalInc(CcoGdaSignal_t id) : signalId(id) {}
};

struct CcoGda_SignalAdd {
  CcoGdaSignal_t signalId;
  uint64_t value;
  __device__ inline CcoGda_SignalAdd(CcoGdaSignal_t id, uint64_t val)
      : signalId(id), value(val) {}
};

struct CcoGda_CounterInc {
  CcoGdaCounter_t counterId;
  __device__ inline CcoGda_CounterInc(CcoGdaCounter_t id) : counterId(id) {}
};

struct CcoGdaCtx {
  int rank;
  int worldSize;
  void* handle;
  int contextId;
};

typedef void* CcoWindow_t;
typedef void* CcoGdaRequest_t;

typedef uint32_t CcoGdaSignal_t;
typedef uint32_t CcoGdaCounter_t;

typedef enum CcoGdaSignalOp_t {
  CcoGdaSignalInc = 0,
  CcoGdaSignalAdd,
} CcoGdaSignalOp_t;

struct CcoGda {
  CcoDevComm const& comm;
  uint32_t contextId;
  CcoGdaCtx ctx;
  void* _gdaHandle;

  // Constructor
  __device__ inline CcoGda(CcoDevComm const&, int contextIndex);

  // Data transfer operations
  template <typename RemoteAction = CcoGda_NoSignal, typename LocalAction = CcoGda_NoCounter>
  __device__ inline void put(int peer, CcoWindow_t dstWin, size_t dstOffset,
                             CcoWindow_t srcWin, size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = CcoGda_NoSignal{},
                             LocalAction localAction = CcoGda_NoCounter{});

  template <typename T, typename RemoteAction = CcoGda_NoSignal>
  __device__ inline void putValue(int peer, CcoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = CcoGda_NoSignal{});

  __device__ inline void get(int peer, CcoWindow_t remoteWin, size_t remoteOffset,
                             CcoWindow_t localWin, size_t localOffset, size_t bytes);

  // Signal operations
  template <typename RemoteAction>
  __device__ inline void signal(int peer, RemoteAction remoteAction);

  __device__ inline uint64_t readSignal(CcoGdaSignal_t signalId, int bits = 64);

  __device__ inline void waitSignal(CcoGdaSignal_t signalId, uint64_t least, int bits = 64);

  __device__ inline void resetSignal(CcoGdaSignal_t signalId);

  // Counter operations
  __device__ inline uint64_t readCounter(CcoGdaCounter_t counterId, int bits = 56);

  __device__ inline void waitCounter(CcoGdaCounter_t counterId, uint64_t least, int bits = 56);

  __device__ inline void resetCounter(CcoGdaCounter_t counterId);

  // Completion operations
  __device__ inline void flush();

  __device__ inline void flushAsync(int peer, CcoGdaRequest_t* outRequest);

  __device__ inline void wait(CcoGdaRequest_t& request);
};

}  // namespace gda
}  // namespace cco
}  // namespace mori