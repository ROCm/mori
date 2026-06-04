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
//
// Low-level GDA type aliases/enums shared by both the primitive layer
// (gda_device_primitive.hpp) and the high-level layer (gda_device_common.hpp).
// Kept in a standalone header so the primitive layer can depend on these
// types without creating a circular include with the common layer.
#pragma once

#include <stdint.h>

namespace mori {
namespace cco {
namespace gda {

typedef void* ccoWindow_t;
typedef void* ccoGdaRequest_t;

typedef uint32_t ccoGdaSignal_t;
typedef uint32_t ccoGdaCounter_t;

enum ccoGdaOptFlags {
  ccoGdaOptFlagsDefault = 0,
  ccoGdaOptFlagsMaySkipCreditCheck = (1 << 0),
  ccoGdaOptFlagsAggregateRequests = (1 << 1),
};

typedef enum ccoGdaSignalOp_t {
  ccoGdaSignalInc = 0,
  ccoGdaSignalAdd,
} ccoGdaSignalOp_t;

struct ccoGda_NoSignal {};
struct ccoGda_NoCounter {};

struct ccoGda_SignalInc {
  ccoGdaSignal_t signalId;
  __device__ inline ccoGda_SignalInc(ccoGdaSignal_t id) : signalId(id) {}
};

struct ccoGda_SignalAdd {
  ccoGdaSignal_t signalId;
  uint64_t value;
  __device__ inline ccoGda_SignalAdd(ccoGdaSignal_t id, uint64_t val) : signalId(id), value(val) {}
};

struct ccoGda_CounterInc {
  ccoGdaCounter_t counterId;
  __device__ inline ccoGda_CounterInc(ccoGdaCounter_t id) : counterId(id) {}
};

struct ccoGdaCtx {
  int rank;
  int worldSize;
  void* handle;
  int contextId;
};

template <core::ProviderType PrvdType>
struct ccoGda {
  ccoDevComm const& comm;
  uint32_t contextId;
  void* _gdaHandle;

  // Constructor
  __device__ inline ccoGda(ccoDevComm const&, int contextIndex);

  // Data transfer operations
  template <typename RemoteAction = ccoGda_NoSignal, typename LocalAction = ccoGda_NoCounter>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = ccoGda_NoSignal{},
                             LocalAction localAction = ccoGda_NoCounter{},
                             uint32_t optFlags = ccoGdaOptFlagsDefault);

  template <typename T, typename RemoteAction = ccoGda_NoSignal>
  __device__ inline void putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = ccoGda_NoSignal{},
                                  uint32_t optFlags = ccoGdaOptFlagsDefault);

  __device__ inline void get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                             uint32_t optFlags = ccoGdaOptFlagsDefault);

  // Signal operations
  template <typename RemoteAction>
  __device__ inline void signal(int peer, RemoteAction remoteAction);

  __device__ inline uint64_t readSignal(ccoGdaSignal_t signalId, int bits = 64);

  __device__ inline void waitSignal(ccoGdaSignal_t signalId, uint64_t least, int bits = 64);

  __device__ inline void resetSignal(ccoGdaSignal_t signalId);

  // Counter operations
  __device__ inline uint64_t readCounter(ccoGdaCounter_t counterId, int bits = 56);

  __device__ inline void waitCounter(ccoGdaCounter_t counterId, uint64_t least, int bits = 56);

  __device__ inline void resetCounter(ccoGdaCounter_t counterId);

  // Completion operations
  __device__ inline void flush();          // Ring doorbell for all peers
  __device__ inline void flush(int peer);  // Ring doorbell for specific peer

  __device__ inline void flushAsync(int peer, ccoGdaRequest_t* outRequest);

  __device__ inline void wait(ccoGdaRequest_t& request);
};

// Type aliases for convenience
using ccoGdaMLX5 = ccoGda<core::ProviderType::MLX5>;
using ccoGdaPSD = ccoGda<core::ProviderType::PSD>;
using ccoGdaBNXT = ccoGda<core::ProviderType::BNXT>;

}  // namespace gda
}  // namespace cco
}  // namespace mori
