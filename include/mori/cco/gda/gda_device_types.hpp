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
// low-level GDA type aliases/enums shared by both the primitive layer
// (gda_device_primitive.hpp) and the high-level layer (gda_device_common.hpp).
// kept in a standalone header so the primitive layer can depend on these
// types without creating a circular include with the common layer.
#pragma once

/* clang-format off */
#include "mori/cco/cco_types.hpp"
#include "mori/cco/cco_coop.hpp"
/* clang-format on */

namespace mori {
namespace cco {
namespace gda {

typedef void* ccoWindow_t;
typedef struct {
  int qpIdx;
  uint64_t postIdx;
} ccoGdaRequest_t;

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
  int rank;    // my index in the GDA team [0, nRanks)
  int nRanks;  // GDA team size, derived from gdaConnType at construction
  uint32_t contextId;
  void* _gdaHandle;

  // constructor
  __device__ inline ccoGda(ccoDevComm const&, int contextIndex);

  // ── data transfer ───────────────────────────────────────────────────────

  // put: rdma write with optional remote signal and local counter.
  template <typename RemoteAction = ccoGda_NoSignal, typename LocalAction = ccoGda_NoCounter,
            typename Coop = ccoCoopThread>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = ccoGda_NoSignal{},
                             LocalAction localAction = ccoGda_NoCounter{}, Coop coop = Coop{},
                             uint32_t optFlags = ccoGdaOptFlagsDefault);

  // putValue: write an immediate value (≤8 bytes) with optional remote signal.
  template <typename T, typename RemoteAction = ccoGda_NoSignal, typename Coop = ccoCoopThread>
  __device__ inline void putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = ccoGda_NoSignal{}, Coop coop = Coop{},
                                  uint32_t optFlags = ccoGdaOptFlagsDefault);

  // get: rdma read — pull peer's window content into our local window.
  template <typename Coop = ccoCoopThread>
  __device__ inline void get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                             Coop coop = Coop{}, uint32_t optFlags = ccoGdaOptFlagsDefault);

  // ── signal ──────────────────────────────────────────────────────────────

  // signal: send a signal-only message to peer (no data payload).
  template <typename RemoteAction, typename Coop = ccoCoopThread>
  __device__ inline void signal(int peer, RemoteAction remoteAction, Coop coop = Coop{});

  // readSignal: read the local value of one signal slot.
  __device__ inline uint64_t readSignal(ccoGdaSignal_t signalId, int bits = 64);

  // waitSignal: block until the local signal slot reaches `least`.
  template <typename Coop = ccoCoopThread>
  __device__ inline void waitSignal(ccoGdaSignal_t signalId, uint64_t least, Coop coop = Coop{},
                                    int bits = 64);

  // resetSignal: zero one local signal slot.
  __device__ inline void resetSignal(ccoGdaSignal_t signalId);

  // ── counter ─────────────────────────────────────────────────────────────

  // readCounter: read the local value of one counter slot.
  __device__ inline uint64_t readCounter(ccoGdaCounter_t counterId, int bits = 56);

  // waitCounter: block until the local counter slot reaches `least`.
  template <typename Coop = ccoCoopThread>
  __device__ inline void waitCounter(ccoGdaCounter_t counterId, uint64_t least, Coop coop = Coop{},
                                     int bits = 56);

  // resetCounter: zero one local counter slot.
  __device__ inline void resetCounter(ccoGdaCounter_t counterId);

  // ── completion ──────────────────────────────────────────────────────────

  // flush = flushAsync + wait per peer.
  // flushAsync rings the doorbell if any WQEs are pending (skips if already
  // rung), then wait polls CQ until all submitted WQEs complete.

  // flush: ring doorbell + poll CQ for every peer.
  // peers are distributed across the Coop group (default: warp).
  // all threads in the group must call flush together.
  template <typename Coop = ccoCoopWarp>
  __device__ inline void flush(Coop coop = Coop{});

  // flush(peer): poll CQ for a single peer until its submitted WQEs complete.
  template <typename Coop = ccoCoopWarp>
  __device__ inline void flush(int peer, Coop coop = Coop{});

  // flushAsync: ring doorbell for peer and return a request handle that
  // wait() can later be used to wait on individually.
  template <typename Coop = ccoCoopThread>
  __device__ inline void flushAsync(int peer, ccoGdaRequest_t* outRequest, Coop coop = Coop{});

  // wait: block on a request handle previously returned by flushAsync.
  template <typename Coop = ccoCoopWarp>
  __device__ inline void wait(ccoGdaRequest_t& request, Coop coop = Coop{});
};

}  // namespace gda
}  // namespace cco
}  // namespace mori
