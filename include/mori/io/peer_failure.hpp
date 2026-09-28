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

// Peer liveness signalling. Deliberately free of any dependency on the verbs,
// msgpack and transport headers so that code which only needs to react to a peer
// failure — and the tests for that logic — need not pull in the IO data plane.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mori {
namespace io {

// Why a peer became unusable. Kept distinct from StatusCode because StatusCode
// describes the fate of one transfer, while this describes the fate of the link
// to a peer: it can be raised when no transfer is outstanding, and it must let
// callers separate "peer/path is gone" from "peer is merely slow" (which is not
// a failure and is therefore never reported here).
enum class PeerFailureReason : uint32_t {
  UNKNOWN = 0,
  // QP transitioned to an unrecoverable error state (IBV_EVENT_QP_FATAL /
  // QP_REQ_ERR / QP_ACCESS_ERR). The peer or the path to it is gone.
  QP_FATAL = 1,
  // Local port went down (IBV_EVENT_PORT_ERR). Every peer reached through it is
  // unreachable until the port comes back.
  PORT_DOWN = 2,
  // Local HCA raised a fatal error (IBV_EVENT_DEVICE_FATAL).
  DEVICE_FATAL = 3,
  // CQ entered an error state (IBV_EVENT_CQ_ERR); completions on it are lost.
  CQ_ERROR = 4,
};

// An asynchronous notification that a peer, or the local resource used to reach
// it, has failed. Raised independently of any in-flight transfer, so a caller
// parked waiting on data that has not started arriving can still learn that it
// will never arrive, without imposing a wall-clock timeout of its own.
struct PeerFailureEvent {
  // EngineKey of the peer this failure is attributed to. Empty when the event is
  // not tied to a single QP (PORT_DOWN and DEVICE_FATAL affect every peer on the
  // device) or when the QP was already torn down and could not be resolved.
  std::string remoteEngineKey;
  PeerFailureReason reason{PeerFailureReason::UNKNOWN};
  // QP number the event arrived on; 0 when the event is not QP-scoped.
  uint32_t qpNum{0};
  // RDMA device the event was observed on, e.g. "mlx5_0".
  std::string deviceName;
  // Human-readable detail for logs and error propagation.
  std::string detail;
};

// Invoked on the async-event monitor thread. Implementations must not block and
// must not re-enter the engine; the intended use is to record the event so an
// application thread can pick it up later.
using PeerFailureCallback = std::function<void(const PeerFailureEvent&)>;

// Records peer failures reported asynchronously and answers liveness questions
// about individual QPs. Written by the async-event monitor thread and read by
// application threads, so every method is internally synchronized.
//
// The asymmetry here is deliberate: a QP is reported dead only once a fatal
// event has actually been observed for it. Silence means alive, so a slow peer
// is never mistaken for a dead one and no elapsed-time heuristic is needed.
class PeerFailureTracker {
 public:
  // Bound on unclaimed failures. A caller that never drains must not be able to
  // grow this without limit; the oldest events are the diagnostic ones, so
  // overflow drops the newest and counts it.
  static constexpr size_t kMaxPending = 1024;

  // Associates a local QP with the peer and device it belongs to, so an event
  // carrying only a QP number can be attributed to a peer.
  void RegisterQp(uint32_t qpNum, std::string remoteEngineKey, std::string deviceName);
  // Drops a QP association. Any recorded failure for it is retained, so a
  // pending event stays drainable after the endpoint is gone.
  void ForgetQp(uint32_t qpNum);

  // Records a failure, filling in remoteEngineKey when the QP can be resolved.
  // QP-scoped events mark that QP dead; events with no QP mark the whole device
  // dead rather than guessing which peer was affected.
  void Record(PeerFailureEvent event);

  // Takes the oldest pending failure. False when none is pending.
  bool Pop(PeerFailureEvent* out);

  // False only once a fatal event has been observed for this QP or for the
  // device it was registered on. Unknown QPs are reported alive.
  bool IsQpAlive(uint32_t qpNum) const;

  // Number of failures dropped because the pending queue was full.
  uint64_t Dropped() const;

  // Pending, not-yet-drained failure count. For tests and diagnostics.
  size_t PendingCount() const;

 private:
  struct QpOwner {
    std::string remoteEngineKey;
    std::string deviceName;
  };

  mutable std::mutex mu_;
  std::unordered_map<uint32_t, QpOwner> qpOwners_;
  std::unordered_set<uint32_t> failedQpns_;
  std::unordered_set<std::string> failedDevices_;
  std::deque<PeerFailureEvent> pending_;
  uint64_t dropped_{0};
};

}  // namespace io
}  // namespace mori
