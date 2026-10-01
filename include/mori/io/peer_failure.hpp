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

// Peer liveness signalling. Kept free of the verbs, msgpack and transport headers
// so consumers and their tests need not pull in the IO data plane.

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

// Why a peer became unusable. Distinct from StatusCode, which describes the fate
// of one transfer: this describes the link, and can be raised with none pending.
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

// Asynchronous notification that a peer, or the resource reaching it, has failed.
// Raised independently of any in-flight transfer, so no timeout is needed.
struct PeerFailureEvent {
  // EngineKey of the attributed peer. Empty for device-wide events (PORT_DOWN,
  // DEVICE_FATAL) or when the QP could not be resolved.
  std::string remoteEngineKey;
  PeerFailureReason reason{PeerFailureReason::UNKNOWN};
  // QP number the event arrived on; 0 when the event is not QP-scoped.
  uint32_t qpNum{0};
  // RDMA device the event was observed on, e.g. "mlx5_0".
  std::string deviceName;
  // Human-readable detail for logs and error propagation.
  std::string detail;
};

// Invoked on the async-event monitor thread. Must not block or re-enter the
// engine; record the event so an application thread can pick it up later.
using PeerFailureCallback = std::function<void(const PeerFailureEvent&)>;

// Records asynchronous peer failures and answers QP liveness; thread-safe. A QP
// is dead only once a fatal event is observed for it, so silence means alive.
class PeerFailureTracker {
 public:
  // Bound on unclaimed failures. Overflow drops the newest and counts it, since
  // the oldest events are the diagnostic ones.
  static constexpr size_t kMaxPending = 1024;

  // Associates a local QP with the peer and device it belongs to, so an event
  // carrying only a QP number can be attributed to a peer.
  void RegisterQp(uint32_t qpNum, std::string remoteEngineKey, std::string deviceName);
  // Drops a QP association. Any recorded failure for it is retained, so a
  // pending event stays drainable after the endpoint is gone.
  void ForgetQp(uint32_t qpNum);

  // Records a failure, resolving remoteEngineKey when possible. QP-scoped events
  // mark that QP dead; events with no QP mark the whole device dead.
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
