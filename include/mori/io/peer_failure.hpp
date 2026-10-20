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
// A passive receiver sees none of these: a WRITE target posts nothing to time out.
enum class PeerFailureReason : uint32_t {
  UNKNOWN = 0,
  // Peer did not ack within the HCA's retry budget (IBV_WC_RETRY_EXC_ERR), seen only when sending.
  PEER_UNREACHABLE = 1,
  // Local QP is unrecoverable (IBV_EVENT_QP_FATAL / QP_REQ_ERR / QP_ACCESS_ERR).
  LOCAL_QP_ERROR = 2,
  // Local port went down (IBV_EVENT_PORT_ERR); cleared once it comes back up.
  LOCAL_PORT_DOWN = 3,
  // Local HCA raised a fatal error (IBV_EVENT_DEVICE_FATAL).
  LOCAL_DEVICE_FATAL = 4,
  // Local CQ is in an error state (IBV_EVENT_CQ_ERR); only QPs sharing it are affected.
  LOCAL_CQ_ERROR = 5,
};

// Asynchronous notification that a peer, or the resource reaching it, has failed.
// Raised independently of any in-flight transfer, so no timeout is needed.
struct PeerFailureEvent {
  // EngineKey of the attributed peer. Empty when the event is not QP-scoped.
  std::string remoteEngineKey;
  PeerFailureReason reason{PeerFailureReason::UNKNOWN};
  // QP number the event arrived on, unique only within deviceName; 0 if not QP-scoped.
  uint32_t qpNum{0};
  // Local port the event concerns; 0 when the port is not known.
  uint32_t portNum{0};
  // RDMA device the event was observed on, e.g. "mlx5_0".
  std::string deviceName;
  // Human-readable detail for logs and error propagation.
  std::string detail;
};

// Which resource a failure invalidates; each covers a different set of sessions.
enum class FailureScope : uint32_t {
  kQp = 0,
  kPort = 1,
  kDevice = 2,
  kCq = 3,
};

// A failure as submitted: the event a consumer sees, plus how to scope it.
struct PeerFailureReport {
  PeerFailureEvent event;
  FailureScope scope{FailureScope::kQp};
  // Opaque ibv_cq* for kCq reports. void* keeps this header verbs-free.
  const void* cqHandle{nullptr};
  // IBV_EVENT_PORT_ACTIVE: clears the matching port failure instead of adding one.
  bool recovery{false};
};

// Invoked on the async-event monitor thread. Must not block or re-enter the
// engine; record the event so an application thread can pick it up later.
using PeerFailureCallback = std::function<void(const PeerFailureReport&)>;

// Records asynchronous peer failures and answers QP liveness; thread-safe. A QP
// is dead only once a failure covering it is observed, so silence means alive.
class PeerFailureTracker {
 public:
  // Bound on unclaimed failures. Overflow drops the newest and counts it, since
  // the oldest events are the diagnostic ones.
  static constexpr size_t kMaxPending = 1024;

  // Identity of a local QP. QP numbers are per device, so a bare qpn would alias.
  struct QpKey {
    std::string deviceName;
    uint32_t qpNum{0};

    bool operator==(const QpKey& rhs) const {
      return qpNum == rhs.qpNum && deviceName == rhs.deviceName;
    }
  };

  // Ties a QP to its peer, port and CQ, so a failure of any of them resolves here.
  void RegisterQp(const QpKey& qp, std::string remoteEngineKey, uint32_t portNum,
                  const void* cqHandle);
  // Drops a QP association. Any recorded failure for it is retained, so a
  // pending event stays drainable after the endpoint is gone.
  void ForgetQp(const QpKey& qp);

  // Records a failure, or clears one on recovery. Repeats fold into the first.
  void Report(const PeerFailureReport& report);

  // Takes the oldest pending failure. False when none is pending.
  bool Pop(PeerFailureEvent* out);

  // False once this QP, or the port, CQ or device it uses, has been seen to fail.
  bool IsQpAlive(const QpKey& qp) const;

  // Number of failures dropped because the pending queue was full.
  uint64_t Dropped() const;

  // Pending, not-yet-drained failure count. For tests and diagnostics.
  size_t PendingCount() const;

 private:
  struct PortKey {
    std::string deviceName;
    uint32_t portNum{0};

    bool operator==(const PortKey& rhs) const {
      return portNum == rhs.portNum && deviceName == rhs.deviceName;
    }
  };

  struct CqKey {
    std::string deviceName;
    const void* cqHandle{nullptr};

    bool operator==(const CqKey& rhs) const {
      return cqHandle == rhs.cqHandle && deviceName == rhs.deviceName;
    }
  };

  // Mixes in the device name so same-numbered resources on other NICs differ.
  static size_t MixHash(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
  }

  struct QpKeyHash {
    size_t operator()(const QpKey& key) const {
      return MixHash(std::hash<std::string>()(key.deviceName), key.qpNum);
    }
  };

  struct PortKeyHash {
    size_t operator()(const PortKey& key) const {
      return MixHash(std::hash<std::string>()(key.deviceName), key.portNum);
    }
  };

  struct CqKeyHash {
    size_t operator()(const CqKey& key) const {
      return MixHash(std::hash<std::string>()(key.deviceName),
                     std::hash<const void*>()(key.cqHandle));
    }
  };

  struct QpOwner {
    std::string remoteEngineKey;
    uint32_t portNum{0};
    const void* cqHandle{nullptr};
  };

  // Appends to the queue, dropping and counting on overflow. Caller holds mu_.
  void Enqueue(PeerFailureEvent event);

  mutable std::mutex mu_;
  std::unordered_map<QpKey, QpOwner, QpKeyHash> qpOwners_;
  std::unordered_set<QpKey, QpKeyHash> failedQps_;
  std::unordered_set<PortKey, PortKeyHash> failedPorts_;
  std::unordered_set<CqKey, CqKeyHash> failedCqs_;
  std::unordered_set<std::string> failedDevices_;
  std::deque<PeerFailureEvent> pending_;
  uint64_t dropped_{0};
};

}  // namespace io
}  // namespace mori
