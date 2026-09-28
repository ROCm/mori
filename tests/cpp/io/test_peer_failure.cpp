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
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/peer_failure.hpp"

using namespace mori::io;

namespace {

struct TestFailure : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

void Require(bool cond, const std::string& msg) {
  if (!cond) throw TestFailure(msg);
}

PeerFailureEvent QpEvent(uint32_t qpNum, const std::string& device = "mlx5_0") {
  PeerFailureEvent event;
  event.reason = PeerFailureReason::QP_FATAL;
  event.qpNum = qpNum;
  event.deviceName = device;
  event.detail = "IBV_EVENT_QP_FATAL";
  return event;
}

PeerFailureEvent DeviceEvent(const std::string& device, PeerFailureReason reason) {
  PeerFailureEvent event;
  event.reason = reason;
  event.qpNum = 0;
  event.deviceName = device;
  return event;
}

// A tracker that has seen nothing reports everything alive. This is the property
// that keeps a slow peer from being mistaken for a dead one.
void CaseSilenceMeansAlive() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(10, "decode-0", "mlx5_0");

  Require(tracker.IsQpAlive(10), "registered QP with no events must be alive");
  Require(tracker.IsQpAlive(999), "unknown QP must be alive");

  PeerFailureEvent event;
  Require(!tracker.Pop(&event), "no failure should be pending");
  Require(tracker.PendingCount() == 0, "pending count must be zero");
}

// A QP-scoped failure kills exactly that QP and is attributed to its peer.
void CaseQpFailureIsAttributed() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(10, "decode-0", "mlx5_0");
  tracker.RegisterQp(11, "decode-1", "mlx5_1");

  tracker.Record(QpEvent(10));

  Require(!tracker.IsQpAlive(10), "failed QP must not be alive");
  Require(tracker.IsQpAlive(11), "unrelated QP must stay alive");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "failure should be pending");
  Require(event.remoteEngineKey == "decode-0",
          "failure must be attributed to the peer that owns the QP, got '" + event.remoteEngineKey +
              "'");
  Require(event.reason == PeerFailureReason::QP_FATAL, "reason must be preserved");
  Require(event.qpNum == 10, "qp number must be preserved");
  Require(!tracker.Pop(&event), "queue must be drained after one pop");
}

// An event for a QP we never registered is still reported, just unattributed.
// Losing the event would be worse than losing the peer's name.
void CaseUnknownQpStillReported() {
  PeerFailureTracker tracker;
  tracker.Record(QpEvent(77));

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "failure for unknown QP must still be reported");
  Require(event.remoteEngineKey.empty(), "unresolvable peer key must be left empty");
  Require(!tracker.IsQpAlive(77), "unknown QP that failed must be dead");
}

// Port and device failures are not tied to one QP, so they kill every QP on the
// device rather than guessing which peer was affected.
void CaseDeviceFailureAffectsWholeDevice() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(10, "decode-0", "mlx5_0");
  tracker.RegisterQp(11, "decode-1", "mlx5_0");
  tracker.RegisterQp(12, "decode-2", "mlx5_1");

  tracker.Record(DeviceEvent("mlx5_0", PeerFailureReason::PORT_DOWN));

  Require(!tracker.IsQpAlive(10), "QP on downed device must be dead");
  Require(!tracker.IsQpAlive(11), "second QP on downed device must be dead");
  Require(tracker.IsQpAlive(12), "QP on a healthy device must stay alive");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "device failure must be reported");
  Require(event.remoteEngineKey.empty(),
          "device-wide failure must not be attributed to a single peer");
  Require(event.reason == PeerFailureReason::PORT_DOWN, "reason must be preserved");
}

// A QP registered after its device already failed must not be reported alive:
// the device is still down regardless of when the endpoint was created.
void CaseQpRegisteredAfterDeviceFailure() {
  PeerFailureTracker tracker;
  tracker.Record(DeviceEvent("mlx5_0", PeerFailureReason::DEVICE_FATAL));
  tracker.RegisterQp(10, "decode-0", "mlx5_0");

  Require(!tracker.IsQpAlive(10), "QP on an already-failed device must be dead");
}

// Forgetting a QP must not resurrect it or discard its pending event.
void CaseForgetQpKeepsFailure() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(10, "decode-0", "mlx5_0");
  tracker.Record(QpEvent(10));
  tracker.ForgetQp(10);

  Require(!tracker.IsQpAlive(10), "forgetting a QP must not mark it alive again");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "pending failure must survive ForgetQp");
  Require(event.remoteEngineKey == "decode-0", "attribution happens at record time");
}

// Events are drained oldest-first so the first cause is seen before its fallout.
void CaseFifoOrder() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(10, "decode-0", "mlx5_0");
  tracker.RegisterQp(11, "decode-1", "mlx5_0");
  tracker.Record(QpEvent(10));
  tracker.Record(QpEvent(11));

  PeerFailureEvent first;
  PeerFailureEvent second;
  Require(tracker.Pop(&first) && tracker.Pop(&second), "both failures must be pending");
  Require(first.qpNum == 10 && second.qpNum == 11, "failures must drain in FIFO order");
}

// A caller that never drains must not grow the queue without bound. Overflow
// drops the newest and counts it, keeping the earliest diagnostic events.
void CaseBoundedQueueDropsNewest() {
  PeerFailureTracker tracker;
  const size_t overflow = 10;
  for (size_t i = 0; i < PeerFailureTracker::kMaxPending + overflow; ++i) {
    // QP numbers start at 1; 0 would be read as a device-scoped event.
    tracker.Record(QpEvent(static_cast<uint32_t>(i + 1)));
  }

  Require(tracker.PendingCount() == PeerFailureTracker::kMaxPending,
          "pending queue must be capped at kMaxPending");
  Require(tracker.Dropped() == overflow,
          "dropped count must equal the overflow, got " + std::to_string(tracker.Dropped()));

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "capped queue must still be drainable");
  Require(event.qpNum == 1, "the oldest event must be retained, not the newest");

  // Liveness must be unaffected by the queue bound: a dropped event still marks
  // its QP dead, otherwise overflow would silently resurrect peers.
  uint32_t droppedQp = static_cast<uint32_t>(PeerFailureTracker::kMaxPending + overflow);
  Require(!tracker.IsQpAlive(droppedQp), "a QP whose event was dropped must still be dead");
}

void CasePopNullptrIsSafe() {
  PeerFailureTracker tracker;
  tracker.Record(QpEvent(10));
  Require(!tracker.Pop(nullptr), "Pop(nullptr) must return false rather than crash");
}

// The monitor records from its own thread while the application drains from
// another; run both concurrently and require every event to be accounted for.
void CaseConcurrentRecordAndPop() {
  PeerFailureTracker tracker;
  const uint32_t total = 500;
  for (uint32_t qp = 1; qp <= total; ++qp) {
    tracker.RegisterQp(qp, "decode-0", "mlx5_0");
  }

  std::thread producer([&tracker] {
    for (uint32_t qp = 1; qp <= total; ++qp) tracker.Record(QpEvent(qp));
  });

  uint32_t drained = 0;
  std::thread consumer([&tracker, &drained] {
    PeerFailureEvent event;
    while (drained < total) {
      if (tracker.Pop(&event)) {
        drained++;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  Require(drained == total, "every recorded failure must be drainable exactly once");
  Require(tracker.Dropped() == 0, "a draining consumer must prevent drops");
  for (uint32_t qp = 1; qp <= total; ++qp) {
    Require(!tracker.IsQpAlive(qp), "every failed QP must be dead after the run");
  }
}

struct TestCase {
  const char* name;
  std::function<void()> run;
};

}  // namespace

int main() {
  std::vector<TestCase> cases = {
      {"SilenceMeansAlive", CaseSilenceMeansAlive},
      {"QpFailureIsAttributed", CaseQpFailureIsAttributed},
      {"UnknownQpStillReported", CaseUnknownQpStillReported},
      {"DeviceFailureAffectsWholeDevice", CaseDeviceFailureAffectsWholeDevice},
      {"QpRegisteredAfterDeviceFailure", CaseQpRegisteredAfterDeviceFailure},
      {"ForgetQpKeepsFailure", CaseForgetQpKeepsFailure},
      {"FifoOrder", CaseFifoOrder},
      {"BoundedQueueDropsNewest", CaseBoundedQueueDropsNewest},
      {"PopNullptrIsSafe", CasePopNullptrIsSafe},
      {"ConcurrentRecordAndPop", CaseConcurrentRecordAndPop},
  };

  for (const TestCase& test : cases) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << "\n";
    } catch (const std::exception& e) {
      std::cerr << "[FAIL] " << test.name << ": " << e.what() << "\n";
      return 1;
    }
  }
  return 0;
}
