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

using QpKey = PeerFailureTracker::QpKey;

constexpr uint32_t kPort1 = 1;
constexpr uint32_t kPort2 = 2;

// Stand-ins for ibv_cq*; the tracker only ever compares them.
void* const kCqA = reinterpret_cast<void*>(0xA000);
void* const kCqB = reinterpret_cast<void*>(0xB000);

QpKey Qp(uint32_t qpNum, const std::string& device = "mlx5_0") { return QpKey{device, qpNum}; }

PeerFailureReport QpFailure(uint32_t qpNum, const std::string& device = "mlx5_0") {
  PeerFailureReport report;
  report.scope = FailureScope::kQp;
  report.event.reason = PeerFailureReason::LOCAL_QP_ERROR;
  report.event.qpNum = qpNum;
  report.event.deviceName = device;
  report.event.detail = "IBV_EVENT_QP_FATAL";
  return report;
}

PeerFailureReport PeerUnreachable(uint32_t qpNum, const std::string& remoteEngineKey,
                                  const std::string& device = "mlx5_0") {
  PeerFailureReport report = QpFailure(qpNum, device);
  report.event.reason = PeerFailureReason::PEER_UNREACHABLE;
  report.event.remoteEngineKey = remoteEngineKey;
  report.event.detail = "completion status transport retry counter exceeded";
  return report;
}

PeerFailureReport PortFailure(const std::string& device, uint32_t portNum) {
  PeerFailureReport report;
  report.scope = FailureScope::kPort;
  report.event.reason = PeerFailureReason::LOCAL_PORT_DOWN;
  report.event.deviceName = device;
  report.event.portNum = portNum;
  return report;
}

PeerFailureReport PortRecovery(const std::string& device, uint32_t portNum) {
  PeerFailureReport report = PortFailure(device, portNum);
  report.recovery = true;
  return report;
}

PeerFailureReport DeviceFatal(const std::string& device) {
  PeerFailureReport report;
  report.scope = FailureScope::kDevice;
  report.event.reason = PeerFailureReason::LOCAL_DEVICE_FATAL;
  report.event.deviceName = device;
  return report;
}

PeerFailureReport CqFailure(const std::string& device, const void* cq) {
  PeerFailureReport report;
  report.scope = FailureScope::kCq;
  report.event.reason = PeerFailureReason::LOCAL_CQ_ERROR;
  report.event.deviceName = device;
  report.cqHandle = cq;
  return report;
}

// A tracker that has seen nothing reports everything alive. This is the property
// that keeps a slow peer from being mistaken for a dead one.
void CaseSilenceMeansAlive() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);

  Require(tracker.IsQpAlive(Qp(10)), "registered QP with no events must be alive");
  Require(tracker.IsQpAlive(Qp(999)), "unknown QP must be alive");

  PeerFailureEvent event;
  Require(!tracker.Pop(&event), "no failure should be pending");
  Require(tracker.PendingCount() == 0, "pending count must be zero");
}

// A QP-scoped failure kills exactly that QP and is attributed to its peer.
void CaseQpFailureIsAttributed() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11), "decode-1", kPort1, kCqB);

  tracker.Report(QpFailure(10));

  Require(!tracker.IsQpAlive(Qp(10)), "failed QP must not be alive");
  Require(tracker.IsQpAlive(Qp(11)), "unrelated QP must stay alive");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "failure should be pending");
  Require(event.remoteEngineKey == "decode-0",
          "failure must be attributed to the peer that owns the QP, got '" + event.remoteEngineKey +
              "'");
  Require(event.reason == PeerFailureReason::LOCAL_QP_ERROR, "reason must be preserved");
  Require(event.qpNum == 10, "qp number must be preserved");
  Require(event.portNum == kPort1, "port must be filled in from the QP's registration");
  Require(!tracker.Pop(&event), "queue must be drained after one pop");
}

// An event for a QP we never registered is still reported, just unattributed.
// Losing the event would be worse than losing the peer's name.
void CaseUnknownQpStillReported() {
  PeerFailureTracker tracker;
  tracker.Report(QpFailure(77));

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "failure for unknown QP must still be reported");
  Require(event.remoteEngineKey.empty(), "unresolvable peer key must be left empty");
  Require(!tracker.IsQpAlive(Qp(77)), "unknown QP that failed must be dead");
}

// A failure on one NIC must not touch the same-numbered QP on another.
void CaseQpNumbersAreScopedPerDevice() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(2048, "ionic_0"), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(2048, "ionic_1"), "decode-1", kPort1, kCqB);

  tracker.Report(QpFailure(2048, "ionic_0"));

  Require(!tracker.IsQpAlive(Qp(2048, "ionic_0")), "failed QP must be dead on its own device");
  Require(tracker.IsQpAlive(Qp(2048, "ionic_1")),
          "a QP sharing a number on another NIC must stay alive");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "failure should be pending");
  Require(
      event.remoteEngineKey == "decode-0",
      "attribution must pick the owner on the failing device, got '" + event.remoteEngineKey + "'");
}

// The HCA is gone, so every QP on it dies, including unregistered ones.
void CaseDeviceFatalAffectsWholeDevice() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10, "mlx5_0"), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11, "mlx5_0"), "decode-1", kPort2, kCqB);
  tracker.RegisterQp(Qp(12, "mlx5_1"), "decode-2", kPort1, kCqA);

  tracker.Report(DeviceFatal("mlx5_0"));

  Require(!tracker.IsQpAlive(Qp(10, "mlx5_0")), "QP on the dead device must be dead");
  Require(!tracker.IsQpAlive(Qp(11, "mlx5_0")), "second QP on the dead device must be dead");
  Require(!tracker.IsQpAlive(Qp(99, "mlx5_0")), "unregistered QP on the dead device must be dead");
  Require(tracker.IsQpAlive(Qp(12, "mlx5_1")), "QP on a healthy device must stay alive");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "device failure must be reported");
  Require(event.remoteEngineKey.empty(),
          "device-wide failure must not be attributed to a single peer");
  Require(event.reason == PeerFailureReason::LOCAL_DEVICE_FATAL, "reason must be preserved");
}

// Condemning the whole NIC would kill healthy sessions on its other ports.
void CasePortFailureScopedToPort() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11), "decode-1", kPort2, kCqA);

  tracker.Report(PortFailure("mlx5_0", kPort1));

  Require(!tracker.IsQpAlive(Qp(10)), "QP on the downed port must be dead");
  Require(tracker.IsQpAlive(Qp(11)), "QP on another port of the same device must stay alive");
}

// A link flap must not condemn the NIC for the rest of the process.
void CasePortRecoveryRevivesSessions() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);

  tracker.Report(PortFailure("mlx5_0", kPort1));
  Require(!tracker.IsQpAlive(Qp(10)), "QP on the downed port must be dead while it is down");

  tracker.RegisterQp(Qp(11), "decode-1", kPort1, kCqA);
  Require(!tracker.IsQpAlive(Qp(11)), "a QP created while the port is down must not be alive");

  tracker.Report(PortRecovery("mlx5_0", kPort1));

  Require(tracker.IsQpAlive(Qp(10)), "port recovery must revive QPs the port had condemned");
  Require(tracker.IsQpAlive(Qp(11)), "port recovery must revive QPs created while it was down");

  // Recovery withdraws the verdict, not the history.
  PeerFailureEvent event;
  Require(tracker.Pop(&event), "the port failure must still be reported after recovery");
  Require(event.reason == PeerFailureReason::LOCAL_PORT_DOWN, "reason must be preserved");
  Require(!tracker.Pop(&event), "recovery itself must not enqueue an event");
}

// A QP in the error state stays there until it is torn down and rebuilt.
void CaseQpFailureSurvivesPortRecovery() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11), "decode-1", kPort1, kCqA);

  tracker.Report(QpFailure(10));
  tracker.Report(PortFailure("mlx5_0", kPort1));
  tracker.Report(PortRecovery("mlx5_0", kPort1));

  Require(!tracker.IsQpAlive(Qp(10)), "an individually failed QP must not be revived by recovery");
  Require(tracker.IsQpAlive(Qp(11)), "a QP condemned only by the port must be revived");
}

// A CQ error loses completions for the QPs sharing that CQ and no others.
void CaseCqFailureScopedToCq() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11), "decode-1", kPort1, kCqB);
  tracker.RegisterQp(Qp(12, "mlx5_1"), "decode-2", kPort1, kCqA);

  tracker.Report(CqFailure("mlx5_0", kCqA));

  Require(!tracker.IsQpAlive(Qp(10)), "QP on the failed CQ must be dead");
  Require(tracker.IsQpAlive(Qp(11)), "QP on another CQ of the same device must stay alive");
  Require(tracker.IsQpAlive(Qp(12, "mlx5_1")), "a CQ pointer is only meaningful within its device");
}

// The reporter already knows the peer, so its attribution is kept.
void CasePeerUnreachableIsAttributed() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "prefill-0", kPort1, kCqA);

  tracker.Report(PeerUnreachable(10, "prefill-0"));

  Require(!tracker.IsQpAlive(Qp(10)), "a peer that stopped acknowledging must not be alive");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "peer unreachable must be reported");
  Require(event.reason == PeerFailureReason::PEER_UNREACHABLE,
          "the reason must distinguish peer death from a local fault");
  Require(event.remoteEngineKey == "prefill-0", "the reporter's attribution must be kept");
}

// One dead peer must not flush every other pending failure out of the queue.
void CaseRepeatFailuresFoldIntoFirst() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11), "decode-1", kPort1, kCqA);

  for (int i = 0; i < 100; ++i) tracker.Report(PeerUnreachable(10, "decode-0"));
  tracker.Report(QpFailure(11));

  Require(tracker.PendingCount() == 2,
          "repeats for one QP must collapse, got " + std::to_string(tracker.PendingCount()));
  Require(tracker.Dropped() == 0, "folding repeats is not dropping them");

  PeerFailureEvent event;
  Require(tracker.Pop(&event) && event.qpNum == 10, "the first report must be the one kept");
  Require(tracker.Pop(&event) && event.qpNum == 11, "an unrelated QP must still be reported");
}

// Attribution is at record time, which is why ConnectEndpoint registers early.
void CaseLateRegistrationCannotAttribute() {
  PeerFailureTracker tracker;
  tracker.Report(QpFailure(10));
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "the failure must still be reported");
  Require(event.remoteEngineKey.empty(),
          "registering after the fact must not retroactively attribute, got '" +
              event.remoteEngineKey + "'");
}

// Forgetting a QP must not resurrect it or discard its pending event.
void CaseForgetQpKeepsFailure() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.Report(QpFailure(10));
  tracker.ForgetQp(Qp(10));

  Require(!tracker.IsQpAlive(Qp(10)), "forgetting a QP must not mark it alive again");

  PeerFailureEvent event;
  Require(tracker.Pop(&event), "pending failure must survive ForgetQp");
  Require(event.remoteEngineKey == "decode-0", "attribution happens at record time");
}

// Events are drained oldest-first so the first cause is seen before its fallout.
void CaseFifoOrder() {
  PeerFailureTracker tracker;
  tracker.RegisterQp(Qp(10), "decode-0", kPort1, kCqA);
  tracker.RegisterQp(Qp(11), "decode-1", kPort1, kCqA);
  tracker.Report(QpFailure(10));
  tracker.Report(QpFailure(11));

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
    tracker.Report(QpFailure(static_cast<uint32_t>(i + 1)));
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
  Require(!tracker.IsQpAlive(Qp(droppedQp)), "a QP whose event was dropped must still be dead");
}

void CasePopNullptrIsSafe() {
  PeerFailureTracker tracker;
  tracker.Report(QpFailure(10));
  Require(!tracker.Pop(nullptr), "Pop(nullptr) must return false rather than crash");
}

// The monitor records from its own thread while the application drains from
// another; run both concurrently and require every event to be accounted for.
void CaseConcurrentRecordAndPop() {
  PeerFailureTracker tracker;
  const uint32_t total = 500;
  for (uint32_t qp = 1; qp <= total; ++qp) {
    tracker.RegisterQp(Qp(qp), "decode-0", kPort1, kCqA);
  }

  std::thread producer([&tracker] {
    for (uint32_t qp = 1; qp <= total; ++qp) tracker.Report(QpFailure(qp));
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
    Require(!tracker.IsQpAlive(Qp(qp)), "every failed QP must be dead after the run");
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
      {"QpNumbersAreScopedPerDevice", CaseQpNumbersAreScopedPerDevice},
      {"DeviceFatalAffectsWholeDevice", CaseDeviceFatalAffectsWholeDevice},
      {"PortFailureScopedToPort", CasePortFailureScopedToPort},
      {"PortRecoveryRevivesSessions", CasePortRecoveryRevivesSessions},
      {"QpFailureSurvivesPortRecovery", CaseQpFailureSurvivesPortRecovery},
      {"CqFailureScopedToCq", CaseCqFailureScopedToCq},
      {"PeerUnreachableIsAttributed", CasePeerUnreachableIsAttributed},
      {"RepeatFailuresFoldIntoFirst", CaseRepeatFailuresFoldIntoFirst},
      {"LateRegistrationCannotAttribute", CaseLateRegistrationCannotAttribute},
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
