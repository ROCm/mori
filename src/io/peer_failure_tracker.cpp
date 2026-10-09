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
#include <utility>

#include "mori/io/peer_failure.hpp"

namespace mori {
namespace io {

void PeerFailureTracker::RegisterQp(const QpKey& qp, std::string remoteEngineKey, uint32_t portNum,
                                    const void* cqHandle) {
  std::lock_guard<std::mutex> lock(mu_);
  qpOwners_[qp] = QpOwner{std::move(remoteEngineKey), portNum, cqHandle};
}

void PeerFailureTracker::ForgetQp(const QpKey& qp) {
  std::lock_guard<std::mutex> lock(mu_);
  qpOwners_.erase(qp);
}

void PeerFailureTracker::Enqueue(PeerFailureEvent event) {
  if (pending_.size() >= kMaxPending) {
    dropped_++;
    return;
  }
  pending_.push_back(std::move(event));
}

void PeerFailureTracker::Report(const PeerFailureReport& report) {
  PeerFailureEvent event = report.event;
  std::lock_guard<std::mutex> lock(mu_);

  if (report.recovery) {
    // Withdraws the port-wide verdict so sessions that died only to a link flap
    // work again. QPs that failed in their own right stay failed: a QP in the
    // error state does not heal.
    failedPorts_.erase(PortKey{event.deviceName, event.portNum});
    return;
  }

  bool firstForResource = false;
  switch (report.scope) {
    case FailureScope::kQp: {
      QpKey key{event.deviceName, event.qpNum};
      firstForResource = failedQps_.insert(key).second;
      auto it = qpOwners_.find(key);
      if (it != qpOwners_.end()) {
        // Attribution from the reporter wins; the registry only fills in what
        // the event could not carry.
        if (event.remoteEngineKey.empty()) event.remoteEngineKey = it->second.remoteEngineKey;
        if (event.portNum == 0) event.portNum = it->second.portNum;
      }
      break;
    }
    case FailureScope::kPort:
      firstForResource = failedPorts_.insert(PortKey{event.deviceName, event.portNum}).second;
      break;
    case FailureScope::kDevice:
      firstForResource = failedDevices_.insert(event.deviceName).second;
      break;
    case FailureScope::kCq:
      firstForResource = failedCqs_.insert(CqKey{event.deviceName, report.cqHandle}).second;
      break;
  }

  // A repeat for a resource already known bad adds nothing, and a CQE error
  // burst would otherwise fill the queue with copies of one failure.
  if (!firstForResource) return;
  Enqueue(std::move(event));
}

bool PeerFailureTracker::Pop(PeerFailureEvent* out) {
  if (out == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  if (pending_.empty()) return false;
  *out = std::move(pending_.front());
  pending_.pop_front();
  return true;
}

bool PeerFailureTracker::IsQpAlive(const QpKey& qp) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (failedQps_.count(qp) != 0) return false;
  // Device-wide, so it applies whether or not this QP was ever registered.
  if (failedDevices_.count(qp.deviceName) != 0) return false;

  // An unregistered QP cannot be tied to a port or a CQ, so nothing else here
  // can condemn it.
  auto it = qpOwners_.find(qp);
  if (it == qpOwners_.end()) return true;

  if (failedPorts_.count(PortKey{qp.deviceName, it->second.portNum}) != 0) return false;
  if (it->second.cqHandle != nullptr &&
      failedCqs_.count(CqKey{qp.deviceName, it->second.cqHandle}) != 0) {
    return false;
  }
  return true;
}

uint64_t PeerFailureTracker::Dropped() const {
  std::lock_guard<std::mutex> lock(mu_);
  return dropped_;
}

size_t PeerFailureTracker::PendingCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  return pending_.size();
}

}  // namespace io
}  // namespace mori
