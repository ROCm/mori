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

void PeerFailureTracker::RegisterQp(uint32_t qpNum, std::string remoteEngineKey,
                                    std::string deviceName) {
  std::lock_guard<std::mutex> lock(mu_);
  qpOwners_[qpNum] = QpOwner{std::move(remoteEngineKey), std::move(deviceName)};
}

void PeerFailureTracker::ForgetQp(uint32_t qpNum) {
  std::lock_guard<std::mutex> lock(mu_);
  qpOwners_.erase(qpNum);
}

void PeerFailureTracker::Record(PeerFailureEvent event) {
  std::lock_guard<std::mutex> lock(mu_);

  if (event.qpNum != 0) {
    failedQpns_.insert(event.qpNum);
    auto it = qpOwners_.find(event.qpNum);
    if (it != qpOwners_.end()) {
      event.remoteEngineKey = it->second.remoteEngineKey;
      // Fall back to the device the QP was created on, so the field is populated
      // even when the monitor could not name it.
      if (event.deviceName.empty()) event.deviceName = it->second.deviceName;
    }
  } else if (!event.deviceName.empty()) {
    failedDevices_.insert(event.deviceName);
  }

  if (pending_.size() >= kMaxPending) {
    dropped_++;
    return;
  }
  pending_.push_back(std::move(event));
}

bool PeerFailureTracker::Pop(PeerFailureEvent* out) {
  if (out == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  if (pending_.empty()) return false;
  *out = std::move(pending_.front());
  pending_.pop_front();
  return true;
}

bool PeerFailureTracker::IsQpAlive(uint32_t qpNum) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (failedQpns_.count(qpNum) != 0) return false;
  if (failedDevices_.empty()) return true;
  auto it = qpOwners_.find(qpNum);
  // An unregistered QP cannot be tied to a failed device, so it stays alive.
  if (it == qpOwners_.end()) return true;
  return failedDevices_.count(it->second.deviceName) == 0;
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
