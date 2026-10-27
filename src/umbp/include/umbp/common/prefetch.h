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

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mori::umbp {

// Values 0..6 match the deployed Router's PrefetchState wire enum.
enum class PrefetchState { kUnknown, kQueued, kRunning, kReady, kPartial, kFailed, kExpired };
enum class PrefetchCode {
  kPending,
  kReady,
  kMissing,
  kNoDram,
  kNoSpace,
  kTransferFailed,
  kFragmented,
  kNotResident,
  kExpired,
  kStopped
};
enum class PrefetchSource { kUnknown, kDram, kSsd, kRemote };

struct PrefetchOptions {
  std::chrono::milliseconds lease_ttl{500};
  std::chrono::milliseconds timeout{5000};
};

struct PrefetchKeyResult {
  PrefetchCode code = PrefetchCode::kPending;
  PrefetchSource source = PrefetchSource::kUnknown;
  uint32_t backend_id = 0;
  uint64_t bytes = 0;
  bool Ready() const { return code == PrefetchCode::kReady; }
};

struct PrefetchSubmitResult {
  bool accepted = false;
  std::string request_id;
  std::string error;
};

struct PrefetchStatus {
  PrefetchState state = PrefetchState::kUnknown;
  std::vector<PrefetchKeyResult> results;
  uint64_t completed_keys = 0;
  std::string error;
  uint64_t queue_latency_us = 0;
  uint64_t execution_latency_us = 0;
  uint64_t total_latency_us = 0;
  // Remaining guaranteed protection, sampled at query time. A historical
  // READY with zero remaining lease does not promise continued residency.
  uint64_t lease_remaining_ms = 0;
};

inline bool PrefetchTerminal(PrefetchState state) {
  return state == PrefetchState::kReady || state == PrefetchState::kPartial ||
         state == PrefetchState::kFailed || state == PrefetchState::kExpired;
}

}  // namespace mori::umbp
