// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
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
