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
//
// MIT License
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"

namespace mori::umbp {

enum class TierTransitionKind {
  kOffload,
  kPromotion,
};

struct TierTransitionResult {
  bool success = false;
  uint32_t target_backend_id = BackendRegistry::kMaxBackends;
  uint64_t bytes_moved = 0;
  uint64_t bytes_freed = 0;
  bool source_draining = false;
};

// Executes one peer-local copy or move. Placement policy and scheduling remain
// in PeerPool; this class owns only the transactional byte-moving sequence.
class TierTransitionExecutor {
 public:
  TierTransitionExecutor(BackendRegistry* backends, TransferEngine* transfer_engine)
      : backends_(backends), transfer_engine_(transfer_engine) {}

  TierTransitionResult Execute(const std::string& key, uint32_t source_backend_id,
                               const std::vector<uint32_t>& target_backend_ids,
                               bool remove_source) const;

 private:
  static bool BuildTransfers(MediumBackend* source, const ResolvedEntry& resolved,
                             MediumBackend* target, const AllocateResult& allocation,
                             std::vector<TransferItem>* items);

  BackendRegistry* backends_;
  TransferEngine* transfer_engine_;
};

}  // namespace mori::umbp
