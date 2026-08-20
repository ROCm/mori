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
