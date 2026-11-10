// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <optional>
#include <vector>

#include "umbp/distributed/peer/backend/engine_hbm_backend.h"
#include "umbp/engine/engine_types.h"

namespace mori::umbp {

class PeerPool;

namespace engine {

// Narrow, state-free view of the existing PoolClient used by the standalone
// engine control channel.  Implementations must not create another PeerPool or
// another backend registry: the fixed engine backends in the compiled graph
// remain the only ownership authority.
class EngineControl {
 public:
  virtual ~EngineControl() = default;

  virtual bool RegisterEngine(EngineRegistration registration) = 0;
  virtual std::optional<uint32_t> BackendId(RankId rank) const = 0;
  virtual std::vector<SharedDramRegion> ExportSharedDram(PoolEpoch epoch) = 0;
  virtual PeerPool* Pool() const = 0;
  virtual ReadPlan Lookup(const std::vector<std::string>& keys, std::string name_space,
                          std::string layout_id, uint32_t requester_backend_id) = 0;
  virtual PreparedLoad PrepareLoad(const ReadPlan& plan, std::string request_id,
                                   const std::vector<AllocationHandle>& targets, uint32_t n_pages,
                                   const ReaderIdentity& reader, OperationId operation) = 0;
  virtual CancelLoadResult CancelLoad(uint64_t prepared_id, OperationId operation) = 0;
  virtual bool CompleteLoad(uint64_t prepared_id, LoadOutcome outcome, OperationId operation) = 0;
  virtual void SessionFailed(RankId rank) = 0;
};

}  // namespace engine
}  // namespace mori::umbp
