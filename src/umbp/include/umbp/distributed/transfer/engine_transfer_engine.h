// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp::engine {
class EngineExecutor;
}

namespace mori::umbp {

// TransferEngine adapter for refs owned by one rank executor. Node code routes
// an executor plan over EngineSession; the rank uses this same adapter against
// its local EngineExecutor. It never dereferences a foreign executor address.
class EngineTransferEngine final : public TransferEngine {
 public:
  explicit EngineTransferEngine(engine::EngineExecutor* executor = nullptr) : executor_(executor) {}

  const char* Name() const override { return "EngineTransferEngine"; }
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override;
  void Deregister(const TransferRef&) override {}
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items = 0) const override;
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;

 private:
  engine::EngineExecutor* executor_ = nullptr;
};

}  // namespace mori::umbp
