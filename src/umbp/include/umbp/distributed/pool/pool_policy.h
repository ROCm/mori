// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"

namespace mori::umbp {

// Policy input for one new logical block. Backend names are an explicit
// override used by topology-aware callers; tier-only requests preserve the
// legacy route contract.
struct PoolPlacementRequest {
  std::string key;
  uint64_t size = 0;
  TierType tier = TierType::UNKNOWN;
  std::string backend_name;
};

// Decision-only interface. A policy returns backend ids; PeerPool owns slot
// lifecycle and data movement. Policy implementations must not call a backend
// or TransferEngine.
class PoolPolicy {
 public:
  virtual ~PoolPolicy() = default;

  virtual std::optional<uint32_t> SelectPutBackend(
      const BackendRegistry& backends, const PoolPlacementRequest& request) const = 0;

  virtual std::vector<uint32_t> ReadOrder(const BackendRegistry& backends) const = 0;
};

// Compatibility policy for the implicit default pool. An explicit backend name
// selects that instance; otherwise the first instance of the routed TierType is
// used, exactly matching the pre-Pool behavior.
class SingleBackendPolicy final : public PoolPolicy {
 public:
  std::optional<uint32_t> SelectPutBackend(
      const BackendRegistry& backends, const PoolPlacementRequest& request) const override {
    MediumBackend* backend = nullptr;
    if (!request.backend_name.empty()) {
      backend = backends.Get(request.backend_name);
      if (backend != nullptr && backend->Tier() != request.tier) backend = nullptr;
    } else {
      backend = backends.Get(request.tier);
    }
    if (backend == nullptr) return std::nullopt;
    const uint32_t id = backends.BackendId(backend);
    return id < BackendRegistry::kMaxBackends ? std::optional<uint32_t>{id} : std::nullopt;
  }

  std::vector<uint32_t> ReadOrder(const BackendRegistry& backends) const override {
    std::vector<uint32_t> order;
    order.reserve(backends.Size());
    for (auto* backend : backends.All()) {
      const uint32_t id = backends.BackendId(backend);
      if (id < BackendRegistry::kMaxBackends) order.push_back(id);
    }
    return order;
  }
};

inline std::unique_ptr<PoolPolicy> MakeSingleBackendPolicy() {
  return std::make_unique<SingleBackendPolicy>();
}

}  // namespace mori::umbp
