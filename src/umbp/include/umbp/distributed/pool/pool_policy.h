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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool/tier_graph.h"

namespace mori::umbp {

// Policy input for one new logical block. Backend names are an explicit
// override used by topology-aware callers; tier-only requests preserve the
// legacy route contract.
struct PoolPlacementRequest {
  std::string key;
  uint64_t size = 0;
  TierType tier = TierType::UNKNOWN;
  std::string backend_name;
  std::string logical_tier;
};

// Decision-only interface. A policy returns backend ids; PeerPool owns slot
// lifecycle and data movement. Policy implementations must not call a backend
// or TransferEngine.
class PoolPolicy {
 public:
  virtual ~PoolPolicy() = default;

  // Ordered candidates for one placement attempt, best first. A single-choice
  // policy returns one id; others append deterministic no-space fallbacks.
  virtual std::vector<uint32_t> PutOrder(const BackendRegistry& backends,
                                         const PoolPlacementRequest& request) const = 0;

  virtual std::vector<uint32_t> ReadOrder(const BackendRegistry& backends) const = 0;
  virtual std::shared_ptr<const LogicalTierGraph> TierGraph() const { return {}; }
};

// Every backend the peer holds, in registry order. Shared by the policies that
// have no read preference of their own.
inline std::vector<uint32_t> AllBackendIds(const BackendRegistry& backends) {
  std::vector<uint32_t> order;
  order.reserve(backends.Size());
  for (auto* backend : backends.All()) {
    const uint32_t id = backends.BackendId(backend);
    if (id < BackendRegistry::kMaxBackends) order.push_back(id);
  }
  return order;
}

// Compatibility policy for the implicit default pool. An explicit backend name
// selects that instance; otherwise the first instance of the routed TierType is
// used, exactly matching the pre-Pool behavior.
class SingleBackendPolicy final : public PoolPolicy {
 public:
  std::vector<uint32_t> PutOrder(const BackendRegistry& backends,
                                 const PoolPlacementRequest& request) const override {
    MediumBackend* backend = nullptr;
    if (!request.backend_name.empty()) {
      backend = backends.Get(request.backend_name);
      if (backend != nullptr && backend->Tier() != request.tier) backend = nullptr;
    } else {
      backend = backends.Get(request.tier);
    }
    if (backend == nullptr) return {};
    const uint32_t id = backends.BackendId(backend);
    return id < BackendRegistry::kMaxBackends ? std::vector<uint32_t>{id} : std::vector<uint32_t>{};
  }

  std::vector<uint32_t> ReadOrder(const BackendRegistry& backends) const override {
    return AllBackendIds(backends);
  }
};

inline std::unique_ptr<PoolPolicy> MakeSingleBackendPolicy() {
  return std::make_unique<SingleBackendPolicy>();
}

struct BackendPlacementWeight {
  std::string backend_name;
  uint32_t weight = 1;
};

// Deterministic weighted placement among configured instances of the requested
// tier. Hashing the logical key makes retries select the same backend without
// shared counters and gives stable distribution independent of request order.
class WeightedPlacementPolicy final : public PoolPolicy {
 public:
  explicit WeightedPlacementPolicy(std::vector<BackendPlacementWeight> weights)
      : weights_(std::move(weights)) {}

  std::vector<uint32_t> PutOrder(const BackendRegistry& backends,
                                 const PoolPlacementRequest& request) const override {
    if (!request.backend_name.empty()) {
      auto* backend = backends.Get(request.backend_name);
      if (backend == nullptr || backend->Tier() != request.tier) return {};
      const uint32_t id = backends.BackendId(backend);
      return id < BackendRegistry::kMaxBackends ? std::vector<uint32_t>{id}
                                                : std::vector<uint32_t>{};
    }

    std::vector<LogicalTierBackendConfig> candidates;
    candidates.reserve(weights_.size());
    for (const auto& configured : weights_) {
      auto* backend = backends.Get(configured.backend_name);
      if (backend == nullptr || backend->Tier() != request.tier) continue;
      candidates.push_back({configured.backend_name, configured.weight});
    }
    return WeightedBackendOrder(candidates, backends, request.key);
  }

  std::vector<uint32_t> ReadOrder(const BackendRegistry& backends) const override {
    return AllBackendIds(backends);
  }

 private:
  std::vector<BackendPlacementWeight> weights_;
};

inline std::unique_ptr<PoolPolicy> MakeWeightedPlacementPolicy(
    std::vector<BackendPlacementWeight> weights) {
  return std::make_unique<WeightedPlacementPolicy>(std::move(weights));
}

class TieredPlacementPolicy final : public PoolPolicy {
 public:
  explicit TieredPlacementPolicy(std::shared_ptr<const LogicalTierGraph> graph)
      : graph_(std::move(graph)) {}

  std::vector<uint32_t> PutOrder(const BackendRegistry& backends,
                                 const PoolPlacementRequest& request) const override {
    if (!request.backend_name.empty()) {
      auto* backend = backends.Get(request.backend_name);
      if (backend == nullptr) return {};
      const uint32_t id = backends.BackendId(backend);
      return id < BackendRegistry::kMaxBackends ? std::vector<uint32_t>{id}
                                                : std::vector<uint32_t>{};
    }
    if (graph_ == nullptr) return {};
    return request.logical_tier.empty()
               ? graph_->PutOrder(request.key)
               : graph_->PutOrderFromTier(request.logical_tier, request.key);
  }

  std::vector<uint32_t> ReadOrder(const BackendRegistry&) const override {
    return graph_ == nullptr ? std::vector<uint32_t>{} : graph_->ReadOrder();
  }
  std::shared_ptr<const LogicalTierGraph> TierGraph() const override { return graph_; }

 private:
  std::shared_ptr<const LogicalTierGraph> graph_;
};

inline std::unique_ptr<PoolPolicy> MakeTieredPlacementPolicy(
    std::shared_ptr<const LogicalTierGraph> graph) {
  return std::make_unique<TieredPlacementPolicy>(std::move(graph));
}

}  // namespace mori::umbp
