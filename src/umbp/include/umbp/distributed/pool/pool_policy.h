// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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

  // Ordered candidates for one placement attempt. The default preserves the
  // single-choice contract; policies may add deterministic no-space fallbacks.
  virtual std::vector<uint32_t> PutOrder(
      const BackendRegistry& backends, const PoolPlacementRequest& request) const {
    auto selected = SelectPutBackend(backends, request);
    return selected.has_value() ? std::vector<uint32_t>{*selected}
                                : std::vector<uint32_t>{};
  }

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

  std::optional<uint32_t> SelectPutBackend(
      const BackendRegistry& backends, const PoolPlacementRequest& request) const override {
    auto order = PutOrder(backends, request);
    return order.empty() ? std::nullopt : std::optional<uint32_t>{order.front()};
  }

  std::vector<uint32_t> PutOrder(
      const BackendRegistry& backends, const PoolPlacementRequest& request) const override {
    if (!request.backend_name.empty()) {
      auto* backend = backends.Get(request.backend_name);
      if (backend == nullptr || backend->Tier() != request.tier) return {};
      const uint32_t id = backends.BackendId(backend);
      return id < BackendRegistry::kMaxBackends ? std::vector<uint32_t>{id}
                                                : std::vector<uint32_t>{};
    }

    struct Candidate {
      uint32_t backend_id;
      uint32_t weight;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(weights_.size());
    uint64_t total_weight = 0;
    for (const auto& configured : weights_) {
      if (configured.weight == 0) continue;
      auto* backend = backends.Get(configured.backend_name);
      if (backend == nullptr || backend->Tier() != request.tier) continue;
      const uint32_t id = backends.BackendId(backend);
      if (id >= BackendRegistry::kMaxBackends) continue;
      candidates.push_back(Candidate{id, configured.weight});
      total_weight += configured.weight;
    }
    if (candidates.empty() || total_weight == 0) return {};

    uint64_t bucket = StableHash(request.key) % total_weight;
    size_t primary = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& candidate = candidates[i];
      if (bucket < candidate.weight) {
        primary = i;
        break;
      }
      bucket -= candidate.weight;
    }

    std::vector<uint32_t> order;
    order.reserve(candidates.size());
    for (size_t offset = 0; offset < candidates.size(); ++offset) {
      order.push_back(candidates[(primary + offset) % candidates.size()].backend_id);
    }
    return order;
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

 private:
  static uint64_t StableHash(std::string_view key) {
    // FNV-1a is deliberately fixed here rather than std::hash, whose output is
    // not a cross-process contract.
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : key) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  std::vector<BackendPlacementWeight> weights_;
};

inline std::unique_ptr<PoolPolicy> MakeWeightedPlacementPolicy(
    std::vector<BackendPlacementWeight> weights) {
  return std::make_unique<WeightedPlacementPolicy>(std::move(weights));
}

}  // namespace mori::umbp
