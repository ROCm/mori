// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/transfer/engine_transfer_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iterator>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/engine/engine_dispatch.h"
#include "umbp/engine/engine_executor.h"

namespace mori::umbp {
namespace {

std::string RefDescription(const TransferRef& ref, size_t offset, size_t bytes) {
  std::ostringstream out;
  out << "kind=" << static_cast<int>(ref.kind) << " valid=" << ref.Valid()
      << " host=" << ref.HasHostPtr() << " mem=" << ref.HasMemoryDesc()
      << " engine_endpoint=" << ref.HasEngineEndpoint() << " executor='" << ref.executor_id
      << "' region='" << ref.region_id << "' address=" << ref.executor_address
      << " size=" << ref.size << " offset=" << offset << " bytes=" << bytes
      << " loc=" << static_cast<int>(ref.loc) << " device=" << ref.device;
  return out.str();
}

void WarnPlanRejection(std::string_view reason, const TransferItem& item, std::string_view detail) {
  struct State {
    uint64_t count = 0;
    uint64_t since_last = 0;
    std::chrono::steady_clock::time_point last{};
  };
  static std::mutex mutex;
  static std::unordered_map<std::string, State> states;
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex);
  auto& state = states[std::string(reason)];
  ++state.count;
  ++state.since_last;
  if (state.last != std::chrono::steady_clock::time_point{} &&
      now - state.last < std::chrono::seconds(10)) {
    return;
  }
  MORI_UMBP_WARN(
      "[EnginePlanBuilder] reject reason={} count={} since_last={} detail='{}' src=[{}] dst=[{}]",
      reason, state.count, state.since_last, detail,
      RefDescription(item.src, item.src_offset, item.size),
      RefDescription(item.dst, item.dst_offset, item.size));
  state.since_last = 0;
  state.last = now;
}

std::pair<std::string, std::string> RejectionReason(const EngineTransferEngine* owner,
                                                    const TransferItem& item) {
  if (owner == nullptr) return {"NO_OWNER", "plan builder has no engine"};
  if (!item.src.Valid()) return {"SRC_INVALID", "source TransferRef::Valid is false"};
  if (!item.dst.Valid()) return {"DST_INVALID", "target TransferRef::Valid is false"};
  if (!item.src.HasEngineEndpoint() && !item.dst.HasEngineEndpoint()) {
    return {"NO_ENGINE_ENDPOINT", "neither endpoint has executor_id/region_id/address"};
  }
  std::string executor;
  for (const auto* ref : {&item.src, &item.dst}) {
    if (!ref->HasEngineEndpoint()) continue;
    if (executor.empty()) executor = ref->executor_id;
    if (executor != ref->executor_id) {
      return {"EXECUTOR_MISMATCH", "source and target name different executors"};
    }
  }
  if (!owner->CanHandle(item.src, item.dst)) {
    return {"CAN_HANDLE_FALSE", "engine rejected endpoint ownership"};
  }
  if (item.src_offset > item.src.size) {
    return {"SRC_OFFSET", "source offset exceeds source size"};
  }
  if (item.size > item.src.size - item.src_offset) {
    return {"SRC_RANGE", "source offset+bytes exceeds source size"};
  }
  if (item.dst_offset > item.dst.size) {
    return {"DST_OFFSET", "target offset exceeds target size"};
  }
  if (item.size > item.dst.size - item.dst_offset) {
    return {"DST_RANGE", "target offset+bytes exceeds target size"};
  }
  return {};
}

std::chrono::milliseconds DispatchTimeout() {
  constexpr uint64_t kDefaultMs = 30000;
  const char* raw = std::getenv("UMBP_ENGINE_DISPATCH_TIMEOUT_MS");
  if (raw == nullptr || raw[0] == '\0') return std::chrono::milliseconds(kDefaultMs);
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(raw, &end, 10);
  return end == raw || parsed == 0 ? std::chrono::milliseconds(kDefaultMs)
                                   : std::chrono::milliseconds(parsed);
}

bool SameEndpoint(const TransferRef& left, const TransferRef& right) {
  return left.kind == right.kind && left.host_ptr == right.host_ptr && left.size == right.size &&
         left.loc == right.loc && left.device == right.device && left.mem == right.mem &&
         left.executor_id == right.executor_id && left.region_id == right.region_id &&
         left.executor_address == right.executor_address && left.file_fd == right.file_fd &&
         left.file_offset == right.file_offset && left.gds_handle == right.gds_handle;
}

class EnginePlanBuilder final : public TransferPlanBuilder {
 public:
  explicit EnginePlanBuilder(const EngineTransferEngine* owner, size_t expected) : owner_(owner) {
    plans_.reserve(expected);
  }

  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    const auto [reason, detail] = RejectionReason(owner_, item);
    if (!reason.empty()) {
      WarnPlanRejection(reason, item, detail);
      rejected_.push_back(item.tag);
      return;
    }
    TransferRef source = item.src;
    TransferRef target = item.dst;
    // A node-owned shared region has no process-local executor identity. The
    // engine endpoint on the other side is the rank that must hydrate that
    // region's memfd mapping before executing the plan. Put that ownership on
    // the wire explicitly instead of relying on an empty-id fallback in the
    // rank process (which same-process tests accidentally made sufficient).
    if (source.HasEngineEndpoint() && !target.region_id.empty() && !target.HasEngineEndpoint()) {
      target.executor_id = source.executor_id;
    } else if (target.HasEngineEndpoint() && !source.region_id.empty() &&
               !source.HasEngineEndpoint()) {
      source.executor_id = target.executor_id;
    }
    auto existing = std::find_if(plans_.begin(), plans_.end(), [&](const TransferPlan& plan) {
      return SameEndpoint(plan.src, source) && SameEndpoint(plan.dst, target);
    });
    if (existing == plans_.end()) {
      TransferPlan plan;
      plan.src = std::move(source);
      plan.dst = std::move(target);
      plan.dir = TransferDirection::kLocal;
      plans_.push_back(std::move(plan));
      existing = std::prev(plans_.end());
    }
    existing->src_offsets.push_back(item.src_offset);
    existing->dst_offsets.push_back(item.dst_offset);
    existing->sizes.push_back(item.size);
    if (std::find(existing->tags.begin(), existing->tags.end(), item.tag) == existing->tags.end()) {
      existing->tags.push_back(item.tag);
    }
  }

  TransferPlanSet Finish() override {
    return TransferPlanSet{std::move(plans_), std::move(rejected_)};
  }

 private:
  const EngineTransferEngine* owner_ = nullptr;
  std::vector<TransferPlan> plans_;
  std::vector<size_t> rejected_;
};

class EngineHandle final : public TransferHandle {
 public:
  EngineHandle(engine::EngineExecutor* executor, std::vector<TransferPlan> plans)
      : result_(std::async(std::launch::async, [executor, plans = std::move(plans)]() mutable {
          std::vector<TransferFailure> failures;
          if (executor == nullptr) {
            failures.push_back({{}, 1, "engine executor is unavailable", {}});
          } else {
            executor->ExecutePlans(plans, &failures);
          }
          return failures;
        })) {}

  ~EngineHandle() override { Wait(nullptr); }

  void Wait(std::vector<TransferFailure>* failures) override {
    if (drained_) return;
    drained_ = true;
    auto settled = result_.get();
    if (failures != nullptr) {
      for (auto& failure : settled) failures->push_back(std::move(failure));
    }
  }

 private:
  std::future<std::vector<TransferFailure>> result_;
  bool drained_ = false;
};

class RemoteEngineHandle final : public TransferHandle {
 public:
  explicit RemoteEngineHandle(std::shared_ptr<engine::DispatchTicket> ticket)
      : ticket_(std::move(ticket)) {}
  ~RemoteEngineHandle() override { Wait(nullptr); }
  void Wait(std::vector<TransferFailure>* failures) override {
    if (drained_) return;
    drained_ = true;
    (void)ticket_->WaitFor(DispatchTimeout(), failures);
  }

 private:
  std::shared_ptr<engine::DispatchTicket> ticket_;
  bool drained_ = false;
};

}  // namespace

TransferRef EngineTransferEngine::RegisterMemory(void* base, size_t size,
                                                 mori::io::MemoryLocationType loc, int device) {
  return TransferRef::HostBytes(base, size, loc, device);
}

bool EngineTransferEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  if (!src.Valid() || !dst.Valid() || (!src.HasEngineEndpoint() && !dst.HasEngineEndpoint())) {
    return false;
  }
  std::string executor_id;
  for (const auto* ref : {&src, &dst}) {
    if (!ref->HasEngineEndpoint()) continue;
    if (executor_id.empty()) executor_id = ref->executor_id;
    if (executor_id != ref->executor_id) return false;
  }
  return executor_ == nullptr || executor_->id() == executor_id;
}

std::unique_ptr<TransferPlanBuilder> EngineTransferEngine::CreatePlanBuilder(
    size_t expected_items) const {
  return std::make_unique<EnginePlanBuilder>(this, expected_items);
}

std::unique_ptr<TransferHandle> EngineTransferEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;
  if (executor_ != nullptr) return std::make_unique<EngineHandle>(executor_, std::move(plans));
  std::string executor_id;
  for (const auto& plan : plans) {
    for (const auto* ref : {&plan.src, &plan.dst}) {
      if (!ref->HasEngineEndpoint()) continue;
      if (executor_id.empty()) executor_id = ref->executor_id;
      if (executor_id != ref->executor_id) return nullptr;
    }
  }
  if (executor_id.empty()) return nullptr;
  return std::make_unique<RemoteEngineHandle>(
      engine::EngineDispatchRegistry::Global().Submit(executor_id, std::move(plans)));
}

}  // namespace mori::umbp
