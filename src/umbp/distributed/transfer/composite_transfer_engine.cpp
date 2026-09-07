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
#include "umbp/distributed/transfer/composite_transfer_engine.h"

#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

// Wraps the per-engine handles a single Submit fanned out into.
//
// Also the transfer layer's measurement point.  Each sub-handle is tagged with
// the engine that produced it and the shape of what it carries, so Wait can
// charge bytes, failures and in-flight time to that engine — the composite is
// the only object that knows the mapping, and Wait is the only moment the
// outcome is known.
class CompositeTransferEngine::FanOutHandle final : public TransferHandle {
 public:
  // One sub-handle plus what it costs.  `bytes` and `plans` are known at submit
  // time; how many plans failed is known only after Wait.
  struct Charge {
    size_t engine_index = 0;
    TransferDirection dir = TransferDirection::kLocal;
    uint64_t bytes = 0;
    uint64_t plans = 0;
  };

  explicit FanOutHandle(CompositeTransferEngine* owner) : owner_(owner) {}

  void Add(std::unique_ptr<TransferHandle> handle, std::vector<Charge> charges) {
    if (handle == nullptr) {
      // Nothing posted: the caller fails these tags, and so do the metrics.
      for (auto& c : charges) {
        owner_->RecordSettled(c.engine_index, c.dir, /*bytes=*/0, /*nanos=*/0, c.plans);
      }
      return;
    }
    handles_.push_back({std::move(handle), std::move(charges)});
  }
  void AddFailure(TransferFailure f) { failures_.push_back(std::move(f)); }
  bool Empty() const { return handles_.empty() && failures_.empty(); }

  void Wait(std::vector<TransferFailure>* failures) override {
    if (drained_) return;
    drained_ = true;
    if (failures != nullptr) {
      for (auto& f : failures_) failures->push_back(std::move(f));
    }
    failures_.clear();
    // Wait every sub-handle, never breaking early: an unwaited handle would
    // leave live statuses behind.
    for (auto& entry : handles_) {
      std::vector<TransferFailure> mine;
      const auto start = std::chrono::steady_clock::now();
      entry.handle->Wait(&mine);
      const uint64_t nanos =
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count());

      // One TransferFailure per failed plan (see TransferHandle::Wait), so the
      // count maps onto the plans this engine was charged.  A handle carries at
      // most one charge per direction, and a submit is usually uniform in
      // direction, so the fill order below is exact in practice; when one
      // engine posted both a push and a pull in the same submit, the failure
      // total is right and its split across the two directions can be off.
      // Wait does not say which plan failed, and adding a plan id to
      // TransferFailure to sharpen a metric would be paying in the interface
      // for something no caller needs.
      uint64_t failed = static_cast<uint64_t>(mine.size());
      for (auto& c : entry.charges) {
        const uint64_t attributed = failed >= c.plans ? c.plans : failed;
        failed -= attributed;
        // Time is charged once per (engine, direction) group, not per plan: it
        // is wall time in flight, and the plans inside a group overlap.
        owner_->RecordSettled(c.engine_index, c.dir, c.bytes, nanos, attributed);
      }
      if (failures != nullptr) {
        for (auto& f : mine) failures->push_back(std::move(f));
      }
    }
  }

 private:
  struct Entry {
    std::unique_ptr<TransferHandle> handle;
    std::vector<Charge> charges;
  };

  CompositeTransferEngine* owner_ = nullptr;
  std::vector<Entry> handles_;
  std::vector<TransferFailure> failures_;
  bool drained_ = false;
};

void CompositeTransferEngine::AddEngine(std::unique_ptr<TransferEngine> engine) {
  if (engine == nullptr) return;
  engines_.push_back(std::move(engine));
  // Grown in lockstep so an engine's index is its counters' index.  This is the
  // whole registration cost of instrumenting a new transport.
  counters_.push_back(std::make_unique<EngineCounters>());
}

void CompositeTransferEngine::RecordSettled(size_t engine_index, TransferDirection dir,
                                            uint64_t bytes, uint64_t nanos, uint64_t failed_plans) {
  if (engine_index >= counters_.size()) return;
  DirectionCounters& c = counters_[engine_index]->by_direction[DirectionIndex(dir)];
  if (bytes > 0) c.bytes.fetch_add(bytes, std::memory_order_relaxed);
  if (nanos > 0) c.nanos.fetch_add(nanos, std::memory_order_relaxed);
  if (failed_plans > 0) c.failed_plans.fetch_add(failed_plans, std::memory_order_relaxed);
}

TransferRef CompositeTransferEngine::RegisterMemory(void* base, size_t size,
                                                    mori::io::MemoryLocationType loc, int device) {
  TransferRef merged;
  for (auto& engine : engines_) {
    TransferRef part = engine->RegisterMemory(base, size, loc, device);
    if (part.HasHostPtr() && !merged.HasHostPtr()) {
      merged.host_ptr = part.host_ptr;
      merged.size = part.size;
      merged.loc = part.loc;
      merged.device = part.device;
    }
    if (part.HasMemoryDesc()) {
      if (!merged.HasMemoryDesc()) {
        merged.mem = part.mem;
      } else {
        // TransferRef currently carries ONE mori::io::MemoryDesc, so a second
        // engine's registration has nowhere to go — and Deregister would never
        // release it.  Unreachable with today's engine set (only MoriIoEngine
        // registers); it becomes real with a second registering transport, at
        // which point TransferRef needs a handle per transport the way
        // mori::io::MemoryDesc already keeps ipcHandle beside fabricHandle.
        MORI_UMBP_WARN(
            "[CompositeTransferEngine] '{}' also registered {} bytes but TransferRef holds only "
            "one descriptor; that registration will leak",
            engine->Name(), size);
        engine->Deregister(part);
      }
    }
  }
  return merged;
}

TransferRef CompositeTransferEngine::RegisterFile(int fd, uint64_t offset, uint64_t size) {
  // Unlike RegisterMemory, a file range fans out to exactly one taker: only a
  // file-capable engine (GdsEngine) claims it, and it owns the fd registration
  // outright.  First match wins; an invalid ref means no file engine is
  // configured and the caller must fall back to a memory path.
  for (auto& engine : engines_) {
    TransferRef part = engine->RegisterFile(fd, offset, size);
    if (part.IsFile()) return part;
  }
  return TransferRef{};
}

void CompositeTransferEngine::Deregister(const TransferRef& ref) {
  for (auto& engine : engines_) engine->Deregister(ref);
}

TransferEngine* CompositeTransferEngine::SelectEngine(const TransferRef& src,
                                                      const TransferRef& dst) const {
  for (const auto& engine : engines_) {
    if (engine->CanHandle(src, dst)) return engine.get();
  }
  return nullptr;
}

bool CompositeTransferEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  return SelectEngine(src, dst) != nullptr;
}

TransferPlanSet CompositeTransferEngine::Plan(const std::vector<TransferItem>& items) const {
  TransferPlanSet out;
  if (items.empty()) return out;

  auto adopt = [&out](TransferPlanSet sub, TransferEngine* engine) {
    for (auto& plan : sub.plans) {
      plan.engine = engine;
      out.plans.push_back(std::move(plan));
    }
    out.rejected_tags.insert(out.rejected_tags.end(), sub.rejected_tags.begin(),
                             sub.rejected_tags.end());
  };

  // Fast path: every item goes to the same engine, so delegate the caller's own
  // vector.  Worth a separate path because a TransferItem holds two
  // TransferRefs and each carries a MemoryDesc with three std::strings —
  // partitioning by copy would allocate per page on the hot path, and in
  // practice a batch is uniform (one peer group is all-remote, a self-target
  // batch is all-local).
  TransferEngine* first = SelectEngine(items.front().src, items.front().dst);
  if (first != nullptr) {
    bool uniform = true;
    for (size_t i = 1; i < items.size() && uniform; ++i) {
      uniform = SelectEngine(items[i].src, items[i].dst) == first;
    }
    if (uniform) {
      adopt(first->Plan(items), first);
      return out;
    }
  }

  // Mixed batch: partition, then plan per engine in ONE call each — grouping is
  // what Plan is for, and planning item-by-item would defeat it.
  std::unordered_map<TransferEngine*, std::vector<TransferItem>> by_engine;
  for (const auto& item : items) {
    TransferEngine* engine = SelectEngine(item.src, item.dst);
    if (engine == nullptr) {
      out.rejected_tags.push_back(item.tag);
      rejected_items_.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    by_engine[engine].push_back(item);
  }

  for (auto& [engine, engine_items] : by_engine) {
    adopt(engine->Plan(engine_items), engine);
  }
  return out;
}

size_t CompositeTransferEngine::IndexOf(const TransferEngine* engine) const {
  for (size_t i = 0; i < engines_.size(); ++i) {
    if (engines_[i].get() == engine) return i;
  }
  return engines_.size();  // not ours; RecordSettled drops it
}

std::unique_ptr<TransferHandle> CompositeTransferEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;

  std::unordered_map<TransferEngine*, std::vector<TransferPlan>> by_engine;
  auto handle = std::make_unique<FanOutHandle>(this);
  for (auto& plan : plans) {
    TransferEngine* engine = plan.engine;
    if (engine == nullptr) {
      // A plan this composite did not produce.  Fail its keys rather than
      // guessing an engine: the endpoints alone cannot tell us which engine
      // built the offsets (a bounce plan's local offsets mean nothing outside
      // the engine that owns that pool).
      TransferFailure f;
      f.tags = plan.tags;
      f.message = "plan has no owning engine";
      handle->AddFailure(std::move(f));
      continue;
    }
    by_engine[engine].push_back(std::move(plan));
  }

  for (auto& [engine, engine_plans] : by_engine) {
    const size_t engine_index = IndexOf(engine);

    // Snapshot the tags BEFORE the move so a Submit that posts nothing can
    // still fail exactly the keys it dropped.  The metric charges are built in
    // the same pass, for the same reason.
    std::vector<size_t> tags;
    FanOutHandle::Charge charges[kDirectionCount];
    for (size_t d = 0; d < kDirectionCount; ++d) {
      charges[d].engine_index = engine_index;
      charges[d].dir = static_cast<TransferDirection>(d);
    }
    for (const auto& plan : engine_plans) {
      tags.insert(tags.end(), plan.tags.begin(), plan.tags.end());
      FanOutHandle::Charge& c = charges[DirectionIndex(plan.dir)];
      ++c.plans;
      for (size_t size : plan.sizes) c.bytes += size;
    }

    std::vector<FanOutHandle::Charge> live;
    for (auto& c : charges) {
      if (c.plans == 0) continue;
      // Posted count goes in now; how many of them failed is decided in Wait.
      if (engine_index < counters_.size()) {
        counters_[engine_index]->by_direction[DirectionIndex(c.dir)].plans.fetch_add(
            c.plans, std::memory_order_relaxed);
      }
      live.push_back(c);
    }

    auto sub = engine->Submit(std::move(engine_plans));
    if (sub != nullptr) {
      handle->Add(std::move(sub), std::move(live));
      continue;
    }
    // Submit posted nothing: the caller will never get a completion for these.
    handle->Add(nullptr, std::move(live));
    TransferFailure f;
    f.message = std::string(engine->Name()) + " submitted nothing";
    f.tags = std::move(tags);
    handle->AddFailure(std::move(f));
  }

  if (handle->Empty()) return nullptr;
  return handle;
}

namespace {

const char* DirectionName(TransferDirection dir) {
  switch (dir) {
    case TransferDirection::kPush:
      return "push";
    case TransferDirection::kPull:
      return "pull";
    case TransferDirection::kLocal:
      return "local";
    default:
      return "unknown";
  }
}

constexpr double kNanosPerSecond = 1e9;

}  // namespace

std::vector<MetricSample> CompositeTransferEngine::SampleMetrics() const {
  std::vector<MetricSample> out;

  for (size_t i = 0; i < engines_.size() && i < counters_.size(); ++i) {
    const char* engine_name = engines_[i]->Name();

    for (size_t d = 0; d < kDirectionCount; ++d) {
      const DirectionCounters& c = counters_[i]->by_direction[d];
      const uint64_t plans = c.plans.load(std::memory_order_relaxed);
      if (plans == 0) continue;
      const char* dir = DirectionName(static_cast<TransferDirection>(d));
      const uint64_t failed = c.failed_plans.load(std::memory_order_relaxed);

      out.push_back(MetricSample{MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL,
                                 MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL_HELP,
                                 {{"engine", engine_name}, {"direction", dir}, {"status", "ok"}},
                                 // Plans posted but not yet settled count as ok until Wait says
                                 // otherwise; the correction lands on the next tick.
                                 plans >= failed ? plans - failed : 0});
      if (failed > 0) {
        out.push_back(
            MetricSample{MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL,
                         MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL_HELP,
                         {{"engine", engine_name}, {"direction", dir}, {"status", "failed"}},
                         failed});
      }

      const uint64_t bytes = c.bytes.load(std::memory_order_relaxed);
      if (bytes > 0) {
        out.push_back(MetricSample{MORI_UMBP_METRIC_TRANSFER_BYTES_TOTAL,
                                   MORI_UMBP_METRIC_TRANSFER_BYTES_TOTAL_HELP,
                                   {{"engine", engine_name}, {"direction", dir}},
                                   bytes});
      }
      const uint64_t nanos = c.nanos.load(std::memory_order_relaxed);
      if (nanos > 0) {
        out.push_back(MetricSample{MORI_UMBP_METRIC_TRANSFER_SECONDS_TOTAL,
                                   MORI_UMBP_METRIC_TRANSFER_SECONDS_TOTAL_HELP,
                                   {{"engine", engine_name}, {"direction", dir}},
                                   nanos,
                                   MetricKind::kCounter,
                                   1.0 / kNanosPerSecond});
      }
    }

    // Whatever the engine publishes about its own internals, stamped with the
    // engine that produced it so it shares the dashboard's engine= dimension.
    for (MetricSample& s : engines_[i]->SampleMetrics()) {
      s.labels.insert(s.labels.begin(), {"engine", engine_name});
      out.push_back(std::move(s));
    }
  }

  // Items no engine would take.  Not an engine failure — a routing one — so it
  // gets its own engine value rather than being blamed on whoever ran last.
  const uint64_t rejected = rejected_items_.load(std::memory_order_relaxed);
  if (rejected > 0) {
    out.push_back(MetricSample{MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL,
                               MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL_HELP,
                               {{"engine", "none"}, {"direction", "none"}, {"status", "rejected"}},
                               rejected});
  }
  return out;
}

}  // namespace mori::umbp
