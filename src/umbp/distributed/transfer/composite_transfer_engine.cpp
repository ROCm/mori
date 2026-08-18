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

#include <string>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

// Wraps the per-engine handles a single Submit fanned out into.
class CompositeTransferEngine::FanOutHandle final : public TransferHandle {
 public:
  void Add(std::unique_ptr<TransferHandle> handle) {
    if (handle != nullptr) handles_.push_back(std::move(handle));
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
    for (auto& h : handles_) h->Wait(failures);
  }

 private:
  std::vector<std::unique_ptr<TransferHandle>> handles_;
  std::vector<TransferFailure> failures_;
  bool drained_ = false;
};

void CompositeTransferEngine::AddEngine(std::unique_ptr<TransferEngine> engine) {
  if (engine == nullptr) return;
  engines_.push_back(std::move(engine));
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
      continue;
    }
    by_engine[engine].push_back(item);
  }

  for (auto& [engine, engine_items] : by_engine) {
    adopt(engine->Plan(engine_items), engine);
  }
  return out;
}

std::unique_ptr<TransferHandle> CompositeTransferEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;

  std::unordered_map<TransferEngine*, std::vector<TransferPlan>> by_engine;
  auto handle = std::make_unique<FanOutHandle>();
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
    // Snapshot the tags BEFORE the move so a Submit that posts nothing can
    // still fail exactly the keys it dropped.
    std::vector<size_t> tags;
    for (const auto& plan : engine_plans)
      tags.insert(tags.end(), plan.tags.begin(), plan.tags.end());
    auto sub = engine->Submit(std::move(engine_plans));
    if (sub != nullptr) {
      handle->Add(std::move(sub));
      continue;
    }
    // Submit posted nothing: the caller will never get a completion for these.
    TransferFailure f;
    f.message = std::string(engine->Name()) + " submitted nothing";
    f.tags = std::move(tags);
    handle->AddFailure(std::move(f));
  }

  if (handle->Empty()) return nullptr;
  return handle;
}

}  // namespace mori::umbp
