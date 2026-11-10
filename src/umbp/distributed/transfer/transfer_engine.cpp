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
#include "umbp/distributed/transfer/transfer_engine.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <sstream>

namespace mori::umbp {

namespace {

// Adds its lifetime to a sink; a null sink makes it inert, so an untimed
// Transfer does not even read the clock.
class StepTimer {
 public:
  explicit StepTimer(double* sink)
      : sink_(sink),
        start_(sink ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
  ~StepTimer() { Stop(); }
  StepTimer(const StepTimer&) = delete;
  StepTimer& operator=(const StepTimer&) = delete;
  void Stop() {
    if (sink_ == nullptr) return;
    *sink_ += std::chrono::duration_cast<std::chrono::duration<double>>(
                  std::chrono::steady_clock::now() - start_)
                  .count();
    sink_ = nullptr;
  }

 private:
  double* sink_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace

TransferPlanSet TransferEngine::Plan(const std::vector<TransferItem>& items) const {
  auto builder = CreatePlanBuilder(items.size());
  for (const auto& item : items) builder->Add(item);
  TransferPlanSet out = builder->Finish();
  for (size_t tag : out.rejected_tags) {
    auto found = std::find_if(items.begin(), items.end(),
                              [&](const TransferItem& item) { return item.tag == tag; });
    if (found == items.end()) continue;
    const auto describe = [&](const TransferRef& ref, size_t offset) {
      std::ostringstream text;
      text << "kind=" << static_cast<int>(ref.kind) << " valid=" << ref.Valid()
           << " host=" << ref.HasHostPtr() << " mem=" << ref.HasMemoryDesc()
           << " endpoint=" << ref.HasEngineEndpoint() << " executor='" << ref.executor_id
           << "' region='" << ref.region_id << "' address=" << ref.executor_address
           << " size=" << ref.size << " offset=" << offset << " bytes=" << found->size
           << " loc=" << static_cast<int>(ref.loc) << " device=" << ref.device;
      return text.str();
    };
    std::string reason;
    if (found->src_offset > found->src.size) {
      reason = "source offset exceeds size";
    } else if (found->size > found->src.size - found->src_offset) {
      reason = "source offset+bytes exceeds size";
    } else if (found->dst_offset > found->dst.size) {
      reason = "target offset exceeds size";
    } else if (found->size > found->dst.size - found->dst_offset) {
      reason = "target offset+bytes exceeds size";
    } else if (!CanHandle(found->src, found->dst)) {
      reason = "CanHandle=false";
    } else {
      reason = "selected builder rejected item";
    }
    out.rejected_details.emplace_back(tag, std::string("engine=") + Name() + " reason=" + reason +
                                               " src=[" + describe(found->src, found->src_offset) +
                                               "] dst=[" + describe(found->dst, found->dst_offset) +
                                               "]");
  }
  return out;
}

bool TransferEngine::Transfer(const std::vector<TransferItem>& items,
                              std::vector<size_t>* failed_tags, const StepTiming* timing,
                              bool* quiesced, bool* submitted,
                              std::vector<TransferFailure>* failure_details) {
  if (quiesced) *quiesced = true;
  if (submitted) *submitted = false;
  if (items.empty()) return true;

  TransferPlanSet planned;
  {
    StepTimer plan_timer(timing ? timing->plan : nullptr);
    planned = Plan(items);
  }
  return Transfer(std::move(planned), failed_tags, timing, quiesced, submitted, failure_details);
}

bool TransferEngine::Transfer(TransferPlanSet planned, std::vector<size_t>* failed_tags,
                              const StepTiming* timing, bool* quiesced, bool* submitted,
                              std::vector<TransferFailure>* failure_details) {
  if (quiesced) *quiesced = true;
  if (submitted) *submitted = false;
  bool ok = planned.rejected_tags.empty();
  if (failed_tags != nullptr) {
    failed_tags->insert(failed_tags->end(), planned.rejected_tags.begin(),
                        planned.rejected_tags.end());
  }
  if (failure_details != nullptr && !planned.rejected_tags.empty()) {
    const std::string detail = planned.rejected_details.empty()
                                   ? std::string{}
                                   : ": " + planned.rejected_details.front().second;
    failure_details->push_back({planned.rejected_tags,
                                1,
                                "transfer planning rejected " +
                                    std::to_string(planned.rejected_tags.size()) + " item(s)" +
                                    detail,
                                {},
                                /*quiesced=*/true,
                                /*submitted=*/false});
  }
  if (planned.plans.empty()) return ok;

  // Snapshot every tag that made it into a plan BEFORE the move, so a Submit
  // that posts nothing can still fail exactly those keys.
  std::vector<size_t> planned_tags;
  for (const auto& plan : planned.plans) {
    planned_tags.insert(planned_tags.end(), plan.tags.begin(), plan.tags.end());
  }

  std::unique_ptr<TransferHandle> handle;
  {
    StepTimer submit_timer(timing ? timing->submit : nullptr);
    handle = Submit(std::move(planned.plans));
  }
  if (handle == nullptr) {
    if (failed_tags != nullptr) {
      failed_tags->insert(failed_tags->end(), planned_tags.begin(), planned_tags.end());
    }
    if (failure_details != nullptr) {
      failure_details->push_back({std::move(planned_tags),
                                  1,
                                  "transfer engine submitted no work",
                                  {},
                                  /*quiesced=*/true,
                                  /*submitted=*/false});
    }
    return false;
  }
  if (submitted) *submitted = true;

  std::vector<TransferFailure> failures;
  {
    StepTimer wait_timer(timing ? timing->wait : nullptr);
    handle->Wait(&failures);
  }
  if (failures.empty()) return ok;
  if (quiesced != nullptr) {
    *quiesced = std::all_of(failures.begin(), failures.end(),
                            [](const auto& failure) { return failure.quiesced; });
  }
  if (submitted != nullptr) {
    *submitted = std::any_of(failures.begin(), failures.end(),
                             [](const auto& failure) { return failure.submitted; });
  }
  if (failed_tags != nullptr) {
    for (const auto& f : failures) {
      failed_tags->insert(failed_tags->end(), f.tags.begin(), f.tags.end());
    }
    std::sort(failed_tags->begin(), failed_tags->end());
    failed_tags->erase(std::unique(failed_tags->begin(), failed_tags->end()), failed_tags->end());
  }
  if (failure_details != nullptr) {
    failure_details->insert(failure_details->end(), std::make_move_iterator(failures.begin()),
                            std::make_move_iterator(failures.end()));
  }
  return false;
}

}  // namespace mori::umbp
