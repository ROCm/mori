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

bool TransferEngine::Transfer(const std::vector<TransferItem>& items,
                              std::vector<size_t>* failed_tags, const StepTiming* timing) {
  if (items.empty()) return true;

  TransferPlanSet planned;
  {
    StepTimer plan_timer(timing ? timing->plan : nullptr);
    planned = Plan(items);
  }
  bool ok = planned.rejected_tags.empty();
  if (failed_tags != nullptr) {
    failed_tags->insert(failed_tags->end(), planned.rejected_tags.begin(),
                        planned.rejected_tags.end());
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
    return false;
  }

  std::vector<TransferFailure> failures;
  {
    StepTimer wait_timer(timing ? timing->wait : nullptr);
    handle->Wait(&failures);
  }
  if (failures.empty()) return ok;
  if (failed_tags != nullptr) {
    for (const auto& f : failures) {
      failed_tags->insert(failed_tags->end(), f.tags.begin(), f.tags.end());
    }
    std::sort(failed_tags->begin(), failed_tags->end());
    failed_tags->erase(std::unique(failed_tags->begin(), failed_tags->end()), failed_tags->end());
  }
  return false;
}

}  // namespace mori::umbp
