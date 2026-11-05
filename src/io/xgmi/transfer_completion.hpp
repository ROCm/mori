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
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "mori/io/common.hpp"
#include "src/io/xgmi/hip_resource_pool.hpp"

namespace mori::io {

// A retained entry is removed only after evidence covers every queued command.
// The process-lifetime registry also owns operations whose status was destroyed.
struct TransferCompletion {
  static std::shared_ptr<TransferCompletion> Begin(TransferStatus* status,
                                                   std::shared_ptr<EventPool> events,
                                                   std::shared_ptr<StreamPool> streams,
                                                   std::shared_ptr<std::atomic<size_t>> pending);
  void Seal();
  void FinalizeBlocking();
  void FinalizeNonBlocking();
  QuiescenceState Quiescence(int waitMs = 0);

  // Written only by the submitting thread before Seal; immutable to readers until then.
  hipStream_t stream{nullptr};
  hipEvent_t event{nullptr};
  int eventDevice{-1};
  bool submitted{false};
  bool eventRecorded{false};
  void* deviceMeta{nullptr};
  std::vector<size_t> hostMeta;

 private:
  void StartFailureLocked();
  bool PollLocked();
  bool CompleteLocked(bool blocking);
  void PublishFailureLocked(const char* operation, hipError_t error);

  TransferStatus* status_{nullptr};
  std::shared_ptr<EventPool> eventPool_;
  std::shared_ptr<StreamPool> streamPool_;
  std::shared_ptr<std::atomic<size_t>> pending_;
  std::atomic<QuiescenceState> proof_{QuiescenceState::Pending};
  bool sealed_{false};
  bool failed_{false};
  bool proven_{false};
  bool fenceRecorded_{false};
  hipEvent_t fence_{nullptr};
  std::mutex eventMu_;
  std::condition_variable submittedCv_;
};

struct XgmiSubmission {
  XgmiSubmission(TransferStatus* status, std::shared_ptr<EventPool> events,
                 std::shared_ptr<StreamPool> streams, std::shared_ptr<std::atomic<size_t>> pending)
      : completion(TransferCompletion::Begin(status, std::move(events), std::move(streams),
                                             std::move(pending))) {}
  ~XgmiSubmission() { completion->Seal(); }
  const std::shared_ptr<TransferCompletion> completion;
};

}  // namespace mori::io
