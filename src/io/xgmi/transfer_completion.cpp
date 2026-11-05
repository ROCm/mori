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
#include "src/io/xgmi/transfer_completion.hpp"

#include <unordered_map>

#include "mori/io/logging.hpp"

namespace mori::io {
namespace {
struct RetainedTransfers {
  std::mutex mu;
  std::unordered_map<TransferCompletion*, std::shared_ptr<TransferCompletion>> operations;
};

RetainedTransfers& Retained() {
  // Do not run GPU cleanup at static destruction without a completion proof.
  static auto* retained = new RetainedTransfers;
  return *retained;
}
}  // namespace

std::shared_ptr<TransferCompletion> TransferCompletion::Begin(
    TransferStatus* status, std::shared_ptr<EventPool> events, std::shared_ptr<StreamPool> streams,
    std::shared_ptr<std::atomic<size_t>> pending) {
  status->CheckReusable();
  auto completion = std::make_shared<TransferCompletion>();
  completion->status_ = status;
  completion->eventPool_ = std::move(events);
  completion->streamPool_ = std::move(streams);
  completion->pending_ = std::move(pending);
  // Allocate and retain before any command can reach a GPU.
  {
    std::lock_guard<std::mutex> lock(Retained().mu);
    Retained().operations.emplace(completion.get(), completion);
    completion->pending_->fetch_add(1, std::memory_order_relaxed);
  }
  try {
    status->SetQuiescenceCallback(
        [completion](int slice) { return completion->Quiescence(slice); });
    status->SetWaitCallback([completion] { completion->FinalizeBlocking(); });
    status->SetProgressCallback([completion] { completion->FinalizeNonBlocking(); });
    status->SetCode(StatusCode::IN_PROGRESS);
  } catch (...) {
    std::lock_guard<std::mutex> lock(Retained().mu);
    completion->pending_->fetch_sub(1, std::memory_order_relaxed);
    Retained().operations.erase(completion.get());
    completion->proof_.store(QuiescenceState::Quiesced, std::memory_order_release);
    throw;
  }
  return completion;
}

void TransferCompletion::PublishFailureLocked(const char* operation, hipError_t error) {
  status_->Update(StatusCode::ERR_GPU_OP,
                  std::string("XGMI: ") + operation + ": " + hipGetErrorString(error));
  StartFailureLocked();
}

void TransferCompletion::StartFailureLocked() {
  if (failed_) return;
  failed_ = true;
  if (!submitted) return;
  // The failure result is already published. Never re-record the original event.
  MORI_IO_WARN("XGMI: queued transfer failed; retaining resources until quiescence is proven");
  int previousDevice = -1;
  (void)hipGetDevice(&previousDevice);
  if (hipSetDevice(eventDevice) == hipSuccess) {
    fence_ = eventPool_->TryGetEvent(eventDevice);
    if (fence_) fenceRecorded_ = hipEventRecord(fence_, stream) == hipSuccess;
  }
  if (previousDevice >= 0) (void)hipSetDevice(previousDevice);
}

void TransferCompletion::Seal() {
  {
    std::lock_guard<std::mutex> lock(eventMu_);
    if (sealed_) return;
    if (status_->Failed())
      StartFailureLocked();
    else if (!eventRecorded) {
      if (!submitted && status_->Succeeded()) {
        proven_ = true;
      } else {
        status_->Update(StatusCode::ERR_GPU_OP, "XGMI: submission interrupted");
        StartFailureLocked();
      }
    }
    sealed_ = true;
    if (!submitted) {
      proven_ = true;
      CompleteLocked(true);
    }
  }
  submittedCv_.notify_all();
}

bool TransferCompletion::PollLocked() {
  if (proven_) return true;
  hipError_t error;
  if (failed_) {
    error = fenceRecorded_ ? hipEventQuery(fence_) : hipStreamQuery(stream);
  } else {
    error = hipEventQuery(event);
  }
  if (error == hipErrorNotReady) {
    (void)hipGetLastError();
    return false;
  }
  if (error != hipSuccess) {
    if (!failed_) {
      PublishFailureLocked("hipEventQuery failed", error);
      return false;
    }
    MORI_IO_ERROR("XGMI: quiescence unprovable ({}); retaining operation until process exit",
                  hipGetErrorString(error));
    proof_.store(QuiescenceState::Unprovable, std::memory_order_release);
    return false;
  }
  proven_ = true;
  return true;
}

bool TransferCompletion::CompleteLocked(bool blocking) {
  std::unique_lock<std::mutex> retained(Retained().mu, std::defer_lock);
  if (blocking)
    retained.lock();
  else if (!retained.try_lock())
    return false;
  // Return both events together; a nonblocking query never waits on a pool mutex.
  if (blocking) {
    if (event) eventPool_->PutEvent(event, eventDevice);
    if (fence_) eventPool_->PutEvent(fence_, eventDevice);
  } else if (!eventPool_->TryPutEvents(event, fence_, eventDevice)) {
    return false;
  }
  event = fence_ = nullptr;
  if (deviceMeta) {
    (void)hipFree(deviceMeta);
    deviceMeta = nullptr;
  }
  std::vector<size_t>().swap(hostMeta);
  if (!failed_) status_->SetCode(StatusCode::SUCCESS);
  pending_->fetch_sub(1, std::memory_order_release);
  Retained().operations.erase(this);
  proof_.store(QuiescenceState::Quiesced, std::memory_order_release);
  return true;
}

void TransferCompletion::FinalizeBlocking() {
  if (proof_.load(std::memory_order_acquire) != QuiescenceState::Pending) return;
  std::unique_lock<std::mutex> lock(eventMu_);
  submittedCv_.wait(lock, [&] { return sealed_; });
  if (proof_.load(std::memory_order_acquire) != QuiescenceState::Pending || failed_) return;
  auto error = proven_ ? hipSuccess : hipEventSynchronize(event);
  if (error != hipSuccess) {
    PublishFailureLocked("hipEventSynchronize failed", error);
    return;
  }
  proven_ = true;
  CompleteLocked(true);
}

void TransferCompletion::FinalizeNonBlocking() {
  if (proof_.load(std::memory_order_acquire) != QuiescenceState::Pending) return;
  std::unique_lock<std::mutex> lock(eventMu_, std::try_to_lock);
  if (!lock || !sealed_ || proof_.load(std::memory_order_acquire) != QuiescenceState::Pending)
    return;
  if (PollLocked()) CompleteLocked(false);
}

QuiescenceState TransferCompletion::Quiescence(int waitMs) {
  FinalizeNonBlocking();
  auto proof = proof_.load(std::memory_order_acquire);
  if (proof == QuiescenceState::Pending && waitMs > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(waitMs));
    FinalizeNonBlocking();
    proof = proof_.load(std::memory_order_acquire);
  }
  return proof;
}
}  // namespace mori::io
