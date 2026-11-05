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
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <thread>

#include "mori/io/logging.hpp"
#include "src/io/rdma/common.hpp"
#include "src/io/rdma/verbs_ops.hpp"
#include "src/io/roctx_mori.hpp"

namespace mori {
namespace io {
namespace {
int* FutexAddress(std::atomic<uint32_t>& word) {
  static_assert(sizeof(word) == sizeof(int));
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  return reinterpret_cast<int*>(&word);
}

void WaitEpoch(std::atomic<uint32_t>& word, uint32_t expected,
               std::chrono::steady_clock::time_point deadline) {
  auto left = deadline - std::chrono::steady_clock::now();
  if (left <= std::chrono::steady_clock::duration::zero()) return;
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(left).count();
  timespec timeout{ns / 1000000000, ns % 1000000000};
  if (syscall(SYS_futex, FutexAddress(word), FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0) <
          0 &&
      errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT) {
    std::this_thread::sleep_for(std::chrono::microseconds(2));
  }
}
}  // namespace

SubmissionLedger::SubmissionLedger(int capacity) : capacity_(capacity) {
  if (capacity < 2) throw std::invalid_argument("SQ capacity must be at least 2");
  size_t size = 1;
  while (size < static_cast<size_t>(capacity)) size <<= 1;
  mask_ = size - 1;
  ring_ = std::make_unique<Entry[]>(size);
}

bool SubmissionLedger::Closed() const {
  return (state_.load(std::memory_order_acquire) & (kClosed | kFrozen)) != 0;
}
bool SubmissionLedger::Frozen() const {
  return (state_.load(std::memory_order_acquire) & kFrozen) != 0;
}
uint64_t SubmissionLedger::Done() const {
  return state_.load(std::memory_order_acquire) & kSqSequenceMask;
}

void SubmissionLedger::NotifyAdmission() {
  if (!admission_.waiters.load(kSqAdmissionOrder)) return;
  admission_.epoch.fetch_add(1, kSqAdmissionOrder);
  syscall(SYS_futex, FutexAddress(admission_.epoch), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr,
          0);
}

void SubmissionLedger::Close(bool freeze) {
  state_.fetch_or(kClosed | (freeze ? kFrozen : 0), std::memory_order_acq_rel);
  NotifyAdmission();
}

void SubmissionLedger::FreezeForShutdown() {
  {
    std::lock_guard<std::mutex> lock(postMu_);
    state_.fetch_or(kClosed | kFrozen, std::memory_order_acq_rel);
  }
  NotifyAdmission();
}

bool SubmissionLedger::TryReserve(int count, int limit) {
  int used = used_.load(kSqAdmissionOrder);
  while (used <= limit - count) {
    if (used_.compare_exchange_weak(used, used + count, kSqAdmissionOrder)) return true;
  }
  return false;
}

bool SubmissionLedger::Reserve(int count, int timeoutUs, std::string* error,
                               ReserveFailure* failure) {
  auto fail = [&](ReserveFailure why, const char* message) {
    if (failure) *failure = why;
    if (error)
      *error = std::string(message) + ": depth=" + std::to_string(Used()) +
               " requested=" + std::to_string(count) + " max=" + std::to_string(OrdinaryLimit());
    return false;
  };
  if (failure) *failure = ReserveFailure::None;
  if (Closed()) return fail(ReserveFailure::Closed, "SQ ledger is closed");
  if (count < 0 || count > OrdinaryLimit())
    return fail(ReserveFailure::ExceedsCapacity, "SQ request exceeds ordinary capacity");
  if (!count || TryReserve(count, OrdinaryLimit())) return true;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeoutUs);
  struct Waiter {
    SqAdmissionEvent& event;
    explicit Waiter(SqAdmissionEvent& e) : event(e) {
      event.waiters.fetch_add(1, kSqAdmissionOrder);
    }
    ~Waiter() { event.waiters.fetch_sub(1, kSqAdmissionOrder); }
  } waiter(admission_);
  while (true) {
    auto epoch = admission_.epoch.load(kSqAdmissionOrder);
    if (Closed()) return fail(ReserveFailure::Closed, "SQ ledger is closed");
    if (TryReserve(count, OrdinaryLimit())) return true;
    if (std::chrono::steady_clock::now() >= deadline)
      return fail(ReserveFailure::Timeout, "SQ full timeout");
    WaitEpoch(admission_.epoch, epoch, deadline);
  }
}

void SubmissionLedger::CancelReservation(int count) {
  if (!count) return;
  int previous = used_.fetch_sub(count, kSqAdmissionOrder);
  assert(count > 0 && previous >= count);
  NotifyAdmission();
}

SubmissionLedger::PostResult SubmissionLedger::PostToSq(ibv_qp* qp, ibv_send_wr* first, int count,
                                                        Kind kind,
                                                        const std::shared_ptr<CqCallbackMeta>& meta,
                                                        int batchSize, bool finalBatch,
                                                        TransferUniqueId transferId, size_t bytes,
                                                        VerbsOps& verbs) {
  PostResult result;
  {
    std::lock_guard<std::mutex> lock(postMu_);
    if (Closed()) {
      result.error = ECANCELED;
    } else {
      result = PostToSqLocked(qp, first, count, kind, meta, batchSize, finalBatch, transferId,
                              bytes, verbs);
    }
  }
  CancelReservation(count - result.posted);
  return result;
}

SubmissionLedger::PostResult SubmissionLedger::PostToSqLocked(
    ibv_qp* qp, ibv_send_wr* first, int count, Kind kind,
    const std::shared_ptr<CqCallbackMeta>& meta, int batchSize, bool finalBatch,
    TransferUniqueId transferId, size_t bytes, VerbsOps& verbs) {
  PostResult result;
  if (count <= 0 || static_cast<uint64_t>(count) > kSqSequenceMask - postedSeq_) {
    result.error = EOVERFLOW;
    return result;
  }
  bool signal = kind != Kind::Data || finalBatch ||
                postedSeq_ + count - lastSignaledSeq_ >= static_cast<uint64_t>(capacity_ / 2);
  auto* wr = first;
  for (int i = 0; i < count; ++i, wr = wr->next) {
    auto seq = postedSeq_ + i + 1;
    auto& entry = ring_[seq & mask_];
    entry.kind = kind == Kind::Data && signal && i == count - 1 ? Kind::DataTail : kind;
    entry.batchSize = entry.kind == Kind::DataTail ? batchSize : 0;
    entry.transferId = transferId;
    assert(!std::atomic_load(&entry.meta));
    std::atomic_exchange(&entry.meta, entry.kind == Kind::DataTail ? meta : nullptr);
    entry.seq.store(seq, std::memory_order_release);
    wr->wr_id = kSqTag | seq;
    wr->send_flags &= ~IBV_SEND_SIGNALED;
    if (signal && i == count - 1) wr->send_flags |= IBV_SEND_SIGNALED;
  }
  uint64_t tail = postedSeq_ + count;
  bool trace = signal && kind == Kind::Data && mori::io::MoriRoctxTransferEnabled();
  if (trace)
    mori::io::MoriRoctxTransferStart(this, tail, transferId, first->opcode == IBV_WR_RDMA_READ,
                                     bytes);
  ibv_send_wr* bad = nullptr;
  result.error = verbs.PostSend(qp, first, &bad);
  result.posted = count;
  if (result.error) {
    result.posted = 0;
    for (auto* cur = first; bad && cur && cur != bad && result.posted < count; cur = cur->next)
      ++result.posted;
  }
  postedSeq_ += result.posted;
  if (kind == Kind::Data && result.posted) result.dataSeq = postedSeq_;
  if (signal && result.posted == count) {
    lastSignaledSeq_ = postedSeq_;
    result.signaledSeq = postedSeq_;
  }
  for (int i = result.posted; i < count; ++i) {
    auto& entry = ring_[(postedSeq_ + i - result.posted + 1) & mask_];
    std::atomic_exchange(&entry.meta, std::shared_ptr<CqCallbackMeta>{});
    entry.seq.store(0, std::memory_order_release);
  }
  if (trace && !result.signaledSeq) mori::io::MoriRoctxTransferStop(this, tail);
  return result;
}

SubmissionLedger::Kind SubmissionLedger::KindForCompletion(uint64_t id) const {
  auto seq = SqSequence(id);
  const auto& entry = ring_[seq & mask_];
  return seq > Done() && entry.seq.load(std::memory_order_acquire) == seq ? entry.kind : Kind::Data;
}

SubmissionLedger::Snapshot SubmissionLedger::TakeEntry(uint64_t seq) {
  auto& entry = ring_[seq & mask_];
  if (entry.seq.load(std::memory_order_acquire) != seq) {
    Close(true);
    return {};
  }
  return {true,
          entry.kind,
          std::atomic_exchange(&entry.meta, std::shared_ptr<CqCallbackMeta>{}),
          entry.batchSize,
          entry.transferId,
          seq};
}

void SubmissionLedger::CompleteEntry(const Snapshot& entry, ibv_wc_status status,
                                     const std::string& message, bool lost) {
  if (entry.kind == Kind::DataTail) {
    mori::io::MoriRoctxTransferStop(this, entry.seq);
    auto meta = entry.meta;
    if (!meta) return;  // A completion consumer already owns the result.
    uint32_t before = meta->finishedBatchSize.fetch_add(entry.batchSize);
    if (status != IBV_WC_SUCCESS) {
      if (status == IBV_WC_WR_FLUSH_ERR && !lost)
        meta->diagnostics.MarkFlushCascade();
      else
        meta->diagnostics.MarkRootCause();
      if (meta->diagnostics.Label() &&
          meta->diagnostics.CurrentFailureKind() == internal::IoFailureKind::FlushCascade &&
          meta->diagnostics.TryMarkLogged(internal::IoFailureKind::FlushCascade)) {
        MORI_IO_DEBUG("{} error {} message {}", meta->diagnostics.Label(),
                      static_cast<uint32_t>(StatusCode::ERR_RDMA_OP), message);
      }
      if (auto* s = meta->status.exchange(nullptr, std::memory_order_acq_rel))
        meta->ReportData(s, StatusCode::ERR_RDMA_OP, message);
    } else if (before + entry.batchSize == static_cast<uint32_t>(meta->totalBatchSize)) {
      if (auto* s = meta->status.exchange(nullptr, std::memory_order_acq_rel))
        meta->ReportData(s, StatusCode::SUCCESS, message);
    }
  } else if (entry.kind == Kind::Notify && (lost || status != IBV_WC_SUCCESS)) {
    MORI_IO_WARN("notification SEND completion failed: transfer_id={} {}", entry.transferId,
                 message);
  } else if (entry.kind == Kind::Marker) {
    std::lock_guard<std::mutex> lock(postMu_);
    markerOutstanding_ = false;
  }
}

bool SubmissionLedger::OnSqCompletion(uint64_t id, ibv_wc_status status,
                                      const std::string& message) {
  uint64_t seq = SqSequence(id);
  uint64_t old = state_.load(std::memory_order_acquire);
  if (!IsSqWrId(id) || (old & kFrozen)) return false;
  auto freezeTracking = [&] {
    MORI_IO_ERROR("completion tracking frozen: ledger={} seq={} done={}",
                  static_cast<const void*>(this), seq, Done());
    FailAll(StatusCode::ERR_RDMA_OP, "completion tracking frozen", true);
    return false;
  };
  if (seq <= (old & kSqSequenceMask)) return freezeTracking();
  auto current = TakeEntry(seq);
  if (!current.valid) return freezeTracking();
  bool trackingFailed = false;
  for (uint64_t prior = (old & kSqSequenceMask) + 1; prior < seq; ++prior) {
    auto& entry = ring_[prior & mask_];
    if (entry.seq.load(std::memory_order_acquire) != prior) {
      Close(true);
      trackingFailed = true;
      break;
    }
    if (entry.kind != Kind::Data) {
      auto missing = TakeEntry(prior);
      if (!missing.valid) {
        trackingFailed = true;
        break;
      }
      MORI_IO_ERROR("signaled completion lost: SQ sequence={} covered_by={}", prior, seq);
      CompleteEntry(missing, IBV_WC_GENERAL_ERR, "signaled completion lost", true);
    }
  }
  while (!(old & kFrozen)) {
    auto next = (old & ~kSqSequenceMask) | seq;
    if (state_.compare_exchange_weak(old, next, std::memory_order_acq_rel)) {
      CancelReservation(static_cast<int>(seq - (old & kSqSequenceMask)));
      break;
    }
  }
  if (trackingFailed) freezeTracking();
  // Taking the entry transfers publication responsibility even if freezing won.
  CompleteEntry(current, status, message);
  return !Frozen();
}

int SubmissionLedger::FailAll(StatusCode code, const std::string& message, bool freeze) {
  std::vector<std::pair<std::shared_ptr<CqCallbackMeta>, TransferStatus*>> claimed;
  claimed.reserve(capacity_);
  int failed = 0;
  {
    std::lock_guard<std::mutex> lock(postMu_);
    state_.fetch_or(kClosed | (freeze ? kFrozen : 0), std::memory_order_acq_rel);
    for (uint64_t seq = Done() + 1; seq <= postedSeq_; ++seq) {
      auto& entry = ring_[seq & mask_];
      if (entry.kind != Kind::DataTail) continue;
      auto meta = std::atomic_load(&entry.meta);
      if (meta) {
        auto* s = meta->status.exchange(nullptr, std::memory_order_acq_rel);
        failed += s != nullptr;
        claimed.emplace_back(std::move(meta), s);
      }
    }
  }
  for (auto& item : claimed) {
    if (item.second) item.first->ReportData(item.second, code, message);
  }
  NotifyAdmission();
  return failed;
}

}  // namespace io
}  // namespace mori
