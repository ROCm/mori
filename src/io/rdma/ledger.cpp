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
#include <algorithm>

#include "mori/io/logging.hpp"
#include "src/io/rdma/common.hpp"
#include "src/io/rdma/telemetry.hpp"

namespace mori {
namespace io {

namespace {
// Smallest power of two >= v, with a floor so tiny SQ depths still get a
// reasonable ring. Capacity must be strictly greater than the max number of
// live records (bounded by maxSqDepth) so recordId & (cap-1) never aliases a
// still-live slot.
uint64_t RingCapacityFor(int maxSqDepth) {
  constexpr uint64_t kSlack = 64;    // headroom above maxSqDepth
  constexpr uint64_t kFloor = 256;   // minimum ring size
  uint64_t need = static_cast<uint64_t>(std::max(0, maxSqDepth)) + kSlack;
  uint64_t cap = kFloor;
  while (cap < need) cap <<= 1;
  return cap;
}
}  // namespace

CqCallbackMeta::~CqCallbackMeta() = default;

SubmissionLedger::SubmissionLedger(uint32_t notifPerQp, int maxSqDepth)
    : nextId_{notifPerQp} {
  const uint64_t cap = RingCapacityFor(maxSqDepth);
  capMask_ = cap - 1;
  ring_.resize(cap);  // default-constructed records have recordId == 0 (empty)
}

uint64_t SubmissionLedger::Insert(int postedWr, bool hasSignaledTail,
                                  std::shared_ptr<CqCallbackMeta> meta, int batchSize,
                                  uint64_t totalBytes) {
  std::lock_guard<SpinLock> lock(mu_);
  uint64_t id = nextId_++;
  SubmissionRecord& slot = ring_[id & capMask_];
  if (MORI_IO_TELEM_UNLIKELY(slot.recordId != 0)) {
    // Should be unreachable: admission control caps live records at maxSqDepth < capacity.
    MORI_IO_ERROR(
        "SubmissionLedger::Insert ring overflow: slot for recordId {} still holds live recordId "
        "{} (capacity {}); increase ring capacity",
        id, slot.recordId, capMask_ + 1);
  }
  slot = SubmissionRecord{id,
                          postedWr,
                          hasSignaledTail,
                          SubmissionState::Posted,
                          std::move(meta),
                          batchSize,
                          totalBytes,
                          /*post_tsc=*/0};
  return id;
}

void SubmissionLedger::RecordPostTimestamp(uint64_t recordId, uint64_t postTsc) {
  if (!timestamping_) return;
  std::lock_guard<SpinLock> lock(mu_);
  SubmissionRecord& slot = ring_[recordId & capMask_];
  if (slot.recordId == recordId) slot.post_tsc = postTsc;
}

void SubmissionLedger::InsertOrphaned(int postedWr, std::shared_ptr<CqCallbackMeta> meta,
                                      int batchSize, uint64_t totalBytes) {
  std::lock_guard<SpinLock> lock(mu_);
  uint64_t id = nextId_++;
  SubmissionRecord& slot = ring_[id & capMask_];
  if (MORI_IO_TELEM_UNLIKELY(slot.recordId != 0)) {
    MORI_IO_ERROR(
        "SubmissionLedger::InsertOrphaned ring overflow: slot for recordId {} still holds live "
        "recordId {} (capacity {}); increase ring capacity",
        id, slot.recordId, capMask_ + 1);
  }
  slot = SubmissionRecord{id,
                          postedWr,
                          false,
                          SubmissionState::Orphaned,
                          std::move(meta),
                          batchSize,
                          totalBytes,
                          /*post_tsc=*/0};
}

std::shared_ptr<CqCallbackMeta> SubmissionLedger::ReleaseByCqe(uint64_t recordId,
                                                               std::atomic<int>* sqDepth,
                                                               int* outBatchSize,
                                                               uint64_t* outTotalBytes,
                                                               uint64_t* out_post_tsc) {
  int postedWr = 0;
  std::shared_ptr<CqCallbackMeta> meta;
  {
    std::lock_guard<SpinLock> lock(mu_);
    SubmissionRecord& slot = ring_[recordId & capMask_];
    if (slot.recordId != recordId) return nullptr;  // stale / already released
    postedWr = slot.postedWr;
    if (outBatchSize) *outBatchSize = slot.batchSize;
    if (outTotalBytes) *outTotalBytes = slot.totalBytes;
    if (out_post_tsc) *out_post_tsc = slot.post_tsc;
    meta = std::move(slot.meta);
    slot.recordId = 0;  // mark empty
  }
  if (sqDepth && postedWr > 0) sqDepth->fetch_sub(postedWr, kSqAdmissionOrder);
  return meta;
}

int SubmissionLedger::ReleaseOrphanedByRecovery(std::atomic<int>* sqDepth) {
  int total = 0;
  {
    std::lock_guard<SpinLock> lock(mu_);
    // Only release Orphaned records.  Posted records still have signaled WRs
    // whose CQEs may arrive later; they must remain so ReleaseByCqe() can update
    // the corresponding TransferStatus and release sqDepth normally.
    for (SubmissionRecord& slot : ring_) {
      if (slot.recordId != 0 && slot.state == SubmissionState::Orphaned) {
        total += slot.postedWr;
        slot.meta.reset();
        slot.recordId = 0;  // mark empty
      }
    }
  }
  if (sqDepth && total > 0) sqDepth->fetch_sub(total, kSqAdmissionOrder);
  return total;
}

bool SubmissionLedger::HasOrphaned() const {
  std::lock_guard<SpinLock> lock(mu_);
  for (const SubmissionRecord& slot : ring_) {
    if (slot.recordId != 0 && slot.state == SubmissionState::Orphaned) return true;
  }
  return false;
}

}  // namespace io
}  // namespace mori
