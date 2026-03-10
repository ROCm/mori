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
#include "src/io/rdma/common.hpp"

namespace mori {
namespace io {

uint64_t SubmissionLedger::Insert(int postedWr, bool hasSignaledTail,
                                  std::shared_ptr<CqCallbackMeta> meta, int batchSize) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t id = nextId_++;
  records_[id] = SubmissionRecord{
      id, postedWr, hasSignaledTail, SubmissionState::Posted, std::move(meta), batchSize};
  return id;
}

void SubmissionLedger::InsertOrphaned(int postedWr, std::shared_ptr<CqCallbackMeta> meta,
                                      int batchSize) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t id = nextId_++;
  records_[id] =
      SubmissionRecord{id, postedWr, false, SubmissionState::Orphaned, std::move(meta), batchSize};
}

std::shared_ptr<CqCallbackMeta> SubmissionLedger::ReleaseByCqe(uint64_t recordId,
                                                               std::atomic<int>* sqDepth,
                                                               int* outBatchSize) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(recordId);
  if (it == records_.end()) return nullptr;
  SubmissionRecord& rec = it->second;
  if (sqDepth && rec.postedWr > 0) sqDepth->fetch_sub(rec.postedWr, std::memory_order_relaxed);
  if (outBatchSize) *outBatchSize = rec.batchSize;
  auto meta = std::move(rec.meta);
  records_.erase(it);
  return meta;
}

int SubmissionLedger::ReleaseOrphanedByRecovery(std::atomic<int>* sqDepth) {
  std::lock_guard<std::mutex> lock(mu_);
  int total = 0;
  // Only erase Orphaned records.  Posted records still have signaled WRs whose
  // CQEs may arrive later; they must remain so ReleaseByCqe() can update the
  // corresponding TransferStatus and release sqDepth normally.
  auto it = records_.begin();
  while (it != records_.end()) {
    if (it->second.state == SubmissionState::Orphaned) {
      total += it->second.postedWr;
      it = records_.erase(it);
    } else {
      ++it;
    }
  }
  if (sqDepth && total > 0) sqDepth->fetch_sub(total, std::memory_order_relaxed);
  return total;
}

bool SubmissionLedger::HasOrphaned() const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& [id, rec] : records_) {
    if (rec.state == SubmissionState::Orphaned) return true;
  }
  return false;
}

}  // namespace io
}  // namespace mori
