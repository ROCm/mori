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
      id, postedWr, hasSignaledTail, SubmissionState::Tentative, std::move(meta), batchSize};
  return id;
}

uint64_t SubmissionLedger::InsertOrphaned(int postedWr, std::shared_ptr<CqCallbackMeta> meta,
                                          int batchSize) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t id = nextId_++;
  records_[id] =
      SubmissionRecord{id, postedWr, false, SubmissionState::Orphaned, std::move(meta), batchSize};
  return id;
}

bool SubmissionLedger::ReleaseByCqe(uint64_t recordId, SubmissionRecord* outRecord) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(recordId);
  if (it == records_.end()) return false;
  if (outRecord != nullptr) *outRecord = std::move(it->second);
  records_.erase(it);
  return true;
}

bool SubmissionLedger::MarkPosted(uint64_t recordId) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(recordId);
  if (it == records_.end()) return false;
  if (it->second.state != SubmissionState::Tentative || !it->second.hasSignaledTail) return false;
  it->second.state = SubmissionState::Posted;
  return true;
}

bool SubmissionLedger::CancelTentative(uint64_t recordId, SubmissionRecord* outRecord) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(recordId);
  if (it == records_.end()) return false;
  if (it->second.state != SubmissionState::Tentative || !it->second.hasSignaledTail) return false;
  if (outRecord != nullptr) *outRecord = std::move(it->second);
  records_.erase(it);
  return true;
}

void SubmissionLedger::ExtractOrphanedRecords(std::vector<SubmissionRecord>* outRecords) {
  std::lock_guard<std::mutex> lock(mu_);
  if (outRecords != nullptr) outRecords->clear();
  auto it = records_.begin();
  while (it != records_.end()) {
    if (it->second.state == SubmissionState::Orphaned) {
      if (outRecords != nullptr) outRecords->push_back(std::move(it->second));
      it = records_.erase(it);
    } else {
      ++it;
    }
  }
}

bool SubmissionLedger::HasOrphaned() const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& [id, rec] : records_) {
    if (rec.state == SubmissionState::Orphaned) return true;
  }
  return false;
}

size_t SubmissionLedger::RecordCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  return records_.size();
}

}  // namespace io
}  // namespace mori
