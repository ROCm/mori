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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/pool/tier_transition.h"

#include <algorithm>
#include <utility>

namespace mori::umbp {

namespace {

// Where byte `offset` of a value lives on one side: a file range holds all of
// it, a page holds the rest of that page.
bool Locate(MediumBackend* backend, const std::vector<PageLocation>& pages, uint64_t page_size,
            const TransferRef& file, uint64_t offset, TransferRef* ref, uint64_t* at,
            uint64_t* run) {
  if (file.IsFile()) {
    *ref = file;
    *at = offset;
    *run = UINT64_MAX;
    return true;
  }
  if (page_size == 0 || offset / page_size >= pages.size()) return false;
  const PageLocation& page = pages[static_cast<size_t>(offset / page_size)];
  *ref = backend->BufferRef(page.buffer_index);
  *at = static_cast<uint64_t>(page.page_index) * page_size + offset % page_size;
  *run = page_size - offset % page_size;
  return ref->Valid();
}

}  // namespace

bool TierTransitionExecutor::BuildTransfers(MediumBackend* source, const ResolvedEntry& resolved,
                                            MediumBackend* target, const AllocateResult& allocation,
                                            std::vector<TransferItem>* items) {
  if (source == nullptr || target == nullptr || items == nullptr || resolved.size == 0 ||
      allocation.size != resolved.size) {
    return false;
  }

  items->clear();
  for (uint64_t copied = 0; copied < resolved.size;) {
    TransferItem item;
    uint64_t src_run = 0;
    uint64_t dst_run = 0;
    if (!Locate(source, resolved.pages, resolved.page_size, resolved.file_ref, copied, &item.src,
                &item.src_offset, &src_run) ||
        !Locate(target, allocation.pages, allocation.page_size, allocation.file_ref, copied,
                &item.dst, &item.dst_offset, &dst_run)) {
      return false;
    }
    item.tag = items->size();
    item.size = std::min({resolved.size - copied, src_run, dst_run});
    copied += item.size;
    items->push_back(std::move(item));
  }
  return true;
}

// An engine joins `backend`'s memory and a file both ways; CanHandle is pure,
// so a placeholder range answers.
bool TierTransitionExecutor::FileReachable(MediumBackend* backend) const {
  const TransferRef mem = backend->BufferRef(0);
  const TransferRef file = TransferRef::File(/*fd=*/0, 0, 1);
  return mem.Valid() && transfer_engine_->CanHandle(mem, file) &&
         transfer_engine_->CanHandle(file, mem);
}

TierTransitionResult TierTransitionExecutor::Execute(
    const std::string& key, uint32_t source_backend_id,
    const std::vector<uint32_t>& target_backend_ids, bool remove_source) const {
  TierTransitionResult result;
  if (backends_ == nullptr || transfer_engine_ == nullptr || target_backend_ids.empty()) {
    return result;
  }

  MediumBackend* source = backends_->Get(source_backend_id);
  if (source == nullptr) return result;
  // A file-backed side skips staging when an engine joins its file to the other
  // side's memory.
  bool targets_reach_files = true;
  for (uint32_t target_backend_id : target_backend_ids) {
    MediumBackend* target = backends_->Get(target_backend_id);
    if (target != nullptr && target != source) targets_reach_files &= FileReachable(target);
  }
  ResolvedEntry resolved;
  if (!source->AcquireMigrationRead(key, &resolved, targets_reach_files) || !resolved.found) {
    return result;
  }
  const bool source_reaches_files = FileReachable(source);

  for (uint32_t target_backend_id : target_backend_ids) {
    MediumBackend* target = backends_->Get(target_backend_id);
    if (target == nullptr || target == source) continue;

    auto allocations =
        target->BatchAllocate({AllocateRequest{key, resolved.size, -1, source_reaches_files}});
    if (allocations.empty()) continue;
    AllocateResult& allocation = allocations.front();

    if (allocation.outcome == AllocateOutcome::kSuccessAllocated) {
      std::vector<TransferItem> transfers;
      if (!BuildTransfers(source, resolved, target, allocation, &transfers) ||
          !transfer_engine_->Transfer(transfers, nullptr)) {
        target->BatchAbort({allocation.slot_id});
        result.retryable = true;
        continue;
      }
      result.bytes_read += resolved.size;
      auto committed = target->BatchCommit({CommitRequest{allocation.slot_id, key}});
      if (committed.empty() || !committed.front().success) {
        target->BatchAbort({allocation.slot_id});
        result.retryable = true;
        continue;
      }
      result.bytes_written.emplace_back(target->Tier(), committed.front().bytes_committed);
    } else if (allocation.outcome != AllocateOutcome::kSuccessAlreadyExists) {
      // No space is pressure (staging full of reads, eviction catching up);
      // a shape failure (kFailed) stays final.
      if (allocation.outcome == AllocateOutcome::kFailedNoSpace) result.retryable = true;
      continue;
    }

    // A pin, not a read-back: the bytes were just written.
    ResolvedEntry target_read;
    if (!target->AcquireMigrationRead(key, &target_read, /*allow_file_ref=*/true) ||
        target_read.size != resolved.size) {
      if (target_read.found) target->ReleaseMigrationRead(key);
      if (allocation.outcome == AllocateOutcome::kSuccessAllocated) target->Evict({key});
      result.retryable = true;
      continue;
    }

    result.success = true;
    result.retryable = false;
    result.target_backend_id = target_backend_id;
    result.bytes_moved = resolved.size;
    source->ReleaseMigrationRead(key);
    if (remove_source) {
      auto evicted = source->Evict({key});
      const uint64_t freed = evicted.empty() ? 0 : evicted.front().bytes_freed;
      result.source_draining = freed == 0 && source->Contains(key);
      result.bytes_freed = freed;
    }
    target->ReleaseMigrationRead(key);
    return result;
  }

  source->ReleaseMigrationRead(key);
  return result;
}

}  // namespace mori::umbp
