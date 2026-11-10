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
#include <chrono>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/peer/backend/engine_hbm_backend.h"
#include "umbp/distributed/peer/backend/instrumented_backend.h"

namespace mori::umbp {

namespace {

struct FailureLogState {
  uint64_t count = 0;
  uint64_t since_last = 0;
  std::chrono::steady_clock::time_point last{};
};

void WarnTransitionFailure(std::string_view reason, const std::string& key, uint32_t source,
                           uint32_t target, std::string_view detail) {
  static std::mutex mutex;
  static std::unordered_map<std::string, FailureLogState> states;
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex);
  auto& state = states[std::string(reason)];
  ++state.count;
  ++state.since_last;
  if (state.last != std::chrono::steady_clock::time_point{} &&
      now - state.last < std::chrono::seconds(10)) {
    return;
  }
  MORI_UMBP_WARN(
      "[TierTransition] failure reason={} count={} since_last={} key='{}' source_backend={} "
      "target_backend={} detail='{}'",
      reason, state.count, state.since_last, key, source, target, detail);
  state.since_last = 0;
  state.last = now;
}

void RecordFailure(TierTransitionResult* result, std::string reason, std::string detail,
                   const std::string& key, uint32_t source, uint32_t target) {
  if (result != nullptr) {
    result->failure_reason = reason;
    result->failure_detail = detail;
    result->failed_target_backend_id = target;
  }
  WarnTransitionFailure(reason, key, source, target, detail);
}

std::string DescribeFailures(const std::vector<TransferFailure>& failures) {
  std::ostringstream out;
  for (size_t i = 0; i < failures.size(); ++i) {
    if (i != 0) out << "; ";
    out << "code=" << failures[i].code << " msg=" << failures[i].message
        << " endpoint=" << failures[i].endpoint << " submitted=" << failures[i].submitted
        << " quiesced=" << failures[i].quiesced;
  }
  return out.str();
}

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
  if (!resolved.view.empty() && !allocation.view.empty()) {
    return BuildObjectViewTransfers(resolved.view, allocation.view, items);
  }
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
    const std::vector<uint32_t>& target_backend_ids, bool remove_source,
    std::optional<engine::OperationId> retire_operation) const {
  TierTransitionResult result;
  if (backends_ == nullptr || transfer_engine_ == nullptr || target_backend_ids.empty()) {
    RecordFailure(&result, "INVALID_EXECUTOR", "missing registry, transfer engine, or target", key,
                  source_backend_id, BackendRegistry::kMaxBackends);
    return result;
  }

  MediumBackend* source = backends_->Get(source_backend_id);
  if (source == nullptr) {
    RecordFailure(&result, "SOURCE_BACKEND_MISSING", "source backend lookup failed", key,
                  source_backend_id, BackendRegistry::kMaxBackends);
    return result;
  }
  // A file-backed side skips staging when an engine joins its file to the other
  // side's memory.
  bool targets_reach_files = true;
  for (uint32_t target_backend_id : target_backend_ids) {
    MediumBackend* target = backends_->Get(target_backend_id);
    if (target != nullptr && target != source) targets_reach_files &= FileReachable(target);
  }
  ResolvedEntry resolved;
  bool source_acquired = false;
  if (retire_operation.has_value()) {
    MediumBackend* unwrapped = source;
    if (auto* instrumented = dynamic_cast<InstrumentedBackend*>(source)) {
      unwrapped = instrumented->Inner();
    }
    auto* engine_source = dynamic_cast<EngineHbmBackend*>(unwrapped);
    source_acquired = engine_source != nullptr &&
                      engine_source->AcquireRetireMigrationRead(key, *retire_operation, &resolved);
  } else {
    source_acquired = source->AcquireMigrationRead(key, &resolved, targets_reach_files);
  }
  if (!source_acquired || !resolved.found) {
    result.source_missing = true;
    RecordFailure(&result, "SOURCE_ACQUIRE_FAILED",
                  source_acquired ? "resolved entry is missing" : "migration read rejected", key,
                  source_backend_id, BackendRegistry::kMaxBackends);
    return result;
  }
  const bool source_reaches_files = FileReachable(source);

  for (uint32_t target_backend_id : target_backend_ids) {
    MediumBackend* target = backends_->Get(target_backend_id);
    if (target == nullptr || target == source) {
      RecordFailure(&result, "TARGET_BACKEND_INVALID",
                    target == nullptr ? "target backend lookup failed" : "target equals source",
                    key, source_backend_id, target_backend_id);
      continue;
    }

    auto allocations =
        target->BatchAllocate({AllocateRequest{key, resolved.size, -1, source_reaches_files,
                                               /*acquire_write_hold=*/true}});
    if (allocations.empty()) {
      RecordFailure(&result, "TARGET_ALLOCATE_EMPTY", "BatchAllocate returned no result", key,
                    source_backend_id, target_backend_id);
      continue;
    }
    AllocateResult& allocation = allocations.front();

    if (allocation.outcome == AllocateOutcome::kSuccessAllocated) {
      std::vector<TransferItem> transfers;
      if (!BuildTransfers(source, resolved, target, allocation, &transfers)) {
        const bool quiesced = !allocation.write_hold ||
                              target->FinishPendingWrite(allocation.slot_id, allocation.write_hold,
                                                         PendingWriteOutcome::kNeverSubmitted);
        if (quiesced) target->BatchAbort({allocation.slot_id});
        result.retryable = false;  // geometry does not change between attempts
        RecordFailure(&result, "BUILD_TRANSFERS_FAILED",
                      "source/target ObjectView intersection is invalid", key, source_backend_id,
                      target_backend_id);
        continue;
      }
      bool transfer_quiesced = true;
      bool transfer_submitted = false;
      std::vector<size_t> failed_tags;
      std::vector<TransferFailure> transfer_failures;
      if (!transfer_engine_->Transfer(transfers, &failed_tags, nullptr, &transfer_quiesced,
                                      &transfer_submitted, &transfer_failures)) {
        const PendingWriteOutcome outcome = !transfer_quiesced ? PendingWriteOutcome::kUnknown
                                            : !transfer_submitted
                                                ? PendingWriteOutcome::kNeverSubmitted
                                                : PendingWriteOutcome::kCompleted;
        const bool quiesced =
            !allocation.write_hold ||
            target->FinishPendingWrite(allocation.slot_id, allocation.write_hold, outcome);
        if (quiesced) target->BatchAbort({allocation.slot_id});
        const std::string detail = transfer_failures.empty()
                                       ? "transfer failed without details; failed_tags=" +
                                             std::to_string(failed_tags.size())
                                       : DescribeFailures(transfer_failures);
        RecordFailure(&result, "TRANSFER_FAILED", detail, key, source_backend_id,
                      target_backend_id);
        if (!transfer_quiesced) {
          result.unknown = true;
          return result;  // keep the source hold: a late DMA may still read it
        }
        const bool deterministic = std::any_of(
            transfer_failures.begin(), transfer_failures.end(), [](const TransferFailure& failure) {
              return failure.message.find("could not hydrate") != std::string::npos ||
                     failure.message.find("planning rejected") != std::string::npos;
            });
        result.retryable = !deterministic;
        continue;
      }
      result.bytes_read += resolved.size;
      if (allocation.write_hold &&
          !target->FinishPendingWrite(allocation.slot_id, allocation.write_hold,
                                      PendingWriteOutcome::kCompleted)) {
        result.retryable = true;
        RecordFailure(&result, "FINISH_PENDING_WRITE_FAILED", "write hold was rejected", key,
                      source_backend_id, target_backend_id);
        continue;
      }
      auto committed = target->BatchCommit({CommitRequest{allocation.slot_id, key}});
      if (committed.empty() || !committed.front().success) {
        target->BatchAbort({allocation.slot_id});
        result.retryable = true;
        RecordFailure(&result, "TARGET_COMMIT_FAILED", "BatchCommit rejected pending slot", key,
                      source_backend_id, target_backend_id);
        continue;
      }
      result.bytes_written.emplace_back(target->Tier(), committed.front().bytes_committed);
    } else if (allocation.outcome != AllocateOutcome::kSuccessAlreadyExists) {
      // No space is pressure (staging full of reads, eviction catching up);
      // a shape failure (kFailed) stays final.
      if (allocation.outcome == AllocateOutcome::kFailedNoSpace) result.retryable = true;
      RecordFailure(
          &result, "TARGET_ALLOCATE_FAILED",
          "allocation outcome=" + std::to_string(static_cast<uint32_t>(allocation.outcome)), key,
          source_backend_id, target_backend_id);
      continue;
    }

    // A pin, not a read-back: the bytes were just written.
    ResolvedEntry target_read;
    if (!target->AcquireMigrationRead(key, &target_read, /*allow_file_ref=*/true) ||
        target_read.size != resolved.size) {
      if (target_read.found) target->ReleaseMigrationRead(key);
      if (allocation.outcome == AllocateOutcome::kSuccessAllocated) target->Evict({key});
      result.retryable = true;
      RecordFailure(&result, "TARGET_READBACK_FAILED", "committed target could not be pinned", key,
                    source_backend_id, target_backend_id);
      continue;
    }

    result.success = true;
    result.retryable = false;
    result.failure_reason.clear();
    result.failure_detail.clear();
    result.failed_target_backend_id = BackendRegistry::kMaxBackends;
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
