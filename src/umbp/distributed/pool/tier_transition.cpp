// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/pool/tier_transition.h"

#include <algorithm>
#include <utility>

namespace mori::umbp {

bool TierTransitionExecutor::BuildTransfers(
    MediumBackend* source, const ResolvedEntry& resolved, MediumBackend* target,
    const AllocateResult& allocation, std::vector<TransferItem>* items) {
  if (source == nullptr || target == nullptr || items == nullptr || resolved.size == 0 ||
      resolved.page_size == 0 || allocation.page_size == 0 || resolved.pages.empty() ||
      allocation.pages.empty() || allocation.size != resolved.size) {
    return false;
  }

  items->clear();
  for (uint64_t copied = 0; copied < resolved.size;) {
    const uint64_t source_page_number = copied / resolved.page_size;
    const uint64_t target_page_number = copied / allocation.page_size;
    if (source_page_number >= resolved.pages.size() ||
        target_page_number >= allocation.pages.size()) {
      return false;
    }

    const auto& source_page = resolved.pages[static_cast<size_t>(source_page_number)];
    const auto& target_page = allocation.pages[static_cast<size_t>(target_page_number)];
    TransferRef source_ref = source->BufferRef(source_page.buffer_index);
    TransferRef target_ref = target->BufferRef(target_page.buffer_index);
    if (!source_ref.Valid() || !target_ref.Valid()) return false;

    const uint64_t source_offset = copied % resolved.page_size;
    const uint64_t target_offset = copied % allocation.page_size;
    const uint64_t size =
        std::min({resolved.size - copied, resolved.page_size - source_offset,
                  allocation.page_size - target_offset});
    TransferItem item;
    item.tag = items->size();
    item.src = std::move(source_ref);
    item.src_offset =
        static_cast<uint64_t>(source_page.page_index) * resolved.page_size + source_offset;
    item.dst = std::move(target_ref);
    item.dst_offset =
        static_cast<uint64_t>(target_page.page_index) * allocation.page_size + target_offset;
    item.size = size;
    items->push_back(std::move(item));
    copied += size;
  }
  return true;
}

TierTransitionResult TierTransitionExecutor::Execute(
    const std::string& key, uint32_t source_backend_id,
    const std::vector<uint32_t>& target_backend_ids, bool remove_source) const {
  TierTransitionResult result;
  if (backends_ == nullptr || transfer_engine_ == nullptr || target_backend_ids.empty()) {
    return result;
  }

  MediumBackend* source = backends_->Get(source_backend_id);
  ResolvedEntry resolved;
  if (source == nullptr || !source->AcquireMigrationRead(key, &resolved) || !resolved.found) {
    return result;
  }

  for (uint32_t target_backend_id : target_backend_ids) {
    MediumBackend* target = backends_->Get(target_backend_id);
    if (target == nullptr || target == source) continue;

    auto allocations = target->BatchAllocate({AllocateRequest{key, resolved.size}});
    if (allocations.empty()) continue;
    AllocateResult& allocation = allocations.front();

    if (allocation.outcome == AllocateOutcome::kSuccessAllocated) {
      std::vector<TransferItem> transfers;
      if (!BuildTransfers(source, resolved, target, allocation, &transfers) ||
          !transfer_engine_->Transfer(transfers, nullptr)) {
        target->BatchAbort({allocation.slot_id});
        continue;
      }
      auto committed = target->BatchCommit({CommitRequest{allocation.slot_id, key}});
      if (committed.empty() || !committed.front().success) {
        target->BatchAbort({allocation.slot_id});
        continue;
      }
    } else if (allocation.outcome != AllocateOutcome::kSuccessAlreadyExists) {
      continue;
    }

    ResolvedEntry target_read;
    if (!target->AcquireMigrationRead(key, &target_read) ||
        target_read.size != resolved.size) {
      if (target_read.found) target->ReleaseMigrationRead(key);
      if (allocation.outcome == AllocateOutcome::kSuccessAllocated) target->Evict({key});
      continue;
    }

    result.success = true;
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
