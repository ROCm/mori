// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "umbp/distributed/pool/fabric_read_staging.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <msgpack.hpp>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {

struct FabricReadStaging::State {
  State(TransferEngine* transfer, std::unique_ptr<PageMemorySource> memory, uint64_t page,
        uint64_t capacity)
      : engine(transfer), source(std::move(memory)), page_size(page), capacity_bytes(capacity) {}

  ~State() { ReleaseRegistration(); }

  void ReleaseRegistration() noexcept {
    if (!ref.HasMemoryDesc()) return;
    auto registered = std::move(ref);
    ref = TransferRef{};
    try {
      engine->Deregister(registered);
    } catch (...) {
      // A failed deregistration provides no proof that freeing this allocation
      // is safe. Its allocation source owns the arena, so retain that owner.
      (void)source.release();
    }
  }

  void Quarantine(const std::shared_ptr<State>& self) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    quarantined.store(true, std::memory_order_release);
    // Copying an existing shared_ptr allocates nothing. A cycle keeps both the
    // mapping and its registrar reference inert after the client shuts down.
    quarantine_hold = self;
  }

  TransferEngine* engine;
  std::unique_ptr<PageMemorySource> source;
  BackendRegistry* backends = nullptr;
  const uint64_t page_size;
  const uint64_t capacity_bytes;
  TransferRef ref;
  std::vector<BufferMemoryDescBytes> descs;
  mutable std::mutex mutex;
  std::vector<uint8_t> allocated;
  size_t free_pages = 0;
  size_t next_page = 0;
  bool initialized = false;
  bool init_attempted = false;
  std::atomic<bool> quarantined{false};
  std::shared_ptr<State> quarantine_hold;
};

struct FabricReadStaging::BatchHold {
  explicit BatchHold(std::shared_ptr<State> arena) : state(std::move(arena)) {}
  ~BatchHold() {
    if (!reserved) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    for (uint32_t page : pages) state->allocated[page] = 0;
    state->free_pages += pages.size();
  }
  std::shared_ptr<State> state;
  std::vector<uint32_t> pages;
  bool reserved = false;
};

// Constructed before posting anything. An uncertain drain retains this entire
// wrapper with release(), so cleanup cannot need a fresh allocation.
struct FabricReadStaging::InFlight {
  std::shared_ptr<BatchHold> hold;
  std::vector<std::shared_ptr<const void>> sources;
  std::unique_ptr<TransferHandle> handle;
};

FabricReadStaging::FabricReadStaging(TransferEngine* engine,
                                     std::unique_ptr<PageMemorySource> source, uint64_t page_size,
                                     uint64_t capacity_bytes)
    : state_(std::make_shared<State>(engine, std::move(source), page_size, capacity_bytes)) {}

FabricReadStaging::~FabricReadStaging() {
  // Stopping an RPC server does not finish a remote GPU read. Never let the
  // last outstanding pin later call a registrar destroyed with this client.
  if (state_.use_count() > 1) state_->Quarantine(state_);
}

bool FabricReadStaging::Init(BackendRegistry* backends, bool require_fabric_handle) {
  auto& state = *state_;
  if (state.initialized) return state.backends == backends;
  if (state.init_attempted) return false;
  state.init_attempted = true;
  if (state.engine == nullptr || state.source == nullptr || backends == nullptr ||
      state.source->LocationType() != mori::io::MemoryLocationType::GPU ||
      state.source->Device() < 0 || state.page_size == 0 || state.capacity_bytes == 0 ||
      state.capacity_bytes % state.page_size != 0 ||
      state.capacity_bytes > std::numeric_limits<size_t>::max() ||
      state.capacity_bytes / state.page_size > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  auto fail_init = [&]() {
    state.ReleaseRegistration();
    if (state.source) {
      try {
        state.source->Release();
      } catch (...) {
        (void)state.source.release();
      }
    }
    state.descs.clear();
    state.allocated.clear();
    state.free_pages = 0;
    return false;
  };
  try {
    state.allocated.resize(static_cast<size_t>(state.capacity_bytes / state.page_size), 0);
    std::vector<PageMemorySource::Buffer> buffers;
    if (!state.source->Allocate({state.capacity_bytes}, &buffers) || buffers.size() != 1 ||
        buffers.front().base == nullptr || buffers.front().size < state.capacity_bytes)
      return fail_init();
    state.ref = state.engine->RegisterMemoryWithBackends(
        buffers.front().base, static_cast<size_t>(state.capacity_bytes),
        state.source->LocationType(), state.source->Device(),
        mori::io::BackendBit(mori::io::BackendType::FABRIC));
    if (!state.ref.HasHostPtr() || !state.ref.HasMemoryDesc() ||
        state.ref.loc != mori::io::MemoryLocationType::GPU ||
        state.ref.mem.loc != mori::io::MemoryLocationType::GPU ||
        (require_fabric_handle &&
         (!(state.ref.mem.registeredBackendMask &
            mori::io::BackendBit(mori::io::BackendType::FABRIC)) ||
          std::all_of(state.ref.mem.fabricHandle.begin(), state.ref.mem.fabricHandle.end(),
                      [](char byte) { return byte == 0; })))) {
      return fail_init();
    }
    msgpack::sbuffer packed;
    msgpack::pack(packed, state.ref.mem);
    state.descs.push_back({0, {packed.data(), packed.data() + packed.size()}, kBackendId});
    state.free_pages = state.allocated.size();
    state.backends = backends;
    state.initialized = true;
    return true;
  } catch (...) {
    return fail_init();
  }
}

bool FabricReadStaging::Stage(std::vector<PoolResolvedEntry>* entries,
                              const std::vector<ObjectExtent>* extents, bool include_descs) {
  if (entries == nullptr) return false;
  auto state = state_;
  struct Candidate {
    size_t index;
    ObjectExtent window;
    ResolvedEntry replacement;
    std::vector<TransferItem> transfers;
  };
  std::vector<Candidate> candidates;
  bool all_ok = true;
  auto fail_entry = [&](size_t index, ResolveOutcome outcome) {
    (*entries)[index].resolved = ResolvedEntry{};
    (*entries)[index].resolved.outcome = outcome;
    all_ok = false;
  };
  if (extents != nullptr && extents->size() != entries->size()) {
    for (size_t i = 0; i < entries->size(); ++i)
      if ((*entries)[i].resolved.found) fail_entry(i, ResolveOutcome::kFailed);
    return false;
  }
  if (!state->initialized || state->quarantined.load(std::memory_order_acquire)) {
    for (size_t i = 0; i < entries->size(); ++i)
      if ((*entries)[i].resolved.found) fail_entry(i, ResolveOutcome::kFailed);
    return false;
  }

  size_t needed_pages = 0;
  std::unordered_map<uint64_t, TransferRef> source_buffers;
  auto local_source = [&](MediumBackend* backend, uint32_t backend_id, uint32_t buffer_index) {
    const uint64_t key = (static_cast<uint64_t>(backend_id) << 32) | buffer_index;
    auto found = source_buffers.find(key);
    if (found == source_buffers.end()) {
      const auto registered = backend->BufferRef(buffer_index);
      // Local H2D needs the raw endpoint only. Avoid copying a registration's
      // descriptor strings/maps for every source page and small KV fragment.
      found = source_buffers
                  .emplace(key, TransferRef::HostBytes(registered.host_ptr, registered.size,
                                                       registered.loc, registered.device))
                  .first;
    }
    return found->second;
  };
  const TransferRef local_destination = TransferRef::HostBytes(state->ref.host_ptr, state->ref.size,
                                                               state->ref.loc, state->ref.device);
  for (size_t i = 0; i < entries->size(); ++i) {
    auto& entry = (*entries)[i];
    const auto& resolved = entry.resolved;
    if (!resolved.found || resolved.size == 0) continue;
    auto* backend = state->backends->Get(entry.backend_id);
    if (backend == nullptr || resolved.pages.empty() || resolved.page_size == 0 ||
        resolved.file_ref.IsFile() || !resolved.hold) {
      fail_entry(i, ResolveOutcome::kFailed);
      continue;
    }
    TransferRef first =
        local_source(backend, entry.backend_id, resolved.pages.front().buffer_index);
    if (first.loc == mori::io::MemoryLocationType::GPU) continue;
    if (first.loc != mori::io::MemoryLocationType::CPU || !first.HasHostPtr()) {
      fail_entry(i, ResolveOutcome::kFailed);
      continue;
    }
    const uint64_t covered_start = resolved.covered_size == 0 ? 0 : resolved.covered_offset;
    const uint64_t covered_bytes =
        resolved.covered_size == 0 ? resolved.size : resolved.covered_size;
    ObjectExtent window{covered_start, covered_bytes};
    if (extents != nullptr) {
      window = (*extents)[i];
      if (window.length == 0) window = {0, resolved.size};
    }
    if (!RangeFits(covered_start, covered_bytes, resolved.size) ||
        !RangeFits(window.offset, window.length, resolved.size) || window.offset < covered_start ||
        !RangeFits(window.offset - covered_start, window.length, covered_bytes)) {
      fail_entry(i, ResolveOutcome::kFailed);
      continue;
    }
    const uint64_t pages = 1 + (window.length - 1) / state->page_size;
    if (pages > state->allocated.size() || needed_pages > state->allocated.size() - pages) {
      // This complete batch can never fit, even after every pin is released.
      // Do not ask the consumer to retry an unchanged impossible request.
      for (auto& candidate : candidates) fail_entry(candidate.index, ResolveOutcome::kFailed);
      candidates.clear();
      fail_entry(i, ResolveOutcome::kFailed);
      for (size_t j = i + 1; j < entries->size(); ++j) {
        auto& later = (*entries)[j];
        if (!later.resolved.found || later.resolved.pages.empty()) continue;
        auto* owner = state->backends->Get(later.backend_id);
        if (owner &&
            local_source(owner, later.backend_id, later.resolved.pages.front().buffer_index).loc ==
                mori::io::MemoryLocationType::CPU)
          fail_entry(j, ResolveOutcome::kFailed);
      }
      return false;
    }
    Candidate candidate{i, window, {}, {}};
    candidate.replacement.outcome = ResolveOutcome::kFound;
    candidate.replacement.found = true;
    candidate.replacement.size = resolved.size;
    candidate.replacement.page_size = state->page_size;
    candidate.replacement.covered_offset = window.offset;
    candidate.replacement.covered_size =
        window.offset == 0 && window.length == resolved.size ? 0 : window.length;
    candidate.replacement.pages.resize(static_cast<size_t>(pages));
    if (include_descs) candidate.replacement.descs = state->descs;
    bool valid = true;
    for (uint64_t copied = 0; copied < window.length;) {
      const uint64_t source_at = window.offset - covered_start + copied;
      const uint64_t source_page = source_at / resolved.page_size;
      if (source_page >= resolved.pages.size()) {
        valid = false;
        break;
      }
      const PageLocation& location = resolved.pages[static_cast<size_t>(source_page)];
      if (location.page_index > std::numeric_limits<uint64_t>::max() / resolved.page_size) {
        valid = false;
        break;
      }
      TransferRef source_ref = local_source(backend, entry.backend_id, location.buffer_index);
      const uint64_t in_page = source_at % resolved.page_size;
      const uint64_t source_offset =
          static_cast<uint64_t>(location.page_index) * resolved.page_size;
      const uint64_t bytes = std::min({window.length - copied, resolved.page_size - in_page,
                                       state->page_size - copied % state->page_size});
      if (!source_ref.HasHostPtr() || source_ref.loc != mori::io::MemoryLocationType::CPU ||
          !RangeFits(source_offset, in_page, source_ref.size) ||
          !RangeFits(source_offset + in_page, bytes, source_ref.size)) {
        valid = false;
        break;
      }
      // Destination offsets are relative to this candidate until admission.
      candidate.transfers.push_back(
          {std::move(source_ref), source_offset + in_page, local_destination, copied, bytes, i});
      copied += bytes;
    }
    if (!valid) {
      fail_entry(i, ResolveOutcome::kFailed);
      continue;
    }
    needed_pages += static_cast<size_t>(pages);
    candidates.push_back(std::move(candidate));
  }
  if (candidates.empty()) return all_ok;

  auto pending = std::make_unique<InFlight>();
  pending->hold = std::make_shared<BatchHold>(state);
  pending->hold->pages.resize(needed_pages);
  pending->sources.reserve(candidates.size());
  for (const auto& candidate : candidates)
    pending->sources.push_back((*entries)[candidate.index].resolved.hold);
  auto fail_batch = [&](ResolveOutcome outcome) {
    for (const auto& candidate : candidates) fail_entry(candidate.index, outcome);
  };
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->quarantined.load(std::memory_order_acquire)) {
      fail_batch(ResolveOutcome::kFailed);
      return false;
    }
    if (needed_pages > state->free_pages) {
      fail_batch(ResolveOutcome::kBusy);
      return false;
    }
    size_t found = 0;
    for (size_t scanned = 0; found < needed_pages; ++scanned) {
      const size_t page = (state->next_page + scanned) % state->allocated.size();
      if (state->allocated[page]) continue;
      state->allocated[page] = 1;
      pending->hold->pages[found++] = static_cast<uint32_t>(page);
    }
    state->next_page = (pending->hold->pages.back() + 1ull) % state->allocated.size();
    state->free_pages -= needed_pages;
    pending->hold->reserved = true;
  }

  // No mutation of published entries until the whole H2D batch is drained.
  // Planning failures release the reservation safely because nothing posted.
  bool post_attempted = false;
  try {
    std::vector<TransferItem> transfers;
    size_t cursor = 0;
    for (auto& candidate : candidates) {
      for (size_t j = 0; j < candidate.replacement.pages.size(); ++j)
        candidate.replacement.pages[j] = {0, pending->hold->pages[cursor + j]};
      for (auto& item : candidate.transfers) {
        const uint64_t at = item.dst_offset;
        item.dst_offset =
            static_cast<uint64_t>(pending->hold->pages[cursor + at / state->page_size]) *
                state->page_size +
            at % state->page_size;
        transfers.push_back(std::move(item));
      }
      cursor += candidate.replacement.pages.size();
      candidate.replacement.hold = pending->hold;
    }
    auto planned = state->engine->Plan(transfers);
    if (!planned.rejected_tags.empty() || planned.plans.empty()) {
      fail_batch(ResolveOutcome::kFailed);
      return false;
    }
    post_attempted = true;
    pending->handle = state->engine->Submit(std::move(planned.plans));
    if (!pending->handle) {
      post_attempted = false;
      fail_batch(ResolveOutcome::kFailed);
      return false;
    }
    std::vector<TransferFailure> failures;
    pending->handle->Wait(&failures);
    const bool quarantined = std::any_of(failures.begin(), failures.end(),
                                         [](const auto& f) { return f.resources_quarantined; });
    if (quarantined) {
      state->Quarantine(state);
      (void)pending.release();
      fail_batch(ResolveOutcome::kFailed);
      return false;
    }
    post_attempted = false;
    if (!failures.empty()) {
      fail_batch(ResolveOutcome::kFailed);
      return false;
    }
    for (auto& candidate : candidates) {
      auto& output = (*entries)[candidate.index];
      output.backend_id = kBackendId;
      // Preserve the object's owner tier: the temporary transport copy is not
      // a promotion or a new key placement.
      output.resolved = std::move(candidate.replacement);
    }
    return all_ok;
  } catch (...) {
    if (post_attempted) {
      state->Quarantine(state);
      (void)pending.release();
    }
    fail_batch(ResolveOutcome::kFailed);
    return false;
  }
}

std::vector<BufferMemoryDescBytes> FabricReadStaging::Descriptors() const {
  if (!state_->initialized || state_->quarantined.load(std::memory_order_acquire)) return {};
  return state_->descs;
}

uint64_t FabricReadStaging::CapacityBytes() const {
  return state_->initialized ? state_->capacity_bytes : 0;
}

uint64_t FabricReadStaging::InUseBytes() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (!state_->initialized) return 0;
  return (state_->allocated.size() - state_->free_pages) * state_->page_size;
}

bool FabricReadStaging::HasQuarantinedTransfers() const {
  return state_->quarantined.load(std::memory_order_acquire);
}

void FabricReadStaging::QuarantineBackingMemory() noexcept { state_->Quarantine(state_); }

}  // namespace mori::umbp
