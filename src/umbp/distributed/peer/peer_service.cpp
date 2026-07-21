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
#include "umbp/distributed/peer/peer_service.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/batch_resolve_codec.h"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp/distributed/peer/ssd_copy_pipeline.h"
#include "umbp/distributed/types.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {
// Shared with master_server.cpp via UMBP_GRPC_SHUTDOWN_DEADLINE_SEC.
std::chrono::seconds GrpcShutdownDeadline() {
  static const auto v =
      GetEnvSeconds("UMBP_GRPC_SHUTDOWN_DEADLINE_SEC", std::chrono::seconds(3), /*min_allowed=*/1);
  return v;
}

// Free -> Preparing -> Leased.  Preparing slots (IO in flight) are never
// reclaimed, so a slow IO can't have its slot reassigned mid-write.  The TTL is
// anchored at request receipt (leased_at = received_at, set on promotion) to
// align the peer's reclaim point (received_at + ttl) with the reader's deadline
// (t_send + ttl, t_send < received_at).  A reader trusting bytes from a slot
// reclaimed mid-use is prevented by Preparing (IO safety) + the reader's own
// lease gating (see ssd_read_lease.h).
enum class SlotState { kFree, kPreparing, kLeased };

struct StagingSlot {
  SlotState state = SlotState::kFree;
  uint64_t lease_id = 0;
  size_t allocated_size = 0;
  std::chrono::steady_clock::time_point leased_at;  // valid only while kLeased
};

// Reclaim TTL-expired leased slots, then claim a free one as Preparing (TTL not
// yet started) and return its index, or -1 if none free.
int ClaimStagingSlot(std::vector<StagingSlot>& slots, std::atomic<uint64_t>& next_lease_id,
                     std::chrono::milliseconds lease_timeout, StagingMetrics& metrics) {
  auto now = std::chrono::steady_clock::now();
  for (auto& slot : slots) {
    if (slot.state == SlotState::kLeased && now - slot.leased_at > lease_timeout) {
      metrics.expired_reclaims.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_WARN("[PeerService] Reclaiming expired slot (lease_id={})", slot.lease_id);
      slot.state = SlotState::kFree;
    }
  }
  for (auto& slot : slots) {
    if (slot.state == SlotState::kFree) {
      slot.state = SlotState::kPreparing;
      slot.lease_id = next_lease_id.fetch_add(1, std::memory_order_relaxed);
      slot.allocated_size = 0;
      return static_cast<int>(&slot - &slots[0]);
    }
  }
  return -1;
}

bool ReleaseSlotByLeaseId(std::vector<StagingSlot>& slots, uint64_t lease_id) {
  for (auto& slot : slots) {
    if (slot.state == SlotState::kLeased && slot.lease_id == lease_id) {
      slot.state = SlotState::kFree;
      return true;
    }
  }
  return false;
}
}  // namespace

namespace {

// Translate proto TierType <-> umbp::TierType.  Defined inline because
// only the peer service handlers need them.
TierType FromProtoTier(::umbp::TierType t) {
  switch (t) {
    case ::umbp::TIER_HBM:
      return TierType::HBM;
    case ::umbp::TIER_DRAM:
      return TierType::DRAM;
    case ::umbp::TIER_SSD:
      return TierType::SSD;
    default:
      return TierType::UNKNOWN;
  }
}
::umbp::TierType ToProtoTier(TierType t) {
  switch (t) {
    case TierType::HBM:
      return ::umbp::TIER_HBM;
    case TierType::DRAM:
      return ::umbp::TIER_DRAM;
    case TierType::SSD:
      return ::umbp::TIER_SSD;
    default:
      return ::umbp::TIER_UNKNOWN;
  }
}

// Map a PeerSsdManager read outcome (data-level: ok/not_found/size/error) to the
// wire status.  NO_SLOT is a staging-layer concern owned by the peer service,
// not PeerSsdManager, and is set directly by the handler.  Lease expiry is not
// a wire status: the reader decides it locally and the peer reclaims by TTL.
::umbp::SsdReadStatus ToProtoReadStatus(SsdReadStatus s) {
  switch (s) {
    case SsdReadStatus::kOk:
      return ::umbp::SSD_READ_OK;
    case SsdReadStatus::kNotFound:
      return ::umbp::SSD_READ_NOT_FOUND;
    case SsdReadStatus::kSizeTooLarge:
      return ::umbp::SSD_READ_SIZE_TOO_LARGE;
    case SsdReadStatus::kError:
      return ::umbp::SSD_READ_ERROR;
  }
  return ::umbp::SSD_READ_ERROR;
}

// Drop a (pages, page_size, descs) tuple into a slot-shaped response
// that exposes those fields directly.  Templated so the same body
// covers AllocateSlotResponse and ResolveKeyResponse.
template <typename Response>
void FillPagesAndDescs(Response* resp, const std::vector<PageLocation>& pages, uint64_t page_size,
                       const std::vector<BufferMemoryDescBytes>& descs) {
  for (const auto& p : pages) {
    auto* pl = resp->add_pages();
    pl->set_buffer_index(p.buffer_index);
    pl->set_page_index(p.page_index);
  }
  resp->set_page_size(page_size);
  for (const auto& d : descs) {
    auto* desc = resp->add_descs();
    desc->set_buffer_index(d.buffer_index);
    desc->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
  }
}

}  // namespace

class PeerServiceServer::UMBPPeerServiceImpl final : public ::umbp::UMBPPeer::Service {
 public:
  UMBPPeerServiceImpl(void* ssd_staging_base, size_t ssd_staging_size,
                      const std::vector<uint8_t>& ssd_staging_mem_desc_bytes,
                      PeerSsdManager* peer_ssd, PeerDramAllocator* dram_alloc,
                      MasterClient* master_client, StagingMetrics& metrics, int num_read_slots,
                      std::chrono::milliseconds lease_timeout,
                      const std::vector<uint8_t>& engine_desc_bytes, SsdCopyPipeline* copy_pipeline)
      : ssd_staging_base_(ssd_staging_base),
        ssd_staging_size_(ssd_staging_size),
        ssd_staging_mem_desc_bytes_(ssd_staging_mem_desc_bytes),
        peer_ssd_(peer_ssd),
        dram_alloc_(dram_alloc),
        copy_pipeline_(copy_pipeline),
        engine_desc_bytes_(engine_desc_bytes),
        metrics_(metrics),
        lease_timeout_(std::max(lease_timeout, std::chrono::milliseconds{1})),
        num_read_slots_(std::max(num_read_slots, 1)),
        // The whole staging buffer is the read region now: the lower half used
        // to be direct-put write staging, removed in the SSD-tier redesign and
        // reclaimed here so reads get more / larger slots.
        read_region_base_(0),
        read_slot_size_(ssd_staging_size / static_cast<size_t>(std::max(num_read_slots, 1))),
        read_slots_(std::max(num_read_slots, 1)),
        master_client_(master_client) {
    if (num_read_slots <= 0) {
      MORI_UMBP_ERROR("[PeerService] num_read_slots={} invalid, clamped to 1", num_read_slots);
    }
  }

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    response->set_ssd_staging_mem_desc(
        std::string(ssd_staging_mem_desc_bytes_.begin(), ssd_staging_mem_desc_bytes_.end()));
    response->set_ssd_staging_size(ssd_staging_size_);
    if (!engine_desc_bytes_.empty()) {
      response->set_engine_desc(std::string(engine_desc_bytes_.begin(), engine_desc_bytes_.end()));
    }
    if (dram_alloc_ != nullptr) {
      // Ship every configured DRAM/HBM buffer's desc so first-contact
      // writers can hydrate without a follow-up Allocate / Resolve.
      // DRAM and HBM share a single page_size in this design, so the
      // single field on the response is sufficient.
      auto dram_descs = dram_alloc_->AllBufferDescs(TierType::DRAM);
      auto hbm_descs = dram_alloc_->AllBufferDescs(TierType::HBM);
      for (const auto& d : dram_descs) {
        auto* out = response->add_dram_memory_descs();
        out->set_buffer_index(d.buffer_index);
        out->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
      }
      for (const auto& d : hbm_descs) {
        auto* out = response->add_dram_memory_descs();
        out->set_buffer_index(d.buffer_index);
        out->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
      }
      response->set_dram_page_size(dram_alloc_->PageSize());
    }
    return grpc::Status::OK;
  }

  // ---- SSD read staging: PrepareSsdRead + ReleaseSsdLease ----
  // Key-based: claim a slot -> PeerSsdManager::PrepareRead fills it -> reader
  // RDMAs the bytes out of the published staging buffer -> best-effort release.
  grpc::Status PrepareSsdRead(grpc::ServerContext* /*context*/,
                              const ::umbp::PrepareSsdReadRequest* request,
                              ::umbp::PrepareSsdReadResponse* response) override {
    if (!SsdRpcAvailable()) {
      response->set_status(::umbp::SSD_READ_ERROR);
      return grpc::Status::OK;
    }

    // Anchor the lease TTL at request receipt (see SlotState comment): the slot
    // is promoted to Leased with this timestamp once the IO completes, so the
    // peer's reclaim point stays aligned with the reader's t_send-based
    // deadline rather than starting only after the (variable) SSD IO.
    const auto received_at = std::chrono::steady_clock::now();

    // Claim a slot first (Preparing — not yet TTL-tracked).  NO_SLOT is a
    // transient/retryable condition, not a miss.
    int slot_idx;
    uint64_t offset, lease_id;
    {
      std::lock_guard<std::mutex> lock(read_slots_mutex_);
      slot_idx = ClaimStagingSlot(read_slots_, next_lease_id_, lease_timeout_, metrics_);
      if (slot_idx < 0) {
        metrics_.slot_full_rejects.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_WARN("[PeerService] PrepareSsdRead: no free staging slots");
        response->set_status(::umbp::SSD_READ_NO_SLOT);
        return grpc::Status::OK;
      }
      offset = read_region_base_ + static_cast<uint64_t>(slot_idx) * read_slot_size_;
      lease_id = read_slots_[slot_idx].lease_id;
    }

    // Cap the read at min(reader capacity, slot size); PrepareRead rejects an
    // over-cap key BEFORE doing any SSD IO, so an oversized key costs no read.
    void* dst = static_cast<uint8_t*>(ssd_staging_base_) + offset;
    size_t cap = std::min<uint64_t>(request->max_size(), read_slot_size_);
    SsdReadOutcome outcome = peer_ssd_->PrepareRead(request->key(), dst, cap);

    if (outcome.status != SsdReadStatus::kOk) {
      std::lock_guard<std::mutex> lock(read_slots_mutex_);
      read_slots_[slot_idx].state = SlotState::kFree;  // give the slot straight back
      response->set_status(ToProtoReadStatus(outcome.status));
      response->set_size(outcome.size);
      return grpc::Status::OK;
    }

    // Data is in the slot: promote Preparing -> Leased with the request-receipt
    // TTL anchor (leased_at = received_at).
    {
      std::lock_guard<std::mutex> lock(read_slots_mutex_);
      auto& slot = read_slots_[slot_idx];
      slot.state = SlotState::kLeased;
      slot.leased_at = received_at;
      slot.allocated_size = outcome.size;
    }
    response->set_status(::umbp::SSD_READ_OK);
    response->set_staging_offset(offset);
    response->set_size(outcome.size);
    response->set_lease_id(lease_id);
    response->set_lease_ttl_ms(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(lease_timeout_).count()));
    MORI_UMBP_DEBUG(
        "[PeerService] PrepareSsdRead: key={}, slot={}, offset={}, size={}, lease_id={}",
        request->key(), slot_idx, offset, outcome.size, lease_id);
    return grpc::Status::OK;
  }

  grpc::Status ReleaseSsdLease(grpc::ServerContext* /*context*/,
                               const ::umbp::ReleaseSsdLeaseRequest* request,
                               ::umbp::ReleaseSsdLeaseResponse* response) override {
    // Best-effort fast release; correctness does not depend on it (the slot is
    // also reclaimed by the lease TTL).  Returns false only when the lease is
    // already gone (released earlier or TTL-reclaimed).
    std::lock_guard<std::mutex> lock(read_slots_mutex_);
    response->set_success(ReleaseSlotByLeaseId(read_slots_, request->lease_id()));
    return grpc::Status::OK;
  }

  // ============================================================
  //  DRAM/HBM allocator + key map (master-as-advisor design)
  // ============================================================

  grpc::Status AllocateSlot(grpc::ServerContext* /*ctx*/,
                            const ::umbp::AllocateSlotRequest* request,
                            ::umbp::AllocateSlotResponse* response) override {
    if (dram_alloc_ == nullptr) {
      response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
      return grpc::Status::OK;
    }
    auto result =
        dram_alloc_->Allocate(request->key(), request->size(), FromProtoTier(request->tier()));
    switch (result.outcome) {
      case PeerDramAllocator::Outcome::kSuccessAlreadyExists:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS);
        return grpc::Status::OK;
      case PeerDramAllocator::Outcome::kFailed:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
        return grpc::Status::OK;
      case PeerDramAllocator::Outcome::kFailedNoSpace:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE);
        return grpc::Status::OK;
      case PeerDramAllocator::Outcome::kSuccessAllocated:
        break;
    }
    const auto& pending = *result.slot;
    auto descs = dram_alloc_->BufferDescsForPages(pending.tier, pending.pages);
    response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
    response->set_slot_id(pending.slot_id);
    FillPagesAndDescs(response, pending.pages, dram_alloc_->PageSize(), descs);
    response->set_pending_ttl_ms(dram_alloc_->PendingTtlMs());
    return grpc::Status::OK;
  }

  grpc::Status CommitSlot(grpc::ServerContext* /*ctx*/, const ::umbp::CommitSlotRequest* request,
                          ::umbp::CommitSlotResponse* response) override {
    if (dram_alloc_ == nullptr) {
      response->set_success(false);
      return grpc::Status::OK;
    }
    uint64_t committed_bytes = 0;
    const bool ok = dram_alloc_->Commit(request->slot_id(), request->key(), committed_bytes);
    response->set_success(ok);
    if (ok) {
      RecordInboundPut(committed_bytes, "remote");
      EnqueueSsdCopy(request->key(), committed_bytes);
    }
    return grpc::Status::OK;
  }

  grpc::Status AbortSlot(grpc::ServerContext* /*ctx*/, const ::umbp::AbortSlotRequest* request,
                         ::umbp::AbortSlotResponse* response) override {
    if (dram_alloc_ == nullptr) {
      response->set_success(true);  // idempotent: nothing to drop
      return grpc::Status::OK;
    }
    response->set_success(dram_alloc_->Abort(request->slot_id()));
    return grpc::Status::OK;
  }

  grpc::Status ResolveKey(grpc::ServerContext* /*ctx*/, const ::umbp::ResolveKeyRequest* request,
                          ::umbp::ResolveKeyResponse* response) override {
    if (dram_alloc_ == nullptr) {
      response->set_found(false);
      return grpc::Status::OK;
    }
    auto r = dram_alloc_->Resolve(request->key());
    response->set_found(r.found);
    if (!r.found) return grpc::Status::OK;
    auto descs = dram_alloc_->BufferDescsForPages(r.tier, r.pages);
    FillPagesAndDescs(response, r.pages, dram_alloc_->PageSize(), descs);
    response->set_size(r.size);
    RecordInboundGet(r.size, "remote");
    return grpc::Status::OK;
  }

  grpc::Status EvictKey(grpc::ServerContext* /*ctx*/, const ::umbp::EvictKeyRequest* request,
                        ::umbp::EvictKeyResponse* response) override {
    if (dram_alloc_ == nullptr) {
      // No DRAM/HBM tier on this peer — nothing to evict, treat as success.
      return grpc::Status::OK;
    }
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    auto results = dram_alloc_->Evict(keys);
    for (const auto& r : results) {
      auto* entry = response->add_evicted();
      entry->set_key(r.key);
      entry->set_bytes_freed(r.bytes_freed);
    }
    return grpc::Status::OK;
  }

  // -------- Batch variants --------

  grpc::Status BatchAllocateSlots(grpc::ServerContext* /*ctx*/,
                                  const ::umbp::BatchAllocateSlotsRequest* request,
                                  ::umbp::BatchAllocateSlotsResponse* response) override {
    if (dram_alloc_ == nullptr) {
      for (int i = 0; i < request->entries_size(); ++i) {
        auto* out = response->add_entries();
        out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
      }
      return grpc::Status::OK;
    }

    std::vector<PeerDramAllocator::AllocateRequest> entries;
    entries.reserve(request->entries_size());
    for (const auto& entry : request->entries()) {
      PeerDramAllocator::AllocateRequest alloc_entry;
      alloc_entry.key = entry.key();
      alloc_entry.size = entry.size();
      alloc_entry.tier = FromProtoTier(entry.tier());
      entries.push_back(std::move(alloc_entry));
    }

    auto results = dram_alloc_->BatchAllocate(entries);
    for (const auto& result : results) {
      auto* out = response->add_entries();
      switch (result.outcome) {
        case PeerDramAllocator::Outcome::kSuccessAlreadyExists:
          out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS);
          continue;
        case PeerDramAllocator::Outcome::kFailed:
          out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
          continue;
        case PeerDramAllocator::Outcome::kFailedNoSpace:
          out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE);
          continue;
        case PeerDramAllocator::Outcome::kSuccessAllocated:
          break;
      }
      const auto& pending = *result.slot;
      out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
      out->set_slot_id(pending.slot_id);
      FillPagesAndDescs(out, pending.pages, dram_alloc_->PageSize(), result.descs);
      out->set_pending_ttl_ms(dram_alloc_->PendingTtlMs());
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchCommitSlots(grpc::ServerContext* /*ctx*/,
                                const ::umbp::BatchCommitSlotsRequest* request,
                                ::umbp::BatchCommitSlotsResponse* response) override {
    if (dram_alloc_ == nullptr) {
      for (int i = 0; i < request->entries_size(); ++i) response->add_success(false);
      return grpc::Status::OK;
    }

    std::vector<PeerDramAllocator::CommitRequest> entries;
    entries.reserve(request->entries_size());
    for (const auto& entry : request->entries()) {
      PeerDramAllocator::CommitRequest commit_entry;
      commit_entry.slot_id = entry.slot_id();
      commit_entry.key = entry.key();
      entries.push_back(std::move(commit_entry));
    }

    auto results = dram_alloc_->BatchCommit(entries);
    uint64_t total_committed = 0;
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& result = results[i];
      response->add_success(result.success);
      if (result.success) {
        total_committed += result.bytes_committed;
        EnqueueSsdCopy(entries[i].key, result.bytes_committed);
      }
    }
    if (total_committed > 0) RecordInboundPut(total_committed, "remote");
    return grpc::Status::OK;
  }

  grpc::Status BatchAbortSlots(grpc::ServerContext* /*ctx*/,
                               const ::umbp::BatchAbortSlotsRequest* request,
                               ::umbp::BatchAbortSlotsResponse* response) override {
    if (dram_alloc_ == nullptr) {
      for (int i = 0; i < request->slot_ids_size(); ++i) response->add_success(true);
      return grpc::Status::OK;
    }

    std::vector<uint64_t> slot_ids(request->slot_ids().begin(), request->slot_ids().end());
    auto results = dram_alloc_->BatchAbort(slot_ids);
    for (bool ok : results) {
      response->add_success(ok);
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchResolveKeys(grpc::ServerContext* /*ctx*/,
                                const ::umbp::BatchResolveKeysRequest* request,
                                ::umbp::BatchResolveKeysResponse* response) override {
    if (dram_alloc_ == nullptr) {
      std::vector<ResolvedKeyEntry> misses(request->keys_size());
      EncodeBatchResolveResponse(misses, /*page_size=*/0, /*descs=*/{}, response);
      return grpc::Status::OK;
    }
    // The client can suppress descs entirely via omit_descs when it already
    // hydrated them from GetPeerInfo — skip BuildBufferDescsLocked for the
    // whole batch in that case rather than computing and discarding it.
    const bool omit_descs = request->omit_descs();
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    auto resolved = dram_alloc_->BatchResolve(keys, /*include_descs=*/!omit_descs);

    // Project to the encoder's host shape and deduplicate the buffer
    // descriptors once for the whole batch (their bytes are identical across
    // keys for a given buffer_index).
    std::vector<ResolvedKeyEntry> entries;
    entries.reserve(resolved.size());
    std::vector<BufferMemoryDescBytes> batch_descs;
    std::vector<bool> desc_seen;  // indexed by buffer_index
    uint64_t total_bytes = 0;
    for (auto& r : resolved) {
      ResolvedKeyEntry e;
      e.found = r.found;
      if (r.found) {
        e.tier = r.tier;
        e.size = r.size;
        e.pages = std::move(r.pages);
        total_bytes += r.size;
        if (!omit_descs) {
          for (const auto& d : r.descs) {
            if (d.buffer_index >= desc_seen.size()) desc_seen.resize(d.buffer_index + 1, false);
            if (desc_seen[d.buffer_index]) continue;
            desc_seen[d.buffer_index] = true;
            batch_descs.push_back(d);
          }
        }
      }
      entries.push_back(std::move(e));
    }
    EncodeBatchResolveResponse(entries, dram_alloc_->PageSize(), batch_descs, response);
    RecordInboundGet(total_bytes, "remote");
    return grpc::Status::OK;
  }

 private:
  bool SsdRpcAvailable() const {
    return peer_ssd_ != nullptr && ssd_staging_base_ != nullptr && ssd_staging_size_ > 0;
  }

  // Best-effort enqueue of an async SSD copy after a successful DRAM commit.
  // No-op when SSD is disabled; never blocks (a full/stopped queue drops).
  void EnqueueSsdCopy(const std::string& key, uint64_t bytes) {
    if (copy_pipeline_ == nullptr) return;
    copy_pipeline_->Enqueue(SsdCopyTask{key, TierType::DRAM, static_cast<size_t>(bytes)});
  }

  void RecordInboundPut(uint64_t bytes, const char* traffic) {
    if (master_client_ == nullptr || bytes == 0) return;
    MasterClient::Labels labels = {{"traffic", std::string(traffic)}};
    master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL,
                               MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL_HELP, labels,
                               static_cast<double>(bytes));
  }

  void RecordInboundGet(uint64_t bytes, const char* traffic) {
    if (master_client_ == nullptr || bytes == 0) return;
    MasterClient::Labels labels = {{"traffic", std::string(traffic)}};
    master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                               MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP, labels,
                               static_cast<double>(bytes));
  }

  void* ssd_staging_base_;
  size_t ssd_staging_size_;
  const std::vector<uint8_t>& ssd_staging_mem_desc_bytes_;
  PeerSsdManager* peer_ssd_;
  PeerDramAllocator* dram_alloc_;
  SsdCopyPipeline* copy_pipeline_;
  MasterClient* master_client_;
  const std::vector<uint8_t>& engine_desc_bytes_;
  StagingMetrics& metrics_;

  const std::chrono::milliseconds lease_timeout_;
  const int num_read_slots_;
  const uint64_t read_region_base_;
  const size_t read_slot_size_;

  std::mutex read_slots_mutex_;
  std::vector<StagingSlot> read_slots_;
  std::atomic<uint64_t> next_lease_id_{1};

 public:
  // SSD read staging slots currently busy (Preparing or Leased).  Sampled once
  // per metrics flush by PoolClient's provider; a brief lock + small scan
  // (num_read_slots, default 16) — not a busy loop.
  size_t ReadSlotsInUse() {
    std::lock_guard<std::mutex> lock(read_slots_mutex_);
    size_t in_use = 0;
    for (const auto& slot : read_slots_) {
      if (slot.state != SlotState::kFree) ++in_use;
    }
    return in_use;
  }
};

PeerServiceServer::PeerServiceServer(PeerDramAllocator* dram_alloc, PeerSsdManager* peer_ssd,
                                     void* ssd_staging_base, size_t ssd_staging_size,
                                     std::vector<uint8_t> ssd_staging_mem_desc_bytes,
                                     int num_read_slots, std::chrono::milliseconds lease_timeout,
                                     std::vector<uint8_t> engine_desc_bytes,
                                     MasterClient* master_client, SsdCopyPipeline* copy_pipeline)
    : ssd_staging_base_(ssd_staging_base),
      ssd_staging_size_(ssd_staging_size),
      peer_ssd_(peer_ssd),
      dram_alloc_(dram_alloc),
      master_client_(master_client),
      copy_pipeline_(copy_pipeline),
      ssd_staging_mem_desc_bytes_(std::move(ssd_staging_mem_desc_bytes)),
      engine_desc_bytes_(std::move(engine_desc_bytes)) {
  service_ = std::make_unique<UMBPPeerServiceImpl>(
      ssd_staging_base_, ssd_staging_size_, ssd_staging_mem_desc_bytes_, peer_ssd_, dram_alloc_,
      master_client_, metrics_, num_read_slots, lease_timeout, engine_desc_bytes_, copy_pipeline_);
}

PeerServiceServer::~PeerServiceServer() { Stop(); }

size_t PeerServiceServer::SnapshotReadSlotsInUse() const {
  return service_ ? service_->ReadSlotsInUse() : 0;
}

bool PeerServiceServer::Start(uint16_t port) {
  std::string address = "0.0.0.0:" + std::to_string(port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();

  if (!server_) {
    MORI_UMBP_ERROR("[PeerService] Failed to start on {} (port may be in use)", address);
    return false;
  }
  MORI_UMBP_INFO("[PeerService] Listening on {}", address);
  return true;
}

void PeerServiceServer::Stop() {
  if (server_) {
    const auto deadline = std::chrono::system_clock::now() + GrpcShutdownDeadline();
    MORI_UMBP_INFO("[PeerService] Shutting down");
    server_->Shutdown(deadline);
    // Block until every in-flight handler has returned (Shutdown's deadline
    // force-cancels any that overrun).  This guarantees no RPC handler is still
    // touching borrowed state (dram_alloc_ / copy_pipeline_) after Stop()
    // returns, so PoolClient can safely tear those down next.
    server_->Wait();
    server_.reset();
  }
}

}  // namespace mori::umbp
