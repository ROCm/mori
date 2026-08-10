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
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/batch_resolve_codec.h"
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

// ---------------------------------------------------------------------------
//  Slot-id tier tagging
//
//  Commit / Abort carry only a slot_id — the proto has no tier field on them,
//  and umbp_peer.proto documents the id as "opaque; echoed back by Commit /
//  Abort".  Every backend numbers its own slots from 1 independently, so a bare
//  id is ambiguous the moment a second medium is live.
//
//  So the peer service tags the tier into the high byte on the way out and
//  strips it on the way back in.  Dispatch then works with NO proto change and
//  NO client change: the client only ever echoes the value it was given.  The
//  tag is peer-local — PoolClient's local fast path talks to a backend directly
//  and never sees a tagged id.
//
//  next_slot_id_ is a per-backend counter starting at 1, so the low 56 bits
//  cannot reach the tag in any realistic process lifetime.
// ---------------------------------------------------------------------------
constexpr int kSlotTierShift = 56;
constexpr uint64_t kSlotLocalMask = (1ULL << kSlotTierShift) - 1;

uint64_t TagSlotId(TierType tier, uint64_t local_id) {
  return (static_cast<uint64_t>(tier) << kSlotTierShift) | (local_id & kSlotLocalMask);
}
TierType TierFromSlotId(uint64_t tagged) { return static_cast<TierType>(tagged >> kSlotTierShift); }
uint64_t LocalSlotId(uint64_t tagged) { return tagged & kSlotLocalMask; }

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
  UMBPPeerServiceImpl(BackendRegistry* registry, MasterClient* master_client,
                      const std::vector<uint8_t>& engine_desc_bytes)
      : registry_(registry),
        // All() allocates, and Resolve is the hot read path — so the medium
        // list is snapshotted once here rather than rebuilt per RPC.  This is
        // why the registry MUST be fully populated before the peer service is
        // constructed (PoolClient::Init registers every backend first, then
        // starts this server).
        media_(registry == nullptr ? std::vector<MediumBackend*>{} : registry->All()),
        engine_desc_bytes_(engine_desc_bytes),
        master_client_(master_client) {}

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    if (!engine_desc_bytes_.empty()) {
      response->set_engine_desc(std::string(engine_desc_bytes_.begin(), engine_desc_bytes_.end()));
    }
    // Ship every configured buffer's desc so first-contact writers can hydrate
    // without a follow-up Allocate / Resolve.
    //
    // The bootstrap wire carries ONE medium: dram_memory_descs is a flat
    // buffer_index space and dram_page_size is a single field, while each
    // backend numbers its buffers from 0 with its own page size (design doc §1
    // item 4 — "no tier dimension").  So descs come from the first backend that
    // has any, and a second buffer-owning backend is reported rather than
    // silently merged into a colliding index space.  A
    // tier dimension here is the prerequisite for a second live medium, which
    // the design doc defers to Phase 6 (§5 Phase 2 option (a)).
    MediumBackend* published = nullptr;
    for (auto* backend : Media()) {
      auto descs = backend->AllBufferDescs();
      if (descs.empty()) continue;
      if (published != nullptr) {
        MORI_UMBP_ERROR(
            "[PeerService] GetPeerInfo: backend tier={} has buffers but the bootstrap wire "
            "carries only one medium (published tier={}); its buffers are NOT advertised",
            static_cast<int>(backend->Tier()), static_cast<int>(published->Tier()));
        continue;
      }
      for (const auto& d : descs) {
        auto* out = response->add_dram_memory_descs();
        out->set_buffer_index(d.buffer_index);
        out->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
      }
      response->set_dram_page_size(backend->PageSize());
      published = backend;
    }
    return grpc::Status::OK;
  }

  // ============================================================
  //  DRAM/HBM allocator + key map (master-as-advisor design)
  // ============================================================

  // The single-key RPCs below are served by one-element batches (design doc §3)
  // — there is no separate single-key path on MediumBackend to keep in sync.

  grpc::Status AllocateSlot(grpc::ServerContext* /*ctx*/,
                            const ::umbp::AllocateSlotRequest* request,
                            ::umbp::AllocateSlotResponse* response) override {
    // A tier with no live backend on this peer is a normal condition (a
    // DRAM-only node), not an error: FAILED, not a crash.
    auto* backend = Backend(FromProtoTier(request->tier()));
    if (backend == nullptr) {
      response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
      return grpc::Status::OK;
    }
    AllocateRequest entry;
    entry.key = request->key();
    entry.size = request->size();
    auto results = backend->BatchAllocate({entry});
    const auto& result = results.front();
    switch (result.outcome) {
      case AllocateOutcome::kSuccessAlreadyExists:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS);
        return grpc::Status::OK;
      case AllocateOutcome::kFailed:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
        return grpc::Status::OK;
      case AllocateOutcome::kFailedNoSpace:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE);
        return grpc::Status::OK;
      case AllocateOutcome::kSuccessAllocated:
        break;
    }
    response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
    response->set_slot_id(TagSlotId(backend->Tier(), result.slot_id));
    FillPagesAndDescs(response, result.pages, result.page_size, result.descs);
    response->set_pending_ttl_ms(result.pending_ttl_ms);
    return grpc::Status::OK;
  }

  grpc::Status CommitSlot(grpc::ServerContext* /*ctx*/, const ::umbp::CommitSlotRequest* request,
                          ::umbp::CommitSlotResponse* response) override {
    auto* backend = Backend(TierFromSlotId(request->slot_id()));
    if (backend == nullptr) {
      response->set_success(false);
      return grpc::Status::OK;
    }
    CommitRequest entry;
    entry.slot_id = LocalSlotId(request->slot_id());
    entry.key = request->key();
    auto results = backend->BatchCommit({entry});
    const auto& result = results.front();
    response->set_success(result.success);
    if (result.success) {
      RecordInboundPut(result.bytes_committed, "remote");
    }
    return grpc::Status::OK;
  }

  grpc::Status AbortSlot(grpc::ServerContext* /*ctx*/, const ::umbp::AbortSlotRequest* request,
                         ::umbp::AbortSlotResponse* response) override {
    auto* backend = Backend(TierFromSlotId(request->slot_id()));
    if (backend == nullptr) {
      response->set_success(true);  // idempotent: nothing to drop
      return grpc::Status::OK;
    }
    auto results = backend->BatchAbort({LocalSlotId(request->slot_id())});
    response->set_success(results.front());
    return grpc::Status::OK;
  }

  grpc::Status ResolveKey(grpc::ServerContext* /*ctx*/, const ::umbp::ResolveKeyRequest* request,
                          ::umbp::ResolveKeyResponse* response) override {
    // Resolve carries no tier: the key may sit in any medium, and may be
    // mirrored across several.  Walk this peer's media and take the first hit.
    // With every medium equivalent (Phase 4) the walk order is deterministic
    // but arbitrary; a real preference would come from the backend advertising
    // one, not from a tier list here.
    for (auto* backend : Media()) {
      auto results = backend->BatchResolve({request->key()}, /*include_descs=*/true);
      const auto& r = results.front();
      if (!r.found) continue;
      response->set_found(true);
      FillPagesAndDescs(response, r.pages, r.page_size, r.descs);
      response->set_size(r.size);
      RecordInboundGet(r.size, "remote");
      return grpc::Status::OK;
    }
    response->set_found(false);
    return grpc::Status::OK;
  }

  grpc::Status EvictKey(grpc::ServerContext* /*ctx*/, const ::umbp::EvictKeyRequest* request,
                        ::umbp::EvictKeyResponse* response) override {
    // Eviction carries no tier either.  A key mirrored across media must be
    // dropped from ALL of them, so every backend is asked and the freed bytes
    // are summed per key — master sizes its next eviction round off this total.
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    if (keys.empty()) return grpc::Status::OK;
    std::vector<uint64_t> freed(keys.size(), 0);
    for (auto* backend : Media()) {
      auto results = backend->Evict(keys);
      for (size_t i = 0; i < results.size() && i < freed.size(); ++i) {
        freed[i] += results[i].bytes_freed;
      }
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      auto* entry = response->add_evicted();
      entry->set_key(keys[i]);
      entry->set_bytes_freed(freed[i]);
    }
    return grpc::Status::OK;
  }

  // -------- Batch variants --------

  grpc::Status BatchAllocateSlots(grpc::ServerContext* /*ctx*/,
                                  const ::umbp::BatchAllocateSlotsRequest* request,
                                  ::umbp::BatchAllocateSlotsResponse* response) override {
    const int n = request->entries_size();
    for (int i = 0; i < n; ++i) {
      response->add_entries()->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
    }

    // Group by requested tier so each backend still sees ONE batched call —
    // splitting a mixed batch into per-entry calls would give up the batching
    // the RPC exists for.  Entries whose tier has no live backend keep the
    // pre-filled FAILED, which is what preserves per-entry result ordering.
    std::map<TierType, std::vector<int>> by_tier;
    for (int i = 0; i < n; ++i) {
      by_tier[FromProtoTier(request->entries(i).tier())].push_back(i);
    }

    for (const auto& [tier, indices] : by_tier) {
      auto* backend = Backend(tier);
      if (backend == nullptr) continue;
      std::vector<AllocateRequest> entries;
      entries.reserve(indices.size());
      for (int i : indices) {
        AllocateRequest alloc_entry;
        alloc_entry.key = request->entries(i).key();
        alloc_entry.size = request->entries(i).size();
        entries.push_back(std::move(alloc_entry));
      }

      auto results = backend->BatchAllocate(entries);
      for (size_t k = 0; k < indices.size() && k < results.size(); ++k) {
        const auto& result = results[k];
        auto* out = response->mutable_entries(indices[k]);
        switch (result.outcome) {
          case AllocateOutcome::kSuccessAlreadyExists:
            out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS);
            continue;
          case AllocateOutcome::kFailed:
            out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
            continue;
          case AllocateOutcome::kFailedNoSpace:
            out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE);
            continue;
          case AllocateOutcome::kSuccessAllocated:
            break;
        }
        out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
        out->set_slot_id(TagSlotId(tier, result.slot_id));
        FillPagesAndDescs(out, result.pages, result.page_size, result.descs);
        out->set_pending_ttl_ms(result.pending_ttl_ms);
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchCommitSlots(grpc::ServerContext* /*ctx*/,
                                const ::umbp::BatchCommitSlotsRequest* request,
                                ::umbp::BatchCommitSlotsResponse* response) override {
    const int n = request->entries_size();
    for (int i = 0; i < n; ++i) response->add_success(false);

    // The tier rides in the slot_id the peer handed out at Allocate (see
    // TagSlotId) — a commit for a tier that has since gone away keeps the
    // pre-filled false, which is exactly "slot unknown".
    std::map<TierType, std::vector<int>> by_tier;
    for (int i = 0; i < n; ++i) {
      by_tier[TierFromSlotId(request->entries(i).slot_id())].push_back(i);
    }

    uint64_t total_committed = 0;
    for (const auto& [tier, indices] : by_tier) {
      auto* backend = Backend(tier);
      if (backend == nullptr) continue;
      std::vector<CommitRequest> entries;
      entries.reserve(indices.size());
      for (int i : indices) {
        CommitRequest commit_entry;
        commit_entry.slot_id = LocalSlotId(request->entries(i).slot_id());
        commit_entry.key = request->entries(i).key();
        entries.push_back(std::move(commit_entry));
      }

      auto results = backend->BatchCommit(entries);
      for (size_t k = 0; k < indices.size() && k < results.size(); ++k) {
        response->set_success(indices[k], results[k].success);
        if (results[k].success) total_committed += results[k].bytes_committed;
      }
    }
    if (total_committed > 0) RecordInboundPut(total_committed, "remote");
    return grpc::Status::OK;
  }

  grpc::Status BatchAbortSlots(grpc::ServerContext* /*ctx*/,
                               const ::umbp::BatchAbortSlotsRequest* request,
                               ::umbp::BatchAbortSlotsResponse* response) override {
    const int n = request->slot_ids_size();
    // Abort is idempotent: an unknown slot (including one whose tier has no
    // live backend) reports true — there is nothing left to drop.
    for (int i = 0; i < n; ++i) response->add_success(true);

    std::map<TierType, std::vector<int>> by_tier;
    for (int i = 0; i < n; ++i) by_tier[TierFromSlotId(request->slot_ids(i))].push_back(i);

    for (const auto& [tier, indices] : by_tier) {
      auto* backend = Backend(tier);
      if (backend == nullptr) continue;
      std::vector<uint64_t> slot_ids;
      slot_ids.reserve(indices.size());
      for (int i : indices) slot_ids.push_back(LocalSlotId(request->slot_ids(i)));

      auto results = backend->BatchAbort(slot_ids);
      for (size_t k = 0; k < indices.size() && k < results.size(); ++k) {
        response->set_success(indices[k], results[k]);
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchResolveKeys(grpc::ServerContext* /*ctx*/,
                                const ::umbp::BatchResolveKeysRequest* request,
                                ::umbp::BatchResolveKeysResponse* response) override {
    // The client can suppress descs entirely via omit_descs when it already
    // hydrated them from GetPeerInfo — skip building them for the whole batch
    // in that case rather than computing and discarding them.
    const bool omit_descs = request->omit_descs();
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());

    // One response carries ONE medium: page_size is a single field and descs
    // share a flat buffer_index space (same wire limitation as GetPeerInfo).
    // So the first backend in read-rank order that has any hit serves the whole
    // response.  A key held only by a lower-ranked medium reports found=false,
    // which the client handles as a miss (it excludes the node and retries) —
    // never as a corrupt read, the same safe direction the omit_descs contract
    // already relies on.
    std::vector<ResolvedEntry> resolved;
    MediumBackend* serving = nullptr;
    for (auto* backend : Media()) {
      auto candidate = backend->BatchResolve(keys, /*include_descs=*/!omit_descs);
      const bool any_hit = std::any_of(candidate.begin(), candidate.end(),
                                       [](const ResolvedEntry& e) { return e.found; });
      if (!any_hit) continue;
      resolved = std::move(candidate);
      serving = backend;
      break;
    }
    if (serving == nullptr) {
      std::vector<ResolvedKeyEntry> misses(request->keys_size());
      EncodeBatchResolveResponse(misses, /*page_size=*/0, /*descs=*/{}, response);
      return grpc::Status::OK;
    }

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
        e.tier = serving->Tier();
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
    EncodeBatchResolveResponse(entries, serving->PageSize(), batch_descs, response);
    RecordInboundGet(total_bytes, "remote");
    return grpc::Status::OK;
  }

 private:
  // Null when the registry is absent or has no backend for `tier` — both are
  // normal (a DRAM-only node), and every caller answers with the RPC's
  // not-here response rather than an error.
  MediumBackend* Backend(TierType tier) const {
    return registry_ == nullptr ? nullptr : registry_->Get(tier);
  }

  // Every medium on this peer, snapshotted at construction (see ctor).  The
  // order is deterministic (ascending TierType) but carries no preference —
  // every medium is equivalent (backend-agnostic refactor Phase 4).
  const std::vector<MediumBackend*>& Media() const { return media_; }

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

  BackendRegistry* registry_;
  const std::vector<MediumBackend*> media_;
  const std::vector<uint8_t>& engine_desc_bytes_;
  MasterClient* master_client_;
};

PeerServiceServer::PeerServiceServer(BackendRegistry* registry,
                                     std::vector<uint8_t> engine_desc_bytes,
                                     MasterClient* master_client)
    : registry_(registry),
      master_client_(master_client),
      engine_desc_bytes_(std::move(engine_desc_bytes)) {
  service_ = std::make_unique<UMBPPeerServiceImpl>(registry_, master_client_, engine_desc_bytes_);
}

PeerServiceServer::~PeerServiceServer() { Stop(); }

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
    // touching borrowed state (registry_ and the backends in it) after Stop()
    // returns, so PoolClient can safely tear it down next.
    server_->Wait();
    server_.reset();
  }
}

}  // namespace mori::umbp
