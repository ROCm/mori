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
#include <limits>
#include <msgpack.hpp>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/common/grpc_limits.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/metrics/metric_sink.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/batch_resolve_codec.h"
#include "umbp/distributed/pool/fabric_read_push.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/tier_proto.h"
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

// ---------------------------------------------------------------------------
//  Slot-id backend tagging
//
//  Commit / Abort carry only a slot_id — the proto has no tier field on them,
//  and umbp_peer.proto documents the id as "opaque; echoed back by Commit /
//  Abort".  Every backend numbers its own slots from 1 independently, so a bare
//  id is ambiguous the moment a second medium is live.
//
//  So the peer service tags the backend instance id into the high byte on the
//  way out and strips it on the way back in.  TierType is not sufficient: a
//  topology may contain ssd_a and ssd_b.  The tag is peer-local — PoolClient's
//  local fast path talks to a backend directly and never sees a tagged id.
//
//  next_slot_id_ is a per-backend counter starting at 1, so the low 56 bits
//  cannot reach the tag in any realistic process lifetime.
// ---------------------------------------------------------------------------
constexpr int kSlotBackendShift = 56;
constexpr uint64_t kSlotLocalMask = (1ULL << kSlotBackendShift) - 1;

uint64_t TagSlotId(uint32_t backend_id, uint64_t local_id) {
  return (static_cast<uint64_t>(backend_id) << kSlotBackendShift) | (local_id & kSlotLocalMask);
}
uint32_t BackendIdFromSlotId(uint64_t tagged) {
  return static_cast<uint32_t>(tagged >> kSlotBackendShift);
}
uint64_t LocalSlotId(uint64_t tagged) { return tagged & kSlotLocalMask; }

// Drop a (pages, page_size, descs) tuple into a slot-shaped response
// that exposes those fields directly.  Templated so the same body
// covers AllocateSlotResponse and ResolveKeyResponse.
//
// `backend_id` is stamped here rather than inside the backend: a backend
// numbers its buffers from 0 and knows nothing of its siblings, so the owning
// backend is a fact only this boundary has.  See BufferMemoryDesc in
// umbp.proto.
template <typename Response>
void FillPagesAndDescs(Response* resp, const std::vector<PageLocation>& pages, uint64_t page_size,
                       const std::vector<BufferMemoryDescBytes>& descs, uint32_t backend_id) {
  for (const auto& p : pages) {
    auto* pl = resp->add_pages();
    pl->set_buffer_index(p.buffer_index);
    pl->set_page_index(p.page_index);
  }
  resp->set_page_size(page_size);
  resp->set_backend_id(backend_id);
  for (const auto& d : descs) {
    auto* desc = resp->add_descs();
    desc->set_buffer_index(d.buffer_index);
    desc->set_backend_id(backend_id);
    desc->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
  }
}

}  // namespace

class PeerServiceServer::UMBPPeerServiceImpl final : public ::umbp::UMBPPeer::Service {
 public:
  UMBPPeerServiceImpl(PeerPool* pool, MetricSink* metric_sink,
                      const std::vector<uint8_t>& engine_desc_bytes)
      : pool_(pool),
        registry_(pool == nullptr ? nullptr : pool->Backends()),
        // All() allocates, and Resolve is the hot read path — so the medium
        // list is snapshotted once here rather than rebuilt per RPC.  This is
        // why the registry MUST be fully populated before the peer service is
        // constructed (PoolClient::Init registers every backend first, then
        // starts this server).
        media_(registry_ == nullptr ? std::vector<MediumBackend*>{} : registry_->All()),
        engine_desc_bytes_(engine_desc_bytes),
        metric_sink_(metric_sink) {}

  grpc::Status GetPeerInfo(grpc::ServerContext* /*context*/,
                           const ::umbp::GetPeerInfoRequest* /*request*/,
                           ::umbp::GetPeerInfoResponse* response) override {
    response->set_owner_incarnation(owner_incarnation_);
    response->set_pool_generation(pool_ == nullptr ? 0 : pool_->Generation());
    response->set_completion_pins(true);
    response->set_read_push(pool_ && pool_->SupportsReadPush());
    if (!engine_desc_bytes_.empty()) {
      response->set_engine_desc(std::string(engine_desc_bytes_.begin(), engine_desc_bytes_.end()));
    }
    // Ship EVERY backend's buffer descs so first-contact writers can hydrate
    // without a follow-up Allocate / Resolve.
    //
    // This used to publish one medium and log an error about the rest, because
    // the wire carried a flat buffer_index space with no way to say which
    // backend an index belonged to.  That was not merely incomplete: the reader
    // caches descriptors by index and asks the peer to omit them once it
    // believes it has them (BatchResolveKeysRequest.omit_descs), so an
    // unadvertised medium's pages were resolved against the advertised
    // medium's memory — a hit, with bytes from the wrong pool.  Every backend
    // publishes exactly one buffer today, so index 0 collided on every
    // mixed-media node and the corruption was deterministic.
    //
    // BufferMemoryDesc.backend_id closes it.  buffer_index stays backend-local
    // and no backend changed; the id is stamped here, at the boundary that
    // knows it.
    for (auto* backend : Media()) {
      const uint32_t backend_id = registry_->BackendId(backend);
      for (const auto& d : backend->AllBufferDescs()) {
        auto* out = response->add_buffer_descs();
        out->set_backend_id(backend_id);
        out->set_buffer_index(d.buffer_index);
        out->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
      }
    }
    if (pool_ != nullptr) {
      for (const auto& d : pool_->FabricReadStagingDescriptors()) {
        auto* out = response->add_buffer_descs();
        out->set_backend_id(d.backend_id);
        out->set_buffer_index(d.buffer_index);
        out->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
      }
    }
    // Uniform across backends; BackendRegistry::Register refuses a backend that
    // disagrees, so there is one page size to report.
    if (registry_ != nullptr) response->set_page_size(registry_->PageSize());
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
    if (pool_ == nullptr) {
      response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
      return grpc::Status::OK;
    }
    PoolPlacementRequest entry;
    if (!request->operation_id().empty() &&
        request->expected_owner_incarnation() != owner_incarnation_)
      return StaleOwner();
    entry.key = request->key();
    entry.size = request->size();
    entry.tier = TierFromProto(request->tier());
    entry.backend_name = request->backend_name();
    entry.logical_tier = request->logical_tier();
    entry.operation_id = request->operation_id();
    if (!entry.operation_id.empty())
      entry.expected_pool_generation = request->expected_pool_generation();
    if (request->has_preferred_numa_node())
      entry.preferred_numa_node = request->preferred_numa_node();
    auto pool_result = pool_->BatchAllocate({entry}).front();
    auto* backend = Backend(pool_result.backend_id);
    const auto& result = pool_result.allocation;
    switch (result.outcome) {
      case AllocateOutcome::kSuccessAlreadyExists:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS);
        return grpc::Status::OK;
      case AllocateOutcome::kFailed:
        response->set_outcome(pool_result.in_flight ? ::umbp::ALLOCATE_SLOT_OUTCOME_IN_FLIGHT
                                                    : ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
        return grpc::Status::OK;
      case AllocateOutcome::kFailedNoSpace:
        response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE);
        return grpc::Status::OK;
      case AllocateOutcome::kSuccessAllocated:
        break;
    }
    if (backend == nullptr) {
      response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
      return grpc::Status::OK;
    }
    response->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
    response->set_slot_id(TagSlotId(pool_result.backend_id, result.slot_id));
    FillPagesAndDescs(response, result.pages, result.page_size, result.descs,
                      pool_result.backend_id);
    response->set_pending_ttl_ms(result.pending_ttl_ms);
    return grpc::Status::OK;
  }

  grpc::Status CommitSlot(grpc::ServerContext* /*ctx*/, const ::umbp::CommitSlotRequest* request,
                          ::umbp::CommitSlotResponse* response) override {
    if (pool_ == nullptr) {
      response->set_success(false);
      return grpc::Status::OK;
    }
    PoolCommitRequest entry;
    if (!request->operation_id().empty() &&
        request->expected_owner_incarnation() != owner_incarnation_)
      return StaleOwner();
    entry.slot.backend_id = BackendIdFromSlotId(request->slot_id());
    entry.slot.local_slot_id = LocalSlotId(request->slot_id());
    entry.key = request->key();
    entry.operation_id = request->operation_id();
    entry.transfer_drained = request->transfer_drained();
    const auto result = pool_->BatchCommit({entry}).front().commit;
    response->set_success(result.success);
    if (result.success) {
      RecordInboundPut(result.bytes_committed, "remote");
    }
    return grpc::Status::OK;
  }

  grpc::Status AbortSlot(grpc::ServerContext* /*ctx*/, const ::umbp::AbortSlotRequest* request,
                         ::umbp::AbortSlotResponse* response) override {
    if (pool_ == nullptr) {
      response->set_success(true);  // idempotent: nothing to drop
      return grpc::Status::OK;
    }
    PoolSlotRef slot{BackendIdFromSlotId(request->slot_id()), LocalSlotId(request->slot_id())};
    auto results = pool_->BatchAbort({slot});
    response->set_success(results.front());
    return grpc::Status::OK;
  }

  grpc::Status ResolveKey(grpc::ServerContext* /*ctx*/, const ::umbp::ResolveKeyRequest* request,
                          ::umbp::ResolveKeyResponse* response) override {
    if (pool_ == nullptr) return grpc::Status::OK;
    auto result = pool_->BatchResolve({request->key()}, /*include_descs=*/true).front();
    if (!result.resolved.found) return grpc::Status::OK;
    response->set_found(true);
    FillPagesAndDescs(response, result.resolved.pages, result.resolved.page_size,
                      result.resolved.descs, result.backend_id);
    response->set_size(result.resolved.size);
    RecordInboundGet(result.resolved.size, "remote");
    return grpc::Status::OK;
  }

  grpc::Status EvictKey(grpc::ServerContext* /*ctx*/, const ::umbp::EvictKeyRequest* request,
                        ::umbp::EvictKeyResponse* response) override {
    // Free each key's copy on the tier the master named (an older master names
    // none: the medium holding it).  bytes_freed is informational; a queued
    // demotion reports 0.
    const int n = request->keys_size();
    if (n == 0) return grpc::Status::OK;
    const bool scoped = request->tiers_size() == n;
    std::vector<PoolEvictRequest> requests(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      requests[i].key = request->keys(i);
      if (scoped) requests[i].tier = static_cast<TierType>(request->tiers(i));
    }
    auto evicted = pool_ == nullptr ? std::vector<PoolEvictResult>{}
                                    : pool_->Evict(requests, PoolEvictMode::kReclaim);
    for (size_t i = 0; i < requests.size(); ++i) {
      auto* entry = response->add_evicted();
      entry->set_key(requests[i].key);
      entry->set_bytes_freed(i < evicted.size() ? evicted[i].bytes_freed : 0);
    }
    return grpc::Status::OK;
  }

  // -------- Batch variants --------

  grpc::Status BatchAllocateSlots(grpc::ServerContext* /*ctx*/,
                                  const ::umbp::BatchAllocateSlotsRequest* request,
                                  ::umbp::BatchAllocateSlotsResponse* response) override {
    const int n = request->entries_size();
    response->set_owner_incarnation(owner_incarnation_);
    response->set_pool_generation(pool_ == nullptr ? 0 : pool_->Generation());
    for (const auto& entry : request->entries()) {
      if (!entry.operation_id().empty() &&
          request->expected_owner_incarnation() != owner_incarnation_)
        return StaleOwner();
    }
    for (int i = 0; i < n; ++i) {
      response->add_entries()->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
    }
    if (pool_ == nullptr) return grpc::Status::OK;

    std::vector<PoolPlacementRequest> entries;
    entries.reserve(n);
    for (int i = 0; i < n; ++i) {
      PoolPlacementRequest entry;
      entry.key = request->entries(i).key();
      entry.size = request->entries(i).size();
      entry.tier = TierFromProto(request->entries(i).tier());
      entry.backend_name = request->entries(i).backend_name();
      entry.logical_tier = request->entries(i).logical_tier();
      entry.operation_id = request->entries(i).operation_id();
      if (!entry.operation_id.empty())
        entry.expected_pool_generation = request->expected_pool_generation();
      if (request->entries(i).has_preferred_numa_node()) {
        entry.preferred_numa_node = request->entries(i).preferred_numa_node();
      }
      entries.push_back(std::move(entry));
    }
    auto results = pool_->BatchAllocate(entries);
    for (size_t i = 0; i < results.size() && i < static_cast<size_t>(n); ++i) {
      const auto& result = results[i].allocation;
      auto* out = response->mutable_entries(static_cast<int>(i));
      switch (result.outcome) {
        case AllocateOutcome::kSuccessAlreadyExists:
          out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS);
          continue;
        case AllocateOutcome::kFailed:
          out->set_outcome(results[i].in_flight ? ::umbp::ALLOCATE_SLOT_OUTCOME_IN_FLIGHT
                                                : ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
          continue;
        case AllocateOutcome::kFailedNoSpace:
          out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE);
          continue;
        case AllocateOutcome::kSuccessAllocated:
          break;
      }
      out->set_outcome(::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
      out->set_slot_id(TagSlotId(results[i].backend_id, result.slot_id));
      FillPagesAndDescs(out, result.pages, result.page_size, result.descs, results[i].backend_id);
      out->set_pending_ttl_ms(result.pending_ttl_ms);
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchCommitSlots(grpc::ServerContext* /*ctx*/,
                                const ::umbp::BatchCommitSlotsRequest* request,
                                ::umbp::BatchCommitSlotsResponse* response) override {
    const int n = request->entries_size();
    for (const auto& entry : request->entries()) {
      if (!entry.operation_id().empty() &&
          request->expected_owner_incarnation() != owner_incarnation_)
        return StaleOwner();
    }
    for (int i = 0; i < n; ++i) response->add_success(false);
    if (pool_ == nullptr) return grpc::Status::OK;

    std::vector<PoolCommitRequest> entries;
    entries.reserve(n);
    for (int i = 0; i < n; ++i) {
      PoolCommitRequest entry;
      entry.slot.backend_id = BackendIdFromSlotId(request->entries(i).slot_id());
      entry.slot.local_slot_id = LocalSlotId(request->entries(i).slot_id());
      entry.key = request->entries(i).key();
      entry.operation_id = request->entries(i).operation_id();
      entry.transfer_drained = request->entries(i).transfer_drained();
      entries.push_back(std::move(entry));
    }

    uint64_t total_committed = 0;
    auto results = pool_->BatchCommit(entries);
    for (size_t i = 0; i < results.size() && i < static_cast<size_t>(n); ++i) {
      response->set_success(static_cast<int>(i), results[i].commit.success);
      if (results[i].commit.success) total_committed += results[i].commit.bytes_committed;
    }
    if (total_committed > 0) RecordInboundPut(total_committed, "remote");
    return grpc::Status::OK;
  }

  grpc::Status BatchAbortSlots(grpc::ServerContext* /*ctx*/,
                               const ::umbp::BatchAbortSlotsRequest* request,
                               ::umbp::BatchAbortSlotsResponse* response) override {
    const int n = request->slot_ids_size();
    // Abort is idempotent: an unknown slot (including one whose backend is no
    // longer live) reports true — there is nothing left to drop.
    for (int i = 0; i < n; ++i) response->add_success(true);
    if (pool_ == nullptr) return grpc::Status::OK;

    std::vector<PoolSlotRef> slots;
    slots.reserve(n);
    for (int i = 0; i < n; ++i) {
      slots.push_back(PoolSlotRef{BackendIdFromSlotId(request->slot_ids(i)),
                                  LocalSlotId(request->slot_ids(i))});
    }

    auto results = pool_->BatchAbort(slots);
    for (size_t i = 0; i < results.size() && i < static_cast<size_t>(n); ++i) {
      response->set_success(static_cast<int>(i), results[i]);
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

    std::vector<ResolvedKeyEntry> entries(keys.size());
    std::vector<BufferMemoryDescBytes> batch_descs;
    std::vector<std::vector<bool>> desc_seen(BackendRegistry::kMaxBackends);
    uint64_t total_bytes = 0;
    auto resolved = pool_ == nullptr ? std::vector<PoolResolvedEntry>{}
                                     : pool_->BatchResolve(keys, /*include_descs=*/!omit_descs);
    for (size_t i = 0; i < resolved.size() && i < entries.size(); ++i) {
      auto& result = resolved[i];
      auto& r = result.resolved;
      entries[i].outcome = EffectiveResolveOutcome(r);
      if (result.backend_id < BackendRegistry::kMaxBackends) {
        entries[i].tier = result.tier;
        entries[i].backend_id = result.backend_id;
      }
      if (!r.found || result.backend_id >= BackendRegistry::kMaxBackends) continue;
      entries[i].found = true;
      entries[i].outcome = ResolveOutcome::kFound;
      entries[i].tier = result.tier;
      entries[i].backend_id = result.backend_id;
      entries[i].size = r.size;
      entries[i].pages = std::move(r.pages);
      entries[i].covered_offset = r.covered_offset;
      entries[i].covered_size = r.covered_size;
      total_bytes += r.size;

      if (omit_descs) continue;
      auto& seen = desc_seen[result.backend_id];
      for (auto& d : r.descs) {
        if (d.buffer_index >= seen.size()) seen.resize(d.buffer_index + 1, false);
        if (seen[d.buffer_index]) continue;
        seen[d.buffer_index] = true;
        d.backend_id = result.backend_id;
        batch_descs.push_back(std::move(d));
      }
    }

    // Page size is uniform across backends (BackendRegistry::Register), so a
    // batch spanning media still has one to report.
    EncodeBatchResolveResponse(entries, registry_ == nullptr ? 0 : registry_->PageSize(),
                               batch_descs, response);
    RecordInboundGet(total_bytes, "remote");
    return grpc::Status::OK;
  }

  grpc::Status BatchAcquireReadKeys(grpc::ServerContext*,
                                    const ::umbp::BatchAcquireReadKeysRequest* request,
                                    ::umbp::BatchAcquireReadKeysResponse* response) override {
    response->set_owner_incarnation(owner_incarnation_);
    response->set_pool_generation(pool_ == nullptr ? 0 : pool_->Generation());
    if (request->expected_owner_incarnation() != owner_incarnation_) return StaleOwner();
    if (pool_ == nullptr) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "no peer pool");
    const auto& resolve = request->resolve();
    std::vector<std::string> keys(resolve.keys().begin(), resolve.keys().end());
    if (resolve.extents_size() != 0 && resolve.extents_size() != static_cast<int>(keys.size()))
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "extents must be parallel to keys");
    std::vector<ObjectExtent> extents;
    for (const auto& extent : resolve.extents())
      extents.push_back({extent.offset(), extent.length()});
    if (resolve.ranges_size() != 0 && resolve.ranges_size() != static_cast<int>(keys.size()))
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "ranges must be parallel to keys");
    std::vector<std::vector<ObjectExtent>> ranges(resolve.ranges_size());
    for (int i = 0; i < resolve.ranges_size(); ++i)
      for (const auto& span : resolve.ranges(i).spans())
        ranges[i].push_back({span.offset(), span.length()});
    FabricReadPushRequest push;
    const bool rdma_push = pool_ != nullptr && pool_->ReadPushRdma();
    if (request->has_push()) {
      const auto& wire = request->push();
      if (request->require_fabric() || wire.reader_id().empty() ||
          wire.keys_size() != static_cast<int>(keys.size()))
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid push shape");
      push.signature = request->SerializeAsString();
      push.reader_id = wire.reader_id();
      push.engine_desc = wire.engine_desc();
      push.ranges.resize(keys.size());
      try {
        mori::io::EngineDesc engine;
        msgpack::unpack(wire.engine_desc().data(), wire.engine_desc().size()).get().convert(engine);
        for (int i = 0; i < wire.keys_size(); ++i) {
          for (const auto& span : wire.keys(i).ranges()) {
            FabricReadPushRange range;
            range.source_offset = span.source_offset();
            range.size = span.length();
            range.destination_offset = span.destination_offset();
            auto& ref = range.destination;
            msgpack::unpack(span.destination_desc().data(), span.destination_desc().size())
                .get()
                .convert(ref.mem);
            ref.loc = ref.mem.loc;
            ref.size = ref.mem.size;
            ref.device = ref.mem.deviceId;
            // FABRIC destinations are fabric-exported GPU memory; RDMA ones
            // only need to be registered with the reader's engine.
            const bool fabric_ok =
                ref.loc == mori::io::MemoryLocationType::GPU &&
                !std::all_of(ref.mem.fabricHandle.begin(), ref.mem.fabricHandle.end(),
                             [](char c) { return c == 0; });
            if (ref.mem.engineKey != engine.key || !range.size ||
                range.destination_offset > ref.size ||
                range.size > ref.size - range.destination_offset ||
                !(rdma_push ? ref.HasMemoryDesc() : fabric_ok))
              return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid push destination");
            push.ranges[i].push_back(std::move(range));
          }
        }
      } catch (...) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid push descriptor");
      }
    }
    auto grant = pool_->AcquireReadBatch(
        keys, request->operation_id(), request->expected_pool_generation(), !resolve.omit_descs(),
        extents.empty() ? nullptr : &extents, request->require_fabric(),
        ranges.empty() ? nullptr : &ranges, request->has_push() ? &push : nullptr);
    response->set_pool_generation(grant.generation);
    if (!grant.accepted) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "stale generation, released operation, or pin capacity exhausted");
    }
    response->set_read_pin_token(grant.token);
    if (request->has_push()) {
      response->set_push_supported(grant.push_supported);
      response->set_push_drained(grant.push_drained);
      response->set_push_submitted_bytes(grant.push_bytes);
      response->set_push_submitted_plans(grant.push_plans);
    }
    std::vector<ResolvedKeyEntry> entries(keys.size());
    std::vector<BufferMemoryDescBytes> descs;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    uint64_t bytes = 0;
    for (size_t i = 0; i < grant.entries.size() && i < entries.size(); ++i) {
      auto& source = grant.entries[i];
      auto& entry = entries[i];
      entry.outcome = EffectiveResolveOutcome(source.resolved);
      entry.found = source.resolved.found;
      entry.backend_id = source.backend_id;
      entry.tier = source.tier;
      entry.size = source.resolved.size;
      if (!request->has_push()) entry.pages = source.resolved.pages;
      entry.covered_offset = source.resolved.covered_offset;
      entry.covered_size = source.resolved.covered_size;
      entry.packed_ranges = source.resolved.packed_ranges;
      if (entry.found) bytes += entry.size;
      for (auto desc : source.resolved.descs) {
        desc.backend_id = source.backend_id;
        if (seen.emplace(desc.backend_id, desc.buffer_index).second)
          descs.push_back(std::move(desc));
      }
    }
    EncodeBatchResolveResponse(entries, registry_ == nullptr ? 0 : registry_->PageSize(), descs,
                               response->mutable_resolved());
    RecordInboundGet(bytes, "remote");
    return grpc::Status::OK;
  }

  grpc::Status BatchReleaseReadPins(grpc::ServerContext*,
                                    const ::umbp::BatchReleaseReadPinsRequest* request,
                                    ::umbp::BatchReleaseReadPinsResponse* response) override {
    if (request->expected_owner_incarnation() != owner_incarnation_) return StaleOwner();
    if (pool_ == nullptr) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "no peer pool");
    std::vector<uint64_t> tokens(request->read_pin_tokens().begin(),
                                 request->read_pin_tokens().end());
    for (bool success : pool_->ReleaseReadPins(tokens)) response->add_success(success);
    std::vector<std::string> abandoned(request->abandoned_operation_ids().begin(),
                                       request->abandoned_operation_ids().end());
    const auto cancelled = request->no_gpu_work_submitted()
                               ? pool_->AbandonReadOperations(abandoned)
                               : std::vector<bool>(abandoned.size(), false);
    // successes are tokens first, then abandoned operation ids.
    for (bool success : cancelled) response->add_success(success);
    return grpc::Status::OK;
  }

  grpc::Status BatchAbortProtectedSlots(
      grpc::ServerContext*, const ::umbp::BatchAbortProtectedSlotsRequest* request,
      ::umbp::BatchAbortProtectedSlotsResponse* response) override {
    if (request->expected_owner_incarnation() != owner_incarnation_) return StaleOwner();
    if (pool_ == nullptr) return grpc::Status(grpc::StatusCode::UNAVAILABLE, "no peer pool");
    std::vector<PoolProtectedAbortRequest> entries;
    for (const auto& entry : request->entries()) {
      entries.push_back({{BackendIdFromSlotId(entry.slot_id()), LocalSlotId(entry.slot_id())},
                         entry.operation_id(),
                         entry.transfer_drained(),
                         entry.no_gpu_work_submitted()});
    }
    for (bool success : pool_->BatchAbortProtected(entries)) response->add_success(success);
    return grpc::Status::OK;
  }

 private:
  grpc::Status StaleOwner() const {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "owner incarnation mismatch");
  }
  MediumBackend* Backend(uint32_t backend_id) const {
    return registry_ == nullptr ? nullptr : registry_->Get(backend_id);
  }

  // Every backend on this peer, snapshotted at construction for bootstrap
  // descriptor publication. Placement/read order lives in PoolPolicy.
  const std::vector<MediumBackend*>& Media() const { return media_; }

  void RecordInboundPut(uint64_t bytes, const char* traffic) {
    if (metric_sink_ == nullptr || bytes == 0) return;
    MetricSink::Labels labels = {{"traffic", std::string(traffic)}};
    metric_sink_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL_HELP, labels,
                             static_cast<double>(bytes));
  }

  void RecordInboundGet(uint64_t bytes, const char* traffic) {
    if (metric_sink_ == nullptr || bytes == 0) return;
    MetricSink::Labels labels = {{"traffic", std::string(traffic)}};
    metric_sink_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP, labels,
                             static_cast<double>(bytes));
  }

  PeerPool* pool_;
  BackendRegistry* registry_;
  const std::vector<MediumBackend*> media_;
  const std::vector<uint8_t>& engine_desc_bytes_;
  MetricSink* metric_sink_;
  const std::string owner_incarnation_ = [] {
    std::random_device random;
    return std::to_string(random()) + "-" + std::to_string(random()) + "-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  }();
};

PeerServiceServer::PeerServiceServer(PeerPool& pool, std::vector<uint8_t> engine_desc_bytes,
                                     MetricSink* metric_sink)
    : pool_(&pool), metric_sink_(metric_sink), engine_desc_bytes_(std::move(engine_desc_bytes)) {
  service_ = std::make_unique<UMBPPeerServiceImpl>(pool_, metric_sink_, engine_desc_bytes_);
}

PeerServiceServer::PeerServiceServer(BackendRegistry* registry,
                                     std::vector<uint8_t> engine_desc_bytes,
                                     MetricSink* metric_sink)
    : owned_pool_(registry == nullptr
                      ? nullptr
                      : std::make_unique<PeerPool>(registry, MakeSingleBackendPolicy())),
      pool_(owned_pool_.get()),
      metric_sink_(metric_sink),
      engine_desc_bytes_(std::move(engine_desc_bytes)) {
  service_ = std::make_unique<UMBPPeerServiceImpl>(pool_, metric_sink_, engine_desc_bytes_);
}

PeerServiceServer::~PeerServiceServer() { Stop(); }

bool PeerServiceServer::Start(uint16_t port) {
  std::string address = "0.0.0.0:" + std::to_string(port);

  grpc::ServerBuilder builder;
  ApplyGrpcLimits(&builder);
  // gRPC turns SO_REUSEPORT ON by default for TCP servers on Linux, which means
  // a SECOND server binding this same port SUCCEEDS and the kernel then splits
  // incoming connections between the two.  For a peer service that is never
  // right: a client dialing this address would be answered, some fraction of
  // the time, by a different process's peer service with different backends
  // registered — a wrong answer rather than a connection error.  It also made
  // the "port may be in use" check below dead code, since BuildAndStart could
  // not fail that way.  Exactly one peer service owns a port.
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);

  // gRPC's default sync poller pool is small, and every peer RPC that stages
  // SSD bytes occupies one of those threads for the whole device read.  On a
  // loopback read (reader and owner are the same node) the default pool is the
  // binding constraint rather than the drive — the same head-of-line blocking
  // the master server widens its pool for.  Tunable for large deployments.
  {
    auto env_pollers = [](const char* name, int def) -> int {
      const char* v = std::getenv(name);
      if (v == nullptr) return def;
      char* end = nullptr;
      long n = std::strtol(v, &end, 10);
      return (end == v || n <= 0) ? def : static_cast<int>(n);
    };
    const int min_pollers = env_pollers("UMBP_PEER_MIN_POLLERS", 8);
    int max_pollers = env_pollers("UMBP_PEER_MAX_POLLERS", 64);
    if (max_pollers < min_pollers) max_pollers = min_pollers;
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, min_pollers);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, max_pollers);
  }

  int selected_port = 0;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();

  if (!server_) {
    MORI_UMBP_ERROR("[PeerService] Failed to start on {} (port may be in use)", address);
    return false;
  }
  if (selected_port <= 0 || selected_port > std::numeric_limits<uint16_t>::max()) {
    MORI_UMBP_ERROR("[PeerService] invalid selected port {} for {}", selected_port, address);
    server_->Shutdown();
    server_->Wait();
    server_.reset();
    return false;
  }
  bound_port_ = static_cast<uint16_t>(selected_port);
  MORI_UMBP_INFO("[PeerService] Listening on 0.0.0.0:{}", bound_port_);
  return true;
}

void PeerServiceServer::Stop() {
  if (server_) {
    const auto deadline = std::chrono::system_clock::now() + GrpcShutdownDeadline();
    MORI_UMBP_INFO("[PeerService] Shutting down");
    server_->Shutdown(deadline);
    // Block until every in-flight handler has returned (Shutdown's deadline
    // force-cancels any that overrun).  This guarantees no RPC handler is still
    // touching borrowed state (pool_ and its backends) after Stop()
    // returns, so PoolClient can safely tear it down next.
    server_->Wait();
    server_.reset();
    bound_port_ = 0;
  }
}

}  // namespace mori::umbp
