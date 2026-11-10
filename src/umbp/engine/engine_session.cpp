// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/engine/engine_session.h"

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <msgpack.hpp>
#include <random>
#include <unordered_map>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/engine/engine_dispatch.h"
#include "umbp/engine/kv_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp_engine.pb.h"

namespace mori::umbp::engine {
namespace wire = ::umbp::engine;
namespace {

constexpr size_t kPeerRegionBatchMaxFds = 192;
constexpr size_t kPreparedFrameMaxBytes = 12 * 1024 * 1024;

std::string PeerRegionKey(const PeerVmmRegion& region) {
  return region.executor_id + '\0' + region.region_id + '\0' + std::to_string(region.mapping_id);
}

std::vector<PeerVmmRegion> TakePeerRegionBatch(std::vector<PeerVmmRegion>* pending) {
  if (pending == nullptr) return {};
  const size_t count = std::min(pending->size(), kPeerRegionBatchMaxFds);
  std::vector<PeerVmmRegion> out(pending->begin(), pending->begin() + count);
  pending->erase(pending->begin(), pending->begin() + count);
  return out;
}

bool FromWire(const wire::PoolEpoch& in, PoolEpoch* out) {
  if (out == nullptr || in.boot_id().size() != static_cast<int>(out->boot_id.size())) return false;
  std::memcpy(out->boot_id.data(), in.boot_id().data(), out->boot_id.size());
  out->reset_seq = in.reset_seq();
  return true;
}

void ToWire(const PoolEpoch& in, wire::PoolEpoch* out) {
  out->set_boot_id(in.boot_id.data(), in.boot_id.size());
  out->set_reset_seq(in.reset_seq);
}

bool FromWire(const wire::OperationId& in, OperationId* out) {
  if (out == nullptr || in.session_id().size() != static_cast<int>(out->session_id.size()) ||
      in.seq() == 0) {
    return false;
  }
  std::memcpy(out->session_id.data(), in.session_id().data(), out->session_id.size());
  out->seq = in.seq();
  return true;
}

void ToWire(const OperationId& in, wire::OperationId* out) {
  out->set_session_id(in.session_id.data(), in.session_id.size());
  out->set_seq(in.seq);
}

TransferRef FromWire(const wire::MemoryRef& in) {
  TransferRef out;
  out.size = in.size();
  out.loc = static_cast<mori::io::MemoryLocationType>(in.location());
  out.device = in.device();
  out.executor_id = in.executor_id();
  out.region_id = in.region_id();
  out.executor_address = in.address();
  if (!in.memory_desc().empty()) {
    try {
      auto packed = msgpack::unpack(in.memory_desc().data(), in.memory_desc().size());
      packed.get().convert(out.mem);
    } catch (...) {
      out.executor_id.clear();
      out.region_id.clear();
    }
  }
  return out;
}

void ToWire(const TransferRef& in, wire::MemoryRef* out) {
  out->set_address(in.HasEngineEndpoint() ? in.executor_address : 0);
  out->set_size(in.size);
  out->set_location(static_cast<uint32_t>(in.loc));
  out->set_device(in.device);
  out->set_executor_id(in.executor_id);
  out->set_region_id(in.region_id);
  if (in.HasMemoryDesc()) {
    msgpack::sbuffer packed;
    msgpack::pack(packed, in.mem);
    out->set_memory_desc(packed.data(), packed.size());
  }
}

void PlanToWire(const TransferPlan& in, wire::TransferPlan* out) {
  ToWire(in.src, out->mutable_source());
  ToWire(in.dst, out->mutable_target());
  for (size_t value : in.src_offsets) out->add_source_offsets(value);
  for (size_t value : in.dst_offsets) out->add_target_offsets(value);
  for (size_t value : in.sizes) out->add_sizes(value);
  for (size_t value : in.tags) out->add_tags(value);
}

void ToWire(const AllocationHandle& in, wire::AllocationHandle* out);

void ViewToWire(const ObjectView& in, wire::ObjectView* out) {
  out->set_size(in.size);
  out->set_layout_id(in.layout_id);
  for (const auto& span : in.spans) {
    auto* converted = out->add_spans();
    ToWire(span.buffer, converted->mutable_buffer());
    converted->set_offset(span.offset);
    converted->set_length(span.length);
    converted->set_logical_offset(span.logical_offset);
    converted->set_layer_id(span.layer_id);
    converted->set_component(span.component);
  }
}

void ReadPlanToWire(const ReadPlan& in, wire::ReadPlan* out) {
  out->set_ok(in.success);
  out->set_error(in.error);
  out->set_plan_id(in.id);
  ToWire(in.epoch, out->mutable_epoch());
  out->set_storage_namespace(in.name_space);
  out->set_layout_id(in.layout_id);
  out->set_requester_backend_id(in.requester_backend_id);
  for (const auto& key : in.keys) out->add_keys(key);
  for (const auto& source : in.sources) {
    auto* converted = out->add_sources();
    converted->set_key(source.key);
    converted->set_kind(static_cast<wire::ReadSourceKind>(source.kind));
    converted->set_backend_id(source.backend_id);
    converted->set_node_id(source.node_id);
    converted->set_peer_address(source.peer_address);
    converted->set_executor_id(source.executor_id);
    converted->set_layout_id(source.layout_id);
    if (source.handle.has_value()) {
      ToWire(*source.handle, converted->mutable_handle());
    }
  }
  out->set_valid_prefix(in.valid_prefix);
  for (uint32_t boundary : in.legal_boundaries) out->add_legal_boundaries(boundary);
}

void PreparedLoadToWire(const PreparedLoad& in, wire::PreparedLoad* out,
                        std::vector<int>* response_fds) {
  out->set_ok(in.success);
  out->set_error(in.error);
  out->set_prepared_id(in.id);
  out->set_request_id(in.request_id);
  for (const auto& target : in.targets) ToWire(target, out->add_targets());
  for (const auto& grant : in.grants) {
    auto* converted = out->add_grants();
    converted->set_grant_id(grant.id);
    converted->set_backend_id(grant.backend_id);
    converted->set_remote(grant.remote);
    converted->set_engine(grant.engine);
    converted->set_node_id(grant.node_id);
    converted->set_peer_address(grant.peer_address);
    converted->set_key(grant.key);
  }
  for (const auto& bootstrap : in.bootstraps) {
    auto* converted = out->add_bootstraps();
    converted->set_executor_id(bootstrap.executor_id);
    converted->set_engine_desc(bootstrap.engine_desc.data(), bootstrap.engine_desc.size());
  }
  std::unordered_map<int, uint32_t> fd_indices;
  for (const auto& region : in.peer_regions) {
    auto* converted = out->add_peer_regions();
    converted->set_executor_id(region.executor_id);
    converted->set_region_id(region.region_id);
    converted->set_mapped_size(region.mapped_size);
    converted->set_allocation_offset(region.allocation_offset);
    converted->set_region_size(region.region_size);
    converted->set_mapping_id(region.mapping_id);
    converted->set_owner_device(region.owner_device);
    converted->set_handle_kind(region.handle_kind);
    if (!region.fabric.empty()) {
      converted->set_fabric(region.fabric.data(), region.fabric.size());
    }
    if (region.fd >= 0 && response_fds != nullptr) {
      auto [fd, inserted] = fd_indices.emplace(region.fd, response_fds->size());
      if (inserted) response_fds->push_back(region.fd);
      converted->set_fd_index(fd->second);
    }
  }
  for (const auto& layer : in.layers) {
    auto* converted = out->add_layers();
    converted->set_layer_id(layer.layer_id);
    converted->set_component(layer.component);
    converted->set_tag(layer.tag);
    for (const auto& plan : layer.plans) PlanToWire(plan, converted->add_plans());
  }
}

void PeerRegionsToWire(const std::vector<PeerVmmRegion>& regions,
                       google::protobuf::RepeatedPtrField<wire::PeerVmmRegion>* out,
                       std::vector<int>* response_fds) {
  std::unordered_map<int, uint32_t> fd_indices;
  for (const auto& region : regions) {
    auto* converted = out->Add();
    converted->set_executor_id(region.executor_id);
    converted->set_region_id(region.region_id);
    converted->set_mapped_size(region.mapped_size);
    converted->set_allocation_offset(region.allocation_offset);
    converted->set_region_size(region.region_size);
    converted->set_mapping_id(region.mapping_id);
    converted->set_owner_device(region.owner_device);
    converted->set_handle_kind(region.handle_kind);
    if (!region.fabric.empty()) converted->set_fabric(region.fabric.data(), region.fabric.size());
    if (region.fd >= 0 && response_fds != nullptr) {
      auto [fd, inserted] = fd_indices.emplace(region.fd, response_fds->size());
      if (inserted) response_fds->push_back(region.fd);
      converted->set_fd_index(fd->second);
    }
  }
}

bool FromWire(const wire::PageGeometry& in, PageGeometry* out) {
  if (out == nullptr) return false;
  out->page_tokens = in.page_tokens();
  out->num_pages = in.num_pages();
  out->object_bytes = in.object_bytes();
  out->layout_id = in.layout_id();
  out->regions.clear();
  out->regions.reserve(in.regions_size());
  for (const auto& region : in.regions()) {
    PageRegion converted;
    converted.region_id = region.region_id();
    converted.buffer = FromWire(region.buffer());
    converted.row_bytes = region.row_bytes();
    converted.rows_per_page = region.rows_per_page();
    converted.logical_offset = region.logical_offset();
    converted.layer_id = region.layer_id();
    converted.component = region.component();
    out->regions.push_back(std::move(converted));
  }
  if (out->layout_id.empty()) out->layout_id = ComputeLayoutId(*out);
  return out->Validate();
}

bool FromWire(const wire::AllocationHandle& in, AllocationHandle* out) {
  if (out == nullptr || !FromWire(in.epoch(), &out->epoch)) return false;
  out->pool_id = in.pool_id();
  out->backend_id = in.backend_id();
  out->page_id = in.page_id();
  out->generation = in.generation();
  return !out->pool_id.empty() && out->page_id != 0;
}

void ToWire(const AllocationHandle& in, wire::AllocationHandle* out) {
  out->set_pool_id(in.pool_id);
  ToWire(in.epoch, out->mutable_epoch());
  out->set_backend_id(in.backend_id);
  out->set_page_id(in.page_id);
  out->set_generation(in.generation);
}

void InitResponse(const wire::EngineFrame& request, const PoolEpoch& epoch,
                  wire::EngineFrame* response) {
  response->Clear();
  response->set_version(1);
  response->mutable_op()->CopyFrom(request.op());
  ToWire(epoch, response->mutable_epoch());
}

void Status(bool ok, std::string message, wire::EngineFrame* response) {
  auto* status = response->mutable_status();
  status->set_ok(ok);
  status->set_error(std::move(message));
}

void TicketToWire(const RetireTicket& ticket, wire::RetireReply* out) {
  out->set_state(static_cast<uint32_t>(ticket.state));
  ToWire(ticket.id, out->mutable_retire_op());
  for (const auto& handle : ticket.handles) ToWire(handle, out->add_handles());
}

bool SameSession(const std::array<uint8_t, 16>& expected, const OperationId& operation) {
  return expected == operation.session_id;
}

std::array<uint8_t, 16> NewBootId() {
  std::array<uint8_t, 16> out{};
  std::random_device random;
  for (auto& byte : out) byte = static_cast<uint8_t>(random());
  out[6] = static_cast<uint8_t>((out[6] & 0x0f) | 0x40);
  out[8] = static_cast<uint8_t>((out[8] & 0x3f) | 0x80);
  return out;
}

}  // namespace

#ifdef MORI_UMBP_TESTING
std::vector<PeerVmmRegion> TakePeerRegionBatchForTest(std::vector<PeerVmmRegion>* pending) {
  return TakePeerRegionBatch(pending);
}
#endif

EngineSession::EngineSession(EngineControl* control, NowFn now,
                             std::array<uint8_t, 16> node_boot_id)
    : control_(control), now_(std::move(now)), last_activity_(now_()), node_boot_id_(node_boot_id) {
  if (std::all_of(node_boot_id_.begin(), node_boot_id_.end(),
                  [](uint8_t value) { return value == 0; })) {
    node_boot_id_ = NewBootId();
  }
}

EngineSession::~EngineSession() { Fail(); }

bool EngineSession::Handle(const wire::EngineFrame& request, const std::vector<int>& request_fds,
                           wire::EngineFrame* response, std::vector<int>* response_fds,
                           std::string* error) {
  if (control_ == nullptr || response == nullptr || response_fds == nullptr || failed_) {
    if (error) *error = "engine session is unavailable";
    return false;
  }
  response_fds->clear();
  if (request.version() != 1) {
    if (error) *error = "unsupported engine protocol version";
    return false;
  }
  OperationId operation;
  if (!FromWire(request.op(), &operation)) {
    if (error) *error = "invalid engine operation id";
    return false;
  }
  if (operation.seq != last_received_seq_ + 1) {
    if (error) *error = "out-of-order engine operation";
    return false;
  }

  if (!registered_) {
    if (!request.has_register_engine()) {
      if (error) *error = "first engine frame must register";
      return false;
    }
    const auto& in = request.register_engine();
    EngineRegistration registration;
    registration.pool_id = in.pool_id();
    registration.owner_rank = in.rank_id();
    registration.executor_id = in.executor_id();
    registration.engine_desc.assign(in.engine_desc().begin(), in.engine_desc().end());
    const auto close_registration_fds = [&] {
      for (auto& held : registration.vmm_regions) {
        if (held.fd >= 0) close(held.fd);
        held.fd = -1;
      }
    };
    for (const auto& region : in.vmm_regions()) {
      const auto kind = static_cast<VmmHandleKind>(region.handle_kind());
      if (kind != VmmHandleKind::kPosixFd && kind != VmmHandleKind::kFabric) {
        if (error) *error = "unsupported engine VMM handle kind";
        close_registration_fds();
        return false;
      }
      if (kind == VmmHandleKind::kPosixFd && region.fd_index() >= request_fds.size()) {
        if (error) *error = "engine VMM fd index is out of range";
        close_registration_fds();
        return false;
      }
      const int retained =
          kind == VmmHandleKind::kPosixFd ? dup(request_fds[region.fd_index()]) : -1;
      if ((kind == VmmHandleKind::kPosixFd && retained < 0) ||
          (kind == VmmHandleKind::kFabric && region.fabric().empty())) {
        if (error) *error = "failed to retain engine VMM fd";
        close_registration_fds();
        return false;
      }
      if (region.region_size() == 0 || region.allocation_offset() > region.mapped_size() ||
          region.region_size() > region.mapped_size() - region.allocation_offset()) {
        if (retained >= 0) close(retained);
        close_registration_fds();
        if (error) *error = "invalid engine VMM region range";
        return false;
      }
      engine::PeerVmmRegion converted;
      converted.executor_id = registration.executor_id;
      converted.region_id = region.region_id();
      converted.fd = retained;
      converted.mapped_size = region.mapped_size();
      converted.allocation_offset = region.allocation_offset();
      converted.region_size = region.region_size();
      converted.mapping_id = region.mapping_id();
      converted.owner_device = region.owner_device();
      converted.handle_kind = region.handle_kind();
      converted.fabric.assign(region.fabric().begin(), region.fabric().end());
      registration.vmm_regions.push_back(std::move(converted));
    }
    PoolEpoch requested_epoch;
    if (registration.pool_id.empty() || !FromWire(in.epoch(), &requested_epoch) ||
        !FromWire(in.geometry(), &registration.geometry)) {
      close_registration_fds();
      if (error) *error = "invalid engine registration";
      return false;
    }
    registration.epoch.boot_id = node_boot_id_;
    registration.epoch.reset_seq = 0;
    auto backend_id = control_->BackendId(registration.owner_rank);
    if (!backend_id.has_value()) {
      close_registration_fds();
      if (error) *error = "engine rank is outside the fixed topology";
      return false;
    }
    registration.backend_id = *backend_id;
    if (!control_->RegisterEngine(registration)) {
      close_registration_fds();
      if (error) *error = "engine backend activation failed";
      return false;
    }
    rank_ = registration.owner_rank;
    backend_id_ = registration.backend_id;
    epoch_ = registration.epoch;
    session_id_ = operation.session_id;
    executor_id_ = in.executor_id();
    if (!EngineDispatchRegistry::Global().Register(executor_id_, session_id_)) {
      control_->SessionFailed(rank_);
      if (error) *error = "executor id is already bound to another session";
      return false;
    }
    heartbeat_timeout_ = std::chrono::milliseconds(
        std::max<uint32_t>(in.heartbeat_timeout_ms(), std::max<uint32_t>(in.heartbeat_ms(), 1)));
    registered_ = true;
    last_received_seq_ = operation.seq;
    last_activity_ = now_();
    InitResponse(request, epoch_, response);
    auto* reply = response->mutable_registered();
    reply->set_ok(true);
    reply->set_backend_id(backend_id_);
    ToWire(epoch_, reply->mutable_epoch());

    std::unordered_map<int, uint32_t> fd_indices;
    const auto dram_regions = control_->ExportSharedDram(epoch_);
    if (dram_regions.empty()) {
      reply->set_ok(false);
      reply->set_error("shared DRAM export returned no regions");
      return true;
    }
    for (const auto& region : dram_regions) {
      if (region.fd < 0) {
        reply->set_ok(false);
        reply->set_error("shared DRAM export is unavailable");
        response_fds->clear();
        return true;
      }
      auto [it, inserted] = fd_indices.emplace(region.fd, response_fds->size());
      if (inserted) response_fds->push_back(region.fd);
      auto* out = reply->add_dram_regions();
      out->set_region_id(region.region_id);
      ToWire(region.epoch, out->mutable_epoch());
      out->set_backend_id(region.backend_id);
      out->set_buffer_index(region.buffer_index);
      out->set_fd_index(it->second);
      out->set_fd_offset(region.fd_offset);
      out->set_length(region.length);
      out->set_mapped_length(region.mapped_length);
      out->set_numa_node(region.numa_node);
      out->set_alignment(region.alignment);
      out->set_actual_backing(static_cast<uint32_t>(region.actual_backing));
      ToWire(region.memory, out->mutable_memory());
    }
    return true;
  }

  PoolEpoch request_epoch;
  if (!SameSession(session_id_, operation) || !FromWire(request.epoch(), &request_epoch) ||
      request_epoch != epoch_) {
    if (error) *error = "stale engine session or pool epoch";
    return false;
  }
  last_received_seq_ = operation.seq;
  last_activity_ = now_();
  InitResponse(request, epoch_, response);
  if (request.has_heartbeat()) {
    if (PeerPool* pool = control_->Pool();
        pool != nullptr && request.heartbeat().last_received_seq() > 0) {
      OperationId watermark{session_id_, request.heartbeat().last_received_seq()};
      pool->AdvanceEngineAckWatermark(backend_id_, watermark);
    }
    auto* heartbeat = response->mutable_heartbeat();
    heartbeat->set_monotonic_seq(request.heartbeat().monotonic_seq());
    heartbeat->set_last_received_seq(last_received_seq_);
    heartbeat->set_last_applied_seq(last_received_seq_);
    return true;
  }
  if (request.has_poll_work()) {
    auto work = EngineDispatchRegistry::Global().Fetch(executor_id_, session_id_);
    if (!work.has_value()) {
      Status(true, {}, response);
      return true;
    }
    auto* out = response->mutable_execute_plans();
    out->set_transfer_id(work->id);
    for (const auto& plan : work->plans) PlanToWire(plan, out->add_plans());
    return true;
  }
  if (request.has_complete_transfer()) {
    std::vector<TransferFailure> failures;
    for (const auto& failure : request.complete_transfer().failures()) {
      TransferFailure converted;
      converted.code = failure.code();
      converted.message = failure.message();
      converted.quiesced = failure.quiesced();
      converted.submitted = failure.submitted();
      for (uint64_t tag : failure.tags()) converted.tags.push_back(tag);
      failures.push_back(std::move(converted));
    }
    const uint32_t raw_outcome = request.complete_transfer().outcome();
    const bool valid_outcome = raw_outcome <= static_cast<uint32_t>(DispatchOutcome::kUnknown);
    const bool ok =
        valid_outcome && EngineDispatchRegistry::Global().Complete(
                             executor_id_, session_id_, request.complete_transfer().transfer_id(),
                             static_cast<DispatchOutcome>(raw_outcome), std::move(failures));
    Status(ok, ok ? std::string{} : "unknown engine transfer", response);
    return true;
  }
  PeerPool* pool = control_->Pool();
  if (pool == nullptr) {
    Status(false, "engine pool is unavailable", response);
    return true;
  }

  if (request.has_lookup()) {
    std::vector<std::string> keys(request.lookup().keys().begin(), request.lookup().keys().end());
    auto plan = control_->Lookup(keys, request.lookup().storage_namespace(),
                                 request.lookup().layout_id(), backend_id_);
    ReadPlanToWire(plan, response->mutable_read_plan());
    return true;
  }
  if (request.has_prepare_load()) {
    std::vector<AllocationHandle> targets;
    targets.reserve(request.prepare_load().targets_size());
    for (const auto& target : request.prepare_load().targets()) {
      AllocationHandle converted;
      if (!FromWire(target, &converted)) {
        response->mutable_prepared()->set_error("invalid load target");
        return true;
      }
      targets.push_back(std::move(converted));
    }
    ReadPlan plan;
    plan.id = request.prepare_load().plan_id();
    plan.epoch = epoch_;
    const std::string reader_session(reinterpret_cast<const char*>(session_id_.data()),
                                     session_id_.size());
    auto prepared = control_->PrepareLoad(
        plan, request.prepare_load().request_id(), targets, request.prepare_load().n_pages(),
        {reader_session, request.prepare_load().request_id()}, operation);
    std::vector<PeerVmmRegion> unsent;
    if (prepared.success) {
      for (const auto& region : prepared.peer_regions) {
        if (sent_peer_regions_.count(PeerRegionKey(region)) == 0) unsent.push_back(region);
      }
      prepared.peer_regions = TakePeerRegionBatch(&unsent);
      if (!unsent.empty()) pending_peer_regions_[prepared.id] = std::move(unsent);
    }
    PreparedLoadToWire(prepared, response->mutable_prepared(), response_fds);
    response->mutable_prepared()->set_peer_regions_more(pending_peer_regions_.count(prepared.id) !=
                                                        0);
    if (prepared.success && (response_fds->size() > kPeerRegionBatchMaxFds ||
                             response->ByteSizeLong() > kPreparedFrameMaxBytes)) {
      pending_peer_regions_.erase(prepared.id);
      response_fds->clear();
      response->clear_prepared();
      auto* rejected = response->mutable_prepared();
      rejected->set_error("prepared load exceeds ordered-frame limit");
      (void)control_->CancelLoad(prepared.id, operation);
      return true;
    }
    for (const auto& region : prepared.peer_regions) {
      sent_peer_regions_.insert(PeerRegionKey(region));
    }
    return true;
  }
  if (request.has_fetch_peer_regions()) {
    const uint64_t prepared_id = request.fetch_peer_regions().prepared_id();
    auto pending = pending_peer_regions_.find(prepared_id);
    auto* batch = response->mutable_peer_region_batch();
    batch->set_prepared_id(prepared_id);
    if (pending == pending_peer_regions_.end()) {
      batch->set_error("unknown peer-region batch");
      return true;
    }
    std::vector<PeerVmmRegion> regions = TakePeerRegionBatch(&pending->second);
    PeerRegionsToWire(regions, batch->mutable_peer_regions(), response_fds);
    batch->set_ok(true);
    batch->set_more(!pending->second.empty());
    for (const auto& region : regions) sent_peer_regions_.insert(PeerRegionKey(region));
    if (pending->second.empty()) pending_peer_regions_.erase(pending);
    return true;
  }
  if (request.has_complete_load()) {
    const uint32_t raw = request.complete_load().outcome();
    const bool valid = raw <= static_cast<uint32_t>(LoadOutcome::kCanceled);
    const bool ok = valid && control_->CompleteLoad(request.complete_load().prepared_id(),
                                                    static_cast<LoadOutcome>(raw), operation);
    if (ok && raw != static_cast<uint32_t>(LoadOutcome::kPending)) {
      pending_peer_regions_.erase(request.complete_load().prepared_id());
    }
    auto* completed = response->mutable_load_completed();
    completed->set_ok(ok);
    if (!ok) completed->set_error("load completion rejected");
    return true;
  }
  if (request.has_cancel_load()) {
    const auto result = control_->CancelLoad(request.cancel_load().prepared_id(), operation);
    auto* cancelled = response->mutable_cancelled();
    cancelled->set_result(static_cast<uint32_t>(result));
    if (result == CancelLoadResult::kRejected) cancelled->set_error("load cancellation rejected");
    if (result == CancelLoadResult::kNeverSubmitted || result == CancelLoadResult::kCompleted) {
      pending_peer_regions_.erase(request.cancel_load().prepared_id());
    }
    return true;
  }

  if (request.has_delegate()) {
    auto delegated =
        pool->DelegateEnginePages(backend_id_, rank_, request.delegate().count(), operation);
    auto* out = response->mutable_delegated();
    out->set_ok(delegated.success);
    for (const auto& handle : delegated.handles) ToWire(handle, out->add_handles());
    return true;
  }
  if (request.has_publish()) {
    std::vector<ResidentRecord> records;
    records.reserve(request.publish().records_size());
    for (const auto& record : request.publish().records()) {
      ResidentRecord converted;
      converted.key = record.key();
      converted.parent_key = record.parent_key();
      if (converted.key.empty() || !FromWire(record.handle(), &converted.handle)) {
        Status(false, "invalid resident record", response);
        return true;
      }
      records.push_back(std::move(converted));
    }
    auto published = pool->PublishResident(backend_id_, records, operation);
    auto* out = response->mutable_published();
    out->set_ok(published.success);
    for (bool accepted : published.accepted) out->add_accepted(accepted);
    return true;
  }
  if (request.has_io_hold()) {
    std::vector<AllocationHandle> handles;
    for (const auto& handle : request.io_hold().handles()) {
      AllocationHandle converted;
      if (!FromWire(handle, &converted)) {
        response->mutable_io_held()->set_ok(false);
        return true;
      }
      handles.push_back(std::move(converted));
    }
    response->mutable_io_held()->set_ok(pool->HoldEngineIO(backend_id_, handles, operation));
    return true;
  }
  if (request.has_io_complete()) {
    OperationId hold;
    const uint32_t raw_proof = request.io_complete().proof();
    const bool valid = FromWire(request.io_complete().hold(), &hold) &&
                       raw_proof <= static_cast<uint32_t>(CompletionProof::kUnknown);
    const bool ok =
        valid && pool->CompleteEngineIO(backend_id_, hold, static_cast<CompletionProof>(raw_proof));
    Status(ok, ok ? std::string{} : "I/O completion rejected", response);
    return true;
  }
  if (request.has_quarantine()) {
    std::vector<AllocationHandle> handles;
    handles.reserve(request.quarantine().handles_size());
    for (const auto& handle : request.quarantine().handles()) {
      AllocationHandle converted;
      if (!FromWire(handle, &converted)) {
        Status(false, "invalid quarantine handle", response);
        return true;
      }
      handles.push_back(std::move(converted));
    }
    const bool ok = !handles.empty() && pool->QuarantineEnginePages(backend_id_, handles);
    Status(ok, ok ? std::string{} : "quarantine rejected", response);
    return true;
  }
  if (request.has_retire()) {
    std::vector<AllocationHandle> handles;
    handles.reserve(request.retire().handles_size());
    for (const auto& handle : request.retire().handles()) {
      AllocationHandle converted;
      if (!FromWire(handle, &converted)) {
        Status(false, "invalid retire handle", response);
        return true;
      }
      handles.push_back(std::move(converted));
    }
    TicketToWire(pool->RetireEnginePages(backend_id_, handles, operation),
                 response->mutable_retired());
    return true;
  }
  if (request.has_poll_retires()) {
    auto* out = response->mutable_released();
    for (const auto& ticket : pool->PollReadyEngineRetires(backend_id_)) {
      TicketToWire(ticket, out->add_tickets());
    }
    for (const auto& handle : pool->PollEngineQuarantines(backend_id_)) {
      ToWire(handle, out->add_quarantined());
    }
    return true;
  }
  if (request.has_acknowledge_released()) {
    auto* reply = response->mutable_acknowledge_reply();
    for (const auto& id : request.acknowledge_released().retire_ops()) {
      OperationId retire;
      reply->add_accepted(FromWire(id, &retire) &&
                          pool->AcknowledgeEngineReleased(backend_id_, retire));
    }
    return true;
  }
  if (request.has_reset()) {
    auto lease = pool->BeginEngineReset(backend_id_, operation);
    auto* out = response->mutable_reset_reply();
    out->set_ready(lease.ready);
    ToWire(lease.id, out->mutable_lease());
    ToWire(lease.old_epoch, out->mutable_old_epoch());
    ToWire(lease.new_epoch, out->mutable_new_epoch());
    if (lease.ready) {
      for (const auto& handle : lease.approved) ToWire(handle, out->add_approved());
    }
    return true;
  }
  if (request.has_reset_finish()) {
    OperationId lease;
    if (!FromWire(request.reset_finish().lease(), &lease)) {
      Status(false, "invalid reset lease", response);
      return true;
    }
    PoolEpoch resulting_epoch;
    std::vector<AllocationHandle> approved;
    const bool ok = pool->FinishEngineReset(backend_id_, lease, request.reset_finish().commit(),
                                            &resulting_epoch, &approved);
    auto* out = response->mutable_reset_finished();
    out->set_ok(ok);
    ToWire(ok ? resulting_epoch : epoch_, out->mutable_epoch());
    if (ok && request.reset_finish().commit()) {
      for (const auto& handle : approved) ToWire(handle, out->add_approved());
    }
    if (ok && request.reset_finish().commit()) epoch_ = resulting_epoch;
    return true;
  }
  Status(false, "unsupported engine frame", response);
  return true;
}

bool EngineSession::Expired() const { return now_() - last_activity_ > heartbeat_timeout_; }

void EngineSession::Fail() {
  if (failed_) return;
  failed_ = true;
  if (registered_) EngineDispatchRegistry::Global().Unregister(executor_id_, session_id_);
  if (registered_ && control_ != nullptr) control_->SessionFailed(rank_);
}

bool EngineSession::PopWorkFrame(wire::EngineFrame* response) {
  if (response == nullptr || !registered_ || failed_) return false;
  auto work = EngineDispatchRegistry::Global().Fetch(executor_id_, session_id_);
  if (!work.has_value()) return false;
  response->Clear();
  response->set_version(1);
  response->set_server_push(true);
  response->mutable_op()->set_session_id(session_id_.data(), session_id_.size());
  response->mutable_op()->set_seq(0);
  ToWire(epoch_, response->mutable_epoch());
  auto* out = response->mutable_execute_plans();
  out->set_transfer_id(work->id);
  for (const auto& plan : work->plans) PlanToWire(plan, out->add_plans());
  return true;
}

int EngineSession::WorkEventFd() const {
  return registered_ && !failed_
             ? EngineDispatchRegistry::Global().EventFd(executor_id_, session_id_)
             : -1;
}

void EngineSession::FailPushedWork(uint64_t transfer_id) {
  (void)EngineDispatchRegistry::Global().FailFetched(executor_id_, session_id_, transfer_id,
                                                     DispatchOutcome::kNeverSubmitted);
}

EngineSessionServer::EngineSessionServer(EngineControl* control)
    : control_(control), node_boot_id_(NewBootId()) {}

EngineSessionServer::~EngineSessionServer() { Shutdown(); }

bool EngineSessionServer::Start(const std::string& path, std::string* error) {
  if (control_ == nullptr || path.empty() || listen_fd_ >= 0) return false;
  if (!standalone::EnsureParentDirectory(path, error)) return false;
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    if (error) *error = "engine UDS path is too long";
    return false;
  }
  unlink(path.c_str());
  listen_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) {
    if (error) *error = std::string("engine socket: ") + std::strerror(errno);
    return false;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
  const socklen_t length = sizeof(sa_family_t) + path.size() + 1;
  if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), length) != 0 ||
      listen(listen_fd_, 8) != 0) {
    if (error) *error = std::string("engine bind/listen: ") + std::strerror(errno);
    close(listen_fd_);
    listen_fd_ = -1;
    unlink(path.c_str());
    return false;
  }
  chmod(path.c_str(), 0600);
  path_ = path;
  stopping_.store(false);
  accept_thread_ = std::thread(&EngineSessionServer::AcceptLoop, this);
  return true;
}

void EngineSessionServer::Shutdown() {
  if (stopping_.exchange(true)) return;
  if (listen_fd_ >= 0) {
    shutdown(listen_fd_, SHUT_RDWR);
    close(listen_fd_);
    listen_fd_ = -1;
  }
  {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (int fd : clients_) shutdown(fd, SHUT_RDWR);
  }
  if (accept_thread_.joinable()) accept_thread_.join();
  {
    std::unique_lock<std::mutex> lock(clients_mutex_);
    clients_cv_.wait(lock, [&] { return clients_.empty(); });
  }
  if (!path_.empty()) unlink(path_.c_str());
}

void EngineSessionServer::AcceptLoop() {
  while (!stopping_.load()) {
    int fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EINTR) continue;
      if (!stopping_.load())
        MORI_UMBP_ERROR("[EngineSessionServer] accept failed: {}", std::strerror(errno));
      break;
    }
    {
      std::lock_guard<std::mutex> lock(clients_mutex_);
      clients_.push_back(fd);
    }
    timeval send_timeout{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
    std::thread(&EngineSessionServer::Serve, this, fd).detach();
  }
}

void EngineSessionServer::Serve(int fd) {
  EngineSession session(control_, EngineSession::Clock::now, node_boot_id_);
  while (!stopping_.load()) {
    wire::EngineFrame pushed;
    while (session.PopWorkFrame(&pushed)) {
      std::string pushed_payload;
      std::string pushed_error;
      if (!pushed.SerializeToString(&pushed_payload) ||
          !standalone::SendFramedMessage(fd, pushed_payload, {}, &pushed_error)) {
        if (pushed.has_execute_plans()) {
          session.FailPushedWork(pushed.execute_plans().transfer_id());
        }
        goto done;
      }
    }
    const int work_fd = session.WorkEventFd();
    pollfd ready[2]{{fd, POLLIN, 0}, {work_fd, POLLIN, 0}};
    const nfds_t ready_count = work_fd >= 0 ? 2 : 1;
    const int poll_rc = poll(ready, ready_count, 100);
    if (poll_rc < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (poll_rc == 0) {
      if (session.Expired()) break;
      continue;
    }
    if (ready_count == 2 && (ready[1].revents & POLLIN) != 0) {
      uint64_t wake = 0;
      while (read(work_fd, &wake, sizeof(wake)) == sizeof(wake)) {
      }
      // Drain immediately at the top of the loop. The rank-side progress loop
      // already coalesces adjacent work; delaying here also stalls control
      // frames that are ready on the session socket.
      continue;
    }
    if ((ready[0].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) break;
    if ((ready[0].revents & POLLIN) == 0) continue;
    timeval frame_timeout{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &frame_timeout, sizeof(frame_timeout));
    std::string payload;
    std::vector<int> fds;
    std::string error;
    if (!standalone::RecvFramedMessage(fd, &payload, &fds, 16 * 1024 * 1024, 256, &error)) {
      for (int attachment : fds) close(attachment);
      break;
    }
    wire::EngineFrame request;
    wire::EngineFrame response;
    std::vector<int> response_fds;
    const bool parsed = request.ParseFromString(payload);
    const bool handled = parsed && session.Handle(request, fds, &response, &response_fds, &error);
    for (int attachment : fds) close(attachment);
    if (!handled) break;
    if (!response.SerializeToString(&payload) ||
        !standalone::SendFramedMessage(fd, payload, response_fds, &error)) {
      break;
    }
  }
done:
  session.Fail();
  std::lock_guard<std::mutex> lock(clients_mutex_);
  auto it = std::find(clients_.begin(), clients_.end(), fd);
  if (it != clients_.end()) clients_.erase(it);
  close(fd);
  clients_cv_.notify_all();
}

}  // namespace mori::umbp::engine
