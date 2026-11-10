// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/engine/engine_client.h"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <msgpack.hpp>
#include <random>
#include <unordered_set>

#include "umbp/engine/engine_dispatch.h"
#include "umbp/engine/engine_executor.h"
#include "umbp/engine/kv_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp_engine.pb.h"

namespace mori::umbp::engine {
namespace wire = ::umbp::engine;
namespace {

bool IsTerminalPublishReceipt(PublishReceiptState state) {
  return state == PublishReceiptState::kAccepted || state == PublishReceiptState::kRejected ||
         state == PublishReceiptState::kCancelled || state == PublishReceiptState::kUnknown;
}

void ToWire(const PoolEpoch& in, wire::PoolEpoch* out) {
  out->set_boot_id(in.boot_id.data(), in.boot_id.size());
  out->set_reset_seq(in.reset_seq);
}

bool FromWire(const wire::PoolEpoch& in, PoolEpoch* out) {
  if (out == nullptr || in.boot_id().size() != static_cast<int>(out->boot_id.size())) return false;
  std::memcpy(out->boot_id.data(), in.boot_id().data(), out->boot_id.size());
  out->reset_seq = in.reset_seq();
  return true;
}

void ToWire(const OperationId& in, wire::OperationId* out) {
  out->set_session_id(in.session_id.data(), in.session_id.size());
  out->set_seq(in.seq);
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

void ToWire(const TransferRef& in, std::string_view executor_id, std::string_view region_id,
            wire::MemoryRef* out) {
  const uint64_t address = in.executor_address != 0
                               ? in.executor_address
                               : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(in.host_ptr));
  out->set_address(address);
  out->set_size(in.size);
  out->set_location(static_cast<uint32_t>(in.loc));
  out->set_device(in.device);
  out->set_executor_id(in.executor_id.empty() ? std::string(executor_id) : in.executor_id);
  out->set_region_id(in.region_id.empty() ? std::string(region_id) : in.region_id);
  if (in.HasMemoryDesc()) {
    msgpack::sbuffer packed;
    msgpack::pack(packed, in.mem);
    out->set_memory_desc(packed.data(), packed.size());
  }
}

void ToWire(const PageGeometry& in, std::string_view executor_id, wire::PageGeometry* out) {
  out->set_page_tokens(in.page_tokens);
  out->set_num_pages(in.num_pages);
  out->set_object_bytes(in.object_bytes);
  out->set_layout_id(in.layout_id.empty() ? ComputeLayoutId(in) : in.layout_id);
  for (const auto& region : in.regions) {
    auto* converted = out->add_regions();
    converted->set_region_id(region.region_id);
    ToWire(region.buffer, executor_id, region.region_id, converted->mutable_buffer());
    converted->set_row_bytes(region.row_bytes);
    converted->set_rows_per_page(region.rows_per_page);
    converted->set_logical_offset(region.logical_offset);
    converted->set_layer_id(region.layer_id);
    converted->set_component(region.component);
  }
}

void ToWire(const AllocationHandle& in, wire::AllocationHandle* out) {
  out->set_pool_id(in.pool_id);
  ToWire(in.epoch, out->mutable_epoch());
  out->set_backend_id(in.backend_id);
  out->set_page_id(in.page_id);
  out->set_generation(in.generation);
}

bool FromWire(const wire::AllocationHandle& in, AllocationHandle* out) {
  if (out == nullptr || !FromWire(in.epoch(), &out->epoch)) return false;
  out->pool_id = in.pool_id();
  out->backend_id = in.backend_id();
  out->page_id = in.page_id();
  out->generation = in.generation();
  return !out->pool_id.empty() && out->page_id != 0;
}

SharedDramRegion FromWire(const wire::SharedDramRegion& in, int fd) {
  SharedDramRegion out;
  out.region_id = in.region_id();
  FromWire(in.epoch(), &out.epoch);
  out.backend_id = in.backend_id();
  out.buffer_index = in.buffer_index();
  out.fd = fd;
  out.fd_offset = in.fd_offset();
  out.length = in.length();
  out.mapped_length = in.mapped_length();
  out.numa_node = in.numa_node();
  out.alignment = in.alignment();
  out.actual_backing = static_cast<HostBufferBacking>(in.actual_backing());
  out.memory.size = in.memory().size();
  out.memory.loc = static_cast<mori::io::MemoryLocationType>(in.memory().location());
  out.memory.device = in.memory().device();
  out.memory.executor_id = in.memory().executor_id();
  out.memory.region_id = in.memory().region_id();
  out.memory.executor_address = in.memory().address();
  if (!in.memory().memory_desc().empty()) {
    try {
      auto packed =
          msgpack::unpack(in.memory().memory_desc().data(), in.memory().memory_desc().size());
      packed.get().convert(out.memory.mem);
    } catch (...) {
      out.memory.mem = {};
    }
  }
  return out;
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

bool PlanFromWire(const wire::TransferPlan& in, TransferPlan* out) {
  if (out == nullptr || in.source_offsets_size() != in.target_offsets_size() ||
      in.source_offsets_size() != in.sizes_size()) {
    return false;
  }
  out->src = FromWire(in.source());
  out->dst = FromWire(in.target());
  out->src_offsets.assign(in.source_offsets().begin(), in.source_offsets().end());
  out->dst_offsets.assign(in.target_offsets().begin(), in.target_offsets().end());
  out->sizes.assign(in.sizes().begin(), in.sizes().end());
  out->tags.assign(in.tags().begin(), in.tags().end());
  out->dir = TransferDirection::kLocal;
  return true;
}

bool ReadPlanFromWire(const wire::ReadPlan& in, ReadPlan* out) {
  if (out == nullptr || !FromWire(in.epoch(), &out->epoch) || in.sources_size() != in.keys_size()) {
    return false;
  }
  out->success = in.ok();
  out->error = in.error();
  out->id = in.plan_id();
  out->name_space = in.storage_namespace();
  out->layout_id = in.layout_id();
  out->requester_backend_id = in.requester_backend_id();
  out->keys.assign(in.keys().begin(), in.keys().end());
  out->sources.clear();
  for (const auto& source : in.sources()) {
    ReadSourceHint converted;
    converted.key = source.key();
    converted.kind = static_cast<ReadSourceKind>(source.kind());
    converted.backend_id = source.backend_id();
    converted.node_id = source.node_id();
    converted.peer_address = source.peer_address();
    converted.executor_id = source.executor_id();
    converted.layout_id = source.layout_id();
    if (source.has_handle()) {
      AllocationHandle handle;
      if (!FromWire(source.handle(), &handle)) return false;
      converted.handle = std::move(handle);
    }
    out->sources.push_back(std::move(converted));
  }
  out->valid_prefix = in.valid_prefix();
  out->legal_boundaries.assign(in.legal_boundaries().begin(), in.legal_boundaries().end());
  return !out->success || (out->id != 0 && out->valid_prefix <= out->keys.size());
}

bool AppendPeerRegionsFromWire(
    const google::protobuf::RepeatedPtrField<wire::PeerVmmRegion>& regions,
    const std::vector<int>& fds, std::vector<PeerVmmRegion>* out) {
  if (out == nullptr) return false;
  for (const auto& region : regions) {
    const auto kind = static_cast<VmmHandleKind>(region.handle_kind());
    if (kind != VmmHandleKind::kPosixFd && kind != VmmHandleKind::kFabric) return false;
    if (kind == VmmHandleKind::kPosixFd && region.fd_index() >= fds.size()) return false;
    PeerVmmRegion converted;
    converted.executor_id = region.executor_id();
    converted.region_id = region.region_id();
    converted.fd = kind == VmmHandleKind::kPosixFd ? fds[region.fd_index()] : -1;
    converted.mapped_size = region.mapped_size();
    converted.allocation_offset = region.allocation_offset();
    converted.region_size = region.region_size();
    converted.mapping_id = region.mapping_id();
    converted.owner_device = region.owner_device();
    converted.handle_kind = region.handle_kind();
    converted.fabric.assign(region.fabric().begin(), region.fabric().end());
    if (converted.region_size == 0 || converted.allocation_offset > converted.mapped_size ||
        converted.region_size > converted.mapped_size - converted.allocation_offset) {
      return false;
    }
    out->push_back(std::move(converted));
  }
  return true;
}

bool PreparedLoadFromWire(const wire::PreparedLoad& in, const std::vector<int>& fds,
                          PreparedLoad* out) {
  if (out == nullptr) return false;
  out->success = in.ok();
  out->error = in.error();
  out->id = in.prepared_id();
  out->request_id = in.request_id();
  out->targets.clear();
  for (const auto& target : in.targets()) {
    AllocationHandle converted;
    if (!FromWire(target, &converted)) return false;
    out->targets.push_back(std::move(converted));
  }
  out->grants.clear();
  for (const auto& grant : in.grants()) {
    out->grants.push_back(LoadGrantRef{grant.grant_id(), grant.backend_id(), grant.remote(),
                                       grant.engine(), grant.node_id(), grant.peer_address(),
                                       grant.key()});
  }
  out->bootstraps.clear();
  for (const auto& bootstrap : in.bootstraps()) {
    out->bootstraps.push_back(ExecutorBootstrap{
        bootstrap.executor_id(),
        std::vector<uint8_t>(bootstrap.engine_desc().begin(), bootstrap.engine_desc().end())});
  }
  out->peer_regions.clear();
  if (!AppendPeerRegionsFromWire(in.peer_regions(), fds, &out->peer_regions)) return false;
  out->layers.clear();
  for (const auto& layer : in.layers()) {
    LayerTransferPlan converted;
    converted.layer_id = layer.layer_id();
    converted.component = layer.component();
    converted.tag = layer.tag();
    for (const auto& plan : layer.plans()) {
      TransferPlan transfer;
      if (!PlanFromWire(plan, &transfer)) return false;
      converted.plans.push_back(std::move(transfer));
    }
    out->layers.push_back(std::move(converted));
  }
  return !out->success || (out->id != 0 && !out->targets.empty() && !out->layers.empty());
}

std::array<uint8_t, 16> NewSessionId() {
  std::array<uint8_t, 16> out{};
  std::random_device random;
  for (auto& byte : out) byte = static_cast<uint8_t>(random());
  out[6] = static_cast<uint8_t>((out[6] & 0x0f) | 0x40);
  out[8] = static_cast<uint8_t>((out[8] & 0x3f) | 0x80);
  return out;
}

void CloseAll(std::vector<int>* fds) {
  if (fds == nullptr) return;
  for (int fd : *fds) close(fd);
  fds->clear();
}

}  // namespace

EngineClient::EngineClient(EngineClientConfig config)
    : config_(std::move(config)), session_id_(NewSessionId()) {}

EngineClient::~EngineClient() { Close(); }

void EngineClient::AttachExecutor(EngineExecutor* executor) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (executor_ != nullptr && executor_ != executor) executor_->AttachClient(nullptr);
  executor_ = executor;
  if (executor_ != nullptr) {
    executor_->AttachClient(this);
    config_.engine_desc = executor_->PackedEngineDesc();
  }
}

std::vector<SharedDramRegion> EngineClient::dram_regions() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return dram_regions_;
}

PoolRegistration::~PoolRegistration() {
  for (auto& region : dram_regions) {
    if (region.fd >= 0) close(region.fd);
    region.fd = -1;
  }
}

PoolRegistration::PoolRegistration(PoolRegistration&& other) noexcept
    : success(other.success),
      backend_id(other.backend_id),
      epoch(other.epoch),
      dram_regions(std::move(other.dram_regions)),
      error(std::move(other.error)) {
  other.success = false;
}

PoolRegistration& PoolRegistration::operator=(PoolRegistration&& other) noexcept {
  if (this == &other) return *this;
  for (auto& region : dram_regions) {
    if (region.fd >= 0) close(region.fd);
  }
  success = other.success;
  backend_id = other.backend_id;
  epoch = other.epoch;
  dram_regions = std::move(other.dram_regions);
  error = std::move(other.error);
  other.success = false;
  return *this;
}

bool EngineClient::Connect(std::string* error) {
  if (failed_.load()) return false;
  if (socket_ >= 0) return true;
  const std::string path = standalone::DeriveEngineSocketPath(config_.node_address);
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    if (error) *error = "engine UDS path is too long";
    return false;
  }
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    if (error) *error = std::string("engine socket: ") + std::strerror(errno);
    return false;
  }
  timeval timeout;
  timeout.tv_sec = config_.heartbeat_timeout_ms / 1000;
  timeout.tv_usec = (config_.heartbeat_timeout_ms % 1000) * 1000;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
  const socklen_t length = sizeof(sa_family_t) + path.size() + 1;
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), length) != 0) {
    if (error) *error = std::string("engine connect: ") + std::strerror(errno);
    close(fd);
    return false;
  }
  socket_ = fd;
  return true;
}

OperationId EngineClient::NextOperation() { return OperationId{session_id_, next_seq_++}; }

bool EngineClient::Call(wire::EngineFrame* request, wire::EngineFrame* response,
                        std::vector<int>* response_fds, std::string* error,
                        const std::vector<int>& request_fds) {
  if (request == nullptr || response == nullptr || response_fds == nullptr || closing_.load()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(call_mutex_);
  return CallLocked(request, response, response_fds, error, request_fds);
}

bool EngineClient::CallLocked(wire::EngineFrame* request, wire::EngineFrame* response,
                              std::vector<int>* response_fds, std::string* error,
                              const std::vector<int>& request_fds) {
  if (request == nullptr || response == nullptr || response_fds == nullptr || closing_.load() ||
      failed_.load()) {
    return false;
  }
  if (!Connect(error)) {
    FailStopLocked();
    return false;
  }
  const OperationId operation = NextOperation();
  request->set_version(1);
  ToWire(operation, request->mutable_op());
  ToWire(epoch_, request->mutable_epoch());
  std::string payload;
  if (!request->SerializeToString(&payload)) return false;
  if (receiver_started_.load()) {
    {
      std::lock_guard<std::mutex> response_lock(response_mutex_);
      expected_response_seq_ = operation.seq;
      received_response_.reset();
      CloseAll(&received_response_fds_);
    }
#ifdef MORI_UMBP_TESTING
    if (before_call_send_hook_) before_call_send_hook_(*request);
#endif
    if (!standalone::SendFramedMessage(socket_, payload, request_fds, error)) {
      FailStopLocked();
      return false;
    }
    std::unique_lock<std::mutex> response_lock(response_mutex_);
    if (!response_cv_.wait_for(
            response_lock, std::chrono::milliseconds(config_.heartbeat_timeout_ms),
            [&] { return received_response_.has_value() || failed_.load() || closing_.load(); }) ||
        !received_response_.has_value()) {
      response_lock.unlock();
      FailStopLocked();
      return false;
    }
    *response = std::move(*received_response_);
    received_response_.reset();
    *response_fds = std::move(received_response_fds_);
    expected_response_seq_ = 0;
  } else {
    if (!standalone::SendFramedMessage(socket_, payload, request_fds, error) ||
        !standalone::RecvFramedMessage(socket_, &payload, response_fds, 16 * 1024 * 1024, 256,
                                       error) ||
        !response->ParseFromString(payload)) {
      CloseAll(response_fds);
      FailStopLocked();
      return false;
    }
  }
  OperationId reply;
  PoolEpoch reply_epoch;
  if (response->version() != 1 || !FromWire(response->op(), &reply) || reply != operation ||
      !FromWire(response->epoch(), &reply_epoch) ||
      (!request->has_register_engine() && reply_epoch != epoch_)) {
    CloseAll(response_fds);
    if (error) *error = "engine response identity mismatch";
    FailStopLocked();
    return false;
  }
  if (request->has_register_engine()) epoch_ = reply_epoch;
  last_received_seq_.store(operation.seq);
  healthy_.store(true);
  return true;
}

void EngineClient::FailStopLocked() {
  healthy_.store(false);
  failed_.store(true);
  const int fd = socket_.exchange(-1);
  if (fd >= 0) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    FinishAllPublishTasksLocked();
  }
  progress_cv_.notify_all();
}

PoolRegistration EngineClient::RegisterPool(RankId rank, std::string pool_id, PoolEpoch epoch,
                                            const PageGeometry& geometry) {
  PoolRegistration out;
  if (config_.executor_id.empty() || pool_id.empty() || !geometry.Validate() ||
      heartbeat_thread_.joinable()) {
    out.error = "invalid or repeated engine pool registration";
    return out;
  }
  rank_ = rank;
  pool_id_ = std::move(pool_id);
  layout_id_ = geometry.layout_id.empty() ? ComputeLayoutId(geometry) : geometry.layout_id;
  epoch_ = epoch;
  wire::EngineFrame request;
  auto* registration = request.mutable_register_engine();
  registration->set_rank_id(rank_);
  registration->set_pool_id(pool_id_);
  registration->set_executor_id(config_.executor_id);
  ToWire(epoch_, registration->mutable_epoch());
  PageGeometry registered_geometry = geometry;
  if (executor_ != nullptr) {
    for (auto& region : registered_geometry.regions) {
      TransferRef registered;
      registered.executor_id = config_.executor_id;
      registered.region_id = region.region_id;
      if (!executor_->Hydrate(&registered) || registered.size != region.buffer.size ||
          registered.loc != region.buffer.loc || registered.device != region.buffer.device) {
        out.error = "executor region does not match pool geometry";
        return out;
      }
      region.buffer = std::move(registered);
    }
  }
  ToWire(registered_geometry, config_.executor_id, registration->mutable_geometry());
  registration->set_heartbeat_ms(config_.heartbeat_ms);
  registration->set_heartbeat_timeout_ms(config_.heartbeat_timeout_ms);
  if (!config_.engine_desc.empty()) {
    registration->set_engine_desc(config_.engine_desc.data(), config_.engine_desc.size());
  }
  std::vector<int> registration_fds;
  for (const auto& region : geometry.regions) {
    if (!region.buffer.HasHostPtr()) continue;
    auto allocation = GlobalKvVmmAllocator().Lookup(region.buffer.host_ptr, region.buffer.size);
    if (!allocation.has_value()) continue;
    auto exported = GlobalKvVmmAllocator().Export(region.buffer.host_ptr);
    if (!exported.has_value()) {
      continue;  // CPU tests and non-VMM request-only regions have no export
    }
    auto* wire_region = registration->add_vmm_regions();
    wire_region->set_region_id(region.region_id);
    wire_region->set_mapped_size(exported->mapped_size);
    wire_region->set_allocation_offset(reinterpret_cast<uintptr_t>(region.buffer.host_ptr) -
                                       reinterpret_cast<uintptr_t>(allocation->base));
    wire_region->set_region_size(region.buffer.size);
    wire_region->set_owner_device(region.buffer.device);
    wire_region->set_handle_kind(static_cast<uint32_t>(exported->kind));
    if (exported->kind == VmmHandleKind::kPosixFd && exported->fd >= 0) {
      struct stat identity {};
      if (fstat(exported->fd, &identity) != 0) {
        close(exported->fd);
        registration->mutable_vmm_regions()->RemoveLast();
        continue;
      }
      wire_region->set_mapping_id(static_cast<uint64_t>(identity.st_dev) ^
                                  (static_cast<uint64_t>(identity.st_ino) << 1));
      wire_region->set_fd_index(registration_fds.size());
      registration_fds.push_back(exported->fd);
    } else if (exported->kind == VmmHandleKind::kFabric) {
      wire_region->set_mapping_id(HashBytes(exported->fabric.data(), exported->fabric.size()));
      wire_region->set_fabric(exported->fabric.data(), exported->fabric.size());
    }
  }

  wire::EngineFrame response;
  std::vector<int> fds;
  const bool registered = Call(&request, &response, &fds, &out.error, registration_fds);
  for (int fd : registration_fds) close(fd);
  if (!registered || !response.has_registered() || !response.registered().ok()) {
    if (out.error.empty() && response.has_registered()) out.error = response.registered().error();
    CloseAll(&fds);
    return out;
  }
  const auto& reply = response.registered();
  backend_id_ = reply.backend_id();
  out.backend_id = backend_id_;
  if (!FromWire(reply.epoch(), &out.epoch) || out.epoch != epoch_) {
    out.error = "registration epoch mismatch";
    CloseAll(&fds);
    return out;
  }
  for (const auto& region : reply.dram_regions()) {
    if (region.fd_index() >= fds.size()) {
      out.error = "shared DRAM fd index is out of range";
      CloseAll(&fds);
      for (auto& exported : out.dram_regions) close(exported.fd);
      out.dram_regions.clear();
      return out;
    }
    const int owned = dup(fds[region.fd_index()]);
    if (owned < 0) {
      out.error = "failed to retain shared DRAM fd";
      CloseAll(&fds);
      for (auto& exported : out.dram_regions) close(exported.fd);
      out.dram_regions.clear();
      return out;
    }
    out.dram_regions.push_back(FromWire(region, owned));
  }
  CloseAll(&fds);
  dram_regions_ = std::move(out.dram_regions);
  std::vector<SharedDramRegion> returned_regions;
  returned_regions.reserve(dram_regions_.size());
  for (const auto& region : dram_regions_) {
    SharedDramRegion returned = region;
    returned.fd = dup(region.fd);
    if (returned.fd < 0) {
      for (auto& retained : returned_regions) close(retained.fd);
      out.error = "failed to duplicate shared DRAM descriptor for caller";
      return out;
    }
    returned_regions.push_back(std::move(returned));
  }
  out.dram_regions = std::move(returned_regions);
  healthy_.store(true);
  receiver_started_.store(true);
  receiver_thread_ = std::thread(&EngineClient::ReceiverLoop, this);
#ifdef MORI_UMBP_TESTING
  std::function<void()> after_receiver_started;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    after_receiver_started = after_receiver_started_hook_;
  }
  if (after_receiver_started) after_receiver_started();
#endif
  auto delegated = RequestDelegation(geometry.num_pages);
  if (!delegated.success || delegated.handles.size() != geometry.num_pages) {
    out.error = "initial fixed-pool delegation failed";
    std::lock_guard<std::mutex> lock(call_mutex_);
    FailStopLocked();
    return out;
  }
  out.success = true;
  progress_thread_ = std::thread(&EngineClient::ProgressLoop, this);
  heartbeat_thread_ = std::thread(&EngineClient::HeartbeatLoop, this);
  return out;
}

DelegateResult EngineClient::RequestDelegation(size_t count) {
  DelegateResult out;
  wire::EngineFrame request;
  request.mutable_delegate()->set_count(count);
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_delegated()) return out;
  CloseAll(&fds);
  out.success = response.delegated().ok();
  for (const auto& handle : response.delegated().handles()) {
    AllocationHandle converted;
    if (!FromWire(handle, &converted)) {
      out.success = false;
      out.handles.clear();
      return out;
    }
    out.handles.push_back(std::move(converted));
  }
  if (out.success) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    AddDelegatedLocked(out.handles);
  }
  return out;
}

ReservationResult EngineClient::Reserve(std::string request_id, size_t load_pages,
                                        size_t extend_pages) {
  ReservationResult out;
  if (request_id.empty() || (load_pages == 0 && extend_pages == 0)) return out;
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!healthy_.load() || failed_.load() || quiescing_) return out;
  const size_t needed = load_pages + extend_pages;
  if (state_counts_[static_cast<size_t>(LocalPageState::kCached)] < needed) {
    out.shortfall = needed - state_counts_[static_cast<size_t>(LocalPageState::kCached)];
    return out;
  }
  std::vector<LocalPage*> available;
  available.reserve(needed);
  while (available.size() < needed && !cached_pages_.empty()) {
    ++reserve_page_checks_;
    const uint32_t page_id = cached_pages_.front();
    cached_pages_.pop_front();
    auto page = pages_.find(page_id);
    if (page != pages_.end() && page->second.state == LocalPageState::kCached) {
      available.push_back(&page->second);
    }
  }
  if (available.size() != needed) return out;  // state index corruption: fail closed
  LocalReservation reservation;
  reservation.request_id = std::move(request_id);
  for (size_t i = 0; i < load_pages; ++i) {
    if (!TransitionLocked(available[i], {LocalPageState::kCached}, LocalPageState::kReserved)) {
      return {};
    }
    reservation.load.push_back(available[i]->handle);
  }
  for (size_t i = 0; i < extend_pages; ++i) {
    LocalPage* page = available[load_pages + i];
    if (!TransitionLocked(page, {LocalPageState::kCached}, LocalPageState::kReserved)) return {};
    reservation.extend.push_back(page->handle);
  }
  out.success = true;
  out.id = next_reservation_id_++;
  reservations_.emplace(out.id, std::move(reservation));
  return out;
}

std::vector<AllocationHandle> EngineClient::Take(uint64_t reservation_id,
                                                 ReservationPurpose purpose, size_t count) {
  std::vector<AllocationHandle> out;
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!healthy_.load() || failed_.load() || quiescing_) return out;
  auto reservation = reservations_.find(reservation_id);
  if (reservation == reservations_.end()) return out;
  auto& source =
      purpose == ReservationPurpose::kLoad ? reservation->second.load : reservation->second.extend;
  if (source.size() < count) return out;
  auto candidate = source.begin();
  for (size_t i = 0; i < count; ++i, ++candidate) {
    auto page = pages_.find(candidate->page_id);
    if (page == pages_.end() || page->second.state != LocalPageState::kReserved ||
        !(page->second.handle == *candidate)) {
      return {};
    }
  }
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    AllocationHandle handle = source.front();
    source.pop_front();
    auto page = pages_.find(handle.page_id);
    ++page->second.handle.generation;
    if (!TransitionLocked(&page->second, {LocalPageState::kReserved}, LocalPageState::kInUse)) {
      return {};
    }
    out.push_back(page->second.handle);
  }
  return out;
}

bool EngineClient::Refund(uint64_t reservation_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto reservation = reservations_.find(reservation_id);
  if (reservation == reservations_.end()) return false;
  for (const auto* queue : {&reservation->second.load, &reservation->second.extend}) {
    for (const auto& handle : *queue) {
      auto page = pages_.find(handle.page_id);
      if (page != pages_.end() && page->second.state == LocalPageState::kReserved &&
          page->second.handle == handle) {
        (void)TransitionLocked(&page->second, {LocalPageState::kReserved}, LocalPageState::kCached);
      }
    }
  }
  reservations_.erase(reservation);
  return true;
}

bool EngineClient::ReturnPages(const std::vector<AllocationHandle>& handles) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (quiescing_) return false;
  std::unordered_set<uint32_t> seen;
  for (const auto& handle : handles) {
    if (!HandleMatchesLocked(handle) || !seen.insert(handle.page_id).second) return false;
    const auto state = pages_.at(handle.page_id).state;
    if (state == LocalPageState::kReserved) return false;
  }
  std::unordered_set<uint64_t> canceled_tasks;
  std::vector<AllocationHandle> cancel_quarantined;
  for (const auto& handle : handles) {
    LocalPage& page = pages_.at(handle.page_id);
    if (page.state == LocalPageState::kQuarantined) continue;
    page.return_requested = true;
    if (page.state == LocalPageState::kInUse && page.pending_publish != 0) {
      canceled_tasks.insert(page.pending_publish);
    }
  }
  for (uint64_t task_id : canceled_tasks) {
    auto task = publish_tasks_.find(task_id);
    if (task == publish_tasks_.end() || task->second->sent) continue;
    task->second->canceled = true;
    for (const auto& record : task->second->records) {
      auto page = pages_.find(record.handle.page_id);
      if (page == pages_.end() || page->second.pending_publish != task_id ||
          !(page->second.handle == record.handle)) {
        continue;
      }
      page->second.pending_publish = 0;
      if (seen.count(record.handle.page_id) != 0) {
        const CompletionProof proof = task->second->readiness.value_or(CompletionProof::kCompleted);
        if (!SettleUnpublishedLocked(&page->second, proof) && proof == CompletionProof::kUnknown &&
            page->second.state == LocalPageState::kQuarantined) {
          cancel_quarantined.push_back(record.handle);
        }
      }
    }
  }
  for (const auto& handle : handles) {
    LocalPage& page = pages_.at(handle.page_id);
    switch (page.state) {
      case LocalPageState::kCached:
        page.return_requested = false;
        break;
      case LocalPageState::kReserved:
        return false;  // validated above; keeps the switch exhaustive
      case LocalPageState::kInUse:
        if (page.pending_publish == 0) {
          (void)SettleUnpublishedLocked(&page, CompletionProof::kCompleted);
        }
        break;
      case LocalPageState::kPendingFree:
        break;
      case LocalPageState::kPublishSent:
        break;
      case LocalPageState::kPublished:
        (void)QueueRetireLocked(&page);
        break;
      case LocalPageState::kRetiring:
        break;
      case LocalPageState::kQuarantined:
        break;
    }
  }
  if (!cancel_quarantined.empty()) quarantine_queue_.push_back(std::move(cancel_quarantined));
  progress_cv_.notify_all();
  return true;
}

uint64_t EngineClient::QueuePublish(const std::vector<ResidentRecord>& records, bool ready) {
  return QueuePublishImpl(records, ready, /*retain_result=*/false);
}

uint64_t EngineClient::QueuePublishImpl(const std::vector<ResidentRecord>& records, bool ready,
                                        bool retain_result) {
  if (records.empty()) return 0;
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!healthy_.load() || failed_.load() || quiescing_ ||
      publish_receipts_.size() >= publish_receipt_limit_) {
    if (publish_receipts_.size() >= publish_receipt_limit_) event_overflowed_ = true;
    return 0;
  }
  std::unordered_set<uint32_t> pages;
  for (const auto& record : records) {
    if (!HandleMatchesLocked(record.handle) || !pages.insert(record.handle.page_id).second) {
      return 0;
    }
    const LocalPage& page = pages_.at(record.handle.page_id);
    if (page.state != LocalPageState::kInUse || page.pending_publish != 0) return 0;
  }
  auto task = std::make_shared<PublishTask>();
  task->id = next_publish_id_++;
  task->records = records;
  task->retain_result = retain_result;
  if (ready) task->readiness = CompletionProof::kCompleted;
  for (const auto& record : records) pages_.at(record.handle.page_id).pending_publish = task->id;
  publish_tasks_[task->id] = task;
  publish_queue_.push_back(task);
  RecordPublishReceiptLocked(task->id,
                             ready ? PublishReceiptState::kReady : PublishReceiptState::kQueued);
  progress_cv_.notify_one();
  return task->id;
}

bool EngineClient::MarkPublishReady(uint64_t ticket, CompletionProof proof) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto task = publish_tasks_.find(ticket);
  if (task == publish_tasks_.end() || task->second->sent || task->second->done ||
      task->second->canceled) {
    return false;
  }
  if (proof == CompletionProof::kUnknown) {
    std::vector<AllocationHandle> quarantined;
    for (const auto& record : task->second->records) {
      auto page = pages_.find(record.handle.page_id);
      if (page == pages_.end() || !(page->second.handle == record.handle) ||
          page->second.pending_publish != ticket) {
        continue;
      }
      page->second.pending_publish = 0;
      if (TransitionLocked(&page->second, {LocalPageState::kInUse, LocalPageState::kPendingFree},
                           LocalPageState::kQuarantined)) {
        quarantined.push_back(record.handle);
      }
    }
    task->second->readiness = proof;
    task->second->canceled = true;
    RecordPublishReceiptLocked(ticket, PublishReceiptState::kUnknown);
    FinishTaskLocked(task->second, {});
    publish_queue_.erase(std::remove(publish_queue_.begin(), publish_queue_.end(), task->second),
                         publish_queue_.end());
    if (!task->second->retain_result) publish_tasks_.erase(task);
    if (!quarantined.empty()) quarantine_queue_.push_back(std::move(quarantined));
    progress_cv_.notify_all();
    state_cv_.notify_all();
    return true;
  }
  task->second->readiness = proof;
  RecordPublishReceiptLocked(
      ticket, proof == CompletionProof::kCompleted ? PublishReceiptState::kReady
              : proof == CompletionProof::kUnknown ? PublishReceiptState::kUnknown
                                                   : PublishReceiptState::kCancelled);
  progress_cv_.notify_one();
  return true;
}

bool EngineClient::CancelPublish(uint64_t ticket) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto task = publish_tasks_.find(ticket);
  if (task == publish_tasks_.end() || task->second->sent || task->second->done) return false;
  task->second->canceled = true;
  RecordPublishReceiptLocked(ticket, PublishReceiptState::kCancelled);
  for (const auto& record : task->second->records) {
    auto page = pages_.find(record.handle.page_id);
    if (page != pages_.end() && page->second.pending_publish == ticket) {
      page->second.pending_publish = 0;
    }
  }
  progress_cv_.notify_one();
  return true;
}

std::optional<PublishReceipt> EngineClient::PublishStatus(uint64_t ticket) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto found = publish_receipts_.find(ticket);
  return found == publish_receipts_.end() ? std::nullopt
                                          : std::optional<PublishReceipt>{found->second};
}

std::vector<PublishReceipt> EngineClient::PollPublishReceipts() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  std::vector<PublishReceipt> out;
  out.reserve(publish_receipt_events_.size());
  while (!publish_receipt_events_.empty()) {
    out.push_back(std::move(publish_receipt_events_.front()));
    publish_receipt_events_.pop_front();
  }
  return out;
}

bool EngineClient::AcknowledgePublishReceipt(uint64_t ticket) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto receipt = publish_receipts_.find(ticket);
  if (receipt == publish_receipts_.end() || !IsTerminalPublishReceipt(receipt->second.state)) {
    return false;
  }
  publish_receipts_.erase(receipt);
  return true;
}

bool EngineClient::ConsumeEventOverflow() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  const bool overflowed = event_overflowed_;
  event_overflowed_ = false;
  return overflowed;
}

PublishResult EngineClient::PublishRpc(const std::vector<ResidentRecord>& records, bool* unknown) {
  PublishResult out;
  if (unknown) *unknown = false;
  wire::EngineFrame request;
  for (const auto& record : records) {
    auto* converted = request.mutable_publish()->add_records();
    converted->set_key(record.key);
    converted->set_parent_key(record.parent_key);
    ToWire(record.handle, converted->mutable_handle());
  }
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_published()) {
    if (unknown) *unknown = true;
    return out;
  }
  CloseAll(&fds);
  out.success = response.published().ok();
  out.accepted.assign(response.published().accepted().begin(),
                      response.published().accepted().end());
  return out;
}

PublishResult EngineClient::Publish(const std::vector<ResidentRecord>& records) {
  PublishResult out;
  const uint64_t ticket = QueuePublishImpl(records, true, /*retain_result=*/true);
  if (ticket == 0) return out;
  std::unique_lock<std::mutex> lock(state_mutex_);
  auto task = publish_tasks_.find(ticket);
  if (task == publish_tasks_.end()) return out;
  std::shared_ptr<PublishTask> pending = task->second;
  pending->done_cv.wait(lock, [&] { return pending->done || failed_.load() || closing_.load(); });
  out = pending->result;
  publish_tasks_.erase(ticket);
  publish_receipts_.erase(ticket);  // synchronous caller consumed the terminal result
  return out;
}

RetireTicket EngineClient::Retire(const std::vector<AllocationHandle>& handles) {
  RetireTicket out;
  std::vector<AllocationHandle> to_send;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const bool already_sent =
        !handles.empty() && std::all_of(handles.begin(), handles.end(), [&](const auto& handle) {
          auto page = pages_.find(handle.page_id);
          return page != pages_.end() && page->second.handle == handle &&
                 page->second.state == LocalPageState::kRetiring && page->second.retire_sent;
        });
    if (already_sent) {
      out.handles = handles;
      out.state = RetireState::kTransitionPending;
      return out;
    }
    for (const auto& handle : handles) {
      if (!HandleMatchesLocked(handle)) {
        out.state = RetireState::kQuarantined;
        return out;
      }
      const auto state = pages_.at(handle.page_id).state;
      if (state != LocalPageState::kPublished && state != LocalPageState::kRetiring) {
        out.state = RetireState::kQuarantined;
        return out;
      }
    }
    for (const auto& handle : handles) {
      auto& page = pages_.at(handle.page_id);
      if (page.retire_sent) continue;
      if (page.state == LocalPageState::kPublished) {
        (void)TransitionLocked(&page, {LocalPageState::kPublished}, LocalPageState::kRetiring);
      }
      page.retire_queued = false;
      page.retire_sent = true;
      to_send.push_back(handle);
    }
  }
#ifdef MORI_UMBP_TESTING
  std::function<void()> before_retire_send;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    before_retire_send = before_retire_send_hook_;
  }
  if (before_retire_send) before_retire_send();
#endif
  wire::EngineFrame request;
  for (const auto& handle : to_send) ToWire(handle, request.mutable_retire()->add_handles());
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_retired()) {
    out.state = RetireState::kQuarantined;
    return out;
  }
  CloseAll(&fds);
  FromWire(response.retired().retire_op(), &out.id);
  out.state = static_cast<RetireState>(response.retired().state());
  for (const auto& handle : response.retired().handles()) {
    AllocationHandle converted;
    if (!FromWire(handle, &converted)) {
      out.state = RetireState::kQuarantined;
      out.handles.clear();
      std::lock_guard<std::mutex> lock(call_mutex_);
      FailStopLocked();
      return out;
    }
    out.handles.push_back(std::move(converted));
  }
#ifdef MORI_UMBP_TESTING
  std::function<void()> after_retire_reply;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    after_retire_reply = after_retire_reply_hook_;
  }
  if (after_retire_reply) after_retire_reply();
#endif
  if (out.state == RetireState::kBusy) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto& handle : handles) {
      auto page = pages_.find(handle.page_id);
      if (page != pages_.end() && page->second.handle == handle) {
        page->second.retire_queued = false;
        page->second.retire_sent = false;
        (void)TransitionLocked(&page->second, {LocalPageState::kRetiring},
                               LocalPageState::kPublished);
        if (!quiescing_) (void)QueueRetireLocked(&page->second);
      }
    }
    return out;
  }
  if (out.state == RetireState::kQuarantined) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto& handle : out.handles) {
      auto page = pages_.find(handle.page_id);
      if (page != pages_.end() && page->second.handle == handle) {
        page->second.retire_queued = false;
        page->second.retire_sent = false;
        (void)TransitionLocked(&page->second, {LocalPageState::kRetiring},
                               LocalPageState::kQuarantined);
      }
    }
  }
  return out;
}

std::vector<RetireTicket> EngineClient::PollRetires() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  std::vector<RetireTicket> out;
  out.reserve(retire_events_.size());
  while (!retire_events_.empty()) {
    out.push_back(std::move(retire_events_.front()));
    retire_events_.pop_front();
  }
  return out;
}

bool EngineClient::AcknowledgeReleased(const std::vector<OperationId>& operations) {
  wire::EngineFrame request;
  for (const auto& operation : operations) {
    ToWire(operation, request.mutable_acknowledge_released()->add_retire_ops());
  }
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  const bool called = Call(&request, &response, &fds, &error) && response.has_acknowledge_reply() &&
                      response.acknowledge_reply().accepted_size() == operations.size();
  CloseAll(&fds);
  if (!called) return false;
  bool all = true;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (size_t i = 0; i < operations.size(); ++i) {
      const auto& operation = operations[i];
      const bool accepted = response.acknowledge_reply().accepted(static_cast<int>(i)) ||
                            released_operations_.count(operation) != 0;
      if (!accepted) {
        all = false;
        continue;
      }
      if (released_operations_.insert(operation).second) {
        released_operation_order_.push_back(operation);
        while (released_operation_order_.size() > 4096) {
          released_operations_.erase(released_operation_order_.front());
          released_operation_order_.pop_front();
        }
      }
      auto released = pending_released_.find(operation);
      if (released == pending_released_.end()) continue;
      for (auto& handle : released->second) {
        auto page = pages_.find(handle.page_id);
        if (page == pages_.end() || page->second.state != LocalPageState::kRetiring ||
            !(page->second.handle == handle)) {
          continue;
        }
        page->second.handle = handle;
        page->second.retire_queued = false;
        page->second.retire_sent = false;
        const LocalPageState target = page->second.io_holds == 0 && page->second.reuse_guards == 0
                                          ? LocalPageState::kCached
                                          : LocalPageState::kPendingFree;
        (void)TransitionLocked(&page->second, {LocalPageState::kRetiring}, target);
        page->second.return_requested = false;
      }
      pending_released_.erase(released);
    }
  }
  return all;
}

ResetLease EngineClient::QuiesceReset() {
  ResetLease out;
  {
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (quiescing_ || failed_.load() || !healthy_.load() || !local_holds_.empty()) return out;
    quiescing_ = true;
    // Seal only the send side.  Unsent tickets remain intact so a Busy reply
    // or explicit abort can resume them without changing tree ownership.
    progress_cv_.notify_all();
    if (!state_cv_.wait_for(lock, std::chrono::milliseconds(config_.heartbeat_timeout_ms), [&] {
          return (state_counts_[static_cast<size_t>(LocalPageState::kPublishSent)] == 0 &&
                  quarantine_queue_.empty() && quarantine_reports_in_flight_ == 0) ||
                 failed_.load();
        })) {
      quiescing_ = false;
      progress_cv_.notify_all();
      return out;
    }
  }
  wire::EngineFrame request;
  request.mutable_reset();
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_reset_reply()) return out;
  CloseAll(&fds);
  const auto& reply = response.reset_reply();
  if (!FromWire(reply.lease(), &out.id)) {
    std::lock_guard<std::mutex> call_lock(call_mutex_);
    FailStopLocked();
    return ResetLease{};
  }
  const auto abort_and_fail = [&] {
    if (reply.ready()) {
      wire::EngineFrame abort;
      ToWire(out.id, abort.mutable_reset_finish()->mutable_lease());
      abort.mutable_reset_finish()->set_commit(false);
      wire::EngineFrame ignored;
      std::vector<int> abort_fds;
      std::string abort_error;
      (void)Call(&abort, &ignored, &abort_fds, &abort_error);
      CloseAll(&abort_fds);
    }
    std::lock_guard<std::mutex> call_lock(call_mutex_);
    FailStopLocked();
  };
  if (!FromWire(reply.old_epoch(), &out.old_epoch) ||
      !FromWire(reply.new_epoch(), &out.new_epoch)) {
    abort_and_fail();
    return ResetLease{};
  }
  out.ready = reply.ready();
  for (const auto& wire_handle : reply.approved()) {
    AllocationHandle handle;
    if (!FromWire(wire_handle, &handle)) {
      abort_and_fail();
      return ResetLease{};
    }
    out.approved.push_back(std::move(handle));
  }
  if (!out.ready) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      quiescing_ = false;
      progress_cv_.notify_all();
    }
    RetryReturnedPublished();
  }
  return out;
}

bool EngineClient::FinishReset(ResetLease& lease, bool commit) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.heartbeat_timeout_ms);
  std::unique_lock<std::mutex> call_lock(call_mutex_, std::defer_lock);
  std::unique_lock<std::mutex> state_lock(state_mutex_);
  for (;;) {
    progress_cv_.notify_all();
    if (!state_cv_.wait_until(state_lock, deadline, [&] {
          return (quarantine_queue_.empty() && quarantine_reports_in_flight_ == 0) ||
                 failed_.load();
        })) {
      return false;
    }
    if (failed_.load()) return false;
    // Acquire the ordered channel without holding state_mutex_, then take the
    // state lock again and recheck. Once both locks are held, no new quarantine
    // can be enqueued between the final empty check and reset commit.
    state_lock.unlock();
    call_lock.lock();
    state_lock.lock();
    if (quarantine_queue_.empty() && quarantine_reports_in_flight_ == 0) break;
    call_lock.unlock();
  }
  wire::EngineFrame request;
  ToWire(lease.id, request.mutable_reset_finish()->mutable_lease());
  request.mutable_reset_finish()->set_commit(commit);
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  PoolEpoch resulting;
  std::vector<AllocationHandle> approved;
  if (!CallLocked(&request, &response, &fds, &error) || !response.has_reset_finished() ||
      !response.reset_finished().ok() || !FromWire(response.reset_finished().epoch(), &resulting)) {
    CloseAll(&fds);
    return false;
  }
  for (const auto& wire_handle : response.reset_finished().approved()) {
    AllocationHandle handle;
    if (!FromWire(wire_handle, &handle)) {
      CloseAll(&fds);
      state_lock.unlock();
      FailStopLocked();
      return false;
    }
    approved.push_back(std::move(handle));
  }
  if (commit) epoch_ = resulting;
  CloseAll(&fds);
  if (commit) {
    std::unordered_map<uint32_t, LocalPage> rebuilt;
    for (const auto& [page_id, page] : pages_) {
      if (page.state == LocalPageState::kQuarantined) rebuilt.emplace(page_id, page);
    }
    for (auto handle : approved) {
      if (rebuilt.count(handle.page_id) != 0) continue;  // local quarantine is authoritative
      auto old = pages_.find(handle.page_id);
      if (old != pages_.end())
        handle.generation = std::max(handle.generation, old->second.handle.generation);
      handle.epoch = resulting;
      rebuilt.emplace(handle.page_id, LocalPage{handle, LocalPageState::kCached});
    }
    pages_ = std::move(rebuilt);
    RebuildStateIndexLocked();
    reservations_.clear();
    pending_released_.clear();
    released_operations_.clear();
    released_operation_order_.clear();
    for (auto task = publish_tasks_.begin(); task != publish_tasks_.end();) {
      auto& pending = task->second;
      pending->canceled = true;
      RecordPublishReceiptLocked(task->first, PublishReceiptState::kCancelled);
      pending->done = true;
      pending->done_cv.notify_all();
      if (!pending->retain_result) {
        task = publish_tasks_.erase(task);
      } else {
        ++task;
      }
    }
    publish_queue_.clear();
    retire_queue_.clear();
    local_holds_.clear();
    load_target_pages_.clear();
  }
  quiescing_ = false;
  state_lock.unlock();
  call_lock.unlock();
  progress_cv_.notify_all();
  if (commit) {
    lease.new_epoch = resulting;
    lease.approved = approved;
  }
  if (!commit) RetryReturnedPublished();
  return true;
}

std::optional<OperationId> EngineClient::HoldIO(const std::vector<AllocationHandle>& handles) {
  if (handles.empty()) return std::nullopt;
  uint64_t id = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (failed_.load() || quiescing_) return std::nullopt;
    for (const auto& handle : handles) {
      if (!HandleMatchesLocked(handle)) return std::nullopt;
      const LocalPage& page = pages_.at(handle.page_id);
      if (page.state != LocalPageState::kPublished && page.state != LocalPageState::kReserved &&
          page.state != LocalPageState::kInUse && page.state != LocalPageState::kPendingFree &&
          page.state != LocalPageState::kPublishSent) {
        return std::nullopt;
      }
    }
    id = next_local_hold_id_++;
    LocalHold local;
    local.handles = handles;
    for (const auto& handle : handles) ++pages_.at(handle.page_id).io_holds;
    local_holds_.emplace(id, std::move(local));
  }
  return OperationId{session_id_, (uint64_t{1} << 63) | id};
}

std::optional<OperationId> EngineClient::HoldLoadTargets(
    const std::vector<AllocationHandle>& handles) {
  if (handles.empty()) return std::nullopt;
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (failed_.load() || quiescing_) return std::nullopt;
  std::unordered_set<uint32_t> seen;
  for (const auto& handle : handles) {
    if (!HandleMatchesLocked(handle) || !seen.insert(handle.page_id).second ||
        load_target_pages_.count(handle.page_id) != 0 ||
        pages_.at(handle.page_id).state != LocalPageState::kInUse) {
      return std::nullopt;
    }
  }
  const uint64_t id = next_local_hold_id_++;
  LocalHold hold;
  hold.handles = handles;
  hold.load_target = true;
  for (const auto& handle : handles) {
    ++pages_.at(handle.page_id).io_holds;
    load_target_pages_.insert(handle.page_id);
  }
  local_holds_.emplace(id, std::move(hold));
  return OperationId{session_id_, (uint64_t{1} << 63) | id};
}

bool EngineClient::CompleteIO(OperationId hold, CompletionProof proof) {
  if (hold.session_id != session_id_ || (hold.seq & (uint64_t{1} << 63)) == 0) return false;
  const uint64_t id = hold.seq & ~(uint64_t{1} << 63);
  return CompleteIOInternal(id, proof, /*deferred=*/false);
}

bool EngineClient::CompleteIOInternal(uint64_t id, CompletionProof proof, bool deferred) {
  std::vector<AllocationHandle> quarantined;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    auto local = local_holds_.find(id);
    if (local == local_holds_.end()) return deferred;
    if (local->second.reuse_guard) return false;
    for (const auto& handle : local->second.handles) {
      auto page = pages_.find(handle.page_id);
      if (page == pages_.end() || page->second.io_holds == 0 || !(page->second.handle == handle)) {
        continue;
      }
      --page->second.io_holds;
      if (proof == CompletionProof::kUnknown) {
        if (TransitionLocked(&page->second,
                             {LocalPageState::kReserved, LocalPageState::kInUse,
                              LocalPageState::kPendingFree, LocalPageState::kPublishSent,
                              LocalPageState::kPublished, LocalPageState::kRetiring},
                             LocalPageState::kQuarantined)) {
          quarantined.push_back(handle);
        }
      } else if (page->second.state == LocalPageState::kPendingFree) {
        (void)SettleUnpublishedLocked(&page->second, proof);
      } else if (page->second.state == LocalPageState::kPublished &&
                 page->second.return_requested) {
        (void)QueueRetireLocked(&page->second);
      }
    }
    if (local->second.load_target) {
      for (const auto& handle : local->second.handles) load_target_pages_.erase(handle.page_id);
    }
    local_holds_.erase(local);
    if (!quarantined.empty()) quarantine_queue_.push_back(quarantined);
    progress_cv_.notify_all();
  }
  return true;
}

std::optional<LocalHoldId> EngineClient::HoldReuseGuard(
    const std::vector<AllocationHandle>& handles) {
  if (handles.empty()) return std::nullopt;
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (failed_.load() || quiescing_) return std::nullopt;
  for (const auto& handle : handles) {
    if (!HandleMatchesLocked(handle)) return std::nullopt;
    const auto state = pages_.at(handle.page_id).state;
    if (state == LocalPageState::kCached || state == LocalPageState::kQuarantined) {
      return std::nullopt;
    }
  }
  const uint64_t id = next_local_hold_id_++;
  LocalHold hold;
  hold.handles = handles;
  hold.reuse_guard = true;
  for (const auto& handle : handles) ++pages_.at(handle.page_id).reuse_guards;
  local_holds_.emplace(id, std::move(hold));
  return LocalHoldId{id};
}

bool EngineClient::CompleteReuseGuard(LocalHoldId hold, CompletionProof proof) {
  std::vector<AllocationHandle> quarantined;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto local = local_holds_.find(hold.value);
    if (local == local_holds_.end() || !local->second.reuse_guard) return false;
    for (const auto& handle : local->second.handles) {
      auto page = pages_.find(handle.page_id);
      if (page == pages_.end() || page->second.reuse_guards == 0 ||
          !(page->second.handle == handle)) {
        continue;
      }
      --page->second.reuse_guards;
      if (proof == CompletionProof::kUnknown) {
        if (TransitionLocked(&page->second,
                             {LocalPageState::kReserved, LocalPageState::kInUse,
                              LocalPageState::kPendingFree, LocalPageState::kPublishSent,
                              LocalPageState::kPublished, LocalPageState::kRetiring},
                             LocalPageState::kQuarantined)) {
          quarantined.push_back(handle);
        }
      } else if (page->second.state == LocalPageState::kPendingFree) {
        (void)SettleUnpublishedLocked(&page->second, proof);
      } else if (page->second.state == LocalPageState::kPublished &&
                 page->second.return_requested) {
        (void)QueueRetireLocked(&page->second);
      }
    }
    local_holds_.erase(local);
    if (!quarantined.empty()) quarantine_queue_.push_back(quarantined);
    progress_cv_.notify_all();
  }
  return true;
}

ReadPlan EngineClient::Lookup(const std::vector<std::string>& keys, std::string name_space) {
  ReadPlan out;
  if (keys.empty() || name_space.empty() || layout_id_.empty()) {
    out.error = "invalid lookup request";
    return out;
  }
  wire::EngineFrame request;
  auto* lookup = request.mutable_lookup();
  for (const auto& key : keys) lookup->add_keys(key);
  lookup->set_storage_namespace(std::move(name_space));
  lookup->set_requester_backend_id(backend_id_);
  lookup->set_layout_id(layout_id_);
  // The registered pool's layout is immutable for the session. The node also
  // validates it against requester_backend_id.
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_read_plan() ||
      !ReadPlanFromWire(response.read_plan(), &out)) {
    CloseAll(&fds);
    out.error = error.empty() ? "lookup failed" : error;
    return out;
  }
  CloseAll(&fds);
  return out;
}

PreparedLoad EngineClient::PrepareLoad(const ReadPlan& plan, std::string request_id,
                                       const std::vector<AllocationHandle>& targets,
                                       uint32_t n_pages) {
  PreparedLoad out;
  if (!plan.success || plan.id == 0 || request_id.empty() || n_pages == 0 ||
      targets.size() != n_pages || n_pages > plan.valid_prefix) {
    out.error = "invalid prepare_load request";
    return out;
  }
  auto target_hold = HoldLoadTargets(targets);
  if (!target_hold.has_value()) {
    out.error = "could not hold load targets";
    return out;
  }
  wire::EngineFrame request;
  auto* prepare = request.mutable_prepare_load();
  prepare->set_plan_id(plan.id);
  prepare->set_request_id(request_id);
  prepare->set_n_pages(n_pages);
  for (const auto& target : targets) ToWire(target, prepare->add_targets());
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_prepared() ||
      !PreparedLoadFromWire(response.prepared(), fds, &out) || !out.success) {
    CloseAll(&fds);
    (void)CompleteIO(*target_hold, CompletionProof::kNeverSubmitted);
    if (out.error.empty()) out.error = error.empty() ? "prepare_load rejected" : error;
    return out;
  }
  EngineExecutor* executor = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    executor = executor_;
  }
  const auto import_regions = [&](const std::vector<PeerVmmRegion>& regions) {
    return executor != nullptr &&
           std::all_of(regions.begin(), regions.end(), [&](const auto& region) {
             return executor->ImportPeerRegion(region, region.fd);
           });
  };
  if (!import_regions(out.peer_regions)) {
    CloseAll(&fds);
    PreparedLoad cancel = out;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      prepared_loads_.emplace(out.id, LocalPreparedLoad{target_hold, targets});
    }
    (void)CancelLoad(cancel);
    {
      std::lock_guard<std::mutex> call_lock(call_mutex_);
      FailStopLocked();
    }
    out.success = false;
    out.error = "failed to import peer VMM region; engine session stopped";
    return out;
  }
  CloseAll(&fds);
  bool more_regions = response.prepared().peer_regions_more();
  while (more_regions) {
    wire::EngineFrame fetch;
    fetch.mutable_fetch_peer_regions()->set_prepared_id(out.id);
    wire::EngineFrame batch_response;
    std::vector<int> batch_fds;
    std::string batch_error;
    if (!Call(&fetch, &batch_response, &batch_fds, &batch_error) ||
        !batch_response.has_peer_region_batch() || !batch_response.peer_region_batch().ok()) {
      CloseAll(&batch_fds);
      PreparedLoad cancel = out;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        prepared_loads_.emplace(out.id, LocalPreparedLoad{target_hold, targets});
      }
      (void)CancelLoad(cancel);
      out.success = false;
      out.error = batch_error.empty() ? "failed to fetch peer VMM region batch" : batch_error;
      return out;
    }
    std::vector<PeerVmmRegion> batch;
    if (!AppendPeerRegionsFromWire(batch_response.peer_region_batch().peer_regions(), batch_fds,
                                   &batch) ||
        !import_regions(batch)) {
      CloseAll(&batch_fds);
      PreparedLoad cancel = out;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        prepared_loads_.emplace(out.id, LocalPreparedLoad{target_hold, targets});
      }
      (void)CancelLoad(cancel);
      {
        std::lock_guard<std::mutex> call_lock(call_mutex_);
        FailStopLocked();
      }
      out.success = false;
      out.error = "failed to import peer VMM region batch; engine session stopped";
      return out;
    }
    out.peer_regions.insert(out.peer_regions.end(), std::make_move_iterator(batch.begin()),
                            std::make_move_iterator(batch.end()));
    more_regions = batch_response.peer_region_batch().more();
    CloseAll(&batch_fds);
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    prepared_loads_.emplace(out.id, LocalPreparedLoad{target_hold, targets});
  }
  return out;
}

CancelLoadResult EngineClient::CancelLoad(const PreparedLoad& prepared) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto local = prepared_loads_.find(prepared.id);
    if (local == prepared_loads_.end()) return CancelLoadResult::kRejected;
    if (local->second.terminal) return CancelLoadResult::kCompleted;
    if (local->second.started) return CancelLoadResult::kInFlight;
  }
  wire::EngineFrame request;
  request.mutable_cancel_load()->set_prepared_id(prepared.id);
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_cancelled()) {
    CloseAll(&fds);
    return CancelLoadResult::kRejected;
  }
  CloseAll(&fds);
  const auto result = static_cast<CancelLoadResult>(response.cancelled().result());
  if (result != CancelLoadResult::kNeverSubmitted) return result;
  std::optional<OperationId> target_hold;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto local = prepared_loads_.find(prepared.id);
    if (local == prepared_loads_.end()) return CancelLoadResult::kCompleted;
    // The node's NeverSubmitted result is authoritative.  MarkLoadStarted may
    // have raced locally, but its ordered RPC can only be rejected after this
    // cancel removed the node record; retaining the target hold here leaks it.
    local->second.terminal = true;
    target_hold = local->second.target_hold;
    prepared_loads_.erase(local);
  }
  if (target_hold.has_value()) (void)CompleteIO(*target_hold, CompletionProof::kNeverSubmitted);
  return result;
}

bool EngineClient::MarkLoadStarted(uint64_t prepared_id) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto prepared = prepared_loads_.find(prepared_id);
    if (prepared == prepared_loads_.end() || prepared->second.started ||
        prepared->second.terminal) {
      return false;
    }
    prepared->second.started = true;
  }
  wire::EngineFrame request;
  auto* complete = request.mutable_complete_load();
  complete->set_prepared_id(prepared_id);
  complete->set_outcome(static_cast<uint32_t>(LoadOutcome::kPending));
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  const bool ok = Call(&request, &response, &fds, &error) && response.has_load_completed() &&
                  response.load_completed().ok();
  CloseAll(&fds);
  if (!ok) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto prepared = prepared_loads_.find(prepared_id);
    if (prepared != prepared_loads_.end()) prepared->second.started = false;
  }
  return ok;
}

bool EngineClient::FinishLoad(uint64_t prepared_id, LoadOutcome outcome) {
  if (outcome == LoadOutcome::kPending) return false;
  std::optional<OperationId> target_hold;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto prepared = prepared_loads_.find(prepared_id);
    if (prepared == prepared_loads_.end() || prepared->second.terminal) return false;
    prepared->second.terminal = true;
    target_hold = prepared->second.target_hold;
  }
  wire::EngineFrame request;
  auto* complete = request.mutable_complete_load();
  complete->set_prepared_id(prepared_id);
  complete->set_outcome(static_cast<uint32_t>(outcome));
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  const bool node_ok = Call(&request, &response, &fds, &error) && response.has_load_completed() &&
                       response.load_completed().ok();
  CloseAll(&fds);
  const CompletionProof proof = outcome == LoadOutcome::kUnknown ? CompletionProof::kUnknown
                                : outcome == LoadOutcome::kCanceled
                                    ? CompletionProof::kNeverSubmitted
                                    : CompletionProof::kCompleted;
  const bool local_ok = !target_hold.has_value() || CompleteIO(*target_hold, proof);
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    prepared_loads_.erase(prepared_id);
    state_cv_.notify_all();
  }
  return node_ok && local_ok;
}

EnginePageAccounting EngineClient::Accounting() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  EnginePageAccounting out;
  out.total = pages_.size();
  out.cached = state_counts_[static_cast<size_t>(LocalPageState::kCached)];
  out.reserved = state_counts_[static_cast<size_t>(LocalPageState::kReserved)];
  out.in_use = state_counts_[static_cast<size_t>(LocalPageState::kInUse)];
  out.pending_free = state_counts_[static_cast<size_t>(LocalPageState::kPendingFree)];
  out.publish_sent = state_counts_[static_cast<size_t>(LocalPageState::kPublishSent)];
  out.published = state_counts_[static_cast<size_t>(LocalPageState::kPublished)];
  out.retiring = state_counts_[static_cast<size_t>(LocalPageState::kRetiring)];
  out.quarantined = state_counts_[static_cast<size_t>(LocalPageState::kQuarantined)];
  out.available = healthy_.load() && !failed_.load() && !quiescing_ ? out.cached : 0;
  return out;
}

bool EngineClient::HandleMatchesLocked(const AllocationHandle& handle) const {
  auto page = pages_.find(handle.page_id);
  return page != pages_.end() && page->second.handle == handle;
}

bool EngineClient::TransitionLocked(LocalPage* page,
                                    std::initializer_list<LocalPageState> allowed_from,
                                    LocalPageState to) {
  if (page == nullptr ||
      std::find(allowed_from.begin(), allowed_from.end(), page->state) == allowed_from.end()) {
    return false;
  }
  if (page->state == to) return true;
  const size_t from_index = static_cast<size_t>(page->state);
  const size_t to_index = static_cast<size_t>(to);
  if (from_index >= state_counts_.size() || to_index >= state_counts_.size() ||
      state_counts_[from_index] == 0) {
    return false;
  }
  --state_counts_[from_index];
  ++state_counts_[to_index];
  page->state = to;
  if (to == LocalPageState::kCached) cached_pages_.push_back(page->handle.page_id);
  state_cv_.notify_all();
  return true;
}

void EngineClient::RebuildStateIndexLocked() {
  state_counts_.fill(0);
  cached_pages_.clear();
  for (auto& [page_id, page] : pages_) {
    ++state_counts_[static_cast<size_t>(page.state)];
    if (page.state == LocalPageState::kCached) cached_pages_.push_back(page_id);
  }
  state_cv_.notify_all();
}

void EngineClient::AddDelegatedLocked(const std::vector<AllocationHandle>& handles) {
  for (const auto& handle : handles) {
    auto [page, inserted] = pages_.try_emplace(handle.page_id);
    if (!inserted) continue;
    page->second = LocalPage{handle, LocalPageState::kCached};
    ++state_counts_[static_cast<size_t>(LocalPageState::kCached)];
    cached_pages_.push_back(handle.page_id);
  }
}

bool EngineClient::SettleUnpublishedLocked(LocalPage* page, CompletionProof proof) {
  if (page == nullptr) return false;
  if (proof == CompletionProof::kUnknown) {
    (void)TransitionLocked(page,
                           {LocalPageState::kReserved, LocalPageState::kInUse,
                            LocalPageState::kPendingFree, LocalPageState::kPublishSent},
                           LocalPageState::kQuarantined);
    return false;
  }
  const LocalPageState target = page->io_holds == 0 && page->reuse_guards == 0
                                    ? LocalPageState::kCached
                                    : LocalPageState::kPendingFree;
  if (!TransitionLocked(
          page,
          {LocalPageState::kInUse, LocalPageState::kPendingFree, LocalPageState::kPublishSent},
          target)) {
    return false;
  }
  page->return_requested = false;
  return target == LocalPageState::kCached;
}

void EngineClient::FinishTaskLocked(const std::shared_ptr<PublishTask>& task,
                                    PublishResult result) {
  if (task == nullptr || task->done) return;
  task->result = std::move(result);
  task->done = true;
  task->done_cv.notify_all();
}

void EngineClient::RecordPublishReceiptLocked(uint64_t ticket, PublishReceiptState state,
                                              std::vector<bool> accepted) {
  PublishReceipt receipt{ticket, state, std::move(accepted)};
  auto current = publish_receipts_.find(ticket);
  if (current != publish_receipts_.end() && current->second.state == state &&
      current->second.accepted == receipt.accepted) {
    return;
  }
  if (current == publish_receipts_.end() && publish_receipts_.size() >= publish_receipt_limit_) {
    event_overflowed_ = true;
    return;
  }
  publish_receipts_[ticket] = receipt;
  publish_receipt_events_.push_back(std::move(receipt));
  while (publish_receipt_events_.size() > event_queue_limit_) {
    publish_receipt_events_.pop_front();
    event_overflowed_ = true;
  }
}

bool EngineClient::QueueRetireLocked(LocalPage* page) {
  if (page == nullptr || quiescing_ || page->state != LocalPageState::kPublished ||
      !page->return_requested || page->io_holds != 0 || page->reuse_guards != 0 ||
      page->retire_queued || page->retire_sent) {
    return false;
  }
  if (!TransitionLocked(page, {LocalPageState::kPublished}, LocalPageState::kRetiring)) {
    return false;
  }
  page->retire_queued = true;
  retire_queue_.push_back(page->handle);
  progress_cv_.notify_one();
  return true;
}

void EngineClient::FinishAllPublishTasksLocked() {
  for (auto& [id, task] : publish_tasks_) {
    if (!task->done) {
      RecordPublishReceiptLocked(
          id, task->sent ? PublishReceiptState::kUnknown : PublishReceiptState::kCancelled);
      FinishTaskLocked(task, {});
    }
  }
  state_cv_.notify_all();
}

std::vector<AllocationHandle> EngineClient::ReturnedPublishedLocked() const {
  std::vector<AllocationHandle> out;
  for (const auto& [page_id, page] : pages_) {
    (void)page_id;
    if (page.state == LocalPageState::kPublished && page.return_requested) {
      out.push_back(page.handle);
    }
  }
  return out;
}

void EngineClient::RetryReturnedPublished() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (quiescing_ || failed_.load() || closing_.load()) return;
  for (auto& [page_id, page] : pages_) {
    (void)page_id;
    (void)QueueRetireLocked(&page);
  }
}

void EngineClient::ProcessRetireProgress() {
  wire::EngineFrame request;
  request.mutable_poll_retires();
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  if (!Call(&request, &response, &fds, &error) || !response.has_released()) {
    CloseAll(&fds);
    return;
  }
  CloseAll(&fds);

  std::vector<RetireTicket> tickets;
  std::vector<AllocationHandle> notices;
  bool valid = true;
  for (const auto& ticket : response.released().tickets()) {
    RetireTicket converted;
    valid = valid && FromWire(ticket.retire_op(), &converted.id);
    converted.state = static_cast<RetireState>(ticket.state());
    for (const auto& handle : ticket.handles()) {
      AllocationHandle item;
      valid = valid && FromWire(handle, &item);
      converted.handles.push_back(std::move(item));
    }
    tickets.push_back(std::move(converted));
  }
  for (const auto& handle : response.released().quarantined()) {
    AllocationHandle converted;
    valid = valid && FromWire(handle, &converted);
    notices.push_back(std::move(converted));
  }
  if (!valid) {
    std::lock_guard<std::mutex> call_lock(call_mutex_);
    FailStopLocked();
    return;
  }

  std::vector<OperationId> ready;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto quarantine_local = [&](const AllocationHandle& handle) {
      auto page = pages_.find(handle.page_id);
      if (page == pages_.end() || !(page->second.handle == handle)) return false;
      page->second.retire_queued = false;
      page->second.retire_sent = false;
      return TransitionLocked(
          &page->second,
          {LocalPageState::kReserved, LocalPageState::kInUse, LocalPageState::kPendingFree,
           LocalPageState::kPublishSent, LocalPageState::kPublished, LocalPageState::kRetiring},
          LocalPageState::kQuarantined);
    };
    const auto push_retire_event = [&](RetireTicket event) {
      retire_events_.push_back(std::move(event));
      while (retire_events_.size() > event_queue_limit_) {
        retire_events_.pop_front();
        event_overflowed_ = true;
      }
    };
    for (const auto& handle : notices) {
      if (quarantine_local(handle)) {
        push_retire_event(RetireTicket{OperationId{}, {handle}, RetireState::kQuarantined});
      }
    }
    for (auto& ticket : tickets) {
      if (ticket.state == RetireState::kQuarantined) {
        auto pending = pending_released_.find(ticket.id);
        if (pending != pending_released_.end()) {
          for (const auto& quarantined : ticket.handles) {
            pending->second.erase(
                std::remove(pending->second.begin(), pending->second.end(), quarantined),
                pending->second.end());
          }
          if (pending->second.empty()) pending_released_.erase(pending);
        }
        for (const auto& handle : ticket.handles) (void)quarantine_local(handle);
      } else if (ticket.state == RetireState::kReady) {
        pending_released_[ticket.id] = ticket.handles;
        ready.push_back(ticket.id);
      }
      push_retire_event(ticket);
    }
  }
  if (!ready.empty()) (void)AcknowledgeReleased(ready);
}

bool EngineClient::ReportQuarantine(const std::vector<AllocationHandle>& handles) {
  if (handles.empty()) return true;
  wire::EngineFrame request;
  for (const auto& handle : handles) ToWire(handle, request.mutable_quarantine()->add_handles());
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  const bool ok =
      Call(&request, &response, &fds, &error) && response.has_status() && response.status().ok();
  CloseAll(&fds);
  if (!ok) {
    std::lock_guard<std::mutex> call_lock(call_mutex_);
    FailStopLocked();
  }
  return ok;
}

bool EngineClient::Heartbeat() {
  wire::EngineFrame request;
  request.mutable_heartbeat()->set_monotonic_seq(last_received_seq_.load() + 1);
  request.mutable_heartbeat()->set_last_received_seq(last_received_seq_.load());
  request.mutable_heartbeat()->set_last_applied_seq(last_received_seq_.load());
  wire::EngineFrame response;
  std::vector<int> fds;
  std::string error;
  const bool ok = Call(&request, &response, &fds, &error) && response.has_heartbeat();
  CloseAll(&fds);
  return ok;
}

bool EngineClient::ExecuteOneWork(wire::EngineFrame response) {
  std::vector<wire::EngineFrame> batch;
  batch.push_back(std::move(response));
  return ExecuteWorkBatch(std::move(batch));
}

bool EngineClient::ExecuteWorkBatch(std::vector<wire::EngineFrame> responses) {
  if (responses.empty()) return true;
  std::vector<TransferPlan> plans;
  std::unordered_map<size_t, std::pair<size_t, size_t>> tag_origins;
  size_t next_tag = 1;
  EngineExecutor* executor = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    executor = executor_;
  }
  bool valid = executor != nullptr;
  for (size_t work = 0; work < responses.size(); ++work) {
    for (const auto& plan : responses[work].execute_plans().plans()) {
      TransferPlan converted;
      if (!PlanFromWire(plan, &converted)) {
        valid = false;
        continue;
      }
      if (converted.tags.empty()) converted.tags.push_back(0);
      for (size_t& tag : converted.tags) {
        const size_t original = tag;
        tag = next_tag;
        tag_origins.emplace(next_tag++, std::make_pair(work, original));
      }
      plans.push_back(std::move(converted));
    }
  }
  std::vector<TransferFailure> failures;
  const bool executed = valid && executor->ExecutePlans(plans, &failures);
  if (!executed && failures.empty()) {
    std::vector<size_t> tags;
    for (const auto& [tag, origin] : tag_origins) {
      (void)origin;
      tags.push_back(tag);
    }
    failures.push_back(
        {std::move(tags),
         1,
         valid ? "executor failed without completion details" : "rank executor is unavailable",
         {},
         /*quiesced=*/!valid,
         /*submitted=*/valid});
  }
  std::vector<std::vector<TransferFailure>> per_work(responses.size());
  for (const auto& failure : failures) {
    std::unordered_map<size_t, std::vector<size_t>> tags_by_work;
    for (size_t tag : failure.tags) {
      auto origin = tag_origins.find(tag);
      if (origin != tag_origins.end()) {
        tags_by_work[origin->second.first].push_back(origin->second.second);
      }
    }
    if (tags_by_work.empty()) {
      for (size_t work = 0; work < responses.size(); ++work) tags_by_work[work] = {};
    }
    for (auto& [work, tags] : tags_by_work) {
      TransferFailure split = failure;
      split.tags = std::move(tags);
      per_work[work].push_back(std::move(split));
    }
  }

  for (size_t work = 0; work < responses.size(); ++work) {
    DispatchOutcome outcome =
        valid ? DispatchOutcome::kCompleted : DispatchOutcome::kNeverSubmitted;
    if (std::any_of(per_work[work].begin(), per_work[work].end(),
                    [](const auto& failure) { return !failure.quiesced; })) {
      outcome = DispatchOutcome::kUnknown;
    } else if (std::any_of(per_work[work].begin(), per_work[work].end(),
                           [](const auto& failure) { return !failure.submitted; })) {
      outcome = DispatchOutcome::kNeverSubmitted;
    }
    wire::EngineFrame completion;
    auto* body = completion.mutable_complete_transfer();
    body->set_transfer_id(responses[work].execute_plans().transfer_id());
    body->set_quiesced(outcome != DispatchOutcome::kUnknown);
    body->set_outcome(static_cast<uint32_t>(outcome));
    for (const auto& failure : per_work[work]) {
      auto* out = body->add_failures();
      out->set_code(failure.code);
      out->set_message(failure.message);
      out->set_quiesced(failure.quiesced);
      out->set_submitted(failure.submitted);
      for (size_t tag : failure.tags) out->add_tags(tag);
    }
    wire::EngineFrame acknowledged;
    std::vector<int> fds;
    std::string error;
    if (!Call(&completion, &acknowledged, &fds, &error)) return false;
    CloseAll(&fds);
    if (!acknowledged.has_status() || !acknowledged.status().ok()) return false;
  }
  return true;
}

void EngineClient::ReceiverLoop() {
  while (!closing_.load() && !failed_.load()) {
    std::string payload;
    std::vector<int> fds;
    std::string error;
    if (!standalone::RecvFramedMessage(socket_, &payload, &fds, 16 * 1024 * 1024, 256, &error)) {
      CloseAll(&fds);
      break;
    }
    wire::EngineFrame frame;
    if (!frame.ParseFromString(payload) || frame.version() != 1) {
      CloseAll(&fds);
      break;
    }
    if (frame.server_push()) {
      CloseAll(&fds);
      if (!frame.has_execute_plans()) break;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        work_queue_.push_back(std::move(frame));
      }
      progress_cv_.notify_one();
      continue;
    }
    OperationId operation;
    if (!FromWire(frame.op(), &operation)) {
      CloseAll(&fds);
      break;
    }
    {
      std::lock_guard<std::mutex> lock(response_mutex_);
      if (expected_response_seq_ == 0 || operation.seq != expected_response_seq_ ||
          received_response_.has_value()) {
        CloseAll(&fds);
        break;
      }
      received_response_ = std::move(frame);
      received_response_fds_ = std::move(fds);
    }
    response_cv_.notify_one();
  }
  if (!closing_.load()) {
    healthy_.store(false);
    failed_.store(true);
    const int fd = socket_.load();
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
    if (call_mutex_.try_lock()) {
      const int owned = socket_.exchange(-1);
      if (owned >= 0) close(owned);
      call_mutex_.unlock();
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      FinishAllPublishTasksLocked();
    }
  }
  response_cv_.notify_all();
  progress_cv_.notify_all();
}

void EngineClient::HeartbeatLoop() {
  const auto interval = std::chrono::milliseconds(std::max<uint32_t>(config_.heartbeat_ms, 1));
  while (!closing_.load()) {
    std::this_thread::sleep_for(interval);
    if (closing_.load()) break;
    if (!Heartbeat()) break;
  }
  if (!closing_.load()) healthy_.store(false);
}

void EngineClient::ProgressLoop() {
  while (!closing_.load() && !failed_.load()) {
    std::shared_ptr<PublishTask> task;
    std::vector<AllocationHandle> retire_batch;
    std::vector<AllocationHandle> quarantine_batch;
    std::vector<AllocationHandle> pre_send_quarantine;
    std::vector<uint64_t> expired_prepared;
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      progress_cv_.wait_for(lock, std::chrono::milliseconds(5), [&] {
        return closing_.load() || failed_.load() || !work_queue_.empty() ||
               (!quarantine_queue_.empty()) || (!quiescing_ && !retire_queue_.empty()) ||
               (!publish_queue_.empty() &&
                (publish_queue_.front()->canceled ||
                 (!quiescing_ && publish_queue_.front()->readiness.has_value())));
      });
      if (closing_.load() || failed_.load()) break;
      if (!publish_queue_.empty() &&
          (publish_queue_.front()->canceled ||
           (!quiescing_ && publish_queue_.front()->readiness.has_value()))) {
        task = publish_queue_.front();
        publish_queue_.pop_front();
        if (task->canceled) {
          task->canceled = true;
          for (const auto& record : task->records) {
            auto page = pages_.find(record.handle.page_id);
            if (page != pages_.end() && page->second.handle == record.handle &&
                page->second.pending_publish == task->id) {
              page->second.pending_publish = 0;
            }
          }
          FinishTaskLocked(task, {});
          RecordPublishReceiptLocked(task->id, PublishReceiptState::kCancelled);
          if (!task->retain_result) publish_tasks_.erase(task->id);
          task.reset();
        } else if (*task->readiness != CompletionProof::kCompleted) {
          for (const auto& record : task->records) {
            auto page = pages_.find(record.handle.page_id);
            if (page == pages_.end() || !(page->second.handle == record.handle) ||
                page->second.pending_publish != task->id) {
              continue;
            }
            page->second.pending_publish = 0;
            if (*task->readiness == CompletionProof::kUnknown &&
                TransitionLocked(&page->second, {LocalPageState::kInUse},
                                 LocalPageState::kQuarantined)) {
              pre_send_quarantine.push_back(record.handle);
            }
          }
          FinishTaskLocked(task, {});
          if (!task->retain_result) publish_tasks_.erase(task->id);
          task.reset();
        } else {
          bool valid = true;
          for (const auto& record : task->records) {
            auto page = pages_.find(record.handle.page_id);
            if (page == pages_.end() || page->second.state != LocalPageState::kInUse ||
                page->second.pending_publish != task->id ||
                !(page->second.handle == record.handle)) {
              valid = false;
              break;
            }
          }
          if (!valid) {
            task->canceled = true;
            FinishTaskLocked(task, {});
            RecordPublishReceiptLocked(task->id, PublishReceiptState::kCancelled);
            if (!task->retain_result) publish_tasks_.erase(task->id);
            task.reset();
          } else {
            for (const auto& record : task->records) {
              LocalPage& page = pages_.at(record.handle.page_id);
              page.pending_publish = 0;
              (void)TransitionLocked(&page, {LocalPageState::kInUse}, LocalPageState::kPublishSent);
            }
            task->sent = true;
            RecordPublishReceiptLocked(task->id, PublishReceiptState::kSent);
          }
        }
      }
      if (!quiescing_) {
        while (!retire_queue_.empty()) {
          retire_batch.push_back(std::move(retire_queue_.front()));
          retire_queue_.pop_front();
        }
      }
      while (!quarantine_queue_.empty()) {
        auto queued = std::move(quarantine_queue_.front());
        quarantine_queue_.pop_front();
        quarantine_batch.insert(quarantine_batch.end(), queued.begin(), queued.end());
      }
      quarantine_batch.insert(quarantine_batch.end(), pre_send_quarantine.begin(),
                              pre_send_quarantine.end());
      pre_send_quarantine.clear();
      if (!quarantine_batch.empty()) ++quarantine_reports_in_flight_;
      const auto now = std::chrono::steady_clock::now();
      for (const auto& [id, prepared] : prepared_loads_) {
        if (!prepared.started && !prepared.terminal &&
            now - prepared.created_at >= std::chrono::seconds(5)) {
          expired_prepared.push_back(id);
        }
      }
    }

    if (!quarantine_batch.empty()) {
      (void)ReportQuarantine(quarantine_batch);
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (quarantine_reports_in_flight_ > 0) --quarantine_reports_in_flight_;
      state_cv_.notify_all();
    }
    if (!retire_batch.empty()) (void)Retire(retire_batch);
    for (uint64_t id : expired_prepared) {
      PreparedLoad prepared;
      prepared.id = id;
      (void)CancelLoad(prepared);
    }

    if (task) {
#ifdef MORI_UMBP_TESTING
      std::function<void()> before_publish_send;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        before_publish_send = before_publish_send_hook_;
      }
      if (before_publish_send) before_publish_send();
#endif
      bool unknown = false;
      PublishResult result = PublishRpc(task->records, &unknown);
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!unknown) {
          for (size_t i = 0; i < task->records.size(); ++i) {
            const auto& record = task->records[i];
            auto page = pages_.find(record.handle.page_id);
            if (page == pages_.end() || !(page->second.handle == record.handle) ||
                page->second.state != LocalPageState::kPublishSent) {
              continue;
            }
            const bool accepted = i < result.accepted.size() && result.accepted[i];
            if (accepted) {
              if (TransitionLocked(&page->second, {LocalPageState::kPublishSent},
                                   LocalPageState::kPublished) &&
                  page->second.return_requested) {
                (void)QueueRetireLocked(&page->second);
              }
            } else {
              if (page->second.return_requested) {
                (void)SettleUnpublishedLocked(&page->second, CompletionProof::kCompleted);
              } else {
                (void)TransitionLocked(&page->second, {LocalPageState::kPublishSent},
                                       LocalPageState::kInUse);
              }
            }
          }
        }
        RecordPublishReceiptLocked(
            task->id,
            unknown ? PublishReceiptState::kUnknown
            : result.success && !result.accepted.empty() &&
                    std::all_of(result.accepted.begin(), result.accepted.end(),
                                [](bool accepted) { return accepted; })
                ? PublishReceiptState::kAccepted
                : PublishReceiptState::kRejected,
            result.accepted);
        FinishTaskLocked(task, result);
        if (!task->retain_result) publish_tasks_.erase(task->id);
      }
    }

    if (!closing_.load() && !failed_.load()) {
      bool poll_retires = false;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_retire_poll_) {
          poll_retires = true;
          const bool active = state_counts_[static_cast<size_t>(LocalPageState::kRetiring)] != 0 ||
                              !pending_released_.empty();
          next_retire_poll_ = now + std::chrono::milliseconds(
                                        active ? 5 : std::max<uint32_t>(10, config_.heartbeat_ms));
        }
      }
      if (poll_retires) ProcessRetireProgress();
    }

    if (!closing_.load() && !failed_.load()) {
      std::vector<wire::EngineFrame> work;
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        if (!work_queue_.empty()) {
          progress_cv_.wait_for(lock, std::chrono::milliseconds(5));
          work.reserve(work_queue_.size());
          while (!work_queue_.empty()) {
            work.push_back(std::move(work_queue_.front()));
            work_queue_.pop_front();
          }
        }
      }
      if (!work.empty() && !ExecuteWorkBatch(std::move(work))) {
        std::lock_guard<std::mutex> call_lock(call_mutex_);
        FailStopLocked();
        break;
      }
    }
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  FinishAllPublishTasksLocked();
}

#ifdef MORI_UMBP_TESTING
void EngineClient::SetBeforePublishSendHookForTest(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  before_publish_send_hook_ = std::move(hook);
}

void EngineClient::SetBeforeRetireSendHookForTest(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  before_retire_send_hook_ = std::move(hook);
}

void EngineClient::SetAfterRetireReplyHookForTest(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  after_retire_reply_hook_ = std::move(hook);
}

void EngineClient::SetAfterReceiverStartedHookForTest(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  after_receiver_started_hook_ = std::move(hook);
}

void EngineClient::SetBeforeCallSendHookForTest(
    std::function<void(const wire::EngineFrame&)> hook) {
  std::lock_guard<std::mutex> lock(call_mutex_);
  before_call_send_hook_ = std::move(hook);
}

bool EngineClient::QuarantineLocalForTest(const AllocationHandle& handle) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto page = pages_.find(handle.page_id);
  return page != pages_.end() && page->second.handle == handle &&
         TransitionLocked(
             &page->second,
             {LocalPageState::kReserved, LocalPageState::kInUse, LocalPageState::kPendingFree,
              LocalPageState::kPublishSent, LocalPageState::kPublished, LocalPageState::kRetiring},
             LocalPageState::kQuarantined);
}

size_t EngineClient::PublishTaskCountForTest() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return publish_tasks_.size();
}

void EngineClient::SetEventQueueLimitForTest(size_t limit) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  event_queue_limit_ = std::max<size_t>(limit, 1);
}

void EngineClient::SetPublishReceiptLimitForTest(size_t limit) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  publish_receipt_limit_ = std::max<size_t>(limit, 1);
}

void EngineClient::InjectPublishReceiptForTest(uint64_t ticket, PublishReceiptState state) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  RecordPublishReceiptLocked(ticket, state);
}
#endif

void EngineClient::Close() {
  if (closing_.load()) return;
  std::vector<uint64_t> unstarted_loads;
  EngineExecutor* attached_executor = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto& [id, load] : prepared_loads_) {
      if (!load.started && !load.terminal) unstarted_loads.push_back(id);
    }
    attached_executor = executor_;
  }
  for (uint64_t id : unstarted_loads) {
    PreparedLoad prepared;
    prepared.id = id;
    (void)CancelLoad(prepared);
  }
  // A load worker calls FinishLoad on this client. Join it while the ordered
  // channel is still usable, then tear down the socket and client state.
  if (attached_executor != nullptr) attached_executor->Close();
  if (closing_.exchange(true)) return;
  progress_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lock(call_mutex_);
    const int fd = socket_.load();
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
  }
  if (receiver_thread_.joinable()) receiver_thread_.join();
  if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
  if (progress_thread_.joinable()) progress_thread_.join();
  {
    std::lock_guard<std::mutex> lock(call_mutex_);
    const int fd = socket_.exchange(-1);
    if (fd >= 0) close(fd);
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    FinishAllPublishTasksLocked();
    pages_.clear();
    state_counts_.fill(0);
    cached_pages_.clear();
    reservations_.clear();
    publish_queue_.clear();
    publish_tasks_.clear();
    local_holds_.clear();
    load_target_pages_.clear();
    prepared_loads_.clear();
    pending_released_.clear();
    released_operations_.clear();
    released_operation_order_.clear();
    if (executor_ != nullptr) executor_->AttachClient(nullptr);
    executor_ = nullptr;
    for (auto& region : dram_regions_) {
      if (region.fd >= 0) close(region.fd);
      region.fd = -1;
    }
    dram_regions_.clear();
  }
  healthy_.store(false);
}

}  // namespace mori::umbp::engine
