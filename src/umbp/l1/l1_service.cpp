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
#include "umbp/l1/l1_service.h"

#include <google/protobuf/arena.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/l1/shm_channel.h"
#include "umbp_l1.grpc.pb.h"

namespace mori::umbp::l1 {

namespace {

namespace pb = ::umbp::l1;

// Everything but Attach is slot bookkeeping or a bounded copy.
constexpr std::chrono::seconds kCallDeadline{120};

// The calls a shared-memory channel carries, as its method numbers. Each
// message is the call's request proto; a reply is its response proto.
enum ShmMethod : uint32_t {
  kShmApply = 1,     // one-way
  kShmCommit = 2,    // one-way
  kShmAllocate = 3,  // call
  kShmFetch = 4,     // call; the client sends only fetches that copy nothing
};

// An Allocate reply is a slot number and, at most, an evicted key per slot
// granted. Calls that could outgrow the reply buffer at this many bytes a
// slot go over gRPC instead.
constexpr uint64_t kReplyBytesPerSlot = 64;

uint64_t EnvU64(const char* name, uint64_t fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  return (end == raw || *end != '\0') ? fallback : value;
}

// What the pool manager did for one kind of call since the last report: how
// often, over how many items (ops, slots, keys), and for how long -- the
// caller's wait for the manager's lock included.
struct CallStats {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> items{0};
  std::atomic<uint64_t> nanos{0};
  std::atomic<uint64_t> max_nanos{0};

  void Add(uint64_t n, std::chrono::steady_clock::time_point started) {
    const uint64_t took =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - started)
                                  .count());
    calls.fetch_add(1, std::memory_order_relaxed);
    items.fetch_add(n, std::memory_order_relaxed);
    nanos.fetch_add(took, std::memory_order_relaxed);
    uint64_t seen = max_nanos.load(std::memory_order_relaxed);
    while (took > seen && !max_nanos.compare_exchange_weak(seen, took, std::memory_order_relaxed)) {
    }
  }
  std::string Take(const char* name) {
    const uint64_t c = calls.exchange(0), i = items.exchange(0), n = nanos.exchange(0),
                   m = max_nanos.exchange(0);
    if (c == 0) return std::string();
    char line[160];
    std::snprintf(line, sizeof(line),
                  " %s calls=%llu items=%llu total_us=%llu us/call=%.1f max_us=%llu;", name,
                  static_cast<unsigned long long>(c), static_cast<unsigned long long>(i),
                  static_cast<unsigned long long>(n / 1000),
                  static_cast<double>(n) / 1e3 / static_cast<double>(c),
                  static_cast<unsigned long long>(m / 1000));
    return line;
  }
};

template <typename Response>
void SetStatus(const Status& status, Response* response) {
  response->set_ok(status.ok);
  if (!status.ok) response->set_error(status.error);
}

class L1PoolService final : public pb::UMBPL1Pool::Service {
 public:
  explicit L1PoolService(std::shared_ptr<PoolManager> manager) : manager_(std::move(manager)) {
    const char* dir = std::getenv("UMBP_L1_SHM_DIR");
    shm_dir_ = (dir != nullptr && *dir != '\0') ? dir : "/dev/shm";
    shm_sizes_.request_bytes = EnvU64("UMBP_L1_SHM_REQUEST_BYTES", shm_sizes_.request_bytes);
    shm_sizes_.response_bytes = EnvU64("UMBP_L1_SHM_RESPONSE_BYTES", shm_sizes_.response_bytes);
    shm_spin_ = std::chrono::microseconds(EnvU64("UMBP_L1_SHM_SERVER_SPIN_US", 50));
    shm_handler_ = [this](uint32_t method, const char* data, size_t size, std::string* reply) {
      return HandleShm(method, data, size, reply);
    };
  }

  // End the shared-memory sessions: their streams never finish by themselves.
  void Stop() {
    stopping_.store(true, std::memory_order_release);
    ReportCalls(/*force=*/true);
  }

  grpc::Status Attach(grpc::ServerContext*, const pb::AttachRequest* request,
                      pb::AttachResponse* response) override {
    DrainShm();
    ExternalMemory external{request->external_handle(), request->external_offset()};
    AttachResult result = manager_->Attach(request->pool_id(), request->group(), request->device(),
                                           request->bytes(), external);
    SetStatus(result.status, response);
    response->set_handle_kind(result.memory.kind);
    response->set_handle(result.memory.handle);
    response->set_bytes(result.bytes);
    response->set_created(result.created);
    return grpc::Status::OK;
  }

  grpc::Status BindLayout(grpc::ServerContext*, const pb::BindLayoutRequest* request,
                          pb::StatusResponse* response) override {
    DrainShm();
    Layout layout;
    layout.num_slots = request->num_slots();
    layout.regions.reserve(request->regions_size());
    for (const pb::Region& region : request->regions()) {
      layout.regions.push_back(Region{region.offset(), region.unit_bytes()});
    }
    SetStatus(manager_->BindLayout(request->pool_id(), layout), response);
    return grpc::Status::OK;
  }

  grpc::Status Open(grpc::ServerContext*, const pb::OpenRequest* request,
                    pb::OpenResponse* response) override {
    DrainShm();
    OpenResult result = manager_->Open(request->pool_id());
    SetStatus(result.status, response);
    response->set_num_slots(result.num_slots);
    return grpc::Status::OK;
  }

  grpc::Status Allocate(grpc::ServerContext*, const pb::AllocateRequest* request,
                        pb::AllocateResponse* response) override {
    DrainShm();
    DoAllocate(*request, response);
    return grpc::Status::OK;
  }

  grpc::Status Apply(grpc::ServerContext*, const pb::ApplyRequest* request,
                     pb::StatusResponse* response) override {
    DrainShm();
    DoApply(*request, response);
    return grpc::Status::OK;
  }

  grpc::Status Commit(grpc::ServerContext*, const pb::CommitRequest* request,
                      pb::StatusResponse* response) override {
    DrainShm();
    DoCommit(*request, response);
    return grpc::Status::OK;
  }

  grpc::Status Fetch(grpc::ServerContext*, const pb::FetchRequest* request,
                     pb::FetchResponse* response) override {
    DrainShm();
    DoFetch(*request, response);
    return grpc::Status::OK;
  }

  grpc::Status Clear(grpc::ServerContext*, const pb::PoolRequest* request,
                     pb::StatusResponse* response) override {
    DrainShm();
    SetStatus(manager_->Clear(request->pool_id()), response);
    return grpc::Status::OK;
  }

  grpc::Status Detach(grpc::ServerContext*, const pb::PoolRequest* request,
                      pb::StatusResponse* response) override {
    DrainShm();
    SetStatus(manager_->Detach(request->pool_id()), response);
    return grpc::Status::OK;
  }

  grpc::Status Stats(grpc::ServerContext*, const pb::StatsRequest*,
                     pb::StatsResponse* response) override {
    DrainShm();
    for (const PoolStats& stats : manager_->Stats()) {
      pb::PoolStats* out = response->add_pools();
      out->set_pool_id(stats.pool_id);
      out->set_group(stats.group);
      out->set_device(stats.device);
      out->set_bytes(stats.bytes);
      out->set_num_slots(stats.num_slots);
      out->set_held(stats.held);
      out->set_cached(stats.cached);
      out->set_vacant(stats.vacant);
      out->set_evicted(stats.evicted);
      out->set_fetched_in_slots(stats.fetched_in_slots);
      out->set_fetched_in_bytes(stats.fetched_in_bytes);
      out->set_served_out_slots(stats.served_out_slots);
      out->set_copy_micros(stats.copy_micros);
    }
    return grpc::Status::OK;
  }

  // One client's shared-memory channel, served on this RPC's own thread for
  // as long as the client keeps the stream open.
  grpc::Status ShmSession(grpc::ServerContext* context, const pb::ShmSessionRequest*,
                          grpc::ServerWriter<pb::ShmSessionInfo>* writer) override {
    pb::ShmSessionInfo info;
    std::string error;
    std::shared_ptr<shm::ServerChannel> channel =
        shm::ServerChannel::Create(shm_dir_, shm_sizes_, &error);
    if (channel == nullptr) {
      MORI_UMBP_WARN("[L1Pool] no shared-memory channel: {}", error);
      info.set_ok(false);
      info.set_error(error);
      writer->Write(info);
      return grpc::Status::OK;
    }
    info.set_ok(true);
    info.set_path(channel->path());
    info.set_nonce(channel->nonce());
    AddChannel(channel);
    if (writer->Write(info)) {
      MORI_UMBP_INFO("[L1Pool] shared-memory channel {} open (requests {} B, replies {} B)",
                     channel->path(), channel->request_bytes(), channel->response_bytes());
      bool linked = true;
      while (!stopping_.load(std::memory_order_acquire) && !context->IsCancelled()) {
        // Its own client's calls only: each channel has its thread, so one
        // client's small call does not queue behind another's large one.
        serving_ = channel.get();
        channel->Drain(shm_handler_);
        serving_ = nullptr;
        ReportCalls(/*force=*/false);
        if (linked && channel->ClientAttached()) {
          // Both sides have it mapped; the name is no longer needed.
          channel->Unlink();
          linked = false;
        }
        channel->Wait(shm_spin_, std::chrono::milliseconds(100));
      }
      // What the client sent before it left still counts.
      channel->Drain(shm_handler_);
    }
    RemoveChannel(channel.get());
    channel->Close();
    MORI_UMBP_INFO("[L1Pool] shared-memory channel {} closed", channel->path());
    return grpc::Status::OK;
  }

 private:
  void DoAllocate(const pb::AllocateRequest& request, pb::AllocateResponse* response) {
    const auto started = std::chrono::steady_clock::now();
    AllocateResult result =
        manager_->Allocate(request.pool_id(), request.count(), request.min_count());
    allocate_stats_.Add(result.slots.size(), started);
    SetStatus(result.status, response);
    response->mutable_slots()->Reserve(static_cast<int>(result.slots.size()));
    for (uint32_t slot : result.slots) response->add_slots(slot);
    for (std::string& key : result.evicted_keys) response->add_evicted_keys(std::move(key));
  }

  void DoApply(const pb::ApplyRequest& request, pb::StatusResponse* response) {
    if (request.kinds_size() != request.slots_size()) {
      SetStatus(Status::Error("apply: kinds and slots differ in length"), response);
      return;
    }
    std::vector<Op> ops;
    ops.reserve(request.kinds_size());
    int next_key = 0;
    for (int i = 0; i < request.kinds_size(); ++i) {
      Op op;
      op.kind = static_cast<OpKind>(request.kinds(i));
      op.slot = request.slots(i);
      if (op.kind == OpKind::kPublish) {
        if (next_key >= request.keys_size()) {
          SetStatus(Status::Error("apply: a publish has no key"), response);
          return;
        }
        op.key = request.keys(next_key++);
      }
      ops.push_back(std::move(op));
    }
    const auto started = std::chrono::steady_clock::now();
    SetStatus(manager_->Apply(request.pool_id(), ops, request.stamp()), response);
    apply_stats_.Add(ops.size(), started);
  }

  void DoCommit(const pb::CommitRequest& request, pb::StatusResponse* response) {
    const auto started = std::chrono::steady_clock::now();
    SetStatus(manager_->Commit(request.pool_id(), request.committed()), response);
    commit_stats_.Add(1, started);
  }

  void DoFetch(const pb::FetchRequest& request, pb::FetchResponse* response) {
    std::vector<std::string> keys(request.keys().begin(), request.keys().end());
    FetchOptions options;
    options.dst_slots.assign(request.dst_slots().begin(), request.dst_slots().end());
    options.probe_only = request.probe_only();
    const auto started = std::chrono::steady_clock::now();
    FetchResult result = manager_->Fetch(request.pool_id(), keys, options);
    (options.probe_only ? probe_stats_ : fetch_stats_).Add(keys.size(), started);
    SetStatus(result.status, response);
    response->set_available(result.available);
    for (uint32_t slot : result.slots) response->add_slots(slot);
    for (std::string& key : result.evicted_keys) response->add_evicted_keys(std::move(key));
    response->set_copied_slots(result.copied_slots);
    response->set_copied_bytes(result.copied_bytes);
    response->set_copy_micros(result.copy_micros);
  }

  template <typename Request, typename Response>
  std::string RunShm(void (L1PoolService::*run)(const Request&, Response*), const char* data,
                     size_t size, std::string* reply) {
    // On an arena: a request of a thousand keys is a thousand small objects,
    // made and freed one by one otherwise.
    google::protobuf::Arena arena;
    auto* request = google::protobuf::Arena::CreateMessage<Request>(&arena);
    auto* response = google::protobuf::Arena::CreateMessage<Response>(&arena);
    if (!request->ParseFromArray(data, static_cast<int>(size))) {
      SetStatus(Status::Error("malformed shared-memory request"), response);
    } else {
      (this->*run)(*request, response);
    }
    if (reply != nullptr) response->SerializeToString(reply);
    return response->ok() ? std::string() : response->error();
  }

  std::string HandleShm(uint32_t method, const char* data, size_t size, std::string* reply) {
    switch (method) {
      case kShmApply:
        return RunShm(&L1PoolService::DoApply, data, size, reply);
      case kShmCommit:
        return RunShm(&L1PoolService::DoCommit, data, size, reply);
      case kShmAllocate:
        return RunShm(&L1PoolService::DoAllocate, data, size, reply);
      case kShmFetch:
        DrainOtherShm();
        return RunShm(&L1PoolService::DoFetch, data, size, reply);
      default:
        // An empty reply does not parse as ok: the caller sees a failure.
        return "unknown shared-memory method " + std::to_string(method);
    }
  }

  // One log line a minute while there is traffic, and one at the end: what
  // the manager did for each kind of call.
  void ReportCalls(bool force) {
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    int64_t last = last_report_.load(std::memory_order_relaxed);
    if (!force && now - last < 60) return;
    if (!last_report_.compare_exchange_strong(last, now)) return;
    const std::string line = apply_stats_.Take("apply") + commit_stats_.Take("commit") +
                             allocate_stats_.Take("allocate") + probe_stats_.Take("probe") +
                             fetch_stats_.Take("fetch");
    if (!line.empty()) MORI_UMBP_INFO("[L1Pool] manager time by call:{}", line);
  }

  // Apply what every client has queued in shared memory, waiting for a thread
  // that is draining a channel. Before an RPC: its sender may have queued
  // calls on any of them, and they come first.
  void DrainShm() {
    ReportCalls(/*force=*/false);
    if (channel_count_.load(std::memory_order_acquire) == 0) return;
    for (const std::shared_ptr<shm::ServerChannel>& channel : *ChannelsNow()) {
      if (channel->Pending()) channel->Drain(shm_handler_);
    }
  }

  // The other clients' queued calls, for a call that asks what other pools
  // hold: it should see what they published before it. Best effort -- a
  // channel another thread is draining is left to that thread -- and only
  // from a channel's own serving thread, which holds no other channel.
  void DrainOtherShm() {
    shm::ServerChannel* own = serving_;
    if (own == nullptr) return;
    serving_ = nullptr;  // calls handled below do not recurse into this
    for (const std::shared_ptr<shm::ServerChannel>& channel : *ChannelsNow()) {
      if (channel.get() != own && channel->Pending()) channel->TryDrain(shm_handler_);
    }
    serving_ = own;
  }

  std::shared_ptr<const std::vector<std::shared_ptr<shm::ServerChannel>>> ChannelsNow() {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    return channels_;
  }

  // The channel this thread is serving, while it drains it.
  static thread_local shm::ServerChannel* serving_;

  using Channels = std::vector<std::shared_ptr<shm::ServerChannel>>;

  void AddChannel(std::shared_ptr<shm::ServerChannel> channel) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    auto next = std::make_shared<Channels>(*channels_);
    next->push_back(std::move(channel));
    channels_ = std::move(next);
    channel_count_.store(channels_->size(), std::memory_order_release);
  }

  void RemoveChannel(const shm::ServerChannel* channel) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    auto next = std::make_shared<Channels>();
    for (const auto& kept : *channels_) {
      if (kept.get() != channel) next->push_back(kept);
    }
    channels_ = std::move(next);
    channel_count_.store(channels_->size(), std::memory_order_release);
  }

  std::shared_ptr<PoolManager> manager_;
  CallStats apply_stats_, commit_stats_, allocate_stats_, probe_stats_, fetch_stats_;
  std::atomic<int64_t> last_report_{0};
  std::string shm_dir_;
  shm::Sizes shm_sizes_;
  std::chrono::microseconds shm_spin_{50};
  shm::Handler shm_handler_;
  std::atomic<bool> stopping_{false};
  std::mutex channels_mutex_;
  // Replaced, never changed in place, so a drain walks it without the lock.
  std::shared_ptr<const Channels> channels_ = std::make_shared<Channels>();
  std::atomic<size_t> channel_count_{0};
};

thread_local shm::ServerChannel* L1PoolService::serving_ = nullptr;

Status FromGrpc(const grpc::Status& status) {
  if (status.ok()) return Status{};
  return Status::Error("UMBP L1 RPC failed: " + status.error_message());
}

template <typename Response>
Status FromResponse(const grpc::Status& rpc, const Response& response) {
  if (!rpc.ok()) return FromGrpc(rpc);
  if (!response.ok()) return Status::Error(response.error());
  return Status{};
}

Status FromShm(shm::SendResult result) {
  switch (result) {
    case shm::SendResult::kOk:
      return Status{};
    case shm::SendResult::kClosed:
      return Status::Error("UMBP L1 shared-memory channel: the server has gone");
    case shm::SendResult::kTimeout:
      return Status::Error("UMBP L1 shared-memory channel: no answer within the deadline");
    case shm::SendResult::kReplyTooLarge:
      return Status::Error(
          "UMBP L1 shared-memory channel: the reply exceeds UMBP_L1_SHM_RESPONSE_BYTES");
    default:
      return Status::Error("UMBP L1 shared-memory channel: request not sent");
  }
}

void AllocateFromResponse(const pb::AllocateResponse& response, AllocateResult* result) {
  result->slots.assign(response.slots().begin(), response.slots().end());
  result->evicted_keys.assign(response.evicted_keys().begin(), response.evicted_keys().end());
}

void FetchFromResponse(const pb::FetchResponse& response, FetchResult* result) {
  result->slots.assign(response.slots().begin(), response.slots().end());
  result->evicted_keys.assign(response.evicted_keys().begin(), response.evicted_keys().end());
  result->copied_slots = response.copied_slots();
  result->copied_bytes = response.copied_bytes();
  result->copy_micros = response.copy_micros();
  result->available = response.available();
}

void FillApply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp,
               pb::ApplyRequest* request) {
  request->set_pool_id(pool_id);
  request->set_stamp(stamp);
  request->mutable_kinds()->Reserve(static_cast<int>(ops.size()));
  request->mutable_slots()->Reserve(static_cast<int>(ops.size()));
  for (const Op& op : ops) {
    request->add_kinds(static_cast<pb::OpKind>(op.kind));
    request->add_slots(op.slot);
    if (op.kind == OpKind::kPublish) request->add_keys(op.key);
  }
}

}  // namespace

std::unique_ptr<grpc::Service> MakeL1PoolService(std::shared_ptr<PoolManager> manager) {
  return std::make_unique<L1PoolService>(std::move(manager));
}

void StopL1PoolService(grpc::Service* service) {
  if (auto* l1 = dynamic_cast<L1PoolService*>(service)) l1->Stop();
}

class L1Server::Impl {
 public:
  Impl(const std::string& address, std::shared_ptr<MemoryApi> api)
      : manager_(std::make_shared<PoolManager>(std::move(api))),
        service_(MakeL1PoolService(manager_)) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
  }
  ~Impl() {
    StopL1PoolService(service_.get());
    if (server_) server_->Shutdown();
  }

  std::shared_ptr<PoolManager> manager_;
  std::unique_ptr<grpc::Service> service_;
  std::unique_ptr<grpc::Server> server_;
};

L1Server::L1Server(const std::string& address, std::shared_ptr<MemoryApi> api)
    : impl_(std::make_unique<Impl>(address, std::move(api))) {}
L1Server::~L1Server() = default;
bool L1Server::ok() const { return impl_->server_ != nullptr; }

L1ClientOptions L1ClientOptions::FromEnv() {
  L1ClientOptions options;
  const char* transport = std::getenv("UMBP_L1_TRANSPORT");
  options.shm = transport != nullptr && std::string(transport) == "shm";
  options.shm_spin = std::chrono::microseconds(
      EnvU64("UMBP_L1_SHM_CLIENT_SPIN_US", static_cast<uint64_t>(options.shm_spin.count())));
  return options;
}

class L1Client::Impl {
 public:
  Impl(const std::string& address, const L1ClientOptions& options)
      : channel_(grpc::CreateChannel(address, grpc::InsecureChannelCredentials())),
        stub_(pb::UMBPL1Pool::NewStub(channel_)) {
    if (options.shm) OpenShm(address, options);
  }

  ~Impl() {
    if (session_ != nullptr) {
      session_context_->TryCancel();
      session_->Finish();
    }
  }

  static void Bound(grpc::ClientContext* context) {
    context->set_deadline(std::chrono::system_clock::now() + kCallDeadline);
  }

  // The parked failure of an earlier one-way call, as this call's status.
  Status Parked(ShmMethod method, const char* what) {
    std::string text;
    if (shm_ != nullptr && shm_->TakeError(method, &text)) {
      return Status::Error(std::string("UMBP L1: an earlier ") + what + " failed: " + text);
    }
    return Status{};
  }

  // The request's bytes, in a buffer this thread reuses.
  template <typename Request>
  static const std::string& Wire(const Request& request) {
    static thread_local std::string wire;
    wire.clear();
    request.SerializeToString(&wire);
    return wire;
  }

  // Whether a call of this size goes on the channel.
  bool Carries(const std::string& wire, uint64_t reply_bytes) const {
    return shm_ != nullptr && wire.size() <= shm_->max_request_bytes() &&
           reply_bytes <= shm_->response_bytes();
  }

  template <typename Response>
  Status CallShm(ShmMethod method, const std::string& wire, Response* response) {
    static thread_local std::string reply;
    const shm::SendResult sent =
        shm_->Call(method, wire.data(), wire.size(), &reply,
                   std::chrono::duration_cast<std::chrono::milliseconds>(kCallDeadline));
    if (sent != shm::SendResult::kOk) return FromShm(sent);
    if (!response->ParseFromString(reply)) {
      return Status::Error("UMBP L1 shared-memory channel: malformed reply");
    }
    // The reply follows everything sent before it, so an Apply that failed
    // has been parked by now. What this call returned cannot be trusted.
    Status parked = Parked(kShmApply, "Apply");
    if (!parked.ok) return parked;
    if (!response->ok()) return Status::Error(response->error());
    return Status{};
  }

  // The caller's hooks around one RPC.
  struct Blocking {
    explicit Blocking(Impl* i) : impl(i) {
      if (impl->enter_) impl->enter_();
    }
    ~Blocking() {
      if (impl->leave_) impl->leave_();
    }
    Blocking(const Blocking&) = delete;
    Blocking& operator=(const Blocking&) = delete;
    Impl* impl;
  };

  std::function<void()> enter_;
  std::function<void()> leave_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<pb::UMBPL1Pool::Stub> stub_;
  std::unique_ptr<grpc::ClientContext> session_context_;
  std::unique_ptr<grpc::ClientReader<pb::ShmSessionInfo>> session_;
  std::unique_ptr<shm::ClientChannel> shm_;

 private:
  void OpenShm(const std::string& address, const L1ClientOptions& options) {
    // A stream cannot carry a deadline for its opening alone, so wait for the
    // server first. Without it the client stays on gRPC.
    if (!channel_->WaitForConnected(std::chrono::system_clock::now() + options.shm_connect)) {
      MORI_UMBP_WARN("[L1Client] {} not reachable; calls stay on gRPC", address);
      return;
    }
    session_context_ = std::make_unique<grpc::ClientContext>();
    session_ = stub_->ShmSession(session_context_.get(), pb::ShmSessionRequest());
    pb::ShmSessionInfo info;
    std::string error;
    if (!session_->Read(&info)) {
      error = "the server does not offer it";
    } else if (!info.ok()) {
      error = info.error();
    } else {
      shm_ = shm::ClientChannel::Open(info.path(), info.nonce(), &error);
    }
    if (shm_ == nullptr) {
      MORI_UMBP_WARN("[L1Client] no shared-memory channel ({}); calls stay on gRPC", error);
      session_context_->TryCancel();
      session_->Finish();
      session_.reset();
      session_context_.reset();
      return;
    }
    shm_->set_spin(options.shm_spin);
    grpc::Channel* channel = channel_.get();
    // The stream is this channel's only tie to the server's lifetime.
    shm_->set_alive([channel] { return channel->GetState(false) == GRPC_CHANNEL_READY; });
    MORI_UMBP_INFO("[L1Client] Apply, Commit, Allocate and probes use shared memory ({})",
                   info.path());
  }
};

L1Client::L1Client(const std::string& address) : L1Client(address, L1ClientOptions::FromEnv()) {}
L1Client::L1Client(const std::string& address, const L1ClientOptions& options)
    : impl_(std::make_unique<Impl>(address, options)) {}
L1Client::~L1Client() = default;

bool L1Client::shm_active() const { return impl_->shm_ != nullptr; }

void L1Client::SetBlockingHooks(std::function<void()> enter, std::function<void()> leave) {
  impl_->enter_ = enter;
  impl_->leave_ = leave;
  if (impl_->shm_ != nullptr) impl_->shm_->set_blocking(std::move(enter), std::move(leave));
}

AttachResult L1Client::Attach(const std::string& pool_id, const std::string& group, int device,
                              uint64_t bytes, const ExternalMemory& external) {
  pb::AttachRequest request;
  request.set_pool_id(pool_id);
  request.set_group(group);
  request.set_device(device);
  request.set_bytes(bytes);
  request.set_external_handle(external.handle);
  request.set_external_offset(external.offset);
  pb::AttachResponse response;
  // A long deadline: the server allocates and zero-fills the whole pool.
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(10));
  AttachResult result;
  {
    Impl::Blocking blocking(impl_.get());
    result.status = FromResponse(impl_->stub_->Attach(&context, request, &response), response);
  }
  if (!result.status.ok) return result;
  result.memory.kind = response.handle_kind();
  result.memory.handle = response.handle();
  result.bytes = response.bytes();
  result.created = response.created();
  return result;
}

Status L1Client::BindLayout(const std::string& pool_id, const Layout& layout) {
  pb::BindLayoutRequest request;
  request.set_pool_id(pool_id);
  request.set_num_slots(layout.num_slots);
  for (const Region& region : layout.regions) {
    pb::Region* out = request.add_regions();
    out->set_offset(region.offset);
    out->set_unit_bytes(region.unit_bytes);
  }
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  Impl::Blocking blocking(impl_.get());
  return FromResponse(impl_->stub_->BindLayout(&context, request, &response), response);
}

OpenResult L1Client::Open(const std::string& pool_id) {
  pb::OpenRequest request;
  request.set_pool_id(pool_id);
  pb::OpenResponse response;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  Impl::Bound(&context);
  OpenResult result;
  {
    Impl::Blocking blocking(impl_.get());
    result.status = FromResponse(impl_->stub_->Open(&context, request, &response), response);
  }
  result.num_slots = response.num_slots();
  return result;
}

AllocateResult L1Client::Allocate(const std::string& pool_id, uint32_t count, uint32_t min_count) {
  pb::AllocateRequest request;
  request.set_pool_id(pool_id);
  request.set_count(count);
  request.set_min_count(min_count);
  // The reply names up to `count` slots and evicted keys: on an arena, not
  // one small allocation each.
  google::protobuf::Arena arena;
  pb::AllocateResponse& response =
      *google::protobuf::Arena::CreateMessage<pb::AllocateResponse>(&arena);
  AllocateResult result;
  if (impl_->shm_ != nullptr) {
    const std::string& wire = Impl::Wire(request);
    if (impl_->Carries(wire, uint64_t{count} * kReplyBytesPerSlot)) {
      result.status = impl_->CallShm(kShmAllocate, wire, &response);
      AllocateFromResponse(response, &result);
      return result;
    }
  }
  grpc::ClientContext context;
  Impl::Bound(&context);
  {
    Impl::Blocking blocking(impl_.get());
    result.status = FromResponse(impl_->stub_->Allocate(&context, request, &response), response);
  }
  AllocateFromResponse(response, &result);
  if (result.status.ok) result.status = impl_->Parked(kShmApply, "Apply");
  return result;
}

Status L1Client::Commit(const std::string& pool_id, uint64_t committed) {
  pb::CommitRequest request;
  request.set_pool_id(pool_id);
  request.set_committed(committed);
  if (impl_->shm_ != nullptr) {
    Status parked = impl_->Parked(kShmCommit, "Commit");
    if (!parked.ok) return parked;
    const std::string& wire = Impl::Wire(request);
    return FromShm(
        impl_->shm_->Send(kShmCommit, wire.data(), wire.size(),
                          std::chrono::duration_cast<std::chrono::milliseconds>(kCallDeadline)));
  }
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  Impl::Blocking blocking(impl_.get());
  return FromResponse(impl_->stub_->Commit(&context, request, &response), response);
}

Status L1Client::Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp) {
  google::protobuf::Arena arena;
  pb::ApplyRequest& request = *google::protobuf::Arena::CreateMessage<pb::ApplyRequest>(&arena);
  FillApply(pool_id, ops, stamp, &request);
  if (impl_->shm_ != nullptr) {
    Status parked = impl_->Parked(kShmApply, "Apply");
    if (!parked.ok) return parked;
    const std::string& wire = Impl::Wire(request);
    const shm::SendResult sent =
        impl_->shm_->Send(kShmApply, wire.data(), wire.size(),
                          std::chrono::duration_cast<std::chrono::milliseconds>(kCallDeadline));
    // A batch larger than the ring takes goes over the socket, where it is
    // applied after everything queued before it.
    if (sent != shm::SendResult::kTooLarge) return FromShm(sent);
  }
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  Impl::Blocking blocking(impl_.get());
  return FromResponse(impl_->stub_->Apply(&context, request, &response), response);
}

FetchResult L1Client::Fetch(const std::string& pool_id, const std::vector<std::string>& keys,
                            const FetchOptions& options) {
  google::protobuf::Arena arena;
  pb::FetchRequest& request = *google::protobuf::Arena::CreateMessage<pb::FetchRequest>(&arena);
  request.set_pool_id(pool_id);
  request.mutable_keys()->Reserve(static_cast<int>(keys.size()));
  for (const std::string& key : keys) request.add_keys(key);
  request.mutable_dst_slots()->Reserve(static_cast<int>(options.dst_slots.size()));
  for (uint32_t slot : options.dst_slots) request.add_dst_slots(slot);
  request.set_probe_only(options.probe_only);
  pb::FetchResponse& response = *google::protobuf::Arena::CreateMessage<pb::FetchResponse>(&arena);
  FetchResult result;
  // Only a probe: a fetch that copies would hold up this client's channel,
  // and with it the Commit its own destinations may be waiting for.
  if (impl_->shm_ != nullptr && options.probe_only) {
    const std::string& wire = Impl::Wire(request);
    if (impl_->Carries(wire, 4096)) {
      result.status = impl_->CallShm(kShmFetch, wire, &response);
      FetchFromResponse(response, &result);
      return result;
    }
  }
  grpc::ClientContext context;
  Impl::Bound(&context);
  {
    Impl::Blocking blocking(impl_.get());
    result.status = FromResponse(impl_->stub_->Fetch(&context, request, &response), response);
  }
  FetchFromResponse(response, &result);
  if (result.status.ok) result.status = impl_->Parked(kShmApply, "Apply");
  return result;
}

Status L1Client::Clear(const std::string& pool_id) {
  pb::PoolRequest request;
  request.set_pool_id(pool_id);
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  Impl::Blocking blocking(impl_.get());
  return FromResponse(impl_->stub_->Clear(&context, request, &response), response);
}

Status L1Client::Detach(const std::string& pool_id) {
  pb::PoolRequest request;
  request.set_pool_id(pool_id);
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  Impl::Blocking blocking(impl_.get());
  return FromResponse(impl_->stub_->Detach(&context, request, &response), response);
}

Status L1Client::Stats(std::vector<PoolStats>* out) {
  pb::StatsRequest request;
  pb::StatsResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  grpc::Status rpc;
  {
    Impl::Blocking blocking(impl_.get());
    rpc = impl_->stub_->Stats(&context, request, &response);
  }
  if (!rpc.ok()) return FromGrpc(rpc);
  out->clear();
  for (const pb::PoolStats& in : response.pools()) {
    PoolStats stats;
    stats.pool_id = in.pool_id();
    stats.group = in.group();
    stats.device = in.device();
    stats.bytes = in.bytes();
    stats.num_slots = in.num_slots();
    stats.held = in.held();
    stats.cached = in.cached();
    stats.vacant = in.vacant();
    stats.evicted = in.evicted();
    stats.fetched_in_slots = in.fetched_in_slots();
    stats.fetched_in_bytes = in.fetched_in_bytes();
    stats.served_out_slots = in.served_out_slots();
    stats.copy_micros = in.copy_micros();
    out->push_back(std::move(stats));
  }
  return Status{};
}

}  // namespace mori::umbp::l1
