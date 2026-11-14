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

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <utility>

#include "umbp_l1.grpc.pb.h"

namespace mori::umbp::l1 {

namespace {

namespace pb = ::umbp::l1;

// Everything but Attach is slot bookkeeping or a bounded copy.
constexpr std::chrono::seconds kCallDeadline{120};

template <typename Response>
void SetStatus(const Status& status, Response* response) {
  response->set_ok(status.ok);
  if (!status.ok) response->set_error(status.error);
}

class L1PoolService final : public pb::UMBPL1Pool::Service {
 public:
  explicit L1PoolService(std::shared_ptr<PoolManager> manager) : manager_(std::move(manager)) {}

  grpc::Status Attach(grpc::ServerContext*, const pb::AttachRequest* request,
                      pb::AttachResponse* response) override {
    AttachResult result =
        manager_->Attach(request->pool_id(), request->group(), request->device(), request->bytes());
    SetStatus(result.status, response);
    response->set_handle_kind(result.memory.kind);
    response->set_handle(result.memory.handle);
    response->set_bytes(result.bytes);
    response->set_created(result.created);
    return grpc::Status::OK;
  }

  grpc::Status BindLayout(grpc::ServerContext*, const pb::BindLayoutRequest* request,
                          pb::StatusResponse* response) override {
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
    OpenResult result = manager_->Open(request->pool_id());
    SetStatus(result.status, response);
    response->set_num_slots(result.num_slots);
    return grpc::Status::OK;
  }

  grpc::Status Allocate(grpc::ServerContext*, const pb::AllocateRequest* request,
                        pb::AllocateResponse* response) override {
    AllocateResult result =
        manager_->Allocate(request->pool_id(), request->count(), request->min_count());
    SetStatus(result.status, response);
    for (uint32_t slot : result.slots) response->add_slots(slot);
    for (std::string& key : result.evicted_keys) response->add_evicted_keys(std::move(key));
    return grpc::Status::OK;
  }

  grpc::Status Apply(grpc::ServerContext*, const pb::ApplyRequest* request,
                     pb::StatusResponse* response) override {
    if (request->kinds_size() != request->slots_size()) {
      SetStatus(Status::Error("apply: kinds and slots differ in length"), response);
      return grpc::Status::OK;
    }
    std::vector<Op> ops;
    ops.reserve(request->kinds_size());
    int next_key = 0;
    for (int i = 0; i < request->kinds_size(); ++i) {
      Op op;
      op.kind = static_cast<OpKind>(request->kinds(i));
      op.slot = request->slots(i);
      if (op.kind == OpKind::kPublish) {
        if (next_key >= request->keys_size()) {
          SetStatus(Status::Error("apply: a publish has no key"), response);
          return grpc::Status::OK;
        }
        op.key = request->keys(next_key++);
      }
      ops.push_back(std::move(op));
    }
    SetStatus(manager_->Apply(request->pool_id(), ops, request->stamp()), response);
    return grpc::Status::OK;
  }

  grpc::Status Commit(grpc::ServerContext*, const pb::CommitRequest* request,
                      pb::StatusResponse* response) override {
    SetStatus(manager_->Commit(request->pool_id(), request->committed()), response);
    return grpc::Status::OK;
  }

  grpc::Status Fetch(grpc::ServerContext*, const pb::FetchRequest* request,
                     pb::FetchResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    FetchResult result = manager_->Fetch(request->pool_id(), keys);
    SetStatus(result.status, response);
    for (uint32_t slot : result.slots) response->add_slots(slot);
    for (std::string& key : result.evicted_keys) response->add_evicted_keys(std::move(key));
    response->set_copied_slots(result.copied_slots);
    response->set_copied_bytes(result.copied_bytes);
    response->set_copy_micros(result.copy_micros);
    return grpc::Status::OK;
  }

  grpc::Status Clear(grpc::ServerContext*, const pb::PoolRequest* request,
                     pb::StatusResponse* response) override {
    SetStatus(manager_->Clear(request->pool_id()), response);
    return grpc::Status::OK;
  }

  grpc::Status Detach(grpc::ServerContext*, const pb::PoolRequest* request,
                      pb::StatusResponse* response) override {
    SetStatus(manager_->Detach(request->pool_id()), response);
    return grpc::Status::OK;
  }

  grpc::Status Stats(grpc::ServerContext*, const pb::StatsRequest*,
                     pb::StatsResponse* response) override {
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

 private:
  std::shared_ptr<PoolManager> manager_;
};

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

}  // namespace

std::unique_ptr<grpc::Service> MakeL1PoolService(std::shared_ptr<PoolManager> manager) {
  return std::make_unique<L1PoolService>(std::move(manager));
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

class L1Client::Impl {
 public:
  explicit Impl(const std::string& address)
      : stub_(pb::UMBPL1Pool::NewStub(
            grpc::CreateChannel(address, grpc::InsecureChannelCredentials()))) {}

  static void Bound(grpc::ClientContext* context) {
    context->set_deadline(std::chrono::system_clock::now() + kCallDeadline);
  }

  std::unique_ptr<pb::UMBPL1Pool::Stub> stub_;
};

L1Client::L1Client(const std::string& address) : impl_(std::make_unique<Impl>(address)) {}
L1Client::~L1Client() = default;

AttachResult L1Client::Attach(const std::string& pool_id, const std::string& group, int device,
                              uint64_t bytes) {
  pb::AttachRequest request;
  request.set_pool_id(pool_id);
  request.set_group(group);
  request.set_device(device);
  request.set_bytes(bytes);
  pb::AttachResponse response;
  // A long deadline: the server allocates and zero-fills the whole pool.
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(10));
  AttachResult result;
  result.status = FromResponse(impl_->stub_->Attach(&context, request, &response), response);
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
  result.status = FromResponse(impl_->stub_->Open(&context, request, &response), response);
  result.num_slots = response.num_slots();
  return result;
}

AllocateResult L1Client::Allocate(const std::string& pool_id, uint32_t count, uint32_t min_count) {
  pb::AllocateRequest request;
  request.set_pool_id(pool_id);
  request.set_count(count);
  request.set_min_count(min_count);
  pb::AllocateResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  AllocateResult result;
  result.status = FromResponse(impl_->stub_->Allocate(&context, request, &response), response);
  result.slots.assign(response.slots().begin(), response.slots().end());
  result.evicted_keys.assign(response.evicted_keys().begin(), response.evicted_keys().end());
  return result;
}

Status L1Client::Commit(const std::string& pool_id, uint64_t committed) {
  pb::CommitRequest request;
  request.set_pool_id(pool_id);
  request.set_committed(committed);
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  return FromResponse(impl_->stub_->Commit(&context, request, &response), response);
}

Status L1Client::Apply(const std::string& pool_id, const std::vector<Op>& ops, uint64_t stamp) {
  pb::ApplyRequest request;
  request.set_pool_id(pool_id);
  request.set_stamp(stamp);
  request.mutable_kinds()->Reserve(static_cast<int>(ops.size()));
  request.mutable_slots()->Reserve(static_cast<int>(ops.size()));
  for (const Op& op : ops) {
    request.add_kinds(static_cast<pb::OpKind>(op.kind));
    request.add_slots(op.slot);
    if (op.kind == OpKind::kPublish) request.add_keys(op.key);
  }
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  return FromResponse(impl_->stub_->Apply(&context, request, &response), response);
}

FetchResult L1Client::Fetch(const std::string& pool_id, const std::vector<std::string>& keys) {
  pb::FetchRequest request;
  request.set_pool_id(pool_id);
  for (const std::string& key : keys) request.add_keys(key);
  pb::FetchResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  FetchResult result;
  result.status = FromResponse(impl_->stub_->Fetch(&context, request, &response), response);
  result.slots.assign(response.slots().begin(), response.slots().end());
  result.evicted_keys.assign(response.evicted_keys().begin(), response.evicted_keys().end());
  result.copied_slots = response.copied_slots();
  result.copied_bytes = response.copied_bytes();
  result.copy_micros = response.copy_micros();
  return result;
}

Status L1Client::Clear(const std::string& pool_id) {
  pb::PoolRequest request;
  request.set_pool_id(pool_id);
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  return FromResponse(impl_->stub_->Clear(&context, request, &response), response);
}

Status L1Client::Detach(const std::string& pool_id) {
  pb::PoolRequest request;
  request.set_pool_id(pool_id);
  pb::StatusResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  return FromResponse(impl_->stub_->Detach(&context, request, &response), response);
}

Status L1Client::Stats(std::vector<PoolStats>* out) {
  pb::StatsRequest request;
  pb::StatsResponse response;
  grpc::ClientContext context;
  Impl::Bound(&context);
  const grpc::Status rpc = impl_->stub_->Stats(&context, request, &response);
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
