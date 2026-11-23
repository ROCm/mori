// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU-only RPC deadline tests: real peer clients and PoolManager grants, with
// host memory standing in for an external allocation. No HIP call is issued.
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "umbp/l1/l1_service.h"
#include "umbp_l1.grpc.pb.h"

namespace mori::umbp::l1 {
namespace {
namespace pb = ::umbp::l1;
using namespace std::chrono_literals;

std::string RpcDescriptor() {
  std::string out = "L1FBR001";
  out.append(64, 'r');
  for (uint64_t field : {uint64_t{16}, uint64_t{0}, uint64_t{16}, uint64_t{2}, uint64_t{4}})
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>(field >> (8 * i)));
  return out + "test";
}

class RpcMemory final : public MemoryApi {
 public:
  bool SupportsFabric() const override { return true; }
  bool Allocate(int, uint64_t, void**, ExportedMemory*, std::string*) override { return false; }
  void Free(int, void*, uint64_t) override { ADD_FAILURE() << "external allocation freed"; }
  bool Import(int, const std::string&, uint64_t offset, void** base, void** mapped,
              std::string*) override {
    *mapped = bytes_;
    *base = bytes_ + offset;
    return true;
  }
  void Unimport(int, void*) override {}
  bool Copy(int, void*, int, const void*, uint64_t, std::string*) override {
    ADD_FAILURE() << "probe/read-grant tests must not copy payload";
    return false;
  }
  bool Synchronize(int, std::string*) override { return true; }

 private:
  char bytes_[16]{};
};

// A controllable peer loses a reply or stalls before handling a release. Its
// wait also observes cancellation, so the passing tests do not leave handlers
// behind. Unblock lets the old long-deadline implementation fail in four
// seconds, rather than making the regression itself wait for 120 seconds.
class ControlledPeer final : public pb::UMBPL1Pool::Service {
 public:
  enum class Block { kNone, kProbe, kAcquireReply, kReleaseRequest };
  explicit ControlledPeer(PoolManager* owner = nullptr) : owner_(owner) {}
  std::atomic<Block> blocked{Block::kNone};

  void Unblock() {
    blocked = Block::kNone;
    wake_.notify_all();
  }
  grpc::Status PeerProbe(grpc::ServerContext* context, const pb::PeerProbeRequest* request,
                         pb::PeerProbeResponse* response) override {
    if (!Wait(context, Block::kProbe)) return Cancelled();
    PeerProbeResult result;
    result.incarnation = "empty-peer";
    if (owner_) {
      result = owner_->Probe(request->group(), {request->keys().begin(), request->keys().end()},
                             {request->unit_bytes().begin(), request->unit_bytes().end()});
    }
    response->set_ok(result.status.ok);
    response->set_error(result.status.error);
    response->set_incarnation(result.incarnation);
    response->set_available(result.available);
    return grpc::Status::OK;
  }
  grpc::Status AcquireRead(grpc::ServerContext* context, const pb::AcquireReadRequest* request,
                           pb::AcquireReadResponse* response) override {
    PeerReadRequest read{request->operation_id(), request->expected_incarnation(), request->group(),
                         {request->keys().begin(), request->keys().end()},
                         {request->unit_bytes().begin(), request->unit_bytes().end()}};
    auto result = owner_->AcquireRead(read);
    response->set_ok(result.status.ok);
    response->set_error(result.status.error);
    response->set_incarnation(result.incarnation);
    response->set_available(result.available);
    for (const auto& region : result.regions) {
      auto* out = response->add_regions();
      out->set_handle(region.handle);
      out->set_offset(region.offset);
      out->set_bytes(region.bytes);
    }
    if (!Wait(context, Block::kAcquireReply)) return Cancelled();
    return grpc::Status::OK;
  }
  grpc::Status ReleaseRead(grpc::ServerContext* context, const pb::ReleaseReadRequest* request,
                           pb::StatusResponse* response) override {
    if (!Wait(context, Block::kReleaseRequest)) return Cancelled();
    auto result = owner_->ReleaseRead(request->operation_id(), request->expected_incarnation(),
                                      request->drained());
    response->set_ok(result.ok);
    response->set_error(result.error);
    return grpc::Status::OK;
  }

 private:
  bool Wait(grpc::ServerContext* context, Block point) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (blocked == point && !context->IsCancelled()) wake_.wait_for(lock, 10ms);
    return !context->IsCancelled();
  }
  static grpc::Status Cancelled() {
    return {grpc::StatusCode::CANCELLED, "injected delayed peer control RPC"};
  }
  PoolManager* owner_;
  std::mutex mutex_;
  std::condition_variable wake_;
};

class PeerServer {
 public:
  explicit PeerServer(ControlledPeer& service) : service_(service) {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    if (!server_ || !port) throw std::runtime_error("could not start CPU test peer");
    peer = MakeL1Peer("127.0.0.1:" + std::to_string(port));
  }
  ~PeerServer() {
    service_.Unblock();
    server_->Shutdown();
  }
  std::shared_ptr<PeerApi> peer;

 private:
  ControlledPeer& service_;
  std::unique_ptr<grpc::Server> server_;
};

class L1PeerRpcTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memory = std::make_shared<RpcMemory>();
    owner = std::make_unique<PoolManager>(memory);
    Attach(*owner, "source");
    ASSERT_EQ(owner->Allocate("source", 1).slots, (std::vector<uint32_t>{0}));
    ASSERT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "a"}}).ok);
  }
  void Attach(PoolManager& manager, const std::string& id) {
    ASSERT_TRUE(manager.Attach(id, "g", 0, 16, {RpcDescriptor(), 0, "hip_fabric"}).status.ok);
    ASSERT_TRUE(manager.BindLayout(id, Layout{1, {{0, 16}}}).ok);
  }
  PeerReadRequest Read(const std::string& operation = "rpc-read") {
    return {operation, owner->Incarnation(), "g", {"a"}, {16}};
  }
  std::shared_ptr<RpcMemory> memory;
  std::unique_ptr<PoolManager> owner;
};

TEST_F(L1PeerRpcTest, SlowProbeLeavesTimeForAHealthyPeer) {
  ControlledPeer slow;
  slow.blocked = ControlledPeer::Block::kProbe;
  ControlledPeer healthy(owner.get());
  PeerServer slow_server(slow), healthy_server(healthy);
  PoolManager destination(std::make_shared<RpcMemory>());
  destination.ConfigurePeers({slow_server.peer, healthy_server.peer});
  Attach(destination, "destination");
  FetchOptions options;
  options.probe_only = true;
  auto pending = std::async(std::launch::async, [&] {
    return destination.Fetch("destination", {"a"}, options);
  });
  const bool bounded = pending.wait_for(4s) == std::future_status::ready;
  slow.Unblock();
  const auto result = pending.get();
  EXPECT_TRUE(bounded) << "one slow peer consumed the local Fetch deadline";
  EXPECT_TRUE(result.status.ok) << result.status.error;
  EXPECT_EQ(result.available, 1u);
}

TEST_F(L1PeerRpcTest, LostAcquireReplyKeepsSourcePinnedUntilConfirmedRelease) {
  ControlledPeer service(owner.get());
  service.blocked = ControlledPeer::Block::kAcquireReply;
  PeerServer server(service);
  const auto request = Read();
  auto pending = std::async(std::launch::async, [&] { return server.peer->AcquireRead(request); });
  const bool bounded = pending.wait_for(4s) == std::future_status::ready;
  service.Unblock();
  const auto read = pending.get();
  EXPECT_TRUE(bounded);
  EXPECT_FALSE(read.status.ok) << "delayed reply should expire before the parent Fetch";
  EXPECT_FALSE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  ASSERT_TRUE(server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true).ok);
  EXPECT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_FALSE(server.peer->AcquireRead(request).status.ok);
}

TEST_F(L1PeerRpcTest, ReleaseTimeoutDoesNotUnpinAndConfirmedRetryIsIdempotent) {
  ControlledPeer service(owner.get());
  PeerServer server(service);
  const auto request = Read();
  ASSERT_TRUE(server.peer->AcquireRead(request).status.ok);
  service.blocked = ControlledPeer::Block::kReleaseRequest;
  auto pending = std::async(std::launch::async, [&] {
    return server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true);
  });
  const bool bounded = pending.wait_for(4s) == std::future_status::ready;
  // Wait for the cancellation-observing handler before enabling the retry.
  const auto released = [&] {
    if (!bounded) service.Unblock();
    return pending.get();
  }();
  EXPECT_TRUE(bounded);
  EXPECT_FALSE(released.ok);
  EXPECT_FALSE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  service.Unblock();
  ASSERT_TRUE(server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true).ok);
  EXPECT_TRUE(server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true).ok);
  EXPECT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_FALSE(server.peer->AcquireRead(request).status.ok);
}
}  // namespace
}  // namespace mori::umbp::l1
