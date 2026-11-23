// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU-only peer RPC tests: real clients and PoolManager grants, with
// host memory standing in for an external allocation. No HIP call is issued.
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "umbp/l1/l1_service.h"
#include "umbp_l1.grpc.pb.h"

namespace mori::umbp::l1 {
namespace {
namespace pb = ::umbp::l1;
using namespace std::chrono_literals;

std::string RpcDescriptor(uint64_t bytes = 16) {
  std::string out = "L1FBR001";
  out.append(64, 'r');
  for (uint64_t field : {bytes, uint64_t{0}, bytes, uint64_t{2}, uint64_t{4}})
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
  char bytes_[32768]{};
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
  std::function<void(pb::AcquireReadResponse*)> rewrite;
  std::atomic<bool> compact_requested{false};
  std::atomic<int> releases{0};

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
    compact_requested = request->compact_handles();
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
    if (rewrite) rewrite(response);
    if (!Wait(context, Block::kAcquireReply)) return Cancelled();
    return grpc::Status::OK;
  }
  grpc::Status ReleaseRead(grpc::ServerContext* context, const pb::ReleaseReadRequest* request,
                           pb::StatusResponse* response) override {
    if (!Wait(context, Block::kReleaseRequest)) return Cancelled();
    ++releases;
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
  explicit PeerServer(ControlledPeer& service)
      : PeerServer(static_cast<grpc::Service&>(service)) {
    controlled_ = &service;
  }
  explicit PeerServer(grpc::Service& service) {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    server_ = builder.BuildAndStart();
    if (!server_ || !port) throw std::runtime_error("could not start CPU test peer");
    peer = MakeL1Peer("127.0.0.1:" + std::to_string(port));
    stub = pb::UMBPL1Pool::NewStub(grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
  }
  ~PeerServer() {
    if (controlled_) controlled_->Unblock();
    server_->Shutdown();
  }
  std::shared_ptr<PeerApi> peer;
  std::unique_ptr<pb::UMBPL1Pool::Stub> stub;

 private:
  ControlledPeer* controlled_ = nullptr;
  std::unique_ptr<grpc::Server> server_;
};

class L1PeerRpcTest : public ::testing::Test {
 protected:
  void SetUp() override {
    owner = std::make_unique<PoolManager>(std::make_shared<RpcMemory>());
    Attach(*owner, "source");
    ASSERT_EQ(owner->Allocate("source", 1).slots, (std::vector<uint32_t>{0}));
    ASSERT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "a"}}).ok);
  }
  void Attach(PoolManager& manager, const std::string& id) {
    ASSERT_TRUE(manager.Attach(id, "g", 0, 16, {RpcDescriptor(), 0, "hip_fabric"}).status.ok);
    ASSERT_TRUE(manager.BindLayout(id, Layout{1, {{0, 16}}}).ok);
  }
  void CheckReadTimeout(ControlledPeer::Block point) {
    ControlledPeer service(owner.get());
    PeerServer server(service);
    const PeerReadRequest request{"rpc-read", owner->Incarnation(), "g", {"a"}, {16}};
    auto release = [&] {
      return server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true);
    };
    const bool acquire = point == ControlledPeer::Block::kAcquireReply;
    SCOPED_TRACE(acquire ? "lost Acquire reply" : "lost Release request");
    if (!acquire) ASSERT_TRUE(server.peer->AcquireRead(request).status.ok);
    service.blocked = point;
    auto pending = std::async(std::launch::async, [&] {
      return acquire ? server.peer->AcquireRead(request).status : release();
    });
    const bool bounded = pending.wait_for(4s) == std::future_status::ready;
    if (!bounded) service.Unblock();  // Bound failures of the old 120-second deadline too.
    EXPECT_FALSE(pending.get().ok);
    EXPECT_TRUE(bounded);
    EXPECT_FALSE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
    service.Unblock();
    ASSERT_TRUE(release().ok);
    EXPECT_TRUE(release().ok);
    EXPECT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
    EXPECT_FALSE(server.peer->AcquireRead(request).status.ok);
  }
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
  CheckReadTimeout(ControlledPeer::Block::kAcquireReply);
}

TEST_F(L1PeerRpcTest, ReleaseTimeoutDoesNotUnpinAndConfirmedRetryIsIdempotent) {
  CheckReadTimeout(ControlledPeer::Block::kReleaseRequest);
}

TEST_F(L1PeerRpcTest, NewClientAcceptsLegacyInlineReply) {
  ControlledPeer service(owner.get());  // Ignores compact_handles, as an older server does.
  PeerServer server(service);
  const PeerReadRequest request{"legacy", owner->Incarnation(), "g", {"a"}, {16}};
  const auto read = server.peer->AcquireRead(request);
  ASSERT_TRUE(read.status.ok);
  ASSERT_EQ(read.regions.size(), 1u);
  EXPECT_TRUE(service.compact_requested.load());
  EXPECT_EQ(read.regions[0].handle, RpcDescriptor());
  EXPECT_TRUE(server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true).ok);
}

class L1PeerWireTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, uint32_t>> {};

TEST_P(L1PeerWireTest, CompactAndLegacyRepliesPreserveRegionsAndTheSameGrant) {
  const auto [slots, regions, pools] = GetParam();
  auto owner = std::make_shared<PoolManager>(std::make_shared<RpcMemory>());
  Layout layout{slots, {}};
  uint64_t bytes = 0;
  PeerReadRequest read{"wire", owner->Incarnation(), "g", {}, {}};
  for (uint32_t r = 0; r < regions; ++r) {
    layout.regions.push_back({bytes, r + 1});
    read.unit_bytes.push_back(r + 1);
    bytes += slots * (r + 1);
  }
  std::vector<std::string> handles;
  for (uint32_t p = 0; p < pools; ++p) {
    // Same opaque token, different complete descriptor: do not dedup a prefix.
    handles.push_back(RpcDescriptor(bytes + p));
    const std::string id = std::to_string(p);
    ASSERT_TRUE(owner->Attach(id, "g", 0, bytes, {handles.back(), 0, "hip_fabric"}).status.ok);
    ASSERT_TRUE(owner->BindLayout(id, layout).ok);
    ASSERT_EQ(owner->Allocate(id, slots).slots.size(), slots);
  }
  for (uint32_t s = 0; s < slots; ++s) {
    read.keys.push_back(std::to_string(s));
    ASSERT_TRUE(owner->Apply(std::to_string(s % pools), {{OpKind::kPublish, s, read.keys.back()}}).ok);
  }
  auto service = MakeL1PoolService(owner);
  PeerServer server(*service);
  pb::AcquireReadRequest request;
  request.set_operation_id(read.operation_id);
  request.set_expected_incarnation(read.expected_incarnation);
  request.set_group(read.group);
  for (const auto& key : read.keys) request.add_keys(key);
  for (uint64_t unit : read.unit_bytes) request.add_unit_bytes(unit);
  size_t legacy_bytes = 0;
  for (bool compact : {false, true, false}) {
    request.set_compact_handles(compact);
    grpc::ClientContext context;
    pb::AcquireReadResponse response;
    ASSERT_TRUE(server.stub->AcquireRead(&context, request, &response).ok());
    ASSERT_TRUE(response.ok()) << response.error();
    ASSERT_EQ(response.available(), slots);
    ASSERT_EQ(response.regions_size(), slots * regions);
    ASSERT_EQ(response.handles_size(), compact ? pools : 0);
    if (!compact) legacy_bytes = response.ByteSizeLong();
    if (compact && slots == 128) EXPECT_LT(response.ByteSizeLong(), legacy_bytes / 5);
    for (uint32_t i = 0; i < slots * regions; ++i) {
      const auto& region = response.regions(i);
      const uint32_t source = (i / regions) % pools;
      EXPECT_EQ(region.handle_index(), compact ? source + 1 : 0);
      EXPECT_EQ(region.handle(), compact ? "" : handles[source]);
      if (compact) EXPECT_EQ(response.handles(region.handle_index() - 1), handles[source]);
      EXPECT_EQ(region.offset(), layout.regions[i % regions].offset +
                                     (i / regions) * read.unit_bytes[i % regions]);
      EXPECT_EQ(region.bytes(), read.unit_bytes[i % regions]);
    }
  }
  const auto decoded = server.peer->AcquireRead(read);
  ASSERT_TRUE(decoded.status.ok);
  ASSERT_EQ(decoded.regions.size(), slots * regions);
  for (uint32_t i = 0; i < slots * regions; ++i) {
    EXPECT_EQ(decoded.regions[i].handle, handles[(i / regions) % pools]);
    EXPECT_EQ(decoded.regions[i].offset, layout.regions[i % regions].offset +
                                            (i / regions) * read.unit_bytes[i % regions]);
    EXPECT_EQ(decoded.regions[i].bytes, read.unit_bytes[i % regions]);
  }
  EXPECT_FALSE(owner->Apply("0", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_TRUE(server.peer->ReleaseRead(read.operation_id, read.expected_incarnation, true).ok);
  EXPECT_TRUE(owner->Apply("0", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_FALSE(server.peer->AcquireRead(read).status.ok);
}

INSTANTIATE_TEST_SUITE_P(Shapes, L1PeerWireTest,
                        ::testing::Values(std::make_tuple(1u, 1u, 1u), std::make_tuple(4u, 3u, 2u),
                                          std::make_tuple(128u, 16u, 1u)));

class L1MalformedHandleTest : public L1PeerRpcTest, public ::testing::WithParamInterface<int> {};

TEST_P(L1MalformedHandleTest, RejectsBeforeCopyAndFetchReleasesTheGrant) {
  ControlledPeer service(owner.get());
  service.rewrite = [kind = GetParam()](pb::AcquireReadResponse* response) {
    if (response->regions_size() == 0) return;
    auto* region = response->mutable_regions(0);
    if (kind != 0) response->add_handles(kind >= 3 ? "" : region->handle());
    if (kind != 1 && kind != 4) region->clear_handle();
    if (kind >= 1 && kind <= 3) region->set_handle_index(kind == 2 ? 2 : 1);
  };
  PeerServer server(service);
  const PeerReadRequest request{"malformed", owner->Incarnation(), "g", {"a"}, {16}};
  const auto read = server.peer->AcquireRead(request);
  EXPECT_FALSE(read.status.ok);
  EXPECT_TRUE(read.regions.empty());
  EXPECT_FALSE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  ASSERT_TRUE(server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true).ok);
  EXPECT_FALSE(server.peer->AcquireRead(request).status.ok);
  PoolManager destination(std::make_shared<RpcMemory>());
  destination.ConfigurePeers({server.peer});
  Attach(destination, "destination");
  FetchOptions options;
  options.dst_slots = destination.Allocate("destination", 1).slots;
  const auto result = destination.Fetch("destination", {"a"}, options);
  EXPECT_TRUE(result.status.ok);  // A rejected remote read permits normal recomputation.
  EXPECT_TRUE(result.slots.empty());
  EXPECT_EQ(result.copied_bytes, 0u);
  EXPECT_FALSE(result.completion_unknown);
  EXPECT_EQ(service.releases.load(), 2);
  EXPECT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_TRUE(destination.Apply("destination", {{OpKind::kPublish, 0, "recomputed"}}).ok);
}

// Missing inline descriptor; inline + index; out-of-range index; empty referenced
// dictionary entry; empty unused dictionary entry.
INSTANTIATE_TEST_SUITE_P(References, L1MalformedHandleTest, ::testing::Range(0, 5));
}  // namespace
}  // namespace mori::umbp::l1
