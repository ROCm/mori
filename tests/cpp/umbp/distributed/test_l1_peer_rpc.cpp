// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU-only peer RPC tests: real clients and PoolManager grants, with
// host memory standing in for an external allocation. No HIP call is issued.
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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
class ControlledPeer : public pb::UMBPL1Pool::Service {
 public:
  enum class Block { kNone, kProbe, kAcquireReply, kReleaseRequest, kCommit };
  explicit ControlledPeer(PoolManager* owner = nullptr) : owner_(owner) {}
  std::atomic<Block> blocked{Block::kNone};
  std::function<void(pb::AcquireReadResponse*)> rewrite;
  std::atomic<int> releases{0};
  std::atomic<int64_t> commit_deadline_ms{0};

  void Unblock() { blocked = Block::kNone; }
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
    response->set_read_epoch(result.read_epoch);
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
    for (const auto& handle : result.handles) response->add_handles(handle);
    for (const auto& region : result.regions) {
      auto* out = response->add_regions();
      out->set_handle_index(region.handle_index + 1);
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
    response->set_owner_incarnation(owner_->Incarnation());
    return grpc::Status::OK;
  }
  grpc::Status Commit(grpc::ServerContext* context, const pb::CommitRequest*,
                      pb::StatusResponse* response) override {
    commit_deadline_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             context->deadline() - std::chrono::system_clock::now()).count();
    if (!Wait(context, Block::kCommit)) return Cancelled();
    response->set_ok(true);
    return grpc::Status::OK;
  }

 private:
  bool Wait(grpc::ServerContext* context, Block point) {
    while (blocked == point && !context->IsCancelled()) std::this_thread::sleep_for(10ms);
    return !context->IsCancelled();
  }
  static grpc::Status Cancelled() {
    return {grpc::StatusCode::CANCELLED, "injected delayed peer control RPC"};
  }
  PoolManager* owner_;
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
    address = "127.0.0.1:" + std::to_string(port);
    peer = MakeL1Peer(address);
  }
  ~PeerServer() {
    if (controlled_) controlled_->Unblock();
    server_->Shutdown();
  }
  std::shared_ptr<PeerApi> peer;
  std::string address;

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
    const PeerReadRequest request{"1:rpc-read", owner->Incarnation(), "g", {"a"}, {16}};
    auto release = [&] {
      return server.peer->ReleaseRead(request.operation_id, request.expected_incarnation, true);
    };
    const bool acquire = point == ControlledPeer::Block::kAcquireReply;
    SCOPED_TRACE(acquire ? "lost Acquire reply" : "lost Release request");
    if (!acquire) ASSERT_TRUE(server.peer->AcquireRead(request).status.ok);
    service.blocked = point;
    auto pending = std::async(std::launch::async, [&] {
      if (!acquire) return release().status;
      const auto read = server.peer->AcquireRead(request);
      EXPECT_TRUE(read.incarnation.empty());  // A lost reply cannot prove an owner restart.
      return read.status;
    });
    const bool bounded = pending.wait_for(4s) == std::future_status::ready;
    if (!bounded) service.Unblock();  // Bound failures of the old 120-second deadline too.
    EXPECT_FALSE(pending.get().ok);
    EXPECT_TRUE(bounded);
    EXPECT_FALSE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
    service.Unblock();
    ASSERT_TRUE(release().status.ok);
    EXPECT_TRUE(release().status.ok);
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

class L1PeerWireTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, uint32_t>> {};

TEST_P(L1PeerWireTest, DictionaryRoundTripPreservesRegionsAndOneReplayableGrant) {
  const auto [slots, regions, pools] = GetParam();
  auto owner = std::make_shared<PoolManager>(std::make_shared<RpcMemory>());
  Layout layout{slots, {}};
  uint64_t bytes = 0;
  PeerReadRequest read{"1:wire", owner->Incarnation(), "g", {}, {}};
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
  for (int attempt = 0; attempt < 2; ++attempt) {
    const auto decoded = server.peer->AcquireRead(read);
    ASSERT_TRUE(decoded.status.ok);
    EXPECT_EQ(decoded.available, slots);
    ASSERT_EQ(decoded.regions.size(), slots * regions);
    EXPECT_EQ(decoded.handles, handles);
    for (uint32_t i = 0; i < slots * regions; ++i) {
      EXPECT_EQ(decoded.regions[i].handle_index, (i / regions) % pools);
      EXPECT_EQ(decoded.regions[i].offset, layout.regions[i % regions].offset +
                                              (i / regions) * read.unit_bytes[i % regions]);
      EXPECT_EQ(decoded.regions[i].bytes, read.unit_bytes[i % regions]);
    }
  }
  EXPECT_FALSE(owner->Apply("0", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_TRUE(
      server.peer->ReleaseRead(read.operation_id, read.expected_incarnation, true).status.ok);
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
    if (kind == 0) region->set_handle_index(0);
    if (kind == 1) region->set_handle_index(response->handles_size() + 1);
    if (kind == 2) response->set_handles(0, "");
    if (kind == 3) response->add_handles("");
  };
  PeerServer server(service);
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
  EXPECT_EQ(service.releases.load(), 1);
  EXPECT_TRUE(owner->Apply("source", {{OpKind::kPublish, 0, "replacement"}}).ok);
  EXPECT_TRUE(destination.Apply("destination", {{OpKind::kPublish, 0, "recomputed"}}).ok);
}

// Missing/out-of-range index; empty referenced/unreferenced dictionary entry.
INSTANTIATE_TEST_SUITE_P(References, L1MalformedHandleTest, ::testing::Range(0, 4));

class ClientPolicyService final : public ControlledPeer {
 public:
  std::atomic<bool> open_fabric{false};
  std::atomic<bool> open_metadata{true};
  grpc::Status Attach(grpc::ServerContext*, const pb::AttachRequest*,
                      pb::AttachResponse* out) override {
    out->set_ok(true);
    out->set_owner_incarnation("before-restart");
    return grpc::Status::OK;
  }
  grpc::Status Open(grpc::ServerContext*, const pb::OpenRequest*, pb::OpenResponse* out) override {
    out->set_ok(true);
    out->set_num_slots(1);
    if (open_metadata) {
      out->set_memory_kind(open_fabric ? "hip_fabric" : "hip_ipc");
      out->set_owner_incarnation("after-restart");
    }
    return grpc::Status::OK;
  }
  grpc::Status Fetch(grpc::ServerContext*, const pb::FetchRequest*, pb::FetchResponse*) override {
    return {grpc::StatusCode::UNAVAILABLE, "injected connection loss"};
  }
  grpc::Status Detach(grpc::ServerContext*, const pb::PoolRequest* in,
                      pb::StatusResponse* out) override {
    out->set_ok(in->expected_incarnation().empty() ||
                in->expected_incarnation() == "after-restart");
    if (!out->ok()) out->set_error("owner incarnation changed");
    return grpc::Status::OK;
  }
};

TEST(L1ClientPolicyTest, IPCPreservesRestartDetachAndFabricKeepsItsFence) {
  ClientPolicyService service;
  PeerServer server(service);
  EXPECT_FALSE(server.peer->Probe("g", {"a"}, {16}).status.ok);
  for (bool fabric : {false, true}) {
    service.open_fabric = fabric;
    L1Client client(server.address);
    const ExternalMemory external{"unused-test-handle", 0, fabric ? "hip_fabric" : "hip_ipc"};
    ASSERT_TRUE(client.Attach("pool", "g", 0, 16, external).status.ok);
    ASSERT_TRUE(client.Open("pool").status.ok);  // Must not replace the original Attach fence.
    FetchOptions options;
    options.dst_slots = {0};
    EXPECT_EQ(client.Fetch("pool", {"a"}, options).completion_unknown, fabric);
    EXPECT_EQ(client.Detach("pool").ok, !fabric);
    L1Client scheduler(server.address);
    ASSERT_TRUE(scheduler.Open("pool").status.ok);
    EXPECT_EQ(scheduler.Fetch("pool", {"a"}, options).completion_unknown, fabric);
    service.open_fabric = true;  // The same client must recognize an IPC-to-FABRIC replacement.
    ASSERT_TRUE(scheduler.Open("pool").status.ok);
    EXPECT_TRUE(scheduler.Fetch("pool", {"a"}, options).completion_unknown);
    EXPECT_TRUE(scheduler.Detach("pool").ok);
  }
  service.open_metadata = false;
  L1Client unknown(server.address);
  ASSERT_TRUE(unknown.Open("pool").status.ok);
  EXPECT_TRUE(unknown.Fetch("pool", {"a"}).completion_unknown);
}

TEST(L1ClientPolicyTest, CommitOptionalDeadlineBoundsAnUnresponsiveControlRPC) {
  ClientPolicyService service;
  PeerServer server(service);
  L1Client client(server.address);
  ASSERT_TRUE(client.Commit("pool", 1).ok);
  EXPECT_GT(service.commit_deadline_ms, 110000);
  service.blocked = ControlledPeer::Block::kCommit;
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(client.Commit("pool", 2, 1000).ok);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_GE(elapsed, 900ms);
  EXPECT_LT(elapsed, 3s);
  // gRPC rounds the wire timeout up to its encoding granularity.
  EXPECT_LE(service.commit_deadline_ms, 1100);
  service.Unblock();
}
}  // namespace
}  // namespace mori::umbp::l1
