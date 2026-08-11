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

// PeerServiceServer tier dispatch (backend-agnostic refactor Phase 3).
//
// The peer service used to hold one typed PageBackend* and serve every RPC from
// it.  It now holds a BackendRegistry* and dispatches on the request's tier
// tag.  These tests drive the real gRPC surface with two MockBackends
// registered, so they assert the wire-visible contract rather than the
// handlers' internals:
//
//   * a request routes to the backend registered for its tier, and a tier with
//     no backend fails cleanly instead of hitting whichever one happens to be
//     there
//   * Commit / Abort carry NO tier on the wire, yet still reach the backend
//     that handed the slot out — the peer tags the tier into the opaque
//     slot_id (this is what makes two backends numbering their slots from 1
//     unambiguous)
//   * a mixed-tier batch is grouped per backend but answered in request order
//   * Resolve / Evict, which carry no tier either, walk the peer-local
//     read-rank order and fan out respectively

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/backend/mock_backend.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

// Every medium is equivalent (Phase 4), so the registry walks them in ascending
// TierType order — HBM(1) before DRAM(2).  The Resolve test depends only on
// that order being deterministic, not on it meaning "faster".
constexpr TierType kFirstByTier = TierType::HBM;
constexpr TierType kSecondByTier = TierType::DRAM;

class PeerServiceDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_.Register(std::make_unique<MockBackend>(kFirstByTier));
    registry_.Register(std::make_unique<MockBackend>(kSecondByTier));

    // OS-assigned ports are not reachable through PeerServiceServer::Start
    // (it does not report the bound port back), so probe a random high range
    // rather than hardcoding one port that another process may already hold.
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> pick(20000, 60000);
    for (int attempt = 0; attempt < 50 && !started_; ++attempt) {
      const uint16_t port = static_cast<uint16_t>(pick(rng));
      server_ = std::make_unique<PeerServiceServer>(&registry_);
      if (server_->Start(port)) {
        started_ = true;
        auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                           grpc::InsecureChannelCredentials());
        stub_ = ::umbp::UMBPPeer::NewStub(channel);
      } else {
        server_.reset();
      }
    }
    ASSERT_TRUE(started_) << "could not bind a peer service port";
  }

  void TearDown() override {
    stub_.reset();
    if (server_) server_->Stop();
    server_.reset();
  }

  MediumBackend* Backend(TierType tier) { return registry_.Get(tier); }

  // The registry hands out MediumBackend; PublishBuffers is test scaffolding
  // that only the mock has.  Safe because SetUp registers nothing else.
  MockBackend* Mock(TierType tier) { return static_cast<MockBackend*>(registry_.Get(tier)); }

  static ::umbp::TierType Proto(TierType t) { return static_cast<::umbp::TierType>(t); }

  // Allocate one slot and return the (opaque, tier-tagged) slot_id.
  ::umbp::AllocateSlotResponse Allocate(const std::string& key, uint64_t size, TierType tier) {
    ::umbp::AllocateSlotRequest req;
    req.set_key(key);
    req.set_size(size);
    req.set_tier(Proto(tier));
    ::umbp::AllocateSlotResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->AllocateSlot(&ctx, req, &resp).ok());
    return resp;
  }

  bool Commit(uint64_t slot_id, const std::string& key) {
    ::umbp::CommitSlotRequest req;
    req.set_slot_id(slot_id);
    req.set_key(key);
    ::umbp::CommitSlotResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->CommitSlot(&ctx, req, &resp).ok());
    return resp.success();
  }

  // Put `key` straight into one backend, bypassing the RPC surface, so a test
  // can set up peer-side state without asserting on the path it is testing.
  void SeedKey(TierType tier, const std::string& key, uint64_t size) {
    auto* backend = Backend(tier);
    ASSERT_NE(backend, nullptr);
    auto allocated = backend->BatchAllocate({{key, size}});
    ASSERT_EQ(allocated[0].outcome, AllocateOutcome::kSuccessAllocated);
    ASSERT_TRUE(backend->BatchCommit({{allocated[0].slot_id, key}})[0].success);
  }

  BackendRegistry registry_;
  std::unique_ptr<PeerServiceServer> server_;
  std::unique_ptr<::umbp::UMBPPeer::Stub> stub_;
  bool started_ = false;
};

// ---- Allocate ---------------------------------------------------------------

TEST_F(PeerServiceDispatchTest, AllocateRoutesToTheBackendForItsTier) {
  auto resp = Allocate("k", 128, kFirstByTier);
  ASSERT_EQ(resp.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
  ASSERT_TRUE(Commit(resp.slot_id(), "k"));

  EXPECT_EQ(Backend(kFirstByTier)->OwnedKeyCount(), 1u);
  EXPECT_EQ(Backend(kSecondByTier)->OwnedKeyCount(), 0u);
}

TEST_F(PeerServiceDispatchTest, AllocateForATierWithNoBackendFailsCleanly) {
  // SSD is a valid wire tier with no backend registered here — a normal
  // deployment shape, so it must be a FAILED outcome and not fall through to
  // whichever backend happens to exist.
  auto resp = Allocate("k", 128, TierType::SSD);
  EXPECT_EQ(resp.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
  EXPECT_EQ(Backend(kFirstByTier)->OwnedKeyCount(), 0u);
  EXPECT_EQ(Backend(kSecondByTier)->OwnedKeyCount(), 0u);
}

// ---- The slot_id carries the tier -------------------------------------------

TEST_F(PeerServiceDispatchTest, ConcurrentSlotsOnTwoBackendsStayDistinct) {
  // Each backend numbers its own slots from 1, so without the tier tag these
  // two allocations would come back with the SAME opaque id and Commit could
  // not tell them apart.
  auto first = Allocate("a", 8, kFirstByTier);
  auto second = Allocate("b", 8, kSecondByTier);
  ASSERT_EQ(first.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
  ASSERT_EQ(second.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
  EXPECT_NE(first.slot_id(), second.slot_id());

  EXPECT_TRUE(Commit(first.slot_id(), "a"));
  EXPECT_TRUE(Commit(second.slot_id(), "b"));
  EXPECT_EQ(Backend(kFirstByTier)->OwnedKeyCount(), 1u);
  EXPECT_EQ(Backend(kSecondByTier)->OwnedKeyCount(), 1u);
}

TEST_F(PeerServiceDispatchTest, CommitOfAnUnknownSlotFails) {
  // Slot 0 decodes to tier UNKNOWN, which has no backend: "slot unknown".
  EXPECT_FALSE(Commit(0, "nope"));
}

// ---- Two live media share one buffer_index space ----------------------------
//
// buffer_index is BACKEND-local: every backend numbers its buffers from 0, and
// every real backend publishes exactly one buffer, so on a peer with two live
// media both claim index 0.  These tests pin the two wire surfaces where that
// used to lose data.  Both fail without BufferMemoryDesc.backend_id.

TEST_F(PeerServiceDispatchTest, GetPeerInfoPublishesEveryBackendsBuffers) {
  // Distinct first descriptor byte per backend so we can tell whose is whose.
  constexpr uint8_t kFirstTag = 0xA1;
  constexpr uint8_t kSecondTag = 0xB2;
  Mock(kFirstByTier)->PublishBuffers(1, kFirstTag);
  Mock(kSecondByTier)->PublishBuffers(1, kSecondTag);

  ::umbp::GetPeerInfoRequest req;
  ::umbp::GetPeerInfoResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->GetPeerInfo(&ctx, req, &resp).ok());

  // Both media must be advertised.  The bootstrap wire used to carry one, and
  // dropping the other was not merely incomplete: the reader caches by index
  // and then asks the peer to omit descriptors it believes it holds, so the
  // unadvertised medium's pages were read against the advertised medium's
  // memory — a hit, with the wrong bytes.
  ASSERT_EQ(resp.buffer_descs_size(), 2);

  std::map<uint8_t, uint32_t> backend_id_by_tag;
  for (const auto& d : resp.buffer_descs()) {
    ASSERT_FALSE(d.desc().empty());
    // Every backend numbers from 0, so the indices collide by design; only the
    // backend_id separates them.
    EXPECT_EQ(d.buffer_index(), 0u);
    backend_id_by_tag[static_cast<uint8_t>(d.desc()[0])] = d.backend_id();
  }
  ASSERT_EQ(backend_id_by_tag.count(kFirstTag), 1u) << "first medium was not advertised";
  ASSERT_EQ(backend_id_by_tag.count(kSecondTag), 1u) << "second medium was not advertised";
  EXPECT_NE(backend_id_by_tag[kFirstTag], backend_id_by_tag[kSecondTag])
      << "two media sharing a backend_id collapses back into one address space";

  // Uniform across backends, so one field still describes the whole peer.
  EXPECT_EQ(resp.page_size(), Backend(kFirstByTier)->PageSize());
}

TEST_F(PeerServiceDispatchTest, BatchResolveAnswersKeysHeldByDifferentMedia) {
  Mock(kFirstByTier)->PublishBuffers(1, 0xA1);
  Mock(kSecondByTier)->PublishBuffers(1, 0xB2);
  SeedKey(kFirstByTier, "in-first", 8);
  SeedKey(kSecondByTier, "in-second", 8);

  ::umbp::BatchResolveKeysRequest req;
  req.add_keys("in-first");
  req.add_keys("in-second");
  ::umbp::BatchResolveKeysResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->BatchResolveKeys(&ctx, req, &resp).ok());

  // One response used to be served entirely by the FIRST medium with any hit,
  // so "in-second" came back found=false purely because another key in the same
  // batch hit earlier — a silent hit-rate loss on every mixed-media node.
  ASSERT_EQ(resp.found_size(), 2);
  ASSERT_EQ(resp.backend_id_size(), 2);
  EXPECT_TRUE(resp.found(0)) << "key held by the first medium";
  EXPECT_TRUE(resp.found(1)) << "key held by the second medium was reported missing";

  // Each key names the medium that served it, so its backend-local pages can be
  // resolved against the right buffers.
  EXPECT_NE(resp.backend_id(0), resp.backend_id(1));
  EXPECT_EQ(resp.tier(0), Proto(kFirstByTier));
  EXPECT_EQ(resp.tier(1), Proto(kSecondByTier));
}

TEST_F(PeerServiceDispatchTest, AllocateNamesTheBackendItsPagesBelongTo) {
  Mock(kFirstByTier)->PublishBuffers(1, 0xA1);
  Mock(kSecondByTier)->PublishBuffers(1, 0xB2);

  auto first = Allocate("a", 8, kFirstByTier);
  auto second = Allocate("b", 8, kSecondByTier);
  ASSERT_EQ(first.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
  ASSERT_EQ(second.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);

  // Same reason Commit needs the tier tagged into slot_id: the writer has to
  // know which buffers its pages index into before it can RDMA into them.
  EXPECT_NE(first.backend_id(), second.backend_id());
  for (const auto& d : first.descs()) EXPECT_EQ(d.backend_id(), first.backend_id());
  for (const auto& d : second.descs()) EXPECT_EQ(d.backend_id(), second.backend_id());
}

TEST_F(PeerServiceDispatchTest, AbortOfAnUnknownSlotIsIdempotentlyTrue) {
  ::umbp::AbortSlotRequest req;
  req.set_slot_id(0);
  ::umbp::AbortSlotResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->AbortSlot(&ctx, req, &resp).ok());
  EXPECT_TRUE(resp.success());
}

TEST_F(PeerServiceDispatchTest, AbortReachesTheBackendThatOwnsTheSlot) {
  auto allocated = Allocate("a", 8, kFirstByTier);
  ASSERT_EQ(allocated.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);

  ::umbp::AbortSlotRequest req;
  req.set_slot_id(allocated.slot_id());
  ::umbp::AbortSlotResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->AbortSlot(&ctx, req, &resp).ok());
  EXPECT_TRUE(resp.success());

  // The slot is gone, so the commit that would have followed now fails.
  EXPECT_FALSE(Commit(allocated.slot_id(), "a"));
  EXPECT_EQ(Backend(kFirstByTier)->OwnedKeyCount(), 0u);
}

// ---- Mixed-tier batches -----------------------------------------------------

TEST_F(PeerServiceDispatchTest, BatchAllocateGroupsByTierButAnswersInRequestOrder) {
  ::umbp::BatchAllocateSlotsRequest req;
  const std::vector<std::pair<std::string, TierType>> wanted = {
      {"e0", kFirstByTier},
      {"e1", TierType::SSD},  // no backend -> FAILED, but keeps its slot in the answer
      {"e2", kSecondByTier},
      {"e3", kFirstByTier},
  };
  for (const auto& [key, tier] : wanted) {
    auto* entry = req.add_entries();
    entry->set_key(key);
    entry->set_size(16);
    entry->set_tier(Proto(tier));
  }

  ::umbp::BatchAllocateSlotsResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->BatchAllocateSlots(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.entries_size(), 4);
  EXPECT_EQ(resp.entries(0).outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
  EXPECT_EQ(resp.entries(1).outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);
  EXPECT_EQ(resp.entries(2).outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);
  EXPECT_EQ(resp.entries(3).outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED);

  // Commit them back through the batch RPC — also mixed-tier, also positional.
  ::umbp::BatchCommitSlotsRequest commit_req;
  for (int i : {0, 2, 3}) {
    auto* entry = commit_req.add_entries();
    entry->set_slot_id(resp.entries(i).slot_id());
    entry->set_key(wanted[i].first);
  }
  ::umbp::BatchCommitSlotsResponse commit_resp;
  grpc::ClientContext commit_ctx;
  ASSERT_TRUE(stub_->BatchCommitSlots(&commit_ctx, commit_req, &commit_resp).ok());
  ASSERT_EQ(commit_resp.success_size(), 3);
  for (int i = 0; i < 3; ++i) EXPECT_TRUE(commit_resp.success(i)) << "entry " << i;

  // e0 and e3 landed on the first backend, e2 on the second.
  EXPECT_EQ(Backend(kFirstByTier)->OwnedKeyCount(), 2u);
  EXPECT_EQ(Backend(kSecondByTier)->OwnedKeyCount(), 1u);
}

TEST_F(PeerServiceDispatchTest, BatchCommitOfAnUnroutableSlotReportsFalseInPlace) {
  auto allocated = Allocate("a", 8, kSecondByTier);
  ::umbp::BatchCommitSlotsRequest req;
  auto* bad = req.add_entries();
  bad->set_slot_id(0);  // tier UNKNOWN
  bad->set_key("bad");
  auto* good = req.add_entries();
  good->set_slot_id(allocated.slot_id());
  good->set_key("a");

  ::umbp::BatchCommitSlotsResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->BatchCommitSlots(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.success_size(), 2);
  EXPECT_FALSE(resp.success(0));
  EXPECT_TRUE(resp.success(1));
}

// ---- Resolve / Evict carry no tier ------------------------------------------

TEST_F(PeerServiceDispatchTest, ResolveTakesTheFirstHitInMediaOrder) {
  // Same key mirrored across both media with different sizes, so the response
  // says which one served it.
  SeedKey(kFirstByTier, "mirrored", 111);
  SeedKey(kSecondByTier, "mirrored", 222);

  ::umbp::ResolveKeyRequest req;
  req.set_key("mirrored");
  ::umbp::ResolveKeyResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->ResolveKey(&ctx, req, &resp).ok());
  ASSERT_TRUE(resp.found());
  EXPECT_EQ(resp.size(), 111u);
}

TEST_F(PeerServiceDispatchTest, ResolveFindsAKeyHeldOnlyByALowerRankedMedium) {
  SeedKey(kSecondByTier, "only-second", 77);

  ::umbp::ResolveKeyRequest req;
  req.set_key("only-second");
  ::umbp::ResolveKeyResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->ResolveKey(&ctx, req, &resp).ok());
  ASSERT_TRUE(resp.found());
  EXPECT_EQ(resp.size(), 77u);
}

TEST_F(PeerServiceDispatchTest, ResolveMissIsNotAnError) {
  ::umbp::ResolveKeyRequest req;
  req.set_key("absent");
  ::umbp::ResolveKeyResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->ResolveKey(&ctx, req, &resp).ok());
  EXPECT_FALSE(resp.found());
}

TEST_F(PeerServiceDispatchTest, BatchResolveReportsTheServingTier) {
  SeedKey(kSecondByTier, "b-only", 55);

  ::umbp::BatchResolveKeysRequest req;
  req.add_keys("absent");
  req.add_keys("b-only");
  ::umbp::BatchResolveKeysResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->BatchResolveKeys(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.found_size(), 2);
  EXPECT_FALSE(resp.found(0));
  EXPECT_TRUE(resp.found(1));
  ASSERT_EQ(resp.tier_size(), 2);
  EXPECT_EQ(resp.tier(1), Proto(kSecondByTier));
  EXPECT_EQ(resp.size(1), 55u);
}

TEST_F(PeerServiceDispatchTest, EvictFansOutAcrossEveryMedium) {
  // A key mirrored across media must disappear from ALL of them, and the freed
  // bytes master sizes its next round from are the sum.
  SeedKey(kFirstByTier, "mirrored", 111);
  SeedKey(kSecondByTier, "mirrored", 222);

  ::umbp::EvictKeyRequest req;
  req.add_keys("mirrored");
  ::umbp::EvictKeyResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->EvictKey(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.evicted_size(), 1);
  EXPECT_EQ(resp.evicted(0).key(), "mirrored");
  EXPECT_EQ(resp.evicted(0).bytes_freed(), 333u);
  EXPECT_EQ(Backend(kFirstByTier)->OwnedKeyCount(), 0u);
  EXPECT_EQ(Backend(kSecondByTier)->OwnedKeyCount(), 0u);
}

TEST_F(PeerServiceDispatchTest, EvictOfAnAbsentKeyReportsZeroBytes) {
  ::umbp::EvictKeyRequest req;
  req.add_keys("absent");
  ::umbp::EvictKeyResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->EvictKey(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.evicted_size(), 1);
  EXPECT_EQ(resp.evicted(0).bytes_freed(), 0u);
}

// ---- A registry-less peer still answers -------------------------------------

TEST(PeerServiceNoRegistry, AnswersWithoutCrashing) {
  PeerServiceServer server(nullptr);
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<int> pick(20000, 60000);
  uint16_t bound = 0;
  for (int attempt = 0; attempt < 50 && bound == 0; ++attempt) {
    const uint16_t port = static_cast<uint16_t>(pick(rng));
    if (server.Start(port)) bound = port;
  }
  ASSERT_NE(bound, 0);

  auto channel =
      grpc::CreateChannel("127.0.0.1:" + std::to_string(bound), grpc::InsecureChannelCredentials());
  auto stub = ::umbp::UMBPPeer::NewStub(channel);

  ::umbp::AllocateSlotRequest alloc_req;
  alloc_req.set_key("k");
  alloc_req.set_size(8);
  alloc_req.set_tier(::umbp::TIER_DRAM);
  ::umbp::AllocateSlotResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  ASSERT_TRUE(stub->AllocateSlot(&alloc_ctx, alloc_req, &alloc_resp).ok());
  EXPECT_EQ(alloc_resp.outcome(), ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED);

  ::umbp::ResolveKeyRequest resolve_req;
  resolve_req.set_key("k");
  ::umbp::ResolveKeyResponse resolve_resp;
  grpc::ClientContext resolve_ctx;
  ASSERT_TRUE(stub->ResolveKey(&resolve_ctx, resolve_req, &resolve_resp).ok());
  EXPECT_FALSE(resolve_resp.found());

  server.Stop();
}

// ---- One peer service owns a port -------------------------------------------
//
// gRPC enables SO_REUSEPORT by default for TCP servers on Linux, so before
// PeerServiceServer::Start passed GRPC_ARG_ALLOW_REUSEPORT=0 a SECOND server
// bound the same port successfully and the kernel split incoming connections
// between the two.  A client dialing that address was then answered, some
// fraction of the time, by a peer service with entirely different backends
// registered — a wrong answer, not a connection error.
//
// That is a production hazard (two UMBP processes sharing a peer port silently
// mis-serve each other's traffic) and it was also the cause of the umbp suite's
// random cross-test flakiness: PeerServiceDispatchTest::SetUp picks a random
// port in [20000,60000) and relies on Start() failing to detect a collision, so
// with reuseport on it never retried and instead talked to the wrong server.
TEST(PeerServicePortExclusivity, SecondServerCannotBindTheSamePort) {
  BackendRegistry registry_a;
  BackendRegistry registry_b;
  PeerServiceServer first(&registry_a);
  PeerServiceServer second(&registry_b);

  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<int> pick(20000, 60000);
  uint16_t port = 0;
  for (int attempt = 0; attempt < 50 && port == 0; ++attempt) {
    const uint16_t candidate = static_cast<uint16_t>(pick(rng));
    if (first.Start(candidate)) port = candidate;
  }
  ASSERT_NE(port, 0) << "could not bind any port for the first server";

  EXPECT_FALSE(second.Start(port))
      << "a second peer service bound port " << port
      << " — SO_REUSEPORT is on, so clients would be split between two servers";

  second.Stop();
  first.Stop();
}

}  // namespace
}  // namespace mori::umbp
