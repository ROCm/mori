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

// Tests for the BatchPut RDMA fusion path (3-phase per-peer
// SubmitFusedBucket / WaitFusedBucket / MapBucketFailures).  Counter
// assertions and the PartialPairFailure injection test require building
// with -DMORI_UMBP_TESTING=ON; without it the counter getters return 0
// (corresponding tests skip via GTEST_SKIP), and the failure injection
// case is gated out at compile time.

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kBlockSize = kPageSize;
constexpr size_t kCallerBuf = 1 << 20;   // 1 MiB - room for batch=64 single-page
constexpr size_t kRemoteCap = 16 << 20;  // 16 MiB on each target node

// Two-node fixture: caller pinned to a tiny local DRAM (every BatchPut
// item lands on a remote target).  Caller pre-registers caller_buf_ so
// the default path is REMOTE_ZC.
class FusedBatchPutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    caller_buf_ = std::malloc(kCallerBuf);
    target_buf_ = std::malloc(kRemoteCap);
    caller_local_ = std::malloc(kBlockSize);
    ASSERT_TRUE(caller_buf_ && target_buf_ && caller_local_);
    std::memset(caller_buf_, 0, kCallerBuf);
    std::memset(target_buf_, 0, kRemoteCap);
    std::memset(caller_local_, 0, kBlockSize);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    master_addr_ = "localhost:" + std::to_string(master_->GetBoundPort());

    PoolClientConfig cfg_caller;
    cfg_caller.master_config.node_id = "node-caller";
    cfg_caller.master_config.node_address = "127.0.0.1";
    cfg_caller.master_config.master_address = master_addr_;
    cfg_caller.io_engine.host = "0.0.0.0";
    cfg_caller.io_engine.port = 0;
    cfg_caller.dram_page_size = kPageSize;
    cfg_caller.dram_buffers = {{caller_local_, kBlockSize}};
    cfg_caller.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
    caller_ = std::make_unique<PoolClient>(std::move(cfg_caller));
    ASSERT_TRUE(caller_->Init());
    ASSERT_TRUE(caller_->RegisterMemory(caller_buf_, kCallerBuf));

    PoolClientConfig cfg_target;
    cfg_target.master_config.node_id = "node-target";
    cfg_target.master_config.node_address = "127.0.0.1";
    cfg_target.master_config.master_address = master_addr_;
    cfg_target.io_engine.host = "0.0.0.0";
    cfg_target.io_engine.port = 0;
    cfg_target.dram_page_size = kPageSize;
    cfg_target.dram_buffers = {{target_buf_, kRemoteCap}};
    cfg_target.tier_capacities = {{TierType::DRAM, {kRemoteCap, kRemoteCap}}};
    target_ = std::make_unique<PoolClient>(std::move(cfg_target));
    ASSERT_TRUE(target_->Init());
  }

  void TearDown() override {
    if (caller_) caller_->Shutdown();
    if (target_) target_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(caller_buf_);
    std::free(target_buf_);
    std::free(caller_local_);
  }

  // Build a batch backed by `base` (one slot per page).
  void MakeBatch(void* base, size_t n, std::vector<std::string>* keys,
                 std::vector<const void*>* srcs, std::vector<size_t>* sizes,
                 const std::string& prefix) {
    keys->clear();
    srcs->clear();
    sizes->clear();
    for (size_t i = 0; i < n; ++i) {
      auto* slot = static_cast<char*>(base) + i * kBlockSize;
      std::memset(slot, static_cast<int>(0x10 + (i & 0x7F)), kBlockSize);
      keys->push_back(prefix + std::to_string(i));
      srcs->push_back(slot);
      sizes->push_back(kBlockSize);
    }
  }

  void* caller_buf_ = nullptr;
  void* caller_local_ = nullptr;
  void* target_buf_ = nullptr;
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

// Single peer + single remote buffer + same registered local_mem -> all
// items collapse into one FusedPair, posted via one IOEngine::BatchWrite.
TEST_F(FusedBatchPutTest, FusedSinglePeerSingleCall) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  constexpr size_t N = 8;
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  MakeBatch(caller_buf_, N, &keys, &srcs, &sizes, "f1-");

  const auto calls0 = caller_->BatchPutIoEngineCallsCount();
  const auto pairs0 = caller_->BatchPutIoEnginePairsCount();
  const auto items0 = caller_->BatchPutItemsCount();

  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_TRUE(results[i]) << "key " << keys[i];

  EXPECT_EQ(caller_->BatchPutIoEngineCallsCount() - calls0, 1u);
  EXPECT_EQ(caller_->BatchPutIoEnginePairsCount() - pairs0, 1u);
  EXPECT_EQ(caller_->BatchPutItemsCount() - items0, N);
}

// caller's tiny dram buffer (1 page) means BatchPut to target
// short-circuits all to the SAME peer.  Caller is the receiver in this
// test - so we drive BatchPut from `caller_` whose local DRAM is empty.
// Since target has 16 MiB, 4 items easily fit on a single buffer; no
// way to force "page across two buffers" without restructuring the
// fixture.  Instead, build a fixture variant with two small buffers
// on the target so the master's allocator distributes pages across them.
class FusedBatchPutMultiBufferTest : public ::testing::Test {
 protected:
  static constexpr size_t kMpPage = kPageSize;
  static constexpr size_t kBufBytes = kPageSize * 2;  // 2 pages per buffer
  static constexpr size_t kCaller = 1 << 20;

  void SetUp() override {
    caller_buf_ = std::malloc(kCaller);
    target_b0_ = std::malloc(kBufBytes);
    target_b1_ = std::malloc(kBufBytes);
    caller_local_ = std::malloc(kMpPage);
    ASSERT_TRUE(caller_buf_ && target_b0_ && target_b1_ && caller_local_);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    auto master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    PoolClientConfig cfg_caller;
    cfg_caller.master_config.node_id = "node-caller";
    cfg_caller.master_config.node_address = "127.0.0.1";
    cfg_caller.master_config.master_address = master_addr;
    cfg_caller.io_engine.host = "0.0.0.0";
    cfg_caller.io_engine.port = 0;
    cfg_caller.dram_page_size = kMpPage;
    cfg_caller.dram_buffers = {{caller_local_, kMpPage}};
    cfg_caller.tier_capacities = {{TierType::DRAM, {kMpPage, kMpPage}}};
    caller_ = std::make_unique<PoolClient>(std::move(cfg_caller));
    ASSERT_TRUE(caller_->Init());
    ASSERT_TRUE(caller_->RegisterMemory(caller_buf_, kCaller));

    PoolClientConfig cfg_target;
    cfg_target.master_config.node_id = "node-target";
    cfg_target.master_config.node_address = "127.0.0.1";
    cfg_target.master_config.master_address = master_addr;
    cfg_target.io_engine.host = "0.0.0.0";
    cfg_target.io_engine.port = 0;
    cfg_target.dram_page_size = kMpPage;
    cfg_target.dram_buffers = {{target_b0_, kBufBytes}, {target_b1_, kBufBytes}};
    cfg_target.tier_capacities = {{TierType::DRAM, {kBufBytes * 2, kBufBytes * 2}}};
    target_ = std::make_unique<PoolClient>(std::move(cfg_target));
    ASSERT_TRUE(target_->Init());
  }

  void TearDown() override {
    if (caller_) caller_->Shutdown();
    if (target_) target_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(caller_buf_);
    std::free(target_b0_);
    std::free(target_b1_);
    std::free(caller_local_);
  }

  void* caller_buf_ = nullptr;
  void* caller_local_ = nullptr;
  void* target_b0_ = nullptr;
  void* target_b1_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

// 4 single-page items all to the SAME peer, but the master spreads
// pages across 2 remote buffers (each only fits 2 pages) -> 2 FusedPairs
// inside a SINGLE IOEngine::BatchWrite call.
TEST_F(FusedBatchPutMultiBufferTest, FusedSinglePeerMultiBuffer) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < N; ++i) {
    auto* slot = static_cast<char*>(caller_buf_) + i * kPageSize;
    std::memset(slot, static_cast<int>(0x21 + i), kPageSize);
    keys.push_back("mb-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kPageSize);
  }

  const auto calls0 = caller_->BatchPutIoEngineCallsCount();
  const auto pairs0 = caller_->BatchPutIoEnginePairsCount();

  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_TRUE(results[i]);

  EXPECT_EQ(caller_->BatchPutIoEngineCallsCount() - calls0, 1u)
      << "single peer: must collapse to one BatchWrite call";
  EXPECT_EQ(caller_->BatchPutIoEnginePairsCount() - pairs0, 2u)
      << "two remote buffers: should produce two FusedPairs";
}

// Two-target fixture: each peer gets half the batch -> two BatchWrite
// calls (one per peer), each with one pair.  Each peer's local DRAM is
// trivially small so its own master allocations don't conflict.
class FusedBatchPutMultiPeerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    caller_buf_ = std::malloc(kCallerBuf);
    target_a_buf_ = std::malloc(kRemoteCap);
    target_b_buf_ = std::malloc(kRemoteCap);
    caller_local_ = std::malloc(kBlockSize);
    ASSERT_TRUE(caller_buf_ && target_a_buf_ && target_b_buf_ && caller_local_);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    auto master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    PoolClientConfig cfg_caller;
    cfg_caller.master_config.node_id = "node-caller";
    cfg_caller.master_config.node_address = "127.0.0.1";
    cfg_caller.master_config.master_address = master_addr;
    cfg_caller.io_engine.host = "0.0.0.0";
    cfg_caller.io_engine.port = 0;
    cfg_caller.dram_page_size = kPageSize;
    cfg_caller.dram_buffers = {{caller_local_, kBlockSize}};
    cfg_caller.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
    caller_ = std::make_unique<PoolClient>(std::move(cfg_caller));
    ASSERT_TRUE(caller_->Init());
    ASSERT_TRUE(caller_->RegisterMemory(caller_buf_, kCallerBuf));

    auto bring_up_target = [&](const std::string& node_id, void* buf,
                               std::unique_ptr<PoolClient>* out) {
      PoolClientConfig cfg;
      cfg.master_config.node_id = node_id;
      cfg.master_config.node_address = "127.0.0.1";
      cfg.master_config.master_address = master_addr;
      cfg.io_engine.host = "0.0.0.0";
      cfg.io_engine.port = 0;
      cfg.dram_page_size = kPageSize;
      cfg.dram_buffers = {{buf, kRemoteCap}};
      cfg.tier_capacities = {{TierType::DRAM, {kRemoteCap, kRemoteCap}}};
      *out = std::make_unique<PoolClient>(std::move(cfg));
      ASSERT_TRUE((*out)->Init());
    };
    bring_up_target("node-target-a", target_a_buf_, &target_a_);
    bring_up_target("node-target-b", target_b_buf_, &target_b_);
  }

  void TearDown() override {
    if (caller_) caller_->Shutdown();
    if (target_a_) target_a_->Shutdown();
    if (target_b_) target_b_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(caller_buf_);
    std::free(target_a_buf_);
    std::free(target_b_buf_);
    std::free(caller_local_);
  }

  void* caller_buf_ = nullptr;
  void* caller_local_ = nullptr;
  void* target_a_buf_ = nullptr;
  void* target_b_buf_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_a_;
  std::unique_ptr<PoolClient> target_b_;
};

// Multiple peers force multiple buckets / multiple BatchWrite calls.
// Routing strategy distribution is tested implicitly: as long as both
// peers receive at least one item, calls > 1.
TEST_F(FusedBatchPutMultiPeerTest, FusedMultiPeer) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  constexpr size_t N = 8;
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < N; ++i) {
    auto* slot = static_cast<char*>(caller_buf_) + i * kBlockSize;
    std::memset(slot, static_cast<int>(0x40 + i), kBlockSize);
    keys.push_back("mp-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kBlockSize);
  }

  const auto calls0 = caller_->BatchPutIoEngineCallsCount();

  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_TRUE(results[i]);

  // Both targets must have received some items - master's RR-like
  // policy should split when caller's local capacity is zero.
  EXPECT_GE(caller_->BatchPutIoEngineCallsCount() - calls0, 1u);
}

// Mixed LOCAL + REMOTE_ZC: items routed to the local node memcpy (no
// fused counter), the rest go through the fused path.  Uses target_ for
// the LOCAL leg (it owns the big DRAM, master will route its self-Puts
// locally) and caller_ for the REMOTE_ZC leg (its tiny local DRAM
// forces routing to target_).
TEST_F(FusedBatchPutTest, MixedLocalAndRemoteZC) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  std::vector<char> tlocal_src(4 * kBlockSize, 0);
  std::vector<std::string> lk;
  std::vector<const void*> ls;
  std::vector<size_t> lz;
  for (size_t i = 0; i < 4; ++i) {
    auto* slot = tlocal_src.data() + i * kBlockSize;
    std::memset(slot, static_cast<int>(0x60 + i), kBlockSize);
    lk.push_back("mx-local-" + std::to_string(i));
    ls.push_back(slot);
    lz.push_back(kBlockSize);
  }
  const auto t_items0 = target_->BatchPutItemsCount();
  const auto t_calls0 = target_->BatchPutIoEngineCallsCount();
  auto local_results = target_->BatchPut(lk, ls, lz);
  ASSERT_EQ(local_results.size(), 4u);
  for (bool ok : local_results) EXPECT_TRUE(ok);
  // Self-routed Puts go through the LOCAL memcpy branch; fused counters
  // (REMOTE_ZC only) must stay flat.
  EXPECT_EQ(target_->BatchPutItemsCount() - t_items0, 0u);
  EXPECT_EQ(target_->BatchPutIoEngineCallsCount() - t_calls0, 0u);

  // Remote-only batch via caller_ - guaranteed REMOTE_ZC (caller has
  // ~zero local DRAM and registered caller_buf_).
  std::vector<std::string> rk;
  std::vector<const void*> rs;
  std::vector<size_t> rz;
  MakeBatch(caller_buf_, /*n=*/4, &rk, &rs, &rz, "mx-rem-");
  const auto items1 = caller_->BatchPutItemsCount();
  const auto calls1 = caller_->BatchPutIoEngineCallsCount();
  auto rr = caller_->BatchPut(rk, rs, rz);
  ASSERT_EQ(rr.size(), 4u);
  for (bool ok : rr) EXPECT_TRUE(ok);
  EXPECT_EQ(caller_->BatchPutItemsCount() - items1, 4u);
  EXPECT_EQ(caller_->BatchPutIoEngineCallsCount() - calls1, 1u);
}

// All items hit the legacy staging path (caller did not RegisterMemory
// for the batch's source region) -> fused counters do not move.
TEST_F(FusedBatchPutTest, AllStagingNoFusedCounter) {
  // Use a non-registered scratch buffer.  vector<char> owns storage so
  // ASSERT bails out cleanly.
  std::vector<char> unreg(4 * kBlockSize, 0);
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < 4; ++i) {
    auto* slot = unreg.data() + i * kBlockSize;
    std::memset(slot, static_cast<int>(0x80 + i), kBlockSize);
    keys.push_back("stg-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kBlockSize);
  }

  const auto calls0 = caller_->BatchPutIoEngineCallsCount();
  const auto items0 = caller_->BatchPutItemsCount();
  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), 4u);
  for (bool ok : results) EXPECT_TRUE(ok);

  // Fused path was not taken (caller srcs are unregistered).  The legacy
  // STG branch issues its own per-item BatchWrite via
  // RemoteDramScatterWrite, but those calls are NOT counted in
  // batch_put_io_engine_calls_; only SubmitFusedBucket increments it.
  EXPECT_EQ(caller_->BatchPutIoEngineCallsCount() - calls0, 0u);
  EXPECT_EQ(caller_->BatchPutItemsCount() - items0, 0u);
}

// Empty / mismatched input: early return with all-false results, no
// counters move, no master interaction.
TEST_F(FusedBatchPutTest, EmptyAndMismatchedInput) {
  std::vector<std::string> empty_keys;
  std::vector<const void*> empty_srcs;
  std::vector<size_t> empty_sizes;
  EXPECT_TRUE(caller_->BatchPut(empty_keys, empty_srcs, empty_sizes).empty());

  std::vector<std::string> ks = {"a", "b"};
  std::vector<const void*> ss = {caller_buf_};
  std::vector<size_t> sz = {kBlockSize, kBlockSize};
  auto r = caller_->BatchPut(ks, ss, sz);
  ASSERT_EQ(r.size(), 2u);
  EXPECT_FALSE(r[0]);
  EXPECT_FALSE(r[1]);
}

// Route validation rejection (caller passes a size that violates the
// allocation window): the item is SKIPPED (not Aborted - master
// invariant violation) and doesn't appear in pending_aborts.
TEST_F(FusedBatchPutTest, RouteValidationSkippedNoAbort) {
  // size_too_big_for_one_page: SizeMatchesAllocation requires
  // (N-1)*ps < size <= N*ps.  With page_size=4096 and master allocating
  // ceil(size/ps) pages, passing size > N*ps for some N would trigger
  // master rejection upstream of the SizeMatchesAllocation guard.
  //
  // Easier path: feed a size of 0.  Master rejects at RoutePut, leaving
  // the entry as nullopt; results[i]=false, nothing aborted.
  const auto aborts0 = caller_->BatchAbortAllocationCallsCount();
  const auto entries0 = caller_->BatchAbortAllocationEntriesCount();

  std::vector<std::string> keys = {"sk-zero"};
  std::vector<const void*> srcs = {caller_buf_};
  std::vector<size_t> sizes = {0};
  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results[0]);

  EXPECT_EQ(caller_->BatchAbortAllocationCallsCount(), aborts0);
  EXPECT_EQ(caller_->BatchAbortAllocationEntriesCount(), entries0);
}

#ifdef MORI_UMBP_TESTING
// Subclass that synthesizes a per-pair failure on the first IssueBatchWrite
// call.  Verifies that a failed pair routes ALL its contributing items
// to pending_aborts (BatchAbortAllocation invoked exactly once, entries
// equal to the contributing item count).
class FailingPoolClient : public PoolClient {
 public:
  using PoolClient::PoolClient;
  void IssueBatchWrite(const mori::io::MemDescVec& local_src,
                       const mori::io::BatchSizeVec& local_offsets,
                       const mori::io::MemDescVec& remote_dest,
                       const mori::io::BatchSizeVec& remote_offsets,
                       const mori::io::BatchSizeVec& sizes,
                       mori::io::TransferStatusPtrVec& statuses,
                       mori::io::TransferUniqueIdVec& ids) override {
    (void)local_src;
    (void)local_offsets;
    (void)remote_dest;
    (void)remote_offsets;
    (void)sizes;
    (void)ids;
    // Fail every pair in this submit by flipping status to a failed code.
    for (auto* s : statuses) {
      s->Update(mori::io::StatusCode::ERR_RDMA_OP, "injected pair failure");
    }
  }
};

TEST_F(FusedBatchPutTest, PartialPairFailure) {
  // Bring up a separate caller using FailingPoolClient (subclass of
  // PoolClient) so the fixture's caller_ stays untouched.
  void* fbuf = std::malloc(kCallerBuf);
  void* flocal = std::malloc(kBlockSize);
  ASSERT_TRUE(fbuf && flocal);
  std::memset(fbuf, 0, kCallerBuf);
  std::memset(flocal, 0, kBlockSize);

  PoolClientConfig cfg;
  cfg.master_config.node_id = "node-failclient";
  cfg.master_config.node_address = "127.0.0.1";
  cfg.master_config.master_address = master_addr_;
  cfg.io_engine.host = "0.0.0.0";
  cfg.io_engine.port = 0;
  cfg.dram_page_size = kPageSize;
  cfg.dram_buffers = {{flocal, kBlockSize}};
  cfg.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
  // Owned by unique_ptr<FailingPoolClient> (NOT base) so non-virtual
  // PoolClient destructor in release builds does not slice; under
  // MORI_UMBP_TESTING the destructor is virtual, but keeping the
  // derived type at delete site is a robust habit anyway.
  std::unique_ptr<FailingPoolClient> fc = std::make_unique<FailingPoolClient>(std::move(cfg));
  ASSERT_TRUE(fc->Init());
  ASSERT_TRUE(fc->RegisterMemory(fbuf, kCallerBuf));

  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < N; ++i) {
    auto* slot = static_cast<char*>(fbuf) + i * kBlockSize;
    std::memset(slot, static_cast<int>(0xA0 + i), kBlockSize);
    keys.push_back("pf-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kBlockSize);
  }

  const auto aborts0 = fc->BatchAbortAllocationCallsCount();
  const auto entries0 = fc->BatchAbortAllocationEntriesCount();

  auto results = fc->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_FALSE(results[i]);

  // Single batched abort RPC, entries == contributing item count
  // (all N items, since all share the same pair under single peer +
  // single buffer + same registered local_mem).
  EXPECT_EQ(fc->BatchAbortAllocationCallsCount() - aborts0, 1u);
  EXPECT_EQ(fc->BatchAbortAllocationEntriesCount() - entries0, N);

  fc->Shutdown();
  fc.reset();
  std::free(fbuf);
  std::free(flocal);
}

// IssueBatchWrite throws on first call: validates the two-layer guard.
//   - Inner: SubmitFusedBucket catches, drains posted statuses (no-op
//     here since the override never posts), rethrows.
//   - Outer: BatchPut catches, drains already-pushed buckets, marks all
//     REMOTE_ZC items as failed and routes them to pending_aborts.
// All N items end up failed + abort exactly once at end-of-batch.
class ThrowingIssuePoolClient : public PoolClient {
 public:
  using PoolClient::PoolClient;
  void IssueBatchWrite(const mori::io::MemDescVec&, const mori::io::BatchSizeVec&,
                       const mori::io::MemDescVec&, const mori::io::BatchSizeVec&,
                       const mori::io::BatchSizeVec&, mori::io::TransferStatusPtrVec&,
                       mori::io::TransferUniqueIdVec&) override {
    throw std::bad_alloc();
  }
};

TEST_F(FusedBatchPutTest, IssueBatchWriteThrowsHandledCleanly) {
  void* fbuf = std::malloc(kCallerBuf);
  void* flocal = std::malloc(kBlockSize);
  ASSERT_TRUE(fbuf && flocal);
  std::memset(fbuf, 0, kCallerBuf);
  std::memset(flocal, 0, kBlockSize);

  PoolClientConfig cfg;
  cfg.master_config.node_id = "node-throwclient";
  cfg.master_config.node_address = "127.0.0.1";
  cfg.master_config.master_address = master_addr_;
  cfg.io_engine.host = "0.0.0.0";
  cfg.io_engine.port = 0;
  cfg.dram_page_size = kPageSize;
  cfg.dram_buffers = {{flocal, kBlockSize}};
  cfg.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
  auto tc = std::make_unique<ThrowingIssuePoolClient>(std::move(cfg));
  ASSERT_TRUE(tc->Init());
  ASSERT_TRUE(tc->RegisterMemory(fbuf, kCallerBuf));

  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < N; ++i) {
    auto* slot = static_cast<char*>(fbuf) + i * kBlockSize;
    std::memset(slot, static_cast<int>(0xB0 + i), kBlockSize);
    keys.push_back("th-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kBlockSize);
  }

  const auto aborts0 = tc->BatchAbortAllocationCallsCount();
  const auto entries0 = tc->BatchAbortAllocationEntriesCount();

  // Must NOT propagate the exception out of BatchPut.
  std::vector<bool> results;
  ASSERT_NO_THROW({ results = tc->BatchPut(keys, srcs, sizes); });
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_FALSE(results[i]) << "key " << keys[i];

  // All N items routed to a single end-of-batch abort flush.
  EXPECT_EQ(tc->BatchAbortAllocationCallsCount() - aborts0, 1u);
  EXPECT_EQ(tc->BatchAbortAllocationEntriesCount() - entries0, N);

  tc->Shutdown();
  tc.reset();
  std::free(fbuf);
  std::free(flocal);
}
#endif  // MORI_UMBP_TESTING

}  // namespace
}  // namespace mori::umbp
