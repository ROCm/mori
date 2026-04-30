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

// Tests for the BatchGet RDMA fusion path (3-phase per-peer
// SubmitFusedBucketRead / WaitFusedBucket / MapBucketFailuresRead).
// Mirrors test_pool_client_batch_put_fused.cpp; counter assertions and
// the PartialPairFailure injection test require -DMORI_UMBP_TESTING=ON.

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
constexpr size_t kCallerBuf = 1 << 20;   // 1 MiB caller dst region (registered for ZC)
constexpr size_t kRemoteCap = 16 << 20;  // 16 MiB on each source (peer) node

// Two-node fixture mirror of FusedBatchPutTest.  The "caller" runs
// BatchGet, the "target" holds the source data (Put-then-Get).  Caller
// has tiny local DRAM so every Put would route to target, then BatchGet
// reads back via REMOTE_ZC.
class FusedBatchGetTest : public ::testing::Test {
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
    // Target also needs a registered region to BatchPut from (its
    // src region for the seed Put step).  Use target_seed_ (separate
    // allocation so it doesn't overlap target_buf_, which is the
    // master-managed remote DRAM exportable region).
    target_seed_ = std::malloc(kCallerBuf);
    ASSERT_TRUE(target_seed_);
    std::memset(target_seed_, 0, kCallerBuf);
    ASSERT_TRUE(target_->RegisterMemory(target_seed_, kCallerBuf));
  }

  void TearDown() override {
    if (caller_) caller_->Shutdown();
    if (target_) target_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(caller_buf_);
    std::free(target_buf_);
    std::free(target_seed_);
    std::free(caller_local_);
  }

  // Seed `n` keys onto `target_` so BatchGet from `caller_` finds them.
  // Returns the canonical byte pattern used per item so tests can
  // verify dst content.
  void SeedKeys(size_t n, const std::string& prefix, std::vector<std::string>* keys,
                std::vector<size_t>* sizes) {
    keys->clear();
    sizes->clear();
    std::vector<const void*> srcs;
    for (size_t i = 0; i < n; ++i) {
      auto* slot = static_cast<char*>(target_seed_) + i * kBlockSize;
      std::memset(slot, static_cast<int>(0x10 + (i & 0x7F)), kBlockSize);
      keys->push_back(prefix + std::to_string(i));
      srcs.push_back(slot);
      sizes->push_back(kBlockSize);
    }
    auto put_results = target_->BatchPut(*keys, srcs, *sizes);
    ASSERT_EQ(put_results.size(), n);
    for (size_t i = 0; i < n; ++i) ASSERT_TRUE(put_results[i]) << "seed put for " << (*keys)[i];
  }

  // Verify caller_buf_ slots match the canonical seed pattern.
  void VerifyDstPattern(size_t n) {
    for (size_t i = 0; i < n; ++i) {
      const auto* slot = static_cast<const char*>(caller_buf_) + i * kBlockSize;
      const char expect = static_cast<char>(0x10 + (i & 0x7F));
      for (size_t b = 0; b < kBlockSize; ++b) {
        ASSERT_EQ(slot[b], expect) << "mismatch at item " << i << " byte " << b;
      }
    }
  }

  void* caller_buf_ = nullptr;    // dst region; registered for ZC
  void* caller_local_ = nullptr;  // tiny local DRAM (forces remote routing)
  void* target_buf_ = nullptr;    // target's exportable DRAM (master-managed)
  void* target_seed_ = nullptr;   // target's src for seed BatchPut
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

// Single peer + single remote buffer + same registered local_mem -> all
// items collapse into one FusedPair, posted via one IOEngine::BatchRead.
TEST_F(FusedBatchGetTest, FusedSinglePeerSingleCall) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  constexpr size_t N = 8;
  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  SeedKeys(N, "g1-", &keys, &sizes);

  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) {
    dsts[i] = static_cast<char*>(caller_buf_) + i * kBlockSize;
  }

  const auto calls0 = caller_->BatchGetIoEngineCallsCount();
  const auto pairs0 = caller_->BatchGetIoEnginePairsCount();
  const auto items0 = caller_->BatchGetItemsCount();

  auto results = caller_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_TRUE(results[i]) << "key " << keys[i];

  EXPECT_EQ(caller_->BatchGetIoEngineCallsCount() - calls0, 1u);
  EXPECT_EQ(caller_->BatchGetIoEnginePairsCount() - pairs0, 1u);
  EXPECT_EQ(caller_->BatchGetItemsCount() - items0, N);
  VerifyDstPattern(N);
}

// batch=1 must produce identical observable behavior to the single-Get
// path: results=[true], dst byte-matches.  Counter values are NOT
// asserted (single-key Get bypasses IoEngine::BatchRead, but the fused
// BatchGet with batch=1 still hits IssueBatchRead - so we only assert
// the externally visible parity: success + dst bytes).
TEST_F(FusedBatchGetTest, BatchEqualsOneIsLegacyParity) {
  constexpr size_t N = 1;
  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  SeedKeys(N, "b1-", &keys, &sizes);
  std::vector<void*> dsts = {caller_buf_};
  auto results = caller_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0]);
  VerifyDstPattern(N);
}

// 4 single-page items all to the SAME peer, but the master spreads
// pages across 2 remote buffers (each only fits 2 pages) -> 2 FusedPairs
// inside a SINGLE IOEngine::BatchRead call.
class FusedBatchGetMultiBufferTest : public ::testing::Test {
 protected:
  static constexpr size_t kMpPage = kPageSize;
  static constexpr size_t kBufBytes = kPageSize * 2;
  static constexpr size_t kCaller = 1 << 20;

  void SetUp() override {
    caller_buf_ = std::malloc(kCaller);
    target_b0_ = std::malloc(kBufBytes);
    target_b1_ = std::malloc(kBufBytes);
    caller_local_ = std::malloc(kMpPage);
    target_seed_ = std::malloc(kCaller);
    ASSERT_TRUE(caller_buf_ && target_b0_ && target_b1_ && caller_local_ && target_seed_);
    std::memset(caller_buf_, 0, kCaller);
    std::memset(target_seed_, 0, kCaller);

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
    ASSERT_TRUE(target_->RegisterMemory(target_seed_, kCaller));
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
    std::free(target_seed_);
  }

  void* caller_buf_ = nullptr;
  void* caller_local_ = nullptr;
  void* target_b0_ = nullptr;
  void* target_b1_ = nullptr;
  void* target_seed_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

TEST_F(FusedBatchGetMultiBufferTest, FusedSinglePeerMultiBuffer) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  constexpr size_t N = 4;
  // Seed first.
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < N; ++i) {
    auto* slot = static_cast<char*>(target_seed_) + i * kPageSize;
    std::memset(slot, static_cast<int>(0x21 + i), kPageSize);
    keys.push_back("mbg-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kPageSize);
  }
  auto put_r = target_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(put_r.size(), N);
  for (bool ok : put_r) ASSERT_TRUE(ok);

  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) dsts[i] = static_cast<char*>(caller_buf_) + i * kPageSize;

  const auto calls0 = caller_->BatchGetIoEngineCallsCount();
  const auto pairs0 = caller_->BatchGetIoEnginePairsCount();

  auto results = caller_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_TRUE(results[i]);

  EXPECT_EQ(caller_->BatchGetIoEngineCallsCount() - calls0, 1u)
      << "single peer: must collapse to one BatchRead call";
  EXPECT_EQ(caller_->BatchGetIoEnginePairsCount() - pairs0, 2u)
      << "two remote buffers: should produce two FusedPairs";

  // Verify dst content matches.
  for (size_t i = 0; i < N; ++i) {
    const auto* slot = static_cast<const char*>(caller_buf_) + i * kPageSize;
    const char expect = static_cast<char>(0x21 + i);
    for (size_t b = 0; b < kPageSize; ++b) {
      ASSERT_EQ(slot[b], expect) << "item " << i << " byte " << b;
    }
  }
}

// Two-source-peer fixture: each peer holds half the seed -> two BatchRead
// calls (one per peer).
class FusedBatchGetMultiPeerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    caller_buf_ = std::malloc(kCallerBuf);
    target_a_buf_ = std::malloc(kRemoteCap);
    target_b_buf_ = std::malloc(kRemoteCap);
    caller_local_ = std::malloc(kBlockSize);
    target_a_seed_ = std::malloc(kCallerBuf);
    target_b_seed_ = std::malloc(kCallerBuf);
    ASSERT_TRUE(caller_buf_ && target_a_buf_ && target_b_buf_ && caller_local_ && target_a_seed_ &&
                target_b_seed_);
    std::memset(caller_buf_, 0, kCallerBuf);
    std::memset(target_a_seed_, 0, kCallerBuf);
    std::memset(target_b_seed_, 0, kCallerBuf);

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

    auto bring_up_target = [&](const std::string& node_id, void* buf, void* seed,
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
      ASSERT_TRUE((*out)->RegisterMemory(seed, kCallerBuf));
    };
    bring_up_target("node-target-a", target_a_buf_, target_a_seed_, &target_a_);
    bring_up_target("node-target-b", target_b_buf_, target_b_seed_, &target_b_);
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
    std::free(target_a_seed_);
    std::free(target_b_seed_);
    std::free(caller_local_);
  }

  void* caller_buf_ = nullptr;
  void* caller_local_ = nullptr;
  void* target_a_buf_ = nullptr;
  void* target_b_buf_ = nullptr;
  void* target_a_seed_ = nullptr;
  void* target_b_seed_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_a_;
  std::unique_ptr<PoolClient> target_b_;
};

TEST_F(FusedBatchGetMultiPeerTest, FusedMultiPeer) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  constexpr size_t N = 8;
  // Seed alternating items to a/b targets so master records each on
  // the corresponding peer.  We force placement by Putting from each
  // target itself (self-Put -> LOCAL on that target -> next BatchGet
  // from caller_ pulls remote).
  std::vector<std::string> keys_a, keys_b;
  std::vector<const void*> srcs_a, srcs_b;
  std::vector<size_t> sizes_a, sizes_b;
  for (size_t i = 0; i < N / 2; ++i) {
    auto* slot_a = static_cast<char*>(target_a_seed_) + i * kBlockSize;
    auto* slot_b = static_cast<char*>(target_b_seed_) + i * kBlockSize;
    std::memset(slot_a, static_cast<int>(0x40 + i), kBlockSize);
    std::memset(slot_b, static_cast<int>(0x50 + i), kBlockSize);
    keys_a.push_back("mpg-a-" + std::to_string(i));
    keys_b.push_back("mpg-b-" + std::to_string(i));
    srcs_a.push_back(slot_a);
    srcs_b.push_back(slot_b);
    sizes_a.push_back(kBlockSize);
    sizes_b.push_back(kBlockSize);
  }
  auto pa = target_a_->BatchPut(keys_a, srcs_a, sizes_a);
  auto pb = target_b_->BatchPut(keys_b, srcs_b, sizes_b);
  for (bool ok : pa) ASSERT_TRUE(ok);
  for (bool ok : pb) ASSERT_TRUE(ok);

  std::vector<std::string> all_keys;
  std::vector<size_t> all_sizes;
  std::vector<void*> all_dsts;
  for (size_t i = 0; i < N / 2; ++i) {
    all_keys.push_back(keys_a[i]);
    all_sizes.push_back(kBlockSize);
    all_dsts.push_back(static_cast<char*>(caller_buf_) + i * kBlockSize);
  }
  for (size_t i = 0; i < N / 2; ++i) {
    all_keys.push_back(keys_b[i]);
    all_sizes.push_back(kBlockSize);
    all_dsts.push_back(static_cast<char*>(caller_buf_) + (N / 2 + i) * kBlockSize);
  }

  const auto calls0 = caller_->BatchGetIoEngineCallsCount();

  auto results = caller_->BatchGet(all_keys, all_dsts, all_sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_TRUE(results[i]) << "key " << all_keys[i];

  // Two source peers -> two BatchRead calls.
  EXPECT_EQ(caller_->BatchGetIoEngineCallsCount() - calls0, 2u);
}

// Mixed LOCAL + REMOTE_ZC: items on caller's own node hit memcpy
// branch (no fused counter), remote items go through fused.
TEST_F(FusedBatchGetTest, MixedLocalAndRemoteZC) {
#ifndef MORI_UMBP_TESTING
  GTEST_SKIP() << "requires -DMORI_UMBP_TESTING=ON for fused counter assertions";
#endif
  // Local-on-target: target_->BatchPut will route locally (self-node).
  // Then target_->BatchGet on the same keys is a same-node Get -> LOCAL
  // branch in BatchGet, no fused counter increment.
  constexpr size_t kLocal = 4;
  std::vector<std::string> lk;
  std::vector<size_t> lz;
  std::vector<const void*> lsrcs;
  for (size_t i = 0; i < kLocal; ++i) {
    auto* slot = static_cast<char*>(target_seed_) + i * kBlockSize;
    std::memset(slot, static_cast<int>(0x60 + i), kBlockSize);
    lk.push_back("mxg-local-" + std::to_string(i));
    lz.push_back(kBlockSize);
    lsrcs.push_back(slot);
  }
  auto put_r = target_->BatchPut(lk, lsrcs, lz);
  for (bool ok : put_r) ASSERT_TRUE(ok);

  // Fresh dst on target's own buffer (use offset past seeded region).
  std::vector<char> ldst(kLocal * kBlockSize, 0);
  std::vector<void*> ldsts(kLocal);
  for (size_t i = 0; i < kLocal; ++i) ldsts[i] = ldst.data() + i * kBlockSize;
  const auto t_items0 = target_->BatchGetItemsCount();
  const auto t_calls0 = target_->BatchGetIoEngineCallsCount();
  auto local_results = target_->BatchGet(lk, ldsts, lz);
  ASSERT_EQ(local_results.size(), kLocal);
  for (bool ok : local_results) EXPECT_TRUE(ok);
  // Same-node Get goes LOCAL branch; fused counters stay flat.
  EXPECT_EQ(target_->BatchGetItemsCount() - t_items0, 0u);
  EXPECT_EQ(target_->BatchGetIoEngineCallsCount() - t_calls0, 0u);

  // Remote-only via caller_ (data lives on target_).
  std::vector<std::string> rk;
  std::vector<size_t> rz;
  SeedKeys(/*n=*/4, "mxg-rem-", &rk, &rz);
  std::vector<void*> rdsts(4);
  for (size_t i = 0; i < 4; ++i) rdsts[i] = static_cast<char*>(caller_buf_) + i * kBlockSize;
  const auto items1 = caller_->BatchGetItemsCount();
  const auto calls1 = caller_->BatchGetIoEngineCallsCount();
  auto rr = caller_->BatchGet(rk, rdsts, rz);
  ASSERT_EQ(rr.size(), 4u);
  for (bool ok : rr) EXPECT_TRUE(ok);
  EXPECT_EQ(caller_->BatchGetItemsCount() - items1, 4u);
  EXPECT_EQ(caller_->BatchGetIoEngineCallsCount() - calls1, 1u);
}

// caller did NOT register caller_buf_ for the dst region -> all items
// fall back to STG path; fused counters do not move.  (Use a fresh
// PoolClient so we don't mutate the fixture's caller_ which is
// pre-registered.)
TEST_F(FusedBatchGetTest, AllStagingNoFusedCounter) {
  // Bring up an unregistered-dst caller alongside the fixture's
  // caller_/target_.
  void* nbuf = std::malloc(kCallerBuf);
  void* nlocal = std::malloc(kBlockSize);
  ASSERT_TRUE(nbuf && nlocal);
  std::memset(nbuf, 0, kCallerBuf);
  std::memset(nlocal, 0, kBlockSize);

  PoolClientConfig cfg;
  cfg.master_config.node_id = "node-stgcaller";
  cfg.master_config.node_address = "127.0.0.1";
  cfg.master_config.master_address = master_addr_;
  cfg.io_engine.host = "0.0.0.0";
  cfg.io_engine.port = 0;
  cfg.dram_page_size = kPageSize;
  cfg.dram_buffers = {{nlocal, kBlockSize}};
  cfg.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
  auto stg_caller = std::make_unique<PoolClient>(std::move(cfg));
  ASSERT_TRUE(stg_caller->Init());
  // Deliberately do NOT call RegisterMemory on nbuf -> ZC find will miss
  // and STG path takes over.

  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  SeedKeys(N, "stgg-", &keys, &sizes);
  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) dsts[i] = static_cast<char*>(nbuf) + i * kBlockSize;

  const auto calls0 = stg_caller->BatchGetIoEngineCallsCount();
  const auto items0 = stg_caller->BatchGetItemsCount();
  auto results = stg_caller->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(results.size(), N);
  for (bool ok : results) EXPECT_TRUE(ok);

  // Fused path was not taken (dsts unregistered).  STG path internally
  // does its own per-item BatchRead via RemoteDramScatterRead, but those
  // are NOT counted in batch_get_io_engine_calls_; only
  // SubmitFusedBucketRead increments it.
  EXPECT_EQ(stg_caller->BatchGetIoEngineCallsCount() - calls0, 0u);
  EXPECT_EQ(stg_caller->BatchGetItemsCount() - items0, 0u);

  // Verify dst content (STG path must still deliver correctly).
  for (size_t i = 0; i < N; ++i) {
    const auto* slot = static_cast<const char*>(nbuf) + i * kBlockSize;
    const char expect = static_cast<char>(0x10 + (i & 0x7F));
    for (size_t b = 0; b < kBlockSize; ++b) {
      ASSERT_EQ(slot[b], expect) << "stg item " << i << " byte " << b;
    }
  }

  stg_caller->Shutdown();
  stg_caller.reset();
  std::free(nbuf);
  std::free(nlocal);
}

// One item passes a wrong size (sizes[i] != stored loc.size).  That
// item must be SKIPPED (results[i]=false); other items in the same
// batch must succeed unchanged.
TEST_F(FusedBatchGetTest, SizeMismatchPerItemSkipped) {
  constexpr size_t N = 3;
  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  SeedKeys(N, "smm-", &keys, &sizes);

  // Mutate item 1's size to something != stored size.
  std::vector<size_t> bad_sizes = sizes;
  bad_sizes[1] = kBlockSize - 1;

  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) dsts[i] = static_cast<char*>(caller_buf_) + i * kBlockSize;

  auto results = caller_->BatchGet(keys, dsts, bad_sizes);
  ASSERT_EQ(results.size(), N);
  EXPECT_TRUE(results[0]);
  EXPECT_FALSE(results[1]) << "size-mismatched item must be SKIPPED, not amplify";
  EXPECT_TRUE(results[2]);

  // Verify item 0 and 2 have correct content; item 1 dst is undefined
  // per legacy contract (we don't assert anything on it).
  const char ex0 = static_cast<char>(0x10);
  const char ex2 = static_cast<char>(0x12);
  const auto* slot0 = static_cast<const char*>(caller_buf_);
  const auto* slot2 = static_cast<const char*>(caller_buf_) + 2 * kBlockSize;
  for (size_t b = 0; b < kBlockSize; ++b) ASSERT_EQ(slot0[b], ex0);
  for (size_t b = 0; b < kBlockSize; ++b) ASSERT_EQ(slot2[b], ex2);
}

// Empty / mismatched input: early return with all-false results.
TEST_F(FusedBatchGetTest, EmptyAndMismatchedInput) {
  std::vector<std::string> empty_keys;
  std::vector<void*> empty_dsts;
  std::vector<size_t> empty_sizes;
  EXPECT_TRUE(caller_->BatchGet(empty_keys, empty_dsts, empty_sizes).empty());

  std::vector<std::string> ks = {"a", "b"};
  std::vector<void*> ds = {caller_buf_};  // mismatched size
  std::vector<size_t> sz = {kBlockSize, kBlockSize};
  auto r = caller_->BatchGet(ks, ds, sz);
  ASSERT_EQ(r.size(), 2u);
  EXPECT_FALSE(r[0]);
  EXPECT_FALSE(r[1]);
}

// Lookup miss: keys never Put.  routes[i] is nullopt for each one;
// per-item false, no counter movement, no exception.
TEST_F(FusedBatchGetTest, RouteMissReturnsAllFalseNoFusedCounter) {
  constexpr size_t N = 3;
  std::vector<std::string> keys = {"missing-1", "missing-2", "missing-3"};
  std::vector<size_t> sizes = {kBlockSize, kBlockSize, kBlockSize};
  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) dsts[i] = static_cast<char*>(caller_buf_) + i * kBlockSize;

  const auto calls0 = caller_->BatchGetIoEngineCallsCount();
  const auto items0 = caller_->BatchGetItemsCount();
  auto results = caller_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(results.size(), N);
  for (bool ok : results) EXPECT_FALSE(ok);

  EXPECT_EQ(caller_->BatchGetIoEngineCallsCount() - calls0, 0u);
  EXPECT_EQ(caller_->BatchGetItemsCount() - items0, 0u);
}

#ifdef MORI_UMBP_TESTING
// Subclass that synthesizes a per-pair Read failure on the first
// IssueBatchRead call.  Verifies that a failed pair routes ALL its
// contributing items to results[i]=false, **without** any Master Abort
// RPC (Get has no reservation to abort).
class FailingReadPoolClient : public PoolClient {
 public:
  using PoolClient::PoolClient;
  void IssueBatchRead(const mori::io::MemDescVec&, const mori::io::BatchSizeVec&,
                      const mori::io::MemDescVec&, const mori::io::BatchSizeVec&,
                      const mori::io::BatchSizeVec&, mori::io::TransferStatusPtrVec& statuses,
                      mori::io::TransferUniqueIdVec&) override {
    for (auto* s : statuses) {
      s->Update(mori::io::StatusCode::ERR_RDMA_OP, "injected pair read failure");
    }
  }
};

TEST_F(FusedBatchGetTest, PartialPairFailure) {
  // Seed first via the normal target_.
  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  SeedKeys(N, "ppf-", &keys, &sizes);

  // Build a separate caller with the failing seam to drive the BatchGet.
  void* fbuf = std::malloc(kCallerBuf);
  void* flocal = std::malloc(kBlockSize);
  ASSERT_TRUE(fbuf && flocal);
  std::memset(fbuf, 0, kCallerBuf);
  std::memset(flocal, 0, kBlockSize);

  PoolClientConfig cfg;
  cfg.master_config.node_id = "node-failreader";
  cfg.master_config.node_address = "127.0.0.1";
  cfg.master_config.master_address = master_addr_;
  cfg.io_engine.host = "0.0.0.0";
  cfg.io_engine.port = 0;
  cfg.dram_page_size = kPageSize;
  cfg.dram_buffers = {{flocal, kBlockSize}};
  cfg.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
  auto fr = std::make_unique<FailingReadPoolClient>(std::move(cfg));
  ASSERT_TRUE(fr->Init());
  ASSERT_TRUE(fr->RegisterMemory(fbuf, kCallerBuf));

  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) dsts[i] = static_cast<char*>(fbuf) + i * kBlockSize;

  const auto aborts0 = fr->BatchAbortAllocationCallsCount();
  const auto entries0 = fr->BatchAbortAllocationEntriesCount();

  auto results = fr->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_FALSE(results[i]) << "key " << keys[i];

  // Get has NO Abort RPC tail - even though all N items failed.
  EXPECT_EQ(fr->BatchAbortAllocationCallsCount(), aborts0);
  EXPECT_EQ(fr->BatchAbortAllocationEntriesCount(), entries0);

  fr->Shutdown();
  fr.reset();
  std::free(fbuf);
  std::free(flocal);
}

// IssueBatchRead throws on first call: validates the two-layer guard
// (inner SubmitFusedBucketRead drains posted statuses + rethrows; outer
// BatchGet catches, drains already-pushed buckets, marks all REMOTE_ZC
// items false).  No exception escapes BatchGet; no Abort RPC.
class ThrowingReadPoolClient : public PoolClient {
 public:
  using PoolClient::PoolClient;
  void IssueBatchRead(const mori::io::MemDescVec&, const mori::io::BatchSizeVec&,
                      const mori::io::MemDescVec&, const mori::io::BatchSizeVec&,
                      const mori::io::BatchSizeVec&, mori::io::TransferStatusPtrVec&,
                      mori::io::TransferUniqueIdVec&) override {
    throw std::bad_alloc();
  }
};

TEST_F(FusedBatchGetTest, IssueBatchReadThrowsHandledCleanly) {
  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  SeedKeys(N, "thg-", &keys, &sizes);

  void* fbuf = std::malloc(kCallerBuf);
  void* flocal = std::malloc(kBlockSize);
  ASSERT_TRUE(fbuf && flocal);
  std::memset(fbuf, 0, kCallerBuf);
  std::memset(flocal, 0, kBlockSize);

  PoolClientConfig cfg;
  cfg.master_config.node_id = "node-throwreader";
  cfg.master_config.node_address = "127.0.0.1";
  cfg.master_config.master_address = master_addr_;
  cfg.io_engine.host = "0.0.0.0";
  cfg.io_engine.port = 0;
  cfg.dram_page_size = kPageSize;
  cfg.dram_buffers = {{flocal, kBlockSize}};
  cfg.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
  auto tr = std::make_unique<ThrowingReadPoolClient>(std::move(cfg));
  ASSERT_TRUE(tr->Init());
  ASSERT_TRUE(tr->RegisterMemory(fbuf, kCallerBuf));

  std::vector<void*> dsts(N);
  for (size_t i = 0; i < N; ++i) dsts[i] = static_cast<char*>(fbuf) + i * kBlockSize;

  const auto aborts0 = tr->BatchAbortAllocationCallsCount();

  std::vector<bool> results;
  ASSERT_NO_THROW({ results = tr->BatchGet(keys, dsts, sizes); });
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) EXPECT_FALSE(results[i]) << "key " << keys[i];

  // No Abort RPC.
  EXPECT_EQ(tr->BatchAbortAllocationCallsCount(), aborts0);

  tr->Shutdown();
  tr.reset();
  std::free(fbuf);
  std::free(flocal);
}
#endif  // MORI_UMBP_TESTING

}  // namespace
}  // namespace mori::umbp
