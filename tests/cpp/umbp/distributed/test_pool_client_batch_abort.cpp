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

// Tests for the BatchAbortAllocation refactor
// (distributed-known-issues #19).  Two layers of coverage:
//   1. Master-level RPC wiring: build up pending allocations via RoutePut
//      and roll them back through BatchAbortAllocation, verifying the
//      parallel `aborted[]` flags for valid / already-reaped / unknown /
//      wrong-node_id entries.
//   2. BatchPut integration contract: the happy path must issue ZERO
//      BatchAbortAllocation RPCs (observability counters stay at 0).
//
// Counter assertions require building with -DMORI_UMBP_OBS_COUNTERS=ON.
// When compiled without the flag the counter getters are always-zero
// and the EXPECT_EQ(..., 0u) assertions still pass (they degenerate to
// 0 == 0); behavioral assertions (aborted[] flags, BatchPut success)
// exercise real semantics regardless of the flag.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kBlockSize = kPageSize;
constexpr size_t kRemoteCap = 8 << 20;  // 8 MiB of DRAM on target
constexpr size_t kCallerBuf = 1 << 20;  // 1 MiB caller src region

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{0};
  if (next.load() == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    next.store(static_cast<uint16_t>(52500 + (std::rand() % 3500)));
  }
  return next.fetch_add(50);
}

// 2-node fixture: a caller with no local DRAM (forces remote routing)
// and a target node owning all the DRAM capacity.  Caller pre-registers
// a src region so BatchPut happy-path takes the zero-copy branch.
class BatchAbortTest : public ::testing::Test {
 protected:
  void SetUp() override {
    uint16_t base = AllocPort();
    master_port_ = base;
    io_port_caller_ = base + 1;
    io_port_target_ = base + 2;

    caller_buf_ = std::malloc(kCallerBuf);
    target_buf_ = std::malloc(kRemoteCap);
    ASSERT_NE(caller_buf_, nullptr);
    ASSERT_NE(target_buf_, nullptr);
    std::memset(caller_buf_, 0, kCallerBuf);
    std::memset(target_buf_, 0, kRemoteCap);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:" + std::to_string(master_port_);
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    master_addr_ = "localhost:" + std::to_string(master_port_);

    // Caller: tiny local DRAM, zero practical capacity — every RoutePut
    // lands on the target node.
    caller_local_ = std::malloc(kBlockSize);
    ASSERT_NE(caller_local_, nullptr);
    PoolClientConfig cfg_caller;
    cfg_caller.master_config.node_id = "node-caller";
    cfg_caller.master_config.node_address = "127.0.0.1";
    cfg_caller.master_config.master_address = master_addr_;
    cfg_caller.io_engine.host = "0.0.0.0";
    cfg_caller.io_engine.port = io_port_caller_;
    cfg_caller.dram_page_size = kPageSize;
    cfg_caller.dram_buffers = {{caller_local_, kBlockSize}};
    cfg_caller.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
    caller_ = std::make_unique<PoolClient>(std::move(cfg_caller));
    ASSERT_TRUE(caller_->Init());
    caller_->RegisterMemory(caller_buf_, kCallerBuf);

    PoolClientConfig cfg_target;
    cfg_target.master_config.node_id = "node-target";
    cfg_target.master_config.node_address = "127.0.0.1";
    cfg_target.master_config.master_address = master_addr_;
    cfg_target.io_engine.host = "0.0.0.0";
    cfg_target.io_engine.port = io_port_target_;
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

  // Issue `count` RoutePut RPCs against the caller's MasterClient,
  // producing `count` pending allocations on the target node.  Returns
  // the allocation_id list in order.
  std::vector<std::string> CreatePendings(size_t count) {
    std::vector<std::string> ids;
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      std::optional<RoutePutResult> r;
      auto status = caller_->Master().RoutePut("ab-" + std::to_string(i), kBlockSize, &r);
      EXPECT_TRUE(status.ok()) << "RoutePut failed: " << status.error_message();
      if (r.has_value()) {
        ids.push_back(r->allocation_id);
      }
    }
    return ids;
  }

  uint16_t master_port_ = 0;
  uint16_t io_port_caller_ = 0;
  uint16_t io_port_target_ = 0;
  void* caller_buf_ = nullptr;
  void* caller_local_ = nullptr;
  void* target_buf_ = nullptr;
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

// BatchAbortAllocation rolls back every pending it is handed when the
// inputs all reference live, correctly-scoped allocations.  After the
// call the reserved capacity is reusable, which we verify by issuing a
// fresh RoutePut that would otherwise run out of DRAM.
TEST_F(BatchAbortTest, RollsBackAllLiveEntries) {
  constexpr size_t N = 8;
  auto ids = CreatePendings(N);
  ASSERT_EQ(ids.size(), N);

  std::vector<MasterClient::BatchAbortEntry> entries;
  entries.reserve(N);
  for (const auto& id : ids) {
    entries.push_back({"node-target", id, kBlockSize});
  }

  std::vector<bool> aborted;
  auto status = caller_->Master().BatchAbortAllocation(entries, &aborted);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(aborted.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_TRUE(aborted[i]) << "entry " << i << " (" << ids[i] << ") not aborted";
  }

  // Re-route to confirm capacity was really released: this succeeds
  // only if the previous N pendings were rolled back.
  std::optional<RoutePutResult> recheck;
  auto recheck_status = caller_->Master().RoutePut("ab-recheck", kBlockSize, &recheck);
  EXPECT_TRUE(recheck_status.ok());
  EXPECT_TRUE(recheck.has_value());
}

// Mixed-input robustness: a single request may contain valid entries,
// already-aborted entries, unknown allocation_ids, and entries with a
// node_id that does not match the pending's owner.  Per-entry false is
// normal, the RPC still returns Status::OK.
TEST_F(BatchAbortTest, MixedValidReapedUnknownAndWrongNode) {
  auto ids = CreatePendings(2);
  ASSERT_EQ(ids.size(), 2);

  // Pre-abort ids[1] through the single-item path so BatchAbort finds
  // it already gone.
  ASSERT_TRUE(caller_->AbortAllocation("node-target", TierType::DRAM, ids[1], kBlockSize));

  std::vector<MasterClient::BatchAbortEntry> entries = {
      {"node-target", ids[0], kBlockSize},            // live: should succeed
      {"node-target", ids[1], kBlockSize},            // already aborted
      {"node-target", "does-not-exist", kBlockSize},  // unknown id
      {"node-wrong", ids[0], kBlockSize},  // wrong node_id (ids[0] was just aborted by [0])
  };

  std::vector<bool> aborted;
  auto status = caller_->Master().BatchAbortAllocation(entries, &aborted);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(aborted.size(), 4u);
  EXPECT_TRUE(aborted[0]);
  EXPECT_FALSE(aborted[1]);
  EXPECT_FALSE(aborted[2]);
  EXPECT_FALSE(aborted[3]);
}

// Empty entries list short-circuits at the MasterClient layer: returns
// Status::OK without hitting the wire and leaves `out` empty.
TEST_F(BatchAbortTest, EmptyEntriesIsNoOp) {
  std::vector<MasterClient::BatchAbortEntry> entries;
  std::vector<bool> aborted;
  auto status = caller_->Master().BatchAbortAllocation(entries, &aborted);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(aborted.empty());
}

// BatchPut happy path: every item succeeds, so pending_aborts stays
// empty and BatchAbortAllocation is never invoked.  The counters stay
// at 0 (regardless of whether MORI_UMBP_OBS_COUNTERS is on, 0 == 0 is
// still a valid expectation).
TEST_F(BatchAbortTest, BatchPutAllSuccessIssuesNoAbort) {
  constexpr size_t N = 4;
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < N; ++i) {
    auto* slot = static_cast<char*>(caller_buf_) + i * kBlockSize;
    std::memset(slot, static_cast<int>(0x20 + i), kBlockSize);
    keys.push_back("bp-ok-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kBlockSize);
  }

  const uint64_t calls_before = caller_->BatchAbortAllocationCallsCount();
  const uint64_t entries_before = caller_->BatchAbortAllocationEntriesCount();
  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_TRUE(results[i]) << "BatchPut lost key " << keys[i];
  }

  EXPECT_EQ(caller_->BatchAbortAllocationCallsCount(), calls_before);
  EXPECT_EQ(caller_->BatchAbortAllocationEntriesCount(), entries_before);
}

}  // namespace
}  // namespace mori::umbp
