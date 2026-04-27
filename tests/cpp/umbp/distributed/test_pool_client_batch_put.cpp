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

// Tests for the batch-level "src not registered" WARN added by
// distributed-known-issues.md #12.  Three scenarios:
//   1. Cross-node BatchPut with un-registered caller src → exactly one
//      batch-level WARN per PoolClient instance, suppressed for 60s on
//      subsequent batches.
//   2. Cross-node BatchPut with registered caller src → no batch-level
//      WARN (zero-copy path).
//   3. Same-node BatchPut (local memcpy branch) with un-registered src →
//      no batch-level WARN (the message is gated on the remote else
//      branch where staging fallback applies).
//
// All assertions filter spdlog output by the substring
// "BatchPut: src not registered for key=" so we don't conflate with the
// per-call WARN inside RemoteDramScatterWrite.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "spdlog/sinks/base_sink.h"
#include "spdlog/spdlog.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kPerKey = kPageSize;
constexpr size_t kCallerBuf = 1 << 20;
constexpr size_t kRemoteCap = 8 << 20;

constexpr const char* kBatchPutWarnSubstr = "BatchPut: src not registered for key=";

// Minimal sink that copies every payload into a vector for later
// substring inspection.  Derived from base_sink (mt) so it is safe to
// share across the umbp logger's worker.
class CapturingSink : public spdlog::sinks::base_sink<std::mutex> {
 public:
  std::vector<std::string> Lines() {
    std::lock_guard<std::mutex> lock(mu_);
    return lines_;
  }

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    std::lock_guard<std::mutex> lock(mu_);
    lines_.emplace_back(msg.payload.data(), msg.payload.size());
  }
  void flush_() override {}

 private:
  std::mutex mu_;
  std::vector<std::string> lines_;
};

// Attaches a CapturingSink to the umbp logger for the helper's lifetime
// and restores the original log level on destruction.
class UmbpLogCapture {
 public:
  UmbpLogCapture() {
    logger_ = mori::ModuleLogger::GetInstance().GetLogger(mori::modules::UMBP);
    saved_level_ = logger_->level();
    // Force WARN-level so MORI_UMBP_WARN reaches sinks regardless of any
    // env override that may have left the module at ERROR.
    logger_->set_level(spdlog::level::warn);
    sink_ = std::make_shared<CapturingSink>();
    sink_->set_level(spdlog::level::warn);
    logger_->sinks().push_back(sink_);
  }

  ~UmbpLogCapture() {
    auto& sinks = logger_->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
    logger_->set_level(saved_level_);
  }

  size_t CountSubstring(const std::string& needle) const {
    auto lines = sink_->Lines();
    size_t n = 0;
    for (const auto& line : lines) {
      if (line.find(needle) != std::string::npos) ++n;
    }
    return n;
  }

 private:
  std::shared_ptr<spdlog::logger> logger_;
  std::shared_ptr<CapturingSink> sink_;
  spdlog::level::level_enum saved_level_ = spdlog::level::off;
};

// 2-node fixture: caller pinned to a tiny local DRAM (forces remote
// routing) + target with full 8 MiB.  Caller owns two src buffers:
// caller_buf_ (ad-hoc malloc) and registered_buf_ (registered with
// caller_->RegisterMemory).  Tests choose which one to feed BatchPut.
class BatchPutWarnTest : public ::testing::Test {
 protected:
  void SetUp() override {
    caller_buf_ = std::malloc(kCallerBuf);
    registered_buf_ = std::malloc(kCallerBuf);
    target_buf_ = std::malloc(kRemoteCap);
    caller_local_ = std::malloc(kPageSize);
    ASSERT_NE(caller_buf_, nullptr);
    ASSERT_NE(registered_buf_, nullptr);
    ASSERT_NE(target_buf_, nullptr);
    ASSERT_NE(caller_local_, nullptr);
    std::memset(caller_buf_, 0, kCallerBuf);
    std::memset(registered_buf_, 0, kCallerBuf);
    std::memset(target_buf_, 0, kRemoteCap);
    std::memset(caller_local_, 0, kPageSize);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0) << "Master failed to start";
    const std::string master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    // Caller: 1 page of local DRAM ~= zero practical capacity, so any
    // multi-key BatchPut is forced to route to the target node.  We do
    // NOT call RegisterMemory in SetUp; tests opt in by registering
    // registered_buf_ explicitly.
    PoolClientConfig cfg_caller;
    cfg_caller.master_config.node_id = "node-caller";
    cfg_caller.master_config.node_address = "127.0.0.1";
    cfg_caller.master_config.master_address = master_addr;
    cfg_caller.io_engine.host = "0.0.0.0";
    cfg_caller.io_engine.port = 0;
    cfg_caller.dram_page_size = kPageSize;
    cfg_caller.dram_buffers = {{caller_local_, kPageSize}};
    cfg_caller.tier_capacities = {{TierType::DRAM, {kPageSize, kPageSize}}};
    caller_ = std::make_unique<PoolClient>(std::move(cfg_caller));
    ASSERT_TRUE(caller_->Init());

    PoolClientConfig cfg_target;
    cfg_target.master_config.node_id = "node-target";
    cfg_target.master_config.node_address = "127.0.0.1";
    cfg_target.master_config.master_address = master_addr;
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
    std::free(registered_buf_);
    std::free(caller_local_);
    std::free(target_buf_);
  }

  // Build a batch backed by `base` (one slot per page).  Caller chooses
  // whether `base` is a registered region (zero-copy) or a fresh malloc
  // (staging fallback).
  void MakeBatch(void* base, size_t n, std::vector<std::string>* keys,
                 std::vector<const void*>* srcs, std::vector<size_t>* sizes,
                 const std::string& key_prefix) {
    keys->clear();
    srcs->clear();
    sizes->clear();
    for (size_t i = 0; i < n; ++i) {
      auto* slot = static_cast<char*>(base) + i * kPerKey;
      std::memset(slot, static_cast<int>(0x10 + i), kPerKey);
      keys->push_back(key_prefix + std::to_string(i));
      srcs->push_back(slot);
      sizes->push_back(kPerKey);
    }
  }

  void* caller_buf_ = nullptr;
  void* registered_buf_ = nullptr;
  void* target_buf_ = nullptr;
  void* caller_local_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

// Un-registered src on a remote-bound batch: exactly one batch-level
// WARN, then a second BatchPut on the same PoolClient is silent (60s
// throttle).  Per-call WARNs from RemoteDramScatterWrite are deliberately
// not counted (substring filter).
TEST_F(BatchPutWarnTest, StagingFallbackEmitsWarnOnceThenThrottles) {
  UmbpLogCapture cap;

  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  MakeBatch(caller_buf_, /*n=*/3, &keys, &srcs, &sizes, "stg-");

  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), 3u);
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_TRUE(results[i]) << "key=" << keys[i];
  }
  EXPECT_EQ(cap.CountSubstring(kBatchPutWarnSubstr), 1u)
      << "Expected exactly one batch-level WARN for the first un-registered "
         "BatchPut (per-batch suppression after first hit).";

  std::vector<std::string> keys2;
  std::vector<const void*> srcs2;
  std::vector<size_t> sizes2;
  MakeBatch(caller_buf_, /*n=*/3, &keys2, &srcs2, &sizes2, "stg2-");
  auto results2 = caller_->BatchPut(keys2, srcs2, sizes2);
  ASSERT_EQ(results2.size(), 3u);
  EXPECT_EQ(cap.CountSubstring(kBatchPutWarnSubstr), 1u)
      << "Second BatchPut within 60s must NOT emit another batch-level WARN.";
}

// Registered caller src goes through the zero-copy branch — no batch
// WARN should fire at any point.
TEST_F(BatchPutWarnTest, RegisteredSrcsNoWarn) {
  ASSERT_TRUE(caller_->RegisterMemory(registered_buf_, kCallerBuf));

  UmbpLogCapture cap;

  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  MakeBatch(registered_buf_, /*n=*/4, &keys, &srcs, &sizes, "zc-");

  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), 4u);
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_TRUE(results[i]) << "key=" << keys[i];
  }
  EXPECT_EQ(cap.CountSubstring(kBatchPutWarnSubstr), 0u);

  caller_->DeregisterMemory(registered_buf_);
}

// All-LOCAL BatchPut: even if srcs are un-registered, the WARN must
// stay silent because the local memcpy branch does not exercise
// FindRegisteredMemory and the staging fallback never applies.  The
// fixture forces this by having target_ run BatchPut on itself.
TEST_F(BatchPutWarnTest, AllLocalBatchNoWarn) {
  UmbpLogCapture cap;

  // Un-registered ad-hoc slot (not in any PoolClient::RegisterMemory
  // region) to confirm the WARN path is gated on the remote else branch
  // and not on the registration check alone.  vector<char> owns the
  // storage so an early gtest ASSERT does not leak it.
  std::vector<char> unreg(4 * kPerKey, 0);
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < 4; ++i) {
    auto* slot = unreg.data() + i * kPerKey;
    std::memset(slot, static_cast<int>(0x70 + i), kPerKey);
    keys.push_back("local-" + std::to_string(i));
    srcs.push_back(slot);
    sizes.push_back(kPerKey);
  }

  auto results = target_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), 4u);
  size_t ok = 0;
  for (bool b : results) {
    if (b) ++ok;
  }
  EXPECT_GT(ok, 0u);
  EXPECT_EQ(cap.CountSubstring(kBatchPutWarnSubstr), 0u)
      << "Local-route BatchPut must not emit the batch-level WARN even with "
         "un-registered srcs (the WARN lives in the remote else branch only).";
}

}  // namespace
}  // namespace mori::umbp
