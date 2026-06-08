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
#include <gtest/gtest.h>

#include <atomic>
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

constexpr size_t kBufSize = 1 << 20;
constexpr size_t kBlockSize = 4096;

// Unique peer-service port per PoolClient.  A non-zero peer_service_port is
// required for the node to register a peer_address and accept remote
// AllocateSlot/CommitSlot RPCs; without it remote BatchPut fails with
// "peer service connection unavailable".
inline uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{52000};
  return next.fetch_add(1);
}

// The master index is eventually consistent: a committed key becomes visible
// only after the owning peer ships its ADD event on the next heartbeat.  Poll
// Exists() until the key shows up (or the timeout elapses) before asserting
// visibility / issuing a Get.
inline bool WaitForExists(PoolClient* client, const std::string& key,
                          std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (client->Exists(key)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return client->Exists(key);
}

class CrossNodeSmoke : public ::testing::Test {
 protected:
  void SetUp() override {
    buf_a_ = std::malloc(kBufSize);
    buf_b_ = std::malloc(kBufSize);
    caller_buf_ = std::malloc(kBufSize);
    read_buf_ = std::malloc(kBufSize);
    ASSERT_NE(buf_a_, nullptr);
    ASSERT_NE(buf_b_, nullptr);
    ASSERT_NE(caller_buf_, nullptr);
    ASSERT_NE(read_buf_, nullptr);
    std::memset(buf_a_, 0, kBufSize);
    std::memset(buf_b_, 0, kBufSize);
    std::memset(caller_buf_, 0, kBufSize);
    std::memset(read_buf_, 0, kBufSize);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    // Short heartbeat so committed keys propagate to the master index quickly
    // (event-driven index converges within ~one heartbeat interval).
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds{1};
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0) << "Master failed to start";

    std::string master_addr = "localhost:" + std::to_string(master_->GetBoundPort());

    // Phase 2: use a tiny page_size so the existing 4KB block size yields
    // exactly one page per Put without needing to size buffers up to the
    // 2 MiB Master default.  Both nodes must agree on page_size since the
    // Master records it per-node at registration time.
    PoolClientConfig cfg_a;
    cfg_a.master_config.node_id = "node-a";
    cfg_a.master_config.node_address = "127.0.0.1";
    cfg_a.master_config.master_address = master_addr;
    cfg_a.io_engine.host = "0.0.0.0";
    cfg_a.io_engine.port = 0;
    cfg_a.peer_service_port = NextPeerServicePort();
    cfg_a.dram_buffers = {{buf_a_, kBlockSize}};
    cfg_a.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
    cfg_a.dram_page_size = kBlockSize;
    client_a_ = std::make_unique<PoolClient>(std::move(cfg_a));
    ASSERT_TRUE(client_a_->Init());

    PoolClientConfig cfg_b;
    cfg_b.master_config.node_id = "node-b";
    cfg_b.master_config.node_address = "127.0.0.1";
    cfg_b.master_config.master_address = master_addr;
    cfg_b.io_engine.host = "0.0.0.0";
    cfg_b.io_engine.port = 0;
    cfg_b.peer_service_port = NextPeerServicePort();
    cfg_b.dram_buffers = {{buf_b_, kBufSize}};
    cfg_b.tier_capacities = {{TierType::DRAM, {kBufSize, kBufSize}}};
    cfg_b.dram_page_size = kBlockSize;
    client_b_ = std::make_unique<PoolClient>(std::move(cfg_b));
    ASSERT_TRUE(client_b_->Init());

    client_a_->RegisterMemory(caller_buf_, kBufSize);
    client_b_->RegisterMemory(read_buf_, kBufSize);
  }

  void TearDown() override {
    if (client_b_) client_b_->Shutdown();
    if (client_a_) client_a_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(buf_a_);
    std::free(buf_b_);
    std::free(caller_buf_);
    std::free(read_buf_);
  }

  void* buf_a_ = nullptr;
  void* buf_b_ = nullptr;
  void* caller_buf_ = nullptr;
  void* read_buf_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> client_a_;
  std::unique_ptr<PoolClient> client_b_;
};

TEST_F(CrossNodeSmoke, PutGetWithRDMA) {
  std::memset(caller_buf_, 0xAB, kBlockSize);

  ASSERT_TRUE(client_a_->Put("rdma-key", caller_buf_, kBlockSize));
  EXPECT_TRUE(WaitForExists(client_a_.get(), "rdma-key"));
  EXPECT_TRUE(WaitForExists(client_b_.get(), "rdma-key"));

  std::memset(read_buf_, 0, kBlockSize);
  ASSERT_TRUE(client_b_->Get("rdma-key", read_buf_, kBlockSize));
  EXPECT_EQ(std::memcmp(caller_buf_, read_buf_, kBlockSize), 0);
}

TEST_F(CrossNodeSmoke, BatchPutGetWithRDMA) {
  auto* src1 = static_cast<char*>(caller_buf_);
  auto* src2 = src1 + kBlockSize;
  auto* src3 = src2 + kBlockSize;
  std::memset(src1, 0x11, kBlockSize);
  std::memset(src2, 0x22, kBlockSize);
  std::memset(src3, 0x33, kBlockSize);

  std::vector<std::string> keys = {"bk1", "bk2", "bk3"};
  std::vector<const void*> srcs = {src1, src2, src3};
  std::vector<size_t> sizes = {kBlockSize, kBlockSize, kBlockSize};

  auto put_results = client_a_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(put_results.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(put_results[i]) << "put failed for " << keys[i];
  }

  for (const auto& key : keys) {
    EXPECT_TRUE(WaitForExists(client_a_.get(), key));
    EXPECT_TRUE(WaitForExists(client_b_.get(), key));
  }

  auto* dst1 = static_cast<char*>(read_buf_);
  auto* dst2 = dst1 + kBlockSize;
  auto* dst3 = dst2 + kBlockSize;
  std::memset(read_buf_, 0, kBlockSize * 3);

  std::vector<void*> dsts = {dst1, dst2, dst3};
  auto get_results = client_b_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(get_results.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(get_results[i]) << "get failed for " << keys[i];
  }
  EXPECT_EQ(std::memcmp(src1, dst1, kBlockSize), 0);
  EXPECT_EQ(std::memcmp(src2, dst2, kBlockSize), 0);
  EXPECT_EQ(std::memcmp(src3, dst3, kBlockSize), 0);
}

TEST_F(CrossNodeSmoke, FinalizeIdempotentE2E) {
  std::memset(caller_buf_, 0xCD, kBlockSize);
  ASSERT_TRUE(client_a_->Put("idem-key", caller_buf_, kBlockSize));
  EXPECT_TRUE(WaitForExists(client_a_.get(), "idem-key"));

  ASSERT_TRUE(client_a_->Put("idem-key", caller_buf_, kBlockSize));
  EXPECT_TRUE(WaitForExists(client_b_.get(), "idem-key"));

  std::memset(read_buf_, 0, kBlockSize);
  ASSERT_TRUE(client_b_->Get("idem-key", read_buf_, kBlockSize));
  EXPECT_EQ(std::memcmp(caller_buf_, read_buf_, kBlockSize), 0);
}

// ===========================================================================
// Phase 2 multi-page scatter-gather tests.  These exercise the
// RemoteDramScatterWrite/Read code path across the PageBitmapAllocator
// strategies:
//   1) same-buffer continuous run        -> MultiPageSameBufferPutGet
//   3) cross-buffer discrete pages       -> CrossBufferScatterPutGet
// Each test stands up its own master + two clients so it can choose the
// per-node buffer layout independently of the single-page fixture.
// ===========================================================================

class CrossNodeMultiPage : public ::testing::Test {
 protected:
  static constexpr size_t kPageSize = 4096;

  struct NodeSetup {
    // Each entry is a buffer size in bytes; the test allocates buffers of
    // exactly these sizes and registers them with the PoolClient.
    std::vector<size_t> buffer_sizes;
  };

  void StartMaster() {
    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds{1};
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0) << "Master failed to start";
  }

  void TearDown() override { TearDownClients(); }

  std::unique_ptr<PoolClient> MakeClient(const std::string& node_id, const NodeSetup& setup,
                                         std::vector<void*>* owned_bufs) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = "localhost:" + std::to_string(master_->GetBoundPort());
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = NextPeerServicePort();
    cfg.dram_page_size = kPageSize;
    uint64_t total = 0;
    for (size_t sz : setup.buffer_sizes) {
      void* p = std::malloc(sz);
      EXPECT_NE(p, nullptr);
      std::memset(p, 0, sz);
      owned_bufs->push_back(p);
      cfg.dram_buffers.push_back({p, sz});
      total += sz;
    }
    cfg.tier_capacities = {{TierType::DRAM, {total, total}}};
    auto cli = std::make_unique<PoolClient>(std::move(cfg));
    EXPECT_TRUE(cli->Init());
    return cli;
  }

  void TearDownClients() {
    if (client_a_) client_a_->Shutdown();
    if (client_b_) client_b_->Shutdown();
    client_a_.reset();
    client_b_.reset();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    master_.reset();
    for (void* p : owned_a_) std::free(p);
    for (void* p : owned_b_) std::free(p);
    owned_a_.clear();
    owned_b_.clear();
  }

  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> client_a_;
  std::unique_ptr<PoolClient> client_b_;
  std::vector<void*> owned_a_;
  std::vector<void*> owned_b_;
};

TEST_F(CrossNodeMultiPage, MultiPageSameBufferPutGet) {
  // Strategy 1: a single buffer with enough contiguous pages for a multi-
  // page block.  node-a is sized so node-b is the obvious most-available
  // target; we then PUT 3 pages from a and GET them back from a.
  StartMaster();
  client_a_ = MakeClient("node-a", NodeSetup{{kPageSize}}, &owned_a_);
  client_b_ = MakeClient("node-b", NodeSetup{{kPageSize * 4}}, &owned_b_);

  constexpr size_t kPayload = kPageSize * 3;
  std::vector<char> src(kPayload);
  for (size_t i = 0; i < kPayload; ++i) src[i] = static_cast<char>(i & 0xFF);

  ASSERT_TRUE(client_a_->Put("mp-same-buf", src.data(), kPayload));
  EXPECT_TRUE(WaitForExists(client_a_.get(), "mp-same-buf"));
  EXPECT_TRUE(WaitForExists(client_b_.get(), "mp-same-buf"));

  std::vector<char> dst(kPayload, 0);
  ASSERT_TRUE(client_a_->Get("mp-same-buf", dst.data(), kPayload));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kPayload), 0);
}

TEST_F(CrossNodeMultiPage, CrossBufferScatterPutGet) {
  // Strategy 3: target node with two single-page buffers.  Allocator must
  // fall through Strategy 1 + 2 (each buffer has only 1 page free, can't
  // satisfy 2-page request inside one buffer) and use cross-buffer scatter.
  StartMaster();
  // node-a is the source; sized so it cannot accept the Put itself
  // (single page) and routes go to node-b.
  client_a_ = MakeClient("node-a", NodeSetup{{kPageSize}}, &owned_a_);
  client_b_ = MakeClient("node-b", NodeSetup{{kPageSize, kPageSize}}, &owned_b_);

  constexpr size_t kPayload = kPageSize * 2;
  std::vector<char> src(kPayload);
  for (size_t i = 0; i < kPayload; ++i) src[i] = static_cast<char>((i * 7) & 0xFF);

  ASSERT_TRUE(client_a_->Put("xbuf-key", src.data(), kPayload));
  EXPECT_TRUE(WaitForExists(client_a_.get(), "xbuf-key"));
  EXPECT_TRUE(WaitForExists(client_b_.get(), "xbuf-key"));

  std::vector<char> dst(kPayload, 0);
  ASSERT_TRUE(client_a_->Get("xbuf-key", dst.data(), kPayload));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kPayload), 0);

  // Sanity: a second multi-page Put on top of an exhausted cross-buffer
  // pool fails (no pages left), surfacing the "no suitable target" path
  // through to the caller.
  EXPECT_FALSE(client_a_->Put("xbuf-overflow", src.data(), kPayload));
}

// ---------------------------------------------------------------------------
// Partial-tail tests.  Master allocates ceil(size / page_size) pages even
// when size is not page-aligned; the last page is partially filled.  These
// tests guard against silently truncating valid bytes or pulling stale
// tail bytes back into the caller's buffer.
// ---------------------------------------------------------------------------

TEST_F(CrossNodeMultiPage, PartialTailSinglePage) {
  // size < page_size: single allocation, partial tail = size.  Covers the
  // N=1 fast path through both Put/Get and the scatter helper.
  StartMaster();
  client_a_ = MakeClient("node-a", NodeSetup{{kPageSize / 2}}, &owned_a_);
  client_b_ = MakeClient("node-b", NodeSetup{{kPageSize}}, &owned_b_);

  constexpr size_t kPayload = 1234;  // arbitrary, < kPageSize
  std::vector<char> src(kPayload);
  for (size_t i = 0; i < kPayload; ++i) src[i] = static_cast<char>((i * 13 + 7) & 0xFF);

  ASSERT_TRUE(client_a_->Put("pt-1", src.data(), kPayload));
  ASSERT_TRUE(WaitForExists(client_a_.get(), "pt-1"));

  // Sentinel bytes after `dst` validate that Get does not write past
  // `size` (would catch a regression that copied a full page back).
  constexpr char kSentinel = 0x5A;
  std::vector<char> dst(kPayload + 64, kSentinel);
  ASSERT_TRUE(client_a_->Get("pt-1", dst.data(), kPayload));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kPayload), 0);
  for (size_t i = kPayload; i < dst.size(); ++i) {
    EXPECT_EQ(dst[i], kSentinel) << "Get wrote past requested size at offset " << i;
  }
}

TEST_F(CrossNodeMultiPage, PartialTailMultiPageSameBuffer) {
  // 2 full pages + a partial tail in a single contiguous buffer (Strategy 1).
  StartMaster();
  client_a_ = MakeClient("node-a", NodeSetup{{kPageSize / 2}}, &owned_a_);
  client_b_ = MakeClient("node-b", NodeSetup{{kPageSize * 4}}, &owned_b_);

  constexpr size_t kTail = 333;
  constexpr size_t kPayload = kPageSize * 2 + kTail;
  std::vector<char> src(kPayload);
  for (size_t i = 0; i < kPayload; ++i) src[i] = static_cast<char>((i * 31 + 1) & 0xFF);

  ASSERT_TRUE(client_a_->Put("pt-mp", src.data(), kPayload));
  ASSERT_TRUE(WaitForExists(client_a_.get(), "pt-mp"));

  constexpr char kSentinel = 0xA5;
  std::vector<char> dst(kPayload + 64, kSentinel);
  ASSERT_TRUE(client_a_->Get("pt-mp", dst.data(), kPayload));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kPayload), 0);
  for (size_t i = kPayload; i < dst.size(); ++i) {
    EXPECT_EQ(dst[i], kSentinel) << "Get wrote past requested size at offset " << i;
  }
}

TEST_F(CrossNodeMultiPage, PartialTailCrossBufferScatter) {
  // Strategy 3: target has two single-page buffers; payload = 1 page + tail
  // forces cross-buffer scatter where the *last* logical page (carrying the
  // partial tail) lands in a different buffer group than the first page.
  // Regression guard: scatter helpers must identify "last page" by spi
  // (global page index), not by the position inside a group.
  StartMaster();
  client_a_ = MakeClient("node-a", NodeSetup{{kPageSize / 2}}, &owned_a_);
  client_b_ = MakeClient("node-b", NodeSetup{{kPageSize, kPageSize}}, &owned_b_);

  constexpr size_t kTail = 777;
  constexpr size_t kPayload = kPageSize + kTail;
  std::vector<char> src(kPayload);
  for (size_t i = 0; i < kPayload; ++i) src[i] = static_cast<char>((i * 53 + 3) & 0xFF);

  ASSERT_TRUE(client_a_->Put("pt-xbuf", src.data(), kPayload));
  ASSERT_TRUE(WaitForExists(client_a_.get(), "pt-xbuf"));

  constexpr char kSentinel = 0x3C;
  std::vector<char> dst(kPayload + 64, kSentinel);
  ASSERT_TRUE(client_a_->Get("pt-xbuf", dst.data(), kPayload));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kPayload), 0);
  for (size_t i = kPayload; i < dst.size(); ++i) {
    EXPECT_EQ(dst[i], kSentinel) << "Get wrote past requested size at offset " << i;
  }
}

TEST_F(CrossNodeMultiPage, PartialTailGetSizeMismatchRejected) {
  // Contract: Get must reject `size != Location.size`.  Without this
  // check the partial-tail code path would either truncate valid bytes
  // (size < stored) or pull stale bytes from the unused tail of the last
  // page (size > stored, still inside the page window).
  StartMaster();
  client_a_ = MakeClient("node-a", NodeSetup{{kPageSize / 2}}, &owned_a_);
  client_b_ = MakeClient("node-b", NodeSetup{{kPageSize}}, &owned_b_);

  constexpr size_t kPayload = 999;
  std::vector<char> src(kPayload, 'X');
  ASSERT_TRUE(client_a_->Put("pt-mismatch", src.data(), kPayload));
  ASSERT_TRUE(WaitForExists(client_a_.get(), "pt-mismatch"));

  // Asking for the rounded-up cap (1 full page) is in the page window but
  // != stored size; must fail rather than leaking the unused tail bytes.
  std::vector<char> dst(kPageSize, 0);
  EXPECT_FALSE(client_a_->Get("pt-mismatch", dst.data(), kPageSize));
  // Asking for fewer bytes than stored must also fail (would truncate).
  EXPECT_FALSE(client_a_->Get("pt-mismatch", dst.data(), kPayload - 1));
  // Sanity: the correct size still works.
  ASSERT_TRUE(client_a_->Get("pt-mismatch", dst.data(), kPayload));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kPayload), 0);
}

}  // namespace
}  // namespace mori::umbp
