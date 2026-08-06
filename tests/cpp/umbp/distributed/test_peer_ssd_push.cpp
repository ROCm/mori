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

// RDMA-WRITE push for the SSD read fan-out.
//
// Background: single flight already collapses the tp_size-way same-key read
// that MLA + TP produces down to ONE device read — but the leader then memcpy'd
// its bytes into every follower's private staging slot, single-threaded.  On a
// measured Kimi-K2 L3 reload that copy was 68% of leader time (2x the device
// read) and moved 159 GB of redundant host-to-host traffic, with followers
// blocked on it for essentially their whole latency.  This suite pins the
// behaviour that replaces the copy with a hardware-DMA'd RDMA WRITE straight
// into each follower's own destination buffer.
//
// The push is an OPT-IN fast path: everything that cannot be pushed (no engine,
// unregistered destination, unresolvable target, failed leader read) must fall
// through to exactly the prior memcpy + staging + pull behaviour.  Several tests
// here exist only to hold that fallback in place.
//
// Needs a working RDMA device (real transfers, loopback between two in-process
// IO engines), so it is registered under the "integration" label.

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <msgpack.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp/local/tiers/tier_backend.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

constexpr int kFanout = 8;          // TP8: one leader + 7 followers
constexpr size_t kSlotSize = 4096;  // one staging slot / one destination window
constexpr size_t kStagingSize = kSlotSize * kFanout;
constexpr size_t kClientBufSize = kSlotSize * kFanout;

// ---------------------------------------------------------------------------
//  Fakes / helpers
// ---------------------------------------------------------------------------

// In-memory backend whose reads can be held open, so every reader is guaranteed
// to be inside the merge window at the same time.  Same shape as the one in
// test_peer_ssd_single_flight.cpp: without it "did they all merge?" is a race
// rather than an assertion.
class GatedBackend : public TierBackend {
 public:
  explicit GatedBackend(size_t capacity)
      : TierBackend(StorageTier::LOCAL_SSD), capacity_(capacity) {}

  bool Write(const std::string& key, const void* data, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (used_ + size > capacity_) return false;
    store_[key].assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
    used_ += size;
    return true;
  }

  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) override {
    {
      std::unique_lock<std::mutex> lk(gate_mu_);
      ++reads_started_;
      started_cv_.notify_all();
      gate_cv_.wait(lk, [this] { return !blocked_; });
      if (fail_) return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    if (it == store_.end() || it->second.size() != size) return false;
    std::memcpy(reinterpret_cast<void*>(dst), it->second.data(), size);
    return true;
  }

  bool Exists(const std::string& key) const override {
    std::lock_guard<std::mutex> lk(mu_);
    return store_.count(key) != 0;
  }
  bool Evict(const std::string& key) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    if (it == store_.end()) return false;
    used_ -= it->second.size();
    store_.erase(it);
    return true;
  }
  std::pair<size_t, size_t> Capacity() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return {used_, capacity_};
  }
  void Clear() override {
    std::lock_guard<std::mutex> lk(mu_);
    store_.clear();
    used_ = 0;
  }

  void BlockReads() {
    std::lock_guard<std::mutex> lk(gate_mu_);
    blocked_ = true;
  }
  void UnblockReads() {
    {
      std::lock_guard<std::mutex> lk(gate_mu_);
      blocked_ = false;
    }
    gate_cv_.notify_all();
  }
  void SetFailReads(bool f) {
    std::lock_guard<std::mutex> lk(gate_mu_);
    fail_ = f;
  }
  void WaitReadsStarted(int n) {
    std::unique_lock<std::mutex> lk(gate_mu_);
    started_cv_.wait(lk, [&] { return reads_started_ >= n; });
  }
  int reads_started() const {
    std::lock_guard<std::mutex> lk(gate_mu_);
    return reads_started_;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<char>> store_;
  size_t used_ = 0;
  size_t capacity_;

  mutable std::mutex gate_mu_;
  std::condition_variable gate_cv_;
  std::condition_variable started_cv_;
  bool blocked_ = false;
  bool fail_ = false;
  int reads_started_ = 0;
};

// Kernel-assigned free port; a hard-coded base collides with whatever else is on
// a shared host (mirrors AllocPort in the other peer tests).
uint16_t AllocPort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd >= 0) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = 0;
    socklen_t len = sizeof(addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
      uint16_t port = ntohs(addr.sin_port);
      ::close(fd);
      return port;
    }
    ::close(fd);
  }
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(53300 + (static_cast<unsigned>(::getpid()) % 4000))};
  return next.fetch_add(1);
}

std::unique_ptr<mori::io::IOEngine> MakeEngine(const std::string& key) {
  mori::io::IOEngineConfig cfg;
  cfg.host = "0.0.0.0";
  cfg.port = 0;
  auto engine = std::make_unique<mori::io::IOEngine>(key, cfg);
  mori::io::RdmaBackendConfig rdma_cfg;
  rdma_cfg.qpPerTransfer = 1;
  engine->CreateBackend(mori::io::BackendType::RDMA, rdma_cfg);
  return engine;
}

template <typename T>
std::string Pack(const T& v) {
  msgpack::sbuffer sbuf;
  msgpack::pack(sbuf, v);
  return std::string(sbuf.data(), sbuf.size());
}

std::vector<std::pair<const void*, size_t>> OneSeg(const std::string& s) {
  return {{s.data(), s.size()}};
}

// Spin until `n` requesters have found a read already in flight — the merge
// window is only deterministic if we wait for it, and polling a public counter
// beats sleeping.
void WaitForDups(const PeerSsdManager& mgr, uint64_t n) {
  while (mgr.ReadDup() < n) std::this_thread::yield();
}

// ---------------------------------------------------------------------------
//  Manager-level fixture: PeerSsdManager driven directly, real RDMA
// ---------------------------------------------------------------------------
//
// Two in-process IO engines stand in for the real topology: `peer_engine_` is
// the storage node that reads the drive and initiates the push, `client_engine_`
// is the TP-rank process being written into.  The peer registers the client's
// engine exactly as the GetPeerInfo handshake would.
class PeerSsdPushTest : public ::testing::Test {
 protected:
  void SetUp() override {
    peer_engine_ = MakeEngine("push-peer-" + std::to_string(::getpid()));
    client_engine_ = MakeEngine("push-client-" + std::to_string(::getpid()));
    peer_engine_->RegisterRemoteEngine(client_engine_->GetEngineDesc());

    staging_ = std::aligned_alloc(4096, kStagingSize);
    client_buf_ = std::aligned_alloc(4096, kClientBufSize);
    ASSERT_NE(staging_, nullptr);
    ASSERT_NE(client_buf_, nullptr);
    std::memset(staging_, 0, kStagingSize);
    std::memset(client_buf_, 0, kClientBufSize);

    staging_mem_ =
        peer_engine_->RegisterMemory(staging_, kStagingSize, -1, mori::io::MemoryLocationType::CPU);
    client_mem_ = client_engine_->RegisterMemory(client_buf_, kClientBufSize, -1,
                                                 mori::io::MemoryLocationType::CPU);

    auto be = std::make_unique<GatedBackend>(1 << 20);
    backend_ = be.get();
    mgr_ = std::make_unique<PeerSsdManager>(std::move(be), 0.9, 0.7, /*single_flight=*/true,
                                            peer_engine_.get());
  }

  void TearDown() override {
    mgr_.reset();
    if (staging_) peer_engine_->DeregisterMemory(staging_mem_);
    if (client_buf_) client_engine_->DeregisterMemory(client_mem_);
    std::free(staging_);
    std::free(client_buf_);
    client_engine_.reset();
    peer_engine_.reset();
  }

  void* SlotPtr(int i) { return static_cast<uint8_t*>(staging_) + i * kSlotSize; }
  void* ClientPtr(int i) { return static_cast<uint8_t*>(client_buf_) + i * kSlotSize; }

  // The slot requester `i` would send: its own staging slot as the RDMA source
  // (it might lead), and its own destination window as the RDMA sink (it might
  // follow).  `eligible=false` models an unregistered / unresolvable
  // destination, which must degrade to memcpy.
  ReadPushSlot SlotFor(int i, bool eligible = true) {
    ReadPushSlot s;
    s.local_desc = staging_mem_;
    s.local_offset = i * kSlotSize;
    s.push_eligible = eligible;
    if (eligible) {
      s.remote_desc = client_mem_;
      s.remote_offset = i * kSlotSize;
    }
    return s;
  }

  std::unique_ptr<mori::io::IOEngine> peer_engine_, client_engine_;
  void* staging_ = nullptr;
  void* client_buf_ = nullptr;
  mori::io::MemoryDesc staging_mem_{}, client_mem_{};
  GatedBackend* backend_ = nullptr;
  std::unique_ptr<PeerSsdManager> mgr_;
};

// The core behaviour: every follower's bytes arrive by RDMA WRITE in its own
// buffer, one device read total, and no follower's staging slot is touched —
// which is what lets the peer service hand that slot straight back.
TEST_F(PeerSsdPushTest, FollowersAreServedByRdmaWriteNotMemcpy) {
  const std::string payload(2048, 'p');
  ASSERT_TRUE(mgr_->Write("K", OneSeg(payload), payload.size()));

  backend_->BlockReads();
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::vector<bool>> pushed(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back([&, i] {
      auto res = mgr_->PrepareReadBatch({"K"}, {SlotPtr(i)}, {kSlotSize}, {SlotFor(i)}, &pushed[i]);
      outs[i] = res[0];
    });
  }
  backend_->WaitReadsStarted(1);    // the leader is on the device
  WaitForDups(*mgr_, kFanout - 1);  // every follower has attached
  backend_->UnblockReads();
  for (auto& t : readers) t.join();

  EXPECT_EQ(backend_->reads_started(), 1) << "followers must not touch the drive";
  EXPECT_EQ(mgr_->ReadPushed(), static_cast<uint64_t>(kFanout - 1));
  EXPECT_EQ(mgr_->ReadPushFailed(), 0u);

  int leaders = 0;
  for (int i = 0; i < kFanout; ++i) {
    ASSERT_EQ(outs[i].status, SsdReadStatus::kOk) << "reader " << i;
    ASSERT_EQ(pushed[i].size(), 1u);
    if (!pushed[i][0]) {
      ++leaders;
      // The leader read into its own staging slot and pulls from there as
      // before; nothing was written to its client-side window.
      EXPECT_EQ(std::string(static_cast<const char*>(SlotPtr(i)), payload.size()), payload);
      continue;
    }
    // A pushed follower: bytes in ITS OWN destination buffer...
    EXPECT_EQ(std::string(static_cast<const char*>(ClientPtr(i)), payload.size()), payload)
        << "follower " << i << " did not receive its bytes";
    // ...and its staging slot untouched, so the slot held nothing worth leasing.
    const auto* slot = static_cast<const uint8_t*>(SlotPtr(i));
    EXPECT_EQ(slot[0], 0) << "follower " << i << " staging slot was written after all";
  }
  EXPECT_EQ(leaders, 1) << "exactly one requester should have led the read";
}

// Push and memcpy followers coexist on one episode: an unregistered destination
// is served exactly as it was before this feature existed, on the same read.
TEST_F(PeerSsdPushTest, MixedPushAndMemcpyFollowersBothLandCorrectly) {
  const std::string payload(1024, 'm');
  ASSERT_TRUE(mgr_->Write("K", OneSeg(payload), payload.size()));

  backend_->BlockReads();
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::vector<bool>> pushed(kFanout);
  std::vector<std::thread> readers;
  // Even indices push, odd indices are ineligible and must fall back.
  auto eligible = [](int i) { return i % 2 == 0; };
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back([&, i] {
      auto res = mgr_->PrepareReadBatch({"K"}, {SlotPtr(i)}, {kSlotSize}, {SlotFor(i, eligible(i))},
                                        &pushed[i]);
      outs[i] = res[0];
    });
  }
  backend_->WaitReadsStarted(1);
  WaitForDups(*mgr_, kFanout - 1);
  backend_->UnblockReads();
  for (auto& t : readers) t.join();

  for (int i = 0; i < kFanout; ++i) {
    ASSERT_EQ(outs[i].status, SsdReadStatus::kOk) << "reader " << i;
    if (pushed[i][0]) {
      EXPECT_TRUE(eligible(i)) << "reader " << i << " was pushed without asking";
      EXPECT_EQ(std::string(static_cast<const char*>(ClientPtr(i)), payload.size()), payload);
    } else {
      // Leader or memcpy-follower: byte-for-byte the pre-push result, in the
      // staging slot the caller will pull from.
      EXPECT_EQ(std::string(static_cast<const char*>(SlotPtr(i)), payload.size()), payload)
          << "reader " << i;
    }
  }
  // Every ineligible follower must have been served the classic way.
  for (int i = 1; i < kFanout; i += 2) EXPECT_FALSE(pushed[i][0]) << "reader " << i;
}

// An unresolvable push target (what the peer service produces for an unknown
// node id, a stale region id, or an expired registration) is not an error — it
// is the memcpy path, with the right bytes.
TEST_F(PeerSsdPushTest, UnresolvableTargetFallsBackToTheClassicPath) {
  const std::string payload(777, 'u');
  ASSERT_TRUE(mgr_->Write("K", OneSeg(payload), payload.size()));

  backend_->BlockReads();
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::vector<bool>> pushed(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back([&, i] {
      // push_eligible, but the destination descriptor never resolved — exactly
      // what ResolvePushTargets leaves behind on a miss.
      ReadPushSlot s = SlotFor(i, /*eligible=*/true);
      s.remote_desc = mori::io::MemoryDesc{};
      auto res = mgr_->PrepareReadBatch({"K"}, {SlotPtr(i)}, {kSlotSize}, {s}, &pushed[i]);
      outs[i] = res[0];
    });
  }
  backend_->WaitReadsStarted(1);
  WaitForDups(*mgr_, kFanout - 1);
  backend_->UnblockReads();
  for (auto& t : readers) t.join();

  EXPECT_EQ(mgr_->ReadPushed(), 0u);
  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kOk) << "reader " << i;
    EXPECT_FALSE(pushed[i][0]) << "reader " << i;
    EXPECT_EQ(std::string(static_cast<const char*>(SlotPtr(i)), payload.size()), payload);
  }
}

// No IO engine: the push arguments are accepted and ignored.  This is the
// configuration every existing caller and unit test runs in, so it has to be
// bit-identical to the old behaviour.
TEST_F(PeerSsdPushTest, NoIoEngineIgnoresPushSlotsEntirely) {
  auto be = std::make_unique<GatedBackend>(1 << 20);
  auto* backend = be.get();
  PeerSsdManager mgr(std::move(be), 0.9, 0.7, /*single_flight=*/true, /*io_engine=*/nullptr);

  const std::string payload(512, 'n');
  ASSERT_TRUE(mgr.Write("K", OneSeg(payload), payload.size()));

  backend->BlockReads();
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::vector<bool>> pushed(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back([&, i] {
      auto res = mgr.PrepareReadBatch({"K"}, {SlotPtr(i)}, {kSlotSize}, {SlotFor(i)}, &pushed[i]);
      outs[i] = res[0];
    });
  }
  backend->WaitReadsStarted(1);
  WaitForDups(mgr, kFanout - 1);
  backend->UnblockReads();
  for (auto& t : readers) t.join();

  EXPECT_EQ(mgr.ReadPushed(), 0u);
  EXPECT_EQ(backend->reads_started(), 1);
  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kOk);
    EXPECT_FALSE(pushed[i][0]);
    EXPECT_EQ(std::string(static_cast<const char*>(SlotPtr(i)), payload.size()), payload);
  }
}

// A leader whose device read fails must post no BatchWrite at all and release
// every follower — push-registered or not — with an error, rather than leaving
// them to the 30s timeout or writing garbage into their buffers.
TEST_F(PeerSsdPushTest, LeaderReadFailurePostsNoPushAndErrorsEveryFollower) {
  const std::string payload(1024, 'f');
  ASSERT_TRUE(mgr_->Write("K", OneSeg(payload), payload.size()));

  backend_->BlockReads();
  backend_->SetFailReads(true);
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::vector<bool>> pushed(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back([&, i] {
      auto res = mgr_->PrepareReadBatch({"K"}, {SlotPtr(i)}, {kSlotSize}, {SlotFor(i)}, &pushed[i]);
      outs[i] = res[0];
    });
  }
  backend_->WaitReadsStarted(1);
  WaitForDups(*mgr_, kFanout - 1);
  backend_->UnblockReads();
  for (auto& t : readers) t.join();  // hangs here if the failure is not published

  EXPECT_EQ(mgr_->ReadPushed(), 0u);
  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kError) << "reader " << i;
    EXPECT_FALSE(pushed[i][0]) << "reader " << i;
    // Nothing was delivered anywhere: a failed read must not leave a partial
    // write in a requester's buffer that it might then trust.
    EXPECT_EQ(static_cast<const uint8_t*>(ClientPtr(i))[0], 0) << "reader " << i;
  }
}

// ---------------------------------------------------------------------------
//  RPC-level fixture: the whole peer stack over gRPC
// ---------------------------------------------------------------------------
//
// Covers what the manager-level tests cannot: the GetPeerInfo handshake, the
// per-request resolution of (node_id, region_id, offset), and — the point of
// the whole exercise — that a pushed key consumes NO read-staging slot.
class PeerSsdPushRpcTest : public ::testing::Test {
 protected:
  void SetUp() override {
    peer_engine_ = MakeEngine("push-rpc-peer-" + std::to_string(::getpid()));
    client_engine_ = MakeEngine("push-rpc-client-" + std::to_string(::getpid()));

    staging_ = std::aligned_alloc(4096, kStagingSize);
    client_buf_ = std::aligned_alloc(4096, kClientBufSize);
    ASSERT_NE(staging_, nullptr);
    ASSERT_NE(client_buf_, nullptr);
    std::memset(staging_, 0, kStagingSize);
    std::memset(client_buf_, 0, kClientBufSize);

    staging_mem_ =
        peer_engine_->RegisterMemory(staging_, kStagingSize, -1, mori::io::MemoryLocationType::CPU);
    client_mem_ = client_engine_->RegisterMemory(client_buf_, kClientBufSize, -1,
                                                 mori::io::MemoryLocationType::CPU);
    const std::string packed = Pack(staging_mem_);
    staging_desc_bytes_.assign(packed.begin(), packed.end());

    auto be = std::make_unique<GatedBackend>(1 << 20);
    backend_ = be.get();
    peer_ssd_ = std::make_unique<PeerSsdManager>(std::move(be), 0.9, 0.7, /*single_flight=*/true,
                                                 peer_engine_.get());

    port_ = AllocPort();
    server_ = std::make_unique<PeerServiceServer>(
        /*dram_alloc=*/nullptr, peer_ssd_.get(), staging_, kStagingSize, staging_desc_bytes_,
        /*num_read_slots=*/kFanout, std::chrono::seconds(5), /*engine_desc_bytes=*/
        std::vector<uint8_t>{}, /*master_client=*/nullptr, /*copy_pipeline=*/nullptr,
        SsdWriteStagingConfig{}, peer_engine_.get());
    ASSERT_TRUE(server_->Start(port_));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    channel_ = grpc::CreateChannel("localhost:" + std::to_string(port_),
                                   grpc::InsecureChannelCredentials());
    stub_ = ::umbp::UMBPPeer::NewStub(channel_);
  }

  void TearDown() override {
    stub_.reset();
    server_->Stop();
    server_.reset();
    peer_ssd_.reset();
    if (staging_) peer_engine_->DeregisterMemory(staging_mem_);
    if (client_buf_) client_engine_->DeregisterMemory(client_mem_);
    std::free(staging_);
    std::free(client_buf_);
    client_engine_.reset();
    peer_engine_.reset();
  }

  // The handshake a PoolClient runs on first contact: our engine plus the
  // destination buffers the peer may write into, keyed by stable region id.
  void Handshake(const std::string& node_id, uint64_t region_id = kRegionId) {
    ::umbp::GetPeerInfoRequest req;
    req.set_node_id(node_id);
    req.set_engine_desc(Pack(client_engine_->GetEngineDesc()));
    auto* buf = req.add_dst_buffers();
    buf->set_region_id(region_id);
    buf->set_desc(Pack(client_mem_));
    ::umbp::GetPeerInfoResponse resp;
    grpc::ClientContext ctx;
    ASSERT_TRUE(stub_->GetPeerInfo(&ctx, req, &resp).ok());
  }

  ::umbp::BatchPrepareSsdReadResponse BatchPrepare(const std::string& node_id,
                                                   const std::string& key, uint64_t size,
                                                   uint64_t region_id, uint64_t offset) {
    ::umbp::BatchPrepareSsdReadRequest req;
    req.set_requester_node_id(node_id);
    req.add_keys(key);
    req.add_max_sizes(size);
    req.add_dst_region_id(region_id);
    req.add_dst_offset(offset);
    ::umbp::BatchPrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->BatchPrepareSsdRead(&ctx, req, &resp).ok());
    return resp;
  }

  static constexpr uint64_t kRegionId = 7;  // arbitrary; only "stable" matters

  std::unique_ptr<mori::io::IOEngine> peer_engine_, client_engine_;
  void* staging_ = nullptr;
  void* client_buf_ = nullptr;
  mori::io::MemoryDesc staging_mem_{}, client_mem_{};
  std::vector<uint8_t> staging_desc_bytes_;
  GatedBackend* backend_ = nullptr;
  std::unique_ptr<PeerSsdManager> peer_ssd_;
  std::unique_ptr<PeerServiceServer> server_;
  uint16_t port_ = 0;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<::umbp::UMBPPeer::Stub> stub_;
};

// The handshake lands and is visible as a live registration.
TEST_F(PeerSsdPushRpcTest, HandshakeRegistersThePushTarget) {
  EXPECT_EQ(server_->SnapshotClientPushTargets(), 0u);
  Handshake("tp-rank-0");
  EXPECT_EQ(server_->SnapshotClientPushTargets(), 1u);
}

// An empty request (an old client, or one with nothing registered) must be a
// silent no-op, not a registration and not an error.
TEST_F(PeerSsdPushRpcTest, EmptyHandshakeRegistersNothing) {
  ::umbp::GetPeerInfoRequest req;
  ::umbp::GetPeerInfoResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->GetPeerInfo(&ctx, req, &resp).ok());
  EXPECT_EQ(server_->SnapshotClientPushTargets(), 0u);
}

// The headline regression: concurrent same-key requesters are pushed to, their
// bytes land in their own buffers, and — because a pushed key never holds a
// lease — every staging slot is free again by the time the calls return.
TEST_F(PeerSsdPushRpcTest, ConcurrentFollowersArePushedAndHoldNoSlots) {
  const std::string payload(2048, 'r');
  ASSERT_TRUE(peer_ssd_->Write("K", OneSeg(payload), payload.size()));
  Handshake("tp-rank-0");

  backend_->BlockReads();
  std::vector<::umbp::BatchPrepareSsdReadResponse> resps(kFanout);
  std::vector<std::thread> callers;
  for (int i = 0; i < kFanout; ++i) {
    callers.emplace_back([&, i] {
      resps[i] = BatchPrepare("tp-rank-0", "K", payload.size(), kRegionId, i * kSlotSize);
    });
  }
  backend_->WaitReadsStarted(1);
  WaitForDups(*peer_ssd_, kFanout - 1);
  backend_->UnblockReads();
  for (auto& t : callers) t.join();

  int pushed = 0, leased = 0;
  for (int i = 0; i < kFanout; ++i) {
    ASSERT_EQ(resps[i].status_size(), 1);
    ASSERT_EQ(resps[i].status(0), ::umbp::SSD_READ_OK) << "caller " << i;
    EXPECT_FALSE(resps[i].push_registration_stale());
    if (resps[i].pushed_size() > 0 && resps[i].pushed(0)) {
      ++pushed;
      // Nothing to pull and nothing to release.
      EXPECT_EQ(resps[i].lease_id(0), 0u) << "caller " << i;
      EXPECT_EQ(resps[i].staging_offset(0), 0u) << "caller " << i;
      EXPECT_EQ(std::string(static_cast<const char*>(client_buf_) + i * kSlotSize, payload.size()),
                payload)
          << "caller " << i;
    } else {
      ++leased;
      EXPECT_GT(resps[i].lease_id(0), 0u) << "caller " << i;
    }
  }
  EXPECT_EQ(pushed, kFanout - 1) << "every follower should have been pushed to";
  EXPECT_EQ(leased, 1) << "only the leader should hold a lease";
  // The leader's lease is still out; the kFanout-1 pushed slots must not be.
  EXPECT_EQ(server_->SnapshotReadSlotsInUse(), 1u);
}

// An unknown requester, and a region id the peer never saw, both fall back to
// the classic staging+lease answer AND raise the re-register flag, so the client
// can recover the fast path instead of silently losing it forever.
TEST_F(PeerSsdPushRpcTest, UnknownNodeOrRegionFallsBackAndAsksForReregistration) {
  const std::string payload(256, 's');
  ASSERT_TRUE(peer_ssd_->Write("K", OneSeg(payload), payload.size()));

  // Never handshaked.
  auto resp = BatchPrepare("ghost-rank", "K", payload.size(), kRegionId, 0);
  ASSERT_EQ(resp.status(0), ::umbp::SSD_READ_OK);
  EXPECT_FALSE(resp.pushed(0));
  EXPECT_GT(resp.lease_id(0), 0u);
  EXPECT_TRUE(resp.push_registration_stale());

  // Handshaked, but naming a region the peer does not hold.
  Handshake("tp-rank-0");
  auto resp2 = BatchPrepare("tp-rank-0", "K", payload.size(), kRegionId + 99, 0);
  ASSERT_EQ(resp2.status(0), ::umbp::SSD_READ_OK);
  EXPECT_FALSE(resp2.pushed(0));
  EXPECT_TRUE(resp2.push_registration_stale());
}

// A destination window that does not fit inside the registered region must be
// refused server-side and served the classic way — a stale or malformed request
// can never turn into a write past the end of a client's buffer.
TEST_F(PeerSsdPushRpcTest, OutOfBoundsOffsetIsRefusedNotWritten) {
  const std::string payload(256, 'b');
  ASSERT_TRUE(peer_ssd_->Write("K", OneSeg(payload), payload.size()));
  Handshake("tp-rank-0");

  // Offset lands inside the region but the read would run off its end.
  auto resp = BatchPrepare("tp-rank-0", "K", payload.size(), kRegionId, kClientBufSize - 8);
  ASSERT_EQ(resp.status(0), ::umbp::SSD_READ_OK);
  EXPECT_FALSE(resp.pushed(0)) << "an out-of-bounds destination must not be pushed to";
  EXPECT_GT(resp.lease_id(0), 0u) << "it must still be served the classic way";
  // Nothing was written near the end of the client's buffer.
  EXPECT_EQ(static_cast<const uint8_t*>(client_buf_)[kClientBufSize - 8], 0);
}

// A single requester with no concurrent fan-out leads its own read: nothing to
// push, classic staging answer, and no false staleness signal.
TEST_F(PeerSsdPushRpcTest, SoleRequesterLeadsAndIsNotPushedTo) {
  const std::string payload(128, 'l');
  ASSERT_TRUE(peer_ssd_->Write("K", OneSeg(payload), payload.size()));
  Handshake("tp-rank-0");

  auto resp = BatchPrepare("tp-rank-0", "K", payload.size(), kRegionId, 0);
  ASSERT_EQ(resp.status(0), ::umbp::SSD_READ_OK);
  EXPECT_FALSE(resp.pushed(0));
  EXPECT_FALSE(resp.push_registration_stale());
  EXPECT_GT(resp.lease_id(0), 0u);
  EXPECT_EQ(
      std::string(static_cast<const char*>(staging_) + resp.staging_offset(0), payload.size()),
      payload);
}

// ---------------------------------------------------------------------------
//  TTL expiry / client-restart safety
// ---------------------------------------------------------------------------
//
// The highest-value new test: the push makes the peer an RDMA initiator against
// the CLIENT role, which is the short-lived, restart-prone side.  A client that
// dies and comes back with the same node_id gets the same per-process
// MemoryUniqueId sequence, so a registration that never expired could be
// resolved against a dead instance's memory.  The TTL is what bounds that
// window, and expiry has to fail into the classic path — not into a hang and
// not into a misdirected write.
//
// Runs in its own fixture because the TTL is read once at server construction,
// so the environment has to be set before the server exists.
class PeerSsdPushTtlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 1000ms is the floor ResolveClientPushTtl enforces; anything lower would be
    // clamped up and make this test sleep for the default 30s instead.
    ::setenv("UMBP_PUSH_TARGET_TTL_MS", "1000", /*overwrite=*/1);

    peer_engine_ = MakeEngine("push-ttl-peer-" + std::to_string(::getpid()));
    client_engine_ = MakeEngine("push-ttl-client-" + std::to_string(::getpid()));

    staging_ = std::aligned_alloc(4096, kStagingSize);
    client_buf_ = std::aligned_alloc(4096, kClientBufSize);
    ASSERT_NE(staging_, nullptr);
    ASSERT_NE(client_buf_, nullptr);
    std::memset(staging_, 0, kStagingSize);
    std::memset(client_buf_, 0, kClientBufSize);
    staging_mem_ =
        peer_engine_->RegisterMemory(staging_, kStagingSize, -1, mori::io::MemoryLocationType::CPU);
    client_mem_ = client_engine_->RegisterMemory(client_buf_, kClientBufSize, -1,
                                                 mori::io::MemoryLocationType::CPU);
    const std::string packed = Pack(staging_mem_);
    staging_desc_bytes_.assign(packed.begin(), packed.end());

    auto be = std::make_unique<GatedBackend>(1 << 20);
    peer_ssd_ = std::make_unique<PeerSsdManager>(std::move(be), 0.9, 0.7, true, peer_engine_.get());
    ASSERT_TRUE(peer_ssd_->Write("K", OneSeg(payload_), payload_.size()));

    port_ = AllocPort();
    // Plenty of read slots: these tests deliberately never release a lease (the
    // classic path is what they assert), and running out would surface as
    // NO_SLOT rather than as the staleness behaviour under test.
    server_ = std::make_unique<PeerServiceServer>(
        nullptr, peer_ssd_.get(), staging_, kStagingSize, staging_desc_bytes_,
        /*num_read_slots=*/64, std::chrono::seconds(5), std::vector<uint8_t>{}, nullptr, nullptr,
        SsdWriteStagingConfig{}, peer_engine_.get());
    ASSERT_TRUE(server_->Start(port_));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    stub_ = ::umbp::UMBPPeer::NewStub(grpc::CreateChannel("localhost:" + std::to_string(port_),
                                                          grpc::InsecureChannelCredentials()));
  }

  void TearDown() override {
    ::unsetenv("UMBP_PUSH_TARGET_TTL_MS");
    stub_.reset();
    server_->Stop();
    server_.reset();
    peer_ssd_.reset();
    if (staging_) peer_engine_->DeregisterMemory(staging_mem_);
    if (client_buf_) client_engine_->DeregisterMemory(client_mem_);
    std::free(staging_);
    std::free(client_buf_);
    client_engine_.reset();
    peer_engine_.reset();
  }

  void Handshake() {
    ::umbp::GetPeerInfoRequest req;
    req.set_node_id("tp-rank-0");
    req.set_engine_desc(Pack(client_engine_->GetEngineDesc()));
    auto* buf = req.add_dst_buffers();
    buf->set_region_id(1);
    buf->set_desc(Pack(client_mem_));
    ::umbp::GetPeerInfoResponse resp;
    grpc::ClientContext ctx;
    ASSERT_TRUE(stub_->GetPeerInfo(&ctx, req, &resp).ok());
  }

  ::umbp::BatchPrepareSsdReadResponse Prepare() {
    ::umbp::BatchPrepareSsdReadRequest req;
    req.set_requester_node_id("tp-rank-0");
    req.add_keys("K");
    req.add_max_sizes(payload_.size());
    req.add_dst_region_id(1);
    req.add_dst_offset(0);
    ::umbp::BatchPrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->BatchPrepareSsdRead(&ctx, req, &resp).ok());
    return resp;
  }

  const std::string payload_ = std::string(256, 't');
  std::unique_ptr<mori::io::IOEngine> peer_engine_, client_engine_;
  void* staging_ = nullptr;
  void* client_buf_ = nullptr;
  mori::io::MemoryDesc staging_mem_{}, client_mem_{};
  std::vector<uint8_t> staging_desc_bytes_;
  std::unique_ptr<PeerSsdManager> peer_ssd_;
  std::unique_ptr<PeerServiceServer> server_;
  uint16_t port_ = 0;
  std::unique_ptr<::umbp::UMBPPeer::Stub> stub_;
};

// A client that goes silent past the TTL has its registration dropped, and the
// next request is answered the classic way with a re-register request — no hang,
// no write against the dead instance's memory.
TEST_F(PeerSsdPushTtlTest, SilentClientRegistrationExpires) {
  Handshake();
  ASSERT_EQ(server_->SnapshotClientPushTargets(), 1u);

  std::this_thread::sleep_for(std::chrono::milliseconds(1400));
  EXPECT_EQ(server_->SnapshotClientPushTargets(), 0u) << "TTL did not age the entry out";

  // The read still succeeds — expiry costs the fast path, never the data.
  auto resp = Prepare();
  ASSERT_EQ(resp.status(0), ::umbp::SSD_READ_OK);
  EXPECT_FALSE(resp.pushed(0));
  EXPECT_TRUE(resp.push_registration_stale());
  EXPECT_GT(resp.lease_id(0), 0u);

  // ...and re-running the handshake (what the client does on that flag) restores
  // it, so a restarted client recovers rather than degrading permanently.
  Handshake();
  EXPECT_EQ(server_->SnapshotClientPushTargets(), 1u);
  EXPECT_FALSE(Prepare().push_registration_stale());
}

// An active client is never expired mid-use: each request refreshes its own
// entry, so traffic alone keeps the registration alive well past one TTL.
TEST_F(PeerSsdPushTtlTest, ActiveClientRegistrationIsRefreshedByItsOwnTraffic) {
  Handshake();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1600);
  while (std::chrono::steady_clock::now() < deadline) {
    auto resp = Prepare();
    ASSERT_EQ(resp.status(0), ::umbp::SSD_READ_OK);
    ASSERT_FALSE(resp.push_registration_stale()) << "an active client was declared stale";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  EXPECT_EQ(server_->SnapshotClientPushTargets(), 1u);
}

}  // namespace
}  // namespace mori::umbp
