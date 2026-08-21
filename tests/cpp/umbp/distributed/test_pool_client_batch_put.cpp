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

// BatchPut staging/registration coverage across three routing scenarios:
//   1. Cross-node BatchPut with un-registered caller src → staging fallback
//      still completes every key.
//   2. Cross-node BatchPut with registered caller src → zero-copy path.
//   3. Same-node BatchPut (local memcpy branch) with un-registered src.
//
// The legacy one-shot "src not registered" batch-level WARN (+ 60s throttle)
// has been removed from PoolClient; these tests now assert that BatchPut
// succeeds and that no such WARN is emitted on any path.  Assertions filter
// spdlog output by the substring "BatchPut: src not registered for key=".

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
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

// Unique peer-service port per PoolClient.  A non-zero peer_service_port is
// required for a node to register a peer_address and serve remote
// AllocateSlot/CommitSlot RPCs; without it the caller's remote BatchPut fails
// with "peer service connection unavailable".
//
// The port must be free *at bind time* (PoolClient binds it directly and
// registers it verbatim), so a hard-coded base collides with concurrent test
// processes / leftover servers on a shared (self-hosted CI) host.  Ask the
// kernel for a currently-free ephemeral port instead.
inline uint16_t NextPeerServicePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd >= 0) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = 0;  // kernel picks a free port
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
      static_cast<uint16_t>(53000 + (static_cast<unsigned>(::getpid()) % 4000))};
  return next.fetch_add(1);
}

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
    ASSERT_NE(caller_buf_, nullptr);
    ASSERT_NE(registered_buf_, nullptr);
    std::memset(caller_buf_, 0, kCallerBuf);
    std::memset(registered_buf_, 0, kCallerBuf);

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
    cfg_caller.peer_service_port = NextPeerServicePort();
    cfg_caller.dram_page_size = kPageSize;
    cfg_caller.dram.buffer_sizes = {kPageSize};
    caller_ = std::make_unique<PoolClient>(std::move(cfg_caller));
    ASSERT_TRUE(caller_->Init());

    PoolClientConfig cfg_target;
    cfg_target.master_config.node_id = "node-target";
    cfg_target.master_config.node_address = "127.0.0.1";
    cfg_target.master_config.master_address = master_addr;
    cfg_target.io_engine.host = "0.0.0.0";
    cfg_target.io_engine.port = 0;
    cfg_target.peer_service_port = NextPeerServicePort();
    cfg_target.dram_page_size = kPageSize;
    cfg_target.dram.buffer_sizes = {kRemoteCap};
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
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
};

// Un-registered src on a remote-bound batch still completes (staging
// fallback): every key succeeds and no batch-level WARN is emitted (the
// previous one-shot "src not registered" WARN + 60s throttle was removed).
TEST_F(BatchPutWarnTest, StagingFallbackSucceedsWithoutWarn) {
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
  EXPECT_EQ(cap.CountSubstring(kBatchPutWarnSubstr), 0u);
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

// Zero-size puts are rejected before local/peer execution: the empty entry
// fails (false) while the surrounding non-zero keys still succeed.  Return
// vector length is preserved.  Runs on target_ (self/local path) so the
// assertion does not depend on remote RDMA.
TEST_F(BatchPutWarnTest, ZeroSizePutEntriesFailOthersSucceed) {
  std::vector<char> buf(3 * kPerKey, 0);
  std::memset(buf.data(), 0x21, kPerKey);
  std::memset(buf.data() + 2 * kPerKey, 0x23, kPerKey);

  std::vector<std::string> keys = {"zs-a", "zs-zero", "zs-b"};
  std::vector<const void*> srcs = {buf.data(), buf.data() + kPerKey, buf.data() + 2 * kPerKey};
  std::vector<size_t> sizes = {kPerKey, 0, kPerKey};

  auto results = target_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), 3u);
  EXPECT_TRUE(results[0]) << "key=" << keys[0];
  EXPECT_FALSE(results[1]) << "zero-size put must be filtered to failure";
  EXPECT_TRUE(results[2]) << "key=" << keys[2];
}

// Zero-size gets are rejected before local fallback or remote read: they fail
// (false) while previously-seeded non-zero keys still resolve.  Return vector
// length is preserved.  Local self put+get on target_ keeps the round trip
// free of RDMA/registration.
TEST_F(BatchPutWarnTest, ZeroSizeGetEntriesFailOthersSucceed) {
  std::vector<char> src(2 * kPerKey, 0);
  std::memset(src.data(), 0x31, kPerKey);
  std::memset(src.data() + kPerKey, 0x32, kPerKey);
  std::vector<std::string> pkeys = {"zg-a", "zg-b"};
  std::vector<const void*> psrcs = {src.data(), src.data() + kPerKey};
  std::vector<size_t> psizes = {kPerKey, kPerKey};
  auto pres = target_->BatchPut(pkeys, psrcs, psizes);
  ASSERT_EQ(pres.size(), 2u);
  ASSERT_TRUE(pres[0]);
  ASSERT_TRUE(pres[1]);

  std::vector<char> dst(3 * kPerKey, 0);
  std::vector<std::string> gkeys = {"zg-a", "zg-zero", "zg-b"};
  std::vector<void*> gdsts = {dst.data(), dst.data() + kPerKey, dst.data() + 2 * kPerKey};
  std::vector<size_t> gsizes = {kPerKey, 0, kPerKey};

  auto gres = target_->BatchGet(gkeys, gdsts, gsizes);
  ASSERT_EQ(gres.size(), 3u);
  EXPECT_TRUE(gres[0]) << "key=" << gkeys[0];
  EXPECT_FALSE(gres[1]) << "zero-size get must be filtered to failure";
  EXPECT_TRUE(gres[2]) << "key=" << gkeys[2];
}

// ---------------------------------------------------------------------------
// Multi-page objects exercise run coalescing in LocalCopyEngine. Every other
// test here uses exactly one page per key, so the merge path never runs there.
//
// A wrong run boundary or a miscomputed run length still returns true, so these
// assert bytes rather than status. The fill pattern varies per byte so a
// shifted, duplicated or truncated run is visible.
// ---------------------------------------------------------------------------

namespace {
void FillPattern(char* buf, size_t bytes, uint8_t seed) {
  for (size_t i = 0; i < bytes; ++i) {
    buf[i] = static_cast<char>((i * 31u + seed) & 0xFFu);
  }
}
}  // namespace

TEST_F(BatchPutWarnTest, MultiPageObjectRoundTripsByteExact) {
  constexpr size_t kPages = 4;
  constexpr size_t kSize = kPages * kPageSize;  // whole pages, no partial tail

  std::vector<char> src(kSize);
  FillPattern(src.data(), kSize, 0x5A);
  std::vector<std::string> keys = {"mp-full"};
  std::vector<const void*> srcs = {src.data()};
  std::vector<size_t> sizes = {kSize};
  auto pres = target_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(pres.size(), 1u);
  ASSERT_TRUE(pres[0]);

  std::vector<char> dst(kSize, 0);
  std::vector<void*> dsts = {dst.data()};
  auto gres = target_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(gres.size(), 1u);
  ASSERT_TRUE(gres[0]);
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kSize), 0)
      << "multi-page object did not round trip byte-exact";
}

TEST_F(BatchPutWarnTest, MultiPageObjectWithPartialTailRoundTripsByteExact) {
  // Deliberately not a page multiple: the final page is short, which is the
  // one place LogicalPageBytes differs and the easiest boundary to get wrong
  // when accumulating a run length.
  constexpr size_t kSize = 3 * kPageSize + 1234;

  std::vector<char> src(kSize);
  FillPattern(src.data(), kSize, 0xA5);
  std::vector<std::string> keys = {"mp-tail"};
  std::vector<const void*> srcs = {src.data()};
  std::vector<size_t> sizes = {kSize};
  auto pres = target_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(pres.size(), 1u);
  ASSERT_TRUE(pres[0]);

  // One extra byte of guard on each side catches a run that copies too much.
  std::vector<char> dst(kSize + 2, 0x7E);
  std::vector<void*> dsts = {dst.data() + 1};
  auto gres = target_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(gres.size(), 1u);
  ASSERT_TRUE(gres[0]);
  EXPECT_EQ(std::memcmp(src.data(), dst.data() + 1, kSize), 0)
      << "partial-tail object did not round trip byte-exact";
  EXPECT_EQ(dst.front(), static_cast<char>(0x7E)) << "run underran the destination";
  EXPECT_EQ(dst.back(), static_cast<char>(0x7E)) << "run overran the destination";
}

// ---------------------------------------------------------------------------
// GPU user buffers. Neither branch had coverage for a device pointer driven
// through PoolClient, which is the whole point of the port: an unregistered
// device pointer used to be described as host bytes and memcpy'd.
// ---------------------------------------------------------------------------

namespace {

// Device memory is not available on every machine that runs this suite (and is
// not available at all in a container without /dev/kfd), so the GPU cases skip
// rather than fail. A skip here is not a pass — it means the case did not run.
bool DeviceMemoryAvailable() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return count > 0;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) {
    if (hipMalloc(&ptr_, bytes) != hipSuccess) {
      (void)hipGetLastError();
      ptr_ = nullptr;
    }
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) (void)hipFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  bool Valid() const { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
};

}  // namespace

// The self-target path with a device source and destination. This is the case
// BuildLocalPageTransfers used to mislabel: the user ref was built with
// TransferRef::HostBytes directly, so LocalCopyEngine claimed the pair and
// memcpy'd from device memory. Multi-page so it also covers coalescing on the
// GPU side.
TEST_F(BatchPutWarnTest, DeviceBufferRoundTripsByteExactOnLocalPath) {
  if (!DeviceMemoryAvailable()) GTEST_SKIP() << "no HIP device available";

  constexpr size_t kSize = 3 * kPageSize + 512;  // multi-page with a short tail
  DeviceBuffer device_src(kSize);
  DeviceBuffer device_dst(kSize);
  if (!device_src.Valid() || !device_dst.Valid()) GTEST_SKIP() << "hipMalloc failed";

  std::vector<char> host(kSize);
  FillPattern(host.data(), kSize, 0x3C);
  ASSERT_EQ(hipMemcpy(device_src.get(), host.data(), kSize, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemset(device_dst.get(), 0, kSize), hipSuccess);

  std::vector<std::string> keys = {"gpu-local"};
  std::vector<const void*> srcs = {device_src.get()};
  std::vector<size_t> sizes = {kSize};
  auto pres = target_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(pres.size(), 1u);
  ASSERT_TRUE(pres[0]) << "device-source put must be handled by HbmCopyEngine";

  std::vector<void*> dsts = {device_dst.get()};
  auto gres = target_->BatchGet(keys, dsts, sizes);
  ASSERT_EQ(gres.size(), 1u);
  ASSERT_TRUE(gres[0]) << "device-destination get must be handled by HbmCopyEngine";

  std::vector<char> back(kSize, 0);
  ASSERT_EQ(hipMemcpy(back.data(), device_dst.get(), kSize, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(std::memcmp(host.data(), back.data(), kSize), 0)
      << "device round trip through the local path was not byte-exact";
}

// An UNREGISTERED device source on a remote-bound batch must be rejected, not
// host-staged. Before the port the bounce predicate tested only HasHostPtr(),
// so this std::memcpy'd out of hipMalloc'd memory — which faults rather than
// corrupting, so the old behaviour would take the whole suite down here.
//
// The fixture pins caller_ to a single page precisely to force remote routing,
// which is what StagingFallbackSucceedsWithoutWarn depends on: the same shape
// with a HOST buffer succeeds for every key by staging. So the contrast is the
// assertion — identical batch, device memory instead of host memory, and every
// key must now fail.
//
// The log check is what makes the failure meaningful rather than incidental: it
// pins the reason to MoriIoEngine finding no engine for the pair. UmbpLogCapture
// forces the module to WARN, which is why the message is observable here and
// not in a default-level run.
TEST_F(BatchPutWarnTest, UnregisteredDeviceSourceIsRejectedNotHostStaged) {
  if (!DeviceMemoryAvailable()) GTEST_SKIP() << "no HIP device available";

  constexpr size_t kKeys = 4;
  DeviceBuffer device_src(kKeys * kPerKey);
  if (!device_src.Valid()) GTEST_SKIP() << "hipMalloc failed";
  ASSERT_EQ(hipMemset(device_src.get(), 0x6B, kKeys * kPerKey), hipSuccess);

  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < kKeys; ++i) {
    keys.push_back("gpu-remote-" + std::to_string(i));
    srcs.push_back(static_cast<const char*>(device_src.get()) + i * kPerKey);
    sizes.push_back(kPerKey);
  }

  UmbpLogCapture cap;
  auto results = caller_->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(results.size(), keys.size()) << "result vector length must be preserved";
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_FALSE(results[i]) << "key=" << keys[i]
                             << ": unregistered device memory must not be host-staged";
  }
  EXPECT_GE(cap.CountSubstring("transfer unplannable"), 1u)
      << "the batch failed, but not because the transfer layer refused the GPU endpoint";

  // None of the keys may have been published either, or a later Get would hand
  // back whatever the staging buffer happened to hold.
  auto exists = caller_->BatchExists(keys);
  ASSERT_EQ(exists.size(), keys.size());
  for (size_t i = 0; i < exists.size(); ++i) {
    EXPECT_FALSE(exists[i]) << "key=" << keys[i] << " was published despite a failed put";
  }
}

}  // namespace
}  // namespace mori::umbp
