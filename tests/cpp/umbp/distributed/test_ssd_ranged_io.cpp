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

// RANGED I/O ON AN SSD-MEDIUM NODE  (doc/design-ssd-ranged-io.md, D0)
//
// DistributedClient::SupportsRangedIO() used to answer false whenever the
// node's medium was SSD, on the reasoning that ranged access maps object
// ranges onto pages a backend publishes as in-process endpoints and SSD
// publishes storage refs instead.  SsdBackend does not do that: it weighs a
// file endpoint against staging and picks staging (ssd_backend.h), so it
// publishes ordinary registered host pages and a resolved SSD key reaches
// BuildLocalRangeTransfers looking exactly like a DRAM one.
//
// The gate was therefore refusing paths that already worked.  This suite pins
// all four of them so they cannot regress:
//
//   1. The capability itself   — the arena is the only condition, on SSD as on
//      DRAM.
//   2. Local ranged put+get    — scattered sources tile an object onto an SSD
//      node's own medium; a subset reads back.
//   3. Remote ranged put+get   — the same across the wire to an SSD-only peer,
//      through the scratch arena.
//   4. Subset semantics        — a ranged get writes the requested bytes and
//      nothing else, which is what distinguishes it from a whole-object read.
//
// WHAT THIS SUITE DELIBERATELY DOES NOT ASSERT: that any of it is faster.  A
// resolve stages the whole object before the ranges are known, so a ranged get
// off SSD still reads the full object from the device.  Making the device read
// follow the requested extent is D1, and the test that belongs with it counts
// bytes read at the TierBackend — an assertion this suite would pass without
// D1's code, which is exactly why it is not written here.

#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

constexpr size_t kPageSize = 4096;
// One key is one page on an SSD node (ssd_backend.h: a key larger than
// page_size is refused at BatchAllocate), so the object is a page.
constexpr size_t kObjectSize = kPageSize;
constexpr size_t kScratchSize = 8 * kObjectSize;
// A single page of DRAM ~= no capacity, so a put from this node must route out.
constexpr size_t kNoCapacity = kPageSize;

uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(58000 + (static_cast<unsigned>(::getpid()) % 4000))};
  return next.fetch_add(1);
}

std::vector<char> Pattern(size_t size, unsigned seed) {
  std::vector<char> out(size);
  for (size_t i = 0; i < size; ++i) {
    out[i] = static_cast<char>(((i * 31) + seed * 7 + 11) & 0xff);
  }
  return out;
}

bool WaitForExists(IUMBPClient* client, const std::string& key) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto found = client->BatchExists({key});
    if (found.size() == 1 && found[0]) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  const auto found = client->BatchExists({key});
  return found.size() == 1 && found[0];
}

uintptr_t Addr(void* p) { return reinterpret_cast<uintptr_t>(p); }
uintptr_t Addr(const void* p) { return reinterpret_cast<uintptr_t>(p); }

class SsdRangedIoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    ssd_dir_ = fs::temp_directory_path() / ("umbp_ssd_ranged_" + std::to_string(::getpid()) + "_" +
                                            std::to_string(counter.fetch_add(1)));
    fs::remove_all(ssd_dir_);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    // The registry derives a 2 s client heartbeat from this TTL.  Left at the
    // default, every WaitForExists below sits out a full publication interval.
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds(4);
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0) << "Master failed to start";
    master_addr_ = "localhost:" + std::to_string(master_->GetBoundPort());
  }

  void TearDown() override {
    for (auto& c : clients_) {
      if (c) c->Close();
    }
    clients_.clear();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::error_code ec;
    fs::remove_all(ssd_dir_, ec);
  }

  // Deliberately the full IUMBPClient, not the bare PoolClient the sibling
  // ranges suite drives.  SupportsRangedIO() lives on DistributedClient, and
  // the point of every test here is to tie the data path to what the client
  // ADVERTISES -- see the note above TEST 2.
  UMBPConfig BaseConfig(const std::string& node_id, size_t ranged_scratch_size) {
    UMBPConfig config;
    config.dram.use_hugepages = false;
    // The SSD tier's own knobs live on UMBPConfig, not on the distributed
    // block: DistributedClient lowers config.ssd into PeerSsdConfig::ssd, and
    // only the *enabled* flag comes from the medium selector.
    config.ssd.storage_dir = ssd_dir_.string();
    config.ssd.capacity_bytes = 64ULL * 1024 * 1024;
    config.ssd.io.backend = UMBPIoBackend::Posix;  // avoid io_uring container flakiness

    UMBPDistributedConfig distributed;
    distributed.master_config.node_id = node_id;
    distributed.master_config.node_address = "127.0.0.1";
    distributed.master_config.master_address = master_addr_;
    distributed.io_engine.host = "0.0.0.0";
    distributed.io_engine.port = 0;
    distributed.peer_service_port = NextPeerServicePort();
    distributed.dram_page_size = kPageSize;
    // Smaller than one object, so a remote ranged transfer can only pass
    // through the registered arena and never the legacy staging path.
    distributed.staging_buffer_size = 1024;
    distributed.ranged_scratch_size = ranged_scratch_size;
    // Off, so a remote read stays remote and the assertions describe the SSD
    // peer rather than a re-cached copy on the reader.
    distributed.cache_remote_fetches = false;
    distributed.ssd_staging_buffer_slots = 8;
    config.distributed = std::move(distributed);
    return config;
  }

  UMBPConfig SsdConfig(const std::string& node_id, size_t ranged_scratch_size = kScratchSize) {
    auto config = BaseConfig(node_id, ranged_scratch_size);
    // The selector IS the opt-in: this is what sets PeerSsdConfig::enabled.
    config.distributed->medium = UMBPMedium::SSD;
    // Present and to be ignored -- an SSD node gets no host pool whatever this
    // says (see test_medium_selection.cpp).
    config.dram.capacity_bytes = 8 << 20;
    return config;
  }

  UMBPConfig DramConfig(const std::string& node_id, size_t capacity,
                        size_t ranged_scratch_size = kScratchSize) {
    auto config = BaseConfig(node_id, ranged_scratch_size);
    config.distributed->medium = UMBPMedium::DRAM;
    config.dram.capacity_bytes = capacity;
    return config;
  }

  IUMBPClient* Start(UMBPConfig config) {
    std::string validation_error;
    EXPECT_TRUE(config.Validate(&validation_error)) << validation_error;
    auto client = CreateUMBPClient(config);
    if (client == nullptr) return nullptr;
    clients_.push_back(std::move(client));
    return clients_.back().get();
  }

  fs::path ssd_dir_;
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::vector<std::unique_ptr<IUMBPClient>> clients_;
};

// ---------------------------------------------------------------------------
//  1. The capability
// ---------------------------------------------------------------------------

// The whole of D0.  Before it, the second EXPECT_TRUE was false: an SSD node
// could configure an arena and still be told it had no ranged I/O.
TEST_F(SsdRangedIoTest, SsdMediumSupportsRangedIoOnceTheArenaExists) {
  auto* without = Start(SsdConfig("ssd-no-scratch", /*ranged_scratch_size=*/0));
  ASSERT_NE(without, nullptr);
  EXPECT_EQ(without->GetDeploymentMode(), UMBPDeploymentMode::Distributed);
  EXPECT_FALSE(without->SupportsRangedIO()) << "the arena is still the opt-in on SSD";

  auto* with = Start(SsdConfig("ssd-with-scratch"));
  ASSERT_NE(with, nullptr);
  EXPECT_TRUE(with->SupportsRangedIO()) << "medium must no longer be a condition";
}

// ---------------------------------------------------------------------------
//  2-4. Local: the node's own SSD medium
// ---------------------------------------------------------------------------
//
// Each of these opens by asserting SupportsRangedIO(), and that is not
// ceremony.  The ranged entry points forward to PoolClient unconditionally
// (distributed_client.cpp) -- the flag is a DECLARATION callers consult, not a
// guard the data path enforces, which is why the sglang tree connector simply
// declined to ask.  So the contract worth pinning is the conjunction: the
// client says yes, AND the bytes are right.  A test that skipped the assertion
// would pass against the pre-D0 gate and pin nothing.

// A ranged put tiles its object out of scattered sources; the whole-object Get
// then proves the bytes were assembled in the right order on the device, which
// a ranged read-back alone would not (a consistently wrong mapping would agree
// with itself).
TEST_F(SsdRangedIoTest, LocalRangedPutAssemblesTheObjectOnSsd) {
  auto* node = Start(SsdConfig("ssd-local-put"));
  ASSERT_NE(node, nullptr);
  ASSERT_TRUE(node->SupportsRangedIO());

  const std::string key = "ssd-ranged-local-put";
  const auto object = Pattern(kObjectSize, 1);
  const size_t quarter = kObjectSize / 4;

  // Deliberately out of order: the ranges tile the object, but the caller
  // supplies them middle-first.
  const std::vector<uintptr_t> srcs = {Addr(object.data() + quarter), Addr(object.data()),
                                       Addr(object.data() + 2 * quarter)};
  const std::vector<size_t> sizes = {quarter, quarter, kObjectSize - 2 * quarter};
  const std::vector<size_t> offsets = {quarter, 0, 2 * quarter};

  auto put = node->BatchPutRanges({key}, {kObjectSize}, {srcs}, {sizes}, {offsets});
  ASSERT_EQ(put, std::vector<bool>({true}));
  ASSERT_TRUE(WaitForExists(node, key));

  std::vector<char> whole(kObjectSize, 0);
  auto got = node->BatchGet({key}, {Addr(whole.data())}, {whole.size()});
  ASSERT_EQ(got, std::vector<bool>({true}));
  EXPECT_EQ(whole, object) << "ranges landed at the wrong offsets on the device";
}

// The read direction, against an object written the ordinary way -- so a
// failure here is the ranged read, not a ranged write that happened to agree.
TEST_F(SsdRangedIoTest, LocalRangedGetReadsASubsetFromSsd) {
  auto* node = Start(SsdConfig("ssd-local-get"));
  ASSERT_NE(node, nullptr);
  ASSERT_TRUE(node->SupportsRangedIO());

  const std::string key = "ssd-ranged-local-get";
  const auto object = Pattern(kObjectSize, 2);
  ASSERT_EQ(node->BatchPut({key}, {Addr(object.data())}, {object.size()}),
            std::vector<bool>({true}));
  ASSERT_TRUE(WaitForExists(node, key));

  // Two disjoint windows that do NOT tile the object -- the read side is a
  // subset, which is what separates it from a whole-object read.
  std::vector<char> head(64, 0);
  std::vector<char> middle(128, 0);
  const std::vector<uintptr_t> dsts = {Addr(head.data()), Addr(middle.data())};
  const std::vector<size_t> sizes = {head.size(), middle.size()};
  const std::vector<size_t> offsets = {0, kObjectSize / 2};

  auto got = node->BatchGetRanges({key}, {dsts}, {sizes}, {offsets});
  ASSERT_EQ(got, std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(head.data(), object.data(), head.size()), 0);
  EXPECT_EQ(std::memcmp(middle.data(), object.data() + kObjectSize / 2, middle.size()), 0);
}

// A ranged get must touch the requested bytes and nothing else.  Guard bytes
// on either side of the destination catch an implementation that quietly
// copies the whole staged object out -- which, on SSD, is precisely the shape
// the staging arena makes easy to do by accident.
TEST_F(SsdRangedIoTest, LocalRangedGetWritesOnlyTheRequestedBytes) {
  auto* node = Start(SsdConfig("ssd-local-subset"));
  ASSERT_NE(node, nullptr);
  ASSERT_TRUE(node->SupportsRangedIO());

  const std::string key = "ssd-ranged-subset";
  const auto object = Pattern(kObjectSize, 3);
  ASSERT_EQ(node->BatchPut({key}, {Addr(object.data())}, {object.size()}),
            std::vector<bool>({true}));
  ASSERT_TRUE(WaitForExists(node, key));

  constexpr size_t kGuard = 32;
  constexpr size_t kWanted = 96;
  std::vector<char> buffer(kGuard + kWanted + kGuard, 0x7E);
  const std::vector<uintptr_t> dsts = {Addr(buffer.data() + kGuard)};
  const std::vector<size_t> sizes = {kWanted};
  const std::vector<size_t> offsets = {kObjectSize - kWanted};  // the object's tail

  auto got = node->BatchGetRanges({key}, {dsts}, {sizes}, {offsets});
  ASSERT_EQ(got, std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(buffer.data() + kGuard, object.data() + kObjectSize - kWanted, kWanted), 0);
  for (size_t i = 0; i < kGuard; ++i) {
    EXPECT_EQ(buffer[i], 0x7E) << "leading guard clobbered at " << i;
    EXPECT_EQ(buffer[kGuard + kWanted + i], 0x7E) << "trailing guard clobbered at " << i;
  }
}

// ---------------------------------------------------------------------------
//  5. Remote: across the wire to an SSD-only peer
// ---------------------------------------------------------------------------

// The caller has a page of DRAM and therefore no capacity, so master routes
// both operations to the SSD peer and they travel through the scratch arena.
// This is the pair the medium gate cost most: a caller honouring the flag
// declined before ever consulting master, on a deployment where EVERY node is
// SSD (pure-ssd-mode.md) and so every key is remote.
TEST_F(SsdRangedIoTest, RemoteRangedRoundTripThroughAnSsdPeer) {
  auto* caller = Start(DramConfig("ssd-remote-caller", kNoCapacity));
  ASSERT_NE(caller, nullptr);
  ASSERT_TRUE(caller->SupportsRangedIO());
  auto* target = Start(SsdConfig("ssd-remote-target"));
  ASSERT_NE(target, nullptr);
  ASSERT_TRUE(target->SupportsRangedIO());

  const std::string key = "ssd-ranged-remote";
  const auto object = Pattern(kObjectSize, 4);
  const size_t half = kObjectSize / 2;

  const std::vector<uintptr_t> srcs = {Addr(object.data()), Addr(object.data() + half)};
  const std::vector<size_t> put_sizes = {half, kObjectSize - half};
  const std::vector<size_t> put_offsets = {0, half};

  auto put = caller->BatchPutRanges({key}, {kObjectSize}, {srcs}, {put_sizes}, {put_offsets});
  ASSERT_EQ(put, std::vector<bool>({true})) << "ranged put did not reach the SSD peer";
  // Polled on the CALLER, not the holder: what the ranged get below needs is
  // not "the target has the bytes" but "the master can route the caller to
  // them", and only the caller's lookup waits for that.  Under local_first the
  // holder answers Exists out of its own backend, which is true as soon as the
  // put commits -- a publication interval before the master knows -- so polling
  // `target` here would stop waiting for the very thing this test needs.
  ASSERT_TRUE(WaitForExists(caller, key)) << "the object never landed on the SSD node";

  std::vector<char> front(200, 0);
  std::vector<char> tail(100, 0);
  const std::vector<uintptr_t> dsts = {Addr(front.data()), Addr(tail.data())};
  const std::vector<size_t> get_sizes = {front.size(), tail.size()};
  const std::vector<size_t> get_offsets = {0, kObjectSize - tail.size()};

  auto got = caller->BatchGetRanges({key}, {dsts}, {get_sizes}, {get_offsets});
  ASSERT_EQ(got, std::vector<bool>({true})) << "ranged get did not reach the SSD peer";
  EXPECT_EQ(std::memcmp(front.data(), object.data(), front.size()), 0);
  EXPECT_EQ(std::memcmp(tail.data(), object.data() + kObjectSize - tail.size(), tail.size()), 0);
}

// ---------------------------------------------------------------------------
//  6. Through the standalone UMBP server
// ---------------------------------------------------------------------------

// The topology the feature is actually deployed in: an engine talks gRPC to a
// standalone server on its node, and the SERVER is the one holding the
// DistributedClient against master.  Everything above drives DistributedClient
// in-process, so none of it exercises the hop that matters here --
// supports_ranged_io is answered by the server's client and carried on the
// Ping response (umbp_standalone.proto), then CACHED in the worker for its
// lifetime.  A capability that propagated wrongly, or not at all, would leave
// the deployment exactly where the medium gate left it: correct code that
// every caller declines to use.
TEST_F(SsdRangedIoTest, StandaloneServerOverAnSsdBackendAdvertisesAndServesRangedIo) {
  const std::string address =
      "unix:///tmp/umbp_ssd_ranged_srv_" + std::to_string(::getpid()) + ".sock";

  // The server's own config: distributed + SSD.  standalone_process is left
  // unset, so CreateUMBPClient inside the server builds a DistributedClient
  // rather than a client pointed at itself.
  auto server_cfg = SsdConfig("ssd-standalone-backend");
  standalone::StandaloneServer server(server_cfg, address);
  ASSERT_TRUE(server.Start());
  std::thread server_thread([&server]() { server.Run(); });

  {
    // The worker's config: standalone_process ONLY.  The factory tests
    // `distributed` first, so a config carrying both would silently produce a
    // second DistributedClient instead of a forwarding one.
    UMBPConfig worker_cfg;
    UMBPStandaloneProcessConfig sp_cfg;
    sp_cfg.address = address;
    sp_cfg.startup_timeout_ms = 5000;
    worker_cfg.standalone_process = sp_cfg;

    auto worker = CreateUMBPClient(worker_cfg);
    ASSERT_NE(worker, nullptr);
    EXPECT_EQ(worker->GetDeploymentMode(), UMBPDeploymentMode::StandaloneProcess);
    EXPECT_EQ(worker->GetBackendMode(), UMBPDeploymentMode::Distributed)
        << "the server must be reporting its DistributedClient, not a local one";
    ASSERT_TRUE(worker->SupportsRangedIO())
        << "the SSD backend's capability did not survive the Ping hop";

    // Standalone-process mode moves bytes over SHARED MEMORY, not over the
    // gRPC message: the worker registers an shm-backed region, the server maps
    // it, and the wire carries (region_base, offset, size) triples that the
    // server resolves back to its own mapping.  Ordinary heap pointers fail
    // ResolveRanges and come back as a per-key false -- which is what an
    // unregistered buffer SHOULD do, and is worth knowing is the failure shape.
    HostMemAllocator allocator;
    HostBufferOptions opts;
    opts.backing = HostBufferBacking::kAnonymousShm;
    opts.prefault = false;
    HostBufferHandle region = allocator.Alloc(4 * kObjectSize, opts);
    ASSERT_TRUE(region.valid());
    ASSERT_TRUE(
        worker->RegisterMemory(reinterpret_cast<uintptr_t>(region.ptr), region.mapped_size));

    auto* base = static_cast<char*>(region.ptr);
    char* src_region = base;                    // [0, kObjectSize)
    char* dst_region = base + 2 * kObjectSize;  // read destinations

    const std::string key = "ssd-ranged-standalone";
    const auto object = Pattern(kObjectSize, 5);
    std::memcpy(src_region, object.data(), kObjectSize);
    const size_t half = kObjectSize / 2;

    const std::vector<uintptr_t> srcs = {Addr(src_region), Addr(src_region + half)};
    const std::vector<size_t> put_sizes = {half, kObjectSize - half};
    const std::vector<size_t> put_offsets = {0, half};
    auto put = worker->BatchPutRanges({key}, {kObjectSize}, {srcs}, {put_sizes}, {put_offsets});
    ASSERT_EQ(put, std::vector<bool>({true}));
    ASSERT_TRUE(WaitForExists(worker.get(), key));

    constexpr size_t kHead = 96;
    constexpr size_t kTail = 64;
    char* head = dst_region;
    char* tail = dst_region + kObjectSize;  // a second, non-adjacent destination
    std::memset(head, 0, kHead);
    std::memset(tail, 0, kTail);
    const std::vector<uintptr_t> dsts = {Addr(head), Addr(tail)};
    const std::vector<size_t> get_sizes = {kHead, kTail};
    const std::vector<size_t> get_offsets = {0, kObjectSize - kTail};
    auto got = worker->BatchGetRanges({key}, {dsts}, {get_sizes}, {get_offsets});
    ASSERT_EQ(got, std::vector<bool>({true}));
    EXPECT_EQ(std::memcmp(head, object.data(), kHead), 0);
    EXPECT_EQ(std::memcmp(tail, object.data() + kObjectSize - kTail, kTail), 0);

    worker->DeregisterMemory(reinterpret_cast<uintptr_t>(region.ptr));
    allocator.Free(region);

    worker->Close();
  }

  server.Shutdown();
  if (server_thread.joinable()) server_thread.join();
}

}  // namespace
}  // namespace mori::umbp
