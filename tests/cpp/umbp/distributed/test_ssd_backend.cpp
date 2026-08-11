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

// SsdBackend: SSD reaching the data plane as an ordinary MediumBackend by
// staging through registered host DRAM.
//
// What is worth asserting here is the part that differs from DRAM — the spill
// on Commit, the fill on Resolve, and the staging arena that both borrow from —
// plus the invariant that makes the whole design work: what the backend
// PUBLISHES is plain host memory, so no transfer-layer change was needed.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/peer/backend/ssd_backend.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

constexpr uint64_t kPageSize = 64 * 1024;

// Hands back process-local refs only — what CompositeTransferEngine degrades to
// on a node with no RDMA, and enough to exercise ownership without an IOEngine.
class LocalOnlyRegistrar final : public MemoryRegistrar {
 public:
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    ++registrations;
    last_loc = loc;
    return TransferRef::HostBytes(base, size, loc, device);
  }
  void Deregister(const TransferRef&) override { ++deregistrations; }

  int registrations = 0;
  int deregistrations = 0;
  mori::io::MemoryLocationType last_loc = mori::io::MemoryLocationType::GPU;
};

class SsdBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = fs::temp_directory_path() / ("umbp_ssd_backend_test_" + std::to_string(::getpid()) +
                                        "_" + std::to_string(counter.fetch_add(1)));
    fs::remove_all(dir_);
  }

  void TearDown() override {
    if (backend_ != nullptr) backend_->Shutdown();
    backend_.reset();
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  // `staging_pages` is the backend's real concurrency limit, so tests that care
  // about exhaustion set it small.
  SsdBackend* Start(uint32_t staging_pages = 8,
                    std::chrono::milliseconds read_lease_ttl = std::chrono::milliseconds{500}) {
    SsdBackend::Config cfg;
    cfg.page_size = kPageSize;
    cfg.staging_pages = staging_pages;
    cfg.read_lease_ttl = read_lease_ttl;
    cfg.ssd.enabled = true;
    cfg.ssd.ssd.enabled = true;
    cfg.ssd.ssd.storage_dir = dir_.string();
    cfg.ssd.ssd.capacity_bytes = 64ULL * 1024 * 1024;
    cfg.ssd.ssd.io.backend = UMBPIoBackend::Posix;  // avoid io_uring container flakiness

    auto owned = std::make_unique<SsdBackend>(std::move(cfg));
    backend_ = std::move(owned);
    EXPECT_TRUE(backend_->Init(&registrar_));
    return backend_.get();
  }

  // The full Put path a real writer takes: reserve a page, move bytes into it
  // through the transfer layer, commit.
  bool Put(SsdBackend* backend, const std::string& key, const std::vector<char>& payload) {
    auto allocated = backend->BatchAllocate({AllocateRequest{key, payload.size()}});
    if (allocated.size() != 1 || allocated[0].outcome != AllocateOutcome::kSuccessAllocated) {
      return false;
    }
    const TransferRef pool = backend->BufferRef(allocated[0].pages[0].buffer_index);
    const uint64_t off = static_cast<uint64_t>(allocated[0].pages[0].page_index) * kPageSize;

    TransferItem item;
    item.src = TransferRef::HostBytes(const_cast<char*>(payload.data()), payload.size());
    item.dst = pool;
    item.dst_offset = off;
    item.size = payload.size();
    std::vector<size_t> failed;
    if (!engine_.Transfer({item}, &failed)) return false;

    auto committed = backend->BatchCommit({CommitRequest{allocated[0].slot_id, key}});
    return committed.size() == 1 && committed[0].success;
  }

  // The full Get path: resolve (which stages off the SSD), then move bytes out.
  bool Get(SsdBackend* backend, const std::string& key, std::vector<char>* out) {
    auto resolved = backend->BatchResolve({key}, /*include_descs=*/false);
    if (resolved.size() != 1 || !resolved[0].found) return false;
    out->assign(resolved[0].size, 0);

    const TransferRef pool = backend->BufferRef(resolved[0].pages[0].buffer_index);
    const uint64_t off = static_cast<uint64_t>(resolved[0].pages[0].page_index) * kPageSize;

    TransferItem item;
    item.src = pool;
    item.src_offset = off;
    item.dst = TransferRef::HostBytes(out->data(), out->size());
    item.size = resolved[0].size;
    std::vector<size_t> failed;
    return engine_.Transfer({item}, &failed);
  }

  static std::vector<char> Payload(size_t n, char seed) {
    std::vector<char> v(n);
    std::iota(v.begin(), v.end(), seed);
    return v;
  }

  SsdBackendTest() { engine_.AddEngine(std::make_unique<LocalCopyEngine>()); }

  fs::path dir_;
  LocalOnlyRegistrar registrar_;
  CompositeTransferEngine engine_;
  std::unique_ptr<SsdBackend> backend_;
};

// ---------------------------------------------------------------------------
//  The design invariant
// ---------------------------------------------------------------------------

// The reason SSD needed no new engine, no new TransferRef kind and no chaining:
// what it publishes is ordinary registered HOST memory.  If this ever changes,
// every claim in ssd_backend.h's header comment stops holding.
TEST_F(SsdBackendTest, PublishesPlainHostMemorySoNoNewEngineIsNeeded) {
  SsdBackend* backend = Start();

  EXPECT_EQ(backend->Tier(), TierType::SSD);
  EXPECT_EQ(registrar_.registrations, 1);  // one arena, one registration
  EXPECT_EQ(registrar_.last_loc, mori::io::MemoryLocationType::CPU);

  ASSERT_EQ(backend->BufferCount(), 1u);
  const TransferRef ref = backend->BufferRef(0);
  ASSERT_TRUE(ref.Valid());
  EXPECT_TRUE(ref.HasHostPtr());
  EXPECT_EQ(ref.loc, mori::io::MemoryLocationType::CPU);

  // Which means the plain host engine claims its pairs, with nothing new added.
  LocalCopyEngine local;
  EXPECT_TRUE(local.CanHandle(ref, ref));
}

TEST_F(SsdBackendTest, ReportsAnOutOfRangeBufferAsInvalid) {
  SsdBackend* backend = Start();
  EXPECT_FALSE(backend->BufferRef(1).Valid());
}

// ---------------------------------------------------------------------------
//  Put / Get
// ---------------------------------------------------------------------------

TEST_F(SsdBackendTest, PutThenGetReturnsTheSameBytes) {
  SsdBackend* backend = Start();
  const auto payload = Payload(4096, 5);

  ASSERT_TRUE(Put(backend, "key-a", payload));

  std::vector<char> readback;
  ASSERT_TRUE(Get(backend, "key-a", &readback));
  ASSERT_EQ(readback.size(), payload.size());
  EXPECT_EQ(std::memcmp(readback.data(), payload.data(), payload.size()), 0);
}

TEST_F(SsdBackendTest, GetOfAnUnknownKeyMisses) {
  SsdBackend* backend = Start();
  auto resolved = backend->BatchResolve({"never-written"}, false);
  ASSERT_EQ(resolved.size(), 1u);
  EXPECT_FALSE(resolved[0].found);
}

TEST_F(SsdBackendTest, CommitQueuesAnSsdAddEvent) {
  SsdBackend* backend = Start();
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));

  const auto events = backend->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[0].key, "key-a");
  EXPECT_EQ(events[0].tier, TierType::SSD);

  EXPECT_TRUE(backend->DrainPendingEvents().empty());  // outbox cleared
}

TEST_F(SsdBackendTest, MultipleKeysRoundTripIndependently) {
  SsdBackend* backend = Start();
  const auto a = Payload(2048, 1);
  const auto b = Payload(4096, 64);

  ASSERT_TRUE(Put(backend, "key-a", a));
  ASSERT_TRUE(Put(backend, "key-b", b));
  EXPECT_EQ(backend->OwnedKeyCount(), 2u);

  std::vector<char> got_a, got_b;
  ASSERT_TRUE(Get(backend, "key-a", &got_a));
  ASSERT_TRUE(Get(backend, "key-b", &got_b));
  EXPECT_EQ(got_a, a);
  EXPECT_EQ(got_b, b);
}

// ---------------------------------------------------------------------------
//  The staging arena — what actually differs from DRAM
// ---------------------------------------------------------------------------

// A staging page is BORROWED for a write, not the key's home.  If Commit leaked
// it the backend would wedge after `staging_pages` puts, which is the bug this
// guards: with 2 pages, 10 sequential puts must all succeed.
TEST_F(SsdBackendTest, CommitReturnsTheStagingPageToTheArena) {
  SsdBackend* backend = Start(/*staging_pages=*/2);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(Put(backend, "key-" + std::to_string(i), Payload(512, static_cast<char>(i))))
        << "put " << i << " failed — staging page leaked?";
  }
  EXPECT_EQ(backend->OwnedKeyCount(), 10u);
}

TEST_F(SsdBackendTest, AbortReturnsTheStagingPage) {
  SsdBackend* backend = Start(/*staging_pages=*/1);

  auto first = backend->BatchAllocate({AllocateRequest{"key-a", 1024}});
  ASSERT_EQ(first[0].outcome, AllocateOutcome::kSuccessAllocated);

  // The only page is taken, so a second reservation must report exhaustion.
  auto blocked = backend->BatchAllocate({AllocateRequest{"key-b", 1024}});
  EXPECT_EQ(blocked[0].outcome, AllocateOutcome::kFailedNoSpace);

  ASSERT_TRUE(backend->BatchAbort({first[0].slot_id})[0]);

  auto after = backend->BatchAllocate({AllocateRequest{"key-b", 1024}});
  EXPECT_EQ(after[0].outcome, AllocateOutcome::kSuccessAllocated);
}

TEST_F(SsdBackendTest, AbortIsIdempotent) {
  SsdBackend* backend = Start();
  auto allocated = backend->BatchAllocate({AllocateRequest{"key-a", 1024}});
  ASSERT_TRUE(backend->BatchAbort({allocated[0].slot_id})[0]);
  EXPECT_TRUE(backend->BatchAbort({allocated[0].slot_id})[0]);  // already gone
  EXPECT_TRUE(backend->BatchAbort({99999})[0]);                 // never existed
}

TEST_F(SsdBackendTest, CommitOfAnAbortedSlotFails) {
  SsdBackend* backend = Start();
  auto allocated = backend->BatchAllocate({AllocateRequest{"key-a", 1024}});
  ASSERT_TRUE(backend->BatchAbort({allocated[0].slot_id})[0]);

  auto committed = backend->BatchCommit({CommitRequest{allocated[0].slot_id, "key-a"}});
  EXPECT_FALSE(committed[0].success);
}

// A key bigger than one page is refused outright rather than reported as "no
// space": no other peer would do better, so the writer must not retry elsewhere.
TEST_F(SsdBackendTest, RejectsAKeyLargerThanOnePage) {
  SsdBackend* backend = Start();
  auto allocated = backend->BatchAllocate({AllocateRequest{"too-big", kPageSize + 1}});
  ASSERT_EQ(allocated.size(), 1u);
  EXPECT_EQ(allocated[0].outcome, AllocateOutcome::kFailed);
}

TEST_F(SsdBackendTest, DedupsAKeyAlreadyOnTheTier) {
  SsdBackend* backend = Start();
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));

  auto again = backend->BatchAllocate({AllocateRequest{"key-a", 1024}});
  EXPECT_EQ(again[0].outcome, AllocateOutcome::kSuccessAlreadyExists);
}

// The read lease is what keeps staged bytes alive while a reader consumes them;
// the reaper is what stops that from being a leak.
TEST_F(SsdBackendTest, ReaperReclaimsStagingPagesFromExpiredReadLeases) {
  SsdBackend* backend = Start(/*staging_pages=*/1, /*read_lease_ttl=*/std::chrono::milliseconds{1});
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));

  auto resolved = backend->BatchResolve({"key-a"}, false);
  ASSERT_TRUE(resolved[0].found);

  // The single page is now leased, so a write cannot get one.
  EXPECT_EQ(backend->BatchAllocate({AllocateRequest{"key-b", 512}})[0].outcome,
            AllocateOutcome::kFailedNoSpace);

  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  backend->RunReaperOnceForTest();

  EXPECT_EQ(backend->BatchAllocate({AllocateRequest{"key-b", 512}})[0].outcome,
            AllocateOutcome::kSuccessAllocated);
}

// A second reader of an already-staged key reuses the page rather than issuing
// a redundant SSD read — and, with one page configured, could not get a second
// page even if it tried.
TEST_F(SsdBackendTest, ConcurrentResolvesShareOneStagedPage) {
  SsdBackend* backend = Start(/*staging_pages=*/1);
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 9)));

  auto first = backend->BatchResolve({"key-a"}, false);
  ASSERT_TRUE(first[0].found);
  auto second = backend->BatchResolve({"key-a"}, false);
  ASSERT_TRUE(second[0].found);

  EXPECT_EQ(first[0].pages[0].page_index, second[0].pages[0].page_index);
  EXPECT_EQ(first[0].size, second[0].size);
}

// Exhaustion degrades a Get to a miss.  Asserted because it is a KNOWN wrong
// shape (the client will retry another peer for a key this node does hold) —
// documented in ssd_backend.h, and pinned here so a future control-plane fix
// has a test to change deliberately rather than discovering the behavior.
TEST_F(SsdBackendTest, ResolveDegradesToAMissWhenStagingIsExhausted) {
  SsdBackend* backend = Start(/*staging_pages=*/1);
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));
  ASSERT_TRUE(Put(backend, "key-b", Payload(1024, 2)));

  auto a = backend->BatchResolve({"key-a"}, false);
  ASSERT_TRUE(a[0].found);  // takes the only page and holds it under a lease

  auto b = backend->BatchResolve({"key-b"}, false);
  EXPECT_FALSE(b[0].found) << "expected the documented degrade-to-miss";
}

// ---------------------------------------------------------------------------
//  Eviction and Clear
// ---------------------------------------------------------------------------

TEST_F(SsdBackendTest, EvictFreesTheKeyAndReportsItsBytes) {
  SsdBackend* backend = Start();
  const auto payload = Payload(2048, 3);
  ASSERT_TRUE(Put(backend, "key-a", payload));
  backend->DrainPendingEvents();  // discard the ADD

  auto evicted = backend->Evict({"key-a"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0].key, "key-a");
  EXPECT_EQ(evicted[0].bytes_freed, payload.size());

  EXPECT_FALSE(backend->BatchResolve({"key-a"}, false)[0].found);

  const auto events = backend->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::REMOVE);
  EXPECT_EQ(events[0].tier, TierType::SSD);
}

TEST_F(SsdBackendTest, EvictOfAnUnknownKeyFreesNothing) {
  SsdBackend* backend = Start();
  auto evicted = backend->Evict({"nope"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0].bytes_freed, 0u);
}

// A staged read holds the bytes live; master retries on a later round.
TEST_F(SsdBackendTest, EvictSkipsAKeyWithALiveReadLease) {
  SsdBackend* backend = Start();
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));
  ASSERT_TRUE(backend->BatchResolve({"key-a"}, false)[0].found);

  auto evicted = backend->Evict({"key-a"});
  EXPECT_EQ(evicted[0].bytes_freed, 0u);
  EXPECT_TRUE(backend->BatchResolve({"key-a"}, false)[0].found);  // still there
}

TEST_F(SsdBackendTest, EvictResultsAreOnePerKeyInRequestOrder) {
  SsdBackend* backend = Start();
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));

  auto evicted = backend->Evict({"missing-1", "key-a", "missing-2"});
  ASSERT_EQ(evicted.size(), 3u);
  EXPECT_EQ(evicted[0].key, "missing-1");
  EXPECT_EQ(evicted[1].key, "key-a");
  EXPECT_EQ(evicted[2].key, "missing-2");
  EXPECT_EQ(evicted[1].bytes_freed, 1024u);
}

TEST_F(SsdBackendTest, ClearLocalDropsEverythingAndGatesAllocation) {
  SsdBackend* backend = Start();
  ASSERT_TRUE(Put(backend, "key-a", Payload(1024, 1)));

  backend->ClearLocal();
  EXPECT_TRUE(backend->IsClearFullSyncPending());
  EXPECT_EQ(backend->OwnedKeyCount(), 0u);

  // Gated: no new owned key may appear before master acks the empty full sync.
  EXPECT_EQ(backend->BatchAllocate({AllocateRequest{"key-b", 512}})[0].outcome,
            AllocateOutcome::kFailed);

  backend->ClearFullSyncAcked();
  EXPECT_FALSE(backend->IsClearFullSyncPending());
  EXPECT_EQ(backend->BatchAllocate({AllocateRequest{"key-b", 512}})[0].outcome,
            AllocateOutcome::kSuccessAllocated);
}

TEST_F(SsdBackendTest, ShutdownDeregistersTheArena) {
  SsdBackend* backend = Start();
  backend->Shutdown();
  EXPECT_EQ(registrar_.deregistrations, 1);
  backend->Shutdown();  // idempotent
  EXPECT_EQ(registrar_.deregistrations, 1);
}

}  // namespace
}  // namespace mori::umbp
