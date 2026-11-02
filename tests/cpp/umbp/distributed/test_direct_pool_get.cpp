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
// Real pool allocation, remote transfer-item construction, range state and
// scatter; only peer resolution and the physical wire are replaced by host RAM.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "umbp/distributed/direct_pool_get.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/transfer/local_copy_engine.h"

namespace mori::umbp {
namespace {
class TestDirectory : public PeerDirectory {
 public:
  TransferRef source;
  std::vector<uint8_t> PackedLocalEngineDesc() const override { return {}; }
  bool EnsureRemoteEngine(const std::string&, const std::string&) override { return true; }
  bool HasRemoteEngine(const std::string&) const override { return true; }
  void ForgetRemote(const std::string&) override {}
  void CacheRemoteBuffers(const std::string&, const std::vector<BufferMemoryDescBytes>&) override {}
  bool HasRemoteBuffers(const std::string&) const override { return true; }
  std::vector<TransferRef> RemoteBufferSnapshot(const std::string&, uint32_t) const override {
    return {source};
  }
};
}  // namespace
class DirectPoolGetTest : public ::testing::Test {
 protected:
  static constexpr size_t kPage = 1024, kSourcePage = 2 * kPage, kSize = 5 * kPage;
  using Plan = PoolClient::BatchGetPlan;
  using Fetch = std::function<void(const Plan&, std::vector<bool>*)>;
  void SetUp() override {
    source_.resize(kSize);
    for (size_t i = 0; i < source_.size(); ++i) source_[i] = (i * 17 + i / kPage) % 251;
    // No allocation can fit in one backing buffer: exercises mapping of
    // destination page boundaries as well as source page boundaries.
    storage_.assign(8, std::vector<char>(2 * kPage, char(0xcc)));
    PageBackend::TierConfig tier;
    for (auto& bytes : storage_) {
      tier.buffer_sizes.push_back(bytes.size());
      tier.buffer_bases.push_back(bytes.data());
      tier.buffer_descs.push_back({});
    }
    auto backend =
        std::make_unique<PageBackend>(TierType::DRAM, kPage, std::move(tier),
                                      std::chrono::milliseconds(1), std::chrono::seconds(60));
    backend_ = backend.get();
    PoolClientConfig cfg;
    cfg.direct_pool_get = true;
    client_ = std::make_unique<PoolClient>(cfg);
    client_->transfer_engine_ = std::make_unique<LocalCopyEngine>();
    ASSERT_TRUE(client_->registry_.Register(std::move(backend)));
    client_->direct_pool_get_cache_ = std::make_unique<DirectPoolGetCache>(16 * kPage);
    directory_.source = TransferRef::HostBytes(source_.data(), source_.size());
    directory_.source.mem.size = source_.size();  // required remote-descriptor presence
    client_->peer_directory_ = &directory_;
  }
  DirectPoolGetCache& Cache() { return *client_->direct_pool_get_cache_; }
  void TearDown() override {
    Cache().Clear();
    client_.reset();
  }
  void Wire(const Plan& plan, std::vector<bool>* results) {
    std::vector<PoolClient::RemoteGetEntry> entries;
    for (const auto& [peer, items] : plan.remote_groups) {
      for (const auto& item : items) {
        PoolClient::RemoteGetEntry entry{};
        entry.item = &item;
        entry.result_index = item.index;
        entry.plan.page_size = kSourcePage;
        for (size_t i = 0; i < (kSize + kSourcePage - 1) / kSourcePage; ++i)
          entry.plan.pages.push_back(PageLocation{0, static_cast<uint32_t>(i)});
        entries.push_back(std::move(entry));
        wire_keys_.fetch_add(1);
      }
    }
    std::vector<TransferItem> items;
    EXPECT_TRUE(client_->BuildRemoteGetTransfers(entries, "source", &items));
    std::vector<size_t> failed;
    EXPECT_TRUE(client_->transfer_engine_->Transfer(items, &failed));
    EXPECT_TRUE(failed.empty());
    for (const auto& item : items) wire_bytes_.fetch_add(item.size);
    for (const auto& entry : entries) {
      EXPECT_FALSE(entry.failed);
      (*results)[entry.result_index] = !entry.failed;
    }
  }
  bool Get(const std::string& key, const std::vector<size_t>& offsets,
           const std::vector<size_t>& sizes, std::vector<char>* out, Fetch fetch = {}) {
    size_t total = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
    out->assign(total, char(0xee));
    EXPECT_TRUE(client_->RegisterMemory(out->data(), out->size()));
    std::vector<void*> dsts;
    size_t cursor = 0;
    for (size_t size : sizes) {
      dsts.push_back(out->data() + cursor);
      cursor += size;
    }
    RouteGetResult route{};
    route.node_id = "source";
    route.size = kSize;
    std::vector<bool> result(1, false);
    double remote_bytes = 0;
    if (!fetch) fetch = [&](const Plan& plan, std::vector<bool>* ok) { Wire(plan, ok); };
    try {
      client_->ServeRangesFromDirectPool({key}, {dsts}, {sizes}, {offsets}, {0}, {route}, &result,
                                         &remote_bytes, fetch);
    } catch (...) {
      client_->DeregisterMemory(out->data());
      throw;
    }
    client_->DeregisterMemory(out->data());
    return result[0];
  }
  void Check(const std::vector<char>& out, const std::vector<size_t>& offsets,
             const std::vector<size_t>& sizes) {
    size_t cursor = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      EXPECT_EQ(0, std::memcmp(out.data() + cursor, source_.data() + offsets[i], sizes[i]));
      cursor += sizes[i];
    }
  }
  std::vector<char> source_;
  std::vector<std::vector<char>> storage_;
  TestDirectory directory_;
  PageBackend* backend_;
  std::unique_ptr<PoolClient> client_;
  std::atomic<size_t> wire_bytes_{0}, wire_keys_{0};
};

TEST_F(DirectPoolGetTest, PartialUnorderedRangesStayPrivateThenFetchOnlyMissingBytes) {
  std::vector<char> out;
  ASSERT_TRUE(Get("partial", {2500, 17}, {1800, 1100}, &out));
  Check(out, {2500, 17}, {1800, 1100});
  EXPECT_FALSE(backend_->Resolve("partial").found);
  EXPECT_EQ(wire_bytes_, 2900u);
  ASSERT_TRUE(Get("partial", {500, 3000}, {400, 1000}, &out));
  Check(out, {500, 3000}, {400, 1000});
  EXPECT_EQ(wire_bytes_, 2900u);
  ASSERT_TRUE(Get("partial", {0}, {kSize}, &out));
  Check(out, {0}, {kSize});
  EXPECT_EQ(wire_bytes_, kSize);
  EXPECT_TRUE(backend_->Resolve("partial").found);
  EXPECT_EQ(Cache().Bytes(), 0u);
}

TEST_F(DirectPoolGetTest, EightReadersShareOneRangeFetch) {
  std::vector<std::future<bool>> readers;
  for (int rank = 0; rank < 8; ++rank)
    readers.push_back(std::async(std::launch::async, [&] {
      std::vector<char> out;
      bool ok = Get("shared", {300}, {2400}, &out, [&](const Plan& p, std::vector<bool>* r) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Wire(p, r);
      });
      if (ok) Check(out, {300}, {2400});
      return ok;
    }));
  for (auto& reader : readers) EXPECT_TRUE(reader.get());
  EXPECT_EQ(wire_keys_, 1u);
  EXPECT_EQ(wire_bytes_, 2400u);
}

TEST_F(DirectPoolGetTest, FailureAndExceptionDiscardPrivateBytesForRetry) {
  std::vector<char> out;
  EXPECT_FALSE(Get("failure", {0}, {kSize}, &out, [](const Plan&, std::vector<bool>*) {}));
  EXPECT_EQ(out, std::vector<char>(kSize, char(0xee)));
  EXPECT_FALSE(backend_->Resolve("failure").found);
  EXPECT_EQ(Cache().Bytes(), 0u);
  EXPECT_THROW(Get("failure", {0}, {2000}, &out,
                   [](const Plan&, std::vector<bool>*) {
                     throw std::runtime_error("injected transport failure");
                   }),
               std::runtime_error);
  ASSERT_TRUE(Get("failure", {0}, {kSize}, &out));
  Check(out, {0}, {kSize});
}

TEST_F(DirectPoolGetTest, ReaperAndDuplicateCommitWaitForLastReader) {
  auto reader = Cache().Acquire(backend_, "pinned", kSize);
  ASSERT_TRUE(reader);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  backend_->RunReaperOnceForTest();
  auto slots = backend_->BatchAllocate({{"too-large", 12 * kPage}});
  EXPECT_EQ(slots[0].outcome, AllocateOutcome::kFailedNoSpace);
  std::vector<char> out;
  ASSERT_TRUE(Get("pinned", {0}, {kSize}, &out));
  Check(out, {0}, {kSize});
  EXPECT_FALSE(backend_->Resolve("pinned").found);
  // A competing PUT commits while a reader still holds the private replica.
  auto writer = backend_->BatchAllocate({{"pinned", kSize}});
  ASSERT_EQ(writer[0].outcome, AllocateOutcome::kSuccessAllocated);
  for (const auto& page : writer[0].pages) {
    auto ref = backend_->BufferRef(page.buffer_index);
    std::memset(static_cast<char*>(ref.host_ptr) + page.page_index * kPage, 0x5a, kPage);
  }
  ASSERT_TRUE(backend_->BatchCommit({{writer[0].slot_id, "pinned"}})[0].success);
  ASSERT_TRUE(Get("pinned", {0}, {kSize}, &out));
  Check(out, {0}, {kSize});  // The outstanding reader still sees its intact pages.
  reader.reset();
  Cache().Trim();
  auto committed = backend_->Resolve("pinned");
  ASSERT_TRUE(committed.found);
  for (const auto& page : committed.pages) {
    auto ref = backend_->BufferRef(page.buffer_index);
    const char* bytes = static_cast<char*>(ref.host_ptr) + page.page_index * kPage;
    EXPECT_EQ(std::vector<char>(bytes, bytes + kPage), std::vector<char>(kPage, char(0x5a)));
  }
  EXPECT_EQ(Cache().Bytes(), 0u);
  EXPECT_EQ(backend_->Capacity().available_bytes, 11 * kPage);
}

TEST_F(DirectPoolGetTest, PressureAndClearReclaimIncompleteReplicas) {
  std::vector<char> out;
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(Get("key-" + std::to_string(i), {123}, {2345}, &out));
    Check(out, {123}, {2345});
    EXPECT_LE(Cache().Bytes(), 16 * kPage);
  }
  EXPECT_FALSE(backend_->Resolve("key-0").found);
  Cache().Clear();
  EXPECT_EQ(Cache().Bytes(), 0u);
  EXPECT_TRUE(backend_->DrainPendingEvents().empty());
  auto slots = backend_->BatchAllocate({{"all", 16 * kPage}});
  EXPECT_EQ(slots[0].outcome, AllocateOutcome::kSuccessAllocated);
  backend_->BatchAbort({slots[0].slot_id});
}

TEST_F(DirectPoolGetTest, EightRanksRestoreMultipleGroupsWithoutDuplicateWireBytes) {
  std::vector<std::future<bool>> ranks;
  for (size_t rank = 0; rank < 8; ++rank)
    ranks.push_back(std::async(std::launch::async, [&, rank] {
      for (size_t group = 0; group < 5; ++group) {
        for (size_t k = 0; k < 3; ++k) {
          const size_t key = rank % 2 ? k : 2 - k;
          const size_t offset = ((group + rank) % 5) * kPage;
          std::vector<char> out;
          if (!Get("multi-" + std::to_string(key), {offset}, {kPage}, &out)) return false;
          Check(out, {offset}, {kPage});
        }
      }
      return true;
    }));
  for (auto& rank : ranks) EXPECT_TRUE(rank.get());
  EXPECT_EQ(wire_bytes_, 3 * kSize);
  EXPECT_EQ(Cache().Bytes(), 0u);
}

}  // namespace mori::umbp
