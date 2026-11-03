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
#include <future>
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
    for (size_t i = 0; i < kSize; ++i) source_[i] = (i * 17 + i / kPage) % 251;
    // Destination spans several buffers; source and destination page sizes differ.
    storage_.assign(8, std::vector<char>(2 * kPage));
    PageBackend::TierConfig tier;
    for (auto& bytes : storage_) {
      tier.buffer_sizes.push_back(bytes.size());
      tier.buffer_bases.push_back(bytes.data());
      tier.buffer_descs.push_back({});
    }
    auto backend = std::make_unique<PageBackend>(TierType::DRAM, kPage, std::move(tier),
                                                 std::chrono::milliseconds(1));
    backend_ = backend.get();
    client_ = std::make_unique<PoolClient>(PoolClientConfig{.direct_pool_get = true});
    client_->transfer_engine_ = std::make_unique<LocalCopyEngine>();
    ASSERT_TRUE(client_->registry_.Register(std::move(backend)));
    client_->direct_pool_get_cache_ = std::make_unique<DirectPoolGetCache>(16 * kPage);
    directory_.source = TransferRef::HostBytes(source_.data(), kSize);
    directory_.source.mem.size = kSize;
    client_->peer_directory_ = &directory_;
  }
  DirectPoolGetCache& Cache() { return *client_->direct_pool_get_cache_; }
  void Wire(const Plan& plan, std::vector<bool>* results) {
    std::vector<PoolClient::RemoteGetEntry> entries;
    for (const auto& item : plan.remote_groups.at("source")) {
      PoolClient::RemoteGetEntry entry{};
      entry.item = &item;
      entry.result_index = item.index;
      entry.plan.page_size = kSourcePage;
      entry.plan.pages = {{0, 0}, {0, 1}, {0, 2}};
      entries.push_back(std::move(entry));
    }
    std::vector<TransferItem> items;
    ASSERT_TRUE(client_->BuildRemoteGetTransfers(entries, "source", &items));
    std::vector<size_t> failed;
    ASSERT_TRUE(client_->transfer_engine_->Transfer(items, &failed));
    ASSERT_TRUE(failed.empty());
    for (const auto& item : items) wire_bytes_ += item.size;
    for (const auto& entry : entries) (*results)[entry.result_index] = !entry.failed;
  }
  bool Get(const std::string& key, const std::vector<PoolClient::ByteSpan>& ranges,
           Fetch fetch = {}) {
    std::vector<char> out(kSize, char(0xee)), expected = out;
    std::vector<size_t> sizes, offsets;
    std::vector<void*> dsts;
    for (const auto& range : ranges) {
      sizes.push_back(range.size);
      offsets.push_back(range.object_offset);
      dsts.push_back(out.data() + range.object_offset);
      std::memcpy(expected.data() + range.object_offset, source_.data() + range.object_offset,
                  range.size);
    }
    RouteGetResult route{.node_id = "source", .size = kSize};
    std::vector<bool> result(1, false);
    double remote_bytes = 0;
    if (!fetch) fetch = [&](const Plan& plan, std::vector<bool>* ok) { Wire(plan, ok); };
    client_->ServeRangesFromDirectPool({key}, {dsts}, {sizes}, {offsets}, {0}, {route}, &result,
                                       &remote_bytes, fetch);
    EXPECT_EQ(out, result[0] ? expected : std::vector<char>(kSize, char(0xee)));
    return result[0];
  }
  std::vector<char> source_;
  std::vector<std::vector<char>> storage_;
  TestDirectory directory_;
  PageBackend* backend_;
  std::unique_ptr<PoolClient> client_;
  std::atomic<size_t> wire_bytes_{0};
};

TEST_F(DirectPoolGetTest, PartialRangesFetchOnlyMissingBytesUntilPublication) {
  ASSERT_TRUE(Get("partial", {{2500, 1800}, {17, 1100}}));
  EXPECT_FALSE(backend_->Resolve("partial").found);
  EXPECT_EQ(wire_bytes_, 2900u);
  ASSERT_TRUE(Get("partial", {{500, 400}, {3000, 1000}}));
  EXPECT_EQ(wire_bytes_, 2900u);
  ASSERT_TRUE(Get("partial", {{0, kSize}}));
  EXPECT_EQ(wire_bytes_, kSize);
  EXPECT_TRUE(backend_->Resolve("partial").found);
  EXPECT_EQ(Cache().Bytes(), 0u);
}

TEST_F(DirectPoolGetTest, ConcurrentReadersShareFetchAndFailedOwnersCanRetry) {
  std::vector<std::future<bool>> readers;
  for (int rank = 0; rank < 8; ++rank)
    readers.push_back(std::async(std::launch::async, [&] {
      return Get("shared", {{300, 2400}}, [&](const Plan& p, std::vector<bool>* r) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Wire(p, r);
      });
    }));
  for (auto& reader : readers) EXPECT_TRUE(reader.get());
  EXPECT_EQ(wire_bytes_, 2400u);
  Cache().Clear();
  EXPECT_FALSE(Get("failure", {{0, kSize}}, [](const Plan&, std::vector<bool>*) {}));
  EXPECT_EQ(Cache().Bytes(), 0u);
  auto fail = [](const Plan&, std::vector<bool>*) { throw std::runtime_error("wire failure"); };
  EXPECT_THROW(Get("failure", {{0, kSize}}, fail), std::runtime_error);
  EXPECT_TRUE(Get("failure", {{0, kSize}}));
}

TEST_F(DirectPoolGetTest, ReaperAndDuplicateCommitWaitForLastReader) {
  auto reader = Cache().Acquire(backend_, "pinned", kSize);
  ASSERT_TRUE(reader);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  backend_->RunReaperOnceForTest();
  EXPECT_EQ(backend_->BatchAllocate({{"too-large", 12 * kPage}})[0].outcome,
            AllocateOutcome::kFailedNoSpace);
  ASSERT_TRUE(Get("pinned", {{0, kSize}}));
  EXPECT_FALSE(backend_->Resolve("pinned").found);
  // A PUT commits first, while the private replica still has an outstanding reader.
  auto writer = backend_->BatchAllocate({{"pinned", kSize}});
  ASSERT_EQ(writer[0].outcome, AllocateOutcome::kSuccessAllocated);
  for (const auto& page : writer[0].pages)
    std::memset(storage_[page.buffer_index].data() + page.page_index * kPage, 0x5a, kPage);
  ASSERT_TRUE(backend_->BatchCommit({{writer[0].slot_id, "pinned"}})[0].success);
  ASSERT_TRUE(Get("pinned", {{0, kSize}}));
  reader.reset();
  Cache().Trim();
  auto committed = backend_->Resolve("pinned");
  ASSERT_TRUE(committed.found);
  for (const auto& page : committed.pages) {
    const char* bytes = storage_[page.buffer_index].data() + page.page_index * kPage;
    EXPECT_EQ(std::vector<char>(bytes, bytes + kPage), std::vector<char>(kPage, char(0x5a)));
  }
  EXPECT_EQ(Cache().Bytes(), 0u);
  EXPECT_EQ(backend_->Capacity().available_bytes, 11 * kPage);
}

TEST_F(DirectPoolGetTest, BudgetAndClearReclaimIncompleteReplicas) {
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(Get("key-" + std::to_string(i), {{123, 2345}}));
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
}  // namespace mori::umbp
