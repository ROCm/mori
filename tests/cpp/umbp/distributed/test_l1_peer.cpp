// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU-only tests of the production peer state machine. The injected backend
// uses host bytes; descriptor parsing, slot ownership and peer grants are real.
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "umbp/l1/l1_pool.h"

namespace mori::umbp::l1 {
namespace {
constexpr uint32_t kSlots = 8;
constexpr uint64_t kBytes = 160;
const Layout kLayout{8, {{0, 16}, {128, 4}}};
using Directory = std::map<std::string, std::vector<unsigned char>>;

// L1FBR001 wire descriptor with a nonzero opaque token, a bounded view and a
// test fabric domain. No HIP handle is ever opened by this test backend.
std::string Descriptor(unsigned char token, uint64_t bytes) {
  std::string out = "L1FBR001";
  out.append(64, static_cast<char>(token));
  for (uint64_t field : {bytes, uint64_t{0}, bytes, uint64_t{2}, uint64_t{4}})
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>(field >> (8 * i)));
  out += "test";
  return out;
}

class TestMemory final : public MemoryApi {
 public:
  explicit TestMemory(std::shared_ptr<Directory> directory) : directory_(directory) {}
  bool SupportsFabric() const override { return true; }
  bool Allocate(int, uint64_t, void**, ExportedMemory*, std::string*) override { return false; }
  void Free(int, void*, uint64_t) override { ADD_FAILURE() << "externally owned allocation freed"; }
  bool Import(int, const std::string& handle, uint64_t offset, void** base, void** mapped,
              std::string* error) override {
    if (fail_import || directory_->count(handle) == 0) {
      *error = "injected import failure";
      return false;
    }
    ++imports;
    *mapped = directory_->at(handle).data();
    *base = static_cast<unsigned char*>(*mapped) + offset;
    return true;
  }
  void Unimport(int, void*) override { ++unimports; }
  bool Copy(int, void* dst, int, const void* src, uint64_t bytes, std::string*) override {
    std::memcpy(dst, src, bytes);
    return true;
  }
  bool Synchronize(int, std::string*) override { return true; }
  CopyOutcome CopyBatchTracked(int, const std::vector<Fragment>& fragments) override {
    ++copies;
    if (on_copy) on_copy();
    if (!outcome.ok) return outcome;
    for (const auto& fragment : fragments)
      std::memcpy(fragment.dst, fragment.src, fragment.bytes);
    return outcome;
  }
  std::atomic<int> imports{0}, unimports{0}, copies{0};
  bool fail_import = false;
  CopyOutcome outcome{true, true, {}};
  std::function<void()> on_copy;
 private:
  std::shared_ptr<Directory> directory_;
};

class LoopPeer final : public PeerApi {
 public:
  explicit LoopPeer(PoolManager* owner) : owner(owner) {}
  PeerProbeResult Probe(const std::string& group, const std::vector<std::string>& keys,
                        const std::vector<uint64_t>& units) override {
    if (on_probe) on_probe();
    return owner->Probe(group, keys, units);
  }
  PeerReadResult AcquireRead(const PeerReadRequest& request) override {
    ++acquires;
    last = request;
    auto result = owner->AcquireRead(request);
    if (lose_acquire) result.status = Status::Error("injected lost Acquire reply");
    if (corrupt && !result.regions.empty()) result.regions[0].offset = UINT64_MAX;
    return result;
  }
  Status ReleaseRead(const std::string& op, const std::string& incarnation, bool drained) override {
    ++releases;
    if (lose_release) return Status::Error("injected lost Release request");
    return owner->ReleaseRead(op, incarnation, drained);
  }
  PoolManager* owner;
  int acquires = 0, releases = 0;
  bool lose_acquire = false, lose_release = false, corrupt = false;
  PeerReadRequest last;
  std::function<void()> on_probe;
};

class L1PeerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory = std::make_shared<Directory>();
    source_api = std::make_shared<TestMemory>(directory);
    dest_api = std::make_shared<TestMemory>(directory);
    source = std::make_unique<PoolManager>(source_api);
    dest = std::make_unique<PoolManager>(dest_api);
    peer = std::make_shared<LoopPeer>(source.get());
    dest->ConfigurePeers({peer});
    Attach(*source, "src", 1);
    Attach(*dest, "dst", 2);
  }
  void Attach(PoolManager& manager, const std::string& id, unsigned char token,
              const std::string& group = "g") {
    handles[id] = Descriptor(token, kBytes);
    (*directory)[handles[id]] = std::vector<unsigned char>(kBytes, 0xDA);
    ASSERT_TRUE(manager.Attach(id, group, token, kBytes, {handles[id], 0, "hip_fabric"}).status.ok);
    ASSERT_TRUE(manager.BindLayout(id, kLayout).ok);
    ASSERT_EQ(manager.Open(id).num_slots, kSlots);
  }
  void Fill(const std::string& id, uint32_t slot, unsigned char value) {
    auto& bytes = directory->at(handles[id]);
    for (const auto& region : kLayout.regions)
      std::memset(bytes.data() + region.offset + slot * region.unit_bytes, value, region.unit_bytes);
  }
  bool Equals(const std::string& id, uint32_t slot, unsigned char value) {
    const auto& bytes = directory->at(handles[id]);
    for (const auto& region : kLayout.regions)
      for (uint64_t i = 0; i < region.unit_bytes; ++i)
        if (bytes[region.offset + slot * region.unit_bytes + i] != value) return false;
    return true;
  }
  std::vector<uint32_t> Publish(PoolManager& manager, const std::string& id,
                                const std::vector<std::string>& keys, uint64_t stamp = 0) {
    auto slots = manager.Allocate(id, keys.size()).slots;
    EXPECT_EQ(slots.size(), keys.size());
    for (size_t i = 0; i < slots.size(); ++i) {
      Fill(id, slots[i], static_cast<unsigned char>(11 + i));
      EXPECT_TRUE(manager.Apply(id, {{OpKind::kPublish, slots[i], keys[i]}}, stamp).ok);
    }
    return slots;
  }
  FetchOptions Destinations(size_t count) {
    FetchOptions options;
    options.dst_slots = dest->Allocate("dst", count).slots;
    EXPECT_EQ(options.dst_slots.size(), count);
    return options;
  }
  PeerReadRequest Read(const std::string& operation = "op") {
    return {operation, source->Incarnation(), "g", {"a"}, {16, 4}};
  }
  std::shared_ptr<Directory> directory;
  std::map<std::string, std::string> handles;
  std::shared_ptr<TestMemory> source_api, dest_api;
  std::unique_ptr<PoolManager> source, dest;
  std::shared_ptr<LoopPeer> peer;
};

TEST_F(L1PeerTest, CopiesOnlyTheLeadingRemotePrefixIntoFinalSlots) {
  Publish(*source, "src", {"a", "b"});
  auto options = Destinations(3);
  auto result = dest->Fetch("dst", {"a", "missing", "b"}, options);
  EXPECT_TRUE(result.status.ok);
  EXPECT_EQ(result.slots, (std::vector<uint32_t>{options.dst_slots[0]}));
  EXPECT_EQ(result.copied_bytes, 20u);
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 11));
  EXPECT_TRUE(Equals("dst", options.dst_slots[1], 0xDA));
  EXPECT_TRUE(Equals("dst", options.dst_slots[2], 0xDA));
  EXPECT_EQ(peer->releases, 1);
  EXPECT_EQ(dest_api->imports, 2);  // own arena + one shared remote descriptor
  EXPECT_EQ(dest_api->unimports, 1);
  EXPECT_EQ(dest->Stats()[0].served_out_slots, 0u);
  EXPECT_EQ(dest->Probe("g", {"a"}, {16, 4}).available, 0u);  // caller must Publish
  EXPECT_TRUE(dest->Apply("dst", {{OpKind::kPublish, options.dst_slots[0], "a"}}).ok);
  EXPECT_EQ(dest->Probe("g", {"a"}, {16, 4}).available, 1u);
}

TEST_F(L1PeerTest, SourceRequiresCommitAndMatchingGroupAndLayout) {
  Publish(*source, "src", {"a"}, 7);
  EXPECT_EQ(source->Probe("g", {"a"}, {16, 4}).available, 0u);
  EXPECT_EQ(source->AcquireRead(Read()).available, 0u);
  ASSERT_TRUE(source->ReleaseRead("op", source->Incarnation(), true).ok);
  EXPECT_TRUE(source->Commit("src", 7).ok);
  EXPECT_EQ(source->Probe("g", {"a"}, {16, 4}).available, 1u);
  EXPECT_EQ(source->Probe("other", {"a"}, {16, 4}).available, 0u);
  EXPECT_FALSE(source->Probe("g", {"a"}, {8, 4}).status.ok);
}

TEST_F(L1PeerTest, RepeatedAcquireAndReleaseAreIdempotentAndPinsPreventReuse) {
  auto slots = Publish(*source, "src", {"a"});
  auto first = source->AcquireRead(Read());
  ASSERT_EQ(first.available, 1u);
  EXPECT_EQ(source->AcquireRead(Read()).regions.size(), first.regions.size());
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, slots[0], "changed"}}).ok);
  EXPECT_FALSE(source->ReleaseRead("op", source->Incarnation(), false).ok);
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, slots[0], "changed"}}).ok);
  EXPECT_TRUE(source->ReleaseRead("op", source->Incarnation(), true).ok);
  EXPECT_TRUE(source->ReleaseRead("op", source->Incarnation(), true).ok);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, slots[0], "changed"}}).ok);
  EXPECT_FALSE(source->AcquireRead(Read()).status.ok);
}

TEST_F(L1PeerTest, PeerReadPinsProtectTheUpstreamKeptEvictionList) {
  source->SetEvictionPolicy(1);
  const auto slots = Publish(*source, "src", {"a", "b", "c", "d", "e", "f", "g", "h"});
  for (uint32_t slot : slots)
    ASSERT_TRUE(source->Apply("src", {{OpKind::kRelease, slot, ""}}).ok);
  // With no spare copies, eviction moves the cached slots to `kept` first.
  ASSERT_EQ(source->Allocate("src", 1).slots, (std::vector<uint32_t>{slots[0]}));
  auto request = Read("kept");
  request.keys = {"b"};
  ASSERT_EQ(source->AcquireRead(request).available, 1u);
  const auto granted = source->Allocate("src", kSlots - 1).slots;
  EXPECT_EQ(granted, (std::vector<uint32_t>{slots[2], slots[3], slots[4], slots[5], slots[6],
                                         slots[7]}));
  EXPECT_TRUE(Equals("src", slots[1], 12));
  ASSERT_TRUE(source->ReleaseRead("kept", source->Incarnation(), true).ok);
  EXPECT_EQ(source->Allocate("src", 1).slots, (std::vector<uint32_t>{slots[1]}));
}

TEST_F(L1PeerTest, RejectsStaleOwnerAndOperationReuseAndFencesDelayedAcquire) {
  Publish(*source, "src", {"a", "b"});
  auto request = Read();
  request.expected_incarnation = "stale";
  EXPECT_FALSE(source->AcquireRead(request).status.ok);
  EXPECT_FALSE(source->ReleaseRead("op", "stale", true).ok);
  EXPECT_TRUE(source->AcquireRead(Read()).status.ok);
  request = Read();
  request.keys = {"b"};
  EXPECT_FALSE(source->AcquireRead(request).status.ok);
  EXPECT_TRUE(source->ReleaseRead("op", source->Incarnation(), true).ok);
  EXPECT_TRUE(source->ReleaseRead("late", source->Incarnation(), true).ok);
  EXPECT_FALSE(source->AcquireRead(Read("late")).status.ok);
}

TEST_F(L1PeerTest, DetachWaitsForPeerReadToDrain) {
  Publish(*source, "src", {"a"});
  ASSERT_EQ(source->AcquireRead(Read()).available, 1u);
  auto detach = std::async(std::launch::async, [&] { return source->Detach("src"); });
  EXPECT_EQ(detach.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_TRUE(source->ReleaseRead("op", source->Incarnation(), true).ok);
  EXPECT_TRUE(detach.get().ok);
  EXPECT_EQ(source_api->unimports, 1);
}

TEST_F(L1PeerTest, CachedSourceIsNotEvictedUntilItsPeerGrantIsReleased) {
  Publish(*source, "src", {"a"});
  ASSERT_EQ(source->Allocate("src", kSlots - 1).slots.size(), kSlots - 1);
  ASSERT_EQ(source->AcquireRead(Read()).available, 1u);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kRelease, 0, ""}}).ok);
  auto allocation = std::async(std::launch::async, [&] { return source->Allocate("src", 1); });
  EXPECT_EQ(allocation.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_TRUE(source->ReleaseRead("op", source->Incarnation(), true).ok);
  auto reused = allocation.get();
  EXPECT_EQ(reused.slots, (std::vector<uint32_t>{0}));
  EXPECT_EQ(reused.evicted_keys, (std::vector<std::string>{"a"}));
}

TEST_F(L1PeerTest, RepeatedSourceKeyCanFillDistinctDestinationsAndUnpinCompletely) {
  Publish(*source, "src", {"a"});
  auto options = Destinations(2);
  auto result = dest->Fetch("dst", {"a", "a"}, options);
  EXPECT_TRUE(result.status.ok);
  EXPECT_EQ(result.copied_slots, 2u);
  for (uint32_t slot : options.dst_slots) EXPECT_TRUE(Equals("dst", slot, 11));
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok);
}

TEST_F(L1PeerTest, RejectsDuplicateDestinationBeforeAcquire) {
  Publish(*source, "src", {"a", "b"});
  auto options = Destinations(1);
  options.dst_slots.push_back(options.dst_slots[0]);
  EXPECT_FALSE(dest->Fetch("dst", {"a", "b"}, options).status.ok);
  EXPECT_EQ(peer->acquires, 0);
  EXPECT_EQ(dest_api->copies, 0);
}

TEST_F(L1PeerTest, IndexedDestinationCannotBeOverwrittenByRemoteFetch) {
  Publish(*source, "src", {"a"});
  auto slots = Publish(*dest, "dst", {"old"});
  Fill("dst", slots[0], 77);
  FetchOptions options;
  options.dst_slots = slots;
  auto fetched = dest->Fetch("dst", {"a"}, options);
  EXPECT_FALSE(fetched.status.ok);
  EXPECT_EQ(fetched.copied_slots, 0u);
  EXPECT_TRUE(Equals("dst", slots[0], 77));
  EXPECT_EQ(dest->Probe("g", {"old"}, {16, 4}).available, 1u);
}

TEST_F(L1PeerTest, DestinationCannotBeReusedWhileCopyIsRunning) {
  Publish(*source, "src", {"a"});
  auto options = Destinations(1);
  std::promise<void> started, finish;
  auto gate = finish.get_future().share();
  dest_api->on_copy = [&] { started.set_value(); gate.wait(); };
  auto result = std::async(std::launch::async, [&] { return dest->Fetch("dst", {"a"}, options); });
  started.get_future().wait();
  EXPECT_FALSE(dest->Apply("dst", {{OpKind::kRelease, options.dst_slots[0], ""}}).ok);
  EXPECT_FALSE(dest->Apply("dst", {{OpKind::kPublish, options.dst_slots[0], "other"}}).ok);
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  finish.set_value();
  EXPECT_EQ(result.get().copied_slots, 1u);
  EXPECT_TRUE(dest->Apply("dst", {{OpKind::kRelease, options.dst_slots[0], ""}}).ok);
}

TEST_F(L1PeerTest, DestinationPublishedDuringRemoteProbeCannotBeOverwritten) {
  Publish(*source, "src", {"a"});
  auto options = Destinations(1);
  peer->on_probe = [&] {
    Fill("dst", options.dst_slots[0], 77);
    EXPECT_TRUE(dest->Apply("dst", {{OpKind::kPublish, options.dst_slots[0], "old"}}).ok);
  };
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 77));
  EXPECT_EQ(peer->acquires, 0);
}

TEST_F(L1PeerTest, DestinationWaitsForPreviousOwnerCommit) {
  Publish(*source, "src", {"a"});
  auto options = Destinations(1);
  EXPECT_TRUE(dest->Apply("dst", {{OpKind::kRelease, options.dst_slots[0], ""}}, 9).ok);
  EXPECT_EQ(dest->Allocate("dst", 1).slots, options.dst_slots);
  auto result = std::async(std::launch::async, [&] { return dest->Fetch("dst", {"a"}, options); });
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_EQ(dest_api->copies, 0);
  EXPECT_TRUE(dest->Commit("dst", 9).ok);
  EXPECT_EQ(result.get().copied_slots, 1u);
}

TEST_F(L1PeerTest, ImportFailureOrMalformedReplyUnpinsSourceAndReleasesWriter) {
  Publish(*source, "src", {"a"});
  auto options = Destinations(1);
  dest_api->fail_import = true;
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  dest_api->fail_import = false;
  peer->corrupt = true;
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  peer->corrupt = false;
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 1u);
  EXPECT_EQ(peer->releases, 3);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok);
}

TEST_F(L1PeerTest, LostAcquireReplyStillReleasesTheOwnerGrant) {
  Publish(*source, "src", {"a"});
  peer->lose_acquire = true;
  EXPECT_EQ(dest->Fetch("dst", {"a"}, Destinations(1)).copied_slots, 0u);
  EXPECT_EQ(peer->releases, 1);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok);
}

TEST_F(L1PeerTest, LostReleaseRetainsSourceButConfirmedDestinationIsUsable) {
  Publish(*source, "src", {"a"});
  peer->lose_release = true;
  auto options = Destinations(1);
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 1u);
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 11));
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok);
  EXPECT_TRUE(dest->Apply("dst", {{OpKind::kPublish, options.dst_slots[0], "a"}}).ok);
  EXPECT_TRUE(source->ReleaseRead(peer->last.operation_id, source->Incarnation(), true).ok);
}

TEST_F(L1PeerTest, DrainedCopyFailureAllowsSafeRetry) {
  Publish(*source, "src", {"a"});
  auto options = Destinations(1);
  dest_api->outcome = {false, true, "injected failed but drained copy"};
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 0xDA));
  dest_api->outcome = {true, true, {}};
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 1u);
}

TEST_F(L1PeerTest, UnknownCompletionRetainsMappingsPinsAndQuarantinesDestination) {
  // Isolate the intentional retention in a child: no normal teardown is safe
  // without external drain proof, and no GPU is involved in this injection.
  EXPECT_EXIT(( [&] {
    Publish(*source, "src", {"a"});
    auto options = Destinations(1);
    dest_api->outcome = {false, false, "injected unknown completion"};
    auto result = dest->Fetch("dst", {"a"}, options);
    bool ok = !result.status.ok && result.completion_unknown && peer->releases == 0 &&
              dest_api->unimports == 0 && !dest->Detach("dst").ok &&
              !dest->Allocate("dst", 1).status.ok &&
              !dest->Apply("dst", {{OpKind::kRelease, options.dst_slots[0], ""}}).ok &&
              !source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok;
    _exit(ok ? 0 : 1);
  }() ), ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace mori::umbp::l1
