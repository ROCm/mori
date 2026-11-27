// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU-only tests of the production peer state machine. The injected backend
// uses host bytes; descriptor parsing, slot ownership and peer grants are real.
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <tuple>
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
  void Unimport(int device, void* mapped) override {
    ++unimports;
    if (on_unimport) on_unimport(device, mapped);
  }
  bool Copy(int, void* dst, int, const void* src, uint64_t bytes, std::string*) override {
    std::memcpy(dst, src, bytes);
    return true;
  }
  bool Synchronize(int, std::string*) override { return true; }
  CopyOutcome CopyBatchTracked(int, const std::vector<Fragment>& fragments) override {
    ++copies;
    last_fragments = fragments;
    if (on_copy) on_copy();
    if (!outcome.ok) return outcome;
    for (const auto& fragment : fragments)
      std::memcpy(fragment.dst, fragment.src, fragment.bytes);
    return outcome;
  }
  std::atomic<int> imports{0}, unimports{0}, copies{0};
  bool fail_import = false;
  CopyOutcome outcome{true, true, {}};
  std::vector<Fragment> last_fragments;
  std::function<void()> on_copy;
  std::function<void(int, void*)> on_unimport;

 private:
  std::shared_ptr<Directory> directory_;
};

class LoopPeer final : public PeerApi {
 public:
  explicit LoopPeer(PoolManager* owner) : owner(owner) {}
  PeerProbeResult Probe(const std::string& group, const std::vector<std::string>& keys,
                        const std::vector<uint64_t>& units) override {
    ++probes;
    if (on_probe) on_probe();
    auto result = owner->Probe(group, keys, units);
    if (on_probe_result) on_probe_result(result);
    return result;
  }
  PeerReadResult AcquireRead(const PeerReadRequest& request) override {
    ++acquires;
    last = request;
    auto result = owner->AcquireRead(request);
    if (on_acquire) on_acquire(result);
    return result;
  }
  PeerReleaseResult ReleaseRead(const std::string& op, const std::string& incarnation,
                                bool drained) override {
    ++releases;
    if (throw_release.exchange(false)) throw std::runtime_error("injected Release exception");
    if (lose_release) return {Status::Error("injected lost Release request"), {}};
    if (restarted) return {Status::Error("owner incarnation changed"), "replacement"};
    return {owner->ReleaseRead(op, incarnation, drained), owner->Incarnation()};
  }
  PoolManager* owner;
  std::atomic<int> probes{0}, acquires{0}, releases{0};
  std::atomic<bool> lose_release{false};
  std::atomic<bool> restarted{false};
  std::atomic<bool> throw_release{false};
  PeerReadRequest last;
  std::function<void()> on_probe;
  std::function<void(PeerProbeResult&)> on_probe_result;
  std::function<void(PeerReadResult&)> on_acquire;
};

class L1PeerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory = std::make_shared<Directory>();
    source_api = std::make_shared<TestMemory>(directory);
    dest_api = std::make_shared<TestMemory>(directory);
    source = std::make_unique<PoolManager>(source_api, [this] { return now; });
    dest = std::make_unique<PoolManager>(dest_api);
    peer = std::make_shared<LoopPeer>(source.get());
    dest->ConfigurePeers({peer});
    Attach(*source, "src", 1);
    Attach(*dest, "dst", 2);
  }
  void Attach(PoolManager& manager, const std::string& id, unsigned char token,
              const std::string& group = "g") {
    handles[id] = Descriptor(token, arena_bytes);
    (*directory)[handles[id]] = std::vector<unsigned char>(arena_bytes, 0xDA);
    ASSERT_TRUE(
        manager.Attach(id, group, token, arena_bytes, {handles[id], 0, "hip_fabric"}).status.ok);
    ASSERT_TRUE(manager.BindLayout(id, layout).ok);
    ASSERT_EQ(manager.Open(id).num_slots, layout.num_slots);
  }
  void Fill(const std::string& id, uint32_t slot, unsigned char value) {
    auto& bytes = directory->at(handles[id]);
    for (const auto& region : layout.regions)
      std::memset(bytes.data() + region.offset + slot * region.unit_bytes, value, region.unit_bytes);
  }
  bool Equals(const std::string& id, uint32_t slot, unsigned char value) {
    const auto& bytes = directory->at(handles[id]);
    for (const auto& region : layout.regions)
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
  FetchResult Probe(const std::vector<std::string>& keys) {
    FetchOptions options;
    options.probe_only = true;
    return dest->Fetch("dst", keys, options);
  }
  PeerReadRequest Read(const std::string& operation = "op") {
    const auto epoch = source->Probe("g", {}, {16, 4}).read_epoch;
    return {std::to_string(epoch) + ":" + operation, source->Incarnation(), "g", {"a"}, {16, 4}};
  }
  std::shared_ptr<Directory> directory;
  PoolManager::Clock::time_point now{};
  Layout layout = kLayout;
  uint64_t arena_bytes = kBytes;
  std::map<std::string, std::string> handles;
  std::shared_ptr<TestMemory> source_api, dest_api;
  std::unique_ptr<PoolManager> source, dest;
  std::shared_ptr<LoopPeer> peer;
};

class L1PeerRunTest : public L1PeerTest, public ::testing::WithParamInterface<std::string> {};

TEST_P(L1PeerRunTest, CoalescesOnlyValidatedContiguousRangesInOneDescriptor) {
  const auto& mode = GetParam();
  if (mode == "Unaligned") {
    layout = {kSlots, {{1, 13}, {112, 3}}};
    ASSERT_TRUE(source->BindLayout("src", layout).ok);
    ASSERT_TRUE(dest->BindLayout("dst", layout).ok);
  }
  std::vector<std::string> keys{"a", "b", "c", "d"}, owners(4, "src");
  std::vector<uint32_t> src_slots;
  if (mode == "DifferentHandle") {
    Attach(*source, "src2", 3);
    ASSERT_EQ(source->Allocate("src2", 2).slots.size(), 2u);
    src_slots = Publish(*source, "src", {"a", "b"});
    const auto more = Publish(*source, "src2", {"c", "d"});
    src_slots.insert(src_slots.end(), more.begin(), more.end());
    owners[2] = owners[3] = "src2";
  } else {
    src_slots = Publish(*source, "src", keys);
  }
  auto options = Destinations(kSlots);
  options.dst_slots.resize(keys.size());
  if (mode == "ReversedDestination")
    std::reverse(options.dst_slots.begin(), options.dst_slots.end());
  if (mode == "GappedDestination") options.dst_slots = {0, 2, 4, 6};
  if (mode == "ReversedSource") {
    std::reverse(keys.begin(), keys.end());
    std::reverse(src_slots.begin(), src_slots.end());
  }
  if (mode == "RepeatedSource") {
    std::fill(keys.begin(), keys.end(), keys.front());
    std::fill(src_slots.begin(), src_slots.end(), src_slots.front());
  }
  auto expected = *directory;
  uint64_t bytes = 0;
  for (size_t k = 0; k < keys.size(); ++k) {
    for (const auto& region : layout.regions) {
      std::memcpy(expected[handles["dst"]].data() + region.offset +
                      options.dst_slots[k] * region.unit_bytes,
                  directory->at(handles[owners[k]]).data() + region.offset +
                      src_slots[k] * region.unit_bytes,
                  region.unit_bytes);
      bytes += region.unit_bytes;
    }
  }
  const auto result = dest->Fetch("dst", keys, options);
  EXPECT_TRUE(result.status.ok);
  EXPECT_EQ(result.copied_slots, keys.size());
  EXPECT_EQ(result.copied_bytes, bytes);
  size_t runs = keys.size();
  if (mode == "Contiguous" || mode == "Unaligned") runs = 1;
  if (mode == "DifferentHandle") runs = 2;
  EXPECT_EQ(dest_api->last_fragments.size(), runs * layout.regions.size());
  EXPECT_EQ(*directory, expected);  // Full destination, guards and source bytes.
  EXPECT_EQ(peer->releases, 1);
  EXPECT_EQ(dest_api->imports.load() - dest_api->unimports.load(), 1);  // Own arena remains mapped.
  for (size_t k = 0; k < keys.size(); ++k) {
    EXPECT_TRUE(source->Apply(owners[k], {{OpKind::kPublish, src_slots[k], keys[k] + "-changed"}}).ok);
    EXPECT_TRUE(dest->Apply("dst", {{OpKind::kRelease, options.dst_slots[k], ""}}).ok);
  }
}
INSTANTIATE_TEST_SUITE_P(Layouts, L1PeerRunTest,
                        ::testing::Values("Contiguous", "Unaligned", "ReversedDestination",
                                          "GappedDestination", "ReversedSource", "DifferentHandle",
                                          "RepeatedSource"),
                        [](const auto& info) { return info.param; });

TEST_F(L1PeerTest, SourceRequiresCommitAndMatchingGroupAndLayout) {
  Publish(*source, "src", {"a"}, 7);
  EXPECT_EQ(source->Probe("g", {"a"}, {16, 4}).available, 0u);
  EXPECT_EQ(source->AcquireRead(Read()).available, 0u);
  ASSERT_TRUE(source->ReleaseRead(Read("op").operation_id, source->Incarnation(), true).ok);
  EXPECT_TRUE(source->Commit("src", 7).ok);
  EXPECT_EQ(source->Probe("g", {"a"}, {16, 4}).available, 1u);
  EXPECT_EQ(source->Probe("other", {"a"}, {16, 4}).available, 0u);
  EXPECT_FALSE(source->Probe("g", {"a"}, {8, 4}).status.ok);
}

TEST_F(L1PeerTest, ReadGrantsAreIdempotentAndRejectStaleOwnersOrOperationReuse) {
  auto slots = Publish(*source, "src", {"a"});
  const auto request = Read();
  auto invalid = request;
  invalid.expected_incarnation = "stale";
  EXPECT_FALSE(source->AcquireRead(invalid).status.ok);
  EXPECT_FALSE(source->ReleaseRead(request.operation_id, "stale", true).ok);
  auto first = source->AcquireRead(request);
  ASSERT_EQ(first.available, 1u);
  auto replay = source->AcquireRead(request);
  EXPECT_EQ(replay.regions.size(), first.regions.size());
  EXPECT_EQ(replay.handles, first.handles);
  invalid = request;
  invalid.keys = {"b"};
  EXPECT_FALSE(source->AcquireRead(invalid).status.ok);
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, slots[0], "changed"}}).ok);
  EXPECT_FALSE(source->ReleaseRead(request.operation_id, source->Incarnation(), false).ok);
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, slots[0], "changed"}}).ok);
  EXPECT_TRUE(source->ReleaseRead(request.operation_id, source->Incarnation(), true).ok);
  EXPECT_TRUE(source->ReleaseRead(request.operation_id, source->Incarnation(), true).ok);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, slots[0], "changed"}}).ok);
  EXPECT_FALSE(source->AcquireRead(request).status.ok);
  ASSERT_TRUE(source->Detach("src").ok);
  // Both snapshots own their descriptors after the grant and source pool are gone.
  EXPECT_EQ(first.handles, (std::vector<std::string>{handles["src"]}));
  EXPECT_EQ(replay.handles, first.handles);
  for (const auto& region : replay.regions) EXPECT_EQ(region.handle_index, 0u);
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
  ASSERT_TRUE(source->ReleaseRead(Read("kept").operation_id, source->Incarnation(), true).ok);
  EXPECT_EQ(source->Allocate("src", 1).slots, (std::vector<uint32_t>{slots[1]}));
}

TEST_F(L1PeerTest, DetachWaitsForPeerReadToDrain) {
  Publish(*source, "src", {"a"});
  ASSERT_EQ(source->AcquireRead(Read()).available, 1u);
  auto detach = std::async(std::launch::async, [&] { return source->Detach("src"); });
  EXPECT_EQ(detach.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_TRUE(source->ReleaseRead(Read("op").operation_id, source->Incarnation(), true).ok);
  EXPECT_TRUE(detach.get().ok);
  EXPECT_EQ(source_api->unimports, 1);
}

TEST_F(L1PeerTest, RetiringMappingDoesNotBlockOtherPoolsOrAcknowledgeItsOwnDetach) {
  Attach(*source, "other", 3);
  Publish(*source, "other", {"a"});
  std::promise<void> unmapping, resume;
  auto resumed = resume.get_future().share();
  std::atomic<bool> first{true};
  source_api->on_unimport = [&](int device, void*) {
    if (device == 1 && first.exchange(false)) {
      unmapping.set_value();
      resumed.wait();
    }
  };
  auto detach = std::async(std::launch::async, [&] { return source->Detach("src"); });
  unmapping.get_future().wait();
  const PeerReadRequest read{"1:while-retiring", source->Incarnation(), "g", {"a"}, {16, 4}};
  auto controls = std::async(std::launch::async, [&] {
    const bool readable = source->Probe("g", {"a"}, {16, 4}).available == 1 &&
                          source->AcquireRead(read).available == 1;
    const bool rejected =
        !source->Attach("src", "g", 1, kBytes, {handles["src"], 0, "hip_fabric"}).status.ok &&
        !source->Detach("src").ok;
    return readable && rejected;
  });
  const bool responsive =
      controls.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
  resume.set_value();
  EXPECT_TRUE(detach.get().ok);
  EXPECT_TRUE(controls.get());
  EXPECT_TRUE(responsive);
  source_api->on_unimport = {};
  EXPECT_TRUE(source->ReleaseRead(read.operation_id, read.expected_incarnation, true).ok);
  EXPECT_TRUE(source->Detach("src").ok);
}

TEST_F(L1PeerTest, UnimportExceptionCannotAcknowledgeDetachOrPermitReuse) {
  EXPECT_EXIT(
      ([&] {
        Attach(*source, "idle", 3);  // Its destructor must also contain an Unimport exception.
        source_api->on_unimport = [](int, void*) { throw std::runtime_error(""); };
        const bool rejected =
            !source->Detach("src").ok && !source->Open("src").status.ok &&
            !source->Detach("src").ok &&
            !source->Attach("src", "g", 1, kBytes, {handles["src"], 0, "hip_fabric"}).status.ok;
        source.reset();
        _exit(rejected && source_api->unimports == 2 ? 0 : 1);
      }()),
      ::testing::ExitedWithCode(0), "");
}

TEST_F(L1PeerTest, AdmissionEpochReclaimsReceiptsAndRefreshesHintsButNeverActivePins) {
  Publish(*source, "src", {"a"});
  const auto active = Read("active"), late = Read("late"), delayed = Read("delayed");
  ASSERT_EQ(source->AcquireRead(active).available, 1u);
  ASSERT_TRUE(source->ReleaseRead(late.operation_id, source->Incarnation(), true).ok);
  EXPECT_FALSE(source->AcquireRead(late).status.ok);
  for (size_t i = 2; i < 65536; ++i)
    ASSERT_TRUE(source->ReleaseRead("1:" + std::to_string(i), source->Incarnation(), true).ok);
  EXPECT_FALSE(source->AcquireRead(delayed).status.ok);
  ASSERT_EQ(Probe({"a"}).available, 1u);

  now += std::chrono::seconds(61);
  EXPECT_FALSE(source->AcquireRead(active).status.ok);
  EXPECT_FALSE(source->AcquireRead(late).status.ok);
  EXPECT_FALSE(source->AcquireRead(delayed).status.ok);
  EXPECT_TRUE(source->ReleaseRead(late.operation_id, source->Incarnation(), true).ok);
  EXPECT_EQ(dest->Fetch("dst", {"a"}, Destinations(1)).copied_slots, 1u);
  EXPECT_EQ(peer->probes, 2);
  EXPECT_EQ(peer->acquires, 2);
  EXPECT_EQ(peer->last.operation_id.substr(0, 2), "2:");
  // New reads work after reclamation, but the old active grant still owns its pin.
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok);
  EXPECT_TRUE(source->ReleaseRead(active.operation_id, source->Incarnation(), true).ok);
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

TEST_F(L1PeerTest, IndexedDestinationCannotBeOverwrittenLocallyOrDuringRemoteProbe) {
  Publish(*source, "src", {"a"});
  Attach(*source, "local", 3);
  FetchOptions local;
  local.dst_slots = Publish(*source, "local", {"old"});
  EXPECT_FALSE(source->Fetch("local", {"a"}, local).status.ok);
  EXPECT_EQ(source_api->copies, 0);  // FABRIC guard also applies without outgoing peers.
  auto options = Destinations(1);
  peer->on_probe = [&] {
    Fill("dst", options.dst_slots[0], 77);
    EXPECT_TRUE(dest->Apply("dst", {{OpKind::kPublish, options.dst_slots[0], "old"}}).ok);
  };
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 77));
  EXPECT_EQ(peer->acquires, 0);
}

TEST_F(L1PeerTest, StopsAfterFirstValidFullPeerButContinuesShortOrFailedProbes) {
  auto second = std::make_shared<LoopPeer>(source.get());
  auto tail = std::make_shared<LoopPeer>(source.get());
  ASSERT_TRUE(dest->Detach("dst").ok);
  dest->ConfigurePeers({peer, second, tail});
  Attach(*dest, "dst", 2);
  Publish(*source, "src", {"a", "b"});
  const auto options = Destinations(2);
  for (int mode : {0, 1, 2}) {  // Full hit, partial prefix, failed reply claiming a full hit.
    SCOPED_TRACE(mode);
    peer->on_probe_result = [mode](PeerProbeResult& result) {
      if (mode == 1) result.available = 1;
      if (mode == 2) result.status = Status::Error("injected Probe failure");
    };
    for (bool with_hint : {false, true}) {
      SCOPED_TRACE(with_hint);
      const int first_probes = peer->probes, second_probes = second->probes;
      const int first_acquires = peer->acquires, second_acquires = second->acquires;
      const int first_releases = peer->releases, second_releases = second->releases;
      if (with_hint) {
        ASSERT_EQ(Probe({"a", "b"}).available, 2u);
      }
      const auto fetched = dest->Fetch("dst", {"a", "b"}, options);
      EXPECT_TRUE(fetched.status.ok);
      EXPECT_FALSE(fetched.completion_unknown);
      EXPECT_EQ(fetched.copied_slots, 2u);
      EXPECT_EQ(peer->probes - first_probes, 1);
      EXPECT_EQ(second->probes - second_probes, mode == 0 ? 0 : 1);
      EXPECT_EQ(tail->probes, 0);
      EXPECT_EQ(peer->acquires - first_acquires, mode == 0 ? 1 : 0);
      EXPECT_EQ(second->acquires - second_acquires, mode == 0 ? 0 : 1);
      EXPECT_EQ(peer->releases - first_releases, mode == 0 ? 1 : 0);
      EXPECT_EQ(second->releases - second_releases, mode == 0 ? 0 : 1);
      EXPECT_TRUE(Equals("dst", options.dst_slots[0], 11));
      EXPECT_TRUE(Equals("dst", options.dst_slots[1], 12));
    }
  }
  EXPECT_TRUE(source->Detach("src").ok);
  EXPECT_TRUE(dest->Detach("dst").ok);
}

TEST_F(L1PeerTest, ProbeHintCoversOnlyItsProvedPrefix) {
  Publish(*source, "src", {"a", "b"});
  const auto options = Destinations(2);
  EXPECT_EQ(Probe({"a", "b", "missing"}).available, 2u);
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 1u);
  EXPECT_EQ(peer->probes, 1);  // A caller may shorten the proved prefix.
  EXPECT_EQ(Probe({"a", "missing", "b"}).available, 1u);
  EXPECT_EQ(dest->Fetch("dst", {"a", "b"}, options).copied_slots, 2u);
  EXPECT_EQ(peer->probes, 3);  // Neither truncate nor cache past the first hole.
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 11));
  EXPECT_TRUE(Equals("dst", options.dst_slots[1], 12));
}

TEST_F(L1PeerTest, ProbeHintRediscoversRestartedOwner) {
  Publish(*source, "src", {"a"});
  ASSERT_EQ(Probe({"a"}).available, 1u);
  ASSERT_TRUE(source->Detach("src").ok);
  source = std::make_unique<PoolManager>(source_api);
  peer->owner = source.get();
  Attach(*source, "src", 1);
  Publish(*source, "src", {"a"});
  const auto options = Destinations(1);
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 1u);
  EXPECT_EQ(peer->probes, 2);
  EXPECT_EQ(peer->acquires, 2);
  EXPECT_EQ(peer->releases, 1);  // The rejected old incarnation created no grant.
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], 11));
  EXPECT_TRUE(source->Detach("src").ok);
}

class L1PeerHintTest : public L1PeerTest,
                       public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(L1PeerHintTest, StaleOwnerRetriesOnlyAfterConfirmedRelease) {
  const auto [rebind, lose_release] = GetParam();
  PoolManager other(source_api);
  auto other_peer = std::make_shared<LoopPeer>(&other);
  ASSERT_TRUE(dest->Detach("dst").ok);
  dest->ConfigurePeers({peer, other_peer});
  Attach(*dest, "dst", 2);
  Attach(other, "other", 3);
  Publish(*source, "src", {"a", "b"});
  Publish(other, "other", {"a", "b"});
  ASSERT_EQ(Probe({"a", "b"}).available, 2u);
  if (rebind) {
    const auto incarnation = source->Incarnation();
    ASSERT_TRUE(source->BindLayout("src", Layout{8, {{0, 8}, {64, 12}}}).ok);
    EXPECT_EQ(source->Incarnation(), incarnation);
  } else {
    ASSERT_TRUE(source->Apply("src", {{OpKind::kPublish, 1, "gone"}}).ok);
  }
  peer->lose_release = lose_release;
  const auto options = Destinations(2);
  EXPECT_EQ(dest->Fetch("dst", {"a", "b"}, options).copied_slots, lose_release ? 0u : 2u);
  EXPECT_EQ(peer->probes, lose_release ? 1 : 2);
  EXPECT_EQ(peer->acquires, 1);
  EXPECT_GE(peer->releases, 1);
  EXPECT_EQ(other_peer->acquires, lose_release ? 0 : 1);
  EXPECT_EQ(dest_api->copies, lose_release ? 0 : 1);
  EXPECT_TRUE(Equals("dst", options.dst_slots[0], lose_release ? 0xDA : 11));
  EXPECT_TRUE(Equals("dst", options.dst_slots[1], lose_release ? 0xDA : 12));
  if (!rebind)
    EXPECT_EQ(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok, !lose_release);
  EXPECT_TRUE(source->ReleaseRead(peer->last.operation_id, source->Incarnation(), true).ok);
  peer->lose_release = false;
  EXPECT_TRUE(source->Detach("src").ok);
  EXPECT_TRUE(other.Detach("other").ok);
  EXPECT_TRUE(dest->Detach("dst").ok);
}
INSTANTIATE_TEST_SUITE_P(Release, L1PeerHintTest,
                        ::testing::Combine(::testing::Bool(), ::testing::Bool()));

class L1PeerFetchTest : public L1PeerTest {
 protected:
  void SetUp() override {
    layout = {300, {{0, 16}, {300 * 16 + 16, 4}}};
    arena_bytes = 300 * 20 + 32;
    L1PeerTest::SetUp();
    for (uint32_t i = 0; i < layout.num_slots; ++i) keys.push_back(std::to_string(i));
    Publish(*source, "src", keys);
    options = Destinations(keys.size());
  }
  void ExpectPrefix(const FetchResult& result, size_t count) {
    EXPECT_TRUE(result.status.ok);
    EXPECT_FALSE(result.completion_unknown);
    EXPECT_EQ(result.slots, (std::vector<uint32_t>(options.dst_slots.begin(),
                                                   options.dst_slots.begin() + count)));
    EXPECT_EQ(result.copied_slots, count);
    EXPECT_EQ(result.copied_bytes, count * 20);
    std::vector<unsigned char> expected(arena_bytes, 0xDA);
    for (const auto& region : layout.regions)
      std::memcpy(expected.data() + region.offset,
                  directory->at(handles["src"]).data() + region.offset, count * region.unit_bytes);
    EXPECT_EQ(directory->at(handles["dst"]),
              expected);  // Full payload, untouched suffix and guards.
  }
  std::vector<std::string> keys;
  FetchOptions options;
};

class L1PeerPrefixTest : public L1PeerFetchTest,
                         public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {};

TEST_P(L1PeerPrefixTest, FetchAndProbeContinueUntilTheFirstMissingKey) {
  const auto [count, local] = GetParam();
  if (local) {
    Attach(*dest, "local", 3);
    Publish(*dest, "local", {keys.begin(), keys.begin() + local});
  }
  if (count < keys.size())
    ASSERT_TRUE(source->Apply("src", {{OpKind::kPublish, uint32_t(count), "hole"}}).ok);
  EXPECT_EQ(Probe(keys).available, count);
  EXPECT_EQ(peer->acquires, 0);
  peer->on_acquire = [&](PeerReadResult&) {
    EXPECT_LE(peer->last.keys.size(), 128u);
    EXPECT_EQ(peer->releases, peer->acquires - 1);
  };
  ExpectPrefix(dest->Fetch("dst", keys, options), count);
  EXPECT_EQ(peer->acquires, (count - local + 127) / 128);
  EXPECT_EQ(peer->probes, 2 * peer->acquires - 1);
  EXPECT_EQ(peer->releases, peer->acquires);
  EXPECT_EQ(dest_api->imports.load() - dest_api->unimports.load(), local ? 2 : 1);
  EXPECT_EQ(dest->Probe("g", {keys[local]}, {16, 4}).available, 0u);  // Caller still owns Publish.
  EXPECT_TRUE(source->Detach("src").ok);
  EXPECT_TRUE(dest->Detach("dst").ok);
}
INSTANTIATE_TEST_SUITE_P(Prefixes, L1PeerPrefixTest,
                        ::testing::Values(std::make_tuple(300u, 0u), std::make_tuple(200u, 0u),
                                          std::make_tuple(300u, 17u)));

class L1PeerFailureTest : public L1PeerFetchTest,
                          public ::testing::WithParamInterface<std::string> {};

TEST_P(L1PeerFailureTest, LaterWindowFailureReleasesPinsAndWritersForRetry) {
  const auto& fault = GetParam();
  peer->on_acquire = [&](PeerReadResult& read) {
    if (peer->acquires != 2) return;
    if (fault == "AcquireReply") read.status = Status::Error("injected lost Acquire reply");
    if (fault == "Import") dest_api->fail_import = true;
    if (fault == "DrainedCopy") dest_api->outcome = {false, true, "injected drained failure"};
    if (fault == "WrongUnit") ++read.regions.back().bytes;
    if (fault == "OutOfRange") read.regions.back().offset = UINT64_MAX;
    if (fault == "HandleIndex") read.regions.back().handle_index = read.handles.size();
    if (fault == "Descriptor") read.handles.back()[0] = '!';
  };
  ExpectPrefix(dest->Fetch("dst", keys, options), 128);
  EXPECT_EQ(peer->acquires, 2);
  EXPECT_EQ(peer->releases, 2);
  EXPECT_EQ(dest_api->imports.load() - dest_api->unimports.load(), 1);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, 128, "changed"}}).ok);
  ASSERT_TRUE(source->Apply("src", {{OpKind::kPublish, 128, keys[128]}}).ok);
  peer->on_acquire = {};
  dest_api->fail_import = false;
  dest_api->outcome = {true, true, {}};
  ExpectPrefix(dest->Fetch("dst", keys, options), keys.size());
  EXPECT_TRUE(source->Detach("src").ok);
  EXPECT_TRUE(dest->Detach("dst").ok);
}
INSTANTIATE_TEST_SUITE_P(Failures, L1PeerFailureTest,
                        ::testing::Values("Import", "AcquireReply", "DrainedCopy", "WrongUnit",
                                          "OutOfRange", "HandleIndex", "Descriptor"),
                        [](const auto& info) { return info.param; });

TEST_F(L1PeerFetchTest, LostReleaseRetriesAfterReturningCompletedPrefix) {
  ASSERT_EQ(Probe(keys).available, keys.size());
  peer->lose_release = true;
  peer->throw_release = true;  // The initial attempt deterministically exercises exception handling.
  ExpectPrefix(dest->Fetch("dst", keys, options), 128);
  EXPECT_EQ(peer->acquires, 1);
  EXPECT_GE(peer->releases, 1);
  EXPECT_FALSE(source->Apply("src", {{OpKind::kPublish, 0, "changed"}}).ok);
  EXPECT_TRUE(source->Apply("src", {{OpKind::kPublish, 128, "unread"}}).ok);
  EXPECT_TRUE(dest->Apply("dst", {{OpKind::kPublish, options.dst_slots[0], keys[0]}}).ok);
  peer->lose_release = false;
  EXPECT_TRUE(source->Detach("src").ok);  // Only the background retry can drain the pin.
  EXPECT_GE(peer->releases, 2);
}

TEST_F(L1PeerTest, FullReleaseQueueDoesNotAcquireAnUntrackedRead) {
  Publish(*source, "src", {"a"});
  const auto options = Destinations(1);
  peer->lose_release = true;
  for (size_t i = 0; i < 1024; ++i) ASSERT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 1u);
  EXPECT_EQ(dest->Fetch("dst", {"a"}, options).copied_slots, 0u);
  EXPECT_EQ(peer->acquires, 1024);
  peer->lose_release = false;
  EXPECT_TRUE(source->Detach("src").ok);
}

TEST_F(L1PeerTest, ConfirmedOwnerRestartStopsReleaseRetries) {
  Publish(*source, "src", {"a"});
  peer->lose_release = true;
  ASSERT_EQ(dest->Fetch("dst", {"a"}, Destinations(1)).copied_slots, 1u);
  peer->restarted = true;
  peer->lose_release = false;
  const auto started = std::chrono::steady_clock::now();
  dest.reset();  // The acknowledged new incarnation drains the pending queue.
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(1));
  EXPECT_GE(peer->releases, 2);
  // The mock keeps the old manager alive; explicitly drain that old incarnation.
  EXPECT_TRUE(source->ReleaseRead(peer->last.operation_id, source->Incarnation(), true).ok);
}

TEST_F(L1PeerFetchTest, UnknownLaterCompletionRetainsCurrentGrantAndQuarantines) {
  // Intentional retention needs child-process teardown; no GPU is involved.
  for (bool unimport_failure : {false, true}) {
    SCOPED_TRACE(unimport_failure);
    EXPECT_EXIT(([&] {
                  dest_api->on_copy = [&] {
                    if (!unimport_failure && dest_api->copies == 2)
                      dest_api->outcome = {false, false, "injected unknown completion"};
                  };
                  dest_api->on_unimport = [&](int, void* mapped) {
                    if (unimport_failure && dest_api->copies == 2 &&
                        mapped == directory->at(handles["src"]).data())
                      throw std::runtime_error("injected remote unimport failure");
                  };
                  const auto result = dest->Fetch("dst", keys, options);
                  const bool ok =
                      !result.status.ok && result.completion_unknown && result.slots.empty() &&
                      peer->acquires == 2 && peer->releases == 1 &&
                      dest_api->unimports == (unimport_failure ? 2 : 1) &&
                      dest_api->imports == 3 && !dest->Detach("dst").ok &&
                      !dest->Allocate("dst", 1).status.ok &&
                      !dest->Apply("dst", {{OpKind::kRelease, options.dst_slots[128], ""}}).ok &&
                      source->Apply("src", {{OpKind::kPublish, 0, "released"}}).ok &&
                      !source->Apply("src", {{OpKind::kPublish, 128, "pinned"}}).ok;
                  _exit(ok ? 0 : 1);
                }()),
                ::testing::ExitedWithCode(0), "");
  }
}

}  // namespace
}  // namespace mori::umbp::l1
