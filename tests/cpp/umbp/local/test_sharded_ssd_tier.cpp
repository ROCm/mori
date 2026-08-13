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
//
// Multi-drive SSD tier: keys spread across N directories (one per drive), read
// back from whichever drive holds them, and batch IO runs on all drives at once.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "umbp/local/tiers/sharded_ssd_tier.h"
#include "umbp/local/tiers/ssd_backend_factory.h"

using namespace mori::umbp;

namespace {

// A fresh set of per-drive directories for one test case.
std::string MakeDirs(const std::string& stem, int n) {
  std::string joined;
  for (int i = 0; i < n; ++i) {
    const std::string dir = "/tmp/umbp_test_" + stem + "_d" + std::to_string(i);
    std::string cmd = "rm -rf " + dir;
    if (system(cmd.c_str()) != 0) { /* fresh tree either way */
    }
    if (!joined.empty()) joined += ',';
    joined += dir;
  }
  return joined;
}

UMBPSsdConfig MakeConfig(const std::string& dirs, size_t capacity) {
  UMBPSsdConfig cfg;
  cfg.enabled = true;
  cfg.storage_dir = dirs;
  cfg.capacity_bytes = capacity;
  cfg.segment_size_bytes = 4ULL * 1024 * 1024;
  return cfg;
}

void test_storage_dirs_parsing() {
  std::cout << "test_storage_dirs_parsing... ";
  UMBPSsdConfig cfg;

  cfg.storage_dir = "/mnt/nvme0";
  assert(cfg.StorageDirs().size() == 1);
  assert(cfg.StorageDirs()[0] == "/mnt/nvme0");

  cfg.storage_dir = "/mnt/nvme0,/mnt/nvme1,/mnt/nvme2";
  auto dirs = cfg.StorageDirs();
  assert(dirs.size() == 3);
  assert(dirs[2] == "/mnt/nvme2");

  // Whitespace around separators is tolerated.
  cfg.storage_dir = "/mnt/a , /mnt/b";
  dirs = cfg.StorageDirs();
  assert(dirs.size() == 2);
  assert(dirs[0] == "/mnt/a");
  assert(dirs[1] == "/mnt/b");

  // Empty never yields an empty list.
  cfg.storage_dir = "";
  assert(cfg.StorageDirs().size() == 1);
  std::cout << "PASSED" << std::endl;
}

// One directory must still produce a plain SSDTier (no sharding overhead, and
// no behavior change for every existing single-drive deployment).
void test_factory_single_dir_is_unsharded() {
  std::cout << "test_factory_single_dir_is_unsharded... ";
  auto cfg = MakeConfig(MakeDirs("factory_single", 1), 64ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);
  assert(backend != nullptr);
  assert(dynamic_cast<ShardedSsdTier*>(backend.get()) == nullptr);
  std::cout << "PASSED" << std::endl;
}

void test_factory_multi_dir_is_sharded() {
  std::cout << "test_factory_multi_dir_is_sharded... ";
  auto cfg = MakeConfig(MakeDirs("factory_multi", 3), 96ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);
  auto* sharded = dynamic_cast<ShardedSsdTier*>(backend.get());
  assert(sharded != nullptr);
  assert(sharded->ShardCount() == 3);
  // capacity_bytes is the TOTAL budget, split across drives.
  assert(backend->Capacity().second == 96ULL * 1024 * 1024);
  std::cout << "PASSED" << std::endl;
}

// The headline case: 1000 same-size keys over 2 drives land ~500/500 and every
// one reads back correctly from whichever drive got it.
void test_balanced_placement_and_readback() {
  std::cout << "test_balanced_placement_and_readback... ";
  auto cfg = MakeConfig(MakeDirs("balance", 2), 128ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);
  auto* sharded = dynamic_cast<ShardedSsdTier*>(backend.get());
  assert(sharded != nullptr);

  constexpr int kKeys = 1000;
  constexpr size_t kSize = 4096;
  int per_shard[2] = {0, 0};
  for (int i = 0; i < kKeys; ++i) {
    std::vector<char> payload(kSize, static_cast<char>('A' + (i % 26)));
    const std::string key = "k" + std::to_string(i);
    assert(backend->Write(key, payload.data(), payload.size()));
    const int s = sharded->ShardOf(key);
    assert(s == 0 || s == 1);
    ++per_shard[s];
  }
  // Exact round-robin for uniform sizes on uniform drives.
  assert(per_shard[0] == kKeys / 2);
  assert(per_shard[1] == kKeys / 2);

  for (int i = 0; i < kKeys; ++i) {
    std::vector<char> out(kSize, 0);
    const std::string key = "k" + std::to_string(i);
    assert(backend->ReadIntoPtr(key, reinterpret_cast<uintptr_t>(out.data()), out.size()));
    assert(out == std::vector<char>(kSize, static_cast<char>('A' + (i % 26))));
  }
  std::cout << "PASSED" << std::endl;
}

// A re-put of a live key must go back to its own drive, never get duplicated
// onto a second one.
void test_rewrite_stays_on_same_shard() {
  std::cout << "test_rewrite_stays_on_same_shard... ";
  auto cfg = MakeConfig(MakeDirs("rewrite", 3), 96ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);
  auto* sharded = dynamic_cast<ShardedSsdTier*>(backend.get());

  std::vector<char> a(2048, 'A');
  std::vector<char> b(2048, 'B');
  assert(backend->Write("sticky", a.data(), a.size()));
  const int first_shard = sharded->ShardOf("sticky");
  assert(first_shard >= 0);
  assert(backend->Write("sticky", b.data(), b.size()));
  assert(sharded->ShardOf("sticky") == first_shard);

  std::vector<char> out(2048, 0);
  assert(backend->ReadIntoPtr("sticky", reinterpret_cast<uintptr_t>(out.data()), out.size()));
  assert(out == b);
  std::cout << "PASSED" << std::endl;
}

void test_batch_write_read_across_shards() {
  std::cout << "test_batch_write_read_across_shards... ";
  auto cfg = MakeConfig(MakeDirs("batch", 4), 128ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);
  auto* sharded = dynamic_cast<ShardedSsdTier*>(backend.get());

  constexpr int kKeys = 64;
  constexpr size_t kSize = 8192;
  std::vector<std::string> keys;
  std::vector<std::vector<char>> payloads;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (int i = 0; i < kKeys; ++i) {
    keys.push_back("bk" + std::to_string(i));
    payloads.emplace_back(kSize, static_cast<char>(i));
    sizes.push_back(kSize);
  }
  for (auto& p : payloads) srcs.push_back(p.data());

  auto wrote = backend->BatchWrite(keys, srcs, sizes);
  assert(wrote.size() == static_cast<size_t>(kKeys));
  for (bool ok : wrote) assert(ok);

  // Every drive should be carrying part of the batch — that is what makes the
  // reads parallel.
  for (const auto& cap : sharded->PerShardCapacity()) assert(cap.first > 0);

  std::vector<std::vector<char>> outs(kKeys, std::vector<char>(kSize, 0));
  std::vector<uintptr_t> dsts;
  for (auto& o : outs) dsts.push_back(reinterpret_cast<uintptr_t>(o.data()));
  auto read = backend->BatchReadIntoPtr(keys, dsts, sizes);
  assert(read.size() == static_cast<size_t>(kKeys));
  for (int i = 0; i < kKeys; ++i) {
    assert(read[i]);
    assert(outs[i] == std::vector<char>(kSize, static_cast<char>(i)));
  }
  std::cout << "PASSED" << std::endl;
}

// A key that was never written must miss, not read someone else's bytes, and a
// batch containing unknown keys must report exactly those as failures.
void test_missing_keys() {
  std::cout << "test_missing_keys... ";
  auto cfg = MakeConfig(MakeDirs("missing", 2), 64ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);

  std::vector<char> payload(1024, 'X');
  assert(backend->Write("present", payload.data(), payload.size()));
  assert(backend->Exists("present"));
  assert(!backend->Exists("absent"));

  std::vector<char> out(1024, 0);
  assert(!backend->ReadIntoPtr("absent", reinterpret_cast<uintptr_t>(out.data()), out.size()));

  std::vector<std::string> keys = {"present", "absent"};
  std::vector<std::vector<char>> outs(2, std::vector<char>(1024, 0));
  std::vector<uintptr_t> dsts = {reinterpret_cast<uintptr_t>(outs[0].data()),
                                 reinterpret_cast<uintptr_t>(outs[1].data())};
  std::vector<size_t> sizes = {1024, 1024};
  auto read = backend->BatchReadIntoPtr(keys, dsts, sizes);
  assert(read[0]);
  assert(!read[1]);
  std::cout << "PASSED" << std::endl;
}

void test_evict_and_clear() {
  std::cout << "test_evict_and_clear... ";
  auto cfg = MakeConfig(MakeDirs("evict", 2), 64ULL * 1024 * 1024);
  auto backend = MakeFileSsdBackend(cfg);

  std::vector<char> payload(4096, 'E');
  for (int i = 0; i < 8; ++i) {
    assert(backend->Write("ek" + std::to_string(i), payload.data(), payload.size()));
  }
  const size_t used_before = backend->Capacity().first;
  assert(used_before > 0);

  assert(backend->Evict("ek3"));
  assert(!backend->Exists("ek3"));
  assert(!backend->Evict("ek3"));  // idempotent-ish: already gone
  assert(backend->Exists("ek4"));

  backend->Clear();
  assert(backend->Capacity().first == 0);
  for (int i = 0; i < 8; ++i) assert(!backend->Exists("ek" + std::to_string(i)));
  std::cout << "PASSED" << std::endl;
}

// Uneven drive sizes: placement is capacity-aware, so the bigger drive takes
// more keys rather than the set failing once the small one fills.
void test_uneven_shards_place_by_free_space() {
  std::cout << "test_uneven_shards_place_by_free_space... ";
  const std::string dirs = MakeDirs("uneven", 2);
  auto cfg = MakeConfig(dirs, 0);  // capacity set per shard below

  std::vector<std::string> parts = cfg.StorageDirs();
  std::vector<std::unique_ptr<TierBackend>> shards;
  UMBPSsdConfig shard_cfg = cfg;
  shard_cfg.capacity_bytes = 8ULL * 1024 * 1024;
  shards.push_back(std::make_unique<SSDTier>(parts[0], 8ULL * 1024 * 1024, shard_cfg));
  shard_cfg.capacity_bytes = 32ULL * 1024 * 1024;
  shards.push_back(std::make_unique<SSDTier>(parts[1], 32ULL * 1024 * 1024, shard_cfg));
  ShardedSsdTier tier(std::move(shards));

  assert(tier.Capacity().second == 40ULL * 1024 * 1024);

  int per_shard[2] = {0, 0};
  std::vector<char> payload(64 * 1024, 'U');
  for (int i = 0; i < 60; ++i) {
    const std::string key = "uk" + std::to_string(i);
    assert(tier.Write(key, payload.data(), payload.size()));
    ++per_shard[tier.ShardOf(key)];
  }
  // The 32MB drive must have absorbed strictly more than the 8MB one.
  assert(per_shard[1] > per_shard[0]);
  std::cout << "PASSED" << std::endl;
}

}  // namespace

int main() {
  std::cout << "=== Sharded (multi-drive) SSD Tier Tests ===" << std::endl;
  test_storage_dirs_parsing();
  test_factory_single_dir_is_unsharded();
  test_factory_multi_dir_is_sharded();
  test_balanced_placement_and_readback();
  test_rewrite_stays_on_same_shard();
  test_batch_write_read_across_shards();
  test_missing_keys();
  test_evict_and_clear();
  test_uneven_shards_place_by_free_space();
  std::cout << "All sharded SSD tier tests passed!" << std::endl;
  return 0;
}
