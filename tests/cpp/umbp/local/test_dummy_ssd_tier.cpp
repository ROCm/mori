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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "umbp/local/tiers/dummy_ssd_tier.h"
#include "umbp/local/tiers/local_storage_manager.h"

using namespace mori::umbp;

// ============================================================
// Unit tests: DummySsdTier directly
// ============================================================

void test_write_and_exists() {
  std::cout << "test_write_and_exists... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> data(4096, 'A');
  assert(tier.Write("key1", data.data(), data.size()));
  assert(tier.Exists("key1"));
  assert(!tier.Exists("key_nonexistent"));

  std::cout << "PASSED" << std::endl;
}

void test_write_duplicate_key() {
  std::cout << "test_write_duplicate_key... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> d1(4096, 'A');
  assert(tier.Write("dup", d1.data(), d1.size()));

  std::vector<char> d2(4096, 'B');
  assert(tier.Write("dup", d2.data(), d2.size()));

  assert(tier.Exists("dup"));

  auto [used, total] = tier.Capacity();
  assert(used == 4096);

  std::cout << "PASSED" << std::endl;
}

void test_read_into_ptr_existing_key() {
  std::cout << "test_read_into_ptr_existing_key... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> data(4096, 'X');
  assert(tier.Write("rk", data.data(), data.size()));

  std::vector<char> buf(4096, 0);
  uintptr_t dst = reinterpret_cast<uintptr_t>(buf.data());
  assert(tier.ReadIntoPtr("rk", dst, buf.size()));

  // buf content is unchanged (dummy tier does not write dst)
  assert(buf == std::vector<char>(4096, 0));

  std::cout << "PASSED" << std::endl;
}

void test_read_into_ptr_missing_key() {
  std::cout << "test_read_into_ptr_missing_key... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> buf(4096, 0);
  uintptr_t dst = reinterpret_cast<uintptr_t>(buf.data());
  assert(!tier.ReadIntoPtr("missing", dst, buf.size()));

  std::cout << "PASSED" << std::endl;
}

void test_evict() {
  std::cout << "test_evict... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> data(512, 'E');
  assert(tier.Write("evict_me", data.data(), data.size()));
  assert(tier.Exists("evict_me"));

  assert(tier.Evict("evict_me"));
  assert(!tier.Exists("evict_me"));

  // Evicting a non-existent key returns false
  assert(!tier.Evict("evict_me"));

  std::cout << "PASSED" << std::endl;
}

void test_evict_frees_capacity() {
  std::cout << "test_evict_frees_capacity... ";
  DummySsdTier tier(1024);

  std::vector<char> d1(512, 'A');
  std::vector<char> d2(512, 'B');
  assert(tier.Write("k1", d1.data(), d1.size()));
  assert(tier.Write("k2", d2.data(), d2.size()));

  auto [used1, total1] = tier.Capacity();
  assert(used1 == 1024);
  assert(total1 == 1024);

  // Full — cannot write more
  std::vector<char> d3(512, 'C');
  assert(!tier.Write("k3", d3.data(), d3.size()));

  // Evict frees space
  assert(tier.Evict("k1"));
  auto [used2, total2] = tier.Capacity();
  assert(used2 == 512);

  assert(tier.Write("k3", d3.data(), d3.size()));

  std::cout << "PASSED" << std::endl;
}

void test_capacity_tracking() {
  std::cout << "test_capacity_tracking... ";
  DummySsdTier tier(10000);

  auto [u0, t0] = tier.Capacity();
  assert(u0 == 0);
  assert(t0 == 10000);

  std::vector<char> data(3000, 'D');
  assert(tier.Write("cap1", data.data(), data.size()));
  auto [u1, t1] = tier.Capacity();
  assert(u1 == 3000);
  assert(t1 == 10000);

  assert(tier.Write("cap2", data.data(), data.size()));
  auto [u2, t2] = tier.Capacity();
  assert(u2 == 6000);

  assert(tier.Evict("cap1"));
  auto [u3, t3] = tier.Capacity();
  assert(u3 == 3000);

  std::cout << "PASSED" << std::endl;
}

void test_capacity_full_rejects_write() {
  std::cout << "test_capacity_full_rejects_write... ";
  DummySsdTier tier(1000);

  std::vector<char> data(600, 'F');
  assert(tier.Write("fit", data.data(), data.size()));

  std::vector<char> big(500, 'G');
  assert(!tier.Write("no_fit", big.data(), big.size()));
  assert(!tier.Exists("no_fit"));

  std::cout << "PASSED" << std::endl;
}

void test_clear() {
  std::cout << "test_clear... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> data(1024, 'C');
  tier.Write("c1", data.data(), data.size());
  tier.Write("c2", data.data(), data.size());
  tier.Write("c3", data.data(), data.size());

  tier.Clear();

  assert(!tier.Exists("c1"));
  assert(!tier.Exists("c2"));
  assert(!tier.Exists("c3"));

  auto [used, total] = tier.Capacity();
  assert(used == 0);

  // Can write again after clear
  assert(tier.Write("c4", data.data(), data.size()));
  assert(tier.Exists("c4"));

  std::cout << "PASSED" << std::endl;
}

void test_get_lru_key() {
  std::cout << "test_get_lru_key... ";
  DummySsdTier tier(1024 * 1024);

  // Empty tier returns ""
  assert(tier.GetLRUKey() == "");

  std::vector<char> data(128, 'L');
  tier.Write("lru1", data.data(), data.size());
  tier.Write("lru2", data.data(), data.size());
  tier.Write("lru3", data.data(), data.size());

  // lru1 was written first → LRU tail
  assert(tier.GetLRUKey() == "lru1");

  // Touch lru1 via Read → lru2 becomes LRU tail
  std::vector<char> buf(128);
  tier.ReadIntoPtr("lru1", reinterpret_cast<uintptr_t>(buf.data()), buf.size());
  assert(tier.GetLRUKey() == "lru2");

  // Evict lru2 → lru3 becomes LRU tail
  tier.Evict("lru2");
  assert(tier.GetLRUKey() == "lru3");

  std::cout << "PASSED" << std::endl;
}

void test_get_lru_candidates() {
  std::cout << "test_get_lru_candidates... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> data(128, 'K');
  tier.Write("a", data.data(), data.size());
  tier.Write("b", data.data(), data.size());
  tier.Write("c", data.data(), data.size());
  tier.Write("d", data.data(), data.size());

  // Order: d(MRU) → c → b → a(LRU)
  auto cands = tier.GetLRUCandidates(3);
  assert(cands.size() == 3);
  assert(cands[0] == "a");
  assert(cands[1] == "b");
  assert(cands[2] == "c");

  // Request more than available
  auto all = tier.GetLRUCandidates(100);
  assert(all.size() == 4);

  // Empty tier
  tier.Clear();
  auto empty = tier.GetLRUCandidates(5);
  assert(empty.empty());

  std::cout << "PASSED" << std::endl;
}

void test_write_from_ptr() {
  std::cout << "test_write_from_ptr... ";
  DummySsdTier tier(1024 * 1024);

  std::vector<char> data(4096, 'W');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());
  assert(tier.WriteFromPtr("wfp", src, data.size()));
  assert(tier.Exists("wfp"));

  auto [used, _] = tier.Capacity();
  assert(used == 4096);

  std::cout << "PASSED" << std::endl;
}

void test_flush() {
  std::cout << "test_flush... ";
  DummySsdTier tier(1024 * 1024);

  // Flush on empty tier should succeed
  assert(tier.Flush());

  std::vector<char> data(512, 'F');
  tier.Write("fk", data.data(), data.size());

  // Flush after writes should succeed
  assert(tier.Flush());

  std::cout << "PASSED" << std::endl;
}

void test_tier_id() {
  std::cout << "test_tier_id... ";
  DummySsdTier tier(1024);
  assert(tier.tier_id() == StorageTier::LOCAL_SSD);

  std::cout << "PASSED" << std::endl;
}

void test_batch_write_and_batch_read() {
  std::cout << "test_batch_write_and_batch_read... ";
  DummySsdTier tier(1024 * 1024);

  constexpr size_t kN = 5;
  constexpr size_t kSize = 1024;

  std::vector<std::string> keys(kN);
  std::vector<std::vector<char>> payloads(kN);
  std::vector<const void*> data_ptrs(kN);
  std::vector<size_t> sizes(kN, kSize);

  for (size_t i = 0; i < kN; ++i) {
    keys[i] = "batch_" + std::to_string(i);
    payloads[i].assign(kSize, static_cast<char>('0' + i));
    data_ptrs[i] = payloads[i].data();
  }

  // BatchWrite (default loops over Write)
  auto write_results = tier.BatchWrite(keys, data_ptrs, sizes);
  assert(write_results.size() == kN);
  for (size_t i = 0; i < kN; ++i) {
    assert(write_results[i]);
    assert(tier.Exists(keys[i]));
  }

  // BatchReadIntoPtr
  std::vector<std::vector<char>> bufs(kN, std::vector<char>(kSize, 0));
  std::vector<uintptr_t> dst_ptrs(kN);
  for (size_t i = 0; i < kN; ++i) {
    dst_ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
  }

  auto read_results = tier.BatchReadIntoPtr(keys, dst_ptrs, sizes);
  assert(read_results.size() == kN);
  for (size_t i = 0; i < kN; ++i) {
    assert(read_results[i]);
  }

  std::cout << "PASSED" << std::endl;
}

void test_batch_read_partial_miss() {
  std::cout << "test_batch_read_partial_miss... ";
  DummySsdTier tier(1024 * 1024);

  constexpr size_t kSize = 512;

  std::vector<char> data(kSize, 'P');
  tier.Write("exist1", data.data(), data.size());
  tier.Write("exist2", data.data(), data.size());

  std::vector<std::string> keys = {"exist1", "missing", "exist2"};
  std::vector<std::vector<char>> bufs(3, std::vector<char>(kSize, 0));
  std::vector<uintptr_t> ptrs = {
      reinterpret_cast<uintptr_t>(bufs[0].data()),
      reinterpret_cast<uintptr_t>(bufs[1].data()),
      reinterpret_cast<uintptr_t>(bufs[2].data()),
  };
  std::vector<size_t> sizes = {kSize, kSize, kSize};

  auto results = tier.BatchReadIntoPtr(keys, ptrs, sizes);
  assert(results[0] == true);
  assert(results[1] == false);
  assert(results[2] == true);

  std::cout << "PASSED" << std::endl;
}

// ============================================================
// Integration tests: DummySsdTier via LocalStorageManager
// ============================================================

static UMBPConfig MakeDummyConfig(size_t dram_bytes, size_t ssd_bytes) {
  UMBPConfig cfg;
  cfg.dram.capacity_bytes = dram_bytes;
  cfg.ssd.enabled = true;
  cfg.ssd_backend = "dummy_storage";
  cfg.ssd.capacity_bytes = ssd_bytes;
  return cfg;
}

void test_lsm_dummy_write_read() {
  std::cout << "test_lsm_dummy_write_read... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 10 * 1024 * 1024);
  LocalStorageManager mgr(cfg);

  std::vector<char> data(4096, 'S');
  assert(mgr.Write("lsm_dk", data.data(), data.size(), StorageTier::LOCAL_SSD));
  assert(mgr.Exists("lsm_dk"));

  // ReadIntoPtr succeeds (dummy returns true, does not write buffer)
  std::vector<char> buf(4096, 0);
  assert(mgr.ReadIntoPtr("lsm_dk", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_capacity() {
  std::cout << "test_lsm_dummy_capacity... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 5 * 1024 * 1024);
  LocalStorageManager mgr(cfg);

  auto [used, total] = mgr.Capacity(StorageTier::LOCAL_SSD);
  assert(used == 0);
  assert(total == 5 * 1024 * 1024);

  std::vector<char> data(1024, 'C');
  mgr.Write("cap_dk", data.data(), data.size(), StorageTier::LOCAL_SSD);

  auto [used2, _] = mgr.Capacity(StorageTier::LOCAL_SSD);
  assert(used2 == 1024);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_dram_auto_demote() {
  std::cout << "test_lsm_dummy_dram_auto_demote... ";
  auto cfg = MakeDummyConfig(1024, 10 * 1024 * 1024);
  LocalBlockIndex index;
  LocalStorageManager mgr(cfg, &index);

  std::vector<char> d1(512, 'A');
  assert(mgr.Write("dk1", d1.data(), d1.size()));
  index.Insert("dk1", {StorageTier::CPU_DRAM, 0, 512});

  std::vector<char> d2(512, 'B');
  assert(mgr.Write("dk2", d2.data(), d2.size()));
  index.Insert("dk2", {StorageTier::CPU_DRAM, 0, 512});

  // DRAM full → dk1 auto-demoted to dummy SSD
  std::vector<char> d3(512, 'C');
  assert(mgr.Write("dk3", d3.data(), d3.size()));
  index.Insert("dk3", {StorageTier::CPU_DRAM, 0, 512});

  // dk1 still exists (demoted to dummy SSD)
  assert(mgr.Exists("dk1"));

  auto loc = index.Lookup("dk1");
  assert(loc.has_value());
  assert(loc->tier == StorageTier::LOCAL_SSD);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_ssd_full_eviction() {
  std::cout << "test_lsm_dummy_ssd_full_eviction... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 1024);
  LocalBlockIndex index;
  LocalStorageManager mgr(cfg, &index);

  std::vector<char> d1(512, 'A'), d2(512, 'B');
  assert(mgr.Write("se1", d1.data(), d1.size(), StorageTier::LOCAL_SSD));
  index.Insert("se1", {StorageTier::LOCAL_SSD, 0, 512});
  assert(mgr.Write("se2", d2.data(), d2.size(), StorageTier::LOCAL_SSD));
  index.Insert("se2", {StorageTier::LOCAL_SSD, 0, 512});

  // SSD full → se1 (LRU) gets evicted to make room for se3
  std::vector<char> d3(512, 'C');
  assert(mgr.Write("se3", d3.data(), d3.size(), StorageTier::LOCAL_SSD));

  assert(!mgr.Exists("se1"));
  assert(mgr.Exists("se2"));
  assert(mgr.Exists("se3"));

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_demote_and_promote() {
  std::cout << "test_lsm_dummy_demote_and_promote... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 10 * 1024 * 1024);
  cfg.eviction.auto_promote_on_read = false;
  LocalBlockIndex index;
  LocalStorageManager mgr(cfg, &index);

  std::vector<char> data(4096, 'D');
  assert(mgr.Write("dp_dk", data.data(), data.size()));
  index.Insert("dp_dk", {StorageTier::CPU_DRAM, 0, 4096});

  // Demote DRAM → dummy SSD
  assert(mgr.Demote("dp_dk"));
  assert(mgr.Exists("dp_dk"));
  auto loc1 = index.Lookup("dp_dk");
  assert(loc1.has_value());
  assert(loc1->tier == StorageTier::LOCAL_SSD);

  // Promote dummy SSD → DRAM
  assert(mgr.Promote("dp_dk"));
  auto loc2 = index.Lookup("dp_dk");
  assert(loc2.has_value());
  assert(loc2->tier == StorageTier::CPU_DRAM);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_batch_write_and_read() {
  std::cout << "test_lsm_dummy_batch_write_and_read... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 10 * 1024 * 1024);
  LocalStorageManager mgr(cfg);

  constexpr size_t kN = 8;
  constexpr size_t kSize = 2048;

  std::vector<std::string> keys(kN);
  std::vector<std::vector<char>> payloads(kN);
  std::vector<const void*> ptrs(kN);
  std::vector<size_t> sizes(kN, kSize);

  for (size_t i = 0; i < kN; ++i) {
    keys[i] = "lsm_batch_" + std::to_string(i);
    payloads[i].assign(kSize, static_cast<char>('a' + i));
    ptrs[i] = payloads[i].data();
  }

  auto wr = mgr.BatchWrite(keys, ptrs, sizes, StorageTier::LOCAL_SSD);
  for (size_t i = 0; i < kN; ++i) {
    assert(wr[i]);
  }

  std::vector<std::vector<char>> bufs(kN, std::vector<char>(kSize, 0));
  std::vector<uintptr_t> dst_ptrs(kN);
  for (size_t i = 0; i < kN; ++i) {
    dst_ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
  }

  auto rd = mgr.ReadBatchIntoPtr(keys, dst_ptrs, sizes);
  for (size_t i = 0; i < kN; ++i) {
    assert(rd[i]);
  }

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_copy_to_ssd() {
  std::cout << "test_lsm_dummy_copy_to_ssd... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 10 * 1024 * 1024);
  LocalStorageManager mgr(cfg);

  // Write to DRAM first
  std::vector<char> data(4096, 'T');
  assert(mgr.Write("copy_dk", data.data(), data.size(), StorageTier::CPU_DRAM));

  // CopyToSSD should succeed with dummy tier
  assert(mgr.CopyToSSD("copy_dk"));

  // Key should exist on SSD tier now
  auto* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
  assert(ssd != nullptr);
  assert(ssd->Exists("copy_dk"));

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_lsm_dummy_flush() {
  std::cout << "test_lsm_dummy_flush... ";
  auto cfg = MakeDummyConfig(1024 * 1024, 10 * 1024 * 1024);
  LocalStorageManager mgr(cfg);

  std::vector<char> data(1024, 'F');
  mgr.Write("fl_dk", data.data(), data.size(), StorageTier::LOCAL_SSD);

  assert(mgr.Flush());

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

// ============================================================
// Main
// ============================================================

int main() {
  std::cout << "=== DummySsdTier Unit Tests ===" << std::endl;
  test_write_and_exists();
  test_write_duplicate_key();
  test_read_into_ptr_existing_key();
  test_read_into_ptr_missing_key();
  test_evict();
  test_evict_frees_capacity();
  test_capacity_tracking();
  test_capacity_full_rejects_write();
  test_clear();
  test_get_lru_key();
  test_get_lru_candidates();
  test_write_from_ptr();
  test_flush();
  test_tier_id();
  test_batch_write_and_batch_read();
  test_batch_read_partial_miss();

  std::cout << "\n=== DummySsdTier Integration Tests (via LocalStorageManager) ===" << std::endl;
  test_lsm_dummy_write_read();
  test_lsm_dummy_capacity();
  test_lsm_dummy_dram_auto_demote();
  test_lsm_dummy_ssd_full_eviction();
  test_lsm_dummy_demote_and_promote();
  test_lsm_dummy_batch_write_and_read();
  test_lsm_dummy_copy_to_ssd();
  test_lsm_dummy_flush();

  std::cout << "\nAll DummySsdTier tests passed!" << std::endl;
  return 0;
}
