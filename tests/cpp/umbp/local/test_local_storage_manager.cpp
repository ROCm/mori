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
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "umbp/storage/local_storage_manager.h"

void test_dram_write_read() {
  std::cout << "test_dram_write_read... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;  // 1 MB
  config.ssd.enabled = false;

  LocalStorageManager mgr(config);

  std::vector<char> data(4096, 'A');
  assert(mgr.Write("key1", data.data(), data.size()));
  assert(mgr.Exists("key1"));

  std::vector<char> buf(4096, 0);
  assert(mgr.ReadIntoPtr("key1", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);

  std::cout << "PASSED" << std::endl;
}

void test_dram_eviction_no_ssd() {
  std::cout << "test_dram_eviction_no_ssd... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1024;  // tiny: 1 KB
  config.ssd.enabled = false;

  LocalStorageManager mgr(config);

  // Write 512 bytes — should fit
  std::vector<char> d1(512, 'A');
  assert(mgr.Write("key1", d1.data(), d1.size()));

  // Write another 512 bytes — should fit
  std::vector<char> d2(512, 'B');
  assert(mgr.Write("key2", d2.data(), d2.size()));

  // Write 512 more — DRAM full, no SSD → last-resort eviction of key1 (LRU)
  std::vector<char> d3(512, 'C');
  assert(mgr.Write("key3", d3.data(), d3.size()));
  // key1 was LRU, should have been evicted (data lost, no SSD)
  assert(!mgr.Exists("key1"));
  assert(mgr.Exists("key3"));

  std::cout << "PASSED" << std::endl;
}

void test_dram_full_auto_demote_to_ssd() {
  std::cout << "test_dram_full_auto_demote_to_ssd... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1024;  // tiny: 1 KB
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_ssd_autodemote";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;

  mori::umbp::LocalBlockIndex index;
  LocalStorageManager mgr(config, &index);

  // Write 512 bytes — fits in DRAM
  std::vector<char> d1(512, 'A');
  assert(mgr.Write("key1", d1.data(), d1.size()));
  index.Insert("key1", {StorageTier::CPU_DRAM, 0, 512});

  // Write another 512 bytes — fits in DRAM
  std::vector<char> d2(512, 'B');
  assert(mgr.Write("key2", d2.data(), d2.size()));
  index.Insert("key2", {StorageTier::CPU_DRAM, 0, 512});

  // Write 512 more — DRAM full, should auto-demote key1 (LRU) to SSD
  std::vector<char> d3(512, 'C');
  assert(mgr.Write("key3", d3.data(), d3.size()));
  index.Insert("key3", {StorageTier::CPU_DRAM, 0, 512});

  // key1 should still exist (demoted to SSD, not lost!)
  assert(mgr.Exists("key1"));
  assert(mgr.Exists("key3"));

  // key1's index should now say LOCAL_SSD
  auto loc = index.Lookup("key1");
  assert(loc.has_value());
  assert(loc->tier == StorageTier::LOCAL_SSD);

  // Verify key1 data is intact (read from SSD)
  std::vector<char> buf(512, 0);
  assert(mgr.ReadIntoPtr("key1", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == d1);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_ssd_write_read() {
  std::cout << "test_ssd_write_read... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_ssd";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;

  LocalStorageManager mgr(config);

  std::vector<char> data(4096, 'X');
  assert(mgr.Write("ssd_key", data.data(), data.size(), StorageTier::LOCAL_SSD));
  assert(mgr.Exists("ssd_key"));

  std::vector<char> buf(4096, 0);
  assert(mgr.ReadIntoPtr("ssd_key", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_demote_promote_with_index() {
  std::cout << "test_demote_promote_with_index... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_ssd_dp2";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;
  config.eviction.auto_promote_on_read = false;

  mori::umbp::LocalBlockIndex index;
  LocalStorageManager mgr(config, &index);

  // Write to DRAM
  std::vector<char> data(4096, 'D');
  assert(mgr.Write("dp_key", data.data(), data.size()));
  index.Insert("dp_key", {StorageTier::CPU_DRAM, 0, 4096});

  // Verify index says DRAM
  auto loc1 = index.Lookup("dp_key");
  assert(loc1->tier == StorageTier::CPU_DRAM);

  // Demote DRAM → SSD
  assert(mgr.Demote("dp_key"));
  assert(mgr.Exists("dp_key"));

  // Index should now say SSD
  auto loc2 = index.Lookup("dp_key");
  assert(loc2.has_value());
  assert(loc2->tier == StorageTier::LOCAL_SSD);

  // Verify data integrity after demote
  std::vector<char> buf(4096, 0);
  assert(mgr.ReadIntoPtr("dp_key", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);

  // Promote SSD → DRAM
  assert(mgr.Promote("dp_key"));

  // Index should now say DRAM again
  auto loc3 = index.Lookup("dp_key");
  assert(loc3.has_value());
  assert(loc3->tier == StorageTier::CPU_DRAM);

  std::vector<char> buf2(4096, 0);
  assert(mgr.ReadIntoPtr("dp_key", reinterpret_cast<uintptr_t>(buf2.data()), buf2.size()));
  assert(buf2 == data);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_capacity() {
  std::cout << "test_capacity... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_ssd_cap";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;

  LocalStorageManager mgr(config);

  auto [dram_used, dram_total] = mgr.Capacity(StorageTier::CPU_DRAM);
  assert(dram_used == 0);
  assert(dram_total == 1 * 1024 * 1024);

  auto [ssd_used, ssd_total] = mgr.Capacity(StorageTier::LOCAL_SSD);
  assert(ssd_used == 0);
  assert(ssd_total == 10 * 1024 * 1024);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_write_from_ptr() {
  std::cout << "test_write_from_ptr... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  LocalStorageManager mgr(config);

  std::vector<char> data(4096, 'P');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());
  assert(mgr.WriteFromPtr("ptr_key", src, data.size()));

  std::vector<char> buf(4096, 0);
  assert(mgr.ReadIntoPtr("ptr_key", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);

  std::cout << "PASSED" << std::endl;
}

// Regression test: SSD was previously self-evicting keys without notifying
// mori::umbp::LocalBlockIndex, leaving the index in a dirty state.  After the fix,
// SSDTier never self-evicts; LocalStorageManager::Write() drives eviction
// with full index synchronization.
void test_ssd_full_evicts_with_index_sync() {
  std::cout << "test_ssd_full_evicts_with_index_sync... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_ssd_idxsync";
  config.ssd.capacity_bytes = 1024;  // tiny: 1 KB

  mori::umbp::LocalBlockIndex index;
  LocalStorageManager mgr(config, &index);

  // Fill SSD with two 512-byte blocks (already demoted from DRAM)
  std::vector<char> d1(512, 'A'), d2(512, 'B');
  assert(mgr.Write("ssd1", d1.data(), d1.size(), StorageTier::LOCAL_SSD));
  index.Insert("ssd1", {StorageTier::LOCAL_SSD, 0, 512});
  assert(mgr.Write("ssd2", d2.data(), d2.size(), StorageTier::LOCAL_SSD));
  index.Insert("ssd2", {StorageTier::LOCAL_SSD, 0, 512});

  // Writing a third block to SSD should evict ssd1 (LRU) with index removal
  std::vector<char> d3(512, 'C');
  assert(mgr.Write("ssd3", d3.data(), d3.size(), StorageTier::LOCAL_SSD));
  index.Insert("ssd3", {StorageTier::LOCAL_SSD, 0, 512});

  // ssd1 must be gone from both storage and index
  assert(!mgr.Exists("ssd1"));
  assert(!index.Lookup("ssd1").has_value());

  // ssd3 must be readable
  std::vector<char> buf(512, 0);
  assert(mgr.ReadIntoPtr("ssd3", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == d3);

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_read_into_ptr_size_mismatch() {
  std::cout << "test_read_into_ptr_size_mismatch... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_ssd_sizemm";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;
  config.eviction.auto_promote_on_read = false;

  LocalStorageManager mgr(config);

  std::vector<char> data(4096, 'Z');
  assert(mgr.Write("mm_key", data.data(), data.size()));

  // Exact size — should succeed
  std::vector<char> buf(4096, 0);
  assert(mgr.ReadIntoPtr("mm_key", reinterpret_cast<uintptr_t>(buf.data()), 4096));

  // Smaller size — must fail (not silently truncate)
  std::vector<char> small(2048, 0);
  assert(!mgr.ReadIntoPtr("mm_key", reinterpret_cast<uintptr_t>(small.data()), 2048));

  // Larger size — must fail
  std::vector<char> large(8192, 0);
  assert(!mgr.ReadIntoPtr("mm_key", reinterpret_cast<uintptr_t>(large.data()), 8192));

  // Same test for a key on SSD
  assert(mgr.Write("ssd_mm", data.data(), data.size(), StorageTier::LOCAL_SSD));
  assert(mgr.ReadIntoPtr("ssd_mm", reinterpret_cast<uintptr_t>(buf.data()), 4096));
  assert(!mgr.ReadIntoPtr("ssd_mm", reinterpret_cast<uintptr_t>(small.data()), 2048));

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_shared_memory_mode() {
  std::cout << "test_shared_memory_mode... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.dram.use_shared_memory = true;
  config.dram.shm_name = "/umbp_test_shm_" + std::to_string(getpid());
  config.ssd.enabled = false;

  try {
    LocalStorageManager mgr(config);

    std::vector<char> data(4096, 'S');
    assert(mgr.Write("shm_key", data.data(), data.size()));
    assert(mgr.Exists("shm_key"));

    std::vector<char> buf(4096, 0);
    assert(mgr.ReadIntoPtr("shm_key", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
    assert(buf == data);

    mgr.Clear();
    std::cout << "PASSED" << std::endl;
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    if (message.find("shm_open failed") != std::string::npos) {
      std::cout << "SKIPPED (" << message << ")" << std::endl;
      return;
    }
    throw;
  }
}

int main() {
  std::cout << "=== LocalStorageManager Tests ===" << std::endl;
  test_dram_write_read();
  test_dram_eviction_no_ssd();
  test_dram_full_auto_demote_to_ssd();
  test_ssd_write_read();
  test_demote_promote_with_index();
  test_capacity();
  test_write_from_ptr();
  test_ssd_full_evicts_with_index_sync();
  test_read_into_ptr_size_mismatch();
  test_shared_memory_mode();
  std::cout << "All LocalStorageManager tests passed!" << std::endl;
  return 0;
}
