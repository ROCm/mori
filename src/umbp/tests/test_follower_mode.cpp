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
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "umbp/storage/ssd_tier.h"
#include "umbp/umbp_client.h"

namespace fs = std::filesystem;

static const std::string SHARED_SSD_DIR = "/tmp/umbp_test_follower_ssd";

static void cleanup_dir(const std::string& dir) {
  if (fs::exists(dir)) {
    fs::remove_all(dir);
  }
}

static std::vector<char> read_file_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  in.seekg(0, std::ios::end);
  std::streamsize sz = in.tellg();
  in.seekg(0, std::ios::beg);
  if (sz <= 0) return {};
  std::vector<char> buf(static_cast<size_t>(sz));
  in.read(buf.data(), sz);
  return buf;
}

void test_ssd_follower_exists_fallback() {
  std::cout << "test_ssd_follower_exists_fallback... ";
  cleanup_dir(SHARED_SSD_DIR);

  // Leader writes a key
  SSDTier leader(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadWrite);
  std::vector<char> data(4096, 'X');
  assert(leader.Write("shared_k", data.data(), data.size()));
  assert(leader.Exists("shared_k"));

  // Follower can discover it via filesystem fallback
  SSDTier follower(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadOnlyShared);
  assert(follower.Exists("shared_k"));
  assert(!follower.Exists("nonexistent"));

  // Non-follower SSDTier at same dir would NOT see leader's key
  SSDTier normal(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadWrite);
  assert(!normal.Exists("shared_k"));  // not in its keys_ map

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_ssd_follower_read_fallback() {
  std::cout << "test_ssd_follower_read_fallback... ";
  cleanup_dir(SHARED_SSD_DIR);

  // Leader writes data
  SSDTier leader(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadWrite);
  std::vector<char> data(4096);
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>(i % 256);
  assert(leader.Write("read_k", data.data(), data.size()));

  // Follower reads via filesystem fallback
  SSDTier follower(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadOnlyShared);
  std::vector<char> buf(4096, 0);
  assert(follower.ReadIntoPtr("read_k", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);

  // Second read should hit fast path (key now in follower's keys_ map)
  std::vector<char> buf2(4096, 0);
  assert(follower.ReadIntoPtr("read_k", reinterpret_cast<uintptr_t>(buf2.data()), buf2.size()));
  assert(buf2 == data);

  // Read with wrong size should fail
  std::vector<char> buf3(2048, 0);
  assert(!follower.ReadIntoPtr("read_k", reinterpret_cast<uintptr_t>(buf3.data()), buf3.size()));

  // Read() (vector version) should also work via fallback
  SSDTier follower2(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadOnlyShared);
  auto read_data = follower2.Read("read_k");
  assert(read_data.size() == data.size());
  assert(read_data == data);

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_ssd_follower_evict_no_delete() {
  std::cout << "test_ssd_follower_evict_no_delete... ";
  cleanup_dir(SHARED_SSD_DIR);

  // Leader writes
  SSDTier leader(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadWrite);
  std::vector<char> data(4096, 'Y');
  assert(leader.Write("evict_k", data.data(), data.size()));

  // Follower reads (ReadIntoPtr registers key in local map via fallback)
  SSDTier follower(SHARED_SSD_DIR, 1 * 1024 * 1024, SSDAccessMode::ReadOnlyShared);
  assert(follower.Exists("evict_k"));
  std::vector<char> buf(4096, 0);
  assert(follower.ReadIntoPtr("evict_k", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));

  // Follower evicts — should NOT delete the file
  assert(follower.Evict("evict_k"));
  auto [used, cap] = follower.Capacity();
  assert(used <= cap);  // guard against size_t underflow

  // File should still exist on disk
  assert(fs::exists(SHARED_SSD_DIR + "/evict_k.bin"));

  // Follower can re-discover via filesystem fallback
  assert(follower.Exists("evict_k"));

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_follower_stale_index_after_file_deleted() {
  std::cout << "test_follower_stale_index_after_file_deleted... ";
  cleanup_dir(SHARED_SSD_DIR);

  // Leader writes to shared SSD (via copy-on-write)
  UMBPConfig leader_cfg;
  leader_cfg.dram_capacity_bytes = 1 * 1024 * 1024;
  leader_cfg.ssd_enabled = true;
  leader_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  leader_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  leader_cfg.role = UMBPRole::SharedSSDLeader;
  UMBPClient leader(leader_cfg);

  std::vector<char> data(4096, 'Q');
  assert(leader.Put("stale_k", data.data(), data.size()));
  assert(fs::exists(SHARED_SSD_DIR + "/stale_k.bin"));

  // Follower reads from SSD but does NOT auto-promote into DRAM.
  UMBPConfig follower_cfg;
  follower_cfg.dram_capacity_bytes = 512 * 1024;
  follower_cfg.ssd_enabled = true;
  follower_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  follower_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  follower_cfg.role = UMBPRole::SharedSSDFollower;
  follower_cfg.auto_promote_on_read = false;
  UMBPClient follower(follower_cfg);

  std::vector<char> buf(4096, 0);
  assert(follower.GetIntoPtr("stale_k", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(follower.Index().MayExist("stale_k"));

  // Simulate leader eviction: delete the file.
  fs::remove(SHARED_SSD_DIR + "/stale_k.bin");
  assert(!fs::exists(SHARED_SSD_DIR + "/stale_k.bin"));

  // Exists() should not get stuck on stale in-memory index/keys_.
  assert(!follower.Exists("stale_k"));

  // A failed read should also clean up stale index entry.
  std::vector<char> buf2(4096, 0);
  assert(!follower.GetIntoPtr("stale_k", reinterpret_cast<uintptr_t>(buf2.data()), buf2.size()));
  assert(!follower.Index().MayExist("stale_k"));

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_leader_copy_to_ssd() {
  std::cout << "test_leader_copy_to_ssd... ";
  cleanup_dir(SHARED_SSD_DIR);

  UMBPConfig cfg;
  cfg.dram_capacity_bytes = 1 * 1024 * 1024;
  cfg.ssd_enabled = true;
  cfg.ssd_storage_dir = SHARED_SSD_DIR;
  cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  cfg.role = UMBPRole::SharedSSDLeader;

  UMBPClient leader(cfg);

  std::vector<char> data(4096, 'Z');
  assert(leader.Put("copy_k", data.data(), data.size()));

  // Data should be in DRAM (index says CPU_DRAM)
  assert(leader.Exists("copy_k"));
  auto loc = leader.Index().Lookup("copy_k");
  assert(loc.has_value());
  assert(loc->tier == StorageTier::CPU_DRAM);

  // File should also exist on SSD
  assert(fs::exists(SHARED_SSD_DIR + "/copy_k.bin"));

  // Verify data on SSD is correct
  SSDTier ssd_check(SHARED_SSD_DIR, 4 * 1024 * 1024, SSDAccessMode::ReadOnlyShared);
  auto ssd_data = ssd_check.Read("copy_k");
  assert(ssd_data.size() == data.size());
  assert(ssd_data == data);

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_e2e_leader_follower() {
  std::cout << "test_e2e_leader_follower... ";
  cleanup_dir(SHARED_SSD_DIR);

  // Leader config
  UMBPConfig leader_cfg;
  leader_cfg.dram_capacity_bytes = 1 * 1024 * 1024;
  leader_cfg.ssd_enabled = true;
  leader_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  leader_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  leader_cfg.role = UMBPRole::SharedSSDLeader;

  UMBPClient leader(leader_cfg);

  // Follower config — separate DRAM, shared SSD dir
  UMBPConfig follower_cfg;
  follower_cfg.dram_capacity_bytes = 512 * 1024;
  follower_cfg.ssd_enabled = true;
  follower_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  follower_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  follower_cfg.role = UMBPRole::SharedSSDFollower;
  follower_cfg.auto_promote_on_read = true;

  UMBPClient follower(follower_cfg);

  // Leader writes 3 keys
  for (int i = 0; i < 3; ++i) {
    std::string key = "e2e_k" + std::to_string(i);
    std::vector<char> data(4096, 'A' + i);
    assert(leader.Put(key.c_str(), data.data(), data.size()));
  }

  // Follower should see all 3 keys
  for (int i = 0; i < 3; ++i) {
    std::string key = "e2e_k" + std::to_string(i);
    assert(follower.Exists(key));
  }

  // Follower should read all 3 keys correctly
  for (int i = 0; i < 3; ++i) {
    std::string key = "e2e_k" + std::to_string(i);
    std::vector<char> buf(4096, 0);
    assert(follower.GetIntoPtr(key, reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
    std::vector<char> expected(4096, 'A' + i);
    assert(buf == expected);
  }

  // After reading with auto_promote, follower's index should have the keys
  for (int i = 0; i < 3; ++i) {
    std::string key = "e2e_k" + std::to_string(i);
    assert(follower.Index().MayExist(key));
  }

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_follower_batch_exists() {
  std::cout << "test_follower_batch_exists... ";
  cleanup_dir(SHARED_SSD_DIR);

  // Leader
  UMBPConfig leader_cfg;
  leader_cfg.dram_capacity_bytes = 1 * 1024 * 1024;
  leader_cfg.ssd_enabled = true;
  leader_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  leader_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  leader_cfg.role = UMBPRole::SharedSSDLeader;
  UMBPClient leader(leader_cfg);

  // Follower
  UMBPConfig follower_cfg;
  follower_cfg.dram_capacity_bytes = 512 * 1024;
  follower_cfg.ssd_enabled = true;
  follower_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  follower_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  follower_cfg.role = UMBPRole::SharedSSDFollower;
  UMBPClient follower(follower_cfg);

  // Leader writes 10 keys
  std::vector<std::string> keys;
  for (int i = 0; i < 10; ++i) {
    std::string key = "batch_k" + std::to_string(i);
    keys.push_back(key);
    std::vector<char> data(1024, 'a' + (i % 26));
    assert(leader.Put(key.c_str(), data.data(), data.size()));
  }

  // Follower BatchExists should find all
  auto results = follower.BatchExists(keys);
  assert(results.size() == 10);
  for (size_t i = 0; i < results.size(); ++i) {
    assert(results[i]);
  }

  // BatchExists with a mix of existing and non-existing
  std::vector<std::string> mixed_keys = {"batch_k0", "batch_k1", "missing_key"};
  auto mixed_results = follower.BatchExists(mixed_keys);
  assert(mixed_results[0] == true);
  assert(mixed_results[1] == true);
  assert(mixed_results[2] == false);

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

void test_follower_autopromote_no_writeback() {
  std::cout << "test_follower_autopromote_no_writeback... ";
  cleanup_dir(SHARED_SSD_DIR);

  UMBPConfig leader_cfg;
  leader_cfg.dram_capacity_bytes = 1 * 1024 * 1024;
  leader_cfg.ssd_enabled = true;
  leader_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  leader_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  leader_cfg.role = UMBPRole::SharedSSDLeader;
  UMBPClient leader(leader_cfg);

  std::vector<char> d0(4096, 'M');
  std::vector<char> d1(4096, 'N');
  assert(leader.Put("auto_k0", d0.data(), d0.size()));
  assert(leader.Put("auto_k1", d1.data(), d1.size()));

  const std::string k0_path = SHARED_SSD_DIR + "/auto_k0.bin";
  assert(fs::exists(k0_path));
  const auto before_time = fs::last_write_time(k0_path);
  const auto before_bytes = read_file_bytes(k0_path);
  assert(before_bytes == d0);

  UMBPConfig follower_cfg;
  follower_cfg.dram_capacity_bytes = 4096;  // fit one block only
  follower_cfg.ssd_enabled = true;
  follower_cfg.ssd_storage_dir = SHARED_SSD_DIR;
  follower_cfg.ssd_capacity_bytes = 4 * 1024 * 1024;
  follower_cfg.role = UMBPRole::SharedSSDFollower;
  follower_cfg.auto_promote_on_read = true;
  follower_cfg.dram_high_watermark = 2.0;  // force promote path on every read
  UMBPClient follower(follower_cfg);

  std::vector<char> buf(4096, 0);
  assert(follower.GetIntoPtr("auto_k0", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  // Ensure filesystem mtime granularity won't mask accidental rewrites.
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  assert(follower.GetIntoPtr("auto_k1", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));

  const auto after_time = fs::last_write_time(k0_path);
  const auto after_bytes = read_file_bytes(k0_path);
  assert(after_bytes == d0);
  assert(after_time == before_time);

  cleanup_dir(SHARED_SSD_DIR);
  std::cout << "PASSED" << std::endl;
}

int main() {
  test_ssd_follower_exists_fallback();
  test_ssd_follower_read_fallback();
  test_ssd_follower_evict_no_delete();
  test_follower_stale_index_after_file_deleted();
  test_leader_copy_to_ssd();
  test_e2e_leader_follower();
  test_follower_batch_exists();
  test_follower_autopromote_no_writeback();

  std::cout << "\nAll follower mode tests PASSED!" << std::endl;
  return 0;
}
