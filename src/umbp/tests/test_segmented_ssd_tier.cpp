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
#include <iostream>
#include <vector>

#include "umbp/storage/local_storage_manager.h"

void test_segmented_recovery() {
  std::cout << "test_segmented_recovery... ";
  const std::string dir = "/tmp/umbp_test_segmented_recovery";

  UMBPConfig cfg;
  cfg.dram_capacity_bytes = 1024 * 1024;
  cfg.ssd_enabled = true;
  cfg.ssd_storage_dir = dir;
  cfg.ssd_capacity_bytes = 64 * 1024 * 1024;

  {
    LocalStorageManager mgr(cfg);
    std::vector<char> payload(4096, 'R');
    assert(mgr.Write("recover_key", payload.data(), payload.size(), StorageTier::LOCAL_SSD));
  }

  {
    LocalStorageManager mgr(cfg);
    std::vector<char> buf(4096, 0);
    assert(mgr.ReadIntoPtr("recover_key", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
    assert(buf == std::vector<char>(4096, 'R'));
    mgr.Clear();
  }
  std::cout << "PASSED" << std::endl;
}

void test_segmented_overwrite_generation() {
  std::cout << "test_segmented_overwrite_generation... ";
  const std::string dir = "/tmp/umbp_test_segmented_overwrite";

  UMBPConfig cfg;
  cfg.dram_capacity_bytes = 1024 * 1024;
  cfg.ssd_enabled = true;
  cfg.ssd_storage_dir = dir;
  cfg.ssd_capacity_bytes = 64 * 1024 * 1024;

  LocalStorageManager mgr(cfg);
  std::vector<char> a(1024, 'A');
  std::vector<char> b(1024, 'B');
  assert(mgr.Write("gen_key", a.data(), a.size(), StorageTier::LOCAL_SSD));
  assert(mgr.Write("gen_key", b.data(), b.size(), StorageTier::LOCAL_SSD));

  std::vector<char> out(1024, 0);
  assert(mgr.ReadIntoPtr("gen_key", reinterpret_cast<uintptr_t>(out.data()), out.size()));
  assert(out == b);
  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_segmented_follower_refresh() {
  std::cout << "test_segmented_follower_refresh... ";
  const std::string dir = "/tmp/umbp_test_segmented_follower";

  UMBPConfig leader_cfg;
  leader_cfg.dram_capacity_bytes = 1024 * 1024;
  leader_cfg.ssd_enabled = true;
  leader_cfg.ssd_storage_dir = dir;
  leader_cfg.ssd_capacity_bytes = 64 * 1024 * 1024;

  leader_cfg.role = UMBPRole::SharedSSDLeader;

  UMBPConfig follower_cfg = leader_cfg;
  follower_cfg.role = UMBPRole::SharedSSDFollower;
  follower_cfg.follower_mode = true;

  LocalStorageManager leader(leader_cfg);
  LocalStorageManager follower(follower_cfg);

  std::vector<char> payload(2048, 'F');
  assert(leader.Write("follower_key", payload.data(), payload.size(), StorageTier::LOCAL_SSD));

  std::vector<char> out(2048, 0);
  assert(follower.ReadIntoPtr("follower_key", reinterpret_cast<uintptr_t>(out.data()), out.size()));
  assert(out == payload);

  leader.Clear();
  follower.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_segmented_io_uring_backend() {
  std::cout << "test_segmented_io_uring_backend... ";
  const std::string dir = "/tmp/umbp_test_segmented_io_uring";

  UMBPConfig cfg;
  cfg.dram_capacity_bytes = 1024 * 1024;
  cfg.ssd_enabled = true;
  cfg.ssd_storage_dir = dir;
  cfg.ssd_capacity_bytes = 64 * 1024 * 1024;

  cfg.ssd_io_backend = UMBPIoBackend::IoUring;
  cfg.ssd_durability_mode = UMBPDurabilityMode::Strict;
  cfg.ssd_queue_depth = 128;

  LocalStorageManager mgr(cfg);
  std::vector<char> payload(4096, 'U');
  assert(mgr.Write("uring_key", payload.data(), payload.size(), StorageTier::LOCAL_SSD));

  std::vector<char> out(4096, 0);
  assert(mgr.ReadIntoPtr("uring_key", reinterpret_cast<uintptr_t>(out.data()), out.size()));
  assert(out == payload);
  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== Segmented SSD Tier Tests ===" << std::endl;
  test_segmented_recovery();
  test_segmented_overwrite_generation();
  test_segmented_follower_refresh();
  test_segmented_io_uring_backend();
  std::cout << "All Segmented SSD Tier tests passed!" << std::endl;
  return 0;
}
