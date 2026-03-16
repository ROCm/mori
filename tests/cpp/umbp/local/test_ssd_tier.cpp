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
  cfg.dram.capacity_bytes = 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = dir;
  cfg.ssd.capacity_bytes = 64 * 1024 * 1024;

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
  cfg.dram.capacity_bytes = 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = dir;
  cfg.ssd.capacity_bytes = 64 * 1024 * 1024;

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
  leader_cfg.dram.capacity_bytes = 1024 * 1024;
  leader_cfg.ssd.enabled = true;
  leader_cfg.ssd.storage_dir = dir;
  leader_cfg.ssd.capacity_bytes = 64 * 1024 * 1024;

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
  cfg.dram.capacity_bytes = 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = dir;
  cfg.ssd.capacity_bytes = 64 * 1024 * 1024;

  cfg.ssd.io.backend = UMBPIoBackend::IoUring;
  cfg.ssd.durability.mode = UMBPDurabilityMode::Strict;
  cfg.ssd.io.queue_depth = 128;

  LocalStorageManager mgr(cfg);
  std::vector<char> payload(4096, 'U');
  assert(mgr.Write("uring_key", payload.data(), payload.size(), StorageTier::LOCAL_SSD));

  std::vector<char> out(4096, 0);
  assert(mgr.ReadIntoPtr("uring_key", reinterpret_cast<uintptr_t>(out.data()), out.size()));
  assert(out == payload);
  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_ssd_batch_read() {
  std::cout << "test_ssd_batch_read... ";
  const std::string dir = "/tmp/umbp_test_ssd_batch_read";

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = dir;
  cfg.ssd.capacity_bytes = 64 * 1024 * 1024;

  LocalStorageManager mgr(cfg);

  // Write 10 keys to SSD.
  constexpr size_t kNumKeys = 10;
  constexpr size_t kValueSize = 4096;
  std::vector<std::vector<char>> payloads(kNumKeys);
  std::vector<std::string> keys(kNumKeys);
  for (size_t i = 0; i < kNumKeys; ++i) {
    keys[i] = "batch_read_key_" + std::to_string(i);
    payloads[i].assign(kValueSize, static_cast<char>('A' + i));
    assert(mgr.Write(keys[i], payloads[i].data(), payloads[i].size(), StorageTier::LOCAL_SSD));
  }

  // Batch read all 10 via ReadBatchIntoPtr.
  auto* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
  assert(ssd != nullptr);

  std::vector<std::vector<char>> buffers(kNumKeys, std::vector<char>(kValueSize, 0));
  std::vector<uintptr_t> ptrs(kNumKeys);
  std::vector<size_t> sizes(kNumKeys, kValueSize);
  for (size_t i = 0; i < kNumKeys; ++i) {
    ptrs[i] = reinterpret_cast<uintptr_t>(buffers[i].data());
  }

  auto results = ssd->ReadBatchIntoPtr(keys, ptrs, sizes);
  for (size_t i = 0; i < kNumKeys; ++i) {
    assert(results[i]);
    assert(buffers[i] == payloads[i]);
  }

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_ssd_batch_read_partial() {
  std::cout << "test_ssd_batch_read_partial... ";
  const std::string dir = "/tmp/umbp_test_ssd_batch_read_partial";

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = dir;
  cfg.ssd.capacity_bytes = 64 * 1024 * 1024;

  LocalStorageManager mgr(cfg);

  // Write only 5 keys to SSD.
  constexpr size_t kExist = 5;
  constexpr size_t kTotal = 7;
  constexpr size_t kValueSize = 2048;
  std::vector<std::string> keys(kTotal);
  std::vector<std::vector<char>> payloads(kExist);
  for (size_t i = 0; i < kTotal; ++i) {
    keys[i] = "partial_key_" + std::to_string(i);
  }
  for (size_t i = 0; i < kExist; ++i) {
    payloads[i].assign(kValueSize, static_cast<char>('X' + i));
    assert(mgr.Write(keys[i], payloads[i].data(), payloads[i].size(), StorageTier::LOCAL_SSD));
  }

  // Batch read all 7 keys — first 5 should succeed, last 2 fail.
  auto* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
  assert(ssd != nullptr);

  std::vector<std::vector<char>> buffers(kTotal, std::vector<char>(kValueSize, 0));
  std::vector<uintptr_t> ptrs(kTotal);
  std::vector<size_t> sizes(kTotal, kValueSize);
  for (size_t i = 0; i < kTotal; ++i) {
    ptrs[i] = reinterpret_cast<uintptr_t>(buffers[i].data());
  }

  auto results = ssd->ReadBatchIntoPtr(keys, ptrs, sizes);
  for (size_t i = 0; i < kExist; ++i) {
    assert(results[i]);
    assert(buffers[i] == payloads[i]);
  }
  for (size_t i = kExist; i < kTotal; ++i) {
    assert(!results[i]);
  }

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_ssd_batch_read_io_uring() {
  std::cout << "test_ssd_batch_read_io_uring... ";
  const std::string dir = "/tmp/umbp_test_ssd_batch_read_uring";

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = dir;
  cfg.ssd.capacity_bytes = 64 * 1024 * 1024;
  cfg.ssd.io.backend = UMBPIoBackend::IoUring;
  cfg.ssd.io.queue_depth = 128;

  LocalStorageManager mgr(cfg);

  constexpr size_t kNumKeys = 8;
  constexpr size_t kValueSize = 8192;
  std::vector<std::vector<char>> payloads(kNumKeys);
  std::vector<std::string> keys(kNumKeys);
  for (size_t i = 0; i < kNumKeys; ++i) {
    keys[i] = "uring_batch_" + std::to_string(i);
    payloads[i].assign(kValueSize, static_cast<char>('0' + i));
    assert(mgr.Write(keys[i], payloads[i].data(), payloads[i].size(), StorageTier::LOCAL_SSD));
  }

  auto* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
  assert(ssd != nullptr);

  std::vector<std::vector<char>> buffers(kNumKeys, std::vector<char>(kValueSize, 0));
  std::vector<uintptr_t> ptrs(kNumKeys);
  std::vector<size_t> sizes(kNumKeys, kValueSize);
  for (size_t i = 0; i < kNumKeys; ++i) {
    ptrs[i] = reinterpret_cast<uintptr_t>(buffers[i].data());
  }

  auto results = ssd->ReadBatchIntoPtr(keys, ptrs, sizes);
  for (size_t i = 0; i < kNumKeys; ++i) {
    assert(results[i]);
    assert(buffers[i] == payloads[i]);
  }

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== Segmented SSD Tier Tests ===" << std::endl;
  test_segmented_recovery();
  test_segmented_overwrite_generation();
  test_segmented_follower_refresh();
  test_segmented_io_uring_backend();
  test_ssd_batch_read();
  test_ssd_batch_read_partial();
  test_ssd_batch_read_io_uring();
  std::cout << "All Segmented SSD Tier tests passed!" << std::endl;
  return 0;
}
