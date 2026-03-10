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
#include <iostream>
#include <thread>
#include <vector>

#include "umbp/block_index/block_index.h"

void test_hash_kv_block() {
  std::cout << "test_hash_kv_block... ";

  // Basic hash generation
  std::vector<int> tokens = {1, 2, 3, 4};
  auto hash1 = BlockIndexClient::HashKVBlock(tokens);
  assert(!hash1.empty());
  assert(hash1.size() == 64);  // SHA-256 hex = 64 chars

  // Same input → same hash
  auto hash2 = BlockIndexClient::HashKVBlock(tokens);
  assert(hash1 == hash2);

  // Different tokens → different hash
  std::vector<int> tokens2 = {5, 6, 7, 8};
  auto hash3 = BlockIndexClient::HashKVBlock(tokens2);
  assert(hash1 != hash3);

  // With prior hash
  auto hash4 = BlockIndexClient::HashKVBlock(tokens2, hash1);
  assert(hash4 != hash3);  // Prior hash changes result
  assert(hash4.size() == 64);

  std::cout << "PASSED" << std::endl;
}

void test_insert_lookup() {
  std::cout << "test_insert_lookup... ";

  BlockIndexClient idx;
  assert(idx.Count() == 0);

  // Insert
  LocalLocation loc{StorageTier::CPU_DRAM, 100, 4096};
  idx.Insert("key1", loc);
  assert(idx.Count() == 1);

  // Lookup existing
  auto result = idx.Lookup("key1");
  assert(result.has_value());
  assert(result->tier == StorageTier::CPU_DRAM);
  assert(result->offset == 100);
  assert(result->size == 4096);

  // Lookup non-existing
  auto result2 = idx.Lookup("key2");
  assert(!result2.has_value());

  // MayExist
  assert(idx.MayExist("key1") == true);
  assert(idx.MayExist("key2") == false);

  std::cout << "PASSED" << std::endl;
}

void test_remove() {
  std::cout << "test_remove... ";

  BlockIndexClient idx;
  idx.Insert("key1", {StorageTier::CPU_DRAM, 0, 100});

  auto removed = idx.Remove("key1");
  assert(removed.has_value());
  assert(removed->size == 100);
  assert(idx.Count() == 0);

  // Remove non-existing
  auto removed2 = idx.Remove("key1");
  assert(!removed2.has_value());

  std::cout << "PASSED" << std::endl;
}

void test_update_tier() {
  std::cout << "test_update_tier... ";

  BlockIndexClient idx;
  idx.Insert("key1", {StorageTier::CPU_DRAM, 0, 100});

  bool ok = idx.UpdateTier("key1", StorageTier::LOCAL_SSD);
  assert(ok);
  auto loc = idx.Lookup("key1");
  assert(loc->tier == StorageTier::LOCAL_SSD);

  // Update non-existing
  bool fail = idx.UpdateTier("nonexist", StorageTier::CPU_DRAM);
  assert(!fail);

  std::cout << "PASSED" << std::endl;
}

void test_clear() {
  std::cout << "test_clear... ";

  BlockIndexClient idx;
  idx.Insert("key1", {StorageTier::CPU_DRAM, 0, 100});
  idx.Insert("key2", {StorageTier::LOCAL_SSD, 0, 200});
  assert(idx.Count() == 2);

  idx.Clear();
  assert(idx.Count() == 0);
  assert(!idx.MayExist("key1"));

  std::cout << "PASSED" << std::endl;
}

void test_concurrent_access() {
  std::cout << "test_concurrent_access... ";

  BlockIndexClient idx;
  const int num_threads = 8;
  const int ops_per_thread = 1000;

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&idx, t, ops_per_thread]() {
      for (int i = 0; i < ops_per_thread; ++i) {
        std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
        idx.Insert(key, {StorageTier::CPU_DRAM, 0, 64});
        idx.MayExist(key);
        idx.Lookup(key);
      }
    });
  }

  for (auto& th : threads) th.join();

  assert(idx.Count() == num_threads * ops_per_thread);

  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== BlockIndex Tests ===" << std::endl;
  test_hash_kv_block();
  test_insert_lookup();
  test_remove();
  test_update_tier();
  test_clear();
  test_concurrent_access();
  std::cout << "All BlockIndex tests passed!" << std::endl;
  return 0;
}
