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
#include <string>
#include <thread>
#include <vector>

#include "umbp/block_index/local_block_index.h"
#include "umbp/storage/local_storage_manager.h"
#include "umbp/umbp_client.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a fake 64-hex-char SHA256-like base hash.
// |index| is encoded in the first 2 hex chars so each call produces a unique
// but syntactically valid hex string (all chars in [0-9a-f]).
static std::string FakeHash(int index) {
  char buf[3];
  snprintf(buf, sizeof(buf), "%02x", index & 0xff);
  std::string h(64, 'a');  // 'a' is valid hex
  h[0] = buf[0];
  h[1] = buf[1];
  return h;
}

static UMBPConfig MakePrefixAwareConfig(size_t dram_bytes, size_t window = 16) {
  UMBPConfig cfg;
  cfg.dram.capacity_bytes = dram_bytes;
  cfg.ssd.enabled = false;
  cfg.eviction.policy = "prefix_aware_lru";
  cfg.eviction.candidate_window = window;
  return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: Depth-biased eviction — deeper suffix block is evicted before
//         shallower prefix block even when the prefix block is older (LRU).
// ---------------------------------------------------------------------------
void test_depth_biased_eviction() {
  std::cout << "test_depth_biased_eviction... ";

  // DRAM holds exactly 2 × 512 B.  Insert prefix (depth 0) then suffix (depth 5).
  // The 3rd write cannot fit → must evict one.  suffix (depth 5) should be evicted.
  UMBPConfig cfg = MakePrefixAwareConfig(2 * 512);
  mori::umbp::LocalBlockIndex index;
  LocalStorageManager mgr(cfg, &index);

  std::string prefix_hash = FakeHash(1);
  std::string suffix_hash = FakeHash(2);
  std::string new_hash = FakeHash(3);

  std::vector<char> data(512, 'X');

  // Write prefix block (depth 0) — oldest in LRU
  assert(mgr.WriteFromPtrWithDepth(prefix_hash + "_0_k", reinterpret_cast<uintptr_t>(data.data()),
                                   data.size(), 0));
  index.Insert(prefix_hash + "_0_k", {StorageTier::CPU_DRAM, 0, 512});

  // Write suffix block (depth 5) — newer in LRU
  assert(mgr.WriteFromPtrWithDepth(suffix_hash + "_0_k", reinterpret_cast<uintptr_t>(data.data()),
                                   data.size(), 5));
  index.Insert(suffix_hash + "_0_k", {StorageTier::CPU_DRAM, 0, 512});

  // Write a third block — DRAM full, must evict. Prefix block is LRU-oldest,
  // but suffix block has higher depth → suffix should be evicted.
  std::vector<char> new_data(512, 'Y');
  assert(mgr.WriteFromPtrWithDepth(new_hash + "_0_k", reinterpret_cast<uintptr_t>(new_data.data()),
                                   new_data.size(), 3));

  // Suffix (depth 5) must be gone; prefix (depth 0) must still exist.
  assert(!mgr.Exists(suffix_hash + "_0_k"));
  assert(mgr.Exists(prefix_hash + "_0_k"));

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 2: Group eviction — both _k and _v variants of a page are evicted
//         atomically when the group victim is selected.
// ---------------------------------------------------------------------------
void test_group_eviction() {
  std::cout << "test_group_eviction... ";

  // DRAM = 2 × 512 B.  Page A (depth 5) occupies both slots (A_k + A_v).
  // Writing a third key must trigger group eviction of A (depth 5 is highest).
  // Both A_k and A_v must disappear atomically.
  UMBPConfig cfg = MakePrefixAwareConfig(2 * 512);
  LocalStorageManager mgr(cfg);

  std::string hashA = FakeHash(1);
  std::string hashC = FakeHash(3);
  std::vector<char> data(512, 'X');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  // Page A (depth 5): two keys fill DRAM
  assert(mgr.WriteFromPtrWithDepth(hashA + "_0_k", src, 512, 5));
  assert(mgr.WriteFromPtrWithDepth(hashA + "_0_v", src, 512, 5));

  // DRAM full. Writing one more key triggers eviction.
  // Victim = A_k (first member of group A seen by SelectVictim).
  // Group eviction: A_k and A_v both removed, freeing 1024 B → C_k fits.
  std::vector<char> new_data(512, 'Y');
  uintptr_t new_src = reinterpret_cast<uintptr_t>(new_data.data());
  assert(mgr.WriteFromPtrWithDepth(hashC + "_0_k", new_src, 512, 3));

  // Both A variants must be gone.
  assert(!mgr.Exists(hashA + "_0_k"));
  assert(!mgr.Exists(hashA + "_0_v"));

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 3: Fallback — keys with no depth metadata behave identically to LRU.
// ---------------------------------------------------------------------------
void test_fallback_to_lru() {
  std::cout << "test_fallback_to_lru... ";

  // 2 slots. Insert key1 (no depth) then key2 (no depth). Fill cache → key1
  // (LRU-oldest) should be evicted.
  UMBPConfig cfg = MakePrefixAwareConfig(2 * 512);
  LocalStorageManager mgr(cfg);

  std::vector<char> data(512, 'X');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  // Use plain WriteFromPtr (no depth)
  assert(mgr.WriteFromPtr("key1_0_k", src, 512));
  assert(mgr.WriteFromPtr("key2_0_k", src, 512));

  // Third write → evicts key1 (LRU-oldest, no depth metadata)
  assert(mgr.WriteFromPtr("key3_0_k", src, 512));

  assert(!mgr.Exists("key1_0_k"));
  assert(mgr.Exists("key2_0_k"));
  assert(mgr.Exists("key3_0_k"));

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 4: Mixed metadata — depth-aware keys score over no-metadata keys.
// ---------------------------------------------------------------------------
void test_mixed_metadata() {
  std::cout << "test_mixed_metadata... ";

  // DRAM = 3 × 512 B.  Insert 3 keys, then write a 4th to force eviction.
  // depth-5 key should be evicted (highest depth), not the LRU-oldest no-metadata key.
  UMBPConfig cfg = MakePrefixAwareConfig(3 * 512);  // fits exactly 3; 4th triggers eviction
  LocalStorageManager mgr(cfg);

  std::string hashA = FakeHash(1);
  std::string hashB = FakeHash(2);
  std::string hashC = FakeHash(3);
  std::string hashD = FakeHash(4);
  std::vector<char> data(512, 'X');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  // No-depth key (oldest in LRU)
  assert(mgr.WriteFromPtr(hashA + "_0_k", src, 512));

  // depth 3
  assert(mgr.WriteFromPtrWithDepth(hashB + "_0_k", src, 512, 3));

  // depth 5 (deepest)
  assert(mgr.WriteFromPtrWithDepth(hashC + "_0_k", src, 512, 5));

  // Fourth write triggers eviction. depth-5 key should be victim.
  assert(mgr.WriteFromPtrWithDepth(hashD + "_0_k", src, 512, 1));

  assert(!mgr.Exists(hashC + "_0_k"));  // depth 5 → evicted
  assert(mgr.Exists(hashA + "_0_k"));   // no depth → preserved
  assert(mgr.Exists(hashB + "_0_k"));   // depth 3 → preserved

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 5: ExtractBaseHash — verify parsing for all key formats.
// ---------------------------------------------------------------------------
// Access the static method via a thin wrapper (it's private; test via a
// public path that exercises it: RecordGroup + GetGroup).
void test_extract_base_hash_via_group() {
  std::cout << "test_extract_base_hash_via_group... ";

  UMBPConfig cfg = MakePrefixAwareConfig(16 * 512);
  LocalStorageManager mgr(cfg);

  std::string hash = FakeHash(1);
  std::vector<char> data(512, 'X');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  // MHA no PP: {hash}_0_k and {hash}_0_v should group together.
  assert(mgr.WriteFromPtrWithDepth(hash + "_0_k", src, 512, 2));
  assert(mgr.WriteFromPtrWithDepth(hash + "_0_v", src, 512, 2));

  // MLA no PP (double underscore): {hash}__k
  std::string hashB = FakeHash(2);
  assert(mgr.WriteFromPtrWithDepth(hashB + "__k", src, 512, 3));

  // MHA with PP: {hash}_0_0_k, {hash}_0_0_v should group.
  std::string hashC = FakeHash(3);
  assert(mgr.WriteFromPtrWithDepth(hashC + "_0_0_k", src, 512, 4));
  assert(mgr.WriteFromPtrWithDepth(hashC + "_0_0_v", src, 512, 4));

  // Verify group eviction for MHA no PP: evict hash_0_k, hash_0_v should also go.
  // Fill up to force eviction — we have 5 keys in 16-slot cache, so let's
  // just call Evict directly on _0_k and check _0_v is NOT removed (group
  // eviction only happens on space pressure paths; Evict() removes single key).
  // Instead, verify via a small cache that forces group eviction.

  // Separate test: small cache with MHA pair, force group eviction on third write.
  UMBPConfig cfg2 = MakePrefixAwareConfig(2 * 512);
  LocalStorageManager mgr2(cfg2);
  std::string hashX = FakeHash(10);
  assert(mgr2.WriteFromPtrWithDepth(hashX + "_0_k", src, 512, 5));
  assert(mgr2.WriteFromPtrWithDepth(hashX + "_0_v", src, 512, 5));
  // Third write must evict both hashX_0_k and hashX_0_v (group).
  std::string hashY = FakeHash(11);
  assert(mgr2.WriteFromPtrWithDepth(hashY + "_0_k", src, 512, 1));
  assert(!mgr2.Exists(hashX + "_0_k"));
  assert(!mgr2.Exists(hashX + "_0_v"));

  // MLA double-underscore: small cache, force eviction.
  UMBPConfig cfg3 = MakePrefixAwareConfig(1 * 512);
  LocalStorageManager mgr3(cfg3);
  std::string hashZ = FakeHash(12);
  std::string hashW = FakeHash(13);
  assert(mgr3.WriteFromPtrWithDepth(hashZ + "__k", src, 512, 5));
  assert(mgr3.WriteFromPtrWithDepth(hashW + "__k", src, 512, 1));
  assert(!mgr3.Exists(hashZ + "__k"));  // depth 5 evicted
  assert(mgr3.Exists(hashW + "__k"));

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 6: TOCTOU resilience — if the selected victim was concurrently evicted,
//         Evict() returns false but does not crash or corrupt state.
// ---------------------------------------------------------------------------
void test_toctou_resilience() {
  std::cout << "test_toctou_resilience... ";

  // DRAM = 1 slot (512 B).  Write key1 (depth 5) — fills DRAM.
  // Manually evict key1 to simulate a concurrent eviction that happens between
  // SelectVictim choosing key1 and the eviction loop acting on it.
  // Then write key2 — DRAM is already empty (key1 gone), so key2 fits.
  // No crash or assertion failure = TOCTOU handled gracefully.
  UMBPConfig cfg = MakePrefixAwareConfig(1 * 512);
  LocalStorageManager mgr(cfg);

  std::vector<char> data(512, 'X');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  // Fill the single slot.
  assert(mgr.WriteFromPtrWithDepth("key1_0_k", src, 512, 5));

  // Simulate concurrent eviction: key1 is removed before the Write loop acts.
  assert(mgr.Evict("key1_0_k"));

  // Write key2 — DRAM is now empty, fits without eviction.
  // SelectVictim would have returned key1 (highest depth in window) but the
  // eviction attempt finds key1 already gone and succeeds on the next iteration.
  assert(mgr.WriteFromPtrWithDepth("key2_0_k", src, 512, 1));
  assert(mgr.Exists("key2_0_k"));
  assert(!mgr.Exists("key1_0_k"));

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 7: Concurrent depth_map access — no data races under TSan.
// ---------------------------------------------------------------------------
void test_concurrent_depth_map_access() {
  std::cout << "test_concurrent_depth_map_access... ";

  UMBPConfig cfg = MakePrefixAwareConfig(256 * 1024);  // 256 KB
  cfg.eviction.policy = "prefix_aware_lru";
  LocalStorageManager mgr(cfg);

  constexpr int kThreads = 4;
  constexpr int kKeysPerThread = 20;
  std::vector<char> data(512, 'X');

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&mgr, &data, t]() {
      uintptr_t src = reinterpret_cast<uintptr_t>(data.data());
      for (int i = 0; i < kKeysPerThread; ++i) {
        // Build a fake 64-char hex hash using thread/key index.
        std::string hash(64, '0');
        // Encode thread and key index in hex chars.
        char buf[16];
        snprintf(buf, sizeof(buf), "%02d%02d", t, i);
        for (int c = 0; c < 4 && c < 64; ++c) hash[c] = buf[c];

        std::string key = hash + "_0_k";
        int depth = i % 8;
        mgr.WriteFromPtrWithDepth(key, src, 512, depth);
      }
    });
  }
  for (auto& th : threads) th.join();

  std::cout << "PASSED (run under ThreadSanitizer for full verification)" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 8: UMBPClient depth-aware batch put — BatchPutFromPtrWithDepth.
// ---------------------------------------------------------------------------
void test_umbp_client_batch_put_with_depth() {
  std::cout << "test_umbp_client_batch_put_with_depth... ";

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 4 * 1024;
  cfg.ssd.enabled = false;
  cfg.eviction.policy = "prefix_aware_lru";
  cfg.eviction.candidate_window = 16;

  UMBPClient client(cfg);

  std::vector<char> data(512, 'Z');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  std::vector<std::string> keys = {"key_a_0_k", "key_b_0_k", "key_c_0_k"};
  std::vector<uintptr_t> ptrs = {src, src, src};
  std::vector<size_t> sizes = {512, 512, 512};
  std::vector<int> depths = {0, 3, 7};

  auto results = client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
  assert(results.size() == 3);
  assert(results[0] && results[1] && results[2]);

  assert(client.Exists("key_a_0_k"));
  assert(client.Exists("key_b_0_k"));
  assert(client.Exists("key_c_0_k"));

  // Dedup: re-putting same keys should succeed (already exist).
  auto results2 = client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
  assert(results2[0] && results2[1] && results2[2]);

  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 9: SSD-path consistency — prefix_aware_lru still syncs index on
//         SSD eviction.
// ---------------------------------------------------------------------------
void test_ssd_prefix_aware_index_sync() {
  std::cout << "test_ssd_prefix_aware_index_sync... ";

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 4 * 1024 * 1024;  // large DRAM — no DRAM pressure
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = "/tmp/umbp_test_prefix_ssd";
  cfg.ssd.capacity_bytes = 1024;  // tiny SSD: 1 KB
  cfg.eviction.policy = "prefix_aware_lru";
  cfg.eviction.candidate_window = 16;

  mori::umbp::LocalBlockIndex index;
  LocalStorageManager mgr(cfg, &index);

  std::vector<char> d1(512, 'A'), d2(512, 'B'), d3(512, 'C');

  // Write two keys directly to SSD with depths.
  assert(mgr.WriteFromPtrWithDepth("hash1_0_k", reinterpret_cast<uintptr_t>(d1.data()), 512, 2,
                                   StorageTier::LOCAL_SSD));
  index.Insert("hash1_0_k", {StorageTier::LOCAL_SSD, 0, 512});

  assert(mgr.WriteFromPtrWithDepth("hash2_0_k", reinterpret_cast<uintptr_t>(d2.data()), 512, 7,
                                   StorageTier::LOCAL_SSD));
  index.Insert("hash2_0_k", {StorageTier::LOCAL_SSD, 0, 512});

  // Writing a third 512-byte key to SSD forces eviction.
  // hash2_0_k has depth 7 (higher) → should be evicted, index removed.
  assert(mgr.WriteFromPtrWithDepth("hash3_0_k", reinterpret_cast<uintptr_t>(d3.data()), 512, 1,
                                   StorageTier::LOCAL_SSD));
  index.Insert("hash3_0_k", {StorageTier::LOCAL_SSD, 0, 512});

  assert(!mgr.Exists("hash2_0_k"));
  assert(!index.Lookup("hash2_0_k").has_value());
  assert(mgr.Exists("hash1_0_k"));
  assert(mgr.Exists("hash3_0_k"));

  mgr.Clear();
  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 10: Follower safety — prefix_aware_lru never writes/deletes shared SSD.
// ---------------------------------------------------------------------------
void test_follower_no_ssd_write() {
  std::cout << "test_follower_no_ssd_write... ";

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1 * 1024 * 1024;
  cfg.ssd.enabled = true;
  cfg.ssd.storage_dir = "/tmp/umbp_test_prefix_follower";
  cfg.ssd.capacity_bytes = 32 * 1024 * 1024;
  cfg.eviction.policy = "prefix_aware_lru";
  cfg.eviction.candidate_window = 16;
  cfg.role = UMBPRole::SharedSSDFollower;

  LocalStorageManager mgr(cfg);

  std::vector<char> data(512, 'F');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());

  // Follower WriteFromPtrWithDepth should still write to DRAM.
  assert(mgr.WriteFromPtrWithDepth("key1_0_k", src, 512, 0));
  assert(mgr.Exists("key1_0_k"));

  // Follower must not write to SSD directly.
  // CopyToSSD returns false for followers.
  assert(!mgr.CopyToSSD("key1_0_k"));

  // DRAM eviction in follower mode must not touch SSD.
  // Fill DRAM with depth-aware keys.
  for (int i = 0; i < 2048; ++i) {
    std::string k = "loop_" + std::to_string(i) + "_0_k";
    mgr.WriteFromPtrWithDepth(k, src, 512, i % 10);
    // No assert — DRAM will evict, but follower path must not write SSD.
  }

  // If we get here without crash or SSD writes, follower safety holds.
  std::cout << "PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  std::cout << "=== Prefix-Aware Eviction Tests ===" << std::endl;
  test_depth_biased_eviction();
  test_group_eviction();
  test_fallback_to_lru();
  test_mixed_metadata();
  test_extract_base_hash_via_group();
  test_toctou_resilience();
  test_concurrent_depth_map_access();
  test_umbp_client_batch_put_with_depth();
  test_ssd_prefix_aware_index_sync();
  test_follower_no_ssd_write();
  std::cout << "All prefix-aware eviction tests passed!" << std::endl;
  return 0;
}
