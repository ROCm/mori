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
#include <climits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "umbp/local/storage/dram_tier.h"
#include "umbp/local/storage/local_storage_manager.h"

using namespace mori::umbp;

void test_default_single_chunk() {
  std::cout << "test_default_single_chunk... ";

  // Constructor initializes a single full-arena chunk — no ConfigureChunks needed.
  DRAMTier tier(4096);

  std::vector<char> data(1024, 'A');
  assert(tier.Write("k1", data.data(), data.size()));
  assert(tier.Exists("k1"));

  auto loc = tier.GetSlotChunkLocation("k1");
  assert(loc.has_value());
  assert(loc->chunk_index == 0);
  assert(loc->offset == 0);

  auto chunks = tier.GetExportableChunks();
  assert(chunks.size() == 1);
  assert(chunks[0].size == 4096);

  std::cout << "PASSED" << std::endl;
}

void test_configure_single_chunk() {
  std::cout << "test_configure_single_chunk... ";

  DRAMTier tier(4096);
  tier.ConfigureChunks(SIZE_MAX);

  std::vector<char> data(1024, 'B');
  assert(tier.Write("k1", data.data(), data.size()));

  auto loc = tier.GetSlotChunkLocation("k1");
  assert(loc.has_value());
  assert(loc->chunk_index == 0);
  assert(loc->offset == 0);

  auto chunks = tier.GetExportableChunks();
  assert(chunks.size() == 1);
  assert(chunks[0].size == 4096);

  // Also test with chunk_size == 0 (should behave the same).
  DRAMTier tier2(4096);
  tier2.ConfigureChunks(0);
  auto chunks2 = tier2.GetExportableChunks();
  assert(chunks2.size() == 1);
  assert(chunks2[0].size == 4096);

  std::cout << "PASSED" << std::endl;
}

void test_configure_multi_chunk() {
  std::cout << "test_configure_multi_chunk... ";

  // 10 KB arena, 4 KB chunks → 3 chunks (4KB + 4KB + 2KB)
  const size_t arena = 10 * 1024;
  const size_t chunk_size = 4 * 1024;
  DRAMTier tier(arena);
  tier.ConfigureChunks(chunk_size);

  auto chunks = tier.GetExportableChunks();
  assert(chunks.size() == 3);
  assert(chunks[0].size == 4096);
  assert(chunks[1].size == 4096);
  assert(chunks[2].size == 2048);

  std::cout << "PASSED" << std::endl;
}

void test_first_fit_across_chunks() {
  std::cout << "test_first_fit_across_chunks... ";

  // 8 KB arena, 4 KB chunks → 2 chunks
  const size_t arena = 8 * 1024;
  const size_t chunk_size = 4 * 1024;
  DRAMTier tier(arena);
  tier.ConfigureChunks(chunk_size);

  // Fill chunk 0 completely
  std::vector<char> d1(4096, 'A');
  assert(tier.Write("k1", d1.data(), d1.size()));
  auto loc1 = tier.GetSlotChunkLocation("k1");
  assert(loc1->chunk_index == 0);

  // Next allocation should go to chunk 1
  std::vector<char> d2(1024, 'B');
  assert(tier.Write("k2", d2.data(), d2.size()));
  auto loc2 = tier.GetSlotChunkLocation("k2");
  assert(loc2->chunk_index == 1);
  assert(loc2->offset == 0);

  std::cout << "PASSED" << std::endl;
}

void test_no_cross_chunk_block() {
  std::cout << "test_no_cross_chunk_block... ";

  // 8 KB arena, 4 KB chunks
  const size_t arena = 8 * 1024;
  const size_t chunk_size = 4 * 1024;
  DRAMTier tier(arena);
  tier.ConfigureChunks(chunk_size);

  // Write a 3 KB block into chunk 0
  std::vector<char> d1(3072, 'A');
  assert(tier.Write("k1", d1.data(), d1.size()));
  auto loc1 = tier.GetSlotChunkLocation("k1");
  assert(loc1->chunk_index == 0);
  assert(loc1->offset == 0);

  // 1 KB remaining in chunk 0. Write a 2 KB block — must go to chunk 1.
  std::vector<char> d2(2048, 'B');
  assert(tier.Write("k2", d2.data(), d2.size()));
  auto loc2 = tier.GetSlotChunkLocation("k2");
  assert(loc2->chunk_index == 1);

  // Verify data integrity
  std::vector<char> buf(2048, 0);
  assert(tier.ReadIntoPtr("k2", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == d2);

  std::cout << "PASSED" << std::endl;
}

void test_reject_oversized_block() {
  std::cout << "test_reject_oversized_block... ";

  // 8 KB arena, 2 KB chunks
  const size_t arena = 8 * 1024;
  const size_t chunk_size = 2 * 1024;
  DRAMTier tier(arena);
  tier.ConfigureChunks(chunk_size);

  // A 3 KB block exceeds the 2 KB chunk size — should fail.
  std::vector<char> data(3072, 'X');
  assert(!tier.Write("big", data.data(), data.size()));

  std::cout << "PASSED" << std::endl;
}

void test_tail_chunk_smaller() {
  std::cout << "test_tail_chunk_smaller... ";

  // 10 KB arena, 4 KB chunks → chunk2 = 2 KB
  const size_t arena = 10 * 1024;
  const size_t chunk_size = 4 * 1024;
  DRAMTier tier(arena);
  tier.ConfigureChunks(chunk_size);

  // Fill chunks 0 and 1
  std::vector<char> d1(4096, 'A');
  assert(tier.Write("k1", d1.data(), d1.size()));
  assert(tier.Write("k2", d1.data(), d1.size()));

  // Write 1 KB into the tail chunk (2 KB available)
  std::vector<char> d3(1024, 'C');
  assert(tier.Write("k3", d3.data(), d3.size()));
  auto loc3 = tier.GetSlotChunkLocation("k3");
  assert(loc3->chunk_index == 2);

  // 3 KB block should not fit in the 2 KB tail chunk (chunks 0,1 are full)
  std::vector<char> d4(3072, 'D');
  assert(!tier.Write("k4", d4.data(), d4.size()));

  // But 1 KB more should fit in chunk 2's remaining 1 KB
  std::vector<char> d5(1024, 'E');
  assert(tier.Write("k5", d5.data(), d5.size()));
  auto loc5 = tier.GetSlotChunkLocation("k5");
  assert(loc5->chunk_index == 2);
  assert(loc5->offset == 1024);

  std::cout << "PASSED" << std::endl;
}

void test_per_chunk_coalescing() {
  std::cout << "test_per_chunk_coalescing... ";

  // 8 KB arena, 4 KB chunks
  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);

  // Write two 2 KB blocks into chunk 0
  std::vector<char> d1(2048, 'A'), d2(2048, 'B');
  assert(tier.Write("k1", d1.data(), d1.size()));
  assert(tier.Write("k2", d2.data(), d2.size()));

  auto loc1 = tier.GetSlotChunkLocation("k1");
  auto loc2 = tier.GetSlotChunkLocation("k2");
  assert(loc1->chunk_index == 0);
  assert(loc2->chunk_index == 0);

  // Evict both — free space should coalesce back to 4 KB
  assert(tier.Evict("k1"));
  assert(tier.Evict("k2"));

  // Should be able to write a full 4 KB block into chunk 0 (coalesced)
  std::vector<char> d3(4096, 'C');
  assert(tier.Write("k3", d3.data(), d3.size()));
  auto loc3 = tier.GetSlotChunkLocation("k3");
  assert(loc3->chunk_index == 0);
  assert(loc3->offset == 0);

  std::cout << "PASSED" << std::endl;
}

void test_get_slot_chunk_location() {
  std::cout << "test_get_slot_chunk_location... ";

  // 8 KB arena, 4 KB chunks
  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);

  std::vector<char> d1(1024, 'A'), d2(2048, 'B'), d3(1024, 'C');
  assert(tier.Write("k1", d1.data(), d1.size()));
  assert(tier.Write("k2", d2.data(), d2.size()));
  // k1: chunk 0, offset 0
  // k2: chunk 0, offset 1024

  auto loc1 = tier.GetSlotChunkLocation("k1");
  assert(loc1.has_value());
  assert(loc1->chunk_index == 0);
  assert(loc1->offset == 0);

  auto loc2 = tier.GetSlotChunkLocation("k2");
  assert(loc2.has_value());
  assert(loc2->chunk_index == 0);
  assert(loc2->offset == 1024);

  // k3 spills to chunk 1 (only 1 KB left in chunk 0, k3 is 1 KB so it fits)
  assert(tier.Write("k3", d3.data(), d3.size()));
  auto loc3 = tier.GetSlotChunkLocation("k3");
  assert(loc3.has_value());
  assert(loc3->chunk_index == 0);
  assert(loc3->offset == 3072);

  // Non-existent key
  assert(!tier.GetSlotChunkLocation("nope").has_value());

  std::cout << "PASSED" << std::endl;
}

void test_get_exportable_chunks() {
  std::cout << "test_get_exportable_chunks... ";

  DRAMTier tier(12 * 1024);
  tier.ConfigureChunks(4 * 1024);

  auto chunks = tier.GetExportableChunks();
  assert(chunks.size() == 3);
  assert(chunks[0].size == 4096);
  assert(chunks[1].size == 4096);
  assert(chunks[2].size == 4096);

  // Verify base pointers are sequential.
  char* base0 = static_cast<char*>(chunks[0].buffer);
  assert(static_cast<char*>(chunks[1].buffer) == base0 + 4096);
  assert(static_cast<char*>(chunks[2].buffer) == base0 + 8192);

  std::cout << "PASSED" << std::endl;
}

void test_reconfigure_before_allocation_ok() {
  std::cout << "test_reconfigure_before_allocation_ok... ";

  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(0);  // single chunk

  // No allocations yet — reconfiguring should succeed.
  tier.ConfigureChunks(4 * 1024);  // split into 2 chunks

  auto chunks = tier.GetExportableChunks();
  assert(chunks.size() == 2);
  assert(chunks[0].size == 4096);
  assert(chunks[1].size == 4096);

  // Write should work with the new layout.
  std::vector<char> data(4096, 'A');
  assert(tier.Write("k1", data.data(), data.size()));
  auto loc = tier.GetSlotChunkLocation("k1");
  assert(loc->chunk_index == 0);

  std::cout << "PASSED" << std::endl;
}

void test_reject_reconfigure_after_allocation() {
  std::cout << "test_reject_reconfigure_after_allocation... ";

  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);

  std::vector<char> data(1024, 'A');
  assert(tier.Write("k1", data.data(), data.size()));

  // Reconfiguring after allocation should throw.
  bool threw = false;
  try {
    tier.ConfigureChunks(2 * 1024);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  std::cout << "PASSED" << std::endl;
}

void test_get_location_id_returns_global_offset() {
  std::cout << "test_get_location_id_returns_global_offset... ";

  // 8 KB arena, 4 KB chunks
  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);

  // Fill chunk 0
  std::vector<char> d1(4096, 'A');
  assert(tier.Write("k1", d1.data(), d1.size()));

  // Write into chunk 1
  std::vector<char> d2(512, 'B');
  assert(tier.Write("k2", d2.data(), d2.size()));

  // GetLocationId returns the global byte offset (base class contract).
  auto lid1 = tier.GetLocationId("k1");
  assert(lid1.has_value());
  assert(*lid1 == "0");

  auto lid2 = tier.GetLocationId("k2");
  assert(lid2.has_value());
  assert(*lid2 == "4096");

  // Non-existent key
  assert(!tier.GetLocationId("nope").has_value());

  std::cout << "PASSED" << std::endl;
}

void test_clear_resets_chunks() {
  std::cout << "test_clear_resets_chunks... ";

  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);

  // Fill both chunks completely
  std::vector<char> d1(4096, 'A');
  assert(tier.Write("k1", d1.data(), d1.size()));
  assert(tier.Write("k2", d1.data(), d1.size()));
  assert(!tier.Write("k3", d1.data(), d1.size()));  // should fail — full

  // Clear resets all chunk free lists
  tier.Clear();

  // Should be able to write again
  assert(tier.Write("k4", d1.data(), d1.size()));
  auto loc4 = tier.GetSlotChunkLocation("k4");
  assert(loc4->chunk_index == 0);
  assert(loc4->offset == 0);

  std::cout << "PASSED" << std::endl;
}

void test_get_slot_offset_global() {
  std::cout << "test_get_slot_offset_global... ";

  // 8 KB arena, 4 KB chunks
  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);

  // Fill chunk 0
  std::vector<char> d1(4096, 'A');
  assert(tier.Write("k1", d1.data(), d1.size()));

  // Write 1 KB into chunk 1
  std::vector<char> d2(1024, 'B');
  assert(tier.Write("k2", d2.data(), d2.size()));

  // k1: global offset should be 0
  auto off1 = tier.GetSlotOffset("k1");
  assert(off1.has_value());
  assert(*off1 == 0);

  // k2: global offset should be 4096 (chunk 1, offset 0 within chunk)
  auto off2 = tier.GetSlotOffset("k2");
  assert(off2.has_value());
  assert(*off2 == 4096);

  std::cout << "PASSED" << std::endl;
}

void test_data_integrity_multi_chunk() {
  std::cout << "test_data_integrity_multi_chunk... ";

  // 8 KB arena, 2 KB chunks → 4 chunks
  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(2 * 1024);

  // Write distinct data to each chunk
  for (int i = 0; i < 4; ++i) {
    std::string key = "k" + std::to_string(i);
    std::vector<char> data(2048, 'A' + i);
    assert(tier.Write(key, data.data(), data.size()));
    auto loc = tier.GetSlotChunkLocation(key);
    assert(loc->chunk_index == static_cast<uint32_t>(i));
  }

  // Read back and verify
  for (int i = 0; i < 4; ++i) {
    std::string key = "k" + std::to_string(i);
    std::vector<char> expected(2048, 'A' + i);
    std::vector<char> buf(2048, 0);
    assert(tier.ReadIntoPtr(key, reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
    assert(buf == expected);
  }

  std::cout << "PASSED" << std::endl;
}

void test_clear_does_not_unlock_reconfigure() {
  std::cout << "test_clear_does_not_unlock_reconfigure... ";

  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);  // 2 chunks

  // Allocate a block — permanently locks the layout.
  std::vector<char> data(1024, 'A');
  assert(tier.Write("k1", data.data(), data.size()));

  // Clear() empties slots but does NOT unlock reconfiguration,
  // because registered MRs depend on the layout staying fixed.
  tier.Clear();

  bool threw = false;
  try {
    tier.ConfigureChunks(2 * 1024);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  // Writing with the original layout still works after Clear().
  assert(tier.Write("k2", data.data(), data.size()));
  auto loc = tier.GetSlotChunkLocation("k2");
  assert(loc.has_value());
  assert(loc->chunk_index == 0);

  std::cout << "PASSED" << std::endl;
}

void test_seal_locks_layout_before_write() {
  std::cout << "test_seal_locks_layout_before_write... ";

  DRAMTier tier(8 * 1024);
  tier.ConfigureChunks(4 * 1024);  // 2 chunks

  // Seal without any writes — simulates distributed init completing.
  tier.SealChunkLayout();

  // Reconfiguration must be rejected even though no writes happened.
  bool threw = false;
  try {
    tier.ConfigureChunks(2 * 1024);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  // Writing with the sealed layout still works.
  std::vector<char> data(4096, 'A');
  assert(tier.Write("k1", data.data(), data.size()));
  auto loc = tier.GetSlotChunkLocation("k1");
  assert(loc->chunk_index == 0);

  std::cout << "PASSED" << std::endl;
}

void test_build_tier_location_info_consistency() {
  std::cout << "test_build_tier_location_info_consistency... ";

  // Create a LocalStorageManager with a chunked DRAMTier (via reconfiguration).
  UMBPConfig config;
  config.dram.capacity_bytes = 8 * 1024;
  config.ssd.enabled = false;
  LocalStorageManager mgr(config);

  // Reconfigure DRAM to use 4 KB chunks (no allocations yet, so this is safe).
  auto* dram = mgr.GetTierAs<DRAMTier>(StorageTier::CPU_DRAM);
  assert(dram != nullptr);
  dram->ConfigureChunks(4 * 1024);

  // Fill chunk 0
  std::vector<char> d1(4096, 'A');
  assert(mgr.Write("k1", d1.data(), d1.size()));

  // Write into chunk 1
  std::vector<char> d2(512, 'B');
  assert(mgr.Write("k2", d2.data(), d2.size()));

  // Verify BuildTierLocationInfo produces chunk-aware location_ids.
  auto info1 = mgr.BuildTierLocationInfo(dram, "k1", d1.size());
  assert(info1.has_value());
  assert(info1->location_id == "0:0");

  auto info2 = mgr.BuildTierLocationInfo(dram, "k2", d2.size());
  assert(info2.has_value());
  assert(info2->location_id == "1:0");

  // Verify GetSlotChunkLocation agrees.
  auto loc1 = dram->GetSlotChunkLocation("k1");
  assert(loc1.has_value());
  std::string expected1 = std::to_string(loc1->chunk_index) + ":" + std::to_string(loc1->offset);
  assert(info1->location_id == expected1);

  auto loc2 = dram->GetSlotChunkLocation("k2");
  assert(loc2.has_value());
  std::string expected2 = std::to_string(loc2->chunk_index) + ":" + std::to_string(loc2->offset);
  assert(info2->location_id == expected2);

  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== DRAMTier Chunk Tests ===" << std::endl;
  test_default_single_chunk();
  test_configure_single_chunk();
  test_configure_multi_chunk();
  test_first_fit_across_chunks();
  test_no_cross_chunk_block();
  test_reject_oversized_block();
  test_tail_chunk_smaller();
  test_per_chunk_coalescing();
  test_get_slot_chunk_location();
  test_get_exportable_chunks();
  test_reconfigure_before_allocation_ok();
  test_reject_reconfigure_after_allocation();
  test_get_location_id_returns_global_offset();
  test_clear_resets_chunks();
  test_get_slot_offset_global();
  test_data_integrity_multi_chunk();
  test_clear_does_not_unlock_reconfigure();
  test_seal_locks_layout_before_write();
  test_build_tier_location_info_consistency();
  std::cout << "All DRAMTier chunk tests passed!" << std::endl;
  return 0;
}
