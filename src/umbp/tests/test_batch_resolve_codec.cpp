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
// Unit tests for the struct-of-arrays BatchResolveKeysResponse codec
// (batch_resolve_codec.h).  This is the wire format shared by the peer service
// (encoder) and the pool client (decoder) after the resolve-keys payload
// optimization: per-key fields hoisted into parallel arrays, page locations
// flattened, buffer descriptors deduplicated once per batch, and page_size
// hoisted to a single field.  The tests assert the encode->decode round-trip is
// lossless (correctness of the optimization), that the optimization actually
// shrinks the wire payload, that the omit_descs path is honored, and that a
// malformed response is rejected rather than over-read.
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/peer/batch_resolve_codec.h"
#include "umbp/distributed/types.h"
#include "umbp_peer.pb.h"

namespace mori::umbp {
namespace {

BufferMemoryDescBytes MakeDesc(uint32_t buffer_index, const std::string& bytes) {
  BufferMemoryDescBytes d;
  d.buffer_index = buffer_index;
  d.desc_bytes.assign(bytes.begin(), bytes.end());
  return d;
}

KvEncodingDescriptor MakeTurboQuantEncoding(uint64_t stored_bytes) {
  KvEncodingDescriptor d;
  d.kind = KvEncodingKind::TURBOQUANT;
  d.original_dtype = KvDType::BF16;
  d.preset = "turboquant_k8v4";
  d.key_bits = 8;
  d.value_bits = 4;
  d.head_dim = 128;
  d.block_size = 16;
  d.num_layers = 32;
  d.num_tokens = 16;
  d.num_heads = 8;
  d.hidden_dim = 1024;
  d.skip_first_layers = 1;
  d.skip_last_layers = 1;
  d.stored_bytes = stored_bytes;
  d.logical_bytes = stored_bytes;
  return d;
}

// A representative batch: 4 keys mixing found / not-found, varying page counts
// (including a found-but-zero-page key), and several tiers.
std::vector<ResolvedKeyEntry> SampleEntries() {
  std::vector<ResolvedKeyEntry> entries(4);

  entries[0].found = true;
  entries[0].tier = TierType::HBM;
  entries[0].size = 3 * 4096;
  entries[0].encoding = MakeTurboQuantEncoding(entries[0].size);
  entries[0].pages = {{0, 10}, {1, 11}, {0, 12}};

  entries[1].found = false;  // miss: no tier / size / pages

  entries[2].found = true;
  entries[2].tier = TierType::DRAM;
  entries[2].size = 4096;
  entries[2].encoding = RawKvEncoding(entries[2].size, KvDType::FP16);
  entries[2].pages = {{2, 7}};

  entries[3].found = true;
  entries[3].tier = TierType::SSD;
  entries[3].size = 0;  // found but zero pages (degenerate but legal)
  entries[3].pages = {};

  return entries;
}

void ExpectKeyMatches(const ResolvedKeyEntry& src, const DecodedResolveKey& dec) {
  EXPECT_EQ(src.found, dec.found);
  if (!src.found) return;
  EXPECT_EQ(src.tier, dec.tier);
  EXPECT_EQ(src.size, dec.size);
  EXPECT_EQ(src.encoding, dec.encoding);
  ASSERT_EQ(src.pages.size(), dec.pages.size());
  for (size_t p = 0; p < src.pages.size(); ++p) {
    EXPECT_EQ(src.pages[p].buffer_index, dec.pages[p].buffer_index);
    EXPECT_EQ(src.pages[p].page_index, dec.pages[p].page_index);
  }
}

TEST(BatchResolveCodec, RoundTripPreservesEveryField) {
  const auto entries = SampleEntries();
  const uint64_t page_size = 4096;
  const std::vector<BufferMemoryDescBytes> descs = {
      MakeDesc(0, "desc-zero"), MakeDesc(1, "desc-one"), MakeDesc(2, "desc-two")};

  ::umbp::BatchResolveKeysResponse resp;
  EncodeBatchResolveResponse(entries, page_size, descs, &resp);

  // Survives a real serialize/parse cycle, not just in-memory copy.
  std::string wire;
  ASSERT_TRUE(resp.SerializeToString(&wire));
  ::umbp::BatchResolveKeysResponse parsed;
  ASSERT_TRUE(parsed.ParseFromString(wire));

  EXPECT_EQ(BatchResolveKeyCount(parsed), static_cast<int>(entries.size()));

  DecodedBatchResolve decoded = DecodeBatchResolveResponse(parsed);
  EXPECT_EQ(decoded.page_size, page_size);
  ASSERT_EQ(decoded.keys.size(), entries.size());
  for (size_t i = 0; i < entries.size(); ++i) ExpectKeyMatches(entries[i], decoded.keys[i]);

  ASSERT_EQ(decoded.descs.size(), descs.size());
  for (size_t i = 0; i < descs.size(); ++i) {
    EXPECT_EQ(decoded.descs[i].buffer_index, descs[i].buffer_index);
    EXPECT_EQ(decoded.descs[i].desc_bytes, descs[i].desc_bytes);
  }
}

TEST(BatchResolveCodec, FlattenedPagesSliceByKey) {
  // Two found keys with different page counts must not bleed into each other.
  std::vector<ResolvedKeyEntry> entries(2);
  entries[0].found = true;
  entries[0].size = 2 * 4096;
  entries[0].pages = {{5, 100}, {6, 101}};
  entries[1].found = true;
  entries[1].size = 4096;
  entries[1].pages = {{7, 200}};

  ::umbp::BatchResolveKeysResponse resp;
  EncodeBatchResolveResponse(entries, 4096, {}, &resp);

  // Flattened arrays hold exactly the concatenation, in order.
  ASSERT_EQ(resp.buffer_index_size(), 3);
  EXPECT_EQ(resp.buffer_index(0), 5u);
  EXPECT_EQ(resp.buffer_index(1), 6u);
  EXPECT_EQ(resp.buffer_index(2), 7u);
  EXPECT_EQ(resp.page_count(0), 2u);
  EXPECT_EQ(resp.page_count(1), 1u);

  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resp);
  ASSERT_EQ(decoded.keys.size(), 2u);
  ASSERT_EQ(decoded.keys[0].pages.size(), 2u);
  ASSERT_EQ(decoded.keys[1].pages.size(), 1u);
  EXPECT_EQ(decoded.keys[1].pages[0].buffer_index, 7u);
  EXPECT_EQ(decoded.keys[1].pages[0].page_index, 200u);
}

TEST(BatchResolveCodec, OmitDescsRoundTrips) {
  const auto entries = SampleEntries();

  ::umbp::BatchResolveKeysResponse resp;
  EncodeBatchResolveResponse(entries, 4096, /*descs=*/{}, &resp);
  EXPECT_EQ(resp.descs_size(), 0);

  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resp);
  EXPECT_TRUE(decoded.descs.empty());
  ASSERT_EQ(decoded.keys.size(), entries.size());
  for (size_t i = 0; i < entries.size(); ++i) ExpectKeyMatches(entries[i], decoded.keys[i]);
}

TEST(BatchResolveCodec, AllNotFound) {
  std::vector<ResolvedKeyEntry> entries(3);  // all default => found=false
  ::umbp::BatchResolveKeysResponse resp;
  EncodeBatchResolveResponse(entries, 4096, {}, &resp);

  EXPECT_EQ(BatchResolveKeyCount(resp), 3);
  EXPECT_EQ(resp.buffer_index_size(), 0);

  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resp);
  ASSERT_EQ(decoded.keys.size(), 3u);
  for (const auto& k : decoded.keys) EXPECT_FALSE(k.found);
}

TEST(BatchResolveCodec, EmptyBatch) {
  ::umbp::BatchResolveKeysResponse resp;
  EncodeBatchResolveResponse({}, 4096, {}, &resp);
  EXPECT_EQ(BatchResolveKeyCount(resp), 0);
  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resp);
  EXPECT_TRUE(decoded.keys.empty());
  EXPECT_EQ(decoded.page_size, 4096u);
}

TEST(BatchResolveCodec, MalformedMismatchedArraysRejected) {
  // found has 2 entries but the per-key arrays disagree -> decoder must refuse.
  ::umbp::BatchResolveKeysResponse resp;
  resp.set_page_size(4096);
  resp.add_found(true);
  resp.add_found(true);
  resp.add_size(4096);  // only one size for two found
  resp.add_page_count(0);
  resp.add_page_count(0);
  // tier intentionally left empty (size 0 != 2)

  EXPECT_EQ(BatchResolveKeyCount(resp), 2);
  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resp);
  EXPECT_TRUE(decoded.keys.empty());  // rejected, not partially decoded
}

TEST(BatchResolveCodec, MalformedTruncatedPagesRejected) {
  // page_count sums to 3 but only 1 flattened page is present.
  ::umbp::BatchResolveKeysResponse resp;
  resp.set_page_size(4096);
  resp.add_found(true);
  resp.add_tier(::umbp::TIER_HBM);
  resp.add_size(3 * 4096);
  resp.add_page_count(3);
  resp.add_buffer_index(0);
  resp.add_page_index(0);

  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resp);
  EXPECT_TRUE(decoded.keys.empty());
}

// The whole point of the optimization: with descriptors shared across many
// keys, the SoA layout that dedupes them once per batch is dramatically smaller
// than repeating a full descriptor set under every key.
TEST(BatchResolveCodec, OptimizedWireIsSmallerThanPerKeyDescs) {
  const int kKeys = 1000;
  const int kPagesPerKey = 8;
  const int kBuffers = 4;
  std::vector<BufferMemoryDescBytes> descs;
  for (int b = 0; b < kBuffers; ++b) descs.push_back(MakeDesc(b, std::string(64, 'x')));

  std::vector<ResolvedKeyEntry> entries(kKeys);
  for (int i = 0; i < kKeys; ++i) {
    entries[i].found = true;
    entries[i].tier = TierType::HBM;
    entries[i].size = kPagesPerKey * 4096;
    for (int p = 0; p < kPagesPerKey; ++p) {
      entries[i].pages.push_back({static_cast<uint32_t>(p % kBuffers), static_cast<uint32_t>(p)});
    }
  }

  ::umbp::BatchResolveKeysResponse opt;
  EncodeBatchResolveResponse(entries, 4096, descs, &opt);
  std::string opt_wire;
  ASSERT_TRUE(opt.SerializeToString(&opt_wire));

  // Naive baseline: the descriptor bytes repeated once per key (what the old
  // per-entry ResolveKeyResponse layout carried).
  const size_t per_key_desc_bytes = descs.size() * (64 + 4);
  const size_t naive_desc_total = per_key_desc_bytes * kKeys;
  const size_t opt_desc_total = per_key_desc_bytes;  // once per batch

  EXPECT_LT(opt_wire.size(), naive_desc_total);
  EXPECT_GT(naive_desc_total - opt_desc_total, 0u);
}

}  // namespace
}  // namespace mori::umbp
