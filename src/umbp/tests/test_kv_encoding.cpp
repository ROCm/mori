// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include "umbp/codec/kv_encoding.h"

namespace mori::umbp {
namespace {

TEST(KvEncoding, RawEncodingCarriesStoredByteSize) {
  auto desc = RawKvEncoding(4096, KvDType::BF16);
  EXPECT_TRUE(ValidateKvEncodingDescriptor(desc));
  EXPECT_EQ(desc.kind, KvEncodingKind::RAW);
  EXPECT_EQ(desc.stored_bytes, 4096u);
  EXPECT_EQ(desc.logical_bytes, 4096u);
}

TEST(KvEncoding, TurboQuantK8V4ComputesSerializedBytes) {
  KvEncodingDescriptor desc;
  desc.kind = KvEncodingKind::TURBOQUANT;
  desc.original_dtype = KvDType::BF16;
  desc.preset = "turboquant_k8v4";
  desc.key_bits = 8;
  desc.value_bits = 4;
  desc.head_dim = 128;
  desc.block_size = 16;
  desc.num_layers = 4;
  desc.num_tokens = 16;
  desc.num_heads = 2;
  desc.hidden_dim = 256;
  desc.skip_first_layers = 1;
  desc.skip_last_layers = 1;

  auto layout = ComputeTurboQuantLayout(desc);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->key_packed_size, 128u);
  EXPECT_EQ(layout->value_packed_size, 68u);
  EXPECT_EQ(layout->slot_size_aligned, 196u);
  EXPECT_EQ(layout->compressed_layers, 2u);
  EXPECT_EQ(layout->raw_layers, 2u);
  EXPECT_EQ(layout->compressed_bytes, 2u * 1u * 16u * 2u * 196u);
  EXPECT_EQ(layout->raw_bytes, 2u * 2u * 16u * 2u * 128u * 2u);
  EXPECT_EQ(layout->stored_bytes, layout->compressed_bytes + layout->raw_bytes);

  desc.stored_bytes = layout->stored_bytes;
  EXPECT_TRUE(ValidateKvEncodingDescriptor(desc));
}

TEST(KvEncoding, RejectsMismatchedTurboQuantStoredBytes) {
  KvEncodingDescriptor desc;
  desc.kind = KvEncodingKind::TURBOQUANT;
  desc.original_dtype = KvDType::BF16;
  desc.preset = "turboquant_4bit_nc";
  desc.key_bits = 4;
  desc.value_bits = 4;
  desc.head_dim = 64;
  desc.block_size = 16;
  desc.num_layers = 2;
  desc.num_tokens = 16;
  desc.num_heads = 1;
  desc.hidden_dim = 64;
  desc.stored_bytes = 1;

  std::string error;
  EXPECT_FALSE(ValidateKvEncodingDescriptor(desc, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace mori::umbp
