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
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "umbp/distributed/types.h"

namespace mori::umbp {

struct TurboQuantLayout {
  uint64_t key_packed_size = 0;
  uint64_t value_packed_size = 0;
  uint64_t slot_size = 0;
  uint64_t slot_size_aligned = 0;
  uint64_t num_blocks = 0;
  uint64_t compressed_layers = 0;
  uint64_t raw_layers = 0;
  uint64_t compressed_bytes = 0;
  uint64_t raw_bytes = 0;
  uint64_t stored_bytes = 0;
};

inline uint64_t PackedBytes(uint64_t elements, uint32_t bits) {
  return (elements * static_cast<uint64_t>(bits) + 7) / 8;
}

inline uint64_t DTypeBytes(KvDType dtype) {
  switch (dtype) {
    case KvDType::FP16:
    case KvDType::BF16:
      return 2;
    case KvDType::FP8:
    case KvDType::UINT8:
      return 1;
    default:
      return 0;
  }
}

inline bool IsRawEncoding(const KvEncodingDescriptor& desc) {
  return desc.kind == KvEncodingKind::RAW || desc.kind == KvEncodingKind::UNKNOWN;
}

inline bool IsTurboQuantPresetSupported(const std::string& preset) {
  return preset == "turboquant_k8v4" || preset == "turboquant_4bit_nc" ||
         preset == "turboquant_k3v4_nc" || preset == "turboquant_3bit_nc";
}

inline std::optional<TurboQuantLayout> ComputeTurboQuantLayout(
    const KvEncodingDescriptor& desc) {
  if (desc.kind != KvEncodingKind::TURBOQUANT) return std::nullopt;
  if (!IsTurboQuantPresetSupported(desc.preset)) return std::nullopt;
  if (desc.head_dim == 0 || desc.block_size == 0 || desc.num_layers == 0 ||
      desc.num_tokens == 0 || desc.num_heads == 0) {
    return std::nullopt;
  }

  const uint32_t key_bits =
      desc.key_bits != 0 ? desc.key_bits : (desc.preset == "turboquant_k8v4" ? 8 : 4);
  const uint32_t value_bits = desc.value_bits != 0 ? desc.value_bits : 4;

  TurboQuantLayout layout;
  const bool key_fp8 = key_bits == 8;
  layout.key_packed_size = key_fp8 ? desc.head_dim : PackedBytes(desc.head_dim, key_bits) + 2;
  layout.value_packed_size = PackedBytes(desc.head_dim, value_bits) + 4;
  layout.slot_size = layout.key_packed_size + layout.value_packed_size;
  layout.slot_size_aligned = layout.slot_size + (layout.slot_size % 2);
  layout.num_blocks = (desc.num_tokens + desc.block_size - 1) / desc.block_size;

  const uint32_t quant_start =
      desc.skip_first_layers < desc.num_layers ? desc.skip_first_layers : desc.num_layers;
  const uint32_t raw_suffix =
      desc.skip_last_layers < desc.num_layers - quant_start ? desc.skip_last_layers
                                                            : desc.num_layers - quant_start;
  layout.compressed_layers = desc.num_layers - quant_start - raw_suffix;
  layout.raw_layers = desc.num_layers - layout.compressed_layers;

  const uint64_t raw_dtype_bytes = DTypeBytes(desc.original_dtype);
  if (layout.raw_layers > 0 && raw_dtype_bytes == 0) return std::nullopt;

  layout.raw_bytes = 2ull * layout.raw_layers * desc.num_tokens *
                     static_cast<uint64_t>(desc.num_heads) * desc.head_dim * raw_dtype_bytes;
  layout.compressed_bytes = layout.compressed_layers * layout.num_blocks * desc.block_size *
                            static_cast<uint64_t>(desc.num_heads) * layout.slot_size_aligned;
  layout.stored_bytes = layout.raw_bytes + layout.compressed_bytes;
  return layout;
}

inline bool ValidateKvEncodingDescriptor(const KvEncodingDescriptor& desc,
                                         std::string* error = nullptr) {
  auto fail = [&](const char* msg) {
    if (error != nullptr) *error = msg;
    return false;
  };
  if (desc.schema_version == 0) return fail("schema_version must be non-zero");
  if (IsRawEncoding(desc)) {
    if (desc.stored_bytes == 0) return fail("raw encoding stored_bytes must be non-zero");
    return true;
  }
  if (desc.kind != KvEncodingKind::TURBOQUANT) return fail("unsupported KV encoding kind");
  auto layout = ComputeTurboQuantLayout(desc);
  if (!layout.has_value()) return fail("invalid TurboQuant descriptor");
  if (desc.stored_bytes != 0 && desc.stored_bytes != layout->stored_bytes) {
    return fail("TurboQuant stored_bytes does not match descriptor layout");
  }
  return true;
}

inline KvEncodingDescriptor RawKvEncoding(uint64_t stored_bytes,
                                          KvDType dtype = KvDType::UNKNOWN) {
  KvEncodingDescriptor desc;
  desc.kind = KvEncodingKind::RAW;
  desc.original_dtype = dtype;
  desc.stored_bytes = stored_bytes;
  desc.logical_bytes = stored_bytes;
  return desc;
}

}  // namespace mori::umbp
