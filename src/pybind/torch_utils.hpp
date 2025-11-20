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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/library_types.h>
#include <torch/torch.h>

namespace mori {

template <typename T>
inline torch::Dtype GetTorchDataType() {
  if constexpr (std::is_same_v<T, float>) {
    return torch::kFloat32;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return torch::kUInt32;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return torch::kInt32;
  } else if constexpr (std::is_same_v<T, size_t>) {
    return torch::kUInt64;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return torch::kUInt64;
  } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
    return torch::kBFloat16;
  } else if constexpr (std::is_same_v<T, __hip_fp8_e4m3>) {
    return torch::kFloat8_e4m3fn;
  } else if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz>) {
    return torch::kFloat8_e4m3fnuz;
  } else {
    static_assert(false, "Unsupported data type");
  }
}

inline hipDataType ScalarTypeToHipDataType(at::ScalarType scalarType) {
  switch (scalarType) {
    case at::kFloat:
      return HIP_R_32F;
    case at::kBFloat16:
      return HIP_R_16BF;
    case at::kFloat8_e4m3fn:
      return HIP_R_8F_E4M3;
    case at::kFloat8_e4m3fnuz:
      return HIP_R_8F_E4M3_FNUZ;
    default:
      throw std::runtime_error("Unsupported scalar type");
  }
}

}  // namespace mori
