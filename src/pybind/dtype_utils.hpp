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
// Minimal dtype utilities without depending on libtorch
#pragma once

#include <hip/library_types.h>

#include <cstdint>

namespace mori {

// Public enum for Python-side mapping
enum class MoriScalarType : int32_t {
  Float32 = 0,
  BFloat16 = 1,
  Float8_e4m3fnuz = 2,
  Int32 = 3,
  UInt32 = 4,
  UInt64 = 5,
};

inline size_t MoriScalarTypeSize(MoriScalarType t) {
  switch (t) {
    case MoriScalarType::Float32:
      return 4;
    case MoriScalarType::BFloat16:
      return 2;
    case MoriScalarType::Float8_e4m3fnuz:
      return 1;
    case MoriScalarType::Int32:
    case MoriScalarType::UInt32:
      return 4;
    case MoriScalarType::UInt64:
      return 8;
    default:
      return 0;
  }
}

inline hipDataType MoriScalarToHipDataType(MoriScalarType t) {
  switch (t) {
    case MoriScalarType::Float32:
      return HIP_R_32F;
    case MoriScalarType::BFloat16:
      return HIP_R_16BF;
    case MoriScalarType::Float8_e4m3fnuz:
      return HIP_R_8F_E4M3_FNUZ;
    case MoriScalarType::Int32:
    case MoriScalarType::UInt32:
    case MoriScalarType::UInt64:
      return HIP_R_32F;  // Not used for integer tensors in kernels
    default:
      return HIP_R_32F;
  }
}

}  // namespace mori
