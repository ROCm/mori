#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <torch/torch.h>

namespace mori {

template <typename T>
inline torch::Dtype GetTorchDataType() {
  if constexpr (std::is_same_v<T, float>) {
    return torch::kFloat32;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return torch::kUInt32;
  } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
    return torch::kBFloat16;
  } else if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz>) {
    return torch::kFloat8_e4m3fnuz;
  } else {
    static_assert(false, "Unsupported data type");
  }
}

}  // namespace mori