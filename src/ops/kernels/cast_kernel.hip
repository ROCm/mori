// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Standalone device kernel for float->fp4 cast.
// Compiled with: hipcc --genco --offload-arch=<arch>
// Loaded at runtime via hipModuleLoad / hipModuleLaunchKernel.

#include "mori/core/core.hpp"
#include "mori/utils/data_types.hpp"

using namespace mori;
using namespace mori::core;

extern "C" __global__
void mori_cast_float_to_fp4(float* src, mori_fp4_e2m1* dst, size_t nelems) {
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;

  constexpr int srcVecBytes = 16;
  constexpr int vecSize = std::min(static_cast<int>(srcVecBytes / sizeof(float)), srcVecBytes * 2);
  constexpr int dstVecBytes = vecSize / 2;

  using SrcLoadType = typename VecTypeSelector<srcVecBytes>::dataType;
  using DstStoreType = typename VecTypeSelector<dstVecBytes>::dataType;
  using SrcVecType = typename VecTypeAdaptor<float, vecSize>::dataType;
  using DstVecType = typename VecTypeAdaptor<mori_fp4_e2m1, vecSize>::dataType;

  for (int i = globalThdId; i < static_cast<int>(nelems / vecSize); i += globalThdNum) {
    SrcLoadType srcVec = load<srcVecBytes>(reinterpret_cast<SrcLoadType*>(src) + i);
    DstVecType dstVec(reinterpret_cast<SrcVecType*>(&srcVec)[0]);
    store<dstVecBytes>(reinterpret_cast<DstStoreType*>(dst) + i,
                       reinterpret_cast<DstStoreType*>(&dstVec)[0]);
  }
}
