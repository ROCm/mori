
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
#include "mori/ops/cast.hpp"

#include <hip/hip_runtime.h>

#include "mori/core/core.hpp"
#include "mori/utils/data_types.hpp"

namespace {

using namespace mori;
using namespace mori::core;

template <typename SrcT, typename DstT>
__global__ void CastKernel(SrcT* src, DstT* dst, size_t nelems);

template <>
__global__ void CastKernel(float* src, mori::mori_fp4_e2m1* dst, size_t nelems) {
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;

  constexpr int srcVecBytes = 16;
  constexpr int vecSize = std::min(static_cast<int>(srcVecBytes / sizeof(float)), srcVecBytes * 2);
  constexpr int dstVecBytes = vecSize / 2;

  using SrcLoadType = typename VecTypeSelector<srcVecBytes>::dataType;
  using DstStoreType = typename VecTypeSelector<dstVecBytes>::dataType;

  using SrcVecType = typename VecTypeAdaptor<float, vecSize>::dataType;
  using DstVecType = typename VecTypeAdaptor<mori::mori_fp4_e2m1, vecSize>::dataType;

  for (int i = globalThdId; i < (nelems / vecSize); i += globalThdNum) {
    SrcLoadType srcVec = load<srcVecBytes>(reinterpret_cast<SrcLoadType*>(src) + i);
    DstVecType dstVec(reinterpret_cast<SrcVecType*>(&srcVec)[0]);
    store<dstVecBytes>(reinterpret_cast<DstStoreType*>(dst) + i,
                       reinterpret_cast<DstStoreType*>(&dstVec)[0]);
  }
}

}  // namespace

namespace mori {

template <typename SrcT, typename DstT>
void LaunchCast(SrcT* src, DstT* dst, size_t nelems, hipStream_t stream) {
  CastKernel<<<1, 1, 0, stream>>>(src, dst, nelems);
}

template void LaunchCast<float, mori::mori_fp4_e2m1>(float* src, mori::mori_fp4_e2m1* dst,
                                                     size_t nelems, hipStream_t stream);
}  // namespace mori
