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
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/internode.hpp"
#include "src/ops/dispatch_combine/internode_v1.hpp"
#include "src/ops/dispatch_combine/intranode.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

/* ---------------------------------------------------------------------------------------------- */
/*                                  EpDispatchLowLatencyAsyncSend                                 */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__global__ void EpDispatchLowLatencyAsyncSend(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Template Specialization                                    */
/* ---------------------------------------------------------------------------------------------- */
template __global__ void EpDispatchLowLatencyAsyncSend<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpDispatchLowLatencyAsyncSend<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpDispatchLowLatencyAsyncSend<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpDispatchLowLatencyAsyncSend<float>(EpDispatchCombineArgs<float> args);

}  // namespace moe
}  // namespace mori
