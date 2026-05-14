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

// CCL collective kernels — JIT compiled via hipcc --genco

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define MORI_SHMEM_NO_STATIC_INIT
#include "mori/shmem/shmem.hpp"

#include "mori/collective/all2all/oneshot_all2all_sdma_async_kernel.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_kernel.hpp"
#include "mori/collective/allgather/oneshot_sdma_async_kernel.hpp"
#include "mori/collective/allgather/oneshot_sdma_kernel.hpp"
#include "mori/collective/allreduce/twoshot_sdma_async_kernel.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"

#include "mori/collective/ccl_kernel_args.hpp"

// Each .hsaco needs its own globalGpuStates, initialized via ShmemModuleInit.
namespace mori {
namespace shmem {
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;
}  // namespace shmem
}  // namespace mori

using namespace mori::collective;

// ============================================================================
// Wrapper macros
// ============================================================================

// All2all: (myPe, npes, input, inputTransitMemObj, outputTransitMemObj,
//           flagsMemObj, elementCount)
#define CCL_WRAP_ALL2ALL(kernel, suffix, T)                                  \
  extern "C" __global__ void kernel##_##suffix(CclAll2allArgs<T> args) {    \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.inputTransitMemObj, \
                     args.outputTransitMemObj, args.flagsMemObj, args.elementCount); \
  }

// AllGather sync: all fields including dstBaseOffset and flagVal
#define CCL_WRAP_ALLGATHER(kernel, suffix, T)                                \
  extern "C" __global__ void kernel##_##suffix(CclAllgatherArgs<T> args) {  \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.srcMemObj,      \
                     args.dstMemObj, args.flagsMemObj, args.elementCount,   \
                     args.dstBaseOffset, args.flagVal);                     \
  }

// AllGather async put: no flagVal parameter
#define CCL_WRAP_ALLGATHER_ASYNC_PUT(kernel, suffix, T)                      \
  extern "C" __global__ void kernel##_##suffix(CclAllgatherArgs<T> args) {  \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.srcMemObj,      \
                     args.dstMemObj, args.flagsMemObj, args.elementCount,   \
                     args.dstBaseOffset);                                   \
  }

// AllReduce reduce-scatter / fused: (myPe, npes, input, dstMemObj,
//                                    flagsMemObj, barrier, elementCount)
#define CCL_WRAP_ALLREDUCE_RS(kernel, suffix, T)                              \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.dstMemObj,       \
                     args.flagsMemObj, args.barrier, args.elementCount);     \
  }

// AllReduce allgather / async-put: (myPe, npes, dstMemObj, flagsMemObj,
//                                   barrier, elementCount)
#define CCL_WRAP_ALLREDUCE_AG(kernel, suffix, T)                              \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, args.dstMemObj, args.flagsMemObj, \
                     args.barrier, args.elementCount);                       \
  }

// AllReduce async wait (non-template): (myPe, npes, flagsMemObj, barrier,
//                                       elementCount)
#define CCL_WRAP_ALLREDUCE_AG_WAIT(kernel, suffix, T)                         \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body(args.myPe, args.npes, args.flagsMemObj, args.barrier,      \
                  args.elementCount);                                        \
  }

// Async RS SDMA put: (myPe, npes, input, dstMemObj, elementCount)
#define CCL_WRAP_ALLREDUCE_RS_PUT(kernel, suffix, T)                          \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, const_cast<T*>(args.input),       \
                     args.dstMemObj, args.elementCount);                     \
  }

// Async RS local reduce: (gathered, input, elementCount, myPe, npes)
#define CCL_WRAP_ALLREDUCE_LOCAL_REDUCE(kernel, suffix, T)                    \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(reinterpret_cast<T*>(args.dstMemObj->localPtr),          \
                     args.input, args.elementCount, args.myPe, args.npes);  \
  }

// Async AG reduced SDMA put: (myPe, npes, dstMemObj, elementCount)
#define CCL_WRAP_ALLREDUCE_AG_REDUCED_PUT(kernel, suffix, T)                  \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, args.dstMemObj, args.elementCount); \
  }

// Expand a WRAP macro across all allreduce types.
#define CCL_ALLREDUCE_ALL_TYPES(WRAP_MACRO, kernel) \
  WRAP_MACRO(kernel, u32, uint32_t)                 \
  WRAP_MACRO(kernel, i32, int32_t)                  \
  WRAP_MACRO(kernel, f32, float)                    \
  WRAP_MACRO(kernel, f16, __half)                   \
  WRAP_MACRO(kernel, bf16, hip_bfloat16)

// ============================================================================
// All2all instantiations (uint32_t)
// ============================================================================

CCL_WRAP_ALL2ALL(OneShotAll2allSdmaKernel, u32, uint32_t)
CCL_WRAP_ALL2ALL(OneShotAll2allSdmaAsyncPutKernel, u32, uint32_t)

extern "C" __global__ void OneShotAll2allSdmaAsyncWaitKernel_u32(
    CclAll2allArgs<uint32_t> args) {
  OneShotAll2allSdmaAsyncWaitKernel_body(args.myPe, args.npes,
                                         args.outputTransitMemObj, args.flagsMemObj);
}

// ============================================================================
// AllGather instantiations (uint32_t)
// ============================================================================

CCL_WRAP_ALLGATHER(OneShotAllGatherSdmaKernel, u32, uint32_t)
CCL_WRAP_ALLGATHER_ASYNC_PUT(OneShotAllGatherSdmaAsyncPutKernel, u32, uint32_t)

extern "C" __global__ void OneShotAllGatherSdmaAsyncWaitKernel_u32(
    CclAllgatherArgs<uint32_t> args) {
  OneShotAllGatherSdmaAsyncWaitKernel_body(args.myPe, args.npes, args.dstMemObj,
                                           args.flagsMemObj, args.flagVal);
}

// ============================================================================
// AllReduce instantiations (uint32_t, int32_t, float, __half, hip_bfloat16)
// ============================================================================

// --- SdmaReduceScatterKernel ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_RS, SdmaReduceScatterKernel)

// --- AllGatherSdmaKernel ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG, AllGatherSdmaKernel)

// --- ReduceScatterSdmaPutKernel (async scatter, no wait/reduce) ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_RS_PUT, ReduceScatterSdmaPutKernel)

// --- ReduceScatterLocalReduceKernel (local reduce after async scatter) ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_LOCAL_REDUCE, ReduceScatterLocalReduceKernel)

// --- AllGatherReducedSdmaPutKernel (SDMA put of reduced shard) ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG_REDUCED_PUT, AllGatherReducedSdmaPutKernel)

// --- AllGatherAsyncPutKernel ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG, AllGatherAsyncPutKernel)

// --- AllGatherAsyncWaitKernel (non-template) ---
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG_WAIT, AllGatherAsyncWaitKernel)
