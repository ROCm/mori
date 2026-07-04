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
#include "mori/collective/inter_node/kernels/all_gather.hpp"

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

#define CCL_WRAP_ALLGATHER_PARAM_CONTIGUOUS(kernel, suffix, T)               \
  extern "C" __global__ void kernel##_##suffix(CclAllgatherArgs<T> args) {  \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.srcMemObj,      \
                     args.dstMemObj, args.flagsMemObj, args.elementCount,   \
                     args.dstBaseOffset, args.flagVal, args.splitSizes,     \
                     args.splitOffsets, args.splitCount);                   \
  }

#define CCL_WRAP_ALLGATHER_PARAM_CONTIGUOUS_ASYNC_PUT(kernel, suffix, T)     \
  extern "C" __global__ void kernel##_##suffix(CclAllgatherArgs<T> args) {  \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.srcMemObj,      \
                     args.dstMemObj, args.flagsMemObj, args.elementCount,   \
                     args.dstBaseOffset, args.splitSizes, args.splitOffsets, \
                     args.splitCount);                                      \
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
CCL_WRAP_ALLGATHER_PARAM_CONTIGUOUS(OneShotAllGatherSdmaParamContiguousKernel, u32, uint32_t)
CCL_WRAP_ALLGATHER_PARAM_CONTIGUOUS_ASYNC_PUT(OneShotAllGatherSdmaParamContiguousAsyncPutKernel,
                                              u32, uint32_t)

// Sub-group intra-node SDMA AllGather. Type-agnostic byte move on
// u32 lanes; the Python layer rounds the byte count up to u32 elements.
extern "C" __global__ void OneShotAllGatherSdmaSubGroupKernel_u32(
    CclAllgatherSubGroupArgs<uint32_t> args) {
  OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
      args.myPe, args.npes, args.groupSize, args.groupPos, args.peBase, args.peStride, args.input,
      args.dstMemObj, args.flagsMemObj, args.elementCount, args.dstBaseOffset,
      args.dstSlotStrideBytes, args.flagVal);
}

// Fused hierarchical param-contiguous SubGroup gather (ONE launch): loops node
// blocks + param splits internally, killing the N_nodes*N_params launch overhead.
extern "C" __global__ void OneShotAllGatherSdmaSubGroupParamContiguousKernel_u32(
    CclAllgatherSubGroupParamContiguousArgs<uint32_t> args) {
  OneShotAllGatherSdmaSubGroupParamContiguousKernel_body<uint32_t>(
      args.myPe, args.npes, args.groupSize, args.groupPos, args.peBase, args.peStride,
      args.numBlocks, args.firstBlock, args.input, args.dstMemObj, args.flagsMemObj,
      args.blockStrideElems, args.worldSize, args.dstBaseOffset, args.flagVal, args.splitSizes,
      args.splitOffsets, args.splitCount);
}

extern "C" __global__ void OneShotAllGatherSdmaAsyncWaitKernel_u32(
    CclAllgatherArgs<uint32_t> args) {
  OneShotAllGatherSdmaAsyncWaitKernel_body(args.myPe, args.npes, args.dstMemObj,
                                           args.flagsMemObj, args.flagVal);
}

// Sub-group intra-node SDMA broadcast. Type-
// agnostic byte move on u32 lanes; the Python layer rounds the byte count up to
// u32 elements. One root (group position 0) fans the full buffer to all members.
extern "C" __global__ void OneShotBroadcastSdmaSubGroupKernel_u32(
    CclBroadcastSubGroupArgs<uint32_t> args) {
  OneShotBroadcastSdmaSubGroupKernel_body<uint32_t>(
      args.myPe, args.groupSize, args.groupPos, args.peBase, args.peStride, args.input,
      args.dstMemObj, args.flagsMemObj, args.elementCount, args.dstBaseOffset, args.flagVal);
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

// ============================================================================
// Inter-node RDMA ring AllGather
// ============================================================================
// Byte-move ring: ``chunkBytes`` per PE, ``npes-1`` rounds over the shmem
// transport. Type-agnostic (raw bytes), so a single u32 entry serves all
// dtypes (the Python layer rounds the byte count up to u32 lanes).
// CU-domain ring finish copy-OUT: copy the gathered ring buffer to the user
// output from the COMPUTE-UNIT coherence domain (vs a copy-engine
// hipMemcpyAsync D2D). The inter-node ring receiver lands every remote chunk
// via RDMA into ``src`` (this PE's ring buffer) and the ring kernel's
// system-scope acquire + __threadfence_system make those bytes visible to the
// CUs on kernel exit. A subsequent copy-ENGINE (SDMA) D2D read is a SEPARATE
// hardware agent whose view is NOT ordered by that CU fence, so under tight
// FSDP overlap it can drain STALE remote-half ring bytes (the MI355
// receiver/copy-out residual after the sender-side quiet fix). A CU copy reads
// the same coherent HBM the ring kernel fenced -> closes the residual
// on-device, no host sync, keeps the ring<->gather overlap.
extern "C" __global__ void RingFinishCopyKernel_u32(
    uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, uint64_t n) {
  uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = idx; i < n; i += stride) {
    dst[i] = src[i];
  }
}

extern "C" __global__ void InterNodeRingAllGatherKernel_u32(CclInterNodeRingArgs args) {
  // Sub-group ring over {peBase, peBase+peStride, ...}; the flat whole-world
  // ring is the special case peBase=0, peStride=1, ringSize=npes, ringPos=myPe.
  AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.peBase, args.peStride,
                                  args.memObj, args.flagsObj, args.chunkBytes, args.numQp,
                                  /*numBlocksOverride=*/-1, /*bidOverride=*/-1,
                                  /*usePutSignal=*/args.usePutSignal != 0);
}

// ============================================================================
// Fused inter-node ring + intra-node LOCAL-block SDMA gather
// ============================================================================
// One launch overlaps Phase A (RDMA ring, NIC) with the Phase-B gather of THIS
// node's own block (SDMA, XGMI) -- the only half of Phase B that does NOT depend
// on the ring (each local rank's own shard is present from the start). Blocks
// [0, ringBlocks) run the ring (with explicit numBlocks=ringBlocks / bid so the
// RCCL-style multi-block channel logic still tiles each chunk exactly); the
// remaining block runs the local SDMA gather as a single self-contained block
// (blockLocal=true). The ring and the gather touch DISJOINT symm objects and
// flag regions, so the two halves never alias -- the union is byte-identical to
// running them as two serial kernels, only now concurrent. This is the fused
// recv+reassemble lever this work proved reaches/beats RCCL @>=32MiB, adopted here
//. NOTE: this fuses the ring with the LOCAL block only; the
// REMOTE-block gather (which DOES depend on the ring) still follows as a
// separate launch -- a later increment may fuse that tail.
extern "C" __global__ void FusedRingLocalGatherKernel_u32(CclFusedRingLocalGatherArgs args) {
  if (blockIdx.x < static_cast<unsigned>(args.ringBlocks)) {
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/args.ringBlocks,
                                    /*bidOverride=*/static_cast<int>(blockIdx.x));
  } else {
    OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
        args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
        args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
        args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true);
  }
}
