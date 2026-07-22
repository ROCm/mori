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
      args.dstSlotStrideBytes, args.flagVal, /*blockLocal=*/false, args.flagBase);
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
// output from the compute-unit coherence domain (vs a copy-engine
// hipMemcpyAsync D2D). The inter-node ring receiver lands every remote chunk
// via RDMA into ``src`` (this PE's ring buffer) and the ring kernel's
// system-scope acquire + __threadfence_system make those bytes visible to the
// CUs on kernel exit. A subsequent copy-engine (SDMA) D2D read is a separate
// hardware agent whose view is not ordered by that CU fence, so under tight
// overlap it can read stale remote-half ring bytes. A CU copy reads the same
// coherent HBM the ring kernel fenced, closing the gap on-device with no host
// sync while preserving the ring/gather overlap.
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
                                  /*chunkReadyFlags=*/nullptr,
                                  /*deepPipe=*/1,
                                  /*deepPipeQuiet=*/0,
                                  /*useWriteFence=*/args.useWriteFence != 0);
}

// ============================================================================
// Fused inter-node ring + intra-node LOCAL-block SDMA gather
// ============================================================================
// One launch overlaps Phase A (RDMA ring, NIC) with the Phase-B gather of this
// node's own block (SDMA, XGMI) -- the only half of Phase B that does not depend
// on the ring (each local rank's own shard is present from the start). Blocks
// [0, ringBlocks) run the ring (with explicit numBlocks=ringBlocks / bid so the
// multi-block channel logic still tiles each chunk exactly); the remaining block
// runs the local SDMA gather as a single self-contained block (blockLocal=true).
// The ring and the gather touch disjoint symm objects and flag regions, so the two
// halves never alias -- the union is byte-identical to running them as two serial
// kernels, only now concurrent. This fuses the ring with the local block only; the
// remote-block gather (which does depend on the ring) still follows as a separate
// launch.
extern "C" __global__ void FusedRingLocalGatherKernel_u32(CclFusedRingLocalGatherArgs args) {
  if (blockIdx.x < static_cast<unsigned>(args.ringBlocks)) {
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/args.ringBlocks,
                                    /*bidOverride=*/static_cast<int>(blockIdx.x),
                                    /*chunkReadyFlags=*/nullptr);
  } else {
    OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
        args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
        args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
        args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/0);
  }
}

// Remote reassembly WORKER (the dedicated reassembly CTAs). Worker ``j`` (of
// ``stride`` total workers) owns ring channels f with (f % stride == j). For each such
// channel it waits ONLY on that channel's landing flag (chunkReadyFlags[f]) and,
// the instant it lands, SDMA-pushes THAT channel's 16B-aligned tile of every
// remote block from this PE's ring buffer straight into the registered output on
// SDMA queue ``qId`` -- overlapping the ring channels still crossing the NIC. The
// tile partition mirrors the ring's multi-block tiling exactly. PUSH-ONLY (no
// cross-rank wait) so the workers never dead-lock; each worker uses a DISTINCT
// qId so their per-queue signal counters never race.
// ``partition`` = number of disjoint 16B-aligned byte sub-ranges the chunk is
// split into for the intra-node SDMA reassembly push. Worker ``j`` (stride
// ``stride``) owns parts f=j, j+stride, ... . ``spatial`` selects the landing
// flag each part waits on: for the temporal deep-pipe (spatial=false) part f
// waits on chunkReadyFlags[f] (its own sub-chunk landed); for the spatial split
// at a single-channel ring (spatial=true) every part waits on the one landing
// flag chunkReadyFlags[0] (the whole chunk landed), so multiple SDMA engines
// reassemble disjoint sub-ranges of that one landed chunk concurrently --
// shrinking the serial intra-reassembly tail without changing the inter-node
// ring (no new inter race). Flag slot region is per-part (flagBase uses ``f``)
// so concurrent parts never race a shared slot.
__device__ inline void FusedRemoteReassembleWorker(
    const CclFusedRingRemoteGatherArgs& args, int partition, int j, int stride, int qId,
    bool spatial) {
  const size_t kAlign = 16;
  const size_t nUnits = (args.chunkBytes + kAlign - 1) / kAlign;
  const size_t unitsPerChan =
      (nUnits + static_cast<size_t>(partition) - 1) / static_cast<size_t>(partition);
  const size_t sliceElems = args.chunkBytes / sizeof(uint32_t);
  const int numRemote = args.numNodes - 1;
  uint32_t* ringBase = reinterpret_cast<uint32_t*>(args.ringMemObj->localPtr);
  // Intra reassembly deep-SQ: when reasmDeepSq!=0 run the owned channels in two
  // passes over the same queue qId -- pass 0 waits each channel's landing flag
  // then submits its copy with no drain (deepSqPhase=1), keeping the SDMA SQ
  // continuously fed across channels; pass 1 replays them drain+flag only
  // (deepSqPhase=2), where a single quiet to the accumulated expected count covers
  // every back-to-back submit (SQ FIFO, signal monotonic). The landing wait stays
  // in pass 0 (never submit before the ring bytes land); the output flag fires only
  // after the pass-1 drain => never precedes its bytes (bit-exact). nPass==1 =>
  // deepSqPhase=0 = the byte-identical single-shot path.
  const int nPass = (args.reasmDeepSq != 0) ? 2 : 1;
  for (int pass = 0; pass < nPass; ++pass) {
    const int deepSqPhase = (nPass == 1) ? 0 : (pass == 0 ? 1 : 2);
    for (int f = j; f < partition; f += stride) {
      size_t startUnit = static_cast<size_t>(f) * unitsPerChan;
      size_t endUnit = startUnit + unitsPerChan;
      if (endUnit > nUnits) endUnit = nUnits;
      const size_t subOff = startUnit * kAlign;
      size_t subEnd = endUnit * kAlign;
      if (subEnd > args.chunkBytes) subEnd = args.chunkBytes;
      const size_t subBytes = (subOff < subEnd) ? (subEnd - subOff) : 0;
      // Landing-flag wait only guards the SUBMIT (deep-SQ pass 0) or the single-shot
      // path; the deep-SQ drain pass (deepSqPhase==2) already waited in pass 0.
      if (deepSqPhase != 2) {
        if (threadIdx.x == 0) {
          int spin = 0;
          const int flagIdx = spatial ? 0 : f;
          // Landing flags publish 1 and are host-zeroed each op.
          while (mori::core::AtomicLoadSeqCstSystem(args.chunkReadyFlags + flagIdx) <
                 static_cast<uint64_t>(1)) {
            if (++spin > 100000000) break;  // bounded fallback
          }
          __threadfence_system();
        }
        __syncthreads();
      }
      if (subBytes == 0) continue;
      const size_t subElems = subBytes / sizeof(uint32_t);  // 16B-aligned => exact
      int mi = 0;
      for (int m = 0; m < args.numNodes; ++m) {
        if (m == args.nodeId) continue;  // local block handled by the bx==rb CTA gather
        uint32_t* src =
            ringBase + static_cast<size_t>(m) * sliceElems + (subOff / sizeof(uint32_t));
        const size_t dstBase =
            static_cast<size_t>(m) * static_cast<size_t>(args.groupSize) * sliceElems *
                sizeof(uint32_t) +
            subOff;
        const size_t flagBase = static_cast<size_t>(args.groupSize) +
                                (static_cast<size_t>(f) * numRemote + mi) *
                                    static_cast<size_t>(args.groupSize);
        OneShotSubGroupPushOnly_body<uint32_t>(
            args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, src, args.gDstMemObj,
            args.gFlagsObj, subElems, dstBase, /*dstSlotStrideBytes=*/args.chunkBytes,
            args.gFlagVal, flagBase, /*qId=*/qId, /*deepSqPhase=*/deepSqPhase,
            /*pushPeerLo=*/0, /*pushPeerHi=*/-1, /*pushFlag=*/1);
        ++mi;
      }
    }
  }
}

// Pipelined ring + remote-block reassembly (see CclFusedRingRemoteGather
// Args). Grid = 2*ringBlocks + 1. Blocks [0,ringBlocks): ring channels, each
// publishing chunkReadyFlags[bid] on landing. Block [ringBlocks]: the ring-
// independent LOCAL block gather (this PE's own input). Blocks
// (ringBlocks, 2*ringBlocks]: remote reassembly block j = blockIdx.x-ringBlocks-1
// spins on chunkReadyFlags[j] then SDMA-pushes sub-range j of every remote block
// from THIS PE's ring buffer (slot m == node m's chunk, ring order) straight into
// the registered output -- overlapping ring channel j+1 still on the NIC. The
// sub-range partition mirrors the ring's multi-block 16B-aligned tiling exactly so
// chunkReadyFlags[j] guards precisely the bytes reassembled here. Each (j, remote
// m) gather uses a disjoint flagBase so the concurrent reassembly blocks never
// race on the shared subgroup flag slots.
extern "C" __global__ void FusedRingRemoteGatherKernel_u32(CclFusedRingRemoteGatherArgs args) {
  const int rb = args.ringBlocks;
  // Deep-SQ temporal pipeline: when deepPipe>1 (only on rb==1) the single ring
  // channel splits the chunk into ``pipe`` temporal sub-chunks each with its own
  // landing flag, so the reassembly workers + completion reader iterate over
  // ``pipe`` sub-chunk flags instead of ``rb`` spatial channel flags. deepPipe<=1
  // => pipe==rb (byte-identical default path).
  const int pipe = args.deepPipe > 1 ? args.deepPipe : rb;
  const int reasm = args.reassemblyBlocks > 0 ? args.reassemblyBlocks : rb;
  const int effReasm = reasm;
  // Spatial reassembly split: on the single-channel giant-AG ring (rb==1) with no
  // temporal deep-pipe, split the one landed chunk into ``effReasm`` disjoint byte
  // sub-ranges so effReasm SDMA engines reassemble it concurrently -- shrinks the
  // serial intra-reassembly tail without touching the inter ring (no new inter
  // race, unlike rb>1 which grows the RDMA fill 1:1). Each part waits the single
  // landing flag chunkReadyFlags[0]. reasm>1 requires sdmaNumQueue>=reasm+1
  // (Python clamp).
  const bool spatialSplit = (rb == 1 && args.deepPipe <= 1 && effReasm > 1);
  // Partition count for the reassembly byte split: temporal pipe channels, or the
  // spatial effReasm sub-ranges when the spatial split engages.
  const int partition = spatialSplit ? effReasm : pipe;
  const unsigned bx = blockIdx.x;
  if (bx < static_cast<unsigned>(rb)) {
    // Ring channel bx: publishes chunkReadyFlags[bx] on landing.
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/rb, /*bidOverride=*/static_cast<int>(bx),
                                    args.chunkReadyFlags,
                                    /*deepPipe=*/args.deepPipe,
                                    /*deepPipeQuiet=*/args.deepPipeQuiet,
                                    /*useWriteFence=*/args.useWriteFence != 0);
  } else if (bx == static_cast<unsigned>(rb)) {
    // Local block (m == nodeId): ring-independent, reads this PE's own input.
    // Collective gather on flag slots [0, G) -- a SINGLE collective (never nested),
    // so it cannot dead-lock. BUT under DEEP_PIPE>=8 the concurrent ring sub-chunk
    // CTAs + reassembly workers can starve the cross-rank flag AMO this coupled
    // per-slot wait spins on -> the "Slow wait for sub-group pos" stall.
    // LOCAL-BLOCK PUSH-ONLY (localPushOnly): push this rank's own column with NO
    // coupled wait (queue 0, distinct from the reasm workers' queues 1..reasm), and
    // let the completion reader below drain the local flag slots [0,G) too. Same
    // pushes + same flags => byte-identical output, deadlock-free at any depth.
    if (args.localPushOnly != 0) {
      OneShotSubGroupPushOnly_body<uint32_t>(
          args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, args.gInput,
          args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*flagBase=*/0, /*qId=*/0,
          /*deepSqPhase=*/0,
          /*pushPeerLo=*/0,
          /*pushPeerHi=*/-1);
    } else if (args.crownRing != 0) {
      // Route the local-block gather through the intra-node nearest-neighbor
      // pipelined ring instead of the flat G-way push. The flat push makes every
      // receiver take a G-1 concurrent-write incast; a one-flow-per-link relay avoids
      // it. The ring self-completes on its own high-base flag region
      // [G + reasmTotal + G, ...) so it never collides with the reassembly flags
      // [G, G+reasmTotal) or the local block flags [0, G). Same final slot layout and
      // bytes, only the copy schedule differs. crownRing default 0 => this branch never
      // taken => byte-identical to the flat push.
      const int _cr_numRemote = args.numNodes - 1;
      const size_t _cr_reasmTotal = static_cast<size_t>(partition) *
                                    static_cast<size_t>(_cr_numRemote > 0 ? _cr_numRemote : 0) *
                                    static_cast<size_t>(args.groupSize);
      const size_t _cr_flagBase =
          static_cast<size_t>(args.groupSize) + _cr_reasmTotal + static_cast<size_t>(args.groupSize);
      OneShotAllGatherSdmaSubGroupRingKernel_body<uint32_t>(
          args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
          args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/_cr_flagBase,
          /*fencedFlag=*/(args.crownRing & 8191));
      // fencedFlag bit meanings (see oneshot_sdma_kernel.hpp for details):
      //   bit2=2x4 half-ring; bit3=fused overlap; bit4=parallel phase2;
      //   bit5=dest-spread phase2; bit6=dest-spread phase1;
      //   bit7=multi-warp dest-spread submission; bit8=concurrent phase1/phase2;
      //   bit9=flat multi-warp broadcast; bit10=fused self-fill;
      //   bit11=batch self-fill; bit12=overlap self-fill.
    } else {
      // Default local-block flat crown gather (localPushOnly OFF).
      OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
          args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
          args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/0,
          /*pushPeerLo=*/0,
          /*pushPeerHi=*/-1);
    }
    // COMPLETION READER: the remote reassembly blocks are PUSH-ONLY (they never
    // wait on a peer) so this CTA does the single cross-rank completion wait for
    // the WHOLE remote reassembly. It spins until every (block j, remote node m,
    // sender g) has published its flag -- guaranteed to finish because every
    // rank's push blocks always make progress (no circular dependency). Flag slot
    // for (j, mIdx, senderG) == G + (j*numRemote + mIdx)*G + senderG, disjoint
    // from the local block's [0, G).
    const int numRemote = args.numNodes - 1;
    if ((numRemote > 0 || args.localPushOnly != 0) && args.chunkReadyFlags != nullptr) {
      uint64_t* rflags = reinterpret_cast<uint64_t*>(args.gFlagsObj->localPtr);
      const int G = args.groupSize;
      // Parallel completion reader: the flag slots (channel f, remote node mi,
      // sender g) occupy the contiguous region [G, G + rb*numRemote*G):
      // idx = G + (f*numRemote + mi)*G + sg with (f,mi,sg) enumerating that range
      // densely. A serial thread-0 spin over all rb*numRemote*G slots would grow
      // the whole-reassembly completion latency linearly with the ring-channel
      // count, collapsing the pipeline's overlap at rb>1. Fan the wait across the
      // CTA's threads (each thread polls a strided subset) so the completion tail
      // is O(total / blockDim) instead of O(total). Bit-exact: same slots, same
      // threshold, just concurrent; a single CTA-wide fence + barrier still orders
      // every landed byte before the finish.
      const size_t reasmTotal = static_cast<size_t>(partition) * static_cast<size_t>(numRemote) *
                                static_cast<size_t>(G);
      // Reassembly flags live in [G, G+reasmTotal). LOCAL-BLOCK PUSH-ONLY also folds
      // the local block's own flag slots [0,G) into this single completion wait
      // (startIdx=0); when OFF the local block did its own coupled wait so this
      // reader starts at G -- byte-identical range.
      const size_t startIdx = (args.localPushOnly != 0) ? 0 : static_cast<size_t>(G);
      const size_t endIdx = static_cast<size_t>(G) + reasmTotal;
      for (size_t idx = startIdx + threadIdx.x; idx < endIdx; idx += blockDim.x) {
        int spin = 0;
        while (mori::core::AtomicLoadSeqCstSystem(rflags + idx) < args.gFlagVal) {
          if (++spin > 100000000) break;  // bounded fallback
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  } else {
    // Remote reassembly block j in [0, reasm): PIPELINED over ring channels.
    // Block j owns the ring channels f with (f % reasm == j). For EACH such
    // channel it waits ONLY on that channel's landing flag (chunkReadyFlags[f])
    // and, the instant it lands, SDMA-pushes THAT channel's exact 16B-aligned
    // tile of every remote block from this PE's ring buffer straight into the
    // registered output -- so channel f's XGMI reassembly overlaps ring channels
    // f+reasm, f+2*reasm, ... still crossing the NIC. This is the true two-phase
    // (NIC<->XGMI) pipeline: waiting for all ring flags before pushing anything
    // would collapse the overlap. The tile partition mirrors the ring's multiBlock
    // tiling (numBlocks==rb,
    // 16B units) so chunkReadyFlags[f] guards precisely the bytes pushed here.
    // PUSH-ONLY (no cross-rank wait) => the blocks never dead-lock. Each block
    // uses SDMA queue qId=j+1 (distinct per block; local CTA uses queue 0), and
    // reuses that one queue serially across its channels (never raced).
    const int j = static_cast<int>(bx) - rb - 1;
    if (j < 0 || j >= reasm) return;
    // Worker j of effReasm total workers (stride effReasm); qId=j+1 (1..reasm) is
    // distinct from the local CTA's queue 0.
    FusedRemoteReassembleWorker(args, partition, /*j=*/j, /*stride=*/effReasm, /*qId=*/j + 1,
                                spatialSplit);
  }
}
