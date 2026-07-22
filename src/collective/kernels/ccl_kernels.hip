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

// Sub-group intra-node SDMA AllGather (type-agnostic byte move on u32 lanes).
extern "C" __global__ void OneShotAllGatherSdmaSubGroupKernel_u32(
    CclAllgatherSubGroupArgs<uint32_t> args) {
  OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
      args.myPe, args.npes, args.groupSize, args.groupPos, args.peBase, args.peStride, args.input,
      args.dstMemObj, args.flagsMemObj, args.elementCount, args.dstBaseOffset,
      args.dstSlotStrideBytes, args.flagVal, /*blockLocal=*/false, args.flagBase);
}

// Fused hierarchical param-contiguous SubGroup gather: one launch loops node
// blocks + param splits internally (vs N_nodes*N_params launches).
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

// Sub-group intra-node SDMA broadcast: root (group pos 0) fans the full buffer
// to all members (type-agnostic byte move on u32 lanes).
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
// Byte-move ring: chunkBytes per PE, npes-1 rounds over the shmem transport
// (raw bytes, one u32 entry serves all dtypes).
// CU-domain ring finish copy-OUT: copy the gathered ring buffer to user output
// from the CU coherence domain. The ring kernel's system-scope acquire +
// __threadfence_system order the RDMA-landed bytes for CUs, but NOT for a
// copy-engine D2D (separate agent) which can read stale ring bytes; a CU copy
// reads the same fenced HBM, closing the gap on-device without host sync.
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
                                  /*deepPipeQuiet=*/0);
}

// ============================================================================
// Fused inter-node ring + intra-node LOCAL-block SDMA gather
// ============================================================================
// One launch overlaps the RDMA ring (NIC) with the local-block SDMA gather
// (XGMI, ring-independent). Blocks [0, ringBlocks) run the ring; the last block
// runs the local gather (blockLocal=true). Ring and gather touch disjoint symm
// objects + flag regions => byte-identical to running them serially. Remote-block
// gather (ring-dependent) still follows as a separate launch.
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

// Remote reassembly worker j (of stride workers) owns parts f=j, j+stride, ...
// of the chunk's `partition` 16B-aligned sub-ranges. Per part it waits its
// landing flag then SDMA-pushes that tile of every remote block into the output.
// PUSH-ONLY (no cross-rank wait) => deadlock-free; distinct qId per worker so
// signal counters never race; per-part flagBase (uses f) so parts never share a
// slot. spatial=false: part f waits chunkReadyFlags[f] (temporal deep-pipe).
// spatial=true (single-channel ring): every part waits chunkReadyFlags[0], so
// multiple SDMA engines reassemble the one landed chunk concurrently.
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
  // Intra reassembly deep-SQ (reasmDeepSq!=0): two passes over queue qId. Pass 0
  // waits each channel's landing flag then submits with no drain (deepSqPhase=1);
  // pass 1 drains+flags only (deepSqPhase=2), one quiet covering all back-to-back
  // submits (SQ FIFO, signal monotonic). Landing wait stays in pass 0; output flag
  // fires only after the pass-1 drain => never precedes its bytes (bit-exact).
  // nPass==1 => deepSqPhase=0, byte-identical single-shot path.
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
      // Landing wait guards only the SUBMIT (pass 0 / single-shot); the drain pass
      // (deepSqPhase==2) already waited in pass 0.
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

// Pipelined ring + remote-block reassembly. Grid = 2*ringBlocks + 1:
//   [0,ringBlocks)          ring channels, publish chunkReadyFlags[bid] on landing
//   [ringBlocks]            ring-independent local-block gather
//   (ringBlocks,2*ringBlocks]  reassembly block j spins chunkReadyFlags[j] then
//                           SDMA-pushes sub-range j of every remote block.
// Sub-range partition mirrors the ring's 16B-aligned tiling so chunkReadyFlags[j]
// guards exactly the reassembled bytes; disjoint flagBase per (j, remote m).
extern "C" __global__ void FusedRingRemoteGatherKernel_u32(CclFusedRingRemoteGatherArgs args) {
  const int rb = args.ringBlocks;
  // Deep-SQ temporal pipeline (deepPipe>1, rb==1 only): the single ring channel
  // splits the chunk into `pipe` temporal sub-chunks, each with its own landing
  // flag, so reassembly + completion iterate over pipe flags. deepPipe<=1 =>
  // pipe==rb (byte-identical default).
  const int pipe = args.deepPipe > 1 ? args.deepPipe : rb;
  const int reasm = args.reassemblyBlocks > 0 ? args.reassemblyBlocks : rb;
  const int effReasm = reasm;
  // Spatial reassembly split (rb==1, no deep-pipe): split the one landed chunk
  // into effReasm disjoint sub-ranges reassembled concurrently; each part waits
  // chunkReadyFlags[0]. reasm>1 requires sdmaNumQueue>=reasm+1 (Python clamp).
  const bool spatialSplit = (rb == 1 && args.deepPipe <= 1 && effReasm > 1);
  // Partition = temporal pipe channels, or spatial effReasm sub-ranges.
  const int partition = spatialSplit ? effReasm : pipe;
  const unsigned bx = blockIdx.x;
  if (bx < static_cast<unsigned>(rb)) {
    // Ring channel bx: publishes chunkReadyFlags[bx] on landing.
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/rb, /*bidOverride=*/static_cast<int>(bx),
                                    args.chunkReadyFlags,
                                    /*deepPipe=*/args.deepPipe,
                                    /*deepPipeQuiet=*/args.deepPipeQuiet);
  } else if (bx == static_cast<unsigned>(rb)) {
    // Local block (m == nodeId): ring-independent, reads this PE's own input.
    // Default: collective gather on flag slots [0, G). LOCAL-BLOCK PUSH-ONLY
    // (localPushOnly): push this rank's column with NO coupled wait (queue 0,
    // distinct from reasm queues 1..reasm) and let the completion reader drain
    // slots [0,G) -- same pushes + flags => byte-identical, deadlock-free at any
    // deep-pipe depth (avoids the coupled per-slot AMO stall under DEEP_PIPE>=8).
    if (args.localPushOnly != 0) {
      OneShotSubGroupPushOnly_body<uint32_t>(
          args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, args.gInput,
          args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*flagBase=*/0, /*qId=*/0,
          /*deepSqPhase=*/0,
          /*pushPeerLo=*/0,
          /*pushPeerHi=*/-1);
    } else if (args.crownRing != 0) {
      // crownRing: route the local-block gather through the intra-node
      // nearest-neighbor pipelined ring (one flow per link) instead of the flat
      // G-way push (G-1 incast). Ring self-completes on its own high-base flag
      // region [G+reasmTotal+G, ...), disjoint from reasm flags [G, G+reasmTotal)
      // and local flags [0, G). crownRing default 0 => byte-identical to flat push.
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
    // COMPLETION READER: reassembly blocks are PUSH-ONLY, so this CTA does the
    // single cross-rank completion wait for the whole remote reassembly, spinning
    // until every (j, remote m, sender g) flag is published. Flag slot
    // (j, mIdx, senderG) == G + (j*numRemote + mIdx)*G + senderG, disjoint from
    // the local block's [0, G).
    const int numRemote = args.numNodes - 1;
    if ((numRemote > 0 || args.localPushOnly != 0) && args.chunkReadyFlags != nullptr) {
      uint64_t* rflags = reinterpret_cast<uint64_t*>(args.gFlagsObj->localPtr);
      const int G = args.groupSize;
      // Parallel completion reader: flag slots (f, mi, sg) fill [G, G+rb*numRemote*G)
      // densely (idx = G + (f*numRemote + mi)*G + sg). Fan the wait across the CTA's
      // threads (strided subsets) instead of a serial thread-0 spin. Bit-exact:
      // same slots + threshold, and a CTA-wide fence + barrier still orders every
      // landed byte before the finish.
      const size_t reasmTotal = static_cast<size_t>(partition) * static_cast<size_t>(numRemote) *
                                static_cast<size_t>(G);
      // Reassembly flags live in [G, G+reasmTotal). localPushOnly folds the local
      // block's slots [0,G) into this wait (startIdx=0); else it did its own coupled
      // wait and this reader starts at G.
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
    // Remote reassembly block j in [0, reasm): pipelined over ring channels f with
    // (f % reasm == j). Per channel it waits chunkReadyFlags[f] then SDMA-pushes
    // that 16B-aligned tile of every remote block into the output, so channel f's
    // XGMI reassembly overlaps channels f+reasm, ... still on the NIC (two-phase
    // NIC<->XGMI pipeline). Tile partition mirrors the ring's multiBlock tiling
    // (numBlocks==rb) so chunkReadyFlags[f] guards exactly the pushed bytes.
    // PUSH-ONLY => deadlock-free; qId=j+1 (local CTA uses queue 0), reused serially.
    const int j = static_cast<int>(bx) - rb - 1;
    if (j < 0 || j >= reasm) return;
    // Worker j of effReasm workers (stride effReasm); qId=j+1 distinct from queue 0.
    FusedRemoteReassembleWorker(args, partition, /*j=*/j, /*stride=*/effReasm, /*qId=*/j + 1,
                                spatialSplit);
  }
}
