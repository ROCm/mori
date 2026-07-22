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

// Wrapper macros: expand a kernel body into an extern "C" __global__ entry per dtype.

#define CCL_WRAP_ALL2ALL(kernel, suffix, T)                                  \
  extern "C" __global__ void kernel##_##suffix(CclAll2allArgs<T> args) {    \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.inputTransitMemObj, \
                     args.outputTransitMemObj, args.flagsMemObj, args.elementCount); \
  }

#define CCL_WRAP_ALLGATHER(kernel, suffix, T)                                \
  extern "C" __global__ void kernel##_##suffix(CclAllgatherArgs<T> args) {  \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.srcMemObj,      \
                     args.dstMemObj, args.flagsMemObj, args.elementCount,   \
                     args.dstBaseOffset, args.flagVal);                     \
  }

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

#define CCL_WRAP_ALLREDUCE_RS(kernel, suffix, T)                              \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, args.input, args.dstMemObj,       \
                     args.flagsMemObj, args.barrier, args.elementCount);     \
  }

#define CCL_WRAP_ALLREDUCE_AG(kernel, suffix, T)                              \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, args.dstMemObj, args.flagsMemObj, \
                     args.barrier, args.elementCount);                       \
  }

// Non-template body (dtype-agnostic wait).
#define CCL_WRAP_ALLREDUCE_AG_WAIT(kernel, suffix, T)                         \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body(args.myPe, args.npes, args.flagsMemObj, args.barrier,      \
                  args.elementCount);                                        \
  }

#define CCL_WRAP_ALLREDUCE_RS_PUT(kernel, suffix, T)                          \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(args.myPe, args.npes, const_cast<T*>(args.input),       \
                     args.dstMemObj, args.elementCount);                     \
  }

#define CCL_WRAP_ALLREDUCE_LOCAL_REDUCE(kernel, suffix, T)                    \
  extern "C" __global__ void kernel##_##suffix(CclAllreduceArgs<T> args) {   \
    kernel##_body<T>(reinterpret_cast<T*>(args.dstMemObj->localPtr),          \
                     args.input, args.elementCount, args.myPe, args.npes);  \
  }

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

// All2all instantiations (uint32_t)
CCL_WRAP_ALL2ALL(OneShotAll2allSdmaKernel, u32, uint32_t)
CCL_WRAP_ALL2ALL(OneShotAll2allSdmaAsyncPutKernel, u32, uint32_t)

extern "C" __global__ void OneShotAll2allSdmaAsyncWaitKernel_u32(
    CclAll2allArgs<uint32_t> args) {
  OneShotAll2allSdmaAsyncWaitKernel_body(args.myPe, args.npes,
                                         args.outputTransitMemObj, args.flagsMemObj);
}

// AllGather instantiations (uint32_t)
CCL_WRAP_ALLGATHER(OneShotAllGatherSdmaKernel, u32, uint32_t)
CCL_WRAP_ALLGATHER_ASYNC_PUT(OneShotAllGatherSdmaAsyncPutKernel, u32, uint32_t)
CCL_WRAP_ALLGATHER_PARAM_CONTIGUOUS(OneShotAllGatherSdmaParamContiguousKernel, u32, uint32_t)
CCL_WRAP_ALLGATHER_PARAM_CONTIGUOUS_ASYNC_PUT(OneShotAllGatherSdmaParamContiguousAsyncPutKernel,
                                              u32, uint32_t)

// Sub-group intra-node SDMA AllGather. Type-agnostic byte move on u32 lanes.
extern "C" __global__ void OneShotAllGatherSdmaSubGroupKernel_u32(
    CclAllgatherSubGroupArgs<uint32_t> args) {
  OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
      args.myPe, args.npes, args.groupSize, args.groupPos, args.peBase, args.peStride, args.input,
      args.dstMemObj, args.flagsMemObj, args.elementCount, args.dstBaseOffset,
      args.dstSlotStrideBytes, args.flagVal, /*blockLocal=*/false, args.flagBase);
}

// Fused param-contiguous SubGroup gather: one launch loops node blocks + param
// splits internally (vs N_nodes*N_params separate launches).
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

// Sub-group intra-node SDMA broadcast: root (group pos 0) fans the full buffer to
// all members. Type-agnostic byte move on u32 lanes.
extern "C" __global__ void OneShotBroadcastSdmaSubGroupKernel_u32(
    CclBroadcastSubGroupArgs<uint32_t> args) {
  OneShotBroadcastSdmaSubGroupKernel_body<uint32_t>(
      args.myPe, args.groupSize, args.groupPos, args.peBase, args.peStride, args.input,
      args.dstMemObj, args.flagsMemObj, args.elementCount, args.dstBaseOffset, args.flagVal);
}

// AllReduce instantiations (uint32_t, int32_t, float, __half, hip_bfloat16)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_RS, SdmaReduceScatterKernel)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG, AllGatherSdmaKernel)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_RS_PUT, ReduceScatterSdmaPutKernel)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_LOCAL_REDUCE, ReduceScatterLocalReduceKernel)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG_REDUCED_PUT, AllGatherReducedSdmaPutKernel)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG, AllGatherAsyncPutKernel)
CCL_ALLREDUCE_ALL_TYPES(CCL_WRAP_ALLREDUCE_AG_WAIT, AllGatherAsyncWaitKernel)

// Inter-node RDMA ring AllGather: byte-move ring, chunkBytes per PE, npes-1
// rounds over the shmem transport; type-agnostic (raw bytes).
//
// RingFinishCopyKernel copies the gathered ring buffer OUT from the CU coherence
// domain, not via a copy-engine D2D: the ring's system-scope acquire + __threadfence_
// system order the RDMA-landed bytes only for the CUs, so a copy-engine read (separate
// hardware agent) can see stale remote-half bytes under tight overlap. A CU copy reads
// the same coherent HBM the ring fenced -- no host sync, overlap preserved.
extern "C" __global__ void RingFinishCopyKernel_u32(
    uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, uint64_t n) {
  uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = idx; i < n; i += stride) {
    dst[i] = src[i];
  }
}

extern "C" __global__ void InterNodeRingAllGatherKernel_u32(CclInterNodeRingArgs args) {
  // Sub-group ring over {peBase, peBase+peStride, ...}; flat whole-world ring is
  // the special case peBase=0, peStride=1, ringSize=npes, ringPos=myPe.
  AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.peBase, args.peStride,
                                  args.memObj, args.flagsObj, args.chunkBytes, args.numQp,
                                  /*numBlocksOverride=*/-1, /*bidOverride=*/-1,
                                  /*chunkReadyFlags=*/nullptr,
                                  /*deepPipe=*/1,
                                  /*deepPipeQuiet=*/0,
                                  /*useWriteFence=*/args.useWriteFence != 0);
}

// Fused inter-node ring + intra-node LOCAL-block SDMA gather. One launch overlaps
// the RDMA ring (blocks [0,ringBlocks)) with the ring-independent local-block gather
// (last block, blockLocal=true) -- the local shard is present from the start. Ring
// and gather touch disjoint symm objects / flag regions, so the union is byte-
// identical to two serial kernels. Remote-block gather (ring-dependent) still
// follows in a separate launch.
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

// Remote reassembly worker. Worker j (stride total) owns parts f=j, j+stride,...;
// each part waits its landing flag then SDMA-pushes that part's 16B-aligned tile of
// every remote block into the output on queue qId, overlapping ring channels still on
// the NIC. PUSH-ONLY => no dead-lock; distinct qId => per-queue counters never race;
// per-part flagBase => concurrent parts never race a slot. spatial=false: part f waits
// chunkReadyFlags[f]; spatial=true (single-channel split): all wait chunkReadyFlags[0].
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
  // Deep-SQ (reasmDeepSq!=0): two passes over queue qId. Pass 0 waits each channel's
  // landing flag then submits with no drain (deepSqPhase=1), keeping the SQ fed; pass 1
  // replays drain+flag only (deepSqPhase=2), one quiet to the accumulated count covering
  // every submit (SQ FIFO, signal monotonic). Wait stays in pass 0 (never submit before
  // bytes land); output flag fires only after the pass-1 drain => never precedes its
  // bytes (bit-exact). nPass==1 => deepSqPhase=0, byte-identical single-shot path.
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
      // Landing-flag wait guards only the SUBMIT; the drain pass already waited.
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
//   [0,rb)         ring channels, each publishes chunkReadyFlags[bid] on landing;
//   [rb]           ring-independent LOCAL block gather (this PE's own input);
//   (rb, 2*rb]     remote reassembly block j spins chunkReadyFlags[j] then SDMA-
//                  pushes sub-range j of every remote block, overlapping ring
//                  channel j+1 still on the NIC.
// The sub-range partition mirrors the ring's 16B-aligned tiling, so chunkReadyFlags[j]
// guards exactly the bytes reassembled here; disjoint flagBase per (j, remote m) =>
// concurrent reassembly blocks never race a subgroup flag slot.
extern "C" __global__ void FusedRingRemoteGatherKernel_u32(CclFusedRingRemoteGatherArgs args) {
  const int rb = args.ringBlocks;
  // deepPipe>1 (rb==1 only): the single channel splits the chunk into `pipe` temporal
  // sub-chunks each with its own landing flag. deepPipe<=1 => pipe==rb (byte-identical).
  const int pipe = args.deepPipe > 1 ? args.deepPipe : rb;
  const int reasm = args.reassemblyBlocks > 0 ? args.reassemblyBlocks : rb;
  const int effReasm = reasm;
  // Spatial split (rb==1, no deep-pipe, effReasm>1): split the one landed chunk into
  // effReasm disjoint byte sub-ranges for concurrent SDMA reassembly, no new inter race.
  // Each part waits chunkReadyFlags[0]. reasm>1 requires sdmaNumQueue>=reasm+1 (Python).
  const bool spatialSplit = (rb == 1 && args.deepPipe <= 1 && effReasm > 1);
  const int partition = spatialSplit ? effReasm : pipe;
  const unsigned bx = blockIdx.x;
  if (bx < static_cast<unsigned>(rb)) {
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/rb, /*bidOverride=*/static_cast<int>(bx),
                                    args.chunkReadyFlags,
                                    /*deepPipe=*/args.deepPipe,
                                    /*deepPipeQuiet=*/args.deepPipeQuiet,
                                    /*useWriteFence=*/args.useWriteFence != 0);
  } else if (bx == static_cast<unsigned>(rb)) {
    // Local block (m == nodeId): ring-independent, gathers this PE's own input on
    // flag slots [0, G). localPushOnly: push with NO coupled wait (queue 0, distinct
    // from reasm workers' 1..reasm) and let the completion reader below drain [0,G) too
    // -- avoids the coupled per-slot wait starving under deep pipe; same pushes + flags
    // => byte-identical, deadlock-free at any depth.
    if (args.localPushOnly != 0) {
      OneShotSubGroupPushOnly_body<uint32_t>(
          args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, args.gInput,
          args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*flagBase=*/0, /*qId=*/0,
          /*deepSqPhase=*/0,
          /*pushPeerLo=*/0,
          /*pushPeerHi=*/-1);
    } else if (args.crownRing != 0) {
      // Route the local-block gather through the intra-node nearest-neighbor ring
      // instead of the flat G-way push (avoids the G-1 concurrent-write incast). Self-
      // completes on its own high-base flag region [G+reasmTotal+G, ...) so it never
      // collides with the reassembly flags [G, G+reasmTotal) or local flags [0, G).
      // Same bytes, different schedule; crownRing default 0 => byte-identical.
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
      // fencedFlag bit meanings: see oneshot_sdma_kernel.hpp.
    } else {
      OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
          args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
          args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/0,
          /*pushPeerLo=*/0,
          /*pushPeerHi=*/-1);
    }
    // Completion reader: reassembly blocks are PUSH-ONLY (never wait a peer), so this
    // CTA does the single cross-rank completion wait for the whole remote reassembly --
    // guaranteed to finish (no circular dependency). Slot (j, mIdx, senderG) ==
    // G + (j*numRemote + mIdx)*G + senderG, disjoint from local block's [0, G).
    const int numRemote = args.numNodes - 1;
    if ((numRemote > 0 || args.localPushOnly != 0) && args.chunkReadyFlags != nullptr) {
      uint64_t* rflags = reinterpret_cast<uint64_t*>(args.gFlagsObj->localPtr);
      const int G = args.groupSize;
      // Completion wait fanned across the CTA's threads (each polls a strided subset).
      // Bit-exact: same slots + threshold, just concurrent; the fence + barrier below
      // still order every landed byte before the finish.
      const size_t reasmTotal = static_cast<size_t>(partition) * static_cast<size_t>(numRemote) *
                                static_cast<size_t>(G);
      // localPushOnly folds the local block's slots [0,G) into this wait (startIdx=0);
      // when OFF the local block did its own coupled wait so we start at G.
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
    // Remote reassembly block j in [0, reasm), pipelined over ring channels: owns
    // channels f with (f % reasm == j), waits chunkReadyFlags[f] then SDMA-pushes that
    // channel's 16B tile of every remote block, overlapping channels f+reasm,... still
    // on the NIC (the NIC<->XGMI pipeline; waiting all ring flags first collapses it).
    // PUSH-ONLY => no dead-lock; queue qId=j+1 (local CTA uses 0), reused serially.
    const int j = static_cast<int>(bx) - rb - 1;
    if (j < 0 || j >= reasm) return;
    FusedRemoteReassembleWorker(args, partition, /*j=*/j, /*stride=*/effReasm, /*qId=*/j + 1,
                                spatialSplit);
  }
}
