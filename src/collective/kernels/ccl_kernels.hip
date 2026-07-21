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
      args.dstSlotStrideBytes, args.flagVal, /*blockLocal=*/false, args.flagBase,
      args.multiQueue);
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

// L2-coherent re-touch. A barrier that orders the RDMA cross-PE landing plus the
// intra-node SDMA gather, composed with a plain CU re-touch (torch.add(out,0)),
// does not reach bit-exactness: the SDMA copy-engine-written output bytes are not
// L2-coherent to the consumer GEMM under output-buffer reuse. A plain add-0
// re-touch cannot fix it because torch.add issues agent-scope cacheable loads --
// it reads the same stale L2 line the previous step's GEMM cached for this reused
// buffer, adds 0, and writes the stale value straight back. This kernel does a
// system-scope acquire load (glc+dlc: bypass/invalidate L2, fetch the
// fabric-coherent HBM value the SDMA/RDMA agents actually wrote) followed by a
// system-scope release store, so the last writer of the consumed buffer is a CU
// op and the value it re-publishes is the freshly-landed one, not a stale cache
// line. Stream-ordered after the barrier, so no host stall. Rewriting the same
// value is bit-exact for every dtype. Targets only the large cross-node
// all-gathers where the stale-L2 read can occur.
// NOTE: system-scope __hip_atomic_* faults/hangs on gfx942 for coarse-grained
// torch CUDA memory (they require fine-grained/coherent allocations). Instead use
// a volatile load, which the CDNA backend lowers to a glc-tagged global load that
// bypasses the L2 line -> reads the fabric-coherent HBM value the SDMA/RDMA agents
// wrote (skipping the stale L2 line the previous step's GEMM cached for this
// reused buffer), then a normal store re-publishes it into the CU/L2 domain the
// consumer GEMM reads. Safe on coarse-grained memory, no atomics.
extern "C" __global__ void L2CoherentRetouchKernel_u32(
    uint32_t* __restrict__ out, uint64_t n) {
  // This reachable Python-path retouch (launch_l2_coherent_retouch, big-mode) must
  // do a real system-scope acquire-load + release-store: a plain `volatile` alias is
  // not enough. On CDNA3/gfx942 a volatile load lowers to a glc-tagged load that
  // bypasses only the per-CU L1 -- it still hits the stale device-scope L2 line the
  // consumer GEMM cached for this reused output buffer, so it would republish the
  // same stale word (the L2InvRetouch comment below documents the same failure mode).
  // Fetch with a system-scope load (glc/dlc == miss agent caches, read the
  // HBM-coherent bytes the SDMA/XGMI/RDMA agents landed) and re-publish with a
  // system-scope store (write-through == invalidate the stale L2 copy system-wide) so
  // the stream-ordered consumer GEMM reads the true landed data. Bit-exact by
  // construction (reads then writes the identical HBM-coherent word, changes no
  // value, every dtype). Default-inert: only launched under the retouch mode.
  uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = idx; i < n; i += stride) {
    uint32_t v = mori::core::AtomicLoadRelaxedSystem(out + i);
    mori::core::AtomicStoreRelaxedSystem(out + i, v);
  }
}

// L2InvRetouchKernel_u32: the consumer-coherence fix the volatile-only
// L2CoherentRetouchKernel above cannot close. On gfx942 a `volatile` load lowers to
// a glc-tagged load that bypasses only the per-CU L1 vector cache -- it still hits
// the stale line in the device-scope L2 that the previous step's consumer GEMM
// cached for this reused output buffer, so the re-touch would republish the same
// stale value and the deep-pipe device-fence path drifts. The fix is a device-side
// L2 invalidate. system-scope __hip_atomic_* faults on coarse-grained torch memory,
// but that is true only for atomic operations (they need fine-grained/coherent
// allocations); a pure __builtin_amdgcn_fence is a cache-control barrier that
// touches no buffer word and is safe on any allocation. A system-scope acquire fence
// lowers on gfx942 to `buffer_inv sc0 sc1` (invalidate both L1 and L2), so the
// subsequent loads miss the stale L2 line and fetch the fabric-coherent HBM value
// the SDMA/RDMA agents landed. Both load and store go through the glc `volatile`
// alias so no dirty L2 line exists for a concurrent wave's whole-L2 invalidate to
// drop; a trailing system-scope release fence (`buffer_wbinvl2`/writeback +
// s_waitcnt) drains every store to HBM before the kernel completes, so the consumer
// GEMM (stream-ordered after) reads the freshly-landed bytes. Rewriting the
// identical value is bit-exact for every dtype.
extern "C" __global__ void L2InvRetouchKernel_u32(
    uint32_t* __restrict__ out, uint64_t n) {
  volatile uint32_t* vp = reinterpret_cast<volatile uint32_t*>(out);
  // Device-side L2 invalidate BEFORE any read: drop stale L2 lines cached by the
  // previous step's GEMM for this reused buffer -> loads fall through to HBM.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = idx; i < n; i += stride) {
    uint32_t v = vp[i];
    vp[i] = v;
  }
  // Writeback/drain every store to HBM before the consumer GEMM (ordered after).
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}

// L2InvOnlyKernel_u32: the cheap consumer-coherence variant, without the full-buffer
// rewrite. The stale data lives in the device-scope L2 (shared across all CUs); the
// freshly-landed SDMA/RDMA bytes are already in HBM. A single system-scope acquire
// fence lowers to `buffer_inv sc0 sc1` on gfx942, which drops the device L2 lines, so
// the stream-ordered consumer GEMM misses L2 and re-fetches the fabric-coherent HBM
// value. buffer_inv is a whole-cache op (not address-ranged) so no bulk traffic is
// needed -- one wave per CU covers every per-CU L1 and the shared L2, at ~0 bytes
// moved (vs L2InvRetouch which rewrites every u32). The `out`/`n` args are accepted
// for launch-signature parity but unused. No writeback fence: this kernel writes
// nothing, so there are no dirty lines to drain.
extern "C" __global__ void L2InvOnlyKernel_u32(
    uint32_t* __restrict__ out, uint64_t n) {
  (void)out; (void)n;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

extern "C" __global__ void InterNodeRingAllGatherKernel_u32(CclInterNodeRingArgs args) {
  // Sub-group ring over {peBase, peBase+peStride, ...}; the flat whole-world
  // ring is the special case peBase=0, peStride=1, ringSize=npes, ringPos=myPe.
  AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.peBase, args.peStride,
                                  args.memObj, args.flagsObj, args.chunkBytes, args.numQp,
                                  /*numBlocksOverride=*/-1, /*bidOverride=*/-1,
                                  /*usePutSignal=*/args.usePutSignal != 0,
                                  /*useWriteImm=*/args.useWriteImm != 0,
                                  /*chunkReadyFlags=*/nullptr, /*opGen=*/args.opGen,
                                  /*useRead=*/args.useRead != 0, /*wqeDepth=*/1, /*deepPipe=*/1,
                                  /*deepPipeImm=*/0, /*deepPipeQuiet=*/0, /*dpSerialDrain=*/0,
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
    // IN-KERNEL COPY-IN (fuseCopyIn): stage this ring channel's own send sub-range
    // of gInput into the local ring slot before the RDMA put, so the host
    // hipMemcpyAsync copy-in can be dropped (launch-collapse). Tiling mirrors
    // AllGatherRingSubGroupKernelBody's multiBlock split so channel bx writes
    // exactly the bytes it sends (no cross-CTA dep; __syncthreads orders stage
    // before send). The local-gather CTA and remote consumers never read this PE's
    // own ring slot, so no other consumer races the stage. Byte-identical when 0.
    if (args.fuseCopyIn != 0) {
      const int rb = args.ringBlocks;
      const unsigned bx = blockIdx.x;
      const int nextPeer =
          args.ringPeBase + ((args.ringPos + 1) % args.ringSize) * args.ringPeStride;
      const mori::application::TransportType nx =
          mori::shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
      const bool peerRdma = (nx == mori::application::TransportType::RDMA);
      const bool mb = (rb > 1 && peerRdma);
      size_t blkOff = 0;
      size_t blkBytes = args.chunkBytes;
      if (mb) {
        const size_t kAlignB = 16;
        size_t nUnits = (args.chunkBytes + kAlignB - 1) / kAlignB;
        size_t unitsPerBlk = (nUnits + static_cast<size_t>(rb) - 1) / static_cast<size_t>(rb);
        size_t startUnit = static_cast<size_t>(bx) * unitsPerBlk;
        size_t endUnit = startUnit + unitsPerBlk;
        if (endUnit > nUnits) endUnit = nUnits;
        blkOff = startUnit * kAlignB;
        size_t blkEnd = endUnit * kAlignB;
        if (blkEnd > args.chunkBytes) blkEnd = args.chunkBytes;
        blkBytes = (blkOff < blkEnd) ? (blkEnd - blkOff) : 0;
      } else if (rb > 1 && !peerRdma && bx != 0) {
        blkBytes = 0;  // single-node sim: only channel 0 sends (mirrors ring body)
      }
      if (blkBytes > 0) {
        char* ringBase = reinterpret_cast<char*>(args.ringMemObj->localPtr) +
                         static_cast<size_t>(args.ringPos) * args.chunkBytes;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(args.gInput) + blkOff);
        uint32_t* dst = reinterpret_cast<uint32_t*>(ringBase + blkOff);
        const size_t n32 = blkBytes / 4;
        for (size_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
      }
      __syncthreads();
    }
    // Device-side gen-ring: when opGenCounter is wired (MORI_HIER_GEN_RING_DEV),
    // this ring channel (blockIdx.x) computes this op's generation on-device by
    // bumping its own counter slot. Because the increment happens inside the kernel,
    // it advances on every execution -- eager warmup and every HIP-graph replay --
    // so the generation stays in lockstep with the sender's per-op flag AMO_ADD(1)
    // even under graph capture (where the host prepare_stream runs only once). One
    // thread per block owns its slot (no cross-block/cross-thread race); broadcast
    // via shared memory. opGen==0 (nullptr counter) keeps the classic reset+barrier
    // path byte-identical.
    uint64_t myGen = 0;
    if (args.opGenCounter != nullptr) {
      __shared__ uint64_t s_gen;
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        uint64_t g = args.opGenCounter[blockIdx.x] + 1;
        args.opGenCounter[blockIdx.x] = g;
        s_gen = g;
      }
      __syncthreads();
      myGen = s_gen;
    }
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/args.ringBlocks,
                                    /*bidOverride=*/static_cast<int>(blockIdx.x),
                                    /*usePutSignal=*/args.usePutSignal != 0,
                                    /*useWriteImm=*/args.useWriteImm != 0,
                                    /*chunkReadyFlags=*/nullptr,
                                    /*opGen=*/myGen);
  } else {
    OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
        args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
        args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
        args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/0,
        /*multiQueue=*/args.multiQueue, /*qFlag=*/args.qFlag,
        /*ringPhased=*/args.ringPhased, /*batchFence=*/args.batchFence,
        /*h2x4=*/args.h2x4);
  }
}

// Remote reassembly WORKER (shared by the dedicated reassembly CTAs and, under
// MORI_HIER_FUSE_REMOTE_ELASTIC, the local-block CTA). Worker ``j`` (of ``stride``
// total workers) owns ring channels f with (f % stride == j). For each such
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
  // Skewed temporal split: mirror the producer's P==2 boundary (bnd = nUnits -
  // nUnits*pct/100) so this worker reassembles exactly the bytes flag slot f
  // guards. Only for the temporal reassembly (partition == deepPipe == 2, not
  // spatial/inline) with 0<pct<100, pct!=50; else uniform (byte-identical).
  // pct>50 head-skews (small first sub-chunk) to cut first-land latency at small
  // sizes; the boundary formula is identical to the producer. See HierDpTailPct.
  const bool dpSkew = (!spatial && args.dpTailPct > 0 && args.dpTailPct < 100 &&
                       args.dpTailPct != 50 && args.deepPipe == 2 && partition == 2);
  const size_t sliceElems = args.chunkBytes / sizeof(uint32_t);
  const int numRemote = args.numNodes - 1;
  uint32_t* ringBase = reinterpret_cast<uint32_t*>(args.ringMemObj->localPtr);
  // Every-GPU-direct write-push: the remote half is delivered by the remote
  // nodes' ring-CTA pushes straight into this rank's output (with a fused AMO_SET of
  // the completion flags); the local reasm workers do nothing (they would otherwise
  // spin on chunkReadyFlags that are never published under every-direct). The bx==rb
  // completion reader waits on the AMO_SET flags. See HierEveryDirectWriteOn.
  if (args.everyDirectWrite != 0) return;
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
      size_t startUnit, endUnit;
      if (dpSkew) {
        size_t tailU = (nUnits * static_cast<size_t>(args.dpTailPct)) / 100;
        size_t bnd = nUnits - tailU;
        if (f == 0) { startUnit = 0; endUnit = bnd; }
        else { startUnit = bnd; endUnit = nUnits; }
      } else {
        startUnit = static_cast<size_t>(f) * unitsPerChan;
        endUnit = startUnit + unitsPerChan;
        if (endUnit > nUnits) endUnit = nUnits;
      }
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
          // Gen-token: when args.opGen>0 the publisher stores the per-op token
          // (not a fixed 1) and the buffer is never host-reset, so this landing wait
          // gates on `< opGen` (slot reaches this op's generation); opGen==0 keeps the
          // legacy `< 1` (host-zeroed each op). Same slot, same one-shot semantics.
          const uint64_t genThresh = args.opGen ? args.opGen : static_cast<uint64_t>(1);
          while (mori::core::AtomicLoadSeqCstSystem(args.chunkReadyFlags + flagIdx) <
                 genThresh) {
            // Cooperative backoff (spinBackoff): yield fabric/L2 to the SDMA pushes +
            // mlx5 CQ drainers so a contended DEEP_PIPE>=8 landing can win. Bit-exact:
            // same flag/threshold gates the fence, only the poll cadence changes.
            if (args.spinBackoff != 0) { __builtin_amdgcn_s_sleep(2); }
            if (++spin > 100000000) {
              // spinBlock: the bounded fallback can fire on under-landed bytes under
              // DEEP_PIPE>=8 contention, causing E2E drift. When on, never abandon:
              // reset the counter and keep polling until the flag actually lands
              // (genuine in-kernel HW-completion fence).
              if (args.spinBlock == 0) break;  // bounded fallback (default path)
              spin = 0;                          // blocking: poll until it lands
            }
          }
          __threadfence_system();
        }
        __syncthreads();
      }
      if (subBytes == 0) continue;
      const size_t subElems = subBytes / sizeof(uint32_t);  // 16B-aligned => exact
      // Pipelined relay-ring reassembly. Replace the flat G-way scatter (below)
      // with a G-1 step relay ring per remote block -- perfect-matching per step,
      // no G-way XGMI incast (native single-node ring pattern). Same final layout
      // and same per-shard completion flags.
      if (args.reasmRing != 0) {
        int miR = 0;
        for (int m = 0; m < args.numNodes; ++m) {
          if (m == args.nodeId) continue;  // local block handled by the bx==rb gather
          const size_t flagBase = static_cast<size_t>(args.groupSize) +
                                  (static_cast<size_t>(f) * numRemote + miR) *
                                      static_cast<size_t>(args.groupSize);
          const size_t blockBaseBytes =
              static_cast<size_t>(m) * static_cast<size_t>(args.groupSize) * args.chunkBytes;
          // Non-direct-land: stage THIS rank's own shard from the ring buffer; direct-
          // land: the inter RDMA already put it in the output self-slot (ringSelfSrc=nullptr).
          const uint32_t* ringSelfSrc =
              (args.directLand != 0)
                  ? nullptr
                  : (ringBase + static_cast<size_t>(m) * sliceElems + (subOff / sizeof(uint32_t)));
          OneShotSubGroupPushRing_body<uint32_t>(
              args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, args.gDstMemObj,
              args.gFlagsObj, blockBaseBytes, subOff, subBytes, /*slotStride=*/args.chunkBytes,
              args.gFlagVal, flagBase, /*qId=*/qId, ringSelfSrc);
          ++miR;
        }
        continue;  // ring handled this sub-range; skip the flat scatter
      }
      int mi = 0;
      for (int m = 0; m < args.numNodes; ++m) {
        if (m == args.nodeId) continue;  // local block handled by the bx==rb CTA gather
        // Direct-land: the inter RDMA WRITE placed block m straight into this GPU's
        // own output self-slot ((m*groupSize+groupPos)*chunkBytes), so read the
        // broadcast source from there (== the bytes the ring would have held) and let
        // the body skip the redundant self column copy (skipSelfCopy below). OFF =>
        // read from the ring buffer as the default path does.
        uint32_t* src =
            (args.directLand != 0)
                ? (reinterpret_cast<uint32_t*>(args.gDstMemObj->localPtr) +
                   ((static_cast<size_t>(m) * static_cast<size_t>(args.groupSize) +
                     static_cast<size_t>(args.groupPos)) *
                        args.chunkBytes +
                    subOff) /
                       sizeof(uint32_t))
                : (ringBase + static_cast<size_t>(m) * sliceElems + (subOff / sizeof(uint32_t)));
        const size_t dstBase =
            static_cast<size_t>(m) * static_cast<size_t>(args.groupSize) * sliceElems *
                sizeof(uint32_t) +
            subOff;
        const size_t flagBase = static_cast<size_t>(args.groupSize) +
                                (static_cast<size_t>(f) * numRemote + mi) *
                                    static_cast<size_t>(args.groupSize);
        // Read-coalescing tile. When MORI_HIER_REASM_L2TILE>0 (and none of the
        // special completion modes are active) split this per-peer copy into L2-sized
        // byte windows and drive all groupSize peer copies through the same tile window
        // together -- the body's own per-call drain+__syncthreads is a per-tile
        // completion barrier, so the tile stays L2-resident across the groupSize-1
        // redundant reads instead of each peer re-reading the whole block from HBM. The
        // completion flag fires only on the last tile (pushFlag) so it never precedes
        // any byte (bit-exact). L2Tile==0 => single call == byte-identical default path.
        const bool l2tile =
            (args.reasmL2Tile > 0 && deepSqPhase == 0 && args.reasmMultiQueue == 0 &&
             args.pushPhased == 0 && args.qFlag == 0 && args.reasmQSplit == 0 &&
             args.fuseSenderFence == 0 && args.directLand == 0);
        if (l2tile) {
          const size_t tileElems = args.reasmL2Tile / sizeof(uint32_t);
          size_t done = 0;
          while (done < subElems) {
            const size_t te = (subElems - done < tileElems) ? (subElems - done) : tileElems;
            const int lastTile = (done + te >= subElems) ? 1 : 0;
            OneShotSubGroupPushOnly_body<uint32_t>(
                args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, src + done,
                args.gDstMemObj, args.gFlagsObj, te, dstBase + done * sizeof(uint32_t),
                /*dstSlotStrideBytes=*/args.chunkBytes, args.gFlagVal, flagBase, /*qId=*/qId,
                /*deepSqPhase=*/0, /*qSplit=*/0, /*qStride=*/stride, /*fuseFence=*/0,
                /*ndesc=*/args.putNdesc, /*pushRotate=*/args.pushRotate, /*qFlag=*/0,
                /*multiQueue=*/0, /*pushPhased=*/0, /*pushPeerLo=*/0, /*pushPeerHi=*/-1,
                /*pushFlag=*/lastTile);
            done += te;
          }
        } else {
          OneShotSubGroupPushOnly_body<uint32_t>(
              args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, src, args.gDstMemObj,
              args.gFlagsObj, subElems, dstBase, /*dstSlotStrideBytes=*/args.chunkBytes,
              args.gFlagVal, flagBase, /*qId=*/qId, /*deepSqPhase=*/deepSqPhase,
              /*qSplit=*/args.reasmQSplit, /*qStride=*/stride,
              /*fuseFence=*/args.fuseSenderFence, /*ndesc=*/args.putNdesc,
            /*pushRotate=*/args.pushRotate, /*qFlag=*/args.qFlag,
            /*multiQueue=*/args.reasmMultiQueue, /*pushPhased=*/args.pushPhased,
            /*pushPeerLo=*/0, /*pushPeerHi=*/-1, /*pushFlag=*/1,
            // Do not skip the self column under direct-land. The RDMA WRITE lands
            // the block into the output self-slot, but a pure NIC write leaves that
            // slot outside the copy-engine coherence path every other slot gets
            // (copy-engine write + system fence), so the consuming CU read observes a
            // stale line at exactly the self-slot -- a bit-exact mismatch at rank
            // (m*groupSize+groupPos). Copying the self column src==dst (output
            // self-slot -> itself, identity) re-touches it through the same
            // copy-engine+fence path as the peer columns => coherent. Direct-land
            // still deletes the ring bounce (RDMA lands straight in output, no ring
            // read/footprint) -- the primary volume win -- only the one self copy is
            // kept for coherence. Set MORI_HIER_DIRECT_LAND_SKIPSELF=1 to skip it
            // (incoherent). OFF (directLand==0) => unchanged default path.
            /*skipSelfCopy=*/(args.directLand != 0 ? args.directLandSkipSelf : 0));
        }
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
// Device double-buffered ring parity bump. Single-thread kernel that
// increments the per-op parity counter on-device. Launched (captured) on the comm
// stream BEFORE FusedRingRemoteGatherKernel_u32 each op, so it advances on every
// HIP-graph replay (a host counter would freeze at capture) and every grid block
// reads one stable post-bump value. Only launched when MORI_HIER_GEN_RING_DBL on.
extern "C" __global__ void RingParityBumpKernel_u32(uint64_t* parityCounter) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    parityCounter[0] = parityCounter[0] + 1;
  }
}

// Fuse-entry-barrier: the in-kernel entry rendezvous. Replaces the separate
// host-launched ShmemBarrierOnStream (_barrier_kernel<<<1,1>>>) with a device-side
// prologue so the whole op is one launch. Correctness is identical to the
// separate barrier: the same PE0-funnel
// cross-PE rendezvous (ShmemBarrierAllBlock, done by block 0 exactly like
// _barrier_kernel) fences every PE before any ring atomic; a manual grid-arrival
// gate makes ALL grid blocks wait for block 0's rendezvous before proceeding.
// ga[0] = arrival counter (returns to 0 each op), ga[1] = monotonic release
// generation (never reset -> graph-replay safe: each op sees a fresh ga[0]==0 and
// a strictly larger release gen). Single barrier per launch => no reuse hazard.
__device__ __forceinline__ void FusedEntryGridRendezvous(unsigned int* ga) {
  const unsigned int nb = static_cast<unsigned int>(gridDim.x);
  __shared__ unsigned int prevRel;
  // Every block records the current release generation BEFORE signalling arrival,
  // so block 0 (which waits for all arrivals) can only bump the gen afterwards.
  if (threadIdx.x == 0) prevRel = atomicAdd(&ga[1], 0u);
  __syncthreads();
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      // Wait for every OTHER block of this grid to arrive (block 0 does not count
      // itself; nb==1 => the predicate is already satisfied).
      while (atomicAdd(&ga[0], 0u) < (nb - 1u)) { /* spin */ }
    }
    __syncthreads();
    // Cross-PE rendezvous -- ALL threads of block 0 participate (block-scoped,
    // thread 0 drives the network op, others __syncthreads inside).
    mori::shmem::ShmemBarrierAllBlock();
    if (threadIdx.x == 0) {
      // Release the grid: reset the arrival counter for the next op, then advance
      // the generation so the waiting blocks fall through.
      atomicExch(&ga[0], 0u);
      __threadfence();
      atomicAdd(&ga[1], 1u);
    }
    __syncthreads();
  } else {
    if (threadIdx.x == 0) {
      atomicAdd(&ga[0], 1u);
      while (atomicAdd(&ga[1], 0u) == prevRel) { /* spin */ }
    }
    __syncthreads();
  }
}

extern "C" __global__ void FusedRingRemoteGatherKernel_u32(CclFusedRingRemoteGatherArgs args) {
  // Fuse-entry-barrier: run the cross-PE entry rendezvous device-side (block 0) behind
  // a grid-arrival gate, before any ring/gather work, when the host skipped the separate
  // barrier launch. Every grid block executes this unconditionally so none races ahead
  // of the rendezvous. Default OFF => byte-identical to the separate-barrier path.
  if (args.fuseEntryBarrier != 0 && args.gridArrival != nullptr) {
    FusedEntryGridRendezvous(args.gridArrival);
  }
  // Device double-buffered ring: select the active ring half by op parity so op N+1's
  // peer pushes land in the other half than op N's still-reassembling half (closes the
  // barrier-free gen-ring reuse race). parityCounter was bumped once for this op by
  // the captured pre-kernel (MoriRingParityBumpKernel) on the same stream, so every
  // block reads one stable, graph-replay-advancing value.
  // args is by-value => reassigning ringMemObj propagates to the in-kernel copy-IN,
  // the ring send/recv (AllGatherRingSubGroupKernelBody), and the reassembly worker
  // with no body changes. null parityCounter => single-buffer (byte-identical).
  if (args.parityCounter != nullptr) {
    if ((args.parityCounter[0] & 1ULL) != 0ULL) {
      args.ringMemObj = args.ringMemObjB;
    }
    // Device flag-token: derive the per-op chunkReadyFlags generation from the graph-safe
    // device parity counter (bumped once/op by RingParityBumpKernel, so it advances on
    // every HIP-graph replay -- a host counter freezes at capture). args is by-value =>
    // overwriting opGen here propagates to both the publisher (ring body AtomicStore
    // chunkReadyFlags = opGen) and the consumer (reasm worker wait `< opGen`), so the
    // landing flags become a monotonic per-op token that supersedes stale slots with no
    // host reset -- closing the barrier-free flag-reset race. Every grid block reads
    // the same post-bump value (single launch). Gated on flagGenDev so the default
    // path stays byte-identical.
    if (args.flagGenDev != 0) {
      args.opGen = args.parityCounter[0];
    }
  }
  const int rb = args.ringBlocks;
  // Deep-SQ temporal pipeline: when deepPipe>1 (only on rb==1) the single ring
  // channel splits the chunk into ``pipe`` temporal sub-chunks each with its own
  // landing flag, so the reassembly workers + completion reader iterate over
  // ``pipe`` sub-chunk flags instead of ``rb`` spatial channel flags. deepPipe<=1
  // => pipe==rb (byte-identical default path).
  const int pipe = args.deepPipe > 1 ? args.deepPipe : rb;
  const int reasm = args.reassemblyBlocks > 0 ? args.reassemblyBlocks : rb;
  // ELASTIC REASSEMBLY: the local-block CTA joins reassembly as an extra worker
  // (index == reasm) on queue 0 after its own gather, so the tail runs on
  // effReasm = reasm+1 SDMA queues. OFF => effReasm == reasm (byte-identical).
  const int effReasm = args.elasticReasm != 0 ? reasm + 1 : reasm;
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
  // Per-QP fine-grain inter-arrival drain (shardDrain): the producer publishes one
  // progressive landing flag per QP shard (sw = min(numQp, warpsPerBlock=512/64=8)),
  // so the reasm worker iterates numQp shards on its single queue, each gated on its
  // own per-QP flag -- shard s pushes while s+1..sw-1 still cross the NIC. Overrides
  // the temporal pipe partition; the per-QP byte ranges (ceil(nUnits/sw)) match the
  // producer's tiling exactly => bit-exact.
  int swShards = args.numQp;
  if (swShards > 512 / warpSize) swShards = 512 / warpSize;
  if (swShards < 1) swShards = 1;
  // Every-GPU-direct write-push: force the sub-range count to numQp so each
  // (receiver, column) push rides a distinct QP (qp=f%numQp) -- the in-flight WQE
  // depth the mlx5 WRITE pipe needs (partition=1 => one QP/peer underfills it). The
  // bx==rb completion reader uses this same partition, so its flag range stays
  // consistent.
  const int partition =
      (args.everyDirectWrite != 0)
          ? (args.numQp > 1 ? args.numQp : 1)
          : ((args.shardDrain != 0) ? swShards : (spatialSplit ? effReasm : pipe));
  const unsigned bx = blockIdx.x;
  if (bx < static_cast<unsigned>(rb)) {
    // Host-proxy inter (hostProxyInter): a host proxy owns the inter-node leg --
    // it posts sub-chunk p's cross-node RDMA writes into the ring buffer, drains
    // its send-CQ (the host-drain landing fence, bit-exact at the very large sizes
    // where the device per-sub-chunk signal races/crashes), and publishes
    // chunkReadyFlags[p] from the host. So the device ring-send blocks do nothing
    // here; the reassembly workers + completion reader (below) consume the
    // host-published flags unchanged. OFF => device posts inter as before.
    if (args.hostProxyInter != 0) {
      // no-op: inter leg owned by the host proxy.
    } else {
    // IN-KERNEL COPY-IN (fuseCopyIn): stage this ring channel's OWN send sub-range
    // of gInput into the local ring slot before the RDMA put, so the host
    // hipMemcpyAsync copy-IN can be dropped (single-launch collapse). The tiling
    // MIRRORS AllGatherRingSubGroupKernelBody's multiBlock split so channel bx
    // writes exactly the bytes it sends (no cross-CTA dependency; __syncthreads
    // orders the stage before the send within this CTA). The remote reassembly /
    // completion reader never read the LOCAL slot (they gate on the received
    // chunks' flags), so no other consumer races this stage.
    if (args.fuseCopyIn != 0) {
      const int nextPeer =
          args.ringPeBase + ((args.ringPos + 1) % args.ringSize) * args.ringPeStride;
      const mori::application::TransportType nx =
          mori::shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
      const bool peerRdma = (nx == mori::application::TransportType::RDMA);
      const bool mb = (rb > 1 && peerRdma);
      size_t blkOff = 0;
      size_t blkBytes = args.chunkBytes;
      if (mb) {
        const size_t kAlignB = 16;
        size_t nUnits = (args.chunkBytes + kAlignB - 1) / kAlignB;
        size_t unitsPerBlk = (nUnits + static_cast<size_t>(rb) - 1) / static_cast<size_t>(rb);
        size_t startUnit = static_cast<size_t>(bx) * unitsPerBlk;
        size_t endUnit = startUnit + unitsPerBlk;
        if (endUnit > nUnits) endUnit = nUnits;
        blkOff = startUnit * kAlignB;
        size_t blkEnd = endUnit * kAlignB;
        if (blkEnd > args.chunkBytes) blkEnd = args.chunkBytes;
        blkBytes = (blkOff < blkEnd) ? (blkEnd - blkOff) : 0;
      } else if (rb > 1 && !peerRdma && bx != 0) {
        // single-node simulation: only channel 0 sends (mirrors the ring body).
        blkBytes = 0;
      }
      if (blkBytes > 0) {
        char* ringBase = reinterpret_cast<char*>(args.ringMemObj->localPtr) +
                         static_cast<size_t>(args.ringPos) * args.chunkBytes;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(args.gInput) + blkOff);
        uint32_t* dst = reinterpret_cast<uint32_t*>(ringBase + blkOff);
        const size_t n32 = blkBytes / 4;
        for (size_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
      }
      __syncthreads();
    }
    // Every-GPU-direct write-push fan-out (MORI_HIER_EVERY_DIRECT_WRITE): delete
    // the inter ring send and the 8-way XGMI reasm scatter. My ring self-slot (ringPos
    // chunk, staged by copy-in above / at entry) already holds my block. Instead of
    // ringing it to my successor then XGMI-broadcasting the received remote chunk to all
    // G local peers, push my block directly to every receiver on every remote node,
    // straight into that receiver's final output self-slot ((nodeId*G+groupPos) chunk),
    // fused with an AMO_SET of the receiver's completion flag on the same QP (RC in-order
    // => the flag can't beat the data). Warp w drives receiver g==w (G warps => G distinct
    // dest NICs). The push target is terminal output (nothing re-reads it), so it sidesteps
    // the self-slot SDMA-read coherence hazard. Sub-range tiling matches the bx==rb
    // completion reader's `partition` so every (sub-range f, my column) flag it waits on is
    // set exactly once. This WRITE variant streams where an every-direct read is latency-bound.
    if (args.everyDirectWrite != 0) {
      const int G = args.groupSize;
      const int edNumRemote = args.numNodes - 1;
      const int edWarpId = static_cast<int>(threadIdx.x) / warpSize;
      const int edLaneId = static_cast<int>(threadIdx.x) % warpSize;
      const size_t kAlignW = 16;
      const size_t nUnitsW = (args.chunkBytes + kAlignW - 1) / kAlignW;
      const size_t unitsPerPart =
          (nUnitsW + static_cast<size_t>(partition) - 1) / static_cast<size_t>(partition);
      // My block's byte base in the receiver's output = my global slot (nodeId*G+groupPos).
      const size_t myOutBase =
          (static_cast<size_t>(args.nodeId) * static_cast<size_t>(G) +
           static_cast<size_t>(args.groupPos)) * args.chunkBytes;
      // My own block lives in ring self-slot ringPos (== nodeId), staged before entry.
      const size_t myRingBase = static_cast<size_t>(args.ringPos) * args.chunkBytes;
      if (edWarpId < G) {
        const int g = edWarpId;
        for (int m = 0; m < args.numNodes; ++m) {
          if (m == args.nodeId) continue;
          const int remotePe = m * G + g;
          // Receiver g on node m: index of MY node among its remote nodes (numRemote list).
          const int miR = (args.nodeId < m) ? args.nodeId : (args.nodeId - 1);
          for (int f = 0; f < partition; ++f) {
            const size_t sU = static_cast<size_t>(f) * unitsPerPart;
            size_t eU = sU + unitsPerPart;
            if (eU > nUnitsW) eU = nUnitsW;
            const size_t subOff = sU * kAlignW;
            size_t subEnd = eU * kAlignW;
            if (subEnd > args.chunkBytes) subEnd = args.chunkBytes;
            if (subOff >= subEnd) continue;
            const size_t subBytes = subEnd - subOff;
            const size_t flagIdx = static_cast<size_t>(G) +
                                   (static_cast<size_t>(f) * static_cast<size_t>(edNumRemote) +
                                    static_cast<size_t>(miR)) * static_cast<size_t>(G) +
                                   static_cast<size_t>(args.groupPos);
            const int qp = (args.numQp > 0) ? (f % args.numQp) : 0;
            mori::shmem::ShmemPutMemNbiSignalWarp(
                args.gDstMemObj, myOutBase + subOff, args.ringMemObj, myRingBase + subOff,
                subBytes, args.gFlagsObj, flagIdx * sizeof(uint64_t), args.gFlagVal,
                mori::core::atomicType::AMO_SET, remotePe, qp);
          }
        }
      }
      __syncthreads();
      // Quiesce this warp's dest QPs (send-CQ drain); the receiver's flag wait is the
      // cross-rank landing fence (AMO rode after the data on the same RC QP).
      if (edLaneId == 0 && edWarpId < G) {
        const int g = edWarpId;
        for (int m = 0; m < args.numNodes; ++m) {
          if (m == args.nodeId) continue;
          const int remotePe = m * G + g;
          for (int qp = 0; qp < args.numQp; ++qp) {
            mori::shmem::ShmemQuietThread(remotePe, qp);
          }
        }
      }
      __syncthreads();
    } else {
    // Ring channel bx: publishes chunkReadyFlags[bx] on landing.
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/rb, /*bidOverride=*/static_cast<int>(bx),
                                    /*usePutSignal=*/args.usePutSignal != 0,
                                    /*useWriteImm=*/args.useWriteImm != 0, args.chunkReadyFlags,
                                    /*opGen=*/args.opGen, /*useRead=*/args.useRead != 0,
                                    /*wqeDepth=*/args.wqeDepth,
                                    /*deepPipe=*/args.deepPipe, /*deepPipeImm=*/args.deepPipeImm,
                                    /*deepPipeQuiet=*/args.deepPipeQuiet,
                                    /*dpSerialDrain=*/args.dpSerialDrain,
                                    /*useWriteFence=*/args.useWriteFence != 0,
                                    /*fifoFullWidth=*/args.fifoFullWidth,
                                    /*dpTailPct=*/args.dpTailPct,
                                    /*fifoProg=*/args.fifoProg,
                                    /*shardDrain=*/args.shardDrain,
                                    /*directLand=*/args.directLand,
                                    /*gOutMemObj=*/args.gDstMemObj,
                                    /*gGroupSize=*/args.groupSize,
                                    /*gGroupPos=*/args.groupPos);
    // Same-CTA inline reassembly: this ring CTA just landed spatial sub-range bx of the
    // remote chunk and made it visible to this GPU (the receiver threadfence_system
    // inside the ring body, then chunkReadyFlags[bx] publish). On the multiBlock ring
    // (rb>1) the dedicated reassembly worker j==bx would push exactly this sub-range
    // (partition==rb, stride==effReasm==rb, qId==bx+1). Do that push here instead -- no
    // cross-CTA chunkReadyFlags relay (avoids a stale-L2 hazard), no extra reassembly CTA
    // (avoids CU/XGMI contention). Byte-identical: the same
    // FusedRemoteReassembleWorker, SAME per-channel tile, SAME per-queue drain + output
    // flags; the worker's landing wait on chunkReadyFlags[bx] returns immediately (this
    // CTA already published it). The completion reader (bx==rb CTA) is unchanged. The
    // Python launcher drops the dedicated reassembly CTAs (grid rb+1). spatial=false so
    // part bx waits its own flag; gate mirrors the C++ builder (rb>1, no host-proxy).
    if (args.inlineReasm != 0 && args.chunkReadyFlags != nullptr) {
      __syncthreads();
      FusedRemoteReassembleWorker(args, /*partition=*/rb, /*j=*/static_cast<int>(bx),
                                  /*stride=*/rb, /*qId=*/static_cast<int>(bx) + 1,
                                  /*spatial=*/false);
    }
    }  // end else (!everyDirectWrite): ring send + inline reasm
    }
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
      // Multi-engine per-link on the local 8-GPU gather (MORI_INTRA_MQ). The dominant
      // intra leg is this local push-only gather (7 XGMI peers). Passing multiQueue here
      // lets mqActive engage (fuseSenderFence defaults OFF here => fuseFence==0)
      // so each peer column is split across all sdmaNumQueue engines via SdmaPutWarp.
      // Disjoint sub-ranges, all nqTop queues drained before the one flag AMO. Default
      // (INTRA_MQ unset) => reasmMultiQueue==0 => byte-identical.
      OneShotSubGroupPushOnly_body<uint32_t>(
          args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, args.gInput,
          args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*flagBase=*/0, /*qId=*/0,
          /*deepSqPhase=*/0, /*qSplit=*/0, /*qStride=*/1,
          /*fuseFence=*/args.fuseSenderFence, /*ndesc=*/args.putNdesc,
          /*pushRotate=*/args.pushRotate, /*qFlag=*/0,
          /*multiQueue=*/args.reasmMultiQueue, /*pushPhased=*/0,
          /*pushPeerLo=*/0,
          /*pushPeerHi=*/(args.offloadPeers > 0 ? args.groupSize - args.offloadPeers : -1));
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
      // Forward qFlag (MORI_HIER_QFLAG) into the local-block gather. This branch
      // (localPushOnly OFF => the default path) passes multiQueue==0 so qFlagActive can
      // engage the copy-engine FIFO-ordered flag delivery (COPY_LINEAR+FENCE flag on the
      // same SDMA queue as its data, no per-peer ShmemQuietThread drain +
      // __threadfence_system + separate P2P AMO). FENCE is FIFO after the copy on one
      // engine so the completion reader's seq-cst SYSTEM acquire never sees the flag ahead
      // of the bytes. Default (QFLAG unset => args.qFlag==0) => byte-identical.
      // FIRST-LAND idle-engine reclamation (see HierLocalOffload): the queue-0 local
      // CTA issues only peer-columns [0, G-offloadPeers); the last offloadPeers columns
      // are pushed by the otherwise-idle reasm CTA during its pre-land spin (below).
      // The completion WAIT inside the body is unchanged (full [0,G)). offloadPeers==0
      // => pushPeerHi==-1 => full-mesh push == byte-identical.
      const int _offPeerHi =
          (args.offloadPeers > 0) ? (args.groupSize - args.offloadPeers) : -1;
      OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
          args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
          args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/0,
          /*multiQueue=*/0, /*qFlag=*/args.qFlag, /*ringPhased=*/args.ringPhased,
          /*batchFence=*/args.batchFence, /*h2x4=*/args.h2x4, /*pushPeerLo=*/0,
          /*pushPeerHi=*/_offPeerHi, /*putNdesc=*/args.putNdesc,
          /*poleSqDepth=*/args.poleSqDepth, /*polePull=*/args.polePull,
          /*putCacheHint=*/args.putCacheHint, /*intra2=*/args.intra2);
    }
    // ELASTIC REASSEMBLY: this CTA's own-shard gather (queue 0) is done, so join
    // remote reassembly as worker index==reasm (stride effReasm) on queue 0 --
    // reassemble the tail channels concurrently with the reasm dedicated blocks,
    // giving the un-hidden reassembly leg effReasm SDMA queues. Must complete
    // before the completion reader below (its published flags are counted there).
    if (args.elasticReasm != 0 && args.chunkReadyFlags != nullptr) {
      FusedRemoteReassembleWorker(args, partition, /*j=*/reasm, /*stride=*/effReasm, /*qId=*/0,
                                  spatialSplit);
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
          // Cooperative backoff (spinBackoff): see chunkReadyFlags wait above.
          if (args.spinBackoff != 0) { __builtin_amdgcn_s_sleep(2); }
          if (++spin > 100000000) {
            // spinBlock: completion reader must not publish on under-landed
            // reassembly output. When on, poll until the flag lands.
            if (args.spinBlock == 0) break;  // bounded fallback (default path)
            spin = 0;                          // blocking: poll until it lands
          }
        }
      }
      __threadfence_system();
    }
    // CU-coherent re-touch (fused copy-out, no host stall): every peer's SDMA
    // push into this PE's output has now landed (all flags set + system fence
    // above) and the local-block collective gather (run earlier in this same CTA)
    // has completed. The bytes are fabric-coherent in HBM but the consumer GEMM
    // may still hold stale L2 lines for the reused FSDP output buffer, causing
    // E2E drift. Re-touch the full local output with a volatile-glc load (bypass
    // stale L2, fetch fresh HBM) + normal store (re-publish into the CU/L2 domain
    // the GEMM reads). Fused in-kernel after the actual HW-completion flag-wait,
    // so no host round-trip. Env-gated (MORI_HIER_FUSE_REMOTE_RETOUCH); default
    // OFF keeps the path byte-identical.
    if (args.retouchOut != 0) {
      // A `volatile` pointer load on CDNA3 (gfx942) is an agent-scope cached load --
      // it reads the same possibly-stale L2 lines the consumer GEMM holds and stores
      // them straight back, re-publishing stale bytes instead of the intended
      // "bypass stale L2, fetch fresh HBM". The SDMA/XGMI reassembly pushes wrote
      // fresh bytes to HBM bypassing L2, so the coherent copy lives in HBM. Fetch it
      // with a system-scope load (glc/dlc == miss the agent caches, read HBM) and
      // re-publish with a system-scope store (write-through == invalidate the stale
      // L2 copy system-wide) so the consumer GEMM's next read sees the true landed
      // bytes. Bit-exact by construction: reads then writes the identical
      // HBM-coherent word, changes no value.
      uint32_t* out = reinterpret_cast<uint32_t*>(args.gDstMemObj->localPtr);
      const size_t totalOut = static_cast<size_t>(args.numNodes) *
                              static_cast<size_t>(args.groupSize) * args.gElementCount;
      for (size_t i = threadIdx.x; i < totalOut; i += blockDim.x) {
        uint32_t v = mori::core::AtomicLoadRelaxedSystem(out + i);
        mori::core::AtomicStoreRelaxedSystem(out + i, v);
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
    // FIRST-LAND idle-engine reclamation (see HierLocalOffload): worker j==0, before
    // it starts spinning on the first inter-node landing flag, issues the offloaded
    // local-block peer-columns [G-offloadPeers, G) that the queue-0 local CTA skipped.
    // PUSH-ONLY on queue 0 (per-peer signal slots disjoint from the local CTA's
    // remaining peers => no counter race), setting flag[groupPos] on each target peer
    // (local flag region [0,G), flagBase=0). This runs during the inter-fill window
    // where this CTA's queue-1 reassembly is otherwise idle, so the queue-0 local
    // gather shrinks by offloadPeers/G off the co-critical path. Bit-exact: the
    // union of pushed local columns is unchanged; only the issuing CTA/moment for
    // the last K columns differs. offloadPeers==0 => skipped (byte-identical).
    if (j == 0 && args.offloadPeers > 0 && args.groupSize > args.offloadPeers) {
      OneShotSubGroupPushOnly_body<uint32_t>(
          args.groupSize, args.groupPos, args.gPeBase, args.gPeStride, args.gInput,
          args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
          args.gDstSlotStrideBytes, args.gFlagVal, /*flagBase=*/0, /*qId=*/0,
          /*deepSqPhase=*/0, /*qSplit=*/0, /*qStride=*/1, /*fuseFence=*/0, /*ndesc=*/1,
          /*pushRotate=*/0, /*qFlag=*/0, /*multiQueue=*/0, /*pushPhased=*/0,
          /*pushPeerLo=*/args.groupSize - args.offloadPeers, /*pushPeerHi=*/args.groupSize);
    }
    // Worker j of effReasm total workers (stride effReasm); under elastic the
    // local CTA is the extra worker j==reasm. qId=j+1 (1..reasm) is distinct from
    // the local CTA's queue 0. Byte-identical to the prior stride==reasm path when
    // elastic is OFF (effReasm==reasm).
    FusedRemoteReassembleWorker(args, partition, /*j=*/j, /*stride=*/effReasm, /*qId=*/j + 1,
                                spatialSplit);
  }
}

// ---------------------------------------------------------------------------
// Device-side landing gate-op
// ---------------------------------------------------------------------------
// On this MI300X/mlx5, kernel-return + on-stream CUDA events do not capture
// SDMA/RDMA hardware completion -- the async copy-engine / RDMA send WQEs outlive
// the enqueuing kernel, so the FSDP consumer reading on the same stream can race
// the still-in-flight landing. Only a host hipStreamSynchronize -- which drains
// the SDMA HSA-signal queues + the RDMA CQ hardware queues -- is bit-exact, but it
// stalls the CPU enqueue of the rest of the step.
//
// This gate is the device equivalent of that host drain, with no CPU round-trip:
// a tiny 1-block kernel launched on the comm stream after the big AG's ring/gather
// kernels. ShmemQuietThread<RDMA>() is the live drain (DrainToLive=true) -- it
// spins on the mlx5 CQ (Mlx5CollapsedCqDrain) for every peer/QP until every
// reserved RDMA WQE has completed, i.e. every inter-node write this rank posted
// has landed at the remote NIC. __threadfence_system() then system-orders those
// landed bytes before the kernel returns, so the FSDP-recorded completion event
// on this stream coincides with the actual hardware landing. Single thread runs
// the (serial) per-QP drain loop, mirroring the ring kernels' quiet calls.
extern "C" __global__ void DeviceLandingGateKernel_u32(int dummy) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    mori::shmem::ShmemQuietThread();
    __threadfence_system();
  }
}
