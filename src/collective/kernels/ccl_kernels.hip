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

// L2-COHERENT RE-TOUCH (Phase-5, Team A -- targets the barriercutouch residual).
// barriercutouch composes shmem_barrier_on_stream (orders the RDMA cross-PE
// landing + intra SDMA gather) with a CU re-touch (torch.add(out,0)) and cut the
// olapfast completion drift ~24x (dL 0.054 -> 0.0023) but never reached bit-exact:
// the residual dL 0.0023 is the SDMA copy-engine-written output bytes not being
// L2-coherent to the consumer GEMM under FSDP buffer reuse. The plain add-0
// re-touch FAILS to fix it because torch.add issues AGENT-scope, cacheable loads
// -- it reads the SAME STALE L2 line the previous step's GEMM cached for this
// reused buffer, adds 0, and writes the stale value straight back. This kernel
// does a SYSTEM-SCOPE ACQUIRE load (glc+dlc: bypass/invalidate L2, fetch the
// fabric-coherent HBM value the SDMA/RDMA agents actually wrote) followed by a
// SYSTEM-SCOPE RELEASE store, so the LAST writer of the consumed buffer is a CU
// op AND the value it re-publishes is the freshly-landed one, not a stale cache
// line. Stream-ordered after the barrier => no host stall (keeps overlap). Rewrite
// of the same value is bit-exact for every dtype. Targets ONLY the big cross-node
// AGs where the residual lives.
// NOTE: system-scope __hip_atomic_* faults/hangs on gfx942 for coarse-grained
// torch CUDA memory (they require fine-grained/coherent allocations). Instead use
// a VOLATILE load, which the CDNA backend lowers to a glc-tagged global load that
// BYPASSES the L2 line -> reads the fabric-coherent HBM value the SDMA/RDMA agents
// wrote (skipping the stale L2 line the previous step's GEMM cached for this
// reused buffer), then a normal store re-publishes it into the CU/L2 domain the
// consumer GEMM reads. Safe on coarse-grained memory, no atomics.
extern "C" __global__ void L2CoherentRetouchKernel_u32(
    uint32_t* __restrict__ out, uint64_t n) {
  volatile uint32_t* vp = reinterpret_cast<volatile uint32_t*>(out);
  uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = idx; i < n; i += stride) {
    uint32_t v = vp[i];
    out[i] = v;
  }
}

// L2InvRetouchKernel_u32 (Team A, director 2026-07-08T14:54Z sole lever):
// the CONCRETE consumer-coherence fix the volatile-only L2CoherentRetouchKernel
// above could NOT close. On gfx942 a `volatile` load lowers to a glc-tagged load
// that bypasses only the per-CU L1 vector cache -- it STILL HITS the stale line in
// the DEVICE-SCOPE L2 that the previous FSDP step's consumer GEMM cached for this
// reused output buffer, so the re-touch republishes the SAME stale value and the
// DEEP_PIPE device-fence path drifts (T55: retouch did not fix it). The real fix is
// a device-side L2 INVALIDATE. The prior author avoided system-scope work because
// __hip_atomic_* faults on coarse-grained torch memory -- but that is true only for
// atomic OPERATIONS (they need fine-grained/coherent allocations); a pure
// __builtin_amdgcn_fence is a CACHE-CONTROL barrier that touches no buffer word and
// is SAFE on any allocation. A system-scope ACQUIRE fence lowers on gfx942 to
// `buffer_inv sc0 sc1` (invalidate BOTH L1 and L2), so the subsequent loads MISS the
// stale L2 line and fetch the fabric-coherent HBM value the SDMA/RDMA agents landed.
// Both load and store go through the glc `volatile` alias so no dirty L2 line exists
// for a concurrent wave's whole-L2 invalidate to drop; a trailing system-scope
// RELEASE fence (`buffer_wbinvl2`/writeback + s_waitcnt) drains every store to HBM
// before the kernel completes, so the consumer GEMM (stream-ordered after) reads the
// freshly-landed bytes. Rewrite of the identical value is bit-exact for every dtype.
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

// L2InvOnlyKernel_u32 (Team A, director 2026-07-08T14:54Z lever — CHEAP variant):
// the DEEP_PIPE consumer-coherence fix WITHOUT the full-buffer rewrite. The stale
// data lives in the DEVICE-SCOPE L2 (shared across all CUs); the freshly-landed
// SDMA/RDMA bytes are already in HBM. A single system-scope ACQUIRE fence lowers to
// `buffer_inv sc0 sc1` on gfx942 (VERIFIED by ISA disasm 2026-07-08) which drops the
// device L2 lines, so the stream-ordered consumer GEMM MISSES L2 and re-fetches the
// fabric-coherent HBM value. buffer_inv is a whole-cache op (not address-ranged) so
// no bulk traffic is needed — one wave per CU covers every per-CU L1 and the shared
// L2, at ~0 bytes moved (vs L2InvRetouch which rewrites every u32 ~= +7% E2E). The
// `out`/`n` args are accepted for launch-signature parity but unused. No writeback
// fence: this kernel writes nothing, so there are no dirty lines to drain.
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
                                  /*useRead=*/args.useRead != 0);
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
                                    /*bidOverride=*/static_cast<int>(blockIdx.x),
                                    /*usePutSignal=*/args.usePutSignal != 0,
                                    /*useWriteImm=*/args.useWriteImm != 0);
  } else {
    OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
        args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
        args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
        args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true);
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
// Wall-decomposition profiler landmarks (MORI_HIER_PROFILE). g_hierProf[1] is an
// atomicMax over reassembly workers of the wall_clock64() at which each ring
// channel's landing flag was OBSERVED (== when the last inter-node RDMA chunk
// became reassemblable); [0] kernel-entry reference, [2] completion-reader done,
// [3] local-gather done. All wall_clock64 (fixed-freq, GPU-uniform). Zeroed at
// module load; the completion reader resets [1] after reading so the atomicMax
// starts clean for the next launch.
__device__ unsigned long long g_hierProf[4] = {0, 0, 0, 0};

// ``partition`` = number of disjoint 16B-aligned byte sub-ranges the chunk is
// split into for the intra-node SDMA reassembly push. Worker ``j`` (stride
// ``stride``) owns parts f=j, j+stride, ... . ``spatial`` selects the LANDING
// FLAG each part waits on: for the temporal deep-pipe (spatial=false) part f
// waits on chunkReadyFlags[f] (its own sub-chunk landed); for the SPATIAL split
// at a single-channel ring (spatial=true) EVERY part waits on the ONE landing
// flag chunkReadyFlags[0] (the whole chunk landed), so multiple SDMA engines
// reassemble disjoint sub-ranges of that one landed chunk CONCURRENTLY --
// shrinking the serial intra_reasm_tail (HIERPROF ~40-45% of the giant-AG wall)
// without changing the inter-node ring (no new inter race). Flag SLOT region is
// per-part (flagBase uses ``f``) so concurrent parts never race a shared slot.
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
  for (int f = j; f < partition; f += stride) {
    size_t startUnit = static_cast<size_t>(f) * unitsPerChan;
    size_t endUnit = startUnit + unitsPerChan;
    if (endUnit > nUnits) endUnit = nUnits;
    const size_t subOff = startUnit * kAlign;
    size_t subEnd = endUnit * kAlign;
    if (subEnd > args.chunkBytes) subEnd = args.chunkBytes;
    const size_t subBytes = (subOff < subEnd) ? (subEnd - subOff) : 0;
    if (threadIdx.x == 0) {
      int spin = 0;
      const int flagIdx = spatial ? 0 : f;
      while (mori::core::AtomicLoadSeqCstSystem(args.chunkReadyFlags + flagIdx) <
             static_cast<uint64_t>(1)) {
        if (++spin > 100000000) break;
      }
      __threadfence_system();
      // PROFILE: record when THIS channel's inter-node RDMA landing flag was
      // observed. atomicMax across all workers/channels => the LAST inter chunk
      // land time (the inter-node RDMA-fill completion boundary).
      if (args.profile != 0) {
        atomicMax(&g_hierProf[1], static_cast<unsigned long long>(wall_clock64()));
      }
    }
    __syncthreads();
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
          args.gFlagVal, flagBase, /*qId=*/qId);
      ++mi;
    }
  }
}

// PHASE 4: pipelined ring + REMOTE-block reassembly (see CclFusedRingRemoteGather
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
  // DEEP-SQ TEMPORAL PIPELINE: when deepPipe>1 (only on rb==1) the single ring
  // channel splits the chunk into ``pipe`` TEMPORAL sub-chunks each with its own
  // landing flag, so the reassembly workers + completion reader iterate over
  // ``pipe`` sub-chunk flags instead of ``rb`` spatial channel flags. deepPipe<=1
  // => pipe==rb (byte-identical shipped path).
  const int pipe = args.deepPipe > 1 ? args.deepPipe : rb;
  const int reasm = args.reassemblyBlocks > 0 ? args.reassemblyBlocks : rb;
  // ELASTIC REASSEMBLY: the local-block CTA joins reassembly as an extra worker
  // (index == reasm) on queue 0 after its own gather, so the tail runs on
  // effReasm = reasm+1 SDMA queues. OFF => effReasm == reasm (byte-identical).
  const int effReasm = args.elasticReasm != 0 ? reasm + 1 : reasm;
  // SPATIAL REASSEMBLY SPLIT (Turn 51, A): on the single-channel giant-AG ring
  // (rb==1) with NO temporal deep-pipe, split the ONE landed chunk into
  // ``effReasm`` disjoint byte sub-ranges so effReasm SDMA engines reassemble it
  // concurrently -- directly attacks the ~40-45% serial intra_reasm_tail HIERPROF
  // localised, WITHOUT touching the inter ring (no new inter race, unlike rb>1
  // which grows the RDMA fill 1:1). Each part waits the single landing flag
  // chunkReadyFlags[0]. reasm>1 requires sdmaNumQueue>=reasm+1 (Python clamp).
  const bool spatialSplit = (rb == 1 && args.deepPipe <= 1 && effReasm > 1);
  // Partition count for the reassembly byte split: temporal pipe channels, or the
  // spatial effReasm sub-ranges when the spatial split engages.
  const int partition = spatialSplit ? effReasm : pipe;
  const unsigned bx = blockIdx.x;
  if (bx < static_cast<unsigned>(rb)) {
    // HOST-PROXY INTER (hostProxyInter): a host proxy owns the inter-node leg --
    // it posts sub-chunk p's cross-node RDMA writes into the ring buffer, drains
    // its send-CQ (the proven host-drain landing fence, bit-exact at 466MB where
    // the device per-sub-chunk signal races/crashes), and publishes
    // chunkReadyFlags[p] from the host. So the device ring-send blocks do NOTHING
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
    // Ring channel bx: publishes chunkReadyFlags[bx] on landing.
    AllGatherRingSubGroupKernelBody(args.ringPos, args.ringSize, args.ringPeBase, args.ringPeStride,
                                    args.ringMemObj, args.ringFlagsObj, args.chunkBytes, args.numQp,
                                    /*numBlocksOverride=*/rb, /*bidOverride=*/static_cast<int>(bx),
                                    /*usePutSignal=*/args.usePutSignal != 0,
                                    /*useWriteImm=*/args.useWriteImm != 0, args.chunkReadyFlags,
                                    /*opGen=*/0, /*useRead=*/false, /*wqeDepth=*/args.wqeDepth,
                                    /*deepPipe=*/args.deepPipe, /*deepPipeImm=*/args.deepPipeImm,
                                    /*deepPipeQuiet=*/args.deepPipeQuiet);
    }
  } else if (bx == static_cast<unsigned>(rb)) {
    // PROFILE: kernel-entry reference for the wall decomposition (this single CTA
    // is launched with all others, so its entry ~= kernel start).
    if (args.profile != 0 && threadIdx.x == 0) {
      g_hierProf[0] = static_cast<unsigned long long>(wall_clock64());
    }
    // Local block (m == nodeId): ring-independent, reads this PE's own input.
    // Collective gather on flag slots [0, G) -- a SINGLE collective (never nested),
    // so it cannot dead-lock.
    OneShotAllGatherSdmaSubGroupKernel_body<uint32_t>(
        args.myPe, args.npes, args.groupSize, args.groupPos, args.gPeBase, args.gPeStride,
        args.gInput, args.gDstMemObj, args.gFlagsObj, args.gElementCount, args.gDstBaseOffset,
        args.gDstSlotStrideBytes, args.gFlagVal, /*blockLocal=*/true, /*flagBase=*/0);
    // PROFILE: local-block SDMA gather done (ring-independent leg).
    if (args.profile != 0 && threadIdx.x == 0) {
      __threadfence();
      g_hierProf[3] = static_cast<unsigned long long>(wall_clock64());
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
    if (numRemote > 0 && args.chunkReadyFlags != nullptr) {
      uint64_t* rflags = reinterpret_cast<uint64_t*>(args.gFlagsObj->localPtr);
      const int G = args.groupSize;
      // PARALLEL completion reader (Turn-15): the flag slots (channel f, remote
      // node mi, sender g) occupy the CONTIGUOUS region [G, G + rb*numRemote*G):
      // idx = G + (f*numRemote + mi)*G + sg with (f,mi,sg) enumerating that range
      // densely. Previously thread 0 spun over all rb*numRemote*G slots SERIALLY,
      // so the whole-reassembly completion latency grew linearly with the ring-
      // channel count -- exactly the per-chunk handshake overhead that collapsed
      // the pipeline's overlap at rb>1. Fan the wait across the CTA's threads
      // (each thread polls a strided subset) so the completion tail is O(total /
      // blockDim) instead of O(total). Bit-exact: same slots, same threshold,
      // just concurrent; a single CTA-wide fence + barrier still orders every
      // landed byte before the finish.
      const size_t total = static_cast<size_t>(partition) * static_cast<size_t>(numRemote) *
                           static_cast<size_t>(G);
      for (size_t t = threadIdx.x; t < total; t += blockDim.x) {
        const size_t idx = static_cast<size_t>(G) + t;
        int spin = 0;
        while (mori::core::AtomicLoadSeqCstSystem(rflags + idx) < args.gFlagVal) {
          if (++spin > 100000000) break;
        }
      }
      __threadfence_system();
      // PROFILE: completion reader done == all intra-node SDMA reassembly pushes
      // have landed in this PE's output. Decompose the wall: t0=entry, t1=last
      // inter-node RDMA land, t2=reassembly drain done, t3=local-gather done.
      if (args.profile != 0 && threadIdx.x == 0) {
        unsigned long long t2 = static_cast<unsigned long long>(wall_clock64());
        g_hierProf[2] = t2;
        __threadfence();
        unsigned long long t0 = g_hierProf[0];
        unsigned long long t1 = g_hierProf[1];  // last inter-node land (atomicMax)
        unsigned long long t3 = g_hierProf[3];  // local-gather done
        unsigned long long tot = (t2 > t0) ? (t2 - t0) : 0;
        unsigned long long inter = (t1 > t0) ? (t1 - t0) : 0;
        unsigned long long intraTail = (t2 > t1) ? (t2 - t1) : 0;
        unsigned long long localLeg = (t3 > t0) ? (t3 - t0) : 0;
        if (args.myPe == 0 && tot > 0) {
          double interPct = 100.0 * static_cast<double>(inter) / static_cast<double>(tot);
          double intraPct = 100.0 * static_cast<double>(intraTail) / static_cast<double>(tot);
          double localPct = 100.0 * static_cast<double>(localLeg) / static_cast<double>(tot);
          printf(
              "[HIERPROF] chunkBytes=%llu nodes=%d rb=%d reasm=%d | wall=%llu cyc | "
              "inter_land=%llu (%.1f%%) intra_reasm_tail=%llu (%.1f%%) local_gather=%llu (%.1f%%)\n",
              static_cast<unsigned long long>(args.chunkBytes), args.numNodes, rb,
              args.reassemblyBlocks, tot, inter, interPct, intraTail, intraPct, localLeg, localPct);
        }
        // Reset the atomicMax slot so the next launch starts clean.
        g_hierProf[1] = 0;
      }
    }
    // CU-COHERENT RE-TOUCH (director-mandated fused copy-out, no host stall):
    // every peer's SDMA push into THIS PE's output has now landed (all flags set
    // + system fence above) AND the local-block collective gather (run earlier in
    // this same CTA) has completed. The bytes are fabric-coherent in HBM but the
    // consumer GEMM may still hold STALE L2 lines for the reused FSDP output
    // buffer -> the +0.018 E2E drift. Re-touch the full local output with a
    // volatile-glc load (bypass stale L2, fetch fresh HBM) + normal store
    // (re-publish into the CU/L2 domain the GEMM reads). Fused in-kernel after
    // the ACTUAL HW-completion flag-wait, so no host round-trip. Env-gated
    // (MORI_HIER_FUSE_REMOTE_RETOUCH); default OFF keeps the path byte-identical.
    if (args.retouchOut != 0) {
      volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(args.gDstMemObj->localPtr);
      const size_t totalOut = static_cast<size_t>(args.numNodes) *
                              static_cast<size_t>(args.groupSize) * args.gElementCount;
      for (size_t i = threadIdx.x; i < totalOut; i += blockDim.x) {
        uint32_t v = out[i];
        out[i] = v;
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
    // (NIC<->XGMI) pipeline: previously the reassembly waited for ALL ring flags
    // before pushing anything, collapsing the overlap (the Turn-38 regression).
    // The tile partition MIRRORS the ring's multiBlock tiling (numBlocks==rb,
    // 16B units) so chunkReadyFlags[f] guards precisely the bytes pushed here.
    // PUSH-ONLY (no cross-rank wait) => the blocks never dead-lock. Each block
    // uses SDMA queue qId=j+1 (distinct per block; local CTA uses queue 0), and
    // reuses that one queue serially across its channels (never raced).
    const int j = static_cast<int>(bx) - rb - 1;
    if (j < 0 || j >= reasm) return;
    // Worker j of effReasm total workers (stride effReasm); under elastic the
    // local CTA is the extra worker j==reasm. qId=j+1 (1..reasm) is distinct from
    // the local CTA's queue 0. Byte-identical to the prior stride==reasm path when
    // elastic is OFF (effReasm==reasm).
    FusedRemoteReassembleWorker(args, partition, /*j=*/j, /*stride=*/effReasm, /*qId=*/j + 1,
                                spatialSplit);
  }
}

// ---------------------------------------------------------------------------
// PHASE 5 (Team A) — device-side landing GATE-OP (director spec 2026-07-07 13:48Z)
// ---------------------------------------------------------------------------
// The campaign's decisive root-cause (T11): on this MI300X/mlx5, kernel-return +
// on-stream CUDA events do NOT capture SDMA/RDMA HARDWARE completion -- the async
// copy-engine / RDMA send WQEs outlive the enqueuing kernel, so the FSDP consumer
// reading on the SAME stream can race the still-in-flight landing (olapfast drift
// dL=-0.054). Only a HOST hipStreamSynchronize -- which drains the SDMA HSA-signal
// queues + the RDMA CQ hardware queues -- was bit-exact, but it stalls the CPU
// enqueue of the rest of the step (-22%).
//
// This gate is the DEVICE equivalent of that host drain, with NO CPU round-trip:
// a tiny 1-block kernel launched on the comm stream AFTER the big AG's ring/gather
// kernels. ShmemQuietThread<RDMA>() is the LIVE drain (DrainToLive=true) -- it
// spins on the mlx5 CQ (Mlx5CollapsedCqDrain) for EVERY peer/QP until every
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
