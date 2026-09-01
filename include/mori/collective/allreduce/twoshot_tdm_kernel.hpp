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
#pragma once

// Two-shot AllReduce whose cross-GPU bytes move through the TDM engine instead of the SDMA copy
// engines. gfx125x only -- the whole file is reached from inside a
// `#if defined(__gfx1250__) || defined(__gfx1251__)' in ccl_kernels.hip, so nothing here tests the
// target architecture.
//
// WHAT IS THE SAME AS twoshot_sdma_kernel.hpp, deliberately: the buffer layout (npes receive slots
// followed by one reduced slot, all at slotStrideElements), the flag protocol (monotonic generation
// tokens in flagsMemObj, index [0,npes) for data-landed and [npes,2*npes) for the scratch-reuse
// handshake), the CrossPeBarrier generation counter, the local reduce loop, and every host-visible
// argument. That is what lets the Python layer pick a transport per call against the same handle,
// the same symmetric buffers and the same CclAllreduceArgs<T>.
//
// WHAT IS DIFFERENT: SDMA moves a whole chunk with one descriptor submitted by one lane, so the
// SDMA kernel does its scatter on block 0 alone and leaves every other block parked on a broadcast.
// TDM has no such form. Its two instructions are tensor_load_to_lds and tensor_store_from_lds, so
// one endpoint is always LDS and a cross-GPU copy is always local-global -> LDS -> remote-global,
// one tile at a time, with the tile as a per-warp resource. Filling the fabric therefore takes many
// independent warps rather than a few deep queues, which is why the scatter here runs on the whole
// grid and needs a grid-wide rendezvous before any peer flag may go up.
//
// SCOPE OF THIS VERSION: two-shot only. The fused single-launch path that the SDMA kernel takes for
// messages at or below 1 MB is not implemented, and the Python layer keeps sending small messages to
// SDMA. The LDS round trip is a fixed cost per tile, so it is worst exactly where the message is too
// small to amortize it, which is the range where SDMA's one-descriptor-per-chunk form is already
// efficient. Fusing would also need a third grid-wide rendezvous inside one kernel, and an LDS or
// barrier mistake here does not produce a wrong answer, it produces an unkillable D-state process.

#include <hip/amd_detail/amd_gfx1250_TDM.h>
#include <hip/hip_runtime.h>

#include <cstddef>

#include "mori/collective/allreduce/twoshot_sdma_common.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/intra_node/kernels/vec_type.cuh"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// Tile geometry, in 4-byte elements. Everything this kernel moves is a byte copy, so the element
// type is uint32 regardless of T; GROUP1's dataSize is a 2-bit log2 that only encodes 1/2/4 bytes,
// making 4 both the widest it can express and the one needing the fewest descriptors.
//
// The ROW WIDTH is the load-bearing number, and it is a measured bandwidth result rather than a
// legality rule: a narrow row costs roughly 3x (TdmCheapDim1's derivation in
// src/ops/dispatch_combine/intranode_1250x.hpp records ~500 GB/s at a 224 B row against ~1500 at
// 256 B). 256 elements is a 1024 B row, well clear of the floor.
#define MORI_CCL_TDM_TILE_W 256
#define MORI_CCL_TDM_TILE_H 8
#define MORI_CCL_TDM_TILE_ELEMS (MORI_CCL_TDM_TILE_W * MORI_CCL_TDM_TILE_H)
#define MORI_CCL_TDM_TILE_BYTES (MORI_CCL_TDM_TILE_ELEMS * 4)

// Dynamic LDS this kernel needs, which the host must reserve. MUST AGREE WITH _tdm_shared_mem() in
// python/mori/ccl/collective.py: reserving less than a block indexes is an LDS overrun, and an LDS
// overrun on this hardware is an unkillable D-state process, not an error return. Rounds up so a
// blockDim that is not a whole number of waves still has a tile for its last partial wave.
inline __host__ __device__ size_t CclTdmSharedMemBytes(int threadsPerBlock, int waveSize) {
  const int warps = (threadsPerBlock + waveSize - 1) / waveSize;
  return static_cast<size_t>(warps) * MORI_CCL_TDM_TILE_BYTES;
}

namespace detail {

// GROUP1 for a dense 2D tile of 4-byte elements. Same construction as TdmShape2D in
// src/ops/dispatch_combine/intranode_1250x.hpp; duplicated rather than shared because that header
// is an EP-private include that also requires MAX_GPUS_PER_NODE and the EP barrier from its
// includer.
__device__ __forceinline__ gfx1250_TDM_GROUP1 CclTdmShape2D(int dim0, int dim1) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(2);
  g1.tensorDim0(dim0);
  g1.tensorDim1(dim1);
  g1.tensorDim0Stride(dim0);
  g1.tensorDim1Stride(dim1);
  g1.tileDim0(dim0);
  g1.tileDim1(dim1);
  return g1;
}

// Both issue functions are executed by every lane of the warp with warp-uniform arguments. The
// engine takes its descriptor from scalar registers and the operation is per-wave, which is why the
// EP payload loop also calls its equivalents without a lane guard.
__device__ __forceinline__ void CclTdmIssueLoad(uint32_t* ldsTile, const uint32_t* src,
                                                const gfx1250_TDM_GROUP1& g1) {
  typedef int _v4i __attribute__((ext_vector_type(4)));
  typedef int _v8i __attribute__((ext_vector_type(8)));
  gfx1250_TDM_GROUP0 g0;
  g0.ldsAddr(reinterpret_cast<uintptr_t>(ldsTile));
  g0.globalAddr(reinterpret_cast<uintptr_t>(src));
  _v4i z4{0, 0, 0, 0};
  _v8i z8{0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
}

__device__ __forceinline__ void CclTdmIssueStore(uint32_t* dst, uint32_t* ldsTile,
                                                 const gfx1250_TDM_GROUP1& g1) {
  typedef int _v4i __attribute__((ext_vector_type(4)));
  typedef int _v8i __attribute__((ext_vector_type(8)));
  gfx1250_TDM_GROUP0 g0;
  g0.ldsAddr(reinterpret_cast<uintptr_t>(ldsTile));
  g0.globalAddr(reinterpret_cast<uintptr_t>(dst));
  _v4i z4{0, 0, 0, 0};
  _v8i z8{0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_store_from_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
}

// One warp copies count4 four-byte elements from local global memory into a peer's global memory,
// bouncing through its own LDS tile.
//
// ONLY s_wait_tensorcnt(0) IS USED. Partial waits would remove a drain per tile and are not an
// option: the tensor queue does not retire in issue order. That was established by measurement in
// benchmark/cco/ualoe_metric/ualoe_bw.cpp, where a deep-pipelined variant built on the in-order
// assumption corrupted data at every buffer count its derivation called safe, including
// configurations with large slack; that file keeps the variant as an explicit refutation.
//
// The tail is split twice, so only the sub-row remainder leaves the TDM path. count4 is always a
// multiple of 4 (it derives from elementCountPerRank, itself a multiple of 16/sizeof(T)), which is
// what makes the remainder uint4-copyable.
__device__ __forceinline__ void CclTdmWarpCopy(uint32_t* __restrict__ dstPeer,
                                               const uint32_t* __restrict__ src, size_t count4,
                                               uint32_t* ldsTile, int laneId, int waveSize) {
  const gfx1250_TDM_GROUP1 gFull = CclTdmShape2D(MORI_CCL_TDM_TILE_W, MORI_CCL_TDM_TILE_H);
  size_t done = 0;

  while (done + MORI_CCL_TDM_TILE_ELEMS <= count4) {
    CclTdmIssueLoad(ldsTile, src + done, gFull);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    CclTdmIssueStore(dstPeer + done, ldsTile, gFull);
    // Cannot be deferred past the next trip: the tile is the only staging this warp owns.
    __builtin_amdgcn_s_wait_tensorcnt(0);
    done += MORI_CCL_TDM_TILE_ELEMS;
  }

  size_t rem = count4 - done;
  const int rows = static_cast<int>(rem / MORI_CCL_TDM_TILE_W);
  if (rows > 0) {
    const gfx1250_TDM_GROUP1 gRows = CclTdmShape2D(MORI_CCL_TDM_TILE_W, rows);
    CclTdmIssueLoad(ldsTile, src + done, gRows);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    CclTdmIssueStore(dstPeer + done, ldsTile, gRows);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    done += static_cast<size_t>(rows) * MORI_CCL_TDM_TILE_W;
    rem = count4 - done;
  }

  if (rem > 0) {
    using Uint4 = uint32_t __attribute__((ext_vector_type(4)));
    const size_t nVec = rem >> 2;
    const Uint4* srcV = reinterpret_cast<const Uint4*>(src + done);
    Uint4* dstV = reinterpret_cast<Uint4*>(dstPeer + done);
    for (size_t i = laneId; i < nVec; i += waveSize) dstV[i] = srcV[i];
  }
}

// Every block runs this handshake independently rather than block 0 running it and broadcasting.
// Both the publish and the wait are on a monotonic generation token, so repeating them across
// blocks is idempotent, and the property each block needs -- that peer P has finished the read that
// the previous operation left outstanding -- is established by P's token alone. Broadcasting it
// would need a second device-scope flag, and CrossPeBarrier cannot grow a field: the host class
// that sizes it is compiled AOT into libmori_collective while these kernels are JIT-compiled, so a
// larger struct would have the kernel writing past what an already-built host allocated.
__device__ __forceinline__ void CclTdmPeerTokenRendezvous(
    const application::SymmMemObjPtr flagsMemObj, const uint64_t* flags, size_t flagBase,
    uint64_t token, int myPe, int npes, int warpId, int laneId) {
  const bool owner = (warpId < npes) && (laneId == 0) && (warpId != myPe);
  if (owner) PublishControlFlag(flagsMemObj, flagBase + myPe, token, warpId);
  // Separating publish from wait means every peer this block is responsible for has been notified
  // before any warp starts spinning, which is how the SDMA kernel orders the same rendezvous.
  __syncthreads();
  if (owner) {
    while (LoadControlFlag(flags + flagBase + warpId) < token);
  }
  __syncthreads();
}

}  // namespace detail

// ============================================================================
// TdmReduceScatterKernel — TDM scatter on the whole grid, then local reduce
//
//   Phase 1 (all blocks):  every warp copies tiles of input[destPe] into
//                          destPe's receive slot myPe, through its LDS tile.
//   Phase 2 (grid, then block 0): grid-wide rendezvous so no flag can rise
//                          before every block's tiles are published, then
//                          block 0 raises the data-landed token on each peer
//                          and waits for theirs.
//   Phase 3 (all blocks):  local reduce out of the receive slots into the
//                          reduced slot. Byte-identical to the SDMA kernel.
//
// Signature matches SdmaReduceScatterKernel_body minus its optional `output',
// so CCL_WRAP_ALLREDUCE_RS in ccl_kernels.hip wraps it unchanged.
// Requirement, inherited from the SDMA kernel: gridDim.x <= multiProcessorCount,
// because Phase 2 spins on blocks being co-resident.
// ============================================================================
template <typename T>
__device__ void TdmReduceScatterKernel_body(int myPe, int npes, const T* __restrict__ input,
                                            const application::SymmMemObjPtr dstMemObj,
                                            const application::SymmMemObjPtr flagsMemObj,
                                            CrossPeBarrier* __restrict__ barrier,
                                            size_t elementCount, size_t slotStrideElements) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  if (elementCountPerRank == 0) return;

  const size_t bytesPerElement = sizeof(T);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t slotStrideBytes = slotStrideElements * bytesPerElement;
  const size_t slotStridePacked = slotStrideElements / pack_size;
  const size_t chunk4 = chunkBytes >> 2;

  const int waveSize = warpSize;
  const int warpId = static_cast<int>(threadIdx.x) / waveSize;
  const int laneId = static_cast<int>(threadIdx.x) % waveSize;
  const int warpsPerBlock = (static_cast<int>(blockDim.x) + waveSize - 1) / waveSize;

  extern __shared__ char _cclTdmSmem[];
  uint32_t* ldsTile = reinterpret_cast<uint32_t*>(_cclTdmSmem) +
                      static_cast<size_t>(warpId) * MORI_CCL_TDM_TILE_ELEMS;

  __shared__ uint64_t s_next;
  __shared__ uint64_t s_reuse;
  __shared__ uint32_t s_needs_reuse;
  if (threadIdx.x == 0) {
    s_next = barrier->flag + 1ULL;
    s_reuse = barrier->reuseGeneration + 1ULL;
    s_needs_reuse = barrier->needsReuseHandshake && chunkBytes > barrier->reuseSafeChunkBytes;
  }
  __syncthreads();

  const uint64_t* __restrict__ flags = reinterpret_cast<const uint64_t*>(flagsMemObj->localPtr);

  // --- scratch-reuse handshake, before any block writes a receive slot ------
  // Same contract as the SDMA kernel: arriving here proves this rank's previous copy-out is done
  // because the stream is ordered, so all that is left to establish is the same about each peer.
  if (s_needs_reuse) {
    detail::CclTdmPeerTokenRendezvous(flagsMemObj, flags, static_cast<size_t>(npes), s_reuse, myPe,
                                      npes, warpId, laneId);
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->reuseGeneration, s_reuse, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_DEVICE);
    }
  }

  // === Phase 1: TDM scatter, whole grid ====================================
  // Destination order is rotated by rank so that at any instant the ranks are driving different
  // peers rather than all piling onto the same one. The same effect was measured from the other
  // side: a combine push reached 782 GB/s with only 2-3 of the links driven at once, while the
  // links themselves were not saturated.
  {
    const size_t globalWarpId =
        static_cast<size_t>(blockIdx.x) * warpsPerBlock + static_cast<size_t>(warpId);
    const size_t totalWarps = static_cast<size_t>(gridDim.x) * warpsPerBlock;
    const size_t tilesPerChunk =
        (chunk4 + MORI_CCL_TDM_TILE_ELEMS - 1) / MORI_CCL_TDM_TILE_ELEMS;

    for (int step = 1; step < npes; ++step) {
      const int destPe = (myPe + step) % npes;
      const uint32_t* srcBase = reinterpret_cast<const uint32_t*>(
          reinterpret_cast<const uint8_t*>(input) + static_cast<size_t>(destPe) * chunkBytes);
      // p2pPeerPtrs (what GetAs(pe) returns), not peerPtrs: TDM's global endpoint is an ordinary
      // GPU virtual address, and this is the array holding the directly addressable peer mapping.
      // The SDMA kernel reads peerPtrs because that is what its descriptors are built from.
      uint32_t* dstBase = reinterpret_cast<uint32_t*>(
          dstMemObj->GetAs<uint8_t*>(destPe) + static_cast<size_t>(myPe) * slotStrideBytes);

      for (size_t tile = globalWarpId; tile < tilesPerChunk; tile += totalWarps) {
        const size_t off = tile * MORI_CCL_TDM_TILE_ELEMS;
        const size_t n =
            (off + MORI_CCL_TDM_TILE_ELEMS <= chunk4) ? MORI_CCL_TDM_TILE_ELEMS : (chunk4 - off);
        detail::CclTdmWarpCopy(dstBase + off, srcBase + off, n, ldsTile, laneId, waveSize);
      }
    }
  }

  // === Phase 2: grid rendezvous, then cross-PE tokens ======================
  // s_wait_tensorcnt(0) retired the engine; it does not order memory. The system-scope fence is
  // what makes this block's tiles visible to a peer, and __syncthreads() is what makes one
  // thread's fence cover the whole block's stores.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    __scoped_atomic_store_n(&barrier->blockDone[blockIdx.x], s_next, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_DEVICE);
  }

  if (blockIdx.x == 0) {
    if (threadIdx.x < gridDim.x) {
      while (__scoped_atomic_load_n(&barrier->blockDone[threadIdx.x], __ATOMIC_ACQUIRE,
                                    __MEMORY_SCOPE_DEVICE) < s_next);
    }
    __syncthreads();

    if (warpId < npes && laneId == 0 && warpId != myPe) {
      PublishControlFlag(flagsMemObj, myPe, s_next, warpId);
    }
    __syncthreads();

    if (warpId < npes && laneId == 0 && warpId != myPe) {
      const int sender = warpId;
      int spin = 0;
      bool warned = false;
      while (LoadControlFlag(flags + sender) < s_next) {
        if (++spin > 100000000 && !warned) {
          printf("PE %d: TdmScatter timeout waiting for peer %d\n", myPe, sender);
          warned = true;
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->flag, s_next, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
  } else {
    if (threadIdx.x == 0) {
      while (__scoped_atomic_load_n(&barrier->flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
             s_next);
    }
    __syncthreads();
  }
  // Peers published their tiles with a system-scope release; match it before reading them.
  __threadfence_system();

  // === Phase 3: local reduce (all blocks) ==================================
  // Byte-identical to SdmaReduceScatterKernel_body's Phase 3, including taking the local
  // contribution straight from input rather than self-transferring it.
  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ reduced = buf + static_cast<size_t>(npes) * slotStridePacked;
  const P* __restrict__ inputSlot =
      reinterpret_cast<const P*>(input) + static_cast<size_t>(myPe) * packedPerRank;

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t k = tid; k < packedPerRank; k += stride) {
    A acc = upcast_v<typename P::type, pack_size>(myPe == 0 ? inputSlot[k] : buf[k]);
    for (int pe = 1; pe < npes; ++pe) {
      const P value =
          pe == myPe ? inputSlot[k] : buf[static_cast<size_t>(pe) * slotStridePacked + k];
      packed_assign_add(acc, upcast_v<typename P::type, pack_size>(value));
    }
    SdmaVisibleStore(reduced + k, downcast_v<typename P::type, pack_size>(acc));
  }
}

template <typename T>
__global__ void TdmReduceScatterKernel(int myPe, int npes, const T* __restrict__ input,
                                       const application::SymmMemObjPtr dstMemObj,
                                       const application::SymmMemObjPtr flagsMemObj,
                                       CrossPeBarrier* __restrict__ barrier, size_t elementCount,
                                       size_t slotStrideElements) {
  TdmReduceScatterKernel_body<T>(myPe, npes, input, dstMemObj, flagsMemObj, barrier, elementCount,
                                 slotStrideElements);
}

// ============================================================================
// TdmAllGatherKernel — second shot: publish the reduced shard to every peer
//
// The source is always the reduced slot, so unlike AllGatherSdmaKernel_body
// there is no sourceOffsetElements parameter: this kernel exists only to finish
// an AllReduce. Its signature therefore matches CCL_WRAP_ALLREDUCE_AG.
//
// The destination arithmetic, including the outputBaseOffsetBytes shear, is
// taken from the SDMA kernel and means the same thing: a nonzero base lies
// wholly inside slot-0 padding and cannot land on a peer's receive data, while
// a compact output at base zero can, and therefore needs the ready rendezvous
// first.
// ============================================================================
template <typename T>
__device__ void TdmAllGatherKernel_body(int myPe, int npes,
                                        const application::SymmMemObjPtr srcMemObj,
                                        const application::SymmMemObjPtr dstMemObj,
                                        const application::SymmMemObjPtr flagsMemObj,
                                        CrossPeBarrier* __restrict__ barrier, size_t elementCount,
                                        size_t slotStrideElements, size_t outputBaseOffsetBytes) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  if (elementCountPerRank == 0) return;

  const size_t bytesPerElement = sizeof(T);
  const size_t agSendBytes = elementCountPerRank * bytesPerElement;
  const size_t send4 = agSendBytes >> 2;

  const int waveSize = warpSize;
  const int warpId = static_cast<int>(threadIdx.x) / waveSize;
  const int laneId = static_cast<int>(threadIdx.x) % waveSize;
  const int warpsPerBlock = (static_cast<int>(blockDim.x) + waveSize - 1) / waveSize;

  extern __shared__ char _cclTdmSmem[];
  uint32_t* ldsTile = reinterpret_cast<uint32_t*>(_cclTdmSmem) +
                      static_cast<size_t>(warpId) * MORI_CCL_TDM_TILE_ELEMS;

  const uint64_t* __restrict__ flags = reinterpret_cast<const uint64_t*>(flagsMemObj->localPtr);
  __shared__ uint64_t s_token;
  if (threadIdx.x == 0) s_token = static_cast<uint64_t>(barrier->flag) + 1ULL;
  __syncthreads();
  const uint64_t ready_val = s_token;
  const uint64_t done_val = s_token + 1ULL;

  if (outputBaseOffsetBytes == 0) {
    detail::CclTdmPeerTokenRendezvous(flagsMemObj, flags, 0, ready_val, myPe, npes, warpId, laneId);
  }

  // --- TDM put: my reduced shard to every peer, whole grid ------------------
  {
    const size_t globalWarpId =
        static_cast<size_t>(blockIdx.x) * warpsPerBlock + static_cast<size_t>(warpId);
    const size_t totalWarps = static_cast<size_t>(gridDim.x) * warpsPerBlock;
    const size_t tilesPerChunk = (send4 + MORI_CCL_TDM_TILE_ELEMS - 1) / MORI_CCL_TDM_TILE_ELEMS;

    const uint32_t* srcBase = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(srcMemObj->localPtr) +
        static_cast<size_t>(npes) * slotStrideElements * bytesPerElement);

    // Whether a rank also sends to ITSELF is decided by the output base, and the two cases are
    // opposite. This mirrors the SDMA kernel's `outputBaseOffsetBytes == 0 || warpId != myPe'.
    //
    // A nonzero base is npes*slotStride - myPe*agSendBytes, so this rank's own shard lands at
    // base + myPe*agSendBytes == npes*slotStride, which is exactly where the reduced slot already
    // is. The copy would be the identity, so it is skipped. At base zero the output region starts
    // at offset 0 instead, the two no longer coincide, and skipping it would leave this rank's
    // shard missing from its own output.
    for (int step = 0; step < npes; ++step) {
      const int remotePe = (myPe + step) % npes;
      if (outputBaseOffsetBytes != 0 && remotePe == myPe) continue;

      const size_t remoteOutputBase =
          outputBaseOffsetBytes == 0
              ? 0
              : static_cast<size_t>(static_cast<ptrdiff_t>(outputBaseOffsetBytes) +
                                    static_cast<ptrdiff_t>(myPe - remotePe) *
                                        static_cast<ptrdiff_t>(agSendBytes));
      // localPtr rather than the peer mapping for the self case, so this does not depend on
      // p2pPeerPtrs[myPe] being populated with this rank's own address.
      uint8_t* dstRoot = (remotePe == myPe) ? reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                                            : dstMemObj->GetAs<uint8_t*>(remotePe);
      uint32_t* dstBase = reinterpret_cast<uint32_t*>(
          dstRoot + remoteOutputBase + static_cast<size_t>(myPe) * agSendBytes);

      for (size_t tile = globalWarpId; tile < tilesPerChunk; tile += totalWarps) {
        const size_t off = tile * MORI_CCL_TDM_TILE_ELEMS;
        const size_t n =
            (off + MORI_CCL_TDM_TILE_ELEMS <= send4) ? MORI_CCL_TDM_TILE_ELEMS : (send4 - off);
        detail::CclTdmWarpCopy(dstBase + off, srcBase + off, n, ldsTile, laneId, waveSize);
      }
    }
  }

  // --- grid rendezvous, then the done handshake ----------------------------
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    __scoped_atomic_store_n(&barrier->blockDone[blockIdx.x], done_val, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_DEVICE);
  }
  if (blockIdx.x != 0) return;

  if (threadIdx.x < gridDim.x) {
    while (__scoped_atomic_load_n(&barrier->blockDone[threadIdx.x], __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) < done_val);
  }
  __syncthreads();

  if (warpId < npes && laneId == 0 && warpId != myPe) {
    PublishControlFlag(flagsMemObj, myPe, done_val, warpId);
  }
  __syncthreads();
  if (warpId < npes && laneId == 0 && warpId != myPe) {
    const int sender = warpId;
    int spin = 0;
    bool warned = false;
    while (LoadControlFlag(flags + sender) < done_val) {
      if (++spin > 100000000 && !warned) {
        printf("PE %d: TdmAllGather timeout waiting for peer %d\n", myPe, sender);
        warned = true;
      }
    }
  }
  __syncthreads();
  __threadfence_system();

  if (threadIdx.x == 0) {
    barrier->flag = done_val;
    barrier->needsReuseHandshake = 1;
    const size_t slotStrideBytes = slotStrideElements * bytesPerElement;
    barrier->reuseSafeChunkBytes =
        outputBaseOffsetBytes == 0 ? 0
                                   : slotStrideBytes - static_cast<size_t>(npes - 1) * agSendBytes;
  }
}

template <typename T>
__global__ void TdmAllGatherKernel(int myPe, int npes, const application::SymmMemObjPtr srcMemObj,
                                   const application::SymmMemObjPtr dstMemObj,
                                   const application::SymmMemObjPtr flagsMemObj,
                                   CrossPeBarrier* __restrict__ barrier, size_t elementCount,
                                   size_t slotStrideElements, size_t outputBaseOffsetBytes) {
  TdmAllGatherKernel_body<T>(myPe, npes, srcMemObj, dstMemObj, flagsMemObj, barrier, elementCount,
                             slotStrideElements, outputBaseOffsetBytes);
}

}  // namespace collective
}  // namespace mori
