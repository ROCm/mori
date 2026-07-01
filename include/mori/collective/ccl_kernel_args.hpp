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

#include <cstddef>
#include <cstdint>

#include "mori/application/application_device_types.hpp"

namespace mori {
namespace collective {

struct CrossPeBarrier;

template <typename T>
struct CclAll2allArgs {
  int myPe;
  int npes;
  T* input;
  application::SymmMemObjPtr inputTransitMemObj;
  application::SymmMemObjPtr outputTransitMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t elementCount;
};

template <typename T>
struct CclAllgatherArgs {
  int myPe;
  int npes;
  T* input;
  application::SymmMemObjPtr srcMemObj;
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t elementCount;
  size_t dstBaseOffset;
  uint64_t flagVal;
  const size_t* splitSizes;
  const size_t* splitOffsets;
  size_t splitCount;
};

// Sub-group intra-node SDMA AllGather. The ``G`` local ranks of a
// node ({peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}) gather
// their shards over the SDMA copy engines; this PE is at position ``groupPos``.
// The destination buffer holds ``groupSize`` contiguous slots; member at
// position ``p`` writes its shard into slot ``p`` of every member. The flat
// whole-world gather is the special case groupSize=npes, groupPos=myPe,
// peBase=0, peStride=1.
template <typename T>
struct CclAllgatherSubGroupArgs {
  int myPe;
  int npes;
  int groupSize;
  int groupPos;
  int peBase;
  int peStride;
  T* input;
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t elementCount;
  size_t dstBaseOffset;
  // M5: per-peer destination SLOT STRIDE in bytes. The kernel writes
  // member ``p``'s shard into slot ``p`` of the destination; by default the
  // slots are packed contiguously (stride == elementCount*sizeof(T) == the copy
  // size). A non-zero ``dstSlotStrideBytes`` decouples the slot stride from the
  // copy size, so a SUB-RANGE (chunk) of a slice can be written into its final
  // strided position within a full-size block. This is the enabler for the
  // chunked inter/intra reassembly pipeline (overlap the remote-block gather of
  // chunk k with the inter ring of chunk k+1): each chunk copies elementCount
  // (= chunk) bytes per peer but lands at slot stride = full slice size. 0 keeps
  // the contiguous-slot contract byte-for-byte unchanged.
  size_t dstSlotStrideBytes;
  uint64_t flagVal;
};

// Sub-group intra-node SDMA broadcast. The root
// (group position 0 == global PE ``peBase``) holds a full buffer of
// ``elementCount`` u32 lanes in ``input`` and SDMA-copies it into the
// ``dstMemObj`` of every member of {peBase, peBase+peStride, ...,
// peBase+(groupSize-1)*peStride}, including itself. This is the intra-node
// placement phase of the hierarchical AllGather's leader-only variant: leader
// rings the inter-node RDMA exchange into a staging buffer, then broadcasts the
// full N*G output to its G local ranks over XGMI (~G x less NIC traffic than
// the every-rank-direct ring).
template <typename T>
struct CclBroadcastSubGroupArgs {
  int myPe;
  int groupSize;
  int groupPos;
  int peBase;
  int peStride;
  T* input;
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t elementCount;
  size_t dstBaseOffset;
  uint64_t flagVal;
};

template <typename T>
struct CclAllreduceArgs {
  int myPe;
  int npes;
  const T* input;
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  CrossPeBarrier* barrier;
  size_t elementCount;
};

// Inter-node RDMA ring AllGather. The ring buffer ``memObj`` holds
// ``ringSize`` contiguous chunks of ``chunkBytes`` each (chunk ``k`` at offset
// ``k * chunkBytes``); on entry only this PE's own chunk (slot ``ringPos``) is
// filled. After ``ringSize-1`` rounds every member holds all ``ringSize`` chunks
// in ring order. The per-element type is irrelevant to the byte-move ring, so
// this struct is not templated -- the kernel moves raw bytes (chunkBytes) over
// shmem (P2P within a node, RDMA across nodes).
//
// Sub-group support (M2b): the ring runs over an arithmetic sub-group of global
// PEs ``{peBase, peBase+peStride, ..., peBase+(ringSize-1)*peStride}``; this
// PE's position within that sub-group is ``ringPos``. The flat whole-world ring
// is just ``peBase=0, peStride=1, ringSize=npes, ringPos=myPe``. The sub-group
// form is what the hierarchical AllGather uses for the inter-node phase
// (ring over node-leaders / same-local-index ranks across nodes).
struct CclInterNodeRingArgs {
  int myPe;
  int npes;
  int ringPos;
  int ringSize;
  int peBase;
  int peStride;
  application::SymmMemObjPtr memObj;
  application::SymmMemObjPtr flagsObj;
  size_t chunkBytes;
  // M4: number of RDMA QPs to fan the per-round ring put across.
  // 1 (default) = the original single-warp / single-QP put (also forced for any
  // same-node P2P/SDMA neighbour). >1 splits the chunk across warps 0..numQp-1,
  // each driving qpId=warpId, but ONLY when the neighbour is reached over RDMA
  // (the kernel checks transportTypes[nextPeer] at runtime so single-node
  // simulation stays single-warp -- see all_gather.hpp).
  int numQp;
};

// FUSED inter-node ring + intra-node LOCAL-block SDMA gather.
// A single grid runs the RDMA ring (Phase A, over the NIC) in blocks
// [0, ringBlocks) and the intra-node SDMA gather of THIS node's own block
// (Phase B for m == node_id -- the half that is INDEPENDENT of the ring, since
// every local rank's own shard is already present) in the remaining block, so
// the XGMI reassembly overlaps the NIC ring in ONE launch with NO host-side
// wait_stream merge. The ring fields mirror CclInterNodeRingArgs; the ``g*``
// fields mirror CclAllgatherSubGroupArgs<uint32_t> (the gather is a type-
// agnostic u32 byte move). ``ringBlocks`` partitions the grid. This is the
// parity lever (RCCL-beating @>=32MiB). Inert until the Python fused launcher
// is wired; this struct + glue
// only enable the fused __global__ to compile + be exercised.
struct CclFusedRingLocalGatherArgs {
  // --- inter-node ring (Phase A) ---
  int ringPos;
  int ringSize;
  int ringPeBase;
  int ringPeStride;
  application::SymmMemObjPtr ringMemObj;
  application::SymmMemObjPtr ringFlagsObj;
  size_t chunkBytes;
  int numQp;
  int ringBlocks;  // grid blocks [0, ringBlocks) run the ring; the rest gather

  // --- intra-node local-block SDMA gather (Phase B, m == node_id) ---
  int myPe;
  int npes;
  int groupSize;
  int groupPos;
  int gPeBase;
  int gPeStride;
  uint32_t* gInput;
  application::SymmMemObjPtr gDstMemObj;
  application::SymmMemObjPtr gFlagsObj;
  size_t gElementCount;
  size_t gDstBaseOffset;
  size_t gDstSlotStrideBytes;
  uint64_t gFlagVal;
};

// cross-handle builder for CclFusedRingLocalGatherArgs. The
// fused __global__ needs ONE args struct that sees BOTH the inter-node ring
// handle's ring memObj/flags AND the intra-node gather handle's
// dst(output)/flags/input -- but those live in two separate C++ classes, each of
// which already builds its own jit_args in prepare_*. Rather than reach into
// either class's privates, this takes the two already-built arg structs (the
// int64_t pointers their prepare_* calls return) and MERGES them, so the existing
// prepare paths stay byte-identical and this is pure additive glue.
//
// The fused launcher (Python, gated MORI_HIER_FUSE_LOCAL, default OFF) will call
// both handles' prepare_* (priming the ring slot + gather flags exactly as the
// shipped serial path does), pass the two returned pointers here, then launch
// FusedRingLocalGatherKernel_u32 once -- replacing the two separate kernel
// launches with one concurrent launch (NIC ring || XGMI local gather), with NO
// host wait_stream merge. ``ringBlocks`` partitions the grid: blocks
// [0,ringBlocks) run the ring, the rest run the local-block SDMA gather.
//
// The returned pointer is a function-local static (the Python launch path is
// single-threaded / single-stream per op, matching how each handle keeps its own
// jit_args_ member alive between prepare and launch). Inert until the launcher is
// wired; default shipped path is untouched.
inline int64_t BuildFusedRingLocalGatherArgs(int64_t ringArgsPtr, int64_t gatherArgsPtr,
                                             int ringBlocks) {
  static CclFusedRingLocalGatherArgs fused;
  const CclInterNodeRingArgs* r = reinterpret_cast<const CclInterNodeRingArgs*>(ringArgsPtr);
  const CclAllgatherSubGroupArgs<uint32_t>* g =
      reinterpret_cast<const CclAllgatherSubGroupArgs<uint32_t>*>(gatherArgsPtr);

  // --- inter-node ring (Phase A) ---
  fused.ringPos = r->ringPos;
  fused.ringSize = r->ringSize;
  fused.ringPeBase = r->peBase;
  fused.ringPeStride = r->peStride;
  fused.ringMemObj = r->memObj;
  fused.ringFlagsObj = r->flagsObj;
  fused.chunkBytes = r->chunkBytes;
  fused.numQp = r->numQp;
  fused.ringBlocks = ringBlocks < 1 ? 1 : ringBlocks;

  // --- intra-node local-block SDMA gather (Phase B, m == node_id) ---
  fused.myPe = g->myPe;
  fused.npes = g->npes;
  fused.groupSize = g->groupSize;
  fused.groupPos = g->groupPos;
  fused.gPeBase = g->peBase;
  fused.gPeStride = g->peStride;
  fused.gInput = g->input;
  fused.gDstMemObj = g->dstMemObj;
  fused.gFlagsObj = g->flagsMemObj;
  fused.gElementCount = g->elementCount;
  fused.gDstBaseOffset = g->dstBaseOffset;
  fused.gDstSlotStrideBytes = g->dstSlotStrideBytes;
  fused.gFlagVal = g->flagVal;

  return reinterpret_cast<int64_t>(&fused);
}

}  // namespace collective
}  // namespace mori
