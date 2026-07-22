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
#include <cstdlib>

namespace mori {
namespace collective {
// Crown local-block flag (MORI_HIER_CROWN). Set => the named size-adaptive crown
// (flatMW bit9=512 + batchSelf bit11=2048 = 2560); unset/0 => byte-identical flat crown.
inline int HierCrownRing() {
  const char* c = std::getenv("MORI_HIER_CROWN");
  if (c != nullptr && c[0] != '\0' && !(c[0] == '0' && c[1] == '\0')) return 2560;
  return 0;
}
// Local-block push-only (MORI_HIER_LOCAL_PUSHONLY, default OFF). The local node-block
// gather (bx==rb) normally uses the coupled push+wait
// OneShotAllGatherSdmaSubGroupKernel_body; under deep pipelining the concurrent ring
// sub-chunk CTAs and reassembly workers can starve the cross-rank flag AMO the coupled
// per-slot wait spins on, causing a circular stall. This lever decouples the local
// block like the remote reassembly already is: the bx==rb CTA pushes its own column
// (no wait), and the completion reader (same CTA) is extended to also drain the local
// flag slots [0,G). Byte-identical output (same pushes, same flags); only the wait
// moves off the coupled path, so it is bit-exact and deadlock-free at any depth.
// Default OFF keeps the coupled path byte-identical.
inline bool HierLocalPushOnly() {
  const char* e = std::getenv("MORI_HIER_LOCAL_PUSHONLY");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Deep-SQ temporal pipeline (MORI_HIER_DEEP_PIPE=P, default 2). The giant AG wall is
// roughly half inter-node RDMA fill and half intra-node SDMA reassembly, fully serial
// at rb=1. Spatial ring-block split is a wash (it grows inter fill by exactly what it
// hides). This lever instead splits the chunk into P temporal sub-chunks issued
// back-to-back on the same full numQp fan-out (deep SQ, full inter BW) with a
// per-sub-chunk put-with-signal, so sub-chunk p's landing flag fires (RC in-order,
// after its data) before p+1's -- a reassembly worker pushes sub-chunk p over XGMI
// while p+1.. still cross the NIC, hiding the intra leg under the inter with no
// inter-fill growth. Returns -1 for "auto" (size-adaptive: caller derives depth from
// chunkBytes), else the clamped explicit depth [1,16]. depth<=1 => path.
inline int HierDeepPipe() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE");
  // Default depth 2. The E2E landing is anchored by the crown DEFER_HOSTSYNC host
  // fence, so the pipeline stays bit-exact with no explicit env.
  if (e == nullptr || e[0] == '\0') return 2;
  if (e[0] == 'a' || e[0] == 'A') return -1;  // "auto"
  int v = std::atoi(e);
  if (v < 1) return 1;
  if (v > 16) return 16;
  return v;
}

// Quiet-fence per-sub-chunk landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_QUIET, default
// OFF). The put-with-signal AMO fails bit-exact for large chunks because the AMO can
// beat its own data landing. This is the scale-robust option: dedicate one
// QP per temporal sub-chunk, issue a plain put, and drain that QP's send-CQ with
// ShmemQuietThread(pe, qpId) (== the sub-chunk's data has landed remotely, RC in-order)
// before the separate flag AMO. So chunkReadyFlags[p] is published only after sub-chunk
// p physically landed -- bit-exact at scale -- while preserving the temporal pipeline
// (p's flag fires before p+1's data finishes, since each sub-chunk drains its own QP in
// order). Only meaningful when deepPipe>1; takes precedence over the put-signal path.
inline bool HierDeepPipeQuietOn() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_QUIET");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
}  // namespace collective
}  // namespace mori

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
  // Per-peer destination slot stride in bytes. The kernel writes
  // member ``p``'s shard into slot ``p`` of the destination; by default the
  // slots are packed contiguously (stride == elementCount*sizeof(T) == the copy
  // size). A non-zero ``dstSlotStrideBytes`` decouples the slot stride from the
  // copy size, so a sub-range (chunk) of a slice can be written into its final
  // strided position within a full-size block. This is the enabler for the
  // chunked inter/intra reassembly pipeline (overlap the remote-block gather of
  // chunk k with the inter ring of chunk k+1): each chunk copies elementCount
  // (= chunk) bytes per peer but lands at slot stride = full slice size. 0 keeps
  // the contiguous-slot contract byte-for-byte unchanged.
  size_t dstSlotStrideBytes;
  uint64_t flagVal;
  // Disjoint flag-slot base for race-free concurrent direct gathers.
  // The device _body uses flag slots [flagBase, flagBase+groupSize). Default 0
  // keeps every classic single-gather caller on [0, groupSize) byte-for-byte.
  // A concurrent Phase-B reassembly lane j sets flagBase = j*groupSize so N
  // simultaneous gather_kernel_direct launches never race on the shared flag
  // slots -- the same mechanism the FUSED reassembly kernel already uses.
  size_t flagBase;
};

// Fused hierarchical param-contiguous SubGroup gather. One launch performs the
// full per-(node-block, param) gather. Each of ``G`` group members
// pushes this PE's shard (groupPos == local rank g) directly into the
// registered user output in param-contiguous layout:
// for node block ``m`` (in [0,numBlocks)) and param split ``s`` with per-rank
// element count ``splitSizes[s]`` (== E_s, u32 lanes) at input element offset
// ``splitOffsets[s]`` (== O_s within a block of ``blockStrideElems`` u32 lanes),
// global rank ``r = m*groupSize + g`` lands at output element offset
// ``O_s*worldSize + r*E_s``. ``input`` is the Phase-A collection buffer
// (numBlocks contiguous blocks of blockStrideElems u32 lanes). Split arrays are
// device pointers (size_t / u32-lane units), shared across all blocks.
template <typename T>
struct CclAllgatherSubGroupParamContiguousArgs {
  int myPe;
  int npes;
  int groupSize;  // G local ranks per node
  int groupPos;   // g == this PE's local rank within the node
  int peBase;
  int peStride;
  int numBlocks;   // N node blocks gathered by Phase A
  int firstBlock;  // global m of input's first block (source i -> m=firstBlock+i)
  T* input;        // Phase-A collection: numBlocks * blockStrideElems u32 lanes
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t blockStrideElems;  // per-node-block stride in input (u32 lanes)
  size_t worldSize;         // W == npes; output param scaling factor
  size_t dstBaseOffset;     // byte offset into the registered output segment
  uint64_t flagVal;
  const size_t* splitSizes;    // device ptr, u32-lane units (E_s)
  const size_t* splitOffsets;  // device ptr, u32-lane units (O_s within a block)
  size_t splitCount;
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
// Sub-group support: the ring runs over an arithmetic sub-group of global
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
  // Number of RDMA QPs to fan the per-round ring put across.
  // 1 (default) = the original single-warp / single-QP put (also forced for any
  // same-node P2P/SDMA neighbour). >1 splits the chunk across warps 0..numQp-1,
  // each driving qpId=warpId, but only when the neighbour is reached over RDMA
  // (the kernel checks transportTypes[nextPeer] at runtime so single-node
  // simulation stays single-warp -- see all_gather.hpp).
  int numQp;
  // WRITE-PUSH (SEND-CQ) per-channel landing fence (env MORI_HIER_RING_WRITE,
  // default 0). On the giant multiBlock AG each channel CTA pushes its sub-range
  // as a fused put-with-signal on qpId=bid then drains that QP's SEND CQE; the
  // receiver spins its per-channel inbound flag. Default 0 = push path unchanged.
  int useWriteFence = 0;
};

// FUSED inter-node ring + intra-node LOCAL-block SDMA gather.
// A single grid runs the RDMA ring (Phase A, over the NIC) in blocks
// [0, ringBlocks) and the intra-node SDMA gather of THIS node's own block
// (Phase B for m == node_id -- the half that is INDEPENDENT of the ring, since
// every local rank's own shard is already present) in the remaining block, so
// the XGMI reassembly overlaps the NIC ring in ONE launch with NO host-side
// wait_stream merge. The ring fields mirror CclInterNodeRingArgs; the ``g*``
// fields mirror CclAllgatherSubGroupArgs<uint32_t> (the gather is a type-
// agnostic u32 byte move). ``ringBlocks`` partitions the grid. Inert until the Python
// fused launcher is wired; this struct + glue only enable the fused __global__ to
// compile and be exercised.
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

// Cross-handle builder for CclFusedRingLocalGatherArgs. The
// fused __global__ needs one args struct that sees both the inter-node ring
// handle's ring memObj/flags AND the intra-node gather handle's
// dst(output)/flags/input -- but those live in two separate C++ classes, each of
// which already builds its own jit_args in prepare_*. Rather than reach into
// either class's privates, this takes the two already-built arg structs (the
// int64_t pointers their prepare_* calls return) and MERGES them, so the existing
// prepare paths stay byte-identical and this is pure additive glue.
//
// The fused launcher (Python, gated MORI_HIER_FUSE_LOCAL, default OFF) will call
// both handles' prepare_* (priming the ring slot + gather flags exactly as the
// serial path does), pass the two returned pointers here, then launch
// FusedRingLocalGatherKernel_u32 once -- replacing the two separate kernel
// launches with one concurrent launch (NIC ring || XGMI local gather), with NO
// host wait_stream merge. ``ringBlocks`` partitions the grid: blocks
// [0,ringBlocks) run the ring, the rest run the local-block SDMA gather.
//
// The returned pointer is a function-local static (the Python launch path is
// single-threaded / single-stream per op, matching how each handle keeps its own
// jit_args_ member alive between prepare and launch). Inert until the launcher is
// wired; default path is untouched.
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

// ============================================================================
// Fused inter-node ring + intra-node remote-block reassembly (pipelined)
// ============================================================================
// The FusedRingLocalGatherKernel_u32 fuses the ring with only the local node-block
// gather (the ring-independent half); the remote-block reassembly otherwise runs as a
// separate launch after the whole ring + its global finish barrier (two serial phases:
// the NIC sits idle during the XGMI reassembly and vice-versa). This kernel closes that
// by pipelining: the ring runs as ``ringBlocks`` channels, each publishing
// ``chunkReadyFlags[bid]`` the instant its sub-range lands; a matching set of
// ``ringBlocks`` reassembly blocks each spin on chunkReadyFlags[j] and, the
// instant sub-range j is ready, SDMA-push that sub-range of EVERY remote block
// straight into the registered output over XGMI -- so sub-range j's reassembly
// overlaps ring channel j+1 still crossing the NIC. Because each PE reassembles
// a remote block by pushing from its own ring buffer (slot m holds node m's chunk
// in ring order, see AllgatherInterNodeRing.full_tensor), the only dependency is
// this PE's own ring landing -- a purely local flag spin, NO global finish
// barrier and NO copy-OUT scratch. Grid = 2*ringBlocks + 1: [0,ringBlocks) ring,
// [ringBlocks] the local-block gather, (ringBlocks, 2*ringBlocks] the remote
// reassembly (block j = blockIdx.x - ringBlocks - 1). Default OFF (env
// MORI_HIER_FUSE_REMOTE); the serial path is untouched.
struct CclFusedRingRemoteGatherArgs {
  // --- inter-node ring (Phase A) ---
  int ringPos;
  int ringSize;
  int ringPeBase;
  int ringPeStride;
  application::SymmMemObjPtr ringMemObj;
  application::SymmMemObjPtr ringFlagsObj;
  size_t chunkBytes;
  int numQp;
  int ringBlocks;
  int useWriteFence = 0;  // RDMA-WRITE-push SEND-CQ landing fence (MORI_HIER_RING_WRITE)
  uint64_t* chunkReadyFlags = nullptr;  // device, >= ringBlocks u64, zeroed

  // --- intra-node local-block SDMA gather (Phase B, m == nodeId) ---
  int myPe;
  int npes;
  int groupSize;
  int groupPos;
  int gPeBase;
  int gPeStride;
  uint32_t* gInput;  // this PE's own input (local-block source, ring-independent)
  application::SymmMemObjPtr gDstMemObj;
  application::SymmMemObjPtr gFlagsObj;
  size_t gElementCount;        // per-slice u32 lanes (== count)
  size_t gDstBaseOffset;       // bytes: local block base (nodeId*blockCount*4)
  size_t gDstSlotStrideBytes;  // bytes: full-slice stride (== chunkBytes)
  uint64_t gFlagVal;

  // --- remote reassembly (Phase B, m != nodeId; reads the ring buffer) ---
  int numNodes;  // N == ringSize
  int nodeId;    // this node's block index (skipped by the remote reassembly)
  // Number of reassembly blocks, DECOUPLED from ringBlocks so the XGMI
  // reassembly can be parallelised (like the multi-block copy-OUT) even when the
  // ring runs as a single channel (ringBlocks==1). Each reassembly block owns a
  // disjoint 16B-aligned byte sub-range of the chunk and waits until all ring
  // channels have landed (spin over the ringBlocks flags) before pushing its
  // sub-range over XGMI. 0 => legacy behaviour (reassemblyBlocks == ringBlocks).
  int reassemblyBlocks = 0;
  // Deep-SQ temporal pipeline depth (see HierDeepPipe). P>1 splits the chunk into
  // P temporal sub-chunks with per-sub-chunk landing flags so reassembly overlaps
  // the still-in-flight later sub-chunks. 1 = OFF (byte-identical path).
  int deepPipe = 1;
  // Deep-SQ temporal pipeline QUIET landing fence (see HierDeepPipeQuietOn). 1 =>
  // each temporal sub-chunk rides its own QP with a plain put; thread drains that
  // QP's send-CQ (ShmemQuietThread(pe,qpId) = sub-chunk landed remotely) BEFORE the
  // separate flag AMO, so the flag never fires ahead of the data landing (bit-exact
  // at scale, unlike the racy fused put-signal). 0 = OFF. Only when deepPipe>1.
  int deepPipeQuiet = 0;
  // Local-block push-only (see HierLocalPushOnly). 1 => the bx==rb local node-block
  // gather is push-only (no coupled per-slot wait) and its completion is folded
  // into the completion reader (which then also drains flag slots [0,G)); this
  // removes the DEEP_PIPE>=8 "Slow wait for sub-group pos" deadlock. Byte-identical
  // output. 0 = OFF (coupled push+wait, byte-identical path).
  int localPushOnly = 0;
  // Intra reassembly deep-SQ (see HierReasmDeepSqOn).
  // 1 => a reassembly worker submits all its owned channels' copy descriptors
  // back-to-back (each after its landing flag) keeping the SDMA SQ continuously
  // fed, then a SINGLE drain covers them all before firing the deferred output
  // flags -- instead of submit+drain per channel. 0 = OFF (byte-identical
  // path). Bit-exact by construction (flag never precedes its bytes).
  int reasmDeepSq = 0;
  // Crown local-block flag (see HierCrownRing / MORI_HIER_CROWN). 0 = OFF
  // (byte-identical flat crown).
  int crownRing = 0;
};

// Builder: merge an already-built ring args (CclInterNodeRingArgs) and gather
// args (CclAllgatherSubGroupArgs<uint32_t>, primed for the LOCAL block) plus the
// pipeline extras into one CclFusedRingRemoteGatherArgs. Mirrors
// BuildFusedRingLocalGatherArgs so the existing prepare_* paths stay byte-
// identical; this is pure additive glue. Inert until the Python launcher is wired.
inline int64_t BuildFusedRingRemoteGatherArgs(int64_t ringArgsPtr, int64_t gatherArgsPtr,
                                              int ringBlocks, int64_t chunkReadyFlagsPtr,
                                              int numNodes, int nodeId, int reassemblyBlocks = 0,
                                              int reasmDeepSq = 0) {
  static CclFusedRingRemoteGatherArgs fused;
  const CclInterNodeRingArgs* r = reinterpret_cast<const CclInterNodeRingArgs*>(ringArgsPtr);
  const CclAllgatherSubGroupArgs<uint32_t>* g =
      reinterpret_cast<const CclAllgatherSubGroupArgs<uint32_t>*>(gatherArgsPtr);

  fused.ringPos = r->ringPos;
  fused.ringSize = r->ringSize;
  fused.ringPeBase = r->peBase;
  fused.ringPeStride = r->peStride;
  fused.ringMemObj = r->memObj;
  fused.ringFlagsObj = r->flagsObj;
  fused.chunkBytes = r->chunkBytes;
  fused.numQp = r->numQp;
  fused.ringBlocks = ringBlocks < 1 ? 1 : ringBlocks;
  fused.useWriteFence =
      r->useWriteFence;  // plumb MORI_HIER_RING_WRITE into the crown/deep-pipe launch
  fused.chunkReadyFlags = reinterpret_cast<uint64_t*>(chunkReadyFlagsPtr);

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

  fused.numNodes = numNodes;
  fused.nodeId = nodeId;
  fused.reassemblyBlocks = reassemblyBlocks > 0 ? reassemblyBlocks : fused.ringBlocks;
  // Deep-SQ temporal pipeline only engages on the single-channel ring (rb==1, the
  // giant-AG fan-out path); RING_BLOCKS>1 (multiBlock) keeps its spatial split.
  {
    int dp = (fused.ringBlocks == 1) ? HierDeepPipe() : 1;
    if (dp < 0) {  // "auto": depth = round(perPE chunkBytes / subBytes), clamp[1,16]
      // 16MiB sub-chunk: fills mlx5 NIC DMA while staying under the coherence window.
      const size_t sub = 16ull * 1024ull * 1024ull;
      size_t d = (fused.chunkBytes + sub / 2) / sub;
      dp = (d < 1) ? 1 : (d > 16 ? 16 : static_cast<int>(d));
    }
    // Snap dp down to the largest divisor of chunkBytes <= dp so every sub-chunk is
    // equal (no ragged remainder tail). Down-only => sub-chunk count never exceeds the
    // Python-sized flag budget; no-op when dp already divides chunkBytes.
    if (dp > 1 && fused.chunkBytes > 0) {
      while (dp > 1 && (fused.chunkBytes % static_cast<size_t>(dp)) != 0) --dp;
    }
    fused.deepPipe = dp;
  }
  // Scale-robust landing fence default. The deep-pipe put-with-signal AMO is not
  // scale-robust -- for large sub-chunks the AMO can beat its own data landing, so the
  // reassembly reader can consume un-landed bytes on the giant AG. Auto-engage the
  // quiet-drain landing fence whenever deep-pipe runs, unless explicitly forced off
  // (MORI_HIER_DEEP_PIPE_QUIET=0). Bit-exact (same drains/AMOs/slots, only
  // independent-completion order relaxed). deepPipe<=1 (the default) never enters this
  // path => byte-identical path.
  {
    const char* q = std::getenv("MORI_HIER_DEEP_PIPE_QUIET");
    const bool quietForcedOff = (q != nullptr && q[0] == '0' && q[1] == '\0');
    fused.deepPipeQuiet =
        (HierDeepPipeQuietOn() || (fused.deepPipe > 1 && !quietForcedOff))
            ? 1
            : 0;
  }
  fused.localPushOnly = HierLocalPushOnly() ? 1 : 0;
  // Intra reassembly deep-SQ: feed all reassembly channels then drain once.
  fused.reasmDeepSq = reasmDeepSq;
  // Crown local-block flag (MORI_HIER_CROWN). Default 0 => OFF (byte-identical
  // flat crown). See HierCrownRing.
  fused.crownRing = HierCrownRing();
  return reinterpret_cast<int64_t>(&fused);
}

}  // namespace collective
}  // namespace mori
