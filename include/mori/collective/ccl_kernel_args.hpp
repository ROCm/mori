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
// Crown local-block flag (MORI_HIER_CROWN). !=0 => size-adaptive crown (2560);
// 0 => byte-identical flat crown.
inline int HierCrownRing() {
  const char* c = std::getenv("MORI_HIER_CROWN");
  if (c != nullptr && c[0] != '\0' && !(c[0] == '0' && c[1] == '\0')) return 2560;
  return 0;
}
// Local-block push-only (MORI_HIER_LOCAL_PUSHONLY, default OFF). Decouples the
// bx==rb local gather to push-only (completion reader drains flag slots [0,G)),
// dodging the deep-pipe stall where ring/reassembly CTAs starve the coupled
// per-slot wait's flag AMO. Byte-identical; 0 => coupled push+wait unchanged.
inline bool HierLocalPushOnly() {
  const char* e = std::getenv("MORI_HIER_LOCAL_PUSHONLY");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Deep-SQ temporal pipeline depth (MORI_HIER_DEEP_PIPE=P, default 2). Splits the
// chunk into P sub-chunks issued back-to-back on the full numQp fan-out with a
// per-sub-chunk put-with-signal (RC in-order: sub-chunk p's flag fires after its
// data, before p+1's), overlapping XGMI reassembly under the NIC. Returns -1 for
// "auto" (caller derives depth from chunkBytes), else clamped [1,16]; <=1 => OFF.
inline int HierDeepPipe() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE");
  // Default 2; E2E landing anchored by the crown DEFER_HOSTSYNC host fence.
  if (e == nullptr || e[0] == '\0') return 2;
  if (e[0] == 'a' || e[0] == 'A') return -1;  // "auto"
  int v = std::atoi(e);
  if (v < 1) return 1;
  if (v > 16) return 16;
  return v;
}

// Per-sub-chunk quiet-fence landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_QUIET,
// default OFF). One QP per sub-chunk + plain put; drain that QP's send-CQ
// (ShmemQuietThread(pe,qpId) == data landed remotely, RC in-order) before the flag
// AMO, so chunkReadyFlags[p] publishes only after sub-chunk p landed -- bit-exact
// at scale, where the put-signal AMO can beat its own data. Only when deepPipe>1.
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

// Sub-group intra-node SDMA AllGather. G local ranks {peBase + i*peStride} gather
// their shards over SDMA; member at position p writes its shard into slot p of
// every member (groupSize contiguous slots). Flat whole-world gather ==
// groupSize=npes, groupPos=myPe, peBase=0, peStride=1.
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
  // Per-peer destination slot stride in bytes; 0 packs slots contiguously
  // (stride == copy size). Non-zero decouples slot stride from copy size so a
  // chunk of a slice lands at its final strided position within a full-size block
  // (chunked inter/intra reassembly pipeline). 0 keeps the contiguous-slot
  // contract byte-for-byte.
  size_t dstSlotStrideBytes;
  uint64_t flagVal;
  // Disjoint flag-slot base for race-free concurrent direct gathers: _body uses
  // slots [flagBase, flagBase+groupSize). Lane j sets flagBase = j*groupSize so
  // concurrent launches never race. 0 keeps single-gather callers on [0,groupSize).
  size_t flagBase;
};

// Fused hierarchical param-contiguous SubGroup gather. One launch does the full
// per-(node-block, param) gather: member g pushes this PE's shard into the
// registered output in param-contiguous layout. For block m and param split s
// (per-rank count splitSizes[s]=E_s at input offset splitOffsets[s]=O_s), global
// rank r = m*groupSize + g lands at output element offset O_s*worldSize + r*E_s.
// input is the Phase-A collection (numBlocks blocks of blockStrideElems u32 lanes);
// split arrays are device pointers (u32-lane units), shared across all blocks.
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

// Sub-group intra-node SDMA broadcast. Root (group position 0 == PE peBase) holds
// a full elementCount-lane buffer and SDMA-copies it into every member's dstMemObj
// (incl. itself). Intra-node placement phase of the leader-only hierarchical AG.
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

// Inter-node RDMA ring AllGather. Ring buffer memObj holds ringSize contiguous
// chunkBytes chunks (chunk k at k*chunkBytes); on entry only this PE's chunk
// (slot ringPos) is filled, after ringSize-1 rounds every member holds all in
// ring order. Byte-move ring => not templated (raw bytes over shmem: P2P intra,
// RDMA inter). Runs over the arithmetic sub-group {peBase + i*peStride}, this PE
// at ringPos. Flat whole-world ring == peBase=0, peStride=1, ringSize=npes,
// ringPos=myPe (hierarchical AG uses the sub-group form for the inter-node phase).
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
  // RDMA QPs to fan the per-round put across. 1 (default) = single-warp/single-QP
  // (also forced for same-node P2P/SDMA neighbours); >1 splits the chunk across
  // warps 0..numQp-1 (qpId=warpId) only when the neighbour is over RDMA (runtime
  // transportTypes[nextPeer] check -- see all_gather.hpp).
  int numQp;
  // WRITE-PUSH (SEND-CQ) per-channel landing fence (MORI_HIER_RING_WRITE, default
  // 0). Each channel CTA pushes its sub-range as a put-with-signal on qpId=bid then
  // drains that QP's SEND CQE; receiver spins its per-channel flag. 0 = unchanged.
  int useWriteFence = 0;
};

// FUSED inter-node ring + intra-node LOCAL-block SDMA gather. One grid runs the
// RDMA ring (Phase A) in blocks [0,ringBlocks) and the SDMA gather of THIS node's
// own block (Phase B, m==node_id -- ring-independent, every local shard already
// present) in the rest, overlapping XGMI reassembly with the NIC ring in one
// launch with no host wait_stream merge. Ring fields mirror CclInterNodeRingArgs;
// g* fields mirror CclAllgatherSubGroupArgs<uint32_t> (type-agnostic u32 move).
// ringBlocks partitions the grid. Inert until the fused launcher is wired.
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

// Merge already-built ring (CclInterNodeRingArgs) + gather
// (CclAllgatherSubGroupArgs<uint32_t>) arg structs into one, so the existing
// prepare_* paths stay byte-identical (pure additive glue). ringBlocks partitions
// the grid: [0,ringBlocks) ring, the rest the local-block SDMA gather. Returned
// pointer is a function-local static (launch path is single-stream per op). Inert
// until the launcher is wired.
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

// Fused inter-node ring + intra-node remote-block reassembly (pipelined). Unlike
// FusedRingLocalGatherKernel_u32 (local block only), this pipelines the remote-block
// reassembly with the ring: the ring runs as ringBlocks channels, each publishing
// chunkReadyFlags[bid] the instant its sub-range lands; ringBlocks reassembly blocks
// each spin on chunkReadyFlags[j] and SDMA-push sub-range j of every remote block
// into the registered output over XGMI, so reassembly of j overlaps ring channel
// j+1 still on the NIC. Each PE reassembles from its own ring buffer (slot m holds
// node m's chunk), so the only dependency is this PE's own ring landing -- a local
// flag spin, NO global finish barrier, NO copy-OUT. Grid = 2*ringBlocks+1:
// [0,ringBlocks) ring, [ringBlocks] local gather, (ringBlocks,2*ringBlocks] remote
// reassembly (block j = blockIdx.x - ringBlocks - 1). Default OFF (MORI_HIER_FUSE_REMOTE).
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
  int nodeId;    // this node's block index (skipped by remote reassembly)
  // Reassembly block count, decoupled from ringBlocks so XGMI reassembly can be
  // parallelised even when the ring is single-channel. Each block owns a disjoint
  // 16B-aligned byte sub-range and waits until all ring channels land (spin over
  // ringBlocks flags) before pushing. 0 => reassemblyBlocks == ringBlocks.
  int reassemblyBlocks = 0;
  // Deep-SQ temporal pipeline depth (see HierDeepPipe). P>1 splits the chunk into
  // P sub-chunks with per-sub-chunk landing flags. 1 = OFF (byte-identical).
  int deepPipe = 1;
  // Deep-SQ QUIET landing fence (see HierDeepPipeQuietOn). 1 => per-sub-chunk QP +
  // plain put, drain send-CQ before the flag AMO so the flag never beats the data
  // (bit-exact at scale). 0 = OFF. Only when deepPipe>1.
  int deepPipeQuiet = 0;
  // Local-block push-only (see HierLocalPushOnly). 1 => bx==rb gather is push-only,
  // completion folded into the reader (drains flag slots [0,G)); removes the
  // DEEP_PIPE>=8 wait deadlock. Byte-identical. 0 = OFF (coupled push+wait).
  int localPushOnly = 0;
  // Intra reassembly deep-SQ (see HierReasmDeepSqOn). 1 => a worker submits all its
  // channels' copy descriptors back-to-back (each after its landing flag), then a
  // SINGLE drain before firing the deferred output flags. 0 = OFF (byte-identical).
  // Bit-exact by construction (flag never precedes its bytes).
  int reasmDeepSq = 0;
  // Crown local-block flag (see HierCrownRing). 0 = OFF (byte-identical flat crown).
  int crownRing = 0;
};

// Merge already-built ring (CclInterNodeRingArgs) + gather
// (CclAllgatherSubGroupArgs<uint32_t>, primed for the LOCAL block) args + pipeline
// extras into one. Mirrors BuildFusedRingLocalGatherArgs; prepare_* paths stay
// byte-identical (pure additive glue). Inert until the launcher is wired.
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
      r->useWriteFence;
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
  // Deep-SQ temporal pipeline only on the single-channel ring (rb==1); RING_BLOCKS>1
  // keeps its spatial split.
  {
    int dp = (fused.ringBlocks == 1) ? HierDeepPipe() : 1;
    if (dp < 0) {  // "auto": depth ~ chunkBytes / 16MiB sub-chunk, clamp[1,16]
      // 16MiB sub-chunk fills the mlx5 NIC DMA under the coherence window.
      const size_t sub = 16ull * 1024ull * 1024ull;
      size_t d = (fused.chunkBytes + sub / 2) / sub;
      dp = (d < 1) ? 1 : (d > 16 ? 16 : static_cast<int>(d));
    }
    // Snap dp down to the largest divisor of chunkBytes so sub-chunks are equal (no
    // ragged tail); no-op when dp already divides chunkBytes.
    if (dp > 1 && fused.chunkBytes > 0) {
      while (dp > 1 && (fused.chunkBytes % static_cast<size_t>(dp)) != 0) --dp;
    }
    fused.deepPipe = dp;
  }
  // Scale-robust landing fence default. The put-with-signal AMO can beat its own
  // data for large sub-chunks, so auto-engage the quiet-drain fence whenever
  // deep-pipe runs unless forced off (MORI_HIER_DEEP_PIPE_QUIET=0). Bit-exact;
  // deepPipe<=1 never enters => byte-identical path.
  {
    const char* q = std::getenv("MORI_HIER_DEEP_PIPE_QUIET");
    const bool quietForcedOff = (q != nullptr && q[0] == '0' && q[1] == '\0');
    fused.deepPipeQuiet =
        (HierDeepPipeQuietOn() || (fused.deepPipe > 1 && !quietForcedOff))
            ? 1
            : 0;
  }
  fused.localPushOnly = HierLocalPushOnly() ? 1 : 0;
  fused.reasmDeepSq = reasmDeepSq;
  fused.crownRing = HierCrownRing();
  return reinterpret_cast<int64_t>(&fused);
}

}  // namespace collective
}  // namespace mori
