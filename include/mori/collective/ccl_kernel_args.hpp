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
// Local-block push-only (MORI_HIER_LOCAL_PUSHONLY, default OFF). Decouples the bx==rb
// local-block gather from the coupled push+wait: CTA pushes its column (no wait), the
// completion reader also drains local flag slots [0,G). Byte-identical output, deadlock-free
// at any pipeline depth. Default OFF keeps the coupled path byte-identical.
inline bool HierLocalPushOnly() {
  const char* e = std::getenv("MORI_HIER_LOCAL_PUSHONLY");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Deep-SQ temporal pipeline (MORI_HIER_DEEP_PIPE=P, default 2). Splits the chunk into P
// temporal sub-chunks issued back-to-back on the full numQp fan-out with per-sub-chunk
// landing flags, so reassembly of sub-chunk p overlaps p+1.. still crossing the NIC.
// Returns -1 for "auto" (caller derives depth from chunkBytes), else clamped [1,16];
// depth<=1 => path off.
inline int HierDeepPipe() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE");
  // Default depth 2; landing anchored by the crown host fence (bit-exact, no explicit env).
  if (e == nullptr || e[0] == '\0') return 2;
  if (e[0] == 'a' || e[0] == 'A') return -1;  // "auto"
  int v = std::atoi(e);
  if (v < 1) return 1;
  if (v > 16) return 16;
  return v;
}

// Quiet-fence per-sub-chunk landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_QUIET, default OFF).
// Each temporal sub-chunk rides its own QP with a plain put; drain that QP's send-CQ
// (ShmemQuietThread == sub-chunk landed remotely, RC in-order) before the flag AMO, so the
// flag never precedes the data (bit-exact at scale). Only when deepPipe>1; takes precedence
// over the put-signal path.
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

// Sub-group intra-node SDMA AllGather over G local ranks
// {peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}; this PE is at groupPos.
// Destination holds groupSize contiguous slots; member p writes its shard into slot p of
// every member. Flat whole-world gather = groupSize=npes, groupPos=myPe, peBase=0, peStride=1.
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
  // Per-peer destination slot stride in bytes. Default 0 packs slots contiguously
  // (stride == elementCount*sizeof(T) == copy size). Non-zero decouples slot stride from
  // copy size so a chunk of a slice lands at its final strided position within a full-size
  // block (enables the chunked inter/intra reassembly pipeline).
  size_t dstSlotStrideBytes;
  uint64_t flagVal;
  // Disjoint flag-slot base for race-free concurrent direct gathers. _body uses slots
  // [flagBase, flagBase+groupSize); default 0 keeps single-gather callers on [0, groupSize).
  // Concurrent reassembly lane j sets flagBase = j*groupSize to avoid racing on shared slots.
  size_t flagBase;
};

// Fused hierarchical param-contiguous SubGroup gather: one launch does the full
// per-(node-block, param) gather. Member g pushes this PE's shard into the registered user
// output in param-contiguous layout: for node block m in [0,numBlocks) and param split s
// (per-rank count splitSizes[s]=E_s at input offset splitOffsets[s]=O_s within
// blockStrideElems u32 lanes), global rank r=m*groupSize+g lands at output element offset
// O_s*worldSize + r*E_s. input is the Phase-A collection (numBlocks blocks of
// blockStrideElems u32 lanes); split arrays are device ptrs (u32-lane units), shared across blocks.
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

// Sub-group intra-node SDMA broadcast. Root (groupPos 0 == PE peBase) holds a full
// elementCount-lane buffer in input and SDMA-copies it into every member's dstMemObj
// {peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}, incl. itself. Intra-node
// placement phase of the hierarchical AllGather leader-only variant.
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

// Inter-node RDMA ring AllGather. Ring buffer memObj holds ringSize contiguous chunks of
// chunkBytes (chunk k at k*chunkBytes); on entry only this PE's chunk (slot ringPos) is
// filled. After ringSize-1 rounds every member holds all chunks in ring order. Not
// templated -- the kernel moves raw bytes over shmem (P2P intra-node, RDMA cross-node).
// Sub-group: the ring runs over {peBase, peBase+peStride, ..., peBase+(ringSize-1)*peStride},
// this PE at ringPos; flat whole-world = peBase=0, peStride=1, ringSize=npes, ringPos=myPe.
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
  // Number of RDMA QPs to fan the per-round ring put across. 1 (default) = single-QP put
  // (also forced for same-node P2P/SDMA). >1 splits the chunk across warps 0..numQp-1
  // (qpId=warpId), only when the neighbour is reached over RDMA (runtime transport check).
  int numQp;
  // WRITE-PUSH (SEND-CQ) per-channel landing fence (MORI_HIER_RING_WRITE, default 0). Each
  // channel CTA pushes its sub-range as a put-with-signal on qpId=bid then drains that QP's
  // SEND CQE; receiver spins its per-channel flag. 0 = push path unchanged.
  int useWriteFence = 0;
};

// FUSED inter-node ring + intra-node LOCAL-block SDMA gather in one grid: RDMA ring
// (Phase A) in blocks [0,ringBlocks), and the intra-node SDMA gather of THIS node's own
// block (Phase B, m==node_id -- ring-independent) in the remaining blocks, so XGMI
// reassembly overlaps the NIC ring with no host wait_stream merge. Ring fields mirror
// CclInterNodeRingArgs; g* fields mirror CclAllgatherSubGroupArgs<uint32_t>. ringBlocks
// partitions the grid.
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

// Cross-handle builder for CclFusedRingLocalGatherArgs: merges the two already-built arg
// structs (ring memObj/flags + gather dst/flags/input, from each handle's prepare_*) so the
// existing prepare paths stay byte-identical. ringBlocks partitions the grid. Returned
// pointer is a function-local static (single-stream launch path).
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
// Pipelines the ring with the remote-block reassembly: the ring runs as ringBlocks channels
// each publishing chunkReadyFlags[bid] when its sub-range lands; matching reassembly blocks
// spin on chunkReadyFlags[j] and SDMA-push sub-range j of every remote block into the output
// over XGMI, so reassembly of j overlaps ring channel j+1. Each PE reassembles from its own
// ring buffer (slot m == node m's chunk in ring order), so the only dependency is a local
// flag spin -- no global barrier, no copy-OUT scratch. Grid = 2*ringBlocks+1: [0,ringBlocks)
// ring, [ringBlocks] local-block gather, (ringBlocks,2*ringBlocks] remote reassembly
// (block j = blockIdx.x-ringBlocks-1). Default OFF (MORI_HIER_FUSE_REMOTE).
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
  // Number of reassembly blocks, decoupled from ringBlocks so the XGMI reassembly can be
  // parallelised even when ringBlocks==1. Each owns a disjoint 16B-aligned sub-range and
  // waits for all ring channels to land before pushing. 0 => reassemblyBlocks == ringBlocks.
  int reassemblyBlocks = 0;
  // Deep-SQ temporal pipeline depth (see HierDeepPipe). P>1 splits the chunk into P temporal
  // sub-chunks with per-sub-chunk landing flags; 1 = OFF (byte-identical path).
  int deepPipe = 1;
  // Deep-SQ QUIET landing fence (see HierDeepPipeQuietOn). 1 => each sub-chunk rides its own
  // QP; drain its send-CQ (ShmemQuietThread = sub-chunk landed remotely) before the flag AMO
  // so the flag never precedes the data (bit-exact at scale). 0 = OFF. Only when deepPipe>1.
  int deepPipeQuiet = 0;
  // Local-block push-only (see HierLocalPushOnly). 1 => bx==rb local-block gather is
  // push-only, completion folded into the reader (drains flag slots [0,G)); byte-identical
  // output. 0 = OFF (coupled push+wait, byte-identical path).
  int localPushOnly = 0;
  // Intra reassembly deep-SQ (see HierReasmDeepSqOn). 1 => a worker submits all its channels'
  // copy descriptors back-to-back then a single drain before firing the deferred output
  // flags. 0 = OFF (byte-identical). Bit-exact by construction (flag never precedes its bytes).
  int reasmDeepSq = 0;
  // Crown local-block flag (see HierCrownRing / MORI_HIER_CROWN). 0 = OFF
  // (byte-identical flat crown).
  int crownRing = 0;
};

// Builder: merge already-built ring args (CclInterNodeRingArgs) and gather args
// (CclAllgatherSubGroupArgs<uint32_t>, primed for the LOCAL block) plus pipeline extras into
// one CclFusedRingRemoteGatherArgs. Mirrors BuildFusedRingLocalGatherArgs; prepare_* paths
// stay byte-identical.
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
  // Deep-SQ temporal pipeline only engages on the single-channel ring (rb==1);
  // ringBlocks>1 keeps its spatial split.
  {
    int dp = (fused.ringBlocks == 1) ? HierDeepPipe() : 1;
    if (dp < 0) {  // "auto": depth = round(perPE chunkBytes / subBytes), clamp[1,16]
      // 16MiB sub-chunk: fills mlx5 NIC DMA while staying under the coherence window.
      const size_t sub = 16ull * 1024ull * 1024ull;
      size_t d = (fused.chunkBytes + sub / 2) / sub;
      dp = (d < 1) ? 1 : (d > 16 ? 16 : static_cast<int>(d));
    }
    // Snap dp down to the largest divisor of chunkBytes so sub-chunks are equal (no ragged
    // tail) and the count never exceeds the Python-sized flag budget.
    if (dp > 1 && fused.chunkBytes > 0) {
      while (dp > 1 && (fused.chunkBytes % static_cast<size_t>(dp)) != 0) --dp;
    }
    fused.deepPipe = dp;
  }
  // Scale-robust landing fence default: the deep-pipe put-with-signal AMO can beat its own
  // data landing for large sub-chunks, so auto-engage the quiet-drain fence whenever
  // deep-pipe runs unless forced off (MORI_HIER_DEEP_PIPE_QUIET=0). Bit-exact; deepPipe<=1
  // never enters this path.
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
  // Crown local-block flag (MORI_HIER_CROWN); default 0 => OFF. See HierCrownRing.
  fused.crownRing = HierCrownRing();
  return reinterpret_cast<int64_t>(&fused);
}

}  // namespace collective
}  // namespace mori
