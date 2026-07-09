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
// Put-signal is default-ON for the STANDALONE ring (UT path, see
// inter_node_ring_class.hpp) but the FUSED FSDP builders must keep it OFF unless
// the env is EXPLICITLY enabled, so the shipped E2E deferred/overlap bytes stay
// byte-identical (enabling put-signal on that path re-opened a consumer-landing
// loss drift). Returns true ONLY when MORI_HIER_RING_PUT_SIGNAL is set to a
// non-"0" value; an UNSET env yields false here (unlike the standalone default).
inline bool HierRingPutSignalExplicitlyOn() {
  const char* e = std::getenv("MORI_HIER_RING_PUT_SIGNAL");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// FUSED-REMOTE E2E COHERENCE lever (MORI_HIER_FUSE_REMOTE_RETOUCH, default OFF).
// After the fused remote-gather completion reader confirms every peer's SDMA
// push has landed in THIS PE's output (all flags set) + a system threadfence,
// re-touch the freshly-landed local output with a volatile-glc load (bypass the
// stale L2 line the consumer GEMM cached for the reused FSDP output buffer) +
// a normal store, republishing the fabric-coherent HBM value into the CU/L2
// domain. This is the director-mandated CU-coherent copy-out FUSED into the
// kernel (no host stall) -- the completion reader already waits the ACTUAL HW
// completion (flags fire only after each push's per-queue SdmaQueitThread), so
// the temporal landing is captured on-device; the re-touch closes the residual
// receiver-side L2 coherence gap that keeps fuse_remote E2E-drifting (+0.018).
inline bool HierFuseRemoteRetouchOn() {
  const char* e = std::getenv("MORI_HIER_FUSE_REMOTE_RETOUCH");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// ELASTIC REASSEMBLY lever (MORI_HIER_FUSE_REMOTE_ELASTIC, default OFF).
// In the pipelined FusedRingRemoteGather kernel the LOCAL-block CTA finishes its
// own-shard SDMA gather early (ring-independent, overlaps the RDMA ring) and then
// merely idles as the completion reader -- so during the reassembly TAIL only the
// reasm reassembly block(s) drive SDMA (queues 1..reasm) while queue 0 sits idle.
// That un-hidden ~50GB/s reassembly leg on the big embed/lm_head AGs is the 0.92x
// device-path wall. With this lever the local CTA, AFTER its own gather completes
// (queue 0 free), JOINS remote reassembly as an extra worker on queue 0, so the
// tail runs on reasm+1 SDMA queues instead of reasm. Byte-identical partition
// (workers 0..reasm handle ring channels f % (reasm+1)); default OFF unchanged.
inline bool HierFuseRemoteElasticOn() {
  const char* e = std::getenv("MORI_HIER_FUSE_REMOTE_ELASTIC");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// IN-KERNEL COPY-IN lever (MORI_HIER_FUSE_COPYIN, default OFF). The PERSISTENT-
// SINGLE-KERNEL port (director 13:44Z): mori's per-op cost is the AGGREGATE ramp of
// several serial GPU ops -- {host hipMemcpyAsync copy-IN of this PE's input into its
// ring slot} -> {entry barrier} -> {fused ring+gather kernel} -> {finish fence} --
// vs RCCL's ONE persistent fused kernel. This lever folds the copy-IN INTO the fused
// kernel: each ring channel CTA stages its OWN send sub-range of gInput into the
// local ring slot then __syncthreads() before the RDMA put, so the put sources valid
// data with NO cross-CTA dependency (channel bx writes exactly the bytes it sends).
// Combined with MORI_HIER_GEN_RING (drops the entry barrier) + slice_defer_fin
// (defers the finish fence) the whole AG collapses to a SINGLE host kernel launch --
// the untested aggregate collapse (copy-IN alone and GEN_RING alone were each
// separately NEUTRAL; the persistent-kernel thesis is that killing the inter-launch
// ramps TOGETHER closes the fixed ~0.3-0.5ms per-op floor that walls 64-128MB).
// Default OFF => the host copy-IN runs, byte-identical shipped path.
inline bool HierFuseCopyInOn() {
  const char* e = std::getenv("MORI_HIER_FUSE_COPYIN");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// DEEP-SQ WQE-DEPTH lever (MORI_HIER_WQE_DEPTH, default 1 = byte-identical).
// The crown big embed/lm_head inter-node put takes the fan-out path: the chunk is
// split across numQp warps -> numQp QPs, each warp issuing ONE whole-sub-range
// RDMA-WRITE WQE per QP. Adding more QPs (numQp 4->8) was neutral (Turn 2/25), so
// QP-count is not the fill lever. The UNtested axis is in-flight WQE DEPTH PER QP:
// with depth d each warp splits its sub-range into d back-to-back non-blocking
// puts on its SAME QP, so the NIC sees d queued WQEs per QP instead of 1 -- the
// device analogue of the host-proxy deep-SQ (depth~32 -> 48 GB/s on this mlx5)
// that beats the ~31 GB/s single-WQE-per-QP device fill. The union of the d
// 16B-aligned sub-puts tiles the sub-range exactly and rides the same QP in RC
// order, so the byte image AND the completion drain (per-QP quiet) are identical;
// only the SQ depth changes. depth<=1 => single put (shipped path unchanged).
inline int HierWqeDepth() {
  const char* e = std::getenv("MORI_HIER_WQE_DEPTH");
  if (e == nullptr || e[0] == '\0') return 1;
  int v = std::atoi(e);
  return v < 1 ? 1 : v;
}
// WALL-DECOMPOSITION PROFILER (MORI_HIER_PROFILE, default OFF = zero overhead).
// Director 18:33Z localization mandate: decompose the giant embed/lm_head AG's
// deferred-fence wait into inter-node-RDMA-land vs intra-node-SDMA-reassembly.
// When on, the FusedRingRemoteGatherKernel records GPU wall_clock64() landmarks
// into a __device__ global (g_hierProf) and rank 0 printf's the per-phase split
// for each big AG: total wall, inter-land fraction (when the LAST RDMA chunk's
// landing flag was observed by a reassembly worker), and intra-reassembly tail
// fraction (SDMA drain after the last inter land). All guarded by this flag so
// the shipped path stays byte- AND cycle-identical.
inline bool HierProfileOn() {
  const char* e = std::getenv("MORI_HIER_PROFILE");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// DEEP-SQ TEMPORAL PIPELINE lever (MORI_HIER_DEEP_PIPE=P, default 1 = off). The
// giant embed/lm_head AG wall is 54% inter-node RDMA fill + 46% intra-node SDMA
// reassembly, FULLY SERIAL at rb=1 (Turn 28 profile). RING_BLOCKS (spatial split)
// is a 1:1 wash because it splits the one full-numQp write into FEWER-QP channels,
// growing inter fill by exactly what it hides. This lever instead splits the chunk
// into P TEMPORAL sub-chunks issued BACK-TO-BACK on the SAME full numQp fan-out
// (deep SQ, full inter BW) with a PER-SUB-CHUNK put-with-signal, so sub-chunk p's
// landing flag fires (RC in-order, after its data) BEFORE p+1's -- a reassembly
// worker pushes sub-chunk p over XGMI while p+1.. still cross the NIC, hiding the
// 46% intra under the 54% inter with NO inter-fill growth. Clamped to [1,8] (the
// ring flags buffer holds npes==8 slots at rb==1). depth<=1 => shipped path.
inline int HierDeepPipe() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE");
  if (e == nullptr || e[0] == '\0') return 1;
  int v = std::atoi(e);
  if (v < 1) return 1;
  if (v > 16) return 16;
  return v;
}

// SIZE GATE for DEEP_PIPE (MORI_HIER_DEEP_PIPE_MAX_MB, default 0 = no limit).
// Turn 31 UT: DEEP_PIPE=2 at 256MB = 1.146x bit-exact but NON-deterministic
// (Turn 30: 64MB/P2 MISMATCH), and 466MB (the E2E giant embed/lm_head AG) CRASHES
// (SIGABRT). The per-sub-chunk device landing flag (send-CQE quiet + AMO) is not a
// scale-robust landing proof on this mlx5 provider (write-with-imm recv-CQE is
// HW-unavailable). So the temporal pipeline (real BW win) is only SAFE below a
// byte threshold. This gate engages deepPipe ONLY for chunks <= MAX_MB per PE and
// forces every larger chunk (incl the 466MB giant AGs) onto the whole-chunk CROWN
// fence (single quiet + AMO, bit-exact at 466MB) so E2E stays bit-exact while the
// medium AGs still ride the 1.1x+ pipeline. 0 => current behaviour (no gate).
inline size_t HierDeepPipeMaxBytes() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_MAX_MB");
  if (e == nullptr || e[0] == '\0') return 0;
  long v = std::atol(e);
  if (v <= 0) return 0;
  return static_cast<size_t>(v) * 1024ull * 1024ull;
}
// WRITE_WITH_IMM per-sub-chunk landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_IMM,
// default OFF). recv-CQE = definitive remote-landing (RC in-order per QP), but the
// WRITE_WITH_IMM path is HW-unavailable on this mlx5 provider (asserts out), so this
// stays OFF here; kept for portability. Only meaningful when deepPipe>1.
inline bool HierDeepPipeImmOn() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_IMM");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// QUIET-FENCE per-sub-chunk landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_QUIET,
// default OFF). The put-with-signal AMO (deepPipeImm==0) fails bit-exact >=64MB
// because the AMO can beat its own large-transfer data landing; WRITE_WITH_IMM
// (deepPipeImm==1) is HW-unavailable on this mlx5 provider. This is the third,
// scale-robust option: dedicate ONE QP per temporal sub-chunk, then instead of
// fusing put+AMO, issue a PLAIN put and drain THAT QP's send-CQ with
// ShmemQuietThread(pe, qpId) (== the sub-chunk's data has landed remotely, the
// proven teamC quiet-drain fence, RC in-order) BEFORE the separate flag AMO. So
// chunkReadyFlags[p] is published only after sub-chunk p physically landed --
// bit-exact at scale (the mechanism teamC's ring uses for its >=32MiB parity)
// while preserving the temporal pipeline (p's flag fires before p+1's data
// finishes, since each sub-chunk drains its own QP in order). Only meaningful
// when deepPipe>1; takes precedence over the racy put-signal path.
inline bool HierDeepPipeQuietOn() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_QUIET");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}

// HOST-PROXY INTER + DEVICE REASSEMBLY composition (MORI_HIER_HOSTPROXY_REASM).
// The device deep-pipe (temporal sub-chunk) needs a per-sub-chunk landing signal;
// BOTH device options are refuted on this mlx5 provider (WRITE_WITH_IMM recv-CQE
// HW-unavailable; per-sub-chunk quiet+AMO races >=64MB / crashes at 466MB). The
// sole scale-robust per-sub-chunk landing proof is the HOST send-CQ drain (the
// proven host-drain fence, bit-exact at 466MB). This lever hands the inter-node
// fill to a HOST proxy that posts sub-chunk p's cross-node RDMA writes into the
// ring buffer, drains its send-CQ (== sub-chunk landed remotely), and publishes
// chunkReadyFlags[p] from the host (system-scope, device-visible). The DEVICE
// kernel then runs ONLY the intra SDMA reassembly pipeline, spinning on the
// host-published chunkReadyFlags[p] exactly as it does for the device-published
// flags today -- so sub-chunk p's XGMI reassembly overlaps sub-chunk p+1 still on
// the NIC, breaking the giant-AG 54/46 serial wall WITHOUT the racy device signal.
// When ON the device ring-send blocks (bx < ringBlocks) SKIP the RDMA send (the
// host owns the inter leg) but the reassembly workers + completion reader are
// UNCHANGED. Default 0 = OFF (byte-identical shipped path: device posts inter and
// publishes the flags as before).
inline int HierHostProxyReasm() {
  const char* e = std::getenv("MORI_HIER_HOSTPROXY_REASM");
  if (e == nullptr || e[0] == '\0') return 0;
  return std::atoi(e) != 0 ? 1 : 0;
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

// Fused hierarchical param-contiguous SubGroup gather. ONE launch replaces the
// per-(node-block, param) loop that ``HierAllGather.enqueue_param_contiguous``
// used to issue (N_nodes * N_params separate SubGroup launches, whose launch
// overhead erased the copy-out saving vs RCCL). Each of ``G`` group members
// pushes this PE's shard (group position ``groupPos`` == this node's local rank
// ``g``) DIRECTLY into the registered user output in PARAM-CONTIGUOUS layout:
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
  // Transport-level flag-can't-beat-data: when non-zero, the single-warp RDMA
  // ring send fuses the data WRITE and the completion-flag AMO into ONE
  // ShmemPutMemNbiSignal call so the signal WQE rides the SAME QP strictly
  // AFTER the data WRITE. RC in-order execution then guarantees the remote
  // peer's data has physically LANDED before its flag is observable -- closing
  // the residual FSDP loss completion race without any host sync (the flag can
  // never beat its data). Default 0 = the historical separate put + quiet + AMO.
  int usePutSignal = 0;
  // Phase-6 WRITE_WITH_IMM (env MORI_HIER_RING_WRITE_IMM, default 0). On the
  // single-warp cross-node (RDMA) ring path, replace the data PUT + QP quiet +
  // flag AMO with an RDMA_WRITE_WITH_IMM and have the receiver consume the
  // recv-CQ completion instead of spinning the flag. The recv-CQE cannot be
  // observed before the write payload has landed globally, so this closes the
  // remote-landing stale-read race that no device-side barrier/quiet fixed
  // (13 avenues refuted; only host sync worked) WITHOUT the host stall. Default
  // 0 = the historical put+quiet+flag path, byte-for-byte unchanged.
  int useWriteImm = 0;
  // RDMA-READ (PULL) ring (env MORI_HIER_RING_READ, default 0). On the
  // single-round (ringSize==2) all-RDMA inter-node phase -- exactly the 2-node
  // hierarchical AG this cluster runs -- the chunk each PE needs is prevPeer's
  // OWN chunk, already present after the intra prepare barrier. Instead of
  // relying on the peer to PUSH it (a GPU-initiated RDMA WRITE, the measured
  // 0.71x per-QP throughput wall on mlx5), PULL it with an RDMA READ. A READ
  // completion drained by our OWN quiet is a CONSUMER-side landing guarantee:
  // the bytes are physically in this PE's ring buffer and, with a system fence,
  // visible to its CUs -- no cross-PE flag AMO, no receiver spin, no remote-
  // landing race (the E2E accuracy race the push+flag path exposes). Byte-
  // identical result (same slot, same bytes). Default 0 = the push path,
  // byte-for-byte unchanged.
  int useRead = 0;
  // GENERATION-COUNTER barrier-free ring (env MORI_HIER_GEN_RING, default 0).
  // When non-zero this holds the monotonically-increasing per-op generation
  // number (op 1, 2, 3, ...). On the classic single-increment flag path
  // (expectedRecvSig==1, no put-signal / write-imm / fan-out) the sender's
  // AMO_ADD(1) is left to ACCUMULATE across ops (the kernel skips the per-op
  // flag reset), so slot k holds exactly ``opGen`` after ``opGen`` ops. The
  // receiver then waits for the slot to reach ``opGen`` instead of 1. Because
  // the flags are never reset, the prepare-time ENTRY barrier (whose sole job
  // was to order every PE's op-end reset BEFORE any peer's next-op increment)
  // is no longer needed and is skipped in prepare_stream -- removing one of the
  // two global on-stream barriers per ring round. The trailing finish reuse
  // barrier is kept, so ring-buffer reuse ordering is unchanged. Default 0
  // keeps the reset+entry-barrier path byte-for-byte identical.
  uint64_t opGen = 0;
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
  // Phase-6: propagate the ring completion-protocol flags into the FUSED path.
  // Historically the fused ring dropped these (defaulted to the plain
  // put+quiet+flag protocol), so the big embed/lm_head cross-node AG -- which
  // under FSDP runs through THIS fused kernel -- never saw usePutSignal /
  // useWriteImm even when the env enabled them. That is exactly why WRITE_IMM
  // was bit-exact in the standalone ring kernel yet never closed the FSDP
  // remote-landing race. Carrying them here lets the fused ring engage the
  // fanOut WRITE_WITH_IMM (recv-CQ landing proof) on the big AG. Default 0 =>
  // byte-identical to the shipped fused path.
  int usePutSignal = 0;
  int useWriteImm = 0;

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
  // Fused FSDP path: put-signal only when EXPLICITLY env-enabled (standalone
  // default-ON must NOT leak into the E2E deferred/overlap bytes -- keeps loss
  // byte-identical to native). See HierRingPutSignalExplicitlyOn.
  fused.usePutSignal = HierRingPutSignalExplicitlyOn() ? r->usePutSignal : 0;
  fused.useWriteImm = r->useWriteImm;

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
// PHASE 4: FUSED inter-node ring + intra-node REMOTE-block reassembly (pipelined)
// ============================================================================
// The shipped ``FusedRingLocalGatherKernel_u32`` fuses the ring with only the
// LOCAL node-block gather (the ring-INDEPENDENT half); the REMOTE-block
// reassembly still runs as a SEPARATE launch AFTER the whole ring + its global
// finish barrier (the two-serial-phases cost: the NIC sits idle during the XGMI
// reassembly and vice-versa -- the 143-vs-168 GB/s gap). This kernel closes that
// by PIPELINING: the ring runs as ``ringBlocks`` channels, each publishing
// ``chunkReadyFlags[bid]`` the instant its sub-range lands; a matching set of
// ``ringBlocks`` reassembly blocks each spin on chunkReadyFlags[j] and, the
// instant sub-range j is ready, SDMA-push that sub-range of EVERY remote block
// straight into the registered output over XGMI -- so sub-range j's reassembly
// overlaps ring channel j+1 still crossing the NIC. Because each PE reassembles
// a remote block by pushing FROM ITS OWN ring buffer (slot m holds node m's chunk
// in ring order, see AllgatherInterNodeRing.full_tensor), the ONLY dependency is
// this PE's own ring landing -- a purely local flag spin, NO global finish
// barrier and NO copy-OUT scratch. Grid = 2*ringBlocks + 1: [0,ringBlocks) ring,
// [ringBlocks] the local-block gather, (ringBlocks, 2*ringBlocks] the remote
// reassembly (block j = blockIdx.x - ringBlocks - 1). Default OFF (env
// MORI_HIER_FUSE_REMOTE); the shipped serial path is untouched.
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
  int usePutSignal = 0;
  int useWriteImm = 0;
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
  size_t gElementCount;         // per-slice u32 lanes (== count)
  size_t gDstBaseOffset;        // bytes: local block base (nodeId*blockCount*4)
  size_t gDstSlotStrideBytes;   // bytes: full-slice stride (== chunkBytes)
  uint64_t gFlagVal;

  // --- remote reassembly (Phase B, m != nodeId; reads the ring buffer) ---
  int numNodes;  // N == ringSize
  int nodeId;    // this node's block index (skipped by the remote reassembly)
  // Number of reassembly blocks, DECOUPLED from ringBlocks so the XGMI
  // reassembly can be parallelised (like the multi-block copy-OUT) even when the
  // ring runs as a single channel (ringBlocks==1). Each reassembly block owns a
  // disjoint 16B-aligned byte sub-range of the chunk and waits until ALL ring
  // channels have landed (spin over the ringBlocks flags) before pushing its
  // sub-range over XGMI. 0 => legacy behaviour (reassemblyBlocks == ringBlocks).
  int reassemblyBlocks = 0;
  // CU-coherent re-touch of the landed local output in the completion reader
  // (see HierFuseRemoteRetouchOn). 0 = OFF (byte-identical shipped path).
  int retouchOut = 0;
  // Elastic reassembly (see HierFuseRemoteElasticOn). 1 => the local-block CTA
  // joins remote reassembly on SDMA queue 0 after its own gather, so the tail
  // uses reasm+1 concurrent SDMA queues. 0 = OFF (byte-identical shipped path).
  int elasticReasm = 0;
  // Deep-SQ WQE depth per QP for the inter-node ring put (see HierWqeDepth).
  // 1 = single WQE per QP (byte-identical shipped path); d>1 posts d back-to-back
  // sub-puts per QP so the NIC SQ carries d in-flight WQEs per QP.
  int wqeDepth = 1;
  // Wall-decomposition profiler (see HierProfileOn). 0 = OFF (cycle-identical).
  int profile = 0;
  // Deep-SQ temporal pipeline depth (see HierDeepPipe). P>1 splits the chunk into
  // P temporal sub-chunks with per-sub-chunk landing flags so reassembly overlaps
  // the still-in-flight later sub-chunks. 1 = OFF (byte-identical shipped path).
  int deepPipe = 1;
  // Deep-SQ temporal pipeline landing fence: 0 = per-sub-chunk put-with-signal AMO
  // (fails bit-exact >=64MB/P4 -- the AMO can beat its own large-transfer data);
  // 1 = per-sub-chunk RDMA_WRITE_WITH_IMM (recv-CQE = definitive remote-landing,
  // RC in-order per QP). See HierDeepPipeImmOn. Only meaningful when deepPipe>1.
  int deepPipeImm = 0;
  // Deep-SQ temporal pipeline QUIET landing fence (see HierDeepPipeQuietOn). 1 =>
  // each temporal sub-chunk rides its OWN QP with a PLAIN put; thread drains that
  // QP's send-CQ (ShmemQuietThread(pe,qpId) = sub-chunk landed remotely) BEFORE the
  // separate flag AMO, so the flag never fires ahead of the data landing (bit-exact
  // at scale, unlike the racy fused put-signal). 0 = OFF. Only when deepPipe>1.
  int deepPipeQuiet = 0;
  // HOST-PROXY INTER + DEVICE REASSEMBLY (see HierHostProxyReasm). 1 => the device
  // ring-send blocks SKIP the RDMA send (a host proxy owns the inter leg and
  // publishes chunkReadyFlags[p] from the host after its send-CQ drains); the
  // device reassembly workers + completion reader are UNCHANGED. 0 = OFF
  // (byte-identical shipped path: device posts inter and publishes the flags).
  int hostProxyInter = 0;
  // IN-KERNEL COPY-IN (see HierFuseCopyInOn). 1 => each ring channel CTA stages its
  // own send sub-range of gInput into the local ring slot before the put (the host
  // hipMemcpyAsync copy-IN is skipped on the Python side, prepare_stream_in_place).
  // 0 = OFF (byte-identical shipped path: host copies input into the slot).
  int fuseCopyIn = 0;
};

// Builder: merge an already-built ring args (CclInterNodeRingArgs) and gather
// args (CclAllgatherSubGroupArgs<uint32_t>, primed for the LOCAL block) plus the
// pipeline extras into one CclFusedRingRemoteGatherArgs. Mirrors
// BuildFusedRingLocalGatherArgs so the existing prepare_* paths stay byte-
// identical; this is pure additive glue. Inert until the Python launcher is wired.
inline int64_t BuildFusedRingRemoteGatherArgs(int64_t ringArgsPtr, int64_t gatherArgsPtr,
                                              int ringBlocks, int64_t chunkReadyFlagsPtr,
                                              int numNodes, int nodeId,
                                              int reassemblyBlocks = 0) {
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
  // Fused FSDP path: put-signal only when EXPLICITLY env-enabled (standalone
  // default-ON must NOT leak into the E2E deferred/overlap bytes -- keeps loss
  // byte-identical to native). See HierRingPutSignalExplicitlyOn.
  fused.usePutSignal = HierRingPutSignalExplicitlyOn() ? r->usePutSignal : 0;
  fused.useWriteImm = r->useWriteImm;
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
  // In-kernel single-CTA re-touch is thread-starved on the big AG (512 threads
  // over the whole output serializes -> step timeout). The re-touch is instead
  // done by a FULL-GRID L2CoherentRetouchKernel epilogue launched from Python
  // after the fused kernel's completion fence (stream-ordered, all bytes landed).
  fused.retouchOut = 0;
  fused.elasticReasm = HierFuseRemoteElasticOn() ? 1 : 0;
  fused.wqeDepth = HierWqeDepth();
  fused.profile = HierProfileOn() ? 1 : 0;
  // Deep-SQ temporal pipeline only engages on the single-channel ring (rb==1, the
  // giant-AG fan-out path); RING_BLOCKS>1 (multiBlock) keeps its spatial split.
  // SIZE GATE (Turn 31): the per-sub-chunk device landing flag is only bit-exact
  // below a byte threshold (466MB giant AGs CRASH, >=64MB races) -- so engage the
  // pipeline ONLY when this PE's chunk <= MORI_HIER_DEEP_PIPE_MAX_MB; every larger
  // chunk falls through to the whole-chunk crown fence (bit-exact at 466MB). 0 => no
  // gate (legacy).
  {
    // dpMax is the PER-SUB-CHUNK coherence window (chunkBytes/dp), NOT the total
    // chunk. The device per-sub-chunk landing flag is a pipeline HINT (E2E landing
    // is anchored by the crown DEFER_HOSTSYNC host fence, T43 fence-role control);
    // the real HW hazards are (a) the CRASH on very large sub-chunks (466MB giant
    // AG) and (b) the >=32MB sub-chunk send-CQE-before-SDMA-coherent race. Gating on
    // the SUB-CHUNK keeps DEEP_PIPE=4 engaged on the 34-67MB steady-state decoder AGs
    // (8.5-16.75MB sub-chunks, strictly under 32MB) while the 466MB giant AG (116MB
    // sub-chunk) falls through to the whole-chunk crown fence. Strict '<' so a 32MB
    // sub-chunk (e.g. 64MB@dp=2) is caged, not engaged. Explicit MORI_HIER_DEEP_PIPE_MAX_MB
    // overrides the window. dp<=1 => byte-identical shipped path.
    size_t dpWindow = HierDeepPipeMaxBytes();
    const int dp = (fused.ringBlocks == 1) ? HierDeepPipe() : 1;
    if (dp > 1 && dpWindow == 0) dpWindow = 32ull * 1024ull * 1024ull;
    size_t subChunk = (dp > 1) ? (fused.chunkBytes / static_cast<size_t>(dp))
                               : fused.chunkBytes;
    fused.deepPipe = (dpWindow == 0 || subChunk < dpWindow) ? dp : 1;
  }
  fused.deepPipeImm = HierDeepPipeImmOn() ? 1 : 0;
  fused.deepPipeQuiet = HierDeepPipeQuietOn() ? 1 : 0;
  fused.hostProxyInter = HierHostProxyReasm();
  fused.fuseCopyIn = HierFuseCopyInOn() ? 1 : 0;

  return reinterpret_cast<int64_t>(&fused);
}

}  // namespace collective
}  // namespace mori
