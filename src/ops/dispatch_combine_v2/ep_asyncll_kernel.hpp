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
//
// EP AsyncLL on cco, ported from src/ops/dispatch_combine/low_latency_async.cpp.
//
// What AsyncLL is, and why it is a separate family from dispatch_ll: the work is
// cut into nine kernels so the SEND half and the RECV half are separate launches
// with a host-visible seam between them. The caller runs
//
//     dispatch_send()  ->  [its own compute]  ->  dispatch_recv()
//
// and the transfers drain under that compute. dispatch_ll does send and recv in
// one kernel with a grid-wide DispatchSync at the end (EpDispatchInterNodeV2LL_body),
// so there is no seam to put anything in. That seam is the whole feature; the
// routing differs too (AsyncLL deduplicates per destination RANK, v2_ll does not
// deduplicate at all and sends one entry per node per token).
//
// The transport is ccoSdma, the intra-node copy engine. v1 reached the same
// engines through shmem's ShmemPutMemNbiThread, which dispatches per peer on
// globalGpuStates->transportTypes[pe] and lands on SDMA for a same-node peer, so
// this is the same hardware path with a different front end. Two deliberate
// departures from v1, both measured as wins on MI355X (see the SendTransfer and
// RecvTransfer comments):
//
//   * the per-peer payload goes out as ONE copy, not sliced per QP;
//   * the token count rides a single slot per source rank rather than one per QP.
//
// Cross-node peers go over ccoGda instead, through the same EpInterNode* shim
// the v2 / v2_ll families use. The protocol is unchanged -- same slots, same
// count word, same epoch -- and the branch is on `destPe / gpuPerNode != myNode`
// in the four transfer passes. Two things differ by transport, both deliberate:
// a cross-node payload is STRIPED over numQpPerPe QPs where a same-node one is
// sent whole, and the remote count and epoch are atomic ADDs because that is the
// only remote update ccoGda offers at an arbitrary window offset.
//
// Note that AsyncLL talks to every peer directly. On a fabric whose rails are not
// connected to each other this cannot work at all, and the symptom is not an
// error: QP setup happens over TCP and succeeds, then no data moves. The
// v1/v1_ll families aggregate through a same-rail proxy and are immune, which is
// why MORI_ENABLE_RAIL_ONLY is only ever claimed for them. tools/ has a probe
// for the cco half of this; the fabric half is an ib_write_bw across two
// different rails.

#pragma once

// Brings DEF_COMMON_VARS, index_t, EpDispatchCombineArgs, the EpInterNode* ccoGda
// shim used by the internode hooks, and the MORI_EP_INTERNODE_CCO_ENTRY* macros
// the generated TU expands. Including it rather than restating any of that is
// what keeps the two families from drifting apart.
#include "src/ops/dispatch_combine_v2/ep_internode_kernel.hpp"
#include "src/ops/dispatch_combine_v2/ep_scatter_quant.hpp"

#ifndef MORI_ASYNCLL_EARLY_NOTIFY
#define MORI_ASYNCLL_EARLY_NOTIFY 0
#endif
#ifndef MORI_ASYNCLL_FUSED_RECV
#define MORI_ASYNCLL_FUSED_RECV 0
#endif
#if MORI_ASYNCLL_FUSED_RECV && !MORI_ASYNCLL_EARLY_NOTIFY
#error "MORI_ASYNCLL_FUSED_RECV requires MORI_ASYNCLL_EARLY_NOTIFY"
#endif
#ifndef MORI_ASYNCLL_DIRECT
#define MORI_ASYNCLL_DIRECT 0
#endif
// Direct rows placed with one ring reservation per peer (cco batched placement, chunked to half
// the ring) instead of one per wave-batch.
#ifndef MORI_ASYNCLL_BATCH_POST
#define MORI_ASYNCLL_BATCH_POST 1
#endif
// Rendered from EpInterNodeCfg::asyncLlFuseMeta. 1 (direct only): segmented dispatch writes the
// metadata rows in SlotAssign, and SendTransferSeg posts the scale rows with the hidden rows and
// the count/mode words; SendCopySeg is a no-op.
#ifndef MORI_ASYNCLL_FUSE_META
#define MORI_ASYNCLL_FUSE_META 0
#endif
// Combine receive reduce: 4 = the shared WarpAccum; 16 = load-first 16 B vectors,
// MORI_ASYNCLL_COMBINE_UNROLL deep; 17 = wave-uniform buffer loads into registers (gfx942/950).
// MORI_ASYNCLL_COMBINE_VEC sets the compact receive, MORI_ASYNCLL_COMBINE_VEC_SEG the segmented.
#ifndef MORI_ASYNCLL_COMBINE_VEC
#define MORI_ASYNCLL_COMBINE_VEC 4
#endif
#ifndef MORI_ASYNCLL_COMBINE_VEC_SEG
#if defined(__gfx942__) || defined(__gfx950__)
#define MORI_ASYNCLL_COMBINE_VEC_SEG 17
#else
#define MORI_ASYNCLL_COMBINE_VEC_SEG MORI_ASYNCLL_COMBINE_VEC
#endif
#endif
#if (MORI_ASYNCLL_COMBINE_VEC == 17 || MORI_ASYNCLL_COMBINE_VEC_SEG == 17) && \
    defined(__HIP_DEVICE_COMPILE__) && !(defined(__gfx942__) || defined(__gfx950__))
#error "MORI_ASYNCLL_COMBINE_VEC(_SEG) 17 uses the gfx942/gfx950 buffer descriptor"
#endif
#ifndef MORI_ASYNCLL_COMBINE_UNROLL
#define MORI_ASYNCLL_COMBINE_UNROLL 2
#endif
// 1: segmented receives load addresses and map entries before the wait, drain beside the poll,
// and skip the wait in blocks with no work.
#ifndef MORI_ASYNCLL_RECV_TRIM
#if defined(CCO_GFX125X)
#define MORI_ASYNCLL_RECV_TRIM 0
#else
#define MORI_ASYNCLL_RECV_TRIM 1
#endif
#endif
#if MORI_ASYNCLL_DIRECT && !MORI_ASYNCLL_EARLY_NOTIFY
#error "MORI_ASYNCLL_DIRECT requires MORI_ASYNCLL_EARLY_NOTIFY"
#endif
#if MORI_ASYNCLL_FUSE_META && !MORI_ASYNCLL_DIRECT
#error "MORI_ASYNCLL_FUSE_META requires MORI_ASYNCLL_DIRECT"
#endif
// Segmented sends put the count, mode and epoch words in the SDMA packets as immediates instead
// of staging them with CU stores and a system fence: 1 = one WRITE_LINEAR per word, 2 = the
// polled word (count, epoch) as WRITE_LINEAR of its high dword then a FENCE of its low dword, so
// it lands only after the queue's earlier writes complete. The dispatch half needs FUSE_META.
#ifndef MORI_ASYNCLL_INLINE_COUNT
#if defined(CCO_GFX125X)
#define MORI_ASYNCLL_INLINE_COUNT 0
#else
#define MORI_ASYNCLL_INLINE_COUNT 2
#endif
#endif
// With INLINE_COUNT, a system release after combine's count clear. 0 drops it, which is safe only
// on a truly uncached window: CCO_UNCACHED_WINDOW unset and a ROCm whose VMM honours uncached.
#ifndef MORI_ASYNCLL_INLINE_CLEAR_FENCE
#define MORI_ASYNCLL_INLINE_CLEAR_FENCE 1
#endif
#if MORI_ASYNCLL_INLINE_COUNT < 0 || MORI_ASYNCLL_INLINE_COUNT > 2
#error "MORI_ASYNCLL_INLINE_COUNT must be 0, 1 or 2"
#endif
#if MORI_ASYNCLL_INLINE_COUNT && defined(CCO_GFX125X)
#error "MORI_ASYNCLL_INLINE_COUNT uses the gfx9 SDMA WRITE/FENCE layouts"
#endif

namespace mori {
namespace ops {
namespace v2 {
namespace asyncll {

// The host resolves this protocol per op and carries it in every Plan's cfg.
// LSA and multi-node configurations retain the original protocol.
template <EpInterNodeKernelCfg kConfig>
inline constexpr bool EarlyNotify =
    MORI_ASYNCLL_EARLY_NOTIFY && kConfig.worldSize == kConfig.gpuPerNode;

// Direct sources: the engines read the caller's tokens where they are instead of a
// gathered staging copy. Dispatch posts one packet per (token, peer) from the input,
// then the peer's metadata block, then its count; combine posts each peer's rows straight
// from the expert output, where dispatch's RecvCopy already laid them out contiguously by
// source rank. The packets are small and many, which only pays off once the ring carries
// them back to back. Both inputs are read until the matching recv drains the queues.
template <EpInterNodeKernelCfg kConfig>
inline constexpr bool Direct = MORI_ASYNCLL_DIRECT && EarlyNotify<kConfig>;

// ---------------------------------------------------------------------------
// Signal slots.
//
// v1 indexes the recv-count region by (srcPe, qpId) because its transfer is
// striped over numQpPerPe RDMA queue pairs and each stripe announces its own
// chunk. An SDMA queue is not a QP: queues exist so independent WARPS can issue
// in parallel, and splitting one peer's copy across them is measurably slower
// (one peer is one xGMI link, so the pieces only get smaller -- 600 KB in one
// op runs at ~46 GB/s on this part, the same bytes in two at ~35). This port
// therefore sends each peer's payload whole and needs exactly one count slot per
// source rank. The region is still allocated at v1's per-QP size, so these
// indices address a prefix of it and the unused tail stays zero.
// ---------------------------------------------------------------------------
__device__ __forceinline__ size_t RecvCountSlot(int srcPe) {
  return static_cast<size_t>(srcPe) * sizeof(uint64_t);
}

// How many tokens this rank actually has for `pe`, clamped to the slice it can
// hold. destPeTokenCounter is cleared once per round by RecvCopy, so a second
// dispatch_send with no recv between keeps climbing: the raw value then names
// more tokens than the per-PE stride, and every reader of it -- the transfer's
// byte count, and the count the peer is told to expect -- would run off the end
// of the slice. SlotAssign refuses to encode a slot past the stride for the same
// reason; this is the other half of that bound, so the two agree on where the
// slice ends. The host's call-order check is what keeps this unreachable.
__device__ __forceinline__ int SendTokenCount(const EpInterNodeDeviceCfg& config, int raw) {
  const int cap = config.MaxNumTokensToSendPerRank();
  return (raw < cap) ? raw : cap;
}

// The local staging for a value that is about to be SDMA'd out. ccoSdma moves
// window to window and has no immediate form, so a number has to exist at a real
// address before it can be sent; v1 put the same number straight in the WQE with
// ShmemPutUint64ImmNbiThread. Two words per destination: [0] the token count,
// [1] the combine barrier epoch. They are in flight at different times, but
// separate slots cost 8 bytes per peer and remove the question.
__device__ __forceinline__ size_t SendCountSlot(int destPe) {
  return static_cast<size_t>(destPe) * 2 * sizeof(uint64_t);
}
__device__ __forceinline__ size_t SendFlagSlot(int destPe) {
  return (static_cast<size_t>(destPe) * 2 + 1) * sizeof(uint64_t);
}

// One issuing warp per queue, which is the rule that matters for SDMA: several
// warps posting to one queue is correct but serialises on the in-order commit
// chain. The transfer kernels run one issuing warp per destination PE, so keying
// the queue on the PE gives each its own as long as the comm was created with
// enough; the modulo only folds peers together when it was not.
// The reverse map's encoding: recv slot -> which rank's which input token landed
// there. This is a PUBLIC output -- callers decode it as `tis / maxTokPerRank`,
// `tis % maxTokPerRank` -- so the stride is maxTokPerRank and NOT the wider
// FlatTokenIndex stride the forward slot map uses. EpSrcTokIndex in
// ep_intranode_kernel.hpp carries the same contract for the intranode family,
// with a comment about FlyDSL publishing the identical encoding; a handle has to
// decode the same way whichever kernel produced it.
__device__ __forceinline__ int SrcTokenIndex(const EpInterNodeDeviceCfg& config, int pe,
                                             int srcTokId) {
  return pe * config.MaxNumTokensToSendPerRank() + srcTokId;
}

// `offset` (args.sdmaQueueOffset) is per op, so ops that opt in post to their own
// queue and a recv's drain no longer waits for another op's copies. One op's
// traffic to a peer still shares one queue, so the protocol's ordering holds.
__device__ __forceinline__ int QueueForPe(const ::mori::cco::ccoDevComm& comm, int destPe,
                                          int offset) {
  const uint32_t n = comm.sdma.sdmaNumQueue;
  if (n == 0) return 0;
  return static_cast<int>((static_cast<uint32_t>(destPe) + static_cast<uint32_t>(offset)) % n);
}

// The first active lane places the payload, the second its ready word, in one
// reservation and with one doorbell. A zero-byte payload still sends the word.
// Keep exceptionally large, multi-packet copies separate: the chunked put's
// per-lane loop must not interleave the ready word with an unfinished payload.
// `dst`/`notifyDst` are peer addresses as mapped here, `src`/`notifySrc` local ones.
__device__ __forceinline__ void PostPayloadAndNotifyPtr(const ::mori::cco::ccoDevComm& comm,
                                                        int peer, int queueId, void* dst, void* src,
                                                        size_t bytes, void* notifyDst,
                                                        void* notifySrc) {
  ::mori::cco::ccoSdma sdma{comm};
  const int lane = threadIdx.x & (warpSize - 1);
  if (bytes > ::mori::cco::CCO_SDMA_MAX_COPY_BYTES) {
    sdma.putPtr<::mori::cco::ccoCoopWarp>(peer, dst, src, bytes, queueId);
    if (lane == 0) sdma.putPtr(peer, notifyDst, notifySrc, sizeof(uint64_t), queueId);
  } else if ((lane == 0 && bytes != 0) || lane == 1) {
    const bool notify = lane == 1;
    sdma.putPtr<::mori::cco::ccoCoopThread, false, false, ::mori::cco::ccoSdmaOptFlagsDefault,
                ::mori::cco::ccoSdmaThreadSameQueue>(peer, notify ? notifyDst : dst,
                                                     notify ? notifySrc : src,
                                                     notify ? sizeof(uint64_t) : bytes, queueId);
  }
}

__device__ __forceinline__ void PostPayloadAndNotify(
    const ::mori::cco::ccoDevComm& comm, int peer, int queueId, ::mori::cco::ccoWindow_t dstWin,
    size_t dstOffset, ::mori::cco::ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
    ::mori::cco::ccoWindow_t notifyDstWin, size_t notifyDstOffset,
    ::mori::cco::ccoWindow_t notifySrcWin, size_t notifySrcOffset) {
  PostPayloadAndNotifyPtr(comm, peer, queueId,
                          ::mori::cco::ccoGetLsaPeerPtr(dstWin, peer, dstOffset),
                          ::mori::cco::ccoGetLocalPtr(srcWin, srcOffset), bytes,
                          ::mori::cco::ccoGetLsaPeerPtr(notifyDstWin, peer, notifyDstOffset),
                          ::mori::cco::ccoGetLocalPtr(notifySrcWin, notifySrcOffset));
}

// A region's address in LSA rank `peer`'s memory as mapped here, and in this rank's.
__device__ __forceinline__ void* RegionPeerPtr(const EpInterNodeRegion region, int peer,
                                               size_t offset) {
  return ::mori::cco::ccoGetLsaPeerPtr(EpInterNodeWindow(region), peer,
                                       EpInterNodeOffset(region, offset));
}
__device__ __forceinline__ void* RegionLocalPtr(const EpInterNodeRegion region, size_t offset) {
  return ::mori::cco::ccoGetLocalPtr(EpInterNodeWindow(region), EpInterNodeOffset(region, offset));
}

// MORI_ASYNCLL_INLINE_COUNT: the engine writes each 64-bit word from the packet itself, so it
// needs no staging store or fence. A plain word (the mode word) is read only after a trigger
// placed behind it on the same queue; a trigger (count, epoch) is what the receiver polls.
template <EpInterNodeKernelCfg kConfig>
inline constexpr bool InlineCount = MORI_ASYNCLL_INLINE_COUNT != 0 && EarlyNotify<kConfig>;
constexpr uint64_t kInlineWordBytes = ::mori::cco::CCO_SDMA_WRITE64_BYTES;
constexpr uint64_t kInlineTriggerBytes =
    MORI_ASYNCLL_INLINE_COUNT == 2
        ? ::mori::cco::CCO_SDMA_WRITE32_BYTES + ::mori::cco::CCO_SDMA_FENCE_BYTES
        : ::mori::cco::CCO_SDMA_WRITE64_BYTES;

__device__ __forceinline__ void PlaceInlineWord(uint32_t* ring, uint64_t at, void* dst,
                                                uint64_t value) {
  ::mori::cco::ccoSdma::placeWrite64(ring, at, dst, value);
}

// Mode 2: high dword by WRITE, then low dword by FENCE, written once every earlier write on the
// queue has completed. A poller seeing new-high/old-low never matches: a count's high half is
// always 0 over a cleared slot, and an epoch's low half changes every round.
__device__ __forceinline__ void PlaceInlineTrigger(uint32_t* ring, uint64_t at, void* dst,
                                                   uint64_t value) {
#if MORI_ASYNCLL_INLINE_COUNT == 2
  ::mori::cco::ccoSdma::placeWrite32(ring, at, static_cast<uint8_t*>(dst) + sizeof(uint32_t),
                                     static_cast<uint32_t>(value >> 32));
  ::mori::cco::ccoSdma::placeFence(ring, at + ::mori::cco::CCO_SDMA_WRITE32_BYTES, dst,
                                   static_cast<uint32_t>(value));
#else
  ::mori::cco::ccoSdma::placeWrite64(ring, at, dst, value);
#endif
}

// One lane: trigger `value` to `dst` (as mapped here) on (peer, queueId), behind everything
// already reserved on that queue. Its own reservation and doorbell.
__device__ __forceinline__ void PostInlineTrigger(const ::mori::cco::ccoDevComm& comm, int peer,
                                                  int queueId, void* dst, uint64_t value) {
  ::mori::cco::ccoSdma sdma{comm};
  ::mori::cco::ccoSdmaQueueDeviceHandle* qh = sdma.queueHandle(peer, queueId);
  const uint64_t base = ::mori::cco::ccoSdma::reserve(qh, kInlineTriggerBytes);
  PlaceInlineTrigger(::mori::cco::ccoSdma::ring(qh), base, dst, value);
  ::mori::cco::ccoSdma::publish(qh, base, base + kInlineTriggerBytes);
}

// PostPayloadAndNotifyPtr with the notify word as an immediate trigger: lane 0 places the payload
// and the trigger in one reservation with one doorbell. Warp-collective; `bytes` warp-uniform.
__device__ __forceinline__ void PostPayloadAndInlineNotify(const ::mori::cco::ccoDevComm& comm,
                                                           int peer, int queueId, void* dst,
                                                           void* src, size_t bytes, void* notifyDst,
                                                           uint64_t notifyValue) {
  const int lane = threadIdx.x & (warpSize - 1);
  const bool big = bytes > ::mori::cco::CCO_SDMA_MAX_COPY_BYTES;
  if (big) {
    ::mori::cco::ccoSdma sdma{comm};
    sdma.putPtr<::mori::cco::ccoCoopWarp>(peer, dst, src, bytes, queueId);
  }
  if (lane != 0) return;
  ::mori::cco::ccoSdma sdma{comm};
  ::mori::cco::ccoSdmaQueueDeviceHandle* qh = sdma.queueHandle(peer, queueId);
  const uint64_t copyBytes = (bytes != 0 && !big) ? ::mori::cco::CCO_SDMA_COPY_BYTES : 0;
  const uint64_t base = ::mori::cco::ccoSdma::reserve(qh, copyBytes + kInlineTriggerBytes);
  uint32_t* ring = ::mori::cco::ccoSdma::ring(qh);
  if (copyBytes != 0) ::mori::cco::ccoSdma::placeCopy(ring, base, dst, src, bytes);
  PlaceInlineTrigger(ring, base + copyBytes, notifyDst, notifyValue);
  ::mori::cco::ccoSdma::publish(qh, base, base + copyBytes + kInlineTriggerBytes);
}

// Direct mode lays each per-PE slice of the landing (dispatchInp) out as all hidden rows, then
// all metadata rows (index | weight | scale | source token): the same bytes as the record
// layout, so one packet carries a peer's metadata and the hidden rows can come from anywhere.
__device__ __forceinline__ size_t DirectHiddenOffset(const EpInterNodeDeviceCfg& config, int pe,
                                                     int slot, size_t hiddenBytes,
                                                     size_t xferBytes) {
  return static_cast<size_t>(SendBufSlotOffset(config, pe, 0)) * xferBytes +
         static_cast<size_t>(slot) * hiddenBytes;
}
__device__ __forceinline__ size_t DirectMetaOffset(const EpInterNodeDeviceCfg& config, int pe,
                                                   int slot, size_t hiddenBytes, size_t xferBytes) {
  return static_cast<size_t>(SendBufSlotOffset(config, pe, 0)) * xferBytes +
         static_cast<size_t>(config.MaxNumTokensToSendPerRank()) * hiddenBytes +
         static_cast<size_t>(slot) * (xferBytes - hiddenBytes);
}
// Direct mode's send staging holds only the metadata records, at their flat slot: each peer's
// block is still contiguous, so one packet carries it.
__device__ __forceinline__ size_t DirectStagingMetaOffset(const EpInterNodeDeviceCfg& config,
                                                          int pe, int slot, size_t metaBytes) {
  return static_cast<size_t>(SendBufSlotOffset(config, pe, slot)) * metaBytes;
}

// Byte offsets of the field blocks in inter_dispatch_inp / inter_staging under the segmented
// layout, each worldSize * MaxNumTokensToSendPerRank rows. Must match asyncll_seg_layout in
// internode_regions.py, which sizes both regions and builds the expert's views.
struct SegLayout {
  size_t hidden;
  size_t index;
  size_t weight;
  size_t scale;
  size_t srcTok;
};

__device__ __forceinline__ size_t SegAlign(size_t bytes) {
  return (bytes + 255) & ~static_cast<size_t>(255);
}

__device__ __forceinline__ SegLayout SegLayoutOf(const EpInterNodeDeviceCfg& config,
                                                 size_t hiddenBytes) {
  const size_t rows = static_cast<size_t>(config.MaxNumTokensToSend());
  SegLayout layout;
  layout.hidden = 0;
  layout.index = SegAlign(rows * hiddenBytes);
  layout.weight = SegAlign(layout.index + rows * config.IndexBytes());
  layout.scale = SegAlign(layout.weight + rows * config.WeightBytes());
  layout.srcTok = SegAlign(layout.scale + rows * config.ScaleBytes());
  return layout;
}

// inter_staging's segmented layout: a direct op stages no hidden rows, so its field blocks start
// at 0 (asyncll_seg_layout(cfg, hidden_nbytes=0)); otherwise it is the landing's.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ SegLayout SegStagingLayoutOf(const EpInterNodeDeviceCfg& config,
                                                        size_t hiddenBytes) {
  if constexpr (Direct<kConfig>)
    return SegLayoutOf(config, 0);
  else
    return SegLayoutOf(config, hiddenBytes);
}

// Direct mode's metadata pass: one group of numEpt lanes per token. Lane j loads the token's
// entry j (its slot) and element j of the index, weight and scale rows once; the group then
// writes those rows into each record the token claimed. stride < 0: compact metadata records
// (DirectStagingMetaOffset); stride >= 0: the segmented field blocks, own rows in the landing,
// peers' in the staging layout, slots at or past the stride dropped (and nulled in the map).
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void DirectWriteMeta(const EpDispatchCombineArgs& args,
                                                const EpInterNodeDeviceCfg& config,
                                                size_t hiddenBytes, size_t xferBytes, int stride) {
  constexpr int numEpt = kConfig.numExpertPerToken;
  constexpr size_t kIndexBytes = numEpt * sizeof(index_t);
  constexpr size_t kWeightBytes = numEpt * sizeof(float);
  constexpr size_t kScaleBytes =
      static_cast<size_t>(kConfig.scaleDim) * static_cast<size_t>(kConfig.scaleTypeSize);
  constexpr int kScaleWords = static_cast<int>((kScaleBytes / 4 + numEpt - 1) / numEpt);
  constexpr bool kWordScales = (kScaleBytes & 3) == 0;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int globalWarpNum = (gridDim.x * blockDim.x) / warpSize;
  const int myPe = args.rank;
  const int groupsPerWarp = warpSize / numEpt;
  const int group = laneId / numEpt;
  const int j = laneId % numEpt;
  const int baseLane = group * numEpt;
  const bool haveScales = (args.scalesBuf != nullptr) && (kScaleBytes > 0);
  const index_t null = NullSendBufSlotOffset(config);
  uint8_t* staging = args.reg(args.offStaging)->template GetAs<uint8_t*>();
  uint8_t* landing = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();
  const SegLayout layout = SegLayoutOf(config, hiddenBytes);
  const SegLayout stagingLayout = SegStagingLayoutOf<kConfig>(config, hiddenBytes);
  for (int first = globalWarpId * groupsPerWarp; first < args.curRankNumToken;
       first += globalWarpNum * groupsPerWarp) {
    const int tok = first + group;
    const bool active = (group < groupsPerWarp) && (tok < args.curRankNumToken);
    index_t dst = null, index = 0;
    float weight = 0.0f;
    uint32_t scaleWords[kScaleWords > 0 ? kScaleWords : 1];
    if (active) {
      dst = args.dispDestTokIdMap[tok * numEpt + j];
      index = args.tokenIndices[tok * numEpt + j];
      if (args.weightsBuf) weight = args.weightsBuf[tok * numEpt + j];
      if constexpr (kWordScales && kScaleWords > 0) {
        if (haveScales) {
          const uint32_t* from = reinterpret_cast<const uint32_t*>(
              args.scalesBuf + static_cast<size_t>(tok) * kScaleBytes);
#pragma unroll
          for (int w = 0; w < kScaleWords; ++w) {
            const size_t word = static_cast<size_t>(j) + static_cast<size_t>(w) * numEpt;
            scaleWords[w] = (word < kScaleBytes / 4) ? from[word] : 0u;
          }
        }
      }
      // Segmented: a slot past the stride is dropped; the receiver never sees it.
      if (stride >= 0 && dst < null && SlotIdFromSendBufSlotOffset(config, dst) >= stride) {
        args.dispDestTokIdMap[tok * numEpt + j] = null;
        dst = null;
      }
    }
    for (int k = 0; k < numEpt; ++k) {
      const index_t d = __shfl(dst, (baseLane + k) & (warpSize - 1));
      if (!active || d >= null) continue;
      const int pe = PeFromSendBufSlotOffset(config, d);
      const int slot = SlotIdFromSendBufSlotOffset(config, d);
      uint8_t *idxRow, *wRow, *scaleRow;
      index_t* srcTokAt;
      if (stride < 0) {
        uint8_t* meta =
            staging + DirectStagingMetaOffset(config, pe, slot, xferBytes - hiddenBytes);
        idxRow = meta;
        wRow = meta + kIndexBytes;
        scaleRow = meta + kIndexBytes + kWeightBytes;
        srcTokAt = reinterpret_cast<index_t*>(meta + kIndexBytes + kWeightBytes + kScaleBytes);
      } else {
        uint8_t* base = (pe == myPe) ? landing : staging;
        const SegLayout& lay = (pe == myPe) ? layout : stagingLayout;
        const size_t row =
            (pe == myPe) ? static_cast<size_t>(myPe) * stride + slot : static_cast<size_t>(d);
        idxRow = base + lay.index + row * kIndexBytes;
        wRow = base + lay.weight + row * kWeightBytes;
        scaleRow = base + lay.scale + row * kScaleBytes;
        srcTokAt = reinterpret_cast<index_t*>(base + lay.srcTok) + row;
      }
      reinterpret_cast<index_t*>(idxRow)[j] = index;
      if (args.weightsBuf) reinterpret_cast<float*>(wRow)[j] = weight;
      if (haveScales) {
        if constexpr (kWordScales && kScaleWords > 0) {
#pragma unroll
          for (int w = 0; w < kScaleWords; ++w) {
            const size_t word = static_cast<size_t>(j) + static_cast<size_t>(w) * numEpt;
            if (word < kScaleBytes / 4) reinterpret_cast<uint32_t*>(scaleRow)[word] = scaleWords[w];
          }
        } else {
          const uint8_t* from = args.scalesBuf + static_cast<size_t>(tok) * kScaleBytes;
          for (size_t b = j; b < kScaleBytes; b += numEpt) scaleRow[b] = from[b];
        }
      }
      if (j == 0) *srcTokAt = static_cast<index_t>(SrcTokenIndex(config, myPe, tok));
    }
  }
}

// DirectWriteMeta's body for one group whose slots are already in registers (SlotAssign's lane
// layout is the same: lane j of the group holds entry j). Segmented layout only. No scale rows:
// they travel with the hidden rows (DirectPostRows), from the input.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void DirectWriteMetaGroup(const EpDispatchCombineArgs& args,
                                                     const EpInterNodeDeviceCfg& config,
                                                     size_t hiddenBytes, int stride, bool active,
                                                     int tok, int j, int baseLane, index_t dst,
                                                     index_t index, float weight, uint8_t* staging,
                                                     uint8_t* landing) {
  constexpr int numEpt = kConfig.numExpertPerToken;
  constexpr size_t kIndexBytes = numEpt * sizeof(index_t);
  constexpr size_t kWeightBytes = numEpt * sizeof(float);
  const int myPe = args.rank;
  const index_t null = NullSendBufSlotOffset(config);
  const SegLayout layout = SegLayoutOf(config, hiddenBytes);
  const SegLayout stagingLayout = SegStagingLayoutOf<kConfig>(config, hiddenBytes);
  for (int k = 0; k < numEpt; ++k) {
    const index_t d = __shfl(dst, (baseLane + k) & (warpSize - 1));
    if (!active || d >= null) continue;
    const int pe = PeFromSendBufSlotOffset(config, d);
    const int slot = SlotIdFromSendBufSlotOffset(config, d);
    uint8_t* base = (pe == myPe) ? landing : staging;
    const SegLayout& lay = (pe == myPe) ? layout : stagingLayout;
    const size_t row =
        (pe == myPe) ? static_cast<size_t>(myPe) * stride + slot : static_cast<size_t>(d);
    reinterpret_cast<index_t*>(base + lay.index + row * kIndexBytes)[j] = index;
    if (args.weightsBuf)
      reinterpret_cast<float*>(base + lay.weight + row * kWeightBytes)[j] = weight;
    if (j == 0)
      reinterpret_cast<index_t*>(base + lay.srcTok)[row] =
          static_cast<index_t>(SrcTokenIndex(config, myPe, tok));
  }
}

// The same records written one group of numEpt lanes per ENTRY: 8x the parallel units of the
// per-token writer, each reloading its token's row. Wins while the entries fit the grid in a
// round or two (few tokens); the per-token writer wins past that.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void DirectWriteMetaPerEntry(const EpDispatchCombineArgs& args,
                                                        const EpInterNodeDeviceCfg& config,
                                                        size_t hiddenBytes, size_t xferBytes,
                                                        int stride) {
  constexpr int numEpt = kConfig.numExpertPerToken;
  constexpr size_t kIndexBytes = numEpt * sizeof(index_t);
  constexpr size_t kWeightBytes = numEpt * sizeof(float);
  constexpr size_t kScaleBytes =
      static_cast<size_t>(kConfig.scaleDim) * static_cast<size_t>(kConfig.scaleTypeSize);
  constexpr int kScaleWords = static_cast<int>((kScaleBytes / 4 + numEpt - 1) / numEpt);
  constexpr bool kWordScales = (kScaleBytes & 3) == 0;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int globalWarpNum = (gridDim.x * blockDim.x) / warpSize;
  const int myPe = args.rank;
  const int groupsPerWarp = warpSize / numEpt;
  const int group = laneId / numEpt;
  const int j = laneId % numEpt;
  const bool haveScales = (args.scalesBuf != nullptr) && (kScaleBytes > 0);
  const index_t null = NullSendBufSlotOffset(config);
  const int totalEntries = args.curRankNumToken * numEpt;
  uint8_t* staging = args.reg(args.offStaging)->template GetAs<uint8_t*>();
  uint8_t* landing = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();
  const SegLayout layout = SegLayoutOf(config, hiddenBytes);
  const SegLayout stagingLayout = SegStagingLayoutOf<kConfig>(config, hiddenBytes);
  for (int first = globalWarpId * groupsPerWarp; first < totalEntries;
       first += globalWarpNum * groupsPerWarp) {
    const int entryId = first + group;
    if (group >= groupsPerWarp || entryId >= totalEntries) continue;
    const index_t d = args.dispDestTokIdMap[entryId];
    const int tok = entryId / numEpt;
    const index_t index = args.tokenIndices[tok * numEpt + j];
    const float weight = args.weightsBuf ? args.weightsBuf[tok * numEpt + j] : 0.0f;
    uint32_t scaleWords[kScaleWords > 0 ? kScaleWords : 1];
    if constexpr (kWordScales && kScaleWords > 0) {
      if (haveScales) {
        const uint32_t* from = reinterpret_cast<const uint32_t*>(
            args.scalesBuf + static_cast<size_t>(tok) * kScaleBytes);
#pragma unroll
        for (int w = 0; w < kScaleWords; ++w) {
          const size_t word = static_cast<size_t>(j) + static_cast<size_t>(w) * numEpt;
          scaleWords[w] = (word < kScaleBytes / 4) ? from[word] : 0u;
        }
      }
    }
    if (d >= null) continue;
    const int pe = PeFromSendBufSlotOffset(config, d);
    const int slot = SlotIdFromSendBufSlotOffset(config, d);
    if (stride >= 0 && slot >= stride) {
      if (j == 0) args.dispDestTokIdMap[entryId] = null;
      continue;
    }
    uint8_t *idxRow, *wRow, *scaleRow;
    index_t* srcTokAt;
    if (stride < 0) {
      uint8_t* meta = staging + DirectStagingMetaOffset(config, pe, slot, xferBytes - hiddenBytes);
      idxRow = meta;
      wRow = meta + kIndexBytes;
      scaleRow = meta + kIndexBytes + kWeightBytes;
      srcTokAt = reinterpret_cast<index_t*>(meta + kIndexBytes + kWeightBytes + kScaleBytes);
    } else {
      uint8_t* base = (pe == myPe) ? landing : staging;
      const SegLayout& lay = (pe == myPe) ? layout : stagingLayout;
      const size_t row =
          (pe == myPe) ? static_cast<size_t>(myPe) * stride + slot : static_cast<size_t>(d);
      idxRow = base + lay.index + row * kIndexBytes;
      wRow = base + lay.weight + row * kWeightBytes;
      scaleRow = base + lay.scale + row * kScaleBytes;
      srcTokAt = reinterpret_cast<index_t*>(base + lay.srcTok) + row;
    }
    reinterpret_cast<index_t*>(idxRow)[j] = index;
    if (args.weightsBuf) reinterpret_cast<float*>(wRow)[j] = weight;
    if (haveScales) {
      if constexpr (kWordScales && kScaleWords > 0) {
#pragma unroll
        for (int w = 0; w < kScaleWords; ++w) {
          const size_t word = static_cast<size_t>(j) + static_cast<size_t>(w) * numEpt;
          if (word < kScaleBytes / 4) reinterpret_cast<uint32_t*>(scaleRow)[word] = scaleWords[w];
        }
      } else {
        const uint8_t* from = args.scalesBuf + static_cast<size_t>(tok) * kScaleBytes;
        for (size_t b = j; b < kScaleBytes; b += numEpt) scaleRow[b] = from[b];
      }
    }
    if (j == 0) *srcTokAt = static_cast<index_t>(SrcTokenIndex(config, myPe, tok));
  }
}

// Per-entry while every entry gets a group within four rounds, per-token past that.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void DirectWriteMetaAuto(const EpDispatchCombineArgs& args,
                                                    const EpInterNodeDeviceCfg& config,
                                                    size_t hiddenBytes, size_t xferBytes,
                                                    int stride) {
  constexpr int numEpt = kConfig.numExpertPerToken;
  const long groups = static_cast<long>(gridDim.x) * (blockDim.x / warpSize) * (warpSize / numEpt);
  if (static_cast<long>(args.curRankNumToken) * numEpt <= 4 * groups)
    DirectWriteMetaPerEntry<kConfig>(args, config, hiddenBytes, xferBytes, stride);
  else
    DirectWriteMeta<kConfig>(args, config, hiddenBytes, xferBytes, stride);
}

// The reduce on source pointers held in registers (static indexing, null = no source): pointers
// read back from LDS cost the loop its overlap (1.6x in a microbenchmark), so the VEC 17 receive
// computes them per token instead.
// The wave's value of a pointer every lane holds the same of, in scalar registers: a null test
// on it is then a scalar branch. As a per-lane value it is a divergent branch, and the compiler
// waits for every outstanding load before each source's load (eight round trips a step).
__device__ __forceinline__ uint64_t WaveUniformU64(uint64_t v) {
  const uint64_t lo = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
  const uint64_t hi =
      static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(v >> 32)));
  return (hi << 32) | lo;
}
template <typename P>
__device__ __forceinline__ P* WaveUniform(P* p) {
  return reinterpret_cast<P*>(WaveUniformU64(reinterpret_cast<uint64_t>(p)));
}

// E floats to E T's in a 16-byte vector. bf16 on gfx950 uses the packed hardware convert (same
// round-to-nearest-even as hip_bfloat16's, without its branchy software path).
template <typename T, int E>
__device__ __forceinline__ uint4 PackAccum(const float (&acc)[E]) {
  uint4 r;
#if defined(__gfx950__)
  if constexpr (std::is_same_v<T, hip_bfloat16> || std::is_same_v<T, __hip_bfloat16>) {
    typedef float F2 __attribute__((ext_vector_type(2)));
    typedef __bf16 B2 __attribute__((ext_vector_type(2)));
    uint32_t* w = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
    for (int p = 0; p < E / 2; ++p)
      w[p] =
          __builtin_bit_cast(uint32_t, __builtin_convertvector(F2{acc[2 * p], acc[2 * p + 1]}, B2));
    return r;
  }
#endif
  T* y = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int e = 0; e < E; ++e) y[e] = T(acc[e]);
  return r;
}

template <typename T, int kTopk>
__device__ __forceinline__ void CombineReduceRegs(T* out_, const T* const (&src_)[kTopk],
                                                  size_t nelems) {
  // Buffer loads and stores off wave-uniform row bases: generic (flat) accesses make the
  // compiler wait for every outstanding access before each load, since a flat store may hit LDS.
  constexpr int E = 16 / sizeof(T);
#if defined(__gfx942__) || defined(__gfx950__)
  constexpr int kRsrcWord3 = 0x00020000;
#else
  constexpr int kRsrcWord3 = 0x31014000;
#endif
  const int lane = threadIdx.x & (warpSize - 1);
  const int bytes = static_cast<int>(nelems * sizeof(T));
  __amdgpu_buffer_rsrc_t rs[kTopk];
  bool has[kTopk];
#pragma unroll
  for (int k = 0; k < kTopk; ++k) {
    const T* p = WaveUniform(src_[k]);
    has[k] = p != nullptr;
    rs[k] = __builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(p), 0, bytes, kRsrcWord3);
  }
  T* out = WaveUniform(out_);
  const __amdgpu_buffer_rsrc_t ro = __builtin_amdgcn_make_buffer_rsrc(out, 0, bytes, kRsrcWord3);
  const size_t span = static_cast<size_t>(warpSize) * E;
  size_t off = 0;
  for (; off + span <= nelems; off += span) {
    const int voff = static_cast<int>((off + lane * E) * sizeof(T));
    uint4 v[kTopk];
#pragma unroll
    for (int k = 0; k < kTopk; ++k)
      if (has[k]) {
        const auto x = __builtin_amdgcn_raw_buffer_load_b128(rs[k], voff, 0, 0);
        v[k] = *reinterpret_cast<const uint4*>(&x);
      }
    float acc[E] = {};
#pragma unroll
    for (int k = 0; k < kTopk; ++k)
      if (has[k]) {
        const T* x = reinterpret_cast<const T*>(&v[k]);
#pragma unroll
        for (int e = 0; e < E; ++e) acc[e] += float(x[e]);
      }
    uint4 r = PackAccum<T, E>(acc);
    __builtin_amdgcn_raw_buffer_store_b128(
        *reinterpret_cast<decltype(__builtin_amdgcn_raw_buffer_load_b128(ro, 0, 0, 0))*>(&r), ro,
        voff, 0, 0);
  }
  for (size_t e = off + lane; e < nelems; e += warpSize) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kTopk; ++k)
      if (has[k]) acc += float(src_[k][e]);
    out[e] = T(acc);
  }
}

template <typename T, int kTopk, int kVec>
__device__ __forceinline__ void CombineReduce(T* out, T* const* srcs, size_t nelems) {
  if constexpr (kVec == 16) {
    core::WarpAccumLF<T, 16, MORI_ASYNCLL_COMBINE_UNROLL>(out, srcs, nullptr, kTopk, nelems);
  } else {
    core::WarpAccum<T, 4>(out, srcs, nullptr, kTopk, nelems);
  }
}
// MultiWarpIter granularity that keeps every warp's slice on a vector boundary.
template <typename T, int kVec>
__device__ __forceinline__ constexpr size_t CombineGranularity() {
  // VEC 17: whole 16 B x 64-lane steps per warp slice, so no slice ends in the scalar tail.
  return kVec == 17 ? 64 * (16 / sizeof(T)) : kVec == 16 ? 16 / sizeof(T) : 1;
}

// Exclusive prefix of the per-peer counts staged in SendCountSlot (combine's early-notify
// bookkeeping): where each source rank's rows start in the expert output. Lane i gets PE i's.
__device__ __forceinline__ uint64_t CombineRowStart(const uint8_t* sendCounts, int npes,
                                                    int laneId) {
  const uint64_t n =
      (laneId < npes) ? *reinterpret_cast<const uint64_t*>(sendCounts + SendCountSlot(laneId)) : 0;
  uint64_t inclusive = n;
  for (int offset = 1; offset < npes; offset <<= 1) {
    const uint64_t up = __shfl_up(inclusive, offset);
    if (laneId >= offset) inclusive += up;
  }
  return inclusive - n;
}

// Copy a short field one 4 B word per lane, falling back to one byte per lane
// unless the length and both ends are 4 B aligned. At the shapes in use they are
// (hidden and scale rows are multiples of 4), but the record stride is the sum
// of every field, so an odd scale row would misalign every record after it.
__device__ __forceinline__ void LaneCopyWords(uint8_t* dst, const uint8_t* src, size_t bytes) {
  const int lane = threadIdx.x & (warpSize - 1);
  if (((bytes | reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 3) == 0) {
    for (size_t b = static_cast<size_t>(lane) * 4; b < bytes;
         b += static_cast<size_t>(warpSize) * 4)
      *reinterpret_cast<uint32_t*>(dst + b) = *reinterpret_cast<const uint32_t*>(src + b);
  } else {
    for (size_t b = lane; b < bytes; b += warpSize) dst[b] = src[b];
  }
}

// ---- Quantized combine (EpInterNodeKernelCfg.combineQGroup) ----
// The combine wire row is the token's hiddenDim fp8 elements, then one scale per
// group (an E8M0 byte per 32 elements, or an f32 per 128), padded to 16 B. Rows
// sit at that stride in inter_combine_staging and inter_combine_inp, which are
// sized for the bf16 row, so they fit. The codec is the scatter combine's
// (ep_scatter_quant.hpp); a lane takes 16 elements, one 16 B fp8 access, and a
// group's lanes share its amax.
template <EpInterNodeKernelCfg kConfig>
inline constexpr bool CombineQuant = kConfig.combineQGroup != 0;

template <EpInterNodeKernelCfg kConfig, typename T>
constexpr size_t CombineRowBytes() {
  if constexpr (CombineQuant<kConfig>) {
    constexpr size_t scale =
        kConfig.combineQGroup == 32 ? kConfig.hiddenDim / 32 : kConfig.hiddenDim / 128 * 4;
    return (static_cast<size_t>(kConfig.hiddenDim) + scale + 15) & ~static_cast<size_t>(15);
  } else {
    return static_cast<size_t>(kConfig.hiddenDim) * sizeof(T);
  }
}

constexpr int kCombineQVec = 16;

// Arena and input pointers are built from integers, so the compiler cannot see
// that they are global: it emits flat accesses, which have no immediate offset
// and keep a 64-bit address per access live. Rebuilding the pointer in address
// space 1 gives global_load / global_store.
template <typename V>
__device__ __forceinline__ V LoadGlobal(const void* p) {
  return *(const __attribute__((address_space(1))) V*)(reinterpret_cast<uintptr_t>(p));
}
template <typename V>
__device__ __forceinline__ void StoreGlobal(void* p, V v) {
  *(__attribute__((address_space(1))) V*)(reinterpret_cast<uintptr_t>(p)) = v;
}

// This warp's index in the grid, as a scalar: what a warp derives from it (its
// token, the token's slots) then stays in SGPRs and loads as one.
__device__ __forceinline__ int UniformWarpId() {
  return __builtin_amdgcn_readfirstlane(blockIdx.x * (blockDim.x / warpSize) +
                                        threadIdx.x / warpSize);
}

// CombineSendCopy, quantizing: one warp per (recv row, 16 * warpSize elements).
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ __forceinline__ void CombineQuantSendCopy(const EpDispatchCombineArgs& args,
                                                     uint8_t* staging, int rows) {
  static_assert(sizeof(T) == 2, "the quantized combine reads bf16 rows");
  constexpr int kGroup = kConfig.combineQGroup;
  constexpr int kHidden = kConfig.hiddenDim;
  constexpr int kLanesPerGroup = kGroup / kCombineQVec;
  constexpr size_t kRowB = CombineRowBytes<kConfig, T>();
  const int lane = threadIdx.x & (warpSize - 1);
  const int warps = gridDim.x * (blockDim.x / warpSize);
  const int chunk = warpSize * kCombineQVec;
  const int steps = (kHidden + chunk - 1) / chunk;
  const T* inp = static_cast<const T*>(args.inpTokenBuf);

  for (int u = UniformWarpId(); u < rows * steps; u += warps) {
    const int tok = u / steps;
    const int e = (u % steps) * chunk + lane * kCombineQVec;
    const bool active = e < kHidden;
    float f[2][8] = {};
    if (active) {
      const T* src = inp + static_cast<size_t>(tok) * kHidden + e;
      EpBf16x8ToF32(LoadGlobal<uint4>(src), f[0]);
      EpBf16x8ToF32(LoadGlobal<uint4>(src + 8), f[1]);
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i / 8][i % 8]));
#pragma unroll
    for (int off = 1; off < kLanesPerGroup; off <<= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
    float scale, inv;
    unsigned e8m0;
    EpScatterScale<kGroup, kEpScatterFp8Max>(amax, scale, inv, e8m0);
    if (!active) continue;
    uint8_t* row = staging + static_cast<size_t>(args.dispReceiverIdxMap[tok]) * kRowB;
    uint2 q[2];
    EpScatterPack<false>(reinterpret_cast<char*>(&q[0]), f[0], inv);
    EpScatterPack<false>(reinterpret_cast<char*>(&q[1]), f[1], inv);
    StoreGlobal(row + e, make_uint4(q[0].x, q[0].y, q[1].x, q[1].y));
    if (lane % kLanesPerGroup == 0) {
      if constexpr (kGroup == 32)
        StoreGlobal(row + kHidden + e / kGroup, static_cast<uint8_t>(e8m0));
      else
        StoreGlobal(row + kHidden + e / kGroup * 4, scale);
    }
  }
}

// CombineRecvCopy, dequantizing: one warp per (local token, 16 * warpSize
// elements) sums the token's top-k rows in f32. The slots are resolved first and
// every row's load is issued before the first is summed: resolving and loading
// one row at a time serialises top-k round trips. A dropped slot loads a harmless
// row and is skipped at the sum, not scaled by 0, since its bytes may be NaN.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ __forceinline__ void CombineQuantRecvCopy(const EpDispatchCombineArgs& args,
                                                     const uint8_t* peerBase,
                                                     const uint8_t* selfBase, T* out) {
  constexpr EpInterNodeDeviceCfg config = EpInterNodeDeviceCfgOf(kConfig);
  constexpr int kGroup = kConfig.combineQGroup;
  constexpr int kHidden = kConfig.hiddenDim;
  constexpr int kTopk = kConfig.numExpertPerToken;
  constexpr size_t kRowB = CombineRowBytes<kConfig, T>();
  const int lane = threadIdx.x & (warpSize - 1);
  const int warps = gridDim.x * (blockDim.x / warpSize);
  const int chunk = warpSize * kCombineQVec;
  const int steps = (kHidden + chunk - 1) / chunk;

  for (int u = UniformWarpId(); u < args.curRankNumToken * steps; u += warps) {
    const int tok = u / steps;
    const int e = (u % steps) * chunk + lane * kCombineQVec;
    if (e >= kHidden) continue;
    // All of the token's slots in flight at once, then warp-uniform: the row
    // addresses are scalar arithmetic and the choice between them a select.
    index_t slots[kTopk];
#pragma unroll
    for (int j = 0; j < kTopk; ++j) slots[j] = args.dispDestTokIdMap[tok * kTopk + j];
    const uint8_t* rows[kTopk];
    bool live[kTopk];
#pragma unroll
    for (int j = 0; j < kTopk; ++j) {
      const index_t destTokId = __builtin_amdgcn_readfirstlane(slots[j]);
      const int pe = PeFromSendBufSlotOffset(config, destTokId);
      live[j] = pe < kConfig.worldSize;
      // Own rows were never transferred: they are still in combine's staging.
      rows[j] =
          live[j] ? (pe != args.rank ? peerBase : selfBase) + static_cast<size_t>(destTokId) * kRowB
                  : selfBase;
    }
    uint4 q[kTopk];
    float scale[kTopk];
#pragma unroll
    for (int j = 0; j < kTopk; ++j) {
      q[j] = LoadGlobal<uint4>(rows[j] + e);
      if constexpr (kGroup == 32)
        scale[j] = __uint_as_float(
            static_cast<unsigned>(LoadGlobal<uint8_t>(rows[j] + kHidden + e / kGroup)) << 23);
      else
        scale[j] = LoadGlobal<float>(rows[j] + kHidden + e / kGroup * 4);
    }
    float acc[2][8] = {};
#pragma unroll
    for (int j = 0; j < kTopk; ++j) {
      if (!live[j]) continue;
      EpScatterAcc<false>(make_uint2(q[j].x, q[j].y), scale[j], acc[0]);
      EpScatterAcc<false>(make_uint2(q[j].z, q[j].w), scale[j], acc[1]);
    }
    T* dst = out + static_cast<size_t>(tok) * kHidden + e;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      unsigned pk[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned lo = __builtin_bit_cast(unsigned short, T(acc[h][2 * i]));
        const unsigned hi = __builtin_bit_cast(unsigned short, T(acc[h][2 * i + 1]));
        pk[i] = lo | (hi << 16);
      }
      StoreGlobal(dst + 8 * h, make_uint4(pk[0], pk[1], pk[2], pk[3]));
    }
  }
}

// ---- Segmented receive (EpDispatchCombineConfig.async_ll_recv_layout) ----
// args.segStride is 0 on an op built without the segmented layout. On one built with it, bit 63
// is set on every call and the low 32 bits are this round's stride S, 0 for a compact round.
constexpr uint64_t kSegCapableBit = uint64_t{1} << 63;
__device__ __forceinline__ bool SegCapable(const EpDispatchCombineArgs& args) {
  return (args.segStride & kSegCapableBit) != 0;
}
// Clamped to the per-source receive capacity. The host refuses a larger stride
// (segment_stride), so this only matters when the arguments come from elsewhere:
// the transfers then stay inside the peers' field blocks instead of writing past
// them, and the round is wrong rather than a corrupted arena.
__device__ __forceinline__ int SegStrideOf(const EpDispatchCombineArgs& args,
                                           const EpInterNodeDeviceCfg& config) {
  const uint64_t stride = args.segStride & 0xffffffffull;
  const uint64_t cap = static_cast<uint64_t>(config.MaxNumTokensToRecvPerRank());
  return static_cast<int>(stride < cap ? stride : cap);
}

// Every dispatch on a segmented-capable op, compact or segmented, sends each peer a mode word:
// the round's epoch in the high half and its stride in the low half. The epoch is
// crossDeviceBarrierFlag + 1 as read before this round's receive bumps it, the same on every
// rank of a round, so a word that was not written this round (a peer that skipped it, or a
// compact path that never sends one) cannot match either. The receive compares each peer's
// word with its own and counts a mismatch: ranks that disagree on segmented vs compact, or on S,
// read each other's rows at the wrong offsets.
__device__ __forceinline__ uint64_t SegModeWord(uint64_t flag, int stride) {
  return ((flag + 1) << 32) | static_cast<uint32_t>(stride);
}

// send_count words past the count/epoch pairs (segmented-capable ops only):
//   [2W, 3W)    the mode word each source sent this rank, written by that source's engine
//   3W          the local word this rank sends its own from
//   3W+1, 3W+2  error counters (SegErrOverflow, SegErrMismatch), local, never sent
//   [3W+3, 4W+3) this round's per-source row counts n_s, local (segment_counts())
// Must match asyncll_seg_words() in internode_regions.py.
__device__ __forceinline__ size_t SegStrideSlot(const EpInterNodeDeviceCfg& config, int srcPe) {
  return (static_cast<size_t>(config.worldSize) * 2 + srcPe) * sizeof(uint64_t);
}
__device__ __forceinline__ size_t SegStrideSourceSlot(const EpInterNodeDeviceCfg& config) {
  return static_cast<size_t>(config.worldSize) * 3 * sizeof(uint64_t);
}
constexpr int SegErrOverflow = 0;  // this rank had more rows for a peer than the stride
constexpr int SegErrMismatch = 1;  // a peer's round did not use this rank's mode and stride
__device__ __forceinline__ size_t SegErrSlot(const EpInterNodeDeviceCfg& config, int which) {
  return (static_cast<size_t>(config.worldSize) * 3 + 1 + which) * sizeof(uint64_t);
}
__device__ __forceinline__ size_t SegCountSlot(const EpInterNodeDeviceCfg& config, int srcPe) {
  return (static_cast<size_t>(config.worldSize) * 3 + 3 + srcPe) * sizeof(uint64_t);
}
// Counters, not sticky bits: the host keeps what it last saw, so it never has to clear a word
// the device may be adding to.
__device__ __forceinline__ void SegCountError(const EpDispatchCombineArgs& args,
                                              const EpInterNodeDeviceCfg& config, int which) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(
          args.reg(args.offSendCount)->template GetAs<uint8_t*>() + SegErrSlot(config, which)),
      1ull);
}
// Compare every peer's mode word with this rank's. Lanes < npes of one warp, before the receive
// bumps crossDeviceBarrierFlag.
__device__ __forceinline__ void SegCheckModeWords(const EpDispatchCombineArgs& args,
                                                  const EpInterNodeDeviceCfg& config, int lane,
                                                  int myPe, int stride) {
  if (lane == myPe) return;
  const uint64_t want = SegModeWord(args.crossDeviceBarrierFlag[0], stride);
  const uint64_t* word = reinterpret_cast<const uint64_t*>(
      args.reg(args.offSendCount)->template GetAs<uint8_t*>() + SegStrideSlot(config, lane));
  if (core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(word)) != want)
    SegCountError(args, config, SegErrMismatch);
}

// DirectPostRows without MORI_ASYNCLL_BATCH_POST: each warp reserves its own 64-slot batch's runs.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void DirectPostRowsPerWarp(
    const EpDispatchCombineArgs& args, const ::mori::cco::ccoDevComm& comm,
    const EpInterNodeDeviceCfg& config, int peer, int q, uint8_t* dst, const uint8_t* srcTokAt,
    size_t srcTokStride, int tokenNum, size_t hiddenBytes, uint8_t* scaleDst, size_t scaleBytes) {
  ::mori::cco::ccoSdma sdma{comm};
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = threadIdx.x / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const uint8_t* input = static_cast<const uint8_t*>(args.inpTokenBuf);
  const index_t firstSrcTok = SrcTokenIndex(config, args.rank, 0);
  for (int base = warpId * warpSize; base < tokenNum; base += warpNum * warpSize) {
    const int slot = base + laneId;
    const int active = min(tokenNum - base, static_cast<int>(warpSize));
    const index_t srcTok = (slot < tokenNum)
                               ? reinterpret_cast<const index_t*>(
                                     srcTokAt + static_cast<size_t>(slot) * srcTokStride)[0] -
                                     firstSrcTok
                               : -2;
    const index_t prev = __shfl_up(srcTok, 1);
    const bool head = (slot < tokenNum) && (laneId == 0 || prev + 1 != srcTok);
    const unsigned long long heads = __ballot(head);
    if (head) {
      const unsigned long long later =
          heads & ~((laneId + 1 >= 64) ? ~0ull : ((1ull << (laneId + 1)) - 1));
      const int end = later ? (__ffsll(static_cast<long long>(later)) - 1) : active;
      sdma.putPtr<::mori::cco::ccoCoopThread, false, false, ::mori::cco::ccoSdmaOptFlagsDefault,
                  ::mori::cco::ccoSdmaThreadSameQueue>(
          peer, dst + static_cast<size_t>(slot) * hiddenBytes,
          const_cast<uint8_t*>(input + static_cast<size_t>(srcTok) * hiddenBytes),
          static_cast<size_t>(end - laneId) * hiddenBytes, q);
      if (scaleBytes)
        sdma.putPtr<::mori::cco::ccoCoopThread, false, false, ::mori::cco::ccoSdmaOptFlagsDefault,
                    ::mori::cco::ccoSdmaThreadSameQueue>(
            peer, scaleDst + static_cast<size_t>(slot) * scaleBytes,
            const_cast<uint8_t*>(args.scalesBuf + static_cast<size_t>(srcTok) * scaleBytes),
            static_cast<size_t>(end - laneId) * scaleBytes, q);
    }
  }
}

// BATCH_POST DirectPostRows: batches per warp kept in registers between passes (reload group).
#ifndef MORI_ASYNCLL_POST_REG_BATCHES
#define MORI_ASYNCLL_POST_REG_BATCHES 4
#endif
// BATCH_POST DirectPostRows: max packets' worth of runs per ring reservation (halved when scale
// rows ride along), 0 = half the ring (else a test hook).
#ifndef MORI_ASYNCLL_POST_CHUNK_RUNS
#define MORI_ASYNCLL_POST_CHUNK_RUNS 0
#endif

// Rows straight from the input, by every warp of the block: slot s goes to dst + s * hiddenBytes
// from the source token whose SrcTokenIndex sits at srcTokAt + s * srcTokStride. Warp w posts
// batches w, w + W, ...; a run of adjacent source tokens is one packet, and with scaleBytes a
// second one carries the run's scale rows from args.scalesBuf to scaleDst. With `parts` > 1 the
// peer's batches are split over that many blocks (this one is `part`), each reserving its own;
// returns true in the one block that reserved last (counted at `arrival`), so what that block
// reserves next lands behind every block's rows. Always true with one part. The caller follows
// with a block barrier before anything that must land behind the rows.
template <EpInterNodeKernelCfg kConfig, bool kSplit = false>
__device__ __forceinline__ bool DirectPostRows(
    const EpDispatchCombineArgs& args, const ::mori::cco::ccoDevComm& comm,
    const EpInterNodeDeviceCfg& config, int peer, int q, uint8_t* dst, const uint8_t* srcTokAt,
    size_t srcTokStride, int tokenNum, size_t hiddenBytes, uint8_t* scaleDst = nullptr,
    size_t scaleBytes = 0, int part = 0, int parts = 1, int* arrival = nullptr) {
#if MORI_ASYNCLL_BATCH_POST
  // All waves place the peer's runs at their run index; one reservation and doorbell per half ring.
  constexpr int kMaxBatches = (kConfig.maxNumInpTokenPerRank + 31) / 32;
  constexpr int kReg = MORI_ASYNCLL_POST_REG_BATCHES;
  static_assert(kReg >= 1, "MORI_ASYNCLL_POST_REG_BATCHES must be at least 1");
  constexpr uint64_t kPkt = ::mori::cco::CCO_SDMA_COPY_BYTES;
  // ReserveSlot waits forever for a span the ring cannot hold.
  constexpr int kHalfRingRuns = static_cast<int>(::mori::cco::CCO_SDMA_QUEUE_SIZE / kPkt / 2);
  constexpr int kChunkRuns =
      (MORI_ASYNCLL_POST_CHUNK_RUNS > 0 && MORI_ASYNCLL_POST_CHUNK_RUNS < kHalfRingRuns)
          ? MORI_ASYNCLL_POST_CHUNK_RUNS
          : kHalfRingRuns;
  __shared__ int runStart[kMaxBatches + 1];
  __shared__ uint64_t runBase;
  ::mori::cco::ccoSdma sdma{comm};
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = threadIdx.x / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int groupStride = kReg * warpNum;
  const uint8_t* input = static_cast<const uint8_t*>(args.inpTokenBuf);
  const index_t firstSrcTok = SrcTokenIndex(config, args.rank, 0);
  // `parts` blocks share the peer, block `part` taking an even share of its 64-slot batches (a
  // run never spans batches, so the split adds no packets). Each reserves and submits its own;
  // the last to finish reserving, counted at `arrival`, returns true: anything the caller then
  // reserves lands behind every block's rows.
  const int nbAll = (tokenNum + warpSize - 1) / warpSize;
  // kSplit is the caller's parts > 1: one part (the common case) compiles none of this.
  const int bLo = kSplit ? nbAll * part / parts : 0;
  const int nb = kSplit ? nbAll * (part + 1) / parts - bLo : nbAll;
  const unsigned long long below = (1ull << laneId) - 1ull;
  // Packets per run, and the runs a half-ring reservation holds.
  const int perRun = scaleBytes ? 2 : 1;
  const int chunkRuns = max(kChunkRuns / perRun, 1);
  // Thread 0, after this block's last row reservation.
  auto arrive = [&]() {
    if constexpr (!kSplit) return true;
    const bool last = atomicAdd(arrival, 1) == parts - 1;
    if (last) *arrival = 0;  // every block has counted; the next round starts from zero
    return last;
  };
  // This lane's source token in batch b; the row is clamped into the slice so loads never branch.
  using GIndex = const __attribute__((address_space(1))) index_t*;
  auto srcTokOf = [&](int b) -> index_t {
    const int row = max(min((bLo + b) * static_cast<int>(warpSize) + laneId, tokenNum - 1), 0);
    return *(GIndex)(srcTokAt + static_cast<size_t>(row) * srcTokStride) - firstSrcTok;
  };
  // Lanes of batch b that start a run of adjacent source tokens.
  auto headsOf = [&](int b, index_t srcTok) -> unsigned long long {
    const int slot = (bLo + b) * warpSize + laneId;
    const index_t prev = __shfl_up(srcTok, 1);
    return __ballot((slot < tokenNum) && (laneId == 0 || prev + 1 != srcTok));
  };

  // Pass 1: run counts per batch; the first kReg batches stay in registers for pass 2.
  ::mori::cco::ccoSdmaQueueDeviceHandle* const qhLoaded = sdma.queueHandle(peer, q);
  index_t tok0[kReg];
  unsigned long long heads0[kReg];
#pragma unroll
  for (int i = 0; i < kReg; ++i) tok0[i] = srcTokOf(warpId + i * warpNum);
  __builtin_amdgcn_sched_barrier(0);  // keep the loads above ahead of the first wait below
  ::mori::cco::ccoSdmaQueueDeviceHandle* const qh = WaveUniform(qhLoaded);
  uint32_t* const ring = WaveUniform(::mori::cco::ccoSdma::ring(qh));
  ::mori::cco::ccoSdmaQueueDeviceHandle h{};  // thread 0's copy for every reserve and publish
  __shared__ bool lastBlock;                  // this block reserved the peer's last rows
  if (threadIdx.x == 0 && tokenNum > 0) h = *::mori::cco::impl::global(qh);
#pragma unroll
  for (int i = 0; i < kReg; ++i) {
    const int b = warpId + i * warpNum;
    heads0[i] = headsOf(b, tok0[i]);
    if (b < nb && laneId == 0) runStart[b + 1] = __popcll(heads0[i]);
  }
  for (int b0 = warpId + groupStride; b0 < nb; b0 += groupStride) {
    index_t tok[kReg];
#pragma unroll
    for (int i = 0; i < kReg; ++i) tok[i] = srcTokOf(b0 + i * warpNum);
#pragma unroll
    for (int i = 0; i < kReg; ++i) {
      const int b = b0 + i * warpNum;
      const unsigned long long heads = headsOf(b, tok[i]);
      if (b < nb && laneId == 0) runStart[b + 1] = __popcll(heads);
    }
  }
  __syncthreads();
  if (warpId == 0) {
    int carry = 0;
    for (int c = 0; c < nb; c += warpSize) {
      const int b = c + laneId;
      int incl = (b < nb) ? runStart[b + 1] : 0;
      for (int o = 1; o < warpSize; o <<= 1) {
        const int up = __shfl_up(incl, o);
        if (laneId >= o) incl += up;
      }
      if (b < nb) runStart[b + 1] = incl + carry;
      carry += __shfl(incl, warpSize - 1);
    }
    if (laneId == 0) {
      runStart[0] = 0;
      runBase =
          carry ? h.ReserveSlot(static_cast<uint64_t>(min(carry, chunkRuns)) * perRun * kPkt, qh)
                : 0;
      if (carry <= chunkRuns) lastBlock = arrive();
    }
  }
  __syncthreads();

  // Pass 2: run g in [lo, hi) to packet g - lo at base; regs: first kReg batches from pass 1.
  auto placeRange = [&](int lo, int hi, uint64_t base, bool regs) {
    auto place = [&](int b, index_t srcTok, unsigned long long heads) {
      const int g = runStart[b] + __popcll(heads & below);
      if (!((heads >> laneId) & 1ull) || g < lo || g >= hi) return;
      const int active =
          min(tokenNum - (bLo + b) * static_cast<int>(warpSize), static_cast<int>(warpSize));
      const unsigned long long later =
          heads & ~((laneId + 1 >= 64) ? ~0ull : ((1ull << (laneId + 1)) - 1));
      const int end = later ? (__ffsll(static_cast<long long>(later)) - 1) : active;
      const int slot = (bLo + b) * warpSize + laneId;
      const uint64_t at = base + static_cast<uint64_t>(g - lo) * perRun * kPkt;
      ::mori::cco::ccoSdma::placeCopy(
          ring, at, dst + static_cast<size_t>(slot) * hiddenBytes,
          const_cast<uint8_t*>(input + static_cast<size_t>(srcTok) * hiddenBytes),
          static_cast<size_t>(end - laneId) * hiddenBytes);
      if (scaleBytes)
        ::mori::cco::ccoSdma::placeCopy(
            ring, at + kPkt, scaleDst + static_cast<size_t>(slot) * scaleBytes,
            const_cast<uint8_t*>(args.scalesBuf + static_cast<size_t>(srcTok) * scaleBytes),
            static_cast<size_t>(end - laneId) * scaleBytes);
    };
    if (regs) {
#pragma unroll
      for (int i = 0; i < kReg; ++i) {
        const int b = warpId + i * warpNum;
        if (b < nb) place(b, tok0[i], heads0[i]);
      }
    }
    for (int b0 = regs ? warpId + groupStride : warpId; b0 < nb; b0 += groupStride) {
      // Skip a group none of whose runs is in range (run starts grow with the batch).
      if (runStart[b0] >= hi || runStart[min(b0 + (kReg - 1) * warpNum, nb - 1) + 1] <= lo)
        continue;
      index_t tok[kReg];
#pragma unroll
      for (int i = 0; i < kReg; ++i) tok[i] = srcTokOf(b0 + i * warpNum);
#pragma unroll
      for (int i = 0; i < kReg; ++i) {
        const int b = b0 + i * warpNum;
        const unsigned long long heads = headsOf(b, tok[i]);
        if (b < nb) place(b, tok[i], heads);
      }
    }
    // Every wave's ring stores drained before thread 0 rings the doorbell.
    ::mori::cco::ccoSdmaPublishStores();
    __syncthreads();
    if (threadIdx.x == 0)
      h.submitPacket(base, base + static_cast<uint64_t>(hi - lo) * perRun * kPkt);
  };
  const int total = runStart[nb];
  if (total <= chunkRuns) {
    if (total > 0) placeRange(0, total, runBase, true);
  } else {
    // Thread 0 reserves a chunk only after publishing the previous one, so chunks stay in order.
    for (int lo = 0; lo < total; lo += chunkRuns) {
      const int hi = min(lo + chunkRuns, total);
      placeRange(lo, hi, runBase, false);
      if (hi < total) {
        if (threadIdx.x == 0) {
          runBase =
              h.ReserveSlot(static_cast<uint64_t>(min(chunkRuns, total - hi)) * perRun * kPkt, qh);
          if (total - hi <= chunkRuns) lastBlock = arrive();
        }
        __syncthreads();
      }
    }
  }
  if constexpr (!kSplit) return true;
  __syncthreads();
  return lastBlock;
#else
  DirectPostRowsPerWarp<kConfig>(args, comm, config, peer, q, dst, srcTokAt, srcTokStride, tokenNum,
                                 hiddenBytes, scaleDst, scaleBytes);
  return true;
#endif
}

// Direct compact dispatch to one peer: the rows (DirectPostRows), then after a block barrier
// warp 0 posts the metadata block and the count, so they are reserved, and executed, behind
// every row. SendCopy staged the metadata and the count word.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void DirectDispatchSend(const EpDispatchCombineArgs& args,
                                                   const ::mori::cco::ccoDevComm& comm,
                                                   const EpInterNodeDeviceCfg& config, int destPe,
                                                   int tokenNum, size_t hiddenBytes,
                                                   size_t xferBytes, size_t srcTokField) {
  ::mori::cco::ccoSdma sdma{comm};
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = threadIdx.x / warpSize;
  const int myPe = args.rank;
  const int peer = destPe - args.lsaBase;
  const int q = QueueForPe(comm, destPe, args.sdmaQueueOffset);
  const EpInterNodeRegion landing = args.reg(args.offDispatchInp);
  const EpInterNodeRegion staging = args.reg(args.offStaging);
  const EpInterNodeRegion words = args.reg(args.offSendCount);

  if (destPe == myPe) {
    if (warpId == 0 && laneId == 0) {
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      core::AtomicStoreReleaseSystem(reinterpret_cast<uint64_t*>(recvCounts + RecvCountSlot(myPe)),
                                     static_cast<uint64_t>(tokenNum) + 1);
    }
    return;
  }
  // Segmented-capable op: the mode word goes first on the queue the count follows on.
  if (SegCapable(args) && warpId == 0 && laneId == 0)
    sdma.put(peer, EpInterNodeWindow(words), EpInterNodeOffset(words, SegStrideSlot(config, myPe)),
             EpInterNodeWindow(words), EpInterNodeOffset(words, SegStrideSourceSlot(config)),
             sizeof(uint64_t), q);
  __syncthreads();

  const size_t metaBytes = xferBytes - hiddenBytes;
  const uint8_t* metaLocal = staging->template GetAs<const uint8_t*>() +
                             DirectStagingMetaOffset(config, destPe, 0, metaBytes);
  DirectPostRows<kConfig>(
      args, comm, config, peer, q,
      static_cast<uint8_t*>(RegionPeerPtr(
          landing, peer, DirectHiddenOffset(config, myPe, 0, hiddenBytes, xferBytes))),
      metaLocal + srcTokField, metaBytes, tokenNum, hiddenBytes);
  __syncthreads();
  if (warpId != 0) return;
  PostPayloadAndNotify(
      comm, peer, q, EpInterNodeWindow(landing),
      EpInterNodeOffset(landing, DirectMetaOffset(config, myPe, 0, hiddenBytes, xferBytes)),
      EpInterNodeWindow(staging),
      EpInterNodeOffset(staging, DirectStagingMetaOffset(config, destPe, 0, metaBytes)),
      static_cast<size_t>(tokenNum) * metaBytes, EpInterNodeWindow(args.reg(args.offRecvTokenNum)),
      EpInterNodeOffset(args.reg(args.offRecvTokenNum), RecvCountSlot(myPe)),
      EpInterNodeWindow(words), EpInterNodeOffset(words, SendCountSlot(destPe)));
}

}  // namespace asyncll

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchAsyncLLSlotAssign -- dispatch pass 1                         */
/* ---------------------------------------------------------------------------------------------- */
// Routing only: no payload moves here. Groups of numExpertPerToken lanes inside a
// warp cooperate on one token, each lane holding its own expert's destination, so
// the deduplication is a register shuffle rather than a reload.
//
// Slots are claimed per BLOCK, not per entry. Every surviving entry needs the
// next slot of its destination PE, and there are only worldSize counters: one
// atomicAdd per entry put ~18k atomics on 8 addresses at EP8, top-6, 4096 tokens
// and the pass took ~58 us for a few hundred KB of routing. Here each warp claims
// its run inside the block (one LDS atomic per destination PE it touches), then
// one thread per PE claims the block's whole run from the global counter: ~6.5 us
// at that shape. The block walks the tokens in lockstep so every __syncthreads
// is reached by all of its threads.
//
// Which slot an entry gets was never deterministic (it was atomic order before),
// and nothing downstream depends on it: RecvCopy compacts each PE's slots in slot
// order and records where each token came from.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSlotAssign_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  static_assert(kConfig.worldSize <= 64,
                "slot claims are tracked in one 64-entry LDS array per destination PE");
  constexpr int numEpt = kConfig.numExpertPerToken;
  const int tokensPerWarp = warpSize / numEpt;
  const int tokensPerBlock = warpNum * tokensPerWarp;
  const int expertIdx = laneId % numEpt;
  const int inWarpTokIdx = laneId / numEpt;
  const int baseLane = inWarpTokIdx * numEpt;
  const unsigned long long lanesBelow = (1ull << laneId) - 1ull;

  __shared__ index_t blockClaims[64];
  __shared__ index_t blockBase[64];

  constexpr bool kFuseMeta = asyncll::Direct<kConfig> && MORI_ASYNCLL_FUSE_META;
  const int segStride = kFuseMeta ? asyncll::SegStrideOf(args, config) : 0;
  // The fused field writes' bases, loaded ahead of the barriers they would otherwise follow.
  uint8_t* const metaStaging =
      kFuseMeta ? args.reg(args.offStaging)->template GetAs<uint8_t*>() : nullptr;
  uint8_t* const metaLanding =
      kFuseMeta ? args.reg(args.offDispatchInp)->template GetAs<uint8_t*>() : nullptr;
  // Fused field writes: spread the tokens over every block (whole warps' worth each), so a
  // few tokens still use the whole grid for the writes.
  int chunk = tokensPerBlock;
  if constexpr (kFuseMeta) {
    const int per = (args.curRankNumToken + blockNum - 1) / blockNum;
    chunk = min(tokensPerBlock, ((per + tokensPerWarp - 1) / tokensPerWarp) * tokensPerWarp);
    chunk = max(chunk, tokensPerWarp);
  }

  for (int blockTokBase = blockId * chunk; blockTokBase < args.curRankNumToken;
       blockTokBase += blockNum * chunk) {
    if (threadIdx.x < kConfig.worldSize) blockClaims[threadIdx.x] = 0;
    __syncthreads();

    const int inChunk = warpId * tokensPerWarp + inWarpTokIdx;
    const int tokenId = blockTokBase + inChunk;
    // warpSize need not be a multiple of numEpt (64 % 6 == 4 at top-k 6): the
    // trailing lanes alias the next warp's first token, so one entry would get two
    // slots and the orphan -- never written by SendCopy -- is read as a token by
    // the receiver. The test at top-k 6/9/10/12 exists for this.
    //
    // Those lanes and the ones past the last token stay in the loop as inactive
    // rather than continuing: the ballots and shuffles below, and the block
    // barriers, need every lane.
    const bool active =
        (inWarpTokIdx < tokensPerWarp) && (inChunk < chunk) && (tokenId < args.curRankNumToken);
    const int i = tokenId * numEpt + expertIdx;
    // A negative expert id is the caller's "drop this slot" sentinel.
    const index_t destExpert = active ? args.tokenIndices[i] : static_cast<index_t>(-1);
    const float metaWeight = (kFuseMeta && active && args.weightsBuf) ? args.weightsBuf[i] : 0.0f;
    const bool dropped = (destExpert < 0);
    const index_t destPe = dropped ? 0 : (destExpert / config.numExpertPerRank);

    // Deduplicate by shuffling the earlier lanes' destinations out of their
    // registers. inWarpTokIdx is group-uniform, so a group is entirely active or
    // entirely not, and an inactive group's result is discarded. The mask keeps
    // the trailing lanes' source lane inside the wave.
    bool isDuplicate = false;
    for (int j = 0; j < numEpt - 1; ++j) {
      const int src = (baseLane + j) & (warpSize - 1);
      const index_t prevPe = __shfl(destPe, src);
      const bool prevDropped = __shfl(static_cast<int>(dropped), src) != 0;
      if ((j < expertIdx) && !prevDropped) isDuplicate = isDuplicate || (prevPe == destPe);
    }

    // An out-of-range expert id (an EPLB physical id past worldSize*numExpertPerRank)
    // would index destPeTokenCounter and SendBufSlotOffset out of bounds. Folded
    // into the same null result as the duplicate rather than skipped early, again
    // to keep the shuffles coherent.
    const bool peOutOfRange = (destPe < 0) || (destPe >= config.worldSize);
    const bool claims = active && !dropped && !isDuplicate && !peOutOfRange;

    // This lane's offset within the block's run for its PE.
    index_t inBlockTokId = 0;
#pragma unroll
    for (int pe = 0; pe < kConfig.worldSize; ++pe) {
      const unsigned long long lanes = __ballot(claims && (destPe == pe));
      if (lanes) {
        const int leader = __ffsll(static_cast<long long>(lanes)) - 1;
        index_t warpBase = 0;
        if (laneId == leader)
          warpBase = atomicAdd(&blockClaims[pe], static_cast<index_t>(__popcll(lanes)));
        warpBase = __shfl(warpBase, leader);
        if (claims && (destPe == pe))
          inBlockTokId = warpBase + static_cast<index_t>(__popcll(lanes & lanesBelow));
      }
    }
    __syncthreads();
    if (threadIdx.x < kConfig.worldSize) {
      const index_t n = blockClaims[threadIdx.x];
      blockBase[threadIdx.x] = n ? atomicAdd(args.destPeTokenCounter + threadIdx.x, n) : 0;
    }
    __syncthreads();

    const index_t destTokId = (active && claims) ? blockBase[destPe] + inBlockTokId : 0;
    if (active) {
      // The counter is cleared by RecvCopy, one per round, so a SECOND
      // dispatch_send with no recv between it and the first keeps climbing from
      // where the first left off. Unbounded, SendBufSlotOffset then encodes a
      // slot past this PE's stride and every later pass writes into the NEXT
      // PE's slice -- silently, because the decoders are plain divide/modulo and
      // a slot one stride over decodes as a legal slot of a legal PE.
      //
      // Out of range is the same condition the dedup already has a sentinel for,
      // so use it: the entry is simply not sent. That makes the round WRONG, not
      // right -- it is the host's call-order check that stops it being reachable
      // (dispatch_combine_op.py) -- but wrong-and-in-bounds beats corrupting a
      // peer's buffer, which is the one thing this kernel can still choose.
      index_t mapped = (claims && destTokId < config.MaxNumTokensToSendPerRank())
                           ? SendBufSlotOffset(config, destPe, destTokId)
                           : NullSendBufSlotOffset(config);
      // Segmented: a slot past the stride is dropped (same rule as DirectWriteMeta*).
      if (kFuseMeta && segStride > 0 && mapped < NullSendBufSlotOffset(config) &&
          destTokId >= segStride)
        mapped = NullSendBufSlotOffset(config);
      args.dispDestTokIdMap[i] = mapped;
    }
    if constexpr (kFuseMeta) {
      if (segStride > 0) {
        const index_t dst = (active && claims && destTokId < config.MaxNumTokensToSendPerRank() &&
                             destTokId < segStride)
                                ? SendBufSlotOffset(config, destPe, destTokId)
                                : NullSendBufSlotOffset(config);
        asyncll::DirectWriteMetaGroup<kConfig>(args, config, hiddenBytes, segStride, active,
                                               tokenId, expertIdx, baseLane, dst, destExpert,
                                               metaWeight, metaStaging, metaLanding);
      }
    }
    // blockClaims is reset at the top of the next round.
    __syncthreads();
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpDispatchAsyncLLSendCopy -- dispatch pass 2                         */
/* ---------------------------------------------------------------------------------------------- */
// Payload only: the slots are already decided, so several warps can split one
// token's hidden dimension and the copy is pure bandwidth.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  // A compact round on a segmented-capable op still sends its mode word (stride 0), so a peer
  // running a segmented round in it notices. SendTransfer hands this word to SDMA after a
  // kernel boundary, like the count below.
  if (asyncll::SegCapable(args) && globalWarpId == 0 && laneId == 0) {
    uint8_t* words = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    *reinterpret_cast<uint64_t*>(words + asyncll::SegStrideSourceSlot(config)) =
        asyncll::SegModeWord(args.crossDeviceBarrierFlag[0], 0);
  }
  if constexpr (asyncll::EarlyNotify<kConfig>) {
    // SlotAssign has completed. The following SendTransfer kernel may hand
    // these words directly to SDMA without an additional system fence.
    if (globalWarpId == 0 && laneId < npes) {
      uint8_t* counts = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
      *reinterpret_cast<uint64_t*>(counts + asyncll::SendCountSlot(laneId)) =
          static_cast<uint64_t>(asyncll::SendTokenCount(
              config, core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId))) +
          1;
    }
  }
  const index_t totalEntries = args.curRankNumToken * config.numExpertPerToken;
  // v1 divides by totalEntries unguarded, which is an integer division by zero on
  // a rank that was handed no tokens -- reachable, and covered by the ragged
  // token-count test.
  if (totalEntries <= 0) return;

  if constexpr (asyncll::Direct<kConfig>) {
    // Metadata only (SendTransfer posts the hidden rows from the input). One group of numEpt
    // lanes per TOKEN: lane j holds the token's entry j and element j of each field, loaded
    // once, and the group writes that row into every record the token claimed. Only stores
    // follow the loads, so a few blocks keep up.
    asyncll::DirectWriteMetaAuto<kConfig>(args, config, hiddenBytes, xferBytes, -1);
    return;
  }

  const index_t warpsPerToken = (globalWarpNum + totalEntries - 1) / totalEntries;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);

  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < totalEntries * warpsPerToken; i += globalWarpNum) {
    const index_t entryId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;

    const index_t destTokOffset = args.dispDestTokIdMap[entryId];
    // Deduplicated / dropped entries carry the out-of-range sentinel.
    if (destTokOffset >= NullSendBufSlotOffset(config)) continue;

    const index_t srcTokId = entryId / config.numExpertPerToken;
    uint8_t* dst = stagingPtr + static_cast<size_t>(destTokOffset) * xferBytes;

    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenOffset,
          static_cast<const uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes + hiddenOffset,
          len);
    }

    // The metadata is small and goes with the first sub-warp, so the others stay
    // on the payload.
    if (inTokenPartId == 0) {
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenBytes, reinterpret_cast<uint8_t*>(args.tokenIndices) + srcTokId * indexBytes,
          indexBytes);
      if (args.weightsBuf)
        core::WarpCopy<uint8_t, 1>(
            dst + hiddenBytes + indexBytes,
            reinterpret_cast<uint8_t*>(args.weightsBuf) + srcTokId * weightBytes, weightBytes);
      if (args.scalesBuf && (scaleBytes > 0))
        core::WarpCopy<uint8_t, 1>(dst + hiddenBytes + indexBytes + weightBytes,
                                   args.scalesBuf + srcTokId * scaleBytes, scaleBytes);
      if (laneId == 0)
        reinterpret_cast<index_t*>(dst + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
            static_cast<index_t>(asyncll::SrcTokenIndex(config, myPe, srcTokId));
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*            EpDispatchAsyncLLSendCopyLsa -- dispatch passes 2 and 3, fused (LSA only)           */
/* ---------------------------------------------------------------------------------------------- */
// The same gather, landed in the peer instead of in local staging.
//
// The unfused pair moves the payload twice: SendCopy gathers scattered tokens
// into staging at SendBufSlotOffset(destPe, slot), then SendTransferLsa bulk
// copies that contiguous range into the peer. The second pass exists because
// ccoSdma needs a local, contiguous source to hand the engine. A peer store has
// no such requirement -- the grid is already doing the move -- so for this
// transport the gather can address the peer directly and the bulk copy
// disappears, taking one of AsyncLL's three passes over the payload with it.
//
// Self is the exception: RecvCopy reads this rank's own slice out of staging,
// not out of dispatchInp, so entries for myPe still land where they did.
//
// It also absorbs the count publication, which is a handful of words and the
// only other thing SendTransferLsa did. RecvTransfer is still what sends those
// counts, one kernel boundary later, so the payload is visible to the peer
// before its count is.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendCopyLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  // Unconditional, not gated on having tokens: a rank handed none still has to
  // publish its zero or every peer waits on it forever.
  if (globalWarpId == 0 && laneId < npes) {
    uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    *reinterpret_cast<uint64_t*>(sendCount + asyncll::SendCountSlot(laneId)) =
        static_cast<uint64_t>(asyncll::SendTokenCount(
            config, core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId))) +
        1;
  }

  const index_t totalEntries = args.curRankNumToken * config.numExpertPerToken;
  if (totalEntries <= 0) return;

  const index_t warpsPerToken = (globalWarpNum + totalEntries - 1) / totalEntries;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);

  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < totalEntries * warpsPerToken; i += globalWarpNum) {
    const index_t entryId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;

    const index_t destTokOffset = args.dispDestTokIdMap[entryId];
    if (destTokOffset >= NullSendBufSlotOffset(config)) continue;

    const int destPe = PeFromSendBufSlotOffset(config, destTokOffset);
    const int slotId = SlotIdFromSendBufSlotOffset(config, destTokOffset);
    const index_t srcTokId = entryId / config.numExpertPerToken;

    // The two halves of SendBufSlotOffset's symmetry, as in the SDMA transfer:
    // locally a slot is indexed by its DESTINATION, in the peer by its SOURCE.
    uint8_t* dst;
    if (destPe == myPe) {
      dst = stagingPtr + static_cast<size_t>(destTokOffset) * xferBytes;
    } else {
      dst = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>(destPe) +
            static_cast<size_t>(SendBufSlotOffset(config, myPe, slotId)) * xferBytes;
    }

    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      // Unroll 4, not 1: see the LSA section's note on store concurrency. This is
      // the peer write, and it is the single biggest lever on this transport.
      core::WarpCopy<uint8_t, 4>(
          dst + hiddenOffset,
          static_cast<const uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes + hiddenOffset,
          len);
    }

    if (inTokenPartId == 0) {
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenBytes, reinterpret_cast<uint8_t*>(args.tokenIndices) + srcTokId * indexBytes,
          indexBytes);
      if (args.weightsBuf)
        core::WarpCopy<uint8_t, 1>(
            dst + hiddenBytes + indexBytes,
            reinterpret_cast<uint8_t*>(args.weightsBuf) + srcTokId * weightBytes, weightBytes);
      if (args.scalesBuf && (scaleBytes > 0))
        core::WarpCopy<uint8_t, 1>(dst + hiddenBytes + indexBytes + weightBytes,
                                   args.scalesBuf + srcTokId * scaleBytes, scaleBytes);
      if (laneId == 0)
        reinterpret_cast<index_t*>(dst + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
            static_cast<index_t>(asyncll::SrcTokenIndex(config, myPe, srcTokId));
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchAsyncLLSendTransfer -- dispatch pass 3                       */
/* ---------------------------------------------------------------------------------------------- */
// Issue and return. Nothing is waited on here: that is what makes the seam between
// dispatch_send and dispatch_recv worth anything -- the copies drain under
// whatever the caller runs next.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendTransfer_body(EpDispatchCombineArgs args,
                                                   const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};

  // One issuing warp per destination: warp 0 of the block that owns the PE. More
  // warps on one peer would not make the copy faster (one peer is one link) and
  // would serialise on that queue's commit chain.
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    const int tokenNum =
        asyncll::SendTokenCount(config, core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe));

    if constexpr (asyncll::Direct<kConfig>) {
      asyncll::DirectDispatchSend<kConfig>(args, comm, config, destPe, tokenNum, hiddenBytes,
                                           xferBytes, indexBytes + weightBytes + scaleBytes);
      continue;
    }
    if (warpId != 0) continue;

    // Segmented-capable op (one node, SDMA): the mode word goes first on the queue the count
    // follows on, so it has landed when the peer sees the count.
    if (asyncll::SegCapable(args) && destPe != myPe && laneId == 0 &&
        destPe / config.gpuPerNode == myNode) {
      const EpInterNodeRegion wordRegion = args.reg(args.offSendCount);
      sdma.put(destPe - args.lsaBase, EpInterNodeWindow(wordRegion),
               EpInterNodeOffset(wordRegion, asyncll::SegStrideSlot(config, myPe)),
               EpInterNodeWindow(wordRegion),
               EpInterNodeOffset(wordRegion, asyncll::SegStrideSourceSlot(config)),
               sizeof(uint64_t), asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset));
    }

    if constexpr (asyncll::EarlyNotify<kConfig>) {
      if (destPe == myPe) {
        if (laneId == 0) {
          uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
          core::AtomicStoreReleaseSystem(
              reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)),
              static_cast<uint64_t>(tokenNum) + 1);
        }
      } else {
        const size_t remoteBase =
            static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes;
        const size_t localBase =
            static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes;
        asyncll::PostPayloadAndNotify(
            comm, destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
            EpInterNodeWindow(args.reg(args.offDispatchInp)),
            EpInterNodeOffset(args.reg(args.offDispatchInp), remoteBase),
            EpInterNodeWindow(args.reg(args.offStaging)),
            EpInterNodeOffset(args.reg(args.offStaging), localBase),
            static_cast<size_t>(tokenNum) * xferBytes,
            EpInterNodeWindow(args.reg(args.offRecvTokenNum)),
            EpInterNodeOffset(args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe)),
            EpInterNodeWindow(args.reg(args.offSendCount)),
            EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendCountSlot(destPe)));
      }
      continue;
    }

    // Publish the count into the local staging word now, while the value is in
    // hand; RecvTransfer is what sends it, after it has drained the payload.
    if (laneId == 0) {
      uint64_t* sendCount = args.reg(args.offSendCount)->template GetAs<uint64_t*>();
      *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sendCount) +
                                   asyncll::SendCountSlot(destPe)) =
          static_cast<uint64_t>(tokenNum) + 1;
    }

    // Self needs no transfer: RecvCopy reads this rank's own slice straight out
    // of staging, exactly as v1 does.
    if (destPe == myPe) continue;
    if (tokenNum <= 0) continue;

    const size_t remoteBase = static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes;
    const size_t localBase = static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes;

    if (destPe / config.gpuPerNode != myNode) {
      // Cross-node: RDMA write, STRIPED over the QPs. This is the opposite of
      // what the same-node branch below does, and both are right for their
      // transport. Splitting an SDMA copy only makes each piece smaller on the
      // one link a peer has; splitting an RDMA transfer spreads it over QPs that
      // really do progress in parallel (the op README records that
      // num_qp_per_pe=1 starves this path by about 1.5x).
      //
      // No signal rides along: this phase only posts. RecvTransfer drains and
      // then announces, which is v1's shape and keeps this launch non-blocking.
      const int chunk = core::CeilDiv(tokenNum, config.numQpPerPe);
      for (int qpId = 0; qpId < config.numQpPerPe; ++qpId) {
        const int first = qpId * chunk;
        const int count = min(chunk, tokenNum - first);
        if (count <= 0) break;
        EpInterNodePut(comm, args.reg(args.offDispatchInp),
                       remoteBase + static_cast<size_t>(first) * xferBytes,
                       args.reg(args.offStaging),
                       localBase + static_cast<size_t>(first) * xferBytes,
                       static_cast<size_t>(count) * xferBytes, destPe, qpId);
      }
      continue;
    }

    const int q = asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset);
    // The destination addresses ITS copy of dispatchInp at this rank's slice, and
    // the source is this rank's staging at the destination's slice -- the two
    // halves of SendBufSlotOffset's symmetry, and the one place a transposed
    // index silently delivers another rank's tokens.
    //
    // Whole, not sliced per QP. This is the measured departure from v1: at 64
    // tokens a peer's payload is ~600 KB, which SDMA moves at ~46 GB/s in one op
    // and ~35 GB/s as two 300 KB ones. Warp scope so lane 0 posts for the warp
    // without an explicit if.
    sdma.put<::mori::cco::ccoCoopWarp>(destPe - args.lsaBase,
                                       EpInterNodeWindow(args.reg(args.offDispatchInp)),
                                       EpInterNodeOffset(args.reg(args.offDispatchInp), remoteBase),
                                       EpInterNodeWindow(args.reg(args.offStaging)),
                                       EpInterNodeOffset(args.reg(args.offStaging), localBase),
                                       static_cast<size_t>(tokenNum) * xferBytes, q);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchAsyncLLRecvTransfer -- dispatch pass 4                       */
/* ---------------------------------------------------------------------------------------------- */
// Drain, announce, then wait for everyone else's announcement.
//
// The quiet lives here rather than in SendTransfer for the same reason v1 puts it
// here: a quiet in the send kernel would block the launch that is supposed to
// return immediately, and the seam would buy nothing.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvTransfer_body(EpDispatchCombineArgs args,
                                                   const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;

    const bool sameNode = (destPe / config.gpuPerNode == myNode);
    // Both drains are WARP-COLLECTIVE and must stay above the laneId guard:
    // ccoSdma::quietQueue syncs the group, and EpInterNodeQuiet takes ccoGda's
    // warp-level CQ poll lock. Called from one lane they deadlock rather than
    // misbehave, which is why the guard is below them and not above.
    // Direct mode also drains the loopback queue: a segmented round lands its own rows there.
    if (destPe == myPe && asyncll::Direct<kConfig>)
      sdma.quietQueue<::mori::cco::ccoCoopWarp>(
          myPe - args.lsaBase, asyncll::QueueForPe(comm, myPe, args.sdmaQueueOffset));
    if (destPe != myPe) {
      if (sameNode) {
        // Our payload to this peer is out of the queue before its count word
        // goes, so the peer seeing a non-zero count has the tokens too.
        sdma.quietQueue<::mori::cco::ccoCoopWarp>(
            destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset));
      } else {
        // Same argument over RDMA: drain every QP the send leg striped across.
        for (int qpId = 0; qpId < config.numQpPerPe; ++qpId) EpInterNodeQuiet(comm, destPe, qpId);
      }
    }

    // Early notifications were already placed after their payloads by SendTransfer.
    // Keep local completion: incoming ready words do not prove our sources are free.
    if constexpr (asyncll::EarlyNotify<kConfig>) continue;
    if (laneId != 0) continue;

    const uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<const uint8_t*>();
    const uint64_t countValue =
        *reinterpret_cast<const uint64_t*>(sendCount + asyncll::SendCountSlot(destPe));

    if (destPe == myPe) {
      // No engine for a local store; the prefix sum still has to see the slot.
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      core::AtomicStoreRelaxedSystem(
          reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)), countValue);
    } else if (sameNode) {
      sdma.put(destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offRecvTokenNum)),
               EpInterNodeOffset(args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe)),
               EpInterNodeWindow(args.reg(args.offSendCount)),
               EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendCountSlot(destPe)),
               sizeof(uint64_t), asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset));
    } else {
      // Cross-node. An ADD rather than a store, because that is the only remote
      // update ccoGda offers at an arbitrary window offset -- and it is exactly
      // equivalent here: combine clears this slot every round, so adding
      // (tokenNum + 1) to zero lands the same value v1 stores with
      // ShmemPutUint64ImmNbiThread.
      EpInterNodeAtomicAdd(comm, args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe),
                           countValue, destPe);
    }
  }

  // Wait for every source to have announced. One lane per source rank; the count
  // is biased by one so that a rank sending zero tokens still publishes a value
  // distinguishable from "has not arrived".
  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();
  if (blockId == 0 && warpId == 0 && laneId < npes) {
    const uint64_t* slot =
        reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
    while (core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(slot)) == 0) {
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpDispatchAsyncLLRecvCopy -- dispatch pass 5                         */
/* ---------------------------------------------------------------------------------------------- */
// Unpack what arrived into the dispatch output, and build the two maps combine
// needs. Per-PE output offsets come from a warp prefix sum rather than an atomic,
// so every block derives the same layout independently.
//
// The whole grid walks the concatenated output rather than giving each source PE
// a fixed 1/worldSize of the blocks: with a static split the PE that sent the
// most sets the kernel's length while the other groups sit idle, and the PEs a
// rank receives from are rarely even. The owning PE of output row r is the last
// one whose exclusive offset is <= r, read off the prefix sum with one ballot.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  // The absolute bound only. The prefix sum below is one wavefront wide, so the
  // real limit is this arch's wave -- 32 on gfx1250, not 64 -- and waveSize is
  // not part of the rendered cfg, so it cannot be said here. The exact check is
  // host-side in EpAsyncLLRenderSource, which every compile goes through and
  // which does know it. This catches only a worldSize no wave could hold.
  static_assert(kConfig.worldSize <= 64,
                "the per-PE offset prefix sum is one wavefront wide; worldSize must fit it");

  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();

  uint64_t myPeTokens = 0;
  if (laneId < npes)
    myPeTokens =
        *reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId)) - 1;

  uint64_t inclusive = myPeTokens;
  for (int offset = 1; offset < npes; offset <<= 1) {
    const uint64_t n = __shfl_up(inclusive, offset);
    if (laneId >= offset) inclusive += n;
  }
  uint64_t peExclusive = __shfl_up(inclusive, 1);
  if (laneId == 0) peExclusive = 0;
  const uint64_t totalTokens = __shfl(inclusive, npes - 1);

  const uint8_t* peerBase = args.reg(args.offDispatchInp)->template GetAs<const uint8_t*>();
  // This rank's own tokens were never transferred; they are still in staging.
  const uint8_t* selfBase = args.reg(args.offStaging)->template GetAs<const uint8_t*>();
  uint8_t* outTokens = args.reg(args.offDispatchOut)->template GetAs<uint8_t*>();
  uint8_t* outIndices = args.reg(args.offOutIndices)->template GetAs<uint8_t*>();
  uint8_t* outWeights = args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>();
  uint8_t* outScales = args.reg(args.offOutScales)->template GetAs<uint8_t*>();

  const index_t warpsPerToken =
      (totalTokens > 0) ? static_cast<index_t>((globalWarpNum + totalTokens - 1) / totalTokens) : 1;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);

  for (int i = globalWarpId; i < static_cast<int>(totalTokens) * warpsPerToken;
       i += globalWarpNum) {
    const index_t destTokId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;
    // Lane 0 always qualifies (its exclusive offset is 0), so the mask is never
    // empty; a PE that sent nothing shares its offset with the next and is
    // skipped by taking the highest qualifying lane.
    const unsigned long long owners =
        __ballot((laneId < npes) && (peExclusive <= static_cast<uint64_t>(destTokId)));
    const int srcPe = 63 - __clzll(static_cast<long long>(owners));
    const index_t tokenId = destTokId - static_cast<index_t>(__shfl(peExclusive, srcPe));
    const uint8_t* hidden;
    const uint8_t* meta;
    if constexpr (asyncll::Direct<kConfig>) {
      // Peers' records are in the landing; this rank's own were only staged.
      meta = (srcPe != myPe) ? peerBase + asyncll::DirectMetaOffset(config, srcPe, tokenId,
                                                                    hiddenBytes, xferBytes)
                             : selfBase + asyncll::DirectStagingMetaOffset(config, srcPe, tokenId,
                                                                           xferBytes - hiddenBytes);
      if (srcPe != myPe) {
        hidden =
            peerBase + asyncll::DirectHiddenOffset(config, srcPe, tokenId, hiddenBytes, xferBytes);
      } else {
        // This rank's own rows were never copied: read them from the input.
        const index_t srcTok =
            reinterpret_cast<const index_t*>(meta + indexBytes + weightBytes + scaleBytes)[0] -
            asyncll::SrcTokenIndex(config, myPe, 0);
        hidden = static_cast<const uint8_t*>(args.inpTokenBuf) +
                 static_cast<size_t>(srcTok) * hiddenBytes;
      }
    } else {
      hidden = ((srcPe != myPe) ? peerBase : selfBase) +
               (static_cast<size_t>(SendBufSlotOffset(config, srcPe, 0)) + tokenId) * xferBytes;
      meta = hidden + hiddenBytes;
    }

    // Four 16 B loads in flight per lane: the source is what SDMA just wrote, so
    // it comes from HBM, not cache, and one load per lane left it latency-bound.
    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 4>(outTokens + destTokId * hiddenBytes + hiddenOffset,
                                 hidden + hiddenOffset, len);
    }

    if (inTokenPartId == 0) {
      // The metadata fields are a few dozen bytes each, below WarpCopy's vector
      // width, which sent them through its byte-at-a-time tail; copy them a word
      // per lane instead.
      asyncll::LaneCopyWords(outIndices + destTokId * indexBytes, meta, indexBytes);
      if (args.weightsBuf)
        asyncll::LaneCopyWords(outWeights + destTokId * weightBytes, meta + indexBytes,
                               weightBytes);
      if (scaleBytes > 0)
        asyncll::LaneCopyWords(outScales + destTokId * scaleBytes, meta + indexBytes + weightBytes,
                               scaleBytes);
      if (laneId == 0) {
        // Where this token came from, so combine's send leg can put the result
        // back in the slot it arrived in.
        args.dispReceiverIdxMap[destTokId] = SendBufSlotOffset(config, srcPe, tokenId);
        args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>()[destTokId] =
            reinterpret_cast<const index_t*>(meta + indexBytes + weightBytes + scaleBytes)[0];
      }
    }
  }

  // One warp in the grid owns the epilogue: publish the total, clear what the
  // next call reads, and advance the combine barrier epoch.
  if (globalWarpId == 0) {
    // Compact round on a segmented-capable op: every peer must have run one too.
    if (asyncll::SegCapable(args) && laneId < npes)
      asyncll::SegCheckModeWords(args, config, laneId, myPe, 0);
    if (laneId == 0) args.totalRecvTokenNum[0] = static_cast<index_t>(totalTokens);
    // The recv counts are deliberately NOT cleared here. Combine's send leg needs
    // to know how many tokens each peer sent us -- that is exactly how many we owe
    // it back -- and this is the only place that number exists. v1 clears it in
    // EpCombineLowLatencyAsyncSendTransfer for the same reason, after reading it.
    if (laneId < npes) args.destPeTokenCounter[laneId] = 0;
    // Direct: combine has no SendCopy to snapshot the counts, so leave them here. This
    // wave drained the sends that read these words before it got here.
    if constexpr (asyncll::Direct<kConfig>) {
      if (laneId < npes)
        *reinterpret_cast<uint64_t*>(args.reg(args.offSendCount)->template GetAs<uint8_t*>() +
                                     asyncll::SendCountSlot(laneId)) = myPeTokens;
    }
    if (laneId == 0) {
      args.dispatchGridBarrier[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1ull);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                   LSA transfer variants -- same protocol, peer stores instead                  */
/* ---------------------------------------------------------------------------------------------- */
//
// Drop-in replacements for the two dispatch transfer passes, for the case where
// every peer is on this node. The protocol, the slot arithmetic and the count
// word are unchanged; only how the bytes cross differs.
//
// The copy engine tops out near 60 GB/s and needs ~6 us to start, while a peer
// store over the flat LSA VA runs at xGMI speed, so at these sizes (a per-peer
// payload of ~600 KB at 64 tokens x 7168 x bf16) the engine is latency-bound and
// a store should win.
//
// It does at 64 tokens and nowhere else. Measured on an IDLE, RESERVED MI355X
// node (crsuse2 v2-008) at EP8, hidden 7168, bf16, dispatch+combine as a CUDA
// graph; two full sweeps agreed to within 0.3 percentage points at every point:
//
//   tok/rank      64     128     256     512    1024    2048   (us)
//   sdma       102.0   136.8   194.3   292.1   523.1   982.0
//   lsa         97.7   138.9   201.5   320.3   581.4  1094.2
//   delta      -4.2%   +1.6%   +3.6%   +9.7%  +11.1%  +11.4%
//
// The reasoning above predicted the opposite sign at the top end -- the engine
// does saturate there, so peer stores "should" win -- and the measurement says
// lsa loses by a widening margin instead. Two things the prediction left out:
// the transfer is a kernel of its own, so nothing overlaps it either way and
// SDMA's separate hardware is pure gain; and a grid doing the copy contends with
// its own HBM traffic, where the engines do not. The ordinary v2 intranode
// dispatch, which writes straight into the peer and stages nothing, reaches
// 856.9 us on the same point, so the grid is not short of bandwidth -- it is
// short of bandwidth *while also* running the staging copies this protocol adds.
//
// That staging is why neither transport closes the gap to v2 native: AsyncLL
// moves each payload THREE times (input -> staging, staging -> peer, peer ->
// dispatch output) because that is what an RDMA leg needs, and changing the
// middle hop leaves the other two alone.
//
// So: sdma is the default on evidence, not just because it is the internode
// shape. lsa is worth keeping for the small-batch case and for a caller with an
// empty seam, since it also costs CUs that an SDMA transfer leaves free.
//
// STORE CONCURRENCY IS THE LEVER ON THIS TRANSPORT. The three peer-facing copies
// here (the fused gather and the two bulk transfers) carry Unroll=4 rather than
// WarpCopy's default of 1. At Unroll=1 a lane has exactly ONE 16 B store in
// flight and the warp walks a 14 KB token as 28 dependent round trips to the
// peer; raising it to 4 keeps four in flight and the latency stops being
// serial. Note that EpCombineAsyncLLSendCopy_body, whose copy is LOCAL, was
// already at 4 -- only the peer-facing sites had been left at 1, which is the
// wrong way round, since a peer store is the one with the long latency to hide.
//
// Measured on an idle gfx1250 node at EP4, hidden 7168, topk 8, bf16, graph;
// five interleaved rounds per arm, medians, arms differing only in
// MORI_SOURCE_ROOT. us for the dispatch+combine pair:
//
//   tok/rank        128      512     2048
//   Unroll=1      646.5    813.6   1364.1
//   Unroll=4      644.3    715.4    974.5
//   delta         -0.3%   -12.1%   -28.6%
//
// The dispatch half alone is -13.5% / -45.1% / -53.9%: 202 -> 438 GB/s at 2048
// in bench_ep.py's accounting. Unroll=8 was measured in the same matrix and is
// NOT better -- it ties at 512 and 2048 and loses 1.4% at 128, so 4 is the
// setting, not merely a direction.
//
// Two things tried in the same matrix and rejected, both documented in
// python/mori/ops/dispatch_combine_v2/README.md: moving the transfer onto the
// gfx1250 TDM engine, and padding the slot stride to a 128 B row.
//
// INTERMITTENT HANG observed on gfx125x UNDER CUDA GRAPH CAPTURE before the
// count-reset ordering fix below. The numbers above are MI355X (gfx950, wave64).
// On gfx1250 at EP4, hidden 7168, 64 tokens:
//
//   lsa  + eager          passes every run, output verifies
//   lsa  + graph capture  hung 6 runs out of 8
//   sdma + graph capture  passes every run
//
// These runs observed hangs only for LSA under graph capture. They do not
// establish that eager execution or other architectures are immune. An earlier
// note incorrectly called LSA broken on wave32 based on one capturing benchmark
// run compared with a single-node test that never captures.
//
// A protocol race exists independently of payload visibility: CombineRecvTransferLsa
// used to publish its epoch, wait for peers, and only THEN clear recv counts. A
// fast peer could pass the epoch handshake and publish its next dispatch count
// before our late clear erased it. SDMA clears in CombineSendTransfer, before
// the epoch announcement, so it does not have this window. LSA now clears each
// peer's count before its release epoch store, paired with an acquire wait.
//
// Fixing it removes the hang. Eight interleaved pairs on an idle gfx1250 at the
// shape above, base and fixed alternating, no contaminated samples:
//
//   baseline  1 pass / 7 hang
//   fixed     8 pass / 0 hang
//
// The two arms shared one host library and differed only in MORI_SOURCE_ROOT, so
// the JIT compiled a different kernel for each. That was checked rather than
// assumed -- an #error planted in the baseline tree fired when the root pointed
// there and did not fire when it pointed at the fixed tree. Without that check
// "both arms passed" cannot be told from "both arms ran the same code", which is
// how an earlier 8/8-vs-8/8 run reached no conclusion.
//
// Why eager never hit it, when the race does not need graphs: the window is the
// gap between a peer passing the epoch handshake and our clear landing, and
// eager's host round trip between kernels is wide enough that ranks rarely sit
// in it. Replay closes that gap and the ranks run shoulder to shoulder. So
// "eager passes" was never evidence against this race -- only evidence that the
// window was wide. Reading it as a contradiction cost a round of disbelief.
//
// A separate payload-visibility defect has not been established; adding a fence
// after WarpCopy alone would not fix a count that a later kernel overwrites with
// zero.
//
// Measure the RATE before believing any claim about this. Several conclusions
// here were drawn from single runs and were wrong; at a 7-in-8 hit rate a
// passing run means nothing, and on a shared box an interfering job can hide the
// failure entirely -- a 55-run matrix on a busy node saw 0 hangs where this
// 16-run matrix on an idle one saw 7.
//
// test_asyncll.py drives eager and the sdma transport, which is why the gfx1250
// enablement passed without touching this.

namespace asyncll {

// Splits one peer's byte range across the warps assigned to it. Chunks are
// 16 B-aligned so every warp's slice starts on a vector boundary; WarpCopy is
// what actually moves them.
struct LsaShare {
  int destPe;
  size_t offset;
  size_t bytes;
};

__device__ __forceinline__ LsaShare LsaShareFor(int globalWarpId, int globalWarpNum, int npes,
                                                size_t totalBytes) {
  const int warpsPerPe = globalWarpNum / npes;
  LsaShare share{globalWarpId / (warpsPerPe > 0 ? warpsPerPe : 1), 0, 0};
  if (warpsPerPe <= 0 || share.destPe >= npes) {
    share.destPe = -1;
    return share;
  }
  const int chunkId = globalWarpId % warpsPerPe;
  const size_t perWarp =
      ((totalBytes + warpsPerPe - 1) / warpsPerPe + 15) & ~static_cast<size_t>(15);
  share.offset = static_cast<size_t>(chunkId) * perWarp;
  share.bytes = (share.offset < totalBytes) ? min(perWarp, totalBytes - share.offset) : 0;
  return share;
}

}  // namespace asyncll

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  // Publish this rank's counts into the local staging word, as the SDMA variant
  // does. RecvTransfer is still what hands them to the peers.
  if (globalWarpId == 0 && laneId < npes) {
    uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    *reinterpret_cast<uint64_t*>(sendCount + asyncll::SendCountSlot(laneId)) =
        static_cast<uint64_t>(asyncll::SendTokenCount(
            config, core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId))) +
        1;
  }

  // Every warp in the grid carries bytes, rather than one warp per peer: a peer
  // store has no engine to hand the work to, so the copy is only as wide as the
  // grid made it.
  const int warpsPerPe = globalWarpNum / npes;
  if (warpsPerPe <= 0) return;
  const int destPe = globalWarpId / warpsPerPe;
  if (destPe >= npes || destPe == myPe) return;
  // Single-node by construction: a peer store needs the peer inside this rank's
  // flat LSA window, which only same-node ranks are. The host gate refuses
  // async_ll_transport="lsa" on a multi-node config, so this is unreachable
  // rather than a silent drop -- kept as a guard, not as a fallback.
  if (destPe / config.gpuPerNode != myNode) return;

  const int tokenNum =
      asyncll::SendTokenCount(config, core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe));
  if (tokenNum <= 0) return;

  const asyncll::LsaShare share = asyncll::LsaShareFor(globalWarpId, globalWarpNum, npes,
                                                       static_cast<size_t>(tokenNum) * xferBytes);
  if (share.bytes == 0) return;

  uint8_t* dst = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>(destPe) +
                 static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes + share.offset;
  const uint8_t* src = args.reg(args.offStaging)->template GetAs<const uint8_t*>() +
                       static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes +
                       share.offset;
  core::WarpCopy<uint8_t, 4>(dst, src, share.bytes);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  // No drain to do: the payload stores were issued by the PREVIOUS kernel, and a
  // kernel boundary already made them visible system-wide. That is the same
  // argument the SDMA variant uses for its count word, applied to the payload --
  // and it is why this variant needs no quiet at all.
  if (globalWarpId == 0 && laneId < npes) {
    const int destPe = laneId;
    const uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<const uint8_t*>();
    const uint64_t countValue =
        *reinterpret_cast<const uint64_t*>(sendCount + asyncll::SendCountSlot(destPe));
    const bool sameNode = (destPe / config.gpuPerNode == myNode);
    if (destPe == myPe || sameNode) {
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>(destPe);
      core::AtomicStoreRelaxedSystem(
          reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)), countValue);
    }
    // No cross-node arm, for the same reason the payload has none: this variant
    // is single-node and the host gate enforces it.
  }

  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();
  if (blockId == 0 && warpId == 0 && laneId < npes) {
    const uint64_t* slot =
        reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
    while (core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(slot)) == 0) {
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();

  // The counts are READ here and cleared in the next pass. They cannot be
  // cleared here: every warp in the grid sizes its copy from them, and a
  // grid-wide "all reads done" point does not exist inside one kernel --
  // __syncthreads is per block. The SDMA variant gets away with clearing in
  // place only because there the whole peer is one warp's business.
  const int warpsPerPe = globalWarpNum / npes;
  if (warpsPerPe <= 0) return;
  const int destPe = globalWarpId / warpsPerPe;
  if (destPe >= npes) return;

  // Biased by one, so 0 means "no dispatch announced this round". Test the raw
  // value rather than subtracting first: an unpaired combine finds 0 here and
  // `announced - 1` would be UINT64_MAX, which the old `tokenNum > 0` test waves
  // through and LsaShareFor then turns into a grid of peer stores running far
  // past the buffer. Same defect as the SDMA combine send; see its comment.
  const uint64_t announced = core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(
      reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe))));

  if (destPe != myPe && destPe / config.gpuPerNode == myNode && announced > 1) {
    const uint64_t tokenNum = announced - 1;
    const asyncll::LsaShare share =
        asyncll::LsaShareFor(globalWarpId, globalWarpNum, npes, tokenNum * hiddenBytes);
    if (share.bytes > 0) {
      uint8_t* dst = args.reg(args.offCombineInp)->template GetAs<uint8_t*>(destPe) +
                     static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes +
                     share.offset;
      const uint8_t* src = args.reg(args.offCombineStaging)->template GetAs<const uint8_t*>() +
                           static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * hiddenBytes +
                           share.offset;
      core::WarpCopy<uint8_t, 4>(dst, src, share.bytes);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  const uint64_t flag = args.crossDeviceBarrierFlag[0];

  // Same epoch handshake as the SDMA variant, with a peer store in place of the
  // engine. The payload stores landed in the previous kernel, so publishing the
  // epoch here is already ordered behind them.
  if (globalWarpId == 0 && laneId < npes) {
    const int destPe = laneId;
    if (destPe == myPe || destPe / config.gpuPerNode == myNode) {
      // All warps finished reading these counts in SendTransferLsa. Re-arm this
      // peer's slot BEFORE allowing it to start the next dispatch: if we clear
      // after the epoch wait, a faster peer can already have written its next
      // count and our late zero would erase it, leaving dispatch spinning forever.
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      core::AtomicStoreRelaxedSystem(
          reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe)), uint64_t{0});
      uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>(destPe);
      // The same lane clears the slot and notifies its writer. The peer's acquire
      // must observe this clear before its next dispatch can reuse the slot.
      core::AtomicStoreReleaseSystem(barrier + myPe, flag);
    }
  }

  if (blockId == 0 && warpId == 0 && laneId < npes) {
    uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    while (__hip_atomic_load(barrier + laneId, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) !=
           flag) {
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpCombineAsyncLLSendCopy -- combine pass 1                           */
/* ---------------------------------------------------------------------------------------------- */
// Put each post-expert token back into the staging slot it was dispatched from.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  if constexpr (asyncll::EarlyNotify<kConfig>) {
    if (globalWarpId == 0 && laneId < npes) {
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      uint64_t* count = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
      uint8_t* sendCounts = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
      // Dispatch recv drained its sends, so its old count source may be reused.
      // Biased by one: 0 means no dispatch was announced this round, and must
      // stay 0 tokens rather than wrap to UINT64_MAX, which SendTransfer would
      // multiply into a ~2^64-byte SDMA copy. Same guard as the legacy path's
      // `announced == 0`; the host call-order check keeps it unreachable.
      const uint64_t announced = core::AtomicLoadRelaxedSystem(count);
      *reinterpret_cast<uint64_t*>(sendCounts + asyncll::SendCountSlot(laneId)) =
          announced == 0 ? 0 : announced - 1;
      core::AtomicStoreRelaxedSystem(count, uint64_t{0});
      *reinterpret_cast<uint64_t*>(sendCounts + asyncll::SendFlagSlot(laneId)) =
          args.crossDeviceBarrierFlag[0];
      // This kernel must complete before SendTransfer publishes the epoch. In
      // particular the count clear must land before a peer starts its next round.
    }
  }
  const index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  // Combine's own staging, not dispatch's: writing dispatch's here would race
  // its transfer still reading it. See asyncll_regions.
  uint8_t* stagingPtr = args.reg(args.offCombineStaging)->template GetAs<uint8_t*>();

  if constexpr (asyncll::CombineQuant<kConfig>) {
    asyncll::CombineQuantSendCopy<kConfig, T>(args, stagingPtr, totalRecvTokenNum);
    return;
  }
  for (int tokenId = globalWarpId; tokenId < totalRecvTokenNum; tokenId += globalWarpNum) {
    index_t stagingTokId = 0;
    if (laneId == 0) stagingTokId = args.dispReceiverIdxMap[tokenId];
    stagingTokId = __shfl(stagingTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        stagingPtr + static_cast<size_t>(stagingTokId) * hiddenBytes,
        static_cast<const uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes, hiddenBytes);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpCombineAsyncLLSendTransfer -- combine pass 2                         */
/* ---------------------------------------------------------------------------------------------- */
// The dispatch transfer with the endpoints swapped: what arrived in dispatchInp
// goes back to combineInp, at the same slot arithmetic and the same stride. The
// payload is hidden only -- no indices, no source id -- so the stride is
// hiddenBytes rather than xferBytes (CombineRowBytes when the combine quantizes).
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendTransfer_body(EpDispatchCombineArgs args,
                                                  const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};
  // One combine row on the wire: hiddenBytes, or the quantized row.
  constexpr size_t rowBytes = asyncll::CombineRowBytes<kConfig, T>();

  uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;

    if constexpr (asyncll::Direct<kConfig>) {
      // The whole combine send. Dispatch's RecvCopy left each source's row count in
      // SendCountSlot; the rows sit contiguously in that order in the expert output. Re-arm
      // this peer's recv count and stage the epoch before the engine is handed either.
      const uint64_t flag = args.crossDeviceBarrierFlag[0];
      uint8_t* words = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
      uint64_t announced = 0;
      if (laneId == 0) {
        uint64_t* slot = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe));
        announced = core::AtomicLoadRelaxedSystem(slot);
        core::AtomicStoreRelaxedSystem(slot, uint64_t{0});
        *reinterpret_cast<uint64_t*>(words + asyncll::SendFlagSlot(destPe)) = flag;
        __threadfence_system();
      }
      announced = __shfl(static_cast<unsigned long long>(announced), 0);
      const uint64_t first = __shfl(
          static_cast<unsigned long long>(asyncll::CombineRowStart(words, npes, laneId)), destPe);
      // 0 is "no dispatch this round": send nothing but the epoch.
      const uint64_t tokenNum =
          (announced == 0)
              ? 0
              : *reinterpret_cast<const uint64_t*>(words + asyncll::SendCountSlot(destPe));
      if (destPe == myPe) {
        if (laneId == 0) {
          uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
          core::AtomicStoreReleaseSystem(barrier + myPe, flag);
        }
        continue;
      }
      const int peer = destPe - args.lsaBase;
      asyncll::PostPayloadAndNotifyPtr(
          comm, peer, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
          asyncll::RegionPeerPtr(
              args.reg(args.offCombineInp), peer,
              static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes),
          const_cast<uint8_t*>(static_cast<const uint8_t*>(args.inpTokenBuf) + first * hiddenBytes),
          static_cast<size_t>(tokenNum) * hiddenBytes,
          asyncll::RegionPeerPtr(args.reg(args.offCrossDeviceBarrier), peer,
                                 myPe * sizeof(uint64_t)),
          asyncll::RegionLocalPtr(args.reg(args.offSendCount), asyncll::SendFlagSlot(destPe)));
      continue;
    }

    // The staged protocols; a direct build has no combine staging (offCombineStaging is 0).
    if constexpr (!asyncll::Direct<kConfig>) {
      if constexpr (asyncll::EarlyNotify<kConfig>) {
        const uint8_t* sendCounts = args.reg(args.offSendCount)->template GetAs<const uint8_t*>();
        uint64_t tokenNum = 0;
        if (laneId == 0)
          tokenNum =
              *reinterpret_cast<const uint64_t*>(sendCounts + asyncll::SendCountSlot(destPe));
        tokenNum = __shfl(static_cast<unsigned long long>(tokenNum), 0);
        if (destPe == myPe) {
          if (laneId == 0) {
            uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
            const uint64_t flag =
                *reinterpret_cast<const uint64_t*>(sendCounts + asyncll::SendFlagSlot(destPe));
            core::AtomicStoreReleaseSystem(barrier + myPe, flag);
          }
        } else {
          const size_t remoteBase =
              static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * rowBytes;
          const size_t localBase =
              static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * rowBytes;
          asyncll::PostPayloadAndNotify(
              comm, destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
              EpInterNodeWindow(args.reg(args.offCombineInp)),
              EpInterNodeOffset(args.reg(args.offCombineInp), remoteBase),
              EpInterNodeWindow(args.reg(args.offCombineStaging)),
              EpInterNodeOffset(args.reg(args.offCombineStaging), localBase),
              static_cast<size_t>(tokenNum) * rowBytes,
              EpInterNodeWindow(args.reg(args.offCrossDeviceBarrier)),
              EpInterNodeOffset(args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t)),
              EpInterNodeWindow(args.reg(args.offSendCount)),
              EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendFlagSlot(destPe)));
        }
        continue;
      }

      // How many tokens this peer sent us during dispatch is how many we owe it
      // back. Dispatch left the slot alone precisely so this read works; clear it
      // now, which is also what re-arms the "0 means not arrived" test the next
      // dispatch's RecvTransfer spins on.
      uint64_t announced = 0;
      if (laneId == 0) {
        uint64_t* slot = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe));
        announced = core::AtomicLoadRelaxedSystem(slot);
        core::AtomicStoreRelaxedSystem(slot, uint64_t{0});
      }
      announced = __shfl(static_cast<unsigned long long>(announced), 0);

      // The count is biased by one, so 0 is "no dispatch announced this round"
      // rather than "zero tokens" -- and subtracting from it wraps. A combine with
      // no dispatch behind it (combine() twice on one routing, combine() before any
      // dispatch, combine_recv() after a whole combine) lands here with 0, and the
      // `tokenNum == 0` test below cannot catch UINT64_MAX: sdma.put would then be
      // handed a ~2^64-byte copy. Reproduced on 8xMI308X as a PERMISSION_FAULT on
      // every GPU in the job. Nothing this kernel can do makes the call correct, so
      // this only keeps a caller's mistake from becoming a memory fault; the host
      // raises on it first (see the call-order check in dispatch_combine_op.py).
      if (announced == 0) continue;
      const uint64_t tokenNum = announced - 1;

      if (destPe == myPe) continue;
      if (tokenNum == 0) continue;

      const size_t remoteBase = static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * rowBytes;
      const size_t localBase = static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * rowBytes;

      if (destPe / config.gpuPerNode != myNode) {
        // Striped over the QPs, as the dispatch send leg is and for the same
        // reason. RecvTransfer drains them.
        const int chunk = core::CeilDiv(static_cast<int>(tokenNum), config.numQpPerPe);
        for (int qpId = 0; qpId < config.numQpPerPe; ++qpId) {
          const int first = qpId * chunk;
          const int count = min(chunk, static_cast<int>(tokenNum) - first);
          if (count <= 0) break;
          EpInterNodePut(comm, args.reg(args.offCombineInp),
                         remoteBase + static_cast<size_t>(first) * rowBytes,
                         args.reg(args.offCombineStaging),
                         localBase + static_cast<size_t>(first) * rowBytes,
                         static_cast<size_t>(count) * rowBytes, destPe, qpId);
        }
        continue;
      }

      const int q = asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset);
      sdma.put<::mori::cco::ccoCoopWarp>(
          destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offCombineInp)),
          EpInterNodeOffset(args.reg(args.offCombineInp), remoteBase),
          EpInterNodeWindow(args.reg(args.offCombineStaging)),
          EpInterNodeOffset(args.reg(args.offCombineStaging), localBase), tokenNum * rowBytes, q);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpCombineAsyncLLRecvTransfer -- combine pass 3                         */
/* ---------------------------------------------------------------------------------------------- */
// Drain our sends, then an all-ranks handshake on a monotone epoch. The epoch
// comes from crossDeviceBarrierFlag, which dispatch's RecvCopy bumped, so it
// never repeats across launches and the slots need no clearing.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvTransfer_body(EpDispatchCombineArgs args,
                                                  const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};
  const uint64_t flag = args.crossDeviceBarrierFlag[0];

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;
    const bool sameNode = (destPe / config.gpuPerNode == myNode);
    // Warp-collective, above the laneId guard. See the note in dispatch's
    // RecvTransfer: from a single lane these deadlock.
    if (destPe != myPe) {
      if (sameNode)
        sdma.quietQueue<::mori::cco::ccoCoopWarp>(
            destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset));
      else
        for (int qpId = 0; qpId < config.numQpPerPe; ++qpId) EpInterNodeQuiet(comm, destPe, qpId);
    }

    if constexpr (asyncll::EarlyNotify<kConfig>) continue;
    if (laneId != 0) continue;

    uint8_t* sendFlags = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    core::AtomicStoreRelaxedSystem(
        reinterpret_cast<uint64_t*>(sendFlags + asyncll::SendFlagSlot(destPe)), flag);
    // The copy engine reads this word through its own path, not through the
    // wavefront's cache hierarchy, and the store above is in THIS kernel rather
    // than a previous launch -- so without the fence the engine can read the slot
    // before the value lands and ship a stale epoch. The peer then waits on a
    // barrier value that never arrives: a hang, not a wrong answer. Dispatch's
    // count word needs no equivalent because it is written in SendTransfer and
    // sent from RecvTransfer, with a kernel boundary in between.
    __threadfence_system();

    if (destPe == myPe) {
      uint8_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint8_t*>();
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint64_t*>(barrier) + myPe, flag);
    } else if (sameNode) {
      sdma.put(destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offCrossDeviceBarrier)),
               EpInterNodeOffset(args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t)),
               EpInterNodeWindow(args.reg(args.offSendCount)),
               EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendFlagSlot(destPe)),
               sizeof(uint64_t), asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset));
    } else {
      // Cross-node publishes the epoch as an ADD OF ONE, which is the same value
      // the two branches above store. crossDeviceBarrierFlag is bumped once per
      // round by dispatch's RecvCopy, and this rank writes a given peer's slot
      // exactly once per round, so after round N the slot has been incremented N
      // times and holds N -- precisely the epoch the wait below compares against.
      // ccoGda offers only an add at an arbitrary window offset, and here that is
      // enough rather than a compromise.
      EpInterNodeAtomicAdd(comm, args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t),
                           1ull, destPe);
    }
  }

  if (blockId == 0 && warpId == 0 && laneId < npes) {
    uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    while (core::AtomicLoadRelaxedSystem(barrier + laneId) != flag) {
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpCombineAsyncLLRecvCopy -- combine pass 4                           */
/* ---------------------------------------------------------------------------------------------- */
// Reduce each local token's top-k copies back into one. The sources are found by
// decoding dispDestTokIdMap, the same map dispatch wrote, so a dropped or
// deduplicated slot decodes to pe == worldSize and contributes nothing.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  constexpr int kTopk = kConfig.numExpertPerToken;

  extern __shared__ char epAsyncLlShared[];
  T** srcPtrs = reinterpret_cast<T**>(epAsyncLlShared) + warpId * kTopk;

  if (args.curRankNumToken == 0) return;

  if constexpr (asyncll::CombineQuant<kConfig>) {
    asyncll::CombineQuantRecvCopy<kConfig, T>(
        args, args.reg(args.offCombineInp)->template GetAs<const uint8_t*>(),
        args.reg(args.offCombineStaging)->template GetAs<const uint8_t*>(),
        args.reg(args.offCombineOut)->template GetAs<T*>());
    return;
  }

  // Direct: own rows never left the expert output; they start where SendTransfer's
  // prefix put this rank's, and a slot is a row within them.
  size_t ownFirst = 0;
  if constexpr (asyncll::Direct<kConfig>) {
    ownFirst =
        __shfl(static_cast<unsigned long long>(asyncll::CombineRowStart(
                   args.reg(args.offSendCount)->template GetAs<const uint8_t*>(), npes, laneId)),
               myPe);
  }

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim,
                       asyncll::CombineGranularity<T, MORI_ASYNCLL_COMBINE_VEC>());

  for (int i = globalWarpId; i < args.curRankNumToken * mwIter.warpsPerItem; i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    if constexpr (MORI_ASYNCLL_COMBINE_VEC == 17) {
      tokenId = __builtin_amdgcn_readfirstlane(tokenId);
      hiddenDimOffset = asyncll::WaveUniformU64(hiddenDimOffset);
      hiddenDimSize = asyncll::WaveUniformU64(hiddenDimSize);
      const T* src[kTopk];
#pragma unroll
      for (int j = 0; j < kTopk; ++j) {
        const index_t destTokId = args.dispDestTokIdMap[tokenId * kTopk + j];
        const index_t destPe = PeFromSendBufSlotOffset(config, destTokId);
        if (destPe >= npes) {
          src[j] = nullptr;
        } else if (destPe != myPe) {
          src[j] = args.reg(args.offCombineInp)->template GetAs<T*>() +
                   static_cast<size_t>(destTokId) * hiddenDim + hiddenDimOffset;
        } else if constexpr (asyncll::Direct<kConfig>) {
          src[j] = static_cast<const T*>(args.inpTokenBuf) +
                   (ownFirst + SlotIdFromSendBufSlotOffset(config, destTokId)) * hiddenDim +
                   hiddenDimOffset;
        } else {
          src[j] = args.reg(args.offCombineStaging)->template GetAs<T*>() +
                   static_cast<size_t>(destTokId) * hiddenDim + hiddenDimOffset;
        }
      }
      asyncll::CombineReduceRegs<T, kTopk>(args.reg(args.offCombineOut)->template GetAs<T*>() +
                                               tokenId * hiddenDim + hiddenDimOffset,
                                           src, hiddenDimSize);
      continue;
    }
    for (int j = laneId; j < kTopk; j += warpSize) {
      const index_t destTokId = args.dispDestTokIdMap[tokenId * kTopk + j];
      const index_t destPe = PeFromSendBufSlotOffset(config, destTokId);

      if (destPe < npes) {
        // Own tokens were never transferred -- they are still in combine's
        // staging where SendCopy put them (direct: in the expert output).
        if (asyncll::Direct<kConfig> && destPe == myPe) {
          srcPtrs[j] = const_cast<T*>(static_cast<const T*>(args.inpTokenBuf)) +
                       (ownFirst + SlotIdFromSendBufSlotOffset(config, destTokId)) * hiddenDim +
                       hiddenDimOffset;
          continue;
        }
        T* base;
        if constexpr (asyncll::Direct<kConfig>)
          base = args.reg(args.offCombineInp)->template GetAs<T*>();
        else
          base = (destPe != myPe) ? args.reg(args.offCombineInp)->template GetAs<T*>()
                                  : args.reg(args.offCombineStaging)->template GetAs<T*>();
        srcPtrs[j] = base + static_cast<size_t>(destTokId) * hiddenDim + hiddenDimOffset;
      } else {
        srcPtrs[j] = nullptr;
      }
    }

    T* outPtr =
        args.reg(args.offCombineOut)->template GetAs<T*>() + tokenId * hiddenDim + hiddenDimOffset;
    asyncll::CombineReduce<T, kTopk, MORI_ASYNCLL_COMBINE_VEC>(outPtr, srcPtrs, hiddenDimSize);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*            Segmented receive -- the expert reads the landing segments in place                 */
/* ---------------------------------------------------------------------------------------------- */
// With stride S > 0 (SegStrideOf; chosen per call by the host: identical on every rank, at least
// every rank's token count, at most the capacity), source s's rows land at [s * S, s * S + n_s)
// of this rank's inter_dispatch_inp, one block per field (SegLayout), so the expert reads plain
// [worldSize * S, ...] arrays and dispatch has no RecvCopy. Rows past n_s still hold an older
// round's data; RecvFinalizeSeg points their experts at a non-local one with weight 0, which the
// expert's mask drops. The expert writes its output in the same layout to inter_dispatch_staging,
// so what goes back to source s is rows [s * S, s * S + n_s) there. CombineSendTransferSeg sends
// that in place and CombineRecvCopySeg reads the replies at [d * S + slot] of inter_combine_inp.
//
// The expert reads the landing region until combine, so a peer must not land its next round's
// rows there before this rank's expert is done. Today that holds through combine: a peer starts
// its next dispatch only after its combine receive, which waits for this rank's epoch (posted by
// CombineSendTransfer[Seg] with early notify, or by CombineRecvTransfer's barrier otherwise),
// and both are issued on this rank's stream after the expert. A combine change that lets a
// peer run ahead of that point must keep this ordering for the segmented layout.
//
// dispDestTokIdMap still encodes SendBufSlotOffset with the capacity stride (and its null
// sentinel); only the row a slot lives in moves. Peers' rows are staged at that encoded slot in
// inter_staging, laid out per field as well (SegStagingLayoutOf: no hidden block under direct
// sources); this rank's own rows go straight into its landing segment. Who writes the rows and
// nulls the slots past the stride:
//   default: SendCopySeg writes every field.
//   direct:  SendCopySeg writes every field but hidden, which SendTransferSeg posts from the input.
//   direct + MORI_ASYNCLL_FUSE_META: SlotAssign writes the index, weight and source-token fields
//            and nulls the slots; SendTransferSeg posts hidden and scale rows from the input and
//            stages the count and mode words; SendCopySeg is a no-op.
namespace asyncll {

constexpr int kSegFields = 5;  // hidden, index, weight, scale, source token id

// One peer's segmented dispatch payload. Lane f < kSegFields posts field block f, lane
// kSegFields the stride word and, with early notify, lane kSegFields + 1 the count: one
// reservation on one queue, in lane order, so the count lands after everything it covers.
// Zero-byte lanes (an empty segment, no scales) drop out. A block too large for one packet
// takes the sequential path, for the reason PostPayloadAndNotify gives.
__device__ __forceinline__ void SegPostSegment(const ::mori::cco::ccoDevComm& comm, int peer,
                                               int queueId, ::mori::cco::ccoWindow_t win,
                                               size_t dstOffset, size_t srcOffset, size_t bytes) {
  ::mori::cco::ccoSdma sdma{comm};
  const bool big = __ballot(bytes > ::mori::cco::CCO_SDMA_MAX_COPY_BYTES) != 0;
  if (big) {
    for (int f = 0; f < kSegFields + 2; ++f) {
      const size_t n = __shfl(static_cast<unsigned long long>(bytes), f);
      const size_t dst = __shfl(static_cast<unsigned long long>(dstOffset), f);
      const size_t src = __shfl(static_cast<unsigned long long>(srcOffset), f);
      if (n != 0) sdma.put<::mori::cco::ccoCoopWarp>(peer, win, dst, win, src, n, queueId);
    }
  } else if (bytes != 0) {
    sdma.put<::mori::cco::ccoCoopThread, false, false, ::mori::cco::ccoSdmaOptFlagsDefault,
             ::mori::cco::ccoSdmaThreadSameQueue>(peer, win, dstOffset, win, srcOffset, bytes,
                                                  queueId);
  }
}

// SegPostSegment for MORI_ASYNCLL_INLINE_COUNT: lanes < kSegFields carry the field blocks, then
// the mode word and the count (the trigger) follow as immediates, in one reservation in order.
// `modeDst` / `countDst` are peer addresses as mapped here. Warp-collective.
__device__ __forceinline__ void SegPostSegmentInline(const ::mori::cco::ccoDevComm& comm, int peer,
                                                     int queueId, ::mori::cco::ccoWindow_t win,
                                                     size_t dstOffset, size_t srcOffset,
                                                     size_t bytes, void* modeDst, uint64_t mode,
                                                     void* countDst, uint64_t count) {
  ::mori::cco::ccoSdma sdma{comm};
  const int lane = threadIdx.x & (warpSize - 1);
  const bool big = __ballot(bytes > ::mori::cco::CCO_SDMA_MAX_COPY_BYTES) != 0;
  if (big) {
    for (int f = 0; f < kSegFields; ++f) {
      const size_t n = __shfl(static_cast<unsigned long long>(bytes), f);
      const size_t dst = __shfl(static_cast<unsigned long long>(dstOffset), f);
      const size_t src = __shfl(static_cast<unsigned long long>(srcOffset), f);
      if (n != 0) sdma.put<::mori::cco::ccoCoopWarp>(peer, win, dst, win, src, n, queueId);
    }
  }
  const unsigned long long copies = big ? 0ull : __ballot(lane < kSegFields && bytes != 0);
  const uint64_t copyBytes =
      static_cast<uint64_t>(__popcll(copies)) * ::mori::cco::CCO_SDMA_COPY_BYTES;
  const uint64_t total = copyBytes + kInlineWordBytes + kInlineTriggerBytes;
  ::mori::cco::ccoSdmaQueueDeviceHandle* qh = sdma.queueHandle(peer, queueId);
  uint32_t* ring = ::mori::cco::ccoSdma::ring(qh);
  uint64_t base = 0;
  if (lane == 0) base = ::mori::cco::ccoSdma::reserve(qh, total);
  base = __shfl(static_cast<unsigned long long>(base), 0);
  if ((copies >> lane) & 1ull)
    ::mori::cco::ccoSdma::placeCopy(
        ring,
        base + static_cast<uint64_t>(__popcll(copies & ((1ull << lane) - 1ull))) *
                   ::mori::cco::CCO_SDMA_COPY_BYTES,
        ::mori::cco::ccoGetLsaPeerPtr(win, peer, dstOffset),
        ::mori::cco::ccoGetLocalPtr(win, srcOffset), bytes);
  if (lane == 0) {
    PlaceInlineWord(ring, base + copyBytes, modeDst, mode);
    PlaceInlineTrigger(ring, base + copyBytes + kInlineWordBytes, countDst, count);
  }
  ::mori::cco::ccoSdmaPublishStores();  // every placer, as ccoSdmaPutThread's SameQueue batch
  __builtin_amdgcn_wave_barrier();
  if (lane == 0) ::mori::cco::ccoSdma::publish(qh, base, base + total);
}

}  // namespace asyncll

/* ------------------------------ EpDispatchAsyncLLSendCopySeg -----------------------------------
 */
// SendCopy with the per-field layout. Also publishes the stride word, and the counts clamped to
// the stride: entries past it are dropped (and nulled in dispDestTokIdMap so combine does not
// read a reply that never comes), which makes the round wrong, so it is flagged.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendCopySeg_body(EpDispatchCombineArgs args) {
  // Fused metadata: SlotAssign and SendTransferSeg do all of this (no double-counted overflow).
  if constexpr (asyncll::Direct<kConfig> && MORI_ASYNCLL_FUSE_META) return;
  DEF_COMMON_VARS;
  const int stride = asyncll::SegStrideOf(args, config);
  if (globalWarpId == 0 && laneId < npes) {
    const int raw = core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId);
    if (raw > stride) asyncll::SegCountError(args, config, asyncll::SegErrOverflow);
    uint8_t* words = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    if constexpr (asyncll::EarlyNotify<kConfig>) {
      *reinterpret_cast<uint64_t*>(words + asyncll::SendCountSlot(laneId)) =
          static_cast<uint64_t>(min(raw, stride)) + 1;
    }
    if (laneId == 0)
      *reinterpret_cast<uint64_t*>(words + asyncll::SegStrideSourceSlot(config)) =
          asyncll::SegModeWord(args.crossDeviceBarrierFlag[0], stride);
  }
  const index_t totalEntries = args.curRankNumToken * config.numExpertPerToken;
  if (totalEntries <= 0) return;

  if constexpr (asyncll::Direct<kConfig>) {
    // Fields only (SendTransferSeg posts every hidden row, own ones on the loopback queue),
    // one group of lanes per token as in SendCopy.
    asyncll::DirectWriteMetaAuto<kConfig>(args, config, hiddenBytes, xferBytes, stride);
    return;
  }

  const index_t warpsPerToken = (globalWarpNum + totalEntries - 1) / totalEntries;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);
  const asyncll::SegLayout layout = asyncll::SegLayoutOf(config, hiddenBytes);
  uint8_t* staging = args.reg(args.offStaging)->template GetAs<uint8_t*>();
  uint8_t* landing = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < totalEntries * warpsPerToken; i += globalWarpNum) {
    const index_t entryId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;

    const index_t destTokOffset = args.dispDestTokIdMap[entryId];
    if (destTokOffset >= NullSendBufSlotOffset(config)) continue;
    const int destPe = PeFromSendBufSlotOffset(config, destTokOffset);
    const int slotId = SlotIdFromSendBufSlotOffset(config, destTokOffset);
    if (slotId >= stride) {
      if (inTokenPartId == 0 && laneId == 0)
        args.dispDestTokIdMap[entryId] = NullSendBufSlotOffset(config);
      continue;
    }

    const bool self = (destPe == myPe);
    uint8_t* base = self ? landing : staging;
    const size_t row =
        self ? static_cast<size_t>(myPe) * stride + slotId : static_cast<size_t>(destTokOffset);
    const index_t srcTokId = entryId / config.numExpertPerToken;

    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 1>(
          base + layout.hidden + row * hiddenBytes + hiddenOffset,
          static_cast<const uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes + hiddenOffset,
          len);
    }

    if (inTokenPartId == 0) {
      asyncll::LaneCopyWords(
          base + layout.index + row * indexBytes,
          reinterpret_cast<const uint8_t*>(args.tokenIndices) + srcTokId * indexBytes, indexBytes);
      if (args.weightsBuf)
        asyncll::LaneCopyWords(
            base + layout.weight + row * weightBytes,
            reinterpret_cast<const uint8_t*>(args.weightsBuf) + srcTokId * weightBytes,
            weightBytes);
      if (args.scalesBuf && (scaleBytes > 0))
        asyncll::LaneCopyWords(base + layout.scale + row * scaleBytes,
                               args.scalesBuf + srcTokId * scaleBytes, scaleBytes);
      if (laneId == 0)
        reinterpret_cast<index_t*>(base + layout.srcTok)[row] =
            static_cast<index_t>(asyncll::SrcTokenIndex(config, myPe, srcTokId));
    }
  }
}

/* ---------------------------- EpDispatchAsyncLLSendTransferSeg ---------------------------------
 */
// SendTransfer with one packet per field block (SegPostSegment) into the peer's landing at
// [myPe * S]. Same counts and queue as the original: with early notify the count rides the same
// reservation, otherwise it is staged here and RecvTransfer sends it after its drain. Own rows
// are already in place, except direct hidden rows, which go on the loopback queue.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendTransferSeg_body(EpDispatchCombineArgs args,
                                                      const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  const int stride = asyncll::SegStrideOf(args, config);
  const asyncll::SegLayout layout = asyncll::SegLayoutOf(config, hiddenBytes);
  uint8_t* words = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
  // The count and mode word go out as packet immediates: nothing to stage or fence.
  constexpr bool kInline =
      asyncll::Direct<kConfig> && MORI_ASYNCLL_FUSE_META && asyncll::InlineCount<kConfig>;

  // Direct sources: blockNum / npes blocks share each peer's rows (the send ceiling sets the
  // grid); the one that reserves last posts the peer's tail. Otherwise one block per peer.
  constexpr bool kShared = asyncll::Direct<kConfig> && MORI_ASYNCLL_BATCH_POST;
  const int parts = kShared ? max(1, blockNum / npes) : 1;
  for (int unit = blockId; unit < npes * parts; unit += blockNum) {
    const int destPe = unit % npes;
    const int part = unit / npes;
    const int raw = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
    const size_t tokenNum = static_cast<size_t>(min(raw, stride));
    // This block reserved the peer's last rows, so it posts what follows them.
    bool postTail = true;
    if constexpr (asyncll::Direct<kConfig> && MORI_ASYNCLL_FUSE_META) {
      // No SendCopySeg: stage this peer's count and the mode word here, written back to memory
      // before any packet that carries them.
      if (part == 0 && warpId == 0 && laneId == 0) {
        if (raw > stride) asyncll::SegCountError(args, config, asyncll::SegErrOverflow);
        if constexpr (!kInline) {
          *reinterpret_cast<uint64_t*>(words + asyncll::SendCountSlot(destPe)) =
              static_cast<uint64_t>(tokenNum) + 1;
          *reinterpret_cast<uint64_t*>(words + asyncll::SegStrideSourceSlot(config)) =
              asyncll::SegModeWord(args.crossDeviceBarrierFlag[0], stride);
          __threadfence_system();
        }
      }
    }
    if constexpr (asyncll::Direct<kConfig>) {
      // The rows straight from the input into the peer's segment, own rows on the loopback
      // queue (RecvTransfer drains it); SendCopySeg, or SlotAssign with fused metadata, wrote
      // the other fields.
      if (destPe / config.gpuPerNode == myNode) {
        const EpInterNodeRegion landing = args.reg(args.offDispatchInp);
        const uint8_t* srcTokAt =
            (destPe == myPe)
                ? landing->template GetAs<const uint8_t*>() + layout.srcTok +
                      static_cast<size_t>(myPe) * stride * srcTokenIdBytes
                : args.reg(args.offStaging)->template GetAs<const uint8_t*>() +
                      asyncll::SegStagingLayoutOf<kConfig>(config, hiddenBytes).srcTok +
                      static_cast<size_t>(destPe) * config.MaxNumTokensToSendPerRank() *
                          srcTokenIdBytes;
        // With fused metadata the scale rows ride the same runs instead of the staging.
        const bool scaleRuns = MORI_ASYNCLL_FUSE_META && args.scalesBuf && scaleBytes > 0;
        auto post = [&](auto split) {
          return asyncll::DirectPostRows<kConfig, decltype(split)::value>(
              args, comm, config, destPe - args.lsaBase,
              asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
              static_cast<uint8_t*>(asyncll::RegionPeerPtr(
                  landing, destPe - args.lsaBase,
                  layout.hidden + static_cast<size_t>(myPe) * stride * hiddenBytes)),
              srcTokAt, srcTokenIdBytes, static_cast<int>(tokenNum), hiddenBytes,
              scaleRuns ? static_cast<uint8_t*>(asyncll::RegionPeerPtr(
                              landing, destPe - args.lsaBase,
                              layout.scale + static_cast<size_t>(myPe) * stride * scaleBytes))
                        : nullptr,
              scaleRuns ? scaleBytes : 0, part, parts, args.blockFlagCounter + destPe);
        };
        postTail = parts > 1 ? post(std::true_type{}) : post(std::false_type{});
      }
      __syncthreads();
    }
    if (warpId != 0) continue;

    if (!postTail) continue;
    if (destPe == myPe) {
      if (laneId == 0) {
        if constexpr (kInline) {
          // Own rows are on the loopback queue: the count follows them there.
          asyncll::PostInlineTrigger(
              comm, myPe - args.lsaBase, asyncll::QueueForPe(comm, myPe, args.sdmaQueueOffset),
              asyncll::RegionPeerPtr(args.reg(args.offRecvTokenNum), myPe - args.lsaBase,
                                     asyncll::RecvCountSlot(myPe)),
              static_cast<uint64_t>(tokenNum) + 1);
        } else if constexpr (asyncll::Direct<kConfig>) {
          // Own rows are on the loopback queue: the count follows them there.
          ::mori::cco::ccoSdma sdma{comm};
          sdma.put(myPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offRecvTokenNum)),
                   EpInterNodeOffset(args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe)),
                   EpInterNodeWindow(args.reg(args.offSendCount)),
                   EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendCountSlot(myPe)),
                   sizeof(uint64_t), asyncll::QueueForPe(comm, myPe, args.sdmaQueueOffset));
        } else if constexpr (asyncll::EarlyNotify<kConfig>) {
          uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
          core::AtomicStoreReleaseSystem(
              reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)),
              static_cast<uint64_t>(tokenNum) + 1);
        } else {
          *reinterpret_cast<uint64_t*>(words + asyncll::SendCountSlot(myPe)) =
              static_cast<uint64_t>(tokenNum) + 1;
        }
      }
      continue;
    }
    // Single node only; the host refuses the segmented layout otherwise.
    if (destPe / config.gpuPerNode != myNode) continue;
    if constexpr (!asyncll::EarlyNotify<kConfig>) {
      if (laneId == 0)
        *reinterpret_cast<uint64_t*>(words + asyncll::SendCountSlot(destPe)) =
            static_cast<uint64_t>(tokenNum) + 1;
    }

    // This lane's packet; see SegPostSegment for the lane assignment.
    const EpInterNodeRegion landing = args.reg(args.offDispatchInp);
    const EpInterNodeRegion staging = args.reg(args.offStaging);
    const EpInterNodeRegion wordRegion = args.reg(args.offSendCount);
    size_t dst = 0, src = 0, bytes = 0;
    if (laneId < asyncll::kSegFields) {
      const size_t fieldOffsets[asyncll::kSegFields] = {layout.hidden, layout.index, layout.weight,
                                                        layout.scale, layout.srcTok};
      const size_t fieldRowBytes[asyncll::kSegFields] = {hiddenBytes, indexBytes, weightBytes,
                                                         scaleBytes, srcTokenIdBytes};
      const size_t field = fieldOffsets[laneId];
      const size_t fieldBytes = fieldRowBytes[laneId];
      dst = EpInterNodeOffset(landing, field + static_cast<size_t>(myPe) * stride * fieldBytes);
      // The peers' rows sit in the staging layout (no hidden block under direct sources).
      size_t stagingField = field;
      if constexpr (asyncll::Direct<kConfig>) {
        const asyncll::SegLayout stagingLayout =
            asyncll::SegStagingLayoutOf<kConfig>(config, hiddenBytes);
        const size_t stagingOffsets[asyncll::kSegFields] = {
            stagingLayout.hidden, stagingLayout.index, stagingLayout.weight, stagingLayout.scale,
            stagingLayout.srcTok};
        stagingField = stagingOffsets[laneId];
      }
      src = EpInterNodeOffset(staging, stagingField + static_cast<size_t>(destPe) *
                                                          config.MaxNumTokensToSendPerRank() *
                                                          fieldBytes);
      // Direct: the hidden block went row by row above, and with fused metadata the scales too.
      const bool posted =
          asyncll::Direct<kConfig> && (laneId == 0 || (MORI_ASYNCLL_FUSE_META && laneId == 3));
      bytes = posted ? 0 : tokenNum * fieldBytes;
    } else if (!kInline && laneId == asyncll::kSegFields) {
      dst = EpInterNodeOffset(wordRegion, asyncll::SegStrideSlot(config, myPe));
      src = EpInterNodeOffset(wordRegion, asyncll::SegStrideSourceSlot(config));
      bytes = sizeof(uint64_t);
    } else if (!kInline && laneId == asyncll::kSegFields + 1 && asyncll::EarlyNotify<kConfig>) {
      dst = EpInterNodeOffset(args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe));
      src = EpInterNodeOffset(wordRegion, asyncll::SendCountSlot(destPe));
      bytes = sizeof(uint64_t);
    }
    if constexpr (kInline) {
      // Same slots, values and queue order: fields, then the mode word, then the count.
      const int peer = destPe - args.lsaBase;
      asyncll::SegPostSegmentInline(
          comm, peer, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
          EpInterNodeWindow(landing), dst, src, bytes,
          asyncll::RegionPeerPtr(wordRegion, peer, asyncll::SegStrideSlot(config, myPe)),
          asyncll::SegModeWord(args.crossDeviceBarrierFlag[0], stride),
          asyncll::RegionPeerPtr(args.reg(args.offRecvTokenNum), peer,
                                 asyncll::RecvCountSlot(myPe)),
          static_cast<uint64_t>(tokenNum) + 1);
    } else {
      asyncll::SegPostSegment(comm, destPe - args.lsaBase,
                              asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
                              EpInterNodeWindow(landing), dst, src, bytes);
    }
  }
}

#if MORI_ASYNCLL_RECV_TRIM
// RecvFinalizeSeg below with a shorter load chain. kWait: wave 0 polls the counts and stages them.
// Block 0 always reads the counts, so the kernel still ends after every one; rowless blocks exit.
template <EpInterNodeKernelCfg kConfig, typename T, bool kWait, typename Drain>
__device__ __forceinline__ void EpDispatchAsyncLLRecvFinalizeSegTrim_body(
    EpDispatchCombineArgs args, Drain drain) {
  drain(false);
  DEF_COMMON_VARS;
  static_assert(kConfig.worldSize <= 64, "the segment counts are staged in a 64-entry LDS array");
  uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
  uint8_t* words = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
  const asyncll::SegLayout layout = asyncll::SegLayoutOf(config, hiddenBytes);
  uint8_t* landing = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();
  index_t* indices = reinterpret_cast<index_t*>(landing + layout.index);
  float* weights = reinterpret_cast<float*>(landing + layout.weight);
  uint32_t* scales = reinterpret_cast<uint32_t*>(landing + layout.scale);
  // A whole number of dwords per scale row: the host refuses the layout otherwise.
  const size_t scaleWords = scaleBytes / 4;
  const index_t* srcTok = reinterpret_cast<const index_t*>(landing + layout.srcTok);
  index_t* reverse = args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>();
  // The next rank's first expert: this rank's expert mask drops it.
  const index_t invalidExpert = ((myPe + 1) % npes) * config.numExpertPerRank;
  drain(true);
  const int stride = asyncll::SegStrideOf(args, config);
  const int rows = npes * stride;
  if (blockId != 0 && blockId * thdNum >= rows) return;
  const bool tail = globalWarpId == 0;

  __shared__ int segRows[64];
  if (warpId == 0) {
    uint64_t announced = 0;
    if (laneId < npes) {
      uint64_t* slot = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
      if constexpr (kWait) {
        while ((announced = __hip_atomic_load(slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) ==
               0)
          __builtin_amdgcn_s_sleep(1);
      } else {
        announced = *slot;
      }
    }
    // Orders the block's later reads after the counts; the barrier hands it on.
    if constexpr (kWait) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // Biased by one; 0 (no count) reads as an empty segment rather than wrapping to "full".
    if (laneId < npes)
      segRows[laneId] = (announced == 0)
                            ? 0
                            : static_cast<int>(min(announced - 1, static_cast<uint64_t>(stride)));
  }
  __syncthreads();

  // Low half of the epoch before the tail's bump: all SegModeWord keeps, no dead half to wait on.
  const uint32_t epoch = __hip_atomic_load(reinterpret_cast<uint32_t*>(args.crossDeviceBarrierFlag),
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // Each peer's mode word landed before its count; loaded now to overlap the row pass.
  uint64_t mode = 0;
  if (tail && laneId < npes && laneId != myPe)
    mode = core::AtomicLoadRelaxedSystem(
        reinterpret_cast<uint64_t*>(words + asyncll::SegStrideSlot(config, laneId)));

  // Holes get an expert id this rank masks off, weight 0 and zeroed scales.
  const int liveThreads = min(blockNum, max(1, (rows + thdNum - 1) / thdNum)) * thdNum;
  if (rows <= liveThreads) {
    // One row per thread: its own stores, issued as soon as its count is known.
    const int r = globalThdId;
    if (r < rows) {
      const int srcPe = r / stride;
      if (r - srcPe * stride < segRows[srcPe]) {
        reverse[r] = srcTok[r];
      } else {
        reverse[r] = -1;
        for (int k = 0; k < config.numExpertPerToken; ++k) {
          indices[static_cast<size_t>(r) * config.numExpertPerToken + k] = invalidExpert;
          weights[static_cast<size_t>(r) * config.numExpertPerToken + k] = 0.0f;
        }
      }
    }
  } else {
    // Several rows per thread. The srcTok loads of kRowBatch rows go out together, and each
    // segment's holes are one contiguous range of words, written by adjacent lanes: per-row
    // stores would cost a memory transaction per word in this window.
    constexpr int kRowBatch = 8;
    for (int base = globalThdId; base < rows; base += liveThreads * kRowBatch) {
      index_t tok[kRowBatch];
#pragma unroll
      for (int u = 0; u < kRowBatch; ++u) {
        const int r = base + u * liveThreads;
        const int srcPe = r < rows ? r / stride : 0;
        tok[u] = (r < rows && r - srcPe * stride < segRows[srcPe]) ? srcTok[r] : index_t{-1};
      }
#pragma unroll
      for (int u = 0; u < kRowBatch; ++u) {
        const int r = base + u * liveThreads;
        if (r < rows) reverse[r] = tok[u];
      }
    }
    const size_t topk = config.numExpertPerToken;
    for (int s = 0; s < npes; ++s) {
      const size_t holeBegin = static_cast<size_t>(s) * stride + segRows[s];
      const size_t holeEnd = static_cast<size_t>(s + 1) * stride;
      for (size_t e = holeBegin * topk + globalThdId; e < holeEnd * topk; e += liveThreads) {
        indices[e] = invalidExpert;
        weights[e] = 0.0f;
      }
    }
  }
  // Scale rows are tens of words: contiguous per segment on either path.
  if (scaleWords > 0) {
    for (int s = 0; s < npes; ++s) {
      const size_t holeBegin = static_cast<size_t>(s) * stride + segRows[s];
      const size_t holeEnd = static_cast<size_t>(s + 1) * stride;
      for (size_t e = holeBegin * scaleWords + globalThdId; e < holeEnd * scaleWords;
           e += liveThreads)
        scales[e] = 0;
    }
  }

  if (tail) {
    if (laneId < npes) {
      if (laneId != myPe && mode != asyncll::SegModeWord(epoch, stride))
        asyncll::SegCountError(args, config, asyncll::SegErrMismatch);
      *reinterpret_cast<uint64_t*>(words + asyncll::SegCountSlot(config, laneId)) =
          static_cast<uint64_t>(segRows[laneId]);
      args.destPeTokenCounter[laneId] = 0;
    }
    if (laneId == 0) {
      args.totalRecvTokenNum[0] = static_cast<index_t>(rows);
      args.dispatchGridBarrier[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1ull);
    }
  }
}
#endif

/* ---------------------------- EpDispatchAsyncLLRecvFinalizeSeg ---------------------------------
 */
// Replaces RecvCopy: nothing is copied. Points every row past its segment's count (a hole) at a
// non-local expert with weight 0 and zeroes its scales, records the counts for combine and for
// segment_counts(), checks every peer's mode word, and runs RecvCopy's epilogue with
// total_recv = worldSize * S, the row count of the views. RecvTransfer has seen every count, and
// each count was queued behind its segment and mode word.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvFinalizeSeg_body(EpDispatchCombineArgs args) {
#if MORI_ASYNCLL_RECV_TRIM
  EpDispatchAsyncLLRecvFinalizeSegTrim_body<kConfig, T, false>(args, [](bool) {});
#else
  DEF_COMMON_VARS;
  static_assert(kConfig.worldSize <= 64, "the segment counts are staged in a 64-entry LDS array");
  const int stride = asyncll::SegStrideOf(args, config);
  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();

  __shared__ int segRows[64];
  if (threadIdx.x < npes) {
    const uint64_t announced =
        *reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(threadIdx.x));
    // Biased by one; RecvTransfer waited for every count, so 0 should not reach here, but if
    // it does, treat the segment as empty rather than letting announced - 1 wrap to "full".
    segRows[threadIdx.x] =
        (announced == 0) ? 0 : static_cast<int>(min(announced - 1, static_cast<uint64_t>(stride)));
  }
  __syncthreads();

  const asyncll::SegLayout layout = asyncll::SegLayoutOf(config, hiddenBytes);
  uint8_t* landing = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();
  index_t* indices = reinterpret_cast<index_t*>(landing + layout.index);
  float* weights = reinterpret_cast<float*>(landing + layout.weight);
  uint32_t* scales = reinterpret_cast<uint32_t*>(landing + layout.scale);
  // A whole number of dwords per scale row: the host refuses the layout otherwise.
  const size_t scaleWords = scaleBytes / 4;
  const index_t* srcTok = reinterpret_cast<const index_t*>(landing + layout.srcTok);
  index_t* reverse = args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>();
  const int rows = npes * stride;
  // The next rank's first expert: this rank's expert mask drops it.
  const index_t invalidExpert = ((myPe + 1) % npes) * config.numExpertPerRank;
  for (int r = globalThdId; r < rows; r += globalThdNum) {
    const int srcPe = r / stride;
    if (r - srcPe * stride < segRows[srcPe]) {
      reverse[r] = srcTok[r];
      continue;
    }
    reverse[r] = -1;
    for (int k = 0; k < config.numExpertPerToken; ++k) {
      indices[static_cast<size_t>(r) * config.numExpertPerToken + k] = invalidExpert;
      weights[static_cast<size_t>(r) * config.numExpertPerToken + k] = 0.0f;
    }
    // Stale scales could be anything (a NaN pattern included) to a consumer that touches every
    // row's scale even for a masked expert; the hidden bytes stay stale, they are 25-50x larger.
    for (size_t w = 0; w < scaleWords; ++w) scales[static_cast<size_t>(r) * scaleWords + w] = 0;
  }

  if (globalWarpId == 0) {
    int mine = 0;
    if (laneId < npes) {
      mine = segRows[laneId];
      // No re-arming: the epoch in the word already tells this round from the last.
      asyncll::SegCheckModeWords(args, config, laneId, myPe, stride);
      *reinterpret_cast<uint64_t*>(args.reg(args.offSendCount)->template GetAs<uint8_t*>() +
                                   asyncll::SegCountSlot(config, laneId)) =
          static_cast<uint64_t>(mine);
      args.destPeTokenCounter[laneId] = 0;
    }
    if (laneId == 0) {
      args.totalRecvTokenNum[0] = static_cast<index_t>(rows);
      args.dispatchGridBarrier[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1ull);
    }
  }
#endif
}

/* --------------------------- EpCombineAsyncLLSendTransferSeg -----------------------------------
 */
// CombineSendTransfer straight from the segmented expert output (inter_dispatch_staging, which
// peers never write) into the peer's inter_combine_inp at [myPe * S]. Without a SendCopy in front
// of it, it also does that pass's early-notify bookkeeping, per peer: read and clear the count,
// then stage the epoch and fence before the engine is handed either.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendTransferSeg_body(EpDispatchCombineArgs args,
                                                     const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};
  const int stride = asyncll::SegStrideOf(args, config);
  uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
  uint8_t* words = args.reg(args.offSendCount)->template GetAs<uint8_t*>();

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;
    uint64_t announced = 0;
    if (laneId == 0) {
      uint64_t* slot = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe));
      announced = core::AtomicLoadRelaxedSystem(slot);
      core::AtomicStoreRelaxedSystem(slot, uint64_t{0});
    }
    announced = __shfl(static_cast<unsigned long long>(announced), 0);
    // 0 is "no dispatch this round" (see the original); never more than the segment.
    const uint64_t tokenNum =
        (announced == 0) ? 0 : min(announced - 1, static_cast<uint64_t>(stride));
    const size_t remoteBase = static_cast<size_t>(myPe) * stride * hiddenBytes;
    const size_t localBase = static_cast<size_t>(destPe) * stride * hiddenBytes;

    if constexpr (asyncll::EarlyNotify<kConfig>) {
      const uint64_t flag = args.crossDeviceBarrierFlag[0];
      if (destPe == myPe) {
        if (laneId == 0) {
          uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
          core::AtomicStoreReleaseSystem(barrier + myPe, flag);
        }
        continue;
      }
      if constexpr (asyncll::InlineCount<kConfig>) {
        // The epoch rides the packet behind the rows. The clear above is a system-scope store
        // this wave drains (s_waitcnt) before the doorbell; a coarse-grained window also needs
        // it written back from L2.
#if MORI_ASYNCLL_INLINE_CLEAR_FENCE
        if (laneId == 0) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
#endif
        const int peer = destPe - args.lsaBase;
        asyncll::PostPayloadAndInlineNotify(
            comm, peer, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
            asyncll::RegionPeerPtr(args.reg(args.offCombineInp), peer, remoteBase),
            asyncll::RegionLocalPtr(args.reg(args.offDispatchStaging), localBase),
            static_cast<size_t>(tokenNum) * hiddenBytes,
            asyncll::RegionPeerPtr(args.reg(args.offCrossDeviceBarrier), peer,
                                   myPe * sizeof(uint64_t)),
            flag);
        continue;
      }
      if (laneId == 0) {
        *reinterpret_cast<uint64_t*>(words + asyncll::SendFlagSlot(destPe)) = flag;
        // The engine reads the epoch from memory, and the clear above must be visible before
        // the peer can see the epoch and start its next dispatch.
        __threadfence_system();
      }
      asyncll::PostPayloadAndNotify(
          comm, destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset),
          EpInterNodeWindow(args.reg(args.offCombineInp)),
          EpInterNodeOffset(args.reg(args.offCombineInp), remoteBase),
          EpInterNodeWindow(args.reg(args.offDispatchStaging)),
          EpInterNodeOffset(args.reg(args.offDispatchStaging), localBase),
          static_cast<size_t>(tokenNum) * hiddenBytes,
          EpInterNodeWindow(args.reg(args.offCrossDeviceBarrier)),
          EpInterNodeOffset(args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t)),
          EpInterNodeWindow(args.reg(args.offSendCount)),
          EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendFlagSlot(destPe)));
      continue;
    }

    if (announced == 0 || destPe == myPe || tokenNum == 0) continue;
    if (destPe / config.gpuPerNode != myNode) continue;
    sdma.put<::mori::cco::ccoCoopWarp>(
        destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offCombineInp)),
        EpInterNodeOffset(args.reg(args.offCombineInp), remoteBase),
        EpInterNodeWindow(args.reg(args.offDispatchStaging)),
        EpInterNodeOffset(args.reg(args.offDispatchStaging), localBase), tokenNum * hiddenBytes,
        asyncll::QueueForPe(comm, destPe, args.sdmaQueueOffset));
  }
}

/* ------------------------------ EpCombineAsyncLLRecvCopySeg ------------------------------------
 */
// CombineRecvCopy for CombineSendTransferSeg: a peer's reply to slot j is at [peer * S + j] of
// inter_combine_inp, and this rank's own never left its expert output, at [myPe * S + j].
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopySeg_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  constexpr int kTopk = kConfig.numExpertPerToken;

  extern __shared__ char epAsyncLlShared[];
  T** srcPtrs = reinterpret_cast<T**>(epAsyncLlShared) + warpId * kTopk;

  if (args.curRankNumToken == 0) return;
  const int stride = asyncll::SegStrideOf(args, config);
  T* peerBase = args.reg(args.offCombineInp)->template GetAs<T*>();
  T* selfBase = args.reg(args.offDispatchStaging)->template GetAs<T*>();

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim,
                       asyncll::CombineGranularity<T, MORI_ASYNCLL_COMBINE_VEC_SEG>());

  for (int i = globalWarpId; i < args.curRankNumToken * mwIter.warpsPerItem; i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    if constexpr (MORI_ASYNCLL_COMBINE_VEC_SEG == 17) {
      tokenId = __builtin_amdgcn_readfirstlane(tokenId);
      hiddenDimOffset = asyncll::WaveUniformU64(hiddenDimOffset);
      hiddenDimSize = asyncll::WaveUniformU64(hiddenDimSize);
      const T* src[kTopk];
#pragma unroll
      for (int j = 0; j < kTopk; ++j) {
        const index_t destTokId = args.dispDestTokIdMap[tokenId * kTopk + j];
        const index_t destPe = PeFromSendBufSlotOffset(config, destTokId);
        src[j] = (destPe < npes) ? ((destPe != myPe) ? peerBase : selfBase) +
                                       (static_cast<size_t>(destPe) * stride +
                                        SlotIdFromSendBufSlotOffset(config, destTokId)) *
                                           hiddenDim +
                                       hiddenDimOffset
                                 : nullptr;
      }
      asyncll::CombineReduceRegs<T, kTopk>(args.reg(args.offCombineOut)->template GetAs<T*>() +
                                               tokenId * hiddenDim + hiddenDimOffset,
                                           src, hiddenDimSize);
      continue;
    }
    for (int j = laneId; j < kTopk; j += warpSize) {
      const index_t destTokId = args.dispDestTokIdMap[tokenId * kTopk + j];
      const index_t destPe = PeFromSendBufSlotOffset(config, destTokId);

      if (destPe < npes) {
        const size_t row =
            static_cast<size_t>(destPe) * stride + SlotIdFromSendBufSlotOffset(config, destTokId);
        srcPtrs[j] = ((destPe != myPe) ? peerBase : selfBase) + row * hiddenDim + hiddenDimOffset;
      } else {
        srcPtrs[j] = nullptr;
      }
    }

    T* outPtr =
        args.reg(args.offCombineOut)->template GetAs<T*>() + tokenId * hiddenDim + hiddenDimOffset;
    asyncll::CombineReduce<T, kTopk, MORI_ASYNCLL_COMBINE_VEC_SEG>(outPtr, srcPtrs, hiddenDimSize);
  }
}

#if MORI_ASYNCLL_FUSED_RECV
// The control wave checks one outgoing queue per lane. Thread-scope quiet has
// no collective barrier and these queues only wait on independently progressing
// SDMA engines. Kernel completion waits for this wave, so sources remain owned
// until all outgoing copies finish, without making other blocks wait on this block.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void EpAsyncLLFusedQuiet(EpDispatchCombineArgs args,
                                                    const ::mori::cco::ccoDevComm& comm) {
  static_assert(asyncll::EarlyNotify<kConfig>,
                "fused AsyncLL receive requires early notifications on a single node");
  const int lane = threadIdx.x & (warpSize - 1);
  // Direct sources also queue this rank's own rows on its loopback queue (segmented layout).
  if (blockIdx.x == 0 && threadIdx.x < warpSize && lane < kConfig.worldSize &&
      (lane != args.rank || asyncll::Direct<kConfig>)) {
    ::mori::cco::ccoSdma sdma{comm};
    sdma.quietQueue<::mori::cco::ccoCoopThread>(
        lane - args.lsaBase, asyncll::QueueForPe(comm, lane, args.sdmaQueueOffset));
  }
}

#if MORI_ASYNCLL_RECV_TRIM
// EpAsyncLLFusedQuiet's queues, drained from wave 1 of the last block beside its poll; that block
// has rows only when a round fills the whole grid. Called first, before any block-size load;
// kSolo: a one-wave block drains from wave 0.
template <EpInterNodeKernelCfg kConfig, bool kSolo>
__device__ __forceinline__ void EpAsyncLLSegDrain(const EpDispatchCombineArgs& args,
                                                  const ::mori::cco::ccoDevComm& comm) {
  static_assert(asyncll::EarlyNotify<kConfig>,
                "fused AsyncLL receive requires early notifications on a single node");
  if (blockIdx.x != gridDim.x - 1) return;
  if constexpr (kSolo) {
    if (threadIdx.x >= warpSize || blockDim.x > warpSize) return;
  } else {
    if (threadIdx.x / warpSize != 1) return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  if (lane >= kConfig.worldSize || (lane == args.rank && !asyncll::Direct<kConfig>)) return;
  ::mori::cco::ccoSdma sdma{comm};
  sdma.quietQueue<::mori::cco::ccoCoopThread>(
      lane - args.lsaBase, asyncll::QueueForPe(comm, lane, args.sdmaQueueOffset));
}

// CombineRecvCopySeg fused, with a shorter chain. Block 0 always waits, so the kernel still ends
// after every epoch; blocks with no work exit.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopySegTrim_body(EpDispatchCombineArgs args,
                                                     const ::mori::cco::ccoDevComm& comm) {
  EpAsyncLLSegDrain<kConfig, false>(args, comm);
  DEF_COMMON_VARS;
  constexpr int kTopk = kConfig.numExpertPerToken;
  constexpr bool kRegs = MORI_ASYNCLL_COMBINE_VEC_SEG == 17;

  extern __shared__ char epAsyncLlShared[];
  T** srcPtrs = reinterpret_cast<T**>(epAsyncLlShared) + warpId * kTopk;

  const uint64_t flag = args.crossDeviceBarrierFlag[0];
  const int stride = asyncll::SegStrideOf(args, config);
  T* peerBase = args.reg(args.offCombineInp)->template GetAs<T*>();
  T* selfBase = args.reg(args.offDispatchStaging)->template GetAs<T*>();
  T* outBase = args.reg(args.offCombineOut)->template GetAs<T*>();
  uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
  EpAsyncLLSegDrain<kConfig, true>(args, comm);
  // MultiWarpIter's split in 32-bit arithmetic (its size_t divisions cost ~400 instructions).
  const int tokens = args.curRankNumToken;
  const int dim = static_cast<int>(hiddenDim);
  constexpr int kGran =
      static_cast<int>(asyncll::CombineGranularity<T, MORI_ASYNCLL_COMBINE_VEC_SEG>());
  int perItem = (globalWarpNum + (tokens > 0 ? tokens : 1) - 1) / (tokens > 0 ? tokens : 1);
  int dimPerWarp = (dim + perItem - 1) / perItem;
  if constexpr (kGran > 1) {
    dimPerWarp = (dimPerWarp + kGran - 1) / kGran * kGran;
    perItem = (dim + dimPerWarp - 1) / dimPerWarp;
  }
  const int items = tokens > 0 ? tokens * perItem : 0;
  if (blockId != 0 && blockId * warpNum >= items) return;

  // A work item: token, hidden slice, and map entries (all kTopk with kRegs, else lane j's).
  struct Item {
    int tokenId;
    size_t dimOffset, dimSize;
    index_t ids[kRegs ? kTopk : 1];
  };
  auto load = [&](int i, Item& it) {
    it.tokenId = i / perItem;
    const int offset = (i - it.tokenId * perItem) * dimPerWarp;
    it.dimOffset = static_cast<size_t>(offset);
    it.dimSize = static_cast<size_t>(offset < dim ? min(dim - offset, dimPerWarp) : 0);
    if constexpr (kRegs) {
      it.tokenId = __builtin_amdgcn_readfirstlane(it.tokenId);
      it.dimOffset = asyncll::WaveUniformU64(it.dimOffset);
      it.dimSize = asyncll::WaveUniformU64(it.dimSize);
#pragma unroll
      for (int j = 0; j < kTopk; ++j) it.ids[j] = args.dispDestTokIdMap[it.tokenId * kTopk + j];
    } else if (laneId < kTopk) {
      it.ids[0] = args.dispDestTokIdMap[it.tokenId * kTopk + laneId];
    }
  };
  auto row = [&](const Item& it, index_t destTokId) -> T* {
    const index_t destPe = PeFromSendBufSlotOffset(config, destTokId);
    if (destPe >= npes) return nullptr;
    return ((destPe != myPe) ? peerBase : selfBase) +
           (static_cast<size_t>(destPe) * stride + SlotIdFromSendBufSlotOffset(config, destTokId)) *
               hiddenDim +
           it.dimOffset;
  };
  auto reduce = [&](const Item& it) {
    T* outPtr = outBase + it.tokenId * hiddenDim + it.dimOffset;
    if constexpr (kRegs) {
      const T* src[kTopk];
#pragma unroll
      for (int j = 0; j < kTopk; ++j) src[j] = row(it, it.ids[j]);
      // Uniform again: the conditional reload hides it, and a per-lane size waterfalls each load.
      asyncll::CombineReduceRegs<T, kTopk>(outPtr, src, asyncll::WaveUniformU64(it.dimSize));
    } else {
      if (laneId < kTopk) srcPtrs[laneId] = row(it, it.ids[0]);
      asyncll::CombineReduce<T, kTopk, MORI_ASYNCLL_COMBINE_VEC_SEG>(outPtr, srcPtrs, it.dimSize);
    }
  };

  // The first item's map entries load beside the poll: this rank's dispatch wrote them.
  Item item{};
  if (globalWarpId < items) load(globalWarpId, item);
  if (warpId == 0) {
    if (laneId < npes) {
      while (__hip_atomic_load(barrier + laneId, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) !=
             flag)
        __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  __syncthreads();
  for (int i = globalWarpId; i < items; i += globalWarpNum) {
    if (i != globalWarpId) load(i, item);
    reduce(item);
  }
}
#endif

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvCopyFused_body(EpDispatchCombineArgs args,
                                                    const ::mori::cco::ccoDevComm& comm) {
  EpAsyncLLFusedQuiet<kConfig>(args, comm);
  const int lane = threadIdx.x & (warpSize - 1);
  if (threadIdx.x < warpSize) {
    if (lane < kConfig.worldSize) {
      uint8_t* counts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      uint64_t* slot = reinterpret_cast<uint64_t*>(counts + asyncll::RecvCountSlot(lane));
      while (__hip_atomic_load(slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0)
        __builtin_amdgcn_s_sleep(1);
    }
    // Pair the successful system-scope observations with an acquire before
    // handing the ready data to every consumer wave in this block.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  // HIP's barrier includes global/local workgroup release and acquire fences:
  // propagate the first wave's system acquire to every consumer in this block.
  // All threads participate, including blocks whose copy body has no work.
  __syncthreads();
  EpDispatchAsyncLLRecvCopy_body<kConfig, T>(args);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopyFused_body(EpDispatchCombineArgs args,
                                                   const ::mori::cco::ccoDevComm& comm) {
  const uint64_t flag = args.crossDeviceBarrierFlag[0];
  EpAsyncLLFusedQuiet<kConfig>(args, comm);
  const int lane = threadIdx.x & (warpSize - 1);
  if (threadIdx.x < warpSize) {
    if (lane < kConfig.worldSize) {
      uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
      while (__hip_atomic_load(barrier + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != flag)
        __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  __syncthreads();
  // Even a zero-token rank must execute both completion paths above before the
  // original copy body returns early and the caller can start the next round.
  EpCombineAsyncLLRecvCopy_body<kConfig, T>(args);
}

// The segmented receives fused the same way: wait in every block, then the pass body.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvFinalizeSegFused_body(EpDispatchCombineArgs args,
                                                           const ::mori::cco::ccoDevComm& comm) {
#if MORI_ASYNCLL_RECV_TRIM
  EpDispatchAsyncLLRecvFinalizeSegTrim_body<kConfig, T, true>(args, [&](bool solo) {
    if (solo)
      EpAsyncLLSegDrain<kConfig, true>(args, comm);
    else
      EpAsyncLLSegDrain<kConfig, false>(args, comm);
  });
#else
  EpAsyncLLFusedQuiet<kConfig>(args, comm);
  const int lane = threadIdx.x & (warpSize - 1);
  if (threadIdx.x < warpSize) {
    if (lane < kConfig.worldSize) {
      uint8_t* counts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      uint64_t* slot = reinterpret_cast<uint64_t*>(counts + asyncll::RecvCountSlot(lane));
      while (__hip_atomic_load(slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0)
        __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  __syncthreads();
  EpDispatchAsyncLLRecvFinalizeSeg_body<kConfig, T>(args);
#endif
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopySegFused_body(EpDispatchCombineArgs args,
                                                      const ::mori::cco::ccoDevComm& comm) {
#if MORI_ASYNCLL_RECV_TRIM
  EpCombineAsyncLLRecvCopySegTrim_body<kConfig, T>(args, comm);
#else
  const uint64_t flag = args.crossDeviceBarrierFlag[0];
  EpAsyncLLFusedQuiet<kConfig>(args, comm);
  const int lane = threadIdx.x & (warpSize - 1);
  if (threadIdx.x < warpSize) {
    if (lane < kConfig.worldSize) {
      uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
      while (__hip_atomic_load(barrier + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != flag)
        __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  __syncthreads();
  EpCombineAsyncLLRecvCopySeg_body<kConfig, T>(args);
#endif
}
#endif

}  // namespace v2
}  // namespace ops
}  // namespace mori

// The entry macros come from ep_internode_kernel.hpp; the generated TU picks the
// _LOCAL form for the passes that take no communicator.
