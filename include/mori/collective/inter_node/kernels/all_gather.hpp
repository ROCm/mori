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
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// CU-yield backoff for the inter-node RDMA landing spins. The inter leg of the
// hierarchical AllGather is CU-driven: warp leaders busy-spin on a system-scope
// atomic landing flag while the cross-node RDMA write (~1.5ms round trip) is in
// flight. Under a concurrent backward GEMM those spinning wavefronts contend with
// the GEMM for CU issue slots, so the AllGather cannot hide behind compute. A
// short s_sleep between polls parks the polling wavefront for ~64*N cycles,
// handing its SIMD issue quanta back to the co-resident GEMM without changing any
// memory ordering (the post-loop __threadfence_system still gates visibility, so
// the result is bit-identical to the pure busy-spin -- this is a timing-only
// lever). 0 => OFF: the guarded call compiles out, byte-identical baseline.
// Rebuild to change (device track). Applied only to the long REMOTE-landing spins,
// never the fast intra-block counter joins.
constexpr int kHierInterPollSleep = 0;

// Ring AllGather data movement over an arithmetic sub-group of global PEs:
// ``{peBase, peBase+peStride, ..., peBase+(ringSize-1)*peStride}``. This PE is
// at position ``ringPos`` within that sub-group. The chunk size is passed
// explicitly (``peChunkSize``) so a single fixed-size symmetric ring buffer can
// serve variable message sizes. The ring buffer holds ``ringSize`` chunks; on
// entry only slot ``ringPos`` is filled. After ``ringSize-1`` rounds every
// member holds all ``ringSize`` chunks in ring order. Equal per-PE chunks => no
// last-chunk special case is needed. The whole-world ring is the special case
// ``peBase=0, peStride=1``.
//
// This is the inter-node phase of the hierarchical AllGather: the ring runs
// over node-leaders (or same-local-index ranks across nodes), so neighbours are
// reached over RDMA while same-node members go P2P.
//
// ``numBlocksOverride`` / ``bidOverride`` let this body run as a sub-range of a
// larger fused grid. When >=0 they replace the grid-derived ``gridDim.x`` /
// ``blockIdx.x`` so the ring can occupy blocks [0, ringBlocks) of a fused launch
// while other blocks of the same grid run the intra-node SDMA local-block gather
// concurrently (NIC and XGMI in one kernel, no host-side wait_stream merge).
// Both default to -1, preserving the grid-derived geometry -- inert until a
// fused launcher passes them.
//
// ``chunkReadyFlags`` (default nullptr, inert for every existing caller) is the
// per-chunk landing signal that pipelines the inter-node RDMA ring with the
// intra-node SDMA remote-block reassembly. When non-null it is a device array of
// at least ``numBlocks`` uint64_t in ordinary (cached) HBM, zeroed by the caller
// before launch. As soon as this block's (channel ``bid``'s) inbound sub-range
// has fully landed in this PE's ring buffer -- i.e. after the receiver's
// system-scope acquire + __threadfence_system that make those bytes coherently
// visible to this GPU's CUs -- block ``bid`` publishes ``chunkReadyFlags[bid]``.
// A concurrent reassembly block in the same fused grid spins on that flag and,
// the instant sub-range ``bid`` is ready, SDMA-pushes exactly that sub-range
// over XGMI while ring channel ``bid+1`` is still crossing the NIC, overlapping
// the two hierarchy legs. Because each PE reassembles a remote block by pushing
// from its own ring buffer, the only dependency is this PE's own ring landing --
// a purely local flag spin, no global barrier. nullptr keeps the standalone ring
// byte-for-byte identical.
inline __device__ void AllGatherRingSubGroupKernelBody(
    int ringPos, int ringSize, int peBase, int peStride,
    const application::SymmMemObjPtr memObj, const application::SymmMemObjPtr flagsObj,
    size_t peChunkSize, int numQp = 1, int numBlocksOverride = -1, int bidOverride = -1,
    bool usePutSignal = false, bool useWriteImm = false, uint64_t* chunkReadyFlags = nullptr,
    uint64_t opGen = 0, bool useRead = false, int wqeDepth = 1, int deepPipe = 1,
    int deepPipeImm = 0, int deepPipeQuiet = 0, int dpSerialDrain = 0,
    bool useWriteFence = false, int fifoFullWidth = 0, int dpTailPct = 0,
    int fifoProg = 0, int shardDrain = 0, int directLand = 0,
    application::SymmMemObjPtr gOutMemObj = application::SymmMemObjPtr{},
    int gGroupSize = 0, int gGroupPos = 0) {
  int nextPos = (ringPos + 1) % ringSize;
  int nextPeer = peBase + nextPos * peStride;
  int maxRounds = ringSize - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  const int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  const int threadLinearId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int warpId = threadLinearId / warpSize;
  const int warpsPerBlock = threadsPerBlock / warpSize;

  // Multi-block ring ("channels", RCCL-style). A single-block ring (numQp warps
  // in one CTA, each on its own QP) under-fills the NIC; driving many CTAs
  // (channels) concurrently fills it. Each block ``bid`` of ``gridDim.x`` handles
  // a disjoint 16B-aligned sub-range of every chunk and uses its own flag region
  // [bid*ringSize, (bid+1)*ringSize), so blocks never alias data or flags. Block
  // bid issues its put on qpId=bid so each channel drives a distinct QP -- the
  // union still tiles each chunk exactly => byte-identical result. Only engaged
  // for true RDMA neighbours: same-node P2P/SDMA lowers to one anvil queue per
  // (src,dst), and multiple CTAs hammering it overflow the retry budget and
  // crash. For a non-RDMA neighbour (single-node simulation) only block 0 runs
  // the single-block path and the other blocks return, keeping single-node
  // bit-exact and crash-free.
  const int numBlocks = (numBlocksOverride >= 0) ? numBlocksOverride : static_cast<int>(gridDim.x);
  const int bid = (bidOverride >= 0) ? bidOverride : static_cast<int>(blockIdx.x);
  application::TransportType nextXportMb =
      shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerRdmaMb = (nextXportMb == application::TransportType::RDMA);
  bool multiBlock = (numBlocks > 1 && peerRdmaMb);
  if (numBlocks > 1 && !peerRdmaMb && bid != 0) {
    // Single-node simulation with a multi-block launch: only block 0 works.
    return;
  }
  // Per-block 16B-aligned sub-range of the chunk (full chunk when single-block).
  size_t blkOff = 0;
  size_t blkBytes = peChunkSize;
  int flagBase = 0;
  if (multiBlock) {
    const size_t kAlignB = 16;
    size_t nUnits = (peChunkSize + kAlignB - 1) / kAlignB;
    size_t unitsPerBlk = (nUnits + numBlocks - 1) / numBlocks;
    size_t startUnit = static_cast<size_t>(bid) * unitsPerBlk;
    size_t endUnit = startUnit + unitsPerBlk;
    if (endUnit > nUnits) endUnit = nUnits;
    blkOff = startUnit * kAlignB;
    size_t blkEnd = endUnit * kAlignB;
    if (blkEnd > peChunkSize) blkEnd = peChunkSize;
    blkBytes = (blkOff < blkEnd) ? (blkEnd - blkOff) : 0;
    flagBase = bid * ringSize;
  }

  // Multi-QP fan-out gate. A single warp on a single QP under-fills the NIC, so
  // we fan the per-round put across ``numQp`` QPs (warp w -> qpId=w, disjoint
  // 16B-aligned sub-range) -- but only when the neighbour is reached over RDMA.
  // For a same-node neighbour ShmemPutMemNbiWarp lowers to a single anvil SDMA
  // queue per (src,dst); multiple warps hammering it overflow the retry budget
  // and crash. So we read the neighbour's transport at runtime: single-node
  // simulation (P2P/SDMA) keeps the single-warp path and stays bit-exact; only a
  // true cross-node (RDMA) neighbour fans out. Gated additionally on numQp>1 so
  // the flat whole-world ring (numQp defaults to 1) is byte-for-byte unchanged.
  application::TransportType nextXport =
      shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerIsRdma = (nextXport == application::TransportType::RDMA);
  // Multi-block and within-block multi-QP fan-out are mutually exclusive: in
  // multi-block mode each CTA already drives its own QP (qpId=bid) on its own
  // sub-range, so a single warp per block is correct.
  int useWarps = (!multiBlock && numQp > 1 && peerIsRdma) ? numQp : 1;
  if (useWarps > warpsPerBlock) useWarps = warpsPerBlock;
  bool fanOut = (useWarps > 1);

  // Put-with-signal on the multi-block channel path (env MORI_HIER_RING_PUT_SIGNAL).
  // The plain channel path puts each CTA's sub-range on qpId=bid, then thread 0
  // does a separate ShmemQuietThread(nextPeer, bid) drain + a separate flag AMO.
  // That quiet-drain stalls the channel until its RDMA send CQ empties before the
  // receiver's flag can fire. Fusing the channel's data WRITE + its completion-
  // flag AMO into one ShmemPutMemNbiSignal on the same QP (qpId=bid) makes the
  // flag ride strictly after the data WRITE (RC in-order on that QP), so the
  // receiver observing the per-channel flag is guaranteed the sub-range has landed
  // globally -- with no separate quiet and no separate AMO. Byte image is
  // identical (same sub-range tiling, same flag slot); only the completion
  // mechanism changes. Same put-signal mechanism used on the single-warp
  // (signalFused) and fan-out (fanOutSignal) paths. Gated on usePutSignal so the
  // default path stays byte-for-byte unchanged.
  bool multiBlockSignal = (usePutSignal && peerIsRdma && multiBlock);

  // Fan-out put-with-signal (env MORI_HIER_RING_PUT_SIGNAL, default off): on the
  // multi-QP fan-out path the chunk is split across ``useWarps`` QPs, so a single
  // completion-flag AMO (even RC-ordered on one QP) only orders after that QP's
  // data -- the other QPs' tail bytes can still be in flight when the receiver
  // observes the flag (a remote-landing race). Fix: have each fan-out warp fuse
  // its data WRITE + a flag AMO_ADD(1) on its own QP via ShmemPutMemNbiSignalWarp.
  // On RC the responder executes each QP's WRITE then its AMO in order, so every
  // QP's data is globally visible before its own +1. The receiver waits for the
  // flag to reach the number of active fan-out warps (``fanActive``) => it can
  // only proceed after all QPs' data has landed, with no host sync (keeps the
  // ring/gather overlap). In the symmetric homogeneous-RDMA subgroup ring
  // (leaders) next/prev are both RDMA, so send/recv counts match.
  bool fanOutSignal = (usePutSignal && peerIsRdma && !multiBlock && fanOut);
  int fanActive = 1;
  if (useWarps > 1) {
    const size_t kAlignS = 16;
    size_t nUnitsS = (peChunkSize + kAlignS - 1) / kAlignS;
    size_t unitsPerWarpS = (nUnitsS + useWarps - 1) / useWarps;
    if (unitsPerWarpS == 0) unitsPerWarpS = 1;
    fanActive = static_cast<int>((nUnitsS + unitsPerWarpS - 1) / unitsPerWarpS);
    if (fanActive > useWarps) fanActive = useWarps;
    if (fanActive < 1) fanActive = 1;
  }
  int prevPos = (ringPos - 1 + ringSize) % ringSize;
  int prevPeer = peBase + prevPos * peStride;
  application::TransportType prevXport =
      shmem::GetGlobalGpuStatesPtr()->transportTypes[prevPeer];
  bool prevIsRdma = (prevXport == application::TransportType::RDMA);
  // Expected increments on our recv slot = active fan-out warps the sender (prev)
  // used, iff prev also fans out with signals; else the single +1.
  int expectedRecvSig = (fanOutSignal && prevIsRdma) ? fanActive : 1;

  // RDMA-READ (pull) ring gate. Engaged only on the single-round (ringSize==2)
  // all-RDMA inter-node phase (the 2-node hierarchical AG). In that case the
  // chunk this PE needs (data-rank recvDataRank) is prevPeer's own chunk, already
  // present in prevPeer's ring slot after the intra prepare barrier -- so it can
  // be pulled with an RDMA READ rather than waiting for the peer to push it. The
  // READ completion, drained by our own quiet, is a consumer-side landing
  // guarantee (bytes physically in this PE's buffer + system-fence-visible to its
  // CUs): no cross-PE flag AMO, no receiver spin, no remote-landing race.
  // Restricted to !multiBlock (single-block fan-out) and maxRounds==1 so larger
  // rings / the multi-channel path / single-node keep the push path byte-for-byte.
  // Force-disabled here (`&& false`): a single-QP/single-outstanding READ has no
  // fan-out concurrency so its round-trip is fully serialized per op and is far
  // slower than the push path at latency-bound sizes. This leaves
  // MORI_HIER_RING_READ to reach only the multiBlockRead landing-fence path below.
  bool useReadRing =
      (useRead && peerIsRdma && prevIsRdma && maxRounds == 1 && !multiBlock) && false;

  // RDMA-READ (pull) landing fence extended to the multi-block channel path. A
  // large multi-block AG (numQp==1) never hits the !multiBlock useReadRing above;
  // it otherwise falls back to flag-spin recv whose flag can be observed before
  // the NIC's WRITE DMA is globally visible (a stale-remote-half race), and the
  // WRITE_WITH_IMM fix is unavailable on this mlx5 provider. RDMA READ sidesteps
  // WRITE_IMM entirely: each channel CTA ``bid`` pulls its own sub-range
  // [blkOff,blkBytes) of prevPeer's own chunk (slot recvDataRank, present after
  // the intra prepare barrier) into our matching slot on qpId=bid, then drains
  // only qpId=bid's READ completion. A READ CQE is produced only after the
  // response payload has physically landed in this PE's buffer, so the quiet is a
  // per-channel consumer-side landing fence -- no flag AMO, no receiver spin, no
  // remote-landing race, no host stall. Byte-identical to the push receive (same
  // slot/offset/bytes; only the completion mechanism changes). Gated on
  // maxRounds==1 so larger rings keep the push path.
  bool multiBlockRead = (useRead && peerIsRdma && prevIsRdma && maxRounds == 1 && multiBlock);

  // T40 (A): WRITE-PUSH (SEND-CQ) per-channel landing fence for the giant
  // multiBlock AG -- the WRITE-side counterpart of T38's multiBlockRead. T38
  // proved the READ CQE is a bit-exact device landing fence on this giant AG, but
  // RDMA-READ underfills on mlx5 (~125 GB/s => 0.70x). This keeps the fast
  // RDMA-WRITE PUSH (~155-170 GB/s) and makes the completion bit-exact by
  // construction: each channel CTA ``bid`` PUSHES its OWN sub-range [blkOff,
  // blkBytes) of my chunk to nextPeer on qpId=bid as a FUSED put-with-signal (the
  // flag AMO rides the SAME QP strictly AFTER the data WRITE, RC in-order, so the
  // receiver's per-channel flag can never beat the data -- the flag-beats-data
  // race that makes the plain multiBlock push E2E-drift), then DRAINS ONLY
  // qpId=bid's SEND CQE (per-channel quiet, no cross-CTA CQ race) so the local
  // WQE has completed before this block publishes. The receiver spins its own
  // per-channel inbound flag (system-acquire) + system-fences. Byte-identical to
  // the multiBlock push receive (same slot/offset/bytes); only the completion
  // mechanism (fused-signal + send-CQ drain vs separate put+quiet+AMO) changes.
  // Gated maxRounds==1 (the N=2 hier AG) so larger rings keep the default path.
  bool multiBlockWrite =
      (useWriteFence && peerIsRdma && prevIsRdma && maxRounds == 1 && multiBlock);

  // Phase-6 WRITE_WITH_IMM (single-warp, single-QP cross-node path only). The
  // sender emits RDMA_WRITE_WITH_IMM instead of put+quiet+flag-AMO; the receiver
  // consumes the recv-CQ completion instead of spinning the flag. The recv-CQE
  // cannot be observed before the write payload has landed globally, so this is
  // the transport-level completion that closes the remote-landing stale-read race
  // (device-side ordering alone was insufficient; only a host sync worked) without the
  // host stall. Gated to !multiBlock && !fanOut (numQp==1) so numQp>1 / multi-channel
  // and single-node (P2P/SDMA) paths stay byte-for-byte on the proven flag path.
  bool writeImm = (useWriteImm && peerIsRdma && !multiBlock && !fanOut);
  bool recvWriteImm = (useWriteImm && prevIsRdma && !multiBlock && !fanOut);
  // Phase-6b: WRITE_WITH_IMM on the MULTI-QP FAN-OUT path. This is the ONLY
  // WRITE_IMM variant that engages on the big embed/lm_head AG, because under
  // FSDP that AG runs with numQp>1 (fanOut) -- the single-warp writeImm above is
  // gated !fanOut and so NEVER fires for it (which is why every prior numQp==1
  // WRITE_IMM FSDP test was inconclusive: at numQp==1 the big AG hits multiBlock,
  // also gated off). Each active fan-out warp emits its disjoint sub-range as an
  // RDMA_WRITE_WITH_IMM on its OWN QP (qpId=warpId); the receiver consumes one
  // recv-CQE per active QP. The recv-CQE is produced only AFTER that QP's payload
  // has landed globally, so the receiver proceeds coherent with NO host sync and
  // NO flag AMO -- the transport completion the 20 device/host-bounded avenues
  // could not give on this path. Gated to fanOut so numQp==1 stays byte-identical.
  bool fanOutWriteImm = (useWriteImm && peerIsRdma && !multiBlock && fanOut);
  bool recvFanOutWriteImm = (useWriteImm && prevIsRdma && !multiBlock && fanOut);
  // WRITE_WITH_IMM on the MULTI-BLOCK CHANNEL path (env MORI_HIER_RING_WRITE_IMM).
  // At N=2 the big AG runs multiBlock (numQp==1), so the single-warp writeImm
  // (gated !multiBlock) and fanOutWriteImm (gated fanOut) NEVER fire for it -- it
  // falls back to the flag-spin recv whose flag can be observed BEFORE the NIC's
  // WRITE DMA is globally visible to the consumer's CU reads (the FSDP E2E stale-
  // remote-half completion race; device-side reader barriers did not fix it, only a
  // host drain did). Extend WRITE_WITH_IMM to the channel path: each CTA bid
  // emits its sub-range as RDMA_WRITE_WITH_IMM on qpId=bid; the receiver consumes
  // THIS channel's recv-CQE (generated only AFTER the write payload has landed
  // globally) instead of spinning the flag. Byte image is identical (same sub-range
  // tiling); only the completion mechanism changes. This is the transport-level
  // completion the big-AG path needs to make fuse_local E2E bit-exact WITHOUT a
  // host sync. Gated on useWriteImm so the default path stays byte-for-byte the same.
  bool multiBlockWriteImm = (useWriteImm && peerIsRdma && multiBlock);
  bool recvMultiBlockWriteImm = (useWriteImm && prevIsRdma && multiBlock);
  // Pre-post one recv WQE per ring round so every remote WRITE_WITH_IMM yields a
  // recv-CQE. bytes=0: a pure write-with-imm does not consume the recv SGL for
  // payload (data lands at the write's addr/rkey); the WQE only produces the CQE.
  // recvPostIdx is persisted in the QP handle (rqPostIdx) across launches.
  if (recvWriteImm && threadLinearId == 0) {
    shmem::ShmemPostRecvImm(reinterpret_cast<uintptr_t>(memObj->localPtr), memObj->lkey,
                            /*bytes=*/0, /*count=*/static_cast<uint32_t>(maxRounds), prevPeer, 0);
  }
  if (recvFanOutWriteImm && threadLinearId == 0) {
    // prev fans out across fanActive QPs, so each of our QPs 0..fanActive-1 must
    // have maxRounds recv WQEs pre-posted (one per round per QP).
    for (int q = 0; q < fanActive; ++q) {
      shmem::ShmemPostRecvImm(reinterpret_cast<uintptr_t>(memObj->localPtr), memObj->lkey,
                              /*bytes=*/0, /*count=*/static_cast<uint32_t>(maxRounds), prevPeer, q);
    }
  }
  if (recvMultiBlockWriteImm && threadLinearId == 0 && blkBytes > 0) {
    // This channel bid receives prev's sub-range on qpId=bid; pre-post maxRounds
    // recv WQEs on that QP so every remote WRITE_WITH_IMM yields a recv-CQE.
    shmem::ShmemPostRecvImm(reinterpret_cast<uintptr_t>(memObj->localPtr), memObj->lkey,
                            /*bytes=*/0, /*count=*/static_cast<uint32_t>(maxRounds), prevPeer, bid);
  }
  __syncthreads();

  // ==========================================================================
  // DEEP-SQ TEMPORAL PIPELINE (MORI_HIER_DEEP_PIPE=P). Split the chunk into P
  // temporal sub-chunks issued back-to-back on the SAME full numQp fan-out with a
  // per-sub-chunk put-with-signal; sub-chunk p's landing flag fires (RC in-order)
  // before p+1's, so a reassembly worker pushes p over XGMI while p+1.. still
  // cross the NIC -- hides the 46% intra reassembly under the 54% inter fill with
  // NO inter-fill growth (unlike the spatial RING_BLOCKS split which drops QPs).
  // Engaged only on the single-round (ringSize==2) all-RDMA fan-out path. Uses the
  // full useWarps QP fan-out per sub-chunk (full inter BW), publishes P
  // chunkReadyFlags in temporal order. Self-contained: returns before the classic
  // round loop. INERT when deepPipe<=1 (byte-identical shipped path).
  bool deepPipeEngaged = (deepPipe > 1 && peerIsRdma && prevIsRdma && maxRounds == 1 &&
                          !multiBlock && chunkReadyFlags != nullptr && !useReadRing &&
                          !useWriteImm && useWarps >= 1);
  if (deepPipeEngaged) {
    // Min-sub-chunk clamp: the deep-SQ temporal FIFO splits the chunk into P sub-chunks
    // issued back-to-back on the full useWarps QP fan-out (the NCCL_STEPS full-width-per-
    // step model). But an unclamped large P at a small total size shrinks each sub-chunk
    // below a useful RDMA transfer granularity -- the tiny per-sub-chunk WQEs starve the
    // NIC and the extra flag round-trips can deadlock small transfers. Clamp P so every
    // temporal sub-chunk carries >= kMinSubChunkB
    // (1 MiB): reqPmax = peChunkSize / kMinSubChunkB. This makes ANY requested depth
    // (incl. RCCL-matched P=8) safe at every size, and is BIT-EXACT + byte-identical
    // for the shipped path -- P only controls the temporal partition of the SAME
    // contiguous bytes issued+landed in order, and the default P=2 already yields
    // >=8 MiB sub-chunks at every UT size >=32MB (clamp is a no-op there).
    const size_t kMinSubChunkB = static_cast<size_t>(1) << 20;
    int reqPmax = static_cast<int>(peChunkSize / kMinSubChunkB);
    if (reqPmax < 1) reqPmax = 1;
    const int P = (deepPipe < reqPmax) ? deepPipe : reqPmax;
    const int sendRank = ringPos;                       // maxRounds==1: send our own chunk
    const size_t chunkBaseOffsetSend = static_cast<size_t>(sendRank) * peChunkSize;
    const size_t kAlignDP = 16;
    const size_t nUnits = (peChunkSize + kAlignDP - 1) / kAlignDP;
    const size_t unitsPerP = (nUnits + static_cast<size_t>(P) - 1) / static_cast<size_t>(P);
    const int sw = useWarps;  // sender fan-out warps per sub-chunk (== numQp fan-out)
    // SKEWED TEMPORAL SPLIT (T29 front-load + T37 head-skew): the P==2 boundary is
    // bnd = nUnits - nUnits*dpTailPct/100, so sub-chunk 0 = (100-pct)% and sub-chunk
    // 1 (last) = pct%. pct<50 FRONT-LOADS (small LAST => shrinks the exposed reasm
    // tail, T29). pct>50 HEAD-SKEWS (small FIRST => the first sub-chunk lands sooner,
    // attacking the T33 first_land latency that is 55% of wall at 32MB where the reasm
    // tail is only 10%): the tiny head crosses the NIC fast so reassembly + the NIC
    // pipe fill start earlier, overlapping the large tail's transfer. Producer +
    // consumer compute this boundary identically => flag slot p guards exactly the
    // reassembled bytes (bit-exact) at ANY pct. pct==0 or P!=2 => uniform (byte-ident).
    auto dpRange = [&](int p, size_t& sU, size_t& eU) {
      if (dpTailPct > 0 && dpTailPct < 100 && dpTailPct != 50 && P == 2) {
        size_t tailU = (nUnits * static_cast<size_t>(dpTailPct)) / 100;
        size_t bnd = nUnits - tailU;
        if (p == 0) { sU = 0; eU = bnd; } else { sU = bnd; eU = nUnits; }
      } else {
        sU = static_cast<size_t>(p) * unitsPerP;
        eU = sU + unitsPerP;
        if (eU > nUnits) eU = nUnits;
        if (sU > eU) sU = eU;
      }
    };
    // Per-sub-chunk active-warp count, computed identically to the sender tiling.
    // active(p) = # of warps that actually get a non-empty tile of sub-chunk p.
    auto activeOf = [&](int p) -> int {
      size_t sU, eU;
      dpRange(p, sU, eU);
      if (sU >= eU) return 0;
      size_t subUnits = eU - sU;
      size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
      if (upw == 0) upw = 1;
      int a = static_cast<int>((subUnits + upw - 1) / upw);
      if (a > sw) a = sw;
      if (a < 1) a = 1;
      return a;
    };
    // ==== PER-QP FINE-GRAIN INTER-ARRIVAL DRAIN (shardDrain, MORI_HIER_SHARD_DRAIN). ====
    // Go FINER than the temporal P: treat each of the sw QPs as its OWN progressive
    // landing shard. Issue the WHOLE chunk at full sw-QP width FIRST (full NIC fill,
    // P-independent), then each warp-leader drains its OWN QP's send-CQ + system-fences
    // and publishes chunkReadyFlags[warpId] the INSTANT that QP lands -- so the single
    // reasm worker (partition==numQp) pushes shard 0 while shards 1..sw-1 still cross
    // the NIC, cutting the first_land latency prefix from ~1/2 chunk (P=2 group drain)
    // to ~1/sw. Bit-exact: the sw per-QP 16B tiles [w*upw,(w+1)*upw) exactly tile the
    // chunk (upw = ceil(nUnits/sw), IDENTICAL to the consumer's partition==sw
    // unitsPerChan), each flag AMO strictly follows its own QP drain+fence. Deadlock-
    // free: full send issued before any wait; every wait is on OUR OWN inbound flags
    // (set by the partner's symmetric per-QP AMO) -- no cross-rank circular ordering.
    // Requires sw>=1 && sw<=warpsPerBlock. Takes precedence over fifoProg/fifoFullWidth.
    if (shardDrain && sw >= 1 && sw <= warpsPerBlock) {
      const size_t upwS = (nUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
      // SEND: warp w owns QP w, sends its 16B-aligned per-QP tile of the chunk.
      if (warpId < sw) {
        size_t wS = static_cast<size_t>(warpId) * upwS;
        size_t wE = wS + upwS;
        if (wE > nUnits) wE = nUnits;
        if (wS < wE) {
          size_t so = wS * kAlignDP;
          size_t eo = wE * kAlignDP;
          if (eo > peChunkSize) eo = peChunkSize;
          size_t off = chunkBaseOffsetSend + so;
          shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, eo - so, nextPeer, warpId);
        }
      }
      __syncthreads();
      // PER-QP PROGRESSIVE DRAIN + PUBLISH: warp-leader w drains QP w (RC in-order =>
      // shard w landed at nextPeer) + system-fence, AMOs the receiver's inbound flag
      // slot w, waits its OWN inbound flag w, then publishes chunkReadyFlags[w]. All P
      // shards fire independently (disjoint QPs/CQs/flag slots) => concurrent, no
      // cross-shard serialize. warpsPerBlock>=sw so a leader warp exists for every QP.
      const bool sdWarpLead = (threadLinearId % warpSize) == 0;
      if (sdWarpLead && warpId < sw) {
        size_t wS = static_cast<size_t>(warpId) * upwS;
        if (wS < nUnits) {  // non-empty shard
          shmem::ShmemQuietThread(nextPeer, warpId);
          __threadfence_system();
          shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
              flagsObj, (flagBase + warpId) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
              nextPeer);
          long long spin = 0;
          while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + warpId) <
                 static_cast<uint64_t>(1)) {
            if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + warpId,
                                        opGen ? opGen : static_cast<uint64_t>(1));
        }
      }
      __syncthreads();
      // Reset the inbound flag slots for the next launch (mirrors the FIFO epilogue).
      for (int idx = threadLinearId; idx < sw; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      return;
    }
    // ==== FULL-WIDTH DEEP-SQ INFLIGHT FIFO (fifoFullWidth, MORI_HIER_FIFO). ====
    // Depth is decoupled from spatial width: the deepPipeQuiet path buys temporal
    // depth by splitting
    // the QP fan-out per sub-chunk (g = sw/P QPs each) -- so a deeper P under-fills
    // the NIC. Here EVERY temporal sub-chunk uses the FULL sw-QP width, and the P
    // sub-chunks are issued BACK-TO-BACK on each QP BEFORE any drain -- so each QP's
    // send queue carries P in-flight WQEs (deep SQ = the NIC-fill lever) at full
    // width. Completion: every sub-chunk on QP q is RC in-order, so a SINGLE
    // parallel per-QP send-CQ drain (warp-leader per QP) proves ALL P*sw sends
    // landed at nextPeer; then thread 0 AMOs all P remote flag slots and one warp
    // publishes all P chunkReadyFlags after its own inbound flags arrive. BIT-EXACT
    // by construction (union of the P*sw tiles == the chunk, every flag AMO follows
    // a full drain of all landings, per-slot flags + per-QP RC order preserved --
    // only the ORDER of independent per-QP completions is relaxed, same relaxation
    // deepPipeQuiet already sanctions). Requires sw>=1 and P<=warpsPerBlock. Takes
    // precedence over deepPipeQuiet/deepPipeImm. Returns before the classic loop.
    // ==== PROGRESSIVE DEEP-PIPE PUBLISH (fifoProg, MORI_HIER_FIFO_PROG). ====
    // The consume-side pincer stacked on the crown. fifoFullWidth (below) issues
    // all P sub-chunks deep then batch-drains + batch-publishes all P flags together
    // -- a completion BARRIER where flag[0] can't fire until sub-chunk P-1 lands, so
    // the receiver's intra XGMI reassembly of sub-chunk 0 never overlaps the inter
    // fill of 1..P-1. This lever instead walks the P sub-chunks in strict temporal
    // order and publishes EACH chunkReadyFlags[p] the instant its own sub-chunk lands,
    // BEFORE issuing p+1: warp w sends its dpRange tile of p on qpId=w (full sw-QP
    // width), a parallel per-QP send-CQ drain proves p landed at nextPeer, thread 0
    // AMOs the receiver's inbound slot, the reader warp waits its own inbound flag[p]
    // and publishes chunkReadyFlags[p] -- so the reassembly worker reassembles p over
    // XGMI while p+1.. are still crossing the NIC (the ~46% intra tail hidden under the
    // ~54% inter fill). BIT-EXACT: same dpRange tiling + flag slots as the crown
    // deep-pipe path, each AMO strictly follows a full drain of its own sub-chunk's
    // landings -- only MORE ordered (strict temporal vs the batch path's relaxed
    // per-QP order). Takes precedence over fifoFullWidth. Requires sw>=1 &&
    // sw<=warpsPerBlock. Returns before the classic round loop.
    if (fifoProg && P > 1 && sw >= 1 && sw <= warpsPerBlock) {
      const bool progWarpLead = (threadLinearId % warpSize) == 0;
      for (int p = 0; p < P; ++p) {
        size_t sU, eU;
        dpRange(p, sU, eU);
        const bool pActive = (sU < eU);
        // SEND sub-chunk p at full sw-QP width (warp w owns QP w).
        if (pActive && warpId < sw) {
          size_t subUnits = eU - sU;
          size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
          if (upw == 0) upw = 1;
          size_t wS = sU + static_cast<size_t>(warpId) * upw;
          size_t wE = wS + upw;
          if (wE > eU) wE = eU;
          if (wS < wE) {
            size_t so = wS * kAlignDP;
            size_t eo = wE * kAlignDP;
            if (eo > peChunkSize) eo = peChunkSize;
            size_t off = chunkBaseOffsetSend + so;
            shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, eo - so, nextPeer, warpId);
          }
        }
        __syncthreads();
        // PER-QP SEND-CQ DRAIN of sub-chunk p only (RC in-order per QP => p landed at
        // nextPeer) + system fence, so the flag can never precede its bytes.
        if (pActive && progWarpLead && warpId < sw) {
          shmem::ShmemQuietThread(nextPeer, warpId);
          __threadfence_system();
        }
        __syncthreads();
        // PUBLISH flag[p] NOW: thread 0 AMOs the receiver's inbound slot; the reader
        // warp waits its own inbound flag[p] then publishes chunkReadyFlags[p] -- all
        // BEFORE issuing p+1 (tail-per-step).
        if (pActive && threadLinearId == 0) {
          shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
              flagsObj, (flagBase + p) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
              nextPeer);
        }
        if (pActive && threadLinearId == warpSize) {
          long long spin = 0;
          while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                 static_cast<uint64_t>(1)) {
            if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                        opGen ? opGen : static_cast<uint64_t>(1));
        }
        __syncthreads();
      }
      // Reset the inbound flag slots so the next launch starts clean (mirrors the
      // fifoFullWidth epilogue).
      for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      return;
    }
    if (fifoFullWidth && P > 1 && sw >= 1 && sw <= warpsPerBlock) {
      // SENDER: warp w in [0,sw) owns QP w. For EACH temporal sub-chunk p it sends
      // its 16B-aligned tile of p on qpId=w. All P tiles ride qpId=w back-to-back
      // => P-deep SQ per QP, full sw-QP width per sub-chunk.
      if (warpId < sw) {
        for (int p = 0; p < P; ++p) {
          size_t sU, eU;
          dpRange(p, sU, eU);
          if (sU >= eU) break;
          size_t subUnits = eU - sU;
          size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
          if (upw == 0) upw = 1;
          size_t wS = sU + static_cast<size_t>(warpId) * upw;
          size_t wE = wS + upw;
          if (wE > eU) wE = eU;
          if (wS >= wE) continue;
          size_t so = wS * kAlignDP;
          size_t eo = wE * kAlignDP;
          if (eo > peChunkSize) eo = peChunkSize;
          size_t off = chunkBaseOffsetSend + so;
          shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, eo - so, nextPeer, warpId);
        }
      }
      __syncthreads();
      // PARALLEL PER-QP SEND-CQ DRAIN: each of the sw warp-leaders quiets its OWN
      // QP (covers ALL P WQEs on it, RC in-order) + system-fences. A block barrier
      // then guarantees every QP's P sends landed globally before any AMO.
      const bool warpLead = (threadLinearId % warpSize) == 0;
      if (warpLead && warpId < sw) {
        shmem::ShmemQuietThread(nextPeer, warpId);
        __threadfence_system();
      }
      __syncthreads();
      // PUBLISH: thread 0 AMOs all P remote flag slots (data already landed, so no
      // flag can beat its bytes); one warp waits on its own inbound P flags and
      // publishes chunkReadyFlags[p]. activeOf(p)>0 => sub-chunk p was sent.
      if (threadLinearId == 0) {
        for (int p = 0; p < P; ++p) {
          if (activeOf(p) > 0) {
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
                flagsObj, (flagBase + p) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
                nextPeer);
          }
        }
      }
      if (threadLinearId == warpSize) {
        for (int p = 0; p < P; ++p) {
          if (activeOf(p) > 0) {
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                        opGen ? opGen : static_cast<uint64_t>(1));
        }
      }
      __syncthreads();
      for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      return;
    }
    // DEEP_PIPE_QUIET: scale-robust per-sub-chunk landing fence via QP quiet-drain.
    // The put-signal AMO can beat its own data >=64MB and WRITE_WITH_IMM is HW-
    // unavailable here, but a SEND-CQ drain of the sub-chunk's QP is a definitive
    // remote-landing proof (the teamC ring's >=32MiB-parity mechanism). Give each
    // temporal sub-chunk p its OWN QP (qpId = p % sw), issue a PLAIN put (no fused
    // AMO), then in temporal order drain qpId=(p%sw) with ShmemQuietThread(nextPeer,
    // qp) (== sub-chunk p landed at nextPeer, RC in-order per QP) and only THEN AMO
    // the receiver's flag slot p. Because distinct sub-chunks ride distinct QPs, the
    // temporal pipeline is preserved (p's flag fires before p+1's data finishes),
    // yet the flag NEVER precedes the landing => bit-exact even at 64-256MB. Self-
    // contained: returns before the IMM/signal branches. Requires chunkReadyFlags.
    if (deepPipeQuiet) {
      // Disjoint QP GROUP per temporal sub-chunk so distinct sub-chunks stay on
      // distinct QPs (temporal landing order preserved => p's flag can fire while
      // p+1.. still cross the NIC) AND each sub-chunk still fans across g = sw/P
      // QPs for full per-sub-chunk BW (the 1-QP mapping under-filled the NIC ->
      // 0.86x). group(p) = QPs [p*g, p*g+g); warp within the group tiles the
      // sub-chunk in 16B units. When sw < P (P>sw) groups wrap (g==1, p%sw) --
      // then some sub-chunks share a QP and the drain also covers the later one
      // (still bit-exact, just less overlap). Draining a sub-chunk's WHOLE group
      // (all g QPs) before its AMO is the landing fence.
      const int g = (sw >= P) ? (sw / P) : 1;               // QPs per sub-chunk
      auto grpBase = [&](int p) -> int { return (sw >= P) ? (p * g) : (p % sw); };
      auto nonEmptyDP = [&](int p) -> bool {
        size_t sU, eU;
        dpRange(p, sU, eU);
        return sU < eU;
      };
      // SENDER: warp w belongs to sub-chunk p = w / g (when sw>=P), lane role wl =
      // w % g drives qpId = grpBase(p)+wl on its disjoint 16B tile of sub-chunk p.
      if (warpId < sw) {
        int p, wl, gg;
        if (sw >= P) {
          p = warpId / g;
          wl = warpId % g;
          gg = g;
          if (p >= P) p = -1;                                // extra warps idle
        } else {
          p = warpId;                                        // P>sw: each of first sw warps -> its own sub-chunk group start
          wl = 0;
          gg = 1;
        }
        if (p >= 0) {
          for (int pp = p; pp < P; pp += (sw >= P ? P : sw)) {  // sw<P: warp handles pp = warpId, warpId+sw, ...
            size_t sU, eU;
            dpRange(pp, sU, eU);
            if (sU >= eU) { if (sw >= P) break; else continue; }
            size_t subUnits = eU - sU;
            // Tile sub-chunk pp across gg QPs; lane wl takes its slice.
            size_t upl = (subUnits + static_cast<size_t>(gg) - 1) / static_cast<size_t>(gg);
            if (upl == 0) upl = 1;
            size_t lS = sU + static_cast<size_t>(wl) * upl;
            size_t lE = lS + upl;
            if (lE > eU) lE = eU;
            if (lS >= lE) { if (sw >= P) continue; else continue; }
            size_t so = lS * kAlignDP;
            size_t eo = lE * kAlignDP;
            if (eo > peChunkSize) eo = peChunkSize;
            size_t off = chunkBaseOffsetSend + so;
            int qp = grpBase(pp) + wl;
            shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, eo - so, nextPeer, qp);
            if (sw >= P) break;   // sw>=P: one warp issues exactly one sub-chunk tile
          }
        }
      }
      __syncthreads();
      // PARALLEL SEND-DRAIN + RECV-PUBLISH: the prior serial thread-0 drain loop
      // (quiet group 0, AMO 0, quiet group 1, AMO 1, ...) forced flag p to wait on
      // EVERY earlier group's drain even when group p landed first, throttling the
      // inter->intra pipeline to ~0.86-0.98x. Give each sub-chunk p its OWN drain
      // warp-leader (thread p*warpSize) and its OWN publish warp-leader (thread
      // (P+p)*warpSize) so a landed group fires its flag the INSTANT its own QP
      // group drains -- concurrently with the other groups still crossing the NIC.
      // Distinct sub-chunks ride DISJOINT QP groups => distinct ep[]/CQ state, so
      // concurrent ShmemQuietThread(nextPeer, qp) calls never share a WQ/CQ =>
      // bit-exact (same drains, same AMOs, same flag slots; only the ORDER of
      // independent completions is relaxed, which the per-slot flags already allow).
      // ONE warp-leader per sub-chunk (thread p*warpSize): it drains its OWN QP
      // group, AMOs the remote landing flag p, THEN waits for its OWN incoming flag
      // p and publishes chunkReadyFlags[p]. Merging send-drain + recv-publish into a
      // SINGLE leader per p (was two leaders, thread p*ws and (P+p)*ws) is required
      // on wave64 HW: block=512 => warpsPerBlock = 512/64 = 8, so the old 2*P-warp
      // split DEADLOCKED at P=8 (recv-publish leaders warps 8..15 do not exist =>
      // chunkReadyFlags never published => hang). P leaders fit P<=warpsPerBlock, so
      // this enables DEEP_PIPE=8 parallel drain. All P leaders still run CONCURRENTLY
      // across sub-chunks (each on its DISJOINT QP group => disjoint WQ/CQ => no
      // shared-completion race, bit-exact); the per-p drain->recv sequence is the
      // natural inter->intra dependency for that sub-chunk, not a cross-p serialize.
      // ONLY safe when sw>=P (disjoint QP group per sub-chunk) AND P<=warpsPerBlock
      // (a leader warp exists for every p). Else fall back to the serial thread-0
      // drain (P>sw groups WRAP grpBase=p%sw sharing a QP; or too few warps).
      if (sw >= P && P <= warpsPerBlock && !dpSerialDrain) {
        // T72 (A): PARALLELISE the per-sub-chunk QP-GROUP send-CQ drain. Each
        // temporal sub-chunk p fans its data across g = sw/P QPs (grpBase(p)..
        // +g-1). The prior parallel path still had ONE leader warp per group
        // drain all g CQs SERIALLY (for q<g: ShmemQuietThread), so the exposed
        // per-op ring-completion poll was g-wide serial -- and at the crown's
        // weak sizes it is WIDEST there: DEEP_PIPE=auto(16MiB sub) gives P=1 at
        // 32/64/128MB total (g == sw == 8, EIGHT serial CQ polls on one thread)
        // and P=2 at 256MB (g=4). That serial poll is the inter-node ring-completion
        // residual. Give each of the g QPs
        // its OWN drain warp so the g completion polls RUN CONCURRENTLY, then a
        // lock-free per-group join (shared arrival counter) lets the group leader
        // AMO the remote flag + publish only AFTER all g QPs of THAT group have
        // landed. Distinct groups keep firing independently (per-group counter,
        // no global barrier) so the inter-sub-chunk pipeline the P>1 case relies
        // on is preserved. BIT-EXACT by construction: identical set of QP drains,
        // identical AMO/flag slots, the AMO still strictly follows ALL g landings
        // of its group (same landing->consume order); only the ORDER of the g
        // independent per-QP completions within a group is relaxed -- exactly the
        // relaxation the cross-group parallel path already sanctioned (each QP has
        // its own WQ/CQ, per-slot flags allow it). warpsPerBlock>=sw here (useWarps
        // clamped to warpsPerBlock), so a distinct drain warp exists per QP.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        // T73 (A): SINGLE-GROUP FAST JOIN. At the crown's weak sizes DEEP_PIPE=auto
        // collapses to P==1 (per-PE 4/8/16MB @ 32/64/128MB total => round(perPE/16MiB)
        // == 1), so g == sw == 8 and ALL sw drain warps belong to ONE group p==0. The
        // general per-group counter join then makes 8 warps atomicAdd the SAME shared
        // slot dpGrpDrained[0] and the leader SPIN-loads it to g -- 8-way atomic
        // write contention + a busy-spin where a plain block barrier already proves
        // "all g QP CQs drained + fenced". Since P==1 has exactly one group spanning
        // the whole block, __syncthreads is the natural (and cheaper) join: every
        // drain warp quiets its QP + __threadfence_system BEFORE the barrier, so after
        // it thread 0 AMOs the remote flag / spins the inbound landing flag / publishes.
        // BIT-EXACT by construction: identical QP-drain set, identical single AMO/flag
        // slot, the AMO still strictly follows ALL g landing fences (barrier orders
        // every drain+fence before the AMO) -- landing->consume order unchanged; only
        // the P==1 join primitive changes (block barrier vs atomic-counter spin). The
        // P>1 pipeline KEEPS the per-group counter join below (a block barrier there
        // would cross-synchronize independent groups and serialize the T72 pipeline).
        if (P == 1) {
          if (warpLead && myWarp < sw && nonEmptyDP(0)) {
            shmem::ShmemQuietThread(nextPeer, grpBase(0) + myWarp);  // g==sw, wl==myWarp
            __threadfence_system();                                 // this QP's bytes visible
          }
          __syncthreads();  // all g QP drains + fences complete (uniform block join)
          if (threadLinearId == 0 && nonEmptyDP(0)) {
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
                flagsObj, (flagBase + 0) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
                nextPeer);
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + 0) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + 0, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
          }
          __syncthreads();  // keep the block uniform before the shared trailing logic
        } else {
        __shared__ unsigned int dpGrpDrained[64];  // arrivals per sub-chunk group
        for (int i = threadLinearId; i < P; i += threadsPerBlock) dpGrpDrained[i] = 0u;
        __syncthreads();
        // Drain warp d (d in [0,sw)) -> group p = d/g, lane wl = d%g, drains QP
        // grpBase(p)+wl. All g lanes of a non-empty group participate (an empty
        // per-lane tile still drains a no-op CQ, so the count always reaches g).
        if (warpLead && myWarp < sw) {
          int d = myWarp;
          int p = d / g;
          int wl = d % g;
          if (p < P && nonEmptyDP(p)) {
            shmem::ShmemQuietThread(nextPeer, grpBase(p) + wl);
            __threadfence_system();                 // this QP's landed bytes visible
            atomicAdd(&dpGrpDrained[p], 1u);         // signal group arrival
            if (wl == 0) {
              // Group leader: wait for all g QPs of this group to land, then AMO
              // the remote flag + spin our own inbound flag + publish. atomicAdd(.,0)
              // is a well-defined atomic load of the shared arrival counter.
              long long gspin = 0;
              while (atomicAdd(&dpGrpDrained[p], 0u) < static_cast<unsigned int>(g)) {
                if (++gspin > 10000000000LL) __builtin_trap();
              }
              shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
                  flagsObj, (flagBase + p) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
                  nextPeer);
              long long spin = 0;
              while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                     static_cast<uint64_t>(1)) {
                // Landing fence MUST wait for the group to land; a timeout escape
                // that published chunkReadyFlags[p] anyway would allow a stale-read
                // (R188-R191). Abort loudly instead of silently corrupting bytes.
                if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
              }
              __threadfence_system();
              core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
            }
          }
        }
        }  // end else (P>1 per-group counter join; P==1 uses the block-barrier fast join)
      } else if (sw <= warpsPerBlock && !dpSerialDrain) {
        // WRAP PARALLEL DRAIN (T23 A): P>sw so sub-chunks share QPs (grpBase=p%sw),
        // but the sw QP GROUPS are DISJOINT. The old fallback serialized ALL P
        // drain->AMO on thread 0 (16 CQ-drains back-to-back @256MB w16 numQp=4,P=16)
        // and all P recv-publishes on warp 1 -- exposing the drain latency ON TOP of
        // the raw NIC-fill wall. Give each QP group w in [0,sw) its OWN merged
        // leader warp (thread w*warpSize): it walks ITS sub-chunks p = w, w+sw, ...
        // in increasing-p order, draining QP w (in-order per QP == landing proof),
        // AMOing remote flag p, then waiting on its own incoming flag p and
        // publishing chunkReadyFlags[p]. The sw leaders run CONCURRENTLY on disjoint
        // QP groups (disjoint WQ/CQ => no shared-completion race). BIT-EXACT: same
        // drains, same AMOs, same flag slots, per-QP completion order preserved;
        // only the ORDER of independent cross-group completions is relaxed, exactly
        // as the sw>=P parallel path above already relies on (per-slot flags allow
        // it). Merged send-drain+recv-publish into ONE leader per group (not two) to
        // stay within warpsPerBlock on wave64. Requires sw<=warpsPerBlock (a leader
        // warp per QP group); else fall to the serial thread-0 drain below.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        if (warpLead && myWarp < sw) {
          const int w = myWarp;
          for (int p = w; p < P; p += sw) {
            if (!nonEmptyDP(p)) continue;
            shmem::ShmemQuietThread(nextPeer, grpBase(p));   // grpBase(p)==p%sw==w
            __threadfence_system();
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
                flagsObj, (flagBase + p) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer);
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
          }
        }
      } else if (threadLinearId == 0) {
        // WRAP fallback (P>sw AND sw>warpsPerBlock): serial drain, groups share QPs.
        for (int p = 0; p < P; ++p) {
          if (!nonEmptyDP(p)) continue;
          shmem::ShmemQuietThread(nextPeer, grpBase(p));
          __threadfence_system();
          shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
              flagsObj, (flagBase + p) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer);
        }
      } else if (threadLinearId == warpSize) {
        for (int p = 0; p < P; ++p) {
          if (nonEmptyDP(p)) {
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
        }
      }
      __syncthreads();
      // T71 (A): SKIP the redundant trailing full-QP re-drain on the PARALLEL
      // deepPipeQuiet paths. The two parallel branches above (sw>=P && P<=warps,
      // and the sw<=warps WRAP path) give EVERY temporal sub-chunk its own leader
      // that ShmemQuietThread(nextPeer, qp)-drains its QP group BEFORE that group's
      // AMO -- and the union of the groups covers EVERY send QP this op used (the
      // sender only issues on qp ∈ groups). So by the time all P leaders have
      // published chunkReadyFlags, all our send WQEs have already completed =>
      // the buffer-reuse safety this trailing ShmemQuietThread(nextPeer) provides
      // is ALREADY satisfied. It is a serial full-numQpPerPe CQ poll on thread 0
      // re-draining already-empty CQs -- pure fixed per-op tail latency (part of
      // the inter-node ring-completion residual) with no effect on the byte image or
      // the landing->consume order. Bit-exact by construction; removed on the
      // parallel paths only. The serial-fallback branches (dpSerialDrain, or
      // sw>warpsPerBlock) KEEP the trailing drain unchanged. Env
      // MORI_HIER_DP_KEEP_TAIL_DRAIN restores it as an escape hatch is N/A here
      // (no env in kernel); the guard is a pure uniform condition.
      const bool dpParallelDrained =
          !dpSerialDrain && ((sw >= P && P <= warpsPerBlock) || (sw <= warpsPerBlock));
      if (!dpParallelDrained) {
        if (threadLinearId == 0) {
          shmem::ShmemQuietThread(nextPeer);
        }
        __syncthreads();
      }
      for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      return;
    }
    // DEEP_PIPE_IMM: per-sub-chunk landing fence via RDMA_WRITE_WITH_IMM instead of
    // put-with-signal. The put-signal AMO can beat its own data on large / many-in-
    // flight transfers (a device flag does NOT order a large RDMA landing -- fails
    // bit-exact >=64MB/P4), whereas a recv-CQE is produced ONLY after the write DMA
    // has landed globally (RC transport guarantee, in-order per QP), so it is a
    // definitive per-sub-chunk landing fence -> bit-exact even at 466MB E2E.
    // Pre-post P recv WQEs on each QP w in [0,sw) so every remote WRITE_WITH_IMM
    // yields a recv-CQE; the entry barrier in prepare orders every PE's post before
    // any peer's write. bytes=0: a pure write-with-imm does not consume the recv
    // SGL for payload (data lands at the write's addr/rkey); the WQE only makes the
    // CQE. On RC per QP, sub-chunk p's CQE precedes p+1's, so polling one CQE per
    // active QP for p, in order, republishes chunkReadyFlags[p] in temporal order.
    if (deepPipeImm) {
      if (threadLinearId == 0) {
        for (int w = 0; w < sw; ++w) {
          int cnt = 0;
          for (int p = 0; p < P; ++p)
            if (activeOf(p) > w) ++cnt;
          if (cnt > 0) {
            shmem::ShmemPostRecvImm(reinterpret_cast<uintptr_t>(memObj->localPtr), memObj->lkey,
                                    /*bytes=*/0, /*count=*/static_cast<uint32_t>(cnt), prevPeer, w);
          }
        }
      }
      __syncthreads();
    }
    // SENDER: warps [0,sw). For each temporal sub-chunk p, warp w sends its
    // 16B-aligned tile on qpId=w. All P sub-chunks ride qpId=w back-to-back (deep
    // SQ) so p's data lands before p+1's -- temporal landing order preserved.
    // IMM path: RDMA_WRITE_WITH_IMM (imm = p+1). Signal path: put-with-signal AMO.
    if (warpId < sw) {
      for (int p = 0; p < P; ++p) {
        size_t sU, eU;
        dpRange(p, sU, eU);
        if (sU >= eU) break;
        size_t subUnits = eU - sU;
        size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
        if (upw == 0) upw = 1;
        size_t wS = sU + static_cast<size_t>(warpId) * upw;
        size_t wE = wS + upw;
        if (wE > eU) wE = eU;
        if (wS >= wE) continue;
        size_t so = wS * kAlignDP;
        size_t eo = wE * kAlignDP;
        if (eo > peChunkSize) eo = peChunkSize;
        size_t off = chunkBaseOffsetSend + so;
        if (deepPipeImm) {
          shmem::ShmemPutMemImmWarp(memObj, off, memObj, off, eo - so,
                                    static_cast<uint32_t>(p + 1), nextPeer, warpId);
        } else if (wqeDepth <= 1 || (eo - so) <= kAlignDP) {
          shmem::ShmemPutMemNbiSignalWarp(memObj, off, memObj, off, eo - so, flagsObj,
                                          (flagBase + p) * sizeof(uint64_t), 1,
                                          core::atomicType::AMO_ADD, nextPeer, warpId);
        } else {
          // Deep SQ on the crown put-with-signal send (MORI_HIER_WQE_DEPTH). The crown's
          // deep-pipe send issues one whole-tile put-with-signal WQE per (sub-chunk p,
          // warp==QP), so each QP holds a single in-flight data WQE per sub-chunk. This
          // splits this warp's per-QP tile of sub-chunk p into up to wqeDepth back-to-back
          // 16B-aligned WRITEs on qpId=warpId: the first are plain non-blocking puts, the
          // last carries the put-with-signal AMO. RC in-order per QP, so the flag-p AMO
          // still lands strictly after all data WQEs of this tile, each warp signals slot
          // p exactly once (receiver's >=active-signals wait unchanged), and the sub-WQEs
          // tile [off,off+bytes) exactly -- only the per-QP SQ depth grows. In practice
          // extra WQEs add per-descriptor overhead that hurts the largest buffer without
          // raising wire fill, so this is kept default-off (WQE_DEPTH=1 => this branch
          // never taken => byte-identical to the single-WQE crown).
          const size_t tileBytes = eo - so;
          const size_t nUw = (tileBytes + kAlignDP - 1) / kAlignDP;
          size_t uPer = (nUw + static_cast<size_t>(wqeDepth) - 1) / static_cast<size_t>(wqeDepth);
          if (uPer == 0) uPer = 1;
          const int nWqe = static_cast<int>((nUw + uPer - 1) / uPer);
          for (int d = 0; d < nWqe; ++d) {
            size_t sUw = static_cast<size_t>(d) * uPer;
            if (sUw >= nUw) break;
            size_t eUw = sUw + uPer;
            if (eUw > nUw) eUw = nUw;
            size_t bo = sUw * kAlignDP;
            size_t be = eUw * kAlignDP;
            if (be > tileBytes) be = tileBytes;
            if (d == nWqe - 1) {
              shmem::ShmemPutMemNbiSignalWarp(memObj, off + bo, memObj, off + bo, be - bo, flagsObj,
                                              (flagBase + p) * sizeof(uint64_t), 1,
                                              core::atomicType::AMO_ADD, nextPeer, warpId);
            } else {
              shmem::ShmemPutMemNbiWarp(memObj, off + bo, memObj, off + bo, be - bo, nextPeer,
                                        warpId);
            }
          }
        }
      }
    }
    // RECEIVER: republish chunkReadyFlags[p] in temporal order so the reassembly
    // worker can push sub-chunk p while later sub-chunks still cross the NIC.
    // IMM path: poll one recv-CQE per active QP for p (RC in-order per QP =>
    // temporal order preserved), then publish. Signal path: spin the flag sum.
    if (deepPipeImm) {
      if (threadLinearId == warpSize) {
        for (int p = 0; p < P; ++p) {
          int active = activeOf(p);
          for (int w = 0; w < active; ++w) {
            shmem::ShmemPollRecvCqImm(prevPeer, w);
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
        }
      }
    } else if (threadLinearId == 0) {
      for (int p = 0; p < P; ++p) {
        int active = activeOf(p);
        if (active > 0) {
          long long spin = 0;
          while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                 static_cast<uint64_t>(active)) {
            if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
          }
        }
        __threadfence_system();
        core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
      }
    }
    __syncthreads();
    // T14 (A): OVERLAP the buffer-reuse send-QP drain (thread 0) with the recv
    // flag-slot reset (threads>0) under ONE trailing block barrier, instead of the
    // old drain-barrier-reset-barrier (3 tail joins -> 2 on the signal path). The
    // two touch DISJOINT state: thread 0 drains only its LOCAL send CQs
    // (ShmemQuietThread(nextPeer), buffer-reuse safety), while threads>0 zero
    // flagsArray[flagBase+idx] which NO reader reads after the __syncthreads above
    // (chunkReadyFlags were already published there). The entry barrier in prepare
    // still orders every PE's reset before any peer's next-op AMO, so the reset is
    // safe; the single trailing join guarantees BOTH the drain and the reset finish
    // before the fence/return. Removes one block barrier on the exposed small-buffer
    // per-op tail (the 32MB fixed-cost floor). Bit-exact by construction: identical
    // set of QP drains + identical flag zeroing, only overlapped with one fewer join.
    if (!deepPipeImm) {
      for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
    }
    if (threadLinearId == 0) {
      shmem::ShmemQuietThread(nextPeer);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    return;
  }
  // ==========================================================================

  for (int i = 0; i < maxRounds; i++) {
    // Chunk slots are indexed by ring position, not global PE.
    int sendDataRank = (ringPos - i + ringSize) % ringSize;
    int recvDataRank = (ringPos - i - 1 + ringSize) % ringSize;

    size_t chunkBaseOffset = static_cast<size_t>(sendDataRank) * peChunkSize;

    if (useReadRing) {
      // PULL: read prevPeer's own chunk (slot recvDataRank, i.e. prevPeer's OWN
      // ring slot, present after the intra prepare barrier) into our matching
      // slot at the SAME offset. Fan the read across the same numQp QPs the push
      // path uses (warp w -> qpId=w, disjoint 16B-aligned sub-ranges) so the
      // union tiles the chunk exactly -> byte-identical to a push receive.
      size_t readBase = static_cast<size_t>(recvDataRank) * peChunkSize;
      int rWarps = fanOut ? useWarps : 1;
      if (warpId < rWarps) {
        const size_t kAlign = 16;
        size_t nUnits = (peChunkSize + kAlign - 1) / kAlign;
        size_t unitsPerWarp = (nUnits + rWarps - 1) / rWarps;
        size_t startUnit = static_cast<size_t>(warpId) * unitsPerWarp;
        size_t endUnit = startUnit + unitsPerWarp;
        if (endUnit > nUnits) endUnit = nUnits;
        if (startUnit < endUnit) {
          size_t subStart = startUnit * kAlign;
          size_t subEnd = endUnit * kAlign;
          if (subEnd > peChunkSize) subEnd = peChunkSize;  // clamp tail
          shmem::ShmemGetMemNbiWarp(memObj, readBase + subStart, memObj, readBase + subStart,
                                    subEnd - subStart, prevPeer, warpId);
        }
      }
      __syncthreads();
      if (threadLinearId == 0) {
        // Drain ALL QPs' READ completions -> every sub-range has physically
        // landed in this PE's ring buffer (a READ CQE is produced only after the
        // response payload is written locally). System fence publishes the landed
        // bytes to this GPU's CUs for the subsequent reassembly / copy-out.
        shmem::ShmemQuietThread(prevPeer);
        __threadfence_system();
      }
      __syncthreads();
      continue;
    }
    if (multiBlockRead) {
      // MULTI-BLOCK PULL: THIS channel CTA reads its OWN sub-range
      // [blkOff,blkBytes) of prevPeer's chunk (slot recvDataRank, present after
      // the intra prepare barrier) into our matching slot on qpId=bid. Warp 0
      // issues; the union of all CTAs' READs tiles the chunk exactly => byte-
      // identical to the multiBlock push receive. Drain ONLY qpId=bid's READ
      // completion (per-channel, RCCL-style: draining all QPs from every CTA
      // would race the same CQ across blocks) -- the READ CQE proves this sub-
      // range physically landed, so the system fence publishes coherent bytes to
      // this GPU's CUs for the subsequent (same-CTA/dedicated) reassembly push,
      // with NO flag AMO, NO receiver spin, NO host sync. The post-loop
      // chunkReadyFlags[bid] publish (below) then releases the reassembly reader.
      if (warpId == 0 && blkBytes > 0) {
        size_t readOff = static_cast<size_t>(recvDataRank) * peChunkSize + blkOff;
        shmem::ShmemGetMemNbiWarp(memObj, readOff, memObj, readOff, blkBytes, prevPeer, bid);
      }
      __syncthreads();
      if (threadLinearId == 0 && blkBytes > 0) {
        shmem::ShmemQuietThread(prevPeer, bid);
        __threadfence_system();
      }
      __syncthreads();
      continue;
    }
    if (multiBlockWrite) {
      // MULTI-BLOCK WRITE-PUSH + SEND-CQ landing fence. Warp 0 pushes THIS
      // channel's sub-range [blkOff,blkBytes) of my chunk (chunkBaseOffset) to
      // nextPeer on qpId=bid as ONE fused put-with-signal: the data WRITE and the
      // flag AMO_ADD(1) ride the SAME QP, RC-ordered, so on the responder the sub-
      // range is globally visible BEFORE its own +1 fires -- the receiver's per-
      // channel flag can never beat the data (bit-exact by construction).
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        shmem::ShmemPutMemNbiSignalWarp(
            memObj, subOff, memObj, subOff, blkBytes, flagsObj,
            (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer,
            bid);
      }
      __syncthreads();
      // T40b: NO explicit send-CQ quiet. The fused put-signal already carries the
      // flag AMO as the last WQE on qpId=bid strictly AFTER the data WRITE (RC in-
      // order), so the receiver observing the per-channel flag is ALREADY
      // guaranteed the sub-range has landed globally -- an explicit
      // ShmemQuietThread(nextPeer,bid) here would only STALL the channel until its
      // send CQ empties (the exact single-round latency the put-signal was built to
      // remove; T40a measured that stall = 123 GB/s / 0.70x, the same underfill as
      // the RDMA-READ pull). Receiver: spin THIS channel's inbound flag (peer's CTA
      // bid bumped slot recvDataRank via its fused signal); system acquire + fence
      // makes the landed sub-range coherently visible to this GPU's CUs. The post-
      // loop chunkReadyFlags[bid] publish then releases the reassembly reader.
      if (threadLinearId == warpSize && blkBytes > 0) {
        int spinCount = 0;
        while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + recvDataRank) < 1) {
          if (++spinCount > 10000000) break;
        }
        __threadfence_system();
      }
      __syncthreads();
      continue;
    }
    // NOTE (M4, ): tried splitting this put across ALL 8 warps in the
    // block (disjoint 16B-aligned sub-ranges) to parallelize the transfer.
    // VALIDATED NEGATIVE RESULT on device (single-node ring, GPUs 0-3): it
    // crashes with anvil "submitPacket: Retry limit exceeded" -> GPU coredump.
    // Cause: the same-node neighbour path of ShmemPutMemNbiWarp lowers to one
    // anvil SDMA queue per (src,dst) pair; all warps target the SAME nextPeer
    // chunk region, so 8 warps hammer ONE SDMA queue and overflow its retry
    // budget. Multi-warp puts only help when each warp drives a distinct queue/
    // QP (as the intra SDMA gather does, one warp per peer). For a single-peer
    // ring round there is one queue, so a single warp is correct. Kept as-is.
    // NOTE (M4, ): this put is the DOMINANT cost of the xnode hier
    // AllGather. True 2-node bench (n09-21+n09-29, fp32 64MiB/rank, world=8
    // N=2,G=4, >=3 reps) of the  overlap commit (4a2feeb9):
    //   mori min=13.59ms 39.5 GB/s  vs  rccl min=3.55ms 151.1 GB/s (~3.8x).
    // Within noise of the  baseline (40.3 GB/s) => the quiet/spin
    // overlap is a NO-OP at this size: the round is BANDWIDTH-bound, not
    // latency-bound. ROOT CAUSE of the under-fill: this single warp issues
    // the entire chunk on a SINGLE QP. ShmemPutMemNbiWarp<RDMA> issues from
    // lane 0 with qpId defaulting to 0, while the transport provisions
    // numQpPerPe (default 4, MORI_NUM_QP_PER_PE) QPs/peer -- we use 1 of 4.
    // NEXT LEVER: fan the chunk across all numQpPerPe QPs (warp w -> disjoint
    // 16B-aligned sub-range on qpId=w) so multiple QPs drive the NIC in
    // parallel. CORRECTNESS GATE (must hold before landing): the flag bump
    // below must follow a quiet that drains ALL used QPs -- ShmemQuietThread
    // (int pe) (RDMA) loops qpId 0..numQpPerPe-1, whereas the current
    // ShmemQuietThread(nextPeer, memObj) SDMA-typed call must be confirmed to
    // cover every QP, else the receiver's flag fires before tail QPs land.
    // Gate fan-out on numQp>1 only for the all-RDMA sub-group ring; the flat
    // single-node ring (P2P, one anvil queue) must stay single-warp.
    // IMPLEMENTED: the runtime-gated fan-out described above.
    //
    // DEEP-SQ WQE-DEPTH (MORI_HIER_WQE_DEPTH, default 1). The plain quiet-drained
    // put path (below) issues ONE whole-sub-range RDMA-WRITE WQE per QP, so the SQ
    // holds a single in-flight WQE per QP -- adding QPs (numQp 4->8) was neutral,
    // so the un-probed axis is SQ DEPTH per QP. putDeep splits a sub-range into
    // `wqeDepth` back-to-back 16B-aligned non-blocking WRITEs on the SAME QP, so
    // the NIC sees `wqeDepth` queued WQEs per QP (device analogue of the host-proxy
    // deep-SQ that reaches ~48 GB/s vs ~31 GB/s single-WQE-per-QP). The union of
    // the d sub-puts tiles [off,off+bytes) EXACTLY and rides the QP in RC order, so
    // the byte image AND the per-QP quiet drain are identical; only SQ depth grows.
    // depth<=1 (or a sub-range too small to split) => single put = shipped path.
    auto putDeep = [&](size_t off, size_t bytes, int qp) {
      // DIRECT-LAND: retarget the DATA WRITE DEST from the receiver's ring slot to
      // its OWN final output self-slot (gOutMemObj at (sendDataRank*gGroupSize+
      // gGroupPos)*peChunkSize + local-sub); SOURCE stays this GPU's local ring
      // chunk (memObj@off). nextPeer is this GPU's same-groupPos counterpart, so
      // that offset is exactly where the reasm self-column push would have written
      // (bit-exact). The post-loop quiet+flag path is unchanged (drains the send =>
      // output landed remotely => flag). OFF => dest==ring (byte-identical crown).
      // DIRECT-LAND RKEY GUARD + one-time DIAGNOSTIC (T58). T56/T57 INFERRED the
      // cross-node RDMA WRITE to gOutMemObj drops for lack of a valid remote rkey
      // (register printf absent + control-works-via-IPC), but never MEASURED it at
      // the write site. Here we read the actual device SymmMemObj: if the dest is an
      // RDMA peer but has no valid peerRkeys[nextPeer], the WRITE would land nowhere
      // -> FALL BACK to the ring dest (byte-identical crown, no drop). The one-shot
      // printf reports rkey/ptr/offset so we can settle the root cause definitively.
      bool dlRkeyOk = true;
      if (directLand) {
        const application::SymmMemObj* g = gOutMemObj.gpu;
        dlRkeyOk = (g != nullptr) && (!peerIsRdma ||
                   (g->peerRkeys != nullptr && g->peerRkeys[nextPeer] != 0));
        if (threadIdx.x == 0 && off == chunkBaseOffset && qp == 0) {
          printf(
              "[DL] pe? sendRank=%d nextPeer=%d rdma=%d gOut=%p peerRkeys=%p "
              "rkey=%u peerPtr=%p size=%zu dBase=%zu peChunk=%zu\n",
              sendDataRank, nextPeer, (int)peerIsRdma, (void*)g,
              (g ? (void*)g->peerRkeys : nullptr),
              (g && g->peerRkeys ? g->peerRkeys[nextPeer] : 0u),
              (g && g->peerPtrs ? (void*)g->peerPtrs[nextPeer] : nullptr),
              (g ? g->size : (size_t)0),
              (static_cast<size_t>(sendDataRank) * static_cast<size_t>(gGroupSize) +
               static_cast<size_t>(gGroupPos)) * peChunkSize + (off - chunkBaseOffset),
              peChunkSize);
        }
      }
      const bool doDirect = directLand && dlRkeyOk;
      const application::SymmMemObjPtr dObj = doDirect ? gOutMemObj : memObj;
      const size_t dBase =
          doDirect ? ((static_cast<size_t>(sendDataRank) * static_cast<size_t>(gGroupSize) +
                         static_cast<size_t>(gGroupPos)) *
                            peChunkSize +
                        (off - chunkBaseOffset))
                     : off;
      if (wqeDepth <= 1 || bytes <= 16) {
        shmem::ShmemPutMemNbiWarp(dObj, dBase, memObj, off, bytes, nextPeer, qp);
        return;
      }
      const size_t kAlignW = 16;
      size_t nU = (bytes + kAlignW - 1) / kAlignW;
      size_t uPer = (nU + static_cast<size_t>(wqeDepth) - 1) / static_cast<size_t>(wqeDepth);
      if (uPer == 0) uPer = 1;
      for (int d = 0; d < wqeDepth; ++d) {
        size_t sU = static_cast<size_t>(d) * uPer;
        if (sU >= nU) break;
        size_t eU = sU + uPer;
        if (eU > nU) eU = nU;
        size_t so = sU * kAlignW;
        size_t eo = eU * kAlignW;
        if (eo > bytes) eo = bytes;
        shmem::ShmemPutMemNbiWarp(dObj, dBase + so, memObj, off + so, eo - so, nextPeer, qp);
      }
    };
    if (multiBlock) {
      // RCCL-style channel: this CTA puts only its sub-range [blkOff, blkOff+
      // blkBytes) of the chunk, on qpId=bid (a distinct QP per channel). Warp 0
      // issues; the union of all CTAs' sub-ranges tiles the chunk exactly =>
      // byte-identical result.
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        if (multiBlockWriteImm) {
          // THIS channel's sub-range as RDMA_WRITE_WITH_IMM on qpId=bid. The
          // receiver's per-channel recv-CQE proves it landed globally -- no quiet,
          // no flag AMO, no host sync. imm carries the chunk id for validation.
          shmem::ShmemPutMemImmWarp(memObj, subOff, memObj, subOff, blkBytes,
                                    static_cast<uint32_t>(sendDataRank + 1), nextPeer, bid);
        } else if (multiBlockSignal) {
          // Fuse THIS channel's data WRITE + its flag AMO_ADD(1) on qpId=bid. RC
          // in-order => the sub-range lands remotely before its own +1 fires, so
          // the receiver's per-channel flag never beats the data. No separate
          // quiet, no separate AMO (skipped in the completion block below).
          shmem::ShmemPutMemNbiSignalWarp(
              memObj, subOff, memObj, subOff, blkBytes, flagsObj,
              (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer,
              bid);
        } else {
          putDeep(subOff, blkBytes, bid);
        }
      }
    } else if (fanOut) {
      // Split the chunk into ``useWarps`` disjoint 16B-aligned sub-ranges; warp
      // w drives its sub-range on qpId=w. The union tiles the chunk exactly (the
      // last warp absorbs the unaligned tail), so the byte image is identical to
      // a single whole-chunk put -- only the QP fan-out differs.
      if (warpId < useWarps) {
        const size_t kAlign = 16;
        size_t nUnits = (peChunkSize + kAlign - 1) / kAlign;  // # of 16B units
        size_t unitsPerWarp = (nUnits + useWarps - 1) / useWarps;
        size_t startUnit = static_cast<size_t>(warpId) * unitsPerWarp;
        size_t endUnit = startUnit + unitsPerWarp;
        if (endUnit > nUnits) endUnit = nUnits;
        if (startUnit < endUnit) {
          size_t subStart = startUnit * kAlign;
          size_t subEnd = endUnit * kAlign;
          if (subEnd > peChunkSize) subEnd = peChunkSize;  // clamp tail
          size_t subOff = chunkBaseOffset + subStart;
          if (fanOutWriteImm) {
            // THIS warp's sub-range as RDMA_WRITE_WITH_IMM on ITS OWN QP
            // (qpId=warpId). The receiver's per-QP recv-CQE proves this sub-range
            // has landed globally -- no quiet, no flag AMO, no host sync.
            shmem::ShmemPutMemImmWarp(memObj, subOff, memObj, subOff, subEnd - subStart,
                                      static_cast<uint32_t>(sendDataRank + 1), nextPeer, warpId);
          } else if (fanOutSignal) {
            // Fuse THIS warp's data WRITE + a flag AMO_ADD(1) on ITS OWN QP
            // (qpId=warpId). RC in-order => this QP's data lands remotely before
            // its +1 fires. Receiver waits for the sum (fanActive) so it proceeds
            // only after EVERY QP's data has landed -- no separate quiet, no host
            // sync, ring<->gather overlap preserved.
            // DIRECT-LAND: retarget the DATA WRITE from the receiver's ring slot to
            // the receiver's OWN final output self-slot (gOutMemObj at
            // (sendDataRank*gGroupSize+gGroupPos)*peChunkSize) -- since nextPeer is
            // this GPU's same-groupPos counterpart on the sender's node, that offset
            // is exactly where the reasm worker's self-column push would have written
            // it (bit-exact). The flag AMO still rides flagsObj (chunkReadyFlags), so
            // the landing signal path is unchanged. src stays this GPU's ring slot.
            if (directLand) {
              size_t outOff =
                  (static_cast<size_t>(sendDataRank) * static_cast<size_t>(gGroupSize) +
                   static_cast<size_t>(gGroupPos)) *
                      peChunkSize +
                  subStart;
              // Signature is (dest, destOff, source, srcOff, ...): DEST is the
              // receiver's OWN output self-slot (gOutMemObj@outOff on nextPeer),
              // SOURCE is this GPU's local ring chunk (memObj@subOff).
              shmem::ShmemPutMemNbiSignalWarp(
                  gOutMemObj, outOff, memObj, subOff, subEnd - subStart, flagsObj,
                  (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
                  nextPeer, warpId);
            } else {
              shmem::ShmemPutMemNbiSignalWarp(
                  memObj, subOff, memObj, subOff, subEnd - subStart, flagsObj,
                  (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
                  nextPeer, warpId);
            }
          } else {
            putDeep(subOff, subEnd - subStart, warpId);
          }
        }
      }
      // All fan-out warps must finish ISSUING their puts before thread 0 drains
      // the QPs and bumps the flag, else the receiver's flag could fire before a
      // tail QP's data lands. (Only added on the fan-out path -- the single-warp
      // path keeps the  thread schedule unchanged.)
      __syncthreads();
    } else if (warpId == 0) {
      if (writeImm) {
        // RDMA_WRITE_WITH_IMM: data WRITE + a 32-bit immediate on ONE QP. On RC
        // the responder produces the recv-CQE only AFTER the payload DMA has
        // landed globally, so the receiver (polling that CQE below) is guaranteed
        // coherent data with NO quiet, NO flag AMO, NO host sync. imm carries the
        // chunk id (sendDataRank+1) for optional validation.
        //
        // DIRECT-LAND (T64 A): the plain-put + send-CQE-quiet direct-land (T56-T58)
        // MISMATCHED at exactly the NIC-landed output self-slot (rank m*groupSize+
        // groupPos) because ShmemQuietThread drains the SEND completion, which does
        // NOT prove the WRITE's payload physically reached the REMOTE HBM before the
        // landing flag fired -> the reasm SDMA read raced the still-draining write
        // and consumed STALE bytes. WRITE_WITH_IMM closes that race STRUCTURALLY: the
        // receiver's recv-CQE (polled below at threadLinearId==warpSize) is produced
        // by the responder ONLY after the payload DMA has landed globally, and the
        // per-chunk chunkReadyFlags publish (end of body) runs AFTER that CQE poll +
        // __threadfence_system + __syncthreads -- so the reasm worker's landing wait
        // now gates on a TRUE remote-landing proof, not a send drain. This is the
        // receiver-side RDMA-completion gate (NOT the banned coherence fence-type
        // knob). Retarget only the DEST to the receiver's output self-slot; SOURCE
        // stays this GPU's local ring chunk. OFF => byte-identical crown.
        if (directLand) {
          const size_t outOff =
              (static_cast<size_t>(sendDataRank) * static_cast<size_t>(gGroupSize) +
               static_cast<size_t>(gGroupPos)) *
              peChunkSize;
          shmem::ShmemPutMemImmWarp(gOutMemObj, outOff, memObj, chunkBaseOffset, peChunkSize,
                                    static_cast<uint32_t>(sendDataRank + 1), nextPeer);
        } else {
          shmem::ShmemPutMemImmWarp(memObj, chunkBaseOffset, memObj, chunkBaseOffset, peChunkSize,
                                    static_cast<uint32_t>(sendDataRank + 1), nextPeer);
        }
      } else if (usePutSignal && peerIsRdma) {
        // FLAG-CAN'T-BEAT-DATA (transport-level): fuse the data WRITE and the
        // completion-flag AMO into ONE ShmemPutMemNbiSignal so the signal WQE
        // rides the SAME QP strictly AFTER the data WRITE. On RC the responder
        // executes them in order and the WRITE's data is globally visible before
        // the atomic -- so the receiver observing the flag is GUARANTEED the
        // remote-half bytes have physically landed, with NO host sync (keeps the
        // ring<->gather overlap). This replaces the separate put + quiet + AMO
        // whose AMO could land before the (independently-drained) data on a race.
        shmem::ShmemPutMemNbiSignalWarp(
            memObj, chunkBaseOffset, memObj, chunkBaseOffset, peChunkSize, flagsObj,
            (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer);
      } else {
        putDeep(chunkBaseOffset, peChunkSize, 0);
      }
    }

    // M4: overlap the OUTBOUND drain (quiet + flag bump to nextPeer)
    // with the INBOUND recv-flag wait (from prevPeer). These two waits are on
    // independent network directions, but the previous code serialized them on
    // thread 0 (quiet+bump, syncthreads, then spin) so their latencies added.
    // Split them across two threads so they proceed concurrently; the single
    // trailing __syncthreads makes both finish before the round ends (this also
    // drops one __syncthreads/round vs the old two-phase form).
    // NOTE: true round-level pipelining (issuing the next round's put during
    // this recv-wait) is IMPOSSIBLE here -- round i+1 sends sendDataRank =
    // (ringPos-i-1), which is EXACTLY the recvDataRank received in round i, a
    // hard data dependency (you forward onward precisely what you just got).
    bool signalFused = (usePutSignal && peerIsRdma && !multiBlock && !fanOut);
    if (threadLinearId == 0 &&
        (signalFused || fanOutSignal || multiBlockSignal || multiBlockWriteImm || writeImm || fanOutWriteImm)) {
      // The put-with-signal path already carried the completion flag as the last
      // WQE on the data QP (RC-ordered after the data WRITE) -- no separate quiet
      // or AMO is needed. Skipping them is what removes the flag-beats-data race.
      // On the fan-out path EVERY active warp already issued its own per-QP
      // signal (fanActive of them), so thread 0 issues no extra AMO here.
      // On the WRITE_WITH_IMM path the completion is signalled by the recv-CQE
      // (polled below), so likewise no quiet + no flag AMO.
    } else if (threadLinearId == 0) {
      // Drain the outbound put before bumping the receiver's flag. On the
      // fan-out path the put used numQp RDMA QPs, so we must quiet ALL of them:
      // ShmemQuietThread(pe) (RDMA) loops qpId 0..numQpPerPe-1. The single-warp
      // path keeps the original SDMA/P2P-typed quiet (memObj overload).
      if (multiBlock) {
        // This CTA put on exactly qpId=bid; drain ONLY that QP. Draining ALL QPs
        // (ShmemQuietThread(pe)) from every block would poll the same completion
        // queues concurrently across CTAs and race. Per-QP quiet keeps each
        // channel independent (RCCL-style), and this block bumps only its own
        // flag region, so the receiver's per-block flag fires only after THIS
        // block's data has landed.
        shmem::ShmemQuietThread(nextPeer, bid);
      } else if (fanOut) {
        // Fan-out issued from numQp QPs within ONE block; ShmemQuietThread(pe)
        // drains ALL QPs so the receiver's flag never fires before a tail QP's
        // data lands.
        shmem::ShmemQuietThread(nextPeer);
      } else if (peerIsRdma) {
        // CORRECTNESS (flag-beats-data): for a CROSS-NODE (RDMA) neighbour the
        // single-warp put rode an RDMA QP, but the SDMA-typed memObj-overload
        // quiet drains only the P2P/SDMA path -- it does NOT wait for the RDMA
        // send-queue completion, so the flag AMO below can be issued (and land
        // remotely, RC-ordered on its own QP) BEFORE the data PUT has drained
        // -> the receiver observes the flag and reads STALE remote-half bytes
        // (the MI355 FSDP loss drift; in-situ probe: exactly the remote half).
        // Use the transport-aware quiet (RDMA -> loops all numQpPerPe QPs) so
        // the outbound put is fully drained before the flag fires. On-device,
        // no host sync -> keeps the ring<->gather overlap (perf) AND orders
        // remote landing (accuracy). Mirrors the fan-out path's RDMA quiet.
        shmem::ShmemQuietThread(nextPeer);
      } else {
        shmem::ShmemQuietThread(nextPeer, memObj);
      }
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
          flagsObj, (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
          nextPeer);
    } else if (threadLinearId == warpSize && recvFanOutWriteImm) {
      // Poll one recv-CQE per active QP: prev issued fanActive WRITE_WITH_IMMs
      // (one per QP), each CQE proving that QP's disjoint sub-range has landed
      // globally. After all fanActive complete, the whole chunk is coherent, so
      // the subsequent forward-put / copy-out reads correct bytes with no host
      // sync. __threadfence_system publishes them to this GPU.
      for (int q = 0; q < fanActive; ++q) {
        shmem::ShmemPollRecvCqImm(prevPeer, q);
      }
      __threadfence_system();
    } else if (threadLinearId == warpSize && recvWriteImm) {
      // WRITE_WITH_IMM receiver: block on the recv-CQ completion for this round's
      // inbound chunk instead of spinning the flag. Observing the CQE PROVES the
      // peer's payload has landed globally (the CQE is generated only after the
      // write DMA completes remotely) -- the transport-level guarantee no device
      // barrier/quiet gave. threadfence_system makes the landed bytes visible to
      // this GPU's subsequent forward-put / copy-out with no host sync.
      shmem::ShmemPollRecvCqImm(prevPeer, 0);
      __threadfence_system();
    } else if (threadLinearId == warpSize && recvMultiBlockWriteImm) {
      // MULTI-BLOCK channel receiver: consume THIS channel's recv-CQE on qpId=bid
      // instead of spinning the flag. The CQE is produced only AFTER prev's sub-
      // range WRITE DMA has landed globally (RC transport guarantee), so the
      // subsequent copy-OUT / remote reassembly reads coherent bytes with NO host
      // sync -- the device-side completion that closes the fuse_local E2E stale-
      // remote race the flag path could not. threadfence_system publishes to CUs.
      if (blkBytes > 0) {
        shmem::ShmemPollRecvCqImm(prevPeer, bid);
      }
      __threadfence_system();
    } else if (threadLinearId == warpSize) {
      // Each round the sender increments a DISTINCT flag slot (index
      // recvDataRank = sendDataRank on the receiver), so every slot is
      // incremented exactly once over the ringSize-1 rounds -- 0 -> 1.
      // Wait for THIS round's slot to become nonzero (not a cumulative count;
      // the previous "!= i+1" form only held for ringSize==2 / a single round).
      int spinCount = 0;
      // SYSTEM-scope acquire: the flag is bumped by a REMOTE peer's RDMA AMO and
      // the chunk it guards is landed by that peer's RDMA put -- both cross-agent
      // writes. A RELAXED load establishes NO happens-before with those data
      // writes, so observing the flag does NOT make the received chunk coherently
      // visible to this GPU's subsequent forward-put / copy-OUT -> the RDMA (remote)
      // half of the output reads STALE bytes under FSDP tight overlap (the
      // MI355-exposed loss drift; in-situ probe showed exactly the remote half
      // stale). A system-scope acquire + system threadfence makes the peer's prior
      // data writes visible without a host sync -- mirrors the intra SDMA gather's
      // proven AtomicLoadSeqCstSystem + __threadfence_system receiver pattern.
      // On the fan-out-signal path the sender adds +1 PER active QP, so wait for
      // the slot to reach ``expectedRecvSig`` (== fanActive); the classic path
      // adds exactly 1 (expectedRecvSig==1), so this is a strict superset that
      // stays byte-identical when signals are off.
      // GEN-RING: when opGen!=0 on the classic single-increment path the flags
      // ACCUMULATE across ops (no per-op reset), so slot k == opGen after opGen
      // ops. Wait for the slot to reach the current generation instead of 1.
      // Restricted to expectedRecvSig==1 (single AMO_ADD(1) per slot per op) so
      // the accumulation is uniform; fan-out/put-signal keep the classic path.
      const bool genRecv = (opGen != 0 && expectedRecvSig == 1);
      const uint64_t recvThreshold = genRecv ? opGen : (uint64_t)expectedRecvSig;
      while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + recvDataRank) <
             recvThreshold) {
        spinCount++;
        if (spinCount > 10000000) {  // Increased timeout threshold
          printf("ringPos %d: Timeout waiting from ringPos %d (round %d, slot=%llu<%d)\n", ringPos,
                 recvDataRank, i, (unsigned long long)core::AtomicLoadSeqCstSystem(
                                      flagsArray + flagBase + recvDataRank),
                 expectedRecvSig);
          break;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  }

  // PHASE-4 per-chunk landing publish. After the round loop this block's inbound
  // sub-range (channel ``bid``) is fully landed AND made visible to this GPU by
  // the receiver's __threadfence_system inside the loop. Publish the ready flag so
  // a concurrent reassembly block can start pushing exactly this sub-range over
  // XGMI without waiting for the whole ring / a global finish barrier. The store
  // is system-scoped and preceded by a fence so the reassembly reader observes the
  // landed bytes coherently. INERT when chunkReadyFlags==nullptr (every existing
  // caller) so the standalone ring stays byte-for-byte identical.
  if (chunkReadyFlags != nullptr && threadLinearId == 0) {
    __threadfence_system();
    core::AtomicStoreSeqCstSystem(chunkReadyFlags + bid, opGen ? opGen : static_cast<uint64_t>(1)); // GEN-TOKEN T28
  }
  // On a WRITE_WITH_IMM completion path the ring flags region
  // [flagBase, flagBase+ringSize) is NEVER touched this op: the sender emits
  // RDMA_WRITE_WITH_IMM (no flag AMO) and the receiver consumes the recv-CQE (no
  // flag spin), so every slot is still at its entry value (0). The per-op reset
  // loop + its trailing __threadfence_system are therefore dead work here. Because
  // useWriteImm is a persistent per-handle env (set once, same path every call),
  // the flags stay 0 across all ops on this handle -> skipping the reset is
  // byte-for-byte identical while removing per-AG overhead that under FSDP's many
  // AGs/step erodes the standalone WIN toward E2E parity. Non-writeImm paths keep
  // the exact reset+fence, so the default path is unchanged.
  // multiBlockRead/useReadRing bump NO ring flag (they PULL + drain the READ CQE),
  // so the flag region stays 0 all op -> its reset+fence is dead work; skip it too.
  bool usedWriteImm = (multiBlockWriteImm || writeImm || fanOutWriteImm || multiBlockRead ||
                       useReadRing);
  // GEN-RING: on the classic single-increment path the flags are intentionally
  // NOT reset -- they accumulate so slot k == opGen after opGen ops (the
  // receiver waits for >= opGen). Skipping the reset is what lets the prepare
  // entry barrier be dropped (no reset -> nothing to order against a peer's
  // next-op increment). Gen-mode only engages when the per-op increment is a
  // uniform +1 (expectedRecvSig==1, no put-signal/fan-out), matching the
  // receiver-side gate above; multiBlockSignal/fanOutSignal keep the reset.
  const bool genReset = (opGen != 0 && !usePutSignal);
  if (!usedWriteImm && !genReset) {
    // Each block resets ONLY its own flag region [flagBase, flagBase+ringSize) so
    // concurrent channels never race (single-block: flagBase=0, ringSize slots).
    for (int idx = threadLinearId; idx < ringSize; idx += threadsPerBlock) {
      flagsArray[flagBase + idx] = 0;
    }
    __syncthreads();
    if (threadLinearId == 0) {
      __threadfence_system();
    }
  }
}

// Whole-world ring (flat): position == global PE, stride 1, base 0.
inline __device__ void AllGatherRingKernelBody(int myPe, int npes,
                                               const application::SymmMemObjPtr memObj,
                                               const application::SymmMemObjPtr flagsObj,
                                               size_t peChunkSize) {
  AllGatherRingSubGroupKernelBody(myPe, npes, /*peBase=*/0, /*peStride=*/1, memObj, flagsObj,
                                  peChunkSize);
}

template <typename T>
__global__ void AllGatherRingKernel(int myPe, int npes, const application::SymmMemObjPtr memObj,
                                    const application::SymmMemObjPtr flagsObj) {
  // Existing executor path: chunk size derived from the (exactly npes*chunk)
  // buffer. Equal-sized chunks, so size/npes is exact.
  AllGatherRingKernelBody(myPe, npes, memObj, flagsObj, memObj->size / npes);
}

}  // namespace collective
}  // namespace mori
// ============================================================================
// Direct-land coherence note. Direct-land (the NIC writing straight into the output
// tensor) can leave the SDMA reassembly read stale: the blocker is SDMA-read coherence
// of the NIC-written output tensor, not a landing/ordering race. Fusing the flag AMO
// onto the same RC QP as the data write (so the flag executes remotely strictly after
// the payload lands in remote HBM) does not fix it, which shows the payload is already
// in remote HBM before the flag is observed. The residual staleness is the copy
// engine's read of the self-slot: the crown reads the ring buffer (fine-grained RDMA
// scratch, SDMA-read-coherent with the NIC write) and is correct, whereas direct-land
// reads the coarse-grained cached output tensor, whose line the NIC write does not
// invalidate for the copy engine. A recv-CQE (WRITE_WITH_IMM) gate would not fix this
// either, since a recv-CQE is only a landing proof, not a copy-engine cache
// invalidation. The writeImm direct-land dest-retarget above is committed but default
// OFF and cannot be exercised until recv-WQE posting is wired for the standalone AG.
// A fix would land the RDMA into fine-grained coherent staging that the SDMA
// reassembly read snoops.
// ============================================================================
