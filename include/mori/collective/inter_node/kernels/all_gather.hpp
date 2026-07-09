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

// Core ring AllGather data movement, parameterized on an explicit per-chunk
// byte size (``peChunkSize``). Factored out as a ``__device__`` helper so it
// can be reused both by the existing ``__global__`` entry point below (used by
// the inter-node executor, which derives the chunk size from the buffer) and by
// the JIT-launched hierarchical entry point (which passes an explicit chunk
// size so a single fixed-size symmetric ring buffer can serve variable message
// sizes). The ring buffer holds ``npes`` chunks of ``peChunkSize`` bytes; on
// entry only PE ``myPe``'s own chunk is filled. After ``npes-1`` rounds every
// PE holds all chunks in PE order. Equal per-PE chunks => no last-chunk special
// case is needed.
// Sub-group ring AllGather over an arithmetic sub-group of global PEs:
// ``{peBase, peBase+peStride, ..., peBase+(ringSize-1)*peStride}``. This PE is
// at position ``ringPos`` within that sub-group. The ring buffer holds
// ``ringSize`` chunks; on entry only slot ``ringPos`` is filled. After
// ``ringSize-1`` rounds every member holds all ``ringSize`` chunks in ring
// order. The whole-world ring is the special case ``peBase=0, peStride=1``.
//
// This is the inter-node phase of the hierarchical AllGather: the ring runs
// over node-leaders (or same-local-index ranks across nodes), so neighbours are
// reached over RDMA while same-node members go P2P -- exactly the DESIGN intent.
// ``numBlocksOverride`` / ``bidOverride`` let this body run as
// a SUB-RANGE of a larger fused grid. When >=0 they replace the grid-derived
// ``gridDim.x`` / ``blockIdx.x`` so the ring can occupy blocks [0, ringBlocks)
// of a fused launch while OTHER blocks of the same grid run the intra-node SDMA
// local-block gather CONCURRENTLY (NIC || XGMI in ONE kernel, no host-side
// wait_stream merge -- the fused recv+reassemble lever this work proved reaches
// RCCL parity, ported here. Both default to -1, preserving the
// historical grid-derived geometry BYTE-FOR-BYTE -- inert until a fused launcher
// passes them.
// ``chunkReadyFlags`` (default nullptr, INERT for every existing caller) is the
// per-chunk landing signal that enables PIPELINING the inter-node RDMA ring with
// the intra-node SDMA remote-block reassembly (SCOPE_A Phase 4). When non-null it
// is a device array of at least ``numBlocks`` uint64_t in ordinary (cached) HBM,
// zeroed by the caller before launch. As soon as THIS block's (channel ``bid``'s)
// inbound sub-range has fully landed in this PE's ring buffer -- i.e. after the
// receiver's system-scope acquire + __threadfence_system that already make those
// bytes coherently visible to this GPU's CUs -- block ``bid`` publishes
// ``chunkReadyFlags[bid] = 1``. A CONCURRENT reassembly block (in the same fused
// grid) spins on that flag and, the instant sub-range ``bid`` is ready, SDMA-
// pushes exactly that sub-range over XGMI while ring channel ``bid+1`` is still
// crossing the NIC. This overlaps the two hierarchy legs that today run as two
// serial phases (the 143-vs-168 GB/s gap). Because each PE reassembles a remote
// block by pushing FROM ITS OWN ring buffer, the only dependency is this PE's own
// ring landing -- a purely local flag spin, no global barrier. nullptr keeps the
// standalone ring byte-for-byte identical.
inline __device__ void AllGatherRingSubGroupKernelBody(
    int ringPos, int ringSize, int peBase, int peStride,
    const application::SymmMemObjPtr memObj, const application::SymmMemObjPtr flagsObj,
    size_t peChunkSize, int numQp = 1, int numBlocksOverride = -1, int bidOverride = -1,
    bool usePutSignal = false, bool useWriteImm = false, uint64_t* chunkReadyFlags = nullptr,
    uint64_t opGen = 0, bool useRead = false, int wqeDepth = 1, int deepPipe = 1,
    int deepPipeImm = 0, int deepPipeQuiet = 0) {
  int nextPos = (ringPos + 1) % ringSize;
  int nextPeer = peBase + nextPos * peStride;
  int maxRounds = ringSize - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  const int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  const int threadLinearId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int warpId = threadLinearId / warpSize;
  const int warpsPerBlock = threadsPerBlock / warpSize;

  // M4: MULTI-BLOCK ring ("channels", RCCL-style). The single-block
  // ring (numQp warps in ONE CTA, each on its own QP) saturates at ~63 GB/s vs
  // RCCL's ~150 (-18: numQp 4->8 gave 0 gain -> warps-in-one-CTA is
  // exhausted). RCCL instead drives many CTAs (channels) concurrently. Here each
  // block ``bid`` of ``gridDim.x`` handles a DISJOINT 16B-aligned sub-range of
  // EVERY chunk and uses its OWN flag region [bid*ringSize, (bid+1)*ringSize), so
  // blocks never alias data or flags. Block bid issues its put on qpId=bid so
  // each channel drives a distinct QP -- the union still tiles each chunk exactly
  // => byte-identical result. ONLY engaged for true RDMA neighbours (same-node
  // P2P/SDMA lowers to ONE anvil queue per (src,dst); multiple CTAs hammering it
  // overflow the retry budget and coredump, ). For a non-RDMA neighbour
  // (single-node simulation) only block 0 runs the proven single-block path and
  // the other blocks return, keeping single-node bit-exact + crash-free.
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

  // M4: multi-QP fan-out gate. The whole motivation (see the long
  // NOTE below) is that a single warp on a single QP under-fills the NIC vs
  // RCCL's many channels. We fan the per-round put across ``numQp`` QPs (warp w
  // -> qpId=w, disjoint 16B-aligned sub-range) -- but ONLY when the neighbour is
  // reached over RDMA. For a same-node neighbour ShmemPutMemNbiWarp lowers to a
  // single anvil SDMA queue per (src,dst); multiple warps hammering it overflow
  // the retry budget and coredump (validated-negative, ). So we read the
  // neighbour's transport at runtime: single-node simulation (P2P/SDMA) keeps
  // the proven single-warp path and stays bit-exact; only a true cross-node
  // (RDMA) neighbour fans out. Gated additionally on numQp>1 so the flat
  // whole-world ring (numQp defaults to 1) is byte-for-byte unchanged.
  application::TransportType nextXport =
      shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerIsRdma = (nextXport == application::TransportType::RDMA);
  // Multi-block and within-block multi-QP fan-out are
  // mutually exclusive: in multi-block mode each CTA already drives its own QP
  // (qpId=bid) on its own sub-range, so a single warp per block is correct.
  int useWarps = (!multiBlock && numQp > 1 && peerIsRdma) ? numQp : 1;
  if (useWarps > warpsPerBlock) useWarps = warpsPerBlock;
  bool fanOut = (useWarps > 1);

  // PUT-WITH-SIGNAL on the MULTI-BLOCK CHANNEL path (env MORI_HIER_RING_PUT_SIGNAL).
  // At N=2 (numQp==1) the big AG runs multiBlock: each CTA ``bid`` puts its 16B-
  // aligned sub-range of the chunk on qpId=bid, then thread 0 does a separate
  // ShmemQuietThread(nextPeer, bid) drain + a separate flag AMO. That quiet-drain
  // stalls the channel until its RDMA send CQ empties before the receiver's flag
  // can fire -- the exposed per-round latency of the single ring round. Fusing the
  // channel's data WRITE + its completion-flag AMO into ONE ShmemPutMemNbiSignal on
  // the SAME QP (qpId=bid) makes the flag ride strictly AFTER the data WRITE (RC
  // in-order on that QP), so the receiver observing the per-channel flag is
  // guaranteed the sub-range has landed globally -- with NO separate quiet and NO
  // separate AMO. Byte image is identical (same sub-range tiling, same flag slot);
  // only the completion mechanism changes. This is the same proven put-signal lever
  // already used on the single-warp (signalFused) and fan-out (fanOutSignal) paths,
  // now extended to the multiBlock channel path the big AG actually takes at N=2.
  // Gated on usePutSignal so the default path stays byte-for-byte unchanged.
  bool multiBlockSignal = (usePutSignal && peerIsRdma && multiBlock);

  // FANOUT PUT-WITH-SIGNAL (env MORI_HIER_RING_PUT_SIGNAL, default OFF): on the
  // multi-QP fan-out path the chunk is split across ``useWarps`` QPs, so a SINGLE
  // completion-flag AMO (even RC-ordered on one QP) only orders after THAT QP's
  // data -- the other QPs' tail bytes can still be in flight when the receiver
  // observes the flag (the residual FSDP loss remote-landing race). Fix: have
  // EACH fan-out warp fuse its data WRITE + a flag AMO_ADD(1) on its OWN QP via
  // ShmemPutMemNbiSignalWarp. On RC the responder executes each QP's WRITE then
  // its AMO in order, so every QP's data is globally visible before its own +1.
  // The receiver waits for the flag to reach the number of active fan-out warps
  // (``fanActive``) => it can only proceed after ALL QPs' data has landed, with
  // NO host sync (keeps the ring<->gather overlap). Symmetric homogeneous-RDMA
  // subgroup ring (leaders): next/prev both RDMA => send/recv counts match.
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
  // Expected increments on OUR recv slot = active fan-out warps the sender (prev)
  // used, iff prev also fans out with signals; else the classic single +1.
  int expectedRecvSig = (fanOutSignal && prevIsRdma) ? fanActive : 1;

  // RDMA-READ (PULL) ring gate. Engaged only on the single-round (ringSize==2)
  // all-RDMA inter-node phase -- the 2-node hierarchical AG this cluster runs.
  // In that case the chunk THIS PE needs (data-rank recvDataRank) is prevPeer's
  // OWN chunk, already present in prevPeer's ring slot after the intra prepare
  // barrier -- so it can be PULLED with an RDMA READ rather than waiting for the
  // peer to PUSH it. The READ completion, drained by our own quiet, is a
  // consumer-side landing guarantee (bytes physically in this PE's buffer +
  // system-fence-visible to its CUs): NO cross-PE flag AMO, NO receiver spin, NO
  // remote-landing race. Restricted to !multiBlock (single-block fan-out) and
  // maxRounds==1 so larger rings / the multi-channel path / single-node keep the
  // proven push path byte-for-byte. Result is byte-identical (same slot, bytes).
  bool useReadRing = (useRead && peerIsRdma && prevIsRdma && maxRounds == 1 && !multiBlock);

  // Phase-6 WRITE_WITH_IMM (single-warp, single-QP cross-node path only). The
  // sender emits RDMA_WRITE_WITH_IMM instead of put+quiet+flag-AMO; the receiver
  // consumes the recv-CQ completion instead of spinning the flag. The recv-CQE
  // cannot be observed before the write payload has landed globally, so this is
  // the transport-level completion that closes the remote-landing stale-read race
  // (13 device-side avenues refuted; only host sync worked) WITHOUT the host
  // stall. Gated to !multiBlock && !fanOut (numQp==1) so numQp>1 / multi-channel
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
  // remote-half completion race; every device-side reader barrier refuted, only a
  // host drain fixed it). Extend WRITE_WITH_IMM to the channel path: each CTA bid
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
    const int P = deepPipe;
    const int sendRank = ringPos;                       // maxRounds==1: send our own chunk
    const size_t chunkBaseOffsetSend = static_cast<size_t>(sendRank) * peChunkSize;
    const size_t kAlignDP = 16;
    const size_t nUnits = (peChunkSize + kAlignDP - 1) / kAlignDP;
    const size_t unitsPerP = (nUnits + static_cast<size_t>(P) - 1) / static_cast<size_t>(P);
    const int sw = useWarps;  // sender fan-out warps per sub-chunk (== numQp fan-out)
    // Per-sub-chunk active-warp count, computed identically to the sender tiling.
    // active(p) = # of warps that actually get a non-empty tile of sub-chunk p.
    auto activeOf = [&](int p) -> int {
      size_t sU = static_cast<size_t>(p) * unitsPerP;
      size_t eU = sU + unitsPerP;
      if (eU > nUnits) eU = nUnits;
      if (sU >= eU) return 0;
      size_t subUnits = eU - sU;
      size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
      if (upw == 0) upw = 1;
      int a = static_cast<int>((subUnits + upw - 1) / upw);
      if (a > sw) a = sw;
      if (a < 1) a = 1;
      return a;
    };
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
        size_t sU = static_cast<size_t>(p) * unitsPerP;
        size_t eU = sU + unitsPerP;
        if (eU > nUnits) eU = nUnits;
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
            size_t sU = static_cast<size_t>(pp) * unitsPerP;
            size_t eU = sU + unitsPerP;
            if (eU > nUnits) eU = nUnits;
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
      if (sw >= P && P <= warpsPerBlock) {
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        if (warpLead && myWarp < P) {
          int p = myWarp;
          if (nonEmptyDP(p)) {
            int base = grpBase(p);
            for (int q = 0; q < g; ++q) shmem::ShmemQuietThread(nextPeer, base + q);
            __threadfence_system();
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
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, static_cast<uint64_t>(1));
          }
        }
      } else if (threadLinearId == 0) {
        // WRAP fallback (P>sw): serial drain, groups share QPs.
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
            }
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, static_cast<uint64_t>(1));
        }
      }
      __syncthreads();
      if (threadLinearId == 0) {
        shmem::ShmemQuietThread(nextPeer);
      }
      __syncthreads();
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
        size_t sU = static_cast<size_t>(p) * unitsPerP;
        size_t eU = sU + unitsPerP;
        if (eU > nUnits) eU = nUnits;
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
        } else {
          shmem::ShmemPutMemNbiSignalWarp(memObj, off, memObj, off, eo - so, flagsObj,
                                          (flagBase + p) * sizeof(uint64_t), 1,
                                          core::atomicType::AMO_ADD, nextPeer, warpId);
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
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, static_cast<uint64_t>(1));
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
          }
        }
        __threadfence_system();
        core::AtomicStoreSeqCstSystem(chunkReadyFlags + p, static_cast<uint64_t>(1));
      }
    }
    __syncthreads();
    // Drain our own send QPs once (buffer-reuse safety), then reset the recv slots
    // (entry barrier in prepare orders every PE's reset before any peer's next-op
    // AMO, so this end-of-op reset is safe).
    if (threadLinearId == 0) {
      shmem::ShmemQuietThread(nextPeer);
    }
    __syncthreads();
    if (!deepPipeImm) {
      for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
      __syncthreads();
    }
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
          shmem::ShmemPutMemNbiWarp(memObj, subOff, memObj, subOff, blkBytes, nextPeer, bid);
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
            shmem::ShmemPutMemNbiSignalWarp(
                memObj, subOff, memObj, subOff, subEnd - subStart, flagsObj,
                (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
                nextPeer, warpId);
          } else {
            shmem::ShmemPutMemNbiWarp(memObj, subOff, memObj, subOff, subEnd - subStart, nextPeer,
                                      warpId);
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
        shmem::ShmemPutMemImmWarp(memObj, chunkBaseOffset, memObj, chunkBaseOffset, peChunkSize,
                                  static_cast<uint32_t>(sendDataRank + 1), nextPeer);
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
        shmem::ShmemPutMemNbiWarp(memObj, chunkBaseOffset, memObj, chunkBaseOffset, peChunkSize,
                                  nextPeer);
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
    core::AtomicStoreSeqCstSystem(chunkReadyFlags + bid, static_cast<uint64_t>(1));
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
  bool usedWriteImm = (multiBlockWriteImm || writeImm || fanOutWriteImm);
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
