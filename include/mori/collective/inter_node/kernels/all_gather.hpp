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
    int ringPos, int ringSize, int peBase, int peStride, const application::SymmMemObjPtr memObj,
    const application::SymmMemObjPtr flagsObj, size_t peChunkSize, int numQp = 1,
    int numBlocksOverride = -1, int bidOverride = -1,
    uint64_t* chunkReadyFlags = nullptr,
    int deepPipe = 1,
    int deepPipeQuiet = 0, bool useWriteFence = false) {
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
  application::TransportType nextXportMb = shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
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
  application::TransportType nextXport = shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerIsRdma = (nextXport == application::TransportType::RDMA);
  // Multi-block and within-block multi-QP fan-out are mutually exclusive: in
  // multi-block mode each CTA already drives its own QP (qpId=bid) on its own
  // sub-range, so a single warp per block is correct.
  int useWarps = (!multiBlock && numQp > 1 && peerIsRdma) ? numQp : 1;
  if (useWarps > warpsPerBlock) useWarps = warpsPerBlock;
  bool fanOut = (useWarps > 1);

  int prevPos = (ringPos - 1 + ringSize) % ringSize;
  int prevPeer = peBase + prevPos * peStride;
  application::TransportType prevXport = shmem::GetGlobalGpuStatesPtr()->transportTypes[prevPeer];
  bool prevIsRdma = (prevXport == application::TransportType::RDMA);

  // WRITE-PUSH (SEND-CQ) per-channel landing fence for the giant multiBlock AG.
  // Each channel CTA ``bid`` pushes its own sub-range [blkOff, blkBytes) of my
  // chunk to nextPeer on qpId=bid as a fused put-with-signal (the flag AMO rides
  // the same QP strictly after the data WRITE, RC in-order, so the receiver's
  // per-channel flag can never beat the data), then drains only qpId=bid's send
  // CQE (per-channel quiet, no cross-CTA CQ race) so the local WQE has completed
  // before this block publishes. The receiver spins its own per-channel inbound
  // flag (system-acquire) + system-fences. Byte-identical to the multiBlock push
  // receive; only the completion mechanism changes. Gated maxRounds==1 (the N=2
  // hier AG) so larger rings keep the default path.
  bool multiBlockWrite =
      (useWriteFence && peerIsRdma && prevIsRdma && maxRounds == 1 && multiBlock);

  __syncthreads();

  // ==========================================================================
  // DEEP-SQ TEMPORAL PIPELINE (MORI_HIER_DEEP_PIPE=P). Split the chunk into P
  // temporal sub-chunks issued back-to-back on the SAME full numQp fan-out with a
  // per-sub-chunk put-with-signal; sub-chunk p's landing flag fires (RC in-order)
  // before p+1's, so a reassembly worker pushes p over XGMI while p+1.. still
  // cross the NIC, overlapping intra reassembly with inter fill. Engaged only on
  // the single-round (ringSize==2) all-RDMA fan-out path; full useWarps QP fan-out
  // per sub-chunk, publishes P chunkReadyFlags in temporal order. Self-contained:
  // returns before the classic round loop. INERT when deepPipe<=1 (byte-identical).
  bool deepPipeEngaged =
      (deepPipe > 1 && peerIsRdma && prevIsRdma && maxRounds == 1 && !multiBlock &&
       chunkReadyFlags != nullptr && useWarps >= 1);
  if (deepPipeEngaged) {
    // Min-sub-chunk clamp: the deep-SQ temporal FIFO splits the chunk into P sub-chunks
    // issued back-to-back on the full useWarps QP fan-out (the NCCL_STEPS full-width-per-
    // step model). But an unclamped large P at a small total size shrinks each sub-chunk
    // below a useful RDMA transfer granularity -- the tiny per-sub-chunk WQEs starve the
    // NIC and the extra flag round-trips can deadlock small transfers. Clamp P so every
    // temporal sub-chunk carries >= kMinSubChunkB
    // (1 MiB): reqPmax = peChunkSize / kMinSubChunkB. This makes any requested
    // depth safe at every size, and is bit-exact -- P only controls the temporal
    // partition of the SAME contiguous bytes issued+landed in order, and the
    // default P=2 already yields >=8 MiB sub-chunks at every size >=32MB (clamp is
    // a no-op there).
    const size_t kMinSubChunkB = static_cast<size_t>(1) << 20;
    int reqPmax = static_cast<int>(peChunkSize / kMinSubChunkB);
    if (reqPmax < 1) reqPmax = 1;
    const int P = (deepPipe < reqPmax) ? deepPipe : reqPmax;
    const int sendRank = ringPos;  // maxRounds==1: send our own chunk
    const size_t chunkBaseOffsetSend = static_cast<size_t>(sendRank) * peChunkSize;
    const size_t kAlignDP = 16;
    const size_t nUnits = (peChunkSize + kAlignDP - 1) / kAlignDP;
    const size_t unitsPerP = (nUnits + static_cast<size_t>(P) - 1) / static_cast<size_t>(P);
    const int sw = useWarps;  // sender fan-out warps per sub-chunk (== numQp fan-out)
    // Uniform temporal split: sub-chunk p = [p*unitsPerP, (p+1)*unitsPerP), clamped.
    auto dpRange = [&](int p, size_t& sU, size_t& eU) {
      sU = static_cast<size_t>(p) * unitsPerP;
      eU = sU + unitsPerP;
      if (eU > nUnits) eU = nUnits;
      if (sU > eU) sU = eU;
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
      // QPs for full per-sub-chunk BW. group(p) = QPs [p*g, p*g+g); warp within the
      // group tiles the sub-chunk in 16B units. When sw < P (P>sw) groups wrap
      // (g==1, p%sw) --
      // then some sub-chunks share a QP and the drain also covers the later one
      // (still bit-exact, just less overlap). Draining a sub-chunk's WHOLE group
      // (all g QPs) before its AMO is the landing fence.
      const int g = (sw >= P) ? (sw / P) : 1;  // QPs per sub-chunk
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
          if (p >= P) p = -1;  // extra warps idle
        } else {
          p = warpId;  // P>sw: each of first sw warps -> its own sub-chunk group start
          wl = 0;
          gg = 1;
        }
        if (p >= 0) {
          for (int pp = p; pp < P;
               pp += (sw >= P ? P : sw)) {  // sw<P: warp handles pp = warpId, warpId+sw, ...
            size_t sU, eU;
            dpRange(pp, sU, eU);
            if (sU >= eU) {
              if (sw >= P)
                break;
              else
                continue;
            }
            size_t subUnits = eU - sU;
            // Tile sub-chunk pp across gg QPs; lane wl takes its slice.
            size_t upl = (subUnits + static_cast<size_t>(gg) - 1) / static_cast<size_t>(gg);
            if (upl == 0) upl = 1;
            size_t lS = sU + static_cast<size_t>(wl) * upl;
            size_t lE = lS + upl;
            if (lE > eU) lE = eU;
            if (lS >= lE) {
              if (sw >= P)
                continue;
              else
                continue;
            }
            size_t so = lS * kAlignDP;
            size_t eo = lE * kAlignDP;
            if (eo > peChunkSize) eo = peChunkSize;
            size_t off = chunkBaseOffsetSend + so;
            int qp = grpBase(pp) + wl;
            shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, eo - so, nextPeer, qp);
            if (sw >= P) break;  // sw>=P: one warp issues exactly one sub-chunk tile
          }
        }
      }
      __syncthreads();
      // PARALLEL SEND-DRAIN + RECV-PUBLISH: each sub-chunk p gets its OWN drain
      // warp-leader (thread p*warpSize) so a landed group fires its flag the INSTANT
      // its own QP group drains -- concurrently with the other groups still crossing
      // the NIC.
      // Distinct sub-chunks ride DISJOINT QP groups => distinct ep[]/CQ state, so
      // concurrent ShmemQuietThread(nextPeer, qp) calls never share a WQ/CQ =>
      // bit-exact (same drains, same AMOs, same flag slots; only the ORDER of
      // independent completions is relaxed, which the per-slot flags already allow).
      // ONE warp-leader per sub-chunk (thread p*warpSize): it drains its OWN QP
      // group, AMOs the remote landing flag p, THEN waits for its OWN incoming flag
      // p and publishes chunkReadyFlags[p]. A SINGLE leader per p (not a separate
      // send-drain and recv-publish leader) is required on wave64 HW: block=512 =>
      // warpsPerBlock = 8, so P<=warpsPerBlock holds and a leader warp exists for
      // every p (a 2*P-warp split would leave P=8 recv leaders without a warp =>
      // chunkReadyFlags never published => hang). All P leaders still run CONCURRENTLY
      // across sub-chunks (each on its DISJOINT QP group => disjoint WQ/CQ => no
      // shared-completion race, bit-exact); the per-p drain->recv sequence is the
      // natural inter->intra dependency for that sub-chunk, not a cross-p serialize.
      // ONLY safe when sw>=P (disjoint QP group per sub-chunk) AND P<=warpsPerBlock
      // (a leader warp exists for every p). Else fall back to the serial thread-0
      // drain (P>sw groups WRAP grpBase=p%sw sharing a QP; or too few warps).
      if (sw >= P && P <= warpsPerBlock) {
        // Parallelise the per-sub-chunk QP-group send-CQ drain. Each temporal
        // sub-chunk p fans its data across g = sw/P QPs (grpBase(p)..+g-1). Give
        // each of the g QPs its own drain warp so the g completion polls run
        // concurrently, then a lock-free per-group join (shared arrival counter)
        // lets the group leader AMO the remote flag + publish only after all g QPs
        // of that group have landed. Distinct groups keep firing independently
        // (per-group counter, no global barrier) so the inter-sub-chunk pipeline is
        // preserved. Identical QP drains and AMO/flag slots as the serial path; only
        // the order of the g independent per-QP completions within a group is relaxed
        // (each QP has its own WQ/CQ, per-slot flags allow it). warpsPerBlock>=sw
        // here, so a distinct drain warp exists per QP.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        // Single-group fast join: when P==1 all sw drain warps belong to one group,
        // so __syncthreads is the natural (and cheaper) join instead of an
        // atomic-counter spin. Every drain warp quiets its QP + __threadfence_system
        // before the barrier, then thread 0 AMOs the remote flag / spins the inbound
        // landing flag / publishes. Same QP-drain set and single AMO/flag slot as the
        // counter join; only the P==1 join primitive changes. The P>1 pipeline keeps
        // the per-group counter join below (a block barrier there would
        // cross-synchronize independent groups and serialize the pipeline).
        if (P == 1) {
          if (warpLead && myWarp < sw && nonEmptyDP(0)) {
            shmem::ShmemQuietThread(nextPeer, grpBase(0) + myWarp);  // g==sw, wl==myWarp
            __threadfence_system();                                  // this QP's bytes visible
          }
          __syncthreads();  // all g QP drains + fences complete (uniform block join)
          if (threadLinearId == 0 && nonEmptyDP(0)) {
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                           (flagBase + 0) * sizeof(uint64_t), 1,
                                                           core::atomicType::AMO_ADD, nextPeer);
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + 0) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
              if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + 0,
                                          static_cast<uint64_t>(1));
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
              __threadfence_system();           // this QP's landed bytes visible
              atomicAdd(&dpGrpDrained[p], 1u);  // signal group arrival
              if (wl == 0) {
                // Group leader: wait for all g QPs of this group to land, then AMO
                // the remote flag + spin our own inbound flag + publish. atomicAdd(.,0)
                // is a well-defined atomic load of the shared arrival counter.
                long long gspin = 0;
                while (atomicAdd(&dpGrpDrained[p], 0u) < static_cast<unsigned int>(g)) {
                  if (++gspin > 10000000000LL) __builtin_trap();
                }
                shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                               (flagBase + p) * sizeof(uint64_t), 1,
                                                               core::atomicType::AMO_ADD, nextPeer);
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
                core::AtomicStoreSeqCstSystem(
                    chunkReadyFlags + p,
                    static_cast<uint64_t>(1));
              }
            }
          }
        }  // end else (P>1 per-group counter join; P==1 uses the block-barrier fast join)
      } else if (sw <= warpsPerBlock) {
        // Wrap parallel drain: P>sw so sub-chunks share QPs (grpBase=p%sw), but the
        // sw QP groups are disjoint. Give each QP group w in [0,sw) its own merged
        // leader warp (thread w*warpSize): it walks its sub-chunks p = w, w+sw, ...
        // in increasing-p order, draining QP w (in-order per QP == landing proof),
        // AMOing remote flag p, then waiting on its own incoming flag p and
        // publishing chunkReadyFlags[p]. The sw leaders run concurrently on disjoint
        // QP groups (disjoint WQ/CQ => no shared-completion race); same drains, AMOs,
        // and flag slots as the serial path with per-QP completion order preserved.
        // Send-drain and recv-publish are merged into one leader per group to stay
        // within warpsPerBlock on wave64. Requires sw<=warpsPerBlock (a leader warp
        // per QP group); else fall to the serial thread-0 drain below.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        if (warpLead && myWarp < sw) {
          const int w = myWarp;
          for (int p = w; p < P; p += sw) {
            if (!nonEmptyDP(p)) continue;
            shmem::ShmemQuietThread(nextPeer, grpBase(p));  // grpBase(p)==p%sw==w
            __threadfence_system();
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                           (flagBase + p) * sizeof(uint64_t), 1,
                                                           core::atomicType::AMO_ADD, nextPeer);
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
              if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                          static_cast<uint64_t>(1));
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
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                        static_cast<uint64_t>(1));
        }
      }
      __syncthreads();
      // Skip the redundant trailing full-QP re-drain on the parallel deepPipeQuiet
      // paths. The two parallel branches above give every temporal sub-chunk its own
      // leader that ShmemQuietThread(nextPeer, qp)-drains its QP group before that
      // group's AMO, and the union of the groups covers every send QP this op used.
      // So by the time all P leaders have published chunkReadyFlags, all our send
      // WQEs have already completed, so the buffer-reuse safety the trailing
      // ShmemQuietThread(nextPeer) provides is already satisfied; re-draining
      // already-empty CQs only adds fixed per-op tail latency. Removed on the
      // parallel paths only; the serial-fallback branch (sw>warpsPerBlock) keeps
      // the trailing drain unchanged.
      const bool dpParallelDrained =
          (sw >= P && P <= warpsPerBlock) || (sw <= warpsPerBlock);
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
    // SENDER: warps [0,sw). For each temporal sub-chunk p, warp w sends its
    // 16B-aligned tile on qpId=w. All P sub-chunks ride qpId=w back-to-back (deep
    // SQ) so p's data lands before p+1's -- temporal landing order preserved.
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
        shmem::ShmemPutMemNbiSignalWarp(memObj, off, memObj, off, eo - so, flagsObj,
                                        (flagBase + p) * sizeof(uint64_t), 1,
                                        core::atomicType::AMO_ADD, nextPeer, warpId);
      }
    }
    // RECEIVER: republish chunkReadyFlags[p] in temporal order so the reassembly
    // worker can push sub-chunk p while later sub-chunks still cross the NIC.
    // Spin the flag sum per sub-chunk, then publish.
    if (threadLinearId == 0) {
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
        core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                      static_cast<uint64_t>(1));
      }
    }
    __syncthreads();
    // Overlap the buffer-reuse send-QP drain (thread 0) with the recv flag-slot
    // reset (threads>0) under one trailing block barrier. The two touch disjoint
    // state: thread 0 drains only its local send CQs (ShmemQuietThread(nextPeer),
    // buffer-reuse safety), while threads>0 zero flagsArray[flagBase+idx] which no
    // reader reads after the __syncthreads above (chunkReadyFlags were already
    // published there). The entry barrier in prepare orders every PE's reset before
    // any peer's next-op AMO, so the reset is safe; the single trailing join
    // guarantees both the drain and the reset finish before the fence/return. Same
    // QP drains and flag zeroing as the separate-barrier path, with one fewer join.
    for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
      flagsArray[flagBase + idx] = 0;
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

    if (multiBlockWrite) {
      // MULTI-BLOCK WRITE-PUSH + SEND-CQ landing fence. Warp 0 pushes THIS
      // channel's sub-range [blkOff,blkBytes) of my chunk (chunkBaseOffset) to
      // nextPeer on qpId=bid as ONE fused put-with-signal: the data WRITE and the
      // flag AMO_ADD(1) ride the SAME QP, RC-ordered, so on the responder the sub-
      // range is globally visible BEFORE its own +1 fires -- the receiver's per-
      // channel flag can never beat the data (bit-exact by construction).
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        shmem::ShmemPutMemNbiSignalWarp(memObj, subOff, memObj, subOff, blkBytes, flagsObj,
                                        (flagBase + sendDataRank) * sizeof(uint64_t), 1,
                                        core::atomicType::AMO_ADD, nextPeer, bid);
      }
      __syncthreads();
      // No explicit send-CQ quiet. The fused put-signal already carries the flag
      // AMO as the last WQE on qpId=bid strictly AFTER the data WRITE (RC in-
      // order), so the receiver observing the per-channel flag is already
      // guaranteed the sub-range has landed globally -- an explicit
      // ShmemQuietThread(nextPeer,bid) here would only stall the channel until its
      // send CQ empties. Receiver: spin THIS channel's inbound flag (peer's CTA
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
    // Single-warp per ring round: the same-node neighbour path of
    // ShmemPutMemNbiWarp lowers to one anvil SDMA queue per (src,dst) pair. All
    // warps would target the same nextPeer chunk region, so splitting the put
    // across warps hammers ONE SDMA queue and overflows its retry budget (anvil
    // "submitPacket: Retry limit exceeded"). Multi-warp puts only help when each
    // warp drives a distinct queue/QP (as the intra SDMA gather does, one warp per
    // peer); a single-peer ring round has one queue, so a single warp is correct.
    //
    // Fan-out (numQp>1) is the alternative for the all-RDMA sub-group ring: a
    // single warp on a single QP underfills the NIC (the transport provisions
    // numQpPerPe QPs/peer). Fanning the chunk across QPs
    // (warp w -> disjoint 16B-aligned sub-range on qpId=w) drives multiple QPs in
    // parallel. Correctness gate: the flag bump below must follow a quiet that
    // drains ALL used QPs. Gated on numQp>1 only for the all-RDMA sub-group ring;
    // the flat single-node ring (P2P, one anvil queue) stays single-warp.
    auto putDeep = [&](size_t off, size_t bytes, int qp) {
      shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, bytes, nextPeer, qp);
    };
    if (multiBlock) {
      // RCCL-style channel: this CTA puts only its sub-range [blkOff, blkOff+
      // blkBytes) of the chunk, on qpId=bid (a distinct QP per channel). Warp 0
      // issues; the union of all CTAs' sub-ranges tiles the chunk exactly =>
      // byte-identical result.
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        putDeep(subOff, blkBytes, bid);
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
          putDeep(subOff, subEnd - subStart, warpId);
        }
      }
      // All fan-out warps must finish ISSUING their puts before thread 0 drains
      // the QPs and bumps the flag, else the receiver's flag could fire before a
      // tail QP's data lands. (Only added on the fan-out path -- the single-warp
      // path keeps the  thread schedule unchanged.)
      __syncthreads();
    } else if (warpId == 0) {
      putDeep(chunkBaseOffset, peChunkSize, 0);
    }

    // Outbound drain (quiet + flag bump to nextPeer) and inbound recv-flag wait
    // (from prevPeer) are on independent network directions; run them on two
    // threads so their latencies overlap, closed by a single trailing __syncthreads.
    // Round-level pipelining is impossible: round i+1 sends sendDataRank =
    // (ringPos-i-1) == the recvDataRank of round i (you forward exactly what you
    // just got), a hard data dependency.
    if (threadLinearId == 0) {
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
        // CORRECTNESS (flag-beats-data): a cross-node (RDMA) neighbour's put
        // rode an RDMA QP, but the SDMA-typed memObj quiet drains only the
        // P2P/SDMA path, not the RDMA send-queue completion -- the flag AMO
        // below could land (RC-ordered on its own QP) before the data PUT has
        // drained, letting the receiver read stale remote-half bytes. Use the
        // transport-aware quiet (RDMA -> all numQpPerPe QPs) so the put fully
        // drains before the flag fires; on-device, no host sync. Mirrors the
        // fan-out path's RDMA quiet.
        shmem::ShmemQuietThread(nextPeer);
      } else {
        shmem::ShmemQuietThread(nextPeer, memObj);
      }
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                     (flagBase + sendDataRank) * sizeof(uint64_t),
                                                     1, core::atomicType::AMO_ADD, nextPeer);
    } else if (threadLinearId == warpSize) {
      // Each round the sender increments a DISTINCT flag slot (index
      // recvDataRank = sendDataRank on the receiver), so every slot is
      // incremented exactly once over the ringSize-1 rounds -- 0 -> 1.
      // Wait for THIS round's slot to become nonzero (per-slot, not a cumulative count).
      int spinCount = 0;
      // SYSTEM-scope acquire: the flag is bumped by a REMOTE peer's RDMA AMO and
      // the chunk it guards is landed by that peer's RDMA put -- both cross-agent
      // writes. A RELAXED load establishes NO happens-before with those data
      // writes, so observing the flag does NOT make the received chunk coherently
      // visible to this GPU's subsequent forward-put / copy-OUT -> the RDMA (remote)
      // half of the output reads STALE bytes under FSDP tight overlap. A
      // system-scope acquire + system threadfence makes the peer's prior data
      // writes visible without a host sync -- mirrors the intra SDMA gather's
      // AtomicLoadSeqCstSystem + __threadfence_system receiver pattern.
      // The sender adds exactly +1 to this round's slot.
      while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + recvDataRank) < 1) {
        spinCount++;
        if (spinCount > 10000000) {  // Increased timeout threshold
          break;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  }

  // Per-chunk landing publish. After the round loop this block's inbound
  // sub-range (channel ``bid``) is fully landed AND made visible to this GPU by
  // the receiver's __threadfence_system inside the loop. Publish the ready flag so
  // a concurrent reassembly block can start pushing exactly this sub-range over
  // XGMI without waiting for the whole ring / a global finish barrier. The store
  // is system-scoped and preceded by a fence so the reassembly reader observes the
  // landed bytes coherently. INERT when chunkReadyFlags==nullptr (every existing
  // caller) so the standalone ring stays byte-for-byte identical.
  if (chunkReadyFlags != nullptr && threadLinearId == 0) {
    __threadfence_system();
    core::AtomicStoreSeqCstSystem(chunkReadyFlags + bid,
                                  static_cast<uint64_t>(1));
  }
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
// engine's read of the self-slot: the default path reads the ring buffer (fine-grained
// RDMA scratch, SDMA-read-coherent with the NIC write) and is correct, whereas direct-land
// reads the coarse-grained cached output tensor, whose line the NIC write does not
// invalidate for the copy engine. A fix would land the RDMA into fine-grained coherent
// staging that the SDMA reassembly read snoops.
// ============================================================================
