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
// Returns true only when MORI_HIER_RING_PUT_SIGNAL is set to a non-"0" value; an
// unset env yields false. The fused FSDP builders require put-signal to be
// explicitly enabled (unlike the standalone ring, which defaults it on) so their
// output bytes are unchanged unless the env opts in.
inline bool HierRingPutSignalExplicitlyOn() {
  const char* e = std::getenv("MORI_HIER_RING_PUT_SIGNAL");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Elastic reassembly (MORI_HIER_FUSE_REMOTE_ELASTIC, default OFF). In the pipelined
// FusedRingRemoteGather kernel the local-block CTA finishes its own-shard SDMA
// gather early and then idles as the completion reader, leaving queue 0 unused
// during the reassembly tail (only queues 1..reasm are active). With this lever the
// local CTA joins remote reassembly as an extra worker on queue 0 once its own
// gather completes, so the tail runs on reasm+1 SDMA queues. Bit-exact: workers
// 0..reasm handle disjoint ring channels f % (reasm+1).
inline bool HierFuseRemoteElasticOn() {
  const char* e = std::getenv("MORI_HIER_FUSE_REMOTE_ELASTIC");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Intra reassembly per-peer queue split (MORI_HIER_REASM_QSPLIT, default OFF). The
// reassembly push (OneShotSubGroupPushOnly_body) copies each member's column on a
// single per-peer SDMA queue (q == qId % nq), leaving the remaining per-peer queues
// idle when nq > effReasm. When set, each worker splits its per-peer column across
// its own disjoint queue class {k in [0,nq): k % qStride == qId % qStride} (qStride
// == effReasm), engaging up to ceil(nq/effReasm) engines per peer link with no
// cross-worker same-queue race. Only active on the single-shot path (reasmDeepSq==0).
// Bit-exact: disjoint 16B-aligned sub-ranges of the same column, all owned queues
// drained before the flag.
inline bool HierReasmQSplitOn() {
  const char* e = std::getenv("MORI_HIER_REASM_QSPLIT");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Batched sender-quiet (MORI_HIER_FUSE_SENDER_FENCE, default OFF). The
// per-peer push warps each run drain -> __threadfence_system -> flag interleaved in
// warp order, so the G peer copies quiet serially with a system fence between each.
// This lever splits the push-only body into two phases: all G warps push+drain, a
// block barrier, then a single completion phase where each peer warp fences+fires
// its flag -- so the copies stay continuously in flight and the fences batch at the
// end. Applies only to the single-shot push path (deepSqPhase==0, qSplit==0).
// Bit-exact: every copy drained before any flag, each flag still preceded by a
// system fence.
inline bool HierFuseSenderFenceOn() {
  const char* e = std::getenv("MORI_HIER_FUSE_SENDER_FENCE");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Copy-engine flag delivery (MORI_HIER_QFLAG, default OFF). 1 => the push-only
// reassembly body rides each peer's completion flag on the same SDMA queue as its
// data copy (COPY_LINEAR + FENCE, FIFO-ordered) instead of drain +
// __threadfence_system + a separate direct P2P AMO, stripping the per-peer
// drain/fence/AMO round-trips from the all-to-all completion critical path. See
// SdmaPutFencedFlagThread. Bit-exact by construction.
inline bool HierQFlagOn() {
  const char* e = std::getenv("MORI_HIER_QFLAG");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Intra-node pipelined ring for the crown local block (MORI_HIER_CROWN_RING,
// default 0=OFF). Replaces the flat all-to-all local gather with a nearest-neighbor
// pipelined ring (one flow per link). Value carries fencedFlag in bit0:
// 1 => copy+local-signal relay (SdmaPutCopySignalThread, per-step drain + P2P AMO),
// 2 => plain per-chunk drain relay (SdmaPutThread + SdmaQueitThread + AMO).
// 0 => OFF (byte-identical flat crown). Bit-exact: same final slot layout and bytes,
// only the intra copy schedule differs.
// MORI_HIER_CROWN=1 selects the named size-adaptive crown (flatMW bit9=512 +
// batchSelf bit11=2048 = 2560) by name instead of the raw 2560 bitmask. A non-empty
// MORI_HIER_CROWN_RING overrides the named flag so raw bitmask experiments still
// work. Neither set => 0 => byte-identical flat crown.
inline int HierCrownRing() {
  const char* e = std::getenv("MORI_HIER_CROWN_RING");
  if (e != nullptr && e[0] != '\0') return std::atoi(e);
  const char* c = std::getenv("MORI_HIER_CROWN");
  if (c != nullptr && c[0] != '\0' && !(c[0] == '0' && c[1] == '\0')) return 2560;
  return 0;
}
// Multi-engine per-link reassembly put (MORI_INTRA_MQ, default OFF). The push-only
// reassembly worker (OneShotSubGroupPushOnly_body) drives only queue 0 of the ~2
// KFD-recommended XGMI SDMA engines per link. 1 => split each peer's column across
// all sdmaNumQueue queues via SdmaPutWarp (lane k -> queue k over a disjoint
// contiguous sub-range), engaging every recommended engine on the link. Bit-exact:
// disjoint sub-ranges of the same bytes, all nq queues drained before the flag AMO
// fires. 0 = OFF (byte-identical path).
inline bool HierReasmMultiQueueOn() {
  const char* e = std::getenv("MORI_INTRA_MQ");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Width-W permutation issue schedule (MORI_HIER_RING as an integer width). The
// crown fires all G-1 peer copies concurrently (full mesh); the boolean
// ringPhased issued one link at a time (perfect matching but 1/(G-1) link
// utilisation). MORI_HIER_RING=W issues the G peers in ceil(G/W) rotated phases of
// W concurrent links each (a W-regular matching per phase: each receiver takes
// exactly W concurrent writers per phase), with a CTA barrier staggering the phases
// -- the middle ground between full incast (W=G-1) and full serialisation (W=1).
// 0/empty => OFF (byte-identical crown). Note: W=2 has been observed to hit
// an HSA device exception in the RDMA CQ drain at the largest size, so W>=4 is
// preferred. Bit-exact: identical bytes/slots/completion, only the issue stagger
// changes.
inline int HierRingWidth() {
  const char* e = std::getenv("MORI_HIER_RING");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = std::atoi(e);
  return v < 0 ? 0 : v;
}
// Batched sender-side completion fence (MORI_HIER_BATCH_FENCE, default 0=OFF). The
// crown sender tail runs, on each of the G issuing warps concurrently:
// ShmemQuietThread(peer) -> __threadfence_system -> AMO_SET(peer flag).
// __threadfence_system is a system-scope (agent-wide) write-ordering barrier that
// flushes this agent's whole outstanding write set, so firing it G times at once is
// redundant and serializes in the memory subsystem. When set, the tail becomes:
// every warp drains its own peer's SDMA queue (parallel) -> CTA barrier -> one thread
// issues a single __threadfence_system -> CTA barrier -> every warp publishes its
// peer's AMO flag (parallel). Bit-exact and in fact a strictly stronger order: the
// single fence happens-after all G drains and happens-before all G flag AMOs, so no
// flag can precede any peer's globally-ordered bytes. 0/empty => OFF (byte-identical
// crown, per-peer fence retained).
inline bool HierBatchFence() {
  const char* e = std::getenv("MORI_HIER_BATCH_FENCE");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Hierarchical 2x4 intra gather (MORI_HIER_H2X4, default 0=OFF). The flat G-way intra
// SDMA gather fires G-1 concurrent outbound copies per GPU (7 at G=8) into the ~2
// recommended XGMI engines, oversubscribing them. This mode splits the node's G ranks
// into two sub-groups of H=G/2 and gathers in 2 phases, halving peak concurrent
// outbound copies to H:
// phase 1: each GPU pushes its own shard to its H-1 sub-group peers + its cross-SG
// partner (partnerPos = groupPos ^ H) -- H concurrent copies.
// (local wait for the partner's shard to land)
// phase 2: each GPU forwards its partner's landed shard to its H-1 sub-group peers
// -- H-1 concurrent copies.
// Every shard reaches all G-1 non-owners; the final completion wait is unchanged
// (still waits G-1 sender flags). Only eligible on the plain single-queue path
// (multiQueue/qFlag/ringPhased/batchFence own their own submission+tail). Requires G
// even and H a power of two. Bit-exact: same final bytes/slots/flags; phase 2 reads
// only after acquiring the partner flag. Default 0 => byte-identical crown.
// Note: the two phases are data-dependent (phase 2 forwards the partner's phase-1
// shard) so they cannot overlap, and the cross-SG bytes traverse XGMI twice (~1.4x
// flat at G=8), so this path is strictly slower than the byte-optimal flat gather and
// is kept off by default.
inline int HierH2x4() {
  // Mode-valued: 0=off, 1=serial 2-phase (data-dependent phases serialize),
  // 2=overlapped 2-phase: P2 rides a distinct SDMA queue (qId=1) so its forward
  // executes concurrently with P1's drain, and the partner-wait overlaps the P1
  // drain. Mode 3 = asymmetric single-relay 2x4: P1 intra-SG 4-way gather, P2 the
  // two sgPos0 relays exchange their H-shard block over the single relay link (no
  // circular dep), P3 each relay broadcasts the other SG's H shards to its H-1 peers.
  // Default 0 => byte-identical crown.
  const char* e = std::getenv("MORI_HIER_H2X4");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = atoi(e);
  return v < 0 ? 0 : v;
}
// 2x4 stacked-flat-body hierarchical intra (MORI_HIER_INTRA2=W, default 0=OFF). The
// crown's local gather is a flat full-mesh push: every rank bursts all G-1 peer copies
// into the XGMI crossbar at once. This lever calls the same flat crown push body in
// ceil(G/W) sequential waves of a W-regular rotated matching: wave p issues only the
// warps with ((w-groupPos+G)%G)/W == p, so across ranks each wave is a perfect
// W-matching (every receiver has exactly W concurrent writers this wave). Unlike the
// ringPhased stagger, each wave drains (ShmemQuiet + system fence + flag AMO) to
// completion before the next wave submits, so only W XGMI egress links are ever in
// flight at once. W==4 at G==8 is the 2x4 case. Bit-exact: same bytes into slot
// groupPos of the same G peers, same per-peer drain+fence+flag, only the issue is
// serialized into W-wide drained waves; the final completion wait is unchanged.
// Kept off by default: the per-wave drain serialization idles the crossbar and the
// flat gather's 7 peer links already run concurrently at link rate, so reducing width
// regresses. Default 0 => byte-identical crown.
inline int HierIntra2() {
  const char* e = std::getenv("MORI_HIER_INTRA2");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = atoi(e);
  return v < 0 ? 0 : v;
}
// Descriptor pipelining (MORI_HIER_PUT_NDESC, default 1 = OFF). Number of
// back-to-back SDMA copy sub-descriptors to place on one queue per per-peer push
// before the single trailing atomic (see SdmaPutThread). >1 keeps the SDMA engine fed
// with queued descriptors so its descriptor-fetch latency overlaps in-flight DMA
// instead of the engine idling after one giant copy (distinct from QSplit's
// multi-queue and DEEP_PIPE's per-sub landing flags). A single COPY_LINEAR already
// saturates an isolated XGMI link, so this knob is typically neutral. Clamped [1,16].
// Bit-exact by construction.
inline int HierPutNdesc() {
  const char* e = std::getenv("MORI_HIER_PUT_NDESC");
  if (e == nullptr || e[0] == '\0') return 1;
  int v = atoi(e);
  if (v < 1) v = 1;
  if (v > 16) v = 16;
  return v;
}
// COPY_LINEAR DW2 coherence hint on the local-gather pole (MORI_HIER_PUT_CACHEHINT).
// bit0->dst_ha, bit1->src_ha (host-access coherence-routing hints). The gfx942
// COPY_LINEAR packet has no L2 cache-policy field (only swap+ha; swap corrupts, so it
// is not exposed), and DW2==0 gives peak D2D bandwidth; this is the only
// non-corrupting DW2 knob. 0 = OFF (byte-identical crown). No FSDP/E2E caller sets it.
inline int HierPutCacheHint() {
  const char* e = std::getenv("MORI_HIER_PUT_CACHEHINT");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = atoi(e);
  if (v < 0) v = 0;
  if (v > 3) v = 3;
  return v;
}
// Pole SQ-depth (MORI_HIER_POLE_SQDEPTH). Distinct from HierPutNdesc: putNdesc places
// K sub-descriptors of one copy in one doorbell (descriptor-fetch depth); poleSqDepth
// issues K independent SdmaPutThread copies (K doorbells, K COPY+ATOMIC pairs)
// back-to-back on the same per-peer queue, building an SQ occupancy of K in-flight work
// items to keep the copy engine full across whole-copy completions. Bit-exact: the K
// chunks are disjoint contiguous sub-ranges of the same bytes; each bumps
// expectedSignals so the single ShmemQuietThread drains all K before the flag AMO.
// A single COPY_LINEAR already continuously feeds the engine, so this is typically
// neutral-to-regressive. 1 = OFF (byte-identical path, one copy/peer).
inline int HierPoleSqDepth() {
  const char* e = std::getenv("MORI_HIER_POLE_SQDEPTH");
  if (e == nullptr || e[0] == '\0') return 1;
  int v = atoi(e);
  if (v < 1) v = 1;
  if (v > 16) v = 16;
  return v;
}
// Pull local gather (MORI_HIER_POLE_PULL). The crown local-gather is a push
// all-gather: each rank reads its local input and writes its shard into slot groupPos
// of all peer outputs (XGMI-egress-heavy). This flips the copy direction to a pull:
// each rank's own copy engines read the peer shards over XGMI into its local output and
// the rank completes by draining its own queues -- no per-peer landing fence (only a
// cheap staged flag, no data move, crosses PEs). Bit-exact: the same shard bytes land
// in the same output slots. On this XGMI fabric the SDMA read direction is slower than
// the write (push), so this is kept off by default. 0 = OFF (byte-identical).
inline int HierPolePull() {
  const char* e = std::getenv("MORI_HIER_POLE_PULL");
  if (e == nullptr || e[0] == '\0') return 0;
  return atoi(e) != 0 ? 1 : 0;
}
// Read-coalescing tile for the remote-reassembly fan-out (MORI_HIER_REASM_L2TILE,
// value in MiB, default 0 = OFF). The reassembly's groupSize concurrent XGMI
// peer-copies each read the same landed remote block from this leader GPU's ring
// buffer, contending for HBM read bandwidth with the concurrent NIC->ring fill writes.
// This lever tiles the per-peer copy into L2-sized byte windows and drives all
// groupSize peer copies through the same tile window together (per-tile completion
// barrier via the body's own drain+__syncthreads), aiming to serve the redundant
// reads from L2. Bit-exact: same bytes/dst, the completion flag fires only after the
// last tile's drain. On this hardware the copy-engine reads bypass L2, so tiling only
// adds per-tile completion overhead; kept off by default. 0 => byte-identical.
inline size_t HierReasmL2Tile() {
  const char* e = std::getenv("MORI_HIER_REASM_L2TILE");
  if (e == nullptr || e[0] == '\0') return 0;
  long v = atol(e);
  if (v < 0) v = 0;
  if (v > 64) v = 64;  // cap at 64 MiB (>= max UT chunk => single tile == off-ish)
  return static_cast<size_t>(v) * 1024 * 1024;
}
// Push-issue rotation (MORI_HIER_PUSH_ROTATE, default OFF). The push-only
// reassembly has every rank's warp w target the same global peer (peBase+w*peStride)
// in lock-step warp order, so all ranks hit the same XGMI destination port on the same
// cycle. This lever rotates each rank's warp->peer map by its own group position
// (effWarp = (warpId + groupPos) % groupSize) so rank g starts pushing to peer g+1,
// spreading the per-cycle destination-port load. Pure permutation of the peer set:
// each warp still writes this rank's column into slot groupPos of a distinct peer, all
// G peers covered exactly once, same bytes/slots/flags. Bit-exact; only the
// warp-order->destination-peer assignment changes. 0 = OFF (byte-identical).
inline bool HierPushRotateOn() {
  const char* e = std::getenv("MORI_HIER_PUSH_ROTATE");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Phased-permutation reassembly push (MORI_HIER_PUSH_PHASED, default OFF). The
// reassembly fires all groupSize peer copies concurrently (warp w -> peer w), so every
// receiver takes an incast of all G writers at once. This lever serialises the peer set
// into G rotated phases with a CTA barrier between each -- in phase p rank g pushes only
// to peer (g+1+p)%G -- so across ranks each phase is a perfect matching (every receiver
// has exactly one writer). This is the same permutation-step schedule as ringPhased,
// applied to the heavier remote reassembly all-to-all instead of the local gather.
// Bit-exact: same column bytes into the same slot groupPos of the same G peers with the
// same per-peer drain+fence+flag; only the issue order changes. 0 = OFF (byte-identical
// path).
inline bool HierPushPhasedOn() {
  const char* e = std::getenv("MORI_HIER_PUSH_PHASED");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// In-kernel copy-in (MORI_HIER_FUSE_COPYIN, default OFF). Folds the host
// hipMemcpyAsync copy-in of this PE's input into its ring slot into the fused kernel:
// each ring channel CTA stages its own send sub-range of gInput into the local ring
// slot then __syncthreads before the RDMA put, so the put sources valid data with no
// cross-CTA dependency. Combined with MORI_HIER_GEN_RING (drops the entry barrier) and
// slice_defer_fin the AG can collapse to a single host kernel launch. Default OFF =>
// the host copy-in runs, byte-identical path.
inline bool HierFuseCopyInOn() {
  const char* e = std::getenv("MORI_HIER_FUSE_COPYIN");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
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
// Same-CTA inline reassembly (MORI_HIER_INLINE_REASM, default OFF). On the multiBlock
// spatial ring (ringBlocks>1, N=2) ring channel CTA bx lands exactly spatial sub-range
// bx of the single remote chunk, and the dedicated reassembly worker j==bx pushes that
// same sub-range over XGMI (1:1 channel<->worker, partition==rb, stride==rb). The
// path relays the landed bytes across CTAs (ring CTA stores chunkReadyFlags[bx],
// a separate reassembly CTA spin-loads, re-fences, and reads the ring buffer), which
// costs extra reassembly CTAs and opens a cross-CTA stale-L2 window. With this lever the
// ring CTA, having already system-fenced its own landed sub-range, calls
// FusedRemoteReassembleWorker(partition=rb, j=bx, stride=rb, qId=bx+1) inline before
// returning -- same per-channel tile, per-queue drain, and output flags as the dedicated
// worker, so the output is byte-identical and the completion reader is unchanged. The
// Python launcher drops the dedicated reassembly CTAs (grid 2*rb+1 -> rb+1). Restricted
// to the multiBlock path (rb>1, deepPipe forced 1 there, no host-proxy inter). Default
// OFF => byte-identical.
inline bool HierInlineReasmOn() {
  const char* e = std::getenv("MORI_HIER_INLINE_REASM");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Blocking landing wait (MORI_HIER_SPIN_BLOCK, default OFF). The reassembly /
// completion-reader landing-flag spins carry a bounded fallback (if(++spin>1e8)break)
// so a hung peer cannot wedge the kernel forever -- but under deep pipelining +
// multi-queue reassembly contention that fallback can fire on bytes that have not yet
// landed, so the caller reads/publishes under-landed data. When ON the spin never
// abandons: it polls the landing flag until it is actually set, giving an in-kernel
// HW-completion landing fence with no host round-trip and no CU payload copy. Default
// OFF => byte-identical bounded path.
inline bool HierSpinBlock() {
  const char* e = std::getenv("MORI_HIER_SPIN_BLOCK");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Cooperative spin-backoff (MORI_HIER_SPIN_BACKOFF, default OFF). The fused ring
// gather's flag-wait spin loops (chunkReadyFlags and the reassembly/local completion
// reader) busy-poll system-scope L2 atomics as tight as the hardware allows. Under
// deep pipelining the many concurrent pipeline CTAs hammering those loads can contend
// for the same L2/fabric path the SDMA reassembly pushes and the mlx5 CQ drainers
// need, stalling the op. Inserting an s_sleep between polls yields SIMD cycles and
// fabric bandwidth to the drainers without changing which flag/value gates the fence,
// so it is bit-exact. Default OFF (no sleep) keeps the tight-spin byte-identical.
inline bool HierSpinBackoff() {
  const char* e = std::getenv("MORI_HIER_SPIN_BACKOFF");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Deep-SQ WQE-depth (MORI_HIER_WQE_DEPTH, default 1 = byte-identical). The big
// inter-node put splits the chunk across numQp warps -> numQp QPs, each warp issuing
// one whole-sub-range RDMA-WRITE WQE per QP. With depth d each warp splits its
// sub-range into d back-to-back non-blocking puts on its same QP, so the NIC sees d
// queued WQEs per QP instead of 1 (the device analogue of the host-proxy deep SQ). The
// union of the d 16B-aligned sub-puts tiles the sub-range exactly and rides the same
// QP in RC order, so the byte image and the completion drain (per-QP quiet) are
// identical; only the SQ depth changes. depth<=1 => single put (path unchanged).
inline int HierWqeDepth() {
  const char* e = std::getenv("MORI_HIER_WQE_DEPTH");
  if (e == nullptr || e[0] == '\0') return 1;
  int v = std::atoi(e);
  return v < 1 ? 1 : v;
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
  // Default depth 2. Self-safe: the per-sub-chunk size gate (HierDeepPipeMaxBytes)
  // cages the giant AG to the whole-chunk crown fence, keeping E2E bit-exact with no
  // explicit env.
  if (e == nullptr || e[0] == '\0') return 2;
  if (e[0] == 'a' || e[0] == 'A') return -1;  // "auto"
  int v = std::atoi(e);
  if (v < 1) return 1;
  if (v > 16) return 16;
  return v;
}

// Size gate for DEEP_PIPE (MORI_HIER_DEEP_PIPE_MAX_MB, default 0 = no limit). The
// per-sub-chunk device landing flag (send-CQE quiet + AMO) is not a scale-robust
// landing proof on this mlx5 provider (write-with-imm recv-CQE is HW-unavailable) and
// can mis-order or crash on very large sub-chunks. This gate engages deepPipe only for
// chunks <= MAX_MB per PE and forces every larger chunk onto the whole-chunk crown
// fence (single quiet + AMO, bit-exact at scale), so E2E stays bit-exact while the
// medium AGs still ride the pipeline. 0 => no gate.
inline size_t HierDeepPipeMaxBytes() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_MAX_MB");
  if (e == nullptr || e[0] == '\0') return 0;
  long v = std::atol(e);
  if (v <= 0) return 0;
  return static_cast<size_t>(v) * 1024ull * 1024ull;
}
// Signal-pipe per-pair coherence ceiling (MORI_HIER_SIGNAL_MAX_MB, default 0 = no cap).
// The signal-pipe (deepPipeQuiet=0 = fused put-with-signal landing) rides the
// completion AMO on the same RC QP as its data, so the flag never precedes the bytes.
// On some node pairs the NIC-DMA->HBM coherence window is tighter and the put-AMO can
// outrun its own data for large chunks (hangs / "Slow wait" stalls). This ceiling lets
// such a pair cap signal-pipe engagement to chunkBytes < ceiling and fall the rest back
// onto the scale-robust quiet-drain landing fence (deepPipeQuiet=1) without a rebuild.
// Default 0 keeps the signal path engaged at all sizes; a hang-prone pair sets e.g.
// MORI_HIER_SIGNAL_MAX_MB=48 to harden. Bit-exact on both branches.
inline size_t HierSignalMaxBytes() {
  const char* e = std::getenv("MORI_HIER_SIGNAL_MAX_MB");
  if (e == nullptr || e[0] == '\0') return 0;
  long v = std::atol(e);
  if (v <= 0) return 0;
  return static_cast<size_t>(v) * 1024ull * 1024ull;
}
// Sub-chunk byte target for DEEP_PIPE=auto (default 16MiB). chunkBytes is the total AG
// bytes; depth = round(chunkBytes / target) clamped [1,16]. A 16MiB sub-chunk fills the
// mlx5 NIC DMA better than smaller sizes while staying under the per-sub-chunk coherence
// window (HierDeepPipeMaxBytes), so the landing fence is unchanged and the path stays
// bit-exact. Only reached on the explicit DEEP_PIPE=auto opt-in.
inline size_t HierDeepPipeSubBytes() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_SUBBYTES");
  if (e == nullptr || e[0] == '\0') return 16ull * 1024ull * 1024ull;
  long long v = std::atoll(e);
  return (v > 0) ? static_cast<size_t>(v) : 16ull * 1024ull * 1024ull;
}
// Lower size gate for DEEP_PIPE (MORI_HIER_DEEP_PIPE_MIN_MB, default 0 = no floor). The
// MAX gate is on the sub-chunk (chunkBytes/dp); this floor gates on the per-PE total
// chunkBytes so deep-pipe can be pinned to a clean [MIN_MB, MAX_MB per PE] window and
// every chunk outside it falls through to the plain/crown path. A gated-off chunk takes
// the same code path as deepPipe<=1, so this is bit-exact. 0 => no floor (byte-identical).
// Only meaningful together with MORI_HIER_DEEP_PIPE>1.
inline size_t HierDeepPipeMinBytes() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_MIN_MB");
  if (e == nullptr || e[0] == '\0') return 0;
  long v = std::atol(e);
  if (v <= 0) return 0;
  return static_cast<size_t>(v) * 1024ull * 1024ull;
}
// Small-chunk single-shot gate (MORI_HIER_DP_SMALL_MB, default 0 = OFF). The default
// runs DEEP_PIPE=2 at every size, so small buffers pay the deep-pipe's second
// per-sub-chunk landing round-trip (extra remote flag AMO + the reassembly reader's
// extra per-slot quiet drain). At small sizes the transfer is latency-bound and there
// is too little inter-fill span to hide the intra reassembly under, so that handshake
// is pure exposed overhead. This gate forces fused.deepPipe=1 (the single-shot
// whole-chunk crown fence, one flag round-trip) for every per-PE chunkBytes <=
// threshold, leaving larger chunks on the pipeline where the overlap pays. Bit-exact:
// deepPipe==1 is the single-shot crown path. Default 0 => byte-identical path.
inline size_t HierDpSmallBytes() {
  const char* e = std::getenv("MORI_HIER_DP_SMALL_MB");
  if (e == nullptr || e[0] == '\0') return 0;
  long v = std::atol(e);
  if (v <= 0) return 0;
  return static_cast<size_t>(v) * 1024ull * 1024ull;
}
// Ragged-sub-chunk guard for DEEP_PIPE=auto (MORI_HIER_DP_CLEAN, default ON). Auto
// resolves depth d=round(chunkBytes/sub) then the kernel splits into
// subChunk=chunkBytes/d with a ragged remainder tail whenever d does not divide
// chunkBytes evenly, and the pipeline stalls draining an under-filled last sub-chunk
// (a severe throughput collapse for non-power-of-2 sizes). This guard snaps the
// resolved auto depth down to the nearest divisor of chunkBytes (16B-aligned units) so
// every sub-chunk is equal and the ragged tail is eliminated. Down-only (dp' <= dp), so
// the sub-chunk count never exceeds what the Python flag sizer allocated. Bit-exact:
// equal valid sub-ranges of the same bytes in the same per-peer RC order, all drained
// before the completion flag. It is a no-op when the auto depth already divides
// chunkBytes evenly.
inline bool HierDpCleanDepthOn() {
  const char* e = std::getenv("MORI_HIER_DP_CLEAN");
  // Default ON: the crown E2E path runs DEEP_PIPE=auto, and a real FSDP layer whose
  // per-PE bytes are not a clean 16MiB*2^n multiple would hit the ragged-tail collapse.
  // The guard is a no-op for power-of-2 sizes and down-only, so default-ON is strictly
  // safe-or-better. Set MORI_HIER_DP_CLEAN=0 to restore the ragged auto-depth path.
  if (e == nullptr || e[0] == '\0') return true;
  return !(e[0] == '0' && e[1] == '\0');
}
// WRITE_WITH_IMM per-sub-chunk landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_IMM,
// default OFF). recv-CQE = definitive remote-landing (RC in-order per QP), but the
// WRITE_WITH_IMM path is HW-unavailable on this mlx5 provider (asserts out), so this
// stays OFF here; kept for portability. Only meaningful when deepPipe>1.
inline bool HierDeepPipeImmOn() {
  const char* e = std::getenv("MORI_HIER_DEEP_PIPE_IMM");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Quiet-fence per-sub-chunk landing for DEEP_PIPE (MORI_HIER_DEEP_PIPE_QUIET, default
// OFF). The put-with-signal AMO (deepPipeImm==0) fails bit-exact for large chunks
// because the AMO can beat its own data landing; WRITE_WITH_IMM (deepPipeImm==1) is
// HW-unavailable on this mlx5 provider. This is the scale-robust option: dedicate one
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
// Deep-pipe serial drain (MORI_HIER_DP_SERIAL_DRAIN, default OFF). The parallel drain
// relaxes the order of independent per-sub-chunk QP-group completions (leader warp p
// drains only its own group before publishing flag p). That relaxation is bit-exact in
// principle (disjoint QP groups, per-slot flags), but under contention it can be a
// source of small residual E2E drift. This lever forces the strictly-ordered thread-0
// serial drain (quiet grp0->AMO0->...->quietP->AMOP, each sub-chunk fully
// drained+published in temporal order). Bit-exact (same drains/AMOs/flag slots, only
// the least-relaxed completion order). Default OFF => byte-identical parallel path.
inline bool HierDpSerialDrainOn() {
  const char* e = std::getenv("MORI_HIER_DP_SERIAL_DRAIN");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Skewed temporal split (MORI_HIER_DP_TAIL_PCT, default 0 = uniform). The deep-pipe
// splits the per-PE chunk into P equal temporal sub-chunks. The reassembly of the last
// sub-chunk cannot overlap the inter NIC fill (nothing is left crossing the NIC after
// it lands), so it runs exposed. This lever front-loads the P==2 split so the last
// sub-chunk carries only dpTailPct% of the chunk, shrinking that exposed tail. Producer
// (all_gather.hpp) and consumer (ccl_kernels.hip) compute the identical boundary
// (nUnits - nUnits*pct/100), so flag slot p guards exactly the reassembled bytes.
// Bit-exact: same union of bytes, same per-slot flags, same landing->consume order,
// only the sub-chunk size ratio changes. Only engages for P==2 with 0<pct<50; 0 or
// out-of-range => byte-identical uniform path.
inline int HierDpTailPct() {
  const char* e = std::getenv("MORI_HIER_DP_TAIL_PCT");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = std::atoi(e);
  // <50 front-loads (small last sub-chunk); >50 head-skews (small first, to cut
  // first_land latency at small sizes). 50 or out-of-range => 0 (uniform, byte-identical).
  if (v <= 0 || v >= 100 || v == 50) return 0;
  return v;
}
// First-land idle-engine reclamation (MORI_HIER_LOCAL_OFFLOAD, default 0 = off). In the
// fused remote crown the reasm CTAs (SDMA queue 1) spin idle during first_land waiting
// for the first inter-node RDMA sub-chunk, while the local-block CTA (queue 0) grinds
// the whole own-block gather alone. This lever offloads the last ``offloadPeers`` of the
// G local-block peer-columns from the queue-0 local CTA onto the otherwise-idle reasm
// CTA, which pushes them (on queue 0, per-peer signal slots disjoint from the main CTA's
// remaining peers) during its pre-land spin, then releases to reasm on queue 1. The
// offload is time-disjoint (local slice runs only in the inter-fill idle window) and
// peer-disjoint (main CTA does [0,G-K), offload does [G-K,G)), so the two engines never
// oversubscribe. Bit-exact: the union of pushed peers is unchanged (all G columns
// written exactly once, same bytes, same flag[groupPos] AMO per target peer), only which
// CTA issues the last K columns. Clamp [0, G-1] (never offload the whole gather). Kept
// off by default: the ring||reasm overlap is HBM-bandwidth-bound, so extra XGMI work in
// the inter window contends with the RDMA fill and regresses. 0 => byte-identical crown.
inline int HierLocalOffload() {
  const char* e = std::getenv("MORI_HIER_LOCAL_OFFLOAD");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = std::atoi(e);
  return v > 0 ? v : 0;
}
// Full-width deep-SQ in-flight FIFO (MORI_HIER_FIFO, default 0 = byte-identical
// path). When on, the temporal deep-pipe (deepPipe>1) drives every sub-chunk over the
// full sw-QP fan-out and issues the P sub-chunks back-to-back so each QP carries P
// in-flight WQEs (deep SQ = NIC fill), then a single parallel per-QP drain + P flag AMOs
// -- pipeline depth decoupled from ring width. Bit-exact (same bytes, same flags, AMOs
// still follow their landings); only engages deepPipe>1.
inline bool HierFifoFullWidthOn() {
  const char* e = std::getenv("MORI_HIER_FIFO");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Progressive deep-pipe publish (MORI_HIER_FIFO_PROG, default 0 = byte-identical
// path). fifoFullWidth issues all P sub-chunks deep then batch-drains + batch-publishes
// all P flags together -- a completion barrier where flag[0] cannot fire until sub-chunk
// P-1 lands, so the receiver's reassembly of sub-chunk 0 never overlaps the inter fill of
// 1..P-1. This lever runs the tail-per-step model instead: for each sub-chunk p in strict
// temporal order, issue it at full sw-QP width, drain its own send-CQ, then AMO+publish
// chunkReadyFlags[p] immediately before issuing p+1, so the reassembly worker reassembles
// p over XGMI while p+1 is still crossing the NIC. Engages only on the deep-pipe ring
// path (deepPipe>1); takes precedence over fifoFullWidth. Kept off by default: the
// default deep-pipe path already captures the overlap, and enabling this has exposed an
// E2E completion drift the standalone bit-exact check does not catch.
inline bool HierFifoProgOn() {
  const char* e = std::getenv("MORI_HIER_FIFO_PROG");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Per-QP fine-grain inter-arrival drain (MORI_HIER_SHARD_DRAIN, default OFF). The
// deep-pipe exposes P temporal sub-chunks (each fans across sw/P QPs, drained as a group
// before its flag), so the fused reasm cannot begin until a full sub-chunk lands. This
// lever goes finer than the temporal P: issue the whole chunk at full sw-QP width, then
// drain + publish each QP's own 16B-aligned shard the instant it lands, so the single
// reasm worker (partition==numQp==sw) can push shard s while shards s+1..sw-1 still cross
// the NIC. It adds no XGMI/HBM bytes (only re-times the consume finer) and attacks the
// latency-bound first_land prefix. Bit-exact: the sw per-QP byte ranges exactly tile the
// chunk and match the consumer's partition==sw unitsPerChan; each flag AMO strictly
// follows its own QP drain + system fence. Deadlock-free: the full send is issued before
// any wait, and every wait is on our own inbound flags. Kept off by default: the extra
// per-shard AMO round-trips and finer reasm drain exceed the first_land latency they save.
// Default 0 => byte-identical crown.
inline bool HierShardDrainOn() {
  const char* e = std::getenv("MORI_HIER_SHARD_DRAIN");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// DIRECT-LAND: RDMA-write the received remote block straight into the final output
// self-slot, deleting the ring->output self copy on the reasm leg (see the field
// comment on CclFusedRingRemoteGatherArgs::directLand). MORI_HIER_DIRECT_LAND=1.
inline bool HierDirectLandOn() {
  const char* e = std::getenv("MORI_HIER_DIRECT_LAND");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Direct-land self-column skip (default 0 = do NOT skip). When direct-land RDMA-lands
// the block into the output self-slot, skipping the reasm self copy leaves that slot
// NIC-written-only, outside the copy-engine+fence coherence path the peer columns get,
// so the consuming CU read can observe a stale line at that slot. Default keeps the
// identity self copy for coherence; MORI_HIER_DIRECT_LAND_SKIPSELF=1 restores the
// incoherent skip. No effect unless direct-land is on.
inline int HierDirectLandSkipSelf() {
  const char* e = std::getenv("MORI_HIER_DIRECT_LAND_SKIPSELF");
  return (e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0')) ? 1 : 0;
}
// Every-GPU-direct write-push fan-out (MORI_HIER_EVERY_DIRECT_WRITE, default OFF). The
// path receives each rank's shard over the ring then XGMI-broadcasts it to all G
// local peers (the reasm scatter). This lever instead has each rank push its own
// already-staged ring self-slot directly to all G receivers on the remote node, straight
// into the receiver's final output self-slot ((nodeId*G+groupPos)*chunkBytes), fused with
// an AMO_SET of the receiver's completion flag on the same QP (RC in-order, so the flag
// can't beat the data). The push target is terminal output (no subsequent SDMA read), so
// nothing re-reads it and the completion reader's full-grid re-touch is the coherence
// fence. This deletes the XGMI scatter: the remote half arrives pre-placed via G distinct
// dest NICs. Default OFF => byte-identical crown.
inline bool HierEveryDirectWriteOn() {
  const char* e = std::getenv("MORI_HIER_EVERY_DIRECT_WRITE");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Device flag-token (MORI_HIER_FLAG_TOKEN_DEV, default OFF; requires
// MORI_HIER_GEN_RING_DBL so the device parity counter exists). The barrier-free gen-ring
// drops the per-op entry ShmemBarrierOnStream that used to drain stale chunkReadyFlags
// each HIP-graph replay. Without it the host flags.zero_ (which under graph capture runs
// once at capture) leaves the landing flags accumulated at the prior op's value, so the
// next op's reassembly wait `< 1` is instantly satisfied by the stale slot and consumes
// before its bytes land. A host-side per-op token has the same problem (the host counter
// also freezes at capture). This makes the flag generation device-derived: the crown
// reads the graph-safe device parity counter (bumped once/op by RingParityBumpKernel) as
// the per-op opGen, so the publisher stores parity[0] and the reasm wait gates on
// `< parity[0]`, letting the higher token supersede the stale slot with no host reset.
// Composes with the data double-buffer. Default OFF => byte-identical crown (opGen=0
// legacy fixed-1 + host reset).
inline bool HierFlagTokenDevOn() {
  const char* e = std::getenv("MORI_HIER_FLAG_TOKEN_DEV");
  return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}
// Pipelined relay ring reassembly (MORI_HIER_REASM_RING, default OFF). The default reasm
// is a flat G-way scatter: every GPU broadcasts its shard of each remote block to all G
// group peers, a G-way XGMI incast the copy engines cannot saturate. This replaces it
// with a G-1 step relay ring where each step is a perfect matching (send the shard you
// hold to your ring successor, receive the next from your predecessor). Peak concurrent
// outbound per GPU drops from G-1 to 1, every link runs at full BW, and it is
// deadlock-free (each rank trails its predecessor by one pipeline step). It sets the same
// per-shard completion flags (flagBase+s) the completion reader expects, so it is a
// drop-in for the scatter. Default OFF => byte-identical crown.
inline int HierReasmRing() {
  const char* e = std::getenv("MORI_HIER_REASM_RING");
  return (e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0')) ? 1 : 0;
}
// Mid-buffer pipe-engage floor (MORI_HIER_DP_MIN_DEPTH, default 1 = off). At large world
// sizes the per-PE ring shard is small, so the auto sub-target can round the pipe depth
// to 1 for mid-size buffers, leaving the temporal pipeline inert and the inter NIC fill
// running serially before the intra XGMI reasm. This floor forces the auto/explicit depth
// up to MIN_DEPTH so those mid buffers pipeline; the MAX_MB window + floor gates still
// cage the sizes that hang/regress. Bit-exact: a deeper valid depth pipelines the same
// bytes in the same per-peer RC order, all sub-chunks drained before the completion flag.
// Default 1 => byte-identical path (dp<=1 skips the block).
inline int HierDeepPipeMinDepth() {
  const char* e = std::getenv("MORI_HIER_DP_MIN_DEPTH");
  if (e == nullptr || e[0] == '\0') return 1;
  int v = std::atoi(e);
  if (v < 1) return 1;
  if (v > 16) return 16;
  return v;
}
// Size-gated big-chunk deep-pipe depth (MORI_HIER_DP_BIG=D, default 0 = off). A flat
// DEEP_PIPE=4 applied at every size helps the large buffers but regresses the small ones
// (at small per-PE sizes the extra per-sub-chunk landing handshake is exposed latency
// with too little inter-fill span to overlap). This lever bumps the resolved depth to D
// only when the per-PE chunk is large enough to hide the deeper pipeline's tail
// (chunkBytes >= HierDpBigBytes, default 6MiB/PE). Bit-exact: a deeper valid depth
// pipelines the same bytes in the same per-peer RC order, every sub-chunk drained before
// the completion flag. The DP_CLEAN ragged guard still snaps D down to a divisor of
// chunkBytes. The residual is largely depth-invariant (transport-fill-bound), so this is
// typically only a marginal lever. Default 0 => byte-identical crown.
inline int HierDpBigDepth() {
  const char* e = std::getenv("MORI_HIER_DP_BIG");
  if (e == nullptr || e[0] == '\0') return 0;
  int v = std::atoi(e);
  if (v < 0) return 0;
  if (v > 16) return 16;
  return v;
}
// Per-PE byte threshold for HierDpBigDepth (MORI_HIER_DP_BIG_MB, default 6 MiB/PE).
// chunkBytes >= this => the big-chunk depth bump engages. The 6MiB default engages
// the large per-PE buffers where the deeper pipeline's tail is hidden and excludes
// the smaller ones where a deeper depth regresses. Only meaningful with
// MORI_HIER_DP_BIG>1.
inline size_t HierDpBigBytes() {
  const char* e = std::getenv("MORI_HIER_DP_BIG_MB");
  if (e == nullptr || e[0] == '\0') return 6ull * 1024ull * 1024ull;
  long v = std::atol(e);
  if (v <= 0) return 6ull * 1024ull * 1024ull;
  return static_cast<size_t>(v) * 1024ull * 1024ull;
}

// Host-proxy inter + device reassembly (MORI_HIER_HOSTPROXY_REASM, default 0 = off).
// The device deep-pipe (temporal sub-chunk) needs a per-sub-chunk landing signal, but
// both device options are unavailable/unreliable on this mlx5 provider (WRITE_WITH_IMM
// recv-CQE is HW-unavailable; per-sub-chunk quiet+AMO races or crashes at large sizes).
// The scale-robust per-sub-chunk landing proof is the host send-CQ drain. This lever
// hands the inter-node fill to a host proxy that posts sub-chunk p's cross-node RDMA
// writes into the ring buffer, drains its send-CQ (== sub-chunk landed remotely), and
// publishes chunkReadyFlags[p] from the host. The device kernel then runs only the intra
// SDMA reassembly pipeline, spinning on the host-published flags exactly as it does for
// device-published flags, so sub-chunk p's XGMI reassembly overlaps sub-chunk p+1 still
// on the NIC. Requires a host proxy thread (the fused E2E path); the standalone crown has
// no host poster and would spin forever. When ON the device ring-send blocks (bx <
// ringBlocks) skip the RDMA send; the reassembly workers + completion reader are
// unchanged. Default 0 = OFF (byte-identical path: device posts inter and
// publishes the flags).
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
  // simultaneous gather_kernel_direct launches (MORI_HIER_REASM_STREAMS) never
  // race on the shared flag slots -- the same mechanism the FUSED reassembly
  // kernel already uses, now available to the direct multi-stream path.
  size_t flagBase;
  // (MORI_INTRA_MQ): !=0 -> split each peer column across all sdmaNumQueue
  // SDMA queues (both KFD-recommended XGMI engines per link) instead of driving
  // only queue 0. Raises per-link intra fill toward the native single-node ring.
  // Default 0 keeps the single-queue put byte-for-byte.
  int multiQueue = 0;
};

// Fused hierarchical param-contiguous SubGroup gather. One launch replaces the
// per-(node-block, param) loop that ``HierAllGather.enqueue_param_contiguous``
// used to issue (N_nodes * N_params separate SubGroup launches, whose launch
// overhead erased the copy-out saving vs native). Each of ``G`` group members
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
  // Transport-level flag-can't-beat-data: when non-zero, the single-warp RDMA
  // ring send fuses the data WRITE and the completion-flag AMO into one
  // ShmemPutMemNbiSignal call so the signal WQE rides the same QP strictly
  // after the data WRITE. RC in-order execution then guarantees the remote
  // peer's data has physically landed before its flag is observable -- closing
  // the residual FSDP loss completion race without any host sync (the flag can
  // never beat its data). Default 0 = the separate put + quiet + AMO.
  int usePutSignal = 0;
  // WRITE_WITH_IMM (env MORI_HIER_RING_WRITE_IMM, default 0). On the single-warp
  // cross-node (RDMA) ring path, replace the data PUT + QP quiet + flag AMO with
  // an RDMA_WRITE_WITH_IMM and have the receiver consume the recv-CQ completion
  // instead of spinning the flag. The recv-CQE cannot be observed before the
  // write payload has landed globally, so this closes the remote-landing
  // stale-read race that no device-side barrier/quiet fixes, without the host
  // stall. Default 0 = the separate put+quiet+flag path, byte-for-byte unchanged.
  int useWriteImm = 0;
  // RDMA-READ (PULL) ring (env MORI_HIER_RING_READ, default 0). On the
  // single-round (ringSize==2) all-RDMA inter-node phase -- exactly the 2-node
  // hierarchical AG this cluster runs -- the chunk each PE needs is prevPeer's
  // own chunk, already present after the intra prepare barrier. Instead of
  // relying on the peer to PUSH it (a GPU-initiated RDMA WRITE, which hits a
  // per-QP throughput wall on mlx5), PULL it with an RDMA READ. A READ
  // completion drained by our own quiet is a consumer-side landing guarantee:
  // the bytes are physically in this PE's ring buffer and, with a system fence,
  // visible to its CUs -- no cross-PE flag AMO, no receiver spin, no remote-
  // landing race (the E2E accuracy race the push+flag path exposes). Byte-
  // identical result (same slot, same bytes). Default 0 = the push path,
  // byte-for-byte unchanged.
  int useRead = 0;
  // WRITE-PUSH (SEND-CQ) per-channel landing fence (env MORI_HIER_RING_WRITE,
  // default 0). The write-side counterpart of useRead/multiBlockRead. On the giant
  // multiBlock AG each channel CTA pushes its sub-range as a fused put-with-signal on
  // qpId=bid then drains that QP's SEND CQE; the receiver spins its per-channel inbound
  // flag. Keeps RDMA-WRITE fill where the READ path underfills, bit-exact by
  // construction. Default 0 = push path unchanged.
  int useWriteFence = 0;
  // Generation-counter barrier-free ring (env MORI_HIER_GEN_RING, default 0).
  // When non-zero this holds the monotonically-increasing per-op generation
  // number (op 1, 2, 3, ...). On the classic single-increment flag path
  // (expectedRecvSig==1, no put-signal / write-imm / fan-out) the sender's
  // AMO_ADD(1) is left to accumulate across ops (the kernel skips the per-op
  // flag reset), so slot k holds exactly ``opGen`` after ``opGen`` ops. The
  // receiver then waits for the slot to reach ``opGen`` instead of 1. Because
  // the flags are never reset, the prepare-time entry barrier (whose sole job
  // was to order every PE's op-end reset before any peer's next-op increment)
  // is no longer needed and is skipped in prepare_stream -- removing one of the
  // two global on-stream barriers per ring round. The trailing finish reuse
  // barrier is kept, so ring-buffer reuse ordering is unchanged. Default 0
  // keeps the reset+entry-barrier path byte-for-byte identical.
  uint64_t opGen = 0;
  // Device-side generation counter (env MORI_HIER_GEN_RING_DEV, default null).
  // The host GEN_RING (opGen above, stamped by ++ringOpGen_ in prepare_*) is
  // graph-incompatible -- under HIP-graph replay prepare_* runs once at capture so
  // opGen is frozen while the accumulating flags keep advancing, so the receiver's
  // `wait flag>=opGen` gate desyncs on every replay after the first. Since the
  // launch-collapse win depends on graph replay, host GEN_RING never reached the
  // fuse_local crown. This pointer instead lets the device increment the per-op
  // generation itself (one uint64 per ring block/channel):
  // each kernel execution (eager warmup OR graph replay) bumps opGenCounter[bx]
  // and uses that as this op's generation, staying in lockstep with the per-op
  // AMO_ADD(1) the sender applies to the same block's flag region -- so the
  // barrier-free accumulating-flag protocol works under graph replay. nullptr =>
  // classic reset+entry-barrier path (byte-identical crown).
  uint64_t* opGenCounter = nullptr;
  // Device double-buffered ring (env MORI_HIER_GEN_RING_DBL, requires GEN_RING_DEV).
  // The barrier-free gen-ring drops both the entry and finish per-op fences, exposing
  // a cross-PE ring-buffer reuse race: a peer's op N+1 RDMA push can overwrite the half
  // this PE's op N reassembly still reads. This buffer B is a second symmetric ring;
  // the crown alternates ring<->ringB by op parity (parityCounter[0] & 1), so op N+1
  // lands in the other half than op N reassembles -- op N+2 reuses op N's half only
  // after 2 ops of flag-chain separation (provably drained). Empty => single-buffer
  // path (byte-identical crown).
  application::SymmMemObjPtr ringMemObjB;
  // Per-op parity counter (one device uint64, host-allocated in the ring handle,
  // bumped once/op by a captured pre-kernel so it advances under HIP-graph replay
  // AND every crown block reads the same stable value). null => no double-buffer.
  uint64_t* parityCounter = nullptr;
  // FUSE-ENTRY-BARRIER: 1 => the crown does the cross-PE entry rendezvous
  // device-side (block 0 -> ShmemBarrierAllBlock) at its prologue instead of the
  // separate host-launched ShmemBarrierOnStream (collapses one graph node/op).
  // gridArrival = 2-word per-PE HBM scratch ([0]=arrival counter, [1]=monotonic
  // release generation) gating the grid-wide arrival barrier. 0/null => the default
  // path (separate host barrier), byte-identical.
  int fuseEntryBarrier = 0;
  unsigned int* gridArrival = nullptr;
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
  // Propagate the ring completion-protocol flags into the fused path. The fused ring
  // previously dropped these (defaulting to the plain put+quiet+flag protocol), so the
  // big cross-node AG that runs through this fused kernel never saw usePutSignal /
  // useWriteImm even when the env enabled them. Carrying them here lets the fused ring
  // engage the fanOut WRITE_WITH_IMM (recv-CQ landing proof) on the big AG. Default 0 =>
  // byte-identical to the fused path.
  int usePutSignal = 0;
  int useWriteImm = 0;
  // In-kernel copy-in launch-collapse: fold the host hipMemcpyAsync copy-in of this
  // PE's input into its ring slot into the fused local kernel (each ring channel CTA
  // stages its own send sub-range of gInput then __syncthreads before the put).
  // Mirrors the FusedRingRemoteGatherKernel fuseCopyIn. Drops one GPU op (the prepare
  // copy-in) while keeping the ring||local-gather concurrency. Kept off by default: the
  // in-kernel stage copies the whole per-PE shard on CU threads (far slower than the
  // copy engine) and __syncthreads serialises it ahead of the RDMA put, forfeiting the
  // ring||local overlap. The copy-engine host copy-in is the correct bit-exact path.
  int fuseCopyIn = 0;

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
  // Multi-engine per-link local gather (see HierReasmMultiQueueOn /
  // MORI_INTRA_MQ). The crown's Phase-B local-block gather calls
  // OneShotAllGatherSdmaSubGroupKernel_body, which supports a multiQueue arg but
  // was previously invoked with the default 0 -> the local 8-shard XGMI gather
  // drove only queue 0 of the link's recommended engines. 1 => split each peer's
  // column across all sdmaNumQueue queues (both KFD-recommended XGMI SDMA engines
  // per link) to raise per-link intra fill. Bit-exact by construction (disjoint
  // sub-ranges of the same bytes; the body's ShmemQuietThread drains every queue
  // before the flag AMO). 0 = OFF (byte-identical crown path).
  int multiQueue = 0;
  // Copy-engine inline-flag completion (see HierQFlagOn / MORI_HIER_QFLAG).
  // The crown's Phase-B local gather delivers each peer flag via ShmemQuietThread
  // drain + __threadfence_system + a separate P2P AMO (3 per-peer round-trips on
  // the 8x7 all-to-all completion critical path). 1 => ride the flag on the same
  // SDMA queue as its data copy (COPY_LINEAR + FENCE, FIFO-ordered) -- the LL-style
  // copy-engine completion model, stripping the per-peer drain/fence/AMO. Bit-exact
  // by construction (fence FIFO-ordered after bytes; reader's seq-cst SYSTEM acquire
  // unchanged). 0 = OFF (byte-identical crown path).
  int qFlag = 0;
  // Staggered-permutation ring schedule (see
  // HierRingWidth / MORI_HIER_RING). 1 => the crown's local gather issues its
  // G-1 peer copies in G-1 rotated PHASES (permutation schedule) instead of the
  // concurrent full-mesh burst. Bit-exact by construction (same bytes/slots/
  // completion, only issue order changes). 0 = OFF (byte-identical crown).
  int ringPhased = 0;
  // Batched sender-side completion fence (see HierBatchFence /
  // MORI_HIER_BATCH_FENCE). 1 => collapse the G concurrent per-peer
  // __threadfence_system in the crown local-gather tail to ONE CTA-wide fence
  // (drains + flag AMOs stay parallel per peer). Bit-exact (strictly stronger
  // order). 0 = OFF (byte-identical crown: per-peer fence retained).
  int batchFence = 0;
  // HIERARCHICAL 2x4 INTRA GATHER (see HierH2x4 / MORI_HIER_H2X4). 1 => the
  // crown local-block gather runs the 2-phase sub-group broadcast (peak concurrent
  // outbound copies H=G/2 instead of G-1). Bit-exact. 0 = OFF (byte-identical crown).
  int h2x4 = 0;
  // 2x4 STACKED-FLAT-BODY intra wave width (see HierIntra2 /
  // MORI_HIER_INTRA2). W>0 => the crown local-block flat push issues in ceil(G/W)
  // drained W-regular-matching waves (per-PE concurrent egress + per-receiver incast
  // G-1 -> W). W==4 at G==8 = the mandated 2x4. Bit-exact. 0 = OFF (byte-identical).
  int intra2 = 0;
  // Device-side gen-ring (MORI_HIER_GEN_RING_DEV): per-ring-block device generation
  // counter. Non-null => ring channel bx computes this op's generation as
  // ++opGenCounter[bx] (device-side, advances on every graph replay), passes it as
  // opGen into AllGatherRingSubGroupKernelBody, and prepare drops the entry barrier.
  // null => classic entry-barrier crown (byte-identical path).
  uint64_t* opGenCounter = nullptr;
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
  // carry the device-side gen-ring counter (null unless GEN_RING_DEV on).
  fused.opGenCounter = r->opGenCounter;
  fused.numQp = r->numQp;
  fused.ringBlocks = ringBlocks < 1 ? 1 : ringBlocks;
  // Fused FSDP path: put-signal only when explicitly env-enabled (the standalone
  // default-on must not leak into the E2E deferred/overlap bytes, keeping the loss
  // byte-identical to native). See HierRingPutSignalExplicitlyOn.
  fused.usePutSignal = HierRingPutSignalExplicitlyOn() ? r->usePutSignal : 0;
  fused.useWriteImm = r->useWriteImm;
  // fold the copy-in into the fused local kernel when MORI_HIER_FUSE_COPYIN.
  fused.fuseCopyIn = HierFuseCopyInOn() ? 1 : 0;

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
  // multi-engine per-link local gather (MORI_INTRA_MQ). Default OFF =>
  // byte-identical crown path; bit-exact by SdmaPutWarp construction.
  fused.multiQueue = HierReasmMultiQueueOn() ? 1 : 0;
  // inline-flag completion on the crown gather (MORI_HIER_QFLAG). Default OFF =>
  // byte-identical crown path; bit-exact by the FIFO-ordered
  // COPY_LINEAR+FENCE construction. Mutually exclusive with multiQueue in the body.
  fused.qFlag = HierQFlagOn() ? 1 : 0;
  // width-W staggered-permutation ring schedule on the crown gather
  // (MORI_HIER_RING=W). ringPhased carries the phase width (0=OFF,
  // 1=single-link matching, W>1=W concurrent links/phase). Default OFF =>
  // byte-identical crown path; bit-exact by construction (same
  // bytes/slots/completion tail, only the issue order/stagger changes).
  fused.ringPhased = HierRingWidth();
  fused.batchFence = HierBatchFence() ? 1 : 0;
  fused.h2x4 = HierH2x4();
  fused.intra2 = HierIntra2();

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
  int usePutSignal = 0;
  int useWriteImm = 0;
  int useRead = 0;        // RDMA-READ (PULL) multiBlockRead landing fence (MORI_HIER_RING_READ)
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
  // CU-coherent re-touch of the landed local output in the completion reader
  // (MORI_HIER_FUSE_REMOTE_RETOUCH; hardcoded 0). 0 = OFF (byte-identical path).
  int retouchOut = 0;
  // Elastic reassembly (see HierFuseRemoteElasticOn). 1 => the local-block CTA
  // joins remote reassembly on SDMA queue 0 after its own gather, so the tail
  // uses reasm+1 concurrent SDMA queues. 0 = OFF (byte-identical path).
  int elasticReasm = 0;
  // Deep-SQ WQE depth per QP for the inter-node ring put (see HierWqeDepth).
  // 1 = single WQE per QP (byte-identical path); d>1 posts d back-to-back
  // sub-puts per QP so the NIC SQ carries d in-flight WQEs per QP.
  int wqeDepth = 1;
  // Deep-SQ temporal pipeline depth (see HierDeepPipe). P>1 splits the chunk into
  // P temporal sub-chunks with per-sub-chunk landing flags so reassembly overlaps
  // the still-in-flight later sub-chunks. 1 = OFF (byte-identical path).
  int deepPipe = 1;
  // Deep-SQ temporal pipeline landing fence: 0 = per-sub-chunk put-with-signal AMO
  // (fails bit-exact >=64MB/P4 -- the AMO can beat its own large-transfer data);
  // 1 = per-sub-chunk RDMA_WRITE_WITH_IMM (recv-CQE = definitive remote-landing,
  // RC in-order per QP). See HierDeepPipeImmOn. Only meaningful when deepPipe>1.
  int deepPipeImm = 0;
  // Deep-SQ temporal pipeline QUIET landing fence (see HierDeepPipeQuietOn). 1 =>
  // each temporal sub-chunk rides its own QP with a plain put; thread drains that
  // QP's send-CQ (ShmemQuietThread(pe,qpId) = sub-chunk landed remotely) BEFORE the
  // separate flag AMO, so the flag never fires ahead of the data landing (bit-exact
  // at scale, unlike the racy fused put-signal). 0 = OFF. Only when deepPipe>1.
  int deepPipeQuiet = 0;
  // Deep-pipe serial drain (see HierDpSerialDrainOn). 1 => force the strictly
  // temporal thread-0 serial drain in the deepPipeQuiet path instead of the
  // parallel/WRAP leader-per-group drain. Bit-exact (same drains/AMOs/flags, most
  // conservative completion order). 0 = OFF = the parallel path. Only deepPipe>1.
  int dpSerialDrain = 0;
  // Skewed temporal split (see HierDpTailPct). >0 && <50 => the P==2 deep-pipe's
  // LAST sub-chunk carries only dpTailPct% of the chunk (front-load), shrinking the
  // exposed intra reassembly tail. 0 => uniform ceil tiling (byte-identical).
  int dpTailPct = 0;
  // FIRST-LAND idle-engine reclamation (see HierLocalOffload). K = # of the G local-
  // block peer-columns pushed by the idle reasm CTA during first_land instead of the
  // queue-0 local CTA. 0 = off => byte-identical crown.
  int offloadPeers = 0;
  // Host-proxy inter + device reassembly (see HierHostProxyReasm). 1 => the device
  // ring-send blocks SKIP the RDMA send (a host proxy owns the inter leg and
  // publishes chunkReadyFlags[p] from the host after its send-CQ drains); the
  // device reassembly workers + completion reader are UNCHANGED. 0 = OFF
  // (byte-identical path: device posts inter and publishes the flags).
  int hostProxyInter = 0;
  // Same-CTA inline reassembly (see HierInlineReasmOn). 1 => on the multiBlock ring
  // (ringBlocks>1) the ring channel CTA runs FusedRemoteReassembleWorker for its own
  // landed sub-range inline (no chunkReadyFlags relay to a dedicated CTA). The Python
  // launcher drops the dedicated reassembly CTAs (grid rb+1). 0 = OFF (2*rb+1 path).
  int inlineReasm = 0;
  // In-kernel copy-in (see HierFuseCopyInOn). 1 => each ring channel CTA stages its
  // own send sub-range of gInput into the local ring slot before the put (the host
  // hipMemcpyAsync copy-IN is skipped on the Python side, prepare_stream_in_place).
  // 0 = OFF (byte-identical path: host copies input into the slot).
  int fuseCopyIn = 0;
  // Local-block push-only (see HierLocalPushOnly). 1 => the bx==rb local node-block
  // gather is push-only (no coupled per-slot wait) and its completion is folded
  // into the completion reader (which then also drains flag slots [0,G)); this
  // removes the DEEP_PIPE>=8 "Slow wait for sub-group pos" deadlock. Byte-identical
  // output. 0 = OFF (coupled push+wait, byte-identical path).
  int localPushOnly = 0;
  // Cooperative spin-backoff (see HierSpinBackoff). 1 => the flag-wait spin loops
  // (chunkReadyFlags + completion reader) issue an s_sleep between polls so waiting
  // waves yield fabric/L2 bandwidth to the SDMA reassembly pushes + mlx5 CQ
  // drainers under DEEP_PIPE>=8 contention. Same flag/value gate => bit-exact.
  // 0 = OFF (tight spin, byte-identical path).
  int spinBackoff = 0;
  // Blocking landing wait (see HierSpinBlock). 1 => the reassembly /
  // completion-reader landing spins never abandon on the bounded fallback; they poll
  // until the flag is actually set (no under-landed read/publish), removing the crown
  // E2E non-determinism. 0 = OFF (byte-identical bounded path).
  int spinBlock = 0;
  // GENERATION-TOKEN chunkReadyFlags (see HierFlagToken). 0 => legacy: the deep-pipe
  // landing flags publish the fixed value 1, are waited with `< 1`, and must be
  // host-zeroed (flags.zero_) before every op (a per-op hipMemset launch on the fixed
  // per-op HIP-launch floor). When opGen>0 the same pattern the classic ring and the
  // reassembly-completion flags (gFlagVal) already use is applied here: publish
  // chunkReadyFlags[p] = opGen (strictly-increasing per op), wait `< opGen`, and skip
  // the host reset -- the higher token supersedes stale slots (also removes the
  // two-distinct-DEEP_PIPE-size carryover hazard without the per-layout re-alloc).
  // Bit-exact by construction. Default 0 => byte-identical path.
  uint64_t opGen = 0;
  // Intra reassembly deep-SQ (see HierReasmDeepSqOn).
  // 1 => a reassembly worker submits all its owned channels' copy descriptors
  // back-to-back (each after its landing flag) keeping the SDMA SQ continuously
  // fed, then a SINGLE drain covers them all before firing the deferred output
  // flags -- instead of submit+drain per channel. 0 = OFF (byte-identical
  // path). Bit-exact by construction (flag never precedes its bytes).
  int reasmDeepSq = 0;
  // Read-coalescing tile bytes for the remote-reassembly fan-out (see
  // HierReasmL2Tile). >0 => tile each per-peer copy into this many bytes and drive all
  // groupSize peer copies through the same tile window (L2-resident) so the redundant
  // reads hit L2 not HBM. 0 = OFF (byte-identical path).
  size_t reasmL2Tile = 0;
  // Intra reassembly per-peer queue split (see HierReasmQSplitOn). 1 => each
  // reassembly worker splits its per-peer column across its disjoint per-peer
  // queue class {k%effReasm==qId%effReasm}, engaging idle per-peer SDMA queues
  // when nq>effReasm. 0 = OFF (byte-identical path, single queue/peer).
  int reasmQSplit = 0;
  // Batched sender-quiet (see HierFuseSenderFenceOn). 1 => push-only body defers
  // fence+flag to a post-drain completion phase so the G peer copies stay in
  // flight without a per-peer fence stall. 0 = OFF (byte-identical path).
  int fuseSenderFence = 0;
  // Descriptor pipelining (see HierPutNdesc). >1 => each per-peer push places this
  // many back-to-back SDMA copy sub-descriptors on one queue + one trailing atomic.
  // 1 = OFF (byte-identical path, single descriptor/peer).
  int putNdesc = 1;
  // Pole SQ-depth (see HierPoleSqDepth). K>1 => the crown local-gather pole
  // issues K independent back-to-back SdmaPutThread copies per peer (K doorbells / K
  // in-flight work items on one queue) to keep the copy engine full across whole-copy
  // completions (the /opt/rccl NCCL_STEPS credit-FIFO mechanism). 1 = OFF (byte-identical).
  int poleSqDepth = 1;
  // Pull local gather (see HierPolePull). 1 => the crown local-gather pole runs
  // the PULL variant (own copy engines read the 7 peer shards over XGMI into local
  // output, self-drain completion; only a cheap staged flag crosses PEs). Bit-exact.
  // Forwarded into OneShotAllGatherSdmaSubGroupKernel_body. 0 = OFF (byte-identical crown).
  int polePull = 0;
  // COPY_LINEAR DW2 coherence hint on the crown local-gather pole (see
  // HierPutCacheHint). bit0->dst_ha, bit1->src_ha. 0 = OFF (byte-identical crown).
  int putCacheHint = 0;
  // Push-issue rotation (see HierPushRotateOn). 1 => rotate each rank's warp->peer
  // map by its group position (native ring stagger) to spread the per-cycle XGMI
  // destination-port load. 0 = OFF (byte-identical path, warp w -> peer w).
  int pushRotate = 0;
  // Copy-engine flag delivery (see HierQFlagOn). 1 => push-only body delivers each
  // peer completion flag via a queued SDMA FENCE packet (FIFO after the copy) with
  // NO drain / system-fence / separate AMO. 0 = OFF (byte-identical path).
  int qFlag = 0;
  // Intra-node pipelined ring for the crown local block (see HierCrownRing /
  // MORI_HIER_CROWN_RING). bit0 = fencedFlag. 0 = OFF (byte-identical flat crown).
  int crownRing = 0;
  // Width-W permutation issue schedule (see HierRingWidth). Phase width W
  // for the crown local-block gather's issue stagger; 0 = OFF (byte-identical
  // path, full-mesh concurrent issue), 1 = single-link matching, W>1 = W
  // concurrent links/phase. Forwarded into OneShotAllGatherSdmaSubGroupKernel_body.
  int ringPhased = 0;
  // Batched sender-side completion fence (see HierBatchFence). 1 => the crown
  // local-block gather collapses its G concurrent per-peer __threadfence_system to
  // ONE CTA-wide fence. Bit-exact. Forwarded into OneShotAllGatherSdmaSubGroupKernel_body.
  int batchFence = 0;
  // HIERARCHICAL 2x4 INTRA GATHER (see HierH2x4). 1 => the crown local-block
  // gather runs the 2-phase sub-group broadcast. Bit-exact. Forwarded into
  // OneShotAllGatherSdmaSubGroupKernel_body. 0 = OFF (byte-identical crown).
  int h2x4 = 0;
  // 2x4 STACKED-FLAT-BODY intra wave width (see HierIntra2 /
  // MORI_HIER_INTRA2). W>0 => the crown local-block flat push issues in ceil(G/W)
  // drained W-regular-matching waves (per-PE egress & per-receiver incast G-1 -> W).
  // W==4 at G==8 = the mandated 2x4. Forwarded into the crown body. 0 = OFF.
  int intra2 = 0;
  // Multi-engine per-link reassembly put (see HierReasmMultiQueueOn / MORI_INTRA_MQ).
  // 1 => the push-only reassembly worker splits each peer's column across all
  // sdmaNumQueue queues via SdmaPutWarp (both recommended XGMI engines/link) instead
  // of a single-lane single-queue put. This is the INTRA_MQ lever ported to the
  // w16 hot path (push-only body). 0 = OFF (byte-identical path).
  int reasmMultiQueue = 0;
  // Phased-permutation reassembly push (see HierPushPhasedOn / MORI_HIER_PUSH_PHASED).
  // 1 => the push-only reassembly issues its groupSize peer copies in groupSize
  // rotated phases (rank g -> peer (g+1+p)%G in phase p) with a CTA barrier between
  // phases, so across ranks each phase is a perfect matching (no receiver-side XGMI
  // incast) -- the native permutation-step schedule on the dominant intra leg. 0 = OFF
  // (byte-identical path, all peers pushed concurrently).
  int pushPhased = 0;
  // Full-width deep-SQ in-flight FIFO (see HierFifoFullWidthOn / MORI_HIER_FIFO).
  // 1 => on the temporal deep-pipe path (deepPipe>1) every sub-chunk uses the FULL
  // sw-QP fan-out and the P sub-chunks are issued back-to-back so each QP carries P
  // in-flight WQEs (deep SQ = NIC fill), then a single parallel per-QP drain + P
  // flag AMOs (native the STEPS pipeline window: depth decoupled from width). 0 = OFF
  // (byte-identical).
  int fifoFullWidth = 0;
  // Progressive deep-pipe publish (see HierFifoProgOn / MORI_HIER_FIFO_PROG). 1 =>
  // on the temporal deep-pipe path (deepPipe>1) issue each sub-chunk p at full sw-QP
  // width, drain its own send-CQ, then publish chunkReadyFlags[p] immediately before
  // issuing p+1 (tail-per-step: the reassembly worker reassembles p while p+1 crosses
  // the NIC). 0 = OFF (byte-identical path). Takes precedence over fifoFullWidth.
  int fifoProg = 0;
  // Per-QP fine-grain inter-arrival drain (see HierShardDrainOn / MORI_HIER_SHARD_DRAIN).
  // 1 => on the single-channel deep-pipe path issue the whole chunk at full sw-QP width,
  // then drain + publish each QP's own 16B-aligned shard the instant it lands, and drive
  // the reasm worker at partition==numQp (single engine) so shard s pushes while s+1..
  // still cross the NIC (attacks the first_land latency prefix). 0 = OFF (byte-identical).
  int shardDrain = 0;
  // DIRECT-LAND (see HierDirectLandOn / MORI_HIER_DIRECT_LAND). 1 => the inter-node
  // RDMA WRITE lands each received remote block straight into this GPU's own final
  // output self-slot (gDstMemObj at (m*groupSize+groupPos)*chunkBytes) instead of the
  // ring buffer, so the reasm worker SKIPS the redundant ring->output self copy (and
  // reads its broadcast source from that output self-slot). Deletes 1 SDMA self-copy
  // + the ring read/footprint for the self column -> cuts the reasm HBM
  // volume that taxes the concurrent inter fill. Bit-exact by construction (same final
  // bytes/layout; the RDMA put-signal AMO still fires the landing flag RC-after data,
  // and the reasm still sets the self flag slot after the landing fence). 0 = OFF
  // (byte-identical crown: NIC writes ring, reasm copies self column).
  int directLand = 0;
  // DIRECT-LAND self-column SKIP toggle (see HierDirectLandSkipSelf). Default
  // 0 => the reasm KEEPS the self-column copy (identity copy of the output self-slot)
  // so that NIC-landed slot is re-touched through the copy-engine+fence coherence
  // path the peer columns get (fixes the stale-line mismatch). 1 => skip it
  // (the incoherent variant; opt-in). Only consulted when directLand!=0.
  int directLandSkipSelf = 0;
  // Pipelined relay ring reassembly (see HierReasmRing / MORI_HIER_REASM_RING).
  // 1 => the reasm broadcasts each remote block's shards with a G-1 step relay ring
  // (perfect-matching per step, no incast) instead of the flat G-way scatter. Same
  // final bytes/layout + same per-shard completion flags => bit-exact. 0 = OFF.
  int reasmRing = 0;
  // Every-GPU-direct write-push fan-out (see HierEveryDirectWriteOn /
  // MORI_HIER_EVERY_DIRECT_WRITE). 1 => the ring CTA skips the inter ring send +
  // inline reasm and instead PUSHES this rank's ring self-slot to all G receivers on
  // every remote node, straight into each receiver's output self-slot with a fused
  // AMO_SET of the completion flag; the local reasm workers become no-ops (the remote
  // half is written by the remote nodes' pushes) and the completion reader waits on
  // the AMO_SET flags. Deletes the 8-way XGMI reasm scatter. 0 = OFF (crown).
  int everyDirectWrite = 0;
  // Device double-buffered ring (MORI_HIER_GEN_RING_DBL). Second symmetric
  // ring buffer + per-op parity counter (see CclInterNodeRingArgs). When
  // parityCounter!=null the crown selects ring<->ringMemObjB by (parityCounter[0]
  // & 1) at kernel entry (args is by-value, so the select propagates to the
  // in-kernel copy-IN, the ring send/recv, and the reassembly read with no body
  // changes). null => single-buffer (byte-identical crown).
  application::SymmMemObjPtr ringMemObjB;
  uint64_t* parityCounter = nullptr;
  // Device flag-token (see HierFlagTokenDevOn). 1 => the crown overrides
  // opGen with the device parity counter value (parityCounter[0]) at kernel entry,
  // so the chunkReadyFlags landing flags become a graph-safe per-op generation
  // (publisher stores parity[0], reasm waits `< parity[0]`, NO host reset). Requires
  // parityCounter!=null (GEN_RING_DBL). 0 => OFF (byte-identical: opGen as passed).
  int flagGenDev = 0;
  // FUSE-ENTRY-BARRIER: carried from the ring args. 1 => the crown runs the
  // cross-PE entry rendezvous in its prologue (block 0 -> ShmemBarrierAllBlock)
  // behind the gridArrival grid-barrier, replacing the separate host barrier launch.
  int fuseEntryBarrier = 0;
  unsigned int* gridArrival = nullptr;
};

// Builder: merge an already-built ring args (CclInterNodeRingArgs) and gather
// args (CclAllgatherSubGroupArgs<uint32_t>, primed for the LOCAL block) plus the
// pipeline extras into one CclFusedRingRemoteGatherArgs. Mirrors
// BuildFusedRingLocalGatherArgs so the existing prepare_* paths stay byte-
// identical; this is pure additive glue. Inert until the Python launcher is wired.
inline int64_t BuildFusedRingRemoteGatherArgs(int64_t ringArgsPtr, int64_t gatherArgsPtr,
                                              int ringBlocks, int64_t chunkReadyFlagsPtr,
                                              int numNodes, int nodeId, int reassemblyBlocks = 0,
                                              int64_t opGen = 0, int reasmDeepSq = 0) {
  static CclFusedRingRemoteGatherArgs fused;
  const CclInterNodeRingArgs* r = reinterpret_cast<const CclInterNodeRingArgs*>(ringArgsPtr);
  const CclAllgatherSubGroupArgs<uint32_t>* g =
      reinterpret_cast<const CclAllgatherSubGroupArgs<uint32_t>*>(gatherArgsPtr);

  // carry the double-buffered ring's second buffer + parity counter (empty/
  // null unless MORI_HIER_GEN_RING_DBL on) so the crown selects the active half.
  fused.ringMemObjB = r->ringMemObjB;
  fused.parityCounter = r->parityCounter;
  // FUSE-ENTRY-BARRIER: carry the in-kernel-rendezvous flag + grid scratch.
  fused.fuseEntryBarrier = r->fuseEntryBarrier;
  fused.gridArrival = r->gridArrival;
  fused.ringPos = r->ringPos;
  fused.ringSize = r->ringSize;
  fused.ringPeBase = r->peBase;
  fused.ringPeStride = r->peStride;
  fused.ringMemObj = r->memObj;
  fused.ringFlagsObj = r->flagsObj;
  fused.chunkBytes = r->chunkBytes;
  fused.numQp = r->numQp;
  fused.ringBlocks = ringBlocks < 1 ? 1 : ringBlocks;
  // Fused FSDP path: put-signal only when explicitly env-enabled (the standalone
  // default-on must not leak into the E2E deferred/overlap bytes, keeping the loss
  // byte-identical to native). See HierRingPutSignalExplicitlyOn.
  fused.usePutSignal = HierRingPutSignalExplicitlyOn() ? r->usePutSignal : 0;
  fused.useWriteImm = r->useWriteImm;
  fused.useRead = r->useRead;  // plumb MORI_HIER_RING_READ into the crown/deep-pipe launch
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
  // In-kernel single-CTA re-touch is thread-starved on the big AG (512 threads
  // over the whole output serializes -> step timeout). The re-touch is instead
  // done by a FULL-GRID L2CoherentRetouchKernel epilogue launched from Python
  // after the fused kernel's completion fence (stream-ordered, all bytes landed).
  fused.retouchOut = 0;
  // elasticReasm resolved AFTER deepPipe (see AUTO-ELASTIC block below) so the
  // auto default can key on the temporal-pipe single-worker tail.
  fused.wqeDepth = HierWqeDepth();
  fused.fifoFullWidth = HierFifoFullWidthOn() ? 1 : 0;
  fused.fifoProg = HierFifoProgOn() ? 1 : 0;
  fused.shardDrain = HierShardDrainOn() ? 1 : 0;
  fused.directLand = HierDirectLandOn() ? 1 : 0;
  fused.directLandSkipSelf = HierDirectLandSkipSelf();
  fused.everyDirectWrite = HierEveryDirectWriteOn() ? 1 : 0;
  // Host-side rkey diagnostic (device printf does not surface in this JIT build).
  // Compare the direct-land dest (g->dstMemObj = the intra transit/output) against
  // the RDMA-registered inter ring buffer (r->memObj, known-good cross-node rkeys).
  // If dstMemObj->peerRkeys is null while ringMemObj->peerRkeys is not, the
  // cross-node RDMA WRITE to the output drops. Prints once.
  if (fused.directLand) {
    static bool dlDumped = false;
    if (!dlDumped) {
      dlDumped = true;
      const application::SymmMemObj* dst = g->dstMemObj.cpu;
      const application::SymmMemObj* ring = r->memObj.cpu;
      printf(
          "[DLHOST] directLand=1 dst(%p): peerRkeys=%p peerPtrs=%p lkey=%u size=%zu | "
          "ring(%p): peerRkeys=%p peerPtrs=%p lkey=%u size=%zu\n",
          (void*)dst, (dst ? (void*)dst->peerRkeys : nullptr),
          (dst ? (void*)dst->peerPtrs : nullptr), (dst ? dst->lkey : 0u),
          (dst ? dst->size : (size_t)0), (void*)ring, (ring ? (void*)ring->peerRkeys : nullptr),
          (ring ? (void*)ring->peerPtrs : nullptr), (ring ? ring->lkey : 0u),
          (ring ? ring->size : (size_t)0));
      fflush(stdout);
    }
  }
  // FIRST-LAND idle-engine reclamation (see HierLocalOffload). Clamp K to [0, G-1]
  // so the queue-0 local CTA always retains at least one peer column (never offloads
  // the whole gather). Only meaningful on the fused remote crown (numNodes>1).
  {
    int k = HierLocalOffload();
    int gmax = fused.groupSize > 1 ? fused.groupSize - 1 : 0;
    fused.offloadPeers = (k > gmax) ? gmax : k;
  }
  // Deep-SQ temporal pipeline only engages on the single-channel ring (rb==1, the
  // giant-AG fan-out path); RING_BLOCKS>1 (multiBlock) keeps its spatial split.
  // Size gate: the per-sub-chunk device landing flag is only bit-exact below a byte
  // threshold (very large sub-chunks crash, >=64MB race) -- so engage the pipeline
  // only when this PE's chunk <= MORI_HIER_DEEP_PIPE_MAX_MB; every larger chunk
  // falls through to the whole-chunk crown fence. 0 => no gate.
  {
    // dpWindow is the per-sub-chunk coherence window (chunkBytes/dp), not the total
    // chunk. The device per-sub-chunk landing flag is a pipeline hint (E2E landing
    // is anchored by the crown DEFER_HOSTSYNC host fence); the HW hazards are
    // (a) a crash on very large sub-chunks and (b) the >=32MB sub-chunk
    // send-CQE-before-SDMA-coherent race. Gating on the sub-chunk keeps the pipeline
    // engaged on the steady-state decoder AGs (sub-chunks under 32MB) while the giant
    // AG (large sub-chunk) falls through to the whole-chunk crown fence. Strict '<'
    // so a 32MB sub-chunk (e.g. 64MB@dp=2) is caged, not engaged. Explicit
    // MORI_HIER_DEEP_PIPE_MAX_MB overrides the window. dp<=1 => byte-identical path.
    size_t dpWindow = HierDeepPipeMaxBytes();
    int dp = (fused.ringBlocks == 1) ? HierDeepPipe() : 1;
    if (dp < 0) {  // "auto": depth = round(perPE chunkBytes / subBytes), clamp[1,16]
      const size_t sub = HierDeepPipeSubBytes();
      size_t d = (fused.chunkBytes + sub / 2) / sub;
      dp = (d < 1) ? 1 : (d > 16 ? 16 : static_cast<int>(d));
    }
    // MID-BUFFER floor: raise the resolved depth to MIN_DEPTH so w16 mid buffers
    // (whose auto depth rounds to 1) actually pipeline. Cap so each sub-chunk stays
    // >= 16B (aligned split) and <= the 16 clamp. Only meaningful on the fused
    // rb==1 path (dp already forced to 1 when ringBlocks>1). Default 1 => no-op.
    {
      int minDepth = HierDeepPipeMinDepth();
      if (minDepth > 1 && fused.ringBlocks == 1 && fused.chunkBytes >= 32) {
        size_t splitCap = fused.chunkBytes / 16;
        int cap = (splitCap > 16) ? 16 : static_cast<int>(splitCap);
        if (cap < 1) cap = 1;
        int tgt = (minDepth < cap) ? minDepth : cap;
        if (tgt > dp) dp = tgt;
      }
    }
    // Size-gated big-chunk depth bump (MORI_HIER_DP_BIG, see getter). Raise
    // the resolved depth to D only for large per-PE chunks (>= HierDpBigBytes),
    // where the deeper pipeline's tail is hidden under the longer inter-fill span
    // (small chunks regress, so they are gated out). Same clamp as the MIN_DEPTH
    // floor (each sub-chunk >= 16B). Only on the fused rb==1 path. dp<=1 (deep-pipe
    // off) is left untouched. Default DP_BIG=0 => no-op (byte-identical crown).
    {
      int bigDepth = HierDpBigDepth();
      if (bigDepth > 1 && dp >= 1 && fused.ringBlocks == 1 &&
          fused.chunkBytes >= HierDpBigBytes() && fused.chunkBytes >= 32) {
        size_t splitCap = fused.chunkBytes / 16;
        int cap = (splitCap > 16) ? 16 : static_cast<int>(splitCap);
        if (cap < 1) cap = 1;
        int tgt = (bigDepth < cap) ? bigDepth : cap;
        if (tgt > dp) dp = tgt;
      }
    }
    // Ragged-sub-chunk guard (MORI_HIER_DP_CLEAN). Snap dp down to the largest
    // divisor of chunkBytes <= dp so every sub-chunk is equal (no ragged remainder
    // tail, which collapses throughput). Down-only => dp' <= dp => sub-chunk count
    // never exceeds the Python-sized flag budget (safe). No-op when dp already
    // divides chunkBytes (all power-of-2 sizes), so only ragged (non-2^n) sizes
    // change.
    if (HierDpCleanDepthOn() && dp > 1 && fused.chunkBytes > 0) {
      while (dp > 1 && (fused.chunkBytes % static_cast<size_t>(dp)) != 0) --dp;
    }
    // No auto default for the size-gate window: dpWindow stays 0 unless
    // MORI_HIER_DEEP_PIPE_MAX_MB is set explicitly, so the default is the
    // conservative byte-identical path. A non-zero window is an explicit opt-in for
    // fuse_remote experiments only.
    size_t subChunk = (dp > 1) ? (fused.chunkBytes / static_cast<size_t>(dp)) : fused.chunkBytes;
    // LOWER floor: gate on the per-PE total chunk so a [MIN,MAX] window can pin
    // deep-pipe to the sizes where it wins and drop the sizes that hang or regress
    // onto the plain path. 0 => no floor.
    size_t dpFloor = HierDeepPipeMinBytes();
    bool floorOk = (dpFloor == 0 || fused.chunkBytes >= dpFloor);
    fused.deepPipe = (floorOk && (dpWindow == 0 || subChunk < dpWindow)) ? dp : 1;
    // Small-chunk single-shot gate (see HierDpSmallBytes). Force the latency-bound
    // small buffers onto the one-flag single-shot crown fence (deepPipe=1) so they do
    // not pay the 2-stage pipeline's extra per-sub-chunk landing handshake that cannot
    // be hidden at small sizes (too little inter-fill span to overlap the intra tail).
    // Bit-exact (single-shot is the crown path). Default 0 => no-op.
    {
      size_t dpSmall = HierDpSmallBytes();
      if (dpSmall > 0 && fused.chunkBytes <= dpSmall) fused.deepPipe = 1;
    }
  }
  // Elastic reassembly resolution. On the temporal deep-pipe tail (deepPipe>1) with a
  // single dedicated reassembly worker, the XGMI drain of all sub-chunks runs on one
  // SDMA queue while queue 0 (the local-block CTA) sits idle after its own-shard
  // gather. The elastic lever would fold that idle local CTA in as a second tail
  // engine (workers 0..reasm handle disjoint sub-chunks f % (reasm+1) on distinct
  // queues, all drained before the completion flag). It is kept strict opt-in (default
  // OFF): the elastic worker was designed for the spatial channel partition
  // (deepPipe<=1) and its flag-slot/queue accounting is not yet correct for the
  // temporal deep-pipe partition, where auto-enabling it can deadlock the completion
  // reader. Default OFF keeps the crown byte-identical. See
  // HierFuseRemoteElasticOn.
  fused.elasticReasm = HierFuseRemoteElasticOn() ? 1 : 0;
  fused.deepPipeImm = HierDeepPipeImmOn() ? 1 : 0;
  // Scale-robust landing fence default. The deep-pipe put-with-signal AMO
  // (deepPipeImm==0) is not scale-robust -- for large sub-chunks the AMO can beat its
  // own data landing, so the reassembly reader can consume un-landed bytes on the giant
  // AG. Since the put-signal path is unreliable at scale (see HierDeepPipeQuietOn),
  // auto-engage the quiet-drain landing fence whenever deep-pipe runs the put-signal
  // path, unless explicitly forced off (MORI_HIER_DEEP_PIPE_QUIET=0). Bit-exact (same
  // drains/AMOs/slots, only independent-completion order relaxed). deepPipe<=1 (the
  // default) never enters this path => byte-identical path. deepPipeImm==1
  // (WRITE_IMM) keeps its own recv-CQE fence, QUIET not added.
  {
    const char* q = std::getenv("MORI_HIER_DEEP_PIPE_QUIET");
    const bool quietForcedOff = (q != nullptr && q[0] == '0' && q[1] == '\0');
    fused.deepPipeQuiet =
        (HierDeepPipeQuietOn() || (fused.deepPipe > 1 && fused.deepPipeImm == 0 && !quietForcedOff))
            ? 1
            : 0;
    // Per-pair coherence ceiling. If the signal path is engaged (deepPipeQuiet==0,
    // deepPipe>1) but the fused per-op chunk is >= the per-pair ceiling, fall back to
    // the bit-exact quiet drain so a tighter-coherence-window pair does not hang or
    // drift. The default ceiling is 0 (no cap), so this is a no-op and the
    // signal path stays byte-identical. A hang-prone pair sets MORI_HIER_SIGNAL_MAX_MB
    // (e.g. 48) to auto-revert large chunks to quiet.
    if (fused.deepPipeQuiet == 0 && fused.deepPipe > 1 && fused.deepPipeImm == 0) {
      const size_t sigMax = HierSignalMaxBytes();
      if (sigMax > 0 && fused.chunkBytes >= sigMax) {
        fused.deepPipeQuiet = 1;
      }
    }
  }
  fused.dpSerialDrain = HierDpSerialDrainOn() ? 1 : 0;
  fused.dpTailPct = HierDpTailPct();
  fused.hostProxyInter = HierHostProxyReasm();
  fused.fuseCopyIn = HierFuseCopyInOn() ? 1 : 0;
  fused.localPushOnly = HierLocalPushOnly() ? 1 : 0;
  fused.spinBackoff = HierSpinBackoff() ? 1 : 0;
  fused.spinBlock = HierSpinBlock() ? 1 : 0;
  // Generation token for chunkReadyFlags: 0 (legacy fixed-1 + host reset) unless
  // the Python launcher passes a strictly-increasing per-op token (HierFlagToken).
  fused.opGen = static_cast<uint64_t>(opGen);
  // Device flag-token: when on and the double-buffer parity counter exists,
  // the crown derives the per-op flag generation from the graph-safe device parity
  // counter at kernel entry (see HierFlagTokenDevOn / FusedRingRemoteGatherKernel).
  fused.flagGenDev = (HierFlagTokenDevOn() && fused.parityCounter != nullptr) ? 1 : 0;
  // Intra reassembly deep-SQ: 0 unless the Python launcher passes
  // MORI_HIER_REASM_DEEPSQ=1.
  fused.reasmDeepSq = reasmDeepSq;
  // Intra reassembly per-peer queue split : env-gated, worker-disjoint,
  // engages idle per-peer SDMA queues when nq>effReasm. REQUIRES elasticReasm so
  // the worker qId space is exactly [0,effReasm) => the queue classes {k%effReasm}
  // partition [0,nq) with queue-0's class reserved to the qId=0 (local) CTA; WITHOUT
  // elastic the remote qId==reasm worker would alias the local push-only's queue 0
  // (same-queue race). Default OFF.
  fused.reasmQSplit = (HierReasmQSplitOn() && fused.elasticReasm != 0) ? 1 : 0;
  // Batched sender-quiet (completion-model lever): env-gated, single-shot path
  // only, bit-exact by construction. Default OFF => byte-identical path.
  fused.fuseSenderFence = HierFuseSenderFenceOn() ? 1 : 0;
  // Descriptor pipelining (single-queue push scheduling axis). Default 1 => OFF.
  fused.putNdesc = HierPutNdesc();
  // pole SQ-depth (independent-work-item issue depth on the local-gather pole).
  // Default 1 => OFF (byte-identical path).
  fused.poleSqDepth = HierPoleSqDepth();
  // pull local gather (copy DIRECTION flip on the pole). Default 0 => OFF.
  fused.polePull = HierPolePull();
  // COPY_LINEAR DW2 coherence hint on the pole. Default 0 => OFF
  // (byte-identical crown).
  fused.putCacheHint = HierPutCacheHint();
  // Push-issue rotation (native ring-stagger of the warp->peer map). Default 0 => OFF.
  fused.pushRotate = HierPushRotateOn() ? 1 : 0;
  // Phased-permutation reassembly (native permutation-step schedule on the intra leg).
  // Default 0 => OFF (byte-identical path, peers pushed concurrently).
  fused.pushPhased = HierPushPhasedOn() ? 1 : 0;
  // Copy-engine flag delivery (queued FENCE-packet completion). Default 0 => OFF
  // (byte-identical path). Bit-exact by FIFO construction (see HierQFlagOn).
  fused.qFlag = HierQFlagOn() ? 1 : 0;
  // intra-node pipelined ring for the crown local block (MORI_HIER_CROWN_RING).
  // Default 0 => OFF (byte-identical flat crown). See HierCrownRing.
  fused.crownRing = HierCrownRing();
  // width-W permutation issue schedule on the crown local-block gather
  // (MORI_HIER_RING=W). Default 0 => OFF (byte-identical path, full-mesh
  // concurrent issue). Forwarded to the crown gather call site in ccl_kernels.hip.
  fused.ringPhased = HierRingWidth();
  fused.batchFence = HierBatchFence() ? 1 : 0;
  fused.h2x4 = HierH2x4();
  fused.intra2 = HierIntra2();
  // Multi-engine per-link reassembly put (INTRA_MQ ported to the w16 hot path).
  // Default OFF => byte-identical path; bit-exact by SdmaPutWarp construction.
  fused.reasmMultiQueue = HierReasmMultiQueueOn() ? 1 : 0;
  // INTRA_MQ takes precedence over deep-SQ on the reassembly leg. mqActive in
  // OneShotSubGroupPushOnly_body requires deepSqPhase==0, but reasmDeepSq is
  // default-on on the fused crown (fuse_local/fuse_remote), so without this the
  // SdmaPutWarp multi-engine spread would never engage on the all-to-all reasm.
  // Forcing INTRA_MQ ahead of deep-SQ makes the mqActive path reachable, spreading
  // the remote-reassembly copies across both KFD-recommended XGMI SDMA engines/link.
  // Both paths are individually bit-exact (deep-SQ: drain-then-flag; MQ: all nqTop
  // queues drained before the one flag AMO), so the precedence stays bit-exact.
  // Default (INTRA_MQ unset) => reasmMultiQueue==0 => no change to reasmDeepSq =>
  // byte-identical crown.
  if (fused.reasmMultiQueue) fused.reasmDeepSq = 0;
  // Read-coalescing tile (MORI_HIER_REASM_L2TILE). The tiled per-tile-completion
  // fan-out needs the single-shot path (deepSqPhase==0) to engage, but reasmDeepSq is
  // default-on on the fused crown, so when the tile lever is set, force reasmDeepSq
  // off so the reassembly worker runs the tileable single-shot path. Default
  // (L2Tile==0) => no change => byte-identical crown.
  fused.reasmL2Tile = HierReasmL2Tile();
  if (fused.reasmL2Tile > 0) fused.reasmDeepSq = 0;
  // Pipelined relay ring reassembly. Its G-1 step relay owns its own submit +
  // per-step drain/fence/flag, so it runs the plain single-shot completion model
  // (force deepSq off, same as L2Tile) and is mutually exclusive with the other reasm
  // completion levers. Default OFF => byte-identical crown.
  fused.reasmRing = HierReasmRing() ? 1 : 0;
  if (fused.reasmRing) fused.reasmDeepSq = 0;
  // Same-CTA inline reassembly: only on the multiBlock spatial ring (rb>1; deepPipe is
  // forced 1 there, line ~1037), classic single-channel-off landing, and NOT under the
  // host-proxy inter leg (E2E path). When engaged the ring CTA reassembles its own
  // sub-range inline via A's FusedRemoteReassembleWorker, so the dedicated reassembly
  // CTAs are redundant (Python drops them). Byte-identical output by construction.
  fused.inlineReasm =
      (HierInlineReasmOn() && fused.ringBlocks > 1 && fused.hostProxyInter == 0 &&
       fused.chunkReadyFlags != nullptr && fused.elasticReasm == 0 && fused.reassemblyBlocks <= 0)
          ? 1
          : 0;

  return reinterpret_cast<int64_t>(&fused);
}

}  // namespace collective
}  // namespace mori
