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

// DEVICE ONLY. Included by the generated TU, never by a host source.
//
// Two parts, in this order:
//
//   1. the transport shim -- one EpInterNode* function per shmem call the port
//      replaced, on ccoGda. New code; nothing in the original corresponds to it.
//   2. EP internode v1 dispatch + combine, ported from
//      src/ops/dispatch_combine/internode_v1.cpp.
//
// Part 2 is mechanical: the algorithm, the block specialisation, the chunk-flag
// protocol and the proxy-PE topology are unchanged from v1; only the transport
// calls and the threaded-down `const ccoDevComm& comm` differ.
//
// mori/shmem is deliberately not included. It reaches its endpoints through a
// device global that the host fills on every hipModuleLoad, and a JIT module
// would have to be taught that handshake; cco takes the communicator as a
// kernel argument, which is the whole reason this port exists.

#pragma once

#include "mori/core/core.hpp"
// The cfg half only. ep_internode_spec.hpp is host-only -- it pulls in the
// Compiler, which this TU is the OUTPUT of and must not depend on.
#include "mori/ops/dispatch_combine_v2/ep_internode_cfg.hpp"
// The argument surface, the config the bodies read, the flat-index helpers and
// MultiWarpIter.
#include "mori/ops/dispatch_combine_v2/ep_internode_args.hpp"

// mori/core/utils/utils.hpp defines `warpSize` as an object-like macro, while
// cco.hpp declares mori::cco::impl::warpSize(). Whichever lands first wins, so
// including cco from EP code is otherwise include-order dependent. Hide the
// macro across the cco headers -- cco never wants it, it uses its own
// wavefront-size builtin -- and restore it for the kernel, which does.
#pragma push_macro("warpSize")
#undef warpSize
#include "mori/cco/cco.hpp"
#include "mori/cco/cco_scale_out.hpp"
#pragma pop_macro("warpSize")

namespace mori {
namespace ops {
namespace v2 {

// v2 types introduced under the names the ported bodies already use.
using index_t = ep_index_t;

// v1's spelling, now a plain alias: the struct stopped depending on T when
// inpTokenBuf became void*, matching the intranode EpArgs.
using EpDispatchCombineArgs = EpInterNodeArgs;

// v1's common.hpp macro, with the config type swapped and the runtime assert
// promoted to a static_assert: numExpertPerToken must fit in a ballot.
#define DEF_COMMON_VARS                                                                          \
  constexpr EpInterNodeDeviceCfg config = EpInterNodeDeviceCfgOf(kConfig);                       \
  int thdId = threadIdx.x;                                                                       \
  int thdNum = blockDim.x;                                                                       \
  int laneId = threadIdx.x & (warpSize - 1);                                                     \
  int warpId = thdId / warpSize;                                                                 \
  int warpNum = blockDim.x / warpSize;                                                           \
  int blockNum = gridDim.x;                                                                      \
  int blockId = blockIdx.x;                                                                      \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;                                       \
  int globalThdNum = gridDim.x * blockDim.x;                                                     \
  int globalWarpId = blockIdx.x * warpNum + warpId;                                              \
  int globalWarpNum = gridDim.x * warpNum;                                                       \
  int nullTokenId = NullFlatTokenIndex(config);                                                  \
  int myPe = args.rank;                                                                          \
  int npes = config.worldSize;                                                                   \
  int myNode = myPe / config.gpuPerNode;                                                         \
  int nNodes = npes / config.gpuPerNode;                                                         \
  int numExpertPerToken = config.numExpertPerToken;                                              \
  static_assert(kConfig.numExpertPerToken < warpSize, "numExpertPerToken must fit in a ballot"); \
  size_t hiddenDim = config.HiddenDimSz();                                                       \
  size_t hiddenBytes = config.HiddenBytes(sizeof(T));                                            \
  size_t indexBytes = config.IndexBytes();                                                       \
  size_t weightBytes = config.WeightBytes();                                                     \
  size_t srcTokenIdBytes = config.SrcTokenIdBytes();                                             \
  size_t scaleBytes = config.ScaleBytes();                                                       \
  size_t xferBytes = config.XferBytesPerToken(sizeof(T));                                        \
  size_t combXferBytes = (args.weightsBuf == nullptr) ? hiddenBytes : hiddenBytes + weightBytes;

/* ---------------------------------------------------------------------------------------------- */
/*                                    Transport shim (cco-GDA)                                    */
/* ---------------------------------------------------------------------------------------------- */
// The five cross-node operations the kernel below needs. Argument order is kept
// identical to the shmem calls they replaced, so the call sites read the same.
//
// Only cross-node traffic is here. Intra-node peer access is untouched: it was
// always a direct store through the flat LSA VA, and the arena window covers
// those from ccoGetPeerPtr on the cco backend, so GetAs<T*>(pe) keeps working.
//
// Scope, matching what the replaced shmem calls actually did:
//
//   EpInterNodePutSignal / EpInterNodePut / EpInterNodeAtomicAdd   thread scope: every active lane
//       posts its own data WQE, and the flag atomic comes once from the group
//       leader. Callers are divergent (they sit under `if (laneId == 0)` and
//       inside the deduplication ballot), so these must not contain a group
//       barrier.
//   EpInterNodeQuiet                                 warp-collective. Its call sites
//       are warp-uniform, and ccoGda::flush requires at least warp scope because
//       it takes a warp-level CQ poll lock.
//
// EpInterNodePut replaces ShmemPutTypeNbiWarp, whose RDMA implementation is
// `if (laneId == 0) <thread put>` -- the warp never split the transfer -- so it
// keeps that shape rather than promising a group barrier it could not honour
// under divergence.

// The NIC provider is fixed at compile time. The JIT toolchain passes the
// matching -DMORI_DEVICE_NIC_*; without it cco_scale_out.hpp falls through to
// its mlx5 #else and a bnxt host silently builds mlx5 WQEs.
inline constexpr ::mori::core::ProviderType kEpInterNodeProvider = CCO_GDA_BUILD_PROVIDER;

// A buffer's position in the arena window. Every EP buffer is a sub-region of
// one registered window (EpDispatchCombineHandle::MallocSymm), and RDMA names a
// remote buffer as (window, byte offset) with iova=0 -- there is no peer VA to
// hand the NIC, which is why a region is (window, offset).
__device__ __forceinline__ ::mori::cco::ccoWindow_t EpInterNodeWindow(
    const EpInterNodeRegion region) {
  return reinterpret_cast<::mori::cco::ccoWindow_t>(region->win);
}

__device__ __forceinline__ size_t EpInterNodeOffset(const EpInterNodeRegion region,
                                                    size_t byteOffset) {
  return region->off + byteOffset;
}

// ── the remote actions ───────────────────────────────────────────────────────
//
// v1's chunk-flag protocol needs a remote atomic add at an *arbitrary* window
// offset: one slot per (node, chunk) in EP's own arena, count scaling with the
// token capacity, polled and cleared by the receiver. ccoGda_SignalInc /
// ccoGda_SignalAdd cannot name that target -- they resolve to
// `signalId * sizeof(uint64_t)` in the DevComm's own resourceWindow -- so EP
// passes ccoGda_WindowSignalAdd, which carries (window, offset, value) instead
// of a slot id. Same NIC atomic, same single reservation; only the raddr/rkey
// lookup differs.
//
// The tradeoff that comes with naming your own window: waitSignal / readSignal /
// resetSignal do not see these, because they are written against the resource
// window's signal pool and its consume-on-read shadow. EP polls and clears the
// flags itself, which is what its protocol did anyway.

__device__ __forceinline__ ::mori::cco::ccoGda_WindowSignalAdd EpInterNodeFlagAdd(
    const EpInterNodeRegion flag, size_t flagOffset, uint64_t value) {
  return ::mori::cco::ccoGda_WindowSignalAdd{EpInterNodeWindow(flag),
                                             EpInterNodeOffset(flag, flagOffset), value};
}

__device__ __forceinline__ void EpInterNodeAtomicAdd(const ::mori::cco::ccoDevComm& comm,
                                                     const EpInterNodeRegion dst, size_t dstOffset,
                                                     uint64_t value, int pe, int qpId = 0) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template signal<::mori::cco::CCO_TEAM_WORLD>(pe, EpInterNodeFlagAdd(dst, dstOffset, value),
                                                   ::mori::cco::ccoCoopThread{});
}

// RDMA write with the flag atomic fused into the same reservation. The ordering
// the protocol wants -- a receiver that sees the counter move has the payload
// too -- comes from the QP: an RC responder executes requests in PSN order, and
// putImpl places the signal WQE at base + numActiveLanes, after every lane's
// write, with the slot index doubling as the PSN (BNXT reserves its signal PSN
// behind the data PSNs for the same reason).
//
// This is the signalled put shmem had, and it stays ONE reservation: issuing a
// bare put and posting the atomic separately would cost an extra doorbell and
// CQE per chunk, on the critical path of a latency-bound kernel.
//
// ccoGdaThreadAggregate, not ThreadIndependent: the latter runs a group-by-peer
// ballot before posting, which every call site here would resolve in a single
// iteration anyway -- all three are warp-uniform in pe and qpId (proxyPe follows
// the warp's node, and startTokenIdx is a multiple of warpSize so
// tokenId / warpSize is too). Aggregate skips the ballot and posts directly,
// which is the shape shmem had. The warp aggregation itself is unaffected: it
// lives in putImpl, below the ThreadMode branch, so one atomic per warp group
// from the leader lane either way -- the shape the deduplication call site
// depends on, since its active lanes carry the same flag slot and the same
// value.
__device__ __forceinline__ void EpInterNodePutSignal(const ::mori::cco::ccoDevComm& comm,
                                                     const EpInterNodeRegion dst, size_t dstOffset,
                                                     const EpInterNodeRegion src, size_t srcOffset,
                                                     size_t bytes, const EpInterNodeRegion signal,
                                                     size_t signalOffset, uint64_t signalValue,
                                                     int pe, int qpId) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template put<::mori::cco::CCO_TEAM_WORLD, ::mori::cco::ccoGdaThreadAggregate>(
      pe, EpInterNodeWindow(dst), EpInterNodeOffset(dst, dstOffset), EpInterNodeWindow(src),
      EpInterNodeOffset(src, srcOffset), bytes,
      EpInterNodeFlagAdd(signal, signalOffset, signalValue), ::mori::cco::ccoCoopThread{});
}

__device__ __forceinline__ void EpInterNodePut(const ::mori::cco::ccoDevComm& comm,
                                               const EpInterNodeRegion dst, size_t dstOffset,
                                               const EpInterNodeRegion src, size_t srcOffset,
                                               size_t bytes, int pe, int qpId) {
  // Nothing to send; mlx5 also reads a zero byte_count as 2^31 and errors the QP.
  if (bytes == 0) return;
  if ((threadIdx.x & (warpSize - 1)) != 0) return;
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template put<::mori::cco::CCO_TEAM_WORLD, ::mori::cco::ccoGdaThreadIndependent>(
      pe, EpInterNodeWindow(dst), EpInterNodeOffset(dst, dstOffset), EpInterNodeWindow(src),
      EpInterNodeOffset(src, srcOffset), bytes, ::mori::cco::ccoGda_NoSignal{},
      ::mori::cco::ccoCoopThread{});
}

// Atomic-free counterparts (kConfig.noRdmaAtomics), for NICs whose RDMA atomics cannot target GPU
// memory. A flag WRITE replaces the add because every flag slot has a single writer per round
// and its owner zeroes it before the next round, so the add always lands on zero. The flag
// goes on the same QP after the data, so RC's in-order execution still publishes it last.
__device__ __forceinline__ void EpInterNodeWriteValue(const ::mori::cco::ccoDevComm& comm,
                                                      const EpInterNodeRegion dst, size_t dstOffset,
                                                      uint64_t value, int pe, int qpId = 0) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template putValue<::mori::cco::CCO_TEAM_WORLD>(pe, EpInterNodeWindow(dst),
                                                     EpInterNodeOffset(dst, dstOffset), value);
}

// Same reservation shape as EpInterNodePutSignal for the data; the group leader then posts the
// flag in a second reservation, so it lands behind every lane's write on this QP.
__device__ __forceinline__ void EpInterNodePutThenWrite(
    const ::mori::cco::ccoDevComm& comm, const EpInterNodeRegion dst, size_t dstOffset,
    const EpInterNodeRegion src, size_t srcOffset, size_t bytes, const EpInterNodeRegion signal,
    size_t signalOffset, uint64_t signalValue, int pe, int qpId) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template put<::mori::cco::CCO_TEAM_WORLD, ::mori::cco::ccoGdaThreadAggregate>(
      pe, EpInterNodeWindow(dst), EpInterNodeOffset(dst, dstOffset), EpInterNodeWindow(src),
      EpInterNodeOffset(src, srcOffset), bytes, ::mori::cco::ccoGda_NoSignal{},
      ::mori::cco::ccoCoopThread{});
  const uint64_t active = __ballot(1);
  if (static_cast<int>(threadIdx.x & (warpSize - 1)) ==
      __ffsll(static_cast<unsigned long long>(active)) - 1) {
    EpInterNodeWriteValue(comm, signal, signalOffset, signalValue, pe, qpId);
  }
}

template <bool NoRdmaAtomics>
__device__ __forceinline__ void EpInterNodeChunkPut(const ::mori::cco::ccoDevComm& comm,
                                                    const EpInterNodeRegion dst, size_t dstOffset,
                                                    const EpInterNodeRegion src, size_t srcOffset,
                                                    size_t bytes, const EpInterNodeRegion signal,
                                                    size_t signalOffset, uint64_t signalValue,
                                                    int pe, int qpId) {
  if constexpr (NoRdmaAtomics) {
    EpInterNodePutThenWrite(comm, dst, dstOffset, src, srcOffset, bytes, signal, signalOffset,
                            signalValue, pe, qpId);
  } else {
    EpInterNodePutSignal(comm, dst, dstOffset, src, srcOffset, bytes, signal, signalOffset,
                         signalValue, pe, qpId);
  }
}

template <bool NoRdmaAtomics>
__device__ __forceinline__ void EpInterNodeCountSignal(const ::mori::cco::ccoDevComm& comm,
                                                       const EpInterNodeRegion dst,
                                                       size_t dstOffset, uint64_t value, int pe) {
  if constexpr (NoRdmaAtomics) {
    EpInterNodeWriteValue(comm, dst, dstOffset, value, pe);
  } else {
    EpInterNodeAtomicAdd(comm, dst, dstOffset, value, pe);
  }
}

// Atomic-free v2_ll dispatch sends a chunk of at least 2 * kDispatchPieceMinTokens tokens as one
// piece per QP. The piece count comes from numQpToDrain, the op's largest active count, so sender
// and receiver agree on it whatever counts their buckets picked. The chunk flag follows piece 0
// and still carries the chunk's token count; each later piece writes its own flag with the
// dispatch's number + 1, after the chunk flags, so it never needs clearing. Those flags are
// WRITEs, which only the atomic-free path may use (it refuses relaxed ordering); with atomics a
// chunk stays one piece behind its fused atomic.
constexpr int kMaxDispatchPieces = 4;
constexpr int kDispatchPieceMinTokens = 32;

template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ int EpInterNodeDispatchPieceTokens(int chunkTokens) {
  constexpr int kPieces = !kConfig.noRdmaAtomics || kConfig.numQpToDrain < 1 ? 1
                          : kConfig.numQpToDrain > kMaxDispatchPieces        ? kMaxDispatchPieces
                                                                             : kConfig.numQpToDrain;
  int pieces = chunkTokens / kDispatchPieceMinTokens;
  pieces = pieces < 1 ? 1 : (pieces > kPieces ? kPieces : pieces);
  return (chunkTokens + pieces - 1) / pieces;
}

__device__ __forceinline__ size_t EpInterNodePieceFlagSlot(int nNodes, int maxChunkNum, int node,
                                                           int chunk, int piece) {
  return static_cast<size_t>(nNodes) * maxChunkNum +
         (static_cast<size_t>(node) * maxChunkNum + chunk) * (kMaxDispatchPieces - 1) + (piece - 1);
}

// A received token past its chunk's piece 0 waits for its own piece's flag.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void EpInterNodeWaitPiece(uint64_t* chunkFlag, int nNodes,
                                                     int maxChunkNum, int node, int chunk,
                                                     int tokenInChunk, uint64_t chunkTokens,
                                                     uint64_t* dispatchNumber) {
  if (tokenInChunk < kDispatchPieceMinTokens) return;  // always in piece 0
  const int piece =
      tokenInChunk / EpInterNodeDispatchPieceTokens<kConfig>(static_cast<int>(chunkTokens));
  if (piece == 0) return;
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    const uint64_t pieceTag = core::AtomicLoadRelaxed(dispatchNumber) + 1;
    uint64_t* flag = chunkFlag + EpInterNodePieceFlagSlot(nNodes, maxChunkNum, node, chunk, piece);
    while (core::AtomicLoadRelaxedSystem(flag) != pieceTag) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
}

// Atomic-free combine barrier. Writes cannot accumulate, so each (sender, qp) gets its own slot,
// after the worldSize per-rank slots the intra-node barrier and the atomic path use. Every rank
// marks the op's largest active count (numQpToDrain), so peers whose buckets picked different
// active counts still agree on which slots to wait for.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ constexpr int EpInterNodeBarrierQps() {
  return kConfig.numQpToDrain > kConfig.numQpPerPe ? kConfig.numQpToDrain : kConfig.numQpPerPe;
}

template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ constexpr size_t EpInterNodeBarrierQpSlot(int senderPe, int qpId) {
  return kConfig.worldSize + static_cast<size_t>(senderPe) * EpInterNodeBarrierQps<kConfig>() +
         qpId;
}

// Drains one stripe: flush(peer) only polls the QP belonging to its own
// context, so each QP a peer was sent on has to be flushed separately.
__device__ __forceinline__ void EpInterNodeQuiet(const ::mori::cco::ccoDevComm& comm, int pe,
                                                 int qpId) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template flush<::mori::cco::CCO_TEAM_WORLD>(pe, ::mori::cco::ccoCoopWarp{});
}

// The value a combine adds on QP qpId; see kCombineBarrierMarkerTotal.
__device__ __forceinline__ constexpr uint64_t CombineBarrierMarker(int qpId, int activeQps) {
  return qpId == 0 ? uint64_t(kCombineBarrierMarkerTotal - (activeQps - 1)) : uint64_t{1};
}

// Spin on a local symmetric slot until a peer publishes a positive value. No
// transport of its own: the write arrives either by intra-node store or by the
// NIC, and ShmemInt32WaitUntilGreaterThan was the same system-scope relaxed load
// in a loop.
__device__ __forceinline__ int32_t EpInterNodeWaitGt(int32_t* addr, int32_t val) {
  int32_t observed;
  while (true) {
    observed = core::AtomicLoadRelaxedSystem(addr);
    if (observed > val) break;
  }
  return observed;
}

// Wait until this wave's stores have completed (gfx9's vmcnt counts them; gfx12 has storecnt).
__device__ __forceinline__ void EpInterNodeWaitStores() {
#if defined(MORI_GFX125X) || defined(__GFX12__)
  __asm__ volatile("s_wait_storecnt 0x0" ::: "memory");
#else
  __builtin_amdgcn_s_waitcnt(0);
#endif
}

// A staged token is [hidden | indices | weights | scales], contiguous: forward it as one round of
// 16 B loads, then the stores. Needs every part 16 B-sized and at most 8 vectors per lane.
template <size_t kHidden, size_t kIndex, size_t kWeight, size_t kScale>
struct ForwardTokenVec16 {
  static constexpr int kVecs = static_cast<int>((kHidden + kIndex + kWeight + kScale) / 16);
  static constexpr int kPerLane = (kVecs + warpSize - 1) / warpSize;
  static constexpr bool kUsable = (kHidden % 16 == 0) && (kIndex % 16 == 0) &&
                                  (kWeight % 16 == 0) && (kScale % 16 == 0) && (kPerLane <= 8);
  // A native vector, not uint4: HIP's uint4 is a union wrapper copied with memcpy, which keeps
  // the array out of registers -- AMDGPU then promotes it to LDS and each 16 B load waits for the
  // previous one's LDS write.
  using Vec = unsigned int __attribute__((ext_vector_type(4)));
  Vec regs[kPerLane];

  __device__ __forceinline__ void Load(const uint8_t* src) {
    const int lane = threadIdx.x & (warpSize - 1);
#pragma unroll
    for (int u = 0; u < kPerLane; u++) {
      const int v = lane + u * warpSize;
      if (v < kVecs) regs[u] = reinterpret_cast<const Vec*>(src)[v];
    }
  }

  __device__ __forceinline__ void Store(uint8_t* hidden, uint8_t* index, uint8_t* weight,
                                        uint8_t* scale) const {
    constexpr int h = kHidden / 16, ix = kIndex / 16, w = kWeight / 16;
    const int lane = threadIdx.x & (warpSize - 1);
#pragma unroll
    for (int u = 0; u < kPerLane; u++) {
      const int v = lane + u * warpSize;
      if (v >= kVecs) continue;
      uint8_t* dst = v < h            ? hidden + 16 * v
                     : v < h + ix     ? index + 16 * (v - h)
                     : v < h + ix + w ? weight + 16 * (v - h - ix)
                                      : scale + 16 * (v - h - ix - w);
      *reinterpret_cast<Vec*>(dst) = regs[u];
    }
  }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpDispatchInterNodeV2                                  */
/* ---------------------------------------------------------------------------------------------- */
namespace internode {

// moeSort: as a warp places a token on a same-node PE it also buckets the token's experts on that
// PE for aiter's moe_sorting. Lane kTopk + j owns expert slot j, so its bucket atomic issues in
// the same instruction as the row atomic of lane 0 (or of the PE's representative lane).
//
// The moe_sort region holds kEpr counters, one per 128 B line so that different experts do not
// queue on one line at the owner; a 128 B line of per-same-node-PE dispatch epochs; then kEpr
// buckets of MaxNumTokensToRecv entries.
constexpr int kMoeSortCountStride = 128 / sizeof(ep_index_t);
constexpr int kMoeSortMaxGpuPerNode = 128 / sizeof(uint64_t);

// Slot `localPe` holds how many dispatches that same-node PE has started. One that has started
// this dispatch no longer reads this PE's combine input -- the expert output, which this dispatch
// zeroes -- whether or not it ran a combine in between.
template <int kEpr>
inline __device__ uint64_t* MoeSortStarted(uint8_t* region) {
  return reinterpret_cast<uint64_t*>(region + kEpr * 128);
}

// At the start of a moeSort dispatch: tell the same-node PEs this one has started it.
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void MoeSortAnnounce(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  static_assert(kConfig.gpuPerNode <= kMoeSortMaxGpuPerNode, "one epoch slot per same-node PE");
  if (globalThdId >= config.gpuPerNode) return;
  // This dispatch bumps the count only in its DispatchSync.
  const uint64_t started = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  uint64_t* slot = MoeSortStarted<kConfig.numExpertPerRank>(
                       args.reg(args.offMoeSort)
                           ->template GetAs<uint8_t*>(myNode * config.gpuPerNode + globalThdId)) +
                   myPe % config.gpuPerNode;
  core::AtomicStoreRelaxedSystem(slot, started);
}

template <EpInterNodeKernelCfg kConfig>
inline __device__ int MoeSortSlot(int laneId) {
  static_assert(!kConfig.moeSort || 2 * kConfig.numExpertPerToken <= warpSize,
                "moeSort needs two lanes per expert slot");
  return laneId - kConfig.numExpertPerToken;
}

template <EpInterNodeKernelCfg kConfig>
inline __device__ int2* MoeSortBuckets(uint8_t* region) {
  return reinterpret_cast<int2*>(region + kConfig.numExpertPerRank * 128 + 128);
}

// What a lane needs to bucket one expert slot, loaded before the row atomic so that the atomic is
// the only round trip the bucketing adds. counter == nullptr: nothing to bucket.
struct MoeSortLane {
  ep_index_t* counter{nullptr};
  int2* bucket{nullptr};
  float weight{0.f};
  int slot{0};
};

template <EpInterNodeKernelCfg kConfig>
inline __device__ MoeSortLane MoeSortPrepare(EpDispatchCombineArgs& args, int pe, ep_index_t expert,
                                             int slot, const float* weight) {
  constexpr EpInterNodeDeviceCfg config = EpInterNodeDeviceCfgOf(kConfig);
  const int local = expert % config.numExpertPerRank;
  uint8_t* region = args.reg(args.offMoeSort)->template GetAs<uint8_t*>(pe);
  MoeSortLane lane;
  lane.counter = reinterpret_cast<ep_index_t*>(region) + local * kMoeSortCountStride;
  lane.bucket =
      MoeSortBuckets<kConfig>(region) + static_cast<size_t>(local) * config.MaxNumTokensToRecv();
  lane.weight = *weight;
  lane.slot = slot;
  return lane;
}

// One bucket entry: (row | topk slot << 24, weight), aiter's sorted id and weight.
template <EpInterNodeKernelCfg kConfig>
inline __device__ void MoeSortPut(const MoeSortLane& lane, int pos, int row) {
  assert(pos < EpInterNodeDeviceCfgOf(kConfig).MaxNumTokensToRecv());
  lane.bucket[pos] = make_int2(row | (lane.slot << 24), __float_as_int(lane.weight));
}

// Sorted-row blocks whose expert the finalize looks up in LDS; beyond this it searches.
constexpr int kMoeSortLdsBlocks = 2048;

// The finalize's LDS: per-expert counts and offsets, and which expert each sorted block holds.
template <int kEpr>
struct MoeSortShared {
  int cnt[kEpr];
  int off[kEpr + 1];
  uint16_t blockExpert[kMoeSortLdsBlocks];
};

// Warp 0 of the last block, once every same-node PE is done: lane e reads and resets expert e's
// counter, in the same round trip as the caller's read of the row count.
template <EpInterNodeKernelCfg kConfig>
inline __device__ int MoeSortLoadCount(EpDispatchCombineArgs& args, int laneId) {
  constexpr int kEpr = kConfig.numExpertPerRank;
  static_assert(kEpr <= warpSize, "one lane per local expert");
  if (laneId >= kEpr) return 0;
  ep_index_t* ctr =
      args.reg(args.offMoeSort)->template GetAs<ep_index_t*>() + laneId * kMoeSortCountStride;
  const int c = __hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(ctr, ep_index_t{0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return c;
}

// Then: pad each expert to sortBlockM and lay them out in expert order, as aiter's moe_sorting does
// with expert_mask, writing sorted_expert_ids and num_valid_ids. MoeSortGather writes the rows.
template <EpInterNodeKernelCfg kConfig>
inline __device__ void MoeSortLayout(EpDispatchCombineArgs& args, int laneId, int c, int recvRows,
                                     MoeSortShared<kConfig.numExpertPerRank>& sh) {
  constexpr int kEpr = kConfig.numExpertPerRank;
  const int bm = args.sortBlockM;
  // aiter's block_m is a power of two; keep the division for anything else.
  const int p = (bm & (bm - 1)) == 0 ? (c + bm - 1) & -bm : (c + bm - 1) / bm * bm;
  int incl = p;
  for (int d = 1; d < warpSize; d <<= 1) {
    const int up = __shfl_up(incl, d);
    if (laneId >= d) incl += up;
  }
  const int off = incl - p;
  if (laneId < kEpr) {
    sh.cnt[laneId] = c;
    sh.off[laneId] = off;
    for (int b = off / bm; b < (off + p) / bm; b++) {
      args.sortedExpertIds[b] = laneId;
      if (b < kMoeSortLdsBlocks) sh.blockExpert[b] = static_cast<uint16_t>(laneId);
    }
  }
  if (laneId == kEpr - 1) {
    sh.off[kEpr] = incl;
    args.sortNumValid[0] = incl;
    args.sortNumValid[1] = recvRows;
    // FC2 accumulates into rows [0, recvRows], the last one from the weight-0 padding rows.
    if (args.sortZeroRowBytes > 0) args.sortZeroState[0] = recvRows + 1;
  }
}

// The whole last block, after MoeSortLayout: copy each expert's bucket to its range and pad it.
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void MoeSortGather(EpDispatchCombineArgs& args, int recvRows,
                                     const MoeSortShared<kConfig.numExpertPerRank>& sh) {
  DEF_COMMON_VARS;
  constexpr int kEpr = kConfig.numExpertPerRank;
  constexpr int kTopk = kConfig.numExpertPerToken;
  constexpr int kU = 16;  // rows per thread per pass, loaded before any is stored
  const int bm = args.sortBlockM;
  const int total = sh.off[kEpr];
  auto expertOf = [&](int i) {
    const int b = i / bm;
    if (b < kMoeSortLdsBlocks) return static_cast<int>(sh.blockExpert[b]);
    int lo = 0, hi = kEpr - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (sh.off[mid] <= i)
        lo = mid;
      else
        hi = mid - 1;
    }
    return lo;
  };
  // Each expert's bucket goes to a contiguous range, so consecutive rows read consecutive entries.
  const int2 pad = make_int2(recvRows | (kTopk << 24), 0);
  const int2* buckets =
      MoeSortBuckets<kConfig>(args.reg(args.offMoeSort)->template GetAs<uint8_t*>());
  for (int base = thdId; base < total; base += kU * thdNum) {
    int2 v[kU];
#pragma unroll
    for (int u = 0; u < kU; u++) {
      const int i = base + u * thdNum;
      v[u] = pad;
      if (i < total) {
        const int e = expertOf(i);
        const int j = i - sh.off[e];
        if (j < sh.cnt[e]) v[u] = buckets[static_cast<size_t>(e) * config.MaxNumTokensToRecv() + j];
      }
    }
#pragma unroll
    for (int u = 0; u < kU; u++) {
      const int i = base + u * thdNum;
      if (i < total) {
        args.sortedTokenIds[i] = v[u].x;
        args.sortedWeights[i] = __int_as_float(v[u].y);
      }
    }
  }
}

// moeSort with zeroing: the intra-node blocks clear the rows of the combine input the last FC2
// accumulated into (sortZeroState[0], set by MoeSortLayout) while the inter-node leg is still in
// flight -- once every same-node PE has started this dispatch (MoeSortAnnounce) and so has
// stopped reading them.
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void MoeSortZeroRows(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  if (thdId < config.gpuPerNode) {
    const uint64_t started = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
    const uint64_t* slots = MoeSortStarted<kConfig.numExpertPerRank>(
        args.reg(args.offMoeSort)->template GetAs<uint8_t*>());
    while (core::AtomicLoadRelaxedSystem(slots + thdId) != started) {
    }
  }
  __syncthreads();
  const size_t rowBytes = static_cast<size_t>(args.sortZeroRowBytes);
  const int rowNum = min(args.sortZeroState[0], config.MaxNumTokensToRecv());
  const size_t vecs = static_cast<size_t>(rowNum) * rowBytes / sizeof(uint4);
  const int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  uint4* dst = static_cast<uint4*>(args.sortZeroOut);
  for (size_t v = static_cast<size_t>(blockId - args.rdmaBlockNum) * thdNum + thdId; v < vecs;
       v += static_cast<size_t>(xgmiBlockNum) * thdNum)
    dst[v] = make_uint4(0, 0, 0, 0);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchIntraNodeBlock(EpDispatchCombineArgs& args, int tokenId, int expId,
                                              int destPe, index_t laneExpert) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (!args.replayMode) {
    // moeSort: lane kTopk + j buckets expert slot j if that expert is on destPe; laneExpert is
    // expert j in lane j.
    [[maybe_unused]] MoeSortLane sort;
    [[maybe_unused]] index_t got = 0;
    if constexpr (kConfig.moeSort) {
      const int j = MoeSortSlot<kConfig>(laneId);
      const index_t e = __shfl(laneExpert, (j >= 0 && j < config.numExpertPerToken) ? j : 0);
      if (j >= 0 && j < config.numExpertPerToken && e >= 0 && e / config.numExpertPerRank == destPe)
        sort = MoeSortPrepare<kConfig>(args, destPe, e, j,
                                       args.weightsBuf + tokenId * config.numExpertPerToken + j);
      index_t* counter = (laneId == 0)
                             ? args.reg(args.offDispTokOffset)->template GetAs<index_t*>(destPe)
                             : sort.counter;
      if (counter) got = atomicAdd(counter, 1);
    }
    if (laneId == 0) {
      // decide token id in dest pe
      if constexpr (kConfig.moeSort)
        destTokId = got;
      else
        destTokId = atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(destPe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.dispDestTokIdMap[tokenExpertId] = FlatTokenIndex(config, destPe, destTokId);

      core::AtomicStoreRelaxedSystem(
          args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(destPe) + destTokId,
          static_cast<index_t>(FlatTokenIndex(config, myPe, tokenId)));
    }
    destTokId = __shfl(destTokId, 0);
    if constexpr (kConfig.moeSort) {
      if (sort.counter) MoeSortPut<kConfig>(sort, got, destTokId);
    }
  } else {
    // Replay routing: reuse the slot recorded by a prior cache-routing dispatch.
    index_t flat = args.dispDestTokIdMap[tokenExpertId];
    destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
  }
  size_t srcTokOffset = tokenId * hiddenDim;
  size_t destTokOffset = destTokId * hiddenDim;

  T* remoteTokenPtr = args.reg(args.offDispatchOut)->template GetAs<T*>(destPe);
  const T* localTokenPtr = static_cast<const T*>(args.inpTokenBuf);
  core::WarpCopy(remoteTokenPtr + destTokOffset, localTokenPtr + srcTokOffset, hiddenDim);

  index_t* remoteIndexPtr = args.reg(args.offOutIndices)->template GetAs<index_t*>(destPe);
  const index_t* localIndexPtr = args.tokenIndices;
  core::WarpCopy(remoteIndexPtr + destTokId * config.numExpertPerToken,
                 localIndexPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  // weightsBuf is null when the caller dispatches without weights.
  if (args.weightsBuf) {
    float* remoteWeightPtr = args.reg(args.offDispatchOutWeights)->template GetAs<float*>(destPe);
    const float* localWeightPtr = args.weightsBuf;
    core::WarpCopy(remoteWeightPtr + destTokId * config.numExpertPerToken,
                   localWeightPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);
  }

  if (args.scalesBuf && (scaleBytes > 0)) {
    core::WarpCopy(
        args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
        args.scalesBuf + tokenId * scaleBytes, scaleBytes);
  }
}

// v2_ll intra-node blocks wait for this rank's RDMA posts, which their XGMI stores would slow;
// bounded (50 us), so a grid that is not fully resident cannot deadlock on it.
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void WaitLLPostsIssued(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  if (laneId != 0) return;
  const int expected = (nNodes - 1) * core::CeilDiv(args.curRankNumToken, warpSize);
  const uint64_t deadline = wall_clock64() + 5000;  // 50 us at the 100 MHz wall clock
  while (wall_clock64() < deadline) {
    int posted = 0;
    for (int n = 0; n < nNodes; n++) posted += core::AtomicLoadRelaxed(args.blockFlagCounter + n);
    if (posted >= expected) break;
    __builtin_amdgcn_s_sleep(16);  // these pollers are idle; keep their loads off the fabric
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchIntraNode(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int i = warpId; i < (endTokenIdx - startTokenIdx) * config.numExpertPerToken; i += warpNum) {
    index_t tokenId = i / config.numExpertPerToken + startTokenIdx;
    index_t expertOffset = startTokenIdx * config.numExpertPerToken + i;
    index_t destExpert = args.tokenIndices[expertOffset];
    if (destExpert < 0) {
      if (!args.replayMode && laneId == 0)
        args.dispDestTokIdMap[expertOffset] = NullFlatTokenIndex(config);
      continue;
    }
    index_t destPe = destExpert / config.numExpertPerRank;
    int destNode = destPe / config.gpuPerNode;

    int lanePe = -1;
    index_t laneExpert = -1;
    if (laneId < numExpertPerToken) {
      laneExpert = args.tokenIndices[tokenId * numExpertPerToken + laneId];
      // Sentinel lanes get a unique impossible destPe so dedup cannot false-match.
      lanePe = (laneExpert < 0) ? (-1 - static_cast<int>(laneId))
                                : (laneExpert / config.numExpertPerRank);
    }

    // Deduplicate
    index_t inTokenExpertId = i % numExpertPerToken;
    if (destNode == myNode) {
      if (__any((laneId < inTokenExpertId) && (destPe == lanePe))) {
        if (!args.replayMode && laneId == 0)
          args.dispDestTokIdMap[expertOffset] = NullFlatTokenIndex(config);
        continue;
      }
      DispatchIntraNodeBlock<kConfig, T>(args, tokenId, inTokenExpertId, destPe, laneExpert);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeSend(EpDispatchCombineArgs& args,
                                             const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);
  int totalChunkNum = core::CeilDiv(args.curRankNumToken, warpSize);
  int blockChunkNum = core::CeilDiv(totalChunkNum, args.rdmaBlockNum);

  int startTokenIdx = blockChunkNum * blockId * warpSize;
  int endTokenIdx = std::min(startTokenIdx + blockChunkNum * warpSize, args.curRankNumToken);

  // Then send to other nodes
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (myPe % config.gpuPerNode);
    for (int tokenId = startTokenIdx + laneId; tokenId < endTokenIdx; tokenId += warpSize) {
      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + e];
        if (laneExpert < 0) continue;
        int destNode = laneExpert / config.numExpertPerRank / config.gpuPerNode;
        if (destNode == i) {
          shouldSend |= true;
          if (!args.replayMode)
            args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = NullFlatTokenIndex(config);
        }
      }
      uint64_t mask = __ballot(shouldSend) & __activemask();
      uint64_t num = __popcll(mask);

      if (num == 0) continue;

      // atomicAdd runs in both paths so blockFlagCounter stays in sync with cache routing.
      index_t flag = 0;
      index_t flagSlotId = 0;
      if (laneId == 0) {
        flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
        flag = num + 1;
      }
      flag = __shfl(flag, 0);
      flagSlotId = __shfl(flagSlotId, 0);

      if (args.replayMode) {
        // Recover the deterministic flag slot from the cached send map.
        int firstSender = __ffsll(static_cast<unsigned long long>(mask)) - 1;
        index_t myCached = shouldSend ? args.interNodeDispSendMap[nNodes * tokenId + i] : 0;
        flagSlotId = __shfl(myCached, firstSender) / warpSize;
      }

      index_t destTokIdOffset = flagSlotId * warpSize;

      uint64_t warpOffset = 0;
      if (laneId > 0) warpOffset = __popcll(mask << (warpSize - laneId));
      index_t destTokId = destTokIdOffset + warpOffset;

      if (shouldSend) {
        bool prev = (laneId > 0) ? ((mask >> (laneId - 1)) & 1ULL) : 0;
        int count = 0;
        if (!prev) {
          count = 1;
          for (int i = laneId + 1; i < warpSize; i++) {
            if ((mask >> i) & 1ULL) {
              count++;
            } else {
              break;
            }
          }
        }
        size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
        if (count > 0) {
          size_t stagingTokOffset = tokenId * xferBytes;
          int qpId = (tokenId / warpSize) % config.numQpPerPe;
          EpInterNodeChunkPut<kConfig.noRdmaAtomics>(
              comm, args.reg(args.offDispatchInp), remoteIdx * xferBytes,
              args.reg(args.offDispatchStaging), stagingTokOffset, count * xferBytes,
              args.reg(args.offChunkFlag), (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t),
              flag, proxyPe, qpId);
        }
        if (!args.replayMode) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    // laneId == myNode would signal slot myNode on a proxy that shares our node,
    // and DispatchInterNodeRecv only ever polls slots of remote nodes, so that
    // write is dead. It is also unroutable: RAIL connects cross-node same-rail
    // peers only, so a local peer has no QP and EpInterNodeAtomicAdd would fault.
    // v1 needed no guard because shmem dispatches per peer on
    // globalGpuStates->transportTypes[pe] (shmem_device_api.hpp): a same-node
    // peer is TransportType::P2P, so the atomic became a local XGMI atomic and
    // never touched a NIC. ccoGda has no such dispatch -- cco_scale_out.hpp
    // mentions neither P2P nor locality -- so scale-out is the only path it can
    // take, and it has nowhere to send this.
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      EpInterNodeCountSignal<kConfig.noRdmaAtomics>(comm, args.reg(args.offNodeRecvTokenNum),
                                                    myNode * sizeof(uint64_t), numTokenSignal,
                                                    proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeLLSend(EpDispatchCombineArgs& args,
                                               const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  // Then send to other nodes
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);
  int totalChunkNum = core::CeilDiv(args.curRankNumToken, warpSize);
  int blockChunkNum = core::CeilDiv(totalChunkNum, args.rdmaBlockNum);
  int chunkStartTokenIdx = blockChunkNum * blockId * warpSize;
  int chunkEndTokenIdx =
      std::min(chunkStartTokenIdx + blockChunkNum * warpSize, args.curRankNumToken);
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (myPe % config.gpuPerNode);

    for (int tokenId = chunkStartTokenIdx + laneId; tokenId < chunkEndTokenIdx;
         tokenId += warpSize) {
      // Every chunk goes to every node here, so slot k is chunk k and the slots are exactly
      // 0 .. S-1 without an atomic to hand them out.
      const index_t flagSlotId = (tokenId - laneId) / warpSize;
      index_t destTokIdOffset = flagSlotId * warpSize;
      index_t destTokId = destTokIdOffset + laneId;

      // Post first: the routing bookkeeping below is read only by combine, and its dependent
      // load per expert would otherwise hold back the chunk's RDMA write.
      size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
      if (laneId == 0) {
        index_t tokenNum = std::min(tokenId + warpSize, chunkEndTokenIdx) - tokenId;
        size_t stagingTokOffset = tokenId * xferBytes;
        const int pieceTokens = EpInterNodeDispatchPieceTokens<kConfig>(tokenNum);
        for (int first = 0, piece = 0; first < tokenNum; first += pieceTokens, piece++) {
          const int count = std::min(pieceTokens, static_cast<int>(tokenNum) - first);
          const int qpId = (flagSlotId + piece) % config.numQpPerPe;
          if (piece == 0) {
            EpInterNodeChunkPut<kConfig.noRdmaAtomics>(
                comm, args.reg(args.offDispatchInp), remoteIdx * xferBytes,
                args.reg(args.offDispatchStaging), stagingTokOffset, count * xferBytes,
                args.reg(args.offChunkFlag), (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t),
                tokenNum + 1, proxyPe, qpId);
          } else {
            const uint64_t pieceTag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
            EpInterNodePutThenWrite(
                comm, args.reg(args.offDispatchInp), (remoteIdx + first) * xferBytes,
                args.reg(args.offDispatchStaging), stagingTokOffset + first * xferBytes,
                count * xferBytes, args.reg(args.offChunkFlag),
                EpInterNodePieceFlagSlot(nNodes, maxChunkNum, myNode, flagSlotId, piece) *
                    sizeof(uint64_t),
                pieceTag, proxyPe, qpId);
          }
        }
        // Counted once posted: WaitLLPostsIssued waits on it, and whoever posts node i's last
        // chunk sends its token count, so no barrier over every RDMA warp is needed.
        const int posted = atomicAdd(args.blockFlagCounter + i, 1) + 1;
        if (posted == totalChunkNum) {
          EpInterNodeCountSignal<kConfig.noRdmaAtomics>(
              comm, args.reg(args.offNodeRecvTokenNum), myNode * sizeof(uint64_t),
              static_cast<uint64_t>(posted) * warpSize + 1, proxyPe);
        }
      }

      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destNode = args.tokenIndices[tokenId * numExpertPerToken + e] /
                       config.numExpertPerRank / config.gpuPerNode;
        if (destNode == i) {
          shouldSend |= true;
          args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = NullFlatTokenIndex(config);
        }
      }
      if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
    }
  }

  // Nothing to post: the count (zero chunks) still has to reach every remote node.
  if (totalChunkNum == 0 && globalWarpId == 0 && laneId < nNodes && laneId != myNode) {
    int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
    EpInterNodeCountSignal<kConfig.noRdmaAtomics>(comm, args.reg(args.offNodeRecvTokenNum),
                                                  myNode * sizeof(uint64_t), 1, proxyPe);
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeRecv(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();

  for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
       bid += args.rdmaBlockNum) {
    int k = bid / (numRecvBlock * (nNodes - 1));
    int i = (bid / numRecvBlock) % (nNodes - 1);

    int node = (myNode + 1 + i) % nNodes;
    int startTokenIdx = k * warpSize;

    uint64_t thisChunkTokenNum = 0;
    index_t nodeFlag = 0;
    if (laneId == 0) {
      while (1) {
        thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkTokenNum > 0) break;

        nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
        if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
          thisChunkTokenNum = 1;
          break;
        }
      }
    }
    thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;

    int endTokenIdx = startTokenIdx + thisChunkTokenNum;

    // bid, NOT blockId: bid strides by rdmaBlockNum, so the two agree only when
    // rdmaBlockNum % numRecvBlock == 0. With any other rdmaBlockNum the eight
    // blocks cooperating on a chunk collide on some sub-indices and never issue
    // others -- tokens silently dropped, and others delivered twice. The combine
    // twin in CombineInterNodeTyped already uses bid.
    for (int j = startTokenIdx + (bid % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
         j += numRecvBlock * warpNum) {
      int tokIdx = SendBufSlotOffset(config, node, j);
      index_t* indices = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
      // Sentinel lanes (-1 expert) get a unique impossible destPe to avoid false dup-matches.
      int lanePe = -1;
      if (laneId < config.numExpertPerToken) {
        index_t laneExpert = indices[laneId];
        lanePe = (laneExpert < 0) ? (-1 - static_cast<int>(laneId))
                                  : (laneExpert / config.numExpertPerRank);
        assert((laneExpert < 0) || ((lanePe < config.worldSize) && (lanePe >= 0)));
      }
      index_t srcTokId = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes +
                                                    indexBytes + weightBytes + scaleBytes)[0];

      // Allocate slots in parallel across distinct destination PEs after deduplication.
      // Payload copies remain sequential per peer.
      bool laneValid = false;
      if (laneId < config.numExpertPerToken) {
        // Validate the peer before GetAs/atomicAdd, including Release builds.
        bool oob = (lanePe < 0) || (lanePe >= config.worldSize);
        laneValid = !oob && (lanePe / config.gpuPerNode == myNode);
      }
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int peOfE = __shfl(lanePe, e);
        // Keep only the first expert slot for each destination PE.
        bool dup = __any((laneId < e) && (peOfE == lanePe));
        if (dup && laneId == e) laneValid = false;
      }

      index_t laneDestTokId = 0;
      if (laneId < config.numExpertPerToken) {
        index_t* mapSlot =
            args.interNodeDispDestTokIdMap + tokIdx * config.numExpertPerToken + laneId;
        if (!args.replayMode) {
          if (laneValid) {
            laneDestTokId =
                atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(lanePe), 1);
            assert(laneDestTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            *mapSlot = FlatTokenIndex(config, lanePe, laneDestTokId);
            args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(lanePe)[laneDestTokId] =
                srcTokId;
          } else {
            *mapSlot = NullFlatTokenIndex(config);
          }
        } else if (laneValid) {
          laneDestTokId = LocalTokIdFromFlatTokenIndex(config, *mapSlot);
        }
      }
      if constexpr (kConfig.moeSort) {
        // Every local expert slot of this token, at the row its PE's valid lane took.
        int rep = 0;
        for (int e = 0; e < config.numExpertPerToken; e++) {
          const int peOfE = __shfl(lanePe, e);
          if (__shfl(static_cast<int>(laneValid), e) && peOfE == lanePe) rep = e;
        }
        const int row = __shfl(laneDestTokId, rep);
        if (laneId < config.numExpertPerToken && !args.replayMode && lanePe >= 0 &&
            lanePe < config.worldSize && lanePe / config.gpuPerNode == myNode) {
          const MoeSortLane sort = MoeSortPrepare<kConfig>(
              args, lanePe, indices[laneId], laneId,
              reinterpret_cast<const float*>(stagingPtr + tokIdx * xferBytes + hiddenBytes +
                                             indexBytes) +
                  laneId);
          MoeSortPut<kConfig>(sort, atomicAdd(sort.counter, 1), row);
        }
      }

      for (int e = 0; e < config.numExpertPerToken; e++) {
        if (!__shfl(static_cast<int>(laneValid), e)) continue;
        int destPe = __shfl(lanePe, e);
        index_t destTokId = __shfl(laneDestTokId, e);
        core::WarpCopy<uint8_t, 4>(args.reg(args.offDispatchOut)->template GetAs<uint8_t*>(destPe) +
                                       destTokId * hiddenBytes,
                                   stagingPtr + tokIdx * xferBytes, hiddenBytes);
        core::WarpCopy<uint8_t, 4>(
            args.reg(args.offOutIndices)->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes, indexBytes);
        core::WarpCopy<uint8_t, 4>(
            args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>(destPe) +
                destTokId * weightBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes, weightBytes);
        if ((scaleBytes > 0)) {
          core::WarpCopy<uint8_t, 4>(
              args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) +
                  destTokId * scaleBytes,
              stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes + weightBytes, scaleBytes);
        }
      }
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeLLRecv(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();

  // Cache nodeRecvTokenNum[node]: unchanged once observed non-zero until
  // combine resets it. One scalar + tag, not a per-node array -- nodeId only
  // increases across this loop, so at most one node is ever "current".
  int cachedNode = -1;
  uint64_t cachedNodeFlag = 0;

  // expert -> token -> node
  for (int i = globalWarpId;
       i < config.MaxNumTokensToSendPerRank() * config.numExpertPerToken * (nNodes - 1);
       i += args.rdmaBlockNum * warpNum) {
    int expertId = i % config.numExpertPerToken;
    int tokenId = i / config.numExpertPerToken % config.MaxNumTokensToSendPerRank();
    int nodeId = i / config.numExpertPerToken / config.MaxNumTokensToSendPerRank();

    int node = (myNode + 1 + nodeId) % nNodes;
    int k = tokenId / warpSize;
    int startTokenIdx = k * warpSize;

    // Poll completion flags
    uint64_t thisChunkTokenNum = 0;
    if (laneId == 0) {
      if (node != cachedNode) {
        cachedNode = node;
        cachedNodeFlag = 0;
      }
      while (1) {
        // Checked before chunkFlag, and that is only safe because both senders number their
        // slots 0 .. S-1 (DispatchInterNodeLLSend by chunk index, DispatchInterNodeSend by
        // atomicAdd(blockFlagCounter)) and send nodeRecvTokenNum = S * warpSize + 1. A slot at
        // or past that bound was never handed out and can never get a flag. Numbering the
        // slots any other way would make this drop tokens silently.
        if ((cachedNodeFlag > 0) && (startTokenIdx >= (cachedNodeFlag - 1))) {
          thisChunkTokenNum = 1;
          break;
        }
        thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkTokenNum > 0) break;

        // Won't change again this round once observed; stop re-reading it.
        if (cachedNodeFlag == 0)
          cachedNodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
      }
    }
    thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;
    // Zero only on the bound branch above: a flag the sender writes is
    // tokenNum + 1 with tokenNum >= 1. So this slot is at or past the node's
    // last one -- and, the slots being 0 .. S-1 and k never decreasing as i
    // walks a node, so is every later iteration of this node. The loop is
    // bounded by capacity, not by what arrived, so at capacity >> load nearly
    // all iterations land here; jump to this warp's first iteration of the
    // next node instead of walking through them one by one.
    if (thisChunkTokenNum == 0) {
      const int stride = args.rdmaBlockNum * warpNum;
      const int nextNodeStart =
          (nodeId + 1) * config.numExpertPerToken * config.MaxNumTokensToSendPerRank();
      const int offset = ((globalWarpId - nextNodeStart) % stride + stride) % stride;
      i = nextNodeStart + offset - stride;  // the loop increment adds stride back
      continue;
    }
    int endTokenIdx = startTokenIdx + thisChunkTokenNum;
    if (tokenId >= endTokenIdx) continue;
    EpInterNodeWaitPiece<kConfig>(chunkFlag, nNodes, maxChunkNum, node, k, tokenId - startTokenIdx,
                                  thisChunkTokenNum, args.crossDeviceBarrierFlag);

    int globalTokenId = SendBufSlotOffset(config, node, tokenId);
    index_t* indices =
        reinterpret_cast<index_t*>(stagingPtr + globalTokenId * xferBytes + hiddenBytes);
    // Sentinel lanes (-1 expert) get a unique impossible destPe to avoid false
    // dup-matches, exactly as DispatchInterNodeRecv does. -1 / numExpertPerRank
    // truncates toward zero, so without this an unfilled slot reads as PE 0.
    int lanePe = -1;
    if (laneId < config.numExpertPerToken) {
      index_t laneExpert = indices[laneId];
      lanePe = (laneExpert < 0) ? (-1 - static_cast<int>(laneId))
                                : (laneExpert / config.numExpertPerRank);
      assert((laneExpert < 0) || ((lanePe < config.worldSize) && (lanePe >= 0)));
    }
    index_t srcTokId =
        reinterpret_cast<index_t*>(stagingPtr + globalTokenId * xferBytes + hiddenBytes +
                                   indexBytes + weightBytes + scaleBytes)[0];

    int destPe = __shfl(lanePe, expertId);
    int destNode = destPe / config.gpuPerNode;
    // HSA-RCA Signature 1 guard, mirroring the DispatchInterNodeRecv site: an
    // out-of-range destPe (the assert above is stripped under NDEBUG) is dropped
    // instead of writing OOB.
    bool peOutOfRange = (destPe < 0) || (destPe >= config.worldSize);
    bool shouldSkip =
        peOutOfRange || (destNode != myNode) || __any((laneId < expertId) && (destPe == lanePe));
    if (shouldSkip) {
      if (laneId == 0)
        args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
            NullFlatTokenIndex(config);
      continue;
    }

    int destTokId = 0;
    if (laneId == 0) {
      destTokId = atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(destPe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
          FlatTokenIndex(config, destPe, destTokId);
      args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
    }
    destTokId = __shfl(destTokId, 0);
    if constexpr (kConfig.moeSort) {
      // Every expert slot of this token on destPe; this path is not latency-critical.
      if (laneId < config.numExpertPerToken && indices[laneId] >= 0 &&
          indices[laneId] / config.numExpertPerRank == destPe) {
        const MoeSortLane sort = MoeSortPrepare<kConfig>(
            args, destPe, indices[laneId], laneId,
            reinterpret_cast<const float*>(stagingPtr + globalTokenId * xferBytes + hiddenBytes +
                                           indexBytes) +
                laneId);
        MoeSortPut<kConfig>(sort, atomicAdd(sort.counter, 1), destTokId);
      }
    }
    core::WarpCopy<uint8_t, 4>(
        args.reg(args.offDispatchOut)->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
        stagingPtr + globalTokenId * xferBytes, hiddenBytes);
    core::WarpCopy<uint8_t, 4>(
        args.reg(args.offOutIndices)->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
        stagingPtr + globalTokenId * xferBytes + hiddenBytes, indexBytes);
    core::WarpCopy<uint8_t, 4>(
        args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>(destPe) +
            destTokId * weightBytes,
        stagingPtr + globalTokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
    if ((scaleBytes > 0)) {
      core::WarpCopy<uint8_t, 4>(
          args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
          stagingPtr + globalTokenId * xferBytes + hiddenBytes + indexBytes + weightBytes,
          scaleBytes);
    }
  }
}

// v2_ll receive, one token per warp: indices read once, one slot atomic per distinct local PE,
// the token loaded once and stored to each. Tokens ForwardTokenVec16 cannot hold take
// DispatchInterNodeLLRecv.
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeLLRecvTok(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  constexpr size_t kH = config.HiddenBytes(sizeof(T)), kI = config.IndexBytes(),
                   kW = config.WeightBytes(), kS = config.ScaleBytes();
  using Forward = ForwardTokenVec16<kH, kI, kW, kS>;
  constexpr int kTopk = kConfig.numExpertPerToken;

  const int maxTok = config.MaxNumTokensToSendPerRank();
  int maxChunkNum = core::CeilDiv(maxTok, warpSize);
  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();

  // nodeRecvTokenNum[node] as in DispatchInterNodeLLRecv: one cached node at a time.
  int cachedNode = -1;
  uint64_t cachedNodeFlag = 0;

  const int stride = args.rdmaBlockNum * warpNum;
  for (int i = globalWarpId; i < maxTok * (nNodes - 1); i += stride) {
    int tokenId = i % maxTok;
    int nodeId = i / maxTok;
    int node = (myNode + 1 + nodeId) % nNodes;
    int k = tokenId / warpSize;
    int startTokenIdx = k * warpSize;

    uint64_t thisChunkTokenNum = 0;
    if (laneId == 0) {
      if (node != cachedNode) {
        cachedNode = node;
        cachedNodeFlag = 0;
      }
      while (1) {
        // Slots are 0 .. S-1 and nodeRecvTokenNum is S * warpSize + 1; see
        // DispatchInterNodeLLRecv.
        if ((cachedNodeFlag > 0) && (startTokenIdx >= (cachedNodeFlag - 1))) {
          thisChunkTokenNum = 1;
          break;
        }
        thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkTokenNum > 0) break;
        if (cachedNodeFlag == 0)
          cachedNodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
        __builtin_amdgcn_s_sleep(4);  // most of the warps wait here; space out their loads
      }
    }
    thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;
    // Past the node's last slot, and so is the rest of this node: go to the next one.
    if (thisChunkTokenNum == 0) {
      const int nextNodeStart = (nodeId + 1) * maxTok;
      const int offset = ((globalWarpId - nextNodeStart) % stride + stride) % stride;
      i = nextNodeStart + offset - stride;  // the loop increment adds stride back
      continue;
    }
    if (tokenId >= startTokenIdx + static_cast<int>(thisChunkTokenNum)) continue;
    EpInterNodeWaitPiece<kConfig>(chunkFlag, nNodes, maxChunkNum, node, k, tokenId - startTokenIdx,
                                  thisChunkTokenNum, args.crossDeviceBarrierFlag);

    const int globalTokenId = SendBufSlotOffset(config, node, tokenId);
    const uint8_t* slot = stagingPtr + globalTokenId * xferBytes;
    Forward fwd;
    fwd.Load(slot);

    // Lane e < topk is expert e. A representative lane is the first expert of a local PE;
    // sentinel (-1) experts get a unique impossible PE, as in DispatchInterNodeLLRecv.
    int lanePe = -1;
    index_t laneExpert = -1;
    if (laneId < kTopk) {
      laneExpert = reinterpret_cast<const index_t*>(slot + kH)[laneId];
      lanePe = (laneExpert < 0) ? (-1 - static_cast<int>(laneId))
                                : (laneExpert / config.numExpertPerRank);
    }
    const index_t srcTokId = reinterpret_cast<const index_t*>(slot + kH + kI + kW + kS)[0];
    bool isRep = (laneId < kTopk) && (lanePe >= 0) && (lanePe < config.worldSize) &&
                 (lanePe / config.gpuPerNode == myNode);
    // moeSort: lane kTopk + j buckets expert slot j on its PE; sortRep is that PE's rep lane.
    [[maybe_unused]] MoeSortLane sort;
    [[maybe_unused]] int sortPe = -1, sortRep = 0;
    if constexpr (kConfig.moeSort) {
      const int j = MoeSortSlot<kConfig>(laneId);
      const index_t e = __shfl(laneExpert, (j >= 0 && j < kTopk) ? j : 0);
      const int pe = e / config.numExpertPerRank;
      if (j >= 0 && j < kTopk && e >= 0 && pe < config.worldSize &&
          pe / config.gpuPerNode == myNode) {
        sortPe = pe;
        sort = MoeSortPrepare<kConfig>(args, pe, e, j,
                                       reinterpret_cast<const float*>(slot + kH + kI) + j);
      }
      sortRep = -1;
    }
#pragma unroll
    for (int j = 0; j < kTopk; j++) {
      const int pj = __shfl(lanePe, j);
      if (j < laneId && pj == lanePe) isRep = false;
      if constexpr (kConfig.moeSort) {
        if (sortRep < 0 && pj == sortPe) sortRep = j;
      }
    }

    int destTokId = 0;
    if constexpr (kConfig.moeSort) {
      index_t* counter =
          isRep ? args.reg(args.offDispTokOffset)->template GetAs<index_t*>(lanePe) : sort.counter;
      int got = 0;
      if (counter) got = atomicAdd(counter, 1);
      const int row = __shfl(got, sortRep < 0 ? 0 : sortRep);
      if (sort.counter) MoeSortPut<kConfig>(sort, got, row);
      destTokId = got;
    }
    if (isRep) {
      if constexpr (!kConfig.moeSort)
        destTokId = atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(lanePe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.interNodeDispDestTokIdMap[globalTokenId * kTopk + laneId] =
          FlatTokenIndex(config, lanePe, destTokId);
      args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(lanePe)[destTokId] = srcTokId;
    } else if (laneId < kTopk) {
      args.interNodeDispDestTokIdMap[globalTokenId * kTopk + laneId] = NullFlatTokenIndex(config);
    }

    uint64_t reps = __ballot(isRep);
    while (reps) {
      const int j = __ffsll(static_cast<unsigned long long>(reps)) - 1;
      reps &= reps - 1;
      const int destPe = __shfl(lanePe, j);
      const int dt = __shfl(destTokId, j);
      fwd.Store(args.reg(args.offDispatchOut)->template GetAs<uint8_t*>(destPe) + dt * kH,
                args.reg(args.offOutIndices)->template GetAs<uint8_t*>(destPe) + dt * kI,
                args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>(destPe) + dt * kW,
                kS > 0 ? args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) + dt * kS
                       : nullptr);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchSync(EpDispatchCombineArgs& args,
                                    const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  // Senders take slots with atomics on the destination's dispTokOffset, so once every local PE
  // is done a PE's own counter is its received total. A warp's slot atomics have returned and
  // its forwarded stores complete here, before its block's one arrival; completed means visible
  // at the peer because the window is uncached.
  EpInterNodeWaitStores();
  __syncthreads();
  // moeSort: the received rows in the last block, -1 in the others, and the finalize's tables.
  [[maybe_unused]] __shared__ int sLastRecv;
  [[maybe_unused]] __shared__ MoeSortShared<kConfig.moeSort ? kConfig.numExpertPerRank : 1> sSort;
  if (warpId == 0) {
    int finishedBlock = 0;
    if (laneId == 0) finishedBlock = atomicAdd(args.dispatchGridBarrier, 1);
    finishedBlock = __shfl(finishedBlock, 0);
    if constexpr (kConfig.moeSort) {
      if (laneId == 0) sLastRecv = -1;
    }
    if ((finishedBlock + 1) == blockNum) {
      const int nodePeOffset = myNode * config.gpuPerNode;
      index_t* doneFlags = args.reg(args.offRecvTokenNum)->template GetAs<index_t*>();
      if (laneId < config.gpuPerNode) {
        __hip_atomic_store(
            args.reg(args.offRecvTokenNum)->template GetAs<index_t*>(nodePeOffset + laneId) + myPe,
            index_t{1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      if (laneId == 0)
        __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      if (laneId < config.gpuPerNode) {
        index_t* done = doneFlags + nodePeOffset + laneId;
        EpInterNodeWaitGt(done, 0);
        __hip_atomic_store(done, index_t{0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      // Read after every local PE said it is done (the wave reconverges past the waits).
      [[maybe_unused]] int sortCount = 0, total = 0;
      if constexpr (kConfig.moeSort) sortCount = MoeSortLoadCount<kConfig>(args, laneId);
      if (laneId == 0) {
        index_t* slots = args.reg(args.offDispTokOffset)->template GetAs<index_t*>();
        total = __hip_atomic_load(slots, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        args.totalRecvTokenNum[0] = total;
        __hip_atomic_store(slots, index_t{0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        atomicAdd(args.crossDeviceBarrierFlag, 1);
        if constexpr (kConfig.moeSort) sLastRecv = total;
      }
      if constexpr (kConfig.moeSort)
        MoeSortLayout<kConfig>(args, laneId, sortCount, __shfl(total, 0), sSort);
    }
  }

  // Drain every QP any kernel of this op may send on (a combine may have used more
  // than this dispatch), capped by the allocation; one (remote node, QP) per warp.
  // Skip the local node: under RAIL its endpoint is null and flush() would fault.
  constexpr int kDrainQps =
      kConfig.numQpToDrain > kConfig.numQpPerPe ? kConfig.numQpToDrain : kConfig.numQpPerPe;
  const int drainQps = min(kDrainQps, comm.ibgda.numQpPerPe);
  for (int pair = globalWarpId; pair < (nNodes - 1) * drainQps; pair += globalWarpNum) {
    int node = (myNode + 1 + pair / drainQps) % nNodes;
    int proxyPe = node * config.gpuPerNode + (myPe % config.gpuPerNode);
    EpInterNodeQuiet(comm, proxyPe, pair % drainQps);
  }

  // After the drain, so the warps draining do not first wait out warp 0's done-wait.
  if constexpr (kConfig.moeSort) {
    __syncthreads();
    if (sLastRecv >= 0) MoeSortGather<kConfig, T>(args, sLastRecv, sSort);
  }
}

}  // namespace internode

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchInterNodeV2_body(EpDispatchCombineArgs args,
                                           const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  if constexpr (kConfig.moeSort) internode::MoeSortAnnounce<kConfig, T>(args);
  if (blockId < args.rdmaBlockNum) {
    internode::DispatchInterNodeSend<kConfig, T>(args, comm);
    internode::DispatchInterNodeRecv<kConfig, T>(args);
  } else {
    internode::DispatchIntraNode<kConfig, T>(args);
    if constexpr (kConfig.moeSort) {
      if (args.sortZeroRowBytes > 0) internode::MoeSortZeroRows<kConfig, T>(args);
    }
  }
  internode::DispatchSync<kConfig, T>(args, comm);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchCopyToStaging_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  if (args.curRankNumToken == 0) return;

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  // First copy to staging buffer
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    uint8_t* stagingPtr = args.reg(args.offDispatchStaging)->template GetAs<uint8_t*>();
    size_t stagingTokOffset = tokenId * xferBytes;
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenDimOffset * sizeof(T),
                               static_cast<const uint8_t*>(args.inpTokenBuf) +
                                   tokenId * hiddenBytes + hiddenDimOffset * sizeof(T),
                               hiddenDimSize * sizeof(T));
    if (inTokenPartId != 0) continue;
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenBytes,
                               reinterpret_cast<uint8_t*>(args.tokenIndices) + tokenId * indexBytes,
                               indexBytes);
    // weightsBuf is null when the caller dispatches without weights; the slot's
    // weight bytes are then only forwarded, never accumulated -- every reader of
    // the weights regions is itself weightsBuf-guarded.
    if (args.weightsBuf)
      core::WarpCopy<uint8_t, 4>(
          stagingPtr + stagingTokOffset + hiddenBytes + indexBytes,
          reinterpret_cast<uint8_t*>(args.weightsBuf) + tokenId * weightBytes, weightBytes);
    if (args.scalesBuf && (scaleBytes > 0))
      core::WarpCopy<uint8_t, 4>(
          stagingPtr + stagingTokOffset + hiddenBytes + indexBytes + weightBytes,
          reinterpret_cast<uint8_t*>(args.scalesBuf) + tokenId * scaleBytes, scaleBytes);
    if (laneId == 0)
      reinterpret_cast<index_t*>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes +
                                 weightBytes + scaleBytes)[0] =
          static_cast<index_t>(FlatTokenIndex(config, myPe, tokenId));
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchInterNodeV2LL_body(EpDispatchCombineArgs args,
                                             const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  if constexpr (kConfig.moeSort) internode::MoeSortAnnounce<kConfig, T>(args);
  if (blockId < args.rdmaBlockNum) {
    internode::DispatchInterNodeLLSend<kConfig, T>(args, comm);
    constexpr EpInterNodeDeviceCfg kCfg = EpInterNodeDeviceCfgOf(kConfig);
    if constexpr (ForwardTokenVec16<kCfg.HiddenBytes(sizeof(T)), kCfg.IndexBytes(),
                                    kCfg.WeightBytes(), kCfg.ScaleBytes()>::kUsable) {
      internode::DispatchInterNodeLLRecvTok<kConfig, T>(args);
    } else {
      internode::DispatchInterNodeLLRecv<kConfig, T>(args);
    }
  } else {
    internode::WaitLLPostsIssued<kConfig, T>(args);
    internode::DispatchIntraNode<kConfig, T>(args);
    if constexpr (kConfig.moeSort) {
      if (args.sortZeroRowBytes > 0) internode::MoeSortZeroRows<kConfig, T>(args);
    }
  }
  internode::DispatchSync<kConfig, T>(args, comm);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpCombineInterNodeV2                                   */
/* ---------------------------------------------------------------------------------------------- */
namespace internode {

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineSync(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  int tokenPerBlock = core::CeilDiv(totalRecvTokenNum, blockNum);
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, totalRecvTokenNum);
  T* combineInp = args.reg(args.offCombineInp)->template GetAs<T*>();
  // A caller that wrote into combine_in_view() already put the rows here.
  if (args.inpTokenBuf != combineInp) {
    for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
      core::WarpCopy(combineInp + tokenId * hiddenDim,
                     static_cast<const T*>(args.inpTokenBuf) + tokenId * hiddenDim, hiddenDim);
    }
  }
  if (args.weightsBuf) {
    for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
      core::WarpCopy(args.reg(args.offInpWeights)->template GetAs<float*>() +
                         tokenId * config.numExpertPerToken,
                     args.weightsBuf + tokenId * config.numExpertPerToken,
                     config.numExpertPerToken);
    }
  }
}

namespace combine_impl {

// The LL gather uses 16-byte loads per lane. Gathering a token from its experts
// can be limited by xGMI peer-read latency: for a topk with its own WarpAccum
// case (1, 2, 4, 6, 8, 10), core::WarpAccum issues AccumNum loads before
// accumulating, and wider loads put more bytes in flight. Other topk (e.g. 9,
// 16) go through WarpAccumDynamic, which loads 16-byte vectors one source at a
// time. The ordinary path has a separate 4-byte load-first policy below.
//
// Alignment and slice sizing matter here:
//   - Both ends must be 16B-aligned. A combine staging slot interleaves the
//     hidden payload with the per-token weights, so its stride is only aligned
//     for some topk/dtype combinations; CombineVecAligned() decides per launch
//     and the caller falls back to the 4B path when it cannot.
//   - A partial vector step is handled by a per-lane scalar tail. LL slices are
//     rounded to CombineVecStep() to avoid short scalar-only gathers. This is a
//     performance choice; a tail does not make the gather incorrect.
// CombineVecStep() controls how work is divided between warps for each token.
// CombineGather() chooses the vector width from the alignment check.
constexpr size_t kCombineVecBytes = 16;

// How many vector steps of a token's hidden dimension go into one warp's slice.
//
// This only sets the split -- CombineVecStep() feeds warpsPerToken below, and
// the slice is rounded up to a whole number of steps so no warp gets less than
// one. It does not reach inside the gather: WarpAccum advances exactly one step
// (warpSize * kCombineVecBytes) per inner iteration regardless of what is set
// here, so a slice of 2 steps simply means each warp runs two iterations.
//
// 2 measures faster than 1 at every token count tried, which is a statement
// about how wide to spread a token, not about the gather's inner loop.
constexpr int kCombineStepsPerWarpSlice = 2;
// The smallest piece a v2_ll relay chunk is split into across QPs.
constexpr int kCombinePieceMinTokens = 8;

inline __device__ bool CombineVecAligned(size_t tokHiddenBytes, size_t tokCombXferBytes) {
  return ((tokHiddenBytes % kCombineVecBytes) == 0) && ((tokCombXferBytes % kCombineVecBytes) == 0);
}

template <typename TokT>
inline __device__ size_t CombineVecStep(int warpSizeRt) {
  return static_cast<size_t>(kCombineStepsPerWarpSlice) * warpSizeRt *
         (kCombineVecBytes / sizeof(TokT));
}

template <typename TokT>
inline __device__ void CombineGather(TokT* dest, TokT** srcPtrs, int accumNum, size_t nelems,
                                     bool vecAligned) {
  if (vecAligned) {
    core::WarpAccum<TokT, kCombineVecBytes>(dest, srcPtrs, nullptr, accumNum, nelems);
  } else {
    core::WarpAccum<TokT, 4>(dest, srcPtrs, nullptr, accumNum, nelems);
  }
}

// Ordinary gather: load-first or plain, fixed at compile time per token bucket
// by the tuning table's combine_load_first. The LL gather uses its separate path.
template <EpInterNodeKernelCfg kConfig, typename TokT>
__forceinline__ __device__ void CombineGatherNormal(TokT* dest, TokT** srcPtrs, int accumNum,
                                                    size_t nelems) {
  if constexpr (kConfig.combineLoadFirst) {
    core::WarpAccumLF<TokT, 4, 2>(dest, srcPtrs, nullptr, accumNum, nelems);
  } else {
    core::WarpAccum<TokT, 4>(dest, srcPtrs, nullptr, accumNum, nelems);
  }
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineIntraNodeTyped(EpDispatchCombineArgs& args,
                                                      size_t tokHiddenBytes,
                                                      size_t tokCombXferBytes) {
  DEF_COMMON_VARS;

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes + myNode, 0) * tokCombXferBytes;

  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
        srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                          destLocalTokId * hiddenDim;
        srcWeightsPtr[laneId] = args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    CombineGatherNormal<kConfig, TokT>(
        reinterpret_cast<TokT*>(stagingPtr + tokenId * tokCombXferBytes), srcPtrs,
        config.numExpertPerToken, hiddenDim);
    if (args.weightsBuf) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * tokCombXferBytes + tokHiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineIntraNodeLLTyped(EpDispatchCombineArgs& args,
                                                        size_t tokHiddenBytes,
                                                        size_t tokCombXferBytes) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  int xgmiWarpNum = xgmiBlockNum * warpNum;

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes + myNode, 0) * tokCombXferBytes;

  // Slices are snapped to a whole vector step so the gather below stays on
  // WarpAccum's vector path instead of its scalar tail.
  MultiWarpIter mwIter(xgmiWarpNum, args.curRankNumToken, hiddenDim,
                       CombineVecStep<TokT>(warpSize));
  const bool vecAligned = CombineVecAligned(tokHiddenBytes, tokCombXferBytes);

  for (int i = globalWarpId - blockOffset * warpNum;
       i < (args.curRankNumToken * mwIter.warpsPerItem); i += xgmiWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
        srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                          destLocalTokId * hiddenDim + hiddenDimOffset;
        srcWeightsPtr[laneId] = args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    CombineGather<TokT>(
        reinterpret_cast<TokT*>(stagingPtr + tokenId * tokCombXferBytes) + hiddenDimOffset, srcPtrs,
        config.numExpertPerToken, hiddenDimSize, vecAligned);
    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * tokCombXferBytes + tokHiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

// End of the combine's cross-node barrier: once the last local RDMA warp has moved the epoch, one
// warp per (remote node, QP) posts that QP's mark and waits for the peer's, so the posts go out
// in parallel rather than one after another from a single lane.
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineInterNodeBarrierPosts(EpDispatchCombineArgs& args,
                                                    const ::mori::cco::ccoDevComm& comm,
                                                    uint64_t barrierFlag, uint32_t epoch) {
  DEF_COMMON_VARS;
  constexpr int kQps =
      kConfig.noRdmaAtomics ? EpInterNodeBarrierQps<kConfig>() : kConfig.numQpPerPe;
  const int pairs = (nNodes - 1) * kQps;
  const int stride = args.rdmaBlockNum * warpNum;
  if (laneId != 0 || globalWarpId >= pairs) return;
  auto proxyPeOf = [&](int p) {
    return ((myNode + 1 + p / kQps) % nNodes) * config.gpuPerNode + (myPe % config.gpuPerNode);
  };
  while (__hip_atomic_load(&args.interNodeBlocksBarrier[1], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT) != epoch) {
  }
  // Every pair this warp owns is posted before any is waited on: a peer's mark can sit behind
  // its own wait on a third node, so post-then-wait per pair deadlocks around the ring.
  for (int p = globalWarpId; p < pairs; p += stride) {
    if constexpr (kConfig.noRdmaAtomics) {
      EpInterNodeWriteValue(
          comm, args.reg(args.offCrossDeviceBarrier),
          EpInterNodeBarrierQpSlot<kConfig>(args.rank, p % kQps) * sizeof(uint64_t), barrierFlag,
          proxyPeOf(p), p % kQps);
    } else {
      EpInterNodeAtomicAdd(comm, args.reg(args.offCrossDeviceBarrier), args.rank * sizeof(uint64_t),
                           CombineBarrierMarker(p % kQps, kQps), proxyPeOf(p), p % kQps);
    }
  }
  uint64_t* localBarrierPtr = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
  for (int p = globalWarpId; p < pairs; p += stride) {
    if constexpr (kConfig.noRdmaAtomics) {
      // A peer cannot write round r+1 before this rank leaves round r, so each slot is exactly
      // barrierFlag; == turns a desync into a hang instead of an early, silent pass.
      while (core::AtomicLoadRelaxedSystem(
                 localBarrierPtr + EpInterNodeBarrierQpSlot<kConfig>(proxyPeOf(p), p % kQps)) !=
             barrierFlag) {
      }
    } else if (p % kQps == 0) {
      // One wait per peer, for the sum of every QP's marker.
      while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPeOf(p)) !=
             (barrierFlag * kCombineBarrierMarkerTotal)) {
      }
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineInterNodeTyped(EpDispatchCombineArgs& args,
                                                      size_t tokHiddenBytes,
                                                      size_t tokCombXferBytes,
                                                      const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  int totalBids = 0;
  for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
       bid += args.rdmaBlockNum) {
    totalBids++;
  }

  int processedCount = 0;
  int batchStart = 0;
  while (processedCount < totalBids) {
    uint32_t processedMask = 0;
    int currentBatchSize = std::min(totalBids - processedCount, 32);

    while (processedMask !=
           ((currentBatchSize == 32) ? 0xFFFFFFFF : ((1u << currentBatchSize) - 1))) {
      int bidIdx = 0;
      for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
           bid += args.rdmaBlockNum) {
        if (bidIdx < batchStart) {
          bidIdx++;
          continue;
        }
        if (bidIdx >= batchStart + currentBatchSize) break;

        int relativeIdx = bidIdx - batchStart;
        if (!((processedMask >> relativeIdx) & 1)) {
          int k = bid / (numRecvBlock * (nNodes - 1));
          int i = (bid / numRecvBlock) % (nNodes - 1);
          int node = (myNode + 1 + i) % nNodes;

          uint64_t thisChunkTokenNum = 0;
          int startTokenIdx = k * warpSize;

          if (laneId == 0) {
            thisChunkTokenNum = chunkFlag[node * maxChunkNum + k];
            if (thisChunkTokenNum == 0) {
              uint64_t nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
              if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
                thisChunkTokenNum = 1;
              }
            }
          }
          thisChunkTokenNum = __shfl(thisChunkTokenNum, 0);

          if (thisChunkTokenNum > 0) {
            thisChunkTokenNum -= 1;
            int endTokenIdx = startTokenIdx + thisChunkTokenNum;

            for (int j = startTokenIdx + (bid % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
                 j += numRecvBlock * warpNum) {
              int tokIdx = SendBufSlotOffset(config, node, j);

              if (laneId < config.numExpertPerToken) {
                srcPtrs[laneId] = nullptr;
                srcWeightsPtr[laneId] = nullptr;
                index_t destTokId =
                    args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId];
                index_t destPe = PeFromFlatTokenIndex(config, destTokId);
                index_t destNode = destPe / config.gpuPerNode;
                if (destNode == myNode) {
                  index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
                  srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                                    destLocalTokId * hiddenDim;
                  srcWeightsPtr[laneId] =
                      args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
                      destLocalTokId * config.numExpertPerToken;
                }
                args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId] = 0;
              }

              CombineGatherNormal<kConfig, TokT>(
                  reinterpret_cast<TokT*>(stagingPtr + tokIdx * tokCombXferBytes), srcPtrs,
                  config.numExpertPerToken, hiddenDim);

              if (args.weightsBuf) {
                core::WarpAccum<float, 4>(
                    reinterpret_cast<float*>(stagingPtr + tokIdx * tokCombXferBytes +
                                             tokHiddenBytes),
                    srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
              }
            }

            // Join the producers before the leader's release publishes their stores.
            ::mori::cco::ccoCoopWarp{}.sync();
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
            index_t finished = 0;
            if (laneId == 0)
              finished = atomicAdd(&args.interNodeChunkFlagCombine[node * maxChunkNum + k], 1);
            finished = __shfl(finished, 0);
            if ((finished + 1) >= (numRecvBlock * warpNum)) {
              if (laneId == 0) {
                // Acquire all producers through the completion-counter RMW chain.
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
                core::AtomicStoreSeqCstSystem(
                    args.reg(args.offChunkFlag)->template GetAs<uint64_t*>() + node * maxChunkNum +
                        k,
                    uint64_t{0});
                core::AtomicStoreRelaxedSystem(
                    args.interNodeChunkFlagCombine + node * maxChunkNum + k, index_t{0});
              }
              int proxyPe = node * config.gpuPerNode + (myPe % config.gpuPerNode);
              int qpId = k % config.numQpPerPe;
              EpInterNodePut(
                  comm, args.reg(args.offStaging),
                  SendBufSlotOffset(config, myNode + nNodes, startTokenIdx) * tokCombXferBytes,
                  args.reg(args.offStaging),
                  SendBufSlotOffset(config, node, startTokenIdx) * tokCombXferBytes,
                  thisChunkTokenNum * tokCombXferBytes, proxyPe, qpId);
            }
          }
          processedMask |= (1u << relativeIdx);
        }
        bidIdx++;
      }
    }
    processedCount += currentBatchSize;
    batchStart += currentBatchSize;
  }

  // Ensure all prior writes (in particular zeroing the chunk-flag region) are visible
  // to other nodes before participating in the cross-device barrier, so a remote node
  // never observes a non-zero flag that is subsequently overwritten with zero
  __threadfence_system();

  int finishedWarp = 0;
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag = __shfl(barrierFlag, 0);
  // Each dispatch bumps the flag, so it also numbers this combine's local release.
  const uint32_t epoch = static_cast<uint32_t>(barrierFlag);

  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    // Neither dispatch nor this combine reads the count when every slot is flagged; consume it
    // before clearing, or a late arrival leaks into the next round.
    if ((laneId < nNodes) && (laneId != myNode)) {
      while (core::AtomicLoadRelaxedSystem(
                 args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>() + laneId) == 0) {
      }
    }
    if (laneId < nNodes) {
      core::AtomicStoreSeqCstSystem(
          args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
    if (laneId == 0) {
      args.interNodeBlocksBarrier[0] = 0;
      __threadfence_system();  // the count resets above, before any barrier post
      __hip_atomic_store(&args.interNodeBlocksBarrier[1], epoch, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  CombineInterNodeBarrierPosts<kConfig, T>(args, comm, barrierFlag, epoch);
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineInterNodeLLTyped(EpDispatchCombineArgs& args,
                                                        size_t tokHiddenBytes,
                                                        size_t tokCombXferBytes,
                                                        const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  int rdmaWarpNum = args.rdmaBlockNum * warpNum;
  for (int n = 0; n < (nNodes - 1); n++) {
    int node = (myNode + n + 1) % nNodes;
    // Full chunk flags can arrive before the final node count. Zero means
    // unpublished, whereas one is the completed empty-node sentinel.
    uint64_t nodeCount = 0;
    if (laneId == 0) {
      do {
        nodeCount = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
      } while (nodeCount == 0);
    }
    nodeCount = __shfl(nodeCount, 0) - 1;
    if (nodeCount == 0) continue;

    // One vector step per part, so no part falls to the gather's scalar tail and 3584 splits
    // 1024/1024/1024/512 rather than 2048/1536 and an empty part.
    const size_t vecStep = CombineVecStep<TokT>(warpSize);
    const int warpsPerToken = static_cast<int>(core::CeilDiv(hiddenDim, vecStep));
    const size_t hiddenDimPerWarp = vecStep;
    const bool vecAligned = CombineVecAligned(tokHiddenBytes, tokCombXferBytes);

    // Each node's items start on a different warp, so a small batch's nodes are gathered by
    // different warps at once rather than by the same warps one node after another.
    const int first = (globalWarpId + n * (rdmaWarpNum - rdmaWarpNum / (nNodes - 1))) % rdmaWarpNum;
    for (int i = first; i < (nodeCount * warpsPerToken); i += rdmaWarpNum) {
      int tokenId = i / warpsPerToken;
      int k = tokenId / warpSize;
      int startTokenIdx = k * warpSize;
      uint64_t thisChunkTokenNum = chunkFlag[node * maxChunkNum + k];
      thisChunkTokenNum -= (thisChunkTokenNum > 0) ? 1 : 0;
      // Padding (past the chunk's tokens, or the flag already cleared) neither gathers nor
      // counts: every real item reads the flag before its own increment, so all of them
      // compute the same completion target.
      if ((tokenId - startTokenIdx) >= thisChunkTokenNum) continue;
      int inTokenPartId = i % warpsPerToken;
      size_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
      size_t hiddenDimSize = std::min(hiddenDim - hiddenDimOffset, hiddenDimPerWarp);

      int globalTokenId = SendBufSlotOffset(config, node, tokenId);
      if (laneId < config.numExpertPerToken) {
        srcPtrs[laneId] = nullptr;
        srcWeightsPtr[laneId] = nullptr;
        index_t destTokId =
            args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + laneId];
        index_t destPe = PeFromFlatTokenIndex(config, destTokId);
        index_t destNode = destPe / config.gpuPerNode;
        if (destNode == myNode) {
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                            destLocalTokId * hiddenDim + hiddenDimOffset;
          srcWeightsPtr[laneId] = args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
                                  destLocalTokId * config.numExpertPerToken;
        }
      }
      CombineGather<TokT>(
          reinterpret_cast<TokT*>(stagingPtr + globalTokenId * tokCombXferBytes) + hiddenDimOffset,
          srcPtrs, config.numExpertPerToken, hiddenDimSize, vecAligned);
      if (args.weightsBuf && (inTokenPartId == warpsPerToken - 1)) {
        core::WarpAccum<float, 4>(
            reinterpret_cast<float*>(stagingPtr + globalTokenId * tokCombXferBytes +
                                     tokHiddenBytes),
            srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
      }

      // Join the producers before the leader's release publishes their stores.
      ::mori::cco::ccoCoopWarp{}.sync();
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      // A chunk goes out as one piece per active QP, each of at least kCombinePieceMinTokens
      // tokens, and each piece posts once its own items are done: a single post per chunk ran
      // a whole node's tokens through one QP at that QP's rate.
      const int chunkTokens = static_cast<int>(thisChunkTokenNum);
      const int pieceTokens =
          max(kCombinePieceMinTokens, core::CeilDiv(chunkTokens, config.numQpPerPe));
      const int piece = (tokenId - startTokenIdx) / pieceTokens;
      const int pieceStart = startTokenIdx + piece * pieceTokens;
      const int pieceCount = min(pieceTokens, startTokenIdx + chunkTokens - pieceStart);
      index_t* pieceDone =
          args.interNodeChunkFlagCombine + (node * maxChunkNum + k) * config.numQpPerPe + piece;
      index_t finished = 0;
      if (laneId == 0) finished = atomicAdd(pieceDone, 1);
      finished = __shfl(finished, 0);
      if ((finished + 1) >= (pieceCount * warpsPerToken)) {
        // Acquire all producers through the completion-counter RMW chain.
        if (laneId == 0) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        int proxyPe = node * config.gpuPerNode + (myPe % config.gpuPerNode);
        int qpId = (k + piece) % config.numQpPerPe;
        EpInterNodePut(comm, args.reg(args.offStaging),
                       SendBufSlotOffset(config, myNode + nNodes, pieceStart) * tokCombXferBytes,
                       args.reg(args.offStaging),
                       SendBufSlotOffset(config, node, pieceStart) * tokCombXferBytes,
                       pieceCount * tokCombXferBytes, proxyPe, qpId);
        if (laneId == 0) core::AtomicStoreRelaxedSystem(pieceDone, index_t{0});
      }
    }
  }

  __threadfence_system();
  int finishedWarp = 0;
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(&args.interNodeBlocksBarrier[0], 1);
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag = __shfl(barrierFlag, 0);
  // Each dispatch bumps the flag, so it also numbers this combine's local release.
  const uint32_t epoch = static_cast<uint32_t>(barrierFlag);

  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    // Every item has read its chunk's flag: clear them all here, before the barrier posts let
    // the next dispatch write them again.
    for (int n = 0; n < (nNodes - 1); n++) {
      const int node = (myNode + n + 1) % nNodes;
      const int chunks =
          static_cast<int>((core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]) - 1) / warpSize);
      for (int c = laneId; c < chunks; c += warpSize) {
        core::AtomicStoreRelaxedSystem(chunkFlag + node * maxChunkNum + c, uint64_t{0});
      }
    }
    if (laneId < nNodes) {
      core::AtomicStoreSeqCstSystem(
          args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
    if (laneId == 0) {
      args.interNodeBlocksBarrier[0] = 0;
      __threadfence_system();  // the resets above, before any barrier post
      __hip_atomic_store(&args.interNodeBlocksBarrier[1], epoch, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  CombineInterNodeBarrierPosts<kConfig, T>(args, comm, barrierFlag, epoch);
}

}  // namespace combine_impl

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineIntraNode(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  combine_impl::CombineIntraNodeTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineIntraNodeLL(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  if (args.curRankNumToken == 0) return;
  combine_impl::CombineIntraNodeLLTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineInterNode(EpDispatchCombineArgs& args,
                                        const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  combine_impl::CombineInterNodeTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes, comm);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineInterNodeLL(EpDispatchCombineArgs& args,
                                          const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  combine_impl::CombineInterNodeLLTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes, comm);
}
}  // namespace internode

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineInterNodeV2_body(EpDispatchCombineArgs args,
                                          const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  if (blockId < args.rdmaBlockNum) {
    internode::CombineInterNode<kConfig, T>(args, comm);
  } else {
    internode::CombineIntraNode<kConfig, T>(args);
  }
}

namespace combine_all_impl {

template <EpInterNodeKernelCfg kConfig, typename T>
__forceinline__ __device__ void EpCombineAllGeneric(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes, 0) * combXferBytes;

  // Aligned rows take 16 B loads on slices of whole vector rounds, every node's load in flight;
  // unsnapped slices left most of each part to the gather's serial scalar tail.
  const bool vecAligned = internode::combine_impl::CombineVecAligned(hiddenBytes, combXferBytes);
  MultiWarpIter mwIter(
      globalWarpNum, args.curRankNumToken, hiddenDim,
      vecAligned ? warpSize * (internode::combine_impl::kCombineVecBytes / sizeof(T)) : 1);

  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    int lanePe = -1, laneNode = -1;
    if (laneId < config.numExpertPerToken) {
      index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + laneId];
      if (laneExpert >= 0) {
        lanePe = laneExpert / config.numExpertPerRank;
        laneNode = lanePe / config.gpuPerNode;
      }
    }

    if (laneId < nNodes) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtrs[laneId] = nullptr;
    }

    for (int n = 0; n < nNodes; n++) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + n];
        uint8_t* base = stagingPtr + SendBufSlotOffset(config, n, mappedId) * combXferBytes;
        srcPtrs[n] = reinterpret_cast<T*>(base) + hiddenDimOffset;
        srcWeightsPtrs[n] = reinterpret_cast<float*>(base + hiddenBytes);
      }
    }
    internode::combine_impl::CombineGather<T>(
        args.reg(args.offCombineOut)->template GetAs<T*>() + tokenId * hiddenDim + hiddenDimOffset,
        srcPtrs, nNodes, hiddenDimSize, vecAligned);
    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(args.reg(args.offCombineOutWeights)->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}

}  // namespace combine_all_impl

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAll_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  if (globalWarpId == 0) {
    if (laneId == 0) args.totalRecvTokenNum[0] = 0;
    if (laneId < nNodes) args.blockFlagCounter[laneId] = 0;
  }
  if (args.curRankNumToken == 0) return;
  combine_all_impl::EpCombineAllGeneric<kConfig, T>(args);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineInterNodeV2LL_body(EpDispatchCombineArgs args,
                                            const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  if (blockId < args.rdmaBlockNum) {
    internode::CombineInterNodeLL<kConfig, T>(args, comm);
  } else {
    internode::CombineIntraNodeLL<kConfig, T>(args);
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineSync_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  internode::CombineSync<kConfig, T>(args);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineSyncBarrier_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  barrierFlag = __shfl(barrierFlag, 0);
  uint64_t* localBarrierPtr = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    core::AtomicStoreRelaxedSystem(
        args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>(destPe) + args.rank,
        barrierFlag);
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + destPe) != barrierFlag) {
    }
  }
}

}  // namespace v2
}  // namespace ops
}  // namespace mori

// ---------------------------------------------------------------------------
// JIT entry points.
//
// The AOT build wraps each body in a `template <typename T> __global__` and lets
// ep_common.hip stamp out one symbol per dtype. A JIT module compiles exactly
// one kernel, so the generated TU invokes one of these macros instead.
//
// The argument is the v1 POD block plus the communicator. Passing the comm by
// value is the whole point of the port: mori-shmem would have needed a device
// global filled by the host after every hipModuleLoad.
// ---------------------------------------------------------------------------

// `kConfig` and `TokT` are not macro arguments. The generated TU defines both
// under those names just above the entry, exactly as ep_spec.cpp emits
// `constexpr EpCfg kCfg` / `using TokT` and then instantiates
// `EpDispatchBody<kCfg, TokT>`. A Cfg could not be a macro argument in any case:
// the commas inside its brace initialiser would be taken as argument separators.

// Kernels that reach the network.
#define MORI_EP_INTERNODE_CCO_ENTRY(entry, body)                                     \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs kernelArgs) { \
    ::mori::ops::v2::body<kConfig, TokT>(kernelArgs.args, kernelArgs.devComm);       \
  }

// Staging, sync and the final reduction: local or intra-node only, so they take
// no communicator. They still take the same argument struct, so the host has one
// launch path for the whole sequence.
#define MORI_EP_INTERNODE_CCO_ENTRY_LOCAL(entry, body)                               \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs kernelArgs) { \
    ::mori::ops::v2::body<kConfig, TokT>(kernelArgs.args);                           \
  }
