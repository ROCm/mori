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
// test: cco gda doorbell modes (ccoGdaDoorbellDeferred / CallerOrdered handles)
// and doorbell coalescing.
//
// A Deferred post writes its WQEs and marks them ready without ringing; the next
// ring that reaches them (an Immediate post or signal after them, flush, counter)
// releases them. A CallerOrdered post writes its WQEs only; a CallerOrdered flush
// rings straight up to its snapshot. Each case is one kernel, then a host barrier and a check of
// what landed:
//
//   probe     one warp, one peer: deferred puts leave dbTouchIdx where it was
//             while postIdx moves; flush brings dbTouchIdx up to postIdx.
//   flush     every warp defers its puts to every peer, then flushes.
//   signal    deferred puts released by a plain signal, no flush: the receiver
//             waits for the signal, which the QP orders after the data.
//   mixed     every block posts 16 B thread-scope puts (lanes grouped by QP),
//             even warps defer and odd warps do not -- the contended,
//             coalescing case.
//   overflow  one warp defers two SQs' worth of putValues: the SQ space wait
//             must ring for them itself or it deadlocks.
//   counter   deferred puts drained by counter() instead of flush.
//   get       deferred gets, then flush.
//   async     thread-scope deferred puts, the lanes of a warp spread over the
//             peers, released by each lane's flushAsync and drained by its wait.
//   ctx       thread-scope puts with the context alternating across lanes, so
//             lanes of a warp post on two QPs of the same peer.
//   split     one warp per peer: 3/4 of the lanes release their deferred puts,
//             the rest release theirs later, then all wait at thread scope on
//             one QP with two different targets.
//   sigthr    every lane signals each peer at thread scope, so many lanes of a
//             wave post to one QP at once.
//   ordered   CallerOrdered puts (some with a window signal) and putValues from
//             several blocks to every peer, each bumping a per-peer count once it
//             returns; one warp per peer waits for the count, then flush(peer).
//   ordthr    one warp per peer posts CallerOrdered puts, then all its lanes
//             flushAsync that one QP at thread scope and wait.
//   ordcnt    CallerOrdered puts with a window signal from one block, released
//             by counter() after a block barrier.
//   ordprobe  probe on a CallerOrdered handle.
//   ordall    ordcnt released by flush() over every peer.
//
// The cases run on a FULL devcomm with the WORLD team and, when the ranks span
// nodes, again on a RAIL devcomm with the GDA team (peer = node index).
// MORI_CCO_SKIP_GDA_FULL (runners without intranode cross-rail RDMA) skips the
// FULL pass.

#include <unistd.h>

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 512ULL * 1024 * 1024;
static constexpr int kChunks = 256;      // chunks per (src, dst) pair
static constexpr int kChunkElems = 32;   // uint64 per chunk (256 B)
static constexpr int kMixedPieces = 16;  // mixed: 16 B puts
static constexpr int kPieceElems = kChunkElems / kMixedPieces;
// Per-source region of the receive buffer; overflow wraps around it.
static constexpr size_t kRegionElems = 16384;
static_assert(kChunks * kChunkElems <= kRegionElems, "region too small");
static constexpr int kContexts = 2;      // ctx case: lanes alternate between two QPs
static constexpr int kTimeoutSec = 30;   // per case; a doorbell nobody rings never completes
static constexpr int kSigThreads = 512;  // sigthr grid: signals each source adds per peer
static constexpr int kValueEvery = 16;   // ordered: every 16th chunk goes as putValues
static constexpr int kSignalEvery = 4;   // ordered: every 4th chunk's put carries a signal
// ordered: WQEs per QP between two flushes (a signal takes up to 2), which must fit in the SQ.
static constexpr int kOrderedWqes = kChunks - kChunks / kValueEvery +
                                    kChunks / kValueEvery * kChunkElems +
                                    2 * (kChunks / kSignalEvery);
static_assert(kValueEvery % kSignalEvery == 0, "a putValue chunk never carries a signal");

enum Case : int {
  kProbe,
  kFlush,
  kSignal,
  kMixed,
  kOverflow,
  kCounter,
  kGet,
  kAsync,
  kCtx,
  kSplit,
  kSigThr,
  kOrdered,
  kOrdThr,
  kOrdCnt,
  kOrdProbe,
  kOrdAll,
  kNumCases
};
static const char* kCaseNames[] = {"probe",  "flush",  "signal",   "mixed", "overflow", "counter",
                                   "get",    "async",  "ctx",      "split", "sigthr",   "ordered",
                                   "ordthr", "ordcnt", "ordprobe", "ordall"};
// sigthr / ordcnt: each source adds into this element of its region on the receiver.
static constexpr size_t kSigElem = kRegionElems - 1;

__host__ __device__ inline uint64_t Value(int src, int dst, int chunk, int i) {
  return (static_cast<uint64_t>(src) << 48) | (static_cast<uint64_t>(dst) << 40) |
         (static_cast<uint64_t>(chunk) << 16) | static_cast<uint64_t>(i);
}

template <mori::core::ProviderType P>
using DeferredGda = ccoGda<P, ccoGdaDoorbellDeferred>;
template <mori::core::ProviderType P>
using OrderedGda = ccoGda<P, ccoGdaDoorbellCallerOrdered>;

// Ops take team peers; buffers and signal ids are laid out by world rank.
// WORLD: every rank. GDA on RAIL: one rank per node, the same local rank.
template <ccoTeamMode TM>
__host__ __device__ inline int TeamSize(int worldSize, int lsaSize) {
  return TM == CCO_TEAM_GDA ? worldSize / lsaSize : worldSize;
}
template <ccoTeamMode TM>
__host__ __device__ inline int TeamRank(int rank, int lsaSize) {
  return TM == CCO_TEAM_GDA ? rank / lsaSize : rank;
}
template <ccoTeamMode TM>
__host__ __device__ inline int ToWorld(int teamPeer, int lsaSize, int lsaRank) {
  return TM == CCO_TEAM_GDA ? teamPeer * lsaSize + lsaRank : teamPeer;
}

template <mori::core::ProviderType P>
__device__ inline mori::core::RdmaEndpointDevice* Endpoint(const ccoDevComm& dc, int worldPeer) {
  return &dc.ibgda.endpoints[worldPeer * dc.ibgda.numQpPerPe];  // context 0
}

// probe / ordprobe: out[0..3] = dbTouchIdx before, dbTouchIdx after the puts on
// a Deferred / CallerOrdered handle, postIdx after them, dbTouchIdx after flush.
template <mori::core::ProviderType P, ccoTeamMode TM, typename Gda>
__global__ void ProbeKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc,
                            uint32_t* out) {
  if (threadIdx.x >= warpSize) return;  // one warp, also on wave32
  Gda gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int peer = (TeamRank<TM>(dc.rank, dc.lsaSize) + 1) % n;
  const int meW = dc.rank, peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
  auto* wq = &Endpoint<P>(dc, peerW)->wqHandle;
  const size_t chunkBytes = kChunkElems * sizeof(uint64_t);
  if (threadIdx.x == 0)
    out[0] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  for (int c = 0; c < kChunks; c++) {
    gda.template put<TM, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopWarp>(
        peer, reinterpret_cast<ccoWindow_t>(recvWin),
        (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
        reinterpret_cast<ccoWindow_t>(sendWin),
        (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), chunkBytes, ccoGda_NoSignal{},
        ccoCoopWarp{});
  }
  if (threadIdx.x == 0) {
    out[1] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    out[2] = __hip_atomic_load(&wq->postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  gda.template flush<TM>(peer, ccoCoopWarp{});
  if (threadIdx.x == 0)
    out[3] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// flush / signal / counter / get: warp-scope ops, chunks striped over all warps
// of the grid, every peer.
template <mori::core::ProviderType P, ccoTeamMode TM, int C>
__global__ void ChunkKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  DeferredGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int warpsPerBlock = blockDim.x / warpSize;
  const int gw = blockIdx.x * warpsPerBlock + threadIdx.x / warpSize;
  const int nw = gridDim.x * warpsPerBlock;
  const size_t chunkBytes = kChunkElems * sizeof(uint64_t);
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);

  for (int k = 1; k < n; k++) {
    const int peer = (me + k) % n;
    const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
    for (int c = gw; c < kChunks; c += nw) {
      if constexpr (C == kGet) {
        // Pull peer's chunk for me into my region for peer.
        gda.template get<TM, ccoGdaThreadIndependent, ccoCoopWarp>(
            peer, send, (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), recv,
            (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), chunkBytes, ccoCoopWarp{});
      } else {
        gda.template put<TM, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopWarp>(
            peer, recv, (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), send,
            (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), chunkBytes,
            ccoGda_NoSignal{}, ccoCoopWarp{});
      }
    }
  }

  if constexpr (C == kSignal) {
    // No flush: one signal per (warp, peer) is the only post that rings.
    for (int k = 1; k < n; k++) {
      const int peer = (me + k) % n;
      gda.template signal<TM>(peer, ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(meW)},
                              ccoCoopWarp{});
    }
    // Every warp of every peer signals me once.
    if (blockIdx.x == 0 && threadIdx.x < n && static_cast<int>(threadIdx.x) != me) {
      const int srcW = ToWorld<TM>(threadIdx.x, dc.lsaSize, dc.lsaRank);
      gda.waitSignal(static_cast<ccoGdaSignal_t>(srcW), static_cast<uint64_t>(nw));
    }
    __syncthreads();
    gda.flush(ccoCoopWarp{});  // drain our own CQs before the kernel exits
  } else if constexpr (C == kCounter) {
    gda.counter(ccoGda_CounterInc{0}, ccoCoopWarp{});
  } else {
    gda.flush(ccoCoopWarp{});
  }
}

// mixed: one lane per (peer, 16 B piece) item, thread scope, so a warp's lanes
// are grouped by QP and every block hits each QP. Even warps defer, odd ones ring.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void MixedKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  ccoGda<P> gda{dc, 0};
  DeferredGda<P> dgda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int items = (n - 1) * kChunks * kMixedPieces;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  const bool defer = ((tid / warpSize) & 1) == 0;
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);

  for (int it = tid; it < items; it += nthreads) {
    const int peer = (me + 1 + it % (n - 1)) % n;
    const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
    const int elem = (it / (n - 1)) * kPieceElems;  // piece p of the pair's chunks
    const size_t dstOff = (meW * kRegionElems + elem) * sizeof(uint64_t);
    const size_t srcOff = (peerW * kRegionElems + elem) * sizeof(uint64_t);
    const size_t bytes = kPieceElems * sizeof(uint64_t);
    if (defer) {
      dgda.template put<TM>(peer, recv, dstOff, send, srcOff, bytes);
    } else {
      gda.template put<TM>(peer, recv, dstOff, send, srcOff, bytes);
    }
  }
  __syncthreads();
  gda.flush(ccoCoopWarp{});
}

// overflow: warp w of block 0 owns peer (me + 1 + w); `puts` deferred 8-byte
// putValues to it, wrapping around the region, then one flush. Warp 0 records
// its QP's dbTouchIdx before and after the puts: it must have moved, i.e. the
// SQ space wait rang.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void OverflowKernel(ccoWindowDevice* recvWin, ccoDevComm dc, int puts, uint32_t* out) {
  DeferredGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int w = threadIdx.x / warpSize;
  if (w >= n - 1) return;
  const int peer = (me + 1 + w) % n;
  const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
  auto* wq = &Endpoint<P>(dc, peerW)->wqHandle;
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  if (threadIdx.x == 0)
    out[0] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  for (int i = 0; i < puts; i++) {
    gda.template putValue<TM, ccoGdaThreadIndependent, uint64_t, ccoGda_NoSignal, ccoCoopWarp>(
        peer, recv, (meW * kRegionElems + i % kRegionElems) * sizeof(uint64_t),
        Value(meW, peerW, 0, i), ccoGda_NoSignal{}, ccoCoopWarp{});
  }
  if (threadIdx.x == 0)
    out[1] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  gda.template flush<TM>(peer, ccoCoopWarp{});
}

// async: one lane per (peer, chunk) item and a grid with a lane for each, so
// the lanes of a warp post to different peers. Each lane releases and drains
// its own QP with thread-scope flushAsync + wait.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void AsyncKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  DeferredGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int it = blockIdx.x * blockDim.x + threadIdx.x;
  if (it >= (n - 1) * kChunks) return;
  const int peer = (me + 1 + it % (n - 1)) % n;
  const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
  const int c = it / (n - 1);
  gda.template put<TM>(peer, reinterpret_cast<ccoWindow_t>(recvWin),
                       (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
                       reinterpret_cast<ccoWindow_t>(sendWin),
                       (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
                       kChunkElems * sizeof(uint64_t));
  ccoGdaRequest_t req;
  gda.template flushAsync<TM>(peer, &req, ccoCoopThread{});
  gda.wait(req, ccoCoopThread{});
}

// ctx: one lane per (peer, chunk) item; every n-1 lanes (one per peer) switch
// context, so lanes with the same peer use two QPs. Deferring switches every
// 2(n-1) lanes, so both contexts meet in every post. One flush per context.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void CtxKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int ctx = static_cast<int>((threadIdx.x / (n - 1)) & 1);
  ccoGda<P> gda{dc, ctx};
  DeferredGda<P> dgda{dc, ctx};
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  for (int it = tid; it < (n - 1) * kChunks; it += nthreads) {
    const int peer = (me + 1 + it % (n - 1)) % n;
    const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
    const int c = it / (n - 1);
    const size_t dstOff = (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t);
    const size_t srcOff = (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t);
    const size_t bytes = kChunkElems * sizeof(uint64_t);
    if ((threadIdx.x / (2 * (n - 1))) & 1) {
      dgda.template put<TM>(peer, recv, dstOff, send, srcOff, bytes);
    } else {
      gda.template put<TM>(peer, recv, dstOff, send, srcOff, bytes);
    }
  }
  __syncthreads();
  for (int ctx = 0; ctx < kContexts; ctx++) ccoGda<P>{dc, ctx}.flush(ccoCoopWarp{});
}

// split: warp w owns peer (me + 1 + w). The first 3/4 of its lanes post most
// chunks deferred and flushAsync; after a pause, so their completions are in,
// the rest post one chunk each and flushAsync. Every lane then waits at thread
// scope: the early lanes on the lower target, the late ones on fewer completions
// than there are early lanes.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void SplitKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  DeferredGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int w = threadIdx.x / warpSize, lane = threadIdx.x % warpSize;
  if (w >= n - 1) return;
  const int peer = (me + 1 + w) % n;
  const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
  const int early = warpSize * 3 / 4, late = warpSize - early;
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  auto post = [&](int c) {
    gda.template put<TM>(peer, recv, (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
                         send, (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
                         kChunkElems * sizeof(uint64_t));
  };
  ccoGdaRequest_t req;
  if (lane < early) {
    for (int c = late + lane; c < kChunks; c += early) post(c);
    gda.template flushAsync<TM>(peer, &req, ccoCoopThread{});
  }
  for (int i = 0; i < 16; i++) __builtin_amdgcn_s_sleep(127);
  if (lane >= early) {
    post(lane - early);
    gda.template flushAsync<TM>(peer, &req, ccoCoopThread{});
  }
  gda.wait(req, ccoCoopThread{});
}

// sigthr: every lane adds 1 into its region's kSigElem on each peer at thread
// scope, then the warps drain.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void SigThrKernel(ccoWindowDevice* recvWin, ccoDevComm dc) {
  ccoGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  for (int k = 1; k < n; k++) {
    gda.template signal<TM>(
        (me + k) % n,
        ccoGda_WindowSignalAdd{recv, (meW * kRegionElems + kSigElem) * sizeof(uint64_t), 1},
        ccoCoopThread{});
  }
  gda.flush(ccoCoopWarp{});
}

// ordered: blocks 1.. post every chunk to every peer, every kValueEvery-th as
// thread-scope putValues, and bump posted[k - 1] once each post has returned.
// Warp k - 1 of block 0 waits for all kChunks posts to peer me + k, then flushes.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void OrderedKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc,
                              uint32_t* posted) {
  OrderedGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int lane = threadIdx.x % warpSize;
  if (blockIdx.x == 0) {
    const int w = threadIdx.x / warpSize;
    if (w >= n - 1) return;
    while (__hip_atomic_load(&posted[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < kChunks) {
    }
    gda.template flush<TM>((me + 1 + w) % n, ccoCoopWarp{});
    return;
  }
  const int warpsPerBlock = blockDim.x / warpSize;
  const int gw = (blockIdx.x - 1) * warpsPerBlock + threadIdx.x / warpSize;
  const int nw = (gridDim.x - 1) * warpsPerBlock;
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  for (int k = 1; k < n; k++) {
    const int peer = (me + k) % n;
    const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
    for (int c = gw; c < kChunks; c += nw) {
      const size_t dstOff = (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t);
      if (c % kValueEvery == kValueEvery - 1) {
        if (lane < kChunkElems) {
          gda.template putValue<TM, ccoGdaThreadIndependent, uint64_t>(
              peer, recv, dstOff + lane * sizeof(uint64_t), Value(meW, peerW, c, lane));
        }
      } else if (c % kSignalEvery == 0) {
        gda.template put<TM, ccoGdaThreadIndependent, ccoGda_WindowSignalAdd, ccoCoopWarp>(
            peer, recv, dstOff, send, (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
            kChunkElems * sizeof(uint64_t),
            ccoGda_WindowSignalAdd{recv, (meW * kRegionElems + kSigElem) * sizeof(uint64_t), 1},
            ccoCoopWarp{});
      } else {
        gda.template put<TM, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopWarp>(
            peer, recv, dstOff, send, (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
            kChunkElems * sizeof(uint64_t), ccoGda_NoSignal{}, ccoCoopWarp{});
      }
      if (lane == 0)
        __hip_atomic_fetch_add(&posted[k - 1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// ordthr: warp w posts every chunk to peer (me + 1 + w), then all of its lanes
// flushAsync that QP at thread scope, contending for its doorbell lock, and wait.
template <mori::core::ProviderType P, ccoTeamMode TM>
__global__ void OrdThrKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  OrderedGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int w = threadIdx.x / warpSize;
  if (w >= n - 1) return;
  const int peer = (me + 1 + w) % n;
  const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
  for (int c = 0; c < kChunks; c++) {
    gda.template put<TM, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopWarp>(
        peer, reinterpret_cast<ccoWindow_t>(recvWin),
        (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
        reinterpret_cast<ccoWindow_t>(sendWin),
        (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), kChunkElems * sizeof(uint64_t),
        ccoGda_NoSignal{}, ccoCoopWarp{});
  }
  ccoGdaRequest_t req;
  gda.template flushAsync<TM>(peer, &req, ccoCoopThread{});
  gda.wait(req, ccoCoopThread{});
}

// ordcnt / ordall: the warps of one block post every chunk to every peer, each put
// adding 1 into kSigElem of my region there; after a block barrier warp 0's
// counter() / flush() releases and drains them all.
template <mori::core::ProviderType P, ccoTeamMode TM, bool FlushAll>
__global__ void OrdCntKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  OrderedGda<P> gda{dc, 0};
  const int n = TeamSize<TM>(dc.worldSize, dc.lsaSize);
  const int me = TeamRank<TM>(dc.rank, dc.lsaSize), meW = dc.rank;
  const int w = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  const ccoGda_WindowSignalAdd sig{recv, (meW * kRegionElems + kSigElem) * sizeof(uint64_t), 1};
  for (int k = 1; k < n; k++) {
    const int peer = (me + k) % n;
    const int peerW = ToWorld<TM>(peer, dc.lsaSize, dc.lsaRank);
    for (int c = w; c < kChunks; c += nw) {
      gda.template put<TM, ccoGdaThreadIndependent, ccoGda_WindowSignalAdd, ccoCoopWarp>(
          peer, recv, (meW * kRegionElems + c * kChunkElems) * sizeof(uint64_t), send,
          (peerW * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
          kChunkElems * sizeof(uint64_t), sig, ccoCoopWarp{});
    }
  }
  __syncthreads();
  if (w != 0) return;
  if constexpr (FlushAll) {
    gda.flush(ccoCoopWarp{});
  } else {
    gda.counter(ccoGda_CounterInc{0}, ccoCoopWarp{});
  }
}

// A doorbell nobody rings shows up as a kernel that never finishes: fail it.
static void WaitStream(hipStream_t stream, int rank, const char* team, const char* name) {
  for (int ms = 0; ms < kTimeoutSec * 1000; ms += 10) {
    hipError_t err = hipStreamQuery(stream);
    if (err == hipSuccess) return;
    if (err != hipErrorNotReady) HIP_CHECK(err);
    usleep(10 * 1000);
  }
  printf("[rank %d] %-5s %-8s HANG (no completion in %d s)\n", rank, team, name, kTimeoutSec);
  _exit(1);
}

// srcs: the world ranks that sent to me in this case.
static bool Verify(int c, const char* team, int rank, const std::vector<int>& srcs,
                   const std::vector<uint64_t>& recv, int overflowPuts) {
  int bad = 0;
  for (int src : srcs) {
    const uint64_t* r = recv.data() + src * kRegionElems;
    if (c == kOverflow) {
      // Same QP, so the last put to each slot wins.
      for (int j = 0; j < static_cast<int>(kRegionElems) && j < overflowPuts; j++) {
        const int last = j + (overflowPuts - 1 - j) / kRegionElems * kRegionElems;
        uint64_t want = Value(src, rank, 0, last);
        if (r[j] != want && bad++ < 4)
          fprintf(stderr, "[rank %d] %s/%s: src %d [%d] got %#lx want %#lx\n", rank, team,
                  kCaseNames[c], src, j, r[j], want);
      }
      continue;
    }
    for (int ch = 0; ch < kChunks; ch++) {
      for (int i = 0; i < kChunkElems; i++) {
        // get pulls the peer's data for me; every other case pushes mine to the peer.
        uint64_t want = Value(src, rank, ch, i);
        uint64_t got = r[ch * kChunkElems + i];
        if (got != want && bad++ < 4)
          fprintf(stderr, "[rank %d] %s/%s: src %d chunk %d [%d] got %#lx want %#lx\n", rank, team,
                  kCaseNames[c], src, ch, i, got, want);
      }
    }
  }
  return bad == 0;
}

template <ccoTeamMode TM>
static bool RunCases(ccoComm* comm, const ccoDevComm& devComm, ccoWindowDevice* sw,
                     ccoWindowDevice* rw, void* recvBuf, size_t bufElems, hipStream_t stream,
                     uint32_t* probe, uint32_t* posted) {
  const char* team = TM == CCO_TEAM_GDA ? "gda" : "world";
  const int rank = devComm.rank;
  const int n = TeamSize<TM>(devComm.worldSize, devComm.lsaSize);
  const int me = TeamRank<TM>(rank, devComm.lsaSize);
  std::vector<int> srcs;
  for (int p = 0; p < n; p++)
    if (p != me) srcs.push_back(ToWorld<TM>(p, devComm.lsaSize, devComm.lsaRank));
  const size_t bufBytes = bufElems * sizeof(uint64_t);

  // Two SQs' worth, whatever this provider's depth is.
  mori::core::RdmaEndpointDevice ep;
  const int nextW = ToWorld<TM>((me + 1) % n, devComm.lsaSize, devComm.lsaRank);
  HIP_CHECK(hipMemcpy(&ep, devComm.ibgda.endpoints + nextW * devComm.ibgda.numQpPerPe, sizeof(ep),
                      hipMemcpyDeviceToHost));
  const int overflowPuts = 2 * static_cast<int>(ep.wqHandle.sqWqeNum) + 5;
  const int asyncBlocks = ((n - 1) * kChunks + 255) / 256;
  const int orderedThreads = std::max(256, 64 * (n - 1));  // block 0: a warp per peer

  bool allOk = true;
  for (int c = 0; c < kNumCases; c++) {
    if (c >= kOrdered && kOrderedWqes >= static_cast<int>(ep.wqHandle.sqWqeNum)) {
      // CallerOrdered posts between two flushes must fit in the SQ.
      printf("[rank %d] %-5s %-8s SKIP (SQ %u)\n", rank, team, kCaseNames[c], ep.wqHandle.sqWqeNum);
      continue;
    }
    HIP_CHECK(hipMemset(recvBuf, 0xff, bufBytes));
    if (c == kOrdered) HIP_CHECK(hipMemset(posted, 0, (n - 1) * sizeof(uint32_t)));
    if (c == kSigThr || c == kOrdered || c == kOrdCnt || c == kOrdAll)
      for (int src : srcs)
        HIP_CHECK(hipMemset(static_cast<uint64_t*>(recvBuf) + src * kRegionElems + kSigElem, 0,
                            sizeof(uint64_t)));
    HIP_CHECK(hipDeviceSynchronize());
    ccoBarrierAll(comm);
    switch (c) {
      case kProbe:
        CCO_GDA_DISPATCH(ProbeKernel<P, TM, DeferredGda<P>>
                         <<<1, 64, 0, stream>>>(sw, rw, devComm, probe));
        break;
      case kFlush:
        CCO_GDA_DISPATCH(ChunkKernel<P, TM, kFlush><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kSignal:
        // Signals accumulate across runs only within this case; 1 block keeps
        // the expected count (warps per grid) simple.
        CCO_GDA_DISPATCH(ChunkKernel<P, TM, kSignal><<<1, 512, 0, stream>>>(sw, rw, devComm));
        break;
      case kMixed:
        CCO_GDA_DISPATCH(MixedKernel<P, TM><<<16, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kOverflow:
        CCO_GDA_DISPATCH(OverflowKernel<P, TM>
                         <<<1, 64 * (n - 1), 0, stream>>>(rw, devComm, overflowPuts, probe));
        break;
      case kCounter:
        CCO_GDA_DISPATCH(ChunkKernel<P, TM, kCounter><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kGet:
        CCO_GDA_DISPATCH(ChunkKernel<P, TM, kGet><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kAsync:
        CCO_GDA_DISPATCH(AsyncKernel<P, TM><<<asyncBlocks, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kCtx:
        CCO_GDA_DISPATCH(CtxKernel<P, TM><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kSplit:
        CCO_GDA_DISPATCH(SplitKernel<P, TM><<<1, 64 * (n - 1), 0, stream>>>(sw, rw, devComm));
        break;
      case kSigThr:
        CCO_GDA_DISPATCH(SigThrKernel<P, TM><<<2, kSigThreads / 2, 0, stream>>>(rw, devComm));
        break;
      case kOrdered:
        CCO_GDA_DISPATCH(OrderedKernel<P, TM>
                         <<<5, orderedThreads, 0, stream>>>(sw, rw, devComm, posted));
        break;
      case kOrdThr:
        CCO_GDA_DISPATCH(OrdThrKernel<P, TM><<<1, 64 * (n - 1), 0, stream>>>(sw, rw, devComm));
        break;
      case kOrdCnt:
        CCO_GDA_DISPATCH(OrdCntKernel<P, TM, false><<<1, 512, 0, stream>>>(sw, rw, devComm));
        break;
      case kOrdProbe:
        CCO_GDA_DISPATCH(ProbeKernel<P, TM, OrderedGda<P>>
                         <<<1, 64, 0, stream>>>(sw, rw, devComm, probe));
        break;
      case kOrdAll:
        CCO_GDA_DISPATCH(OrdCntKernel<P, TM, true><<<1, 512, 0, stream>>>(sw, rw, devComm));
        break;
    }
    HIP_CHECK(hipGetLastError());
    WaitStream(stream, rank, team, kCaseNames[c]);
    ccoBarrierAll(comm);

    std::vector<uint64_t> hostRecv(bufElems);
    HIP_CHECK(hipMemcpy(hostRecv.data(), recvBuf, bufBytes, hipMemcpyDeviceToHost));
    bool ok = true;
    uint32_t p[4];
    if (c == kProbe || c == kOrdProbe || c == kOverflow)
      HIP_CHECK(hipMemcpy(p, probe, sizeof(p), hipMemcpyDeviceToHost));
    if (c == kProbe || c == kOrdProbe) {
      // Only the probed pair is filled; check the doorbell indices, then the
      // data the previous team member sent me.
      if (p[1] != p[0] || p[3] != p[2] || p[2] - p[0] != static_cast<uint32_t>(kChunks)) {
        fprintf(stderr,
                "[rank %d] %s/%s: dbTouchIdx %u -> %u after the puts (want unchanged), "
                "postIdx %u, dbTouchIdx after flush %u (want == postIdx)\n",
                rank, team, kCaseNames[c], p[0], p[1], p[2], p[3]);
        ok = false;
      }
      const int src = ToWorld<TM>((me - 1 + n) % n, devComm.lsaSize, devComm.lsaRank);
      for (int ch = 0; ch < kChunks && ok; ch++)
        for (int i = 0; i < kChunkElems && ok; i++)
          if (hostRecv[src * kRegionElems + ch * kChunkElems + i] != Value(src, rank, ch, i)) {
            fprintf(stderr, "[rank %d] %s/%s: data from %d chunk %d [%d] wrong\n", rank, team,
                    kCaseNames[c], src, ch, i);
            ok = false;
          }
    } else {
      if (c == kOverflow && p[1] == p[0]) {
        fprintf(stderr,
                "[rank %d] %s/overflow: dbTouchIdx stayed at %u over %d deferred puts (SQ %u): "
                "the SQ space wait never rang\n",
                rank, team, p[0], overflowPuts, ep.wqHandle.sqWqeNum);
        ok = false;
      }
      if (c == kSigThr) {
        for (int src : srcs) {
          const uint64_t got = hostRecv[src * kRegionElems + kSigElem];
          if (got != kSigThreads) {
            fprintf(stderr, "[rank %d] %s/sigthr: src %d added %lu, want %d\n", rank, team, src,
                    got, kSigThreads);
            ok = false;
          }
        }
      } else {
        ok = Verify(c, team, rank, srcs, hostRecv, overflowPuts) && ok;
      }
      if (c == kOrdered || c == kOrdCnt || c == kOrdAll) {
        const uint64_t want = c == kOrdered ? kChunks / kSignalEvery : kChunks;
        for (int src : srcs) {
          const uint64_t got = hostRecv[src * kRegionElems + kSigElem];
          if (got != want) {
            fprintf(stderr, "[rank %d] %s/%s: src %d signalled %lu, want %lu\n", rank, team,
                    kCaseNames[c], src, got, want);
            ok = false;
          }
        }
      }
    }
    printf("[rank %d] %-5s %-8s %s\n", rank, team, kCaseNames[c], ok ? "PASS" : "FAIL");
    allOk = allOk && ok;
  }
  return allOk;
}

static bool CreateDevComm(ccoComm* comm, ccoGdaConnectionType type, int nranks,
                          ccoDevComm* devComm) {
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = type;
  reqs.gdaContextCount = kContexts;
  reqs.gdaSignalCount = nranks;  // signal ids are world ranks
  reqs.gdaCounterCount = 1;
  return ccoDevCommCreate(comm, &reqs, devComm) == 0;
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // the fork harness's children leave via _exit
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);
  const bool skipFull = getenv("MORI_CCO_SKIP_GDA_FULL") != nullptr;

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) return 1;

  const size_t bufElems = kRegionElems * nranks;
  const size_t bufBytes = bufElems * sizeof(uint64_t);
  void *sendBuf = nullptr, *recvBuf = nullptr;
  if (ccoMemAlloc(comm, bufBytes, &sendBuf) != 0 || ccoMemAlloc(comm, bufBytes, &recvBuf) != 0)
    return 1;

  // send[dst region][chunk][i] = Value(me, dst, chunk, i). For get, a peer pulls
  // from my send region for it, which holds the same value: Value(me, peer, ...).
  std::vector<uint64_t> hostSend(bufElems, 0);
  for (int dst = 0; dst < nranks; dst++)
    for (int ch = 0; ch < kChunks; ch++)
      for (int i = 0; i < kChunkElems; i++)
        hostSend[dst * kRegionElems + ch * kChunkElems + i] = Value(rank, dst, ch, i);
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufBytes, hipMemcpyHostToDevice));

  ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (ccoWindowRegister(comm, sendBuf, bufBytes, &sendWin) != 0 ||
      ccoWindowRegister(comm, recvBuf, bufBytes, &recvWin) != 0)
    return 1;

  // FULL unless the runner cannot route it; RAIL whenever the ranks span nodes
  // (on one node it collapses to NONE).
  ccoDevComm full{}, rail{};
  bool haveFull = false, haveRail = false;
  if (!skipFull) {
    if (!CreateDevComm(comm, CCO_GDA_CONNECTION_FULL, nranks, &full)) return 1;
    if (full.gdaConnType != CCO_GDA_CONNECTION_FULL) {
      fprintf(stderr, "[rank %d] FULL devcomm got gdaConnType %d\n", rank, full.gdaConnType);
      return 1;
    }
    haveFull = true;
  }
  if (skipFull || nranks / full.lsaSize > 1) {
    if (!CreateDevComm(comm, CCO_GDA_CONNECTION_RAIL, nranks, &rail)) return 1;
    haveRail = rail.gdaConnType == CCO_GDA_CONNECTION_RAIL;
    if (!haveRail) ccoDevCommDestroy(comm, &rail);
  }
  printf("[rank %d] world=%s gda=%s qpsPerPe=%d\n", rank, haveFull ? "FULL" : "skipped",
         haveRail ? "RAIL" : "skipped (one node)",
         haveFull ? full.ibgda.numQpPerPe : (haveRail ? rail.ibgda.numQpPerPe : 0));

  uint32_t *probe = nullptr, *posted = nullptr;
  HIP_CHECK(hipMalloc(&probe, 4 * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&posted, nranks * sizeof(uint32_t)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  auto* sw = reinterpret_cast<ccoWindowDevice*>(sendWin);
  auto* rw = reinterpret_cast<ccoWindowDevice*>(recvWin);

  bool allOk = true;
  if (haveFull)
    allOk =
        RunCases<CCO_TEAM_WORLD>(comm, full, sw, rw, recvBuf, bufElems, stream, probe, posted) &&
        allOk;
  if (haveRail)
    allOk = RunCases<CCO_TEAM_GDA>(comm, rail, sw, rw, recvBuf, bufElems, stream, probe, posted) &&
            allOk;

  ccoBarrierAll(comm);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(probe));
  HIP_CHECK(hipFree(posted));
  if (haveRail) ccoDevCommDestroy(comm, &rail);
  if (haveFull) ccoDevCommDestroy(comm, &full);
  ccoWindowDeregister(comm, sendWin);
  ccoWindowDeregister(comm, recvWin);
  ccoMemFree(comm, sendBuf);
  ccoMemFree(comm, recvBuf);
  ccoCommDestroy(comm);
  if (!haveFull && !haveRail)
    printf("[rank %d] SKIPPED (MORI_CCO_SKIP_GDA_FULL on one node)\n", rank);
  else
    printf("[rank %d] %s\n", rank, allOk ? "PASSED" : "FAILED");
  return allOk ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA defer doorbell", "/tmp/cco_gda_defer_doorbell_uid", 0);
}
