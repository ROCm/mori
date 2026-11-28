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

// test: ccoProxy put / putValue / signal / flush, positive and negative cases.
//
// Block u of a put kernel is unit u and sends a distinct chunk to every peer;
// signal slot Slot(src, u) on the receiver counts it. The host verifies the
// receive window after a barrier.
//
// MORI_CCO_PROXY_TEST_CASES=<list> runs only those cases of kCases (default
// "all"; "atomic_free" selects the cases that post no NIC atomic).
// MORI_CCO_PROXY_TEST_NEGATIVE=all or ids ("n1,n3") adds n1-n9, then
// put_sig_after rechecks dc. A name or id that matches nothing fails the run,
// and so does a selection that runs no case. Give every rank the same values.
//
// Launch: `mpirun -np N test_proxy_put`, `test_proxy_put [N]` (fork), or
// `--gen-uid` then `--rank r --world N --uid-file f` per node.

#include <infiniband/verbs.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 512ULL * 1024 * 1024;
static constexpr int kSpscQueues = 4;
static constexpr int kMpscQueues = 2;
static constexpr int kProxyThreads = 2;
static constexpr int kMaxUnits = 8;
static constexpr size_t kChunk = 64 * 1024;
static constexpr size_t kLargeChunk = 4 * 1024 * 1024;
static constexpr int kSignalRounds = 16;
static constexpr int kStressBlocks = 8;
static constexpr int kStressWarps = 4;  // wavefronts per block
static constexpr int kStressRounds = 8;
static constexpr size_t kStressChunk = 16 * 1024;
static constexpr int kSmallCapacity = 16;  // ring lines: a put + signal takes 2
static constexpr int kPieces = 64;         // puts per chunk in the small-ring cases
static constexpr int kSqRepeats = 2048;    // puts per peer in sq_full (SQ depth: 512)
static constexpr int kQpsPerPeer = 3;      // dcQps, per (thread, peer): one per queue of a thread
static constexpr uint32_t kClobber = 0xdeadbeefu;
// Long enough for any proxy answer, short enough not to hang.
static constexpr uint64_t kTimeoutCycles = 40ull * 1000 * 1000 * 1000;

enum PutMode {
  kPutSignal,         // put carrying SignalInc
  kPutThenSignal,     // put, then signal()
  kPutWindowSignal,   // put carrying WindowSignalAdd
  kPutFlushCount,     // put, flush (flushDone advances by one), signal()
  kPutFlushClobber,   // put, flush, overwrite the source, signal()
  kPutKernelPayload,  // the block writes the source, then put carrying SignalInc
  kPutRepeat,         // kSqRepeats puts of the same chunk, the last carrying SignalInc
  kPutPieces          // kPieces puts per chunk, the last carrying SignalInc
};

__host__ __device__ inline uint32_t Pattern(uint32_t salt, int src, int dst, int u, size_t i) {
  return salt * 0x9E3779B1u + static_cast<uint32_t>(src) * 0x85EBCA77u +
         static_cast<uint32_t>(dst) * 0xC2B2AE3Du + static_cast<uint32_t>(u) * 0x27D4EB2Fu +
         static_cast<uint32_t>(i);
}

__host__ __device__ inline ccoGdaSignal_t Slot(int src, int unit) {
  return static_cast<ccoGdaSignal_t>(src * kMaxUnits + unit);
}

// Lane 0 waits until the caller-window flag reaches `least`. Returns, on every
// lane, 0 = reached, 1 = timed out after kTimeoutCycles, 2 = the DevComm failed.
template <typename P>
__device__ inline int WaitFlag(P& proxy, const uint64_t* flag, uint64_t least) {
  int result = 0;
  if (__lane_id() == 0) {
    impl::TimeoutClock clk;
    clk.start();
    for (uint32_t polls = 1;
         __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < least; polls++) {
      if (polls % 256 == 0 && proxy.failed()) {
        result = 2;
        break;
      }
      if (clk.expired(kTimeoutCycles)) {
        result = proxy.failed() ? 2 : 1;
        break;
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // the data the flag covers
  return __builtin_amdgcn_readfirstlane(result);
}

__host__ __device__ inline const char* WaitResultName(int result) {
  return result == 1 ? "timed out" : "DevComm failed";
}

// A queue's satisfied-FLUSH count.
__device__ inline uint64_t FlushDone(const uint64_t* flushDone) {
  return __hip_atomic_load(flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Flags `err` unless the count moved from `before` by exactly `want`; `what`
// names the case and the kind of `idx` (e.g. "n3: unit").
__device__ inline void CheckFlushDone(const uint64_t* flushDone, uint64_t before, uint64_t want,
                                      const char* what, int idx, int me, int* err) {
  const uint64_t after = FlushDone(flushDone);
  if (after == before + want) return;
  printf("[rank %d] %s %d flushDone %llu -> %llu, want +%llu\n", me, what, idx,
         (unsigned long long)before, (unsigned long long)after, (unsigned long long)want);
  atomicExch(err, 1);
}

// Overwrites words first, first + stride, ... below `words` of `src`.
__device__ inline void ClobberChunk(uint32_t* src, size_t words, size_t first, size_t stride) {
  for (size_t i = first; i < words; i += stride) src[i] = kClobber;
}

// Block u = unit u on queue u (MPSC: modulo the pool). Sends one chunk to every peer,
// then waits for every sender's chunk on this unit (signal slot or flag word).
template <typename Q, typename Coop, int Mode>
__global__ void ProxyPutKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                               ccoWindow_t flagWin, uint32_t* sendLocal, uint64_t* flagLocal,
                               uint32_t salt, size_t chunk, int nUnits, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  // Linear thread id: the block may be 2-D (block_2d_*).
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nThreads = blockDim.x * blockDim.y * blockDim.z;
  ccoProxy<Q> proxy{dc, u};
  auto srcOff = [&](int p) { return (static_cast<size_t>(p) * nUnits + u) * chunk; };
  const size_t dstOff = (static_cast<size_t>(me) * nUnits + u) * chunk;
  if constexpr (Mode == kPutKernelPayload) {
    // Every thread of the block writes the chunks; put() publishes them.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
      for (size_t i = tid; i < chunk / sizeof(uint32_t); i += nThreads) {
        src[i] = Pattern(salt, me, p, u, i);
      }
    }
  }
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    if constexpr (Mode == kPutSignal || Mode == kPutKernelPayload) {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_SignalInc{Slot(me, u)},
                Coop{});
    } else if constexpr (Mode == kPutWindowSignal) {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk,
                ccoGda_WindowSignalAdd{flagWin, Slot(me, u) * sizeof(uint64_t), 1}, Coop{});
    } else if constexpr (Mode == kPutRepeat) {
      for (int k = 0; k < kSqRepeats - 1; k++) {
        proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_NoSignal{}, Coop{});
      }
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_SignalInc{Slot(me, u)},
                Coop{});
    } else if constexpr (Mode == kPutPieces) {
      const size_t piece = chunk / kPieces;
      for (int k = 0; k < kPieces - 1; k++) {
        proxy.put(p, recvWin, dstOff + k * piece, sendWin, srcOff(p) + k * piece, piece,
                  ccoGda_NoSignal{}, Coop{});
      }
      const size_t last = (kPieces - 1) * piece;
      proxy.put(p, recvWin, dstOff + last, sendWin, srcOff(p) + last, chunk - last,
                ccoGda_SignalInc{Slot(me, u)}, Coop{});
    } else {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_NoSignal{}, Coop{});
      if constexpr (Mode == kPutThenSignal) {
        proxy.signal(p, ccoGda_SignalAdd{Slot(me, u), 1}, Coop{});
      }
    }
  }
  if constexpr (Mode == kPutFlushCount) {
    uint64_t before = 0;
    if constexpr (ccoProxy<Q>::kSpsc) before = FlushDone(proxy.flushDone);
    proxy.flush(Coop{});
    // This wavefront owns the queue: its FLUSH is the only one in flight.
    if constexpr (ccoProxy<Q>::kSpsc) {
      if (tid == 0) CheckFlushDone(proxy.flushDone, before, 1, "flush_count: unit", u, me, err);
    }
  } else if constexpr (Mode == kPutFlushClobber) {
    proxy.flush(Coop{});
    // Every put of this unit completed: its sources may be reused right away.
    for (int p = 0; p < n; p++) {
      if (p != me) {
        ClobberChunk(sendLocal + srcOff(p) / sizeof(uint32_t), chunk / sizeof(uint32_t), tid,
                     nThreads);
      }
    }
    __threadfence_system();
    __syncthreads();
  }
  if constexpr (Mode == kPutFlushCount || Mode == kPutFlushClobber) {
    for (int p = 0; p < n; p++) {
      if (p != me) proxy.signal(p, ccoGda_SignalInc{Slot(me, u)}, Coop{});
    }
  }
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    if constexpr (Mode == kPutWindowSignal) {
      if (tid < 64) {  // wavefront 0
        const int waited = WaitFlag(proxy, &flagLocal[Slot(s, u)], 1);
        if (waited != 0) {
          if (tid == 0) {
            printf("[rank %d] win_signal: src=%d unit=%d: %s\n", me, s, u, WaitResultName(waited));
            atomicExch(err, 1);
          }
          return;
        }
      }
    } else {
      proxy.waitSignal(Slot(s, u), 1, Coop{});  // one waiter per group
    }
  }
}

// ccoCoopLaneMasked cases: one wavefront per block. Lane L owns kLaneSlice bytes
// at L * kLaneSlice of the unit's chunk to every peer; if ThreadActive(L) it puts
// the first ThreadBytes(L) of them, else the rest of its slice stays untouched.
static constexpr size_t kLaneSlice = kChunk / 64;
static constexpr int kThreadCapacity = 128;  // ring lines: 51 lanes' put + signal take 102
static constexpr int kThreadRounds = 16;     // put rounds on kThreadCapacity-line rings
static constexpr int kContendedRounds = 16;  // tm_branch_contended's rounds of the same puts

__host__ __device__ inline bool ThreadActive(int lane) { return lane % 5 != 0; }  // 0, 5, .. idle

__host__ __device__ inline size_t ThreadBytes(int lane) {  // 4 B .. kLaneSlice, mixed
  return 4 * (1 + (static_cast<size_t>(lane) * 37) % (kLaneSlice / 4));
}

// Sum of (L + 1) over the active lanes: the SignalAdd total of one unit.
__host__ __device__ inline uint64_t ThreadSignalSum() {
  uint64_t s = 0;
  for (int l = 0; l < 64; l++) s += ThreadActive(l) ? l + 1 : 0;
  return s;
}

// Per-(sender, unit, lane) signal slots, after the per-(sender, unit) slots.
__host__ __device__ inline ccoGdaSignal_t LaneSlot(int nranks, int src, int unit, int lane) {
  return static_cast<ccoGdaSignal_t>(nranks * kMaxUnits + (src * kMaxUnits + unit) * 64 + lane);
}

enum ThreadCase {
  kThreadPutSignal,       // `rounds` x put carrying SignalAdd{LaneSlot(me, u, L), L + 1}
  kThreadPutGroupSignal,  // `rounds` x ccoCoopLaneMaskedGroup put carrying SignalInc{Slot(me, u)}:
                          // one signal per call and peer group
  kThreadPutLaneSignal,   // put, then signal(SignalInc{LaneSlot(me, u, L)})
  kThreadPutFlush         // put, flush, overwrite the source, signal(SignalAdd{Slot(me, u), L + 1})
};

// Lane L's k-th peer: every lane sends to every peer once, and with more than
// two ranks the lanes of one call target different peers.
__device__ inline int ThreadPeer(int me, int n, int k, int lane) {
  return (me + 1 + (k + lane) % (n - 1)) % n;
}

// Calls k of rank s with an active lane targeting me: one per-group signal each.
__device__ inline uint64_t GroupSignals(int s, int n, int me) {
  uint64_t want = 0;
  for (int k = 0; k < n - 1; k++) {
    bool any = false;
    for (int l = 0; l < 64; l++) any = any || (ThreadActive(l) && ThreadPeer(s, n, k, l) == me);
    want += any ? 1 : 0;
  }
  return want;
}

// Checks this lane's written words of chunk slot (s, v), nSlots slots per sender.
__device__ inline void CheckLaneSlice(const uint32_t* recvLocal, uint32_t salt, int s, int me,
                                      int v, int nSlots, size_t chunk, int lane, const char* what,
                                      int* err) {
  const size_t w0 = ((static_cast<size_t>(s) * nSlots + v) * chunk + lane * kLaneSlice) / 4;
  for (size_t i = 0; i < ThreadBytes(lane) / 4; i++) {
    const uint32_t got =
        __hip_atomic_load(recvLocal + w0 + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const uint32_t want = Pattern(salt, s, me, v, lane * kLaneSlice / 4 + i);
    if (got != want) {
      printf("[rank %d] %s: src=%d slot=%d lane %d word %zu: got %08x want %08x\n", me, what, s, v,
             lane, i, got, want);
      atomicExch(err, 1);
      return;
    }
  }
}

// Idle lanes pass a valid put (kThreadPutSignal) or garbage (other modes): neither
// may be issued or read. Round r of unit u uses chunk slot u * rounds + r.
template <typename Q, int Mode>
__global__ void ProxyThreadKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                  uint32_t* sendLocal, const uint32_t* recvLocal, uint32_t salt,
                                  size_t chunk, int nUnits, int rounds, int* err) {
  const int u = blockIdx.x;
  const int lane = __lane_id();
  const int me = dc.rank;
  const int n = dc.worldSize;
  const bool active = ThreadActive(lane);
  ccoProxy<Q> proxy{dc, u};
  const int nSlices = nUnits * rounds;
  auto off = [&](int p, int r) {  // lane's slice of chunk slot (p, u * rounds + r)
    return (static_cast<size_t>(p) * nSlices + u * rounds + r) * chunk + lane * kLaneSlice;
  };
  const bool garbage = Mode != kThreadPutSignal && !active;
  for (int r = 0; r < rounds; r++) {
    for (int k = 0; k < n - 1; k++) {
      const int p = ThreadPeer(me, n, k, lane);
      const int peer = garbage ? (1 << 20) : p;
      const ccoWindow_t dst = garbage ? nullptr : recvWin;
      const ccoWindow_t src = garbage ? nullptr : sendWin;
      const size_t bytes = garbage ? ~size_t{0} : active ? ThreadBytes(lane) : kLaneSlice;
      if constexpr (Mode == kThreadPutSignal) {
        proxy.put(peer, dst, off(me, r), src, off(p, r), bytes,
                  ccoGda_SignalAdd{LaneSlot(n, me, u, lane), static_cast<uint64_t>(lane + 1)},
                  ccoCoopLaneMasked{active});
      } else if constexpr (Mode == kThreadPutGroupSignal) {
        proxy.put(peer, dst, off(me, r), src, off(p, r), bytes, ccoGda_SignalInc{Slot(me, u)},
                  ccoCoopLaneMaskedGroup{active});
      } else {
        proxy.put(peer, dst, off(me, r), src, off(p, r), bytes, ccoGda_NoSignal{},
                  ccoCoopLaneMasked{active});
      }
    }
  }
  if constexpr (Mode == kThreadPutFlush) {
    uint64_t before = 0;
    if constexpr (ccoProxy<Q>::kSpsc) before = FlushDone(proxy.flushDone);
    proxy.flush();  // one flush for the wavefront
    if constexpr (ccoProxy<Q>::kSpsc) {
      if (lane == 0) CheckFlushDone(proxy.flushDone, before, 1, "thread flush: unit", u, me, err);
    }
    // Every lane's puts completed: each lane overwrites its own sources.
    for (int p = 0; p < n; p++) {
      if (p != me) ClobberChunk(sendLocal + off(p, 0) / sizeof(uint32_t), kLaneSlice / 4, 0, 1);
    }
    __threadfence_system();
  }
  for (int k = 0; k < n - 1; k++) {
    const int peer = garbage ? (1 << 20) : ThreadPeer(me, n, k, lane);
    if constexpr (Mode == kThreadPutLaneSignal) {
      proxy.signal(peer, ccoGda_SignalInc{LaneSlot(n, me, u, lane)}, ccoCoopLaneMasked{active});
    } else if constexpr (Mode == kThreadPutFlush) {
      proxy.signal(peer, ccoGda_SignalAdd{Slot(me, u), static_cast<uint64_t>(lane + 1)},
                   ccoCoopLaneMasked{active});
    }
  }
  // Consume exactly the active lanes' signals, then check their slices.
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    if constexpr (Mode == kThreadPutFlush) {
      if (lane == 0) proxy.waitSignal(Slot(s, u), ThreadSignalSum());
    } else if constexpr (Mode == kThreadPutGroupSignal) {
      if (lane == 0)
        proxy.waitSignal(Slot(s, u), static_cast<uint64_t>(rounds) * GroupSignals(s, n, me));
    } else if (active) {
      proxy.waitSignal(LaneSlot(n, s, u, lane),
                       Mode == kThreadPutSignal ? static_cast<uint64_t>(rounds) * (lane + 1) : 1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    if (active) {
      for (int r = 0; r < rounds; r++) {
        CheckLaneSlice(recvLocal, salt, s, me, u * rounds + r, nSlices, chunk, lane, "early check",
                       err);
      }
    }
  }
  // Complete every WR before exit, for the host checks after the barrier.
  proxy.flush();
}

// ccoCoopThread cases (MPSC; blocks share queues): the lane-masked layout, but
// only active lanes call, from divergent code. Lane L's k-th put goes to
// ThreadPeer(me, n, k, L) unless noted.
enum ThreadModeCase {
  kTmBranch,           // odd / even active lanes put in the two arms of a branch;
                       // the converged wavefront flushes, then sets Slot(me, u)
  kTmBranchContended,  // as kTmBranch, but the arms make the same call with other
                       // arguments (the compiler may merge their tails), kContendedRounds
                       // times, every unit on MPSC queue 0 of kThreadCapacity-line rings
  kTmSingle,           // one active lane at a time puts; then as kTmBranch
  kTmGroupSignal,      // puts carry SignalInc{Slot(me, u)}: per call, one signal per
                       // peer its lanes target (ccoGda's per-group signal)
  kTmLaneSignal,       // puts, then signal(SignalInc{LaneSlot(me, u, L)}): per lane
  kTmAggregate,        // call k: every active lane puts to ThreadPeer(me, n, k, 0)
                       // with ccoGdaThreadAggregate and SignalSet{Slot(me, u), 1}
  kTmLaneAggregate     // as kTmAggregate, from converged code with ccoCoopLaneMaskedGroup
};

template <int Mode>
__global__ void ProxyThreadModeKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                      const uint32_t* recvLocal, uint32_t salt, size_t chunk,
                                      int nUnits, int zero, int* err) {
  const int u = blockIdx.x;
  const int lane = __lane_id();
  const int me = dc.rank;
  const int n = dc.worldSize;
  const bool active = ThreadActive(lane);
  ccoProxy<ccoProxyMpsc> proxy{dc, Mode == kTmBranchContended ? 0 : u};
  auto off = [&](int p) {  // this lane's slice of chunk slot (p, u)
    return (static_cast<size_t>(p) * nUnits + u) * chunk + lane * kLaneSlice;
  };
  constexpr bool kAgg = Mode == kTmAggregate || Mode == kTmLaneAggregate;
  auto peerOf = [&](int k) { return ThreadPeer(me, n, k, kAgg ? 0 : lane); };
  auto putTo = [&](int k, auto action) {
    const int p = peerOf(k);
    if constexpr (Mode == kTmAggregate) {
      proxy.template put<CCO_TEAM_WORLD, ccoGdaThreadAggregate>(
          p, recvWin, off(me), sendWin, off(p), ThreadBytes(lane), action, ccoCoopThread{});
    } else if constexpr (Mode == kTmLaneAggregate) {
      proxy.template put<CCO_TEAM_WORLD, ccoGdaThreadAggregate>(p, recvWin, off(me), sendWin,
                                                                off(p), ThreadBytes(lane), action,
                                                                ccoCoopLaneMaskedGroup{active});
    } else {
      proxy.put(p, recvWin, off(me), sendWin, off(p), ThreadBytes(lane), action, ccoCoopThread{});
    }
  };
  if constexpr (Mode == kTmBranch) {
    if (active) {
      if (lane & 1) {
        for (int k = 0; k < n - 1; k++) putTo(k, ccoGda_NoSignal{});
      } else {
        for (int k = n - 2; k >= 0; k--) putTo(k, ccoGda_NoSignal{});  // differs: no arm merging
      }
    }
  } else if constexpr (Mode == kTmBranchContended) {
    // A push whose stores the compiler sinks past the join hangs: both arms
    // claim and wait for credit before either writes its lines.
    if (active) {
      for (int r = 0; r < kContendedRounds; r++) {  // the same puts every round
        for (int k = 0; k < n - 1; k++) {
          if (lane & 1) {
            putTo(k, ccoGda_NoSignal{});
          } else {
            putTo(k + zero, ccoGda_NoSignal{});  // zero is 0: only the arguments differ
          }
        }
      }
    }
  } else if constexpr (Mode == kTmSingle) {
    for (int l = 0; l < 64; l++) {
      if (lane == l && active) {
        for (int k = 0; k < n - 1; k++) putTo(k, ccoGda_NoSignal{});
      }
    }
  } else if constexpr (Mode == kTmGroupSignal) {
    if (active) {
      for (int k = 0; k < n - 1; k++) putTo(k, ccoGda_SignalInc{Slot(me, u)});
    }
  } else if constexpr (Mode == kTmLaneSignal) {
    if (active) {
      for (int k = 0; k < n - 1; k++) {
        putTo(k, ccoGda_NoSignal{});
        proxy.signal(peerOf(k), ccoGda_SignalInc{LaneSlot(n, me, u, lane)}, ccoCoopThread{});
      }
    }
  } else if constexpr (Mode == kTmLaneAggregate) {
    for (int k = 0; k < n - 1; k++) putTo(k, ccoGda_SignalSet{Slot(me, u), 1});  // every lane calls
  } else {
    if (active) {
      for (int k = 0; k < n - 1; k++) putTo(k, ccoGda_SignalSet{Slot(me, u), 1});
    }
  }
  if constexpr (Mode == kTmBranch || Mode == kTmBranchContended || Mode == kTmSingle) {
    proxy.flush(ccoCoopWarp{});  // covers the divergent puts
    for (int p = 0; p < n; p++) {
      if (p != me) proxy.signal(p, ccoGda_SignalSet{Slot(me, u), 1}, ccoCoopWarp{});
    }
  }
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    if constexpr (Mode == kTmLaneSignal) {
      if (active) proxy.waitSignal(LaneSlot(n, s, u, lane), 1);  // one per lane
    } else {
      uint64_t want = 1;
      if constexpr (Mode == kTmGroupSignal) want = GroupSignals(s, n, me);
      proxy.waitSignal(Slot(s, u), want, ccoCoopWarp{});
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    if (active) CheckLaneSlice(recvLocal, salt, s, me, u, nUnits, chunk, lane, "thread mode", err);
  }
  proxy.flush(ccoCoopWarp{});  // complete every work request before the kernel ends
}

// putValue: unit u sends kValueSlots values to every peer, slot k of
// ((me * nUnits + u) * kValueSlots + k) * 8 holding ValueBytes(k) bytes; slots
// below kValueSlots / 2 carry no signal, the others signal Slot(me, u):
// SignalInc, or (kSet, atomic-free) SignalSet of 1, 2, ... in slot order.
// ccoCoopWarp sends slot by slot to each peer in turn; ccoCoopLaneMasked sends each slot to every
// peer in one call, lane j to the j-th next rank. After the signals, each receiver checks every
// slot of every sender: its low bytes hold the value, the rest keep the 0xff fill.
static constexpr int kValueSlots = 8;

__host__ __device__ inline int ValueBytes(int k) { return 1 << (k % 4); }  // 1, 2, 4, 8

__host__ __device__ inline uint64_t ValueOf(uint32_t salt, int src, int dst, int u, int k) {
  return (static_cast<uint64_t>(Pattern(salt, src, dst, u, k)) << 32) |
         Pattern(salt, src, dst, u, k + kValueSlots);
}

template <typename P, typename RemoteAction, typename Coop>
__device__ inline void PutValueSized(P& proxy, int peer, ccoWindow_t win, size_t off, uint64_t v,
                                     int bytes, RemoteAction ra, Coop coop) {
  switch (bytes) {
    case 1:
      proxy.putValue(peer, win, off, static_cast<uint8_t>(v), ra, coop);
      break;
    case 2:
      proxy.putValue(peer, win, off, static_cast<uint16_t>(v), ra, coop);
      break;
    case 4:
      proxy.putValue(peer, win, off, static_cast<uint32_t>(v), ra, coop);
      break;
    default:
      proxy.putValue(peer, win, off, v, ra, coop);
      break;
  }
}

template <bool kSet>
__device__ inline auto ValueSignal(int me, int u, int k) {
  if constexpr (kSet) {
    return ccoGda_SignalSet{Slot(me, u), static_cast<uint64_t>(k - kValueSlots / 2 + 1)};
  } else {
    return ccoGda_SignalInc{Slot(me, u)};
  }
}

// Unit u's slots from sender s (one thread, after the signal wait): values,
// untouched high bytes, and the signal slot fully consumed.
template <typename P>
__device__ inline void CheckValues(const ccoDevComm& dc, P& proxy, const uint64_t* recvLocal,
                                   uint32_t salt, int s, int u, int nUnits, int* err) {
  const int me = dc.rank;
  for (int k = 0; k < kValueSlots; k++) {
    const size_t w = (static_cast<size_t>(s) * nUnits + u) * kValueSlots + k;
    const uint64_t got =
        __hip_atomic_load(recvLocal + w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const uint64_t mask =
        ValueBytes(k) == 8 ? ~uint64_t{0} : (uint64_t{1} << (8 * ValueBytes(k))) - 1;
    const uint64_t want = (ValueOf(salt, s, me, u, k) & mask) | ~mask;
    if (got != want) {
      printf("[rank %d] putValue: src=%d unit=%d slot %d (%d B): got %016llx want %016llx\n", me, s,
             u, k, ValueBytes(k), (unsigned long long)got, (unsigned long long)want);
      atomicExch(err, 1);
    }
  }
  const uint64_t left = proxy.readSignal(Slot(s, u));
  if (left != 0) {
    printf("[rank %d] putValue: src=%d unit=%d signal %llu past the expected value\n", me, s, u,
           (unsigned long long)left);
    atomicExch(err, 1);
  }
}

template <typename Q, typename Coop, bool kSet>
__global__ void ProxyValueKernel(ccoDevComm dc, ccoWindow_t recvWin, const uint64_t* recvLocal,
                                 uint32_t salt, int nUnits, int* err) {
  constexpr bool kLaneMasked =
      std::is_same_v<Coop, ccoCoopLaneMasked> || std::is_same_v<Coop, ccoCoopLaneMaskedGroup>;
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<Q> proxy{dc, u};
  auto slotOff = [&](int src, int k) {
    return ((static_cast<size_t>(src) * nUnits + u) * kValueSlots + k) * sizeof(uint64_t);
  };
  auto send = [&](int p, int k, auto coop) {
    const uint64_t v = ValueOf(salt, me, p, u, k);
    if (k < kValueSlots / 2) {
      PutValueSized(proxy, p, recvWin, slotOff(me, k), v, ValueBytes(k), ccoGda_NoSignal{}, coop);
    } else {
      PutValueSized(proxy, p, recvWin, slotOff(me, k), v, ValueBytes(k),
                    ValueSignal<kSet>(me, u, k), coop);
    }
  };
  if constexpr (kLaneMasked) {
    for (int base = 0; base < n - 1; base += 64) {
      const int j = base + __lane_id();
      const bool active = j < n - 1;
      const int p = active ? (me + 1 + j) % n : me;  // an idle lane's peer is not read
      for (int k = 0; k < kValueSlots; k++) send(p, k, Coop{active});
    }
  } else if constexpr (std::is_same_v<Coop, ccoCoopThread>) {
    // Lane j writes to the j-th next rank from a divergent branch: one lane per
    // peer group, so every value carries its own signal.
    for (int base = 0; base < n - 1; base += 64) {
      const int j = base + __lane_id();
      if (j < n - 1) {
        for (int k = 0; k < kValueSlots; k++) send((me + 1 + j) % n, k, ccoCoopThread{});
      }
    }
  } else {
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      for (int k = 0; k < kValueSlots; k++) send(p, k, Coop{});
    }
  }
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    proxy.waitSignal(Slot(s, u), kValueSlots / 2, ccoCoopWarp{});  // acquires on lane 0
    if (threadIdx.x == 0) CheckValues(dc, proxy, recvLocal, salt, s, u, nUnits, err);
  }
}

// SignalSet: unit u sends its chunk to every peer as kSetRounds pieces, piece
// r followed on the same queue by a set of Slot(me, u) to r + 1: carried by the
// put (kSetSlot), a separate signal() (kSetSignal), or a WindowSignalSet into
// the flag window (kSetWindow). The receiver waits for each value in turn and
// checks that piece right away; at the end the slot holds exactly kSetRounds.
static constexpr int kSetRounds = 8;

enum SetMode { kSetSlot, kSetSignal, kSetWindow };

template <typename Q, int Mode>
__global__ void ProxySetKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                               ccoWindow_t flagWin, const uint32_t* recvLocal,
                               const uint64_t* flagLocal, uint32_t salt, size_t chunk, int nUnits,
                               int* err) {
  const int u = blockIdx.x;
  const int lane = __lane_id();
  const int me = dc.rank;
  const int n = dc.worldSize;
  const size_t piece = chunk / kSetRounds;
  ccoProxy<Q> proxy{dc, u};
  for (int r = 0; r < kSetRounds; r++) {
    const uint64_t value = r + 1;
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      const size_t src = (static_cast<size_t>(p) * nUnits + u) * chunk + r * piece;
      const size_t dst = (static_cast<size_t>(me) * nUnits + u) * chunk + r * piece;
      if constexpr (Mode == kSetSlot) {
        proxy.put(p, recvWin, dst, sendWin, src, piece, ccoGda_SignalSet{Slot(me, u), value},
                  ccoCoopWarp{});
      } else if constexpr (Mode == kSetSignal) {
        proxy.put(p, recvWin, dst, sendWin, src, piece, ccoGda_NoSignal{}, ccoCoopWarp{});
        proxy.signal(p, ccoGda_SignalSet{Slot(me, u), value}, ccoCoopWarp{});
      } else {
        proxy.put(p, recvWin, dst, sendWin, src, piece,
                  ccoGda_WindowSignalSet{flagWin, Slot(me, u) * sizeof(uint64_t), value},
                  ccoCoopWarp{});
      }
    }
  }
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    for (int r = 0; r < kSetRounds; r++) {
      int waited;  // 0 = arrived, 1 = timed out, 2 = DevComm failed
      if constexpr (Mode == kSetWindow) {
        waited = WaitFlag(proxy, &flagLocal[Slot(s, u)], static_cast<uint64_t>(r + 1));
      } else {
        // Consumed r so far: the slot is >= r + 1.
        waited = proxy.waitSignal(Slot(s, u), 1, ccoCoopWarp{}, kTimeoutCycles);
      }
      if (waited != 0) {
        if (lane == 0) {
          printf("[rank %d] signal set: src=%d unit=%d round %d: %s\n", me, s, u, r,
                 WaitResultName(waited));
          atomicExch(err, 1);
        }
        return;
      }
      const size_t w0 = ((static_cast<size_t>(s) * nUnits + u) * chunk + r * piece) / 4;
      for (size_t i = lane; i < piece / 4; i += 64) {
        const size_t word = r * piece / 4 + i;
        const uint32_t got = recvLocal[w0 + i];
        if (got != Pattern(salt, s, me, u, word)) {
          printf("[rank %d] signal set: src=%d unit=%d round %d word %zu: got %08x\n", me, s, u, r,
                 word, got);
          atomicExch(err, 1);
          break;
        }
      }
    }
    const uint64_t left = Mode == kSetWindow
                              ? __hip_atomic_load(&flagLocal[Slot(s, u)], __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_SYSTEM) -
                                    kSetRounds
                              : proxy.readSignal(Slot(s, u));
    if (lane == 0 && left != 0) {
      printf("[rank %d] signal set: src=%d unit=%d slot is %llu past the last value\n", me, s, u,
             (unsigned long long)left);
      atomicExch(err, 1);
    }
  }
}

// Every signal slot must read 0: every signal sent was consumed by a wait.
__global__ void ProxyCheckSignalsKernel(ccoDevComm dc, int nSlots, int* err) {
  ccoProxy<> proxy{dc, 0};  // no push: reads the slots' signal state only
  for (int i = threadIdx.x; i < nSlots; i += blockDim.x) {
    const uint64_t v = proxy.readSignal(static_cast<ccoGdaSignal_t>(i));
    if (v != 0) {
      printf("[rank %d] signal slot %d: %llu left over\n", dc.rank, i, (unsigned long long)v);
      atomicExch(err, 1);
    }
  }
}

// The handle's GDA team is the proxy's (comm.proxy.connType), not the GDA one.
__global__ void ProxyTeamKernel(ccoDevComm dc, int wantRank, int wantRanks, int* err) {
  ccoProxy<> proxy{dc, 0};  // no push
  if (threadIdx.x == 0 && (proxy.rank != wantRank || proxy.nRanks != wantRanks)) {
    printf("[rank %d] conn_team: proxy team %d/%d, want %d/%d\n", dc.rank, proxy.rank, proxy.nRanks,
           wantRank, wantRanks);
    atomicExch(err, 1);
  }
}

// All wavefronts share MPSC queue 0; each puts a unit per round, flushes in most
// rounds and then overwrites every source since its last flush. Contended
// FLUSHes may land out of ticket order.
__global__ void ProxyStressKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                  uint32_t* sendLocal, ccoGdaSignal_t slot, int* err) {
  const int lane = __lane_id();
  const int waveSize = __builtin_amdgcn_wavefrontsize();
  const int w = blockIdx.x * (blockDim.x / waveSize) + threadIdx.x / waveSize;
  const int nWaves = gridDim.x * (blockDim.x / waveSize);
  const int nUnits = nWaves * kStressRounds;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<ccoProxyMpsc> proxy{dc, 0};
  auto off = [&](int rank, int u) {
    return (static_cast<size_t>(rank) * nUnits + u) * kStressChunk;
  };
  int clean = 0;  // first round whose sources are not yet overwritten
  for (int r = 0; r < kStressRounds; r++) {
    const int u = w * kStressRounds + r;
    for (int p = 0; p < n; p++) {
      if (p != me) {
        proxy.put(p, recvWin, off(me, u), sendWin, off(p, u), kStressChunk, ccoGda_NoSignal{},
                  ccoCoopWarp{});
      }
    }
    if (r == kStressRounds - 1 || (w + r) % 3 != 2) {
      proxy.flush(ccoCoopWarp{});
      for (int k = w * kStressRounds + clean; k <= u; k++) {
        for (int p = 0; p < n; p++) {
          if (p != me) {
            ClobberChunk(sendLocal + off(p, k) / sizeof(uint32_t), kStressChunk / sizeof(uint32_t),
                         lane, waveSize);
          }
        }
      }
      __threadfence_system();
      clean = r + 1;
    }
  }
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.signal(p, ccoGda_SignalAdd{slot, 1}, ccoCoopWarp{});
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    proxy.waitSignal(slot, static_cast<uint64_t>(nWaves) * (n - 1));
    if (proxy.readSignal(slot) != 0) {
      printf("[rank %d] mpsc_stress: extra signals\n", me);
      atomicExch(err, 1);
    }
  }
}

// Every SPSC queue sends kSignalRounds SignalAdd{u+1} to `slot` on every peer,
// then block 0 consumes the expected total and checks nothing is left over.
__global__ void ProxySignalKernel(ccoDevComm dc, ccoGdaSignal_t slot, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<> proxy{dc, u};
  for (int k = 0; k < kSignalRounds; k++) {
    for (int p = 0; p < n; p++) {
      if (p != me) {
        proxy.signal(p, ccoGda_SignalAdd{slot, static_cast<uint64_t>(u + 1)}, ccoCoopWarp{});
      }
    }
  }
  if (u == 0 && threadIdx.x == 0) {
    const uint64_t perPeer = static_cast<uint64_t>(kSignalRounds) * gridDim.x * (gridDim.x + 1) / 2;
    proxy.waitSignal(slot, perPeer * (n - 1));
    const uint64_t rest = proxy.readSignal(slot);
    if (rest != 0) {
      printf("[rank %d] signal_only: %llu left after consuming the expected total\n", me,
             (unsigned long long)rest);
      atomicExch(err, 1);
    }
  }
}

// One block of two wavefronts, unit 0 on SPSC queue 0, handed from wavefront 0
// to wavefront 1 (the release() / __syncthreads() handoff, MORI-CCO-GUIDE.md §10
// "SPSC queues").
__global__ void ProxyHandoffKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                   size_t chunk, int* err) {
  __shared__ uint64_t before;
  const int wave = threadIdx.x / __builtin_amdgcn_wavefrontsize();
  const int me = dc.rank;
  const int n = dc.worldSize;
  const size_t half = chunk / 2;
  uint64_t* flushDone = dc.proxy.flushDone;  // SPSC queue 0
  if (wave == 0) {
    if (threadIdx.x == 0) before = FlushDone(flushDone);
    ccoProxy<> proxy{dc, 0};
    for (int p = 0; p < n; p++) {
      if (p != me) {
        proxy.put(p, recvWin, me * chunk, sendWin, p * chunk, half, ccoGda_NoSignal{},
                  ccoCoopWarp{});
      }
    }
    proxy.flush();
    proxy.release();
  }
  __syncthreads();
  if (wave == 1) {
    ccoProxy<> proxy{dc, 0};
    for (int p = 0; p < n; p++) {
      if (p != me) {
        proxy.put(p, recvWin, me * chunk + half, sendWin, p * chunk + half, chunk - half,
                  ccoGda_SignalInc{Slot(me, 0)}, ccoCoopWarp{});
      }
    }
    proxy.flush();
    if (__lane_id() == 0) {
      CheckFlushDone(flushDone, before, 2, "spsc_handoff: wave", wave, me, err);
      for (int s = 0; s < n; s++) {
        if (s != me) proxy.waitSignal(Slot(s, 0), 1);
      }
    }
  }
}

// One block of kBlockFlushWaves wavefronts; wavefront w is unit w on SPSC
// queue w. Each puts its chunk to every peer (ccoCoopWarp, own handle), then
// the block calls flush(ccoCoopBlock) once: every wavefront's queue must be
// flushed (flushDone +1 each), so the block may overwrite all the sources
// before each wavefront signals its peers.
static constexpr int kBlockFlushWaves = 4;
static_assert(kBlockFlushWaves <= kSpscQueues, "a queue per wavefront");

__global__ void ProxyBlockFlushKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                      uint32_t* sendLocal, size_t chunk, int* err) {
  const int w = threadIdx.x / 64;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<> proxy{dc, w};
  auto srcOff = [&](int p, int u) {
    return (static_cast<size_t>(p) * kBlockFlushWaves + u) * chunk;
  };
  const uint64_t before = FlushDone(proxy.flushDone);
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    proxy.put(p, recvWin, srcOff(me, w), sendWin, srcOff(p, w), chunk, ccoGda_NoSignal{},
              ccoCoopWarp{});
  }
  proxy.flush(ccoCoopBlock{});
  if (__lane_id() == 0) CheckFlushDone(proxy.flushDone, before, 1, "block_flush: wave", w, me, err);
  // Every wavefront's puts completed: overwrite all of the block's sources.
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    for (int u = 0; u < kBlockFlushWaves; u++) {
      ClobberChunk(sendLocal + srcOff(p, u) / sizeof(uint32_t), chunk / sizeof(uint32_t),
                   threadIdx.x, blockDim.x);
    }
  }
  __threadfence_system();
  __syncthreads();
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.signal(p, ccoGda_SignalInc{Slot(me, w)}, ccoCoopWarp{});
  }
  for (int s = 0; s < n; s++) {
    if (s != me) proxy.waitSignal(Slot(s, w), 1, ccoCoopWarp{});
  }
}

// n1 / n2: one wavefront puts `bytes` to `peer`, then flushes twice with a
// timeout, then waits with a timeout on slot 0, which nobody signals.
// res = {first flush, second flush, failed(), waitSignal}.
__global__ void ProxyFailKernel(ccoDevComm dc, int peer, ccoWindow_t dstWin, size_t dstOffset,
                                ccoWindow_t srcWin, size_t bytes, int* res) {
  ccoProxy<> proxy{dc, 0};
  proxy.put(peer, dstWin, dstOffset, srcWin, 0, bytes, ccoGda_NoSignal{}, ccoCoopWarp{});
  const int first = proxy.flush(ccoCoopWarp{}, kTimeoutCycles);
  const int second = proxy.flush(ccoCoopWarp{}, kTimeoutCycles);
  const int waited = proxy.waitSignal(0, 1, ccoCoopWarp{}, kTimeoutCycles);
  if (__lane_id() == 0) {
    res[0] = first;
    res[1] = second;
    res[2] = proxy.failed() ? 1 : 0;
    res[3] = waited;
  }
}

// n7 push shapes: a slot signal past gdaSignalCount through each engine entry
// (signal, put + signal, putValue + signal), a window signal at the same offset
// of a window, which the slot check must not touch, and a window signal at an
// offset that is not 8-byte aligned.
enum BadSlotShape {
  kBadSignalSet,
  kBadSignalInc,
  kBadPutAdd,
  kBadValueInc,
  kWindowPastSlots,
  kWindowUnaligned
};

template <int Shape>
__global__ void ProxyBadSlotKernel(ccoDevComm dc, int peer, ccoGdaSignal_t slot,
                                   ccoWindow_t sendWin, ccoWindow_t recvWin, int* res) {
  ccoProxy<> proxy{dc, 0};
  const ccoCoopWarp warp{};
  if constexpr (Shape == kBadSignalSet) {
    proxy.signal(peer, ccoGda_SignalSet{slot, 0x5a5a5a5aull}, warp);
  } else if constexpr (Shape == kBadSignalInc) {
    proxy.signal(peer, ccoGda_SignalInc{slot}, warp);
  } else if constexpr (Shape == kBadPutAdd) {
    proxy.put(peer, recvWin, 0, sendWin, 0, 64, ccoGda_SignalAdd{slot, 1}, warp);
  } else if constexpr (Shape == kBadValueInc) {
    proxy.putValue(peer, recvWin, 0, uint64_t{1}, ccoGda_SignalInc{slot}, warp);
  } else if constexpr (Shape == kWindowPastSlots) {
    proxy.signal(peer, ccoGda_WindowSignalSet{recvWin, slot * sizeof(uint64_t), 1}, warp);
  } else {
    proxy.signal(peer, ccoGda_WindowSignalSet{recvWin, 4, 1}, warp);
  }
  const int flushed = proxy.flush(warp, kTimeoutCycles);
  if (__lane_id() == 0) res[0] = flushed;
}

// n3: timeout overloads (ccoCoopBlock): flush(100) = 1, then flush() completes
// it, then flush(long) = 0 with no FLUSH pushed (no wavefront pushed since), so
// flushDone +2; waitSignal 1/0/1, consuming only on 0.
__global__ void ProxyTimeoutKernel(ccoDevComm dc, ccoWindow_t dstWin, ccoWindow_t srcWin,
                                   size_t chunk, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<> proxy{dc, u};
  const uint64_t before = FlushDone(proxy.flushDone);
  const size_t off = (static_cast<size_t>(me) * gridDim.x + u) * chunk;
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.put(p, dstWin, off, srcWin, off, chunk, ccoGda_NoSignal{}, ccoCoopBlock{});
  }
  const int timedOut = proxy.flush(ccoCoopBlock{}, 100);
  proxy.flush(ccoCoopBlock{});
  const int done = proxy.flush(ccoCoopBlock{}, kTimeoutCycles);
  if (timedOut != 1 || done != 0 || proxy.failed()) {
    if (__lane_id() == 0) {
      printf("[rank %d] n3: unit %d wave %d: flush(100) = %d, flush(long) = %d, failed %d\n", me, u,
             static_cast<int>(threadIdx.x) / 64, timedOut, done, proxy.failed() ? 1 : 0);
    }
    atomicExch(err, 1);
  }
  if (threadIdx.x == 0) CheckFlushDone(proxy.flushDone, before, 2, "n3: unit", u, me, err);
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.signal(p, ccoGda_SignalSet{Slot(me, u), 1}, ccoCoopBlock{});
  }
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    const int early = proxy.waitSignal(Slot(s, u), 2, ccoCoopBlock{}, 100);
    const int got = proxy.waitSignal(Slot(s, u), 1, ccoCoopBlock{}, kTimeoutCycles);
    const int again = proxy.waitSignal(Slot(s, u), 1, ccoCoopBlock{}, 100);
    const uint64_t left = proxy.readSignal(Slot(s, u));
    if (early != 1 || got != 0 || again != 1 || left != 0) {
      if (__lane_id() == 0) {
        printf(
            "[rank %d] n3: unit %d src %d wave %d: waitSignal(2, short) = %d, (1, long) = %d, "
            "(1, short) = %d, left %llu\n",
            me, u, s, static_cast<int>(threadIdx.x) / 64, early, got, again,
            (unsigned long long)left);
      }
      atomicExch(err, 1);
    }
  }
  proxy.flush(ccoCoopBlock{});
}

__global__ void ProxyResetKernel(ccoDevComm dc, int nSlots) {
  ccoProxy<> proxy{dc, 0};  // no push
  for (int i = threadIdx.x; i < nSlots; i += blockDim.x) {
    proxy.resetSignal(static_cast<ccoGdaSignal_t>(i));
  }
}

struct Ctx {
  ccoComm* comm;
  ccoDevComm dc;        // default rings, 1 QP per (proxy thread, peer)
  ccoDevComm dcSmall;   // kSmallCapacity-line rings
  ccoDevComm dcThread;  // kThreadCapacity-line rings
  ccoDevComm dcQps;     // kQpsPerPeer QPs per (proxy thread, peer)
  ccoDevComm dcNoGda;   // GDA NONE, proxy FULL
  ccoDevComm dcSplit;   // GDA CROSSNODE (NONE on one node), proxy FULL
  hipStream_t stream;
  int rank;
  int nranks;
  ccoWindow_t sendWin, recvWin, flagWin;
  void *sendBuf, *recvBuf, *flagBuf;
  size_t bufSize;
  int nSlots;
  int* dErr;

  void Sync() {
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
  }

  // Reset d's signal slots, flags, recv window; fill the send window for this
  // salt (or clobber it, when the kernel writes the sources itself).
  void Fresh(const ccoDevComm& d, uint32_t salt, size_t chunk, int nUnits, bool fillSend = true) {
    ProxyResetKernel<<<1, 256, 0, stream>>>(d, nSlots);
    HIP_CHECK(hipMemsetAsync(flagBuf, 0, sizeof(uint64_t) * nSlots, stream));
    HIP_CHECK(hipMemsetAsync(recvBuf, 0xff, bufSize, stream));
    if (!fillSend) {
      HIP_CHECK(hipMemsetAsync(sendBuf, 0x5a, bufSize, stream));
      Sync();
      return;
    }
    std::vector<uint32_t> host(bufSize / sizeof(uint32_t), 0);
    for (int p = 0; p < nranks; p++) {
      for (int u = 0; u < nUnits; u++) {
        uint32_t* dst =
            host.data() + (static_cast<size_t>(p) * nUnits + u) * chunk / sizeof(uint32_t);
        for (size_t i = 0; i < chunk / sizeof(uint32_t); i++) dst[i] = Pattern(salt, rank, p, u, i);
      }
    }
    HIP_CHECK(hipMemcpyAsync(sendBuf, host.data(), bufSize, hipMemcpyHostToDevice, stream));
    Sync();
  }

  // laneSlices: check only active lanes' first ThreadBytes(L).
  int Verify(const char* name, uint32_t salt, size_t chunk, int nUnits, bool laneSlices = false) {
    std::vector<uint32_t> host(bufSize / sizeof(uint32_t));
    HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int s = 0; s < nranks && !bad; s++) {
      if (s == rank) continue;
      for (int u = 0; u < nUnits && !bad; u++) {
        const uint32_t* got =
            host.data() + (static_cast<size_t>(s) * nUnits + u) * chunk / sizeof(uint32_t);
        for (size_t i = 0; i < chunk / sizeof(uint32_t); i++) {
          const size_t b = i * sizeof(uint32_t);
          const int lane = static_cast<int>(b / kLaneSlice);
          const bool written =
              !laneSlices || (ThreadActive(lane) && b % kLaneSlice < ThreadBytes(lane));
          const uint32_t want = written ? Pattern(salt, s, rank, u, i) : 0xffffffffu;
          if (got[i] != want) {
            fprintf(stderr, "[rank %d] %s: src=%d unit=%d word %zu: got %08x want %08x\n", rank,
                    name, s, u, i, got[i], want);
            bad = 1;
            break;
          }
        }
      }
    }
    return Harvest(name, bad);
  }

  int Harvest(const char* name, int hostBad) {
    int dBad = 0;
    HIP_CHECK(hipMemcpy(&dBad, dErr, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));
    const int bad = (hostBad || dBad) ? 1 : 0;
    printf("[rank %d]   [%-13s] %s\n", rank, name, bad ? "FAIL" : "PASS");
    ccoBarrierAll(comm);
    return bad;
  }
};

// A positive case. `run` gets a copy of the DevComm `dev` names (dc when null,
// for a case that picks its own) with the put split size lowered to
// maxPutBytes when nonzero; the proxy posts any size up to the real limit, so
// that only makes put() split.
struct Case;
using CaseFn = int (*)(Ctx& c, const Case& k, const ccoDevComm& d);

struct Case {
  const char* name;
  uint32_t salt;
  bool atomicFree;  // posts no NIC atomic (MORI_CCO_PROXY_TEST_CASES=atomic_free)
  ccoDevComm Ctx::* dev;
  uint32_t maxPutBytes;
  CaseFn run;
};

static int RunCase(Ctx& c, const Case& k) {
  ccoDevComm d = k.dev != nullptr ? c.*k.dev : c.dc;
  if (k.maxPutBytes != 0) d.proxy.maxPutBytes = k.maxPutBytes;
  return k.run(c, k, d);
}

// ProxyPutKernel on kUnits blocks of kBlockX x kBlockY threads, kBytes per
// (sender, unit).
template <typename Q, typename Coop, int Mode, int kUnits, int kBlockX, int kBlockY = 1,
          size_t kBytes = kChunk>
static int PutCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  c.Fresh(d, k.salt, kBytes, kUnits, /*fillSend=*/Mode != kPutKernelPayload);
  ProxyPutKernel<Q, Coop, Mode><<<kUnits, dim3(kBlockX, kBlockY), 0, c.stream>>>(
      d, c.sendWin, c.recvWin, c.flagWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<uint64_t*>(c.flagBuf), k.salt, kBytes, kUnits, c.dErr);
  c.Sync();
  return c.Verify(k.name, k.salt, kBytes, kUnits);
}

// Per-lane cases: `launch()` on `nSlots` chunk slots per sender, then exact
// signal counts and the lane slices.
template <typename Launch>
static int LaneCase(Ctx& c, const Case& k, const ccoDevComm& d, int nSlots, Launch launch) {
  c.Fresh(d, k.salt, kChunk, nSlots);
  launch();
  c.Sync();  // every rank's kernel, and so its final flush, is done
  ProxyCheckSignalsKernel<<<1, 256, 0, c.stream>>>(d, c.nSlots, c.dErr);
  HIP_CHECK(hipStreamSynchronize(c.stream));
  return c.Verify(k.name, k.salt, kChunk, nSlots, /*laneSlices=*/true);
}

// ProxyThreadKernel on kUnits one-wavefront blocks, kRounds put rounds: a
// chunk slot per (unit, round).
template <typename Q, int Mode, int kUnits, int kRounds = 1>
static int ThreadCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return LaneCase(c, k, d, kUnits * kRounds, [&] {
    ProxyThreadKernel<Q, Mode><<<kUnits, 64, 0, c.stream>>>(
        d, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf),
        static_cast<const uint32_t*>(c.recvBuf), k.salt, kChunk, kUnits, kRounds, c.dErr);
  });
}

template <int Mode>
static int ThreadModeCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return LaneCase(c, k, d, kMaxUnits, [&] {
    ProxyThreadModeKernel<Mode><<<kMaxUnits, 64, 0, c.stream>>>(
        d, c.sendWin, c.recvWin, static_cast<const uint32_t*>(c.recvBuf), k.salt, kChunk, kMaxUnits,
        /*zero=*/0, c.dErr);
  });
}

// `launch()` on `nUnits` chunks, then Verify, or only the device checks when
// !verify.
template <typename Launch>
static int KernelCase(Ctx& c, const Case& k, const ccoDevComm& d, size_t chunk, int nUnits,
                      bool verify, Launch launch) {
  c.Fresh(d, k.salt, chunk, nUnits);
  launch();
  c.Sync();
  return verify ? c.Verify(k.name, k.salt, chunk, nUnits) : c.Harvest(k.name, 0);
}

template <typename Q, typename Coop, bool kSet, int kUnits>
static int ValueCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return KernelCase(c, k, d, kChunk, kUnits, /*verify=*/false, [&] {
    ProxyValueKernel<Q, Coop, kSet><<<kUnits, 64, 0, c.stream>>>(
        d, c.recvWin, static_cast<const uint64_t*>(c.recvBuf), k.salt, kUnits, c.dErr);
  });
}

template <typename Q, int Mode, int kUnits>
static int SetCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return KernelCase(c, k, d, kChunk, kUnits, /*verify=*/true, [&] {
    ProxySetKernel<Q, Mode><<<kUnits, 64, 0, c.stream>>>(
        d, c.sendWin, c.recvWin, c.flagWin, static_cast<const uint32_t*>(c.recvBuf),
        static_cast<const uint64_t*>(c.flagBuf), k.salt, kChunk, kUnits, c.dErr);
  });
}

static int StressCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  const int nUnits = kStressBlocks * kStressWarps * kStressRounds;
  return KernelCase(c, k, d, kStressChunk, nUnits, /*verify=*/true, [&] {
    ProxyStressKernel<<<kStressBlocks, kStressWarps * 64, 0, c.stream>>>(
        d, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf),
        static_cast<ccoGdaSignal_t>(c.nSlots - 1), c.dErr);
  });
}

static int SignalCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return KernelCase(c, k, d, kChunk, kSpscQueues, /*verify=*/false, [&] {
    ProxySignalKernel<<<kSpscQueues, 64, 0, c.stream>>>(
        d, static_cast<ccoGdaSignal_t>(c.nSlots - 1), c.dErr);
  });
}

static int HandoffCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return KernelCase(c, k, d, kChunk, /*nUnits=*/1, /*verify=*/true, [&] {
    ProxyHandoffKernel<<<1, 128, 0, c.stream>>>(d, c.sendWin, c.recvWin, kChunk, c.dErr);
  });
}

static int BlockFlushCase(Ctx& c, const Case& k, const ccoDevComm& d) {
  return KernelCase(c, k, d, kChunk, kBlockFlushWaves, /*verify=*/true, [&] {
    ProxyBlockFlushKernel<<<1, kBlockFlushWaves * 64, 0, c.stream>>>(
        d, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf), kChunk, c.dErr);
  });
}

// The handle's team on dcNoGda and dcSplit: the proxy's FULL team.
static int TeamCase(Ctx& c, const Case& k, const ccoDevComm&) {
  for (const ccoDevComm* d : {&c.dcNoGda, &c.dcSplit}) {
    ProxyTeamKernel<<<1, 64, 0, c.stream>>>(*d, c.rank, c.nranks, c.dErr);
  }
  HIP_CHECK(hipStreamSynchronize(c.stream));
  return c.Harvest(k.name, 0);
}

using Spsc = ccoProxySpsc;
using Mpsc = ccoProxyMpsc;

// Every positive case, in run order. The atomic-free ones come first: where
// NIC atomics fail, the first atomic case traps the process.
static const Case kCases[] = {
    // name, salt, atomicFree, DevComm, maxPutBytes, launcher
    {"lane_value_set", 46, true, &Ctx::dc, 0,
     ValueCase<Spsc, ccoCoopLaneMasked, true, kSpscQueues>},
    {"mpsc_lane_value_set", 47, true, &Ctx::dc, 0,
     ValueCase<Mpsc, ccoCoopLaneMasked, true, kMaxUnits>},
    {"set_put", 34, true, &Ctx::dc, 0, SetCase<Spsc, kSetSlot, kSpscQueues>},
    {"set_signal", 35, true, &Ctx::dc, 0, SetCase<Spsc, kSetSignal, kSpscQueues>},
    {"set_window", 36, true, &Ctx::dc, 0, SetCase<Spsc, kSetWindow, kSpscQueues>},
    {"mpsc_set_put", 37, true, &Ctx::dc, 0, SetCase<Mpsc, kSetSlot, kMaxUnits>},
    {"tm_branch", 51, true, &Ctx::dc, 0, ThreadModeCase<kTmBranch>},
    {"tm_branch_contended", 59, true, &Ctx::dcThread, 0, ThreadModeCase<kTmBranchContended>},
    {"tm_single", 52, true, &Ctx::dc, 0, ThreadModeCase<kTmSingle>},
    {"tm_agg_set", 56, true, &Ctx::dc, 0, ThreadModeCase<kTmAggregate>},
    {"tm_lane_agg_set", 60, true, &Ctx::dc, 0, ThreadModeCase<kTmLaneAggregate>},
    {"mpsc_lane_group_value_set", 61, true, &Ctx::dc, 0,
     ValueCase<Mpsc, ccoCoopLaneMaskedGroup, true, kMaxUnits>},
    {"tm_value_set", 58, true, &Ctx::dc, 0, ValueCase<Mpsc, ccoCoopThread, true, kMaxUnits>},
    // Proxy QPs without GDA QPs (dcNoGda).
    {"conn_set_put", 48, true, &Ctx::dcNoGda, 0, SetCase<Spsc, kSetSlot, kSpscQueues>},
    {"conn_team", 0, true, nullptr, 0, TeamCase},
    {"put_sig_warp", 1, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopWarp, kPutSignal, kSpscQueues, 64>},
    {"put_then_sig", 2, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopBlock, kPutThenSignal, kSpscQueues, 256>},
    {"win_signal", 3, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopWarp, kPutWindowSignal, kSpscQueues, 64>},
    {"large_put", 4, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopWarp, kPutSignal, 1, 64, 1, kLargeChunk>},
    {"flush_count", 5, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopWarp, kPutFlushCount, kSpscQueues, 64>},
    {"flush_reuse", 6, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopBlock, kPutFlushClobber, kSpscQueues, 256>},
    {"kernel_payload", 11, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopBlock, kPutKernelPayload, kSpscQueues, 256>},
    {"sq_full", 12, false, &Ctx::dc, 0, PutCase<Spsc, ccoCoopWarp, kPutRepeat, 1, 64>},
    // 32x4 blocks: wavefront 0 is linear threads 0..63 (rows 0 and 1).
    {"block_2d_flush", 38, false, &Ctx::dc, 0,
     PutCase<Spsc, ccoCoopBlock, kPutFlushCount, kSpscQueues, 32, 4>},
    {"block_2d_payload", 39, false, &Ctx::dc, 0,
     PutCase<Mpsc, ccoCoopBlock, kPutKernelPayload, kMaxUnits, 32, 4>},
    // kMaxUnits blocks over kMpscQueues MPSC queues: several concurrent producers per queue.
    {"mpsc_put_sig", 7, false, &Ctx::dc, 0, PutCase<Mpsc, ccoCoopWarp, kPutSignal, kMaxUnits, 64>},
    {"mpsc_then_sig", 8, false, &Ctx::dc, 0,
     PutCase<Mpsc, ccoCoopBlock, kPutThenSignal, kMaxUnits, 256>},
    {"mpsc_flush", 9, false, &Ctx::dc, 0,
     PutCase<Mpsc, ccoCoopBlock, kPutFlushClobber, kMaxUnits, 256>},
    {"mpsc_payload", 13, false, &Ctx::dc, 0,
     PutCase<Mpsc, ccoCoopBlock, kPutKernelPayload, kMaxUnits, 256>},
    {"split_warp", 17, false, &Ctx::dc, 4096,
     PutCase<Spsc, ccoCoopWarp, kPutSignal, kSpscQueues, 64>},
    // ccoCoopLaneMasked: per-lane offsets, sizes and signals, idle lanes; 256-byte
    // split pieces in the kThreadPutSignal cases (lanes need 1 to 4 pieces).
    {"thr_put_sig", 18, false, &Ctx::dc, 256, ThreadCase<Spsc, kThreadPutSignal, kSpscQueues>},
    {"thr_lane_sig", 19, false, &Ctx::dc, 0, ThreadCase<Spsc, kThreadPutLaneSignal, kSpscQueues>},
    {"thr_flush", 20, false, &Ctx::dc, 0, ThreadCase<Spsc, kThreadPutFlush, kSpscQueues>},
    {"mpsc_thr_sig", 21, false, &Ctx::dc, 256, ThreadCase<Mpsc, kThreadPutSignal, kMaxUnits>},
    {"mpsc_thr_lane", 22, false, &Ctx::dc, 0, ThreadCase<Mpsc, kThreadPutLaneSignal, kMaxUnits>},
    {"mpsc_thr_flush", 23, false, &Ctx::dc, 0, ThreadCase<Mpsc, kThreadPutFlush, kMaxUnits>},
    // ccoCoopLaneMaskedGroup: one signal per call and peer group, SPSC and MPSC.
    {"lane_group_sig", 62, false, &Ctx::dc, 0,
     ThreadCase<Spsc, kThreadPutGroupSignal, kSpscQueues>},
    {"lane_group_split", 63, false, &Ctx::dc, 256,
     ThreadCase<Spsc, kThreadPutGroupSignal, kSpscQueues>},
    {"mpsc_lane_group_sig", 64, false, &Ctx::dcThread, 0,
     ThreadCase<Mpsc, kThreadPutGroupSignal, kMaxUnits, kThreadRounds>},
    {"lane_group_value", 65, false, &Ctx::dc, 0,
     ValueCase<Spsc, ccoCoopLaneMaskedGroup, false, kSpscQueues>},
    // Lane-masked pushes on kThreadCapacity-line rings, kThreadRounds rounds: the
    // rings wrap, SPSC pushes retry for credit, MPSC claims wait.
    {"thr_small_spsc", 24, false, &Ctx::dcThread, 256,
     ThreadCase<Spsc, kThreadPutSignal, kSpscQueues, kThreadRounds>},
    {"thr_small_mpsc", 25, false, &Ctx::dcThread, 256,
     ThreadCase<Mpsc, kThreadPutSignal, kMaxUnits, kThreadRounds>},
    {"tm_group_sig", 53, false, &Ctx::dc, 0, ThreadModeCase<kTmGroupSignal>},
    {"tm_group_split", 54, false, &Ctx::dc, 256, ThreadModeCase<kTmGroupSignal>},
    {"tm_lane_sig", 55, false, &Ctx::dc, 0, ThreadModeCase<kTmLaneSignal>},
    {"tm_value", 57, false, &Ctx::dc, 0, ValueCase<Mpsc, ccoCoopThread, false, kMaxUnits>},
    {"put_value", 32, false, &Ctx::dc, 0, ValueCase<Spsc, ccoCoopWarp, false, kSpscQueues>},
    {"mpsc_stress", 10, false, &Ctx::dc, 0, StressCase},
    {"signal_only", 0, false, &Ctx::dc, 0, SignalCase},
    {"spsc_handoff", 16, false, &Ctx::dc, 0, HandoffCase},
    {"block_flush", 40, false, &Ctx::dc, 0, BlockFlushCase},
    {"small_spsc", 14, false, &Ctx::dcSmall, 0,
     PutCase<Spsc, ccoCoopWarp, kPutPieces, kSpscQueues, 64>},
    {"qps_put_sig", 29, false, &Ctx::dcQps, 0,
     PutCase<Spsc, ccoCoopWarp, kPutSignal, kSpscQueues, 64>},
    {"qps_flush", 30, false, &Ctx::dcQps, 0,
     PutCase<Spsc, ccoCoopBlock, kPutFlushClobber, kSpscQueues, 256>},
    {"qps_mpsc_flush", 31, false, &Ctx::dcQps, 0,
     PutCase<Mpsc, ccoCoopBlock, kPutFlushClobber, kMaxUnits, 256>},
    {"small_mpsc", 15, false, &Ctx::dcSmall, 0,
     PutCase<Mpsc, ccoCoopWarp, kPutPieces, kMaxUnits, 64>},
    // Proxy QPs without GDA QPs (dcNoGda) and with another GDA peer set (dcSplit).
    {"conn_put_sig", 49, false, &Ctx::dcNoGda, 0,
     PutCase<Spsc, ccoCoopWarp, kPutSignal, kSpscQueues, 64>},
    {"conn_split", 50, false, &Ctx::dcSplit, 0,
     PutCase<Mpsc, ccoCoopWarp, kPutSignal, kMaxUnits, 64>},
};

// Run after any negative case: dc still works.
static const Case kAfterNegatives[] = {
    {"put_sig_after", 26, true, &Ctx::dc, 0, SetCase<Spsc, kSetSlot, kSpscQueues>},
};

// MORI_CCO_PROXY_TEST_CASES / MORI_CCO_PROXY_TEST_NEGATIVE, split at commas.
struct Selection {
  bool allCases = true;  // CASES unset, empty or "all"
  std::vector<std::string> cases;
  std::vector<std::string> negatives;

  static std::vector<std::string> Split(const char* list) {
    std::vector<std::string> out;
    std::string item;
    for (const char* p = list;; p++) {
      if (*p == ',' || *p == '\0') {
        if (!item.empty()) out.push_back(item);
        item.clear();
        if (*p == '\0') break;
      } else {
        item += *p;
      }
    }
    return out;
  }

  static Selection FromEnv() {
    Selection s;
    const char* cases = getenv("MORI_CCO_PROXY_TEST_CASES");
    const char* negatives = getenv("MORI_CCO_PROXY_TEST_NEGATIVE");
    if (cases != nullptr && cases[0] != '\0') {
      s.cases = Split(cases);
      s.allCases = Has(s.cases, "all");
    }
    if (negatives != nullptr) s.negatives = Split(negatives);
    return s;
  }

  static bool Has(const std::vector<std::string>& list, const char* name) {
    return std::find(list.begin(), list.end(), name) != list.end();
  }

  bool Selected(const Case& k) const {
    return allCases || Has(cases, k.name) || (k.atomicFree && Has(cases, "atomic_free"));
  }

  bool NegativeSelected(const char* id) const {
    return Has(negatives, "all") || Has(negatives, id);
  }
};

static constexpr const char* kNegativeIds[] = {"n1", "n2", "n3", "n4", "n5",
                                               "n6", "n7", "n8", "n9"};

// Reports every name or id that matches nothing, and a selection without any
// case. Returns whether the selection is usable.
static bool CheckSelection(const Selection& s, int rank) {
  bool ok = true;
  int selected = 0;
  for (const std::string& t : s.cases) {
    bool known = t == "all" || t == "atomic_free";
    for (const Case& k : kCases) known = known || t == k.name;
    if (!known) {
      fprintf(stderr, "[rank %d] MORI_CCO_PROXY_TEST_CASES: no case \"%s\"\n", rank, t.c_str());
      ok = false;
    }
  }
  for (const std::string& t : s.negatives) {
    bool known = t == "all";
    for (const char* id : kNegativeIds) known = known || t == id;
    if (!known) {
      fprintf(stderr, "[rank %d] MORI_CCO_PROXY_TEST_NEGATIVE: no case \"%s\" (n1..n9 or all)\n",
              rank, t.c_str());
      ok = false;
    }
  }
  for (const Case& k : kCases) selected += s.Selected(k) ? 1 : 0;
  for (const char* id : kNegativeIds) selected += s.NegativeSelected(id) ? 1 : 0;
  if (ok && selected == 0) {
    fprintf(stderr, "[rank %d] MORI_CCO_PROXY_TEST_CASES / _NEGATIVE select no case\n", rank);
    ok = false;
  }
  return ok;
}

static void PrintError(int rank, const char* name, int got, const ccoDevCommError& e) {
  printf(
      "[rank %d]   %s: ccoDevCommGetError %d: code %d peer %d queue %d op %d wc %d vendor_err "
      "0x%x errno %d raddr 0x%llx len %u suppressed %llu \"%s\"\n",
      rank, name, got, e.code, e.peer, e.queue, e.op, e.wcStatus, e.vendorErr, e.sysErrno,
      (unsigned long long)e.raddr, e.len, (unsigned long long)e.suppressed, e.message);
}

// A negative case's own DevComm; reports a failed create.
static bool CreateOwn(Ctx& c, const char* name, const ccoDevCommRequirements& reqs, ccoDevComm* d) {
  if (ccoDevCommCreate(c.comm, &reqs, d) == 0) return true;
  fprintf(stderr, "[rank %d] %s: DevCommCreate failed\n", c.rank, name);
  return false;
}

// Destroys it; returns 1 (and reports) on failure.
static int DestroyOwn(Ctx& c, const char* name, ccoDevComm* d) {
  if (ccoDevCommDestroy(c.comm, d) == 0) return 0;
  fprintf(stderr, "[rank %d] %s: DevCommDestroy failed\n", c.rank, name);
  return 1;
}

// n1 (selfPut false): rank 0 puts past the end of peer 1's receive window;
// flush(coop, timeout) returns 2 twice, failed() is true, waitSignal(...,
// timeout) returns 2, and ccoDevCommGetError reports COMPLETION /
// IBV_WC_REM_ACCESS_ERR for peer 1.
// n2 (selfPut true): every rank puts to itself, which has no proxy QP: as n1,
// but BAD_REQUEST naming the rank itself. ccoDevCommDestroy must return 0.
static int RunFailCase(Ctx& c, const char* name, const ccoDevCommRequirements& reqs, bool selfPut) {
  ccoDevComm d{};
  if (!CreateOwn(c, name, reqs, &d)) return c.Harvest(name, 1);
  const bool fails = selfPut || c.rank == 0;  // ranks whose DevComm must fail
  int* dRes = nullptr;
  HIP_CHECK(hipMalloc(&dRes, 4 * sizeof(int)));
  HIP_CHECK(hipMemset(dRes, 0xff, 4 * sizeof(int)));
  ccoBarrierAll(c.comm);
  if (selfPut) {
    ProxyFailKernel<<<1, 64, 0, c.stream>>>(d, c.rank, c.recvWin, 0, c.sendWin, 4096, dRes);
  } else if (c.rank == 0) {
    ProxyFailKernel<<<1, 64, 0, c.stream>>>(d, 1, c.recvWin, c.bufSize, c.sendWin, 4096, dRes);
  }
  HIP_CHECK(hipStreamSynchronize(c.stream));
  int res[4];
  HIP_CHECK(hipMemcpy(res, dRes, sizeof(res), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dRes));
  ccoDevCommError e{};
  const int got = ccoDevCommGetError(c.comm, &d, &e);  // published before any flush sees it
  int bad = 0;
  if (fails) {
    PrintError(c.rank, name, got, e);
    const int wantPeer = selfPut ? c.rank : 1;
    bad |= res[0] != 2 || res[1] != 2 || res[2] != 1 || res[3] != 2;
    bad |= got != 1 || e.rank != c.rank || e.peer != wantPeer || e.queue != 0;
    if (selfPut) {
      bad |= e.code != CCO_PROXY_ERR_BAD_REQUEST || e.op != ccoProxyOpPut;
    } else {
      bad |= e.code != CCO_PROXY_ERR_COMPLETION || e.wcStatus != IBV_WC_REM_ACCESS_ERR ||
             e.op != IBV_WR_RDMA_WRITE || e.raddr != c.bufSize || e.len != 4096;
    }
    if (bad) {
      printf("[rank %d]   %s: flush returned %d then %d, failed() %d, waitSignal %d\n", c.rank,
             name, res[0], res[1], res[2], res[3]);
    }
  } else if (got != 0) {
    PrintError(c.rank, name, got, e);  // e.g. a responder-side error; not checked
  }
  ccoBarrierAll(c.comm);
  bad |= DestroyOwn(c, name, &d);
  return c.Harvest(name, bad);
}

static int RunTimeoutCase(Ctx& c, const ccoDevCommRequirements& reqs) {
  const char* name = "n3_timeout";
  ccoDevComm d{};
  if (!CreateOwn(c, name, reqs, &d)) return c.Harvest(name, 1);
  ProxyResetKernel<<<1, 256, 0, c.stream>>>(d, c.nSlots);
  c.Sync();
  ProxyTimeoutKernel<<<kSpscQueues, 256, 0, c.stream>>>(d, c.recvWin, c.sendWin, kChunk, c.dErr);
  c.Sync();
  ccoDevCommError e{};
  const int got = ccoDevCommGetError(c.comm, &d, &e);
  if (got != 0) PrintError(c.rank, name, got, e);
  const int destroyed = DestroyOwn(c, name, &d);
  return c.Harvest(name, got != 0 || destroyed != 0);
}

// n4-n6, n8: ccoDevCommCreate(r) must fail on every rank; Harvest's barrier
// checks the comm still works. failConnect sets the engine's hook that makes
// rank 1 fail to connect, for the create call only.
static int RunCreateFails(Ctx& c, const char* name, const ccoDevCommRequirements& r,
                          bool failConnect = false) {
  if (failConnect) setenv("MORI_CCO_PROXY_TEST_FAIL_CONNECT", "1", 1);
  ccoDevComm d{};
  const int rc = ccoDevCommCreate(c.comm, &r, &d);
  if (failConnect) unsetenv("MORI_CCO_PROXY_TEST_FAIL_CONNECT");
  if (rc == 0) {
    fprintf(stderr, "[rank %d] %s: DevCommCreate succeeded\n", c.rank, name);
    ccoDevCommDestroy(c.comm, &d);
  }
  return c.Harvest(name, rc == 0);
}

// n7: rank 0 pushes a bad signal to peer 1, one push shape per DevComm: a slot
// id of gdaSignalCount, or a window offset that is not 8-byte aligned, is
// BAD_REQUEST before posting and leaves peer 1's slots untouched; a window
// signal at the slot-id offset succeeds.
template <int Shape>
static int RunBadSlotShape(Ctx& c, const ccoDevCommRequirements& reqs, const char* name,
                           ccoProxyOp wantOp) {
  ccoDevComm d{};
  if (!CreateOwn(c, name, reqs, &d)) return c.Harvest(name, 1);
  ProxyResetKernel<<<1, 256, 0, c.stream>>>(d, c.nSlots);
  c.Sync();
  int* dRes = nullptr;
  HIP_CHECK(hipMalloc(&dRes, sizeof(int)));
  HIP_CHECK(hipMemset(dRes, 0xff, sizeof(int)));
  if (c.rank == 0) {
    ProxyBadSlotKernel<Shape><<<1, 64, 0, c.stream>>>(d, 1, static_cast<ccoGdaSignal_t>(c.nSlots),
                                                      c.sendWin, c.recvWin, dRes);
  }
  c.Sync();
  int flushed = -1;
  HIP_CHECK(hipMemcpy(&flushed, dRes, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dRes));
  ProxyCheckSignalsKernel<<<1, 256, 0, c.stream>>>(d, c.nSlots, c.dErr);
  HIP_CHECK(hipStreamSynchronize(c.stream));
  ccoDevCommError e{};
  const int got = ccoDevCommGetError(c.comm, &d, &e);
  constexpr bool kRejected = Shape != kWindowPastSlots;
  // A window's raddr carries its MR bias; only the low bits are known.
  const bool raddrOk = Shape == kWindowUnaligned
                           ? (e.raddr & 7) != 0
                           : e.raddr == static_cast<uint64_t>(c.nSlots) * sizeof(uint64_t);
  int bad = 0;
  if (c.rank == 0 && kRejected) {
    PrintError(c.rank, name, got, e);
    bad |= flushed != 2 || got != 1 || e.code != CCO_PROXY_ERR_BAD_REQUEST || e.peer != 1 ||
           e.queue != 0 || e.op != wantOp || !raddrOk;
  } else if (got != 0) {
    PrintError(c.rank, name, got, e);
    bad = 1;
  }
  if (c.rank == 0 && flushed != (kRejected ? 2 : 0)) {
    printf("[rank %d]   %s: flush returned %d\n", c.rank, name, flushed);
    bad = 1;
  }
  ccoBarrierAll(c.comm);
  bad |= DestroyOwn(c, name, &d);
  return c.Harvest(name, bad);
}

static int RunBadSlotCase(Ctx& c, const ccoDevCommRequirements& reqs) {
  int fails = 0;
  fails += RunBadSlotShape<kBadSignalSet>(c, reqs, "n7_sig_set", ccoProxyOpSignal);
  fails += RunBadSlotShape<kBadSignalInc>(c, reqs, "n7_sig_inc", ccoProxyOpSignal);
  fails += RunBadSlotShape<kBadPutAdd>(c, reqs, "n7_put_add", ccoProxyOpPutSignal);
  fails += RunBadSlotShape<kBadValueInc>(c, reqs, "n7_val_inc", ccoProxyOpPutValueSignal);
  fails += RunBadSlotShape<kWindowPastSlots>(c, reqs, "n7_win_ok", ccoProxyOpSignal);
  fails += RunBadSlotShape<kWindowUnaligned>(c, reqs, "n7_win_align", ccoProxyOpSignal);
  return fails;
}

// n9, first part: ccoDevCommGetError on a copy of a destroyed DevComm returns
// 0. Then creates `leaked` (proxy-only) for the teardown part in run_test.
static int RunStaleCopyCase(Ctx& c, const ccoDevCommRequirements& reqs, ccoDevComm* leaked) {
  ccoDevComm d{};
  int bad = ccoDevCommCreate(c.comm, &reqs, &d) != 0;
  const ccoDevComm copy = d;
  if (!bad) bad |= ccoDevCommDestroy(c.comm, &d) != 0;
  ccoDevCommError e{};
  if (!bad && ccoDevCommGetError(c.comm, &copy, &e) != 0) {
    printf("[rank %d]   n9: GetError on a destroyed DevComm's copy did not return 0\n", c.rank);
    bad = 1;
  }
  int fails = c.Harvest("n9_stale_copy", bad);
  if (ccoDevCommCreate(c.comm, &reqs, leaked) != 0) {
    fprintf(stderr, "[rank %d] n9: DevCommCreate failed\n", c.rank);
    fails++;
  }
  return fails;
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  const Selection sel = Selection::FromEnv();
  if (!CheckSelection(sel, rank)) return 1;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  const int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  Ctx c{};
  c.rank = rank;
  c.nranks = nranks;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &c.comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  // One buffer size covers every chunk layout used below.
  c.bufSize = std::max({nranks * kMaxUnits * kThreadRounds * kChunk, nranks * kLargeChunk,
                        nranks * kStressBlocks * kStressWarps * kStressRounds * kStressChunk});
  // Per (sender, unit), per (sender, unit, lane), one contended slot.
  c.nSlots = nranks * kMaxUnits * (1 + 64) + 1;
  if (ccoWindowRegister(c.comm, c.bufSize, &c.sendWin, &c.sendBuf) != 0 ||
      ccoWindowRegister(c.comm, c.bufSize, &c.recvWin, &c.recvBuf) != 0 ||
      ccoWindowRegister(c.comm, sizeof(uint64_t) * c.nSlots, &c.flagWin, &c.flagBuf) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = c.nSlots;
  reqs.proxySpscQueueCount = kSpscQueues;
  reqs.proxyMpscQueueCount = kMpscQueues;
  reqs.proxyThreadCount = kProxyThreads;
  reqs.proxyQpsPerPeer = 1;
  ccoDevCommRequirements smallReqs = reqs;
  smallReqs.proxyQueueCapacity = kSmallCapacity;
  ccoDevCommRequirements threadReqs = reqs;
  threadReqs.proxyQueueCapacity = kThreadCapacity;
  ccoDevCommRequirements qpsReqs = reqs;
  qpsReqs.proxyQpsPerPeer = kQpsPerPeer;
  ccoDevCommRequirements noGdaReqs = reqs;
  noGdaReqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  noGdaReqs.proxyConnectionType = CCO_GDA_CONNECTION_FULL;
  noGdaReqs.proxyThreadCount = 1;  // fewer busy-polling threads
  ccoDevCommRequirements splitReqs = noGdaReqs;
  splitReqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
  // dcNoGda first: on one node its proxy FULL alone back-fills the window MRs.
  if (ccoDevCommCreate(c.comm, &noGdaReqs, &c.dcNoGda) != 0 ||
      ccoDevCommCreate(c.comm, &reqs, &c.dc) != 0 ||
      ccoDevCommCreate(c.comm, &smallReqs, &c.dcSmall) != 0 ||
      ccoDevCommCreate(c.comm, &threadReqs, &c.dcThread) != 0 ||
      ccoDevCommCreate(c.comm, &qpsReqs, &c.dcQps) != 0 ||
      ccoDevCommCreate(c.comm, &splitReqs, &c.dcSplit) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  // proxyConnectionType -1 follows gdaConnectionType; the conn_* DevComms resolve their own.
  const bool multiNode = c.dc.lsaSize < nranks;
  const struct {
    const ccoDevComm* d;
    int gda, proxy;
  } want[] = {
      {&c.dc, CCO_GDA_CONNECTION_FULL, CCO_GDA_CONNECTION_FULL},
      {&c.dcSmall, CCO_GDA_CONNECTION_FULL, CCO_GDA_CONNECTION_FULL},
      {&c.dcThread, CCO_GDA_CONNECTION_FULL, CCO_GDA_CONNECTION_FULL},
      {&c.dcQps, CCO_GDA_CONNECTION_FULL, CCO_GDA_CONNECTION_FULL},
      {&c.dcNoGda, CCO_GDA_CONNECTION_NONE, CCO_GDA_CONNECTION_FULL},
      {&c.dcSplit, multiNode ? CCO_GDA_CONNECTION_CROSSNODE : CCO_GDA_CONNECTION_NONE,
       CCO_GDA_CONNECTION_FULL},
  };
  for (const auto& w : want) {
    const ccoDevComm* d = w.d;
    const bool gdaOk = d->gdaConnType == w.gda &&
                       (w.gda == CCO_GDA_CONNECTION_NONE) == (d->ibgda.endpoints == nullptr);
    if (!gdaOk || d->proxy.connType != w.proxy || d->ibgda.signalBuf == nullptr ||
        d->proxy.numSpscQueues != kSpscQueues || d->proxy.numMpscQueues != kMpscQueues) {
      fprintf(stderr,
              "[rank %d] DevComm %d: gdaConnType=%d (want %d) endpoints=%p proxy connType=%d "
              "(want %d) signalBuf=%p spsc=%d mpsc=%d\n",
              rank, static_cast<int>(&w - want), (int)d->gdaConnType, w.gda,
              (void*)d->ibgda.endpoints, d->proxy.connType, w.proxy, (void*)d->ibgda.signalBuf,
              d->proxy.numSpscQueues, d->proxy.numMpscQueues);
      return 1;
    }
  }
  HIP_CHECK(hipMalloc(&c.dErr, sizeof(int)));
  HIP_CHECK(hipMemset(c.dErr, 0, sizeof(int)));
  HIP_CHECK(hipStreamCreate(&c.stream));
  ccoBarrierAll(c.comm);

  int fails = 0;
  for (const Case& k : kCases) {
    if (sel.Selected(k)) fails += RunCase(c, k);
  }

  // Negative cases, each on a DevComm of its own; then dc must still work.
  // n5, n6, n8: rank 0 disagrees on QPs per peer, proxy connection type or
  // gdaSignalCount.
  ccoDevCommRequirements qpsMismatch = reqs, connMismatch = reqs, slotMismatch = reqs;
  qpsMismatch.proxyQpsPerPeer = rank == 0 ? 1 : 2;
  connMismatch.proxyConnectionType =
      rank == 0 ? CCO_GDA_CONNECTION_FULL : CCO_GDA_CONNECTION_CROSSNODE;
  if (rank == 0) slotMismatch.gdaSignalCount += 1;
  bool n9 = false;
  ccoDevComm leaked{};  // n9: left to ccoCommDestroy on rank 0
  const std::pair<const char*, std::function<int()>> negatives[] = {
      {"n1", [&] { return RunFailCase(c, "n1_bad_raddr", reqs, /*selfPut=*/false); }},
      {"n2", [&] { return RunFailCase(c, "n2_no_qp", reqs, /*selfPut=*/true); }},
      {"n3", [&] { return RunTimeoutCase(c, reqs); }},
      {"n4", [&] { return RunCreateFails(c, "n4_fail_connect", reqs, /*failConnect=*/true); }},
      {"n5", [&] { return RunCreateFails(c, "n5_qps_mismatch", qpsMismatch); }},
      {"n6", [&] { return RunCreateFails(c, "n6_conn_mismatch", connMismatch); }},
      {"n7", [&] { return RunBadSlotCase(c, reqs); }},
      {"n8", [&] { return RunCreateFails(c, "n8_slot_count", slotMismatch); }},
      {"n9",
       [&] {
         n9 = true;
         return RunStaleCopyCase(c, noGdaReqs, &leaked);
       }},
  };
  static_assert(std::size(negatives) == std::size(kNegativeIds), "one entry per id");
  bool anyNegative = false;
  for (const auto& [id, run] : negatives) {
    if (!sel.NegativeSelected(id)) continue;
    anyNegative = true;
    fails += run();
  }
  if (anyNegative) {
    for (const Case& k : kAfterNegatives) fails += RunCase(c, k);
  }

  // n9, teardown part: teardown is local and survives the destroy orders below.
  // Rank 0 leaves `leaked` to ccoCommDestroy while rank 1 is busy for
  // kBusyPeerSec before destroying its own: rank 0 must not wait for it. Then
  // ccoDevCommDestroy(nullptr, &leaked) must return 0.
  constexpr int kBusyPeerSec = 3;
  HIP_CHECK(hipStreamDestroy(c.stream));
  HIP_CHECK(hipFree(c.dErr));
  ccoDevCommDestroy(c.comm, &c.dcSplit);
  ccoDevCommDestroy(c.comm, &c.dcNoGda);
  ccoDevCommDestroy(c.comm, &c.dcQps);
  ccoDevCommDestroy(c.comm, &c.dcThread);
  ccoDevCommDestroy(c.comm, &c.dcSmall);
  ccoDevCommDestroy(c.comm, &c.dc);
  ccoWindowDeregister(c.comm, c.flagWin);
  ccoWindowDeregister(c.comm, c.recvWin);
  ccoWindowDeregister(c.comm, c.sendWin);
  ccoMemFree(c.comm, c.flagBuf);
  ccoMemFree(c.comm, c.recvBuf);
  ccoMemFree(c.comm, c.sendBuf);
  if (n9 && rank != 0) {
    if (rank == 1) sleep(kBusyPeerSec);  // a busy peer
    if (ccoDevCommDestroy(c.comm, &leaked) != 0) fails++;
  }
  const auto t0 = std::chrono::steady_clock::now();
  ccoCommDestroy(c.comm);
  const double commDestroySec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (n9 && rank == 0) {
    // The leaked engine was stopped above; this must skip it, not free it again.
    const int rc = ccoDevCommDestroy(nullptr, &leaked);
    const bool bad = commDestroySec >= kBusyPeerSec - 1 || rc != 0;
    printf("[rank %d]   [%-13s] %s (ccoCommDestroy %.2f s, then DevCommDestroy %d)\n", rank,
           "n9_teardown", bad ? "FAIL" : "PASS", commDestroySec, rc);
    fails += bad;
  }

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  // MORI_CCO_SKIP_GDA_FULL marks runners without intranode cross-rail RDMA. A
  // fork launch puts every rank on this node, where the DevComms need FULL
  // connections, so skip it there.
  if (getenv("MORI_CCO_SKIP_GDA_FULL") != nullptr && ccoTestIsForkLaunch(argc, argv)) {
    printf("=== CCO proxy put: SKIPPED (MORI_CCO_SKIP_GDA_FULL) ===\n");
    return 0;
  }
  const int rc = ccoTestMain(argc, argv, "CCO proxy put", "/tmp/cco_proxy_put_uid", 19890);
#ifdef MORI_WITH_MPI
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) MPI_Finalize();
#endif
  return rc;
}
