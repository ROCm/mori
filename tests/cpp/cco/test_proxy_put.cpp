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

// test: ccoProxy (host-proxy RDMA over the KIWI queue) put / signal / flush.
//
// FULL connectivity; kSpscQueues SPSC + kMpscQueues MPSC proxy queues polled by
// kProxyThreads proxy threads. Block u of a put kernel is a "unit": it uses
// queue u (MPSC: modulo the pool size) and sends a distinct chunk to every peer:
//   recv[(src * nUnits + u) * chunk + i] = Pattern(salt, src, dst, u, i)
// Signal slot (src * kMaxUnits + u) on the receiver counts that chunk's arrival
// (put-carried or a trailing signal on the same queue, hence after the data).
// Each block waits for every sender's chunk on its unit, then the host verifies
// the receive window after a barrier. The host fills the send window, except
// in the kernel_payload cases.
//
// Cases:
//   put_sig_warp   put carrying SignalInc (one descriptor), ccoCoopWarp, SPSC
//   put_then_sig   put, then a separate signal() on the same queue, ccoCoopBlock
//                  (per-(queue, peer) ordering)
//   win_signal     put carrying WindowSignalAdd into a caller flag window
//   large_put      one 4 MiB put per peer on a single queue
//   flush_count    put, flush (checks it advanced flushDone by exactly one),
//                  then signal
//   flush_reuse    put, flush, overwrite the source right away, then signal:
//                  the peer must still see the original data
//   kernel_payload every wavefront of a 4-wavefront block writes part of the
//                  chunk, then the block puts it (ccoCoopBlock): the put's
//                  fence must publish stores of the non-issuing wavefronts
//   caller_fence,  ccoProxyFenceCaller handles (SPSC ccoCoopBlock, MPSC
//   mpsc_caller    ccoCoopWarp): the block writes the chunk as in
//                  kernel_payload, calls fence() once, then puts it as
//                  kFencePieces puts per peer (the last carrying SignalInc)
//                  and flushes
//   block_2d_flush flush_count and mpsc_payload with ccoCoopBlock in 32x4 blocks
//   block_2d_payload (wavefront 0 is linear threads 0..63); flushDone advances
//                  by exactly one per flush
//   sq_full        one queue puts its chunk kSqRepeats times per peer with no
//                  flush in between, more WRs than a proxy QP's SQ holds, so
//                  the proxy waits for SQ credit
//   mpsc_put_sig   put_sig_warp on the MPSC pool, 4 blocks per queue pushing
//                  concurrently
//   mpsc_then_sig  put_then_sig on the MPSC pool, 4 blocks per queue
//   mpsc_flush     flush_reuse on the MPSC pool, 4 blocks per queue flushing
//                  concurrently
//   mpsc_payload   kernel_payload on the MPSC pool
//   mpsc_stress    kStressBlocks x kStressWarps wavefronts on one MPSC queue,
//                  each doing kStressRounds rounds of put, flushing after most
//                  rounds. After each flush a wavefront overwrites the sources
//                  of every put since its previous flush. Ticket fetch-adds and
//                  FLUSH claims are separate atomics, so under this contention
//                  FLUSHes can land in the ring out of ticket order.
//   signal_only    every queue sends kSignalRounds SignalAdd{u+1} to one
//                  contended slot on every peer; total checked via waitSignal
//   spsc_handoff   in-kernel SPSC handoff: wavefront 0 puts half the chunk,
//                  flushes and release()s; after __syncthreads() wavefront 1
//                  constructs a ccoProxy on the same queue, puts the other half
//                  carrying SignalInc and flushes; flushDone must advance by
//                  exactly 2
//   block_flush    one block of 4 wavefronts, wavefront w on SPSC queue w:
//                  ccoCoopWarp puts on per-wavefront handles, one
//                  flush(ccoCoopBlock) (flushDone +1 on every queue), then the
//                  block overwrites every source before the signals
//   block_flush_async  block_flush with flushAsync + wait (ccoCoopBlock), each
//                  wavefront holding its own queue's request
//   async_reverse, put half the chunk, flushAsync, the other half, flushAsync,
//   block_async,   wait for the second request and then the first (with a
//   mpsc_async     timeout, must return 0); SPSC flushDone +2. Then the
//                  sources are overwritten and a SignalSet tells the peer.
//                  ccoCoopWarp, ccoCoopBlock, and MPSC (4 blocks per queue)
//   counter,       the same puts, then counter(CounterInc{unit}) and
//   mpsc_counter   waitCounter: the counter reads exactly 1 (SPSC flushDone +1)
//   small_spsc,    on a second DevComm whose rings hold kSmallCapacity lines:
//   small_mpsc     each unit sends its chunk as kPieces puts per peer (the
//                  last carrying the signal), so every ring wraps many times
//                  and producers wait for ring space (SPSC: 1 producer per
//                  queue; MPSC: 4)
//   qps_put_sig,   put_sig_warp, flush_reuse and mpsc_flush on a DevComm with
//   qps_flush,     proxyQpsPerPeer = kQpsPerPeer: each proxy thread owns 3
//   qps_mpsc_flush queues (2 SPSC + 1 MPSC), each bound to its own QP per peer
//   put_value,     putValue of 1, 2, 4 and 8 bytes to every peer, the second
//   mpsc_put_value half carrying SignalInc; the receiver checks each value,
//                  that the bytes past it are untouched and that the signal
//                  slot is fully consumed (MPSC: 4 blocks per queue)
//   value_set,     put_value with the second half carrying SignalSet of 1..4
//   block_value_set  (atomic-free): ccoCoopWarp, ccoCoopBlock, and
//   lane_value_set,  ccoCoopLaneMasked where lane j of one call sends to the
//   mpsc_lane_value_set  j-th next rank (SPSC, and MPSC with 4 blocks per queue)
//   set_put,       the chunk as kSetRounds pieces, piece r followed by a
//   set_signal,    SignalSet of r + 1 (carried by the put, a separate
//   set_window,    signal(), or WindowSignalSet into the flag window); the
//   mpsc_set_put   receiver waits for each value and checks that piece
//   split_warp     put_sig_warp with the put split size lowered to 4 KiB
//                  (16 pieces per put)
//   thr_*          ccoCoopLaneMasked, one wavefront per unit: lane L puts its own
//                  offset and size (ThreadBytes(L), 4 B .. 1 KiB) and every
//                  fifth lane is idle (active = false); with more than two
//                  ranks the lanes of one call target different peers. A
//                  receiver checks the data right after its signal wait;
//                  after the kernel, idle lanes' slices must be untouched and
//                  every signal slot fully consumed. SPSC (thr_*) and MPSC
//                  with 4 wavefronts per queue (mpsc_thr_*):
//     thr_put_sig    put carrying SignalAdd{LaneSlot(unit, L), L + 1} (per-lane
//                    slot and value), split size lowered to 256 B so lanes need
//                    1..4 pieces; idle lanes pass a valid put that must not be
//                    issued
//     thr_lane_sig   put, then a signal to a per-lane slot
//     thr_flush      put, flush() from all lanes (SPSC: flushDone +1), each
//                    lane overwrites its source, then SignalAdd{L + 1} to a
//                    shared slot
//                  thr_lane_sig and thr_flush idle lanes pass garbage
//                  arguments (bad peer, null windows, a huge size)
//   thr_small_*    thr_put_sig for kThreadRounds rounds, each to its own
//                  destination, on a third DevComm whose rings hold
//                  kThreadCapacity lines: rings wrap, SPSC lane-masked pushes
//                  retry for credit, MPSC claims wait
//
// The atomic-free cases (value_set .. mpsc_counter) run first.
// MORI_CCO_PROXY_TEST_CASES=<list> runs only the listed positive cases, where
// "atomic_free" stands for all of them: for NICs whose atomics into GPU memory
// fail (the first atomic case would fail the shared DevComm and trap).
//
// Negative cases, only with MORI_CCO_PROXY_TEST_NEGATIVE=all (or a list such as
// "n1,n3"). They run after the positive cases, each on a DevComm of its own,
// and put_sig_after (a put_sig_warp case) then runs on the first DevComm:
//   n1  rank 0 puts past the end of peer 1's window, then flush(coop,
//       timeout) returns 2 twice and failed() is true; ccoDevCommGetError
//       reports COMPLETION / IBV_WC_REM_ACCESS_ERR for peer 1, stderr gets
//       exactly one error line, and ccoDevCommDestroy returns 0
//   n2  every rank puts to itself, which has no proxy QP: as n1, but
//       BAD_REQUEST naming the rank itself
//   n3  put to every peer, then flush(coop, 100 cycles) returns 1, a void
//       flush still completes, and flush(coop, long timeout) returns 0; SPSC
//       flushDone advances by exactly 3 (ccoCoopBlock), MPSC with 4 blocks
//       per queue (ccoCoopWarp), where flush(coop, 100 cycles) may also
//       return 0 (a later ticket's FLUSH can cover it first)
//   n4  MORI_CCO_PROXY_TEST_FAIL_CONNECT=1: ccoDevCommCreate fails on every
//       rank, and the comm still works afterwards
//   n5  MORI_CCO_PROXY_QPS_PER_PEER differs between ranks (1 on rank 0, 2
//       elsewhere): ccoDevCommCreate fails on every rank, and the comm still
//       works afterwards
//
// Launch: `mpirun -np N test_proxy_put` (one rank per MPI process, any number
// of nodes; built with MORI_WITH_MPI when WITH_MPI=ON), or `test_proxy_put [N]`
// to fork N local ranks.

#include <infiniband/verbs.h>

#include <string>

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;
using mori::cco::proxy::queue::MPSCD2HQueue;
using mori::cco::proxy::queue::SPSCD2HQueue;

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
static constexpr int kFencePieces = 4;     // puts per chunk in the caller-fenced cases
static constexpr int kQpsPerPeer = 3;      // per (proxy thread, peer): one per queue of a thread
static constexpr uint32_t kClobber = 0xdeadbeefu;

enum PutMode {
  kPutSignal,         // put carrying SignalInc
  kPutThenSignal,     // put, then signal()
  kPutWindowSignal,   // put carrying WindowSignalAdd
  kPutFlushCount,     // put, flush (flushDone advances by one), signal()
  kPutFlushClobber,   // put, flush, overwrite the source, signal()
  kPutKernelPayload,  // the block writes the source, then put carrying SignalInc
  kPutRepeat,         // kSqRepeats puts of the same chunk, the last carrying SignalInc
  kPutPieces,         // kPieces puts per chunk, the last carrying SignalInc
  kPutCallerFence     // the block writes the source, fence(), kFencePieces puts, flush
};

__host__ __device__ inline uint32_t Pattern(uint32_t salt, int src, int dst, int u, size_t i) {
  return salt * 0x9E3779B1u + static_cast<uint32_t>(src) * 0x85EBCA77u +
         static_cast<uint32_t>(dst) * 0xC2B2AE3Du + static_cast<uint32_t>(u) * 0x27D4EB2Fu +
         static_cast<uint32_t>(i);
}

__host__ __device__ inline ccoGdaSignal_t Slot(int src, int unit) {
  return static_cast<ccoGdaSignal_t>(src * kMaxUnits + unit);
}

// Lane 0 waits until the caller-window flag reaches `least`, polling the
// DevComm's failed bit. Returns false on every lane if the DevComm failed.
template <typename P>
__device__ inline bool WaitFlag(P& proxy, const uint64_t* flag, uint64_t least) {
  int ok = 1;
  if (__lane_id() == 0) {
    for (uint32_t polls = 1;
         __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < least; polls++) {
      if (polls % 256 == 0 && proxy.failed()) {
        ok = 0;
        break;
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // the data the flag covers
  return __builtin_amdgcn_readfirstlane(ok) != 0;
}

// Block u = unit u on queue u (MPSC: modulo the pool). Sends one chunk to every peer,
// then waits for every sender's chunk on this unit (signal slot or flag word).
template <typename Q, typename Coop, int Mode, ccoProxyFenceMode F = ccoProxyFenceEachOp>
__global__ void ProxyPutKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                               ccoWindow_t flagWin, uint32_t* sendLocal, uint64_t* flagLocal,
                               uint32_t salt, size_t chunk, int nUnits, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  // Linear thread id: the block may be 2-D (block_2d_*).
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nThreads = blockDim.x * blockDim.y * blockDim.z;
  ccoProxy<Q, F> proxy{dc, u};
  auto srcOff = [&](int p) { return (static_cast<size_t>(p) * nUnits + u) * chunk; };
  const size_t dstOff = (static_cast<size_t>(me) * nUnits + u) * chunk;
  if constexpr (Mode == kPutKernelPayload || Mode == kPutCallerFence) {
    // Every thread of the block writes the chunks; put() (or fence()) publishes them.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
      for (size_t i = tid; i < chunk / sizeof(uint32_t); i += nThreads) {
        src[i] = Pattern(salt, me, p, u, i);
      }
    }
    if constexpr (Mode == kPutCallerFence) proxy.fence(Coop{});
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
    } else if constexpr (Mode == kPutPieces || Mode == kPutCallerFence) {
      constexpr int kN = Mode == kPutPieces ? kPieces : kFencePieces;
      const size_t piece = chunk / kN;
      for (int k = 0; k < kN - 1; k++) {
        proxy.put(p, recvWin, dstOff + k * piece, sendWin, srcOff(p) + k * piece, piece,
                  ccoGda_NoSignal{}, Coop{});
      }
      const size_t last = (kN - 1) * piece;
      proxy.put(p, recvWin, dstOff + last, sendWin, srcOff(p) + last, chunk - last,
                ccoGda_SignalInc{Slot(me, u)}, Coop{});
    } else {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_NoSignal{}, Coop{});
      if constexpr (Mode == kPutThenSignal) {
        proxy.signal(p, ccoGda_SignalAdd{Slot(me, u), 1}, Coop{});
      }
    }
  }
  if constexpr (Mode == kPutCallerFence) proxy.flush(Coop{});
  if constexpr (Mode == kPutFlushCount) {
    uint64_t before = 0;
    if constexpr (ccoProxy<Q>::kSpsc) {
      before = __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    proxy.flush(Coop{});
    if constexpr (ccoProxy<Q>::kSpsc) {
      // This wavefront owns the queue: its FLUSH is the only one in flight.
      const uint64_t after =
          __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (tid == 0 && after != before + 1) {
        printf("[rank %d] flush_count: unit %d flushDone %llu -> %llu\n", me, u,
               (unsigned long long)before, (unsigned long long)after);
        atomicExch(err, 1);
      }
    }
  } else if constexpr (Mode == kPutFlushClobber) {
    proxy.flush(Coop{});
    // Every put of this unit completed: its sources may be reused right away.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
      for (size_t i = tid; i < chunk / sizeof(uint32_t); i += nThreads) src[i] = kClobber;
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
      if (tid < 64 && !WaitFlag(proxy, &flagLocal[Slot(s, u)], 1)) {  // wavefront 0
        if (tid == 0) {
          printf("[rank %d] win_signal: DevComm failed\n", me);
          atomicExch(err, 1);
        }
        return;
      }
    } else {
      proxy.waitSignal(Slot(s, u), 1, Coop{});  // one waiter per group
    }
  }
}

// ccoCoopLaneMasked cases: one wavefront per block, block u = unit u on queue u
// (MPSC: modulo the pool). Lane L owns kLaneSlice bytes at L * kLaneSlice of
// the unit's chunk to every peer; if ThreadActive(L) it puts the first
// ThreadBytes(L) of them, else the rest of its slice stays untouched.
static constexpr size_t kLaneSlice = kChunk / 64;
static constexpr int kThreadCapacity = 128;  // ring lines: 51 lanes' put + signal take 102
static constexpr int kThreadRounds = 16;     // put rounds on kThreadCapacity-line rings

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
  kThreadPutSignal,      // `rounds` x put carrying SignalAdd{LaneSlot(me, u, L), L + 1}
  kThreadPutLaneSignal,  // put, then signal(SignalInc{LaneSlot(me, u, L)})
  kThreadPutFlush        // put, flush, overwrite the source, signal(SignalAdd{Slot(me, u), L + 1})
};

// Lane L's k-th peer: every lane sends to every peer once, and with more than
// two ranks the lanes of one call target different peers.
__device__ inline int ThreadPeer(int me, int n, int k, int lane) {
  return (me + 1 + (k + lane) % (n - 1)) % n;
}

// kThreadPutSignal: idle lanes pass a valid put (their whole slice), which must
// not be issued. The other cases: idle lanes pass garbage that must not be read
// (an out-of-range peer, null windows, a huge size that would never finish
// splitting).
// Round r of unit u writes chunk slot v = u * rounds + r (nUnits * rounds slots
// per sender). A receiver block checks the data a signal covers right after
// waiting for it, so a signal that overtakes its data is caught unless the data
// happens to land first.
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
      } else {
        proxy.put(peer, dst, off(me, r), src, off(p, r), bytes, ccoGda_NoSignal{},
                  ccoCoopLaneMasked{active});
      }
    }
  }
  if constexpr (Mode == kThreadPutFlush) {
    uint64_t before = 0;
    if constexpr (ccoProxy<Q>::kSpsc) {
      before = __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    proxy.flush();  // one flush for the wavefront
    if constexpr (ccoProxy<Q>::kSpsc) {
      const uint64_t after =
          __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (lane == 0 && after != before + 1) {
        printf("[rank %d] thread flush: unit %d flushDone %llu -> %llu\n", me, u,
               (unsigned long long)before, (unsigned long long)after);
        atomicExch(err, 1);
      }
    }
    // Every lane's puts completed: each lane overwrites its own sources.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* s = sendLocal + off(p, 0) / sizeof(uint32_t);
      for (size_t i = 0; i < kLaneSlice / sizeof(uint32_t); i++) s[i] = kClobber;
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
  // Consume exactly what the senders' active lanes signalled (leftovers are
  // checked after the case by ProxyCheckSignalsKernel), then check the data
  // those signals cover: active lane slices only, as an idle-lane put could
  // still be in flight (the host checks idle slices after the barrier).
  auto check = [&](int s, int r) {  // this lane's slice of chunk slot (s, u * rounds + r)
    const int v = u * rounds + r;
    const size_t w0 = ((static_cast<size_t>(s) * nSlices + v) * chunk + lane * kLaneSlice) / 4;
    for (size_t i = 0; i < ThreadBytes(lane) / 4; i++) {
      const uint32_t got =
          __hip_atomic_load(recvLocal + w0 + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      const uint32_t want = Pattern(salt, s, me, v, lane * kLaneSlice / 4 + i);
      if (got != want) {
        printf("[rank %d] early check: src=%d slot=%d lane %d word %zu: got %08x want %08x\n", me,
               s, v, lane, i, got, want);
        atomicExch(err, 1);
        break;
      }
    }
  };
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    if constexpr (Mode == kThreadPutFlush) {
      if (lane == 0) proxy.waitSignal(Slot(s, u), ThreadSignalSum());
    } else if (active) {
      proxy.waitSignal(LaneSlot(n, s, u, lane),
                       Mode == kThreadPutSignal ? static_cast<uint64_t>(rounds) * (lane + 1) : 1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    if (active) {
      for (int r = 0; r < rounds; r++) check(s, r);
    }
  }
  // Complete every work request before the kernel ends, so the checks after
  // the barrier see all this rank sent (including anything an idle lane might
  // wrongly have issued).
  proxy.flush();
}

// putValue: unit u sends kValueSlots values to every peer, slot k of
// ((me * nUnits + u) * kValueSlots + k) * 8 holding ValueBytes(k) bytes; slots
// below kValueSlots / 2 carry no signal, the others signal Slot(me, u):
// SignalInc, or (kSet, atomic-free) SignalSet of 1, 2, ... in slot order.
// ccoCoopWarp / ccoCoopBlock send slot by slot to each peer in turn;
// ccoCoopLaneMasked sends each slot to every peer in one call, lane j to the
// j-th next rank. After the signals, each receiver checks every slot of every
// sender: its low bytes hold the value, the rest keep the 0xff fill.
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
  constexpr bool kLaneMasked = std::is_same_v<Coop, ccoCoopLaneMasked>;
  using WaitCoop =
      std::conditional_t<std::is_same_v<Coop, ccoCoopBlock>, ccoCoopBlock, ccoCoopWarp>;
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
      for (int k = 0; k < kValueSlots; k++) send(p, k, ccoCoopLaneMasked{active});
    }
  } else {
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      for (int k = 0; k < kValueSlots; k++) send(p, k, Coop{});
    }
  }
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    proxy.waitSignal(Slot(s, u), kValueSlots / 2, WaitCoop{});  // acquires on thread 0
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
      if constexpr (Mode == kSetWindow) {
        if (!WaitFlag(proxy, &flagLocal[Slot(s, u)], static_cast<uint64_t>(r + 1))) {
          if (lane == 0) {
            printf("[rank %d] set_window: DevComm failed\n", me);
            atomicExch(err, 1);
          }
          return;
        }
      } else {
        proxy.waitSignal(Slot(s, u), 1, ccoCoopWarp{});  // consumed r so far: the slot is >= r + 1
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

// Every wavefront w (of kStressBlocks x kStressWarps) shares MPSC queue 0. In
// round r it puts unit u = w * kStressRounds + r to every peer, then flushes in
// the last round and in two of every three others. A flush covers every
// earlier put of the wavefront, so it then overwrites the sources of all units
// since its previous flush.
// Finally each wavefront adds 1 to `slot` on every peer, and block 0 waits for
// every peer's wavefronts.
__global__ void ProxyStressKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                  uint32_t* sendLocal, ccoGdaSignal_t slot, int* err) {
  const int lane = __lane_id();
  const int waveSize = __builtin_amdgcn_wavefrontsize();
  const int w = blockIdx.x * (blockDim.x / waveSize) + threadIdx.x / waveSize;
  const int nWaves = gridDim.x * (blockDim.x / waveSize);
  const int nUnits = nWaves * kStressRounds;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<MPSCD2HQueue> proxy{dc, 0};
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
          if (p == me) continue;
          uint32_t* src = sendLocal + off(p, k) / sizeof(uint32_t);
          for (size_t i = lane; i < kStressChunk / sizeof(uint32_t); i += waveSize)
            src[i] = kClobber;
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
      if (p != me) proxy.signal(p, ccoGda_SignalAdd{slot, static_cast<uint64_t>(u + 1)});
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
// to wavefront 1 (the release() / __syncthreads() handoff in cco_proxy.hpp).
__global__ void ProxyHandoffKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                   size_t chunk, int* err) {
  __shared__ uint64_t before;
  const int wave = threadIdx.x / __builtin_amdgcn_wavefrontsize();
  const int me = dc.rank;
  const int n = dc.worldSize;
  const size_t half = chunk / 2;
  uint64_t* flushDone = dc.proxy.flushDone;  // SPSC queue 0
  if (wave == 0) {
    if (threadIdx.x == 0) {
      before = __hip_atomic_load(flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    ccoProxy<> proxy{dc, 0};
    for (int p = 0; p < n; p++) {
      if (p != me) proxy.put(p, recvWin, me * chunk, sendWin, p * chunk, half);
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
                  ccoGda_SignalInc{Slot(me, 0)});
      }
    }
    proxy.flush();
    const uint64_t after =
        __hip_atomic_load(flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (__lane_id() == 0) {
      if (after != before + 2) {
        printf("[rank %d] spsc_handoff: flushDone %llu -> %llu, want +2\n", me,
               (unsigned long long)before, (unsigned long long)after);
        atomicExch(err, 1);
      }
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

template <bool kAsync>
__global__ void ProxyBlockFlushKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                      uint32_t* sendLocal, size_t chunk, int* err) {
  const int w = threadIdx.x / 64;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<> proxy{dc, w};
  auto srcOff = [&](int p, int u) {
    return (static_cast<size_t>(p) * kBlockFlushWaves + u) * chunk;
  };
  const uint64_t before =
      __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    proxy.put(p, recvWin, srcOff(me, w), sendWin, srcOff(p, w), chunk, ccoGda_NoSignal{},
              ccoCoopWarp{});
  }
  if constexpr (kAsync) {
    ccoGdaRequest_t req;  // per wavefront: its own queue's
    proxy.flushAsync(0, &req, ccoCoopBlock{});
    proxy.wait(req, ccoCoopBlock{});
  } else {
    proxy.flush(ccoCoopBlock{});
  }
  const uint64_t after =
      __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  if (__lane_id() == 0 && after != before + 1) {
    printf("[rank %d] block_flush%s: wave %d flushDone %llu -> %llu\n", me, kAsync ? "_async" : "",
           w, (unsigned long long)before, (unsigned long long)after);
    atomicExch(err, 1);
  }
  // Every wavefront's puts completed: overwrite all of the block's sources.
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    for (int u = 0; u < kBlockFlushWaves; u++) {
      uint32_t* src = sendLocal + srcOff(p, u) / sizeof(uint32_t);
      for (size_t i = threadIdx.x; i < chunk / sizeof(uint32_t); i += blockDim.x) src[i] = kClobber;
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

// Long enough for any proxy answer, short enough not to hang.
static constexpr uint64_t kNegTimeoutCycles = 40ull * 1000 * 1000 * 1000;

// flushAsync / wait and counter(). Unit u (queue u; MPSC: modulo the pool)
// puts the first half of its chunk to every peer, then
//   kAsyncReverse: flushAsync (r1), the second half, flushAsync (r2), wait(r2),
//                  then wait(r1, coop, timeout), which must return 0;
//   kAsyncCounter: the second half, counter(CounterInc{u}), waitCounter(u, 1),
//                  and the counter must read exactly 1.
// SPSC: flushDone advances by exactly 2 (1 for counter). The block then
// overwrites its sources and sets Slot(me, u) to 1 on every peer (SignalSet,
// atomic-free); the host checks that the original data arrived.
enum AsyncMode { kAsyncReverse, kAsyncCounter };

template <typename Q, typename Coop, int Mode>
__global__ void ProxyAsyncKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                 uint32_t* sendLocal, size_t chunk, int nUnits, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  const int tid = threadIdx.x;
  ccoProxy<Q> proxy{dc, u};
  auto srcOff = [&](int p) { return (static_cast<size_t>(p) * nUnits + u) * chunk; };
  const size_t dstOff = (static_cast<size_t>(me) * nUnits + u) * chunk;
  const size_t half = chunk / 2;
  uint64_t before = 0;
  if constexpr (ccoProxy<Q>::kSpsc) {
    before = __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if constexpr (Mode == kAsyncCounter) {
    if (tid == 0) proxy.resetCounter(u);
    __syncthreads();
  }
  auto putHalf = [&](size_t at, size_t bytes) {
    for (int p = 0; p < n; p++) {
      if (p != me) {
        proxy.put(p, recvWin, dstOff + at, sendWin, srcOff(p) + at, bytes, ccoGda_NoSignal{},
                  Coop{});
      }
    }
  };
  putHalf(0, half);
  if constexpr (Mode == kAsyncReverse) {
    ccoGdaRequest_t r1, r2;
    proxy.flushAsync(0, &r1, Coop{});
    putHalf(half, chunk - half);
    proxy.flushAsync(0, &r2, Coop{});
    proxy.wait(r2, Coop{});
    const int rc = proxy.wait(r1, Coop{}, kNegTimeoutCycles);
    if (tid == 0 && rc != 0) {
      printf("[rank %d] async: unit %d wait(r1) returned %d\n", me, u, rc);
      atomicExch(err, 1);
    }
  } else {
    putHalf(half, chunk - half);
    proxy.counter(ccoGda_CounterInc{static_cast<ccoGdaCounter_t>(u)}, Coop{});
    proxy.waitCounter(u, 1, Coop{});
    if (tid == 0 && proxy.readCounter(u) != 1) {
      printf("[rank %d] counter: unit %d reads %llu, want 1\n", me, u,
             (unsigned long long)proxy.readCounter(u));
      atomicExch(err, 1);
    }
  }
  if constexpr (ccoProxy<Q>::kSpsc) {
    const uint64_t after =
        __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const uint64_t want = before + (Mode == kAsyncReverse ? 2 : 1);
    if (tid == 0 && after != want) {
      printf("[rank %d] async: unit %d flushDone %llu -> %llu, want %llu\n", me, u,
             (unsigned long long)before, (unsigned long long)after, (unsigned long long)want);
      atomicExch(err, 1);
    }
  }
  // Every put of this unit completed: its sources may be reused right away.
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
    for (size_t i = tid; i < chunk / sizeof(uint32_t); i += blockDim.x) src[i] = kClobber;
  }
  __threadfence_system();
  __syncthreads();
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.signal(p, ccoGda_SignalSet{Slot(me, u), 1}, Coop{});
  }
  for (int s = 0; s < n; s++) {
    if (s != me) proxy.waitSignal(Slot(s, u), 1, Coop{});
  }
}

// n1 / n2: one wavefront puts `bytes` to `peer`, then flushes twice with a
// timeout. res = {first flush, second flush, failed()}.
__global__ void ProxyFailKernel(ccoDevComm dc, int peer, ccoWindow_t dstWin, size_t dstOffset,
                                ccoWindow_t srcWin, size_t bytes, int* res) {
  ccoProxy<> proxy{dc, 0};
  proxy.put(peer, dstWin, dstOffset, srcWin, 0, bytes, ccoGda_NoSignal{}, ccoCoopWarp{});
  const int first = proxy.flush(ccoCoopWarp{}, kNegTimeoutCycles);
  const int second = proxy.flush(ccoCoopWarp{}, kNegTimeoutCycles);
  if (__lane_id() == 0) {
    res[0] = first;
    res[1] = second;
    res[2] = proxy.failed() ? 1 : 0;
  }
}

// n3: block u (queue u, MPSC: modulo the pool) puts its chunk to every peer,
// then flushes with a timeout the proxy cannot meet, without one, and with a
// long one. Every thread checks the results it got.
template <typename Q, typename Coop>
__global__ void ProxyTimeoutKernel(ccoDevComm dc, ccoWindow_t dstWin, ccoWindow_t srcWin,
                                   size_t chunk, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<Q> proxy{dc, u};
  uint64_t before = 0;
  if constexpr (ccoProxy<Q>::kSpsc) {
    before = __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  const size_t off = (static_cast<size_t>(me) * gridDim.x + u) * chunk;
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.put(p, dstWin, off, srcWin, off, chunk, ccoGda_NoSignal{}, Coop{});
  }
  const int timedOut = proxy.flush(Coop{}, 100);
  proxy.flush(Coop{});
  const int done = proxy.flush(Coop{}, kNegTimeoutCycles);
  // MPSC: another block's later-ticket FLUSH may already cover ours.
  const bool timedOutOk = timedOut == 1 || (!ccoProxy<Q>::kSpsc && timedOut == 0);
  if (!timedOutOk || done != 0 || proxy.failed()) {
    if (__lane_id() == 0) {
      printf("[rank %d] n3: unit %d wave %d: flush(100) = %d, flush(long) = %d, failed %d\n", me, u,
             static_cast<int>(threadIdx.x) / 64, timedOut, done, proxy.failed() ? 1 : 0);
    }
    atomicExch(err, 1);
  }
  if constexpr (ccoProxy<Q>::kSpsc) {
    const uint64_t after =
        __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (threadIdx.x == 0 && after != before + 3) {
      printf("[rank %d] n3: unit %d flushDone %llu -> %llu, want +3\n", me, u,
             (unsigned long long)before, (unsigned long long)after);
      atomicExch(err, 1);
    }
  }
}

__global__ void ProxyResetKernel(ccoDevComm dc, int nSlots) {
  ccoProxy<> proxy{dc, 0};  // no push: reads the slots' signal state only
  for (int i = threadIdx.x; i < nSlots; i += blockDim.x) {
    proxy.resetSignal(static_cast<ccoGdaSignal_t>(i));
  }
}

// Positive cases that use no NIC atomics (MORI_CCO_PROXY_TEST_CASES=atomic_free).
static const char* const kAtomicFreeCases[] = {
    "value_set",  "block_value_set", "lane_value_set", "mpsc_lane_value_set", "set_put",
    "set_signal", "set_window",      "mpsc_set_put",   "async_reverse",       "block_async",
    "mpsc_async", "counter",         "mpsc_counter"};

// MORI_CCO_PROXY_TEST_CASES (same on every rank): unset runs every positive
// case; else a comma-separated list of case names, where "atomic_free" stands
// for kAtomicFreeCases (e.g. on platforms whose NIC atomics into GPU memory
// fail, which would fail the DevComm the other cases share).
static bool CaseSelected(const char* name) {
  const char* env = getenv("MORI_CCO_PROXY_TEST_CASES");
  if (env == nullptr || env[0] == '\0') return true;
  const std::string list = std::string(",") + env + ",";
  auto listed = [&](const char* n) {
    return list.find(std::string(",") + n + ",") != std::string::npos;
  };
  if (listed(name)) return true;
  if (!listed("atomic_free")) return false;
  for (const char* n : kAtomicFreeCases) {
    if (!strcmp(n, name)) return true;
  }
  return false;
}

struct Ctx {
  ccoComm* comm;
  ccoDevComm dc;        // default rings
  ccoDevComm dcSmall;   // kSmallCapacity-line rings
  ccoDevComm dcThread;  // kThreadCapacity-line rings
  ccoDevComm dcQps;     // kQpsPerPeer QPs per (proxy thread, peer)
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

  // laneSlices: the ccoCoopLaneMasked layout, where only the first ThreadBytes(L)
  // of an active lane L's slice are written and the rest keeps the fill.
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

// maxPutBytes != 0 lowers the DevComm's put split size (a kernel-argument copy;
// the proxy posts any size up to the real limit), so put() splits.
template <typename Q, typename Coop, int Mode, ccoProxyFenceMode F = ccoProxyFenceEachOp>
static int RunPutCase(Ctx& c, const char* name, uint32_t salt, size_t chunk, int nUnits,
                      dim3 blockDim, const ccoDevComm* dev = nullptr, uint64_t maxPutBytes = 0) {
  if (!CaseSelected(name)) return 0;
  ccoDevComm d = dev ? *dev : c.dc;
  if (maxPutBytes != 0) d.proxy.maxPutBytes = maxPutBytes;
  c.Fresh(d, salt, chunk, nUnits,
          /*fillSend=*/Mode != kPutKernelPayload && Mode != kPutCallerFence);
  ProxyPutKernel<Q, Coop, Mode, F><<<nUnits, blockDim, 0, c.stream>>>(
      d, c.sendWin, c.recvWin, c.flagWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<uint64_t*>(c.flagBuf), salt, chunk, nUnits, c.dErr);
  c.Sync();
  return c.Verify(name, salt, chunk, nUnits);
}

template <typename Q, int Mode>
static int RunThreadCase(Ctx& c, const char* name, uint32_t salt, int nUnits,
                         uint64_t maxPutBytes = 0, bool smallRings = false) {
  if (!CaseSelected(name)) return 0;
  ccoDevComm d = smallRings ? c.dcThread : c.dc;
  if (maxPutBytes != 0) d.proxy.maxPutBytes = maxPutBytes;
  const int rounds = smallRings ? kThreadRounds : 1;
  c.Fresh(d, salt, kChunk, nUnits * rounds);  // a chunk slot per (unit, round)
  ProxyThreadKernel<Q, Mode><<<nUnits, 64, 0, c.stream>>>(
      d, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<const uint32_t*>(c.recvBuf), salt, kChunk, nUnits, rounds, c.dErr);
  c.Sync();  // every rank's kernel, and so its final flush, is done
  ProxyCheckSignalsKernel<<<1, 256, 0, c.stream>>>(d, c.nSlots, c.dErr);
  HIP_CHECK(hipStreamSynchronize(c.stream));
  return c.Verify(name, salt, kChunk, nUnits * rounds, /*laneSlices=*/true);
}

template <typename Q, typename Coop, bool kSet>
static int RunValueCase(Ctx& c, const char* name, uint32_t salt, int nUnits, int blockDim = 64) {
  if (!CaseSelected(name)) return 0;
  c.Fresh(c.dc, salt, kChunk, nUnits);
  ProxyValueKernel<Q, Coop, kSet><<<nUnits, blockDim, 0, c.stream>>>(
      c.dc, c.recvWin, static_cast<const uint64_t*>(c.recvBuf), salt, nUnits, c.dErr);
  c.Sync();
  return c.Harvest(name, 0);
}

template <typename Q, int Mode>
static int RunSetCase(Ctx& c, const char* name, uint32_t salt, int nUnits) {
  if (!CaseSelected(name)) return 0;
  c.Fresh(c.dc, salt, kChunk, nUnits);
  ProxySetKernel<Q, Mode><<<nUnits, 64, 0, c.stream>>>(
      c.dc, c.sendWin, c.recvWin, c.flagWin, static_cast<const uint32_t*>(c.recvBuf),
      static_cast<const uint64_t*>(c.flagBuf), salt, kChunk, nUnits, c.dErr);
  c.Sync();
  return c.Verify(name, salt, kChunk, nUnits);
}

static int RunStressCase(Ctx& c) {
  if (!CaseSelected("mpsc_stress")) return 0;
  const int nWaves = kStressBlocks * kStressWarps;
  const int nUnits = nWaves * kStressRounds;
  c.Fresh(c.dc, /*salt=*/10, kStressChunk, nUnits);
  ProxyStressKernel<<<kStressBlocks, kStressWarps * 64, 0, c.stream>>>(
      c.dc, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<ccoGdaSignal_t>(c.nSlots - 1), c.dErr);
  c.Sync();
  return c.Verify("mpsc_stress", 10, kStressChunk, nUnits);
}

template <bool kAsync>
static int RunBlockFlushCase(Ctx& c, const char* name, uint32_t salt) {
  if (!CaseSelected(name)) return 0;
  c.Fresh(c.dc, salt, kChunk, kBlockFlushWaves);
  ProxyBlockFlushKernel<kAsync><<<1, kBlockFlushWaves * 64, 0, c.stream>>>(
      c.dc, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf), kChunk, c.dErr);
  c.Sync();
  return c.Verify(name, salt, kChunk, kBlockFlushWaves);
}

template <typename Q, typename Coop, int Mode>
static int RunAsyncCase(Ctx& c, const char* name, uint32_t salt, int nUnits, int blockDim) {
  if (!CaseSelected(name)) return 0;
  c.Fresh(c.dc, salt, kChunk, nUnits);
  ProxyAsyncKernel<Q, Coop, Mode><<<nUnits, blockDim, 0, c.stream>>>(
      c.dc, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf), kChunk, nUnits, c.dErr);
  c.Sync();
  return c.Verify(name, salt, kChunk, nUnits);
}

static int RunHandoffCase(Ctx& c) {
  if (!CaseSelected("spsc_handoff")) return 0;
  c.Fresh(c.dc, /*salt=*/16, kChunk, /*nUnits=*/1);
  ProxyHandoffKernel<<<1, 128, 0, c.stream>>>(c.dc, c.sendWin, c.recvWin, kChunk, c.dErr);
  c.Sync();
  return c.Verify("spsc_handoff", 16, kChunk, 1);
}

// Sends this process's stderr (proxy threads included) to a temporary file
// until Stop(), which copies it back and counts the lines containing `needle`.
struct StderrCapture {
  int saved = -1;
  char path[64] = {};

  void Start() {
    fflush(stderr);
    snprintf(path, sizeof(path), "/tmp/cco_proxy_neg_XXXXXX");
    const int fd = mkstemp(path);
    if (fd < 0) return;
    saved = dup(STDERR_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
  }

  // -1 if nothing was captured.
  int Stop(const char* needle) {
    if (saved < 0) return -1;
    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    saved = -1;
    int count = 0;
    if (FILE* f = fopen(path, "r")) {
      char line[1024];
      while (fgets(line, sizeof(line), f) != nullptr) {
        fputs(line, stderr);
        if (strstr(line, needle) != nullptr) count++;
      }
      fclose(f);
    }
    unlink(path);
    return count;
  }
};

static void PrintError(int rank, const char* name, int got, const ccoDevCommError& e) {
  printf(
      "[rank %d]   %s: ccoDevCommGetError %d: code %d peer %d queue %d op %d wc %d vendor_err "
      "0x%x errno %d raddr 0x%llx len %u suppressed %llu \"%s\"\n",
      rank, name, got, e.code, e.peer, e.queue, e.op, e.wcStatus, e.vendorErr, e.sysErrno,
      (unsigned long long)e.raddr, e.len, (unsigned long long)e.suppressed, e.message);
}

// n1 (selfPut false): rank 0 puts past the end of peer 1's receive window.
// n2 (selfPut true): every rank puts to itself, which has no proxy QP.
static int RunFailCase(Ctx& c, const char* name, const ccoDevCommRequirements& reqs, bool selfPut) {
  ccoDevComm d{};
  if (ccoDevCommCreate(c.comm, &reqs, &d) != 0) {
    fprintf(stderr, "[rank %d] %s: DevCommCreate failed\n", c.rank, name);
    return c.Harvest(name, 1);
  }
  const bool fails = selfPut || c.rank == 0;  // ranks whose DevComm must fail
  int* dRes = nullptr;
  HIP_CHECK(hipMalloc(&dRes, 3 * sizeof(int)));
  HIP_CHECK(hipMemset(dRes, 0xff, 3 * sizeof(int)));
  StderrCapture capture;
  if (fails) capture.Start();
  ccoBarrierAll(c.comm);
  if (selfPut) {
    ProxyFailKernel<<<1, 64, 0, c.stream>>>(d, c.rank, c.recvWin, 0, c.sendWin, 4096, dRes);
  } else if (c.rank == 0) {
    ProxyFailKernel<<<1, 64, 0, c.stream>>>(d, 1, c.recvWin, c.bufSize, c.sendWin, 4096, dRes);
  }
  HIP_CHECK(hipStreamSynchronize(c.stream));
  int res[3];
  HIP_CHECK(hipMemcpy(res, dRes, sizeof(res), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dRes));
  ccoDevCommError e{};
  const int got = ccoDevCommGetError(c.comm, &d, &e);  // published before any flush sees it
  int bad = 0;
  if (fails) {
    PrintError(c.rank, name, got, e);
    const int wantPeer = selfPut ? c.rank : 1;
    bad |= res[0] != 2 || res[1] != 2 || res[2] != 1;
    bad |= got != 1 || e.rank != c.rank || e.peer != wantPeer || e.queue != 0;
    if (selfPut) {
      bad |= e.code != CCO_PROXY_ERR_BAD_REQUEST || e.op != ccoProxyOpPut;
    } else {
      bad |= e.code != CCO_PROXY_ERR_COMPLETION || e.wcStatus != IBV_WC_REM_ACCESS_ERR ||
             e.op != IBV_WR_RDMA_WRITE || e.raddr != c.bufSize || e.len != 4096;
    }
    if (bad) {
      printf("[rank %d]   %s: flush returned %d then %d, failed() %d\n", c.rank, name, res[0],
             res[1], res[2]);
    }
  } else if (got != 0) {
    PrintError(c.rank, name, got, e);  // e.g. a responder-side error; not checked
  }
  ccoBarrierAll(c.comm);
  if (ccoDevCommDestroy(c.comm, &d) != 0) {
    fprintf(stderr, "[rank %d] %s: DevCommDestroy failed\n", c.rank, name);
    bad = 1;
  }
  const int lines = capture.Stop("[mori cco] ERROR: ccoProxy rank");
  if (fails && lines >= 0 && lines != 1) {
    fprintf(stderr, "[rank %d] %s: %d proxy error lines on stderr, want 1\n", c.rank, name, lines);
    bad = 1;
  }
  return c.Harvest(name, bad);
}

template <typename Q, typename Coop>
static int RunTimeoutCase(Ctx& c, const char* name, const ccoDevCommRequirements& reqs, int nUnits,
                          int blockDim) {
  ccoDevComm d{};
  if (ccoDevCommCreate(c.comm, &reqs, &d) != 0) {
    fprintf(stderr, "[rank %d] %s: DevCommCreate failed\n", c.rank, name);
    return c.Harvest(name, 1);
  }
  ccoBarrierAll(c.comm);
  ProxyTimeoutKernel<Q, Coop>
      <<<nUnits, blockDim, 0, c.stream>>>(d, c.recvWin, c.sendWin, kChunk, c.dErr);
  c.Sync();
  ccoDevCommError e{};
  const int got = ccoDevCommGetError(c.comm, &d, &e);
  if (got != 0) PrintError(c.rank, name, got, e);
  const int destroyed = ccoDevCommDestroy(c.comm, &d);
  return c.Harvest(name, got != 0 || destroyed != 0);
}

static int RunFailConnectCase(Ctx& c, const ccoDevCommRequirements& reqs) {
  setenv("MORI_CCO_PROXY_TEST_FAIL_CONNECT", "1", 1);
  ccoDevComm d{};
  const int rc = ccoDevCommCreate(c.comm, &reqs, &d);
  unsetenv("MORI_CCO_PROXY_TEST_FAIL_CONNECT");
  if (rc == 0) {
    fprintf(stderr, "[rank %d] n4_fail_connect: DevCommCreate succeeded\n", c.rank);
    ccoDevCommDestroy(c.comm, &d);
  }
  return c.Harvest("n4_fail_connect", rc == 0);  // its barrier checks the comm still works
}

// n5: ranks disagree on QPs per peer through the env (the field is 0).
static int RunQpsMismatchCase(Ctx& c, const ccoDevCommRequirements& reqs) {
  const char* prev = getenv("MORI_CCO_PROXY_QPS_PER_PEER");
  const std::string saved = prev ? prev : "";
  setenv("MORI_CCO_PROXY_QPS_PER_PEER", c.rank == 0 ? "1" : "2", 1);
  ccoDevCommRequirements r = reqs;
  r.proxyQpsPerPeer = 0;
  ccoDevComm d{};
  const int rc = ccoDevCommCreate(c.comm, &r, &d);
  if (prev) {
    setenv("MORI_CCO_PROXY_QPS_PER_PEER", saved.c_str(), 1);
  } else {
    unsetenv("MORI_CCO_PROXY_QPS_PER_PEER");
  }
  if (rc == 0) {
    fprintf(stderr, "[rank %d] n5_qps_mismatch: DevCommCreate succeeded\n", c.rank);
    ccoDevCommDestroy(c.comm, &d);
  }
  return c.Harvest("n5_qps_mismatch", rc == 0);  // its barrier checks the comm still works
}

// MORI_CCO_PROXY_TEST_NEGATIVE: "all" or a comma-separated list of case names.
static bool NegativeSelected(const char* name) {
  const char* env = getenv("MORI_CCO_PROXY_TEST_NEGATIVE");
  if (env == nullptr || env[0] == '\0') return false;
  if (!strcmp(env, "all")) return true;
  const std::string list = std::string(",") + env + ",";
  return list.find(std::string(",") + name + ",") != std::string::npos;
}

static int RunSignalCase(Ctx& c) {
  if (!CaseSelected("signal_only")) return 0;
  c.Fresh(c.dc, /*salt=*/0, kChunk, kSpscQueues);
  const ccoGdaSignal_t slot = static_cast<ccoGdaSignal_t>(c.nSlots - 1);
  ProxySignalKernel<<<kSpscQueues, 64, 0, c.stream>>>(c.dc, slot, c.dErr);
  c.Sync();
  return c.Harvest("signal_only", 0);
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
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
  reqs.gdaCounterCount = kMaxUnits;  // one per unit (counter cases)
  reqs.proxySpscQueueCount = kSpscQueues;
  reqs.proxyMpscQueueCount = kMpscQueues;
  reqs.proxyThreadCount = kProxyThreads;
  ccoDevCommRequirements smallReqs = reqs;
  smallReqs.proxyQueueCapacity = kSmallCapacity;
  ccoDevCommRequirements threadReqs = reqs;
  threadReqs.proxyQueueCapacity = kThreadCapacity;
  ccoDevCommRequirements qpsReqs = reqs;
  qpsReqs.proxyQpsPerPeer = kQpsPerPeer;
  if (ccoDevCommCreate(c.comm, &reqs, &c.dc) != 0 ||
      ccoDevCommCreate(c.comm, &smallReqs, &c.dcSmall) != 0 ||
      ccoDevCommCreate(c.comm, &threadReqs, &c.dcThread) != 0 ||
      ccoDevCommCreate(c.comm, &qpsReqs, &c.dcQps) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  for (const ccoDevComm* d : {&c.dc, &c.dcSmall, &c.dcThread, &c.dcQps}) {
    if (d->gdaConnType == CCO_GDA_CONNECTION_NONE || d->proxy.numSpscQueues != kSpscQueues ||
        d->proxy.numMpscQueues != kMpscQueues) {
      fprintf(stderr, "[rank %d] proxy queues missing (gdaConnType=%d spsc=%d mpsc=%d)\n", rank,
              (int)d->gdaConnType, d->proxy.numSpscQueues, d->proxy.numMpscQueues);
      return 1;
    }
  }
  HIP_CHECK(hipMalloc(&c.dErr, sizeof(int)));
  HIP_CHECK(hipMemset(c.dErr, 0, sizeof(int)));
  HIP_CHECK(hipStreamCreate(&c.stream));
  ccoBarrierAll(c.comm);

  int fails = 0;
  // Atomic-free cases first: on platforms where NIC atomics fail, they still
  // run before the first atomic fails the shared DevComm.
  fails += RunValueCase<SPSCD2HQueue, ccoCoopWarp, true>(c, "value_set", 44, kSpscQueues);
  fails +=
      RunValueCase<SPSCD2HQueue, ccoCoopBlock, true>(c, "block_value_set", 45, kSpscQueues, 256);
  fails +=
      RunValueCase<SPSCD2HQueue, ccoCoopLaneMasked, true>(c, "lane_value_set", 46, kSpscQueues);
  fails +=
      RunValueCase<MPSCD2HQueue, ccoCoopLaneMasked, true>(c, "mpsc_lane_value_set", 47, kMaxUnits);
  fails += RunSetCase<SPSCD2HQueue, kSetSlot>(c, "set_put", 34, kSpscQueues);
  fails += RunSetCase<SPSCD2HQueue, kSetSignal>(c, "set_signal", 35, kSpscQueues);
  fails += RunSetCase<SPSCD2HQueue, kSetWindow>(c, "set_window", 36, kSpscQueues);
  fails += RunSetCase<MPSCD2HQueue, kSetSlot>(c, "mpsc_set_put", 37, kMaxUnits);
  fails += RunAsyncCase<SPSCD2HQueue, ccoCoopWarp, kAsyncReverse>(c, "async_reverse", 48,
                                                                  kSpscQueues, 64);
  fails += RunAsyncCase<SPSCD2HQueue, ccoCoopBlock, kAsyncReverse>(c, "block_async", 49,
                                                                   kSpscQueues, 256);
  fails +=
      RunAsyncCase<MPSCD2HQueue, ccoCoopBlock, kAsyncReverse>(c, "mpsc_async", 50, kMaxUnits, 256);
  fails +=
      RunAsyncCase<SPSCD2HQueue, ccoCoopWarp, kAsyncCounter>(c, "counter", 51, kSpscQueues, 64);
  fails += RunAsyncCase<MPSCD2HQueue, ccoCoopBlock, kAsyncCounter>(c, "mpsc_counter", 52, kMaxUnits,
                                                                   256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "put_sig_warp", 1, kChunk,
                                                             kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutThenSignal>(c, "put_then_sig", 2, kChunk,
                                                                  kSpscQueues, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutWindowSignal>(c, "win_signal", 3, kChunk,
                                                                   kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "large_put", 4, kLargeChunk, 1, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutFlushCount>(c, "flush_count", 5, kChunk,
                                                                 kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutFlushClobber>(c, "flush_reuse", 6, kChunk,
                                                                    kSpscQueues, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutKernelPayload>(c, "kernel_payload", 11,
                                                                     kChunk, kSpscQueues, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutCallerFence, ccoProxyFenceCaller>(
      c, "caller_fence", 27, kChunk, kSpscQueues, 256);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopWarp, kPutCallerFence, ccoProxyFenceCaller>(
      c, "mpsc_caller", 28, kChunk, kMaxUnits, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutRepeat>(c, "sq_full", 12, kChunk, 1, 64);
  // 32x4 blocks: wavefront 0 is linear threads 0..63 (rows 0 and 1).
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutFlushCount>(c, "block_2d_flush", 38, kChunk,
                                                                  kSpscQueues, dim3(32, 4));
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutKernelPayload>(
      c, "block_2d_payload", 39, kChunk, kMaxUnits, dim3(32, 4));
  // kMaxUnits blocks over kMpscQueues MPSC queues: several concurrent producers per queue.
  fails += RunPutCase<MPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "mpsc_put_sig", 7, kChunk,
                                                             kMaxUnits, 64);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutThenSignal>(c, "mpsc_then_sig", 8, kChunk,
                                                                  kMaxUnits, 256);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutFlushClobber>(c, "mpsc_flush", 9, kChunk,
                                                                    kMaxUnits, 256);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutKernelPayload>(c, "mpsc_payload", 13, kChunk,
                                                                     kMaxUnits, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "split_warp", 17, kChunk,
                                                             kSpscQueues, 64, nullptr, 4096);
  // ccoCoopLaneMasked: per-lane offsets, sizes and signals, idle lanes; 256-byte
  // split pieces in the *_put_sig cases (lanes need 1 to 4 pieces).
  fails += RunThreadCase<SPSCD2HQueue, kThreadPutSignal>(c, "thr_put_sig", 18, kSpscQueues, 256);
  fails += RunThreadCase<SPSCD2HQueue, kThreadPutLaneSignal>(c, "thr_lane_sig", 19, kSpscQueues);
  fails += RunThreadCase<SPSCD2HQueue, kThreadPutFlush>(c, "thr_flush", 20, kSpscQueues);
  fails += RunThreadCase<MPSCD2HQueue, kThreadPutSignal>(c, "mpsc_thr_sig", 21, kMaxUnits, 256);
  fails += RunThreadCase<MPSCD2HQueue, kThreadPutLaneSignal>(c, "mpsc_thr_lane", 22, kMaxUnits);
  fails += RunThreadCase<MPSCD2HQueue, kThreadPutFlush>(c, "mpsc_thr_flush", 23, kMaxUnits);
  // Lane-masked pushes on kThreadCapacity-line rings, kThreadRounds rounds: the
  // rings wrap, SPSC pushes retry for credit, MPSC claims wait.
  fails += RunThreadCase<SPSCD2HQueue, kThreadPutSignal>(c, "thr_small_spsc", 24, kSpscQueues, 256,
                                                         true);
  fails +=
      RunThreadCase<MPSCD2HQueue, kThreadPutSignal>(c, "thr_small_mpsc", 25, kMaxUnits, 256, true);
  // putValue (1/2/4/8 bytes, with and without SignalInc).
  fails += RunValueCase<SPSCD2HQueue, ccoCoopWarp, false>(c, "put_value", 32, kSpscQueues);
  fails += RunValueCase<MPSCD2HQueue, ccoCoopWarp, false>(c, "mpsc_put_value", 33, kMaxUnits);
  fails += RunStressCase(c);
  fails += RunSignalCase(c);
  fails += RunHandoffCase(c);
  fails += RunBlockFlushCase<false>(c, "block_flush", 40);
  fails += RunBlockFlushCase<true>(c, "block_flush_async", 41);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutPieces>(c, "small_spsc", 14, kChunk,
                                                             kSpscQueues, 64, &c.dcSmall);
  // kQpsPerPeer QPs per (thread, peer): each of a thread's 3 queues on its own QP.
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "qps_put_sig", 29, kChunk,
                                                             kSpscQueues, 64, &c.dcQps);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutFlushClobber>(c, "qps_flush", 30, kChunk,
                                                                    kSpscQueues, 256, &c.dcQps);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutFlushClobber>(c, "qps_mpsc_flush", 31, kChunk,
                                                                    kMaxUnits, 256, &c.dcQps);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopWarp, kPutPieces>(c, "small_mpsc", 15, kChunk, kMaxUnits,
                                                             64, &c.dcSmall);

  // Negative cases, each on a DevComm of its own; then dc must still work.
  bool anyNegative = false;
  if (NegativeSelected("n1")) {
    anyNegative = true;
    fails += RunFailCase(c, "n1_bad_raddr", reqs, /*selfPut=*/false);
  }
  if (NegativeSelected("n2")) {
    anyNegative = true;
    fails += RunFailCase(c, "n2_no_qp", reqs, /*selfPut=*/true);
  }
  if (NegativeSelected("n3")) {
    anyNegative = true;
    fails += RunTimeoutCase<SPSCD2HQueue, ccoCoopBlock>(c, "n3_timeout", reqs, kSpscQueues, 256);
    fails += RunTimeoutCase<MPSCD2HQueue, ccoCoopWarp>(c, "n3_mpsc_timeout", reqs, kMaxUnits, 64);
  }
  if (NegativeSelected("n4")) {
    anyNegative = true;
    fails += RunFailConnectCase(c, reqs);
  }
  if (NegativeSelected("n5")) {
    anyNegative = true;
    fails += RunQpsMismatchCase(c, reqs);
  }
  if (anyNegative) {
    fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "put_sig_after", 26, kChunk,
                                                               kSpscQueues, 64);
  }

  HIP_CHECK(hipStreamDestroy(c.stream));
  HIP_CHECK(hipFree(c.dErr));
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
  ccoCommDestroy(c.comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  // MORI_CCO_SKIP_GDA_FULL marks runners without intranode cross-rail RDMA; the
  // single-node (fork) run needs FULL connections, so skip it there.
  bool singleRank = false;
  for (int i = 1; i < argc; i++) singleRank |= !strcmp(argv[i], "--rank");
  if (getenv("MORI_CCO_SKIP_GDA_FULL") != nullptr && !singleRank) {
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
