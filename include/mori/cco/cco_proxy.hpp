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
//
// cco_proxy.hpp — ccoProxy: host-proxy RDMA transport with the ccoGda window API.
//
// Kernels invoke host callbacks through KIWI invoke (the GPU->CPU callback
// layer on the KIWI device-to-host queues): each op is one fixed-size
// descriptor; host proxy threads run the callbacks, which post ibverbs work
// requests on host-created RC QPs, proxyQpsPerPeer per (proxy thread, peer),
// on the PD that owns the cco window MRs. Keys and remote addresses are
// resolved on the GPU exactly like ccoGda (raddr = window offset + iovaBias),
// so the proxy needs no key tables.
//
// Queues: ccoProxy<SPSCD2HQueue> (default) indexes the DevComm's SPSC pool,
// ccoProxy<MPSCD2HQueue> its MPSC pool; each pool is one KIWI invoke context.
// An MPSC queueIndex is taken modulo the pool size, like ccoGda's
// contextIndex; an SPSC one must be in [0, proxySpscQueueCount), since
// wrapping would let two wavefronts share an SPSC queue and corrupt it.
// Only one wavefront may use an SPSC queue at a time: its producer state lives
// in the context's device handle, loaded by the constructor and saved back by
// the destructor (or release()). Handing an SPSC queue to another wavefront of
// the same kernel needs a release/acquire between them: release() (or the
// destructor) then an agent-scope release (e.g. a flag store) in the old owner,
// and an agent-scope acquire of it before the new owner constructs its
// ccoProxy (__syncthreads() suffices within one block). Across kernel launches
// stream order is enough.
//
// Usage errors trap instead of corrupting a queue or faulting: an empty pool,
// an SPSC queueIndex out of range, a push from a partial wavefront and (builds
// without NDEBUG) differing arguments across the issuing lanes. Builds without
// NDEBUG print the reason first; release builds only trap, since an inlined
// device printf would add registers to every kernel using ccoProxy.
//
// Ordering: ops pushed by one wavefront to one queue are popped in program
// order and land on one QP per peer, so a put followed by a signal (or a put
// carrying a signal) to the same peer through the same queue is delivered in
// order -- the same guarantee ccoGda gives within one context. Different queues
// are unordered, as different ccoGda contexts are.
//
// Proxy QPs go to the peers of proxyConnectionType (ccoDevCommRequirements),
// independent of the GDA QPs. Signals land in the ccoGda signal slots
// (devComm.ibgda.signalBuf), which exist when the GDA or the proxy connection
// resolves to non-NONE. SignalInc / SignalAdd / WindowSignalAdd are NIC atomic
// fetch-adds; SignalSet / WindowSignalSet are 8-byte inline RDMA writes for NICs
// whose atomics into GPU memory fail (see ccoGda_SignalSet), and need window MRs
// without relaxed ordering.
//
// Flush: the proxy satisfies each queue's FLUSHes in queue order and adds 1 to
// that queue's flushDone per FLUSH satisfied (it is the only writer). A flush
// takes a ticket t from the queue's flushTicket counter right before pushing
// and waits for flushDone >= t, so it never needs its FLUSH's ring position
// and an earlier FLUSH need not have been waited for.
//
// Failure: the DevComm fails for good at the first proxy error (see
// ccoDevCommGetError): the proxy drops every later op and sets
// ccoProxyFailedBit in each queue's flushDone. failed() then returns true,
// flush(coop, timeoutCycles) and waitSignal(..., timeoutCycles) return 2, an
// SPSC push waiting for ring space drops its op, and flush() / waitSignal()
// trap (waitSignal within 256 polls). A trap, like any usage-error trap, ends
// the process: HIP reports the queue error and aborts. Only this rank sees the
// failure; peers are not told, so a receiver needs its own timeout
// (waitSignal(..., timeoutCycles)). Recovery: detect it (2, failed(), or
// ccoDevCommGetError), agree out of band, then every rank calls
// ccoDevCommDestroy (collective) and creates a new DevComm. Not covered: an
// MPSC push waiting for ring space (only after a corrupted ring).
//
// Fences: each put / putValue / signal does a system-scope release fence
// before its push, so the data it sends (and every earlier store) is visible
// to the NIC. Reads of NIC-written memory need an acquire: waitSignal() and
// flush() end with a system-scope one; after readSignal() or a poll of a
// caller window, issue it yourself.
//
// Calling convention: put / putValue / signal / flush are called by the whole
// Coop group from converged code, and the issuing wavefront (the calling
// wavefront for ccoCoopWarp and ccoCoopLaneMasked, wavefront 0 -- linear
// thread ids 0..63, also in 2-D and 3-D blocks -- for ccoCoopBlock) must have
// all 64 lanes active: a KIWI push is wavefront-collective, so a
// ccoCoopLaneMasked call from a divergent branch traps too. put / putValue /
// signal take no default Coop (ccoGda's default is ccoCoopThread), and
// ccoCoopThread is rejected at compile time. With ccoCoopWarp / ccoCoopBlock
// the group issues one op and every issuing lane must pass identical arguments
// (lane i writes word i of the descriptor; only builds without NDEBUG check
// it); ccoGda reads only thread_rank 0's.
// ccoCoopLaneMasked{active} (put / putValue / signal) issues one op per active
// lane, all in one lane-masked push, so they enter the queue in lane order; a
// put larger than maxPutBytes is split as with the other Coops, the wavefront
// pushing while any lane has pieces left. ThreadMode has no effect. After
// lane-masked ops, call flush() (ccoCoopWarp) from all 64 lanes.
//
// Requires a BUILD_CCO_PROXY build and a wave64 GFX9 target (KIWI push).

#pragma once

#include "mori/cco/cco_scale_out.hpp"

#if defined(__HIPCC__)
#include "mori/cco/proxy/invoke/device_context.hip.hpp"
#include "mori/cco/proxy/queue/device_to_host_queue.hip.hpp"
#include "mori/cco/proxy/queue/mpsc_device_to_host_queue.hip.hpp"
#endif

namespace mori {
namespace cco {

/* ────────────────────────────────────────────────────────────────────────────
 *  Wire format (shared by the device API and the host proxy engine)
 *
 *  Each op is one KIWI invoke callback: the op is the callback id and the
 *  descriptor below its single argument (FLUSH has none). The host engine
 *  registers the callbacks in enum order, so the ids match; the op enum stays
 *  open for later ops.
 * ──────────────────────────────────────────────────────────────────────────── */

enum ccoProxyOp : uint8_t {
  ccoProxyOpPut = 0,             // ccoProxyPutDesc: RDMA write
  ccoProxyOpPutSignal = 1,       // ccoProxyPutSignalDesc: RDMA write + signal, same QP
  ccoProxyOpSignal = 2,          // ccoProxySignalDesc: signal
  ccoProxyOpFlush = 3,           // no arguments: flushDone += 1 once satisfied
  ccoProxyOpPutValue = 4,        // ccoProxyPutValueDesc: inline RDMA write
  ccoProxyOpPutValueSignal = 5,  // ccoProxyPutValueSignalDesc: inline RDMA write + signal
  ccoProxyOpCount
};

struct ccoProxyPutDesc {
  uint64_t laddr;  // local MR address: window offset + iovaBias
  uint64_t raddr;  // remote MR address: window offset + iovaBias
  uint32_t lkey;
  uint32_t rkey;
  uint32_t bytes;
  uint32_t peer;  // world rank
};

struct ccoProxySignalDesc {
  uint64_t raddr;  // remote signal slot (MR address, as raddr above)
  uint64_t value;  // atomic add operand, or the value written (set)
  uint32_t rkey;
  uint32_t peer;  // world rank
  uint32_t set;   // 0: atomic fetch-add; 1: 8-byte inline write of value
  uint32_t pad;
};

struct ccoProxyPutSignalDesc {
  ccoProxyPutDesc put;
  ccoProxySignalDesc signal;  // signal.peer == put.peer
};

struct ccoProxyPutValueDesc {
  uint64_t raddr;  // remote MR address: window offset + iovaBias
  uint64_t value;  // the low `bytes` bytes are written (little endian)
  uint32_t rkey;
  uint32_t bytes;  // 1..8
  uint32_t peer;   // world rank
  uint32_t pad;
};

struct ccoProxyPutValueSignalDesc {
  ccoProxyPutValueDesc put;
  ccoProxySignalDesc signal;  // signal.peer == put.peer
};

static_assert(sizeof(ccoProxyPutDesc) == 32, "one queue line");
static_assert(sizeof(ccoProxySignalDesc) == 32, "one queue line");
static_assert(sizeof(ccoProxyPutSignalDesc) == 64, "two queue lines");
static_assert(sizeof(ccoProxyPutValueDesc) == 32, "one queue line");
static_assert(sizeof(ccoProxyPutValueSignalDesc) == 64, "two queue lines");

// Upper bound on the bytes of one RDMA write a descriptor carries. The engine
// lowers it to the port's max_msg_sz (ccoProxyContext::maxPutBytes), and put()
// splits larger transfers into pieces of that size.
inline constexpr uint64_t ccoProxyMaxPutBytes = 1ull << 30;
static_assert(ccoProxyMaxPutBytes <= UINT32_MAX, "ccoProxyContext::maxPutBytes is 32-bit");

// Set in every flushDone word once the DevComm has failed (see
// ccoDevCommGetError); the low bits keep the satisfied-FLUSH count.
inline constexpr uint64_t ccoProxyFailedBit = 1ull << 63;

#if defined(__HIPCC__)

// Usage error or failed DevComm: print the message from the lanes where `who`
// holds (builds without NDEBUG only; see the file comment), then trap.
#ifndef NDEBUG
#define CCO_PROXY_TRAP(who, ...)  \
  do {                            \
    if (who) printf(__VA_ARGS__); \
    __builtin_trap();             \
  } while (0)
#else
#define CCO_PROXY_TRAP(who, ...) __builtin_trap()
#endif

namespace impl {

// Linear thread id in the block (x fastest): wavefront w holds ids
// [64w, 64w + 64) in 1-D, 2-D and 3-D blocks alike.
__device__ inline unsigned proxyLinearTid() {
  return __builtin_amdgcn_workitem_id_x() +
         __builtin_amdgcn_workgroup_size_x() *
             (__builtin_amdgcn_workitem_id_y() +
              __builtin_amdgcn_workgroup_size_y() * __builtin_amdgcn_workitem_id_z());
}

// Resolve a remote action to a signal descriptor, as ccoGda::put does.
template <typename RemoteAction>
__device__ inline ccoProxySignalDesc proxyResolveSignal(ccoDevComm const& comm, int worldPeer,
                                                        const RemoteAction& action) {
  ccoProxySignalDesc s{};
  s.peer = static_cast<uint32_t>(worldPeer);
  if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
    s.raddr = static_cast<uint64_t>(action.signalId) * sizeof(uint64_t);
    s.rkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
    s.value = 1;
  } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
    s.raddr = static_cast<uint64_t>(action.signalId) * sizeof(uint64_t);
    s.rkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
    s.value = action.value;
  } else if constexpr (std::is_same_v<RemoteAction, ccoGda_WindowSignalAdd>) {
    // Caller's window: its MR starts at iova iovaBias, as in ccoGda.
    const ccoWindowDevice* w = reinterpret_cast<ccoWindowDevice*>(action.win);
    s.raddr = action.offset + w->ibgdaWin.iovaBias;
    s.rkey = w->ibgdaWin.peerRkeys[worldPeer];
    s.value = action.value;
  } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalSet>) {
    s.raddr = static_cast<uint64_t>(action.signalId) * sizeof(uint64_t);
    s.rkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
    s.value = action.value;
    s.set = 1;
  } else if constexpr (std::is_same_v<RemoteAction, ccoGda_WindowSignalSet>) {
    const ccoWindowDevice* w = reinterpret_cast<ccoWindowDevice*>(action.win);
    s.raddr = action.offset + w->ibgdaWin.iovaBias;
    s.rkey = w->ibgdaWin.peerRkeys[worldPeer];
    s.value = action.value;
    s.set = 1;
  } else {
    static_assert(std::is_same_v<RemoteAction, ccoGda_NoSignal>,
                  "ccoProxy supports ccoGda_NoSignal / SignalInc / SignalAdd / WindowSignalAdd / "
                  "SignalSet / WindowSignalSet");
  }
  return s;
}

// Traps on device targets that cannot push (not wave64 gfx942/gfx950);
// ccoDevCommCreate rejects proxy queues on such GPUs.
__device__ inline void proxyCheckTarget() {
#ifdef MORI_CCO_PROXY_PUSH_UNSUPPORTED
  CCO_PROXY_TRAP(proxyLinearTid() == 0,
                 "ccoProxy: unsupported GPU target; ccoProxy needs a wave64 gfx942/gfx950 GPU\n");
#endif
}

// Whether this is the lowest active lane: the one lane that reports.
__device__ inline bool proxyFirstLane(uint64_t lanes = __builtin_amdgcn_read_exec()) {
  return __lane_id() == static_cast<unsigned>(__builtin_ctzll(lanes));
}

}  // namespace impl

// Device-side transport handle. Construct per thread; call put / signal /
// flush from the whole Coop group. Not copyable: an SPSC handle writes its
// producer state back exactly once.
template <typename Queue = mori::cco::proxy::queue::SPSCD2HQueue>
struct ccoProxy {
  static constexpr bool kSpsc = std::is_same_v<Queue, mori::cco::proxy::queue::SPSCD2HQueue>;
  static_assert(kSpsc || std::is_same_v<Queue, mori::cco::proxy::queue::MPSCD2HQueue>,
                "ccoProxy queue must be mori::cco::proxy::queue::SPSCD2HQueue or MPSCD2HQueue");

  using DeviceContext = mori::cco::proxy::invoke::DeviceContext<Queue>;

  ccoDevComm const& comm;
  int rank;    // my index in the proxy's GDA team (comm.proxy.connType) [0, nRanks)
  int nRanks;  // that team's size
  uint32_t queueId;
  uint64_t maxPutBytes;   // largest RDMA write one descriptor carries
  DeviceContext* ctx;     // this pool's invoke context
  uint64_t* flushDone;    // this queue's satisfied-FLUSH count (fine-grained device memory)
  uint64_t* flushTicket;  // this queue's flush ticket counter (device memory)
  bool dirty;             // SPSC: pushed since the handle was loaded / saved
  bool unflushed;         // pushed since the last completed flush (or construction)
  typename DeviceContext::DeviceHandle handle;

  // SPSC: queueIndex must be in [0, numSpscQueues); MPSC: taken modulo the pool
  // size, like ccoGda's contextIndex. Traps if this pool is empty or (SPSC) the
  // index is out of range.
  __device__ inline ccoProxy(ccoDevComm const& comm_, int queueIndex);
  __device__ inline ~ccoProxy() { release(); }
  ccoProxy(const ccoProxy&) = delete;
  ccoProxy& operator=(const ccoProxy&) = delete;

  // SPSC: save the handle (producer state) back so the next ccoProxy on this
  // queue (later in the kernel, after the release/acquire described in the
  // file comment, or in a later launch) continues from it. Called by the
  // destructor; a no-op for MPSC and for wavefronts that never pushed.
  __device__ inline void release() {
    if constexpr (kSpsc) {
      if (dirty) ctx->save_device_handle(queueId, handle);
      dirty = false;
    }
  }

  template <ccoTeamMode TeamMode>
  __device__ inline int resolveWorldPeer(int peer) const {
    if constexpr (TeamMode == CCO_TEAM_WORLD) {
      return peer;
    } else {
      return impl::GdaPeerToWorld(comm, comm.proxy.connType, peer);
    }
  }

  // RDMA write of [srcWin+srcOffset, +bytes) to peer's [dstWin+dstOffset, ...),
  // optionally followed (same QP, so after the data) by a remote signal.
  // Template parameters follow ccoGda::put; ThreadMode has no effect. Coop has
  // no default (ccoGda's is ccoCoopThread): pass ccoCoopWarp, ccoCoopBlock, or
  // ccoCoopLaneMasked (per-lane puts; see the file comment).
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent, typename RemoteAction,
            typename Coop>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes, RemoteAction remoteAction, Coop coop);

  // Inline RDMA write of `value` (sizeof(T) <= 8 bytes, no padding) to peer's
  // [dstWin+dstOffset, ...), optionally followed by a remote signal, as put().
  // Template parameters follow ccoGda::putValue; Coop as in put().
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent, typename T,
            typename RemoteAction, typename Coop>
  __device__ inline void putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction, Coop coop);

  // Remote signal only. Coop as in put().
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename RemoteAction, typename Coop>
  __device__ inline void signal(int peer, RemoteAction remoteAction, Coop coop);

  // Push a FLUSH and wait until every work request this queue posted before it
  // has completed, on every peer: the sources of earlier puts are reusable
  // (see the file comment). Waits only for this queue; with ccoCoopBlock, also
  // for the queue of every wavefront whose handle pushed since its last
  // completed flush. Coop: ccoCoopWarp or ccoCoopBlock (after lane-masked ops,
  // flush() from all 64 lanes).
  template <typename Coop = ccoCoopWarp, std::enable_if_t<!std::is_integral_v<Coop>, int> = 0>
  __device__ inline void flush(Coop coop = Coop{});

  // flush() that never traps: 0 = done, 1 = timed out after timeoutCycles
  // (impl::clock64 cycles; earlier puts may still be in flight, and a later
  // flush covers them), 2 = the DevComm failed. Every thread of the Coop gets
  // the same value (ccoCoopBlock: 2 if any flushing wavefront got 2, else 1 if
  // any got 1).
  template <typename Coop = ccoCoopWarp, std::enable_if_t<!std::is_integral_v<Coop>, int> = 0>
  __device__ inline int flush(Coop coop, uint64_t timeoutCycles);

  // Whether the DevComm has failed (one load).
  __device__ inline bool failed() const {
    return (__hip_atomic_load(flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) &
            ccoProxyFailedBit) != 0;
  }

  // v1: identical to flush() -- covers every peer.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename Coop = ccoCoopWarp>
  __device__ inline void flush(int peer, Coop coop = Coop{}) {
    (void)peer;
    flush(coop);
  }

  // Local signal slots: identical to ccoGda (remote signals land there).
  __device__ inline uint64_t readSignal(ccoGdaSignal_t signalId, int bits = 64) {
    return impl::readSignalImpl<CCO_GDA_BUILD_PROVIDER>(comm.ibgda.signalBuf,
                                                        comm.ibgda.signalShadows, signalId, bits);
  }
  // As ccoGda::waitSignal: waits until `least` more than already consumed have
  // arrived, then consumes them (readSignal returns the unconsumed amount); one
  // waiter per slot at a time. Also traps once the DevComm failed, and acquires.
  // One thread of the group waits (lane 0 / linear thread 0 for ccoCoopWarp /
  // ccoCoopBlock, also in 2-D and 3-D blocks). Full 64-bit slots (no `bits`).
  template <typename Coop = ccoCoopThread>
  __device__ inline void waitSignal(ccoGdaSignal_t signalId, uint64_t least, Coop coop = Coop{}) {
    coop.sync();
    if (groupLeader(coop) && waitSignalLeader</*kTimeout=*/false>(signalId, least, 0) != 0) {
      CCO_PROXY_TRAP(true,
                     "ccoProxy: waitSignal: the DevComm failed; see the host's stderr or "
                     "ccoDevCommGetError\n");
    }
    coop.sync();
  }

  // waitSignal() that never traps: 0 = consumed, 1 = timed out after
  // timeoutCycles (impl::clock64 cycles), 2 = the DevComm failed; nothing is
  // consumed unless it returns 0. Every thread of a ccoCoopWarp / ccoCoopBlock
  // group gets the same value.
  template <typename Coop>
  [[nodiscard]] __device__ inline int waitSignal(ccoGdaSignal_t signalId, uint64_t least, Coop coop,
                                                 uint64_t timeoutCycles);
  __device__ inline void resetSignal(ccoGdaSignal_t signalId) {
    impl::resetSignalImpl<CCO_GDA_BUILD_PROVIDER>(comm.ibgda.signalBuf, comm.ibgda.signalShadows,
                                                  signalId);
  }

 private:
  template <typename Coop>
  __device__ static inline bool issuingWave() {
    static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                  "ccoProxy put/signal/flush need ccoCoopWarp or ccoCoopBlock: a KIWI queue push "
                  "is wavefront-collective. For per-thread put/signal, call them from all 64 "
                  "lanes (converged) with ccoCoopLaneMasked{active}.");
    static_assert(std::is_same_v<Coop, ccoCoopThread> || std::is_same_v<Coop, ccoCoopWarp> ||
                      std::is_same_v<Coop, ccoCoopBlock> || std::is_same_v<Coop, ccoCoopLaneMasked>,
                  "ccoProxy supports ccoCoopWarp, ccoCoopBlock and (put/signal) ccoCoopLaneMasked");
    if constexpr (std::is_same_v<Coop, ccoCoopBlock>) {
      // Wavefront 0, by linear id: in a 2-D block threadIdx.x < 64 spans
      // several wavefronts. (Not impl::warpSize(): core/utils.hpp, pulled in
      // by cco_scale_out.hpp, defines a `warpSize` macro.)
      return impl::proxyLinearTid() < static_cast<unsigned>(__builtin_amdgcn_wavefrontsize());
    } else {
      return true;  // the calling wavefront
    }
  }

  // Whether this wavefront flushes for a flush(Coop): wavefront 0 of a block,
  // plus any other wavefront that pushed since its last completed flush.
  template <typename Coop>
  __device__ inline bool flushingWave() const {
    if constexpr (std::is_same_v<Coop, ccoCoopBlock>) {
      return issuingWave<Coop>() || unflushed;  // wave-uniform
    } else {
      return issuingWave<Coop>();
    }
  }

  // The one thread of the group that waits: ccoCoopWarp and ccoCoopBlock
  // count lanes / threads by threadIdx.x only, wrong in 2-D blocks.
  template <typename Coop>
  __device__ static inline bool groupLeader(const Coop& coop) {
    if constexpr (std::is_same_v<Coop, ccoCoopBlock>) {
      return impl::proxyLinearTid() == 0;
    } else if constexpr (std::is_same_v<Coop, ccoCoopWarp>) {
      return __lane_id() == 0;
    } else {
      return coop.thread_rank() == 0;
    }
  }

  // Whether this lane issues an op: coop.active for ccoCoopLaneMasked; true
  // for the group Coops, whose group issues one op.
  template <typename Coop>
  __device__ static inline bool laneActive(const Coop& coop) {
    if constexpr (std::is_same_v<Coop, ccoCoopLaneMasked>) {
      return coop.active;
    } else {
      return true;
    }
  }

  // Loop condition of put()'s split: wave-uniform. Lane-masked: true while any
  // lane has `more`.
  template <bool kLaneMasked>
  __device__ static inline bool anyLane(bool more) {
    if constexpr (kLaneMasked) {
      return __ballot(more) != 0;
    } else {
      return more;  // already identical in every issuing lane
    }
  }

  // Invoke callback `op` with its descriptor, if any (whole wavefront):
  // kLaneMasked (ccoCoopLaneMasked): every lane with `active` sends its own
  // descriptor; otherwise the wavefront sends one, and `active` is true. An
  // SPSC push on a full ring retries (a lane-masked push writes all of the
  // active lanes' messages or none), and drops the op once the DevComm failed;
  // an MPSC push waits inside KIWI (and always succeeds). Traps on a partial
  // wavefront, which KIWI does not check in release builds, and (builds without
  // NDEBUG, not kLaneMasked) on lanes passing differing descriptors.
  template <bool kLaneMasked, typename... Desc>
  __device__ inline void invoke([[maybe_unused]] bool active, ccoProxyOp op, const Desc&... desc) {
    impl::proxyCheckTarget();  // before the exec check, which misfires on wave32
    const uint64_t exec = __builtin_amdgcn_read_exec();
    if (__builtin_expect(exec != ~uint64_t{0}, 0)) {
      CCO_PROXY_TRAP(impl::proxyFirstLane(exec),
                     "ccoProxy: put/signal/flush called with a partial wavefront (exec 0x%llx); "
                     "all 64 lanes of the issuing wavefront must call it\n",
                     static_cast<unsigned long long>(exec));
    }
#ifndef NDEBUG  // see the file comment
    if constexpr (!kLaneMasked) (checkUniformDesc(op, desc), ...);
#endif
    auto push = [&] {
      if constexpr (kLaneMasked) {
        return handle.invoke_lane_masked(op, active, desc...);
      } else {
        return handle.invoke(op, desc...);
      }
    };
    if constexpr (kSpsc) {
      while (!push()) {
        // A failed proxy may have stopped popping: drop the op.
        if (__builtin_amdgcn_readfirstlane(failed() ? 1 : 0)) return;
        __builtin_amdgcn_s_sleep(1);
      }
      dirty = true;
    } else {
      push();
    }
  }

#ifndef NDEBUG
  // Trap unless every lane of the (full) issuing wavefront passes lane 0's
  // descriptor: lane i pushes 8-byte word i, so a mismatch would mix lanes.
  template <typename Desc>
  __device__ static inline void checkUniformDesc(ccoProxyOp op, const Desc& desc) {
    static_assert(
        sizeof(Desc) % sizeof(uint64_t) == 0 && std::has_unique_object_representations_v<Desc>,
        "descriptor must be whole 8-byte words without padding");
    constexpr int kWords = sizeof(Desc) / sizeof(uint64_t);
    uint64_t w[kWords];
    __builtin_memcpy(w, &desc, sizeof(Desc));
    int badWord = -1;  // first descriptor word differing from lane 0's
    for (int i = 0; i < kWords && badWord < 0; ++i) {
      const uint32_t lo = static_cast<uint32_t>(w[i]);
      const uint32_t hi = static_cast<uint32_t>(w[i] >> 32);
      if (lo != __builtin_amdgcn_readfirstlane(lo) || hi != __builtin_amdgcn_readfirstlane(hi)) {
        badWord = i;
      }
    }
    const uint64_t badLanes = __ballot(badWord >= 0);
    if (__builtin_expect(badLanes != 0, 0)) {
      CCO_PROXY_TRAP(impl::proxyFirstLane(badLanes),
                     "ccoProxy: op %u: lane %u passed arguments that differ from lane 0's "
                     "(descriptor word %d); every issuing lane must pass identical arguments\n",
                     static_cast<unsigned>(op), __lane_id(), badWord);
    }
  }
#endif

  // Push a FLUSH and wait for it (issuing wavefront, all 64 lanes). Returns 0
  // done, 1 timed out (kTimeout only), 2 failed; the same in every lane.
  template <bool kTimeout>
  __device__ inline int flushIssue(uint64_t timeoutCycles) {
    return flushWait<kTimeout>(flushDone, flushPush(), timeoutCycles);
  }

  // Take a ticket and push a FLUSH (issuing wavefront, all 64 lanes). Returns,
  // in lane 0, the flushDone value that proves it satisfied.
  __device__ inline uint64_t flushPush();

  // Lane 0 waits until *done >= target, then the wavefront acquires. Returns
  // as flushIssue.
  template <bool kTimeout>
  __device__ static inline int flushWait(uint64_t* done, uint64_t target, uint64_t timeoutCycles);

  // The group leader's wait. Returns 0 consumed, 1 timed out (kTimeout only),
  // 2 failed.
  template <bool kTimeout>
  __device__ inline int waitSignalLeader(ccoGdaSignal_t signalId, uint64_t least,
                                         uint64_t timeoutCycles);

  // invoke() for a data-carrying op: first make the source data (and any
  // earlier stores the peer will read) visible to the NIC, since the proxy may
  // post the WR as soon as the descriptor lands. Release only: nothing here
  // reads NIC-written memory (waitSignal and flush acquire).
  template <bool kLaneMasked, typename Desc>
  __device__ inline void fenceAndInvoke(bool active, ccoProxyOp op, const Desc& desc) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    invoke<kLaneMasked>(active, op, desc);
    unflushed = true;
  }
};

template <typename Queue>
__device__ inline ccoProxy<Queue>::ccoProxy(ccoDevComm const& comm_, int queueIndex)
    : comm(comm_),
      maxPutBytes(comm_.proxy.maxPutBytes),
      ctx(nullptr),
      flushDone(nullptr),
      flushTicket(nullptr),
      dirty(false),
      unflushed(false) {
  impl::proxyCheckTarget();
  impl::GdaTeamRankSize(comm, comm.proxy.connType, rank, nRanks);
  const int poolSize = kSpsc ? comm.proxy.numSpscQueues : comm.proxy.numMpscQueues;
  if (__builtin_expect(poolSize <= 0, 0)) {
    CCO_PROXY_TRAP(impl::proxyLinearTid() == 0,
                   "ccoProxy: the DevComm has no %s proxy queues (set %s in "
                   "ccoDevCommRequirements)\n",
                   kSpsc ? "SPSC" : "MPSC", kSpsc ? "proxySpscQueueCount" : "proxyMpscQueueCount");
  }
  if constexpr (kSpsc) {
    // No wrap: two wavefronts sharing an SPSC queue would write the same ring
    // lines and corrupt it. One unsigned compare also catches a negative index.
    if (__builtin_expect(static_cast<uint32_t>(queueIndex) >= static_cast<uint32_t>(poolSize), 0)) {
      CCO_PROXY_TRAP(impl::proxyFirstLane(),
                     "ccoProxy: SPSC queueIndex %d out of range [0, %d) (proxySpscQueueCount)\n",
                     queueIndex, poolSize);
    }
    queueId = static_cast<uint32_t>(queueIndex);
  } else {
    queueId = static_cast<uint32_t>(queueIndex) % static_cast<uint32_t>(poolSize);
  }
  ctx = static_cast<DeviceContext*>(kSpsc ? comm.proxy.spscContext : comm.proxy.mpscContext);
  handle = ctx->get_device_handle(queueId);
  // flushDone and flushTicket: SPSC pool, then MPSC pool.
  const int pool = kSpsc ? 0 : comm.proxy.numSpscQueues;
  flushDone = comm.proxy.flushDone + pool + queueId;
  flushTicket = comm.proxy.flushTicket + pool + queueId;
}

template <typename Queue>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename RemoteAction, typename Coop>
__device__ inline void ccoProxy<Queue>::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                            ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                            RemoteAction remoteAction, Coop coop) {
  constexpr bool kLaneMasked = std::is_same_v<Coop, ccoCoopLaneMasked>;
  coop.sync();
  const bool on = laneActive(coop);  // constant true unless kLaneMasked
  if (issuingWave<Coop>()) {
    int worldPeer = 0;
    ccoProxyPutDesc d{};
    if (on) {  // an inactive lane's arguments are not read
      worldPeer = resolveWorldPeer<TeamMode>(peer);
      const ccoWindowDevice* sw = reinterpret_cast<ccoWindowDevice*>(srcWin);
      const ccoWindowDevice* dw = reinterpret_cast<ccoWindowDevice*>(dstWin);
      d.lkey = sw->ibgdaWin.lkey;
      d.rkey = dw->ibgdaWin.peerRkeys[worldPeer];
      d.laddr = srcOffset + sw->ibgdaWin.iovaBias;  // MR iova of window offset 0
      d.raddr = dstOffset + dw->ibgdaWin.iovaBias;
      d.peer = static_cast<uint32_t>(worldPeer);
    } else {
      bytes = 0;
    }
    // Split transfers one WR cannot carry; the signal rides the last piece.
    // Lane-masked: the wavefront pushes while any lane has more than a piece
    // left, and only those lanes take part.
    while (anyLane<kLaneMasked>(bytes > maxPutBytes)) {
      const bool piece = !kLaneMasked || bytes > maxPutBytes;
      d.bytes = static_cast<uint32_t>(maxPutBytes);
      fenceAndInvoke<kLaneMasked>(piece, ccoProxyOpPut, d);
      if (piece) {
        d.laddr += maxPutBytes;
        d.raddr += maxPutBytes;
        bytes -= maxPutBytes;
      }
    }
    d.bytes = static_cast<uint32_t>(bytes);
    if constexpr (std::is_same_v<RemoteAction, ccoGda_NoSignal>) {
      fenceAndInvoke<kLaneMasked>(on, ccoProxyOpPut, d);
    } else {
      ccoProxyPutSignalDesc ds{};
      ds.put = d;
      if (on) ds.signal = impl::proxyResolveSignal(comm, worldPeer, remoteAction);
      fenceAndInvoke<kLaneMasked>(on, ccoProxyOpPutSignal, ds);
    }
  }
  coop.sync();
}

template <typename Queue>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename T, typename RemoteAction,
          typename Coop>
__device__ inline void ccoProxy<Queue>::putValue(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                                 T value, RemoteAction remoteAction, Coop coop) {
  static_assert(sizeof(T) >= 1 && sizeof(T) <= 8 && std::is_trivially_copyable_v<T>,
                "putValue only supports trivially copyable types of 1..8 bytes");
  // Padding bytes would be sent, and fail the uniform-argument check.
  static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_pointer_v<T> ||
                    std::has_unique_object_representations_v<T>,
                "putValue does not take types with padding bytes");
  constexpr bool kLaneMasked = std::is_same_v<Coop, ccoCoopLaneMasked>;
  coop.sync();
  const bool on = laneActive(coop);  // constant true unless kLaneMasked
  if (issuingWave<Coop>()) {
    int worldPeer = 0;
    ccoProxyPutValueDesc d{};
    if (on) {  // an inactive lane's arguments are not read
      worldPeer = resolveWorldPeer<TeamMode>(peer);
      const ccoWindowDevice* dw = reinterpret_cast<ccoWindowDevice*>(dstWin);
      d.raddr = dstOffset + dw->ibgdaWin.iovaBias;
      d.rkey = dw->ibgdaWin.peerRkeys[worldPeer];
      __builtin_memcpy(&d.value, &value, sizeof(T));
      d.bytes = sizeof(T);
      d.peer = static_cast<uint32_t>(worldPeer);
    }
    if constexpr (std::is_same_v<RemoteAction, ccoGda_NoSignal>) {
      fenceAndInvoke<kLaneMasked>(on, ccoProxyOpPutValue, d);
    } else {
      ccoProxyPutValueSignalDesc ds{};
      ds.put = d;
      if (on) ds.signal = impl::proxyResolveSignal(comm, worldPeer, remoteAction);
      fenceAndInvoke<kLaneMasked>(on, ccoProxyOpPutValueSignal, ds);
    }
  }
  coop.sync();
}

template <typename Queue>
template <ccoTeamMode TeamMode, typename RemoteAction, typename Coop>
__device__ inline void ccoProxy<Queue>::signal(int peer, RemoteAction remoteAction, Coop coop) {
  static_assert(!std::is_same_v<RemoteAction, ccoGda_NoSignal>, "signal() needs a remote action");
  constexpr bool kLaneMasked = std::is_same_v<Coop, ccoCoopLaneMasked>;
  coop.sync();
  const bool on = laneActive(coop);  // constant true unless kLaneMasked
  if (issuingWave<Coop>()) {
    ccoProxySignalDesc s{};
    if (on) s = impl::proxyResolveSignal(comm, resolveWorldPeer<TeamMode>(peer), remoteAction);
    fenceAndInvoke<kLaneMasked>(on, ccoProxyOpSignal, s);
  }
  coop.sync();
}

template <typename Queue>
__device__ inline uint64_t ccoProxy<Queue>::flushPush() {
  // Lane 0 picks the flushDone value that proves our FLUSH (or a later one
  // that covers it) is satisfied.
  uint64_t target = 0;
  if (__lane_id() == 0) {
    if constexpr (kSpsc) {
      // Tickets follow FLUSH order in the ring: one producer at a time, and a
      // handoff's release/acquire orders the old owner's tickets first.
      target = __hip_atomic_fetch_add(flushTicket, uint64_t{1}, __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_AGENT) +
               1;
    } else {
      // Ticket t: we wait for flushDone >= t, i.e. for the first t FLUSHes of
      // this queue (in queue order) to be satisfied. Those t FLUSHes hold t
      // distinct tickets, so at least one of them holds a ticket t' >= t: ours
      // (t' == t), or one whose caller took its ticket after us (t' > t).
      // That FLUSH was claimed after our puts, so it covers them: the proxy
      // pops the queue in order, so it has posted our puts by the time it
      // pops that FLUSH, and the FLUSH waits for the queue's last WR per QP
      // as of that pop.
      //
      // Why that FLUSH was claimed after our puts: every ticket fetch-add is
      // acq_rel (agent scope: all producers of a queue run on this GPU). The
      // later caller's fetch-add reads ours (or a later RMW of the same
      // release sequence), so our release synchronizes with its acquire:
      // our put claims (relaxed fetch-adds on the ring's producer index,
      // earlier in program order) happen before its ticket, which comes
      // before its FLUSH claim in program order (and, by the acquire, is
      // complete before that claim issues). Coherence on the producer index
      // then puts our puts earlier in the ring. seq_cst would add nothing:
      // the argument uses only this one release/acquire pair per caller.
      target = __hip_atomic_fetch_add(flushTicket, uint64_t{1}, __ATOMIC_ACQ_REL,
                                      __HIP_MEMORY_SCOPE_AGENT) +
               1;
    }
  }
  invoke</*kLaneMasked=*/false>(true, ccoProxyOpFlush);
  return target;
}

template <typename Queue>
template <bool kTimeout>
__device__ inline int ccoProxy<Queue>::flushWait(uint64_t* done, uint64_t target,
                                                 uint64_t timeoutCycles) {
  int result = 0;
  if (__lane_id() == 0) {
    [[maybe_unused]] uint64_t start = 0;
    if constexpr (kTimeout) start = static_cast<uint64_t>(impl::clock64());
    // The failed bit is above any target, so it also ends the wait.
    uint64_t v;
    while ((v = __hip_atomic_load(done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) < target) {
      if constexpr (kTimeout) {
        if (static_cast<uint64_t>(impl::clock64()) - start >= timeoutCycles) {
          result = 1;
          break;
        }
      }
      __builtin_amdgcn_s_sleep(1);
    }
    // Exact: a FLUSH satisfied before the failure still succeeds.
    if (result == 0 && (v & ~ccoProxyFailedBit) < target) result = 2;
  }
  // Later stores (e.g. overwriting a source buffer) stay after the wait.
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return __builtin_amdgcn_readfirstlane(result);
}

template <typename Queue>
template <typename Coop, std::enable_if_t<!std::is_integral_v<Coop>, int>>
__device__ inline void ccoProxy<Queue>::flush(Coop coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopLaneMasked>,
                "ccoProxy::flush takes ccoCoopWarp or ccoCoopBlock: after lane-masked put/signal, "
                "call flush() (ccoCoopWarp) from all 64 lanes");
  coop.sync();
  if (flushingWave<Coop>()) {
    if (__builtin_expect(flushIssue</*kTimeout=*/false>(0) != 0, 0)) {
      CCO_PROXY_TRAP(__lane_id() == 0,
                     "ccoProxy: flush: the DevComm failed; see the host's stderr or "
                     "ccoDevCommGetError\n");
    }
    unflushed = false;
  }
  coop.sync();
}

template <typename Queue>
template <typename Coop, std::enable_if_t<!std::is_integral_v<Coop>, int>>
__device__ inline int ccoProxy<Queue>::flush(Coop coop, uint64_t timeoutCycles) {
  static_assert(!std::is_same_v<Coop, ccoCoopLaneMasked>,
                "ccoProxy::flush takes ccoCoopWarp or ccoCoopBlock: after lane-masked put/signal, "
                "call flush() (ccoCoopWarp) from all 64 lanes");
  coop.sync();
  int result = 0;
  if (flushingWave<Coop>()) {
    result = flushIssue</*kTimeout=*/true>(timeoutCycles);
    if (result == 0) unflushed = false;  // else a later flush must still cover it
  }
  coop.sync();
  if constexpr (std::is_same_v<Coop, ccoCoopBlock>) {
    // The flushing wavefronts' worst result to the whole block.
    if (__syncthreads_or(result != 0)) result = __syncthreads_or(result == 2) ? 2 : 1;
  }
  return result;
}

template <typename Queue>
template <bool kTimeout>
__device__ inline int ccoProxy<Queue>::waitSignalLeader(ccoGdaSignal_t signalId, uint64_t least,
                                                        uint64_t timeoutCycles) {
  volatile uint64_t* signalBuf = comm.ibgda.signalBuf;
  volatile uint64_t* signalShadows = comm.ibgda.signalShadows;
  const uint64_t shadow = signalShadows[signalId];
  [[maybe_unused]] uint64_t start = 0;
  if constexpr (kTimeout) start = static_cast<uint64_t>(impl::clock64());
  int result = 0;
  for (uint32_t polls = 1;; polls++) {
    const uint64_t val =
        __hip_atomic_load(&signalBuf[signalId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (val - shadow >= least) {
      signalShadows[signalId] = shadow + least;  // consume
      break;
    }
    if (polls % 256 == 0 && failed()) {
      result = 2;
      break;
    }
    if constexpr (kTimeout) {
      if (static_cast<uint64_t>(impl::clock64()) - start >= timeoutCycles) {
        result = failed() ? 2 : 1;
        break;
      }
    }
    asm volatile("" ::: "memory");
  }
  // Data the signal covers is read after it (system scope: the NIC wrote it).
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  return result;
}

template <typename Queue>
template <typename Coop>
__device__ inline int ccoProxy<Queue>::waitSignal(ccoGdaSignal_t signalId, uint64_t least,
                                                  Coop coop, uint64_t timeoutCycles) {
  coop.sync();
  int result = 0;
  if (groupLeader(coop))
    result = waitSignalLeader</*kTimeout=*/true>(signalId, least, timeoutCycles);
  coop.sync();
  // The leader's result to the whole group.
  if constexpr (std::is_same_v<Coop, ccoCoopBlock>) {
    if (__syncthreads_or(result != 0)) result = __syncthreads_or(result == 2) ? 2 : 1;
  } else if constexpr (std::is_same_v<Coop, ccoCoopWarp>) {
    result = __shfl(result, 0);
  }
  return result;
}

#undef CCO_PROXY_TRAP

#endif  // defined(__HIPCC__)

}  // namespace cco
}  // namespace mori
