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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

// C-style device wrappers over the cco GDA device API, compiled to
// libmori_cco_device.bc and linked into DSL (FlyDSL/...) kernels.
//
// Design (device-IR binding layer adapted to a scalar-only DSL FFI):
//   * Device objects cross the boundary as scalar handles (uint64 = intptr):
//     ccoDevComm*, ccoWindow_t — reinterpreted back here.
//   * Template axes (Coop / ThreadMode / RemoteAction) are MONOMORPHIZED: one
//     extern "C" symbol per valid combination, name-mangled with a tag suffix
//     (see the tables below). Each symbol body is a single fully-specialized
//     facade call — no runtime branch, no type erasure. The Python bindings
//     pick the symbol by name from the (compile-time) coop/thread_mode/signal_op
//     the kernel author passes, so the kernel emits one direct llvm.call to one
//     instantiation. This is how C++ templates cross a scalar FFI with zero
//     dispatch overhead, and lets ThreadMode be selected freely.
//   * Provider is fixed at build time via CCO_GDA_BUILD_PROVIDER (NIC macro), so
//     only one ccoGda<P> specialization is built (same model as the C++ kernels).
//
// Symbol tags (registered in python/mori/cco/device/ops.py):
//   GDA data path (put/put_value/get) — (ThreadMode,Coop) tag, since aggregate is
//   only valid with thread coop (static_assert in cco_scale_out.hpp):
//     it = independent+thread   iw = independent+warp
//     ib = independent+block    at = aggregate+thread
//   signal/wait/flush — coop-only tag: thread / warp / block.
//   put/put_value/signal also carry a remote-action tag: none / inc / add.
//   SDMA put/get on resolved addresses — coop tag only; their other axes ride
//   in a `flags` argument (SdmaWithFlags), constant-folded once inlined.
//   Side-effect-only wrappers return int status 0 rather than void. Triton's
//   extern_elementwise requires a concrete return element type; FlyDSL ignores
//   the status.

#include <type_traits>

#include "mori/cco/cco_scale_out.hpp"

namespace {
using namespace mori::cco;
using Gda = ccoGda<CCO_GDA_BUILD_PROVIDER>;
#if BUILD_CCO_SDMA
using Sdma = ccoSdma;
#endif

inline __device__ const ccoDevComm* AsDevComm(uint64_t h) {
  return reinterpret_cast<const ccoDevComm*>(h);
}
inline __device__ ccoWindow_t AsWindow(uint64_t h) { return reinterpret_cast<ccoWindow_t>(h); }
inline __device__ ccoGdaSignal_t AsSig(int id) { return static_cast<ccoGdaSignal_t>(id); }
}  // namespace

// Inlined into the kernel after link (thin forwarder, like a header-only call).
#define CCO_DEV extern "C" __device__ __attribute__((always_inline, visibility("default")))

// Remote-action constructors, keyed by the signal-op tag (sid/sv are the
// signal-id / signal-value parameters present in each wrapper signature).
#define CCO_RA_none \
  ccoGda_NoSignal {}
#define CCO_RA_inc \
  ccoGda_SignalInc { AsSig(sid) }
#define CCO_RA_add \
  ccoGda_SignalAdd { AsSig(sid), sv }

// Valid (tag, ThreadMode, Coop) combos for the data path.
#define CCO_TC_LIST(X)                          \
  X(it, ccoGdaThreadIndependent, ccoCoopThread) \
  X(iw, ccoGdaThreadIndependent, ccoCoopWarp)   \
  X(ib, ccoGdaThreadIndependent, ccoCoopBlock)  \
  X(at, ccoGdaThreadAggregate, ccoCoopThread)

// Coop-only tags (signal / wait).
#define CCO_COOP_LIST(X)   \
  X(thread, ccoCoopThread) \
  X(warp, ccoCoopWarp)     \
  X(block, ccoCoopBlock)

// ── LSA: expose only the peer's load/store-accessible VA. The copy/reduce is
//    done directly on this pointer in the DSL kernel (examples 04/05) — cco does
//    NOT move data for LSA. GDA below is opaque RDMA, so it IS exposed as ops.
//    peer_va = winBase + peerLsaRank*(stride4G<<32) + offset
CCO_DEV uint64_t cco_lsa_ptr(uint64_t window, int peer, uint64_t offset) {
  ccoWindowDevice* w = AsWindow(window);
  uint64_t stride = static_cast<uint64_t>(w->stride4G) << 32;
  return reinterpret_cast<uint64_t>(w->winBase) + static_cast<uint64_t>(peer) * stride + offset;
}

// Our own VA inside the window (ccoGetLocalPtr): cco_lsa_ptr at our LSA rank.
CCO_DEV uint64_t cco_local_ptr(uint64_t window, uint64_t offset) {
  return reinterpret_cast<uint64_t>(ccoGetLocalPtr(AsWindow(window), offset));
}

// ── LSA window geometry ───────────────────────────────────────────────────────
// `cco_lsa_ptr` above loads winBase and stride4G on *every* call. A kernel that
// needs many addresses from one window can take the geometry once with these two
// and compute `base + peer*stride + offset` itself.
//
// The pointer is explicitly address_space(1). `AsWindow` casts a uint64 to a
// *generic* pointer, and a generic access must be a flat_load: the compiler
// cannot prove the target is not LDS, so it counts against lgkmcnt as well as
// vmcnt. Naming the address space lets it emit global_load, which is vmcnt-only
// -- one fewer counter for any following s_waitcnt to wait on.
typedef const ccoWindowDevice __attribute__((address_space(1))) * ccoWindowGlobalPtr;

__device__ __attribute__((always_inline)) inline ccoWindowGlobalPtr AsWindowGlobal(uint64_t h) {
  return reinterpret_cast<ccoWindowGlobalPtr>(h);
}

CCO_DEV uint64_t cco_lsa_win_base(uint64_t window) {
  return reinterpret_cast<uint64_t>(AsWindowGlobal(window)->winBase);
}

// Returns the *byte* stride between peers, i.e. stride4G already shifted.
CCO_DEV uint64_t cco_lsa_stride(uint64_t window) {
  return static_cast<uint64_t>(AsWindowGlobal(window)->stride4G) << 32;
}

// System-scope publication fence used by direct LSA benchmark/store paths.
// Latency kernels fence from block lane 0; bandwidth kernels fence every lane.
// This primitive is NOT a barrier: callers must synchronize participating
// threads before entering it. leaderOnly orders only thread 0's operations
// under the HIP memory model; it must not be used as a general replacement for
// a release fence issued by every producer lane. It exists for the matched C++
// and Triton benchmark protocol, which brackets it with block barriers.
CCO_DEV int cco_system_fence(int leaderOnly) {
  if (!leaderOnly || threadIdx.x == 0) __threadfence_system();
  return 0;
}

// Expose SDMA C API. Symbol tags kept in sync with _bindings.py:
//   put/get carry a coop tag (thread/warp/block); a "_ns" suffix selects the
//   no-signal (fire-and-forget) variant.
// optFlags is a template parameter on the device API but stays a runtime `flags`
// argument here, so the C ABI and the Python binding are unchanged: the wrapper
// picks the instantiation. The branch is on a value uniform across the group.
// Entire block compile-gated on BUILD_CCO_SDMA — when off, no cco_sdma_* symbols
// are emitted (matches the host lib, which builds no SDMA queues).
#if BUILD_CCO_SDMA
#define CCO_DEF_SDMA_XFER(OP, TAG, COOP, SIG)                                                      \
  CCO_DEV int cco_sdma_##OP##__##TAG(uint64_t dc, int peer, uint64_t dW, uint64_t dO, uint64_t sW, \
                                     uint64_t sO, uint64_t n, int qid, int flags) {                \
    Sdma sdma{*AsDevComm(dc)};                                                                     \
    if (static_cast<uint32_t>(flags) & ccoSdmaOptFlagsAggregate) {                                 \
      sdma.OP<COOP, SIG, false, ccoSdmaOptFlagsAggregate>(peer, AsWindow(dW), dO, AsWindow(sW),    \
                                                          sO, n, qid);                             \
    } else {                                                                                       \
      sdma.OP<COOP, SIG, false, ccoSdmaOptFlagsDefault>(peer, AsWindow(dW), dO, AsWindow(sW), sO,  \
                                                        n, qid);                                   \
    }                                                                                              \
    return 0;                                                                                      \
  }

CCO_DEF_SDMA_XFER(put, thread, ccoCoopThread, true)
CCO_DEF_SDMA_XFER(put, warp, ccoCoopWarp, true)
CCO_DEF_SDMA_XFER(put, block, ccoCoopBlock, true)
CCO_DEF_SDMA_XFER(put, thread_ns, ccoCoopThread, false)
CCO_DEF_SDMA_XFER(put, warp_ns, ccoCoopWarp, false)
CCO_DEF_SDMA_XFER(put, block_ns, ccoCoopBlock, false)
CCO_DEF_SDMA_XFER(get, thread, ccoCoopThread, true)
CCO_DEF_SDMA_XFER(get, warp, ccoCoopWarp, true)
CCO_DEF_SDMA_XFER(get, block, ccoCoopBlock, true)
CCO_DEF_SDMA_XFER(get, thread_ns, ccoCoopThread, false)
CCO_DEF_SDMA_XFER(get, warp_ns, ccoCoopWarp, false)
CCO_DEF_SDMA_XFER(get, block_ns, ccoCoopBlock, false)

#undef CCO_DEF_SDMA_XFER

// 2D put/get: same tags; dP/sP are row pitches in bytes, w (bytes) x h (rows)
// the rect. They cross the FFI as int32 (the scalar types the bindings declare).
#define CCO_DEF_SDMA_XFER2D(OP, TAG, COOP, SIG)                                                 \
  CCO_DEV int cco_sdma_##OP##2d__##TAG(uint64_t dc, int peer, uint64_t dW, uint64_t dO, int dP, \
                                       uint64_t sW, uint64_t sO, int sP, int w, int h, int qid, \
                                       int flags) {                                             \
    Sdma sdma{*AsDevComm(dc)};                                                                  \
    const uint32_t dPitch = static_cast<uint32_t>(dP), sPitch = static_cast<uint32_t>(sP);      \
    const uint32_t width = static_cast<uint32_t>(w), height = static_cast<uint32_t>(h);         \
    if (static_cast<uint32_t>(flags) & ccoSdmaOptFlagsAggregate) {                              \
      sdma.OP##2d < COOP, SIG, false,                                                           \
          ccoSdmaOptFlagsAggregate >                                                            \
              (peer, AsWindow(dW), dO, dPitch, AsWindow(sW), sO, sPitch, width, height, qid);   \
    } else {                                                                                    \
      sdma.OP##2d < COOP, SIG, false,                                                           \
          ccoSdmaOptFlagsDefault >                                                              \
              (peer, AsWindow(dW), dO, dPitch, AsWindow(sW), sO, sPitch, width, height, qid);   \
    }                                                                                           \
    return 0;                                                                                   \
  }

CCO_DEF_SDMA_XFER2D(put, thread, ccoCoopThread, true)
CCO_DEF_SDMA_XFER2D(put, warp, ccoCoopWarp, true)
CCO_DEF_SDMA_XFER2D(put, block, ccoCoopBlock, true)
CCO_DEF_SDMA_XFER2D(put, thread_ns, ccoCoopThread, false)
CCO_DEF_SDMA_XFER2D(put, warp_ns, ccoCoopWarp, false)
CCO_DEF_SDMA_XFER2D(put, block_ns, ccoCoopBlock, false)
CCO_DEF_SDMA_XFER2D(get, thread, ccoCoopThread, true)
CCO_DEF_SDMA_XFER2D(get, warp, ccoCoopWarp, true)
CCO_DEF_SDMA_XFER2D(get, block, ccoCoopBlock, true)
CCO_DEF_SDMA_XFER2D(get, thread_ns, ccoCoopThread, false)
CCO_DEF_SDMA_XFER2D(get, warp_ns, ccoCoopWarp, false)
CCO_DEF_SDMA_XFER2D(get, block_ns, ccoCoopBlock, false)

#undef CCO_DEF_SDMA_XFER2D

// ── SDMA quiet: cco_sdma_quiet__<coop> (wait for outstanding ops to peer) ──
#define CCO_DEF_SDMA_QUIET(TAG, COOP)                        \
  CCO_DEV int cco_sdma_quiet__##TAG(uint64_t dc, int peer) { \
    Sdma sdma{*AsDevComm(dc)};                               \
    sdma.quiet<COOP>(peer);                                  \
    return 0;                                                \
  }
CCO_DEF_SDMA_QUIET(thread, ccoCoopThread)
CCO_DEF_SDMA_QUIET(warp, ccoCoopWarp)
CCO_DEF_SDMA_QUIET(block, ccoCoopBlock)
#undef CCO_DEF_SDMA_QUIET

// quiet a single (peer, queueId) queue only.
CCO_DEV int cco_sdma_quiet_queue(uint64_t dc, int peer, int qid) {
  Sdma sdma{*AsDevComm(dc)};
  sdma.quietQueue(peer, qid);
  return 0;
}

// ── SDMA commit: ring the doorbell for packets posted with the Aggregate flag ──
#define CCO_DEF_SDMA_COMMIT(TAG, COOP)                                 \
  CCO_DEV int cco_sdma_commit__##TAG(uint64_t dc, int peer, int qid) { \
    Sdma sdma{*AsDevComm(dc)};                                         \
    sdma.commit<COOP>(peer, qid);                                      \
    return 0;                                                          \
  }
CCO_DEF_SDMA_COMMIT(thread, ccoCoopThread)
CCO_DEF_SDMA_COMMIT(warp, ccoCoopWarp)
CCO_DEF_SDMA_COMMIT(block, ccoCoopBlock)
#undef CCO_DEF_SDMA_COMMIT

// ── SDMA put/get on resolved addresses: cco_sdma_{put,get}_ptr__<coop> ──
// ccoSdma::putPtr / getPtr with every remaining template axis carried in
// `flags` (below). The wrappers are inlined into the caller's kernel, so a
// compile-time constant `flags` folds the dispatch down to the one
// instantiation; only a runtime `flags` pays for the branches. Unknown bits, and
// SAME_QUEUE off thread scope, trap. Mirrored by SdmaFlags in ops.py.
namespace {
// Bits 0-1 are ccoSdmaOptFlags itself.
constexpr int kSdmaFlagOptMask = ccoSdmaOptFlagsAggregate | ccoSdmaOptFlagsSignalPerCopy;
constexpr int kSdmaFlagLocalSignal = 1 << 2;
constexpr int kSdmaFlagRemoteSignal = 1 << 3;
constexpr int kSdmaFlagSameQueue = 1 << 4;
constexpr int kSdmaFlagAll =
    kSdmaFlagOptMask | kSdmaFlagLocalSignal | kSdmaFlagRemoteSignal | kSdmaFlagSameQueue;

template <typename T, T v>
using Const = std::integral_constant<T, v>;

// Calls fn(localSignal, remoteSignal, optFlags, threadMode), each a
// std::integral_constant, for the instantiation `flags` names.
template <typename Coop, typename Fn>
inline __device__ void SdmaWithFlags(int flags, Fn&& fn) {
  if (flags & ~kSdmaFlagAll) __builtin_trap();  // unknown SdmaFlags bit
  auto withMode = [&](auto local, auto remote, auto opt) {
    if (flags & kSdmaFlagSameQueue) {
      if constexpr (std::is_same_v<Coop, ccoCoopThread>) {
        fn(local, remote, opt, Const<ccoSdmaThreadMode, ccoSdmaThreadSameQueue>{});
      } else {
        __builtin_trap();  // SAME_QUEUE needs thread scope
      }
    } else {
      fn(local, remote, opt, Const<ccoSdmaThreadMode, ccoSdmaThreadIndependent>{});
    }
  };
  auto withOpt = [&](auto local, auto remote) {
    switch (flags & kSdmaFlagOptMask) {
      case 0:
        withMode(local, remote, Const<uint32_t, 0>{});
        break;
      case 1:
        withMode(local, remote, Const<uint32_t, 1>{});
        break;
      case 2:
        withMode(local, remote, Const<uint32_t, 2>{});
        break;
      default:
        withMode(local, remote, Const<uint32_t, 3>{});
        break;
    }
  };
  auto withRemote = [&](auto local) {
    if (flags & kSdmaFlagRemoteSignal) {
      withOpt(local, std::true_type{});
    } else {
      withOpt(local, std::false_type{});
    }
  };
  if (flags & kSdmaFlagLocalSignal) {
    withRemote(std::true_type{});
  } else {
    withRemote(std::false_type{});
  }
}
}  // namespace

#define CCO_DEF_SDMA_PTR(OP, CPP_OP, TAG, COOP)                                                \
  CCO_DEV int cco_sdma_##OP##_ptr__##TAG(uint64_t dc, int peer, uint64_t dst, uint64_t src,    \
                                         uint64_t n, int qid, int flags) {                     \
    Sdma sdma{*AsDevComm(dc)};                                                                 \
    void* d = reinterpret_cast<void*>(dst);                                                    \
    void* s = reinterpret_cast<void*>(src);                                                    \
    SdmaWithFlags<COOP>(flags, [&](auto local, auto remote, auto opt, auto mode) {             \
      sdma.CPP_OP<COOP, decltype(local)::value, decltype(remote)::value, decltype(opt)::value, \
                  decltype(mode)::value>(peer, d, s, n, qid);                                  \
    });                                                                                        \
    return 0;                                                                                  \
  }
CCO_DEF_SDMA_PTR(put, putPtr, thread, ccoCoopThread)
CCO_DEF_SDMA_PTR(put, putPtr, warp, ccoCoopWarp)
CCO_DEF_SDMA_PTR(put, putPtr, block, ccoCoopBlock)
CCO_DEF_SDMA_PTR(get, getPtr, thread, ccoCoopThread)
CCO_DEF_SDMA_PTR(get, getPtr, warp, ccoCoopWarp)
CCO_DEF_SDMA_PTR(get, getPtr, block, ccoCoopBlock)
#undef CCO_DEF_SDMA_PTR

// Poll our signal counter for (src lsa rank, qid) until it reaches `expected`
// (ccoSdma::waitSignal; returns at once for an out-of-range qid).
CCO_DEV int cco_sdma_wait_signal(uint64_t dc, int src, int qid, uint64_t expected) {
  Sdma sdma{*AsDevComm(dc)};
  sdma.waitSignal(src, qid, expected);
  return 0;
}

// ── SDMA put on raw addresses, signalling the peer only ──
// Kept for existing callers; cco_sdma_put_ptr__thread with REMOTE_SIGNAL is the
// same instantiation.
// `dst` is the peer's address as mapped here (cco_lsa_ptr), `src` a local one;
// neither is checked. The trailing ATOMIC bumps the peer's
// cco_sdma_signal_slot(dc, <my lsa rank>, qid), not ours.
CCO_DEV int cco_sdma_put_ptr_rsig__thread(uint64_t dc, int peer, uint64_t dst, uint64_t src,
                                          uint64_t n, int qid) {
  Sdma sdma{*AsDevComm(dc)};
  sdma.putPtr<ccoCoopThread, false, true>(peer, reinterpret_cast<void*>(dst),
                                          reinterpret_cast<void*>(src), n, qid);
  return 0;
}

// Address of our signal counter for (src lsa rank, qid); 0 when out of range.
CCO_DEV uint64_t cco_sdma_signal_slot(uint64_t dc, int src, int qid) {
  Sdma sdma{*AsDevComm(dc)};
  return reinterpret_cast<uint64_t>(sdma.signalSlot(src, qid));
}

// ── SDMA engine-side wait and broadcast ──
// cco_sdma_{post_wait_mem,post_wait_signal,put_broadcast}__<coop>:
// ccoSdma::postWaitMem / postWaitSignal / putBroadcast. They take only the
// Aggregate opt flag (the broadcast also the two signals), so `flags` is the
// SdmaFlags subset below; any other bit traps. A compile-time constant `flags`
// folds to one instantiation once inlined, as for cco_sdma_put_ptr__<coop>.
// On gfx12.5 these trap when reached (the packet layouts are not validated).
namespace {
template <typename Fn>
inline __device__ void SdmaWaitWithFlags(int flags, Fn&& fn) {
  if (flags & ~static_cast<int>(ccoSdmaOptFlagsAggregate)) __builtin_trap();  // unknown flag
  if (flags & ccoSdmaOptFlagsAggregate) {
    fn(Const<uint32_t, ccoSdmaOptFlagsAggregate>{});
  } else {
    fn(Const<uint32_t, ccoSdmaOptFlagsDefault>{});
  }
}

// Calls fn(localSignal, remoteSignal, optFlags) for the instantiation `flags` names.
template <typename Fn>
inline __device__ void SdmaBroadcastWithFlags(int flags, Fn&& fn) {
  constexpr int kAll = ccoSdmaOptFlagsAggregate | kSdmaFlagLocalSignal | kSdmaFlagRemoteSignal;
  if (flags & ~kAll) __builtin_trap();  // unknown flag, or one putBroadcast has no template for
  auto withOpt = [&](auto local, auto remote) {
    if (flags & ccoSdmaOptFlagsAggregate) {
      fn(local, remote, Const<uint32_t, ccoSdmaOptFlagsAggregate>{});
    } else {
      fn(local, remote, Const<uint32_t, ccoSdmaOptFlagsDefault>{});
    }
  };
  auto withRemote = [&](auto local) {
    if (flags & kSdmaFlagRemoteSignal) {
      withOpt(local, std::true_type{});
    } else {
      withOpt(local, std::false_type{});
    }
  };
  if (flags & kSdmaFlagLocalSignal) {
    withRemote(std::true_type{});
  } else {
    withRemote(std::false_type{});
  }
}
}  // namespace

// Park queue (peer, qid) until (*(win + off) & mask) <cmp> value. `win`/`off`
// address memory on THIS GPU; value and mask are the low 32 bits of the
// arguments. `cmp` is ccoSdmaCmp (0 Always .. 6 Gt); anything else traps.
#define CCO_DEF_SDMA_POST_WAIT_MEM(TAG, COOP)                                                     \
  CCO_DEV int cco_sdma_post_wait_mem__##TAG(uint64_t dc, int peer, int qid, uint64_t win,         \
                                            uint64_t off, uint64_t value, int cmp, uint64_t mask, \
                                            int flags) {                                          \
    Sdma sdma{*AsDevComm(dc)};                                                                    \
    if (static_cast<uint32_t>(cmp) > static_cast<uint32_t>(ccoSdmaCmpGt)) __builtin_trap();       \
    SdmaWaitWithFlags(flags, [&](auto opt) {                                                      \
      sdma.postWaitMem<COOP, decltype(opt)::value>(                                               \
          peer, qid, AsWindow(win), off, static_cast<uint32_t>(value),                            \
          static_cast<ccoSdmaCmp>(cmp), static_cast<uint32_t>(mask));                             \
    });                                                                                           \
    return 0;                                                                                     \
  }
CCO_DEF_SDMA_POST_WAIT_MEM(thread, ccoCoopThread)
CCO_DEF_SDMA_POST_WAIT_MEM(warp, ccoCoopWarp)
CCO_DEF_SDMA_POST_WAIT_MEM(block, ccoCoopBlock)
#undef CCO_DEF_SDMA_POST_WAIT_MEM

// Park queue (peer, qid) until our signal slot (src lsa rank, sig_qid) holds
// `expected` or more. The engine compares 32 bits: `expected` must stay below 2^32.
#define CCO_DEF_SDMA_POST_WAIT_SIGNAL(TAG, COOP)                                           \
  CCO_DEV int cco_sdma_post_wait_signal__##TAG(uint64_t dc, int peer, int qid, int src,    \
                                               int sigQid, uint64_t expected, int flags) { \
    Sdma sdma{*AsDevComm(dc)};                                                             \
    SdmaWaitWithFlags(flags, [&](auto opt) {                                               \
      sdma.postWaitSignal<COOP, decltype(opt)::value>(peer, qid, src, sigQid, expected);   \
    });                                                                                    \
    return 0;                                                                              \
  }
CCO_DEF_SDMA_POST_WAIT_SIGNAL(thread, ccoCoopThread)
CCO_DEF_SDMA_POST_WAIT_SIGNAL(warp, ccoCoopWarp)
CCO_DEF_SDMA_POST_WAIT_SIGNAL(block, ccoCoopBlock)
#undef CCO_DEF_SDMA_POST_WAIT_SIGNAL

// One local src -> two destinations in one COPY packet. (peer, dW, dO) is the
// first destination and selects the queue; (peer2, dW2, dO2) the second.
#define CCO_DEF_SDMA_PUT_BROADCAST(TAG, COOP)                                                    \
  CCO_DEV int cco_sdma_put_broadcast__##TAG(uint64_t dc, int peer, uint64_t dW, uint64_t dO,     \
                                            int peer2, uint64_t dW2, uint64_t dO2, uint64_t sW,  \
                                            uint64_t sO, uint64_t n, int qid, int flags) {       \
    Sdma sdma{*AsDevComm(dc)};                                                                   \
    SdmaBroadcastWithFlags(flags, [&](auto local, auto remote, auto opt) {                       \
      sdma.putBroadcast<COOP, decltype(local)::value, decltype(remote)::value,                   \
                        decltype(opt)::value>(peer, AsWindow(dW), dO, peer2, AsWindow(dW2), dO2, \
                                              AsWindow(sW), sO, n, qid);                         \
    });                                                                                          \
    return 0;                                                                                    \
  }
CCO_DEF_SDMA_PUT_BROADCAST(thread, ccoCoopThread)
CCO_DEF_SDMA_PUT_BROADCAST(warp, ccoCoopWarp)
CCO_DEF_SDMA_PUT_BROADCAST(block, ccoCoopBlock)
#undef CCO_DEF_SDMA_PUT_BROADCAST

// Fused wait + put: ccoSdma::postWaitMemPutPtr (POLL_REGMEM, COPY and its
// signals in one reservation, one doorbell). Thread scope only. `flags`:
// LOCAL_SIGNAL / REMOTE_SIGNAL / AGGREGATE; other bits trap, as does a bad cmp.
CCO_DEV int cco_sdma_post_wait_mem_put_ptr__thread(uint64_t dc, int peer, int qid, uint64_t win,
                                                   uint64_t off, uint64_t value, int cmp,
                                                   uint64_t mask, uint64_t dst, uint64_t src,
                                                   uint64_t n, int flags) {
  Sdma sdma{*AsDevComm(dc)};
  if (static_cast<uint32_t>(cmp) > static_cast<uint32_t>(ccoSdmaCmpGt)) __builtin_trap();
  const auto w = AsWindow(win);
  void* d = reinterpret_cast<void*>(dst);
  void* s = reinterpret_cast<void*>(src);
  const auto c = static_cast<ccoSdmaCmp>(cmp);
  const auto v = static_cast<uint32_t>(value), m = static_cast<uint32_t>(mask);
  SdmaBroadcastWithFlags(flags, [&](auto local, auto remote, auto opt) {
    sdma.postWaitMemPutPtr<decltype(local)::value, decltype(remote)::value, decltype(opt)::value>(
        peer, qid, w, off, v, c, m, d, s, n);
  });
  return 0;
}
#endif  // BUILD_CCO_SDMA

// ── ccoDevComm field accessors ──
CCO_DEV int cco_devcomm_rank(uint64_t dc) { return AsDevComm(dc)->rank; }
CCO_DEV int cco_devcomm_world_size(uint64_t dc) { return AsDevComm(dc)->worldSize; }
CCO_DEV int cco_devcomm_lsa_rank(uint64_t dc) { return AsDevComm(dc)->lsaRank; }
CCO_DEV int cco_devcomm_lsa_size(uint64_t dc) { return AsDevComm(dc)->lsaSize; }

// ── GDA put: cco_gda_put__<tc>__<sig> ──
#define CCO_DEF_PUT(TAG, TM, COOP, SIG)                                                      \
  CCO_DEV int cco_gda_put__##TAG##__##SIG(uint64_t dc, int ctx, int peer, uint64_t dW,       \
                                          uint64_t dO, uint64_t sW, uint64_t sO, uint64_t n, \
                                          int sid, uint64_t sv) {                            \
    Gda gda{*AsDevComm(dc), ctx};                                                            \
    gda.put<CCO_TEAM_WORLD, TM>(peer, AsWindow(dW), dO, AsWindow(sW), sO, n, CCO_RA_##SIG,   \
                                COOP{});                                                     \
    return 0;                                                                                \
  }
#define CCO_DEF_PUT_SIGS(TAG, TM, COOP) \
  CCO_DEF_PUT(TAG, TM, COOP, none)      \
  CCO_DEF_PUT(TAG, TM, COOP, inc)       \
  CCO_DEF_PUT(TAG, TM, COOP, add)
CCO_TC_LIST(CCO_DEF_PUT_SIGS)
#undef CCO_DEF_PUT_SIGS
#undef CCO_DEF_PUT

// ── GDA put_value: cco_gda_put_value__<tc>__<sig> ──
#define CCO_DEF_PUTV(TAG, TM, COOP, SIG)                                                     \
  CCO_DEV int cco_gda_put_value__##TAG##__##SIG(uint64_t dc, int ctx, int peer, uint64_t dW, \
                                                uint64_t dO, uint64_t value, int sid,        \
                                                uint64_t sv) {                               \
    Gda gda{*AsDevComm(dc), ctx};                                                            \
    gda.putValue<CCO_TEAM_WORLD, TM>(peer, AsWindow(dW), dO, value, CCO_RA_##SIG, COOP{});   \
    return 0;                                                                                \
  }
#define CCO_DEF_PUTV_SIGS(TAG, TM, COOP) \
  CCO_DEF_PUTV(TAG, TM, COOP, none)      \
  CCO_DEF_PUTV(TAG, TM, COOP, inc)       \
  CCO_DEF_PUTV(TAG, TM, COOP, add)
CCO_TC_LIST(CCO_DEF_PUTV_SIGS)
#undef CCO_DEF_PUTV_SIGS
#undef CCO_DEF_PUTV

// ── GDA get (no remote action): cco_gda_get__<tc> ──
#define CCO_DEF_GET(TAG, TM, COOP)                                                         \
  CCO_DEV int cco_gda_get__##TAG(uint64_t dc, int ctx, int peer, uint64_t rW, uint64_t rO, \
                                 uint64_t lW, uint64_t lO, uint64_t n) {                   \
    Gda gda{*AsDevComm(dc), ctx};                                                          \
    gda.get<CCO_TEAM_WORLD, TM>(peer, AsWindow(rW), rO, AsWindow(lW), lO, n, COOP{});      \
    return 0;                                                                              \
  }
CCO_TC_LIST(CCO_DEF_GET)
#undef CCO_DEF_GET

// ── GDA signal (inc/add only): cco_gda_signal__<coop>__<sig> ──
#define CCO_DEF_SIGNAL(TAG, COOP, SIG)                                                \
  CCO_DEV int cco_gda_signal__##TAG##__##SIG(uint64_t dc, int ctx, int peer, int sid, \
                                             uint64_t sv) {                           \
    Gda gda{*AsDevComm(dc), ctx};                                                     \
    gda.signal<CCO_TEAM_WORLD>(peer, CCO_RA_##SIG, COOP{});                           \
    return 0;                                                                         \
  }
#define CCO_DEF_SIGNAL_SIGS(TAG, COOP) \
  CCO_DEF_SIGNAL(TAG, COOP, inc)       \
  CCO_DEF_SIGNAL(TAG, COOP, add)
CCO_COOP_LIST(CCO_DEF_SIGNAL_SIGS)
#undef CCO_DEF_SIGNAL_SIGS
#undef CCO_DEF_SIGNAL

// ── signal slot local ops (no template axis) ──
CCO_DEV uint64_t cco_gda_read_signal(uint64_t dc, int ctx, int sigId, int bits) {
  Gda gda{*AsDevComm(dc), ctx};
  return gda.readSignal(AsSig(sigId), bits);
}

CCO_DEV int cco_gda_reset_signal(uint64_t dc, int ctx, int sigId) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.resetSignal(AsSig(sigId));
  return 0;
}

// ── GDA wait_signal: cco_gda_wait_signal__<coop> ──
#define CCO_DEF_WAIT(TAG, COOP)                                                         \
  CCO_DEV int cco_gda_wait_signal__##TAG(uint64_t dc, int ctx, int sid, uint64_t least, \
                                         int bits) {                                    \
    Gda gda{*AsDevComm(dc), ctx};                                                       \
    gda.waitSignal(AsSig(sid), least, COOP{}, bits);                                    \
    return 0;                                                                           \
  }
CCO_COOP_LIST(CCO_DEF_WAIT)
#undef CCO_DEF_WAIT

// ── GDA completion (>= warp): cco_gda_flush__<coop> / cco_gda_flush_peer__<coop> ──
#define CCO_DEF_FLUSH(TAG, COOP)                                          \
  CCO_DEV int cco_gda_flush__##TAG(uint64_t dc, int ctx) {                \
    Gda gda{*AsDevComm(dc), ctx};                                         \
    gda.flush(COOP{});                                                    \
    return 0;                                                             \
  }                                                                       \
  CCO_DEV int cco_gda_flush_peer__##TAG(uint64_t dc, int ctx, int peer) { \
    Gda gda{*AsDevComm(dc), ctx};                                         \
    gda.flush(peer, COOP{});                                              \
    return 0;                                                             \
  }
CCO_DEF_FLUSH(warp, ccoCoopWarp)
CCO_DEF_FLUSH(block, ccoCoopBlock)
#undef CCO_DEF_FLUSH
