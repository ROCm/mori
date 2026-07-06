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
// Symbol tags (must stay in sync with python/.../_bindings.py):
//   data path (put/put_value/get) — (ThreadMode,Coop) tag, since aggregate is
//   only valid with thread coop (static_assert in cco_scale_out.hpp):
//     it = independent+thread   iw = independent+warp
//     ib = independent+block    at = aggregate+thread
//   signal/wait/flush — coop-only tag: thread / warp / block.
//   put/put_value/signal also carry a remote-action tag: none / inc / add.

#include "mori/cco/cco_scale_out.hpp"

namespace {
using namespace mori::cco;
using Gda = ccoGda<CCO_GDA_BUILD_PROVIDER>;

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

// ── ccoDevComm field accessors ──
CCO_DEV int cco_devcomm_rank(uint64_t dc) { return AsDevComm(dc)->rank; }
CCO_DEV int cco_devcomm_world_size(uint64_t dc) { return AsDevComm(dc)->worldSize; }
CCO_DEV int cco_devcomm_lsa_rank(uint64_t dc) { return AsDevComm(dc)->lsaRank; }
CCO_DEV int cco_devcomm_lsa_size(uint64_t dc) { return AsDevComm(dc)->lsaSize; }

// ── GDA put: cco_gda_put__<tc>__<sig> ──
#define CCO_DEF_PUT(TAG, TM, COOP, SIG)                                                       \
  CCO_DEV void cco_gda_put__##TAG##__##SIG(uint64_t dc, int ctx, int peer, uint64_t dW,       \
                                           uint64_t dO, uint64_t sW, uint64_t sO, uint64_t n, \
                                           int sid, uint64_t sv) {                            \
    Gda gda{*AsDevComm(dc), ctx};                                                             \
    gda.put<CCO_TEAM_WORLD, TM>(peer, AsWindow(dW), dO, AsWindow(sW), sO, n, CCO_RA_##SIG,    \
                                COOP{});                                                      \
  }
#define CCO_DEF_PUT_SIGS(TAG, TM, COOP) \
  CCO_DEF_PUT(TAG, TM, COOP, none)      \
  CCO_DEF_PUT(TAG, TM, COOP, inc)       \
  CCO_DEF_PUT(TAG, TM, COOP, add)
CCO_TC_LIST(CCO_DEF_PUT_SIGS)
#undef CCO_DEF_PUT_SIGS
#undef CCO_DEF_PUT

// ── GDA put_value: cco_gda_put_value__<tc>__<sig> ──
#define CCO_DEF_PUTV(TAG, TM, COOP, SIG)                                                      \
  CCO_DEV void cco_gda_put_value__##TAG##__##SIG(uint64_t dc, int ctx, int peer, uint64_t dW, \
                                                 uint64_t dO, uint64_t value, int sid,        \
                                                 uint64_t sv) {                               \
    Gda gda{*AsDevComm(dc), ctx};                                                             \
    gda.putValue<CCO_TEAM_WORLD, TM>(peer, AsWindow(dW), dO, value, CCO_RA_##SIG, COOP{});    \
  }
#define CCO_DEF_PUTV_SIGS(TAG, TM, COOP) \
  CCO_DEF_PUTV(TAG, TM, COOP, none)      \
  CCO_DEF_PUTV(TAG, TM, COOP, inc)       \
  CCO_DEF_PUTV(TAG, TM, COOP, add)
CCO_TC_LIST(CCO_DEF_PUTV_SIGS)
#undef CCO_DEF_PUTV_SIGS
#undef CCO_DEF_PUTV

// ── GDA get (no remote action): cco_gda_get__<tc> ──
#define CCO_DEF_GET(TAG, TM, COOP)                                                          \
  CCO_DEV void cco_gda_get__##TAG(uint64_t dc, int ctx, int peer, uint64_t rW, uint64_t rO, \
                                  uint64_t lW, uint64_t lO, uint64_t n) {                   \
    Gda gda{*AsDevComm(dc), ctx};                                                           \
    gda.get<CCO_TEAM_WORLD, TM>(peer, AsWindow(rW), rO, AsWindow(lW), lO, n, COOP{});       \
  }
CCO_TC_LIST(CCO_DEF_GET)
#undef CCO_DEF_GET

// ── GDA signal (inc/add only): cco_gda_signal__<coop>__<sig> ──
#define CCO_DEF_SIGNAL(TAG, COOP, SIG)                                                 \
  CCO_DEV void cco_gda_signal__##TAG##__##SIG(uint64_t dc, int ctx, int peer, int sid, \
                                              uint64_t sv) {                           \
    Gda gda{*AsDevComm(dc), ctx};                                                      \
    gda.signal<CCO_TEAM_WORLD>(peer, CCO_RA_##SIG, COOP{});                            \
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

CCO_DEV void cco_gda_reset_signal(uint64_t dc, int ctx, int sigId) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.resetSignal(AsSig(sigId));
}

// ── GDA wait_signal: cco_gda_wait_signal__<coop> ──
#define CCO_DEF_WAIT(TAG, COOP)                                                          \
  CCO_DEV void cco_gda_wait_signal__##TAG(uint64_t dc, int ctx, int sid, uint64_t least, \
                                          int bits) {                                    \
    Gda gda{*AsDevComm(dc), ctx};                                                        \
    gda.waitSignal(AsSig(sid), least, COOP{}, bits);                                     \
  }
CCO_COOP_LIST(CCO_DEF_WAIT)
#undef CCO_DEF_WAIT

// ── GDA completion (>= warp): cco_gda_flush__<coop> / cco_gda_flush_peer__<coop> ──
#define CCO_DEF_FLUSH(TAG, COOP)                                           \
  CCO_DEV void cco_gda_flush__##TAG(uint64_t dc, int ctx) {                \
    Gda gda{*AsDevComm(dc), ctx};                                          \
    gda.flush(COOP{});                                                     \
  }                                                                        \
  CCO_DEV void cco_gda_flush_peer__##TAG(uint64_t dc, int ctx, int peer) { \
    Gda gda{*AsDevComm(dc), ctx};                                          \
    gda.flush(peer, COOP{});                                               \
  }
CCO_DEF_FLUSH(warp, ccoCoopWarp)
CCO_DEF_FLUSH(block, ccoCoopBlock)
#undef CCO_DEF_FLUSH
