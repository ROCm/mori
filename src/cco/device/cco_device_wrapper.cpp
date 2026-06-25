// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

// C-style device wrappers over the cco GDA device API, compiled to
// libmori_cco_device.bc and linked into DSL (FlyDSL/...) kernels.
//
// Design (device-IR binding layer adapted to a scalar-only DSL FFI):
//   * Device objects cross the boundary as scalar handles (uint64 = intptr):
//     ccoDevComm*, ccoWindow_t — reinterpreted back here.
//   * Template axes (Coop / RemoteAction) are erased to int enums and dispatched
//     at runtime (CCO_BY_COOP / signalOp branch).
//   * Provider is fixed at build time via CCO_GDA_BUILD_PROVIDER (NIC macro), so
//     only one ccoGda<P> specialization is built (same model as the C++ kernels).

#include "mori/cco/cco_scale_out.hpp"

namespace {
using namespace mori::cco;
using Gda = ccoGda<CCO_GDA_BUILD_PROVIDER>;

// Mirrored 1:1 in the Python bindings (CoopScope / SignalOp).
enum CcoCoopScope : int { CCO_COOP_THREAD = 0, CCO_COOP_WARP = 1, CCO_COOP_BLOCK = 2 };
enum CcoSignalOp : int { CCO_SIGNAL_NONE = 0, CCO_SIGNAL_INC = 1, CCO_SIGNAL_ADD = 2 };

inline __device__ const ccoDevComm* AsDevComm(uint64_t h) {
  return reinterpret_cast<const ccoDevComm*>(h);
}
inline __device__ ccoWindow_t AsWindow(uint64_t h) { return reinterpret_cast<ccoWindow_t>(h); }
inline __device__ ccoGdaSignal_t AsSig(int id) { return static_cast<ccoGdaSignal_t>(id); }
}  // namespace

// Export macro + runtime coop dispatch (binds `Coop` in each branch; variadic so
// the statement may contain commas).
#define CCO_DEV extern "C" __device__ __attribute__((visibility("default")))
#define CCO_BY_COOP(scope, ...)                                              \
  do {                                                                       \
    if ((scope) == CCO_COOP_WARP) {                                          \
      using Coop = ccoCoopWarp;                                              \
      __VA_ARGS__;                                                           \
    } else if ((scope) == CCO_COOP_BLOCK) {                                  \
      using Coop = ccoCoopBlock;                                             \
      __VA_ARGS__;                                                           \
    } else {                                                                 \
      using Coop = ccoCoopThread;                                            \
      __VA_ARGS__;                                                           \
    }                                                                        \
  } while (0)

namespace {
// Leaf posters: erase the RemoteAction template via the signalOp enum.
template <typename Coop>
__device__ void PutCoop(Gda& gda, int peer, ccoWindow_t dst, size_t dstOff, ccoWindow_t src,
                        size_t srcOff, size_t bytes, int op, int sigId, uint64_t sigVal) {
  if (op == CCO_SIGNAL_INC)
    gda.put(peer, dst, dstOff, src, srcOff, bytes, ccoGda_SignalInc{AsSig(sigId)}, Coop{});
  else if (op == CCO_SIGNAL_ADD)
    gda.put(peer, dst, dstOff, src, srcOff, bytes, ccoGda_SignalAdd{AsSig(sigId), sigVal}, Coop{});
  else
    gda.put(peer, dst, dstOff, src, srcOff, bytes, ccoGda_NoSignal{}, Coop{});
}

template <typename Coop>
__device__ void PutValueCoop(Gda& gda, int peer, ccoWindow_t dst, size_t dstOff, uint64_t value,
                             int op, int sigId, uint64_t sigVal) {
  if (op == CCO_SIGNAL_INC)
    gda.putValue(peer, dst, dstOff, value, ccoGda_SignalInc{AsSig(sigId)}, Coop{});
  else if (op == CCO_SIGNAL_ADD)
    gda.putValue(peer, dst, dstOff, value, ccoGda_SignalAdd{AsSig(sigId), sigVal}, Coop{});
  else
    gda.putValue(peer, dst, dstOff, value, ccoGda_NoSignal{}, Coop{});
}

template <typename Coop>
__device__ void SignalCoop(Gda& gda, int peer, int op, int sigId, uint64_t sigVal) {
  if (op == CCO_SIGNAL_ADD)
    gda.signal(peer, ccoGda_SignalAdd{AsSig(sigId), sigVal}, Coop{});
  else
    gda.signal(peer, ccoGda_SignalInc{AsSig(sigId)}, Coop{});
}
}  // namespace

// ── LSA: only expose the peer's load/store-accessible VA. The copy/reduce is
//    done directly on this pointer in the DSL kernel (see examples 04/05) — cco
//    does NOT move data for LSA. GDA below is opaque RDMA, so it IS exposed as ops.
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

// ── GDA data path ──
CCO_DEV void cco_gda_put(uint64_t dc, int ctx, int peer, uint64_t dstWin, uint64_t dstOff,
                         uint64_t srcWin, uint64_t srcOff, uint64_t bytes, int op, int sigId,
                         uint64_t sigVal, int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  CCO_BY_COOP(scope, PutCoop<Coop>(gda, peer, AsWindow(dstWin), dstOff, AsWindow(srcWin), srcOff,
                                   bytes, op, sigId, sigVal));
}

CCO_DEV void cco_gda_put_value(uint64_t dc, int ctx, int peer, uint64_t dstWin, uint64_t dstOff,
                               uint64_t value, int op, int sigId, uint64_t sigVal, int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  CCO_BY_COOP(scope,
              PutValueCoop<Coop>(gda, peer, AsWindow(dstWin), dstOff, value, op, sigId, sigVal));
}

CCO_DEV void cco_gda_get(uint64_t dc, int ctx, int peer, uint64_t remoteWin, uint64_t remoteOff,
                         uint64_t localWin, uint64_t localOff, uint64_t bytes, int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  CCO_BY_COOP(scope, gda.get(peer, AsWindow(remoteWin), remoteOff, AsWindow(localWin), localOff,
                             bytes, Coop{}));
}

// ── GDA signal ──
CCO_DEV void cco_gda_signal(uint64_t dc, int ctx, int peer, int op, int sigId, uint64_t sigVal,
                            int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  CCO_BY_COOP(scope, SignalCoop<Coop>(gda, peer, op, sigId, sigVal));
}

CCO_DEV uint64_t cco_gda_read_signal(uint64_t dc, int ctx, int sigId, int bits) {
  Gda gda{*AsDevComm(dc), ctx};
  return gda.readSignal(AsSig(sigId), bits);
}

CCO_DEV void cco_gda_reset_signal(uint64_t dc, int ctx, int sigId) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.resetSignal(AsSig(sigId));
}

CCO_DEV void cco_gda_wait_signal(uint64_t dc, int ctx, int sigId, uint64_t least, int bits,
                                 int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  CCO_BY_COOP(scope, gda.waitSignal(AsSig(sigId), least, Coop{}, bits));
}

// ── GDA completion (>= warp; THREAD falls back to warp) ──
CCO_DEV void cco_gda_flush(uint64_t dc, int ctx, int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  if (scope == CCO_COOP_BLOCK)
    gda.flush(ccoCoopBlock{});
  else
    gda.flush(ccoCoopWarp{});
}

CCO_DEV void cco_gda_flush_peer(uint64_t dc, int ctx, int peer, int scope) {
  Gda gda{*AsDevComm(dc), ctx};
  if (scope == CCO_COOP_BLOCK)
    gda.flush(peer, ccoCoopBlock{});
  else
    gda.flush(peer, ccoCoopWarp{});
}
