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

// ===========================================================================
// push_timing.hpp
//
// Phase timestamps of the SDMA push kernels (reduce-scatter, all-reduce,
// all-gather, all-to-all), for debugging only: build with -DMORI_PUSH_TIMING=1.
// With the macro at its default 0 this header is empty and every MORI_PT /
// MORI_PT_ARGS call site compiles to nothing, so the kernels are unchanged.
//
// Kernels store wall_clock64() into one __device__ global, gPushTiming; no
// kernel argument is added. Each launch overwrites it, so after a stream sync it
// holds the last push launch of this device. The host reads it back with
// PushTimingFetch / PushTimingDump (hipMemcpyFromSymbol) and prints there.
//
// The symbol is `static`, i.e. one copy per translation unit: fetch it from the
// TU that launched the kernels (the facade is header-only, so that is the TU
// including collectives_facade.hpp).
//
// Recorded points (us in the report, relative to the launch entry on that PE;
// wall_clock64 is per GPU, so times are comparable only within one PE):
//   tStart       entry of an async start launch (scatter issued in its own launch)
//   t0[b]        entry of block b (b < kPushTimingBlocks)
//   tPost        block 0 after posting its arrival to every peer
//   tIssue       block 0 after ringing every SDMA doorbell of the scatter
//   arr[s][p]    first poll that saw sender p's slice s land (one block per PE)
//   t3/t4[b][s]  RS/AR: block b after slice s's wait + acquire / after its reduce
//   dlv[p], tDlv my last scatter copy seen landed at peer p / all of them
//   AR only:     broadcast pre-post, per-block count-in, tail owner, broadcast
//                arrivals and delivery
//   tEnd         AG/A2A: after the completion wait + acquire; AR: tail done
// ===========================================================================
#pragma once

#include "mori/collective/XLA/collectives_common.hpp"

#if MORI_PUSH_TIMING && (defined(__HIPCC__) || defined(__HIP__))

#include <hip/hip_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <string>

// wall_clock64 rate if hipDeviceAttributeWallClockRate reports 0 (gfx1250).
#ifndef MORI_PUSH_TIMING_MHZ
#define MORI_PUSH_TIMING_MHZ 100
#endif

namespace mori {
namespace ccl {

static constexpr int kPushTimingBlocks = 4;

enum class PushTimingKind : uint32_t {
  kNone = 0,
  kReduceScatter,
  kAllReduce,
  kAllGather,
  kAllToAll,
};

struct PushTiming {
  uint32_t kind, npes, S, grid, async;
  uint64_t bytes;  // per-peer copy bytes (RS/AR: my shard)
  uint64_t tStart;
  uint64_t t0[kPushTimingBlocks];
  uint64_t tPost, tIssue;
  uint64_t arr[kPushMaxSlices][kCCLMaxPeers];
  uint64_t t3[kPushTimingBlocks][kPushMaxSlices];
  uint64_t t4[kPushTimingBlocks][kPushMaxSlices];
  uint64_t dlv[kCCLMaxPeers], tDlv;
  // All-reduce push. tBPost: block 0 after pre-posting both broadcasts.
  // tCount[b][blk]: just before that block's count-in for broadcast b. The tail
  // owner (last b=1 count) records tOwn after its count, bArr[b][p] / tBDone[b]
  // as broadcast b lands from peer p / every peer, and bDlv[p] once its own b=1
  // landed at peer p.
  uint64_t tBPost;
  uint64_t tCount[2][kPushTimingBlocks];
  uint64_t tOwn, tBDone[2];
  uint64_t bArr[2][kCCLMaxPeers];
  uint64_t bDlv[kCCLMaxPeers];
  uint32_t ownBlk;
  uint64_t tEnd;
};

namespace detail {
static __device__ PushTiming gPushTiming;
}  // namespace detail

__device__ __forceinline__ PushTiming& PT() { return detail::gPushTiming; }

__device__ __forceinline__ bool PtTimedBlock() {
  return blockIdx.x < static_cast<uint32_t>(kPushTimingBlocks);
}

// The block that records per-sender arrivals: not block 0, which also issues
// the SDMA scatter, unless it is the only one.
__device__ __forceinline__ uint32_t PtArrBlock() { return gridDim.x > 1 ? 1 : 0; }

// Launch shape; thread 0 of block 0 of the launch that issues the scatter.
__device__ __forceinline__ void PtShape(PushTimingKind kind, int npes, uint32_t S, size_t bytes,
                                        bool async) {
  PushTiming& T = PT();
  T.kind = static_cast<uint32_t>(kind);
  T.npes = static_cast<uint32_t>(npes);
  T.S = S;
  T.grid = gridDim.x;
  T.async = async ? 1u : 0u;
  T.bytes = bytes;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
inline hipError_t PushTimingReset() {
  void* p = nullptr;
  hipError_t e = hipGetSymbolAddress(&p, HIP_SYMBOL(detail::gPushTiming));
  if (e != hipSuccess) return e;
  return hipMemset(p, 0, sizeof(PushTiming));
}

inline hipError_t PushTimingFetch(PushTiming* out) {
  return hipMemcpyFromSymbol(out, HIP_SYMBOL(detail::gPushTiming), sizeof(*out));
}

namespace detail {

inline void PtAppend(std::string& s, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
inline void PtAppend(std::string& s, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  s += buf;
}

// Earliest / latest non-zero per-peer stamp (zero: that lane recorded nothing).
struct PtSpan {
  int pMin = -1, pMax = -1;
  uint64_t aMin = 0, aMax = 0;
};
inline PtSpan PtPeerSpan(const uint64_t* a, uint32_t npes) {
  PtSpan r;
  for (uint32_t p = 0; p < npes && p < static_cast<uint32_t>(kCCLMaxPeers); p++) {
    if (a[p] == 0) continue;
    if (r.pMin < 0 || a[p] < r.aMin) r.aMin = a[p], r.pMin = static_cast<int>(p);
    if (r.pMax < 0 || a[p] > r.aMax) r.aMax = a[p], r.pMax = static_cast<int>(p);
  }
  return r;
}

}  // namespace detail

// Reads gPushTiming of the current device and prints it as "[PT <coll> pe=N]"
// lines in one fputs, so PEs printing from concurrent threads to the same
// stream do not interleave. Prints nothing if no push kernel ran since the
// last PushTimingReset.
inline void PushTimingDump(int myPe, FILE* out = stderr) {
  using detail::PtAppend;
  PushTiming T{};
  if (PushTimingFetch(&T) != hipSuccess) return;
  const auto kind = static_cast<PushTimingKind>(T.kind);
  const char* name = kind == PushTimingKind::kReduceScatter ? "rs"
                     : kind == PushTimingKind::kAllReduce   ? "ar"
                     : kind == PushTimingKind::kAllGather   ? "ag"
                     : kind == PushTimingKind::kAllToAll    ? "a2a"
                                                            : nullptr;
  if (name == nullptr) return;

  int dev = 0, khz = 0;
  (void)hipGetDevice(&dev);
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess) khz = 0;
  const double mhz = khz > 0 ? khz / 1000.0 : MORI_PUSH_TIMING_MHZ;
  const uint64_t base = T.async ? T.tStart : T.t0[0];
  auto us = [&](uint64_t t, uint64_t from) {
    return static_cast<double>(static_cast<int64_t>(t - from)) / mhz;
  };
  auto at = [&](uint64_t t) { return us(t, base); };

  char tag[48];
  snprintf(tag, sizeof(tag), "[PT %s pe=%d]", name, myPe);
  std::string s;
  const uint32_t S = T.S < 1 ? 1 : (T.S > kPushMaxSlices ? kPushMaxSlices : T.S);
  PtAppend(s, "%s npes=%u S=%u grid=%u bytes=%lu async=%u post=%.2f issue=%.2f us\n", tag,
           T.npes, S, T.grid, T.bytes, T.async,
           at(T.tPost), us(T.tIssue, T.tPost));
  if (T.async) PtAppend(s, "%s done-launch entry=%.2f us\n", tag, at(T.t0[0]));

  auto spanLine = [&](const char* what, const uint64_t* a, uint64_t doneT) {
    const detail::PtSpan r = detail::PtPeerSpan(a, T.npes);
    if (r.pMin < 0) return;
    PtAppend(s, "%s %s first=p%d@%.2f last=p%d@%.2f spread=%.2f", tag, what, r.pMin, at(r.aMin),
             r.pMax, at(r.aMax), us(r.aMax, r.aMin));
    if (doneT != 0) PtAppend(s, " done=%.2f", at(doneT));
    s += " us\n";
  };

  if (kind == PushTimingKind::kAllGather || kind == PushTimingKind::kAllToAll) {
    spanLine("arrive", T.arr[0], 0);
    spanLine("delivered", T.dlv, 0);
    PtAppend(s, "%s end=%.2f us\n", tag, at(T.tEnd));
  } else {
    const int nb = std::min<uint32_t>(T.grid, kPushTimingBlocks);
    for (uint32_t sl = 0; sl < S; sl++) {
      char what[32];
      snprintf(what, sizeof(what), "s=%u arrive", sl);
      spanLine(what, T.arr[sl], 0);
      for (int b = 0; b < nb; b++) {
        const uint64_t prev = sl == 0 ? T.t0[b] : T.t4[b][sl - 1];
        PtAppend(s, "%s s=%u blk%d wait=%.2f reduce=%.2f ready=%.2f done=%.2f us\n", tag, sl, b,
                 us(T.t3[b][sl], prev), us(T.t4[b][sl], T.t3[b][sl]), at(T.t3[b][sl]),
                 at(T.t4[b][sl]));
      }
    }
    if (kind == PushTimingKind::kReduceScatter) {
      spanLine("delivered", T.dlv, T.tDlv);
    } else {
      PtAppend(s, "%s bcast prepost=%.2f (done at %.2f) us\n", tag, us(T.tBPost, T.tIssue),
               at(T.tBPost));
      for (uint32_t b = (S > 1 ? 0u : 1u); b < 2; b++) {
        const uint32_t sl = b == 1 ? S - 1 : S - 2;
        for (int bl = 0; bl < nb; bl++) {
          PtAppend(s, "%s count b=%u blk%d fence=%.2f at=%.2f us\n", tag, b, bl,
                   us(T.tCount[b][bl], T.t4[bl][sl]), at(T.tCount[b][bl]));
        }
      }
      PtAppend(s, "%s tail owner=blk%u counted=%.2f us\n", tag, T.ownBlk, at(T.tOwn));
      for (uint32_t b = (S > 1 ? 0u : 1u); b < 2; b++) {
        char what[32];
        snprintf(what, sizeof(what), "bcast b=%u arrive", b);
        spanLine(what, T.bArr[b], T.tBDone[b]);
      }
      spanLine("bcast delivered", T.bDlv, 0);
      PtAppend(s, "%s end=%.2f after-own=%.2f us\n", tag, at(T.tEnd), us(T.tEnd, T.tOwn));
    }
  }
  fputs(s.c_str(), out);
  fflush(out);
}

}  // namespace ccl
}  // namespace mori

#endif  // MORI_PUSH_TIMING && HIP
