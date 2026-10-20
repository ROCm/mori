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

// Single-GPU test + micro-benchmark for WarpAccum / WarpAccumLF (and WarpAccumDynamic, which
// WarpAccumLF falls back to for accumNum outside its switch). No MPI / shmem needed.
//
// Build standalone (fastest edit-compile loop):
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 -I include -I . \
//     tests/cpp/ops/test_warp_accum.cpp -o test_warp_accum
//   gfx950 (matches the JIT build, which enables the HW fp4 converts):
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 -DHIP_ENABLE_GFX950_OCP_BUILTINS=1 -I include -I . \
//     tests/cpp/ops/test_warp_accum.cpp -o test_warp_accum
//   Add -DWARP_ACCUM_UNROLL=1 to mimic the CMake build (the JIT build keeps the header default 2);
//   it only affects WarpAccumLF calls that do not pass Unroll explicitly, which this file never
//   does.
//
// Usage:
//   ./test_warp_accum                      full correctness sweep (all types/vecs/modes)
//   ./test_warp_accum --type bf16 --vec 16 sweep restricted to one type / vec / mode
//   ./test_warp_accum --bench --type fp8 --vec 16 --mode lf2 --accum 8 --nelems 7168 \
//       --tokens 4096 --blocks 80 --warps 4 [--split 1] [--peer] [--scales] [--iters 50] [--check]
//
//   --type   f32 | bf16 | fp8 (fnuz on gfx94x, ocp otherwise) | fp8_ocp | fp8_fnuz | fp4 (fp4x2)
//   --mode   wa  = WarpAccum<T, Vec>
//            lf1 = WarpAccumLF<T, Vec, 1>
//            lf2 = WarpAccumLF<T, Vec, 2>
//   --nelems row length in T elements (for fp4 one element is a packed fp4x2 byte)
//   --split  warps per token: each warp takes a contiguous slice of the row with granularity 1,
//            like the intranode combine MultiWarpIter (slices are not vector aligned)
//   --peer   place source i on GPU (i % numGpus) so the reads go over XGMI, like the combine
//   --nulls  bitmask of sources passed as nullptr

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mori/core/transport/p2p/device_primitives.hpp"

#define HIP_CHECK(cmd)                                                                          \
  do {                                                                                          \
    hipError_t e_ = (cmd);                                                                      \
    if (e_ != hipSuccess) {                                                                     \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, \
              #cmd);                                                                            \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

namespace {

using Fp8Ocp = __hip_fp8_e4m3;
using Fp8Fnuz = __hip_fp8_e4m3_fnuz;
using Fp4x2 = mori::mori_fp4x2_e2m1;

constexpr int kMaxWarps = 16;
constexpr int kMaxAccum = 32;

// Each gfx target supports only one fp8 flavour on device; the other one gets an empty kernel
// body and the host skips it at runtime.
template <typename T>
constexpr bool kDeviceSupports = true;
#if defined(__HIP_DEVICE_COMPILE__)
#if !HIP_FP8_TYPE_OCP
template <>
constexpr bool kDeviceSupports<Fp8Ocp> = false;
#endif
#if !HIP_FP8_TYPE_FNUZ
template <>
constexpr bool kDeviceSupports<Fp8Fnuz> = false;
#endif
#endif

enum Mode { kModeWA = 0, kModeLF1 = 1, kModeLF2 = 2 };
const char* kModeNames[] = {"wa", "lf1", "lf2"};

// srcBase[i] + token * stride (+ slice offset) is the source row; nullptr stays nullptr.
template <typename T, int VecBytes, int M>
__global__ void AccumKernel(T* dest, T* const* srcBase, const float* scales, int accumNum,
                            int nTokens, size_t nelems, size_t stride, int warpsPerToken) {
  if constexpr (kDeviceSupports<T>) {
    __shared__ T* srcPtrs[kMaxWarps][kMaxAccum];
    const int laneId = threadIdx.x & (warpSize - 1);
    const int warpId = threadIdx.x / warpSize;
    const int globalWarp = blockIdx.x * (blockDim.x / warpSize) + warpId;
    const int totalWarps = gridDim.x * (blockDim.x / warpSize);
    const size_t slice = (nelems + warpsPerToken - 1) / warpsPerToken;

    for (int w = globalWarp; w < nTokens * warpsPerToken; w += totalWarps) {
      const int token = w / warpsPerToken;
      const size_t ofs = static_cast<size_t>(w % warpsPerToken) * slice;
      if (ofs >= nelems) continue;
      const size_t len = (nelems - ofs < slice) ? nelems - ofs : slice;
      const size_t rowOfs = token * stride + ofs;
      for (int i = laneId; i < accumNum; i += warpSize) {
        T* base = srcBase[i];
        srcPtrs[warpId][i] = base ? base + rowOfs : nullptr;
      }
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
      if constexpr (M == kModeWA) {
        mori::core::WarpAccum<T, VecBytes>(dest + rowOfs, srcPtrs[warpId], scales, accumNum, len);
      } else {
        mori::core::WarpAccumLF<T, VecBytes, M == kModeLF1 ? 1 : 2>(dest + rowOfs, srcPtrs[warpId],
                                                                    scales, accumNum, len);
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Host-side codecs. Values live as raw bytes on the host; every type decodes to float and knows
// the set of values it can represent, which is all the reference check needs.
// ---------------------------------------------------------------------------------------------

float Bf16ToFloat(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

float E4M3ToFloat(uint8_t c, bool fnuz) {
  const int s = c >> 7, e = (c >> 3) & 0xF, m = c & 7;
  if (fnuz && c == 0x80) return NAN;
  if (!fnuz && e == 15 && m == 7) return NAN;
  const int bias = fnuz ? 8 : 7;
  float v = (e == 0) ? std::ldexp(m / 8.0f, 1 - bias) : std::ldexp(1.0f + m / 8.0f, e - bias);
  return s ? -v : v;
}

float E2M1ToFloat(uint8_t nibble) {
  static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = kMag[nibble & 7];
  return (nibble & 8) ? -v : v;
}

struct TypeInfo {
  const char* name;
  int bytes;                         // sizeof(T)
  int lanes;                         // scalar values per T (2 for fp4x2)
  float maxAbsSum;                   // keep reference sums below this so nothing saturates
  std::vector<float> representable;  // sorted finite values, empty for f32/bf16
  std::vector<uint32_t> inputCodes;  // codes the input generator draws from (per lane)
  float (*decode)(uint32_t code);    // per-lane code -> float
  uint32_t (*laneCode)(const uint8_t* elem, int lane);
  void (*setLane)(uint8_t* elem, int lane, uint32_t code);
};

uint32_t LaneCodeBytes(const uint8_t* p, int, int bytes) {
  uint32_t c = 0;
  memcpy(&c, p, bytes);
  return c;
}

TypeInfo MakeTypeInfo(const std::string& t) {
  TypeInfo ti{};
  if (t == "f32") {
    ti = {"f32", 4, 1, 1e30f, {}, {}, nullptr, nullptr, nullptr};
    ti.decode = [](uint32_t c) {
      float f;
      memcpy(&f, &c, 4);
      return f;
    };
    ti.laneCode = [](const uint8_t* p, int) { return LaneCodeBytes(p, 0, 4); };
    ti.setLane = [](uint8_t* p, int, uint32_t c) { memcpy(p, &c, 4); };
    for (int m = 0; m < 8; ++m)
      for (int e = -3; e <= 2; ++e)
        for (int s = 0; s < 2; ++s) {
          float v = std::ldexp(1.0f + m / 8.0f, e) * (s ? -1.f : 1.f);
          uint32_t c;
          memcpy(&c, &v, 4);
          ti.inputCodes.push_back(c);
        }
  } else if (t == "bf16") {
    ti = {"bf16", 2, 1, 1e30f, {}, {}, nullptr, nullptr, nullptr};
    ti.decode = [](uint32_t c) { return Bf16ToFloat(static_cast<uint16_t>(c)); };
    ti.laneCode = [](const uint8_t* p, int) { return LaneCodeBytes(p, 0, 2); };
    ti.setLane = [](uint8_t* p, int, uint32_t c) { memcpy(p, &c, 2); };
    for (int m = 0; m < 8; ++m)
      for (int e = -3; e <= 2; ++e)
        for (int s = 0; s < 2; ++s)
          ti.inputCodes.push_back((s << 15) | ((127 + e) << 7) | (m << 4));
  } else if (t == "fp8_ocp" || t == "fp8_fnuz") {
    const bool fnuz = (t == "fp8_fnuz");
    ti = {fnuz ? "fp8_fnuz" : "fp8_ocp",
          1,
          1,
          fnuz ? 200.f : 400.f,
          {},
          {},
          nullptr,
          nullptr,
          nullptr};
    ti.decode = fnuz ? +[](uint32_t c) { return E4M3ToFloat(c, true); }
                     : +[](uint32_t c) { return E4M3ToFloat(c, false); };
    ti.laneCode = [](const uint8_t* p, int) { return static_cast<uint32_t>(p[0]); };
    ti.setLane = [](uint8_t* p, int, uint32_t c) { p[0] = static_cast<uint8_t>(c); };
    for (uint32_t c = 0; c < 256; ++c) {
      float v = ti.decode(c);
      if (std::isnan(v)) continue;
      ti.representable.push_back(v);
      if (v == 0.f || (std::fabs(v) >= 0.125f && std::fabs(v) <= 8.f)) ti.inputCodes.push_back(c);
    }
  } else if (t == "fp4") {
    ti = {"fp4", 1, 2, 6.f, {}, {}, nullptr, nullptr, nullptr};
    ti.decode = [](uint32_t c) { return E2M1ToFloat(static_cast<uint8_t>(c)); };
    ti.laneCode = [](const uint8_t* p, int lane) {
      return static_cast<uint32_t>(p[0] >> (4 * lane)) & 0xF;
    };
    ti.setLane = [](uint8_t* p, int lane, uint32_t c) {
      p[0] = static_cast<uint8_t>((p[0] & ~(0xF << (4 * lane))) | ((c & 0xF) << (4 * lane)));
    };
    for (uint32_t c = 0; c < 16; ++c) {
      float v = ti.decode(c);
      ti.representable.push_back(v);
      if (std::fabs(v) <= 2.f) ti.inputCodes.push_back(c);
    }
  } else {
    fprintf(stderr, "unknown type %s\n", t.c_str());
    exit(1);
  }
  std::sort(ti.representable.begin(), ti.representable.end());
  return ti;
}

// Distance from r to the nearest value T can represent.
double NearestDist(const TypeInfo& ti, double r) {
  if (!strcmp(ti.name, "f32")) return std::fabs(static_cast<double>(static_cast<float>(r)) - r);
  if (!strcmp(ti.name, "bf16")) {
    float f = static_cast<float>(r);
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lo = u & 0xFFFF0000u, hi = lo + 0x10000u;
    float flo, fhi;
    memcpy(&flo, &lo, 4);
    memcpy(&fhi, &hi, 4);
    return std::min(std::fabs(flo - r), std::fabs(fhi - r));
  }
  const auto& rep = ti.representable;
  auto it = std::lower_bound(rep.begin(), rep.end(), r);
  double best = 1e300;
  if (it != rep.end()) best = std::fabs(*it - r);
  if (it != rep.begin()) best = std::min(best, std::fabs(*(it - 1) - r));
  return best;
}

// ---------------------------------------------------------------------------------------------
// Kernel dispatch
// ---------------------------------------------------------------------------------------------

struct LaunchArgs {
  void* dest;
  void* const* srcBase;
  const float* scales;
  int accumNum, nTokens, warpsPerToken, blocks, warps;
  size_t nelems, stride;
  hipStream_t stream;
};
using LaunchFn = void (*)(const LaunchArgs&);

template <typename T, int V, int M>
void Launch(const LaunchArgs& a) {
  AccumKernel<T, V, M><<<a.blocks, a.warps * 64, 0, a.stream>>>(
      static_cast<T*>(a.dest), reinterpret_cast<T* const*>(a.srcBase), a.scales, a.accumNum,
      a.nTokens, a.nelems, a.stride, a.warpsPerToken);
}

template <typename T, int V>
LaunchFn PickMode(int mode) {
  switch (mode) {
    case kModeWA:
      return Launch<T, V, kModeWA>;
    case kModeLF1:
      return Launch<T, V, kModeLF1>;
    default:
      return Launch<T, V, kModeLF2>;
  }
}

template <typename T>
LaunchFn PickVec(int vec, int mode) {
  switch (vec) {
    case 4:
      return PickMode<T, 4>(mode);
    case 8:
      return PickMode<T, 8>(mode);
    default:
      return PickMode<T, 16>(mode);
  }
}

LaunchFn PickLaunch(const std::string& type, int vec, int mode) {
  if (type == "f32") return PickVec<float>(vec, mode);
  if (type == "bf16") return PickVec<hip_bfloat16>(vec, mode);
  if (type == "fp8_ocp") return PickVec<Fp8Ocp>(vec, mode);
  if (type == "fp8_fnuz") return PickVec<Fp8Fnuz>(vec, mode);
  return PickVec<Fp4x2>(vec, mode);
}

// ---------------------------------------------------------------------------------------------
// Test case: host data, device buffers, reference check
// ---------------------------------------------------------------------------------------------

constexpr uint8_t kSentinel = 0xA5;

// The sweep builds tens of thousands of cases; reuse device buffers instead of re-allocating.
void* CachedAlloc(int dev, int slot, size_t bytes) {
  struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
  };
  static std::vector<std::vector<Buf>> cache;
  if (cache.size() <= static_cast<size_t>(dev)) cache.resize(dev + 1);
  auto& bufs = cache[dev];
  if (bufs.size() <= static_cast<size_t>(slot)) bufs.resize(slot + 1);
  Buf& b = bufs[slot];
  if (b.bytes < bytes) {
    HIP_CHECK(hipSetDevice(dev));
    if (b.p) HIP_CHECK(hipFree(b.p));
    HIP_CHECK(hipMalloc(&b.p, bytes));
    b.bytes = bytes;
    HIP_CHECK(hipSetDevice(0));
  }
  return b.p;
}

enum Slot { kSlotDest = kMaxAccum, kSlotTable, kSlotScales };

struct Case {
  std::string type;
  int vec = 16, mode = kModeLF2, accumNum = 8, nTokens = 3, warpsPerToken = 1;
  int blocks = 2, warps = 4;
  size_t nelems = 7168;
  uint32_t nullMask = 0;
  bool scales = false, peer = false;
};

class Runner {
 public:
  Runner(const Case& c, uint64_t seed) : c_(c), ti_(MakeTypeInfo(c.type)), rng_(seed) {
    // Rows padded to 16B so every row starts vector aligned; the guard bytes catch overruns.
    const size_t rowBytes = c.nelems * ti_.bytes;
    stride_ = ((rowBytes + 15) / 16 * 16 + 16) / ti_.bytes;
    srcBytes_ = c.nTokens * stride_ * ti_.bytes;
    HIP_CHECK(hipGetDeviceCount(&numGpus_));
    hostSrc_.assign(c.accumNum, std::vector<uint8_t>(srcBytes_, 0));
    devSrc_.assign(c.accumNum, nullptr);
    for (int i = 0; i < c.accumNum; ++i)
      if (!(c.nullMask >> i & 1)) devSrc_[i] = CachedAlloc(c.peer ? i % numGpus_ : 0, i, srcBytes_);
    if (c.peer)
      for (int d = 1; d < numGpus_; ++d) {
        hipError_t e = hipDeviceEnablePeerAccess(d, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
      }
    devDest_ = CachedAlloc(0, kSlotDest, srcBytes_);
    devSrcTable_ = static_cast<void**>(CachedAlloc(0, kSlotTable, kMaxAccum * sizeof(void*)));
    devScales_ = static_cast<float*>(CachedAlloc(0, kSlotScales, kMaxAccum * sizeof(float)));
    HIP_CHECK(
        hipMemcpy(devSrcTable_, devSrc_.data(), c.accumNum * sizeof(void*), hipMemcpyHostToDevice));
    static const float kScales[] = {1.f, 0.5f, 0.75f, 0.25f, 1.f, 0.625f, 0.375f, 0.875f};
    for (int i = 0; i < c.accumNum; ++i) hostScales_.push_back(kScales[i % 8]);
    HIP_CHECK(hipMemcpy(devScales_, hostScales_.data(), c.accumNum * sizeof(float),
                        hipMemcpyHostToDevice));
    launch_ = PickLaunch(c.type, c.vec, c.mode);
  }

  void FillInputs() {
    std::uniform_int_distribution<size_t> pick(0, ti_.inputCodes.size() - 1);
    for (int t = 0; t < c_.nTokens; ++t)
      for (size_t e = 0; e < c_.nelems; ++e)
        for (int l = 0; l < ti_.lanes; ++l) {
          double sum = 0;
          for (int i = 0; i < c_.accumNum; ++i) {
            uint8_t* p = Elem(hostSrc_[i], t, e);
            uint32_t code = ti_.inputCodes[pick(rng_)];
            if (c_.nullMask >> i & 1) continue;
            double v = ti_.decode(code) * Scale(i);
            // Zero the term if it would push the sum out of range, so nothing saturates.
            if (std::fabs(sum + v) > ti_.maxAbsSum) code = 0, v = 0;
            ti_.setLane(p, l, code);
            sum += v;
          }
        }
    for (int i = 0; i < c_.accumNum; ++i)
      if (devSrc_[i])
        HIP_CHECK(hipMemcpy(devSrc_[i], hostSrc_[i].data(), srcBytes_, hipMemcpyHostToDevice));
  }

  void Run(int iters = 1) {
    LaunchArgs a{devDest_,    devSrcTable_, c_.scales ? devScales_ : nullptr,
                 c_.accumNum, c_.nTokens,   c_.warpsPerToken,
                 c_.blocks,   c_.warps,     c_.nelems,
                 stride_,     nullptr};
    for (int i = 0; i < iters; ++i) launch_(a);
    HIP_CHECK(hipGetLastError());
  }

  void ClearDest() { HIP_CHECK(hipMemset(devDest_, kSentinel, srcBytes_)); }

  // Returns the number of mismatching scalar values (+ clobbered guard bytes); prints a few.
  size_t Check(int maxPrint = 5) {
    std::vector<uint8_t> out(srcBytes_);
    HIP_CHECK(hipMemcpy(out.data(), devDest_, srcBytes_, hipMemcpyDeviceToHost));
    size_t bad = 0;
    auto report = [&](const char* what, int t, size_t e, int l, double ref, double got) {
      if (bad++ < static_cast<size_t>(maxPrint))
        printf("    %s token %d elem %zu lane %d: ref %.8g got %.8g\n", what, t, e, l, ref, got);
    };
    for (int t = 0; t < c_.nTokens; ++t) {
      for (size_t e = 0; e < c_.nelems; ++e)
        for (int l = 0; l < ti_.lanes; ++l) {
          double ref = 0, sumAbs = 0;
          for (int i = 0; i < c_.accumNum; ++i) {
            if (c_.nullMask >> i & 1) continue;
            double v = ti_.decode(ti_.laneCode(Elem(hostSrc_[i], t, e), l)) * Scale(i);
            ref += v;
            sumAbs += std::fabs(v);
          }
          double got = ti_.decode(ti_.laneCode(Elem(out, t, e), l));
          // got must be (one of) the representable values nearest to ref; the slack covers the
          // fp32 accumulation order inside the kernel.
          double slack = 8.0 * c_.accumNum * 1.2e-7 * sumAbs + 1e-30;
          if (!(std::fabs(got - ref) <= NearestDist(ti_, ref) * (1 + 1e-6) + slack))
            report("mismatch", t, e, l, ref, got);
        }
      for (size_t b = c_.nelems * ti_.bytes; b < stride_ * ti_.bytes; ++b)
        if (out[t * stride_ * ti_.bytes + b] != kSentinel)
          report("guard clobbered", t, b / ti_.bytes, 0, kSentinel,
                 out[t * stride_ * ti_.bytes + b]);
    }
    return bad;
  }

  size_t BytesMoved() const {
    int live = 0;
    for (int i = 0; i < c_.accumNum; ++i) live += !(c_.nullMask >> i & 1);
    return static_cast<size_t>(live + 1) * c_.nTokens * c_.nelems * ti_.bytes;
  }

 private:
  uint8_t* Elem(std::vector<uint8_t>& buf, int t, size_t e) {
    return buf.data() + (t * stride_ + e) * ti_.bytes;
  }
  double Scale(int i) const { return c_.scales ? hostScales_[i] : 1.0; }

  Case c_;
  TypeInfo ti_;
  std::mt19937_64 rng_;
  size_t stride_ = 0, srcBytes_ = 0;
  int numGpus_ = 1;
  std::vector<std::vector<uint8_t>> hostSrc_;
  std::vector<void*> devSrc_;
  std::vector<float> hostScales_;
  void* devDest_ = nullptr;
  void** devSrcTable_ = nullptr;
  float* devScales_ = nullptr;
  LaunchFn launch_ = nullptr;
};

std::string Describe(const Case& c) {
  char buf[256];
  snprintf(buf, sizeof(buf),
           "type=%s vec=%d mode=%s accum=%d nelems=%zu tokens=%d split=%d nulls=0x%x scales=%d",
           c.type.c_str(), c.vec, kModeNames[c.mode], c.accumNum, c.nelems, c.nTokens,
           c.warpsPerToken, c.nullMask, c.scales);
  return buf;
}

// Row lengths that land in each pass of WarpAccumLF: full Unroll*warp*vec steps, the dword pass,
// the scalar remainder, rows shorter than one dword pass, and a realistic hidden size.
std::vector<size_t> SweepNelems(const TypeInfo& ti, int vec, int mode) {
  const size_t unroll = (mode == kModeLF2) ? 2 : 1;
  const size_t mainStep = unroll * 64 * vec / ti.bytes;
  const size_t dwordStep = 64 * 4 / ti.bytes;
  std::vector<size_t> n = {1,
                           7,
                           37,
                           dwordStep - 1,
                           dwordStep + 5,
                           mainStep,
                           2 * mainStep,
                           2 * mainStep + dwordStep,
                           2 * mainStep + dwordStep + 3,
                           mainStep + mainStep / 2 - 1,
                           size_t{7168} / ti.lanes};
  std::sort(n.begin(), n.end());
  n.erase(std::unique(n.begin(), n.end()), n.end());
  return n;
}

std::string DefaultFp8() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  return strncmp(prop.gcnArchName, "gfx94", 5) == 0 ? "fp8_fnuz" : "fp8_ocp";
}

int RunSweep(const std::vector<std::string>& types, const std::vector<int>& vecs,
             const std::vector<int>& modes, const std::vector<int>& accums) {
  size_t cases = 0, failed = 0;
  uint64_t seed = 1;
  for (const auto& type : types) {
    const TypeInfo ti = MakeTypeInfo(type);
    for (int vec : vecs)
      for (int mode : modes) {
        size_t groupFailed = 0, groupCases = 0;
        for (int accum : accums)
          for (size_t nelems : SweepNelems(ti, vec, mode))
            for (uint32_t nulls : {0u, 0x92492492u /* every third source */, ~0u})
              for (bool scales : {false, true})
                for (int split : {1, 3}) {
                  if (nulls == ~0u && (accum != 4 || scales)) continue;
                  if (split > 1 && nelems < 64) continue;
                  Case c;
                  c.type = type;
                  c.vec = vec;
                  c.mode = mode;
                  c.accumNum = accum;
                  c.nelems = nelems;
                  c.nullMask = accum >= 32 ? nulls : nulls & ((1u << accum) - 1);
                  c.scales = scales;
                  c.warpsPerToken = split;
                  Runner r(c, seed++);
                  r.FillInputs();
                  r.ClearDest();
                  r.Run();
                  HIP_CHECK(hipDeviceSynchronize());
                  ++groupCases;
                  size_t bad = r.Check(groupFailed < 3 ? 5 : 0);
                  if (bad) {
                    if (groupFailed < 3)
                      printf("  FAIL %s: %zu bad values\n", Describe(c).c_str(), bad);
                    ++groupFailed;
                  }
                }
        printf("%-8s vec=%-2d mode=%-3s : %zu/%zu cases passed\n", type.c_str(), vec,
               kModeNames[mode], groupCases - groupFailed, groupCases);
        cases += groupCases;
        failed += groupFailed;
      }
  }
  printf("%s: %zu/%zu cases passed\n", failed ? "FAILED" : "PASSED", cases - failed, cases);
  return failed ? 1 : 0;
}

int RunBench(Case c, int iters, bool check) {
  Runner r(c, 42);
  r.FillInputs();
  r.ClearDest();
  r.Run(3);
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  r.Run(iters);
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  const double us = 1e3 * ms / iters;
  printf("%s blocks=%d warps=%d peer=%d\n", Describe(c).c_str(), c.blocks, c.warps, c.peer);
  printf("  %.2f us/launch, %.1f GB/s (live sources read + dest written)\n", us,
         r.BytesMoved() / (us * 1e3));
  int rc = 0;
  if (check) {
    size_t bad = r.Check();
    printf("  check: %s (%zu bad values)\n", bad ? "FAILED" : "ok", bad);
    rc = bad ? 1 : 0;
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return rc;
}

int ParseMode(const std::string& s) {
  for (int m = 0; m < 3; ++m)
    if (s == kModeNames[m]) return m;
  fprintf(stderr, "unknown mode %s (wa|lf1|lf2)\n", s.c_str());
  exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  bool bench = false, check = false;
  int iters = 50;
  std::string type, vecArg, modeArg, accumArg;
  Case c;
  c.type = "bf16";
  c.nTokens = 4096;
  c.blocks = 80;
  c.warps = 4;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", a.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (a == "--bench")
      bench = true;
    else if (a == "--check")
      check = true;
    else if (a == "--peer")
      c.peer = true;
    else if (a == "--scales")
      c.scales = true;
    else if (a == "--type")
      type = next();
    else if (a == "--vec")
      vecArg = next();
    else if (a == "--mode")
      modeArg = next();
    else if (a == "--accum")
      accumArg = next();
    else if (a == "--nelems")
      c.nelems = std::stoul(next());
    else if (a == "--tokens")
      c.nTokens = std::stoi(next());
    else if (a == "--blocks")
      c.blocks = std::stoi(next());
    else if (a == "--warps")
      c.warps = std::stoi(next());
    else if (a == "--split")
      c.warpsPerToken = std::stoi(next());
    else if (a == "--nulls")
      c.nullMask = std::stoul(next(), nullptr, 0);
    else if (a == "--iters")
      iters = std::stoi(next());
    else {
      fprintf(stderr, "unknown argument %s (see the header of %s)\n", a.c_str(), __FILE__);
      return 1;
    }
  }
  if (type == "fp8") type = DefaultFp8();
  if (c.warps > kMaxWarps) c.warps = kMaxWarps;

  if (bench) {
    c.type = type.empty() ? "bf16" : type;
    c.vec = vecArg.empty() ? 16 : std::stoi(vecArg);
    c.mode = modeArg.empty() ? kModeLF2 : ParseMode(modeArg);
    c.accumNum = accumArg.empty() ? 8 : std::min(std::stoi(accumArg), kMaxAccum);
    return RunBench(c, iters, check);
  }

  std::vector<std::string> types = {"f32", "bf16", DefaultFp8(), "fp4"};
  std::vector<int> vecs = {4, 8, 16}, modes = {kModeWA, kModeLF1, kModeLF2};
  std::vector<int> accums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16};
  if (!type.empty()) types = {type};
  if (!vecArg.empty()) vecs = {std::stoi(vecArg)};
  if (!modeArg.empty()) modes = {ParseMode(modeArg)};
  if (!accumArg.empty()) accums = {std::min(std::stoi(accumArg), kMaxAccum)};
  return RunSweep(types, vecs, modes, accums);
}
