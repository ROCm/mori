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
// test: cco sdma put2d over the shapes the gfx12.5 packet encodes differently.
//
// gfx12.5 counts in elements of 1..16 bytes picked from the width, pitches and
// sub-dword start offsets, has 16-bit pitch/extent fields, and falls back to the
// slice pitch for wide pitches; put2d splits a rect past one packet. gfx9's
// 32-bit byte packet needs none of that, so every case must pass on both.
//
// Cases (each checks every byte of the destination slot):
//   align_*   sub-dword src/dst offsets x widths x pitch parities
//   e4_misal  16B elements on bases that are only 4B/8B aligned
//   wide_x    70001B rows at a 70003B pitch: 1B elements, slice-pitch rows, 2 pieces
//   xz_rows   480 rows at pitches past 64Ki elements
//   tall_y    70000 rows: 2 pieces along y
//   wide_e4   2MiB rows: 16B elements, 2 pieces along x
//   batch     8 lanes, half of them 2 pieces, one remote signal for the lot
//   wide_sig  wide_x with a remote signal: the signal counts once
//
// Requires MORI_ENABLE_SDMA=1; otherwise the comm has no SDMA queues and the
// test SKIPs.

#include <string>

#include "cco_test_harness.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 1024ULL * 1024 * 1024;
static const uint8_t SENTINEL = 0xEE;

__host__ __device__ inline uint8_t pat(int rank, uint32_t row, uint32_t col) {
  return static_cast<uint8_t>(rank * 73 + row * 13 + col * 7 + (col >> 8) + (row >> 8) * 3);
}

struct Rect {
  uint32_t sx, sy;
  uint32_t dx, dy;
  uint32_t w, h;
};

struct Shape {
  uint32_t srcPitch, dstPitch;
  uint32_t srcRows, dstRows;
  __host__ __device__ size_t SrcBytes() const { return size_t(srcRows) * srcPitch; }
  __host__ __device__ size_t SlotBytes() const { return size_t(dstRows) * dstPitch; }
};

// Block p posts to peer p. lanes == 1: thread 0 posts every rect in turn;
// otherwise thread t posts rect t, all as one thread-scope batch.
template <bool kSignal>
__global__ void PutRectsKernel(ccoWindowDevice* src, ccoWindowDevice* dst, const Rect* rects,
                               int nrects, int lanes, Shape sh, ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank) return;
  auto dW = reinterpret_cast<ccoWindow_t>(dst);
  auto sW = reinterpret_cast<ccoWindow_t>(src);
  const size_t slot = size_t(dc.rank) * sh.SlotBytes();
  auto post = [&](const Rect& r) {
    sdma.put2d<ccoCoopThread, false, kSignal>(
        p, dW, slot + size_t(r.dy) * sh.dstPitch + r.dx, sh.dstPitch, sW,
        size_t(r.sy) * sh.srcPitch + r.sx, sh.srcPitch, r.w, r.h);
  };
  if (lanes == 1) {
    if (threadIdx.x != 0) return;
    for (int k = 0; k < nrects; k++) post(rects[k]);
  } else if (int(threadIdx.x) < nrects) {
    post(rects[threadIdx.x]);
  }
  __syncthreads();
  if (threadIdx.x == 0) sdma.quiet(p);
}

// Wait for each sender's queue-0 signal, then record the slot values.
__global__ void WaitSignalsKernel(ccoDevComm dc, uint64_t expected, uint64_t* seen) {
  ccoSdma sdma{dc};
  const int s = threadIdx.x;
  if (s >= dc.lsaSize || s == dc.rank) return;
  sdma.waitSignal(s, 0, expected);
  seen[s] = __hip_atomic_load(dc.sdma.signalBuf + s * dc.sdma.sdmaNumQueue, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_SYSTEM);
}

static bool CheckSlot(const std::vector<uint8_t>& host, const Shape& sh, int s,
                      const std::vector<Rect>& rects, const char* name, int rank) {
  const size_t slotBytes = sh.SlotBytes();
  std::vector<uint8_t> want(slotBytes, SENTINEL);
  for (const Rect& r : rects)
    for (uint32_t y = 0; y < r.h; y++)
      for (uint32_t x = 0; x < r.w; x++)
        want[size_t(r.dy + y) * sh.dstPitch + r.dx + x] = pat(s, r.sy + y, r.sx + x);
  const uint8_t* slot = host.data() + size_t(s) * slotBytes;
  if (memcmp(slot, want.data(), slotBytes) == 0) return true;
  long bad = 0;
  for (size_t i = 0; i < slotBytes; i++) {
    if (slot[i] == want[i]) continue;
    if (bad < 5)
      fprintf(stderr, "[rank %d] %s slot %d mismatch at (row %zu, byte %zu): got 0x%02x want 0x%02x\n",
              rank, name, s, i / sh.dstPitch, i % sh.dstPitch, slot[i], want[i]);
    bad++;
  }
  fprintf(stderr, "[rank %d] %s slot %d: %ld bad bytes\n", rank, name, s, bad);
  return false;
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  // Sized for the largest shape below (xz_rows).
  const size_t srcCap = 36ULL << 20;
  const size_t slotCap = 36ULL << 20;
  const size_t dstCap = slotCap * nranks;
  void* srcBuf = nullptr;
  void* dstBuf = nullptr;
  if (ccoMemAlloc(comm, srcCap, &srcBuf) != 0 || ccoMemAlloc(comm, dstCap, &dstBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  ccoWindow_t srcWin = nullptr, dstWin = nullptr;
  if (ccoWindowRegister(comm, srcBuf, srcCap, &srcWin) != 0 ||
      ccoWindowRegister(comm, dstBuf, dstCap, &dstWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = 0;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  ccoDevComm devComm{};
  if (ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  bool ok = true;
  if (devComm.sdma.sdmaNumQueue == 0) {
    printf("[rank %d] SKIP — no SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank);
  } else {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    auto* sW = reinterpret_cast<ccoWindowDevice*>(srcWin);
    auto* dW = reinterpret_cast<ccoWindowDevice*>(dstWin);
    Rect* dRects = nullptr;
    uint64_t* dSeen = nullptr;
    HIP_CHECK(hipMalloc(&dRects, 1024 * sizeof(Rect)));
    HIP_CHECK(hipMalloc(&dSeen, 64 * sizeof(uint64_t)));
    uint64_t expectedSig = 0;

    // signal: one remote signal per sender per call; checked to land exactly once.
    auto runCase = [&](const char* name, const Shape& sh, const std::vector<Rect>& rects,
                       int lanes, bool signal) {
      if (sh.SrcBytes() > srcCap || sh.SlotBytes() > slotCap) {
        fprintf(stderr, "[rank %d] %s: shape too big\n", rank, name);
        return false;
      }
      std::vector<uint8_t> h(sh.SrcBytes());
      for (uint32_t y = 0; y < sh.srcRows; y++)
        for (uint32_t x = 0; x < sh.srcPitch; x++) h[size_t(y) * sh.srcPitch + x] = pat(rank, y, x);
      HIP_CHECK(hipMemcpy(srcBuf, h.data(), h.size(), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dstBuf, SENTINEL, sh.SlotBytes() * nranks));
      HIP_CHECK(hipMemcpy(dRects, rects.data(), rects.size() * sizeof(Rect),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipDeviceSynchronize());
      ccoBarrierAll(comm);
      if (signal)
        PutRectsKernel<true><<<nranks, 64, 0, stream>>>(sW, dW, dRects, int(rects.size()), lanes,
                                                        sh, devComm);
      else
        PutRectsKernel<false><<<nranks, 64, 0, stream>>>(sW, dW, dRects, int(rects.size()), lanes,
                                                         sh, devComm);
      bool pass = true;
      if (signal) {
        expectedSig++;
        HIP_CHECK(hipMemset(dSeen, 0, 64 * sizeof(uint64_t)));
        WaitSignalsKernel<<<1, 64, 0, stream>>>(devComm, expectedSig, dSeen);
      }
      HIP_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      std::vector<uint8_t> host(sh.SlotBytes() * nranks);
      HIP_CHECK(hipMemcpy(host.data(), dstBuf, host.size(), hipMemcpyDeviceToHost));
      for (int s = 0; s < nranks; s++) {
        if (s == rank) continue;
        pass &= CheckSlot(host, sh, s, rects, name, rank);
      }
      if (signal) {
        uint64_t seen[64];
        HIP_CHECK(hipMemcpy(seen, dSeen, sizeof(seen), hipMemcpyDeviceToHost));
        for (int s = 0; s < nranks; s++) {
          if (s == rank || seen[s] == expectedSig) continue;
          fprintf(stderr, "[rank %d] %s: signal from %d is %lu, want %lu\n", rank, name, s,
                  (unsigned long)seen[s], (unsigned long)expectedSig);
          pass = false;
        }
      }
      printf("[rank %d] %-10s %s\n", rank, name, pass ? "PASS" : "FAIL");
      return pass;
    };

    // align: every sub-dword src/dst offset x a spread of widths, 3 rows each,
    // one band of 4 rows per rect.
    auto alignRects = [](uint32_t rowBase) {
      std::vector<Rect> v;
      const uint32_t widths[] = {1, 2, 3, 4, 6, 8, 16, 17, 64, 1000};
      uint32_t band = 0;
      for (uint32_t so = 0; so < 4; so++)
        for (uint32_t dof = 0; dof < 4; dof++)
          for (uint32_t w : widths) {
            v.push_back(Rect{16 * so + so, 1 + band % 5, 32 + dof, rowBase + 4 * band, w, 3});
            band++;
          }
      return v;
    };
    const auto aligned = alignRects(0);  // 160 rects, 640 rows
    ok &= runCase("align_A", Shape{4096, 4096, 16, 640}, aligned, 1, false);
    ok &= runCase("align_B", Shape{4100, 4098, 16, 640}, aligned, 1, false);
    ok &= runCase("align_C", Shape{4097, 4096, 16, 640}, aligned, 1, false);

    // 16B elements on bases only 4B/8B aligned (ROCr: dword alignment suffices).
    ok &= runCase("e4_misal", Shape{4096, 4096, 64, 64},
                  {Rect{4, 1, 8, 2, 2048, 16}, Rect{12, 20, 40, 30, 512, 9}}, 1, false);
    ok &= runCase("wide_x", Shape{70003, 70003, 3, 3}, {Rect{1, 0, 2, 0, 70001, 3}}, 1, false);
    ok &= runCase("xz_rows", Shape{70003, 69999, 500, 500}, {Rect{5, 7, 3, 11, 100, 480}}, 1,
                  false);
    ok &= runCase("tall_y", Shape{16, 24, 70000, 70000}, {Rect{8, 0, 16, 0, 8, 70000}}, 1,
                  false);
    ok &= runCase("wide_e4", Shape{2097216, 2097216, 2, 2}, {Rect{0, 0, 0, 0, 2097152, 2}}, 1,
                  false);
    {
      std::vector<Rect> v;
      for (uint32_t l = 0; l < 8; l++)
        v.push_back(Rect{32 * l, 0, 32 * l, 0, 24, l < 4 ? 70000u : 1000u});
      ok &= runCase("batch", Shape{256, 256, 70000, 70000}, v, 8, true);
    }
    ok &= runCase("wide_sig", Shape{70003, 70003, 3, 3}, {Rect{1, 0, 2, 0, 70001, 3}}, 1, true);

    HIP_CHECK(hipFree(dRects));
    HIP_CHECK(hipFree(dSeen));
    HIP_CHECK(hipStreamDestroy(stream));
    printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  }

  ccoDevCommDestroy(comm, &devComm);
  ccoWindowDeregister(comm, dstWin);
  ccoWindowDeregister(comm, srcWin);
  ccoMemFree(comm, dstBuf);
  ccoMemFree(comm, srcBuf);
  ccoCommDestroy(comm);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO SDMA put2d shapes", "/tmp/cco_sdma_put2d_shapes_uid",
                     19901);
}
