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
// test: cco sdma put2d/get2d — one rectangle per LARGE_SUB_WINDOW packet.
//
// Every rank owns a source matrix src[row][byte] = pat(rank, row, byte) and a
// destination window with one slot per peer, pre-filled with SENTINEL. A case
// copies a rectangle of the sender's matrix into its slot on the receiver, and
// the receiver checks every byte of the slot: the rectangle must hold the
// sender's pattern and everything around it must still be SENTINEL, so a wrong
// pitch, origin or extent shows up as either a bad value or a stray write.
//
// Cases:
//   thread      one lane posts one rect (odd origin, src/dst pitches differ)
//   block_wide  block leader, rect wider than 16K bytes -- beyond the 14-bit
//               extents of the 13-dword sub_op 4 packet
//   warp_batch  64 lanes post 64 tiles, one run and one remote signal per
//               wavefront; the receiver waits on the signal, not on a barrier
//   get2d       pull a rect from the peer
//   aggregate   four rects posted without the doorbell, then commit()
//   ring_wrap   thousands of small rects interleaved with 1-unit linear puts,
//               so runs start at every unit alignment and wrap the ring
//
// Requires MORI_ENABLE_SDMA=1; otherwise the comm has no SDMA queues and the
// test SKIPs.

#include "cco_test_harness.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 512ULL * 1024 * 1024;
static const uint8_t SENTINEL = 0xEE;

// Source matrix: SRC_ROWS x SRC_PITCH bytes.
static const uint32_t SRC_ROWS = 2048;
static const uint32_t SRC_PITCH = 24576;
// Destination slot per sender: DST_ROWS x DST_PITCH bytes.
static const uint32_t DST_ROWS = 512;
static const uint32_t DST_PITCH = 32768;
static const size_t SLOT_BYTES = size_t(DST_ROWS) * DST_PITCH;

__host__ __device__ inline uint8_t pat(int rank, uint32_t row, uint32_t col) {
  return static_cast<uint8_t>(rank * 73 + row * 13 + col * 7 + (col >> 8));
}

struct Rect {
  uint32_t sx, sy;  // origin in the source matrix (bytes, rows)
  uint32_t dx, dy;  // origin in the destination slot
  uint32_t w, h;    // bytes x rows
};

__device__ inline size_t SrcOff(const Rect& r) { return size_t(r.sy) * SRC_PITCH + r.sx; }
__device__ inline size_t DstOff(int slot, const Rect& r) {
  return size_t(slot) * SLOT_BYTES + size_t(r.dy) * DST_PITCH + r.dx;
}

// thread: lane 0 of block p posts to peer p.
__global__ void Put2DThreadKernel(ccoWindowDevice* src, ccoWindowDevice* dst, Rect r,
                                  ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank || threadIdx.x != 0) return;
  sdma.put2d(p, reinterpret_cast<ccoWindow_t>(dst), DstOff(dc.rank, r), DST_PITCH,
             reinterpret_cast<ccoWindow_t>(src), SrcOff(r), SRC_PITCH, r.w, r.h);
  sdma.quiet(p);
}

// block: every thread enters, the leader posts.
__global__ void Put2DBlockKernel(ccoWindowDevice* src, ccoWindowDevice* dst, Rect r,
                                 ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank) return;
  sdma.put2d<ccoCoopBlock>(p, reinterpret_cast<ccoWindow_t>(dst), DstOff(dc.rank, r), DST_PITCH,
                           reinterpret_cast<ccoWindow_t>(src), SrcOff(r), SRC_PITCH, r.w, r.h);
  sdma.quiet<ccoCoopBlock>(p);
}

// warp batch: lane i of block p posts tile i of an 8x8 grid to peer p, all on
// queue 0, one run and one remote signal per wavefront. Rect r is the whole grid.
__global__ void Put2DWarpBatchKernel(ccoWindowDevice* src, ccoWindowDevice* dst, Rect r,
                                     ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank) return;
  const uint32_t lane = threadIdx.x;
  const uint32_t tw = r.w / 8, th = r.h / 8;
  Rect t{r.sx + (lane % 8) * tw,
         r.sy + (lane / 8) * th,
         r.dx + (lane % 8) * tw,
         r.dy + (lane / 8) * th,
         tw,
         th};
  sdma.put2d<ccoCoopThread, false, true>(p, reinterpret_cast<ccoWindow_t>(dst), DstOff(dc.rank, t),
                                         DST_PITCH, reinterpret_cast<ccoWindow_t>(src), SrcOff(t),
                                         SRC_PITCH, t.w, t.h);
}

// Receiver side of the batch: wait for every sender's single signal.
__global__ void WaitSignalsKernel(ccoDevComm dc, uint64_t expected) {
  ccoSdma sdma{dc};
  const int s = threadIdx.x;
  if (s >= dc.lsaSize || s == dc.rank) return;
  sdma.waitSignal(s, 0, expected);
}

// get2d: lane 0 of block p pulls peer p's rect into our slot p.
__global__ void Get2DKernel(ccoWindowDevice* src, ccoWindowDevice* dst, Rect r, ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank || threadIdx.x != 0) return;
  sdma.get2d(p, reinterpret_cast<ccoWindow_t>(dst), DstOff(p, r), DST_PITCH,
             reinterpret_cast<ccoWindow_t>(src), SrcOff(r), SRC_PITCH, r.w, r.h);
  sdma.quiet(p);
}

// aggregate: four row bands of r, posted unrung, then one commit.
__global__ void Put2DAggregateKernel(ccoWindowDevice* src, ccoWindowDevice* dst, Rect r,
                                     ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank || threadIdx.x != 0) return;
  const uint32_t bh = r.h / 4;
  for (uint32_t b = 0; b < 4; b++) {
    Rect t{r.sx, r.sy + b * bh, r.dx, r.dy + b * bh, r.w, bh};
    sdma.put2d<ccoCoopThread, false, false, ccoSdmaOptFlagsAggregate>(
        p, reinterpret_cast<ccoWindow_t>(dst), DstOff(dc.rank, t), DST_PITCH,
        reinterpret_cast<ccoWindow_t>(src), SrcOff(t), SRC_PITCH, t.w, t.h);
  }
  sdma.commit(p, 0);
  sdma.quiet(p);
}

// ring wrap: iteration i copies source row i % DST_ROWS (WRAP_W bytes) to the
// same slot row, and every third iteration also posts a 32B linear copy of that
// row into a side column, shifting the next run's alignment by one unit. Rows
// repeat, but each repeat writes the same bytes, so the end state is fixed.
static const uint32_t WRAP_ITERS = 4000;
static const uint32_t WRAP_W = 64;
static const uint32_t WRAP_SIDE_X = 1024;  // linear copies land at this column
__global__ void RingWrapKernel(ccoWindowDevice* src, ccoWindowDevice* dst, ccoDevComm dc) {
  ccoSdma sdma{dc};
  const int p = blockIdx.x;
  if (p == dc.rank || threadIdx.x != 0) return;
  auto dW = reinterpret_cast<ccoWindow_t>(dst);
  auto sW = reinterpret_cast<ccoWindow_t>(src);
  for (uint32_t i = 0; i < WRAP_ITERS; i++) {
    const uint32_t row = i % DST_ROWS;
    Rect t{0, row, 0, row, WRAP_W, 1};
    sdma.put2d(p, dW, DstOff(dc.rank, t), DST_PITCH, sW, SrcOff(t), SRC_PITCH, t.w, t.h);
    if (i % 3 == 0)
      sdma.put<ccoCoopThread, false, false>(p, dW, DstOff(dc.rank, t) + WRAP_SIDE_X, sW, SrcOff(t),
                                            32);
  }
  sdma.quiet(p);
}

// Check slot s (written by sender `owner`): paint the rects onto a SENTINEL
// image and compare, so bytes outside every rect must be untouched.
static bool CheckSlot(const std::vector<uint8_t>& host, int s, int owner,
                      const std::vector<Rect>& rects, const char* name, int rank) {
  std::vector<uint8_t> want(SLOT_BYTES, SENTINEL);
  for (const Rect& r : rects)
    for (uint32_t y = 0; y < r.h; y++)
      for (uint32_t x = 0; x < r.w; x++)
        want[size_t(r.dy + y) * DST_PITCH + r.dx + x] = pat(owner, r.sy + y, r.sx + x);
  const uint8_t* slot = host.data() + size_t(s) * SLOT_BYTES;
  if (memcmp(slot, want.data(), SLOT_BYTES) == 0) return true;
  long bad = 0;
  for (size_t i = 0; i < SLOT_BYTES; i++) {
    if (slot[i] == want[i]) continue;
    if (bad < 5)
      fprintf(stderr,
              "[rank %d] %s slot %d mismatch at (row %zu, byte %zu): got 0x%02x want 0x%02x\n",
              rank, name, s, i / DST_PITCH, i % DST_PITCH, slot[i], want[i]);
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

  const size_t srcBytes = size_t(SRC_ROWS) * SRC_PITCH;
  const size_t dstBytes = SLOT_BYTES * nranks;
  void* srcBuf = nullptr;
  void* dstBuf = nullptr;
  if (ccoMemAlloc(comm, srcBytes, &srcBuf) != 0 || ccoMemAlloc(comm, dstBytes, &dstBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  {
    std::vector<uint8_t> h(srcBytes);
    for (uint32_t y = 0; y < SRC_ROWS; y++)
      for (uint32_t x = 0; x < SRC_PITCH; x++) h[size_t(y) * SRC_PITCH + x] = pat(rank, y, x);
    HIP_CHECK(hipMemcpy(srcBuf, h.data(), srcBytes, hipMemcpyHostToDevice));
  }

  ccoWindow_t srcWin = nullptr, dstWin = nullptr;
  if (ccoWindowRegister(comm, srcBuf, srcBytes, &srcWin) != 0 ||
      ccoWindowRegister(comm, dstBuf, dstBytes, &dstWin) != 0) {
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
    std::vector<uint8_t> host(dstBytes);

    // Reset our slots, then make sure every peer has before anyone writes:
    // hipMemset is async and ccoBarrierAll only syncs hosts.
    auto reset = [&]() {
      HIP_CHECK(hipMemset(dstBuf, SENTINEL, dstBytes));
      HIP_CHECK(hipDeviceSynchronize());
      ccoBarrierAll(comm);
    };
    auto fetch = [&]() {
      HIP_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      HIP_CHECK(hipMemcpy(host.data(), dstBuf, dstBytes, hipMemcpyDeviceToHost));
    };
    // slots written by peers (put) or by us (get) all carry the peer's pattern.
    auto checkAll = [&](const char* name, const std::vector<Rect>& rects) {
      bool pass = true;
      for (int s = 0; s < nranks; s++) {
        if (s == rank) continue;
        pass &= CheckSlot(host, s, s, rects, name, rank);
      }
      printf("[rank %d] %-10s %s\n", rank, name, pass ? "PASS" : "FAIL");
      return pass;
    };

    // thread: odd origin and width, distinct src/dst pitches.
    {
      const Rect r{37, 11, 129, 5, 1000, 37};
      reset();
      Put2DThreadKernel<<<nranks, 64, 0, stream>>>(sW, dW, r, devComm);
      fetch();
      ok &= checkAll("thread", {r});
    }
    // block_wide: 20000 bytes x 300 rows, far past sub_op 4's 16383 limit.
    {
      const Rect r{4096, 100, 8192, 200, 20000, 300};
      reset();
      Put2DBlockKernel<<<nranks, 256, 0, stream>>>(sW, dW, r, devComm);
      fetch();
      ok &= checkAll("block_wide", {r});
    }
    // warp_batch: 8x8 tiles of 16 rows x 96 bytes, one signal per wavefront
    // (two on wave32).
    {
      const Rect r{512, 64, 3072, 32, 8 * 96, 8 * 16};
      int warpSize = 64;
      HIP_CHECK(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, rank % numDevices));
      reset();
      Put2DWarpBatchKernel<<<nranks, 64, 0, stream>>>(sW, dW, r, devComm);
      WaitSignalsKernel<<<1, 64, 0, stream>>>(devComm, 64 / warpSize);
      fetch();
      ok &= checkAll("warp_batch", {r});
    }
    // get2d
    {
      const Rect r{200, 3, 64, 300, 4000, 129};
      reset();
      Get2DKernel<<<nranks, 64, 0, stream>>>(sW, dW, r, devComm);
      fetch();
      ok &= checkAll("get2d", {r});
    }
    // aggregate
    {
      const Rect r{8, 1000, 16, 7, 2048, 64};
      reset();
      Put2DAggregateKernel<<<nranks, 64, 0, stream>>>(sW, dW, r, devComm);
      fetch();
      ok &= checkAll("aggregate", {r});
    }
    // ring_wrap
    {
      reset();
      RingWrapKernel<<<nranks, 64, 0, stream>>>(sW, dW, devComm);
      fetch();
      std::vector<Rect> rects;
      rects.push_back(Rect{0, 0, 0, 0, WRAP_W, DST_ROWS});
      for (uint32_t i = 0; i < WRAP_ITERS; i += 3)
        rects.push_back(Rect{0, i % DST_ROWS, WRAP_SIDE_X, i % DST_ROWS, 32, 1});
      ok &= checkAll("ring_wrap", rects);
    }

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
  return ccoTestMain(argc, argv, "CCO SDMA put2d", "/tmp/cco_sdma_put2d_uid", 19897);
}
