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
// test: cco sdma edge cases — three behaviors not covered by the put/get tests:
//
//   1. Ring-buffer wraparound. Post WRAP_ITERS puts on a single queue so the
//      cumulative command bytes far exceed CCO_SDMA_QUEUE_SIZE (256KB), forcing
//      ReserveQueueSpace to wrap and pad NOPs at the tail, plus the hw-read-index
//      backpressure path. All iterations write the same slice, so the result is
//      deterministic.
//   2. Zero-byte transfer. put(bytes=0) then quiet must NOT hang and must write
//      nothing (validates the size==0 guard in ccoSdmaPostCopy: it returns before
//      posting and leaves expectedSignals untouched, so quiet drains instantly).
//   3. copy_size < queNum. A warp-coop put smaller than the queue count drives
//      the rand_size==0 branch (queue 0 sends the whole thing, the rest idle).
//
// Requires MORI_ENABLE_SDMA=1; otherwise the comm has no SDMA queues and SKIPs.

#include "cco_test_harness.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 64;      // elements per rank-pair (small; wrap is about packet count)
static const int WRAP_ITERS = 8192;  // ~60B/put*8192 ≈ 480KB > 256KB queue → wraps ~1.9x
static const float ZERO_MARKER = 7.0f;  // recv preset; must survive a 0-byte put untouched

// 1. wraparound: thread p issues WRAP_ITERS identical puts on peer p's queue 0,
//    then a single quiet(p) drains all of them. Last-writer-wins; all writes are
//    identical, so peer p's recv slot myRank ends up = our slice for p.
__global__ void SdmaWrapKernel(mori::cco::ccoWindowDevice* sendWin,
                               mori::cco::ccoWindowDevice* recvWin, size_t count, int iters,
                               mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  int myRank = devComm.rank;
  int nRanks = devComm.lsaSize;
  size_t perPair = count * sizeof(float);

  int p = threadIdx.x;
  if (p >= nRanks || p == myRank) return;
  for (int it = 0; it < iters; it++) {
    sdma.put(p, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPair,
             reinterpret_cast<ccoWindow_t>(sendWin), p * perPair, perPair);
  }
  sdma.quiet(p);
}

// 2. zero-byte: thread p issues a 0-byte put to peer p then quiet(p). The guard
//    must skip the post entirely, so quiet returns immediately and peer p's recv
//    slot stays at ZERO_MARKER.
__global__ void SdmaZeroKernel(mori::cco::ccoWindowDevice* sendWin,
                               mori::cco::ccoWindowDevice* recvWin, size_t count,
                               mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  int myRank = devComm.rank;
  int nRanks = devComm.lsaSize;
  size_t perPair = count * sizeof(float);

  int p = threadIdx.x;
  if (p >= nRanks || p == myRank) return;
  sdma.put(p, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPair,
           reinterpret_cast<ccoWindow_t>(sendWin), p * perPair, /*bytes=*/0);
  sdma.quiet(p);
}

// 3. tiny (copy_size < queNum): warp-coop put of `bytes` (< sdmaNumQueue) into
//    the first bytes of peer p's recv slot. Drives ccoSdmaPutMultiQueue's
//    rand_size==0 path (queue 0 sends everything). Byte-granular so it works for
//    any small `bytes`.
__global__ void SdmaTinyKernel(mori::cco::ccoWindowDevice* sendWin,
                               mori::cco::ccoWindowDevice* recvWin, size_t slotBytes, size_t bytes,
                               mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  int myRank = devComm.rank;
  int nRanks = devComm.lsaSize;

  int p = blockIdx.x;  // one warp (block) per peer
  if (p >= nRanks || p == myRank) return;
  // src = start of our per-pair send region for p; dst = start of peer p's recv slot myRank.
  sdma.put<ccoCoopWarp>(p, reinterpret_cast<ccoWindow_t>(recvWin), myRank * slotBytes,
                        reinterpret_cast<ccoWindow_t>(sendWin), p * slotBytes, bytes);
  sdma.quiet<ccoCoopWarp>(p);
}

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  g_rank = rank;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  size_t bufSize = COUNT * nranks * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  std::vector<float> hostSend(COUNT * nranks);
  for (int p = 0; p < nranks; p++)
    for (size_t i = 0; i < COUNT; i++)
      hostSend[p * COUNT + i] = static_cast<float>(rank * 1000 + p * 100 + i);
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));

  mori::cco::ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = 0;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  bool ok = true;
  if (devComm.sdma.sdmaNumQueue == 0) {
    printf("[rank %d] SKIP — no SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank);
  } else {
    const uint32_t queNum = devComm.sdma.sdmaNumQueue;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ── phase 1: wraparound ──────────────────────────────────────────────────
    // recv slot s ends up written by rank s (last-writer-wins over WRAP_ITERS).
    HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));
    mori::cco::ccoBarrierAll(comm);
    SdmaWrapKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, WRAP_ITERS, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    bool ok_wrap = true;
    {
      std::vector<float> host(COUNT * nranks);
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
      for (int s = 0; s < nranks && ok_wrap; s++) {
        if (s == rank) continue;
        for (size_t i = 0; i < COUNT; i++) {
          float expected = static_cast<float>(s * 1000 + rank * 100 + i);
          if (host[s * COUNT + i] != expected) {
            fprintf(stderr, "[rank %d] WRAP mismatch [src=%d][%zu]: got %.0f expected %.0f\n", rank,
                    s, i, host[s * COUNT + i], expected);
            ok_wrap = false;
            break;
          }
        }
      }
    }

    // ── phase 2: zero-byte (guard must skip post; quiet must not hang) ────────
    // Preset recv to a marker; a 0-byte put must leave it untouched.
    std::vector<float> marker(COUNT * nranks, ZERO_MARKER);
    HIP_CHECK(hipMemcpy(recvBuf, marker.data(), bufSize, hipMemcpyHostToDevice));
    mori::cco::ccoBarrierAll(comm);
    SdmaZeroKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));  // hangs here if the guard is broken
    mori::cco::ccoBarrierAll(comm);
    bool ok_zero = true;
    {
      std::vector<float> host(COUNT * nranks);
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < COUNT * static_cast<size_t>(nranks); i++) {
        if (host[i] != ZERO_MARKER) {
          fprintf(stderr, "[rank %d] ZERO wrote data at [%zu]: got %.0f expected %.0f\n", rank, i,
                  host[i], ZERO_MARKER);
          ok_zero = false;
          break;
        }
      }
    }

    // ── phase 3: copy_size < queNum (rand_size==0 branch) ────────────────────
    // Only meaningful when queNum >= 2 (queNum==1 never hits the branch).
    bool ok_tiny = true;
    if (queNum >= 2) {
      const size_t slotBytes = COUNT * sizeof(float);
      const size_t tinyBytes = queNum - 1;  // < queNum → rand_size == 0
      // Byte-level pattern in send region; recv preset to 0x00.
      std::vector<unsigned char> hostSendB(bufSize);
      for (int p = 0; p < nranks; p++)
        for (size_t i = 0; i < slotBytes; i++)
          hostSendB[p * slotBytes + i] = static_cast<unsigned char>((rank * 17 + i) & 0xff);
      HIP_CHECK(hipMemcpy(sendBuf, hostSendB.data(), bufSize, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(recvBuf, 0x00, bufSize));

      mori::cco::ccoBarrierAll(comm);
      SdmaTinyKernel<<<nranks, 64, 0, stream>>>(sendWin, recvWin, slotBytes, tinyBytes, devComm);
      HIP_CHECK(hipStreamSynchronize(stream));
      mori::cco::ccoBarrierAll(comm);

      std::vector<unsigned char> hostB(bufSize);
      HIP_CHECK(hipMemcpy(hostB.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
      for (int s = 0; s < nranks && ok_tiny; s++) {
        if (s == rank) continue;
        for (size_t i = 0; i < slotBytes; i++) {
          unsigned char expected =
              (i < tinyBytes) ? static_cast<unsigned char>((s * 17 + i) & 0xff) : 0x00;
          unsigned char got = hostB[s * slotBytes + i];
          if (got != expected) {
            fprintf(stderr, "[rank %d] TINY mismatch [src=%d][byte %zu]: got %u expected %u\n",
                    rank, s, i, got, expected);
            ok_tiny = false;
            break;
          }
        }
      }
    } else {
      printf("[rank %d] TINY skipped (queNum=%u < 2)\n", rank, queNum);
    }

    ok = ok_wrap && ok_zero && ok_tiny;
    HIP_CHECK(hipStreamDestroy(stream));
    printf("[rank %d] wrap=%s zero=%s tiny=%s %s\n", rank, ok_wrap ? "PASS" : "FAIL",
           ok_zero ? "PASS" : "FAIL", ok_tiny ? "PASS" : "FAIL", ok ? "PASSED" : "FAILED");
  }

  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO SDMA edge cases", "/tmp/cco_sdma_edge_uid", 19893);
}
