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
// test: cco sdma thread-scope put with several issuers per wavefront.
//
// Thread scope posts per lane, so a wavefront can have many lanes calling put at
// once. The shapes that matter:
//   1. same queue, many lanes of one warp — the case that deadlocks if each lane
//      runs the commit chain itself: the lane that must publish committedWptr
//      cannot leave the spin loop until the lane waiting on it does.
//   2. mixed queues in one warp — lanes must be grouped per queue, and a warp
//      must not end up holding two un-published reservations at once.
//   3. same queue across warps — the pre-existing cross-warp shape.
//   4. lanes partially masked off — grouping has to use the active lanes only.
//   5. the uniformQueue promise — same shape as 1, but the caller asserts every
//      active lane shares the queue, which takes a different code path.
// Every lane writes its own chunk, so the receiver can tell them apart.
//
// Requires MORI_ENABLE_SDMA=1; otherwise SKIPs.

#include "cco_test_harness.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t CHUNK = 256;    // bytes per issuing lane
static const int MAX_ISSUERS = 64;  // one wavefront
static const size_t BUF_BYTES = CHUNK * MAX_ISSUERS;

__host__ __device__ static inline uint32_t Pattern(int owner, size_t idx) {
  return (static_cast<uint32_t>(owner) << 20) | static_cast<uint32_t>(idx & 0xfffff);
}

// 1 & 4: `issuers` lanes of one wavefront post to the same queue. `stride` > 1
// leaves gaps so the active set is not a prefix.
__global__ void SdmaSameQueueKernel(mori::cco::ccoWindowDevice* sendWin,
                                    mori::cco::ccoWindowDevice* recvWin, int issuers, int stride,
                                    mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int peer = (devComm.lsaRank + 1) % devComm.lsaSize;
  const int lane = threadIdx.x;
  const int slot = lane / stride;
  if (lane % stride == 0 && slot < issuers) {
    sdma.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), slot * CHUNK,
             reinterpret_cast<ccoWindow_t>(sendWin), slot * CHUNK, CHUNK, 0);
  }
  sdma.quiet<ccoCoopBlock>(peer);
}

// 5: same as 1, but on the uniformQueue path. Every active lane really does share
//    the queue, so the promise holds; a build with MORI_CCO_SDMA_DEBUG traps if it
//    ever stops holding.
__global__ void SdmaUniformQueueKernel(mori::cco::ccoWindowDevice* sendWin,
                                       mori::cco::ccoWindowDevice* recvWin, int issuers,
                                       mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int peer = (devComm.lsaRank + 1) % devComm.lsaSize;
  const int lane = threadIdx.x;
  if (lane < issuers) {
    sdma.put<ccoCoopThread, false, false, ccoSdmaOptFlagsDefault, ccoSdmaThreadSameQueue>(
        peer, reinterpret_cast<ccoWindow_t>(recvWin), lane * CHUNK,
        reinterpret_cast<ccoWindow_t>(sendWin), lane * CHUNK, CHUNK, 0);
  }
  sdma.quiet<ccoCoopBlock>(peer);
}

// 2: lanes of one wavefront spread over every queue, round robin, so the warp
// has several groups to post in turn.
__global__ void SdmaMixedQueueKernel(mori::cco::ccoWindowDevice* sendWin,
                                     mori::cco::ccoWindowDevice* recvWin, int issuers,
                                     mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int peer = (devComm.lsaRank + 1) % devComm.lsaSize;
  const int nq = static_cast<int>(devComm.sdma.sdmaNumQueue);
  const int lane = threadIdx.x;
  if (lane < issuers) {
    sdma.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), lane * CHUNK,
             reinterpret_cast<ccoWindow_t>(sendWin), lane * CHUNK, CHUNK, lane % nq);
  }
  sdma.quiet<ccoCoopBlock>(peer);
}

// 3: one issuer per wavefront, all on the same queue.
__global__ void SdmaCrossWarpKernel(mori::cco::ccoWindowDevice* sendWin,
                                    mori::cco::ccoWindowDevice* recvWin, int issuers,
                                    mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int peer = (devComm.lsaRank + 1) % devComm.lsaSize;
  const int warp = threadIdx.x / 64;
  if ((threadIdx.x % 64) == 0 && warp < issuers) {
    sdma.put(peer, reinterpret_cast<ccoWindow_t>(recvWin), warp * CHUNK,
             reinterpret_cast<ccoWindow_t>(sendWin), warp * CHUNK, CHUNK, 0);
  }
  sdma.quiet<ccoCoopBlock>(peer);
}

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  g_rank = rank;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) return 1;

  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, BUF_BYTES, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, BUF_BYTES, &recvBuf) != 0)
    return 1;

  const size_t words = BUF_BYTES / sizeof(uint32_t);
  std::vector<uint32_t> hostSend(words);
  for (size_t i = 0; i < words; i++) hostSend[i] = Pattern(rank, i);
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), BUF_BYTES, hipMemcpyHostToDevice));

  mori::cco::ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, BUF_BYTES, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, BUF_BYTES, &recvWin) != 0)
    return 1;

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = 0;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) return 1;

  bool ok = true;
  if (devComm.sdma.sdmaNumQueue == 0) {
    printf("[rank %d] SKIP — no SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank);
    fflush(stdout);
  } else {
    const int src = (devComm.lsaRank - 1 + devComm.lsaSize) % devComm.lsaSize;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    std::vector<uint32_t> host(words);

    // Chunks land at the same offset the sender used, so chunk c must hold the
    // source's pattern for that chunk.
    auto verify = [&](int chunks, const char* what) {
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, BUF_BYTES, hipMemcpyDeviceToHost));
      for (int c = 0; c < chunks; c++)
        for (size_t i = 0; i < CHUNK / sizeof(uint32_t); i++) {
          const size_t idx = c * (CHUNK / sizeof(uint32_t)) + i;
          if (host[idx] != Pattern(src, idx)) {
            fprintf(stderr, "[rank %d] %s chunk %d word %zu: got %08x want %08x\n", rank, what, c,
                    i, host[idx], Pattern(src, idx));
            return false;
          }
        }
      return true;
    };

    auto run = [&](const char* what, int chunks, auto launch) {
      HIP_CHECK(hipMemset(recvBuf, 0xff, BUF_BYTES));
      mori::cco::ccoBarrierAll(comm);
      launch();
      HIP_CHECK(hipStreamSynchronize(stream));  // hangs here if issuers deadlock
      mori::cco::ccoBarrierAll(comm);
      const bool good = verify(chunks, what);
      if (!good) ok = false;
      return good;
    };

    bool okSame = true, okGap = true, okMixed = true, okCross = true, okUniform = true;
    for (int n : {2, 4, 16, 64}) {
      char tag[64];
      snprintf(tag, sizeof(tag), "same-queue x%d", n);
      okSame = run(tag, n,
                   [&] {
                     SdmaSameQueueKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, n, 1, devComm);
                   }) &&
               okSame;
    }
    // active lanes 0,4,8,... — the group is not a prefix of the wavefront
    okGap = run("gapped lanes", 16, [&] {
      SdmaSameQueueKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, 16, 4, devComm);
    });

    for (int n : {1, 2, 16, 64}) {
      char tag[64];
      snprintf(tag, sizeof(tag), "uniform-queue x%d", n);
      okUniform = run(tag, n,
                      [&] {
                        SdmaUniformQueueKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, n, devComm);
                      }) &&
                  okUniform;
    }
    for (int n : {4, 16, 64}) {
      char tag[64];
      snprintf(tag, sizeof(tag), "mixed-queue x%d", n);
      okMixed =
          run(tag, n,
              [&] { SdmaMixedQueueKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, n, devComm); }) &&
          okMixed;
    }
    for (int n : {2, 8}) {
      char tag[64];
      snprintf(tag, sizeof(tag), "cross-warp x%d", n);
      okCross = run(tag, n,
                    [&] {
                      SdmaCrossWarpKernel<<<1, n * 64, 0, stream>>>(sendWin, recvWin, n, devComm);
                    }) &&
                okCross;
    }

    printf("[rank %d] same=%s gapped=%s uniform=%s mixed=%s cross-warp=%s %s\n", rank,
           okSame ? "PASS" : "FAIL", okGap ? "PASS" : "FAIL", okUniform ? "PASS" : "FAIL",
           okMixed ? "PASS" : "FAIL", okCross ? "PASS" : "FAIL", ok ? "PASSED" : "FAILED");
    fflush(stdout);
    HIP_CHECK(hipStreamDestroy(stream));
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
  return ccoTestMain(argc, argv, "CCO SDMA warp issuers", "/tmp/cco_sdma_warpissue_uid", 19919);
}
