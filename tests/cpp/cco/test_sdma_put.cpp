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
// test: cco sdma put — intra-node copy-engine (ccoSdma::put), single node.
//
// alltoall via SDMA put. Layout (owner, dest-slot, idx):
//   sendBuf[p*COUNT + i] = myRank*1000 + p*100 + i
// Each rank copies its slice for peer p into peer p's recv slot `myRank`, so
//   recv[s*COUNT+i] == s*1000 + myRank*100 + i   (written by rank s)
//
// One thread per peer (distinct SDMA queue per peer); quiet() drains completion.
// Requires MORI_ENABLE_SDMA=1; otherwise the comm has no SDMA queues and the
// test SKIPs.

#include "cco_test_harness.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;  // elements per rank-pair

// thread scope: thread p writes our slice into peer p's recv slot indexed by
// myRank, over peer p's queue 0.
__global__ void SdmaPutKernel(mori::cco::ccoWindowDevice* sendWin,
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
           reinterpret_cast<ccoWindow_t>(sendWin), p * perPair, perPair);
  sdma.quiet(p);
}

// warp scope: one warp (block) per peer p writes our slice into peer p's recv
// slot, with the transfer split across all of peer p's SDMA queues internally.
__global__ void SdmaPutWarpKernel(mori::cco::ccoWindowDevice* sendWin,
                                  mori::cco::ccoWindowDevice* recvWin, size_t count,
                                  mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  int myRank = devComm.rank;
  int nRanks = devComm.lsaSize;
  size_t perPair = count * sizeof(float);

  int p = blockIdx.x;  // one warp per peer
  if (p >= nRanks || p == myRank) return;
  sdma.put<ccoCoopWarp>(p, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPair,
                        reinterpret_cast<ccoWindow_t>(sendWin), p * perPair, perPair);
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
  HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));

  mori::cco::ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // SDMA needs no GDA connectivity; the signal pool is materialized whenever the
  // comm has SDMA queues (set up in ccoCommCreate for canSDMA peers).
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
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // recv is filled by peers writing into our window; validate the alltoall.
    auto verify = [&](const char* scope) {
      std::vector<float> host(COUNT * nranks);
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
      for (int s = 0; s < nranks; s++) {
        if (s == rank) continue;
        for (size_t i = 0; i < COUNT; i++) {
          float expected = static_cast<float>(s * 1000 + rank * 100 + i);
          if (host[s * COUNT + i] != expected) {
            fprintf(stderr, "[rank %d] PUT(%s) mismatch [src=%d][%zu]: got %.0f expected %.0f\n",
                    rank, scope, s, i, host[s * COUNT + i], expected);
            return false;
          }
        }
      }
      return true;
    };

    // thread scope: one thread per peer, peer's queue 0.
    mori::cco::ccoBarrierAll(comm);
    SdmaPutKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    const bool ok_thread = verify("thread");

    // warp scope: one warp per peer, transfer split across all of the peer's queues.
    HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));
    mori::cco::ccoBarrierAll(comm);
    SdmaPutWarpKernel<<<nranks, 64, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    const bool ok_warp = verify("warp");

    ok = ok_thread && ok_warp;
    HIP_CHECK(hipStreamDestroy(stream));
    printf("[rank %d] thread=%s warp=%s %s\n", rank, ok_thread ? "PASS" : "FAIL",
           ok_warp ? "PASS" : "FAIL", ok ? "PASSED" : "FAILED");
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
  return ccoTestMain(argc, argv, "CCO SDMA put", "/tmp/cco_sdma_put_uid", 19890);
}
