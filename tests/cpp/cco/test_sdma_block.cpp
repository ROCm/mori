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
// test: cco sdma put/get — BLOCK coop scope (ccoSdma::put/get<ccoCoopBlock>).
//
// Same alltoall as test_sdma_put/get.cpp, but driven at block scope. One block
// per peer, launched with 128 threads (2 warps > warpSize) so the block path is
// genuinely exercised — a block-collective transfer distributes internally
// across every SDMA queue (one thread per queue, queueId == threadIdx.x).
// Requires MORI_ENABLE_SDMA=1; otherwise the comm has no SDMA queues and SKIPs.

#include "cco_test_harness.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;       // elements per rank-pair
static const int BLOCK_THREADS = 128;  // 2 warps, > warpSize, to exercise block path

// block scope: block p writes our slice into peer p's recv slot myRank, split
// across all of peer p's SDMA queues by the block collectively.
__global__ void SdmaPutBlockKernel(mori::cco::ccoWindowDevice* sendWin,
                                   mori::cco::ccoWindowDevice* recvWin, size_t count,
                                   mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  int myRank = devComm.rank;
  int nRanks = devComm.lsaSize;
  size_t perPair = count * sizeof(float);

  int p = blockIdx.x;  // one block per peer
  if (p >= nRanks || p == myRank) return;
  sdma.put<ccoCoopBlock>(p, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPair,
                         reinterpret_cast<ccoWindow_t>(sendWin), p * perPair, perPair);
  sdma.quiet<ccoCoopBlock>(p);
}

// block scope: block p pulls peer p's slice destined for us into recv slot p,
// split across all of peer p's SDMA queues by the block collectively.
__global__ void SdmaGetBlockKernel(mori::cco::ccoWindowDevice* sendWin,
                                   mori::cco::ccoWindowDevice* recvWin, size_t count,
                                   mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  int myRank = devComm.rank;
  int nRanks = devComm.lsaSize;
  size_t perPair = count * sizeof(float);

  int p = blockIdx.x;  // one block per peer
  if (p >= nRanks || p == myRank) return;
  sdma.get<ccoCoopBlock>(p, reinterpret_cast<ccoWindow_t>(recvWin), p * perPair,
                         reinterpret_cast<ccoWindow_t>(sendWin), myRank * perPair, perPair);
  sdma.quiet<ccoCoopBlock>(p);
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

    // put: recv is filled by peers writing into our window (dest-slot myRank).
    auto verifyPut = [&]() {
      std::vector<float> host(COUNT * nranks);
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
      for (int s = 0; s < nranks; s++) {
        if (s == rank) continue;
        for (size_t i = 0; i < COUNT; i++) {
          float expected = static_cast<float>(s * 1000 + rank * 100 + i);
          if (host[s * COUNT + i] != expected) {
            fprintf(stderr, "[rank %d] PUT(block) mismatch [src=%d][%zu]: got %.0f expected %.0f\n",
                    rank, s, i, host[s * COUNT + i], expected);
            return false;
          }
        }
      }
      return true;
    };
    // get: we pull peers' slices into our recv slot p.
    auto verifyGet = [&]() {
      std::vector<float> host(COUNT * nranks);
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
      for (int p = 0; p < nranks; p++) {
        if (p == rank) continue;
        for (size_t i = 0; i < COUNT; i++) {
          float expected = static_cast<float>(p * 1000 + rank * 100 + i);
          if (host[p * COUNT + i] != expected) {
            fprintf(stderr,
                    "[rank %d] GET(block) mismatch [peer=%d][%zu]: got %.0f expected %.0f\n", rank,
                    p, i, host[p * COUNT + i], expected);
            return false;
          }
        }
      }
      return true;
    };

    // block-coop put: one block per peer, transfer split across the peer's queues.
    mori::cco::ccoBarrierAll(comm);
    SdmaPutBlockKernel<<<nranks, BLOCK_THREADS, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    const bool ok_put = verifyPut();

    // block-coop get: one block per peer, transfer split across the peer's queues.
    HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));
    mori::cco::ccoBarrierAll(comm);
    SdmaGetBlockKernel<<<nranks, BLOCK_THREADS, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    const bool ok_get = verifyGet();

    ok = ok_put && ok_get;
    HIP_CHECK(hipStreamDestroy(stream));
    printf("[rank %d] put=%s get=%s %s\n", rank, ok_put ? "PASS" : "FAIL", ok_get ? "PASS" : "FAIL",
           ok ? "PASSED" : "FAILED");
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
  return ccoTestMain(argc, argv, "CCO SDMA put/get (block)", "/tmp/cco_sdma_block_uid", 19892);
}
