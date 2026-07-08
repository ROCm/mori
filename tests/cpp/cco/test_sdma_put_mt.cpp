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
// test: cco sdma put — single process, N host threads, one GPU per thread.
//
// Same alltoall-via-SDMA-put semantics as test_sdma_put.cpp, but instead of
// forking one process per rank, this launches nRanks std::threads inside ONE
// process. Each thread binds its own GPU (hipSetDevice), builds its own ccoComm
// against the shared uniqueId, and runs put/quiet. Every comm reserves its own
// flat VA and maps all peers relative to its own flatBase, so the per-thread
// comms are self-consistent — nothing about the LSA peer addressing assumes a
// one-process-per-rank layout.
//
// Layout (owner, dest-slot, idx):
//   sendBuf[p*COUNT + i] = myRank*1000 + p*100 + i
// Each rank copies its slice for peer p into peer p's recv slot `myRank`, so
//   recv[s*COUNT+i] == s*1000 + myRank*100 + i   (written by rank s)
//
// Requires MORI_ENABLE_SDMA=1; otherwise the comms have no SDMA queues and the
// test SKIPs.

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/cco/cco.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;  // elements per rank-pair

// Thread-scoped HIP check: on a hard HIP error, abort the whole process (a
// half-initialized rank would otherwise deadlock its peers at the next barrier).
#define HIP_CHECK_MT(rank, cmd)                                                                  \
  do {                                                                                           \
    hipError_t e = (cmd);                                                                        \
    if (e != hipSuccess) {                                                                       \
      fprintf(stderr, "[rank %d] HIP error %d (%s) at %s:%d\n", (rank), e, hipGetErrorString(e), \
              __FILE__, __LINE__);                                                               \
      fflush(stderr);                                                                            \
      _exit(1);                                                                                  \
    }                                                                                            \
  } while (0)

// thread scope: thread p writes our slice into peer p's recv slot indexed by
// myRank, over peer p's queue 0.
__global__ void SdmaPutMtKernel(mori::cco::ccoWindowDevice* sendWin,
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
__global__ void SdmaPutMtWarpKernel(mori::cco::ccoWindowDevice* sendWin,
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

struct ThreadResult {
  bool skipped = false;
  bool ok = false;
};

static void worker(int rank, int nranks, int dev, const mori::cco::ccoUniqueId& uid,
                   ThreadResult* res) {
  HIP_CHECK_MT(rank, hipSetDevice(dev));
  printf("[rank %d/%d] tid=%ld GPU=%d\n", rank, nranks, (long)gettid(), dev);
  fflush(stdout);

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    _exit(1);
  }

  size_t bufSize = COUNT * nranks * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    _exit(1);
  }

  std::vector<float> hostSend(COUNT * nranks);
  for (int p = 0; p < nranks; p++)
    for (size_t i = 0; i < COUNT; i++)
      hostSend[p * COUNT + i] = static_cast<float>(rank * 1000 + p * 100 + i);
  HIP_CHECK_MT(rank, hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));
  HIP_CHECK_MT(rank, hipMemset(recvBuf, 0xff, bufSize));

  mori::cco::ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    _exit(1);
  }

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = 0;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    _exit(1);
  }

  bool ok = true;
  if (devComm.sdma.sdmaNumQueue == 0) {
    printf("[rank %d] SKIP — no SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank);
    res->skipped = true;
  } else {
    hipStream_t stream;
    HIP_CHECK_MT(rank, hipStreamCreate(&stream));

    // recv is filled by peers writing into our window; validate the alltoall.
    auto verify = [&](const char* scope) {
      std::vector<float> host(COUNT * nranks);
      HIP_CHECK_MT(rank, hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
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
    SdmaPutMtKernel<<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK_MT(rank, hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    const bool ok_thread = verify("thread");

    // warp scope: one warp per peer, transfer split across all of the peer's queues.
    HIP_CHECK_MT(rank, hipMemset(recvBuf, 0xff, bufSize));
    mori::cco::ccoBarrierAll(comm);
    SdmaPutMtWarpKernel<<<nranks, 64, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
    HIP_CHECK_MT(rank, hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
    const bool ok_warp = verify("warp");

    ok = ok_thread && ok_warp;
    HIP_CHECK_MT(rank, hipStreamDestroy(stream));
    printf("[rank %d] thread=%s warp=%s %s\n", rank, ok_thread ? "PASS" : "FAIL",
           ok_warp ? "PASS" : "FAIL", ok ? "PASSED" : "FAILED");
    res->ok = ok;
  }

  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);
}

int main(int argc, char** argv) {
  int numDevices = 0;
  if (hipGetDeviceCount(&numDevices) != hipSuccess || numDevices < 2) {
    printf("Need at least 2 GPUs.\n");
    return 1;
  }
  int nranks = numDevices;
  if (argc > 1) {
    int req = atoi(argv[1]);
    if (req >= 2 && req < nranks) nranks = req;
  }

  printf("=== CCO SDMA put Test (threads, %d ranks / %d GPUs) ===\n", nranks, numDevices);
  fflush(stdout);

  // One process => one uniqueId, shared by every thread (rank 0's socket
  // rendezvous; created here on the main thread, connected during CommCreate).
  mori::cco::ccoUniqueId uid;
  if (mori::cco::ccoGetUniqueId(&uid) != 0) {
    fprintf(stderr, "ccoGetUniqueId failed (set MORI_SOCKET_IFNAME=<iface>)\n");
    return 1;
  }

  std::vector<ThreadResult> results(nranks);
  std::vector<std::thread> threads;
  threads.reserve(nranks);
  for (int r = 0; r < nranks; r++) {
    threads.emplace_back(worker, r, nranks, r % numDevices, std::cref(uid), &results[r]);
  }
  for (auto& t : threads) t.join();

  bool anySkipped = false;
  int passed = 0;
  for (int r = 0; r < nranks; r++) {
    if (results[r].skipped) {
      anySkipped = true;
    } else if (results[r].ok) {
      passed++;
    }
  }

  if (anySkipped) {
    printf("\n=== SKIPPED (no SDMA queues) ===\n");
    return 0;
  }
  printf("\n=== %d/%d PASSED ===\n", passed, nranks);
  return passed == nranks ? 0 : 1;
}
