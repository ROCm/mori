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
// test: cco gda get — gpu-initiated rdma read (one-sided pull).
//
// each rank stages sendBuf[r*COUNT + i] = rank*1000 + r*100 + i, then each
// rank launches one kernel that for every peer p != myRank does:
//   1. get(peer=p, peer's sendWin offset=myRank*perPairBytes,
//          local recvWin offset=p*perPairBytes, perPairBytes)
//      — reads the slice peer p staged for me, into my recv slot for peer p
//   2. flush() — poll cq so recvBuf is reusable / readable on host
//
// no signal: get is fully one-sided. ccoBarrierAll BEFORE the kernel guarantees
// every peer's sendBuf is initialized before anyone reads. ccoBarrierAll AFTER
// the kernel is hygiene so cleanup happens in lock-step.
//
// host verification: recv[peer*COUNT + i] should equal peer*1000 + myRank*100 + i
// — the inverse of the put test's check.

#ifdef MORI_WITH_MPI
#include <mpi.h>

#include "mori/application/bootstrap/mpi_bootstrap.hpp"
#endif

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/cco/cco_scale_out.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;  // elements per rank-pair

// Dispatch a kernel launch to the ccoGda<PrvdType> instantiation matching the
// DevComm's RDMA backend (devComm.ibgda.providerType), resolved at runtime. `P`
// is a constexpr ProviderType usable as a template argument in the launch expr.
#define CCO_GDA_DISPATCH(prvd, ...)                                                              \
  do {                                                                                           \
    switch (prvd) {                                                                              \
      case mori::core::ProviderType::BNXT: {                                                     \
        constexpr auto P = mori::core::ProviderType::BNXT;                                       \
        __VA_ARGS__;                                                                             \
      } break;                                                                                   \
      case mori::core::ProviderType::MLX5: {                                                     \
        constexpr auto P = mori::core::ProviderType::MLX5;                                       \
        __VA_ARGS__;                                                                             \
      } break;                                                                                   \
      case mori::core::ProviderType::PSD: {                                                      \
        constexpr auto P = mori::core::ProviderType::PSD;                                        \
        __VA_ARGS__;                                                                             \
      } break;                                                                                   \
      default:                                                                                   \
        fprintf(stderr, "[cco gda test] unsupported GDA provider %d\n", static_cast<int>(prvd)); \
        _exit(1);                                                                                \
    }                                                                                            \
  } while (0)

// alltoall-via-get kernel — single warp does the work.
// each thread pulls one peer's destined slice into our recvBuf.
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaAlltoAllGetKernel(mori::cco::ccoWindowDevice* sendWin,
                                     mori::cco::ccoWindowDevice* recvWin, size_t count,
                                     mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;

  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  size_t perPairBytes = count * sizeof(T);

  // step 1: each thread issues a get from a distinct peer.
  // we read peer's sendBuf slot indexed by myRank (the data peer staged for us),
  // and land it in our recvBuf slot indexed by peer.
  for (int r = tid; r < nRanks; r += nthreads) {
    if (r == myRank) continue;
    gda.get(r, reinterpret_cast<ccoWindow_t>(sendWin), myRank * perPairBytes,
            reinterpret_cast<ccoWindow_t>(recvWin), r * perPairBytes, perPairBytes);
  }

  // step 2: flush — ring doorbell + poll CQ, ensures rdma reads have landed.
  gda.flush(mori::cco::ccoCoopBlock{});
}

int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
  g_rank = rank;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  for (int i = 0; i < numDevices; i++) {
    if (i == dev) continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dev, i));
    if (canAccess) (void)hipDeviceEnablePeerAccess(i, 0);
  }

  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  // setup: comm, send/recv buffers, windows
  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
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

  // sendBuf[r*COUNT + i] = rank*1000 + r*100 + i — encodes (owner, dest-slot, idx)
  std::vector<float> hostSend(COUNT * nranks);
  for (int r = 0; r < nranks; r++) {
    for (size_t i = 0; i < COUNT; i++) {
      hostSend[r * COUNT + i] = static_cast<float>(rank * 1000 + r * 100 + i);
    }
  }
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));

  mori::cco::ccoWindow_t sendWin = nullptr;
  mori::cco::ccoWindow_t recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // devcomm: full gda connectivity. get is one-sided so no signals needed.
  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  printf("[rank %d] DevCommCreate OK (worldSize=%d, lsaSize=%d, gdaConnType=%d, numQpPerPe=%d)\n",
         rank, devComm.worldSize, devComm.lsaSize, (int)devComm.gdaConnType,
         devComm.ibgda.numQpPerPe);

  if (devComm.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE — check peer mask / rdma support\n",
            rank);
    return 1;
  }

  // critical: peers' sendBuf must be initialized before any get is issued.
  mori::cco::ccoBarrierAll(comm);

  // launch
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  CCO_GDA_DISPATCH(devComm.ibgda.providerType, GdaAlltoAllGetKernel<P, float>
                   <<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, devComm));
  HIP_CHECK(hipStreamSynchronize(stream));
  printf("[rank %d] kernel completed\n", rank);

  mori::cco::ccoBarrierAll(comm);

  // verify: recv[peer*COUNT + i] should be peer*1000 + myRank*100 + i.
  // peer staged sendBuf[myRank*COUNT + i] = peer*1000 + myRank*100 + i, which
  // is exactly what we just pulled into our recv[peer*COUNT + i].
  std::vector<float> hostRecv(COUNT * nranks);
  HIP_CHECK(hipMemcpy(hostRecv.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));

  bool ok = true;
  for (int peer = 0; peer < nranks && ok; peer++) {
    if (peer == rank) continue;
    for (size_t i = 0; i < COUNT; i++) {
      float expected = static_cast<float>(peer * 1000 + rank * 100 + i);
      float got = hostRecv[peer * COUNT + i];
      if (got != expected) {
        fprintf(stderr, "[rank %d] mismatch at [peer=%d][%zu]: got %.0f expected %.0f\n", rank,
                peer, i, got, expected);
        ok = false;
        break;
      }
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA get", "/tmp/cco_gda_get_uid", 19879);
}
