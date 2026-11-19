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

// CCO C++ example 03 — ccoProxy put (host-proxy RDMA) + SignalSet / wait
//
// ccoProxy model: the kernel queues each op and a host proxy thread posts the
// RDMA work request. Rank 0's block b owns SPSC queue b and sends its chunk to
// rank 1 in ROUNDS pieces; each piece is followed, on the same queue, by a
// SignalSet of slot b to the round number (an RDMA write, so no NIC atomics).
// Rank 1's block b calls waitSignal(b, 1) once per round: each call consumes one.
//
// Both kernels use the timeout overloads (flush / waitSignal with timeoutCycles),
// which return 2 on a failed DevComm instead of trapping: a trap would end the
// process. The host then prints ccoDevCommGetError's report.
//
// The DevComm is proxy-only: gdaConnectionType NONE (no GDA QPs) and
// proxyConnectionType FULL (a proxy QP to every rank), so the 2 ranks may share
// a node or sit on separate nodes. Needs a build with the ccoProxy engine
// (BUILD_CCO_PROXY, gfx942 / gfx950).
//
//   mpirun -n 2 ./cco_proxy_put     (set MORI_SOCKET_IFNAME=<iface>)

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

#define HIP_CHECK(x)                                                      \
  do {                                                                    \
    hipError_t _e = (x);                                                  \
    if (_e != hipSuccess) {                                               \
      fprintf(stderr, "HIP error %s at %s\n", hipGetErrorString(_e), #x); \
      MPI_Abort(MPI_COMM_WORLD, 1);                                       \
    }                                                                     \
  } while (0)

#define CCO_CHECK(x)                                   \
  do {                                                 \
    int _r = (x);                                      \
    if (_r != 0) {                                     \
      fprintf(stderr, "cco error %d at %s\n", _r, #x); \
      MPI_Abort(MPI_COMM_WORLD, 1);                    \
    }                                                  \
  } while (0)

static constexpr int DST_RANK = 1;
static constexpr int BLOCKS = 4;  // one SPSC queue and one signal slot per block
static constexpr int ROUNDS = 4;
static constexpr size_t PIECE = 64 * 1024;  // bytes per (block, round)
static constexpr size_t CHUNK = ROUNDS * PIECE;
static constexpr size_t NBYTES = BLOCKS * CHUNK;
static constexpr size_t PER_RANK_VMM = 256ULL * 1024 * 1024;
// clock64 cycles; long enough for any proxy answer.
static constexpr uint64_t TIMEOUT_CYCLES = 40ULL * 1000 * 1000 * 1000;

#ifdef MORI_CCO_PROXY

// Rank 0: block b puts piece r of its chunk, then sets slot b to r + 1. The
// flush makes the sources reusable; status[b] = its result (0 done, 1 timed
// out, 2 DevComm failed).
__global__ void ProxyPutKernel(ccoDevComm devComm, ccoWindow_t sendWin, ccoWindow_t recvWin,
                               int* status) {
  const int b = blockIdx.x;
  ccoProxy<> proxy{devComm, b};
  for (int r = 0; r < ROUNDS; r++) {
    const size_t off = b * CHUNK + r * PIECE;
    proxy.put(DST_RANK, recvWin, off, sendWin, off, PIECE,
              ccoGda_SignalSet{static_cast<ccoGdaSignal_t>(b), static_cast<uint64_t>(r + 1)},
              ccoCoopBlock{});
  }
  const int rc = proxy.flush(ccoCoopBlock{}, TIMEOUT_CYCLES);
  if (threadIdx.x == 0) status[b] = rc;
}

// Rank 1: block b waits for each round's piece (consumes one per round).
// status[b] = the first nonzero result, else 0.
__global__ void ProxyWaitKernel(ccoDevComm devComm, int* status) {
  const int b = blockIdx.x;
  ccoProxy<> proxy{devComm, b};  // never pushes: only reads the signal slots
  int rc = 0;
  for (int r = 0; r < ROUNDS && rc == 0; r++) {
    rc = proxy.waitSignal(static_cast<ccoGdaSignal_t>(b), 1, ccoCoopBlock{}, TIMEOUT_CYCLES);
  }
  if (threadIdx.x == 0) status[b] = rc;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks != 2) {
    if (rank == 0) fprintf(stderr, "This example needs exactly 2 ranks.\n");
    MPI_Finalize();
    return 1;
  }

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  HIP_CHECK(hipSetDevice(rank % ndev));

  ccoUniqueId uid;
  if (rank == 0) CCO_CHECK(ccoGetUniqueId(&uid));
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

  ccoComm* comm = nullptr;
  CCO_CHECK(ccoCommCreate(uid, nranks, rank, PER_RANK_VMM, &comm));

  ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  void* sendLocal = nullptr;
  void* recvLocal = nullptr;
  CCO_CHECK(ccoWindowRegister(comm, NBYTES, &sendWin, &sendLocal));
  CCO_CHECK(ccoWindowRegister(comm, NBYTES, &recvWin, &recvLocal));

  const size_t count = NBYTES / sizeof(uint64_t);
  std::vector<uint64_t> host(count);
  if (rank == 0) {
    for (size_t i = 0; i < count; i++) host[i] = i + 1;
    HIP_CHECK(hipMemcpy(sendLocal, host.data(), NBYTES, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemset(recvLocal, 0, NBYTES));

  // Proxy-only DevComm. Every rank must pass the same proxy fields.
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;    // no GDA QPs
  reqs.proxyConnectionType = CCO_GDA_CONNECTION_FULL;  // proxy QPs to every rank
  reqs.proxySpscQueueCount = BLOCKS;
  reqs.gdaSignalCount = BLOCKS;  // the proxy's signal slots
  ccoDevComm devComm{};
  if (ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    // Fails on every rank alike: no ccoProxy on this GPU, MORI_CCO_DISABLE_PROXY=1,
    // no RDMA to the peer, or ranks passing different proxy fields.
    fprintf(stderr, "[rank %d] ccoDevCommCreate failed for a proxy-only DevComm\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  CCO_CHECK(ccoBarrierAll(comm));

  int* dStatus = nullptr;
  HIP_CHECK(hipMalloc(&dStatus, BLOCKS * sizeof(int)));
  HIP_CHECK(hipMemset(dStatus, 0xff, BLOCKS * sizeof(int)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  if (rank == 0) {
    ProxyPutKernel<<<BLOCKS, 256, 0, stream>>>(devComm, sendWin, recvWin, dStatus);
  } else {
    ProxyWaitKernel<<<BLOCKS, 64, 0, stream>>>(devComm, dStatus);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  int errors = 0;
  std::vector<int> status(BLOCKS);
  HIP_CHECK(hipMemcpy(status.data(), dStatus, BLOCKS * sizeof(int), hipMemcpyDeviceToHost));
  for (int b = 0; b < BLOCKS; b++) {
    if (status[b] != 0) {
      fprintf(stderr, "[rank %d] block %d: %s\n", rank, b,
              status[b] == 1 ? "timed out" : "DevComm failed");
      errors++;
    }
  }
  ccoDevCommError err{};
  if (ccoDevCommGetError(comm, &devComm, &err) == 1) {
    fprintf(stderr, "[rank %d] ccoProxy error: code %d peer %d queue %d op %d wc %d: %s\n", rank,
            err.code, err.peer, err.queue, err.op, err.wcStatus, err.message);
    errors++;
  }

  CCO_CHECK(ccoBarrierAll(comm));

  if (rank == DST_RANK) {
    HIP_CHECK(hipMemcpy(host.data(), recvLocal, NBYTES, hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < count; i++)
      if (host[i] != i + 1) bad++;
    errors += bad != 0;
    printf("[rank 1] proxy put %s — %zu of %zu words wrong, sample[0,-1]=%lu,%lu\n",
           bad ? "FAILED" : "verified", bad, count, host[0], host[count - 1]);
  }

  // A failed DevComm is destroyed the same way: collectively, on every rank.
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dStatus));
  CCO_CHECK(ccoDevCommDestroy(comm, &devComm));
  CCO_CHECK(ccoWindowDeregister(comm, sendWin));
  CCO_CHECK(ccoWindowDeregister(comm, recvWin));
  CCO_CHECK(ccoCommDestroy(comm));

  int total = 0;
  MPI_Reduce(&errors, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) printf("%s\n", total == 0 ? "SUCCESS" : "FAILED");
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}

#else  // !MORI_CCO_PROXY

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) fprintf(stderr, "This build has no ccoProxy engine (BUILD_CCO_PROXY).\n");
  MPI_Finalize();
  return 1;
}

#endif  // MORI_CCO_PROXY
