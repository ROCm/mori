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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

// CCO C++ example 01 — LSA put (direct peer pointer via flat VA)
//
// LSA model: cco hands the kernel the peer's load/store-accessible VA via
// ccoGetLsaPeerPtr(); the kernel writes it *directly* — cco does NOT move the
// data. No GDA / RDMA / devComm needed: ccoCommCreate + ccoWindowRegister set up
// the symmetric flat VA, and the window device handle already carries
// winBase / stride4G / lsaRank.
//
// Rank 0 copies its send window into rank 1's recv window slot (2 ranks).
// With fabric cross-node LSA enabled, ranks may span multiple nodes.
//
//   mpirun -n 2 ./cco_lsa_put        (set MORI_SOCKET_IFNAME=<iface>)

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/cco/cco.hpp"

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

static constexpr size_t COUNT = 1024;
static constexpr size_t NBYTES = COUNT * sizeof(uint64_t);
static constexpr size_t PER_RANK_VMM = 256ULL * 1024 * 1024;

// One block stores src -> peer's recv slot via the peer's LSA pointer.
__global__ void LsaPutKernel(ccoWindow_t sendWin, ccoWindow_t recvWin, int peerLsaRank,
                             size_t count) {
  const uint64_t* src = static_cast<const uint64_t*>(ccoGetLocalPtr(sendWin));
  uint64_t* dst = static_cast<uint64_t*>(ccoGetLsaPeerPtr(recvWin, peerLsaRank));
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) dst[i] = src[i];
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

  // Bootstrap: rank 0 makes the UniqueId, broadcast it, everyone creates the comm.
  ccoUniqueId uid;
  if (rank == 0) CCO_CHECK(ccoGetUniqueId(&uid));
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

  ccoComm* comm = nullptr;
  CCO_CHECK(ccoCommCreate(uid, nranks, rank, PER_RANK_VMM, &comm));

  // Symmetric windows (cco allocs + registers; localPtr is for host memcpy).
  ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  void* sendLocal = nullptr;
  void* recvLocal = nullptr;
  CCO_CHECK(ccoWindowRegister(comm, NBYTES, &sendWin, &sendLocal));
  CCO_CHECK(ccoWindowRegister(comm, NBYTES, &recvWin, &recvLocal));

  std::vector<uint64_t> host(COUNT);
  if (rank == 0) {
    for (size_t i = 0; i < COUNT; i++) host[i] = i + 1;
    HIP_CHECK(hipMemcpy(sendLocal, host.data(), NBYTES, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemset(recvLocal, 0, NBYTES));

  CCO_CHECK(ccoBarrierAll(comm));

  // Rank 0 writes into rank 1's window slot (peer lsaRank = 1).
  if (rank == 0) {
    LsaPutKernel<<<1, 256>>>(sendWin, recvWin, /*peerLsaRank=*/1, COUNT);
    HIP_CHECK(hipDeviceSynchronize());
  }

  CCO_CHECK(ccoBarrierAll(comm));

  int errors = 0;
  if (rank == 1) {
    HIP_CHECK(hipMemcpy(host.data(), recvLocal, NBYTES, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < COUNT; i++)
      if (host[i] != i + 1) errors++;
    printf("[rank 1] LSA put %s — sample[0,1,-1]=%lu,%lu,%lu\n", errors ? "FAILED" : "verified",
           host[0], host[1], host[COUNT - 1]);
  }

  CCO_CHECK(ccoWindowDeregister(comm, sendWin));
  CCO_CHECK(ccoWindowDeregister(comm, recvWin));
  CCO_CHECK(ccoCommDestroy(comm));

  int total = 0;
  MPI_Reduce(&errors, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) printf("%s\n", total == 0 ? "SUCCESS" : "FAILED");
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
