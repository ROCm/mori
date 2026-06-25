// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

// CCO C++ example 02 — GDA put (GPU-initiated RDMA) + signal/wait
//
// GDA model: opaque network ops. Rank 0's kernel issues one GDA put into rank
// 1's recv window with a completion signal; rank 1's kernel waits on the signal,
// then the host verifies the payload. Needs a devComm with a GDA connection
// (FULL for a single node; CROSSNODE for two physical nodes).
//
// Only cco_scale_out.hpp is included (it pulls in cco.hpp). The provider is
// selected at launch by CCO_GDA_DISPATCH from the build-time NIC macro, so build
// with -DMORI_DEVICE_NIC_* (the CMake target adds it).
//
//   mpirun -n 2 ./cco_gda_put     (single node: MORI_CCO_GDA_CONN unused here —
//                                  we set CCO_GDA_CONNECTION_FULL directly;
//                                  set MORI_SOCKET_IFNAME=<iface>)

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

#define HIP_CHECK(x)                                                       \
  do {                                                                     \
    hipError_t _e = (x);                                                   \
    if (_e != hipSuccess) {                                                \
      fprintf(stderr, "HIP error %s at %s\n", hipGetErrorString(_e), #x);  \
      MPI_Abort(MPI_COMM_WORLD, 1);                                        \
    }                                                                      \
  } while (0)

#define CCO_CHECK(x)                                          \
  do {                                                        \
    int _r = (x);                                             \
    if (_r != 0) {                                            \
      fprintf(stderr, "cco error %d at %s\n", _r, #x);        \
      MPI_Abort(MPI_COMM_WORLD, 1);                           \
    }                                                         \
  } while (0)

static constexpr int DST_RANK = 1;
static constexpr ccoGdaSignal_t SIG = 0;
static constexpr size_t COUNT = 1024;
static constexpr size_t NBYTES = COUNT * sizeof(uint64_t);
static constexpr size_t PER_RANK_VMM = 256ULL * 1024 * 1024;

// Rank 0: put send -> rank 1's recv, bump signal SIG; then drain the local CQ.
template <mori::core::ProviderType PrvdType>
__global__ void GdaPutKernel(ccoWindow_t sendWin, ccoWindow_t recvWin, size_t bytes,
                             ccoDevComm devComm) {
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0) {
    gda.put(DST_RANK, recvWin, 0, sendWin, 0, bytes, ccoGda_SignalInc{SIG});
  }
  gda.flush(ccoCoopBlock{});
}

// Rank 1: wait until the signal lands (proves the remote write completed).
template <mori::core::ProviderType PrvdType>
__global__ void GdaWaitKernel(ccoDevComm devComm) {
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0) gda.waitSignal(SIG, 1);
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

  std::vector<uint64_t> host(COUNT);
  if (rank == 0) {
    for (size_t i = 0; i < COUNT; i++) host[i] = i + 1;
    HIP_CHECK(hipMemcpy(sendLocal, host.data(), NBYTES, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemset(recvLocal, 0, NBYTES));

  // devComm with a GDA connection (FULL = single node; CROSSNODE for 2 nodes).
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = 4;
  reqs.gdaCounterCount = 0;
  ccoDevComm devComm{};
  CCO_CHECK(ccoDevCommCreate(comm, &reqs, &devComm));
  if (devComm.gdaConnType == CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] GDA connection collapsed to NONE — check RDMA support\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  CCO_CHECK(ccoBarrierAll(comm));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  if (rank == 0) {
    CCO_GDA_DISPATCH(GdaPutKernel<P><<<1, 64, 0, stream>>>(sendWin, recvWin, NBYTES, devComm));
  } else {
    CCO_GDA_DISPATCH(GdaWaitKernel<P><<<1, 64, 0, stream>>>(devComm));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  CCO_CHECK(ccoBarrierAll(comm));

  int errors = 0;
  if (rank == DST_RANK) {
    HIP_CHECK(hipMemcpy(host.data(), recvLocal, NBYTES, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < COUNT; i++)
      if (host[i] != i + 1) errors++;
    printf("[rank 1] GDA put %s — sample[0,1,-1]=%lu,%lu,%lu\n", errors ? "FAILED" : "verified",
           host[0], host[1], host[COUNT - 1]);
  }

  HIP_CHECK(hipStreamDestroy(stream));
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
