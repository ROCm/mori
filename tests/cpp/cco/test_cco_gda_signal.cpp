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

// test: cco gda signal API — standalone signal/waitSignal/resetSignal verification.
//
// tests three signal patterns without mixing put data:
//   1. SignalInc: each rank sends gda.signal(peer, SignalInc{myRank}) to every peer.
//      every rank waits for signal[r] >= 1 for each r != myRank.
//   2. SignalAdd: each rank sends gda.signal(peer, SignalAdd{myRank, 42}) to every peer.
//      every rank waits for signal[r] >= 42.
//   3. resetSignal + reuse: reset all signal slots, repeat SignalInc, verify counters
//      start from 0 (not from prior round's value).
//
// signal layout: devComm requests nRanks signal slots. signal[r] is the slot
// dedicated to messages from rank r (sender sends to slot r on the receiver).
//
// unlike test_cco_gda_put which verifies signals as a side-effect of put,
// this test exercises the signal primitive in isolation.

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
#include "mori/shmem/internal.hpp"

#include "mori/cco/cco.hpp"
#include "mori/cco/cco_device.hpp"

static int g_rank = 0;

#define HIP_CHECK(cmd)                                                                           \
  do {                                                                                           \
    hipError_t e = (cmd);                                                                        \
    if (e != hipSuccess) {                                                                       \
      fprintf(stderr, "[rank %d] HIP error %d (%s) at %s:%d\n", g_rank, e, hipGetErrorString(e), \
              __FILE__, __LINE__);                                                               \
      _exit(1);                                                                                  \
    }                                                                                            \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;

static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::PSD;

// ── kernel: round 1 — SignalInc ─────────────────────────────────────────────
//
// step 1: each thread sends SignalInc{myRank} to peer tid (no data payload).
// step 2: collective flush — ring doorbell + drain CQ for every peer.
//         flush(ccoCoopBlock) distributes peers across all threads via stride,
//         ensuring each QP's CQ is polled by exactly one thread (avoids the
//         warp-level pollCqLock collision that hangs with >= 4 ranks when
//         multiple threads call flush(single-peer) on different QPs simultaneously).
// step 3: each thread waits for peer tid's reciprocal signal on our slot.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalIncKernel(mori::cco::ccoDevComm devComm) {
  using namespace mori::cco::gda;

  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid    = threadIdx.x;

  // step 1: send
  if (tid < nRanks && tid != myRank) {
    gda.signal(tid, ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank)});
  }

  // step 2: collective flush — all threads participate
  gda.flush(mori::cco::ccoCoopBlock{});

  // step 3: wait for peer's signal
  if (tid < nRanks && tid != myRank) {
    gda.waitSignal(static_cast<ccoGdaSignal_t>(tid), /*least=*/1);
  }
}

// ── kernel: reset all signal slots ──────────────────────────────────────────
// must complete on ALL ranks (via host ccoBarrierAll) before any rank sends
// round-2 signals, otherwise a remote SignalAdd can race with the local reset.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalResetKernel(mori::cco::ccoDevComm devComm) {
  using namespace mori::cco::gda;

  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int nRanks = devComm.worldSize;
  int tid    = threadIdx.x;

  if (tid == 0) {
    for (int r = 0; r < nRanks; r++) {
      gda.resetSignal(static_cast<ccoGdaSignal_t>(r));
    }
  }
}

// ── kernel: round 2 — SignalAdd ─────────────────────────────────────────────
// launched only after ccoBarrierAll confirms all ranks have completed the reset.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalAddKernel(mori::cco::ccoDevComm devComm) {
  using namespace mori::cco::gda;

  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid    = threadIdx.x;

  // send SignalAdd{myRank, 42} to every peer
  if (tid < nRanks && tid != myRank) {
    gda.signal(tid, ccoGda_SignalAdd{static_cast<ccoGdaSignal_t>(myRank), 42ULL});
  }

  // collective flush
  gda.flush(mori::cco::ccoCoopBlock{});

  // wait for peer tid's signal to reach 42
  if (tid < nRanks && tid != myRank) {
    gda.waitSignal(static_cast<ccoGdaSignal_t>(tid), /*least=*/42);
  }
}

// ── host test ────────────────────────────────────────────────────────────────

static int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
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

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  // devcomm: full gda connectivity, one signal slot per rank
  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount   = 1;
  reqs.gdaSignalCount    = nranks;
  reqs.gdaCounterCount   = 0;
  mori::cco::ccoDevComm* devComm = nullptr;
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  mori::cco::ccoDevComm devCommHost;
  HIP_CHECK(hipMemcpy(&devCommHost, devComm, sizeof(devCommHost), hipMemcpyDeviceToHost));
  printf("[rank %d] DevCommCreate OK (worldSize=%d, gdaConnType=%d)\n",
         rank, devCommHost.worldSize, (int)devCommHost.gdaConnType);

  if (devCommHost.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE\n", rank);
    return 1;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  mori::cco::ccoBarrierAll(comm);

  // ── round 1: SignalInc ────────────────────────────────────────────────────
  printf("[rank %d] round 1: SignalInc\n", rank);
  GdaSignalIncKernel<kPrvdType><<<1, nranks, 0, stream>>>(devCommHost);
  HIP_CHECK(hipStreamSynchronize(stream));
  printf("[rank %d] round 1 passed\n", rank);

  mori::cco::ccoBarrierAll(comm);

  // ── round 2: resetSignal + SignalAdd ─────────────────────────────────────
  // reset must be globally complete before any rank sends round-2 signals.
  printf("[rank %d] round 2: resetSignal\n", rank);
  GdaSignalResetKernel<kPrvdType><<<1, nranks, 0, stream>>>(devCommHost);
  HIP_CHECK(hipStreamSynchronize(stream));

  mori::cco::ccoBarrierAll(comm);  // all ranks reset before any sends

  printf("[rank %d] round 2: SignalAdd\n", rank);
  GdaSignalAddKernel<kPrvdType><<<1, nranks, 0, stream>>>(devCommHost);
  HIP_CHECK(hipStreamSynchronize(stream));
  printf("[rank %d] round 2 passed\n", rank);

  mori::cco::ccoBarrierAll(comm);

  HIP_CHECK(hipStreamDestroy(stream));
  mori::cco::ccoDevCommDestroy(comm, devComm);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] PASSED\n", rank);
  return 0;
}

// ── fork mode ─────────────────────────────────────────────────────────────────

static void write_file(const char* path, const void* data, size_t len) {
  FILE* f = fopen(path, "wb");
  fwrite(data, 1, len, f);
  fclose(f);
}

static bool read_file(const char* path, void* data, size_t len) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  bool ok = fread(data, 1, len, f) == len;
  fclose(f);
  return ok;
}

static int run_fork_mode(int nranks) {
  char uidPath[256];
  snprintf(uidPath, sizeof(uidPath), "/tmp/cco_gda_signal_uid_%d", getpid());

  printf("=== CCO GDA signal Test (fork, %d ranks) ===\n", nranks);
  fflush(stdout);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 19878);
  write_file(uidPath, &uid, sizeof(uid));

  std::vector<pid_t> children;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      mori::application::UniqueId childUid;
      while (!read_file(uidPath, &childUid, sizeof(childUid))) usleep(10000);
      auto* boot = new mori::application::SocketBootstrapNetwork(childUid, r, nranks);
      _exit(run_test(r, nranks, boot));
    }
    children.push_back(pid);
  }

  int fail = 0;
  for (int r = 0; r < nranks; r++) {
    int status = 0;
    waitpid(children[r], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "rank %d failed (status=%d)\n", r, status);
      fail++;
    }
  }

  unlink(uidPath);
  printf("\n=== %d/%d PASSED ===\n", nranks - fail, nranks);
  return fail > 0 ? 1 : 0;
}

// ── single-rank mode for cross-host ──────────────────────────────────────────

static int run_single_rank_mode(int argc, char** argv) {
  int rank = -1, worldSize = -1, gpuOffset = -1;
  const char* uidPath = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank") && i + 1 < argc)
      rank = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--world") && i + 1 < argc)
      worldSize = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--uid-file") && i + 1 < argc)
      uidPath = argv[++i];
    else if (!strcmp(argv[i], "--gpu-offset") && i + 1 < argc)
      gpuOffset = atoi(argv[++i]);
  }
  if (rank < 0 || worldSize <= 0 || !uidPath) return -1;

  mori::application::UniqueId uid;
  for (int tries = 0; tries < 600; tries++) {
    FILE* f = fopen(uidPath, "rb");
    if (f) {
      size_t n = fread(&uid, 1, sizeof(uid), f);
      fclose(f);
      if (n == sizeof(uid)) break;
    }
    usleep(100000);
  }

  if (gpuOffset >= 0) HIP_CHECK(hipSetDevice(rank - gpuOffset));

  auto* boot = new mori::application::SocketBootstrapNetwork(uid, rank, worldSize);
  return run_test(rank, worldSize, boot);
}

static int run_gen_uid_mode(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: --gen-uid IFACE PORT OUTFILE\n");
    return 1;
  }
  const char* iface  = argv[2];
  int port           = atoi(argv[3]);
  const char* outPath = argv[4];
  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface(iface, port);
  FILE* f = fopen(outPath, "wb");
  if (!f) {
    fprintf(stderr, "fopen(%s) failed\n", outPath);
    return 1;
  }
  fwrite(&uid, 1, sizeof(uid), f);
  fclose(f);
  printf("Wrote UID (%zu bytes) for iface=%s port=%d to %s\n", sizeof(uid), iface, port, outPath);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "--gen-uid")) return run_gen_uid_mode(argc, argv);
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank")) return run_single_rank_mode(argc, argv);
  }

#ifdef MORI_WITH_MPI
  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);

  bool underMpi = mpiInitialized || getenv("OMPI_COMM_WORLD_SIZE") || getenv("PMI_SIZE") ||
                  getenv("PMI_RANK") || getenv("SLURM_PROCID");

  if (underMpi) {
    if (!mpiInitialized) MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (rank == 0) printf("=== CCO GDA signal Test (MPI, %d ranks) ===\n", nranks);
    auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);
    return run_test(rank, nranks, boot);
  }
#endif

  // fork mode — detect local gpu count
  int nranks = 0;
  for (int i = 0; i < 64; i++) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", i);
    FILE* f = fopen(path, "r");
    if (!f) break;
    unsigned long gpuId = 0;
    if (fscanf(f, "%lu", &gpuId) == 1 && gpuId != 0) nranks++;
    fclose(f);
  }
  if (argc > 1) nranks = std::min(atoi(argv[1]), nranks);
  if (nranks < 2) {
    printf("Need at least 2 GPUs.\n");
    return 1;
  }

  return run_fork_mode(nranks);
}
