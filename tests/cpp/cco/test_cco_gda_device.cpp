// Test: CCO GDA device API — AlltoAll via GPU-initiated RDMA put + signal.
//
// Multi-process (fork or MPI). Each rank launches a HIP kernel that:
//   1. put(peer, ..., SignalInc{signalId}) to every peer
//   2. waitSignal to confirm all peers have written
//   3. flush to ensure source buffers are reusable
//
// Host side verifies data correctness after kernel completion.
// No BarrierSession — uses host ccoBarrierAll for pre/post sync.

#ifdef MORI_WITH_MPI
#include <mpi.h>

#include "mori/application/bootstrap/mpi_bootstrap.hpp"
#endif

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/cco/cco_api.hpp"
#include "mori/cco/gda/gda_device_api.hpp"

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
static const size_t COUNT = 256;  // elements per rank-pair

// NIC provider type selection (compile-time, same as shmem kernels)
#ifdef MORI_DEVICE_NIC_BNXT
static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::BNXT;
#elif defined(MORI_DEVICE_NIC_IONIC)
static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::PSD;
#else
static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::MLX5;
#endif

// AlltoAll kernel: each rank puts its data to every peer's recv buffer.
// Single warp (threadIdx.x < 64) does the work.
// Signal layout: one signal per rank — signal[r] is incremented by peer r.
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaAlltoAllKernel(mori::cco::ccoWindowDevice* sendWin,
                                  mori::cco::ccoWindowDevice* recvWin, size_t count,
                                  mori::cco::ccoDevComm devComm) {
  using namespace mori::cco::gda;

  int ginContext = 0;
  ccoGda<PrvdType> gda{devComm, ginContext};

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  size_t perPairBytes = count * sizeof(T);

  // Each thread handles a subset of peers
  for (int r = tid; r < nRanks; r += nthreads) {
    if (r == myRank) continue;

    // put my data for peer r into peer r's recv buffer at slot [myRank]
    gda.put(r, reinterpret_cast<ccoWindow_t>(recvWin),  // dst: peer's recv window
            myRank * perPairBytes,                      // dst offset: slot for myRank
            reinterpret_cast<ccoWindow_t>(sendWin),     // src: my send window
            r * perPairBytes,                           // src offset: data destined for peer r
            perPairBytes, ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank)});
  }

  // Flush all peers — ring doorbells and wait for CQ completion
  if (tid == 0) {
    gda.flush();
  }

  // Wait for all peers to have written to us (each peer increments signal[peerRank])
  // Only one thread does the waiting to avoid redundant polls
  if (tid == 0) {
    for (int r = 0; r < nRanks; r++) {
      if (r == myRank) continue;
      gda.waitSignal(static_cast<ccoGdaSignal_t>(r), 1);
    }
  }
}

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
    if (canAccess) hipDeviceEnablePeerAccess(i, 0);
  }

  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  // Phase 1: CommCreate
  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  // Phase 2: Allocate send and recv buffers
  size_t bufSize = COUNT * nranks * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  // Initialize send data: sendBuf[r*COUNT + i] = rank*1000 + r*100 + i
  std::vector<float> hostSend(COUNT * nranks);
  for (int r = 0; r < nranks; r++) {
    for (size_t i = 0; i < COUNT; i++) {
      hostSend[r * COUNT + i] = static_cast<float>(rank * 1000 + r * 100 + i);
    }
  }
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(recvBuf, 0, bufSize));

  // Phase 3: Register windows
  mori::cco::ccoWindow_t sendWin = nullptr;
  mori::cco::ccoWindow_t recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // Phase 4: DevCommCreate with FULL GDA connectivity + signals
  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;  // one signal per peer
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm* devComm = nullptr;
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  // Copy DevComm to host for kernel launch (passed by value like NCCL)
  mori::cco::ccoDevComm devCommHost;
  HIP_CHECK(hipMemcpy(&devCommHost, devComm, sizeof(devCommHost), hipMemcpyDeviceToHost));
  printf("[rank %d] DevCommCreate OK (worldSize=%d, lsaSize=%d)\n", rank, devCommHost.worldSize,
         devCommHost.lsaSize);

  // Verify IBGDA endpoints are allocated
  if (devCommHost.ibgda.endpoints == nullptr) {
    fprintf(stderr, "[rank %d] IBGDA endpoints are null — GDA not initialized\n", rank);
    return 1;
  }

  // Host barrier to ensure all ranks have initialized
  mori::cco::ccoBarrierAll(comm);

  // Phase 5: Launch AlltoAll kernel
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  GdaAlltoAllKernel<kPrvdType, float><<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, devCommHost);

  HIP_CHECK(hipStreamSynchronize(stream));
  printf("[rank %d] Kernel completed\n", rank);

  // Host barrier after kernel to ensure all ranks finished
  mori::cco::ccoBarrierAll(comm);

  // Phase 6: Verify recv buffer
  std::vector<float> hostRecv(COUNT * nranks);
  HIP_CHECK(hipMemcpy(hostRecv.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));

  bool ok = true;
  for (int srcRank = 0; srcRank < nranks; srcRank++) {
    if (srcRank == rank) continue;
    for (size_t i = 0; i < COUNT; i++) {
      size_t idx = srcRank * COUNT + i;
      // srcRank sent: srcRank*1000 + rank*100 + i  (data destined for us)
      float expected = static_cast<float>(srcRank * 1000 + rank * 100 + i);
      if (hostRecv[idx] != expected) {
        fprintf(stderr, "[rank %d] Mismatch at [src=%d][%zu]: got %.0f expected %.0f\n", rank,
                srcRank, i, hostRecv[idx], expected);
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }

  if (ok) {
    printf("[rank %d] Data verification PASSED\n", rank);
  }

  // Cleanup
  HIP_CHECK(hipStreamDestroy(stream));
  mori::cco::ccoDevCommDestroy(comm, devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

// ── Fork mode ──

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
  snprintf(uidPath, sizeof(uidPath), "/tmp/cco_gda_device_uid_%d", getpid());

  printf("=== CCO GDA Device API Test (fork, %d ranks) ===\n", nranks);
  fflush(stdout);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 19877);
  write_file(uidPath, &uid, sizeof(uid));

  std::vector<pid_t> children;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      mori::application::UniqueId childUid;
      while (!read_file(uidPath, &childUid, sizeof(childUid))) {
        usleep(10000);
      }
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

// ── Single-rank mode for cross-host ──

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
  const char* iface = argv[2];
  int port = atoi(argv[3]);
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
    if (rank == 0) printf("=== CCO GDA Device API Test (MPI, %d ranks) ===\n", nranks);

    auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);
    return run_test(rank, nranks, boot);
  }
#endif

  // Fork mode
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
