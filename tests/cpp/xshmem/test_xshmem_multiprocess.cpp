// Test: XSHMEM host API — multi-process, one GPU per rank.
//
// Two modes, auto-detected:
//   mpirun -np 8 ./test_xshmem_multiprocess          (MPI bootstrap)
//   ./test_xshmem_multiprocess [nranks]               (fork, socket bootstrap)

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
#include "mori/xshmem/xshmem_api.hpp"

static int g_rank = 0;

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "[rank %d] HIP error %d (%s) at %s:%d\n", g_rank, e,    \
              hipGetErrorString(e), __FILE__, __LINE__);                       \
      _exit(1);                                                                \
    }                                                                          \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t WINDOW_SIZE = 4096;

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

  mori::xshmem::XshmemComm* comm = nullptr;
  if (mori::xshmem::XshmemCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }
  printf("[rank %d] CommCreate OK\n", rank);

  void* buf = nullptr;
  if (mori::xshmem::XshmemMemAlloc(comm, WINDOW_SIZE, &buf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  uint8_t pattern = static_cast<uint8_t>((rank + 1) * 10);
  HIP_CHECK(hipMemset(buf, pattern, WINDOW_SIZE));

  mori::xshmem::XshmemWindow_t win = nullptr;
  if (mori::xshmem::XshmemWindowRegister(comm, buf, WINDOW_SIZE, &win) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  mori::xshmem::XshmemDevComm* devComm = nullptr;
  if (mori::xshmem::XshmemDevCommCreate(comm, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  printf("[rank %d] init OK\n", rank);

  // Verify DevComm
  mori::xshmem::XshmemDevComm dcHost;
  HIP_CHECK(hipMemcpy(&dcHost, devComm, sizeof(dcHost), hipMemcpyDeviceToHost));
  if (dcHost.rank != rank || dcHost.worldSize != nranks) {
    fprintf(stderr, "[rank %d] DevComm mismatch\n", rank);
    return 1;
  }

  // P2P cross-read via flat addressing
  mori::xshmem::XshmemWindowDevice winHost;
  HIP_CHECK(hipMemcpy(&winHost, win, sizeof(winHost), hipMemcpyDeviceToHost));

  mori::xshmem::XshmemBarrierAll(comm);

  int p2pOk = 0;
  for (int pe = 0; pe < nranks; pe++) {
    if (pe == rank) continue;
    void* peerVa = winHost.winBase + (static_cast<uint64_t>(pe) * winHost.stride4G << 32);
    uint8_t got = 0;
    HIP_CHECK(hipMemcpy(&got, peerVa, 1, hipMemcpyDeviceToHost));
    uint8_t want = static_cast<uint8_t>((pe + 1) * 10);
    if (got != want) {
      fprintf(stderr, "[rank %d] P2P read PE %d: got %u want %u\n", rank, pe, got, want);
      return 1;
    }
    p2pOk++;
  }
  printf("[rank %d] P2P OK from %d peers\n", rank, p2pOk);

  mori::xshmem::XshmemDevCommDestroy(devComm);
  mori::xshmem::XshmemWindowDeregister(comm, win);
  mori::xshmem::XshmemMemFree(comm, buf);
  mori::xshmem::XshmemCommDestroy(comm);

  printf("[rank %d] PASSED\n", rank);
  return 0;
}

// ── Socket bootstrap: fork mode ──

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
  snprintf(uidPath, sizeof(uidPath), "/tmp/xshmem_test_uid_%d", getpid());

  printf("=== XSHMEM Multi-Process Test (fork, %d ranks) ===\n", nranks);
  fflush(stdout);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 19876);
  write_file(uidPath, &uid, sizeof(uid));

  std::vector<pid_t> children;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      // Child: read uid, run test
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

// ── Main: auto-detect MPI or fork ──

int main(int argc, char** argv) {
#ifdef MORI_WITH_MPI
  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);

  // Check if launched under mpirun (PMI/OMPI env vars present)
  bool underMpi = mpiInitialized || getenv("OMPI_COMM_WORLD_SIZE") || getenv("PMI_SIZE") ||
                  getenv("PMI_RANK") || getenv("SLURM_PROCID");

  if (underMpi) {
    if (!mpiInitialized) MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (rank == 0) printf("=== XSHMEM Multi-Process Test (MPI, %d ranks) ===\n", nranks);

    auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);
    int ret = run_test(rank, nranks, boot);
    // MPI_Finalize is called by MpiBootstrapNetwork::Finalize() inside CommDestroy
    return ret;
  }
#endif

  // Fork mode: detect GPU count without initializing HIP (avoids fork+HIP corruption)
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
  if (nranks < 1) { printf("No GPUs found.\n"); return 1; }

  return run_fork_mode(nranks);
}
