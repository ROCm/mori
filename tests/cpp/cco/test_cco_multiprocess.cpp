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
// Test: CCO host API — multi-process, one GPU per rank.
//
// Two modes, auto-detected:
//   mpirun -np 8 ./test_cco_multiprocess          (MPI bootstrap)
//   ./test_cco_multiprocess [nranks]               (fork, socket bootstrap)

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
#include "mori/cco/cco.hpp"

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
static const size_t WINDOW_SIZE = 4096 * 1024;

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

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }
  printf("[rank %d] CommCreate OK\n", rank);

  void* buf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, WINDOW_SIZE, &buf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  uint8_t pattern = static_cast<uint8_t>((rank + 1) * 10);
  HIP_CHECK(hipMemset(buf, pattern, WINDOW_SIZE));

  mori::cco::ccoWindow_t win = nullptr;
  if (mori::cco::ccoWindowRegister(comm, buf, WINDOW_SIZE, &win) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // ── Create DevComm #1 (default requirements) ──
  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  mori::cco::ccoDevComm* devComm1 = nullptr;
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm1) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate #1 failed\n", rank);
    return 1;
  }

  // ── Create DevComm #2 (fresh QPs, independent from #1) ──
  mori::cco::ccoDevComm* devComm2 = nullptr;
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm2) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate #2 failed\n", rank);
    return 1;
  }
  printf("[rank %d] 2x DevCommCreate OK\n", rank);

  // Verify both DevComms have correct rank/worldSize
  mori::cco::ccoDevComm dc1Host, dc2Host;
  HIP_CHECK(hipMemcpy(&dc1Host, devComm1, sizeof(dc1Host), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&dc2Host, devComm2, sizeof(dc2Host), hipMemcpyDeviceToHost));
  if (dc1Host.rank != rank || dc2Host.rank != rank) {
    fprintf(stderr, "[rank %d] DevComm rank mismatch\n", rank);
    return 1;
  }

  // Verify DevComm #1 and #2 have different QP resources (different GPU endpoint pointers)
  if (dc1Host.ibgda.endpoints == dc2Host.ibgda.endpoints && dc1Host.ibgda.endpoints != nullptr) {
    fprintf(stderr, "[rank %d] DevComm #1 and #2 share same endpoint pointer — NOT independent!\n",
            rank);
    return 1;
  }

  // Verify signal buffers are also independent
  if (dc1Host.ibgda.signalBuf == dc2Host.ibgda.signalBuf && dc1Host.ibgda.signalBuf != nullptr) {
    fprintf(stderr, "[rank %d] DevComm #1 and #2 share same signalBuf!\n", rank);
    return 1;
  }
  printf("[rank %d] DevComm independence verified\n", rank);

  // ── GDA connection mode verification (NONE / FULL / CROSSNODE / RAIL) ──
  // Re-issues DevCommCreate with each connType and counts the QPs that ended
  // up non-zero in ibgda.endpoints. Expected on a uniform N-nodes × lsaSize
  // layout:
  //   NONE      : 0
  //   FULL      : (worldSize - 1) * qpsPerPe
  //   CROSSNODE : (worldSize - lsaSize) * qpsPerPe   (collapses to 0 on single-node)
  //   RAIL      : (nNodes - 1) * qpsPerPe            (collapses to 0 on single-node)
  {
    auto countQps = [&](mori::cco::ccoDevComm* dc) -> int {
      mori::cco::ccoDevComm h;
      HIP_CHECK(hipMemcpy(&h, dc, sizeof(h), hipMemcpyDeviceToHost));
      if (!h.ibgda.endpoints || h.ibgda.numQpPerPe == 0) return 0;
      size_t n = static_cast<size_t>(h.worldSize) * h.ibgda.numQpPerPe;
      std::vector<mori::application::RdmaEndpointDevice> eps(n);
      HIP_CHECK(
          hipMemcpy(eps.data(), h.ibgda.endpoints, n * sizeof(eps[0]), hipMemcpyDeviceToHost));
      int c = 0;
      for (auto& ep : eps)
        if (ep.qpn != 0) c++;
      return c;
    };
    auto mkReqs = [](mori::cco::ccoGdaConnectionType ct) {
      mori::cco::ccoDevCommRequirements r = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
      r.gdaConnectionType = ct;
      return r;
    };
    mori::cco::ccoDevComm *dcNone = nullptr, *dcFull = nullptr;
    mori::cco::ccoDevComm *dcXnode = nullptr, *dcRail = nullptr;
    auto rNone = mkReqs(mori::cco::CCO_GDA_CONNECTION_NONE);
    auto rFull = mkReqs(mori::cco::CCO_GDA_CONNECTION_FULL);
    auto rXnode = mkReqs(mori::cco::CCO_GDA_CONNECTION_CROSSNODE);
    auto rRail = mkReqs(mori::cco::CCO_GDA_CONNECTION_RAIL);
    const int qpsPerPe = rFull.gdaContextCount;
    if (mori::cco::ccoDevCommCreate(comm, &rNone, &dcNone) != 0 ||
        mori::cco::ccoDevCommCreate(comm, &rFull, &dcFull) != 0 ||
        mori::cco::ccoDevCommCreate(comm, &rXnode, &dcXnode) != 0 ||
        mori::cco::ccoDevCommCreate(comm, &rRail, &dcRail) != 0) {
      fprintf(stderr, "[rank %d] connType DevCommCreate failed\n", rank);
      return 1;
    }
    const int nNodes = dc1Host.worldSize / dc1Host.lsaSize;
    const int qNone = countQps(dcNone);
    const int qFull = countQps(dcFull);
    const int qXnode = countQps(dcXnode);
    const int qRail = countQps(dcRail);
    const int eFull = (dc1Host.worldSize - 1) * qpsPerPe;
    const int eXnode = (dc1Host.worldSize - dc1Host.lsaSize) * qpsPerPe;
    const int eRail = (nNodes - 1) * qpsPerPe;
    if (qNone != 0 || qFull != eFull || qXnode != eXnode || qRail != eRail) {
      fprintf(stderr,
              "[rank %d] connType MISMATCH: NONE got=%d exp=0, FULL got=%d exp=%d, "
              "XNODE got=%d exp=%d, RAIL got=%d exp=%d (nNodes=%d qpsPerPe=%d)\n",
              rank, qNone, qFull, eFull, qXnode, eXnode, qRail, eRail, nNodes, qpsPerPe);
      return 1;
    }
    printf("[rank %d] connType OK: NONE=0 FULL=%d XNODE=%d RAIL=%d (nNodes=%d qpsPerPe=%d)\n", rank,
           qFull, qXnode, qRail, nNodes, qpsPerPe);
    mori::cco::ccoDevCommDestroy(comm, dcRail);
    mori::cco::ccoDevCommDestroy(comm, dcXnode);
    mori::cco::ccoDevCommDestroy(comm, dcFull);
    mori::cco::ccoDevCommDestroy(comm, dcNone);
  }

  // P2P cross-read via flat addressing — LSA peers only (intra-node).
  mori::cco::ccoWindowDevice winHost;
  HIP_CHECK(hipMemcpy(&winHost, win, sizeof(winHost), hipMemcpyDeviceToHost));
  mori::cco::ccoDevComm devCommSnap;
  HIP_CHECK(hipMemcpy(&devCommSnap, devComm1, sizeof(devCommSnap), hipMemcpyDeviceToHost));

  mori::cco::ccoBarrierAll(comm);

  int p2pOk = 0;
  int lsaSize = devCommSnap.lsaSize;
  int myNodeStart = devCommSnap.myNodeStart;
  for (int lsa = 0; lsa < lsaSize; lsa++) {
    if (lsa == winHost.lsaRank) continue;
    int pe = myNodeStart + lsa;
    void* peerVa = winHost.winBase + (static_cast<uint64_t>(lsa) * winHost.stride4G << 32);
    uint8_t got = 0;
    HIP_CHECK(hipMemcpy(&got, peerVa, 1, hipMemcpyDeviceToHost));
    uint8_t want = static_cast<uint8_t>((pe + 1) * 10);
    if (got != want) {
      fprintf(stderr, "[rank %d] P2P read PE %d (lsa=%d): got %u want %u\n", rank, pe, lsa, got,
              want);
      return 1;
    }
    p2pOk++;
  }
  printf("[rank %d] P2P OK from %d LSA peers\n", rank, p2pOk);

  mori::cco::ccoDevCommDestroy(comm, devComm2);
  mori::cco::ccoDevCommDestroy(comm, devComm1);
  mori::cco::ccoWindowDeregister(comm, win);
  mori::cco::ccoMemFree(comm, buf);
  mori::cco::ccoCommDestroy(comm);

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
  snprintf(uidPath, sizeof(uidPath), "/tmp/cco_test_uid_%d", getpid());

  printf("=== CCO Multi-Process Test (fork, %d ranks) ===\n", nranks);
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

// ── Single-rank mode: --rank R --world N --uid-file F [--gpu-offset G] ──
// Used for cross-host launches where mpirun/MPI bootstrap isn't available;
// an outside coordinator writes the UID file and spawns one process per rank.
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
    usleep(100000);  // wait up to 60s for the launcher to write the UID
  }

  // Pin this process to a GPU. If --gpu-offset is given, use rank - offset
  // (so two-host launches can map rank 0..7 -> GPU 0..7 on node A and rank
  // 8..15 -> GPU 0..7 on node B). Otherwise fall back to rank % numDevices.
  if (gpuOffset >= 0) HIP_CHECK(hipSetDevice(rank - gpuOffset));

  auto* boot = new mori::application::SocketBootstrapNetwork(uid, rank, worldSize);
  return run_test(rank, worldSize, boot);
}

// ── --gen-uid IFACE PORT OUTFILE ──
// Generate a SocketBootstrap UniqueId bound to the given interface/port and
// write it to OUTFILE. The cross-host launcher uses this on one node, then
// distributes the file to the other node.
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

// ── Main: auto-detect MPI / single-rank / gen-uid / fork ──

int main(int argc, char** argv) {
  // Special CLI modes (cross-host launcher uses these)
  if (argc >= 2 && !strcmp(argv[1], "--gen-uid")) return run_gen_uid_mode(argc, argv);
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank")) return run_single_rank_mode(argc, argv);
  }

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
    if (rank == 0) printf("=== CCO Multi-Process Test (MPI, %d ranks) ===\n", nranks);

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
  if (nranks < 1) {
    printf("No GPUs found.\n");
    return 1;
  }

  return run_fork_mode(nranks);
}
