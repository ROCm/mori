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

// Shared launch harness for multi-rank CCO tests.
//
// Each test provides only:
//   - its __global__ kernel(s)
//   - int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid)
//   - a one-line main() that calls ccoTestMain(...)
// This header owns the rest: HIP_CHECK and the fork / single-rank / gen-uid /
// MPI launch modes. It obtains a ccoUniqueId (rank 0 / parent), distributes it
// over the launch channel (MPI_Bcast or a file), and hands it to run_test, which
// creates the comm via the cco-native ccoCommCreate(uid, nRanks, rank, ...) API —
// no application:: bootstrap types are used.

#pragma once

#ifdef MORI_WITH_MPI
#include <mpi.h>
#endif

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/cco/cco.hpp"

// Current rank, set at the top of each run_test; used by HIP_CHECK diagnostics.
inline int g_rank = 0;

#define HIP_CHECK(cmd)                                                                           \
  do {                                                                                           \
    hipError_t e = (cmd);                                                                        \
    if (e != hipSuccess) {                                                                       \
      fprintf(stderr, "[rank %d] HIP error %d (%s) at %s:%d\n", g_rank, e, hipGetErrorString(e), \
              __FILE__, __LINE__);                                                               \
      _exit(1);                                                                                  \
    }                                                                                            \
  } while (0)

// GDA provider dispatch (CCO_GDA_DISPATCH) lives in mori/cco/cco_scale_out.hpp
// now — it is compile-time (per-NIC build), so GDA tests include that header and
// call CCO_GDA_DISPATCH(Kernel<P, ...><<<...>>>(...)) with no runtime provider.

// Each test implements this; the harness invokes it for every rank with the
// shared cco unique id (rank 0's socket rendezvous, distributed out-of-band).
int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid);

// ── small file helpers (UID exchange for fork / cross-host launches) ──────────

static inline void ccoTestWriteFile(const char* path, const void* data, size_t len) {
  FILE* f = fopen(path, "wb");
  fwrite(data, 1, len, f);
  fclose(f);
}

static inline bool ccoTestReadFile(const char* path, void* data, size_t len) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  bool ok = fread(data, 1, len, f) == len;
  fclose(f);
  return ok;
}

// ── fork mode: spawn nranks children, each binds one GPU ─────────────────────

static inline int ccoTestForkMode(int nranks, const char* name, const char* uidPrefix,
                                  int /*port*/) {
  char uidPath[256];
  snprintf(uidPath, sizeof(uidPath), "%s_%d", uidPrefix, getpid());

  printf("=== %s Test (fork, %d ranks) ===\n", name, nranks);
  fflush(stdout);

  mori::cco::ccoUniqueId uid;
  if (mori::cco::ccoGetUniqueId(&uid) != 0) {
    fprintf(stderr, "ccoGetUniqueId failed (set MORI_SOCKET_IFNAME=<iface>)\n");
    return 1;
  }
  ccoTestWriteFile(uidPath, &uid, sizeof(uid));

  std::vector<pid_t> children;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      mori::cco::ccoUniqueId childUid;
      while (!ccoTestReadFile(uidPath, &childUid, sizeof(childUid))) {
        usleep(10000);
      }
      _exit(run_test(r, nranks, childUid));
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

// ── single-rank mode: one process per rank, for cross-host launches ──────────

static inline int ccoTestSingleRankMode(int argc, char** argv) {
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

  mori::cco::ccoUniqueId uid;
  bool got = false;
  for (int tries = 0; tries < 600; tries++) {
    if (ccoTestReadFile(uidPath, &uid, sizeof(uid))) {
      got = true;
      break;
    }
    usleep(100000);
  }
  if (!got) return -1;

  if (gpuOffset >= 0) HIP_CHECK(hipSetDevice(rank - gpuOffset));

  return run_test(rank, worldSize, uid);
}

// ── gen-uid mode: emit a UID file for cross-host single-rank launches ─────────

static inline int ccoTestGenUidMode(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: --gen-uid IFACE PORT OUTFILE\n");
    return 1;
  }
  const char* iface = argv[2];
  const char* outPath = argv[4];
  // ccoGetUniqueId reads the interface from MORI_SOCKET_IFNAME and picks a free
  // port itself; honour the iface arg by exporting it. (PORT is ignored.)
  setenv("MORI_SOCKET_IFNAME", iface, /*overwrite=*/1);
  mori::cco::ccoUniqueId uid;
  if (mori::cco::ccoGetUniqueId(&uid) != 0) {
    fprintf(stderr, "ccoGetUniqueId failed for iface=%s\n", iface);
    return 1;
  }
  FILE* f = fopen(outPath, "wb");
  if (!f) {
    fprintf(stderr, "fopen(%s) failed\n", outPath);
    return 1;
  }
  fwrite(&uid, 1, sizeof(uid), f);
  fclose(f);
  printf("Wrote UID (%zu bytes) for iface=%s to %s\n", sizeof(uid), iface, outPath);
  return 0;
}

// ── unified entry point: MPI > single-rank > fork ────────────────────────────
//
// name      — human label printed in headers (e.g. "CCO GDA flushAsync")
// uidPrefix — fork-mode UID file prefix    (e.g. "/tmp/cco_gda_flush_async_uid")
// port      — unused (ccoGetUniqueId picks its own free port); kept for API compat
static inline int ccoTestMain(int argc, char** argv, const char* name, const char* uidPrefix,
                              int port) {
  if (argc >= 2 && !strcmp(argv[1], "--gen-uid")) return ccoTestGenUidMode(argc, argv);
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank")) return ccoTestSingleRankMode(argc, argv);
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
    if (rank == 0) printf("=== %s Test (MPI, %d ranks) ===\n", name, nranks);
    // Rank 0 mints the cco unique id (its socket rendezvous) and broadcasts the
    // POD to every rank; all ranks then create the comm via the uniqueId API.
    mori::cco::ccoUniqueId uid;
    if (rank == 0) {
      if (mori::cco::ccoGetUniqueId(&uid) != 0) {
        fprintf(stderr, "ccoGetUniqueId failed (set MORI_SOCKET_IFNAME=<iface>)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    return run_test(rank, nranks, uid);
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

  return ccoTestForkMode(nranks, name, uidPrefix, port);
}
