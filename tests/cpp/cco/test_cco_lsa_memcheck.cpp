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

/*
 * LSA resource leak checker
 *
 * Repeatedly creates and destroys:
 *   - ccoComm               (once, outer)
 *   - ccoWindow pairs       (WINDOW_ITERS times)
 *   - ccoDevComm            (DEVCOMM_ITERS times, inside each window iter)
 *
 * After each destroy, hipMemGetInfo() is sampled on rank 0 and printed.
 * A leak shows as monotonically decreasing free memory across iterations.
 *
 * Run:
 *   mpirun -np <N> --allow-run-as-root \
 *     -x LD_PRELOAD=<libamdhip64.so> \
 *     ./build/examples/lsa_memcheck [--window-iters W] [--devcomm-iters D]
 */

<<<<<<<<HEAD : tests / cpp / cco / test_lsa_memcheck.cpp
#include <cassert>
#include <cstdio>
#include <cstring>

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"  // CCO single header (host + device)
        == == == ==
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <unistd.h>

#include <cstdlib>

// Always-on check: these tests build under -DNDEBUG (Release), where CCO_MUST()
// would drop the wrapped expression together with its side effects. CCO_MUST
// always evaluates expr and aborts the rank on failure (mpirun then tears down
// the whole job).
#define CCO_MUST(expr)                                                                    \
  do {                                                                                    \
    if (!(expr)) {                                                                        \
      std::fprintf(stderr, "[cco lsa test] CHECK FAILED: %s at %s:%d\n", #expr, __FILE__, \
                   __LINE__);                                                             \
      std::abort();                                                                       \
    }                                                                                     \
  } while (0)
#include <cstdio>
#include <cstring>

#include "mori/cco/cco.hpp"  // CCO core header (host + LSA device; no GDA/RDMA)
        >>>>>>>> dev /
    cco : tests / cpp / cco /
          test_cco_lsa_memcheck.cpp

// Tests build with -DNDEBUG (Release), which strips assert(). Re-define an
// always-on check so the assert(...)-style error handling below stays effective.
#undef assert
#define assert(expr)                                                                         \
  do {                                                                                       \
    if (!(expr)) {                                                                           \
      std::fprintf(stderr, "[rank %d] check failed: %s at %s:%d\n", g_rank, #expr, __FILE__, \
                   __LINE__);                                                                \
      std::exit(1);                                                                          \
    }                                                                                        \
  } while (0)

          using namespace mori::cco;

// ── tiny barrier kernel — just enough to exercise the DevComm ──────────────
__global__ void lsa_barrier_kernel(ccoDevComm devComm) {
  ccoCoopBlock coop;
  ccoLsaBarrierSession<ccoCoopBlock> bar(coop, &devComm, ccoTeamLsa(devComm), devComm.lsaBarrier,
                                         0);
  bar.sync(coop);
}

// ── main ───────────────────────────────────────────────────────────────────
int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
  g_rank = rank;

  int window_iters = 100;
  int devcomm_iters = 1;

  if (rank == 0) {
    printf("lsa_memcheck: nranks=%d  window_iters=%d  devcomm_iters=%d\n\n", nranks, window_iters,
           devcomm_iters);
    fflush(stdout);
  }

  < < < < < < < < HEAD : tests / cpp / cco / test_lsa_memcheck.cpp == == == ==
      // ── Phase 1: ccoComm (self-contained bootstrap) ──
      // MPI is only the launcher + a one-shot broadcast of the cco unique id.
      ccoUniqueId uid;
  if (rank == 0) CCO_MUST(ccoGetUniqueId(&uid) == 0);
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

  >>>>>>>> dev / cco : tests / cpp / cco /
                       test_cco_lsa_memcheck.cpp
                       // Bind each rank to its own GPU BEFORE ccoCommCreate. ccoCommCreate calls
                       // hipGetDevice() and pins all allocations to the current device, so without
                       // this every rank would land on GPU 0 (all 8 GB windows stacked on PE0).
                       int hipDevCount = 0;
  CCO_MUST(hipGetDeviceCount(&hipDevCount) == hipSuccess);
  CCO_MUST(hipSetDevice(rank % hipDevCount) == hipSuccess);

  mori::cco::ccoComm* comm = nullptr;
  < < < < < < < <
      HEAD : tests / cpp / cco /
             test_lsa_memcheck.cpp if (ccoCommCreate(bootNet, /*perRankVmmSize=*/0, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }
  == == == == CCO_MUST(ccoCommCreate(uid, nranks, rank, 0, &comm) == 0);
  >>>>>>>> dev / cco : tests / cpp / cco /
                       test_cco_lsa_memcheck.cpp

                       // ── Phase 2: window + devcomm leak loop ──
                       for (int wi = 0; wi < window_iters; wi++) {
    // Register a window pair.
    const size_t winBytes = 8ULL << 30;  // 8 GB
    void* buf = nullptr;
    ccoWindow_t win = nullptr;
    CCO_MUST(ccoWindowRegister(comm, winBytes, &win, &buf) == 0);

    // ── DevComm loop ──
    for (int di = 0; di < devcomm_iters; di++) {
      ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
      reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
      reqs.lsaBarrierCount = 1;

      // Host struct, filled in place; kernel takes it by value.
      ccoDevComm devComm{};
      CCO_MUST(ccoDevCommCreate(comm, &reqs, &devComm) == 0);

      // Exercise the barrier so the DevComm is actually used.
      lsa_barrier_kernel<<<1, 64>>>(devComm);
      CCO_MUST(hipDeviceSynchronize() == hipSuccess);

      ccoDevCommDestroy(comm, &devComm);
    }

    printf("rank[%d] ccoWindowDeregister %d/%d\n", rank, wi, window_iters);

    // Deregister window.
    ccoWindowDeregister(comm, win);
    ccoMemFree(comm, buf);

    // Barrier: ensure every rank has torn down its peer mappings of this
    // window (in ccoWindowDeregister) and freed its slot before any rank
    // reuses the same flat VA in the next iteration. Without this, one rank's
    // next-iter hipMemMap at the reused VA can race a peer that still holds the
    // old dma-buf mapping, causing hsa_amd_vmem_map to fail.
    ccoBarrierAll(comm);
  }

  // ── Teardown ──
  // cco owns the internal socket bootstrap (built from the unique id) and tears
  // it down in ccoCommDestroy. MPI is only our launcher + id broadcast, so we
  // finalize it ourselves.
  ccoCommDestroy(comm);
  printf("[rank %d] PASSED\n", rank);
  return 0;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO LSA memcheck", "/tmp/cco_lsa_memcheck_uid", 19884);
}
