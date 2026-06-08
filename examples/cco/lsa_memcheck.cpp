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

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstring>

#include "args_parser.hpp"
#include "mori/cco/cco.hpp"         // host control-plane
#include "mori/cco/cco_device.hpp"  // device-side (kernel) API

using namespace mori::cco;

// ── tiny barrier kernel — just enough to exercise the DevComm ──────────────
__global__ void lsa_barrier_kernel(ccoDevComm* devComm) {
  ccoCoopBlock coop;
  ccoLsaBarrierSession<ccoCoopBlock> bar(coop, devComm, ccoTeamLsa(*devComm), devComm->lsaBarrier,
                                         0);
  bar.sync(coop);
}

// ── main ───────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
#ifndef MORI_WITH_MPI
  fprintf(stderr, "lsa_memcheck requires MORI_WITH_MPI (enable WITH_MPI).\n");
  return 1;
#endif

  // ── parse optional args ──
  int window_iters = 100;
  int devcomm_iters = 1;

  // ── MPI init ──
  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  if (rank == 0) {
    printf("lsa_memcheck: nranks=%d  window_iters=%d  devcomm_iters=%d\n\n", nranks, window_iters,
           devcomm_iters);
    fflush(stdout);
  }

  // ── Phase 1: ccoComm (created once, destroyed at the end) ──
  auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);

  // Bind each rank to its own GPU BEFORE ccoCommCreate. ccoCommCreate calls
  // hipGetDevice() and pins all allocations to the current device, so without
  // this every rank would land on GPU 0 (all 8 GB windows stacked on PE0).
  int hipDevCount = 0;
  assert(hipGetDeviceCount(&hipDevCount) == hipSuccess);
  assert(hipSetDevice(boot->GetLocalRank() % hipDevCount) == hipSuccess);

  mori::cco::ccoComm* comm = nullptr;
  assert(ccoCommCreate(boot, 0, &comm) == 0);

  // ── Phase 2: window + devcomm leak loop ──
  for (int wi = 0; wi < window_iters; wi++) {
    // Register a window pair.
    const size_t winBytes = 8ULL << 30;  // 8 GB
    void* buf = nullptr;
    ccoWindow_t win = nullptr;
    assert(ccoWindowRegister(comm, winBytes, &win, &buf) == 0);

    // ── DevComm loop ──
    for (int di = 0; di < devcomm_iters; di++) {
      ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
      reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
      reqs.lsaBarrierCount = 1;

      ccoDevComm* devComm = nullptr;
      assert(ccoDevCommCreate(comm, &reqs, &devComm) == 0);

      // Exercise the barrier so the DevComm is actually used.
      lsa_barrier_kernel<<<1, 64>>>(devComm);
      assert(hipDeviceSynchronize() == hipSuccess);

      ccoDevCommDestroy(comm, devComm);
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
  // bootstrap ownership transfers to ccoComm; ccoCommDestroy calls
  // bootNet->Finalize() + delete bootNet, which calls MPI_Finalize().
  ccoCommDestroy(comm);
  return 0;
}
