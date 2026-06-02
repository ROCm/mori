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
 * CCo Device API AllReduce Example (intra-node LSA)
 *
 * Demonstrates three variants of the same allreduce-sum, one per CCo group
 * abstraction (mori/include/mori/cco/cco_group.hpp):
 *
 *   BLOCK  : ccoBlockGroup   ── CTA-wide cooperation, __syncthreads()
 *   WARP   : ccoWarpGroup    ── wavefront-wide cooperation, wave_barrier()
 *   THREAD : ccoThreadGroup  ── single-thread, no-op sync
 *
 * Per-rank input vector: (rank, rank, rank, rank).
 * Expected per-rank output: (N(N-1)/2, ...) on every rank.
 */

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

#include "args_parser.hpp"
#include "mori/cco/cco_api.hpp"
#include "mori/cco/cco_device_api.hpp"
#include "mori/cco/cco_coop.hpp"
#include "mori/cco/cco_lsa_impl.hpp"
#include "mori/cco/cco_lsa_types.hpp"
#include "mori/cco/cco_team.hpp"
#include "mori/cco/cco_types.hpp"

#define NELEMS  (32)  // tiny vector: rank r contributes (r,r,r,r)

using namespace mori::cco;

// ===========================================================================
// Three allreduce kernels — one per CCo group type.
// ===========================================================================
//
// All three follow the same protocol:
//   1. open a ccoLsaBarrierSession on barrier slot 0
//   2. sync (acquire — wait for peers' sendBuf to be ready)
//   3. for each owned element i:
//        v = sum over peers of sendBuf[i]
//        write v to every peer's recvBuf[i]
//   4. sync (release — signal recvBuf is fully written)
//
// They differ only in:
//   * the launch shape they expect
//   * the Group type used for the barrier session
//   * how work is distributed inside the group
// ---------------------------------------------------------------------------

// ─── BLOCK variant ─────────────────────────────────────────────────────────
//   Launch: <<<1, blockDim>>>
//   All threads of the CTA cooperate. Stride loop over threadIdx.x.
__global__ void lsa_allreduce_block_kernel(ccoDevComm* devComm,
                                           ccoWindow_t sendWin, size_t sendOff,
                                           ccoWindow_t recvWin, size_t recvOff,
                                           size_t count) {
  ccoCoopBlock coop;
  ccoLsaBarrierSession<ccoCoopBlock> bar(coop, devComm, devComm->lsaBarrier, 0);
  bar.sync(coop);

  const int lsaSize = devComm->lsaSize;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    float v = 0.f;
    for (int peer = 0; peer < lsaSize; peer++) {
      v += reinterpret_cast<float*>(getLsaPeerPtr(sendWin, peer, sendOff))[i];
    }
    for (int peer = 0; peer < lsaSize; peer++) {
      reinterpret_cast<float*>(getLsaPeerPtr(recvWin, peer, recvOff))[i] = v;
    }
  }

  bar.sync(coop);
}

// ─── WARP variant ──────────────────────────────────────────────────────────
//   Launch: <<<1, 64>>>  (exactly one wavefront on AMD)
//   The 64 lanes of one warp cooperate. Stride loop over lane id.
__global__ void lsa_allreduce_warp_kernel(ccoDevComm* devComm,
                                          ccoWindow_t sendWin, size_t sendOff,
                                          ccoWindow_t recvWin, size_t recvOff,
                                          size_t count) {
  ccoCoopWarp coop;
  ccoLsaBarrierSession<ccoCoopWarp> bar(coop, devComm, devComm->lsaBarrier, 0);
  bar.sync(coop);

  const int lsaSize = devComm->lsaSize;
  const int lane    = __lane_id();
  // `warpSize` is a HIP built-in __device__ const int (64 on AMD gfx9+).

  for (size_t i = lane; i < count; i += warpSize) {
    float v = 0.f;
    for (int peer = 0; peer < lsaSize; peer++) {
      v += reinterpret_cast<float*>(getLsaPeerPtr(sendWin, peer, sendOff))[i];
    }
    for (int peer = 0; peer < lsaSize; peer++) {
      reinterpret_cast<float*>(getLsaPeerPtr(recvWin, peer, recvOff))[i] = v;
    }
  }

  bar.sync(coop);
}

// ─── THREAD variant ────────────────────────────────────────────────────────
//   Launch: <<<1, 1>>>  (one single thread per rank)
//   That thread does the whole allreduce serially.
__global__ void lsa_allreduce_thread_kernel(ccoDevComm* devComm,
                                            ccoWindow_t sendWin, size_t sendOff,
                                            ccoWindow_t recvWin, size_t recvOff,
                                            size_t count) {
  ccoCoopThread coop;
  ccoLsaBarrierSession<ccoCoopThread> bar(coop, devComm, devComm->lsaBarrier, 0);
  bar.sync(coop);

  const int lsaSize = devComm->lsaSize;
  for (size_t i = 0; i < count; i++) {
    float v = 0.f;
    for (int peer = 0; peer < lsaSize; peer++) {
      v += reinterpret_cast<float*>(getLsaPeerPtr(sendWin, peer, sendOff))[i];
    }
    for (int peer = 0; peer < lsaSize; peer++) {
      reinterpret_cast<float*>(getLsaPeerPtr(recvWin, peer, recvOff))[i] = v;
    }
  }

  bar.sync(coop);
}

// ===========================================================================
// Host driver
// ===========================================================================
int main(int argc, char* argv[]) {
#ifndef MORI_WITH_MPI
  std::fprintf(stderr, "lsa_allreduce requires MORI_WITH_MPI (enable WITH_MPI).\n");
  return 1;
#endif

  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // ── Phase 1: communicator ──
  auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);
  ccoComm* comm = nullptr;
  assert(ccoCommCreate(boot, 0, &comm) == 0);

  const size_t sizeBytes = NELEMS * sizeof(float);

  // ── Phase 2: register send/recv windows + init send buffer ──
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  ccoWindow_t sendWin = nullptr;
  ccoWindow_t recvWin = nullptr;
  assert(ccoWindowRegister(comm, sizeBytes, &sendWin, &sendBuf) == 0);
  assert(ccoWindowRegister(comm, sizeBytes, &recvWin, &recvBuf) == 0);

  // Each rank's send vector is (rank, rank, rank, rank).
  std::vector<float> sendHost(NELEMS, static_cast<float>(rank));
  assert(hipMemcpy(sendBuf, sendHost.data(), sizeBytes, hipMemcpyHostToDevice) == hipSuccess);

  // Print input (rank by rank in order).
  for (int r = 0; r < nranks; r++) {
    ccoBarrierAll(comm);
    if (rank == r) {
      char buf[256]; int n = 0;
      n += snprintf(buf + n, sizeof(buf) - n, "  Rank %d INPUT  (", rank);
      for (size_t i = 0; i < NELEMS; i++)
        n += snprintf(buf + n, sizeof(buf) - n, "%s%.0f", i ? "," : "", sendHost[i]);
      n += snprintf(buf + n, sizeof(buf) - n, ")\n");
      fputs(buf, stdout); fflush(stdout);
    }
  }

  ccoBarrierAll(comm);
  const float expected = static_cast<float>(nranks * (nranks - 1)) / 2.f;
  if (rank == 0) {
    printf("AllReduce-SUM over %d ranks of %zu-elem vectors  ⇒  expected = (%.0f",
           nranks, (size_t)NELEMS, expected);
    for (size_t i = 1; i < NELEMS; i++) printf(",%.0f", expected);
    printf(")\n");
    fflush(stdout);
  }

  ccoBarrierAll(comm);



  // ── Phase 3: device communicator (1 barrier slot is enough for all 3) ──
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.lsaBarrierCount   = 1;

  ccoDevComm* devComm = nullptr;
  assert(ccoDevCommCreate(comm, &reqs, &devComm) == 0);

  if (rank == 0) {
    printf("DevComm ready, lsaSize=%d  (running 3 group variants back-to-back)\n",
           devComm->lsaSize);
  }

  // ── Helper: launch one variant, verify, print ──
  int totalErrors = 0;
  auto run_variant = [&](const char* name, auto launch_fn) {
    // Zero recvBuf so each variant is independently verified.
    assert(hipMemset(recvBuf, 0, sizeBytes) == hipSuccess);

    launch_fn();
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> recvHost(NELEMS);
    assert(hipMemcpy(recvHost.data(), recvBuf, sizeBytes,
                     hipMemcpyDeviceToHost) == hipSuccess);
    int errors = 0;
    for (size_t i = 0; i < NELEMS; i++)
      if (recvHost[i] != expected) errors++;
    totalErrors += errors;

    char buf[256]; int n = 0;
    n += snprintf(buf + n, sizeof(buf) - n, "  Rank %d [%-6s] RESULT (", rank, name);
    for (size_t i = 0; i < NELEMS; i++)
      n += snprintf(buf + n, sizeof(buf) - n, "%s%.0f", i ? "," : "", recvHost[i]);
    n += snprintf(buf + n, sizeof(buf) - n, ")  %s  (expected=%.0f errors=%d)\n",
                  errors == 0 ? "PASS" : "FAIL", expected, errors);
    fputs(buf, stdout); fflush(stdout);
  };

  // ── BLOCK variant ──
  run_variant("block",  [&] {
    lsa_allreduce_block_kernel<<<1, 64>>>(devComm, sendWin, 0, recvWin, 0, NELEMS);
  });

  // ── WARP variant ──
  run_variant("warp",   [&] {
    lsa_allreduce_warp_kernel<<<1, 64>>>(devComm, sendWin, 0, recvWin, 0, NELEMS);
  });

  // ── THREAD variant ──
  run_variant("thread", [&] {
    lsa_allreduce_thread_kernel<<<1, 1>>>(devComm, sendWin, 0, recvWin, 0, NELEMS);
  });

  // ── Teardown ──
  ccoDevCommDestroy(comm, devComm);
  ccoWindowDeregister(comm, sendWin);
  ccoWindowDeregister(comm, recvWin);
  ccoMemFree(comm, sendBuf);
  ccoMemFree(comm, recvBuf);

  // bootstrap ownership transfers to ccoComm at ccoCommCreate; ccoCommDestroy
  // does `bootNet->Finalize()` + `delete bootNet`, which calls MPI_Finalize().
  // Don't double-free `boot` or call MPI_Finalize() a second time here.
  ccoCommDestroy(comm);
  return totalErrors != 0 ? 1 : 0;
}
