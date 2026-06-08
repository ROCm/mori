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
 * CCo Device API AllReduce Example (intra-node LSA, multi-block / multi-slot)
 *
 * Mirrors NCCL's device-API allreduce: the kernel is launched with CTA_COUNT
 * blocks, each block drives its OWN lsaBarrier slot (slot = blockIdx.x), and
 * the workload is spread across the full (block × rank × thread) grid via a
 * grid-stride loop. lsaBarrierCount must therefore equal CTA_COUNT.
 *
 * Three variants of the same allreduce-sum, one per CCo coop type — identical
 * apart from the barrier cooperation granularity:
 *   BLOCK  : ccoCoopBlock   ── whole CTA cooperates  (launch THREADS_PER_CTA)
 *   WARP   : ccoCoopWarp    ── one wavefront per CTA (launch 64)
 *   THREAD : ccoCoopThread  ── one thread per CTA    (launch 1)
 *
 * Per-rank input vector: all elements = rank.
 * Expected per-rank output: all elements = N(N-1)/2 on every rank.
 */

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

#include "args_parser.hpp"
#include "mori/cco/cco_api.hpp"         // host control-plane
#include "mori/cco/cco_device_api.hpp"  // device-side (kernel) API

// Larger vector so the multi-block grid-stride loop actually spreads work
// across blocks (each rank r contributes a vector of all r's).
#define NELEMS (4096)

// Launch geometry. CTA_COUNT blocks ⇒ CTA_COUNT barrier slots; must match
// reqs.lsaBarrierCount. Each block owns slot == blockIdx.x.
#define CTA_COUNT (8)
#define THREADS_PER_CTA (256)

using namespace mori::cco;

// ===========================================================================
// Multi-block / multi-slot allreduce kernel — generic over CCo coop type.
// ===========================================================================
//
// Launched with CTA_COUNT blocks. Each block:
//   1. opens a ccoLsaBarrierSession on its OWN slot (blockIdx.x)
//   2. sync (acquire — wait for peers' sendBuf to be ready)
//   3. grid-stride over elements it owns:
//        v = sum over peers of sendBuf[i]; write v to every peer's recvBuf[i]
//   4. sync (release — signal recvBuf is fully written)
//
// Work is spread across the whole (block × rank × thread) grid like the NCCL
// device-API example: every barrier slot is exercised concurrently, and each
// element is owned by exactly one (rank, block, lane) triple.
//
// The Coop type only changes the cooperation granularity within a block:
//   ccoCoopBlock  → all THREADS_PER_CTA threads stride together
//   ccoCoopWarp   → the first wavefront (64 lanes) strides
//   ccoCoopThread → lane 0 alone strides
template <typename Coop>
__global__ void lsa_allreduce_kernel(ccoDevComm* devComm, ccoWindow_t sendWin, size_t sendOff,
                                     ccoWindow_t recvWin, size_t recvOff, size_t count) {
  Coop coop;
  ccoLsaBarrierSession<Coop> bar(coop, devComm, ccoTeamLsa(*devComm), devComm->lsaBarrier,
                                 blockIdx.x);
  bar.sync(coop);

  const int lsaSize = devComm->lsaSize;
  const int lane = coop.thread_rank();
  const int stride = coop.size();

  // Global element ownership: block b, lane l of this rank starts at
  // (b * stride + l) and steps by (gridDim.x * stride). Peers cover the same
  // index set on their own GPUs, and the cross-GPU reduction below makes the
  // result identical on every rank, so no inter-rank work partition is needed.
  for (size_t i = blockIdx.x * stride + lane; i < count; i += gridDim.x * stride) {
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

  // Bind each rank to its own GPU BEFORE ccoCommCreate (which calls
  // hipGetDevice() and pins allocations to the current device).
  int hipDevCount = 0;
  assert(hipGetDeviceCount(&hipDevCount) == hipSuccess);
  assert(hipSetDevice(boot->GetLocalRank() % hipDevCount) == hipSuccess);

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

  // Print input (rank by rank in order). Show only the first few elements.
  const size_t kShow = std::min<size_t>(NELEMS, 8);
  for (int r = 0; r < nranks; r++) {
    ccoBarrierAll(comm);
    if (rank == r) {
      char buf[256];
      int n = 0;
      n += snprintf(buf + n, sizeof(buf) - n, "  Rank %d INPUT  (", rank);
      for (size_t i = 0; i < kShow; i++)
        n += snprintf(buf + n, sizeof(buf) - n, "%s%.0f", i ? "," : "", sendHost[i]);
      n += snprintf(buf + n, sizeof(buf) - n, "%s)\n", NELEMS > kShow ? ",..." : "");
      fputs(buf, stdout);
      fflush(stdout);
    }
  }

  ccoBarrierAll(comm);
  const float expected = static_cast<float>(nranks * (nranks - 1)) / 2.f;
  if (rank == 0) {
    printf("AllReduce-SUM over %d ranks of %zu-elem vectors  ⇒  expected all = %.0f\n", nranks,
           (size_t)NELEMS, expected);
    fflush(stdout);
  }

  ccoBarrierAll(comm);

  // ── Phase 3: device communicator (one barrier slot per CTA) ──
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.lsaBarrierCount = CTA_COUNT;

  ccoDevComm* devComm = nullptr;
  assert(ccoDevCommCreate(comm, &reqs, &devComm) == 0);

  if (rank == 0) {
    printf("DevComm ready, lsaSize=%d  grid=%d blocks × %d slots  (3 coop variants)\n",
           devComm->lsaSize, CTA_COUNT, CTA_COUNT);
  }

  // ── Helper: launch one variant, verify, print ──
  int totalErrors = 0;
  auto run_variant = [&](const char* name, auto launch_fn) {
    // Zero recvBuf so each variant is independently verified.
    assert(hipMemset(recvBuf, 0, sizeBytes) == hipSuccess);

    launch_fn();
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> recvHost(NELEMS);
    assert(hipMemcpy(recvHost.data(), recvBuf, sizeBytes, hipMemcpyDeviceToHost) == hipSuccess);
    int errors = 0;
    for (size_t i = 0; i < NELEMS; i++)
      if (recvHost[i] != expected) errors++;
    totalErrors += errors;

    // Print only the first few elements (NELEMS is large now).
    const size_t kShow = std::min<size_t>(NELEMS, 8);
    char buf[256];
    int n = 0;
    n += snprintf(buf + n, sizeof(buf) - n, "  Rank %d [%-6s] RESULT (", rank, name);
    for (size_t i = 0; i < kShow; i++)
      n += snprintf(buf + n, sizeof(buf) - n, "%s%.0f", i ? "," : "", recvHost[i]);
    n += snprintf(buf + n, sizeof(buf) - n, "%s)  %s  (expected=%.0f errors=%d)\n",
                  NELEMS > kShow ? ",..." : "", errors == 0 ? "PASS" : "FAIL", expected, errors);
    fputs(buf, stdout);
    fflush(stdout);
  };

  // All three launch CTA_COUNT blocks (one barrier slot each); they differ only
  // in block width, which fixes how many threads the coop type cooperates over.

  // ── BLOCK variant ── full CTA cooperates.
  run_variant("block", [&] {
    lsa_allreduce_kernel<ccoCoopBlock>
        <<<CTA_COUNT, THREADS_PER_CTA>>>(devComm, sendWin, 0, recvWin, 0, NELEMS);
  });

  // ── WARP variant ── one wavefront (64 lanes) per block.
  run_variant("warp", [&] {
    lsa_allreduce_kernel<ccoCoopWarp><<<CTA_COUNT, 64>>>(devComm, sendWin, 0, recvWin, 0, NELEMS);
  });

  // ── THREAD variant ── one thread per block.
  run_variant("thread", [&] {
    lsa_allreduce_kernel<ccoCoopThread><<<CTA_COUNT, 1>>>(devComm, sendWin, 0, recvWin, 0, NELEMS);
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
