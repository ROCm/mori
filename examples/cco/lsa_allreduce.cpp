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
 * Mirrors the NCCL docs/examples/06_device_api/01_allreduce_lsa pattern:
 *  - One barrier slot per CTA (blockIdx.x selects the slot).
 *  - Separate send / recv windows.
 *  - Two barrier syncs (acquire at start, release at end).
 *  - getLsaPeerPtr() for direct peer memory access.
 *
 * Host flow:
 *   CcoCommCreate → CcoWindowRegister (x2) → CcoDevCommCreate
 *   → kernel → CcoDevCommDestroy → CcoWindowDeregister → CcoCommDestroy
 */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <vector>

#include "args_parser.hpp"
#include "mori/cco/cco_api.hpp"
#include "mori/cco/cco_device_api.hpp"
#include "mori/cco/cco_lsa_impl.hpp"
#include "mori/cco/cco_lsa_types.hpp"
#include "mori/cco/cco_types.hpp"

#define CTA_COUNT        (16)
#define THREADS_PER_CTA  (512)
#define NELEMS           (1024 * 1024)  // 1M floats = 4 MB

using namespace mori::cco;
namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Kernel — direct CCo Device API AllReduce over LSA
//
// Each CTA owns barrier slot blockIdx.x.
// Threads spread work via a global-thread-ID stride loop across all lsaSize
// ranks, matching the NCCL example's globalTid / globalNthreads pattern.
// ---------------------------------------------------------------------------
__global__ void lsa_allreduce_kernel(CcoDevComm* devComm,
                                     CcoWindow_t sendWin, size_t sendOff,
                                     CcoWindow_t recvWin,  size_t recvOff,
                                     size_t count) {
  cg::thread_block blk = cg::this_thread_block();

  // Each CTA gets its own dedicated barrier slot.
  CcoLsaBarrierSession<cg::thread_block> bar(blk, devComm, devComm->lsaBarrier, blockIdx.x);

  // Acquire barrier — wait until all peers have written their send buffers.
  bar.sync(blk);

  const int lsaSize = devComm->lsaSize;
  const int lsaRank = devComm->lsaRank;

  // Global thread ID spread across all ranks, matching NCCL example.
  const size_t globalTid      = threadIdx.x + blockDim.x * (lsaRank + blockIdx.x * lsaSize);
  const size_t globalNthreads = blockDim.x * gridDim.x * lsaSize;

  for (size_t i = globalTid; i < count; i += globalNthreads) {
    float v = 0.f;
    // Reduce: read element i from every peer's send buffer.
    for (int peer = 0; peer < lsaSize; peer++) {
      float* src = reinterpret_cast<float*>(getLsaPeerPtr(sendWin, peer, sendOff));
      v += src[i];
    }
    // Scatter result into every peer's recv buffer.
    for (int peer = 0; peer < lsaSize; peer++) {
      float* dst = reinterpret_cast<float*>(getLsaPeerPtr(recvWin, peer, recvOff));
      dst[i] = v;
    }
  }

  // Release barrier — signal peers that recv buffers are fully written.
  bar.sync(blk);
}

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
  CcoComm* comm = nullptr;
  assert(CcoCommCreate(boot, 0, &comm) == 0);

  const size_t sizeBytes = NELEMS * sizeof(float);

  // ── Phase 2: register send and recv windows ──
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  CcoWindow_t sendWin = nullptr;
  CcoWindow_t recvWin = nullptr;
  assert(CcoWindowRegister(comm, sizeBytes, &sendWin, &sendBuf) == 0);
  assert(CcoWindowRegister(comm, sizeBytes, &recvWin, &recvBuf) == 0);

  // Initialize send buffer: each rank fills with its rank value.
  // sendBuf is GPU VMM memory — use a host staging buffer + hipMemcpy.
  {
    std::vector<float> host(NELEMS, static_cast<float>(rank));
    assert(hipMemcpy(sendBuf, host.data(), sizeBytes, hipMemcpyHostToDevice) == hipSuccess);
  }

  if (rank == 0) {
    printf("Starting LSA AllReduce: %zu elements (%.0f MB), %d ranks\n",
           (size_t)NELEMS, sizeBytes / 1e6, nranks);
    printf("Expected result per element: %.0f\n",
           (float)(nranks * (nranks - 1)) / 2.f);
  }

  // ── Phase 3: device communicator — one barrier slot per CTA ──
  CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.lsaBarrierCount   = CTA_COUNT;

  CcoDevComm* devComm = nullptr;
  assert(CcoDevCommCreate(comm, &reqs, &devComm) == 0);

  printf("  Rank %d: DevComm ready, lsaSize=%d lsaRank=%d\n",
         rank, devComm->lsaSize, devComm->lsaRank);

  // ── Launch ──
  lsa_allreduce_kernel<<<CTA_COUNT, THREADS_PER_CTA>>>(
      devComm, sendWin, 0, recvWin, 0, NELEMS);
  assert(hipDeviceSynchronize() == hipSuccess);

  // ── Verify ──
  float expected = static_cast<float>(nranks * (nranks - 1)) / 2.f;
  int errors = 0;
  {
    std::vector<float> host(NELEMS);
    assert(hipMemcpy(host.data(), recvBuf, sizeBytes, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < NELEMS; i++) {
      if (host[i] != expected) errors++;
    }
  }
  printf("  Rank %d: %s (expected=%.0f errors=%d)\n",
         rank, errors == 0 ? "PASS" : "FAIL", expected, errors);

  // ── Teardown ──
  CcoDevCommDestroy(comm, devComm);
  CcoWindowDeregister(comm, sendWin);
  CcoWindowDeregister(comm, recvWin);
  CcoMemFree(comm, sendBuf);
  CcoMemFree(comm, recvBuf);
  CcoCommDestroy(comm);
  delete boot;
  MPI_Finalize();
  return errors != 0 ? 1 : 0;
}
