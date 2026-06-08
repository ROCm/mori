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
// Test: CCO GDA connection modes (NONE / CROSSNODE / FULL).
// Validates that ccoDevCommCreate honors reqs.gdaConnectionType:
//   - NONE      : 0 QPs allocated
//   - CROSSNODE : 0 QPs allocated on a single-node deployment (auto-collapsed
//                 to NONE because lsaSize == worldSize), otherwise per-cross-node-peer
//   - FULL      : (worldSize - 1) × numQpPerPe QPs allocated (one per non-self peer)
//
// Single process, N threads.

#include <cstdio>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/cco/cco.hpp"
#include "mori/utils/mori_log.hpp"

#define HIP_CHECK(cmd)                                                            \
  do {                                                                            \
    hipError_t e = (cmd);                                                         \
    if (e != hipSuccess) {                                                        \
      fprintf(stderr, "[rank ?] HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 64ULL * 1024 * 1024;

struct Result {
  int rank;
  bool passed;
  char detail[256];
};

// Read DevComm back to host, count non-zero QPs in the IBGDA endpoint array.
static int CountQpsFor(mori::cco::ccoDevComm* devComm, int worldSize) {
  mori::cco::ccoDevComm host;
  HIP_CHECK(hipMemcpy(&host, devComm, sizeof(host), hipMemcpyDeviceToHost));
  if (host.ibgda.endpoints == nullptr || host.ibgda.numQpPerPe == 0) return 0;
  size_t total = static_cast<size_t>(worldSize) * host.ibgda.numQpPerPe;
  std::vector<mori::shmem::ShmemRdmaEndpoint> eps(total);
  HIP_CHECK(
      hipMemcpy(eps.data(), host.ibgda.endpoints, total * sizeof(eps[0]), hipMemcpyDeviceToHost));
  int count = 0;
  for (const auto& ep : eps) {
    if (ep.qpn != 0) count++;
  }
  return count;
}

static void run_rank(int rank, int nranks, const mori::application::UniqueId& uid, Result* r) {
  r->rank = rank;
  r->passed = false;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  auto* bootNet = new mori::application::SocketBootstrapNetwork(uid, rank, nranks);

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    snprintf(r->detail, sizeof(r->detail), "CommCreate failed");
    return;
  }

  // Build three DevComms with different connection types.
  auto makeReqs = [](mori::cco::ccoGdaConnectionType ct) {
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = ct;
    reqs.lsaBarrierCount = 4;      // LSA barrier slab in resource window
    reqs.railGdaBarrierCount = 2;  // rail GDA barrier → IBGDA signal pool
    reqs.barrierCount = 3;         // hybrid LSA + rail GDA
    return reqs;
  };

  mori::cco::ccoDevComm* dcNone = nullptr;
  mori::cco::ccoDevComm* dcFull = nullptr;
  mori::cco::ccoDevComm* dcRail = nullptr;
  auto reqsNone = makeReqs(mori::cco::CCO_GDA_CONNECTION_NONE);
  auto reqsFull = makeReqs(mori::cco::CCO_GDA_CONNECTION_FULL);
  auto reqsRail = makeReqs(mori::cco::CCO_GDA_CONNECTION_RAIL);
  const int numQpPerPe = reqsFull.gdaContextCount;

  if (mori::cco::ccoDevCommCreate(comm, &reqsNone, &dcNone) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate NONE failed");
    mori::cco::ccoCommDestroy(comm);
    return;
  }
  if (mori::cco::ccoDevCommCreate(comm, &reqsFull, &dcFull) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate FULL failed");
    mori::cco::ccoDevCommDestroy(comm, dcNone);
    mori::cco::ccoCommDestroy(comm);
    return;
  }
  if (mori::cco::ccoDevCommCreate(comm, &reqsRail, &dcRail) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate RAIL failed");
    mori::cco::ccoDevCommDestroy(comm, dcFull);
    mori::cco::ccoDevCommDestroy(comm, dcNone);
    mori::cco::ccoCommDestroy(comm);
    return;
  }

  // Expectations on a uniform N-nodes × lsaSize layout:
  //   NONE : 0
  //   FULL : (worldSize - 1) * qpsPerPe
  //   RAIL : (nNodes - 1) * qpsPerPe  (one peer per other node at same lsaRank)
  // On single-node (nNodes == 1), RAIL collapses to NONE: expected 0.
  const int nNodes = comm->worldSize / comm->lsaSize;
  const int qpsNone = CountQpsFor(dcNone, comm->worldSize);
  const int qpsFull = CountQpsFor(dcFull, comm->worldSize);
  const int qpsRail = CountQpsFor(dcRail, comm->worldSize);
  const int expectedFull = (comm->worldSize - 1) * numQpPerPe;
  const int expectedRail = (nNodes - 1) * numQpPerPe;

  bool ok = true;
  if (qpsNone != 0) {
    snprintf(r->detail, sizeof(r->detail), "NONE: expected 0, got %d", qpsNone);
    ok = false;
  } else if (qpsFull != expectedFull) {
    snprintf(r->detail, sizeof(r->detail), "FULL: expected %d, got %d", expectedFull, qpsFull);
    ok = false;
  } else if (qpsRail != expectedRail) {
    snprintf(r->detail, sizeof(r->detail), "RAIL: expected %d ((nNodes-1)*qpsPerPe=%d*%d), got %d",
             expectedRail, nNodes - 1, numQpPerPe, qpsRail);
    ok = false;
  } else {
    snprintf(r->detail, sizeof(r->detail),
             "NONE=0 FULL=%d RAIL=%d (worldSize=%d lsaSize=%d nNodes=%d qpsPerPe=%d)", qpsFull,
             qpsRail, comm->worldSize, comm->lsaSize, nNodes, numQpPerPe);
  }

  printf("[rank %d] NONE=%d FULL=%d RAIL=%d (expected: 0 / %d / %d)\n", rank, qpsNone, qpsFull,
         qpsRail, expectedFull, expectedRail);

  mori::cco::ccoDevCommDestroy(comm, dcRail);
  mori::cco::ccoDevCommDestroy(comm, dcFull);
  mori::cco::ccoDevCommDestroy(comm, dcNone);
  mori::cco::ccoCommDestroy(comm);

  r->passed = ok;
  if (ok) printf("[rank %d] PASSED\n", rank);
}

int main(int argc, char** argv) {
  mori::ModuleLogger::GetInstance().GetLogger(mori::modules::APPLICATION);
  mori::ModuleLogger::GetInstance().GetLogger(mori::modules::SHMEM);

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int nranks = numDevices;
  if (argc > 1) nranks = std::min(atoi(argv[1]), numDevices);
  if (nranks < 2) {
    printf("Need at least 2 GPUs.\n");
    return 1;
  }

  printf("=== CCO GDA Connection Modes Test (%d ranks) ===\n\n", nranks);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 18458);

  std::vector<Result> results(nranks);
  std::vector<std::thread> threads;
  for (int r = 0; r < nranks; r++) {
    threads.emplace_back(run_rank, r, nranks, std::cref(uid), &results[r]);
  }
  for (auto& t : threads) t.join();

  printf("\n=== Summary ===\n");
  int pass = 0, fail = 0;
  for (auto& r : results) {
    printf("  rank %d: [%s] %s\n", r.rank, r.passed ? "PASS" : "FAIL", r.detail);
    r.passed ? pass++ : fail++;
  }
  printf("\n%d passed, %d failed\n", pass, fail);
  return (fail == 0) ? 0 : 1;
}
