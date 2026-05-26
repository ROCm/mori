// Test: CCO GDA connection modes (NONE / CROSSNODE / FULL).
// Validates that CcoDevCommCreate honors reqs.gdaConnectionType:
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
#include "mori/cco/cco_api.hpp"
#include "mori/utils/mori_log.hpp"

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "[rank ?] HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 64ULL * 1024 * 1024;

struct Result {
  int rank;
  bool passed;
  char detail[256];
};

// Read DevComm back to host, count non-zero QPs in the IBGDA endpoint array.
static int CountQpsFor(mori::cco::CcoDevComm* devComm, int worldSize) {
  mori::cco::CcoDevComm host;
  HIP_CHECK(hipMemcpy(&host, devComm, sizeof(host), hipMemcpyDeviceToHost));
  if (host.ibgda.endpoints == nullptr || host.ibgda.numQpPerPe == 0) return 0;
  size_t total = static_cast<size_t>(worldSize) * host.ibgda.numQpPerPe;
  std::vector<mori::shmem::ShmemRdmaEndpoint> eps(total);
  HIP_CHECK(hipMemcpy(eps.data(), host.ibgda.endpoints, total * sizeof(eps[0]),
                      hipMemcpyDeviceToHost));
  int count = 0;
  for (const auto& ep : eps) {
    if (ep.qpn != 0) count++;
  }
  return count;
}

static void run_rank(int rank, int nranks, const mori::application::UniqueId& uid,
                     Result* r) {
  r->rank = rank;
  r->passed = false;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  auto* bootNet = new mori::application::SocketBootstrapNetwork(uid, rank, nranks);

  mori::cco::CcoComm* comm = nullptr;
  if (mori::cco::CcoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    snprintf(r->detail, sizeof(r->detail), "CommCreate failed");
    return;
  }

  // Build three DevComms with different connection types.
  auto makeReqs = [](mori::cco::CcoGdaConnectionType ct) {
    mori::cco::CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = ct;
    return reqs;
  };

  mori::cco::CcoDevComm* dcNone = nullptr;
  mori::cco::CcoDevComm* dcFull = nullptr;
  auto reqsNone = makeReqs(mori::cco::CCO_GDA_CONNECTION_NONE);
  auto reqsFull = makeReqs(mori::cco::CCO_GDA_CONNECTION_FULL);
  const int numQpPerPe = reqsFull.gdaContextCount;

  if (mori::cco::CcoDevCommCreate(comm, &reqsNone, &dcNone) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate NONE failed");
    mori::cco::CcoCommDestroy(comm);
    return;
  }
  if (mori::cco::CcoDevCommCreate(comm, &reqsFull, &dcFull) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate FULL failed");
    mori::cco::CcoDevCommDestroy(comm, dcNone);
    mori::cco::CcoCommDestroy(comm);
    return;
  }

  int qpsNone = CountQpsFor(dcNone, comm->worldSize);
  int qpsFull = CountQpsFor(dcFull, comm->worldSize);
  int expectedFull = (comm->worldSize - 1) * numQpPerPe;  // every non-self peer

  bool ok = true;
  if (qpsNone != 0) {
    snprintf(r->detail, sizeof(r->detail), "NONE: expected 0 QPs, got %d", qpsNone);
    ok = false;
  } else if (qpsFull != expectedFull) {
    snprintf(r->detail, sizeof(r->detail),
             "FULL: expected %d QPs ((worldSize-1)*numQpPerPe=%d*%d), got %d",
             expectedFull, comm->worldSize - 1, numQpPerPe, qpsFull);
    ok = false;
  } else {
    snprintf(r->detail, sizeof(r->detail), "NONE=0 FULL=%d (worldSize=%d, qpsPerPe=%d)",
             qpsFull, comm->worldSize, numQpPerPe);
  }

  printf("[rank %d] NONE qps=%d FULL qps=%d (expected: 0 / %d)\n", rank, qpsNone,
         qpsFull, expectedFull);

  mori::cco::CcoDevCommDestroy(comm, dcFull);
  mori::cco::CcoDevCommDestroy(comm, dcNone);
  mori::cco::CcoCommDestroy(comm);

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

  auto uid =
      mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 18458);

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
