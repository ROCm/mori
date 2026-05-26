// Test: CCO host-side API lifecycle
// Single process, N threads (one per GPU) via SocketBootstrapNetwork.
// Validates: CommCreate → MemAlloc → WindowRegister → DevCommCreate → P2P read → teardown.

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/utils/mori_log.hpp"
#include "mori/cco/cco_api.hpp"

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "[rank ?] HIP error %d (%s) at %s:%d\n", e,             \
              hipGetErrorString(e), __FILE__, __LINE__);                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t WINDOW_SIZE = 4096;

struct ThreadResult {
  int rank;
  bool passed;
  char detail[512];
};

static void run_rank(int rank, int nranks, const mori::application::UniqueId& uid,
                     ThreadResult* result) {
  result->rank = rank;
  result->passed = false;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  for (int i = 0; i < numDevices; i++) {
    if (i == dev) continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dev, i));
    if (canAccess) {
      hipError_t err = hipDeviceEnablePeerAccess(i, 0);
      (void)err;
    }
  }

  printf("[rank %d] GPU %d\n", rank, dev);

  auto* bootNet = new mori::application::SocketBootstrapNetwork(uid, rank, nranks);

  // Phase 1: CommCreate
  mori::cco::CcoComm* comm = nullptr;
  int ret = mori::cco::CcoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm);
  if (ret != 0) {
    snprintf(result->detail, sizeof(result->detail), "CommCreate failed: %d", ret);
    return;
  }
  printf("[rank %d] CommCreate OK\n", rank);

  // Phase 1.5: MemAlloc
  void* buf = nullptr;
  ret = mori::cco::CcoMemAlloc(comm, WINDOW_SIZE, &buf);
  if (ret != 0) {
    snprintf(result->detail, sizeof(result->detail), "MemAlloc failed: %d", ret);
    mori::cco::CcoCommDestroy(comm);
    return;
  }
  printf("[rank %d] MemAlloc OK: buf=%p\n", rank, buf);

  // Phase 1.6: Allocator path coverage (size==0 + freelist reuse).
  {
    void* z = reinterpret_cast<void*>(0x1);
    if (mori::cco::CcoMemAlloc(comm, 0, &z) != 0 || z != nullptr) {
      snprintf(result->detail, sizeof(result->detail), "size==0 alloc didn't return nullptr");
      mori::cco::CcoMemFree(comm, buf);
      mori::cco::CcoCommDestroy(comm);
      return;
    }
    void* a = nullptr;
    void* b = nullptr;
    if (mori::cco::CcoMemAlloc(comm, WINDOW_SIZE, &a) != 0 ||
        mori::cco::CcoMemFree(comm, a) != 0 ||
        mori::cco::CcoMemAlloc(comm, WINDOW_SIZE, &b) != 0) {
      snprintf(result->detail, sizeof(result->detail), "alloc/free churn failed");
      mori::cco::CcoMemFree(comm, buf);
      mori::cco::CcoCommDestroy(comm);
      return;
    }
    if (a != b) {
      snprintf(result->detail, sizeof(result->detail), "slot reuse: a=%p b=%p", a, b);
      mori::cco::CcoMemFree(comm, b);
      mori::cco::CcoMemFree(comm, buf);
      mori::cco::CcoCommDestroy(comm);
      return;
    }
    mori::cco::CcoMemFree(comm, b);
  }

  // Write unique pattern: byte[0] = (rank+1)*10
  std::vector<uint8_t> pattern(WINDOW_SIZE);
  for (size_t i = 0; i < WINDOW_SIZE; i++) {
    pattern[i] = static_cast<uint8_t>((rank + 1) * 10 + (i % 256));
  }
  HIP_CHECK(hipMemcpy(buf, pattern.data(), WINDOW_SIZE, hipMemcpyHostToDevice));

  // Phase 2: WindowRegister (ptr overload)
  mori::cco::CcoWindow_t win = nullptr;
  ret = mori::cco::CcoWindowRegister(comm, buf, WINDOW_SIZE, &win);
  if (ret != 0) {
    snprintf(result->detail, sizeof(result->detail), "WindowRegister failed: %d", ret);
    mori::cco::CcoMemFree(comm, buf);
    mori::cco::CcoCommDestroy(comm);
    return;
  }

  // Phase 2: WindowRegister (convenience overload)
  mori::cco::CcoWindow_t win2 = nullptr;
  void* buf2 = nullptr;
  ret = mori::cco::CcoWindowRegister(comm, WINDOW_SIZE, &win2, &buf2);
  if (ret != 0) {
    snprintf(result->detail, sizeof(result->detail), "WindowRegister(convenience) failed: %d", ret);
    mori::cco::CcoWindowDeregister(comm, win);
    mori::cco::CcoMemFree(comm, buf);
    mori::cco::CcoCommDestroy(comm);
    return;
  }
  printf("[rank %d] WindowRegister x2 OK\n", rank);

  // Phase 3: DevCommCreate with default requirements
  mori::cco::CcoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  mori::cco::CcoDevComm* devComm = nullptr;
  ret = mori::cco::CcoDevCommCreate(comm, &reqs, &devComm);
  if (ret != 0) {
    snprintf(result->detail, sizeof(result->detail), "DevCommCreate failed: %d", ret);
    mori::cco::CcoWindowDeregister(comm, win2);
    mori::cco::CcoWindowDeregister(comm, win);
    mori::cco::CcoMemFree(comm, buf);
    mori::cco::CcoCommDestroy(comm);
    return;
  }

  // Verify DevComm on GPU
  mori::cco::CcoDevComm devCommHost;
  HIP_CHECK(
      hipMemcpy(&devCommHost, devComm, sizeof(devCommHost), hipMemcpyDeviceToHost));
  if (devCommHost.rank != rank || devCommHost.worldSize != nranks) {
    snprintf(result->detail, sizeof(result->detail),
             "DevComm mismatch: rank=%d(want %d) world=%d(want %d)", devCommHost.rank, rank,
             devCommHost.worldSize, nranks);
    goto cleanup;
  }

  {
    // Verify WindowDevice on GPU — uses LSA-sized flat VA, addressed by lsaRank.
    mori::cco::CcoWindowDevice winHost;
    HIP_CHECK(hipMemcpy(&winHost, win, sizeof(winHost), hipMemcpyDeviceToHost));
    mori::cco::CcoDevComm devCommSnap;
    HIP_CHECK(hipMemcpy(&devCommSnap, devComm, sizeof(devCommSnap), hipMemcpyDeviceToHost));

    // Verify local ptr via flat addressing (uses lsaRank now).
    void* localVa =
        winHost.winBase + (static_cast<uint64_t>(winHost.lsaRank) * winHost.stride4G << 32);
    if (localVa != buf) {
      snprintf(result->detail, sizeof(result->detail), "flat localVa mismatch: %p != %p", localVa,
               buf);
      goto cleanup;
    }

    // Barrier before P2P cross-read
    mori::cco::CcoBarrierAll(comm);

    // P2P read from every LSA peer via flat addressing (intra-node only;
    // cross-node peers would need RDMA path, not exercised by this test).
    int p2pChecked = 0;
    int lsaSize = devCommSnap.lsaSize;
    int myNodeStart = devCommSnap.myNodeStart;
    for (int lsa = 0; lsa < lsaSize; lsa++) {
      if (lsa == winHost.lsaRank) continue;
      int pe = myNodeStart + lsa;
      void* peerVa =
          winHost.winBase + (static_cast<uint64_t>(lsa) * winHost.stride4G << 32);
      uint8_t got = 0;
      HIP_CHECK(hipMemcpy(&got, peerVa, 1, hipMemcpyDeviceToHost));
      uint8_t want = static_cast<uint8_t>((pe + 1) * 10);
      if (got != want) {
        snprintf(result->detail, sizeof(result->detail), "P2P read PE %d (lsa=%d): got %u want %u",
                 pe, lsa, got, want);
        goto cleanup;
      }
      p2pChecked++;
    }
    printf("[rank %d] P2P read OK from %d LSA peers\n", rank, p2pChecked);
  }

  result->passed = true;
  snprintf(result->detail, sizeof(result->detail), "all OK (%d ranks)", nranks);

cleanup:
  mori::cco::CcoDevCommDestroy(comm, devComm);
  mori::cco::CcoWindowDeregister(comm, win2);
  mori::cco::CcoWindowDeregister(comm, win);
  mori::cco::CcoMemFree(comm, buf2);
  mori::cco::CcoMemFree(comm, buf);
  mori::cco::CcoCommDestroy(comm);

  if (result->passed) printf("[rank %d] PASSED\n", rank);
}

int main(int argc, char** argv) {
  // Pre-init loggers to avoid spdlog thread-safety race
  mori::ModuleLogger::GetInstance().GetLogger(mori::modules::APPLICATION);
  mori::ModuleLogger::GetInstance().GetLogger(mori::modules::SHMEM);

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  int nranks = numDevices;
  if (argc > 1) nranks = std::min(atoi(argv[1]), numDevices);
  if (nranks < 1) {
    printf("No GPUs available.\n");
    return 1;
  }

  printf("=== CCO Host API Test (%d ranks on %d GPUs) ===\n\n", nranks, numDevices);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 18456);

  std::vector<ThreadResult> results(nranks);
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
  return fail > 0 ? 1 : 0;
}
