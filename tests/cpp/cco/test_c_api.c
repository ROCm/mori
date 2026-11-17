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

// The MORI-CCO C ABI from C, the way a runtime loader uses it: dlopen the SONAME,
// fetch the table, and drive an N-rank comm (one forked process per GPU) through
// windows over hipMemCreate memory and dev comms. Bad arguments must come back
// as error codes with a message, and the process must survive them.
//
//   test_c_api [nranks]   (default 2; rank r uses HIP device r)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "mori/cco/cco_c.h"

static moriCcoApi_t api;
static int g_rank = -1;

#define FAIL(...)                                                     \
  do {                                                                \
    fprintf(stderr, "[rank %d] %s:%d: ", g_rank, __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__);                                     \
    fprintf(stderr, "\n");                                            \
    return 1;                                                         \
  } while (0)

// A call that must succeed.
#define OK(call)                                                                       \
  do {                                                                                 \
    int rc_ = (call);                                                                  \
    if (rc_ != MORI_CCO_SUCCESS) FAIL("%s -> %d: %s", #call, rc_, api.getLastError()); \
    if (api.getLastError()[0] != '\0') FAIL("%s left last error set", #call);          \
  } while (0)

// A call that must fail with `want` and say why.
#define EXPECT_ERR(want, call)                                                                 \
  do {                                                                                         \
    int rc_ = (call);                                                                          \
    if (rc_ != (want)) FAIL("%s -> %d, want %d (%s)", #call, rc_, (want), api.getLastError()); \
    if (api.getLastError()[0] == '\0') FAIL("%s failed without a last error", #call);          \
  } while (0)

#define HIP(call)                                                         \
  do {                                                                    \
    hipError_t e_ = (call);                                               \
    if (e_ != hipSuccess) FAIL("%s -> %s", #call, hipGetErrorString(e_)); \
  } while (0)

// ---- cross-process barrier on a shared page ----

typedef struct {
  int count;
  int generation;
  int failed;
} Barrier;

static Barrier* g_bar;
static int g_nranks;

static int barrier(void) {
  int gen = __atomic_load_n(&g_bar->generation, __ATOMIC_ACQUIRE);
  if (__atomic_add_fetch(&g_bar->count, 1, __ATOMIC_ACQ_REL) == g_nranks) {
    __atomic_store_n(&g_bar->count, 0, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_bar->generation, 1, __ATOMIC_RELEASE);
    return 0;
  }
  for (int i = 0; i < 600000; i++) {  // 60 s
    if (__atomic_load_n(&g_bar->generation, __ATOMIC_ACQUIRE) != gen) return 0;
    if (__atomic_load_n(&g_bar->failed, __ATOMIC_ACQUIRE)) return -1;
    usleep(100);
  }
  return -1;
}

#define BARRIER()                                       \
  do {                                                  \
    if (barrier() != 0) FAIL("barrier: a peer failed"); \
  } while (0)

// ---- checks that need no GPU (run in the parent before fork) ----

static int test_table(moriCcoGetApiFn_t get) {
  // A newer caller's bigger table: ours is filled, the rest zeroed.
  unsigned char big[sizeof(moriCcoApi_t) + 64];
  memset(big, 0xab, sizeof(big));
  if (get(MORI_CCO_ABI_MAJOR, (moriCcoApi_t*)big, sizeof(big)) != MORI_CCO_SUCCESS)
    FAIL("bigger table refused");
  moriCcoApi_t bigApi;
  memcpy(&bigApi, big, sizeof(bigApi));
  if (bigApi.size != sizeof(moriCcoApi_t)) FAIL("table size %zu", bigApi.size);
  for (size_t i = sizeof(moriCcoApi_t); i < sizeof(big); i++)
    if (big[i] != 0) FAIL("table tail byte %zu not zeroed", i);

  EXPECT_ERR(MORI_CCO_ERROR_ABI_MISMATCH, get(MORI_CCO_ABI_MAJOR + 1, &bigApi, sizeof(bigApi)));
  EXPECT_ERR(MORI_CCO_ERROR_ABI_MISMATCH, get(0, &bigApi, sizeof(bigApi)));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, get(MORI_CCO_ABI_MAJOR, NULL, sizeof(bigApi)));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT,
             get(MORI_CCO_ABI_MAJOR, &bigApi, sizeof(moriCcoApi_t) - sizeof(void*)));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, get(MORI_CCO_ABI_MAJOR, &bigApi, sizeof(size_t)));

  int major = 0, minor = -1;
  const char* version = NULL;
  OK(api.getVersion(&major, &minor, &version));
  if (major != MORI_CCO_ABI_MAJOR || minor < 0 || version == NULL) FAIL("getVersion");
  OK(api.getVersion(NULL, NULL, NULL));
  printf("libmori_cco ABI %d.%d, build %s\n", major, minor, version);
  return 0;
}

static int test_bad_args(void) {
  moriCcoUniqueId_t uid;
  memset(&uid, 0, sizeof(uid));
  moriCcoComm_t comm = (moriCcoComm_t)(uintptr_t)0x1;
  moriCcoComm_t bogusComm = (moriCcoComm_t)(uintptr_t)0xdead0000;
  moriCcoWindow_t bogusWin = (moriCcoWindow_t)(uintptr_t)0xdead0001;
  void* ptr = NULL;

  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.getUniqueId(NULL));

  // commCreate rejects these before touching HIP or the network.
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 2, 0, NULL, NULL));
  if (comm != (moriCcoComm_t)(uintptr_t)0x1) FAIL("comm written on a NULL-out failure");
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(NULL, 2, 0, NULL, &comm));
  if (comm != NULL) FAIL("comm not cleared on failure");
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 2, 2, NULL, &comm));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 0, 0, NULL, &comm));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 2, -1, NULL, &comm));

  moriCcoCommConfig_t cfg = MORI_CCO_COMM_CONFIG_INIT;
  cfg.size = sizeof(size_t);
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 2, 0, &cfg, &comm));
  cfg.size = sizeof(cfg);
  cfg.flags = (uint64_t)1 << 40;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 2, 0, &cfg, &comm));
  cfg.flags = 0;
  cfg.bootstrap = (const struct moriCcoBootstrap*)&cfg;
  EXPECT_ERR(MORI_CCO_ERROR_NOT_SUPPORTED, api.commCreate(&uid, 2, 0, &cfg, &comm));
  cfg.bootstrap = NULL;
  cfg.reserved = 1;
  EXPECT_ERR(MORI_CCO_ERROR_NOT_SUPPORTED, api.commCreate(&uid, 2, 0, &cfg, &comm));
  cfg.reserved = 0;
  {
    // A field from a newer minor that this library does not know.
    unsigned char newer[sizeof(moriCcoCommConfig_t) + 8];
    memset(newer, 0, sizeof(newer));
    memcpy(newer, &cfg, sizeof(cfg));
    ((moriCcoCommConfig_t*)newer)->size = sizeof(newer);
    newer[sizeof(cfg)] = 1;
    EXPECT_ERR(MORI_CCO_ERROR_NOT_SUPPORTED,
               api.commCreate(&uid, 2, 0, (const moriCcoCommConfig_t*)newer, &comm));
  }

  // Handles that were never issued.
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commDestroy(NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commDestroy(bogusComm));
  moriCcoCommInfo_t info;
  info.size = sizeof(info);
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commGetInfo(bogusComm, &info));
  moriCcoWindow_t win = NULL;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT,
             api.windowImport(bogusComm, &uid, 4096, 0, &win, NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowRelease(NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowRelease(bogusWin));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowPeerPtr(bogusWin, 0, &ptr));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowPeerPtr(bogusWin, 0, NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommCreate(bogusComm, NULL, &ptr));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommDestroy(bogusComm, &uid));
  moriCcoDevCommInfo_t dinfo;
  dinfo.size = sizeof(dinfo);
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommGetInfo(bogusComm, &uid, &dinfo));

  // A success clears the previous error.
  OK(api.getVersion(NULL, NULL, NULL));
  return 0;
}

// ---- one rank ----

static int fill(void* dst, uint32_t value, size_t bytes) {
  HIP(hipMemsetD32((hipDeviceptr_t)dst, (int)value, bytes / 4));
  HIP(hipDeviceSynchronize());
  return 0;
}

// Every word of [src, src + bytes) must equal value.
static int check(const void* src, uint32_t value, size_t bytes, const char* what) {
  uint32_t* host = (uint32_t*)malloc(bytes);
  if (!host) FAIL("malloc");
  hipError_t e = hipMemcpy(host, src, bytes, hipMemcpyDeviceToHost);
  if (e != hipSuccess) {
    free(host);
    FAIL("%s: hipMemcpy -> %s", what, hipGetErrorString(e));
  }
  for (size_t i = 0; i < bytes / 4; i++) {
    if (host[i] != value) {
      uint32_t got = host[i];
      free(host);
      FAIL("%s: word %zu is %#x, want %#x", what, i, got, value);
    }
  }
  free(host);
  return 0;
}

static int run_rank(int rank, int nranks, const moriCcoUniqueId_t* uid) {
  g_rank = rank;
  HIP(hipSetDevice(rank));

  moriCcoCommConfig_t cfg = MORI_CCO_COMM_CONFIG_INIT;
  cfg.flags = MORI_CCO_COMM_LSA_ONLY | MORI_CCO_COMM_NO_CPU_AFFINITY;
  cfg.perRankVaBytes = (size_t)8 << 30;
  moriCcoComm_t comm = NULL;
  OK(api.commCreate(uid, nranks, rank, &cfg, &comm));
  if (comm == NULL) FAIL("NULL comm");
  if (hipPeekAtLastError() != hipSuccess) FAIL("commCreate left a sticky HIP error");

  moriCcoCommInfo_t info;
  info.size = sizeof(info);
  OK(api.commGetInfo(comm, &info));
  if (info.rank != rank || info.nRanks != nranks || info.lsaSize != nranks ||
      info.lsaRank != rank || info.lsaFirstRank != 0 || info.device != rank ||
      info.flags != cfg.flags || info.perRankVaBytes != cfg.perRankVaBytes)
    FAIL("commGetInfo: rank %d/%d lsa %d/%d first %d dev %d flags %#llx va %zu", info.rank,
         info.nRanks, info.lsaRank, info.lsaSize, info.lsaFirstRank, info.device,
         (unsigned long long)info.flags, info.perRankVaBytes);
  if (info.memHandleType != hipMemHandleTypePosixFileDescriptor && info.memHandleType != 8)
    FAIL("memHandleType %d", info.memHandleType);
  info.size = 4;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commGetInfo(comm, &info));
  {
    unsigned char bigger[sizeof(moriCcoCommInfo_t) + 16];
    memset(bigger, 0xab, sizeof(bigger));
    ((moriCcoCommInfo_t*)bigger)->size = sizeof(bigger);
    OK(api.commGetInfo(comm, (moriCcoCommInfo_t*)bigger));
    for (size_t i = sizeof(moriCcoCommInfo_t); i < sizeof(bigger); i++)
      if (bigger[i] != 0) FAIL("commGetInfo tail byte %zu not zeroed", i);
  }

  // A VMM allocation the way torch.symm_mem makes one, exportable as the comm needs.
  hipMemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = (hipMemAllocationHandleType)info.memHandleType;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = rank;
  size_t gran = 0;
  HIP(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
  const size_t half = gran, bytes = 2 * gran;
  size_t freeBefore = 0, total = 0;
  HIP(hipMemGetInfo(&freeBefore, &total));
  hipMemGenericAllocationHandle_t handle;
  void* buf = NULL;
  HIP(hipMemCreate(&handle, bytes, &prop, 0));
  HIP(hipMemAddressReserve(&buf, bytes, gran, NULL, 0));
  HIP(hipMemMap(buf, bytes, 0, handle, 0));
  hipMemAccessDesc access;
  memset(&access, 0, sizeof(access));
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = rank;
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP(hipMemSetAccess(buf, bytes, &access, 1));
  if (fill(buf, 0x1000u + (uint32_t)rank, bytes)) return 1;

  // Local argument errors, caught before the collective.
  moriCcoWindow_t win = NULL, win2 = NULL;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowImport(comm, buf, bytes, 1, &win, NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowImport(comm, NULL, bytes, 0, &win, NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowImport(comm, buf, 0, 0, &win, NULL));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowImport(comm, buf, bytes, 0, NULL, NULL));
  moriCcoWindowInfo_t winfo;
  winfo.size = 4;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowImport(comm, buf, bytes, 0, &win, &winfo));
  {
    // Not VMM memory: every rank fails the same local step, so nothing hangs.
    void* plain = NULL;
    HIP(hipMalloc(&plain, bytes));
    EXPECT_ERR(MORI_CCO_ERROR_INTERNAL, api.windowImport(comm, plain, bytes, 0, &win, NULL));
    if (!strstr(api.getLastError(), "hipMemRetainAllocationHandle"))
      FAIL("last error does not name the cause: %s", api.getLastError());
    (void)hipGetLastError();
    HIP(hipFree(plain));
  }

  // The whole allocation, and its second half (an import from inside it).
  winfo.size = sizeof(winfo);
  OK(api.windowImport(comm, buf, bytes, 0, &win, &winfo));
  if (win == NULL || winfo.localPtr == NULL || winfo.deviceWindow == NULL || winfo.bytes != bytes)
    FAIL("windowImport info");
  OK(api.windowImport(comm, (char*)buf + half, half, 0, &win2, NULL));
  if (hipPeekAtLastError() != hipSuccess) FAIL("windowImport left a sticky HIP error");

  void* self = NULL;
  OK(api.windowPeerPtr(win, rank, &self));
  if (self != winfo.localPtr) FAIL("self peer pointer %p != localPtr %p", self, winfo.localPtr);
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowPeerPtr(win, nranks, &self));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowPeerPtr(win, -1, &self));

  // Read every peer's buffer through its peer pointer.
  BARRIER();
  for (int p = 0; p < nranks; p++) {
    void* peer = NULL;
    OK(api.windowPeerPtr(win, p, &peer));
    if (peer == NULL) FAIL("no peer pointer for rank %d", p);
    if (check(peer, 0x1000u + (uint32_t)p, bytes, "read peer")) return 1;
    void* peer2 = NULL;
    OK(api.windowPeerPtr(win2, p, &peer2));
    if (check(peer2, 0x1000u + (uint32_t)p, half, "read peer, offset window")) return 1;
  }

  // Write into the next rank's second half; it checks through its own mapping.
  BARRIER();
  const int next = (rank + 1) % nranks, prev = (rank + nranks - 1) % nranks;
  void* nextPtr = NULL;
  OK(api.windowPeerPtr(win2, next, &nextPtr));
  if (fill(nextPtr, 0x2000u + (uint32_t)rank, half)) return 1;
  BARRIER();
  if (check((char*)buf + half, 0x2000u + (uint32_t)prev, half, "peer write, owner view")) return 1;
  if (check(buf, 0x1000u + (uint32_t)rank, half, "untouched first half")) return 1;
  if (check((char*)winfo.localPtr + half, 0x2000u + (uint32_t)prev, half, "peer write, alias"))
    return 1;

  // Dev comms.
  void* dev = NULL;
  void* dev2 = NULL;
  moriCcoDevCommRequirements_t req = MORI_CCO_DEV_COMM_REQUIREMENTS_INIT;
  req.size = 4;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommCreate(comm, &req, &dev));
  req.size = sizeof(req);
  req.lsaBarrierCount = -1;
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommCreate(comm, &req, &dev));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommCreate(comm, NULL, NULL));
  req.lsaBarrierCount = 2;
  OK(api.devCommCreate(comm, &req, &dev));
  OK(api.devCommCreate(comm, NULL, &dev2));
  if (dev == NULL || dev2 == NULL || dev == dev2) FAIL("devCommCreate pointers");
  if (hipPeekAtLastError() != hipSuccess) FAIL("devCommCreate left a sticky HIP error");
  hipPointerAttribute_t attr;
  HIP(hipPointerGetAttributes(&attr, dev));
  if (attr.type != hipMemoryTypeDevice || attr.device != rank) FAIL("devComm is not device memory");

  moriCcoDevCommInfo_t dinfo;
  dinfo.size = sizeof(dinfo);
  OK(api.devCommGetInfo(comm, dev, &dinfo));
  if (dinfo.layoutVersion != 1 || dinfo.devCommBytes < 5 * sizeof(int) ||
      dinfo.lsaBarrierCount != 2)
    FAIL("devCommGetInfo: version %u bytes %zu barriers %d", dinfo.layoutVersion,
         dinfo.devCommBytes, dinfo.lsaBarrierCount);
  {
    // ccoDevComm starts with {rank, worldSize, lsaSize, lsaRank}.
    int head[4];
    HIP(hipMemcpy(head, dev, sizeof(head), hipMemcpyDeviceToHost));
    if (head[0] != rank || head[1] != nranks || head[2] != nranks || head[3] != rank)
      FAIL("devComm header %d %d %d %d", head[0], head[1], head[2], head[3]);
  }
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommDestroy(comm, buf));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommGetInfo(comm, buf, &dinfo));
  OK(api.devCommDestroy(comm, dev));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommDestroy(comm, dev));

  // The owner lets go of its handle first, then the window (see cco_c.h, Lifetime).
  BARRIER();
  HIP(hipDeviceSynchronize());
  HIP(hipMemUnmap(buf, bytes));
  HIP(hipMemAddressFree(buf, bytes));
  HIP(hipMemRelease(handle));
  OK(api.windowRelease(win));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowRelease(win));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowPeerPtr(win, rank, &self));

  // win2 and dev2 are left for commDestroy to release.
  OK(api.commDestroy(comm));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commDestroy(comm));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commGetInfo(comm, &info));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowPeerPtr(win2, rank, &self));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.windowRelease(win2));
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommDestroy(comm, dev2));

  size_t freeAfter = 0;
  HIP(hipMemGetInfo(&freeAfter, &total));
  // Informational: other processes on the device move this number too.
  printf("[rank %d] device free memory %+lld KiB across the test (allocation %zu KiB)\n", rank,
         ((long long)freeAfter - (long long)freeBefore) / 1024, bytes / 1024);
  printf("[rank %d] PASS\n", rank);
  return 0;
}

int main(int argc, char** argv) {
  int nranks = argc > 1 ? atoi(argv[1]) : 2;
  if (nranks < 2) {
    printf("Need at least 2 ranks.\n");
    return 1;
  }
  g_nranks = nranks;
  setvbuf(stdout, NULL, _IOLBF, 0);

  void* lib = dlopen(MORI_CCO_LIBRARY_SONAME, RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "dlopen(%s): %s\n", MORI_CCO_LIBRARY_SONAME, dlerror());
    return 1;
  }
  // What a second loader in the process gets: the same library.
  void* again = dlopen(MORI_CCO_LIBRARY_SONAME, RTLD_NOW | RTLD_NOLOAD);
  if (again != lib) FAIL("RTLD_NOLOAD did not find the loaded library");
  dlclose(again);
  moriCcoGetApiFn_t get = NULL;
  *(void**)(&get) = dlvsym(lib, MORI_CCO_GET_API_SYMBOL, "MORI_CCO_1.0");
  if (!get) FAIL("dlvsym(%s@MORI_CCO_1.0): %s", MORI_CCO_GET_API_SYMBOL, dlerror());

  if (get(MORI_CCO_ABI_MAJOR, &api, sizeof(api)) != MORI_CCO_SUCCESS) FAIL("moriCcoGetApi");
  if (api.size != sizeof(api) || api.abiMajor != MORI_CCO_ABI_MAJOR) FAIL("table header");
  if (test_table(get) || test_bad_args()) return 1;
  printf("=== table and argument checks PASSED ===\n");

  moriCcoUniqueId_t uid;
  OK(api.getUniqueId(&uid));

  g_bar = (Barrier*)mmap(NULL, sizeof(Barrier), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                         -1, 0);
  if (g_bar == MAP_FAILED) FAIL("mmap");
  memset(g_bar, 0, sizeof(*g_bar));

  printf("=== C ABI test (fork, %d ranks) ===\n", nranks);
  pid_t* pids = (pid_t*)calloc((size_t)nranks, sizeof(pid_t));
  for (int r = 0; r < nranks; r++) {
    pids[r] = fork();
    if (pids[r] == 0) {
      int rc = run_rank(r, nranks, &uid);
      if (rc) __atomic_store_n(&g_bar->failed, 1, __ATOMIC_RELEASE);
      _exit(rc);
    }
  }

  // Reap with a deadline, so a hang fails the test instead of the CI job.
  int failed = 0, left = nranks;
  time_t deadline = time(NULL) + 240;
  while (left > 0) {
    int status = 0;
    pid_t pid = waitpid(-1, &status, WNOHANG);
    if (pid > 0) {
      left--;
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "child %d failed (status %d)\n", (int)pid, status);
        failed++;
      }
      continue;
    }
    if (time(NULL) > deadline) {
      fprintf(stderr, "timed out; killing %d rank(s)\n", left);
      for (int r = 0; r < nranks; r++) kill(pids[r], SIGKILL);
      while (waitpid(-1, NULL, 0) > 0) {
      }
      failed += left;
      break;
    }
    usleep(10000);
  }
  free(pids);
  printf("\n=== %d/%d PASSED ===\n", nranks - failed, nranks);
  return failed ? 1 : 0;
}
