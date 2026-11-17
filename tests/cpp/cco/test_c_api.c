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
#include <dirent.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <pthread.h>
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
  moriCcoUniqueId_t uids[4];  // rank 0's ids for the later comms
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
  cfg.flags = MORI_CCO_COMM_DEFAULT_FLAGS | ((uint64_t)1 << 40);
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.commCreate(&uid, 2, 0, &cfg, &comm));
  // ABI 1.0 has no RDMA comm.
  cfg.flags = MORI_CCO_COMM_NO_CPU_AFFINITY;
  EXPECT_ERR(MORI_CCO_ERROR_NOT_SUPPORTED, api.commCreate(&uid, 2, 0, &cfg, &comm));
  if (!strstr(api.getLastError(), "MORI_CCO_COMM_LSA_ONLY")) FAIL("%s", api.getLastError());
  cfg.flags = 0;
  EXPECT_ERR(MORI_CCO_ERROR_NOT_SUPPORTED, api.commCreate(&uid, 2, 0, &cfg, &comm));
  cfg.flags = MORI_CCO_COMM_DEFAULT_FLAGS;
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

// ---- the ABI from an atexit hook (no GPU) ----

// By now exit() has destroyed the C++ thread_locals; the error text must outlive them.
static void exit_hook(void) {
  for (int i = 0; i < 4; i++) {
    if (api.commDestroy((moriCcoComm_t)(uintptr_t)0xdead0000) != MORI_CCO_ERROR_INVALID_ARGUMENT ||
        strcmp(api.getLastError(), "commDestroy: unknown comm handle") != 0)
      _exit(3);
    if (api.windowRelease((moriCcoWindow_t)(uintptr_t)0xdead0001) !=
            MORI_CCO_ERROR_INVALID_ARGUMENT ||
        strcmp(api.getLastError(), "windowRelease: unknown or released window") != 0)
      _exit(4);
  }
  if (api.getVersion(NULL, NULL, NULL) != MORI_CCO_SUCCESS || api.getLastError()[0] != '\0')
    _exit(5);
}

static int test_exit_hook(void) {
  pid_t pid = fork();
  if (pid < 0) FAIL("fork");
  if (pid == 0) {
    // Leave error text too long for the small-string buffer on this thread.
    (void)api.commDestroy((moriCcoComm_t)(uintptr_t)0xdead0000);
    if (atexit(exit_hook) != 0) _exit(2);
    exit(0);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid) FAIL("waitpid");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    FAIL("C ABI called from an atexit hook: child status %#x", status);
  return 0;
}

// ---- VMM buffers, and what a process holds ----

typedef struct {
  hipMemGenericAllocationHandle_t handle;
  void* ptr;
  size_t bytes;
} VmmBuf;

// A VMM allocation the way torch.symm_mem makes one.
static int vmm_alloc(const hipMemAllocationProp* prop, size_t bytes, VmmBuf* b) {
  b->bytes = bytes;
  HIP(hipMemCreate(&b->handle, bytes, prop, 0));
  HIP(hipMemAddressReserve(&b->ptr, bytes, 0, NULL, 0));
  HIP(hipMemMap(b->ptr, bytes, 0, b->handle, 0));
  hipMemAccessDesc access;
  memset(&access, 0, sizeof(access));
  access.location = prop->location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP(hipMemSetAccess(b->ptr, bytes, &access, 1));
  return 0;
}

static int vmm_free(VmmBuf* b) {
  HIP(hipMemUnmap(b->ptr, b->bytes));
  HIP(hipMemAddressFree(b->ptr, b->bytes));
  HIP(hipMemRelease(b->handle));
  return 0;
}

// This process's VRAM in KiB from its amdgpu DRM fdinfo, which other users of
// the GPU do not move; -1 if the kernel does not report it.
static long long vram_kib(void) {
  DIR* dir = opendir("/proc/self/fdinfo");
  if (!dir) return -1;
  long long total = -1;
  unsigned long long seen[64];
  int nseen = 0;
  struct dirent* e;
  while ((e = readdir(dir)) != NULL) {
    char path[300], line[256], unit[8];
    snprintf(path, sizeof(path), "/proc/self/fdinfo/%s", e->d_name);
    FILE* f = e->d_name[0] == '.' ? NULL : fopen(path, "r");
    if (!f) continue;
    int amdgpu = 0, have = 0, dup = 0;
    unsigned long long client = 0, n = 0, bytes = 0;
    while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "drm-driver:", 11) == 0) amdgpu = strstr(line, "amdgpu") != NULL;
      (void)sscanf(line, "drm-client-id: %llu", &client);
      unit[0] = '\0';
      if (sscanf(line, "drm-memory-vram: %llu %7s", &n, unit) >= 1) {
        have = 1;
        bytes = n << (strcmp(unit, "KiB") == 0 ? 10 : strcmp(unit, "MiB") == 0 ? 20 : 0);
      }
    }
    fclose(f);
    if (!amdgpu || !have) continue;
    for (int i = 0; i < nseen; i++) dup |= seen[i] == client;
    if (dup) continue;
    if (nseen < 64) seen[nseen++] = client;
    total = (total < 0 ? 0 : total) + (long long)(bytes >> 10);
  }
  closedir(dir);
  return total;
}

static int open_fds(void) {
  DIR* dir = opendir("/proc/self/fd");
  int n = 0;
  if (!dir) return -1;
  while (readdir(dir) != NULL) n++;
  closedir(dir);
  return n;
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

// Whether this process has an RDMA library mapped.
static int rdma_loaded(void) {
  FILE* f = fopen("/proc/self/maps", "r");
  char line[512];
  int found = 0;
  if (!f) return -1;
  while (!found && fgets(line, sizeof(line), f)) found = strstr(line, "libibverbs") != NULL;
  fclose(f);
  return found;
}

// Comms with a NULL config and with pinning asked for. Neither maps an RDMA library.
static int default_comms(int rank, int nranks) {
  if (rank == 0) {
    OK(api.getUniqueId(&g_bar->uids[0]));
    OK(api.getUniqueId(&g_bar->uids[1]));
  }
  BARRIER();
  cpu_set_t before, after;
  if (sched_getaffinity(0, sizeof(before), &before) != 0) FAIL("sched_getaffinity");
  moriCcoComm_t comm = NULL;
  OK(api.commCreate(&g_bar->uids[0], nranks, rank, NULL, &comm));
  if (sched_getaffinity(0, sizeof(after), &after) != 0) FAIL("sched_getaffinity");
  if (!CPU_EQUAL(&before, &after))
    FAIL("NULL config: %d CPUs became %d", CPU_COUNT(&before), CPU_COUNT(&after));
  moriCcoCommInfo_t info;
  info.size = sizeof(info);
  OK(api.commGetInfo(comm, &info));
  if (info.flags != MORI_CCO_COMM_DEFAULT_FLAGS || info.nRanks != nranks)
    FAIL("NULL-config comm: flags %#llx", (unsigned long long)info.flags);
  OK(api.commDestroy(comm));

  // Pins this thread unless MORI_IGNORE_CPU_AFFINITY is set, which shows the
  // affinity checks have teeth.
  moriCcoCommConfig_t cfg = MORI_CCO_COMM_CONFIG_INIT;
  cfg.flags = MORI_CCO_COMM_LSA_ONLY;
  OK(api.commCreate(&g_bar->uids[1], nranks, rank, &cfg, &comm));
  if (sched_getaffinity(0, sizeof(after), &after) != 0) FAIL("sched_getaffinity");
  printf("[rank %d] pinning config: %d -> %d CPUs\n", rank, CPU_COUNT(&before), CPU_COUNT(&after));
  OK(api.commDestroy(comm));
  if (rdma_loaded() != 0) FAIL("an LSA-only comm mapped libibverbs");
  return 0;
}

// Import/release cycles in both teardown orders. A window holds its own
// reference, so neither order may leave VRAM or fds behind in this process.
static int teardown_orders(moriCcoComm_t comm, const hipMemAllocationProp* prop, size_t bytes) {
  enum { kCycles = 8 };
  const long long kib = (long long)(bytes >> 10);
  for (int windowFirst = 0; windowFirst <= 1; windowFirst++) {
    const char* order = windowFirst ? "window first" : "owner first";
    BARRIER();
    const long long vram0 = vram_kib();
    const int fds0 = open_fds();
    long long seen = 0;  // what one allocation adds, to know fdinfo counts it
    for (int i = 0; i < kCycles; i++) {
      VmmBuf b;
      moriCcoWindow_t w = NULL;
      if (vmm_alloc(prop, bytes, &b)) return 1;
      if (i == 0 && vram0 >= 0) seen = vram_kib() - vram0;
      OK(api.windowImport(comm, b.ptr, bytes, 0, &w, NULL));
      if (windowFirst) OK(api.windowRelease(w));
      if (vmm_free(&b)) return 1;
      if (!windowFirst) OK(api.windowRelease(w));
    }
    BARRIER();  // peers have released their imports of this rank's buffers
    const long long grew = vram_kib() - vram0;
    const int fdsLeft = open_fds() - fds0;
    printf("[rank %d] %s x%d: VRAM %+lld KiB (one allocation shows as %lld), fds %+d\n", g_rank,
           order, kCycles, grew, seen, fdsLeft);
    if (fdsLeft > 0) FAIL("%s: %d fds left open", order, fdsLeft);
    if (seen >= kib && grew >= kib) FAIL("%s: %lld KiB of VRAM not returned", order, grew);
  }
  return 0;
}

// Ranks release different windows, so their next imports land at different
// slot offsets; the import's rendezvous must still meet.
static int divergent_release(moriCcoComm_t comm, const hipMemAllocationProp* prop, size_t bytes) {
  VmmBuf b[3];
  moriCcoWindow_t w[3];
  for (int i = 0; i < 2; i++) {
    if (vmm_alloc(prop, bytes, &b[i])) return 1;
    OK(api.windowImport(comm, b[i].ptr, bytes, 0, &w[i], NULL));
  }
  const int gone = g_rank % 2;
  OK(api.windowRelease(w[gone]));
  if (vmm_free(&b[gone])) return 1;
  if (vmm_alloc(prop, bytes, &b[2])) return 1;
  if (fill(b[2].ptr, 0x3000u + (uint32_t)g_rank, bytes)) return 1;
  OK(api.windowImport(comm, b[2].ptr, bytes, 0, &w[2], NULL));
  for (int p = 0; p < g_nranks; p++) {
    void* peer = NULL;
    OK(api.windowPeerPtr(w[2], p, &peer));
    if (check(peer, 0x3000u + (uint32_t)p, bytes, "after divergent release")) return 1;
  }
  BARRIER();
  for (int i = 0; i < 3; i++) {
    if (i == gone) continue;
    OK(api.windowRelease(w[i]));
    if (vmm_free(&b[i])) return 1;
  }
  return 0;
}

// Two live comms that share rank 0, importing windows at the same time from two
// threads of each rank. Their rendezvous must not cross.
typedef struct {
  int rank, idx, rc;
  hipMemAllocationProp prop;
  size_t bytes;
} PairArg;

static int pair_comm(PairArg* a) {
  HIP(hipSetDevice(a->rank));
  moriCcoCommConfig_t cfg = MORI_CCO_COMM_CONFIG_INIT;
  cfg.perRankVaBytes = (size_t)4 << 30;
  moriCcoComm_t comm = NULL;
  OK(api.commCreate(&g_bar->uids[2 + a->idx], g_nranks, a->rank, &cfg, &comm));
  for (int i = 0; i < 16; i++) {
    const uint32_t tag = 0x40000000u | ((uint32_t)a->idx << 16) | ((uint32_t)i << 8);
    VmmBuf b;
    moriCcoWindow_t w = NULL;
    if (vmm_alloc(&a->prop, a->bytes, &b)) return 1;
    if (fill(b.ptr, tag | (uint32_t)a->rank, a->bytes)) return 1;
    OK(api.windowImport(comm, b.ptr, a->bytes, 0, &w, NULL));
    for (int p = 0; p < g_nranks; p++) {
      void* peer = NULL;
      OK(api.windowPeerPtr(w, p, &peer));
      if (check(peer, tag | (uint32_t)p, a->bytes, "comm pair")) return 1;
    }
    OK(api.windowRelease(w));
    if (vmm_free(&b)) return 1;
  }
  OK(api.commDestroy(comm));
  return 0;
}

static void* pair_thread(void* p) {
  PairArg* a = (PairArg*)p;
  a->rc = pair_comm(a);
  return NULL;
}

static int comm_pair(int rank, const hipMemAllocationProp* prop, size_t bytes) {
  if (rank == 0) {
    OK(api.getUniqueId(&g_bar->uids[2]));
    OK(api.getUniqueId(&g_bar->uids[3]));
  }
  BARRIER();
  PairArg args[2];
  pthread_t threads[2];
  for (int t = 0; t < 2; t++) {
    args[t].rank = rank;
    args[t].idx = t;
    args[t].rc = 1;
    args[t].prop = *prop;
    args[t].bytes = bytes;
    if (pthread_create(&threads[t], NULL, pair_thread, &args[t]) != 0) FAIL("pthread_create");
  }
  for (int t = 0; t < 2; t++) pthread_join(threads[t], NULL);
  if (args[0].rc || args[1].rc) FAIL("comm pair: thread rc %d %d", args[0].rc, args[1].rc);
  return 0;
}

static int run_rank(int rank, int nranks, const moriCcoUniqueId_t* uid) {
  g_rank = rank;
  HIP(hipSetDevice(rank));
  cpu_set_t cpusBefore, cpusAfter;
  if (sched_getaffinity(0, sizeof(cpusBefore), &cpusBefore) != 0) FAIL("sched_getaffinity");

  moriCcoCommConfig_t cfg = MORI_CCO_COMM_CONFIG_INIT;
  cfg.flags = MORI_CCO_COMM_LSA_ONLY | MORI_CCO_COMM_NO_CPU_AFFINITY;
  cfg.perRankVaBytes = (size_t)8 << 30;
  moriCcoComm_t comm = NULL;
  OK(api.commCreate(uid, nranks, rank, &cfg, &comm));
  if (comm == NULL) FAIL("NULL comm");
  if (hipPeekAtLastError() != hipSuccess) FAIL("commCreate left a sticky HIP error");
  if (sched_getaffinity(0, sizeof(cpusAfter), &cpusAfter) != 0) FAIL("sched_getaffinity");
  if (!CPU_EQUAL(&cpusBefore, &cpusAfter))
    FAIL("NO_CPU_AFFINITY: %d CPUs became %d", CPU_COUNT(&cpusBefore), CPU_COUNT(&cpusAfter));
  if (rdma_loaded() != 0) FAIL("LSA_ONLY comm mapped libibverbs");

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
    if (hipPeekAtLastError() != hipSuccess) FAIL("failed windowImport left a HIP error pending");
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
  // A HIP error the caller has pending stays for the caller.
  if (hipSetDevice(-1) == hipSuccess) FAIL("hipSetDevice(-1) succeeded");
  OK(api.devCommDestroy(comm, dev));
  if (hipGetLastError() != hipErrorInvalidDevice)
    FAIL("devCommDestroy took the caller's HIP error");
  EXPECT_ERR(MORI_CCO_ERROR_INVALID_ARGUMENT, api.devCommDestroy(comm, dev));

  if (teardown_orders(comm, &prop, bytes) || divergent_release(comm, &prop, bytes)) return 1;

  // The owner lets go of its handle first, then the window.
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

  if (default_comms(rank, nranks) || comm_pair(rank, &prop, bytes)) return 1;

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
  if (test_table(get) || test_bad_args() || test_exit_hook()) return 1;
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
