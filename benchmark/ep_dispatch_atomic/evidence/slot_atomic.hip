// Minimal reproduction of the EPv2 gfx1250 dispatch SlotReserve cost, all 4 GPUs at once.
//
// What the kernel does (ep_intranode_1250x.hpp, the args.offTokOff RMW), per block:
//
//   for (int p = thdId; p < npes; p += blockDim.x) {          // lanes 0..npes-1 of warp 0
//     index_t n = s_N[p];
//     if (n > 0) {
//       s_base[p] = __hip_atomic_fetch_add(EpPeer<index_t>(win, p, args.offTokOff), n,
//                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
//       atomicAdd(&args.destPeTokenCounter[p], n);             // local, AGENT scope
//     }
//   }
//   __syncthreads();
//
// Every block of every rank runs it, so each PE's ONE tokOff word takes
// gridDim.x * npes returning SYSTEM-scope RMWs per dispatch: 64 x 4 = 256 at the
// tuned default.
//
// Here one process drives all 4 GPUs with peer access on. Every iteration, an
// in-kernel barrier (all blocks of a GPU, then all GPUs) releases every block of
// every GPU into the region together -- the way ranks leave the previous combine's
// barrier -- and each block times region start -> after the __syncthreads with
// wall_clock64, i.e. exactly the SlotReserve span in the profiler trace.
//
// Modes (the timed region only; everything else is identical):
//   none    only the __syncthreads: the floor
//   stock   the kernel's code above
//   varA    variant A: only the local AGENT add; its return is the in-segment offset
//   spread  stock, but every (source GPU, block) owns its own 128 B-padded word on
//           each PE: the same remote SYSTEM round trip, no two RMWs on one address
// stock vs spread separates same-word contention from remote latency.
//
// Two switches for what the real kernel has and an isolated loop does not:
//   memory  ext: hipExtMallocWithFlags(hipDeviceMallocUncached), the mori-shmem heap
//           vmm: hipMemCreate(Uncached, gpuDirectRDMACapable) + hipMemSetAccess for all
//                GPUs -- how the cco window the op binds is allocated (cco_init.cpp)
//   load    a second kernel on every GPU streams 16 B stores into the peers' uncached
//           buffers for the whole run, the way dispatch's payload and combine's
//           gather keep the fabric busy while some blocks are still reserving
// destPeTokenCounter is plain hipMalloc in every case (a torch.zeros tensor in
// hip_backend.py).
//
//   hipcc -O3 --offload-arch=gfx1250 -std=c++17 slot_atomic.hip -o slot_atomic
//   ./slot_atomic [iters=300] [warm=50] [attr]   # attr: only the memory-attribute matrix
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                       \
  do {                                                                                     \
    hipError_t _e = (x);                                                                   \
    if (_e != hipSuccess) {                                                                \
      fprintf(stderr, "HIP error: %s (line %d)\n", hipGetErrorString(_e), __LINE__);      \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

constexpr int MAXG = 8;
constexpr int MAXBLK = 256;
constexpr int PAD = 32;                              // ints: 128 B between spread words
constexpr size_t SPREAD_OFF = PAD;                   // word 0 is the shared allocator
constexpr size_t TOK_INTS = SPREAD_OFF + (size_t)MAXG * MAXBLK * PAD;
constexpr unsigned long long SPIN_TIMEOUT = 500000000ull;  // ~5 s of wall_clock64
constexpr size_t LOAD_VEC = (size_t)1 << 20;         // uint4 per source slice: 16 MB
constexpr int LOAD_BLOCKS = 64;

enum Mode { NONE = 0, STOCK = 1, VARA = 2, SPREAD = 3 };
static const char* kModeName[] = {"none", "stock", "varA", "spread"};

struct Peers {
  int* tok[MAXG];                   // PE g's tokOff region
  unsigned long long* xflag[MAXG];  // PE g's cross-GPU barrier slots (uncached)
  uint4* load[MAXG];                // PE g's landing buffer for the background traffic
};

// Spin until *p >= want, or give up after ~5 s and flag it: a broken run must not
// wedge a shared GPU.
template <int SCOPE, typename T>
__device__ __forceinline__ bool SpinGe(T* p, T want, int* err) {
  const unsigned long long t0 = wall_clock64();
  while (__hip_atomic_load(p, __ATOMIC_RELAXED, SCOPE) < want) {
    if (wall_clock64() - t0 > SPIN_TIMEOUT) {
      atomicExch(err, 1);
      return false;
    }
  }
  return true;
}

template <int MODE>
__global__ __launch_bounds__(256) void k_slot(Peers P, int me, int nact, int npes, int iters,
                                             int n, int* destCnt, unsigned* arrive,
                                             unsigned* go, unsigned long long* tout,
                                             int* sink, int* err, unsigned* stop) {
  const int t = threadIdx.x;
  __shared__ int s_base[MAXG];
  for (int it = 0; it < iters; ++it) {
    const unsigned phase = (unsigned)it + 1;
    // ---- release: every block of this GPU, then every active GPU, then go ----
    __syncthreads();
    if (t == 0) {
      __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (blockIdx.x == 0) {
        SpinGe<__HIP_MEMORY_SCOPE_AGENT>(arrive, phase * gridDim.x, err);
        for (int g = 0; g < nact; ++g)
          __hip_atomic_store(P.xflag[g] + me, (unsigned long long)phase, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_SYSTEM);
        for (int g = 0; g < nact; ++g)
          SpinGe<__HIP_MEMORY_SCOPE_SYSTEM>(P.xflag[me] + g, (unsigned long long)phase, err);
        __hip_atomic_store(go, phase, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      SpinGe<__HIP_MEMORY_SCOPE_AGENT>(go, phase, err);
    }
    __syncthreads();

    // ---- the timed region: SlotReserve ----
    const unsigned long long t0 = wall_clock64();
    if (MODE != NONE) {
      for (int p = t; p < npes; p += blockDim.x) {
        if (MODE == STOCK) {
          s_base[p] = __hip_atomic_fetch_add(P.tok[p], n, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_SYSTEM);
          atomicAdd(&destCnt[p], n);
        } else if (MODE == VARA) {
          s_base[p] = atomicAdd(&destCnt[p], n);
        } else {  // SPREAD
          int* w = P.tok[p] + SPREAD_OFF + ((size_t)me * MAXBLK + blockIdx.x) * PAD;
          s_base[p] =
              __hip_atomic_fetch_add(w, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          atomicAdd(&destCnt[p], n);
        }
      }
    }
    __syncthreads();
    const unsigned long long t1 = wall_clock64();

    if (t == 0) {
      tout[((size_t)it * gridDim.x + blockIdx.x) * 2 + 0] = t0;
      tout[((size_t)it * gridDim.x + blockIdx.x) * 2 + 1] = t1;
    }
    // Consume the returned base, as the kernel does, so every RMW stays a returning one.
    if (MODE != NONE && t < npes) sink[blockIdx.x * MAXG + t] ^= s_base[t];
  }
  if (blockIdx.x == 0 && t == 0)
    __hip_atomic_store(stop, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Background fabric traffic: each block streams 16 B stores into one peer's buffer
// (this GPU's own slice of it) until the measured kernel says stop.
__global__ __launch_bounds__(256) void k_load(Peers P, int me, int npes, const unsigned* stop,
                                             unsigned long long* bytes) {
  const int dst = (me + 1 + (int)blockIdx.x % (npes - 1)) % npes;
  uint4* buf = P.load[dst] + (size_t)me * LOAD_VEC;
  const uint4 v = make_uint4(blockIdx.x, threadIdx.x, me, 0);
  size_t i = (size_t)(blockIdx.x / (npes - 1)) * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)((gridDim.x + npes - 2) / (npes - 1)) * blockDim.x;
  unsigned long long cnt = 0;
  const unsigned long long t0 = wall_clock64();
  for (;;) {
    for (int k = 0; k < 64; ++k) {
      buf[i] = v;
      i += stride;
      if (i >= LOAD_VEC) i -= LOAD_VEC;
    }
    cnt += 64;
    if (__hip_atomic_load(stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) break;
    if (wall_clock64() - t0 > SPIN_TIMEOUT) break;
  }
  atomicAdd(bytes, cnt * 16ull);
}

struct Stats {
  double mean, p50, p95, max;
};
static Stats Summ(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double s = 0;
  for (double x : v) s += x;
  return {s / v.size(), v[v.size() / 2], v[(size_t)(0.95 * (v.size() - 1))], v.back()};
}

struct Dev {
  int* tok[8];  // one tokOff region per memory kind (kMem)
  unsigned long long* xflag;
  uint4* load;
  int* destCnt;
  unsigned* arrive;
  unsigned* go;
  unsigned* stop;
  unsigned long long* tout;
  unsigned long long* bytes;
  int* sink;
  int* err;
  hipStream_t st, st2;
  double mhz;
};

static int g_npes, g_iters, g_warm;
static Dev g_dev[MAXG];
static Peers g_peers;
// Where the tokOff words live. 0 is the mori-shmem heap's allocation; the rest are
// hipMemCreate the way ccoMemAlloc does it, one per setting of its two env vars:
//   CCO_UNCACHED_WINDOW=0 -> hipMemAllocationTypePinned instead of Uncached
//   CCO_GDR_CAPABLE=0     -> allocFlags.gpuDirectRDMACapable = 0
struct MemKind {
  const char* name;
  bool vmm;
  hipMemAllocationType type;
  int gdr;
};
static const MemKind kMem[] = {
    {"ext   hipExtMallocWithFlags(Uncached)       [shmem heap]", false,
     hipMemAllocationTypePinned, 0},
    {"vmm   Uncached + gpuDirectRDMA              [cco default]", true,
     static_cast<hipMemAllocationType>(0x40000000), 1},
    {"vmm   Uncached, no gpuDirectRDMA            [CCO_GDR_CAPABLE=0]", true,
     static_cast<hipMemAllocationType>(0x40000000), 0},
    {"vmm   Pinned + gpuDirectRDMA                [CCO_UNCACHED_WINDOW=0]", true,
     hipMemAllocationTypePinned, 1},
    {"vmm   Pinned, no gpuDirectRDMA              [both =0]", true, hipMemAllocationTypePinned,
     0},
};
constexpr int NMEM = sizeof(kMem) / sizeof(kMem[0]);
static bool g_memOk[NMEM];

// hipMemCreate + one mapping made accessible to every GPU, as ccoMemAlloc does
// (the raw 0x40000000 is hipMemAllocationTypeUncached, as cco_init.cpp spells it
// when the symbol is missing). No fallback: a refused combination is reported and
// its rows are skipped, never silently measured as something else.
static int* VmmAlloc(int dev, size_t bytes, const MemKind& k) {
  hipMemAllocationProp prop = {};
  prop.type = k.type;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  prop.allocFlags.gpuDirectRDMACapable = k.gdr;
  size_t gran = 0;
  hipError_t e = hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum);
  if (e != hipSuccess) {
    printf("  [%s] granularity query failed on GPU %d: %s\n", k.name, dev, hipGetErrorString(e));
    (void)hipGetLastError();
    return nullptr;
  }
  const size_t sz = (bytes + gran - 1) / gran * gran;
  hipMemGenericAllocationHandle_t h;
  e = hipMemCreate(&h, sz, &prop, 0);
  if (e != hipSuccess) {
    printf("  [%s] hipMemCreate refused on GPU %d: %s\n", k.name, dev, hipGetErrorString(e));
    (void)hipGetLastError();
    return nullptr;
  }
  void* va = nullptr;
  HIP_CHECK(hipMemAddressReserve(&va, sz, gran, nullptr, 0));
  HIP_CHECK(hipMemMap(va, sz, 0, h, 0));
  std::vector<hipMemAccessDesc> acc(g_npes);
  for (int g = 0; g < g_npes; ++g) {
    acc[g].location.type = hipMemLocationTypeDevice;
    acc[g].location.id = g;
    acc[g].flags = hipMemAccessFlagsProtReadWrite;
  }
  e = hipMemSetAccess(va, sz, acc.data(), g_npes);
  if (e != hipSuccess) {
    printf("  [%s] hipMemSetAccess refused on GPU %d: %s\n", k.name, dev, hipGetErrorString(e));
    (void)hipGetLastError();
    return nullptr;
  }
  return static_cast<int*>(va);
}

static void Run(int mode, int nblk, int nact, int mem, bool load) {
  const int n = 7;  // ~512 tok/rank over 64 blocks, deduped per destination
  Peers P = g_peers;
  for (int d = 0; d < g_npes; ++d) P.tok[d] = g_dev[d].tok[mem];
  for (int d = 0; d < g_npes; ++d) {
    Dev& D = g_dev[d];
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipMemset(P.tok[d], 0, TOK_INTS * sizeof(int)));
    HIP_CHECK(hipMemset(D.xflag, 0, MAXG * sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(D.destCnt, 0, MAXG * sizeof(int)));
    HIP_CHECK(hipMemset(D.arrive, 0, sizeof(unsigned)));
    HIP_CHECK(hipMemset(D.go, 0, sizeof(unsigned)));
    HIP_CHECK(hipMemset(D.stop, 0, sizeof(unsigned)));
    HIP_CHECK(hipMemset(D.bytes, 0, sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(D.err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());
  }
  // The measured kernel first, so its blocks are resident before the load takes CUs.
  for (int d = 0; d < nact; ++d) {
    Dev& D = g_dev[d];
    HIP_CHECK(hipSetDevice(d));
    auto k = mode == NONE    ? k_slot<NONE>
             : mode == STOCK ? k_slot<STOCK>
             : mode == VARA  ? k_slot<VARA>
                             : k_slot<SPREAD>;
    hipLaunchKernelGGL(k, dim3(nblk), dim3(256), 0, D.st, P, d, nact, g_npes, g_iters, n,
                       D.destCnt, D.arrive, D.go, D.tout, D.sink, D.err, D.stop);
    HIP_CHECK(hipGetLastError());
  }
  if (load)
    for (int d = 0; d < g_npes; ++d) {
      Dev& D = g_dev[d];
      HIP_CHECK(hipSetDevice(d));
      hipLaunchKernelGGL(k_load, dim3(LOAD_BLOCKS), dim3(256), 0, D.st2, P, d, g_npes,
                         D.stop, D.bytes);
      HIP_CHECK(hipGetLastError());
    }
  for (int d = 0; d < g_npes; ++d) {
    HIP_CHECK(hipSetDevice(d));
    if (d < nact) HIP_CHECK(hipStreamSynchronize(g_dev[d].st));
    // An inactive GPU never sets its stop flag; release its load kernel by hand.
    if (d >= nact && load) HIP_CHECK(hipMemset(g_dev[d].stop, 0xff, sizeof(unsigned)));
    HIP_CHECK(hipStreamSynchronize(g_dev[d].st2));
  }

  // ---- per-block durations, per-GPU grid span and start skew, pooled over GPUs ----
  std::vector<double> blk, span, skew;
  int errs = 0;
  double loadGBs = 0;
  for (int d = 0; d < nact; ++d) {
    Dev& D = g_dev[d];
    HIP_CHECK(hipSetDevice(d));
    std::vector<unsigned long long> h((size_t)g_iters * nblk * 2);
    HIP_CHECK(hipMemcpy(h.data(), D.tout, h.size() * 8, hipMemcpyDeviceToHost));
    int e = 0;
    HIP_CHECK(hipMemcpy(&e, D.err, sizeof(int), hipMemcpyDeviceToHost));
    errs += e;
    unsigned long long first = ~0ull, last = 0;
    for (int it = 0; it < g_iters; ++it) {
      unsigned long long lo0 = ~0ull, hi0 = 0, hi1 = 0;
      for (int b = 0; b < nblk; ++b) {
        unsigned long long t0 = h[((size_t)it * nblk + b) * 2], t1 = h[((size_t)it * nblk + b) * 2 + 1];
        lo0 = std::min(lo0, t0);
        hi0 = std::max(hi0, t0);
        hi1 = std::max(hi1, t1);
        if (it >= g_warm) blk.push_back((t1 - t0) / D.mhz);
      }
      first = std::min(first, lo0);
      last = std::max(last, hi1);
      if (it >= g_warm) {
        span.push_back((hi1 - lo0) / D.mhz);
        skew.push_back((hi0 - lo0) / D.mhz);
      }
    }
    if (load) {
      unsigned long long b = 0;
      HIP_CHECK(hipMemcpy(&b, D.bytes, sizeof(b), hipMemcpyDeviceToHost));
      // bytes over the measured kernel's lifetime: approximate, the load runs a
      // little longer than the window
      loadGBs += b / ((last - first) / D.mhz * 1e3) / nact;
    }
  }

  // ---- every RMW landed: the words must hold exactly what was added ----
  const long long per = (long long)g_iters * n;
  long long bad = 0;
  for (int d = 0; d < g_npes; ++d) {
    HIP_CHECK(hipSetDevice(d));
    std::vector<int> tok(TOK_INTS), dc(MAXG);
    HIP_CHECK(hipMemcpy(tok.data(), P.tok[d], TOK_INTS * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dc.data(), g_dev[d].destCnt, MAXG * sizeof(int), hipMemcpyDeviceToHost));
    if (mode == STOCK) bad += tok[0] != per * nact * nblk;
    if (mode == SPREAD)
      for (int g = 0; g < nact; ++g)
        for (int b = 0; b < nblk; ++b)
          bad += tok[SPREAD_OFF + ((size_t)g * MAXBLK + b) * PAD] != per;
    if (mode != NONE && d < nact)
      for (int p = 0; p < g_npes; ++p) bad += dc[p] != per * nblk;
  }

  Stats a = Summ(blk), s = Summ(span), k = Summ(skew);
  const int rmw = mode == STOCK ? nact * nblk : (mode == SPREAD ? 1 : 0);
  printf("%-6s %4d %6d %8d | %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f | %5.2f | %s%s",
         kModeName[mode], nact, nblk, rmw, a.mean, a.p50, a.p95, a.max, s.mean, s.p50, s.p95,
         k.mean, bad ? "COUNT MISMATCH" : "ok", errs ? " TIMEOUT" : "");
  if (load) printf("  load ~%.0f GB/s/GPU", loadGBs);
  printf("\n");
  fflush(stdout);
}

int main(int argc, char** argv) {
  g_iters = argc > 1 ? atoi(argv[1]) : 300;
  g_warm = argc > 2 ? atoi(argv[2]) : 50;
  // "attr" runs only section 8, the memory-attribute matrix
  const bool attrOnly = argc > 3 && std::string(argv[3]) == "attr";
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  g_npes = std::min(ndev, 4);
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device %s (%s), %d CUs, %d GPUs used, iters %d (first %d dropped)\n", prop.name,
         prop.gcnArchName, prop.multiProcessorCount, g_npes, g_iters, g_warm);

  for (int a = 0; a < g_npes; ++a)
    for (int b = 0; b < g_npes; ++b) {
      if (a == b) continue;
      int can = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can, a, b));
      if (!can) {
        printf("no peer access %d -> %d\n", a, b);
        return 1;
      }
      HIP_CHECK(hipSetDevice(a));
      hipError_t e = hipDeviceEnablePeerAccess(b, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
    }

  for (int d = 0; d < g_npes; ++d) {
    Dev& D = g_dev[d];
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipExtMallocWithFlags((void**)&D.tok[0], TOK_INTS * sizeof(int),
                                    hipDeviceMallocUncached));
    for (int k = 1; k < NMEM; ++k) D.tok[k] = VmmAlloc(d, TOK_INTS * sizeof(int), kMem[k]);
    HIP_CHECK(hipExtMallocWithFlags((void**)&D.xflag, MAXG * sizeof(unsigned long long),
                                    hipDeviceMallocUncached));
    HIP_CHECK(hipExtMallocWithFlags((void**)&D.load, (size_t)g_npes * LOAD_VEC * sizeof(uint4),
                                    hipDeviceMallocUncached));
    HIP_CHECK(hipMalloc(&D.destCnt, MAXG * sizeof(int)));
    HIP_CHECK(hipMalloc(&D.arrive, sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&D.go, sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&D.stop, sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&D.bytes, sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&D.tout, (size_t)g_iters * MAXBLK * 2 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&D.sink, MAXBLK * MAXG * sizeof(int)));
    HIP_CHECK(hipMalloc(&D.err, sizeof(int)));
    HIP_CHECK(hipStreamCreateWithFlags(&D.st, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&D.st2, hipStreamNonBlocking));
    int khz = 0;
    HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, d));
    D.mhz = khz / 1000.0;
    g_peers.xflag[d] = D.xflag;
    g_peers.load[d] = D.load;
  }
  for (int k = 0; k < NMEM; ++k) {
    g_memOk[k] = true;
    for (int d = 0; d < g_npes; ++d) g_memOk[k] = g_memOk[k] && g_dev[d].tok[k] != nullptr;
  }
  printf("wall_clock64 rate %.3f MHz (hipDeviceAttributeWallClockRate)\n", g_dev[0].mhz);
  for (int k = 0; k < NMEM; ++k)
    printf("  mem %d: %s  %s\n", k, kMem[k].name, g_memOk[k] ? "allocated" : "UNAVAILABLE");
  printf("\n");

  const char* hdr =
      "mode   gpus blocks rmw/word |  per-block SlotReserve us   | grid span us (last block)| skew  |\n"
      "                            |  mean    p50    p95    max  |  mean    p50    p95     |  us   |\n"
      "----------------------------+-----------------------------+--------------------------+-------+";
  if (!attrOnly) {
  printf("== 1. the real shape: 4 GPUs x 64 blocks x 256 threads, ext memory, idle fabric ==\n%s\n",
         hdr);
  for (int m : {NONE, STOCK, VARA, SPREAD}) Run(m, 64, g_npes, 0, false);

  printf("\n== 2. contenders per word: stock, 64 blocks, 1/2/4 GPUs running ==\n%s\n", hdr);
  for (int g : {1, 2, 4})
    if (g <= g_npes) Run(STOCK, 64, g, 0, false);

  printf("\n== 3. block sweep, 4 GPUs: stock vs spread vs varA ==\n%s\n", hdr);
  for (int b : {8, 16, 32, 64, 128})
    for (int m : {STOCK, SPREAD, VARA}) Run(m, b, g_npes, 0, false);

  printf("\n== 4. the cco window's memory (hipMemCreate Uncached + gpuDirectRDMA), idle fabric ==\n%s\n",
         hdr);
  for (int m : {NONE, STOCK, VARA, SPREAD}) Run(m, 64, g_npes, 1, false);

  printf("\n== 5. ext memory, fabric busy (%d blocks/GPU streaming 16 B stores to peers) ==\n%s\n",
         LOAD_BLOCKS, hdr);
  for (int m : {NONE, STOCK, VARA, SPREAD}) Run(m, 64, g_npes, 0, true);

  printf("\n== 6. cco window memory AND fabric busy ==\n%s\n", hdr);
  for (int m : {NONE, STOCK, VARA, SPREAD}) Run(m, 64, g_npes, 1, true);

  printf("\n== 7. contenders per word under load: stock, vmm, 64 blocks ==\n%s\n", hdr);
  for (int g : {1, 2, 4})
    if (g <= g_npes) Run(STOCK, 64, g, 1, true);

  }

  printf("\n== 8. memory attributes: which hipMemCreate setting makes same-word RMWs slow ==\n"
         "   stock = 256 RMWs on ONE int per GPU; spread = same remote RMW, own int per block\n");
  for (bool load : {false, true}) {
    printf("\n-- fabric %s --\n%s\n", load ? "BUSY" : "idle", hdr);
    Run(VARA, 64, g_npes, 0, load);
    for (int k = 0; k < NMEM; ++k) {
      printf("   mem %d: %s\n", k, kMem[k].name);
      if (!g_memOk[k]) {
        printf("   (unavailable, skipped)\n");
        continue;
      }
      Run(STOCK, 64, g_npes, k, load);
      Run(SPREAD, 64, g_npes, k, load);
    }
  }

  printf("\nrmw/word = returning SYSTEM RMWs landing on ONE address per iteration.\n"
         "grid span = first block in -> last block out on one GPU; skew = spread of the\n"
         "blocks' start times after the release (should be small).\n");
  return 0;
}
