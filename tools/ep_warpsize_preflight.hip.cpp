// EP v1 (C++ IntraNode) warpSize pre-flight — single GPU, single process, watchdog.
//
// WHY THIS EXISTS
// ---------------
// The Python launch layer hard-codes WARP_SIZE = 64
//   (python/mori/ops/dispatch_combine.py:33), and every v1 kernel launch does
//   block = (WARP_SIZE * warp_num_per_block,)  and sizes the *dynamic* shared
//   memory as warp_num_per_block warps' worth of pointer arrays
//   (EpDispatchCombineOp._combine_shared_mem).
// The device kernels instead recompute the warp count at runtime:
//   warpNum = blockDim.x / warpSize   (src/ops/dispatch_combine/intranode.hpp).
// On gfx942 warpSize==64, so host and device agree. On gfx1250 (MI450/MI455)
//   warpSize==32, so the device sees TWICE as many warps as the host budgeted.
//   The combine kernel then indexes srcPtrs[warpId*ept] and
//   srcWeightsPtr[warpNum*ept + warpId*ept] past the host-allocated LDS
//   (intranode.hpp:441-456) -> LDS overflow -> garbage src pointers ->
//   invalid device memory access in WarpAccum. Because all ranks rendezvous in
//   CrossDeviceBarrierIntraNodeKernel via ShmemUint32WaitUntilEquals spin-waits
//   (intranode.hpp:46-77, 215-227), one faulting rank leaves the other three
//   spinning forever -> the whole EP4 job HANGS (and can wedge the node).
//
// This tool reproduces the *cause* on ONE GPU, deterministically, with a host
// watchdog so it can NEVER wedge the node, so you can decide whether the real
// EP4 test is safe to launch BEFORE launching it.
//
// Build (pick your arch):
//   hipcc -O3 --offload-arch=gfx1250 ep_warpsize_preflight.hip.cpp -o ep_pf
//   hipcc -O3 --offload-arch=gfx942  ep_warpsize_preflight.hip.cpp -o ep_pf
// Run:
//   ./ep_pf [warp_num_per_block=8] [num_experts_per_token=8] [use_weights=1]
// Exit codes: 0 SAFE, 2 UNSAFE (host/device warp mismatch -> LDS overflow),
//             3 watchdog timeout, 1 HIP error.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

// Mirror of the Python launch-layer constants (dispatch_combine.py).
static constexpr int kHostWarpSize = 64;  // WARP_SIZE
static constexpr int kPtrSize = 8;        // _PTR_SIZE (sizeof(T**))
static constexpr int kWatchdogSec = 15;

#define CK(x)                                                                        \
  do {                                                                               \
    hipError_t _e = (x);                                                             \
    if (_e != hipSuccess) {                                                          \
      printf("HIP err %d (%s) at %s:%d\n", (int)_e, hipGetErrorString(_e), __FILE__, \
             __LINE__);                                                              \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)

// Replicates the combine kernel's LDS index arithmetic (intranode.hpp:441-456)
// but on a *generously* sized real LDS buffer, so we only MEASURE the byte range
// the real kernel would touch — we never actually overflow anything here.
//   srcPtrs[j]       -> sharedMem + warpId*ept ... (array 0)
//   srcWeightsPtr[j] -> sharedMem + warpNum*ept + warpId*ept ... (array 1, if weights)
__global__ void ProbeCombineLdsFootprint(unsigned* outWarpSize, unsigned* outWarpNum,
                                         unsigned long long* outMaxEndByte, int ept,
                                         int useWeights) {
  const int wsz = warpSize;  // device builtin: 64 on gfx9xx, 32 on gfx1250
  const int warpId = threadIdx.x / wsz;
  const int warpNum = blockDim.x / wsz;
  const int lane = threadIdx.x % wsz;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outWarpSize = (unsigned)wsz;
    *outWarpNum = (unsigned)warpNum;
  }
  if (lane == 0) {
    // Last byte (exclusive) each warp writes into the srcPtrs array.
    unsigned long long endPtrs = (unsigned long long)(warpId * ept + ept) * kPtrSize;
    unsigned long long hi = endPtrs;
    if (useWeights) {
      unsigned long long endW =
          (unsigned long long)(warpNum * ept + warpId * ept + ept) * kPtrSize;
      if (endW > hi) hi = endW;
    }
    atomicMax(outMaxEndByte, hi);
  }
}

int main(int argc, char** argv) {
  int wpb = (argc > 1) ? atoi(argv[1]) : 8;         // warp_num_per_block
  int ept = (argc > 2) ? atoi(argv[2]) : 8;         // num_experts_per_token
  int useWeights = (argc > 3) ? atoi(argv[3]) : 1;  // combine with weights

  int nDev = 0;
  CK(hipGetDeviceCount(&nDev));
  if (nDev < 1) {
    printf("no GPU visible\n");
    return 1;
  }
  CK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  CK(hipGetDeviceProperties(&prop, 0));

  const int numPtrArrays = 1 + (useWeights ? 1 : 0);
  // Exactly EpDispatchCombineOp._combine_shared_mem(wpb): host budgets for `wpb` warps.
  const long hostLds = (long)wpb * ept * numPtrArrays * kPtrSize;
  const int hostBlockThreads = kHostWarpSize * wpb;  // block = (WARP_SIZE * wpb,)

  printf("=== EP v1 IntraNode warpSize pre-flight ===\n");
  printf("device            : %s (%s)\n", prop.name, prop.gcnArchName);
  printf("prop.warpSize     : %d\n", prop.warpSize);
  printf("host assumes      : WARP_SIZE=%d, warp_num_per_block=%d, ept=%d, weights=%d\n",
         kHostWarpSize, wpb, ept, useWeights);
  printf("host launch block : %d threads  (= %d * %d)\n", hostBlockThreads, kHostWarpSize, wpb);
  printf("host LDS budgeted : %ld bytes  (_combine_shared_mem: %d*%d*%d*%d)\n", hostLds, wpb, ept,
         numPtrArrays, kPtrSize);

  // Give the probe kernel plenty of real LDS so it can never overflow; we only
  // read back the footprint it computes. Size for the worst case (warpSize==32).
  const int worstWarpNum = hostBlockThreads / 32;
  const size_t probeLds = (size_t)numPtrArrays * worstWarpNum * ept * kPtrSize + 256;

  unsigned *dWarpSize = nullptr, *dWarpNum = nullptr;
  unsigned long long* dMaxEnd = nullptr;
  CK(hipMalloc(&dWarpSize, sizeof(unsigned)));
  CK(hipMalloc(&dWarpNum, sizeof(unsigned)));
  CK(hipMalloc(&dMaxEnd, sizeof(unsigned long long)));
  CK(hipMemset(dWarpSize, 0, sizeof(unsigned)));
  CK(hipMemset(dWarpNum, 0, sizeof(unsigned)));
  CK(hipMemset(dMaxEnd, 0, sizeof(unsigned long long)));

  hipStream_t stream;
  CK(hipStreamCreate(&stream));
  ProbeCombineLdsFootprint<<<1, hostBlockThreads, probeLds, stream>>>(dWarpSize, dWarpNum, dMaxEnd,
                                                                      ept, useWeights);
  // ---- Watchdog: never let a wedged kernel hang the harness / node ----
  auto t0 = std::chrono::steady_clock::now();
  hipError_t q = hipErrorNotReady;
  while ((q = hipStreamQuery(stream)) == hipErrorNotReady) {
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(kWatchdogSec)) {
      printf("WATCHDOG_TIMEOUT after %ds — probe kernel did not finish; aborting.\n",
             kWatchdogSec);
      fflush(stdout);
      _exit(3);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  CK(q);

  unsigned hWarpSize = 0, hWarpNum = 0;
  unsigned long long hMaxEnd = 0;
  CK(hipMemcpy(&hWarpSize, dWarpSize, sizeof(unsigned), hipMemcpyDeviceToHost));
  CK(hipMemcpy(&hWarpNum, dWarpNum, sizeof(unsigned), hipMemcpyDeviceToHost));
  CK(hipMemcpy(&hMaxEnd, dMaxEnd, sizeof(unsigned long long), hipMemcpyDeviceToHost));

  printf("device warpSize   : %u\n", hWarpSize);
  printf("device warpNum    : %u  (blockDim %d / warpSize %u)\n", hWarpNum, hostBlockThreads,
         hWarpSize);
  printf("device combine LDS: %llu bytes actually indexed by the combine layout\n", hMaxEnd);

  bool unsafe = ((long long)hMaxEnd > hostLds);
  if (unsafe) {
    double factor = hostLds > 0 ? (double)hMaxEnd / (double)hostLds : 0.0;
    printf("\nRESULT: UNSAFE ❌\n");
    printf("  device indexes %llu bytes of dynamic LDS but the host only allocated %ld.\n", hMaxEnd,
           hostLds);
    printf("  overflow = %lld bytes  (%.2fx the budget).\n", (long long)hMaxEnd - hostLds, factor);
    printf("  In the real EpCombineIntraNodeKernel this overwrites/reads past the shared-mem\n");
    printf("  pointer arrays -> garbage src pointers -> invalid device memory access, and the\n");
    printf("  cross-device spin barrier never completes -> EP4 HANGS.\n");
    printf("  Root cause: WARP_SIZE=64 hard-coded in dispatch_combine.py vs device warpSize=%u.\n",
           hWarpSize);
    printf("  DO NOT launch the v1 C++ IntraNode EP4 test on this arch until fixed.\n");
  } else {
    printf("\nRESULT: SAFE ✅  (host WARP_SIZE matches device warpSize; no LDS overflow)\n");
  }

  CK(hipFree(dWarpSize));
  CK(hipFree(dWarpNum));
  CK(hipFree(dMaxEnd));
  CK(hipStreamDestroy(stream));
  return unsafe ? 2 : 0;
}
