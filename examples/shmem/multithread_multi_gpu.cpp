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

// Multithread multi-GPU exploration test.
//
// Spawns one host thread per GPU within a single process. Each thread sets its
// own HIP device and attempts ShmemInit / ShmemFinalize through the socket
// bootstrap.  The purpose is to observe how the current single-GPU-per-process
// design behaves and to serve as a starting point for multi-GPU-per-process
// integration work.
//
// Run (no MPI needed):
//   ./multithread_multi_gpu [num_gpus]
//
// If num_gpus is omitted, all visible GPUs are used.

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::shmem;
using namespace mori::application;
using namespace mori::core;

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

// Collective permute kernel: each PE writes its own pe_id into the next PE's
// destination buffer (ring: pe -> (pe+1) % npes).  A single thread is enough
// for this smoke test.
__global__ void CollectivePermuteKernel(int myPe, int nPes,
                                        uint32_t* dst) {
  int nextPe = (myPe + 1) % nPes;
  uint64_t dstPtr = ShmemPtrP2p(reinterpret_cast<uint64_t>(dst), myPe, nextPe);
  auto *peerDst = reinterpret_cast<uint32_t*>(dstPtr);

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  // printf("pe: %d rank=%d mydst=%p in [%lx; %lx] dstPtr=%p heapObj=%p\n",
  //       myPe, globalGpuStates->rank, 
  //       dst, 
  //       globalGpuStates->heapBaseAddr,
  //       globalGpuStates->heapEndAddr,
  //       reinterpret_cast<void*>(dstPtr), reinterpret_cast<void*>(globalGpuStates->heapObj));
  // printf("pe: %d rank=%d mydst=%p peerDst=%p\n",
  //   myPe, globalGpuStates->rank, dst, peerDst);
  *peerDst = myPe;
  ShmemFenceThread();
}

// C++17-friendly reusable barrier (std::barrier is C++20)
class ThreadBarrier {
 public:
  explicit ThreadBarrier(int count) : threshold_(count), count_(count), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    int gen = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  int threshold_;
  int count_;
  int generation_;
};

struct ThreadResult {
  int gpu_id{-1};
  int init_status{-1};
  int my_pe{-1};
  int n_pes{-1};
  int finalize_status{-1};
  std::string error;
  std::thread::id thread_id;
};

static void GpuThreadFunc(int thread_id, int num_threads, const UniqueId& uid,
                          ThreadBarrier& barrier, ThreadResult& result) {
  result.gpu_id = thread_id;
  result.thread_id = std::this_thread::get_id();
  // --- Phase 1: Set HIP device ---
  XPUT("[thread %d] hipSetDevice(%d)", thread_id, thread_id);
  hipError_t hip_err = hipSetDevice(thread_id);
  if (hip_err != hipSuccess) {
    result.error = std::string("hipSetDevice failed: ") + hipGetErrorString(hip_err);
    XPUT("[thread %d] ERROR: %s", thread_id, result.error.c_str());
    barrier.Wait();  // still participate in barrier so others don't hang
    barrier.Wait();
    return;
  }

  // Verify the device is set correctly
  int actual_device = -1;
  HIP_RUNTIME_CHECK(hipGetDevice(&actual_device));
  XPUT("[thread %d] hipGetDevice() = %d", thread_id, actual_device);

  // --- Phase 2: All threads synchronize before ShmemInit ---
  barrier.Wait();

  XPUT("[thread %d] Calling ShmemInit (rank=%d, nranks=%d)", thread_id, thread_id, num_threads);

  //auto* bootstrap = new LocalBootstrapNetwork(thread_id, num_threads);
  auto* bootstrap = new SocketBootstrapNetwork(uid, thread_id, num_threads);
  result.init_status = ShmemInit(bootstrap);

  // XPUT("[thread %d] ShmemInit returned %d", thread_id, result.init_status);

  if (result.init_status == 0) {
    result.my_pe = ShmemMyPe();
    result.n_pes = ShmemNPes();
    XPUT("[thread %d] ShmemMyPe()=%d, ShmemNPes()=%d", thread_id, result.my_pe, result.n_pes);
  }

  // Verify HIP device hasn't been changed behind our back
  int device_after_init = -1;
  HIP_RUNTIME_CHECK(hipGetDevice(&device_after_init));
  if (device_after_init != thread_id) {
    XPUT("[thread %d] WARNING: HIP device changed from %d to %d during ShmemInit!", thread_id,
         thread_id, device_after_init);
  }

  // --- Phase 3: Allocate symmetric buffers and run collective permute ---
  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  constexpr size_t kBufBytes = sizeof(uint32_t);
  void* dstBuff = ShmemMalloc(kBufBytes);
  XPUT("[thread %d] dstBuff=%p", thread_id, dstBuff);

  // src = my pe id, dst = sentinel
  HIP_RUNTIME_CHECK(
      hipMemsetD32Async(reinterpret_cast<uint32_t*>(dstBuff), 0xDEADBEEF, 1, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  SymmMemObjPtr dstObj = ShmemQueryMemObjPtr(dstBuff);
  assert(dstObj.IsValid());

  // All threads must be ready before the permute
  barrier.Wait();

  XPUT("[thread %d] Launching CollectivePermuteKernel (pe %d -> pe %d)",
       thread_id, thread_id, (thread_id + 1) % num_threads);
  CollectivePermuteKernel<<<1, 1, 0, stream>>>(thread_id, num_threads, 
        reinterpret_cast<uint32_t*>(dstBuff));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  // Wait for all PEs to finish writing
  barrier.Wait();

  // --- Phase 4: Verify ---
  uint32_t got = 0;
  HIP_RUNTIME_CHECK(hipMemcpy(&got, dstBuff, sizeof(uint32_t), hipMemcpyDeviceToHost));
  int expected_sender = (thread_id - 1 + num_threads) % num_threads;
  if (got == static_cast<uint32_t>(expected_sender)) {
    XPUT("[thread %d] PASS: dst=0x%08x (from pe %d)", thread_id, got, expected_sender);
  } else {
    XPUT("[thread %d] FAIL: dst=0x%08x, expected 0x%08x (from pe %d)",
         thread_id, got, static_cast<uint32_t>(expected_sender), expected_sender);
  }

  // --- Phase 5: Cleanup ---
  barrier.Wait();

  ShmemFree(dstBuff);
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));

  if (result.init_status == 0) {
    result.finalize_status = ShmemFinalize();
    XPUT("[thread %d] ShmemFinalize returned %d", thread_id, result.finalize_status);
  }

  XPUT("[thread %d] Done", thread_id);
}

int main(int argc, char* argv[]) {
  // --- Determine how many GPUs to use ---
  int device_count = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&device_count));
  XPUT("Detected %d GPU device(s)", device_count);

  int num_gpus = device_count;
  if (argc > 1) {
    num_gpus = std::atoi(argv[1]);
    if (num_gpus < 1 || num_gpus > device_count) {
      XPUT("Usage: %s [num_gpus]  (1..%d)", argv[0], device_count);
      return 1;
    }
  }

  if (num_gpus < 2) {
    XPUT("Need at least 2 GPUs for this test (found %d)", device_count);
    return 1;
  }

  XPUT("\n=== Multi-thread multi-GPU exploration test ===");
  XPUT("Using %d GPUs in a single process\n", num_gpus);

  // --- Generate UniqueId for socket bootstrap (from "rank 0" perspective) ---
  mori_shmem_uniqueid_t uid_bytes;
  int ret = ShmemGetUniqueId(&uid_bytes);
  if (ret != 0) {
    XPUT("ShmemGetUniqueId failed (ret=%d)", ret);
    return 1;
  }

  UniqueId uid;
  static_assert(sizeof(uid) == sizeof(uid_bytes), "UniqueId size mismatch");
  std::memcpy(&uid, uid_bytes.data(), sizeof(uid));

  XPUT("Socket bootstrap UniqueId generated successfully\n");

  // --- Launch one thread per GPU ---
  ThreadBarrier barrier(num_gpus);
  std::vector<ThreadResult> results(num_gpus);
  std::vector<std::thread> threads;

  for (int i = 0; i < num_gpus; i++) {
    threads.emplace_back(GpuThreadFunc, i, num_gpus, std::cref(uid), std::ref(barrier),
                         std::ref(results[i]));
  }

  for (auto& t : threads) {
    t.join();
  }

  // --- Summary ---
  XPUT("\n=== Results summary ===");
  XPUT("%-16s %-8s %-14s %-8s %-8s %-16s", "Thread", "GPU", "ShmemInit", "MyPe", "NPes",
       "ShmemFinalize");
  XPUT("-------- -------- -------------- -------- -------- ---------------- --------");

  for (int i = 0; i < num_gpus; i++) {
    auto& r = results[i];
    std::ostringstream oss;
    oss << r.thread_id;
    XPUT("%-16s %-8d %-14s %-8d %-8d %-16s %s", oss.str().c_str(), r.gpu_id,
         (r.init_status == 0 ? "OK" : "SKIPPED/FAIL"), r.my_pe, r.n_pes,
         (r.finalize_status == 0 ? "OK" : "N/A"), r.error.c_str());
  }

  XPUT("\n=== Observations ===");
  int init_ok_count = 0;
  for (auto& r : results) {
    if (r.init_status == 0) init_ok_count++;
  }
  XPUT("  Threads that completed ShmemInit: %d / %d", init_ok_count, num_gpus);

  if (init_ok_count == 1) {
    XPUT("  -> As expected: ShmemStates is a process-wide singleton.");
    XPUT("     Only the first thread to call ShmemInit succeeds;");
    XPUT("     subsequent threads see status==Initialized and skip.");
    XPUT("     The globalGpuStates __device__ symbol is also per-process,");
    XPUT("     not per-GPU.");
  } else if (init_ok_count == num_gpus) {
    XPUT("  -> All threads initialized (unexpected with current singleton design).");
  } else {
    XPUT("  -> Partial initialization (%d/%d).", init_ok_count, num_gpus);
  }

  XPUT("\nDone.");
  return 0;
}
