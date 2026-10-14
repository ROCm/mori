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

// test: ccoCommCreate must not leak the fabric-handle probe's HIP error.
//
// ccoCommCreate probes hipMemHandleTypeFabricCompat and falls back to the FD
// path when it is unsupported. The failed probe leaves the runtime's sticky
// last-error set; if it is not consumed, the next framework call that checks
// hipGetLastError() (e.g. a torch kernel launch) reports it as its own error.
// On hosts where the probe succeeds there is nothing to leak and the test
// passes trivially.

#include <cstdio>

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  HIP_CHECK(hipFree(nullptr));  // force context creation
  (void)hipGetLastError();      // start from a clean slate

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] ccoCommCreate failed\n", rank);
    return 1;
  }

  int rc = 0;
  hipError_t leaked = hipGetLastError();
  if (leaked != hipSuccess) {
    fprintf(stderr,
            "[rank %d] FAIL: ccoCommCreate leaked sticky HIP error %d (%s), handleType=%d\n", rank,
            leaked, hipGetErrorString(leaked), comm->handleType);
    rc = 1;
  }

  // Mimic the framework allocation that surfaced the bug.
  void* p = nullptr;
  hipError_t mallocErr = hipMalloc(&p, 1 << 20);
  hipError_t afterMalloc = hipGetLastError();
  if (mallocErr != hipSuccess || afterMalloc != hipSuccess) {
    fprintf(stderr, "[rank %d] FAIL: hipMalloc after CommCreate: ret=%d lastError=%d\n", rank,
            mallocErr, afterMalloc);
    rc = 1;
  }
  if (p) (void)hipFree(p);

  int handleType = comm->handleType;
  ccoCommDestroy(comm);
  if (rc == 0) printf("[rank %d] PASS (handleType=%d)\n", rank, handleType);
  return rc;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO fabric probe last-error", "/tmp/cco_fabric_probe_uid", 0);
}
