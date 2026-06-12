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

// test: cco gda barrier — signal-based cross-node barrier.
//
// GDA barrier uses the existing signal infrastructure: each rank RDMA
// atomic-adds to every peer's signalBuf slot, then polls for reciprocal
// signals. Shadow-based delta semantics allow repeated sync() calls on the
// same session (shadows auto-advance on each wait).
//
//   basic    — all ranks enter and exit a single sync().
//   multi    — N consecutive sync() calls on the same session.
//   data_vis — rank puts data, barrier, peer reads; data is visible.

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;

static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::PSD;

// BASIC: every rank does one sync(). If any rank hangs, the test times out.
template <mori::core::ProviderType PrvdType>
__global__ void GdaBarrierBasicKernel(mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, 0};
  ccoGdaBarrierSession<PrvdType, ccoCoopBlock> session(ccoCoopBlock{}, gda,
                                                       devComm.railGdaBarrier, 0);
  session.sync(ccoCoopBlock{});
}

// MULTI: N consecutive sync() calls on the same session.
template <mori::core::ProviderType PrvdType>
__global__ void GdaBarrierMultiKernel(mori::cco::ccoDevComm devComm, int rounds) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, 0};
  ccoGdaBarrierSession<PrvdType, ccoCoopBlock> session(ccoCoopBlock{}, gda,
                                                       devComm.railGdaBarrier, 0);
  for (int r = 0; r < rounds; r++) {
    session.sync(ccoCoopBlock{});
  }
}

// DATA_VIS: rank 0 puts data to every peer, barrier, then every peer reads
// and verifies. Proves that barrier guarantees put visibility.
// sync() ends with coop.sync(), so the check is safe immediately after.
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaBarrierDataVisKernel(mori::cco::ccoWindowDevice* sendWin,
                                        mori::cco::ccoWindowDevice* recvWin, size_t count,
                                        mori::cco::ccoDevComm devComm, int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, 0};
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;

  if (myRank == 0) {
    // Each thread issues the put to a different peer (one QP per peer).
    for (int r = 1 + threadIdx.x; r < nRanks; r += blockDim.x) {
      gda.put(r, reinterpret_cast<ccoWindow_t>(recvWin), 0,
              reinterpret_cast<ccoWindow_t>(sendWin), 0, count * sizeof(T));
    }
    // flush is block-cooperative: all threads of the block must participate.
    gda.flush(ccoCoopBlock{});
  }

  ccoGdaBarrier(ccoCoopBlock{}, gda, devComm.railGdaBarrier, 0);

  if (threadIdx.x == 0 && myRank != 0) {
    char* base = recvWin->winBase + ((uint64_t)recvWin->lsaRank * recvWin->stride4G << 32);
    T* buf = reinterpret_cast<T*>(base);
    for (size_t i = 0; i < count; i++) {
      T expected = static_cast<T>(i + 1);
      if (buf[i] != expected) {
        printf("[rank %d] data_vis: buf[%zu] = %f, expected %f\n", myRank, i, (double)buf[i],
               (double)expected);
        atomicExch(errorFlag, 1);
        return;
      }
    }
  }
}

// ── UT context + helpers ─────────────────────────────────────────────────────
struct UtCtx {
  mori::cco::ccoComm* comm;
  mori::cco::ccoDevComm dc;
  mori::cco::ccoWindow_t sendWin;
  mori::cco::ccoWindow_t recvWin;
  void* sendBuf;
  void* recvBuf;
  hipStream_t stream;
  int* dErr;
  int nranks;
  int rank;

  template <typename LaunchFn>
  void step(LaunchFn&& launch) {
    launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
  }
  int harvest(const char* name) {
    int hErr = 0;
    HIP_CHECK(hipMemcpy(&hErr, dErr, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));
    if (rank == 0) printf("  [%-10s] %s\n", name, hErr == 0 ? "PASS" : "FAIL");
    mori::cco::ccoBarrierAll(comm);
    return hErr == 0 ? 0 : 1;
  }
};

// ── UT cases ─────────────────────────────────────────────────────────────────

static int ut_basic(UtCtx& c) {
  c.step([&] { GdaBarrierBasicKernel<kPrvdType><<<1, 64, 0, c.stream>>>(c.dc); });
  return c.harvest("basic");
}

static int ut_multi(UtCtx& c) {
  const int N = 5;
  c.step([&] { GdaBarrierMultiKernel<kPrvdType><<<1, 64, 0, c.stream>>>(c.dc, N); });
  return c.harvest("multi");
}

static int ut_data_vis(UtCtx& c) {
  // Rank 0 fills sendBuf with [1..COUNT].
  if (c.rank == 0) {
    std::vector<float> hostBuf(COUNT);
    for (size_t i = 0; i < COUNT; i++) hostBuf[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpy(c.sendBuf, hostBuf.data(), COUNT * sizeof(float), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipStreamSynchronize(c.stream));
  mori::cco::ccoBarrierAll(c.comm);

  c.step([&] {
    GdaBarrierDataVisKernel<kPrvdType, float>
        <<<1, 64, 0, c.stream>>>(c.sendWin, c.recvWin, COUNT, c.dc, c.dErr);
  });
  return c.harvest("data_vis");
}

using UtFn = int (*)(UtCtx&);

static int run_all_tests(UtCtx& ctx) {
  static const struct {
    const char* name;
    UtFn fn;
  } kCases[] = {
      {"basic", ut_basic},
      {"multi", ut_multi},
      {"data_vis", ut_data_vis},
  };
  int fails = 0;
  for (const auto& c : kCases) fails += c.fn(ctx);
  const int n = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  if (ctx.rank == 0) printf("=== %d/%d PASS ===\n", n - fails, n);
  return fails;
}

// ── host driver ──────────────────────────────────────────────────────────────

int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
  g_rank = rank;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  for (int i = 0; i < numDevices; i++) {
    if (i == dev) continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dev, i));
    if (canAccess) (void)hipDeviceEnablePeerAccess(i, 0);
  }

  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  size_t bufSize = COUNT * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  HIP_CHECK(hipMemset(sendBuf, 0, bufSize));
  HIP_CHECK(hipMemset(recvBuf, 0, bufSize));

  mori::cco::ccoWindow_t sendWin = nullptr;
  mori::cco::ccoWindow_t recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;
  reqs.railGdaBarrierCount = 2;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  if (devComm.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE\n", rank);
    return 1;
  }

  int* dErr = nullptr;
  HIP_CHECK(hipMalloc(&dErr, sizeof(int)));
  HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));

  UtCtx ctx{comm, devComm, sendWin, recvWin, sendBuf, recvBuf, /*stream=*/nullptr, dErr, nranks, rank};
  HIP_CHECK(hipStreamCreate(&ctx.stream));

  mori::cco::ccoBarrierAll(comm);
  int fails = run_all_tests(ctx);

  HIP_CHECK(hipStreamDestroy(ctx.stream));
  HIP_CHECK(hipFree(dErr));
  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA barrier", "/tmp/cco_gda_barrier_uid", 19887);
}
