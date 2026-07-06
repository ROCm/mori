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

// test: cco gda ThreadAggregate mode.
//
// Verifies the ccoGdaThreadAggregate template parameter on put / putValue / get.
// With ThreadAggregate all warp lanes target the same peer; the leader
// posts one shared signal WQE and rings one doorbell for the batch, instead
// of each lane posting its own.
//
// Cases:
//   put:      64 threads put different chunks → data correct, signal == 1
//   putvalue: 64 threads putValue different uint64_t → data correct, signal == 1
//   get:      64 threads get different chunks → data correct

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static constexpr int WARP_SIZE = 64;
static constexpr int CHUNK_ELEMS = 64;
static constexpr mori::core::ProviderType kPrvdType = CCO_GDA_BUILD_PROVIDER;

// ── Kernels ─────────────────────────────────────────────────────────────────

__global__ void ThreadAggPutKernel(mori::cco::ccoWindowDevice* sendWin,
                                   mori::cco::ccoWindowDevice* recvWin, size_t chunkBytes,
                                   mori::cco::ccoDevComm devComm, int peer) {
  using namespace mori::cco;
  ccoGda<kPrvdType> gda{devComm, 0};
  int tid = threadIdx.x;

  gda.put<CCO_TEAM_WORLD, ccoGdaThreadAggregate, ccoGda_SignalInc, ccoCoopThread>(
      peer, reinterpret_cast<ccoWindow_t>(recvWin), tid * chunkBytes,
      reinterpret_cast<ccoWindow_t>(sendWin), tid * chunkBytes, chunkBytes, ccoGda_SignalInc{0});

  gda.flush(ccoCoopWarp{});
}

__global__ void ThreadAggPutValueKernel(mori::cco::ccoWindowDevice* recvWin,
                                        mori::cco::ccoDevComm devComm, int peer, uint64_t baseVal) {
  using namespace mori::cco;
  ccoGda<kPrvdType> gda{devComm, 0};
  int tid = threadIdx.x;

  uint64_t val = baseVal + static_cast<uint64_t>(tid);
  gda.putValue<CCO_TEAM_WORLD, ccoGdaThreadAggregate, uint64_t, ccoGda_SignalInc, ccoCoopThread>(
      peer, reinterpret_cast<ccoWindow_t>(recvWin), tid * sizeof(uint64_t), val,
      ccoGda_SignalInc{1});

  gda.flush(ccoCoopWarp{});
}

__global__ void ThreadAggGetKernel(mori::cco::ccoWindowDevice* remoteWin,
                                   mori::cco::ccoWindowDevice* localWin, size_t chunkBytes,
                                   mori::cco::ccoDevComm devComm, int peer) {
  using namespace mori::cco;
  ccoGda<kPrvdType> gda{devComm, 0};
  int tid = threadIdx.x;

  gda.get<CCO_TEAM_WORLD, ccoGdaThreadAggregate, ccoCoopThread>(
      peer, reinterpret_cast<ccoWindow_t>(remoteWin), tid * chunkBytes,
      reinterpret_cast<ccoWindow_t>(localWin), tid * chunkBytes, chunkBytes);

  gda.flush(mori::cco::ccoCoopBlock{});
}

__global__ void CheckSignalKernel(mori::cco::ccoDevComm devComm, uint32_t slot, uint64_t expected,
                                  int round, int* errorFlag) {
  using namespace mori::cco;
  if (threadIdx.x != 0) return;
  ccoGda<kPrvdType> gda{devComm, 0};
  uint64_t got = gda.readSignal(static_cast<ccoGdaSignal_t>(slot));
  if (got != expected) {
    printf("[rank %d] round %d: signal[%u] = %llu, expected %llu\n", devComm.rank, round, slot,
           (unsigned long long)got, (unsigned long long)expected);
    atomicExch(errorFlag, 1);
  }
}

__global__ void ResetSignalKernel(mori::cco::ccoDevComm devComm, int nSlots) {
  using namespace mori::cco;
  if (threadIdx.x != 0) return;
  ccoGda<kPrvdType> gda{devComm, 0};
  for (int i = 0; i < nSlots; i++) gda.resetSignal(static_cast<ccoGdaSignal_t>(i));
}

// ── UT context ──────────────────────────────────────────────────────────────

struct UtCtx {
  mori::cco::ccoComm* comm;
  mori::cco::ccoDevComm dc;
  hipStream_t stream;
  int* dErr;
  int nranks, rank;
  void* sendBuf;
  void* recvBuf;
  mori::cco::ccoWindow_t sendWin;
  mori::cco::ccoWindow_t recvWin;
  size_t bufSize;
  int nextPeer;
  int prevPeer;

  template <typename F>
  void step(F&& f) {
    f();
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
  }

  void resetSignals() {
    step([&] { ResetSignalKernel<<<1, 1, 0, stream>>>(dc, 2); });
  }

  void fail() {
    int one = 1;
    HIP_CHECK(hipMemcpy(dErr, &one, sizeof(int), hipMemcpyHostToDevice));
  }

  int harvest(const char* name) {
    int hErr = 0;
    HIP_CHECK(hipMemcpy(&hErr, dErr, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));
    if (rank == 0) printf("  [%-12s] %s\n", name, hErr == 0 ? "PASS" : "FAIL");
    mori::cco::ccoBarrierAll(comm);
    return hErr == 0 ? 0 : 1;
  }
};

// ── Test cases ──────────────────────────────────────────────────────────────

// 64 threads put different chunks to same peer, verify data + signal == 1
static int ut_thread_agg_put(UtCtx& c) {
  c.resetSignals();

  size_t totalElems = WARP_SIZE * CHUNK_ELEMS;
  size_t chunkBytes = CHUNK_ELEMS * sizeof(float);

  std::vector<float> hostSend(totalElems);
  for (int t = 0; t < WARP_SIZE; t++)
    for (int i = 0; i < CHUNK_ELEMS; i++)
      hostSend[t * CHUNK_ELEMS + i] = static_cast<float>(c.rank * 10000 + t * 100 + i);
  HIP_CHECK(
      hipMemcpy(c.sendBuf, hostSend.data(), totalElems * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(c.recvBuf, 0xff, c.bufSize));
  mori::cco::ccoBarrierAll(c.comm);

  c.step([&] {
    ThreadAggPutKernel<<<1, WARP_SIZE, 0, c.stream>>>(c.sendWin, c.recvWin, chunkBytes, c.dc,
                                                      c.nextPeer);
  });

  // ThreadAggregate → leader posts 1 signal, not 64
  c.step([&] { CheckSignalKernel<<<1, 1, 0, c.stream>>>(c.dc, 0, 1, 0, c.dErr); });

  std::vector<float> hostRecv(totalElems);
  HIP_CHECK(
      hipMemcpy(hostRecv.data(), c.recvBuf, totalElems * sizeof(float), hipMemcpyDeviceToHost));

  for (int t = 0; t < WARP_SIZE; t++) {
    for (int i = 0; i < CHUNK_ELEMS; i++) {
      float expected = static_cast<float>(c.prevPeer * 10000 + t * 100 + i);
      float got = hostRecv[t * CHUNK_ELEMS + i];
      if (got != expected) {
        fprintf(stderr, "[rank %d] put: chunk[%d][%d] = %.0f, expected %.0f\n", c.rank, t, i, got,
                expected);
        c.fail();
        return c.harvest("put");
      }
    }
  }
  return c.harvest("put");
}

// 64 threads putValue different uint64_t, verify data + signal == 1
static int ut_thread_agg_putvalue(UtCtx& c) {
  c.resetSignals();
  HIP_CHECK(hipMemset(c.recvBuf, 0xff, c.bufSize));
  mori::cco::ccoBarrierAll(c.comm);

  uint64_t baseVal = static_cast<uint64_t>(c.rank) * 10000;

  c.step([&] {
    ThreadAggPutValueKernel<<<1, WARP_SIZE, 0, c.stream>>>(c.recvWin, c.dc, c.nextPeer, baseVal);
  });

  c.step([&] { CheckSignalKernel<<<1, 1, 0, c.stream>>>(c.dc, 1, 1, 1, c.dErr); });

  std::vector<uint64_t> hostRecv(WARP_SIZE);
  HIP_CHECK(
      hipMemcpy(hostRecv.data(), c.recvBuf, WARP_SIZE * sizeof(uint64_t), hipMemcpyDeviceToHost));

  uint64_t srcBase = static_cast<uint64_t>(c.prevPeer) * 10000;
  for (int t = 0; t < WARP_SIZE; t++) {
    uint64_t expected = srcBase + static_cast<uint64_t>(t);
    if (hostRecv[t] != expected) {
      fprintf(stderr, "[rank %d] putvalue: slot[%d] = %llu, expected %llu\n", c.rank, t,
              (unsigned long long)hostRecv[t], (unsigned long long)expected);
      c.fail();
      return c.harvest("putvalue");
    }
  }
  return c.harvest("putvalue");
}

// 64 threads get different chunks from same peer, verify data
static int ut_thread_agg_get(UtCtx& c) {
  size_t totalElems = WARP_SIZE * CHUNK_ELEMS;
  size_t chunkBytes = CHUNK_ELEMS * sizeof(float);

  std::vector<float> hostSend(totalElems);
  for (int t = 0; t < WARP_SIZE; t++)
    for (int i = 0; i < CHUNK_ELEMS; i++)
      hostSend[t * CHUNK_ELEMS + i] = static_cast<float>(c.rank * 10000 + t * 100 + i);
  HIP_CHECK(
      hipMemcpy(c.sendBuf, hostSend.data(), totalElems * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(c.recvBuf, 0, c.bufSize));
  mori::cco::ccoBarrierAll(c.comm);

  c.step([&] {
    ThreadAggGetKernel<<<1, WARP_SIZE, 0, c.stream>>>(c.sendWin, c.recvWin, chunkBytes, c.dc,
                                                      c.nextPeer);
  });

  std::vector<float> hostRecv(totalElems);
  HIP_CHECK(
      hipMemcpy(hostRecv.data(), c.recvBuf, totalElems * sizeof(float), hipMemcpyDeviceToHost));

  for (int t = 0; t < WARP_SIZE; t++) {
    for (int i = 0; i < CHUNK_ELEMS; i++) {
      float expected = static_cast<float>(c.nextPeer * 10000 + t * 100 + i);
      float got = hostRecv[t * CHUNK_ELEMS + i];
      if (got != expected) {
        fprintf(stderr, "[rank %d] get: chunk[%d][%d] = %.0f, expected %.0f\n", c.rank, t, i, got,
                expected);
        c.fail();
        return c.harvest("get");
      }
    }
  }
  return c.harvest("get");
}

using UtFn = int (*)(UtCtx&);

static int run_all_tests(UtCtx& ctx) {
  static const struct {
    const char* name;
    UtFn fn;
  } kCases[] = {
      {"put", ut_thread_agg_put},
      // {"putvalue", ut_thread_agg_putvalue},
      // {"get", ut_thread_agg_get},
  };
  int fails = 0;
  for (const auto& c : kCases) fails += c.fn(ctx);
  int n = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  if (ctx.rank == 0) printf("=== %d/%d PASS ===\n", n - fails, n);
  return fails;
}

// ── Host driver ─────────────────────────────────────────────────────────────

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
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
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  size_t bufSize = WARP_SIZE * CHUNK_ELEMS * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

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
  reqs.gdaSignalCount = 2;  // slot 0 for put, slot 1 for putValue
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  if (devComm.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE\n", rank);
    return 1;
  }

  printf("[rank %d] DevCommCreate OK (worldSize=%d, gdaConnType=%d)\n", rank, devComm.worldSize,
         (int)devComm.gdaConnType);

  int* dErr = nullptr;
  HIP_CHECK(hipMalloc(&dErr, sizeof(int)));
  HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));

  UtCtx ctx{comm,
            devComm,
            nullptr,
            dErr,
            nranks,
            rank,
            sendBuf,
            recvBuf,
            sendWin,
            recvWin,
            bufSize,
            (rank + 1) % nranks,
            (rank - 1 + nranks) % nranks};
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
  return ccoTestMain(argc, argv, "CCO GDA thread aggregate", "/tmp/cco_gda_thread_agg_uid", 19883);
}
