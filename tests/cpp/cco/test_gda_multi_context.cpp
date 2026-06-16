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

// test: cco gda multi-context DevComm lifecycle + multi-context send.
//
// Two things under test, exercised together over several iterations:
//   1. Lifecycle: repeatedly ccoDevCommCreate(gdaContextCount=NCTX) →
//      use → ccoDevCommDestroy. Each create allocates NCTX independent QP sets
//      per peer and connects them; each destroy must tear them down cleanly.
//      Looping ITERS times catches QP/endpoint leaks and non-reentrant setup.
//   2. Multi-context send: within each iteration, an all-to-all RDMA put is
//      issued where context c (a distinct ccoGda{devComm, c} → distinct QP via
//      qpIdx = teamPeer*numQpPerPe + c%numQpPerPe) carries its OWN slice of the
//      payload. All NCTX contexts run concurrently to the same peer over
//      different QPs; the receiver verifies every byte, so a misrouted or
//      dropped context surfaces as a data mismatch.
//
// Windows/buffers are created once; only the DevComm is recreated each iter
// (that is the connection create/destroy path we want to stress).

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;  // float elements per (peer, context) slice
static const int NCTX = 4;        // GDA contexts == gdaContextCount == QPs/peer
static const int ITERS = 8;       // DevComm create/destroy cycles

static constexpr mori::core::ProviderType kPrvdType = CCO_GDA_BUILD_PROVIDER;  // per-NIC build

// All-to-all where each GDA context sends its own payload slice over its own QP.
//
// Buffer layout per rank: [peer][context][COUNT] floats.
//   send slice for (peer p, context c) = sendBuf[(p*NCTX + c)*COUNT ..]
//   recv slice from (src s, context c) = recvBuf[(s*NCTX + c)*COUNT ..]
//
// Block has NCTX warps; warp c owns context c. Warp c constructs ccoGda{dc, c}
// and puts the (peer, c) slice to every peer. Distinct contexts → distinct QPs.
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaMultiCtxAlltoAllKernel(mori::cco::ccoWindowDevice* sendWin,
                                          mori::cco::ccoWindowDevice* recvWin, size_t count,
                                          int nctx, mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;

  const int warpSize_ = warpSize;
  int warpId = threadIdx.x / warpSize_;  // == context id
  int lane = threadIdx.x % warpSize_;
  if (warpId >= nctx) return;

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  size_t sliceBytes = count * sizeof(T);
  int c = warpId;

  ccoGda<PrvdType> gda{devComm, /*ginContext=*/c};

  // Lane 0 of each warp issues this context's put to every peer.
  if (lane == 0) {
    for (int p = 0; p < nRanks; p++) {
      if (p == myRank) continue;
      size_t dstOff = (static_cast<size_t>(myRank) * nctx + c) * sliceBytes;
      size_t srcOff = (static_cast<size_t>(p) * nctx + c) * sliceBytes;
      gda.put(p, reinterpret_cast<ccoWindow_t>(recvWin), dstOff,
              reinterpret_cast<ccoWindow_t>(sendWin), srcOff, sliceBytes,
              ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank * nctx + c)});
    }
  }

  // Each warp drains its own context's QPs.
  gda.flush(ccoCoopWarp{});

  // Wait for every peer's write on THIS context to land in our signal slot.
  if (lane == 0) {
    for (int p = 0; p < nRanks; p++) {
      if (p == myRank) continue;
      gda.waitSignal(static_cast<ccoGdaSignal_t>(p * nctx + c), 1);
    }
  }
}

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

  // Buffers/windows created ONCE; per-(peer,context) slices.
  size_t bufSize = static_cast<size_t>(nranks) * NCTX * COUNT * sizeof(float);
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

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const int warpSz = 64;  // gfx warp size
  dim3 grid(1);
  dim3 block(NCTX * warpSz);  // one warp per context

  bool ok = true;

  // ── lifecycle loop: recreate the DevComm (and its QP connections) each iter ──
  for (int it = 0; it < ITERS && ok; it++) {
    // encode iteration into payload so a stale buffer from a prior iter is caught.
    // sendBuf[(p*NCTX + c)*COUNT + i] = rank*1e6 + p*1e4 + c*1e2 + i + it
    std::vector<float> hostSend(static_cast<size_t>(nranks) * NCTX * COUNT);
    for (int p = 0; p < nranks; p++)
      for (int c = 0; c < NCTX; c++)
        for (size_t i = 0; i < COUNT; i++)
          hostSend[(p * NCTX + c) * COUNT + i] =
              static_cast<float>(rank * 1000000 + p * 10000 + c * 100 + (int)i + it);
    HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));

    // Create a fresh multi-context DevComm: NCTX QP sets per peer + one signal
    // slot per (peer, context).
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
    reqs.gdaContextCount = NCTX;
    reqs.gdaSignalCount = nranks * NCTX;
    reqs.gdaCounterCount = 0;
    mori::cco::ccoDevComm devComm{};
    if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
      fprintf(stderr, "[rank %d] iter %d: DevCommCreate failed\n", rank, it);
      ok = false;
      break;
    }
    if (devComm.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
      fprintf(stderr, "[rank %d] iter %d: gdaConnType collapsed to NONE\n", rank, it);
      mori::cco::ccoDevCommDestroy(comm, &devComm);
      ok = false;
      break;
    }
    if (rank == 0 && it == 0) {
      printf("[rank 0] DevComm OK: worldSize=%d gdaConnType=%d numQpPerPe=%d (NCTX=%d)\n",
             devComm.worldSize, (int)devComm.gdaConnType, devComm.ibgda.numQpPerPe, NCTX);
    }

    mori::cco::ccoBarrierAll(comm);
    GdaMultiCtxAlltoAllKernel<kPrvdType, float>
        <<<grid, block, 0, stream>>>(sendWin, recvWin, COUNT, NCTX, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);

    // verify: recv slice from (src s, context c) == s's send slice for (me, c).
    // expected = s*1e6 + rank*1e4 + c*1e2 + i + it
    std::vector<float> hostRecv(static_cast<size_t>(nranks) * NCTX * COUNT);
    HIP_CHECK(hipMemcpy(hostRecv.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
    for (int s = 0; s < nranks && ok; s++) {
      if (s == rank) continue;  // self slices never written
      for (int c = 0; c < NCTX && ok; c++) {
        for (size_t i = 0; i < COUNT; i++) {
          float exp = static_cast<float>(s * 1000000 + rank * 10000 + c * 100 + (int)i + it);
          float got = hostRecv[(s * NCTX + c) * COUNT + i];
          if (got != exp) {
            fprintf(stderr, "[rank %d] iter %d mismatch [src=%d][ctx=%d][%zu]: got %.0f exp %.0f\n",
                    rank, it, s, c, i, got, exp);
            ok = false;
            break;
          }
        }
      }
    }

    // Tear down the DevComm (and all NCTX*peer QP connections) before next iter.
    mori::cco::ccoDevCommDestroy(comm, &devComm);
    mori::cco::ccoBarrierAll(comm);
    if (rank == 0) printf("  [iter %d] %s\n", it, ok ? "PASS" : "FAIL");
  }

  HIP_CHECK(hipStreamDestroy(stream));
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA multi-context", "/tmp/cco_gda_multi_context_uid", 19881);
}
