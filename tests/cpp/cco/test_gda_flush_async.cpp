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
// test: cco gda flushAsync + wait — overlap doorbell ring with peer-signal wait.
//
// each rank launches one kernel that:
//   1. put(peer, ..., SignalInc{myRank}) to every peer
//   2. flushAsync(peer, &req[w]) — warp w's lane 0 rings doorbell, keeps the handle
//   3. waitSignal on every incoming signal (data+signal are ordered on the same qp,
//      so a landed signal implies the data write also landed)
//   4. wait(req[w]) — warp w's lane 0 drains its own cq, confirming the outbound
//      put has completed locally and sendBuf is reusable
//
// overlap pattern: between step 2 (doorbell rung) and step 4 (cq drained), step 3
// blocks on remote signal landings — our outbound WQEs are in flight on the nic
// during that window, so wait() in step 4 usually returns immediately.
//
// difference from test_cco_gda_put:
//   - that test calls flush() (rings doorbell AND polls cq, fused)
//   - this test splits into flushAsync(ring only) → waitSignal → wait(poll only),
//     letting the cq-poll be hidden behind the signal wait

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;  // elements per rank-pair

// force psd (ionic) provider
static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::PSD;

// alltoall kernel using flushAsync — one warp per peer for the doorbell ring.
// signal layout: signal[r] is incremented by peer r.
//
// block layout: blockDim.x = nRanks * warpSize. warp w owns peer w:
//   - step 2: warp w's lane 0 calls flushAsync(peer=w) — different warps
//     fire doorbells in parallel rather than serialized on thread 0
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaAlltoAllFlushAsyncKernel(mori::cco::ccoWindowDevice* sendWin,
                                            mori::cco::ccoWindowDevice* recvWin, size_t count,
                                            mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;

  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int warpId = tid / warpSize;
  int laneId = tid % warpSize;

  size_t perPairBytes = count * sizeof(T);

  // per-thread handle slot. only lane 0 of each "active" warp writes it in step 2
  // and reads it in step 4; other threads leave it null.
  ccoGdaRequest_t myReq{};

  // step 1: each thread issues put to a distinct peer
  for (int r = tid; r < nRanks; r += nthreads) {
    if (r == myRank) continue;
    gda.put(r, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPairBytes,
            reinterpret_cast<ccoWindow_t>(sendWin), r * perPairBytes, perPairBytes,
            ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank)});
  }
  __syncthreads();

  // step 2: warp w's lane 0 rings doorbell for peer w — parallel across warps.
  if (laneId == 0 && warpId < nRanks && warpId != myRank) {
    gda.flushAsync(warpId, &myReq);
    gda.wait(myReq);
  }

  // step 3: wait for every peer's signal. since data+signal share the qp and
  // are ordered, a landed signal implies the data write also landed.
  if (tid < nRanks && tid != myRank) {
    gda.waitSignal(static_cast<ccoGdaSignal_t>(tid), 1);
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

  // setup: comm, send/recv buffers, windows
  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  size_t bufSize = COUNT * nranks * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  // sendBuf[r*COUNT + i] = rank*1000 + r*100 + i — encodes (sender, dest-slot, idx)
  std::vector<float> hostSend(COUNT * nranks);
  for (int r = 0; r < nranks; r++) {
    for (size_t i = 0; i < COUNT; i++) {
      hostSend[r * COUNT + i] = static_cast<float>(rank * 1000 + r * 100 + i);
    }
  }
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(recvBuf, 0xff, bufSize));

  mori::cco::ccoWindow_t sendWin = nullptr;
  mori::cco::ccoWindow_t recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // devcomm: full gda connectivity + one signal per peer
  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  printf("[rank %d] DevCommCreate OK (worldSize=%d, lsaSize=%d, gdaConnType=%d, numQpPerPe=%d)\n",
         rank, devComm.worldSize, devComm.lsaSize, (int)devComm.gdaConnType,
         devComm.ibgda.numQpPerPe);

  if (devComm.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE — check peer mask / rdma support\n",
            rank);
    return 1;
  }

  mori::cco::ccoBarrierAll(comm);

  // launch — one warp per peer so flushAsync calls fan out across warps.
  // AMD warpSize = 64; block max = 1024 threads, so up to 16 peers.
  constexpr int kWarpSize = 64;
  if (nranks * kWarpSize > 1024) {
    fprintf(stderr, "[rank %d] nranks=%d exceeds 1024/%d warp budget\n", rank, nranks, kWarpSize);
    return 1;
  }
  int blockDim = nranks * kWarpSize;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  GdaAlltoAllFlushAsyncKernel<kPrvdType, float>
      <<<1, blockDim, 0, stream>>>(sendWin, recvWin, COUNT, devComm);
  HIP_CHECK(hipStreamSynchronize(stream));
  printf("[rank %d] kernel completed\n", rank);

  mori::cco::ccoBarrierAll(comm);

  // verify: recv[src*COUNT + i] should be src*1000 + rank*100 + i for every src != rank
  std::vector<float> hostRecv(COUNT * nranks);
  HIP_CHECK(hipMemcpy(hostRecv.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));

  bool ok = true;
  for (int srcRank = 0; srcRank < nranks && ok; srcRank++) {
    if (srcRank == rank) continue;
    for (size_t i = 0; i < COUNT; i++) {
      float expected = static_cast<float>(srcRank * 1000 + rank * 100 + i);
      float got = hostRecv[srcRank * COUNT + i];
      if (got != expected) {
        fprintf(stderr, "[rank %d] mismatch at [src=%d][%zu]: got %.0f expected %.0f\n", rank,
                srcRank, i, got, expected);
        ok = false;
        break;
      }
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA flushAsync", "/tmp/cco_gda_flush_async_uid", 19878);
}
