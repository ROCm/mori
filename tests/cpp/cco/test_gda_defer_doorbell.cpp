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
//
// test: cco gda ccoGdaOptFlagsDeferDoorbell and doorbell coalescing.
//
// A deferred post writes its WQEs and marks them ready without ringing; the next
// non-deferred post, signal, flush or counter on the QP rings for them. Each case
// is one kernel, then a host barrier and a check of what landed:
//
//   probe     one warp, one peer: deferred puts leave dbTouchIdx where it was
//             while postIdx moves; flush brings dbTouchIdx up to postIdx.
//   flush     every warp defers its puts to every peer, then flushes.
//   signal    deferred puts released by a plain signal, no flush: the receiver
//             waits for the signal, which the QP orders after the data.
//   mixed     many blocks, thread-scope puts (lanes grouped by peer), even warps
//             defer and odd warps do not -- the contended, coalescing case.
//   overflow  one warp defers more putValues than the SQ holds: the SQ space wait
//             must ring for them itself or it deadlocks.
//   counter   deferred puts drained by counter() instead of flush.
//   get       deferred gets, then flush.

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 512ULL * 1024 * 1024;
static constexpr int kChunks = 256;          // chunks per (src, dst) pair
static constexpr int kChunkElems = 32;       // uint64 per chunk (256 B)
static constexpr int kOverflowPuts = 16384;  // > any SQ depth here (bnxt 4096, mlx5 1024)
// Per-source region of the receive buffer, sized for the largest case.
static constexpr size_t kRegionElems = kOverflowPuts;
static_assert(kChunks * kChunkElems <= kRegionElems, "region too small");

enum Case : int { kProbe, kFlush, kSignal, kMixed, kOverflow, kCounter, kGet, kNumCases };
static const char* kCaseNames[] = {"probe",    "flush",   "signal", "mixed",
                                   "overflow", "counter", "get"};

__host__ __device__ inline uint64_t Value(int src, int dst, int chunk, int i) {
  return (static_cast<uint64_t>(src) << 48) | (static_cast<uint64_t>(dst) << 40) |
         (static_cast<uint64_t>(chunk) << 16) | static_cast<uint64_t>(i);
}

constexpr uint32_t kDefer = ccoGdaOptFlagsDeferDoorbell;

template <mori::core::ProviderType P>
__device__ inline mori::core::RdmaEndpointDevice* Endpoint(const ccoDevComm& dc, int peer) {
  return &dc.ibgda.endpoints[peer * dc.ibgda.numQpPerPe];  // context 0
}

// probe: out[0..3] = dbTouchIdx before, dbTouchIdx after the deferred puts,
// postIdx after them, dbTouchIdx after flush.
template <mori::core::ProviderType P>
__global__ void ProbeKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc,
                            uint32_t* out) {
  ccoGda<P> gda{dc, 0};
  const int me = dc.rank;
  const int peer = (me + 1) % dc.worldSize;
  auto* wq = &Endpoint<P>(dc, peer)->wqHandle;
  const size_t chunkBytes = kChunkElems * sizeof(uint64_t);
  if (threadIdx.x == 0)
    out[0] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  for (int c = 0; c < kChunks; c++) {
    gda.template put<CCO_TEAM_WORLD, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopWarp, kDefer>(
        peer, reinterpret_cast<ccoWindow_t>(recvWin),
        (me * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
        reinterpret_cast<ccoWindow_t>(sendWin),
        (peer * kRegionElems + c * kChunkElems) * sizeof(uint64_t), chunkBytes, ccoGda_NoSignal{},
        ccoCoopWarp{});
  }
  if (threadIdx.x == 0) {
    out[1] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    out[2] = __hip_atomic_load(&wq->postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  gda.flush(peer, ccoCoopWarp{});
  if (threadIdx.x == 0)
    out[3] = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// flush / signal / counter / get: warp-scope ops, chunks striped over all warps
// of the grid, every peer.
template <mori::core::ProviderType P, int C>
__global__ void ChunkKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  ccoGda<P> gda{dc, 0};
  const int me = dc.rank, n = dc.worldSize;
  const int warpsPerBlock = blockDim.x / warpSize;
  const int gw = blockIdx.x * warpsPerBlock + threadIdx.x / warpSize;
  const int nw = gridDim.x * warpsPerBlock;
  const size_t chunkBytes = kChunkElems * sizeof(uint64_t);
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);

  for (int k = 1; k < n; k++) {
    const int peer = (me + k) % n;
    for (int c = gw; c < kChunks; c += nw) {
      if constexpr (C == kGet) {
        // Pull peer's chunk for me into my region for peer.
        gda.template get<CCO_TEAM_WORLD, ccoGdaThreadIndependent, ccoCoopWarp, kDefer>(
            peer, send, (me * kRegionElems + c * kChunkElems) * sizeof(uint64_t), recv,
            (peer * kRegionElems + c * kChunkElems) * sizeof(uint64_t), chunkBytes, ccoCoopWarp{});
      } else {
        gda.template put<CCO_TEAM_WORLD, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopWarp,
                         kDefer>(peer, recv,
                                 (me * kRegionElems + c * kChunkElems) * sizeof(uint64_t), send,
                                 (peer * kRegionElems + c * kChunkElems) * sizeof(uint64_t),
                                 chunkBytes, ccoGda_NoSignal{}, ccoCoopWarp{});
      }
    }
  }

  if constexpr (C == kSignal) {
    // No flush: one signal per (warp, peer) is the only non-deferred post.
    for (int k = 1; k < n; k++) {
      const int peer = (me + k) % n;
      gda.signal(peer, ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(me)}, ccoCoopWarp{});
    }
    // Every warp of every peer signals me once.
    if (blockIdx.x == 0 && threadIdx.x < n && static_cast<int>(threadIdx.x) != me) {
      gda.waitSignal(static_cast<ccoGdaSignal_t>(threadIdx.x), static_cast<uint64_t>(nw));
    }
    __syncthreads();
    gda.flush(ccoCoopWarp{});  // drain our own CQs before the kernel exits
  } else if constexpr (C == kCounter) {
    gda.counter(ccoGda_CounterInc{0}, ccoCoopWarp{});
  } else {
    gda.flush(ccoCoopWarp{});
  }
}

// mixed: one lane per (peer, chunk) item, thread scope, so a warp's lanes are
// grouped by peer and many blocks hit each QP. Even warps defer, odd ones ring.
template <mori::core::ProviderType P>
__global__ void MixedKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, ccoDevComm dc) {
  ccoGda<P> gda{dc, 0};
  const int me = dc.rank, n = dc.worldSize;
  const int items = (n - 1) * kChunks;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  const bool defer = ((tid / warpSize) & 1) == 0;
  const size_t chunkBytes = kChunkElems * sizeof(uint64_t);
  auto send = reinterpret_cast<ccoWindow_t>(sendWin);
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);

  for (int it = tid; it < items; it += nthreads) {
    const int peer = (me + 1 + it % (n - 1)) % n;
    const int c = it / (n - 1);
    const size_t dstOff = (me * kRegionElems + c * kChunkElems) * sizeof(uint64_t);
    const size_t srcOff = (peer * kRegionElems + c * kChunkElems) * sizeof(uint64_t);
    if (defer) {
      gda.template put<CCO_TEAM_WORLD, ccoGdaThreadIndependent, ccoGda_NoSignal, ccoCoopThread,
                       kDefer>(peer, recv, dstOff, send, srcOff, chunkBytes);
    } else {
      gda.put(peer, recv, dstOff, send, srcOff, chunkBytes);
    }
  }
  __syncthreads();
  gda.flush(ccoCoopWarp{});
}

// overflow: warp w of block 0 owns peer (me + 1 + w); kOverflowPuts deferred
// 8-byte putValues to it, then one flush.
template <mori::core::ProviderType P>
__global__ void OverflowKernel(ccoWindowDevice* recvWin, ccoDevComm dc) {
  ccoGda<P> gda{dc, 0};
  const int me = dc.rank, n = dc.worldSize;
  const int w = threadIdx.x / warpSize;
  if (w >= n - 1) return;
  const int peer = (me + 1 + w) % n;
  auto recv = reinterpret_cast<ccoWindow_t>(recvWin);
  for (int i = 0; i < kOverflowPuts; i++) {
    gda.template putValue<CCO_TEAM_WORLD, ccoGdaThreadIndependent, uint64_t, ccoGda_NoSignal,
                          ccoCoopWarp, kDefer>(
        peer, recv, (me * kRegionElems + i) * sizeof(uint64_t), Value(me, peer, 0, i),
        ccoGda_NoSignal{}, ccoCoopWarp{});
  }
  gda.flush(peer, ccoCoopWarp{});
}

static bool Verify(int c, int rank, int nranks, const std::vector<uint64_t>& recv) {
  int bad = 0;
  for (int src = 0; src < nranks; src++) {
    if (src == rank) continue;
    const uint64_t* r = recv.data() + src * kRegionElems;
    if (c == kOverflow) {
      for (int i = 0; i < kOverflowPuts; i++) {
        uint64_t want = Value(src, rank, 0, i);
        if (r[i] != want && bad++ < 4)
          fprintf(stderr, "[rank %d] %s: src %d [%d] got %#lx want %#lx\n", rank, kCaseNames[c],
                  src, i, r[i], want);
      }
      continue;
    }
    for (int ch = 0; ch < kChunks; ch++) {
      for (int i = 0; i < kChunkElems; i++) {
        // get pulls the peer's data for me; every other case pushes mine to the peer.
        uint64_t want = Value(src, rank, ch, i);
        uint64_t got = r[ch * kChunkElems + i];
        if (got != want && bad++ < 4)
          fprintf(stderr, "[rank %d] %s: src %d chunk %d [%d] got %#lx want %#lx\n", rank,
                  kCaseNames[c], src, ch, i, got, want);
      }
    }
  }
  return bad == 0;
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // the fork harness's children leave via _exit
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) return 1;

  const size_t bufElems = kRegionElems * nranks;
  const size_t bufBytes = bufElems * sizeof(uint64_t);
  void *sendBuf = nullptr, *recvBuf = nullptr;
  if (ccoMemAlloc(comm, bufBytes, &sendBuf) != 0 || ccoMemAlloc(comm, bufBytes, &recvBuf) != 0)
    return 1;

  // send[dst region][chunk][i] = Value(me, dst, chunk, i). For get, a peer pulls
  // from my send region for it, which holds the same value: Value(me, peer, ...).
  std::vector<uint64_t> hostSend(bufElems, 0);
  for (int dst = 0; dst < nranks; dst++)
    for (int ch = 0; ch < kChunks; ch++)
      for (int i = 0; i < kChunkElems; i++)
        hostSend[dst * kRegionElems + ch * kChunkElems + i] = Value(rank, dst, ch, i);
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), bufBytes, hipMemcpyHostToDevice));

  ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (ccoWindowRegister(comm, sendBuf, bufBytes, &sendWin) != 0 ||
      ccoWindowRegister(comm, recvBuf, bufBytes, &recvWin) != 0)
    return 1;

  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;
  reqs.gdaCounterCount = 1;
  ccoDevComm devComm{};
  if (ccoDevCommCreate(comm, &reqs, &devComm) != 0) return 1;
  if (devComm.gdaConnType == CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE\n", rank);
    return 1;
  }

  uint32_t* probe = nullptr;
  HIP_CHECK(hipMalloc(&probe, 4 * sizeof(uint32_t)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  auto* sw = reinterpret_cast<ccoWindowDevice*>(sendWin);
  auto* rw = reinterpret_cast<ccoWindowDevice*>(recvWin);

  bool allOk = true;
  for (int c = 0; c < kNumCases; c++) {
    HIP_CHECK(hipMemset(recvBuf, 0xff, bufBytes));
    HIP_CHECK(hipDeviceSynchronize());
    ccoBarrierAll(comm);
    switch (c) {
      case kProbe:
        CCO_GDA_DISPATCH(ProbeKernel<P><<<1, 64, 0, stream>>>(sw, rw, devComm, probe));
        break;
      case kFlush:
        CCO_GDA_DISPATCH(ChunkKernel<P, kFlush><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kSignal:
        // Signals accumulate across runs only within this case; 1 block keeps
        // the expected count (warps per grid) simple.
        CCO_GDA_DISPATCH(ChunkKernel<P, kSignal><<<1, 512, 0, stream>>>(sw, rw, devComm));
        break;
      case kMixed:
        CCO_GDA_DISPATCH(MixedKernel<P><<<16, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kOverflow:
        CCO_GDA_DISPATCH(OverflowKernel<P><<<1, 64 * (nranks - 1), 0, stream>>>(rw, devComm));
        break;
      case kCounter:
        CCO_GDA_DISPATCH(ChunkKernel<P, kCounter><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
      case kGet:
        CCO_GDA_DISPATCH(ChunkKernel<P, kGet><<<4, 256, 0, stream>>>(sw, rw, devComm));
        break;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);

    std::vector<uint64_t> hostRecv(bufElems);
    HIP_CHECK(hipMemcpy(hostRecv.data(), recvBuf, bufBytes, hipMemcpyDeviceToHost));
    bool ok = true;
    if (c == kProbe) {
      // Only the probed pair is filled; check the doorbell indices, then the
      // data rank-1 sent me.
      uint32_t p[4];
      HIP_CHECK(hipMemcpy(p, probe, sizeof(p), hipMemcpyDeviceToHost));
      if (p[1] != p[0] || p[3] != p[2] || p[2] - p[0] != static_cast<uint32_t>(kChunks)) {
        fprintf(stderr,
                "[rank %d] probe: dbTouchIdx %u -> %u after deferred puts (want unchanged), "
                "postIdx %u, dbTouchIdx after flush %u (want == postIdx)\n",
                rank, p[0], p[1], p[2], p[3]);
        ok = false;
      }
      const int src = (rank - 1 + nranks) % nranks;
      for (int ch = 0; ch < kChunks && ok; ch++)
        for (int i = 0; i < kChunkElems && ok; i++)
          if (hostRecv[src * kRegionElems + ch * kChunkElems + i] != Value(src, rank, ch, i)) {
            fprintf(stderr, "[rank %d] probe: data from %d chunk %d [%d] wrong\n", rank, src, ch,
                    i);
            ok = false;
          }
    } else {
      ok = Verify(c, rank, nranks, hostRecv);
    }
    printf("[rank %d] %-8s %s\n", rank, kCaseNames[c], ok ? "PASS" : "FAIL");
    allOk = allOk && ok;
  }

  ccoBarrierAll(comm);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(probe));
  ccoDevCommDestroy(comm, &devComm);
  ccoWindowDeregister(comm, sendWin);
  ccoWindowDeregister(comm, recvWin);
  ccoMemFree(comm, sendBuf);
  ccoMemFree(comm, recvBuf);
  ccoCommDestroy(comm);
  printf("[rank %d] %s\n", rank, allOk ? "PASSED" : "FAILED");
  return allOk ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA defer doorbell", "/tmp/cco_gda_defer_doorbell_uid", 0);
}
