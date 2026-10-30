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
// test: cco sdma putBroadcast + postWaitMem / postWaitSignal.
//
// Ranks form a ring: each rank sends to next=(rank+1)%n and receives from
// prev=(rank-1+n)%n. Parts:
//   A  putBroadcast: one source to {next, self} (and {next, next+1} for n >= 3),
//      coop x signal matrix x size.
//   B  postWaitMem: a put posted behind an engine-side wait on a flag. The kernel
//      returns right after posting; the data must NOT have moved until the host
//      raises the flag, and must all arrive afterwards. Default and Aggregate.
//   C  postWaitSignal: each rank's engine relays what it received from prev on to
//      next once prev's signal lands, with no CU in the forwarding path.
//   E  thread scope with every lane of a wave posting a wait to one queue: must
//      not deadlock (the lanes are serialized).
//   F  same, with putBroadcast: every lane broadcasts its own slice to {next, self}
//      on one queue; both destinations must hold every slice.
//   D  broadcast ring wrap: enough 3-unit broadcasts that one is placed on the
//      last unit of the ring, which must be padded rather than split.
//   G  postWaitMemPutPtr: a wait, a copy and a signal in one run, held behind a flag
//      like B (thread scope, Default and Aggregate).

#include "cco_test_harness.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t BUF_BYTES = 1ULL << 20;
static const int NUM_QUEUES = 8;

__host__ __device__ static inline uint32_t Pattern(int owner, size_t idx) {
  return (static_cast<uint32_t>(owner) << 20) | static_cast<uint32_t>(idx & 0xfffff);
}

using mori::cco::ccoCoopBlock;
using mori::cco::ccoCoopThread;
using mori::cco::ccoCoopWarp;
using mori::cco::ccoDevComm;
using mori::cco::ccoSdmaOptFlagsAggregate;
using mori::cco::ccoSdmaOptFlagsDefault;
using mori::cco::ccoWindow_t;
using mori::cco::ccoWindowDevice;

template <typename Coop>
static dim3 BlockFor() {
  return dim3(std::is_same_v<Coop, ccoCoopThread> ? 1 : 64);
}

/* ---------------------------------- part A ---------------------------------- */

template <typename Coop, bool LocalSig, bool RemoteSig>
__global__ void BroadcastKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                                ccoWindowDevice* recv2Win, size_t bytes, int qid, int twoRemote,
                                uint64_t expLocal, uint64_t expRemote1, uint64_t expRemote2,
                                ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int n = devComm.lsaSize, me = devComm.lsaRank;
  const int next = (me + 1) % n;
  const int peer2 = twoRemote ? (me + 2) % n : me;

  sdma.putBroadcast<Coop, LocalSig, RemoteSig>(next, reinterpret_cast<ccoWindow_t>(recvWin), 0,
                                               peer2, reinterpret_cast<ccoWindow_t>(recv2Win), 0,
                                               reinterpret_cast<ccoWindow_t>(sendWin), 0, bytes,
                                               qid);
  sdma.quiet<Coop>(next);
  if constexpr (LocalSig) sdma.waitSignal(me, qid, expLocal);
  if constexpr (RemoteSig) {
    sdma.waitSignal((me - 1 + n) % n, qid, expRemote1);
    if (twoRemote) sdma.waitSignal((me - 2 + n) % n, qid, expRemote2);
  }
}

/* ---------------------------------- part B ---------------------------------- */

template <typename Coop, uint32_t Flags>
__global__ void PostWaitPutKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                                  ccoWindowDevice* flagWin, size_t bytes, int qid,
                                  ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int next = (devComm.lsaRank + 1) % devComm.lsaSize;
  sdma.postWaitMem<Coop, Flags>(next, qid, reinterpret_cast<ccoWindow_t>(flagWin), 0, 1u);
  sdma.put<Coop, false, true, Flags>(next, reinterpret_cast<ccoWindow_t>(recvWin), 0,
                                     reinterpret_cast<ccoWindow_t>(sendWin), 0, bytes, qid);
  if constexpr (Flags & ccoSdmaOptFlagsAggregate) sdma.commit<Coop>(next, qid);
}

// Part G: postWaitMemPutPtr holds a copy and its remote signal behind the flag.
template <uint32_t Flags>
__global__ void PostWaitPutPtrKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                                     ccoWindowDevice* flagWin, size_t bytes, int qid,
                                     ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int next = (devComm.lsaRank + 1) % devComm.lsaSize;
  sdma.postWaitMemPutPtr<false, true, Flags>(
      next, qid, reinterpret_cast<ccoWindow_t>(flagWin), 0, 1u, ccoSdmaCmpEq, 0xFFFFFFFFu,
      ccoGetLsaPeerPtr(reinterpret_cast<ccoWindow_t>(recvWin), next, 0),
      ccoGetLocalPtr(reinterpret_cast<ccoWindow_t>(sendWin), 0), bytes);
  if constexpr (Flags & ccoSdmaOptFlagsAggregate) sdma.commit<ccoCoopThread>(next, qid);
}

__global__ void WaitIncomingKernel(int qid, uint64_t expected, ccoDevComm devComm) {
  mori::cco::ccoSdma sdma{devComm};
  const int n = devComm.lsaSize;
  sdma.waitSignal((devComm.lsaRank - 1 + n) % n, qid, expected);
}

/* ---------------------------------- part C ---------------------------------- */

__global__ void RelayKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                            ccoWindowDevice* fwdWin, size_t bytes, int hopQ, int relayQ,
                            uint64_t expHop, uint64_t expRelay, ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int n = devComm.lsaSize, me = devComm.lsaRank;
  const int next = (me + 1) % n, prev = (me - 1 + n) % n;

  // Engine side: once prev's hop signal lands, forward what it delivered.
  sdma.postWaitSignal(next, relayQ, prev, hopQ, expHop);
  sdma.put<ccoCoopThread, false, true>(next, reinterpret_cast<ccoWindow_t>(fwdWin), 0,
                                       reinterpret_cast<ccoWindow_t>(recvWin), 0, bytes, relayQ);
  // First hop, which is what releases the neighbour's wait.
  sdma.put<ccoCoopThread, false, true>(next, reinterpret_cast<ccoWindow_t>(recvWin), 0,
                                       reinterpret_cast<ccoWindow_t>(sendWin), 0, bytes, hopQ);
  sdma.waitSignal(prev, relayQ, expRelay);
}

/* ---------------------------------- part D ---------------------------------- */

constexpr int WRAP_ITERS = 9000;  // 3 units each: more than 3 laps of the 8192-unit ring

__global__ void BroadcastWrapKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                                    ccoWindowDevice* recv2Win, int qid, uint32_t* straddles,
                                    ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int n = devComm.lsaSize, me = devComm.lsaRank;
  const int next = (me + 1) % n;

  ccoSdmaQueueDeviceHandle h =
      ccoSdmaLoadQueueHandle(devComm.sdma.deviceHandles + next * devComm.sdma.sdmaNumQueue + qid);
  uint64_t pos = __hip_atomic_load(h.cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  constexpr uint64_t kRingUnits = CCO_SDMA_QUEUE_SIZE / CCO_SDMA_UNIT;
  uint32_t count = 0;
  for (int i = 0; i < WRAP_ITERS; i++, pos += 3 * CCO_SDMA_UNIT)
    count += (pos / CCO_SDMA_UNIT) % kRingUnits == kRingUnits - 1;
  *straddles = count;

  for (int i = 0; i < WRAP_ITERS; i++) {
    sdma.putBroadcast(next, reinterpret_cast<ccoWindow_t>(recvWin), 0, me,
                      reinterpret_cast<ccoWindow_t>(recv2Win), 0,
                      reinterpret_cast<ccoWindow_t>(sendWin), 0, 64, qid);
  }
  sdma.quiet(next);
}

/* ---------------------------------- part E ---------------------------------- */

// Every lane posts a wait to the same queue. The wait never blocks (Always), so
// this only returns if the lanes are serialized instead of racing in ReserveSlot.
__global__ void MultiLaneWaitKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                                    ccoWindowDevice* flagWin, size_t bytes, int qid,
                                    ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int next = (devComm.lsaRank + 1) % devComm.lsaSize;
  sdma.postWaitMem(next, qid, reinterpret_cast<ccoWindow_t>(flagWin), 0, 0, ccoSdmaCmpAlways);
  if (threadIdx.x == 0) {
    sdma.put(next, reinterpret_cast<ccoWindow_t>(recvWin), 0,
             reinterpret_cast<ccoWindow_t>(sendWin), 0, bytes, qid);
    sdma.quiet(next);
  }
}

/* ---------------------------------- part F ---------------------------------- */

constexpr int F_LANES = 32;
constexpr size_t F_SLICE = 128;  // bytes per lane

// Every lane broadcasts its own slice on the same queue. putBroadcast reserves a
// slack unit as well, so this pins the serialization for the 3-unit path.
__global__ void MultiLaneBroadcastKernel(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                                         ccoWindowDevice* recv2Win, int qid, ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int me = devComm.lsaRank, next = (me + 1) % devComm.lsaSize;
  const size_t off = static_cast<size_t>(threadIdx.x) * F_SLICE;
  sdma.putBroadcast(next, reinterpret_cast<ccoWindow_t>(recvWin), off, me,
                    reinterpret_cast<ccoWindow_t>(recv2Win), off,
                    reinterpret_cast<ccoWindow_t>(sendWin), off, F_SLICE, qid);
  if (threadIdx.x == 0) sdma.quiet(next);
}

/* ---------------------------------- driver ---------------------------------- */

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  g_rank = rank;
  using namespace mori::cco;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  void *sendBuf = nullptr, *recvBuf = nullptr, *recv2Buf = nullptr, *flagBuf = nullptr,
       *fwdBuf = nullptr;
  if (ccoMemAlloc(comm, BUF_BYTES, &sendBuf) || ccoMemAlloc(comm, BUF_BYTES, &recvBuf) ||
      ccoMemAlloc(comm, BUF_BYTES, &recv2Buf) || ccoMemAlloc(comm, BUF_BYTES, &flagBuf) ||
      ccoMemAlloc(comm, BUF_BYTES, &fwdBuf)) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  ccoWindow_t sendWin = nullptr, recvWin = nullptr, recv2Win = nullptr, flagWin = nullptr,
              fwdWin = nullptr;
  if (ccoWindowRegister(comm, sendBuf, BUF_BYTES, &sendWin) ||
      ccoWindowRegister(comm, recvBuf, BUF_BYTES, &recvWin) ||
      ccoWindowRegister(comm, recv2Buf, BUF_BYTES, &recv2Win) ||
      ccoWindowRegister(comm, flagBuf, BUF_BYTES, &flagWin) ||
      ccoWindowRegister(comm, fwdBuf, BUF_BYTES, &fwdWin)) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = reqs.gdaSignalCount = reqs.gdaCounterCount = 0;
  reqs.sdmaQueueCount = NUM_QUEUES;
  ccoDevComm devComm{};
  if (ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  bool ok = true;
  if (devComm.sdma.sdmaNumQueue < static_cast<uint32_t>(NUM_QUEUES)) {
    printf("[rank %d] SKIP — %u SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank,
           devComm.sdma.sdmaNumQueue);
    fflush(stdout);
  } else {
    const int n = nranks, me = rank;
    const int next = (me + 1) % n, prev = (me - 1 + n) % n, prev2 = (me - 2 + n) % n;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<uint32_t> hostSend(BUF_BYTES / 4);
    for (size_t i = 0; i < hostSend.size(); i++) hostSend[i] = Pattern(me, i);
    HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), BUF_BYTES, hipMemcpyHostToDevice));

    auto resetRecv = [&]() {
      HIP_CHECK(hipMemset(recvBuf, 0xff, BUF_BYTES));
      HIP_CHECK(hipMemset(recv2Buf, 0xff, BUF_BYTES));
      HIP_CHECK(hipMemset(fwdBuf, 0xff, BUF_BYTES));
      HIP_CHECK(hipMemset(flagBuf, 0, 4096));
      HIP_CHECK(hipDeviceSynchronize());
    };
    auto matches = [&](void* buf, size_t bytes, int owner, const char* what) {
      std::vector<uint32_t> host(bytes / 4);
      HIP_CHECK(hipMemcpy(host.data(), buf, bytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < host.size(); i++) {
        if (host[i] != Pattern(owner, i)) {
          fprintf(stderr, "[rank %d] %s: word %zu got 0x%08x expected 0x%08x\n", rank, what, i,
                  host[i], Pattern(owner, i));
          return false;
        }
      }
      return true;
    };
    auto untouched = [&](void* buf, size_t bytes) {
      std::vector<uint32_t> host(bytes / 4);
      HIP_CHECK(hipMemcpy(host.data(), buf, bytes, hipMemcpyDeviceToHost));
      for (uint32_t w : host)
        if (w != 0xffffffffu) return false;
      return true;
    };
    // Cumulative signal bookkeeping, one running total per (source offset, queue).
    uint64_t sigLocal[NUM_QUEUES] = {}, sigR1[NUM_QUEUES] = {}, sigR2[NUM_QUEUES] = {};

    /* ---- A ---- */
    int aPass = 0, aTotal = 0;
    auto runBroadcast = [&](auto coop, auto localSig, auto remoteSig, size_t bytes, int twoRemote,
                            int q) {
      using Coop = decltype(coop);
      constexpr bool L = decltype(localSig)::value, R = decltype(remoteSig)::value;
      resetRecv();
      ccoBarrierAll(comm);
      if (L) sigLocal[q]++;
      if (R) {
        sigR1[q]++;
        if (twoRemote) sigR2[q]++;
      }
      BroadcastKernel<Coop, L, R>
          <<<1, BlockFor<Coop>(), 0, stream>>>(sendWin, recvWin, recv2Win, bytes, q, twoRemote,
                                               sigLocal[q], sigR1[q], sigR2[q], devComm);
      HIP_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      bool good = matches(recvBuf, bytes & ~3ull, prev, "broadcast dst1");
      good = good && matches(recv2Buf, bytes & ~3ull, twoRemote ? prev2 : me, "broadcast dst2");
      aTotal++;
      aPass += good;
      if (!good)
        fprintf(stderr, "[rank %d] A FAIL coop=%s L=%d R=%d bytes=%zu twoRemote=%d q=%d\n", rank,
                std::is_same_v<Coop, ccoCoopThread> ? "thread"
                : std::is_same_v<Coop, ccoCoopWarp> ? "warp"
                                                    : "block",
                L, R, bytes, twoRemote, q);
    };
    using T = std::true_type;
    using F = std::false_type;
    int qi = 0;
    for (size_t bytes : {size_t(100), size_t(4096), BUF_BYTES}) {
      auto allSigs = [&](auto coop) {
        runBroadcast(coop, F{}, F{}, bytes, 0, qi++ % NUM_QUEUES);
        runBroadcast(coop, T{}, F{}, bytes, 0, qi++ % NUM_QUEUES);
        runBroadcast(coop, F{}, T{}, bytes, 0, qi++ % NUM_QUEUES);
        runBroadcast(coop, T{}, T{}, bytes, 0, qi++ % NUM_QUEUES);
      };
      allSigs(ccoCoopThread{});
      allSigs(ccoCoopWarp{});
      allSigs(ccoCoopBlock{});
      if (n >= 3) {
        runBroadcast(ccoCoopThread{}, F{}, T{}, bytes, 1, qi++ % NUM_QUEUES);
        runBroadcast(ccoCoopWarp{}, T{}, T{}, bytes, 1, qi++ % NUM_QUEUES);
      }
    }
    const bool okA = aPass == aTotal;

    /* ---- B ---- */
    int bPass = 0, bTotal = 0;
    auto runWait = [&](auto coop, auto flags, int q) {
      using Coop = decltype(coop);
      constexpr uint32_t Flags = decltype(flags)::value;
      resetRecv();
      ccoBarrierAll(comm);
      // Baseline rather than A's per-queue totals, so B holds at any rank count.
      uint64_t sigBefore = 0;
      HIP_CHECK(hipMemcpy(&sigBefore, devComm.sdma.signalBuf + prev * devComm.sdma.sdmaNumQueue + q,
                          8, hipMemcpyDeviceToHost));
      ccoBarrierAll(comm);
      PostWaitPutKernel<Coop, Flags>
          <<<1, BlockFor<Coop>(), 0, stream>>>(sendWin, recvWin, flagWin, BUF_BYTES, q, devComm);
      HIP_CHECK(hipStreamSynchronize(stream));  // posted; the CU is out of the picture now
      ccoBarrierAll(comm);
      usleep(100 * 1000);
      uint64_t sig = 0;
      HIP_CHECK(hipMemcpy(&sig, devComm.sdma.signalBuf + prev * devComm.sdma.sdmaNumQueue + q, 8,
                          hipMemcpyDeviceToHost));
      const bool held = untouched(recvBuf, BUF_BYTES) && sig == sigBefore;
      ccoBarrierAll(comm);
      const uint32_t one = 1;
      HIP_CHECK(hipMemcpy(flagBuf, &one, 4, hipMemcpyHostToDevice));
      WaitIncomingKernel<<<1, 1, 0, stream>>>(q, sigBefore + 1, devComm);
      HIP_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      const bool landed = matches(recvBuf, BUF_BYTES, prev, "waited put");
      bTotal++;
      bPass += held && landed;
      if (!(held && landed))
        fprintf(stderr, "[rank %d] B FAIL coop=%s agg=%d held=%d landed=%d\n", rank,
                std::is_same_v<Coop, ccoCoopThread> ? "thread" : "warp",
                (int)(Flags & ccoSdmaOptFlagsAggregate), held, landed);
    };
    using Def = std::integral_constant<uint32_t, ccoSdmaOptFlagsDefault>;
    using Agg = std::integral_constant<uint32_t, ccoSdmaOptFlagsAggregate>;
    runWait(ccoCoopThread{}, Def{}, 0);
    runWait(ccoCoopThread{}, Agg{}, 1);
    runWait(ccoCoopWarp{}, Def{}, 2);
    runWait(ccoCoopWarp{}, Agg{}, 3);
    const bool okB = bPass == bTotal;

    /* ---- C ---- */
    resetRecv();
    ccoBarrierAll(comm);
    // Queues 6 and 7 were already signalled by part A: wait for one more on each.
    uint64_t base6 = 0, base7 = 0;  // baselines, as in B
    HIP_CHECK(hipMemcpy(&base6, devComm.sdma.signalBuf + prev * devComm.sdma.sdmaNumQueue + 6, 8,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&base7, devComm.sdma.signalBuf + prev * devComm.sdma.sdmaNumQueue + 7, 8,
                        hipMemcpyDeviceToHost));
    ccoBarrierAll(comm);
    RelayKernel<<<1, 1, 0, stream>>>(sendWin, recvWin, fwdWin, BUF_BYTES, 6, 7, base6 + 1,
                                     base7 + 1, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
    const bool okC = matches(recvBuf, BUF_BYTES, prev, "relay hop") &&
                     matches(fwdBuf, BUF_BYTES, prev2, "relay forward");

    /* ---- D ---- */
    resetRecv();
    uint32_t* straddleDev = nullptr;
    HIP_CHECK(hipMalloc(&straddleDev, sizeof(uint32_t)));
    ccoBarrierAll(comm);
    BroadcastWrapKernel<<<1, 1, 0, stream>>>(sendWin, recvWin, recv2Win, 5, straddleDev, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
    uint32_t straddles = 0;
    HIP_CHECK(hipMemcpy(&straddles, straddleDev, sizeof(straddles), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(straddleDev));
    std::vector<uint32_t> host(16);
    HIP_CHECK(hipMemcpy(host.data(), recvBuf, 64, hipMemcpyDeviceToHost));
    bool okD = straddles >= 1;
    for (size_t i = 0; i < 16; i++) okD = okD && host[i] == Pattern(prev, i);
    HIP_CHECK(hipMemcpy(host.data(), recv2Buf, 64, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < 16; i++) okD = okD && host[i] == Pattern(me, i);

    /* ---- E ---- */
    resetRecv();
    ccoBarrierAll(comm);
    MultiLaneWaitKernel<<<1, 32, 0, stream>>>(sendWin, recvWin, flagWin, BUF_BYTES, 4, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
    const bool okE = matches(recvBuf, BUF_BYTES, prev, "multi-lane wait");

    /* ---- G ---- */
    int gPass = 0, gTotal = 0;
    auto runWaitPtr = [&](auto flags, int q) {
      constexpr uint32_t Flags = decltype(flags)::value;
      resetRecv();
      ccoBarrierAll(comm);
      uint64_t sigBefore = 0;
      HIP_CHECK(hipMemcpy(&sigBefore, devComm.sdma.signalBuf + prev * devComm.sdma.sdmaNumQueue + q,
                          8, hipMemcpyDeviceToHost));
      ccoBarrierAll(comm);
      PostWaitPutPtrKernel<Flags>
          <<<1, 1, 0, stream>>>(sendWin, recvWin, flagWin, BUF_BYTES, q, devComm);
      HIP_CHECK(hipStreamSynchronize(stream));  // posted; the CU is out of the picture now
      ccoBarrierAll(comm);
      usleep(100 * 1000);
      uint64_t sig = 0;
      HIP_CHECK(hipMemcpy(&sig, devComm.sdma.signalBuf + prev * devComm.sdma.sdmaNumQueue + q, 8,
                          hipMemcpyDeviceToHost));
      const bool held = untouched(recvBuf, BUF_BYTES) && sig == sigBefore;
      ccoBarrierAll(comm);
      const uint32_t one = 1;
      HIP_CHECK(hipMemcpy(flagBuf, &one, 4, hipMemcpyHostToDevice));
      WaitIncomingKernel<<<1, 1, 0, stream>>>(q, sigBefore + 1, devComm);
      HIP_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      const bool landed = matches(recvBuf, BUF_BYTES, prev, "waitMemPutPtr");
      gTotal++;
      gPass += held && landed;
      if (!(held && landed))
        fprintf(stderr, "[rank %d] G FAIL agg=%d held=%d landed=%d\n", rank,
                (int)(Flags & ccoSdmaOptFlagsAggregate), held, landed);
    };
    runWaitPtr(Def{}, 5);
    runWaitPtr(Agg{}, 6);
    const bool okG = gPass == gTotal;

    /* ---- F ---- */
    resetRecv();
    ccoBarrierAll(comm);
    MultiLaneBroadcastKernel<<<1, F_LANES, 0, stream>>>(sendWin, recvWin, recv2Win, 4, devComm);
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
    constexpr size_t kFBytes = F_LANES * F_SLICE;
    bool okF = matches(recvBuf, kFBytes, prev, "multi-lane broadcast (peer)") &&
               matches(recv2Buf, kFBytes, me, "multi-lane broadcast (self)");
    {
      // nothing may land past the last slice
      std::vector<uint32_t> tail(16);
      for (void* buf : {recvBuf, recv2Buf}) {
        HIP_CHECK(
            hipMemcpy(tail.data(), static_cast<char*>(buf) + kFBytes, 64, hipMemcpyDeviceToHost));
        for (uint32_t w : tail) okF = okF && w == 0xffffffffu;
      }
    }

    ok = okA && okB && okC && okD && okE && okF && okG;
    HIP_CHECK(hipStreamDestroy(stream));
    printf(
        "[rank %d] A(broadcast)=%s(%d/%d) B(waitMem)=%s(%d/%d) C(relay)=%s "
        "D(wrap, %u straddles)=%s E(multi-lane wait)=%s F(multi-lane bcast)=%s "
        "G(waitMemPutPtr)=%s(%d/%d) %s\n",
        rank, okA ? "PASS" : "FAIL", aPass, aTotal, okB ? "PASS" : "FAIL", bPass, bTotal,
        okC ? "PASS" : "FAIL", straddles, okD ? "PASS" : "FAIL", okE ? "PASS" : "FAIL",
        okF ? "PASS" : "FAIL", okG ? "PASS" : "FAIL", gPass, gTotal, ok ? "PASSED" : "FAILED");
    fflush(stdout);
  }

  ccoDevCommDestroy(comm, &devComm);
  ccoWindowDeregister(comm, fwdWin);
  ccoWindowDeregister(comm, flagWin);
  ccoWindowDeregister(comm, recv2Win);
  ccoWindowDeregister(comm, recvWin);
  ccoWindowDeregister(comm, sendWin);
  ccoMemFree(comm, fwdBuf);
  ccoMemFree(comm, flagBuf);
  ccoMemFree(comm, recv2Buf);
  ccoMemFree(comm, recvBuf);
  ccoMemFree(comm, sendBuf);
  ccoCommDestroy(comm);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO SDMA wait/broadcast", "/tmp/cco_sdma_wait_bcast_uid", 19893);
}
