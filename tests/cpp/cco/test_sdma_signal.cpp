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
// test: cco sdma quiet(rptr) + put local/remote signal + waitSignal.
//
// Ranks form a ring: each rank puts to peer=(rank+1)%n and receives from
// src=(rank-1+n)%n. Four parts:
//   A  queueId x scope x aggregate x size matrix, no signals — proves quiet()
//      drains via rptr for every shape (host-verified after a barrier).
//   B  remote signal only: N puts with remoteSignal, receiver waitSignal(src)
//      then verifies IN-KERNEL with NO barrier — proves the signal alone
//      guarantees the data landed at the peer.
//   C  local signal only: waitSignal(self) returns; the peer's remote slot is
//      untouched (src-dimension slots don't collide).
//   D  both signals (128B slot): local and remote slots each advance by exactly N.
// Every part also calls quiet() after signals that target the peer — the core
// requirement that a remote-targeted signal no longer deadlocks the sender.

#include "cco_test_harness.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t MAX_CHUNK = 1ULL << 20;  // 1MB
static const int NPUT = 4;                   // chunks per signal test
static const size_t BUF_BYTES = MAX_CHUNK * NPUT;

// sendBuf word i of rank r == (r << 20) | (i & 0xfffff)
__host__ __device__ static inline uint32_t Pattern(int owner, size_t idx) {
  return (static_cast<uint32_t>(owner) << 20) | static_cast<uint32_t>(idx & 0xfffff);
}

/* ------------------------------- part A: matrix ------------------------------ */

template <typename Coop, uint32_t Flags>
__global__ void SdmaMatrixKernel(mori::cco::ccoWindowDevice* sendWin,
                                 mori::cco::ccoWindowDevice* recvWin, size_t bytes, int qid,
                                 mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int peer = (devComm.lsaRank + 1) % devComm.lsaSize;

  sdma.put<Coop, false, false, Flags>(peer, reinterpret_cast<ccoWindow_t>(recvWin), 0,
                                      reinterpret_cast<ccoWindow_t>(sendWin), 0, bytes, qid);
  if constexpr (Flags & ccoSdmaOptFlagsAggregate) sdma.commit<Coop>(peer, qid);
  sdma.quiet<Coop>(peer);
}

/* ------------------------- parts B/C/D: signal + waitSignal ------------------ */

// Result slots written by the kernel so the host can report without a barrier.
struct SignalResult {
  uint32_t dataOk;      // in-kernel verify of the received chunks
  uint32_t badIdx;      // first mismatching word (valid when dataOk == 0)
  uint64_t localSlot;   // signalBuf[myLsaRank*n + q] observed after the waits
  uint64_t remoteSlot;  // signalBuf[src*n    + q] observed after the waits
};

// localSignal/remoteSignal are compile-time; expLocal/expRemote are the caller's
// cumulative counts (slots are never reset).
template <bool LocalSig, bool RemoteSig>
__global__ void SdmaSignalKernel(mori::cco::ccoWindowDevice* sendWin,
                                 mori::cco::ccoWindowDevice* recvWin, const uint32_t* recvLocal,
                                 size_t bytes, int qid, uint64_t expLocal, uint64_t expRemote,
                                 int verifyInKernel, SignalResult* out,
                                 mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int n = devComm.lsaSize;
  // Ring over LSA ranks: signal slots and put()'s peer are both LSA-indexed.
  const int peer = (devComm.lsaRank + 1) % n;
  const int src = (devComm.lsaRank - 1 + n) % n;

  for (int i = 0; i < NPUT; i++) {
    sdma.put<ccoCoopThread, LocalSig, RemoteSig>(peer, reinterpret_cast<ccoWindow_t>(recvWin),
                                                 i * bytes, reinterpret_cast<ccoWindow_t>(sendWin),
                                                 i * bytes, bytes, qid);
  }
  if constexpr (LocalSig) sdma.waitSignal(devComm.lsaRank, qid, expLocal);
  if constexpr (RemoteSig) sdma.waitSignal(src, qid, expRemote);

  // No barrier here: for RemoteSig the incoming waitSignal above is the only
  // thing standing between us and reading the peer-written data.
  out->dataOk = 1;
  out->badIdx = 0;
  if (verifyInKernel) {
    const size_t words = bytes / sizeof(uint32_t);
    for (int c = 0; c < NPUT && out->dataOk; c++) {
      const size_t base = c * words;
      for (size_t i = 0; i < words; i++) {
        if (recvLocal[base + i] != Pattern(src, base + i)) {
          out->dataOk = 0;
          out->badIdx = static_cast<uint32_t>(base + i);
          break;
        }
      }
    }
  }

  const uint32_t nq = devComm.sdma.sdmaNumQueue;
  out->localSlot = devComm.sdma.signalBuf[devComm.lsaRank * nq + qid];
  out->remoteSlot = devComm.sdma.signalBuf[src * nq + qid];

  // The point of the rework: quiet() drains via rptr, so it returns even when
  // this put's signal went to the peer instead of our own signalBuf.
  sdma.quiet(peer);
}

/* ---------------------------------- driver ----------------------------------- */

/* ---------------------- part E: one signal per group ------------------------- */

// Several lanes of one wavefront signalling together. By default the group emits
// one signal after all of its copies, so the counter advances by one per put()
// call whatever the lane count; ccoSdmaOptFlagsSignalPerCopy gives every copy its
// own and it advances by the lane count. Nothing else in this file posts from
// more than one lane, so this is the only cover for that layout.
template <uint32_t Flags>
__global__ void SdmaGroupSignalKernel(mori::cco::ccoWindowDevice* sendWin,
                                      mori::cco::ccoWindowDevice* recvWin, int lanes, size_t bytes,
                                      int qid, mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoSdma sdma{devComm};
  const int peer = (devComm.lsaRank + 1) % devComm.lsaSize;
  const int lane = threadIdx.x;
  if (lane < lanes) {
    sdma.put<ccoCoopThread, /*localSignal=*/true, /*remoteSignal=*/false, Flags,
             ccoSdmaThreadSameQueue>(peer, reinterpret_cast<ccoWindow_t>(recvWin), lane * bytes,
                                     reinterpret_cast<ccoWindow_t>(sendWin), lane * bytes, bytes,
                                     qid);
  }
  sdma.quiet<ccoCoopBlock>(peer);
}

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  g_rank = rank;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, BUF_BYTES, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, BUF_BYTES, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  const size_t words = BUF_BYTES / sizeof(uint32_t);
  std::vector<uint32_t> hostSend(words);
  for (size_t i = 0; i < words; i++) hostSend[i] = Pattern(rank, i);
  HIP_CHECK(hipMemcpy(sendBuf, hostSend.data(), BUF_BYTES, hipMemcpyHostToDevice));

  mori::cco::ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, BUF_BYTES, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, BUF_BYTES, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = 0;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  // Host patterns are keyed on `rank`, the device side on lsaRank; cco SDMA is
  // intra-node so they coincide. Fail loudly if that ever stops holding.
  if (devComm.lsaRank != rank || devComm.lsaSize != nranks) {
    fprintf(stderr, "[rank %d] unexpected LSA mapping: lsaRank=%d lsaSize=%d\n", rank,
            devComm.lsaRank, devComm.lsaSize);
    return 1;
  }

  bool ok = true;
  if (devComm.sdma.sdmaNumQueue == 0) {
    printf("[rank %d] SKIP — no SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank);
    fflush(stdout);
  } else {
    const int nq = static_cast<int>(devComm.sdma.sdmaNumQueue);
    const int src = (rank - 1 + nranks) % nranks;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    SignalResult* devRes = nullptr;
    HIP_CHECK(hipMalloc(&devRes, sizeof(SignalResult)));

    std::vector<uint32_t> host(words);
    // Host-side check of the first `bytes` received from src (part A only).
    auto verifyHost = [&](size_t bytes, const char* what) {
      HIP_CHECK(hipMemcpy(host.data(), recvBuf, bytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < bytes / sizeof(uint32_t); i++) {
        if (host[i] != Pattern(src, i)) {
          fprintf(stderr, "[rank %d] %s mismatch at word %zu: got %08x want %08x\n", rank, what, i,
                  host[i], Pattern(src, i));
          return false;
        }
      }
      return true;
    };

    /* ---- part A: queueId x scope x aggregate x size, quiet via rptr ---- */
    const char* partsEnv = getenv("CCO_TEST_PARTS");
    const char* parts = partsEnv ? partsEnv : "ABCDE";
    const bool verbose = getenv("CCO_TEST_VERBOSE") != nullptr;

    const size_t kSizes[] = {8, 64, 1024, 4096, 65536, 262144, MAX_CHUNK};
    int aPass = 0, aTotal = 0;
    for (int scope = 0; scope < 3 && ok && strchr(parts, 'A'); scope++) {
      for (int q = 0; q < nq && ok; q++) {
        for (int agg = 0; agg < 2 && ok; agg++) {
          for (size_t bytes : kSizes) {
            char what[96];
            snprintf(what, sizeof(what), "A[scope=%d q=%d agg=%d %zuB]", scope, q, agg, bytes);
            if (verbose) printf("[rank %d] %s\n", rank, what), fflush(stdout);
            HIP_CHECK(hipMemset(recvBuf, 0xff, BUF_BYTES));
            mori::cco::ccoBarrierAll(comm);
#define MATRIX_LAUNCH(FLAGS)                                            \
  do {                                                                  \
    if (scope == 0)                                                     \
      SdmaMatrixKernel<mori::cco::ccoCoopThread, FLAGS>                 \
          <<<1, 1, 0, stream>>>(sendWin, recvWin, bytes, q, devComm);   \
    else if (scope == 1)                                                \
      SdmaMatrixKernel<mori::cco::ccoCoopWarp, FLAGS>                   \
          <<<1, 64, 0, stream>>>(sendWin, recvWin, bytes, q, devComm);  \
    else                                                                \
      SdmaMatrixKernel<mori::cco::ccoCoopBlock, FLAGS>                  \
          <<<1, 256, 0, stream>>>(sendWin, recvWin, bytes, q, devComm); \
  } while (0)
            if (agg)
              MATRIX_LAUNCH(mori::cco::ccoSdmaOptFlagsAggregate);
            else
              MATRIX_LAUNCH(mori::cco::ccoSdmaOptFlagsDefault);
#undef MATRIX_LAUNCH
            HIP_CHECK(hipStreamSynchronize(stream));
            mori::cco::ccoBarrierAll(comm);
            aTotal++;
            if (verifyHost(bytes, what)) {
              aPass++;
            } else {
              ok = false;
              break;
            }
          }
        }
      }
    }
    const bool okA = ok;
    printf("[rank %d] A matrix (scope x queueId x agg x size): %d/%d\n", rank, aPass, aTotal);
    fflush(stdout);

    /* ---- parts B/C/D: signals ---- */
    // Cumulative expectations, PER QUEUE — each (src, queueId) is its own slot
    // and is never reset.
    std::vector<uint64_t> expLocal(nq, 0), expRemote(nq, 0);
    const size_t kSigSizes[] = {8, 4096, MAX_CHUNK};
    SignalResult res{};

    auto runSignalPart = [&](int mode, size_t bytes, int q, const char* tag) {
      // mode: 0 = remote only, 1 = local only, 2 = both
      if (mode != 1) expRemote[q] += NPUT;  // src sends remoteSignal to us
      if (mode != 0) expLocal[q] += NPUT;   // we signal ourselves
      const int verifyInKernel = (mode != 1);
      if (verbose) printf("[rank %d] %s\n", rank, tag), fflush(stdout);
      HIP_CHECK(hipMemset(recvBuf, 0xff, BUF_BYTES));
      HIP_CHECK(hipMemset(devRes, 0, sizeof(SignalResult)));
      mori::cco::ccoBarrierAll(comm);  // before the puts, not before the verify
      if (mode == 0)
        SdmaSignalKernel<false, true>
            <<<1, 1, 0, stream>>>(sendWin, recvWin, static_cast<uint32_t*>(recvBuf), bytes, q,
                                  expLocal[q], expRemote[q], verifyInKernel, devRes, devComm);
      else if (mode == 1)
        SdmaSignalKernel<true, false>
            <<<1, 1, 0, stream>>>(sendWin, recvWin, static_cast<uint32_t*>(recvBuf), bytes, q,
                                  expLocal[q], expRemote[q], verifyInKernel, devRes, devComm);
      else
        SdmaSignalKernel<true, true>
            <<<1, 1, 0, stream>>>(sendWin, recvWin, static_cast<uint32_t*>(recvBuf), bytes, q,
                                  expLocal[q], expRemote[q], verifyInKernel, devRes, devComm);
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipMemcpy(&res, devRes, sizeof(res), hipMemcpyDeviceToHost));

      bool good = true;
      if (verifyInKernel && !res.dataOk) {
        fprintf(stderr, "[rank %d] %s in-kernel data mismatch at word %u\n", rank, tag, res.badIdx);
        good = false;
      }
      // Slots are src-dimensioned: local and remote must not contaminate each other.
      if (res.localSlot != expLocal[q] || res.remoteSlot != expRemote[q]) {
        fprintf(stderr, "[rank %d] %s slot mismatch: local=%lu (want %lu) remote=%lu (want %lu)\n",
                rank, tag, res.localSlot, expLocal[q], res.remoteSlot, expRemote[q]);
        good = false;
      }
      if (mode == 1) {
        mori::cco::ccoBarrierAll(comm);
        if (!verifyHost(bytes, tag)) good = false;
      }
      return good;
    };

    bool okB = true, okC = true, okD = true;
    for (int q = 0; q < nq; q++) {
      for (size_t bytes : kSigSizes) {
        char tag[64];
        if (strchr(parts, 'B')) {
          snprintf(tag, sizeof(tag), "B[remote q=%d %zuB]", q, bytes);
          okB = runSignalPart(0, bytes, q, tag) && okB;
        }
        if (strchr(parts, 'C')) {
          snprintf(tag, sizeof(tag), "C[local q=%d %zuB]", q, bytes);
          okC = runSignalPart(1, bytes, q, tag) && okC;
        }
        if (strchr(parts, 'D')) {
          snprintf(tag, sizeof(tag), "D[both q=%d %zuB]", q, bytes);
          okD = runSignalPart(2, bytes, q, tag) && okD;
        }
      }
    }

    /* ---- part E: a group of lanes signals once, or once per copy ---- */
    // Read the slot before and after rather than resetting it: parts B-D keep a
    // running total in the same slot.
    bool okE = true;
    if (strchr(parts, 'E')) {
      const size_t kChunk = 256;
      hipDeviceProp_t props{};
      HIP_CHECK(hipGetDeviceProperties(&props, rank % numDevices));
      const int waveSize = props.warpSize;
      HSAuint64* slot = devComm.sdma.signalBuf + devComm.lsaRank * nq;  // queue 0
      for (int lanes : {1, 2, 8, 64}) {
        for (int perCopy = 0; perCopy < 2; perCopy++) {
          uint64_t before = 0, after = 0;
          HIP_CHECK(hipMemcpy(&before, slot, sizeof(before), hipMemcpyDeviceToHost));
          HIP_CHECK(hipMemset(recvBuf, 0xff, BUF_BYTES));
          mori::cco::ccoBarrierAll(comm);
          if (perCopy)
            SdmaGroupSignalKernel<mori::cco::ccoSdmaOptFlagsSignalPerCopy>
                <<<1, 64, 0, stream>>>(sendWin, recvWin, lanes, kChunk, 0, devComm);
          else
            SdmaGroupSignalKernel<mori::cco::ccoSdmaOptFlagsDefault>
                <<<1, 64, 0, stream>>>(sendWin, recvWin, lanes, kChunk, 0, devComm);
          HIP_CHECK(hipStreamSynchronize(stream));
          mori::cco::ccoBarrierAll(comm);
          HIP_CHECK(hipMemcpy(&after, slot, sizeof(after), hipMemcpyDeviceToHost));

          // One signal per wavefront, so a lane count above the wave width is
          // several groups. wave64 folds this back to 1.
          const uint64_t waves = (lanes + waveSize - 1) / waveSize;
          const uint64_t want = perCopy ? static_cast<uint64_t>(lanes) : waves;
          if (after - before != want) {
            fprintf(stderr, "[rank %d] E[%s lanes=%d] signal advanced by %lu, want %lu\n", rank,
                    perCopy ? "per-copy" : "group", lanes, after - before, want);
            okE = false;
          }
          // Every lane must still have delivered its own chunk.
          HIP_CHECK(hipMemcpy(host.data(), recvBuf, BUF_BYTES, hipMemcpyDeviceToHost));
          for (int c = 0; c < lanes && okE; c++) {
            for (size_t i = 0; i < kChunk / sizeof(uint32_t); i++) {
              const size_t idx = c * (kChunk / sizeof(uint32_t)) + i;
              if (host[idx] != Pattern(src, idx)) {
                fprintf(stderr, "[rank %d] E[%s lanes=%d] chunk %d word %zu: got %08x want %08x\n",
                        rank, perCopy ? "per-copy" : "group", lanes, c, i, host[idx],
                        Pattern(src, idx));
                okE = false;
                break;
              }
            }
          }
        }
      }
    }

    ok = okA && okB && okC && okD && okE;
    printf("[rank %d] A=%s B(remote)=%s C(local)=%s D(both)=%s E(group)=%s %s\n", rank,
           okA ? "PASS" : "FAIL", okB ? "PASS" : "FAIL", okC ? "PASS" : "FAIL",
           okD ? "PASS" : "FAIL", okE ? "PASS" : "FAIL", ok ? "PASSED" : "FAILED");
    fflush(stdout);

    HIP_CHECK(hipFree(devRes));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO SDMA signal/quiet", "/tmp/cco_sdma_signal_uid", 19897);
}
