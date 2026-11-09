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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "cco_test_harness.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t SLICE = 256 * 1024;  // bytes per (rank, peer) pair
static const int CALLS = 200;
static const int CHAINS_PER_CALL = 4;
static const int STEP_EVERY = 50;  // release one chain at a time in every 50th call
static const uint64_t TIMEOUT_NS = 10ull * 1000 * 1000 * 1000;
static const int MAX_RANKS = 16;
static const uint64_t kUnit = 32;  // ring bytes per packet
static const uint32_t kPollIntervalRetry = 0x10u | (0xFFFu << 16);

// Window layout (bytes). Each word on its own 64-byte line.
static const size_t GATE_OFF = 0;           // gate[p]: my queue to p waits on it
static const size_t GATE_E1_OFF = 4096;     // the same for the second queue per peer
static const size_t GATE_ABORT_OFF = 8192;  // [0] timeout, [1] wrap, [2] never opened
static const size_t ARRIVE_OFF = 12288;     // arrive[s]: peer s's engine adds here
static const size_t ARRIVE_E1_OFF = 16384;
static const size_t ARRIVE_ABORT_OFF = 20480;
static const size_t ARRIVE_WRAP_OFF = 24576;
static const size_t STATUS_OFF = 28672;  // status[s], written by rank s
static const size_t DATA_OFF = 65536;    // send | recv | recvE1 | scratch, n slices each

__global__ void OpenGates(uint64_t* gates, int nranks, int self, uint64_t add) {
  int p = threadIdx.x;
  if (p >= nranks || p == self) return;
  __hip_atomic_fetch_add(gates + p * 8, add, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void OpenGatesByOne(uint64_t* gates, int nranks, int self) {
  int p = blockIdx.x;
  if (threadIdx.x != 0 || p >= nranks || p == self) return;
  __hip_atomic_fetch_add(gates + p * 8, uint64_t(1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void StoreWord(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static uint32_t Pattern(int src, int dst, int epoch, size_t i) {
  return (uint32_t(src) << 24) ^ (uint32_t(dst) << 16) ^ (uint32_t(epoch) * 2654435761u) ^
         uint32_t(i);
}

namespace {

struct Ctx {
  int rank, nranks;
  ccoComm* comm;
  char* win;  // local window base
  hipStream_t stream;
  uint64_t round = 0;

  char* At(size_t off) { return win + off; }
  uint64_t* Gate(size_t base, int p) { return reinterpret_cast<uint64_t*>(At(base + p * 64)); }
  char* Send(int p) { return At(DATA_OFF + p * SLICE); }
  char* Recv(int s) { return At(DATA_OFF + (nranks + s) * SLICE); }
  char* RecvE1(int s) { return At(DATA_OFF + (2 * nranks + s) * SLICE); }
  char* Scratch(int s) { return At(DATA_OFF + (3 * nranks + s) * SLICE); }
  void* Peer(void* local, int p) { return ccoGetPeerPtr(comm, local, p); }

  uint64_t ReadWord(void* p) {
    uint64_t v = 0;
    HIP_CHECK(hipMemcpy(&v, p, sizeof(v), hipMemcpyDeviceToHost));
    return v;
  }

  // Collective: true on every rank iff `ok` on every rank.
  bool AllAgree(bool ok) {
    const uint64_t mine = (++round << 1) | (ok ? 1 : 0);
    for (int p = 0; p < nranks; p++) {
      auto* slot = static_cast<uint64_t*>(Peer(Gate(STATUS_OFF, rank), p));
      if (slot != nullptr) StoreWord<<<1, 1, 0, stream>>>(slot, mine);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
    bool all = true;
    for (int s = 0; s < nranks; s++)
      all = all && ReadWord(Gate(STATUS_OFF, s)) == ((round << 1) | 1);
    ccoBarrierAll(comm);
    return all;
  }

  void FillSend(int epoch) {
    std::vector<uint32_t> h(SLICE / 4);
    for (int p = 0; p < nranks; p++) {
      for (size_t i = 0; i < h.size(); i++) h[i] = Pattern(rank, p, epoch, i);
      HIP_CHECK(hipMemcpy(Send(p), h.data(), SLICE, hipMemcpyHostToDevice));
    }
  }

  bool CheckRecv(char* (Ctx::*slot)(int), int epoch, const char* what) {
    std::vector<uint32_t> h(SLICE / 4);
    for (int s = 0; s < nranks; s++) {
      if (s == rank) continue;
      HIP_CHECK(hipMemcpy(h.data(), (this->*slot)(s), SLICE, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < h.size(); i++) {
        if (h[i] != Pattern(s, rank, epoch, i)) {
          fprintf(stderr, "[rank %d] %s epoch %d: from %d word %zu = %08x, want %08x\n", rank, what,
                  epoch, s, i, h[i], Pattern(s, rank, epoch, i));
          return false;
        }
      }
    }
    return true;
  }

  bool Arrived(size_t base, uint64_t want, const char* what) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(TIMEOUT_NS);
    for (int s = 0; s < nranks; s++) {
      if (s == rank) continue;
      uint64_t v = ReadWord(Gate(base, s));
      while (v < want && std::chrono::steady_clock::now() < deadline) v = ReadWord(Gate(base, s));
      if (v != want) {
        fprintf(stderr, "[rank %d] %s: arrivals from %d = %lu, want %lu\n", rank, what, s, v, want);
        return false;
      }
    }
    return true;
  }

  bool Landed(char* (Ctx::*slot)(int), size_t base, uint64_t want, int epoch, const char* what) {
    return Arrived(base, want, what) && CheckRecv(slot, epoch, what);
  }
};

std::vector<uint32_t> ExpectedChain(const void* gate, uint32_t target, uint32_t func,
                                    const std::vector<ccoTriggerCopy>& copies, void* arrival) {
  auto lo = [](const void* p) { return uint32_t(reinterpret_cast<uintptr_t>(p)); };
  auto hi = [](const void* p) { return uint32_t(reinterpret_cast<uintptr_t>(p) >> 32); };
  const uint32_t poll = CCO_SDMA_OP_POLL_REGMEM | (func << 28) | (1u << 31);
  std::vector<uint32_t> w = {poll,        lo(gate),           hi(gate), target,
                             0xFFFFFFFFu, kPollIntervalRetry, 0,        0};
  const uint64_t kMax = CCO_SDMA_MAX_COPY_BYTES;
  for (const auto& c : copies) {
    for (uint64_t off = 0; off < c.bytes; off += kMax) {
      const uint64_t n = std::min<uint64_t>(c.bytes - off, kMax);
      const char* s = static_cast<const char*>(c.src) + off;
      const char* d = static_cast<const char*>(c.dst) + off;
      std::vector<uint32_t> u = {
          CCO_SDMA_OP_COPY, uint32_t(n - 1), 0, lo(s), hi(s), lo(d), hi(d), 0};
      w.insert(w.end(), u.begin(), u.end());
    }
  }
  std::vector<uint32_t> a = {
      CCO_SDMA_OP_ATOMIC | (CCO_SDMA_ATOMIC_ADD64 << 25), lo(arrival), hi(arrival), 1, 0, 0, 0, 0};
  w.insert(w.end(), a.begin(), a.end());
  return w;
}

bool RingMatches(ccoTriggerQueue_t q, uint64_t start, const std::vector<uint32_t>& want, int rank,
                 const char* what) {
  std::vector<uint32_t> got(want.size());
  if (ccoTriggerQueueReadRing(q, start, got.data(), got.size() * 4) != 0) return false;
  for (size_t i = 0; i < want.size(); i++) {
    if (got[i] != want[i]) {
      fprintf(stderr, "[rank %d] %s: ring dword %zu = %08x, want %08x\n", rank, what, i, got[i],
              want[i]);
      return false;
    }
  }
  return true;
}

}  // namespace

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  if (nranks > MAX_RANKS) return 1;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  const int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  const bool gfx942 = std::strncmp(prop.gcnArchName, "gfx942", 6) == 0;

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }
  const size_t winBytes = DATA_OFF + 4 * nranks * SLICE;
  void* buf = nullptr;
  ccoWindow_t win = nullptr;
  if (ccoMemAlloc(comm, winBytes, &buf) != 0 || ccoWindowRegister(comm, buf, winBytes, &win) != 0) {
    fprintf(stderr, "[rank %d] window setup failed\n", rank);
    return 1;
  }
  HIP_CHECK(hipMemset(buf, 0, winBytes));
  HIP_CHECK(hipDeviceSynchronize());

  Ctx c{rank, nranks, comm, static_cast<char*>(buf), nullptr};
  HIP_CHECK(hipStreamCreate(&c.stream));
  ccoBarrierAll(comm);

  std::vector<ccoTriggerQueue_t> q(nranks, nullptr), q1(nranks, nullptr);
  std::vector<ccoTriggerQueueInfo> info(nranks), info1(nranks);
  bool created = true;
  for (int p = 0; p < nranks; p++) {
    if (p == rank) continue;
    created = created && ccoTriggerQueueCreate(comm, p, 0, 0, &q[p]) == 0;
    if (created) ccoTriggerQueueGetInfo(q[p], &info[p]);
  }
  if (!c.AllAgree(created)) {
    printf("[rank %d] SKIP — no trigger queues on some rank (set MORI_ENABLE_SDMA=1)\n", rank);
    for (auto x : q) ccoTriggerQueueDestroy(x);
    return 0;
  }

  bool ok = true;
  auto report = [&](const char* name, bool pass) {
    printf("[rank %d] %s %s\n", rank, name, pass ? "PASS" : "FAIL");
    fflush(stdout);
    ok = ok && pass;
  };
  auto quiesceAll = [&](std::vector<ccoTriggerQueue_t>& qs) {
    bool pass = true;
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      if (ccoTriggerQuiesce(qs[p], UINT64_MAX, TIMEOUT_NS) != 0) {
        fprintf(stderr, "[rank %d] quiesce of queue to %d timed out\n", rank, p);
        pass = false;
      }
    }
    return pass;
  };

  {
    c.FillSend(1);
    HIP_CHECK(hipDeviceSynchronize());
    ccoBarrierAll(comm);
    bool ring = true, parked = true;
    std::vector<uint64_t> start(nranks), end(nranks);
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      ccoTriggerQueueState(q[p], nullptr, &start[p]);
      std::vector<ccoTriggerCopy> cp = {{c.Send(p), c.Peer(c.Recv(rank), p), SLICE}};
      void* arr = c.Peer(c.Gate(ARRIVE_OFF, rank), p);
      if (ccoTriggerAppendChain(q[p], c.Gate(GATE_OFF, p), 1, ccoSdmaCmpGe, cp.data(), 1, arr,
                                &end[p]) != 0)
        ring = false;
      ring =
          ring && end[p] - start[p] == 3 * kUnit &&
          RingMatches(q[p], start[p], ExpectedChain(c.Gate(GATE_OFF, p), 1, ccoSdmaCmpGe, cp, arr),
                      rank, "ring contents");
      ccoTriggerKick(q[p]);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      uint64_t r = 0;
      ccoTriggerQueueState(q[p], &r, nullptr);
      parked = parked && r < end[p];
    }
    ccoBarrierAll(comm);
    for (int s = 0; s < nranks; s++)
      if (s != rank) parked = parked && c.ReadWord(c.Gate(ARRIVE_OFF, s)) == 0;
    ccoBarrierAll(comm);
    OpenGates<<<1, 64, 0, c.stream>>>(c.Gate(GATE_OFF, 0), nranks, rank, 1);
    HIP_CHECK(hipStreamSynchronize(c.stream));
    bool data = c.Landed(&Ctx::Recv, ARRIVE_OFF, 1, 1, "closed gate");
    bool drained = quiesceAll(q);
    report("ring holds the chain packets", c.AllAgree(ring));
    report("engine waits at a closed gate until a kernel opens it",
           c.AllAgree(parked && drained && data));
  }

  {
    c.FillSend(2);
    HIP_CHECK(hipDeviceSynchronize());
    ccoBarrierAll(comm);
    bool appended = true;
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      ccoTriggerCopy cp{c.Send(p), c.Peer(c.Recv(rank), p), SLICE};
      appended = ccoTriggerAppendChain(q[p], c.Gate(GATE_OFF, p), 1, ccoSdmaCmpGe, &cp, 1,
                                       c.Peer(c.Gate(ARRIVE_OFF, rank), p), nullptr) == 0 &&
                 appended;
      ccoTriggerKick(q[p]);
    }
    bool data = c.Landed(&Ctx::Recv, ARRIVE_OFF, 2, 2, "open gate");
    bool drained = quiesceAll(q);
    report("gate already open at the kick", c.AllAgree(appended && drained && data));
  }

  {
    bool pass = true;
    uint64_t gate = 1, arrivals = 2;
    const size_t piece = SLICE / CHAINS_PER_CALL;
    std::vector<std::vector<uint64_t>> ends(nranks, std::vector<uint64_t>(CHAINS_PER_CALL));
    std::vector<ccoTriggerQueue_t> live;
    for (int p = 0; p < nranks; p++)
      if (p != rank) live.push_back(q[p]);
    for (int call = 0; call < CALLS; call++) {
      const int epoch = 100 + call;
      const bool stepwise = call % STEP_EVERY == 0;
      c.FillSend(epoch);
      HIP_CHECK(hipDeviceSynchronize());
      for (int p = 0; p < nranks; p++) {
        if (p == rank) continue;
        for (int j = 0; j < CHAINS_PER_CALL; j++) {
          ccoTriggerCopy cp{c.Send(p) + j * piece,
                            static_cast<char*>(c.Peer(c.Recv(rank), p)) + j * piece, piece};
          if (ccoTriggerAppendChain(q[p], c.Gate(GATE_OFF, p), uint32_t(gate + j + 1), ccoSdmaCmpGe,
                                    &cp, 1, c.Peer(c.Gate(ARRIVE_OFF, rank), p), &ends[p][j]) != 0)
            pass = false;
        }
        if (call % 2 == 0) ccoTriggerKick(q[p]);
      }
      if (call % 2 == 1) pass = ccoTriggerKickAll(live.data(), int(live.size())) == 0 && pass;
      ccoBarrierAll(comm);
      if (stepwise) {
        for (int j = 0; j < CHAINS_PER_CALL; j++) {
          OpenGatesByOne<<<nranks, 64, 0, c.stream>>>(c.Gate(GATE_OFF, 0), nranks, rank);
          HIP_CHECK(hipStreamSynchronize(c.stream));
          for (int p = 0; p < nranks; p++)
            if (p != rank) pass = pass && ccoTriggerQuiesce(q[p], ends[p][j], TIMEOUT_NS) == 0;
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          ccoBarrierAll(comm);
          pass = pass && c.Arrived(ARRIVE_OFF, arrivals + j + 1, "stepwise");
          ccoBarrierAll(comm);
        }
      } else {
        for (int j = 0; j < CHAINS_PER_CALL; j++)
          OpenGatesByOne<<<nranks, 64, 0, c.stream>>>(c.Gate(GATE_OFF, 0), nranks, rank);
        HIP_CHECK(hipStreamSynchronize(c.stream));
      }
      gate += CHAINS_PER_CALL;
      arrivals += CHAINS_PER_CALL;
      pass = pass && c.Landed(&Ctx::Recv, ARRIVE_OFF, arrivals, epoch, "calls");
      pass = quiesceAll(q) && pass;
      ccoBarrierAll(comm);
    }
    report("cumulative targets over 200 calls, 4 chains each, KickAll and stepwise release",
           c.AllAgree(pass));
  }

  // Engine offsets are gfx942-only (see ccoTriggerQueueCreate).
  bool second = gfx942;
  if (second) {
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      second = second && ccoTriggerQueueCreate(comm, p, 1, 0, &q1[p]) == 0;
      if (second) ccoTriggerQueueGetInfo(q1[p], &info1[p]);
    }
  }
  if (!c.AllAgree(second)) {
    printf("[rank %d] engine offset skipped: no queue on engine e+1 here\n", rank);
  } else {
    bool pass = true;
    uint32_t used = 0;
    for (int p = 0; p < nranks && pass; p++) {
      if (p == rank) continue;
      pass = pass && info1[p].engineId == info[p].engineId + 1;
      uint32_t bits = (1u << info[p].engineId) | (1u << info1[p].engineId);
      pass = pass && (used & bits) == 0;
      used |= bits;
    }
    if (!pass) {
      printf("[rank %d] engines:", rank);
      for (int p = 0; p < nranks; p++)
        if (p != rank) printf(" ->%d:%u/%u", p, info[p].engineId, info1[p].engineId);
      printf("\n");
    }
    report("engine offset 1 gives every peer queue its own engine", c.AllAgree(pass));

    c.FillSend(3);
    HIP_CHECK(hipDeviceSynchronize());
    ccoBarrierAll(comm);
    for (int p = 0; p < nranks && pass; p++) {
      if (p == rank) continue;
      ccoTriggerCopy cp{c.Send(p), c.Peer(c.RecvE1(rank), p), SLICE};
      pass = ccoTriggerAppendChain(q1[p], c.Gate(GATE_E1_OFF, p), 1, ccoSdmaCmpGe, &cp, 1,
                                   c.Peer(c.Gate(ARRIVE_E1_OFF, rank), p), nullptr) == 0;
      ccoTriggerKick(q1[p]);
    }
    ccoBarrierAll(comm);
    OpenGates<<<1, 64, 0, c.stream>>>(c.Gate(GATE_E1_OFF, 0), nranks, rank, 1);
    HIP_CHECK(hipStreamSynchronize(c.stream));
    pass = pass && c.Landed(&Ctx::RecvE1, ARRIVE_E1_OFF, 1, 3, "engine offset");
    pass = quiesceAll(q1) && pass;
    report("copy through a queue on engine e+1", c.AllAgree(pass));
  }

  {
    const int p = (rank + 1) % nranks;
    const int s = (rank + nranks - 1) % nranks;
    uint64_t* gate = c.Gate(GATE_ABORT_OFF, 0);
    ccoTriggerCopy cp{c.Send(p), c.Peer(c.Scratch(rank), p), SLICE};
    uint64_t end = 0;
    const bool appended =
        ccoTriggerAppendChain(q[p], gate, 100, ccoSdmaCmpGe, &cp, 1,
                              c.Peer(c.Gate(ARRIVE_ABORT_OFF, rank), p), &end) == 0;
    ccoTriggerKick(q[p]);
    const bool timedOut = ccoTriggerQuiesce(q[p], end, 50ull * 1000 * 1000) == 1;
    OpenGates<<<1, 64, 0, c.stream>>>(gate, 1, -1, 100);
    HIP_CHECK(hipStreamSynchronize(c.stream));
    const bool released = ccoTriggerQuiesce(q[p], end, TIMEOUT_NS) == 0;
    ccoBarrierAll(comm);
    uint64_t v = c.ReadWord(c.Gate(ARRIVE_ABORT_OFF, s));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(TIMEOUT_NS);
    while (v < 1 && std::chrono::steady_clock::now() < deadline)
      v = c.ReadWord(c.Gate(ARRIVE_ABORT_OFF, s));
    const bool arrived = v == 1;
    report("quiesce times out at a closed gate, then drains once it opens",
           c.AllAgree(appended && timedOut && released && arrived));
  }

  {
    const int p = (rank + 1) % nranks;
    const int s = (rank + nranks - 1) % nranks;
    bool split = false, refusal = false;

    ccoTriggerQueue_t unkicked = nullptr;
    if (ccoTriggerQueueCreate(comm, p, 0, 4096, &unkicked) == 0) {
      uint64_t* never = c.Gate(GATE_ABORT_OFF, 2);
      uint64_t* arr = c.Gate(ARRIVE_ABORT_OFF, rank);
      uint64_t start = 0, at = 0, end = 0;
      ccoTriggerQueueState(unkicked, nullptr, &start);
      split = true;
      // 2.5 GiB + 4 (3 packets); exactly the limit (1); the limit + 1 (2, the second 1 byte).
      const uint64_t sizes[3] = {(5ull << 29) + 4, CCO_SDMA_MAX_COPY_BYTES,
                                 CCO_SDMA_MAX_COPY_BYTES + 1};
      const uint64_t units[3] = {5, 3, 4};
      at = start;
      for (int k = 0; k < 3; k++) {
        std::vector<ccoTriggerCopy> cp = {{reinterpret_cast<void*>(0x100000000ull),
                                           reinterpret_cast<void*>(0x900000000ull), sizes[k]}};
        split = split &&
                ccoTriggerAppendChain(unkicked, never, 0xFFFFFFFFu, ccoSdmaCmpEq, cp.data(), 1, arr,
                                      &end) == 0 &&
                end - at == units[k] * kUnit &&
                RingMatches(unkicked, at, ExpectedChain(never, 0xFFFFFFFFu, ccoSdmaCmpEq, cp, arr),
                            rank, "split");
        at = end;
      }
      ccoTriggerCopy one{c.Send(p), c.Peer(c.Scratch(rank), p), 64};
      int appended = 0;
      while (ccoTriggerAppendChain(unkicked, never, 0xFFFFFFFFu, ccoSdmaCmpEq, &one, 1, arr,
                                   nullptr) == 0)
        appended++;
      uint64_t before = 0, after = 0;
      ccoTriggerQueueState(unkicked, nullptr, &before);
      const int rc =
          ccoTriggerAppendChain(unkicked, never, 0xFFFFFFFFu, ccoSdmaCmpEq, &one, 1, arr, nullptr);
      ccoTriggerQueueState(unkicked, nullptr, &after);
      // 12 units of split chains + 38 chains x 3 = 126 of 128 units; a 39th does not fit.
      refusal =
          rc == 1 && before == after && appended == 38 && before - start == (12 + 38 * 3) * kUnit;
      ccoTriggerQueueDestroy(unkicked);
    }
    report("copies over CCO_SDMA_MAX_COPY_BYTES are split", c.AllAgree(split));

    ccoTriggerQueue_t small = nullptr;
    bool wrap = ccoTriggerQueueCreate(comm, p, 0, 4096, &small) == 0;
    uint64_t gateVal = 0;
    ccoBarrierAll(comm);
    for (int call = 0; call < CALLS; call++) {
      const int epoch = 1000 + call;
      std::vector<uint32_t> h(64);
      for (size_t i = 0; i < h.size(); i++) h[i] = Pattern(rank, p, epoch, i);
      HIP_CHECK(hipMemcpy(c.Send(p), h.data(), 256, hipMemcpyHostToDevice));
      ccoTriggerCopy cp{c.Send(p), c.Peer(c.Scratch(rank), p), 256};
      uint64_t end = 0;
      if (wrap)
        wrap = ccoTriggerAppendChain(small, c.Gate(GATE_ABORT_OFF, 1), uint32_t(gateVal + 1),
                                     ccoSdmaCmpGe, &cp, 1, c.Peer(c.Gate(ARRIVE_WRAP_OFF, rank), p),
                                     &end) == 0;
      if (wrap) ccoTriggerKick(small);
      OpenGates<<<1, 64, 0, c.stream>>>(c.Gate(GATE_ABORT_OFF, 1), 1, -1, 1);
      HIP_CHECK(hipStreamSynchronize(c.stream));
      gateVal++;
      uint64_t v = c.ReadWord(c.Gate(ARRIVE_WRAP_OFF, s));
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(TIMEOUT_NS);
      while (wrap && v < uint64_t(call + 1) && std::chrono::steady_clock::now() < deadline)
        v = c.ReadWord(c.Gate(ARRIVE_WRAP_OFF, s));
      wrap = wrap && v == uint64_t(call + 1);
      HIP_CHECK(hipMemcpy(h.data(), c.Scratch(s), 256, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < h.size() && wrap; i++) wrap = h[i] == Pattern(s, rank, epoch, i);
      wrap = wrap && (small == nullptr || ccoTriggerQuiesce(small, end, TIMEOUT_NS) == 0);
      ccoBarrierAll(comm);
    }
    uint64_t w = 0;
    if (small != nullptr) ccoTriggerQueueState(small, nullptr, &w);
    wrap = wrap && w > 4 * 4096;
    if (small != nullptr) ccoTriggerQueueDestroy(small);
    report("200 calls through a 4 KiB ring", c.AllAgree(wrap));
    report("a chain that would fill the ring is refused", c.AllAgree(refusal));
  }

  for (int p = 0; p < nranks; p++) {
    if (q[p] != nullptr && ccoTriggerQueueDestroy(q[p]) != 0) ok = false;
    if (q1[p] != nullptr && ccoTriggerQueueDestroy(q1[p]) != 0) ok = false;
  }
  HIP_CHECK(hipStreamDestroy(c.stream));
  printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  ccoBarrierAll(comm);
  ccoWindowDeregister(comm, win);
  ccoMemFree(comm, buf);
  ccoCommDestroy(comm);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO SDMA trigger", "/tmp/cco_sdma_trigger_uid", 19893);
}
