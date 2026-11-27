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
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

// Device helpers use the compiler's attribute and builtins, not the HIP runtime's macros:
// this file is also compiled without the HIP runtime headers (the cco device bitcode).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define MORI_COUNTER_DEVICE_FN __attribute__((device)) inline __attribute__((always_inline))
#endif

// Without MORI_ENABLE_METRICS (BUILD_METRICS=OFF) the device helpers of this file are empty:
// the kernels keep their calls and carry no counting code.

// ===========================================================================
//  Every traffic counter MORI keeps: memory a data path adds to, without a lock,
//  an allocation or a library. With metrics off the pointer to a block is null.
//  Who owns a block and how it becomes series: metrics.hpp, section 3.
//
//  Sampling: one event in N is counted, N-fold, so every total stays an estimate
//  of the same quantity. IO decides per thread (section 2), EP on the device
//  (section 4). The per-NIC bytes and the GPU-issued RDMA (section 3) stay exact.
//
//  The FlyDSL kernels repeat the EP block layout of section 4
//  (python/mori/ops/dispatch_combine_v2/intranode_kernels.py): change both together.
//
//  Sections: 1 metric names and help   2 IO   3 RDMA issued by the GPU   4 EP
// ===========================================================================

/* ------------------------------------------------------------------------ */
/*  1. Metric names and help                                                 */
/* ------------------------------------------------------------------------ */

namespace mori {
namespace metrics {
namespace names {

// ---- IO (section 2) ----
inline constexpr const char* kIoSubmitted = "mori_io_submitted_total";
inline constexpr const char* kIoSubmittedHelp =
    "IO transfers handed to a backend, by flow. A flow is recorded once, by its initiator, with "
    "both ends: sum by the src_* labels for what a GPU sent, by the dst_* labels for what it "
    "received";
inline constexpr const char* kIoSubmittedBytes = "mori_io_submitted_bytes_total";
inline constexpr const char* kIoSubmittedBytesHelp =
    "Bytes of the IO transfers handed to a backend, by flow. Use this for the per-GPU traffic of "
    "xgmi and fabric, which settle a transfer only when the caller looks at its status";
inline constexpr const char* kIoCompleted = "mori_io_completed_total";
inline constexpr const char* kIoCompletedHelp =
    "IO transfers that reached a terminal state, by flow and status. unobserved: the status was "
    "destroyed or reused before anything saw the transfer finish";
inline constexpr const char* kIoCompletedBytes = "mori_io_completed_bytes_total";
inline constexpr const char* kIoCompletedBytesHelp =
    "Bytes of the IO transfers that completed successfully, by flow";
inline constexpr const char* kIoLatency = "mori_io_latency_seconds";
inline constexpr const char* kIoLatencyHelp =
    "Time from submission to the terminal state of an IO transfer. rdma: until the completion "
    "was processed. xgmi and fabric: until the caller observed the completion";
inline constexpr const char* kIoInflight = "mori_io_inflight";
inline constexpr const char* kIoInflightHelp =
    "IO transfers submitted and not yet seen in a terminal state";
inline constexpr const char* kIoNicBytes = "mori_io_nic_bytes_total";
inline constexpr const char* kIoNicBytesHelp =
    "Bytes of RDMA data work requests posted, by the pair of NICs that carried them. Recorded "
    "once, by the initiator: sum by the src_* labels for a NIC's transmit, by the dst_* labels "
    "for its receive";
inline constexpr const char* kIoNicLinkSpeed = "mori_io_nic_link_speed_bytes_per_second";
inline constexpr const char* kIoNicLinkSpeedHelp =
    "Link speed of a local NIC in bytes per second, as the system reports it: what the NIC's "
    "bytes per second are held against to say whether the link is full";

// ---- RDMA issued by the GPU itself (section 3) ----
inline constexpr const char* kGpuRdmaBytes = "mori_gpu_rdma_bytes_total";
inline constexpr const char* kGpuRdmaBytesHelp =
    "Bytes a GPU moved over RDMA itself (IBGDA), counted in the communication library (api) "
    "that issued the operation, by the pair of ranks. Recorded once, by the initiator, with "
    "both ends: sum by the src_* labels for what a NIC transmitted, by the dst_* labels for "
    "what it received. A get flows from the peer to the initiator";
inline constexpr const char* kGpuRdmaNicInfo = "mori_gpu_rdma_nic_info";
inline constexpr const char* kGpuRdmaNicInfoHelp =
    "Always 1: which NIC a rank's GPU-initiated RDMA goes through (every queue pair of a rank "
    "is on this one device)";

// ---- EP (section 4) ----
inline constexpr const char* kEpTokens = "mori_ep_tokens_total";
inline constexpr const char* kEpTokensHelp =
    "Tokens an EP rank sent to (tx) or received from (rx) a peer rank, per physical hop: "
    "path=local stays on the GPU, xgmi goes to a GPU of the same node, rdma to the proxy rank "
    "of another node. Counted on the device by the dispatch and combine kernels; a rank's tx "
    "towards a peer equals that peer's rx from it";
inline constexpr const char* kEpXgmiBytes = "mori_ep_xgmi_bytes_total";
inline constexpr const char* kEpXgmiBytesHelp =
    "Bytes EP moved over XGMI between this GPU and a GPU of the same node (tokens times the "
    "bytes a token carries). Includes tokens forwarded for other nodes; excludes tokens that "
    "stay on this GPU";
inline constexpr const char* kEpRdmaBytes = "mori_ep_rdma_bytes_total";
inline constexpr const char* kEpRdmaBytesHelp =
    "Bytes EP moved over RDMA between this rank and the proxy rank of another node (tokens "
    "times the bytes a token carries). ibdev is this rank's own NIC";
inline constexpr const char* kEpCalls = "mori_ep_calls_total";
inline constexpr const char* kEpCallsHelp = "EP kernel launches, counted on the device";
inline constexpr const char* kEpKernelSeconds = "mori_ep_kernel_seconds_total";
inline constexpr const char* kEpKernelSecondsHelp =
    "Time the EP kernels of a rank ran, measured on the device from the entry of a launch to "
    "its end. Bytes over wall time say how much EP moves; bytes over this say how fast the "
    "links carry it while a kernel runs, which is the bandwidth an EP benchmark reports";

// ---- The sampling in effect (mori/metrics/metrics.hpp, section 2) ----
inline constexpr const char* kSamplePeriod = "mori_metrics_sample_period";
inline constexpr const char* kSamplePeriodHelp =
    "One in this many events is counted, and counts this many times: IO transfers (module=io), "
    "EP dispatch and combine pairs (module=ep). 1 counts everything, 0 nothing. Set while the "
    "process runs: POST /sampling on this port, or mori.metrics.set_sampling()";
inline constexpr const char* kSampleInterval = "mori_metrics_sample_interval_seconds";
inline constexpr const char* kSampleIntervalHelp =
    "How often the metrics thread of this process reads the counters";

}  // namespace names

// Upper limit of a sampling period: a count taken is multiplied by it.
inline constexpr uint32_t kMaxSamplePeriod = 1u << 20;

// A 64-bit mix (the MurmurHash3 finalizer), to draw sampling gaps from.
constexpr uint64_t SampleMix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

}  // namespace metrics
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  2. IO                                                                    */
/* ------------------------------------------------------------------------ */

// Fixed arrays of relaxed atomics: no lock, allocation or string on the data path.
// A block's identity (`id`) is filled in once, when IoMetrics (metrics.hpp) creates it.

namespace mori {
namespace io {

enum class IoOp : uint8_t { kRead = 0, kWrite = 1 };
inline constexpr int kIoOpCount = 2;

// A flow is recorded once, by the initiator. Its local end is on this process's host; a write
// moves data local -> remote, a read remote -> local.

// A flow through the engine: GPU to GPU (or host memory) over one backend.
struct IoFlowIdentity {
  std::string backend;  // "xgmi", "rdma", "tcp", ...
  // Physical GPU index ("4"), "pci:<bdf>" when it is unknown, "cpu" for host memory.
  std::string localGpu;
  std::string remoteHost;  // the peer's host label
  std::string remoteGpu;

  bool operator<(const IoFlowIdentity& o) const {
    return std::tie(backend, localGpu, remoteHost, remoteGpu) <
           std::tie(o.backend, o.localGpu, o.remoteHost, o.remoteGpu);
  }
};

// The RDMA traffic between a local NIC and the remote NIC it is connected to.
struct NicFlowIdentity {
  std::string ibdev;   // the local RDMA device, "mlx5_4"
  std::string netdev;  // its network interface, "ens4np0"; empty when it has none
  std::string remoteHost;
  std::string remoteIbdev;  // "dev<n>" from a peer that does not name it
  std::string remoteNetdev;

  bool operator<(const NicFlowIdentity& o) const {
    return std::tie(ibdev, netdev, remoteHost, remoteIbdev, remoteNetdev) <
           std::tie(o.ibdev, o.netdev, o.remoteHost, o.remoteIbdev, o.remoteNetdev);
  }
};

// Upper bounds of the latency histogram, in ns: 10 us to 10 s. A block's bucket array has
// one more counter, the +Inf bucket.
inline constexpr uint64_t kIoLatencyBoundsNs[] = {
    10'000ULL,        25'000ULL,        50'000ULL,        100'000ULL,       250'000ULL,
    500'000ULL,       1'000'000ULL,     2'500'000ULL,     5'000'000ULL,     10'000'000ULL,
    25'000'000ULL,    50'000'000ULL,    100'000'000ULL,   250'000'000ULL,   500'000'000ULL,
    1'000'000'000ULL, 2'500'000'000ULL, 5'000'000'000ULL, 10'000'000'000ULL};
inline constexpr int kIoLatencyBoundCount =
    static_cast<int>(sizeof(kIoLatencyBoundsNs) / sizeof(kIoLatencyBoundsNs[0]));

inline uint64_t IoNowNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

// One direction (read or write) of one flow.
struct alignas(64) IoOpCounters {
  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> submittedBytes{0};
  // `unobserved`: the status was destroyed or reused before the transfer was seen to finish.
  std::atomic<uint64_t> completedOk{0};
  std::atomic<uint64_t> completedFailed{0};
  std::atomic<uint64_t> completedUnobserved{0};
  // Of the transfers that ended ok.
  std::atomic<uint64_t> completedBytes{0};
  // Submit to terminal state, ok and failed. Buckets are NOT cumulative here.
  std::atomic<uint64_t> latencyNs{0};
  std::atomic<uint64_t> latencyBuckets[kIoLatencyBoundCount + 1] = {};
};

// The counters of one flow, shared by all its transfers. Process-wide, never freed.
// `weight`: how many transfers the recorded one stands for (1, or the sampling period).
struct IoFlowCounters {
  IoFlowIdentity id;
  // The process's IO sampling period; null meters every transfer.
  const std::atomic<uint32_t>* samplePeriod{nullptr};
  IoOpCounters op[kIoOpCount];

  void Submit(IoOp o, uint64_t bytes, uint64_t weight = 1) {
    IoOpCounters& c = op[static_cast<int>(o)];
    c.submitted.fetch_add(weight, std::memory_order_relaxed);
    c.submittedBytes.fetch_add(bytes * weight, std::memory_order_relaxed);
  }

  void Complete(IoOp o, bool ok, uint64_t bytes, uint64_t latencyNs, uint64_t weight = 1) {
    IoOpCounters& c = op[static_cast<int>(o)];
    if (ok) {
      c.completedOk.fetch_add(weight, std::memory_order_relaxed);
      c.completedBytes.fetch_add(bytes * weight, std::memory_order_relaxed);
    } else {
      c.completedFailed.fetch_add(weight, std::memory_order_relaxed);
    }
    c.latencyNs.fetch_add(latencyNs * weight, std::memory_order_relaxed);
    int b = 0;
    while (b < kIoLatencyBoundCount && latencyNs > kIoLatencyBoundsNs[b]) ++b;
    c.latencyBuckets[b].fetch_add(weight, std::memory_order_relaxed);
  }

  void Abandon(IoOp o, uint64_t weight = 1) {
    op[static_cast<int>(o)].completedUnobserved.fetch_add(weight, std::memory_order_relaxed);
  }
};

// Whether the calling thread's next transfer is metered, one in `period`: returns how many
// transfers it stands for (`period`), or 0 to pass it over. The gap to the next one is drawn
// from 1 to 2 * period - 1, per thread, so alternating kinds of work are both sampled.
inline uint32_t IoSampleWeight(uint32_t period) {
  if (period <= 1) return period;
  struct State {
    uint32_t left = 0;    // transfers to pass over before the next one taken
    uint32_t period = 0;  // the period `left` was drawn for
    uint64_t seed = 0;
  };
  static thread_local State st;
  const uint64_t span = 2 * static_cast<uint64_t>(period) - 1;
  auto draw = [&] {
    if (st.seed == 0) st.seed = IoNowNs() ^ reinterpret_cast<uintptr_t>(&st);
    st.seed = mori::metrics::SampleMix(st.seed + 0x9e3779b97f4a7c15ULL);
    return static_cast<uint32_t>(st.seed % span);
  };
  if (st.period != period) {
    st.period = period;
    const uint32_t a = draw(), b = draw();
    st.left = a < b ? a : b;
  }
  if (st.left == 0) {
    st.left = draw();
    return period;
  }
  --st.left;
  return 0;
}

// Meters one transfer from its submission to its end; embedded in its TransferStatus.
//   Start    before the status goes to a backend, which may finish it inside submit.
//   Finish   on a terminal state; takes effect once, whoever races.
//   Abandon  the status was destroyed or reused with the transfer never seen to finish.
// The sampling weight is kept with the transfer, so a period change cannot unbalance inflight.
class IoPerTransferMeter {
 public:
  void Start(IoFlowCounters* flow, IoOp op, uint64_t bytes) {
    if (flow == nullptr) return;
    Abandon();
    uint32_t weight = 1;
    if (flow->samplePeriod != nullptr) {
      weight = IoSampleWeight(flow->samplePeriod->load(std::memory_order_relaxed));
      if (weight == 0) return;
    }
    flow_ = flow;
    bytes_ = bytes;
    submitNs_ = IoNowNs();
    op_ = op;
    weight_ = weight;
    flow->Submit(op, bytes, weight);
    armed_.store(true, std::memory_order_release);
  }

  void Finish(bool ok) {
    if (!armed_.load(std::memory_order_relaxed)) return;
    if (armed_.exchange(false, std::memory_order_acq_rel)) {
      flow_->Complete(op_, ok, bytes_, IoNowNs() - submitNs_, weight_);
    }
  }

  void Abandon() {
    if (!armed_.load(std::memory_order_relaxed)) return;
    if (armed_.exchange(false, std::memory_order_acq_rel)) flow_->Abandon(op_, weight_);
  }

 private:
  IoFlowCounters* flow_{nullptr};
  uint64_t bytes_{0};
  uint64_t submitNs_{0};
  IoOp op_{IoOp::kRead};
  uint32_t weight_{1};              // transfers this one stands for
  std::atomic<bool> armed_{false};  // a transfer was started and has not ended
};

// The counters of one pair of NICs. Bytes are counted when the data WRs are posted.
struct alignas(64) NicFlowCounters {
  NicFlowIdentity id;
  // On a cache line of their own: `id` is only read, these are added to.
  alignas(64) std::atomic<uint64_t> bytes[kIoOpCount] = {};

  void Posted(bool isRead, uint64_t n) {
    bytes[isRead ? 0 : 1].fetch_add(n, std::memory_order_relaxed);
  }
};

}  // namespace io
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  3. RDMA issued by the GPU itself (shmem, cco GDA)                        */
/* ------------------------------------------------------------------------ */

// Bytes a GPU moves over RDMA itself (IBGDA), counted in the library that issues the
// operation (shmem, cco GDA). One flat uint64 block on the device, [op][pe]: `n` is the WORLD
// size, `pe` a world rank. The NIC is not recorded: every QP of a rank is on one NIC, which
// the host adds when it publishes. A null block pointer means counting is off.

namespace mori {
namespace application {

enum RdmaTrafficOp : int {
  kRdmaTrafficPut = 0,     // data written to the peer
  kRdmaTrafficInline = 1,  // a small value carried in the request itself
  kRdmaTrafficAtomic = 2,  // an atomic on the peer's memory, counted as its operand size
  kRdmaTrafficGet = 3,     // data read FROM the peer: the only op that flows towards this rank
};
inline constexpr int kRdmaTrafficOpCount = 4;

constexpr size_t RdmaTrafficSlot(int op, int n, int pe) {
  return static_cast<size_t>(op) * static_cast<size_t>(n) + static_cast<size_t>(pe);
}
constexpr size_t RdmaTrafficSlots(int n) {
  return static_cast<size_t>(kRdmaTrafficOpCount) * static_cast<size_t>(n);
}

#if defined(__HIPCC__) || defined(__CUDACC__)
#ifdef MORI_ENABLE_METRICS
MORI_COUNTER_DEVICE_FN void RdmaTrafficCount(uint64_t* block, int n, int op, int pe,
                                             uint64_t bytes) {
  if (block == nullptr) return;
  __hip_atomic_fetch_add(block + RdmaTrafficSlot(op, n, pe), bytes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
}
#else
MORI_COUNTER_DEVICE_FN void RdmaTrafficCount(uint64_t*, int, int, int, uint64_t) {}
#endif
#endif

}  // namespace application
}  // namespace mori

/* ------------------------------------------------------------------------ */
/*  4. EP                                                                    */
/* ------------------------------------------------------------------------ */

// ---------------------------------------------------------------------------
//  EP statistics, counted ON THE DEVICE by the dispatch and combine kernels: a
//  captured graph replays the kernels with no host code in between.
//
//  One flat uint64 block, the same for v1 HIP, v2 HIP and v2 FlyDSL. `n` is the
//  world size, `op` dispatch (0) or combine (1):
//
//    sent[op][pe]   n * 2   tokens this rank sent to pe           (cumulative)
//    recv[op][pe]   n * 2   tokens this rank received from pe     (cumulative)
//    lastSent[pe]   n       what the latest dispatch sent to pe, until combined
//    lastRecv[pe]   n       what the latest dispatch got from pe, until combined
//    calls[op]      2       kernel launches                       (cumulative)
//    ticks[op]      2       their time, in device clock ticks     (cumulative)
//    lastEnd        1       when the latest timed launch ended
//    gapLimit       1       host-set: a launch this soon after lastEnd is timed from lastEnd
//    mirror         1       host-set: address of the host-mapped copy of the counters
//    arrived        1       blocks of the running launch that are done (EpStatsSpan)
//    seqStart       1       when a dispatch or combine of several launches started
//    period         1       host-set: one dispatch in so many counts
//    weight         1       what the dispatch in progress and its combine count for; 0 = skipped
//    seq            1       dispatches so far, counted or not
//    mark           1       1 in a skipped dispatch right before one that counts
//
//  Hops. Each PHYSICAL hop is counted, `pe` being the rank at its other end: a
//  token forwarded by a proxy counts on the RDMA hop and again on the proxy's
//  XGMI hop. Combine has no per-peer counting point: it returns what its
//  dispatch left in lastSent / lastRecv, swapped.
//
//  Time. Lane 0 of one warp reads the clock at entry and exit (EpStatsLaunch);
//  back-to-back launches are timed from the previous end (gapLimit). An
//  inter-node dispatch or combine is several launches with no thread known to
//  leave last: timed from EpStatsSeqBegin to the last block of the last launch
//  (EpStatsSpan).
//
//  Where the block lives. In device memory; the timing warp copies the counters
//  to `mirror` with plain stores and the host reads only that: a HIP read from
//  the metrics thread would invalidate a graph capture.
//
//  Sampling. The host writes `period` into host-mapped memory (the mirror's
//  slot). One thread per dispatch turns it into `weight` (EpStatsDecide) before
//  any counting site runs; the combine goes by its dispatch's weight. Which
//  dispatch of every N counts is a hash of its number, the same on every rank.
//  A skipped dispatch right before one that counts (`mark`) records its end.
//
//  A null block pointer means statistics are off.
// ---------------------------------------------------------------------------

namespace mori {
namespace moe {

enum EpStatsOp : int { kEpStatsDispatch = 0, kEpStatsCombine = 1 };
inline constexpr int kEpStatsOpCount = 2;

constexpr size_t EpStatsSentOffset(int op, int n) { return static_cast<size_t>(op) * n; }
constexpr size_t EpStatsRecvOffset(int op, int n) { return static_cast<size_t>(2 + op) * n; }
constexpr size_t EpStatsLastSentOffset(int n) { return static_cast<size_t>(4) * n; }
constexpr size_t EpStatsLastRecvOffset(int n) { return static_cast<size_t>(5) * n; }
constexpr size_t EpStatsCallsOffset(int n) { return static_cast<size_t>(6) * n; }
constexpr size_t EpStatsTicksOffset(int n) { return static_cast<size_t>(6) * n + kEpStatsOpCount; }
// End of the counters (what the mirror copies); timing and sampling state follows.
constexpr size_t EpStatsCounterSlots(int n) {
  return static_cast<size_t>(6) * n + 2 * kEpStatsOpCount;
}
constexpr size_t EpStatsLastEndOffset(int n) { return EpStatsCounterSlots(n); }
constexpr size_t EpStatsGapLimitOffset(int n) { return EpStatsCounterSlots(n) + 1; }
constexpr size_t EpStatsMirrorOffset(int n) { return EpStatsCounterSlots(n) + 2; }
constexpr size_t EpStatsArrivedOffset(int n) { return EpStatsCounterSlots(n) + 3; }
constexpr size_t EpStatsSeqStartOffset(int n) { return EpStatsCounterSlots(n) + 4; }
constexpr size_t EpStatsPeriodOffset(int n) { return EpStatsCounterSlots(n) + 5; }
constexpr size_t EpStatsWeightOffset(int n) { return EpStatsCounterSlots(n) + 6; }
constexpr size_t EpStatsSeqOffset(int n) { return EpStatsCounterSlots(n) + 7; }
constexpr size_t EpStatsMarkOffset(int n) { return EpStatsCounterSlots(n) + 8; }
constexpr size_t EpStatsSlots(int n) { return EpStatsCounterSlots(n) + 9; }

// Place (0 .. period - 1) of the dispatch that counts within row `row` of `period`
// dispatches. The same on every rank; the FlyDSL kernels compute it too (20-bit steps).
constexpr uint64_t EpStatsSamplePlace(uint64_t row, uint64_t period) {
  uint64_t h = row & 0xFFFFF;
  h = (h * 40503 + 0x9E37) & 0xFFFFF;
  h ^= h >> 9;
  h = (h * 69621) & 0xFFFFF;
  h ^= h >> 11;
  return (h * period) >> 20;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
#ifdef MORI_ENABLE_METRICS

// The device's wall clock (what wall_clock64() reads), declared here for builds without
// the HIP runtime headers.
extern "C" __attribute__((device)) unsigned long long __ockl_steadyctr_u64();

MORI_COUNTER_DEVICE_FN void EpStatsAdd(uint64_t* slot, uint64_t v) {
  __hip_atomic_fetch_add(slot, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Weight of the dispatch or combine in progress; 0 when skipped or statistics are off.
MORI_COUNTER_DEVICE_FN uint64_t EpStatsWeight(uint64_t* stats, int n) {
  if (stats == nullptr) return 0;
  return __hip_atomic_load(stats + EpStatsWeightOffset(n), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}

// True when the launch must record its end: it counts, or the next dispatch does (`mark`).
MORI_COUNTER_DEVICE_FN bool EpStatsActive(uint64_t* stats, int n) {
  if (stats == nullptr) return false;
  return (__hip_atomic_load(stats + EpStatsWeightOffset(n), __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) |
          __hip_atomic_load(stats + EpStatsMarkOffset(n), __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT)) != 0;
}

// Decides whether this dispatch and its combine count. Call from ONE thread, before any
// counting site of the dispatch runs.
MORI_COUNTER_DEVICE_FN void EpStatsDecide(uint64_t* stats, int n) {
  if (stats == nullptr) return;
  // The host writes the period into the mirror.
  uint64_t* mirror = reinterpret_cast<uint64_t*>(stats[EpStatsMirrorOffset(n)]);
  const uint64_t period = mirror[EpStatsPeriodOffset(n)];
  const uint64_t seq = stats[EpStatsSeqOffset(n)];
  stats[EpStatsSeqOffset(n)] = seq + 1;
  uint64_t weight = period;  // 0: nothing counts; 1: everything does
  uint64_t mark = 0;
  if (period > 1 && seq % period != EpStatsSamplePlace(seq / period, period)) {
    weight = 0;
    // The next dispatch counts: this one and its combine leave their end for it.
    const uint64_t next = seq + 1;
    if (next % period == EpStatsSamplePlace(next / period, period)) mark = 1;
  }
  __hip_atomic_store(stats + EpStatsMarkOffset(n), mark, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(stats + EpStatsWeightOffset(n), weight, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
}

// Dispatch: this rank tells `pe` how many tokens it sent. One thread per (dispatch, pe);
// lastSent is a plain store, so a replayed dispatch is not combined twice.
MORI_COUNTER_DEVICE_FN void EpStatsDispatchSent(uint64_t* stats, int n, int pe, uint64_t tokens) {
  const uint64_t weight = EpStatsWeight(stats, n);
  if (weight == 0) return;
  tokens *= weight;
  EpStatsAdd(stats + EpStatsSentOffset(kEpStatsDispatch, n) + pe, tokens);
  stats[EpStatsLastSentOffset(n) + pe] = tokens;
}

// Dispatch: this rank reads how many tokens `pe` sent it.
MORI_COUNTER_DEVICE_FN void EpStatsDispatchRecv(uint64_t* stats, int n, int pe, uint64_t tokens) {
  const uint64_t weight = EpStatsWeight(stats, n);
  if (weight == 0) return;
  tokens *= weight;
  EpStatsAdd(stats + EpStatsRecvOffset(kEpStatsDispatch, n) + pe, tokens);
  stats[EpStatsLastRecvOffset(n) + pe] = tokens;
}

// The same two for a hop that reaches a peer in several pieces per dispatch (the RDMA
// chunks of an inter-node dispatch): the pieces add up in last*.
MORI_COUNTER_DEVICE_FN void EpStatsDispatchSentChunk(uint64_t* stats, int n, int pe,
                                                     uint64_t tokens) {
  const uint64_t weight = EpStatsWeight(stats, n);
  if (weight == 0) return;
  tokens *= weight;
  EpStatsAdd(stats + EpStatsSentOffset(kEpStatsDispatch, n) + pe, tokens);
  EpStatsAdd(stats + EpStatsLastSentOffset(n) + pe, tokens);
}

MORI_COUNTER_DEVICE_FN void EpStatsDispatchRecvChunk(uint64_t* stats, int n, int pe,
                                                     uint64_t tokens) {
  const uint64_t weight = EpStatsWeight(stats, n);
  if (weight == 0) return;
  tokens *= weight;
  EpStatsAdd(stats + EpStatsRecvOffset(kEpStatsDispatch, n) + pe, tokens);
  EpStatsAdd(stats + EpStatsLastRecvOffset(n) + pe, tokens);
}

// Counts one dispatch or combine, for kernels timed with EpStatsSpan (which does not
// count). Call from one thread.
MORI_COUNTER_DEVICE_FN void EpStatsCall(uint64_t* stats, int n, int op) {
  const uint64_t weight = EpStatsWeight(stats, n);
  if (weight == 0) return;
  EpStatsAdd(stats + EpStatsCallsOffset(n) + op, weight);
}

// One kernel launch, counted and timed. Every thread constructs it first thing in the
// kernel body; `timing` is true in warp 0 of block 0, `lane` the thread's place in it.
// Lane 0 decides (in a dispatch), counts and times the launch; on exit the warp copies the
// counters to the mirror.
struct EpStatsLaunch {
  uint64_t* stats;
  uint64_t start, weight;
  int n, op, lane, wave;

  MORI_COUNTER_DEVICE_FN EpStatsLaunch(uint64_t* block, int worldSize, int statsOp, bool timing,
                                       int laneId, int waveSize)
      : stats(timing ? block : nullptr),
        start(0),
        weight(0),
        n(worldSize),
        op(statsOp),
        lane(laneId),
        wave(waveSize) {
    if (stats == nullptr) return;
    if (op == kEpStatsDispatch && lane == 0) EpStatsDecide(stats, n);
    // Lane 0's verdict, stored by the line above.
    weight = EpStatsWeight(stats, n);
    if (weight == 0) {
      // Skipped: stays on only to record its end (lane 0, when `mark` is set).
      if (lane != 0 || !EpStatsActive(stats, n)) stats = nullptr;
      return;
    }
    if (lane != 0) return;
    EpStatsAdd(stats + EpStatsCallsOffset(n) + op, weight);
    start = __ockl_steadyctr_u64();
    const uint64_t prev = stats[EpStatsLastEndOffset(n)];
    if (prev != 0 && start - prev < stats[EpStatsGapLimitOffset(n)]) start = prev;
  }
  MORI_COUNTER_DEVICE_FN ~EpStatsLaunch() {
    if (stats == nullptr) return;
    if (weight == 0) {
      stats[EpStatsLastEndOffset(n)] = __ockl_steadyctr_u64();
      return;
    }
    if (lane == 0) {
      const uint64_t end = __ockl_steadyctr_u64();
      EpStatsAdd(stats + EpStatsTicksOffset(n) + op, (end - start) * weight);
      stats[EpStatsLastEndOffset(n)] = end;
    }
    uint64_t* mirror = reinterpret_cast<uint64_t*>(stats[EpStatsMirrorOffset(n)]);
    const int slots = static_cast<int>(EpStatsCounterSlots(n));
    for (int i = lane; i < slots; i += wave) mirror[i] = stats[i];
  }
  EpStatsLaunch(const EpStatsLaunch&) = delete;
  EpStatsLaunch& operator=(const EpStatsLaunch&) = delete;
};

// First launch of a dispatch or combine that is several launches: decides (in a dispatch)
// and records when it started. Call from one thread, first thing.
MORI_COUNTER_DEVICE_FN void EpStatsSeqBegin(uint64_t* stats, int n, int op) {
  if (stats == nullptr) return;
  if (op == kEpStatsDispatch) EpStatsDecide(stats, n);
  if (EpStatsWeight(stats, n) == 0) return;
  uint64_t start = __ockl_steadyctr_u64();
  const uint64_t prev = stats[EpStatsLastEndOffset(n)];
  if (prev != 0 && start - prev < stats[EpStatsGapLimitOffset(n)]) start = prev;
  stats[EpStatsSeqStartOffset(n)] = start;
}

// A sequence is charged from seqStart only if its last launch starts within this many gap
// limits of it.
inline constexpr uint64_t kEpStatsSequenceGap = 1u << 16;

// Last launch of a dispatch or combine that is several launches. Every thread constructs it
// first thing in the kernel body. On exit each block reports in after a barrier; the last
// block charges the time since seqStart and its first warp copies the counters to the
// mirror. Does not count the launch: EpStatsCall does.
struct EpStatsSpan {
  uint64_t* stats;
  int n, op, thread, block, wave;
  uint64_t blocks;

  MORI_COUNTER_DEVICE_FN EpStatsSpan(uint64_t* counters, int worldSize, int statsOp, int threadId,
                                     int blockId, int waveSize, int blockNum)
      : stats(EpStatsActive(counters, worldSize) ? counters : nullptr),
        n(worldSize),
        op(statsOp),
        thread(threadId),
        block(blockId),
        wave(waveSize),
        blocks(static_cast<uint64_t>(blockNum)) {
    if (stats == nullptr || block != 0 || thread != 0) return;
    if (EpStatsWeight(stats, n) == 0) return;  // only its end is wanted
    uint64_t start = __ockl_steadyctr_u64();
    const uint64_t prev = stats[EpStatsLastEndOffset(n)];
    const uint64_t limit = stats[EpStatsGapLimitOffset(n)];
    const uint64_t seq = stats[EpStatsSeqStartOffset(n)];
    if (seq != 0 && seq >= prev && start - seq < limit * kEpStatsSequenceGap) {
      start = seq;
    } else if (prev != 0 && start - prev < limit) {
      start = prev;
    }
    // An atomic: read back by another block of this launch.
    __hip_atomic_store(stats + EpStatsSeqStartOffset(n), start, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  MORI_COUNTER_DEVICE_FN ~EpStatsSpan() {
    if (stats == nullptr) return;
    // __syncthreads(), without the HIP runtime headers.
    __builtin_amdgcn_s_barrier();
    if (thread >= wave) return;
    int last = 0;
    if (thread == 0) {
      uint64_t* arrived = stats + EpStatsArrivedOffset(n);
      const uint64_t before = __hip_atomic_fetch_add(arrived, static_cast<uint64_t>(1),
                                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (before + 1 == blocks) {
        last = 1;
        __hip_atomic_store(arrived, static_cast<uint64_t>(0), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        const uint64_t end = __ockl_steadyctr_u64();
        const uint64_t start = __hip_atomic_load(stats + EpStatsSeqStartOffset(n), __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
        const uint64_t weight = EpStatsWeight(stats, n);
        if (weight != 0 && start != 0 && end > start) {
          EpStatsAdd(stats + EpStatsTicksOffset(n) + op, (end - start) * weight);
        }
        stats[EpStatsLastEndOffset(n)] = end;
        stats[EpStatsSeqStartOffset(n)] = 0;
        // Nothing was counted in a launch that only leaves its end: no copy.
        if (weight == 0) last = 0;
      }
    }
    // Thread 0's `last`, to its whole warp (__shfl(last, 0)).
    last = __builtin_amdgcn_ds_bpermute(0, last);
    if (last == 0) return;
    uint64_t* mirror = reinterpret_cast<uint64_t*>(stats[EpStatsMirrorOffset(n)]);
    const int slots = static_cast<int>(EpStatsCounterSlots(n));
    for (int i = thread; i < slots; i += wave) {
      mirror[i] = __hip_atomic_load(stats + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  EpStatsSpan(const EpStatsSpan&) = delete;
  EpStatsSpan& operator=(const EpStatsSpan&) = delete;
};

// Combine, once per (combine, pe), from one thread: returns what the latest dispatch
// exchanged with `pe`, swapped. The counts are taken (zeroed), so a second combine, or one
// without a dispatch, counts nothing.
MORI_COUNTER_DEVICE_FN void EpStatsCombinePeer(uint64_t* stats, int n, int pe) {
  if (EpStatsWeight(stats, n) == 0) return;
  const uint64_t back =
      __hip_atomic_exchange(stats + EpStatsLastRecvOffset(n) + pe, static_cast<uint64_t>(0),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const uint64_t home =
      __hip_atomic_exchange(stats + EpStatsLastSentOffset(n) + pe, static_cast<uint64_t>(0),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (back != 0) EpStatsAdd(stats + EpStatsSentOffset(kEpStatsCombine, n) + pe, back);
  if (home != 0) EpStatsAdd(stats + EpStatsRecvOffset(kEpStatsCombine, n) + pe, home);
}

#else  // no MORI_ENABLE_METRICS: what the kernels call, empty

MORI_COUNTER_DEVICE_FN bool EpStatsActive(uint64_t*, int) { return false; }
MORI_COUNTER_DEVICE_FN void EpStatsDispatchSent(uint64_t*, int, int, uint64_t) {}
MORI_COUNTER_DEVICE_FN void EpStatsDispatchRecv(uint64_t*, int, int, uint64_t) {}
MORI_COUNTER_DEVICE_FN void EpStatsDispatchSentChunk(uint64_t*, int, int, uint64_t) {}
MORI_COUNTER_DEVICE_FN void EpStatsDispatchRecvChunk(uint64_t*, int, int, uint64_t) {}
MORI_COUNTER_DEVICE_FN void EpStatsCall(uint64_t*, int, int) {}
MORI_COUNTER_DEVICE_FN void EpStatsSeqBegin(uint64_t*, int, int) {}
MORI_COUNTER_DEVICE_FN void EpStatsCombinePeer(uint64_t*, int, int) {}
struct EpStatsLaunch {
  MORI_COUNTER_DEVICE_FN EpStatsLaunch(uint64_t*, int, int, bool, int, int) {}
};
struct EpStatsSpan {
  MORI_COUNTER_DEVICE_FN EpStatsSpan(uint64_t*, int, int, int, int, int, int) {}
};

#endif  // MORI_ENABLE_METRICS
#endif  // __HIPCC__ || __CUDACC__

}  // namespace moe
}  // namespace mori
