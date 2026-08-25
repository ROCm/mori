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
#include "umbp/local/tiers/dram_tier.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "device_copy_run.h"
#include "device_gather.h"
#include "host_registration.h"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_copy.h"
#include "umbp/common/range_utils.h"

namespace mori::umbp {

namespace {

#if defined(__x86_64__) || defined(__i386__)
// Non-temporal AVX2 (256-bit) copy: streaming stores bypass the cache and skip
// the read-for-ownership (RFO) on dst. This is the right choice for the real
// batch-get path, where each KV block is read ONCE from cold DRAM and the
// working set far exceeds L3: a cached copy moves 3x the block bytes through
// memory (read src + RFO dst + writeback dst), NT moves only 2x (read src +
// stream-write dst).
//
// Width: AVX2 (256-bit), not AVX-512. On Zen4 the 512-bit datapath is
// double-pumped over 256-bit units, so 512-bit stream stores give no real
// width advantage and can trip AVX-512 frequency throttling; the NT bottleneck
// is the write-combining buffer drain rate, which 256-bit already saturates.
// Measured cold 4 MiB blocks, 8 threads (no pinning) on Zen4 EPYC:
//   avx2_nt ~134  >  avx512_nt ~130  >  glibc memcpy ~88  >  cached storeu ~77.
// dst is a host pinned buffer; sfence orders the streaming stores before the
// subsequent host->device DMA reads it.
__attribute__((target("avx2"))) void NtCopyAvx2(char* d, const char* s, size_t n) {
  size_t head = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
  if (head > n) head = n;
  std::memcpy(d, s, head);
  size_t i = head;
  for (; i + 128 <= n; i += 128) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
    __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
    __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), b);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), c);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), e);
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
  }
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}
bool Avx2Supported() { return __builtin_cpu_supports("avx2"); }
#else
void NtCopyAvx2(char* d, const char* s, size_t n) { std::memcpy(d, s, n); }
bool Avx2Supported() { return false; }
#endif

// Copy one KV block. Large blocks (>= 256 KiB, the real KV-page regime, always
// cold DRAM) use non-temporal stores (~1.5x over memcpy on Zen4). Tiny blocks
// fall back to glibc memcpy (its small-copy path is faster and they may be hot).
// Disable NT via UMBP_DRAM_NT_COPY=0.
inline void CopyBlock(void* dst, const void* src, size_t size) {
  static const bool kNt = Avx2Supported() && !(std::getenv("UMBP_DRAM_NT_COPY") &&
                                               std::getenv("UMBP_DRAM_NT_COPY")[0] == '0');
  static const size_t kNtMinBytes = 256ull << 10;
  if (kNt && size >= kNtMinBytes) {
    NtCopyAvx2(static_cast<char*>(dst), static_cast<const char*>(src), size);
  } else {
    std::memcpy(dst, src, size);
  }
}

struct CopyJob {
  void* dst;
  const void* src;
  const void* classified_ptr;
  size_t size;
  size_t idx;
  bool copied{false};
};

using detail::DeviceCopyRun;
using detail::DeviceCopyRunKind;
using detail::FindDeviceCopyRun;
using detail::PitchedCopyEnabled;
using detail::PointerAddress;

// ---------------------------------------------------------------------------
// Opt-in profiling (UMBP_DRAM_PROFILE=1). Answers the two questions that gate
// the next round of optimization work:
//   1. how much of a batch's wall time is spent waiting for mu_ (i.e. how much
//      the 8 ranks serialize against each other);
//   2. how many device-copy submissions survive 1D and pitched-2D merging.
// Zero cost when disabled: one predictable branch per batch.
// ---------------------------------------------------------------------------

bool ProfileEnabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("UMBP_DRAM_PROFILE");
    return raw != nullptr && raw[0] == '1';
  }();
  return enabled;
}

// Report every N batches so a normal E2E run produces a handful of lines.
constexpr uint64_t kProfileReportEvery = 512;

// How many batches are inside the tier at the same instant. Zero lock wait can
// mean either "no contention" or "requests never arrive together"; this tells
// them apart, which decides whether finer locking could buy anything at all.
struct InFlightStats {
  std::atomic<int> current{0};
  std::atomic<int> peak{0};
  std::atomic<uint64_t> samples{0};
  std::atomic<uint64_t> sum{0};
};

InFlightStats& BatchInFlight() {
  static InFlightStats stats;
  return stats;
}

class ScopedInFlight {
 public:
  ScopedInFlight() {
    if (!ProfileEnabled()) return;
    active_ = true;
    InFlightStats& stats = BatchInFlight();
    const int now = stats.current.fetch_add(1, std::memory_order_relaxed) + 1;
    int observed = stats.peak.load(std::memory_order_relaxed);
    while (now > observed &&
           !stats.peak.compare_exchange_weak(observed, now, std::memory_order_relaxed)) {
    }
    stats.sum.fetch_add(static_cast<uint64_t>(now), std::memory_order_relaxed);
    stats.samples.fetch_add(1, std::memory_order_relaxed);
  }
  ~ScopedInFlight() {
    if (!active_) return;
    BatchInFlight().current.fetch_sub(1, std::memory_order_relaxed);
  }

 private:
  bool active_{false};
};

struct LockStats {
  const char* name;
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> wait_us{0};
  std::atomic<uint64_t> held_us{0};
  std::atomic<uint64_t> objects{0};
};

LockStats& ReadLockStats() {
  static LockStats stats{"ReadBatchIntoPtr"};
  return stats;
}

LockStats& WriteLockStats() {
  static LockStats stats{"BatchWrite"};
  return stats;
}

void RecordLockStats(LockStats& stats, uint64_t wait_us, uint64_t held_us, size_t objects) {
  stats.wait_us.fetch_add(wait_us, std::memory_order_relaxed);
  stats.held_us.fetch_add(held_us, std::memory_order_relaxed);
  stats.objects.fetch_add(objects, std::memory_order_relaxed);
  const uint64_t n = stats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % kProfileReportEvery != 0) return;

  const uint64_t wait = stats.wait_us.load(std::memory_order_relaxed);
  const uint64_t held = stats.held_us.load(std::memory_order_relaxed);
  const uint64_t objs = stats.objects.load(std::memory_order_relaxed);
  const uint64_t total = wait + held;
  const InFlightStats& flight = BatchInFlight();
  const uint64_t samples = flight.samples.load(std::memory_order_relaxed);
  const uint64_t sum = flight.sum.load(std::memory_order_relaxed);
  MORI_UMBP_INFO(
      "[DRAMTier/profile] {} batches={} objects={} avg_lock_wait={}us avg_lock_held={}us "
      "wait_share={}% | in_flight avg={}.{:02d} peak={}",
      stats.name, n, objs, wait / n, held / n, total == 0 ? 0 : (wait * 100 / total),
      samples == 0 ? 0 : sum / samples, samples == 0 ? 0 : (sum * 100 / samples) % 100,
      flight.peak.load(std::memory_order_relaxed));
}

struct MergeStats {
  const char* name;
  std::atomic<uint64_t> batches{0};
  std::atomic<uint64_t> objects{0};
  std::atomic<uint64_t> runs_gpu{0};   // runs if only the GPU side had to be adjacent
  std::atomic<uint64_t> runs_dram{0};  // runs if only the DRAM side had to be adjacent
  std::atomic<uint64_t> runs_both{0};  // runs if both sides had to be adjacent
  std::atomic<uint64_t> actual_submissions{0};
  std::atomic<uint64_t> pitched_submissions{0};
  std::atomic<uint64_t> gather_batches{0};  // batches carried by one kernel launch instead
};

MergeStats& OffloadMergeStats() {
  static MergeStats stats{"offload D2H (src=GPU dst=DRAM)"};
  return stats;
}

MergeStats& LoadMergeStats() {
  static MergeStats stats{"load    H2D (src=DRAM dst=GPU)"};
  return stats;
}

// Counts both the legacy adjacency-only view and the actual 1D/2D plan.
// runs_gpu / runs_dram show which side blocks flat 1D merging; actual_submissions
// includes pitched runs recognized by the same planner used for execution.
void RecordMergeability(const std::vector<CopyJob*>& jobs, hipMemcpyKind kind) {
  if (jobs.empty()) return;
  const bool src_is_gpu = (kind == hipMemcpyDeviceToHost);
  MergeStats& stats = src_is_gpu ? OffloadMergeStats() : LoadMergeStats();

  auto gpu_of = [src_is_gpu](const CopyJob* j) {
    return static_cast<const char*>(src_is_gpu ? j->src : static_cast<const void*>(j->dst));
  };
  auto dram_of = [src_is_gpu](const CopyJob* j) {
    return static_cast<const char*>(src_is_gpu ? static_cast<const void*>(j->dst) : j->src);
  };

  // Sort by the DRAM side: that is the ordering we could actually impose.
  std::vector<const CopyJob*> sorted(jobs.begin(), jobs.end());
  std::sort(sorted.begin(), sorted.end(),
            [&](const CopyJob* a, const CopyJob* b) { return dram_of(a) < dram_of(b); });

  uint64_t runs_gpu = 1, runs_dram = 1, runs_both = 1;
  for (size_t i = 1; i < sorted.size(); ++i) {
    const CopyJob* prev = sorted[i - 1];
    const CopyJob* cur = sorted[i];
    const bool gpu_adjacent = gpu_of(prev) + prev->size == gpu_of(cur);
    const bool dram_adjacent = dram_of(prev) + prev->size == dram_of(cur);
    if (!gpu_adjacent) ++runs_gpu;
    if (!dram_adjacent) ++runs_dram;
    if (!gpu_adjacent || !dram_adjacent) ++runs_both;
  }

  stats.objects.fetch_add(sorted.size(), std::memory_order_relaxed);
  stats.runs_gpu.fetch_add(runs_gpu, std::memory_order_relaxed);
  stats.runs_dram.fetch_add(runs_dram, std::memory_order_relaxed);
  stats.runs_both.fetch_add(runs_both, std::memory_order_relaxed);

  std::vector<CopyJob*> ordered(jobs.begin(), jobs.end());
  std::sort(ordered.begin(), ordered.end(), [](const CopyJob* a, const CopyJob* b) {
    return PointerAddress(a->src) < PointerAddress(b->src);
  });
  uint64_t actual_submissions = 0;
  uint64_t pitched_submissions = 0;
  for (size_t begin = 0; begin < ordered.size();) {
    const DeviceCopyRun run = FindDeviceCopyRun(ordered, begin, PitchedCopyEnabled());
    ++actual_submissions;
    if (run.kind == DeviceCopyRunKind::kPitched) ++pitched_submissions;
    begin = run.end;
  }
  stats.actual_submissions.fetch_add(actual_submissions, std::memory_order_relaxed);
  stats.pitched_submissions.fetch_add(pitched_submissions, std::memory_order_relaxed);
  const uint64_t n = stats.batches.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % kProfileReportEvery != 0) return;

  const uint64_t objs = stats.objects.load(std::memory_order_relaxed);
  auto pct = [objs](const std::atomic<uint64_t>& runs) {
    return objs == 0 ? 100 : runs.load(std::memory_order_relaxed) * 100 / objs;
  };
  MORI_UMBP_INFO(
      "[DRAMTier/profile] mergeability {} batches={} objects={} | submissions kept: "
      "gpu-side-only={}% dram-side-only={}% adjacent-both={}% actual={}% 2d_runs={} "
      "gather_batches={}",
      stats.name, n, objs, pct(stats.runs_gpu), pct(stats.runs_dram), pct(stats.runs_both),
      pct(stats.actual_submissions), stats.pitched_submissions.load(std::memory_order_relaxed),
      stats.gather_batches.load(std::memory_order_relaxed));
}

// Where a batch's time goes once it is inside the tier. Added because the
// end-to-end loader was running at a small fraction of the copy path's
// measured ceiling under 8-rank concurrency, and the copy itself was the only
// part anyone had timed. Classification is called out separately because it
// costs one hipPointerGetAttributes per range, which scales with objects times
// layers rather than with bytes.
struct PhaseStats {
  const char* name;
  std::atomic<uint64_t> batches{0};
  std::atomic<uint64_t> ranges{0};
  std::atomic<uint64_t> bytes{0};        // measured, not inferred from a mean range size
  std::atomic<uint64_t> lookup_us{0};    // resolving keys to slots
  std::atomic<uint64_t> classify_us{0};  // DetectPointerLocation per range
  std::atomic<uint64_t> copy_us{0};      // the transfer itself
};

PhaseStats& ReadPhaseStats() {
  static PhaseStats stats{"read"};
  return stats;
}

PhaseStats& WritePhaseStats() {
  static PhaseStats stats{"write"};
  return stats;
}

void RecordPhases(PhaseStats& stats, size_t ranges, size_t bytes, uint64_t lookup_us,
                  uint64_t classify_us, uint64_t copy_us) {
  stats.ranges.fetch_add(ranges, std::memory_order_relaxed);
  stats.bytes.fetch_add(bytes, std::memory_order_relaxed);
  stats.lookup_us.fetch_add(lookup_us, std::memory_order_relaxed);
  stats.classify_us.fetch_add(classify_us, std::memory_order_relaxed);
  stats.copy_us.fetch_add(copy_us, std::memory_order_relaxed);
  const uint64_t n = stats.batches.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % kProfileReportEvery != 0) return;

  const uint64_t lookup = stats.lookup_us.load(std::memory_order_relaxed);
  const uint64_t classify = stats.classify_us.load(std::memory_order_relaxed);
  const uint64_t copy = stats.copy_us.load(std::memory_order_relaxed);
  const uint64_t moved = stats.bytes.load(std::memory_order_relaxed);
  const uint64_t total = lookup + classify + copy;
  MORI_UMBP_INFO(
      "[DRAMTier/profile] phases {} batches={} ranges={} | per batch: KiB={} lookup={}us "
      "classify={}us copy={}us | share: lookup={}% classify={}% copy={}% | copy_MBps={}",
      stats.name, n, stats.ranges.load(std::memory_order_relaxed), moved / n / 1024, lookup / n,
      classify / n, copy / n, total == 0 ? 0 : lookup * 100 / total,
      total == 0 ? 0 : classify * 100 / total, total == 0 ? 0 : copy * 100 / total,
      copy == 0 ? 0 : moved / copy);
}

// Reads the clock only when profiling is on: these sit on the per-batch path,
// so an unconditional steady_clock::now() in the constructor would be a cost
// paid by every batch in production to serve a switch that is normally off.
class PhaseTimer {
 public:
  explicit PhaseTimer(bool enabled) : enabled_(enabled) {
    if (enabled_) started_ = std::chrono::steady_clock::now();
  }

  uint64_t Lap() {
    if (!enabled_) return 0;
    const auto now = std::chrono::steady_clock::now();
    const uint64_t elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - started_).count();
    started_ = now;
    return elapsed;
  }

 private:
  bool enabled_;
  std::chrono::steady_clock::time_point started_{};
};

// Times mu_ acquisition and the critical section, then reports periodically.
template <typename Lock>
class ProfiledLock {
 public:
  template <typename Mutex>
  ProfiledLock(Mutex& mutex, LockStats& stats, size_t objects)
      : lock_(mutex, std::defer_lock), stats_(stats), objects_(objects) {
    if (!ProfileEnabled()) {
      lock_.lock();
      return;
    }
    profiling_ = true;
    const auto before = std::chrono::steady_clock::now();
    lock_.lock();
    acquired_ = std::chrono::steady_clock::now();
    wait_us_ = std::chrono::duration_cast<std::chrono::microseconds>(acquired_ - before).count();
  }

  ~ProfiledLock() {
    if (!profiling_) return;
    const auto held = std::chrono::steady_clock::now() - acquired_;
    RecordLockStats(stats_, wait_us_,
                    std::chrono::duration_cast<std::chrono::microseconds>(held).count(), objects_);
  }

 private:
  Lock lock_;
  LockStats& stats_;
  size_t objects_;
  bool profiling_{false};
  uint64_t wait_us_{0};
  std::chrono::steady_clock::time_point acquired_{};
};

using ProfiledSharedLock = ProfiledLock<std::shared_lock<std::shared_mutex>>;
using ProfiledUniqueLock = ProfiledLock<std::unique_lock<std::shared_mutex>>;

void CopyHostJobs(const std::vector<CopyJob*>& jobs, int num_threads) {
  if (jobs.empty()) return;
  num_threads = std::max(1, std::min(num_threads, static_cast<int>(jobs.size())));
  if (num_threads == 1) {
    for (CopyJob* job : jobs) {
      CopyBlock(job->dst, job->src, job->size);
      job->copied = true;
    }
    return;
  }

  std::atomic<size_t> next{0};
  auto worker = [&]() {
    size_t i = 0;
    while ((i = next.fetch_add(1)) < jobs.size()) {
      CopyJob* job = jobs[i];
      CopyBlock(job->dst, job->src, job->size);
      job->copied = true;
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) pool.emplace_back(worker);
  for (auto& thread : pool) thread.join();
}

// One reusable stream per (thread, device).
//
// Reusing a stream is what makes the batch path affordable: creating and
// destroying one around every batch costs ~6.5 ms per pair on MI355X/ROCm 7.2
// (measured), which dwarfs the copies themselves.
//
// Per thread, not just per device, because a batch ends with
// hipStreamSynchronize. Sharing one stream across threads makes every batch
// wait for every other batch queued on that device -- loads behind loads, and
// loads behind the offload path, which writes continuously under
// write_through_threshold=1. That coupling, not the DMA, was what held the
// measured copy phase to 1.7 GB/s against 49 GB/s for the same shape in
// isolation.
//
// The streams are intentionally never destroyed: tearing down HIP resources
// from a thread or static destructor races with runtime teardown. The count is
// bounded by the server's worker threads times the devices they touch.
std::map<int, hipStream_t>& DeviceStreamCache() {
  static thread_local std::map<int, hipStream_t>* streams = new std::map<int, hipStream_t>();
  return *streams;
}

hipStream_t DeviceStream(int device_id) {
  std::map<int, hipStream_t>& streams = DeviceStreamCache();
  auto it = streams.find(device_id);
  if (it != streams.end()) return it->second;

  hipStream_t stream = nullptr;
  const hipError_t status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[DRAMTier] hipStreamCreateWithFlags failed on device {}: {}", device_id,
                    hipGetErrorString(status));
    (void)hipGetLastError();
    return nullptr;
  }
  streams.emplace(device_id, stream);
  return stream;
}

// Forget a stream that reported a failure so the next batch builds a fresh one.
// The old handle is intentionally not destroyed, for the same reason the cache
// never destroys any of them. This does not repair a context that a fault has
// already made unusable; it only avoids reusing a stream known to have failed.
void DiscardDeviceStream(int device_id) { DeviceStreamCache().erase(device_id); }

// Circuit breaker for the gather kernel.
//
// Not a general retry policy -- it exists because the caller upstream cannot
// distinguish a failed copy from a full tier (LocalStorageManager answers both
// by evicting a victim and retrying), so any systematic data-path failure turns
// into a tier-wide eviction storm. One latch, never reset: a kernel that has
// failed once on this host is not something to keep probing on the request
// path.
std::atomic<bool>& GatherDisabled() {
  static std::atomic<bool> disabled{false};
  return disabled;
}

bool GatherPathHealthy() { return !GatherDisabled().load(std::memory_order_relaxed); }

void DisableGatherPath() {
  if (GatherDisabled().exchange(true, std::memory_order_relaxed)) return;
  MORI_UMBP_ERROR(
      "[DRAMTier] disabling the GPU gather path after a failed copy; "
      "falling back to the copy engine for the rest of this process");
}

// The side of the copy that lives in the tier, and so the only side that
// HostTierRegistration can vouch for.
const void* TierSide(const CopyJob* job, hipMemcpyKind kind) {
  return kind == hipMemcpyDeviceToHost ? job->dst : job->src;
}

// The copy engine still wins on a handful of large runs: measured break-even
// with 8 ranks active is around 128 KiB per run, below which per-submission
// cost dominates and above which one hipMemcpyAsync already amortizes it.
constexpr size_t kGatherRunBytesThreshold = 128ull << 10;

// Flattens the batch into one fragment per merged run and decides whether a
// single kernel launch should carry them. Returns false to leave the batch on
// the copy-engine path -- which is also what happens for any run whose tier
// side is not registered, since a kernel would fault on it.
bool PlanGatherFragments(const std::vector<CopyJob*>& ordered, hipMemcpyKind kind, int device_id,
                         const HostTierRegistration* registration,
                         std::vector<DeviceGatherFragment>* fragments) {
  if (registration == nullptr || !DeviceGatherEnabled()) return false;
  if (!registration->GatherableOn(device_id)) return false;

  fragments->clear();
  fragments->reserve(ordered.size());
  size_t total_bytes = 0;
  for (size_t begin = 0; begin < ordered.size();) {
    const DeviceCopyRun run = FindDeviceCopyRun(ordered, begin, /*allow_pitched=*/false);
    const CopyJob* first = ordered[begin];
    // Only the tier side is host memory and gets translated; the other side is
    // already a device allocation. Translating the wrong one silently writes to
    // the wrong place, so both directions are spelled out. Null also covers the
    // unregistered case, which is why there is no separate Covers() test.
    void* const tier_alias =
        registration->DeviceAddress(TierSide(first, kind), run.bytes, device_id);
    if (tier_alias == nullptr) return false;
    if (kind == hipMemcpyDeviceToHost) {
      fragments->push_back({first->src, tier_alias, run.bytes});
    } else {
      fragments->push_back({tier_alias, first->dst, run.bytes});
    }
    total_bytes += run.bytes;
    begin = run.end;
  }
  return fragments->size() > 1 && total_bytes / fragments->size() < kGatherRunBytesThreshold;
}

bool SubmitCopyRuns(const std::vector<CopyJob*>& ordered, int device_id, hipMemcpyKind kind,
                    hipStream_t stream) {
  for (size_t begin = 0; begin < ordered.size();) {
    const DeviceCopyRun run = FindDeviceCopyRun(ordered, begin, PitchedCopyEnabled());
    hipError_t status = hipSuccess;
    if (run.kind == DeviceCopyRunKind::kPitched) {
      status = hipMemcpy2DAsync(ordered[begin]->dst, run.dpitch, ordered[begin]->src, run.spitch,
                                run.width, run.end - begin, kind, stream);
    } else {
      status = hipMemcpyAsync(ordered[begin]->dst, ordered[begin]->src, run.bytes, kind, stream);
    }
    if (status != hipSuccess) {
      // Worth logging the tier side explicitly: the caller cannot tell a copy
      // failure from a full tier, so it responds by evicting and retrying, and
      // a systematic failure here turns into a tier-wide eviction storm.
      MORI_UMBP_ERROR(
          "[DRAMTier] device copy failed on device {} "
          "(dir={} shape={} width={} rows={} spitch={} dpitch={} bytes={} tier_side={}): {}",
          device_id, kind == hipMemcpyDeviceToHost ? "d2h" : "h2d",
          run.kind == DeviceCopyRunKind::kPitched ? "2d" : "1d", run.width, run.end - begin,
          run.spitch, run.dpitch, run.bytes, TierSide(ordered[begin], kind),
          hipGetErrorString(status));
      (void)hipGetLastError();
      return false;
    }
    begin = run.end;
  }
  return true;
}

void CopyDeviceJobs(const std::vector<CopyJob*>& jobs, int device_id, hipMemcpyKind kind,
                    const HostTierRegistration* registration) {
  if (jobs.empty()) return;
  ScopedHipDevice device_guard(device_id);
  if (!device_guard.IsValid()) {
    MORI_UMBP_ERROR("[DRAMTier] failed to select GPU device {}", device_id);
    return;
  }

  // Created under the device guard above, so the stream belongs to device_id.
  hipStream_t stream = DeviceStream(device_id);
  if (stream == nullptr) return;

  // Coalesce runs whose source AND destination are both adjacent into one
  // submission, which is worth doing on either path: it is what the copy engine
  // needs to avoid paying submission cost per object, and it shortens the
  // descriptor list the kernel walks. How much it finds depends on the object
  // model -- with page-granular objects the tier side strides by the object
  // size while reading one layer, so runs stay short and the kernel carries
  // most batches.
  std::vector<CopyJob*> ordered(jobs.begin(), jobs.end());
  std::sort(ordered.begin(), ordered.end(), [](const CopyJob* a, const CopyJob* b) {
    return PointerAddress(a->src) < PointerAddress(b->src);
  });

  // Page-granular objects leave runs that no memcpy shape can merge: one layer
  // per object means the tier side strides by the object size. One kernel
  // launch is ~10x the per-fragment submissions and, unlike hipMemcpy2DAsync,
  // does not collapse when the source pages are cold.
  bool enqueued_all = true;
  std::vector<DeviceGatherFragment> fragments;
  const bool gathered = GatherPathHealthy() &&
                        PlanGatherFragments(ordered, kind, device_id, registration, &fragments) &&
                        LaunchDeviceGather(fragments.data(), fragments.size(), device_id, stream);
  if (gathered) {
    if (ProfileEnabled()) {
      MergeStats& stats = kind == hipMemcpyDeviceToHost ? OffloadMergeStats() : LoadMergeStats();
      stats.gather_batches.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    enqueued_all = SubmitCopyRuns(ordered, device_id, kind, stream);
  }

  const hipError_t sync_status = hipStreamSynchronize(stream);
  if (sync_status != hipSuccess) {
    MORI_UMBP_ERROR("[DRAMTier] hipStreamSynchronize failed on device {}: {}", device_id,
                    hipGetErrorString(sync_status));
    (void)hipGetLastError();
    enqueued_all = false;
    // The caller cannot tell a failed copy from a full tier: it answers both by
    // evicting a victim and retrying, so a systematic data-path failure drains
    // the tier one entry at a time. Taking the kernel out of service means the
    // retry runs on the copy engine, which usually recovers before any eviction
    // -- but it is a mitigation, not a guarantee: if the fault also left the
    // context unusable the retry fails too. The complete fix is to distinguish
    // the failure reasons at the tier/manager boundary.
    if (gathered) DisableGatherPath();
    DiscardDeviceStream(device_id);
  }

  if (enqueued_all) {
    for (CopyJob* job : jobs) job->copied = true;
  }
}

void CopyJobs(std::vector<CopyJob>* jobs, int host_threads, hipMemcpyKind device_copy_kind,
              const HostTierRegistration* registration, uint64_t* out_classify_us = nullptr,
              uint64_t* out_copy_us = nullptr) {
  if (jobs->empty()) return;
  const bool profiling = ProfileEnabled();
  PhaseTimer timer(profiling);
  std::vector<CopyJob*> host_jobs;
  std::map<int, std::vector<CopyJob*>> device_jobs;
  host_jobs.reserve(jobs->size());
  for (CopyJob& job : *jobs) {
    const PointerLocation location = DetectPointerLocation(job.classified_ptr);
    if (location.IsDevice()) {
      device_jobs[location.device_id].push_back(&job);
    } else {
      host_jobs.push_back(&job);
    }
  }
  if (out_classify_us != nullptr) *out_classify_us = timer.Lap();

  if (profiling) {
    // Sorting and re-planning every batch is profiling-only work. Do it before
    // the copy measurement starts rather than inside the loop, where it would
    // be charged to the copy it is supposed to describe.
    for (auto& [device_id, grouped_jobs] : device_jobs) {
      RecordMergeability(grouped_jobs, device_copy_kind);
    }
    timer.Lap();
  }

  CopyHostJobs(host_jobs, host_threads);
  for (auto& [device_id, grouped_jobs] : device_jobs) {
    CopyDeviceJobs(grouped_jobs, device_id, device_copy_kind, registration);
  }
  if (out_copy_us != nullptr) *out_copy_us = timer.Lap();
}

}  // namespace

DRAMTier::DRAMTier(size_t capacity, bool use_shm, const std::string& shm_name, bool use_hugepages,
                   size_t hugepage_size, int numa_node, bool prefault)
    : TierBackend(StorageTier::CPU_DRAM),
      base_ptr_(nullptr),
      capacity_(capacity),
      mapped_size_(0),
      used_(0),
      shm_fd_(-1),
      use_shm_(use_shm),
      shm_name_(shm_name) {
  if (use_shm_) {
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) {
      throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
    }
    if (ftruncate(shm_fd_, capacity_) < 0) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
      throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
    }
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (base_ptr_ == MAP_FAILED) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
      throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
    }
    mapped_size_ = capacity_;
  } else {
    HostMemAllocator allocator;
    HostBufferOptions opts;
    opts.backing =
        use_hugepages ? HostBufferBacking::kAnonymousHugetlb : HostBufferBacking::kAnonymous;
    opts.hugepage_size = hugepage_size;
    opts.numa_node = numa_node;
    opts.prefault = prefault;

    host_buf_handle_ = allocator.Alloc(capacity_, opts);
    if (!host_buf_handle_.valid()) {
      throw std::runtime_error("DRAMTier: memory allocation failed for " +
                               std::to_string(capacity_) + " bytes");
    }
    base_ptr_ = host_buf_handle_.ptr;
    mapped_size_ = host_buf_handle_.mapped_size;
  }

  // Initialize free list with entire capacity
  free_list_.push_back({0, capacity_});

  // Must come after base_ptr_ is final. Registering the region is what lets the
  // gather kernel dereference it, and it also moves every hipMemcpy on this
  // tier off the pageable staging path.
  host_registration_ = std::make_unique<HostTierRegistration>(base_ptr_, mapped_size_);

  // Threads for parallel batch-read CopyBlock. Default 8, override via env,
  // capped to hardware concurrency. >1 breaks the single-core memcpy ceiling.
  if (const char* e = std::getenv("UMBP_DRAM_READ_THREADS")) {
    int v = std::atoi(e);
    if (v >= 1) read_threads_ = v;
  }
  if (const char* e = std::getenv("UMBP_DRAM_WRITE_THREADS")) {
    int v = std::atoi(e);
    if (v >= 1) write_threads_ = v;
  }
  unsigned hc = std::thread::hardware_concurrency();
  if (hc > 0 && read_threads_ > static_cast<int>(hc)) read_threads_ = static_cast<int>(hc);
  if (read_threads_ < 1) read_threads_ = 1;
  if (hc > 0 && write_threads_ > static_cast<int>(hc)) write_threads_ = static_cast<int>(hc);
  if (write_threads_ < 1) write_threads_ = 1;
}

DRAMTier::~DRAMTier() {
  // Unregister before the mapping goes away, and before any in-flight
  // background registration can touch an unmapped address.
  host_registration_.reset();
  if (use_shm_) {
    if (base_ptr_ && base_ptr_ != MAP_FAILED) {
      munmap(base_ptr_, mapped_size_);
    }
    if (shm_fd_ >= 0) close(shm_fd_);
    shm_unlink(shm_name_.c_str());
  } else {
    HostMemAllocator allocator;
    allocator.Free(host_buf_handle_);
  }
}

size_t DRAMTier::Allocate(size_t size) {
  // First-fit allocation
  for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
    if (it->size >= size) {
      size_t offset = it->offset;
      if (it->size == size) {
        free_list_.erase(it);
      } else {
        it->offset += size;
        it->size -= size;
      }
      return offset;
    }
  }
  return static_cast<size_t>(-1);  // Allocation failed
}

void DRAMTier::Deallocate(size_t offset, size_t size) {
  // Insert into sorted position and coalesce adjacent blocks
  auto it = free_list_.begin();
  while (it != free_list_.end() && it->offset < offset) {
    ++it;
  }

  auto new_it = free_list_.insert(it, {offset, size});

  // Coalesce with next block
  auto next = std::next(new_it);
  if (next != free_list_.end() && new_it->offset + new_it->size == next->offset) {
    new_it->size += next->size;
    free_list_.erase(next);
  }

  // Coalesce with previous block
  if (new_it != free_list_.begin()) {
    auto prev = std::prev(new_it);
    if (prev->offset + prev->size == new_it->offset) {
      prev->size += new_it->size;
      free_list_.erase(new_it);
    }
  }
}

void DRAMTier::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
}

void DRAMTier::EvictLRU() {
  std::lock_guard<std::mutex> lru_lock(lru_mu_);
  if (lru_list_.empty()) return;

  const std::string& victim = lru_list_.back();
  auto slot_it = slots_.find(victim);
  if (slot_it != slots_.end()) {
    Deallocate(slot_it->second.offset, slot_it->second.size);
    used_ -= slot_it->second.size;
    slots_.erase(slot_it);
  }
  lru_map_.erase(victim);
  lru_list_.pop_back();
}

bool DRAMTier::Write(const std::string& key, const void* data, size_t size) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  // If key already exists, free its old slot first
  auto existing = slots_.find(key);
  if (existing != slots_.end()) {
    Deallocate(existing->second.offset, existing->second.size);
    used_ -= existing->second.size;
    slots_.erase(existing);
    {
      std::lock_guard<std::mutex> lru_lock(lru_mu_);
      auto lru_it = lru_map_.find(key);
      if (lru_it != lru_map_.end()) {
        lru_list_.erase(lru_it->second);
        lru_map_.erase(lru_it);
      }
    }
  }

  // Try to allocate — do NOT self-evict.
  // If no space, return false so upper layer can demote keys to SSD.
  size_t offset = Allocate(size);
  if (offset == static_cast<size_t>(-1)) {
    return false;
  }

  const PointerLocation source_location = DetectPointerLocation(data);
  if (source_location.IsDevice()) {
    ScopedHipDevice device_guard(source_location.device_id);
    if (!device_guard.IsValid() || !DeviceCopy(static_cast<char*>(base_ptr_) + offset, data, size,
                                               hipMemcpyDeviceToHost, source_location.device_id)) {
      Deallocate(offset, size);
      return false;
    }
  } else {
    std::memcpy(static_cast<char*>(base_ptr_) + offset, data, size);
  }
  slots_[key] = {offset, size};
  used_ += size;
  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    TouchLRU(key);
  }
  return true;
}

bool DRAMTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  // Reject if caller's buffer size does not match the stored block size.
  // A mismatch indicates a caller bug (wrong page size); silently truncating
  // would produce a partially-filled KV block with no error signal.
  if (size != it->second.size) return false;

  void* dst = reinterpret_cast<void*>(dst_ptr);
  const PointerLocation destination_location = DetectPointerLocation(dst);
  if (destination_location.IsDevice()) {
    ScopedHipDevice device_guard(destination_location.device_id);
    if (!device_guard.IsValid() ||
        !DeviceCopy(dst, static_cast<char*>(base_ptr_) + it->second.offset, size,
                    hipMemcpyHostToDevice, destination_location.device_id)) {
      return false;
    }
  } else {
    std::memcpy(dst, static_cast<char*>(base_ptr_) + it->second.offset, size);
  }
  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    TouchLRU(key);
  }
  return true;
}

std::vector<bool> DRAMTier::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (n == 0) return results;

  // Hold shared ownership for the whole batch so slot offsets remain valid
  // during CopyJobs. Other readers may proceed concurrently; writers and
  // eviction wait until every copy has finished.
  ScopedInFlight in_flight;
  ProfiledSharedLock lock(mu_, ReadLockStats(), n);

  std::vector<CopyJob> jobs;
  jobs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto it = slots_.find(keys[i]);
    if (it == slots_.end()) continue;
    if (sizes[i] != it->second.size) continue;
    void* dst = reinterpret_cast<void*>(dst_ptrs[i]);
    jobs.push_back(
        {dst, static_cast<char*>(base_ptr_) + it->second.offset, dst, sizes[i], i, false});
  }

  CopyJobs(&jobs, read_threads_, hipMemcpyHostToDevice, host_registration_.get());
  if (!jobs.empty()) {
    // One acquisition per batch avoids recreating the per-object lock convoy
    // that the server-side range-resolution change removes.
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    for (const auto& job : jobs) {
      if (!job.copied) continue;
      results[job.idx] = true;
      TouchLRU(keys[job.idx]);
    }
  }
  return results;
}

std::vector<bool> DRAMTier::ReadBatchRangesIntoPtr(
    const std::vector<std::string>& keys, const std::vector<std::vector<uintptr_t>>& dst_ptrs,
    const std::vector<std::vector<size_t>>& sizes,
    const std::vector<std::vector<size_t>>& src_offsets) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (!RangeBatchShapeValid(n, dst_ptrs, sizes, src_offsets) || n == 0) return results;

  size_t total_ranges = 0;
  for (const auto& entry : sizes) {
    if (entry.size() > std::numeric_limits<size_t>::max() - total_ranges) return results;
    total_ranges += entry.size();
  }

  ScopedInFlight in_flight;
  ProfiledSharedLock lock(mu_, ReadLockStats(), n);

  const bool profiling = ProfileEnabled();
  PhaseTimer phase(profiling);
  size_t job_bytes = 0;
  std::vector<CopyJob> jobs;
  jobs.reserve(total_ranges);
  std::vector<size_t> expected(n, 0);
  for (size_t i = 0; i < n; ++i) {
    if (sizes[i].empty()) continue;
    auto it = slots_.find(keys[i]);
    if (it == slots_.end()) continue;

    bool valid = true;
    for (size_t j = 0; j < sizes[i].size(); ++j) {
      if (sizes[i][j] == 0 ||
          IsObjectRangeOverflow(src_offsets[i][j], sizes[i][j], it->second.size)) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    expected[i] = sizes[i].size();
    for (size_t j = 0; j < sizes[i].size(); ++j) {
      void* dst = reinterpret_cast<void*>(dst_ptrs[i][j]);
      const void* src = static_cast<const char*>(base_ptr_) + it->second.offset + src_offsets[i][j];
      jobs.push_back({dst, src, dst, sizes[i][j], i, false});
      job_bytes += sizes[i][j];
    }
  }

  const uint64_t lookup_us = phase.Lap();
  uint64_t classify_us = 0;
  uint64_t copy_us = 0;
  CopyJobs(&jobs, read_threads_, hipMemcpyHostToDevice, host_registration_.get(), &classify_us,
           &copy_us);
  if (profiling) {
    RecordPhases(ReadPhaseStats(), total_ranges, job_bytes, lookup_us, classify_us, copy_us);
  }
  std::vector<size_t> copied(n, 0);
  for (const auto& job : jobs) {
    if (job.copied) ++copied[job.idx];
  }

  std::lock_guard<std::mutex> lru_lock(lru_mu_);
  for (size_t i = 0; i < n; ++i) {
    if (expected[i] != 0 && copied[i] == expected[i]) {
      results[i] = true;
      TouchLRU(keys[i]);
    }
  }
  return results;
}

std::vector<bool> DRAMTier::BatchWrite(const std::vector<std::string>& keys,
                                       const std::vector<const void*>& data_ptrs,
                                       const std::vector<size_t>& sizes) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (n == 0) return results;

  // Hold unique ownership for the whole batch. Slot allocation and payload
  // publication mutate shared state, and reserved slots must not be reused
  // while CopyJobs is filling them.
  //
  // Phase 2 could in principle run outside the lock (the reserved-but-
  // unregistered region is unreachable to other threads); ProfiledLock measures
  // what that would be worth. See UMBP_STANDALONE_PROCESS_DESIGN.md 11.1.
  ScopedInFlight in_flight;
  ProfiledUniqueLock lock(mu_, WriteLockStats(), n);

  struct WriteJob {
    CopyJob copy;
    size_t offset;
  };
  std::vector<WriteJob> write_jobs;
  std::vector<CopyJob> copy_jobs;
  write_jobs.reserve(n);

  // Phase 1 (serial): free any existing slot for the key, then allocate. Does
  // NOT self-evict — a key that doesn't fit is left false so the upper layer
  // (LocalStorageManager) can demote LRU keys and retry per-key.
  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    for (size_t i = 0; i < n; ++i) {
      auto existing = slots_.find(keys[i]);
      if (existing != slots_.end()) {
        Deallocate(existing->second.offset, existing->second.size);
        used_ -= existing->second.size;
        slots_.erase(existing);
        auto lru_it = lru_map_.find(keys[i]);
        if (lru_it != lru_map_.end()) {
          lru_list_.erase(lru_it->second);
          lru_map_.erase(lru_it);
        }
      }
      size_t offset = Allocate(sizes[i]);
      if (offset == static_cast<size_t>(-1)) continue;
      write_jobs.push_back(
          {{static_cast<char*>(base_ptr_) + offset, data_ptrs[i], data_ptrs[i], sizes[i], i, false},
           offset});
    }
  }

  // Phase 2: preserve the existing parallel host path; GPU jobs are grouped by
  // device and submitted to one stream with one synchronization per device.
  copy_jobs.reserve(write_jobs.size());
  for (const auto& job : write_jobs) copy_jobs.push_back(job.copy);
  CopyJobs(&copy_jobs, write_threads_, hipMemcpyDeviceToHost, host_registration_.get());

  // Phase 3 (serial): register slots + LRU, mark successes.
  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    for (size_t i = 0; i < write_jobs.size(); ++i) {
      const WriteJob& write_job = write_jobs[i];
      const CopyJob& copy_job = copy_jobs[i];
      if (!copy_job.copied) {
        Deallocate(write_job.offset, copy_job.size);
        continue;
      }
      slots_[keys[copy_job.idx]] = {write_job.offset, copy_job.size};
      used_ += copy_job.size;
      TouchLRU(keys[copy_job.idx]);
      results[copy_job.idx] = true;
    }
  }
  return results;
}

std::vector<bool> DRAMTier::BatchWriteRanges(const std::vector<std::string>& keys,
                                             const std::vector<size_t>& object_sizes,
                                             const std::vector<std::vector<const void*>>& src_ptrs,
                                             const std::vector<std::vector<size_t>>& sizes,
                                             const std::vector<std::vector<size_t>>& dst_offsets) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (object_sizes.size() != n || !RangeBatchShapeValid(n, src_ptrs, sizes, dst_offsets) ||
      n == 0) {
    return results;
  }

  size_t total_ranges = 0;
  for (const auto& entry : sizes) {
    if (entry.size() > std::numeric_limits<size_t>::max() - total_ranges) return results;
    total_ranges += entry.size();
  }

  std::unordered_set<std::string> seen;
  std::unordered_set<std::string> duplicates;
  for (const auto& key : keys) {
    if (!seen.insert(key).second) duplicates.insert(key);
  }

  ScopedInFlight in_flight;
  ProfiledUniqueLock lock(mu_, WriteLockStats(), n);

  const bool profiling = ProfileEnabled();
  PhaseTimer phase(profiling);
  size_t job_bytes = 0;
  struct Reservation {
    size_t index;
    size_t offset;
    size_t object_size;
    size_t job_begin;
    size_t job_count;
  };
  std::vector<Reservation> reservations;
  std::vector<CopyJob> jobs;
  reservations.reserve(n);
  jobs.reserve(total_ranges);

  // Reserve new storage while any previous value remains published. This costs
  // temporary extra space, but it is what makes replacement failure atomic.
  for (size_t i = 0; i < n; ++i) {
    if (duplicates.count(keys[i]) != 0 ||
        !RangesTileObject(object_sizes[i], sizes[i], dst_offsets[i])) {
      continue;
    }
    const size_t offset = Allocate(object_sizes[i]);
    if (offset == static_cast<size_t>(-1)) continue;

    const size_t job_begin = jobs.size();
    for (size_t j = 0; j < sizes[i].size(); ++j) {
      void* dst = static_cast<char*>(base_ptr_) + offset + dst_offsets[i][j];
      jobs.push_back({dst, src_ptrs[i][j], src_ptrs[i][j], sizes[i][j], i, false});
      job_bytes += sizes[i][j];
    }
    reservations.push_back({i, offset, object_sizes[i], job_begin, sizes[i].size()});
  }

  const uint64_t lookup_us = phase.Lap();
  uint64_t classify_us = 0;
  uint64_t copy_us = 0;
  CopyJobs(&jobs, write_threads_, hipMemcpyDeviceToHost, host_registration_.get(), &classify_us,
           &copy_us);
  if (profiling) {
    RecordPhases(WritePhaseStats(), total_ranges, job_bytes, lookup_us, classify_us, copy_us);
  }

  std::lock_guard<std::mutex> lru_lock(lru_mu_);
  for (const Reservation& reservation : reservations) {
    bool copied = true;
    for (size_t j = 0; j < reservation.job_count; ++j) {
      if (!jobs[reservation.job_begin + j].copied) {
        copied = false;
        break;
      }
    }
    if (!copied) {
      Deallocate(reservation.offset, reservation.object_size);
      continue;
    }

    const std::string& key = keys[reservation.index];
    auto existing = slots_.find(key);
    if (existing != slots_.end()) {
      Deallocate(existing->second.offset, existing->second.size);
      used_ -= existing->second.size;
    }
    slots_[key] = {reservation.offset, reservation.object_size};
    used_ += reservation.object_size;
    TouchLRU(key);
    results[reservation.index] = true;
  }
  return results;
}

const void* DRAMTier::ReadPtr(const std::string& key, size_t* out_size) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return nullptr;

  if (out_size) *out_size = it->second.size;
  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    TouchLRU(key);
  }
  return static_cast<char*>(base_ptr_) + it->second.offset;
}

std::vector<char> DRAMTier::Read(const std::string& key) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return {};

  size_t sz = it->second.size;
  std::vector<char> buf(sz);
  std::memcpy(buf.data(), static_cast<char*>(base_ptr_) + it->second.offset, sz);
  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    TouchLRU(key);
  }
  return buf;
}

TierCapabilities DRAMTier::Capabilities() const {
  TierCapabilities caps;
  caps.zero_copy_read = true;
  caps.batch_read = true;   // use the multi-threaded ReadBatchIntoPtr above
  caps.batch_write = true;  // use the multi-threaded BatchWrite below
  caps.ranged_read = true;
  return caps;
}

bool DRAMTier::Exists(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return slots_.count(key) > 0;
}

bool DRAMTier::Evict(const std::string& key) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  Deallocate(it->second.offset, it->second.size);
  used_ -= it->second.size;
  slots_.erase(it);

  {
    std::lock_guard<std::mutex> lru_lock(lru_mu_);
    auto lru_it = lru_map_.find(key);
    if (lru_it != lru_map_.end()) {
      lru_list_.erase(lru_it->second);
      lru_map_.erase(lru_it);
    }
  }
  return true;
}

std::pair<size_t, size_t> DRAMTier::Capacity() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return {used_, capacity_};
}

void DRAMTier::Clear() {
  std::unique_lock<std::shared_mutex> lock(mu_);
  std::lock_guard<std::mutex> lru_lock(lru_mu_);
  slots_.clear();
  lru_list_.clear();
  lru_map_.clear();
  free_list_.clear();
  free_list_.push_back({0, capacity_});
  used_ = 0;
}

std::vector<std::string> DRAMTier::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::lock_guard<std::mutex> lru_lock(lru_mu_);
  std::vector<std::string> result;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  // Walk from the back (LRU end) up to max_candidates entries.
  auto it = lru_list_.rbegin();
  for (size_t i = 0; i < max_candidates && it != lru_list_.rend(); ++i, ++it) {
    result.push_back(*it);
  }
  return result;
}

std::string DRAMTier::GetLRUKey() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::lock_guard<std::mutex> lru_lock(lru_mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::optional<size_t> DRAMTier::GetSlotOffset(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = slots_.find(key);
  if (it == slots_.end()) return std::nullopt;
  return it->second.offset;
}

std::optional<std::string> DRAMTier::GetLocationId(const std::string& key) const {
  auto offset = GetSlotOffset(key);
  if (!offset.has_value()) {
    return std::nullopt;
  }
  return std::to_string(*offset);
}

}  // namespace mori::umbp
