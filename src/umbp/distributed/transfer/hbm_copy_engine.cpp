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
#include "umbp/distributed/transfer/hbm_copy_engine.h"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {

namespace {

inline bool IsGpu(const TransferRef& r) { return r.loc == mori::io::MemoryLocationType::GPU; }

// The copy kind is a function of the PAIR, like engine selection itself.
// Spelled explicitly rather than relying on hipMemcpyDefault + unified
// addressing so a mis-registered endpoint (loc=GPU on host memory, say) fails
// loudly at the copy instead of silently taking a slower inferred path.
inline hipMemcpyKind KindFor(const TransferRef& src, const TransferRef& dst) {
  const bool s = IsGpu(src);
  const bool d = IsGpu(dst);
  if (s && d) return hipMemcpyDeviceToDevice;
  if (s) return hipMemcpyDeviceToHost;
  if (d) return hipMemcpyHostToDevice;
  return hipMemcpyHostToHost;  // unreachable: CanHandle requires one GPU side.
}

// ---------------------------------------------------------------------------
//  Gather fast path
//
//  A plan that decomposed into many small segments is the case hipMemcpy is
//  worst at: each call costs a fixed submission (~5.4 us with 8 ranks active)
//  that a 4 KiB copy cannot amortize.  One kernel doing all of them measures
//  ~0.57 us per fragment and is flat in fragment size — see the rationale on
//  LaunchDeviceGather.  Above roughly 128 KiB per fragment the copy engine wins
//  again, because one large hipMemcpyAsync already amortizes its own
//  submission.
//
//  This is what makes ranged I/O worth having on a GPU caller: reading one
//  layer out of page-granular objects is exactly "dozens of small strided
//  fragments", and it is why device_gather moved out of the DRAM tier into
//  common/ (upstream 93f2998a).
// ---------------------------------------------------------------------------

constexpr size_t kGatherFragmentThreshold = 128ULL << 10;

// Staging each prewarmed stream starts with.  A long prefix read across every
// layer reaches 8192 fragments; a bigger batch regrows its stream's staging.
constexpr size_t kPrewarmGatherFragments = 8192;
// Large enough that hipMemcpy uses a copy engine rather than a CPU write.
constexpr size_t kPrewarmCopyBytes = 4ULL << 20;

// Mirrors device_gather.hip's kVectorBytes.  Used ONLY to report, in debug
// mode, whether a segment would take the kernel's uint4 fast path or fall to
// its byte loop; it never affects a copy.  If the kernel's vector width ever
// changes, this becomes a reporting inaccuracy, not a correctness bug.
constexpr size_t kVectorBytes = 16;

// ---------------------------------------------------------------------------
//  Debug mode  (UMBP_HBM_COPY_DEBUG=1)
//
//  Turns this engine into an instrument.  For every batch it reports which of
//  the two paths carried it, why the other one declined, the exact segment
//  geometry needed to rebuild the case in a unit test, and the bandwidth
//  ACTUALLY achieved.
//
//  On measurement honesty: the per-fragment costs quoted in device_gather.h
//  (~0.57 us, "flat across fragment sizes from 8 KiB to 221 KiB") are host-side
//  submission costs, not data-movement costs -- a figure that is flat across a
//  27x size range cannot be a bandwidth.  So dividing bytes by it overstates
//  throughput badly.  What is timed here instead is launch-to-completion, and
//  both paths are already synchronous at this point -- the gather bucket does
//  its own hipStreamSynchronize, and the fallback loop uses blocking hipMemcpy
//  -- so enabling debug adds clock reads and logging but imposes NO extra
//  synchronization and does not itself change which path runs.
//
//  Knobs (all read once, on first use):
//    UMBP_HBM_COPY_DEBUG=1            enable
//    UMBP_HBM_COPY_DEBUG_SAMPLE=N     emit a per-call line every Nth call (1)
//    UMBP_HBM_COPY_DEBUG_FRAGMENTS=N  per-call segment detail lines (16; 0=off)
//    UMBP_HBM_COPY_DEBUG_SUMMARY_SEC=N cumulative rollup cadence (10; 0=off)
//
//  Sampling exists because a serving run calls this millions of times; the
//  rollup is what you read for a bandwidth answer, the per-call lines are what
//  you read to build the reproducer.
// ---------------------------------------------------------------------------

bool DebugEnvFlag(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  const std::string text(value);
  return text != "0" && text != "off" && text != "false";
}

size_t DebugEnvSize(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed > std::numeric_limits<size_t>::max()) return fallback;
  return static_cast<size_t>(parsed);
}

bool DebugEnabled() {
  static const bool enabled = DebugEnvFlag("UMBP_HBM_COPY_DEBUG", false);
  return enabled;
}

size_t DebugSampleEvery() {
  static const size_t n = std::max<size_t>(DebugEnvSize("UMBP_HBM_COPY_DEBUG_SAMPLE", 1), 1);
  return n;
}

size_t DebugMaxSegments() {
  static const size_t n = DebugEnvSize("UMBP_HBM_COPY_DEBUG_FRAGMENTS", 16);
  return n;
}

size_t DebugSummarySeconds() {
  static const size_t n = DebugEnvSize("UMBP_HBM_COPY_DEBUG_SUMMARY_SEC", 10);
  return n;
}

// One counter for the whole engine, so sampling thins the combined stream
// rather than letting a rarely-taken path go unseen behind a common one.
bool DebugSampleThisCall() {
  static std::atomic<uint64_t> counter{0};
  return (counter.fetch_add(1, std::memory_order_relaxed) % DebugSampleEvery()) == 0;
}

const char* KindName(hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToDevice:
      return "H2D";
    case hipMemcpyDeviceToHost:
      return "D2H";
    case hipMemcpyDeviceToDevice:
      return "D2D";
    case hipMemcpyHostToHost:
      return "H2H";
    default:
      return "?";
  }
}

inline double Seconds(std::chrono::steady_clock::time_point start,
                      std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

inline double GiBps(double bytes, double seconds) {
  if (seconds <= 0.0) return 0.0;
  return (bytes / seconds) / static_cast<double>(1ULL << 30);
}

// Cumulative per-path totals.  A serving run's per-call numbers are noisy and
// bimodal (a cold page fault dwarfs a warm copy), so the aggregate is the one
// worth quoting.
class DebugStats {
 public:
  void Record(bool gather, uint64_t segments, double bytes, double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    Totals& t = gather ? gather_ : memcpy_;
    t.calls += 1;
    t.segments += segments;
    t.bytes += bytes;
    t.seconds += seconds;
    ReportIfDueLocked();
  }

 private:
  struct Totals {
    uint64_t calls = 0;
    uint64_t segments = 0;
    double bytes = 0.0;
    double seconds = 0.0;
  };

  void ReportIfDueLocked() {
    const size_t cadence = DebugSummarySeconds();
    if (cadence == 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (Seconds(last_report_, now) < static_cast<double>(cadence)) return;
    last_report_ = now;
    MORI_UMBP_INFO(
        "[HbmCopyEngine][dbg] SUMMARY gather[calls={} segs={} bytes={:.3f}GiB busy={:.3f}s "
        "mean_seg={}B agg={:.3f}GiB/s] hipMemcpy[calls={} segs={} bytes={:.3f}GiB busy={:.3f}s "
        "mean_seg={}B agg={:.3f}GiB/s]",
        gather_.calls, gather_.segments, gather_.bytes / static_cast<double>(1ULL << 30),
        gather_.seconds, gather_.segments ? gather_.bytes / gather_.segments : 0.0,
        GiBps(gather_.bytes, gather_.seconds), memcpy_.calls, memcpy_.segments,
        memcpy_.bytes / static_cast<double>(1ULL << 30), memcpy_.seconds,
        memcpy_.segments ? memcpy_.bytes / memcpy_.segments : 0.0,
        GiBps(memcpy_.bytes, memcpy_.seconds));
  }

  std::mutex mutex_;
  Totals gather_;
  Totals memcpy_;
  std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
};

DebugStats& Stats() {
  // Never destroyed: worker threads can still be copying during static
  // teardown, same reasoning as the gather streams.
  static DebugStats* stats = new DebugStats();
  return *stats;
}

// Everything a unit test needs to rebuild this exact copy: the direction, the
// device, the two base pointers (for alignment), and the segment geometry.
void LogPlanGeometry(const char* path, size_t plan_index, const TransferPlan& plan, int device_id) {
  const size_t limit = DebugMaxSegments();
  if (limit == 0) return;
  const auto src_addr = reinterpret_cast<uintptr_t>(plan.src.host_ptr);
  const auto dst_addr = reinterpret_cast<uintptr_t>(plan.dst.host_ptr);
  MORI_UMBP_INFO(
      "[HbmCopyEngine][dbg]   {} plan={} kind={} dev={} src_base={:#x} dst_base={:#x} "
      "src_loc={} dst_loc={} src_dev={} dst_dev={} src_size={} dst_size={} segs={} tags={}",
      path, plan_index, KindName(KindFor(plan.src, plan.dst)), device_id, src_addr, dst_addr,
      static_cast<int>(plan.src.loc), static_cast<int>(plan.dst.loc), plan.src.device,
      plan.dst.device, plan.src.size, plan.dst.size, plan.sizes.size(), plan.tags.size());
  const size_t shown = std::min(limit, plan.sizes.size());
  for (size_t i = 0; i < shown; ++i) {
    // vec16 reports whether the kernel's uint4 fast path applies to this
    // segment; a single unaligned segment drops it to the byte loop.
    const uintptr_t mask = (src_addr + plan.src_offsets[i]) | (dst_addr + plan.dst_offsets[i]) |
                           static_cast<uintptr_t>(plan.sizes[i]);
    MORI_UMBP_INFO("[HbmCopyEngine][dbg]     seg[{}] src_off={} dst_off={} bytes={} vec16={}", i,
                   plan.src_offsets[i], plan.dst_offsets[i], plan.sizes[i],
                   (mask & (kVectorBytes - 1)) == 0 ? 1 : 0);
  }
  if (plan.sizes.size() > shown) {
    MORI_UMBP_INFO("[HbmCopyEngine][dbg]     ... {} more segs", plan.sizes.size() - shown);
  }
}

// Non-blocking gather streams, pooled per device and lent to one gather at a
// time, so concurrent copies still run on separate streams.  They are not tied
// to the calling thread: the standalone server calls Submit from gRPC's sync
// pool, which retires and spawns threads under load, and every HIP stream holds
// ~6.4 MiB of device memory for its kernarg pool.  Kept per thread, they leaked
// until the GPU was full and stream creation failed (which HIP 7.2 turns into a
// segfault).  A device's pool grows only to its peak number of concurrent
// gathers, or, once a cap is set, to the cap: past it a gather waits for a
// stream instead of creating one, because creating one allocates device
// memory, and with the GPU full that can stall for seconds.  Never destroyed:
// worker threads can still be copying during static teardown.
class GatherStreamPool {
 public:
  static GatherStreamPool& Instance() {
    static GatherStreamPool* pool = new GatherStreamPool();
    return *pool;
  }

  // Streams per device; 0, the default, means no cap.
  void SetCap(size_t cap) {
    std::lock_guard<std::mutex> lock(mutex_);
    cap_ = cap;
  }

  // The current device must already be `device_id`.  nullptr if no stream is
  // free and a new one cannot be created.
  hipStream_t Acquire(int device_id) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      Device& device = devices_[device_id];
      device.returned.wait(lock,
                           [&] { return !device.idle.empty() || cap_ == 0 || device.live < cap_; });
      if (!device.idle.empty()) {
        hipStream_t stream = device.idle.back();
        device.idle.pop_back();
        return stream;
      }
      ++device.live;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
      (void)hipGetLastError();
      Discard(device_id);
      return nullptr;
    }
    created_.fetch_add(1, std::memory_order_relaxed);
    return stream;
  }

  void Release(int device_id, hipStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    Device& device = devices_[device_id];
    device.idle.push_back(stream);
    device.returned.notify_one();
  }

  // For an acquired stream that will not be released.
  void Discard(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Device& device = devices_[device_id];
    --device.live;
    device.returned.notify_one();
  }

  uint64_t Created() const { return created_.load(std::memory_order_relaxed); }

 private:
  struct Device {
    std::vector<hipStream_t> idle;
    size_t live = 0;  // created and not discarded
    std::condition_variable returned;
  };
  std::mutex mutex_;
  std::map<int, Device> devices_;
  size_t cap_ = 0;
  std::atomic<uint64_t> created_{0};
};

// One copy each way on every copy engine between these two buffers, so that
// ROCr creates each engine's queue here instead of on the engine's first use.
void PrewarmCopyEngines(void* device_bytes, void* host_bytes, size_t bytes) {
  hsa_amd_pointer_info_t device_info{};
  hsa_amd_pointer_info_t host_info{};
  device_info.size = sizeof(device_info);
  host_info.size = sizeof(host_info);
  if (hsa_amd_pointer_info(device_bytes, &device_info, nullptr, nullptr, nullptr) !=
          HSA_STATUS_SUCCESS ||
      hsa_amd_pointer_info(host_bytes, &host_info, nullptr, nullptr, nullptr) !=
          HSA_STATUS_SUCCESS) {
    return;
  }
  hsa_signal_t signal;
  if (hsa_signal_create(1, 0, nullptr, &signal) != HSA_STATUS_SUCCESS) return;
  const hsa_agent_t gpu = device_info.agentOwner;
  const hsa_agent_t cpu = host_info.agentOwner;
  struct Direction {
    void* dst;
    hsa_agent_t dst_agent;
    const void* src;
    hsa_agent_t src_agent;
  };
  const Direction directions[] = {{device_bytes, gpu, host_bytes, cpu},
                                  {host_bytes, cpu, device_bytes, gpu}};
  for (const Direction& d : directions) {
    uint32_t engines = 0;
    if (hsa_amd_memory_copy_engine_status(d.dst_agent, d.src_agent, &engines) !=
        HSA_STATUS_SUCCESS) {
      continue;
    }
    for (uint32_t bit = 0; bit < 32; ++bit) {
      const uint32_t engine = 1u << bit;
      if ((engines & engine) == 0) continue;
      hsa_signal_store_screlease(signal, 1);
      if (hsa_amd_memory_async_copy_on_engine(
              d.dst, d.dst_agent, d.src, d.src_agent, bytes, 0, nullptr, signal,
              static_cast<hsa_amd_sdma_engine_id_t>(engine), true) == HSA_STATUS_SUCCESS) {
        hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                  HSA_WAIT_STATE_BLOCKED);
      }
    }
  }
  hsa_signal_destroy(signal);
}

// Latches on the first gather failure and never tries again.  A kernel that
// cannot launch here will not start working later, and the fallback is always
// available, so one failed launch should not cost every later batch a retry.
std::atomic<bool>& GatherDisabled() {
  static std::atomic<bool> disabled{false};
  return disabled;
}

// Which device a plan's copies should run on.  A D2D pair between two devices
// needs a current device with peer access to both; we pick the destination's,
// which is the one hipMemcpy attributes the copy to.
inline int DeviceFor(const TransferPlan& plan) {
  if (IsGpu(plan.dst) && plan.dst.device >= 0) return plan.dst.device;
  if (IsGpu(plan.src) && plan.src.device >= 0) return plan.src.device;
  return -1;
}

// Every plan this engine produces is executed before Submit returns, so the
// handle only has to replay the outcome.  (Same shape as LocalCopyEngine's;
// kept file-local in both because it is an implementation detail of "this
// engine completes inline", not a shared contract.)
class SettledHandle final : public TransferHandle {
 public:
  explicit SettledHandle(std::vector<TransferFailure> failures) : failures_(std::move(failures)) {}

  void Wait(std::vector<TransferFailure>* failures) override {
    if (reported_) return;
    reported_ = true;
    if (failures != nullptr) {
      for (auto& f : failures_) failures->push_back(std::move(f));
    }
    failures_.clear();
  }

 private:
  std::vector<TransferFailure> failures_;
  bool reported_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
//  TransferEngine
// ---------------------------------------------------------------------------

bool HbmCopyEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  if (!src.HasHostPtr() || !dst.HasHostPtr()) return false;
  // At least one GPU endpoint.  The both-CPU case is LocalCopyEngine's and is
  // left to it: hipMemcpy would be correct but strictly slower than the
  // NT-AVX2 path, and overlapping the two engines would make composite
  // selection order load-bearing for performance.
  return IsGpu(src) || IsGpu(dst);
}

namespace {
// Merges adjacent ranges, but a segment never crosses a host registration piece.
void AppendWithinHostPieces(TransferPlan* plan, const TransferItem& item, size_t piece) {
  const bool host_is_src = !IsGpu(item.src);
  const bool host_is_dst = !host_is_src && !IsGpu(item.dst);
  const uintptr_t host_base = host_is_src   ? reinterpret_cast<uintptr_t>(item.src.host_ptr)
                              : host_is_dst ? reinterpret_cast<uintptr_t>(item.dst.host_ptr)
                                            : 0;
  const auto host_piece = [&](size_t src_offset, size_t dst_offset) {
    return (host_base + (host_is_src ? src_offset : dst_offset)) / piece;
  };
  size_t so = static_cast<size_t>(item.src_offset);
  size_t dof = static_cast<size_t>(item.dst_offset);
  size_t remaining = static_cast<size_t>(item.size);
  while (remaining > 0) {
    size_t sz = remaining;
    if (host_base != 0) {
      const uintptr_t host = host_base + (host_is_src ? so : dof);
      sz = std::min<size_t>(sz, piece - host % piece);
    }
    const bool same_piece =
        host_base == 0 || plan->sizes.empty() ||
        host_piece(plan->src_offsets.back(), plan->dst_offsets.back()) == host_piece(so, dof);
    AppendSegment(plan, so, dof, sz, item.tag, same_piece);
    so += sz;
    dof += sz;
    remaining -= sz;
  }
}
}  // namespace

std::unique_ptr<TransferPlanBuilder> HbmCopyEngine::CreatePlanBuilder(
    size_t /*expected_items*/) const {
  const size_t piece = HostRegistrationChunkBytes();
  return MakePairPlanBuilder(*this, [piece](TransferPlan* plan, const TransferItem& item) {
    AppendWithinHostPieces(plan, item, piece);
  });
}

void HbmCopyEngine::AddHostGatherRegion(void* base, size_t bytes) {
  if (base == nullptr || bytes == 0) return;
  std::lock_guard<std::mutex> lock(host_regions_mutex_);
  host_regions_.push_back(std::make_unique<HostTierRegistration>(base, bytes));
}

void HbmCopyEngine::ClearHostGatherRegions() {
  std::lock_guard<std::mutex> lock(host_regions_mutex_);
  host_regions_.clear();
}

void* HbmCopyEngine::HostRegionDeviceAddress(const void* ptr, size_t size, int device_id) const {
  std::lock_guard<std::mutex> lock(host_regions_mutex_);
  for (const auto& region : host_regions_) {
    if (region == nullptr) continue;
    if (void* alias = region->DeviceAddress(ptr, size, device_id)) return alias;
  }
  return nullptr;
}

uint64_t HbmCopyEngine::GatherStreamsCreated() { return GatherStreamPool::Instance().Created(); }

void HbmCopyEngine::Prewarm(size_t gather_streams_per_device) {
  if (gather_streams_per_device == 0) return;
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess) {
    (void)hipGetLastError();
    return;
  }
  // The copy engines are warmed either way: with the kernel off, every copy
  // takes the hipMemcpy fallback.
  const bool gather = !GatherDisabled().load(std::memory_order_relaxed) && DeviceGatherEnabled();
  int previous_device = 0;
  (void)hipGetDevice(&previous_device);
  if (gather) GatherStreamPool::Instance().SetCap(gather_streams_per_device);
  size_t warmed = 0;
  for (int device_id = 0; device_id < devices; ++device_id) {
    if (hipSetDevice(device_id) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    std::vector<hipStream_t> streams;
    for (size_t i = 0; gather && i < gather_streams_per_device; ++i) {
      hipStream_t stream = GatherStreamPool::Instance().Acquire(device_id);
      if (stream == nullptr) break;
      streams.push_back(stream);
      (void)PrewarmDeviceGather(device_id, stream, kPrewarmGatherFragments);
    }
    void* host_bytes = nullptr;
    void* device_bytes = nullptr;
    if (hipHostMalloc(&host_bytes, kPrewarmCopyBytes, hipHostMallocDefault) == hipSuccess &&
        hipMalloc(&device_bytes, kPrewarmCopyBytes) == hipSuccess) {
      PrewarmCopyEngines(device_bytes, host_bytes, kPrewarmCopyBytes);
      // The hipMemcpy fallback runs on the null stream.
      (void)hipMemcpy(device_bytes, host_bytes, kPrewarmCopyBytes, hipMemcpyHostToDevice);
      (void)hipMemcpy(host_bytes, device_bytes, kPrewarmCopyBytes, hipMemcpyDeviceToHost);
    }
    for (hipStream_t stream : streams) {
      if (hipStreamSynchronize(stream) == hipSuccess) {
        GatherStreamPool::Instance().Release(device_id, stream);
        ++warmed;
      } else {
        (void)hipGetLastError();
        GatherStreamPool::Instance().Discard(device_id);
      }
    }
    if (device_bytes != nullptr) (void)hipFree(device_bytes);
    if (host_bytes != nullptr) (void)hipHostFree(host_bytes);
    (void)hipGetLastError();
  }
  (void)hipSetDevice(previous_device);
  MORI_UMBP_INFO("[HbmCopyEngine] prewarmed {} gather streams and the copy engines of {} devices",
                 warmed, devices);
}

// Run the gather kernel over every eligible segment in the batch, and report
// which plans it completed.
//
// Bucketing is per DEVICE, not per plan, and that is the whole point.  A plan
// is one (src base, dst base) pair, so a caller reading three ranges into three
// separate GPU allocations produces three single-segment plans — nothing to
// gather within any of them, while together they are exactly the scattered
// small-fragment batch the kernel wins on.  Direction does not partition the
// buckets either: once the host side is registered both endpoints are
// dereferenceable from the device, so one kernel can carry H2D and D2H
// fragments at once.
//
// Never reports a plan it did not fully complete.  Anything declined or failed
// is left for the hipMemcpy loop, which reaches the same result whether or not
// some fragments already landed.
const char* HbmCopyEngine::GatherSkipName(GatherSkip reason) {
  switch (reason) {
    case GatherSkip::kTaken:
      return "taken";
    case GatherSkip::kUnrecorded:
      return "unrecorded";
    case GatherSkip::kDisabled:
      return "gather-disabled";
    case GatherSkip::kKindNotHostDevice:
      return "kind-not-h2d-or-d2h";
    case GatherSkip::kNoDevice:
      return "no-device";
    case GatherSkip::kHostNotGatherable:
      return "host-not-gatherable";
    case GatherSkip::kNoFragments:
      return "no-fragments";
    case GatherSkip::kTooFewFragments:
      return "too-few-fragments";
    case GatherSkip::kFragmentAtOrAboveThreshold:
      return "mean-seg-at-or-above-threshold";
    case GatherSkip::kSetDeviceFailed:
      return "hipSetDevice-failed";
    case GatherSkip::kNoStream:
      return "no-stream";
    case GatherSkip::kLaunchFailed:
      return "launch-failed";
    case GatherSkip::kSyncFailed:
      return "sync-failed";
  }
  return "?";
}

std::vector<char> HbmCopyEngine::GatherEligiblePlans(const std::vector<TransferPlan>& plans,
                                                     std::vector<GatherSkip>* skip_reasons) {
  std::vector<char> done(plans.size(), 0);
  if (skip_reasons != nullptr) skip_reasons->assign(plans.size(), GatherSkip::kUnrecorded);
  // Submit passes nullptr unless debug mode is on, so the hot path pays one
  // predicted null check per verdict and stores nothing.
  auto note = [skip_reasons](size_t plan_index, GatherSkip reason) {
    if (skip_reasons != nullptr) (*skip_reasons)[plan_index] = reason;
  };
  if (GatherDisabled().load(std::memory_order_relaxed) || !DeviceGatherEnabled()) {
    for (size_t p = 0; p < plans.size(); ++p) note(p, GatherSkip::kDisabled);
    return done;
  }

  struct Bucket {
    std::vector<DeviceGatherFragment> fragments;
    std::vector<size_t> plan_indices;
    size_t total_bytes = 0;
    // Debug-only shape.  A bucket may legitimately mix directions: once the
    // host side is registered both endpoints are device-dereferenceable, so
    // one kernel can carry H2D and D2H fragments at once.
    size_t min_fragment = std::numeric_limits<size_t>::max();
    size_t max_fragment = 0;
    size_t h2d_fragments = 0;
    size_t d2h_fragments = 0;
  };
  std::map<int, Bucket> buckets;

  for (size_t p = 0; p < plans.size(); ++p) {
    const auto& plan = plans[p];
    const hipMemcpyKind kind = KindFor(plan.src, plan.dst);
    // D2D has no host side to register, and H2H never reaches this engine.
    if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost) {
      note(p, GatherSkip::kKindNotHostDevice);
      continue;
    }
    const int device_id = DeviceFor(plan);
    if (device_id < 0) {
      note(p, GatherSkip::kNoDevice);
      continue;
    }

    const char* src = static_cast<const char*>(plan.src.host_ptr);
    char* dst = static_cast<char*>(plan.dst.host_ptr);
    const bool host_is_src = kind == hipMemcpyHostToDevice;

    std::vector<DeviceGatherFragment> fragments;
    fragments.reserve(plan.sizes.size());
    size_t plan_bytes = 0;
    bool eligible = true;
    for (size_t i = 0; i < plan.sizes.size() && eligible; ++i) {
      const void* fragment_src = src + plan.src_offsets[i];
      void* fragment_dst = dst + plan.dst_offsets[i];
      const void* host_side = host_is_src ? fragment_src : fragment_dst;
      // A kernel cannot dereference plain mmap memory — it faults the GPU — so
      // an uncovered host side disqualifies the whole plan. Nor can it use the
      // host address: only the host side is replaced by its per-device alias.
      void* const host_alias = HostRegionDeviceAddress(host_side, plan.sizes[i], device_id);
      if (host_alias == nullptr) {
        eligible = false;
        break;
      }
      if (host_is_src) {
        fragments.push_back({host_alias, fragment_dst, plan.sizes[i]});
      } else {
        fragments.push_back({fragment_src, host_alias, plan.sizes[i]});
      }
      plan_bytes += plan.sizes[i];
    }
    if (!eligible) {
      note(p, GatherSkip::kHostNotGatherable);
      continue;
    }
    if (fragments.empty()) {
      note(p, GatherSkip::kNoFragments);
      continue;
    }

    auto& bucket = buckets[device_id];
    for (const auto& fragment : fragments) {
      bucket.min_fragment = std::min(bucket.min_fragment, fragment.bytes);
      bucket.max_fragment = std::max(bucket.max_fragment, fragment.bytes);
    }
    (host_is_src ? bucket.h2d_fragments : bucket.d2h_fragments) += fragments.size();
    bucket.fragments.insert(bucket.fragments.end(), fragments.begin(), fragments.end());
    bucket.plan_indices.push_back(p);
    bucket.total_bytes += plan_bytes;
  }

  const bool debug = DebugEnabled();
  for (auto& [device_id, bucket] : buckets) {
    // Bound to a plain reference first: capturing the structured binding
    // `bucket` directly is a C++20 extension and warns under this build.
    auto& bucket_plans = bucket.plan_indices;
    auto note_bucket = [&bucket_plans, &note](GatherSkip reason) {
      for (size_t p : bucket_plans) note(p, reason);
    };
    // One fragment is a plain hipMemcpy's best case; above the threshold the
    // copy engine wins because a single large async copy amortizes its own
    // submission.
    if (bucket.fragments.size() < 2) {
      note_bucket(GatherSkip::kTooFewFragments);
      continue;
    }
    if (bucket.total_bytes / bucket.fragments.size() >= kGatherFragmentThreshold) {
      note_bucket(GatherSkip::kFragmentAtOrAboveThreshold);
      continue;
    }
    if (hipSetDevice(device_id) != hipSuccess) {
      (void)hipGetLastError();
      note_bucket(GatherSkip::kSetDeviceFailed);
      continue;
    }
    hipStream_t stream = GatherStreamPool::Instance().Acquire(device_id);
    if (stream == nullptr) {
      note_bucket(GatherSkip::kNoStream);
      continue;
    }
    // Timed launch-to-completion.  The synchronize below is NOT added by debug
    // mode -- it is what makes the gather path synchronous in the first place
    // -- so the interval is the kernel's real cost, and enabling debug does not
    // perturb what is being measured.
    const auto started = std::chrono::steady_clock::now();
    if (!LaunchDeviceGather(bucket.fragments.data(), bucket.fragments.size(), device_id, stream)) {
      GatherStreamPool::Instance().Release(device_id, stream);
      note_bucket(GatherSkip::kLaunchFailed);
      continue;
    }
    const hipError_t sync = hipStreamSynchronize(stream);
    if (sync != hipSuccess) {
      // Latch off: a kernel that cannot complete here will not start working
      // later, and every caller still has the fallback.  The stream is dropped
      // rather than returned to the pool.
      (void)hipGetLastError();
      GatherStreamPool::Instance().Discard(device_id);
      GatherDisabled().store(true, std::memory_order_relaxed);
      MORI_UMBP_ERROR("[HbmCopyEngine] gather sync dev={} failed: {}", device_id,
                      hipGetErrorString(sync));
      note_bucket(GatherSkip::kSyncFailed);
      continue;
    }
    GatherStreamPool::Instance().Release(device_id, stream);
    for (size_t p : bucket.plan_indices) done[p] = 1;
    if (!debug) continue;

    const double seconds = Seconds(started, std::chrono::steady_clock::now());
    const double bytes = static_cast<double>(bucket.total_bytes);
    Stats().Record(/*gather=*/true, bucket.fragments.size(), bytes, seconds);
    note_bucket(GatherSkip::kTaken);
    if (!DebugSampleThisCall()) continue;
    MORI_UMBP_INFO(
        "[HbmCopyEngine][dbg] path=gather dev={} plans={} segs={} h2d={} d2h={} bytes={} "
        "seg_min={} seg_mean={} seg_max={} seconds={:.6f} GiBps={:.3f}",
        device_id, bucket.plan_indices.size(), bucket.fragments.size(), bucket.h2d_fragments,
        bucket.d2h_fragments, bucket.total_bytes, bucket.min_fragment,
        bucket.total_bytes / bucket.fragments.size(), bucket.max_fragment, seconds,
        GiBps(bytes, seconds));
    for (size_t p : bucket.plan_indices) LogPlanGeometry("gather", p, plans[p], device_id);
  }
  return done;
}

std::unique_ptr<TransferHandle> HbmCopyEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;

  std::vector<TransferFailure> failures;

  // Restore the caller's current device on the way out.  Submit runs on
  // PoolClient's executor threads, which are not ours to leave re-pointed at
  // another device.
  int entry_device = -1;
  bool device_touched = false;
  int current_device = -1;

  // The gather pass sets the device itself, so the save has to happen before
  // it rather than at the first plan that needs one.
  if (hipGetDevice(&entry_device) != hipSuccess) {
    entry_device = -1;
  } else {
    device_touched = true;
  }
  const bool debug = DebugEnabled();
  std::vector<GatherSkip> skips;
  const std::vector<char> gathered = GatherEligiblePlans(plans, debug ? &skips : nullptr);
  // The pass may have left the current device anywhere.
  current_device = -1;

  for (size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
    const auto& plan = plans[plan_index];
    if (gathered[plan_index] != 0) continue;
    const int want_device = DeviceFor(plan);
    if (want_device >= 0 && want_device != current_device) {
      if (!device_touched) {
        if (hipGetDevice(&entry_device) != hipSuccess) entry_device = -1;
        device_touched = true;
      }
      const hipError_t derr = hipSetDevice(want_device);
      if (derr != hipSuccess) {
        failures.push_back(TransferFailure{plan.tags, static_cast<uint32_t>(derr),
                                           std::string("hipSetDevice(") +
                                               std::to_string(want_device) +
                                               "): " + hipGetErrorString(derr),
                                           std::string("hbm:dev") + std::to_string(want_device)});
        continue;
      }
      current_device = want_device;
    }

    const hipMemcpyKind kind = KindFor(plan.src, plan.dst);
    char* dst = static_cast<char*>(plan.dst.host_ptr);
    const char* src = static_cast<const char*>(plan.src.host_ptr);

    // One failure per PLAN, not per segment: the plan is the unit a tag set
    // maps back to, and a partially-copied key is failed wholesale anyway.
    //
    // hipMemcpy is BLOCKING, so this loop is already launch-to-completion and
    // the debug clock below needs no synchronize of its own.
    size_t plan_bytes = 0;
    const auto started = std::chrono::steady_clock::now();
    hipError_t err = hipSuccess;
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      err = hipMemcpy(dst + plan.dst_offsets[i], src + plan.src_offsets[i], plan.sizes[i], kind);
      if (err != hipSuccess) break;
      plan_bytes += plan.sizes[i];
    }
    if (debug && err == hipSuccess) {
      const double seconds = Seconds(started, std::chrono::steady_clock::now());
      Stats().Record(/*gather=*/false, plan.sizes.size(), static_cast<double>(plan_bytes), seconds);
      if (DebugSampleThisCall()) {
        const size_t mean = plan.sizes.empty() ? 0 : plan_bytes / plan.sizes.size();
        MORI_UMBP_INFO(
            "[HbmCopyEngine][dbg] path=hipMemcpy dev={} kind={} segs={} bytes={} seg_mean={} "
            "seconds={:.6f} GiBps={:.3f} why_not_gather={}",
            want_device, KindName(kind), plan.sizes.size(), plan_bytes, mean, seconds,
            GiBps(static_cast<double>(plan_bytes), seconds),
            GatherSkipName(plan_index < skips.size() ? skips[plan_index]
                                                     : GatherSkip::kUnrecorded));
        LogPlanGeometry("hipMemcpy", plan_index, plan, want_device);
      }
    }
    if (err != hipSuccess) {
      MORI_UMBP_ERROR("[HbmCopyEngine] hipMemcpy kind={} dev={} failed: {}", static_cast<int>(kind),
                      want_device, hipGetErrorString(err));
      failures.push_back(TransferFailure{plan.tags, static_cast<uint32_t>(err),
                                         std::string("hipMemcpy: ") + hipGetErrorString(err),
                                         std::string("hbm:dev") + std::to_string(want_device)});
    }
  }

  if (device_touched && entry_device >= 0) {
    // Best effort: a failure here cannot be attributed to any one plan, and the
    // transfers themselves already settled.
    (void)hipSetDevice(entry_device);
  }

  return std::make_unique<SettledHandle>(std::move(failures));
}

}  // namespace mori::umbp
