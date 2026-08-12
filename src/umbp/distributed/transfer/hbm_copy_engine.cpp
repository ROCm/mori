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

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"

namespace mori::umbp {

namespace {

// True iff [next, ...) is exactly adjacent after [base, base+len).  Same guard
// LocalCopyEngine uses; duplicated rather than shared because the two engines
// are otherwise independent and this is three lines.
inline bool AdjacentNoOverflow(size_t base, size_t len, size_t next) {
  return len <= std::numeric_limits<size_t>::max() - base && base + len == next;
}

struct PtrPairKey {
  const void* src;
  const void* dst;
  bool operator==(const PtrPairKey& o) const noexcept { return src == o.src && dst == o.dst; }
};
struct PtrPairKeyHash {
  size_t operator()(const PtrPairKey& k) const noexcept {
    size_t h = std::hash<const void*>{}(k.src);
    h ^= std::hash<const void*>{}(k.dst) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

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

// One non-blocking stream per device, per thread.  Submit runs on PoolClient's
// executor threads, so a shared stream would serialize copies that the
// hipMemcpy path ran independently.
std::map<int, hipStream_t>& GatherStreams() {
  static thread_local std::map<int, hipStream_t>* streams = new std::map<int, hipStream_t>();
  return *streams;
}

hipStream_t GatherStream(int device_id) {
  auto& streams = GatherStreams();
  auto it = streams.find(device_id);
  if (it != streams.end()) return it->second;
  hipStream_t stream = nullptr;
  if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  streams.emplace(device_id, stream);
  return stream;
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

TransferPlanSet HbmCopyEngine::Plan(const std::vector<TransferItem>& items) const {
  TransferPlanSet out;
  std::unordered_map<PtrPairKey, size_t, PtrPairKeyHash> pair_to_plan;
  pair_to_plan.reserve(items.size() * 2);

  for (const auto& item : items) {
    if (item.size == 0) continue;
    if (!CanHandle(item.src, item.dst)) {
      out.rejected_tags.push_back(item.tag);
      continue;
    }
    // Bounds are checked here, once, rather than per copy: a page index that
    // does not fit its buffer is a bug in the backend's slot bookkeeping, and
    // the key must be failed rather than copied past the end.  On a GPU
    // endpoint that matters more than on a host one — an overrun is not a
    // segfault but silent corruption of whatever else lives in the pool.
    if (item.size > item.src.size || item.src_offset > item.src.size - item.size ||
        item.size > item.dst.size || item.dst_offset > item.dst.size - item.size) {
      out.rejected_tags.push_back(item.tag);
      continue;
    }

    const PtrPairKey key{item.src.host_ptr, item.dst.host_ptr};
    auto it = pair_to_plan.find(key);
    size_t pi;
    if (it == pair_to_plan.end()) {
      pi = out.plans.size();
      pair_to_plan.emplace(key, pi);
      TransferPlan plan;
      plan.src = item.src;
      plan.dst = item.dst;
      plan.dir = TransferDirection::kLocal;
      out.plans.push_back(std::move(plan));
    } else {
      pi = it->second;
    }
    TransferPlan& plan = out.plans[pi];
    const size_t so = static_cast<size_t>(item.src_offset);
    const size_t dof = static_cast<size_t>(item.dst_offset);
    const size_t sz = static_cast<size_t>(item.size);
    const bool can_coalesce = !plan.sizes.empty() &&
                              AdjacentNoOverflow(plan.src_offsets.back(), plan.sizes.back(), so) &&
                              AdjacentNoOverflow(plan.dst_offsets.back(), plan.sizes.back(), dof);
    if (can_coalesce) {
      plan.sizes.back() += sz;
    } else {
      plan.src_offsets.push_back(so);
      plan.dst_offsets.push_back(dof);
      plan.sizes.push_back(sz);
    }
    if (plan.tags.empty() || plan.tags.back() != item.tag) plan.tags.push_back(item.tag);
  }
  return out;
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

bool HbmCopyEngine::HostRegionCovers(const void* ptr, size_t size) const {
  std::lock_guard<std::mutex> lock(host_regions_mutex_);
  for (const auto& region : host_regions_) {
    if (region && region->Covers(ptr, size)) return true;
  }
  return false;
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
std::vector<char> HbmCopyEngine::GatherEligiblePlans(const std::vector<TransferPlan>& plans) {
  std::vector<char> done(plans.size(), 0);
  if (GatherDisabled().load(std::memory_order_relaxed) || !DeviceGatherEnabled()) return done;

  struct Bucket {
    std::vector<DeviceGatherFragment> fragments;
    std::vector<size_t> plan_indices;
    size_t total_bytes = 0;
  };
  std::map<int, Bucket> buckets;

  for (size_t p = 0; p < plans.size(); ++p) {
    const auto& plan = plans[p];
    const hipMemcpyKind kind = KindFor(plan.src, plan.dst);
    // D2D has no host side to register, and H2H never reaches this engine.
    if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost) continue;
    const int device_id = DeviceFor(plan);
    if (device_id < 0) continue;

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
      // an uncovered host side disqualifies the whole plan.
      if (!HostRegionCovers(host_side, plan.sizes[i])) {
        eligible = false;
        break;
      }
      fragments.push_back({fragment_src, fragment_dst, plan.sizes[i]});
      plan_bytes += plan.sizes[i];
    }
    if (!eligible || fragments.empty()) continue;

    auto& bucket = buckets[device_id];
    bucket.fragments.insert(bucket.fragments.end(), fragments.begin(), fragments.end());
    bucket.plan_indices.push_back(p);
    bucket.total_bytes += plan_bytes;
  }

  for (auto& [device_id, bucket] : buckets) {
    // One fragment is a plain hipMemcpy's best case; above the threshold the
    // copy engine wins because a single large async copy amortizes its own
    // submission.
    if (bucket.fragments.size() < 2) continue;
    if (bucket.total_bytes / bucket.fragments.size() >= kGatherFragmentThreshold) continue;
    if (hipSetDevice(device_id) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    hipStream_t stream = GatherStream(device_id);
    if (stream == nullptr) continue;
    if (!LaunchDeviceGather(bucket.fragments.data(), bucket.fragments.size(), device_id, stream)) {
      continue;
    }
    const hipError_t sync = hipStreamSynchronize(stream);
    if (sync != hipSuccess) {
      // Latch off: a kernel that cannot complete here will not start working
      // later, and every caller still has the fallback.
      (void)hipGetLastError();
      GatherDisabled().store(true, std::memory_order_relaxed);
      GatherStreams().erase(device_id);
      MORI_UMBP_ERROR("[HbmCopyEngine] gather sync dev={} failed: {}", device_id,
                      hipGetErrorString(sync));
      continue;
    }
    for (size_t p : bucket.plan_indices) done[p] = 1;
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
  const std::vector<char> gathered = GatherEligiblePlans(plans);
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
    hipError_t err = hipSuccess;
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      err = hipMemcpy(dst + plan.dst_offsets[i], src + plan.src_offsets[i], plan.sizes[i], kind);
      if (err != hipSuccess) break;
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
