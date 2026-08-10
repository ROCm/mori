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

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"

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

std::unique_ptr<TransferHandle> HbmCopyEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;

  std::vector<TransferFailure> failures;

  // Restore the caller's current device on the way out.  Submit runs on
  // PoolClient's executor threads, which are not ours to leave re-pointed at
  // another device.
  int entry_device = -1;
  bool device_touched = false;
  int current_device = -1;

  for (const auto& plan : plans) {
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
