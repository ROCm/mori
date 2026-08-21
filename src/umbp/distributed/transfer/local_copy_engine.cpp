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
#include "umbp/distributed/transfer/local_copy_engine.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  Block copy
//
//  In distributed mode 1 key == 1 page (master page_size == KV block size), so
//  a self-target copy moves one ~MiB-scale block per call.  Local mode's
//  DRAMTier optimization parallelizes WITHIN a call's pages (always 1 here) and
//  so never applied; the cross-key parallelism lives in PoolClient's batch
//  executors instead, and what is left to win per key is the cache bypass.
// ---------------------------------------------------------------------------

namespace {

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) inline void NtCopyAvx2(char* d, const char* s, size_t n) {
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
inline bool NtSupported() { return __builtin_cpu_supports("avx2"); }
#else
inline void NtCopyAvx2(char* d, const char* s, size_t n) { std::memcpy(d, s, n); }
inline bool NtSupported() { return false; }
#endif

// True iff [next, ...) is exactly adjacent after [base, base+len).
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

// Every plan this engine produces is executed before Submit returns, so the
// handle only has to replay the outcome.
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

void HostCopyBlock(void* dst, const void* src, size_t size) {
  static const bool kNt = NtSupported() && !(std::getenv("UMBP_DRAM_NT_COPY") &&
                                             std::getenv("UMBP_DRAM_NT_COPY")[0] == '0');
  static const size_t kNtMinBytes = 256ull << 10;
  if (kNt && size >= kNtMinBytes) {
    NtCopyAvx2(static_cast<char*>(dst), static_cast<const char*>(src), size);
  } else {
    std::memcpy(dst, src, size);
  }
}

// ---------------------------------------------------------------------------
//  TransferEngine
// ---------------------------------------------------------------------------

bool LocalCopyEngine::CanHandle(const TransferRef& src, const TransferRef& dst) const {
  return src.HasHostPtr() && dst.HasHostPtr() && src.loc == mori::io::MemoryLocationType::CPU &&
         dst.loc == mori::io::MemoryLocationType::CPU;
}

TransferPlanSet LocalCopyEngine::Plan(const std::vector<TransferItem>& items) const {
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
    // the key must be failed rather than copied past the end.
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

std::unique_ptr<TransferHandle> LocalCopyEngine::Submit(std::vector<TransferPlan> plans) {
  if (plans.empty()) return nullptr;
  for (const auto& plan : plans) {
    char* dst = static_cast<char*>(plan.dst.host_ptr);
    const char* src = static_cast<const char*>(plan.src.host_ptr);
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      HostCopyBlock(dst + plan.dst_offsets[i], src + plan.src_offsets[i], plan.sizes[i]);
    }
  }
  // No failure mode: bounds were validated at plan time and a memcpy cannot
  // fail afterwards.
  return std::make_unique<SettledHandle>(std::vector<TransferFailure>{});
}

}  // namespace mori::umbp
