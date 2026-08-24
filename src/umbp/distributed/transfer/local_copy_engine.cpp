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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Tuning knobs for the copy itself.  Kept internal: they describe how this
// engine schedules work, not anything a caller can act on.

// Smallest block the non-temporal path is used for.
//
// This was 256 KiB, a figure that came from whole-object copies where one key
// is one ~MiB page.  A RANGED copy moves one LAYER of one page, and layers are
// small: DeepSeek-V4-Pro's three pools are 37,440 B, 8,448 B and 1,728 B.  So
// every ranged copy fell under the old threshold and took the ordinary memcpy,
// paying read-for-ownership on a destination it completely overwrites.
//
// The threshold has to be set against a real model's fragment sizes, not a
// synthetic one: a uniform 73,728 B shape is larger than anything the model
// actually has, so a threshold that looks right there never fires in practice.
//
// 2 KiB, measured on DeepSeek-V4-Pro's geometry (8 ranks, 32 pages, served
// entirely from the local medium, medians of 3): restore 12.04 -> 11.43 ms and
// TTFL 1,529 -> 1,353 us.  Below 2 KiB the remaining movement is inside this
// machine's run-to-run spread.  Tunable because the right answer is a property
// of the machine's cache, not of the code.
size_t NtMinBytes() {
  static const size_t kDefault = size_t{2} << 10;
  static const size_t n = [] {
    const char* raw = std::getenv("UMBP_DRAM_NT_COPY_MIN_BYTES");
    if (raw == nullptr) return kDefault;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(raw, &end, 10);
    if (end == raw || v == 0) return kDefault;
    return static_cast<size_t>(v);
  }();
  return n;
}

// Backing store for MarkThreadBackgroundCopies(); see the header for why a
// thread would declare itself background.
bool& ThreadBackgroundFlag() {
  static thread_local bool background = false;
  return background;
}

// A persistent set of helper threads that Submit can lend its copies to.
//
// Persistent, not spawned per call, because spawning was measured to cost more
// than the parallelism gained on any workload with a remote half.  Creating and
// joining threads maps and unmaps their stacks, which serializes on the
// process's address space -- so the damage is not proportional to how many
// helpers are used, and indeed capping the whole process to FOUR concurrent
// spawned helpers still cost 25% of TTFL on a remote layer-wise restore.  The
// pool makes fan-out cost a condition-variable wake instead.
class CopyPool {
 public:
  struct Batch {
    const std::function<void(size_t)>* fn = nullptr;
    size_t count = 0;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
  };

  static CopyPool& Instance() {
    // Never destroyed: the helpers outlive any translation unit's static
    // destructor order, and tearing them down during exit buys nothing.
    static CopyPool* pool = new CopyPool();
    return *pool;
  }

  // Publishes `batch` to the helpers, works on it from the calling thread, and
  // returns once every item has been run.  Helpers are best-effort: if they are
  // all busy the caller simply does the whole batch itself.
  void Run(const std::shared_ptr<Batch>& batch, int helpers) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      StartThreadsLocked(helpers);
      queue_.push_back(batch);
    }
    cv_.notify_all();
    Drain(*batch);
    // Drain() returning only means this thread found nothing left to claim; a
    // helper may still be inside the last item.  The batch is shared_ptr-owned
    // precisely so that is safe to wait for.
    while (batch->done.load(std::memory_order_acquire) < batch->count) {
      std::this_thread::yield();
    }
  }

 private:
  static void Drain(Batch& batch) {
    for (size_t i = batch.next.fetch_add(1, std::memory_order_relaxed); i < batch.count;
         i = batch.next.fetch_add(1, std::memory_order_relaxed)) {
      (*batch.fn)(i);
      batch.done.fetch_add(1, std::memory_order_release);
    }
  }

  void StartThreadsLocked(int wanted) {
    while (static_cast<int>(threads_.size()) < wanted) {
      threads_.emplace_back([this] { WorkerLoop(); });
    }
  }

  void WorkerLoop() {
    // A helper exists to serve other threads' copies, so its own copies -- of
    // which it has none -- must never recurse into fanning out.
    MarkThreadBackgroundCopies();
    for (;;) {
      std::shared_ptr<Batch> batch;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        batch = queue_.front();
        // Popped on claim rather than on completion: the caller is draining it
        // too, and a batch that is already exhausted must not keep waking
        // helpers.
        queue_.pop_front();
      }
      Drain(*batch);
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Batch>> queue_;
  std::vector<std::thread> threads_;
};

// How many threads to spread one Submit's copies over, counting the caller.
//
// A single memcpy stream tops out well below what the memory system can do --
// measured at ~6 GB/s here, and the same ~6 GB/s whether 1 rank or 8 are
// copying concurrently, so the ceiling is the stream and not the DRAM.  Local
// mode already knew this: DRAMTier fans its batch reads out over
// UMBP_DRAM_READ_THREADS (default 4), and its comment says ">1 breaks the
// single-core memcpy ceiling".  The distributed path never did, because
// LocalCopyEngine was written when one key was one whole page and the batch
// executors above it supplied the parallelism.  Ranged I/O broke that
// assumption: a layer-wise reader hands one thread thousands of small copies.
//
// The default is 2 rather than local mode's 4.  On DeepSeek-V4-Pro's real pool
// geometry 2 is the best value measured everywhere, not a compromise between
// workloads (medians of 3, restore ms / TTFL us):
//
//                     threads=1        threads=2        threads=4
//   local  1 rank   10.78 / 1340     10.28 / 1093     10.30 / 1118
//   local  8 ranks  13.03 / 1810     11.24 / 1492     11.56 / 1669
//   remote 1 rank   15.29 / 2404     13.61 / 2486     13.64 / 2342
//   remote 8 ranks  81.92 / 10812    83.78 / 10787    84.16 / 11790
//
// Going past 2 costs twice: it starves a concurrent remote half (the NIC's DMA
// into the scratch arena loses to the copies, and the first layer group -- the
// one nothing overlaps -- waits), and on the real geometry it does not even pay
// for the local case, because fragments this small do not keep four threads
// usefully busy.  An earlier uniform-73,728 B shape did show 4 winning locally;
// that was the synthetic shape, not the model's.
//
// Below the byte floor the threads cost more than they save, so a small batch
// stays on the calling thread and pays nothing.
int CopyThreads(size_t total_bytes, size_t copy_count) {
  if (ThreadBackgroundFlag()) return 1;
  static const int kMax = [] {
    int n = 2;
    if (const char* raw = std::getenv("UMBP_DRAM_COPY_THREADS")) {
      const int v = std::atoi(raw);
      if (v >= 1) n = v;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc > 0 && n > static_cast<int>(hc)) n = static_cast<int>(hc);
    return n < 1 ? 1 : n;
  }();
  static const size_t kMinBytes = [] {
    if (const char* raw = std::getenv("UMBP_DRAM_COPY_THREADS_MIN_BYTES")) {
      char* end = nullptr;
      const unsigned long long v = std::strtoull(raw, &end, 10);
      if (end != raw) return static_cast<size_t>(v);
    }
    return size_t{1} << 20;
  }();
  if (kMax <= 1 || copy_count < 2 || total_bytes < kMinBytes) return 1;
  return std::min(kMax, static_cast<int>(copy_count));
}

}  // namespace

// Set on a worker thread at start-up; read by CopyThreads above.
void MarkThreadBackgroundCopies() { ThreadBackgroundFlag() = true; }

bool ThreadDoesBackgroundCopies() { return ThreadBackgroundFlag(); }

void HostCopyBlock(void* dst, const void* src, size_t size) {
  static const bool kNt = NtSupported() && !(std::getenv("UMBP_DRAM_NT_COPY") &&
                                             std::getenv("UMBP_DRAM_NT_COPY")[0] == '0');
  static const size_t kNtMinBytes = NtMinBytes();
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

  // Flatten first.  A ranged call arrives as many plans of one copy each, a
  // whole-object one as a single plan of many; spreading work over threads has
  // to see past that shape difference or it parallelizes only one of them.
  struct Copy {
    char* dst;
    const char* src;
    size_t size;
  };
  std::vector<Copy> copies;
  size_t total_copies = 0;
  for (const auto& plan : plans) total_copies += plan.sizes.size();
  copies.reserve(total_copies);
  size_t total_bytes = 0;
  for (const auto& plan : plans) {
    char* dst = static_cast<char*>(plan.dst.host_ptr);
    const char* src = static_cast<const char*>(plan.src.host_ptr);
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      copies.push_back({dst + plan.dst_offsets[i], src + plan.src_offsets[i], plan.sizes[i]});
      total_bytes += plan.sizes[i];
    }
  }

  const int threads = CopyThreads(total_bytes, copies.size());
  if (threads <= 1) {
    for (const Copy& c : copies) HostCopyBlock(c.dst, c.src, c.size);
  } else {
    // Items are claimed off one shared cursor rather than split up front: the
    // copies in a batch are not the same size (a range that straddles a page
    // boundary is emitted as fragments), so equal counts would not be equal
    // work.
    const std::function<void(size_t)> fn = [&copies](size_t i) {
      HostCopyBlock(copies[i].dst, copies[i].src, copies[i].size);
    };
    auto batch = std::make_shared<CopyPool::Batch>();
    batch->fn = &fn;
    batch->count = copies.size();
    CopyPool::Instance().Run(batch, threads - 1);
  }

  // No failure mode: bounds were validated at plan time and a memcpy cannot
  // fail afterwards.
  return std::make_unique<SettledHandle>(std::vector<TransferFailure>{});
}

}  // namespace mori::umbp
