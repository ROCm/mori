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
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_copy.h"

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

// ---------------------------------------------------------------------------
// Opt-in profiling (UMBP_DRAM_PROFILE=1). Answers the two questions that gate
// the next round of optimization work:
//   1. how much of a batch's wall time is spent waiting for mu_ (i.e. how much
//      the 8 ranks serialize against each other);
//   2. how many hipMemcpyAsync submissions would survive if adjacent objects
//      were merged (i.e. whether contiguous merging is worth implementing).
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
  std::atomic<uint64_t> runs_both{0};  // runs with both adjacent -- the real submission count
};

MergeStats& OffloadMergeStats() {
  static MergeStats stats{"offload D2H (src=GPU dst=DRAM)"};
  return stats;
}

MergeStats& LoadMergeStats() {
  static MergeStats stats{"load    H2D (src=DRAM dst=GPU)"};
  return stats;
}

// Counts how many contiguous runs the batch would collapse into. A run of k
// objects becomes one hipMemcpyAsync, so runs_both/objects is exactly the
// submission-count reduction that merging would buy. runs_gpu / runs_dram show
// which side is the obstacle: the DRAM side is ours to control (slot placement
// follows the order objects are sent), the GPU side is the tree allocator's.
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
  const uint64_t n = stats.batches.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % kProfileReportEvery != 0) return;

  const uint64_t objs = stats.objects.load(std::memory_order_relaxed);
  auto pct = [objs](const std::atomic<uint64_t>& runs) {
    return objs == 0 ? 100 : runs.load(std::memory_order_relaxed) * 100 / objs;
  };
  MORI_UMBP_INFO(
      "[DRAMTier/profile] mergeability {} batches={} objects={} | submissions kept: "
      "gpu-side-only={}% dram-side-only={}% both={}%",
      stats.name, n, objs, pct(stats.runs_gpu), pct(stats.runs_dram), pct(stats.runs_both));
}

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

// One reusable stream per device.
//
// Creating and destroying a stream around every batch costs ~6.5 ms per pair on
// MI355X/ROCm 7.2 (measured), which dwarfs the copies themselves: a layer-wise
// KV load issues 2 * num_layers batches per request, so the churn alone added
// ~1 s per rank and, because ReadBatchIntoPtr holds mu_ for the whole batch,
// ~8 s once 8 ranks serialize behind it. Reusing the stream removes that.
//
// The streams are intentionally never destroyed: tearing down HIP resources
// from a static destructor races with runtime teardown. They are a fixed,
// per-device cost that ends with the process.
hipStream_t DeviceStream(int device_id) {
  static std::mutex stream_mu;
  static std::map<int, hipStream_t> streams;
  std::lock_guard<std::mutex> lock(stream_mu);
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

void CopyDeviceJobs(const std::vector<CopyJob*>& jobs, int device_id, hipMemcpyKind kind) {
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
  // submission. Submission cost (~10us) dominates the DMA itself (~1.5us for a
  // 72 KiB KV object), so collapsing a run of k objects is close to a k-fold
  // win on the layer-wise load path. This pays off only because the connector
  // sends offloads in GPU-address order, which makes the tier's slot layout
  // mirror the GPU layout -- see UMBP_STANDALONE_PROCESS_DESIGN.md 11.2.
  std::vector<CopyJob*> ordered(jobs.begin(), jobs.end());
  std::sort(ordered.begin(), ordered.end(),
            [](const CopyJob* a, const CopyJob* b) { return a->src < b->src; });

  hipError_t status = hipSuccess;
  bool enqueued_all = true;
  for (size_t begin = 0; begin < ordered.size();) {
    size_t end = begin + 1;
    size_t bytes = ordered[begin]->size;
    while (end < ordered.size()) {
      const CopyJob* prev = ordered[end - 1];
      const CopyJob* cur = ordered[end];
      const bool src_adjacent =
          static_cast<const char*>(prev->src) + prev->size == static_cast<const char*>(cur->src);
      const bool dst_adjacent =
          static_cast<char*>(prev->dst) + prev->size == static_cast<char*>(cur->dst);
      if (!src_adjacent || !dst_adjacent) break;
      bytes += cur->size;
      ++end;
    }

    status = hipMemcpyAsync(ordered[begin]->dst, ordered[begin]->src, bytes, kind, stream);
    if (status != hipSuccess) {
      MORI_UMBP_ERROR("[DRAMTier] hipMemcpyAsync failed on device {} ({} bytes, {} objects): {}",
                      device_id, bytes, end - begin, hipGetErrorString(status));
      (void)hipGetLastError();
      enqueued_all = false;
      break;
    }
    begin = end;
  }

  const hipError_t sync_status = hipStreamSynchronize(stream);
  if (sync_status != hipSuccess) {
    MORI_UMBP_ERROR("[DRAMTier] hipStreamSynchronize failed on device {}: {}", device_id,
                    hipGetErrorString(sync_status));
    (void)hipGetLastError();
    enqueued_all = false;
  }

  if (enqueued_all) {
    for (CopyJob* job : jobs) job->copied = true;
  }
}

void CopyJobs(std::vector<CopyJob>* jobs, int host_threads, hipMemcpyKind device_copy_kind) {
  if (jobs->empty()) return;
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

  CopyHostJobs(host_jobs, host_threads);
  for (auto& [device_id, grouped_jobs] : device_jobs) {
    if (ProfileEnabled()) RecordMergeability(grouped_jobs, device_copy_kind);
    CopyDeviceJobs(grouped_jobs, device_id, device_copy_kind);
  }
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

  CopyJobs(&jobs, read_threads_, hipMemcpyHostToDevice);
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
  CopyJobs(&copy_jobs, write_threads_, hipMemcpyDeviceToHost);

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
