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
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

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
  static const bool kNt =
      Avx2Supported() &&
      !(std::getenv("UMBP_DRAM_NT_COPY") && std::getenv("UMBP_DRAM_NT_COPY")[0] == '0');
  static const size_t kNtMinBytes = 256ull << 10;
  if (kNt && size >= kNtMinBytes) {
    NtCopyAvx2(static_cast<char*>(dst), static_cast<const char*>(src), size);
  } else {
    std::memcpy(dst, src, size);
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
  std::lock_guard<std::mutex> lock(mu_);

  // If key already exists, free its old slot first
  auto existing = slots_.find(key);
  if (existing != slots_.end()) {
    Deallocate(existing->second.offset, existing->second.size);
    used_ -= existing->second.size;
    slots_.erase(existing);
    auto lru_it = lru_map_.find(key);
    if (lru_it != lru_map_.end()) {
      lru_list_.erase(lru_it->second);
      lru_map_.erase(lru_it);
    }
  }

  // Try to allocate — do NOT self-evict.
  // If no space, return false so upper layer can demote keys to SSD.
  size_t offset = Allocate(size);
  if (offset == static_cast<size_t>(-1)) {
    return false;
  }

  std::memcpy(static_cast<char*>(base_ptr_) + offset, data, size);
  slots_[key] = {offset, size};
  used_ += size;
  TouchLRU(key);
  return true;
}

bool DRAMTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  // Reject if caller's buffer size does not match the stored block size.
  // A mismatch indicates a caller bug (wrong page size); silently truncating
  // would produce a partially-filled KV block with no error signal.
  if (size != it->second.size) return false;

  std::memcpy(reinterpret_cast<void*>(dst_ptr), static_cast<char*>(base_ptr_) + it->second.offset,
              size);
  TouchLRU(key);
  return true;
}

std::vector<bool> DRAMTier::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (n == 0) return results;

  // Hold mu_ for the whole batch. This tier uses a single mutex (reads and
  // writes are mutually exclusive), so holding it keeps slot offsets valid
  // during the parallel copy without a use-after-free against Write/Evict/
  // Clear. The per-block copies still run in parallel within the batch, which
  // is what breaks the single-core memcpy ceiling on cold DRAM.
  std::lock_guard<std::mutex> lock(mu_);

  struct Job {
    void* dst;
    const void* src;
    size_t size;
    size_t idx;
  };
  std::vector<Job> jobs;
  jobs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto it = slots_.find(keys[i]);
    if (it == slots_.end()) continue;
    if (sizes[i] != it->second.size) continue;
    jobs.push_back({reinterpret_cast<void*>(dst_ptrs[i]),
                    static_cast<char*>(base_ptr_) + it->second.offset, sizes[i], i});
  }

  int num_threads = read_threads_;
  if (num_threads > static_cast<int>(jobs.size())) num_threads = static_cast<int>(jobs.size());

  if (num_threads <= 1) {
    for (const auto& j : jobs) {
      CopyBlock(j.dst, j.src, j.size);
      results[j.idx] = true;
    }
  } else {
    std::atomic<size_t> next{0};
    auto worker = [&]() {
      size_t i;
      while ((i = next.fetch_add(1)) < jobs.size()) {
        CopyBlock(jobs[i].dst, jobs[i].src, jobs[i].size);
      }
    };
    std::vector<std::thread> pool;
    pool.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    for (const auto& j : jobs) results[j.idx] = true;
  }

  for (const auto& j : jobs) TouchLRU(keys[j.idx]);
  return results;
}

std::vector<bool> DRAMTier::BatchWrite(const std::vector<std::string>& keys,
                                       const std::vector<const void*>& data_ptrs,
                                       const std::vector<size_t>& sizes) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (n == 0) return results;

  // Hold mu_ for the whole batch (single-mutex model: serializes against other
  // reads/writes). Slot allocation mutates free_list_/slots_ and must be serial;
  // only the per-block payload copies run in parallel, which is what breaks the
  // single-core memcpy ceiling on the backup path.
  std::lock_guard<std::mutex> lock(mu_);

  struct Job {
    void* dst;
    const void* src;
    size_t size;
    size_t idx;
    size_t offset;
  };
  std::vector<Job> jobs;
  jobs.reserve(n);

  // Phase 1 (serial): free any existing slot for the key, then allocate. Does
  // NOT self-evict — a key that doesn't fit is left false so the upper layer
  // (LocalStorageManager) can demote LRU keys and retry per-key.
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
    jobs.push_back({static_cast<char*>(base_ptr_) + offset, data_ptrs[i], sizes[i], i, offset});
  }

  // Phase 2 (parallel): non-temporal CopyBlock each payload into its slot.
  int num_threads = write_threads_;
  if (num_threads > static_cast<int>(jobs.size())) num_threads = static_cast<int>(jobs.size());

  if (num_threads <= 1) {
    for (const auto& j : jobs) CopyBlock(j.dst, j.src, j.size);
  } else {
    std::atomic<size_t> next{0};
    auto worker = [&]() {
      size_t i;
      while ((i = next.fetch_add(1)) < jobs.size()) {
        CopyBlock(jobs[i].dst, jobs[i].src, jobs[i].size);
      }
    };
    std::vector<std::thread> pool;
    pool.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }

  // Phase 3 (serial): register slots + LRU, mark successes.
  for (const auto& j : jobs) {
    slots_[keys[j.idx]] = {j.offset, j.size};
    used_ += j.size;
    TouchLRU(keys[j.idx]);
    results[j.idx] = true;
  }
  return results;
}

const void* DRAMTier::ReadPtr(const std::string& key, size_t* out_size) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return nullptr;

  if (out_size) *out_size = it->second.size;
  TouchLRU(key);
  return static_cast<char*>(base_ptr_) + it->second.offset;
}

std::vector<char> DRAMTier::Read(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return {};

  size_t sz = it->second.size;
  std::vector<char> buf(sz);
  std::memcpy(buf.data(), static_cast<char*>(base_ptr_) + it->second.offset, sz);
  TouchLRU(key);
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
  std::lock_guard<std::mutex> lock(mu_);
  return slots_.count(key) > 0;
}

bool DRAMTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  Deallocate(it->second.offset, it->second.size);
  used_ -= it->second.size;
  slots_.erase(it);

  auto lru_it = lru_map_.find(key);
  if (lru_it != lru_map_.end()) {
    lru_list_.erase(lru_it->second);
    lru_map_.erase(lru_it);
  }
  return true;
}

std::pair<size_t, size_t> DRAMTier::Capacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return {used_, capacity_};
}

void DRAMTier::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  slots_.clear();
  lru_list_.clear();
  lru_map_.clear();
  free_list_.clear();
  free_list_.push_back({0, capacity_});
  used_ = 0;
}

std::vector<std::string> DRAMTier::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::lock_guard<std::mutex> lock(mu_);
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
  std::lock_guard<std::mutex> lock(mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::optional<size_t> DRAMTier::GetSlotOffset(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
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
