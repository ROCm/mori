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
#include "umbp/local/host_mem_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/common/parallel_for.h"

namespace mori::umbp {
namespace {

constexpr size_t kDefaultPageSize = 4096;
constexpr size_t kThpSize = 2ULL * 1024 * 1024;
// Prefault and verify a THP buffer this much at a time, so a shortfall is
// found (and, with require_full, acted on) a chunk in rather than after the
// whole buffer.  On fragmented hosts THP faults direct-compact and run at
// ~0.1 GB/s (n06-21, 2026-11-09: 16 GiB in 137-168 s vs 0.8 s on base
// pages), so the chunk bounds how long require_full takes to give up; a
// MADV_COLLAPSE over an already-huge chunk costs little.
constexpr size_t kThpVerifyChunk = 64ULL << 20;

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25  // Linux 6.1+; older kernels return EINVAL.
#endif

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

struct ShmAllocRecord {
  int fd = -1;
  size_t size = 0;
  size_t active_registrations = 0;
  bool pending_free = false;
};

std::mutex& ShmRegistryMutex() {
  static std::mutex mu;
  return mu;
}

std::map<void*, ShmAllocRecord>& ShmRegistry() {
  static std::map<void*, ShmAllocRecord> registry;
  return registry;
}

#ifdef __linux__
int MemfdCreate(const char* name, unsigned int flags) {
#if defined(__NR_memfd_create)
  return static_cast<int>(syscall(__NR_memfd_create, name, flags));
#else
  (void)name;
  (void)flags;
  errno = ENOSYS;
  return -1;
#endif
}
#else
int MemfdCreate(const char* name, unsigned int flags) {
  (void)name;
  (void)flags;
  errno = ENOSYS;
  return -1;
}
#endif

size_t GetPageSize() {
  const long page = sysconf(_SC_PAGESIZE);
  return page > 0 ? static_cast<size_t>(page) : kDefaultPageSize;
}

bool IsPowerOfTwo(size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::optional<size_t> AlignUpChecked(size_t size, size_t alignment) {
  if (!IsPowerOfTwo(alignment)) return std::nullopt;
  if (size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return std::nullopt;
  }
  return (size + (alignment - 1)) & ~(alignment - 1);
}

std::string ReadHugepageMeminfoSummary() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) return "HugePages_Total=? HugePages_Free=? Hugepagesize=?";

  std::string line;
  std::string total = "HugePages_Total=?";
  std::string free = "HugePages_Free=?";
  std::string size = "Hugepagesize=?";
  while (std::getline(meminfo, line)) {
    if (line.rfind("HugePages_Total:", 0) == 0) {
      total = line;
    } else if (line.rfind("HugePages_Free:", 0) == 0) {
      free = line;
    } else if (line.rfind("Hugepagesize:", 0) == 0) {
      size = line;
    }
  }
  return total + " " + free + " " + size;
}

void LogHugepageFallbackOnce(size_t size, size_t hugepage_size, int err, bool to_thp = false) {
  static std::once_flag once;
  std::call_once(once, [size, hugepage_size, err, to_thp] {
    const std::string meminfo = ReadHugepageMeminfoSummary();
    MORI_UMBP_ERROR(
        "HostMemAllocator: MAP_HUGETLB allocation failed for size={} "
        "hugepage_size={} ({} pages; {}: {}); falling back to {}. "
        "If this buffer is registered with a GPU, kernel compaction can migrate those "
        "pages, and every migration stops this process's GPU queues until the driver "
        "re-validates the registration. Reserve enough hugepages (vm.nr_hugepages) "
        "before starting. {}",
        size, hugepage_size, hugepage_size ? (size + hugepage_size - 1) / hugepage_size : 0, err,
        std::strerror(err),
        to_thp ? "anonymous pages with transparent hugepages (UMBP_USE_THP; best effort)"
               : "anonymous 4 KiB pages",
        meminfo);
  });
}

// ---------------------------------------------------------------------------
// Transparent hugepages (kAnonymousThp)

// The bracketed word of a sysfs THP knob ("always [madvise] never" ->
// "madvise"); "" when the file is missing (kernel without THP).
std::string ReadSelectedSysfsMode(const char* path) {
  std::ifstream in(path);
  std::string line;
  if (!in || !std::getline(in, line)) return "";
  const size_t open = line.find('[');
  const size_t close = line.find(']', open);
  if (open == std::string::npos || close == std::string::npos) return "";
  return line.substr(open + 1, close - open - 1);
}

std::mutex& ThpOverrideMutex() {
  static std::mutex mu;
  return mu;
}

std::optional<std::string>& ThpEnabledOverride() {
  static std::optional<std::string> mode;
  return mode;
}

std::string ThpEnabledMode() {
  {
    std::lock_guard<std::mutex> lock(ThpOverrideMutex());
    if (ThpEnabledOverride().has_value()) return *ThpEnabledOverride();
  }
  static const std::string mode =
      ReadSelectedSysfsMode("/sys/kernel/mm/transparent_hugepage/enabled");
  return mode;
}

const std::string& ThpDefragMode() {
  static const std::string mode =
      ReadSelectedSysfsMode("/sys/kernel/mm/transparent_hugepage/defrag");
  return mode;
}

bool ThpRequested(const HostBufferOptions& opts) {
  switch (opts.thp) {
    case ThpMode::kOn:
      return true;
    case ThpMode::kOff:
      return false;
    case ThpMode::kFromEnv:
    default: {
      static const bool env = GetEnvUint32("UMBP_USE_THP", 0) != 0;
      return env;
    }
  }
}

bool ThpRequireFull(const HostBufferOptions& opts) {
  static const bool env = GetEnvUint32("UMBP_THP_REQUIRE_FULL", 0) != 0;
  return opts.thp_require_full || env;
}

void LogThpUnusableOnce(const std::string& mode) {
  static std::once_flag once;
  std::call_once(once, [mode] {
    MORI_UMBP_WARN(
        "HostMemAllocator: transparent hugepages requested but "
        "/sys/kernel/mm/transparent_hugepage/enabled is {}; using anonymous 4 KiB pages",
        mode.empty() ? "missing" : "[" + mode + "]");
  });
}

void LogThpModesOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    const std::string& defrag = ThpDefragMode();
    MORI_UMBP_INFO(
        "HostMemAllocator: transparent hugepages requested; THP enabled=[{}] defrag=[{}]{}",
        ThpEnabledMode(), defrag.empty() ? "?" : defrag,
        defrag == "madvise" || defrag == "defer+madvise" || defrag == "always"
            ? " (faults compact synchronously; prefault can be slow on fragmented memory)"
            : "");
  });
}

void LogThpUnmeasuredOnce(bool require_full) {
  static std::once_flag once;
  std::call_once(once, [require_full] {
    MORI_UMBP_WARN(
        "HostMemAllocator: THP buffer allocated without prefault, so its hugepage coverage is "
        "unknown; a NIC that needs one page size per memory region may reject it at "
        "registration{}",
        require_full ? ". UMBP_THP_REQUIRE_FULL has no effect without prefault" : "");
  });
}

std::string ReadThpMeminfoSummary() {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  std::string out;
  while (std::getline(meminfo, line)) {
    for (const char* key :
         {"MemFree:", "Cached:", "AnonHugePages:", "HugePages_Total:", "HugePages_Free:"}) {
      if (line.rfind(key, 0) == 0) {
        std::istringstream fields(line);
        std::string name, value, unit;
        fields >> name >> value >> unit;
        if (!out.empty()) out += ' ';
        out += name + value + (unit.empty() ? "" : " " + unit);
      }
    }
  }
  return out.empty() ? "meminfo unavailable" : out;
}

// Under the default task policy, interleave the buffer over every allowed
// node.  First touch would otherwise fill the local node first, after which
// each THP fault direct-compacts on that node and mostly fails (Mooncake
// #4393 measured 12-23+ minute starts).  An explicit task policy (numactl
// --membind, ...) is left alone.  Returns the node count interleaved over, 0
// when not applied.
int InterleaveIfDefaultPolicy(void* ptr, size_t mapped_size) {
#if defined(__linux__) && defined(__NR_get_mempolicy) && defined(__NR_mbind)
  int mode = -1;
  if (syscall(__NR_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0) return 0;
  if (mode != MPOL_DEFAULT) return 0;

  constexpr size_t kMaxNodes = 1024;
  constexpr size_t kBitsPerWord = sizeof(unsigned long) * 8;
  std::vector<unsigned long> mask(kMaxNodes / kBitsPerWord, 0);
  if (syscall(__NR_get_mempolicy, nullptr, mask.data(), static_cast<unsigned long>(kMaxNodes),
              nullptr, static_cast<unsigned long>(MPOL_F_MEMS_ALLOWED)) != 0) {
    return 0;
  }
  int nodes = 0;
  for (unsigned long word : mask) nodes += __builtin_popcountl(word);
  if (nodes <= 1) return 0;

  if (syscall(__NR_mbind, ptr, mapped_size, MPOL_INTERLEAVE, mask.data(),
              static_cast<unsigned long>(kMaxNodes), 0UL) != 0) {
    const int err = errno;
    MORI_UMBP_WARN("HostMemAllocator: mbind(MPOL_INTERLEAVE, {} nodes) failed ({}: {})", nodes, err,
                   std::strerror(err));
    return 0;
  }
  return nodes;
#else
  (void)ptr;
  (void)mapped_size;
  return 0;
#endif
}

std::string FormatCoverage(double coverage) {
  if (coverage < 0) return "unknown";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f%%", coverage * 100.0);
  return buf;
}

std::optional<ThpCoverage> MeasureThpCoverage(void* ptr, size_t mapped_size) {
  std::ifstream smaps("/proc/self/smaps");
  if (!smaps) return std::nullopt;
  return ThpCoverageFromSmaps(smaps, reinterpret_cast<uintptr_t>(ptr), mapped_size);
}

void LogNumaUnavailableOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    MORI_UMBP_WARN("HostMemAllocator: NUMA binding unavailable; continuing without binding");
  });
}

void TouchPages(void* ptr, size_t mapped_size, size_t stride) {
  volatile char* bytes = static_cast<volatile char*>(ptr);
  for (size_t offset = 0; offset < mapped_size; offset += stride) {
    bytes[offset] = bytes[offset];
  }
}

void PrefaultRange(void* ptr, size_t mapped_size, size_t stride) {
  if (ptr == nullptr || mapped_size == 0) return;

#ifdef MADV_POPULATE_WRITE
  if (madvise(ptr, mapped_size, MADV_POPULATE_WRITE) == 0) return;
  const int err = errno;
  MORI_UMBP_WARN(
      "HostMemAllocator: madvise(MADV_POPULATE_WRITE) failed ({}: {}); falling back "
      "to manual page touching",
      err, std::strerror(err));
#endif

  TouchPages(ptr, mapped_size, stride);
}

// ParallelFor also uses the calling thread. Restore its affinity after each
// chunk; a pool allocation must never leave a gRPC/startup thread pinned.
class ScopedNumaAffinity {
 public:
  explicit ScopedNumaAffinity(int node) {
#ifdef __linux__
    if (node < 0 || sched_getaffinity(0, sizeof(saved_), &saved_) != 0) return;
    std::ifstream input("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string token;
    cpu_set_t local;
    CPU_ZERO(&local);
    while (std::getline(input, token, ',')) {
      std::istringstream part(token);
      int first = -1, last = -1;
      if (!(part >> first)) continue;
      last = first;
      if (part.peek() == '-') {
        part.get();
        part >> last;
      }
      for (int cpu = std::max(0, first); cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &saved_)) CPU_SET(cpu, &local);
      }
    }
    if (CPU_COUNT(&local) > 0) changed_ = sched_setaffinity(0, sizeof(local), &local) == 0;
#endif
  }
  ~ScopedNumaAffinity() {
#ifdef __linux__
    if (changed_) (void)sched_setaffinity(0, sizeof(saved_), &saved_);
#endif
  }

 private:
#ifdef __linux__
  cpu_set_t saved_{};
  bool changed_ = false;
#endif
};

// Nodes that have CPUs to run a prefault worker on (memory-only nodes have an
// empty cpulist).
const std::vector<int>& NodesWithCpus() {
  static const std::vector<int> nodes = [] {
    std::vector<int> found;
    for (int node = 0; node < 256; ++node) {
      std::ifstream input("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
      std::string cpus;
      if (input && std::getline(input, cpus) && !cpus.empty()) found.push_back(node);
    }
    return found;
  }();
  return nodes;
}

// Splits [ptr, ptr + mapped_size) into per-worker slices on `stride`
// boundaries and runs `fn(slice_ptr, slice_len)` for each, on the slice's
// NUMA node when spreading.
template <typename SliceFn>
void ForEachPrefaultSlice(void* ptr, size_t mapped_size, size_t stride,
                          const HostBufferOptions& opts, SliceFn&& fn) {
  // Avoid creating threads for small pools. Chunk boundaries respect hugetlb
  // alignment as well as base pages, so MADV_POPULATE_WRITE can serve both.
  const size_t pages = mapped_size / stride;
  size_t workers = std::min<size_t>(std::clamp(opts.prefault_threads, 1, 16),
                                    std::max<size_t>(1, mapped_size / (64ULL << 20)));
  const auto& nodes = NodesWithCpus();
  const bool spread = opts.numa_node < 0 && opts.spread_prefault_across_nodes && nodes.size() > 1;
  // Whole rounds over the nodes, so first touch splits the mapping evenly.
  if (spread && workers >= nodes.size()) workers -= workers % nodes.size();
  if (workers == 1) {
    fn(ptr, mapped_size);
    return;
  }
  ParallelFor(workers, workers, [&](size_t i) {
    ScopedNumaAffinity affinity(spread ? nodes[i % nodes.size()] : opts.numa_node);
    const size_t begin = (pages / workers * i + std::min(i, pages % workers)) * stride;
    const size_t count = (pages / workers + (i < pages % workers)) * stride;
    fn(static_cast<char*>(ptr) + begin, count);
  });
}

void PrefaultPages(void* ptr, size_t mapped_size, size_t stride, const HostBufferOptions& opts) {
  ForEachPrefaultSlice(ptr, mapped_size, stride, opts,
                       [stride](void* p, size_t n) { PrefaultRange(p, n, stride); });
}

// What prefaulting a THP buffer found.
struct ThpPrefaultResult {
  bool collapse_supported = true;  // MADV_COLLAPSE ran on every chunk it reached
  bool collapse_failed = false;    // some chunk could not be made all-hugepage
  bool aborted = false;            // require_full stopped at the first such chunk
  int collapse_errno = 0;
};

// Prefaults a madvise(MADV_HUGEPAGE)d buffer a chunk at a time, then asks
// MADV_COLLAPSE to make each chunk all-hugepage.  On a chunk the faults
// already covered that is a cheap no-op; on one they did not, it retries
// synchronously and either succeeds or reports the shortfall, which is exact
// where smaps can only be read for the whole VMA.
ThpPrefaultResult PrefaultThp(void* ptr, size_t mapped_size, const HostBufferOptions& opts,
                              bool require_full) {
  std::atomic<bool> collapse_on{true};
  std::atomic<bool> collapse_supported{true};
  std::atomic<bool> collapse_failed{false};
  std::atomic<bool> abort{false};
  std::atomic<int> collapse_errno{0};
  ForEachPrefaultSlice(ptr, mapped_size, kThpSize, opts, [&](void* slice, size_t len) {
    char* base = static_cast<char*>(slice);
    for (size_t off = 0; off < len; off += kThpVerifyChunk) {
      if (abort.load(std::memory_order_relaxed)) return;
      const size_t n = std::min(kThpVerifyChunk, len - off);
      PrefaultRange(base + off, n, kThpSize);
      if (!collapse_on.load(std::memory_order_relaxed)) continue;
      if (madvise(base + off, n, MADV_COLLAPSE) == 0) continue;
      const int err = errno;
      if (err == EINVAL || err == ENOSYS) {
        collapse_supported = false;  // kernel < 6.1: smaps is the only measure
        collapse_on = false;
        continue;
      }
      collapse_failed = true;
      int expected = 0;
      collapse_errno.compare_exchange_strong(expected, err);
      if (require_full) {
        abort = true;
      } else {
        collapse_on = false;  // keep going on what faults give; don't retry each chunk
      }
    }
  });
  ThpPrefaultResult result;
  result.collapse_supported = collapse_supported;
  result.collapse_failed = collapse_failed;
  result.aborted = abort;
  result.collapse_errno = collapse_errno;
  return result;
}

#ifdef __linux__
int MbindMemory(void* ptr, size_t mapped_size, int numa_node, NumaBindMode mode) {
#if defined(__NR_mbind)
  if (ptr == nullptr || mapped_size == 0 || numa_node < 0) return 0;

  constexpr size_t kBitsPerWord = sizeof(unsigned long) * 8;
  const size_t word_count = static_cast<size_t>(numa_node) / kBitsPerWord + 1;
  std::vector<unsigned long> nodemask(word_count, 0);
  nodemask[static_cast<size_t>(numa_node) / kBitsPerWord] =
      1UL << (static_cast<size_t>(numa_node) % kBitsPerWord);

  const long rc = syscall(
      __NR_mbind, ptr, mapped_size, mode == NumaBindMode::kPreferred ? MPOL_PREFERRED : MPOL_BIND,
      nodemask.data(), static_cast<unsigned long>(word_count * kBitsPerWord), 0UL);
  if (rc == 0) return 0;
  return -errno;
#else
  (void)ptr;
  (void)mapped_size;
  (void)numa_node;
  return -ENOSYS;
#endif
}
#else
int MbindMemory(void* ptr, size_t mapped_size, int numa_node, NumaBindMode mode) {
  (void)ptr;
  (void)mapped_size;
  (void)numa_node;
  return -ENOSYS;
}
#endif

void ApplyPostMappingPolicies(HostBufferHandle& handle, const HostBufferOptions& opts) {
  if (!handle.valid()) return;

  if (opts.numa_node >= 0) {
    const int rc = MbindMemory(handle.ptr, handle.mapped_size, opts.numa_node, opts.numa_bind_mode);
    if (rc != 0 && opts.require_numa_binding) {
      MORI_UMBP_ERROR("HostMemAllocator: required mbind(node={}) failed ({}: {})", opts.numa_node,
                      -rc, std::strerror(-rc));
      HostMemAllocator{}.Free(handle);
      return;
    } else if (rc == -ENOSYS) {
      LogNumaUnavailableOnce();
    } else if (rc != 0 && opts.numa_bind_mode == NumaBindMode::kPreferred) {
      MORI_UMBP_ERROR(
          "HostMemAllocator: preferred mbind(node={}) failed ({}: {}); placement degraded",
          opts.numa_node, -rc, std::strerror(-rc));
    } else if (rc != 0) {
      MORI_UMBP_WARN("HostMemAllocator: mbind(node={}) failed ({}: {})", opts.numa_node, -rc,
                     std::strerror(-rc));
    }
  }

  if (opts.prefault) {
    const bool is_hugetlb = handle.actual_backing == HostBufferBacking::kAnonymousHugetlb ||
                            handle.actual_backing == HostBufferBacking::kAnonymousShmHugetlb ||
                            handle.actual_backing == HostBufferBacking::kAnonymousThp;
    const size_t stride = is_hugetlb ? handle.actual_alignment : GetPageSize();
    PrefaultPages(handle.ptr, handle.mapped_size, stride, opts);
  }
}

bool TryBuildHugepageFlags(size_t hugepage_size, int* out_flags) {
  if (!out_flags || !IsPowerOfTwo(hugepage_size)) return false;
#ifdef MAP_HUGE_SHIFT
  *out_flags = __builtin_ctzll(hugepage_size) << MAP_HUGE_SHIFT;
#else
  *out_flags = 0;
#endif
  return true;
}

HostBufferHandle AllocAnonymousBasePages(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;

  const size_t page_size = GetPageSize();
  const std::optional<size_t> mapped_size = AlignUpChecked(size, page_size);
  if (!mapped_size.has_value()) return handle;

  void* ptr =
      mmap(nullptr, *mapped_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (ptr == MAP_FAILED) return handle;

  handle.ptr = ptr;
  handle.requested_size = size;
  handle.mapped_size = *mapped_size;
  handle.actual_backing = HostBufferBacking::kAnonymous;
  handle.actual_alignment = page_size;
  ApplyPostMappingPolicies(handle, opts);
  return handle;
}

// kAnonymous with transparent hugepages: a 2 MiB aligned mapping (so every
// PMD can be a hugepage), madvise(MADV_HUGEPAGE) before any fault, interleave
// when nothing else places it, then prefault + verify.  See
// doc/design-thp-host-buffer.md.
HostBufferHandle AllocAnonymousThp(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;

  const std::optional<size_t> mapped_size = AlignUpChecked(size, kThpSize);
  if (!mapped_size.has_value() || *mapped_size > std::numeric_limits<size_t>::max() - kThpSize) {
    return handle;
  }

  // Over-map by one hugepage and trim to a 2 MiB aligned window.
  const size_t reserve = *mapped_size + kThpSize;
  void* raw = mmap(nullptr, reserve, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) return handle;
  const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
  const uintptr_t aligned = (raw_addr + kThpSize - 1) & ~(uintptr_t{kThpSize} - 1);
  const size_t head = aligned - raw_addr;
  const size_t tail = reserve - head - *mapped_size;
  if (head > 0) (void)munmap(raw, head);
  if (tail > 0) (void)munmap(reinterpret_cast<void*>(aligned + *mapped_size), tail);
  void* ptr = reinterpret_cast<void*>(aligned);

  LogThpModesOnce();
#ifdef MADV_HUGEPAGE
  if (madvise(ptr, *mapped_size, MADV_HUGEPAGE) != 0) {
    const int err = errno;
    MORI_UMBP_WARN(
        "HostMemAllocator: madvise(MADV_HUGEPAGE) failed ({}: {}); the buffer stays correct but "
        "its hugepage coverage will show the shortfall",
        err, std::strerror(err));
  }
#endif

  handle.ptr = ptr;
  handle.requested_size = size;
  handle.mapped_size = *mapped_size;
  handle.actual_backing = HostBufferBacking::kAnonymousThp;
  handle.actual_alignment = GetPageSize();  // raised to 2 MiB once coverage is proven

  int interleaved = 0;
  if (opts.numa_node < 0 && !opts.spread_prefault_across_nodes) {
    interleaved = InterleaveIfDefaultPolicy(ptr, *mapped_size);
  }

  // numa_node binding (and require_numa_binding) as for every backing; the
  // THP prefault below replaces the generic one.
  HostBufferOptions no_prefault = opts;
  no_prefault.prefault = false;
  ApplyPostMappingPolicies(handle, no_prefault);
  if (!handle.valid()) return handle;

  const bool require_full = ThpRequireFull(opts);
  if (!opts.prefault) {
    LogThpUnmeasuredOnce(require_full);
    return handle;
  }

  const auto start = std::chrono::steady_clock::now();
  const ThpPrefaultResult prefault = PrefaultThp(ptr, *mapped_size, opts, require_full);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const std::optional<ThpCoverage> measured = MeasureThpCoverage(ptr, *mapped_size);

  // smaps is exact when the buffer is its own VMA; otherwise trust a clean
  // MADV_COLLAPSE pass, which checked every chunk.
  const bool collapse_clean =
      prefault.collapse_supported && !prefault.collapse_failed && !prefault.aborted;
  bool full = false;
  double coverage = -1.0;
  if (measured.has_value() && measured->exact) {
    coverage = measured->coverage;
    full = coverage >= 1.0 - 1e-9 && !prefault.aborted;
  } else if (collapse_clean) {
    coverage = 1.0;
    full = true;
  } else if (measured.has_value()) {
    coverage = measured->coverage;
  }
  handle.thp_coverage = coverage;
  if (full) handle.actual_alignment = kThpSize;

  MORI_UMBP_INFO(
      "HostMemAllocator: THP buffer {} MiB, prefault {:.1f} s, hugepage coverage {}{}{}",
      *mapped_size >> 20, seconds, FormatCoverage(coverage),
      measured.has_value() && !measured->exact ? " (smaps approximate)" : "",
      interleaved > 0 ? ", interleaved over " + std::to_string(interleaved) + " nodes" : "");

  if (full) return handle;

  const std::string why =
      prefault.collapse_failed
          ? std::string("MADV_COLLAPSE failed (") + std::strerror(prefault.collapse_errno) + ")"
          : std::string("faults did not get hugepages");
  if (require_full) {
    MORI_UMBP_ERROR(
        "HostMemAllocator: THP buffer of {} MiB is not fully hugepage-backed ({}; {}); "
        "failing the allocation because UMBP_THP_REQUIRE_FULL is set. Reserve HugeTLB pages "
        "(vm.nr_hugepages) for a deterministic pool. {}",
        *mapped_size >> 20,
        prefault.aborted ? std::string("stopped at the first short chunk")
                         : "coverage " + FormatCoverage(coverage),
        why, ReadThpMeminfoSummary());
    HostMemAllocator{}.Free(handle);
    return handle;
  }
  static std::once_flag once;
  std::call_once(once, [&] {
    MORI_UMBP_WARN(
        "HostMemAllocator: THP buffer of {} MiB only {} hugepage-backed ({}); a NIC that needs "
        "one page size per memory region (AINIC) will reject it at registration. Options: "
        "reserve HugeTLB pages (vm.nr_hugepages), or have an operator compact memory "
        "(/proc/sys/vm/compact_memory) before start; set UMBP_THP_REQUIRE_FULL=1 to fail here "
        "instead. {}",
        *mapped_size >> 20, FormatCoverage(coverage), why, ReadThpMeminfoSummary());
  });
  return handle;
}

HostBufferHandle AllocAnonymous(size_t size, const HostBufferOptions& opts) {
  if (ThpRequested(opts)) {
    const std::string mode = ThpEnabledMode();
    if (mode == "always" || mode == "madvise") return AllocAnonymousThp(size, opts);
    LogThpUnusableOnce(mode);
  }
  return AllocAnonymousBasePages(size, opts);
}

// THP covers private anonymous memory only; memfd (shmem) THP follows
// shmem_enabled and has not been validated, so the shm backings keep their
// pages.
void LogThpIgnoredForShmOnce(const HostBufferOptions& opts) {
  if (!ThpRequested(opts)) return;
  static std::once_flag once;
  std::call_once(once, [] {
    MORI_UMBP_INFO(
        "HostMemAllocator: transparent hugepages apply to anonymous buffers only; shm "
        "(standalone process mode) buffers keep their pages");
  });
}

HostBufferHandle AllocAnonymousShm(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;
  LogThpIgnoredForShmOnce(opts);

  const size_t page_size = GetPageSize();
  const std::optional<size_t> mapped_size = AlignUpChecked(size, page_size);
  if (!mapped_size.has_value()) return handle;

  const int fd = MemfdCreate("umbp_host_buffer", MFD_CLOEXEC);
  if (fd < 0) {
    const int err = errno;
    MORI_UMBP_WARN("HostMemAllocator: memfd_create failed ({}: {})", err, std::strerror(err));
    return handle;
  }

  if (ftruncate(fd, static_cast<off_t>(*mapped_size)) != 0) {
    const int err = errno;
    MORI_UMBP_WARN("HostMemAllocator: ftruncate(memfd, {}) failed ({}: {})", *mapped_size, err,
                   std::strerror(err));
    close(fd);
    return handle;
  }

  void* ptr = mmap(nullptr, *mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    const int err = errno;
    MORI_UMBP_WARN("HostMemAllocator: mmap(memfd, {}) failed ({}: {})", *mapped_size, err,
                   std::strerror(err));
    close(fd);
    return handle;
  }

  handle.ptr = ptr;
  handle.requested_size = size;
  handle.mapped_size = *mapped_size;
  handle.actual_backing = HostBufferBacking::kAnonymousShm;
  handle.actual_alignment = page_size;

  {
    std::lock_guard<std::mutex> lock(ShmRegistryMutex());
    ShmAllocRecord record;
    record.fd = fd;
    record.size = *mapped_size;
    record.active_registrations = 0;
    ShmRegistry()[ptr] = record;
  }

  ApplyPostMappingPolicies(handle, opts);
  return handle;
}

// MFD_HUGE_*/MAP_HUGE_* share the same bit-shift encoding, so
// TryBuildHugepageFlags() covers both calls below. Falls back to
// AllocAnonymousShm, not AllocAnonymousHugetlb's AllocAnonymous, so
// fd-handoff still works if hugepages are unavailable.
HostBufferHandle AllocAnonymousShmHugetlb(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;
  LogThpIgnoredForShmOnce(opts);
  if (!IsPowerOfTwo(opts.hugepage_size)) {
    MORI_UMBP_WARN(
        "HostMemAllocator: invalid hugepage_size={}; falling back to AnonymousShm (4 KiB pages)",
        opts.hugepage_size);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymousShm;
    return AllocAnonymousShm(size, fallback);
  }

  const std::optional<size_t> mapped_size = AlignUpChecked(size, opts.hugepage_size);
  if (!mapped_size.has_value()) return handle;

#if !defined(MFD_HUGETLB) || !defined(MAP_HUGETLB)
  LogHugepageFallbackOnce(size, opts.hugepage_size, ENOTSUP);
  HostBufferOptions fallback = opts;
  fallback.backing = HostBufferBacking::kAnonymousShm;
  return AllocAnonymousShm(size, fallback);
#else
  int hugepage_flags = 0;
  if (!TryBuildHugepageFlags(opts.hugepage_size, &hugepage_flags)) {
    MORI_UMBP_WARN(
        "HostMemAllocator: cannot encode hugepage_size={} with this libc/kernel header set; "
        "falling back to AnonymousShm (4 KiB pages)",
        opts.hugepage_size);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymousShm;
    return AllocAnonymousShm(size, fallback);
  }

  const int fd =
      MemfdCreate("umbp_host_buffer_hugetlb", MFD_CLOEXEC | MFD_HUGETLB | hugepage_flags);
  if (fd < 0) {
    LogHugepageFallbackOnce(size, opts.hugepage_size, errno);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymousShm;
    return AllocAnonymousShm(size, fallback);
  }

  if (ftruncate(fd, static_cast<off_t>(*mapped_size)) != 0) {
    const int err = errno;
    MORI_UMBP_WARN("HostMemAllocator: ftruncate(hugetlb memfd, {}) failed ({}: {})", *mapped_size,
                   err, std::strerror(err));
    close(fd);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymousShm;
    return AllocAnonymousShm(size, fallback);
  }

  void* ptr = mmap(nullptr, *mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, fd, 0);
  if (ptr == MAP_FAILED) {
    LogHugepageFallbackOnce(size, opts.hugepage_size, errno);
    close(fd);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymousShm;
    return AllocAnonymousShm(size, fallback);
  }

  handle.ptr = ptr;
  handle.requested_size = size;
  handle.mapped_size = *mapped_size;
  handle.actual_backing = HostBufferBacking::kAnonymousShmHugetlb;
  handle.actual_alignment = opts.hugepage_size;

  {
    std::lock_guard<std::mutex> lock(ShmRegistryMutex());
    ShmAllocRecord record;
    record.fd = fd;
    record.size = *mapped_size;
    record.active_registrations = 0;
    ShmRegistry()[ptr] = record;
  }

  ApplyPostMappingPolicies(handle, opts);
  return handle;
#endif
}

HostBufferHandle AllocAnonymousHugetlb(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;
  if (!IsPowerOfTwo(opts.hugepage_size)) {
    MORI_UMBP_WARN("HostMemAllocator: invalid hugepage_size={}; falling back to anonymous pages",
                   opts.hugepage_size);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymous;
    return AllocAnonymous(size, fallback);
  }

  const std::optional<size_t> mapped_size = AlignUpChecked(size, opts.hugepage_size);
  if (!mapped_size.has_value()) return handle;

#ifndef MAP_HUGETLB
  LogHugepageFallbackOnce(size, opts.hugepage_size, ENOTSUP, ThpRequested(opts));
  HostBufferOptions fallback = opts;
  fallback.backing = HostBufferBacking::kAnonymous;
  return AllocAnonymous(size, fallback);
#else
  int hugepage_flags = 0;
  if (!TryBuildHugepageFlags(opts.hugepage_size, &hugepage_flags)) {
    MORI_UMBP_WARN(
        "HostMemAllocator: cannot encode hugepage_size={} with this libc/kernel header set; "
        "falling back to anonymous pages",
        opts.hugepage_size);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymous;
    return AllocAnonymous(size, fallback);
  }

  const int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | hugepage_flags;
  void* ptr = mmap(nullptr, *mapped_size, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (ptr == MAP_FAILED) {
    LogHugepageFallbackOnce(size, opts.hugepage_size, errno, ThpRequested(opts));
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymous;
    return AllocAnonymous(size, fallback);
  }

  handle.ptr = ptr;
  handle.requested_size = size;
  handle.mapped_size = *mapped_size;
  handle.actual_backing = HostBufferBacking::kAnonymousHugetlb;
  handle.actual_alignment = opts.hugepage_size;
  ApplyPostMappingPolicies(handle, opts);
  return handle;
#endif
}

}  // namespace

const char* HostBufferBackingName(HostBufferBacking backing) {
  switch (backing) {
    case HostBufferBacking::kAnonymous:
      return "anonymous";
    case HostBufferBacking::kAnonymousHugetlb:
      return "hugetlb";
    case HostBufferBacking::kAnonymousShm:
      return "shm";
    case HostBufferBacking::kAnonymousShmHugetlb:
      return "shm-hugetlb";
    case HostBufferBacking::kAnonymousThp:
      return "thp";
  }
  return "unknown";
}

std::optional<ThpCoverage> ThpCoverageFromSmaps(std::istream& smaps, uintptr_t begin, size_t size) {
  if (size == 0) return std::nullopt;
  const uintptr_t end = begin + size;
  double huge_bytes = 0.0;
  bool exact = true;
  bool any = false;

  // Current VMA, if it overlaps the range.
  bool in_vma = false;
  uintptr_t vma_begin = 0, vma_end = 0;
  auto overlap = [&] { return std::min(end, vma_end) - std::max(begin, vma_begin); };

  std::string line;
  while (std::getline(smaps, line)) {
    const size_t space = line.find(' ');
    const std::string first = line.substr(0, space);
    const size_t dash = first.find('-');
    if (dash != std::string::npos && first.find(':') == std::string::npos) {
      // "start-end perms offset dev inode path"
      char* stop = nullptr;
      const unsigned long long lo = std::strtoull(first.c_str(), &stop, 16);
      if (stop != first.c_str() + dash) {
        in_vma = false;
        continue;
      }
      const unsigned long long hi = std::strtoull(first.c_str() + dash + 1, &stop, 16);
      vma_begin = static_cast<uintptr_t>(lo);
      vma_end = static_cast<uintptr_t>(hi);
      in_vma = vma_end > begin && vma_begin < end && vma_end > vma_begin;
      if (in_vma) {
        any = true;
        if (vma_begin < begin || vma_end > end) exact = false;
      }
      continue;
    }
    if (!in_vma || line.rfind("AnonHugePages:", 0) != 0) continue;
    std::istringstream fields(line.substr(sizeof("AnonHugePages:") - 1));
    unsigned long long kb = 0;
    if (!(fields >> kb)) continue;
    const double bytes = static_cast<double>(kb) * 1024.0;
    const double vma_size = static_cast<double>(vma_end - vma_begin);
    huge_bytes += bytes * (static_cast<double>(overlap()) / vma_size);
  }
  if (!any) return std::nullopt;
  ThpCoverage result;
  result.coverage = std::min(1.0, huge_bytes / static_cast<double>(size));
  result.exact = exact;
  return result;
}

void SetThpEnabledModeForTesting(std::optional<std::string> mode) {
  std::lock_guard<std::mutex> lock(ThpOverrideMutex());
  ThpEnabledOverride() = std::move(mode);
}

HostBufferHandle HostMemAllocator::Alloc(size_t size, const HostBufferOptions& opts) {
  switch (opts.backing) {
    case HostBufferBacking::kAnonymous:
      return AllocAnonymous(size, opts);
    case HostBufferBacking::kAnonymousHugetlb:
      return AllocAnonymousHugetlb(size, opts);
    case HostBufferBacking::kAnonymousShm:
      return AllocAnonymousShm(size, opts);
    case HostBufferBacking::kAnonymousShmHugetlb:
      return AllocAnonymousShmHugetlb(size, opts);
    case HostBufferBacking::kAnonymousThp: {
      HostBufferOptions thp = opts;
      thp.backing = HostBufferBacking::kAnonymous;
      thp.thp = ThpMode::kOn;
      return AllocAnonymous(size, thp);
    }
    default:
      return {};
  }
}

void HostMemAllocator::Free(HostBufferHandle& handle) {
  if (!handle.valid()) return;

  void* ptr_to_unmap = handle.ptr;
  size_t size_to_unmap = handle.mapped_size;
  bool defer_unmap = false;
  int fd_to_close = -1;
  if (handle.actual_backing == HostBufferBacking::kAnonymousShm ||
      handle.actual_backing == HostBufferBacking::kAnonymousShmHugetlb) {
    std::lock_guard<std::mutex> lock(ShmRegistryMutex());
    auto it = ShmRegistry().find(handle.ptr);
    if (it != ShmRegistry().end()) {
      if (it->second.active_registrations > 0) {
        MORI_UMBP_WARN(
            "HostMemAllocator: freeing AnonymousShm buffer ptr={} while {} active registration(s) "
            "still reference it; deferring munmap/close until deregistration",
            handle.ptr, it->second.active_registrations);
        it->second.pending_free = true;
        defer_unmap = true;
      } else {
        fd_to_close = it->second.fd;
        ShmRegistry().erase(it);
      }
    }
  }

  handle.ptr = nullptr;
  handle.requested_size = 0;
  handle.mapped_size = 0;

  if (defer_unmap) return;

  if (munmap(ptr_to_unmap, size_to_unmap) != 0) {
    const int err = errno;
    MORI_UMBP_WARN(
        "HostMemAllocator: munmap failed ({}: {}); invalidating handle anyway "
        "to prevent a possible double-free of a reused VA range",
        err, std::strerror(err));
    // Keeping a stale-but-valid handle would let the next Free() munmap an
    // address the kernel may have already handed back to a later mmap().
  }

  if (fd_to_close >= 0) close(fd_to_close);
}

std::optional<HostMemAllocator::ShmAllocation> HostMemAllocator::LookupShmAllocation(uintptr_t ptr,
                                                                                     size_t size) {
  if (ptr == 0 || size == 0) return std::nullopt;
  if (ptr > std::numeric_limits<uintptr_t>::max() - (size - 1)) return std::nullopt;

  std::lock_guard<std::mutex> lock(ShmRegistryMutex());
  auto& registry = ShmRegistry();
  auto it = registry.upper_bound(reinterpret_cast<void*>(ptr));
  if (it == registry.begin()) return std::nullopt;
  --it;

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t end = base + it->second.size;
  uintptr_t req_end = ptr + size;
  if (ptr < base || req_end > end || it->second.fd < 0 || it->second.pending_free) {
    return std::nullopt;
  }

  ShmAllocation allocation;
  allocation.base = it->first;
  allocation.mapped_size = it->second.size;
  allocation.fd = it->second.fd;
  return allocation;
}

std::optional<HostMemAllocator::ShmAllocation> HostMemAllocator::AcquireShmAllocation(uintptr_t ptr,
                                                                                      size_t size) {
  if (ptr == 0 || size == 0) return std::nullopt;
  if (ptr > std::numeric_limits<uintptr_t>::max() - (size - 1)) return std::nullopt;

  std::lock_guard<std::mutex> lock(ShmRegistryMutex());
  auto& registry = ShmRegistry();
  auto it = registry.upper_bound(reinterpret_cast<void*>(ptr));
  if (it == registry.begin()) return std::nullopt;
  --it;

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t end = base + it->second.size;
  uintptr_t req_end = ptr + size;
  if (ptr < base || req_end > end || it->second.fd < 0 || it->second.pending_free) {
    return std::nullopt;
  }

  ++it->second.active_registrations;
  ShmAllocation allocation;
  allocation.base = it->first;
  allocation.mapped_size = it->second.size;
  allocation.fd = it->second.fd;
  return allocation;
}

void HostMemAllocator::ReleaseShmAllocation(uintptr_t base_ptr) {
  void* ptr_to_unmap = nullptr;
  size_t size_to_unmap = 0;
  int fd_to_close = -1;
  {
    std::lock_guard<std::mutex> lock(ShmRegistryMutex());
    auto it = ShmRegistry().find(reinterpret_cast<void*>(base_ptr));
    if (it == ShmRegistry().end()) return;
    if (it->second.active_registrations > 0) --it->second.active_registrations;
    if (it->second.active_registrations == 0 && it->second.pending_free) {
      ptr_to_unmap = it->first;
      size_to_unmap = it->second.size;
      fd_to_close = it->second.fd;
      ShmRegistry().erase(it);
    }
  }

  if (ptr_to_unmap) {
    if (munmap(ptr_to_unmap, size_to_unmap) != 0) {
      const int err = errno;
      MORI_UMBP_WARN("HostMemAllocator: deferred munmap failed ({}: {})", err, std::strerror(err));
    }
    if (fd_to_close >= 0) close(fd_to_close);
  }
}

}  // namespace mori::umbp
