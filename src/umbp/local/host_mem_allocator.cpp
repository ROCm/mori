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

#include <sys/mman.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#endif

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "umbp/common/log.h"

namespace mori::umbp {
namespace {

constexpr size_t kDefaultPageSize = 4096;

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

void LogHugepageFallbackOnce(size_t size, size_t hugepage_size, int err) {
  static std::once_flag once;
  std::call_once(once, [size, hugepage_size, err] {
    const std::string meminfo = ReadHugepageMeminfoSummary();
    UMBP_LOG_WARN(
        "HostMemAllocator: MAP_HUGETLB allocation failed for size=%zu "
        "hugepage_size=%zu (%d: %s); falling back to anonymous pages. %s",
        size, hugepage_size, err, std::strerror(err), meminfo.c_str());
  });
}

void LogNumaUnavailableOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    UMBP_LOG_WARN("HostMemAllocator: NUMA binding unavailable on this build; ignoring numa_node");
  });
}

void TouchPages(void* ptr, size_t mapped_size, size_t stride) {
  volatile char* bytes = static_cast<volatile char*>(ptr);
  for (size_t offset = 0; offset < mapped_size; offset += stride) {
    bytes[offset] = bytes[offset];
  }
}

void PrefaultPages(void* ptr, size_t mapped_size, size_t stride) {
  if (ptr == nullptr || mapped_size == 0) return;

#ifdef MADV_POPULATE_WRITE
  if (madvise(ptr, mapped_size, MADV_POPULATE_WRITE) == 0) return;
  const int err = errno;
  UMBP_LOG_WARN(
      "HostMemAllocator: madvise(MADV_POPULATE_WRITE) failed (%d: %s); falling back "
      "to manual page touching",
      err, std::strerror(err));
#endif

  TouchPages(ptr, mapped_size, stride);
}

#ifdef __linux__
int MbindMemory(void* ptr, size_t mapped_size, int numa_node) {
#if defined(__NR_mbind)
  if (ptr == nullptr || mapped_size == 0 || numa_node < 0) return 0;

  constexpr size_t kBitsPerWord = sizeof(unsigned long) * 8;
  const size_t word_count = static_cast<size_t>(numa_node) / kBitsPerWord + 1;
  std::vector<unsigned long> nodemask(word_count, 0);
  nodemask[static_cast<size_t>(numa_node) / kBitsPerWord] =
      1UL << (static_cast<size_t>(numa_node) % kBitsPerWord);

  const long rc = syscall(__NR_mbind, ptr, mapped_size, MPOL_BIND, nodemask.data(),
                          static_cast<unsigned long>(word_count * kBitsPerWord), 0UL);
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
int MbindMemory(void* ptr, size_t mapped_size, int numa_node) {
  (void)ptr;
  (void)mapped_size;
  (void)numa_node;
  return -ENOSYS;
}
#endif

void ApplyPostMappingPolicies(HostBufferHandle& handle, const HostBufferOptions& opts) {
  if (!handle.valid()) return;

  if (opts.numa_node >= 0) {
    const int rc = MbindMemory(handle.ptr, handle.mapped_size, opts.numa_node);
    if (rc == -ENOSYS) {
      LogNumaUnavailableOnce();
    } else if (rc != 0) {
      UMBP_LOG_WARN("HostMemAllocator: mbind(node=%d) failed (%d: %s)", opts.numa_node, -rc,
                    std::strerror(-rc));
    }
  }

  if (opts.prefault) {
    const size_t stride = handle.actual_backing == HostBufferBacking::kAnonymousHugetlb
                              ? handle.actual_alignment
                              : GetPageSize();
    PrefaultPages(handle.ptr, handle.mapped_size, stride);
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

HostBufferHandle AllocAnonymous(size_t size, const HostBufferOptions& opts) {
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

HostBufferHandle AllocAnonymousHugetlb(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;
  if (!IsPowerOfTwo(opts.hugepage_size)) {
    UMBP_LOG_WARN("HostMemAllocator: invalid hugepage_size=%zu; falling back to anonymous pages",
                  opts.hugepage_size);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymous;
    return AllocAnonymous(size, fallback);
  }

  const std::optional<size_t> mapped_size = AlignUpChecked(size, opts.hugepage_size);
  if (!mapped_size.has_value()) return handle;

#ifndef MAP_HUGETLB
  LogHugepageFallbackOnce(size, opts.hugepage_size, ENOTSUP);
  HostBufferOptions fallback = opts;
  fallback.backing = HostBufferBacking::kAnonymous;
  return AllocAnonymous(size, fallback);
#else
  int hugepage_flags = 0;
  if (!TryBuildHugepageFlags(opts.hugepage_size, &hugepage_flags)) {
    UMBP_LOG_WARN(
        "HostMemAllocator: cannot encode hugepage_size=%zu with this libc/kernel header set; "
        "falling back to anonymous pages",
        opts.hugepage_size);
    HostBufferOptions fallback = opts;
    fallback.backing = HostBufferBacking::kAnonymous;
    return AllocAnonymous(size, fallback);
  }

  const int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | hugepage_flags;
  void* ptr = mmap(nullptr, *mapped_size, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (ptr == MAP_FAILED) {
    LogHugepageFallbackOnce(size, opts.hugepage_size, errno);
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

HostBufferHandle HostMemAllocator::Alloc(size_t size, const HostBufferOptions& opts) {
  switch (opts.backing) {
    case HostBufferBacking::kAnonymous:
      return AllocAnonymous(size, opts);
    case HostBufferBacking::kAnonymousHugetlb:
      return AllocAnonymousHugetlb(size, opts);
    default:
      return {};
  }
}

void HostMemAllocator::Free(HostBufferHandle& handle) {
  if (!handle.valid()) return;

  if (munmap(handle.ptr, handle.mapped_size) != 0) {
    const int err = errno;
    UMBP_LOG_WARN(
        "HostMemAllocator: munmap failed (%d: %s); invalidating handle anyway "
        "to prevent a possible double-free of a reused VA range",
        err, std::strerror(err));
    // Fall through to invalidation below: keeping a stale-but-valid handle
    // would let the next Free() munmap an address the kernel may have
    // already handed back to a later mmap().  Accepting a small leak on
    // the (already-rare) munmap-failure path is the lesser evil.
  }

  handle.ptr = nullptr;
  handle.requested_size = 0;
  handle.mapped_size = 0;
}

}  // namespace mori::umbp
