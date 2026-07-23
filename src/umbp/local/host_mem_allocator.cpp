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
#include <sys/syscall.h>
#endif

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {

constexpr size_t kDefaultPageSize = 4096;

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

void LogHugepageFallbackOnce(size_t size, size_t hugepage_size, int err) {
  static std::once_flag once;
  std::call_once(once, [size, hugepage_size, err] {
    const std::string meminfo = ReadHugepageMeminfoSummary();
    MORI_UMBP_WARN(
        "HostMemAllocator: MAP_HUGETLB allocation failed for size={} "
        "hugepage_size={} ({}: {}); falling back to anonymous pages. {}",
        size, hugepage_size, err, std::strerror(err), meminfo);
  });
}

void LogNumaUnavailableOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    MORI_UMBP_WARN("HostMemAllocator: NUMA binding unavailable on this build; ignoring numa_node");
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
  MORI_UMBP_WARN(
      "HostMemAllocator: madvise(MADV_POPULATE_WRITE) failed ({}: {}); falling back "
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
      MORI_UMBP_WARN("HostMemAllocator: mbind(node={}) failed ({}: {})", opts.numa_node, -rc,
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

HostBufferHandle AllocAnonymousShm(size_t size, const HostBufferOptions& opts) {
  HostBufferHandle handle;
  if (size == 0) return handle;

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
  LogHugepageFallbackOnce(size, opts.hugepage_size, ENOTSUP);
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
    case HostBufferBacking::kAnonymousShm:
      return AllocAnonymousShm(size, opts);
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
  if (handle.actual_backing == HostBufferBacking::kAnonymousShm) {
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
