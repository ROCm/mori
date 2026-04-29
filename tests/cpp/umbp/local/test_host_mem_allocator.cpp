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
#include <unistd.h>

#include "umbp/local/host_mem_allocator.h"

#ifdef __linux__
#include <dirent.h>
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace mori::umbp;

namespace {

constexpr size_t kHugepage2Mb = 2ULL * 1024 * 1024;

struct HugepageInfo {
  size_t total = 0;
  size_t free = 0;
  size_t size_bytes = 0;
};

size_t GetPageSize() {
  const long page = sysconf(_SC_PAGESIZE);
  return page > 0 ? static_cast<size_t>(page) : 4096ULL;
}

size_t AlignUp(size_t size, size_t alignment) { return (size + alignment - 1) & ~(alignment - 1); }

HugepageInfo ReadHugepageInfo() {
  HugepageInfo info;
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) return info;

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.rfind("HugePages_Total:", 0) == 0) {
      std::istringstream iss(line.substr(std::strlen("HugePages_Total:")));
      iss >> info.total;
    } else if (line.rfind("HugePages_Free:", 0) == 0) {
      std::istringstream iss(line.substr(std::strlen("HugePages_Free:")));
      iss >> info.free;
    } else if (line.rfind("Hugepagesize:", 0) == 0) {
      std::istringstream iss(line.substr(std::strlen("Hugepagesize:")));
      size_t value = 0;
      std::string unit;
      if (iss >> value) {
        info.size_bytes = (iss >> unit) && unit == "kB" ? value * 1024ULL : value;
      }
    }
  }
  return info;
}

bool CanAttemptHugepageAlloc(size_t bytes, size_t hugepage_size) {
  const HugepageInfo info = ReadHugepageInfo();
  if (info.total == 0 || info.free == 0 || info.size_bytes == 0) return false;
  if (info.size_bytes != hugepage_size) return false;
  return info.free * info.size_bytes >= AlignUp(bytes, hugepage_size);
}

void TouchRange(const HostBufferHandle& handle) {
  assert(handle.valid());
  auto* bytes = static_cast<unsigned char*>(handle.ptr);
  bytes[0] = 0x1A;
  bytes[handle.requested_size - 1] = 0xC3;
  assert(bytes[0] == 0x1A);
  assert(bytes[handle.requested_size - 1] == 0xC3);
}

void TestAnonymousAllocFreeRoundTrip() {
  std::printf("  AnonymousAllocFreeRoundTrip...\n");
  HostMemAllocator allocator;

  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymous;
  auto handle = allocator.Alloc(12345, opts);
  assert(handle.valid());
  assert(handle.requested_size == 12345);
  assert(handle.actual_backing == HostBufferBacking::kAnonymous);
  assert(handle.actual_alignment == GetPageSize());
  assert(handle.mapped_size >= handle.requested_size);
  assert(handle.mapped_size % GetPageSize() == 0);
  assert(reinterpret_cast<uintptr_t>(handle.ptr) % handle.actual_alignment == 0);
  TouchRange(handle);

  allocator.Free(handle);
  assert(!handle.valid());
  assert(handle.requested_size == 0);
  assert(handle.mapped_size == 0);
  assert(handle.actual_backing == HostBufferBacking::kAnonymous);
  assert(handle.actual_alignment == GetPageSize());
  std::printf("    PASS\n");
}

void TestAnonymousHugetlbWhenAvailable() {
  std::printf("  AnonymousHugetlbWhenAvailable...\n");
  if (!CanAttemptHugepageAlloc(kHugepage2Mb, kHugepage2Mb)) {
    std::printf("    SKIPPED (no free 2 MiB hugepages)\n");
    return;
  }

  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousHugetlb;
  opts.hugepage_size = kHugepage2Mb;
  auto handle = allocator.Alloc(kHugepage2Mb, opts);
  if (!handle.valid() || handle.actual_backing != HostBufferBacking::kAnonymousHugetlb) {
    std::printf("    SKIPPED (hugetlb request fell back on this host)\n");
    return;
  }

  assert(handle.actual_alignment == kHugepage2Mb);
  assert(handle.mapped_size == kHugepage2Mb);
  assert(reinterpret_cast<uintptr_t>(handle.ptr) % handle.actual_alignment == 0);
  TouchRange(handle);
  allocator.Free(handle);
  assert(!handle.valid());
  std::printf("    PASS\n");
}

void TestHugetlbFallsBackToAnonymous() {
  std::printf("  HugetlbFallsBackToAnonymous...\n");
  if (CanAttemptHugepageAlloc(kHugepage2Mb, kHugepage2Mb)) {
    std::printf("    SKIPPED (host has free 2 MiB hugepages)\n");
    return;
  }

  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousHugetlb;
  opts.hugepage_size = kHugepage2Mb;
  auto handle = allocator.Alloc(64 * 1024, opts);
  assert(handle.valid());
  assert(handle.actual_backing == HostBufferBacking::kAnonymous);
  assert(handle.actual_alignment == GetPageSize());
  TouchRange(handle);
  allocator.Free(handle);
  std::printf("    PASS\n");
}

#ifdef __linux__
int CountOnlineNumaNodes() {
  DIR* dir = opendir("/sys/devices/system/node");
  if (!dir) return 0;

  int count = 0;
  while (dirent* entry = readdir(dir)) {
    if (std::strncmp(entry->d_name, "node", 4) != 0) continue;
    bool all_digits = true;
    for (const char* p = entry->d_name + 4; *p != '\0'; ++p) {
      if (*p < '0' || *p > '9') {
        all_digits = false;
        break;
      }
    }
    if (all_digits) ++count;
  }
  closedir(dir);
  return count;
}

std::optional<std::vector<int>> QueryNodesForPages(const std::vector<void*>& pages) {
#if defined(__NR_move_pages)
  std::vector<int> status(pages.size(), -1);
  const long rc = syscall(__NR_move_pages, 0, pages.size(), const_cast<void**>(pages.data()),
                          nullptr, status.data(), 0);
  if (rc != 0) return std::nullopt;
  return status;
#else
  (void)pages;
  return std::nullopt;
#endif
}
#else
int CountOnlineNumaNodes() { return 0; }

std::optional<std::vector<int>> QueryNodesForPages(const std::vector<void*>& pages) {
  (void)pages;
  return std::nullopt;
}
#endif

void TestNumaBindingActuallyBinds() {
  std::printf("  NumaBindingActuallyBinds...\n");
  if (CountOnlineNumaNodes() <= 1) {
    std::printf("    SKIPPED (single-node or NUMA sysfs unavailable)\n");
    return;
  }

  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymous;
  opts.numa_node = 0;
  opts.prefault = true;
  auto handle = allocator.Alloc(GetPageSize() * 4, opts);
  assert(handle.valid());

  std::vector<void*> pages;
  for (size_t offset = 0; offset < handle.mapped_size; offset += GetPageSize()) {
    pages.push_back(static_cast<char*>(handle.ptr) + offset);
  }

  const auto nodes = QueryNodesForPages(pages);
  if (!nodes.has_value()) {
    allocator.Free(handle);
    std::printf("    SKIPPED (move_pages query unavailable on this host)\n");
    return;
  }

  for (int node : *nodes) {
    if (node < 0) {
      allocator.Free(handle);
      std::printf("    SKIPPED (move_pages returned per-page error)\n");
      return;
    }
    assert(node == 0);
  }

  allocator.Free(handle);
  std::printf("    PASS\n");
}

void TestNullHandleAfterAllocFailure() {
  std::printf("  NullHandleAfterAllocFailure...\n");
  HostMemAllocator allocator;
  HostBufferOptions opts;
  auto handle = allocator.Alloc(std::numeric_limits<size_t>::max(), opts);
  assert(!handle.valid());
  assert(handle.ptr == nullptr);
  assert(handle.requested_size == 0);
  assert(handle.mapped_size == 0);
  allocator.Free(handle);
  assert(!handle.valid());
  std::printf("    PASS\n");
}

void TestMappedSizeRoundsUp() {
  std::printf("  MappedSizeRoundsUp...\n");
  if (!CanAttemptHugepageAlloc(kHugepage2Mb * 2, kHugepage2Mb)) {
    std::printf("    SKIPPED (no free 2 MiB hugepages)\n");
    return;
  }

  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousHugetlb;
  opts.hugepage_size = kHugepage2Mb;
  auto handle = allocator.Alloc(kHugepage2Mb + 1, opts);
  if (!handle.valid() || handle.actual_backing != HostBufferBacking::kAnonymousHugetlb) {
    std::printf("    SKIPPED (hugetlb request fell back on this host)\n");
    return;
  }

  assert(handle.requested_size == kHugepage2Mb + 1);
  assert(handle.mapped_size == kHugepage2Mb * 2);
  allocator.Free(handle);
  std::printf("    PASS\n");
}

void TestDoubleFreeIsSafe() {
  std::printf("  DoubleFreeIsSafe...\n");
  HostMemAllocator allocator;
  auto handle = allocator.Alloc(4096, HostBufferOptions{});
  assert(handle.valid());

  allocator.Free(handle);
  assert(!handle.valid());
  assert(handle.requested_size == 0);
  assert(handle.mapped_size == 0);

  allocator.Free(handle);
  assert(!handle.valid());
  assert(handle.requested_size == 0);
  assert(handle.mapped_size == 0);
  std::printf("    PASS\n");
}

// Pins the contract that Free invalidates the handle even when the
// underlying munmap fails — without this guarantee, a follow-up Free could
// munmap an address the kernel has since reused for a later mmap().  We
// induce munmap failure by corrupting `mapped_size` to 0 (man munmap:
// length==0 → EINVAL).  This intentionally leaks the original 4 KiB
// mapping; the OS reclaims it at process exit.
void TestFreeInvalidatesEvenOnMunmapFailure() {
  std::printf("  FreeInvalidatesEvenOnMunmapFailure...\n");
  HostMemAllocator allocator;
  auto handle = allocator.Alloc(4096, HostBufferOptions{});
  assert(handle.valid());

  handle.mapped_size = 0;  // munmap(ptr, 0) → EINVAL
  allocator.Free(handle);

  assert(!handle.valid());
  assert(handle.ptr == nullptr);
  assert(handle.requested_size == 0);
  assert(handle.mapped_size == 0);

  // A subsequent Free must also short-circuit cleanly.
  allocator.Free(handle);
  assert(!handle.valid());
  std::printf("    PASS\n");
}

void TestFreeInvalidatesHandle() {
  std::printf("  FreeInvalidatesHandle...\n");
  if (!CanAttemptHugepageAlloc(kHugepage2Mb, kHugepage2Mb)) {
    std::printf("    SKIPPED (no free 2 MiB hugepages)\n");
    return;
  }

  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousHugetlb;
  opts.hugepage_size = kHugepage2Mb;
  auto handle = allocator.Alloc(kHugepage2Mb, opts);
  if (!handle.valid() || handle.actual_backing != HostBufferBacking::kAnonymousHugetlb) {
    std::printf("    SKIPPED (hugetlb request fell back on this host)\n");
    return;
  }

  const auto original_backing = handle.actual_backing;
  const auto original_alignment = handle.actual_alignment;
  allocator.Free(handle);
  assert(!handle.valid());
  assert(handle.requested_size == 0);
  assert(handle.mapped_size == 0);
  assert(handle.actual_backing == original_backing);
  assert(handle.actual_alignment == original_alignment);
  std::printf("    PASS\n");
}

}  // namespace

int main() {
  std::printf("=== test_host_mem_allocator ===\n");
  TestAnonymousAllocFreeRoundTrip();
  TestAnonymousHugetlbWhenAvailable();
  TestHugetlbFallsBackToAnonymous();
  TestNumaBindingActuallyBinds();
  TestNullHandleAfterAllocFailure();
  TestMappedSizeRoundsUp();
  TestDoubleFreeIsSafe();
  TestFreeInvalidatesEvenOnMunmapFailure();
  TestFreeInvalidatesHandle();
  std::printf("=== ALL PASSED ===\n");
  return 0;
}
