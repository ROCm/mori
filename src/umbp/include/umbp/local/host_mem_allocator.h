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
#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace mori::umbp {

enum class HostBufferBacking : int {
  kAnonymous = 0,
  kAnonymousHugetlb = 1,
  kAnonymousShm = 2,
  // Like kAnonymousShm (fd-shareable via memfd+SCM_RIGHTS) but hugetlbfs-
  // backed (MFD_HUGETLB+MAP_HUGETLB): some RDMA NICs have a small MTT table
  // and cannot register a large 4 KiB-paged region.
  kAnonymousShmHugetlb = 3,
  // A kAnonymous mapping, 2 MiB aligned and madvise(MADV_HUGEPAGE)d, so the
  // kernel backs it with transparent hugepages when it can.  kAnonymous (or a
  // kAnonymousHugetlb fallback) becomes this when THP is on
  // (`HostBufferOptions::thp`, `UMBP_USE_THP`); requesting it directly turns
  // THP on for that call.  If the host has THP disabled ([never]) the result
  // is plain kAnonymous.  THP is best effort;
  // `HostBufferHandle::thp_coverage` says how much of the buffer got it.
  kAnonymousThp = 4,
};

const char* HostBufferBackingName(HostBufferBacking backing);

// Whether the kAnonymous path asks for transparent hugepages.
enum class ThpMode : int { kFromEnv = 0, kOff = 1, kOn = 2 };

enum class NumaBindMode { kBind, kPreferred };

struct HostBufferOptions {
  HostBufferBacking backing = HostBufferBacking::kAnonymous;
  size_t hugepage_size = 2ULL * 1024 * 1024;
  int numa_node = -1;
  bool prefault = true;
  NumaBindMode numa_bind_mode = NumaBindMode::kBind;
  bool require_numa_binding = false;  // Preserve best-effort binding for existing callers.
  int prefault_threads = 1;
  // Unbound mappings only: run prefault worker i on NUMA node i % N, so first
  // touch spreads the chunks evenly instead of filling one node first.
  bool spread_prefault_across_nodes = false;
  // kFromEnv reads UMBP_USE_THP once per process; tests set kOn/kOff.
  ThpMode thp = ThpMode::kFromEnv;
  // THP only, and only with prefault: fail the allocation (invalid handle)
  // unless every 2 MiB of the buffer is a hugepage.  OR'ed with
  // UMBP_THP_REQUIRE_FULL.  NICs that take one page size per memory region
  // (AINIC) reject a region holding even one 4 KiB page, so this moves that
  // failure from registration to allocation, where it says why.
  bool thp_require_full = false;
};

struct HostBufferHandle {
  void* ptr = nullptr;
  size_t requested_size = 0;
  size_t mapped_size = 0;
  HostBufferBacking actual_backing = HostBufferBacking::kAnonymous;
  // kAnonymousThp: 2 MiB only when the whole buffer is hugepage-backed, else
  // the base page size, so page-entry estimates (nic_limits) stay correct.
  size_t actual_alignment = 0;
  // kAnonymousThp with prefault: fraction of mapped_size backed by
  // transparent hugepages after prefault, in [0, 1].  -1 when not measured
  // (not THP, prefault off, or /proc/self/smaps unreadable).
  double thp_coverage = -1.0;

  bool valid() const { return ptr != nullptr; }
};

class HostMemAllocator {
 public:
  struct ShmAllocation {
    void* base = nullptr;
    size_t mapped_size = 0;
    int fd = -1;
  };

  HostMemAllocator() = default;
  ~HostMemAllocator() = default;

  // Allocates a host buffer per `opts`. Returns an invalid handle
  // (`!handle.valid()`) on failure; never throws. On `kAnonymousHugetlb`
  // failure (e.g. nr_hugepages == 0), retries with `kAnonymous` and
  // returns a valid handle whose `actual_backing == kAnonymous` (or
  // `kAnonymousThp` when THP is on).
  HostBufferHandle Alloc(size_t size, const HostBufferOptions& opts);

  // Releases the mapping described by `handle` and invalidates the handle in
  // place: on return, `handle.ptr == nullptr` and `mapped_size == 0`.
  // Idempotent — calling Free with an already-invalid handle is a no-op.
  // Crucially, the handle is invalidated even when the underlying munmap
  // fails (a WARN is logged), to prevent a subsequent Free from munmap'ing
  // an address the kernel may have reused for a later mmap().
  void Free(HostBufferHandle& handle);

  // Look up a live kAnonymousShm allocation containing [ptr, ptr + size).
  // The returned fd is owned by the allocator registry; callers may pass it
  // through SCM_RIGHTS but must not close it.
  static std::optional<ShmAllocation> LookupShmAllocation(uintptr_t ptr, size_t size);

  // Acquire/Release a live shm allocation while its fd is being handed to, or
  // remains registered with, another process.  Acquire increments the registry's
  // active reference count before returning the fd, closing the lookup/use
  // race with Free().
  static std::optional<ShmAllocation> AcquireShmAllocation(uintptr_t ptr, size_t size);
  static void ReleaseShmAllocation(uintptr_t base);
};

// Hugepage coverage of [begin, begin + size) from /proc/<pid>/smaps text:
// the AnonHugePages of every VMA overlapping the range over `size`.  A VMA
// that extends past the range (the kernel merged it with a neighbour) is
// counted pro rata and makes the result inexact.  nullopt when no VMA
// overlaps the range.  Exposed for tests; the allocator reads
// /proc/self/smaps.
struct ThpCoverage {
  double coverage = 0.0;
  bool exact = true;
};
std::optional<ThpCoverage> ThpCoverageFromSmaps(std::istream& smaps, uintptr_t begin, size_t size);

// Test only: pretend /sys/kernel/mm/transparent_hugepage/enabled selects
// `mode` ("always", "madvise", "never", or "" for "THP not built in");
// nullopt restores the real file.
void SetThpEnabledModeForTesting(std::optional<std::string> mode);

}  // namespace mori::umbp
