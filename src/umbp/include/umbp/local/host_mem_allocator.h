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

namespace mori::umbp {

enum class HostBufferBacking : int {
  kAnonymous = 0,
  kAnonymousHugetlb = 1,
};

struct HostBufferOptions {
  HostBufferBacking backing = HostBufferBacking::kAnonymous;
  size_t hugepage_size = 2ULL * 1024 * 1024;
  int numa_node = -1;
  bool prefault = true;
};

struct HostBufferHandle {
  void* ptr = nullptr;
  size_t requested_size = 0;
  size_t mapped_size = 0;
  HostBufferBacking actual_backing = HostBufferBacking::kAnonymous;
  size_t actual_alignment = 0;

  bool valid() const { return ptr != nullptr; }
};

class HostMemAllocator {
 public:
  HostMemAllocator() = default;
  ~HostMemAllocator() = default;

  // Allocates a host buffer per `opts`. Returns an invalid handle
  // (`!handle.valid()`) on failure; never throws. On `kAnonymousHugetlb`
  // failure (e.g. nr_hugepages == 0), retries with `kAnonymous` and
  // returns a valid handle whose `actual_backing == kAnonymous`.
  HostBufferHandle Alloc(size_t size, const HostBufferOptions& opts);

  // Releases the mapping described by `handle` and invalidates the handle in
  // place: on return, `handle.ptr == nullptr` and `mapped_size == 0`.
  // Idempotent — calling Free with an already-invalid handle is a no-op.
  // Crucially, the handle is invalidated even when the underlying munmap
  // fails (a WARN is logged), to prevent a subsequent Free from munmap'ing
  // an address the kernel may have reused for a later mmap().
  void Free(HostBufferHandle& handle);
};

}  // namespace mori::umbp
