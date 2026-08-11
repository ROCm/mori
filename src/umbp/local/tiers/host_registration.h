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

// Source-private to the DRAM tier; not part of the installed API.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace mori::umbp {

// Makes a tier's host region addressable from GPU kernels via hipHostRegister.
//
// Two things depend on this. A gather kernel cannot dereference plain mmap
// memory at all -- it faults the GPU -- so registration is a hard prerequisite
// for the scatter-gather copy path. Independently, registering takes
// hipMemcpyAsync off the pageable staging path, which alone is worth ~6x on the
// per-fragment copies that remain.
//
// Registration runs at roughly 13 GiB/s and does not parallelize: splitting it
// across threads measured slower, because it serializes inside the driver. A
// 512 GiB tier would therefore add ~40 s to server startup, so anything larger
// than the synchronous threshold is registered on a background thread and the
// tier simply behaves as it did before until that finishes. It must be a single
// registration covering the whole region -- see Register() -- so this is
// all-or-nothing rather than a progressive watermark.
class HostTierRegistration {
 public:
  // Registers [base, base + bytes). `bytes` must be the mapped (page-aligned)
  // length, not the usable capacity. Never throws and never fails hard: if
  // registration is unavailable the object simply reports Covers() == false
  // forever.
  HostTierRegistration(void* base, size_t bytes);
  ~HostTierRegistration();

  HostTierRegistration(const HostTierRegistration&) = delete;
  HostTierRegistration& operator=(const HostTierRegistration&) = delete;

  // True when [ptr, ptr + size) is registered, and therefore safe to hand to a
  // kernel. False is always the safe answer.
  bool Covers(const void* ptr, size_t size) const;

  // Zero until registration succeeds, then the full region. For logging and
  // tests.
  size_t RegisteredBytes() const { return registered_bytes_.load(std::memory_order_acquire); }

 private:
  void Register();

  char* base_{nullptr};
  size_t bytes_{0};
  std::atomic<size_t> registered_bytes_{0};
  std::thread worker_;
};

}  // namespace mori::umbp
