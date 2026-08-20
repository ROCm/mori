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

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace mori::umbp {

// Makes a host region addressable from GPU kernels via one whole-region
// hipHostRegister call.
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
// caller simply behaves as it did before until that finishes. It must be a
// single registration covering the whole region -- see Register() -- so this is
// all-or-nothing rather than a progressive watermark.
//
// The device need not see the region at the host's address -- hipHostRegister's
// own documentation says the two "will have a different value" on many systems
// and that device code must use the device pointer. Equal addresses are a CUDA
// UVA property ROCm never promised. So the alias is recorded per device and
// DeviceAddress() translates by offset, which is sound because the whole region
// is one registration of one contiguous range.
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

  // True when [ptr, ptr + size) is registered, and therefore safe to copy from
  // or to without the pageable staging path. False is always the safe answer.
  // Weaker than "safe to hand to a kernel" -- that needs DeviceAddress().
  bool Covers(const void* ptr, size_t size) const;

  // Whether a kernel on `device_id` can reach this region at all, i.e. whether
  // DeviceAddress() will return non-null for a covered pointer.
  bool GatherableOn(int device_id) const;

  // The address `device_id` must use to reach [host_ptr, host_ptr + size), or
  // null when that range is outside the region or the device has no alias, in
  // which case the caller must stay on the copy-engine path. Subsumes Covers(),
  // so a caller that needs the address should not test both.
  void* DeviceAddress(const void* host_ptr, size_t size, int device_id) const;

  // Zero until registration succeeds, then the full region. For logging and
  // tests.
  size_t RegisteredBytes() const { return registered_bytes_.load(std::memory_order_acquire); }

 private:
  void Register();
  // Fills alias_bases_, one entry per device. False only when the devices
  // cannot be enumerated; a device without an alias gets a null entry.
  bool RecordDeviceAliases();
  // Bounds check against an already-loaded registered_bytes_, so an entry point
  // that needs the value anyway reads the atomic once.
  bool CoversRegistered(const void* ptr, size_t size, size_t registered) const;

  char* base_{nullptr};
  size_t bytes_{0};
  std::atomic<size_t> registered_bytes_{0};
  // Indexed by device ordinal. Written before registered_bytes_ is published
  // and read-only after, so that release/acquire pair is what publishes it.
  std::vector<char*> alias_bases_;
  std::thread worker_;
};

}  // namespace mori::umbp
