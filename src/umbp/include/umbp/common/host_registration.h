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
#include <cstdint>
#include <thread>
#include <vector>

namespace mori::umbp {

bool HostRegistrationEnabled();

// The size of the pieces a HostTierRegistration registers, and so the
// alignment of the boundaries no single copy of registered memory may cross:
// hipMemcpy rejects a host range spanning two registrations, and a gather
// fragment can use only one piece's device alias. Pieces are cut at absolute
// multiples of this, so a caller needs nothing but the address to split at
// them. UMBP_DRAM_HOST_REGISTER_CHUNK_MB, default 1024; 0 registers each
// region whole.
size_t HostRegistrationChunkBytes();

// The most host memory this process registers for GPU access, over every
// region: UMBP_DRAM_HOST_REGISTER_MAX_MB, in MiB. Unset or 0 is no limit
// (SIZE_MAX), and nothing below changes.
//
// Registered host memory is not free: the driver accounts it against the
// host, and an engine on the same node registers host memory of its own. A
// store too large to register whole can still be used whole: what is over the
// limit stays plain memory, which a copy reaches on the pageable path and a
// gather kernel never touches.
size_t HostRegistrationMaxBytes();
// The limit `value` asks for; SIZE_MAX for null, empty, 0 or anything that is
// not a number.
size_t ParseHostRegistrationMaxMb(const char* value);

// What is left of that limit. A region takes its share when it is declared,
// before it registers anything, so the regions are served in the order they
// are declared whatever order their background registrations finish in: the
// first regions whole, one region up to the last piece that fits, the rest
// not at all.
class HostRegistrationBudget {
 public:
  explicit HostRegistrationBudget(size_t limit_bytes) : limit_(limit_bytes) {}

  // Reserves as many of a region's leading pieces as fit in what is left.
  // Returns how many; `*bytes` is their total. A piece is taken whole or not
  // at all, and none after the first that does not fit.
  size_t Reserve(const std::vector<size_t>& piece_bytes, size_t* bytes);
  // Gives back what a region reserved, when the region goes.
  void Release(size_t bytes);

  bool limited() const { return limit_ != SIZE_MAX; }
  size_t reserved() const { return reserved_.load(std::memory_order_relaxed); }

 private:
  const size_t limit_;
  std::atomic<size_t> reserved_{0};
};

// Makes a host region addressable from GPU kernels via hipHostRegister.
//
// Two things depend on this. A gather kernel cannot dereference plain mmap
// memory at all -- it faults the GPU -- so registration is a hard prerequisite
// for the scatter-gather copy path. Independently, registering takes
// hipMemcpyAsync off the pageable staging path, which alone is worth ~6x on the
// per-fragment copies that remain.
//
// The region is registered in HostRegistrationChunkBytes() pieces, not whole,
// because the GPU driver does not pin what it registers: it mirrors the range,
// and when the kernel moves any page inside it (compaction does this
// continuously on a host with fragmented free memory) it stops every GPU queue
// of the process until it has re-validated that entire registration. Over a
// terabyte pool that re-validation takes minutes and is restarted by the next
// moved page, so the process gets no GPU work done at all; over one piece it
// takes a fraction of a second, though a burst of compaction can still touch
// many pieces at once. A pool on hugetlbfs pages avoids all of it: compaction
// never moves them.
//
// Registration does not parallelize -- splitting it across threads measured
// slower, because it serializes inside the driver -- so anything larger than
// the synchronous threshold is registered on a background thread, piece by
// piece in address order. Each piece is usable as soon as it is registered;
// until then a copy touching it simply takes the path it took before. So does
// a copy touching a piece that is never registered: one that failed, or one
// past the process's limit (`HostRegistrationMaxBytes`). A region registered
// in part, from its lowest address up, is an ordinary state.
//
// The device need not see a piece at the host's address -- hipHostRegister's
// own documentation says the two "will have a different value" on many systems
// and that device code must use the device pointer. Equal addresses are a CUDA
// UVA property ROCm never promised. So each piece records its alias per device
// and DeviceAddress() translates by offset within the piece.
class HostTierRegistration {
 public:
  // Registers [base, base + bytes). `bytes` must be the mapped (page-aligned)
  // length, not the usable capacity. Never throws and never fails hard: if
  // registration is unavailable DeviceAddress() simply returns null forever.
  HostTierRegistration(void* base, size_t bytes);
  // The same against `budget` instead of the process's; for tests.
  HostTierRegistration(void* base, size_t bytes, HostRegistrationBudget* budget);
  ~HostTierRegistration();

  HostTierRegistration(const HostTierRegistration&) = delete;
  HostTierRegistration& operator=(const HostTierRegistration&) = delete;

  // Whether a kernel on `device_id` can reach this region at all, i.e. whether
  // DeviceAddress() will return non-null for a pointer in a registered piece.
  bool GatherableOn(int device_id) const;

  // Bytes registered so far, and bytes the limit keeps unregistered for good.
  size_t RegisteredBytes() const { return registered_bytes_.load(std::memory_order_relaxed); }
  size_t BytesOverLimit() const { return bytes_ - allowed_bytes_; }

  // The address `device_id` must use to reach [host_ptr, host_ptr + size), or
  // null when that range is not inside one registered piece or the device has
  // no alias, in which case the caller must stay on the copy-engine path.
  void* DeviceAddress(const void* host_ptr, size_t size, int device_id) const;

 private:
  struct Piece {
    char* base = nullptr;
    size_t bytes = 0;
    // Indexed by device ordinal; null where the device has no mapping.
    std::vector<char*> aliases;
  };

  void Register();
  // The registered piece holding all of [ptr, ptr + size), or null.
  const Piece* FindPiece(const void* ptr, size_t size) const;

  char* base_{nullptr};
  size_t bytes_{0};
  size_t piece_bytes_{0};
  // Laid out by the constructor, before any reader or the worker exists.
  std::vector<Piece> pieces_;
  // Pieces [0, allowed_pieces_) may be registered: all of them, unless the
  // process's limit was reached inside or before this region.
  HostRegistrationBudget* budget_{nullptr};
  size_t allowed_pieces_{0};
  size_t allowed_bytes_{0};
  // Pieces [0, registered_pieces_) are registered. A piece, aliases included,
  // is written before the count that covers it is released, and is read-only
  // after, so that release/acquire pair is what publishes it.
  std::atomic<size_t> registered_pieces_{0};
  std::atomic<size_t> registered_bytes_{0};
  std::thread worker_;
};

}  // namespace mori::umbp
