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

// The size of the pieces a HostTierRegistration registers, and so the
// alignment of the boundaries no single copy of registered memory may cross:
// hipMemcpy rejects a host range spanning two registrations, and a gather
// fragment can use only one piece's device alias. Pieces are cut at absolute
// multiples of this, so a caller needs nothing but the address to split at
// them. UMBP_DRAM_HOST_REGISTER_CHUNK_MB, default 1024; 0 registers each
// region whole.
size_t HostRegistrationChunkBytes();

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
// until then a copy touching it simply takes the path it took before.
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
  // registration is unavailable the object simply reports Covers() == false
  // forever.
  HostTierRegistration(void* base, size_t bytes);
  ~HostTierRegistration();

  HostTierRegistration(const HostTierRegistration&) = delete;
  HostTierRegistration& operator=(const HostTierRegistration&) = delete;

  // True when [ptr, ptr + size) lies inside one registered piece, and is
  // therefore safe to copy from or to without the pageable staging path. False
  // is always the safe answer. Weaker than "safe to hand to a kernel" -- that
  // needs DeviceAddress().
  bool Covers(const void* ptr, size_t size) const;

  // Whether a kernel on `device_id` can reach this region at all, i.e. whether
  // DeviceAddress() will return non-null for a covered pointer.
  bool GatherableOn(int device_id) const;

  // The address `device_id` must use to reach [host_ptr, host_ptr + size), or
  // null when that range is not inside one registered piece or the device has
  // no alias, in which case the caller must stay on the copy-engine path.
  // Subsumes Covers(), so a caller that needs the address should not test both.
  void* DeviceAddress(const void* host_ptr, size_t size, int device_id) const;

  // Bytes registered so far: zero until the first piece is, the full region
  // once every piece is. For logging and tests.
  size_t RegisteredBytes() const { return registered_bytes_.load(std::memory_order_acquire); }

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
  // Pieces [0, registered_pieces_) are registered. A piece, aliases included,
  // is written before the count that covers it is released, and is read-only
  // after, so that release/acquire pair is what publishes it.
  std::atomic<size_t> registered_pieces_{0};
  std::atomic<size_t> registered_bytes_{0};
  std::thread worker_;
};

}  // namespace mori::umbp
