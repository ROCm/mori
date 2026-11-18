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

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"

namespace mori::umbp {

// GPU HBM, allocated with hipMalloc on a fixed device.
//
// THE WHOLE HBM BACKEND IS THIS CLASS.  There is no HbmBackend type, and that
// is the finding rather than a shortcut: PageBackend's slot lifecycle, bitmap
// allocator, event outbox, reaper, read leases and copy pins never consult the
// tier, so a second paged medium needs no second copy of any of it.  What HBM
// actually changes is three facts — hipMalloc instead of mmap, loc=GPU instead
// of CPU, and a real device ordinal instead of -1 — which is exactly the
// surface PageMemorySource exposes.
//
// The device is fixed at construction rather than read from the ambient current
// device at Allocate time.  A backend is called from the peer service's gRPC
// handler threads and the heartbeat thread (MediumBackend's threading note), and
// none of those has a meaningful current device — inheriting one would make the
// pool land on whichever thread happened to allocate first.
class HbmPageMemorySource final : public PageMemorySource {
 public:
  // How the pool is allocated. kHip is a plain hipMalloc. The two fabric
  // modes allocate through mori-io's fabric allocator, so the pool can be
  // exported over a UALink scale-up fabric and read by a GPU on another tray;
  // kFabricOrHip falls back to hipMalloc on a machine without one.
  enum class Allocation { kHip, kFabric, kFabricOrHip };

  explicit HbmPageMemorySource(int device, Allocation allocation = Allocation::kHip)
      : device_(device), allocation_(allocation) {}
  ~HbmPageMemorySource() override { Release(); }

  // One buffer per size, on device_.  All-or-nothing.
  bool Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) override;
  void Release() override;

  mori::io::MemoryLocationType LocationType() const override {
    return mori::io::MemoryLocationType::GPU;
  }
  int Device() const override { return device_; }
  const char* Name() const override { return "HbmPageMemorySource"; }

 private:
  struct Owned {
    void* ptr = nullptr;
    bool fabric = false;
  };
  int device_ = 0;
  Allocation allocation_ = Allocation::kHip;
  std::vector<Owned> ptrs_;
};

// The one place a caller outside this file obtains an HBM backend — the
// counterpart of MakePageBackend, and subject to the same Phase 5 Rule A: it
// returns MediumBackend, so PoolClient::Init names no concrete backend type.
//
// `device` is the GPU ordinal the pool is allocated on.  `buffer_sizes` is one
// hipMalloc per entry, in bytes.
std::unique_ptr<MediumBackend> MakeHbmBackend(
    uint64_t page_size, int device, std::vector<uint64_t> buffer_sizes,
    std::chrono::milliseconds pending_ttl, std::chrono::milliseconds read_lease_ttl,
    HbmPageMemorySource::Allocation allocation = HbmPageMemorySource::Allocation::kHip);

}  // namespace mori::umbp
