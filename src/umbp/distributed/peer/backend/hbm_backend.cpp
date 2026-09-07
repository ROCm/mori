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
#include "umbp/distributed/peer/backend/hbm_backend.h"

#include <hip/hip_runtime.h>

#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// Set the current device for the duration of a scope and put it back.  Peer
// service threads do not own a current device (see the header), so every HIP
// call that depends on one has to establish it explicitly and leave no trace.
class ScopedDevice {
 public:
  explicit ScopedDevice(int device) {
    if (hipGetDevice(&previous_) != hipSuccess) {
      previous_ = -1;
      return;
    }
    if (device == previous_) return;  // nothing to restore
    if (hipSetDevice(device) != hipSuccess) {
      previous_ = -1;
      return;
    }
    restore_ = true;
  }
  ~ScopedDevice() {
    if (restore_ && previous_ >= 0) (void)hipSetDevice(previous_);
  }

  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int previous_ = -1;
  bool restore_ = false;
};

}  // namespace

bool HbmPageMemorySource::Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) {
  ScopedDevice scope(device_);

  std::vector<void*> taken;
  std::vector<Buffer> staged;
  for (uint64_t size : sizes) {
    if (size == 0) continue;
    void* ptr = nullptr;
    const hipError_t err = hipMalloc(&ptr, static_cast<size_t>(size));
    if (err != hipSuccess || ptr == nullptr) {
      MORI_UMBP_ERROR("[HbmPageMemorySource] hipMalloc({}) on device {} failed: {}", size, device_,
                      hipGetErrorString(err));
      // All-or-nothing, matching the PageMemorySource contract: unwind only
      // what this call took, leaving `out` and any earlier pool untouched.
      for (void* p : taken) (void)hipFree(p);
      return false;
    }
    // hipMalloc returns exactly the requested size — no rounding to report, in
    // contrast to the host source's hugepage mapped_size.
    staged.push_back(Buffer{ptr, size});
    taken.push_back(ptr);
  }

  ptrs_.insert(ptrs_.end(), taken.begin(), taken.end());
  out->insert(out->end(), staged.begin(), staged.end());
  MORI_UMBP_INFO("[HbmPageMemorySource] allocated {} buffers on device {}", staged.size(), device_);
  return true;
}

void HbmPageMemorySource::Release() {
  if (ptrs_.empty()) return;
  ScopedDevice scope(device_);
  for (void* p : ptrs_) {
    const hipError_t err = hipFree(p);
    if (err != hipSuccess) {
      MORI_UMBP_ERROR("[HbmPageMemorySource] hipFree on device {} failed: {}", device_,
                      hipGetErrorString(err));
    }
  }
  ptrs_.clear();
}

std::unique_ptr<MediumBackend> MakeHbmBackend(uint64_t page_size, int device,
                                              std::vector<uint64_t> buffer_sizes,
                                              std::chrono::milliseconds pending_ttl,
                                              std::chrono::milliseconds read_lease_ttl) {
  return std::make_unique<PageBackend>(TierType::HBM, page_size,
                                       std::make_unique<HbmPageMemorySource>(device),
                                       std::move(buffer_sizes), pending_ttl, read_lease_ttl);
}

}  // namespace mori::umbp
