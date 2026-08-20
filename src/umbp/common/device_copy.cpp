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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/common/device_copy.h"

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

PointerLocation DetectPointerLocation(const void* ptr) {
  if (ptr == nullptr) return {};

  hipPointerAttribute_t attributes{};
  const hipError_t status = hipPointerGetAttributes(&attributes, ptr);
  if (status == hipSuccess) {
    if (attributes.type == hipMemoryTypeDevice && attributes.device >= 0) {
      return {true, attributes.device};
    }
    return {};
  }

  // An attribute miss is normal for malloc/mmap memory. Clear HIP's
  // thread-local error so it cannot affect a later runtime call.
  (void)hipGetLastError();
  return {};
}

ScopedHipDevice::ScopedHipDevice(int device_id) {
  if (device_id < 0) return;
  if (hipGetDevice(&previous_device_) != hipSuccess) {
    (void)hipGetLastError();
    return;
  }
  if (previous_device_ != device_id) {
    if (hipSetDevice(device_id) != hipSuccess) {
      (void)hipGetLastError();
      return;
    }
    changed_ = true;
  }
  valid_ = true;
}

ScopedHipDevice::~ScopedHipDevice() {
  if (!changed_) return;
  if (hipSetDevice(previous_device_) != hipSuccess) (void)hipGetLastError();
}

bool DeviceCopy(void* dst, const void* src, size_t size, hipMemcpyKind kind, int device_id) {
  const hipError_t status = hipMemcpy(dst, src, size, kind);
  if (status != hipSuccess) {
    MORI_UMBP_ERROR("[UMBP] hipMemcpy failed on device {}: {}", device_id,
                    hipGetErrorString(status));
    return false;
  }
  return true;
}

}  // namespace mori::umbp
