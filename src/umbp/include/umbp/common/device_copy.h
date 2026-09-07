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
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>

namespace mori::umbp {

struct PointerLocation {
  bool is_device{false};
  int device_id{-1};

  bool IsDevice() const { return is_device; }
};

// Managed memory is intentionally reported as host memory: it is not a stable
// GPU registration target because its backing pages may migrate.
PointerLocation DetectPointerLocation(const void* ptr);

class ScopedHipDevice {
 public:
  explicit ScopedHipDevice(int device_id);
  ~ScopedHipDevice();

  ScopedHipDevice(const ScopedHipDevice&) = delete;
  ScopedHipDevice& operator=(const ScopedHipDevice&) = delete;

  bool IsValid() const { return valid_; }

 private:
  int previous_device_{-1};
  bool changed_{false};
  bool valid_{false};
};

bool DeviceCopy(void* dst, const void* src, size_t size, hipMemcpyKind kind, int device_id);

}  // namespace mori::umbp
