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

namespace mori::umbp {

// Makes a host tier addressable from GPU kernels via one whole-region
// hipHostRegister call. Registration is best-effort: Covers() remains false
// until it succeeds, allowing callers to fall back to the copy engine.
class HostTierRegistration {
 public:
  HostTierRegistration(void* base, size_t bytes);
  ~HostTierRegistration();

  HostTierRegistration(const HostTierRegistration&) = delete;
  HostTierRegistration& operator=(const HostTierRegistration&) = delete;

  bool Covers(const void* ptr, size_t size) const;
  size_t RegisteredBytes() const { return registered_bytes_.load(std::memory_order_acquire); }

 private:
  void Register();

  char* base_{nullptr};
  size_t bytes_{0};
  std::atomic<size_t> registered_bytes_{0};
  std::thread worker_;
};

}  // namespace mori::umbp
