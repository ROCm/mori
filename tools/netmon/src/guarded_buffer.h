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
// A writable mapping followed by a PROT_NONE guard page.
//
// Used for ETHTOOL_GSTATS / ETHTOOL_GSTRINGS: the kernel writes as many entries as the driver
// currently reports and ignores the count passed in from user space, so a fixed-size heap
// buffer cannot be proven safe. With a guard page, an oversized kernel copy stops at the guard
// (copy_to_user returns a short count and the ioctl fails with EFAULT) instead of corrupting
// neighbouring memory.
#pragma once

#include <cstddef>
#include <cstdint>

#include "util.h"

namespace netmon {

class GuardedBuffer {
 public:
  GuardedBuffer() = default;
  ~GuardedBuffer();
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;

  // Map at least `bytes` writable bytes (rounded up to pages) plus one guard page.
  Status Allocate(size_t bytes);
  uint8_t* data() const { return data_; }
  size_t capacity() const { return capacity_; }  // writable bytes before the guard page

 private:
  uint8_t* base_ = nullptr;
  uint8_t* data_ = nullptr;
  size_t capacity_ = 0;
  size_t mapped_ = 0;
};

}  // namespace netmon
