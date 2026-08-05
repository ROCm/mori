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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

#include "umbp/local/tiers/segment/segment_format.h"

namespace mori::umbp {

// Heap buffer whose base address is aligned to segment::kRecordAlign.
//
// O_DIRECT requires the user buffer, the file offset and the length to all be
// alignment multiples; std::vector<char> only guarantees max_align_t (16 B), so
// it cannot back a direct read or write.  Size is rounded up to the alignment as
// well, which is what lets a caller hand the whole buffer to a direct I/O op
// without a separate length check.
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(size_t size) { Resize(size); }

  ~AlignedBuffer() { std::free(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      std::free(data_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  // Grows (never shrinks) the allocation and sets the logical size.  Bytes
  // between `size` and the padded capacity are zeroed so record padding never
  // leaks heap contents to disk.
  void Resize(size_t size) {
    const size_t padded = static_cast<size_t>(segment::AlignUp(size));
    if (padded > capacity_) {
      void* p = nullptr;
      if (posix_memalign(&p, static_cast<size_t>(segment::kRecordAlign), padded) != 0) {
        throw std::bad_alloc();
      }
      std::free(data_);
      data_ = static_cast<char*>(p);
      capacity_ = padded;
    }
    size_ = size;
    if (padded > size_) std::memset(data_ + size_, 0, padded - size_);
  }

  char* data() { return data_; }
  const char* data() const { return data_; }
  size_t size() const { return size_; }
  // Allocation rounded up to kRecordAlign — the length to hand a direct I/O op.
  size_t padded_size() const { return static_cast<size_t>(segment::AlignUp(size_)); }
  bool empty() const { return size_ == 0; }

 private:
  char* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

// True when `ptr` and `size` both satisfy the direct-I/O alignment rules.  File
// offsets are aligned by construction in the v3 layout, so a caller that passes
// this can issue the read straight into the user buffer with no bounce.
inline bool IsDirectIoCompatible(const void* ptr, size_t size) {
  return (reinterpret_cast<uintptr_t>(ptr) % segment::kRecordAlign) == 0 &&
         (size % segment::kRecordAlign) == 0;
}

}  // namespace mori::umbp
