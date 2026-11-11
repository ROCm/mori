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
