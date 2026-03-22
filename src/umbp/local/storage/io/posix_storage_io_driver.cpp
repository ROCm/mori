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
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "storage_io_driver_impl.h"

namespace mori::umbp {

namespace {

IoStatus ErrnoIoStatus(const std::string& message) {
  return IoStatus::IoError(message + ": " + std::string(strerror(errno)), errno);
}

class PosixStorageIoDriver final : public StorageIoDriver {
 public:
  IoStatus WriteAt(int fd, const void* data, size_t size, uint64_t offset) override {
    size_t written = 0;
    const char* ptr = static_cast<const char*>(data);
    while (written < size) {
      ssize_t n = pwrite(fd, ptr + written, size - written, static_cast<off_t>(offset + written));
      if (n < 0) return ErrnoIoStatus("pwrite failed");
      if (n == 0) return IoStatus::ShortWrite("pwrite returned 0");
      written += static_cast<size_t>(n);
    }
    return IoStatus::Ok();
  }

  IoStatus ReadAt(int fd, void* data, size_t size, uint64_t offset) override {
    size_t total = 0;
    char* ptr = static_cast<char*>(data);
    while (total < size) {
      ssize_t n = pread(fd, ptr + total, size - total, static_cast<off_t>(offset + total));
      if (n < 0) return ErrnoIoStatus("pread failed");
      if (n == 0) return IoStatus::ShortRead("pread returned EOF");
      total += static_cast<size_t>(n);
    }
    return IoStatus::Ok();
  }

  IoStatus Sync(int fd) override {
    if (fdatasync(fd) != 0) return ErrnoIoStatus("fdatasync failed");
    return IoStatus::Ok();
  }

  IoCapabilities Capabilities() const override {
    IoCapabilities caps;
    caps.thread_safe = true;
    return caps;
  }
};

}  // namespace

std::unique_ptr<StorageIoDriver> CreatePosixStorageIoDriver() {
  return std::make_unique<PosixStorageIoDriver>();
}

}  // namespace mori::umbp
