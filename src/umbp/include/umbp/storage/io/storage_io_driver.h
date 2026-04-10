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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/storage/io/status.h"

namespace mori::umbp {

struct IoWriteOp {
  int fd = -1;
  const void* data = nullptr;
  size_t size = 0;
  uint64_t offset = 0;
};

struct IoReadOp {
  int fd = -1;
  void* data = nullptr;
  size_t size = 0;
  uint64_t offset = 0;
};

struct IoCapabilities {
  bool thread_safe = true;
  bool batch_write = false;
  bool batch_read = false;
  bool native_async = false;
};

class StorageIoDriver {
 public:
  virtual ~StorageIoDriver() = default;

  virtual IoStatus WriteAt(int fd, const void* data, size_t size, uint64_t offset) = 0;
  virtual IoStatus ReadAt(int fd, void* data, size_t size, uint64_t offset) = 0;
  virtual IoStatus Sync(int fd) = 0;
  virtual IoCapabilities Capabilities() const = 0;

  virtual IoStatus WriteBatch(const std::vector<IoWriteOp>& ops);
  virtual IoStatus ReadBatch(const std::vector<IoReadOp>& ops);
  virtual IoStatus SyncMany(const std::vector<int>& fds);
};

std::unique_ptr<StorageIoDriver> CreateStorageIoDriver(UMBPIoBackend backend, uint32_t queue_depth);

}  // namespace mori::umbp
