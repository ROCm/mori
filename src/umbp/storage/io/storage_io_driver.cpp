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
#include "umbp/storage/io/storage_io_driver.h"

#include <memory>
#include <unordered_set>

#include "storage_io_driver_impl.h"

IoStatus StorageIoDriver::WriteBatch(const std::vector<IoWriteOp>& ops) {
  for (const auto& op : ops) {
    IoStatus status = WriteAt(op.fd, op.data, op.size, op.offset);
    if (!status.ok()) return status;
  }
  return IoStatus::Ok();
}

IoStatus StorageIoDriver::ReadBatch(const std::vector<IoReadOp>& ops) {
  for (const auto& op : ops) {
    IoStatus status = ReadAt(op.fd, op.data, op.size, op.offset);
    if (!status.ok()) return status;
  }
  return IoStatus::Ok();
}

IoStatus StorageIoDriver::SyncMany(const std::vector<int>& fds) {
  std::unordered_set<int> unique_fds(fds.begin(), fds.end());
  for (int fd : unique_fds) {
    IoStatus status = Sync(fd);
    if (!status.ok()) return status;
  }
  return IoStatus::Ok();
}

std::unique_ptr<StorageIoDriver> CreateStorageIoDriver(UMBPIoBackend backend,
                                                       uint32_t queue_depth) {
  if (backend == UMBPIoBackend::IoUring) {
    auto driver = CreateIoUringStorageIoDriver(queue_depth);
    if (driver && driver->Capabilities().native_async) return driver;
  }
  return CreatePosixStorageIoDriver();
}
