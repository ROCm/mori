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

#include <linux/io_uring.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "umbp/common/config.h"

struct IoOp {
  int fd;
  const void* data;
  size_t size;
  uint64_t offset;
};

class IoBackend {
 public:
  virtual ~IoBackend() = default;

  virtual bool PWriteAll(int fd, const void* data, size_t size, uint64_t offset) = 0;
  virtual bool PReadAll(int fd, void* data, size_t size, uint64_t offset) = 0;
  virtual bool Sync(int fd) = 0;

  // Batch write: submit multiple write operations. Default loops over PWriteAll.
  virtual bool PWriteBatch(const std::vector<IoOp>& ops);
};

// POSIX synchronous backend. This is the default fallback for all platforms.
class PosixIoBackend final : public IoBackend {
 public:
  bool PWriteAll(int fd, const void* data, size_t size, uint64_t offset) override;
  bool PReadAll(int fd, void* data, size_t size, uint64_t offset) override;
  bool Sync(int fd) override;
};

// io_uring backend backed by raw Linux io_uring syscalls (no liburing dependency).
class IoUringBackend final : public IoBackend {
 public:
  explicit IoUringBackend(uint32_t queue_depth);
  ~IoUringBackend() override;

  IoUringBackend(const IoUringBackend&) = delete;
  IoUringBackend& operator=(const IoUringBackend&) = delete;

  bool IsReady() const { return ready_; }

  bool PWriteAll(int fd, const void* data, size_t size, uint64_t offset) override;
  bool PReadAll(int fd, void* data, size_t size, uint64_t offset) override;
  bool Sync(int fd) override;
  bool PWriteBatch(const std::vector<IoOp>& ops) override;

 private:
  bool SubmitRwChunked(uint8_t opcode, int fd, void* base, size_t size, uint64_t offset);
  bool SubmitFsync(int fd, uint64_t user_data, int* out_res);
  bool SubmitAndWait(uint32_t to_submit, uint32_t min_complete);

  int ring_fd_ = -1;
  bool ready_ = false;
  bool single_mmap_ = false;
  uint32_t sq_ring_entries_ = 0;
  uint32_t cq_ring_entries_ = 0;

  size_t sq_ring_sz_ = 0;
  size_t cq_ring_sz_ = 0;
  size_t sqes_sz_ = 0;
  void* sq_ring_ptr_ = nullptr;
  void* cq_ring_ptr_ = nullptr;
  void* sqes_ptr_ = nullptr;

  uint32_t* sq_head_ = nullptr;
  uint32_t* sq_tail_ = nullptr;
  uint32_t* sq_ring_mask_ = nullptr;
  uint32_t* sq_ring_entries_ptr_ = nullptr;
  uint32_t* sq_flags_ = nullptr;
  uint32_t* sq_dropped_ = nullptr;
  uint32_t* sq_array_ = nullptr;

  uint32_t* cq_head_ = nullptr;
  uint32_t* cq_tail_ = nullptr;
  uint32_t* cq_ring_mask_ = nullptr;
  uint32_t* cq_ring_entries_ptr_ = nullptr;
  uint32_t* cq_overflow_ = nullptr;

  struct io_uring_sqe* sqes_ = nullptr;
  struct io_uring_cqe* cqes_ = nullptr;
};

std::unique_ptr<IoBackend> CreateIoBackend(UMBPIoBackend backend, uint32_t queue_depth);
std::string BuildSegmentFileName(uint64_t segment_id);
