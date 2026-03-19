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
#include "umbp/storage/io_backend.h"

#include <errno.h>
#include <linux/io_uring.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

bool PosixIoBackend::PWriteAll(int fd, const void* data, size_t size, uint64_t offset) {
  size_t written = 0;
  const char* ptr = static_cast<const char*>(data);
  while (written < size) {
    ssize_t n = pwrite(fd, ptr + written, size - written, static_cast<off_t>(offset + written));
    if (n <= 0) return false;
    written += static_cast<size_t>(n);
  }
  return true;
}

bool PosixIoBackend::PReadAll(int fd, void* data, size_t size, uint64_t offset) {
  size_t total = 0;
  char* ptr = static_cast<char*>(data);
  while (total < size) {
    ssize_t n = pread(fd, ptr + total, size - total, static_cast<off_t>(offset + total));
    if (n <= 0) return false;
    total += static_cast<size_t>(n);
  }
  return true;
}

bool PosixIoBackend::Sync(int fd) { return fdatasync(fd) == 0; }

namespace {
int IoUringSetup(uint32_t entries, struct io_uring_params* p) {
  return static_cast<int>(syscall(__NR_io_uring_setup, entries, p));
}

int IoUringEnter(int fd, uint32_t to_submit, uint32_t min_complete, uint32_t flags) {
  return static_cast<int>(
      syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, sizeof(sigset_t)));
}
}  // namespace

IoUringBackend::IoUringBackend(uint32_t queue_depth) {
  if (queue_depth == 0) queue_depth = 128;
  struct io_uring_params p{};
  ring_fd_ = IoUringSetup(queue_depth, &p);
  if (ring_fd_ < 0) {
    ready_ = false;
    return;
  }

  single_mmap_ = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
  sq_ring_entries_ = p.sq_entries;
  cq_ring_entries_ = p.cq_entries;

  sq_ring_sz_ = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
  cq_ring_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
  if (single_mmap_) {
    if (cq_ring_sz_ > sq_ring_sz_) sq_ring_sz_ = cq_ring_sz_;
    cq_ring_sz_ = sq_ring_sz_;
  }
  sqes_sz_ = p.sq_entries * sizeof(struct io_uring_sqe);

  sq_ring_ptr_ = mmap(nullptr, sq_ring_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      ring_fd_, IORING_OFF_SQ_RING);
  if (sq_ring_ptr_ == MAP_FAILED) {
    sq_ring_ptr_ = nullptr;
    close(ring_fd_);
    ring_fd_ = -1;
    return;
  }

  if (single_mmap_) {
    cq_ring_ptr_ = sq_ring_ptr_;
  } else {
    cq_ring_ptr_ = mmap(nullptr, cq_ring_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                        ring_fd_, IORING_OFF_CQ_RING);
    if (cq_ring_ptr_ == MAP_FAILED) {
      cq_ring_ptr_ = nullptr;
      munmap(sq_ring_ptr_, sq_ring_sz_);
      sq_ring_ptr_ = nullptr;
      close(ring_fd_);
      ring_fd_ = -1;
      return;
    }
  }

  sqes_ptr_ = mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_,
                   IORING_OFF_SQES);
  if (sqes_ptr_ == MAP_FAILED) {
    sqes_ptr_ = nullptr;
    if (!single_mmap_ && cq_ring_ptr_) munmap(cq_ring_ptr_, cq_ring_sz_);
    if (sq_ring_ptr_) munmap(sq_ring_ptr_, sq_ring_sz_);
    sq_ring_ptr_ = nullptr;
    cq_ring_ptr_ = nullptr;
    close(ring_fd_);
    ring_fd_ = -1;
    return;
  }

  sq_head_ = reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.head);
  sq_tail_ = reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.tail);
  sq_ring_mask_ =
      reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.ring_mask);
  sq_ring_entries_ptr_ =
      reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.ring_entries);
  sq_flags_ = reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.flags);
  sq_dropped_ = reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.dropped);
  sq_array_ = reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.array);

  cq_head_ = reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.head);
  cq_tail_ = reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.tail);
  cq_ring_mask_ =
      reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.ring_mask);
  cq_ring_entries_ptr_ =
      reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.ring_entries);
  cq_overflow_ = reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.overflow);

  sqes_ = reinterpret_cast<struct io_uring_sqe*>(sqes_ptr_);
  cqes_ = reinterpret_cast<struct io_uring_cqe*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.cqes);
  ready_ = true;
}

IoUringBackend::~IoUringBackend() {
  if (sqes_ptr_) munmap(sqes_ptr_, sqes_sz_);
  if (!single_mmap_ && cq_ring_ptr_) munmap(cq_ring_ptr_, cq_ring_sz_);
  if (sq_ring_ptr_) munmap(sq_ring_ptr_, sq_ring_sz_);
  if (ring_fd_ >= 0) close(ring_fd_);
}

bool IoUringBackend::SubmitAndWait(uint32_t to_submit, uint32_t min_complete) {
  if (!ready_) return false;
  uint32_t flags = (min_complete > 0) ? IORING_ENTER_GETEVENTS : 0;
  int ret = IoUringEnter(ring_fd_, to_submit, min_complete, flags);
  return ret >= 0;
}

bool IoUringBackend::SubmitRwChunked(uint8_t opcode, int fd, void* base, size_t size,
                                     uint64_t offset) {
  if (!ready_) return false;
  constexpr size_t kChunkBytes = 1 << 20;  // 1 MiB per SQE

  const uint32_t ring_cap = *sq_ring_entries_ptr_;
  const uint32_t max_inflight = (ring_cap > 1) ? (ring_cap - 1) : 1;
  const size_t total_ops = (size + kChunkBytes - 1) / kChunkBytes;

  std::vector<struct iovec> iovs(total_ops);
  std::vector<size_t> expected(total_ops);
  for (size_t i = 0; i < total_ops; ++i) {
    size_t chunk_off = i * kChunkBytes;
    size_t chunk_sz = std::min(kChunkBytes, size - chunk_off);
    iovs[i].iov_base = static_cast<char*>(base) + chunk_off;
    iovs[i].iov_len = chunk_sz;
    expected[i] = chunk_sz;
  }

  size_t submit_idx = 0;
  size_t complete_cnt = 0;
  size_t inflight = 0;

  while (complete_cnt < total_ops) {
    uint32_t tail = *sq_tail_;
    uint32_t head = *sq_head_;
    uint32_t avail = ring_cap - (tail - head);
    size_t remain_cap = static_cast<size_t>(avail);
    size_t inflight_cap = static_cast<size_t>(max_inflight) - inflight;
    size_t remaining_ops = total_ops - submit_idx;
    uint32_t can_submit =
        static_cast<uint32_t>(std::min(remain_cap, std::min(inflight_cap, remaining_ops)));

    for (uint32_t s = 0; s < can_submit; ++s) {
      uint32_t idx = tail & *sq_ring_mask_;
      struct io_uring_sqe* sqe = &sqes_[idx];
      std::memset(sqe, 0, sizeof(*sqe));
      sqe->opcode = opcode;
      sqe->fd = fd;
      sqe->off = offset + (submit_idx * kChunkBytes);
      sqe->addr = reinterpret_cast<uint64_t>(&iovs[submit_idx]);
      sqe->len = 1;
      sqe->user_data = static_cast<uint64_t>(submit_idx + 1);
      sq_array_[idx] = idx;
      ++tail;
      ++submit_idx;
    }

    if (can_submit > 0) {
      std::atomic_thread_fence(std::memory_order_release);
      *sq_tail_ = tail;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      inflight += can_submit;
      if (!SubmitAndWait(can_submit, 0)) return false;
    }

    if (inflight > 0) {
      if (!SubmitAndWait(0, 1)) return false;
    }

    uint32_t cq_h = *cq_head_;
    uint32_t cq_t = *cq_tail_;
    while (cq_h != cq_t) {
      uint32_t cidx = cq_h & *cq_ring_mask_;
      struct io_uring_cqe* cqe = &cqes_[cidx];
      if (cqe->user_data == 0) return false;
      size_t op_idx = static_cast<size_t>(cqe->user_data - 1);
      if (op_idx >= total_ops) return false;
      if (cqe->res < 0) return false;
      if (static_cast<size_t>(cqe->res) != expected[op_idx]) return false;
      ++complete_cnt;
      --inflight;
      ++cq_h;
    }
    *cq_head_ = cq_h;
  }
  return true;
}

bool IoUringBackend::SubmitFsync(int fd, uint64_t user_data, int* out_res) {
  if (!ready_) return false;
  uint32_t tail = *sq_tail_;
  uint32_t head = *sq_head_;
  if (tail - head >= *sq_ring_entries_ptr_) return false;

  uint32_t idx = tail & *sq_ring_mask_;
  struct io_uring_sqe* sqe = &sqes_[idx];
  std::memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IORING_OP_FSYNC;
  sqe->fd = fd;
  sqe->user_data = user_data;
  sq_array_[idx] = idx;

  std::atomic_thread_fence(std::memory_order_release);
  *sq_tail_ = tail + 1;
  std::atomic_thread_fence(std::memory_order_seq_cst);

  if (!SubmitAndWait(1, 1)) return false;

  uint32_t cq_head = *cq_head_;
  if (cq_head == *cq_tail_) return false;
  uint32_t cidx = cq_head & *cq_ring_mask_;
  struct io_uring_cqe* cqe = &cqes_[cidx];
  if (cqe->user_data != user_data) return false;
  if (out_res) *out_res = cqe->res;
  *cq_head_ = cq_head + 1;
  return cqe->res >= 0;
}

bool IoUringBackend::PWriteAll(int fd, const void* data, size_t size, uint64_t offset) {
  return SubmitRwChunked(IORING_OP_WRITEV, fd, const_cast<void*>(data), size, offset);
}

bool IoUringBackend::PReadAll(int fd, void* data, size_t size, uint64_t offset) {
  return SubmitRwChunked(IORING_OP_READV, fd, data, size, offset);
}

bool IoUringBackend::Sync(int fd) {
  int res = 0;
  return SubmitFsync(fd, 0xFFFFFFFFFFFFFFFFULL, &res) && res == 0;
}

// Default PWriteBatch: loop over PWriteAll.
bool IoBackend::PWriteBatch(const std::vector<IoOp>& ops) {
  for (const auto& op : ops) {
    if (!PWriteAll(op.fd, op.data, op.size, op.offset)) return false;
  }
  return true;
}

bool IoUringBackend::PWriteBatch(const std::vector<IoOp>& ops) {
  if (!ready_ || ops.empty()) return ops.empty();

  // For each op, prepare iovec(s) and submit all SQEs in one batch.
  constexpr size_t kChunkBytes = 1 << 20;  // 1 MiB per SQE, matching SubmitRwChunked

  // Build flat list of SQE descriptors.
  struct SqeDesc {
    int fd;
    uint64_t file_offset;
    struct iovec iov;
  };
  std::vector<SqeDesc> descs;
  descs.reserve(ops.size());  // optimistic: 1 SQE per op

  for (const auto& op : ops) {
    size_t remaining = op.size;
    size_t buf_offset = 0;
    while (remaining > 0) {
      size_t chunk = std::min(kChunkBytes, remaining);
      SqeDesc d;
      d.fd = op.fd;
      d.file_offset = op.offset + buf_offset;
      d.iov.iov_base = const_cast<void*>(
          static_cast<const void*>(static_cast<const char*>(op.data) + buf_offset));
      d.iov.iov_len = chunk;
      descs.push_back(d);
      buf_offset += chunk;
      remaining -= chunk;
    }
  }

  const size_t total_sqes = descs.size();
  const uint32_t ring_cap = *sq_ring_entries_ptr_;
  const uint32_t max_inflight = (ring_cap > 1) ? (ring_cap - 1) : 1;

  size_t submit_idx = 0;
  size_t complete_cnt = 0;
  size_t inflight = 0;

  while (complete_cnt < total_sqes) {
    uint32_t tail = *sq_tail_;
    uint32_t head = *sq_head_;
    uint32_t avail = ring_cap - (tail - head);
    size_t remain_cap = static_cast<size_t>(avail);
    size_t inflight_cap = static_cast<size_t>(max_inflight) - inflight;
    size_t remaining_ops = total_sqes - submit_idx;
    uint32_t can_submit =
        static_cast<uint32_t>(std::min(remain_cap, std::min(inflight_cap, remaining_ops)));

    for (uint32_t s = 0; s < can_submit; ++s) {
      uint32_t idx = tail & *sq_ring_mask_;
      struct io_uring_sqe* sqe = &sqes_[idx];
      std::memset(sqe, 0, sizeof(*sqe));
      sqe->opcode = IORING_OP_WRITEV;
      sqe->fd = descs[submit_idx].fd;
      sqe->off = descs[submit_idx].file_offset;
      sqe->addr = reinterpret_cast<uint64_t>(&descs[submit_idx].iov);
      sqe->len = 1;
      sqe->user_data = static_cast<uint64_t>(submit_idx + 1);
      sq_array_[idx] = idx;
      ++tail;
      ++submit_idx;
    }

    if (can_submit > 0) {
      std::atomic_thread_fence(std::memory_order_release);
      *sq_tail_ = tail;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      inflight += can_submit;
      if (!SubmitAndWait(can_submit, 0)) return false;
    }

    if (inflight > 0) {
      if (!SubmitAndWait(0, 1)) return false;
    }

    uint32_t cq_h = *cq_head_;
    uint32_t cq_t = *cq_tail_;
    while (cq_h != cq_t) {
      uint32_t cidx = cq_h & *cq_ring_mask_;
      struct io_uring_cqe* cqe = &cqes_[cidx];
      if (cqe->user_data == 0) return false;
      size_t op_idx = static_cast<size_t>(cqe->user_data - 1);
      if (op_idx >= total_sqes) return false;
      if (cqe->res < 0) return false;
      if (static_cast<size_t>(cqe->res) != descs[op_idx].iov.iov_len) return false;
      ++complete_cnt;
      --inflight;
      ++cq_h;
    }
    *cq_head_ = cq_h;
  }
  return true;
}

std::unique_ptr<IoBackend> CreateIoBackend(UMBPIoBackend backend, uint32_t queue_depth) {
  if (backend == UMBPIoBackend::IoUring) {
    auto uring = std::make_unique<IoUringBackend>(queue_depth);
    if (uring->IsReady()) return uring;
  }
  return std::make_unique<PosixIoBackend>();
}

std::string BuildSegmentFileName(uint64_t segment_id) {
  return "segment_" + std::to_string(segment_id) + ".log";
}
