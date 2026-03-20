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
#include <linux/io_uring.h>
#include <signal.h>
#include <string.h>
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

#include "storage_io_driver_impl.h"

namespace {

class IoUringStorageIoDriver final : public StorageIoDriver {
 public:
  explicit IoUringStorageIoDriver(uint32_t queue_depth) {
    if (queue_depth == 0) queue_depth = 128;
    struct io_uring_params p{};
    ring_fd_ = static_cast<int>(syscall(__NR_io_uring_setup, entriesOrDefault(queue_depth), &p));
    if (ring_fd_ < 0) return;

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
    sq_array_ = reinterpret_cast<uint32_t*>(static_cast<char*>(sq_ring_ptr_) + p.sq_off.array);

    cq_head_ = reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.head);
    cq_tail_ = reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.tail);
    cq_ring_mask_ =
        reinterpret_cast<uint32_t*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.ring_mask);

    sqes_ = reinterpret_cast<struct io_uring_sqe*>(sqes_ptr_);
    cqes_ =
        reinterpret_cast<struct io_uring_cqe*>(static_cast<char*>(cq_ring_ptr_) + p.cq_off.cqes);
    ready_ = true;
  }

  ~IoUringStorageIoDriver() override {
    if (sqes_ptr_) munmap(sqes_ptr_, sqes_sz_);
    if (!single_mmap_ && cq_ring_ptr_) munmap(cq_ring_ptr_, cq_ring_sz_);
    if (sq_ring_ptr_) munmap(sq_ring_ptr_, sq_ring_sz_);
    if (ring_fd_ >= 0) close(ring_fd_);
  }

  IoStatus WriteAt(int fd, const void* data, size_t size, uint64_t offset) override {
    return SubmitRwChunked(IORING_OP_WRITEV, fd, const_cast<void*>(data), size, offset);
  }

  IoStatus ReadAt(int fd, void* data, size_t size, uint64_t offset) override {
    return SubmitRwChunked(IORING_OP_READV, fd, data, size, offset);
  }

  IoStatus Sync(int fd) override {
    int res = 0;
    IoStatus status = SubmitFsync(fd, 0xFFFFFFFFFFFFFFFFULL, &res);
    if (!status.ok()) return status;
    if (res < 0) return IoStatus::IoError("io_uring fsync failed", -res);
    return IoStatus::Ok();
  }

  IoStatus WriteBatch(const std::vector<IoWriteOp>& ops) override {
    if (!ready_ || ops.empty()) {
      return ops.empty() ? IoStatus::Ok() : IoStatus::Unavailable("io_uring unavailable");
    }

    constexpr size_t kChunkBytes = 1 << 20;
    struct SqeDesc {
      int fd;
      uint64_t file_offset;
      struct iovec iov;
    };

    std::vector<SqeDesc> descs;
    descs.reserve(ops.size());
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
        IoStatus status = SubmitAndWait(can_submit, 0);
        if (!status.ok()) return status;
      }

      if (inflight > 0) {
        IoStatus status = SubmitAndWait(0, 1);
        if (!status.ok()) return status;
      }

      uint32_t cq_h = *cq_head_;
      uint32_t cq_t = *cq_tail_;
      while (cq_h != cq_t) {
        uint32_t cidx = cq_h & *cq_ring_mask_;
        struct io_uring_cqe* cqe = &cqes_[cidx];
        if (cqe->user_data == 0) return IoStatus::Corruption("io_uring cqe user_data missing");
        size_t op_idx = static_cast<size_t>(cqe->user_data - 1);
        if (op_idx >= total_sqes)
          return IoStatus::Corruption("io_uring cqe user_data out of range");
        if (cqe->res < 0) return IoStatus::IoError("io_uring batch write failed", -cqe->res);
        if (static_cast<size_t>(cqe->res) != descs[op_idx].iov.iov_len) {
          return IoStatus::ShortWrite("io_uring batch write completed partially");
        }
        ++complete_cnt;
        --inflight;
        ++cq_h;
      }
      *cq_head_ = cq_h;
    }
    return IoStatus::Ok();
  }

  IoStatus ReadBatch(const std::vector<IoReadOp>& ops) override {
    if (!ready_ || ops.empty()) {
      return ops.empty() ? IoStatus::Ok() : IoStatus::Unavailable("io_uring unavailable");
    }

    constexpr size_t kChunkBytes = 1 << 20;
    struct SqeDesc {
      int fd;
      uint64_t file_offset;
      struct iovec iov;
    };

    std::vector<SqeDesc> descs;
    descs.reserve(ops.size());
    for (const auto& op : ops) {
      size_t remaining = op.size;
      size_t buf_offset = 0;
      while (remaining > 0) {
        size_t chunk = std::min(kChunkBytes, remaining);
        SqeDesc d;
        d.fd = op.fd;
        d.file_offset = op.offset + buf_offset;
        d.iov.iov_base = static_cast<char*>(op.data) + buf_offset;
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
        sqe->opcode = IORING_OP_READV;
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
        IoStatus status = SubmitAndWait(can_submit, 0);
        if (!status.ok()) return status;
      }

      if (inflight > 0) {
        IoStatus status = SubmitAndWait(0, 1);
        if (!status.ok()) return status;
      }

      uint32_t cq_h = *cq_head_;
      uint32_t cq_t = *cq_tail_;
      while (cq_h != cq_t) {
        uint32_t cidx = cq_h & *cq_ring_mask_;
        struct io_uring_cqe* cqe = &cqes_[cidx];
        if (cqe->user_data == 0) return IoStatus::Corruption("io_uring cqe user_data missing");
        size_t op_idx = static_cast<size_t>(cqe->user_data - 1);
        if (op_idx >= total_sqes)
          return IoStatus::Corruption("io_uring cqe user_data out of range");
        if (cqe->res < 0) return IoStatus::IoError("io_uring batch read failed", -cqe->res);
        if (static_cast<size_t>(cqe->res) != descs[op_idx].iov.iov_len) {
          return IoStatus::ShortRead("io_uring batch read completed partially");
        }
        ++complete_cnt;
        --inflight;
        ++cq_h;
      }
      *cq_head_ = cq_h;
    }
    return IoStatus::Ok();
  }

  IoCapabilities Capabilities() const override {
    IoCapabilities caps;
    caps.thread_safe = false;
    caps.batch_write = true;
    caps.batch_read = true;
    caps.native_async = ready_;
    return caps;
  }

 private:
  static uint32_t entriesOrDefault(uint32_t queue_depth) {
    return queue_depth == 0 ? 128 : queue_depth;
  }

  IoStatus SubmitAndWait(uint32_t to_submit, uint32_t min_complete) {
    if (!ready_) return IoStatus::Unavailable("io_uring driver is not ready");
    uint32_t flags = (min_complete > 0) ? IORING_ENTER_GETEVENTS : 0;
    int ret = static_cast<int>(syscall(__NR_io_uring_enter, ring_fd_, to_submit, min_complete,
                                       flags, nullptr, sizeof(sigset_t)));
    if (ret < 0) return IoStatus::IoError("io_uring_enter failed", errno);
    return IoStatus::Ok();
  }

  IoStatus SubmitRwChunked(uint8_t opcode, int fd, void* base, size_t size, uint64_t offset) {
    if (!ready_) return IoStatus::Unavailable("io_uring driver is not ready");
    constexpr size_t kChunkBytes = 1 << 20;

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
        IoStatus status = SubmitAndWait(can_submit, 0);
        if (!status.ok()) return status;
      }

      if (inflight > 0) {
        IoStatus status = SubmitAndWait(0, 1);
        if (!status.ok()) return status;
      }

      uint32_t cq_h = *cq_head_;
      uint32_t cq_t = *cq_tail_;
      while (cq_h != cq_t) {
        uint32_t cidx = cq_h & *cq_ring_mask_;
        struct io_uring_cqe* cqe = &cqes_[cidx];
        if (cqe->user_data == 0) return IoStatus::Corruption("io_uring cqe user_data missing");
        size_t op_idx = static_cast<size_t>(cqe->user_data - 1);
        if (op_idx >= total_ops) return IoStatus::Corruption("io_uring cqe user_data out of range");
        if (cqe->res < 0) return IoStatus::IoError("io_uring read/write failed", -cqe->res);
        if (static_cast<size_t>(cqe->res) != expected[op_idx]) {
          return (opcode == IORING_OP_READV)
                     ? IoStatus::ShortRead("io_uring read completed partially")
                     : IoStatus::ShortWrite("io_uring write completed partially");
        }
        ++complete_cnt;
        --inflight;
        ++cq_h;
      }
      *cq_head_ = cq_h;
    }
    return IoStatus::Ok();
  }

  IoStatus SubmitFsync(int fd, uint64_t user_data, int* out_res) {
    if (!ready_) return IoStatus::Unavailable("io_uring driver is not ready");
    uint32_t tail = *sq_tail_;
    uint32_t head = *sq_head_;
    if (tail - head >= *sq_ring_entries_ptr_) return IoStatus::Unavailable("io_uring SQ ring full");

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

    IoStatus status = SubmitAndWait(1, 1);
    if (!status.ok()) return status;

    uint32_t cq_head = *cq_head_;
    if (cq_head == *cq_tail_) return IoStatus::Unavailable("io_uring fsync completion missing");
    uint32_t cidx = cq_head & *cq_ring_mask_;
    struct io_uring_cqe* cqe = &cqes_[cidx];
    if (cqe->user_data != user_data)
      return IoStatus::Corruption("io_uring fsync completion mismatch");
    if (out_res) *out_res = cqe->res;
    *cq_head_ = cq_head + 1;
    return IoStatus::Ok();
  }

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
  uint32_t* sq_array_ = nullptr;

  uint32_t* cq_head_ = nullptr;
  uint32_t* cq_tail_ = nullptr;
  uint32_t* cq_ring_mask_ = nullptr;

  struct io_uring_sqe* sqes_ = nullptr;
  struct io_uring_cqe* cqes_ = nullptr;
};

}  // namespace

std::unique_ptr<StorageIoDriver> CreateIoUringStorageIoDriver(uint32_t queue_depth) {
  return std::make_unique<IoUringStorageIoDriver>(queue_depth);
}
