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
#include "src/ras/event_queue.hpp"

#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>
#include <new>

#include "src/ras/fd_registry.hpp"

namespace mori {
namespace ras {

// Dmitry Vyukov's bounded MPMC ring, used with one consumer. `seq` says who owns the cell:
// pos means free for the producer claiming pos, pos + 1 means filled for the consumer.
struct EventQueue::Cell {
  std::atomic<size_t> seq{0};
  alignas(RasEvent) unsigned char storage[sizeof(RasEvent)];
  RasEvent* Event() { return std::launder(reinterpret_cast<RasEvent*>(storage)); }
};

namespace {
size_t RoundUpPow2(size_t n) {
  size_t c = 2;
  while (c < n) c <<= 1;
  return c;
}
}  // namespace

EventQueue::EventQueue(size_t capacity) {
  size_t cap = RoundUpPow2(capacity);
  cells_.reset(new Cell[cap]);
  mask_ = cap - 1;
  for (size_t i = 0; i < cap; ++i) cells_[i].seq.store(i, std::memory_order_relaxed);
  wakeFd_ = fdreg::EventFd(0, EFD_NONBLOCK);
}

EventQueue::~EventQueue() {
  RasEvent ev;
  while (TryPop(&ev)) {
  }
  fdreg::Close(wakeFd_);
}

bool EventQueue::TryPush(RasEvent&& ev) {
  size_t pos = enqueuePos_.load(std::memory_order_relaxed);
  for (;;) {
    Cell& cell = cells_[pos & mask_];
    size_t seq = cell.seq.load(std::memory_order_acquire);
    intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
    if (diff == 0) {
      if (enqueuePos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        new (cell.storage) RasEvent(std::move(ev));
        cell.seq.store(pos + 1, std::memory_order_release);
        return true;
      }
    } else if (diff < 0) {
      return false;  // the consumer hasn't freed this cell yet: full
    } else {
      pos = enqueuePos_.load(std::memory_order_relaxed);
    }
  }
}

bool EventQueue::TryPop(RasEvent* out) {
  Cell& cell = cells_[dequeuePos_ & mask_];
  size_t seq = cell.seq.load(std::memory_order_acquire);
  if (static_cast<intptr_t>(seq) - static_cast<intptr_t>(dequeuePos_ + 1) < 0) return false;
  RasEvent* ev = cell.Event();
  *out = std::move(*ev);
  ev->~RasEvent();
  cell.seq.store(dequeuePos_ + mask_ + 1, std::memory_order_release);
  ++dequeuePos_;
  return true;
}

bool EventQueue::Enqueue(RasEvent&& ev) {
  size_t kind = ev.index();
  if (!TryPush(std::move(ev))) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    droppedByKind_[kind].fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  enqueued_.fetch_add(1, std::memory_order_relaxed);
  // The exchange orders the push before the consumer's exchange in ConsumeWakeup, so a
  // producer that skips the write still has its event seen by the drain that follows.
  if (!wakePending_.exchange(true, std::memory_order_acq_rel)) Wake();
  return true;
}

void EventQueue::Wake() {
  wakePending_.store(true, std::memory_order_release);
  int fd = wakeFd_;
  if (fd < 0) return;
  uint64_t one = 1;
  ssize_t rc;
  do {
    rc = write(fd, &one, sizeof(one));
  } while (rc < 0 && errno == EINTR);
}

void EventQueue::ConsumeWakeup() {
  if (wakeFd_ >= 0) {
    uint64_t count;
    while (read(wakeFd_, &count, sizeof(count)) > 0) {
    }
  }
  wakePending_.exchange(false, std::memory_order_acq_rel);
}

size_t EventQueue::Drain(const std::function<void(RasEvent&)>& fn, size_t max) {
  size_t depth = enqueuePos_.load(std::memory_order_relaxed) - dequeuePos_;
  if (depth > mask_ + 1) depth = mask_ + 1;
  if (depth > depthHighWater_.load(std::memory_order_relaxed)) {
    depthHighWater_.store(depth, std::memory_order_relaxed);
  }
  size_t n = 0;
  RasEvent ev;
  while (n < max && TryPop(&ev)) {
    fn(ev);
    ++n;
  }
  return n;
}

EventQueueStats EventQueue::Stats() const {
  EventQueueStats s;
  s.capacity = mask_ + 1;
  s.enqueued = enqueued_.load(std::memory_order_relaxed);
  s.dropped = dropped_.load(std::memory_order_relaxed);
  for (size_t i = 0; i < kNumEventKinds; ++i) {
    s.droppedByKind[i] = droppedByKind_[i].load(std::memory_order_relaxed);
  }
  s.depthHighWater = depthHighWater_.load(std::memory_order_relaxed);
  return s;
}

}  // namespace ras
}  // namespace mori
