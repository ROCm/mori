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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "mori/ras/events.hpp"

namespace mori {
namespace ras {

struct EventQueueStats {
  size_t capacity{0};
  uint64_t enqueued{0};
  uint64_t dropped{0};
  std::array<uint64_t, kNumEventKinds> droppedByKind{};
  size_t depthHighWater{0};  // deepest the queue has been when the consumer drained it
};

// Bounded multi-producer, single-consumer queue of RasEvents, plus an eventfd that wakes the
// consumer. Producers never block or wait for the consumer: Enqueue claims a slot with one CAS
// and returns, and a full queue drops the event and counts the drop by kind. The ring is a
// fixed array sized at construction, so the queue allocates nothing after that.
//
// Wakeups coalesce: the first Enqueue after the consumer calls ConsumeWakeup writes the
// eventfd, and later ones don't, so a burst costs one write.
class EventQueue {
 public:
  // Capacity rounds up to a power of two, at least 2.
  explicit EventQueue(size_t capacity);
  ~EventQueue();
  EventQueue(const EventQueue&) = delete;
  EventQueue& operator=(const EventQueue&) = delete;

  // False if the eventfd couldn't be created. The queue still works, but nothing wakes the
  // consumer; it has to poll.
  bool HasWakeFd() const { return wakeFd_ >= 0; }
  int WakeFd() const { return wakeFd_; }

  // Never blocks. Returns false, and counts a drop, if the queue is full.
  bool Enqueue(RasEvent&& ev);

  // Consumer side, one thread only. Re-arms the wakeup; call it before Drain.
  void ConsumeWakeup();
  // Pops up to `max` events in FIFO order. Returns how many it delivered.
  size_t Drain(const std::function<void(RasEvent&)>& fn, size_t max = SIZE_MAX);

  EventQueueStats Stats() const;

 private:
  struct Cell;
  bool TryPush(RasEvent&& ev);
  bool TryPop(RasEvent* out);

  std::unique_ptr<Cell[]> cells_;
  size_t mask_{0};
  int wakeFd_{-1};
  alignas(64) std::atomic<size_t> enqueuePos_{0};
  alignas(64) size_t dequeuePos_{0};       // consumer only
  std::atomic<size_t> depthHighWater_{0};  // written by the consumer only
  alignas(64) std::atomic<bool> wakePending_{false};
  std::atomic<uint64_t> enqueued_{0};
  std::atomic<uint64_t> dropped_{0};
  std::array<std::atomic<uint64_t>, kNumEventKinds> droppedByKind_{};
};

}  // namespace ras
}  // namespace mori
