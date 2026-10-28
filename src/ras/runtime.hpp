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

// One RAS runtime: its identity, event queue, link manager, and thread. The process-wide
// RasRuntime in include/mori/ras/ras.hpp wraps exactly one; tests build several in one
// process to link them to each other.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "mori/ras/events.hpp"
#include "mori/ras/types.hpp"
#include "src/ras/config.hpp"
#include "src/ras/event_queue.hpp"
#include "src/ras/identity.hpp"
#include "src/ras/link.hpp"

namespace mori {
namespace ras {

struct CoreParams {
  RasConfig config;
  std::string nodeId;
  std::string hostname;
  std::string host;
  int32_t pid{0};
  IdentitySources identity{LiveIdentitySources()};
  size_t eventCapacity{1024};
};

class RasCore {
 public:
  explicit RasCore(CoreParams params);
  ~RasCore();
  RasCore(const RasCore&) = delete;
  RasCore& operator=(const RasCore&) = delete;

  // Binds the listener and starts the RAS thread. On failure, *error says why and nothing is
  // left running.
  bool Start(std::string* error);
  // Stops the thread, which sends LEAVE on its links first if `sendLeave`. Waits at most 2 s;
  // a thread that doesn't finish by then is abandoned rather than holding up exit.
  void Stop(bool sendLeave);

  bool Enqueue(RasEvent&& ev);
  RasEndpoint Endpoint() const;
  RuntimeSnapshot Snapshot() const;
  bool SetLabels(const Labels& labels);
  void AddDegraded(uint32_t reasons);

  // In a forked child: the thread is gone and the fd registry has closed the fds, so the core
  // must never touch either again. It is leaked rather than destroyed.
  void OrphanAfterFork();

 private:
  void ThreadMain();

  CoreParams params_;
  RuntimeUuid uuid_;
  std::string token_;
  std::string advertisedHost_;
  std::unique_ptr<EventQueue> queue_;
  std::unique_ptr<LinkManager> links_;
  std::thread* thread_{nullptr};  // never deleted in a forked child, where it isn't running
  std::atomic<bool> running_{false};
  std::atomic<bool> stopped_{false};
  std::atomic<bool> orphaned_{false};
  std::atomic<bool> sendLeave_{true};
  std::mutex doneMu_;
  std::condition_variable doneCv_;
  bool threadDone_{false};
};

}  // namespace ras
}  // namespace mori
