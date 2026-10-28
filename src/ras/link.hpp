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

// The RAS thread's event loop: the peer listener, the event queue, and the links, in one
// epoll set. Everything here runs on the RAS thread unless marked otherwise, so link state
// needs no locks; other threads see it only through published snapshots.
//
// Every timer reads CLOCK_MONOTONIC, so an NTP step can't fire or suppress one.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "mori/ras/types.hpp"
#include "src/ras/config.hpp"
#include "src/ras/event_queue.hpp"

namespace mori {
namespace ras {

uint64_t MonoNowNs();
int64_t WallNowNs();

class LinkManager {
 public:
  struct Options {
    RasConfig config;
    RuntimeUuid uuid;
    std::string token;
    EventQueue* events{nullptr};
  };

  explicit LinkManager(Options options);
  // Closes every fd still open, without LEAVE.
  ~LinkManager();
  LinkManager(const LinkManager&) = delete;
  LinkManager& operator=(const LinkManager&) = delete;

  // On the starting thread, before the RAS thread runs. Binds and listens on `bindHost` (an
  // IPv4 or IPv6 literal; empty means every interface) at the configured port.
  bool Listen(const std::string& bindHost, std::string* error);
  uint16_t Port() const { return port_; }
  bool InitLoop(std::string* error);

  // RAS thread.
  void SetIdentity(const RasProcessId& self);
  void RunOnce(int maxWaitMs);
  // Closes everything. Logs nothing: it also runs from the atexit path.
  void Shutdown(bool sendLeave);

  // Any thread.
  void RequestStop();
  bool StopRequested() const { return stop_.load(std::memory_order_acquire); }
  RuntimeSnapshot Snapshot() const;
  bool SetLabels(const Labels& labels);
  void AddDegraded(uint32_t reasons);
  uint32_t Degraded() const { return degraded_.load(std::memory_order_acquire); }
  // In a forked child, whose copies of these fds the fork handler already closed.
  void ForgetFds();

 private:
  void AcceptAll();
  void HandleEvent(RasEvent& ev);
  void Publish(uint64_t nowNs);
  void MarkDirty() { dirty_ = true; }

  Options opts_;
  int listenFd_{-1};
  int epollFd_{-1};
  uint16_t port_{0};
  RasProcessId self_;
  std::set<std::string> localEngines_;
  uint64_t inboundAccepted_{0};
  uint64_t inboundRejected_{0};
  uint64_t protocolErrors_{0};
  uint64_t peersWithoutRas_{0};
  uint64_t sameProcessPeers_{0};
  bool dirty_{true};
  bool moreEvents_{false};  // the last drain stopped at its limit
  uint64_t lastPublishNs_{0};

  std::atomic<bool> stop_{false};
  std::atomic<uint32_t> degraded_{0};
  mutable std::mutex labelsMu_;
  Labels labels_;
  std::atomic<bool> labelsChanged_{false};
  mutable std::mutex snapshotMu_;
  std::shared_ptr<const RuntimeSnapshot> published_;
};

}  // namespace ras
}  // namespace mori
