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
#include "umbp/distributed/transfer/restore_scheduler.h"

#include <hip/hip_runtime.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {
namespace {
using HostMapper = std::function<void*(const void*, size_t, int)>;

struct RestoreRequest {
  std::vector<DeviceGatherFragment> fragments;
  HostMapper map_host;
  int destination = -1;
  size_t unassigned_bytes = 0;
  size_t fragment_index = 0;
  size_t fragment_offset = 0;
  size_t in_flight = 0;
  bool failed = false;
  bool finished = false;
  std::condition_variable done;
};

struct RestoreWorker {
  int device;
  hipStream_t stream = nullptr;
  hipEvent_t complete = nullptr;
  std::vector<bool> can_write;
  std::vector<bool> peer_enabled;
  std::shared_ptr<RestoreRequest> request;
  std::vector<DeviceGatherFragment> fragments;
  size_t bytes = 0;
};

class RestoreScheduler {
 public:
  explicit RestoreScheduler(size_t chunk_bytes)
      : chunk_bytes_(chunk_bytes), thread_([this] { Run(); }) {}

  ~RestoreScheduler() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_one();
    thread_.join();
  }

  bool Restore(std::shared_ptr<RestoreRequest> request) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_) return false;
    requests_.push_back(request);
    wake_.notify_one();
    request->done.wait(lock, [&] { return request->finished; });
    return !request->failed;
  }

 private:
  // Called with mutex_ held. A failed request must retain its source/destination
  // buffers until every chunk that was already submitted has stopped using them.
  void Finish(const std::shared_ptr<RestoreRequest>& request) {
    if (request->in_flight != 0 || (!request->failed && request->unassigned_bytes != 0)) return;
    request->finished = true;
    requests_.erase(std::find(requests_.begin(), requests_.end(), request));
    request->done.notify_one();
  }

  void Complete(RestoreWorker& worker, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto request = std::move(worker.request);
    request->failed |= !success;
    --request->in_flight;
    Finish(request);
  }

  // Only the scheduler thread touches streams, events, descriptor storage and
  // per-device peer state. Each GPU has at most one submitted chunk at a time.
  bool Launch(RestoreWorker& worker) {
    auto& request = *worker.request;
    if (hipSetDevice(worker.device) != hipSuccess) return false;
    if (worker.stream == nullptr &&
        hipStreamCreateWithFlags(&worker.stream, hipStreamNonBlocking) != hipSuccess)
      return false;
    if (worker.complete == nullptr &&
        hipEventCreateWithFlags(&worker.complete, hipEventDisableTiming) != hipSuccess)
      return false;
    if (!worker.peer_enabled[request.destination]) {
      auto status = hipDeviceEnablePeerAccess(request.destination, 0);
      if (status != hipSuccess && status != hipErrorPeerAccessAlreadyEnabled) return false;
      if (status == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
      worker.peer_enabled[request.destination] = true;
    }
    for (auto& fragment : worker.fragments) {
      fragment.src = request.map_host(fragment.src, fragment.bytes, worker.device);
      if (fragment.src == nullptr) return false;
    }
    if (!LaunchDeviceGather(worker.fragments.data(), worker.fragments.size(), worker.device,
                            worker.stream))
      return false;
    return hipEventRecord(worker.complete, worker.stream) == hipSuccess;
  }

  // Advance the request cursor across ranges, reusing this GPU's descriptor
  // storage. A chunk may end partway through a range.
  void PrepareChunk(RestoreWorker& worker) {
    auto& request = *worker.request;
    worker.fragments.clear();
    size_t remaining = worker.bytes;
    while (remaining != 0) {
      const auto& fragment = request.fragments[request.fragment_index];
      const auto bytes = std::min(remaining, fragment.bytes - request.fragment_offset);
      worker.fragments.push_back({static_cast<const char*>(fragment.src) + request.fragment_offset,
                                  static_cast<char*>(fragment.dst) + request.fragment_offset,
                                  bytes});
      remaining -= bytes;
      request.fragment_offset += bytes;
      if (request.fragment_offset == fragment.bytes) {
        ++request.fragment_index;
        request.fragment_offset = 0;
      }
    }
  }

  // Queue order provides round-robin service: move each selected request to
  // the back, skipping failed requests and those with every chunk assigned.
  // Called with mutex_ held.
  bool AssignRequest(RestoreWorker& worker) {
    const auto selected =
        std::find_if(requests_.begin(), requests_.end(), [&](const auto& request) {
          return !request->failed && request->unassigned_bytes != 0 &&
                 request->destination < static_cast<int>(worker.can_write.size()) &&
                 worker.can_write[request->destination];
        });
    if (selected == requests_.end()) return false;
    worker.request = *selected;
    requests_.splice(requests_.end(), requests_, selected);
    auto& request = *worker.request;
    worker.bytes = std::min(chunk_bytes_, request.unassigned_bytes);
    request.unassigned_bytes -= worker.bytes;
    ++request.in_flight;
    PrepareChunk(worker);
    return true;
  }

  void Run() {
#ifdef __linux__
    // Linux's default timer slack can add ~50 us to a short timed wait. Limit
    // coalescing on this thread only, so completion polls need not oversleep.
    (void)prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) count = 0;
    std::vector<RestoreWorker> workers(count);
    for (int device = 0; device < count; ++device) {
      auto& worker = workers[device];
      worker.device = device;
      worker.can_write.resize(count);
      worker.peer_enabled.resize(count);
      for (int destination = 0; destination < count; ++destination) {
        int accessible = 0;
        worker.can_write[destination] =
            device == destination ||
            (hipDeviceCanAccessPeer(&accessible, device, destination) == hipSuccess && accessible);
      }
      worker.peer_enabled[device] = true;
    }
    for (;;) {
      for (auto& worker : workers) {
        if (!worker.request) continue;
        (void)hipSetDevice(worker.device);
        const auto status = hipEventQuery(worker.complete);
        if (status == hipErrorNotReady) continue;
        if (status != hipSuccess) {
          (void)hipStreamSynchronize(worker.stream);
          (void)hipGetLastError();
        }
        Complete(worker, status == hipSuccess);
      }
      for (auto& worker : workers) {
        if (worker.request) continue;
        bool launched = false;
        try {
          {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!AssignRequest(worker)) continue;
          }
          launched = Launch(worker);
        } catch (const std::exception& error) {
          MORI_UMBP_WARN("[RestoreScheduler] chunk submission failed: {}", error.what());
        }
        if (!launched) {
          MORI_UMBP_WARN(
              "[RestoreScheduler] restore chunk failed on device {}; draining before fallback",
              worker.device);
          (void)hipSetDevice(worker.device);
          if (worker.stream != nullptr) (void)hipStreamSynchronize(worker.stream);
          (void)hipGetLastError();
          Complete(worker, false);
        }
      }
      std::unique_lock<std::mutex> lock(mutex_);
      // No device can ever serve these requests. Do not leave their callers waiting.
      for (auto it = requests_.begin(); it != requests_.end();) {
        const auto request = *it++;
        const bool reachable = std::any_of(workers.begin(), workers.end(), [&](const auto& worker) {
          return request->destination < count && worker.can_write[request->destination];
        });
        if (!reachable && request->in_flight == 0) {
          request->failed = true;
          Finish(request);
        }
      }
      if (stopping_ && requests_.empty()) break;
      if (requests_.empty()) {
        wake_.wait(lock, [&] { return stopping_ || !requests_.empty(); });
      } else {
        // Sleep between completion checks rather than occupying a CPU core.
        // New requests still wake idle workers immediately.
        wake_.wait_for(lock, std::chrono::microseconds(10));
      }
    }
    for (auto& worker : workers) {
      (void)hipSetDevice(worker.device);
      if (worker.complete != nullptr) (void)hipEventDestroy(worker.complete);
      if (worker.stream != nullptr) (void)hipStreamDestroy(worker.stream);
    }
  }

  const size_t chunk_bytes_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::list<std::shared_ptr<RestoreRequest>> requests_;
  bool stopping_ = false;
  std::thread thread_;
};
}  // namespace

bool TryStripedRestore(const std::vector<TransferPlan>& plans, const HostMapper& map_host,
                       size_t chunk_bytes) {
  if (plans.empty() || chunk_bytes == 0 || !DeviceGatherEnabled()) return false;
  auto request = std::make_shared<RestoreRequest>();
  request->map_host = map_host;
  size_t fragment_count = 0;
  for (const auto& plan : plans) {
    if (plan.src.loc != mori::io::MemoryLocationType::CPU ||
        plan.dst.loc != mori::io::MemoryLocationType::GPU || plan.dst.device < 0 ||
        (request->destination >= 0 && request->destination != plan.dst.device))
      return false;
    if (plan.src.host_ptr == nullptr || plan.dst.host_ptr == nullptr ||
        plan.src_offsets.size() != plan.sizes.size() ||
        plan.dst_offsets.size() != plan.sizes.size())
      return false;
    request->destination = plan.dst.device;
    if (plan.sizes.size() > request->fragments.max_size() - fragment_count) return false;
    fragment_count += plan.sizes.size();
  }
  // Allocate descriptor storage once. Zero-byte ranges are skipped below, so
  // the total range count is an upper bound on the number of fragments.
  request->fragments.reserve(fragment_count);
  for (const auto& plan : plans) {
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      const size_t bytes = plan.sizes[i];
      if (bytes == 0) continue;
      if (bytes > std::numeric_limits<size_t>::max() - request->unassigned_bytes) return false;
      request->unassigned_bytes += bytes;
      request->fragments.push_back(
          {static_cast<const char*>(plan.src.host_ptr) + plan.src_offsets[i],
           static_cast<char*>(plan.dst.host_ptr) + plan.dst_offsets[i], bytes});
    }
  }
  if (request->unassigned_bytes == 0) return false;
  // One process-wide scheduler owns all restore streams. Configuration is fixed
  // for the process, like the existing environment-based gather dispatch.
  static RestoreScheduler scheduler(chunk_bytes);
  return scheduler.Restore(std::move(request));
}
}  // namespace mori::umbp
