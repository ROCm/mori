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
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {
namespace {
using HostMapper = std::function<void*(const void*, size_t, int)>;

void ConfigurePollingThread() {
#ifdef __linux__
  // Reduce timed-wait coalescing on this thread. In our MI355X mixed-restore
  // microbenchmark, this lowered mean 8 MiB restore latency from 253 us to 209 us.
  constexpr unsigned long kTimerSlackNs = 1;
  if (prctl(PR_SET_TIMERSLACK, kTimerSlackNs, 0UL, 0UL, 0UL) != 0) {
    MORI_UMBP_WARN("[RestoreScheduler] failed to configure polling timer slack: {}",
                   std::strerror(errno));
  }
#endif
}

struct RestoreRequest {
  std::vector<size_t> object_page_offsets;
  std::vector<const void*> pages;
  std::vector<DeviceGatherRange> ranges;
  size_t page_size = 0;
  HostMapper map_host;
  int destination = -1;
  size_t unassigned_bytes = 0;
  size_t range_index = 0;
  size_t range_offset = 0;
  size_t in_flight = 0;
  bool failed = false;
  bool finished = false;
  std::condition_variable done;
};

struct RestoreWorker {
  int device;
  hipStream_t stream = nullptr;
  hipEvent_t complete = nullptr;
  std::shared_ptr<RestoreRequest> request;
  std::vector<DeviceGatherFragment> fragments;
  std::vector<size_t> object_page_offsets;
  std::vector<const void*> pages;
  std::vector<DeviceGatherRange> ranges;
  size_t bytes = 0;
};

class RestoreScheduler {
 public:
  explicit RestoreScheduler(size_t chunk_bytes) : chunk_bytes_(chunk_bytes) {
    auto check = [](hipError_t status, const std::string& operation) {
      if (status != hipSuccess)
        throw std::runtime_error(operation + ": " + hipGetErrorString(status));
    };
    int count = 0;
    check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count == 0) throw std::runtime_error("Striped restores require at least one GPU");
    workers_.resize(count);
    int previous_device = -1;
    check(hipGetDevice(&previous_device), "hipGetDevice");
    // Require full peer connectivity before starting the scheduler thread.
    // Initialization errors are thrown to TryStripedRestore, with no background
    // thread to join.
    try {
      for (int device = 0; device < count; ++device) {
        workers_[device].device = device;
        check(hipSetDevice(device), "hipSetDevice");
        for (int destination = 0; destination < count; ++destination) {
          if (device == destination) continue;
          const auto status = hipDeviceEnablePeerAccess(destination, 0);
          if (status == hipErrorPeerAccessAlreadyEnabled) {
            (void)hipGetLastError();
          } else {
            check(status, "hipDeviceEnablePeerAccess(" + std::to_string(device) + " -> " +
                              std::to_string(destination) + ")");
          }
        }
      }
    } catch (...) {
      (void)hipSetDevice(previous_device);
      throw;
    }
    check(hipSetDevice(previous_device), "hipSetDevice");
    thread_ = std::thread([this] { Run(); });
  }

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
    if (stopping_ || request->destination >= static_cast<int>(workers_.size())) return false;
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

  // Only the scheduler thread touches streams, events and descriptor storage.
  // Each GPU has at most one submitted chunk at a time.
  bool Launch(RestoreWorker& worker) {
    auto& request = *worker.request;
    if (hipSetDevice(worker.device) != hipSuccess) return false;
    if (worker.stream == nullptr &&
        hipStreamCreateWithFlags(&worker.stream, hipStreamNonBlocking) != hipSuccess)
      return false;
    if (worker.complete == nullptr &&
        hipEventCreateWithFlags(&worker.complete, hipEventDisableTiming) != hipSuccess)
      return false;
    bool launched;
    if (request.page_size != 0) {
      launched = LaunchPagedChunk(worker);
    } else {
      worker.fragments.clear();
      for (const auto& range : worker.ranges) {
        const auto* source = static_cast<const char*>(request.pages[range.object]) + range.offset;
        const void* mapped = request.map_host(source, range.bytes, worker.device);
        if (!mapped) return false;
        worker.fragments.push_back({mapped, range.dst, range.bytes});
      }
      launched = LaunchDeviceGather(worker.fragments.data(), worker.fragments.size(),
                                    worker.device, worker.stream);
    }
    return launched && hipEventRecord(worker.complete, worker.stream) == hipSuccess;
  }

  // Map only the pages touched by this chunk, leaving page expansion to the kernel.
  bool LaunchPagedChunk(RestoreWorker& worker) {
    auto& request = *worker.request;
    worker.object_page_offsets.clear();
    worker.pages.clear();
    for (auto& range : worker.ranges) {
      const size_t first = request.object_page_offsets[range.object] + range.offset / request.page_size;
      range.offset %= request.page_size;
      const size_t count = 1 + (range.offset + range.bytes - 1) / request.page_size;
      range.object = worker.object_page_offsets.size();
      worker.object_page_offsets.push_back(worker.pages.size());
      for (size_t p = first; p < first + count; ++p) {
        const void* mapped = request.map_host(request.pages[p], request.page_size, worker.device);
        if (!mapped) return false;
        worker.pages.push_back(mapped);
      }
    }
    return LaunchDevicePagedGather(
        worker.object_page_offsets.data(), worker.object_page_offsets.size(), request.page_size,
        worker.pages.data(), worker.pages.size(), worker.ranges.data(), worker.ranges.size(),
        worker.device, worker.stream);
  }

  // Split logical ranges by byte count, independent of the source layout.
  void PrepareChunk(RestoreWorker& worker) {
    auto& request = *worker.request;
    worker.ranges.clear();
    size_t remaining = worker.bytes;
    while (remaining != 0) {
      const auto& range = request.ranges[request.range_index];
      const auto bytes = std::min(remaining, range.bytes - request.range_offset);
      const size_t offset = range.offset + request.range_offset;
      auto* dst = static_cast<char*>(range.dst) + request.range_offset;
      worker.ranges.push_back({dst, range.object, offset, bytes});
      remaining -= bytes;
      request.range_offset += bytes;
      if (request.range_offset == range.bytes) {
        ++request.range_index;
        request.range_offset = 0;
      }
    }
  }

  // Queue order provides round-robin service: move each selected request to
  // the back, skipping failed requests and those with every chunk assigned.
  // Called with mutex_ held.
  bool AssignRequest(RestoreWorker& worker) {
    const auto selected = std::find_if(requests_.begin(), requests_.end(), [](const auto& request) {
      return !request->failed && request->unassigned_bytes != 0;
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
    ConfigurePollingThread();
    for (;;) {
      for (auto& worker : workers_) {
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
      for (auto& worker : workers_) {
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
      if (stopping_ && requests_.empty()) break;
      if (requests_.empty()) {
        wake_.wait(lock, [&] { return stopping_ || !requests_.empty(); });
      } else {
        // Sleep between completion checks rather than occupying a CPU core.
        // New requests still wake idle workers immediately.
        wake_.wait_for(lock, std::chrono::microseconds(10));
      }
    }
    for (auto& worker : workers_) {
      (void)hipSetDevice(worker.device);
      if (worker.complete != nullptr) (void)hipEventDestroy(worker.complete);
      if (worker.stream != nullptr) (void)hipStreamDestroy(worker.stream);
    }
  }

  const size_t chunk_bytes_;
  std::vector<RestoreWorker> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::list<std::shared_ptr<RestoreRequest>> requests_;
  bool stopping_ = false;
  std::thread thread_;
};

bool SubmitRestore(std::shared_ptr<RestoreRequest> request, size_t chunk_bytes) {
  // Both descriptor formats share one queue, worker pool and failure handling.
  static const std::unique_ptr<RestoreScheduler> scheduler =
      [chunk_bytes]() -> std::unique_ptr<RestoreScheduler> {
    try {
      return std::make_unique<RestoreScheduler>(chunk_bytes);
    } catch (const std::exception& error) {
      MORI_UMBP_WARN("[RestoreScheduler] striped restores unavailable, using direct restores: {}",
                     error.what());
      return nullptr;
    }
  }();
  return scheduler != nullptr && scheduler->Restore(std::move(request));
}
}  // namespace

bool TryStripedRestore(const std::vector<TransferPlan>& plans, const HostMapper& map_host,
                       size_t chunk_bytes) {
  if (plans.empty() || chunk_bytes == 0 || !DeviceGatherEnabled()) return false;
  auto request = std::make_shared<RestoreRequest>();
  request->map_host = map_host;
  size_t range_count = 0;
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
    if (plan.sizes.size() > request->ranges.max_size() - range_count) return false;
    range_count += plan.sizes.size();
  }
  // Allocate descriptor storage once. Zero-byte ranges are skipped below, so
  // the total range count is an upper bound on the number of descriptors.
  request->ranges.reserve(range_count);
  request->pages.reserve(plans.size());
  for (const auto& plan : plans) {
    const size_t object = request->pages.size();
    request->pages.push_back(plan.src.host_ptr);
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      const size_t bytes = plan.sizes[i];
      if (bytes == 0) continue;
      if (bytes > std::numeric_limits<size_t>::max() - request->unassigned_bytes) return false;
      request->unassigned_bytes += bytes;
      request->ranges.push_back(
          {static_cast<char*>(plan.dst.host_ptr) + plan.dst_offsets[i], object,
           plan.src_offsets[i], bytes});
    }
  }
  if (request->unassigned_bytes == 0) return false;
  // One process-wide scheduler owns all restore streams. Configuration is fixed
  // for the process, like the existing environment-based gather dispatch.
  //
  // The scheduler cannot start without peer access between every pair of GPUs,
  // but the caller's direct path needs none. A failed start is therefore kept
  // as the answer for the process: every restore is declined, and the caller
  // copies through its direct path instead of seeing an exception.
  return SubmitRestore(std::move(request), chunk_bytes);
}

bool TryStripedRestore(
    const std::vector<size_t>& object_page_offsets, size_t page_size,
    const std::vector<const void*>& pages, const std::vector<DeviceGatherRange>& ranges,
    int destination, const HostMapper& map_host, size_t chunk_bytes) {
  if (ranges.empty() || page_size == 0 || chunk_bytes == 0 || destination < 0 ||
      !DeviceGatherEnabled()) return false;
  auto request = std::make_shared<RestoreRequest>();
  request->object_page_offsets = object_page_offsets;
  request->pages = pages;
  request->page_size = page_size;
  request->destination = destination;
  request->map_host = map_host;
  for (const auto& range : ranges) {
    if (range.bytes == 0) continue;
    if (range.bytes > std::numeric_limits<size_t>::max() - request->unassigned_bytes) return false;
    request->ranges.push_back(range);
    request->unassigned_bytes += range.bytes;
  }
  return request->unassigned_bytes != 0 && SubmitRestore(std::move(request), chunk_bytes);
}
}  // namespace mori::umbp
