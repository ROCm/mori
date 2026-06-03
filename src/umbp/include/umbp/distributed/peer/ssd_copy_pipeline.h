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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

class PeerDramAllocator;
class PeerSsdManager;

// One async SSD copy request, enqueued after a DRAM commit succeeds.  It
// deliberately carries NO source pointer: the worker re-derives the readable
// segments from `key` via PeerDramAllocator::AcquireDramCopyPin, so an
// already-evicted key is simply dropped (the pin acquisition fails).
struct SsdCopyTask {
  std::string key;
  TierType source_tier = TierType::DRAM;  // informational; worker reads via key
  size_t size = 0;                        // informational; pin reports true size
};

// Peer-local copy-on-commit pipeline.  After a key's DRAM pages are
// committed on this owner peer, the commit path enqueues an SsdCopyTask; a
// background worker copies the bytes to the local SSD tier best-effort.
//
// Key properties:
//   - Enqueue never blocks commit: a full queue (or a stopped/quiescing
//     pipeline) drops the task and counts it.
//   - The worker holds a DramCopyPin across the backend Write so eviction
//     cannot free the pages underneath it; a RAII guard guarantees the pin is
//     released on every exit (success, failure, or early return).
//   - Only a successful PeerSsdManager::Write records the SSD location and
//     emits ADD SSD.
class SsdCopyPipeline {
 public:
  SsdCopyPipeline(PeerDramAllocator* dram, PeerSsdManager* ssd, size_t queue_depth = 4096,
                  size_t worker_threads = 1);
  ~SsdCopyPipeline();

  SsdCopyPipeline(const SsdCopyPipeline&) = delete;
  SsdCopyPipeline& operator=(const SsdCopyPipeline&) = delete;

  // Spawn worker threads and begin accepting tasks.  Idempotent.
  void Start();

  // Stop accepting, drop any queued tasks, let the in-flight task finish (its
  // pin is released by the RAII guard), then join the workers.  Idempotent.
  void Stop();

  // Pause intake and drain in-flight work without tearing down the workers:
  // stop accepting, drop queued tasks, and block until no task is running.
  // Used by PoolClient::Clear so no in-flight copy re-populates SSD state
  // right after the managers are cleared.  Pair with Resume().
  void Quiesce();

  // Resume accepting tasks after Quiesce().
  void Resume();

  // Enqueue a copy task.  Returns false (without blocking) when the pipeline
  // is not accepting (stopped/quiescing) or the bounded queue is full; a
  // full-queue drop is counted in dropped().
  bool Enqueue(SsdCopyTask task);

  // Prometheus-only observability snapshots (see metrics_ below).  Not
  // correctness state; sampled once per metrics tick by PublishSsdMetrics().
  uint64_t Enqueued() const { return metrics_.enqueued.load(std::memory_order_relaxed); }
  // CopiedOk counts completed copies: a real backend Write OR a content-addressed
  // dedup hit (already resident) — both report success, only the former emits ADD SSD.
  uint64_t CopiedOk() const { return metrics_.copied_ok.load(std::memory_order_relaxed); }
  uint64_t Failed() const { return metrics_.failed.load(std::memory_order_relaxed); }
  // Dropped because the bounded queue was full (kept name for existing tests).
  uint64_t Dropped() const { return metrics_.dropped_queue_full.load(std::memory_order_relaxed); }
  // Dropped because the pipeline was stopped / quiescing at enqueue time.
  uint64_t DroppedStopped() const {
    return metrics_.dropped_stopped.load(std::memory_order_relaxed);
  }

 private:
  void WorkerLoop();
  void RunTask(const SsdCopyTask& task);

  PeerDramAllocator* dram_;
  PeerSsdManager* ssd_;
  const size_t queue_depth_;
  const size_t worker_threads_;

  std::mutex mu_;
  std::condition_variable cv_;       // wakes workers on new work / stop
  std::condition_variable idle_cv_;  // wakes Quiesce when no task is running
  std::deque<SsdCopyTask> queue_;
  // Intake is open unless stopped (Stop) or paused (Quiesce).  Default-open so
  // the bounded-queue drop path is exercised even before Start spawns workers;
  // in production Start() is always called before the first commit.
  bool stop_ = false;
  bool paused_ = false;
  size_t active_ = 0;  // tasks currently being processed by workers

  // Prometheus-only observability counters (see getters above): relaxed atomics
  // bumped at the enqueue/run events, read once per metrics tick.  NOT
  // correctness state (queue_/stop_/paused_ are authoritative).
  struct MetricsCounters {
    std::atomic<uint64_t> enqueued{0};            // tasks accepted into the queue
    std::atomic<uint64_t> copied_ok{0};           // copy completed (write OK or dedup hit)
    std::atomic<uint64_t> failed{0};              // backend Write failed / unusable pin
    std::atomic<uint64_t> dropped_queue_full{0};  // dropped: bounded queue full
    std::atomic<uint64_t> dropped_stopped{0};     // dropped: pipeline stopped/quiescing
  };
  MetricsCounters metrics_;

  std::vector<std::thread> workers_;
};

}  // namespace mori::umbp
