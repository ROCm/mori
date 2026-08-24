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
#include "src/io/rdma/executor.hpp"

#include <pthread.h>
#include <sched.h>

#include <cstring>
#include <vector>

#include "mori/core/utils/utils.hpp"
#include "mori/io/logging.hpp"
#include "mori/utils/env_utils.hpp"

namespace mori {
namespace io {

namespace {

// Allowed CPUs of the current process, sorted ascending. Reflects cgroup/cpuset
// limits. Empty means affinity could not be read.
std::vector<int> GetAllowedCpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  std::vector<int> cpus;
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    return cpus;
  }
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
  }
  return cpus;
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                   MultithreadExecutor::Worker                                  */
/* ---------------------------------------------------------------------------------------------- */
namespace {
constexpr size_t kInitialRingCapacity = 1024;  // power of two
}  // namespace

MultithreadExecutor::Worker::Worker(int wid) : workerId(wid) {
  ring.resize(kInitialRingCapacity);
  mask = kInitialRingCapacity - 1;
}

void MultithreadExecutor::Worker::GrowRing() {
  const size_t oldCap = ring.size();
  const size_t newCap = oldCap * 2;
  std::vector<Task> next(newCap);
  for (size_t i = 0; i < count; ++i) {
    next[i] = std::move(ring[(head + i) & mask]);
  }
  ring.swap(next);
  head = 0;
  tail = count;
  mask = newCap - 1;
}

MultithreadExecutor::Worker::~Worker() { Shutdown(); }

void MultithreadExecutor::Worker::Start() {
  if (running.load()) return;
  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void MultithreadExecutor::Worker::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) return;
    running.store(false);
    cond.notify_all();
  }
  if (thd.joinable()) thd.join();
}

void MultithreadExecutor::Worker::MainLoop() {
  // MORI_CORE_OFFSET is relative to the allowed CPU list, so binding stays within the cpuset.
  if (auto coreOffset = mori::env::GetInt("MORI_CORE_OFFSET")) {
    std::vector<int> allowed = GetAllowedCpus();
    if (allowed.empty()) {
      MORI_IO_WARN(
          "worker {} could not read allowed CPU set (sched_getaffinity failed); "
          "worker will run on any available core.",
          workerId);
    } else {
      int n = static_cast<int>(allowed.size());
      int idx = ((workerId + *coreOffset) % n + n) % n;
      int targetCore = allowed[idx];

      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(targetCore, &cpuset);

      int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      if (rc != 0) {
        MORI_IO_WARN(
            "worker {} failed to set affinity to core {} (allowed[{}], allowed size {}): "
            "errno={} ({}). Worker will run on any available core. "
            "This is usually caused by NUMA configuration or container CPU limits.",
            workerId, targetCore, idx, n, rc, strerror(rc));
      } else {
        MORI_IO_INFO("worker {} bound to core {} (allowed[{}] of {} allowed CPUs, offset {})",
                     workerId, targetCore, idx, n, *coreOffset);
      }
    }
  }

  MORI_IO_INFO("worker {} enter main loop, running on core {}", workerId, sched_getcpu());

  Task task{};
  SizeVec tLoclOffsets, tRemoteOffsets, tSizes;
  std::vector<application::RdmaMemoryRegion> localMrPerEp(1), remoteMrPerEp(1);

  while (true) {
    {
      std::unique_lock<std::mutex> lock(mu);
      cond.wait(lock, [this]() { return count > 0 || !running.load(); });
      if (!running.load()) {
        MORI_IO_INFO("worker {} shutdown", workerId);
        break;
      }
      task = std::move(ring[head & mask]);
      ++head;
      --count;
    }

    auto *preq = task.req;
    tLoclOffsets.assign(preq->localOffsets.begin() + task.begin,
                         preq->localOffsets.begin() + task.end);
    tRemoteOffsets.assign(preq->remoteOffsets.begin() + task.begin,
                           preq->remoteOffsets.begin() + task.end);
    tSizes.assign(preq->sizes.begin() + task.begin, preq->sizes.begin() + task.end);

    const bool chunk = task.req->chunkBytes > 0;
    RdmaTransferControl control{};
    control.chunkBytes = preq->chunkBytes;
    control.maxChunks = preq->maxChunks;
    control.creditByWrCount = chunk;
    control.ownsTotalBatchSize = false;
    control.disableMerge = chunk;

    localMrPerEp[0] = preq->local;
    remoteMrPerEp[0] = preq->remote;

    RdmaOpRet ret = mori::io::RdmaBatchReadWrite(
        {task.req->eps[task.epId]}, localMrPerEp, remoteMrPerEp, tLoclOffsets, tRemoteOffsets,
        tSizes, preq->callbackMeta, preq->id, preq->isRead, preq->postBatchSize,
        control);
    task.latch->Complete(ret);
    MORI_IO_TRACE("Worker {} execute task {} begin {} end {} ret code {}", workerId, preq->id,
                  task.begin, task.end, static_cast<uint32_t>(ret.code));
  }
}

void MultithreadExecutor::Worker::Submit(Task&& task) {
  MORI_IO_FUNCTION_TIMER;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) {
      task.latch->Complete({StatusCode::ERR_BAD_STATE, "worker not started yet"});
      return;
    }
    if (MORI_UNLIKELY(count == ring.size())) {
      GrowRing();
    }
    ring[tail & mask] = std::move(task);
    ++tail;
    ++count;
    cond.notify_one();
  }
  MORI_IO_TRACE("Submit to worker {} task {} begin {} end {}", workerId, task.req->id, task.begin,
                task.end);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       MultithreadExecutor                                      */
/* ---------------------------------------------------------------------------------------------- */
MultithreadExecutor::MultithreadExecutor(int n) : numWorker(n) {
  assert(n > 0);
  for (int i = 0; i < numWorker; i++) {
    pool.emplace_back(new Worker(i));
  }
}

MultithreadExecutor::~MultithreadExecutor() { Shutdown(); }

RdmaOpRet MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  MORI_IO_FUNCTION_TIMER;

  int numEps = static_cast<int>(req.eps.size());
  int totalBatchSize = static_cast<int>(req.sizes.size());
  assert(numEps > 0);

  // Split the batch across at most one worker per EP. Ranges are derived inline
  // (previously SplitWork returned a heap std::vector<std::pair> per call).
  int numActiveWorkers = std::min(numEps, numWorker);
  int perWorkerBatchSize = (totalBatchSize + numActiveWorkers - 1) / numActiveWorkers;
  // Number of non-empty splits. An empty batch still yields one {0,0} split so
  // the caller gets a well-formed (empty) completion, matching prior behavior.
  int numSplits =
      (totalBatchSize == 0) ? 1 : (totalBatchSize + perWorkerBatchSize - 1) / perWorkerBatchSize;

  // Rotate the starting EP by transfer id so single-segment transfers spread
  // evenly across all QPs instead of always landing on eps[0].
  int epOffset = static_cast<int>(req.id % static_cast<uint64_t>(numEps));

  // One stack-local latch shared by all splits: no per-split heap allocation and
  // a single blocking wait for the whole batch (see BatchLatch in executor.hpp).
  BatchLatch latch(numSplits);
  for (int i = 0; i < numSplits; i++) {
    int begin = i * perWorkerBatchSize;
    int end = std::min(begin + perWorkerBatchSize, totalBatchSize);
    int epId = (i + epOffset) % numEps;
    Task task{&req, epId, begin, end, &latch};
    // Keep each QP owned by a stable worker to preserve QP affinity.
    pool[epId % numWorker]->Submit(std::move(task));
  }

  RdmaOpRet ret = latch.Wait(numSplits);
  MORI_IO_TRACE("MultithreadExecutor submit request for RdmaBatchReadWrite done");
  return ret;
}

void MultithreadExecutor::Start() {
  for (auto& worker : pool) {
    worker->Start();
  }
}

void MultithreadExecutor::Shutdown() {
  for (auto& worker : pool) {
    worker->Shutdown();
  }
}

}  // namespace io
}  // namespace mori
