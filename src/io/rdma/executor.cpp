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
#include <future>
#include <vector>

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
MultithreadExecutor::Worker::Worker(int wid) : workerId(wid) {}

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

  Task task{nullptr, 0, 0, 0};
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mu);
      cond.wait(lock, [this]() { return !q.empty() || !running.load(); });

      if (!running.load()) {
        MORI_IO_INFO("worker {} shutdown", workerId);
        break;
      }
      task = std::move(q.front());
      q.pop();
    }

    SizeVec tLoclOffsets(task.req->localOffsets.begin() + task.begin,
                         task.req->localOffsets.begin() + task.end);
    SizeVec tRemoteOffsets(task.req->remoteOffsets.begin() + task.begin,
                           task.req->remoteOffsets.begin() + task.end);
    SizeVec tSizes(task.req->sizes.begin() + task.begin, task.req->sizes.begin() + task.end);

    const bool preChunked = task.req->inputsArePreChunked;
    const bool chunk = task.req->chunkBytes > 0 && !preChunked;
    RdmaTransferControl control{};
    control.chunkBytes = preChunked ? 0 : task.req->chunkBytes;
    control.maxChunks = task.req->maxChunks;
    control.creditByWrCount = chunk || preChunked;
    control.ownsTotalBatchSize = false;
    control.disableMerge = chunk || preChunked;

    thread_local std::vector<application::RdmaMemoryRegion> localMrPerEp(1);
    thread_local std::vector<application::RdmaMemoryRegion> remoteMrPerEp(1);
    localMrPerEp[0] = task.req->local;
    remoteMrPerEp[0] = task.req->remote;

    RdmaOpRet ret = mori::io::RdmaBatchReadWrite(
        {task.req->eps[task.epId]}, localMrPerEp, remoteMrPerEp, tLoclOffsets, tRemoteOffsets,
        tSizes, task.req->callbackMeta, task.req->id, task.req->isRead, task.req->postBatchSize,
        control);
    MORI_IO_TRACE("Worker {} execute task {} begin {} end {} ret code {}", workerId, task.req->id,
                  task.begin, task.end, static_cast<uint32_t>(ret.code));
    task.ret.set_value(ret);
  }
}

void MultithreadExecutor::Worker::Submit(Task&& task) {
  MORI_IO_FUNCTION_TIMER;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) {
      task.ret.set_value({StatusCode::ERR_BAD_STATE, "worker not started yet"});
      return;
    }
    q.push(std::move(task));
    cond.notify_all();
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

std::vector<std::pair<int, int>> MultithreadExecutor::SplitWork(const ExecutorReq& req) {
  int numEps = req.eps.size();
  int totalBatchSize = req.sizes.size();

  assert(numEps > 0);

  int numActiveWorkers = std::min(numEps, numWorker);
  int perWorkerBatchSize = (totalBatchSize + numActiveWorkers - 1) / numActiveWorkers;

  std::vector<std::pair<int, int>> splits;
  for (int i = 0; i < numActiveWorkers; i++) {
    int begin = i * perWorkerBatchSize;
    int end = std::min(begin + perWorkerBatchSize, totalBatchSize);
    splits.push_back({begin, end});
    if (end >= totalBatchSize) break;
  }

  return splits;
}

std::vector<std::pair<int, int>> MultithreadExecutor::SplitWorkExact(int total, int n) {
  const int numSplits = std::min(n, total);
  std::vector<std::pair<int, int>> splits;
  splits.reserve(static_cast<size_t>(std::max(numSplits, 0)));
  for (int i = 0; i < numSplits; ++i) {
    const int begin = static_cast<int>(static_cast<int64_t>(i) * total / numSplits);
    const int end = static_cast<int>(static_cast<int64_t>(i + 1) * total / numSplits);
    splits.push_back({begin, end});
  }
  return splits;
}

RdmaOpRet MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  MORI_IO_FUNCTION_TIMER;

  auto splits = req.inputsArePreChunked
                    ? SplitWorkExact(static_cast<int>(req.sizes.size()), numWorker)
                    : SplitWork(req);
  int numSplits = splits.size();
  int numEps = static_cast<int>(req.eps.size());
  // Rotate the starting EP by transfer id so single-segment transfers spread
  // evenly across all QPs instead of always landing on eps[0].
  int epOffset = static_cast<int>(req.id % static_cast<uint64_t>(numEps));
  std::vector<std::future<RdmaOpRet>> futs;

  for (int i = 0; i < numSplits; i++) {
    int epId = (i + epOffset) % numEps;
    Task task{&req, epId, splits[i].first, splits[i].second};
    futs.push_back(std::move(task.ret.get_future()));
    // Keep each QP owned by a stable worker to preserve QP affinity.
    pool[epId % numWorker]->Submit(std::move(task));
  }

  bool hasFail = false;
  int numSucc = 0;
  RdmaOpRet failedRet;
  for (auto& fut : futs) {
    RdmaOpRet ret = fut.get();
    if (ret.Failed()) {
      hasFail = true;
      failedRet = ret;
    } else if (ret.Succeeded()) {
      numSucc++;
    }
  }
  if (hasFail) return failedRet;

  if (numSucc == numSplits) {
    return {StatusCode::SUCCESS, ""};
  }

  MORI_IO_TRACE("MultithreadExecutor submit request for RdmaBatchReadWrite done");
  return {StatusCode::IN_PROGRESS, ""};
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
