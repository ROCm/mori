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

#include "mori/io/logging.hpp"

namespace mori {
namespace io {

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
  int coreOffset = 0;
  const char* env = std::getenv("MORI_CORE_OFFSET");
  if (env) {
    coreOffset = std::stoi(env);
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int targetCore = workerId + coreOffset;
  CPU_SET(targetCore, &cpuset);

  int rc = pthread_setaffinity_np(thd.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    MORI_IO_WARN(
        "worker {} failed to set affinity to core {}: errno={} ({}). "
        "Worker will run on any available core. "
        "This is usually caused by: CPU not available in cpuset, "
        "NUMA configuration, or container CPU limits.",
        workerId, targetCore, rc, strerror(rc));
  }

  MORI_IO_INFO("worker {} enter main loop, running on core {}", workerId, sched_getcpu());

  Task task;
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

    RdmaOpRet ret;
    TransferUniqueId taskId = 0;

    if (task.kind == Task::Kind::SingleMr) {
      SizeVec tLoclOffsets(task.singleReq->localOffsets.begin() + task.begin,
                           task.singleReq->localOffsets.begin() + task.end);
      SizeVec tRemoteOffsets(task.singleReq->remoteOffsets.begin() + task.begin,
                             task.singleReq->remoteOffsets.begin() + task.end);
      SizeVec tSizes(task.singleReq->sizes.begin() + task.begin,
                     task.singleReq->sizes.begin() + task.end);

      ret = mori::io::RdmaBatchReadWrite({task.singleReq->eps[task.epId]}, task.singleReq->local,
                                         tLoclOffsets, task.singleReq->remote, tRemoteOffsets,
                                         tSizes, task.singleReq->callbackMeta, task.singleReq->id,
                                         task.singleReq->isRead, task.singleReq->postBatchSize);
      taskId = task.singleReq->id;
    } else {
      ret = mori::io::RdmaPostResolvedSlicesToSingleEp(
          task.resolvedReq->eps[task.epId], task.resolvedReq->slices, task.begin, task.end,
          task.resolvedReq->callbackMeta, task.resolvedReq->id, task.resolvedReq->isRead,
          task.resolvedReq->postBatchSize);
      taskId = task.resolvedReq->id;
    }

    task.ret.set_value(ret);
    MORI_IO_TRACE("Worker {} execute task {} begin {} end {} ret code {}", workerId, taskId,
                  task.begin, task.end, static_cast<uint32_t>(ret.code));
  }
}

void MultithreadExecutor::Worker::Submit(Task&& task) {
  MORI_IO_FUNCTION_TIMER;
  TransferUniqueId taskId =
      task.kind == Task::Kind::SingleMr ? task.singleReq->id : task.resolvedReq->id;
  size_t begin = task.begin;
  size_t end = task.end;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) {
      task.ret.set_value({StatusCode::ERR_BAD_STATE, "worker not started yet"});
      return;
    }
    q.push(std::move(task));
    cond.notify_all();
  }
  MORI_IO_TRACE("Submit to worker {} task {} begin {} end {}", workerId, taskId, begin, end);
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

std::vector<std::pair<size_t, size_t>> MultithreadExecutor::SplitWork(const ExecutorReq& req) {
  size_t numEps = req.eps.size();
  size_t totalBatchSize = req.sizes.size();

  assert(numEps > 0);

  size_t numActiveWorkers = std::min(numEps, static_cast<size_t>(numWorker));
  size_t perWorkerBatchSize = (totalBatchSize + numActiveWorkers - 1) / numActiveWorkers;

  std::vector<std::pair<size_t, size_t>> splits;
  for (size_t i = 0; i < numActiveWorkers; i++) {
    size_t begin = i * perWorkerBatchSize;
    size_t end = std::min(begin + perWorkerBatchSize, totalBatchSize);
    splits.push_back({begin, end});
    if (end >= totalBatchSize) break;
  }

  return splits;
}

RdmaOpRet MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  MORI_IO_FUNCTION_TIMER;

  auto splits = SplitWork(req);
  size_t numSplits = splits.size();
  std::vector<std::future<RdmaOpRet>> futs;

  for (size_t i = 0; i < numSplits; i++) {
    Task task{&req, static_cast<int>(i), splits[i].first, splits[i].second};
    futs.push_back(std::move(task.ret.get_future()));
    pool[i]->Submit(std::move(task));
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

RdmaOpRet MultithreadExecutor::RdmaBatchReadWriteResolved(const ResolvedExecutorReq& req) {
  MORI_IO_FUNCTION_TIMER;

  if (req.slices.empty()) return {StatusCode::SUCCESS, ""};
  if (req.lanes.empty()) return {StatusCode::ERR_INVALID_ARGS, "resolved transfer lanes are empty"};

  std::vector<std::future<RdmaOpRet>> futs;
  futs.reserve(req.lanes.size());

  for (size_t i = 0; i < req.lanes.size(); ++i) {
    const auto& lane = req.lanes[i];
    Task task{&req, lane.epId, lane.begin, lane.end};
    futs.push_back(std::move(task.ret.get_future()));
    pool[i % pool.size()]->Submit(std::move(task));
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
  if (numSucc == static_cast<int>(req.lanes.size())) return {StatusCode::SUCCESS, ""};
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
