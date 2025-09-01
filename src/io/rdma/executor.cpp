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
  MORI_IO_INFO("worker {} enter main loop", workerId);
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mu);
      cond.wait(lock, [this]() { return !q.empty() || !running.load(); });

      if (!running.load()) {
        MORI_IO_INFO("worker {} shutdown", workerId);
        break;
      }

      while (!q.empty()) {
        Task task = q.front();
        q.pop();

        SizeVec tLoclOffsets(task.req.localOffsets.begin() + task.begin,
                             task.req.localOffsets.begin() + task.end);
        SizeVec tRemoteOffsets(task.req.remoteOffsets.begin() + task.begin,
                               task.req.remoteOffsets.begin() + task.end);
        SizeVec tSizes(task.req.sizes.begin() + task.begin, task.req.sizes.begin() + task.end);
        task.ret = mori::io::RdmaBatchReadWrite(
            {task.req.eps[task.epId]}, task.req.local, tLoclOffsets, task.req.remote,
            tRemoteOffsets, tSizes, task.req.status, task.req.id, task.req.isRead,
            task.expectedNumCqe, task.req.postBatchSize);
        MORI_IO_TRACE("Worker {} execute task {} begin {} end {} ret code {}", workerId,
                      task.req.id, task.begin, task.end, static_cast<uint32_t>(task.ret.code));
      }
    }
  }
}

void MultithreadExecutor::Worker::Submit(Task task) {
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) {
      task.ret.code = StatusCode::ERR_BAD_STATE;
      task.ret.message = "worker not started yet";
      return;
    }
    q.push(task);
    cond.notify_all();
  }
  MORI_IO_TRACE("Submit to worker {} task {} begin {} end {}", workerId, task.req.id, task.begin,
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

RdmaOpRet MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  auto splits = SplitWork(req);
  int numSplits = splits.size();
  std::vector<RdmaOpRet> rets(numSplits);

  for (int i = 0; i < numSplits; i++) {
    pool[i]->Submit({req, rets[i], i, splits[i].first, splits[i].second, numSplits});
  }

  bool hasFail = false;
  int numSucc = 0;
  RdmaOpRet failedRet;
  for (auto& ret : rets) {
    while (ret.Init()) {
    }
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
