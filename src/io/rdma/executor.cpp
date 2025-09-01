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

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                   MultithreadExecutor::Worker                                  */
/* ---------------------------------------------------------------------------------------------- */
MultithreadExecutor::Worker::Worker() {}

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
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mu);
      cond.wait(lock, [this]() { return !q.empty() || !running.load(); });

      if (!running.load()) break;

      while (!q.empty()) {
        Task task = q.front();
        q.pop();

        SizeVec tLoclOffsets(task.req.localOffsets.begin() + task.begin,
                             task.req.localOffsets.begin() + task.end);
        SizeVec tRemoteOffsets(task.req.remoteOffsets.begin() + task.begin,
                               task.req.remoteOffsets.begin() + task.end);
        SizeVec tSizes(task.req.sizes.begin() + task.begin, task.req.sizes.begin() + task.end);
        mori::io::RdmaBatchReadWrite({task.req.eps[task.epId]}, task.req.local, tLoclOffsets,
                                     task.req.remote, tRemoteOffsets, tSizes, task.req.status,
                                     task.req.id, task.req.isRead, task.expectedNumCqe,
                                     task.req.postBatchSize);
      }
    }
  }
}

void MultithreadExecutor::Worker::Submit(Task task) {
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) {
      task.status.SetCode(StatusCode::ERR_BAD_STATE);
      task.status.SetMessage("worker not started yet");
      return;
    }
    q.push(task);
    cond.notify_all();
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       MultithreadExecutor                                      */
/* ---------------------------------------------------------------------------------------------- */
MultithreadExecutor::MultithreadExecutor(int n) : numWorker(n), pool(5) { assert(n > 0); }

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

void MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  std::vector<std::unique_ptr<TransferStatus>> resps;
  for (int i = 0; i < numWorker; i++) resps.emplace_back(new TransferStatus());

  auto splits = SplitWork(req);
  int expectedNumCqe = splits.size();
  for (int i = 0; i < splits.size(); i++) {
    pool[i].Submit({req, *resps[i], i, splits[i].first, splits[i].second, expectedNumCqe});
  }

  bool hasFail = false;
  int numSucc = 0;
  for (auto& status : resps) {
    while (status->Init()) {
    }
    if (status->Failed()) {
      hasFail = true;
      req.status->SetCode(status->Code());
      req.status->SetMessage(status->Message());
    } else if (status->Succeeded()) {
      numSucc++;
    }
  }
  if (hasFail) return;

  if (numSucc == numWorker) {
    req.status->SetCode(StatusCode::SUCCESS);
    return;
  }

  req.status->SetCode(StatusCode::IN_PROGRESS);
}

void MultithreadExecutor::Start() {
  for (auto& worker : pool) {
    worker.Start();
  }
}

void MultithreadExecutor::Shutdown() {
  for (auto& worker : pool) {
    worker.Shutdown();
  }
}

}  // namespace io
}  // namespace mori
