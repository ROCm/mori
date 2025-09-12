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

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/io/common.hpp"
#include "src/io/rdma/common.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Data Structures                                        */
/* ---------------------------------------------------------------------------------------------- */
struct ExecutorReq {
  const EpPairVec& eps;
  const application::RdmaMemoryRegion& local;
  const SizeVec& localOffsets;
  const application::RdmaMemoryRegion& remote;
  const SizeVec& remoteOffsets;
  const SizeVec& sizes;
  CqCallbackMeta* callbackMeta;
  TransferUniqueId id;
  int postBatchSize;
  bool isRead;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                            Executor                                            */
/* ---------------------------------------------------------------------------------------------- */
class Executor {
 public:
  Executor() = default;
  virtual ~Executor() = default;

  virtual void Start() = 0;
  virtual void Shutdown() = 0;
  virtual RdmaOpRet RdmaBatchReadWrite(const ExecutorReq& req) = 0;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       MultithreadExecutor                                      */
/* ---------------------------------------------------------------------------------------------- */
class MultithreadExecutor : public Executor {
 public:
  MultithreadExecutor(int numWorker);
  ~MultithreadExecutor();

  RdmaOpRet RdmaBatchReadWrite(const ExecutorReq& req);
  void Start();
  void Shutdown();

 private:
  struct Task {
    const ExecutorReq* req;
    int epId{-1};
    int begin{-1};
    int end{-1};
    std::promise<RdmaOpRet> ret;

    Task(const ExecutorReq* req_, int epId_, int begin_, int end_)
        : req(req_), epId(epId_), begin(begin_), end(end_) {}
  };

  std::vector<std::pair<int, int>> SplitWork(const ExecutorReq& req);

  class Worker {
   public:
    Worker(int wid);
    ~Worker();
    void MainLoop();
    void Start();
    void Shutdown();

    void Submit(Task&&);

   private:
    int workerId{-1};
    std::atomic<bool> running{false};
    mutable std::mutex mu;
    std::condition_variable cond;
    std::queue<Task> q;
    std::thread thd;
  };

 public:
  int numWorker{1};

 private:
  std::atomic<bool> running{false};
  std::vector<std::unique_ptr<Worker>> pool;
};

}  // namespace io
}  // namespace mori
