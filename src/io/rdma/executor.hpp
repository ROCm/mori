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
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/io/common.hpp"
#include "src/io/rdma/common.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Data Structures                                        */
/* ---------------------------------------------------------------------------------------------- */
struct AdmissionToken {
  std::shared_ptr<SqController> sq;
  int estimatedWr{0};
  bool active{false};

  AdmissionToken() = default;
  AdmissionToken(std::shared_ptr<SqController> sq_, int wr);
  AdmissionToken(const AdmissionToken&) = delete;
  AdmissionToken& operator=(const AdmissionToken&) = delete;
  AdmissionToken(AdmissionToken&& rhs) noexcept;
  AdmissionToken& operator=(AdmissionToken&& rhs) noexcept;
  ~AdmissionToken();

  void Release();
};

struct ExecutorReq {
  const EpPairVec& eps;
  const application::RdmaMemoryRegion& local;
  const SizeVec& localOffsets;
  const application::RdmaMemoryRegion& remote;
  const SizeVec& remoteOffsets;
  const SizeVec& sizes;
  std::shared_ptr<CqCallbackMeta> callbackMeta;
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
    const ExecutorReq* req{nullptr};
    int epId{-1};
    int begin{-1};
    int end{-1};
    AdmissionToken admissionToken;
    std::promise<RdmaOpRet> ret;

    Task() = default;
    Task(const ExecutorReq* req_, int epId_, int begin_, int end_)
        : req(req_), epId(epId_), begin(begin_), end(end_) {}
    Task(const ExecutorReq* req_, int epId_, int begin_, int end_, AdmissionToken token)
        : req(req_), epId(epId_), begin(begin_), end(end_), admissionToken(std::move(token)) {}
  };

  struct WorkSplit {
    int workerId{-1};
    int epId{-1};
    int begin{-1};
    int end{-1};
    int admissionWr{0};
  };

  std::vector<WorkSplit> SplitWork(const ExecutorReq& req);
  RdmaOpRet RdmaBatchReadWriteWithAdmission(const ExecutorReq& req);

  friend void TestWorkerShutdownDrainsTokensAndPromisesForTest();

  class Worker {
   public:
    Worker(int wid);
    ~Worker();
    void MainLoop();
    void Start();
    void Shutdown();

    void Submit(Task&&);

   private:
    friend void TestWorkerShutdownDrainsTokensAndPromisesForTest();

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
