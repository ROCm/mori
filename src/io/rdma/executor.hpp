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
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

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
  std::shared_ptr<CqCallbackMeta> callbackMeta;
  TransferUniqueId id;
  int postBatchSize;
  bool isRead;
  size_t chunkBytes{0};
  int maxChunks{1};
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
  // Single fork-join completion primitive shared by all splits of ONE
  // RdmaBatchReadWrite call. Replaces the previous per-split
  // std::promise/std::future: those heap-allocated an atomically-refcounted
  // shared state (+mutex/condvar) PER split and forced the caller to block
  // (futex_wait) once per future. Here the caller blocks at most once and there
  // is no per-split heap allocation. Lives on the submitting thread's stack and
  // outlives all its splits because Wait() only returns after every split has
  // called Complete() (so no worker touches it afterward).
  struct BatchLatch {
    explicit BatchLatch(int total) : remaining(total) {}

    // Called once by whichever thread finishes a split (worker, or the submit
    // path on failure). notify happens under mu, and the waiter re-acquires mu
    // before returning, so the notifier is fully done with the latch by the
    // time Wait() returns and the object is destroyed.
    void Complete(const RdmaOpRet& ret) {
      std::lock_guard<std::mutex> lock(mu);
      if (ret.Failed()) {
        if (!anyFail) {
          anyFail = true;
          failedRet = ret;
        }
      } else if (ret.Succeeded()) {
        ++numSucc;
      }
      if (--remaining == 0) cv.notify_one();
    }

    RdmaOpRet Wait(int total) {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [this] { return remaining == 0; });
      if (anyFail) return std::move(failedRet);
      if (numSucc == total) return {StatusCode::SUCCESS, ""};
      return {StatusCode::IN_PROGRESS, ""};
    }

    std::mutex mu;
    std::condition_variable cv;
    int remaining{0};
    int numSucc{0};
    bool anyFail{false};
    RdmaOpRet failedRet;
  };

  struct Task {
    const ExecutorReq* req{nullptr};
    int epId{-1};
    int begin{-1};
    int end{-1};
    BatchLatch* latch{nullptr};

    Task() = default;
    Task(const ExecutorReq* req_, int epId_, int begin_, int end_, BatchLatch* latch_)
        : req(req_), epId(epId_), begin(begin_), end(end_), latch(latch_) {}
  };

  class Worker {
   public:
    Worker(int wid);
    ~Worker();
    void MainLoop();
    void Start();
    void Shutdown();

    void Submit(Task&&);

   private:
    // Grow the ring to the next power-of-two capacity, re-linearizing live
    // entries [head, tail) into [0, count). Caller must hold mu.
    void GrowRing();

    int workerId{-1};
    std::atomic<bool> running{false};
    mutable std::mutex mu;
    std::condition_variable cond;
    // Preallocated power-of-two ring buffer replacing std::queue<Task> (deque):
    // no per-Submit/pop malloc/free node churn. Free-running head/tail masked by
    // (capacity-1); grows by doubling only when full (rare, amortized), so the
    // steady state is allocation-free. All access is under mu (MPSC).
    std::vector<Task> ring;
    size_t head{0};
    size_t tail{0};
    size_t count{0};
    size_t mask{0};
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
