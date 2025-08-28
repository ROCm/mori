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
/*                                       MultithreadExecutor                                      */
/* ---------------------------------------------------------------------------------------------- */
MultithreadExecutor::MultithreadExecutor(int n) : numThd(n) {}

MultithreadExecutor::~MultithreadExecutor() { Shutdown(); }

void MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  assert(running.load());
  std::lock_guard<std::mutex> lock(mu);
  std::vector<TransferStatus> statusVec(numThd, TransferStatus{});
  q.push({req, statusVec});

  bool hasFail = false;
  int numSucc = 0;
  for (auto& status : statusVec) {
    while (status.Init()) {
    }
    if (status.Failed()) {
      hasFail = true;
      req.status->Code(status.Code());
      req.status->Message(status.Message());
    } else if (status.Succeeded()) {
      numSucc++;
    }
  }
  if (hasFail) return;

  if (numSucc == numThd) {
    req.status->Code(StatusCode::SUCCESS);
    return;
  }

  req.status->Code(StatusCode::IN_PROGRESS);
}

void MultithreadExecutor::Start() {
  if (running.load()) return;
  running.store(true);
  for (int i = 0; i < numThd; i++) {
    std::thread worker(MainLoop);
    pool.push_back(std::move(worker));
  }
}

void MultithreadExecutor::Shutdown() {
  running.store(false);
  if (thd.joinable()) thd.join();
}

void MultithreadExecutor::MainLoop() {}

}  // namespace io
}  // namespace mori
