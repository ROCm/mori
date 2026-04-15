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
#include <mutex>
#include <thread>

#include "umbp/distributed/config.h"

namespace mori::umbp {

class GlobalBlockIndex;
class ClientRegistry;

class EvictionManager {
 public:
  EvictionManager(GlobalBlockIndex& index, ClientRegistry& registry, const EvictionConfig& config);
  ~EvictionManager();

  EvictionManager(const EvictionManager&) = delete;
  EvictionManager& operator=(const EvictionManager&) = delete;

  void Start();
  void Stop();

 private:
  void EvictionLoop();
  void RunOnce();

  GlobalBlockIndex& index_;
  ClientRegistry& registry_;
  EvictionConfig config_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::mutex cv_mutex_;
  std::condition_variable cv_;
};

}  // namespace mori::umbp
