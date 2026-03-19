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
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/storage/local_storage_manager.h"

class CopyPipeline {
 public:
  CopyPipeline(LocalStorageManager& storage, const UMBPCopyPipelineConfig& config, UMBPRole role);
  ~CopyPipeline();

  bool MaybeCopyToSharedSSD(const std::string& key);
  void MaybeBatchCopyToSharedSSD(const std::vector<std::string>& keys);

 private:
  struct CopyTask {
    std::string key;
  };

  bool EnqueueCopyToSSD(const std::string& key);
  size_t EnqueueCopyToSSDBatch(const std::vector<std::string>& keys);
  void CopyWorkerLoop();

  LocalStorageManager& storage_;
  UMBPCopyPipelineConfig config_;
  UMBPRole role_;

  std::atomic<bool> stop_copy_worker_{false};
  std::vector<std::thread> copy_workers_;
  std::mutex copy_mu_;
  std::condition_variable copy_cv_;
  std::deque<CopyTask> copy_queue_;
};
