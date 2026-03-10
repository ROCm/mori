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

#include <string>
#include <vector>

#include "umbp/block_index/block_index.h"
#include "umbp/common/config.h"
#include "umbp/storage/local_storage_manager.h"

class UMBPClient {
 public:
  explicit UMBPClient(const UMBPConfig& config = UMBPConfig{});

  // Core API
  bool Put(const std::string& key, const void* data, size_t size);
  bool PutFromPtr(const std::string& key, uintptr_t src, size_t size);
  bool GetIntoPtr(const std::string& key, uintptr_t dst, size_t size);
  bool Exists(const std::string& key) const;
  bool Remove(const std::string& key);

  // Batch API
  std::vector<bool> BatchPutFromPtr(const std::vector<std::string>& keys,
                                    const std::vector<uintptr_t>& ptrs,
                                    const std::vector<size_t>& sizes);
  std::vector<bool> BatchGetIntoPtr(const std::vector<std::string>& keys,
                                    const std::vector<uintptr_t>& ptrs,
                                    const std::vector<size_t>& sizes);
  std::vector<bool> BatchExists(const std::vector<std::string>& keys) const;

  void Clear();

  // Access sub-modules (for testing/debugging)
  BlockIndexClient& Index();
  LocalStorageManager& Storage();

 private:
  static UMBPConfig NormalizeConfig(const UMBPConfig& config);

  UMBPConfig config_;
  UMBPRole role_;
  BlockIndexClient index_;
  LocalStorageManager storage_;
};
