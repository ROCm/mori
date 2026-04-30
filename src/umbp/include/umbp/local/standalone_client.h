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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/tiers/copy_pipeline.h"
#include "umbp/local/tiers/local_storage_manager.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {

/// Standalone IUMBPClient implementation — purely local DRAM + SSD storage
/// with no networking or master coordination.
class StandaloneClient : public IUMBPClient {
 public:
  explicit StandaloneClient(const UMBPConfig& config = UMBPConfig{});
  ~StandaloneClient() override;

  // ---- IUMBPClient interface ----
  bool Put(const std::string& key, uintptr_t src, size_t size) override;
  bool Get(const std::string& key, uintptr_t dst, size_t size) override;
  bool Exists(const std::string& key) const override;

  std::vector<bool> BatchPut(const std::vector<std::string>& keys,
                             const std::vector<uintptr_t>& srcs,
                             const std::vector<size_t>& sizes) override;
  std::vector<bool> BatchPutWithDepth(const std::vector<std::string>& keys,
                                      const std::vector<uintptr_t>& srcs,
                                      const std::vector<size_t>& sizes,
                                      const std::vector<int>& depths) override;
  std::vector<bool> BatchGet(const std::vector<std::string>& keys,
                             const std::vector<uintptr_t>& dsts,
                             const std::vector<size_t>& sizes) override;
  std::vector<bool> BatchExists(const std::vector<std::string>& keys) const override;
  size_t BatchExistsConsecutive(const std::vector<std::string>& keys) const override;

  void Clear() override;
  bool Flush() override;
  void Close() override;
  bool IsDistributed() const override;

  bool ReportExternalKvBlocks(const std::vector<std::string>& /*hashes*/,
                              TierType /*tier*/) override {
    return true;
  }
  bool RevokeExternalKvBlocks(const std::vector<std::string>& /*hashes*/) override { return true; }
  std::vector<ExternalKvMatch> MatchExternalKv(
      const std::vector<std::string>& /*hashes*/) override {
    return {};
  }

  // ---- Extra methods (not in IUMBPClient, for C++ tests and debugging) ----

  bool Put(const std::string& key, const void* data, size_t size);
  bool Remove(const std::string& key);

  mori::umbp::LocalBlockIndex& Index();
  LocalStorageManager& Storage();

 private:
  static UMBPConfig NormalizeConfig(const UMBPConfig& config);

  UMBPConfig config_;
  UMBPRole role_;
  mori::umbp::LocalBlockIndex index_;
  LocalStorageManager storage_;
  std::unique_ptr<CopyPipeline> copy_pipeline_;
  bool closed_ = false;
};

}  // namespace mori::umbp
