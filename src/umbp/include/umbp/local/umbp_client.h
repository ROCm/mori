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

#include "umbp/common/config.h"
#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/storage/copy_pipeline.h"
#include "umbp/local/storage/local_storage_manager.h"

namespace mori::umbp {

class PoolClient;  // forward declaration — full include in .cpp only
class PeerServiceServer;

class UMBPClient {
 public:
  explicit UMBPClient(const UMBPConfig& config = UMBPConfig{});
  ~UMBPClient();

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
  // Depth-aware variant: depths[i] is the chain depth for keys[i].
  // depth == -1 (or empty depths vector) means no metadata — falls back to plain LRU.
  std::vector<bool> BatchPutFromPtrWithDepth(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& ptrs,
                                             const std::vector<size_t>& sizes,
                                             const std::vector<int>& depths);
  std::vector<bool> BatchGetIntoPtr(const std::vector<std::string>& keys,
                                    const std::vector<uintptr_t>& ptrs,
                                    const std::vector<size_t>& sizes);
  std::vector<bool> BatchExists(const std::vector<std::string>& keys) const;
  // Returns the number of keys that exist consecutively from index 0.
  // Stops at the first key that does not exist (early-stop).
  size_t BatchExistsConsecutive(const std::vector<std::string>& keys) const;

  void Clear();

  // Ensure all pending write-back data is persisted and visible to other ranks.
  // Must be called before any cross-rank read barrier in write-back mode.
  bool Flush();

  // Access sub-modules (for testing/debugging)
  mori::umbp::LocalBlockIndex& Index();
  LocalStorageManager& Storage();

  // Returns true when distributed mode is active (PoolClient connected to Master).
  bool IsDistributed() const;

 private:
  static UMBPConfig NormalizeConfig(const UMBPConfig& config);

  // Publish a locally-written block to the Master for remote discovery.
  void MaybePublishLocal(const std::string& key, size_t size);

  UMBPConfig config_;
  UMBPRole role_;
  mori::umbp::LocalBlockIndex index_;
  LocalStorageManager storage_;
  std::unique_ptr<CopyPipeline> copy_pipeline_;
  std::unique_ptr<PoolClient> pool_client_;  // non-null iff distributed mode
  std::unique_ptr<PeerServiceServer> peer_service_;
  size_t dram_chunk_size_ = 0;  // effective DRAM MR chunk size from PoolClient
};

}  // namespace mori::umbp
