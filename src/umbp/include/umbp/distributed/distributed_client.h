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

#include "umbp/umbp_client.h"

namespace mori::io {
class IOEngine;
}

namespace mori::umbp {

class MasterClient;

/// Distributed IUMBPClient implementation — master-led global routing
/// with RDMA/MORI-IO data plane.  All routing decisions go through the
/// Master; this client does not use LocalStorageManager or LocalBlockIndex.
class DistributedClient : public IUMBPClient {
 public:
  explicit DistributedClient(const UMBPConfig& config);
  ~DistributedClient() override;

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

  bool ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) override;
  bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes) override;
  std::vector<ExternalKvMatch> MatchExternalKv(const std::vector<std::string>& hashes) override;

 private:
  UMBPConfig config_;
  bool closed_ = false;

  std::unique_ptr<MasterClient> master_client_;
  std::unique_ptr<mori::io::IOEngine> io_engine_;
};

}  // namespace mori::umbp
