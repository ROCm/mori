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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "umbp/umbp_client.h"
#include "umbp_standalone.grpc.pb.h"

namespace mori::umbp::standalone {

class StandaloneProcessClient : public IUMBPClient {
 public:
  explicit StandaloneProcessClient(const UMBPConfig& config);
  ~StandaloneProcessClient() override;

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

  bool Clear() override;
  bool Flush() override;
  void Close() override;
  bool IsDistributed() const override { return false; }
  UMBPDeploymentMode GetDeploymentMode() const override {
    return UMBPDeploymentMode::StandaloneProcess;
  }

  bool RegisterMemory(uintptr_t ptr, size_t size) override;
  void DeregisterMemory(uintptr_t ptr) override;

  bool ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) override;
  bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) override;
  bool RevokeAllExternalKvBlocksAtTier(TierType tier) override;
  std::vector<ExternalKvMatch> MatchExternalKv(const std::vector<std::string>& hashes,
                                               bool count_as_hit = false) override;
  std::vector<ExternalKvHitCountEntry> GetExternalKvHitCounts(
      const std::vector<std::string>& hashes) override;

 private:
  // Resolves `ptr` against the registered host regions. On success writes the
  // region-relative `offset` and the matched region's worker VA `region_base`.
  bool OffsetFor(uintptr_t ptr, size_t size, uint64_t* offset, uint64_t* region_base) const;
  bool WaitReady(int timeout_ms) const;
  void MaybeAutoStart();
  std::string ClientId();
  void DeregisterMemoryLocked();

  UMBPConfig config_;
  UMBPStandaloneProcessConfig standalone_config_;
  std::string address_;
  std::string fd_socket_path_;
  std::shared_ptr<::grpc::ChannelInterface> channel_;
  std::unique_ptr<::umbp::UMBPStandalone::Stub> stub_;

  mutable std::shared_mutex op_mutex_;
  std::atomic<bool> closing_{false};
  bool closed_ = false;

  // One registered host shared-memory region. A hybrid HiCache worker (e.g.
  // DeepSeek-V4) registers several non-contiguous host pools per rank, so the
  // client tracks N regions and resolves each data op to the one that owns its
  // pointer. `base` is the worker VA base (== allocation->base), `size` its
  // mapped size.
  struct RegisteredRegion {
    uintptr_t base = 0;
    size_t size = 0;
  };

  mutable std::mutex registration_mu_;
  std::string client_id_;
  std::vector<RegisteredRegion> regions_;
};

}  // namespace mori::umbp::standalone
