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
  std::vector<bool> BatchGetRanges(const std::vector<std::string>& keys,
                                   const std::vector<std::vector<uintptr_t>>& dsts,
                                   const std::vector<std::vector<size_t>>& sizes,
                                   const std::vector<std::vector<size_t>>& src_offsets) override;
  std::vector<bool> BatchPutRanges(const std::vector<std::string>& keys,
                                   const std::vector<size_t>& object_sizes,
                                   const std::vector<std::vector<uintptr_t>>& srcs,
                                   const std::vector<std::vector<size_t>>& sizes,
                                   const std::vector<std::vector<size_t>>& dst_offsets) override;
  std::vector<bool> BatchExists(const std::vector<std::string>& keys) const override;
  size_t BatchExistsConsecutive(const std::vector<std::string>& keys) const override;

  bool Clear() override;
  bool Flush() override;
  void Close() override;
  bool IsDistributed() const override { return false; }
  UMBPDeploymentMode GetDeploymentMode() const override {
    return UMBPDeploymentMode::StandaloneProcess;
  }
  UMBPDeploymentMode GetBackendMode() const override { return backend_mode_; }
  bool SupportsRangedIO() const override { return supports_ranged_io_; }

  // `mode` is accepted and ignored: this client owns nothing to pin. It forwards
  // the region to the server, which decides how to declare it to the backend.
  bool RegisterMemory(uintptr_t ptr, size_t size,
                      mori::io::MemoryLocationType loc = mori::io::MemoryLocationType::CPU,
                      int device = -1,
                      MemoryRegistration mode = MemoryRegistration::kPinned) override;
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
  // Not const: a successful Ping is where the server's backend mode and ranged
  // capability become known, and they are cached on the client.
  bool WaitReady(int timeout_ms);
  void MaybeAutoStart();
  std::string ClientId();
  void DeregisterMemoryLocked();
  bool RegisterDeviceMemory(uintptr_t ptr, size_t size, int device_id);
  bool RegisterHostShmMemory(uintptr_t ptr, size_t size);

  UMBPConfig config_;
  UMBPStandaloneProcessConfig standalone_config_;
  std::string address_;
  std::string fd_socket_path_;
  std::shared_ptr<::grpc::ChannelInterface> channel_;
  std::unique_ptr<::umbp::UMBPStandalone::Stub> stub_;

  mutable std::shared_mutex op_mutex_;
  std::atomic<bool> closing_{false};
  bool closed_ = false;
  UMBPDeploymentMode backend_mode_ = UMBPDeploymentMode::StandaloneProcess;
  bool supports_ranged_io_ = false;

  enum class RegionKind { kHostShm, kGpuIpc };

  // A hybrid worker can register several non-contiguous host or GPU regions.
  // `base` is the worker VA used as region_base in data requests.
  struct RegisteredRegion {
    uintptr_t base = 0;
    size_t size = 0;
    RegionKind kind = RegionKind::kHostShm;
  };

  mutable std::mutex registration_mu_;
  std::string client_id_;
  std::vector<RegisteredRegion> regions_;

  // The key lists this client has already sent, and the handles the server
  // gave back for them.
  //
  // A layer-wise restore asks about one key set once per layer group, so the
  // keys are the only part of a ranged get that does not change between those
  // calls -- and at a thousand-odd ~128-byte keys they are the expensive part
  // to serialize. Naming a remembered list instead costs a comparison against
  // what was sent last time.
  //
  // Matching is by full equality of the key vector, not by hash, so nothing
  // here can select the wrong list: the fingerprint travels only so the server
  // can make the same check on its side. A handle the server has dropped comes
  // back as key_handle_unknown, the entry is discarded, and the call is simply
  // repeated carrying the keys.
  struct KeyHandle {
    std::vector<std::string> keys;
    uint64_t handle = 0;
    uint64_t fingerprint = 0;
  };

  // A restore has one key set in flight per pool it reads. Small enough that a
  // linear scan is cheaper than any index, and a miss only costs a resend.
  static constexpr size_t kKeyHandleSlots = 8;

  // Returns 0 when this set has not been sent before, and fills *fingerprint
  // either way.
  uint64_t LookupKeyHandle(const std::vector<std::string>& keys, uint64_t* fingerprint);
  void RememberKeyHandle(const std::vector<std::string>& keys, uint64_t handle,
                         uint64_t fingerprint);
  void ForgetKeyHandle(uint64_t handle);

  std::mutex key_handle_mu_;
  std::vector<KeyHandle> key_handles_;  // most recently used first
};

}  // namespace mori::umbp::standalone
