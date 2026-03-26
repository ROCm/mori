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
//
// SpdkProxyTier: TierBackend that communicates with an external spdk_proxy
// daemon via POSIX shared memory. Does NOT depend on SPDK headers/libraries.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/local/storage/tier_backend.h"
#include "umbp/proxy/spdk_proxy_protocol.h"
#include "umbp/proxy/spdk_proxy_shm.h"

namespace mori {
namespace umbp {

class SpdkProxyTier : public TierBackend {
 public:
  explicit SpdkProxyTier(const UMBPConfig& config);
  ~SpdkProxyTier() override;

  bool IsValid() const { return connected_; }
  uint32_t channel_id() const { return channel_id_; }
  uint32_t rank_id() const { return channel_id_; }  // legacy alias
  uint32_t tenant_id() const { return tenant_id_; }

  bool Write(const std::string& key, const void* data, size_t size) override;
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;
  TierCapabilities Capabilities() const override;

  bool WriteBatch(const std::vector<std::string>& keys, const std::vector<const void*>& data_ptrs,
                  const std::vector<size_t>& sizes) override;

  std::vector<bool> BatchWrite(const std::vector<std::string>& keys,
                               const std::vector<const void*>& data_ptrs,
                               const std::vector<size_t>& sizes) override;

  std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;

  std::vector<bool> BatchReadIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;

  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;

  bool Flush() override;
  void SetColdRead(bool enable) override { cold_read_ = enable; }

  static bool WaitForProxy(const std::string& shm_name, int timeout_ms);

 private:
  ::umbp::proxy::ResultCode SubmitAndWait(::umbp::proxy::RequestType type, const std::string& key,
                                          const void* write_data, size_t write_size,
                                          void* read_buf, size_t read_buf_size,
                                          uint64_t request_aux = 0,
                                          uint64_t* out_result_size = nullptr,
                                          uint64_t* out_result_aux = nullptr) const;

  std::vector<bool> SubmitBatch(::umbp::proxy::RequestType type,
                                const std::vector<std::string>& keys,
                                const std::vector<const void*>& data_ptrs,
                                const std::vector<uintptr_t>& dst_ptrs,
                                const std::vector<size_t>& sizes) const;

  uint32_t NextSeqId() const;
  bool IsProxyAlive() const;
  void ReleaseChannel() const;

  // Per-item ring cache read — seqlock, returns true on hit.
  bool TryShmCacheReadOne(const std::string& key, uintptr_t dst, size_t size) const;

  bool connected_ = false;
  bool cold_read_ = false;  // SetColdRead(true): skip ring cache, always read from NVMe
  uint32_t channel_id_ = 0;
  uint32_t tenant_id_ = 0;
  uint32_t tenant_slot_ = 0;
  uint64_t session_id_ = 0;
  mutable ::umbp::proxy::ProxyShmRegion shm_;
  mutable std::mutex submit_mu_;
  mutable uint64_t seq_counter_ = 0;
};

}  // namespace umbp
}  // namespace mori
