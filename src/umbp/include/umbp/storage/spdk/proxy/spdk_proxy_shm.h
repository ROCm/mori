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
// Shared memory region management for SPDK Proxy IPC.
// Server (proxy daemon) creates; clients attach.
//
#pragma once

#include <cstddef>
#include <string>

#include "umbp/storage/spdk/proxy/spdk_proxy_protocol.h"

namespace umbp {
namespace proxy {

class ProxyShmRegion {
 public:
  ProxyShmRegion() = default;
  ~ProxyShmRegion();

  ProxyShmRegion(const ProxyShmRegion&) = delete;
  ProxyShmRegion& operator=(const ProxyShmRegion&) = delete;

  // Server: create and initialize shared memory.
  // cache_budget_mb: total cache budget in MB (ring buffer + hash index).
  //                  0 = no shared cache.
  // Returns 0 on success, -errno on failure.
  int Create(const std::string& name, uint32_t max_channels, uint32_t max_tenants,
             size_t data_per_channel = kDefaultDataRegionPerChannel, bool try_hugepage = true,
             size_t cache_budget_mb = 0);

  // Client: attach to existing shared memory created by server.
  // Returns 0 on success, -errno on failure.
  int Attach(const std::string& name);

  // Detach (unmap). Server also unlinks the name.
  void Detach();

  bool IsValid() const { return base_ != nullptr; }
  bool IsServer() const { return is_server_; }
  bool IsHugepage() const { return is_hugepage_; }

  // Access the header
  ProxyShmHeader* Header() { return static_cast<ProxyShmHeader*>(base_); }
  const ProxyShmHeader* Header() const { return static_cast<const ProxyShmHeader*>(base_); }

  // Access a channel
  ClientChannel* Channel(uint32_t channel) { return GetChannel(base_, Header(), channel); }

  TenantInfo* Tenant(uint32_t tenant_slot) { return GetTenantInfo(base_, Header(), tenant_slot); }

  const TenantInfo* Tenant(uint32_t tenant_slot) const {
    return GetTenantInfo(base_, Header(), tenant_slot);
  }

  // Access a channel's data region
  void* DataRegion(uint32_t channel) { return proxy::GetDataRegion(base_, Header(), channel); }

  void* Base() { return base_; }
  size_t Size() const { return size_; }

  // ---- Lifecycle helpers (static, for use before attach) ----

  // Probe whether a live proxy already owns SHM |name|.
  //   1 = proxy alive and READY
  //   0 = no SHM or proxy is dead
  //  -1 = proxy alive but not READY yet
  //  -2 = proxy alive but protocol version mismatches
  static int ProbeExisting(const std::string& name);

  // Forcefully unlink any stale SHM (hugepage file + shm_open).
  static void CleanupStale(const std::string& name);

  static std::string BootstrapLockPath(const std::string& name);
  static bool ValidateHeaderLayout(const ProxyShmHeader* hdr, size_t mapped_size,
                                   std::string* error_message = nullptr);

 private:
  void* base_ = nullptr;
  size_t size_ = 0;
  std::string name_;
  std::string hp_path_;  // hugepage file path (empty if using shm_open)
  bool is_server_ = false;
  bool is_hugepage_ = false;
  int fd_ = -1;
};

}  // namespace proxy
}  // namespace umbp
