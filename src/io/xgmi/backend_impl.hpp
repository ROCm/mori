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

#include <hip/hip_runtime.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "src/io/xgmi/hip_resource_pool.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                        XgmiBackendSession                                      */
/* ---------------------------------------------------------------------------------------------- */
class XgmiBackendSession : public BackendSession {
 public:
  XgmiBackendSession() = default;
  XgmiBackendSession(const XgmiBackendConfig& config, void* localAddr, void* remoteAddr,
                     int localDevice, int remoteDevice, StreamPool* streamPool,
                     EventPool* eventPool);
  ~XgmiBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead) override;

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override;

  bool Alive() const override;

 private:
  XgmiBackendConfig config{};
  void* localAddr{nullptr};
  void* remoteAddr{nullptr};
  int localDevice{-1};
  int remoteDevice{-1};
  StreamPool* streamPool{nullptr};
  EventPool* eventPool{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           XgmiBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

class XgmiBackend : public Backend {
 public:
  XgmiBackend(EngineKey, const IOEngineConfig&, const XgmiBackendConfig&);
  ~XgmiBackend();

  void RegisterRemoteEngine(const EngineDesc&) override;
  void DeregisterRemoteEngine(const EngineDesc&) override;
  void RegisterMemory(MemoryDesc& desc) override;
  void DeregisterMemory(const MemoryDesc& desc) override;
  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead) override;
  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override;
  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) override;
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                TransferStatus* status) override;
  bool CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const override;

  bool IsP2PAccessible(int srcDevice, int dstDevice) const;

 private:
  void InitializeP2PAccess();
  void* GetRemappedAddress(const MemoryDesc& desc);

  struct SessionCacheKey {
    EngineKey remoteEngineKey;
    MemoryUniqueId localMemId;
    MemoryUniqueId remoteMemId;
    bool operator==(const SessionCacheKey& o) const {
      return remoteEngineKey == o.remoteEngineKey && localMemId == o.localMemId &&
             remoteMemId == o.remoteMemId;
    }
  };
  struct SessionCacheKeyHash {
    std::size_t operator()(const SessionCacheKey& k) const noexcept {
      auto hash_combine = [](std::size_t& seed, std::size_t v) {
        seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      };
      std::size_t seed = 0;
      hash_combine(seed, std::hash<std::string>()(k.remoteEngineKey));
      hash_combine(seed, std::hash<uint64_t>()(k.localMemId));
      hash_combine(seed, std::hash<uint64_t>()(k.remoteMemId));
      return seed;
    }
  };
  XgmiBackendSession* GetOrCreateSessionCached(const MemoryDesc& local, const MemoryDesc& remote);
  void InvalidateSessionsForMemory(MemoryUniqueId id);

 private:
  EngineKey myEngKey;
  std::string myHostname;
  XgmiBackendConfig config;

  std::unique_ptr<StreamPool> streamPool;
  std::unique_ptr<EventPool> eventPool;

  int numDevices{0};
  std::vector<std::vector<bool>> p2pMatrix;

  struct IpcHandleEntry {
    hipIpcMemHandle_t handle;
    void* remappedAddr{nullptr};
    size_t size{0};
  };
  mutable std::shared_mutex ipcMutex;
  std::unordered_map<MemoryUniqueId, hipIpcMemHandle_t> localIpcHandles;
  std::unordered_map<MemoryUniqueId, IpcHandleEntry> remoteIpcHandles;

  std::unordered_map<SessionCacheKey, std::unique_ptr<XgmiBackendSession>, SessionCacheKeyHash>
      sessionCache;
  std::mutex sessionCacheMu;

  std::unordered_map<EngineKey, EngineDesc> remoteEngines;
  mutable std::mutex remoteEnginesMu;
};

}  // namespace io
}  // namespace mori
