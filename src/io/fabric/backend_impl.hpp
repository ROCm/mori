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

#include <hip/hip_runtime_api.h>

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "src/io/fabric/vpod_topology.hpp"
#include "src/io/xgmi/hip_resource_pool.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                  Fabric allocation helpers                                     */
/* ---------------------------------------------------------------------------------------------- */
// Allocate GPU memory that is exportable over a UALink super-node fabric, using
// the low-level VMM API (hipMemCreate + reserve + map + setAccess) with
// requestedHandleType = fabric. This is what makes a buffer registerable with
// FabricBackend: only fabric-exportable allocations can be shared cross-node.
//
// Returns the mapped device pointer (nullptr on failure). The allocation is
// tracked internally so FabricFree can tear it down. Note: do NOT hipMemset a
// fabric-exportable buffer (returns OOM on ROCm 7.15); zero it via host copies
// if needed.
void* FabricMalloc(size_t size, int device);
void FabricFree(void* ptr);

/* ---------------------------------------------------------------------------------------------- */
/*                                     FabricBackendSession                                       */
/* ---------------------------------------------------------------------------------------------- */
class FabricBackend;

class FabricBackendSession : public BackendSession {
 public:
  FabricBackendSession() = default;
  FabricBackendSession(const FabricBackendConfig& config, void* localAddr, void* remoteAddr,
                       int localDevice, FabricBackend* backend, StreamPool* streamPool,
                       EventPool* eventPool);
  ~FabricBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead) override;

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override;

  bool Alive() const override;

 private:
  // Launch the copy kernel over [srcBase+srcOff, dstBase+dstOff) for `size`
  // bytes on `stream`. Returns hipSuccess or the launch error.
  hipError_t LaunchCopy(void* dst, const void* src, size_t size, hipStream_t stream);

  FabricBackendConfig config{};
  void* localAddr{nullptr};   // local registered pointer
  void* remoteAddr{nullptr};  // imported peer VA (aliases remote HBM over fabric)
  int localDevice{-1};        // device that drives the copy kernel
  FabricBackend* backend{nullptr};
  StreamPool* streamPool{nullptr};
  EventPool* eventPool{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         FabricBackend                                          */
/* ---------------------------------------------------------------------------------------------- */
class FabricBackend : public Backend {
 public:
  FabricBackend(EngineKey, const IOEngineConfig&, const FabricBackendConfig&);
  ~FabricBackend();

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

  // JIT-loaded contiguous copy kernel (fabric_copy.hip), one module per device.
  void LoadFabricCopyModule(const std::string& hsacoPath);
  hipFunction_t GetFabricCopyFunc(int deviceId);

 private:
  void Initialize();
  std::optional<int> LookupVisibleDevice(const std::string& busId) const;
  fabric::VpodKey LocalVpodKey(int deviceId) const;

  // Import a peer's fabric handle and map it into a fresh local VA; cached per
  // (remote engine, remote mem id, local device). Returns the mapped pointer, or
  // the local pointer directly when the memory is owned by this engine.
  void* GetImportedAddress(const MemoryDesc& desc, int localDeviceId);

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
      std::size_t seed = 0;
      auto hashCombine = [](std::size_t& s, std::size_t v) {
        s ^= v + 0x9e3779b97f4a7c15ULL + (s << 6) + (s >> 2);
      };
      hashCombine(seed, std::hash<std::string>()(k.remoteEngineKey));
      hashCombine(seed, std::hash<uint64_t>()(k.localMemId));
      hashCombine(seed, std::hash<uint64_t>()(k.remoteMemId));
      return seed;
    }
  };
  FabricBackendSession* GetOrCreateSessionCached(const MemoryDesc& local, const MemoryDesc& remote);
  void InvalidateSessionsForMemory(MemoryUniqueId id);

  struct ImportedRegion {
    hipMemGenericAllocationHandle_t handle{};
    void* mappedAddr{nullptr};
    size_t size{0};
  };
  struct ImportCacheKey {
    EngineKey engineKey;
    MemoryUniqueId memId;
    int deviceId;
    bool operator==(const ImportCacheKey& o) const {
      return engineKey == o.engineKey && memId == o.memId && deviceId == o.deviceId;
    }
  };
  struct ImportCacheKeyHash {
    std::size_t operator()(const ImportCacheKey& k) const noexcept {
      std::size_t seed = 0;
      auto hashCombine = [](std::size_t& s, std::size_t v) {
        s ^= v + 0x9e3779b97f4a7c15ULL + (s << 6) + (s >> 2);
      };
      hashCombine(seed, std::hash<std::string>()(k.engineKey));
      hashCombine(seed, std::hash<uint64_t>()(k.memId));
      hashCombine(seed, std::hash<int>()(k.deviceId));
      return seed;
    }
  };

 private:
  EngineKey myEngKey;
  std::string myNodeId;
  std::string myHostname;
  FabricBackendConfig config;
  int myPid{0};

  std::unique_ptr<StreamPool> streamPool;
  std::unique_ptr<EventPool> eventPool;

  int numDevices{0};
  std::vector<char> deviceSupportsFabric;       // indexed by device id
  std::vector<fabric::VpodKey> deviceVpodKeys;  // indexed by device id
  std::unordered_map<std::string, int> localDeviceByBusId;

  mutable std::shared_mutex importMutex;
  std::unordered_map<ImportCacheKey, ImportedRegion, ImportCacheKeyHash> importedRegions;

  std::unordered_map<SessionCacheKey, std::unique_ptr<FabricBackendSession>, SessionCacheKeyHash>
      sessionCache;
  std::mutex sessionCacheMu;

  std::unordered_map<EngineKey, EngineDesc> remoteEngines;
  mutable std::mutex remoteEnginesMu;

  std::string fabricCopyHsacoPath_;
  std::vector<hipModule_t> fabricCopyModules_;
  std::vector<hipFunction_t> fabricCopyFuncs_;
  std::mutex moduleMu_;
};

}  // namespace io
}  // namespace mori
