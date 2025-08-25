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

#include <infiniband/verbs.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

struct IOEngineConfig {
  // Out of band TCP network configuration
  std::string host;
  uint16_t port;
};

class IOEngine;

// This is a low latency session between a pair of memory descriptor, it caches
// necessary meta data to avoid the overhead of
class IOEngineSession {
 public:
  ~IOEngineSession() = default;

  TransferUniqueId AllocateTransferUniqueId();
  void Read(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
            TransferUniqueId id);
  void Write(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
             TransferUniqueId id);

  void BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets, const SizeVec& sizes,
                 TransferStatus* status, TransferUniqueId id);
  bool Alive();

  friend class IOEngine;

 protected:
  IOEngineSession() = default;

  IOEngine* engine{nullptr};
  std::unordered_map<BackendType, BackendSession*> backendSess;
};

class IOEngine {
 public:
  IOEngine(EngineKey, IOEngineConfig);
  ~IOEngine();

  void CreateBackend(BackendType, const BackendConfig&);
  void RemoveBackend(BackendType);

  EngineDesc GetEngineDesc() const { return desc; }

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  MemoryDesc RegisterMemory(void* data, size_t size, int device, MemoryLocationType loc);
  void DeregisterMemory(const MemoryDesc& desc);

  TransferUniqueId AllocateTransferUniqueId();
  void Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
            size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id);
  void Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
             size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id);

  void BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                 const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets, const SizeVec& sizes,
                 TransferStatus* status, TransferUniqueId id);

  // Take the transfer status of an inbound op
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status);

  IOEngineSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote);
  void DestroySession(IOEngineSession*);

 public:
  // Config and descriptors
  IOEngineConfig config;
  EngineDesc desc;

 private:
  std::atomic<uint32_t> nextTransferUid{0};
  std::atomic<uint32_t> nextMemUid{0};
  std::unordered_map<MemoryUniqueId, MemoryDesc> memPool;
  std::unordered_map<BackendType, std::unique_ptr<Backend>> backends;
  std::vector<std::unique_ptr<IOEngineSession>> sessions;
};

}  // namespace io
}  // namespace mori
