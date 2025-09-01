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

#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"

namespace mori {
namespace io {

class IOEngineConfig;

/* ---------------------------------------------------------------------------------------------- */
/*                                          BackendConfig                                         */
/* ---------------------------------------------------------------------------------------------- */
struct BackendConfig {
  BackendConfig(BackendType t) : type(t) {}
  ~BackendConfig() = default;

  BackendType Type() const { return type; }

 private:
  BackendType type;
};

struct RdmaBackendConfig : public BackendConfig {
  RdmaBackendConfig() : BackendConfig(BackendType::RDMA) {}
  RdmaBackendConfig(int qpPerTransfer_, int postBatchSize_, int numWorkerThreads_)
      : BackendConfig(BackendType::RDMA),
        qpPerTransfer(qpPerTransfer_),
        postBatchSize(postBatchSize_),
        numWorkerThreads(numWorkerThreads_) {}

  int qpPerTransfer{1};
  int postBatchSize{-1};
  int numWorkerThreads{1};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         BackendSession                                         */
/* ---------------------------------------------------------------------------------------------- */
class BackendSession {
 public:
  BackendSession() = default;
  virtual ~BackendSession() = default;

  virtual void Read(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) = 0;
  virtual void Write(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) = 0;

  virtual void BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) = 0;

  virtual bool Alive() const = 0;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                             Backend                                            */
/* ---------------------------------------------------------------------------------------------- */
class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;

  virtual void RegisterRemoteEngine(const EngineDesc&) = 0;
  virtual void DeregisterRemoteEngine(const EngineDesc&) = 0;

  virtual void RegisterMemory(const MemoryDesc& desc) = 0;
  virtual void DeregisterMemory(const MemoryDesc& desc) = 0;

  virtual void Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) = 0;
  virtual void Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) = 0;

  virtual void BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                         const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) = 0;

  virtual BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) = 0;

  // Take the transfer status of an inbound op
  virtual bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) = 0;
};

}  // namespace io
}  // namespace mori
