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
  RdmaBackendConfig(int qpPerTransfer_, int postBatchSize_, int numWorkerThreads_,
                    PollCqMode pollCqMode_, bool enableNotification_)
      : BackendConfig(BackendType::RDMA),
        qpPerTransfer(qpPerTransfer_),
        postBatchSize(postBatchSize_),
        numWorkerThreads(numWorkerThreads_),
        pollCqMode(pollCqMode_),
        enableNotification(enableNotification_) {}

  int qpPerTransfer{1};
  int postBatchSize{-1};
  int numWorkerThreads{1};
  PollCqMode pollCqMode{PollCqMode::POLLING};
  bool enableNotification{true};  // Enable/disable notification mechanism for transfer completion
};

inline std::ostream& operator<<(std::ostream& os, const RdmaBackendConfig& c) {
  return os << "qpPerTransfer[" << c.qpPerTransfer << "] postBatchSize[" << c.postBatchSize
            << "] numWorkerThreads[" << c.numWorkerThreads << "] enableNotification["
            << c.enableNotification << "]";
}

struct XgmiBackendConfig : public BackendConfig {
  XgmiBackendConfig() : BackendConfig(BackendType::XGMI) {}
  XgmiBackendConfig(int numStreams_, int numEvents_)
      : BackendConfig(BackendType::XGMI), numStreams(numStreams_), numEvents(numEvents_) {}

  int numStreams{64};
  int numEvents{64};
};

inline std::ostream& operator<<(std::ostream& os, const XgmiBackendConfig& c) {
  return os << "numStreams[" << c.numStreams << "] numEvents[" << c.numEvents << "]";
}

struct TcpBackendConfig : public BackendConfig {
  TcpBackendConfig() : BackendConfig(BackendType::TCP) {}
  // For future extension (e.g., parallelism, buffer sizing)
  int numWorkerThreads{4};
  bool preconnect{true};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         BackendSession                                         */
/* ---------------------------------------------------------------------------------------------- */
class BackendSession {
 public:
  BackendSession() = default;
  virtual ~BackendSession() = default;

  virtual void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                         TransferStatus* status, TransferUniqueId id, bool isRead) = 0;
  inline void Write(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) {
    ReadWrite(localOffset, remoteOffset, size, status, id, false);
  }
  inline void Read(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                   TransferUniqueId id) {
    ReadWrite(localOffset, remoteOffset, size, status, id, true);
  }

  virtual void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                              const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                              bool isRead) = 0;
  inline void BatchWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, false);
  }
  inline void BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                        const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, true);
  }
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

  virtual void RegisterMemory(MemoryDesc& desc) = 0;
  virtual void DeregisterMemory(const MemoryDesc& desc) = 0;

  virtual void ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                         const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                         TransferStatus* status, TransferUniqueId id, bool isRead) = 0;
  inline void Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
    ReadWrite(localSrc, localOffset, remoteDest, remoteOffset, size, status, id, false);
  }
  inline void Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                   size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
    ReadWrite(localDest, localOffset, remoteSrc, remoteOffset, size, status, id, true);
  }

  virtual void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                              const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                              const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                              bool isRead) = 0;
  inline void BatchWrite(const MemoryDesc& localSrc, const SizeVec& localOffsets,
                         const MemoryDesc& remoteDest, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localSrc, localOffsets, remoteDest, remoteOffsets, sizes, status, id, false);
  }
  inline void BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                        const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                        const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id, true);
  }

  virtual BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) = 0;

  virtual bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) = 0;

  virtual bool CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const { return true; }
};

}  // namespace io
}  // namespace mori
