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

#include <memory>
#include <optional>
#include <string>

#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"

namespace mori {
namespace io {

class TcpTransport;

/* ---------------------------------------------------------------------------------------------- */
/*                                       TcpBackendSession                                        */
/* ---------------------------------------------------------------------------------------------- */
class TcpBackendSession : public BackendSession {
 public:
  TcpBackendSession() = default;
  TcpBackendSession(const TcpBackendConfig& config, const MemoryDesc& local, const MemoryDesc& remote,
                    TcpTransport* transport);
  ~TcpBackendSession() override = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead) override;

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets, const SizeVec& sizes,
                      TransferStatus* status, TransferUniqueId id, bool isRead) override;

  bool Alive() const override;

 private:
  TcpBackendConfig config{};
  MemoryDesc local{};
  MemoryDesc remote{};
  TcpTransport* transport{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           TcpBackend                                           */
/* ---------------------------------------------------------------------------------------------- */
class TcpBackend : public Backend {
 public:
  TcpBackend(EngineKey, const IOEngineConfig&, const TcpBackendConfig&);
  ~TcpBackend() override;

  std::optional<uint16_t> GetListenPort() const;

  void RegisterRemoteEngine(const EngineDesc&) override;
  void DeregisterRemoteEngine(const EngineDesc&) override;

  void RegisterMemory(MemoryDesc& desc) override;
  void DeregisterMemory(const MemoryDesc& desc) override;

  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead) override;

  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets, const SizeVec& sizes,
                      TransferStatus* status, TransferUniqueId id, bool isRead) override;

  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) override;

  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status) override;

 private:
  EngineKey myEngKey;
  TcpBackendConfig config{};
  std::unique_ptr<TcpTransport> transport{nullptr};
};

}  // namespace io
}  // namespace mori
