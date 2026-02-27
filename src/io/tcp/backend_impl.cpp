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
#include "src/io/tcp/backend_impl.hpp"

#include "src/io/tcp/transport.hpp"

namespace mori {
namespace io {

TcpBackendSession::TcpBackendSession(const TcpBackendConfig& cfg, const MemoryDesc& l,
                                     const MemoryDesc& r, TcpTransport* t)
    : config(cfg), local(l), remote(r), transport(t) {}

void TcpBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitReadWrite(local, localOffset, remote, remoteOffset, size, status, id, isRead);
}

void TcpBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitBatchReadWrite(local, localOffsets, remote, remoteOffsets, sizes, status, id,
                                  isRead);
}

bool TcpBackendSession::Alive() const { return true; }

TcpBackend::TcpBackend(EngineKey k, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
    : myEngKey(std::move(k)), config(cfg) {
  transport = std::make_unique<TcpTransport>(myEngKey, engCfg, cfg);
  transport->Start();
  MORI_IO_INFO("TcpBackend created key={}", myEngKey.c_str());
}

TcpBackend::~TcpBackend() { transport->Shutdown(); }

std::optional<uint16_t> TcpBackend::GetListenPort() const { return transport->GetListenPort(); }

void TcpBackend::RegisterRemoteEngine(const EngineDesc& desc) {
  transport->RegisterRemoteEngine(desc);
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& desc) {
  transport->DeregisterRemoteEngine(desc);
}

void TcpBackend::RegisterMemory(MemoryDesc& desc) { transport->RegisterMemory(desc); }

void TcpBackend::DeregisterMemory(const MemoryDesc& desc) { transport->DeregisterMemory(desc); }

void TcpBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                           const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitReadWrite(localDest, localOffset, remoteSrc, remoteOffset, size, status, id,
                             isRead);
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitBatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status,
                                  id, isRead);
}

BackendSession* TcpBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  auto* sess = new TcpBackendSession(config, local, remote, transport.get());
  return sess;
}

bool TcpBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                          TransferStatus* status) {
  return transport->PopInboundTransferStatus(remote, id, status);
}

}  // namespace io
}  // namespace mori
