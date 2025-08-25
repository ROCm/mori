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
#include "mori/io/engine.hpp"

#include "src/io/rdma/backend_impl_v1.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                         IOEngineSession                                        */
/* ---------------------------------------------------------------------------------------------- */
TransferUniqueId IOEngineSession::AllocateTransferUniqueId() {
  return engine->AllocateTransferUniqueId();
}

void IOEngineSession::Read(size_t localOffset, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id) {
  for (auto& it : backendSess) {
    return it.second->Read(localOffset, remoteOffset, size, status, id);
  }
}

void IOEngineSession::Write(size_t localOffset, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id) {
  for (auto& it : backendSess) {
    return it.second->Write(localOffset, remoteOffset, size, status, id);
  }
}

void IOEngineSession::BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  for (auto& it : backendSess) {
    return it.second->BatchRead(localOffsets, remoteOffsets, sizes, status, id);
  }
}

bool IOEngineSession::Alive() {
  for (auto& it : backendSess) {
    if (it.second->Alive()) return true;
  }
  return false;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            IOEngine                                            */
/* ---------------------------------------------------------------------------------------------- */

IOEngine::IOEngine(EngineKey key, IOEngineConfig config) : config(config) {
  // Initialize descriptor
  desc.key = key;
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  desc.hostname = std::string(hostname);
  desc.host = config.host;
  desc.port = config.port;
}

IOEngine::~IOEngine() {}

void IOEngine::CreateBackend(BackendType type, const BackendConfig& beConfig) {
  if (type == BackendType::RDMA) {
    assert(backends.find(type) == backends.end());
    backends.insert({type, std::make_unique<RdmaBackend>(
                               desc.key, config, static_cast<const RdmaBackendConfig&>(beConfig))});
  } else
    assert(false && "not implemented");
}

void IOEngine::RemoveBackend(BackendType type) { backends.erase(type); }

void IOEngine::RegisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->RegisterRemoteEngine(remote);
  }
}

void IOEngine::DeregisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->DeregisterRemoteEngine(remote);
  }
}

MemoryDesc IOEngine::RegisterMemory(void* data, size_t size, int device, MemoryLocationType loc) {
  MemoryDesc memDesc;
  memDesc.engineKey = desc.key;
  memDesc.id = nextMemUid.fetch_add(1, std::memory_order_relaxed);
  memDesc.deviceId = device;
  memDesc.data = reinterpret_cast<uintptr_t>(data);
  memDesc.size = size;
  memDesc.loc = loc;

  for (auto& it : backends) {
    it.second->RegisterMemory(memDesc);
  }

  memPool.insert({memDesc.id, memDesc});
  return memDesc;
}

void IOEngine::DeregisterMemory(const MemoryDesc& desc) {
  for (auto& it : backends) {
    it.second->DeregisterMemory(desc);
  }
  memPool.erase(desc.id);
}

TransferUniqueId IOEngine::AllocateTransferUniqueId() {
  return nextTransferUid.fetch_add(1, std::memory_order_relaxed);
}

void IOEngine::Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
  for (auto& it : backends) {
    return it.second->Read(localDest, localOffset, remoteSrc, remoteOffset, size, status, id);
  }
}

void IOEngine::Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  for (auto& it : backends) {
    return it.second->Write(localSrc, localOffset, remoteDest, remoteOffset, size, status, id);
  }
}

void IOEngine::BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                         const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  for (auto& it : backends) {
    return it.second->BatchRead(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status,
                                id);
  }
}

IOEngineSession* IOEngine::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  IOEngineSession* sess = new IOEngineSession{};
  sess->engine = this;
  for (auto& it : backends) {
    BackendSession* bsess = it.second->CreateSession(local, remote);
    sess->backendSess.insert({it.first, bsess});
  }
  sessions.emplace_back(sess);
  return sess;
}

bool IOEngine::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) {
  status->SetCode(StatusCode::SUCCESS);
  return true;
}

}  // namespace io
}  // namespace mori
