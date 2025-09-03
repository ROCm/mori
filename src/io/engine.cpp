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

#include "mori/io/logging.hpp"
#include "src/io/rdma/backend_impl.hpp"

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
  MORI_IO_FUNCTION_TIMER;
  backendSess->Read(localOffset, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session read error {} message {}", status->CodeUint32(), status->Message());
  }
}

void IOEngineSession::Write(size_t localOffset, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  backendSess->Write(localOffset, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session write error {} message {}", status->CodeUint32(), status->Message());
  }
  return;
}

void IOEngineSession::BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  backendSess->BatchRead(localOffsets, remoteOffsets, sizes, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Session batch read error {} message {}", status->CodeUint32(),
                  status->Message());
  }
}

bool IOEngineSession::Alive() { return backendSess->Alive(); }

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
  MORI_IO_INFO("Create engine key {} hostname {} host {}, port {}", key, hostname, config.host,
               config.port);
}

IOEngine::~IOEngine() {}

void IOEngine::CreateBackend(BackendType type, const BackendConfig& beConfig) {
  if (type == BackendType::RDMA) {
    assert(backends.find(type) == backends.end());
    backends.insert({type, std::make_unique<RdmaBackend>(
                               desc.key, config, static_cast<const RdmaBackendConfig&>(beConfig))});
  } else
    assert(false && "not implemented");
  MORI_IO_INFO("Create backend type {}", static_cast<uint32_t>(type));
}

void IOEngine::RemoveBackend(BackendType type) { backends.erase(type); }

void IOEngine::RegisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->RegisterRemoteEngine(remote);
  }
  MORI_IO_INFO("Register remote engine {}", remote.key.c_str());
}

void IOEngine::DeregisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->DeregisterRemoteEngine(remote);
  }
  MORI_IO_INFO("Deregister remote engine {}", remote.key.c_str());
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
  MORI_IO_TRACE("Register memory address {} size {} device {} loc {} with id {}", data, size,
                device, static_cast<uint32_t>(loc), memDesc.id);
  return memDesc;
}

void IOEngine::DeregisterMemory(const MemoryDesc& desc) {
  for (auto& it : backends) {
    it.second->DeregisterMemory(desc);
  }
  memPool.erase(desc.id);
  MORI_IO_TRACE("Deregister memory {} at address {}", desc.id, desc.data);
}

TransferUniqueId IOEngine::AllocateTransferUniqueId() {
  TransferUniqueId id = nextTransferUid.fetch_add(1, std::memory_order_relaxed);
  MORI_IO_TRACE("Allocate transfer uid {}", id);
  return id;
}

Backend* IOEngine::SelectBackend(const MemoryDesc& local, const MemoryDesc& remote) {
  if (backends.empty()) {
    return nullptr;
  }
  return backends.begin()->second.get();
}

#define SELECT_BACKEND_AND_RETURN_IF_NONE(local, remote, status, backend)     \
  backend = SelectBackend(local, remote);                                     \
  if (backend == nullptr) {                                                   \
    if (status != nullptr) {                                                  \
      status->SetCode(StatusCode::ERR_BAD_STATE);                             \
      status->SetMessage("No available backend found, create backend first"); \
    }                                                                         \
    MORI_IO_ERROR("No available backend found, please create backend first"); \
    return;                                                                   \
  }

void IOEngine::Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localDest, remoteSrc, status, backend);
  backend->Read(localDest, localOffset, remoteSrc, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Engine read error {} message {}", status->CodeUint32(), status->Message());
  }
}

void IOEngine::Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localSrc, remoteDest, status, backend);
  backend->Write(localSrc, localOffset, remoteDest, remoteOffset, size, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Engine write error {} message {}", status->CodeUint32(), status->Message());
  }
}

void IOEngine::BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                         const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localDest, remoteSrc, status, backend);
  backend->BatchRead(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id);
  if (status->Failed()) {
    MORI_IO_ERROR("Engine batch read error {} message {}", status->CodeUint32(), status->Message());
  }
}

std::optional<IOEngineSession> IOEngine::CreateSession(const MemoryDesc& local,
                                                       const MemoryDesc& remote) {
  IOEngineSession sess{};
  sess.engine = this;

  Backend* backend = SelectBackend(local, remote);
  if (backend == nullptr) {
    return std::nullopt;
  }
  sess.backendSess.reset(backend->CreateSession(local, remote));

  return sess;
}

bool IOEngine::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) {
  // status->SetCode(StatusCode::SUCCESS);
  for (auto& it : backends) {
    bool popped = it.second->PopInboundTransferStatus(remote, id, status);
    if (popped) return true;
  }
  return false;
}

}  // namespace io
}  // namespace mori
