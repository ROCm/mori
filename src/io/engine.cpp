#include "mori/io/engine.hpp"

#include "src/io/rdma/backend_impl_v1.hpp"

#include <cstdint>
#include <iostream>

namespace mori {
namespace io {

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

void IOEngine::CreateBackend(BackendType type, void* params) {
  if (type == BackendType::RDMA) {
    assert(backends.find(type) == backends.end());
    backends.insert({type, std::make_unique<RdmaBackend>(desc.key, config)});
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
  memDesc.data = data;
  memDesc.size = size;
  memDesc.loc = loc;
  std::cout<<"\n\n\nzovlog:moriio engine:IOEngine::RegisterMemory----> data = "<<reinterpret_cast<uintptr_t>(data)<<",size = "<<size<<"\n\n\n";
  for (auto& it : backends) {
    it.second->RegisterMemory(memDesc);
  }

  memPool.insert({memDesc.id, memDesc});
  return memDesc;
}

void IOEngine::DeregisterMemory(MemoryDesc& desc) {
  for (auto& it : backends) {
    it.second->DeregisterMemory(desc);
  }
  memPool.erase(desc.id);
}

TransferUniqueId IOEngine::AllocateTransferUniqueId() {
  return nextTransferUid.fetch_add(1, std::memory_order_relaxed);
}

void IOEngine::Read(MemoryDesc localDest, size_t localOffset, MemoryDesc remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
  for (auto& it : backends) {
    return it.second->Read(localDest, localOffset, remoteSrc, remoteOffset, size, status, id);
  }
}

void IOEngine::Write(MemoryDesc localSrc, size_t localOffset, MemoryDesc remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  for (auto& it : backends) {
    return it.second->Write(localSrc, localOffset, remoteDest, remoteOffset, size, status, id);
  }
}

bool IOEngine::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) {
  status->SetCode(StatusCode::SUCCESS);
  return true;
}

}  // namespace io
}  // namespace mori