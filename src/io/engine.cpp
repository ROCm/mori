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

#include <hip/hip_runtime_api.h>
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"
#include "mori/ras/ras.hpp"
#include "mori/utils/host_utils.hpp"
#include "src/io/call_diagnostics_internal.hpp"
#include "src/io/fabric/backend_impl.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/roctx_mori.hpp"  // ADDITIVE: MORI_ROCTX-gated host-send roctx markers
#include "src/io/sim/backend_impl.hpp"
#include "src/io/xgmi/backend_impl.hpp"
#ifdef MORI_USE_LIBFABRIC
#include "src/io/ofi/backend_impl.hpp"
#endif

namespace mori {
namespace io {

namespace {

template <typename... Args>
inline void LogTransferFailure(const std::shared_ptr<internal::IoCallDiagnostics>& diagnostics,
                               const char* fmt, Args&&... args) {
  internal::IoFailureKind failureKind =
      diagnostics ? diagnostics->CurrentFailureKind() : internal::IoFailureKind::None;
  if (failureKind == internal::IoFailureKind::None) {
    failureKind = internal::IoFailureKind::RootCause;
  }
  if (diagnostics && !diagnostics->TryMarkLogged(failureKind)) {
    return;
  }
  if (failureKind == internal::IoFailureKind::FlushCascade) {
    MORI_IO_DEBUG(fmt, std::forward<Args>(args)...);
  } else {
    MORI_IO_ERROR(fmt, std::forward<Args>(args)...);
  }
}

std::string QueryDeviceBusId(int deviceId) {
  if (deviceId < 0) {
    return "";
  }

  char busId[32] = {0};
  hipError_t err = hipDeviceGetPCIBusId(busId, sizeof(busId), deviceId);
  if (err != hipSuccess) {
    MORI_IO_WARN("Failed to query PCI bus id for device {}: {}", deviceId, hipGetErrorString(err));
    return "";
  }
  std::string result(busId);
  for (auto& c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

bool IsAutoXgmiEnabled() {
  const char* v = std::getenv("MORI_DISABLE_AUTO_XGMI");
  return v != nullptr && v[0] == '0';
}

int DetectNumaNode(void* addr) {
#if defined(SYS_get_mempolicy)
  int node = -1;
  long rc = syscall(SYS_get_mempolicy, &node, nullptr, 0,
                    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(addr)),
                    MPOL_F_NODE | MPOL_F_ADDR);
  return rc == 0 ? node : -1;
#else
  (void)addr;
  return -1;
#endif
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                         IOEngineSession                                        */
/* ---------------------------------------------------------------------------------------------- */
TransferUniqueId IOEngineSession::AllocateTransferUniqueId() {
  return engine->AllocateTransferUniqueId();
}

void IOEngineSession::Read(size_t localOffset, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
  internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Session read");
  auto registration = engine->RetainLocalRegistration(localMemory, status);
  if (!registration) {
    status->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
    return;
  }
  backendSess->Read(localOffset, remoteOffset, size, status, id);
  if (status->Failed()) {
    LogTransferFailure(diagnostics, "Session read error {} message {}", status->CodeUint32(),
                       status->Message());
  }
}

void IOEngineSession::Write(size_t localOffset, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
  internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Session write");
  auto registration = engine->RetainLocalRegistration(localMemory, status);
  if (!registration) {
    status->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
    return;
  }
  backendSess->Write(localOffset, remoteOffset, size, status, id);
  if (status->Failed()) {
    LogTransferFailure(diagnostics, "Session write error {} message {}", status->CodeUint32(),
                       status->Message());
  }
  return;
}

void IOEngineSession::BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
  internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Session batch read");
  auto registration = engine->RetainLocalRegistration(localMemory, status);
  if (!registration) {
    status->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
    return;
  }
  backendSess->BatchRead(localOffsets, remoteOffsets, sizes, status, id);
  if (status->Failed()) {
    LogTransferFailure(diagnostics, "Session batch read error {} message {}", status->CodeUint32(),
                       status->Message());
  }
}

void IOEngineSession::BatchWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                 const SizeVec& sizes, TransferStatus* status,
                                 TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  // ADDITIVE (MORI_ROCTX=1): brackets the host write dispatch for this transfer.
  mori::io::MoriRoctxRange _mori_roctx_("mori.io.session_batch_write", static_cast<uint64_t>(id));
  std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
  internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Session batch write");
  auto registration = engine->RetainLocalRegistration(localMemory, status);
  if (!registration) {
    status->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
    return;
  }
  backendSess->BatchWrite(localOffsets, remoteOffsets, sizes, status, id);
  if (status->Failed()) {
    LogTransferFailure(diagnostics, "Session batch write error {} message {}", status->CodeUint32(),
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
  desc.nodeId = mori::ResolveNodeId(hostname);
  desc.hostname = std::string(hostname);
  desc.host = config.host;
  desc.port = config.port;
  desc.pid = static_cast<int>(getpid());
  // RAS starts with the process's first engine. Built out, disabled, or failed to start, it
  // returns nullptr and the engine advertises no RAS endpoint.
  if (ras::RasRuntime* rt =
          ras::RasRuntime::EnsureStarted({desc.nodeId, desc.hostname, desc.host, desc.pid})) {
    ras::RasEndpoint ep = rt->Endpoint();
    desc.ras = RasDesc{ep.host, ep.port, ep.token, ep.uuid.hi, ep.uuid.lo};
    rt->Enqueue(ras::EngineRegistered{desc.key});
  }
  MORI_IO_INFO("Create engine key {} node_id {} hostname {}", key, desc.nodeId, hostname);
}

IOEngine::~IOEngine() {
  if (ras::RasRuntime* rt = ras::RasRuntime::Get()) rt->Enqueue(ras::EngineUnregistered{desc.key});
}

void IOEngine::CreateBackend(BackendType type, const BackendConfig& beConfig) {
  if (sim::IsEnabled()) {
    CreateSimBackend(type, beConfig);
    return;
  }

  if (backends.find(type) != backends.end()) {
    MORI_IO_WARN("Backend type {} already exists, skip duplicate creation",
                 static_cast<uint32_t>(type));
    return;
  }

  if (type == BackendType::RDMA) {
    if (!RdmaBackend::HasActiveDevices()) {
      if (!IsAutoXgmiEnabled()) {
        throw std::runtime_error(
            "no active RDMA device on this host; "
            "set MORI_DISABLE_AUTO_XGMI=0 to enable XGMI-only fallback");
      }

      if (!SupportsXgmiBackendByP2P()) {
        throw std::runtime_error(
            "no active RDMA device and no usable GPU P2P; cannot create any backend");
      }

      desc.port = internal::kXgmiOnlyFallbackPlaceholderPort;
      this->config.port = internal::kXgmiOnlyFallbackPlaceholderPort;

      if (backends.find(BackendType::XGMI) != backends.end()) {
        InvalidateRouteCache();
        return;
      }

      MORI_IO_WARN("No active RDMA device; falling back to XGMI-only mode (port={})",
                   internal::kXgmiOnlyFallbackPlaceholderPort);

      XgmiBackendConfig xgmiConfig{};
      auto backend = std::make_unique<XgmiBackend>(desc.key, config, xgmiConfig);
      backends.insert({BackendType::XGMI, std::move(backend)});
      InvalidateRouteCache();
      CaptureBackendDesc(BackendType::XGMI);
      return;
    }

    if (config.port == internal::kXgmiOnlyFallbackPlaceholderPort) {
      throw std::runtime_error(
          "IOEngineConfig.port=" + std::to_string(internal::kXgmiOnlyFallbackPlaceholderPort) +
          " is reserved as XGMI-only fallback sentinel");
    }

    auto backend = std::make_unique<RdmaBackend>(desc.key, config,
                                                 static_cast<const RdmaBackendConfig&>(beConfig));

    if (config.port == 0) {
      auto bound_port_opt = backend->GetListenPort();
      if (!bound_port_opt.has_value() || bound_port_opt.value() == 0) {
        MORI_IO_ERROR("IOEngine key {} failed to retrieve bound port after RDMA backend init",
                      desc.key);
        assert(false && "Failed to retrieve bound port after RDMA backend init");
      } else {
        uint16_t bound_port = bound_port_opt.value();
        if (bound_port == internal::kXgmiOnlyFallbackPlaceholderPort) {
          throw std::runtime_error("RDMA control-plane bound to sentinel port " +
                                   std::to_string(bound_port));
        }
        desc.port = bound_port;
        this->config.port = bound_port;
        MORI_IO_INFO("IOEngine key {} bound ephemeral port {}", desc.key, bound_port);
      }
    }

    backends.insert({type, std::move(backend)});
    InvalidateRouteCache();
    CaptureBackendDesc(type);
    EnsureXgmiBackendCreatedIfSupported();
  } else if (type == BackendType::XGMI) {
    auto backend = std::make_unique<XgmiBackend>(desc.key, config,
                                                 static_cast<const XgmiBackendConfig&>(beConfig));
    backends.insert({type, std::move(backend)});
    InvalidateRouteCache();
    CaptureBackendDesc(type);
  } else if (type == BackendType::FABRIC) {
    auto backend = std::make_unique<FabricBackend>(
        desc.key, config, static_cast<const FabricBackendConfig&>(beConfig));
    backends.insert({type, std::move(backend)});
    InvalidateRouteCache();
    CaptureBackendDesc(type);
  } else if (type == BackendType::OFI) {
#ifdef MORI_USE_LIBFABRIC
    const auto& ofiCfg = static_cast<const OfiBackendConfig&>(beConfig);
    auto be = std::make_unique<OfiBackend>(desc.key, config, ofiCfg);
    // OFI carries peer addresses inline in EngineDesc; no control-plane port.
    backends.insert({type, std::move(be)});
    InvalidateRouteCache();
    CaptureBackendDesc(type);
#else
    throw std::runtime_error("MORI was built without libfabric support (MORI_USE_LIBFABRIC=OFF)");
#endif
  } else if (type == BackendType::TCP) {
    throw std::runtime_error(
        "TCP backend is only available in simulation mode (MORI_IO_SIM_ENABLE=1)");
  } else {
    assert(false && "not implemented");
  }
  MORI_IO_INFO("Create backend type {}", static_cast<uint32_t>(type));
}

void IOEngine::CreateSimBackend(BackendType requested, const BackendConfig& beConfig) {
  BackendType type = sim::ResolveBackendType(requested);
  if (type != BackendType::RDMA && type != BackendType::TCP && type != BackendType::XGMI) {
    throw std::runtime_error("MORI_IO_SIM_ENABLE supports RDMA, TCP and XGMI backends, not " +
                             sim::BackendTypeName(type));
  }
  if (backends.find(type) != backends.end()) {
    MORI_IO_WARN("Simulated backend type {} already exists, skip duplicate creation",
                 sim::BackendTypeName(type));
    return;
  }
  // Both serve their control plane on the engine port, so one engine can only have one of them.
  for (BackendType other : {BackendType::RDMA, BackendType::TCP}) {
    if (type != BackendType::XGMI && other != type && backends.count(other) != 0) {
      throw std::runtime_error("Simulated " + sim::BackendTypeName(type) +
                               " cannot be added to an engine that has simulated " +
                               sim::BackendTypeName(other) +
                               "; use MORI_IO_SIM_BACKEND to choose the network instead");
    }
  }

  std::optional<RdmaBackendConfig> rdmaConfig;
  if (requested == BackendType::RDMA) {
    rdmaConfig = ResolveRdmaBackendConfig(static_cast<const RdmaBackendConfig&>(beConfig));
  }
  bool enableNotification = rdmaConfig ? rdmaConfig->enableNotification : true;
  auto backend = std::make_unique<SimBackend>(type, desc, config, enableNotification);

  // RDMA timing depends on the QP and worker counts, so a catalog entry measured with other
  // counts is only an approximation.
  const sim::LinkProfile& profile = backend->Profile();
  if (type == BackendType::RDMA && rdmaConfig &&
      ((profile.measuredQpPerTransfer > 0 &&
        profile.measuredQpPerTransfer != rdmaConfig->qpPerTransfer) ||
       (profile.measuredNumWorkerThreads > 0 &&
        profile.measuredNumWorkerThreads != rdmaConfig->numWorkerThreads))) {
    MORI_IO_WARN(
        "Simulated RDMA timing from {} was measured with qp_per_transfer={} num_worker_threads={}, "
        "this engine uses {} and {}",
        profile.source, profile.measuredQpPerTransfer, profile.measuredNumWorkerThreads,
        rdmaConfig->qpPerTransfer, rdmaConfig->numWorkerThreads);
  }

  if (requested == BackendType::RDMA && type == BackendType::XGMI) {
    // Same engine desc as the XGMI-only fallback taken on a host without RDMA devices.
    desc.port = internal::kXgmiOnlyFallbackPlaceholderPort;
    config.port = internal::kXgmiOnlyFallbackPlaceholderPort;
  } else if (auto port = backend->GetListenPort(); port.has_value() && config.port == 0) {
    desc.port = *port;
    config.port = *port;
  }

  backends.insert({type, std::move(backend)});
  InvalidateRouteCache();
  CaptureBackendDesc(type);
  MORI_IO_INFO("Create simulated backend type {} (requested {})", sim::BackendTypeName(type),
               sim::BackendTypeName(requested));
  if (type == BackendType::RDMA) EnsureXgmiBackendCreatedIfSupported();
}

bool IOEngine::SupportsXgmiBackendByP2P() const {
  // Simulated GPUs on one node are all peers of each other.
  if (sim::IsEnabled()) return true;

  int numDevices = 0;
  hipError_t err = hipGetDeviceCount(&numDevices);
  if (err != hipSuccess) {
    MORI_IO_WARN("XGMI probe skipped: hipGetDeviceCount failed: {}", hipGetErrorString(err));
    return false;
  }

  if (numDevices <= 0) {
    MORI_IO_INFO("XGMI probe skipped: no GPU device found");
    return false;
  }

  if (numDevices == 1) {
    MORI_IO_INFO("XGMI probe succeeded with single GPU device");
    return true;
  }

  for (int src = 0; src < numDevices; ++src) {
    for (int dst = 0; dst < numDevices; ++dst) {
      if (src == dst) continue;
      int canAccess = 0;
      err = hipDeviceCanAccessPeer(&canAccess, src, dst);
      if (err != hipSuccess) {
        MORI_IO_WARN("XGMI probe cannot query P2P from device {} to {}: {}", src, dst,
                     hipGetErrorString(err));
        continue;
      }
      if (canAccess != 0) {
        MORI_IO_INFO("XGMI probe succeeded: P2P is available between GPU {} and {}", src, dst);
        return true;
      }
    }
  }

  MORI_IO_INFO("XGMI probe skipped: no GPU peer access found");
  return false;
}

void IOEngine::EnsureXgmiBackendCreatedIfSupported() {
  if (!IsAutoXgmiEnabled()) {
    const char* disableAutoXgmi = std::getenv("MORI_DISABLE_AUTO_XGMI");
    if (disableAutoXgmi != nullptr && disableAutoXgmi[0] != '\0' && disableAutoXgmi[0] != '0') {
      MORI_IO_INFO("Auto XGMI creation is disabled by MORI_DISABLE_AUTO_XGMI");
    }
    return;
  }

  MORI_IO_INFO("Auto XGMI creation is enabled by MORI_DISABLE_AUTO_XGMI=0");

  if (backends.find(BackendType::XGMI) != backends.end()) {
    return;
  }

  if (!SupportsXgmiBackendByP2P()) {
    return;
  }

  try {
    std::unique_ptr<Backend> backend;
    if (sim::IsEnabled()) {
      backend = std::make_unique<SimBackend>(BackendType::XGMI, desc, config, false);
    } else {
      XgmiBackendConfig xgmiConfig{};
      backend = std::make_unique<XgmiBackend>(desc.key, config, xgmiConfig);
    }
    backends.insert({BackendType::XGMI, std::move(backend)});
    InvalidateRouteCache();
    CaptureBackendDesc(BackendType::XGMI);
    MORI_IO_INFO("Auto-created XGMI backend after RDMA initialization");
  } catch (const std::exception& e) {
    MORI_IO_WARN("Auto-create XGMI backend failed: {}", e.what());
  } catch (...) {
    MORI_IO_WARN("Auto-create XGMI backend failed due to unknown error");
  }
}

void IOEngine::RemoveBackend(BackendType type) {
  backends.erase(type);
  desc.backendDescs.erase(type);
  InvalidateRouteCache();
}

void IOEngine::CaptureBackendDesc(BackendType type) {
  auto it = backends.find(type);
  if (it != backends.end()) {
    desc.backendDescs[type] = it->second->GetEngineDescBlob();
  }
}

void IOEngine::RegisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->RegisterRemoteEngine(remote);
  }
  InvalidateRouteCache();
  MORI_IO_INFO("Register remote engine {} node_id {} hostname {}", remote.key.c_str(),
               remote.nodeId.c_str(), remote.hostname.c_str());
}

void IOEngine::DeregisterRemoteEngine(const EngineDesc& remote) {
  for (auto& it : backends) {
    it.second->DeregisterRemoteEngine(remote);
  }
  InvalidateRouteCache();
  MORI_IO_INFO("Deregister remote engine {}", remote.key.c_str());
}

MemoryDesc IOEngine::RegisterMemory(void* data, size_t size, int device, MemoryLocationType loc) {
  return RegisterMemoryWithBackends(data, size, device, loc, kAllBackendMask);
}

MemoryDesc IOEngine::RegisterMemoryWithBackends(void* data, size_t size, int device,
                                                MemoryLocationType loc,
                                                uint32_t allowedBackendMask) {
  if (data == nullptr || size == 0) throw std::invalid_argument("Invalid memory registration");
  MemoryDesc memDesc;
  memDesc.registeredBackendMask = 0;
  memDesc.engineKey = desc.key;
  memDesc.id = nextMemUid.fetch_add(1, std::memory_order_relaxed);
  memDesc.deviceId = device;
  if (loc == MemoryLocationType::GPU) {
    memDesc.deviceBusId =
        sim::IsEnabled() ? "sim:" + std::to_string(device) : QueryDeviceBusId(device);
  }
  memDesc.data = reinterpret_cast<uintptr_t>(data);
  memDesc.size = size;
  memDesc.loc = loc;
  if (loc == MemoryLocationType::CPU && data != nullptr) {
    memDesc.numaNode = DetectNumaNode(data);
  }

  try {
    for (auto& it : backends) {
      if ((allowedBackendMask & BackendBit(it.first)) == 0) continue;
      it.second->RegisterMemory(memDesc);
      memDesc.registeredBackendMask |= BackendBit(it.first);
    }
  } catch (...) {
    for (auto& it : backends) {
      if (memDesc.registeredBackendMask & BackendBit(it.first)) {
        it.second->DeregisterMemory(memDesc);
      }
    }
    throw;
  }

  {
    std::lock_guard<std::mutex> lock(memPoolMu);
    memPool.insert({memDesc.id, memDesc});
    registrationStates.emplace(memDesc.id, std::make_shared<RegistrationState>());
  }
  MORI_IO_TRACE("Register memory address {} size {} device {} loc {} with id {}", data, size,
                device, static_cast<uint32_t>(loc), memDesc.id);
  return memDesc;
}

std::shared_ptr<void> IOEngine::RetainLocalRegistration(const MemoryDesc& local,
                                                        TransferStatus* status) {
  if (local.engineKey != desc.key || status == nullptr) return {};
  std::shared_ptr<RegistrationState> state;
  {
    std::lock_guard<std::mutex> lock(memPoolMu);
    auto it = registrationStates.find(local.id);
    if (it == registrationStates.end()) return {};
    state = it->second;
  }
  auto ticket = std::make_shared<RegistrationTicket>();
  ticket->status = status;
  // Construct the submission guard before publishing a ticket or allocating
  // callbacks. Any subsequent setup exception must still wake deregistration.
  auto registration = std::shared_ptr<void>(ticket.get(), [ticket](void*) {
    {
      std::lock_guard<std::mutex> lock(ticket->mutex);
      ticket->submissionComplete = true;
    }
    ticket->submissionReady.notify_all();
  });
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->accepting) return {};
  state->tickets.erase(std::remove_if(state->tickets.begin(), state->tickets.end(),
                                      [](const auto& item) { return item.expired(); }),
                       state->tickets.end());
  // Install both status callbacks before publishing the ticket. In particular,
  // a callback allocation failure must never expose an unprotected raw status.
  status->SetDestructionCallback([state, ticket](bool quarantined) {
    std::unique_lock<std::mutex> lock(ticket->mutex);
    ticket->status = nullptr;
    ticket->submissionReady.wait(lock, [&] { return ticket->activeWaiters == 0; });
    // An active waiter may have proved completion while destruction waited.
    // Otherwise an abandoned IN_PROGRESS status must quarantine its backing.
    if (ticket->terminalObserved ? ticket->quarantined : quarantined) {
      std::lock_guard<std::mutex> stateLock(state->mutex);
      state->quarantined = true;
    }
  });
  status->AddLifetimeObserver([state, ticket](bool quarantined) {
    std::lock_guard<std::mutex> ticketLock(ticket->mutex);
    ticket->status = nullptr;
    ticket->terminalObserved = true;
    ticket->quarantined = quarantined;
    if (quarantined) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->quarantined = true;
    }
  });
  state->tickets.push_back(ticket);
  return registration;
}

void IOEngine::DeregisterMemory(const MemoryDesc& memory) {
  if (memory.engineKey != desc.key) {
    throw std::invalid_argument("DeregisterMemory requires a local descriptor");
  }
  std::shared_ptr<RegistrationState> state;
  {
    std::lock_guard<std::mutex> lock(memPoolMu);
    auto it = registrationStates.find(memory.id);
    if (it == registrationStates.end()) return;
    state = it->second;
  }
  std::lock_guard<std::mutex> deregisterLock(state->deregisterMutex);
  if (state->deregistered) return;
  std::vector<std::shared_ptr<RegistrationTicket>> tickets;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->accepting = false;
    for (auto& item : state->tickets)
      if (auto ticket = item.lock()) tickets.push_back(ticket);
  }
  for (auto& ticket : tickets) {
    std::unique_lock<std::mutex> lock(ticket->mutex);
    ticket->submissionReady.wait(lock, [&] { return ticket->submissionComplete; });
    if (auto* status = ticket->status) {
      ++ticket->activeWaiters;
      // The destruction callback waits for this guard before status members
      // disappear. GPU completion callbacks may take ticket->mutex, so never
      // invoke Wait, progress, or stream callbacks while holding that mutex.
      struct WaiterGuard {
        RegistrationTicket& ticket;
        ~WaiterGuard() {
          {
            std::lock_guard<std::mutex> lock(ticket.mutex);
            --ticket.activeWaiters;
          }
          ticket.submissionReady.notify_all();
        }
      } guard{*ticket};
      lock.unlock();
      status->Wait();
      if (status->ResourcesQuarantined()) {
        std::lock_guard<std::mutex> stateLock(state->mutex);
        state->quarantined = true;
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->quarantined) throw std::runtime_error("DRAIN_UNCONFIRMED: registration quarantined");
  }
  for (auto& it : backends) {
    if (memory.registeredBackendMask & BackendBit(it.first)) it.second->DeregisterMemory(memory);
  }
  {
    std::lock_guard<std::mutex> lock(memPoolMu);
    memPool.erase(memory.id);
    registrationStates.erase(memory.id);
  }
  state->deregistered = true;
  InvalidateRouteCache();
}

TransferUniqueId IOEngine::AllocateTransferUniqueId() {
  TransferUniqueId id = nextTransferUid.fetch_add(1, std::memory_order_relaxed);
  MORI_IO_TRACE("Allocate transfer uid {}", id);
  return id;
}

void IOEngine::SetTransferBackendMask(uint32_t mask) {
  transferBackendMask.store(mask, std::memory_order_release);
  InvalidateRouteCache();
}

std::optional<BackendType> IOEngine::GetSelectedBackend(const MemoryDesc& local,
                                                        const MemoryDesc& remote) {
  Backend* selected = SelectBackend(local, remote);
  for (const auto& entry : backends) {
    if (entry.second.get() == selected) return entry.first;
  }
  return std::nullopt;
}

void IOEngine::InvalidateRemoteMemory(const MemoryDesc& remote) {
  if (remote.engineKey == desc.key) {
    throw std::invalid_argument("InvalidateRemoteMemory requires a remote descriptor");
  }
  for (auto& entry : backends) entry.second->InvalidateRemoteMemory(remote);
  InvalidateRouteCache();
}

bool IOEngine::IsFabricReady(int device) {
  auto it = backends.find(BackendType::FABRIC);
  auto* fabric = it == backends.end() ? nullptr : dynamic_cast<FabricBackend*>(it->second.get());
  return fabric != nullptr && fabric->IsReady(device);
}

Backend* IOEngine::SelectBackend(const MemoryDesc& local, const MemoryDesc& remote) {
  const uint32_t eligible = transferBackendMask.load(std::memory_order_acquire) &
                            local.registeredBackendMask & remote.registeredBackendMask;
  RouteCacheKey routeKey{remote.engineKey, local.loc, remote.loc, local.deviceId,
                         remote.deviceId,  local.id,  remote.id};
  auto usable = [&](BackendType type) -> Backend* {
    auto it = backends.find(type);
    if ((eligible & BackendBit(type)) == 0 || it == backends.end() ||
        !it->second->CanHandle(local, remote))
      return nullptr;
    return it->second.get();
  };
  if (auto cached = QueryRouteCache(routeKey)) {
    if (Backend* backend = usable(*cached)) return backend;
  }
  for (BackendType type : {BackendType::XGMI, BackendType::FABRIC, BackendType::RDMA,
                           BackendType::OFI, BackendType::TCP}) {
    if (Backend* backend = usable(type)) {
      UpdateRouteCache(routeKey, type);
      return backend;
    }
  }
  return nullptr;
}

void IOEngine::InvalidateRouteCache() {
  std::unique_lock<std::shared_mutex> lock(routeCacheMu);
  routeCache.clear();
}

void IOEngine::UpdateRouteCache(const RouteCacheKey& key, BackendType backendType) {
  std::unique_lock<std::shared_mutex> lock(routeCacheMu);
  routeCache[key] = backendType;
}

std::optional<BackendType> IOEngine::QueryRouteCache(const RouteCacheKey& key) const {
  std::shared_lock<std::shared_mutex> lock(routeCacheMu);
  auto it = routeCache.find(key);
  if (it == routeCache.end()) {
    return std::nullopt;
  }
  return it->second;
}

#define SELECT_BACKEND_AND_RETURN_IF_NONE(local, remote, status, backend)     \
  backend = SelectBackend(local, remote);                                     \
  if (backend == nullptr) {                                                   \
    if (status != nullptr) {                                                  \
      status->Update(StatusCode::ERR_BAD_STATE,                               \
                     "No available backend found, create backend first");     \
    }                                                                         \
    MORI_IO_ERROR("No available backend found, please create backend first"); \
    return;                                                                   \
  }

void IOEngine::Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
  internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Engine read");
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localDest, remoteSrc, status, backend);
  auto registration = RetainLocalRegistration(localDest, status);
  if (!registration) {
    status->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
    return;
  }
  backend->Read(localDest, localOffset, remoteSrc, remoteOffset, size, status, id);
  if (status->Failed()) {
    LogTransferFailure(diagnostics, "Engine read error {} message {}", status->CodeUint32(),
                       status->Message());
  }
}

void IOEngine::Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
  internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Engine write");
  Backend* backend = nullptr;
  SELECT_BACKEND_AND_RETURN_IF_NONE(localSrc, remoteDest, status, backend);
  auto registration = RetainLocalRegistration(localSrc, status);
  if (!registration) {
    status->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
    return;
  }
  backend->Write(localSrc, localOffset, remoteDest, remoteOffset, size, status, id);
  if (status->Failed()) {
    LogTransferFailure(diagnostics, "Engine write error {} message {}", status->CodeUint32(),
                       status->Message());
  }
}

void IOEngine::BatchRead(const MemDescVec& localDest, const BatchSizeVec& localOffsets,
                         const MemDescVec& remoteSrc, const BatchSizeVec& remoteOffsets,
                         const BatchSizeVec& sizes, TransferStatusPtrVec& status,
                         TransferUniqueIdVec& ids) {
  MORI_IO_FUNCTION_TIMER;
  size_t batchSize = localDest.size();
  assert(batchSize == remoteSrc.size());
  assert(batchSize == localOffsets.size());
  assert(batchSize == remoteOffsets.size());
  assert(batchSize == sizes.size());
  assert(batchSize == status.size());
  assert(batchSize == ids.size());

  for (size_t i = 0; i < batchSize; i++) {
    std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
    internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Engine batch read");
    Backend* backend = nullptr;
    backend = SelectBackend(localDest[i], remoteSrc[i]);
    if (backend == nullptr) {
      status[i]->Update(StatusCode::ERR_BAD_STATE, "No eligible backend for batch group");
      continue;
    }
    auto registration = RetainLocalRegistration(localDest[i], status[i]);
    if (!registration) {
      status[i]->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
      continue;
    }
    backend->BatchRead(localDest[i], localOffsets[i], remoteSrc[i], remoteOffsets[i], sizes[i],
                       status[i], ids[i]);
    if (status[i]->Failed()) {
      LogTransferFailure(diagnostics, "Engine batch read error {} message {}",
                         status[i]->CodeUint32(), status[i]->Message());
    }
  }
}

void IOEngine::BatchWrite(const MemDescVec& localSrc, const BatchSizeVec& localOffsets,
                          const MemDescVec& remoteDest, const BatchSizeVec& remoteOffsets,
                          const BatchSizeVec& sizes, TransferStatusPtrVec& status,
                          TransferUniqueIdVec& ids) {
  MORI_IO_FUNCTION_TIMER;
  // ADDITIVE (MORI_ROCTX=1): brackets the host engine-level batch write.
  mori::io::MoriRoctxRange _mori_roctx_("mori.io.engine_batch_write");
  size_t batchSize = localSrc.size();
  assert(batchSize == remoteDest.size());
  assert(batchSize == localOffsets.size());
  assert(batchSize == remoteOffsets.size());
  assert(batchSize == sizes.size());
  assert(batchSize == status.size());
  assert(batchSize == ids.size());

  for (size_t i = 0; i < batchSize; i++) {
    std::shared_ptr<internal::IoCallDiagnostics> diagnostics;
    internal::ScopedIoCallDiagnosticsCapture capture(&diagnostics, "Engine batch write");
    Backend* backend = nullptr;
    backend = SelectBackend(localSrc[i], remoteDest[i]);
    if (backend == nullptr) {
      status[i]->Update(StatusCode::ERR_BAD_STATE, "No eligible backend for batch group");
      continue;
    }
    auto registration = RetainLocalRegistration(localSrc[i], status[i]);
    if (!registration) {
      status[i]->Update(StatusCode::ERR_BAD_STATE, "Local registration unavailable");
      continue;
    }
    backend->BatchWrite(localSrc[i], localOffsets[i], remoteDest[i], remoteOffsets[i], sizes[i],
                        status[i], ids[i]);
    if (status[i]->Failed()) {
      LogTransferFailure(diagnostics, "Engine batch write error {} message {}",
                         status[i]->CodeUint32(), status[i]->Message());
    }
  }
}

std::optional<IOEngineSession> IOEngine::CreateSession(const MemoryDesc& local,
                                                       const MemoryDesc& remote) {
  IOEngineSession sess{};
  sess.engine = this;
  sess.localMemory = local;

  Backend* backend = SelectBackend(local, remote);
  if (backend == nullptr) {
    return std::nullopt;
  }
  sess.backendSess.reset(backend->CreateSession(local, remote));
  if (sess.backendSess == nullptr) {
    return std::nullopt;
  }

  return sess;
}

void IOEngine::LoadScatterGatherModule(const std::string& hsacoPath) {
  auto it = backends.find(BackendType::XGMI);
  if (it == backends.end()) return;
  // A simulated XGMI backend has no kernels to load.
  if (auto* xgmi = dynamic_cast<XgmiBackend*>(it->second.get())) {
    xgmi->LoadScatterGatherModule(hsacoPath);
  }
}

void IOEngine::LoadFabricCopyModule(const std::string& hsacoPath) {
  auto it = backends.find(BackendType::FABRIC);
  if (it == backends.end()) return;
  if (auto* fabric = dynamic_cast<FabricBackend*>(it->second.get())) {
    fabric->LoadFabricCopyModule(hsacoPath);
  }
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

bool IOEngine::PopPeerFailure(PeerFailureEvent* out) {
  if (out == nullptr) return false;
  for (auto& it : backends) {
    if (it.second->PopPeerFailure(out)) return true;
  }
  return false;
}

StatusCode IOEngine::WaitAll(const std::vector<TransferStatus*>& statuses, int timeoutMs) {
  if (statuses.empty()) return StatusCode::SUCCESS;

  if (timeoutMs == 0) {
    bool anyInProgress = false;
    for (TransferStatus* status : statuses) {
      if (status == nullptr) continue;
      StatusCode rc = status->WaitFor(0);
      if (rc == StatusCode::IN_PROGRESS) {
        anyInProgress = true;
        continue;
      }
      if (rc != StatusCode::SUCCESS) return rc;
    }
    return anyInProgress ? StatusCode::IN_PROGRESS : StatusCode::SUCCESS;
  }

  if (timeoutMs < 0) {
    StatusCode firstError = StatusCode::SUCCESS;
    for (TransferStatus* status : statuses) {
      if (status == nullptr) continue;
      StatusCode rc = status->WaitFor(-1);
      if (rc != StatusCode::SUCCESS && firstError == StatusCode::SUCCESS) {
        firstError = rc;
      }
    }
    return firstError;
  }

  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
  StatusCode firstError = StatusCode::SUCCESS;
  for (TransferStatus* status : statuses) {
    if (status == nullptr) continue;

    const auto now = Clock::now();
    if (now >= deadline) {
      return firstError != StatusCode::SUCCESS ? firstError : StatusCode::IN_PROGRESS;
    }

    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const int remainingMs = remaining.count() > std::numeric_limits<int>::max()
                                ? std::numeric_limits<int>::max()
                                : static_cast<int>(remaining.count());
    StatusCode rc = status->WaitFor(remainingMs);
    if (rc == StatusCode::IN_PROGRESS) {
      return firstError != StatusCode::SUCCESS ? firstError : StatusCode::IN_PROGRESS;
    }
    if (rc != StatusCode::SUCCESS && firstError == StatusCode::SUCCESS) {
      firstError = rc;
    }
  }
  return firstError;
}

}  // namespace io
}  // namespace mori
