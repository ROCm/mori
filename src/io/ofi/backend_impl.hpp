// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see root LICENSE file.
//
// MORI-IO OFI (libfabric) backend — Slingshot / CXI compatible.
// Capabilities: FI_EP_RDM, FI_MR_ENDPOINT (no FI_MR_VIRT_ADDR), fi_write / fi_read.
#pragma once

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace application {
class TopoSystem;  // fwd-decl: GPU↔NIC affinity for local NIC selection
}  // namespace application

namespace io {

/* ── helpers ──────────────────────────────────────────────────────────────── */
#define OFI_CHECK(call, msg)                                           \
  do {                                                                 \
    int _rc = (call);                                                  \
    if (_rc < 0) {                                                     \
      throw std::runtime_error(std::string(msg) + ": " +              \
                               fi_strerror(-_rc));                     \
    }                                                                  \
  } while (0)

/* ── per-registered-memory info ───────────────────────────────────────────── */
struct OfiMemRegion {
  struct fid_mr* mr{nullptr};
  uint64_t       key{FI_KEY_NOTAVAIL};
  uintptr_t      bufBase{0};  // virtual address of registered buffer start
  size_t         size{0};
  uint32_t       nicId{0};    // local NIC/endpoint this MR is bound to
};

/* ── per-in-flight operation context ─────────────────────────────────────── */
struct OfiOpCtx {
  // Must be first: when FI_CONTEXT|FI_CONTEXT2 mode is set, the provider
  // uses the first sizeof(fi_context2)=64 bytes at msg.context for internal
  // bookkeeping.  Placing fi_ctx first keeps our fields safe.
  struct fi_context2 fi_ctx{};
  TransferStatus*   status{nullptr};
  TransferUniqueId  id{0};
  // back-link to the parent counter (for batches)
  std::atomic<int>* remaining{nullptr};
};

/* ── OFI fabric / domain / endpoint manager ──────────────────────────────── */
// Per-local-NIC endpoint resources. One is opened per unique provider domain
// returned by fi_getinfo, so each physical NIC gets its own fabric/domain/av/
// cq/ep. fi_info is aliased into the owned info_ list and must not be freed here.
struct OfiNic {
  uint32_t           nicId{0};
  std::string        domainName;  // libfabric domain name (e.g. cxi0)
  struct fi_info*    info{nullptr};
  struct fid_fabric* fabric{nullptr};
  struct fid_domain* domain{nullptr};
  struct fid_av*     av{nullptr};
  struct fid_ep*     ep{nullptr};
  struct fid_cq*     cq{nullptr};
};

class OfiManager {
 public:
  explicit OfiManager(const OfiBackendConfig& cfg);
  ~OfiManager();

  size_t NumNics() const { return nics_.size(); }
  const std::vector<OfiNic>& Nics() const { return nics_; }

  // Local endpoint address bytes for a given NIC (variable-length, ≤64 B CXI).
  std::vector<uint8_t> GetAddrName(uint32_t nic) const;

  // msgpack OfiEngineBlob: {nicId, addrName} for every local endpoint.
  std::vector<uint8_t> BuildEngineBlob() const;

  // Insert a remote endpoint address into a specific local NIC's AV. The
  // returned fi_addr_t is only valid for fi_write/fi_read from that same NIC.
  fi_addr_t InsertAddr(uint32_t localNic, const std::vector<uint8_t>& addrBytes);

  // Register a memory region on NIC `nic` and bind it to that endpoint.
  // isDevice selects FI_HMEM_ROCR (GPU) vs FI_HMEM_SYSTEM (host).
  OfiMemRegion RegisterMr(uint32_t nic, void* buf, size_t size, uint64_t accessFlags,
                          bool isDevice = false, int deviceId = 0);
  void DeregisterMr(OfiMemRegion& mr);

  // Pick the local NIC/endpoint index for a buffer by GPU/NUMA affinity.
  // deviceId < 0 selects by numaNode (host memory); falls back to NIC 0.
  uint32_t SelectLocalNic(int deviceId, int numaNode);

  struct fid_ep* Ep(uint32_t nic) const { return nics_[nic].ep; }
  struct fid_cq* Cq(uint32_t nic) const { return nics_[nic].cq; }
  bool VirtAddr() const { return virtAddr_; }
  bool HmemEnabled() const { return hmemEnabled_; }

 private:
  void OpenNic(struct fi_info* info, uint32_t nicId);

  struct fi_info* info_{nullptr};  // owns the fi_getinfo linked list
  std::vector<OfiNic> nics_;
  std::unordered_map<std::string, uint32_t> domainToNic_;  // domainName → index
  bool virtAddr_{false};
  bool hmemEnabled_{false};
  std::unique_ptr<application::TopoSystem> topo_;
};

/* ── OFI address registry (no out-of-band control plane) ─────────────────── */
// Peer endpoint addresses (EngineDesc) and MR keys (MemoryDesc) are carried
// inline in backendDescs, so no TCP exchange is needed. This class just inserts
// advertised peer addrs into the local AVs and returns fi_addr_t per transfer.
class OfiControlPlane {
 public:
  explicit OfiControlPlane(OfiManager* ofi);
  ~OfiControlPlane();

  // Seed a remote engine's per-NIC fi_addr_t values from its advertised
  // (nicId, addrName) list. Each remote addr is inserted into every local NIC's
  // AV. Returns true if a new engine entry was inserted.
  using NicAddr = std::pair<uint32_t, std::vector<uint8_t>>;
  bool SeedRemoteAddrs(const EngineKey& key, const std::vector<NicAddr>& nicAddrs);
  void DisconnectRemote(const EngineKey& key);

  // Cached fi_addr_t for sending from localNic to (remote engine, remoteNic).
  fi_addr_t GetRemoteAddr(uint32_t localNic, const EngineKey& key, uint32_t remoteNic) const;

 private:
  // Insert a remote engine's per-NIC addrs into every local NIC's AV.
  bool InsertRemoteNicAddrs(const EngineKey& key, const std::vector<NicAddr>& nicAddrs);

  OfiManager*  ofi_{nullptr};

  mutable std::mutex mu_;
  // localNic → engineKey → remoteNic → fi_addr_t (AV index is per local NIC).
  std::unordered_map<uint32_t,
      std::unordered_map<EngineKey, std::unordered_map<uint32_t, fi_addr_t>>> remoteAddrs_;
};

/* ── OFI completion poller (background thread) ────────────────────────────── */
class OfiPoller {
 public:
  explicit OfiPoller(OfiManager* ofi, int batchSize = 64);
  ~OfiPoller();

  void Start();
  void Shutdown();

 private:
  void PollLoop();

  OfiManager* ofi_;
  int         batchSize_;
  std::atomic<bool> running_{false};
  std::thread  thread_;
};

/* ── BackendSession ───────────────────────────────────────────────────────── */
class OfiBackendSession : public BackendSession {
 public:
  OfiBackendSession() = default;
  OfiBackendSession(OfiManager* ofi, OfiPoller* poller,
                    OfiMemRegion localMr, uint64_t remoteMrKey,
                    fi_addr_t remoteAddr, uintptr_t remoteBufBase, size_t remoteSize);
  ~OfiBackendSession() override;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                 TransferStatus* status, TransferUniqueId id, bool isRead) override;

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status,
                      TransferUniqueId id, bool isRead) override;

  bool Alive() const override { return alive_; }

 private:
  OfiManager*  ofi_{nullptr};
  OfiPoller*   poller_{nullptr};
  OfiMemRegion localMr_;
  uint64_t     remoteMrKey_{FI_KEY_NOTAVAIL};
  fi_addr_t    remoteAddr_{FI_ADDR_UNSPEC};
  uintptr_t    remoteBufBase_{0};
  size_t       remoteSize_{0};
  bool         alive_{true};
};

/* ── OfiBackend : public Backend ─────────────────────────────────────────── */
class OfiBackend : public Backend {
 public:
  OfiBackend(EngineKey key, const IOEngineConfig& cfg, const OfiBackendConfig& ofiCfg);
  ~OfiBackend() override;

  void RegisterRemoteEngine(const EngineDesc& desc) override;
  void DeregisterRemoteEngine(const EngineDesc& desc) override;
  void RegisterMemory(MemoryDesc& desc) override;
  void DeregisterMemory(const MemoryDesc& desc) override;

  void ReadWrite(const MemoryDesc& local, size_t localOffset,
                 const MemoryDesc& remote, size_t remoteOffset,
                 size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead) override;

  void BatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                      const MemoryDesc& remote, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status,
                      TransferUniqueId id, bool isRead) override;

  BackendSession* CreateSession(const MemoryDesc& local,
                                const MemoryDesc& remote) override;

  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                TransferStatus* status) override;

  DescBlob GetEngineDescBlob() const override;

 private:
  OfiMemRegion* GetLocalMr(MemoryUniqueId id);
  // Fetch or cache remote MR info: key + buf base + owning remote NIC.
  struct RemoteMrInfo { uint64_t key; uintptr_t base; size_t size; uint32_t nicId; };
  RemoteMrInfo GetRemoteMrInfo(const MemoryDesc& remote);

  EngineKey         myKey_;
  OfiBackendConfig  cfg_;
  std::unique_ptr<OfiManager>      ofi_;
  std::unique_ptr<OfiControlPlane> cp_;
  std::unique_ptr<OfiPoller>       poller_;

  mutable std::mutex mrMu_;
  std::unordered_map<MemoryUniqueId, OfiMemRegion> localMrs_;
  // Cache of remote MR info: engineKey → memId → RemoteMrInfo
  mutable std::mutex remoteMrMu_;
  std::unordered_map<EngineKey,
    std::unordered_map<MemoryUniqueId, RemoteMrInfo>> remoteMrCache_;
};

}  // namespace io
}  // namespace mori
