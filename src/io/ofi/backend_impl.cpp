// SPDX-FileCopyrightText: © Hewlett Packard Enterprise Development LP
// MIT License — see root LICENSE file.
//
// MORI-IO OFI (libfabric) backend — Slingshot/CXI compatible.
// Uses FI_EP_RDM + FI_MR_ENDPOINT + FI_MR_PROV_KEY (no FI_MR_VIRT_ADDR).
// Data plane: fi_writemsg / fi_readmsg with offset-based RMA addresses.
// Control plane: none — peer addr-names and MR keys travel inline in EngineDesc/MemoryDesc.

#include "backend_impl.hpp"

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "mori/application/topology/system.hpp"

namespace mori {
namespace io {

namespace {
// Inline MR metadata carried in MemoryDesc.backendDescs[OFI]. Base/size are the
// desc's own data/size, so only the provider MR key + owning NIC need publishing.
struct OfiMrBlob {
  uint64_t key{0};
  uint32_t nicId{0};  // remote endpoint that owns/serves this MR
  MSGPACK_DEFINE(key, nicId);
};

// One advertised local endpoint address. Peers insert addrName into their AVs
// and reference it by nicId (matched against OfiMrBlob.nicId).
struct OfiNicAddr {
  uint32_t             nicId{0};
  std::vector<uint8_t> addrName;
  MSGPACK_DEFINE(nicId, addrName);
};

// Engine-level OFI descriptor: the full per-NIC address list, published inline
// in EngineDesc.backendDescs[OFI].
struct OfiEngineBlob {
  std::vector<OfiNicAddr> nics;
  MSGPACK_DEFINE(nics);
};

// Parse an EngineDesc/HELLO OFI blob into (nicId, addr) pairs. Falls back to
// treating the bytes as a single legacy addr_name (nicId 0) if it is not a
// valid OfiEngineBlob.
std::vector<OfiControlPlane::NicAddr> ParseOfiEngineBlob(const uint8_t* data, size_t len) {
  std::vector<OfiControlPlane::NicAddr> out;
  try {
    auto oh = msgpack::unpack(reinterpret_cast<const char*>(data), len);
    OfiEngineBlob blob = oh.get().as<OfiEngineBlob>();
    out.reserve(blob.nics.size());
    for (auto& n : blob.nics) out.emplace_back(n.nicId, n.addrName);
  } catch (...) {
    out.clear();
    out.emplace_back(0u, std::vector<uint8_t>(data, data + len));  // legacy single addr
  }
  return out;
}

// HPE Slingshot (Cassini) NIC discovered from sysfs. These NICs are not exposed
// through ibverbs, so the OFI backend injects them into the topology so GPU→NIC
// affinity can rank them.
struct CxiNicInfo {
  std::string           name;
  application::PciBusId  busId{0};
  double                gbps{0};
};

// Parse a CXI link-speed token (e.g. "ck400G", "BS200G", "cd50G") into Gbps.
// The 2-char media prefix and trailing 'G' are ignored; only the digits carry
// the rate. Returns 0 if none is found.
double ParseCxiLinkGbps(const std::string& token) {
  std::string digits;
  for (char c : token)
    if (std::isdigit(static_cast<unsigned char>(c))) digits.push_back(c);
  return digits.empty() ? 0.0 : static_cast<double>(std::stol(digits));
}

// Enumerate CXI NICs from /sys/class/cxi/cxiN. Each has a `device` symlink to
// its PCI node; the entry name (cxi0, cxi1, …) matches the libfabric CXI domain.
std::vector<CxiNicInfo> DiscoverCxiNics() {
  std::vector<CxiNicInfo> out;
  const std::filesystem::path cxiRoot{"/sys/class/cxi"};
  std::error_code ec;
  if (!std::filesystem::exists(cxiRoot, ec)) return out;

  constexpr double kCxiDefaultGbps = 200.0;  // fallback when link speed unreadable
  for (const auto& entry : std::filesystem::directory_iterator(cxiRoot, ec)) {
    std::filesystem::path devPath = std::filesystem::canonical(entry.path() / "device", ec);
    if (ec) {
      ec.clear();
      continue;
    }

    application::PciBusId busId = application::ParseBusIdFromSysfs(devPath);
    if (busId.packed == 0) continue;

    // e.g. /sys/class/cxi/cxi0/device/port/0/link/speed -> "ck400G".
    double gbps = kCxiDefaultGbps;
    std::ifstream speedFile(entry.path() / "device" / "port" / "0" / "link" / "speed");
    std::string tok;
    if (speedFile && (speedFile >> tok)) {
      double parsed = ParseCxiLinkGbps(tok);
      if (parsed > 0.0) gbps = parsed;
    }
    out.push_back({entry.path().filename().string(), busId, gbps});
  }
  return out;
}

// Close a NIC's libfabric handles in reverse open order (safe on partial init).
void CloseNic(OfiNic& nic) {
  if (nic.ep)     { fi_close(&nic.ep->fid);     nic.ep     = nullptr; }
  if (nic.cq)     { fi_close(&nic.cq->fid);     nic.cq     = nullptr; }
  if (nic.av)     { fi_close(&nic.av->fid);     nic.av     = nullptr; }
  if (nic.domain) { fi_close(&nic.domain->fid); nic.domain = nullptr; }
  if (nic.fabric) { fi_close(&nic.fabric->fid); nic.fabric = nullptr; }
}
}  // namespace

/* ═══════════════════════════════════════════════════════════════════════════
 *  OfiManager
 * ═══════════════════════════════════════════════════════════════════════════ */

OfiManager::OfiManager(const OfiBackendConfig& cfg) {
  // Build hints for a given HMEM setting so we can retry host-only on failure.
  auto makeHints = [&cfg](bool withHmem) -> struct fi_info* {
    struct fi_info* hints = fi_allocinfo();
    if (!hints) throw std::runtime_error("fi_allocinfo failed");

    hints->ep_attr->type = FI_EP_RDM;
    hints->caps          = FI_RMA;  // one-sided only; no two-sided messaging
    hints->mode          = FI_CONTEXT | FI_CONTEXT2;

    // Slingshot/CXI: FI_MR_ENDPOINT + FI_MR_PROV_KEY.  No FI_MR_VIRT_ADDR.
    hints->domain_attr->mr_mode =
        FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_ENDPOINT | FI_MR_PROV_KEY;

    if (withHmem) {
      // Enable device-memory (ROCm/HBM) registration and transfers.
      hints->caps |= FI_HMEM;
      hints->domain_attr->mr_mode |= FI_MR_HMEM;
    }

    // FI_THREAD_SAFE: provider serializes internally, so multiple threads may
    // share a domain's ep/cq (e.g. lock-free ReadWrite from several threads).
    hints->domain_attr->threading = FI_THREAD_SAFE;

    if (!cfg.providerHint.empty())
      hints->fabric_attr->prov_name = strdup(cfg.providerHint.c_str());

    return hints;
  };

  // Require FI_HMEM so GPU buffers can be registered/transferred; no host-only
  // fallback — if the provider can't negotiate HMEM, fail.
  struct fi_info* hints = makeHints(/*withHmem=*/true);
  int ret = fi_getinfo(FI_VERSION(2, 6), nullptr, nullptr, 0, hints, &info_);
  fi_freeinfo(hints);
  if (ret)
    throw std::runtime_error(std::string("fi_getinfo (FI_HMEM): ") + fi_strerror(-ret));
  hmemEnabled_ = true;

  // A throwing constructor does not run ~OfiManager, so clean up info_ and any
  // NICs already opened on failure.
  try {
    topo_.reset(new application::TopoSystem());

    // Slingshot: CXI NICs aren't exposed via ibverbs; inject them so GPU→NIC
    // affinity can rank them (only when verbs discovered none).
    if (topo_->GetTopoSystemNet()->NumNics() == 0) {
      for (const auto& cxi : DiscoverCxiNics())
        topo_->GetTopoSystemNet()->AddNic(cxi.name, cxi.busId, cxi.gbps);
    }

    // Open one endpoint per unique provider domain (i.e. per physical NIC).
    // fi_getinfo returns a linked list; dedup by domain name.
    uint32_t nicId = 0;
    for (struct fi_info* fi = info_; fi != nullptr; fi = fi->next) {
      const char* dn = (fi->domain_attr && fi->domain_attr->name) ? fi->domain_attr->name : nullptr;
      if (!dn) continue;
      std::string dname(dn);
      if (domainToNic_.count(dname)) continue;
      OpenNic(fi, nicId);
      domainToNic_[dname] = nicId;
      ++nicId;
    }
    if (nics_.empty()) throw std::runtime_error("OfiManager: no usable OFI domains found");
  } catch (...) {
    CloseAll();
    throw;
  }

  MORI_IO_INFO("OfiManager: provider={} nics={} mr_mode=0x{:x} hmem={}",
               info_->fabric_attr->prov_name, nics_.size(),
               (unsigned)info_->domain_attr->mr_mode, hmemEnabled_);
}

void OfiManager::OpenNic(struct fi_info* fi, uint32_t nicId) {
  OfiNic nic;
  nic.nicId      = nicId;
  nic.info       = fi;
  nic.domainName = fi->domain_attr->name;

  // Close whatever was opened if any step throws, so a partial NIC never leaks.
  try {
    OFI_CHECK(fi_fabric(fi->fabric_attr, &nic.fabric, nullptr), "fi_fabric");
    OFI_CHECK(fi_domain(nic.fabric, fi, &nic.domain, nullptr),  "fi_domain");

    struct fi_av_attr av_attr{};
    av_attr.type  = FI_AV_TABLE;
    av_attr.count = 1024;
    OFI_CHECK(fi_av_open(nic.domain, &av_attr, &nic.av, nullptr), "fi_av_open");

    struct fi_cq_attr cq_attr{};
    cq_attr.format = FI_CQ_FORMAT_DATA;
    cq_attr.size   = 4096;
    OFI_CHECK(fi_cq_open(nic.domain, &cq_attr, &nic.cq, nullptr), "fi_cq_open");

    OFI_CHECK(fi_endpoint(nic.domain, fi, &nic.ep, nullptr),        "fi_endpoint");
    // RMA ops (fi_read/fi_writemsg) complete on the transmit context; we never
    // post receives, so bind the CQ to FI_TRANSMIT only.
    OFI_CHECK(fi_ep_bind(nic.ep, &nic.cq->fid, FI_TRANSMIT), "fi_ep_bind CQ");
    OFI_CHECK(fi_ep_bind(nic.ep, &nic.av->fid, 0),                  "fi_ep_bind AV");
    OFI_CHECK(fi_enable(nic.ep),                                    "fi_enable ep");
  } catch (...) {
    CloseNic(nic);
    throw;
  }

  MORI_IO_INFO("OfiManager: opened NIC {} domain={}", nicId, nic.domainName);
  nics_.push_back(nic);
}

void OfiManager::CloseAll() {
  for (auto& nic : nics_) CloseNic(nic);
  nics_.clear();
  if (info_) { fi_freeinfo(info_); info_ = nullptr; }
}

OfiManager::~OfiManager() { CloseAll(); }

std::vector<uint8_t> OfiManager::GetAddrName(uint32_t nic) const {
  const OfiNic& n = nics_.at(nic);
  // Probe with a NULL buffer to learn the exact address length (provider sets
  // len and returns -FI_ETOOSMALL), then fetch into a right-sized buffer.
  size_t len = 0;
  int ret = fi_getname(&n.ep->fid, nullptr, &len);
  if (ret != -FI_ETOOSMALL)
    throw std::runtime_error(std::string("fi_getname (size probe): ") + fi_strerror(-ret));

  std::vector<uint8_t> buf(len);
  ret = fi_getname(&n.ep->fid, buf.data(), &len);
  if (ret) throw std::runtime_error(std::string("fi_getname: ") + fi_strerror(-ret));
  buf.resize(len);
  return buf;
}

std::vector<uint8_t> OfiManager::BuildEngineBlob() const {
  OfiEngineBlob blob;
  blob.nics.reserve(nics_.size());
  for (const auto& nic : nics_) {
    OfiNicAddr na;
    na.nicId    = nic.nicId;
    na.addrName = GetAddrName(nic.nicId);
    blob.nics.push_back(std::move(na));
  }
  msgpack::sbuffer sb;
  msgpack::pack(sb, blob);
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(sb.data()),
                              reinterpret_cast<const uint8_t*>(sb.data()) + sb.size());
}

fi_addr_t OfiManager::InsertAddr(uint32_t localNic, const std::vector<uint8_t>& addrBytes) {
  fi_addr_t addr = FI_ADDR_UNSPEC;
  int ret = fi_av_insert(nics_.at(localNic).av, addrBytes.data(), 1, &addr, 0, nullptr);
  if (ret != 1)
    throw std::runtime_error(std::string("fi_av_insert returned ") + std::to_string(ret));
  return addr;
}

uint32_t OfiManager::SelectLocalNic(int deviceId, int numaNode) {
  if (nics_.size() <= 1) return 0;

  std::vector<std::string> names;
  try {
    if (deviceId >= 0) {
      names = topo_->MatchGpuAndNics(deviceId, static_cast<int>(nics_.size()));
    } else {
      names = topo_->MatchCpuNics(numaNode, static_cast<int>(nics_.size()));
    }
  } catch (const std::exception& e) {
    MORI_IO_WARN("OfiManager: NIC affinity lookup failed ({}); using fallback NIC", e.what());
  }
  // TopoSystem NIC names (e.g. cxi0) match the provider domain name.
  for (const auto& name : names) {
    auto it = domainToNic_.find(name);
    if (it != domainToNic_.end()) return it->second;
  }
  // No topology match (e.g. tcp/verbs dev providers): stable fallback.
  return deviceId >= 0 ? static_cast<uint32_t>(deviceId % nics_.size()) : 0u;
}

OfiMemRegion OfiManager::RegisterMr(uint32_t nic, void* buf, size_t size, uint64_t accessFlags,
                                    bool isDevice, int deviceId) {
  if (accessFlags == 0)
    accessFlags = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (isDevice && !hmemEnabled_)
    throw std::runtime_error(
        "OfiManager::RegisterMr: GPU buffer registration requested but the OFI "
        "provider did not negotiate FI_HMEM. Rebuild libfabric with ROCm/HMEM "
        "support or use a host-memory KV cache.");

  OfiNic& n = nics_.at(nic);
  OfiMemRegion mr{};
  mr.bufBase = reinterpret_cast<uintptr_t>(buf);
  mr.size    = size;
  mr.nicId   = nic;

  // Use fi_mr_regattr so we can set the HMEM interface for device buffers.
  struct iovec iov = {.iov_base = buf, .iov_len = size};
  struct fi_mr_attr attr{};
  attr.mr_iov     = &iov;
  attr.iov_count  = 1;
  attr.access     = accessFlags;
  attr.offset     = 0;
  attr.requested_key = 0;
  attr.context    = nullptr;
  // ROCr needs no per-device index, so attr.device stays zeroed.
  attr.iface      = isDevice ? FI_HMEM_ROCR : FI_HMEM_SYSTEM;
  (void)deviceId;

  OFI_CHECK(fi_mr_regattr(n.domain, &attr, 0, &mr.mr), "fi_mr_regattr");
  // Close the MR if any post-registration step fails, so it never leaks.
  try {
    OFI_CHECK(fi_mr_bind(mr.mr, &n.ep->fid, 0), "fi_mr_bind");
    OFI_CHECK(fi_mr_enable(mr.mr), "fi_mr_enable MR");
    mr.key = fi_mr_key(mr.mr);
    if (mr.key == FI_KEY_NOTAVAIL)
      throw std::runtime_error("fi_mr_key returned FI_KEY_NOTAVAIL");
  } catch (...) {
    if (mr.mr) { fi_close(&mr.mr->fid); mr.mr = nullptr; }
    throw;
  }
  MORI_IO_INFO("RegisterMr: nic={} buf=0x{:x} sz={} key=0x{:x} device={}",
               nic, (uintptr_t)buf, size, mr.key, isDevice);
  return mr;
}

void OfiManager::DeregisterMr(OfiMemRegion& mr) {
  if (mr.mr) { fi_close(&mr.mr->fid); mr.mr = nullptr; }
  mr.key = FI_KEY_NOTAVAIL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  OfiControlPlane — in-memory AV/address registry
 *  Peer endpoint addresses (EngineDesc) and MR keys (MemoryDesc) are carried
 *  inline in backendDescs, so no out-of-band TCP exchange is needed. This class
 *  only inserts advertised peer addrs into the local AVs and returns fi_addr_t.
 * ═══════════════════════════════════════════════════════════════════════════ */

OfiControlPlane::OfiControlPlane(OfiManager* ofi) : ofi_(ofi) {}

OfiControlPlane::~OfiControlPlane() = default;

bool OfiControlPlane::InsertRemoteNicAddrs(const EngineKey& key,
                                           const std::vector<NicAddr>& nicAddrs) {
  if (nicAddrs.empty()) return false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    // Already seeded for local NIC 0 → treat the whole engine as known.
    if (remoteAddrs_.count(0) && remoteAddrs_.at(0).count(key)) return false;
  }

  // Each remote addr must be inserted into every local NIC's AV, since the
  // fi_addr_t an AV returns is only usable for sends from that NIC.
  const size_t localNics = ofi_->NumNics();
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, fi_addr_t>> perLocal;
  for (uint32_t localNic = 0; localNic < localNics; ++localNic) {
    for (const auto& [remoteNic, addr] : nicAddrs) {
      perLocal[localNic][remoteNic] = ofi_->InsertAddr(localNic, addr);
    }
  }

  std::lock_guard<std::mutex> lk(mu_);
  if (remoteAddrs_.count(0) && remoteAddrs_.at(0).count(key)) return false;
  for (auto& [localNic, byRemote] : perLocal) {
    remoteAddrs_[localNic][key] = std::move(byRemote);
  }
  return true;
}

bool OfiControlPlane::SeedRemoteAddrs(const EngineKey& key, const std::vector<NicAddr>& nicAddrs) {
  return InsertRemoteNicAddrs(key, nicAddrs);
}

void OfiControlPlane::RemoveRemoteAddrs(const EngineKey& key) {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [localNic, byEngine] : remoteAddrs_) byEngine.erase(key);
}

fi_addr_t OfiControlPlane::GetRemoteAddr(uint32_t localNic, const EngineKey& key,
                                         uint32_t remoteNic) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto lIt = remoteAddrs_.find(localNic);
  if (lIt == remoteAddrs_.end())
    throw std::runtime_error("OfiCP: no addrs for local nic " + std::to_string(localNic));
  auto eIt = lIt->second.find(key);
  if (eIt == lIt->second.end())
    throw std::runtime_error("OfiCP: no addr for remote engine " + key);
  auto rIt = eIt->second.find(remoteNic);
  if (rIt == eIt->second.end())
    throw std::runtime_error("OfiCP: no addr for remote engine " + key + " nic " +
                             std::to_string(remoteNic));
  return rIt->second;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Helpers: submit + drain fi_read / fi_writemsg
 * ═══════════════════════════════════════════════════════════════════════════ */

// Finalize one op: update its status / shared batch counter and free the ctx.
static void CompleteOp(OfiOpCtx* ctx, StatusCode code, const std::string& msg = {}) {
  if (!ctx) return;
  // For batches, only the op that drives `remaining` to 0 finalizes the status.
  const bool last =
      !ctx->remaining || ctx->remaining->fetch_sub(1, std::memory_order_acq_rel) == 1;
  if (code != StatusCode::SUCCESS) {
    ctx->status->Update(code, msg);
  } else if (last && !ctx->status->Failed()) {
    ctx->status->SetCode(StatusCode::SUCCESS);
  }
  if (ctx->remaining && last) delete ctx->remaining;
  delete ctx;
}

// Post one RMA op without waiting for completion. Returns false (and finalizes
// ctx) if the post itself fails; true if it is in flight and will complete on
// the CQ. Uses fi_writemsg + FI_DELIVERY_COMPLETE for writes, fi_read for reads.
static bool SubmitOfiRw(struct fid_ep* ep, void* localBuf, size_t size, void* localDesc,
                        fi_addr_t peer, uint64_t remoteOffset, uint64_t remoteKey, bool isRead,
                        OfiOpCtx* ctx) {
  ctx->status->SetCode(StatusCode::IN_PROGRESS);

  ssize_t ret;
  if (isRead) {
    ret = fi_read(ep, localBuf, size, localDesc, peer, remoteOffset, remoteKey, ctx);
  } else {
    struct iovec iov = {.iov_base = localBuf, .iov_len = size};
    struct fi_rma_iov rma = {.addr = remoteOffset, .key = remoteKey, .len = size};
    struct fi_msg_rma msg{};
    msg.msg_iov       = &iov;
    msg.desc          = &localDesc;
    msg.iov_count     = 1;
    msg.addr          = peer;
    msg.rma_iov       = &rma;
    msg.rma_iov_count = 1;
    msg.context       = ctx;
    ret = fi_writemsg(ep, &msg, FI_DELIVERY_COMPLETE);
  }

  MORI_IO_DEBUG("SubmitOfiRw: {} buf={} len={} peer=0x{:x} raddr={} key=0x{:x} ret={}",
                isRead ? "read" : "write", localBuf, size, (uint64_t)peer, remoteOffset,
                remoteKey, ret);

  if (ret) {
    CompleteOp(ctx, StatusCode::ERR_RDMA_OP,
               std::string("fi_write/read: ") + fi_strerror((int)-ret));
    return false;
  }
  return true;
}

// Wait for `expected` completions on cq (one per in-flight op), finalizing each.
static void DrainCq(struct fid_cq* cq, size_t expected) {
  size_t done = 0;
  while (done < expected) {
    struct fi_cq_data_entry comp{};
    ssize_t n = fi_cq_read(cq, &comp, 1);
    if (n == 1) {
      CompleteOp(reinterpret_cast<OfiOpCtx*>(comp.op_context), StatusCode::SUCCESS);
      ++done;
    } else if (n == -FI_EAGAIN) {
      continue;
    } else {
      struct fi_cq_err_entry err{};
      fi_cq_readerr(cq, &err, 0);
      const char* es = fi_cq_strerror(cq, err.prov_errno, err.err_data, nullptr, 0);
      MORI_IO_WARN("DrainCq CQ error: {} (prov_errno={})", es, err.prov_errno);
      CompleteOp(reinterpret_cast<OfiOpCtx*>(err.op_context), StatusCode::ERR_RDMA_OP,
                 es ? es : "");
      ++done;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  OfiBackendSession
 * ═══════════════════════════════════════════════════════════════════════════ */

OfiBackendSession::OfiBackendSession(OfiManager* ofi,
                                     OfiMemRegion localMr, uint64_t remoteMrKey,
                                     fi_addr_t remoteAddr, size_t remoteSize)
    : ofi_(ofi), localMr_(localMr), remoteMrKey_(remoteMrKey),
      remoteAddr_(remoteAddr), remoteSize_(remoteSize) {}

OfiBackendSession::~OfiBackendSession() {
  // The session borrows all fabric state: ofi_ is owned by OfiBackend,
  // localMr_ aliases the backend's registered MR (freed by DeregisterMemory),
  // and remoteAddr_ is an AV index owned by the control plane — so none of it is
  // released here (closing the MR would double-free the backend's registration).
  // Transfers use synchronous inline CQ drain, so no op outlives the session.
  alive_ = false;
  ofi_ = nullptr;
  MORI_APP_INFO("Destroying OFI backend session: remoteAddr={} remoteSize={}",
                remoteAddr_, remoteSize_);
}

void OfiBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {
  status->SetCode(StatusCode::IN_PROGRESS);

  const uint32_t nic = localMr_.nicId;
  auto* ctx  = new OfiOpCtx{{}, status, id, nullptr};
  void* lBuf = reinterpret_cast<void*>(localMr_.bufBase + localOffset);
  void* lDesc = fi_mr_desc(localMr_.mr);

  if (SubmitOfiRw(ofi_->Ep(nic), lBuf, size, lDesc, remoteAddr_, remoteOffset, remoteMrKey_,
                  isRead, ctx))
    DrainCq(ofi_->Cq(nic), 1);
}

void OfiBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {
  const size_t n = sizes.size();
  if (n == 0) { status->SetCode(StatusCode::SUCCESS); return; }

  status->SetCode(StatusCode::IN_PROGRESS);

  // Shared counter; last completion signals status.
  const uint32_t nic = localMr_.nicId;
  auto* remaining = new std::atomic<int>(static_cast<int>(n));
  void* lDesc     = fi_mr_desc(localMr_.mr);

  // Pipeline all sub-ops into the NIC, then wait for their completions.
  size_t submitted = 0;
  for (size_t i = 0; i < n; i++) {
    auto* ctx  = new OfiOpCtx{{}, status, id, remaining};
    void* lBuf = reinterpret_cast<void*>(localMr_.bufBase + localOffsets[i]);
    if (SubmitOfiRw(ofi_->Ep(nic), lBuf, sizes[i], lDesc, remoteAddr_, remoteOffsets[i],
                    remoteMrKey_, isRead, ctx))
      ++submitted;
  }
  DrainCq(ofi_->Cq(nic), submitted);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  OfiBackend
 * ═══════════════════════════════════════════════════════════════════════════ */

OfiBackend::OfiBackend(EngineKey key, const IOEngineConfig& cfg,
                       const OfiBackendConfig& ofiCfg)
    : myKey_(key), cfg_(ofiCfg) {
  mori::io::SetLogLevel("debug");
  (void)cfg;  // OFI needs no control-plane host/port; addrs travel inline.
  ofi_    = std::make_unique<OfiManager>(ofiCfg);
  cp_     = std::make_unique<OfiControlPlane>(ofi_.get());

  MORI_IO_INFO("OfiBackend: key={} nics={}", key, ofi_->NumNics());
}

OfiBackend::~OfiBackend() {
  cp_.reset();
  // deregister all MRs
  std::lock_guard<std::mutex> lk(mrMu_);
  for (auto& [id, mr] : localMrs_) ofi_->DeregisterMr(mr);
  localMrs_.clear();
}

void OfiBackend::RegisterRemoteEngine(const EngineDesc& desc) {
  // Peer endpoint addresses travel inline in EngineDesc.backendDescs[OFI]; seed
  // the local AVs directly. There is no TCP fallback.
  auto it = desc.backendDescs.find(BackendType::OFI);
  if (it == desc.backendDescs.end() || it->second.empty()) {
    throw std::runtime_error(
        "OFI: remote EngineDesc for " + desc.key +
        " has no inline OFI descriptor (backendDescs[OFI]); cannot seed addresses");
  }
  auto nicAddrs = ParseOfiEngineBlob(it->second.data(), it->second.size());
  MORI_IO_INFO("OfiBackend: seeding remote {} from EngineDesc ({} nics, {} bytes)", desc.key,
               nicAddrs.size(), it->second.size());
  cp_->SeedRemoteAddrs(desc.key, nicAddrs);
}

void OfiBackend::DeregisterRemoteEngine(const EngineDesc& desc) {
  cp_->RemoveRemoteAddrs(desc.key);
  // Drop this peer's cached remote MR info. Within a single remote engine's
  // lifetime its memory ids are monotonically increasing, so an entry can never
  // go stale in place — a purge is only needed when the remote engine restarts
  // and reuses the same engineKey with ids restarting from 0, which would
  // otherwise alias a dead registration's key/base/size.
  std::lock_guard<std::mutex> lk(remoteMrMu_);
  remoteMrCache_.erase(desc.key);
}

void OfiBackend::RegisterMemory(MemoryDesc& desc) {
  auto* buf = reinterpret_cast<void*>(desc.data);
  const bool isDevice = (desc.loc == MemoryLocationType::GPU);
  // Pick the local NIC/endpoint by GPU/NUMA affinity; the MR is bound to it.
  const uint32_t nic = ofi_->SelectLocalNic(desc.deviceId, desc.numaNode);
  MORI_IO_INFO("Registering memory: id={} size={} loc={} deviceId={}, deviceBusId={} nic={}",
               desc.id, desc.size, static_cast<int>(desc.loc), desc.deviceId, desc.deviceBusId,
               nic);
  OfiMemRegion mr = ofi_->RegisterMr(nic, buf, desc.size, cfg_.mrAccessFlags, isDevice,
                                     desc.deviceId);

  {
    std::lock_guard<std::mutex> lk(mrMu_);
    localMrs_[desc.id] = mr;
  }

  // Publish the MR key + owning NIC inline so peers resolve them from the
  // transported MemoryDesc. Base/size are desc.data/desc.size.
  OfiMrBlob blob{mr.key, mr.nicId};
  msgpack::sbuffer sb;
  msgpack::pack(sb, blob);
  auto& out = desc.backendDescs[BackendType::OFI];
  out.assign(reinterpret_cast<const uint8_t*>(sb.data()),
             reinterpret_cast<const uint8_t*>(sb.data()) + sb.size());

  MORI_IO_DEBUG("OfiBackend: registered mem id={} key=0x{:x} size={} nic={}", desc.id, mr.key,
                desc.size, mr.nicId);
}

void OfiBackend::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lk(mrMu_);
  auto it = localMrs_.find(desc.id);
  if (it != localMrs_.end()) {
    ofi_->DeregisterMr(it->second);
    localMrs_.erase(it);
  }
}

OfiMemRegion* OfiBackend::GetLocalMr(MemoryUniqueId id) {
  auto it = localMrs_.find(id);
  if (it == localMrs_.end()) return nullptr;
  return &it->second;
}

OfiBackend::RemoteMrInfo OfiBackend::GetRemoteMrInfo(const MemoryDesc& remote) {
  {
    std::lock_guard<std::mutex> lk(remoteMrMu_);
    auto eIt = remoteMrCache_.find(remote.engineKey);
    if (eIt != remoteMrCache_.end()) {
      auto mIt = eIt->second.find(remote.id);
      if (mIt != eIt->second.end()) return mIt->second;
    }
  }

  MORI_IO_DEBUG("Fetching remote MR info for engine={} id={}", remote.engineKey, remote.id);
  RemoteMrInfo info{};
  auto it = remote.backendDescs.find(BackendType::OFI);
  if (it != remote.backendDescs.end() && !it->second.empty()) {
    // MR key + owning NIC advertised inline; base/size come from the desc.
    auto oh = msgpack::unpack(reinterpret_cast<const char*>(it->second.data()), it->second.size());
    OfiMrBlob blob = oh.get().as<OfiMrBlob>();
    info = RemoteMrInfo{blob.key, static_cast<uintptr_t>(remote.data), remote.size, blob.nicId};
  } else {
    throw std::runtime_error(
        "OFI: remote MemoryDesc for engine " + remote.engineKey + " id " +
        std::to_string(remote.id) + " has no inline OFI descriptor (backendDescs[OFI])");
  }

  {
    std::lock_guard<std::mutex> lk(remoteMrMu_);
    remoteMrCache_[remote.engineKey][remote.id] = info;
  }
  return info;
}

void OfiBackend::ReadWrite(const MemoryDesc& local, size_t localOffset,
                           const MemoryDesc& remote, size_t remoteOffset,
                           size_t size, TransferStatus* status,
                           TransferUniqueId id, bool isRead) {
  // Copy the local MR handle out under the lock; the transfer + CQ drain then
  // run without holding mrMu_ (matches the lock-free session path).
  OfiMemRegion lmr;
  {
    std::lock_guard<std::mutex> lk(mrMu_);
    auto* p = GetLocalMr(local.id);
    if (!p) {
      status->Update(StatusCode::ERR_NOT_FOUND,
                     "OFI: local MR not found id=" + std::to_string(local.id));
      return;
    }
    lmr = *p;
  }

  auto rInfo  = GetRemoteMrInfo(remote);
  const uint32_t localNic = lmr.nicId;
  fi_addr_t   rAddr = cp_->GetRemoteAddr(localNic, remote.engineKey, rInfo.nicId);
  MORI_IO_DEBUG("ReadWrite: eng={} id={} key=0x{:x} base=0x{:x} lnic={} rnic={} fa=0x{:x}",
                remote.engineKey, (unsigned)remote.id, rInfo.key, (unsigned long)rInfo.base,
                localNic, rInfo.nicId, (uint64_t)rAddr);

  auto* ctx  = new OfiOpCtx{{}, status, id, nullptr};
  void* lBuf = reinterpret_cast<void*>(lmr.bufBase + localOffset);
  void* lDesc = fi_mr_desc(lmr.mr);

  if (SubmitOfiRw(ofi_->Ep(localNic), lBuf, size, lDesc, rAddr, remoteOffset, rInfo.key,
                  isRead, ctx))
    DrainCq(ofi_->Cq(localNic), 1);
}

void OfiBackend::BatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                                const MemoryDesc& remote, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status,
                                TransferUniqueId id, bool isRead) {
  const size_t n = sizes.size();
  if (n == 0) { status->SetCode(StatusCode::SUCCESS); return; }

  // Copy the local MR handle out under the lock; submit/drain run lock-free.
  OfiMemRegion lmr;
  {
    std::lock_guard<std::mutex> lk(mrMu_);
    auto* p = GetLocalMr(local.id);
    if (!p) {
      status->Update(StatusCode::ERR_NOT_FOUND, "OFI: local MR not found");
      return;
    }
    lmr = *p;
  }

  auto      rInfo = GetRemoteMrInfo(remote);
  const uint32_t localNic = lmr.nicId;
  fi_addr_t rAddr = cp_->GetRemoteAddr(localNic, remote.engineKey, rInfo.nicId);
  auto*     rem   = new std::atomic<int>(static_cast<int>(n));
  void*     lDesc = fi_mr_desc(lmr.mr);

  status->SetCode(StatusCode::IN_PROGRESS);
  size_t submitted = 0;
  for (size_t i = 0; i < n; i++) {
    auto* ctx  = new OfiOpCtx{{}, status, id, rem};
    void* lBuf = reinterpret_cast<void*>(lmr.bufBase + localOffsets[i]);
    if (SubmitOfiRw(ofi_->Ep(localNic), lBuf, sizes[i], lDesc, rAddr, remoteOffsets[i],
                    rInfo.key, isRead, ctx))
      ++submitted;
  }
  DrainCq(ofi_->Cq(localNic), submitted);
}

BackendSession* OfiBackend::CreateSession(const MemoryDesc& local,
                                          const MemoryDesc& remote) {
  MORI_APP_INFO("Creating OFI backend session for local engine={} id={} remote engine={} id={}",
                local.engineKey, local.id, remote.engineKey, remote.id);
  // Copy the local MR handle out under the lock; the rest runs lock-free.
  OfiMemRegion lmr;
  {
    std::lock_guard<std::mutex> lk(mrMu_);
    auto* p = GetLocalMr(local.id);
    if (!p) throw std::runtime_error("OFI: local MR not found for session");
    lmr = *p;
  }

  auto      rInfo = GetRemoteMrInfo(remote);
  fi_addr_t rAddr = cp_->GetRemoteAddr(lmr.nicId, remote.engineKey, rInfo.nicId);

  return new OfiBackendSession(ofi_.get(), lmr,
                               rInfo.key, rAddr, rInfo.size);
}

bool OfiBackend::PopInboundTransferStatus(EngineKey /*remote*/, TransferUniqueId /*id*/,
                                          TransferStatus* /*status*/) {
  // fi_write/fi_read are initiator-side completions only; no inbound tracking needed.
  return false;
}

DescBlob OfiBackend::GetEngineDescBlob() const {
  // Publish every local endpoint's addr_name (per-NIC list) so peers can seed
  // their AVs pre-handshake and target the right endpoint per remote MR.
  return ofi_ ? ofi_->BuildEngineBlob() : DescBlob{};
}

}  // namespace io
}  // namespace mori
