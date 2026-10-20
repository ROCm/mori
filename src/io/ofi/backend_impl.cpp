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
// SPDX-FileCopyrightText: © Hewlett Packard Enterprise Development LP
// MIT License — see root LICENSE file.
//
// MORI-IO OFI (libfabric) backend — Slingshot/CXI compatible.
// Uses FI_EP_RDM + FI_MR_ENDPOINT + FI_MR_PROV_KEY (no FI_MR_VIRT_ADDR).
// Data plane: fi_writemsg / fi_readmsg with offset-based RMA addresses.
// Control plane: none — peer addr-names and MR keys travel inline in EngineDesc/MemoryDesc.

#include "backend_impl.hpp"

#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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
  uint32_t nicId{0};
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

// A NIC discovered from fi_info->nic. Providers that expose a fabric NIC (e.g.
// CXI/Slingshot) carry PCI bus coordinates and link speed inline, so the
// topology can rank GPU→NIC affinity without provider-specific sysfs probing.
struct OfiNicInfo {
  std::string name;
  application::PciBusId busId{0};
  double gbps{0};
};

// Extract per-NIC PCI location + link speed from a fi_info list. Each entry's
// fi->nic (when present) carries a PCI bus_attr and a link_attr whose speed is
// in bits/sec. Entries without a PCI NIC are skipped; duplicates (same BDF) are
// collapsed since fi_getinfo returns one entry per domain.
std::vector<OfiNicInfo> DiscoverNicsFromInfo(struct fi_info* info) {
  std::vector<OfiNicInfo> out;
  for (struct fi_info* fi = info; fi != nullptr; fi = fi->next) {
    const struct fid_nic* nic = fi->nic;
    if (!nic || !nic->bus_attr || nic->bus_attr->bus_type != FI_BUS_PCI) continue;

    const struct fi_pci_attr& pci = nic->bus_attr->attr.pci;
    application::PciBusId busId(pci.domain_id, pci.bus_id, pci.device_id, pci.function_id);
    if (busId.packed == 0) continue;

    // link_attr->speed is bits/sec; 0 means the provider didn't report it.
    double gbps = (nic->link_attr && nic->link_attr->speed > 0)
                      ? static_cast<double>(nic->link_attr->speed) / 1e9
                      : 0.0;

    const char* nm = (nic->device_attr && nic->device_attr->name) ? nic->device_attr->name
                     : (fi->domain_attr && fi->domain_attr->name) ? fi->domain_attr->name
                                                                  : "";
    bool dup = false;
    for (const auto& e : out)
      if (e.busId.packed == busId.packed) {
        dup = true;
        break;
      }
    if (!dup) out.push_back({nm, busId, gbps});
  }
  return out;
}

// Close a NIC's libfabric handles in reverse open order (safe on partial init).
void CloseNic(OfiNic& nic) {
  if (nic.ep) {
    fi_close(&nic.ep->fid);
    nic.ep = nullptr;
  }
  if (nic.cq) {
    fi_close(&nic.cq->fid);
    nic.cq = nullptr;
  }
  if (nic.av) {
    fi_close(&nic.av->fid);
    nic.av = nullptr;
  }
  if (nic.domain) {
    fi_close(&nic.domain->fid);
    nic.domain = nullptr;
  }
  if (nic.fabric) {
    fi_close(&nic.fabric->fid);
    nic.fabric = nullptr;
  }
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
    hints->caps = FI_RMA;  // one-sided only; no two-sided messaging
    hints->mode = FI_CONTEXT | FI_CONTEXT2;

    // Slingshot/CXI: FI_MR_ENDPOINT + FI_MR_PROV_KEY.  No FI_MR_VIRT_ADDR.
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_ENDPOINT | FI_MR_PROV_KEY;

    if (withHmem) {
      // Enable device-memory (ROCm/HBM) registration and transfers.
      hints->caps |= FI_HMEM;
      hints->domain_attr->mr_mode |= FI_MR_HMEM;
    }

    // FI_THREAD_SAFE: provider serializes internally, so multiple threads may
    // share a domain's ep/cq (e.g. lock-free ReadWrite from several threads).
    hints->domain_attr->threading = FI_THREAD_SAFE;

    if (!cfg.providerHint.empty()) hints->fabric_attr->prov_name = strdup(cfg.providerHint.c_str());

    return hints;
  };

  // Require FI_HMEM so GPU buffers can be registered/transferred;
  // if the provider can't negotiate HMEM, fail then fallback to host only
  struct fi_info* hints = makeHints(/*withHmem=*/true);
  int ret = fi_getinfo(FI_VERSION(2, 6), nullptr, nullptr, 0, hints, &info_);
  fi_freeinfo(hints);
  if (ret) {
    // FI_HMEM negotiation failed; try again without HMEM.
    hints = makeHints(/*withHmem=*/false);
    ret = fi_getinfo(FI_VERSION(2, 6), nullptr, nullptr, 0, hints, &info_);
    fi_freeinfo(hints);
    if (ret) {
      throw std::runtime_error(std::string("fi_getinfo (FI_HMEM fallback): ") + fi_strerror(-ret));
    }
    hmemEnabled_ = false;
  } else {
    hmemEnabled_ = true;
  }

  // A throwing constructor does not run ~OfiManager, so clean up info_ and any
  // NICs already opened on failure.
  try {
    topo_.reset(new application::TopoSystem());

    // NICs exposed via fi_info->nic (e.g. CXI/Slingshot) aren't discovered
    // through ibverbs; inject them so GPU→NIC affinity can rank them (only when
    // verbs discovered none).
    if (topo_->GetTopoSystemNet()->NumNics() == 0) {
      for (const auto& n : DiscoverNicsFromInfo(info_))
        topo_->GetTopoSystemNet()->AddNic(n.name, n.busId, n.gbps);
    }

    // Open one endpoint per unique provider domain (i.e. per physical NIC).
    // fi_getinfo returns providers in priority order and may append same-node
    // helpers (e.g. shm/sm2) after the fabric NIC. Lock onto the first (highest
    // priority) provider and skip the rest: the shm provider doesn't support the
    // one-sided RMA + FI_MR_ENDPOINT/FI_MR_PROV_KEY setup we configure and its
    // fi_av_open returns -FI_ENOSYS.
    const char* primaryProv = (info_->fabric_attr && info_->fabric_attr->prov_name)
                                  ? info_->fabric_attr->prov_name
                                  : nullptr;

    uint32_t nicId = 0;
    for (struct fi_info* fi = info_; fi != nullptr; fi = fi->next) {
      const char* dn = (fi->domain_attr && fi->domain_attr->name) ? fi->domain_attr->name : nullptr;
      if (!dn) continue;
      const char* pv =
          (fi->fabric_attr && fi->fabric_attr->prov_name) ? fi->fabric_attr->prov_name : "";
      if (primaryProv && std::strcmp(pv, primaryProv) != 0) {
        MORI_IO_INFO("OfiManager: skipping non-primary provider {} domain {}", pv, dn);
        continue;
      }
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
               info_->fabric_attr->prov_name, nics_.size(), (unsigned)info_->domain_attr->mr_mode,
               hmemEnabled_);

  // Idle-spin budget before the progress thread blocks on the CQ fds.
  if (const char* e = std::getenv("MORI_IO_OFI_BUSY_POLL_ITERS")) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(e, &end, 10);
    if (end != e && v > 0) busyPollIters_ = static_cast<size_t>(v);
  }

  // Completions reaped per fi_cq_read call.
  if (cfg.cqReadBatch > 0) cqReadBatch_ = static_cast<size_t>(cfg.cqReadBatch);

  // Blocking fallback needs a poll()-able fd on every CQ plus an eventfd to wake
  // the thread on shutdown; otherwise the thread yields when idle instead.
  blockingEnabled_ = true;
  for (const auto& nic : nics_)
    if (nic.cqFd < 0) {
      blockingEnabled_ = false;
      break;
    }
  if (blockingEnabled_) {
    shutdownEventFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (shutdownEventFd_ < 0) blockingEnabled_ = false;
  }
  MORI_IO_INFO("OfiManager: progress blocking={} busyPollIters={}", blockingEnabled_,
               busyPollIters_);

  // All NICs are open: start the single CQ-draining thread. Op threads submit
  // and block on their TransferStatus; this thread reaps completions + signals.
  progressRun_.store(true, std::memory_order_release);
  progressThread_ = std::thread(&OfiManager::ProgressLoop, this);
}

void OfiManager::OpenNic(struct fi_info* fi, uint32_t nicId) {
  OfiNic nic;
  nic.nicId = nicId;
  nic.info = fi;
  nic.domainName = fi->domain_attr->name;

  const char* provName =
      (fi->fabric_attr && fi->fabric_attr->prov_name) ? fi->fabric_attr->prov_name : "?";
  MORI_IO_INFO("OpenNic: nicId={} provider={} domain={}", nicId, provName, nic.domainName);

  // Close whatever was opened if any step throws, so a partial NIC never leaks.
  try {
    OFI_CHECK(fi_fabric(fi->fabric_attr, &nic.fabric, nullptr), "fi_fabric");
    OFI_CHECK(fi_domain(nic.fabric, fi, &nic.domain, nullptr), "fi_domain");

    struct fi_av_attr av_attr{};
    av_attr.type = FI_AV_TABLE;
    av_attr.count = 1024;
    OFI_CHECK(fi_av_open(nic.domain, &av_attr, &nic.av, nullptr), "fi_av_open");

    struct fi_cq_attr cq_attr{};
    cq_attr.format = FI_CQ_FORMAT_DATA;
    cq_attr.size = 4096;
    // Request a poll()-able fd so the progress thread can block when idle. Fall
    // back to a wait-less CQ (busy-poll only) if the provider lacks FD waits.
    cq_attr.wait_obj = FI_WAIT_FD;
    cq_attr.wait_cond = FI_CQ_COND_NONE;
    if (fi_cq_open(nic.domain, &cq_attr, &nic.cq, nullptr) == 0) {
      int fd = -1;
      if (fi_control(&nic.cq->fid, FI_GETWAIT, &fd) == 0 && fd >= 0) nic.cqFd = fd;
    } else {
      cq_attr.wait_obj = FI_WAIT_NONE;
      OFI_CHECK(fi_cq_open(nic.domain, &cq_attr, &nic.cq, nullptr), "fi_cq_open");
    }

    OFI_CHECK(fi_endpoint(nic.domain, fi, &nic.ep, nullptr), "fi_endpoint");
    // RMA ops (fi_read/fi_writemsg) complete on the transmit context; we never
    // post receives, so bind the CQ to FI_TRANSMIT only.
    OFI_CHECK(fi_ep_bind(nic.ep, &nic.cq->fid, FI_TRANSMIT), "fi_ep_bind CQ");
    OFI_CHECK(fi_ep_bind(nic.ep, &nic.av->fid, 0), "fi_ep_bind AV");
    OFI_CHECK(fi_enable(nic.ep), "fi_enable ep");
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
  if (info_) {
    fi_freeinfo(info_);
    info_ = nullptr;
  }
}

OfiManager::~OfiManager() {
  // Stop the drain thread before tearing down the CQs it reads. Signal the
  // eventfd so a thread blocked in poll() wakes immediately instead of waiting.
  progressRun_.store(false, std::memory_order_release);
  if (shutdownEventFd_ >= 0) {
    uint64_t one = 1;
    ssize_t w = write(shutdownEventFd_, &one, sizeof(one));
    (void)w;
  }
  if (progressThread_.joinable()) progressThread_.join();
  if (shutdownEventFd_ >= 0) {
    close(shutdownEventFd_);
    shutdownEventFd_ = -1;
  }
  CloseAll();
}

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
    na.nicId = nic.nicId;
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

void OfiManager::RemoveAddr(uint32_t localNic, fi_addr_t addr) {
  if (addr == FI_ADDR_UNSPEC) return;
  int ret = fi_av_remove(nics_.at(localNic).av, &addr, 1, 0);
  if (ret)
    MORI_IO_WARN("fi_av_remove(nic={}, addr=0x{:x}) failed: {}", localNic, (uint64_t)addr,
                 fi_strerror(-ret));
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
  if (accessFlags == 0) accessFlags = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (isDevice && !hmemEnabled_)
    throw std::runtime_error(
        "OfiManager::RegisterMr: GPU buffer registration requested but the OFI "
        "provider did not negotiate FI_HMEM. Rebuild libfabric with ROCm/HMEM "
        "support or use a host-memory KV cache.");

  OfiNic& n = nics_.at(nic);
  OfiMemRegion mr{};
  mr.bufBase = reinterpret_cast<uintptr_t>(buf);
  mr.size = size;
  mr.nicId = nic;

  // Use fi_mr_regattr so we can set the HMEM interface for device buffers.
  struct iovec iov = {.iov_base = buf, .iov_len = size};
  struct fi_mr_attr attr{};
  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access = accessFlags;
  attr.offset = 0;
  attr.requested_key = 0;
  attr.context = nullptr;
  // ROCr needs no per-device index, so attr.device stays zeroed.
  attr.iface = isDevice ? FI_HMEM_ROCR : FI_HMEM_SYSTEM;
  (void)deviceId;

  OFI_CHECK(fi_mr_regattr(n.domain, &attr, 0, &mr.mr), "fi_mr_regattr");
  // Close the MR if any post-registration step fails, so it never leaks.
  try {
    OFI_CHECK(fi_mr_bind(mr.mr, &n.ep->fid, 0), "fi_mr_bind");
    OFI_CHECK(fi_mr_enable(mr.mr), "fi_mr_enable MR");
    mr.key = fi_mr_key(mr.mr);
    if (mr.key == FI_KEY_NOTAVAIL) throw std::runtime_error("fi_mr_key returned FI_KEY_NOTAVAIL");
  } catch (...) {
    if (mr.mr) {
      fi_close(&mr.mr->fid);
      mr.mr = nullptr;
    }
    throw;
  }
  MORI_IO_INFO("RegisterMr: nic={} buf=0x{:x} sz={} key=0x{:x} device={}", nic, (uintptr_t)buf,
               size, mr.key, isDevice);
  return mr;
}

void OfiManager::DeregisterMr(OfiMemRegion& mr) {
  if (mr.mr) {
    fi_close(&mr.mr->fid);
    mr.mr = nullptr;
  }
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
  // Free each AV slot before dropping the cached handle; erasing alone leaks the
  // insertion, eventually exhausting the fixed-size AV on repeated re-register.
  for (auto& [localNic, byEngine] : remoteAddrs_) {
    auto eIt = byEngine.find(key);
    if (eIt == byEngine.end()) continue;
    for (const auto& [remoteNic, addr] : eIt->second) ofi_->RemoveAddr(localNic, addr);
    byEngine.erase(eIt);
  }
}

bool OfiControlPlane::HasEngine(const EngineKey& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = remoteAddrs_.find(0);
  return it != remoteAddrs_.end() && it->second.count(key) > 0;
}

fi_addr_t OfiControlPlane::GetRemoteAddr(uint32_t localNic, const EngineKey& key,
                                         uint32_t remoteNic) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto lIt = remoteAddrs_.find(localNic);
  if (lIt == remoteAddrs_.end())
    throw std::runtime_error("OfiCP: no addrs for local nic " + std::to_string(localNic));
  auto eIt = lIt->second.find(key);
  if (eIt == lIt->second.end()) throw std::runtime_error("OfiCP: no addr for remote engine " + key);
  auto rIt = eIt->second.find(remoteNic);
  if (rIt == eIt->second.end())
    throw std::runtime_error("OfiCP: no addr for remote engine " + key + " nic " +
                             std::to_string(remoteNic));
  return rIt->second;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Helpers: submit + drain fi_read / fi_writemsg
 * ═══════════════════════════════════════════════════════════════════════════ */

// Shared state for a batch of sub-ops. `remaining` counts sub-ops not yet
// accounted for; the first error is latched here so the batch's TransferStatus
// is only published terminal when the final sub-op completes — until then the
// status stays IN_PROGRESS so any drainer keeps reaping this batch's CQ entries.
struct OfiBatchState {
  std::atomic<int> remaining;
  std::atomic<bool> hasError{false};
  StatusCode errCode{StatusCode::SUCCESS};
  std::string errMsg;
  std::mutex errMu;
  explicit OfiBatchState(int n) : remaining(n) {}
};

// Finalize one op: publish its status and free the ctx (and batch, when last).
// A single op's status is decided by its one completion. A batch op latches the
// first error and only publishes the terminal status on the last sub-op, so the
// status never goes terminal while sibling completions are still on the CQ.
static void CompleteOp(OfiOpCtx* ctx, StatusCode code, const std::string& msg = {}) {
  if (!ctx) return;
  OfiBatchState* batch = ctx->batch;
  TransferStatus* status = ctx->status;

  if (!batch) {
    if (code != StatusCode::SUCCESS) {
      status->Update(code, msg);
    } else if (!status->Failed()) {
      status->SetCode(StatusCode::SUCCESS);
    }
    delete ctx;
    return;
  }

  if (code != StatusCode::SUCCESS && !batch->hasError.exchange(true)) {
    std::lock_guard<std::mutex> lk(batch->errMu);
    batch->errCode = code;
    batch->errMsg = msg;
  }
  const bool last = batch->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1;
  if (last) {
    if (batch->hasError.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lk(batch->errMu);
      status->Update(batch->errCode, batch->errMsg);
    } else if (!status->Failed()) {
      status->SetCode(StatusCode::SUCCESS);
    }
    delete batch;
  }
  delete ctx;
}

// Post one RMA op without waiting for completion. Returns false (and finalizes
// ctx) if the post itself fails; true if it is in flight and will complete on
// the CQ. Uses fi_writemsg + FI_DELIVERY_COMPLETE for writes, fi_read for reads.
static bool SubmitOfiRw(struct fid_ep* ep, void* localBuf, size_t size, void* localDesc,
                        fi_addr_t peer, uint64_t remoteOffset, uint64_t remoteKey, bool isRead,
                        OfiOpCtx* ctx) {
  ctx->status->SetCode(StatusCode::IN_PROGRESS);

  struct iovec iov = {.iov_base = localBuf, .iov_len = size};
  struct fi_rma_iov rma = {.addr = remoteOffset, .key = remoteKey, .len = size};
  struct fi_msg_rma msg{};
  msg.msg_iov = &iov;
  msg.desc = &localDesc;
  msg.iov_count = 1;
  msg.addr = peer;
  msg.rma_iov = &rma;
  msg.rma_iov_count = 1;
  msg.context = ctx;

  // -FI_EAGAIN is transient TX/CQ backpressure (a batch posts every sub-op
  // before the CQ is drained). Yield so the progress thread reaps completions
  // and frees queue space, then retry the post rather than failing the op.
  ssize_t ret;
  for (;;) {
    ret = isRead ? fi_read(ep, localBuf, size, localDesc, peer, remoteOffset, remoteKey, ctx)
                 : fi_writemsg(ep, &msg, FI_DELIVERY_COMPLETE);
    if (ret != -FI_EAGAIN) break;
    std::this_thread::yield();
  }

  MORI_IO_DEBUG("SubmitOfiRw: {} buf={} len={} peer=0x{:x} raddr={} key=0x{:x} ret={}",
                isRead ? "read" : "write", localBuf, size, (uint64_t)peer, remoteOffset, remoteKey,
                ret);

  if (ret) {
    CompleteOp(ctx, StatusCode::ERR_RDMA_OP,
               std::string("fi_write/read: ") + fi_strerror((int)-ret));
    return false;
  }
  return true;
}

// Drain one NIC's CQ: reap up to `batch` ready completions (and error entries),
// finalizing each op's TransferStatus via CompleteOp. Returns true if any entry
// was reaped. Only the progress thread calls this, so the CQ has a single reader
// — no concurrent fi_cq_read / fi_cq_readerr races.
static bool ProcessNicCq(struct fid_cq* cq, size_t batch) {
  std::vector<struct fi_cq_data_entry> comp(batch);
  ssize_t n = fi_cq_read(cq, comp.data(), batch);
  if (n > 0) {
    for (ssize_t i = 0; i < n; ++i)
      CompleteOp(reinterpret_cast<OfiOpCtx*>(comp[i].op_context), StatusCode::SUCCESS);
    return true;
  }
  if (n == -FI_EAVAIL) {
    struct fi_cq_err_entry err{};
    while (fi_cq_readerr(cq, &err, 0) == 1) {
      const char* es = fi_cq_strerror(cq, err.prov_errno, err.err_data, nullptr, 0);
      MORI_IO_WARN("OFI CQ error: {} (prov_errno={})", es, err.prov_errno);
      CompleteOp(reinterpret_cast<OfiOpCtx*>(err.op_context), StatusCode::ERR_RDMA_OP,
                 es ? es : "");
      err = {};
    }
    return true;
  }
  if (n != -FI_EAGAIN) MORI_IO_WARN("OFI fi_cq_read failed: {}", fi_strerror((int)-n));
  return false;
}

// Single CQ-draining thread: sweep every NIC's CQ, finalizing completed ops so
// their waiters (blocked in TransferStatus::Wait) are signaled. After
// busyPollIters_ idle sweeps it blocks on the CQ fds (when available) instead of
// spinning; otherwise it yields.
void OfiManager::ProgressLoop() {
  uint64_t emptySweeps = 0;
  while (progressRun_.load(std::memory_order_acquire)) {
    bool didWork = false;
    for (auto& nic : nics_)
      if (ProcessNicCq(nic.cq, cqReadBatch_)) didWork = true;
    if (didWork) {
      emptySweeps = 0;
      continue;
    }
    if (!blockingEnabled_) {
      std::this_thread::yield();
      continue;
    }
    if (++emptySweeps < busyPollIters_) continue;
    BlockUntilActivity();
    emptySweeps = 0;
  }
}

// Block until any NIC CQ has activity or shutdown is signaled. libfabric requires
// fi_trywait to succeed for a CQ before sleeping on its fd; -FI_EAGAIN (or an
// unsupported trywait) means a completion may be pending, so resume polling
// rather than risk sleeping on an already-ready CQ.
void OfiManager::BlockUntilActivity() {
  for (auto& nic : nics_) {
    struct fid* f = &nic.cq->fid;
    if (fi_trywait(nic.fabric, &f, 1) != FI_SUCCESS) return;
  }

  std::vector<struct pollfd> pfds;
  pfds.reserve(nics_.size() + 1);
  for (auto& nic : nics_) pfds.push_back({nic.cqFd, POLLIN, 0});
  pfds.push_back({shutdownEventFd_, POLLIN, 0});

  int ret = poll(pfds.data(), pfds.size(), -1);
  if (ret <= 0) return;
  // Drain the shutdown eventfd if it fired; its value is irrelevant.
  if (pfds.back().revents & POLLIN) {
    uint64_t v;
    while (read(shutdownEventFd_, &v, sizeof(v)) == static_cast<ssize_t>(sizeof(v))) {
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  OfiBackendSession
 * ═══════════════════════════════════════════════════════════════════════════ */

OfiBackendSession::OfiBackendSession(OfiManager* ofi, OfiMemRegion localMr, uint64_t remoteMrKey,
                                     fi_addr_t remoteAddr, size_t remoteSize)
    : ofi_(ofi),
      localMr_(localMr),
      remoteMrKey_(remoteMrKey),
      remoteAddr_(remoteAddr),
      remoteSize_(remoteSize) {}

OfiBackendSession::~OfiBackendSession() {
  // The session borrows all fabric state: ofi_ is owned by OfiBackend,
  // localMr_ aliases the backend's registered MR (freed by DeregisterMemory),
  // and remoteAddr_ is an AV index owned by the control plane — so none of it is
  // released here (closing the MR would double-free the backend's registration).
  // Completions are reaped by OfiManager's progress thread; callers block on the
  // TransferStatus, so no op outlives the session under correct use.
  alive_ = false;
  ofi_ = nullptr;
  MORI_APP_INFO("Destroying OFI backend session: remoteAddr={} remoteSize={}", remoteAddr_,
                remoteSize_);
}

void OfiBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {
  status->SetCode(StatusCode::IN_PROGRESS);

  const uint32_t nic = localMr_.nicId;
  auto* ctx = new OfiOpCtx{{}, status, id, nullptr};
  void* lBuf = reinterpret_cast<void*>(localMr_.bufBase + localOffset);
  void* lDesc = fi_mr_desc(localMr_.mr);

  // Submit and return; the progress thread signals `status` on completion.
  SubmitOfiRw(ofi_->Ep(nic), lBuf, size, lDesc, remoteAddr_, remoteOffset, remoteMrKey_, isRead,
              ctx);
}

void OfiBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {
  const size_t n = sizes.size();
  if (localOffsets.size() != n || remoteOffsets.size() != n) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "OFI: batch offset/size count mismatch");
    return;
  }
  if (n == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  status->SetCode(StatusCode::IN_PROGRESS);

  // Shared batch state; status goes terminal only when the last sub-op completes.
  const uint32_t nic = localMr_.nicId;
  auto* batch = new OfiBatchState(static_cast<int>(n));
  void* lDesc = fi_mr_desc(localMr_.mr);

  // Pipeline all sub-ops into the NIC; the progress thread signals `status`
  // once the final sub-op completes.
  for (size_t i = 0; i < n; i++) {
    auto* ctx = new OfiOpCtx{{}, status, id, batch};
    void* lBuf = reinterpret_cast<void*>(localMr_.bufBase + localOffsets[i]);
    SubmitOfiRw(ofi_->Ep(nic), lBuf, sizes[i], lDesc, remoteAddr_, remoteOffsets[i], remoteMrKey_,
                isRead, ctx);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  OfiBackend
 * ═══════════════════════════════════════════════════════════════════════════ */

OfiBackend::OfiBackend(EngineKey key, const IOEngineConfig& cfg, const OfiBackendConfig& ofiCfg)
    : myKey_(key), cfg_(ofiCfg) {
  (void)cfg;  // OFI needs no control-plane host/port; addrs travel inline.
  ofi_ = std::make_unique<OfiManager>(ofiCfg);
  cp_ = std::make_unique<OfiControlPlane>(ofi_.get());

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
  // the local AVs directly. IOEngine fans this call out to every backend, so a
  // peer with no OFI metadata (e.g. RDMA-only) is simply skipped here rather
  // than aborting registration for the other backends. CanHandle then rejects
  // any route to such a peer.
  auto it = desc.backendDescs.find(BackendType::OFI);
  if (it == desc.backendDescs.end() || it->second.empty()) {
    MORI_IO_INFO("OfiBackend: remote {} has no inline OFI descriptor; skipping", desc.key);
    return;
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
  OfiMemRegion mr =
      ofi_->RegisterMr(nic, buf, desc.size, cfg_.mrAccessFlags, isDevice, desc.deviceId);

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
    throw std::runtime_error("OFI: remote MemoryDesc for engine " + remote.engineKey + " id " +
                             std::to_string(remote.id) +
                             " has no inline OFI descriptor (backendDescs[OFI])");
  }

  {
    std::lock_guard<std::mutex> lk(remoteMrMu_);
    remoteMrCache_[remote.engineKey][remote.id] = info;
  }
  return info;
}

void OfiBackend::ReadWrite(const MemoryDesc& local, size_t localOffset, const MemoryDesc& remote,
                           size_t remoteOffset, size_t size, TransferStatus* status,
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

  auto rInfo = GetRemoteMrInfo(remote);
  const uint32_t localNic = lmr.nicId;
  fi_addr_t rAddr = cp_->GetRemoteAddr(localNic, remote.engineKey, rInfo.nicId);
  MORI_IO_DEBUG("ReadWrite: eng={} id={} key=0x{:x} base=0x{:x} lnic={} rnic={} fa=0x{:x}",
                remote.engineKey, (unsigned)remote.id, rInfo.key, (unsigned long)rInfo.base,
                localNic, rInfo.nicId, (uint64_t)rAddr);

  auto* ctx = new OfiOpCtx{{}, status, id, nullptr};
  void* lBuf = reinterpret_cast<void*>(lmr.bufBase + localOffset);
  void* lDesc = fi_mr_desc(lmr.mr);

  // Submit and return; the progress thread signals `status` on completion.
  SubmitOfiRw(ofi_->Ep(localNic), lBuf, size, lDesc, rAddr, remoteOffset, rInfo.key, isRead, ctx);
}

void OfiBackend::BatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                                const MemoryDesc& remote, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {
  const size_t n = sizes.size();
  if (n == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

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

  auto rInfo = GetRemoteMrInfo(remote);
  const uint32_t localNic = lmr.nicId;
  fi_addr_t rAddr = cp_->GetRemoteAddr(localNic, remote.engineKey, rInfo.nicId);
  auto* batch = new OfiBatchState(static_cast<int>(n));
  void* lDesc = fi_mr_desc(lmr.mr);

  status->SetCode(StatusCode::IN_PROGRESS);
  for (size_t i = 0; i < n; i++) {
    auto* ctx = new OfiOpCtx{{}, status, id, batch};
    void* lBuf = reinterpret_cast<void*>(lmr.bufBase + localOffsets[i]);
    SubmitOfiRw(ofi_->Ep(localNic), lBuf, sizes[i], lDesc, rAddr, remoteOffsets[i], rInfo.key,
                isRead, ctx);
  }
}

BackendSession* OfiBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
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

  auto rInfo = GetRemoteMrInfo(remote);
  fi_addr_t rAddr = cp_->GetRemoteAddr(lmr.nicId, remote.engineKey, rInfo.nicId);

  return new OfiBackendSession(ofi_.get(), lmr, rInfo.key, rAddr, rInfo.size);
}

bool OfiBackend::PopInboundTransferStatus(EngineKey /*remote*/, TransferUniqueId /*id*/,
                                          TransferStatus* /*status*/) {
  // fi_write/fi_read are initiator-side completions only; no inbound tracking needed.
  return false;
}

bool OfiBackend::CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const {
  (void)local;
  // Only route to a peer that advertised OFI endpoints (seeded into the AVs) and
  // whose remote memory carries an inline OFI MR descriptor.
  return cp_->HasEngine(remote.engineKey) && remote.backendDescs.count(BackendType::OFI) > 0;
}

DescBlob OfiBackend::GetEngineDescBlob() const {
  // Publish every local endpoint's addr_name (per-NIC list) so peers can seed
  // their AVs pre-handshake and target the right endpoint per remote MR.
  return ofi_ ? ofi_->BuildEngineBlob() : DescBlob{};
}

}  // namespace io
}  // namespace mori
