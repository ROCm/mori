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
#include "src/io/sim/backend_impl.hpp"

#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <msgpack.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/flat_yaml.hpp"
#include "src/io/rdma/backend_impl.hpp"

namespace mori {
namespace io {

namespace {

std::optional<double> ParseNonNegativeDouble(const char* raw) {
  errno = 0;
  char* end = nullptr;
  double v = std::strtod(raw, &end);
  if (errno != 0 || end == raw || *end != '\0' || !std::isfinite(v) || v < 0) return std::nullopt;
  return v;
}

std::optional<double> ParsePositiveDouble(const char* raw) {
  auto v = ParseNonNegativeDouble(raw);
  if (!v.has_value() || *v == 0) return std::nullopt;
  return v;
}

std::optional<BackendType> ParseSimBackendType(const char* raw) {
  using mori::env::detail::EqualsIgnoreCase;
  if (EqualsIgnoreCase(raw, "rdma")) return BackendType::RDMA;
  if (EqualsIgnoreCase(raw, "tcp")) return BackendType::TCP;
  if (EqualsIgnoreCase(raw, "xgmi")) return BackendType::XGMI;
  return std::nullopt;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Simulated control plane                                    */
/* ---------------------------------------------------------------------------------------------- */
// Frame: 1-byte type, 4-byte big-endian body length, msgpack body. Both ends are SimBackends.
enum class SimMsgType : uint8_t {
  AskMemory = 1,
  AskMemoryReply = 2,
  Notif = 3,
};

struct SimAskMemory {
  EngineKey asker;
  MemoryUniqueId id{0};
  MSGPACK_DEFINE(asker, id);
};

struct SimAskMemoryReply {
  MemoryUniqueId id{0};
  bool found{false};
  uint64_t size{0};
  MSGPACK_DEFINE(id, found, size);
};

struct SimNotif {
  EngineKey sender;
  TransferUniqueId id{0};
  int totalNum{1};
  MSGPACK_DEFINE(sender, id, totalNum);
};

constexpr uint32_t kMaxSimMsgBytes = 1 << 20;
constexpr size_t kMaxPendingFailures = 1024;

class SimPeerClosed : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

template <typename T>
void WriteSimMsg(int fd, SimMsgType type, const T& body) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, body);
  std::vector<char> frame(5 + buf.size());
  frame[0] = static_cast<char>(type);
  uint32_t len = htonl(static_cast<uint32_t>(buf.size()));
  std::memcpy(frame.data() + 1, &len, sizeof(len));
  std::memcpy(frame.data() + 5, buf.data(), buf.size());
  application::TCPEndpoint ep({fd, {}});
  if (ep.Send(frame.data(), frame.size()) != 0) {
    throw std::system_error(errno, std::generic_category(), "sim control-plane send");
  }
}

std::pair<SimMsgType, std::vector<char>> ReadSimMsg(int fd) {
  application::TCPEndpoint ep({fd, {}});
  char hdr[5];
  size_t got = 0;
  int rc = ep.Recv(hdr, sizeof(hdr), &got);
  if (rc == 1 && got == 0) throw SimPeerClosed("peer closed the connection");
  if (rc != 0) throw std::system_error(errno, std::generic_category(), "sim control-plane recv");
  uint32_t len = 0;
  std::memcpy(&len, hdr + 1, sizeof(len));
  len = ntohl(len);
  if (len > kMaxSimMsgBytes) {
    throw std::runtime_error("sim control-plane message of " + std::to_string(len) +
                             " bytes exceeds limit");
  }
  std::vector<char> body(len);
  if (len > 0 && ep.Recv(body.data(), len) != 0) {
    throw std::system_error(errno, std::generic_category(), "sim control-plane recv body");
  }
  return {static_cast<SimMsgType>(hdr[0]), std::move(body)};
}

template <typename T>
T UnpackSimMsg(const std::vector<char>& body) {
  auto out = msgpack::unpack(body.data(), body.size());
  return out.get().as<T>();
}

bool InBounds(size_t offset, size_t size, size_t limit) {
  return offset <= limit && size <= limit - offset;
}

std::chrono::nanoseconds ToNs(double ns) {
  return std::chrono::nanoseconds(static_cast<int64_t>(std::llround(ns)));
}

}  // namespace

namespace sim {

bool IsEnabled() { return mori::env::IsEnvVarEnabled("MORI_IO_SIM_ENABLE"); }

BackendType ResolveBackendType(BackendType requested) {
  // Only the RDMA request is remapped: it is the one inference engines make for KV transfer.
  if (requested != BackendType::RDMA) return requested;
  BackendType type = requested;
  env::Override("MORI_IO_SIM_BACKEND", type, ParseSimBackendType);
  return type;
}

std::string BackendTypeName(BackendType type) {
  switch (type) {
    case BackendType::XGMI:
      return "XGMI";
    case BackendType::RDMA:
      return "RDMA";
    case BackendType::TCP:
      return "TCP";
    case BackendType::FABRIC:
      return "FABRIC";
    case BackendType::OFI:
      return "OFI";
    default:
      return "Unknown";
  }
}

namespace {

const std::set<std::string> kProfileKeys = {"backend", "bandwidth_gbps", "latency_us",
                                            "per_segment_ns", "per_op_us", "qp_per_transfer",
                                            "num_worker_threads",
                                            // Free text: what the numbers were measured on, since
                                            // they change with each GPU and NIC generation.
                                            "name", "gpu", "nic", "link", "measured", "notes"};

struct ProfileEntry {
  double gbps, latencyUs, segmentNs, opUs;
  int qpPerTransfer, numWorkerThreads;
};

// Validates every entry of the file and returns the one for `backend`, if any.
std::optional<ProfileEntry> ReadProfileFile(const std::string& path, const std::string& backend) {
  std::optional<ProfileEntry> found;
  auto entries = yaml::LoadFlatList(path);
  for (size_t i = 0; i < entries.size(); ++i) {
    const yaml::FlatMap& e = entries[i];
    const std::string where = path + ": entry " + std::to_string(i + 1);
    yaml::CheckKeys(e, kProfileKeys, where);
    std::string name = yaml::GetString(e, "backend");
    if (name != "rdma" && name != "tcp" && name != "xgmi") {
      throw std::runtime_error(where + ": backend must be rdma, tcp or xgmi, got '" + name + "'");
    }
    for (const char* key : {"bandwidth_gbps", "latency_us", "per_segment_ns", "per_op_us"}) {
      if (e.count(key) == 0) throw std::runtime_error(where + ": missing " + key);
    }
    ProfileEntry p{yaml::GetDouble(e, "bandwidth_gbps", 0, where),
                   yaml::GetDouble(e, "latency_us", 0, where),
                   yaml::GetDouble(e, "per_segment_ns", 0, where),
                   yaml::GetDouble(e, "per_op_us", 0, where),
                   static_cast<int>(yaml::GetInt(e, "qp_per_transfer", -1, where)),
                   static_cast<int>(yaml::GetInt(e, "num_worker_threads", -1, where))};
    if (!(p.gbps > 0) || p.latencyUs < 0 || p.segmentNs < 0 || p.opUs < 0) {
      throw std::runtime_error(where + ": bandwidth must be positive and costs non-negative");
    }
    if (name != backend) continue;
    if (found) throw std::runtime_error(where + ": second entry for backend " + name);
    found = p;
  }
  return found;
}

}  // namespace

LinkProfile LoadLinkProfile(BackendType type) {
  // Uncalibrated guesses, for when no profile is given: a 400G RoCE NIC, kernel TCP on a 100G
  // NIC, one MI3xx XGMI link. Measured profiles are in python/mori/io/sim_profiles/.
  ProfileEntry p{45.0, 5.0, 20.0, 0.0, -1, -1};
  if (type == BackendType::TCP) {
    p = {10.0, 30.0, 200.0, 0.0, -1, -1};
  } else if (type == BackendType::XGMI) {
    p = {50.0, 10.0, 0.0, 0.0, -1, -1};
  }
  LinkProfile out;
  out.source = "built-in";

  std::string backend = BackendTypeName(type);
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::stringstream paths(mori::env::GetString("MORI_IO_SIM_PROFILE").value_or(""));
  for (std::string path; std::getline(paths, path, ',');) {
    if (path.empty()) continue;
    if (auto entry = ReadProfileFile(path, backend)) {
      p = *entry;
      out.source = path;
    }
  }

  const std::string prefix = "MORI_IO_SIM_" + BackendTypeName(type);
  env::Override((prefix + "_BW_GBPS").c_str(), p.gbps, ParsePositiveDouble);
  env::Override((prefix + "_LATENCY_US").c_str(), p.latencyUs, ParseNonNegativeDouble);
  env::Override((prefix + "_SEGMENT_NS").c_str(), p.segmentNs, ParseNonNegativeDouble);
  env::Override((prefix + "_OP_US").c_str(), p.opUs, ParseNonNegativeDouble);

  out.bytesPerNs = p.gbps;
  out.latency = ToNs(p.latencyUs * 1000.0);
  out.perSegment = ToNs(p.segmentNs);
  out.perOp = ToNs(p.opUs * 1000.0);
  out.measuredQpPerTransfer = p.qpPerTransfer;
  out.measuredNumWorkerThreads = p.numWorkerThreads;
  return out;
}

std::map<std::string, double> ProfileValues(BackendType type) {
  LinkProfile p = LoadLinkProfile(type);
  return {{"bw_gbps", p.bytesPerNs},
          {"latency_us", p.latency.count() / 1e3},
          {"per_segment_ns", static_cast<double>(p.perSegment.count())},
          {"per_op_us", p.perOp.count() / 1e3}};
}

}  // namespace sim

/* ---------------------------------------------------------------------------------------------- */
/*                                        SimBackendSession                                       */
/* ---------------------------------------------------------------------------------------------- */
void SimBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {
  backend_->Submit(local_, {localOffset}, remote_, {remoteOffset}, {size}, status, id, isRead);
}

void SimBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {
  backend_->Submit(local_, localOffsets, remote_, remoteOffsets, sizes, status, id, isRead);
}

bool SimBackendSession::Alive() const { return backend_->PeerAlive(remote_.engineKey); }

/* ---------------------------------------------------------------------------------------------- */
/*                                           SimBackend                                           */
/* ---------------------------------------------------------------------------------------------- */
SimBackend::SimBackend(BackendType type, const EngineDesc& self, const IOEngineConfig& engConfig,
                       bool enableNotification)
    : type_(type),
      self_(self),
      // XGMI never notifies the remote side, matching XgmiBackend.
      enableNotification_(enableNotification && type != BackendType::XGMI),
      profile_(sim::LoadLinkProfile(type)) {
  // Same meaning as the GPU-free sglang simulator's VGPU_TIME_SCALE: run S times faster.
  env::Override("MORI_IO_SIM_TIME_SCALE", timeScale_, ParsePositiveDouble);

  if (HasControlPlane()) {
    ctx_ = std::make_unique<application::TCPContext>(engConfig.host, engConfig.port);
    ctx_->Listen();
    epfd_ = epoll_create1(EPOLL_CLOEXEC);
    if (epfd_ < 0) throw std::system_error(errno, std::generic_category(), "epoll_create1");
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = ctx_->GetListenFd();
    if (epoll_ctl(epfd_, EPOLL_CTL_ADD, ctx_->GetListenFd(), &ev) != 0) {
      throw std::system_error(errno, std::generic_category(), "epoll_ctl(listen fd)");
    }
    serverRunning_.store(true);
    serverThd_ = std::thread([this] { ServerLoop(); });
  }
  timerThd_ = std::thread([this] { TimerLoop(); });

  MORI_IO_INFO(
      "SimBackend({}) created for engine {}: profile {}, bw {} GB/s latency {} ns segment {} ns "
      "op {} ns time_scale {} notification {} port {}",
      sim::BackendTypeName(type_), self_.key, profile_.source, profile_.bytesPerNs,
      profile_.latency.count(), profile_.perSegment.count(), profile_.perOp.count(), timeScale_,
      enableNotification_, ctx_ ? static_cast<int>(ctx_->GetPort()) : -1);
  if (profile_.source == "built-in") {
    MORI_IO_WARN(
        "SimBackend({}) uses uncalibrated built-in timing; set MORI_IO_SIM_PROFILE to a measured "
        "profile (python/mori/io/sim_profiles/)",
        sim::BackendTypeName(type_));
  }
}

SimBackend::~SimBackend() {
  {
    std::lock_guard<std::mutex> lock(timerMu_);
    stopping_ = true;
  }
  timerCv_.notify_all();
  if (timerThd_.joinable()) timerThd_.join();

  serverRunning_.store(false);
  if (serverThd_.joinable()) serverThd_.join();
  if (epfd_ >= 0) close(epfd_);
  // TCPContext::Close also closes the peer connections it opened in Connect.
  ctx_.reset();
}

std::optional<uint16_t> SimBackend::GetListenPort() const {
  if (!ctx_) return std::nullopt;
  return ctx_->GetPort();
}

void SimBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(peersMu_);
  auto& peer = peers_[rdesc.key];
  if (peer && peer->desc == rdesc) return;
  peer = std::make_shared<Peer>();
  peer->desc = rdesc;
}

void SimBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  // In-flight transfers keep their Peer alive, as RDMA QPs outlive the engine registration.
  std::lock_guard<std::mutex> lock(peersMu_);
  peers_.erase(rdesc.key);
}

void SimBackend::RegisterMemory(MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memsMu_);
  mems_[desc.id] = desc;
}

void SimBackend::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memsMu_);
  mems_.erase(desc.id);
}

void SimBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                           const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id, bool isRead) {
  Submit(localDest, {localOffset}, remoteSrc, {remoteOffset}, {size}, status, id, isRead);
}

void SimBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {
  Submit(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id, isRead);
}

BackendSession* SimBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  if (HasControlPlane()) {
    auto peer = GetPeer(remote.engineKey);
    if (!peer) {
      throw std::runtime_error("SimBackend::CreateSession: remote engine " + remote.engineKey +
                               " is not registered");
    }
    EnsurePeerReady(peer, remote);
  }
  return new SimBackendSession(this, local, remote);
}

bool SimBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                          TransferStatus* status) {
  std::lock_guard<std::mutex> lock(notifMu_);
  auto engineIt = notifPool_.find(remote);
  if (engineIt == notifPool_.end()) return false;
  auto it = engineIt->second.find(id);
  if (it == engineIt->second.end() || it->second != 0) return false;
  status->SetCode(StatusCode::SUCCESS);
  return true;
}

bool SimBackend::PopPeerFailure(PeerFailureEvent* out) {
  std::lock_guard<std::mutex> lock(failuresMu_);
  if (failures_.empty()) return false;
  *out = std::move(failures_.front());
  failures_.pop_front();
  return true;
}

bool SimBackend::CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const {
  if (type_ == BackendType::XGMI) {
    return local.loc == MemoryLocationType::GPU && remote.loc == MemoryLocationType::GPU &&
           IsSameNode(remote.engineKey);
  }
  auto peer = GetPeer(remote.engineKey);
  return peer && peer->desc.port != internal::kXgmiOnlyFallbackPlaceholderPort;
}

bool SimBackend::IsSameNode(const EngineKey& remote) const {
  if (remote == self_.key) return true;
  auto peer = GetPeer(remote);
  if (!peer) return false;
  if (!self_.nodeId.empty() && !peer->desc.nodeId.empty()) {
    return self_.nodeId == peer->desc.nodeId;
  }
  return self_.hostname == peer->desc.hostname;
}

std::shared_ptr<SimBackend::Peer> SimBackend::GetPeer(const EngineKey& remote) const {
  std::lock_guard<std::mutex> lock(peersMu_);
  auto it = peers_.find(remote);
  return it == peers_.end() ? nullptr : it->second;
}

bool SimBackend::PeerAlive(const EngineKey& remote) const {
  if (!HasControlPlane()) return true;
  auto peer = GetPeer(remote);
  return peer && !peer->dead.load();
}

void SimBackend::EnsurePeerReady(const std::shared_ptr<Peer>& peer, const MemoryDesc& remote) {
  std::lock_guard<std::mutex> lock(peer->mu);
  if (peer->dead.load() || peer->verifiedMems.count(remote.id) != 0) return;

  const EngineKey& ekey = peer->desc.key;
  if (!peer->ep.has_value()) {
    peer->ep = ctx_->Connect(peer->desc.host, peer->desc.port);
    MORI_IO_INFO("SimBackend({}) connected engine {} to {} at {}:{}", sim::BackendTypeName(type_),
                 self_.key, ekey, peer->desc.host, peer->desc.port);
  }

  SimAskMemoryReply reply;
  try {
    WriteSimMsg(peer->ep->fd, SimMsgType::AskMemory, SimAskMemory{self_.key, remote.id});
    auto [type, body] = ReadSimMsg(peer->ep->fd);
    if (type != SimMsgType::AskMemoryReply) {
      throw std::runtime_error("unexpected reply type " + std::to_string(static_cast<int>(type)));
    }
    reply = UnpackSimMsg<SimAskMemoryReply>(body);
  } catch (const std::exception& e) {
    MarkPeerDead(*peer, e.what());
    throw std::runtime_error("SimBackend: control plane to engine " + ekey +
                             " failed: " + e.what());
  }

  // Same failure RdmaBackend::CreateSession raises when the fetched MR length differs.
  if (!reply.found || reply.size != remote.size) {
    std::stringstream ss;
    ss << "SimBackend::CreateSession: remote memory size mismatch for engine " << ekey
       << " memId=" << remote.id << " expected=" << remote.size
       << " got=" << (reply.found ? reply.size : 0);
    throw std::runtime_error(ss.str());
  }
  peer->verifiedMems[remote.id] = reply.size;
}

bool SimBackend::PeerSocketClosedLocked(Peer& peer) {
  if (!peer.ep.has_value()) return true;
  char b;
  ssize_t n = recv(peer.ep->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
  if (n == 0) return true;
  return n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}

void SimBackend::MarkPeerDead(Peer& peer, const std::string& detail) {
  if (peer.dead.exchange(true)) return;
  MORI_IO_WARN("SimBackend({}) peer {} unreachable: {}", sim::BackendTypeName(type_), peer.desc.key,
               detail);
  PeerFailureEvent ev;
  ev.remoteEngineKey = peer.desc.key;
  ev.reason = PeerFailureReason::PEER_UNREACHABLE;
  ev.deviceName = "sim";
  ev.detail = detail;
  std::lock_guard<std::mutex> lock(failuresMu_);
  if (failures_.size() < kMaxPendingFailures) failures_.push_back(std::move(ev));
}

void SimBackend::Submit(const MemoryDesc& local, const SizeVec& localOffsets,
                        const MemoryDesc& remote, const SizeVec& remoteOffsets,
                        const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                        bool isRead) {
  if (localOffsets.size() != sizes.size() || remoteOffsets.size() != sizes.size()) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "offset and size vectors differ in length");
    return;
  }
  if (sizes.empty()) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  // A real NIC or DMA engine rejects out-of-range access; catch it here as there is no copy.
  size_t bytes = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (!InBounds(localOffsets[i], sizes[i], local.size) ||
        !InBounds(remoteOffsets[i], sizes[i], remote.size)) {
      std::stringstream ss;
      ss << "simulated transfer " << id << " segment " << i << " out of bounds: local "
         << localOffsets[i] << "+" << sizes[i] << "/" << local.size << ", remote "
         << remoteOffsets[i] << "+" << sizes[i] << "/" << remote.size;
      status->Update(StatusCode::ERR_INVALID_ARGS, ss.str());
      return;
    }
    bytes += sizes[i];
  }

  std::shared_ptr<Peer> peer;
  if (HasControlPlane()) {
    peer = GetPeer(remote.engineKey);
    if (!peer) {
      status->Update(StatusCode::ERR_BAD_STATE,
                     "remote engine " + remote.engineKey + " is not registered");
      return;
    }
    EnsurePeerReady(peer, remote);
    if (peer->dead.load()) {
      status->Update(StatusCode::ERR_RDMA_OP,
                     "simulated peer " + remote.engineKey + " is unreachable");
      return;
    }
  }

  status->SetCode(StatusCode::IN_PROGRESS);
  auto deadline = ReserveLink(isRead, bytes, sizes.size());
  {
    std::lock_guard<std::mutex> lock(timerMu_);
    timerQ_.push(Completion{deadline, timerSeq_++, status, id, std::move(peer)});
  }
  timerCv_.notify_one();
  MORI_IO_TRACE("SimBackend({}) transfer {} {} {} bytes in {} segments to {}",
                sim::BackendTypeName(type_), id, isRead ? "read" : "write", bytes, sizes.size(),
                remote.engineKey);
}

std::chrono::steady_clock::time_point SimBackend::ReserveLink(bool isRead, size_t bytes,
                                                              size_t segments) {
  auto busy =
      ToNs((static_cast<double>(bytes) / profile_.bytesPerNs +
            static_cast<double>(segments) * profile_.perSegment.count() + profile_.perOp.count()) /
           timeScale_);
  auto latency = ToNs(profile_.latency.count() / timeScale_);
  std::lock_guard<std::mutex> lock(linkMu_);
  auto& freeAt = isRead ? rxFreeAt_ : txFreeAt_;
  freeAt = std::max(std::chrono::steady_clock::now(), freeAt) + busy;
  return freeAt + latency;
}

void SimBackend::TimerLoop() {
  // The default 50 us timer slack would make every simulated transfer complete that much late.
  prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
  std::unique_lock<std::mutex> lock(timerMu_);
  while (!stopping_) {
    if (timerQ_.empty()) {
      timerCv_.wait(lock);
      continue;
    }
    auto deadline = timerQ_.top().deadline;
    if (std::chrono::steady_clock::now() < deadline) {
      timerCv_.wait_until(lock, deadline);
      continue;
    }
    Completion c = timerQ_.top();
    timerQ_.pop();
    lock.unlock();
    Complete(c);
    lock.lock();
  }
}

void SimBackend::Complete(Completion& c) {
  if (c.peer) {
    std::string err;
    {
      std::lock_guard<std::mutex> lock(c.peer->mu);
      if (c.peer->dead.load()) {
        err = "peer marked unreachable";
      } else if (PeerSocketClosedLocked(*c.peer)) {
        err = "control-plane connection closed";
      } else if (enableNotification_) {
        try {
          WriteSimMsg(c.peer->ep->fd, SimMsgType::Notif, SimNotif{self_.key, c.id, 1});
        } catch (const std::exception& e) {
          err = e.what();
        }
      }
    }
    if (!err.empty()) {
      MarkPeerDead(*c.peer, err);
      c.status->Update(StatusCode::ERR_RDMA_OP,
                       "simulated transfer to " + c.peer->desc.key + " failed: " + err);
      return;
    }
  }
  c.status->Update(StatusCode::SUCCESS, "");
}

void SimBackend::ServerLoop() {
  constexpr int kMaxEvents = 64;
  epoll_event events[kMaxEvents];
  while (serverRunning_.load()) {
    int nfds = epoll_wait(epfd_, events, kMaxEvents, 5 /*ms*/);
    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;
      if (fd == ctx_->GetListenFd()) {
        for (auto& ep : ctx_->Accept()) {
          epoll_event ev{};
          ev.events = EPOLLIN;
          ev.data.fd = ep.fd;
          if (epoll_ctl(epfd_, EPOLL_CTL_ADD, ep.fd, &ev) != 0) {
            ctx_->CloseEndpointNoThrow(ep);
            continue;
          }
          serverConns_[ep.fd] = ep;
        }
        continue;
      }
      if (!HandleServerMessage(fd)) DropServerConn(fd);
    }
  }
  for (auto& [fd, ep] : serverConns_) ctx_->CloseEndpointNoThrow(ep);
  serverConns_.clear();
}

bool SimBackend::HandleServerMessage(int fd) {
  try {
    // Drain every message already buffered so notification bursts cost one wakeup.
    char b;
    do {
      auto [type, body] = ReadSimMsg(fd);
      switch (type) {
        case SimMsgType::AskMemory: {
          auto req = UnpackSimMsg<SimAskMemory>(body);
          SimAskMemoryReply reply{req.id, false, 0};
          {
            std::lock_guard<std::mutex> lock(memsMu_);
            auto it = mems_.find(req.id);
            if (it != mems_.end()) {
              reply.found = true;
              reply.size = it->second.size;
            }
          }
          WriteSimMsg(fd, SimMsgType::AskMemoryReply, reply);
          break;
        }
        case SimMsgType::Notif: {
          auto msg = UnpackSimMsg<SimNotif>(body);
          std::lock_guard<std::mutex> lock(notifMu_);
          auto& pool = notifPool_[msg.sender];
          auto it = pool.try_emplace(msg.id, msg.totalNum).first;
          it->second -= 1;
          MORI_IO_TRACE("SimBackend({}) notif from {} id {} remaining {}",
                        sim::BackendTypeName(type_), msg.sender, msg.id, it->second);
          break;
        }
        default:
          throw std::runtime_error("unsupported sim control-plane message type " +
                                   std::to_string(static_cast<int>(type)));
      }
    } while (recv(fd, &b, 1, MSG_PEEK | MSG_DONTWAIT) > 0);
    return true;
  } catch (const SimPeerClosed&) {
    return false;
  } catch (const std::exception& e) {
    MORI_IO_WARN("SimBackend({}) control-plane fd {} failed: {}; dropping connection",
                 sim::BackendTypeName(type_), fd, e.what());
    return false;
  }
}

void SimBackend::DropServerConn(int fd) noexcept {
  epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, nullptr);
  serverConns_.erase(fd);
  ctx_->CloseEndpointNoThrow(fd);
}

}  // namespace io
}  // namespace mori
