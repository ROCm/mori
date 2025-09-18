#include "src/io/tcp/backend_impl.hpp"

#include <sys/epoll.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

namespace mori {
namespace io {

TcpBackend::TcpBackend(EngineKey key, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
    : myEngKey(key), config(cfg), engConfig(engCfg) {
  ctx.reset(new application::TCPContext(engConfig.host, engConfig.port));
  ctx->Listen();
  StartService();
  MORI_IO_INFO("TcpBackend created host {} port {}", engConfig.host.c_str(), engConfig.port);
}

TcpBackend::~TcpBackend() { StopService(); }

void TcpBackend::StartService() {
  if (running.load()) return;
  running.store(true);
  serviceThread = std::thread([this] { ServiceLoop(); });
}

void TcpBackend::StopService() {
  running.store(false);
  if (serviceThread.joinable()) serviceThread.join();
  if (ctx) ctx->Close();
}

void TcpBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  remotes[rdesc.key] = rdesc;
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  remotes.erase(rdesc.key);
  conns.erase(rdesc.key);
}

void TcpBackend::RegisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  localMems[desc.id] = desc;
}

void TcpBackend::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  localMems.erase(desc.id);
}

TcpConnection TcpBackend::GetOrCreateConnection(const EngineDesc& rdesc) {
  if (conns.find(rdesc.key) != conns.end() && conns[rdesc.key].Valid()) return conns[rdesc.key];
  auto handle = ctx->Connect(rdesc.host, rdesc.port);
  TcpConnection c(handle);
  conns[rdesc.key] = c;
  return c;
}

void TcpBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                           const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id, bool isRead) {
  status->SetCode(StatusCode::IN_PROGRESS);
  EngineKey remoteKey = remoteSrc.engineKey;
  EngineDesc rdesc;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (remotes.find(remoteKey) == remotes.end()) {
      status->SetCode(StatusCode::ERR_NOT_FOUND);
      status->SetMessage("remote engine not registered");
      return;
    }
    rdesc = remotes[remoteKey];
  }
  TcpConnection conn = GetOrCreateConnection(rdesc);
  if (!conn.Valid()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("tcp connection invalid");
    return;
  }

  TcpMessageHeader hdr{};
  hdr.opcode = isRead ? 0 : 1; // read_req or write_req
  hdr.id = id;
  hdr.offset = remoteOffset;
  hdr.size = size;

  application::TCPEndpoint ep(conn.handle);
  // send header
  ep.Send(&hdr, sizeof(hdr));
  if (!isRead) {
    // write request includes data payload from local memory at localOffset
    const char* src = reinterpret_cast<const char*>(localDest.data + localOffset);
    ep.Send(src, size);
    // Wait for write response
    TcpMessageHeader resp{};
    ep.Recv(&resp, sizeof(resp));
    status->SetCode(StatusCode::SUCCESS);
    return;
  } else {
    // read request, expect read response header + payload
    TcpMessageHeader resp{};
    ep.Recv(&resp, sizeof(resp));
    if (resp.opcode != 2 || resp.size != size) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("unexpected read response");
      return;
    }
    char* dst = reinterpret_cast<char*>(localDest.data + localOffset);
    ep.Recv(dst, size);
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {
  if (sizes.empty()) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  // naive sequential
  for (size_t i = 0; i < sizes.size(); ++i) {
    TransferStatus s;
    ReadWrite(localDest, localOffsets[i], remoteSrc, remoteOffsets[i], sizes[i], &s, id, isRead);
    if (s.Failed()) {
      status->SetCode(s.Code());
      status->SetMessage(s.Message());
      return;
    }
  }
  status->SetCode(StatusCode::SUCCESS);
}

BackendSession* TcpBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  return new TcpBackendSession(this, local, remote);
}

bool TcpBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                          TransferStatus* status) {
  return false; // simplistic synchronous model
}

void TcpBackend::ServiceLoop() {
  int epfd = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd >= 0);
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ev.data.fd = ctx->GetListenFd();
  epoll_ctl(epfd, EPOLL_CTL_ADD, ctx->GetListenFd(), &ev);

  constexpr int maxEvents = 128;
  epoll_event events[maxEvents];

  while (running.load()) {
    int nfds = epoll_wait(epfd, events, maxEvents, 10);
    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;
      if (fd == ctx->GetListenFd()) {
        auto newEps = ctx->Accept();
        for (auto& h : newEps) {
          epoll_event nev{};
            nev.events = EPOLLIN | EPOLLET;
            nev.data.fd = h.fd;
            epoll_ctl(epfd, EPOLL_CTL_ADD, h.fd, &nev);
        }
        continue;
      }
      // handle request
      TcpMessageHeader hdr{};
      ssize_t r = ::recv(fd, &hdr, sizeof(hdr), MSG_WAITALL);
      if (r != sizeof(hdr)) { ::close(fd); continue; }
      if (hdr.opcode == 0) { // read request: send data back
        // find memory (assume only one memory id for now)
        // For now we ignore memory id (not transmitted). Use first memory.
        MemoryDesc target{};
        {
          std::lock_guard<std::mutex> lock(mu);
          if (localMems.empty()) { ::close(fd); continue; }
          target = localMems.begin()->second;
        }
        const char* src = reinterpret_cast<const char*>(target.data + hdr.offset);
        TcpMessageHeader resp{2, hdr.id, hdr.offset, hdr.size};
        ::send(fd, &resp, sizeof(resp), 0);
        ::send(fd, src, hdr.size, 0);
      } else if (hdr.opcode == 1) { // write request: receive payload and respond
        MemoryDesc target{};
        {
          std::lock_guard<std::mutex> lock(mu);
          if (localMems.empty()) { ::close(fd); continue; }
          target = localMems.begin()->second;
        }
        char* dst = reinterpret_cast<char*>(target.data + hdr.offset);
        ssize_t r2 = ::recv(fd, dst, hdr.size, MSG_WAITALL);
        if (r2 != (ssize_t)hdr.size) { ::close(fd); continue; }
        TcpMessageHeader resp{3, hdr.id, hdr.offset, hdr.size};
        ::send(fd, &resp, sizeof(resp), 0);
      } else {
        ::close(fd);
      }
    }
  }
  ::close(epfd);
}

void TcpBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {
  backend->ReadWrite(local, localOffset, remote, remoteOffset, size, status, id, isRead);
}

void TcpBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {
  backend->BatchReadWrite(local, localOffsets, remote, remoteOffsets, sizes, status, id, isRead);
}

} // namespace io
} // namespace mori
