#include "src/io/tcp/backend_impl.hpp"

#include <sys/epoll.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <netinet/tcp.h>

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
  {
    std::lock_guard<std::mutex> lock(mu);
    remotes[rdesc.key] = rdesc;
  }
  if (config.preconnect) {
    // Establish persistent data connection immediately
    (void)GetOrCreateConnection(rdesc);
  }
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  remotes.erase(rdesc.key);
  auto it = conns.find(rdesc.key);
  if (it != conns.end()) {
    if (it->second.handle.fd >= 0) {
      ::close(it->second.handle.fd);
    }
    conns.erase(it);
  }
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
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = conns.find(rdesc.key);
    if (it != conns.end() && it->second.Valid()) return it->second;
  }

  // Connect outside lock to avoid blocking other operations
  auto handle = ctx->Connect(rdesc.host, rdesc.port);
  if (handle.fd >= 0) {
    int flag = 1;
    setsockopt(handle.fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));  // for small msg
    setsockopt(handle.fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    TcpConnection c(handle);
    conns[rdesc.key] = c;
    MORI_IO_INFO("TCP persistent connection established to {}:{} (fd={})", rdesc.host.c_str(),
                 rdesc.port, handle.fd);
    return conns[rdesc.key];
  }
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
  // host staging buffers when GPU memory involved
  bool localIsGpu = (localDest.loc == MemoryLocationType::GPU);
  // For remote side we don't know its loc explicitly; protocol does not carry it yet.
  // We will always send/recv host buffers over TCP.
  std::unique_ptr<char[]> hostBuf(new char[size]);
  char* hostPtr = hostBuf.get();

  // If write (sending data), and source is GPU, copy device->host first.
  if (!isRead) {
    if (localIsGpu) {
      const void* devPtr = reinterpret_cast<const void*>(localDest.data + localOffset);
      hipError_t e = hipMemcpy(hostPtr, devPtr, size, hipMemcpyDeviceToHost);
      if (e != hipSuccess) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        status->SetMessage(std::string("hipMemcpy D2H failed: ") + hipGetErrorString(e));
        return;
      }
    } else {
      const char* src = reinterpret_cast<const char*>(localDest.data + localOffset);
      std::memcpy(hostPtr, src, size);
    }
  }

  // send header
  ep.Send(&hdr, sizeof(hdr));
  if (!isRead) {
    ep.Send(hostPtr, size);
    TcpMessageHeader resp{};
    ep.Recv(&resp, sizeof(resp));
    status->SetCode(StatusCode::SUCCESS);
    return;
  } else {
    TcpMessageHeader resp{};
    ep.Recv(&resp, sizeof(resp));
    if (resp.opcode != 2 || resp.size != size) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("unexpected read response");
      return;
    }
    ep.Recv(hostPtr, size);
    if (localIsGpu) {
      void* devPtr = reinterpret_cast<void*>(localDest.data + localOffset);
      hipError_t e = hipMemcpy(devPtr, hostPtr, size, hipMemcpyHostToDevice);
      if (e != hipSuccess) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        status->SetMessage(std::string("hipMemcpy H2D failed: ") + hipGetErrorString(e));
        return;
      }
    } else {
      char* dst = reinterpret_cast<char*>(localDest.data + localOffset);
      std::memcpy(dst, hostPtr, size);
    }
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
      if (hdr.opcode == 0 || hdr.opcode == 1) { // read or write
        MemoryDesc target{};
        {
          std::lock_guard<std::mutex> lock(mu);
          if (localMems.empty()) { ::close(fd); continue; }
          target = localMems.begin()->second;
        }
        bool targetIsGpu = (target.loc == MemoryLocationType::GPU);
        std::unique_ptr<char[]> hostBuf(new char[hdr.size]);
        if (hdr.opcode == 0) { // read request: copy from target to host and send
          if (targetIsGpu) {
            const void* devPtr = reinterpret_cast<const void*>(target.data + hdr.offset);
            if (hipMemcpy(hostBuf.get(), devPtr, hdr.size, hipMemcpyDeviceToHost) != hipSuccess) {
              ::close(fd); continue; }
          } else {
            const char* src = reinterpret_cast<const char*>(target.data + hdr.offset);
            std::memcpy(hostBuf.get(), src, hdr.size);
          }
          TcpMessageHeader resp{2, hdr.id, hdr.offset, hdr.size};
            ::send(fd, &resp, sizeof(resp), 0);
            ::send(fd, hostBuf.get(), hdr.size, 0);
        } else { // write request: recv payload into host then copy to device if needed
          ssize_t r2 = ::recv(fd, hostBuf.get(), hdr.size, MSG_WAITALL);
          if (r2 != (ssize_t)hdr.size) { ::close(fd); continue; }
          if (targetIsGpu) {
            void* devPtr = reinterpret_cast<void*>(target.data + hdr.offset);
            if (hipMemcpy(devPtr, hostBuf.get(), hdr.size, hipMemcpyHostToDevice) != hipSuccess) {
              ::close(fd); continue; }
          } else {
            char* dst = reinterpret_cast<char*>(target.data + hdr.offset);
            std::memcpy(dst, hostBuf.get(), hdr.size);
          }
          TcpMessageHeader resp{3, hdr.id, hdr.offset, hdr.size};
          ::send(fd, &resp, sizeof(resp), 0);
        }
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
