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
#include "src/io/tcp/backend_impl.hpp"

#include <netinet/tcp.h>
#include <sys/epoll.h>

int FullSend(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = ::send(fd, p, remaining, 0);
    if (n == 0) {
      errno = ECONNRESET;
      return -1;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;  // spin; could epoll later
      return -1;
    }
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return 0;
}

int FullWritev(int fd, struct iovec* iov, int iovcnt) {
  int idx = 0;
  while (idx < iovcnt) {
    ssize_t n = ::writev(fd, &iov[idx], iovcnt - idx);
    if (n == 0) {
      errno = ECONNRESET;
      return -1;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return -1;
    }
    ssize_t consumed = n;
    while (consumed > 0 && idx < iovcnt) {
      if (consumed >= static_cast<ssize_t>(iov[idx].iov_len)) {
        consumed -= static_cast<ssize_t>(iov[idx].iov_len);
        ++idx;
      } else {
        iov[idx].iov_base = static_cast<char*>(iov[idx].iov_base) + consumed;
        iov[idx].iov_len -= consumed;
        consumed = 0;
      }
    }
  }
  return 0;
}

namespace mori {
namespace io {

TcpBackend::TcpBackend(EngineKey key, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
    : myEngKey(key), config(cfg), engConfig(engCfg) {
  ctx.reset(new application::TCPContext(engConfig.host, engConfig.port));
  // Set basic socket options on listening socket
  int lfd = ctx->GetListenFd();
  if (lfd >= 0) {
    int one = 1;
    ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    // Optional: enlarge listen socket buffers (kernel will clone defaults to accepted sockets)
    int bufSz = 4 * 1024 * 1024;  // 4MB
    ::setsockopt(lfd, SOL_SOCKET, SO_RCVBUF, &bufSz, sizeof(bufSz));
    ::setsockopt(lfd, SOL_SOCKET, SO_SNDBUF, &bufSz, sizeof(bufSz));
  }
  bufferPool.Configure(config.buffer_pool_max_buffers, config.buffer_pool_max_bytes,
                       config.buffer_pool_pinned);
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
    std::lock_guard<std::mutex> lock(remotesMu);
    remotes[rdesc.key] = rdesc;
  }
  if (config.preconnect) {
    // Establish persistent data connection immediately
    (void)GetOrCreateConnection(rdesc);
  }
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(remotesMu);
  remotes.erase(rdesc.key);
  std::lock_guard<std::mutex> lock2(connsMu);
  auto it = conns.find(rdesc.key);
  if (it != conns.end()) {
    if (it->second.handle.fd >= 0) {
      ::close(it->second.handle.fd);
    }
    conns.erase(it);
  }
}

void TcpBackend::RegisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems[desc.id] = desc;
}

void TcpBackend::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems.erase(desc.id);
}

TcpConnection TcpBackend::GetOrCreateConnection(const EngineDesc& rdesc) {
  {
    std::lock_guard<std::mutex> lock(connsMu);
    auto it = conns.find(rdesc.key);
    if (it != conns.end() && it->second.Valid()) return it->second;
  }

  // Connect outside lock to avoid blocking other operations
  auto handle = ctx->Connect(rdesc.host, rdesc.port);
  if (handle.fd >= 0) {
    int flag = 1;
    setsockopt(handle.fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));  // for small msg
    setsockopt(handle.fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
    int bufSz = 4 * 1024 * 1024;  // 4MB send/recv buffers
    setsockopt(handle.fd, SOL_SOCKET, SO_RCVBUF, &bufSz, sizeof(bufSz));
    setsockopt(handle.fd, SOL_SOCKET, SO_SNDBUF, &bufSz, sizeof(bufSz));

    int qack = 1;
    setsockopt(handle.fd, IPPROTO_TCP, TCP_QUICKACK, &qack, sizeof(qack));
  }
  {
    std::lock_guard<std::mutex> lock(connsMu);
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
    std::lock_guard<std::mutex> lock(remotesMu);
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
  hdr.opcode = isRead ? 0 : 1;  // read_req or write_req
  hdr.id = id;
  hdr.mem_id = remoteSrc.id;  // specify remote memory id explicitly
  hdr.offset = remoteOffset;
  hdr.size = size;

  application::TCPEndpoint ep(conn.handle);
  bool localIsGpu = (localDest.loc == MemoryLocationType::GPU);
  BufferBlock bufBlock;  // only allocate if GPU staging needed
  char* stagingPtr = nullptr;

  if (!isRead) {
    const char* sendPtr = nullptr;
    if (localIsGpu) {
      bufBlock = bufferPool.Acquire(size);
      stagingPtr = bufBlock.data;
      const void* devPtr = reinterpret_cast<const void*>(localDest.data + localOffset);
      hipError_t e = hipMemcpy(stagingPtr, devPtr, size, hipMemcpyDeviceToHost);
      if (e != hipSuccess) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        status->SetMessage(std::string("hipMemcpy D2H failed: ") + hipGetErrorString(e));
        if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
        return;
      }
      sendPtr = stagingPtr;
    } else {
      // zero-copy host path
      sendPtr = reinterpret_cast<const char*>(localDest.data + localOffset);
    }
    struct iovec iov[2];
    iov[0].iov_base = &hdr;
    iov[0].iov_len = sizeof(hdr);
    iov[1].iov_base = const_cast<char*>(sendPtr);
    iov[1].iov_len = size;
    if (FullWritev(conn.handle.fd, iov, 2) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("writev failed");
      if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
      return;
    }
    TcpMessageHeader resp{};
    if (ep.Recv(&resp, sizeof(resp)) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("write ack recv failed");
      if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
      return;
    }
    status->SetCode(StatusCode::SUCCESS);
    if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
    return;
  } else {
    if (FullSend(conn.handle.fd, &hdr, sizeof(hdr)) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("read header send failed");
      return;
    }
    TcpMessageHeader resp{};
    if (ep.Recv(&resp, sizeof(resp)) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("read resp header recv failed");
      return;
    }
    if (resp.opcode != 2 || resp.size != size) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("unexpected read response");
      return;
    }
    if (localIsGpu) {
      bufBlock = bufferPool.Acquire(size);
      stagingPtr = bufBlock.data;
      if (ep.Recv(stagingPtr, size) != 0) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        status->SetMessage("read payload recv failed");
        if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
        return;
      }
      void* devPtr = reinterpret_cast<void*>(localDest.data + localOffset);
      hipError_t e = hipMemcpy(devPtr, stagingPtr, size, hipMemcpyHostToDevice);
      if (e != hipSuccess) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        status->SetMessage(std::string("hipMemcpy H2D failed: ") + hipGetErrorString(e));
        if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
        return;
      }
      status->SetCode(StatusCode::SUCCESS);
      if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
      return;
    } else {
      char* dst = reinterpret_cast<char*>(localDest.data + localOffset);
      if (ep.Recv(dst, size) != 0) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        status->SetMessage("read payload recv failed");
        return;
      }
      status->SetCode(StatusCode::SUCCESS);
      return;
    }
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
  return false;  // simplistic synchronous model
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
          if (h.fd >= 0) {
            int flag = 1;
            setsockopt(h.fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
            setsockopt(h.fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
            int bufSz = 4 * 1024 * 1024;
            setsockopt(h.fd, SOL_SOCKET, SO_RCVBUF, &bufSz, sizeof(bufSz));
            setsockopt(h.fd, SOL_SOCKET, SO_SNDBUF, &bufSz, sizeof(bufSz));

            int qack = 1;
            setsockopt(h.fd, IPPROTO_TCP, TCP_QUICKACK, &qack, sizeof(qack));
          }
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
      if (r != sizeof(hdr)) {
        ctx->CloseFd(fd);
        continue;
      }
      if (hdr.opcode == 0 || hdr.opcode == 1) {  // read or write
        MemoryDesc target{};
        {
          std::lock_guard<std::mutex> lock(memMu);
          auto it = localMems.find(hdr.mem_id);
          if (it == localMems.end()) {
            ctx->CloseFd(fd);
            continue;
          }
          target = it->second;
        }
        bool targetIsGpu = (target.loc == MemoryLocationType::GPU);
        auto bufBlock = bufferPool.Acquire(hdr.size);
        char* hostBuf = bufBlock.data;
        if (hdr.opcode == 0) {  // read request: copy from target to host and send
          if (targetIsGpu) {
            const void* devPtr = reinterpret_cast<const void*>(target.data + hdr.offset);
            if (hipMemcpy(hostBuf, devPtr, hdr.size, hipMemcpyDeviceToHost) != hipSuccess) {
              bufferPool.Release(std::move(bufBlock));
              ctx->CloseFd(fd);
              continue;
            }
          } else {
            const char* src = reinterpret_cast<const char*>(target.data + hdr.offset);
            std::memcpy(hostBuf, src, hdr.size);
          }
          TcpMessageHeader resp{};
          resp.opcode = 2;
          resp.id = hdr.id;
          resp.mem_id = hdr.mem_id;
          resp.offset = hdr.offset;
          resp.size = hdr.size;
          ::send(fd, &resp, sizeof(resp), 0);
          ::send(fd, hostBuf, hdr.size, 0);
          bufferPool.Release(std::move(bufBlock));
        } else {  // write request: recv payload into host then copy to device if needed
          ssize_t r2 = ::recv(fd, hostBuf, hdr.size, MSG_WAITALL);
          if (r2 != (ssize_t)hdr.size) {
            bufferPool.Release(std::move(bufBlock));
            ctx->CloseFd(fd);
            continue;
          }
          if (targetIsGpu) {
            void* devPtr = reinterpret_cast<void*>(target.data + hdr.offset);
            if (hipMemcpy(devPtr, hostBuf, hdr.size, hipMemcpyHostToDevice) != hipSuccess) {
              bufferPool.Release(std::move(bufBlock));
              ctx->CloseFd(fd);
              continue;
            }
          } else {
            char* dst = reinterpret_cast<char*>(target.data + hdr.offset);
            std::memcpy(dst, hostBuf, hdr.size);
          }
          TcpMessageHeader resp{};
          resp.opcode = 3;
          resp.id = hdr.id;
          resp.mem_id = hdr.mem_id;
          resp.offset = hdr.offset;
          resp.size = hdr.size;
          ::send(fd, &resp, sizeof(resp), 0);
          bufferPool.Release(std::move(bufBlock));
        }
        break;
      } else {
        ctx->CloseFd(fd);
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

}  // namespace io
}  // namespace mori
