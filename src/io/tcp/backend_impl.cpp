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
    // Set listening socket non-blocking for EPOLLET safety
    int flags = fcntl(lfd, F_GETFL, 0);
    if (flags >= 0) fcntl(lfd, F_SETFL, flags | O_NONBLOCK);
  }
  ctx->Listen();
  StartService();
  MORI_IO_INFO("TcpBackend created host {} port {}", engConfig.host.c_str(), engConfig.port);
}

TcpBackend::~TcpBackend() { StopService(); }

void TcpBackend::StartService() {
  if (running.load()) return;
  running.store(true);
  executor.reset(new MultithreadTCPExecutor(ctx.get(), config.numWorkerThreads));
  serviceThread = std::thread([this] { ServiceLoop(); });
}

void TcpBackend::StopService() {
  running.store(false);
  if (serviceThread.joinable()) serviceThread.join();
  executor->Shutdown();
  if (ctx) ctx->Close();
}

void TcpBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  executor->RegisterRemoteEngine(rdesc);
  if (auto* mexec = dynamic_cast<MultithreadTCPExecutor*>(executor.get())) {
    mexec->EnsureConnections(rdesc, config.numWorkerThreads);
  }
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  executor->DeregisterRemoteEngine(rdesc);
  if (auto* mexec = dynamic_cast<MultithreadTCPExecutor*>(executor.get())) {
    mexec->CloseConnections(rdesc.key);
  }
}

void TcpBackend::RegisterMemory(const MemoryDesc& desc) { executor->RegisterMemory(desc); }

void TcpBackend::DeregisterMemory(const MemoryDesc& desc) { executor->DeregisterMemory(desc); }

void TcpBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                           const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id, bool isRead) {
  status->SetCode(StatusCode::IN_PROGRESS);
  ReadWriteWork work{localDest, localOffset, remoteSrc, remoteOffset, size, status, id, isRead};
  if (executor->SubmitReadWriteWork(work) != 0) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("executor shutdown");
    return;
  }
  // synchronous wait for completion
  while (status->Code() == StatusCode::IN_PROGRESS) {
    std::this_thread::yield();
  }

  return;
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {
  if (sizes.empty()) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  // Submit all work items
  if (localOffsets.size() != sizes.size() || remoteOffsets.size() != sizes.size()) {
    status->SetCode(StatusCode::ERR_INVALID_ARGS);
    status->SetMessage("BatchReadWrite: offsets and sizes vector size mismatch");
    return;
  }
  status->SetCode(StatusCode::IN_PROGRESS);
  std::vector<TransferStatus> itemStatuses(sizes.size());

  for (size_t i = 0; i < sizes.size(); ++i) {
    itemStatuses[i].SetCode(StatusCode::IN_PROGRESS);
    ReadWriteWork work{localDest, localOffsets[i],  remoteSrc, remoteOffsets[i],
                       sizes[i],  &itemStatuses[i], id,        isRead};
    if (executor->SubmitReadWriteWork(work) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      status->SetMessage("executor shutdown");
      return;
    }
  }

  // Busy wait; TODO replace with condition variable once TransferStatus supports signaling
  bool anyFailed = false;
  for (auto& s : itemStatuses) {
    s.Wait();
    if (s.Failed()) anyFailed = true;
  }
  if (anyFailed) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("one or more batch items failed");
    return;
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
            // Non-blocking for edge-triggered epoll
            int cflags = fcntl(h.fd, F_GETFL, 0);
            if (cflags >= 0) fcntl(h.fd, F_SETFL, cflags | O_NONBLOCK);
          }
          epoll_event nev{};
          nev.events = EPOLLIN | EPOLLET;
          nev.data.fd = h.fd;
          epoll_ctl(epfd, EPOLL_CTL_ADD, h.fd, &nev);
        }
        continue;
      }
      // submit processing to service worker
      executor->SubmitServiceWork(fd);
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
