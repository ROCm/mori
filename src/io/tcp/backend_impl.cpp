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
  epollFd = epoll_create1(0);
  if (epollFd == -1) {
    MORI_IO_ERROR("TcpBackend: epoll_create1 failed: {}", strerror(errno));
    return;
  }

  bufferPool.Configure(128, 16 * 1024 * 1024, true);  // TODO: make configurable
  for (int i = 0; i < config.numWorkerThreads; ++i) {
    application::TCPContext* listenctx =
        new application::TCPContext(engConfig.host, engConfig.port);
    listenctx->Listen();
    int lfd = listenctx->GetListenFd();
    if (lfd > 0) {
      SetNonBlocking(lfd);
    } else {
      MORI_IO_ERROR("TcpBackend: failed to get listen fd");
      delete listenctx;
      return;
    }
    listeners.push_back(listenctx);
    workerThreads.push_back(std::thread([this, listenctx] { WorkerLoop(listenctx); }));
  }
  running.store(true);
}

TcpBackend::~TcpBackend() {
  running.store(false);
  for (auto& t : workerThreads) {
    if (t.joinable()) t.join();
  }
  workerThreads.clear();
}

void TcpBackend::WorkerLoop(application::TCPContext* ctx) {
  struct epoll_event events;

  int workerEpollFd = epoll_create1(0);
  if (workerEpollFd == -1) {
    MORI_IO_ERROR("TcpBackend: epoll_create1 failed in worker: {}", strerror(errno));
    return;
  }
  int listenFd = ctx->GetListenFd();
  // add listen fd to epoll
  events.data.fd = listenFd;
  events.events = EPOLLIN | EPOLLET;
  if (epoll_ctl(workerEpollFd, EPOLL_CTL_ADD, listenFd, &events) == -1) {
    MORI_IO_ERROR("TcpBackend: epoll_ctl ADD listen fd failed: {}", strerror(errno));
    close(workerEpollFd);
    return;
  }

  while (running.load()) {
    int n = epoll_wait(workerEpollFd, &events, 10, 1000);  // 1 second timeout
    if (n == -1) {
      if (errno == EINTR) continue;
      MORI_IO_ERROR("TcpBackend: epoll_wait failed: {}", strerror(errno));
      break;
    } else if (n == 0) {
      continue;  // timeout
    }

    if (events.data.fd == listenFd) {
      HandleNewConnection(ctx, workerEpollFd);
    } else {
      ConnectionState* conn = reinterpret_cast<ConnectionState*>(events.data.ptr);
      if (events.events & EPOLLIN) {
        HandleReadable(conn);
      }
      if (events.events & EPOLLOUT) {
        HandleWritable(conn);
      }
      if (events.events & (EPOLLHUP | EPOLLERR)) {
        MORI_IO_ERROR("TcpBackend: connection closed or error on fd {}", conn->handle.fd);
        conn->Close();
        delete conn;
      }

      // Re-arm the socket since we used EPOLLONESHOT
      RearmSocket(workerEpollFd, conn, EPOLLIN | EPOLLOUT);
    }
  }
  close(workerEpollFd);
}

void TcpBackend::HandleNewConnection(application::TCPContext* listener_ctx, int epoll_fd) {
  auto newEps = listener_ctx->Accept();
  for (const auto& ep : newEps) {
    std::unique_ptr<ConnectionState> conn = std::make_unique<ConnectionState>();
    conn->recvstate = ConnectionState::RecvState::PARSING_HEADER;
    conn->handle = ep;
    conn->listener = listener_ctx;

    SetNonBlocking(ep.fd);
    SetSocketOptions(ep.fd);

    struct epoll_event event;
    event.data.ptr = conn.get();
    event.events = EPOLLIN | EPOLLET | EPOLLOUT | EPOLLONESHOT;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ep.fd, &event) == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_ctl ADD new connection fd failed: {}", strerror(errno));
      listener_ctx->CloseEndpoint(ep);
      continue;
    }

    std::lock_guard<std::mutex> lock(inConnsMu);
    inboundConnections[ep.fd] = std::move(conn);
  }
}

void TcpBackend::HandleReadable(ConnectionState* conn) {
  // This function implements the protocol parsing state machine.
  if (conn->recvstate == ConnectionState::RecvState::PARSING_HEADER) {
    application::TCPEndpoint ep(conn->handle);
    if (ep.Recv(&conn->pendingheader, sizeof(TcpMessageHeader)) != 0) {
      MORI_IO_ERROR("TcpBackend: recv header failed on fd {}: {}", conn->handle.fd,
                    strerror(errno));
      return;
    }
    conn->recvstate = ConnectionState::RecvState::PARSING_PAYLOAD;
  }

  if (conn->recvstate == ConnectionState::RecvState::PARSING_PAYLOAD) {
    auto& header = conn->pendingheader;
    switch (header.opcode) {
      case 0:  // READ_REQ
      {
      } break;
      case 1:  // WRITE_REQ
      {
      } break;
      case 2:  // READ RESP
      {
      } break;
      case 3:  // WRITE RESP
      {
        TransferOp& op = conn->pendingOps[header.id];
        op.status->SetCode(StatusCode::SUCCESS);
        conn->pendingOps.erase(header.id);
        conn->recvstate = ConnectionState::RecvState::PARSING_HEADER;
      } break;
    }
  }

  conn->Reset();
}

void TcpBackend::HandleWritable(ConnectionState* conn) {
  TransferOp& op = conn->PopTransfer();
  application::TCPEndpoint ep(conn->handle);
  switch (op.opType) {
    case 0:  // READ_REQ
    {
      TcpMessageHeader hdr{};
      hdr.opcode = 0;  // read_req or write_req
      hdr.id = op.id;
      hdr.mem_id = op.remoteDest.id;  // specify remote memory id explicitly
      hdr.offset = op.remoteOffset;
      hdr.size = op.size;
      if (ep.Send(&hdr, sizeof(hdr)) != 0) {
        MORI_IO_ERROR("TcpBackend: send read_req header failed on fd {}: {}", conn->handle.fd,
                      strerror(errno));
        op.status->SetCode(StatusCode::ERR_BAD_STATE);
        op.status->SetMessage("send read_req header failed");
        return;
      }
    } break;
    case 1:  // WRITE_REQ
    {
      bool localIsGpu = (op.localDest.loc == MemoryLocationType::GPU);
      const char* sendPtr = nullptr;
      if (localIsGpu) {
        if (!op.stagingBuffer.data) {
          op.stagingBuffer = bufferPool.Acquire(op.size);
          if (!op.stagingBuffer.data) {
            MORI_IO_ERROR("TcpBackend: staging buffer allocation failed for write_req of size {}",
                          op.size);
            op.status->SetCode(StatusCode::ERR_BAD_STATE);
            op.status->SetMessage("staging buffer allocation failed");
            return;
          }
        }
        const void* devPtr = reinterpret_cast<const void*>(op.localDest.data + op.localOffset);
        hipError_t e = hipMemcpy(op.stagingBuffer.data, devPtr, op.size, hipMemcpyDeviceToHost);
        if (e != hipSuccess) {
          MORI_IO_ERROR("TcpBackend: hipMemcpy D2H failed for write_req of size {}: {}", op.size,
                        hipGetErrorString(e));
          op.status->SetCode(StatusCode::ERR_BAD_STATE);
          op.status->SetMessage(std::string("hipMemcpy D2H failed: ") + hipGetErrorString(e));
          if (op.stagingBuffer.data) bufferPool.Release(std::move(op.stagingBuffer));
          return;
        }
        sendPtr = op.stagingBuffer.data;
      } else
        sendPtr = reinterpret_cast<const char*>(op.localDest.data + op.localOffset);
      TcpMessageHeader hdr{};
      hdr.opcode = 1;
      hdr.id = op.id;
      hdr.mem_id = op.remoteDest.id;  // specify remote memory id explicitly
      hdr.offset = op.remoteOffset;
      hdr.size = op.size;

      if (ep.Send(&hdr, sizeof(hdr)) != 0) {
        MORI_IO_ERROR("TcpBackend: send write_req header failed on fd {}: {}", conn->handle.fd,
                      strerror(errno));
        op.status->SetCode(StatusCode::ERR_BAD_STATE);
        op.status->SetMessage("send write_req header failed");
        return;
      }
      if (ep.Send(sendPtr, op.size) != 0) {
        MORI_IO_ERROR("TcpBackend: send write_req payload failed on fd {}: {}", conn->handle.fd,
                      strerror(errno));
        op.status->SetCode(StatusCode::ERR_BAD_STATE);
        op.status->SetMessage("send write_req payload failed");
      }
    } break;
    case 2:  // READ RESP
    {
    } break;
    case 3:  // WRITE RESP
    {
      TcpMessageHeader hdr{};
      hdr.opcode = 3;  // write_resp
      hdr.id = op.id;
      hdr.mem_id = op.remoteDest.id;
      hdr.offset = op.remoteOffset;
      hdr.size = op.size;

      if (ep.Send(&hdr, sizeof(hdr)) != 0) {
        MORI_IO_ERROR("TcpBackend: send write_resp header failed on fd {}: {}", conn->handle.fd,
                      strerror(errno));
        op.status->SetCode(StatusCode::ERR_BAD_STATE);
        op.status->SetMessage("send write_resp header failed");
        return;
      }

    } break;
  }
}

void TcpBackend::SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) {
    MORI_IO_ERROR("TcpBackend: fcntl F_GETFL failed: {}", strerror(errno));
    return;
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
    MORI_IO_ERROR("TcpBackend: fcntl F_SETFL failed: {}", strerror(errno));
    return;
  }
}

void TcpBackend::SetSocketOptions(int fd) {
  int flag = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
  int bufSz = 4 * 1024 * 1024;
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufSz, sizeof(bufSz));
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufSz, sizeof(bufSz));

  int qack = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &qack, sizeof(qack));
}

void TcpBackend::RearmSocket(int epoll_fd, ConnectionState* conn, uint32_t events) {
  struct epoll_event event;
  event.data.ptr = conn;
  event.events = events | EPOLLET | EPOLLONESHOT;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, conn->handle.fd, &event) == -1) {
    MORI_IO_ERROR("TcpBackend: epoll_ctl MOD fd {} failed: {}", conn->handle.fd, strerror(errno));
    conn->Close();
  }
}

TcpBackendSession* TcpBackend::GetOrCreateSessionCached(const MemoryDesc& local,
                                                        const MemoryDesc& remote) {
  SessionCacheKey key{remote.engineKey, local.id, remote.id};
  {
    std::lock_guard<std::mutex> lock(sessionCacheMu);
    auto it = sessionCache.find(key);
    if (it != sessionCache.end()) {
      return it->second.get();
    }
  }

  BackendSession* rawBase = CreateSession(local, remote);
  if (!rawBase) {
    MORI_IO_ERROR("TcpBackend: CreateSession failed for local mem {} remote mem {}", local.id,
                  remote.id);
    return nullptr;
  }

  std::unique_ptr<TcpBackendSession> newSess(dynamic_cast<TcpBackendSession*>(rawBase));
  if (!newSess) {
    MORI_IO_ERROR(
        "TcpBackend: CreateSession returned incompatible session type (local {}, remote {})",
        local.id, remote.id);
    delete rawBase;
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(sessionCacheMu);
  auto it = sessionCache.find(key);
  if (it != sessionCache.end()) {
    return it->second.get();  // Another thread won the race
  }

  auto [emplacedIt, inserted] = sessionCache.emplace(key, std::move(newSess));
  return emplacedIt->second.get();
}

void TcpBackend::EnsureConnections(const EngineDesc& rdesc, size_t minCount) {
  size_t existing = connPools[rdesc.key]->ConnectionCount();
  if (existing >= minCount) return;

  size_t toCreate = minCount - existing;
  if (toCreate == 0) return;

  // Create connections without holding the lock.
  std::vector<std::shared_ptr<ConnectionState>> pending;
  pending.reserve(toCreate);

  for (size_t i = 0; i < toCreate; ++i) {
    auto handle = application::TCPContext().Connect(rdesc.host, rdesc.port);
    if (handle.fd < 0) {
      MORI_IO_ERROR("TcpBackend: connect to {}:{} failed (attempt {} of {})", rdesc.host,
                    rdesc.port, i + 1, toCreate);
      break;
    }

    SetNonBlocking(handle.fd);
    SetSocketOptions(handle.fd);

    auto conn = std::make_unique<ConnectionState>();
    conn->recvstate = ConnectionState::RecvState::PARSING_HEADER;
    conn->handle = handle;
    conn->listener = nullptr;  // outbound

    struct epoll_event event;
    event.data.ptr = conn.get();
    event.events = EPOLLIN | EPOLLET | EPOLLOUT | EPOLLONESHOT;
    if (epoll_ctl(epollFd, EPOLL_CTL_ADD, handle.fd, &event) == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_ctl ADD new connection fd failed: {}", strerror(errno));
      conn->Close();
      continue;
    }
    pending.push_back(std::move(conn));
  }

  connPools[rdesc.key]->SetConnections(pending);
}

void TcpBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  {
    std::lock_guard<std::mutex> lock(remotesMu);
    auto [it, inserted] = remotes.emplace(rdesc.key, rdesc);
    if (!inserted) return;
  }

  if (config.preconnect) {
    EnsureConnections(rdesc, config.numWorkerThreads);
  }
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  {
    std::lock_guard<std::mutex> lock(remotesMu);
    remotes.erase(rdesc.key);
  }
  auto it = connPools.find(rdesc.key);
  if (it != connPools.end()) {
    it->second->ClearConnections();
    connPools.erase(it);
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

void TcpBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                           const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id, bool isRead) {
  if (size == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  status->SetCode(StatusCode::IN_PROGRESS);
  EngineKey remoteKey = remoteSrc.engineKey;
  EngineDesc rdesc;
  {
    std::lock_guard<std::mutex> lock(remotesMu);
    auto it = remotes.find(remoteKey);
    if (it == remotes.end()) {
      status->SetCode(StatusCode::ERR_NOT_FOUND);
      status->SetMessage("remote engine not registered");
      return;
    }
    rdesc = it->second;
  }

  TransferOp op{localDest, localOffset, remoteSrc, remoteOffset, size, status, id};
  op.opType = isRead ? 0 : 1;

  auto connection = connPools[rdesc.key]->GetNextConnection();
  connection->SubmitTransfer(op);

  // For reference:
  //
  // TcpConnection& conn = *connPtr;
  // // Always lock the connection to fully serialize request-response sequences. This
  // // prevents header/payload interleaving during connection fan-out warm-up or if any
  // // future logic accidentally shares a connection.
  // std::lock_guard<std::mutex> connLock(conn.ioMu);
  // application::TCPEndpoint ep(conn.handle);
  // bool localIsGpu = (localDest.loc == MemoryLocationType::GPU);
  // BufferBlock bufBlock;  // only allocate if GPU staging needed
  // char* stagingPtr = nullptr;
  //
  // TcpMessageHeader hdr{};
  // hdr.opcode = isRead ? 0 : 1;  // read_req or write_req
  // hdr.id = id;
  // hdr.mem_id = remoteSrc.id;  // specify remote memory id explicitly
  // hdr.offset = remoteOffset;
  // hdr.size = size;
  //
  // if (!isRead) {
  //   const char* sendPtr = nullptr;
  //   if (localIsGpu) {
  //     bufBlock = bufferPool.Acquire(size);
  //     stagingPtr = bufBlock.data;
  //     const void* devPtr = reinterpret_cast<const void*>(localDest.data + localOffset);
  //     hipError_t e = hipMemcpy(stagingPtr, devPtr, size, hipMemcpyDeviceToHost);
  //     if (e != hipSuccess) {
  //       status->SetCode(StatusCode::ERR_BAD_STATE);
  //       printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //       status->SetMessage(std::string("hipMemcpy D2H failed: ") + hipGetErrorString(e));
  //       if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //       return;
  //     }
  //     sendPtr = stagingPtr;
  //   } else {
  //     sendPtr = reinterpret_cast<const char*>(localDest.data + localOffset);
  //   }
  //   struct iovec iov[2];
  //   iov[0].iov_base = &hdr;
  //   iov[0].iov_len = sizeof(hdr);
  //   iov[1].iov_base = const_cast<char*>(sendPtr);
  //   iov[1].iov_len = size;
  //   if (FullWritev(conn.handle.fd, iov, 2) != 0) {
  //     int e = errno;
  //     status->SetCode(StatusCode::ERR_BAD_STATE);
  //     printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //     status->SetMessage(std::string("writev failed errno=") + std::to_string(e) + " " +
  //                        std::strerror(e));
  //     if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //     return;
  //   }
  //   TcpMessageHeader resp{};
  //   if (ep.Recv(&resp, sizeof(resp)) != 0) {
  //     status->SetCode(StatusCode::ERR_BAD_STATE);
  //     printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //     status->SetMessage("write ack recv failed");
  //     if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //     return;
  //   }
  //   status->SetCode(StatusCode::SUCCESS);
  //   if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //   return;
  // } else {
  //   if (FullSend(conn.handle.fd, &hdr, sizeof(hdr)) != 0) {
  //     status->SetCode(StatusCode::ERR_BAD_STATE);
  //     printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //     status->SetMessage("read header send failed");
  //     return;
  //   }
  //   TcpMessageHeader resp{};
  //   if (ep.Recv(&resp, sizeof(resp)) != 0) {
  //     status->SetCode(StatusCode::ERR_BAD_STATE);
  //     printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //     status->SetMessage("read resp header recv failed");
  //     return;
  //   }
  //   if (resp.opcode != 2 || resp.size != size) {
  //     status->SetCode(StatusCode::ERR_BAD_STATE);
  //     printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //     status->SetMessage("unexpected read response");
  //     return;
  //   }
  //   if (localIsGpu) {
  //     bufBlock = bufferPool.Acquire(size);
  //     stagingPtr = bufBlock.data;
  //     if (ep.Recv(stagingPtr, size) != 0) {
  //       status->SetCode(StatusCode::ERR_BAD_STATE);
  //       printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //       status->SetMessage("read payload recv failed");
  //       if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //       return;
  //     }
  //     void* devPtr = reinterpret_cast<void*>(localDest.data + localOffset);
  //     hipError_t e = hipMemcpy(devPtr, stagingPtr, size, hipMemcpyHostToDevice);
  //     if (e != hipSuccess) {
  //       status->SetCode(StatusCode::ERR_BAD_STATE);
  //       printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //       status->SetMessage(std::string("hipMemcpy H2D failed: ") + hipGetErrorString(e));
  //       if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //       return;
  //     }
  //     status->SetCode(StatusCode::SUCCESS);
  //     if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
  //     return;
  //   } else {
  //     char* dst = reinterpret_cast<char*>(localDest.data + localOffset);
  //     if (ep.Recv(dst, size) != 0) {
  //       status->SetCode(StatusCode::ERR_BAD_STATE);
  //       printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
  //       status->SetMessage("read payload recv failed");
  //       return;
  //     }
  //     status->SetCode(StatusCode::SUCCESS);
  //     return;
  //   }
  // }
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {}

BackendSession* TcpBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  TcpBackendSession* sess = new TcpBackendSession();
  EngineDesc rdesc = remotes[remote.engineKey];
  EnsureConnections(rdesc, config.numWorkerThreads);
  return sess;
}

bool TcpBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                          TransferStatus* status) {}

void TcpBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {}

void TcpBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {}

}  // namespace io
}  // namespace mori
