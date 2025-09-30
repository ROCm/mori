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

#include <sys/eventfd.h>

#include <array>

namespace mori {
namespace io {

TcpBackend::TcpBackend(EngineKey key, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
    : myEngKey(key), config(cfg), engConfig(engCfg) {
  running.store(true);  // set before threads start
  bufferPool.Configure(128, 16 * 1024 * 1024, true);
  hipStreams.Initialize(config.numWorkerThreads);
  listeners.reserve(config.numWorkerThreads);
  workerThreads.reserve(config.numWorkerThreads);
  workerCtxs.reserve(config.numWorkerThreads);

  for (int i = 0; i < config.numWorkerThreads; ++i) {
    auto* listenctx = new application::TCPContext(engConfig.host, engConfig.port);
    if (!listenctx->Listen()) {
      MORI_IO_ERROR("TcpBackend: Listen failed for worker {}", i);
      delete listenctx;
      continue;
    }
    int lfd = listenctx->GetListenFd();
    if (lfd <= 0) {
      MORI_IO_ERROR("TcpBackend: failed to get listen fd worker {}", i);
      delete listenctx;
      continue;
    }
    SetNonBlocking(lfd);
    listeners.push_back(listenctx);

    // Allocate worker context
    auto* wctx = new WorkerContext();
    wctx->listenCtx = listenctx;
    wctx->id = static_cast<size_t>(i);
    wctx->epollFd = epoll_create1(0);
    if (wctx->epollFd == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_create1 failed for worker {}: {}", i, strerror(errno));
      delete wctx;
      continue;
    }
    wctx->wakeFd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (wctx->wakeFd == -1) {
      MORI_IO_ERROR("TcpBackend: eventfd failed for worker {}: {}", i, strerror(errno));
      close(wctx->epollFd);
      delete wctx;
      continue;
    }

    // Register listen fd
    struct epoll_event lev{};
    lev.data.fd = lfd;
    lev.events = EPOLLIN | EPOLLET;
    if (epoll_ctl(wctx->epollFd, EPOLL_CTL_ADD, lfd, &lev) == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_ctl ADD listen fd failed for worker {}: {}", i,
                    strerror(errno));
      close(wctx->wakeFd);
      close(wctx->epollFd);
      delete wctx;
      continue;
    }
    // Register wake fd
    struct epoll_event wev{};
    wev.data.fd = wctx->wakeFd;
    wev.events = EPOLLIN | EPOLLET;
    if (epoll_ctl(wctx->epollFd, EPOLL_CTL_ADD, wctx->wakeFd, &wev) == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_ctl ADD wake fd failed for worker {}: {}", i,
                    strerror(errno));
      close(wctx->wakeFd);
      close(wctx->epollFd);
      delete wctx;
      continue;
    }

    workerCtxs.push_back(wctx);
    workerThreads.emplace_back([this, wctx] { WorkerLoop(wctx); });
  }
}

TcpBackend::~TcpBackend() {
  running.store(false);
  for (auto& t : workerThreads)
    if (t.joinable()) t.join();
  workerThreads.clear();
  for (auto* w : workerCtxs) {
    if (w->epollFd >= 0) close(w->epollFd);
    if (w->wakeFd >= 0) close(w->wakeFd);
  }
  for (auto* l : listeners) delete l;
  for (auto* w : workerCtxs) delete w;
  workerCtxs.clear();
  listeners.clear();
  for (auto& kv : connPools)
    if (kv.second) kv.second->ClearConnections();
  connPools.clear();
  hipStreams.Destroy();
}

void TcpBackend::WorkerLoop(WorkerContext* wctx) {
  constexpr int kMaxEvents = 64;
  std::array<epoll_event, kMaxEvents> events{};
  int workerEpollFd = wctx->epollFd;
  int listenFd = wctx->listenCtx->GetListenFd();
  int wakeFd = wctx->wakeFd;
  while (running.load()) {
    int n = epoll_wait(workerEpollFd, events.data(), kMaxEvents, 500);
    if (n == -1) {
      if (errno == EINTR) continue;
      MORI_IO_ERROR("TcpBackend: epoll_wait failed: {}", strerror(errno));
      break;
    }
    if (n == 0) continue;
    for (int i = 0; i < n; ++i) {
      auto& ev = events[i];
      if (ev.data.fd == listenFd) {
        HandleNewConnection(wctx->listenCtx, workerEpollFd);
        continue;
      }
      if (ev.data.fd == wakeFd) {
        // Drain eventfd (edge triggered)
        uint64_t tmp;
        while (read(wakeFd, &tmp, sizeof(tmp)) > 0) {
        }
        // Register any pending outbound connections (currently unused placeholder)
        std::vector<std::shared_ptr<ConnectionState>> toAdd;
        {
          std::lock_guard<std::mutex> lk(wctx->pendingAddMu);
          toAdd.swap(wctx->pendingAdd);
        }
        for (auto& c : toAdd) {
          struct epoll_event cev{};
          cev.data.ptr = c.get();
          cev.events = EPOLLIN | EPOLLOUT | EPOLLET | EPOLLONESHOT;
          if (epoll_ctl(workerEpollFd, EPOLL_CTL_ADD, c->handle.fd, &cev) == -1) {
            MORI_IO_ERROR("TcpBackend: epoll_ctl ADD pending outbound fd failed: {}",
                          strerror(errno));
            c->Close();
          }
        }
        continue;
      }
      auto* conn = reinterpret_cast<ConnectionState*>(ev.data.ptr);
      bool closed = false;
      if (ev.events & (EPOLLERR | EPOLLHUP)) {
        MORI_IO_ERROR("TcpBackend: connection err/hup fd {} events=0x{:x}", conn->handle.fd,
                      ev.events);
        conn->Close();
        closed = true;
      } else {
        // Complete non-blocking connect if still in progress.
        if ((ev.events & (EPOLLIN | EPOLLOUT)) &&
            conn->connecting.load(std::memory_order_acquire)) {
          int err = 0;
          socklen_t len = sizeof(err);
          if (getsockopt(conn->handle.fd, SOL_SOCKET, SO_ERROR, &err, &len) == -1 || err != 0) {
            MORI_IO_ERROR("TcpBackend: outbound connect failed fd {} so_error={} errno={}",
                          conn->handle.fd, err, errno);
            conn->Close();
            closed = true;
          } else {
            conn->connecting.store(false, std::memory_order_release);
            conn->ready.store(true, std::memory_order_release);
            // Promote in its owning pool (linear scan across pools acceptable for now)
            for (auto& kv : connPools) {
              kv.second->Promote(conn);
            }
          }
        }
        if (!closed && (ev.events & EPOLLIN)) HandleReadable(conn);
        if (!closed && (ev.events & EPOLLOUT)) HandleWritable(conn);
      }
      if (!closed) RearmSocket(workerEpollFd, conn, EPOLLIN | EPOLLOUT);
    }
  }
}

void TcpBackend::HandleNewConnection(application::TCPContext* listener_ctx, int epoll_fd) {
  auto newEps = listener_ctx->Accept();
  for (const auto& ep : newEps) {
    std::unique_ptr<ConnectionState> conn = std::make_unique<ConnectionState>();
    conn->recvState = ConnectionState::RecvState::PARSING_HEADER;
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

    // Accepted sockets are immediately usable for request handling.
    conn->ready.store(true, std::memory_order_release);

    std::lock_guard<std::mutex> lock(inConnsMu);
    inboundConnections[ep.fd] = std::move(conn);
  }
}

void TcpBackend::HandleReadable(ConnectionState* conn) {
  // This function implements the protocol parsing state machine.
  application::TCPEndpoint ep(conn->handle);
  if (conn->recvState == ConnectionState::RecvState::PARSING_HEADER) {
    if (ep.Recv(&conn->pendingHeader, sizeof(TcpMessageHeader)) != 0) {
      MORI_IO_ERROR("TcpBackend: recv header failed on fd {}: {}", conn->handle.fd,
                    strerror(errno));
      return;
    }
    conn->recvState = ConnectionState::RecvState::PARSING_PAYLOAD;
  }

  if (conn->recvState == ConnectionState::RecvState::PARSING_PAYLOAD) {
    auto& header = conn->pendingHeader;
    size_t sz = header.size;
    switch (header.opcode) {
      case 0:  // READ_REQ
      {
        // Fetch memory meta
        MemoryDesc target{};
        {
          std::lock_guard<std::mutex> lock(memMu);
          auto it = localMems.find(header.mem_id);
          if (it == localMems.end()) {
            MORI_IO_ERROR(
                "tcp service: close fd {} reason=mem_not_found mem_id={} opcode={} size={} ",
                conn->handle.fd, header.mem_id, (int)header.opcode, header.size);
            conn->Close();
            return;
          }
          target = it->second;
        }
        bool targetIsGpu = (target.loc == MemoryLocationType::GPU);

        BufferBlock block = bufferPool.Acquire(sz);
        char* payload = block.data;
        if (targetIsGpu) {
          const void* devPtr = reinterpret_cast<const void*>(target.data + header.offset);
          if (hipMemcpy(payload, devPtr, sz, hipMemcpyDeviceToHost) != hipSuccess) {
            bufferPool.Release(std::move(block));
            MORI_IO_ERROR(
                "tcp service: close fd {} reason=hipMemcpy_D2H_fail mem={} size={} errno={}",
                conn->handle.fd, header.mem_id, sz, errno);
            conn->Close();
            return;
          }
        } else {
          const char* src = reinterpret_cast<const char*>(target.data + header.offset);
          std::memcpy(payload, src, sz);
        }
        // Build response (header + payload) into out_buf
        TcpMessageHeader resp{};
        resp.opcode = 2;
        resp.id = header.id;
        resp.mem_id = header.mem_id;
        resp.offset = header.offset;
        resp.size = header.size;

        if (ep.Send(&resp, sizeof(resp)) != 0) {
          MORI_IO_ERROR("TcpBackend: send write_req header failed on fd {}: {}", conn->handle.fd,
                        strerror(errno));
          conn->Close();
          return;
        }
        if (ep.Send(payload, sz) != 0) {
          MORI_IO_ERROR("TcpBackend: send write_req payload failed on fd {}: {}", conn->handle.fd,
                        strerror(errno));
          conn->Close();
        }
        bufferPool.Release(std::move(block));

      } break;
      case 1:  // WRITE_REQ
      {
        BufferBlock block = bufferPool.Acquire(sz);
        char* payload = block.data;
        if (ep.Recv(payload, sz) != 0) {
          MORI_IO_ERROR("tcp service: close fd {} reason=recv_payload_fail mem={} size={} errno={}",
                        conn->handle.fd, header.mem_id, sz, errno);
          conn->Close();
          return;
        }

        // Complete write: copy to target memory
        MemoryDesc target{};
        {
          std::lock_guard<std::mutex> lock(memMu);
          auto it = localMems.find(header.mem_id);
          if (it == localMems.end()) {
            MORI_IO_ERROR("tcp service: close fd {} reason=mem_not_found(write) mem={} size={}",
                          conn->handle.fd, header.mem_id, sz);
            conn->Close();
            return;
          }
          target = it->second;
        }
        bool targetIsGpu = (target.loc == MemoryLocationType::GPU);
        if (targetIsGpu) {
          void* devPtr = reinterpret_cast<void*>(target.data + header.offset);
          if (hipMemcpy(devPtr, payload, sz, hipMemcpyHostToDevice) != hipSuccess) {
            MORI_IO_ERROR(
                "tcp service: close fd {} reason=hipMemcpy_H2D_fail mem={} size={} errno={}",
                conn->handle.fd, header.mem_id, sz, errno);
            conn->Close();
            return;
          }
        } else {
          char* dst = reinterpret_cast<char*>(target.data + header.offset);
          std::memcpy(dst, payload, sz);
        }
        TcpMessageHeader resp{};
        resp.opcode = 3;
        resp.id = header.id;
        resp.mem_id = header.mem_id;
        resp.offset = header.offset;
        resp.size = header.size;
        ep.Send(&resp, sizeof(resp));
        bufferPool.Release(std::move(block));
      } break;
      case 2:  // READ RESP
      {
        if (conn->recvOp == std::nullopt) {
          MORI_IO_ERROR("TcpBackend: unexpected READ RESP on fd {}", conn->handle.fd);
          conn->Close();
          return;
        }
        TransferOp& op = *(conn->recvOp);
        if (sz != op.size) {
          MORI_IO_ERROR("TcpBackend: READ RESP size mismatch on fd {}: expected {}, got {}",
                        conn->handle.fd, op.size, sz);
          op.status->SetCode(StatusCode::ERR_BAD_STATE);
          op.status->SetMessage("READ RESP size mismatch");
          conn->recvOp.reset();
          conn->recvState = ConnectionState::RecvState::PARSING_HEADER;
          return;
        }
        BufferBlock block = bufferPool.Acquire(sz);
        char* payload = block.data;
        if (ep.Recv(payload, sz) != 0) {
          MORI_IO_ERROR("tcp service: close fd {} reason=recv_payload_fail mem={} size={} errno={}",
                        conn->handle.fd, header.mem_id, sz, errno);
          bufferPool.Release(std::move(block));
          conn->Close();
          return;
        }

        bool localIsGpu = (op.localDest.loc == MemoryLocationType::GPU);
        if (localIsGpu) {
          void* devPtr = reinterpret_cast<void*>(op.localDest.data + op.localOffset);
          if (hipMemcpy(devPtr, payload, sz, hipMemcpyHostToDevice) != hipSuccess) {
            MORI_IO_ERROR("TcpBackend: hipMemcpy H2D failed for read_resp of size {}: {}", sz,
                          hipGetErrorString(hipGetLastError()));
            op.status->SetCode(StatusCode::ERR_BAD_STATE);
            op.status->SetMessage(std::string("hipMemcpy H2D failed: ") +
                                  hipGetErrorString(hipGetLastError()));
            conn->recvOp.reset();
            conn->recvState = ConnectionState::RecvState::PARSING_HEADER;
            bufferPool.Release(std::move(block));
            return;
          }
        } else {
          char* dst = reinterpret_cast<char*>(op.localDest.data + op.localOffset);
          std::memcpy(dst, payload, sz);
        }
        op.status->SetCode(StatusCode::SUCCESS);
        conn->recvOp.reset();
        bufferPool.Release(std::move(block));
      } break;
      case 3:  // WRITE RESP
      {
        TransferOp& op = conn->pendingOps[header.id];
        op.status->SetCode(StatusCode::SUCCESS);
        conn->pendingOps.erase(header.id);

      } break;
      default: {
        MORI_IO_ERROR("TcpBackend: unknown opcode {} fd {} closing", (int)header.opcode,
                      conn->handle.fd);
        conn->Close();
      } break;
    }
  }
  conn->Reset();
}

void TcpBackend::HandleWritable(ConnectionState* conn) {
  auto optOp = conn->PopTransfer();
  if (!optOp) return;  // no pending work
  TransferOp op = std::move(*optOp);
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
      conn->recvOp = std::move(op);  // wait for incoming READ RESP
      conn->recvState = ConnectionState::RecvState::PARSING_HEADER;
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
      // Track pending write to match WRITE RESP
      conn->pendingOps.emplace(op.id, std::move(op));
    } break;
    default: {
      MORI_IO_ERROR("TcpBackend: unknown opType {} on fd {}", (int)op.opType, conn->handle.fd);
      op.status->SetCode(StatusCode::ERR_BAD_STATE);
      op.status->SetMessage("unknown opType");
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
  if (connPools.find(rdesc.key) == connPools.end()) {
    connPools.emplace(rdesc.key, std::make_unique<ConnectionPool>());
  }
  size_t existing = connPools[rdesc.key]->ConnectionCount();
  if (existing >= minCount) return;
  size_t toCreate = minCount - existing;
  std::vector<std::shared_ptr<ConnectionState>> pending;
  pending.reserve(toCreate);
  for (size_t i = 0; i < toCreate; ++i) {
    int sock = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (sock < 0) {
      MORI_IO_ERROR("TcpBackend: socket() failed: {}", strerror(errno));
      break;
    }
    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_port = htons(rdesc.port);
    peer.sin_addr.s_addr = inet_addr(rdesc.host.c_str());
    if (::connect(sock, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)) < 0) {
      if (errno != EINPROGRESS) {
        MORI_IO_ERROR("TcpBackend: connect %s:%u failed: %s", rdesc.host.c_str(), rdesc.port,
                      strerror(errno));
        ::close(sock);
        continue;
      }
    }
    application::TCPEndpointHandle handle{sock, peer};
    SetSocketOptions(handle.fd);
    auto connPtr = std::make_shared<ConnectionState>();
    connPtr->recvState = ConnectionState::RecvState::PARSING_HEADER;
    connPtr->handle = handle;
    connPtr->listener = nullptr;
    connPtr->connecting.store(true, std::memory_order_release);
    // Assign to a worker in round-robin fashion; queue for registration.
    if (workerCtxs.empty()) {
      MORI_IO_ERROR("TcpBackend: no worker contexts available for outbound connection");
      connPtr->Close();
      continue;
    }
    size_t idx = nextWorker.fetch_add(1, std::memory_order_relaxed) % workerCtxs.size();
    auto* wctx = workerCtxs[idx];
    {
      std::lock_guard<std::mutex> lk(wctx->pendingAddMu);
      wctx->pendingAdd.push_back(connPtr);
    }
    uint64_t one = 1;
    if (write(wctx->wakeFd, &one, sizeof(one)) < 0) {
      if (errno != EAGAIN) {
        MORI_IO_ERROR("TcpBackend: write wakeFd failed: {}", strerror(errno));
      }
    }
    pending.push_back(connPtr);  // still track in pool
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
  if (!connection) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("no valid tcp connection");
    return;
  }
  connection->SubmitTransfer(std::move(op));
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {}

BackendSession* TcpBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  return new TcpBackendSession(this, local, remote);
}

bool TcpBackend::PopInboundTransferStatus(EngineKey /*remote*/, TransferUniqueId /*id*/,
                                          TransferStatus* /*status*/) {
  return false;  // Not implemented yet
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
