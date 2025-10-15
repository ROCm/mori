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

#include <errno.h>
#include <sys/eventfd.h>

#include <array>

namespace mori {
namespace io {

namespace {
// Helper: mark one sub-operation success within a batch.
inline void TcpBatchOpSuccess(TransferOp& op) {
  if (op.batchCtx) {
    // Only decrement when status was IN_PROGRESS.
    size_t prev = op.batchCtx->remaining.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
      // Last one to finish and no failure flagged.
      if (!op.batchCtx->failed.load(std::memory_order_acquire)) {
        op.batchCtx->userStatus->SetCode(StatusCode::SUCCESS);
      } else {
        op.batchCtx->userStatus->SetCode(StatusCode::ERR_BAD_STATE);
        std::lock_guard<std::mutex> lk(op.batchCtx->msgMu);
        if (!op.batchCtx->failMsg.empty())
          op.batchCtx->userStatus->SetMessage(op.batchCtx->failMsg);
      }
      delete op.batchCtx;  // free context
      op.batchCtx = nullptr;
    }
  } else if (op.status) {
    op.status->SetCode(StatusCode::SUCCESS);
  }
}

// Helper: mark one sub-operation failure within a batch.
inline void TcpBatchOpFail(TransferOp& op, StatusCode code, const std::string& msg) {
  if (op.batchCtx) {
    bool expected = false;
    if (op.batchCtx->failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      // first failure captures message
      std::lock_guard<std::mutex> lk(op.batchCtx->msgMu);
      op.batchCtx->failMsg = msg;
    }
    size_t prev = op.batchCtx->remaining.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
      op.batchCtx->userStatus->SetCode(code);
      op.batchCtx->userStatus->SetMessage(msg);
      delete op.batchCtx;
      op.batchCtx = nullptr;
    }
  } else if (op.status) {
    op.status->SetCode(code);
    op.status->SetMessage(msg);
  }
}
}  // namespace

void BackendServer::Start() {
  running.store(true);  // set before threads start
  bufferPool.Configure(128, 16 * 1024 * 1024, true);
  listeners.reserve(config.numWorkerThreads);
  workerThreads.reserve(config.numWorkerThreads);
  workerCtxs.reserve(config.numWorkerThreads);

  for (int i = 0; i < config.numWorkerThreads; ++i) {
    auto* listenctx = new application::TCPContext(engConfig.host, engConfig.port);
    listenctx->Listen();
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

    // Register listen fd
    struct epoll_event lev{};
    lev.data.fd = lfd;
    lev.events = EPOLLIN | EPOLLET;
    if (epoll_ctl(wctx->epollFd, EPOLL_CTL_ADD, lfd, &lev) == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_ctl ADD listen fd failed for worker {}: {}", i,
                    strerror(errno));
      close(wctx->epollFd);
      delete wctx;
      continue;
    }

    workerCtxs.push_back(wctx);
    workerThreads.emplace_back([this, wctx] { WorkerLoop(wctx); });
  }
}

void BackendServer::Stop() {
  running.store(false);
  for (auto& t : workerThreads)
    if (t.joinable()) t.join();
  workerThreads.clear();
  for (auto* w : workerCtxs) {
    if (w->epollFd >= 0) close(w->epollFd);
  }
  for (auto* l : listeners) delete l;
  for (auto* w : workerCtxs) delete w;
  workerCtxs.clear();
  listeners.clear();
  for (auto& kv : connPools)
    if (kv.second) kv.second->Shutdown();

  connPools.clear();
}

void BackendServer::CloseInbound(ConnectionState* conn) {
  std::unique_ptr<ConnectionState> victim;
  {
    std::lock_guard<std::mutex> lk(inConnsMu);
    auto it = inboundConnections.find(conn->handle.fd);
    if (it != inboundConnections.end()) {
      victim = std::move(it->second);
      inboundConnections.erase(it);
    }
  }
  if (victim) victim->Close();
}

void BackendServer::WorkerLoop(WorkerContext* wctx) {
  constexpr int kMaxEvents = 64;
  std::array<epoll_event, kMaxEvents> events{};
  int workerEpollFd = wctx->epollFd;
  int listenFd = wctx->listenCtx->GetListenFd();
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
      auto* conn = reinterpret_cast<ConnectionState*>(ev.data.ptr);
      if (conn) conn->lastEpollFd = workerEpollFd;
      bool closed = false;
      if (ev.events & (EPOLLERR | EPOLLHUP)) {
        MORI_IO_ERROR("TcpBackend: connection err/hup fd {} events=0x{:x}", conn->handle.fd,
                      ev.events);
        conn->Close();
        closed = true;
      } else {
        if (!closed && (ev.events & EPOLLIN)) HandleReadable(conn);
        if (!closed && (ev.events & EPOLLOUT)) HandleWritable(conn);
      }
      if (conn->handle.fd < 0) {
        closed = true;
      }
      if (!closed) RearmSocket(workerEpollFd, conn, EPOLLIN | EPOLLOUT);
    }
  }
}

void BackendServer::HandleNewConnection(application::TCPContext* listener_ctx, int epoll_fd) {
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

void BackendServer::HandleReadable(ConnectionState* conn) {
  application::TCPEndpoint ep(conn->handle);
  constexpr size_t kHeaderSize = sizeof(TcpMessageHeader);
  bool needRearm = false;
  while (true) {
    // Phase 1: read header incrementally
    if (conn->recvState == ConnectionState::RecvState::PARSING_HEADER) {
      int r =
          ep.RecvSomeExact(reinterpret_cast<char*>(&conn->pendingHeader) + conn->headerBytesRead,
                           kHeaderSize - conn->headerBytesRead);
      if (r > 0) {
        conn->headerBytesRead += static_cast<size_t>(r);
        if (conn->headerBytesRead < kHeaderSize) {
          // Need more data later
          needRearm = true;
          break;
        }
        // Full header acquired
        auto& header = conn->pendingHeader;
        if (header.opcode == 0 && header.id == 0 && header.mem_id == 0 && header.size == 0) {
          // Treat as peer close / invalid; close
          CloseInbound(conn);
          return;
        }
        conn->expectedPayloadSize = header.size;
        conn->payloadBytesRead = 0;
        // For WRITE_REQ and READ_RESP we expect a payload from peer.
        OpType opt = static_cast<OpType>(header.opcode);
        if (opt == WRITE_REQ || opt == READ_RESP) {
          if (conn->expectedPayloadSize > 0) {
            conn->inboundPayload = bufferPool.Acquire(conn->expectedPayloadSize);
            if (!conn->inboundPayload.data) {
              MORI_IO_ERROR("TcpBackend: buffer allocation failed size {} fd {}",
                            conn->expectedPayloadSize, conn->handle.fd);
              CloseInbound(conn);
              return;
            }
          }
        }
        conn->recvState = ConnectionState::RecvState::PARSING_PAYLOAD;
        // Fall through to payload loop same iteration
      } else if (r == 0) {
        // Peer closed during header
        CloseInbound(conn);
        return;
      } else if (r == -EAGAIN) {
        needRearm = true;
        break;
      } else {  // real error
        MORI_IO_ERROR("TcpBackend: header recv error fd {} err={} errno={} ", conn->handle.fd, r,
                      errno);
        CloseInbound(conn);
        return;
      }
    }

    // Phase 2: payload (if any)
    if (conn->recvState == ConnectionState::RecvState::PARSING_PAYLOAD) {
      auto& header = conn->pendingHeader;
      OpType opt = static_cast<OpType>(header.opcode);
      size_t need = conn->expectedPayloadSize - conn->payloadBytesRead;
      if (opt == WRITE_REQ || opt == READ_RESP) {
        if (need > 0) {
          int r = ep.RecvSomeExact(conn->inboundPayload.data + conn->payloadBytesRead, need);
          if (r > 0) {
            conn->payloadBytesRead += static_cast<size_t>(r);
            need = conn->expectedPayloadSize - conn->payloadBytesRead;
            if (need > 0) {
              needRearm = true;  // need more data later
              break;             // exit loop now
            }
          } else if (r == 0) {
            // Peer closed mid-payload
            MORI_IO_ERROR("TcpBackend: peer closed mid-payload fd {}", conn->handle.fd);
            CloseInbound(conn);
            return;
          } else if (r == -EAGAIN) {
            needRearm = true;
            break;
          } else {  // real error
            MORI_IO_ERROR("TcpBackend: payload recv error fd {} err={} errno={} ", conn->handle.fd,
                          r, errno);
            CloseInbound(conn);
            return;
          }
        }
      }

      // If we reach here, payload (if any) is complete. Execute operation.
      switch (opt) {
        case READ_REQ: {
          // Perform memory read and send back (READ_RESP opcode =2)
          MemoryDesc target{};
          {
            std::lock_guard<std::mutex> lock(memMu);
            auto it = localMems.find(header.mem_id);
            if (it == localMems.end()) {
              MORI_IO_ERROR(
                  "tcp service: close fd {} reason=mem_not_found mem_id={} opcode={} size={}",
                  conn->handle.fd, header.mem_id, (int)header.opcode, header.size);
              CloseInbound(conn);
              return;
            }
            target = it->second;
          }
          bool targetIsGpu = (target.loc == MemoryLocationType::GPU);
          BufferBlock block = bufferPool.Acquire(header.size);
          if (header.size && !block.data) {
            MORI_IO_ERROR("TcpBackend: buffer alloc fail for read_resp size {} fd {}", header.size,
                          conn->handle.fd);
            CloseInbound(conn);
            return;
          }
          char* payload = block.data;
          if (header.size) {
            if (targetIsGpu) {
              const void* devPtr = reinterpret_cast<const void*>(target.data + header.offset);
              if (hipMemcpy(payload, devPtr, header.size, hipMemcpyDeviceToHost) != hipSuccess) {
                bufferPool.Release(std::move(block));
                MORI_IO_ERROR(
                    "tcp service: close fd {} reason=hipMemcpy_D2H_fail mem={} size={} errno={}",
                    conn->handle.fd, header.mem_id, header.size, errno);
                CloseInbound(conn);
                return;
              }
            } else {
              const char* src = reinterpret_cast<const char*>(target.data + header.offset);
              std::memcpy(payload, src, header.size);
            }
          }
          TcpMessageHeader resp{};
          resp.opcode = 2;  // READ_RESP
          resp.id = header.id;
          resp.mem_id = header.mem_id;
          resp.offset = header.offset;
          resp.size = header.size;
          if (ep.Send(&resp, sizeof(resp)) != 0 ||
              (header.size && ep.Send(payload, header.size) != 0)) {
            MORI_IO_ERROR("TcpBackend: send read_resp failed fd {}", conn->handle.fd);
            bufferPool.Release(std::move(block));
            CloseInbound(conn);
            return;
          }
          bufferPool.Release(std::move(block));
        } break;
        case WRITE_REQ: {
          // inboundPayload holds data
          MemoryDesc target{};
          {
            std::lock_guard<std::mutex> lock(memMu);
            auto it = localMems.find(header.mem_id);
            if (it == localMems.end()) {
              MORI_IO_ERROR("tcp service: close fd {} reason=mem_not_found(write) mem={} size={}",
                            conn->handle.fd, header.mem_id, header.size);
              if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
              CloseInbound(conn);
              return;
            }
            target = it->second;
          }
          bool targetIsGpu = (target.loc == MemoryLocationType::GPU);
          if (header.size) {
            if (targetIsGpu) {
              void* devPtr = reinterpret_cast<void*>(target.data + header.offset);
              if (hipMemcpy(devPtr, conn->inboundPayload.data, header.size,
                            hipMemcpyHostToDevice) != hipSuccess) {
                MORI_IO_ERROR(
                    "tcp service: close fd {} reason=hipMemcpy_H2D_fail mem={} size={} errno={}",
                    conn->handle.fd, header.mem_id, header.size, errno);
                if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
                CloseInbound(conn);
                return;
              }
            } else {
              char* dst = reinterpret_cast<char*>(target.data + header.offset);
              std::memcpy(dst, conn->inboundPayload.data, header.size);
            }
          }
          TcpMessageHeader resp{};
          resp.opcode = 3;  // WRITE_RESP
          resp.id = header.id;
          resp.mem_id = header.mem_id;
          resp.offset = header.offset;
          resp.size = header.size;
          ep.Send(&resp, sizeof(resp));  // best effort
          if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
        } break;
        case READ_RESP: {
          if (conn->recvOp == std::nullopt) {
            MORI_IO_ERROR("TcpBackend: unexpected READ RESP fd {}", conn->handle.fd);
            if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
            CloseInbound(conn);
            return;
          }
          TransferOp& op = *(conn->recvOp);
          if (header.size != op.size) {
            MORI_IO_ERROR("TcpBackend: READ RESP size mismatch fd {} expected {} got {}",
                          conn->handle.fd, op.size, header.size);
            TcpBatchOpFail(op, StatusCode::ERR_BAD_STATE, "READ RESP size mismatch");
            if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
            conn->recvOp.reset();
          } else if (header.size) {
            bool localIsGpu = (op.localDest.loc == MemoryLocationType::GPU);
            if (localIsGpu) {
              void* devPtr = reinterpret_cast<void*>(op.localDest.data + op.localOffset);
              if (hipMemcpy(devPtr, conn->inboundPayload.data, header.size,
                            hipMemcpyHostToDevice) != hipSuccess) {
                MORI_IO_ERROR("TcpBackend: hipMemcpy H2D failed read_resp size {} fd {}",
                              header.size, conn->handle.fd);
                TcpBatchOpFail(op, StatusCode::ERR_BAD_STATE, "hipMemcpy H2D failed");
                if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
                conn->recvOp.reset();
                CloseInbound(conn);
                return;
              }
            } else {
              char* dst = reinterpret_cast<char*>(op.localDest.data + op.localOffset);
              std::memcpy(dst, conn->inboundPayload.data, header.size);
            }
            TcpBatchOpSuccess(op);
            if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
            connPools[op.remoteDest.engineKey]->ReleaseConnection(conn);
            conn->recvOp.reset();
          } else {  // zero-size success
            TcpBatchOpSuccess(op);
            connPools[op.remoteDest.engineKey]->ReleaseConnection(conn);
            conn->recvOp.reset();
            if (conn->inboundPayload.data) bufferPool.Release(std::move(conn->inboundPayload));
          }
        } break;
        case WRITE_RESP: {
          std::lock_guard<std::mutex> lk(conn->mu);
          auto it = conn->pendingOps.find(header.id);
          if (it == conn->pendingOps.end()) {
            MORI_IO_ERROR("WRITE_RESP unknown id {} fd {}", header.id, conn->handle.fd);
          } else {
            TransferOp& op = it->second;
            TcpBatchOpSuccess(op);
            conn->pendingOps.erase(it);
            connPools[op.remoteDest.engineKey]->ReleaseConnection(conn);
          }
        } break;
        default: {
          MORI_IO_ERROR("TcpBackend: unknown opcode {} fd {} closing", (int)header.opcode,
                        conn->handle.fd);
          CloseInbound(conn);
          return;
        }
      }
      // Reset for next message
      conn->pendingHeader = TcpMessageHeader{};
      conn->headerBytesRead = 0;
      conn->payloadBytesRead = 0;
      conn->expectedPayloadSize = 0;
      conn->recvState = ConnectionState::RecvState::PARSING_HEADER;
      // Continue loop to drain further messages in same epoll tick
      continue;
    }
  }
  if (needRearm && conn->handle.fd >= 0 && conn->lastEpollFd >= 0) {
    RearmSocket(conn->lastEpollFd, conn, EPOLLIN | EPOLLOUT);
  }
}

void BackendServer::HandleWritable(ConnectionState* conn) {
  application::TCPEndpoint ep(conn->handle);
  bool needRearm = false;

  // Try to start a new op if none active
  if (!conn->activeSendOp) {
    auto optOp = conn->PopTransfer();
    if (!optOp) return;  // nothing to do
    conn->activeSendOp = std::move(*optOp);
    TransferOp& op = *conn->activeSendOp;
    // Prepare header
    TcpMessageHeader hdr{};
    hdr.opcode = (op.opType == READ_REQ) ? 0 : 1;  // READ_REQ=0, WRITE_REQ=1
    hdr.id = op.id;
    hdr.mem_id = op.remoteDest.id;
    hdr.offset = (op.opType == READ_REQ) ? op.remoteOffset : op.remoteOffset;  // same field usage
    hdr.size = op.size;
    std::memcpy(conn->outgoingHeader, &hdr, sizeof(hdr));
    conn->headerBytesSent = 0;
    conn->payloadBytesSent = 0;
    conn->payloadBytesTotal = 0;
    conn->payloadPtr = nullptr;

    if (op.opType == READ_REQ) {
      // No payload to send; after header goes out we wait for READ_RESP
    } else if (op.opType == WRITE_REQ) {
      bool localIsGpu = (op.localDest.loc == MemoryLocationType::GPU);
      if (localIsGpu) {
        if (!op.stagingBuffer.data) {
          op.stagingBuffer = bufferPool.Acquire(op.size);
          if (!op.stagingBuffer.data) {
            MORI_IO_ERROR("TcpBackend: staging buffer allocation failed for write_req size {}",
                          op.size);
            TcpBatchOpFail(op, StatusCode::ERR_BAD_STATE, "staging buffer allocation failed");
            conn->activeSendOp.reset();
            return;
          }
        }
        const void* devPtr = reinterpret_cast<const void*>(op.localDest.data + op.localOffset);
        hipError_t e = hipMemcpy(op.stagingBuffer.data, devPtr, op.size, hipMemcpyDeviceToHost);
        if (e != hipSuccess) {
          MORI_IO_ERROR("TcpBackend: hipMemcpy D2H failed for write_req size {}: {}", op.size,
                        hipGetErrorString(e));
          TcpBatchOpFail(op, StatusCode::ERR_BAD_STATE,
                         std::string("hipMemcpy D2H failed: ") + hipGetErrorString(e));
          if (op.stagingBuffer.data) bufferPool.Release(std::move(op.stagingBuffer));
          conn->activeSendOp.reset();
          return;
        }
        conn->payloadPtr = op.stagingBuffer.data;
      } else {
        conn->payloadPtr = reinterpret_cast<const char*>(op.localDest.data + op.localOffset);
      }
      conn->payloadBytesTotal = op.size;
    }
  }

  // Now attempt to progress active operation
  if (!conn->activeSendOp) return;  // nothing active
  TransferOp& aop = *conn->activeSendOp;

  // 1. Send header (partial allowed)
  while (conn->headerBytesSent < sizeof(TcpMessageHeader)) {
    size_t remaining = sizeof(TcpMessageHeader) - conn->headerBytesSent;
    ssize_t n = ::send(conn->handle.fd, conn->outgoingHeader + conn->headerBytesSent, remaining, 0);
    if (n > 0) {
      conn->headerBytesSent += static_cast<size_t>(n);
      continue;  // try to finish header
    }
    if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      needRearm = true;
      break;
    }
    // real error or connection closed
    MORI_IO_ERROR("TcpBackend: send header failed fd {} errno={}", conn->handle.fd, errno);
    TcpBatchOpFail(aop, StatusCode::ERR_BAD_STATE, "send header failed");
    conn->activeSendOp.reset();
    return;
  }

  if (conn->headerBytesSent < sizeof(TcpMessageHeader)) {
    // Need rearm for EPOLLOUT
    if (needRearm && conn->lastEpollFd >= 0)
      RearmSocket(conn->lastEpollFd, conn, EPOLLIN | EPOLLOUT);
    return;
  }

  // 2. Send payload if WRITE_REQ
  if (aop.opType == WRITE_REQ && conn->payloadBytesSent < conn->payloadBytesTotal) {
    while (conn->payloadBytesSent < conn->payloadBytesTotal) {
      size_t remaining = conn->payloadBytesTotal - conn->payloadBytesSent;
      ssize_t n = ::send(conn->handle.fd, conn->payloadPtr + conn->payloadBytesSent, remaining, 0);
      if (n > 0) {
        conn->payloadBytesSent += static_cast<size_t>(n);
        continue;
      }
      if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        needRearm = true;
        break;
      }
      MORI_IO_ERROR("TcpBackend: send payload failed fd {} errno={} sent={} remaining={} size={} ",
                    conn->handle.fd, errno, conn->payloadBytesSent, remaining,
                    conn->payloadBytesTotal);
      TcpBatchOpFail(aop, StatusCode::ERR_BAD_STATE, "send payload failed");
      conn->activeSendOp.reset();
      return;
    }
    if (conn->payloadBytesSent < conn->payloadBytesTotal) {
      if (needRearm && conn->lastEpollFd >= 0)
        RearmSocket(conn->lastEpollFd, conn, EPOLLIN | EPOLLOUT);
      return;  // wait for next EPOLLOUT
    }
  }

  // 3. Operation header (+payload if any) fully sent.
  if (aop.opType == READ_REQ) {
    // Expect READ_RESP later -> keep status IN_PROGRESS
    conn->recvOp = aop;  // store to match incoming READ_RESP
  } else if (aop.opType == WRITE_REQ) {
    // Track pending write for WRITE_RESP
    {
      std::lock_guard<std::mutex> lk(conn->mu);
      conn->pendingOps.emplace(aop.id, aop);
    }
  }
  // Clear active (but keep copies in recvOp or pendingOps)
  conn->activeSendOp.reset();

  // If more queued ops, attempt next within same EPOLLOUT tick (tail recursion style)
  if (conn->sendQueue.size() > 0) {
    HandleWritable(conn);  // re-enter to try next (bounded by queued ops; recursion depth small)
    return;
  }
}

void BackendServer::SetNonBlocking(int fd) {
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

void BackendServer::SetSocketOptions(int fd) {
  int flag = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
  int bufSz = 4 * 1024 * 1024;
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufSz, sizeof(bufSz));
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufSz, sizeof(bufSz));

  int qack = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &qack, sizeof(qack));
}

void BackendServer::RearmSocket(int epoll_fd, ConnectionState* conn, uint32_t events) {
  if (conn->handle.fd < 0) return;
  struct epoll_event event;
  event.data.ptr = conn;
  event.events = events | EPOLLET | EPOLLONESHOT;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, conn->handle.fd, &event) == -1) {
    MORI_IO_ERROR("TcpBackend: epoll_ctl MOD fd {} failed: {}", conn->handle.fd, strerror(errno));
    conn->Close();
  }
}

void BackendServer::EnsureConnections(const EngineDesc& rdesc, size_t minCount) {
  if (connPools.find(rdesc.key) == connPools.end()) {
    connPools.emplace(rdesc.key, std::make_unique<ConnectionPool>());
  }
  size_t existing = connPools[rdesc.key]->ConnectionCount();
  if (existing >= minCount) return;
  size_t toCreate = minCount - existing;
  std::vector<ConnectionState*> pending;
  pending.reserve(toCreate);
  for (size_t i = 0; i < toCreate; ++i) {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
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
    SetNonBlocking(handle.fd);
    ConnectionState* connPtr = new ConnectionState();
    connPtr->recvState = ConnectionState::RecvState::PARSING_HEADER;
    connPtr->handle = handle;
    connPtr->listener = nullptr;
    connPtr->ready.store(true, std::memory_order_release);
    // Assign to a worker in round-robin fashion; queue for registration.
    if (workerCtxs.empty()) {
      MORI_IO_ERROR("TcpBackend: no worker contexts available for outbound connection");
      connPtr->Close();
      delete connPtr;
      continue;
    }
    size_t idx = nextWorker.fetch_add(1, std::memory_order_relaxed) % workerCtxs.size();
    auto* wctx = workerCtxs[idx];
    struct epoll_event cev{};
    cev.data.ptr = connPtr;
    cev.events = EPOLLIN | EPOLLOUT | EPOLLET | EPOLLONESHOT;
    if (epoll_ctl(wctx->epollFd, EPOLL_CTL_ADD, handle.fd, &cev) == -1) {
      MORI_IO_ERROR("TcpBackend: epoll_ctl ADD pending outbound fd failed: {}", strerror(errno));
      connPtr->Close();
      delete connPtr;
      continue;
    }
    pending.push_back(connPtr);  // still track in pool
  }
  connPools[rdesc.key]->SetConnections(pending);
}

void BackendServer::RegisterRemoteEngine(const EngineDesc& rdesc) {
  {
    std::lock_guard<std::mutex> lock(remotesMu);
    auto [it, inserted] = remotes.emplace(rdesc.key, rdesc);
    if (!inserted) return;
  }

  if (config.preconnect) {
    EnsureConnections(rdesc, config.numWorkerThreads);
  }
}

void BackendServer::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  {
    std::lock_guard<std::mutex> lock(remotesMu);
    remotes.erase(rdesc.key);
  }
  auto it = connPools.find(rdesc.key);
  if (it != connPools.end()) {
    it->second->Shutdown();
    connPools.erase(it);
  }
}

void BackendServer::RegisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems[desc.id] = desc;
}

void BackendServer::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems.erase(desc.id);
}

void BackendServer::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                   const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                   const SizeVec& sizes, TransferStatus* status,
                                   TransferUniqueId id, bool isRead) {
  // Basic validation
  if (sizes.empty()) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  if (localOffsets.size() != sizes.size() || remoteOffsets.size() != sizes.size()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("BatchReadWrite: vector size mismatch");
    return;
  }

  // Lookup remote engine descriptor
  EngineDesc rdesc;
  {
    std::lock_guard<std::mutex> lock(remotesMu);
    auto it = remotes.find(remoteSrc.engineKey);
    if (it == remotes.end()) {
      status->SetCode(StatusCode::ERR_NOT_FOUND);
      status->SetMessage("remote engine not registered");
      return;
    }
    rdesc = it->second;
  }

  // Snapshot connections (both ready & pending). We'll attempt to push ops into their queues;
  // even pending ones will pick them up after promotion.
  auto poolIt = connPools.find(rdesc.key);
  if (poolIt == connPools.end()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("no connection pool for remote engine");
    return;
  }
  ConnectionPool* pool = poolIt->second.get();
  std::deque<ConnectionState*> conns = pool->GetAllConnections();
  if (conns.empty()) {
    // Try to create at least one connection lazily
    EnsureConnections(rdesc, config.numWorkerThreads);
    conns = pool->GetAllConnections();
  }
  if (conns.empty()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("no tcp connections available");
    return;
  }

  // Create batch context and mark user status in-progress.
  status->SetCode(StatusCode::IN_PROGRESS);
  TcpBatchContext* batchCtx = new TcpBatchContext(status, sizes.size());

  // Round-robin distribute operations across all connections without blocking.
  size_t connIdx = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    size_t sz = sizes[i];
    if (sz == 0) {
      // Zero-size op counts as immediate success.
      TransferOp tmp{localDest,
                     localOffsets[i],
                     remoteSrc,
                     remoteOffsets[i],
                     0,
                     nullptr,
                     NextUniqueTransferId(),
                     isRead ? READ_REQ : WRITE_REQ};
      tmp.batchCtx = batchCtx;
      TcpBatchOpSuccess(tmp);  // decrements remaining; may delete batchCtx
      continue;
    }
    TransferStatus* dummy = nullptr;  // per-op status not used in batch
    TransferOp op{localDest,
                  localOffsets[i],
                  remoteSrc,
                  remoteOffsets[i],
                  sz,
                  dummy,
                  NextUniqueTransferId(),
                  isRead ? READ_REQ : WRITE_REQ};
    op.batchCtx = batchCtx;
    ConnectionState* c = conns[connIdx % conns.size()];
    connIdx++;
    c->SubmitTransfer(std::move(op));
  }
}

TcpBackend::TcpBackend(EngineKey k, const IOEngineConfig& engineCfg,
                       const TcpBackendConfig& tcpCfg) {
  myEngKey = k;
  server = new BackendServer(engineCfg, tcpCfg);
  server->Start();
}

TcpBackend::~TcpBackend() {
  server->Stop();
  delete server;
}

void TcpBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  server->RegisterRemoteEngine(rdesc);
}

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  server->DeregisterRemoteEngine(rdesc);
}

void TcpBackend::RegisterMemory(const MemoryDesc& desc) { server->RegisterMemory(desc); }

void TcpBackend::DeregisterMemory(const MemoryDesc& desc) { server->DeregisterMemory(desc); }

void TcpBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                           const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                           TransferStatus* status, TransferUniqueId id, bool isRead) {
  SizeVec localOffsets{localOffset};
  SizeVec remoteOffsets{remoteOffset};
  SizeVec sizes{size};
  server->BatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id,
                         isRead);
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                bool isRead) {
  server->BatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id,
                         isRead);
}

bool TcpBackend::PopInboundTransferStatus(EngineKey /*remote*/, TransferUniqueId /*id*/,
                                          TransferStatus* /*status*/) {
  return false;  // Not implemented yet
}

BackendSession* TcpBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  return new TcpBackendSession(server, local, remote);
}

void TcpBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                  TransferStatus* status, TransferUniqueId id, bool isRead) {
  SizeVec localOffsets{localOffset};
  SizeVec remoteOffsets{remoteOffset};
  SizeVec sizes{size};
  backend->BatchReadWrite(local, localOffsets, remote, remoteOffsets, sizes, status, id, isRead);
}

void TcpBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status,
                                       TransferUniqueId id, bool isRead) {
  backend->BatchReadWrite(local, localOffsets, remote, remoteOffsets, sizes, status, id, isRead);
}

}  // namespace io
}  // namespace mori
