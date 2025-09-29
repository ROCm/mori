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
#include "src/io/tcp/executor.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>

namespace mori {
namespace io {

void MultithreadTCPExecutor::DoReadWrite(const ReadWriteWork& work, BufferPool& bufferPool) {
  const MemoryDesc& localDest = work.localDest;
  size_t localOffset = work.localOffset;
  const MemoryDesc& remoteSrc = work.remoteSrc;
  size_t remoteOffset = work.remoteOffset;
  size_t size = work.size;
  TransferStatus* status = work.status;
  TransferUniqueId id = work.id;
  bool isRead = work.isRead;
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
  // Ensure at least one connection exists; if creation fails we error out.
  TcpConnection* connPtr = nullptr;
  // Ensure we have at least one connection per worker thread. This prevents multiple
  // workers from interleaving protocol frames on the same TCP stream.
  auto& vec = EnsureConnections(rdesc, numThreads);
  if (!vec.empty()) {
    // Assign a unique connection index to each thread on first use. We purposely avoid
    // hashing the thread id (which can collide) and instead hand out sequential indices.
    static std::atomic<size_t> globalThreadConnCounter{0};
    static thread_local size_t threadConnIndex = static_cast<size_t>(-1);
    if (threadConnIndex == static_cast<size_t>(-1)) {
      threadConnIndex = globalThreadConnCounter.fetch_add(1, std::memory_order_relaxed);
    }
    if (vec.size() < numThreads) {
      if (threadConnIndex >= vec.size()) {
        size_t idx = 0;
        connPtr = &vec[idx];
      } else {
        size_t idx = threadConnIndex % vec.size();
        connPtr = &vec[idx];
      }
    } else {
      size_t idx = threadConnIndex % vec.size();
      connPtr = &vec[idx];
    }
  }
  if (!connPtr || !connPtr->Valid()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
    status->SetMessage("no valid tcp connection");
    return;
  }
  if (vec.empty()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("no tcp connections established");
    return;
  }
  TcpConnection& conn = *connPtr;
  // Always lock the connection to fully serialize request-response sequences. This
  // prevents header/payload interleaving during connection fan-out warm-up or if any
  // future logic accidentally shares a connection.
  std::lock_guard<std::mutex> connLock(conn.ioMu);
  application::TCPEndpoint ep(conn.handle);
  bool localIsGpu = (localDest.loc == MemoryLocationType::GPU);
  BufferBlock bufBlock;  // only allocate if GPU staging needed
  char* stagingPtr = nullptr;

  TcpMessageHeader hdr{};
  hdr.opcode = isRead ? 0 : 1;  // read_req or write_req
  hdr.id = id;
  hdr.mem_id = remoteSrc.id;  // specify remote memory id explicitly
  hdr.offset = remoteOffset;
  hdr.size = size;

  if (!isRead) {
    const char* sendPtr = nullptr;
    if (localIsGpu) {
      bufBlock = bufferPool.Acquire(size);
      stagingPtr = bufBlock.data;
      const void* devPtr = reinterpret_cast<const void*>(localDest.data + localOffset);
      hipError_t e = hipMemcpy(stagingPtr, devPtr, size, hipMemcpyDeviceToHost);
      if (e != hipSuccess) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
        status->SetMessage(std::string("hipMemcpy D2H failed: ") + hipGetErrorString(e));
        if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
        return;
      }
      sendPtr = stagingPtr;
    } else {
      sendPtr = reinterpret_cast<const char*>(localDest.data + localOffset);
    }
    struct iovec iov[2];
    iov[0].iov_base = &hdr;
    iov[0].iov_len = sizeof(hdr);
    iov[1].iov_base = const_cast<char*>(sendPtr);
    iov[1].iov_len = size;
    if (FullWritev(conn.handle.fd, iov, 2) != 0) {
      int e = errno;
      status->SetCode(StatusCode::ERR_BAD_STATE);
      printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
      status->SetMessage(std::string("writev failed errno=") + std::to_string(e) + " " +
                         std::strerror(e));
      if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
      return;
    }
    TcpMessageHeader resp{};
    if (ep.Recv(&resp, sizeof(resp)) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
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
      printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
      status->SetMessage("read header send failed");
      return;
    }
    TcpMessageHeader resp{};
    if (ep.Recv(&resp, sizeof(resp)) != 0) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
      status->SetMessage("read resp header recv failed");
      return;
    }
    if (resp.opcode != 2 || resp.size != size) {
      status->SetCode(StatusCode::ERR_BAD_STATE);
      printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
      status->SetMessage("unexpected read response");
      return;
    }
    if (localIsGpu) {
      bufBlock = bufferPool.Acquire(size);
      stagingPtr = bufBlock.data;
      if (ep.Recv(stagingPtr, size) != 0) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
        status->SetMessage("read payload recv failed");
        if (bufBlock.data) bufferPool.Release(std::move(bufBlock));
        return;
      }
      void* devPtr = reinterpret_cast<void*>(localDest.data + localOffset);
      hipError_t e = hipMemcpy(devPtr, stagingPtr, size, hipMemcpyHostToDevice);
      if (e != hipSuccess) {
        status->SetCode(StatusCode::ERR_BAD_STATE);
        printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
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
        printf("set code bad state at file %s line %d\n", __FILE__, __LINE__);
        status->SetMessage("read payload recv failed");
        return;
      }
      status->SetCode(StatusCode::SUCCESS);
      return;
    }
  }
}

void MultithreadTCPExecutor::DoServiceWork(int fd, BufferPool& bufferPool) {
  ServiceConnState st;
  // Read remaining header bytes
  if (FullRecv(fd, reinterpret_cast<char*>(&st.hdr), sizeof(TcpMessageHeader)) != 0) {
    st.closed = true;
    MORI_IO_ERROR("tcp service: close fd {} reason=recv_header_fail errno={}", fd, errno);
    if (ctx)
      ctx->CloseFd(fd);
    else
      ::close(fd);
    return;
  }

  // Validate opcode
  if (st.hdr.opcode != 0 && st.hdr.opcode != 1) {
    st.closed = true;
    MORI_IO_ERROR("tcp service: close fd {} reason=bad_opcode={} id={} mem={} size={}", fd,
                  (int)st.hdr.opcode, st.hdr.id, st.hdr.mem_id, st.hdr.size);
    if (ctx)
      ctx->CloseFd(fd);
    else
      ::close(fd);
    return;
  }
  // Fetch memory meta
  MemoryDesc target{};
  {
    std::lock_guard<std::mutex> lock(memMu);
    auto it = localMems.find(st.hdr.mem_id);
    if (it == localMems.end()) {
      st.closed = true;
      MORI_IO_ERROR("tcp service: close fd {} reason=mem_not_found mem_id={} opcode={} size={} ",
                    fd, st.hdr.mem_id, (int)st.hdr.opcode, st.hdr.size);
      if (ctx)
        ctx->CloseFd(fd);
      else
        ::close(fd);
      return;
    }
    target = it->second;
  }
  st.target_is_gpu = (target.loc == MemoryLocationType::GPU);
  if (st.hdr.opcode == 0) {  // READ_REQ: prepare response buffer now
    // Stage data into host memory if needed
    size_t sz = st.hdr.size;
    BufferBlock block = bufferPool.Acquire(sz);
    std::vector<char> payload(sz);
    if (st.target_is_gpu) {
      const void* devPtr = reinterpret_cast<const void*>(target.data + st.hdr.offset);
      if (hipMemcpy(payload.data(), devPtr, sz, hipMemcpyDeviceToHost) != hipSuccess) {
        bufferPool.Release(std::move(block));
        st.closed = true;
        MORI_IO_ERROR("tcp service: close fd {} reason=hipMemcpy_D2H_fail mem={} size={} errno={}",
                      fd, st.hdr.mem_id, sz, errno);
        if (ctx)
          ctx->CloseFd(fd);
        else
          ::close(fd);
        return;
      }
    } else {
      const char* src = reinterpret_cast<const char*>(target.data + st.hdr.offset);
      std::memcpy(payload.data(), src, sz);
    }
    // Build response (header + payload) into out_buf
    TcpMessageHeader resp{};
    resp.opcode = 2;
    resp.id = st.hdr.id;
    resp.mem_id = st.hdr.mem_id;
    resp.offset = st.hdr.offset;
    resp.size = st.hdr.size;
    st.out_buf.resize(sizeof(resp) + payload.size());
    std::memcpy(st.out_buf.data(), &resp, sizeof(resp));
    std::memcpy(st.out_buf.data() + sizeof(resp), payload.data(), payload.size());
    bufferPool.Release(std::move(block));
    st.phase = ConnPhase::SEND_RESP;
  } else {  // WRITE_REQ
    st.in_payload.resize(st.hdr.size);
    st.phase = ConnPhase::RECV_PAYLOAD;
  }

  if (st.phase == ConnPhase::RECV_PAYLOAD) {
    if (FullRecv(fd, st.in_payload.data(), st.in_payload.size()) != 0) {
      st.closed = true;
      MORI_IO_ERROR("tcp service: close fd {} reason=recv_payload_fail mem={} size={} errno={}", fd,
                    st.hdr.mem_id, st.in_payload.size(), errno);
      if (ctx)
        ctx->CloseFd(fd);
      else
        ::close(fd);
      return;
    }

    // Complete write: copy to target memory
    MemoryDesc target{};
    {
      std::lock_guard<std::mutex> lock(memMu);
      auto it = localMems.find(st.hdr.mem_id);
      if (it == localMems.end()) {
        st.closed = true;
        MORI_IO_ERROR("tcp service: close fd {} reason=mem_not_found(write) mem={} size={}", fd,
                      st.hdr.mem_id, st.in_payload.size());
        if (ctx)
          ctx->CloseFd(fd);
        else
          ::close(fd);
        return;
      }
      target = it->second;
    }
    if (st.target_is_gpu) {
      void* devPtr = reinterpret_cast<void*>(target.data + st.hdr.offset);
      if (hipMemcpy(devPtr, st.in_payload.data(), st.in_payload.size(), hipMemcpyHostToDevice) !=
          hipSuccess) {
        st.closed = true;
        MORI_IO_ERROR("tcp service: close fd {} reason=hipMemcpy_H2D_fail mem={} size={} errno={}",
                      fd, st.hdr.mem_id, st.in_payload.size(), errno);
        if (ctx)
          ctx->CloseFd(fd);
        else
          ::close(fd);
        return;
      }
    } else {
      char* dst = reinterpret_cast<char*>(target.data + st.hdr.offset);
      std::memcpy(dst, st.in_payload.data(), st.in_payload.size());
    }
    TcpMessageHeader resp{};
    resp.opcode = 3;
    resp.id = st.hdr.id;
    resp.mem_id = st.hdr.mem_id;
    resp.offset = st.hdr.offset;
    resp.size = st.hdr.size;
    st.out_buf.resize(sizeof(resp));
    std::memcpy(st.out_buf.data(), &resp, sizeof(resp));
    st.phase = ConnPhase::SEND_RESP;
  }
  if (st.phase == ConnPhase::SEND_RESP) {
    if (FullSend(fd, st.out_buf.data(), st.out_buf.size()) != 0) {
      st.closed = true;
      MORI_IO_ERROR("tcp service: close fd {} reason=send_resp_fail opcode={} size={} errno={}", fd,
                    (int)st.hdr.opcode, st.out_buf.size(), errno);
      if (ctx)
        ctx->CloseFd(fd);
      else
        ::close(fd);
      return;
    }
  }
}

std::vector<TcpConnection>& MultithreadTCPExecutor::EnsureConnections(const EngineDesc& rdesc,
                                                                      size_t minCount) {
  std::lock_guard<std::mutex> lk(connsMu);
  auto& vec = conns[rdesc.key];
  // Remove any dead connections first
  vec.erase(
      std::remove_if(vec.begin(), vec.end(), [](const TcpConnection& c) { return !c.Valid(); }),
      vec.end());
  while (vec.size() < minCount) {
    if (!ctx) break;
    auto handle = ctx->Connect(rdesc.host, rdesc.port);
    if (handle.fd < 0) {
      break;  // fail silently; caller will detect invalid
    }
    int flag = 1;
    setsockopt(handle.fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    setsockopt(handle.fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
    int bufSz = 4 * 1024 * 1024;
    setsockopt(handle.fd, SOL_SOCKET, SO_RCVBUF, &bufSz, sizeof(bufSz));
    setsockopt(handle.fd, SOL_SOCKET, SO_SNDBUF, &bufSz, sizeof(bufSz));

    int qack = 1;
    setsockopt(handle.fd, IPPROTO_TCP, TCP_QUICKACK, &qack, sizeof(qack));

    vec.emplace_back(handle);
  }
  return vec;
}

void MultithreadTCPExecutor::CloseConnections(const EngineKey& key) {
  std::lock_guard<std::mutex> lk(connsMu);
  auto it = conns.find(key);
  if (it == conns.end()) return;
  for (auto& c : it->second) {
    if (c.handle.fd >= 0) {
      ctx->CloseEndpoint(c.handle);  // CloseEndpoint performs shutdown/close and removes from map
      c.handle.fd = -1;
    }
  }
  conns.erase(it);
}

}  // namespace io
}  // namespace mori
