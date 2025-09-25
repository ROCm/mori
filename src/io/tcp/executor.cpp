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

namespace mori {
namespace io {

void MultithreadExecutor::DoReadWrite(const ReadWriteWork& work, BufferPool& bufferPool) {
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
    if (remotes.find(remoteKey) == remotes.end()) {
      status->SetCode(StatusCode::ERR_NOT_FOUND);
      status->SetMessage("remote engine not registered");
      return;
    }
    rdesc = remotes[remoteKey];
  }
  TcpConnection* connPtr = nullptr;
  {
    std::lock_guard<std::mutex> lk(connsMu);
    auto it = conns.find(remoteKey);
    if (it != conns.end() && !it->second.empty()) {
      size_t idx = std::hash<std::thread::id>{}(std::this_thread::get_id()) % it->second.size();
      connPtr = &it->second[idx];
    }
  }
  if (!connPtr || !connPtr->Valid()) {
    status->SetCode(StatusCode::ERR_BAD_STATE);
    status->SetMessage("no valid tcp connection");
    return;
  }
  TcpConnection& conn = *connPtr;
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

void MultithreadExecutor::DoServiceWork(int fd, BufferPool& bufferPool) {
  // handle request
  TcpMessageHeader hdr{};
  ssize_t r = ::recv(fd, &hdr, sizeof(hdr), MSG_WAITALL);
  if (r != sizeof(hdr)) {
    ::close(fd);
    return;
  }
  if (hdr.opcode == 0 || hdr.opcode == 1) {  // read or write
    MemoryDesc target{};
    {
      std::lock_guard<std::mutex> lock(memMu);
      auto it = localMems.find(hdr.mem_id);
      if (it == localMems.end()) {
        ::close(fd);
        return;
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
          ::close(fd);
          return;
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
        ::close(fd);
        return;
      }
      if (targetIsGpu) {
        void* devPtr = reinterpret_cast<void*>(target.data + hdr.offset);
        if (hipMemcpy(devPtr, hostBuf, hdr.size, hipMemcpyHostToDevice) != hipSuccess) {
          bufferPool.Release(std::move(bufBlock));
          ::close(fd);
          return;
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
  } else {
    ::close(fd);
  }
}

void MultithreadExecutor::RegisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(remotesMu);
  remotes[rdesc.key] = rdesc;
}

void MultithreadExecutor::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(remotesMu);
  remotes.erase(rdesc.key);
}

void MultithreadExecutor::RegisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems[desc.id] = desc;
}

void MultithreadExecutor::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems.erase(desc.id);
}
}  // namespace io
}  // namespace mori
