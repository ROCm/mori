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
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cassert>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/io.hpp"

using namespace mori::io;

int GetFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }

  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    close(fd);
    return -1;
  }

  int port = ntohs(addr.sin_port);

  close(fd);
  return port;
}

void TestMoriIOEngine() {
  SetLogLevel("trace");

  IOEngineConfig config;
  config.host = "127.0.0.1";
  config.port = GetFreePort();
  IOEngine initiator("initiator", config);

  RdmaBackendConfig rdmaConfig{};
  initiator.CreateBackend(BackendType::RDMA, rdmaConfig);

  int newPort = GetFreePort();
  assert(newPort != config.port);
  config.port = newPort;
  IOEngine target("target", config);
  target.CreateBackend(BackendType::RDMA, rdmaConfig);

  EngineDesc initiatorEngineDesc = initiator.GetEngineDesc();
  EngineDesc targetEngineDesc = target.GetEngineDesc();

  initiator.RegisterRemoteEngine(targetEngineDesc);
  target.RegisterRemoteEngine(initiatorEngineDesc);

  void *initiatorBuf, *targetBuf;
  size_t bufSize = 1024 * 1024 * 4;
  HIP_RUNTIME_CHECK(hipMalloc(&initiatorBuf, bufSize));
  HIP_RUNTIME_CHECK(hipMalloc(&targetBuf, bufSize));
  HIP_RUNTIME_CHECK(hipMemset(targetBuf, 1, bufSize));

  MemoryDesc initiatorMem =
      initiator.RegisterMemory(initiatorBuf, bufSize, 0, MemoryLocationType::GPU);
  MemoryDesc targetMem = target.RegisterMemory(targetBuf, bufSize, 0, MemoryLocationType::GPU);

  int transferCnt = 64;

  for (int i = 0; i < transferCnt; i++) {
    TransferStatus initiatorStatus, targetStatus;
    TransferUniqueId id = initiator.AllocateTransferUniqueId();
    initiator.Read(initiatorMem, 0, targetMem, 0, bufSize, &initiatorStatus, id);
    printf("read %d id %d\n", i, id);
    while (initiatorStatus.Code() == StatusCode::INIT) {
    }
    while (targetStatus.Code() == StatusCode::INIT) {
      target.PopInboundTransferStatus(initiator.GetEngineDesc().key, id, &targetStatus);
    }
    printf("Status message initiator %s target %s read value %d\n",
           initiatorStatus.Message().c_str(), targetStatus.Message().c_str(),
           reinterpret_cast<uint8_t*>(initiatorBuf)[511]);
  }

  std::vector<TransferStatus> initiatorStatusVec(transferCnt);
  std::vector<TransferStatus> targetStatusVec(transferCnt);
  std::vector<TransferUniqueId> trsfIds(transferCnt);

  for (int i = 0; i < transferCnt; i++) {
    TransferUniqueId id = initiator.AllocateTransferUniqueId();
    trsfIds[i] = id;
    initiator.Read(initiatorMem, 0, targetMem, 0, bufSize, &initiatorStatusVec[i], id);
  }

  for (int i = 0; i < transferCnt; i++) {
    while (initiatorStatusVec[i].Code() == StatusCode::INIT) {
    }
    while (targetStatusVec[i].Code() == StatusCode::INIT) {
      target.PopInboundTransferStatus(initiator.GetEngineDesc().key, trsfIds[i],
                                      &targetStatusVec[i]);
    }
    printf("Status message initiator %s target %s read value %d\n",
           initiatorStatusVec[i].Message().c_str(), targetStatusVec[i].Message().c_str(),
           reinterpret_cast<uint8_t*>(initiatorBuf)[511]);
  }

  initiator.DeregisterMemory(initiatorMem);
  target.DeregisterMemory(targetMem);
}

int main() { TestMoriIOEngine(); }
