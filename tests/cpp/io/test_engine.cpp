#include <fcntl.h>

#include <cassert>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/engine.hpp"

using namespace mori::io;

void TestMoriIOEngine() {
  IOEngineConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.backends = {BackendType::RDMA};
  config.gpuId = 0;
  IOEngine initiator("initiator", config);
  config.gpuId = 1;
  IOEngine target("target", config);

  EngineDesc initiatorEngineDesc = initiator.GetEngineDesc();
  EngineDesc targetEngineDesc = target.GetEngineDesc();

  initiator.RegisterRemoteEngine(targetEngineDesc);
  target.RegisterRemoteEngine(initiatorEngineDesc);

  void *initiatorBuf, *targetBuf;
  size_t bufSize = 1024 * 1024 * 4;
  HIP_RUNTIME_CHECK(hipMalloc(&initiatorBuf, bufSize));
  HIP_RUNTIME_CHECK(hipMalloc(&targetBuf, bufSize));
  HIP_RUNTIME_CHECK(hipMemset(targetBuf, 1, bufSize));

  MemoryDesc initatorMem =
      initiator.RegisterMemory(initiatorBuf, bufSize, 0, MemoryLocationType::GPU);
  MemoryDesc targetMem = target.RegisterMemory(targetBuf, bufSize, 0, MemoryLocationType::GPU);

  TransferStatus status;
  initiator.Read(initatorMem, 0, targetMem, 0, bufSize, status);
  printf("%d\n", reinterpret_cast<uint8_t*>(initiatorBuf)[511]);
  initiator.DeRegisterMemory(initatorMem);
  target.DeRegisterMemory(targetMem);
}

int main() { TestMoriIOEngine(); }