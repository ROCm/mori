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

  for (int i = 0; i < 64; i++) {
    TransferStatus initiatorStatus, targetStatus;
    TransferUniqueId id = initiator.AllocateTransferUniqueId();
    initiator.Read(initatorMem, 0, targetMem, 0, bufSize, &initiatorStatus, id);
    while (initiatorStatus.Code() == StatusCode::INIT) {
    }
    while (targetStatus.Code() == StatusCode::INIT) {
      target.QueryAndAckInboundTransferStatus(initiator.GetEngineDesc().key, id, &targetStatus);
    }
    printf("Status message initiator %s target %s read value %d\n", initiatorStatus.Message().c_str(),targetStatus.Message().c_str(),
           reinterpret_cast<uint8_t*>(initiatorBuf)[511]);
  }

  std::vector<TransferStatus> initiatorStatusVec(64);
  std::vector<TransferStatus> targetStatusVec(64);
  std::vector<TransferUniqueId> trsfIds(64);

  for (int i = 0; i < 64; i++) {
    TransferUniqueId id = initiator.AllocateTransferUniqueId();
    trsfIds[i] = id;
    initiator.Read(initatorMem, 0, targetMem, 0, bufSize, &initiatorStatusVec[i], id);
  }

  for (int i = 0; i < 64; i++) {
    while (initiatorStatusVec[i].Code() == StatusCode::INIT) {
    }
    while (targetStatusVec[i].Code() == StatusCode::INIT) {
      target.QueryAndAckInboundTransferStatus(initiator.GetEngineDesc().key, trsfIds[i],
                                              &targetStatusVec[i]);
    }
    printf("Status message %s read value %d\n", initiatorStatusVec[i].Message().c_str(),
           reinterpret_cast<uint8_t*>(initiatorBuf)[511]);
  }

  initiator.DeRegisterMemory(initatorMem);
  target.DeRegisterMemory(targetMem);
}

int main() { TestMoriIOEngine(); }