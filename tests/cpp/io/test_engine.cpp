#include <fcntl.h>

#include <cassert>
#include <vector>

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
}

int main() { TestMoriIOEngine(); }