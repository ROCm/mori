#include "mori/io/engine.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/logging.hpp"

#include <thread>
#include <vector>
#include <cstring>

using namespace mori::io;

int main() {
  SetLogLevel(spdlog::level::info);
  IOEngineConfig cfgA{"127.0.0.1", 34567};
  IOEngineConfig cfgB{"127.0.0.1", 34568};
  IOEngine engineA("engineA", cfgA);
  IOEngine engineB("engineB", cfgB);

  TcpBackendConfig tcfg; // default
  engineA.CreateBackend(BackendType::TCP, tcfg);
  engineB.CreateBackend(BackendType::TCP, tcfg);

  EngineDesc descA = engineA.GetEngineDesc();
  EngineDesc descB = engineB.GetEngineDesc();

  engineA.RegisterRemoteEngine(descB);
  engineB.RegisterRemoteEngine(descA);

  std::vector<char> bufA(1024, 0);
  std::vector<char> bufB(1024, 0);
  strcpy(bufA.data(), "Hello TCP Backend");

  auto memA = engineA.RegisterMemory(bufA.data(), bufA.size(), 0, MemoryLocationType::CPU);
  auto memB = engineB.RegisterMemory(bufB.data(), bufB.size(), 0, MemoryLocationType::CPU);

  TransferStatus st;
  auto tid = engineA.AllocateTransferUniqueId();
  engineA.Write(memA, 0, memB, 0, strlen(bufA.data())+1, &st, tid);
  if (st.Code() != StatusCode::SUCCESS) {
    printf("Write failed code %u message %s\n", (unsigned)st.Code(), st.Message().c_str());
  }

  // For simplicity, only demonstrate write path; read path in current TCP backend
  // mirrors write but synchronous; a complete round-trip would require memory id propagation.
  printf("Buffer on engineB after write: %s\n", bufB.data());
  return 0;
}
