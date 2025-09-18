#include "mori/io/engine.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/logging.hpp"

#include <thread>
#include <vector>
#include <cstring>
#include <hip/hip_runtime.h>

using namespace mori::io;

int main() {
  SetLogLevel("info");
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

  const size_t bufSize = 1024;
  std::vector<char> hostA(bufSize, 0), hostB(bufSize, 0);
  const char* msg = "Hello TCP Backend";
  std::memcpy(hostA.data(), msg, std::strlen(msg)+1);

  void* devA = nullptr; void* devB = nullptr;
  hipMalloc(&devA, bufSize);
  hipMalloc(&devB, bufSize);
  hipMemset(devB, 0, bufSize);
  hipMemcpy(devA, hostA.data(), bufSize, hipMemcpyHostToDevice);

  auto memA = engineA.RegisterMemory(devA, bufSize, 0, MemoryLocationType::GPU);
  auto memB = engineB.RegisterMemory(devB, bufSize, 0, MemoryLocationType::GPU);

  TransferStatus st;
  auto tid = engineA.AllocateTransferUniqueId();
  engineA.Write(memA, 0, memB, 0, std::strlen(msg)+1, &st, tid);
  if (st.Code() != StatusCode::SUCCESS) {
    printf("Write failed code %u message %s\n", (unsigned)st.Code(), st.Message().c_str());
  }

  // Copy device B buffer back to host to verify
  hipMemcpy(hostB.data(), devB, bufSize, hipMemcpyDeviceToHost);
  printf("Buffer on engineB after write: %s\n", hostB.data());

  // Cleanup
  engineA.DeregisterMemory(memA);
  engineB.DeregisterMemory(memB);
  hipFree(devA);
  hipFree(devB);
  return 0;
}
