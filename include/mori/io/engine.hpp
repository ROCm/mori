#pragma once

#include <thread>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace ioengine {

struct IOEngineConfig {
  std::string host;
  int port;
  BackendTypeVec backends;
};

class IOEngine {
 public:
  IOEngine(EngineKey, IOEngineConfig);
  ~IOEngine();

  EngineDesc GetEngineDesc();
  void RegisterRemoteEngine(EngineDesc);
  void DeRegisterRemoteEngine(EngineDesc);

  //   MemoryDesc RegisterMemory(void* data, size_t length, MemLoc loc);
  //   void DeRegisterMemory(MemoryDesc);

  //   Status Write(MemoryDesc local, MemoryDesc remote, EngineDesc agent);

 private:
  void ControlPlaneLoop();
  void StartControlPlane();
  void ShutdownControlPlane();

 public:
  IOEngineConfig config;
  EngineDesc desc;

 private:
  std::unique_ptr<application::TCPContext> tcpContext;

  // Control plane related members
  int epfd{-1};
  std::thread ctrlPlaneThd;
  std::atomic<bool> running{false};
};

}  // namespace ioengine
}  // namespace mori