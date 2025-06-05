#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace mori {
namespace application {

class TorchBootstrapNetwork : public BootstrapNetwork {
 public:
  TorchBootstrapNetwork(const std::string& groupName);
  ~TorchBootstrapNetwork();

  void Initialize();
  void Finalize();

  void Allgather(void* sendbuf, void* recvbuf, size_t sendcount);
  void AllToAll(void* sendbuf, void* recvbuf, size_t sendcount);
  void Barrier();

 private:
  c10::intrusive_ptr<c10d::ProcessGroup> group;
};

}  // namespace application
}  // namespace mori