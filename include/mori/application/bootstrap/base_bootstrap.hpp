#pragma once

#include <cstddef>

namespace mori {
namespace application {

class BootstrapNetwork {
 public:
  BootstrapNetwork() = default;
  virtual ~BootstrapNetwork() = default;

  virtual void Initialize() = 0;
  virtual void Finalize() = 0;

  int GetLocalRank() const { return localRank; }
  int GetWorldSize() const { return worldSize; }

  virtual void Allgather(void* sendbuf, void* recvbuf, size_t sendcount) = 0;
  virtual void AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) = 0;

  virtual void Barrier() = 0;

 protected:
  int localRank{0};
  int worldSize{0};
};

}  // namespace application
}  // namespace mori