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

  int GetLocalRank() const { return local_rank; }
  int GetWorldSize() const { return world_size; }

  virtual void Allgather(void* sendbuf, void* recvbuf, size_t sendcount) = 0;
  virtual void Barrier() = 0;

 protected:
  int local_rank{0};
  int world_size{0};
};

}  // namespace application
}  // namespace mori