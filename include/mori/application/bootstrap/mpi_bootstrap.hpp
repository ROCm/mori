#pragma once

#include <mpi.h>

#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace mori {
namespace application {

class MpiBootstrapNetwork : public BootstrapNetwork {
 public:
  MpiBootstrapNetwork();
  ~MpiBootstrapNetwork();

  void Initialize();
  void Finalize();

  void Allgather(void* sendbuf, void* recvbuf, size_t sendcount);
  void Barrier();
};

}  // namespace application
}  // namespace mori