#pragma once

#include <mpi.h>

#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace mori {
namespace application {

class MpiBootstrapNetwork : public BootstrapNetwork {
 public:
  MpiBootstrapNetwork(MPI_Comm mpi_comm);
  ~MpiBootstrapNetwork();

  void Initialize();
  void Finalize();

  void Allgather(void* sendbuf, void* recvbuf, size_t sendcount);
  void Barrier();

 private:
  MPI_Comm mpi_comm;
};

}  // namespace application
}  // namespace mori