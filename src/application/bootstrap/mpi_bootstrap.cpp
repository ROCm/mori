#include "mori/application/bootstrap/mpi_bootstrap.hpp"

#include <mpi.h>

#include <cassert>

namespace mori {
namespace application {

MpiBootstrapNetwork::MpiBootstrapNetwork() {}

MpiBootstrapNetwork::~MpiBootstrapNetwork() {}

void MpiBootstrapNetwork::Initialize() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
}

void MpiBootstrapNetwork::Finalize() { MPI_Finalize(); }

void MpiBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  int status =
      MPI_Allgather(sendbuf, sendcount, MPI_CHAR, recvbuf, sendcount, MPI_CHAR, MPI_COMM_WORLD);
  assert(!status);
}

}  // namespace application
}  // namespace mori