#include "mori/application/bootstrap/mpi_bootstrap.hpp"

#include <mpi.h>

#include <cassert>

namespace mori {
namespace application {

MpiBootstrapNetwork::MpiBootstrapNetwork(MPI_Comm mpi_comm) : mpi_comm(mpi_comm) {}

MpiBootstrapNetwork::~MpiBootstrapNetwork() {}

void MpiBootstrapNetwork::Initialize() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(mpi_comm, &world_size);
  MPI_Comm_rank(mpi_comm, &local_rank);
}

void MpiBootstrapNetwork::Finalize() { MPI_Finalize(); }

void MpiBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  int status = MPI_Allgather(sendbuf, sendcount, MPI_CHAR, recvbuf, sendcount, MPI_CHAR, mpi_comm);
  assert(!status);
}

void MpiBootstrapNetwork::Barrier() {
  int status = MPI_Barrier(mpi_comm);
  assert(!status);
}

}  // namespace application
}  // namespace mori