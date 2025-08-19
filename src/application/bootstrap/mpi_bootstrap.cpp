#include "mori/application/bootstrap/mpi_bootstrap.hpp"

#include <mpi.h>

#include <cassert>

namespace mori {
namespace application {

MpiBootstrapNetwork::MpiBootstrapNetwork(MPI_Comm mpi_comm) : mpi_comm(mpi_comm) { Initialize(); }

MpiBootstrapNetwork::~MpiBootstrapNetwork() { Finalize(); }

void MpiBootstrapNetwork::Initialize() {
  int initialized;
  int status = MPI_Initialized(&initialized);
  assert(!status);
  if (!initialized) {
    MPI_Init(NULL, NULL);
  }
  MPI_Comm_size(mpi_comm, &worldSize);
  MPI_Comm_rank(mpi_comm, &localRank);
}

void MpiBootstrapNetwork::Finalize() {
  int finalized = false;
  int status = MPI_Finalized(&finalized);
  assert(!status);

  if (!finalized) MPI_Finalize();
}

void MpiBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  int status = MPI_Allgather(sendbuf, sendcount, MPI_CHAR, recvbuf, sendcount, MPI_CHAR, mpi_comm);
  assert(!status);
}

void MpiBootstrapNetwork::AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) {
  int status = MPI_Alltoall(sendbuf, sendcount, MPI_CHAR, recvbuf, sendcount, MPI_CHAR, mpi_comm);
  assert(!status);
}

void MpiBootstrapNetwork::Barrier() {
  int status = MPI_Barrier(mpi_comm);
  assert(!status);
}

}  // namespace application
}  // namespace mori