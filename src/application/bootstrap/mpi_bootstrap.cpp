// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
