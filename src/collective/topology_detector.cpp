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
#include "mori/collective/topology_detector.hpp"

#include <mpi.h>

#include <cassert>

#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

void TopologyDetector::Initialize() {
  if (initialized) {
    return;
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(NULL, NULL);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &myPe);
  MPI_Comm_size(MPI_COMM_WORLD, &nPes);

  int status = shmem::ShmemMpiInit(MPI_COMM_WORLD);
  assert(status == 0 && "ShmemMpiInit failed");

  int shmem_pe = shmem::ShmemMyPe();
  int shmem_npes = shmem::ShmemNPes();
  assert(shmem_pe == myPe && "SHMEM PE mismatch with MPI rank");
  assert(shmem_npes == nPes && "SHMEM nPEs mismatch with MPI size");

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, myPe, MPI_INFO_NULL, &nodeComm);

  MPI_Comm_rank(nodeComm, &localRank);
  MPI_Comm_size(nodeComm, &localSize);

  int myLeaderRank = (localRank == 0) ? myPe : -1;
  int leaderRank = -1;
  MPI_Allreduce(&myLeaderRank, &leaderRank, 1, MPI_INT, MPI_MAX, nodeComm);
  nodeId = leaderRank;

  int isLeader = (localRank == 0) ? 1 : 0;
  MPI_Allreduce(&isLeader, &numNodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  gpusPerNode = localSize;

  intraNode = (numNodes == 1);

  initialized = true;
}

bool TopologyDetector::IsIntraNode() {
  assert(initialized && "TopologyDetector not initialized");
  return intraNode;
}

int TopologyDetector::GetNodeId() {
  assert(initialized && "TopologyDetector not initialized");
  return nodeId;
}

int TopologyDetector::GetLocalRank() {
  assert(initialized && "TopologyDetector not initialized");
  return localRank;
}

int TopologyDetector::GetLocalSize() {
  assert(initialized && "TopologyDetector not initialized");
  return localSize;
}

int TopologyDetector::GetNodeCount() {
  assert(initialized && "TopologyDetector not initialized");
  return numNodes;
}

std::pair<int, int> TopologyDetector::Get2DTopology() {
  assert(initialized && "TopologyDetector not initialized");
  return std::make_pair(numNodes, gpusPerNode);
}

int TopologyDetector::GetMyPe() {
  assert(initialized && "TopologyDetector not initialized");
  return myPe;
}

int TopologyDetector::GetNPes() {
  assert(initialized && "TopologyDetector not initialized");
  return nPes;
}

void TopologyDetector::Finalize() {
  if (!initialized) {
    return;
  }

  if (nodeComm != MPI_COMM_NULL) {
    MPI_Comm_free(&nodeComm);
    nodeComm = MPI_COMM_NULL;
  }

  shmem::ShmemFinalize();

  initialized = false;
  myPe = -1;
  nPes = -1;
  nodeId = -1;
  localRank = -1;
  localSize = -1;
  numNodes = -1;
  gpusPerNode = -1;
  intraNode = false;
}

}  // namespace collective
}  // namespace mori
