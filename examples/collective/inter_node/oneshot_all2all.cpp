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
#include <mpi.h>

#include <cassert>

#include "mori/application/utils/check.hpp"
#include "mori/collective/core/all2all_manager.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::collective;

void testAll2all() {
  int status;
  // Initialize All2allManager
  All2allManager<uint32_t> all2allManager;
  All2allConfig config;
  config.algorithm = All2allAlgorithm::INTER_ONE_SHOT;
  status = all2allManager.Initialize(config);
  assert(!status);

  int myPe = TopologyDetector::GetMyPe();
  int npes = TopologyDetector::GetNPes();

  printf("PE %d of %d started\n", myPe, npes);

  // Configuration
  int elemsPerPe = npes;
  int buffSize = elemsPerPe * sizeof(uint32_t);

  // Allocate device memory for data buffer
  uint32_t* dataBuff = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&dataBuff, buffSize));
  assert(dataBuff != nullptr);

  // Initialize data buffer: each PE initializes all chunks with its PE number + 1
  uint32_t* hostData = new uint32_t[elemsPerPe];
  for (int i = 0; i < elemsPerPe; i++) {
    hostData[i] = myPe + 1;
  }
  HIP_RUNTIME_CHECK(hipMemcpy(dataBuff, hostData, buffSize, hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Print initial data
  printf("PE %d: Initial data: ", myPe);
  for (int i = 0; i < elemsPerPe; i++) {
    printf("%u ", hostData[i]);
  }
  printf("\n");

  MPI_Barrier(MPI_COMM_WORLD);

  // Perform AllReduce operation
  status = all2allManager.All2all(dataBuff, dataBuff, elemsPerPe);
  assert(!status);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  // Copy result back to host for printing
  HIP_RUNTIME_CHECK(hipMemcpy(hostData, dataBuff, buffSize, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  printf("PE %d: All2all result: ", myPe);
  for (int i = 0; i < elemsPerPe; i++) {
    printf("%u ", hostData[i]);
  }
  printf("\n");

  MPI_Barrier(MPI_COMM_WORLD);

  // Cleanup
  HIP_RUNTIME_CHECK(hipFree(dataBuff));
  delete[] hostData;

  all2allManager.Finalize();

  if (myPe == 0) {
    printf("\nTest completed successfully!\n");
  }
}

int main(int argc, char* argv[]) {
  testAll2all();
  return 0;
}
