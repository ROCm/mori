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
//
// Minimal test for C++ AOT launch API.
// Requires: BUILD_OPS_DEVICE=ON, shmem initialized, 8 GPUs.
//
// Usage:
//   mpiexec --allow-run-as-root -np 8 ./test_aot_launch <hsaco_dir>

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/launch.hpp"
#include "mori/shmem/shmem_api.hpp"

using namespace mori::moe;
using namespace mori::shmem;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  hipSetDevice(rank);

  // Initialize shmem via MPI
  ShmemMpiInit(MPI_COMM_WORLD);

  // Load AOT-compiled kernels
  // AutoLoad searches lib/<arch>_<nic>/ for matching GPU architecture
  std::string base_dir = (argc > 1) ? argv[1] : "lib";
  auto& reg = KernelRegistry::Instance();
  reg.AutoLoad(base_dir);
  printf("[Rank %d] Loaded kernels from %s\n", rank, base_dir.c_str());

  // Configuration (DeepSeek V3 style: 256 experts, top-8, 8 GPUs)
  const int hidden_dim = 7168;
  const int num_tokens = 128;
  const int num_experts_per_rank = 32;
  const int num_experts_per_token = 8;

  EpDispatchCombineConfig config;
  config.rank = rank;
  config.worldSize = world_size;
  config.hiddenDim = hidden_dim;
  config.scaleDim = 0;
  config.scaleTypeSize = 1;
  config.maxTokenTypeSize = 4;
  config.maxNumInpTokenPerRank = num_tokens;
  config.numExpertPerRank = num_experts_per_rank;
  config.numExpertPerToken = num_experts_per_token;
  config.warpNumPerBlock = 16;
  config.blockNum = 80;
  config.useExternalInpBuffer = true;
  config.kernelType = KernelType::IntraNode;
  config.gpuPerNode = world_size;
  config.rdmaBlockNum = 0;
  config.numQpPerPe = 1;

  {  // Scope handle so it destructs before ShmemFinalize
  EpDispatchCombineHandle handle(config);

  // Allocate test data
  size_t inp_size = num_tokens * hidden_dim * sizeof(hip_bfloat16);
  size_t w_size = num_tokens * num_experts_per_token * sizeof(float);
  size_t idx_size = num_tokens * num_experts_per_token * sizeof(index_t);

  void *input, *weights_buf, *indices_buf;
  hipMalloc(&input, inp_size);
  hipMalloc(&weights_buf, w_size);
  hipMalloc(&indices_buf, idx_size);

  // Generate valid test data on host then copy to device
  {
    std::vector<float> h_weights(num_tokens * num_experts_per_token, 1.0f);
    hipMemcpy(weights_buf, h_weights.data(), w_size, hipMemcpyHostToDevice);

    // Random top-K expert indices (each token picks K unique experts)
    int total_experts = num_experts_per_rank * world_size;
    std::vector<index_t> h_indices(num_tokens * num_experts_per_token);
    srand(42 + rank);
    for (int t = 0; t < num_tokens; t++) {
      for (int k = 0; k < num_experts_per_token; k++) {
        h_indices[t * num_experts_per_token + k] = rand() % total_experts;
      }
    }
    hipMemcpy(indices_buf, h_indices.data(), idx_size, hipMemcpyHostToDevice);
    hipMemset(input, 0, inp_size);
  }

  hipStream_t stream;
  hipStreamCreate(&stream);

  // Launch dispatch via AOT C++ API
  printf("[Rank %d] Launching dispatch...\n", rank);
  LaunchDispatch(handle, input, weights_buf, nullptr, indices_buf,
                 num_tokens, HIP_R_16BF, /*block_num=*/80,
                 /*rdma_block_num=*/0, /*warp_per_block=*/16,
                 stream, hidden_dim);
  hipStreamSynchronize(stream);

  // After dispatch, get the actual received token count
  index_t recv_count = handle.GetCurRankNumToken();
  printf("[Rank %d] Dispatch completed, received %d tokens\n", rank, recv_count);

  // Get dispatch output buffer (shmem buffer where received tokens landed)
  void* dispatch_out = handle.shmemDispatchOutTokMemObj->Get();
  void* dispatch_out_weights = handle.shmemDispatchOutWeightsMemObj->Get();

  // Simulate expert computation (identity: just use dispatch output as-is)
  // In real usage, expert MLP would produce new output here.

  // Launch combine via AOT C++ API
  // Key: use recv_count (not num_tokens) and dispatch output buffers
  printf("[Rank %d] Launching combine...\n", rank);
  LaunchCombine(handle, dispatch_out, dispatch_out_weights, indices_buf,
                recv_count, HIP_R_16BF, /*block_num=*/80,
                /*rdma_block_num=*/0, /*warp_per_block=*/8,
                /*use_external_inp_buf=*/-1, stream, hidden_dim);
  hipStreamSynchronize(stream);
  printf("[Rank %d] Combine completed\n", rank);

  // Reset
  LaunchReset(handle, stream);
  hipStreamSynchronize(stream);

  // Cleanup: destroy handle BEFORE ShmemFinalize (handle frees shmem buffers)
  hipStreamDestroy(stream);
  hipFree(input);
  hipFree(weights_buf);
  hipFree(indices_buf);
  }  // handle goes out of scope here, freeing shmem buffers

  ShmemFinalize();

  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }

  if (rank == 0) {
    printf("\n=== AOT launch test PASSED ===\n");
  }
  return 0;
}
