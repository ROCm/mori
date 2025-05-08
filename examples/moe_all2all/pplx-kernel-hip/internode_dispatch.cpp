#include <hip/hip_runtime.h>

namespace {

#define PPLX_HOST_DEVICE __host__ __device__

template <typename T>
PPLX_HOST_DEVICE T ceil_div(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
PPLX_HOST_DEVICE T round_up(T x, T y) {
  return ceil_div<T>(x, y) * y;
}

#define PPLX_DEVICE_ASSERT(cond)                                           \
  do {                                                                     \
    if (!(cond)) {                                                         \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      asm("trap;");                                                        \
    }                                                                      \
  } while (0)

__forceinline__ __device__ unsigned warp_sum(unsigned value) {
  // value += __shfl_xor_sync(0xffffffff, value, 16);
  // value += __shfl_xor_sync(0xffffffff, value, 8);
  // value += __shfl_xor_sync(0xffffffff, value, 4);
  // value += __shfl_xor_sync(0xffffffff, value, 2);
  // value += __shfl_xor_sync(0xffffffff, value, 1);
  value += __shfl_xor(0xffffffff, value, 32);
  value += __shfl_xor(0xffffffff, value, 16);
  value += __shfl_xor(0xffffffff, value, 8);
  value += __shfl_xor(0xffffffff, value, 4);
  value += __shfl_xor(0xffffffff, value, 2);
  value += __shfl_xor(0xffffffff, value, 1);
  return value;
}

template <unsigned NUM_WARPS, bool DO_SEND, bool DO_RECV>
__global__ __launch_bounds__(NUM_WARPS* warpSize, 1) void dispatchKernel(
    /* clang-format off */
    int32_t* outNumTokensPerExpert, 
    size_t outNumTokensPerExpertStrideElem, 
    std::byte* expertX,
    size_t expertXStrideElem, 
    size_t expertXStrideRow, 
    std::byte* dpX, 
    size_t dpXStrideElem,
    uint32_t* indices, 
    size_t indicesStrideElem, 
    size_t indicesStrideRow,
    size_t maxNumTokens,
    size_t numExperts, 
    unsigned rank, 
    unsigned worldSize, 
    unsigned dpSize, 
    size_t hiddenDim,
    size_t numExpertsPerToken, 
    unsigned* boundM, 
    unsigned m, 
    uint32_t* numTokensPerDP,
    uint32_t* sourceExpert, 
    uint32_t* sourceIndex, 
    uint32_t* sourceOffset, 
    uint32_t* sourceGroup,
    uint32_t* sourceToken, 
    uint64_t* numTokensBuffer, 
    uint64_t* numRecvBuffer,
    uint32_t& globalTokenIndex, 
    std::byte* xBufferIn, 
    std::byte* xBufferOut) {
  /* clang-format on */

  // Determine the rank, DP rank and per-rank constants.
  const unsigned numLocalExperts = numExperts / worldSize;
  const unsigned numDPGroups = worldSize / dpSize;
  const unsigned dpGroup = rank / dpSize;
  const unsigned dpRank = rank % dpSize;
  const unsigned tokenDim = hiddenDim;  // + hiddenDimScale;
  const unsigned tokenStride = round_up<unsigned>(tokenDim + sizeof(uint32_t), sizeof(int4));
  const unsigned WARP_SIZE = warpSize;
  const unsigned warpId = threadIdx.x / WARP_SIZE;
  const unsigned laneId = threadIdx.x % WARP_SIZE;

  // Determine the number of tokens populated which are to be sent.
  const unsigned numSendTokens = boundM ? __ldg(boundM) : m;
  PPLX_DEVICE_ASSERT(numSendTokens <= maxNumTokens);

  // Zero out the shared memory buffer.
  extern __shared__ std::byte sharedMemory[];
  if constexpr (DO_SEND) {
    uint32_t* tokenIndex = reinterpret_cast<uint32_t*>(sharedMemory);
    for (uint32_t i = threadIdx.x; i < numExperts; i += blockDim.x) {
      tokenIndex[i] = 0;
    }
    __syncthreads();

    if (warpId + 1 == NUM_WARPS) {
      // The experts are split across the available blocks.
      // The warp counts the number of tokens assigned to each expert.
      for (unsigned dstExpert = blockIdx.x * dpSize + dpRank; dstExpert < numExperts;
           dstExpert += gridDim.x * dpSize) {
        const uint32_t dstRank = dstExpert / numLocalExperts;
        const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

        unsigned count = 0;

#pragma unroll
        for (uint32_t i = laneId; i < numSendTokens * numExpertsPerToken; i += WARP_SIZE) {
          unsigned expert = __ldg(&indices[i]);
          if (expert == dstExpert) {
            count += 1;
          }
        }

        unsigned numTokensPerExpert = warp_sum(count);
        uint64_t* dstCount = &numTokensBuffer[dstLocalExpert * numDPGroups + dpGroup];

        if (laneId == 0) {
          // TODO
          // nvshmemx_signal_op(dstCount, numTokensPerExpert + 1, NVSHMEM_SIGNAL_SET, dstRank);
        }
      }

      // Clear out some buffers.
      if (blockIdx.x == 0) {
        for (uint32_t i = laneId; i < numLocalExperts; i += WARP_SIZE) {
          outNumTokensPerExpert[i] = 0;
        }
      }
    } else {
      // Send the tokens to the destination ranks through RDMA.
      const unsigned numGroupWarps = NUM_WARPS - 1;
      const unsigned numGroupThreads = numGroupWarps * WARP_SIZE;
      for (unsigned i = 0; i < numSendTokens; i++) {
        // Replicate the token count calculation across all blocks.
        if (threadIdx.x < numExpertsPerToken) {
          uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + threadIdx.x]);
          tokenIndex[dstExpert]++;
        }
        // If the token is assigned to this block, handle it.
        if (i % (gridDim.x * dpSize) == (blockIdx.x * dpSize + dpRank)) {
          // Copy the token to the symmetric buffer.
          std::byte* xInPtr = xBufferIn + i * tokenStride;
          const int4* srcX = (int4*)(dpX + i * dpXStrideElem);
          for (unsigned d = threadIdx.x; d * sizeof(int4) < hiddenDim; d += numGroupThreads) {
            ((int4*)xInPtr)[d] = srcX[d];
          }

          if (threadIdx.x == 0) {
            *((uint32_t*)(xInPtr + tokenDim)) = i;
          }

          // Synchronize the warps within this warp group.
          asm volatile("bar.sync 1, %0;" ::"r"(numGroupThreads));

          // Send the token to the other ranks, one send per warp.
          for (unsigned j = warpId; j < numExpertsPerToken; j += numGroupWarps) {
            const uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + j]);
            const uint32_t dstRank = dstExpert / numLocalExperts;
            const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

            const uint32_t index = tokenIndex[dstExpert] - 1;
            const uint32_t group = dstLocalExpert * numDPGroups + dpGroup;
            const unsigned loc = group * maxNumTokens + index;

            std::byte* destPointer = xBufferOut + loc * tokenStride;
            // TODO
            // nvshmemx_putmem_signal_nbi_warp(destPointer, xInPtr, tokenStride,
            // &numRecvBuffer[group],
            //                                 1, NVSHMEM_SIGNAL_ADD, dstRank);
          }
        }
      }
    }

    if (DO_RECV) {
      // TODO
      // cooperative_groups::this_grid().sync();
    }
  }

  if constexpr (DO_RECV) {
    // Wait for the token counts to be sent.
    const size_t numExpertsAndGroups = numLocalExperts * numDPGroups;
    const size_t expertsPerBlock = ceil_div<size_t>(numExpertsAndGroups, gridDim.x);
    uint32_t* sharedExpert = reinterpret_cast<uint32_t*>(sharedMemory);
    uint32_t* sharedToken = sharedExpert + expertsPerBlock;

    unsigned firstGroup = blockIdx.x * expertsPerBlock;
    unsigned lastGroup = std::min(firstGroup + expertsPerBlock, numExpertsAndGroups);

    for (unsigned group = firstGroup + threadIdx.x; group < lastGroup;
         group += gridDim.x * expertsPerBlock) {
      const uint32_t expert = group / numDPGroups;

      // Fetch the token count per DP, which is non-zero to indicate receipt.
      // Afterwards, wait for exactly that many tokens to be sent to us.
      // TODO
      // nvshmem_uint64_wait_until(&numTokensBuffer[group], NVSHMEM_CMP_NE, 0);
      size_t numTokens = numTokensBuffer[group] - 1;
      // TODO
      // nvshmem_uint64_wait_until(&numRecvBuffer[group], NVSHMEM_CMP_EQ, numTokens);

      numTokensPerDP[group] = numTokens;
      numTokensBuffer[group] = 0;
      numRecvBuffer[group] = 0;
      sharedExpert[group - firstGroup] = atomicAdd(&outNumTokensPerExpert[expert], numTokens);
      sharedToken[group - firstGroup] = atomicAdd(&globalTokenIndex, numTokens);
    }

    __syncthreads();

    for (unsigned group = firstGroup; group < lastGroup; group++) {
      const uint32_t expert = group / numDPGroups;
      const uint32_t dp = group % numDPGroups;
      const size_t numTokens = numTokensPerDP[group];
      auto expertStart = sharedExpert[group - firstGroup];
      auto tokenStart = sharedToken[group - firstGroup];

      for (unsigned i = threadIdx.x; i < numTokens; i += blockDim.x) {
        std::byte* xTokenBuffer = xBufferOut + (group * maxNumTokens + i) * tokenStride;
        uint32_t token = tokenStart + i;
        sourceIndex[token] = *((uint32_t*)(xTokenBuffer + tokenDim));
        sourceExpert[token] = expert;
        sourceOffset[token] = expertStart + i;
        sourceGroup[token] = dp;
        sourceToken[token] = i;
      }
    }

    // TODO
    // cooperative_groups::this_grid().sync();
    unsigned numRecvTokens = globalTokenIndex;

    for (unsigned i = blockIdx.x; i < numRecvTokens; i += gridDim.x) {
      auto expertLoc = sourceOffset[i];
      auto expert = sourceExpert[i];
      auto group = expert * numDPGroups + sourceGroup[i];

      std::byte* xTokenBuffer = xBufferOut + (group * maxNumTokens + sourceToken[i]) * tokenStride;
      std::byte* dstXExpert = expertX + expert * expertXStrideRow;

      const int4* srcX = (int4*)xTokenBuffer;
      int4* dstX = (int4*)(dstXExpert + expertLoc * expertXStrideElem);
      for (unsigned k = threadIdx.x; k * sizeof(int4) < hiddenDim; k += blockDim.x) {
        dstX[k] = srcX[k];
      }
    }
  }
}

}  // namespace

int main() {}