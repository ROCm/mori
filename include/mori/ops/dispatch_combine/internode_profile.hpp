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
#pragma once

// Format: X(EnumName, PyBindName)
// The PyBindName is used for automatic Python binding and visualization.

#define INTERNODE_V1_SLOTS(X)                  \
  X(CombineInterNode, "combine_inter_node")    \
  X(BatchProcessing, "batch_processing")       \
  X(ChunkPolling, "chunk_polling")             \
  X(ChunkReady, "chunk_ready")                 \
  X(TokenProcessing, "token_processing")       \
  X(PointerSetup, "pointer_setup")             \
  X(TokenAccumulation, "token_accumulation")   \
  X(WeightAccumulation, "weight_accumulation") \
  X(ChunkCompletion, "chunk_completion")       \
  X(AtomicIncrement, "atomic_increment")       \
  X(FlagReset, "flag_reset")                   \
  X(ShmemPutOp, "shmem_put_op")                \
  X(BarrierSync, "barrier_sync")               \
  X(BarrierWait, "barrier_wait")

namespace mori {
namespace moe {
namespace v1 {

enum class InterNodeSlot : int {
#define X(name, str) name,
  INTERNODE_V1_SLOTS(X)
#undef X
      MAX_SLOTS
};

}  // namespace v1
}  // namespace moe
}  // namespace mori
