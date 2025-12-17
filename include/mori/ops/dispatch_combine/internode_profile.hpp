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

#define PROFILER_TAG_ALL (1 << 0)

#define COMBINE_INTER_SLOTS(X)                \
  X(DispatchInterSend, "dispatch_inter_send") \
  X(DispSendStaging, "disp_send_staging")     \
  X(DispSendRDMA, "disp_send_rdma")           \
  X(DispatchInterRecv, "dispatch_inter_recv") \
  X(DispRecvPoll, "disp_recv_poll")           \
  X(DispRecvProcess, "disp_recv_process")     \
  X(DispatchIntra, "dispatch_intra")          \
  X(DispIntraToken, "disp_intra_token")       \
  X(DispatchSync, "dispatch_sync")            \
  X(DispSyncBarrier, "disp_sync_barrier")     \
  X(DispSyncQuiet, "disp_sync_quiet")         \
  X(CombineSync, "combine_sync")              \
  X(CombSyncCopy, "comb_sync_copy")           \
  X(CombSyncBarrier, "comb_sync_barrier")     \
  X(CombineIntra, "combine_intra")            \
  X(CombIntraToken, "comb_intra_token")       \
  X(CombineInterNode, "combine_inter_node")   \
  X(CombInterChunk, "comb_inter_chunk")       \
  X(CombInterToken, "comb_inter_token")       \
  X(CombInterShmem, "comb_inter_shmem")       \
  X(CombineAll, "combine_all")                \
  X(CombAllToken, "comb_all_token")

namespace mori {
namespace moe {
namespace v1 {

enum class InterNodeSlot : int {
#define X(name, str) name,
  COMBINE_INTER_SLOTS(X)
#undef X
      MAX_SLOTS
};

}  // namespace v1
}  // namespace moe
}  // namespace mori
