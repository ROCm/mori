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

// Sub-group intra-node SDMA broadcast host handle (this work, M4 leader-only
// design).
//
// This is the intra-node *placement* phase of the hierarchical cross-node
// AllGather's leader-only variant: the node leader (local_rank 0) gathers the
// full N*G output into a staging buffer (inter-node RDMA ring), then this
// broadcast SDMA-copies that whole buffer over the XGMI copy engines into every
// local rank's output. It is the dual of ``IntraNodeSubGroupAllgatherSdma``
// (one source -> all members, instead of all members -> all members) and is
// driven from Python with the same prepare -> launch kernel -> finish pattern.
//
// Compared to the "every-rank-direct" decomposition (every local rank rings its
// node-block over the NIC => G x redundant inter-node traffic), leader-only ring
// + this broadcast crosses the NIC only once per node-block.

#ifndef INTRA_NODE_SUBGROUP_BROADCAST_SDMA_CLASS_HPP
#define INTRA_NODE_SUBGROUP_BROADCAST_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

class IntraNodeSubGroupBroadcastSdma {
 private:
  int myPe_;
  int npes_;

  // Sub-group descriptor. The broadcast runs over the arithmetic sub-group of
  // global PEs {peBase_, peBase_+peStride_, ..., peBase_+(groupSize_-1)*peStride_};
  // this PE is at position groupPos_. The root (source) is groupPos_==0.
  int groupSize_;
  int groupPos_;
  int peBase_;
  int peStride_;

  // Symmetric output buffer: holds the full broadcast payload (elementCount u32
  // lanes). Registered (not just malloc'd) so the SDMA queue handles populate.
  void* out_;
  size_t outBytes_;
  application::SymmMemObjPtr outObj_;

  // Single arrival flag slot (one source). npes_ slots allocated for uniformity
  // with the gather handle. Monotonic generation token avoids a per-call reset.
  void* flags_;
  application::SymmMemObjPtr flagsObj_;
  uint64_t seq_;

  CclBroadcastSubGroupArgs<uint32_t> jit_args_;

  IntraNodeSubGroupBroadcastSdma(const IntraNodeSubGroupBroadcastSdma&) = delete;
  IntraNodeSubGroupBroadcastSdma& operator=(const IntraNodeSubGroupBroadcastSdma&) = delete;

 public:
  // groupSize<0 selects the flat whole-world broadcast (groupSize=npes,
  // groupPos=myPe, peBase=0, peStride=1).
  IntraNodeSubGroupBroadcastSdma(int myPe, int npes, size_t out_buffer_bytes, int groupSize = -1,
                                 int groupPos = -1, int peBase = 0, int peStride = 1)
      : myPe_(myPe),
        npes_(npes),
        out_(nullptr),
        outBytes_(out_buffer_bytes),
        flags_(nullptr),
        seq_(0) {
    if (groupSize < 0) {
      groupSize_ = npes_;
      groupPos_ = myPe_;
      peBase_ = 0;
      peStride_ = 1;
    } else {
      if (groupSize < 1 || groupPos < 0 || groupPos >= groupSize || peStride < 1) {
        throw std::runtime_error("IntraNodeSubGroupBroadcastSdma: invalid sub-group descriptor");
      }
      groupSize_ = groupSize;
      groupPos_ = groupPos;
      peBase_ = peBase;
      peStride_ = peStride;
    }

    out_ = shmem::ShmemMalloc(outBytes_);
    if (out_ == nullptr)
      throw std::runtime_error("IntraNodeSubGroupBroadcastSdma: out ShmemMalloc failed");
    // The broadcast payload buffer is gathered/consumed ONLY via P2P/SDMA (XGMI)
    // within the node — never an RDMA src/dst. Register P2P/SDMA-only
    // (rdmaRegister=false) so it dodges the ionic single-MR limit at large sizes
    // (matches IntraNodeSubGroupAllgatherSdma::out_).
    outObj_ = shmem::ShmemSymmetricRegister(out_, outBytes_, /*rdmaRegister=*/false);
    if (!outObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupBroadcastSdma: out register failed");

    size_t flagsBytes = static_cast<size_t>(npes_) * sizeof(uint64_t);
    flags_ = shmem::ShmemMalloc(flagsBytes);
    if (flags_ == nullptr)
      throw std::runtime_error("IntraNodeSubGroupBroadcastSdma: flags ShmemMalloc failed");
    (void)hipMemset(flags_, 0, flagsBytes);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_);
    if (!flagsObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupBroadcastSdma: flags query failed");
  }

  ~IntraNodeSubGroupBroadcastSdma() {
    if (out_) shmem::ShmemFree(out_);
    if (flags_) shmem::ShmemFree(flags_);
  }

  // Barrier so all members are primed (flags already monotonic), stage the
  // root's payload into the symmetric buffer, then build the kernel args. On the
  // root ``input`` is the full payload to broadcast (device ptr, count_u32 u32
  // lanes); on non-root members ``input`` is ignored.
  int64_t prepare_sync(uintptr_t input, size_t count_u32, hipStream_t stream) {
    if (count_u32 * sizeof(uint32_t) > outBytes_) {
      throw std::runtime_error("IntraNodeSubGroupBroadcastSdma: message exceeds out capacity");
    }
    uint64_t flag_token = ++seq_;

    // The root stages its payload into its own symmetric out buffer so the
    // kernel reads a stable source (and the root's own output is already filled
    // when the kernel writes peerPtrs[root] back to it -- idempotent).
    if (groupPos_ == 0 && input != 0) {
      (void)hipMemcpyAsync(out_, reinterpret_cast<void*>(input), count_u32 * sizeof(uint32_t),
                           hipMemcpyDeviceToDevice, stream);
      (void)hipStreamSynchronize(stream);
    }

    // All members enter the broadcast together (flags are monotonic; the token
    // distinguishes this call from the previous one without a reset).
    shmem::ShmemBarrierAll();

    jit_args_.myPe = myPe_;
    jit_args_.groupSize = groupSize_;
    jit_args_.groupPos = groupPos_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.input = reinterpret_cast<uint32_t*>(out_);
    jit_args_.dstMemObj = outObj_;
    jit_args_.flagsMemObj = flagsObj_;
    jit_args_.elementCount = count_u32;
    jit_args_.dstBaseOffset = 0;
    jit_args_.flagVal = flag_token;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // Copy the full broadcast payload out to the user buffer and synchronize, then
  // barrier so no peer reuses the buffer early.
  double finish_sync(uintptr_t output, size_t count_u32, hipStream_t stream) {
    size_t total = count_u32 * sizeof(uint32_t);
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    shmem::ShmemBarrierAll();
    return 0.0;
  }

  int npes() const { return npes_; }
};

}  // namespace collective
}  // namespace mori

#endif  // INTRA_NODE_SUBGROUP_BROADCAST_SDMA_CLASS_HPP
