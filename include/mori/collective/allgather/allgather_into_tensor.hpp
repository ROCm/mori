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

#ifndef MORI_COLLECTIVE_ALLGATHER_INTO_TENSOR_HPP
#define MORI_COLLECTIVE_ALLGATHER_INTO_TENSOR_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mori/collective/allgather/oneshot_allgather_sdma_class.hpp"

namespace mori {
namespace collective {

// ---------------------------------------------------------------------------
// DataType
// ---------------------------------------------------------------------------
// Mirrors the subset of ncclDataType_t / at::ScalarType we actually need to
// dispatch over.  Keeping the enum self-contained avoids dragging the torch
// headers into the collective core (cf. ncclAllGather, which takes a plain
// ncclDataType_t and never sees an at::Tensor).
enum class DataType : int {
  kInt8 = 0,
  kUint8 = 1,
  kInt16 = 2,
  kUint16 = 3,
  kInt32 = 4,
  kUint32 = 5,
  kInt64 = 6,
  kUint64 = 7,
  kFloat16 = 8,
  kBFloat16 = 9,
  kFloat32 = 10,
  kFloat64 = 11,
};

// Element size in bytes for `dtype`, or 0 for unknown values.
size_t SizeOf(DataType dtype);

// ---------------------------------------------------------------------------
// AllGatherIntoTensor
// ---------------------------------------------------------------------------
//
// SDMA-backed equivalent of ncclAllGather / torch.distributed.all_gather_into_tensor.
//
//   sendbuff has `sendcount * SizeOf(dtype)` bytes of input on every rank.
//   recvbuff has `npes * sendcount * SizeOf(dtype)` bytes of output on every rank.
//
// The instance owns the SDMA "communicator" state (symmetric input/output
// transit buffers, completion flags) — analogous to ncclComm_t.
//
// Example::
//
//   AllGatherIntoTensor ag(my_pe, npes, /*max_input_bytes=*/256*1024*1024);
//   ag(send_ptr, recv_ptr, numel, DataType::kBFloat16, stream);
//
// Internally this is a thin dispatcher on top of AllgatherSdma<uint32_t>:
// the SDMA kernel is byte/uint32-aligned and dtype-agnostic, exactly the
// way ncclAllGather treats `datatype` only as a multiplier on `sendcount`.
class AllGatherIntoTensor {
 public:
  // Same constructor flavours as AllgatherSdma: either give one combined
  // transit buffer size (split internally 50/50 between input and output),
  // or give input/output sizes separately.
  //
  // ``auto_register`` (default false) controls the experimental zero-copy
  // path: the first time a given ``recvbuff`` is seen, the class
  // internally invokes the *collective* ``register_output_buffer`` step
  // (every rank reaches it simultaneously inside the allgather entry
  // point, so the collective is well-formed); subsequent calls with the
  // same buffer hit the cache and the SDMA kernel writes directly into
  // the user's recv tensor — no transit copy.
  //
  // IMPORTANT: this only produces correct results if ``recvbuff`` was
  // allocated with ``hipExtMallocWithFlags(hipDeviceMallocUncached)``
  // (i.e. via ``mori.shmem.shmem_malloc`` or equivalent).  PyTorch's
  // caching allocator uses cached device memory, so the SDMA copy engine
  // writes will sit in the GPU L2 cache stale and downstream kernels
  // read zeros.  Hence the safe default is ``false`` — the class falls
  // back to the transit-buffer + post-memcpy path, which is byte-exact
  // and still strictly faster than RCCL on intra-node SDMA-capable links.
  AllGatherIntoTensor(int my_pe, int npes, size_t input_buffer_size, size_t output_buffer_size,
                      bool copy_output_to_user = true, bool auto_register = false);

  AllGatherIntoTensor(int my_pe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024,
                      bool copy_output_to_user = true, bool auto_register = false);

  ~AllGatherIntoTensor();

  AllGatherIntoTensor(const AllGatherIntoTensor&) = delete;
  AllGatherIntoTensor& operator=(const AllGatherIntoTensor&) = delete;

  // ----- ncclAllGather-equivalent entry point -----------------------------
  //
  // Returns true on success, false on failure.  Synchronization is the
  // caller's responsibility (record an event on `stream` and wait on it
  // from the consumer stream), matching NCCL's own contract.
  bool operator()(const void* sendbuff, void* recvbuff, size_t sendcount, DataType dtype,
                  hipStream_t stream = nullptr);

  // ----- Two-phase async path (mirrors AllgatherSdma) ---------------------
  bool start_async(const void* sendbuff, void* recvbuff, size_t sendcount, DataType dtype,
                   hipStream_t stream = nullptr);
  double wait_async(hipStream_t stream = nullptr);
  bool is_async_in_progress() const;
  void cancel_async();

  // ----- Direct-write registration (mirrors AllgatherSdma) ----------------
  //
  // Collective: every rank must call register/deregister with matching
  // pointers and sizes for the kernel to bypass the output transit buffer.
  void register_output_buffer(void* ptr, size_t size);
  void deregister_output_buffer(void* ptr);
  bool is_output_registered(void* ptr) const;

  void resetFlags();

  // ----- Introspection ----------------------------------------------------
  int my_pe() const { return my_pe_; }
  int npes() const { return npes_; }

  // Toggle the lazy auto-registration of recv buffers.  See class header
  // comment for the collective-correctness contract.
  void set_auto_register(bool enable) { auto_register_ = enable; }
  bool auto_register() const { return auto_register_; }

  // Exposed for diagnostics / advanced reuse.
  AllgatherSdma<uint32_t>* impl() { return impl_.get(); }

 private:
  // Registers ``recvbuff`` with mori's symmetric memory the first time
  // we see it, so the next allgather kernel can write directly to the
  // caller's tensor (zero-copy).  Idempotent and silent on failure (the
  // call falls back to the transit-buffer path in that case).
  void maybe_auto_register(void* recvbuff, size_t output_bytes);

  int my_pe_;
  int npes_;
  bool auto_register_;
  std::unique_ptr<AllgatherSdma<uint32_t>> impl_;
};

}  // namespace collective
}  // namespace mori

#endif  // MORI_COLLECTIVE_ALLGATHER_INTO_TENSOR_HPP
