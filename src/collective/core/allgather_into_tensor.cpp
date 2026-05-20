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

#include "mori/collective/allgather/allgather_into_tensor.hpp"

#include <cstdio>
#include <stdexcept>

namespace mori {
namespace collective {

// ---------------------------------------------------------------------------
// DataType helpers
// ---------------------------------------------------------------------------
size_t SizeOf(DataType dtype) {
  switch (dtype) {
    case DataType::kInt8:
    case DataType::kUint8:
      return 1;
    case DataType::kInt16:
    case DataType::kUint16:
    case DataType::kFloat16:
    case DataType::kBFloat16:
      return 2;
    case DataType::kInt32:
    case DataType::kUint32:
    case DataType::kFloat32:
      return 4;
    case DataType::kInt64:
    case DataType::kUint64:
    case DataType::kFloat64:
      return 8;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

// Convert (sendcount, dtype) into a uint32-element count, throwing if the
// total byte length isn't a multiple of sizeof(uint32_t).  The underlying
// SDMA kernel walks the buffer as packed uint32 lanes; non-aligned byte
// lengths would silently corrupt one or more PEs' shards.
size_t bytes_to_u32_count(size_t sendcount, DataType dtype) {
  size_t bytes = sendcount * SizeOf(dtype);
  if (bytes == 0) return 0;
  if ((bytes & 0x3u) != 0) {
    throw std::runtime_error(
        "AllGatherIntoTensor: per-rank byte length must be a multiple of 4 "
        "(SDMA kernel walks the buffer as uint32 lanes)");
  }
  return bytes / sizeof(uint32_t);
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructors / destructor
// ---------------------------------------------------------------------------
AllGatherIntoTensor::AllGatherIntoTensor(int my_pe, int npes, size_t input_buffer_size,
                                         size_t output_buffer_size, bool copy_output_to_user,
                                         bool auto_register)
    : my_pe_(my_pe),
      npes_(npes),
      auto_register_(auto_register),
      impl_(std::make_unique<AllgatherSdma<uint32_t>>(my_pe, npes, input_buffer_size,
                                                      output_buffer_size, copy_output_to_user)) {}

AllGatherIntoTensor::AllGatherIntoTensor(int my_pe, int npes, size_t transit_buffer_size,
                                         bool copy_output_to_user, bool auto_register)
    : my_pe_(my_pe),
      npes_(npes),
      auto_register_(auto_register),
      impl_(std::make_unique<AllgatherSdma<uint32_t>>(my_pe, npes, transit_buffer_size,
                                                      copy_output_to_user)) {}

AllGatherIntoTensor::~AllGatherIntoTensor() = default;

// ---------------------------------------------------------------------------
// maybe_auto_register
// ---------------------------------------------------------------------------
//
// Idempotent best-effort registration of ``recvbuff`` with mori's symmetric
// memory.  Registration is collective: every rank must call
// ShmemSymmetricRegister with matching size at the same time.  This method
// is invoked from inside ``operator()`` / ``start_async``, the natural
// allgather barrier point — all ranks reach it together, so the collective
// is well-formed.
//
// Discipline expected from the caller:
//   * The *sequence* of recv buffers passed across allgather calls must
//     match across ranks (their VAs differ, their order does not).  ZeRO-3
//     parameter prefetching satisfies this naturally.
//   * Recv buffers must remain live until shmem_finalize().  We never
//     deregister, so freeing a buffer and reallocating into the same VA
//     would silently use a stale IPC mapping.  ZeRO-3's prefetch pool
//     reuses the same buffers across steps so this is fine in practice.
//
// On registration failure we log once and leave the buffer unregistered;
// the underlying AllgatherSdma falls back to its transit-buffer path,
// which is still correct, just one hipMemcpyAsync slower.
void AllGatherIntoTensor::maybe_auto_register(void* recvbuff, size_t output_bytes) {
  if (!auto_register_ || recvbuff == nullptr) return;
  if (impl_->is_output_registered(recvbuff)) return;
  try {
    impl_->register_output_buffer(recvbuff, output_bytes);
  } catch (const std::exception& e) {
    static thread_local bool warned = false;
    if (!warned) {
      fprintf(stderr,
              "[mori] AllGatherIntoTensor PE %d: auto-register failed (%s); "
              "falling back to transit-buffer path for this and future calls\n",
              my_pe_, e.what());
      warned = true;
    }
    auto_register_ = false;
  }
}

// ---------------------------------------------------------------------------
// ncclAllGather-style synchronous entry point
// ---------------------------------------------------------------------------
bool AllGatherIntoTensor::operator()(const void* sendbuff, void* recvbuff, size_t sendcount,
                                     DataType dtype, hipStream_t stream) {
  size_t u32_count = bytes_to_u32_count(sendcount, dtype);
  size_t output_bytes = sendcount * SizeOf(dtype) * static_cast<size_t>(npes_);
  maybe_auto_register(recvbuff, output_bytes);
  // The const_cast is safe: AllgatherSdma::operator() doesn't write through
  // the input pointer; we drop const only to reuse the existing entry point
  // (NCCL has the same situation: ncclAllGather declares sendbuff const but
  // forwards to internal kernels that accept non-const pointers).
  auto* in = reinterpret_cast<uint32_t*>(const_cast<void*>(sendbuff));
  auto* out = reinterpret_cast<uint32_t*>(recvbuff);
  return (*impl_)(in, out, u32_count, stream);
}

// ---------------------------------------------------------------------------
// Async two-phase API
// ---------------------------------------------------------------------------
bool AllGatherIntoTensor::start_async(const void* sendbuff, void* recvbuff, size_t sendcount,
                                      DataType dtype, hipStream_t stream) {
  size_t u32_count = bytes_to_u32_count(sendcount, dtype);
  size_t output_bytes = sendcount * SizeOf(dtype) * static_cast<size_t>(npes_);
  maybe_auto_register(recvbuff, output_bytes);
  auto* in = reinterpret_cast<uint32_t*>(const_cast<void*>(sendbuff));
  auto* out = reinterpret_cast<uint32_t*>(recvbuff);
  return impl_->start_async(in, out, u32_count, stream);
}

double AllGatherIntoTensor::wait_async(hipStream_t stream) { return impl_->wait_async(stream); }

bool AllGatherIntoTensor::is_async_in_progress() const { return impl_->is_async_in_progress(); }

void AllGatherIntoTensor::cancel_async() { impl_->cancel_async(); }

// ---------------------------------------------------------------------------
// External output buffer registration
// ---------------------------------------------------------------------------
void AllGatherIntoTensor::register_output_buffer(void* ptr, size_t size) {
  impl_->register_output_buffer(ptr, size);
}

void AllGatherIntoTensor::deregister_output_buffer(void* ptr) {
  impl_->deregister_output_buffer(ptr);
}

bool AllGatherIntoTensor::is_output_registered(void* ptr) const {
  return impl_->is_output_registered(ptr);
}

void AllGatherIntoTensor::resetFlags() { impl_->resetFlags(); }

}  // namespace collective
}  // namespace mori
