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

// all2all_kernel.hip -- one-shot all-to-all against a flat symmetric window.
//
// Pure HIP: no mori shmem or cco. All it needs from the backend is (flat_base, stride),
// after which peers are addressed arithmetically, so no N-entry pointer array is needed
// and the destination is computed at run time:
//
//     recv slot of rank p, chunk from rank r  ==  flat_base + p*stride + r*chunk_bytes
//
// Push semantics: each rank writes its own chunk into every peer's window. No remote
// reads, no per-peer handshake -- just one barrier before anyone reads what landed.
//
// Both addressing modes are built so they can be compared: the flat form above, and the
// N-entry pointer array that torch's buffer_ptrs_dev() forces when a backend maps peers
// at unrelated addresses. The kernels are otherwise identical.

#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

constexpr int kThreads = 256;

// Grid is 2-D over (peer, slice of chunk). One block per peer would idle all but
// `world_size` CUs and keep too few writes in flight to cover interconnect latency.
__global__ void All2AllPush(const uint4* __restrict__ send, char* __restrict__ flat_base,
                            size_t stride, size_t chunk_bytes, int rank_id, int world_size,
                            int blocks_per_peer) {
  const size_t n_vec = chunk_bytes / sizeof(uint4);
  const int peer = blockIdx.x / blocks_per_peer;
  const int sub = blockIdx.x % blocks_per_peer;
  if (peer >= world_size) return;

  // Our chunk destined for `peer` goes into peer's window, in the slot reserved for us.
  auto* dst = reinterpret_cast<uint4*>(flat_base + static_cast<size_t>(peer) * stride +
                                       static_cast<size_t>(rank_id) * chunk_bytes);
  const uint4* src = send + static_cast<size_t>(peer) * n_vec;

  const size_t span = static_cast<size_t>(blocks_per_peer) * blockDim.x;
  for (size_t i = static_cast<size_t>(sub) * blockDim.x + threadIdx.x; i < n_vec; i += span) {
    dst[i] = src[i];
  }
}

// Same work, addressed through an N-entry pointer array instead of arithmetic.
__global__ void All2AllPushPtrs(const uint4* __restrict__ send, void** __restrict__ peers,
                                size_t chunk_bytes, int rank_id, int world_size,
                                int blocks_per_peer) {
  const size_t n_vec = chunk_bytes / sizeof(uint4);
  const int peer = blockIdx.x / blocks_per_peer;
  const int sub = blockIdx.x % blocks_per_peer;
  if (peer >= world_size) return;

  auto* dst = reinterpret_cast<uint4*>(static_cast<char*>(peers[peer]) +
                                       static_cast<size_t>(rank_id) * chunk_bytes);
  const uint4* src = send + static_cast<size_t>(peer) * n_vec;

  const size_t span = static_cast<size_t>(blocks_per_peer) * blockDim.x;
  for (size_t i = static_cast<size_t>(sub) * blockDim.x + threadIdx.x; i < n_vec; i += span) {
    dst[i] = src[i];
  }
}

// Two blocks per CU, split across peers, capped by how much data there is to slice.
int BlocksPerPeer(int64_t chunk_bytes, int64_t world_size) {
  static const int cus = [] {
    int v = 64, dev = 0;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev);
    return v;
  }();
  const int64_t n_vec = chunk_bytes / static_cast<int64_t>(sizeof(uint4));
  int64_t bpp = std::max<int64_t>(1, cus * 2 / std::max<int64_t>(1, world_size));
  return static_cast<int>(std::min<int64_t>(bpp, std::max<int64_t>(1, n_vec / kThreads)));
}

void CheckArgs(const at::Tensor& send, int64_t chunk_bytes, int64_t world_size) {
  TORCH_CHECK(send.is_contiguous(), "send must be contiguous");
  TORCH_CHECK(chunk_bytes % sizeof(uint4) == 0, "chunk_bytes must be a multiple of 16, got ",
              chunk_bytes);
  TORCH_CHECK(send.nbytes() == static_cast<size_t>(world_size * chunk_bytes), "send holds ",
              send.nbytes(),
              " bytes, expected world_size*chunk_bytes = ", world_size * chunk_bytes);
}

}  // namespace

// send is local, world_size*chunk_bytes. flat_base/stride come from flat_layout() on the
// rendezvous'd receive tensor. Caller barriers afterwards.
void all2all_push(const at::Tensor& send, int64_t flat_base, int64_t stride, int64_t chunk_bytes,
                  int64_t rank_id, int64_t world_size) {
  CheckArgs(send, chunk_bytes, world_size);
  const int bpp = BlocksPerPeer(chunk_bytes, world_size);
  hipLaunchKernelGGL(
      All2AllPush, dim3(world_size * bpp), dim3(kThreads), 0, c10::hip::getCurrentHIPStream(),
      reinterpret_cast<const uint4*>(send.data_ptr()), reinterpret_cast<char*>(flat_base),
      static_cast<size_t>(stride), static_cast<size_t>(chunk_bytes), static_cast<int>(rank_id),
      static_cast<int>(world_size), bpp);
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "all2all_push launch failed: ", hipGetErrorString(err));
}

// peers_dev is hdl.buffer_ptrs_dev: a device array of world_size peer base pointers.
void all2all_push_ptrs(const at::Tensor& send, int64_t peers_dev, int64_t chunk_bytes,
                       int64_t rank_id, int64_t world_size) {
  CheckArgs(send, chunk_bytes, world_size);
  const int bpp = BlocksPerPeer(chunk_bytes, world_size);
  hipLaunchKernelGGL(All2AllPushPtrs, dim3(world_size * bpp), dim3(kThreads), 0,
                     c10::hip::getCurrentHIPStream(),
                     reinterpret_cast<const uint4*>(send.data_ptr()),
                     reinterpret_cast<void**>(peers_dev), static_cast<size_t>(chunk_bytes),
                     static_cast<int>(rank_id), static_cast<int>(world_size), bpp);
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "all2all_push_ptrs launch failed: ", hipGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("all2all_push", &all2all_push, "One-shot all-to-all addressed as flat_base + peer*stride");
  m.def("all2all_push_ptrs", &all2all_push_ptrs,
        "Same, addressed through an N-entry peer pointer array");
}
