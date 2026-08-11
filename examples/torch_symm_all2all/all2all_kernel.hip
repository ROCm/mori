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

// all2all_kernel.hip -- one-shot all-to-all written against a flat symmetric window.
//
// Pure HIP. It does not use mori's shmem or cco; the only thing it needs from the
// symmetric memory backend is (flat_base, stride), after which every peer is addressed
// arithmetically:
//
//     recv slot of rank p, chunk from rank r  ==  flat_base + p*stride + r*chunk_bytes
//
// so the kernel takes two pointers and three integers rather than an N-entry pointer
// array, and the destination rank can be computed at run time.
//
// Push semantics: each rank writes its own chunk into every peer's receive window. That
// needs no reads from remote memory and no per-peer handshake -- just one barrier after
// the kernel, before anyone reads what landed.

#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

// Vectorised copy of one chunk. uint4 is 16 B, the widest single global access.
__device__ inline void CopyChunk(uint4* __restrict__ dst, const uint4* __restrict__ src,
                                 size_t n_vec) {
  for (size_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// One block per destination rank; grid-stride if the world is larger than the grid.
__global__ void All2AllPush(const uint4* __restrict__ send, char* __restrict__ flat_base,
                            size_t stride, size_t chunk_bytes, int rank, int world) {
  const size_t n_vec = chunk_bytes / sizeof(uint4);
  for (int peer = blockIdx.x; peer < world; peer += gridDim.x) {
    // Our chunk destined for `peer` goes into peer's window, in the slot reserved for us.
    auto* dst = reinterpret_cast<uint4*>(flat_base + static_cast<size_t>(peer) * stride +
                                         static_cast<size_t>(rank) * chunk_bytes);
    const uint4* src = send + static_cast<size_t>(peer) * n_vec;
    CopyChunk(dst, src, n_vec);
  }
}

}  // namespace

// send: local, world*chunk_bytes. flat_base/stride come from mori.allocator.flat_layout()
// on the rendezvous'd receive tensor. Caller barriers afterwards.
void all2all_push(const at::Tensor& send, int64_t flat_base, int64_t stride, int64_t chunk_bytes,
                  int64_t rank, int64_t world) {
  TORCH_CHECK(send.is_contiguous(), "send must be contiguous");
  TORCH_CHECK(chunk_bytes % sizeof(uint4) == 0, "chunk_bytes must be a multiple of 16, got ",
              chunk_bytes);
  TORCH_CHECK(send.nbytes() == static_cast<size_t>(world * chunk_bytes), "send holds ",
              send.nbytes(), " bytes, expected world*chunk_bytes = ", world * chunk_bytes);

  const int blocks = static_cast<int>(world);
  hipLaunchKernelGGL(All2AllPush, dim3(blocks), dim3(kThreads), 0, c10::hip::getCurrentHIPStream(),
                     reinterpret_cast<const uint4*>(send.data_ptr()),
                     reinterpret_cast<char*>(flat_base), static_cast<size_t>(stride),
                     static_cast<size_t>(chunk_bytes), static_cast<int>(rank),
                     static_cast<int>(world));
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "all2all_push launch failed: ", hipGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("all2all_push", &all2all_push,
        "One-shot all-to-all: push each chunk into every peer's symmetric window");
}
