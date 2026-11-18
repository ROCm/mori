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

#include <hip/hip_runtime_api.h>
#include <sys/types.h>

#include <cstddef>

// HIP device memory for buffers that RDMA NICs pin, such as a KV cache, kept in
// amdgpu GEM BOs instead of KFD BOs (ROCm/mori#721, known-issues Issue 4).
//
// amdgpu-dkms counts a hipMalloc buffer that a NIC pins twice: once as used
// (kfd.vram_used) and once as pinned (vram_pin_size). On 31.10 - 31.60 the KFD
// allocation limit never credits the second count back, whichever way the
// buffer was registered; on 30.x it does for peer-memory MRs only. Once the
// registered memory exceeds the VRAM left after allocating it, every later
// device allocation of the process fails.
//
// Each buffer here is an amdgpu GEM BO created on the GPU's render node and
// imported into HIP through its dma-buf. KFD does not count an imported BO as
// used, so once pinned it is counted once, on every driver. A one-page dma-buf
// MR pins the BO at allocation, which keeps it from being evicted and makes
// hipMemGetInfo cover it from the start.
//
// These buffers must be registered through the BO's own dma-buf (mori does so
// for every buffer HipGemLookup finds) and never through ibv_reg_mr: the amdgpu
// peer-memory client dereferences the NULL kfd_bo of a GEM BO. HIP has no IPC
// handle for them, so they cannot go through XGMI or IPC transports either.
//
// Freeing a buffer unpins it, so KFD's counts drop right away. Whether the VRAM
// itself comes back before the process exits depends on HIP and the driver: ROCm
// 7.2.4 on amdgpu 31.40 keeps imported memory mapped until then, on 30.x and with
// ROCm 10.0 it comes back on free. Allocate long-lived buffers (a KV pool) with it.
//
// The C functions below are a torch.cuda.memory.CUDAPluggableAllocator backend
// (mori.io.HipGemAllocator). MORI_HIP_GEM_PIN_DEVICE names the RDMA device used
// for the one-page pin; by default it is the first of the devices MORI_RDMA_DEVICES
// selects (all if unset) that accepts it.

namespace mori {
namespace application {

struct HipGemBuffer {
  void* base{nullptr};
  size_t size{0};
  int device{-1};
  int dmabufFd{-1};  // a new descriptor of the BO's dma-buf if requested; caller closes
};

// True when addr lies in a buffer of the HIP GEM allocator. With wantFd, out->dmabufFd
// receives a new descriptor of the buffer's dma-buf.
bool HipGemLookup(const void* addr, HipGemBuffer* out, bool wantFd);

}  // namespace application
}  // namespace mori

extern "C" {
void* mori_hip_gem_malloc(ssize_t size, int device, hipStream_t stream);
void mori_hip_gem_free(void* ptr, ssize_t size, int device, hipStream_t stream);
// 0 when addr lies in a buffer of the allocator; dmabuf_fd, if not NULL, receives a new
// descriptor of its dma-buf.
int mori_hip_gem_lookup(const void* addr, void** base, size_t* size, int* dmabuf_fd, int* device);
}
