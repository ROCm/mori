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

// torch_allocator.cpp -- back a torch.cuda.MemPool with the mori symmetric heap.
//
// torch.cuda.memory.CUDAPluggableAllocator dlopens a .so and binds two C symbols, so
// tensors allocated inside `torch.cuda.use_mem_pool(pool)` come from ShmemMalloc instead
// of the caching allocator. That matters for tensors an engine does not allocate by hand
// -- a KV cache, or a GEMM output -- which otherwise have to be copied into symmetric
// memory before they can be communicated.
//
// The expected signatures are fixed by torch:
//     void* alloc(size_t size, int device, hipStream_t stream)
//     void  free (void* ptr, size_t size, int device, hipStream_t stream)
//
// ShmemMalloc is collective and requires shmem to be initialised first, so every rank
// must enter and leave the mem-pool context in the same order with the same sizes. See
// python/mori/allocator for the wrapper and the usage notes.

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>

#include "mori/shmem/shmem.hpp"

namespace {

std::atomic<bool> g_warned_uninitialized{false};

// ShmemMalloc dereferences runtime state that does not exist before ShmemInit, so probe
// first and fail the allocation cleanly instead of taking the process down.
bool ShmemReady() { return mori::shmem::ShmemIsInitialized(); }

}  // namespace

extern "C" {

// Is the mori symmetric heap usable in this process right now? Mirrors the probe symbol
// convention other torch allocators use, so the Python side can degrade gracefully
// rather than discovering the problem inside a tensor constructor.
//   1  ready
//   0  shmem present but not initialised
int mori_allocator_probe() { return ShmemReady() ? 1 : 0; }

void* mori_allocator_malloc(size_t size, int device, hipStream_t stream) {
  (void)stream;  // ShmemMalloc is not stream-ordered

  if (!ShmemReady()) {
    if (!g_warned_uninitialized.exchange(true)) {
      fprintf(stderr,
              "[mori] torch allocator: ShmemMalloc requested before shmem was "
              "initialised. Call mori.shmem.shmem_torch_process_group_init() (or another "
              "shmem init) before entering torch.cuda.use_mem_pool().\n");
    }
    return nullptr;
  }

  int prev = -1;
  if (hipGetDevice(&prev) == hipSuccess && prev != device) {
    // torch may route an allocation for a device other than the current one
    if (hipSetDevice(device) != hipSuccess) {
      return nullptr;
    }
  }

  void* ptr = mori::shmem::ShmemMalloc(size);

  if (prev >= 0 && prev != device) {
    (void)hipSetDevice(prev);
  }
  return ptr;
}

void mori_allocator_free(void* ptr, size_t size, int device, hipStream_t stream) {
  (void)size;
  (void)device;
  (void)stream;
  if (ptr == nullptr) {
    return;
  }
  mori::shmem::ShmemFree(ptr);
}

}  // extern "C"
