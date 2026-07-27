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

// C-ABI allocator shims matching PyTorch's CUDAPluggableAllocator hooks
//   void* alloc(size_t size, int device, hipStream_t stream)
//   void  free (void* ptr,  size_t size, int device, hipStream_t stream)
//
// They back torch tensors with fabric-exportable VMM memory (see FabricMalloc),
// so a tensor allocated under this allocator (e.g. via torch.cuda.MemPool) can be
// registered and transferred by the mori-io FabricBackend across a UALink
// super-node. Plain torch/hipMalloc tensors are NOT fabric-exportable, so this
// allocator is the bridge that lets frameworks (e.g. SGLang) route their KV pool
// over the fabric backend.
//
// The stream argument is unused: VMM allocations are not stream-ordered and, once
// hipMemSetAccess grants the owning device read/write, are usable on any stream.

#include <hip/hip_runtime_api.h>

#include <cstddef>

#include "src/io/fabric/backend_impl.hpp"

extern "C" {

void* mori_io_fabric_malloc(size_t size, int device, hipStream_t /*stream*/) {
  return mori::io::FabricMalloc(size, device);
}

void mori_io_fabric_free(void* ptr, size_t /*size*/, int /*device*/, hipStream_t /*stream*/) {
  mori::io::FabricFree(ptr);
}

}  // extern "C"
