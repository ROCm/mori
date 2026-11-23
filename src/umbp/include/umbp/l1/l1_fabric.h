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

#include <memory>
#include <string>

#include "umbp/l1/l1_pool.h"

namespace mori::umbp::l1 {

// Export one contiguous view of fabric-exportable HBM. The token includes the
// whole VMM allocation size, view bounds and fabric-domain identity. Ordinary
// hipMalloc / default PyTorch allocations are rejected; allocate the KV tensors
// using mori.io.fabric_allocator.make_fabric_mem_pool() with torch.cuda.use_mem_pool;
// keep that pool alive until L1 detaches.
// Export takes no ownership and does not copy or initialize the payload.
bool ExportFabricMemory(int device, void* ptr, uint64_t bytes, ExportedMemory* exported,
                        std::string* error);
bool IsFabricHandle(const std::string& handle);
bool ValidateFabricRange(const std::string& handle, uint64_t offset, uint64_t bytes,
                         std::string* error);

// HBM-only, externally owned native L1 memory. Imports use HIP FABRIC VMM for
// both the local service and peer trays. Copies launch the existing mori-io
// fabric_copy.hip kernel on the destination GPU directly from imported source
// HBM into final destination slots. No host/device staging or hipMemcpy fallback.
// kernel_path must name an already compiled fabric_copy.hip .hsaco.
std::shared_ptr<MemoryApi> MakeFabricMemoryApi(const std::string& kernel_path);

}  // namespace mori::umbp::l1
