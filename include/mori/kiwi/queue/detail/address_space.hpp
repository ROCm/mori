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
/**
 * @file address_space.hpp
 * @brief AMDGPU global address-space pointer alias for device handles
 *
 * Queue memory (host-mapped slots, progress words, device counters) is always
 * reachable through the global address space. A generic pointer that the
 * compiler cannot trace to a kernel argument (for example, a DeviceHandle
 * loaded from a device-resident context) lowers to flat_* instructions. A flat
 * access also counts against lgkmcnt, so a later LDS wait blocks on the
 * flat store's PCIe acknowledgement. Declaring the handle pointers in the
 * global address space makes every access a global_* instruction no matter
 * how the handle reaches the kernel.
 */

#pragma once

namespace mori::kiwi::queue::detail {

/// Pointer to T in the AMDGPU global address space (address space 1).
template <typename T>
using global_ptr = T __attribute__((address_space(1)))*;

/// Convert a generic pointer to global memory into a global_ptr.
template <typename T>
inline global_ptr<T> to_global(T* p) {
  return (global_ptr<T>)p;
}

}  // namespace mori::kiwi::queue::detail
