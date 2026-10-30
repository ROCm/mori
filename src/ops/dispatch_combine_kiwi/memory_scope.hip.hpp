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

#include <cstdint>

namespace mori::kiwi::ep {

using ep_u32x4_t = uint32_t __attribute__((ext_vector_type(4)));

template <typename T>
struct ep_system_scope_width_supported {
  static constexpr bool value =
      sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16;
};

// EP communication data can be consumed by the NIC or by peer GPUs. Use
// explicit system-scope global memory instructions for publication and
// polling. This intentionally requests sc0/sc1 and does not set AMD's NT bit.
template <typename PtrT, typename T>
__device__ inline void ep_store_system_scope(PtrT dst, T value) {
  const unsigned long long addr = reinterpret_cast<unsigned long long>(dst);
  if constexpr (sizeof(T) == 1) {
    const unsigned int raw = static_cast<unsigned int>(*reinterpret_cast<const uint8_t*>(&value));
    asm volatile("global_store_byte %0, %1, off sc0 sc1" ::"v"(addr), "v"(raw) : "memory");
  } else if constexpr (sizeof(T) == 2) {
    const unsigned int raw = static_cast<unsigned int>(*reinterpret_cast<const uint16_t*>(&value));
    asm volatile("global_store_short %0, %1, off sc0 sc1" ::"v"(addr), "v"(raw) : "memory");
  } else if constexpr (sizeof(T) == 4) {
    const unsigned int raw = *reinterpret_cast<const unsigned int*>(&value);
    asm volatile("global_store_dword %0, %1, off sc0 sc1" ::"v"(addr), "v"(raw) : "memory");
  } else if constexpr (sizeof(T) == 8) {
    const unsigned long long raw = *reinterpret_cast<const unsigned long long*>(&value);
    asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1" ::"v"(addr), "v"(raw) : "memory");
  } else if constexpr (sizeof(T) == 16) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&value);
    const ep_u32x4_t raw = {words[0], words[1], words[2], words[3]};
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(addr), "v"(raw) : "memory");
  } else {
    static_assert(ep_system_scope_width_supported<T>::value,
                  "unsupported ep_store_system_scope width");
  }
}

template <typename T, typename PtrT>
__device__ inline T ep_load_system_scope(PtrT src) {
  const unsigned long long addr = reinterpret_cast<unsigned long long>(src);
  T out;
  if constexpr (sizeof(T) == 1) {
    unsigned int raw;
    asm volatile(
        "global_load_ubyte %0, %1, off sc0 sc1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(raw)
        : "v"(addr)
        : "memory");
    *reinterpret_cast<uint8_t*>(&out) = static_cast<uint8_t>(raw);
  } else if constexpr (sizeof(T) == 2) {
    unsigned int raw;
    asm volatile(
        "global_load_ushort %0, %1, off sc0 sc1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(raw)
        : "v"(addr)
        : "memory");
    *reinterpret_cast<uint16_t*>(&out) = static_cast<uint16_t>(raw);
  } else if constexpr (sizeof(T) == 4) {
    unsigned int raw;
    asm volatile(
        "global_load_dword %0, %1, off sc0 sc1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(raw)
        : "v"(addr)
        : "memory");
    *reinterpret_cast<unsigned int*>(&out) = raw;
  } else if constexpr (sizeof(T) == 8) {
    unsigned long long raw;
    asm volatile(
        "global_load_dwordx2 %0, %1, off sc0 sc1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(raw)
        : "v"(addr)
        : "memory");
    *reinterpret_cast<unsigned long long*>(&out) = raw;
  } else if constexpr (sizeof(T) == 16) {
    ep_u32x4_t raw;
    asm volatile(
        "global_load_dwordx4 %0, %1, off sc0 sc1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(raw)
        : "v"(addr)
        : "memory");
    uint32_t* words = reinterpret_cast<uint32_t*>(&out);
    words[0] = raw[0];
    words[1] = raw[1];
    words[2] = raw[2];
    words[3] = raw[3];
  } else {
    static_assert(ep_system_scope_width_supported<T>::value,
                  "unsupported ep_load_system_scope width");
  }
  return out;
}

template <bool SystemScopeStore, typename T>
__device__ inline void ep_store(T* dst, T value) {
  if constexpr (SystemScopeStore) {
    ep_store_system_scope(dst, value);
  } else {
    *dst = value;
  }
}

template <bool SystemScopeLoad, typename T>
__device__ inline T ep_load(const T* src) {
  if constexpr (SystemScopeLoad) {
    return ep_load_system_scope<T>(src);
  } else {
    return *src;
  }
}

}  // namespace mori::kiwi::ep
