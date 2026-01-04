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

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar upcast
template <typename T>
DINLINE float upcast_s(T val) {
  return static_cast<float>(val);
}

template <>
DINLINE float upcast_s<half>(half val) {
  return __half2float(val);
}

template <>
DINLINE float upcast_s<__hip_bfloat16>(__hip_bfloat16 val) {
  return __bfloat162float(val);
}

// scalar downcast
template <typename T>
DINLINE T downcast_s(float val) {
  return static_cast<T>(val);
}

template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

template <>
DINLINE __hip_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}

// scalar add (a value changed)
template <typename T>
DINLINE T& assign_add(T& a, T b) {
  return a += b;
}

template <>
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}

template <>
DINLINE __hip_bfloat16& assign_add(__hip_bfloat16& a, __hip_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}

// pack add (a value changed)
template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add<T>(a.data[i], b.data[i]);
  }
  return a;
}

// pack upcast
template <typename T, int N>
DINLINE array_t<float, N> upcast_v(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = upcast_s<T>(val.data[i]);
    }
    return out;
  }
}

// pack downcast
template <typename T, int N>
DINLINE array_t<T, N> downcast_v(array_t<float, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<T, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = downcast_s<T>(val.data[i]);
    }
    return out;
  }
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast_v(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; ++i) {
    packed_assign_add(tmp, upcast_v(ptrs[i][idx]));
  }
  return downcast_v<typename P::type, P::size>(tmp);
}
