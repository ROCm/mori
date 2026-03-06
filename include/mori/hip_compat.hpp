// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

// HIP compiler compatibility macros.
// When compiled with a standard C++ compiler (clang/g++), HIP qualifiers
// become no-ops so that headers with __device__/__host__ annotations can
// be included without requiring hipcc.

#if defined(__HIPCC__) || defined(__CUDACC__)
// HIP/CUDA compiler — qualifiers already defined
#else
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif
