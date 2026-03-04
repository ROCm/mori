// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Common includes, shims, and wrapper macros shared by dispatch/combine kernel files.
// Not compiled directly — included by ep_*.hip kernel files.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#include "src/ops/dispatch_combine/internode.hpp"
#include "src/ops/dispatch_combine/intranode.hpp"

#define MORI_JIT_GENCO
#include "src/ops/dispatch_combine/internode_v1.cpp"
#include "src/ops/dispatch_combine/low_latency_async.cpp"
#undef MORI_JIT_GENCO

// globalGpuStates is defined per-file via MORI_DEFINE_GPU_STATES macro.
// Each .hsaco needs its own copy, initialized via ShmemModuleInit.
#define MORI_DEFINE_GPU_STATES \
  namespace mori { namespace shmem { \
  __device__ __attribute__((visibility("default"))) GpuStates globalGpuStates; \
  } }

using namespace mori::moe;

// ---------------------------------------------------------------------------
// Conditional FP8 macros
// ---------------------------------------------------------------------------
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
#define MORI_FP8_FNUZ(...) __VA_ARGS__
#else
#define MORI_FP8_FNUZ(...)
#endif

#ifdef MORI_FP8_TYPE_OCP_ENABLED
#define MORI_FP8_OCP(...) __VA_ARGS__
#else
#define MORI_FP8_OCP(...)
#endif

#if defined(MORI_FP8_TYPE_FNUZ_ENABLED) || defined(MORI_FP8_TYPE_OCP_ENABLED)
#define MORI_FP8_ANY(...) __VA_ARGS__
#else
#define MORI_FP8_ANY(...)
#endif

#ifdef MORI_HAS_OCP_FP
#define MORI_FP4(...) __VA_ARGS__
#else
#define MORI_FP4(...)
#endif

// ---------------------------------------------------------------------------
// Wrapper macros
// ---------------------------------------------------------------------------
#define WRAP(kernel, suffix, T) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T>(args); \
  }

#define WRAP_ALL_TYPES(kernel) \
  WRAP(kernel, bf16, hip_bfloat16) \
  MORI_FP8_FNUZ(WRAP(kernel, fp8_fnuz, __hip_fp8_e4m3_fnuz)) \
  MORI_FP8_OCP(WRAP(kernel, fp8_ocp, __hip_fp8_e4m3)) \
  MORI_FP4(WRAP(kernel, fp4, mori::mori_fp4x2_e2m1)) \
  WRAP(kernel, f32, float)

#define WRAP_BOOL(kernel, suffix, T, B) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T, B>(args); \
  }

#define WRAP_ALL_TYPES_BOOL(kernel, bool_suffix, B) \
  WRAP_BOOL(kernel, bf16##bool_suffix, hip_bfloat16, B) \
  MORI_FP8_FNUZ(WRAP_BOOL(kernel, fp8_fnuz##bool_suffix, __hip_fp8_e4m3_fnuz, B)) \
  MORI_FP8_OCP(WRAP_BOOL(kernel, fp8_ocp##bool_suffix, __hip_fp8_e4m3, B)) \
  MORI_FP4(WRAP_BOOL(kernel, fp4##bool_suffix, mori::mori_fp4x2_e2m1, B)) \
  WRAP_BOOL(kernel, f32##bool_suffix, float, B)

#define WRAP_BOOL3(kernel, suffix, T, B1, B2, B3) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T, B1, B2, B3>(args); \
  }

#define WRAP_ALL_TYPES_BOOL3(kernel, bool_suffix, B1, B2, B3) \
  WRAP_BOOL3(kernel, bf16##bool_suffix, hip_bfloat16, B1, B2, B3) \
  MORI_FP8_FNUZ(WRAP_BOOL3(kernel, fp8_fnuz##bool_suffix, __hip_fp8_e4m3_fnuz, B1, B2, B3)) \
  MORI_FP8_OCP(WRAP_BOOL3(kernel, fp8_ocp##bool_suffix, __hip_fp8_e4m3, B1, B2, B3)) \
  MORI_FP4(WRAP_BOOL3(kernel, fp4##bool_suffix, mori::mori_fp4x2_e2m1, B1, B2, B3)) \
  WRAP_BOOL3(kernel, f32##bool_suffix, float, B1, B2, B3)
