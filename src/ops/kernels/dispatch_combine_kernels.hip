// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Standalone device kernels for dispatch/combine operations.
// Compiled with: hipcc --genco --offload-arch=<arch>
// Loaded at runtime via hipModuleLoad / hipModuleLaunchKernel.
//
// Each template instantiation is wrapped with extern "C" to provide
// a predictable symbol name for hipModuleGetFunction.

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

// Include .cpp files for _body device function definitions.
// The MORI_JIT_GENCO guard skips template instantiation sections.
#define MORI_JIT_GENCO
#include "src/ops/dispatch_combine/internode_v1.cpp"
#include "src/ops/dispatch_combine/low_latency_async.cpp"
#undef MORI_JIT_GENCO

// Provide globalGpuStates definition (initialized at runtime via shmem_module_init)
namespace mori {
namespace shmem {
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;
}
}

using namespace mori::moe;

// ---------------------------------------------------------------------------
// Conditional FP8 macros (same as internode_v1.cpp)
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

// ---------------------------------------------------------------------------
// Wrapper macros
// ---------------------------------------------------------------------------

// Kernel_body<T>(args)
#define WRAP(kernel, suffix, T) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T>(args); \
  }

#define WRAP_ALL_TYPES(kernel) \
  WRAP(kernel, bf16, hip_bfloat16) \
  MORI_FP8_FNUZ(WRAP(kernel, fp8_fnuz, __hip_fp8_e4m3_fnuz)) \
  MORI_FP8_OCP(WRAP(kernel, fp8_ocp, __hip_fp8_e4m3)) \
  WRAP(kernel, fp4, mori::mori_fp4x2_e2m1) \
  WRAP(kernel, f32, float)

// Kernel_body<T, BoolParam>(args)
#define WRAP_BOOL(kernel, suffix, T, B) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T, B>(args); \
  }

#define WRAP_ALL_TYPES_BOOL(kernel, bool_suffix, B) \
  WRAP_BOOL(kernel, bf16##bool_suffix, hip_bfloat16, B) \
  MORI_FP8_FNUZ(WRAP_BOOL(kernel, fp8_fnuz##bool_suffix, __hip_fp8_e4m3_fnuz, B)) \
  MORI_FP8_OCP(WRAP_BOOL(kernel, fp8_ocp##bool_suffix, __hip_fp8_e4m3, B)) \
  WRAP_BOOL(kernel, fp4##bool_suffix, mori::mori_fp4x2_e2m1, B) \
  WRAP_BOOL(kernel, f32##bool_suffix, float, B)

// Kernel_body<T, B1, B2>(args) — 3 template params
#define WRAP_BOOL2(kernel, suffix, T, B1, B2) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T, B1, B2>(args); \
  }

// Kernel_body<T, B1, B2, B3>(args) — 4 template params
#define WRAP_BOOL3(kernel, suffix, T, B1, B2, B3) \
  extern "C" __global__ void kernel##_##suffix(EpDispatchCombineArgs<T> args) { \
    kernel##_body<T, B1, B2, B3>(args); \
  }

#define WRAP_ALL_TYPES_BOOL3(kernel, bool_suffix, B1, B2, B3) \
  WRAP_BOOL3(kernel, bf16##bool_suffix, hip_bfloat16, B1, B2, B3) \
  MORI_FP8_FNUZ(WRAP_BOOL3(kernel, fp8_fnuz##bool_suffix, __hip_fp8_e4m3_fnuz, B1, B2, B3)) \
  MORI_FP8_OCP(WRAP_BOOL3(kernel, fp8_ocp##bool_suffix, __hip_fp8_e4m3, B1, B2, B3)) \
  WRAP_BOOL3(kernel, fp4##bool_suffix, mori::mori_fp4x2_e2m1, B1, B2, B3) \
  WRAP_BOOL3(kernel, f32##bool_suffix, float, B1, B2, B3)

// ---------------------------------------------------------------------------
// Dispatch kernels — simple <T> template
// ---------------------------------------------------------------------------
WRAP_ALL_TYPES(EpDispatchInterNodeKernel)
WRAP_ALL_TYPES(EpDispatchInterNodeV1Kernel)
WRAP_ALL_TYPES(EpDispatchCopyToStaging)

// EpDispatchIntraNodeKernel<T, EnableStdMoE=false>
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, , false)
// EpDispatchIntraNodeKernel<T, EnableStdMoE=true>
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, _stdmoe, true)

// EpDispatchInterNodeV1KernelLowLatency<T, EnableStdMoE=false>
WRAP_ALL_TYPES_BOOL(EpDispatchInterNodeV1KernelLowLatency, , false)
// EpDispatchInterNodeV1KernelLowLatency<T, EnableStdMoE=true>
WRAP_ALL_TYPES_BOOL(EpDispatchInterNodeV1KernelLowLatency, _stdmoe, true)

// Low-latency async dispatch
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSend)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecv)

// ---------------------------------------------------------------------------
// Combine kernels — simple <T> template
// ---------------------------------------------------------------------------
WRAP_ALL_TYPES(EpCombineInterNodeKernel)
WRAP_ALL_TYPES(EpCombineInterNodeV1Kernel)
WRAP_ALL_TYPES(EpCombineSync)
WRAP_ALL_TYPES(EpCombineSyncBarrier)
WRAP_ALL_TYPES(EpCombineAll)

// EpCombineInterNodeV1KernelLowLatency<T, EnableStdMoE=false>
WRAP_ALL_TYPES_BOOL(EpCombineInterNodeV1KernelLowLatency, , false)
// EpCombineInterNodeV1KernelLowLatency<T, EnableStdMoE=true>
WRAP_ALL_TYPES_BOOL(EpCombineInterNodeV1KernelLowLatency, _stdmoe, true)

// EpCombineIntraNodeKernel<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast>
// Variants actually used in dispatch_combine.cpp:
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _nop2p, false, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
// Special: bf16 with Fp8DirectCast
WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)

// Low-latency async combine
// EpCombineLowLatencyAsyncSend<T, UseFp8DirectCast=false>
WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncSend, , false)
// EpCombineLowLatencyAsyncSend<hip_bfloat16, true> — FP8 direct cast
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSend, bf16_fp8cast, hip_bfloat16, true))

// EpCombineLowLatencyAsyncRecv<T, UseFp8DirectCast=false>
WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncRecv, , false)
// EpCombineLowLatencyAsyncRecv<hip_bfloat16, true>
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecv, bf16_fp8cast, hip_bfloat16, true))

// ---------------------------------------------------------------------------
// Convert kernels (non-templated or different args type)
// ---------------------------------------------------------------------------
extern "C" __global__ void mori_ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}

#define WRAP_CONVERT_COMBINE(suffix, T) \
  extern "C" __global__ void ConvertCombineInputKernel_##suffix(ConvertCombineInputArgs args) { \
    ConvertCombineInputDevice<T, true>(args); \
  }

WRAP_CONVERT_COMBINE(f32, float)
WRAP_CONVERT_COMBINE(bf16, hip_bfloat16)
MORI_FP8_OCP(WRAP_CONVERT_COMBINE(fp8_ocp, __hip_fp8_e4m3))
MORI_FP8_FNUZ(WRAP_CONVERT_COMBINE(fp8_fnuz, __hip_fp8_e4m3_fnuz))
