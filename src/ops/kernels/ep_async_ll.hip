// KernelType::AsyncLL — async low-latency send/recv.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopy)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopySlotAssign)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopyMultiBlock)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendTransfer)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvCopyMultiBlock)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvTransfer)

// AsyncLL combine kernels are templated on <T, UseFp8DirectCast, UseFp8BlockwiseQuant>.
// Use WRAP_BOOL2 to instantiate. Default (no quant) wraps both booleans as false.
#define WRAP_ALL_TYPES_ASYNC_COMBINE(kernel)                                              \
  WRAP_BOOL2(kernel, bf16, hip_bfloat16, false, false)                                    \
  MORI_FP8_FNUZ(WRAP_BOOL2(kernel, fp8_fnuz, __hip_fp8_e4m3_fnuz, false, false))          \
  MORI_FP8_OCP(WRAP_BOOL2(kernel, fp8_ocp, __hip_fp8_e4m3, false, false))                 \
  MORI_FP4(WRAP_BOOL2(kernel, fp4, mori::mori_fp4x2_e2m1, false, false))                  \
  WRAP_BOOL2(kernel, f32, float, false, false)

WRAP_ALL_TYPES_ASYNC_COMBINE(EpCombineLowLatencyAsyncSendCopy)
WRAP_ALL_TYPES_ASYNC_COMBINE(EpCombineLowLatencyAsyncSendTransfer)
MORI_FP8_ANY(WRAP_BOOL2(EpCombineLowLatencyAsyncSendCopy, bf16_fp8cast, hip_bfloat16, true, false))
MORI_FP8_ANY(
    WRAP_BOOL2(EpCombineLowLatencyAsyncSendTransfer, bf16_fp8cast, hip_bfloat16, true, false))
MORI_FP8_ANY(WRAP_BOOL2(EpCombineLowLatencyAsyncSendCopy, bf16_fp8bwq, hip_bfloat16, false, true))
MORI_FP8_ANY(
    WRAP_BOOL2(EpCombineLowLatencyAsyncSendTransfer, bf16_fp8bwq, hip_bfloat16, false, true))

WRAP_ALL_TYPES_ASYNC_COMBINE(EpCombineLowLatencyAsyncRecvCopy)
WRAP_ALL_TYPES_ASYNC_COMBINE(EpCombineLowLatencyAsyncRecvTransfer)
MORI_FP8_ANY(WRAP_BOOL2(EpCombineLowLatencyAsyncRecvCopy, bf16_fp8cast, hip_bfloat16, true, false))
MORI_FP8_ANY(
    WRAP_BOOL2(EpCombineLowLatencyAsyncRecvTransfer, bf16_fp8cast, hip_bfloat16, true, false))
MORI_FP8_ANY(WRAP_BOOL2(EpCombineLowLatencyAsyncRecvCopy, bf16_fp8bwq, hip_bfloat16, false, true))
MORI_FP8_ANY(
    WRAP_BOOL2(EpCombineLowLatencyAsyncRecvTransfer, bf16_fp8bwq, hip_bfloat16, false, true))
