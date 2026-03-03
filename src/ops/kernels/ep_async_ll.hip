// KernelType::AsyncLL — async low-latency send/recv.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSend)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecv)

WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncSend, , false)
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSend, bf16_fp8cast, hip_bfloat16, true))

WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncRecv, , false)
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecv, bf16_fp8cast, hip_bfloat16, true))
