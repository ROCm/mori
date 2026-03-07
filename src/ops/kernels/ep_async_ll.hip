// KernelType::AsyncLL — async low-latency send/recv.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopy)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendTransfer)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvCopy)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvTransfer)

WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncSendCopy, , false)
WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncSendTransfer, , false)
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSendCopy, bf16_fp8cast, hip_bfloat16, true))
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSendTransfer, bf16_fp8cast, hip_bfloat16, true))

WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncRecvCopy, , false)
WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncRecvTransfer, , false)
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecvCopy, bf16_fp8cast, hip_bfloat16, true))
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecvTransfer, bf16_fp8cast, hip_bfloat16, true))
