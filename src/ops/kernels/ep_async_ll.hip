// KernelType::AsyncLL — async low-latency send/recv.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopy)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopySlotAssign)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopyMultiBlock)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendTransfer)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvCopyMultiBlock)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvTransfer)

WRAP_ALL_TYPES(EpCombineLowLatencyAsyncSendCopy)
WRAP_ALL_TYPES(EpCombineLowLatencyAsyncSendTransfer)

WRAP_ALL_TYPES(EpCombineLowLatencyAsyncRecvCopy)
WRAP_ALL_TYPES(EpCombineLowLatencyAsyncRecvTransfer)

// Blockwise-FP8 combine variants (bf16 in/out, fp8 on the wire): UseFp8BlockwiseQuant=true.
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSendCopy, bf16_fp8_blockwise, hip_bfloat16, true))
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSendTransfer, bf16_fp8_blockwise, hip_bfloat16,
                       true))
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecvCopy, bf16_fp8_blockwise, hip_bfloat16, true))
