// KernelType::AsyncLL — async low-latency send/recv.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopy)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopySlotAssign)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendCopyMultiBlock)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncSendTransfer)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvCopyMultiBlock)
WRAP_ALL_TYPES(EpDispatchLowLatencyAsyncRecvTransfer)

WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncSendCopy, , false)
WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncSendTransfer, , false)
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSendCopy, bf16_fp8cast, hip_bfloat16, true))
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncSendTransfer, bf16_fp8cast, hip_bfloat16, true))

WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncRecvCopy, , false)
WRAP_ALL_TYPES_BOOL(EpCombineLowLatencyAsyncRecvTransfer, , false)
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecvCopy, bf16_fp8cast, hip_bfloat16, true))
MORI_FP8_ANY(WRAP_BOOL(EpCombineLowLatencyAsyncRecvTransfer, bf16_fp8cast, hip_bfloat16, true))

// Fp8BlockwiseQuant family (mirrors IntraNode `_bf16_nop2p_fp8bwq` template flag pattern).
// Template args are <T, UseFp8DirectCast=false, UseFp8BlockwiseQuant=true>. Gated on FP8
// support because the producer/consumer rely on core::CombineInternalFp8.
#define WRAP_ASYNC_FP8BWQ(kernel)                                                                  \
  extern "C" __global__ void kernel##_bf16_fp8bw_quant(EpDispatchCombineArgs<hip_bfloat16> args) { \
    kernel##_body<hip_bfloat16, /*UseFp8DirectCast=*/false, /*UseFp8BlockwiseQuant=*/true>(args);  \
  }
MORI_FP8_ANY(WRAP_ASYNC_FP8BWQ(EpCombineLowLatencyAsyncSendCopy))
MORI_FP8_ANY(WRAP_ASYNC_FP8BWQ(EpCombineLowLatencyAsyncSendTransfer))
MORI_FP8_ANY(WRAP_ASYNC_FP8BWQ(EpCombineLowLatencyAsyncRecvTransfer))
MORI_FP8_ANY(WRAP_ASYNC_FP8BWQ(EpCombineLowLatencyAsyncRecvCopy))
#undef WRAP_ASYNC_FP8BWQ
