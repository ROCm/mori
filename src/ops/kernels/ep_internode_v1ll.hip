// KernelType::InterNodeV1LL — v1 internode low-latency dispatch + combine.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchCopyToStaging)

WRAP_ALL_TYPES_BOOL(EpDispatchInterNodeV1KernelLowLatency, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchInterNodeV1KernelLowLatency, _stdmoe, true)

WRAP_ALL_TYPES(EpCombineSync)
WRAP_ALL_TYPES(EpCombineSyncBarrier)
WRAP_ALL_TYPES_BOOL(EpCombineInterNodeV1KernelLowLatency, , false)
WRAP_ALL_TYPES_BOOL(EpCombineInterNodeV1KernelLowLatency, _stdmoe, true)
WRAP_ALL_TYPES(EpCombineAll)
