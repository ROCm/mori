// KernelType::InterNodeV1LL — v1 internode low-latency dispatch + combine.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchCopyToStaging)

WRAP_ALL_TYPES_BOOL2(EpDispatchInterNodeV1KernelLowLatency, , false, false)
WRAP_ALL_TYPES_BOOL2(EpDispatchInterNodeV1KernelLowLatency, _stdmoe, true, false)
WRAP_ALL_TYPES_BOOL2(EpDispatchInterNodeV1KernelLowLatency, _atomicfree, false, true)
WRAP_ALL_TYPES_BOOL2(EpDispatchInterNodeV1KernelLowLatency, _stdmoe_atomicfree, true, true)

WRAP_ALL_TYPES(EpCombineSync)
WRAP_ALL_TYPES(EpCombineSyncBarrier)
WRAP_ALL_TYPES_BOOL2(EpCombineInterNodeV1KernelLowLatency, , false, false)
WRAP_ALL_TYPES_BOOL2(EpCombineInterNodeV1KernelLowLatency, _stdmoe, true, false)
WRAP_ALL_TYPES_BOOL2(EpCombineInterNodeV1KernelLowLatency, _atomicfree, false, true)
WRAP_ALL_TYPES_BOOL2(EpCombineInterNodeV1KernelLowLatency, _stdmoe_atomicfree, true, true)
WRAP_ALL_TYPES(EpCombineAll)
