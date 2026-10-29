// KernelType::InterNodeV1 — v1 internode dispatch + combine.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchCopyToStaging)
WRAP_ALL_TYPES_BOOL(EpDispatchInterNodeV1Kernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchInterNodeV1Kernel, _atomicfree, true)

WRAP_ALL_TYPES(EpCombineSync)
WRAP_ALL_TYPES(EpCombineSyncBarrier)
WRAP_ALL_TYPES_BOOL(EpCombineInterNodeV1Kernel, , false)
WRAP_ALL_TYPES_BOOL(EpCombineInterNodeV1Kernel, _atomicfree, true)
WRAP_ALL_TYPES(EpCombineAll)
