// KernelType::InterNodeV1 — v1 internode dispatch + combine.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchCopyToStaging)
WRAP_ALL_TYPES(EpDispatchInterNodeV1Kernel)

WRAP_ALL_TYPES(EpCombineSync)
WRAP_ALL_TYPES(EpCombineSyncBarrier)
WRAP_ALL_TYPES(EpCombineInterNodeV1Kernel)
WRAP_ALL_TYPES(EpCombineAll)
