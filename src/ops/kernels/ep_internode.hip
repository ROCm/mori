// KernelType::InterNode — legacy internode dispatch + combine.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_ALL_TYPES(EpDispatchInterNodeKernel)
WRAP_ALL_TYPES(EpCombineInterNodeKernel)
