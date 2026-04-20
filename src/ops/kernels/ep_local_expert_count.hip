// Standalone wrapper for LocalExpertCountKernel.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

#include "src/ops/dispatch_combine/ep_local_expert_count.hpp"

extern "C" __global__ __attribute__((visibility("default"))) void LocalExpertCountKernel(
    mori::moe::LocalExpertCountArgs args) {
  mori::moe::LocalExpertCountKernel_body(args);
}
