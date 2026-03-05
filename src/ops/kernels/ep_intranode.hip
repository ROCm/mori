// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, _stdmoe, true)

// Combine
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _nop2p, false, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)

// Convert (used with ENABLE_STANDARD_MOE_ADAPT)
extern "C" __global__ void mori_ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}
#define WRAP_CONVERT_COMBINE(suffix, T)                                                         \
  extern "C" __global__ void ConvertCombineInputKernel_##suffix(ConvertCombineInputArgs args) { \
    ConvertCombineInputDevice<T, true>(args);                                                   \
  }
WRAP_CONVERT_COMBINE(f32, float)
WRAP_CONVERT_COMBINE(bf16, hip_bfloat16)
MORI_FP8_OCP(WRAP_CONVERT_COMBINE(fp8_ocp, __hip_fp8_e4m3))
MORI_FP8_FNUZ(WRAP_CONVERT_COMBINE(fp8_fnuz, __hip_fp8_e4m3_fnuz))
