// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, _stdmoe, true)

// Dispatch Low Lantency (SMALL TOKEN)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, _stdmoe, true)

// Combine
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _nop2p, false, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)
MORI_FP8_ANY(WRAP_BOOL4(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq, hip_bfloat16, false, false,
                        false, true))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block128_vec8,
                        hip_bfloat16, false, false, false, true, false, 128))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block256_vec8,
                        hip_bfloat16, false, false, false, true, false, 256))
// AccumNum=9 variants: shared-expert fusion routes every token through 9 sources
// (8 routed + 1 fused shared expert, topk 8->9). Same weightless vec8 path as top8.
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block128_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 128, 9))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block256_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 256, 9))

// Blockwise FP4 (E2M1) combine: same as the fp8bwq variants but with UseFp4Combine=true (8th
// template arg), transporting packed FP4 (0.5 byte/elem) over the shared blockwise staging path.
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4bwq, hip_bfloat16, false, false,
                        false, true, true, 0, 8, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4bwq_noweight_block128_vec8,
                        hip_bfloat16, false, false, false, true, false, 128, 8, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4bwq_noweight_block256_vec8,
                        hip_bfloat16, false, false, false, true, false, 256, 8, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4bwq_noweight_block128_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 128, 9, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4bwq_noweight_block256_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 256, 9, true))

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
