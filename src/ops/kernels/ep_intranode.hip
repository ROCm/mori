// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch (block-local exact count + one remote fetch_add(N) per destPe + local slot
// distribution). _ENTRY because the body behind this symbol depends on the architecture:
// EpDispatchIntraNodeKernel_entry in intranode_entry.hpp picks it. Same for combine below.
WRAP_ALL_TYPES_BOOL_ENTRY(EpDispatchIntraNodeKernel, , false)
WRAP_ALL_TYPES_BOOL_ENTRY(EpDispatchIntraNodeKernel, _stdmoe, true)

// Dispatch Low Lantency (SMALL TOKEN)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, _stdmoe, true)

// Combine
WRAP_ALL_TYPES_BOOL3_ENTRY(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3_ENTRY(EpCombineIntraNodeKernel, _nop2p, false, false, false)
WRAP_ALL_TYPES_BOOL3_ENTRY(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
WRAP_BOOL3_ENTRY(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)
MORI_FP8_ANY(WRAP_BOOL4_ENTRY(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise, hip_bfloat16,
                              false, false, false, true))
#if defined(MORI_FP8_TYPE_FNUZ_ENABLED) || defined(MORI_FP8_TYPE_OCP_ENABLED)
extern "C" __global__ void EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise_sbo(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpCombineIntraNodeKernel_entry<hip_bfloat16, false, false, false, true, true, 0, 8, false,
                                 true>(args);
}
#endif
// PULL blockwise and its separately-launched quantise pass used to be registered here. Both existed
// only to put blockwise on the gfx125x TDM tile fold -- that was worth 3646.2us -> 1410.9us, and
// splitting the quantise out took it to 367.6us. intranode_entry.hpp now keeps quantization out of
// the TDM body entirely, because 367.6us is still 2.2x bf16 (169.2us) for half the bytes, so
// neither symbol has a caller left.
MORI_FP8_ANY(WRAP_BOOL6_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp8_blockwise_noweight_block128_vec8, hip_bfloat16, false,
                              false, false, true, false, 128))
MORI_FP8_ANY(WRAP_BOOL6_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp8_blockwise_noweight_block256_vec8, hip_bfloat16, false,
                              false, false, true, false, 256))
// AccumNum=9 variants: shared-expert fusion routes every token through 9 sources
// (8 routed + 1 fused shared expert, topk 8->9). Same weightless vec8 path as top8.
MORI_FP8_ANY(WRAP_BOOL7_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp8_blockwise_noweight_block128_vec8_top9, hip_bfloat16,
                              false, false, false, true, false, 128, 9))
MORI_FP8_ANY(WRAP_BOOL7_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp8_blockwise_noweight_block256_vec8_top9, hip_bfloat16,
                              false, false, false, true, false, 256, 9))

// Blockwise FP4 (E2M1) combine: same as the fp8_blockwise variants but with UseFp4Combine=true (8th
// template arg), transporting packed FP4 (0.5 byte/elem) over the shared blockwise staging path.
MORI_FP8_ANY(WRAP_BOOL8_ENTRY(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise, hip_bfloat16,
                              false, false, false, true, true, 0, 8, true))
MORI_FP8_ANY(WRAP_BOOL8_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp4_blockwise_noweight_block128_vec8, hip_bfloat16, false,
                              false, false, true, false, 128, 8, true))
MORI_FP8_ANY(WRAP_BOOL8_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp4_blockwise_noweight_block256_vec8, hip_bfloat16, false,
                              false, false, true, false, 256, 8, true))
MORI_FP8_ANY(WRAP_BOOL8_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp4_blockwise_noweight_block128_vec8_top9, hip_bfloat16,
                              false, false, false, true, false, 128, 9, true))
MORI_FP8_ANY(WRAP_BOOL8_ENTRY(EpCombineIntraNodeKernel,
                              bf16_nop2p_fp4_blockwise_noweight_block256_vec8_top9, hip_bfloat16,
                              false, false, false, true, false, 256, 9, true))

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
