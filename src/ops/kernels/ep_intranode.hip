// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch (block-local exact count + one remote fetch_add(N) per destPe + local slot
// distribution).
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, _stdmoe, true)

// Dispatch Low Lantency (SMALL TOKEN)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, _stdmoe, true)

// Combine
WRAP_ALL_TYPES_BOOL2(EpCombineIntraNodeKernel, _p2p, true, false)
WRAP_ALL_TYPES_BOOL2(EpCombineIntraNodeKernel, _nop2p, false, false)
WRAP_ALL_TYPES_BOOL2(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true)
MORI_FP8_ANY(WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise, hip_bfloat16, false,
                        false, true))
MORI_FP8_ANY(WRAP_BOOL5(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise_noweight_block128_vec8,
                        hip_bfloat16, false, false, true, false, 128))
MORI_FP8_ANY(WRAP_BOOL5(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise_noweight_block256_vec8,
                        hip_bfloat16, false, false, true, false, 256))
// AccumNum=9 variants: shared-expert fusion routes every token through 9 sources
// (8 routed + 1 fused shared expert, topk 8->9). Same weightless vec8 path as top8.
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel,
                        bf16_nop2p_fp8_blockwise_noweight_block128_vec8_top9, hip_bfloat16, false,
                        false, true, false, 128, 9))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel,
                        bf16_nop2p_fp8_blockwise_noweight_block256_vec8_top9, hip_bfloat16, false,
                        false, true, false, 256, 9))

// Blockwise FP4 (E2M1) combine: same as the fp8_blockwise variants but with UseFp4Combine=true (7th
// template arg), transporting packed FP4 (0.5 byte/elem) over the shared blockwise staging path.
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise, hip_bfloat16, false,
                        false, true, true, 0, 8, true))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise_noweight_block128_vec8,
                        hip_bfloat16, false, false, true, false, 128, 8, true))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise_noweight_block256_vec8,
                        hip_bfloat16, false, false, true, false, 256, 8, true))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel,
                        bf16_nop2p_fp4_blockwise_noweight_block128_vec8_top9, hip_bfloat16, false,
                        false, true, false, 128, 9, true))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel,
                        bf16_nop2p_fp4_blockwise_noweight_block256_vec8_top9, hip_bfloat16, false,
                        false, true, false, 256, 9, true))
// AccumNum=6 variants. The top8/top9 set above covers DeepSeek-V3/R1 (8 routed experts, or 8+1
// with shared-expert fusion); a model routing 6 has no AccumNum=6 symbol to select, so its
// combine falls off the vectorized dequant entirely onto WarpAccumCombineDequantSegment, which
// unpacks one nibble per lane per step. Registered for FP4 only: FP8 blockwise moves twice the
// bytes per element, so the reduce is not the same bottleneck there.
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel,
                        bf16_nop2p_fp4_blockwise_noweight_block128_vec8_top6, hip_bfloat16, false,
                        false, true, false, 128, 6, true))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel,
                        bf16_nop2p_fp4_blockwise_noweight_block256_vec8_top6, hip_bfloat16, false,
                        false, true, false, 256, 6, true))

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
