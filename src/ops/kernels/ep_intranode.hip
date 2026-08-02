// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch (block-local exact count + one remote fetch_add(N) per destPe + local slot
// distribution + 1D TDM). The body behind this symbol is EpDispatchIntraNodeKernel_body in
// intranode.hpp.
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeBatchKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeBatchKernel, _stdmoe, true)

// Dispatch Low Lantency (SMALL TOKEN)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, _stdmoe, true)

// Combine
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _nop2p, false, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)
MORI_FP8_ANY(WRAP_BOOL4(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise, hip_bfloat16, false, false,
                        false, true))
// PULL blockwise. The point of quantising is to move half the bytes, and _nop2p spends that saving
// and more: its send quantises straight into the PEER's staging slot, a per-lane cross-card scatter
// that skips the TDM store the plain path gets, so fp8 ends up SLOWER than bf16 (measured at 64x8
// EP4: bf16 push 417.5us against fp8_blockwise 3679.2us for half the payload). With UseP2PRead=true the
// same kernel instead quantises into its OWN combineInp (intranode.hpp:2237, taken because
// useExternalInpBuffer stays true) and the consumers read peer fp8 + peer scales -- a path the
// gather already implements in full (intranode.hpp:3509-3521). Only the symbol was missing.
//
// MEASURED at 64x8 EP4, MORI_COMB_TDM=4, check armed, both rc=0: _nop2p 3667.8us against 1466.4us
// here under MORI_COMB_QPULL=1, plus dispatch 237.7 -> 175.9.
//
// It only pays WITH the tile path. Switching transport alone, while _cPullType still refused
// 1-byte TokT, measured 4748.6us -- worse than PUSH, because it turned a cross-card scatter of
// writes into a cross-card scatter of reads and PUSH at least gathers from local staging. The win
// is the bulk TDM move, not the direction of travel.
MORI_FP8_ANY(WRAP_BOOL4(EpCombineIntraNodeKernel, bf16_p2p_fp8_blockwise, hip_bfloat16, true, false,
                        false, true))
// The blockwise quantise pass as a kernel of its own, so it can be launched wider than the gather
// wants to be. Paired with useExternalInpBuffer=0 on the combine launch, which is the flag that
// switches the in-kernel staging arm off; see EpCombineQuantizeInputKernel_body in intranode.hpp.
MORI_FP8_ANY(WRAP(EpCombineQuantizeInputKernel, bf16, hip_bfloat16))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise_noweight_block128_vec8,
                        hip_bfloat16, false, false, false, true, false, 128))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise_noweight_block256_vec8,
                        hip_bfloat16, false, false, false, true, false, 256))
// AccumNum=9 variants: shared-expert fusion routes every token through 9 sources
// (8 routed + 1 fused shared expert, topk 8->9). Same weightless vec8 path as top8.
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise_noweight_block128_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 128, 9))
MORI_FP8_ANY(WRAP_BOOL7(EpCombineIntraNodeKernel, bf16_nop2p_fp8_blockwise_noweight_block256_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 256, 9))

// Blockwise FP4 (E2M1) combine: same as the fp8_blockwise variants but with UseFp4Combine=true (8th
// template arg), transporting packed FP4 (0.5 byte/elem) over the shared blockwise staging path.
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise, hip_bfloat16, false, false,
                        false, true, true, 0, 8, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise_noweight_block128_vec8,
                        hip_bfloat16, false, false, false, true, false, 128, 8, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise_noweight_block256_vec8,
                        hip_bfloat16, false, false, false, true, false, 256, 8, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise_noweight_block128_vec8_top9,
                        hip_bfloat16, false, false, false, true, false, 128, 9, true))
MORI_FP8_ANY(WRAP_BOOL8(EpCombineIntraNodeKernel, bf16_nop2p_fp4_blockwise_noweight_block256_vec8_top9,
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
