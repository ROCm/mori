// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

#if defined(__gfx1201__)
#include "src/ops/dispatch_combine/intranode_ep2.hpp"
#include "src/ops/dispatch_combine/intranode_ep4.hpp"

WRAP(EpDispatchIntraNodeEp4HybridKernel, bf16, hip_bfloat16)
WRAP(EpCombineIntraNodeEp4PushKernel, bf16, hip_bfloat16)
WRAP_BOOL(EpCombineIntraNodeEp4PushKernel, bf16_tile512, hip_bfloat16, 512)
WRAP_BOOL(EpCombineIntraNodeEp4PushKernel, bf16_tile1024, hip_bfloat16, 1024)
WRAP(EpDispatchIntraNodeEp2PushKernel, bf16, hip_bfloat16)
WRAP_BOOL2(EpDispatchIntraNodeEp2PushKernel, bf16_small_v2, hip_bfloat16, true, true)
WRAP_BOOL2(EpDispatchIntraNodeEp2PushKernel, bf16_small_v2_compat, hip_bfloat16, true, false)
WRAP(EpCombineIntraNodeEp2Kernel, bf16_push, hip_bfloat16)
WRAP_BOOL2(EpCombineIntraNodeEp2Kernel, bf16_push_small_v2_512, hip_bfloat16, true, 512)
WRAP_BOOL2(EpCombineIntraNodeEp2Kernel, bf16_push_small_v2_1024, hip_bfloat16, true, 1024)
WRAP(EpDispatchIntraNodeEp4HybridKernel, fp16, __half)
WRAP(EpCombineIntraNodeEp4PushKernel, fp16, __half)
WRAP_BOOL(EpCombineIntraNodeEp4PushKernel, fp16_tile512, __half, 512)
WRAP_BOOL(EpCombineIntraNodeEp4PushKernel, fp16_tile1024, __half, 1024)
WRAP(EpDispatchIntraNodeEp2PushKernel, fp16, __half)
WRAP_BOOL2(EpDispatchIntraNodeEp2PushKernel, fp16_small_v2, __half, true, true)
WRAP_BOOL2(EpDispatchIntraNodeEp2PushKernel, fp16_small_v2_compat, __half, true, false)
WRAP(EpCombineIntraNodeEp2Kernel, fp16_push, __half)
WRAP_BOOL2(EpCombineIntraNodeEp2Kernel, fp16_push_small_v2_512, __half, true, 512)
WRAP_BOOL2(EpCombineIntraNodeEp2Kernel, fp16_push_small_v2_1024, __half, true, 1024)

// Runtime top-k (1..64), separate from the measured top-8 instantiations.
#define WRAP_RDNA4_TOPK(sfx, T)                                                               \
  WRAP_BOOL(EpDispatchIntraNodeEp4HybridKernel, sfx##_topk, T, 0)                              \
  WRAP_BOOL2(EpCombineIntraNodeEp4PushKernel, sfx##_topk, T, 0, 0)                             \
  WRAP_BOOL2(EpCombineIntraNodeEp4PushKernel, sfx##_topk_tile512, T, 512, 0)                    \
  WRAP_BOOL2(EpCombineIntraNodeEp4PushKernel, sfx##_topk_tile1024, T, 1024, 0)                  \
  WRAP_BOOL3(EpDispatchIntraNodeEp2PushKernel, sfx##_topk, T, false, false, 0)                  \
  WRAP_BOOL3(EpDispatchIntraNodeEp2PushKernel, sfx##_topk_small_v2, T, true, true, 0)            \
  WRAP_BOOL3(EpDispatchIntraNodeEp2PushKernel, sfx##_topk_small_v2_compat, T, true, false, 0)     \
  WRAP_BOOL3(EpCombineIntraNodeEp2Kernel, sfx##_topk_push, T, false, 512, 0)                    \
  WRAP_BOOL3(EpCombineIntraNodeEp2Kernel, sfx##_topk_push_small_v2_512, T, true, 512, 0)          \
  WRAP_BOOL3(EpCombineIntraNodeEp2Kernel, sfx##_topk_push_small_v2_1024, T, true, 1024, 0)
WRAP_RDNA4_TOPK(bf16, hip_bfloat16)
WRAP_RDNA4_TOPK(fp16, __half)
#undef WRAP_RDNA4_TOPK
#endif

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
