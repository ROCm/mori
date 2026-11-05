# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Host-side (torch) weight preparation and reference helpers: mxfp4 quantisation with e8m0
scales, the a16w4 GEMM weight / scale layouts, and their dequantisation.

Copied from ROCm/aiter (``aiter/ops/quant.py``, ``aiter/ops/shuffle.py``,
``aiter/utility/fp4_utils.py``, ``dtypes``, ``mx_types``); each definition carries a
``# from`` comment naming its source module.
"""

from torch import Tensor
import torch
import torch.nn.functional as F


# from aiter.utility.dtypes
_8bit_fallback = torch.uint8


# from aiter.utility.dtypes
fp4x2 = getattr(torch, "float4_e2m1fn_x2", _8bit_fallback)


# from aiter.utility.mx_types
class MxScaleRoundModeInt:
    """Bare-int mirror of C++ ``MxScaleRoundMode`` (mx_quant_utils.h).

    Values must stay 1:1 with the C++ enum and the pybind ``MxScaleRoundMode``
    class; the lazy loader below verifies this on first import of the
    pybind enum and raises :class:`AssertionError` on drift.
    """

    RoundDown = 0
    RoundUp = 1
    Even = 2
    Ceil = 3


# from aiter.utility.mx_types
MX_DEFAULT_ROUND_MODE = MxScaleRoundModeInt.RoundUp


# from aiter.ops.quant
RoundModeLike = int  # an MxScaleRoundModeInt value


# from aiter.utility.dtypes
fp32 = torch.float32


# from aiter.utility.fp4_utils
def e8m0_to_f32(scale_e8m0_biased):
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 255
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 4194304
    scale_f32[nan_case] = 2139095041
    scale_f32 = scale_f32.view(fp32)
    return scale_f32


# from aiter.utility.fp4_utils
def _n_ones(n: int) -> int:
    return (1 << n) - 1


# from aiter.utility.fp4_utils
EBITS_F32, MBITS_F32 = 8, 23


# from aiter.utility.fp4_utils
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


# from aiter.utility.fp4_utils
def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


# from aiter.utility.fp4_utils
def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# from aiter.utility.fp4_utils
def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# from aiter.utility.fp4_utils
def f32_to_mxfp4(x):
    (FP4_EBITS, FP4_MBITS) = (2, 1)
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(fp4x2)
    return x


# from aiter.utility.dtypes
fp8_e8m0 = getattr(torch, "float8_e8m0fnu", _8bit_fallback)


# from aiter.utility.mx_types
class MxDtypeInt:
    """Bare-int mirror of C++ ``MxDtype`` (mx_quant_utils.h)."""

    FP4_E2M1 = 0
    FP8_E4M3 = 1
    FP8_E4M3_FNUZ = 2


# from aiter.utility.fp4_utils
_DTYPE_CFG = {
    MxDtypeInt.FP4_E2M1: (2, 6.0, 1),  # OCP MXFP4 / DSv4 / FlashInfer
    MxDtypeInt.FP8_E4M3: (8, 448.0, 3),  # OCP / NVIDIA H100 / gfx950+ (e4m3fn)
    MxDtypeInt.FP8_E4M3_FNUZ: (7, 240.0, 3),  # AMD gfx942 hardware FP8 (e4m3fnuz)
}


# from aiter.utility.fp4_utils
def _f32_to_e8m0_ceil_impl(x: Tensor) -> Tensor:
    """Ceil pow2 of x as E8M0 (caller passes ``amax / divisor``).

    Bumps the biased exponent by 1 when any f32 mantissa bit is set,
    except for NaN/Inf (exponent == 0xFF, kept as NaN); never rolls past
    0xFF.
    """
    u32 = x.view(torch.int32)
    exponent = (u32 >> 23 & 255).view(torch.uint32).to(torch.uint8)
    nan_case = exponent == 255
    mantissa_nonzero = u32 & 8388607 != 0
    nonmax_exp = exponent < 255
    bump = mantissa_nonzero & nonmax_exp
    exponent = torch.where(bump, exponent + 1, exponent)
    exponent[nan_case] = 255
    return exponent.view(fp8_e8m0)


# from aiter.utility.fp4_utils
def _f32_to_e8m0_floor_impl(x: Tensor) -> Tensor:
    """Floor pow2 of x as E8M0 (caller passes ``amax / divisor``).

    NaN/Inf inputs (biased exponent == 0xFF) pass through unchanged --
    the floor operation is the identity on the exponent field.
    """
    u32 = x.view(torch.int32)
    exponent = (u32 >> 23 & 255).view(torch.uint32).to(torch.uint8)
    return exponent.view(fp8_e8m0)


# from aiter.utility.fp4_utils
def f32_to_mx_e8m0_scale(
    amax: Tensor,
    *,
    mode: int = MX_DEFAULT_ROUND_MODE,
    dtype: int = MxDtypeInt.FP4_E2M1,
) -> Tensor:
    """Compute the per-block E8M0 scale for an MX format (CPU torch ref).

    Mirrors PyTorch torchao ``to_mx(scaling_mode, elem_dtype)`` semantics
    1:1, and is the CPU-side analogue of the FlyDSL
    :func:`aiter.ops.flydsl.kernels.quant_utils.emit_mx_e8m0_scale` IR
    builder. The four rounding formulas are dtype-agnostic; ``dtype``
    only selects ``target_max_pow2`` / ``max_pos`` / ``mbits`` constants
    from :data:`_DTYPE_CFG`.

    See :class:`MxScaleRoundMode` (``aiter.utility.mx_types``) for the four
    formulas and cross-stack mapping (PyTorch torchao / NV / DSv4 /
    FlashInfer / AMD Quark naming).

    Args:
        amax: f32 (or castable) non-negative tensor of per-block
            ``max(|x|)`` values. Caller is responsible for the per-block
            reduction **and** for taking abs -- negative inputs produce
            undefined results in Even mode (bit-level rounding on a
            negative f32 sign bit).
        mode: ``MxScaleRoundMode`` int or pybind enum. Default ``RoundUp``
            (industry consensus for MXFP4 and MXFP8).
        dtype: ``MxDtype`` int or pybind enum. Default ``FP4_E2M1``.

    Returns:
        E8M0-encoded biased exponent tensor (``dtypes.fp8_e8m0``), same
        shape as ``amax``. NaN/Inf inputs map to ``0xFF`` (E8M0 NaN).
        Inputs near FLT_MAX may also map to ``0xFF`` after ceil rounding.
    """
    # Normalise int / pybind enum into a plain int -- pybind11 enum classes
    # do not auto-compare equal to ``int`` (unlike ``IntEnum``), so callers
    # passing ``mode=1`` would otherwise mis-dispatch.
    mode_int = int(mode)
    dtype_int = int(dtype)
    if dtype_int not in _DTYPE_CFG:
        raise ValueError(
            f"f32_to_mx_e8m0_scale: unsupported dtype {dtype!r}; "
            f"supported: {list(_DTYPE_CFG)}"
        )
    target_max_pow2, max_pos, mbits = _DTYPE_CFG[dtype_int]
    target_pow2_factor = float(1 << target_max_pow2)  # 2^target_max_pow2

    if mode_int == MxScaleRoundModeInt.RoundUp:
        # ceil_pow2(amax / max_pos) -- NV / DSv4 / FlashInfer / torchao RCEIL.
        return _f32_to_e8m0_ceil_impl(amax / max_pos)

    if mode_int == MxScaleRoundModeInt.RoundDown:
        # floor_pow2(amax) / 2^target_max_pow2 -- OCP MX / torchao FLOOR.
        return _f32_to_e8m0_floor_impl(amax / target_pow2_factor)

    if mode_int == MxScaleRoundModeInt.Ceil:
        # ceil_pow2(amax) / 2^target_max_pow2 -- torchao CEIL.
        return _f32_to_e8m0_ceil_impl(amax / target_pow2_factor)

    if mode_int == MxScaleRoundModeInt.Even:
        # round_pow2_special(amax) / 2^target_max_pow2 -- torchao EVEN /
        # Quark EVEN. val_to_add is mantissa-precision-aware:
        # FP4 (mbits=1) -> 0x200000 (1.5x threshold)
        # FP8 e4m3 (mbits=3) -> 0x80000  (~1.0625x threshold).
        val_to_add = 1 << (23 - mbits - 1)
        u32 = amax.view(torch.int32)
        rounded = (u32 + val_to_add) & 0xFF800000
        rounded_f = rounded.view(torch.float32)
        return _f32_to_e8m0_floor_impl(rounded_f / target_pow2_factor)

    raise ValueError(
        f"f32_to_mx_e8m0_scale: unknown mode {mode!r} "
        f"(expected 0=RoundDown, 1=RoundUp, 2=Even, 3=Ceil)"
    )


# from aiter.ops.quant
def per_1x32_f4_quant(
    x,
    scale=None,
    quant_dtype=fp4x2,
    shuffle=False,
    pack_dim=-1,
    round_mode: RoundModeLike = MX_DEFAULT_ROUND_MODE,
):
    """Torch reference for MXFP4 (E2M1) per-1x32 block-scale quantization.

    Mirrors the HIP path (:func:`per_1x32_f4_quant_hip` /
    :func:`quant_mxfp4_hip`) and the Triton path
    (:func:`per_1x32_f4_quant_triton`); all share the E8M0 block layout and
    the :class:`MxScaleRoundMode` enum. PyTorch equivalent:
    ``torchao/prototype/mx_formats/{config,mx_tensor}.py``.

    Args:
        x: Input of shape ``(..., N)`` or ``(M, N)``.
        scale: Pre-computed scale (optional, usually ``None``).
        quant_dtype: Must be ``dtypes.fp4x2``.
        shuffle: Apply e8m0 scale shuffling for hardware.
        pack_dim: ``-1`` (default) packs the last dim -> ``tl.dot_scaled``
            **LHS** ``A(M,K) -> fp4=(M,K//2), scale=(M,K//32)``;
            ``0`` packs the first dim -> **RHS**
            ``B(K,N) -> fp4=(K//2,N), scale=(K//32,N)``.
        round_mode: :class:`MxScaleRoundMode` or bare ``int`` 0..3
            (interchangeable). Default ``RoundUp`` (industry default:
            NV ROUND_UP / DSv4 Pro / FlashInfer / torchao RCEIL).
            See :class:`MxScaleRoundMode` for the four formulas and
            cross-stack mapping.

    Returns:
        ``(quantized_tensor, scale_tensor)``.
    """
    assert quant_dtype == fp4x2
    assert not shuffle, "scale shuffling is not vendored (use shuffle_scale_a16w4)"
    if x.dim() >= 3 and x.numel() * x.element_size() > 4 * 1024**3:
        assert pack_dim == -1, "pack_dim=0 requires a 2D input tensor (K, N)"
        parts = [
            per_1x32_f4_quant(
                x[i],
                scale=scale,
                quant_dtype=quant_dtype,
                shuffle=False,
                pack_dim=-1,
                round_mode=round_mode,
            )
            for i in range(x.shape[0])
        ]
        y = torch.stack([p[0] for p in parts], dim=0)
        scale = torch.cat([p[1].view(torch.uint8) for p in parts], dim=0)
        scale = scale.view(fp8_e8m0)
        return (y, scale)
    block_size = 32
    transposed = False
    if pack_dim == 0:
        assert x.dim() == 2, "pack_dim=0 requires a 2D input tensor (K, N)"
        x = x.T.contiguous()
        transposed = True
    shape_original = x.shape
    x = x.view(-1, shape_original[-1])
    (m, _n) = x.shape
    x = x.view(-1, block_size)
    max_abs = torch.amax(torch.abs(x.float()), 1)
    try:
        scale_e8m0_biased = f32_to_mx_e8m0_scale(
            max_abs, mode=round_mode, dtype=MxDtypeInt.FP4_E2M1
        )
    except ValueError as e:
        raise ValueError(
            f"per_1x32_f4_quant: invalid round_mode={round_mode!r} (type={type(round_mode).__name__}); expected 0 (RoundDown/FLOOR), 1 (RoundUp/RCEIL), 2 (Even), 3 (Ceil), or any MxScaleRoundMode value."
        ) from e
    scale_f32 = e8m0_to_f32(scale_e8m0_biased)
    y = x.float() / scale_f32.view(-1, 1)
    y = f32_to_mxfp4(y)
    y = y.view(*shape_original[:-1], -1)
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    scale = scale.view(fp8_e8m0)
    if transposed:
        y = y.T.contiguous()
        scale = scale.view(torch.uint8).T.contiguous().view(fp8_e8m0)
    return (y, scale)


# from aiter.ops.shuffle
def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    gate_up: bool = False,
    pad_k_to: int = 0,
) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    original_k = x.shape[-1]
    if pad_k_to:
        if pad_k_to < 0:
            raise ValueError(f"pad_k_to must be non-negative, got {pad_k_to}")
        if use_int4:
            raise NotImplementedError("pad_k_to is not supported with use_int4=True")
        if is_guinterleave:
            raise NotImplementedError(
                "pad_k_to is not supported with is_guinterleave=True"
            )
        padded_k = ((original_k + pad_k_to - 1) // pad_k_to) * pad_k_to
        if padded_k != original_k:
            x = F.pad(x.contiguous(), (0, padded_k - original_k), value=0)

    if is_guinterleave:
        experts_cnt, N, K_pk = x.shape
        if gate_up:
            N = N // 2
        NLane, KPack = layout
        KLane = 64 // NLane
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        if gate_up:
            x_ = x.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)
            x_ = x_.permute(0, 2, 1, 4, 5, 3, 6).contiguous()
        else:
            x_ = x.view(experts_cnt, N0, NLane, K0, KLane, KPack)
            x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
        x_ = x_.view(*x.shape).contiguous().view(x_type)
        x_.is_shuffled = True
        x_.gate_up_interleave = gate_up
        return x_

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    if pad_k_to:
        x_.aiter_original_k = original_k
        x_.aiter_padded_k = x.shape[-1]
    return x_


# from aiter.ops.shuffle
def shuffle_weight_a16w4(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_weight(..., is_guinterleave=True)`."""
    return shuffle_weight(
        src, layout=(NLane, 16), is_guinterleave=True, gate_up=gate_up
    )


# from aiter.ops.shuffle
def shuffle_scale(
    src: torch.Tensor,
    experts_cnt: int | None = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> torch.Tensor:
    if src is None:
        return src
    if src.dtype == torch.float32:
        return src
    assert src.ndim == 2, "scale must be a 2D tensor"

    if not is_guinterleave:
        m, n = src.shape
        scale_padded = torch.empty(
            (m + 255) // 256 * 256,
            (n + 7) // 8 * 8,
            dtype=src.dtype,
            device=src.device,
        )
        # Initialise the pad. torch.empty leaves it undefined, and kernels that
        # read whole 8-group tiles then consume garbage (E8M0 0xFF is NaN), so
        # the result depends on stale device memory. 0x7F is E8M0 1.0, matching
        # the is_guinterleave branch below.
        if scale_padded.element_size() == 1:
            scale_padded.view(torch.uint8).fill_(0x7F)
        else:
            scale_padded.fill_(1.0)
        scale_padded[:m, :n] = src
        scale = scale_padded
        sm, sn = scale.shape
        scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
        scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
        return scale.view(sm, sn)

    if experts_cnt is None:
        raise ValueError("experts_cnt is required when is_guinterleave=True")

    n_experts, k_ = src.shape
    if k_ % 8 != 0:
        k_padded = (k_ + 7) // 8 * 8
        scale_padded = torch.empty(
            n_experts, k_padded, dtype=src.dtype, device=src.device
        )
        if scale_padded.element_size() == 1:
            scale_padded.view(torch.uint8).fill_(0x7F)
        else:
            scale_padded.fill_(1)
        scale_padded[:, :k_] = src
        src = scale_padded
        k_ = k_padded
    n_ = n_experts // experts_cnt
    # MXFP4 constants.  The scale layout packs two 4-scale dwords per tile-K
    # group, so shapes with K//32 not divisible by 8 are padded above.
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane
    N1 = n_ // N_Lane // N_Pack  # n_ // 32
    real_k = 32 * k_ * K_Pack * K_Lane  # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    # print("src shape", src.shape)
    # Reshape based on moe_kind
    if gate_up:
        # Reshape to: [E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane]
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        # Reshape to: [E, K1, K_Pack, K_Lane, N1, N_Pack, N_Lane]
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    # print("shf_scale shape:", shfl_scale.shape)
    return shfl_scale.view(*src.shape).contiguous()


# from aiter.ops.shuffle
def shuffle_scale_a16w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_scale(..., is_guinterleave=True)`."""
    return shuffle_scale(
        src, experts_cnt=experts_cnt, is_guinterleave=True, gate_up=gate_up
    )


# from aiter.utility.fp4_utils
def mxfp4_to_f32(x):
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]
