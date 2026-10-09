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
"""Put B in the layout the 8-wave GEMM's ``b_preshuffled=True`` path reads.

Equivalent to aiter's ``shuffle_weight(x, layout=(16, 16))`` for the one case
this op needs -- a 2-D fp8 weight, no int4, no gate/up interleave -- vendored
so mori does not depend on aiter.
"""

import torch


def preshuffle_b(w: torch.Tensor) -> torch.Tensor:
    """Reorder ``[N, K]`` into the MFMA 16x16x128 preshuffled B layout.

    The emitted order is ``N0 K0 KLane(4) NLane(16) KPack(16)``: lane ``l`` of a
    wave reads ``n1 = l % 16`` and ``k = (l // 16) * 16 + [0..15]``, so a whole
    16x128 B tile is one ``buffer_load_dwordx4`` per lane. This is the same byte
    order CK's ``preShuffleBuffer`` emits with ``NXdl=16``, which is why a
    weight shuffled here can be handed to either kernel.

    Shape is preserved; only the element order changes. ``K`` counts *elements*,
    so for fp8 (1 byte) the ``BK = 32`` and ``KPack = 16`` below are in
    elements too.
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D [N, K] weight, got {w.ndim}-D")
    dtype = w.dtype
    n, k = w.shape
    if w.element_size() != 1:
        raise ValueError(
            f"expected a 1-byte element type (fp8), got {dtype} "
            f"({w.element_size()} bytes)"
        )
    # NLane = 16 lanes over N, KLane = 64 // NLane = 4 lanes over K, and
    # KPack = 16 elements per lane -- so a lane group spans BK = KLane * KPack.
    n_lane, k_pack = 16, 16
    bk = 64 // n_lane * k_pack
    if n % n_lane:
        raise ValueError(f"N={n} must be a multiple of {n_lane}")
    if k % bk:
        raise ValueError(f"K={k} must be a multiple of {bk}")

    out = w.view(n // n_lane, n_lane, k // bk, bk // k_pack, k_pack)
    out = out.permute(0, 2, 3, 1, 4).contiguous()
    return out.view(n, k).view(dtype)


#: ue8m0 block size along K, fixed by the checkpoint and by the MFMA.
MXFP8_BLOCK = 32
#: M rows a wave's four 16-row A tiles span, and so the packing group.
_A_SCALE_GROUP = 64


def preshuffle_a_scale(exps: torch.Tensor) -> torch.Tensor:
    """Put the ue8m0 A scales in the layout ``--quant mxfp8`` reads.

    Takes the exponent bytes as ``[M, K/32]`` -- one per 32-wide K block, the
    orientation every quantiser emits -- and returns a flat int32 buffer to hand
    the kernel as its ``A_scale`` argument.

    Two things happen, for two different reasons.

    **K-block major.** A block group's sixteen lanes want sixteen consecutive
    rows of one K block, so K major puts them at consecutive addresses and the
    load coalesces. Row major spreads them ``K/32`` bytes apart, which measured
    +50-67% on the whole GEMM -- the addresses, not the bytes: same instruction
    count, 3.2x the cache accesses.

    **Four M tiles to a dword.** A lane's four A tiles differ only by sixteen
    rows and share the K block, and ``opsel_b`` on the scaled MFMA names which
    byte of the 32-bit scale operand the instruction reads. So packing the four
    into one dword turns four loads into one and the byte select costs nothing
    -- it is the instruction's own field, not a shift. Worth -5 to -7.5%, and
    it also shrinks the buffer 4x by storing bytes rather than int32. CK does
    the same thing in ``preShuffleScaleBuffer_gfx950`` (its ``MNXdlPack``).

    So within each 64-row group the order goes from ``ti*16 + r`` to
    ``r*4 + ti``, which is a ``(4, 16) -> (16, 4)`` transpose and nothing else.
    """
    if exps.ndim != 2:
        raise ValueError(f"expected a 2-D [M, K/32] scale, got {exps.ndim}-D")
    m, kb = exps.shape
    if m % _A_SCALE_GROUP:
        raise ValueError(f"M={m} must be a multiple of {_A_SCALE_GROUP}")
    if (m * kb) % 4:
        raise ValueError(f"M*K/32 = {m * kb} must be a multiple of 4")
    out = exps.to(torch.uint8).t().contiguous()  # [K/32, M], K-block major
    out = out.view(kb, m // _A_SCALE_GROUP, 4, 16)  # split M into (group, ti, r)
    return out.permute(0, 1, 3, 2).contiguous().reshape(-1).view(torch.int32)
