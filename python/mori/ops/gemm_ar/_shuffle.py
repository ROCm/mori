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
