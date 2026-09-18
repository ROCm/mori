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
"""The fp8 GEMM for the all-to-all: the plain one now, the fused ones next.

The GEMM is ``gemm_ar``'s. This operation differs in *where C goes*, not in how
it is computed, so the pipeline, the MFMA schedule, the C-store ladder and the
block-scale handling are all imported from there rather than duplicated.

Three C destinations exist for this op, and only the first is implemented here:

===============  ==================================  =========================
destination      address of element ``(row, col)``    used by
===============  ==================================  =========================
``local``        ``row*N + col``                     ``gemm-only``, ``split-*``
``staging``      ``dest*M*shard_n + row*shard_n      ``fused-sdma``
                 + (col - dest*shard_n)``
``peer``         same, but in peer ``dest``'s window  ``fused-lsa``
===============  ==================================  =========================

where ``dest = col // shard_n``, which is uniform across a GEMM tile because
``layout`` requires ``N % (world*BLOCK_N) == 0``.

The two fused destinations are a small change to the C-store ladder rather than
a new one, and that is worth stating precisely because it is not obvious from
the gemm_ar source. ``_PermlaneStoreC._emit`` -- the default store variant --
already computes its own flat index and already takes the scales as separate
arguments::

    idx = select(col + 7 < self.c_cols, row * self.c_cols + col, oob)

So constructing the store with ``c_cols = shard_n`` and handing ``_emit`` a
**destination-local** base column makes that one line produce the a2a address,
the bounds test correct (``local_col + 7 < shard_n``) and ``oob`` land one past
the slab. The B scale still needs the global column, and ``store()`` passes the
scales separately, so the override is::

    def store(self, c_frag, base_row, base_col):
        self._emit(c_frag, base_row, base_col - dest * shard_n,
                   self._a_scales(base_row), self._b_scales(base_col))

That is the whole of the address-map difference. It is written down here because
the next commit implements it, and because discovering it is what decided this
file would be ~200 lines instead of a 1500-line copy.
"""

from .layout import DEFAULT_BLOCK_M, DEFAULT_BLOCK_N


def compile_gemm_local(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
    b_preshuffled: bool = True,
    quant: str = "ptpc",
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    swap_ab: bool = True,
    permlane: bool = True,
    lane_transpose: bool = False,
    hoist_scales: bool = False,
):
    """The GEMM alone, writing a plain ``[M, N]`` row-major C.

    This is ``gemm_ar.compile_fused_gemm_scatter`` with ``fuse=False``: that
    compiles its whole epilogue tail out (``if const_expr(fuse and not
    direct_lsa)``) and leaves exactly the unmodified 8-wave fp8 GEMM. Calling it
    rather than re-deriving it is the point -- ``gemm-only`` here and
    ``gemm-only`` there have to be the *same* kernel or neither number means
    anything next to the other.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)``. ``dev_comm`` and ``win`` are unused on this path but stay in
    the signature because they are the shared kernel's.

    The ``ArConfig`` built below is a **placeholder**. With ``fuse=False`` the
    only things read from it are folded into code that is then discarded, so it
    does not have to describe the window that is actually allocated -- and it
    cannot, because an all-to-all's window has a different shape. If a future
    change makes the unfused path read an offset, this is where it will go
    wrong, so the config is deliberately built to *fail loudly* rather than
    silently address something: its m/n are this op's, so any offset it produced
    would point inside the real window rather than at random.
    """
    from ..gemm_ar import ArConfig
    from ..gemm_ar.kernels_fused import compile_fused_gemm_scatter

    placeholder = ArConfig(world_size=cfg.world_size, m=cfg.m, n=cfg.n)
    return compile_fused_gemm_scatter(
        placeholder,
        rank,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        b_preshuffled=b_preshuffled,
        quant=quant,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
        fuse=False,
        emit_put=False,
        swap_ab=swap_ab,
        permlane=permlane,
        lane_transpose=lane_transpose,
        hoist_scales=hoist_scales,
    )


__all__ = ["compile_gemm_local"]
