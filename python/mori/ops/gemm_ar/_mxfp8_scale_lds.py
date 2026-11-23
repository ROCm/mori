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
#
# Adapted from aiter's ``ScaleRawLds``
# (aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py), where it carries the notice
# below. It is reproduced here as the upstream attribution; the MIT grant above
# is mori's, and does not restate or replace it.
#
#     SPDX-License-Identifier: Apache-2.0
#     Copyright (c) 2025 FlyDSL Project Contributors
#     Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""mxfp8 scales in the quantiser's own layout, staged through LDS.

``quant="mxfp8_raw"`` reads A's scales as ``[M, K/32]`` and B's as
``[N/32, K/32]`` ue8m0 bytes, row major -- what an mxfp8 quantiser emits, with
no host shuffle. ``_Mxfp8ScaleK`` reads its layouts straight from global
memory instead, one load per K step, and that costs it twice:

* the MFMA that consumes a scale waits for it with ``vmcnt``, which retires in
  issue order, so it also waits for the tile prefetches issued before it --
  63 ``vmcnt`` waits in the K=2048 kernel against 20 here;
* the scale registers are live across the step, and the kernel is already at
  256 VGPRs: at K=512 it spills 8, this spills none.

Here each wave copies its share of a chunk of ``CHUNK`` K steps into LDS with
``buffer_load ... lds``, two chunks double-buffered, and the MFMA operands come
from ``ds_read_u8``. The copies are vmcnt-counted like the tile loads, so the
pipeline's own ``wait_barrier`` covers them.

Differences from aiter: only the 1x32 form is kept, and the staged rows start
at the block's first row rather than ``block * 256``, so a 128-wide N tile
stages the right B rows.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from ._gemm_a8w8_8wave import make_fp8_buffer_tensor


class Mxfp8ScaleLds:
    """Row-major ue8m0 scales for one block, a chunk of K steps at a time in LDS.

    One slot holds 256 A rows then 256 B rows, 16 bytes each: four K steps of
    four 32-blocks. B's ``[N/32, K/32]`` row repeats over its 32 N rows, so a
    lane reads its own row's byte either way. 256 rows cover any tile up to
    256x256; the kernel only builds 256x256 and 256x128.
    """

    #: K steps per staged chunk: 16 bytes of a row, one dword copy per lane.
    CHUNK = 4
    ROWS = 256
    #: One slot: A's rows, then B's. Two slots double-buffer the chunks.
    BYTES = 2 * ROWS * 16

    def __init__(
        self,
        A_scale,
        B_scale,
        K,
        row0,
        col0,
        wave_id,
        a_row,
        b_row,
        lds,
        fp8_ir_t,
        *,
        n_tiles_a,
        n_tiles_b,
    ):
        """``row0`` / ``col0``: the block's first M row and N column.

        ``a_row`` / ``b_row``: this wave's first row inside the block; ``lds``:
        the two slots, ``2 * BYTES``.
        """
        row_bytes = K // 32
        lane = fx.thread_idx.x % 64
        self.n_tiles_a, self.n_tiles_b = n_tiles_a, n_tiles_b
        self.wave_id = wave_id
        self.lds = lds
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        self.lds_ptr_t = fx.PointerType.get(fp8_ir_t, 2, 512)
        self.srcs = []
        for arg, first, rows_per in ((A_scale, row0, 1), (B_scale, col0, 32)):
            # The descriptor takes the tensor's own extent, so rows past M or
            # N/32, and the last chunk's bytes past the tensor's end, read as
            # zero. That needs the chunk offset in the VGPR offset (see dma).
            src = fx.logical_divide(
                make_fp8_buffer_tensor(arg, fp8_ir_t), fx.make_layout(1, 1)
            )
            offs = []
            for step in range_constexpr(2):
                # 512 lanes x 2 copies = 1024 dwords = 256 rows x 4 dwords.
                word = step * 512 + wave_id * 64 + lane
                row = first + word // 4
                if const_expr(rows_per > 1):
                    row = row // rows_per
                offs.append(row * row_bytes + (word % 4) * 4)
            self.srcs.append((src, offs))
        # Lane 16*s + r reads block s of row r. Every read below adds only a
        # constant to this base, which folds into the ds_read offset while the
        # slots sit in the first 64 KB of LDS.
        lane_off = (lane % 16) * 16 + lane // 16
        self.bases = [
            fx.add_offset(lds.ptr, fx.make_int_tuple(a_row * 16 + lane_off)),
            fx.add_offset(
                lds.ptr, fx.make_int_tuple(self.ROWS * 16 + b_row * 16 + lane_off)
            ),
        ]

    def dma(self, chunk):
        """Copy chunk ``chunk`` into slot ``chunk % 2``.

        The chunk offset goes in the per-lane offset, not ``soffset``: the raw
        buffer range check covers the VGPR and instruction offsets only, so via
        ``soffset`` the last row's bytes past K would come from beyond the
        tensor. K % 128 == 0 keeps each row whole dwords, so a dword is either
        in the tensor or zero-filled; the bytes past K are never read anyway.
        Not the immediate offset either: on an LDS-direct load it also moves
        the LDS destination, and slot 0 starts at LDS 0, so it cannot be
        compensated in M0.
        """
        base_i32 = fx.Int32(fx.ptrtoint(self.lds.ptr))
        for op in range_constexpr(2):
            src, offs = self.srcs[op]
            for step in range_constexpr(2):
                dst_off = (
                    chunk % 2 * self.BYTES
                    + op * self.ROWS * 16
                    + step * 2048
                    + self.wave_id * 256
                )
                dst = fx.make_view(
                    fx.inttoptr(self.lds_ptr_t, base_i32 + fx.Int32(dst_off)),
                    fx.make_layout(1, 1),
                )
                fx.copy(
                    self.atom,
                    fx.slice(src, (None, fx.Int32(offs[step]) + chunk * 16)),
                    dst,
                )

    def _read(self, op, half_row, n_tiles, k):
        slot = k // self.CHUNK % 2 * self.BYTES
        words = []
        for t in range_constexpr(n_tiles):
            off = slot + (half_row + t * 16) * 16 + (k % self.CHUNK) * 4
            v = fx.make_view(
                fx.recast_iter(
                    fx.Uint8, fx.add_offset(self.bases[op], fx.make_int_tuple(off))
                ),
                fx.make_layout(1, 1),
            )
            words.append(fx.Int32(v.load()[0]))
        return words

    def step(self, base_row, base_col, ks, lds_block_m, lds_block_n):
        """``_Mxfp8ScaleK.step``'s signature and result, read at op_sel 0.

        ``base_row`` / ``base_col`` are unused: the rows were fixed at
        construction, when the copy offsets were.
        """
        return (
            self._read(0, 0, self.n_tiles_a, ks),
            self._read(0, lds_block_m, self.n_tiles_a, ks),
            self._read(1, 0, self.n_tiles_b, ks),
            self._read(1, lds_block_n, self.n_tiles_b, ks),
        )
