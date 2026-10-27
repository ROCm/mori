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
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, I001
"""GEMM2 whose epilogue writes weighted bf16 rows into a local, rank-sorted send buffer.

Same GEMM2 mainloop and input layout as ``mega_moe_stage2``; only the output side
differs. Instead of scattering each row into the origin rank's ``comb_inp`` it
writes the row to ``send[out_row[row]]``, where ``out_row`` was assigned once per
row beforehand (``sorted_combine.build_rank_sort_plan``) so every N
tile of a row agrees on it. A negative ``out_row`` drops the row.

With ``comm`` the kernel also carries ``comm.comm_wgs`` extra leading workgroups that
ship finished chunks of the send buffer to the peers with cco SDMA while the GEMM
workgroups are still computing. A GEMM unit signals ``cdone[chunk]`` after its stores.
Each comm WAVE owns its own peers (peer ``p`` belongs to wave ``p % n_comm_waves``): it
spins on the chunk counter itself, then puts that chunk's rows for its peers on its own
SDMA queue, so peers neither wait for each other's spin nor share a wave's issue slot.

Two things about the publish are load-bearing and were each worth more than a
factor on this kernel. The epilogue stores the rows write-through (sc0|sc1) so
that retiring them is the release a copy engine needs; releasing at system scope
instead is a whole-L2 write-back per unit. And the per-chunk counters sit a
cache line apart, because every unit of a chunk bumps one.
"""

from dataclasses import dataclass

#: i32 slots between consecutive chunk-completion counters. Every unit of a
#: chunk bumps its counter, so packing them adjacent puts all of a launch's
#: atomics on one or two cache lines: 128 B apart instead is worth ~8% of the
#: GEMM2 kernel at bs=4096. Spreading a chunk's count over several counters on
#: top of this buys nothing -- the atomic itself is not the cost.
CDONE_STRIDE = 32

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    buf_copy_store,
    ptr_buf_tensor,
)

from aiter.ops.flydsl.kernels.mega_moe.gemm2 import (
    _resolve_g2_knobs,
    _spart_output_tile_index,
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
)
from sorted_combine import (
    ArenaLayout,
    sdma_finish_peer,
    sdma_put_run,
    sdma_wait_consumed,
)
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
    _BUFFER_OFFSET_ABI_BYTES,
    _fp8_scale_for_leader,
    _stage2_lds_bytes,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _fabs_f32 as fabs_f32


@dataclass(frozen=True)
class CommSpec:
    """Compile-time description of the fused SDMA comm workgroups.

    ``comm_wgs`` workgroups of 4 waves each; one wave per peer needs
    ``comm_wgs * 4 >= npes``. ``queues`` is the SDMA channels reserved per
    (src, dst) pair; peer ``p`` uses channel ``p % queues`` for everything it
    sends, so a chunk is never split across channels -- splitting measured
    slower, and on this topology KFD recommends one engine per peer link
    anyway, which makes extra channels pure overhead.
    """

    layout: ArenaLayout
    rank: int
    queues: int
    chunk_tiles: int
    comm_wgs: int = 1
    #: How a sender tells a peer its rows have landed; see
    #: ``sorted_combine.sdma_finish_peer``. "remote" needs the signal to ride
    #: the channel the data went out on, which ``queues`` already guarantees.
    signal_mode: str = "remote"


# fmt: off
def local_sort_epilog(lds_acc_base, accm, n_block_idx, wave, lane, send_base, *, N_OUT, BM, BN,
    send_nbytes, lds_outrow_off, lds_weight_off, g2_bf16_lds=False, scatter_vec=8, store_cm=0,
    quant="none"):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and store them at their send-buffer index."""
    kMChunks = BM // 16
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )
    quant_fp8 = quant == "fp8_blockwise_1x32"
    if quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(f"unsupported quant={quant!r}")
    if quant_fp8 and g2_bf16_lds:
        raise ValueError("fp8_blockwise_1x32 needs the f32 CShuffle (g2_bf16_lds=False)")
    out_elem_bytes = 1 if quant_fp8 else 2
    # Row layout matches aiter's P2P scatter: N_OUT payload bytes followed by the
    # N_OUT/32 e8m0 block scales, so one contiguous SDMA put ships both.
    row_nbytes = N_OUT + N_OUT // 32 if quant_fp8 else N_OUT * 2

    fx.barrier()

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        if const_expr(g2_bf16_lds):
            w_row = [
                fx.ptr_load(
                    lds_typed_ptr(
                        fx.Int32(lds_weight_off) + (row_base + v) * fx.Int32(4),
                        T.f32,
                        align=4,
                    )
                )
                for v in range_constexpr(4)
            ]
        for J in range_constexpr(numAccN):
            col = wave * fx.Int32(wave_n) + J * 16 + lane_mod_16
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                if const_expr(g2_bf16_lds):
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
                else:
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    fx.barrier()

    # ``buf_copy_store``'s cache_modifier is a BufferCopy enum (0=cached, 2=nt),
    # so sc0|sc1 cannot be reached through it -- flydsl's own type stub says so,
    # and so does mori/ops/gemm_ar/_compat.py. ``buffer_store`` passes the value
    # through as the instruction's aux field, so the write-through path uses it.
    if const_expr(store_cm or quant_fp8):
        output_buf = create_buffer_resource_from_addr(
            send_base, num_records_bytes=send_nbytes
        )
    else:
        output_buf = ptr_buf_tensor(
            send_base,
            fx.BFloat16,
            unit_elems=scatter_vec,
            num_records_bytes=send_nbytes,
        )
    # Inactive lanes read safe LDS and issue a bounded OOB store.
    active = lane < fx.Int32(BN // scatter_vec)
    col = active.select(lane * fx.Int32(scatter_vec), fx.Int32(0))

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        row_byte_off = row * fx.Int32(4)
        out_row = fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_outrow_off) + row_byte_off, T.i32, align=4)
        )
        out_row = rocdl.readfirstlane(T.i32, out_row.ir_value())
        idx0 = row * fx.Int32(BN) + col
        if const_expr(g2_bf16_lds):
            pk = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(2),
                    fx.Vector.make_type(scatter_vec, fx.BFloat16),
                    fx.BFloat16,
                    align=16,
                )
            )
        else:
            weight = fx.ptr_load(
                lds_typed_ptr(fx.Int32(lds_weight_off) + row_byte_off, T.f32, align=4)
            )
            weight = rocdl.readfirstlane(T.f32, weight.ir_value())
            v8 = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(4),
                    fx.Vector.make_type(scatter_vec, fx.Float32),
                    fx.Float32,
                    align=16,
                )
            )
            weighted = fx.Vector.from_elements(
                [v8[i] * weight for i in range_constexpr(scatter_vec)],
                fx.Float32,
            )
            if const_expr(not quant_fp8):
                pk = weighted.to(fx.BFloat16)
        row_base = out_row * fx.Int32(row_nbytes)
        valid = (out_row >= fx.Int32(0)) & active
        if const_expr(quant_fp8):
            # One e8m0 per 32 contiguous values, i.e. per 32//scatter_vec lanes.
            vals = [fx.Float32(weighted[i]) for i in range_constexpr(scatter_vec)]
            local_max = fabs_f32(vals[0])
            for q in range_constexpr(1, scatter_vec):
                local_max = local_max.maximumf(fabs_f32(vals[q]))
            max_bits = local_max.bitcast(fx.Int32)
            for xor_lane in (1, 2):
                if xor_lane < 32 // scatter_vec:
                    remote = rocdl.ds_bpermute(
                        T.i32, (lane ^ fx.Int32(xor_lane)) * fx.Int32(4), max_bits
                    )
                    local_max = local_max.maximumf(fx.Int32(remote).bitcast(fx.Float32))
                    max_bits = local_max.bitcast(fx.Int32)
            group_lanes = 32 // scatter_vec
            leader_lane = lane & fx.Int32(~(group_lanes - 1))
            is_leader = (lane & fx.Int32(group_lanes - 1)) == fx.Int32(0)
            e8m0 = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32,
                    leader_lane * fx.Int32(4),
                    _fp8_scale_for_leader(is_leader, local_max),
                )
            )
            block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            pk_ty = T.vec(2, T.i16)
            words = []
            for word in range_constexpr(scatter_vec // 4):
                packed = fx.Vector.filled(2, 0, fx.Int16).ir_value()
                for pair in range_constexpr(2):
                    e = word * 4 + pair * 2
                    packed = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty, packed, vals[e].ir_value(), vals[e + 1].ir_value(),
                        block_scale.ir_value(), pair,
                    )
                words.append(fx.Vector(packed).bitcast(fx.Int32)[0])
            payload = fx.Vector.from_elements(words, fx.Int32)
            off = valid.select(
                row_base + n_block_idx * fx.Int32(BN) + col, fx.Int32(send_nbytes)
            )
            buffer_store(
                payload, output_buf, off,
                cache_modifier=store_cm, offset_is_bytes=True,
            )
            # Only the lane that owns a 32-value block writes its scale byte.
            scale_off = (valid & is_leader).select(
                row_base + fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN // 32)
                + lane // fx.Int32(group_lanes),
                fx.Int32(send_nbytes),
            )
            buffer_store(
                e8m0.to(fx.Int8), output_buf, scale_off,
                cache_modifier=store_cm, offset_is_bytes=True,
            )
        else:
            off = valid.select(
                row_base + n_block_idx * fx.Int32(BN * 2) + col * fx.Int32(2),
                fx.Int32(send_nbytes),
            )
            if const_expr(store_cm):
                buffer_store(pk, output_buf, off, cache_modifier=store_cm, offset_is_bytes=True)
            else:
                buf_copy_store(
                    output_buf,
                    off // fx.Int32(scatter_vec * 2),
                    pk,
                    fx.BFloat16,
                    unit_elems=scatter_vec,
                    cache_modifier=0,
                )


# fmt: off
def compile_moe_gemm2_localsort(*, model_dim: int, inter_dim: int, experts: int, rank: int,
    send_nbytes: int, BM: int = 32, BN: int = 256, BK: int = 256, use_nt: bool = True,
    HIDDEN_MAX: int = 8192, INTER_MAX: int = 8192, a_dtype: str = "fp8", SBM: int | None = None,
    persist: bool = False, cu_num: int = 0, has_pad: bool = False, g2_bhoist=None,
    g2_ascale_pf=None, g2_spart=None, persist_strided: bool = False, g2_bf16_lds: bool = False,
    skew_cu: int = 0, scatter_vec: int = 8, comm=None, quant: str = "none"):
# fmt: on
    """Compile GEMM2 with the rank-sorted local-output epilogue (optionally fused with SDMA)."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage2 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    assert model_dim % BN == 0 and HIDDEN_MAX % BN == 0
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}"
    if BM not in (16, 32, 64, 128):
        raise AssertionError(f"BM must be in {{16,32,64,128}}, got {BM}")
    SBM = BM if SBM is None else int(SBM)
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    if scatter_vec not in (8, 16):
        raise ValueError("Stage2 scatter_vec must be 8 or 16")
    if BN % scatter_vec:
        raise ValueError("Stage2 scatter_vec must divide BN")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    if skew_cu and (not persist or not 0 < skew_cu <= cu_num):
        raise AssertionError(f"skew_cu={skew_cu} requires persist=True and 0<skew_cu<=cu_num={cu_num}")
    if not 0 < send_nbytes < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE v2 stage2 send buffer exceeds the 32-bit buffer-resource ABI")
    fuse = comm is not None
    NC = comm.comm_wgs if fuse else 0
    # comm_wgs == 0 is a diagnostic build: the GEMM units still publish each chunk
    # (s_waitcnt + barrier + atomic), but no comm wave is traced and nothing ships.
    if fuse and (comm.comm_wgs < 1 or comm.chunk_tiles < 1):
        raise ValueError("fused comm needs comm_wgs >= 1 and chunk_tiles >= 1")
    # The rows are stored sc0|sc1 -- past L1 (GLC) and past L2 (SCC) -- so they
    # are in memory when the store retires and the ``s_waitcnt vmcnt(0)`` in
    # front of the publish is the release the copy engine needs. Leaving them in
    # L2 and issuing a system-scope release instead costs a whole-L2 write-back
    # per unit (``buffer_wbl2`` on gfx950): measured 2.4x on the GEMM2 kernel.
    # See mori/ops/gemm_ar/_compat.py for the encoding, and why sc1 is 16 not 4.
    STORE_CM = 1 | 16 if fuse else 0
    N_OUT = model_dim
    g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, _g2_bf16_lds = _resolve_g2_knobs(
        g2_bhoist, g2_ascale_pf, g2_spart, False, False
    )
    is_f8 = a_dtype == "fp8"
    aStages = kStages + 1
    KH_TILE_A = BK // (1 if is_f8 else 2)
    compute_lds_bytes = _stage2_lds_bytes(BM, BN, BK, a_dtype, aStages, g2_bf16_lds)
    lds_outrow_off = compute_lds_bytes
    lds_weight_off = lds_outrow_off + BM * 4
    lds_bytes = lds_weight_off + BM * 4
    if lds_bytes > 160 * 1024:
        raise ValueError(f"Stage2 LDS use {lds_bytes} exceeds 160 KiB")
    _expert_offset = rank * experts

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    kernel_name = (
        f"megamoe_stage2_localsort_t{BM}x{BN}x{BK}"
        f"_sbm{SBM}_{a_dtype}_nt{int(use_nt)}"
        f"_p{int(persist)}cu{cu_num}s{int(persist_strided)}_pad{int(has_pad)}"
        f"_sk{skew_cu}"
        f"_bh{int(g2_bhoist)}apf{int(g2_ascale_pf)}sp{g2_group_num}x{g2_m01}"
        f"_bf16lds{int(g2_bf16_lds)}_sv{scatter_vec}_{quant}"
        + (
            f"_fuse{comm.chunk_tiles}q{comm.queues}w{comm.comm_wgs}"
            f"sm{comm.signal_mode[0]}r{comm.rank}"
            if fuse
            else ""
        )
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel_localsort(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_send: fx.Int64, arg_out_row: fx.Int64,
        arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_cdone: fx.Int64, arg_tile_base: fx.Int64,
        arg_cursor: fx.Int64, i32_max_m_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32,
        i32_kpad: fx.Int32, i32_npad: fx.Int32):
    # fmt: on
        tx_i32 = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(1 if is_f8 else 2)
        trb_buf = ptr_buf_tensor(arg_trb, fx.Int32)
        sweights_buf = ptr_buf_tensor(arg_sweights, fx.Float32)
        outrow_buf = ptr_buf_tensor(arg_out_row, fx.Int32)

        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_row0, wave, lane,
                    is_f8, KH_TILE_A, k_bytes, BM=BM)

        def run_unit(unit_bx, m_block_idx):
            # Map each Stage2 BM sub-tile to its Stage1 SBM metadata row.
            m_row = m_block_idx * fx.Int32(BM)
            sort_block_idx = m_row // fx.Int32(SBM)
            row_in_sort_block = m_row - sort_block_idx * fx.Int32(SBM)
            srcmap_row_base = trb_buf[sort_block_idx] + row_in_sort_block
            if tx_i32 < fx.Int32(BM):
                sorted_pos = srcmap_row_base + tx_i32
                fx.ptr_store(
                    outrow_buf[m_row + tx_i32],
                    lds_typed_ptr(
                        fx.Int32(lds_outrow_off) + tx_i32 * fx.Int32(4),
                        T.i32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    sweights_buf[sorted_pos],
                    lds_typed_ptr(
                        fx.Int32(lds_weight_off) + tx_i32 * fx.Int32(4),
                        T.f32,
                        align=4,
                    ),
                )
            # fmt: off
            accm_vecs, m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
                arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
                i32_kpad, i32_npad, BM=BM, BN=BN, BK=BK, use_nt=use_nt, INTER_MAX=INTER_MAX, aStages=aStages,
                a_dtype=a_dtype, has_pad=has_pad, SBM=SBM, g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
                expert_offset=_expert_offset)
            send_base = rocdl.readfirstlane(T.i64, fx.Int64(arg_send).ir_value())
            local_sort_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, send_base,
                N_OUT=N_OUT, BM=BM, BN=BN, send_nbytes=send_nbytes, lds_outrow_off=lds_outrow_off,
                lds_weight_off=lds_weight_off, g2_bf16_lds=g2_bf16_lds, scatter_vec=scatter_vec,
                store_cm=STORE_CM, quant=quant)
            # fmt: on

        def run_one(unit_bx, m_block_idx):
            issue_all_a_loads(m_block_idx * fx.Int32(BM))
            rocdl.sched_barrier(0)
            run_unit(unit_bx, m_block_idx)
            if const_expr(fuse):
                # The rows went past L2 on the way out, so retiring the stores
                # is all the release the copy engine needs; the barrier makes it
                # the whole workgroup's stores, not just this wave's.
                rocdl.s_waitcnt(vmcnt=0)
                fx.barrier()
                if tx_i32 == fx.Int32(0):
                    chunk = (m_block_idx * fx.Int32(BM)) // fx.Int32(SBM * comm.chunk_tiles)
                    comm_ops.atomic_add_agent(
                        fx.Int64(arg_cdone) + fx.Int64(chunk) * fx.Int64(CDONE_STRIDE * 4),
                        fx.Int32(1),
                    )

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = (cumsum0 + fx.Int32(BM - 1)) // fx.Int32(BM)

        def gemm_role(bx_i32):

            if const_expr(not persist and g2_spart <= 0):
                bound = total_m_blocks * fx.Int32(num_n_blocks)
                if fx.Int32(bx_i32) < bound:
                    run_one(bx_i32, bx_i32 // num_n_blocks)
            elif const_expr(not persist):
                bound = total_m_blocks * fx.Int32(num_n_blocks)
                if fx.Int32(bx_i32) < bound:
                    m_block_idx, n_block_idx = _spart_output_tile_index(
                        bx_i32, total_m_blocks, num_n_blocks, g2_group_num, g2_m01
                    )
                    unit_bx = m_block_idx * fx.Int32(num_n_blocks) + n_block_idx
                    run_one(unit_bx, m_block_idx)
            elif const_expr(skew_cu > 0):
                total_stage1_tiles = (cumsum0 + fx.Int32(SBM - 1)) // fx.Int32(SBM)
                max_expert_tiles = global_typed_ptr(arg_max_expert_tiles, T.i32)[0]
                skewed = max_expert_tiles * fx.Int32(4) > total_stage1_tiles
                skew_n_block = bx_i32 // fx.Int32(cu_num)
                skew_m_slot = bx_i32 - skew_n_block * fx.Int32(cu_num)
                normal_m_slot = bx_i32 // fx.Int32(num_n_blocks)
                normal_n_block = bx_i32 - normal_m_slot * fx.Int32(num_n_blocks)
                n_block = skewed.select(skew_n_block, normal_n_block)
                m_slot = skewed.select(skew_m_slot, normal_m_slot)
                active_cu = skewed.select(fx.Int32(skew_cu), fx.Int32(cu_num))
                strided_diff = total_m_blocks - m_slot
                strided_rem = (strided_diff > fx.Int32(0)).select(strided_diff, fx.Int32(0))
                strided_iters = (strided_rem + active_cu - fx.Int32(1)) // active_cu
                tiles_per_slot = (
                    total_m_blocks + active_cu - fx.Int32(1)
                ) // active_cu
                m_tile0 = m_slot * tiles_per_slot
                contiguous_diff = total_m_blocks - m_tile0
                contiguous_rem = (contiguous_diff > fx.Int32(0)).select(
                    contiguous_diff, fx.Int32(0)
                )
                contiguous_iters = (contiguous_rem < tiles_per_slot).select(
                    contiguous_rem, tiles_per_slot
                )
                n_iters = skewed.select(strided_iters, contiguous_iters)
                active = m_slot < active_cu
                for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
                    strided_m = m_slot + fx.Int32(_it) * active_cu
                    contiguous_m = m_tile0 + fx.Int32(_it)
                    m_block = skewed.select(strided_m, contiguous_m)
                    if active:
                        unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                        fx.barrier()
                        if m_block < total_m_blocks:
                            run_one(unit_bx, m_block)
            else:
                m_slot = bx_i32 // fx.Int32(num_n_blocks)
                n_block = bx_i32 - m_slot * fx.Int32(num_n_blocks)
                if const_expr(persist_strided):
                    diff = total_m_blocks - m_slot
                    rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
                    n_iters = (rem + fx.Int32(cu_num - 1)) // fx.Int32(cu_num)
                else:
                    tiles_per_slot = (
                        total_m_blocks + fx.Int32(cu_num - 1)
                    ) // fx.Int32(cu_num)
                    m_tile0 = m_slot * tiles_per_slot
                    diff = total_m_blocks - m_tile0
                    rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
                    n_iters = (rem < tiles_per_slot).select(rem, tiles_per_slot)
                for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
                    if const_expr(persist_strided):
                        m_block = m_slot + fx.Int32(_it) * fx.Int32(cu_num)
                    else:
                        m_block = m_tile0 + fx.Int32(_it)
                    unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                    fx.barrier()  # separate prev-iter epilog LDS reads from this iter's A-load into the LDS union
                    if fx.Int32(m_block) < total_m_blocks:
                        run_one(unit_bx, m_block)

        def comm_role(cw):
            import mori.cco.device.flydsl as cco

            lay, rank, npes, CT = comm.layout, comm.rank, comm.layout.npes, comm.chunk_tiles
            n_waves = NC * 4
            n_mine = (npes + n_waves - 1) // n_waves
            mb_per_tile = SBM // BM
            # This wave owns peers owner, owner + n_waves, ...; a wave past the last
            # peer has nothing to do.
            owner = cw * fx.Int32(4) + wave
            is_lane0 = lane == fx.Int32(0)
            if owner < fx.Int32(npes):
                w = cco.Window(arg_win)
                sdma = cco.DevComm(arg_dev_comm).sdma()
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(rank, lay.off_counter))
                )
                # Read-only here: the combine kernel advances the counter after the call,
                # so every comm wave sees the same phase whenever it starts.
                phase = fx.Int32(buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)) + fx.Int32(1)
                cursor_buf = ptr_buf_tensor(arg_cursor, fx.Int32)
                tile_base_buf = ptr_buf_tensor(arg_tile_base, fx.Int32)
                # Do not overwrite a peer's landing buffer until it has read the last call's.
                for j in range_constexpr(n_mine):
                    peer = owner + fx.Int32(j * n_waves)
                    if peer < fx.Int32(npes):
                        if peer != fx.Int32(rank):
                            if is_lane0:
                                sdma_wait_consumed(w, peer, phase, layout=lay, rank=rank)
                total_tiles = (cumsum0 + fx.Int32(SBM - 1)) // fx.Int32(SBM)
                n_chunks = (total_tiles + fx.Int32(CT - 1)) // fx.Int32(CT)
                for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                    ci = fx.Int32(c)
                    m_lo = ci * fx.Int32(CT * mb_per_tile)
                    m_hi = m_lo + fx.Int32(CT * mb_per_tile)
                    m_hi = (m_hi < total_m_blocks).select(m_hi, total_m_blocks)
                    # Every (m_block, n_block) unit of this chunk has stored.
                    if is_lane0:
                        comm_ops.spin_until_ge_i32_agent(
                            fx.Int64(arg_cdone) + fx.Int64(ci) * fx.Int64(CDONE_STRIDE * 4),
                            (m_hi - m_lo) * num_n_blocks,
                        )
                    comm_ops.fence_agent_acquire()
                    t0 = ci * fx.Int32(CT)
                    t1 = t0 + fx.Int32(CT)
                    t1 = (t1 < total_tiles).select(t1, total_tiles)
                    for j in range_constexpr(n_mine):
                        peer = owner + fx.Int32(j * n_waves)
                        if peer < fx.Int32(npes):
                            if peer != fx.Int32(rank):
                                lo = tile_base_buf[t0 * fx.Int32(npes) + peer]
                                hi = tile_base_buf[t1 * fx.Int32(npes) + peer]
                                if hi > lo:
                                    if is_lane0:
                                        sdma_put_run(sdma, arg_win, peer,
                                                     peer % fx.Int32(comm.queues),
                                                     layout=lay, rank=rank, lo=lo, hi=hi, cco=cco)
                for j in range_constexpr(n_mine):
                    peer = owner + fx.Int32(j * n_waves)
                    if peer < fx.Int32(npes):
                        if peer != fx.Int32(rank):
                            if is_lane0:
                                sdma_finish_peer(sdma, w, arg_win, peer,
                                                 peer % fx.Int32(comm.queues),
                                                 cursor_buf[peer], phase, layout=lay, rank=rank,
                                                 cco=cco, signal_mode=comm.signal_mode)

        if const_expr(fuse):
            if bid < fx.Int32(NC):
                comm_role(bid)
            else:
                gemm_role(bid - fx.Int32(NC))
        else:
            gemm_role(bid)

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_send: fx.Int64, arg_out_row: fx.Int64,
        arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_cdone: fx.Int64, arg_tile_base: fx.Int64,
        arg_cursor: fx.Int64, i32_max_m_blocks: fx.Int32, i32_grid_blocks: fx.Int32, i32_inter: fx.Int32,
        i32_hidden: fx.Int32, i32_kpad: fx.Int32, i32_npad: fx.Int32, stream: fx.Stream):
    # fmt: on
        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        grid_x = i32_grid_blocks * num_n_blocks + fx.Int32(NC)
        kernel_localsort(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_max_expert_tiles,
            arg_sweights, arg_trb, arg_send, arg_out_row, arg_dev_comm, arg_win, arg_cdone,
            arg_tile_base, arg_cursor, i32_max_m_blocks, i32_inter, i32_hidden, i32_kpad, i32_npad,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_LAUNCH_CACHE = {}


# fmt: off
def run_moe_gemm2_localsort(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
    arg_max_expert_tiles, arg_sweights, arg_trb, arg_send, arg_out_row, arg_dev_comm, arg_win,
    arg_cdone, arg_tile_base, arg_cursor, row_capacity, i32_inter, i32_hidden, stream, *,
    model_dim, inter_dim, experts, rank, send_nbytes, BM, SBM, HIDDEN_MAX, INTER_MAX, cu_num,
    BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False,
    persist_cu=0, persist_strided=False, g2_bf16_lds=False, skew_cu=0, scatter_vec=8, comm=None,
    quant="none"):
    # fmt: on
    """Compile or reuse one configuration and launch it."""
    launch_cu_num = min(cu_num, persist_cu) if persist and persist_cu > 0 else cu_num
    compile_kw = dict(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, rank=rank,
        send_nbytes=send_nbytes, BM=BM, BN=BN, BK=BK, use_nt=use_nt, HIDDEN_MAX=HIDDEN_MAX,
        INTER_MAX=INTER_MAX, SBM=SBM, persist=persist, cu_num=launch_cu_num,
        g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf, g2_spart=g2_spart,
        persist_strided=persist_strided, g2_bf16_lds=g2_bf16_lds, skew_cu=skew_cu,
        scatter_vec=scatter_vec, comm=comm, quant=quant,
    )
    key = tuple(sorted(compile_kw.items()))
    launch = _LAUNCH_CACHE.get(key)
    if launch is None:
        launch = _LAUNCH_CACHE[key] = compile_moe_gemm2_localsort(**compile_kw)
    max_m_blocks = (row_capacity + BM - 1) // BM
    grid_blocks = launch_cu_num if persist else max_m_blocks
    _run_compiled(
        launch, arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
        arg_max_expert_tiles, arg_sweights, arg_trb, arg_send, arg_out_row, arg_dev_comm, arg_win,
        arg_cdone, arg_tile_base, arg_cursor, fx.Int32(max_m_blocks), fx.Int32(grid_blocks), fx.Int32(i32_inter), fx.Int32(i32_hidden),
        fx.Int32(0), fx.Int32(0), stream,
    )
