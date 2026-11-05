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
"""The persistent MegaMoE kernel: dispatch, GEMM1, GEMM2 and the SDMA sends in one launch.

One workgroup per CU (``num_cu`` of them), 8 waves each. Every workgroup draws a ticket:

* ticket 0, the owner -- the launch handshake with the peers, the per-launch resets, then
  the dispatch plan (``dispatch.emit_dispatch_plan``): per local expert the rows each source
  sends, the sorted row layout (``expert_tile_end``), and each source's base row there;
* tickets 1..``dispatch_blocks``, the producers -- group this rank's routes per destination
  expert, then copy the token rows to the destinations over P2P
  (``emit_dispatch_group`` / ``emit_dispatch_payload``); a sort tile is ready once its rows
  have all arrived (``TILE_READY`` reaches ``TILE_EXPECTED``);
* every workgroup but the last then takes GEMM1 units (256 sorted rows x 256 of the
  ``2 * inter`` columns, 8-wave A8W4 + SwiGLU) off a shared counter. A unit waits for its
  sort tiles' payload; when done it counts once toward each of its sort tiles;
* once the GEMM1 counter runs dry a workgroup takes GEMM2 units (256 rows x 256 of the
  ``model_dim`` columns). A unit waits until its sort tiles' GEMM1 units are all done, so
  GEMM2 overlaps the GEMM1 tail. Its rows go to a local send buffer ``S`` grouped by origin
  rank (``sdma_parts``), at ``sorted row + D[expert][origin]``: within an expert the rows are
  grouped by origin rank, so that offset table is the whole Stage2 plan;
* the last workgroup builds that table (from the count matrix, after the dispatch plan) and
  is then the comm workgroup: one wave per peer puts each finished chunk of sort tiles of its
  run with one SDMA copy, the last one signalled.

The combine (``sdma_parts.build_combine``) is a separate kernel on the same stream.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

import flyutil as comm_ops
from dispatch import (
    DispatchSlot,
    _wave_uniform_i64,
    emit_dispatch_group,
    emit_dispatch_payload,
    emit_dispatch_plan,
)
from flyutil import (
    G2SLoader,
    S2RLoader,
    _buffer_load,
    _buffer_store,
    _make_buffer_from_addr,
    _wave_inclusive_scan_i32,
    buffer_load,
    buffer_store,
    compute_global_swizzle,
    create_buffer_resource_from_addr,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
    make_fp8_buffer_tensor,
    ptr_buf_tensor,
    wait_barrier,
)
from gemm_parts import BLOCK_K as G1_BLOCK_K
from gemm_parts import BLOCK_M as G1_BLOCK_M
from gemm_parts import BLOCK_N as G1_BLOCK_N
from gemm_parts import SEND_CM as G2_SEND_CM
from gemm_parts import (
    MxMfma,
    S2RF4G,
    ScaleLdsWordS2R,
    ScalePreshuffledS2R,
    lds_barrier,
    opaque_i32,
    tile_find,
    tile_table,
    tile_table_to_lds,
)
from gemm_parts import epi_quant_half as g2_epi_quant_half
from sdma_parts import _decode as s2_decode
from sdma_parts import (
    CDONE_STRIDE,
    INV_P2P,
    OUT_ROW_P2P,
    ArenaLayout,
    publish_flag,
    sdma_put_run,
    sdma_signal_peer,
    sdma_wait_consumed,
)

__all__ = ["compile_mega_moe", "sdma_layout", "PTR_TABLE"]

#: Slots of the kernel's int64 pointer table (``ptr_tab``).
PTR_TABLE = ("win", "dev_comm", "send", "cdone", "plan")
#: Sort tiles per shipped chunk; ``S`` rows reserved per (source, destination) pair over the even share.
CHUNK_TILES = 16
ROW_SLACK = 2.0
G1_OUT_CACHE = 17  # GEMM1 output stores: write-through, read by GEMM2 on any XCD
_SC0_CACHE = 1
LDS_LIMIT_BYTES = 160 * 1024


def sdma_layout(*, npes, mtpr, topk, model_dim):
    """The cco window layout of the send / landing buffers (``sdma_parts.ArenaLayout``)."""
    share = -(-(mtpr * topk) // npes)
    cap = -(-int(share * ROW_SLACK) // 128) * 128
    return ArenaLayout(npes, cap, model_dim + model_dim // 32, mtpr * topk)


def compile_mega_moe(
    *,
    model_dim,
    inter_dim,
    rank,
    experts_per_rank,
    fuse_npes,
    fuse_topk,
    fuse_mtpr,
    num_valid_max,
    num_cu,
    swiglu_limit,
    sort_block_m=128,
    num_dispatch_cu=96,
    payload_chunk_rows=384,
    work_shards=4,
):
    """Build the launcher of the persistent kernel for one rank (shapes are compile-time)."""
    NUM_WAVES = 8
    TOTAL_THREADS = NUM_WAVES * 64
    assert sort_block_m == 128 and model_dim % 256 == 0 and inter_dim % 256 == 0
    G1_NB = (2 * inter_dim) // 256  # GEMM1 units per 256-row M-tile
    G1_PUB_TARGET = G1_NB  # GEMM1 completions a sort tile needs
    grid_epoch_slot = 0
    dispatch_blocks = int(num_dispatch_cu)
    payload_chunk_rows = int(payload_chunk_rows)
    assert dispatch_blocks % fuse_npes == 0 and payload_chunk_rows % sort_block_m == 0
    launch_grid_x = int(num_cu)
    assert 1 + dispatch_blocks < launch_grid_x - 1
    WORK_SHARDS = int(work_shards)
    G2_SHARDS = WORK_SHARDS
    # [0, MAX_SORT_TILES): GEMM1 completions per sort tile; then the GEMM2 claim heads, one
    # 64-byte line per shard (only shard 0's is used here).
    MAX_SORT_TILES = (num_valid_max + sort_block_m - 1) // sort_block_m + 1
    G2_HEAD_BYTE_OFF = (MAX_SORT_TILES * 4 + 63) // 64 * 64
    G2_LAY = sdma_layout(
        npes=fuse_npes, mtpr=fuse_mtpr, topk=fuse_topk, model_dim=model_dim
    )
    G2_NB2 = model_dim // 256  # GEMM2 units per 256-row M-tile
    G2_CHUNK = CHUNK_TILES
    G2_NCH = (MAX_SORT_TILES + G2_CHUNK - 1) // G2_CHUNK + 1
    G2_P = experts_per_rank * fuse_npes
    assert G2_P <= TOTAL_THREADS and fuse_npes <= NUM_WAVES and experts_per_rank <= 64

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_mtpr, fz_rank = int(fuse_mtpr), int(rank)
    assert fz_npes * fz_mtpr <= 1 << 24 and fz_k <= 1 << 8, "srcmap word encoding"
    fz_tile_m = int(sort_block_m)
    fz_total_experts = fz_npes * fz_epr
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_n_i32 = (model_dim // 32 + 3) // 4
    fz_enable_scales = True
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512

    G1_E_STAGE = max(128 * 256, 256 * (model_dim // 32))
    G1_A_LDS = (G1_BLOCK_M // 2) * G1_BLOCK_K
    G1_B_LDS = (G1_BLOCK_N // 2) * (G1_BLOCK_K // 2)
    CTRL_OFF = 32  # control words in A_scale (bytes 0..31 are scratch)

    @fx.struct
    class SharedStorage:
        # The largest field sits at LDS 0: the ticket scratch at the start of the launch.
        E_stage: fx.Array[fx.Float8E4M3FN, G1_E_STAGE, 16]
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, G1_A_LDS, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, G1_A_LDS, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, G1_A_LDS, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, G1_A_LDS, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, G1_B_LDS, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, G1_B_LDS, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, G1_B_LDS, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, G1_B_LDS, 16]
        E_tab: fx.Array[fx.Float8E4M3FN, 4 * 64 * 4 + 16, 16]
        A_scale: fx.Array[fx.Int8, 64, 16]
        # GEMM2 epilogue: e8m0 of a 128-row half, out_row / srcmap word / weight per row.
        E_scale: fx.Array[fx.Float8E4M3FN, 128 * 8, 16]
        E_orow: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]
        E_src: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]
        E_wts: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]

    swiglu_suffix = f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    kernel_name = (
        f"mori_ex14_mega_moe_r{rank}p{fz_npes}e{fz_epr}k{fz_k}m{fz_mtpr}"
        f"_h{model_dim}i{inter_dim}_dcu{dispatch_blocks}_cu{launch_grid_x}{swiglu_suffix}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        x: fx.Tensor,
        w: fx.Tensor,
        scale_w: fx.Tensor,
        out: fx.Tensor,
        out_scale: fx.Tensor,
        xw2: fx.Tensor,
        xw2s: fx.Tensor,
        tokens: fx.Int32,
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64,
        addr_in_wts: fx.Int64,
        addr_in_sc: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_mtile_ctr: fx.Int64,
        s2_aq: fx.Int64,
        s2_ascale: fx.Int64,
        scale_raw: fx.Int64,
        s2_stids: fx.Int64,
        s2_sweights: fx.Int64,
        s2_trb: fx.Int64,
        s2_p2p_comb_inp: fx.Int64,
        ptr_tab: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.E_stage

        def _g1_build(
            A,
            B_T,
            B_scale,
            g1_out,
            g1_out_scale,
            g1_scale_raw,
            tib,
            g1_tokens,
            g1_lds,
            g1_tile_ready,
            g1_tile_expected,
        ):
            """The 8-wave GEMM1 + SwiGLU item on the sorted payload: A = the payload rows, their raw
            e8m0 rows at ``g1_scale_raw``, sort tile bases ``tib`` (tile_row_base); output rows and
            shuffled e8m0 into the Stage1 output (write-through, for GEMM2 on any XCD). Each item
            stages its M-tile's A scales into LDS in the shuffled word layout."""
            # Flat 1-D views: the loaders index A / B with linear byte offsets, and on the
            # [rows, K] / [experts, N, K/2] tensors logical_divide would read them as coordinates.
            A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(2**31 - 1, 1)))
            B_T = fx.Tensor(
                fx.make_view(
                    fx.get_iter(B_T),
                    fx.make_layout(fz_epr * 2 * inter_dim * (model_dim // 2), 1),
                )
            )
            K = model_dim
            N = 2 * inter_dim
            SBM = sort_block_m
            experts = fz_epr
            INTER = N // 2
            SC_COLS32 = (INTER // 32 + 7) // 8 * 8 * 32
            K_ITERS = K // G1_BLOCK_K
            BLOCK_M, BLOCK_N, BLOCK_K = G1_BLOCK_M, G1_BLOCK_N, G1_BLOCK_K
            N_TILES_A = BLOCK_M // 64
            N_TILES_B = BLOCK_N // 128
            N_ACCUMS = N_TILES_A * N_TILES_B
            LDS_BLOCK_M = BLOCK_M // 2
            LDS_BLOCK_N = BLOCK_N // 2
            N_LDS_STEPS_A = LDS_BLOCK_M // 64
            B_ROW_BYTES = BLOCK_K // 2
            N_LDS_STEPS_B = LDS_BLOCK_N * B_ROW_BYTES // (8 * 1024)
            N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
            A_GRP_ROWS = N_TILES_A * 16
            B_GRP_ROWS = N_TILES_B * 16
            B_K_STEP = 1024
            n_blocks = fx.Int32(N // BLOCK_N)
            F8_IR_t = fx.Float8E4M3FN.ir_type
            g1_scale_bytes = g1_tokens * fx.Int32(
                (INTER // 32 + 7) // 8 * 8
            ) + fx.Int32(8192)

            def lds_addr(arr):
                return fx.Int32(fx.ptrtoint(arr.ptr))

            def lane_ids():
                # Recomputed per work item and per epilogue (see opaque_i32).
                tid = opaque_i32(fx.thread_idx.x)
                wave_id = tid // fx.Int32(64)
                return (
                    tid,
                    tid % fx.Int32(64),
                    wave_id,
                    wave_id // fx.Int32(4),
                    wave_id % fx.Int32(4),
                )

            def process_tile(block_m, block_n, tab_lds):
                _, lane_id, wave_id, wave_m, wave_n = lane_ids()
                a_cur0, a_cur1 = g1_lds.A_lds_cur_0, g1_lds.A_lds_cur_1
                a_next0, a_next1 = g1_lds.A_lds_next_0, g1_lds.A_lds_next_1
                b_cur0, b_cur1 = g1_lds.B_lds_cur_0, g1_lds.B_lds_cur_1
                b_next0, b_next1 = g1_lds.B_lds_next_0, g1_lds.B_lds_next_1
                row_base, expert, n_valid = tile_find(
                    tab_lds, block_m, lane_id, SBM=SBM, experts=experts
                )
                b_row_block = expert * n_blocks + block_n
                tid0 = fx.Int32(fx.thread_idx.x) == fx.Int32(0)
                # The M-tile's sort tiles (SBM rows each) have landed once TILE_READY reaches
                # TILE_EXPECTED for each of them.
                st0 = row_base // fx.Int32(SBM)
                if tid0:
                    for i in range_constexpr(256 // SBM):
                        if n_valid > fx.Int32(i * SBM):
                            t_i = st0 + fx.Int32(i)
                            exp_i = fx.Int32(
                                global_typed_ptr(g1_tile_expected, T.i32)[t_i]
                            )
                            comm_ops.wait_i32_until_equals(
                                g1_tile_ready + fx.Int64(t_i) * fx.Int64(4), exp_i
                            )
                fx.barrier()
                comm_ops.fence_system_acquire()
                tib_p = global_typed_ptr(tib, T.i32)

                B0_gl_offset = (b_row_block * BLOCK_N) * (K // 2)
                B1_gl_offset = (b_row_block * BLOCK_N + LDS_BLOCK_N) * (K // 2)
                a_div = fx.logical_divide(
                    make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
                )
                b_div = fx.logical_divide(
                    make_fp8_buffer_tensor(B_T, F8_IR_t), fx.make_layout(1, 1)
                )
                gl_off_a = compute_global_swizzle(
                    lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
                )
                # a16w4 layout: wave w copies 16-row group w's contiguous 1 KB of a K step.
                gl_off_b = [wave_id * (16 * (K // 2)) + lane_id * 16]

                mfma = MxMfma(N_TILES_A, N_TILES_B)
                # Route-major payload: an SBM tile's rows are contiguous, so a lane's offset is
                # the contiguous one (gl_off_a) and the gather is a uniform per (half, step)
                # base in soffset -- no extra VGPRs.
                a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
                a_soff = []
                for h in range_constexpr(2):
                    per_step = []
                    for step in range_constexpr(N_LDS_STEPS_A):
                        g0 = row_base + fx.Int32(h * LDS_BLOCK_M + step * 64)
                        st = g0 // fx.Int32(SBM)
                        rel = row_base + fx.Int32(h * LDS_BLOCK_M) - st * fx.Int32(SBM)
                        per_step.append(
                            fx.Int32(
                                rocdl.readfirstlane(
                                    T.i32, (fx.Int32(tib_p[st]) + rel) * fx.Int32(K)
                                )
                            )
                        )
                    a_soff.append(per_step)

                def load_a(h, lds_dst, k_byte):
                    for step in range_constexpr(N_LDS_STEPS_A):
                        a_g2s.load_one(
                            lds_dst, a_soff[h][step] + fx.Int32(k_byte), step
                        )

                b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
                a_s2r = S2RLoader(wave_m, N_TILES_A)
                a_s2r.lane_id = lane_id
                b_s2r = S2RF4G(wave_n, N_TILES_B, lane_id)
                a_sc = ScaleLdsWordS2R(N_TILES_A, lane_id, K)
                b_sc = ScalePreshuffledS2R(B_scale, experts * N, K, N_TILES_B, lane_id)
                # 16-row tile index of each wave's first row, per LDS half.
                a_base16 = [
                    (row_base + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
                    for h in range_constexpr(2)
                ]
                b_base16 = [
                    (b_row_block * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS)
                    // 16
                    for h in range_constexpr(2)
                ]
                sc_lds = lds_addr(g1_lds.E_stage)
                # The M-tile's scales into LDS in the shuffled word layout; the waves then
                # address it with tile-local 16-row indices.
                tib_bases = [
                    fx.Int32(
                        rocdl.readfirstlane(
                            T.i32,
                            fx.Int32(tib_p[row_base // fx.Int32(SBM) + fx.Int32(i)]),
                        )
                    )
                    for i in range_constexpr(256 // SBM)
                ]
                # Loads only; written to LDS after the prologue's A/B loads are issued.
                sc_staged = a_sc.load_wide(
                    g1_scale_raw,
                    tib_bases,
                    SBM,
                    n_valid,
                    wave_id * fx.Int32(64) + lane_id,
                )
                a_base16 = [
                    a_base16[h] - row_base // fx.Int32(16) for h in range_constexpr(2)
                ]
                # The wait_barrier counts below assume 6 scale loads per K-pair.
                assert 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2) == 6

                def scale_prefetch(sc, k0):
                    if const_expr(k0 >= K_ITERS or k0 // 2 in sc):
                        return sc
                    words = {
                        w: (
                            a_sc.read(a_base16[int(w[1])], k0, sc_lds)
                            if w[0] == "a"
                            else b_sc.read(b_base16[int(w[1])], k0)
                        )
                        for w in ("a0", "a1", "b0", "b1")
                    }
                    return {**sc, k0 // 2: words}

                def scale_read(sc, k, which):
                    return sc[k // 2][which]

                c00_frag = [mfma.zero_value] * N_ACCUMS
                c01_frag = [mfma.zero_value] * N_ACCUMS
                c10_frag = [mfma.zero_value] * N_ACCUMS
                c11_frag = [mfma.zero_value] * N_ACCUMS

                b_g2s.load(b_cur0, B0_gl_offset)
                load_a(0, a_cur0, 0)
                b_g2s.load(b_cur1, B1_gl_offset)
                load_a(1, a_cur1, 0)
                a_sc.write_wide(sc_lds, sc_staged)
                lds_barrier()
                sc_pf = scale_prefetch({}, 0)
                # The wave_m == 1 group runs one barrier behind the other from here on.
                if wave_m == 1:
                    rocdl.s_barrier()
                wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
                b_g2s.load(b_next0, B0_gl_offset + B_K_STEP)
                load_a(0, a_next0, BLOCK_K)
                b_g2s.load(b_next1, B1_gl_offset + B_K_STEP)
                wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

                for k in range_constexpr(K_ITERS - 2):
                    if const_expr(k % 2 == 1):
                        sc_pf = scale_prefetch(sc_pf, k + 1)
                    sa0 = scale_read(sc_pf, k, "a0")
                    sb0 = scale_read(sc_pf, k, "b0")
                    b0_frag = b_s2r.load(b_cur0)
                    a0_frag = a_s2r.load(a_cur0)
                    load_a(1, a_next1, (k + 1) * BLOCK_K)
                    rocdl.s_barrier()

                    c00_frag = mfma.call(
                        a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2
                    )

                    sb1 = scale_read(sc_pf, k, "b1")
                    b1_frag = b_s2r.load(b_cur1)
                    b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
                    rocdl.s_barrier()

                    c01_frag = mfma.call(
                        a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2
                    )

                    sa1 = scale_read(sc_pf, k, "a1")
                    a1_frag = a_s2r.load(a_cur1)
                    load_a(0, a_cur0, (k + 2) * BLOCK_K)
                    # This barrier of the lagging group pairs with the other group's
                    # end-of-step wait, after which that group reads step k + 1's
                    # b_cur0 / a_cur0, loaded in step k - 1: drain everything but this
                    # step's three loads and step k - 1's b_cur1.
                    wait_barrier(2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

                    c10_frag = mfma.call(
                        a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2
                    )

                    b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
                    wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                    c11_frag = mfma.call(
                        a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2
                    )

                    a_cur0, a_next0 = a_next0, a_cur0
                    a_cur1, a_next1 = a_next1, a_cur1
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1

                # Step K_ITERS - 2: no loads left to overlap but the last A1 tile.
                k = K_ITERS - 2
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                # The lagging group loaded step K_ITERS - 1's tiles in step K_ITERS - 3:
                # drain them here, the barrier the other group's next reads pair with.
                wait_barrier(0)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                load_a(1, a_next1, (K_ITERS - 1) * BLOCK_K)
                rocdl.s_barrier()
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)
                b0_frag = b_s2r.load(b_next0)
                rocdl.s_barrier()
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

                # Step K_ITERS - 1
                k = K_ITERS - 1
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                a0_frag = a_s2r.load(a_cur0)
                wait_barrier(0)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c10_frag = mfma.call(
                    a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2, set_prio=False
                )
                c11_frag = mfma.call(
                    a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2, set_prio=False
                )
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                # Even the two groups' barrier counts again before the epilogue's barriers.
                if wave_m == 0:
                    rocdl.s_barrier()

                epilogue_swiglu(
                    (c00_frag, c01_frag),
                    (c10_frag, c11_frag),
                    row_base,
                    n_valid,
                    block_n,
                    mfma.idx,
                )

                return row_base, n_valid

            def act_mul(g, u):
                """MegaMoEV2 SwiGLU, as SiluQuantEpilogue._act_mul / _silu compute it."""

                def silu(x):
                    emu = (x * fx.Float32(-1.4426950408889634)).exp2()
                    return x * (fx.Float32(1.0) / (fx.Float32(1.0) + emu))

                if const_expr(swiglu_limit <= 0):
                    return silu(g) * u
                limit = fx.Float32(swiglu_limit)
                return silu(-(-g).maximumf(-limit)) * fx.clampf(u, -limit, limit)

            def epilogue_swiglu(half0, half1, row_base, n_valid, block_n, mfma_idx):
                """SwiGLU in registers, then per 64-row quarter (one wave_m group's rows) as f32
                through LDS [64][128]; every lane reads 8 contiguous columns of a row, the 1x32
                |max| is 4 lanes (2 xor shuffles per 8 values, as the Stage1 vec8 store) and the
                8 fp8 bytes and the group's e8m0 go straight to global."""
                tid, lane_id, wave_id, wave_m, wave_n = lane_ids()
                stage_lds = lds_addr(g1_lds.E_stage)
                f_p = lds_typed_ptr(stage_lds, T.f32)
                out_rsrc = create_buffer_resource_from_addr(
                    fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(g1_out).ir_value())),
                    num_records_bytes=fx.Int64(g1_tokens) * fx.Int64(INTER),
                )
                osc_rsrc = create_buffer_resource_from_addr(
                    fx.Int64(
                        rocdl.readfirstlane(T.i64, fx.Int64(g1_out_scale).ir_value())
                    ),
                    num_records_bytes=g1_scale_bytes,
                )
                l16 = lane_id % fx.Int32(16)
                c64 = fx.Int32(64)
                # Read side: thread t takes row t // 16 (+32 per pass) and columns (t % 16) * 8.
                rd_row = tid // fx.Int32(16)
                rd_col = (tid % fx.Int32(16)) * fx.Int32(8)
                col_s = block_n * fx.Int32(4) + (tid % fx.Int32(16)) // fx.Int32(4)
                writer = (tid % fx.Int32(4)) == fx.Int32(0)
                d3 = col_s >> fx.Int32(3)
                d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
                d5 = col_s & fx.Int32(3)
                for h_m in range_constexpr(2):
                    frags = half1 if h_m else half0
                    ys = []
                    for ti in range_constexpr(N_TILES_A):
                        for i in range_constexpr(4):
                            rq = (
                                ti * 16 + (lane_id // 16) * 4 + i
                            )  # row within the wave_m quarter
                            for h_n in range_constexpr(2):
                                g = Vec(frags[h_n][mfma_idx(ti, 0)])[i]
                                u = Vec(frags[h_n][mfma_idx(ti, 1)])[i]
                                ys.append(
                                    (
                                        rq * 128 + h_n * 64 + wave_n * 16 + l16,
                                        act_mul(g, u),
                                    )
                                )
                    for q in range_constexpr(2):
                        lds_barrier()
                        if wave_m == fx.Int32(q):
                            for idx, y in ys:
                                f_p[idx] = y
                        lds_barrier()
                        for ps in range_constexpr(2):
                            row_q = rd_row + fx.Int32(ps * 32)
                            v = Vec(
                                lds_vec_load(
                                    stage_lds,
                                    (row_q * 128 + rd_col) * 4,
                                    Vec.make_type(8, fx.Float32),
                                    fx.Float32,
                                    align=16,
                                )
                            )
                            vals = [fx.Float32(v[k]) for k in range_constexpr(8)]
                            m = vals[0].maximumf(fx.Float32(0.0) - vals[0])
                            for k in range_constexpr(1, 8):
                                m = m.maximumf(
                                    vals[k].maximumf(fx.Float32(0.0) - vals[k])
                                )
                            for off in (1, 2):
                                m = m.maximumf(m.shuffle_xor(fx.Int32(off), c64))
                            max_rounded = (
                                m.bitcast(fx.Int32) + fx.Int32(0x400000)
                            ) & fx.Int32(0xFF800000)
                            e = (max_rounded >> fx.Int32(23)) - fx.Int32(8)
                            e8m0 = (e > fx.Int32(0)).select(e, fx.Int32(0))
                            qs = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(
                                fx.Float32
                            )
                            words = []
                            for w in range_constexpr(2):
                                sc = [vals[w * 4 + k] * qs for k in range_constexpr(4)]
                                pk = rocdl.cvt_pk_fp8_f32(
                                    T.i32, sc[0], sc[1], fx.Int32(0), 0
                                )
                                pk = rocdl.cvt_pk_fp8_f32(T.i32, sc[2], sc[3], pk, 1)
                                words.append(fx.Int32(pk))
                            row_t = fx.Int32(h_m * 128 + q * 64) + row_q
                            row_g = row_base + row_t
                            ok = row_t < n_valid
                            off = ok.select(
                                row_g * fx.Int32(INTER)
                                + block_n * fx.Int32(128)
                                + rd_col,
                                g1_tokens * fx.Int32(INTER),
                            )
                            buffer_store(
                                Vec.from_elements(words, fx.Int32),
                                out_rsrc,
                                off,
                                offset_is_bytes=True,
                                cache_modifier=G1_OUT_CACHE,
                            )
                            d0 = row_g >> fx.Int32(5)
                            d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
                            d2 = row_g & fx.Int32(15)
                            b_off = (
                                d0 * fx.Int32(SC_COLS32)
                                + d3 * fx.Int32(256)
                                + d5 * fx.Int32(64)
                                + d2 * fx.Int32(4)
                                + d4 * fx.Int32(2)
                                + d1
                            )
                            b_off = (ok & writer).select(b_off, g1_scale_bytes)
                            buffer_store(
                                e8m0.to(fx.Int8),
                                osc_rsrc,
                                b_off,
                                offset_is_bytes=True,
                                cache_modifier=G1_OUT_CACHE,
                            )

            return process_tile

        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        def _g2_build(
            A,
            B_T,
            A_scale,
            B_scale,
            c_rows,
            stids,
            sweights,
            trb,
            p2p,
            send_base,
            win,
            cdone,
            plan,
        ):
            """The 8-wave GEMM2 item (256 sorted rows x 256 output columns) on the GEMM1 output:
            fp8 1x32 rows into the send buffer ``S`` at their run position (the plan's ``D``), or
            straight into the origin rank's ``comb_inp`` past the reservation."""
            import mori.cco.device.flydsl as cco

            A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(2**31 - 1, 1)))
            K, N, experts, topk, npes, max_tok = (
                inter_dim,
                model_dim,
                fz_epr,
                fuse_topk,
                fuse_npes,
                fuse_mtpr,
            )
            SBM, rank, layout, chunk_tiles = sort_block_m, fz_rank, G2_LAY, G2_CHUNK
            c_m = (c_rows + fx.Int32(255)) // fx.Int32(256) * fx.Int32(256)
            BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
            K_ITERS = K // BLOCK_K
            N_TILES_A = BLOCK_M // 64
            N_TILES_B = BLOCK_N // 128
            N_ACCUMS = N_TILES_A * N_TILES_B
            LDS_BLOCK_M = BLOCK_M // 2
            LDS_BLOCK_N = BLOCK_N // 2
            N_LDS_STEPS_A = LDS_BLOCK_M // 64
            B_ROW_BYTES = BLOCK_K // 2
            N_LDS_STEPS_B = LDS_BLOCK_N * B_ROW_BYTES // (8 * 1024)
            N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
            A_GRP_ROWS = N_TILES_A * 16
            B_GRP_ROWS = N_TILES_B * 16
            B_K_STEP = 1024
            send_nbytes = layout.n_rows * layout.row_bytes
            row_bytes = layout.row_bytes
            log2_max_tok = max_tok.bit_length() - 1
            comb_bytes = max_tok * topk * row_bytes
            F8_IR_t = fx.Float8E4M3FN.ir_type
            n_blocks = fx.Int32(N // BLOCK_N)

            def lds_addr(arr):
                return fx.Int32(fx.ptrtoint(arr.ptr))

            def lane_ids():
                # Recomputed per work item and per epilogue (see opaque_i32).
                tid = opaque_i32(fx.thread_idx.x)
                wave_id = tid // fx.Int32(64)
                return (
                    tid,
                    tid % fx.Int32(64),
                    wave_id,
                    wave_id // fx.Int32(4),
                    wave_id % fx.Int32(4),
                )

            def process_tile(block_m, block_n, tab_lds):
                _, lane_id, wave_id, wave_m, wave_n = lane_ids()
                a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
                a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
                b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
                b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
                row_base, expert, n_valid = tile_find(
                    tab_lds, block_m, lane_id, SBM=SBM, experts=experts
                )
                b_row_block = expert * n_blocks + block_n

                A0_gl_offset = row_base * K
                A1_gl_offset = (row_base + LDS_BLOCK_M) * K
                B0_gl_offset = (b_row_block * BLOCK_N) * (K // 2)
                B1_gl_offset = (b_row_block * BLOCK_N + LDS_BLOCK_N) * (K // 2)
                a_div = fx.logical_divide(
                    make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
                )
                b_div = fx.logical_divide(
                    make_fp8_buffer_tensor(B_T, F8_IR_t), fx.make_layout(1, 1)
                )
                gl_off_a = compute_global_swizzle(
                    lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
                )
                # a16w4 layout: wave w copies 16-row group w's contiguous 1 KB of a K step.
                gl_off_b = [wave_id * (16 * (K // 2)) + lane_id * 16]

                mfma = MxMfma(N_TILES_A, N_TILES_B)
                a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
                b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
                a_s2r = S2RLoader(wave_m, N_TILES_A)
                a_s2r.lane_id = lane_id
                b_s2r = S2RF4G(wave_n, N_TILES_B, lane_id)
                a_sc = ScalePreshuffledS2R(A_scale, c_m, K, N_TILES_A, lane_id)
                b_sc = ScalePreshuffledS2R(B_scale, experts * N, K, N_TILES_B, lane_id)
                # 16-row tile index of each wave's first row, per LDS half.
                a_base16 = [
                    (row_base + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
                    for h in range_constexpr(2)
                ]
                b_base16 = [
                    (b_row_block * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS)
                    // 16
                    for h in range_constexpr(2)
                ]
                # The wait_barrier counts below assume 6 scale loads per K-pair.
                assert 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2) == 6

                def scale_prefetch(sc, k0):
                    if const_expr(k0 >= K_ITERS or k0 // 2 in sc):
                        return sc
                    words = {
                        w: (a_sc if w[0] == "a" else b_sc).read(
                            (a_base16 if w[0] == "a" else b_base16)[int(w[1])], k0
                        )
                        for w in ("a0", "a1", "b0", "b1")
                    }
                    return {**sc, k0 // 2: words}

                def scale_read(sc, k, which):
                    return sc[k // 2][which]

                c00_frag = [mfma.zero_value] * N_ACCUMS
                c01_frag = [mfma.zero_value] * N_ACCUMS
                c10_frag = [mfma.zero_value] * N_ACCUMS
                c11_frag = [mfma.zero_value] * N_ACCUMS

                sc_pf = scale_prefetch({}, 0)
                b_g2s.load(b_cur0, B0_gl_offset)
                a_g2s.load(a_cur0, A0_gl_offset)
                b_g2s.load(b_cur1, B1_gl_offset)
                a_g2s.load(a_cur1, A1_gl_offset)
                # The wave_m == 1 group runs one barrier behind the other from here on.
                if wave_m == 1:
                    rocdl.s_barrier()
                wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
                b_g2s.load(b_next0, B0_gl_offset + B_K_STEP)
                a_g2s.load(a_next0, A0_gl_offset + BLOCK_K)
                b_g2s.load(b_next1, B1_gl_offset + B_K_STEP)
                wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

                for k in range_constexpr(K_ITERS - 2):
                    if const_expr(k % 2 == 1):
                        sc_pf = scale_prefetch(sc_pf, k + 1)
                    sa0 = scale_read(sc_pf, k, "a0")
                    sb0 = scale_read(sc_pf, k, "b0")
                    b0_frag = b_s2r.load(b_cur0)
                    a0_frag = a_s2r.load(a_cur0)
                    a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                    rocdl.s_barrier()

                    c00_frag = mfma.call(
                        a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2
                    )

                    sb1 = scale_read(sc_pf, k, "b1")
                    b1_frag = b_s2r.load(b_cur1)
                    b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
                    rocdl.s_barrier()

                    c01_frag = mfma.call(
                        a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2
                    )

                    sa1 = scale_read(sc_pf, k, "a1")
                    a1_frag = a_s2r.load(a_cur1)
                    a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                    # This barrier of the lagging group pairs with the other group's
                    # end-of-step wait, after which that group reads step k + 1's
                    # b_cur0 / a_cur0, loaded in step k - 1: drain everything but this
                    # step's three loads and step k - 1's b_cur1.
                    wait_barrier(2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

                    c10_frag = mfma.call(
                        a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2
                    )

                    b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
                    wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                    c11_frag = mfma.call(
                        a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2
                    )

                    a_cur0, a_next0 = a_next0, a_cur0
                    a_cur1, a_next1 = a_next1, a_cur1
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1

                # Step K_ITERS - 2: no loads left to overlap but the last A1 tile.
                k = K_ITERS - 2
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                # The lagging group loaded step K_ITERS - 1's tiles in step K_ITERS - 3:
                # drain them here, the barrier the other group's next reads pair with.
                wait_barrier(0)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
                rocdl.s_barrier()
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)
                b0_frag = b_s2r.load(b_next0)
                rocdl.s_barrier()
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

                # Step K_ITERS - 1
                k = K_ITERS - 1
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                a0_frag = a_s2r.load(a_cur0)
                wait_barrier(0)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c10_frag = mfma.call(
                    a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2, set_prio=False
                )
                c11_frag = mfma.call(
                    a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2, set_prio=False
                )
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                # Even the two groups' barrier counts again before the epilogue's barriers.
                if wave_m == 0:
                    rocdl.s_barrier()

                epilogue(
                    (c00_frag, c01_frag),
                    (c10_frag, c11_frag),
                    row_base,
                    n_valid,
                    block_n,
                    mfma.idx,
                    expert,
                )

            def epilogue(half0, half1, row_base, n_valid, block_n, mfma_idx, expert):
                tid, lane_id, wave_id, wave_m, wave_n = lane_ids()
                orow_lds, src_lds, wts_lds = (
                    lds_addr(lds.E_orow),
                    lds_addr(lds.E_src),
                    lds_addr(lds.E_wts),
                )
                stage_lds, scale_lds = lds_addr(lds.E_stage), lds_addr(lds.E_scale)

                # Row descriptors of the tile: out_row, srcmap word, weight.
                row = tid % fx.Int32(256)
                in_tile = (row < n_valid) & (tid < fx.Int32(256))
                grow = row_base + in_tile.select(row, fx.Int32(0))
                sm_idx = global_typed_ptr(trb, T.i32)[
                    grow // fx.Int32(SBM)
                ] + grow % fx.Int32(SBM)
                packed = fx.Int32(global_typed_ptr(stids, T.i32)[sm_idx])
                # The plan, inline: an expert's rows are grouped by origin rank, so the row's index
                # in its origin's run is the sorted row plus a per-(expert, origin) offset.
                xv, xpe, xlid, xk = s2_decode(
                    packed,
                    log2_max_tok=log2_max_tok,
                    recv_cap=npes * max_tok,
                    topk=topk,
                    npes=npes,
                )
                xv = xv & in_tile
                xpe = xv.select(xpe, fx.Int32(0))
                place = grow + fx.Int32(
                    global_typed_ptr(plan, T.i32)[
                        fx.Int32(2 * G2_P) + expert * fx.Int32(npes) + xpe
                    ]
                )
                fits = place < fx.Int32(layout.cap)
                orow = xv.select(
                    fits.select(
                        xpe * fx.Int32(layout.cap) + place, fx.Int32(OUT_ROW_P2P)
                    ),
                    fx.Int32(-1),
                )
                # The origin rank's inv entry: my landing row there (on this rank, the S row itself).
                # Retired before publish() counts the item, hence before the chunk's put and flag.
                if block_n == fx.Int32(0):
                    wv = cco.Window(win)
                    inv_base = fx.Int64(wv.lsa_ptr(0, layout.off_inv))
                    for pp in range_constexpr(1, npes):
                        inv_base = (xpe == fx.Int32(pp)).select(
                            fx.Int64(wv.lsa_ptr(pp, layout.off_inv)), inv_base
                        )
                    if xv & fits:
                        comm_ops.store_i32_global_system_monotonic(
                            inv_base
                            + fx.Int64(xlid * fx.Int32(topk) + xk) * fx.Int64(4),
                            (xpe == fx.Int32(rank)).select(
                                orow,
                                fx.Int32(layout.n_rows + rank * layout.cap) + place,
                            ),
                        )
                weight = fx.Float32(global_typed_ptr(sweights, T.f32)[sm_idx])
                if tid < fx.Int32(256):
                    lds_typed_ptr(orow_lds, T.i32)[row] = orow
                    lds_typed_ptr(src_lds, T.i32)[row] = packed
                    lds_typed_ptr(wts_lds, T.f32)[row] = weight
                lds_barrier()

                send = create_buffer_resource_from_addr(
                    fx.Int64(
                        rocdl.readfirstlane(T.i64, fx.Int64(send_base).ir_value())
                    ),
                    num_records_bytes=send_nbytes,
                )
                orow_p = lds_typed_ptr(orow_lds, T.i32)
                src_p = lds_typed_ptr(src_lds, T.i32)
                half = lane_id // fx.Int32(32)
                col = (lane_id % fx.Int32(32)) * fx.Int32(8)
                row_lead = (lane_id % fx.Int32(32)) == fx.Int32(0)
                for h_m in range_constexpr(2):
                    if const_expr(h_m > 0):
                        lds_barrier()
                    frags = half1 if h_m else half0
                    g2_epi_quant_half(
                        frags[0],
                        frags[1],
                        h_m=h_m,
                        wave_m=wave_m,
                        wave_n=wave_n,
                        lane_id=lane_id,
                        mfma_idx=mfma_idx,
                        n_tiles_a=N_TILES_A,
                        wts_lds=wts_lds,
                        stage_lds=stage_lds,
                        scale_lds=scale_lds,
                    )
                    lds_barrier()
                    # Each wave copies 16 rows of the half, two per store (lanes 0-31 / 32-63).
                    for it in range_constexpr(8):
                        r = wave_id * fx.Int32(16) + fx.Int32(2 * it) + half
                        o = fx.Int32(orow_p[fx.Int32(h_m * 128) + r])
                        payload = Vec(
                            lds_vec_load(
                                stage_lds,
                                r * 256 + col,
                                Vec.make_type(2, fx.Int32),
                                fx.Int32,
                                align=8,
                            )
                        )
                        scales = Vec(
                            lds_vec_load(
                                scale_lds,
                                r * 8,
                                Vec.make_type(2, fx.Int32),
                                fx.Int32,
                                align=8,
                            )
                        )
                        base = o * fx.Int32(row_bytes)
                        ok = o >= fx.Int32(0)
                        p_off = ok.select(
                            base + block_n * fx.Int32(256) + col, fx.Int32(send_nbytes)
                        )
                        buffer_store(
                            payload,
                            send,
                            p_off,
                            cache_modifier=G2_SEND_CM,
                            offset_is_bytes=True,
                        )
                        s_off = (ok & row_lead).select(
                            base + fx.Int32(N) + block_n * fx.Int32(8),
                            fx.Int32(send_nbytes),
                        )
                        buffer_store(
                            scales,
                            send,
                            s_off,
                            cache_modifier=G2_SEND_CM,
                            offset_is_bytes=True,
                        )
                        # Rows past the per-pair reservation go straight to the origin rank.
                        for h in range_constexpr(2):
                            rh = (
                                wave_id * fx.Int32(16)
                                + fx.Int32(2 * it + h)
                                + fx.Int32(h_m * 128)
                            )
                            if fx.Int32(
                                rocdl.readfirstlane(T.i32, orow_p[rh])
                            ) == fx.Int32(OUT_ROW_P2P):
                                direct_row(
                                    fx.Int32(rocdl.readfirstlane(T.i32, src_p[rh])),
                                    payload,
                                    scales,
                                    half == fx.Int32(h),
                                    row_lead,
                                    col,
                                    block_n,
                                )
                publish(row_base, n_valid)

            def direct_row(p, payload, scales, mine, row_lead, col, block_n):
                """Store one row's 256 columns into ``comb_inp[slot]`` of its origin rank, as
                the default Stage2 does, and point that rank's ``inv[slot]`` at it."""
                t = p & fx.Int32(0x00FFFFFF)
                s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
                dest_pe = t >> fx.Int32(log2_max_tok)
                slot = (t & fx.Int32(max_tok - 1)) * fx.Int32(topk) + s
                peer = fx.Int64(
                    rocdl.readfirstlane(
                        T.i64, global_typed_ptr(p2p, T.i64, align=8)[dest_pe]
                    )
                )
                w = cco.Window(win)
                # Like the comm waves: not before the origin rank has read the last call's
                # comb_inp and inv (its combine publishes done[] to this rank).
                if dest_pe != fx.Int32(rank):
                    rsrc_counter = create_buffer_resource_from_addr(
                        fx.Int64(w.lsa_ptr(rank, layout.off_counter))
                    )
                    phase = fx.Int32(
                        buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                    ) + fx.Int32(1)
                    sdma_wait_consumed(w, dest_pe, phase, layout=layout, rank=rank)
                # Written through like S: retired stores are in the peer's memory before the
                # item publishes, with no system-scope release.
                buf = create_buffer_resource_from_addr(
                    peer, num_records_bytes=comb_bytes
                )
                base = slot * fx.Int32(row_bytes)
                p_off = mine.select(
                    base + block_n * fx.Int32(256) + col, fx.Int32(comb_bytes)
                )
                buffer_store(
                    payload, buf, p_off, cache_modifier=G2_SEND_CM, offset_is_bytes=True
                )
                s_off = (mine & row_lead).select(
                    base + fx.Int32(N) + block_n * fx.Int32(8), fx.Int32(comb_bytes)
                )
                buffer_store(
                    scales, buf, s_off, cache_modifier=G2_SEND_CM, offset_is_bytes=True
                )
                if (block_n == fx.Int32(0)) & mine & row_lead:
                    comm_ops.store_i32_global_system_monotonic(
                        fx.Int64(w.lsa_ptr(dest_pe, layout.off_inv))
                        + fx.Int64(slot) * fx.Int64(4),
                        fx.Int32(INV_P2P),
                    )

            def publish(row_base, n_valid):
                """Every wave's rows have retired (written through); count each sort tile
                of the item once toward its chunk."""
                rocdl.s_waitcnt(vmcnt=0)
                fx.barrier()
                if fx.thread_idx.x == fx.Int32(0):
                    st0 = row_base // fx.Int32(SBM)
                    n_st = n_valid // fx.Int32(SBM)
                    for i in range_constexpr(256 // SBM):
                        if fx.Int32(i) < n_st:
                            comm_ops.atomic_add_agent(
                                fx.Int64(cdone)
                                + fx.Int64((st0 + fx.Int32(i)) // fx.Int32(chunk_tiles))
                                * fx.Int64(CDONE_STRIDE * 4),
                                fx.Int32(1),
                            )

            return process_tile

        def _g2_plan_tables(g2_plan, g2_flag, g2_epoch):
            """SEG[e][s] (first sorted row of origin s in expert e), CNT[e][s], and
            D[e][s] = (rows of origin s in experts before e) - SEG[e][s]; then the flag.
            """
            if tid == fx.Int32(0):
                comm_ops.fence_system_acquire()
            fx.barrier()
            cm = ptr_buf_tensor(_disp_ptr(DispatchSlot.COUNT_MATRIX), fx.Int32)
            te = ptr_buf_tensor(_disp_ptr(DispatchSlot.EXPERT_TILE_END), fx.Int32)
            pl = ptr_buf_tensor(g2_plan, fx.Int32)
            if tid < fx.Int32(G2_P):
                e = tid // fx.Int32(fuse_npes)
                src = tid % fx.Int32(fuse_npes)
                cnt = cm[src * fx.Int32(fz_epr) + e]
                prev_end = te[(e > fx.Int32(0)).select(e - fx.Int32(1), fx.Int32(0))]
                seg = (e > fx.Int32(0)).select(prev_end, fx.Int32(0)) * fx.Int32(
                    sort_block_m
                )
                for s2 in range_constexpr(fuse_npes):
                    seg = seg + (fx.Int32(s2) < src).select(
                        cm[fx.Int32(s2 * fz_epr) + e], fx.Int32(0)
                    )
                cp = fx.Int32(0)
                for e2 in range(fx.Int32(0), e, fx.Int32(1)):
                    cp = cp + cm[src * fx.Int32(fz_epr) + e2]
                pl[tid] = seg
                pl[fx.Int32(G2_P) + tid] = cnt
                pl[fx.Int32(2 * G2_P) + tid] = cp - seg
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                comm_ops.store_i32_system(g2_flag, fx.Int32(0), g2_epoch)

        def _g2_comm_role(g2_win, g2_dev_comm, g2_cdone, g2_plan, g2_num_valid):
            """The comm role, one wave per peer, with each chunk's run bounds from SEG / CNT
            (the inv entries were written by the GEMM2 epilogues)."""
            import mori.cco.device.flydsl as cco

            lay = G2_LAY
            lane = tid % fx.Int32(64)
            peer = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // fx.Int32(64)))
            is_lane0 = lane == fx.Int32(0)
            queue = fx.Int32(0)
            pl = ptr_buf_tensor(g2_plan, fx.Int32)
            n_blocks = fx.Int32(G2_NB2)

            def run_end(t):
                """Rows of my run for ``peer`` in the sort tiles before ``t``, capped (wave-wide)."""
                g0 = t * fx.Int32(sort_block_m)
                acc = fx.Int32(0)
                for ec in range_constexpr((fz_epr + 63) // 64):
                    e = fx.Int32(ec * 64) + lane
                    ok = e < fx.Int32(fz_epr)
                    idx = ok.select(e, fx.Int32(0)) * fx.Int32(fuse_npes) + peer
                    v = g0 - pl[idx]
                    c = pl[fx.Int32(G2_P) + idx]
                    v = (v < fx.Int32(0)).select(fx.Int32(0), v)
                    v = (v > c).select(c, v)
                    acc = acc + ok.select(v, fx.Int32(0))
                tot = fx.Int32(
                    fx.rocdl.readlane(T.i32, _wave_inclusive_scan_i32(acc, lane), 63)
                )
                return (tot < fx.Int32(lay.cap)).select(tot, fx.Int32(lay.cap))

            if (peer < fx.Int32(fuse_npes)) & (peer != fx.Int32(fz_rank)):
                xw = cco.Window(g2_win)
                sdma = cco.DevComm(g2_dev_comm).sdma()
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(xw.lsa_ptr(fz_rank, lay.off_counter))
                )
                phase = fx.Int32(
                    buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                ) + fx.Int32(1)
                if is_lane0:
                    sdma_wait_consumed(xw, peer, phase, layout=lay, rank=fz_rank)
                total_tiles = (g2_num_valid + fx.Int32(sort_block_m - 1)) // fx.Int32(
                    sort_block_m
                )
                n_chunks = (total_tiles + fx.Int32(G2_CHUNK - 1)) // fx.Int32(G2_CHUNK)
                last = n_chunks - fx.Int32(1)
                for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                    ci = fx.Int32(c)
                    t0 = ci * fx.Int32(G2_CHUNK)
                    t1 = t0 + fx.Int32(G2_CHUNK)
                    t1 = (t1 < total_tiles).select(t1, total_tiles)
                    if is_lane0:
                        comm_ops.spin_until_ge_i32_agent(
                            g2_cdone + fx.Int64(ci) * fx.Int64(CDONE_STRIDE * 4),
                            (t1 - t0) * n_blocks,
                        )
                    comm_ops.fence_agent_acquire()
                    lo = run_end(t0)
                    hi = run_end(t1)
                    if (hi > lo) & is_lane0:
                        if ci == last:
                            sdma_put_run(
                                sdma,
                                g2_win,
                                peer,
                                queue,
                                layout=lay,
                                rank=fz_rank,
                                lo=lo,
                                hi=hi,
                                cco=cco,
                                remote_signal=True,
                            )
                        else:
                            sdma_put_run(
                                sdma,
                                g2_win,
                                peer,
                                queue,
                                layout=lay,
                                rank=fz_rank,
                                lo=lo,
                                hi=hi,
                                cco=cco,
                            )
                t_last = last * fx.Int32(G2_CHUNK)
                t_last = (t_last > fx.Int32(0)).select(t_last, fx.Int32(0))
                lo_l = run_end(t_last)
                hi_l = run_end(total_tiles)
                if is_lane0:
                    if (n_chunks <= fx.Int32(0)) | (hi_l <= lo_l):
                        sdma_signal_peer(
                            sdma, g2_win, peer, queue, layout=lay, rank=fz_rank, cco=cco
                        )
                    publish_flag(xw, peer, phase, layout=lay, rank=fz_rank)

        def _disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_work_tail = _disp_ptr(DispatchSlot.WORK_TAIL)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(
            DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION
        )
        a_payload_chunks_per_destination = _disp_ptr(
            DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION
        )
        a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
        p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_payload_ready_rows = _disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(
                    a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1)
                )
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid_x)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        owner = ticket == fx.Int32(0)
        producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)

        if owner:
            next_parity_lane = fx.Int32(0)
            launch_epoch_lane = fx.Int32(0)
            if tid == fx.Int32(0):
                old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                next_parity_lane = old_parity ^ fx.Int32(1)
                previous_expected = _buffer_load(
                    expected_rsrc, next_parity_lane, fx.Int32
                )
                next_expected = previous_expected + fx.Int32(fz_npes)
                _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
                launch_epoch_lane = (next_expected // fx.Int32(fz_npes)) * fx.Int32(
                    2
                ) - next_parity_lane
            next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
            launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
            if tid == fx.Int32(0):
                comm_ops.store_i32_system(
                    a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m)
                )
                comm_ops.fence_system_release()
            fx.barrier()
            if tid < fx.Int32(fz_npes):
                peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                comm_ops.fence_system_release()
                launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
                remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
                comm_ops.store_i32_system(
                    remote_launch_ready, fx.Int32(fz_rank), launch_epoch
                )
                # Equality, not >=, so the 32-bit epoch may wrap: no peer can be a launch of this slot
                # ahead, since finishing this one needs this rank's dispatch.
                comm_ops.wait_i32_until_equals(
                    a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch
                )
                comm_ops.fence_system_acquire()
            if tid == fx.Int32(0):
                work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                for shard in range_constexpr(8):
                    _buffer_store(
                        work_head_rsrc, fx.Int32(shard * 16), fx.Int32(0), fx.Int32
                    )
                _buffer_store(
                    _make_buffer_from_addr(a_work_tail, fx.Int32),
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32,
                )
                _buffer_store(
                    _make_buffer_from_addr(a_group_done, fx.Int32),
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32,
                )
            # Rank-local G1 completion entries and G2 claim heads restart every launch; they are
            # cleared before the gate so no CTA of this launch can observe stale values.
            ctr_rsrc = _make_buffer_from_addr(addr_mtile_ctr, fx.Int32)
            if tid == fx.Int32(0):
                for shard in range_constexpr(G2_SHARDS):
                    _buffer_store(
                        ctr_rsrc,
                        fx.Int32(G2_HEAD_BYTE_OFF // 4 + shard * 16),
                        fx.Int32(0),
                        fx.Int32,
                    )
            for _ci in range_constexpr(
                (MAX_SORT_TILES + TOTAL_THREADS - 1) // TOTAL_THREADS
            ):
                clr = tid + fx.Int32(_ci * TOTAL_THREADS)
                if clr < fx.Int32(MAX_SORT_TILES):
                    _buffer_store(ctr_rsrc, clr, fx.Int32(0), fx.Int32)
            # The SDMA chunk counters.
            g2_cd_rsrc = _make_buffer_from_addr(
                _buffer_load(
                    _make_buffer_from_addr(ptr_tab, fx.Int64), fx.Int32(3), fx.Int64
                ),
                fx.Int32,
            )
            for _ci in range_constexpr((G2_NCH + TOTAL_THREADS - 1) // TOTAL_THREADS):
                clr = tid + fx.Int32(_ci * TOTAL_THREADS)
                if clr < fx.Int32(G2_NCH):
                    _buffer_store(
                        g2_cd_rsrc, clr * fx.Int32(CDONE_STRIDE), fx.Int32(0), fx.Int32
                    )
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            comm_ops.fence_system_release()
            fx.barrier()
            if tid == fx.Int32(0):
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                comm_ops.fence_agent_release()
                _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                comm_ops.fence_agent_release()
                comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.barrier()
        else:
            if tid == fx.Int32(0):
                comm_ops.wait_i32_until_equals(gate_addr, gate_epoch)
                comm_ops.fence_agent_acquire()
            fx.barrier()

        payload_parity = _buffer_load(
            parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE
        )
        payload_expected = _buffer_load(
            expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE
        )

        if owner:
            emit_dispatch_plan(
                num_waves=NUM_WAVES,
                fz_npes=fz_npes,
                fz_epr=fz_epr,
                fz_mtpr=fz_mtpr,
                fz_rank=fz_rank,
                fz_tile_m=fz_tile_m,
                fz_total_experts=fz_total_experts,
                addr_disp=addr_disp,
                parity=payload_parity,
                expected=payload_expected,
                dispatch_blocks=dispatch_blocks,
                payload_chunk_rows=payload_chunk_rows,
            )

        if producer:
            emit_dispatch_group(
                num_waves=NUM_WAVES,
                fz_k=fz_k,
                fz_total_experts=fz_total_experts,
                addr_disp=addr_disp,
                i32_cur_tok=i32_cur_tok,
                addr_in_idx=addr_in_idx,
                dispatch_blocks=dispatch_blocks,
                producer_slot=producer_slot,
                parity=payload_parity,
                expected=payload_expected,
            )
            producers_per_destination = fx.Int32(dispatch_blocks // fz_npes)
            chunks_per_destination = fx.Int32(
                (fz_mtpr + payload_chunk_rows - 1) // payload_chunk_rows
            )
            payload_active = fx.Int32(0) == fx.Int32(0)
            if const_expr(dispatch_blocks > 32):
                producer_destination = producer_slot % fx.Int32(fz_npes)
                producer_round = producer_slot // fx.Int32(fz_npes)
                producers_per_destination = _buffer_load(
                    _make_buffer_from_addr(a_payload_blocks_per_destination, fx.Int32),
                    producer_destination,
                    fx.Int32,
                )
                chunks_per_destination = _buffer_load(
                    _make_buffer_from_addr(a_payload_chunks_per_destination, fx.Int32),
                    producer_destination,
                    fx.Int32,
                )
                payload_active = producer_round < producers_per_destination
            if payload_active:
                emit_dispatch_payload(
                    num_waves=NUM_WAVES,
                    fz_epr=fz_epr,
                    fz_k=fz_k,
                    fz_mtpr=fz_mtpr,
                    fz_rank=fz_rank,
                    fz_total_experts=fz_total_experts,
                    fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_safe_end_i32=fz_safe_end_i32,
                    fz_scale_n_i32=fz_scale_n_i32,
                    fz_enable_scales=fz_enable_scales,
                    addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok,
                    addr_in_wts=addr_in_wts,
                    addr_in_sc=addr_in_sc,
                    dispatch_blocks=dispatch_blocks,
                    producer_slot=producer_slot,
                    parity=payload_parity,
                    expected=payload_expected,
                    producers_per_destination=producers_per_destination,
                    payload_chunk_rows=payload_chunk_rows,
                    chunks_per_destination=chunks_per_destination,
                )
        addr_tile_ready = _disp_ptr(DispatchSlot.TILE_READY)
        addr_tile_expected = _disp_ptr(DispatchSlot.TILE_EXPECTED)

        if tid == fx.Int32(0):
            local_plan_ready = _buffer_load(
                disp_rsrc, fx.Int32(int(DispatchSlot.PLAN_READY)), fx.Int64
            )
            ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.wait_i32_until_equals(
                local_plan_ready + fx.Int64(ready_index) * fx.Int64(4), payload_expected
            )
            comm_ops.fence_agent_acquire()
        fx.barrier()

        num_valid = _buffer_load(
            _make_buffer_from_addr(_disp_ptr(DispatchSlot.NUM_VALID), fx.Int32),
            fx.Int32(0),
            fx.Int32,
        )

        # Thread-0-private LDS word holding the last G1 stripe this CTA finished (-1 = none).
        # Its completion is published at the next claim so no atomic sits in the GEMM1 tail.
        pub_slot = lds_typed_ptr(
            fx.Int32(fx.ptrtoint(lds.A_scale.ptr)) + fx.Int32(CTRL_OFF), T.i32
        )
        # The launch epoch sits in a thread-0-private LDS word next to pub_slot, not in a register
        # that would stay live across the GEMM1 tiles.
        epoch_slot_ptr = lds_typed_ptr(
            fx.Int32(fx.ptrtoint(lds.A_scale.ptr)) + fx.Int32(CTRL_OFF + 8), T.i32
        )
        # Thread 0 broadcasts each claimed unit through an LDS word of its own (the tile buffers are
        # written by the first waves to start the tile).
        work_word = lds_typed_ptr(
            fx.Int32(fx.ptrtoint(lds.A_scale.ptr)) + fx.Int32(CTRL_OFF + 16), T.i32
        )
        if tid == fx.Int32(0):
            fx.ptr_store(fx.Int32(-1), pub_slot)
            fx.ptr_store(
                (payload_expected // fx.Int32(fz_npes)) * fx.Int32(2) - payload_parity,
                epoch_slot_ptr,
            )
        # Our buffers are native fields of SharedStorage in this mode (see there).
        g1_lds = lds
        g1_tile = _g1_build(
            x,
            w,
            scale_w,
            s2_aq,
            s2_ascale,
            scale_raw,
            s2_trb,
            tokens,
            g1_lds,
            addr_tile_ready,
            addr_tile_expected,
        )
        g1_lane = fx.thread_idx.x % fx.Int32(64)
        g1_tab = fx.Int32(fx.ptrtoint(g1_lds.E_tab.ptr))
        g1_vals, g1_ntiles = tile_table(
            _disp_ptr(DispatchSlot.EXPERT_TILE_END),
            g1_lane,
            SBM=sort_block_m,
            experts=fz_epr,
        )
        tile_table_to_lds(g1_vals, g1_tab, g1_lane)
        lds_barrier()
        g1_total = g1_ntiles * fx.Int32(G1_NB)
        # The last workgroup is the comm workgroup: no GEMM work.
        g1_active = ticket != fx.Int32(launch_grid_x - 1)
        while g1_active:
            # Publish the previous unit's sort tiles once its stores have drained (pub_slot
            # holds st0 * 4 + count, -1 for none), then claim the next unit.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.barrier()
            if tid == fx.Int32(0):
                g1_done = fx.Int32(fx.ptr_load(pub_slot, result_type=T.i32))
                if g1_done >= fx.Int32(0):
                    g1_st = g1_done >> fx.Int32(2)
                    comm_ops.atomic_add_system(
                        addr_mtile_ctr + fx.Int64(g1_st) * fx.Int64(4), fx.Int32(1)
                    )
                    if (g1_done & fx.Int32(3)) == fx.Int32(2):
                        comm_ops.atomic_add_system(
                            addr_mtile_ctr
                            + fx.Int64(g1_st + fx.Int32(1)) * fx.Int64(4),
                            fx.Int32(1),
                        )
                fx.ptr_store(
                    fx.Int32(comm_ops.atomic_add_agent(a_work_head, fx.Int32(1))),
                    work_word,
                )
            fx.barrier()
            g1_work = fx.Int32(
                fx.rocdl.readfirstlane(
                    T.i32, fx.Int32(fx.ptr_load(work_word, result_type=T.i32))
                )
            )
            g1_has = g1_work < g1_total
            if g1_has:
                g1_rb, g1_nv = g1_tile(
                    g1_work // fx.Int32(G1_NB), g1_work % fx.Int32(G1_NB), g1_tab
                )
                if tid == fx.Int32(0):
                    fx.ptr_store(
                        (g1_rb // fx.Int32(sort_block_m)) * fx.Int32(4)
                        + (g1_nv > fx.Int32(sort_block_m)).select(
                            fx.Int32(2), fx.Int32(1)
                        ),
                        pub_slot,
                    )
                fx.barrier()
            g1_active = g1_has
        g2_tab = _make_buffer_from_addr(ptr_tab, fx.Int64)
        g2_win = _wave_uniform_i64(_buffer_load(g2_tab, fx.Int32(0), fx.Int64))
        g2_dev_comm = _wave_uniform_i64(_buffer_load(g2_tab, fx.Int32(1), fx.Int64))
        g2_send = _wave_uniform_i64(_buffer_load(g2_tab, fx.Int32(2), fx.Int64))
        g2_cdone = _wave_uniform_i64(_buffer_load(g2_tab, fx.Int32(3), fx.Int64))
        g2_plan = _wave_uniform_i64(_buffer_load(g2_tab, fx.Int32(4), fx.Int64))
        g2_flag = g2_plan + fx.Int64(3 * G2_P * 4)
        if ticket == fx.Int32(launch_grid_x - 1):
            g2_epoch = fx.Int32(0)
            if tid == fx.Int32(0):
                g2_epoch = fx.Int32(fx.ptr_load(epoch_slot_ptr, result_type=T.i32))
            _g2_plan_tables(g2_plan, g2_flag, g2_epoch)
            _g2_comm_role(g2_win, g2_dev_comm, g2_cdone, g2_plan, num_valid)
        else:
            g2_tile = _g2_build(
                out,
                xw2,
                out_scale,
                xw2s,
                tokens,
                s2_stids,
                s2_sweights,
                s2_trb,
                s2_p2p_comb_inp,
                g2_send,
                g2_win,
                g2_cdone,
                g2_plan,
            )
            if tid == fx.Int32(0):
                comm_ops.wait_i32_until_equals(
                    g2_flag, fx.Int32(fx.ptr_load(epoch_slot_ptr, result_type=T.i32))
                )
                comm_ops.fence_agent_acquire()
            g2_head = addr_mtile_ctr + fx.Int64(G2_HEAD_BYTE_OFF)
            g2_total = g1_ntiles * fx.Int32(G2_NB2)
            g2_active = fx.Int32(1) == fx.Int32(1)
            while g2_active:
                fx.barrier()
                if tid == fx.Int32(0):
                    fx.ptr_store(
                        fx.Int32(comm_ops.atomic_add_agent(g2_head, fx.Int32(1))),
                        work_word,
                    )
                fx.barrier()
                g2_u = fx.Int32(
                    fx.rocdl.readfirstlane(
                        T.i32, fx.Int32(fx.ptr_load(work_word, result_type=T.i32))
                    )
                )
                g2_has = g2_u < g2_total
                if g2_has:
                    g2_bm = g2_u // fx.Int32(G2_NB2)
                    g2_rb, g2_ex, g2_nv = tile_find(
                        g1_tab, g2_bm, g1_lane, SBM=sort_block_m, experts=fz_epr
                    )
                    if tid == fx.Int32(0):
                        g2_st = g2_rb // fx.Int32(sort_block_m)
                        comm_ops.wait_i32_until_greater_than(
                            addr_mtile_ctr + fx.Int64(g2_st) * fx.Int64(4),
                            fx.Int32(G1_PUB_TARGET - 1),
                        )
                        if g2_nv > fx.Int32(sort_block_m):
                            comm_ops.wait_i32_until_greater_than(
                                addr_mtile_ctr
                                + fx.Int64(g2_st + fx.Int32(1)) * fx.Int64(4),
                                fx.Int32(G1_PUB_TARGET - 1),
                            )
                        comm_ops.fence_agent_acquire()
                    fx.barrier()
                    g2_tile(g2_bm, g2_u % fx.Int32(G2_NB2), g1_tab)
                g2_active = g2_has

    @flyc.jit
    def launch(
        x: fx.Tensor,
        w: fx.Tensor,
        scale_w: fx.Tensor,
        out: fx.Tensor,
        out_scale: fx.Tensor,
        xw2: fx.Tensor,
        xw2s: fx.Tensor,
        tokens: fx.Int32,
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64,
        addr_in_wts: fx.Int64,
        addr_in_sc: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_mtile_ctr: fx.Int64,
        s2_aq: fx.Int64,
        s2_ascale: fx.Int64,
        scale_raw: fx.Int64,
        s2_stids: fx.Int64,
        s2_sweights: fx.Int64,
        s2_trb: fx.Int64,
        s2_p2p_comb_inp: fx.Int64,
        ptr_tab: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            x,
            w,
            scale_w,
            out,
            out_scale,
            xw2,
            xw2s,
            tokens,
            addr_disp,
            i32_cur_tok,
            addr_in_tok,
            addr_in_idx,
            addr_in_wts,
            addr_in_sc,
            addr_parity,
            addr_expected,
            addr_mtile_ctr,
            s2_aq,
            s2_ascale,
            scale_raw,
            s2_stids,
            s2_sweights,
            s2_trb,
            s2_p2p_comb_inp,
            ptr_tab,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch
