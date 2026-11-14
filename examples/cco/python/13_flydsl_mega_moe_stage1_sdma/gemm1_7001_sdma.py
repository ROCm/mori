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
"""``--gemm1 7001``: aiter#7001's fused Stage1 GEMM1 tiles behind the SDMA dispatch.

One persistent launch, after aiter's compact prepare (plan and quantised rows). It is
aiter#7001's compact ``compile_mega_moe_stage1`` with the CU dispatch replaced:

* workgroups 1..``producers`` pack this rank's rows, one (destination, segment) at a time in
  expert order (group segment first): rows for a peer into the send slab of the cco window,
  this rank's own rows straight into its landing rows, route headers to the destination over
  P2P (as ``pack.build_pack``); a flag per segment says it is packed;
* workgroup 0, one wave per peer, puts each of the peer's segments with SDMA as soon as its
  flag is up (the fp8 rows, then the e8m0 carrying the remote signal), and waits for its
  puts at the end;
* every workgroup then takes GEMM1 work items, #7001's own queue and tiles
  (``build_fused_gemm1``, the tuned compact route-major FP8 bodies). An item waits until
  every peer has signalled the group segment and experts 0..e of its M-tile, and this rank's
  own segments of e are packed, where #7001 waits for its producers' tile-ready counters.

So the Stage1 output matches aiter#7001's byte for byte. Needs aiter with aiter#7001's
``mega_moe/gemm1.py``; the rows come from the prepare's quantisation (no ``--fuse-quant``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mega_moe import envs
from aiter.ops.flydsl.kernels.mega_moe.dispatch import DispatchSlot, _load_fanout_pair
from aiter.ops.flydsl.kernels.mega_moe.gemm1 import (
    _LdsF32View,
    build_fused_gemm1,
    gemm1_a_lds_stages,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_load, ptr_buf_tensor

_SC0_CACHE = 1


def compile_gemm1_7001_sdma(
    *,
    model_dim,
    inter_dim,
    rank,
    experts_per_rank,
    fuse_npes,
    fuse_mtpr,
    sort_block_m,
    tile_n,
    tile_k,
    num_waves,
    pipe_weights,
    mfma_amajor,
    swizzle_a,
    async_a_copy,
    use_tile_resource,
    waves_per_eu_hint,
    num_cu,
    b_nt,
    work_shards,
    swiglu_limit,
    off_slab_x,
    off_slab_sc,
    off_land_x,
    off_land_sc,
    off_spare,
    topk=6,
    producers=32,
):
    """The launcher for one Stage1 config (``sort_block_m`` .. ``work_shards`` as aiter's
    Stage1Config) and this rank's cco window offsets. ``producers``: packing workgroups.
    """
    NUM_WAVES = int(num_waves)
    P = int(producers)
    PACK_B = (
        4  # rows per step of a packing wave: their loads all go out before the stores
    )
    assert NUM_WAVES == 8, "the comm role puts with one wave per peer"
    assert P > 0 and model_dim % 1024 == 0
    assert tile_k == 256
    n_per_wave = tile_n // NUM_WAVES
    N_TILES = (2 * inter_dim) // tile_n
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    A_K_STEP_BYTES = tile_k  # fp8
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    WORK_SHARDS = int(work_shards) if work_shards is not None else 4
    K = model_dim
    npes, epr = int(fuse_npes), int(experts_per_rank)
    te = npes * epr

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_size = sort_block_m * (tile_n // 2)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    a_lds_stages = gemm1_a_lds_stages(
        tuned=True,
        sort_block_m=sort_block_m,
        pipe_weights=pipe_weights,
        async_a_copy=async_a_copy,
        indexed_input=False,
        a_dtype="fp8",
        total_threads=TOTAL_THREADS,
    )
    lds_pool_bytes = max(a_lds_stages * a_lds_size, cs_size * 4)
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fuse_mtpr <= 512 else 0)
    EPI_EVEC = envs.AITER_MEGA_S1_EPI_EVEC

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
        bc_work: fx.Array[fx.Int8, 16, 16]

    sl = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    kernel_name = (
        f"sdma_s1_g7001_p{P}_r{rank}_t{sort_block_m}x{tile_n}x{tile_k}_w{NUM_WAVES}"
        f"_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}_bnt{b_cache_modifier}_ws{WORK_SHARDS}_ev{EPI_EVEC}{sl}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor,
        x: fx.Tensor,
        w: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor,
        tokens: fx.Int32,
        addr_disp: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        win: fx.Int64,
        dev_comm: fx.Int64,
        sig_base: fx.Int64,
        x_q: fx.Int64,
        x_sc: fx.Int64,
        wts_in: fx.Int64,
        ids_in: fx.Int64,
        seg_done: fx.Int64,
        win_local: fx.Int64,
    ):
        import mori.cco.device.flydsl as cco

        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        disp_rsrc = ptr_buf_tensor(addr_disp, fx.Int64)
        parity_rsrc = ptr_buf_tensor(addr_parity, fx.Int32)
        expected_rsrc = ptr_buf_tensor(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return disp_rsrc[fx.Int32(int(slot))]

        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        payload_parity = buf_copy_load(
            parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE
        )
        payload_expected = buf_copy_load(
            expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE
        )
        wave_id = fx.thread_idx.x // 64

        w_rsrc = ptr_buf_tensor(fx.get_iter(w), fx.Int32, unit_elems=4)
        sx_rsrc = ptr_buf_tensor(fx.get_iter(scale_x), fx.Int32, unit_elems=4)
        sw_rsrc = ptr_buf_tensor(fx.get_iter(scale_w), fx.Int32)
        trb_rsrc = ptr_buf_tensor(fx.get_iter(sorted_token_ids), fx.Int32)
        tib_rsrc = ptr_buf_tensor(_disp_ptr(DispatchSlot.TILE_INPUT_BASE), fx.Int32)
        srcmap_rsrc = ptr_buf_tensor(_disp_ptr(DispatchSlot.SRCMAP), fx.Int32)
        expert_rsrc = ptr_buf_tensor(fx.get_iter(expert_ids), fx.Int32)
        nv_rsrc = ptr_buf_tensor(fx.get_iter(num_valid_ids), fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + 8192
        assert use_tile_resource
        out_rsrc = None
        os_rsrc = ptr_buf_tensor(
            fx.get_iter(out_scale), fx.Int8, num_records_bytes=os_nbytes
        )

        _, _do_scheduled_tile = build_fused_gemm1(
            x_tensor=x,
            w_rsrc=w_rsrc,
            sw_rsrc=sw_rsrc,
            sx_rsrc=sx_rsrc,
            out_rsrc=out_rsrc,
            os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc,
            expert_rsrc=expert_rsrc,
            out_tensor=out,
            tib_rsrc=tib_rsrc,
            a_buf=a_buf,
            a_scale_lds=a_scale_lds,
            c_tile=c_tile,
            model_dim=model_dim,
            inter_dim=inter_dim,
            sort_block_m=sort_block_m,
            tile_n=tile_n,
            num_waves=NUM_WAVES,
            n_per_wave=n_per_wave,
            wave_id=wave_id,
            m_repeat=M_REPEAT,
            num_acc_n=NUM_ACC_N,
            a_k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS,
            k_iters=K_ITERS,
            a_lds_i32=a_lds_i32,
            n_tiles=N_TILES,
            expert_offset=rank * epr,
            b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a,
            pipe_weights=pipe_weights,
            mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy,
            use_tile_resource=use_tile_resource,
            pair_k=False,
            evec=EPI_EVEC,
            indirect_input=True,
            indexed_input=False,
            row_map_rsrc=srcmap_rsrc,
            source_rows=0,
            a_dtype="fp8",
            out_dtype="fp8",
            swiglu_limit=swiglu_limit,
            tuned=True,
        )

        # This rank's Stage1 plan of this call (prepare flips the parity), as #7001.
        if tid == fx.Int32(0):
            local_plan_ready = disp_rsrc[fx.Int32(int(DispatchSlot.PLAN_READY))]
            ready_index = payload_parity * fx.Int32(npes) + fx.Int32(rank)
            comm_ops.wait_i32_until_equals(
                local_plan_ready + fx.Int64(ready_index) * 4, payload_expected
            )
            comm_ops.fence_system_acquire()
        fx.barrier()

        sig0 = fx.Int64(
            rocdl.readfirstlane(
                T.i64,
                (
                    fx.Int64(comm_ops.load_i64_global(sig_base)) - fx.Int64(epr + 1)
                ).ir_value(),
            )
        )
        # One value per call for the segment-packed flags (the signal target advances by
        # epr + 1 per call).
        epoch = fx.Int32(sig0) + fx.Int32(1)
        lane = fx.thread_idx.x % fx.Int32(64)
        peer = fx.Int32(rocdl.readfirstlane(T.i32, fx.thread_idx.x // fx.Int32(64)))
        is_comm = fx.Int32(fx.block_idx.x) == fx.Int32(0)
        bid = fx.Int32(fx.block_idx.x)
        if (bid >= fx.Int32(1)) & (bid <= fx.Int32(P)):
            # Producers: segment items in expert order (group segment first), one
            # destination after another; item k -> segment slot k // npes, dest k % npes.
            wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.thread_idx.x // fx.Int32(64)))
            pdisp = ptr_buf_tensor(addr_disp, fx.Int64)

            def pdp(slot):
                return pdisp[fx.Int32(int(slot))]

            p_hist = ptr_buf_tensor(pdp(DispatchSlot.LOCAL_HIST), fx.Int32)
            p_pair_base = ptr_buf_tensor(pdp(DispatchSlot.PAIR_BASE), fx.Int32)
            p_task_base = ptr_buf_tensor(pdp(DispatchSlot.TASK_ROW_BASE), fx.Int32)
            p_group_base = ptr_buf_tensor(pdp(DispatchSlot.GROUP_TASK_BASE), fx.Int32)
            p_order = ptr_buf_tensor(pdp(DispatchSlot.PAIR_ORDER), fx.Int32)
            p_srcmap = ptr_buf_tensor(pdp(DispatchSlot.P2P_SRCMAP), fx.Int64)
            p_weight = ptr_buf_tensor(pdp(DispatchSlot.P2P_WEIGHT), fx.Int64)
            rs_x = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x_q).ir_value()))
            )
            rs_sc = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x_sc).ir_value()))
            )
            wts_p = ptr_buf_tensor(wts_in, fx.Float32)
            idx_p = ptr_buf_tensor(ids_in, fx.Int32)
            PER = K // 1024  # 16-byte units per lane of a row
            SC_I32 = K // 128
            ITEMS = (epr + 1) * npes
            for k in range(bid - fx.Int32(1), fx.Int32(ITEMS), fx.Int32(P)):
                jj = k // fx.Int32(npes)
                pdest = k - jj * fx.Int32(npes)
                pgroup = jj == fx.Int32(0)
                pseg = pgroup.select(
                    fx.Int32(te) + pdest, pdest * fx.Int32(epr) + jj - fx.Int32(1)
                )
                _, pair_a, _, _, _ = _load_fanout_pair(
                    pdp(DispatchSlot.FANOUT_PAIR_CONFIG),
                    pdest,
                    payload_parity,
                    npes=npes,
                )
                pcount = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(p_hist[pseg])))
                psrc_base = fx.Int32(
                    rocdl.readfirstlane(T.i32, fx.Int32(p_pair_base[pseg]))
                )
                pdst_base = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32,
                        pgroup.select(
                            fx.Int32(p_group_base[pdest * fx.Int32(epr) + pair_a]),
                            fx.Int32(p_task_base[pseg]),
                        ),
                    )
                )
                pown = pdest == fx.Int32(rank)
                pout_x = create_buffer_resource_from_addr(
                    fx.Int64(
                        rocdl.readfirstlane(
                            T.i64,
                            fx.Int64(
                                pown.select(
                                    win_local + fx.Int64(off_land_x),
                                    win_local + fx.Int64(off_slab_x),
                                )
                            ).ir_value(),
                        )
                    )
                )
                pout_sc = create_buffer_resource_from_addr(
                    fx.Int64(
                        rocdl.readfirstlane(
                            T.i64,
                            fx.Int64(
                                pown.select(
                                    win_local + fx.Int64(off_land_sc),
                                    win_local + fx.Int64(off_slab_sc),
                                )
                            ).ir_value(),
                        )
                    )
                )
                srcmap_remote = ptr_buf_tensor(p_srcmap[pdest], fx.Int32)
                weight_remote = ptr_buf_tensor(p_weight[pdest], fx.Int32)
                n_wave = (pcount - wave + fx.Int32(NUM_WAVES - 1)) // fx.Int32(
                    NUM_WAVES
                )
                for g in range(fx.Int32(0), n_wave, fx.Int32(64)):
                    n_g = n_wave - g
                    n_g = (n_g < fx.Int32(64)).select(n_g, fx.Int32(64))
                    valid = lane < n_g
                    row_l = wave + (g + valid.select(lane, fx.Int32(0))) * fx.Int32(
                        NUM_WAVES
                    )
                    wk_l = fx.Int32(p_order[psrc_base + row_l])
                    token_l = pgroup.select(
                        wk_l & fx.Int32(0xFFFFFF), wk_l // fx.Int32(topk)
                    )
                    if valid:
                        source_key = fx.Int32(rank * fuse_mtpr) + token_l
                        if pgroup:
                            members = (wk_l >> fx.Int32(24)) & fx.Int32(0xFF)
                            for slot in range_constexpr(topk):
                                if (
                                    (members >> fx.Int32(slot)) & fx.Int32(1)
                                ) != fx.Int32(0):
                                    route = token_l * fx.Int32(topk) + fx.Int32(slot)
                                    member_ge = fx.Int32(idx_p[route])
                                    hdr_row = fx.Int32(p_group_base[member_ge]) + row_l
                                    srcmap_remote[hdr_row] = source_key | (
                                        fx.Int32(slot) << fx.Int32(24)
                                    )
                                    weight_remote[hdr_row] = fx.Float32(
                                        wts_p[route]
                                    ).bitcast(fx.Int32)
                        else:
                            slot = wk_l % fx.Int32(topk)
                            srcmap_remote[pdst_base + row_l] = source_key | (
                                slot << fx.Int32(24)
                            )
                            weight_remote[pdst_base + row_l] = fx.Float32(
                                wts_p[wk_l]
                            ).bitcast(fx.Int32)
                    for b0 in range(fx.Int32(0), n_g, fx.Int32(PACK_B)):
                        toks = []
                        out_rows = []
                        for jb in range_constexpr(PACK_B):
                            idx = b0 + fx.Int32(jb)
                            idx = (idx < n_g).select(idx, n_g - fx.Int32(1))
                            toks.append(fx.Int32(rocdl.readlane(T.i32, token_l, idx)))
                            prow = wave + (g + idx) * fx.Int32(NUM_WAVES)
                            out_rows.append(
                                pown.select(pdst_base + prow, psrc_base + prow)
                            )
                        vss = []
                        scs = []
                        for jb in range_constexpr(PACK_B):
                            vss.append(
                                [
                                    buffer_load(
                                        rs_x,
                                        toks[jb] * fx.Int32(K // 4)
                                        + (lane + fx.Int32(64 * i)) * fx.Int32(4),
                                        vec_width=4,
                                        dtype=T.i32,
                                    )
                                    for i in range_constexpr(PER)
                                ]
                            )
                            scs.append(
                                fx.Int32(
                                    buffer_load(
                                        rs_sc,
                                        toks[jb] * fx.Int32(SC_I32)
                                        + (lane < fx.Int32(SC_I32)).select(
                                            lane, fx.Int32(SC_I32 - 1)
                                        ),
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                            )
                        for jb in range_constexpr(PACK_B):
                            for i in range_constexpr(PER):
                                buffer_store(
                                    vss[jb][i],
                                    pout_x,
                                    out_rows[jb] * fx.Int32(K // 4)
                                    + (lane + fx.Int32(64 * i)) * fx.Int32(4),
                                )
                            buffer_store(
                                scs[jb],
                                pout_sc,
                                out_rows[jb] * fx.Int32(SC_I32)
                                + (lane < fx.Int32(SC_I32)).select(
                                    lane, fx.Int32(SC_I32 - 1)
                                ),
                            )
                # The segment is in memory (and its headers at the destination): flag it.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i32_system(seg_done, pseg, epoch)
                fx.barrier()
        put_lane = (
            is_comm
            & (peer < fx.Int32(npes))
            & (peer != fx.Int32(rank))
            & (lane == fx.Int32(0))
        )
        if put_lane:
            # This rank's rows for ``peer``, one put pair per segment (group segment first,
            # then experts 0..epr-1) as soon as it is packed, the e8m0 put signalled.
            disp = ptr_buf_tensor(addr_disp, fx.Int64)
            local_hist = ptr_buf_tensor(
                disp[fx.Int32(int(DispatchSlot.LOCAL_HIST))], fx.Int32
            )
            pair_base = ptr_buf_tensor(
                disp[fx.Int32(int(DispatchSlot.PAIR_BASE))], fx.Int32
            )
            task_base = ptr_buf_tensor(
                disp[fx.Int32(int(DispatchSlot.TASK_ROW_BASE))], fx.Int32
            )
            group_base = ptr_buf_tensor(
                disp[fx.Int32(int(DispatchSlot.GROUP_TASK_BASE))], fx.Int32
            )
            c_parity = fx.Int32(comm_ops.load_i32_system(addr_parity, fx.Int32(0)))
            _, pair_a, _, _, _ = _load_fanout_pair(
                disp[fx.Int32(int(DispatchSlot.FANOUT_PAIR_CONFIG))],
                peer,
                c_parity,
                npes=npes,
            )
            sd = cco.DevComm(dev_comm).sdma()
            q = fx.Int32(0)
            spare = fx.Int64(off_spare)
            for j in range(fx.Int32(0), fx.Int32(epr + 1), fx.Int32(1)):
                grp = j == fx.Int32(0)
                seg = grp.select(
                    fx.Int32(te) + peer, peer * fx.Int32(epr) + j - fx.Int32(1)
                )
                comm_ops.wait_i32_until_equals(seg_done + fx.Int64(seg) * 4, epoch)
                cnt = fx.Int32(local_hist[seg])
                src_row = fx.Int64(fx.Int32(pair_base[seg]))
                dst_row = fx.Int64(
                    grp.select(
                        fx.Int32(group_base[peer * fx.Int32(epr) + pair_a]),
                        fx.Int32(task_base[seg]),
                    )
                )
                has = cnt > fx.Int32(0)
                nx = fx.Int64(has.select(cnt * fx.Int32(K), fx.Int32(4)))
                nsc = fx.Int64(has.select(cnt * fx.Int32(K // 32), fx.Int32(4)))
                sx = has.select(fx.Int64(off_slab_x) + src_row * fx.Int64(K), spare)
                dx = has.select(fx.Int64(off_land_x) + dst_row * fx.Int64(K), spare)
                ss = has.select(
                    fx.Int64(off_slab_sc) + src_row * fx.Int64(K // 32), spare
                )
                ds = has.select(
                    fx.Int64(off_land_sc) + dst_row * fx.Int64(K // 32), spare
                )
                sd.put(
                    peer,
                    win,
                    dx,
                    win,
                    sx,
                    nx,
                    q,
                    coop=cco.CoopScope.THREAD,
                    local_signal=False,
                    remote_signal=False,
                )
                sd.put(
                    peer,
                    win,
                    ds,
                    win,
                    ss,
                    nsc,
                    q,
                    coop=cco.CoopScope.THREAD,
                    local_signal=False,
                    remote_signal=True,
                )

        # The peers' signal counts are one array indexed by the sender: its address and
        # stride, wave uniform, once per workgroup.
        sdh = cco.DevComm(dev_comm).sdma()
        slot0 = fx.Int64(
            rocdl.readfirstlane(T.i64, fx.Int64(sdh.signal_slot(0, 0)).ir_value())
        )
        slot1 = fx.Int64(
            rocdl.readfirstlane(T.i64, fx.Int64(sdh.signal_slot(1, 0)).ir_value())
        )
        slot_stride = slot1 - slot0

        def _wait_tile_sdma(work):
            """Every peer has put the group segment and experts 0..e of this call, e = the
            local expert of the work item's M-tile: all signal counts in one round trip,
            then a spin only on a short one."""
            e = fx.Int32(expert_rsrc[work // fx.Int32(N_TILES)]) - fx.Int32(rank * epr)
            e = (e < fx.Int32(0)).select(fx.Int32(epr - 1), e)
            e = (e < fx.Int32(epr)).select(e, fx.Int32(epr - 1))
            need = sig0 + fx.Int64(e + fx.Int32(2))
            addrs = [
                slot0 + slot_stride * fx.Int64(src)
                for src in range(npes)
                if src != rank
            ]
            sigs = [fx.Int64(comm_ops.load_i64_acquire(a)) for a in addrs]
            short = sigs[0] < need
            for v in sigs[1:]:
                short = short | (v < need)
            if short:
                for a in addrs:
                    comm_ops.spin_until_ge_i64(a, need)
            comm_ops.wait_i32_until_equals(seg_done + fx.Int64(te + rank) * 4, epoch)
            comm_ops.wait_i32_until_equals(
                seg_done + fx.Int64(fx.Int32(rank * epr) + e) * 4, epoch
            )

        num_valid = nv_rsrc[fx.Int32(0)]
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)
        consumer_id = fx.Int32(fx.block_idx.x)
        consumer_active = consumer_id < total_work
        work_scratch = fx.recast_iter(fx.Int32, lds.bc_work.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        work_shard = consumer_id & fx.Int32(WORK_SHARDS - 1)

        while consumer_active:
            if tid == fx.Int32(0):
                local_work = fx.Int32(
                    comm_ops.atomic_add_agent(
                        a_work_head + fx.Int64(work_shard) * 64, fx.Int32(1)
                    )
                )
                fetched = work_shard + local_work * fx.Int32(WORK_SHARDS)
                fx.ptr_store(Vec.from_elements([fetched], fx.Int32), work_scratch)
            fx.barrier()
            first_work = Vec(work_scratch_view.load())[0]
            if first_work < total_work:
                # As #7001: one system acquire by the waiting thread, the barrier orders
                # every other wave's loads after it.
                if tid == fx.Int32(0):
                    _wait_tile_sdma(first_work)
                    comm_ops.fence_system_acquire()
                fx.barrier()
                _do_scheduled_tile(first_work)
            consumer_active = first_work < total_work

        if put_lane:
            # Every put has read the slab before the next call's pack rewrites it.
            cco.DevComm(dev_comm).sdma().quiet_queue(peer, fx.Int32(0))

    @flyc.jit
    def launch(
        out: fx.Tensor,
        x: fx.Tensor,
        w: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor,
        tokens: fx.Int32,
        addr_disp: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        win: fx.Int64,
        dev_comm: fx.Int64,
        sig_base: fx.Int64,
        x_q: fx.Int64,
        x_sc: fx.Int64,
        wts_in: fx.Int64,
        ids_in: fx.Int64,
        seg_done: fx.Int64,
        win_local: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            out,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            num_valid_ids,
            out_scale,
            tokens,
            addr_disp,
            addr_parity,
            addr_expected,
            win,
            dev_comm,
            sig_base,
            x_q,
            x_sc,
            wts_in,
            ids_in,
            seg_done,
            win_local,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(num_cu, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch
