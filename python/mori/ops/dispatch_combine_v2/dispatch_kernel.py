"""cco-LSA intranode dispatch kernel (FlyDSL), bf16 basic path.

Port of FlyDSL's ep_dispatch_intranode (phases 1-3; no quant/scales/StdMoE),
with every P2P peer-pointer table replaced by cco's single-window flat-VA
addressing: peer `pe`'s copy of arena region R lives at
    cco.Window(arena).lsa_ptr(pe, OFF_R)  (+ element/byte offset)

Config is baked as compile-time constants (DeepEP-style: JIT one kernel per
config). Symmetric buffers live in the SymmArena; local metadata (inputs,
dest_tok_map, counters, barriers, total_recv) are plain device tensors.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, T
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from flydsl.expr.rocdl import ballot, readlane
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
import flydsl_prims as P


def make_dispatch(*, rank, npes, experts_per_rank, experts_per_token, hidden_dim,
                  hidden_elem_size, max_tok_per_rank, max_recv, block_num, warp_num_per_block,
                  off_tok_off, off_recv_num, off_tis, off_out_idx, off_out_wts, off_out_tok,
                  off_out_scales=0, scale_dim=0, scale_type_size=0, enable_signal=True):
    nbytes = hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4
    sentinel_val = npes * max_recv
    # Optional per-token scales (e.g. fp4/blockwise quant inputs): forwarded
    # verbatim alongside the token to the dest peer's out_scales (mori parity).
    scale_bytes = scale_dim * scale_type_size
    scale_n_i32 = (scale_bytes + 3) // 4
    enable_scales = scale_bytes > 0

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def ep_dispatch(arena: Int64, addr_inp_tok: Int64, addr_inp_idx: Int64, addr_inp_wts: Int64,
                    addr_tok_map: Int64, addr_dest_pe_ctr: Int64, addr_disp_bar: Int64,
                    addr_total_recv: Int64, addr_inp_scales: Int64,
                    my_lsa_rank: Int32, inp_cur_tok: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        work_limit = inp_cur_tok * experts_per_token

        w = cco.Window(arena)
        _r_idx = create_buffer_resource_from_addr(addr_inp_idx)
        _r_wts = create_buffer_resource_from_addr(addr_inp_wts)
        _r_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        _r_dest_ctr = create_buffer_resource_from_addr(addr_dest_pe_ctr)
        _r_disp_bar = create_buffer_resource_from_addr(addr_disp_bar)

        # ── Phase 1: P2P-scatter each (src_tok, k_slot) to its dest PE ──
        for work_idx in range(global_warp_id, work_limit, global_warp_num):
            src_tok = work_idx // experts_per_token
            k_slot = work_idx % experts_per_token
            dest_expert = buffer_load(_r_idx, work_idx, vec_width=1, dtype=T.i32())
            safe_lane = arith.select(lane < k_slot, lane, 0)
            lane_expert = buffer_load(_r_idx, src_tok * experts_per_token + safe_lane,
                                      vec_width=1, dtype=T.i32())
            dest_pe = dest_expert // experts_per_rank
            lane_dest_pe = lane_expert // experts_per_rank
            dup_per_lane = arith.select(lane_dest_pe == dest_pe,
                                        arith.select(lane < k_slot, lane, 64), 64)
            dup_ballot = ballot(T.i64(), dup_per_lane < 64)
            is_dup = dup_ballot != 0

            dest_tok_lane0 = arith.constant(0)
            if lane == 0:
                if dup_ballot == 0:
                    peer_tok_off = fx.Int64(w.lsa_ptr(dest_pe, off_tok_off))
                    dest_tok_lane0 = P.atomic_add_global(peer_tok_off, fx.Int32(1))
            dest_tok_id = readlane(T.i32(), dest_tok_lane0, 0)

            overflow = dest_tok_id >= max_recv
            is_dup_or_overflow = arith.select(is_dup, is_dup, overflow)
            no_dup = dup_ballot == 0
            in_cap = dest_tok_id < max_recv
            do_publish = arith.select(no_dup, in_cap, no_dup)

            tok_map_entry = arith.select(is_dup_or_overflow, sentinel_val,
                                         dest_pe * max_recv + dest_tok_id)
            if lane == 0:
                buffer_store(tok_map_entry, _r_tok_map, work_idx)
                if do_publish:
                    src_tok_enc = rank * max_tok_per_rank + src_tok
                    peer_tis = fx.Int64(w.lsa_ptr(dest_pe, off_tis))
                    buffer_store(src_tok_enc, create_buffer_resource_from_addr(peer_tis), dest_tok_id)
                    dest_ctr_addr = fx.Int64(addr_dest_pe_ctr) + fx.Int64(dest_pe) * fx.Int64(4)
                    P.atomic_add_global(dest_ctr_addr, fx.Int32(1))

            # Per-lane (weight, expert-idx) scatter (lanes < k).
            if lane < experts_per_token:
                if do_publish:
                    wt_src_off = src_tok * experts_per_token + lane
                    wt_val = buffer_load(_r_wts, wt_src_off, vec_width=1, dtype=T.f32())
                    idx_val = buffer_load(_r_idx, wt_src_off, vec_width=1, dtype=T.i32())
                    dest_slot = dest_tok_id * experts_per_token + lane
                    peer_wts = fx.Int64(w.lsa_ptr(dest_pe, off_out_wts))
                    buffer_store(arith.bitcast(T.i32(), wt_val),
                                 create_buffer_resource_from_addr(peer_wts), dest_slot)
                    peer_idx = fx.Int64(w.lsa_ptr(dest_pe, off_out_idx))
                    buffer_store(idx_val, create_buffer_resource_from_addr(peer_idx), dest_slot)

            # Per-token scales scatter: forward the src token's scale_n_i32 dwords
            # to the dest peer's out_scales[dest_tok_id] (lane-strided to cover
            # scale_dim > one wavefront). Verbatim copy (opaque bytes).
            if const_expr(enable_scales):
                if do_publish:
                    _r_inp_sc = create_buffer_resource_from_addr(addr_inp_scales)
                    peer_sc = fx.Int64(w.lsa_ptr(dest_pe, off_out_scales))
                    _r_peer_sc = create_buffer_resource_from_addr(peer_sc)
                    for k_off in range(lane, scale_n_i32, 64):
                        sc_val = buffer_load(_r_inp_sc, src_tok * scale_n_i32 + k_off,
                                             vec_width=1, dtype=T.i32())
                        buffer_store(sc_val, _r_peer_sc, dest_tok_id * scale_n_i32 + k_off)

            # Token-embedding scatter: each lane owns 4 i32 (16B). Dual-issue the
            # main body (2 vec4 loads then 2 vec4 stores, stride 512 i32) for
            # memory-level parallelism; a stride-256 tail covers the remainder.
            # Dropped slots (dup/overflow) set copy_end == lane_i32_off → no-op.
            peer_tok_base = fx.Int64(w.lsa_ptr(dest_pe, off_out_tok))
            remote_tok_addr = peer_tok_base + fx.Int64(dest_tok_id) * fx.Int64(nbytes)
            local_tok_addr = fx.Int64(addr_inp_tok) + fx.Int64(src_tok) * fx.Int64(nbytes)
            rsrc_src = create_buffer_resource_from_addr(local_tok_addr)
            rsrc_dst = create_buffer_resource_from_addr(remote_tok_addr)
            lane_i32_off = lane * 4
            safe_end_i32 = (n_i32 // 512) * 512
            if const_expr(n_i32 >= 512 and safe_end_i32 > 0):
                copy_end_main = arith.select(is_dup_or_overflow, lane_i32_off, safe_end_i32)
                for chunk in range(lane_i32_off, copy_end_main, 512):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    vec_b = buffer_load(rsrc_src, chunk + 256, vec_width=4, dtype=T.i32())
                    buffer_store(vec_a, rsrc_dst, chunk)
                    buffer_store(vec_b, rsrc_dst, chunk + 256)
            if const_expr(safe_end_i32 < n_i32):
                copy_end_tail = arith.select(is_dup_or_overflow, lane_i32_off, n_i32)
                for chunk in range(lane_i32_off + safe_end_i32, copy_end_tail, 256):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    buffer_store(vec_a, rsrc_dst, chunk)
            elif const_expr(n_i32 < 512):
                copy_end_small = arith.select(is_dup_or_overflow, lane_i32_off, n_i32)
                for chunk in range(lane_i32_off, copy_end_small, 256):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    buffer_store(vec_a, rsrc_dst, chunk)

        if const_expr(enable_signal):
            # ── Phase 2: grid barrier + per-peer count signal ──
            fx.barrier()
            if tid == 0:
                P.atomic_add_global(fx.Int64(addr_disp_bar), arith.constant(1))

            local_recv_num = fx.Int64(w.lsa_ptr(my_lsa_rank, off_recv_num))
            for dest_pe in range(lane, npes, 64):
                if global_warp_id == 0:
                    P.spin_until_eq_i32(fx.Int64(addr_disp_bar), block_num)
                    P.fence_system_acquire()
                    buffer_store(arith.constant(0), _r_disp_bar, 0)
                    signal_value = buffer_load(_r_dest_ctr, dest_pe, vec_width=1, dtype=T.i32()) + 1
                    peer_recv_num = fx.Int64(w.lsa_ptr(dest_pe, off_recv_num))
                    recv_num_remote_addr = peer_recv_num + fx.Int64(rank) * fx.Int64(4)
                    P.spin_until_eq_i32(recv_num_remote_addr, 0)
                    P.store_i32_system(recv_num_remote_addr, arith.constant(0), signal_value)

            # ── Phase 3: collect per-source counts into total_recv ──
            for src_pe in range(lane, npes, 64):
                if global_warp_id == 0:
                    recv_num_src_addr = local_recv_num + fx.Int64(src_pe) * fx.Int64(4)
                    signal_value = P.spin_until_gt_i32(recv_num_src_addr, 0)
                    peer_recv_count = signal_value - 1
                    P.store_i32_system(recv_num_src_addr, arith.constant(0), arith.constant(0))
                    P.atomic_add_global(fx.Int64(addr_total_recv), peer_recv_count)
                    buffer_store(arith.constant(0), _r_dest_ctr, src_pe)

            if global_warp_id == 0:
                if lane == 0:
                    local_tok_off = fx.Int64(w.lsa_ptr(my_lsa_rank, off_tok_off))
                    P.store_i32_system(local_tok_off, arith.constant(0), arith.constant(0))

    @flyc.jit
    def run(arena: Int64, addr_inp_tok: Int64, addr_inp_idx: Int64, addr_inp_wts: Int64,
            addr_tok_map: Int64, addr_dest_pe_ctr: Int64, addr_disp_bar: Int64,
            addr_total_recv: Int64, addr_inp_scales: Int64, my_lsa_rank: Int32,
            inp_cur_tok: Int32, stream=fx.Stream(None)):
        ep_dispatch(arena, addr_inp_tok, addr_inp_idx, addr_inp_wts, addr_tok_map,
                    addr_dest_pe_ctr, addr_disp_bar, addr_total_recv, addr_inp_scales,
                    my_lsa_rank, inp_cur_tok).launch(
            grid=(block_num, 1, 1), block=[warp_num_per_block * 64, 1, 1], stream=stream)

    return run
