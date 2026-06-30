"""cco-LSA intranode combine kernel (FlyDSL), bf16 — P2P-read (gather) path.

Inverse of dispatch. Matches mori's UseP2PRead intranode combine: ONE remote
pass (Stage-2 gather reads), no Stage-1 P2P scatter.

  Stage 1  cross-device entry barrier: grid barrier (comb_bar) + per-peer u64
           flag exchange on xdb_mem, so every peer's expert-output buffer
           (out_tok, written by the prior dispatch/expert kernel) is visible.
  Stage 2  warp-partitioned gather: for each local token, decode its k expert
           dests (dest_pe, dest_tok_id) from tok_map and read the k expert
           outputs REMOTELY from peer[dest_pe].out_tok[dest_tok_id]; reduce in
           f32 (vec4 = 8 bf16 per step), narrow to bf16, write comb_out.

vs the earlier scatter path (Stage-1 remote write + Stage-3 local read = two
serial passes), gather is one remote pass ≈ dispatch bandwidth.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, T, vector
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
import flydsl_prims as P

_V2BF16 = lambda: T.VectorType.get([2], T.bf16())
_V2F32 = lambda: T.VectorType.get([2], T.f32())
_V1I32 = lambda: T.VectorType.get([1], T.i32())


def _to_accum2(i32_scalar):        # i32 (2 bf16) -> v2f32
    return vector.bitcast(_V2BF16(), vector.from_elements(_V1I32(), [i32_scalar])).extf(_V2F32())


def _from_accum2(v2f32):           # v2f32 -> i32 (2 bf16)
    return vector.extract(vector.bitcast(_V1I32(), v2f32.truncf(_V2BF16())), static_position=[0])


def make_combine(*, rank, npes, experts_per_token, hidden_dim, hidden_elem_size,
                 max_tok_per_rank, max_recv, block_num, warp_num_per_block,
                 off_out_tok, off_xdb_mem, reset_total_recv=True, _s3_cache=2, _unroll=2):
    assert hidden_elem_size == 2, "basic combine path is bf16-only"
    nbytes = hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4
    n_chunks = nbytes // 16          # vec4 (16B) chunks per token

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def ep_combine(arena: Int64, addr_tok_map: Int64, addr_comb_bar: Int64,
                   addr_xdb_flag: Int64, addr_total_recv: Int64, addr_out: Int64,
                   my_lsa_rank: Int32, cur_rank_num_token: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        grid_thread_id = bid * (warp_num_per_block * 64) + tid

        w = cco.Window(arena)
        _r_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        _r_comb_bar = create_buffer_resource_from_addr(addr_comb_bar)
        _r_trecv = create_buffer_resource_from_addr(addr_total_recv)
        rsrc_out = create_buffer_resource_from_addr(addr_out)
        xdb_cur_flag = P.load_i64_acquire(fx.Int64(addr_xdb_flag))

        # ── Stage 1: cross-device entry barrier ──
        # Gather reads only out_tok from the prior dispatch/expert kernel, so a
        # cross-device barrier suffices (no Stage-1 scatter). The grid barrier
        # (comb_bar) is REQUIRED: it guarantees every block has already read
        # xdb_cur_flag before block 0 increments xdb_flag — otherwise a slow
        # block reads the bumped flag and the per-peer == handshake deadlocks.
        fx.barrier()
        if tid == 0:
            P.atomic_add_global(fx.Int64(addr_comb_bar), arith.constant(1))
        if grid_thread_id < npes:
            P.spin_until_eq_i32(fx.Int64(addr_comb_bar), block_num)
            P.fence_system_acquire()
            buffer_store(arith.constant(0), _r_comb_bar, 0)
            xdb_remote = (fx.Int64(w.lsa_ptr(grid_thread_id, off_xdb_mem))
                          + fx.Int64(rank) * fx.Int64(8))
            P.store_i64_system(xdb_remote, arith.constant(0), xdb_cur_flag)
        if grid_thread_id == 0:
            P.atomic_add_global(fx.Int64(addr_xdb_flag), arith.constant(1, type=T.i64()))
        if tid < npes:
            xdb_peer_slot = fx.Int64(w.lsa_ptr(my_lsa_rank, off_xdb_mem)) + fx.Int64(tid) * fx.Int64(8)
            P.spin_until_eq_i64(xdb_peer_slot, xdb_cur_flag)
            P.fence_system_acquire()
        fx.barrier()
        P.fence_system_acquire()           # ALL threads: peers' out_tok visible
        if const_expr(reset_total_recv):
            if tid == 0:
                buffer_store(arith.constant(0), _r_trecv, 0)

        # ── Stage 2: warp-partitioned remote gather + f32 accumulate ──
        # Register-light i32 (2 bf16, v2f32) reads + `_unroll`-way unroll: each
        # lane keeps `_unroll` independent loads/accumulators in flight per k so
        # a warp hides xGMI read latency with fewer warps (mori WarpAccum,
        # VecBytes=4, Unroll=2). Partition each token's hidden across
        # warps_per_tok warps so small batches still fill the grid.
        STEP = _unroll * 64
        safe_tok = arith.select(cur_rank_num_token == arith.constant(0),
                                arith.constant(1), cur_rank_num_token)
        warps_per_tok = (arith.constant(global_warp_num) + safe_tok - arith.constant(1)) // safe_tok
        units_per_warp = (arith.constant(n_i32) + warps_per_tok - arith.constant(1)) // warps_per_tok
        s3_total = cur_rank_num_token * warps_per_tok
        for s3_idx in range(global_warp_id, s3_total, global_warp_num):
            tok_id = s3_idx // warps_per_tok
            part_id = s3_idx % warps_per_tok
            unit_base = part_id * units_per_warp
            tm_base = tok_id * experts_per_token
            expert_rsrcs = []
            expert_vlds = []
            for k_slot in range_constexpr(experts_per_token):
                enc_k = buffer_load(_r_tok_map, tm_base + k_slot, vec_width=1, dtype=T.i32())
                dest_pe_k = enc_k // max_recv             # sentinel: dest_pe == npes
                dest_tok_k = enc_k % max_recv             # peer-local recv slot
                vld_k = dest_pe_k < npes
                safe_pe = arith.select(vld_k, dest_pe_k, arith.constant(rank))
                safe_tok_k = arith.select(vld_k, dest_tok_k, arith.constant(0))
                # REMOTE: peer[dest_pe].out_tok[dest_tok_id]
                slot_addr = (fx.Int64(w.lsa_ptr(safe_pe, off_out_tok))
                             + fx.Int64(safe_tok_k) * fx.Int64(nbytes))
                expert_rsrcs.append(create_buffer_resource_from_addr(slot_addr))
                expert_vlds.append(vld_k)
            rem = arith.constant(n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)   # i32 units this warp
            out_base = tok_id * n_i32
            # Nested fn: closure over expert_rsrcs/vlds (lists can't be loop-carried).
            def _one(off):           # reduce k contributions for one i32 (2 bf16) element
                acc = _to_accum2(arith.constant(0))
                for k_slot in range_constexpr(experts_per_token):
                    v = buffer_load(expert_rsrcs[k_slot], off, vec_width=1, dtype=T.i32(),
                                    cache_modifier=_s3_cache)
                    v = arith.select(expert_vlds[k_slot], v, arith.constant(0))
                    acc = acc + _to_accum2(v)
                buffer_store(_from_accum2(acc), rsrc_out, out_base + off)

            def _accum_loop():
                # main: _unroll independent elements per lane per iter
                main_end = (eff // STEP) * STEP
                for u in range(lane, main_end, STEP):
                    accs = []
                    base = unit_base + u
                    for r in range_constexpr(_unroll):
                        accs.append(_to_accum2(arith.constant(0)))
                    for k_slot in range_constexpr(experts_per_token):
                        vld = expert_vlds[k_slot]
                        rsc = expert_rsrcs[k_slot]
                        for r in range_constexpr(_unroll):
                            v = buffer_load(rsc, base + r * 64, vec_width=1, dtype=T.i32(),
                                            cache_modifier=_s3_cache)
                            accs[r] = accs[r] + _to_accum2(arith.select(vld, v, arith.constant(0)))
                    for r in range_constexpr(_unroll):
                        buffer_store(_from_accum2(accs[r]), rsrc_out, out_base + base + r * 64)
                # tail: leftover elements, one per lane
                for u in range(main_end + lane, eff, 64):
                    _one(unit_base + u)
            _accum_loop()

    @flyc.jit
    def run(arena: Int64, addr_tok_map: Int64, addr_comb_bar: Int64, addr_xdb_flag: Int64,
            addr_total_recv: Int64, addr_out: Int64, my_lsa_rank: Int32,
            cur_rank_num_token: Int32, stream=fx.Stream(None)):
        ep_combine(arena, addr_tok_map, addr_comb_bar, addr_xdb_flag, addr_total_recv,
                   addr_out, my_lsa_rank, cur_rank_num_token).launch(
            grid=(block_num, 1, 1), block=[warp_num_per_block * 64, 1, 1], stream=stream)

    return run
