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
from flydsl.expr.rocdl import (
    cvt_pk_f32_fp8,
    cvt_pk_fp8_f32,
    cvt_scalef32_pk_f32_fp4,
    cvt_scalef32_pk_fp4_f32,
)
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
import flydsl_prims as P

_V2BF16 = lambda: T.VectorType.get([2], T.bf16())
_V2F32 = lambda: T.VectorType.get([2], T.f32())
_V4F32 = lambda: T.VectorType.get([4], T.f32())
_V8F32 = lambda: T.VectorType.get([8], T.f32())
_V1I32 = lambda: T.VectorType.get([1], T.i32())


def _accum_funcs(hidden_elem_size, fp8_direct_cast=False, fp4=False):
    if fp4:                            # fp4 e2m1: i32 = 8 packed fp4 -> v8f32
        # NOTE: cvt_scalef32_pk_*_fp4 are gfx950-only (MI350). On gfx942
        # (MI300X) codegen fails "instruction not supported on this GPU".
        # Faithful port of the FlyDSL reference fp4 branch; opt-in (fp4=True).
        def to_accum(i32_scalar):
            one = arith.constant(1.0, type=T.f32())
            pr = [cvt_scalef32_pk_f32_fp4(res=_V2F32(), src=i32_scalar, scale=one,
                                          src_sel_index=s) for s in range(4)]
            lo4 = vector.shuffle(pr[0], pr[1], [0, 1, 2, 3])
            hi4 = vector.shuffle(pr[2], pr[3], [0, 1, 2, 3])
            return vector.shuffle(lo4, hi4, [0, 1, 2, 3, 4, 5, 6, 7])

        def from_accum(acc):
            one = arith.constant(1.0, type=T.f32())
            old = arith.constant(0, type=T.i32())
            for s in range(4):
                fa = vector.extract(acc, static_position=[s * 2])
                fb = vector.extract(acc, static_position=[s * 2 + 1])
                old = cvt_scalef32_pk_fp4_f32(res=T.i32(), old_vdst=old, src0=fa, src1=fb,
                                              scale=one, dst_sel_index=s)
            return old

        def zero_accum():
            return arith.constant_vector(0.0, _V8F32())
        return to_accum, from_accum, zero_accum
    return _accum_funcs_int(hidden_elem_size, fp8_direct_cast)


def _accum_funcs_int(hidden_elem_size, fp8_direct_cast=False):
    """Return (to_accum, from_accum, zero_accum) for one i32 'unit' of the
    transport dtype, mirroring the FlyDSL reference's per-dtype branches.

    Each i32 packs: 2 bf16 (v2f32), 1 f32 (scalar), or 4 fp8 (v4f32).
    """
    if hidden_elem_size == 2:          # bf16: i32 = 2 bf16
        def to_accum(i32_scalar):
            return vector.bitcast(_V2BF16(),
                                  vector.from_elements(_V1I32(), [i32_scalar])).extf(_V2F32())

        def from_accum(acc):
            return vector.extract(vector.bitcast(_V1I32(), acc.truncf(_V2BF16())),
                                  static_position=[0])

        def zero_accum():
            return to_accum(arith.constant(0))
    elif hidden_elem_size == 4:        # f32: i32 = 1 f32
        def to_accum(i32_scalar):
            return fx.Float32(arith.bitcast(T.f32(), arith.unwrap(i32_scalar)))

        def from_accum(acc):
            return fx.Int32(arith.bitcast(T.i32(), arith.unwrap(acc)))

        def zero_accum():
            return fx.Float32(arith.constant(0.0, type=T.f32()))
    elif hidden_elem_size == 1:        # fp8 (OCP e4m3): i32 = 4 fp8
        def to_accum(i32_scalar):
            lo = cvt_pk_f32_fp8(res=_V2F32(), src=i32_scalar, word_sel=False)
            hi = cvt_pk_f32_fp8(res=_V2F32(), src=i32_scalar, word_sel=True)
            return vector.shuffle(lo, hi, [0, 1, 2, 3])

        def from_accum(acc):
            f0 = vector.extract(acc, static_position=[0])
            f1 = vector.extract(acc, static_position=[1])
            f2 = vector.extract(acc, static_position=[2])
            f3 = vector.extract(acc, static_position=[3])
            if fp8_direct_cast:        # wire fp8 -> external bf16: v4f32 -> v4bf16 -> 2 i32
                v4bf16 = acc.truncf(T.VectorType.get([4], T.bf16()))
                return vector.bitcast(T.VectorType.get([2], T.i32()), v4bf16)
            zero = arith.constant(0, type=T.i32())
            lo = cvt_pk_fp8_f32(res=T.i32(), src_a=f0, src_b=f1, old=zero, word_sel=False)
            return cvt_pk_fp8_f32(res=T.i32(), src_a=f2, src_b=f3, old=lo, word_sel=True)

        def zero_accum():
            return arith.constant_vector(0.0, _V4F32())
    else:
        raise ValueError(f"unsupported hidden_elem_size {hidden_elem_size}")
    return to_accum, from_accum, zero_accum


def make_combine(*, rank, npes, experts_per_token, hidden_dim, hidden_elem_size,
                 max_tok_per_rank, max_recv, block_num, warp_num_per_block,
                 off_out_tok, off_xdb_mem, off_out_wts=0, enable_weights=True,
                 fp8_direct_cast=False, fp4=False, reset_total_recv=True,
                 _s3_cache=2, _unroll=2):
    # Transport dtype = external dtype, except fp8_direct_cast wires fp8 while
    # the output (comb_out) stays bf16 (2 i32 per fp8 i32 unit). fp4: i32 = 8 fp4.
    _to_accum2, _from_accum2, _zero_accum = _accum_funcs(hidden_elem_size, fp8_direct_cast, fp4)
    nbytes = hidden_dim // 2 if fp4 else hidden_dim * hidden_elem_size
    n_i32 = hidden_dim // 8 if fp4 else nbytes // 4
    n_chunks = nbytes // 16          # vec4 (16B) chunks per token
    # fp8_direct_cast: output stride is bf16 (2 i32 per fp8 unit) vs input fp8.
    out_n_i32 = (hidden_dim * 2) // 4 if fp8_direct_cast else n_i32
    out_step_mult = 2 if fp8_direct_cast else 1

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def ep_combine(arena: Int64, addr_tok_map: Int64, addr_comb_bar: Int64,
                   addr_xdb_flag: Int64, addr_total_recv: Int64, addr_out: Int64,
                   addr_out_wts: Int64, my_lsa_rank: Int32, cur_rank_num_token: Int32):
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

        rsrc_owts = create_buffer_resource_from_addr(addr_out_wts)

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
            expert_pes = []
            expert_tks = []
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
                expert_pes.append(safe_pe)
                expert_tks.append(safe_tok_k)

            # Weights (mori UseWeights): once per token (part 0), reduce the K
            # forwarded weight vectors -> out_weights[tok][e]. Reuses the decode
            # above and overlaps with this warp's hidden gather.
            if const_expr(enable_weights):
                if part_id == arith.constant(0):
                    if lane < experts_per_token:
                        wt_acc = arith.constant(0.0, type=T.f32())
                        for k_slot in range_constexpr(experts_per_token):
                            waddr = (fx.Int64(w.lsa_ptr(expert_pes[k_slot], off_out_wts))
                                     + (fx.Int64(expert_tks[k_slot]) * fx.Int64(experts_per_token)
                                        + fx.Int64(lane)) * fx.Int64(4))
                            wv = buffer_load(create_buffer_resource_from_addr(waddr), 0,
                                             vec_width=1, dtype=T.f32())
                            wt_acc = wt_acc + arith.select(
                                expert_vlds[k_slot], wv, arith.constant(0.0, type=T.f32()))
                        buffer_store(wt_acc, rsrc_owts, tm_base + lane)
            rem = arith.constant(n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)   # i32 units this warp
            out_base = tok_id * out_n_i32
            # Nested fn: closure over expert_rsrcs/vlds (lists can't be loop-carried).
            def _one(off):           # reduce k contributions for one i32 unit
                acc = _zero_accum()
                for k_slot in range_constexpr(experts_per_token):
                    v = buffer_load(expert_rsrcs[k_slot], off, vec_width=1, dtype=T.i32(),
                                    cache_modifier=_s3_cache)
                    v = arith.select(expert_vlds[k_slot], v, arith.constant(0))
                    acc = acc + _to_accum2(v)
                buffer_store(_from_accum2(acc), rsrc_out, out_base + off * out_step_mult)

            def _accum_loop():
                # main: _unroll independent elements per lane per iter
                main_end = (eff // STEP) * STEP
                for u in range(lane, main_end, STEP):
                    accs = []
                    base = unit_base + u
                    for r in range_constexpr(_unroll):
                        accs.append(_zero_accum())
                    for k_slot in range_constexpr(experts_per_token):
                        vld = expert_vlds[k_slot]
                        rsc = expert_rsrcs[k_slot]
                        for r in range_constexpr(_unroll):
                            v = buffer_load(rsc, base + r * 64, vec_width=1, dtype=T.i32(),
                                            cache_modifier=_s3_cache)
                            accs[r] = accs[r] + _to_accum2(arith.select(vld, v, arith.constant(0)))
                    for r in range_constexpr(_unroll):
                        buffer_store(_from_accum2(accs[r]), rsrc_out,
                                     out_base + (base + r * 64) * out_step_mult)
                # tail: leftover elements, one per lane
                for u in range(main_end + lane, eff, 64):
                    _one(unit_base + u)
            _accum_loop()

    @flyc.jit
    def run(arena: Int64, addr_tok_map: Int64, addr_comb_bar: Int64, addr_xdb_flag: Int64,
            addr_total_recv: Int64, addr_out: Int64, addr_out_wts: Int64, my_lsa_rank: Int32,
            cur_rank_num_token: Int32, stream=fx.Stream(None)):
        ep_combine(arena, addr_tok_map, addr_comb_bar, addr_xdb_flag, addr_total_recv,
                   addr_out, addr_out_wts, my_lsa_rank, cur_rank_num_token).launch(
            grid=(block_num, 1, 1), block=[warp_num_per_block * 64, 1, 1], stream=stream)

    return run


def make_combine_scatter(*, rank, npes, experts_per_token, hidden_dim, hidden_elem_size,
                         max_tok_per_rank, max_recv, block_num, warp_num_per_block,
                         off_out_tok, off_comb_inp, off_tis, off_xdb_mem, off_out_wts=0,
                         off_comb_wts=0, enable_weights=True, fp8_direct_cast=False,
                         reset_total_recv=True, _s3_cache=2):
    """Scatter combine (mori useExternalInpBuffer / _nop2p path).

    Stage 1  each computing rank P2P-WRITES its post-expert tokens back to the
             ORIGIN rank's comb_inp[computing_rank*M + origin_lid] (origin from
             tis); under fp8_direct_cast the bf16 token is cast to fp8 on write.
    Stage 2  cross-device barrier.
    Stage 3  origin rank LOCAL-reads comb_inp[dest_pe*M + tok] for its token's k
             expert PEs (from tok_map) and reduces (fp8->bf16 dequant if cast).

    vs the gather path: 2 passes (remote write + local read) but compresses the
    transport to fp8; the natural home for fp8_direct_cast (gather has no
    Stage-1 writer to compress at)."""
    wire_esz = 1 if fp8_direct_cast else hidden_elem_size
    to_acc, from_acc, zero_acc = _accum_funcs(wire_esz, fp8_direct_cast)
    M = max_tok_per_rank
    inp_nbytes = hidden_dim * hidden_elem_size      # source out_tok (bf16/f32)
    wire_nbytes = hidden_dim * wire_esz             # comb_inp transport
    src_n_i32 = inp_nbytes // 4
    wire_n_i32 = wire_nbytes // 4
    out_n_i32 = (hidden_dim * 2) // 4 if fp8_direct_cast else wire_n_i32
    out_step_mult = 2 if fp8_direct_cast else 1

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def ep_combine_s(arena: Int64, addr_tok_map: Int64, addr_comb_bar: Int64,
                     addr_xdb_flag: Int64, addr_total_recv: Int64, addr_out: Int64,
                     addr_out_wts: Int64, my_lsa_rank: Int32, cur_rank_num_token: Int32):
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
        _r_tis = create_buffer_resource_from_addr(fx.Int64(w.lsa_ptr(my_lsa_rank, off_tis)))
        rsrc_out = create_buffer_resource_from_addr(addr_out)
        rsrc_owts = create_buffer_resource_from_addr(addr_out_wts)
        xdb_cur_flag = P.load_i64_acquire(fx.Int64(addr_xdb_flag))
        total_recv = buffer_load(_r_trecv, 0, vec_width=1, dtype=T.i32())

        # ── Stage 1: scatter post-expert tokens back to origin's comb_inp ──
        src_tok_base = fx.Int64(w.lsa_ptr(my_lsa_rank, off_out_tok))
        for rt in range(global_warp_id, total_recv, global_warp_num):
            enc = buffer_load(_r_tis, rt, vec_width=1, dtype=T.i32())   # origin = src_pe*M+lid
            origin_pe = enc // M
            origin_lid = enc % M
            dst = (fx.Int64(w.lsa_ptr(origin_pe, off_comb_inp))
                   + (fx.Int64(rank * M + origin_lid)) * fx.Int64(wire_nbytes))
            src = src_tok_base + fx.Int64(rt) * fx.Int64(inp_nbytes)
            rsrc_s = create_buffer_resource_from_addr(src)
            rsrc_d = create_buffer_resource_from_addr(dst)
            if const_expr(fp8_direct_cast):
                # 2 bf16 i32 -> v4f32 -> cvt_pk_fp8 x2 -> 1 fp8 i32
                for e in range(lane, wire_n_i32, 64):
                    bf = buffer_load(rsrc_s, e * 2, vec_width=2, dtype=T.i32())
                    v4 = vector.bitcast(T.VectorType.get([4], T.bf16()), bf).extf(_V4F32())
                    f0 = vector.extract(v4, static_position=[0])
                    f1 = vector.extract(v4, static_position=[1])
                    f2 = vector.extract(v4, static_position=[2])
                    f3 = vector.extract(v4, static_position=[3])
                    z = arith.constant(0, type=T.i32())
                    lo = cvt_pk_fp8_f32(res=T.i32(), src_a=f0, src_b=f1, old=z, word_sel=False)
                    fp8 = cvt_pk_fp8_f32(res=T.i32(), src_a=f2, src_b=f3, old=lo, word_sel=True)
                    buffer_store(fp8, rsrc_d, e)
            else:
                for e in range(lane, wire_n_i32, 64):
                    v = buffer_load(rsrc_s, e, vec_width=1, dtype=T.i32())
                    buffer_store(v, rsrc_d, e)
            if const_expr(enable_weights):
                # forward this recv slot's weights (dispatch put them in out_wts[rt])
                # to the ORIGIN's comb_wts[computing_rank*M + lid] (dedicated
                # region; reusing out_wts would collide with dispatch's layout).
                wsrc = (fx.Int64(w.lsa_ptr(my_lsa_rank, off_out_wts))
                        + fx.Int64(rt) * fx.Int64(experts_per_token) * fx.Int64(4))
                wdst = (fx.Int64(w.lsa_ptr(origin_pe, off_comb_wts))
                        + fx.Int64(rank * M + origin_lid) * fx.Int64(experts_per_token) * fx.Int64(4))
                if lane < experts_per_token:
                    wv = buffer_load(create_buffer_resource_from_addr(wsrc), lane,
                                     vec_width=1, dtype=T.i32())
                    buffer_store(wv, create_buffer_resource_from_addr(wdst), lane)

        # ── Stage 2: cross-device barrier ──
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
            xdb_slot = fx.Int64(w.lsa_ptr(my_lsa_rank, off_xdb_mem)) + fx.Int64(tid) * fx.Int64(8)
            P.spin_until_eq_i64(xdb_slot, xdb_cur_flag)
            P.fence_system_acquire()
        fx.barrier()
        P.fence_system_acquire()
        if const_expr(reset_total_recv):
            if tid == 0:
                buffer_store(arith.constant(0), _r_trecv, 0)

        # ── Stage 3: local read of comb_inp + reduce ──
        comb_inp_base = fx.Int64(w.lsa_ptr(my_lsa_rank, off_comb_inp))
        safe_tok = arith.select(cur_rank_num_token == arith.constant(0),
                                arith.constant(1), cur_rank_num_token)
        warps_per_tok = (arith.constant(global_warp_num) + safe_tok - arith.constant(1)) // safe_tok
        units_per_warp = (arith.constant(wire_n_i32) + warps_per_tok - arith.constant(1)) // warps_per_tok
        s3_total = cur_rank_num_token * warps_per_tok
        for s3_idx in range(global_warp_id, s3_total, global_warp_num):
            tok_id = s3_idx // warps_per_tok
            part_id = s3_idx % warps_per_tok
            unit_base = part_id * units_per_warp
            tm_base = tok_id * experts_per_token
            ex_rsrcs = []
            ex_vlds = []
            ex_pes = []
            for k_slot in range_constexpr(experts_per_token):
                enc_k = buffer_load(_r_tok_map, tm_base + k_slot, vec_width=1, dtype=T.i32())
                dpe = enc_k // max_recv
                vld = dpe < npes
                spe = arith.select(vld, dpe, arith.constant(rank))
                # LOCAL comb_inp[computing_pe*M + tok_id]
                saddr = comb_inp_base + (fx.Int64(spe) * fx.Int64(M) + fx.Int64(tok_id)) * fx.Int64(wire_nbytes)
                ex_rsrcs.append(create_buffer_resource_from_addr(saddr))
                ex_vlds.append(vld)
                ex_pes.append(spe)
            if const_expr(enable_weights):
                if part_id == arith.constant(0):
                    if lane < experts_per_token:
                        wt_acc = arith.constant(0.0, type=T.f32())
                        for k_slot in range_constexpr(experts_per_token):
                            # LOCAL comb_wts[computing_pe*M + tok_id] (scattered in S1)
                            waddr = (fx.Int64(w.lsa_ptr(my_lsa_rank, off_comb_wts))
                                     + (fx.Int64(ex_pes[k_slot]) * fx.Int64(M) + fx.Int64(tok_id))
                                     * fx.Int64(experts_per_token) * fx.Int64(4)
                                     + fx.Int64(lane) * fx.Int64(4))
                            wv = buffer_load(create_buffer_resource_from_addr(waddr), 0,
                                             vec_width=1, dtype=T.f32())
                            wt_acc = wt_acc + arith.select(ex_vlds[k_slot], wv,
                                                           arith.constant(0.0, type=T.f32()))
                        buffer_store(wt_acc, rsrc_owts, tm_base + lane)
            rem = arith.constant(wire_n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)
            out_base = tok_id * out_n_i32

            def _one(off):
                acc = zero_acc()
                for k_slot in range_constexpr(experts_per_token):
                    v = buffer_load(ex_rsrcs[k_slot], off, vec_width=1, dtype=T.i32(),
                                    cache_modifier=_s3_cache)
                    v = arith.select(ex_vlds[k_slot], v, arith.constant(0))
                    acc = acc + to_acc(v)
                buffer_store(from_acc(acc), rsrc_out, out_base + off * out_step_mult)

            def _loop():
                for u in range(lane, eff, 64):
                    _one(unit_base + u)
            _loop()

    @flyc.jit
    def run(arena: Int64, addr_tok_map: Int64, addr_comb_bar: Int64, addr_xdb_flag: Int64,
            addr_total_recv: Int64, addr_out: Int64, addr_out_wts: Int64, my_lsa_rank: Int32,
            cur_rank_num_token: Int32, stream=fx.Stream(None)):
        ep_combine_s(arena, addr_tok_map, addr_comb_bar, addr_xdb_flag, addr_total_recv,
                     addr_out, addr_out_wts, my_lsa_rank, cur_rank_num_token).launch(
            grid=(block_num, 1, 1), block=[warp_num_per_block * 64, 1, 1], stream=stream)

    return run
