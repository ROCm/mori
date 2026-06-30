"""StdMoE convert kernels (FlyDSL) — local re-layout around the expert GEMM.

Port of mori intranode ConvertDispatchOutput / ConvertCombineInput (the
ENABLE_STANDARD_MOE_ADAPT path). Both are LOCAL (operate only on this rank's
received buffers — no cross-device sync), so they are far simpler than
dispatch/combine: plain grid-stride loops + atomics, no barriers/spin-waits.

Flow (per rank):
  dispatch  -> out_tok[recv], out_idx[recv,K], out_wts[recv,K], tis[recv], total_recv
  convert_dispatch_output -> packed_x[localExpert, slot, hidden]   (per-expert packed)
                             packed_cnt[localExpert], packed_src[slot], slot_map[recv,K]
  <expert GEMM overwrites packed_x[slot] in place with f_expert(token)>
  convert_combine_input   -> out_tok[recv] = sum_{k: expert_k local} wts[k]*packed_x[slot_k]
  combine (P2P-read gather) sums the per-rank partials across dest PEs

So the per-(token,expert) weighting happens here (locally, over local experts);
the cross-rank combine is a plain unweighted sum of the partials. End to end:
  comb_out[s] = sum over all k of wts[s][k] * f_{expert_k}(input[s]).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, T, vector
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from flydsl.expr.rocdl import readlane
from flydsl.expr.typing import Int32, Int64

import flydsl_prims as P

_V2BF16 = lambda: T.VectorType.get([2], T.bf16())
_V2F32 = lambda: T.VectorType.get([2], T.f32())
_V1I32 = lambda: T.VectorType.get([1], T.i32())


def _to_accum2(i32_scalar):        # i32 (2 bf16) -> v2f32
    return vector.bitcast(_V2BF16(), vector.from_elements(_V1I32(), [i32_scalar])).extf(_V2F32())


def _from_accum2(v2f32):           # v2f32 -> i32 (2 bf16)
    return vector.extract(vector.bitcast(_V1I32(), v2f32.truncf(_V2BF16())), static_position=[0])


def _splat2(f32_scalar):           # f32 -> v2f32 (broadcast)
    return vector.from_elements(_V2F32(), [f32_scalar, f32_scalar])


def make_convert_dispatch_output(*, rank, experts_per_rank, experts_per_token,
                                 hidden_dim, hidden_elem_size, max_tok_per_expert,
                                 block_num, warp_num_per_block):
    """Per-expert packing of dispatched tokens. One warp per (recv_tok, k_slot):
    lane0 bumps packed_cnt[localExpert] to claim a slot, then the warp copies the
    token embedding into packed_x[slot]."""
    assert hidden_elem_size == 2, "stdmoe convert is bf16-only"
    nbytes = hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def convert_disp(addr_out_tok: Int64, addr_out_idx: Int64, addr_tis: Int64,
                     addr_total_recv: Int64, addr_packed_x: Int64, addr_packed_cnt: Int64,
                     addr_packed_src: Int64, addr_slot_map: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gwid = bid * warp_num_per_block + warp
        gwarps = block_num * warp_num_per_block

        _r_out_idx = create_buffer_resource_from_addr(addr_out_idx)
        _r_tis = create_buffer_resource_from_addr(addr_tis)
        _r_trecv = create_buffer_resource_from_addr(addr_total_recv)
        _r_psrc = create_buffer_resource_from_addr(addr_packed_src)
        _r_smap = create_buffer_resource_from_addr(addr_slot_map)

        trecv = buffer_load(_r_trecv, 0, vec_width=1, dtype=T.i32())
        work = trecv * experts_per_token
        for i in range(gwid, work, gwarps):
            recv_tok = i // experts_per_token
            expert = buffer_load(_r_out_idx, i, vec_width=1, dtype=T.i32())
            local_e = expert - arith.constant(rank * experts_per_rank)
            # MUST be unsigned ult: signed slt would treat negative local_e
            # (non-local experts) as in-range and trigger an OOB copy.
            is_local = arith.cmpi(arith.CmpIPredicate.ult, local_e,
                                  arith.constant(experts_per_rank))
            # lane0 claims a per-expert packing slot, then broadcasts.
            slot_lane0 = arith.constant(0)
            if lane == 0:
                if is_local:
                    cnt_addr = fx.Int64(addr_packed_cnt) + fx.Int64(local_e) * fx.Int64(4)
                    slot_lane0 = P.atomic_add_global(cnt_addr, fx.Int32(1))
            slot = readlane(T.i32(), slot_lane0, 0)
            safe_local = arith.select(is_local, local_e, arith.constant(0))
            lin = safe_local * arith.constant(max_tok_per_expert) + slot
            slot_val = arith.select(is_local, fx.Int64(lin), arith.constant(-1, type=T.i64()))
            if lane == 0:
                P.store_i64_system(fx.Int64(addr_slot_map) + fx.Int64(i) * fx.Int64(8),
                                   arith.constant(0), slot_val)
                if is_local:
                    src = buffer_load(_r_tis, recv_tok, vec_width=1, dtype=T.i32())
                    buffer_store(src, _r_psrc, lin)
            if is_local:
                dst = fx.Int64(addr_packed_x) + fx.Int64(lin) * fx.Int64(nbytes)
                src_t = fx.Int64(addr_out_tok) + fx.Int64(recv_tok) * fx.Int64(nbytes)
                rsrc_d = create_buffer_resource_from_addr(dst)
                rsrc_s = create_buffer_resource_from_addr(src_t)
                for c in range(lane * 4, n_i32, 256):
                    v = buffer_load(rsrc_s, c, vec_width=4, dtype=T.i32())
                    buffer_store(v, rsrc_d, c)

    @flyc.jit
    def run(addr_out_tok: Int64, addr_out_idx: Int64, addr_tis: Int64, addr_total_recv: Int64,
            addr_packed_x: Int64, addr_packed_cnt: Int64, addr_packed_src: Int64,
            addr_slot_map: Int64, stream=fx.Stream(None)):
        convert_disp(addr_out_tok, addr_out_idx, addr_tis, addr_total_recv, addr_packed_x,
                     addr_packed_cnt, addr_packed_src, addr_slot_map).launch(
            grid=(block_num, 1, 1), block=[warp_num_per_block * 64, 1, 1], stream=stream)

    return run


def make_convert_combine_input(*, rank, experts_per_rank, experts_per_token,
                               hidden_dim, hidden_elem_size, max_tok_per_expert,
                               block_num, warp_num_per_block):
    """Inverse: reduce each recv token's local-expert outputs with routing
    weights back into out_tok (the combine input staging). Warp-partitioned over
    hidden; out_tok[recv_t][e] = sum_{k: slot_k valid} wts[k]*packed_x[slot_k][e]."""
    assert hidden_elem_size == 2, "stdmoe convert is bf16-only"
    nbytes = hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def convert_comb(addr_out_tok: Int64, addr_out_wts: Int64, addr_total_recv: Int64,
                     addr_packed_x: Int64, addr_slot_map: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gwid = bid * warp_num_per_block + warp
        gwarps = block_num * warp_num_per_block

        _r_owts = create_buffer_resource_from_addr(addr_out_wts)
        _r_smap = create_buffer_resource_from_addr(addr_slot_map)
        _r_trecv = create_buffer_resource_from_addr(addr_total_recv)

        trecv = buffer_load(_r_trecv, 0, vec_width=1, dtype=T.i32())
        safe_recv = arith.select(trecv == arith.constant(0), arith.constant(1), trecv)
        warps_per_tok = (arith.constant(gwarps) + safe_recv - arith.constant(1)) // safe_recv
        units_per_warp = (arith.constant(n_i32) + warps_per_tok - arith.constant(1)) // warps_per_tok
        total = trecv * warps_per_tok
        for s_idx in range(gwid, total, gwarps):
            recv_t = s_idx // warps_per_tok
            part = s_idx % warps_per_tok
            unit_base = part * units_per_warp
            sm_base = recv_t * experts_per_token
            ex_rsrcs = []
            ex_vlds = []
            ex_wts = []
            for k_slot in range_constexpr(experts_per_token):
                slot = P.load_i64_acquire(fx.Int64(addr_slot_map)
                                          + fx.Int64(sm_base + k_slot) * fx.Int64(8))
                vld = slot != arith.constant(-1, type=T.i64())
                safe_slot = arith.select(vld, slot, arith.constant(0, type=T.i64()))
                xaddr = fx.Int64(addr_packed_x) + safe_slot * fx.Int64(nbytes)
                ex_rsrcs.append(create_buffer_resource_from_addr(xaddr))
                ex_vlds.append(vld)
                ex_wts.append(buffer_load(_r_owts, sm_base + k_slot, vec_width=1, dtype=T.f32()))
            rsrc_out = create_buffer_resource_from_addr(
                fx.Int64(addr_out_tok) + fx.Int64(recv_t) * fx.Int64(nbytes))
            rem = arith.constant(n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)

            def _one(off):
                acc = _to_accum2(arith.constant(0))
                for k_slot in range_constexpr(experts_per_token):
                    v = buffer_load(ex_rsrcs[k_slot], off, vec_width=1, dtype=T.i32())
                    wv = arith.select(ex_vlds[k_slot], ex_wts[k_slot],
                                      arith.constant(0.0, type=T.f32()))
                    # v2f32 vector * f32 scalar (broadcast), matching the
                    # FlyDSL reference _weighted_accum_experts.
                    acc = acc + _to_accum2(v) * wv
                buffer_store(_from_accum2(acc), rsrc_out, off)

            def _loop():
                for u in range(lane, eff, 64):
                    _one(unit_base + u)
            _loop()

    @flyc.jit
    def run(addr_out_tok: Int64, addr_out_wts: Int64, addr_total_recv: Int64,
            addr_packed_x: Int64, addr_slot_map: Int64, stream=fx.Stream(None)):
        convert_comb(addr_out_tok, addr_out_wts, addr_total_recv, addr_packed_x,
                     addr_slot_map).launch(
            grid=(block_num, 1, 1), block=[warp_num_per_block * 64, 1, 1], stream=stream)

    return run
