"""Local expert token count (FlyDSL) — mori ep_local_expert_count parity.

For each received token's k expert assignments (out_idx[recv, K], populated by
dispatch), count how many land on each local expert. Local, no cross-device sync.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T
from flydsl.expr.buffer_ops import buffer_load, create_buffer_resource_from_addr
from flydsl.expr.typing import Int64

import flydsl_prims as P


def make_local_expert_count(*, rank, experts_per_rank, experts_per_token,
                            block_num, warp_num_per_block):
    expert_base = rank * experts_per_rank
    bsz = warp_num_per_block * 64

    @flyc.kernel(known_block_size=[bsz, 1, 1])
    def klec(addr_out_idx: Int64, addr_total_recv: Int64, addr_count: Int64):
        gid = fx.block_idx.x * bsz + fx.thread_idx.x
        gnum = block_num * bsz
        _r_idx = create_buffer_resource_from_addr(addr_out_idx)
        _r_tr = create_buffer_resource_from_addr(addr_total_recv)
        limit = buffer_load(_r_tr, 0, vec_width=1, dtype=T.i32()) * experts_per_token
        for i in range(gid, limit, gnum):
            le = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32()) - expert_base
            if le >= 0:
                if le < experts_per_rank:
                    P.atomic_add_global(fx.Int64(addr_count) + fx.Int64(le) * fx.Int64(4),
                                        fx.Int32(1))

    @flyc.jit
    def run(addr_out_idx: Int64, addr_total_recv: Int64, addr_count: Int64,
            stream=fx.Stream(None)):
        klec(addr_out_idx, addr_total_recv, addr_count).launch(
            grid=(block_num, 1, 1), block=[bsz, 1, 1], stream=stream)

    return run
