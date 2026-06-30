"""Per-shape block/warp tuning for the cco-LSA dispatch/combine kernels.

Analogue of mori's tuning_configs/: a small table of measured-good launch
geometries keyed by (world_size, hidden_dim, topk), with a fallback default.
Dispatch's posted remote WRITES saturate with fewer blocks; combine's remote
READS are latency-bound and want more warps/blocks — hence separate counts.

Measured on MI300X (gfx942), EP8, identity-expert round-trip.
"""

_DEFAULT = dict(dispatch_block_num=64, combine_block_num=128, warp_num_per_block=16)

# (world_size, hidden_dim, topk) -> overrides applied on top of _DEFAULT.
_TABLE = {
    (8, 7168, 8): dict(dispatch_block_num=64, combine_block_num=128, warp_num_per_block=16),
    (8, 4096, 8): dict(dispatch_block_num=64, combine_block_num=128, warp_num_per_block=16),
    (8, 2048, 8): dict(dispatch_block_num=48, combine_block_num=96, warp_num_per_block=16),
}


def lookup(world_size, hidden_dim, topk):
    """Return {dispatch_block_num, combine_block_num, warp_num_per_block}."""
    cfg = dict(_DEFAULT)
    cfg.update(_TABLE.get((world_size, hidden_dim, topk), {}))
    return cfg
