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
"""The config's tuned geometry must belong to the backend that ends up serving it.

``__post_init__`` resolves block/warp/schedule before any op exists, so it has to
guess the backend from ``cfg.kernel_backend`` / ``MORI_V2_KERNEL_BACKEND`` / the
default. Naming a backend subclass directly bypasses all three, and the op then
runs one backend's kernels on the other's schedule -- correct results, wrong
geometry, and nothing anywhere says so. ``retune_for`` closes that, and the op
calls it from ``__new__``.

No communicator and no peers: this is config arithmetic.

    pytest test_config_geometry.py -v
"""

import pytest
import torch

from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig

hip_tuning = pytest.importorskip("mori.ops.dispatch_combine_v2.hip_tuning_configs")
flydsl_tuning = pytest.importorskip("mori.ops.dispatch_combine_v2.tuning_configs")

SHAPE = dict(
    rank=0,
    world_size=8,
    hidden_dim=7168,
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,
    num_experts_per_token=8,
    data_type=torch.bfloat16,
)


def _geom(cfg):
    return (
        cfg.dispatch_block_num,
        cfg.warp_num_per_block,
        cfg.combine_block_num,
        cfg.combine_warp_num_per_block,
        cfg.schedule,
    )


def _table(mod, cfg):
    kw = dict(dtype=cfg.dtype_str)
    if mod is hip_tuning:
        kw["experts_per_rank"] = cfg.num_experts_per_rank
    t = mod.lookup(cfg.world_size, cfg.hidden_dim, cfg.num_experts_per_token, **kw)
    return (
        t["dispatch_block_num"],
        t["warp_num_per_block"],
        t["combine_block_num"],
        t["combine_warp_num_per_block"],
        t["schedule"],
    )


def test_named_backend_gets_its_own_table():
    cfg = EpDispatchCombineConfig(**SHAPE, kernel_backend="hip")
    assert _geom(cfg) == _table(hip_tuning, cfg)

    cfg = EpDispatchCombineConfig(**SHAPE, kernel_backend="flydsl")
    assert _geom(cfg) == _table(flydsl_tuning, cfg)


def test_retune_switches_the_table():
    """The case the op hits when a subclass is named directly: the config was
    tuned for the default backend and has to be moved to the real one."""
    cfg = EpDispatchCombineConfig(**SHAPE)  # no backend named -> flydsl's table
    assert _geom(cfg) == _table(flydsl_tuning, cfg)

    cfg.retune_for("hip")
    assert cfg.kernel_backend == "hip"
    assert _geom(cfg) == _table(hip_tuning, cfg)


def test_retune_is_idempotent_and_a_no_op_for_the_same_backend():
    cfg = EpDispatchCombineConfig(**SHAPE, kernel_backend="hip")
    before = _geom(cfg)
    cfg.retune_for("hip")
    cfg.retune_for("hip")
    assert _geom(cfg) == before


def test_a_pinned_geometry_outranks_every_table():
    """An explicit block/warp is the caller overriding the tuner. Re-tuning for
    another backend must not quietly take it back."""
    cfg = EpDispatchCombineConfig(**SHAPE, dispatch_block_num=13)
    assert cfg.dispatch_block_num == 13
    assert cfg.schedule is None  # pinning opts out of the schedule entirely

    cfg.retune_for("hip")
    assert cfg.kernel_backend == "hip"
    assert cfg.dispatch_block_num == 13
    assert cfg.schedule is None


def test_a_pinned_schedule_survives_retuning():
    sched = ((128, 32, 4, 32, 4), (None, 64, 8, 64, 8))
    cfg = EpDispatchCombineConfig(**SHAPE, schedule=sched)
    cfg.retune_for("hip")
    assert cfg.schedule == sched


def test_the_op_retunes_before_it_is_initialised():
    """The wiring, not just the method: __new__ must call retune_for, and it must
    do so for a directly-named subclass too -- that is the whole failure case.
    __new__ alone, so no communicator and no arena are needed."""
    hip_backend = pytest.importorskip(
        "mori.ops.dispatch_combine_v2.hip_backend",
        reason="libmori_ops_v2.so not built",
    )
    cls = hip_backend.EpDispatchCombineOpHip

    cfg = EpDispatchCombineConfig(**SHAPE)  # no backend named
    assert cfg.kernel_backend is None
    cls.__new__(cls, cfg, None)
    assert cfg.kernel_backend == "hip"
    assert _geom(cfg) == _table(hip_tuning, cfg)
