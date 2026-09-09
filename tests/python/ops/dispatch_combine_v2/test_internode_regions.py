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
"""Invariants of the v2 internode arena layout.

``internode_regions()`` is the whole of the op's symmetric memory: SymmArena bump
-allocates exactly these regions and the kernel addresses them by name, so a
missing name is an AttributeError at build time but a *wrong size* is silent
corruption of whichever region follows.

This used to compare the sizes against what v1's ``EpDispatchCombineHandle``
allocated, via a pybind probe. Both are gone -- the v2 path no longer constructs a
v1 handle, which was the point of the refactor -- so the oracle is gone with them.
What is checked here instead are the properties that do not need a second
implementation to state: the name contract with the backend, and the capacity
bounds the kernel's own indexing arithmetic implies.

Pure function of the config: no GPU, no process group, no op.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mori.ops.dispatch_combine_v2.internode_regions import internode_regions

# Every name `EpDispatchCombineOpHip._internode_static_args` resolves, transcribed
# independently. A rename on either side has to be made twice or this fails --
# which is the only cheap guard left now that the v1 allocator is not around to
# be compared against.
_REQUIRED = {
    "inter_dispatch_inp",
    "inter_combine_inp",
    "inter_staging",
    "inter_dispatch_out",
    "inter_combine_out",
    "inter_dispatch_staging",
    "inp_weights",
    "dispatch_out_weights",
    "combine_out_weights",
    "out_indices",
    "recv_token_num",
    "node_recv_token_num",
    "disp_tok_offset",
    "disp_tok_id_to_src_tok_id",
    "cross_device_barrier",
    "inter_node_chunk_flag",
}
_OPTIONAL = {"out_scales"}


def _cfg(**over):
    base = dict(
        world_size=16,
        gpu_per_node=8,
        hidden_dim=7168,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        max_token_type_size=2,
        scale_dim=32,
        scale_type_size=4,
        num_qp_per_pe=2,
        max_total_recv_tokens=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# Chosen to exercise the branches the formulas actually have: scales on/off, one
# node vs many, and the max_total_recv_tokens clamp.
_CASES = {
    "base": _cfg(),
    "single_node": _cfg(world_size=8, gpu_per_node=8),
    "two_nodes_no_scale": _cfg(
        world_size=8, gpu_per_node=4, scale_dim=0, scale_type_size=0, num_qp_per_pe=1
    ),
    "recv_cap_clamped": _cfg(max_total_recv_tokens=256, num_experts_per_token=4),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_region_names_match_the_backend_contract(name):
    cfg = _CASES[name]
    got = {n for n, _ in internode_regions(cfg)}
    assert _REQUIRED <= got, f"missing: {sorted(_REQUIRED - got)}"
    assert (
        got <= _REQUIRED | _OPTIONAL
    ), f"unexpected: {sorted(got - _REQUIRED - _OPTIONAL)}"


@pytest.mark.parametrize("name", sorted(_CASES))
def test_no_duplicate_or_empty_regions(name):
    regions = internode_regions(_CASES[name])
    names = [n for n, _ in regions]
    assert len(names) == len(set(names)), "SymmArena would silently keep only the last"
    for n, sz in regions:
        assert sz > 0, f"{n} is zero-sized; the arena hands out an aliasing pointer"


@pytest.mark.parametrize("name", sorted(_CASES))
def test_scales_absent_rather_than_zero_sized(name):
    cfg = _CASES[name]
    present = "out_scales" in dict(internode_regions(cfg))
    assert present == bool(cfg.scale_dim * cfg.scale_type_size)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_staging_holds_every_slot_the_kernel_indexes(name):
    """The combine half of ``inter_staging`` starts at slot ``nNodes * m`` and
    runs to ``2 * nNodes * m`` (``SendBufSlotOffset(cfg, nNodes, 0)`` onward), at
    a stride of ``combXferBytes = hidden + weights``. Dispatch uses the same
    region at the larger ``xferBytes`` stride, so the region has to cover the
    worse of the two."""
    cfg = _CASES[name]
    sizes = dict(internode_regions(cfg))
    n_nodes = cfg.world_size // cfg.gpu_per_node
    slots = 2 * n_nodes * cfg.max_num_inp_token_per_rank
    comb_xfer = cfg.hidden_dim * cfg.max_token_type_size + cfg.num_experts_per_token * 4
    assert sizes["inter_staging"] >= slots * comb_xfer


def test_gpu_per_node_must_divide_the_world():
    with pytest.raises(ValueError):
        internode_regions(_cfg(world_size=16, gpu_per_node=6))
