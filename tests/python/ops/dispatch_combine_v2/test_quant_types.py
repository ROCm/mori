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
"""Combine quant type names: resolution per arch, refusal per backend. No GPU."""

import pytest
import torch

from mori.ops.dispatch_combine_v2 import (
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
    supported_quant_types,
)
from mori.ops.dispatch_combine_v2 import dispatch_combine_op as op_mod
from mori.ops.dispatch_combine_v2 import hip_tuning_configs as tc
from mori.ops.dispatch_combine_v2.quant_types import (
    QUANT_FORMATS,
    QUANT_TYPES,
    e8m0_round_up,
    fp8_max,
    is_gfx125x_arch,
    resolve_quant_type,
)

_RESOLVED = {
    # name: (on gfx125x, elsewhere)
    "none": ("none", "none"),
    "fp8_blockwise": ("fp8_blockwise_1x32", "fp8_blockwise_1x128"),
    "fp4_blockwise": ("fp4_blockwise_1x32", "fp4_blockwise_1x128"),
    "fp4_blockwise_fp32": ("fp4_blockwise_1x128", "fp4_blockwise_1x128"),
    **{name: (name, name) for name in QUANT_FORMATS},
}


def test_every_name_is_covered():
    assert set(_RESOLVED) == set(QUANT_TYPES)


@pytest.mark.parametrize("name", sorted(_RESOLVED))
def test_resolution(name):
    on_1250, elsewhere = _RESOLVED[name]
    assert resolve_quant_type(name, gfx125x=True) == on_1250
    assert resolve_quant_type(name, gfx125x=False) == elsewhere


def test_unknown_and_removed_names_are_refused():
    with pytest.raises(ValueError, match="quant_type must be one of"):
        resolve_quant_type("mxfp8", gfx125x=True)
    with pytest.raises(ValueError, match="was removed"):
        _cfg("fp8_direct_cast")


def _cfg(quant_type, **kw):
    # Pinned geometry: no tuning lookup, so no GPU.
    return EpDispatchCombineConfig(
        rank=0,
        world_size=8,
        hidden_dim=7168,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        quant_type=quant_type,
        dispatch_block_num=64,
        combine_block_num=64,
        warp_num_per_block=8,
        combine_warp_num_per_block=8,
        **kw,
    )


@pytest.mark.parametrize("gfx125x", [True, False])
@pytest.mark.parametrize("name", sorted(_RESOLVED))
def test_config_holds_the_explicit_name(monkeypatch, name, gfx125x):
    monkeypatch.setattr(op_mod, "is_gfx125x", lambda: gfx125x)
    cfg = _cfg(name)
    want = _RESOLVED[name][0 if gfx125x else 1]
    assert cfg.quant_type == want
    assert cfg.is_scatter == (want != "none")
    # Rebuilding from the stored name is a no-op.
    assert _cfg(cfg.quant_type).quant_type == want


def test_an_unknown_arch_refuses_only_the_arch_default_names(monkeypatch):
    for name in ("fp8_blockwise", "fp4_blockwise"):
        with pytest.raises(ValueError, match="could not be read"):
            resolve_quant_type(name, gfx125x=None)
    for name in ("none", "fp4_blockwise_fp32", *QUANT_FORMATS):
        assert resolve_quant_type(name, gfx125x=None) == _RESOLVED[name][1]
    monkeypatch.setattr(op_mod, "is_gfx125x", lambda: None)
    with pytest.raises(ValueError, match="could not be read"):
        _cfg("fp8_blockwise")
    assert _cfg("fp8_blockwise_1x32").quant_type == "fp8_blockwise_1x32"


@pytest.mark.parametrize(
    "arch, want",
    [
        ("gfx1250", True),
        ("gfx1251", True),
        ("gfx1250:sramecc+:xnack-", True),
        ("gfx950", False),
        ("gfx942:sramecc+:xnack-", False),
        ("", False),
    ],
)
def test_one_gfx125x_check(arch, want):
    assert is_gfx125x_arch(arch) is want


@pytest.mark.parametrize(
    "name, element, group, scales, scale_bytes",
    [
        ("none", None, 0, 0, 0),
        ("fp8_blockwise_1x32", "fp8", 32, 7168 // 32, 7168 // 32),
        ("fp8_blockwise_1x128", "fp8", 128, 7168 // 128, 7168 // 128 * 4),
        ("fp4_blockwise_1x32", "fp4", 32, 0, 0),
        ("fp4_blockwise_1x128", "fp4", 128, 0, 0),
    ],
)
def test_format_properties(name, element, group, scales, scale_bytes):
    cfg = _cfg(name)
    assert (cfg.quant_element, cfg.quant_group) == (element, group)
    assert cfg.fp8_blockwise == (element == "fp8")
    assert cfg.combine_scale_dim == scales
    assert cfg.combine_scale_bytes == scale_bytes


@pytest.mark.parametrize(
    "arch, fmax",
    [("gfx942", 240.0), ("gfx950", 448.0), ("gfx1250", 448.0), ("", 240.0)],
)
def test_fp8_max(arch, fmax):
    assert fp8_max(arch) == fmax


@pytest.mark.parametrize("fmax", [240.0, 448.0])
def test_e8m0_round_up_is_the_smallest_power_of_two(fmax):
    # Every positive finite bf16, the values a group amax can take.
    bits = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16)
    amax = bits.view(torch.bfloat16).double()
    top = torch.full_like(amax, fmax)
    # Smallest k with amax <= fmax * 2^k, exact in f64 after a one-step fix-up.
    k = torch.ceil(torch.log2(amax / fmax))
    k = torch.where(amax > torch.ldexp(top, k), k + 1, k)
    k = torch.where(amax <= torch.ldexp(top, k - 1), k - 1, k)
    want = (k + 127).clamp(1, 254).long()
    got = torch.tensor([e8m0_round_up(a, fmax) for a in amax.tolist()])
    assert torch.equal(got, want)
    # amax / scale never exceeds fmax, so the quantized group needs no clamp.
    assert bool((amax <= torch.ldexp(top, got.double() - 127)).all())
    assert e8m0_round_up(0.0, fmax) == 1  # an all-zero group


_SERVED = {
    ("hip", "gfx1250"): {
        "fp8_blockwise_1x32",
        "fp4_blockwise_1x32",
        "fp4_blockwise_1x128",
    },
    ("hip", "gfx942"): {"fp8_blockwise_1x32", "fp8_blockwise_1x128"},
    ("hip", "gfx950"): {
        "fp8_blockwise_1x32",
        "fp8_blockwise_1x128",
        "fp4_blockwise_1x32",
        "fp4_blockwise_1x128",
    },
    ("flydsl", "gfx1250"): set(),
    ("flydsl", "gfx942"): {"fp8_blockwise_1x32", "fp8_blockwise_1x128"},
    ("flydsl", "gfx950"): {"fp8_blockwise_1x32", "fp8_blockwise_1x128"},
}


@pytest.mark.parametrize("backend, arch", sorted(_SERVED))
def test_supported_quant_types_match_the_refusal(backend, arch):
    try:
        cls = EpDispatchCombineOp._resolve_backend(backend)
    except ImportError as e:
        pytest.skip(str(e))
    served = _SERVED[backend, arch]
    assert set(cls._quant_formats(arch)) == served
    names = supported_quant_types(backend, arch)
    gfx125x = arch.startswith("gfx125")
    assert names == {
        n: resolve_quant_type(n, gfx125x)
        for n in QUANT_TYPES
        if resolve_quant_type(n, gfx125x) in served | {"none"}
    }
    for name in QUANT_TYPES:
        cfg = _cfg(resolve_quant_type(name, gfx125x))
        refused = cls._quant_refusal(cfg, arch)
        assert bool(refused) == (name not in names), name
        if refused:
            assert arch in refused[0] and cfg.quant_type in refused[0]


@pytest.fixture
def gfx1250(monkeypatch):
    monkeypatch.delenv("MORI_EP_TUNING_PROFILE", raising=False)
    monkeypatch.setattr(tc, "_device_key", lambda: "gfx1250")
    monkeypatch.setattr(tc._gpu, "asic_revision", lambda: 1)
    monkeypatch.setattr(tc._gpu, "cu_count", lambda: 0)  # 0: no CU clamp


def test_lookup_resolves_the_arch_default_name(gfx1250):
    kw = dict(dtype="fp4", experts_per_rank=None)
    for bare, explicit in (
        ("fp4_blockwise", "fp4_blockwise_1x32"),
        ("fp8_blockwise", "fp8_blockwise_1x32"),
        ("fp4_blockwise_fp32", "fp4_blockwise_1x128"),
    ):
        assert tc.lookup(4, 7168, 6, quant_type=bare, **kw) == tc.lookup(
            4, 7168, 6, quant_type=explicit, **kw
        )
    assert tc.lookup(4, 7168, 6, quant_type="fp4_blockwise", **kw) != tc.lookup(
        4, 7168, 6, quant_type="none", **kw
    )


def test_push_tables_use_explicit_names():
    for table in (tc._COMBINE_PUSH_TABLE, tc._LL128_TABLE):
        assert set(table) - {"none"} <= set(QUANT_FORMATS)


@pytest.mark.parametrize("model", ["mi300x", "mi355x"])
@pytest.mark.parametrize("name", [*sorted(QUANT_FORMATS), "none"])
@pytest.mark.parametrize("scatter", [False, True])
def test_scatter_rows_serve_every_format(monkeypatch, name, model, scatter):
    monkeypatch.delenv("MORI_EP_TUNING_PROFILE", raising=False)
    monkeypatch.setattr(tc, "_device_key", lambda: model)
    monkeypatch.setattr(tc._gpu, "cu_count", lambda: 0)  # 0: no CU clamp
    # A quantized combine takes the scatter rows whatever scatter= says (callers that
    # predate it omit it); "none" takes them only as the unquantized scatter.
    got = tc.lookup(
        8, 7168, 8, dtype="bf16", experts_per_rank=32, quant_type=name, scatter=scatter
    )
    if name != "none" or scatter:
        for edge, block, warp in tc._SCATTER_COMBINE_TABLE[model][8, 7168, 8, None]:
            assert tc.pick_bucket(got["schedule"], edge or 1 << 20)[3:5] == (block, warp)
    else:  # the gather keeps its own rows
        scat = tc.lookup(8, 7168, 8, dtype="bf16", experts_per_rank=32, scatter=True)
        assert got["schedule"] != scat["schedule"]


def test_flydsl_refuses_a_quantized_combine_that_is_not_bf16(monkeypatch):
    try:
        cls = EpDispatchCombineOp._resolve_backend("flydsl")
    except ImportError as e:
        pytest.skip(str(e))
    from mori.ops.dispatch_combine_v2 import flydsl_backend

    monkeypatch.setattr(flydsl_backend, "_target_arch", lambda: "gfx942")
    op = object.__new__(cls)
    cfg = _cfg("fp8_blockwise_1x128", data_type=torch.float32)
    assert any("bf16 only" in r for r in op._unsupported(cfg))
    assert not op._unsupported(_cfg("fp8_blockwise_1x128"))
