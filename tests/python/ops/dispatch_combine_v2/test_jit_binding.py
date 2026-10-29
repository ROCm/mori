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
"""The ctypes boundary itself, not the kernels behind it.

Every test here pins a property of the binding: that the Plan class is *derived*
from what C++ publishes rather than written down twice, that a typo'd knob is an
error instead of a silent default, and that a config which cannot run is rejected
before anything is compiled.

Single rank, no communicator: creating a Plan compiles and loads a module but
launches nothing, so `world_size=1` needs no arena and no peers. It does need
hipcc and a GPU to load onto.
"""

import ctypes

import pytest

torch = pytest.importorskip("torch")

try:
    from mori.ops.dispatch_combine_v2 import ep_plans as cb
except OSError as e:  # library not built
    pytest.skip(f"libmori_ops_v2.so not available: {e}", allow_module_level=True)


def _plan(**kw):
    """A minimal valid EpDispatchPlan. world_size=1 is the degenerate case of the
    peer path, not a separate one, so it renders the same kernel EP8 does."""
    args = dict(
        world_size=1,
        hidden_dim=1024,
        max_tok_per_rank=64,
        num_expert_per_rank=4,
        num_expert_per_token=4,
        max_recv=64,
    )
    args.update(kw)
    return cb.EpDispatchPlan(**args)


# --------------------------------------------------------------------------
# The Plan class is generated, not written
# --------------------------------------------------------------------------


def test_registry_lists_both_ep_kernels():
    plans = cb.registered_plans()
    assert "ep_dispatch" in plans and "ep_combine" in plans


def test_plan_classes_materialise_from_the_registry():
    # Nothing in the binding names a kernel: the class exists because C++
    # registered one.
    assert cb.EpDispatchPlan.__name__ == "EpDispatchPlan"
    assert cb.EpCombinePlan.__name__ == "EpCombinePlan"
    with pytest.raises(AttributeError):
        cb.NoSuchKernelPlan


def test_constructor_signature_comes_from_the_cpp_request_schema():
    doc = cb.EpDispatchPlan.__doc__
    for field in (
        "world_size",
        "hidden_dim",
        "num_expert_per_token",
        "max_recv",
        "dtype",
    ):
        assert field in doc, f"{field} missing from the generated signature"


def test_args_struct_is_built_from_the_cpp_schema_and_size_checked():
    # The struct is derived from the published schema and its size asserted
    # against C++'s own sizeof; a hand-written mirror is the layout bug this
    # replaced. Building the class is what runs that assertion.
    cls = cb.make_plan("ep_dispatch")
    assert ctypes.sizeof(cls._args_t) > 0
    for name in ("window", "rank", "offTokOff", "offOutTok", "numTokens"):
        assert name in cls._arg_names


# --------------------------------------------------------------------------
# Errors surface where the mistake was made
# --------------------------------------------------------------------------


def test_unknown_request_field_is_rejected_not_ignored():
    # A typo'd knob that silently does nothing is how a measurement ends up
    # describing a different binary than the one it names.
    with pytest.raises(TypeError, match="unknown argument"):
        _plan(hiden_dim=2048)


def test_missing_request_field_takes_the_cpp_default():
    p = _plan()
    assert p.info["useWeights"] is True  # EpRequest's default, never named here
    p.close()


def test_unknown_kernel_is_an_error():
    with pytest.raises(RuntimeError):
        cb.make_plan("not_a_kernel")


def test_unknown_launch_arg_is_rejected():
    p = _plan()
    with pytest.raises(TypeError, match="unknown launch argument"):
        p.launch(nonsense=1)
    p.close()


def test_bad_dtype_is_rejected_before_reaching_cpp():
    with pytest.raises(ValueError, match="unsupported dtype"):
        _plan(dtype=torch.int8)


def test_invalid_shape_reports_a_useful_error():
    # hidden_dim=7 -> token bytes not 16 B aligned; EpCfgIsValid rejects it in
    # C++ and the message names what to change.
    with pytest.raises(RuntimeError, match="16 B aligned|inconsistent config"):
        _plan(hidden_dim=7)


def test_closed_plan_refuses_to_launch():
    p = _plan()
    p.close()
    with pytest.raises(RuntimeError, match="closed plan"):
        p.launch()


# --------------------------------------------------------------------------
# info reports what C++ decided, never what Python asked
# --------------------------------------------------------------------------


def test_info_reports_every_cfg_field_and_the_resolved_geometry():
    p = _plan()
    info = p.info
    for field in (
        "worldSize",
        "hiddenDim",
        "maxRecv",
        "waveSize",
        "blockNum",
        "warpPerBlock",
    ):
        assert field in info, f"{field} missing from info"
    assert info["blockX"] == info["warpPerBlock"] * info["waveSize"]
    assert info["gridX"] == info["blockNum"]
    p.close()


def test_geometry_defaults_come_from_cpp_not_from_the_caller():
    p = _plan()  # no block_num / warp_per_block given
    assert p.info["blockNum"] > 0 and p.info["warpPerBlock"] > 0
    p.close()


def test_dtype_accepts_a_name_or_a_torch_dtype():
    a, b = _plan(dtype="bf16"), _plan(dtype=torch.bfloat16)
    assert a.info["dtype"] == b.info["dtype"]
    a.close()
    b.close()


# --------------------------------------------------------------------------
# The cache key follows the Cfg, and only the Cfg
# --------------------------------------------------------------------------


def test_different_geometry_is_a_different_binary():
    a = _plan(block_num=32, warp_per_block=4)
    b = _plan(block_num=64, warp_per_block=4)
    assert a.info["cacheDir"] != b.info["cacheDir"]
    a.close()
    b.close()


def test_same_cfg_reuses_the_cache_entry():
    a, b = _plan(), _plan()
    assert a.info["cacheDir"] == b.info["cacheDir"]
    a.close()
    b.close()


def test_dispatch_and_combine_are_different_binaries():
    d = _plan()
    c = cb.EpCombinePlan(
        world_size=1,
        hidden_dim=1024,
        max_tok_per_rank=64,
        num_expert_per_rank=4,
        num_expert_per_token=4,
        max_recv=64,
        block_num=d.info["blockNum"],
        warp_per_block=d.info["warpPerBlock"],
    )
    # Same Cfg, same geometry: only the kernel name and body differ, and the
    # name is one of the hashed components.
    assert d.info["cacheDir"] != c.info["cacheDir"]
    d.close()
    c.close()


def test_arena_offsets_are_launch_defaults_not_part_of_the_key():
    """Two arena layouts of the same shape share one binary.

    This is the design's §1.5(2) decision made concrete: the offsets are runtime
    arguments, so a different layout rebinds a scalar instead of forcing a
    recompile -- which is also why one binary serves all eight ranks.
    """

    class FakeArena:
        def __init__(self, base):
            self._base = base
            self.handle = 0xDEAD0000

        def offset(self, name):
            return self._base + 256 * (len(name) % 7)

    a = _plan(arena=FakeArena(0))
    b = _plan(arena=FakeArena(4096))
    assert a.info["cacheDir"] == b.info["cacheDir"], "offsets must not be in the key"
    assert a._defaults["offTokOff"] != b._defaults["offTokOff"], "offsets must be bound"
    a.close()
    b.close()


# --------------------------------------------------------------------------
# Sizes: a plan reports what it binds (EpRegion*, EpStagingTotalBytes), and
# describe() reports it before the plan exists, so a caller sizes nothing by hand
# --------------------------------------------------------------------------

_SMALL = dict(
    world_size=1,
    hidden_dim=1024,
    max_tok_per_rank=64,
    num_expert_per_rank=4,
    num_expert_per_token=4,
    max_recv=64,
)


def test_describe_reports_what_the_plan_would_without_building_it():
    for cls in (cb.EpDispatchPlan, cb.EpCombinePlan):
        described = cls.describe(**_SMALL)
        p = cls(**_SMALL)
        built = p.info
        p.close()
        assert described.pop("cacheDir") == ""
        built.pop("cacheDir")
        assert described == built


def test_describe_rejects_what_the_constructor_rejects():
    with pytest.raises(TypeError, match="unknown argument"):
        cb.EpDispatchPlan.describe(world_size=1, bogus=1)
    with pytest.raises(RuntimeError, match="inconsistent config"):
        cb.EpDispatchPlan.describe(hidden_dim=7)


def test_every_arena_region_is_reported_in_args_order():
    stems = [
        n[3].lower() + n[4:]
        for n in cb.EpDispatchPlan._arg_names
        if n.startswith("off") and n[3:4].isupper()
    ]
    for cls in (cb.EpDispatchPlan, cb.EpCombinePlan):
        info = cls.describe()
        assert [k[len("region.") :] for k in info if k.startswith("region.")] == stems


def _reference_regions(
    ws, hidden, max_tok, topk, disp_tok, comb_tok, scale, push, wave
):
    """hip_backend._regions as it was before the plans reported sizes.

    The one deliberate change: the push combine's row flags are only reserved
    where the overlapped reduce can run (EpCombinePushOvlAny).
    """
    cap = ws * max_tok
    out_tok = cap * comb_tok
    if push:
        q, bits, sb = {"fp4": (32, 4, 1), "fp4b": (128, 4, 4), "fp8": (32, 8, 1)}[push]
        wire = (hidden * bits // 8 + hidden // q * sb + 127) // 128 * 128
        rows = max_tok * min(ws, topk)
        out_tok += rows * wire
        if push != "fp4b" and 4 <= ws <= wave:
            out_tok += rows * 8
    regions = [
        ("tok_off", 4),
        ("recv_num", 2 * ws * 4),
        ("recv_to_src_token", cap * 4),
        ("out_idx", cap * topk * 4),
        ("out_wts", cap * topk * 4),
        ("disp_out", cap * disp_tok),
        ("out_tok", out_tok),
        ("cross_device_barrier", ws * 8),
    ]
    if scale:
        regions.append(("out_scales", cap * ((scale + 127) // 128 * 128)))
    return regions


@pytest.mark.parametrize("ws", [1, 2, 4, 8])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize(
    "dispatch,scale,push",
    [
        ("bf16", 0, None),
        ("fp8", 224, None),
        ("fp4x2", 32, None),
        ("fp32", 0, None),
        ("fp4x2", 32, "fp4"),
        ("fp4x2", 32, "fp4b"),
        ("fp8", 224, "fp8"),
    ],
)
def test_arena_regions_match_the_layout_the_op_used_to_build(
    ws, topk, dispatch, scale, push
):
    from mori.ops.dispatch_combine_v2.hip_backend import _REGIONS

    hidden, max_tok = 7168, 128
    common = dict(
        world_size=ws,
        max_tok_per_rank=max_tok,
        num_expert_per_rank=8,
        num_expert_per_token=topk,
        max_recv=ws * max_tok,
    )
    disp_hidden = hidden // 2 if dispatch == "fp4x2" else hidden
    d = cb.EpDispatchPlan.describe(
        **common, hidden_dim=disp_hidden, dtype=dispatch, scale_bytes=scale
    )
    comb = {"fp4": dict(combine_fp4=1), "fp8": dict(combine_fp8=1)}.get(push, {})
    if push == "fp4b":
        comb = dict(combine_fp4=1, combine_fp4_f32_scale=1)
    if push and not d["sharedBytes"]:
        pytest.skip("the push combine is gfx125x only")
    c = cb.EpCombinePlan.describe(**common, hidden_dim=hidden, dtype="bf16", **comb)
    elem = {"bf16": 2, "fp32": 4, "fp8": 1, "fp4x2": 1}[dispatch]
    want = _reference_regions(
        ws,
        hidden,
        max_tok,
        topk,
        disp_hidden * elem,
        hidden * 2,
        scale,
        push,
        d["waveSize"],
    )
    assert cb.arena_regions(d, c, region_names=_REGIONS) == want


def test_staging_bytes_come_from_the_kernel_layout():
    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    d = cb.EpDispatchPlan.describe(
        world_size=8, max_tok_per_rank=512, max_recv=4096, scale_bytes=224
    )
    # Only the gfx125x dispatch stages.
    want = 6455296 if d["sharedBytes"] else 0
    assert d["stagingBytes"] == ep_staging_bytes(8, 4096, 224) == want


def test_scale_stride_matches_the_plan():
    # scale_stride_bytes() stays for callers that size without a plan.
    from mori.ops.dispatch_combine_v2.hip_backend import scale_stride_bytes

    for scale, stride in (
        (0, 0),
        (4, 128),
        (32, 128),
        (224, 256),
        (256, 256),
        (260, 384),
    ):
        info = cb.EpDispatchPlan.describe(scale_bytes=scale)
        assert info["scaleStride"] == scale_stride_bytes(scale) == stride


_PUSH_KW = {
    "fp4_blockwise": dict(combine_fp4=1),
    "fp4_blockwise_fp32": dict(combine_fp4=1, combine_fp4_f32_scale=1),
    "fp8_blockwise": dict(combine_fp8=1),
}


@pytest.mark.parametrize("quant", sorted(_PUSH_KW))
@pytest.mark.parametrize("ws", [2, 4, 8])
def test_push_compat_sizes_match_the_cpp_layout(quant, ws):
    # push_wire_nbytes()/push_land_rows() stay for callers that size without a
    # plan (aiter): describe()'s values, which Sizes.PushWireRowsAndLanding pins
    # in C++.
    from types import SimpleNamespace

    from mori.ops.dispatch_combine_v2.hip_backend import (
        push_land_rows,
        push_wire_nbytes,
    )

    cfg = SimpleNamespace(
        hidden_dim=7168,
        quant_type=quant,
        max_num_inp_token_per_rank=128,
        world_size=ws,
        num_experts_per_token=8,
    )
    if not cb.EpDispatchPlan.describe()["sharedBytes"]:
        # The push combine is gfx125x only, and so are the sizes of it.
        with pytest.raises(RuntimeError, match="gfx125x only"):
            push_wire_nbytes(cfg)
        return
    assert push_wire_nbytes(cfg) == (7424 if quant == "fp8_blockwise" else 3840)
    assert push_land_rows(cfg) == 128 * min(ws, 8)
    info = cb.EpCombinePlan.describe(
        world_size=ws, max_tok_per_rank=128, hidden_dim=7168, **_PUSH_KW[quant]
    )
    assert info["pushWireBytes"] == push_wire_nbytes(cfg)
    assert info["pushLandRows"] == push_land_rows(cfg)


class _SizedArena:
    handle = 0xDEAD0000

    def __init__(self, sizes):
        self.sizes = sizes

    def offset(self, name):
        if name not in self.sizes:
            raise KeyError(name)  # as SymmArena does for a region it lacks
        return 0

    def size(self, name):
        return self.sizes[name]


def _small_regions():
    # Every plan the arena serves: dispatch and combine each size part of it.
    return dict(
        cb.arena_regions(
            cb.EpDispatchPlan.describe(**_SMALL), cb.EpCombinePlan.describe(**_SMALL)
        )
    )


def test_an_arena_too_small_for_the_plan_is_refused():
    sizes = _small_regions()
    cb.EpDispatchPlan(**_SMALL, arena=_SizedArena(sizes)).close()
    short = dict(sizes, dispOut=sizes["dispOut"] - 1)
    with pytest.raises(ValueError, match="'dispOut'"):
        cb.EpDispatchPlan(**_SMALL, arena=_SizedArena(short))


def test_an_arena_without_size_is_bound_with_a_warning():
    class Unsized:
        handle = _SizedArena.handle
        offset = _SizedArena.offset

        def __init__(self, sizes):
            self.sizes = sizes

    with pytest.warns(FutureWarning, match="no size"):
        cb.EpDispatchPlan(**_SMALL, arena=Unsized(_small_regions())).close()


def test_a_region_the_other_kernel_sizes_is_not_checked():
    # dispatch does not size outTok (combine does), so a dispatch-only caller may
    # bind it to a region of any size -- aiter binds it to its own comb_inp.
    d = cb.EpDispatchPlan.describe(**_SMALL)
    assert "outTok" not in dict(cb.arena_regions(d))
    sizes = _small_regions()
    cb.EpDispatchPlan(**_SMALL, arena=_SizedArena(dict(sizes, outTok=0))).close()


def test_one_plans_regions_do_not_make_an_arena_it_can_bind():
    # A dispatch-only arena lacks outTok, which the dispatch args still bind.
    sizes = dict(cb.arena_regions(cb.EpDispatchPlan.describe(**_SMALL)))
    with pytest.raises(KeyError):
        cb.EpDispatchPlan(**_SMALL, arena=_SizedArena(sizes))


# --------------------------------------------------------------------------
# aggReserve: the blocks meet in a module global of the kernel, so the caller
# allocates and binds nothing for it
# --------------------------------------------------------------------------


def test_agg_reserve_is_the_default_and_takes_no_argument():
    p = _plan()
    on = p.info.get("aggReserve")
    p.close()
    if not on:
        pytest.skip("aggReserve resolves on for the gfx125x dispatch only")
    assert "resState" not in cb.EpDispatchPlan._arg_names


# --------------------------------------------------------------------------
# The gfx1250 dispatch staging buffer: the caller sizes it, so the size has to
# be the one the kernel lays out
# --------------------------------------------------------------------------


def test_staging_bytes_matches_the_device_layout():
    # EpStagingTotalBytes (ep_cfg.hpp): six sub-arrays, each rounded up to
    # EpScaleAlign. EpArgs carries the base but no length, so the kernel cannot
    # bounds-check what it was given -- a size that drifts low is not an error but
    # a silent write past the end of a torch tensor, into whatever the caching
    # allocator handed out next. If the layout moves, re-derive these from it
    # rather than pasting whatever ep_staging_bytes() starts returning.
    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    if not cb.EpDispatchPlan.describe()["sharedBytes"]:
        pytest.skip("only the gfx125x dispatch stages")
    assert ep_staging_bytes(8, 4096, 224) == 6455296
    assert ep_staging_bytes(8, 4096, 0) == 4358144
    assert ep_staging_bytes(32, 4096, 224) == 19529728
    assert ep_staging_bytes(2, 600, 224, 300) == 690944
    # max_recv not a multiple of 32 is the only shape among realistic configs
    # where a sub-array size is not already a multiple of 128, so it is the only
    # case that exercises the mirror's rounding at all.
    assert ep_staging_bytes(1, 7, 0) == 5248


def test_staging_bytes_resolves_max_recv_the_way_the_kernel_does():
    # EpCfg.maxRecv == 0 means "worldSize * maxTokPerRank" (EpMaxRecv), and it is
    # the Plan default -- so this is the spelling an out-of-tree caller sizing the
    # buffer from a Cfg actually passes. Returning a pool of 0 for it would hand
    # back a 4 KiB buffer the kernel then writes megabytes into.
    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    assert ep_staging_bytes(8, 0, 224, 512) == ep_staging_bytes(8, 8 * 512, 224)
    # A resolved cap wins over the fallback, as EpMaxRecv's ternary does.
    assert ep_staging_bytes(8, 4096, 0, 1) == ep_staging_bytes(8, 4096, 0)
    # Neither spelling given is the case that used to return a short buffer in
    # silence; there is no size to compute, so it cannot be a return value.
    with pytest.raises(ValueError, match="max_tok_per_rank"):
        ep_staging_bytes(8, 0, 224)


# --------------------------------------------------------------------------
# A caller that binds no staging_base: the plan's default buffer
# --------------------------------------------------------------------------


@pytest.fixture
def launched(monkeypatch):
    """The ABI crossing stubbed out: launch() fills the args struct and stops."""
    monkeypatch.setattr(cb.plan_api._load(), "mori_jit_plan_launch", lambda *a: 0)
    cb._FALLBACK_STAGING.clear()


def _shared_bytes(monkeypatch, n):
    # The staging decision on any arch: the gfx125x dispatch is the one with
    # dynamic LDS and a staging size, so faking both runs the fallback where CI runs.
    real = cb._EpDispatchJitPlan.info.fget

    def info(self):
        return {**real(self), "sharedBytes": n, "stagingBytes": 1 << 16 if n else 0}

    monkeypatch.setattr(cb.EpDispatchPlan, "info", property(info))


@pytest.fixture
def stages(monkeypatch):
    _shared_bytes(monkeypatch, 1)


@pytest.fixture
def no_staging(monkeypatch):
    _shared_bytes(monkeypatch, 0)


def test_a_dispatch_that_does_not_stage_binds_nothing(launched, no_staging):
    p = _plan()
    assert "stagingBase" not in p._defaults
    p.launch()
    assert not p._buf.stagingBase
    assert not cb._FALLBACK_STAGING
    p.close()


def test_an_unbound_plan_gets_staging_at_its_first_launch(launched, stages):
    p = _plan()
    assert not cb._FALLBACK_STAGING  # nothing until a launch needs it
    with pytest.warns(FutureWarning, match="staging_base is not bound") as warned:
        p.launch()
    assert warned[0].filename == __file__  # the caller's line, not mori's
    buf = p._fallback_staging
    assert p._buf.stagingBase == buf.data_ptr()
    assert buf.numel() == p.info["stagingBytes"]
    # Settled to an address: later launches reuse the cached struct untouched.
    assert p._defaults["stagingBase"] == buf.data_ptr()
    assert "stagingBase" not in p._dyn_defs
    p.close()


def test_calling_the_plan_resolves_it_too(launched, stages):
    p = _plan()
    with pytest.warns(FutureWarning) as warned:
        p()
    assert warned[0].filename == __file__
    assert p._buf.stagingBase == p._fallback_staging.data_ptr()
    p.close()


def test_unbound_plans_of_one_layout_share_one_buffer(launched, stages):
    import warnings

    a = _plan()
    b = _plan(block_num=2 * a.info["blockNum"])
    resolved = _plan(max_recv=0)  # EpMaxRecv answers a's 1 * 64
    other = _plan(max_tok_per_rank=128, max_recv=128)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        for p in (a, b, resolved, other):
            p.launch()
    # Another geometry, same layout: one buffer, as an op's plans share the op's.
    assert a._buf.stagingBase == b._buf.stagingBase == resolved._buf.stagingBase
    assert a._buf.stagingBase != other._buf.stagingBase
    # One warning per buffer allocated, not per plan.
    assert [w.category for w in warned] == [FutureWarning, FutureWarning]
    for p in (a, b, resolved, other):
        p.close()


@pytest.mark.parametrize("as_tensor", [False, True])
def test_a_bound_staging_base_is_left_alone(launched, stages, as_tensor):
    import warnings

    p = _plan()
    staging = torch.zeros(p.info["stagingBytes"], dtype=torch.uint8, device="cuda")
    # An address, as aiter binds it, or the tensor itself, as mori's op layer does.
    p.bind(staging_base=staging if as_tensor else staging.data_ptr())
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        p.launch()
    assert p._buf.stagingBase == staging.data_ptr()
    assert not cb._FALLBACK_STAGING
    p.close()


def test_a_staging_tensor_is_checked_against_the_plan(launched, stages):
    p = _plan()
    need = p.info["stagingBytes"]
    with pytest.raises(ValueError, match="stagingBytes"):
        p.bind(staging_base=torch.zeros(need - 128, dtype=torch.uint8, device="cuda"))
    buf = torch.zeros(need + 1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="128 B aligned"):
        p.bind(staging_base=buf[1:])
    pool = torch.zeros(need * 2, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="stagingBytes"):
        p.bind(staging_base=pool[: need - 128])  # short, however large its pool
    p.bind(staging_base=buf)
    # Held, and bound by address so the launch struct caches it.
    assert p._staging is buf
    assert p._defaults["stagingBase"] == buf.data_ptr()
    p.bind(staging_base=1 << 20)  # an int is taken on trust
    assert p._defaults["stagingBase"] == 1 << 20
    p.close()
    assert p._staging is None


def test_a_staging_tensor_off_the_plans_device_is_refused(launched, stages):
    p = _plan()
    with pytest.raises(ValueError, match="launches on"):
        p.bind(staging_base=torch.zeros(p.info["stagingBytes"], dtype=torch.uint8))
    assert p._staging is None
    p.close()


def test_a_failed_bind_holds_no_staging(launched, stages):
    p = _plan()
    buf = torch.zeros(p.info["stagingBytes"], dtype=torch.uint8, device="cuda")
    with pytest.raises(TypeError, match="no_such_arg"):
        p.bind(staging_base=buf, no_such_arg=1)
    assert p._staging is None
    p.close()


def test_staging_passed_per_launch_is_left_alone(launched, stages):
    p = _plan()
    for _ in range(2):  # the second launch takes the cached struct
        p.launch(staging_base=1 << 20)
        assert p._buf.stagingBase == 1 << 20
    assert not cb._FALLBACK_STAGING
    p.close()


def test_a_launch_argument_wins_over_a_bound_default_every_launch(launched, no_staging):
    p = _plan()
    weights = torch.zeros(64, device="cuda")
    p.bind(weights_buf=weights)  # a non-int default
    for _ in range(2):  # the second launch takes the cached struct
        p.launch(weights_buf=1 << 20)
        assert p._buf.weightsBuf == 1 << 20
    # Omitting it again is a new argument set: the struct is rebuilt from defaults.
    p.launch()
    assert p._buf.weightsBuf == weights.data_ptr()
    p.launch(weights_buf=1 << 20)
    assert p._buf.weightsBuf == 1 << 20
    p.close()


def test_the_fallback_goes_with_the_last_plan_holding_it(launched, stages):
    import gc
    import warnings

    p = _plan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        p.launch()
    assert len(cb._FALLBACK_STAGING) == 1
    gc.disable()
    try:
        del p  # unclosed: refcounting alone has to free it
        assert not cb._FALLBACK_STAGING
    finally:
        gc.enable()


def test_the_fallback_is_not_allocated_inside_a_capture(launched, stages, monkeypatch):
    p = _plan()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="graph capture"):
        p.launch()
    p.close()


def test_a_capture_reuses_the_buffer_of_its_layout(launched, stages, monkeypatch):
    import warnings

    a = _plan()
    b = _plan(block_num=2 * a.info["blockNum"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        a.launch()  # eager, before any capture
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    b.launch()  # first launched inside a capture
    assert b._buf.stagingBase == a._buf.stagingBase
    a.close()
    b.close()


def test_closing_an_unlaunched_plan_allocates_nothing(launched, stages):
    p = _plan()
    p.close()
    assert p._fallback_staging is None
    assert not cb._FALLBACK_STAGING


def test_closing_the_plans_gives_their_buffer_back(launched, stages):
    import warnings

    a = _plan()
    b = _plan(block_num=2 * a.info["blockNum"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        a.launch()
        b.launch()
    a.close()
    assert len(cb._FALLBACK_STAGING) == 1  # b still holds it
    b.close()
    assert not cb._FALLBACK_STAGING  # both plans still referenced, both closed


def test_a_group_compares_defaults_by_the_address_launch_writes(launched, stages):
    import warnings

    a = _plan()
    b = _plan(block_num=2 * a.info["blockNum"])
    other = _plan(max_tok_per_rank=128, max_recv=128)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        # Neither launched: they compare by layout, and nothing is allocated.
        cb.plan_api.make_launch_group([a, b])
        assert not cb._FALLBACK_STAGING
        with pytest.raises(ValueError, match="stagingBase"):
            cb.plan_api.make_launch_group([a, other])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        a.launch()  # a's default is an address now, b's not yet resolved
    cb.plan_api.make_launch_group([a, b])
    assert isinstance(b._defaults["stagingBase"], cb._FallbackStaging)
    weights = torch.zeros(64, device="cuda")
    a.bind(weights_buf=weights)
    b.bind(weights_buf=weights.data_ptr())  # a tensor and its address agree
    cb.plan_api.make_launch_group([a, b])
    for p in (a, b, other):
        p.close()
