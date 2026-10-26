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


def test_per_block_reservation_follows_the_slot_word():
    # The slot word outside the cco window: every block reserves its own.
    p = _plan(tok_off_ext=1)
    assert not p.info.get("aggReserve")
    p.close()


# --------------------------------------------------------------------------
# The gfx1250 dispatch staging buffer: the caller sizes it, so the size has to
# be the one the kernel lays out
# --------------------------------------------------------------------------


def test_staging_bytes_matches_the_device_layout():
    # ep_staging_bytes() hand-mirrors EpStagingTotalBytes (ep_cfg.hpp): six
    # sub-arrays, each rounded up to EpScaleAlign. EpArgs carries the base but no
    # length, so the kernel cannot bounds-check what it was given -- a mirror that
    # drifts low is not an error but a silent write past the end of a torch tensor,
    # into whatever the caching allocator handed out next. The numbers below come
    # from EpStagingTotalBytes and are the only thing holding the mirror to it; if
    # the layout moves, re-derive them there rather than pasting whatever this
    # function starts returning.
    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    assert ep_staging_bytes(8, 4096, 224) == 12746752
    assert ep_staging_bytes(8, 4096, 0) == 4358144
    assert ep_staging_bytes(32, 4096, 224) == 50987008
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
    assert ep_staging_bytes(8, 4096, 224, 1) == ep_staging_bytes(8, 4096, 224)
    # Neither spelling given is the case that used to return a short buffer in
    # silence; there is no size to compute, so it cannot be a return value.
    with pytest.raises(ValueError, match="max_tok_per_rank"):
        ep_staging_bytes(8, 0, 224)


def test_staging_bytes_agg_reserve_scale_rows():
    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    assert ep_staging_bytes(8, 4096, 224, 512, agg_reserve=True) == 6455296
    assert ep_staging_bytes(32, 4096, 224, 128, agg_reserve=True) == 19529728
    assert ep_staging_bytes(8, 4096, 0, 512, agg_reserve=True) == 4358144
    assert ep_staging_bytes(2, 600, 224, 300, agg_reserve=True) == 690944
    assert ep_staging_bytes(2, 600, 224, 300) == 473856
    with pytest.raises(ValueError, match="max_tok_per_rank"):
        ep_staging_bytes(8, 4096, 224, agg_reserve=True)


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
    # dynamic LDS, so faking it runs the fallback where CI runs.
    real = cb._EpDispatchJitPlan.info.fget
    monkeypatch.setattr(
        cb.EpDispatchPlan,
        "info",
        property(lambda self: {**real(self), "sharedBytes": n}),
    )


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
    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    p = _plan()
    assert not cb._FALLBACK_STAGING  # nothing until a launch needs it
    with pytest.warns(FutureWarning, match="staging_base is not bound") as warned:
        p.launch()
    assert warned[0].filename == __file__  # the caller's line, not mori's
    buf = p._fallback_staging
    assert p._buf.stagingBase == buf.data_ptr()
    info = p.info
    assert buf.numel() == ep_staging_bytes(
        info["worldSize"],
        info["maxRecv"],
        info["scaleBytes"],
        info["maxTokPerRank"],
        info["aggReserve"],
    )
    # Settled to an address: later launches reuse the cached struct untouched.
    assert p._defaults["stagingBase"] == buf.data_ptr()
    assert "stagingBase" not in p._dyn_defs
    p.close()


def test_the_fallback_follows_the_scale_row_numbering(launched, stages):
    import warnings

    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    p = _plan(scale_bytes=224)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        p.launch()
    agg = p.info["aggReserve"]
    assert p._fallback_staging.numel() == ep_staging_bytes(1, 64, 224, 64, agg)
    if agg:
        assert p._fallback_staging.numel() > ep_staging_bytes(1, 64, 224, 64)
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
    staging = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    # An address, as mori's op layer and aiter bind it, or the tensor itself.
    p.bind(staging_base=staging if as_tensor else staging.data_ptr())
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        p.launch()
    assert p._buf.stagingBase == staging.data_ptr()
    assert not cb._FALLBACK_STAGING
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
