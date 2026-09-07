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
"""Calibration for the primitive gate in umbp_prim_ci.py.

The floor is only worth having if it separates the two builds it was drawn
from, so it is checked against both of them here rather than asserted to be
a sensible-looking number.  The throughputs below are the measured
`umbp_prim_bench.py` results for the client-path regression that motivated
the gate: an exclusive lock over the resolve path left `batch_exists`
throughput flat in rank count, while the build before it scaled ~3.9x.

Pure arithmetic over recorded numbers -- no mori, no server, no GPU -- so it
runs in the ordinary pytest step and fails loudly if someone retunes the
floor past the regression it exists to catch.
"""
import pytest

umbp_prim_ci = pytest.importorskip("umbp_prim_ci")

# Aggregate batch_exists keys/s by rank count, both arms, from one idle node
# with the same geometry and image pair.  The 1- and 8-rank figures are the
# measured ones; 2 and 4 are reconstructed from the recorded scaling factors
# (1.91 / 3.27 and 1.37 / 0.91).  The gate reads only the ends, so the middle
# is here to keep the fixture the shape of a real sweep.
GOOD_EXISTS = {1: 2_643_915, 2: 5_049_878, 4: 8_645_601, 8: 10_329_315}
BAD_EXISTS = {1: 964_177, 2: 1_320_922, 4: 877_401, 8: 720_637}


def _payload(exists_by_rank, failures=0, mismatches=0, absent_frac=0.0):
    """A result file shaped like the one umbp_prim_bench.py writes."""
    rows = []
    for ranks, keys_per_s in exists_by_rank.items():
        for op in ("put", "get", "exists", "put_ranges", "get_ranges"):
            rows.append(
                {
                    "op": op,
                    "ranks": ranks,
                    "calls": 40,
                    "p50_ms": 1.0,
                    "p99_ms": 2.0,
                    "keys_per_s": keys_per_s if op == "exists" else 100_000.0,
                    "mib_per_s": 0.0 if op == "exists" else 1_000.0,
                    "failures": failures if op == "exists" else 0,
                    "mismatches": mismatches if op == "get" else 0,
                }
            )
    return {
        "backend": "umbp-server",
        "address": "unix:///tmp/x.grpc.sock",
        "geometry": {"exists_absent_frac": absent_frac},
        "rows": rows,
    }


def test_default_floor_separates_the_two_arms():
    """The whole point: the good build passes and the bad one does not."""
    floor = umbp_prim_ci.DEFAULT_EXISTS_SCALING_FLOOR
    # These are idle-node numbers; the floor also has to clear the much
    # lower scores a contended runner produces -- see the test below.
    good = umbp_prim_ci.check_exists_scaling("good", _payload(GOOD_EXISTS), floor)[0]
    bad = umbp_prim_ci.check_exists_scaling("bad", _payload(BAD_EXISTS), floor)[0]
    assert good.ok, f"the known-good arm must pass: {good}"
    assert not bad.ok, f"the known-regressed arm must fail: {bad}"
    assert good.value == "3.91x"
    assert bad.value == "0.75x"


def test_floor_has_margin_under_the_good_arm():
    """A floor set flush against the good arm turns a busy runner into a red
    build, so keep the distance from it visible and enforced."""
    good_scale = GOOD_EXISTS[8] / GOOD_EXISTS[1]
    bad_scale = BAD_EXISTS[8] / BAD_EXISTS[1]
    floor = umbp_prim_ci.DEFAULT_EXISTS_SCALING_FLOOR
    assert bad_scale < floor < good_scale
    assert good_scale / floor >= 1.5, "less than 1.5x of headroom under the good arm"


def test_correctness_gate_catches_a_failed_operation():
    findings = umbp_prim_ci.check_correctness("arm", _payload(GOOD_EXISTS, failures=3))
    assert not findings[0].ok
    assert "3 failed" in findings[0].detail


def test_correctness_gate_catches_a_readback_mismatch():
    findings = umbp_prim_ci.check_correctness(
        "arm", _payload(GOOD_EXISTS, mismatches=1)
    )
    assert not findings[0].ok
    assert "1 mismatched" in findings[0].detail


def _arm(exists_by_rank, scaling_by_frac=None, **kwargs):
    """An arm as run_arm returns it: one payload per pass.

    The sweep contributes one scaling pass per absent fraction; only the
    0.0 one is gated.
    """
    arm = {"correctness": _payload(exists_by_rank, **kwargs)}
    for frac in (0.0, 0.5, 1.0):
        rows = (scaling_by_frac or {}).get(frac, exists_by_rank)
        arm[f"scaling@{frac:g}"] = _payload(rows, absent_frac=frac)
    return arm


def test_clean_run_passes_every_check():
    findings = umbp_prim_ci.gate({"nomaster": _arm(GOOD_EXISTS)}, 2.0)
    assert all(f.ok for f in findings)
    assert umbp_prim_ci.report_findings(findings)


def test_only_the_zero_absent_fraction_is_gated():
    """The other fractions have no measured baseline, so they are reported
    and must not be able to fail the build."""
    findings = umbp_prim_ci.gate({"arm": _arm(GOOD_EXISTS)}, 2.0)
    scaling = [f for f in findings if "exists scaling" in f.check]
    assert len(scaling) == 3
    gated = {f.check.split("absent=")[1].split()[0] for f in scaling if f.gated}
    assert gated == {"0"}, f"expected only absent=0 gated, got {gated}"


def test_a_regressed_absent_fraction_reports_but_does_not_fail():
    findings = umbp_prim_ci.gate(
        {"arm": _arm(GOOD_EXISTS, scaling_by_frac={0.5: BAD_EXISTS, 1.0: BAD_EXISTS})},
        2.0,
    )
    assert umbp_prim_ci.report_findings(findings), "ungated fractions must not gate"
    ungated = [f for f in findings if not f.gated]
    assert ungated and all(f.value == "0.75x" for f in ungated)


def test_a_regressed_zero_fraction_does_fail():
    findings = umbp_prim_ci.gate(
        {"arm": _arm(GOOD_EXISTS, scaling_by_frac={0.0: BAD_EXISTS})}, 2.0
    )
    assert not umbp_prim_ci.report_findings(findings)


def test_correctness_is_checked_on_the_absent_fraction_passes_too():
    """A False for a present key, or True for an absent one, lands in
    `failures` on those passes -- a false positive being the worse bug."""
    arm = _arm(GOOD_EXISTS)
    arm["scaling@1"]["rows"][0]["failures"] = 7
    findings = umbp_prim_ci.gate({"arm": arm}, 2.0)
    bad = [f for f in findings if f.check.startswith("correctness") and not f.ok]
    assert len(bad) == 1 and "scaling@1" in bad[0].check
    assert not umbp_prim_ci.report_findings(findings)


def test_one_bad_arm_fails_the_whole_gate():
    findings = umbp_prim_ci.gate(
        {"nomaster": _arm(GOOD_EXISTS), "master": _arm(BAD_EXISTS)}, 2.0
    )
    assert not umbp_prim_ci.report_findings(findings)


def test_correctness_reads_the_correctness_pass_not_the_scaling_one():
    """The scaling pass runs `exists` alone, so a correctness failure that
    only the five-primitive pass can see must not be looked for in it."""
    findings = umbp_prim_ci.gate({"arm": _arm(GOOD_EXISTS, mismatches=1)}, 1.0)
    correctness = [f for f in findings if f.check.startswith("correctness")]
    assert not correctness[0].ok


#: Worst 1->8 `exists` scaling seen from an unmodified build on a contended
#: node, over three runs through the scaling pass's geometry (3.50 / 5.39 /
#: 4.33).  The floor has to clear this, or a busy runner turns the gate red
#: on a healthy build.
WORST_HEALTHY_SCALING = 3.50


def test_shipped_floor_sits_between_a_busy_healthy_run_and_the_regression():
    floor = umbp_prim_ci.DEFAULT_EXISTS_SCALING_FLOOR
    assert BAD_EXISTS[8] / BAD_EXISTS[1] < floor < WORST_HEALTHY_SCALING
    assert WORST_HEALTHY_SCALING / floor >= 1.4, "under 40% headroom on a busy runner"
    assert floor / (BAD_EXISTS[8] / BAD_EXISTS[1]) >= 2.0, "too close to the regression"


def test_single_rank_count_is_not_a_pass():
    """A ratio needs two points; silently reporting PASS on one would make
    the gate disappear the moment someone trims the sweep."""
    finding = umbp_prim_ci.check_exists_scaling("arm", _payload({8: 10_000.0}), 2.0)[0]
    assert not finding.ok
    assert "two rank counts" in finding.detail


def test_zero_baseline_is_a_failure_not_a_crash():
    finding = umbp_prim_ci.check_exists_scaling(
        "arm", _payload({1: 0.0, 8: 10_000.0}), 2.0
    )[0]
    assert not finding.ok


def test_pool_is_sized_from_the_whole_sweep():
    """The pool must cover every rank count's keys at once, not just the
    largest -- each writes into its own namespace and none are freed."""
    from types import SimpleNamespace

    args = SimpleNamespace(
        ranks=[1, 2, 4, 8],
        ops=["put", "get", "exists", "put_ranges", "get_ranges"],
        keys_per_rank=512,
        batch=128,
        passes=3,
        ranges=4,
        range_bytes=16384,
        scaling_keys_per_rank=8192,
        scaling_batch=256,
        scaling_passes=100,
        scaling_range_bytes=4096,
        page_bytes=65536,
        exists_absent_fracs=[0.0],
        scaling_key_prefix="s00001",
    )
    correctness, scaling = umbp_prim_ci.passes_for(args)
    page = args.page_bytes
    # 15 rank-slots x 512 keys x (3 seeded + 2 writers x 3 passes), one page each
    assert correctness.resident_bytes(args.ranks, page) == 15 * 512 * 9 * page
    # The scaling pass seeds exists once and never writes again, so its
    # residency does not grow with its (large) pass count.
    assert scaling.resident_bytes(args.ranks, page) == 15 * 8192 * 1 * page

    resident = umbp_prim_ci.pool_bytes(args, headroom=1.0)
    assert resident == sum(
        p.resident_bytes(args.ranks, page) for p in (correctness, scaling)
    )
    assert umbp_prim_ci.pool_bytes(args, headroom=3.0) == 3 * resident

    largest_only = SimpleNamespace(**{**vars(args), "ranks": [8]})
    assert umbp_prim_ci.pool_bytes(largest_only, 1.0) < resident


def test_pool_is_charged_in_whole_pages_not_bytes():
    """The regression this guards cost a false A/B result.

    The DRAM medium is paged and hands out whole pages, so a 4 KiB value in a
    2 MiB page occupies 2 MiB.  Sizing the pool by `keys * object_bytes`
    understates demand by the padding ratio; the pool then evicts continuously
    and, because the just-seeded keys are the least recently used, the run
    dies in the visibility wait rather than reporting anything about space.
    """
    from types import SimpleNamespace

    tiny = umbp_prim_ci.Pass(
        "scaling",
        ["exists"],
        keys_per_rank=1000,
        batch=100,
        passes=1,
        ranges=1,
        range_bytes=4096,
    )
    # One 4 KiB object still costs a whole 2 MiB page: 512x what bytes suggest.
    assert tiny.resident_bytes([1], 2 << 20) == 1000 * (2 << 20)
    assert tiny.resident_bytes([1], 2 << 20) == 512 * tiny.resident_bytes([1], 4096)

    # An object larger than a page spans pages rather than being rounded down.
    wide = umbp_prim_ci.Pass(
        "correctness",
        ["exists"],
        keys_per_rank=10,
        batch=10,
        passes=1,
        ranges=1,
        range_bytes=5 * 65536,
    )
    assert wide.resident_bytes([1], 65536) == 10 * 5 * 65536

    # And the pool the harness asks for tracks the page size it pins.
    args = SimpleNamespace(
        ranks=[1, 2],
        ops=["exists"],
        keys_per_rank=64,
        batch=64,
        passes=1,
        ranges=1,
        range_bytes=4096,
        scaling_keys_per_rank=64,
        scaling_batch=64,
        scaling_passes=1,
        scaling_range_bytes=4096,
        page_bytes=65536,
        exists_absent_fracs=[0.0],
        scaling_key_prefix="s00001",
    )
    small_page = umbp_prim_ci.pool_bytes(args, 1.0)
    args.page_bytes = 2 << 20
    assert umbp_prim_ci.pool_bytes(args, 1.0) == 32 * small_page


# ---------------------------------------------------------------------------
#  Media selection
# ---------------------------------------------------------------------------


def _media(spec, dirs=("/d0", "/d1"), devs=(0,)):
    return umbp_prim_ci.parse_media(spec, list(dirs), list(devs))


def test_single_medium_gets_the_whole_pool():
    for spec in ("dram", "ssd", "hbm"):
        (b,) = _media(spec)
        assert b.kind == spec and b.weight == 100.0


def test_unweighted_media_split_the_remainder_evenly():
    assert [b.weight for b in _media("dram,ssd")] == [50.0, 50.0]
    assert [round(b.weight) for b in _media("dram,ssd,hbm")] == [33, 33, 33]


def test_explicit_weights_are_normalised_to_a_hundred():
    got = {b.kind: b.weight for b in _media("dram:70,ssd:30")}
    assert got == {"dram": 70.0, "ssd": 30.0}
    # Relative weights, not percentages: 3:1 is the same as 75:25.
    got = {b.kind: b.weight for b in _media("dram:3,ssd:1")}
    assert got == {"dram": 75.0, "ssd": 25.0}


def test_a_partial_weight_leaves_the_rest_to_the_unweighted():
    got = {b.kind: b.weight for b in _media("dram:80,ssd")}
    assert got == {"dram": 80.0, "ssd": 20.0}


@pytest.mark.parametrize(
    "spec", ["", "tape", "dram:60,ssd:60,hbm", "dram,dram", "dram:0"]
)
def test_bad_media_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        _media(spec)


def test_ssd_backends_consume_the_supplied_directories():
    with pytest.raises(ValueError):
        _media("ssd", dirs=[])
    (b,) = _media("ssd", dirs=["/mnt/one"])
    assert b.path == "/mnt/one"


def test_policy_splits_capacity_by_weight_and_names_each_medium():
    backends = _media("dram:70,ssd:30", dirs=["/mnt/nvme"])
    pol = umbp_prim_ci.backend_policy(backends, 100 << 30)
    assert pol["schema_version"] == 1
    assert pol["entry_tier"] in pol["tiers"][0]["name"]
    # The unit suffix is mandatory; a bare byte count is rejected by the loader.
    assert pol["backends"]["dram"]["capacity"].endswith("B")
    assert int(pol["backends"]["dram"]["capacity"][:-1]) == int((100 << 30) * 0.7)
    assert int(pol["backends"]["ssd"]["capacity"][:-1]) == int((100 << 30) * 0.3)
    assert pol["backends"]["ssd"]["path"] == "/mnt/nvme"
    # One logical tier: this measures placement across media, not promotion
    # between tiers, which would put an eviction policy inside a latency number.
    assert len(pol["tiers"]) == 1
    assert pol["tiers"][0]["backends"] == {"dram": 70.0, "ssd": 30.0}


def test_policy_gives_hbm_its_device_and_no_path():
    backends = _media("hbm", devs=[3])
    pol = umbp_prim_ci.backend_policy(backends, 8 << 30)
    assert pol["backends"]["hbm3"]["devices"] == [3]
    assert "path" not in pol["backends"]["hbm3"]


def test_tiny_shares_still_get_a_usable_backend():
    """A 1% share of a small pool must not round down to nothing."""
    backends = _media("dram:99,ssd:1", dirs=["/mnt/nvme"])
    pol = umbp_prim_ci.backend_policy(backends, 4 << 20)
    assert int(pol["backends"]["ssd"]["capacity"][:-1]) >= (1 << 20)
