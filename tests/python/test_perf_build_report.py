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
"""Tests for the atomic-free vs atomics comparison in tools/perf/build_report.py.

The nightly runs the internode EP bench twice, the second time with
MORI_EP_DISABLE_RDMA_ATOMICS=1, and the report pairs the two. These pin the three ways that
pairing can quietly go wrong: the atomic-free record being deduplicated away as a copy of its
baseline, the delta being computed against a baseline from a different CI run, and the
atomic-free rows merging into the baseline rows in the main EP tables.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_report",
    Path(__file__).resolve().parents[2] / "tools" / "perf" / "build_report.py",
)
build_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_report)


def _rec(
    run_id="100",
    kernel="v1",
    tokens=128,
    atomic_free=False,
    disp_lat=100.0,
    comb_lat=200.0,
    disp_bw=50.0,
    comb_bw=40.0,
    platform="MI355X_AINIC",
    python="3.12",
    ts=1.0,
):
    params = {
        "world_size": 16,
        "max_tokens": tokens,
        "kernel_type": kernel,
        "dtype": "bfloat16",
        "hidden_dim": 7168,
    }
    if atomic_free:
        params["atomic_free"] = True
    return {
        "category": "internode_ep",
        "params": params,
        "metrics": {
            "dispatch_lat_us": disp_lat,
            "combine_lat_us": comb_lat,
            "dispatch_rdma_bw_gbps": disp_bw,
            "combine_rdma_bw_gbps": comb_bw,
        },
        "run_id": run_id,
        "platform": platform,
        "python": python,
        "ts": ts,
    }


def _write(tmp_path, records):
    (tmp_path / "perf.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return build_report.load_records(str(tmp_path))


def test_atomic_free_record_survives_dedup(tmp_path):
    """Same run, same config, only atomic_free differs: both must load."""
    loaded = _write(tmp_path, [_rec(), _rec(atomic_free=True)])
    assert len(loaded) == 2


def test_delta_is_atomic_free_over_baseline(tmp_path):
    records = _write(
        tmp_path,
        [
            _rec(disp_lat=100.0, disp_bw=50.0),
            _rec(atomic_free=True, disp_lat=110.0, disp_bw=45.0),
        ],
    )
    ((base, af),) = build_report._atomic_free_pairs(records)
    assert base["atomic_free"] is False and af["atomic_free"] is True
    assert build_report._delta_pct(base["disp_lat"], af["disp_lat"]) == pytest.approx(
        10.0
    )
    assert build_report._delta_pct(base["disp_bw"], af["disp_bw"]) == pytest.approx(
        -10.0
    )


def test_baseline_from_another_run_is_not_paired(tmp_path):
    """An atomic-free run whose own baseline is missing must not borrow an older night's."""
    records = _write(
        tmp_path,
        [
            _rec(run_id="99", ts=1.0),
            _rec(run_id="100", atomic_free=True, ts=2.0),
        ],
    )
    assert build_report._atomic_free_pairs(records) == []
    assert build_report._atomic_free_section(records) == ""


def test_pairs_do_not_cross_python_versions_or_token_counts(tmp_path):
    records = _write(
        tmp_path,
        [
            _rec(python="3.10", tokens=128),
            _rec(python="3.12", tokens=128),
            _rec(python="3.12", tokens=1024),
            _rec(python="3.12", tokens=128, atomic_free=True, disp_lat=120.0),
        ],
    )
    pairs = build_report._atomic_free_pairs(records)
    assert len(pairs) == 1
    base, af = pairs[0]
    assert (base["python"], base["tokens"]) == ("3.12", 128)
    assert (af["python"], af["tokens"]) == ("3.12", 128)


def test_main_ep_table_lists_both_variants_separately(tmp_path):
    records = _write(
        tmp_path,
        [
            _rec(kernel="v1"),
            _rec(kernel="v1", atomic_free=True),
            _rec(kernel="v1_ll"),
        ],
    )
    report = build_report.build_markdown(records)
    assert "EP16-V1 (atomic-free)" in report
    # Atomic-free row sits right after its own baseline, before the next kernel.
    order = [
        report.index(">EP16-V1<"),
        report.index("EP16-V1 (atomic-free)"),
        report.index(">EP16-V1-LL<"),
    ]
    assert order == sorted(order)


def test_section_reports_mean_latency_change(tmp_path):
    records = _write(
        tmp_path,
        [
            _rec(tokens=128, disp_lat=100.0, comb_lat=200.0),
            _rec(tokens=128, atomic_free=True, disp_lat=110.0, comb_lat=220.0),
            _rec(tokens=1024, disp_lat=100.0, comb_lat=200.0),
            _rec(tokens=1024, atomic_free=True, disp_lat=130.0, comb_lat=200.0),
        ],
    )
    section = build_report._atomic_free_section(records)
    assert "100 → 110 µs (+10.0%)" in section
    assert "100 → 130 µs (+30.0%)" in section
    # mean of +10% and +30% dispatch, +10% and 0% combine
    assert "dispatch latency +20.0%" in section
    assert "combine latency +5.0%" in section


def test_no_atomic_free_records_leaves_report_unchanged(tmp_path):
    records = _write(tmp_path, [_rec(), _rec(kernel="v1_ll")])
    assert build_report._atomic_free_section(records) == ""
    assert "atomic-free" not in build_report.build_markdown(records)


def test_zero_baseline_does_not_divide_by_zero():
    assert build_report._delta_pct(0, 5) is None
    assert build_report._delta_pct(None, 5) is None
    assert build_report._cmp_cell(0, 5, "µs") == "0 → 5 µs (n/a)"
    assert build_report._cmp_cell(None, 5, "µs") == "—"
