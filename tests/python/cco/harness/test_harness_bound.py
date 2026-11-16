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
"""benchmark/cco/harness/bound.py on the CPU.

The worked example is gemm_ar's measured phase breakdown (G950-GAR-01 in
.claude/skills/cco-fusion-facts/gfx950.md): the model has to put the measured
fused call inside its range, or it is not a bound worth trusting.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[4] / "benchmark" / "cco" / "harness"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"cco_harness_{name}", HARNESS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bound = _load("bound")

# DeepSeek-V4.1-Flash wo_b, TP4, M=16384, split-SDMA phases in us (G950-GAR-01).
GAR = {
    "m": 16384,
    "split_us": 1638.8,
    "tiles": 1280,
    "phases": [
        {"name": "gemm", "us": 149.9, "role": "compute"},
        {"name": "scatter", "us": 728.0, "role": "overlap"},
        {"name": "reduce", "us": 32.4, "role": "serial"},
        {"name": "gather", "us": 727.8, "role": "serial"},
    ],
}
FUSED_MEASURED_US = 1549.6  # fused-SDMA whole call, 8 chunks, same run


def config(**overrides):
    data = {
        "t0_us": [4, 9],
        "n_max": 16,
        "noise_band_pct": 1.0,
        "min_gain_pct": 5.0,
        "cus": 256,
    }
    data.update(overrides)
    return bound.make_config(data)


def test_pipeline_reduces_to_both_limits():
    # compute-bound: the last chunk's transfer is exposed
    assert bound.pipeline_us(1000.0, 400.0, 8, 5.0) == pytest.approx(
        1000.0 + (400.0 + 7 * 5.0) / 8
    )
    # communication-bound: the first chunk's compute is exposed, every extra put adds t0
    assert bound.pipeline_us(150.0, 728.0, 8, 5.0) == pytest.approx(
        150.0 / 8 + 728.0 + 7 * 5.0
    )
    # one chunk is the split path
    assert bound.pipeline_us(150.0, 728.0, 1, 5.0) == pytest.approx(878.0)


def test_worked_example_brackets_the_measured_fused_call():
    r = bound.evaluate_bucket(GAR, config())
    assert r["serial_us"] == pytest.approx(760.2)
    assert r["t_ideal_us"] == pytest.approx(760.2 + 728.0)
    row = next(x for x in r["sweep"] if x["n"] == 8)
    low, high = row["t_real_us"]
    assert low == pytest.approx(760.2 + 149.9 / 8 + 728.0 + 7 * 4)
    assert high == pytest.approx(760.2 + 149.9 / 8 + 728.0 + 7 * 9)
    assert low <= FUSED_MEASURED_US <= high
    assert row["bound"] == "communication"


def test_decision_takes_the_high_t0_and_its_best_n():
    r = bound.evaluate_bucket(GAR, config())
    assert r["best_n"] == {"t0_low": 6, "t0_high": 4}
    assert r["saving_real_pct"][0] == pytest.approx(
        100 * (1638.8 - (760.2 + 149.9 / 4 + 728.0 + 3 * 9)) / 1638.8
    )
    assert r["decision"] == "go"
    assert bound.evaluate_bucket(GAR, config(min_gain_pct=6.0))["decision"] == "no-go"


def test_noise_band_raises_the_threshold():
    r = bound.evaluate_bucket(GAR, config(min_gain_pct=1.0, noise_band_pct=7.0))
    assert r["decision"] == "no-go"
    assert any("7.0%" in reason for reason in r["reasons"])


def test_fewer_tiles_than_cus_is_no_go():
    r = bound.evaluate_bucket(dict(GAR, tiles=200), config())
    assert r["decision"] == "no-go"
    assert any("200 tiles" in reason for reason in r["reasons"])


def test_contention_stretches_the_larger_side():
    r = bound.evaluate_bucket(dict(GAR, concurrent_us=800.0), config())
    assert (r["a_eff_us"], r["b_eff_us"]) == (149.9, 800.0)
    assert r["t_ideal_us"] == pytest.approx(760.2 + 800.0)
    r = bound.evaluate_bucket(dict(GAR, concurrent_us=700.0), config())
    assert r["b_eff_us"] == 728.0
    assert any("no contention" in note for note in r["notes"])


def test_nothing_to_overlap_is_no_go():
    phases = [p for p in GAR["phases"] if p["role"] != "overlap"]
    r = bound.evaluate_bucket(dict(GAR, phases=phases, split_us=None), config())
    assert r["decision"] == "no-go"
    assert any("nothing to overlap" in reason for reason in r["reasons"])


def test_split_defaults_to_the_phase_sum():
    r = bound.evaluate_bucket(dict(GAR, split_us=None), config())
    assert r["t_split_us"] == pytest.approx(149.9 + 728.0 + 32.4 + 727.8)
    assert any("sum of the phases" in note for note in r["notes"])


@pytest.mark.parametrize(
    "bucket",
    [
        dict(GAR, phases=[{"name": "gemm", "us": 1.0, "role": "gemm"}]),
        dict(GAR, phases=[]),
        dict(GAR, phases=[{"name": "gemm", "us": -1.0, "role": "compute"}]),
    ],
)
def test_bad_buckets_are_rejected(bucket):
    with pytest.raises(bound.InputError):
        bound.evaluate_bucket(bucket, config())


def test_cli_writes_markdown_and_json(tmp_path):
    source = tmp_path / "bounds.json"
    source.write_text(json.dumps({"t0_us": [4, 9], "cus": 256, "buckets": [GAR]}))
    md, js = tmp_path / "bounds.md", tmp_path / "out.json"
    assert bound.main([str(source), "--out-md", str(md), "--out-json", str(js)]) == 0
    result = json.loads(js.read_text())["buckets"][0]
    assert result["decision"] == "go"
    assert "| 16384 | 1638.8 |" in md.read_text()


def test_cli_reports_bad_input_with_status_2(tmp_path, capsys):
    source = tmp_path / "bounds.json"
    source.write_text(json.dumps({"buckets": [dict(GAR, phases=[{"us": 1.0}])]}))
    assert bound.main([str(source)]) == 2
    assert "role" in capsys.readouterr().err


def test_example_input_runs(tmp_path):
    pytest.importorskip("yaml")
    out = tmp_path / "bounds.json"
    assert (
        bound.main(
            [
                str(HARNESS / "bounds.example.yaml"),
                "--out-md",
                str(tmp_path / "b.md"),
                "--out-json",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["buckets"][0]["m"] == 16384
