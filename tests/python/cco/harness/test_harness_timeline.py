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
"""timeline.py on the CPU, with made-up stamps."""

import importlib.util
import json
import sys
from pathlib import Path


HARNESS = Path(__file__).resolve().parents[4] / "benchmark" / "cco" / "harness"
spec = importlib.util.spec_from_file_location(
    "harness_timeline", HARNESS / "timeline.py"
)
timeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = timeline
spec.loader.exec_module(timeline)


def _call(late):
    """Four GEMM blocks, three puts, a drain and a reduce; ``late`` delays the tail."""
    events = []
    for b in range(4):
        events += [[f"gemm:start@b{b}", float(b)], [f"gemm:end@b{b}", 50.0 + b + late]]
    events += [[f"put@d{d}", 55.0 + 5 * d + late] for d in range(3)]
    events += [["drain:start", 70.0 + late], ["drain:end", 120.0 + late]]
    events += [["reduce:start", 130.0 + late], ["reduce:end", 160.0 + late]]
    return events


DATA = {"ranks": [[_call(0), _call(2)], [_call(5), _call(7)]]}


def test_groups_of_one_call():
    g = timeline.groups_of_call(_call(0))
    assert g["gemm"]["kind"] == "interval" and g["gemm"]["count"] == 4
    assert (g["gemm"]["first"], g["gemm"]["last"], g["gemm"]["span"]) == (
        0.0,
        53.0,
        53.0,
    )
    assert g["gemm"]["dur"] == 50.0
    assert g["put"] == {
        "kind": "instant",
        "count": 3,
        "first": 55.0,
        "median": 60.0,
        "last": 65.0,
    }


def test_summary_follows_the_critical_rank_and_shows_gaps():
    s = timeline.summarize(DATA)
    assert s["critical_rank"] == 1 and s["calls"] == 2
    assert s["call_us"] == {"min": 161.0, "median": 163.5, "max": 166.0}
    order = [p["group"] for p in s["phases"]]
    assert order == ["gemm", "(gap)", "put", "(gap)", "drain", "(gap)", "reduce"]
    drain = next(p for p in s["phases"] if p["group"] == "drain")
    assert drain["span"] == 50.0 and drain["ranks"]["last"] == {
        "min": 121.0,
        "median": 123.5,
        "max": 126.0,
    }
    text = timeline.render(s)
    assert "critical rank 1" in text and "(nothing stamped)" in text
    assert timeline.summarize({"ranks": [[], []]})["phases"] == []


def test_cli(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(DATA))
    out = tmp_path / "s.json"
    assert timeline.main([str(f), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["critical_rank"] == 1
    assert timeline.main([str(tmp_path / "absent.json")]) == 2
