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
"""Check calibration math without a UMBP build or performance assertions in CI."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
spec = importlib.util.spec_from_file_location(
    "calibrate", Path(__file__).resolve().parents[3] / "examples/umbp/calibrate.py"
)
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


def measurements():
    return [
        dict(
            phase="train",
            mode="real",
            effective_threads=p,
            block_bytes=block,
            batch=batch,
            elapsed_ns=1000 + block * batch / (40 * p) + 50 * batch,
        )
        for p in (1, 4)
        for block in (4096, 65536, 1048576)
        for batch in (1, 8, 32)
    ]


def test_fit_recovers_model_without_validation_leakage():
    rows = measurements()
    # Even extreme API/validation/sim observations must not affect the fit.
    rows += [dict(row, phase="api", elapsed_ns=1e12) for row in rows]
    rows += [dict(row, phase="engine", elapsed_ns=1e12) for row in measurements()]
    rows += [dict(row, mode="sim", elapsed_ns=1e12) for row in measurements()]
    rows += [dict(row, concurrency=4, elapsed_ns=1e12) for row in measurements()]
    path = calibrate.fit_profile(rows, [1, 4])
    assert path["latency_ns"] == pytest.approx([1000] * 4)
    assert path["copy_latency_ns"] == pytest.approx([50] * 4)
    assert path["bandwidth_bytes_per_second"] == pytest.approx(
        [
            40e9,
            1 / (1 / 40e9 * 2 / 3 + 1 / 160e9 / 3),
            1 / (1 / 40e9 / 3 + 1 / 160e9 * 2 / 3),
            160e9,
        ]
    )


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_fit_rejects_invalid_measurements(bad):
    rows = measurements()
    rows[0]["elapsed_ns"] = bad
    with pytest.raises(ValueError, match="invalid training"):
        calibrate.fit_profile(rows, [1, 4])


def test_fit_requires_independent_data_for_each_thread_count():
    with pytest.raises(ValueError, match="for 2 threads"):
        calibrate.fit_profile(measurements(), [1, 2, 4])
    with pytest.raises(ValueError, match="insufficient"):
        calibrate.fit_profile([r for r in measurements() if r["batch"] == 1], [1, 4])


def test_report_keeps_absolute_floor_and_large_relative_errors_visible():
    rows = [
        dict(
            phase="api",
            threads=1,
            block_bytes=4096,
            batch=1,
            api="get",
            mode=mode,
            elapsed_ns=ns,
        )
        for mode, ns in [
            ("real", 5000),
            ("real", 6000),
            ("real", 7000),
            ("sim", 8000),
            ("sim", 9000),
            ("sim", 10000),
        ]
    ]
    result = calibrate.compare(rows)[0]
    assert result["error_pct"] == 50
    assert not result["within_tolerance"]
    rows = [dict(row, elapsed_ns=row["elapsed_ns"] / 10) for row in rows]
    assert calibrate.compare(rows)[0]["within_tolerance"]


def test_report_separates_concurrency():
    rows = [
        dict(
            phase="api",
            threads=1,
            concurrency=c,
            block_bytes=4096,
            batch=1,
            api="get",
            mode=mode,
            elapsed_ns=ns,
        )
        for c, mode, ns in [
            (1, "real", 10000),
            (1, "sim", 10000),
            (4, "real", 20000),
            (4, "sim", 30000),
        ]
    ]
    reports = calibrate.compare(rows)
    assert [row["concurrency"] for row in reports] == [1, 4]
    assert [row["error_pct"] for row in reports] == [0, 50]
    assert [row["within_tolerance"] for row in reports] == [True, False]


@pytest.mark.parametrize("overlap", [False, True])
def test_custom_grids_reach_probe_and_keep_validation_independent(
    tmp_path, monkeypatch, overlap
):
    probe = tmp_path / "probe"
    probe.touch()
    output = tmp_path / "output"
    train = [4096, 65536, 1048576]
    validation = [4096] if overlap else [8192, 131072]
    argv = ["calibrate", "--probe", str(probe), "--output", str(output)]
    argv += "--cpus 0 --threads 1 --concurrency 1 --rounds 1 --max-mib 1".split()
    argv += ["--train-block-bytes", *map(str, train), "--train-batches", "1", "8", "32"]
    argv += [
        "--validation-block-bytes",
        *map(str, validation),
        "--validation-batches",
        "1",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(calibrate.os, "sched_getaffinity", lambda _: {0})
    phases = set()

    def run(command, **kwargs):
        phase = command[command.index(str(probe)) + 1]
        phases.add(phase)
        blocks = train if phase == "train" else validation
        batches = [1, 8, 32] if phase == "train" else [1, 4]
        assert command[command.index("--block-bytes") + 1] == ",".join(map(str, blocks))
        assert command[command.index("--batches") + 1] == ",".join(map(str, batches))
        assert command[command.index("--max-mib") + 1] == "1"
        csv = (
            "block_bytes,batch,effective_threads,concurrency,api,iteration,elapsed_ns\n"
        )
        for block, batch in itertools.product(blocks, batches):
            if block * batch > 1048576:
                continue
            for api in ("put", "get") if phase == "api" else ("transfer",):
                csv += f"{block},{batch},1,1,{api},0,{int(1000 + block * batch / 40 + batch * 50)}\n"
        return SimpleNamespace(stdout=csv)

    monkeypatch.setattr(calibrate.subprocess, "run", run)
    if overlap:
        with pytest.raises(ValueError, match="training and validation overlap"):
            calibrate.main()
        assert not (output / "profile.json").exists()
        assert not (output / "report.json").exists()
    else:
        calibrate.main()
        assert phases == {"train", "engine", "api"}
        report = json.loads((output / "report.json").read_text())
        assert report["workloads"]["validation"]["block-bytes"] == validation
        assert {r["block_bytes"] for r in report["comparisons"]} == set(validation)
        assert (output / "profile.json").exists()
