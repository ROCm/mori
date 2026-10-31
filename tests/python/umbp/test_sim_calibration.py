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

np = pytest.importorskip("numpy")
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


@pytest.mark.parametrize("path", ["cpu_staging", "cpu_staging_read"])
@pytest.mark.parametrize("extra", [0, 0.03])
def test_staging_knee_uses_only_identifiable_training_sizes(path, extra):
    knee = 8 * 1048576
    rows = [
        dict(
            phase="train",
            mode="real",
            path=path,
            effective_threads=1,
            block_bytes=block,
            batch=batch,
            elapsed_ns=500
            + 80 * batch
            + block * batch / 80
            + max(0, block * batch - knee) * extra,
        )
        for block in (4096, 65536, 1048576, 4194304, 8388608, 16777216)
        for batch in (1, 2, 4, 8)
    ]
    rows += [dict(row, phase="engine", elapsed_ns=1e12) for row in rows]
    rows += [dict(row, phase="train", mode="sim", elapsed_ns=1e12) for row in rows]
    rows += [
        dict(row, phase="train", mode="real", concurrency=4, elapsed_ns=1e12)
        for row in rows
    ]
    curve = calibrate.fit_paths(rows, [1], {})[path]
    assert curve["latency_ns"] == pytest.approx(500)
    assert curve["copy_latency_ns"] == pytest.approx(80)
    assert curve["bandwidth_bytes_per_second"] == pytest.approx(80e9)
    if extra:
        assert curve["size_knee_bytes"] == knee
        assert curve["large_size_ns_per_byte"] == pytest.approx(extra)
    else:
        assert "size_knee_bytes" not in curve
        assert "large_size_ns_per_byte" not in curve


def test_unresolved_byte_cost_keeps_the_measured_copy_latency():
    # Small measurement drift can make the unconstrained byte coefficient negative.
    # The constrained fit must retain per-copy cost instead of dropping that model.
    rows = [
        dict(
            row,
            elapsed_ns=1000
            + 50 * row["batch"]
            - row["block_bytes"] * row["batch"] / 10485760,
        )
        for row in measurements()
    ]
    curve = calibrate.fit_profile(rows, [1, 4])
    assert curve["copy_latency_ns"] == pytest.approx([50] * 4, rel=0.01)
    assert curve["bandwidth_bytes_per_second"] == [calibrate.np.finfo(float).max] * 4
    json.dumps(curve, allow_nan=False)


def test_gpu_interleaving_only_enables_measured_non_gather_paths():
    names = ("cpu_gpu", "gpu_cpu", "cpu_gpu_gather", "gpu_cpu_gather")
    rows = [
        dict(row, path=name)
        for row in measurements()
        if row["effective_threads"] == 1
        for name in names
    ]
    curves = calibrate.fit_paths(rows, [1], {})
    for name in names:
        assert curves[name].get("interleave_copies", False) == (
            name in ("cpu_gpu", "gpu_cpu")
        )
    base = {"cpu_gpu": {"channel": "unmeasured"}}
    assert (
        calibrate.fit_paths(measurements(), [1, 4], base)["cpu_gpu"] == base["cpu_gpu"]
    )


def test_file_registration_is_fitted_separately_and_uses_only_training():
    rows = [
        dict(row, path="file_gpu")
        for row in measurements()
        if row["effective_threads"] == 1
    ]
    rows += [
        dict(
            rows[0],
            api="register",
            elapsed_ns=ns,
            model_bytes=0,
            segments=0,
            phase=phase,
        )
        for phase, ns in (("train", 46000), ("train", 47000), ("engine", 1e12))
    ]
    curve = calibrate.fit_paths(rows, [1], {})["file_gpu"]
    assert curve["registration_latency_ns"] == 46500
    assert curve["copy_latency_ns"] == pytest.approx(50)
    assert curve["bandwidth_bytes_per_second"] == pytest.approx(40e9)


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


@pytest.mark.parametrize(
    "path,batch_only",
    [
        ("cpu_file_staged", False),
        ("file_cpu_staged_no_crc", False),
        ("file_cpu_staged_no_crc", True),
    ],
)
def test_ssd_fit_uses_physical_bytes_and_reports_gpu_callers_separately(
    path, batch_only
):
    logarithmic = path.endswith("_no_crc")
    rows = [
        dict(
            row,
            path=path,
            model_bytes=(row["block_bytes"] + 4096) * row["batch"],
            elapsed_ns=1000
            + (row["block_bytes"] + 4096) * row["batch"] / 2
            + 50 * row["batch"]
            + (
                ((0 if batch_only else 12000 * row["batch"]) + 8000)
                * np.log2((row["block_bytes"] + 4096) / 4096)
                if logarithmic
                else 0
            ),
        )
        for row in measurements()
        if row["effective_threads"] == 1
        and (not batch_only or row["block_bytes"] != 1048576)
    ]
    curve = calibrate.fit_paths(
        rows,
        [1],
        {"cpu_file": {"channel": "drive0"}, "file_cpu": {"channel": "drive0"}},
        ssd_io="io_uring" if batch_only else "posix",
    )[path]
    assert curve.get("serialize_latency", False) == batch_only
    assert curve["channel"] == "drive0"
    assert curve["copy_latency_ns"] == pytest.approx(50)
    assert curve["bandwidth_bytes_per_second"] == pytest.approx(2e9)
    if logarithmic:
        assert curve.get("log_size_latency_ns", 0) == pytest.approx(
            0 if batch_only else 12000
        )
        assert curve["batch_log_size_latency_ns"] == pytest.approx(8000)
    validation = [
        dict(rows[0], phase="api", threads=1, api="put", mode=mode, caller_gpu=gpu)
        for gpu, mode in itertools.product((0, 1), ("real", "sim"))
    ]
    assert [r["caller_gpu"] for r in calibrate.compare(validation)] == [0, 1]


def test_profile_merge_keeps_estimates_and_fits_each_copy_implementation():
    rows = [dict(row, path="cpu_cpu") for row in measurements()]
    models = (
        ("cpu_gpu", 25),
        ("cpu_gpu_gather", 45),
        ("cpu_staging", 15),
        ("cpu_staging_read", 12),
    )
    for name, bandwidth in models:
        for row in measurements():
            if row["effective_threads"] != 1:
                continue
            # Coalescing changes copy count while payload bytes stay block*batch.
            for segments in (1, row["batch"]):
                ns = (
                    3000 + row["block_bytes"] * row["batch"] / bandwidth + 70 * segments
                )
                rows.append(dict(row, path=name, segments=segments, elapsed_ns=ns))
    estimate = {
        "channel": "network-link",
        "latency_ns": 9000,
        "bandwidth_bytes_per_second": 20e9,
    }
    paths = calibrate.fit_paths(rows, [1, 4], {"remote_read": estimate})
    assert paths["remote_read"] == estimate
    assert paths["cpu_cpu"]["bandwidth_bytes_per_second"][0] == pytest.approx(40e9)
    for name, bandwidth in models:
        assert paths[name]["latency_ns"] == pytest.approx(3000)
        assert paths[name]["copy_latency_ns"] == pytest.approx(70)
        assert paths[name]["bandwidth_bytes_per_second"] == pytest.approx(
            bandwidth * 1e9
        )
    rows.append(dict(measurements()[0], path="gpu_cpu", phase="engine", mode="sim"))
    base = {"gpu_cpu": estimate, "cpu_cpu": dict(estimate, channel="custom-host")}
    paths = calibrate.fit_paths(rows, [1, 4], base)
    assert paths["gpu_cpu"] == estimate
    assert paths["cpu_cpu"]["channel"] == "custom-host"
    assert paths["cpu_staging"]["channel"] == "custom-host"
    assert paths["cpu_staging_read"]["channel"] == "custom-host"


def test_comparison_keeps_direct_staged_and_memory_paths_separate():
    rows = []
    for path, staged, real, sim in (
        ("cpu_gpu", 0, 10000, 10000),
        ("remote_read", 0, 10000, 10000),
        ("remote_read", 1, 20000, 30000),
    ):
        for mode, ns in (("real", real), ("sim", sim)):
            rows.append(
                dict(
                    path=path,
                    staged=staged,
                    phase="engine",
                    threads=1,
                    block_bytes=4096,
                    batch=1,
                    api="transfer",
                    mode=mode,
                    elapsed_ns=ns,
                )
            )
    results = calibrate.compare(rows)
    assert len(results) == 3
    assert [row["error_pct"] for row in results] == [0, 0, 50]


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "path", ["cpu_cpu", "cpu_gpu", "file_cpu", "file_cpu_staged", "remote_read"]
)
def test_custom_grids_reach_probe_and_keep_validation_independent(
    tmp_path, monkeypatch, overlap, path
):
    probe = tmp_path / "probe"
    probe.touch()
    output = tmp_path / "output"
    train = [4096, 65536, 1048576]
    validation = [4096] if overlap else [8192, 131072]
    argv = ["calibrate", "--probe", str(probe), "--output", str(output)]
    argv += "--cpus 0,1,2,3 --threads 1 --concurrency 1 --rounds 1 --max-mib 1".split()
    argv += ["--paths", path]
    remote = path.startswith("remote_")
    if remote:
        argv += [
            "--max-mib",
            "128",
            "--remote-probe",
            str(probe),
            "--local-address",
            "10.0.0.1",
            "--remote-address",
            "10.0.0.2",
            "--remote-command",
            "ssh owner probe",
        ]

        def start_owner(command, **kwargs):
            assert "--max-mib 128" in command[-1]
            return SimpleNamespace(wait=lambda **_: 0, poll=lambda: 0)

        monkeypatch.setattr(calibrate.subprocess, "Popen", start_owner)
    if "file" in path:
        argv += ["--ssd-dir", str(tmp_path), "--ssd-read-prefill-mib", "256"]
    argv += ["--train-block-bytes", *map(str, train), "--train-batches", "1", "8", "32"]
    argv += [
        "--validation-block-bytes",
        *map(str, validation),
        "--validation-batches",
        "1",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(calibrate.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    phases = set()

    def run(command, **kwargs):
        phase = command[command.index(str(probe)) + (5 if remote else 1)]
        selected_path = path if remote else command[command.index("--path") + 1]
        assert command[2] == ("0" if path == "cpu_cpu" else "0,1,2,3")
        phases.add(phase)
        blocks = train if phase == "train" else validation
        batches = [1, 8, 32] if phase == "train" else [1, 4]
        assert command[command.index("--block-bytes") + 1] == ",".join(map(str, blocks))
        assert command[command.index("--batches") + 1] == ",".join(map(str, batches))
        assert command[command.index("--max-mib") + 1] == ("128" if remote else "1")
        if "file" in path:
            assert command[command.index("--ssd-dir") + 1] == str(tmp_path)
            assert command[command.index("--ssd-durability") + 1] == "strict"
            assert command[command.index("--ssd-api-settle-ms") + 1] == "0"
            assert command[command.index("--ssd-read-prefill-mib") + 1] == "256"
        gpu = int("--gpu-caller" in command)
        csv = "block_bytes,batch,effective_threads,concurrency,api,iteration,elapsed_ns,path,caller_gpu"
        csv += ",ssd_tier_threads\n" if "file" in path else "\n"
        for block, batch in itertools.product(blocks, batches):
            if block * batch > 1048576:
                continue
            apis = ("get",) if "file" in path else ("put", "get")
            for api in apis if phase == "api" else ("transfer",):
                model = (
                    "cpu_gpu_gather"
                    if selected_path == "cpu_gpu" and phase == "train"
                    else selected_path
                )
                csv += f"{block},{batch},1,1,{api},0,{int(1000 + block * batch / 40 + batch * 50)},{model},{gpu}"
                csv += ",4\n" if "file" in path else "\n"
        return SimpleNamespace(stdout=csv)

    monkeypatch.setattr(calibrate.subprocess, "run", run)
    if overlap:
        with pytest.raises(ValueError, match="training and validation overlap"):
            calibrate.main()
        assert not (output / "profile.json").exists()
        assert not (output / "report.json").exists()
    else:
        calibrate.main()
        assert phases == (
            {"train", "engine"}
            if path == "file_cpu" or remote
            else {"train", "engine", "api"}
        )
        report = json.loads((output / "report.json").read_text())
        assert report["workloads"]["validation"]["block-bytes"] == validation
        assert {r["block_bytes"] for r in report["comparisons"]} == set(validation)
        if path.endswith("_staged"):
            assert report["ssd"]["api_read_mode"] == "retained-after-prefill"
            assert report["ssd"]["read_prefill_mib"] == 256
            assert report["ssd"]["tier_io_threads"] == 4
            profile = json.loads((output / "profile.json").read_text())
            assert "file_cpu_staged_no_crc" in profile["paths"]
            assert not any(
                r["path"].endswith("_no_crc") and r["phase"] == "api"
                for r in report["comparisons"]
            )
            assert {
                r["caller_gpu"] for r in report["comparisons"] if r["phase"] == "api"
            } == {0, 1}
        assert (output / "profile.json").exists()
