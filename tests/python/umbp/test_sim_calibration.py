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

import hashlib
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


def measurements(threads=(1, 4)):
    return [
        dict(
            phase="train",
            mode="real",
            effective_threads=p,
            block_bytes=block,
            batch=batch,
            elapsed_ns=1000 + block * batch / (40 * p) + 50 * batch,
        )
        for p in threads
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


def test_file_registration_is_fitted_separately_and_uses_only_training():
    rows = [dict(row, path="file_gpu") for row in measurements((1,))]
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


def test_report_uses_medians_and_keeps_workloads_separate():
    cases = [
        ({"path": "cpu_gpu"}, 6000, 9000, False),
        ({"path": "cpu_gpu", "concurrency": 4}, 20000, 30000, False),
        ({"path": "cpu_gpu", "caller_gpu": 1}, 10000, 10000, True),
        ({"path": "remote_read"}, 10000, 10000, True),
        ({"path": "remote_read", "staged": 1}, 20000, 30000, False),
        ({"path": "cpu_cpu"}, 600, 900, True),
    ]
    rows = [
        dict(
            measurements()[0],
            phase="api",
            threads=1,
            api="get",
            mode=mode,
            elapsed_ns=ns * factor,
            **fields,
        )
        for fields, real, sim, _ in cases
        for mode, ns in (("real", real), ("sim", sim))
        for factor in (0.5, 1, 2)
    ]
    reports = calibrate.compare(rows)
    assert len(reports) == len(cases)
    for fields, real, sim, within_tolerance in cases:
        expected = {"concurrency": 1, "caller_gpu": 0, "staged": 0, **fields}
        report = next(r for r in reports if all(r[k] == v for k, v in expected.items()))
        assert report["real_median_ns"] == real
        assert report["sim_median_ns"] == sim
        assert report["error_pct"] == pytest.approx(100 * (sim / real - 1))
        assert report["within_tolerance"] == within_tolerance


@pytest.mark.parametrize(
    "path,batch_only",
    [
        ("cpu_file_staged", False),
        ("file_cpu_staged_no_crc", False),
        ("file_cpu_staged_no_crc", True),
    ],
)
def test_ssd_fit_uses_physical_bytes_and_identifiable_log_costs(path, batch_only):
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
        for row in measurements((1,))
        if not batch_only or row["block_bytes"] != 1048576
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


def test_profile_merge_keeps_estimates_and_fits_each_copy_implementation():
    models = dict(
        cpu_gpu=25,
        gpu_cpu=30,
        cpu_gpu_gather=45,
        gpu_cpu_gather=40,
        cpu_staging=15,
        cpu_staging_read=12,
    )
    rows = [dict(row, path="cpu_cpu") for row in measurements()]
    rows += [
        dict(
            row,
            path=name,
            segments=segments,
            elapsed_ns=3000
            + row["block_bytes"] * row["batch"] / bandwidth
            + 70 * segments,
        )
        for name, bandwidth in models.items()
        for row in measurements((1,))
        for segments in (1, row["batch"])
    ]
    estimate = {
        "channel": "network-link",
        "latency_ns": 9000,
        "bandwidth_bytes_per_second": 20e9,
    }
    base = {
        "remote_read": estimate,
        "gpu_gpu": estimate,
        "cpu_cpu": dict(estimate, channel="custom-host"),
    }
    rows.append(dict(measurements()[0], path="gpu_gpu", phase="engine", mode="sim"))
    paths = calibrate.fit_paths(rows, [1, 4], base)
    assert paths["remote_read"] == paths["gpu_gpu"] == estimate
    for name in ("cpu_cpu", "cpu_staging", "cpu_staging_read"):
        assert paths[name]["channel"] == "custom-host"
    for name, bandwidth in models.items():
        assert paths[name]["latency_ns"] == pytest.approx(3000)
        assert paths[name]["copy_latency_ns"] == pytest.approx(70)
        assert paths[name]["bandwidth_bytes_per_second"] == pytest.approx(
            bandwidth * 1e9
        )
        assert paths[name].get("interleave_copies", False) == (
            name in ("cpu_gpu", "gpu_cpu")
        )
    unmeasured = {name: estimate for name in ("cpu_gpu", "gpu_cpu")}
    fitted = calibrate.fit_paths(measurements(), [1, 4], unmeasured)
    assert all(fitted[name] == curve for name, curve in unmeasured.items())


@pytest.mark.parametrize(
    "path,no_crc,overlap",
    [
        ("cpu_cpu", False, False),
        ("cpu_gpu", False, False),
        ("file_cpu_staged", False, False),
        ("file_cpu_staged", True, False),
        ("remote_read", False, False),
        ("cpu_gpu", False, True),
    ],
)
def test_calibration_workflow(tmp_path, monkeypatch, path, no_crc, overlap):
    probe = tmp_path / "probe"
    probe.touch()
    output = tmp_path / "output"
    base = Path(calibrate.__file__).with_name("sim_profile.json").read_bytes()
    estimates = json.loads(base)["paths"]
    if no_crc:
        estimates[path] = estimates["file_cpu"]
        base = json.dumps({"paths": estimates}).encode()
    base_profile = tmp_path / "base.json"
    base_profile.write_bytes(base)
    train, validation = [4096, 65536, 1048576], [4096] if overlap else [8192, 131072]
    remote = path.startswith("remote_")
    argv = [
        "calibrate",
        "--probe",
        str(probe),
        "--output",
        str(output),
        "--base-profile",
        str(base_profile),
        "--paths",
        path,
    ]
    argv += "--cpus 0,1,2,3 --threads 1 --concurrency 1 --rounds 1 --max-mib".split()
    argv += ["128" if remote else "1"]
    for phase, blocks, batches in (
        ("train", train, [1, 8, 32]),
        ("validation", validation, [1, 4]),
    ):
        argv += [
            f"--{phase}-block-bytes",
            *map(str, blocks),
            f"--{phase}-batches",
            *map(str, batches),
        ]
    if remote:
        argv += [
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
    if no_crc:
        argv.append("--ssd-no-crc")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(calibrate.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    phases = set()

    def run(command, **kwargs):
        # Report provenance must still identify the original, now deleted base.
        base_profile.unlink(missing_ok=True)
        phase = next(p for p in ("train", "engine", "api") if p in command)
        selected = path if remote else command[command.index("--path") + 1]
        if selected == "file_cpu_staged" and "--ssd-no-crc" in command:
            selected += "_no_crc"
        if path == "cpu_gpu" and phase == "train":
            selected += "_gather"
        phases.add(phase)
        blocks, batches = (
            (train, [1, 8, 32]) if phase == "train" else (validation, [1, 4])
        )
        for option, values in (("--block-bytes", blocks), ("--batches", batches)):
            assert command[command.index(option) + 1] == ",".join(map(str, values))
        assert command[command.index("--max-mib") + 1] == ("128" if remote else "1")
        assert command[2] == ("0" if path == "cpu_cpu" else "0,1,2,3")
        if "file" in path:
            assert ("--ssd-no-crc" in command) == no_crc
            assert command[command.index("--ssd-dir") + 1] == str(tmp_path)
            assert command[command.index("--ssd-read-prefill-mib") + 1] == "256"
        csv = "block_bytes,batch,effective_threads,concurrency,api,iteration,elapsed_ns,path,caller_gpu,ssd_tier_threads,staged\n"
        for block, batch in itertools.product(blocks, batches):
            if block * batch > 1048576:
                continue
            for api in ("put", "get") if phase == "api" else ("transfer",):
                ns = int(1000 + block * batch / 40 + batch * 50)
                csv += f"{block},{batch},1,1,{api},0,{ns},{selected},{int('--gpu-caller' in command)},4,{int('--staged' in command)}\n"
        return SimpleNamespace(stdout=csv)

    monkeypatch.setattr(calibrate.subprocess, "run", run)
    if overlap:
        with pytest.raises(ValueError, match="training and validation overlap"):
            calibrate.main()
        assert not (output / "profile.json").exists()
        assert not (output / "report.json").exists()
        return
    calibrate.main()
    assert phases == ({"train", "engine"} if remote else {"train", "engine", "api"})
    report = json.loads((output / "report.json").read_text())
    profile = json.loads((output / "profile.json").read_text())
    assert report["base_profile_sha256"] == hashlib.sha256(base).hexdigest()
    assert {r["block_bytes"] for r in report["comparisons"]} == set(validation)
    if "file" in path:
        assert report["ssd"]["api_read_mode"] == "retained-after-prefill"
        assert report["ssd"]["tier_io_threads"] == 4
        assert {
            r["caller_gpu"] for r in report["comparisons"] if r["phase"] == "api"
        } == {0, 1}
        assert {r["path"] for r in report["comparisons"] if r["phase"] == "api"} == {
            path + "_no_crc" if no_crc else path
        }
        assert "file_cpu_staged_no_crc" in profile["paths"]
        if no_crc:
            assert profile["paths"][path] == estimates[path]
            assert report["path_sources"][path] == "estimate"
            assert report["path_sources"][path + "_no_crc"] == "measured"
