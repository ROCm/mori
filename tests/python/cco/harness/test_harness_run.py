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
"""benchmark/cco/harness/run.py on the CPU, with a stand-in for the torchrun launch.

The GPU path itself (torchrun, bench_driver.py, the benchmarks) is not exercised here;
bench_driver.py is run in-process against two stand-in benchmarks.
"""

import importlib.util
import json
import sys
import time
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


run = _load("run")
driver = _load("bench_driver")
SPEC = run.OPS["gemm_ar"]


def arms(*modes):
    return run.arms_from_op("gemm_ar", list(modes))


def record(mode, us, validated=True):
    return {
        "mode": mode,
        "max_rank_time_us": us,
        "per_rank_time_us": [us, us - 1.0],
        "critical_rank": 0,
        "validated": validated,
        "rel_l2": 1e-3,
    }


def fake_launch(times, validated=True, drop=0):
    """A build_command stand-in whose 'launch' prints one RESULT_JSON per planned run."""

    def build(plan, nproc, timeout_s, launcher, python, env):
        lines = [
            "RESULT_JSON " + json.dumps(record(p["mode"], times(p), validated))
            for p in plan[: len(plan) - drop]
        ]
        text = "\n".join(["rank 0 log line"] + lines) + "\n"
        return [sys.executable, "-c", f"import sys; sys.stdout.write({text!r})"]

    return build


def split_fused(p):
    base = 100.0 if p["mode"] == "split-sdma" else 80.0
    return base + p["round"] + (0.5 if p["repeat"] else 0.0)


def test_plan_interleaves_rounds_and_repeats_the_baseline():
    plan = run.plan_runs(arms("split-sdma", "fused-sdma"), 2, True, ["-m", "64"])
    assert [(p["round"], p["mode"], p["repeat"]) for p in plan] == [
        (1, "split-sdma", False),
        (1, "fused-sdma", False),
        (1, "split-sdma", True),
        (2, "split-sdma", False),
        (2, "fused-sdma", False),
        (2, "split-sdma", True),
    ]
    assert plan[1]["args"] == ["--mode", "fused-sdma", "-m", "64"]


def test_command_shape():
    plan = run.plan_runs(arms("split-sdma"), 1, False, [])
    cmd = run.build_command(
        plan, 8, 900, ["docker", "exec", "c"], "python3", {"B": "2", "A": "1"}
    )
    assert cmd[:3] == ["docker", "exec", "c"]
    assert cmd[3:8] == ["timeout", "--signal=TERM", "--kill-after=30", "900", "env"]
    assert cmd[8:10] == ["A=1", "B=2"]
    assert cmd[10:15] == [
        "python3",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
    ]
    assert cmd[15] == str(run.REPO / run.DRIVER) and (run.REPO / run.DRIVER).is_file()
    assert json.loads(cmd[16]) == [
        {"bench": str(run.REPO / SPEC["bench"]), "args": ["--mode", "split-sdma"]}
    ]


def test_aggregate_medians_noise_and_comparison():
    plan = run.plan_runs(arms("split-sdma", "fused-sdma"), 3, True, [])
    records = [record(p["mode"], split_fused(p)) for p in plan]
    rows, summary, noise, comparisons = run.aggregate(plan, records, True)
    assert (
        summary["split-sdma"]["median_us"] == 102.0
        and summary["split-sdma"]["rounds"] == 3
    )
    assert summary["fused-sdma"]["median_us"] == 82.0
    assert summary["split-sdma"]["spread_pct"] == pytest.approx(100 * 2 / 102)
    assert noise["source"].startswith("A/A") and noise["pct"] == pytest.approx(
        100 * 0.5 / 102.25
    )
    (c,) = comparisons
    assert c["speedup"] == pytest.approx(102 / 82) and c["outside_noise"] is True
    assert rows[2]["repeat"] is True and rows[2]["time_us"] == 101.5


def test_aggregate_without_aa_uses_the_baseline_spread_and_flags_missing_records():
    plan = run.plan_runs(arms("split-sdma", "fused-sdma"), 2, False, [])
    records = [
        record("split-sdma", 100.0),
        record("fused-sdma", 90.0),
        record("split-sdma", 104.0),
    ]
    rows, summary, noise, _ = run.aggregate(plan, records, False)
    assert noise["source"].startswith("spread of split-sdma") and noise[
        "pct"
    ] == pytest.approx(100 * 4 / 102)
    assert rows[-1]["record"] is None and summary["fused-sdma"]["rounds"] == 1


def test_execute_logs_output(tmp_path):
    status, rc, leftover = run.execute(
        [sys.executable, "-c", "print('RESULT_JSON {}')"], tmp_path, 30, tmp_path
    )
    assert (status, rc, leftover) == ("done", 0, [])
    assert run.parse_results((tmp_path / "stdout.log").read_text()) == [{}]


def test_timeout_kills_the_group_and_what_it_started_in_another_session(
    tmp_path, monkeypatch
):
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    real_wait = run.subprocess.Popen.wait

    def short_wait(
        self, timeout=None
    ):  # the outer timeout fires after 2 s instead of timeout_s + 60
        return real_wait(self, timeout=None if timeout is None else min(timeout, 2))

    monkeypatch.setattr(run.subprocess.Popen, "wait", short_wait)
    status, rc, leftover = run.execute(
        [sys.executable, "-c", code], tmp_path, 1, tmp_path
    )
    child = int((tmp_path / "stdout.log").read_text().split()[0])
    assert status == "timeout" and leftover == []
    assert not run._alive(child)


def test_main_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_command", fake_launch(split_fused))
    monkeypatch.setattr(run, "GAP_FILE", tmp_path / "gap")
    out = tmp_path / "r"
    args = [
        "--op",
        "gemm_ar",
        "--modes",
        "split-sdma,fused-sdma",
        "--rounds",
        "3",
        "--aa",
        "--no-preflight",
    ]
    assert (
        run.main([*args, "--bench-args", "-m 64 -n 128 -k 256", "--out", str(out)]) == 0
    )
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "ok" and result["preflight"] == "skipped"
    assert result["bench_args"] == ["-m", "64", "-n", "128", "-k", "256"]
    assert (
        len(result["runs"]) == 9
        and result["comparisons"][0]["candidate"] == "fused-sdma"
    )
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((HARNESS / "schemas" / "result.schema.json").read_text())
    jsonschema.validate(result, schema)
    # an existing record is evidence: no silent overwrite
    assert run.main([*args, "--out", str(out)]) == 2


@pytest.mark.parametrize(
    "build, status, code",
    [
        (fake_launch(split_fused, validated=False), "validation-failed", 1),
        (fake_launch(split_fused, drop=2), "incomplete", 3),
    ],
)
def test_main_reports_failed_and_missing_runs(
    tmp_path, monkeypatch, build, status, code
):
    monkeypatch.setattr(run, "build_command", build)
    monkeypatch.setattr(run, "GAP_FILE", tmp_path / "gap")
    args = [
        "--op",
        "gemm_ar",
        "--modes",
        "split-sdma,fused-sdma",
        "--no-preflight",
        "--out",
        str(tmp_path / "r"),
    ]
    assert run.main(args) == code
    assert json.loads((tmp_path / "r" / "result.json").read_text())["status"] == status


def test_preflight_refusal_stops_before_the_launch(tmp_path, monkeypatch):
    def refuse(args, launcher, require_sdma, out_dir):
        return {
            "info": {"host": "n"},
            "findings": [{"level": "FAIL", "code": "busy", "message": "x"}],
            "result": "FAIL",
        }

    def never(*a, **k):
        raise AssertionError("launched despite a refusal")

    monkeypatch.setattr(run, "run_preflight", refuse)
    monkeypatch.setattr(run, "execute", never)
    out = tmp_path / "r"
    assert (
        run.main(["--op", "gemm_ar", "--modes", "split-sdma", "--out", str(out)]) == 2
    )
    assert json.loads((out / "result.json").read_text())["status"] == "refused"


@pytest.mark.parametrize("modes", ["nope", "split-sdma,split-sdma", ""])
def test_usage_errors(tmp_path, modes):
    assert run.main(["--op", "gemm_ar", "--modes", modes, "--out", str(tmp_path)]) == 2


def test_unwritable_run_directory_is_a_usage_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    args = [
        "--op",
        "gemm_ar",
        "--modes",
        "split-sdma",
        "--no-preflight",
        "--out",
        str(blocker / "r"),
    ]
    assert run.main(args) == 2


def test_dry_run_prints_the_plan(tmp_path, capsys):
    assert (
        run.main(
            [
                "--op",
                "gemm_ar",
                "--modes",
                "gemm-only,split-sdma",
                "--dry-run",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert len(printed["plan"]) == 6 and "torch.distributed.run" in printed["command"]
    assert not (tmp_path / "result.json").exists()


def test_gap_between_multirank_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "GAP_FILE", tmp_path / "gap")
    monkeypatch.setattr(run, "GAP_S", 0.3)
    run.mark_gap(8)
    start = time.time()
    assert run.wait_gap(8) > 0 and time.time() - start >= 0.25
    assert run.wait_gap(1) == 0.0


def test_collect_isa_hashes_and_checks_each_dump(tmp_path):
    kernel = tmp_path / "isa" / "k1"
    kernel.mkdir(parents=True)
    text = (
        '\t.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"\n\t.type\tk1,@function\nk1:\n'
        "\ts_endpgm\n.Lfunc_end0:\n\t.amdgpu_metadata\n---\namdhsa.kernels:\n"
        "  - .agpr_count:     0\n    .name:           k1\n    .private_segment_fixed_size: 0\n"
        "    .vgpr_count:     8\n    .vgpr_spill_count: 0\n    .sgpr_spill_count: 0\n"
        "amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx950\n...\n\t.end_amdgpu_metadata\n"
    )
    (kernel / "22_final_isa.s").write_text(text)
    (item,) = run.collect_isa(tmp_path / "isa")
    assert (
        item["kernel"] == "k1"
        and item["isa_check"] == "PASS"
        and len(item["sha256"]) == 64
    )


def test_candidate_and_verdict_schemas_accept_filled_in_records():
    jsonschema = pytest.importorskip("jsonschema")
    schemas = {
        n: json.loads((HARNESS / "schemas" / f"{n}.schema.json").read_text())
        for n in ("candidate", "verdict")
    }
    candidate = {
        "task": "t",
        "candidate": "c1",
        "commit": "f" * 40,
        "code_objects": [{"kernel": "k", "sha256": "0" * 64}],
        "run_command": "python run.py ...",
        "baseline_command": "python run.py ...",
    }
    jsonschema.validate(candidate, schemas["candidate"])
    verdict = {
        "task": "t",
        "candidate": "c1",
        "commit": "f" * 40,
        "identity": {"code_objects_match": True, "mismatches": []},
        "environment": {"node": "n"},
        "correctness": {"rule": "bit-exact", "fused_path_ran": True, "failures": []},
        "verdict": "meets-contract",
    }
    jsonschema.validate(verdict, schemas["verdict"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(verdict, verdict="probably fine"), schemas["verdict"])


def test_templates_carry_every_required_field():
    skill = HARNESS.parents[2] / ".claude" / "skills" / "cco-fusion-loop" / "templates"
    for name in ("candidate", "verdict"):
        schema = json.loads((HARNESS / "schemas" / f"{name}.schema.json").read_text())
        template = json.loads((skill / f"{name}.example.json").read_text())
        assert set(schema["required"]) <= set(template), name
        if name == "verdict":
            assert set(schema["properties"]["correctness"]["required"]) <= set(
                template["correctness"]
            )


def test_baseline_need_not_be_the_first_mode(tmp_path, monkeypatch):
    plan = run.plan_runs(
        arms("gemm-only", "split-sdma", "fused-sdma"), 1, True, [], "split-sdma"
    )
    assert [(p["mode"], p["repeat"]) for p in plan][-1] == ("split-sdma", True)

    def times(p):
        return {"gemm-only": 358.0, "split-sdma": 1466.0, "fused-sdma": 1162.0}[
            p["mode"]
        ] + p["round"]

    records = [record(p["mode"], times(p)) for p in plan]
    _, _, noise, comparisons = run.aggregate(plan, records, True, "split-sdma")
    assert {c["candidate"]: round(c["speedup"], 3) for c in comparisons} == {
        "gemm-only": round(1467 / 359, 3),
        "fused-sdma": round(1467 / 1163, 3),
    }
    assert all(c["baseline"] == "split-sdma" for c in comparisons)
    monkeypatch.setattr(run, "build_command", fake_launch(times))
    monkeypatch.setattr(run, "GAP_FILE", tmp_path / "gap")
    args = [
        "--op",
        "gemm_ar",
        "--modes",
        "gemm-only,split-sdma,fused-sdma",
        "--baseline",
        "split-sdma",
    ]
    assert (
        run.main([*args, "--aa", "--no-preflight", "--out", str(tmp_path / "r")]) == 0
    )
    assert (
        json.loads((tmp_path / "r" / "result.json").read_text())["baseline"]
        == "split-sdma"
    )
    assert (
        run.main([*args[:4], "--baseline", "fused-lsa", "--out", str(tmp_path / "x")])
        == 2
    )


STAND_IN = """
import argparse, json


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="{mode}")
    p.add_argument("-m", type=int, default=1)
    return p


def run(args):
    print("RESULT_JSON " + json.dumps({{"mode": args.mode, "us": {us} + args.m, "validated": True}}))
    return 0
"""


def stand_in(tmp_path, name, mode, us):
    path = tmp_path / f"{name}.py"
    path.write_text(STAND_IN.format(mode=mode, us=us))
    return path


def test_driver_runs_two_benchmarks_interleaved_in_one_process(tmp_path, capsys):
    a, b = stand_in(tmp_path, "old", "fused", 100), stand_in(
        tmp_path, "new", "cand", 90
    )
    runs = [
        {"bench": str(a), "args": ["--mode", "fused", "-m", "4"]},
        {"bench": str(b), "args": ["-m", "4"]},
        {"bench": str(a), "args": ["--mode", "fused", "-m", "4"]},
    ]
    assert driver.main([json.dumps(runs)]) == 0
    got = run.parse_results(capsys.readouterr().out)
    assert [(r["mode"], r["us"]) for r in got] == [
        ("fused", 104),
        ("cand", 94),
        ("fused", 104),
    ]


def test_arms_file_and_end_to_end(tmp_path, monkeypatch):
    a, b = stand_in(tmp_path, "old", "fused", 100), stand_in(
        tmp_path, "new", "cand", 90
    )
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(
        json.dumps(
            {
                "arms": [
                    {
                        "name": "existing",
                        "bench": str(a),
                        "args": "--mode fused",
                        "time_field": "us",
                    },
                    {
                        "name": "candidate",
                        "bench": str(b),
                        "time_field": "us",
                        "sdma": False,
                    },
                ]
            }
        )
    )
    loaded = run.load_arms(arms_file)
    assert [x["args"] for x in loaded] == [["--mode", "fused"], []]
    plan = run.plan_runs(loaded, 2, True, ["-m", "4"])
    assert [p["expect_mode"] for p in plan] == ["fused", None, "fused"] * 2

    def times(p):
        return 100.0 if p["mode"] == "existing" else 95.0

    def build(plan, nproc, timeout_s, launcher, python, env):
        lines = []
        for p in plan:
            mode = "fused" if p["mode"] == "existing" else "cand"
            lines.append(
                "RESULT_JSON "
                + json.dumps({"mode": mode, "us": times(p), "validated": True})
            )
        text = "\n".join(lines) + "\n"
        return [sys.executable, "-c", f"import sys; sys.stdout.write({text!r})"]

    monkeypatch.setattr(run, "build_command", build)
    monkeypatch.setattr(run, "GAP_FILE", tmp_path / "gap")
    out = tmp_path / "r"
    argv = [
        "--arms",
        str(arms_file),
        "--aa",
        "--no-preflight",
        "--bench-args",
        "-m 4",
        "--out",
        str(out),
    ]
    assert run.main(argv) == 0
    result = json.loads((out / "result.json").read_text())
    assert (
        result["status"] == "ok"
        and result["op"] is None
        and result["modes"] == ["existing", "candidate"]
    )
    assert result["comparisons"][0]["speedup"] == pytest.approx(100 / 95)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(
        result, json.loads((HARNESS / "schemas" / "result.schema.json").read_text())
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--modes", "split-sdma"],
        ["--op", "gemm_ar"],
        ["--op", "gemm_ar", "--modes", "split-sdma", "--arms", "x.json"],
        ["--arms", "missing.json"],
    ],
)
def test_op_and_arms_are_exclusive(tmp_path, argv):
    assert run.main([*argv, "--out", str(tmp_path / "r")]) == 2


def test_arms_need_existing_benchmarks(tmp_path):
    bad = tmp_path / "arms.json"
    bad.write_text(
        json.dumps({"arms": [{"name": "x", "bench": str(tmp_path / "nope.py")}]})
    )
    with pytest.raises(run.UsageError):
        run.load_arms(bad)


def test_arms_take_variables_from_the_environment(tmp_path, monkeypatch):
    bench = stand_in(tmp_path, "new", "cand", 90)
    arms_file = tmp_path / "arms.yaml.json"
    arms_file.write_text(
        json.dumps(
            {
                "arms": [
                    {
                        "name": "candidate",
                        "bench": "${BENCH_DIR}/" + bench.name,
                        "args": "--impl ${CANDIDATE} --tag $TAG",
                    },
                    {"name": "listed", "bench": str(bench), "args": ["${CANDIDATE}"]},
                ]
            }
        )
    )
    monkeypatch.setenv("BENCH_DIR", str(bench.parent))
    monkeypatch.setenv("CANDIDATE", "pkg.mod:Fast")
    monkeypatch.setenv("TAG", "t1")
    arms = run.load_arms(arms_file)
    assert arms[0]["bench"] == str(bench)
    assert arms[0]["args"] == ["--impl", "pkg.mod:Fast", "--tag", "t1"]
    assert arms[1]["args"] == ["pkg.mod:Fast"] and arms[0]["validate"] is True
    monkeypatch.delenv("CANDIDATE")
    with pytest.raises(run.UsageError, match=r"\$\{CANDIDATE\} is not set"):
        run.load_arms(arms_file)


def test_diagnostic_arms_are_timed_but_do_not_fail_the_run(tmp_path, monkeypatch):
    a, b = stand_in(tmp_path, "old", "fused", 100), stand_in(tmp_path, "new", "cut", 60)
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(
        json.dumps(
            {
                "arms": [
                    {"name": "existing", "bench": str(a), "time_field": "us"},
                    {
                        "name": "stop-after-gemm",
                        "bench": str(b),
                        "time_field": "us",
                        "validate": False,
                    },
                ]
            }
        )
    )
    checked = {"existing": True}

    def build(plan, nproc, timeout_s, launcher, python, env):
        lines = [
            "RESULT_JSON "
            + json.dumps(
                {
                    "us": 100.0 if p["mode"] == "existing" else 60.0,
                    "validated": checked.get(p["mode"], False),
                }
            )
            for p in plan
        ]
        text = "\n".join(lines) + "\n"
        return [sys.executable, "-c", f"import sys; sys.stdout.write({text!r})"]

    monkeypatch.setattr(run, "build_command", build)
    monkeypatch.setattr(run, "GAP_FILE", tmp_path / "gap")
    monkeypatch.setattr(run, "GAP_S", 0.0)
    argv = ["--arms", str(arms_file), "--rounds", "2", "--no-preflight", "--out"]
    assert run.main(argv + [str(tmp_path / "ok")]) == 0
    result = json.loads((tmp_path / "ok" / "result.json").read_text())
    diag = result["summary"]["stop-after-gemm"]
    assert result["status"] == "ok" and diag["diagnostic"] is True
    assert diag["validated"] is None and diag["median_us"] == 60.0
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(
        result, json.loads((HARNESS / "schemas" / "result.schema.json").read_text())
    )
    checked["existing"] = False  # a checked arm that fails still fails the run
    assert run.main(argv + [str(tmp_path / "bad")]) == 1
