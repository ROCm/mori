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
"""cases/evaluate.py on the CPU, with a made-up task directory and transcripts."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
CASES = REPO / "benchmark" / "cco" / "harness" / "cases"
CASE = CASES / "gemm_ar_bf16"

spec = importlib.util.spec_from_file_location("cases_evaluate", CASES / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["cases_evaluate"] = ev
spec.loader.exec_module(ev)

DECISIONS = """# decisions
| Round | Candidate | Hypothesis | Change | Predicted | Measured | Decision | Reason | Runs |
|---|---|---|---|---|---|---|---|---|
| dev-1 | c1 | first build | | 1300 | 1310 | keep | | r0 |
| 1 | c1 | overlap the gather | chunks | 1250 | 1290 | revise | | r1 |
| freeze | c2 | | | | | freeze | | |
| 2 | c2 | fewer puts | 8 chunks | 1200 | 1195 | keep | | r2 |
| 3 | | | | | | | | |
"""


def _verdict(verdict="meets-contract", fused=1195.0, existing=1170.0, buckets=True):
    v = {"task": "t", "candidate": "c2", "verdict": verdict, "runs": ["runs/r2"]}
    if buckets:
        v["timing"] = {
            "buckets": [
                {
                    "m": 16384,
                    "fused_us": {"median": fused, "spread": 0.1},
                    "baseline": {"impl": "existing", "median": existing, "spread": 0.1},
                }
            ]
        }
    return v


def _result(duration, summary=None, m=16384):
    return {
        "nproc": 8,
        "duration_s": duration,
        "bench_args": ["-m", str(m)],
        "summary": summary or {},
    }


def _transcript(path, kickoff="human", interventions=2):
    def prompt(kind, ts, text):
        r = {"type": "user", "timestamp": ts, "message": {"content": text}}
        if kind == "human":
            r["origin"] = {"kind": "human"}
        else:
            r["turnOrigin"] = kind
        return r

    def reply(mid, ts, out, model="claude-opus-5-5"):
        usage = {
            "input_tokens": 10,
            "output_tokens": out,
            "cache_read_input_tokens": 100,
        }
        msg = {"id": mid, "model": model, "usage": usage, "content": []}
        return {"type": "assistant", "timestamp": ts, "message": msg}

    use = reply("m1", "2026-11-16T10:00:05Z", 50)
    use["message"]["content"] = [{"type": "tool_use", "id": "t1", "name": "Bash"}]
    records = [
        prompt(kickoff, "2026-11-16T10:00:00Z", "go"),
        use,
        reply("m1", "2026-11-16T10:00:06Z", 50),  # the same message, another block
        {
            "type": "user",
            "timestamp": "2026-11-16T10:00:07Z",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "t1"}]},
        },
        {"type": "user", "isMeta": True, "message": {"content": "<reminder>"}},
        {"type": "cost-state", "totalCostUSD": 3.0},
        reply("m2", "2026-11-16T11:00:00Z", 70),
        {"type": "cost-state", "totalCostUSD": 12.5},
    ]
    for i in range(interventions):
        records.append(prompt("human", f"2026-11-16T12:0{i}:00Z", "and now?"))
    records.append(reply("m3", "2026-11-16T13:00:00Z", 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    sub = path.with_suffix("") / "subagents" / "agent-1.jsonl"
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text(
        json.dumps(reply("s1", "2026-11-16T10:30:00Z", 20, "claude-sonnet-5-5")) + "\n"
    )
    return path, path.with_suffix("")


@pytest.fixture
def task(tmp_path):
    pytest.importorskip("yaml")
    records = tmp_path / "task"
    (records / "verdicts").mkdir(parents=True)
    shutil.copy(CASE / "contract.yaml", records / "contract.yaml")
    (records / "decisions.md").write_text(DECISIONS)
    for name, duration in (("r1", 90.0), ("r2", 45.0)):
        (records / "runs" / name).mkdir(parents=True)
        (records / "runs" / name / "result.json").write_text(
            json.dumps(_result(duration))
        )
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict()))
    main, session = _transcript(tmp_path / "session" / "main.jsonl")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"verdict": "clean", "counts": {"leak": 0}, "base": "a" * 40})
    )
    return records, main, session, audit


def _run(records, main, session, audit, *extra):
    argv = ["--case", str(CASE), "--records", str(records), "--transcript", str(main)]
    argv += ["--transcript", str(session)]
    argv += ["--audit", str(audit)] if audit else []
    assert ev.main(argv + list(extra)) == 0
    return json.loads((records / "evaluation.json").read_text())


def test_a_passing_run_is_recorded(task, tmp_path):
    out = _run(*task, "--append", str(tmp_path / "results.md"))
    assert out["result"] == "pass" and out["case"] == "gemm_ar_bf16"
    t = out["target"]
    assert (
        t["bucket"] == 16384
        and t["met"] is True
        and t["gap_pct"] == pytest.approx(2.14)
    )
    assert out["rounds"] == 2 and out["dev_rows"] == 2
    assert out["gpu_hours"] == pytest.approx(0.3)
    assert out["runpy_launches"] == 2 and out["commit"] == "a" * 40
    s = out["session"]
    assert s["owner_prompts"] == 2 and s["wall_hours"] == 3.0 and s["tool_calls"] == 1
    assert s["tokens"]["output_tokens"] == 50 + 70 + 30 + 20  # m1 once
    assert s["cost_usd"] == 12.5
    assert s["models"] == {"claude-opus-5-5": 3, "claude-sonnet-5-5": 1}
    text = (tmp_path / "results.md").read_text()
    assert (
        text.startswith("| Date |") and "| pass | 1195.0 | 1170.0 | +2.1% | 2 |" in text
    )
    _run(*task, "--append", str(tmp_path / "results.md"))
    assert (tmp_path / "results.md").read_text().count("| pass |") == 2
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (REPO / "benchmark/cco/harness/schemas/evaluation.schema.json").read_text()
    )
    jsonschema.validate(out, schema)


def test_results_follow_the_verdict_and_the_audit(task):
    records, main, session, audit = task
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict(fused=1250.0)))
    assert _run(*task)["result"] == "fail"  # +6.8% against a 5% target
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict("slower")))
    assert _run(*task)["result"] == "fail"
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict()))
    audit.write_text(json.dumps({"verdict": "review", "counts": {"review": 1}}))
    assert _run(*task)["result"] == "pending-review"
    assert _run(*task, "--audit-reviewed")["result"] == "pass"
    audit.write_text(json.dumps({"verdict": "void", "counts": {"leak": 3}}))
    assert _run(*task)["result"] == "void"
    assert _run(records, main, session, None)["result"] == "unaudited"
    (records / "verdicts" / "c2.json").unlink()
    assert _run(records, main, session, None)["result"] == "incomplete"


def test_timing_falls_back_to_the_verdicts_run(task):
    records, main, session, audit = task
    summary = {"existing": {"median_us": 1170.0}, "candidate": {"median_us": 1160.0}}
    (records / "runs" / "r2" / "result.json").write_text(
        json.dumps(_result(45.0, summary))
    )
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict(buckets=False)))
    out = _run(*task)
    assert out["target"]["source"].endswith("runs/r2/result.json")
    assert out["target"]["gap_pct"] == pytest.approx(-0.85) and out["result"] == "pass"


def test_a_driver_kickoff_is_not_an_owner_prompt(tmp_path):
    main, _ = _transcript(tmp_path / "s" / "main.jsonl", kickoff="sdk", interventions=1)
    assert ev.session_stats([main])["owner_prompts"] == 1
    main, _ = _transcript(
        tmp_path / "h" / "main.jsonl", kickoff="human", interventions=0
    )
    assert ev.session_stats([main])["owner_prompts"] == 0
    with pytest.raises(FileNotFoundError):
        ev.session_stats([tmp_path / "absent.jsonl"])


def test_rounds_count_only_rows_with_a_candidate(tmp_path):
    p = tmp_path / "decisions.md"
    assert ev.rounds_of(p) is None
    p.write_text(DECISIONS)
    assert ev.rounds_of(p) == 2


def test_an_optimization_task_reports_its_starting_point(task, tmp_path):
    records, main, session, audit = task
    text = (records / "contract.yaml").read_text()
    text = text.replace(
        "within_existing_pct: 5", "reference: existing\n  within_pct: 0"
    )
    text += "\nkind: optimize\nstart_from: {measured_us: {16384: 1208.2}}\n"
    (records / "contract.yaml").write_text(text)
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict(fused=1160.0)))
    out = _run(*task)
    t = out["target"]
    assert (
        out["result"] == "pass"
        and t["reference"] == "existing"
        and t["within_pct"] == 0
    )
    assert t["start_us"] == 1208.2 and t["improvement_pct"] == pytest.approx(3.99)
    (records / "verdicts" / "c2.json").write_text(json.dumps(_verdict(fused=1171.0)))
    assert _run(*task)["result"] == "fail"  # any distance above the reference fails
