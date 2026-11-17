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
"""loop_state.py and stop_hook.py on the CPU, with made-up task directories."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
HARNESS = REPO / "benchmark" / "cco" / "harness"
sys.path.insert(0, str(HARNESS))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"harness_{name}", HARNESS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loop_state = _load("loop_state")
stop_hook = _load("stop_hook")
CONTRACT = (HARNESS / "cases" / "gemm_ar_bf16" / "contract.yaml").read_text()


@pytest.fixture
def task(tmp_path):
    pytest.importorskip("yaml")
    t = tmp_path / "task"
    t.mkdir()
    (t / "contract.yaml").write_text(
        CONTRACT.replace("approved: false", "approved: true")
    )
    return t


def _touch(path, text="x", age=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    stamp = 1_800_000_000 - age
    os.utime(path, (stamp, stamp))


def _decisions(*rows):
    head = "| Round | Candidate | Hypothesis | Change | Predicted | Measured | Decision | Reason | Runs |\n"
    head += "|---|---|---|---|---|---|---|---|---|\n"
    head += "| dev-1 | c0 | | | | | keep | | |\n"  # the implementer's log: not a round
    return head + "".join(
        f"| {i} | c{i} | | | | | {d} | | |\n" for i, d in enumerate(rows, 1)
    )


def _next(t):
    s = loop_state.state(t)
    assert s["state"] == "continue", s
    return s["next"]


def test_the_owner_comes_first(tmp_path, task):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert loop_state.state(empty)["state"] == "needs-owner"
    (task / "contract.yaml").write_text(CONTRACT)
    assert "not approved" in loop_state.state(task)["reasons"][0]
    (task / "contract.yaml").write_text(
        CONTRACT.replace("approved: false", "approved: true")
    )
    _touch(task / "report.md")
    assert loop_state.state(task)["state"] == "done"
    _touch(task / "BLOCKED", "the lease on v3-046 ended; reserve a node")
    s = loop_state.state(task)
    assert s["state"] == "needs-owner" and "lease" in s["reasons"][0]
    _touch(task / "STOP")
    assert loop_state.state(task)["state"] == "stopped"


def test_the_next_step_follows_the_records(task):
    assert _next(task).startswith("step 1")
    _touch(task / "bounds.md", "decision: no-go, the gain is inside the noise band")
    assert "no-go" in _next(task)
    _touch(task / "bounds.md", "decision: go")
    assert _next(task).startswith("step 2")
    _touch(task / "design" / "c1.md")
    assert _next(task).startswith("step 3")
    _touch(task / "candidates" / "c1.json", age=100)
    assert _next(task) == "step 4: verify the frozen candidate c1"
    _touch(task / "verdicts" / "c1.json", age=50)
    assert _next(task).startswith("step 4: decide on verdict c1")
    _touch(task / "decisions.md", _decisions("revise"))
    assert _next(task).startswith("route the last decision (revise)")
    _touch(task / "candidates" / "c2.json", age=10)
    assert _next(task) == "step 4: verify the frozen candidate c2"
    _touch(task / "verdicts" / "c2.json", age=5)
    _touch(task / "decisions.md", _decisions("revise", "keep"))
    assert _next(task).startswith("step 5")
    _touch(task / "decisions.md", _decisions("revise", "stop"))
    assert "write report.md" in _next(task)


def test_a_spent_budget_asks_for_the_report(task):
    _touch(task / "bounds.md", "go")
    _touch(task / "decisions.md", _decisions(*["revise"] * 8))
    s = loop_state.state(task)
    assert s["state"] == "continue" and s["budget"]["rounds"] == 8
    assert "rounds 8/8" in s["reasons"][0] and "report.md" in s["next"]
    _touch(task / "decisions.md", _decisions("revise"))
    run = task / "runs" / "r1" / "result.json"
    _touch(run, json.dumps({"duration_s": 3600 * 3.1, "nproc": 8}))
    s = loop_state.state(task)
    assert (
        s["budget"]["gpu_hours"] == pytest.approx(24.8)
        and "GPU hours" in s["reasons"][0]
    )
    assert loop_state.main([str(task)]) == 0 and loop_state.main([str(task / "x")]) == 2


def test_the_driver_continues_until_a_brake(task):
    env = {"CCO_FUSION_RECORDS": str(task), "CCO_FUSION_MAX_FLAT": "2"}
    assert stop_hook.decide({"session_id": "s"}, {}) == (None, None)

    def step(session="s"):
        out, book = stop_hook.decide({"session_id": session}, env)
        if book is not None:
            (task / stop_hook.BOOK).write_text(json.dumps(book))
        return out

    out = step()
    assert out["decision"] == "block" and "step 1" in out["reason"]
    _touch(task / "bounds.md", "go")  # progress
    assert step()["decision"] == "block"
    assert step()["decision"] == "block"  # flat 1
    out = step()  # flat 2: brake
    assert (
        "decision" not in out
        and "nothing in the records changed" in out["systemMessage"]
    )
    assert loop_state.state(task)["state"] == "needs-owner"
    (task / "BLOCKED").unlink()
    assert step("another session")["decision"] == "block"  # a new session starts afresh
    env["CCO_FUSION_MAX_CONTINUES"] = "1"
    _touch(task / "design" / "c1.md")
    out = step("another session")
    assert "continued this session 1 times" in out["systemMessage"]
    (task / "BLOCKED").unlink()
    _touch(task / "report.md")
    out = step("third")
    assert "decision" not in out and out["systemMessage"].startswith(
        "cco-fusion driver: done"
    )


def test_the_hook_speaks_the_protocol_and_is_registered(task):
    def call(env):
        return subprocess.run(
            [sys.executable, str(HARNESS / "stop_hook.py")],
            input=json.dumps({"session_id": "s", "hook_event_name": "Stop"}),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, **env},
        ).stdout

    assert call({"CCO_FUSION_RECORDS": ""}) == ""
    out = json.loads(call({"CCO_FUSION_RECORDS": str(task)}))
    assert out["decision"] == "block" and (task / stop_hook.BOOK).is_file()
    for settings in (
        REPO / ".claude/skills/cco-fusion-loop/settings.example.json",
        HARNESS / "cases/gemm_ar_bf16/settings.implementer.json",
    ):
        (entry,) = json.loads(settings.read_text())["hooks"]["Stop"]
        assert entry["hooks"][0]["command"].endswith(
            '/benchmark/cco/harness/stop_hook.py"'
        )
