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
"""Where a cco-fusion-loop task stands, read from its records directory alone.

    python benchmark/cco/harness/loop_state.py <records>/<task> [--json]

The unattended driver (``stop_hook.py``) asks this after every turn; anyone can. It prints
one of four states, the reasons, the next step, and the budget used:

* ``stopped``: the owner created ``STOP`` in the task directory.
* ``needs-owner``: the session wrote ``BLOCKED`` (its question for the owner: GPUs, a
  decision, a permission), or the contract is not approved.
* ``done``: ``report.md`` exists (step 5, or the report of a stopped loop).
* ``continue``: anything else, with the next step from the records: no bounds, no design,
  no frozen candidate, a candidate without a verdict, a verdict without a decision, the
  last decision. A spent budget (rounds or GPU hours from the contract) is ``continue``
  too, with "write report.md and stop" as the next step, so that the run ends with its
  report.
"""

import argparse
import json
import re
import sys
from pathlib import Path

TERMINAL = ("stopped", "needs-owner", "done")


def decision_rows(decisions):
    """(round, candidate, decision) for every row of decisions.md that names a candidate;
    None without the file."""
    if not decisions.is_file():
        return None
    header, rows = None, []
    for line in decisions.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].lower() == "round":
            header = [c.lower() for c in cells]
        elif len(cells) >= 2 and cells[0].strip("-: ") and cells[1]:
            decision = None
            if header and "decision" in header and len(cells) == len(header):
                decision = cells[header.index("decision")].lower()
            rows.append((cells[0], cells[1], decision))
    return rows


def rounds_used(decisions):
    """Numbered rows of decisions.md, one per pass through step 4. Rows labelled otherwise
    (dev-1, freeze) are the implementer's log and do not count. None without the file.
    """
    rows = decision_rows(decisions)
    return None if rows is None else sum(r[0].isdigit() for r in rows)


def last_decision(decisions):
    """The Decision cell of the last numbered row, lower case."""
    numbered = [r for r in decision_rows(decisions) or [] if r[0].isdigit()]
    return numbered[-1][2] if numbered else None


def gpu_hours(task):
    """(GPU hours, launches) over the run.py results under runs/."""
    seconds, launches = 0.0, 0
    for p in sorted(Path(task).glob("runs/**/result.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if r.get("duration_s") and r.get("nproc"):
            seconds += float(r["duration_s"]) * int(r["nproc"])
            launches += 1
    return round(seconds / 3600, 3), launches


def _contract(task):
    p = Path(task) / "contract.yaml"
    if not p.is_file():
        return None
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # an unreadable contract is the owner's to fix
        return {}


def _newest(paths):
    paths = list(paths)
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def state(task):
    task = Path(task)
    out = {"task": str(task), "state": "continue", "reasons": [], "next": None}
    contract = _contract(task)
    budget = (contract or {}).get("budget") or {}
    rounds = rounds_used(task / "decisions.md")
    hours, launches = gpu_hours(task)
    out["budget"] = {
        "rounds": rounds or 0,
        "rounds_max": budget.get("rounds"),
        "gpu_hours": hours,
        "gpu_hours_max": budget.get("gpu_hours"),
        "runpy_launches": launches,
    }

    if (task / "STOP").exists():
        out["state"] = "stopped"
        out["reasons"].append("the owner created STOP")
        return out
    if (task / "BLOCKED").exists():
        out["state"] = "needs-owner"
        text = (task / "BLOCKED").read_text(encoding="utf-8", errors="replace").strip()
        out["reasons"].append("BLOCKED: " + (text[:300] or "(empty)"))
        return out
    if contract is None:
        out["state"] = "needs-owner"
        out["reasons"].append("no contract.yaml in the task directory (step 0)")
        return out
    if contract.get("approved") is not True:
        out["state"] = "needs-owner"
        out["reasons"].append("the contract is not approved (step 0)")
        return out
    if (task / "report.md").is_file():
        out["state"] = "done"
        out["reasons"].append("report.md exists")
        return out

    spent = []
    if budget.get("rounds") and (rounds or 0) >= budget["rounds"]:
        spent.append(f"rounds {rounds}/{budget['rounds']}")
    if budget.get("gpu_hours") and hours >= budget["gpu_hours"]:
        spent.append(f"GPU hours {hours:.1f}/{budget['gpu_hours']}")
    if spent:
        out["reasons"].append("budget spent: " + ", ".join(spent))
        out["next"] = (
            "stop rule: write report.md with the best verified result, then stop"
        )
        return out

    bounds = task / "bounds.md"
    candidates = list((task / "candidates").glob("*.json"))
    verdicts = list((task / "verdicts").glob("*.json"))
    decision = last_decision(task / "decisions.md")
    newest_candidate, newest_verdict = _newest(candidates), _newest(verdicts)
    if not bounds.is_file():
        out["next"] = "step 1: baseline and bounds (bounds.md)"
    elif re.search(
        r"(?i)\bno-go\b", bounds.read_text(encoding="utf-8", errors="replace")
    ):
        out["next"] = (
            "step 1 says no-go: write report.md (the negative result), then stop"
        )
    elif not list((task / "design").glob("*.md")):
        out["next"] = "step 2: design (design/<candidate>.md)"
    elif not candidates:
        out["next"] = (
            "step 3: build, check and freeze a candidate (candidates/<id>.json)"
        )
    elif (
        not verdicts
        or newest_candidate.stat().st_mtime > newest_verdict.stat().st_mtime
    ):
        out["next"] = f"step 4: verify the frozen candidate {newest_candidate.stem}"
    elif (rounds or 0) < len(verdicts):
        out["next"] = (
            f"step 4: decide on verdict {newest_verdict.stem} (a row in decisions.md)"
        )
    elif decision == "stop":
        out["next"] = "the last decision is stop: write report.md, then stop"
    elif decision == "keep":
        out["next"] = "step 5: deliver (operator, dispatch rule, tests, report.md)"
    else:
        out["next"] = f"route the last decision ({decision or 'none'}): step 2 or 3"
    return out


def summary(s):
    b = s["budget"]
    rounds = f"{b['rounds']}/{b['rounds_max'] or '?'}"
    hours = f"{b['gpu_hours']:.1f}/{b['gpu_hours_max'] or '?'}"
    lines = [f"state: {s['state']}"]
    lines += [f"why: {r}" for r in s["reasons"]]
    if s["next"]:
        lines.append(f"next: {s['next']}")
    lines.append(f"budget: rounds {rounds}, GPU hours {hours}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("task", help="the task's records directory")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if not Path(args.task).is_dir():
        print(f"loop_state.py: no such directory {args.task}", file=sys.stderr)
        return 2
    s = state(args.task)
    print(json.dumps(s, indent=2) if args.json else summary(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
