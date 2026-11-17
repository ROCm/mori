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
"""Turn a finished evaluation-case run into one record and one row of the case's results.

    python benchmark/cco/harness/cases/evaluate.py --case <case dir> --records <task dir> \
        --transcript <session.jsonl> [--transcript <session dir>] --audit <audit.json> \
        [--audit-reviewed] [--node-hours H] [--out <task dir>/evaluation.json] \
        [--append <case dir>/results.md]

It reads:

* the contract: the approved copy in the task directory, else the case's (the target bucket,
  ``targets.within_existing_pct``, the budget);
* the verifier's verdicts (``verdicts/*.json``, the newest decides): the verdict and the
  target bucket's timing, candidate against the existing op; when a verdict holds no
  timing for that bucket, the run it lists whose arms include ``existing`` and ``candidate``;
* ``decisions.md`` for the rounds (its numbered rows; ``dev-1`` and the like are the
  implementer's log, counted apart), and every ``runs/**/result.json`` for the GPU time
  spent in run.py launches (duration times ranks);
* the session's transcripts, the main one first: wall time, the owner's prompts after the
  first one (interventions), tool calls, tokens by kind, cost, models;
* the audit report (``audit.py --json``).

The result is ``pass``, ``fail``, ``incomplete`` (no verdict), ``unaudited`` (no audit
report), ``pending-review`` (the audit has items a person has not cleared;
``--audit-reviewed`` says they have) or ``void`` (the audit found a leak).
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_state import gpu_hours as gpu_time  # noqa: E402
from loop_state import decision_rows  # noqa: E402
from loop_state import rounds_used as rounds_of  # noqa: E402

SCHEMA = "cco-fusion/evaluation/v1"
TOKEN_KINDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
HEADER = (
    "| Date | Commit | Models | Result | Candidate (µs) | Existing (µs) | Gap | Rounds "
    "| GPU h | Owner prompts | Wall h | Cost (USD) | Audit |\n"
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
)


def _yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def contract_of(case, records):
    for p in (Path(records) / "contract.yaml", Path(case) / "contract.yaml"):
        if p.is_file():
            return _yaml(p), str(p)
    raise FileNotFoundError("no contract.yaml in the task directory or the case")


def _bench_m(result):
    args = result.get("bench_args") or ""
    args = " ".join(args) if isinstance(args, list) else str(args)
    m = re.search(r"(?:^|\s)-m\s+(\d+)", args)
    return int(m.group(1)) if m else None


def target_timing(verdict, records, bucket):
    """Candidate and existing medians at the bucket, from the verdict or one of its runs."""
    for b in (verdict.get("timing") or {}).get("buckets") or []:
        if b.get("m") == bucket and (b.get("fused_us") or {}).get("median"):
            base = b.get("baseline") or {}
            return {
                "candidate_us": b["fused_us"]["median"],
                "existing_us": base.get("median"),
                "baseline": base.get("impl"),
                "source": "verdict",
            }
    for run in verdict.get("runs") or []:
        p = Path(run)
        p = p if p.is_absolute() else Path(records) / p
        p = p / "result.json" if p.is_dir() else p
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = r.get("summary") or {}
        if "existing" in s and "candidate" in s and _bench_m(r) == bucket:
            return {
                "candidate_us": s["candidate"]["median_us"],
                "existing_us": s["existing"]["median_us"],
                "baseline": "existing",
                "source": str(p),
            }
    return None


def _prompt_kind(record):
    """ "human" or "sdk" for a prompt that starts a turn, None for anything else."""
    if record.get("type") != "user" or record.get("isMeta"):
        return None
    if record.get("isCompactSummary"):
        return None
    content = (record.get("message") or {}).get("content")
    blocks = content if isinstance(content, list) else [content]
    if not any(isinstance(b, str) or (b or {}).get("type") == "text" for b in blocks):
        return None
    if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks):
        return None
    origin = (record.get("origin") or {}).get("kind") or record.get("turnOrigin")
    return origin if origin in ("human", "sdk") else None


def session_stats(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.rglob("*.jsonl"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError("no transcripts")
    usage, models, tool_ids = {}, Counter(), set()
    prompts, first, stamps, cost = [], None, [], None
    for i, f in enumerate(files):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                if i == 0:
                    if r.get("timestamp"):
                        stamps.append(r["timestamp"])
                    kind = _prompt_kind(r)
                    if kind and first is None:
                        first = kind  # the kickoff, by the owner or by a driver
                    elif kind == "human":
                        prompts.append(r.get("timestamp"))
                    if (
                        r.get("type") == "cost-state"
                        and r.get("totalCostUSD") is not None
                    ):
                        cost = max(cost or 0.0, float(r["totalCostUSD"]))
                if r.get("type") != "assistant":
                    continue
                msg = r.get("message") or {}
                key = msg.get("id") or r.get("uuid")
                if key not in usage:
                    usage[key] = msg.get("usage") or {}
                    if msg.get("model") and msg["model"] != "<synthetic>":
                        models[msg["model"]] += 1
                for b in msg.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tool_ids.add(b.get("id"))
    tokens = {k: sum(int(u.get(k) or 0) for u in usage.values()) for k in TOKEN_KINDS}
    wall = None
    if len(stamps) >= 2:
        t = sorted(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps)
        wall = round((t[-1] - t[0]).total_seconds() / 3600, 2)
    return {
        "transcripts": [str(f) for f in files],
        "wall_hours": wall,
        "owner_prompts": len(prompts),
        "owner_prompt_times": prompts,
        "assistant_messages": len(usage),
        "tool_calls": len(tool_ids),
        "tokens": tokens,
        "cost_usd": None if cost is None else round(cost, 2),
        "models": dict(models.most_common()),
    }


def _commit(case):
    try:
        return subprocess.run(
            ["git", "-C", str(case), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(args):
    case, records = Path(args.case), Path(args.records)
    contract, contract_path = contract_of(case, records)
    targets = contract.get("targets") or {}
    bucket = ((contract.get("workload") or {}).get("target_buckets") or [None])[0]
    pct = targets.get("within_existing_pct")

    verdicts = sorted(
        Path(records).glob("verdicts/*.json"), key=lambda p: (p.stat().st_mtime, p.name)
    )
    verdict = json.loads(verdicts[-1].read_text(encoding="utf-8")) if verdicts else None
    timing = target_timing(verdict, records, bucket) if verdict else None
    gap = met = None
    if timing and timing["candidate_us"] and timing["existing_us"]:
        gap = 100.0 * (timing["candidate_us"] / timing["existing_us"] - 1.0)
        met = pct is not None and gap <= pct

    audit = (
        json.loads(Path(args.audit).read_text(encoding="utf-8")) if args.audit else None
    )
    audit_verdict = (audit or {}).get("verdict")
    if audit_verdict == "void":
        result = "void"
    elif verdict is None:
        result = "incomplete"
    elif audit is None:
        result = "unaudited"
    elif audit_verdict == "review" and not args.audit_reviewed:
        result = "pending-review"
    elif verdict.get("verdict") == "meets-contract" and met:
        result = "pass"
    else:
        result = "fail"

    gpu_hours, launches = gpu_time(records)
    budget = contract.get("budget") or {}
    return {
        "schema": SCHEMA,
        "case": case.name,
        "task": contract.get("task"),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "commit": args.commit or (audit or {}).get("base") or _commit(case),
        "contract": contract_path,
        "result": result,
        "target": {
            "bucket": bucket,
            "within_existing_pct": pct,
            "candidate_us": timing and timing["candidate_us"],
            "existing_us": timing and timing["existing_us"],
            "gap_pct": None if gap is None else round(gap, 2),
            "met": met,
            "source": timing and timing["source"],
        },
        "verdict": {
            "file": str(verdicts[-1]) if verdicts else None,
            "verdict": (verdict or {}).get("verdict"),
            "candidate": (verdict or {}).get("candidate"),
            "verdicts": len(verdicts),
        },
        "rounds": rounds_of(Path(records) / "decisions.md"),
        "dev_rows": sum(
            not r[0].isdigit()
            for r in decision_rows(Path(records) / "decisions.md") or []
        ),
        "gpu_hours": gpu_hours,
        "runpy_launches": launches,
        "node_hours": args.node_hours,
        "budget": {
            "rounds": budget.get("rounds"),
            "gpu_hours": budget.get("gpu_hours"),
        },
        "session": session_stats(args.transcript),
        "audit": (
            None
            if audit is None
            else {
                "verdict": audit_verdict,
                "counts": audit.get("counts"),
                "base": audit.get("base"),
                "reviewed": bool(args.audit_reviewed),
            }
        ),
    }


def _num(value, fmt):
    return "" if value is None else format(value, fmt)


def row(ev):
    t, s, a = ev["target"], ev["session"], ev["audit"] or {}
    models = ", ".join(s["models"]) or "?"
    counts = a.get("counts") or {}
    audit = a.get("verdict") or "none"
    if counts:
        audit += " (" + ", ".join(f"{v} {k}" for k, v in counts.items() if v) + ")"
    gap = "" if t["gap_pct"] is None else f"{t['gap_pct']:+.1f}%"
    return (
        f"| {ev['date']} | {(ev['commit'] or '?')[:10]} | {models} | {ev['result']} "
        f"| {_num(t['candidate_us'], '.1f')} | {_num(t['existing_us'], '.1f')} | {gap} "
        f"| {_num(ev['rounds'], 'd')} | {_num(ev['gpu_hours'], '.2f')} "
        f"| {s['owner_prompts']} | {_num(s['wall_hours'], '.1f')} "
        f"| {_num(s['cost_usd'], '.2f')} | {audit} |\n"
    )


def append_row(path, line):
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if HEADER.splitlines()[0] not in text:
        text = text.rstrip("\n") + ("\n\n" if text else "") + HEADER
    path.write_text(text.rstrip("\n") + "\n" + line, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", required=True, help="the case directory")
    p.add_argument("--records", required=True, help="the task's records directory")
    p.add_argument(
        "--transcript",
        action="append",
        required=True,
        help="the main session's .jsonl first, then its session directory (repeatable)",
    )
    p.add_argument("--audit", help="audit.py --json report")
    p.add_argument("--audit-reviewed", action="store_true")
    p.add_argument("--node-hours", type=float, help="reserved node time, if recorded")
    p.add_argument("--commit", help="the harness commit the session ran")
    p.add_argument("--out", help="default: <records>/evaluation.json")
    p.add_argument("--append", help="add the row to this results.md")
    args = p.parse_args(argv)
    try:
        ev = evaluate(args)
    except (OSError, ValueError, KeyError) as e:
        print(f"evaluate.py: {e}", file=sys.stderr)
        return 2
    out = Path(args.out or Path(args.records) / "evaluation.json")
    out.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    line = row(ev)
    if args.append:
        append_row(args.append, line)
    print(HEADER + line, end="")
    print(f"evaluate: {ev['result']} -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
