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
"""Phase timeline of a fused operator, from the device timestamps it records.

    python benchmark/cco/harness/timeline.py <timeline.json> [--json <out>]

``op_harness.py --mode timeline`` writes the input: for every rank, for every call, the
events the implementation stamped, each ``[name, us]`` counted from that call's first
stamp on that rank. Clocks are per GPU, so ranks are compared by offsets, never by
absolute times.

Event names:

* ``phase:start`` and ``phase:end`` bound an interval;
* any other name is an instant;
* ``name@key`` repeats an event per key (a GEMM block, a destination and chunk): the
  part before ``@`` is the group.

Per call, an interval group gives its first start, last end, span (last end minus first
start) and the median of its keys' durations; an instant group gives its first, median
and last time and its count. Each value is the median over calls on a rank; ranks are
then summarized as min, median and max. The critical rank is the one whose calls end
last, and the table follows the order of its groups' first times. A gap row is shown
where nothing in any group runs on the critical rank.

The device side is the implementation's: ``s_memrealtime`` (a 100 MHz clock) stored by
one thread into a u64 slot, mapped to names by ``timeline_read()``; see the loop skill's
``reference/diagnosis.md``.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

INTERVAL_FIELDS = ("first", "last", "span", "dur")
INSTANT_FIELDS = ("first", "median", "last")


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def groups_of_call(events):
    """{group: summary} for one call's ``[name, us]`` events on one rank."""
    raw = defaultdict(lambda: {"starts": {}, "ends": {}, "times": []})
    for name, us in events:
        base, _, key = str(name).partition("@")
        if base.endswith(":start"):
            raw[base[: -len(":start")]]["starts"][key] = us
        elif base.endswith(":end"):
            raw[base[: -len(":end")]]["ends"][key] = us
        else:
            raw[base]["times"].append(us)
    out = {}
    for group, d in raw.items():
        if d["starts"] or d["ends"]:
            both = set(d["starts"]) & set(d["ends"])
            first = min(d["starts"].values()) if d["starts"] else None
            last = max(d["ends"].values()) if d["ends"] else None
            out[group] = {
                "kind": "interval",
                "count": max(len(d["starts"]), len(d["ends"])),
                "first": first,
                "last": last,
                "span": None if first is None or last is None else last - first,
                "dur": _median(d["ends"][k] - d["starts"][k] for k in both),
            }
        else:
            times = sorted(d["times"])
            out[group] = {
                "kind": "instant",
                "count": len(times),
                "first": times[0],
                "median": _median(times),
                "last": times[-1],
            }
    return out


def rank_summary(calls):
    """Median over calls of every group's numbers, and of the call's end, on one rank."""
    parsed = [groups_of_call(c) for c in calls if c]
    groups = {}
    for group in sorted({g for p in parsed for g in p}):
        items = [p[group] for p in parsed if group in p]
        kind = items[0]["kind"]
        fields = INTERVAL_FIELDS if kind == "interval" else INSTANT_FIELDS
        groups[group] = {
            "kind": kind,
            "count": _median(i["count"] for i in items),
            **{f: _median(i.get(f) for i in items) for f in fields},
        }
    end = _median(max(us for _, us in c) for c in calls if c)
    return {"end": end, "groups": groups}


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def summarize(data):
    """The phase table for a timeline file's contents."""
    ranks = [rank_summary(calls) for calls in data["ranks"]]
    live = [i for i, r in enumerate(ranks) if r["end"] is not None]
    if not live:
        return {
            "ranks": len(ranks),
            "critical_rank": None,
            "call_us": None,
            "phases": [],
        }
    critical = max(live, key=lambda i: ranks[i]["end"])
    mine = ranks[critical]["groups"]
    order = sorted(mine, key=lambda g: (mine[g]["first"] is None, mine[g]["first"]))
    phases, busy_until = [], None
    for group in order:
        row = {"group": group, **mine[group]}
        fields = INTERVAL_FIELDS if row["kind"] == "interval" else INSTANT_FIELDS
        row["ranks"] = {
            f: _spread(r["groups"].get(group, {}).get(f) for r in ranks) for f in fields
        }
        start = row["first"]
        if busy_until is not None and start is not None and start > busy_until:
            phases.append(
                {"group": "(gap)", "kind": "gap", "first": busy_until, "last": start}
            )
        end = row["last"]
        if end is not None:
            busy_until = end if busy_until is None else max(busy_until, end)
        phases.append(row)
    return {
        "ranks": len(ranks),
        "calls": max((len(c) for c in data["ranks"]), default=0),
        "critical_rank": critical,
        "call_us": _spread(r["end"] for r in ranks),
        "phases": phases,
    }


def _num(v):
    return "" if v is None else f"{v:.1f}"


def render(summary):
    lines = [
        f"critical rank {summary['critical_rank']}; call "
        + (
            "{min:.1f} / {median:.1f} / {max:.1f} us (min / median / max over ranks)".format(
                **summary["call_us"]
            )
            if summary["call_us"]
            else "?"
        ),
        f"{'group':28} {'kind':8} {'first':>9} {'last':>9} {'span':>9} {'dur':>8} "
        f"{'count':>6}  last over ranks",
    ]
    for p in summary["phases"]:
        if p["kind"] == "gap":
            lines.append(
                f"{'  (nothing stamped)':28} {'gap':8} {_num(p['first']):>9} "
                f"{_num(p['last']):>9} {_num(p['last'] - p['first']):>9}"
            )
            continue
        last = (p.get("ranks") or {}).get("last") or {}
        across = (
            f"{last['min']:.1f}-{last['max']:.1f}"
            if last and last.get("min") is not None
            else ""
        )
        if p["kind"] == "interval":
            cols = (p["first"], p["last"], p["span"], p["dur"])
        else:
            cols = (p["first"], p["last"], None, None)
        lines.append(
            f"{p['group']:28} {p['kind']:8} "
            + " ".join(f"{_num(c):>{w}}" for c, w in zip(cols, (9, 9, 9, 8)))
            + f" {_num(p['count']):>6}  {across}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("timeline", help="the file op_harness.py --mode timeline wrote")
    parser.add_argument("--json", help="write the summary here")
    args = parser.parse_args(argv)
    try:
        data = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
        summary = summarize(data)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"timeline.py: {e}", file=sys.stderr)
        return 2
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
