#!/usr/bin/env python3
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
"""Per-phase statistics from the traces bench_ep PROFILE=1 writes.

    python3 trace_stats.py mori_traces/trace_hip_ct512_rank*.json

One row per phase: how long a warp spends in it, over every warp in every file
given. The interesting column is usually p95 rather than mean -- a phase that
serializes across GPUs is cheap for the warp that wins and expensive for the one
that loses, and only the tail shows that.

"share" is of the summed per-warp span, so the rows add to 100%: it says where a
warp's time goes, NOT what fraction of the kernel's wall clock a phase owns
(warps overlap, so those are different questions).
"""
import argparse
import json
import sys
from collections import defaultdict


def pair_events(path):
    """-> {slot: [(round, duration us)]}, {warp: span us}.

    B/E per (warp, slot) in ts order. A warp meets each phase once per traced round,
    so the k-th time it closes a phase is round k -- which is what lets the caller
    ask whether round 0 differs from the rest rather than averaging that away.
    """
    with open(path) as f:
        doc = json.load(f)
    events = doc["traceEvents"] if isinstance(doc, dict) else doc
    events = [e for e in events if e.get("ph") in ("B", "E")]
    events.sort(key=lambda e: e["ts"])

    open_at = defaultdict(list)
    seen = defaultdict(int)
    durs = defaultdict(list)
    first_last = {}
    for e in events:
        key = (e["tid"], e["name"])
        if e["ph"] == "B":
            open_at[key].append(e["ts"])
        else:
            if not open_at[key]:
                continue  # an END whose BEGIN was dropped by the ring wrapping
            durs[e["name"]].append((seen[key], e["ts"] - open_at[key].pop()))
            seen[key] += 1
        lo, hi = first_last.get(e["tid"], (e["ts"], e["ts"]))
        first_last[e["tid"]] = (min(lo, e["ts"]), max(hi, e["ts"]))
    spans = {w: hi - lo for w, (lo, hi) in first_last.items()}
    return durs, spans


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("traces", nargs="+", help="trace_*.json from bench_ep PROFILE=1")
    ap.add_argument(
        "--order",
        default="",
        help="comma-separated phase names to print in this order; "
        "default is by descending mean",
    )
    ap.add_argument(
        "--by-round",
        action="store_true",
        help="also print each phase's mean per traced round, which is how a warm-up "
        "artefact shows itself: round 0 out of line with the rest",
    )
    a = ap.parse_args()

    rounds = defaultdict(lambda: defaultdict(list))  # phase -> round -> durations
    durs = defaultdict(list)
    spans = {}
    for i, p in enumerate(a.traces):
        d, s = pair_events(p)
        for k, v in d.items():
            for rnd, dur in v:
                durs[k].append(dur)
                rounds[k][rnd].append(dur)
        for w, v in s.items():
            spans[(i, w)] = v  # warp ids repeat across ranks; file index separates them

    if not durs:
        sys.exit("no paired B/E events found")

    total = sum(sum(v) for v in durs.values())
    names = (
        [n for n in a.order.split(",") if n in durs]
        if a.order
        else sorted(durs, key=lambda n: -sum(durs[n]) / len(durs[n]))
    )

    warps = len(spans)
    span_mean = sum(spans.values()) / warps if warps else 0.0
    print(
        f"{len(a.traces)} file(s), {warps} warps, "
        f"per-warp span mean {span_mean:.2f} us, p95 {pct(list(spans.values()), 0.95):.2f} us"
    )
    print(
        f"{'phase':14s} {'n':>6s} {'mean':>8s} {'p50':>8s} {'p95':>8s} "
        f"{'max':>8s} {'share':>7s}"
    )
    for n in names:
        v = durs[n]
        print(
            f"{n:14s} {len(v):6d} {sum(v)/len(v):8.3f} {pct(v,0.5):8.3f} "
            f"{pct(v,0.95):8.3f} {max(v):8.3f} {100*sum(v)/total:6.1f}%"
        )

    if a.by_round:
        n_rounds = max((max(r) for r in rounds.values() if r), default=-1) + 1
        print(f"\nmean us per round (0 = first traced launch), {n_rounds} rounds")
        print(
            f"{'phase':14s} " + " ".join(f"{'r' + str(k):>8s}" for k in range(n_rounds))
        )
        for n in names:
            cells = []
            for k in range(n_rounds):
                v = rounds[n].get(k, [])
                cells.append(f"{sum(v)/len(v):8.3f}" if v else f"{'-':>8s}")
            print(f"{n:14s} " + " ".join(cells))


if __name__ == "__main__":
    main()
