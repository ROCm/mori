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
"""Aggregate MORI-VIZ kernel traces across all EP ranks.

analyze_ep_kernel_trace.py answers "what did rank N's timeline look like".
This answers "where does the time go, across the whole job" -- for each
instrumented phase, the wall-clock duration seen on every rank/iteration, plus
what fraction of the dispatch and combine windows it accounts for.

Usage: aggregate_ep_kernel_trace.py 'traces/trace_rank_*.json' [--per-rank]
"""

import argparse
import glob
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ep_kernel_trace import _merge_spans, parse_events  # noqa: E402

# The v1_ll launch order, from src/ops/dispatch_combine/launch.cpp:485-489
# (dispatch: 2 kernels) and :614-621 (combine: 4 kernels).
DISPATCH_PHASES = [
    "ep_dispatch_copy_to_staging",
    "dispatch_inter_node_ll_send",
    "dispatch_inter_node_ll_recv",
    "dispatch_intra",
    "dispatch_sync",
]
COMBINE_PHASES = [
    "combine_sync",
    "ep_combine_sync_barrier",
    "combine_inter_node_ll",
    "combine_intra_node_ll",
    "ep_combine_all",
]
PHASE_ORDER = DISPATCH_PHASES + COMBINE_PHASES

# One iteration is anchored on the first and last kernel of each half, so the
# windows line up with what the torch.cuda.Event timers in run_bench_once
# measure: dispatch = copy_to_staging .. end of the fused LL dispatch kernel,
# combine = EpCombineSync .. end of EpCombineAll.
DISPATCH_ANCHOR = "ep_dispatch_copy_to_staging"
COMBINE_ANCHOR = "combine_sync"
COMBINE_END_ANCHOR = "ep_combine_all"


def collect(path):
    """Yield one dict per iteration in this rank's trace."""
    intervals = parse_events(path)
    by_phase = defaultdict(list)
    for name, ts, te, _ in intervals:
        by_phase[name].append((ts, te))

    disp_starts = [s for s, _ in _merge_spans(by_phase[DISPATCH_ANCHOR])]
    comb_starts = [s for s, _ in _merge_spans(by_phase[COMBINE_ANCHOR])]
    comb_ends = [e for _, e in _merge_spans(by_phase[COMBINE_END_ANCHOR])]
    n = min(len(disp_starts), len(comb_starts), len(comb_ends))

    # Assign every span to the iteration it *starts* in. Overlap-based binning
    # double-counts: a warp still spinning in dispatch_sync when the next
    # iteration's copy_to_staging begins would otherwise land in both.
    bins = [defaultdict(list) for _ in range(n)]
    bounds = disp_starts[:n] + [float("inf")]
    for name, ts, te, _ in intervals:
        for i in range(n):
            if bounds[i] <= ts < bounds[i + 1]:
                bins[i][name].append((ts, te))
                break

    for i in range(n):
        merged = {k: _merge_spans(v) for k, v in bins[i].items()}
        # Union of the per-warp spans: the wall-clock time during which any warp
        # was inside this phase. Summing the disjoint pieces rather than taking
        # last_end - first_start avoids charging a phase for gaps in which every
        # warp had already moved on.
        durs = {k: sum(e - s for s, e in spans) for k, spans in merged.items()}
        disp_end = max(
            (e for p in DISPATCH_PHASES if p in merged for _, e in merged[p]),
            default=None,
        )
        yield {
            "durs": durs,
            "disp_us": disp_end - disp_starts[i] if disp_end else None,
            "comb_us": comb_ends[i] - comb_starts[i],
            "gap_us": comb_starts[i] - disp_end if disp_end else None,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", help="glob matching trace_rank_*.json")
    ap.add_argument(
        "--skip",
        type=int,
        default=1,
        help="drop the first N iterations per rank (still warming up)",
    )
    ap.add_argument(
        "--per-rank",
        action="store_true",
        help="also print each rank's mean dispatch/combine window",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"no traces matched {args.pattern!r}")

    per_phase = defaultdict(list)
    disp_totals, comb_totals, gaps = [], [], []
    by_rank = []
    n_iters = 0
    for path in files:
        its = list(collect(path))[args.skip :]
        m = re.search(r"rank_(\d+)_", Path(path).name)
        by_rank.append(
            (
                int(m.group(1)) if m else -1,
                statistics.mean(i["disp_us"] for i in its),
                statistics.mean(i["comb_us"] for i in its),
            )
        )
        for it in its:
            n_iters += 1
            for k, v in it["durs"].items():
                per_phase[k].append(v)
            if it["disp_us"] is not None:
                disp_totals.append(it["disp_us"])
                gaps.append(it["gap_us"])
            comb_totals.append(it["comb_us"])

    print(
        f"{len(files)} rank traces, {n_iters} rank-iterations "
        f"(first {args.skip} per rank dropped)\n"
    )
    disp_mean = statistics.mean(disp_totals)
    comb_mean = statistics.mean(comb_totals)
    print(
        f"  dispatch window (kernel-side): {disp_mean:7.2f} us   "
        f"[{min(disp_totals):.1f} .. {max(disp_totals):.1f}]"
    )
    print(
        f"  combine  window (kernel-side): {comb_mean:7.2f} us   "
        f"[{min(comb_totals):.1f} .. {max(comb_totals):.1f}]"
    )
    print(
        f"  gap between the two (host-side convert + launch): "
        f"{statistics.mean(gaps):.2f} us"
    )
    print()

    hdr = f"{'phase':<32}{'mean':>9}{'min':>9}{'max':>9}{'p90':>9}{'share':>8}  n"
    print(hdr)
    print("-" * len(hdr))
    known = [p for p in PHASE_ORDER if p in per_phase]
    rest = sorted(k for k in per_phase if k not in PHASE_ORDER)
    for phase in known + rest:
        d = sorted(per_phase[phase])
        base = disp_mean if phase in DISPATCH_PHASES else comb_mean
        p90 = d[min(len(d) - 1, int(0.9 * len(d)))]
        if phase == COMBINE_ANCHOR:
            print("-" * len(hdr))
        print(
            f"{phase:<32}{statistics.mean(d):9.2f}{d[0]:9.2f}{d[-1]:9.2f}"
            f"{p90:9.2f}{statistics.mean(d) / base * 100:7.1f}%  {len(d)}"
        )
    print("-" * len(hdr))
    print("share = phase mean / (dispatch or combine) window mean; phases that")
    print("run concurrently on different blocks make the shares sum past 100%.")

    if args.per_rank:
        print(f"\n{'rank':>4}{'dispatch':>10}{'combine':>10}{'sum':>10}")
        print("-" * 34)
        for r, d, c in sorted(by_rank):
            print(f"{r:>4}{d:10.2f}{c:10.2f}{d + c:10.2f}")


if __name__ == "__main__":
    main()
