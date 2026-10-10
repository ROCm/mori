#!/usr/bin/env bash
# ==========================================================================
# EP AsyncLL comparison benchmark (single node, EP8 by default)
#
# Runs ONE token sweep across four implementations of the same dispatch +
# combine and emits one JSON record plus a human-readable table:
#
#   v2_asyncll_sdma  EPv2 AsyncLL on cco, transfers over ccoSdma (the default)
#   v2_asyncll_lsa   EPv2 AsyncLL on cco, transfers as peer stores over LSA
#   v2_native        EPv2's ordinary intranode kernels -- no staging, no
#                    send/recv split. The floor: what the hardware does when
#                    the protocol is not carrying an RDMA leg's shape.
#   v1_asyncll       EPv1 AsyncLL on mori-shmem. The thing v2 had to beat.
#
# Why this exists: the four were measured piecemeal on a shared box while the
# AsyncLL port was being written, and the open question could not be settled
# there. At 64-128 tokens per rank the transfer is a small part of the time, so
# sdma and lsa land within a few microseconds of each other. At 1024+ the SDMA
# engine saturates (~60 GB/s per link, measured) while peer stores do not, so
# lsa is EXPECTED to pull ahead -- and nothing has measured it. The sweep here
# covers both regimes on purpose.
#
# Usage (on an idle node, inside the mori container):
#   bash tools/bench_ep_asyncll_compare.sh
#   bash tools/bench_ep_asyncll_compare.sh --tokens 64,512,4096
#   bash tools/bench_ep_asyncll_compare.sh --impls v2_asyncll_sdma,v2_asyncll_lsa
#   bash tools/bench_ep_asyncll_compare.sh --hidden 4096 --topk 6 --epr 48
#   bash tools/bench_ep_asyncll_compare.sh --dry-run        # preflight only, no GPU work
#   bash tools/bench_ep_asyncll_compare.sh --force          # skip the idle check
#
# Output:
#   <output-dir>/
#     raw/<impl>[_<tokens>].txt   full stdout of every run
#     env.txt                     provenance: GPU, ROCm, kernel, known issues
#     results.json                one row per (impl, tokens)
#     summary.txt                 the table printed at the end
#     bench.log                   run log
#
# NOTE ON COMPARABILITY. The two harnesses report bandwidth with numerators
# that are close but not identical, so LATENCY is the number to compare across
# implementations and bandwidth is best read WITHIN one. Both are recorded.
#   * bench_ep.py (v2):  recv_tokens * hidden * elem_size / time, per rank.
#   * the v1 harness:    its "XGMI Bandwidth" column, averaged over ranks.
# Recomputing the v1 column by hand lands within ~3% of the v2 definition at
# the same point, which is why they are tabulated side by side at all.
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
V2_BENCH="$REPO_ROOT/tests/python/ops/dispatch_combine_v2/bench_ep.py"
V1_BENCH="$REPO_ROOT/examples/ops/dispatch_combine/test_dispatch_combine_internode.py"

# ---- Defaults ----
WORLD_SIZE=8
# Spans both regimes: latency/staging-bound at the low end, SDMA-bandwidth-bound
# at the high end. 2048 is where the sdma transport was last seen still climbing.
TOKENS="64,128,256,512,1024,2048"
IMPLS="v2_asyncll_sdma,v2_asyncll_lsa,v2_native,v1_asyncll"
HIDDEN=7168
TOPK=8
EPR=32
ITERS=50
OUTPUT_DIR=""
TIMEOUT=1800
FORCE=0
DRY_RUN=0
PYTHON="${PYTHON:-python3}"
TORCHRUN="${TORCHRUN:-torchrun}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)       TOKENS="$2";      shift 2 ;;
        --impls)        IMPLS="$2";       shift 2 ;;
        --world-size)   WORLD_SIZE="$2";  shift 2 ;;
        --hidden)       HIDDEN="$2";      shift 2 ;;
        --topk)         TOPK="$2";        shift 2 ;;
        --epr)          EPR="$2";         shift 2 ;;
        --iters)        ITERS="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --timeout)      TIMEOUT="$2";     shift 2 ;;
        --force)        FORCE=1;          shift 1 ;;
        --dry-run)      DRY_RUN=1;        shift 1 ;;
        -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    # bench_results/ is not in .gitignore, so results land outside the tree by
    # default -- a benchmark should not leave the working copy dirty.
    OUTPUT_DIR="${MORI_BENCH_OUTPUT_ROOT:-/tmp/mori_bench}/asyncll_ep${WORLD_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR/raw"
LOG="$OUTPUT_DIR/bench.log"
ENVFILE="$OUTPUT_DIR/env.txt"
SUMMARY="$OUTPUT_DIR/summary.txt"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT:${PYTHONPATH:-}"
# Required, not optional: without it cco creates no SDMA queues and every
# ccoSdma put is a silent no-op. The op's own gate rejects that config, but set
# it here so a caller who forgot does not get a confusing refusal instead of a
# benchmark.
export MORI_ENABLE_SDMA=1
export GPU_PER_NODE="$WORLD_SIZE"
export HSA_NO_SCRATCH_RECLAIM=1

# ======================================================================
# Preflight
# ======================================================================

# A benchmark that shares the GPUs with someone else produces numbers nobody can
# use, and the sharing is invisible after the fact. This is the single most
# valuable check in the script.
if [[ "$FORCE" -eq 0 ]]; then
    if command -v rocm-smi >/dev/null 2>&1; then
        PIDS=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || true)
        if [[ "${PIDS:-0}" -gt 0 ]]; then
            echo "ERROR: other processes are using the GPUs:" >&2
            rocm-smi --showpids >&2
            echo "Benchmark results would be meaningless. Re-run with --force to override." >&2
            exit 1
        fi
    else
        echo "WARNING: rocm-smi not found; cannot confirm the node is idle." | tee -a "$LOG"
    fi
fi

if [[ ! -f "$V2_BENCH" ]]; then echo "missing $V2_BENCH" >&2; exit 1; fi
if [[ ! -f "$V1_BENCH" ]]; then echo "missing $V1_BENCH" >&2; exit 1; fi

# Fail now, with a readable message, rather than inside the first torchrun.
"$PYTHON" - <<'PY' || { echo "ERROR: mori is not importable; build it first." >&2; exit 1; }
import sys
try:
    import torch  # noqa: F401
    import mori  # noqa: F401
    from mori.jit.v2 import plan_api
    plan_api.load_library("libmori_ops_v2.so")
except Exception as exc:
    print(f"import check failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY

# ---- Provenance -------------------------------------------------------
# Recorded because two numbers from two machines are not comparable without it,
# and because two of the three known environment issues change these results
# materially. See the known-issues skill.
{
    echo "# collected $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host:            $(hostname)"
    echo "kernel:          $(uname -r)"
    echo "rocm:            $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
    echo "git_commit:      $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty:       $(test -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo ""
    echo "## GPU"
    "$PYTHON" -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'model:           {p.name}')
print(f'arch:            {getattr(p, \"gcnArchName\", \"?\")}')
print(f'cu_count:        {p.multi_processor_count}')
print(f'visible_devices: {torch.cuda.device_count()}')
" 2>/dev/null || echo "(torch device query failed)"
    echo ""
    echo "## known-issues checks"
    # Issue 1: VMM peer traffic pinned to host memory. Silent, and worth ~9x on
    # every cco number, so a run on an affected node is simply void.
    CFG="/boot/config-$(uname -r)"
    PROBE_SRC="$REPO_ROOT/.claude/skills/known-issues/vmm_peer_probe.cpp"
    if [[ -r "$CFG" ]]; then
        grep -E 'CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY)=' "$CFG" || \
            echo "issue1: CONFIG_DMABUF_MOVE_NOTIFY / PCI_P2PDMA NOT SET -- cco numbers are void"
    elif [[ -r "$PROBE_SRC" ]] && command -v hipcc >/dev/null 2>&1; then
        # /boot is usually not mounted in the container, and this is too important
        # to leave unanswered: on an affected node every cco number here is ~9x
        # low and nothing says so. The probe allocates VMM memory on one GPU,
        # grants peer access from another and reports whether it stayed in VRAM.
        echo "issue1: /boot not readable, running vmm_peer_probe instead"
        PROBE_DIR="$(mktemp -d)"
        if hipcc -O3 --offload-arch="$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*')" \
                "$PROBE_SRC" -o "$PROBE_DIR/probe" >/dev/null 2>&1; then
            "$PROBE_DIR/probe" 2>&1 | sed 's/^/  /' || echo "  probe failed to run"
        else
            echo "  probe failed to build"
        fi
        rm -rf "$PROBE_DIR"
    else
        echo "issue1: UNDETERMINED -- $CFG unreadable and no probe available."
        echo "        An affected node reports ~1/9 of the real cco bandwidth, silently."
    fi
    # Issue 3: on ROCm 7.2.x the VMM path ignores the uncached request, costing
    # cco combine ~10-15us. The tell is that forcing cached changes nothing.
    echo "issue3: ROCm $(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1-2) -- 7.0-7.2.x is affected, 7.14+ is not"
} > "$ENVFILE" 2>&1

{
    echo "============================================================"
    echo "EP${WORLD_SIZE} AsyncLL comparison"
    echo "============================================================"
    cat "$ENVFILE"
    echo ""
    echo "  tokens:     $TOKENS"
    echo "  impls:      $IMPLS"
    echo "  shape:      hidden=$HIDDEN topk=$TOPK epr=$EPR dtype=bf16"
    echo "  iters:      $ITERS (CUDA graph)"
    echo "  output:     $OUTPUT_DIR"
    echo "============================================================"
} | tee "$LOG"

# ======================================================================
# Runners
# ======================================================================
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS"
IFS=',' read -ra IMPL_ARRAY <<< "$IMPLS"

# Heap for the v1 harness, which allocates its symmetric buffers up front.
v1_heap() {
    local t=$1
    if   (( t >= 16384 )); then echo "48G"
    elif (( t >= 4096  )); then echo "24G"
    else echo "8G"
    fi
}

# The three v2 variants differ only in two config fields, and bench_ep.py sweeps
# every token count inside ONE process -- so each is a single run.
run_v2() {
    local impl=$1 async_ll=$2 transport=$3
    local raw="$OUTPUT_DIR/raw/${impl}.txt"
    echo "[$impl] sweep tokens=$TOKENS  $(date +%H:%M:%S)" | tee -a "$LOG"
    set +e
    MORI_V2_KERNEL_BACKEND=hip \
    BACKENDS=hip \
    ASYNC_LL="$async_ll" \
    ASYNC_LL_TRANSPORT="$transport" \
    DISP=bf16 QUANT=none COMB_MODE=pull COMBINE_IN=inplace CHECK=1 \
    HIDDEN="$HIDDEN" TOPK="$TOPK" EPR="$EPR" \
    SWEEP="$TOKENS" MODES=graph ITERS="$ITERS" \
    timeout "$TIMEOUT" "$TORCHRUN" --standalone --nproc_per_node="$WORLD_SIZE" \
        "$V2_BENCH" > "$raw" 2>&1
    local rc=$?
    set -e
    grep -E "^  ct=|^# (PASS|FAIL)" "$raw" | tee -a "$LOG" || true
    [[ $rc -ne 0 ]] && echo "  !!! $impl exited $rc !!!" | tee -a "$LOG"
    return 0
}

# v1's harness benches one token count per process, so this one loops.
run_v1() {
    local impl=v1_asyncll
    for t in "${TOKEN_ARRAY[@]}"; do
        local raw="$OUTPUT_DIR/raw/${impl}_${t}.txt"
        echo "[$impl] tokens=$t  $(date +%H:%M:%S)" | tee -a "$LOG"
        set +e
        MORI_SHMEM_HEAP_SIZE="$(v1_heap "$t")" \
        timeout "$TIMEOUT" "$TORCHRUN" \
            --nnodes=1 --node_rank=0 --nproc_per_node=1 \
            --master_addr=127.0.0.1 --master_port=$((29500 + RANDOM % 1000)) \
            "$V1_BENCH" \
            --kernel-type async_ll --cmd bench --dtype bf16 --num-qp 2 \
            --hidden-dim "$HIDDEN" --topk "$TOPK" \
            --max-tokens "$t" > "$raw" 2>&1
        local rc=$?
        set -e
        grep -E "^  E2E " "$raw" | tee -a "$LOG" || true
        [[ $rc -ne 0 ]] && echo "  !!! $impl tokens=$t exited $rc !!!" | tee -a "$LOG"
    done
    return 0
}

# --dry-run stops here: preflight, provenance and the plan, but no GPU work. It
# is what to run first on a borrowed node, where finding out that an argument was
# wrong after forty minutes of benchmarking is the expensive failure.
if [[ "$DRY_RUN" -eq 1 ]]; then
    {
        echo ""
        echo "DRY RUN -- would execute:"
        for impl in "${IMPL_ARRAY[@]}"; do
            case "$impl" in
                v2_asyncll_sdma|v2_asyncll_lsa|v2_native)
                    echo "  [$impl] 1 process, SWEEP=$TOKENS"
                    echo "      $TORCHRUN --standalone --nproc_per_node=$WORLD_SIZE $V2_BENCH" ;;
                v1_asyncll)
                    echo "  [$impl] ${#TOKEN_ARRAY[@]} processes, one per token count"
                    echo "      $TORCHRUN --nnodes=1 --nproc_per_node=1 $V1_BENCH --cmd bench --kernel-type async_ll ..." ;;
                *) echo "Unknown impl: $impl" >&2; exit 1 ;;
            esac
        done
        echo ""
        echo "Preflight passed. Drop --dry-run to collect."
    } | tee -a "$LOG"
    exit 0
fi

for impl in "${IMPL_ARRAY[@]}"; do
    case "$impl" in
        v2_asyncll_sdma) run_v2 "$impl" 1 sdma ;;
        v2_asyncll_lsa)  run_v2 "$impl" 1 lsa  ;;
        v2_native)       run_v2 "$impl" 0 sdma ;;   # async_ll=0 ignores transport
        v1_asyncll)      run_v1 ;;
        *) echo "Unknown impl: $impl" >&2; exit 1 ;;
    esac
done

# ======================================================================
# Parse -> results.json + summary.txt
# ======================================================================
"$PYTHON" - "$OUTPUT_DIR" "$HIDDEN" "$TOPK" "$EPR" "$WORLD_SIZE" <<'PY'
"""Collect both harnesses' output into one record.

bench_ep.py already prints a JSON summary line, so the v2 variants are a load.
The v1 harness prints a prettytable instead, so its Average row and its E2E line
are scraped -- deliberately the AVERAGE row, because the JSON that harness
records internally mixes best-bandwidth with average-latency and the two cannot
be put in one column honestly.
"""
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
hidden, topk, epr, world = (int(x) for x in sys.argv[2:6])
raw = out / "raw"
rows = []

# ---- v2 variants: one JSON line per run, one entry per token count ----
for path in sorted(raw.glob("v2_*.txt")):
    impl = path.stem
    for line in path.read_text(errors="replace").splitlines():
        if '"mori ep dispatch_combine_v2 summary"' not in line:
            continue
        for r in json.loads(line.strip()).get("rows", []):
            if "err_msg" in r:
                rows.append({"impl": impl, "tokens": r.get("m"), "error": r["err_msg"]})
                continue
            rows.append(
                {
                    "impl": impl,
                    "tokens": r["m"],
                    "dispatch_us": r["dispatch_us"],
                    "combine_us": r["combine_us"],
                    "pair_us": r["pair_us"],
                    "dispatch_gbps": r["dispatch_gbps"],
                    "combine_gbps": r["combine_gbps"],
                    "recv_tokens": r["recv_tokens"],
                    "bw_definition": "recv_tokens*hidden*elem/time (bench_ep.py)",
                }
            )

# ---- v1: scrape the table ----
# Section header, then the Average row of that section. The columns are
# RDMA | XGMI | LL | Latency; XGMI is the one whose numerator matches the v2
# definition closely enough to tabulate beside it.
_SECTION = re.compile(r"\|\s+(Dispatch|Combine) Performance")
_AVERAGE = re.compile(
    r"\|\s*Average\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)
_E2E = re.compile(r"E2E \(dispatch\+combine, CUDA graph x\d+\):\s*([\d.]+) us")

for path in sorted(raw.glob("v1_asyncll_*.txt")):
    tokens = int(path.stem.rsplit("_", 1)[1])
    text = path.read_text(errors="replace")
    section, found = None, {}
    for line in text.splitlines():
        m = _SECTION.search(line)
        if m:
            section = m.group(1).lower()
            continue
        if section:
            a = _AVERAGE.search(line)
            if a:
                found[section] = {"xgmi_gbps": float(a.group(2)), "us": float(a.group(4))}
                section = None
    e2e = _E2E.search(text)
    if not found:
        rows.append({"impl": "v1_asyncll", "tokens": tokens, "error": "no table parsed"})
        continue
    rows.append(
        {
            "impl": "v1_asyncll",
            "tokens": tokens,
            "dispatch_us": found.get("dispatch", {}).get("us"),
            "combine_us": found.get("combine", {}).get("us"),
            "pair_us": float(e2e.group(1)) if e2e else None,
            "dispatch_gbps": found.get("dispatch", {}).get("xgmi_gbps"),
            "combine_gbps": found.get("combine", {}).get("xgmi_gbps"),
            "recv_tokens": None,
            "bw_definition": "XGMI column, rank-average (v1 harness)",
        }
    )

env = (out / "env.txt").read_text(errors="replace")
record = {
    "name": "mori ep asyncll comparison",
    "shape": {
        "world_size": world,
        "hidden": hidden,
        "topk": topk,
        "experts_per_rank": epr,
        "dtype": "bf16",
    },
    "env": env,
    "rows": rows,
}
(out / "results.json").write_text(json.dumps(record, indent=2))

# ---- Human table ----
impls = [i for i in ("v2_asyncll_sdma", "v2_asyncll_lsa", "v2_native", "v1_asyncll")
         if any(r["impl"] == i for r in rows)]
tokens = sorted({r["tokens"] for r in rows if r.get("tokens") is not None})
by = {(r["impl"], r.get("tokens")): r for r in rows}

lines = []
lines.append(f"# EP{world} hidden={hidden} topk={topk} epr={epr} bf16, CUDA graph")
lines.append("# pair_us = dispatch + combine. Compare LATENCY across impls;")
lines.append("# bandwidth definitions differ slightly between the harnesses.")
lines.append("")
head = f"{'tokens':>7} " + "".join(f"{i:>22}" for i in impls)
lines.append(head)
lines.append("-" * len(head))
for t in tokens:
    cells = []
    for i in impls:
        r = by.get((i, t))
        if r is None:
            cells.append(f"{'-':>22}")
        elif "error" in r:
            cells.append(f"{'FAIL':>22}")
        else:
            cells.append(f"{r['pair_us']:>14.1f} us".rjust(22))
    lines.append(f"{t:>7} " + "".join(cells))
lines.append("")
lines.append("per-rank payload bandwidth, dispatch / combine (GB/s)")
lines.append(head)
lines.append("-" * len(head))
for t in tokens:
    cells = []
    for i in impls:
        r = by.get((i, t))
        if r is None or "error" in r or r.get("dispatch_gbps") is None:
            cells.append(f"{'-':>22}")
        else:
            cells.append(f"{r['dispatch_gbps']:>9.1f} /{r['combine_gbps']:>8.1f}".rjust(22))
    lines.append(f"{t:>7} " + "".join(cells))

# The question this sweep exists to answer.
sd = {t: by.get(("v2_asyncll_sdma", t)) for t in tokens}
ls = {t: by.get(("v2_asyncll_lsa", t)) for t in tokens}
cross = [
    (t, sd[t]["pair_us"], ls[t]["pair_us"])
    for t in tokens
    if sd.get(t) and ls.get(t) and "error" not in sd[t] and "error" not in ls[t]
]
if cross:
    lines.append("")
    lines.append("sdma vs lsa (negative = lsa faster):")
    for t, a, b in cross:
        lines.append(f"  {t:>6} tok: {b - a:+7.1f} us  ({100 * (b - a) / a:+.1f}%)")

text = "\n".join(lines)
(out / "summary.txt").write_text(text + "\n")
print()
print(text)
PY

echo "" | tee -a "$LOG"
echo "Results: $OUTPUT_DIR/results.json" | tee -a "$LOG"
echo "Summary: $SUMMARY" | tee -a "$LOG"
echo "Env:     $ENVFILE" | tee -a "$LOG"
