# SDMA Dispatch Experiment Runbook

This document explains how to reproduce the SDMA dispatch experiments and how
to parse the results.

The commands below assume:

- repo root: `/home/dasidler/mori`
- running benchmark container: `mori_dev_bench`
- EP world size: `8`
- target GPU arch in the container is usable by the existing build/JIT setup

## Current Repository State

The latest no-return atomic experiment intentionally uses:

```bash
git switch --detach f56a005e0776c10113986a76614c3a6d8def0677
```

with local edits:

```text
include/mori/application/transport/sdma/sdma_pkt_struct.h
include/mori/application/transport/sdma/anvil_device.hpp
src/ops/dispatch_combine/intranode.hpp
```

Those edits add:

```cpp
SDMA_ATOMIC_ADD64_NO_RETURN = 111
CreateAtomicAddNoReturnPacket(...)
```

and use it for SDMA completion packets.

To see the current local diff:

```bash
git diff -- \
  include/mori/application/transport/sdma/sdma_pkt_struct.h \
  include/mori/application/transport/sdma/anvil_device.hpp \
  src/ops/dispatch_combine/intranode.hpp
```

## Build / Rebuild

### Fast HIP Compile Checks

Non-profiler:

```bash
cd /home/dasidler/mori
hipcc -c --cuda-device-only -emit-llvm \
  --offload-arch=gfx942 \
  -fgpu-rdc \
  -std=c++17 \
  -O2 \
  -Ibuild/generated/include \
  -Iinclude \
  -I. \
  src/ops/kernels/ep_intranode.hip \
  -o /tmp/ep_intranode.bc
```

Profiler:

```bash
cd /home/dasidler/mori
hipcc -c --cuda-device-only -emit-llvm \
  --offload-arch=gfx942 \
  -fgpu-rdc \
  -std=c++17 \
  -O2 \
  -DENABLE_PROFILER \
  -Ibuild/generated/include \
  -Iinclude \
  -I. \
  src/ops/kernels/ep_intranode.hip \
  -o /tmp/ep_intranode_prof.bc
```

Whitespace check:

```bash
git diff --check
```

### Rebuild Extension In The Container

Use this after changing C++ headers, args layout, pybinds, profiler bindings, or
anything the installed extension needs to see.

```bash
docker exec mori_dev_bench bash -lc \
  'cd /home/dasidler/mori && ENABLE_PROFILER=ON python3 -m pip install -e .'
```

For profiler-disabled benchmark-only work, the editable build can still use
`ENABLE_PROFILER=ON`; the runtime `ENABLE_PROFILER=0/1` controls JIT kernel
variant selection.

## Focused Benchmark Command

This is the main correctness/performance smoke test used repeatedly:

```bash
docker exec \
  -e MORI_ENABLE_SDMA=1 \
  -e ENABLE_PROFILER=0 \
  mori_dev_bench \
  bash -lc 'rm -rf /root/.mori/jit &&
    cd /home/dasidler/mori &&
    PYTHONPATH=/home/dasidler/mori/python:/home/dasidler/mori
    MORI_OPS_LOG_LEVEL=INFO
    MORI_SHMEM_HEAP_SIZE=6G
    timeout 240
    python3 tests/python/ops/bench_dispatch_combine.py
      --cmd bench
      --world-size 8
      --max-tokens 128
      --dtype bf16
      --hidden-dim 7168
      --zero-copy 1
      --dispatch-block-num 32
      --dispatch-warp-per-block 16
      --combine-block-num 32
      --combine-warp-per-block 4'
```

Expected shape for the no-return atomic experiment from `f56a`:

```text
dispatch around ~88 us for max_tokens=128, blocks=32, wpb=16
```

## Focused Profile Command

Use this to export Perfetto JSON traces:

```bash
docker exec \
  -e MORI_ENABLE_SDMA=1 \
  -e ENABLE_PROFILER=1 \
  mori_dev_bench \
  bash -lc 'rm -rf /root/.mori/jit &&
    cd /home/dasidler/mori &&
    PYTHONPATH=/home/dasidler/mori/python:/home/dasidler/mori
    MORI_OPS_LOG_LEVEL=INFO
    MORI_SHMEM_HEAP_SIZE=6G
    timeout 360
    python3 tests/python/ops/bench_dispatch_combine.py
      --cmd profile
      --world-size 8
      --max-tokens 128
      --dtype bf16
      --hidden-dim 7168
      --zero-copy 1
      --dispatch-block-num 32
      --dispatch-warp-per-block 16
      --combine-block-num 32
      --combine-warp-per-block 4'
```

Trace files are emitted in the repo root:

```text
trace_intranode_rank0_<timestamp>.json
...
trace_intranode_rank7_<timestamp>.json
```

Example no-return atomic trace:

```text
trace_intranode_rank*_0616_215652.json
```

## Parsing Perfetto Trace JSON

Use this parser snippet to summarize SDMA completion, copy warp, submit, and
global-warp post-loop waits:

```bash
cd /home/dasidler/mori
python3 - <<'PY'
import json, pathlib, statistics, collections, re

stamp = "0616_215652"  # change this
files = sorted(pathlib.Path(".").glob(f"trace_intranode_rank*_{stamp}.json"))

start = "dispatch_sdma_completion_timestamp_start"
end = "dispatch_sdma_completion_timestamp_end"
slots = [
    "dispatch_sdma_completion",
    "dispatch_sdma_copy_warp",
    "dispatch_wait_peer_token",
    "dispatch_sdma_post_loop",
    "dispatch_sdma_submit",
]

durs = collections.defaultdict(list)
starts = collections.defaultdict(list)
timestamp_deltas = []
counts = collections.Counter()

for p in files:
    rank = int(re.search(r"rank(\d+)", p.name).group(1))
    data = json.load(open(p))
    stacks = collections.defaultdict(list)
    counts.update(e["name"] for e in data["traceEvents"])
    for ev in sorted(data["traceEvents"], key=lambda e: e.get("ts", 0)):
        name = ev["name"]
        tid = ev.get("tid")
        ts = ev.get("ts")

        if name == start:
            starts[(rank, tid)].append(ts)
        elif name == end and starts[(rank, tid)]:
            timestamp_deltas.append(ts - starts[(rank, tid)].pop(0))

        if name in slots:
            key = (rank, tid, name)
            if ev.get("ph") == "B":
                stacks[key].append(ts)
            elif ev.get("ph") == "E" and stacks[key]:
                s = stacks[key].pop()
                if ts >= s:
                    durs[name].append(ts - s)

def pct(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    f = k - lo
    return vals[lo] * (1 - f) + vals[hi] * f

def stat(vals):
    if not vals:
        return "n=0"
    return (
        f"n={len(vals)} med={statistics.median(vals):.2f} "
        f"p90={pct(vals,90):.2f} min={min(vals):.2f} max={max(vals):.2f}"
    )

print("files", len(files))
print("timestamp counts", counts[start], counts[end], "delta", stat([x for x in timestamp_deltas if x >= 0]))
for s in slots:
    print(s, stat(durs[s]))

for name in ["dispatch_sdma_post_loop", "dispatch_notify_peer", "dispatch_wait_peer_token"]:
    vals = []
    for p in files:
        data = json.load(open(p))
        stack = []
        for ev in data["traceEvents"]:
            if ev.get("tid") != 0 or ev.get("name") != name:
                continue
            if ev.get("ph") == "B":
                stack.append(ev["ts"])
            elif ev.get("ph") == "E" and stack:
                s = stack.pop()
                if ev["ts"] >= s:
                    vals.append(ev["ts"] - s)
    print("tid0", name, stat(vals))
PY
```

## Sweeps Created During This Work

### SDMA vs Non-SDMA Blocks/Tokens Sweep

Directory:

```text
bench_results/sdma_vs_nosdma_blocks_tokens_20260612/
```

Scripts:

```text
bench_results/sdma_vs_nosdma_blocks_tokens_20260612/run_remaining_sweep.sh
bench_results/sdma_vs_nosdma_blocks_tokens_20260612/plot_dispatch_sweep.py
```

Summary:

```text
bench_results/sdma_vs_nosdma_blocks_tokens_20260612/summary.csv
```

Plot outputs:

```text
bench_results/sdma_vs_nosdma_blocks_tokens_20260612/dispatch_vs_max_tokens.png
bench_results/sdma_vs_nosdma_blocks_tokens_20260612/dispatch_vs_max_tokens.svg
```

This sweep compares `MORI_ENABLE_SDMA=0` and `1`.

### Post-Barrier Per-Queue Completion Sweep

Directory:

```text
bench_results/sdma_postbarrier_blocks_tokens_20260615/
```

Script:

```text
bench_results/sdma_postbarrier_blocks_tokens_20260615/run_sweep.sh
```

Summary:

```text
bench_results/sdma_postbarrier_blocks_tokens_20260615/summary.csv
```

This corresponds to the version that moved completion after a grid barrier and
aggregated by `queuePe`, while still using remote completion atomics.

### Local Completion Sweep

Directory:

```text
bench_results/sdma_local_completion_blocks_tokens_20260615/
```

Script:

```text
bench_results/sdma_local_completion_blocks_tokens_20260615/run_sweep.sh
```

Summary:

```text
bench_results/sdma_local_completion_blocks_tokens_20260615/summary.csv
```

This corresponds to the local queue-completion protocol where senders wait for
queue drain before publishing `numTokenSignal`.

## Generic Sweep Script Template

The sweep scripts follow this pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/dasidler/mori
mkdir -p bench_results/<result-dir>/raw

for blocks in 8 16 32; do
  for tokens in 64 128 256 512 1024 2048 4096; do
    log="bench_results/<result-dir>/raw/blocks_${blocks}_tokens_${tokens}.log"
    if [[ -f "${log}" ]] && grep -q 'Round 9 e2e' "${log}"; then
      echo "SKIP complete blocks=${blocks} tokens=${tokens}"
      continue
    fi

    echo "RUN blocks=${blocks} tokens=${tokens}"
    docker exec \
      -e MORI_ENABLE_SDMA=1 \
      -e ENABLE_PROFILER=0 \
      mori_dev_bench \
      bash -lc "rm -rf /root/.mori/jit && cd /home/dasidler/mori && PYTHONPATH=/home/dasidler/mori/python:/home/dasidler/mori MORI_OPS_LOG_LEVEL=INFO MORI_SHMEM_HEAP_SIZE=6G timeout 360 python3 tests/python/ops/bench_dispatch_combine.py --cmd bench --world-size 8 --max-tokens '${tokens}' --dtype bf16 --hidden-dim 7168 --zero-copy 1 --dispatch-block-num '${blocks}' --dispatch-warp-per-block 16 --combine-block-num '${blocks}' --combine-warp-per-block 4" \
      2>&1 | tee "${log}"
  done
done
```

## Parsing Sweep Logs

Use this parser for sweep logs:

```bash
cd /home/dasidler/mori
python3 - <<'PY'
import pathlib, re, statistics, csv

root = pathlib.Path("bench_results/sdma_local_completion_blocks_tokens_20260615/raw")
out = root.parent / "summary.csv"
rows = []
missing = []

for blocks in (8, 16, 32):
    for tokens in (64, 128, 256, 512, 1024, 2048, 4096):
        p = root / f"blocks_{blocks}_tokens_{tokens}.log"
        if not p.exists():
            missing.append((blocks, tokens, "missing"))
            continue

        text = p.read_text(errors="ignore")
        dispatch = re.search(r"Dispatch result:\n(.*?)(?:\n\nCombine result:)", text, re.S)
        dispatch_lat = []
        dispatch_bw = []
        if dispatch:
            for m in re.finditer(
                r"Round\s+(\d+)\s+duration\(us\).*?\blat\s+([0-9.]+)\s+bw\s+([0-9.]+)",
                dispatch.group(1),
            ):
                if int(m.group(1)) == 0:
                    continue
                dispatch_lat.append(float(m.group(2)))
                dispatch_bw.append(float(m.group(3)))

        e2e = []
        e2e_block = re.search(r"End-to-end result:.*?(?=\n\[mori-jit\]|\Z)", text, re.S)
        if e2e_block:
            for m in re.finditer(r"Round\s+(\d+)\s+e2e\(us\)\s+\[([^\]]+)\]", e2e_block.group(0)):
                if int(m.group(1)) == 0:
                    continue
                vals = [float(x.strip()) for x in m.group(2).split(",") if x.strip()]
                if vals:
                    e2e.append(statistics.median(vals))

        if not dispatch_lat or not e2e:
            missing.append((blocks, tokens, "incomplete"))

        rows.append(
            {
                "blocks": blocks,
                "max_tokens": tokens,
                "dispatch_us_med": round(statistics.median(dispatch_lat), 1) if dispatch_lat else "",
                "dispatch_bw_gbs_med": round(statistics.median(dispatch_bw), 2) if dispatch_bw else "",
                "e2e_us_med": round(statistics.median(e2e), 1) if e2e else "",
                "rounds": len(dispatch_lat),
                "log": str(p),
            }
        )

with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("wrote", out, "missing", missing)
print("| blocks | max_tokens | dispatch us | dispatch GB/s | e2e us |")
print("|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['blocks']} | {r['max_tokens']} | {r['dispatch_us_med']} | {r['dispatch_bw_gbs_med']} | {r['e2e_us_med']} |")
PY
```

## Key Result Tables

### Local Completion Variant

From:

```text
bench_results/sdma_local_completion_blocks_tokens_20260615/summary.csv
```

| blocks | max_tokens | dispatch us | dispatch GB/s | e2e us |
|---:|---:|---:|---:|---:|
| 8 | 64 | 82.9 | 57.85 | 185.1 |
| 8 | 128 | 117.2 | 82.15 | 328.8 |
| 8 | 256 | 185.7 | 104.08 | 609.8 |
| 8 | 512 | 318.4 | 121.86 | 1171.7 |
| 8 | 1024 | 592.4 | 131.07 | 2299.8 |
| 8 | 2048 | 1129.8 | 137.47 | 4539.8 |
| 8 | 4096 | 2201.0 | 141.15 | 9018.2 |
| 16 | 64 | 69.1 | 69.38 | 117.0 |
| 16 | 128 | 91.5 | 105.25 | 194.6 |
| 16 | 256 | 134.5 | 143.64 | 346.6 |
| 16 | 512 | 222.3 | 174.53 | 649.5 |
| 16 | 1024 | 397.8 | 195.23 | 1252.0 |
| 16 | 2048 | 745.1 | 208.44 | 2452.1 |
| 16 | 4096 | 1422.0 | 218.47 | 4848.6 |
| 32 | 64 | 66.0 | 72.72 | 89.5 |
| 32 | 128 | 88.8 | 108.39 | 138.6 |
| 32 | 256 | 127.9 | 151.13 | 233.9 |
| 32 | 512 | 211.2 | 183.70 | 425.4 |
| 32 | 1024 | 374.2 | 207.47 | 806.1 |
| 32 | 2048 | 696.9 | 222.85 | 1565.7 |
| 32 | 4096 | 1349.4 | 230.22 | 3079.6 |

## Gotchas

1. Always clear the JIT cache when changing HIP code:

   ```bash
   rm -rf /root/.mori/jit
   ```

2. Rebuild the extension when changing exported C++ layout or profiler slots.

3. The Python/JIT source root should be `/home/dasidler/mori`. Verify with:

   ```bash
   docker exec mori_dev_bench bash -lc 'python3 - <<PY
from mori.jit.config import get_mori_source_root
print(get_mori_source_root())
PY'
   ```

4. Profiler traces can be misleading if slot maps and generated headers are out
   of sync. Regenerate and rebuild after adding/removing profiler slots:

   ```bash
   docker exec mori_dev_bench bash -lc \
     'cd /home/dasidler/mori &&
      python3 tools/profiler/generate_profiler_bindings.py \
        . \
        src/ops/dispatch_combine \
        build/generated/include/mori/profiler \
        build/generated/src/pybind/profiler_bindings_generated.cpp'
   ```

5. Timestamps are useful for relative SDMA-engine measurements, but absolute
   units need calibration.

6. The current no-return atomic experiment is on detached `f56a005e`, not on the
   later local-completion commit.

