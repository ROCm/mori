#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-mori_dev_bench}"
REPO_ROOT="${REPO_ROOT:-/home/dasidler/mori}"
WORLD_SIZE="${WORLD_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-128}"
DTYPE="${DTYPE:-bf16}"
COMBINE_DTYPE="${COMBINE_DTYPE:-}"
HIDDEN_DIM="${HIDDEN_DIM:-7168}"
ZERO_COPY="${ZERO_COPY:-1}"
DISPATCH_WARPS="${DISPATCH_WARPS:-16}"
COMBINE_WARPS="${COMBINE_WARPS:-4}"
TIMEOUT="${TIMEOUT:-120}"
BLOCKS="${BLOCKS:-8 16 32 64 128}"

HOST_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$HOST_REPO_ROOT/bench_results/intranode_block_sweep_$(date +%Y%m%d_%H%M%S)}"
RAW_DIR="$OUT_DIR/raw"
SUMMARY="$OUT_DIR/summary.tsv"

mkdir -p "$RAW_DIR"

cat > "$SUMMARY" <<'EOF'
blocks	dispatch_warps	combine_warps	dispatch_lat_us	dispatch_bw_gbps	dispatch_bw2_gbps	combine_lat_us	combine_bw_gbps	combine_bw2_gbps	e2e_lat_us	raw_log
EOF

parse_raw() {
  local blocks="$1"
  local raw_file="$2"
  python3 - "$blocks" "$DISPATCH_WARPS" "$COMBINE_WARPS" "$raw_file" <<'PY'
import re
import statistics
import sys

blocks, dispatch_warps, combine_warps, path = sys.argv[1:]
phase = None
dispatch = []
combine = []
e2e = []

round_re = re.compile(r"Round\s+(\d+).*?\slat\s+([0-9.]+)\s+bw\s+([0-9.]+)\s*/\s*([0-9.]+)")
e2e_re = re.compile(r"Round\s+(\d+)\s+e2e\(us\)\s+\[([^\]]+)\]")

with open(path, "r", errors="replace") as f:
    for line in f:
        if line.startswith("Dispatch result:"):
            phase = "dispatch"
            continue
        if line.startswith("Combine result:"):
            phase = "combine"
            continue
        if line.startswith("End-to-end result:"):
            phase = "e2e"
            continue

        if phase in ("dispatch", "combine"):
            m = round_re.search(line)
            if not m:
                continue
            round_id = int(m.group(1))
            if round_id == 0:
                continue
            row = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            if phase == "dispatch":
                dispatch.append(row)
            else:
                combine.append(row)
        elif phase == "e2e":
            m = e2e_re.search(line)
            if not m:
                continue
            round_id = int(m.group(1))
            if round_id == 0:
                continue
            vals = [float(x.strip()) for x in m.group(2).split(",")]
            e2e.append(statistics.mean(vals))

def avg(rows, idx):
    if not rows:
        return float("nan")
    return statistics.mean(r[idx] for r in rows)

print(
    "\t".join(
        [
            blocks,
            dispatch_warps,
            combine_warps,
            f"{avg(dispatch, 0):.3f}",
            f"{avg(dispatch, 1):.3f}",
            f"{avg(dispatch, 2):.3f}",
            f"{avg(combine, 0):.3f}",
            f"{avg(combine, 1):.3f}",
            f"{avg(combine, 2):.3f}",
            f"{statistics.mean(e2e):.3f}" if e2e else "nan",
            path,
        ]
    )
)
PY
}

echo "Writing results to $OUT_DIR"
echo "Container: $CONTAINER"
echo "Blocks: $BLOCKS"
echo "Dispatch dtype: $DTYPE"
echo "Combine dtype: ${COMBINE_DTYPE:-<same as dispatch>}"

for blocks in $BLOCKS; do
  raw_file="$RAW_DIR/blocks_${blocks}.log"
  echo "=== Running blocks=$blocks ==="
  extra_args=""
  if [[ -n "$COMBINE_DTYPE" ]]; then
    extra_args="--combine-dtype '$COMBINE_DTYPE'"
  fi
  set +e
  docker exec "$CONTAINER" bash -lc "
    cd '$REPO_ROOT' &&
    PYTHONPATH='$REPO_ROOT/python:$REPO_ROOT' \
    MORI_SHMEM_HEAP_SIZE=6G \
    timeout '$TIMEOUT' python3 tests/python/ops/bench_dispatch_combine.py \
      --cmd bench \
      --world-size '$WORLD_SIZE' \
      --max-tokens '$MAX_TOKENS' \
      --dtype '$DTYPE' \
      $extra_args \
      --hidden-dim '$HIDDEN_DIM' \
      --zero-copy '$ZERO_COPY' \
      --dispatch-block-num '$blocks' \
      --dispatch-warp-per-block '$DISPATCH_WARPS' \
      --combine-block-num '$blocks' \
      --combine-warp-per-block '$COMBINE_WARPS'
  " 2>&1 | tee "$raw_file"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "blocks=$blocks failed with status $status" | tee -a "$OUT_DIR/errors.log"
    printf "%s\t%s\t%s\tFAILED\tFAILED\tFAILED\tFAILED\tFAILED\tFAILED\tFAILED\t%s\n" \
      "$blocks" "$DISPATCH_WARPS" "$COMBINE_WARPS" "$raw_file" >> "$SUMMARY"
    continue
  fi

  parse_raw "$blocks" "$raw_file" >> "$SUMMARY"
done

echo "=== Summary ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "Summary written to $SUMMARY"
