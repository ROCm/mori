#!/bin/bash
# Run collectives_benchmark over a dtype x op x size matrix for one collective
# and mode, and fail if any run fails verification, crashes or times out.
#
# Usage: run_collective_bench.sh --coll <rs|ar|ag|a2a> [--mode push|pull] [--logS n]
#
# Overridable from the environment:
#   BIN          benchmark binary          [./build/examples/collectives_benchmark]
#   NPES         GPUs per run              [8]
#   DTYPES       dtypes to sweep           [f32 bf16 f16]
#   OPS          ops to sweep (rs/ar only) [sum prod]
#   MIN_SIZE     smallest total elem count [1048576]
#   MAX_SIZE     largest total elem count  [134217728]
#   SIZE_STEP    size growth factor num/den [2/1] (e.g. 3/2 = 1.5x)
#   SIZE_ALIGN   round generated sizes down to a multiple of this [256]
#   SIZES        explicit element counts   [MIN_SIZE..MAX_SIZE, x SIZE_STEP]
#   EXTRA_SIZES  extra counts appended to SIZES (e.g. non-power-of-two) []
#   WARMUP/ITERS benchmark iterations      [2/5]
#   RUN_TIMEOUT  seconds per run           [120]
#   LOG_DIR      per-run logs + summary.md [ci-logs/collective-bench]
#   LABEL        prefix for run names and the summary title (e.g. "timing ") []
#
# Logs from a binary built with RS_PUSH_TIMING=1 carry "[RS-T ...]" lines; each
# such run also gets a per-slice wait/reduce table in summary.md.
set -uo pipefail

COLL=
MODE=push
LOGS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --coll) COLL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --logS) LOGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$COLL" ]] || { echo "--coll is required" >&2; exit 2; }

BIN=${BIN:-./build/examples/collectives_benchmark}
NPES=${NPES:-8}
DTYPES=${DTYPES:-"f32 bf16 f16"}
OPS=${OPS:-"sum prod"}
MIN_SIZE=${MIN_SIZE:-1048576}
MAX_SIZE=${MAX_SIZE:-134217728}
SIZE_STEP=${SIZE_STEP:-2/1}
SIZE_ALIGN=${SIZE_ALIGN:-256}
STEP_NUM=${SIZE_STEP%/*}; STEP_DEN=${SIZE_STEP#*/}
[[ "$SIZE_STEP" == */* ]] || STEP_DEN=1
if (( STEP_NUM <= STEP_DEN )); then
  echo "SIZE_STEP must be > 1: $SIZE_STEP" >&2; exit 2
fi
if [[ -z "${SIZES:-}" ]]; then
  SIZES=
  for ((s = MIN_SIZE; s <= MAX_SIZE; s = s * STEP_NUM / STEP_DEN)); do
    SIZES+="$(( s / SIZE_ALIGN * SIZE_ALIGN )) "
  done
fi
# Append extra (e.g. non-power-of-two) element counts to whatever SIZES holds.
[[ -n "${EXTRA_SIZES:-}" ]] && SIZES+=" $EXTRA_SIZES"
WARMUP=${WARMUP:-2}
ITERS=${ITERS:-5}
RUN_TIMEOUT=${RUN_TIMEOUT:-120}
LOG_DIR=${LOG_DIR:-ci-logs/collective-bench}

case "$COLL" in
  rs|reduce_scatter|ar|all_reduce) ;;
  *) OPS="sum" ;;  # op is ignored by the non-reducing collectives
esac

[[ -x "$BIN" ]] || { echo "benchmark binary not found: $BIN" >&2; exit 2; }
mkdir -p "$LOG_DIR"

export HSA_NO_SCRATCH_RECLAIM=1
export HIP_FORCE_DEV_KERNARG=1
export MORI_SHMEM_MODE=${MORI_SHMEM_MODE:-STATIC_HEAP}
export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-5G}
export MORI_DISABLE_P2P=0
export MORI_ENABLE_SDMA=1
export MORI_SDMA_NUM_CHANNELS=${MORI_SDMA_NUM_CHANNELS:-1}
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-lo}
# Avoids "request to allocate mask for invalid number" from libnuma.
if [[ -f /lib/x86_64-linux-gnu/libnuma.so.1 ]]; then
  export LD_PRELOAD=/lib/x86_64-linux-gnu/libnuma.so.1${LD_PRELOAD:+:$LD_PRELOAD}
fi

LABEL=${LABEL:-}
title="${LABEL}$COLL mode=$MODE logS=$LOGS npes=$NPES"
tag="${LABEL// /_}${COLL}_${MODE}_logS${LOGS}"
rows=()
timingTables=()

# Condenses the RS_PUSH_TIMING printf's of one run into a markdown table: per
# slice, mean/max over PEs of the waits and reduces of blocks 1.. (block 0 also
# issues the SDMA scatter, so it gets its own column) and of the sender arrival
# spread. All values in us.
rs_timing_table() {
  awk '
    function num(key,   r) {
      if (match($0, key "=-?[0-9.]+")) {
        r = substr($0, RSTART + length(key) + 1, RLENGTH - length(key) - 1); return r + 0
      }
      return 0
    }
    function acc(name, s, v) {
      sum[name, s] += v; cnt[name, s]++
      if (!((name, s) in mx) || v > mx[name, s]) mx[name, s] = v
    }
    function cell(name, s) {
      if (!cnt[name, s]) return "-"
      return sprintf("%.2f / %.2f", sum[name, s] / cnt[name, s], mx[name, s])
    }
    # Device printf (stdout) and the host PASS line (stderr) share the log, so a
    # line can be cut by the other stream: only take lines ending in " us".
    !/\[RS-T / || !/ us( \(seen by blk[0-9]+\))?$/ { next }
    / S=[0-9]+ grid=/ {
      S = num("S"); grid = num("grid"); acc("barrier", 0, num("barrier")); acc("issue", 0, num("issue"))
      next
    }
    / s=[0-9]+ arrive first=/ {
      s = num(" s"); acc("spread", s, num("spread"))
      if (match($0, /last=p[0-9]+@-?[0-9.]+/)) acc("last", s, substr($0, RSTART + index(substr($0, RSTART), "@"), RLENGTH) + 0)
      if (match($0, /first=p[0-9]+@-?[0-9.]+/)) acc("first", s, substr($0, RSTART + index(substr($0, RSTART), "@"), RLENGTH) + 0)
      next
    }
    # All-reduce push broadcast phase.
    / bcast prepost=/ { ar = 1; acc("prepost", 0, num("prepost")); next }
    / count b=[01] blk[1-9][0-9]* fence=/ { acc("fence", num(" b"), num("fence")); next }
    / tail owner=/ { acc("own", 0, num("counted")); next }
    / bcast b=[01] arrive first=/ {
      b = num(" b"); acc("bdone", b, num("done")); acc("bafter", b, num("after-own"))
      next
    }
    / blk[0-9]+ wait=/ {
      s = num(" s"); b = ($0 ~ / blk0 /) ? "0" : "n"
      acc("wait" b, s, num("wait")); acc("reduce" b, s, num("reduce")); acc("done", s, num("done"))
      if (S > nS) nS = S
      next
    }
    END {
      if (!nS) exit
      printf "S=%d grid=%d | blk0 entry barrier %s, SDMA issue %s (mean / max over PEs)\n\n", \
             nS, grid, cell("barrier", 0), cell("issue", 0)
      print "| slice | wait blk1+ | reduce blk1+ | wait blk0 | reduce blk0 | first arrival | last arrival | arrival spread | slice done |"
      print "|---|---|---|---|---|---|---|---|---|"
      for (s = 0; s < nS; s++)
        printf "| %d | %s | %s | %s | %s | %s | %s | %s | %s |\n", s, cell("waitn", s), \
               cell("reducen", s), cell("wait0", s), cell("reduce0", s), cell("first", s), \
               cell("last", s), cell("spread", s), cell("done", s)
      if (ar) {
        printf "\nbroadcast: prepost %s | count-in fence+sync blk1+ b0 %s, b1 %s | tail owner counted at %s | b0 landed at %s (+%s after count) | b1 landed at %s (+%s after count)\n", \
               cell("prepost", 0), cell("fence", 0), cell("fence", 1), cell("own", 0), \
               cell("bdone", 0), cell("bafter", 0), cell("bdone", 1), cell("bafter", 1)
      }
    }
  ' "$1"
}
nPass=0; nFail=0; nSkip=0

for size in $SIZES; do
  for dtype in $DTYPES; do
    for op in $OPS; do
      name="${tag}_${dtype}_${op}_${size}"
      log="$LOG_DIR/$name.log"
      cmd=("$BIN" --coll "$COLL" --npes "$NPES" --size "$size" --dtype "$dtype" --op "$op"
           --mode "$MODE" --logS "$LOGS" --warmup "$WARMUP" --iters "$ITERS")

      echo "::group::$name"
      echo "+ ${cmd[*]}"
      timeout -k 10 "$RUN_TIMEOUT" "${cmd[@]}" >"$log" 2>&1
      rc=$?
      cat "$log"
      echo "::endgroup::"

      passLine=$(grep -m1 ': PASS' "$log" || true)
      avg=$(sed -nE 's/.*avg ([0-9.]+) ms \(([0-9.]+) GB\/s\).*/\1 ms | \2 GB\/s/p' <<<"$passLine")
      if grep -q 'not supported by this build' "$log"; then
        status=SKIP; nSkip=$((nSkip + 1)); avg="- | -"
      elif [[ $rc -eq 0 && -n "$passLine" ]] && ! grep -q ': FAIL' "$log"; then
        status=PASS; nPass=$((nPass + 1))
      else
        [[ $rc -eq 124 || $rc -eq 137 ]] && status=TIMEOUT || status=FAIL
        nFail=$((nFail + 1)); avg="- | -"
        echo "::error title=collective bench $status::$name (exit $rc)"
      fi
      printf '%-8s %s\n' "$status" "$name"
      rows+=("| $dtype | $op | $size | $status | $avg |")
      if grep -q '\[RS-T ' "$log"; then
        table=$(rs_timing_table "$log")
        echo "$table"
        timingTables+=("#### $name ($avg)" "" "$table" "")
      fi
    done
  done
done

summary="$LOG_DIR/summary.md"
{
  echo "### $title: $nPass pass, $nFail fail, $nSkip skip"
  echo
  echo "| dtype | op | size | result | avg | avg BW |"
  echo "|---|---|---|---|---|---|"
  printf '%s\n' "${rows[@]}"
  echo
  if [[ ${#timingTables[@]} -gt 0 ]]; then
    echo "RS push slice timing, us, mean / max over PEs (and over blocks 1.. for blk1+)."
    echo "Arrival and done times are relative to block 0 entry on the receiving PE."
    echo
    printf '%s\n' "${timingTables[@]}"
  fi
} >>"$summary"

echo "== $title: $nPass pass, $nFail fail, $nSkip skip =="
[[ $nFail -eq 0 ]]
