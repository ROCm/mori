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
# Logs from a binary built with MORI_PUSH_TIMING=1 carry "[PT <coll> pe=N]"
# lines (push kernels only); each such run also gets a timing table in summary.md.
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

# Condenses the MORI_PUSH_TIMING report of one run into markdown: every cell is
# "mean / max" over PEs (and over blocks 1.. for the blk1+ columns; block 0 also
# issues the SDMA scatter, so it gets its own). All values in us, relative to
# the launch entry on the reporting PE.
push_timing_table() {
  awk '
    function num(key,   r) {
      if (match($0, "(^| )" key "=-?[0-9.]+")) {
        r = substr($0, RSTART, RLENGTH); sub(/.*=/, "", r); return r + 0
      }
      return 0
    }
    function at(key,   r) {
      if (match($0, key "=p[0-9]+@-?[0-9.]+")) {
        r = substr($0, RSTART, RLENGTH); sub(/.*@/, "", r); return r + 0
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
    !/^\[PT [a-z0-9]+ pe=[0-9]+\] / || !/ us$/ { next }
    { coll = $2 }
    / npes=[0-9]+ S=/ { S = num("S"); grid = num("grid"); acc("post", 0, num("post")); acc("issue", 0, num("issue")); next }
    / done-launch entry=/ { acc("dentry", 0, num("entry")); next }
    / s=[0-9]+ arrive first=/ {
      s = num("s"); acc("first", s, at("first")); acc("last", s, at("last")); acc("spread", s, num("spread")); next
    }
    / s=[0-9]+ blk[0-9]+ wait=/ {
      s = num("s"); b = ($0 ~ / blk0 /) ? "0" : "n"
      acc("wait" b, s, num("wait")); acc("reduce" b, s, num("reduce")); acc("sdone", s, num("done")); next
    }
    / bcast prepost=/ { acc("prepost", 0, num("prepost")); next }
    / count b=[01] blk[1-9][0-9]* fence=/ { acc("fence", num("b"), num("fence")); next }
    / tail owner=/ { acc("own", 0, num("counted")); next }
    / bcast b=[01] arrive first=/ { b = num("b"); acc("blast", b, at("last")); acc("bdone", b, num("done")); next }
    / bcast delivered first=/ { acc("bdlv", 0, at("last")); next }
    / arrive first=/ { acc("first", 0, at("first")); acc("last", 0, at("last")); acc("spread", 0, num("spread")); next }
    / delivered first=/ { acc("dlv", 0, at("last")); if (num("done")) acc("dlvdone", 0, num("done")); next }
    / end=/ { acc("end", 0, num("end")); if ($0 ~ /after-own=/) acc("afterown", 0, num("after-own")); next }
    END {
      if (!cnt["post", 0]) exit
      printf "%s S=%d grid=%d | blk0 arrival post %s, SDMA issue %s", coll, S, grid, cell("post", 0), cell("issue", 0)
      if (cnt["dentry", 0]) printf ", done-launch entry %s", cell("dentry", 0)
      printf "\n\n"
      if (coll == "ag" || coll == "a2a") {
        print "| first arrival | last arrival | arrival spread | last delivered | end |"
        print "|---|---|---|---|---|"
        printf "| %s | %s | %s | %s | %s |\n", cell("first", 0), cell("last", 0), cell("spread", 0), cell("dlv", 0), cell("end", 0)
        exit
      }
      print "| slice | wait blk1+ | reduce blk1+ | wait blk0 | reduce blk0 | first arrival | last arrival | arrival spread | slice done |"
      print "|---|---|---|---|---|---|---|---|---|"
      for (s = 0; s < S; s++)
        printf "| %d | %s | %s | %s | %s | %s | %s | %s | %s |\n", s, cell("waitn", s), cell("reducen", s), \
               cell("wait0", s), cell("reduce0", s), cell("first", s), cell("last", s), cell("spread", s), cell("sdone", s)
      if (coll == "rs") printf "\nblk0 last copy delivered at %s, wait done at %s\n", cell("dlv", 0), cell("dlvdone", 0)
      if (coll == "ar") {
        printf "\nbroadcast: prepost %s | count-in fence+sync blk1+ b0 %s, b1 %s | tail owner counted at %s\n", \
               cell("prepost", 0), cell("fence", 0), cell("fence", 1), cell("own", 0)
        printf "b0 last arrival %s (done %s) | b1 last arrival %s (done %s) | own b1 delivered %s | end %s (+%s after count)\n", \
               cell("blast", 0), cell("bdone", 0), cell("blast", 1), cell("bdone", 1), cell("bdlv", 0), cell("end", 0), cell("afterown", 0)
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
      if grep -q '^\[PT ' "$log"; then
        table=$(push_timing_table "$log")
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
    echo "Push timing, us, mean / max over PEs; times relative to the launch entry on each PE."
    echo
    printf '%s\n' "${timingTables[@]}"
  fi
} >>"$summary"

echo "== $title: $nPass pass, $nFail fail, $nSkip skip =="
[[ $nFail -eq 0 ]]
