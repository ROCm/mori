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
#   SIZES        explicit element counts   [MIN_SIZE..MAX_SIZE, doubling]
#   EXTRA_SIZES  extra counts appended to SIZES (e.g. non-power-of-two) []
#   WARMUP/ITERS benchmark iterations      [2/5]
#   RUN_TIMEOUT  seconds per run           [120]
#   LOG_DIR      per-run logs + summary.md [ci-logs/collective-bench]
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
if [[ -z "${SIZES:-}" ]]; then
  SIZES=
  for ((s = MIN_SIZE; s <= MAX_SIZE; s *= 2)); do SIZES+="$s "; done
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

title="$COLL mode=$MODE logS=$LOGS npes=$NPES"
tag="${COLL}_${MODE}_logS${LOGS}"
rows=()
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
} >>"$summary"

echo "== $title: $nPass pass, $nFail fail, $nSkip skip =="
[[ $nFail -eq 0 ]]
