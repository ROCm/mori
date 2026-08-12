#!/usr/bin/env bash
# How many blocks each transport needs to reach its ceiling on one xGMI link.
#
# CU, TDM, TDMc2 and TDMms are measured at the same block count, back to back, inside one process, so
# the curves are directly comparable. Across processes they are not: LOOP, build flags and clock state
# all move the absolute numbers by more than the differences this sweep is about.
#
# A block is not the same amount of hardware on the two sides. The CU kernel runs WTH=512 threads per
# block all moving data; the TDM kernel runs TWTH=256 threads per block of which MWSISS issue
# descriptors and the rest return immediately. Reporting per block is the point -- that asymmetry is
# what the table is meant to show.
#
# LOOP is cut from 50 to 10 because the low end of the sweep is slow. The full-grid row that BLKONLY
# prints at the end of each round is measured with the same LOOP, so it is the curve's own reference
# point rather than a number borrowed from another build.
set -uo pipefail
cd "$(dirname "$0")/.."

GRID="$(printf '%s' "${GRID:--DBLKMUL=64+-DWTH=512+-DTWBLK=32+-DTWTH=256+-DRTD0N=256+-DRTD1N=8+-DRPIPEN=4}" | tr '+' ' ')"
BASEX="${BASEX:--DSWEEP_16 -DONLY_1WAY -DBLKONLY -DNOVERIFY -DLOOP=10 -DWARMUP=3}"
ROUNDS="${ROUNDS:-2}"
BLKS="${BLKS:-16,32,64,128,256,512,1024,2048,4096,8192,16384}"
PORT="${PORT:-55637}"
GPUA="${GPUA:-0}"; GPUB="${GPUB:-1}"
ARCH="${ARCH:-gfx1250}"
OUT="${OUT:-/tmp}"

# The sweep allocates 16 GB per side and drives one xGMI link flat out for ~10 minutes. Anything
# already on these two GPUs will be perturbed by it and will perturb it back, so refuse to start
# rather than produce a number nobody can interpret.
if pgrep -f 'ualoe|ubk' >/dev/null 2>&1; then
  echo "REFUSING: a previous ualoe/ubk process is still alive"; exit 1
fi
# The partition-stride half of this check is silent at launch; skipping it is how a node gets wedged.
# The -D flags are the single source of truth for the geometry, so the preflight is fed from them
# rather than from a second copy of the same constants that could drift out of sync.
GRID_ENV=$(printf '%s\n' $GRID | sed -n 's/^-D\([A-Z0-9_]*\)=\(.*\)$/\1=\2/p' | tr '\n' ' ')
env $GRID_ENV bash tools/lds_preflight.sh || { echo "REFUSING: LDS preflight failed"; exit 1; }
[ -n "${PREFLIGHT_ONLY:-}" ] && { echo "PREFLIGHT_ONLY set, not running"; exit 0; }

hipcc -std=c++17 -O3 --offload-arch="$ARCH" $BASEX $GRID -DAB_ROUNDS="$ROUNDS" \
      ualoe_bw.cpp -o "$OUT/ubk" || { echo "COMPILE FAILED"; exit 1; }

BLKSWEEP=$BLKS "$OUT/ubk" listen -port="$PORT" -gpu="$GPUA" > "$OUT/bk_listen.log" 2>&1 &
LP=$!
sleep 3
BLKSWEEP=$BLKS timeout 1650 "$OUT/ubk" connect 127.0.0.1 -port="$PORT" -gpu="$GPUB" 2>&1 \
  | grep -E '^\[BLK\]|^\[KV\]|grid:|FATAL'
wait $LP 2>/dev/null
pkill -f "$OUT/ubk" 2>/dev/null
echo BLKSWEEP_DONE
