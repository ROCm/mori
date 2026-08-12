#!/usr/bin/env bash
# Transfer-size x grid-width sweep of the two transports: the CU uint4 copy and the staged TDM copy,
# GPU0 -> GPU1 over xGMI, one direction's bytes only. The block sweep answers "how wide a grid does
# each transport need"; this one answers the same question at every transfer size, which is what
# decides whether a given message is large enough for TDM to be worth using at all.
#
# The CU axis is a grid width, not a physical CU count: the sweep launches CUMUL blocks per unit on
# the CU side and TDMMUL on the TDM side, so the two sides are deliberately not the same amount of
# hardware. CUMUL=64 / TDMMUL=32 reproduces the recorded tables. Set MATRIX_CUMASK=1 to turn the axis
# into an actual CU count instead.
set -uo pipefail
cd "$(dirname "$0")/.."

GRID="$(printf '%s' "${GRID:--DBLKMUL=64+-DWTH=512+-DTWBLK=32+-DTWTH=256+-DRTD0N=256+-DRTD1N=8+-DRPIPEN=4}" | tr '+' ' ')"
BASEX="${BASEX:--DSWEEP_MATRIX -DONLY_1WAY}"
CUS="${CUS:-1,2,4,8,16,32,64,128,256}"
SZS="${SZS:-}"                      # empty => 1KB doubling up to the allocation
BUDGET="${BUDGET:-34359738368}"     # bytes moved per cell; iteration count follows from it
MINIT="${MINIT:-5}"; MAXIT="${MAXIT:-200}"
CUMUL="${CUMUL:-64}"; TDMMUL="${TDMMUL:-32}"
MAXB="${MAXB:-8589934592}"          # allocation = largest cell the sweep may ask for
GSRC="${GSRC:-0}"; GDST="${GDST:-1}"
PORT="${PORT:-55643}"
ARCH="${ARCH:-gfx1250}"
OUT="${OUT:-/tmp/uamatrix.txt}"
BIN="${BIN:-/tmp/umx}"

if pgrep -f 'ualoe|umx' >/dev/null 2>&1; then
  echo "REFUSING: a previous ualoe/umx process is still alive"; exit 1
fi
# The matrix runs the same kernels as the block sweep, so it is gated on the same preflight. Its
# partition-stride half is silent at launch; skipping it is how a node gets wedged.
GRID_ENV=$(printf '%s\n' $GRID | sed -n 's/^-D\([A-Z0-9_]*\)=\(.*\)$/\1=\2/p' | tr '\n' ' ')
env $GRID_ENV bash tools/lds_preflight.sh || { echo "REFUSING: LDS preflight failed"; exit 1; }
[ -n "${PREFLIGHT_ONLY:-}" ] && { echo "PREFLIGHT_ONLY set, not running"; exit 0; }

hipcc -std=c++17 -O3 --offload-arch="$ARCH" $BASEX -DMATRIX_MAXB="${MAXB}UL" $GRID \
      ualoe_bw.cpp -o "$BIN" || { echo "COMPILE FAILED"; exit 1; }

# ulimit -c 0: a crash here would otherwise drop a multi-GB core into /tmp, which has filled the
# node's disk before and taken dockerd down with it.
ulimit -c 0
export MATRIX=1 MATRIX_BYTES=$BUDGET MATRIX_MINIT=$MINIT MATRIX_MAXIT=$MAXIT
export MATRIX_CUMUL=$CUMUL MATRIX_TDMMUL=$TDMMUL
export MATRIX_HYB=${HYB:-0} MATRIX_C2=${C2:-0} MATRIX_NT=${NT:-0}
# Which kernel the TDM column reports: 1 = tdm_write, one issuing wave per block (every recorded
# table); 9 = tdmmws, MWSISS waves per block. Issuer count and per-wave LDS span are compile time, so
# this has to travel with GRID rather than be set on its own.
export MATRIX_TDMKIND=${TDMKIND:-1}
export MATRIX_CUMASK=${CUMASK:-0} MATRIX_SO=${SO:-0}
# Each CU-masked stream takes a dedicated hardware queue and the sweep rebuilds one per row; the
# default allowance of 4 ran out on the second row.
export GPU_MAX_HW_QUEUES=${HWQ:-16}
export MATRIX_CU=$CUS
[ -n "$SZS" ] && export MATRIX_SZ=$SZS

"$BIN" listen -port="$PORT" -gpu="$GDST" > /tmp/mx_listen.log 2>&1 &
LP=$!
sleep 3
"$BIN" connect 127.0.0.1 -port="$PORT" -gpu="$GSRC" > "$OUT" 2>&1
wait $LP 2>/dev/null
pkill -f "$BIN" 2>/dev/null

echo -n "cells="; grep -cE '^\[MX\] ' "$OUT"
grep -E '^\[MXCFG\]|^\[MXV\]|^\[VERIFY\]|FATAL' "$OUT"
grep -E '^\[MX\] ' "$OUT"
echo UAMATRIX_DONE
