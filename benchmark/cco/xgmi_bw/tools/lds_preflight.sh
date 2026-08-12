#!/usr/bin/env bash
# LDS preflight for the block sweep. Pure arithmetic, touches no GPU -- run it before every sweep.
#
# Two separate things have to hold, and only the first one fails loudly at launch:
#   1. no kernel may ask for more LDS than the 320 KB a CU has; hipLaunchKernel refuses and the run
#      stops with an error, which is the harmless case;
#   2. every per-wave partition stride must cover what the kernel addresses inside it. This one is
#      silent: shrinking the constant shrinks the allocation but not the addressing, so the kernel
#      hands an out-of-range LDS offset to the TDM engine, and the process wedges in an unkillable
#      D state that survives everything short of a node reboot. It cost a node on 2026-08-11.
#
# The kernel list below is not "the kernels the table reports". BLKSWEEP runs kinds 0, 1, 8 and 9 at
# every block count (ualoe_bw.cpp, the BLKSWEEP branch), so all four are checked here.
set -uo pipefail

# Defaults are the ones compiled into ualoe_bw.cpp; override to match the GRID the sweep is built with.
D_RTD0N=256; D_RTD1N=8; D_RPIPEN=4; D_LDSPART=16384
D_MWSSPAN=8192; D_MWSPIPE=1; D_MWSISS=8; D_TWTH=256

RTD0N="${RTD0N:-$D_RTD0N}"; RTD1N="${RTD1N:-$D_RTD1N}"; RPIPEN="${RPIPEN:-$D_RPIPEN}"
LDSPART="${LDSPART:-$D_LDSPART}"; MWSSPAN="${MWSSPAN:-$D_MWSSPAN}"
MWSPIPE="${MWSPIPE:-$D_MWSPIPE}"; MWSISS="${MWSISS:-$D_MWSISS}"
TWTH="${TWTH:-$D_TWTH}"; WARP="${WARP:-32}"  # gfx1250 is wave32
TDM_NBUF="${TDM_NBUF:-2}"
LIMIT="${LIMIT:-327680}"                     # 320 KB per CU

# The values above are a second copy of numbers that live in the source, which is how a preflight ends
# up certifying a build that does not exist. Compare them against the source and say so on a mismatch.
# A read that finds nothing is ignored rather than fatal: this check must not be able to block a sweep
# because the file was reformatted.
SRC="$(dirname "$0")/../ualoe_bw.cpp"
if [ -r "$SRC" ]; then
  for m in RTD0N RTD1N RPIPEN LDSPART MWSSPAN MWSPIPE MWSISS TWTH; do
    dv="D_$m"
    sv=$(sed -n "s/^#define $m[[:space:]]\{1,\}\([0-9]\{1,\}\)[[:space:]]*\$/\1/p" "$SRC" | head -1)
    if [ -n "$sv" ] && [ "$sv" != "${!dv}" ]; then
      echo "[WARN] $m: preflight assumes ${!dv}, ualoe_bw.cpp compiles $sv -- fix tools/lds_preflight.sh"
    fi
  done
fi

TILE=$((RTD0N*RTD1N*4))
MW_SPAN=$(( LDSPART > RPIPEN*TILE ? LDSPART : RPIPEN*TILE ))
MWS_SPAN=$(( MWSSPAN > MWSPIPE*TILE ? MWSSPAN : MWSPIPE*TILE ))
fail=0

echo "tile=${TILE}B pipe=$RPIPEN ldspart=${LDSPART}B mws_span=${MWS_SPAN}B waves/block=$((TWTH/WARP))"

chk() { # name bytes
  if [ "$2" -gt "$LIMIT" ]; then
    printf '[FAIL] %-26s %8s B > limit %s\n' "$1" "$2" "$LIMIT"; fail=1
  else
    printf '[ok]   %-26s %8s B\n' "$1" "$2"
  fi
}
chk "copyk (kind 0)"            0
chk "tdm_write (kind 1)"        $((RPIPEN*TILE))
chk "tdm_store_cuload (kind 8)" $((2*LDSPART))
chk "tdmmws (kind 9)"           $((MWSISS*MWS_SPAN))

if [ "$LDSPART" -lt $((TDM_NBUF*TILE)) ]; then
  echo "[FAIL] LDSPART=$LDSPART < TDM_NBUF*tile=$((TDM_NBUF*TILE)); tdm_store_cuload would address past its partition"; fail=1
else
  echo "[ok]   LDSPART=$LDSPART covers TDM_NBUF*tile=$((TDM_NBUF*TILE))"
fi
if [ "$MWS_SPAN" -lt $((MWSPIPE*TILE)) ]; then
  echo "[FAIL] MWS_SPAN=$MWS_SPAN < MWSPIPE*tile=$((MWSPIPE*TILE)); tdmmws would address past its partition"; fail=1
else
  echo "[ok]   MWS_SPAN=$MWS_SPAN covers MWSPIPE*tile=$((MWSPIPE*TILE))"
fi
# Fewer waves in a block than issuers leaves the missing issuers' tiles uncopied: not unsafe, but it
# would be reported as a bandwidth for a copy that never happened.
if [ $((TWTH/WARP)) -lt "$MWSISS" ]; then
  echo "[FAIL] block holds $((TWTH/WARP)) waves < MWSISS=$MWSISS"; fail=1
else
  echo "[ok]   block holds $((TWTH/WARP)) waves >= MWSISS=$MWSISS"
fi

if [ "$fail" = 0 ]; then echo "LDS PREFLIGHT OK"; else echo "LDS PREFLIGHT FAILED"; fi
exit $fail
