#!/usr/bin/env bash
# Self-contained reproduction check for the gfx1250 xGMI bandwidth results.
#
# Runs entirely inside the container: compiles both benchmarks from the sources shipped alongside it,
# measures, and compares against the baseline recorded on the reference node. Exits non-zero if any
# measurement falls outside tolerance, so it is usable as a gate.
#
# The baselines below were measured on ctheliosr-1b114-f01-2 with the pinned base image
# (HIP 7.14.60850, clang 23.0.0git). They are per-configuration, not general truths: the grid and tile
# parameters are compiled in, and changing them invalidates the comparison.
set -uo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-gfx1250}"
TOL="${TOL:-5}"                       # percent band around each baseline
PORT="${PORT:-55571}"
# 8GB workset needs ~24GB of VRAM on each of two GPUs (peer window + src + dst).
XF="${EPCHECK_XFLAGS:--DSWEEP_ONE -DONLY_1WAY}"
WORK="$(mktemp -d /tmp/epcheck.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

# --- baselines (GB/s) ---
BL_A2A=1627.6      # tdma2a, 4 rank, grid=512, aggregate incl. self-write
BL_CU=1641.0       # ualoe, 8GB, GPU0->GPU1 one-way CU copy
BL_TDM=1637.3      # ualoe, 8GB, GPU0->GPU1 one-way TDM copy (256x8, pipe 4)
BL_TDMNL=1643.0    # ualoe, 8GB, one-way TDM store without staging load

pass=0; fail=0; skip=0
chk(){
  local n="$1" m="$2" b="$3" lo hi ok
  if [ -z "$m" ]; then printf "  %-26s MISSING (could not parse output)\n" "$n"; fail=$((fail+1)); return; fi
  lo=$(awk -v b="$b" -v t="$TOL" 'BEGIN{printf "%.1f", b*(1-t/100)}')
  hi=$(awk -v b="$b" -v t="$TOL" 'BEGIN{printf "%.1f", b*(1+t/100)}')
  ok=$(awk -v m="$m" -v lo="$lo" -v hi="$hi" 'BEGIN{print (m>=lo && m<=hi)?"PASS":"FAIL"}')
  printf "  %-26s %8s   baseline %8s   band [%s, %s]   %s\n" "$n" "$m" "$b" "$lo" "$hi" "$ok"
  if [ "$ok" = PASS ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
}

echo "=================== L0  environment ==================="
hipcc --version 2>&1 | sed -n '1,2p' | sed 's/^/  /'
cat > "$WORK/dc.cc" <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int n=0; if(hipGetDeviceCount(&n)!=hipSuccess){printf("count=0\n");return 1;}
  hipDeviceProp_t p;
  for(int i=0;i<n;i++){ hipGetDeviceProperties(&p,i);
    printf("  gpu%d arch=%s CU=%d vram=%.0fGB\n",i,p.gcnArchName,p.multiProcessorCount,p.totalGlobalMem/1073741824.0); }
  printf("count=%d\n",n); return 0; }
EOF
if ! hipcc -O1 "$WORK/dc.cc" -o "$WORK/dc" 2>"$WORK/dc.err"; then
  echo "  FATAL hipcc cannot build a trivial program"; sed 's/^/  /' "$WORK/dc.err" | head -10; exit 2
fi
"$WORK/dc" | sed '/^count=/d'
NG=$("$WORK/dc" | sed -n 's/^count=//p')
GFX=$("$WORK/dc" | sed -n '1s/.*arch=\([^ ]*\).*/\1/p')
echo "  gpus=$NG arch=$GFX"
if [ "${GFX%%:*}" != "$ARCH" ]; then
  echo "  FATAL arch mismatch: found ${GFX%%:*}, expected $ARCH (TDM is gfx1250-specific)"; exit 2
fi
[ "${NG:-0}" -ge 2 ] || { echo "  FATAL need at least 2 GPUs, found $NG"; exit 2; }

echo "=================== L1  build ==================="
if hipcc --offload-arch="$ARCH" -std=c++17 -O3 -DTILE_S=64 tdma2a.cc -o "$WORK/tdma2a" 2>"$WORK/b1.err"; then
  echo "  tdma2a      OK"
else
  echo "  tdma2a      FAILED"; sed 's/^/    /' "$WORK/b1.err" | head -15; exit 3
fi
# shellcheck disable=SC2086
if hipcc --offload-arch="$ARCH" -std=c++17 -O3 $XF \
     -DBLKMUL=64 -DWTH=512 -DTWBLK=32 -DTWTH=256 -DRTD0N=256 -DRTD1N=8 -DRPIPEN=4 \
     ualoe_bw.cpp -o "$WORK/ualoe_bw" 2>"$WORK/b2.err"; then
  echo "  ualoe_bw    OK  ($XF, tile 256x8, pipe 4)"
else
  echo "  ualoe_bw    FAILED"; sed 's/^/    /' "$WORK/b2.err" | head -15; exit 3
fi

echo "=================== L2  TDM all-to-all ==================="
if [ "$NG" -ge 4 ]; then
  A2A_NR=4 TDM_NUM_TILES=4096 TDM_ITER=50 TDM_GRIDS="256 512 1024" \
    "$WORK/tdma2a" >"$WORK/a2a.log" 2>&1
  sed -n '/grid=/s/^/  /p' "$WORK/a2a.log"
  A2A=$(sed -nE 's@.*grid=512 +: +([0-9.]+) GB/s evt.*@\1@p' "$WORK/a2a.log" | head -1)
  chk "a2a grid=512 aggregate" "$A2A" "$BL_A2A"
else
  echo "  SKIP: needs 4 GPUs, found $NG"; skip=$((skip+1))
fi

echo "=================== L3  GPU0 -> GPU1 one-way copy ==================="
"$WORK/ualoe_bw" listen -port="$PORT" -gpu=1 >"$WORK/ua_l.log" 2>&1 &
LP=$!
sleep 3
"$WORK/ualoe_bw" connect 127.0.0.1 -port="$PORT" -gpu=0 >"$WORK/ua_c.log" 2>&1
wait $LP 2>/dev/null
sed -n '/GB\|size\|----/{/^\[KV\]/d; s/^/  /p;}' "$WORK/ua_c.log"
# Parsed from the [KV] line, not from column positions: the table width changes with ONLY_1WAY and
# with every variant added, and a positional parser silently checks the wrong figure against the
# wrong baseline when it does.
KV=$(grep -m1 '^\[KV\] size=8.00GB' "$WORK/ua_c.log")
kv(){ sed -n "s/.* $1=\([0-9.]*\).*/\1/p" <<<"$KV"; }
CU=$(kv CU1); TDM=$(kv TDM1); TDMNL=$(kv TDMnl1); TDMCU=$(kv TDMcu1)
chk "CU copy 1way"    "$CU"    "$BL_CU"
chk "TDM copy 1way"   "$TDM"   "$BL_TDM"
chk "TDM store 1way"  "$TDMNL" "$BL_TDMNL"
# Reported but not gated: no baseline has been established for this variant yet.
printf "  %-26s %8s   (informational, no baseline)\n" "TDMcu copy 1way" "${TDMCU:-?}"

echo "=================== L4  low-occupancy TDM a2a ==================="
# WHY THIS EXISTS. On 2026-08-04 every check above passed -- CU copy 1640.8 against a baseline of
# 1641.0, a2a grid=512 at 1617.7 -- while mori's combine was running 18% slow on the same node and
# the same commit that had produced its reference figure hours earlier. The node was idle, no
# neighbour held a GPU, and the regression was reproducible to within 0.5% across four runs.
#
# What L0-L3 miss is a matter of where they sample. Three of the four checks are two-process point
# to point, and the a2a one runs grid=512. All of them sit on the saturated part of the bandwidth
# curve, where per-warp throughput does not matter because concurrency covers for it. Measured that
# day with epsim (16KB tile, 243MB working set), against the same tool's historical curve:
#
#   grid      32      64      96     128     256
#   healthy 1346    1748    1773    1766    1604
#   that day 896.1  1408.5  1520.3  1545.7  1541.0
#   delta    -33%    -19%    -14%   -12.5%   -3.9%
#
# The deficit closes as grid rises: at 256 it is within noise, at 64 it is 19%. Fixing grid=64 and
# sweeping op size located it further -- 512B ops were normal (72.3 vs 74.8, inside the ~3.5% run
# to run spread) while 16KB+ ops were uniformly 11-13% down. Fixed per-op cost intact, asymptotic
# bandwidth down: it is PER-WARP THROUGHPUT that fell, and only low concurrency exposes it.
#
# That is exactly where mori's combine lives. Its geometry is pinned at 64 blocks because the rest
# of the CUs belong to the GEMM it overlaps with, so it cannot buy the loss back with concurrency
# the way a benchmark can. A gate that only samples the saturated part of the curve will keep
# certifying a node as healthy while every collective on it runs 18% slow.
#
# BASELINES ARE DELIBERATELY EMPTY. They must be recorded on a node known to be good, and the node
# this was written on was not: it was mid-regression, so its numbers would enshrine the fault as
# the reference. epcheck has also never run grid=64 at all, so there is no historical figure to
# fall back on. Until filled, this section reports and skips -- it cannot fail, and the exit code
# keeps its old meaning. Fill them and it becomes a gate automatically.
#
# TO CALIBRATE: run this on a node that passes L0-L3 *and* whose epsim mode0 GRIDS=64 BLOCK=256
# reads in its healthy band (1545-1610 with default NT; see HANDOFF §10 and the recovery ladder).
# Both conditions matter -- L0-L3 passing is precisely what failed to detect this.
BL_A2A_G64=""      # tdma2a, 4 rank, grid=64,  aggregate incl. self-write
BL_A2A_G128=""     # tdma2a, 4 rank, grid=128, aggregate incl. self-write
BL_SAT64=""        # BW(grid=64) / BW(grid=512), a shape metric -- see below

# Same report format as chk(), but tolerates an unset baseline instead of inventing one. An unset
# check is a skip, never a pass: printing PASS for something nobody has measured is worse than
# printing nothing, because it is the reassuring output that gets trusted.
chk_opt(){
  local n="$1" m="$2" b="$3" lo hi ok
  if [ -z "$m" ]; then printf "  %-26s MISSING (could not parse output)\n" "$n"; skip=$((skip+1)); return; fi
  if [ -z "$b" ]; then
    printf "  %-26s %8s   baseline    UNSET   (record on a known-good node)\n" "$n" "$m"
    skip=$((skip+1)); return
  fi
  lo=$(awk -v b="$b" -v t="$TOL" 'BEGIN{printf "%.1f", b*(1-t/100)}')
  hi=$(awk -v b="$b" -v t="$TOL" 'BEGIN{printf "%.1f", b*(1+t/100)}')
  ok=$(awk -v m="$m" -v lo="$lo" -v hi="$hi" 'BEGIN{print (m>=lo && m<=hi)?"PASS":"FAIL"}')
  printf "  %-26s %8s   baseline %8s   band [%s, %s]   %s\n" "$n" "$m" "$b" "$lo" "$hi" "$ok"
  if [ "$ok" = PASS ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
}

if [ "$NG" -ge 4 ]; then
  # Only the grids L2 does not already cover. Same tile count and iteration count as L2 so the two
  # sections' numbers sit on one curve; grid=512 is reused from L2 rather than re-measured.
  A2A_NR=4 TDM_NUM_TILES=4096 TDM_ITER=50 TDM_GRIDS="64 128" \
    "$WORK/tdma2a" >"$WORK/a2a_low.log" 2>&1
  sed -n '/grid=/s/^/  /p' "$WORK/a2a_low.log"
  G64=$(sed -nE 's@.*grid=64 +: +([0-9.]+) GB/s evt.*@\1@p' "$WORK/a2a_low.log" | head -1)
  G128=$(sed -nE 's@.*grid=128 +: +([0-9.]+) GB/s evt.*@\1@p' "$WORK/a2a_low.log" | head -1)
  chk_opt "a2a grid=64 aggregate"  "$G64"  "$BL_A2A_G64"
  chk_opt "a2a grid=128 aggregate" "$G128" "$BL_A2A_G128"

  # Absolute numbers move with tile count, working set and iteration count; this ratio does not,
  # which is what makes it worth having next to them. It asks a question the absolutes cannot:
  # does low concurrency reach what high concurrency reaches? A healthy node answers yes -- epsim's
  # historical curve peaked at grid=64 (1748 against a 1773 peak, 98.6%), whereas on the bad day it
  # only reached 91% and needed grid=128 to level off.
  #
  # The threshold is NOT transferable from epsim: tdma2a is a different kernel with a different
  # tile shape, and its healthy grid=64 may legitimately sit well below its grid=512. That is why
  # BL_SAT64 is empty rather than set to 0.9 -- a plausible-looking threshold that has never been
  # observed on a good node would just relocate the false confidence, not remove it.
  if [ -n "${A2A:-}" ] && [ -n "${G64:-}" ]; then
    SAT64=$(awk -v a="$G64" -v b="$A2A" 'BEGIN{printf "%.3f", a/b}')
    chk_opt "sat64 = g64/g512"      "$SAT64" "$BL_SAT64"
  else
    printf "  %-26s %8s   (grid=512 unavailable, cannot form ratio)\n" "sat64 = g64/g512" "-"
    skip=$((skip+1))
  fi
else
  echo "  SKIP: needs 4 GPUs, found $NG"; skip=$((skip+1))
fi

echo "=================== result ==================="
echo "  pass=$pass fail=$fail skip=$skip  (tolerance +-${TOL}%)"
if [ "$fail" -gt 0 ]; then echo "  EPCHECK FAIL"; exit 1; fi
echo "  EPCHECK PASS"
