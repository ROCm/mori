#!/usr/bin/env bash
# Where the blockwise quantise pass loses its 2.6x, priced in the standalone pass rather than as a
# difference between two runs of the whole combine.
#
# WHY THIS AND NOT THE KERNEL. The pass is the larger of the two halves of the quantised combine and
# it is the one with a hard floor: 319 MB of local traffic at the 6.3 TB/s a plain d2d copy gets is
# 51us, and the best figure on record is 134. Every candidate below is a compile-time -D, which in
# the real kernel costs a ~890s JIT rebuild each; here a build is seconds, so the whole grid is
# cheaper than one kernel rebuild.
#
# THE FOUR QUESTIONS, in the order the modes answer them:
#   mode 2 (cast) vs mode 0 (quant)  -- is it the block-max reduction, or the streaming itself
#   mode 1 (scales to LDS) vs mode 0 -- is it the 224 B per token going to uncached memory
#   modes 3/4 (read alone, write alone) -- if 3+4 is already 134 the pass is at ITS floor and the
#                                         floor is not 51, and the target has to move to the gather
#   Q_UNC=0 vs 2                     -- does the allocator cost anything once the geometry is not
#                                       starved. The earlier answer (720.5 vs 720.4, no difference)
#                                       was taken at 64x8, where the pass is latency-bound and any
#                                       bandwidth effect is hidden -- so it settled nothing.
#
# Then QSTGU (loads in flight) and QSCW (how the scales are stored) at whatever geometry wins,
# because both were tuned at 64x8 for the same reason and neither has been seen anywhere else.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"
G="${G:-256 512 1024}"
W="${W:-8 16}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"

build() { # $1 = tag, rest = extra -D
  local tag=$1; shift
  docker exec "$CTR" bash -lc "/opt/rocm/bin/hipcc --offload-arch=gfx1250 -std=c++17 -O3 \
    -I $SRC/include -I $SRC $* $SRC/tools/q_micro.cc -o /tmp/q_$tag 2>&1 | grep -E 'error' | head -8; \
    ls -l /tmp/q_$tag >/dev/null && echo BUILT_$tag"
}
run() { # $1 = binary tag, $2 = env
  echo "########## $1  $2"
  docker exec "$CTR" bash -lc "$2 Q_GRIDS='$G' Q_WPB='$W' timeout 900 /tmp/q_$1" 2>&1
}

build base
run base "Q_MODES='0 1 2 3 4' Q_UNC=2"
run base "Q_MODES='0 2' Q_UNC=0"

# QSTGU was measured 1/4/7 inside the fused kernel, where the LDS reservation held a CU to two waves
# per SIMD. With no LDS the depth that hides the loads is a different number, and 7 is also the
# point where the exact fit runs out, so higher depths fall into the general loop -- which is why 14
# and 28 are here and not 8 and 16.
for d in 1 4 7 14 28; do
  build stg$d -DMORI_COMB_QSTGU=$d
  run stg$d "Q_MODES=0 Q_UNC=2"
done

# QSCW: 0 = a 4 B store per subwarp per block, 1 = one grouped store per group of blocks,
# 2 = DIAGNOSTIC, do not store the scales at all. WRONG RESULTS on 2; it is the deletion price.
for s in 0 1 2; do
  build scw$s -DMORI_COMB_QSCW=$s
  run scw$s "Q_MODES=0 Q_UNC=2 Q_CHECK=$([ $s = 2 ] && echo 0 || echo 1)"
done
echo "QQUANT_DONE"
