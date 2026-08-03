#!/usr/bin/env bash
# Which scale-broadcast strategy the fp8 fold should use, decided in the standalone gather rather
# than in the kernel. This binary builds in seconds; adding one -D to the kernel costs a ~890s JIT
# rebuild, so every strategy that can be rejected here is 15 minutes not spent.
#
# The modes, and what separates them (see the header of g_micro.cc):
#   1  a scale load per source per vector, straight off the peer  -- what shipped
#   2  the row into LDS once per token, read from LDS
#   5  the row into REGISTERS once per token, read by shuffle     -- what the kernel now does
#   6  no fetching scheme at all: 4 elements per lane instead of 8, so a wave covers exactly one
#      scale block, sb is wave-uniform and the address is scalar
#   3  the transport floor, fold deleted -- WRONG OUTPUT BY CONSTRUCTION
#   bf16 at the same points, moving twice the bytes, for the thing this has to beat
#
# Mode 6 is the one worth watching. Modes 2, 4 and 5 all accept that sb varies across the wave and
# spend something to distribute a value; 6 removes the divergence instead, by changing the fold's
# own lane mapping so a wave covers one block. If it wins, the register prefetch in the kernel is
# solving a problem the kernel created.
#
# The check is armed (G_CHECK=1, the default) and its scale row now VARIES per entry -- it used to
# be a constant, which meant reading the wrong entry of the right row still passed, and mode 5
# shipped with exactly that bug. A row that says "ok" is a row whose entry indexing is right.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "/opt/rocm/bin/hipcc --offload-arch=gfx1250 -std=c++17 -O3 \
  -I $SRC/include -I $SRC $SRC/tools/g_micro.cc -o /tmp/g_micro 2>&1 | grep -E 'error|warning: unused' | head -12; \
  ls -l /tmp/g_micro | tail -1"

G="${G:-64 128 256}"
W="${W:-8 16}"
C="${C:-1792 3584}"
for d in 1 2 5 6 3; do
  echo "########## fp8 DEQ=$d"
  docker exec "$CTR" bash -lc "G_BYTES=1 G_PIPE=0 G_DEQ=$d G_GRIDS='$G' G_WPB='$W' G_CHUNKS='$C' \
    G_TOK=4096 G_ITERS=20 timeout 600 /tmp/g_micro" 2>&1
done
echo "########## bf16 reference"
docker exec "$CTR" bash -lc "G_BYTES=2 G_PIPE=0 G_DEQ=0 G_GRIDS='$G' G_WPB='$W' G_CHUNKS='$C' \
  G_TOK=4096 G_ITERS=20 timeout 600 /tmp/g_micro" 2>&1
echo "GSCALE_DONE"
