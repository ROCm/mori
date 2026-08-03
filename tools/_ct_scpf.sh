#!/usr/bin/env bash
# First run of the scale-row prefetch on real hardware. CHECK BEFORE TIMING, in that order and in
# that many rows: the prefetch changes which memory a scale comes from, and the microbench that
# motivated it could not have caught a wrong-entry bug (its scale row was constant, so reading the
# wrong entry of the right row still produced the right answer). Its row varies now, but this is the
# kernel and the bench's own comparison is the thing that decides.
#
# The reference to beat is bf16 zero-copy PULL at this geometry, 169.2us / 1254.7 GB/s.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
date -u +'START %H:%M:%S'

# 1. Correctness, both blockwise transports, at the geometry the complaint is about. Tags end in '!'
#    so the bench runs its comparison; rc=0 is the only pass.
echo "== correctness =="
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='bwq!=; bwq_pipe!=MORI_COMB_PIPE=2' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '

# 2. The same two rows with the prefetch disabled, to price it rather than assume it. There is no
#    env gate for that, so QNOSC -- which folds with a scale of 1 -- is the closest available floor
#    and is WRONG BY CONSTRUCTION; it brackets the prefetch rather than measuring it.
echo "== price =="
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='bwq=; nosc=MORI_COMB_QNOSC=1' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '

# 3. bf16 at the same geometry, the number quantising has to beat.
echo "== bf16 reference =="
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='bf16!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '

date -u +'END %H:%M:%S'
echo "SCPF_DONE"
