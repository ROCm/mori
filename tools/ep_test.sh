#!/usr/bin/env bash
# EP intra-node dispatch/combine bench driver (A/B by gate set, "deletion method").
#
# Runs tests/python/ops/bench_dispatch_combine.py inside a container, once per SPEC, and prints one
# aligned row per spec with BOTH phases' latency and bandwidth. Meant to be run ON THE NODE (it
# drives docker), not inside the container.
#
# Why deletion method rather than the in-kernel TIMING buckets: [CSPLIT] is per-warp max with no
# __syncthreads at either end, so it systematically understates a phase's wall clock (cPush reads
# 113us there against 231us by deletion) and the TIMING build is itself ~2x slower overall. Price
# phases by deleting them from a noTIMING build and diffing, never by reading the buckets.
#
# Usage:
#   SPECS="tag=GATES; tag2=GATES2"   one run per spec; MORI_BENCH_SKIPCHECK=1 unless tag ends in '!'
#   tag ending in '!'                run this spec WITH the correctness check (rc=0 is the pass)
#   CBN/CWPB/DBN/DWPB                geometry; DBN/DWPB accept SAME to follow the combine values
#   CBNS="64 128 256"                sweep combine block count, running every spec at each
#   BASE                             gates shared by every spec
#   ZC                               0 = PUSH (_nop2p), 1 = PULL (_p2p, cross-card read)
#   QT                               none / fp8_blockwise / fp8_direct_cast
#   REV=<branch|commit>              fetch and hard-checkout that rev in the container before running;
#                                    empty (default) runs whatever is already checked out there
# Examples:
#   REV=tdm-dispatch ./tools/ep_test.sh
#   ./tools/ep_test.sh
#   SPECS="full=; nopush=MORI_COMB_NOPUSH=1" CBNS="64 128" ./tools/ep_test.sh
#   SPECS="check!=" ZC=1 ./tools/ep_test.sh
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
CBN="${CBN:-64}"        # single value; use CBNS to sweep
CBNS="${CBNS:-}"        # non-empty: run every spec once per block count (diagnostic; default is 64x8)
CWPB="${CWPB:-8}"
DBN="${DBN:-64}"        # SAME = follow the combine block count being swept (per-row geometry)
DWPB="${DWPB:-8}"       # SAME = follow CWPB
WS="${WS:-4}"           # peer count; 2 is the case where round-robin has the least imbalance to recover
ZC="${ZC:-0}"           # 0 = PUSH (_nop2p), 1 = PULL (_p2p, cross-card read)
QT="${QT:-none}"        # none / fp8_blockwise / fp8_direct_cast; blockwise only pairs with ZC=0
# BARSLEEP is deliberately NOT set here. It used to default to 127 in this script, which silently
# overrode the library's own default of 15 (jit/core.py) and cost 2.2us on every PULL reading:
# 127 was tuned on the PUSH path where the barrier is ~58us, but QUAD PULL waits only 7.6us, so the
# backoff oversleeps. MEASURED 64x8 ZC=1 bf16 EP4, check armed: nothing set 169.4us / 1195 GB/s,
# RUNRR alone 168.9 / 1199, RUNRR+BARSLEEP=127 171.1 / 1183. Passing it explicitly here made every
# geometry in this script read ~2us slow and produced a phantom "regression" against the recorded
# 168.9 / 1199 baseline. Set it per-spec if a run is actually studying the barrier.
# BASE is empty on purpose. It used to carry MORI_COMB_RUNRR=1, which was worth 22% on PUSH and so
# looked mandatory; the push loop now always uses that ordering (the queued variant, measured better
# at both EP2 and EP4), so there is nothing left to pass. Anything set here applies to every spec.
BASE="${BASE:-}"
SPECS="${SPECS:-full=}"
REV="${REV:-}"

# Source comes from git, never from a docker cp of the working tree. Copying had two silent failure
# modes and both fired: a stage step that landed files in the node's /tmp without entering the
# container produced a gfx942 compile log identical to the pre-fix one (the fix was never read), and
# the copy list was a fixed set that did not include launch.cpp, so editing it deployed nothing. Both
# read as "the change did not work" rather than "the change never arrived". A rev is verifiable on
# both ends, and carries whatever was touched.
echo "######## source"
docker exec "$CTR" bash -lc "cd $SRC || exit 1
if [ -n '$REV' ]; then
  timeout 120 git fetch --quiet origin 2>/dev/null || echo '  fetch failed (no network/key), using local objects'
  git checkout -f --detach '$REV' >/dev/null 2>&1 || git checkout -f --detach 'origin/$REV' >/dev/null 2>&1 || {
    echo '  no such rev: $REV'; exit 1; }
fi
printf '  HEAD %s  %s\n' \"\$(git rev-parse --short HEAD)\" \"\$(git log -1 --format=%s | cut -c1-72)\"
# Printed unconditionally, and this is the point of the block: a result is only attributable if the
# run says which tree produced it. A dirty tree is not fatal (it is how you test an unpushed idea),
# it just means the rev above is not what ran.
d=\$(git status --porcelain -- src include python tests)
if [ -n \"\$d\" ]; then
  echo '  DIRTY, the run below is NOT the commit above:'
  echo \"\$d\" | head -12 | sed 's/^/    /'
fi" || exit 1
# No JIT cache drop after a checkout: the cache key hashes the content of src/ops and include/mori
# (jit/cache.py), so a different tree already resolves to a different cache dir on its own.

docker exec -i "$CTR" bash -lc 'cat > /tmp/ep_test_inner.sh' <<INNER
#!/bin/bash
cd $SRC || exit 1
# Nothing that changes kernel behaviour is exported here any more, and that is the point: what is
# left is bootstrap only (loopback interface, rendezvous address, heap size, unbuffered stdout).
# The five that used to sit here are gone for cause -- MORI_DISP_BATCH was read by nothing at all,
# MORI_DISP_TDM is implied by the arch, MORI_COMB_TDM=1 was not "on" but an override of the resolved
# default of 2 chunks, MORI_COMB_RUNRR is now the only push order, and MORI_EP_COMM=cco is the
# gfx125x default. Each of them was a fact about the hardware or a settled choice being spelled as
# something the caller had to remember.
export MORI_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo PYTHONPATH=$SRC
export MASTER_ADDR=127.0.0.1 MORI_SHMEM_HEAP_SIZE=6G PYTHONUNBUFFERED=1
HN=\$(hostname); grep -q "\$HN" /etc/hosts || echo "127.0.0.1 \$HN" >> /etc/hosts
idle() {
  for _ in \$(seq 1 40); do
    u=\$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -1)
    [ -n "\$u" ] && [ "\$u" -lt 400000000 ] && return 0; sleep 3
  done
  echo "ABORT VRAM=\$u"; exit 1
}
P=\$(( 37000 + RANDOM % 900 ))
CB=$CBN
run() { # \$1=tag \$2=gates \$3=1 means run WITH the correctness check
  pkill -9 -f bench_dispatch_combine 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null; sleep 4; idle
  P=\$((P+1))
  sk=MORI_BENCH_SKIPCHECK=1; [ "\${3:-0}" = 1 ] && sk=MORI_BENCH_SKIPCHECK=0
  db=$DBN; [ "\$db" = SAME ] && db=\$CB
  dw=$DWPB; [ "\$dw" = SAME ] && dw=$CWPB
  env \$sk $BASE \$2 MASTER_PORT=\$P \
    timeout 900 python3 tests/python/ops/bench_dispatch_combine.py \
    --cmd bench --world-size $WS --dtype bf16 --max-tokens 4096 --hidden-dim 7168 \
    --num-experts-per-rank 64 --num-experts-per-token 8 --zero-copy $ZC --quant-type $QT \
    --dispatch-block-num \$db --dispatch-warp-per-block \$dw \
    --combine-block-num \$CB --combine-warp-per-block $CWPB > /tmp/ep_test_\$1.log 2>&1
  rc=\$?
  cmb=\$(awk '/^Combine result/{c=1;next} /^Dispatch result|^End-to-end/{c=0}
              /^Round [0-9]+ duration/&&c{n++;for(i=1;i<=NF;i++)if(\$i=="lat")s+=\$(i+1)}
              END{if(n)printf "%.1f",s/n; else print "NA"}' /tmp/ep_test_\$1.log)
  dsp=\$(awk '/^Dispatch result/{c=1;next} /^Combine result|^End-to-end/{c=0}
              /^Round [0-9]+ duration/&&c{n++;for(i=1;i<=NF;i++)if(\$i=="lat")s+=\$(i+1)}
              END{if(n)printf "%.1f",s/n; else print "NA"}' /tmp/ep_test_\$1.log)
  # Take the bench's OWN bw field. This used to recompute 202.47MB/lat, which understated every
  # number this script ever printed by exactly 4.86%: the bench used to print that payload through
  # 1024**2 while labelling it MB, so 202.47 was MiB and the real figure is 212.3 MB -- the same
  # bytes, and 1048576/1000000 = 1.048576 is the whole discrepancy. Both phases move the same
  # total_recv_num_token x hidden x elemsize, reconstructing to recv ~14813 against the 14804 the
  # topk draw predicts, so there is no missing-token puzzle here either.
  cgb=\$(awk '/^Combine result/{c=1;next} /^Dispatch result|^End-to-end/{c=0}
              /^Round [0-9]+ duration/&&c{n++;for(i=1;i<=NF;i++)if(\$i=="bw")s+=\$(i+1)}
              END{if(n)printf "%.1f",s/n; else print "NA"}' /tmp/ep_test_\$1.log)
  dgb=\$(awk '/^Dispatch result/{c=1;next} /^Combine result|^End-to-end/{c=0}
              /^Round [0-9]+ duration/&&c{n++;for(i=1;i<=NF;i++)if(\$i=="bw")s+=\$(i+1)}
              END{if(n)printf "%.1f",s/n; else print "NA"}' /tmp/ep_test_\$1.log)
  printf "  %-18s rc=%-3s combine=%8s us (%7s GB/s)  disp=%7s us (%7s GB/s)\n" \
    "\$1" "\$rc" "\$cmb" "\$cgb" "\$dsp" "\$dgb"
  grep -oE "AssertionError|Memory access fault.*|HSA_STATUS[A-Z_]*|LDS.*exceed.*|out of memory|invalid configuration" /tmp/ep_test_\$1.log | head -2 | sed 's/^/       /'
  grep -oE "\[BENCHW\].*|Weight mismatch for token [0-9]+" /tmp/ep_test_\$1.log | head -3 | sed 's/^/       /'
}
echo "## geometry comb <cbn>x${CWPB}  disp ${DBN}x${DWPB}  ZC=$ZC  QT=$QT  BASE='$BASE'"
SPECS='$SPECS'
for CB in ${CBNS:-$CBN}; do
  echo "## --- combine block = \$CB ---"
  OLDIFS=\$IFS; IFS=';'
  for sp in \$SPECS; do
    IFS=\$OLDIFS
    sp="\$(echo \$sp | sed 's/^ *//;s/ *\$//')"
    [ -z "\$sp" ] && { IFS=';'; continue; }
    tag="\${sp%%=*}"; gates="\${sp#*=}"
    chk=0; case "\$tag" in *!) chk=1; tag="\${tag%!}";; esac
    run "\${tag}_b\$CB" "\$gates" "\$chk"
    IFS=';'
  done
  IFS=\$OLDIFS
done
echo EP_TEST_DONE
INNER
docker exec "$CTR" bash -lc 'sed -i "s/\r$//" /tmp/ep_test_inner.sh; bash /tmp/ep_test_inner.sh'
