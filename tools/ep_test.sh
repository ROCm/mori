#!/usr/bin/env bash
# EP intra-node dispatch/combine bench driver (A/B by gate set, "deletion method").
#
# Runs tests/python/ops/bench_dispatch_combine.py once per SPEC and prints one aligned row per spec
# with BOTH phases' latency and bandwidth.
#
# Run it INSIDE the container, from anywhere, with no arguments:
#   ./tools/ep_test.sh
# It finds the repo from its own path, so there is nothing to configure to get a first number. Every
# knob below is an optional override, and the defaults are the 64x8 bf16 EP4 PUSH case.
#
# Why deletion method rather than the in-kernel TIMING buckets: [CSPLIT] is per-warp max with no
# __syncthreads at either end, so it systematically understates a phase's wall clock (cPush reads
# 113us there against 231us by deletion) and the TIMING build is itself ~2x slower overall. Price
# phases by deleting them from a noTIMING build and diffing, never by reading the buckets.
#
# Optional overrides:
#   SPECS="tag=GATES; tag2=GATES2"   one run per spec; MORI_BENCH_SKIPCHECK=1 unless tag ends in '!'
#   tag ending in '!'                run this spec WITH the correctness check (rc=0 is the pass)
#   CBN/CWPB/DBN/DWPB                geometry; DBN/DWPB accept SAME to follow the combine values
#   CBNS="64 128 256"                sweep combine block count, running every spec at each
#   BASE                             gates shared by every spec
#   WS                               peer count
#   ZC                               0 = PUSH (_nop2p), 1 = PULL (_p2p, cross-card read)
#   QT                               none / fp8_blockwise / fp8_direct_cast
# Examples:
#   ./tools/ep_test.sh
#   ZC=1 ./tools/ep_test.sh
#   SPECS="full=; nopush=MORI_COMB_NOPUSH=1" CBNS="64 128" ./tools/ep_test.sh
#   SPECS="check!=" ZC=1 ./tools/ep_test.sh
# From the node, without giving this script any docker knowledge of its own:
#   docker exec MORI-EPV2 bash -lc 'cd /root/mori_tdm && ./tools/ep_test.sh'
set -uo pipefail
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SRC" || exit 1
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

# Printed so a number is attributable to a tree. Deploying by copying files in used to make this
# unknowable, and it cost a wrong conclusion: a recheck once compiled a file the copy had never
# reached and printed a log identical to the pre-fix one, which reads as "the fix does not work"
# rather than "the fix is not here". Checking out a rev is git's job, not this script's; a dirty
# tree still runs, it just says so.
if git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1; then
  printf '## HEAD %s  %s\n' "$(git -C "$SRC" rev-parse --short HEAD)" \
    "$(git -C "$SRC" log -1 --format=%s | cut -c1-72)"
  d=$(git -C "$SRC" status --porcelain -- src include python tests)
  [ -n "$d" ] && { echo '## DIRTY, not the commit above:'; echo "$d" | head -12 | sed 's/^/##   /'; }
fi

# Bootstrap only: loopback interface, rendezvous address, heap size, unbuffered stdout. Nothing that
# changes kernel behaviour is exported, and that is the point. The five that used to sit here are
# gone for cause -- MORI_DISP_BATCH was read by nothing at all, MORI_DISP_TDM is implied by the arch,
# MORI_COMB_TDM=1 was not "on" but an override of the resolved default of 2 chunks, MORI_COMB_RUNRR
# is now the only push order, and MORI_EP_COMM=cco is the gfx125x default. Each was a fact about the
# hardware, or a settled choice, spelled as something the caller had to remember.
export MORI_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo PYTHONPATH="$SRC"
export MASTER_ADDR=127.0.0.1 MORI_SHMEM_HEAP_SIZE=6G PYTHONUNBUFFERED=1
HN=$(hostname)
grep -q "$HN" /etc/hosts 2>/dev/null || { echo "127.0.0.1 $HN" >> /etc/hosts; } 2>/dev/null ||
  echo "## warn: '$HN' is not in /etc/hosts and it is not writable, rendezvous may hang"

idle() {
  local u=""
  for _ in $(seq 1 40); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -1)
    # No figure at all is a broken probe, not a busy GPU. Waiting out all 40 rounds on it burns two
    # minutes per spec and then aborts with "VRAM=" and nothing after it, which says neither.
    [ -z "$u" ] && { echo "## warn: rocm-smi gave no VRAM figure, skipping the idle wait"; return 0; }
    [ "$u" -lt 400000000 ] && return 0
    sleep 3
  done
  echo "ABORT: $u bytes of VRAM still in use after 120s"; exit 1
}

P=$(( 37000 + RANDOM % 900 ))
run() { # $1=tag  $2=gates  $3=1 means run WITH the correctness check
  pkill -9 -f bench_dispatch_combine 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null
  sleep 4; idle
  P=$((P+1))
  local sk=MORI_BENCH_SKIPCHECK=1; [ "${3:-0}" = 1 ] && sk=MORI_BENCH_SKIPCHECK=0
  local db=$DBN; [ "$db" = SAME ] && db=$CB
  local dw=$DWPB; [ "$dw" = SAME ] && dw=$CWPB
  local log=/tmp/ep_test_$1.log
  env $sk $BASE $2 MASTER_PORT=$P \
    timeout 900 python3 tests/python/ops/bench_dispatch_combine.py \
    --cmd bench --world-size "$WS" --dtype bf16 --max-tokens 4096 --hidden-dim 7168 \
    --num-experts-per-rank 64 --num-experts-per-token 8 --zero-copy "$ZC" --quant-type "$QT" \
    --dispatch-block-num "$db" --dispatch-warp-per-block "$dw" \
    --combine-block-num "$CB" --combine-warp-per-block "$CWPB" > "$log" 2>&1
  local rc=$?
  # Take the bench's OWN bw field. This used to recompute 202.47MB/lat, which understated every
  # number this script ever printed by exactly 4.86%: the bench used to print that payload through
  # 1024**2 while labelling it MB, so 202.47 was MiB and the real figure is 212.3 MB -- the same
  # bytes, and 1048576/1000000 = 1.048576 is the whole discrepancy. Both phases move the same
  # total_recv_num_token x hidden x elemsize, reconstructing to recv ~14813 against the 14804 the
  # topk draw predicts, so there is no missing-token puzzle here either.
  avg() { # $1=phase section header  $2=field name
    awk -v sec="^$1 result" -v fld="$2" \
      '$0~sec{c=1;next} /^Dispatch result|^Combine result|^End-to-end/{c=0}
       /^Round [0-9]+ duration/&&c{n++;for(i=1;i<=NF;i++)if($i==fld)s+=$(i+1)}
       END{if(n)printf "%.1f",s/n; else print "NA"}' "$log"
  }
  printf "  %-18s rc=%-3s combine=%8s us (%7s GB/s)  disp=%7s us (%7s GB/s)\n" \
    "$1" "$rc" "$(avg Combine lat)" "$(avg Combine bw)" "$(avg Dispatch lat)" "$(avg Dispatch bw)"
  grep -oE "AssertionError|Memory access fault.*|HSA_STATUS[A-Z_]*|LDS.*exceed.*|out of memory|invalid configuration" "$log" | head -2 | sed 's/^/       /'
  grep -oE "\[BENCHW\].*|Weight mismatch for token [0-9]+" "$log" | head -3 | sed 's/^/       /'
}

echo "## geometry comb <cbn>x${CWPB}  disp ${DBN}x${DWPB}  WS=$WS  ZC=$ZC  QT=$QT  BASE='$BASE'"
for CB in ${CBNS:-$CBN}; do
  echo "## --- combine block = $CB ---"
  OLDIFS=$IFS; IFS=';'
  for sp in $SPECS; do
    IFS=$OLDIFS
    sp="$(echo "$sp" | sed 's/^ *//;s/ *$//')"
    [ -z "$sp" ] && { IFS=';'; continue; }
    tag="${sp%%=*}"; gates="${sp#*=}"
    chk=0; case "$tag" in *!) chk=1; tag="${tag%!}";; esac
    run "${tag}_b$CB" "$gates" "$chk"
    IFS=';'
  done
  IFS=$OLDIFS
done
echo EP_TEST_DONE
