#!/usr/bin/env bash
# EP intra-node dispatch/combine bench driver (A/B by gate set, "deletion method").
#
# Drives tests/python/ops/bench_dispatch_combine.py and prints a token sweep, three columns wide:
#
#   tokens/rank   dispatch                combine PUSH ZC=0       combine PULL ZC=1
#                 GB/s (lat us)           GB/s (lat us)           GB/s (lat us)
#            64   38.3 (90.3)             97.7 (33.7)             126.6 (26.0)
#           ...
#
# Two runs per row, one per transport; dispatch is read off the PUSH run because ZC does not touch
# that phase. Nine token counts by default, so a default invocation is ~18 bench runs and takes
# roughly a quarter of an hour -- TOKS=4096 for a single row.
#
# Run it INSIDE the container, from anywhere, with no arguments:
#   ./tools/ep_test.sh
# It finds the repo from its own path, so there is nothing to configure to get a first number. Every
# knob below is an optional override, and with none of them set this measures WHAT SHIPS: bf16 EP4
# at whatever geometry the library picks for itself.
#
# Why deletion method rather than the in-kernel TIMING buckets: [CSPLIT] is per-warp max with no
# __syncthreads at either end, so it systematically understates a phase's wall clock (cPush reads
# 113us there against 231us by deletion) and the TIMING build is itself ~2x slower overall. Price
# phases by deleting them from a noTIMING build and diffing, never by reading the buckets.
#
# Optional overrides:
#   SPECS="tag=GATES; tag2=GATES2"   one run per spec; MORI_BENCH_SKIPCHECK=1 unless tag ends in '!'
#   tag ending in '!'                run this spec WITH the correctness check (rc=0 is the pass)
#   CBN/CWPB/DBN/DWPB                geometry; unset sends 0, which is how you ask the library for
#                                    its own per-body default, i.e. what ships (see run() below).
#                                    DBN/DWPB accept SAME to follow the combine values
#   CBNS="64 128 256"                sweep combine block count, running every spec at each
#   BASE                             gates shared by every spec
#   WS                               peer count
#   TOKS="4096 8192" / MAXTOK=4096   which token counts make up the rows
#   ZCS                              which combine columns to fill: "0 1" (default), "0", or "1"
#   QT                               none / fp8_blockwise / fp8_direct_cast
# Examples:
#   ./tools/ep_test.sh
#   TOKS=4096 ./tools/ep_test.sh
#   ZCS=1 ./tools/ep_test.sh
#   SPECS="full=; nopush=MORI_COMB_NOPUSH=1" CBNS="64 128" ./tools/ep_test.sh
#   SPECS="check!=" TOKS="4096 16384" ./tools/ep_test.sh
# From the node, without giving this script any docker knowledge of its own:
#   docker exec MORI-EPV2 bash -lc 'cd /root/mori_tdm && ./tools/ep_test.sh'
set -uo pipefail
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SRC" || exit 1

# ONE of these at a time on a node, enforced rather than remembered. Two overlapping runs put two
# four-rank jobs on the same four GPUs; each spins on its own cross-device barrier, neither can be
# scheduled to let the other finish, every rank lands in D state where SIGKILL does not reach it,
# and the recovery is docker stop plus a per-device rocm-smi --gpureset. That happened because a
# remote `timeout` killed the ssh-side bash and left THIS loop running inside the container, and
# the next script started anyway. The per-run pkill below cannot help: by then both loops are live
# and they take turns killing each other's bench. It cost two hours and a node reboot.
if command -v flock >/dev/null 2>&1; then
  exec 9>/tmp/ep_test.lock
  if ! flock -n 9; then
    echo "ABORT: another ep_test.sh holds /tmp/ep_test.lock (pid $(cat /tmp/ep_test.pid 2>/dev/null))"
    exit 1
  fi
  echo $$ > /tmp/ep_test.pid
fi
# Geometry. Empty means no opinion, which run() spells as an explicit 0 so _resolve_launch_params
# falls through to the per-body default, which IS the shipping configuration. This used to force 64x8 on
# both phases, and because an explicit value outranks the per-body default, the headline number was
# a geometry that ships to nobody: gfx125x combine defaults to 64x16 for PUSH (64x8 only for PULL,
# where 16 warps want 458KB against a 320KB budget), and every non-gfx125x arch defaults to 256x16
# for dispatch. Forcing 8 warps on a PUSH run reads 234.7us where the shipping default reads ~193.
# Deferring also keeps the ZC=0/ZC=1 width pairing out of the caller's memory, which is where the
# rest of this script tries to keep such things.
CBN="${CBN:-}"          # single value; use CBNS to sweep
CBNS="${CBNS:-}"        # non-empty: run every spec once per block count (diagnostic)
CWPB="${CWPB:-}"
DBN="${DBN:-}"          # SAME = follow the combine block count being swept (per-row geometry)
DWPB="${DWPB:-}"        # SAME = follow CWPB
g() { [ -n "$1" ] && printf '%s' "$1" || printf 'lib'; }   # 'lib' = deferred to the library
WS="${WS:-4}"           # peer count; 2 is the case where round-robin has the least imbalance to recover
# Which combine transports to put in the table. Both by default, because the interesting quantity is
# the gap between them, not either alone. ZC is who owns the combine input buffer: 0 = the caller
# (PUSH, _nop2p), 1 = mori (PULL, _p2p, cross-card read). It means exactly "PUSH or PULL" again --
# for a while on gfx125x it did not, because ZC=0 staged the caller-owned buffer into the registered
# one and pulled anyway, and MORI_COMB_PULL=off was how you got the PUSH transport back. That route
# and the knob are gone (bed30dcf). ZC=0 or ZC=1 alone still works and leaves the other column empty.
ZCS="${ZCS:-${ZC:-0 1}}"
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
# Tokens per rank, swept. The whole curve is the deliverable: both transports change rank with size
# (PULL wins throughout, and PUSH used to collapse past 8192 until the tile fix), so a single point
# invites reading a crossover that is not there. 4096 is the point every older recorded number on
# this tree was taken at. MAXTOK=<n> still pins it to one column's worth of work.
# The headline GB/s stays honest across the sweep because the printer takes the bench's OWN bw field
# rather than dividing a hardcoded 212.3 MB, and that payload is proportional to the token count.
TOKS="${TOKS:-${MAXTOK:-64 128 256 512 1024 2048 4096 8192 16384}}"

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
# The package is python/mori, so $SRC alone does NOT put it on the path: with the tree also editable
# -installed elsewhere, `import mori` then resolves to that other tree and the "## HEAD" line above
# attributes the numbers to a commit that never ran. That is how a post-fix reading of 687.9us once
# passed for a pre-fix baseline. $SRC stays on the path behind it for the bench's own imports.
export MORI_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo PYTHONPATH="$SRC/python:$SRC"
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

# A breadcrumb that survives the node, written before each spec and closed out after it. Two
# gfx1250 boxes have now stopped answering ssh in the middle of a sweep, and in both cases the only
# record of which spec was running lived in /tmp on the machine that was gone -- so "what kills the
# node" is still unknown after paying for it twice. This file is inside the container's own
# filesystem, which survives a host reboot, and a line without a matching "done" names the suspect.
CRUMB="$SRC/.ep_test_last"
crumb() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$CRUMB" 2>/dev/null; sync 2>/dev/null; }
crumb "sweep start geometry=$(g "${CBNS:-$CBN}")x$(g "$CWPB") WS=$WS ZCS='$ZCS' QT=$QT toks='$TOKS'"

P=$(( 37000 + RANDOM % 900 ))
# Leaves the reading in R_RC / R_CLAT / R_CBW / R_DLAT / R_DBW rather than printing it, because one
# table row is assembled from two runs (PUSH and PULL) and cannot be printed until both are in.
run() { # $1=tag  $2=gates  $3=1 means run WITH the correctness check  $4=tokens  $5=zero-copy
  pkill -9 -f bench_dispatch_combine 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null
  sleep 4; idle
  crumb "run  $1  gates='$2'"
  P=$((P+1))
  local sk=MORI_BENCH_SKIPCHECK=1; [ "${3:-0}" = 1 ] && sk=MORI_BENCH_SKIPCHECK=0
  local db=$DBN; [ "$db" = SAME ] && db=$CB
  local dw=$DWPB; [ "$dw" = SAME ] && dw=$CWPB
  # "No opinion" is the literal value 0, NOT an absent flag. _resolve_launch_params treats bn/wpb
  # <= 0 as "apply the per-body default", which is the shipping geometry; but an ABSENT flag never
  # reaches it, because the bench substitutes _get_default_launch_config() first and then passes
  # that as an explicit value. That table is a gfx942/FlyDSL artefact and is wrong here twice over:
  # at WS<=4, maxtok>128 it hands ZC=0 a 768x8 dispatch with a 256x14 combine, and hands ZC=1 a
  # 192x32 dispatch -- 32 warps is 32*7168*2 = 458,752B of LDS against a 327,680B budget, so the
  # ZC=1 column does not read slow, it dies on hipModuleLaunchKernel with HIP error 1.
  local geo="--combine-block-num ${CB:-0} --combine-warp-per-block ${CWPB:-0}"
  geo="$geo --dispatch-block-num ${db:-0} --dispatch-warp-per-block ${dw:-0}"
  local log=/tmp/ep_test_$1.log
  env $sk $BASE $2 MASTER_PORT=$P \
    timeout 900 python3 tests/python/ops/bench_dispatch_combine.py \
    --cmd bench --world-size "$WS" --dtype bf16 --max-tokens "$4" --hidden-dim 7168 \
    --num-experts-per-rank 64 --num-experts-per-token 8 --zero-copy "$5" --quant-type "$QT" \
    $geo > "$log" 2>&1
  R_RC=$?
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
  R_CLAT=$(avg Combine lat); R_CBW=$(avg Combine bw)
  R_DLAT=$(avg Dispatch lat); R_DBW=$(avg Dispatch bw)
  crumb "done $1  rc=$R_RC"
  # Tagged and out of band so a clean sweep prints nothing but the table, and a dirty one still
  # says which cell went wrong.
  grep -oE "AssertionError|Memory access fault.*|HSA_STATUS[A-Z_]*|LDS.*exceed.*|out of memory|invalid configuration|dispatch metadata staging holds.*" "$log" | head -2 | sed "s|^|## $1: |"
  grep -oE "\[BENCHW\].*|Weight mismatch for token [0-9]+" "$log" | head -3 | sed "s|^|## $1: |"
}

cell() { # $1=bw $2=lat $3=rc -- one table cell. A failure has to be visible as a failure, not as a
         # plausible number, so it replaces the reading rather than sitting beside it.
  [ "${3:-0}" != 0 ] && { printf 'FAIL rc=%s' "$3"; return; }
  [ "$1" = NA ] && { printf 'no rounds parsed'; return; }
  printf '%s (%s)' "$1" "$2"
}
ROW='%11s   %-21s %-21s %-21s\n'

cbs="${CBNS:-$CBN}"; case "$cbs" in *' '*) cbs="[$cbs]";; esac   # a swept list, not one block count
echo "## geometry comb $(g "$cbs")x$(g "$CWPB")  disp $(g "$DBN")x$(g "$DWPB")" \
     " WS=$WS  QT=$QT  BASE='$BASE'   (lib = the library's own default, i.e. what ships)"
# A '-' placeholder so the loop still runs exactly once when no block count was asked for; an empty
# list would iterate zero times and the script would silently do nothing.
CBLIST="${CBNS:-$CBN}"; [ -z "$CBLIST" ] && CBLIST='-'
for CB in $CBLIST; do
  [ "$CB" = '-' ] && CB=''
  OLDIFS=$IFS; IFS=';'
  for sp in $SPECS; do
    IFS=$OLDIFS
    sp="$(echo "$sp" | sed 's/^ *//;s/ *$//')"
    [ -z "$sp" ] && { IFS=';'; continue; }
    tag="${sp%%=*}"; gates="${sp#*=}"
    chk=0; case "$tag" in *!) chk=1; tag="${tag%!}";; esac
    ck=off; [ "$chk" = 1 ] && ck=armed
    echo "## --- spec '$tag'  gates='$gates'  comb block $(g "$CB")  check $ck ---"
    # shellcheck disable=SC2059
    printf "$ROW" 'tokens/rank' 'dispatch' 'combine PUSH ZC=0' 'combine PULL ZC=1'
    printf "$ROW" ''            'GB/s (lat us)' 'GB/s (lat us)' 'GB/s (lat us)'
    for T in $TOKS; do
      dcell='-'; pcell='-'; lcell='-'
      for z in $ZCS; do
        run "${tag}_t${T}_zc${z}_b$(g "$CB")" "$gates" "$chk" "$T" "$z"
        if [ "$z" = 0 ]; then pcell="$(cell "$R_CBW" "$R_CLAT" "$R_RC")"
                         else lcell="$(cell "$R_CBW" "$R_CLAT" "$R_RC")"; fi
        # ZC picks the combine transport and does not touch dispatch, so the dispatch column comes
        # from the PUSH run; the fallback only matters when PUSH was not asked for at all.
        { [ "$z" = 0 ] || [ "$dcell" = '-' ]; } && dcell="$(cell "$R_DBW" "$R_DLAT" "$R_RC")"
      done
      printf "$ROW" "$T" "$dcell" "$pcell" "$lcell"
    done
    IFS=';'
  done
  IFS=$OLDIFS
done
crumb "sweep end"
echo EP_TEST_DONE
