#!/usr/bin/env bash
# Generic 2-node RDMA cross-rail test worker. Runs on BOTH nodes (srun -N2 -n2).
# Self-organizes by hostname: A = tester (lowest name), B = target/server.
#
# Rail discovery is delegated to rail_detect.sh, shared with probe_topology.sh
# and xrail_matrix.sh: each device's global GID index is auto-detected (IPv4-mapped
# OR IPv6 ULA/GUA, skipping fe80:: link-local) along with its address, so ping vs
# ping6 is chosen automatically and no device or netdev name is hardcoded. Works
# on ionic/bnxt/mlx5 alike.
#
# This is the lighter of the two cross-rail tools: one same-rail and one
# cross-rail RDMA probe plus a full IP ping matrix. Use xrail_matrix.sh when you
# want the NxN RDMA matrix and the SL/MTU/TC sweeps.
#
# Args:  $1 = JOB label   $2 = RUNDIR (per-run output folder, on a shared filesystem)
#
# Optional env overrides:
#   RAIL_LIB           path to rail_detect.sh                          (default: alongside this script)
#   RAIL_DEV_REGEX     only consider RDMA devices matching this ERE     (default: .*)
#   EXCLUDE_DEV_REGEX  drop RDMA devices matching this ERE              (default: none)
#   INCLUDE_MGMT=1     keep the device on the default-route netdev      (default: drop it)
#   GID_INDEX          force this GID index for every device            (default: auto)
#   SRC_RAILS          space-separated source rail indices to ping      (default: all)
#   PORT_BASE          base TCP port for ibv_rc_pingpong                (default: 18500)
set -uo pipefail
JOB=${1:?job label}; RUNDIR=${2:?run dir}
me=$(hostname)
PORT_BASE=${PORT_BASE:-18500}
log() { echo "[$me] $*"; }

# ---------- detect ACTIVE rail devices (rail index = position in DEVS) ----------
RAIL_LIB="${RAIL_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rail_detect.sh}"
[ -r "$RAIL_LIB" ] || { log "cannot read $RAIL_LIB (set RAIL_LIB)"; exit 1; }
# shellcheck source=rail_detect.sh
. "$RAIL_LIB"
rail_detect
DEVS=(${RAIL_ACTIVE[@]+"${RAIL_ACTIVE[@]}"})
NR=${#DEVS[@]}
log "detected $NR rail device(s): ${DEVS[*]:-none}"

# ---------- publish my rails: "idx dev ndev fam gididx addr" ----------
: > "$RUNDIR/addrs.$me"
for i in $(seq 0 $((NR - 1))); do
  d=${DEVS[$i]}
  echo "$i $d ${RAIL_NDEV[$d]} ${RAIL_FAM[$d]} ${RAIL_GIDIDX[$d]} ${RAIL_ADDR[$d]}" >> "$RUNDIR/addrs.$me"
done
touch "$RUNDIR/host.$me"

# ---------- discover the two participating hostnames ----------
# Schedulers that dispatch the batch body per-node (Spur >= 0.10) can start the peer
# tens of seconds after us, so wait generously.
PEER_WAIT=${PEER_WAIT:-180}
for _ in $(seq 1 "$PEER_WAIT"); do
  [ "$(ls "$RUNDIR"/host.* 2>/dev/null | wc -l)" -ge 2 ] && break; sleep 1
done
HOSTS=()
for f in "$RUNDIR"/host.*; do b=$(basename "$f"); HOSTS+=("${b#host.}"); done
IFS=$'\n' HOSTS=($(printf '%s\n' "${HOSTS[@]}" | sort -u)); unset IFS
# Defaulting B to ourselves here would produce a plausible-looking loopback "result"
# that silently answers a different question. Fail loudly instead.
[ "${#HOSTS[@]}" -ge 2 ] || {
  log "FATAL: only ${#HOSTS[@]} host(s) registered in $RUNDIR after ${PEER_WAIT}s: ${HOSTS[*]:-none}"
  log "       the peer node never started — check the allocation, do not trust a 1-node run"
  exit 1
}
A=${HOSTS[0]}; B=${HOSTS[1]}
log "A(tester)=$A B(target)=$B"

have_rdma=0; command -v ibv_rc_pingpong >/dev/null 2>&1 && have_rdma=1
# iputils older than ~2020 (and busybox) have no `ping -6`; the IPv6 rails would then
# all read FAIL and the fabric would be misreported as rail-only. Probe once and fall
# back to the separate ping6 binary.
PING6=(ping -6)
ping -6 -c1 -W1 ::1 >/dev/null 2>&1 || { command -v ping6 >/dev/null 2>&1 && PING6=(ping6); }
# Publish the local tool inventory so the tester can tell "peer cannot run the test"
# apart from "fabric refused the connection" — they look identical from A's side.
echo "ibv_rc_pingpong=$have_rdma ping6=${PING6[*]}" > "$RUNDIR/tools.$me"

# ---------- Node B: re-listening RDMA servers for same-rail & cross-rail ----------
# The server must outlive A's worst case, not its best case: A may retry the same-rail
# probe 5x(8s+2s) before it even starts the cross-rail one. A server sized for the happy
# path exits first, and A then reports UNREACHABLE for a fabric that was fine — a false
# rail-only verdict, which is the most expensive way for this tool to be wrong. Serve
# until the tester says it is finished, bounded so a dead tester cannot hang the job.
if [ "$me" = "$B" ] && [ "$have_rdma" = 1 ] && [ "$NR" -ge 1 ]; then
  serve() { local dev=$1 gi=$2 port=$3 r
    for r in $(seq 1 60); do
      [ -e "$RUNDIR/rdma_done" ] && break
      timeout 12 ibv_rc_pingpong -d "$dev" -g "$gi" -p "$port" >/dev/null 2>&1
      sleep 1
    done; }
  d0=${DEVS[0]}; g0=${RAIL_GIDIDX[$d0]}
  serve "$d0" "$g0" "$PORT_BASE" &
  if [ "$NR" -ge 2 ]; then
    d1=${DEVS[1]}; g1=${RAIL_GIDIDX[$d1]}
    serve "$d1" "$g1" "$((PORT_BASE+1))" &
  fi
fi

# ---------- Node B: outlive the tester ----------
# B has nothing left to do once the RDMA phase ends, but it must not EXIT: on per-node
# dispatch the allocation is torn down when the bodies finish, and an early B kills A
# in the middle of the IP matrix — which truncates result.txt silently, leaving a file
# that still parses and still looks like a verdict. Wait for A's end-of-run sentinel.
if [ "$me" = "$B" ]; then
  END_WAIT=${END_WAIT:-600}
  for _ in $(seq 1 "$END_WAIT"); do
    [ -e "$RUNDIR/all_done" ] && break
    sleep 1
  done
  [ -e "$RUNDIR/all_done" ] || log "WARNING: tester never signalled completion after ${END_WAIT}s"
fi

# ---------- wait for both addr files ----------
for _ in $(seq 1 30); do
  [ -s "$RUNDIR/addrs.$A" ] && [ -s "$RUNDIR/addrs.$B" ] && break; sleep 1
done

# ---------- Node A only, once (atomic lock) ----------
# The lock guards against a scheduler spawning this wrapper twice on one node. It is
# released on exit: sbatch retries the step on failure, and a lock left behind by a
# crashed attempt would make every retry a silent no-op that still reports success.
if [ "$me" = "$A" ] && mkdir "$RUNDIR/lock" 2>/dev/null; then
  # Also signal the servers on the way out, so a tester that dies mid-probe releases B
  # instead of leaving it to burn its bounded loop.
  # Trap INT/TERM too: the scheduler kills the step rather than letting it exit, and a
  # peer left waiting on a sentinel that never arrives burns the rest of the time limit.
  trap 'rmdir "$RUNDIR/lock" 2>/dev/null; touch "$RUNDIR/rdma_done" "$RUNDIR/all_done" 2>/dev/null' EXIT INT TERM
  R="$RUNDIR/result.txt"; : > "$R"
  # B's inventory, if it published one (older peers may not have)
  b_rdma=$(sed -n 's/.*ibv_rc_pingpong=\([01]\).*/\1/p' "$RUNDIR/tools.$B" 2>/dev/null)
  b_rdma=${b_rdma:-1}
  # peer (B) rails
  declare -A BADDR BFAM
  while read -r idx dev ndev fam gi addr; do BADDR[$idx]=$addr; BFAM[$idx]=$fam; done < "$RUNDIR/addrs.$B"
  BNR=$(wc -l < "$RUNDIR/addrs.$B")

  {
    echo "cross-rail fabric test  tester=$A  target=$B  (job $JOB)"
    echo "rails on A=$NR  rails on B=$BNR"
    echo "rail map (A): $(for i in $(seq 0 $((NR - 1))); do d=${DEVS[$i]}; printf 'r%s=%s/%s/gid%s ' "$i" "$d" "${RAIL_NDEV[$d]}" "${RAIL_GIDIDX[$d]}"; done)"
    echo ""
  } >> "$R"

  # ---- RDMA layer (authoritative) ----
  # A peer with no ibv_rc_pingpong never binds, so every probe would time out and read
  # exactly like a dead fabric. Report the missing tool instead of inventing a verdict.
  if [ "$b_rdma" != 1 ]; then
    { echo "### RDMA: SKIPPED (peer $B has no ibv_rc_pingpong — cannot serve)"; echo ""; } >> "$R"
  elif [ "$have_rdma" = 1 ] && [ "$NR" -ge 1 ]; then
    d0=${DEVS[0]}; g0=${RAIL_GIDIDX[$d0]}
    try_rdma() { local dev=$1 gi=$2 port=$3 r; for r in 1 2 3 4 5; do
        timeout 8 ibv_rc_pingpong -d "$dev" -g "$gi" -p "$port" "$B" >/dev/null 2>&1 && { echo REACHABLE; return; }; sleep 2
      done; echo UNREACHABLE; }
    sleep 3
    s1=$(try_rdma "$d0" "$g0" "$PORT_BASE")
    { echo "### RDMA (ibv_rc_pingpong)"; echo "same-rail  A.$d0 -> B.rail0 : $s1"; } >> "$R"
    if [ "$NR" -ge 2 ] && [ "$BNR" -ge 2 ]; then
      s2=$(try_rdma "$d0" "$g0" "$((PORT_BASE+1))")
      echo "cross-rail A.$d0 -> B.rail1 : $s2" >> "$R"
    else
      echo "cross-rail : SKIPPED (need >=2 rails on both nodes)" >> "$R"
    fi
    echo "" >> "$R"
  else
    echo "### RDMA: SKIPPED (ibv_rc_pingpong not found or no rails)" >> "$R"; echo "" >> "$R"
  fi
  # Release B's servers now that the RDMA phase is over; the IP matrix does not need them.
  touch "$RUNDIR/rdma_done"

  # ---- IP layer matrix (ping/ping6, bound to source rail netdev) ----
  echo "### IP reachability (ping bound to source rail netdev), A -> B" >> "$R"
  printf "%-22s %-8s %-6s %s\n" src_rail dst_rail stat kind >> "$R"
  SRCS="${SRC_RAILS:-$(seq 0 $((NR-1)))}"
  for s in $SRCS; do
    case $s in *[!0-9]*) log "ignoring non-numeric SRC_RAILS entry '$s'"; continue ;; esac
    [ "$s" -lt "$NR" ] || continue
    sdev=${DEVS[$s]}; sndev=${RAIL_NDEV[$sdev]}
    for t in $(seq 0 $((BNR-1))); do
      tip=${BADDR[$t]:-}; tfam=${BFAM[$t]:-}
      if [ -z "$tip" ]; then st=NO_DST_IP
      else
        if [ "$tfam" = 6 ]; then PCMD=("${PING6[@]}"); else PCMD=(ping -4); fi
        if "${PCMD[@]}" -c 2 -W 2 -I "$sndev" "$tip" >/dev/null 2>&1; then st=OK; else st=FAIL; fi
      fi
      kind=$([ "$s" = "$t" ] && echo same-rail || echo cross-rail)
      printf "%-22s %-8s %-6s %s\n" "rail$s($sndev)" "rail$t" "$st" "$kind" >> "$R"
    done
  done
  echo "DONE" >> "$R"
  touch "$RUNDIR/all_done"
fi

wait 2>/dev/null || true
