#!/bin/bash
# Full cross-rail RDMA verification. Runs on BOTH nodes (srun -N2 -n2).
# A = tester (lowest hostname), B = target/server.
#
# Answers one question the earlier probe could not: is cross-rail RoCE dead on
# every rail pair, or only under the default SL/TC/MTU? The earlier run tested
# exactly one pair (r0->r1) at pingpong defaults and generalized from it.
#
#   Phase 1  environment: per-rail MTU/rate/GID, QoS/PFC/DSCP/ECN state
#   Phase 2  full 8x8 RDMA reachability matrix
#   Phase 3  SL sweep 0-7 on a cross-rail pair
#   Phase 4  path-MTU sweep on a cross-rail pair
#   Phase 5  traffic-class sweep via ib_write_bw (if perftest present)
#
# Args: $1 = JOB label   $2 = RUNDIR (shared filesystem)
set -uo pipefail
JOB=${1:?job label}; RUNDIR=${2:?run dir}
me=$(hostname)
PORT_BASE=${PORT_BASE:-18600}
BW_PORT_BASE=${BW_PORT_BASE:-19600}
log() { echo "[$me] $*"; }

# ---------- detect ACTIVE rail devices, excluding the mgmt NIC ----------
# Shared with probe_topology.sh and xrail_worker.sh: each device's GID index is
# auto-detected, and rails addressed as IPv4-mapped are found as well as IPv6.
RAIL_LIB="${RAIL_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rail_detect.sh}"
[ -r "$RAIL_LIB" ] || { log "cannot read $RAIL_LIB (set RAIL_LIB)"; exit 1; }
# shellcheck source=rail_detect.sh
. "$RAIL_LIB"
rail_detect
IB_ROOT="$RAIL_IB_ROOT"
DEVS=(${RAIL_ACTIVE[@]+"${RAIL_ACTIVE[@]}"})
NR=${#DEVS[@]}
log "rails: ${DEVS[*]:-none}"

# Every sweep below is cross-rail, so it needs a second rail to aim at. Say why
# and write it down: a pre-flight tool that meets a machine it does not
# understand should produce a diagnosis, not an unbound-variable abort.
if [ "$NR" -lt 2 ]; then
  log "need >=2 active rails, found $NR — nothing to sweep"
  {
    echo "cross-rail RDMA verification   node=$me  job=$JOB"
    echo "date=$(date -u +%FT%TZ)"
    echo
    echo "ABORTED: need >=2 active RoCE rails, found $NR."
    echo "all RDMA devices seen: ${RAIL_ALL[*]:-none}"
    echo "rails (addressable):   ${RAIL_DEVS[*]:-none}"
    echo "rails (ACTIVE):        ${RAIL_ACTIVE[*]:-none}"
    echo "mgmt netdev excluded:  ${RAIL_MGMT_NDEV:-none}"
    echo
    echo "Run 'rail_detect.sh --dump' on this node to see why devices were dropped."
  } > "$RUNDIR/matrix.txt"
  exit 0
fi
d0=${DEVS[0]}; d1=${DEVS[1]}

: > "$RUNDIR/addrs.$me"
for i in "${!DEVS[@]}"; do d=${DEVS[$i]}; echo "$i $d ${RAIL_NDEV[$d]} ${RAIL_ADDR[$d]}" >> "$RUNDIR/addrs.$me"; done
touch "$RUNDIR/host.$me"

# Schedulers that dispatch the batch body per-node (Spur >= 0.10) can start the peer
# tens of seconds after us, so wait generously.
PEER_WAIT=${PEER_WAIT:-180}
for _ in $(seq 1 "$PEER_WAIT"); do [ "$(ls "$RUNDIR"/host.* 2>/dev/null | wc -l)" -ge 2 ] && break; sleep 1; done
HOSTS=(); for f in "$RUNDIR"/host.*; do b=$(basename "$f"); HOSTS+=("${b#host.}"); done
IFS=$'\n' HOSTS=($(printf '%s\n' "${HOSTS[@]}" | sort -u)); unset IFS
# Defaulting B to ourselves here would produce a plausible-looking loopback matrix
# that silently answers a different question. Fail loudly instead.
[ "${#HOSTS[@]}" -ge 2 ] || {
  log "FATAL: only ${#HOSTS[@]} host(s) registered in $RUNDIR after ${PEER_WAIT}s: ${HOSTS[*]:-none}"
  log "       the peer node never started — check the allocation, do not trust a 1-node run"
  exit 1
}
A=${HOSTS[0]}; B=${HOSTS[1]}
log "A(tester)=$A  B(target)=$B"

# ---------- Phase 1: environment dump (both nodes) ----------
{
  echo "===== NODE $me ====="
  echo "--- kernel/driver ---"
  uname -r
  for d in "${DEVS[@]}"; do
    P="$IB_ROOT/$d/ports/1"
    printf "%-9s fw=%-16s mtu_active=%-8s rate=%-14s ndev=%-11s gid%s=%s\n" \
      "$d" "$(cat "$IB_ROOT/$d/fw_ver" 2>/dev/null)" \
      "$(awk '{print $2}' "$P/rate" 2>/dev/null; cat "$P/active_mtu" 2>/dev/null)" \
      "$(cat "$P/rate" 2>/dev/null)" "${RAIL_NDEV[$d]}" "${RAIL_GIDIDX[$d]}" "${RAIL_ADDR[$d]}"
  done
  echo "--- netdev MTU ---"
  for d in "${DEVS[@]}"; do
    printf "%-9s %-11s mtu=%s\n" "$d" "${RAIL_NDEV[$d]}" "$(cat /sys/class/net/${RAIL_NDEV[$d]}/mtu 2>/dev/null)"
  done
  echo "--- QoS / PFC / DSCP / ECN ---"
  echo "nicctl: $(command -v nicctl || echo ABSENT)"
  echo "mlnx_qos: $(command -v mlnx_qos || echo ABSENT)"
  echo "dcb: $(command -v dcb || echo ABSENT)"
  n0=${RAIL_NDEV[$d0]:-}
  if [ -n "$n0" ]; then
    echo "[dcb pfc show dev $n0]"; dcb pfc show dev "$n0" 2>&1 | head -12
    echo "[dcb app show dev $n0]"; dcb app show dev "$n0" 2>&1 | head -12
    echo "[tc qdisc show dev $n0]"; tc qdisc show dev "$n0" 2>&1 | head -6
  fi
  if command -v nicctl >/dev/null 2>&1; then echo "[nicctl]"; nicctl show qos 2>&1 | head -30; fi
  echo "--- tools ---"
  for t in ibv_rc_pingpong ib_write_bw ib_send_bw ibv_devinfo; do
    printf "%-18s %s\n" "$t" "$(command -v $t || echo ABSENT)"
  done
  echo
} > "$RUNDIR/env.$me" 2>&1

have_pp=0; command -v ibv_rc_pingpong >/dev/null 2>&1 && have_pp=1
have_bw=0; command -v ib_write_bw     >/dev/null 2>&1 && have_bw=1

# ---------- Node B: one persistent re-listening server per rail ----------
# Each ibv_rc_pingpong server exits after a single connection, so loop it.
# Port PORT_BASE+t is served by B's rail t -> the client's choice of port
# selects the DESTINATION rail; its -d flag selects the SOURCE rail.
if [ "$me" = "$B" ] && [ "$have_pp" = 1 ]; then
  for t in "${!DEVS[@]}"; do
    d=${DEVS[$t]}
    ( for _ in $(seq 1 200); do
        timeout 10 ibv_rc_pingpong -d "$d" -g "${RAIL_GIDIDX[$d]}" -p "$((PORT_BASE+t))" -n 20 >/dev/null 2>&1
      done ) &
  done
  # servers for the SL sweep on rail 1 (dest), one per SL
  for sl in 0 1 2 3 4 5 6 7; do
    ( for _ in $(seq 1 12); do
        timeout 10 ibv_rc_pingpong -d "$d1" -g "${RAIL_GIDIDX[$d1]}" -p "$((PORT_BASE+40+sl))" -n 20 -l "$sl" >/dev/null 2>&1
      done ) &
  done
  # servers for the MTU sweep on rail 1 (dest)
  mi=0
  for m in 256 512 1024 2048 4096; do
    ( for _ in $(seq 1 12); do
        timeout 10 ibv_rc_pingpong -d "$d1" -g "${RAIL_GIDIDX[$d1]}" -p "$((PORT_BASE+60+mi))" -n 20 -m "$m" >/dev/null 2>&1
      done ) &
    mi=$((mi+1))
  done
  if [ "$have_bw" = 1 ]; then
    ti=0
    for tc in 0 8 16 26 46 96 106 136; do
      ( for _ in $(seq 1 12); do
          timeout 15 ib_write_bw -d "$d1" -x "${RAIL_GIDIDX[$d1]}" -p "$((BW_PORT_BASE+ti))" \
            -n 200 -s 4096 --tclass="$tc" >/dev/null 2>&1
        done ) &
      ti=$((ti+1))
    done
  fi
  touch "$RUNDIR/servers_up.$me"
fi

for _ in $(seq 1 40); do [ -s "$RUNDIR/addrs.$A" ] && [ -s "$RUNDIR/addrs.$B" ] && break; sleep 1; done

# ---------- Node A: run the sweeps ----------
if [ "$me" = "$A" ] && mkdir "$RUNDIR/lock" 2>/dev/null; then
  for _ in $(seq 1 30); do [ -f "$RUNDIR/servers_up.$B" ] && break; sleep 1; done
  sleep 5
  R="$RUNDIR/matrix.txt"; : > "$R"
  BNR=$(wc -l < "$RUNDIR/addrs.$B")

  {
    echo "cross-rail RDMA verification   tester=$A  target=$B  job=$JOB"
    echo "rails A=$NR  rails B=$BNR   gid_index=${GID_INDEX:-auto}"
    for d in ${DEVS[@]+"${DEVS[@]}"}; do
      echo "  A.$d  gid=${RAIL_GIDIDX[$d]}  ndev=${RAIL_NDEV[$d]}  addr=${RAIL_ADDR[$d]}"
    done
    echo "date=$(date -u +%FT%TZ)"
    echo
  } >> "$R"

  # try: $1=src dev  $2=port  -> REACHABLE / unreachable
  try_pp() {
    local dev=$1 port=$2 extra=${3:-} r
    for r in 1 2; do
      # shellcheck disable=SC2086
      timeout 8 ibv_rc_pingpong -d "$dev" -g "${RAIL_GIDIDX[$dev]}" -p "$port" -n 20 $extra "$B" >/dev/null 2>&1 \
        && { echo OK; return; }
      sleep 1
    done
    echo FAIL
  }

  # ---- Phase 2: full src-rail x dst-rail matrix ----
  {
    echo "### Phase 2 — RDMA reachability matrix (ibv_rc_pingpong, default SL/MTU)"
    echo "rows = A source rail (-d), cols = B destination rail (port)"
    printf "%-10s" "src\\dst"; for t in $(seq 0 $((BNR-1))); do printf "%6s" "r$t"; done; echo
  } >> "$R"
  okc=0; failc=0; xokc=0; xfailc=0
  for s in "${!DEVS[@]}"; do
    sdev=${DEVS[$s]}
    printf "%-10s" "r$s" >> "$R"
    for t in $(seq 0 $((BNR-1))); do
      st=$(try_pp "$sdev" "$((PORT_BASE+t))")
      if [ "$st" = OK ]; then printf "%6s" "OK" >> "$R"; okc=$((okc+1)); [ "$s" != "$t" ] && xokc=$((xokc+1))
      else printf "%6s" "--" >> "$R"; failc=$((failc+1)); [ "$s" != "$t" ] && xfailc=$((xfailc+1)); fi
    done
    echo >> "$R"
  done
  {
    echo "totals: OK=$okc FAIL=$failc   cross-rail OK=$xokc cross-rail FAIL=$xfailc"
    echo
  } >> "$R"

  # ---- Phase 3: SL sweep on cross-rail r0 -> r1 ----
  {
    echo "### Phase 3 — service-level sweep, A.$d0 -> B.rail1 (cross-rail)"
    for sl in 0 1 2 3 4 5 6 7; do
      printf "sl=%-3s %s\n" "$sl" "$(try_pp "$d0" "$((PORT_BASE+40+sl))" "-l $sl")"
    done
    echo
  } >> "$R"

  # ---- Phase 4: path-MTU sweep on cross-rail r0 -> r1 ----
  {
    echo "### Phase 4 — path-MTU sweep, A.$d0 -> B.rail1 (cross-rail)"
    mi=0
    for m in 256 512 1024 2048 4096; do
      printf "mtu=%-6s %s\n" "$m" "$(try_pp "$d0" "$((PORT_BASE+60+mi))" "-m $m")"
      mi=$((mi+1))
    done
    echo
  } >> "$R"

  # ---- Phase 5: traffic-class sweep via ib_write_bw ----
  {
    echo "### Phase 5 — traffic-class (DSCP) sweep, A.$d0 -> B.rail1 (cross-rail)"
    if [ "$have_bw" = 1 ]; then
      ti=0
      for tc in 0 8 16 26 46 96 106 136; do
        out=$(timeout 20 ib_write_bw -d "$d0" -x "${RAIL_GIDIDX[$d0]}" -p "$((BW_PORT_BASE+ti))" \
                -n 200 -s 4096 --tclass="$tc" "$B" 2>&1)
        bw=$(echo "$out" | awk '/^ *4096/{print $4" MB/s"; found=1} END{if(!found) print "no-data"}')
        printf "tclass=%-5s %s\n" "$tc" "$bw"
        ti=$((ti+1)); sleep 1
      done
    else
      echo "SKIPPED — ib_write_bw not installed"
    fi
    echo
  } >> "$R"

  echo "DONE" >> "$R"
fi

wait 2>/dev/null || true
