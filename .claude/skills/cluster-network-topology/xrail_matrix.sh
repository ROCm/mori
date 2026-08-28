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
IB_ROOT=/sys/class/infiniband
GID_INDEX=${GID_INDEX:-1}          # RoCEv2 IPv6-ULA rails are commonly at index 1
log() { echo "[$me] $*"; }

# ---------- detect ACTIVE rail devices, excluding the mgmt NIC ----------
MGMT_NDEV=$(ip route show default 2>/dev/null | awk '/default/{print $5; exit}')
DEVS=(); declare -A NDEV ADDR
for d in $(ls "$IB_ROOT" 2>/dev/null | sort -V); do
  P="$IB_ROOT/$d/ports/1"
  [ "$(awk '{print $2}' "$P/state" 2>/dev/null)" = ACTIVE ] || continue
  [ "$(cat "$P/gid_attrs/types/$GID_INDEX" 2>/dev/null)" = "RoCE v2" ] || continue
  nd=$(cat "$P/gid_attrs/ndevs/$GID_INDEX" 2>/dev/null)
  [ -n "$nd" ] || continue
  [ -n "$MGMT_NDEV" ] && [ "$nd" = "$MGMT_NDEV" ] && continue
  a=$(ip -o -6 addr show "$nd" scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
  [ -n "$a" ] || continue
  DEVS+=("$d"); NDEV[$d]=$nd; ADDR[$d]=$a
done
NR=${#DEVS[@]}
log "rails: ${DEVS[*]:-none}"

: > "$RUNDIR/addrs.$me"
for i in "${!DEVS[@]}"; do d=${DEVS[$i]}; echo "$i $d ${NDEV[$d]} ${ADDR[$d]}" >> "$RUNDIR/addrs.$me"; done
touch "$RUNDIR/host.$me"

for _ in $(seq 1 30); do [ "$(ls "$RUNDIR"/host.* 2>/dev/null | wc -l)" -ge 2 ] && break; sleep 1; done
HOSTS=(); for f in "$RUNDIR"/host.*; do b=$(basename "$f"); HOSTS+=("${b#host.}"); done
IFS=$'\n' HOSTS=($(printf '%s\n' "${HOSTS[@]}" | sort -u)); unset IFS
A=${HOSTS[0]:-$me}; B=${HOSTS[1]:-$me}
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
      "$(cat "$P/rate" 2>/dev/null)" "${NDEV[$d]}" "$GID_INDEX" "${ADDR[$d]}"
  done
  echo "--- netdev MTU ---"
  for d in "${DEVS[@]}"; do
    printf "%-9s %-11s mtu=%s\n" "$d" "${NDEV[$d]}" "$(cat /sys/class/net/${NDEV[$d]}/mtu 2>/dev/null)"
  done
  echo "--- QoS / PFC / DSCP / ECN ---"
  echo "nicctl: $(command -v nicctl || echo ABSENT)"
  echo "mlnx_qos: $(command -v mlnx_qos || echo ABSENT)"
  echo "dcb: $(command -v dcb || echo ABSENT)"
  n0=${NDEV[${DEVS[0]}]:-}
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
        timeout 10 ibv_rc_pingpong -d "$d" -g "$GID_INDEX" -p "$((PORT_BASE+t))" -n 20 >/dev/null 2>&1
      done ) &
  done
  # servers for the SL sweep on rail 1 (dest), one per SL
  for sl in 0 1 2 3 4 5 6 7; do
    ( for _ in $(seq 1 12); do
        timeout 10 ibv_rc_pingpong -d "${DEVS[1]}" -g "$GID_INDEX" -p "$((PORT_BASE+40+sl))" -n 20 -l "$sl" >/dev/null 2>&1
      done ) &
  done
  # servers for the MTU sweep on rail 1 (dest)
  mi=0
  for m in 256 512 1024 2048 4096; do
    ( for _ in $(seq 1 12); do
        timeout 10 ibv_rc_pingpong -d "${DEVS[1]}" -g "$GID_INDEX" -p "$((PORT_BASE+60+mi))" -n 20 -m "$m" >/dev/null 2>&1
      done ) &
    mi=$((mi+1))
  done
  if [ "$have_bw" = 1 ]; then
    ti=0
    for tc in 0 8 16 26 46 96 106 136; do
      ( for _ in $(seq 1 12); do
          timeout 15 ib_write_bw -d "${DEVS[1]}" -x "$GID_INDEX" -p "$((BW_PORT_BASE+ti))" \
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
    echo "rails A=$NR  rails B=$BNR   gid_index=$GID_INDEX"
    echo "date=$(date -u +%FT%TZ)"
    echo
  } >> "$R"

  # try: $1=src dev  $2=port  -> REACHABLE / unreachable
  try_pp() {
    local dev=$1 port=$2 extra=${3:-} r
    for r in 1 2; do
      # shellcheck disable=SC2086
      timeout 8 ibv_rc_pingpong -d "$dev" -g "$GID_INDEX" -p "$port" -n 20 $extra "$B" >/dev/null 2>&1 \
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
    echo "### Phase 3 — service-level sweep, A.${DEVS[0]} -> B.rail1 (cross-rail)"
    for sl in 0 1 2 3 4 5 6 7; do
      printf "sl=%-3s %s\n" "$sl" "$(try_pp "${DEVS[0]}" "$((PORT_BASE+40+sl))" "-l $sl")"
    done
    echo
  } >> "$R"

  # ---- Phase 4: path-MTU sweep on cross-rail r0 -> r1 ----
  {
    echo "### Phase 4 — path-MTU sweep, A.${DEVS[0]} -> B.rail1 (cross-rail)"
    mi=0
    for m in 256 512 1024 2048 4096; do
      printf "mtu=%-6s %s\n" "$m" "$(try_pp "${DEVS[0]}" "$((PORT_BASE+60+mi))" "-m $m")"
      mi=$((mi+1))
    done
    echo
  } >> "$R"

  # ---- Phase 5: traffic-class sweep via ib_write_bw ----
  {
    echo "### Phase 5 — traffic-class (DSCP) sweep, A.${DEVS[0]} -> B.rail1 (cross-rail)"
    if [ "$have_bw" = 1 ]; then
      ti=0
      for tc in 0 8 16 26 46 96 106 136; do
        out=$(timeout 20 ib_write_bw -d "${DEVS[0]}" -x "$GID_INDEX" -p "$((BW_PORT_BASE+ti))" \
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
