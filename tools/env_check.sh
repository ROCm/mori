#!/bin/bash

# NIC version recommendations for cross-node MORI (EP over RDMA / IBGDA).
# The wrong NIC firmware/driver version is a common blocker, so `mori check`
# validates the detected version against these known-good/known-bad ranges:
#   - AINIC     : >= 1.117.5-a-45 is solid. The 1.117.1 major does NOT support IBGDA.
#   - Broadcom  : solid on 237.1.137.x (official release) and 235.2.86.x
#                 (customer-specific build); 231.x is too old for IBGDA.
#   - Mellanox  : no real version requirement — mlx5 just works out of the box.
# For all NICs, the userspace library must match the corresponding kernel driver.

set -uo pipefail

# ============================ config ============================

PEER_IP="${1:-}"
BW_THRESHOLD=300    # Gbps
LAT_THRESHOLD=10    # microseconds
MSG_SIZE=65536      # 64K
LAT_MSG_SIZE=2      # bytes (small message for latency)
IB_PORT=18515       # base port for ib_write_bw / ib_write_lat
AINIC_MIN_VER="1.117.5-a-45"       # minimum recommended AINIC firmware for IBGDA
BNXT_MIN_VER_235="235.2.86.0"      # minimum solid version on the 235.x branch
BNXT_MIN_VER_237="237.1.137.0"     # minimum solid version on the 237.x branch
# Kept <= sshd MaxSessions (default 10): all remote servers multiplex over one
# ssh master connection, so too many concurrent exec sessions would be refused.
MESH_PARALLEL=8     # max concurrent pair probes for mesh tests
MESH_CLI_TIMEOUT=3  # per-pair client timeout (s); unreachable pair -> fail
MESH_SRV_TIMEOUT=6  # per-pair server timeout (s); must exceed client timeout
MESH_SRV_WAIT_REMOTE=0.5  # wait (s) for ssh-launched remote server to bind (master is warm)
MESH_RETRIES=2      # extra attempts for a pair when the server wasn't ready (anti false-negative)
MORI_RDMA_SL=0      # overwritten by check_qos()
MORI_RDMA_TC=0      # overwritten by check_qos()

# ========================== logging / ui ========================

STEP=0
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step()      { STEP=$((STEP + 1)); echo ""; echo -e "${CYAN}=== Step $STEP: $* ===${NC}"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_skip()  { echo -e "${YELLOW}[SKIP]${NC} $*"; }

die() { log_fail "$@"; exit 1; }

require_cmd() {
    command -v "$1" > /dev/null 2>&1 || die "$1 not found. Please install it first."
}

# =============== ssh: identity + connection multiplexing ========

# SSH identity for peer access. Under sudo, $(whoami) is root (no authorized key),
# so prefer the invoking user ($SUDO_USER). Override with SSH_USER=... if needed.
SSH_USER="${SSH_USER:-${SUDO_USER:-$(whoami)}}"
# Connection multiplexing: a single persistent master carries every ssh in the mesh,
# so per-pair runs skip the TCP+auth handshake (~0.5-1s each) and don't hit sshd
# MaxStartups. The socket must be writable by the user that actually runs ssh.
SSH_MUX_DIR="${TMPDIR:-/tmp}/.envcheck_sshmux.${SUDO_USER:-${USER:-$(id -un)}}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR \
-o ControlMaster=auto -o ControlPath=$SSH_MUX_DIR/%r@%h:%p -o ControlPersist=60"
# When running under sudo (HOME=/root), ssh would use root's keys. Drop back to the
# invoking user so their ~/.ssh key is used. SSH_PREFIX is empty otherwise.
if [[ "$EUID" -eq 0 && -n "${SUDO_USER:-}" ]]; then
    SSH_PREFIX=(sudo -u "$SUDO_USER")
    install -d -o "$SUDO_USER" -m 700 "$SSH_MUX_DIR" 2>/dev/null
else
    SSH_PREFIX=()
    mkdir -p -m 700 "$SSH_MUX_DIR" 2>/dev/null
fi

# Open one master connection up front so the first mesh pairs reuse it immediately.
ssh_warm() {
    local host="$1"
    "${SSH_PREFIX[@]}" ssh $SSH_OPTS "$SSH_USER@$host" true 2>/dev/null
}
ssh_cleanup() {
    [[ -d "$SSH_MUX_DIR" ]] || return 0
    local s
    for s in "$SSH_MUX_DIR"/*; do
        [[ -S "$s" ]] || continue
        "${SSH_PREFIX[@]}" ssh $SSH_OPTS -O exit -o ControlPath="$s" x 2>/dev/null
    done
    rm -rf "$SSH_MUX_DIR" 2>/dev/null
}
trap ssh_cleanup EXIT

# ================== rdma device & gid discovery =================

# query_rdma_devices [host]
#   prints one RDMA device name per line; empty host = local
query_rdma_devices() {
    local host="${1:-}"
    if [[ -z "$host" || "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        ibv_devices 2>/dev/null | awk 'NR>2 && NF {print $1}'
    else
        # A login banner (e.g. Conductor MOTD) can precede the command output, so
        # emit a sentinel and parse only the ibv_devices table after it.
        "${SSH_PREFIX[@]}" ssh $SSH_OPTS "$SSH_USER@$host" 'echo __IBV__; ibv_devices 2>/dev/null' 2>/dev/null \
            | sed -n '/^__IBV__$/,$p' | awk 'NR>3 && NF {print $1}'
    fi
}

# Self-contained snippet ($1=ib device) that prints the best GID index for that
# device, mirroring MORI's ScoreGidCandidate (rdma.cpp): RoCEv2 (+1000) over
# RoCEv1 (+500); IPv4-mapped (+200) over global IPv6 (+100) over link-local (+0);
# smaller index wins ties. Hardcoding "-x 1" picks the RoCEv2 *link-local* GID
# (fe80::), which is NOT routable through the ToR and makes QPs fail at RTR.
_GID_PICK_SNIPPET='dev="$1"; g="/sys/class/infiniband/$dev/ports/1"; best=""; bs=-1;
for tf in "$g/gid_attrs/types/"*; do
  [ -e "$tf" ] || continue; i=$(basename "$tf");
  t=$(cat "$g/gid_attrs/types/$i" 2>/dev/null); gid=$(cat "$g/gids/$i" 2>/dev/null);
  [ -z "$gid" ] && continue;
  [ "$gid" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && continue;
  s=0; case "$t" in *v2*) s=$((s+1000));; *) s=$((s+500));; esac;
  case "$gid" in 0000:0000:0000:0000:0000:ffff:*) s=$((s+200));; fe80:*) : ;; *) s=$((s+100));; esac;
  s=$((s-i)); if [ "$s" -gt "$bs" ]; then bs=$s; best=$i; fi;
done; echo "$best"'

# gid_index <dev> [host]   -> best GID index (default 1 if detection fails)
gid_index() {
    local dev="$1" host="${2:-}" idx
    if [[ -z "$host" || "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        idx=$(bash -c "$_GID_PICK_SNIPPET" _ "$dev" 2>/dev/null)
    else
        idx=$("${SSH_PREFIX[@]}" ssh $SSH_OPTS "$SSH_USER@$host" \
              "echo __GID__; bash -c '$_GID_PICK_SNIPPET' _ $dev" 2>/dev/null \
              | sed -n '/^__GID__$/,$p' | sed -n '2p')
    fi
    echo "${idx:-1}"
}

# build_gid_map <assoc_name> <host> <dev...>
#   fills the named assoc array with each device's best GID index.
build_gid_map() {
    local -n _m="$1"; local host="$2"; shift 2
    # gid_index is expensive (scans hundreds of sysfs GID entries, or an ssh round
    # trip per device). Probe all devices in parallel; throttle to MESH_PARALLEL so
    # remote probes stay under sshd MaxSessions.
    local d tmpd running=0
    tmpd=$(mktemp -d)
    for d in "$@"; do
        ( gid_index "$d" "$host" > "$tmpd/$d" ) &
        running=$(( running + 1 ))
        if (( running >= MESH_PARALLEL )); then wait -n 2>/dev/null; running=$(( running - 1 )); fi
    done
    wait
    for d in "$@"; do
        _m["$d"]=$(cat "$tmpd/$d" 2>/dev/null)
        [[ -n "${_m[$d]}" ]] || _m["$d"]=1
    done
    rm -rf "$tmpd"
}

# nic_ipv4 <dev>   -> first IPv4 of the device's netdev (empty if none)
nic_ipv4() {
    local dev="$1" nd
    nd=$(ls "/sys/class/infiniband/$dev/device/net" 2>/dev/null | head -1)
    [[ -n "$nd" ]] || { echo ""; return; }
    ip -o -4 addr show dev "$nd" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1
}

# intra_reachable <src_dev> <dst_dev>
#   0 if src can route to dst on this host (same backend fabric), 1 otherwise.
#   Uses the per-source policy routing table (the same path RoCEv2 traffic takes
#   when it hairpins through the ToR), so it matches what ib_write_bw will do.
intra_reachable() {
    local sip dip stable
    sip=$(nic_ipv4 "$1"); dip=$(nic_ipv4 "$2")
    [[ -n "$sip" && -n "$dip" ]] || return 0   # can't tell -> don't skip
    stable=$(ip rule show 2>/dev/null | awk -v ip="$sip" \
        '$0 ~ ("from "ip"[ /]") {for(i=1;i<=NF;i++) if($i=="lookup") print $(i+1)}' | head -1)
    [[ -n "$stable" ]] || return 0             # no policy table -> don't skip
    # dst is reachable if it falls inside any prefix present in src's table
    local pfx
    while read -r pfx; do
        [[ -n "$pfx" ]] || continue
        if python3 -c "import ipaddress,sys; sys.exit(0 if ipaddress.ip_address('$dip') in ipaddress.ip_network('$pfx',strict=False) else 1)" 2>/dev/null; then
            return 0
        fi
    done < <(ip route show table "$stable" 2>/dev/null | awk '$1 ~ /\//{print $1}')
    return 1
}

# ==================== parallel full-mesh runner =================

# mesh_execute fills <cell_assoc>[i,j] with a numeric metric, "x" (fail) or "-" (self).
#   mesh_execute <cell_assoc> <tool> <server_host> <self_skip> <iters> \
#                <row_gid_assoc> <col_gid_assoc> <row_arr> <col_arr>
#   tool=ib_write_bw -> BW avg (Gbps);  tool=ib_write_lat -> avg latency (us).
#   iters: empty = ib_* default; otherwise "-n <iters>".
#   Probes run in parallel, throttled to MESH_PARALLEL. Each pair uses a unique
#   port so concurrent runs don't collide.
mesh_execute() {
    local -n _cell="$1" _rgid="$6" _cgid="$7" _row="$8" _col="$9"
    local tool="$2" shost="$3" self_skip="$4" iters="$5"
    local nr=${#_row[@]} nc=${#_col[@]}
    local extra key col
    if [[ "$tool" == "ib_write_bw" ]]; then
        extra="-s $MSG_SIZE --report_gbits"; key="$MSG_SIZE"; col='$(NF-1)'
    else
        extra="-s $LAT_MSG_SIZE"; key="$LAT_MSG_SIZE"; col='$6'
    fi
    [[ -n "$iters" ]] && extra="$extra -n $iters"

    local tmpd; tmpd=$(mktemp -d)
    local i j port running=0
    for (( i=0; i<nr; i++ )); do
        for (( j=0; j<nc; j++ )); do
            if [[ "$self_skip" == "true" && "${_row[$i]}" == "${_col[$j]}" ]]; then
                printf -- '-' > "$tmpd/$i.$j"; continue
            fi
            port=$(( IB_PORT + i * nc + j ))   # unique per pair
            (
                local is_local=false base_wait="$MESH_SRV_WAIT_REMOTE"
                [[ "$shost" == "localhost" || "$shost" == "127.0.0.1" ]] && { is_local=true; base_wait=0.5; }
                local result="x" try sp out rc m tport sw
                # Retry only transient "server not ready yet" races (client couldn't
                # open the out-of-band socket because the server hadn't bound the port).
                # A genuine unreachable pair connects on TCP but the RDMA QP never comes
                # up -> the client hits its timeout (rc=124); those are NOT retried, so
                # dead NICs don't waste attempts.
                for (( try=0; try<=MESH_RETRIES; try++ )); do
                    tport=$(( port + try * nr * nc ))   # fresh port each try (old server may linger)
                    local sa="-p $tport -x ${_cgid[${_col[$j]}]} --sl $MORI_RDMA_SL $extra"
                    local ca="-p $tport -x ${_rgid[${_row[$i]}]} --sl $MORI_RDMA_SL $extra"
                    # Server is wrapped in `timeout` so an unreachable pair can never make
                    # `wait` block forever (no client ever connects -> server self-exits).
                    if $is_local; then
                        timeout "$MESH_SRV_TIMEOUT" $tool -d "${_col[$j]}" $sa &>/dev/null &
                    else
                        "${SSH_PREFIX[@]}" ssh $SSH_OPTS "$SSH_USER@$shost" \
                            "timeout $MESH_SRV_TIMEOUT $tool -d ${_col[$j]} $sa" &>/dev/null &
                    fi
                    sp=$!
                    sw=$(awk "BEGIN{print $base_wait + $try*0.8}")   # wait longer on later tries
                    sleep "$sw"
                    rc=0
                    out=$(timeout "$MESH_CLI_TIMEOUT" $tool -d "${_row[$i]}" $ca "$shost" 2>&1) || rc=$?
                    kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
                    if (( rc == 0 )); then
                        m=$(echo "$out" | grep "^[[:space:]]*$key" | awk "{print $col}" | head -1)
                        [[ "$m" =~ ^[0-9.]+$ ]] && { result="$m"; break; }
                    fi
                    # stop unless this looks like a server-not-ready race
                    grep -qiE "couldn'?t connect|unable to (init|open).*socket|connection refused|read.*server" <<<"$out" || break
                done
                echo "$result" > "$tmpd/$i.$j"
            ) &
            running=$(( running + 1 ))
            if (( running >= MESH_PARALLEL )); then wait -n 2>/dev/null; running=$(( running - 1 )); fi
        done
    done
    wait

    for (( i=0; i<nr; i++ )); do
        for (( j=0; j<nc; j++ )); do
            local v; v=$(cat "$tmpd/$i.$j" 2>/dev/null)
            [[ "$v" =~ ^[0-9.]+$ || "$v" == "-" ]] || v="x"
            _cell["$i,$j"]="$v"
        done
    done
    rm -rf "$tmpd"
}

# mesh_report prints a reachability matrix + a value matrix + summary.
#   mesh_report <cell_assoc> <row_arr> <col_arr> <title> <unit> <fmt> [show_reach]
#   fmt: "int" (round) or "f1" (1 decimal) for the value matrix.
#   show_reach: "no" skips the reachability matrix (e.g. inter-node latency, where
#   the BW step already reported it); the summary line is still printed. Default "yes".
mesh_report() {
    local -n _cell="$1" _row="$2" _col="$3"
    local title="$4" unit="$5" fmt="$6" show_reach="${7:-yes}"
    local nr=${#_row[@]} nc=${#_col[@]}
    local i j ok=0 fail=0
    local hdr; hdr=$(printf "  %-10s" "")
    for (( j=0; j<nc; j++ )); do hdr+=$(printf " %6s" "$(echo "${_col[$j]}" | sed 's/bnxt_//')"); done

    if [[ "$show_reach" != "no" ]]; then
        echo ""
        echo -e "  $title reachability  (${GREEN}✓${NC}=ok ${RED}✗${NC}=fail '-'=self)  rows=client cols=server"
        echo "$hdr"
    fi
    for (( i=0; i<nr; i++ )); do
        local row; row=$(printf "  %-10s" "$(echo "${_row[$i]}" | sed 's/bnxt_//')")
        for (( j=0; j<nc; j++ )); do
            local c="${_cell[$i,$j]}"
            # 6 spaces + 1 glyph = 7 cols, matching the header's `printf " %6s"`
            # columns. Color escapes have zero display width, so pad by hand.
            if [[ "$c" == "-" ]]; then row+="      -"
            elif [[ "$c" == "x" ]]; then row+="      ${RED}✗${NC}"; (( fail++ ))
            else row+="      ${GREEN}✓${NC}"; (( ok++ )); fi
        done
        [[ "$show_reach" != "no" ]] && echo -e "$row"
    done

    echo ""
    echo "  $title $unit matrix"
    echo "$hdr"
    for (( i=0; i<nr; i++ )); do
        local row; row=$(printf "  %-10s" "$(echo "${_row[$i]}" | sed 's/bnxt_//')")
        for (( j=0; j<nc; j++ )); do
            local c="${_cell[$i,$j]}"
            if [[ "$c" =~ ^[0-9.]+$ ]]; then
                if [[ "$fmt" == "int" ]]; then c=$(printf "%.0f" "$c"); else c=$(printf "%.1f" "$c"); fi
            fi
            row+=$(printf " %6s" "$c")
        done
        echo "$row"
    done

    echo ""
    local total=$(( ok + fail ))
    if (( fail == 0 )); then log_ok "$title: all $total ordered pairs reachable"
    else log_warn "$title: $ok/$total reachable, $fail unreachable (see ✗ cells)"; fi
}

# version_ge <candidate> <min>
#   true if <candidate> >= <min>, comparing dotted/hyphenated version strings
#   (e.g. "1.117.5-a-45", "237.1.145.0") via `sort -V`.
version_ge() {
    local cand="$1" min="$2"
    [[ "$cand" == "$min" ]] && return 0
    [[ "$(printf '%s\n%s\n' "$cand" "$min" | sort -V | head -1)" == "$min" ]]
}

# check_ainic_version_recommendation <fw_version>
#   warns if the AINIC firmware is on the IBGDA-incapable 1.117.1 branch, or
#   below the recommended minimum for cross-node MORI (EP over RDMA / IBGDA).
check_ainic_version_recommendation() {
    local ver="$1"
    [[ -n "$ver" ]] || { log_warn "cannot verify AINIC firmware version against recommendation (empty)"; return; }
    if [[ "$ver" == 1.117.1* ]]; then
        log_warn "AINIC firmware $ver is on the 1.117.1 branch, which does NOT support IBGDA — upgrade to >= $AINIC_MIN_VER"
    elif version_ge "$ver" "$AINIC_MIN_VER"; then
        log_ok "AINIC firmware $ver meets the recommended minimum (>= $AINIC_MIN_VER) for cross-node IBGDA"
    else
        log_warn "AINIC firmware $ver is below the recommended minimum (>= $AINIC_MIN_VER) for cross-node IBGDA"
    fi
}

# check_bnxt_version_recommendation <fw_version>
#   classifies Broadcom firmware by major branch against known-good/known-bad
#   ranges for cross-node MORI (EP over RDMA / IBGDA).
check_bnxt_version_recommendation() {
    local ver="$1" major="${1%%.*}"
    [[ -n "$ver" ]] || { log_warn "cannot verify Broadcom firmware version against recommendation (empty)"; return; }
    case "$major" in
        231)
            log_warn "Broadcom firmware $ver is on the 231.x branch, which is too old for IBGDA — upgrade to >= $BNXT_MIN_VER_235 or >= $BNXT_MIN_VER_237"
            ;;
        235)
            version_ge "$ver" "$BNXT_MIN_VER_235" \
                && log_ok "Broadcom firmware $ver is solid (>= $BNXT_MIN_VER_235 on the 235.x branch)" \
                || log_warn "Broadcom firmware $ver is below the solid minimum on the 235.x branch (>= $BNXT_MIN_VER_235)"
            ;;
        237)
            version_ge "$ver" "$BNXT_MIN_VER_237" \
                && log_ok "Broadcom firmware $ver is solid (>= $BNXT_MIN_VER_237 on the 237.x branch)" \
                || log_warn "Broadcom firmware $ver is below the solid minimum on the 237.x branch (>= $BNXT_MIN_VER_237)"
            ;;
        *)
            log_warn "Broadcom firmware $ver is on an unverified branch ($major.x) — known-solid: $BNXT_MIN_VER_235, $BNXT_MIN_VER_237; known-bad: 231.x"
            ;;
    esac
}

# dominant_group <out_array_name> <dev...>
#   sets the named array to the largest same-vendor-prefix subset of the inputs.
dominant_group() {
    local -n _out="$1"; shift
    local -A _pref=(); local d p best="" bc=0
    for d in "$@"; do p=$(echo "$d" | sed 's/[0-9]*$//'); _pref["$p"]+="$d "; done
    for p in "${!_pref[@]}"; do
        local g=(); read -ra g <<< "${_pref[$p]}"
        (( ${#g[@]} > bc )) && { bc=${#g[@]}; best="$p"; }
    done
    read -ra _out <<< "${_pref[$best]}"
}

# ======================== check functions =======================

check_versions() {
    step "check ainic firmware and driver version"

    local fw_output sw_output
    fw_output=$(sudo nicctl show version firmware)
    sw_output=$(sudo nicctl show version host-software)

    local fw_versions fw_count
    fw_versions=$(echo "$fw_output" | grep -i "firmware" | awk '{print $NF}' | sort -u)
    fw_count=$(echo "$fw_versions" | wc -l)
    if [[ $fw_count -ne 1 ]]; then
        log_warn "firmware versions not consistent across NICs:"
        echo "$fw_versions"
        local v
        while read -r v; do check_ainic_version_recommendation "$v"; done <<< "$fw_versions"
    else
        log_ok "firmware         : $fw_versions"
        check_ainic_version_recommendation "$fw_versions"
    fi

    local nicctl_ver
    nicctl_ver=$(echo "$sw_output" | grep "nicctl" | awk '{print $NF}')
    [[ -n "$nicctl_ver" ]] && log_ok "nicctl           : $nicctl_ver" \
                           || log_fail "cannot determine nicctl version"

    local ionic_ver
    ionic_ver=$(echo "$sw_output" | grep "ionic driver" | awk '{print $NF}')
    [[ -n "$ionic_ver" ]] && log_ok "ionic driver     : $ionic_ver" \
                           || log_fail "cannot determine ionic driver version"
}

check_qos() {
    step "check QoS and derive SL/TC"

    local qos_output
    qos_output=$(sudo nicctl show qos)

    # classification type
    local class_type
    class_type=$(echo "$qos_output" | grep "Classification type" | head -1 | awk '{print $NF}')
    [[ "$class_type" == "DSCP" ]] || die "classification type is '$class_type', expected 'DSCP'"
    log_ok "classification type : DSCP"

    # no-drop priorities (may be a comma-separated list, e.g. "0,3")
    local nd_prio_raw
    nd_prio_raw=$(echo "$qos_output" | grep "PFC no-drop priorities" | head -1 | awk '{print $NF}')
    [[ -n "$nd_prio_raw" ]] || die "cannot find PFC no-drop priority"
    local nd_prios=()
    IFS=',' read -ra nd_prios <<< "$nd_prio_raw"
    log_ok "no-drop priorities : ${nd_prios[*]}"

    # PFC bitmap must cover every no-drop priority
    local pfc_bitmap
    pfc_bitmap=$(echo "$qos_output" | grep "PFC priority bitmap" | head -1 | awk '{print $NF}')
    [[ -n "$pfc_bitmap" && "$pfc_bitmap" != "0x0" ]] || die "PFC is not enabled (bitmap=$pfc_bitmap)"
    local p
    for p in "${nd_prios[@]}"; do
        (( pfc_bitmap & (1 << p) )) || die "PFC bitmap $pfc_bitmap does not cover priority $p"
    done
    log_ok "PFC enabled for priorities ${nd_prios[*]} (bitmap=$pfc_bitmap)"

    # For each no-drop priority, look up scheduling info and the DSCP list,
    # then pick the priority with the largest bandwidth share as our RDMA SL.
    local best_prio="" best_bw=-1 best_dscp=""
    for p in "${nd_prios[@]}"; do
        # Scheduling table rows look like:  "    3         DWRR        90        N/A"
        local sched_line sched_type sched_bw
        sched_line=$(echo "$qos_output" \
            | grep -E "^[[:space:]]+${p}[[:space:]]+(DWRR|SP|STRICT)[[:space:]]+" \
            | head -1)
        if [[ -z "$sched_line" ]]; then
            log_warn "cannot find scheduling info for priority $p"
            continue
        fi
        sched_type=$(echo "$sched_line" | awk '{print $2}')
        sched_bw=$(echo "$sched_line"   | awk '{print $3}')
        log_ok "scheduling for priority $p : $sched_type bw=${sched_bw}%"

        # DSCP list lines look like:  "    DSCP                      : 24, 46 ==> priority : 0"
        # (skip the "DSCP bitmap ..." variant)
        local dscp_line dscp_list first_dscp
        dscp_line=$(echo "$qos_output" \
            | grep -E "^[[:space:]]+DSCP[[:space:]]+:" \
            | grep -E "==>[[:space:]]+priority[[:space:]]+:[[:space:]]+${p}[[:space:]]*$" \
            | head -1)
        if [[ -z "$dscp_line" ]]; then
            log_warn "cannot find DSCP list for priority $p"
            continue
        fi
        # extract the chunk between "DSCP : " and " ==>"
        dscp_list=$(echo "$dscp_line" | sed -E 's/.*DSCP[[:space:]]+:[[:space:]]*//; s/[[:space:]]*==>.*//')

        # Pick a DSCP for this priority. Preference order:
        #   1) DSCP 26 (RoCEv2 convention; also what env_setup.sh explicitly maps)
        #   2) first concrete (non-range) token
        #   3) first range's lower bound
        local picked="" tok lo hi _toks=()
        IFS=',' read -ra _toks <<< "$dscp_list"
        for tok in "${_toks[@]}"; do
            tok=$(echo "$tok" | tr -d ' ')
            if [[ "$tok" == *-* ]]; then
                lo=${tok%-*}; hi=${tok#*-}
                if (( 26 >= lo && 26 <= hi )); then picked=26; break; fi
            elif [[ "$tok" == "26" ]]; then
                picked=26; break
            fi
        done
        if [[ -z "$picked" ]]; then
            for tok in "${_toks[@]}"; do
                tok=$(echo "$tok" | tr -d ' ')
                if [[ "$tok" != *-* && -n "$tok" ]]; then picked="$tok"; break; fi
            done
        fi
        if [[ -z "$picked" ]]; then
            tok=$(echo "$dscp_list" | awk -F',' '{print $1}' | tr -d ' ')
            picked=${tok%-*}
        fi
        first_dscp="$picked"
        if [[ -z "$first_dscp" ]]; then
            log_warn "cannot parse DSCP list '$dscp_list' for priority $p"
            continue
        fi
        log_ok "priority $p : DSCPs=[$dscp_list] -> picking DSCP $first_dscp"

        if (( sched_bw > best_bw )); then
            best_bw="$sched_bw"
            best_prio="$p"
            best_dscp="$first_dscp"
        fi
    done

    [[ -n "$best_prio" ]] || die "could not derive SL/TC from QoS info"

    # TC = DSCP << 2 (DSCP occupies the upper 6 bits of the 8-bit TC/TOS field)
    MORI_RDMA_SL="$best_prio"
    MORI_RDMA_TC=$(( best_dscp * 4 ))
    log_ok "selected SL=$MORI_RDMA_SL  TC=$MORI_RDMA_TC  (priority $best_prio, DSCP $best_dscp, bw=${best_bw}%)"
}

check_dcqcn() {
    step "check DCQCN"

    local dcqcn_output
    dcqcn_output=$(sudo nicctl show dcqcn)

    local total
    total=$(echo "$dcqcn_output" | grep -c "ROCE device")
    [[ $total -gt 0 ]] || die "no ROCE devices found in dcqcn output"

    local disabled
    disabled=$(echo "$dcqcn_output" | grep "Status" | grep -v "Enabled" || true)
    [[ -z "$disabled" ]] || { log_fail "some ROCE devices have DCQCN disabled:"; echo "$disabled"; exit 1; }
    log_ok "DCQCN enabled on all $total ROCE devices"

    local cnp_values cnp_count
    cnp_values=$(echo "$dcqcn_output" | grep "DSCP value used for CNP" | awk '{print $NF}' | sort -u)
    cnp_count=$(echo "$cnp_values" | wc -l)
    [[ $cnp_count -eq 1 ]] || die "CNP DSCP not consistent across NICs: $cnp_values"
    log_ok "CNP DSCP = $cnp_values (consistent across all NICs)"
}

check_intra_node_bw() {
    step "intra-node bandwidth check (full mesh)"

    command -v ib_write_bw > /dev/null 2>&1 || { log_warn "ib_write_bw not found, skipping"; return 0; }

    local all_devs=()
    mapfile -t all_devs < <(query_rdma_devices)
    [[ ${#all_devs[@]} -gt 0 ]] || { log_fail "no local RDMA devices found (check ibv_devices)"; return 1; }
    log_ok "local RDMA devices (${#all_devs[@]}): ${all_devs[*]}"

    # Pick the largest same-vendor group; exported via LOCAL_DEVS for inter-node tests.
    dominant_group LOCAL_DEVS "${all_devs[@]}"
    if (( ${#all_devs[@]} != ${#LOCAL_DEVS[@]} )); then
        log_warn "mixed NIC vendors detected; using ${#LOCAL_DEVS[@]} devices for tests: ${LOCAL_DEVS[*]}"
    fi
    [[ ${#LOCAL_DEVS[@]} -ge 2 ]] || { log_skip "only 1 device in dominant group, skipping"; return 0; }

    local n=${#LOCAL_DEVS[@]}
    # RDMA writes per pair scales with the test size (number of devices squared).
    local iters=$(( n * n ))
    # Intra-node pairs are all local (no ssh), so they aren't bound by sshd
    # MaxSessions -> use more parallelism than the inter-node mesh.
    local MESH_PARALLEL=$(( MESH_PARALLEL * 4 ))
    log_ok "full mesh over $n devices ($((n*(n-1))) ordered pairs, ${iters} writes/pair, parallel=$MESH_PARALLEL)"

    local -A GID=(); build_gid_map GID "" "${LOCAL_DEVS[@]}"
    local -A CELL=()
    mesh_execute CELL ib_write_bw localhost true "$iters" GID GID LOCAL_DEVS LOCAL_DEVS
    mesh_report CELL LOCAL_DEVS LOCAL_DEVS "intra-node BW" "Gbps" int
}

check_inter_node_bw() {
    step "inter-node bandwidth check (full mesh)"

    command -v ib_write_bw > /dev/null 2>&1 || { log_warn "ib_write_bw not found, skipping"; return 0; }
    if [[ -z "$PEER_IP" ]]; then
        log_skip "no peer IP provided (usage: $0 <peer_ip>)"; return 0
    fi
    ping -c 2 -W 2 "$PEER_IP" > /dev/null 2>&1 || die "cannot ping $PEER_IP, skip inter-node bandwidth test"
    log_ok "ping $PEER_IP reachable"
    ssh_warm "$PEER_IP"

    local all_remote=()
    mapfile -t all_remote < <(query_rdma_devices "$PEER_IP")
    [[ ${#all_remote[@]} -gt 0 ]] || { log_fail "no RDMA devices on $PEER_IP (check ibv_devices / ssh)"; return 1; }
    local REMOTE_DEVS=(); dominant_group REMOTE_DEVS "${all_remote[@]}"
    [[ ${#LOCAL_DEVS[@]} -gt 0 ]] || { log_fail "no local RDMA devices available"; return 1; }
    log_ok "local ${#LOCAL_DEVS[@]} x remote ${#REMOTE_DEVS[@]} mesh (parallel=$MESH_PARALLEL): ${REMOTE_DEVS[*]}"

    local -A LGID=(); build_gid_map LGID "" "${LOCAL_DEVS[@]}"
    local -A RGID=(); build_gid_map RGID "$PEER_IP" "${REMOTE_DEVS[@]}"
    local -A CELL=()
    # 1000 writes is plenty for a reachability + rough-BW probe; the default (5000)
    # just makes each working pair ~5x slower. no self-skip (distinct hosts).
    mesh_execute CELL ib_write_bw "$PEER_IP" false 1000 LGID RGID LOCAL_DEVS REMOTE_DEVS
    mesh_report CELL LOCAL_DEVS REMOTE_DEVS "inter-node BW" "Gbps" int
}

# =================== bnxt_re (Broadcom) checks =================
# All three functions below skip gracefully on non-bnxt hosts.
# They share the BNXT_DEVS / BNXT_ETH_DEVS arrays populated by
# check_bnxt_versions(); call that one first.

BNXT_DEVS=()        # bnxt_re IB device names      (e.g. bnxt_re_bond0)
BNXT_ETH_DEVS=()    # corresponding net devices    (e.g. enp30s0f0np0)
BNXT_NICCLI_IDX=1   # niccli index for first bond  (resolved in check_bnxt_versions)

# Populate BNXT_DEVS and BNXT_ETH_DEVS, check fw/driver/lib versions.
check_bnxt_versions() {
    step "check bnxt_re firmware and driver version (Broadcom NICs)"

    local ib_root="/sys/class/infiniband"
    if [[ ! -d "$ib_root" ]]; then
        log_skip "RDMA stack not loaded ($ib_root absent)"; return 0
    fi

    mapfile -t BNXT_DEVS < <(
        find "$ib_root" -maxdepth 1 -mindepth 1 -type l -printf '%f\n' \
        | grep '^bnxt_re' | sort -V)

    if [[ ${#BNXT_DEVS[@]} -eq 0 ]]; then
        log_skip "no bnxt_re devices found, skipping Broadcom checks"; return 0
    fi
    log_ok "bnxt_re devices (${#BNXT_DEVS[@]}): ${BNXT_DEVS[*]}"

    # --- kernel modules ---
    local m disk_ver load_ver
    for m in bnxt_re bnxt_en; do
        disk_ver=$(modinfo -F version "$m" 2>/dev/null || true)
        if [[ -r "/sys/module/$m/version" ]]; then
            load_ver=$(cat "/sys/module/$m/version")
        elif lsmod | awk '{print $1}' | grep -qx "$m"; then
            load_ver="(loaded, no version node)"
        else
            load_ver="(not loaded)"
        fi
        if [[ "${load_ver:0:1}" != "(" ]]; then
            log_ok "$m driver : $load_ver"
            [[ -n "$disk_ver" && "$disk_ver" != "$load_ver" ]] \
                && log_warn "$m on-disk ($disk_ver) differs from loaded ($load_ver) — reboot needed?"
        else
            [[ -n "$disk_ver" ]] && log_ok "$m driver (on-disk) : $disk_ver" \
                                 || log_fail "$m : not installed"
        fi
    done

    # --- RoCE userspace library ---
    local roce_lib_dir="/usr/local/lib"
    local found_ver
    found_ver=$(find "$roce_lib_dir" -maxdepth 2 -name 'libbnxt_re-*.so' 2>/dev/null \
        | sed -n 's|.*libbnxt_re-\([0-9][0-9.]*\)\.so$|\1|p' | sort -V | tail -1)
    if [[ -n "$found_ver" ]]; then
        log_ok "libbnxt_re userspace : $found_ver"
    else
        log_warn "libbnxt_re-<ver>.so not found under $roce_lib_dir"
    fi

    # --- firmware version via niccli -i <idx> show per NIC ---
    # Use the running "Firmware Version" / "RoCE Firmware Version" reported by
    # `niccli -i <idx> show` (the actual FW the NIC is running, e.g. 236.1.173.0),
    # NOT the "Active Package Version" from `show -p` (the NVM bundle version,
    # e.g. 36.11.73.00) which is a different identifier and confusing for RoCE.
    if ! command -v niccli >/dev/null 2>&1; then
        log_fail "niccli not found — cannot check firmware version"; return 1
    fi

    # get list of NIC indices from niccli --list (first column, skip header)
    local nic_indices=()
    mapfile -t nic_indices < <(sudo niccli --list 2>/dev/null | awk 'NR>1 && /^[[:space:]]*[0-9]/{gsub(/[^0-9]/,"",$1); print $1}')
    if [[ ${#nic_indices[@]} -eq 0 ]]; then
        log_warn "niccli --list returned no devices; defaulting to index 1"
        nic_indices=(1)
    fi

    # field <output> <label> -> value after the ':' for the line starting with <label>
    _niccli_field() { awk -F: -v k="$2" 'index($0,k)==1 {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}' <<<"$1"; }

    local fw_versions=() roce_versions=() failed_idxs=()
    local idx
    for idx in "${nic_indices[@]}"; do
        local show_out fw_ver roce_ver
        show_out=$(sudo niccli -i "$idx" show 2>/dev/null || true)
        fw_ver=$(_niccli_field "$show_out" "Firmware Version")
        roce_ver=$(_niccli_field "$show_out" "RoCE Firmware Version")
        if [[ -n "$fw_ver" ]]; then
            fw_versions+=("$fw_ver"); roce_versions+=("${roce_ver:-$fw_ver}")
        else
            failed_idxs+=("$idx")
        fi
    done

    _report_fw() {  # <label> <versions...>
        local label="$1"; shift
        [[ $# -gt 0 ]] || return 0
        local uniq; uniq=$(printf '%s\n' "$@" | sort -u)
        if [[ $(grep -c . <<<"$uniq") -gt 1 ]]; then
            log_warn "$label inconsistent across NICs:"; printf '         %s\n' $uniq
        else
            log_ok "$label : $uniq (consistent across all ${#nic_indices[@]} NICs)"
        fi
    }
    _report_fw "firmware"      "${fw_versions[@]}"
    _report_fw "RoCE firmware" "${roce_versions[@]}"
    [[ ${#failed_idxs[@]} -gt 0 ]] && log_warn "could not read firmware from NIC(s): ${failed_idxs[*]}"

    if [[ ${#roce_versions[@]} -gt 0 ]]; then
        local v
        while read -r v; do check_bnxt_version_recommendation "$v"; done \
            < <(printf '%s\n' "${roce_versions[@]}" | sort -u)
    fi

    # --- port state, net device, and niccli index mapping via sysfs ---
    # Build a PCI->niccli_index map from "niccli --list" output:
    #   "  1) BCM57608  <mac>  235.2.40.0  0000:06:00.0  NIC  PCI"
    declare -A _pci2idx=()
    local niccli_list
    niccli_list=$(sudo niccli --list 2>/dev/null || true)
    while IFS= read -r line; do
        local idx pci
        idx=$(echo "$line" | awk '/^[[:space:]]*[0-9]+\)/{gsub(/[^0-9]/,"",$1); print $1}')
        pci=$(echo "$line" | grep -oiE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | tr '[:upper:]' '[:lower:]')
        [[ -n "$idx" && -n "$pci" ]] && _pci2idx["$pci"]="$idx"
    done < <(echo "$niccli_list")

    local first_idx_set=false
    local dev state link eth_dev pci_addr
    local active_count=0 inactive_devs=()
    for dev in "${BNXT_DEVS[@]}"; do
        state=$(awk -F': *' '{print $2}' "$ib_root/$dev/ports/1/state" 2>/dev/null || true)
        link=$(cat "$ib_root/$dev/ports/1/link_layer" 2>/dev/null || true)
        pci_addr=$(basename "$(readlink -f "$ib_root/$dev/device")" 2>/dev/null || true)
        eth_dev=$(basename "$(readlink -f "$ib_root/$dev/device/net/"* 2>/dev/null)" 2>/dev/null || true)

        if [[ -n "${_pci2idx[$pci_addr]+x}" ]]; then
            local dev_idx="${_pci2idx[$pci_addr]}"
            [[ "$first_idx_set" == "false" ]] && { BNXT_NICCLI_IDX="$dev_idx"; first_idx_set=true; }
        fi

        local idx_label=""
        [[ -n "${_pci2idx[$pci_addr]+x}" ]] && idx_label="  niccli_idx=${_pci2idx[$pci_addr]}"

        if [[ "${state:-}" == "ACTIVE" ]]; then
            (( active_count++ ))
            log_ok "$dev : pci=$pci_addr  eth=${eth_dev:-?}  state=$state  link=${link:-?}$idx_label"
        else
            inactive_devs+=("$dev")
            log_warn "$dev : pci=$pci_addr  eth=${eth_dev:-?}  state=${state:-?}  link=${link:-?}$idx_label"
        fi
        [[ -n "$eth_dev" ]] && BNXT_ETH_DEVS+=("$eth_dev")
    done

    local total=${#BNXT_DEVS[@]}
    if [[ ${#inactive_devs[@]} -eq 0 ]]; then
        log_ok "all $total bnxt_re ports ACTIVE"
    else
        log_fail "$active_count/$total ports ACTIVE; not active: ${inactive_devs[*]}"
    fi
}

# Check PFC and DSCP-based QoS for bnxt_re via niccli; derive MORI_RDMA_SL / MORI_RDMA_TC.
check_bnxt_qos() {
    step "check bnxt_re QoS / PFC (Broadcom NICs)"

    if [[ ${#BNXT_DEVS[@]} -eq 0 ]]; then
        log_skip "no bnxt_re devices, run check_bnxt_versions first"; return 0
    fi

    if ! command -v niccli >/dev/null 2>&1; then
        log_warn "niccli not found, cannot check QoS/PFC"; return 0
    fi

    local fail=0

    # Use the first bond as the representative (QoS config is homogeneous across NICs).
    local rep_dev="${BNXT_DEVS[0]}"
    local nic_idx="$BNXT_NICCLI_IDX"

    # --- lossless TC check: ingress CoSQ ---
    # Output format:  TC   State     Mode
    #                  0   Enabled   Lossy
    #                  1   Enabled   Lossless
    local ingress_out lossless_tcs=()
    ingress_out=$(sudo niccli -i "$nic_idx" qos --ingress --cosq --show 2>/dev/null || true)
    if [[ -z "$ingress_out" ]]; then
        log_fail "$rep_dev : niccli qos --ingress --cosq --show returned nothing"; return 1
    fi
    while IFS= read -r line; do
        local tc mode
        tc=$(echo "$line"   | awk '/^[[:space:]]+[0-9]+[[:space:]]+Enabled/{print $1}')
        mode=$(echo "$line" | awk '/^[[:space:]]+[0-9]+[[:space:]]+Enabled/{print $3}')
        [[ -n "$tc" && "${mode,,}" == "lossless" ]] && lossless_tcs+=("$tc")
    done < <(echo "$ingress_out")

    if [[ ${#lossless_tcs[@]} -eq 0 ]]; then
        log_fail "$rep_dev : no lossless TC configured — PFC not active on NIC"; fail=1
    else
        log_ok "$rep_dev : lossless TC(s): ${lossless_tcs[*]}"
    fi

    # --- lossless bandwidth check: ETS TC Bandwidth ---
    local ets_out
    ets_out=$(sudo niccli -i "$nic_idx" qos --ets --show 2>/dev/null || true)
    if [[ -n "$ets_out" && ${#lossless_tcs[@]} -gt 0 ]]; then
        local bw_line
        bw_line=$(echo "$ets_out" | grep -i "TC Bandwidth")
        if [[ -n "$bw_line" ]]; then
            local bw_vals=()
            mapfile -t bw_vals < <(echo "$bw_line" | grep -oE '[0-9]+%' | tr -d '%')

            local total_bw=0 lossless_bw=0
            for tc in "${lossless_tcs[@]}"; do
                local bw="${bw_vals[$tc]:-0}"
                (( lossless_bw += bw ))
            done
            for bw in "${bw_vals[@]}"; do
                (( total_bw += bw ))
            done

            if [[ $total_bw -eq 0 ]]; then
                log_ok "$rep_dev : all TCs strict priority (ETS BW = 0%) — lossless TC has absolute precedence"
            else
                local pct=$(( lossless_bw * 100 / total_bw ))
                if (( pct >= 90 )); then
                    log_ok "$rep_dev : lossless TC bandwidth ${lossless_bw}% / ${total_bw}% total (${pct}% >= 90%)"
                else
                    log_warn "$rep_dev : lossless TC bandwidth ${lossless_bw}% / ${total_bw}% total (${pct}% < 90%)"
                fi
            fi
        fi

        # PFC status from ETS output
        local pfc_line
        pfc_line=$(echo "$ets_out" | grep -i "PFC enabled")
        if [[ "$pfc_line" == *none* ]]; then
            log_warn "$rep_dev : DCBx PFC = none"
        else
            local pfc_prios
            pfc_prios=$(echo "$pfc_line" | grep -oE '[0-9]+' | tr '\n' ' ')
            log_ok "$rep_dev : DCBx PFC enabled on priorities: ${pfc_prios% }"
        fi
    fi

    [[ $fail -eq 0 ]]
}


check_bnxt_dcqcn() {
    step "check bnxt_re DCQCN (Broadcom NICs)"

    if [[ ${#BNXT_DEVS[@]} -eq 0 ]]; then
        log_skip "no bnxt_re devices, run check_bnxt_versions first"; return 0
    fi

    local fail=0

    for dev in "${BNXT_DEVS[@]}"; do
        # Method 1: configfs (CNP_SERVICE_TYPE=0, driver-managed CC)
        local cc_path="/sys/kernel/config/bnxt_re/$dev/ports/1/cc"
        if mkdir -p "/sys/kernel/config/bnxt_re/$dev" 2>/dev/null && [[ -d "$cc_path" ]]; then
            local ecn_enable cc_mode
            ecn_enable=$(cat "$cc_path/ecn_enable" 2>/dev/null || true)
            cc_mode=$(cat    "$cc_path/cc_mode"    2>/dev/null || true)
            rmdir -p "/sys/kernel/config/bnxt_re/$dev" 2>/dev/null || true

            if [[ "$ecn_enable" == "0x1" || "$ecn_enable" == "1" ]] && \
               [[ "$cc_mode"    == "0x1" || "$cc_mode" == "1" ]]; then
                log_ok "$dev : DCQCN enabled (ecn_enable=$ecn_enable cc_mode=$cc_mode) [configfs]"
            else
                log_fail "$dev : DCQCN not enabled (ecn_enable=${ecn_enable:-?} cc_mode=${cc_mode:-?}) [configfs]"
                fail=1
            fi
            continue
        fi

        # Method 2: debugfs (CNP_SERVICE_TYPE=1, firmware-managed CC)
        local debug_info="/sys/kernel/debug/bnxt_re/$dev/info"
        if [[ -r "$debug_info" ]]; then
            local prof_type
            prof_type=$(grep "fw_service_prof_type_sup" "$debug_info" 2>/dev/null | awk '{print $3}')
            if [[ "$prof_type" == "1" ]]; then
                log_ok "$dev : DCQCN managed by firmware (fw_service_prof_type_sup=1) [debugfs]"
            else
                log_warn "$dev : cannot confirm DCQCN via debugfs (fw_service_prof_type_sup=${prof_type:-?})"
            fi
            continue
        fi

        log_warn "$dev : cannot determine DCQCN status (no configfs or debugfs access)"
    done

    [[ $fail -eq 0 ]]
}

check_inter_node_lat() {
    step "inter-node latency check (full mesh)"

    command -v ib_write_lat > /dev/null 2>&1 || { log_warn "ib_write_lat not found, skipping"; return 0; }
    if [[ -z "$PEER_IP" ]]; then
        log_skip "no peer IP provided (usage: $0 <peer_ip>)"; return 0
    fi
    ping -c 1 -W 2 "$PEER_IP" > /dev/null 2>&1 || die "cannot ping $PEER_IP, skip inter-node latency test"
    ssh_warm "$PEER_IP"

    local all_remote=()
    mapfile -t all_remote < <(query_rdma_devices "$PEER_IP")
    [[ ${#all_remote[@]} -gt 0 ]] || { log_fail "no RDMA devices on $PEER_IP"; return 1; }
    local REMOTE_DEVS=(); dominant_group REMOTE_DEVS "${all_remote[@]}"
    [[ ${#LOCAL_DEVS[@]} -gt 0 ]]  || { log_fail "no local RDMA devices available"; return 1; }
    log_ok "local ${#LOCAL_DEVS[@]} x remote ${#REMOTE_DEVS[@]} latency mesh (parallel=$MESH_PARALLEL)"

    local -A LGID=(); build_gid_map LGID "" "${LOCAL_DEVS[@]}"
    local -A RGID=(); build_gid_map RGID "$PEER_IP" "${REMOTE_DEVS[@]}"
    local -A CELL=()
    # latency uses ib_write_lat default iterations. Skip the reachability matrix
    # here: Step 5 (inter-node BW) already reported inter-node reachability.
    mesh_execute CELL ib_write_lat "$PEER_IP" false "" LGID RGID LOCAL_DEVS REMOTE_DEVS
    mesh_report CELL LOCAL_DEVS REMOTE_DEVS "inter-node latency" "us" f1 no
}

# ============================= main =============================

# ionic checks require nicctl AND real AMD/ionic NICs on this host.
# nicctl exits 0 even when no NIC is present, so check the output.
# [[ $EUID -eq 0 ]] || die "please run as root"

LOCAL_DEVS=()

# detect NIC vendor and run the matching checks
# Detect NICs by PCI vendor id, not by IB device name: ionic cards may show up as
# ionic_*, roceensp*, etc., so name matching is not reliable. The vendor id under
# /sys/class/infiniband/<dev>/device/vendor is stable.
_ib_has_vendor() {
    local vid="$1" d
    [[ -d /sys/class/infiniband ]] || return 1
    for d in /sys/class/infiniband/*; do
        [[ -e "$d/device/vendor" ]] || continue
        [[ "$(cat "$d/device/vendor" 2>/dev/null)" == "$vid" ]] && return 0
    done
    return 1
}

_have_ionic=false
_ib_has_vendor 0x1dd8 && _have_ionic=true   # AMD/Pensando (ionic)

# The ionic firmware/QoS/DCQCN checks all shell out to `sudo nicctl`. Probe that
# nicctl is installed AND can actually enumerate the cards; otherwise skip those
# checks gracefully instead of spewing "Invalid card handle" errors.
_nicctl_ok=false
if [[ "$_have_ionic" == "true" ]] && command -v nicctl >/dev/null 2>&1; then
    _nicctl_out=$(sudo nicctl show version firmware 2>&1 || true)
    if ! echo "$_nicctl_out" | grep -qiE 'No AMD NICs|Invalid card handle|Failed to get NIC'; then
        _nicctl_ok=true
    fi
    unset _nicctl_out
fi

_have_bnxt=false
_ib_has_vendor 0x14e4 && _have_bnxt=true     # Broadcom (bnxt_re)

_have_mlx=false
_ib_has_vendor 0x15b3 && _have_mlx=true      # Mellanox (mlx5)

if [[ "$_have_ionic" == "true" ]]; then
    if [[ "$_nicctl_ok" == "true" ]]; then
        check_versions
        check_qos
        check_dcqcn
    else
        log_warn "ionic NICs present but nicctl is unavailable or cannot access them — skipping nicctl-based checks (firmware / QoS / DCQCN)"
    fi
elif [[ "$_have_bnxt" == "true" ]]; then
    check_bnxt_versions
    check_bnxt_qos
    check_bnxt_dcqcn
elif [[ "$_have_mlx" == "true" ]]; then
    step "check mlx5 (Mellanox) NIC"
    log_ok "Mellanox ConnectX (mlx5) detected — no specific firmware/driver version requirement, IBGDA works out of the box via libmlx5"
else
    log_warn "no ionic, bnxt_re, or mlx5 NICs detected — skipping NIC-specific checks"
fi

check_intra_node_bw
check_inter_node_bw
check_inter_node_lat

echo ""
echo "=== All checks completed ==="
