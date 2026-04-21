#!/bin/bash

# TODO: adapt for MLX (Mellanox/NVIDIA) and BRCM (Broadcom) NICs —
#       currently ionic-specific (device detection, PFC/DCQCN knobs, port enumeration).

set -uo pipefail

# ============================ config ============================

PEER_IP="${1:-}"
BW_THRESHOLD=300    # Gbps
LAT_THRESHOLD=10    # microseconds
MSG_SIZE=65536      # 64K
LAT_MSG_SIZE=2      # bytes (small message for latency)
LAT_ITERS=5000      # iterations for latency test
TEST_DURATION=2     # seconds
IB_PORT=18515       # base port for ib_write_bw / ib_write_lat
MORI_RDMA_SL=0      # overwritten by check_qos()
MORI_RDMA_TC=0      # overwritten by check_qos()

# ============================ helpers ===========================

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

# query_rdma_devices [host]
#   prints one RDMA device name per line; empty host = local
query_rdma_devices() {
    local host="${1:-}"
    if [[ -z "$host" || "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        ibv_devices 2>/dev/null | awk '/^\s+[a-z]/ && $1 != "device" {print $1}'
    else
        ssh -o ConnectTimeout=5 "$(whoami)"@"$host" "ibv_devices 2>/dev/null" \
            | awk '/^\s+[a-z]/ && $1 != "device" {print $1}'
    fi
}

# run_ib_bw_test <client_dev> <server_dev> <server_host> [check_threshold]
#   check_threshold: if "true" (default), compare BW against BW_THRESHOLD
run_ib_bw_test() {
    local client_dev="$1" server_dev="$2" server_host="$3"
    local check="${4:-true}"
    local port=$((IB_PORT++))
    local label="$client_dev -> $server_dev@$server_host"
    local ib_args="-x 1 -p $port -s $MSG_SIZE -D $TEST_DURATION \
        --report_gbits --sl $MORI_RDMA_SL"

    if [[ "$server_host" == "localhost" || "$server_host" == "127.0.0.1" ]]; then
        ib_write_bw -d "$server_dev" $ib_args &>/dev/null &
    else
        ssh "$(whoami)"@"$server_host" "ib_write_bw -d $server_dev $ib_args" &>/dev/null &
    fi
    local server_pid=$!
    sleep 1

    local output rc=0
    output=$(ib_write_bw -d "$client_dev" $ib_args "$server_host" 2>&1) || rc=$?
    wait "$server_pid" 2>/dev/null

    if [[ $rc -ne 0 ]]; then
        log_fail "$label : ib_write_bw failed (rc=$rc)"; return 1
    fi

    # parse BW from the data line (format: "65536  ...  <BW_avg>  ...")
    local bw
    bw=$(echo "$output" | grep "^[[:space:]]*$MSG_SIZE" | awk '{print $(NF-1)}')
    if [[ -z "$bw" ]]; then
        log_fail "$label : cannot parse bandwidth"; return 1
    fi

    if [[ "$check" == "true" ]]; then
        if (( $(echo "$bw >= $BW_THRESHOLD" | bc -l) )); then
            log_ok "$label : ${bw} Gbps"
        else
            log_fail "$label : ${bw} Gbps (threshold: ${BW_THRESHOLD} Gbps)"; return 1
        fi
    else
        log_ok "$label : ${bw} Gbps"
    fi
}

# run_ib_lat_test <client_dev> <server_dev> <server_host>
#   returns 0 if avg latency <= threshold, 1 otherwise
run_ib_lat_test() {
    local client_dev="$1" server_dev="$2" server_host="$3"
    local port=$((IB_PORT++))
    local label="$client_dev -> $server_dev@$server_host"
    local ib_args="-x 1 -p $port -s $LAT_MSG_SIZE -n $LAT_ITERS --sl $MORI_RDMA_SL"

    ssh "$(whoami)"@"$server_host" "ib_write_lat -d $server_dev $ib_args" &>/dev/null &
    local server_pid=$!
    sleep 1

    local output rc=0
    output=$(ib_write_lat -d "$client_dev" $ib_args "$server_host" 2>&1) || rc=$?
    wait "$server_pid" 2>/dev/null

    if [[ $rc -ne 0 ]]; then
        log_fail "$label : ib_write_lat failed (rc=$rc)"; return 1
    fi

    # columns: #bytes #iterations t_min t_max t_typical t_avg t_stdev 99% 99.9%
    local avg_lat
    avg_lat=$(echo "$output" | grep "^[[:space:]]*$LAT_MSG_SIZE" | awk '{print $6}')
    if [[ -z "$avg_lat" ]]; then
        log_fail "$label : cannot parse latency"; return 1
    fi

    if (( $(echo "$avg_lat <= $LAT_THRESHOLD" | bc -l) )); then
        log_ok "$label : ${avg_lat} us"
    else
        log_fail "$label : ${avg_lat} us (threshold: ${LAT_THRESHOLD} us)"; return 1
    fi
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
    else
        log_ok "firmware         : $fw_versions"
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

    # no-drop priority
    local nd_prio
    nd_prio=$(echo "$qos_output" | grep "PFC no-drop priorities" | head -1 | awk '{print $NF}')
    [[ -n "$nd_prio" ]] || die "cannot find PFC no-drop priority"
    log_ok "no-drop priority : $nd_prio"

    # PFC bitmap
    local pfc_bitmap
    pfc_bitmap=$(echo "$qos_output" | grep "PFC priority bitmap" | head -1 | awk '{print $NF}')
    [[ -n "$pfc_bitmap" && "$pfc_bitmap" != "0x0" ]] || die "PFC is not enabled (bitmap=$pfc_bitmap)"
    (( pfc_bitmap & (1 << nd_prio) )) || die "PFC bitmap $pfc_bitmap does not cover priority $nd_prio"
    log_ok "PFC enabled for priority $nd_prio (bitmap=$pfc_bitmap)"

    # scheduling info
    local sched_line
    sched_line=$(echo "$qos_output" | grep -E "^\s+${nd_prio}\s+" | head -1)
    if [[ -n "$sched_line" ]]; then
        log_ok "scheduling for priority $nd_prio : $(echo "$sched_line" | awk '{print $2}')  bw=$(echo "$sched_line" | awk '{print $3}')"
    else
        log_warn "cannot find scheduling info for priority $nd_prio"
    fi

    # DSCP -> traffic class
    local dscp_line nd_dscp
    dscp_line=$(echo "$qos_output" | grep "DSCP" | grep "==>" | grep -v "bitmap" | grep ": ${nd_prio}$" | head -1)
    nd_dscp=$(echo "$dscp_line" | awk -F': ' '{print $2}' | grep -o '[0-9]*' | head -1)
    [[ -n "$nd_dscp" ]] || die "cannot find DSCP mapped to no-drop priority $nd_prio"

    # TC = DSCP << 2 (DSCP occupies the upper 6 bits of the 8-bit TC/TOS field)
    MORI_RDMA_SL="$nd_prio"
    MORI_RDMA_TC=$(( nd_dscp * 4 ))

    log_ok "no-drop DSCP     : $nd_dscp"
    log_ok "traffic class    : $MORI_RDMA_TC"
    log_ok "SL=$MORI_RDMA_SL  TC=$MORI_RDMA_TC"
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
    step "intra-node bandwidth check"

    command -v ib_write_bw > /dev/null 2>&1 || { log_warn "ib_write_bw not found, skipping"; return 0; }

    LOCAL_DEVS=($(query_rdma_devices))
    local count=${#LOCAL_DEVS[@]}
    [[ $count -gt 0 ]] || { log_fail "no local RDMA devices found (check ibv_devices)"; return 1; }
    log_ok "local RDMA devices ($count): ${LOCAL_DEVS[*]}"

    if [[ $count -lt 2 ]]; then
        log_skip "only 1 local RDMA device, skipping intra-node test"; return 0
    fi

    for (( i=1; i<count; i++ )); do
        run_ib_bw_test "${LOCAL_DEVS[0]}" "${LOCAL_DEVS[$i]}" "localhost" "false"
    done
}

check_inter_node_bw() {
    step "inter-node bandwidth check"

    command -v ib_write_bw > /dev/null 2>&1 || { log_warn "ib_write_bw not found, skipping"; return 0; }

    if [[ -z "$PEER_IP" ]]; then
        log_skip "no peer IP provided (usage: $0 <peer_ip>)"; return 0
    fi

    ping -c 2 -W 2 "$PEER_IP" > /dev/null 2>&1 || die "cannot ping $PEER_IP, skip inter-node bandwidth test"
    log_ok "ping $PEER_IP reachable"

    local remote_devs
    remote_devs=($(query_rdma_devices "$PEER_IP"))
    [[ ${#remote_devs[@]} -gt 0 ]] || { log_fail "no RDMA devices on $PEER_IP (check ibv_devices / ssh)"; return 1; }
    log_ok "remote RDMA devices: ${remote_devs[*]}"

    [[ ${#LOCAL_DEVS[@]} -gt 0 ]] || { log_fail "no local RDMA devices available"; return 1; }

    local fail=0
    for rdev in "${remote_devs[@]}"; do
        run_ib_bw_test "${LOCAL_DEVS[0]}" "$rdev" "$PEER_IP" || fail=1
    done

    [[ $fail -eq 0 ]] && log_ok  "all inter-node pairs passed (>= ${BW_THRESHOLD} Gbps)" \
                       || log_fail "some inter-node pairs failed"
}

check_inter_node_lat() {
    step "inter-node latency check"

    command -v ib_write_lat > /dev/null 2>&1 || { log_warn "ib_write_lat not found, skipping"; return 0; }

    if [[ -z "$PEER_IP" ]]; then
        log_skip "no peer IP provided (usage: $0 <peer_ip>)"; return 0
    fi

    ping -c 1 -W 2 "$PEER_IP" > /dev/null 2>&1 || die "cannot ping $PEER_IP, skip inter-node latency test"

    local remote_devs
    remote_devs=($(query_rdma_devices "$PEER_IP"))
    [[ ${#remote_devs[@]} -gt 0 ]] || { log_fail "no RDMA devices on $PEER_IP"; return 1; }
    [[ ${#LOCAL_DEVS[@]} -gt 0 ]]  || { log_fail "no local RDMA devices available"; return 1; }

    local fail=0
    for rdev in "${remote_devs[@]}"; do
        run_ib_lat_test "${LOCAL_DEVS[0]}" "$rdev" "$PEER_IP" || fail=1
    done

    [[ $fail -eq 0 ]] && log_ok  "all inter-node pairs passed (<= ${LAT_THRESHOLD} us)" \
                       || log_fail "some inter-node pairs failed"
}

# ============================= main =============================

require_cmd nicctl
# [[ $EUID -eq 0 ]] || die "please run as root"

LOCAL_DEVS=()

check_versions
check_qos
check_dcqcn
check_intra_node_bw
check_inter_node_bw
check_inter_node_lat

echo ""
echo "=== All checks completed ==="
