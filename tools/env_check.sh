#!/bin/bash

# TODO: adapt for MLX (Mellanox/NVIDIA) NICs.

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
        ibv_devices 2>/dev/null | awk 'NR>2 && NF {print $1}'
    else
        ssh -o ConnectTimeout=5 "$(whoami)"@"$host" "ibv_devices 2>/dev/null" \
            | awk 'NR>2 && NF {print $1}'
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
        if awk "BEGIN{exit !($bw >= $BW_THRESHOLD)}"; then
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

    if awk "BEGIN{exit !($avg_lat <= $LAT_THRESHOLD)}"; then
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

    # --- firmware version via niccli -i <idx> show -p per NIC ---
    if ! command -v niccli >/dev/null 2>&1; then
        log_fail "niccli not found — cannot check firmware version"; return 1
    fi

    # get list of NIC indices from niccli -l (first column, skip header)
    local nic_indices=()
    mapfile -t nic_indices < <(sudo niccli -l 2>/dev/null | awk 'NR>1 && /^[[:space:]]*[0-9]/{gsub(/[^0-9]/,"",$1); print $1}')
    if [[ ${#nic_indices[@]} -eq 0 ]]; then
        log_warn "niccli -l returned no devices; defaulting to index 1"
        nic_indices=(1)
    fi

    local pkg_versions=() failed_idxs=()
    local idx
    for idx in "${nic_indices[@]}"; do
        local fw_out pkg_ver
        fw_out=$(sudo niccli -i "$idx" show -p 2>/dev/null || true)
        pkg_ver=$(echo "$fw_out" | awk '/Active Package Version/{print $NF}')
        if [[ -n "$pkg_ver" ]]; then
            pkg_versions+=("$pkg_ver")
        else
            failed_idxs+=("$idx")
        fi
    done

    local uniq_fw
    uniq_fw=$(printf '%s\n' "${pkg_versions[@]}" | sort -u)
    local uniq_count
    uniq_count=$(echo "$uniq_fw" | grep -c . || true)
    if [[ $uniq_count -gt 1 ]]; then
        log_warn "firmware versions inconsistent across NICs:"
        for idx in "${nic_indices[@]}"; do
            local fw_out pkg_ver
            fw_out=$(sudo niccli -i "$idx" show -p 2>/dev/null || true)
            pkg_ver=$(echo "$fw_out" | awk '/Active Package Version/{print $NF}')
            log_warn "  NIC $idx : ${pkg_ver:-unknown}"
        done
    elif [[ $uniq_count -eq 1 ]]; then
        log_ok "firmware : $uniq_fw (consistent across all ${#nic_indices[@]} NICs)"
    fi
    [[ ${#failed_idxs[@]} -gt 0 ]] && log_warn "could not read firmware from NIC(s): ${failed_idxs[*]}"

    # --- port state, net device, and niccli index mapping via sysfs ---
    # Build a PCI->niccli_index map from "niccli -l" output:
    #   "  1) BCM57608  <mac>  235.2.40.0  0000:06:00.0  NIC  PCI"
    declare -A _pci2idx=()
    local niccli_list
    niccli_list=$(sudo niccli -l 2>/dev/null || true)
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
                    log_fail "$rep_dev : lossless TC bandwidth ${lossless_bw}% / ${total_bw}% total (${pct}% < 90%)"; fail=1
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
               [[ "$cc_mode"    == "1" ]]; then
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

# ionic checks require nicctl AND real AMD/ionic NICs on this host.
# nicctl exits 0 even when no NIC is present, so check the output.
# [[ $EUID -eq 0 ]] || die "please run as root"

LOCAL_DEVS=()

# detect NIC vendor and run the matching checks
_have_ionic=false
if command -v nicctl >/dev/null 2>&1; then
    _nicctl_out=$(nicctl show version firmware 2>&1 || true)
    echo "$_nicctl_out" | grep -qi "No AMD NICs" || _have_ionic=true
    unset _nicctl_out
fi

_have_bnxt=false
if [[ -d /sys/class/infiniband ]] && \
   find /sys/class/infiniband -maxdepth 1 -name 'bnxt_re*' 2>/dev/null | grep -q .; then
    _have_bnxt=true
fi

if [[ "$_have_ionic" == "true" ]]; then
    check_versions
    check_qos
    check_dcqcn
elif [[ "$_have_bnxt" == "true" ]]; then
    check_bnxt_versions
    check_bnxt_qos
    check_bnxt_dcqcn
else
    log_warn "no ionic or bnxt_re NICs detected — skipping NIC-specific checks"
fi

check_intra_node_bw
check_inter_node_bw
check_inter_node_lat

echo ""
echo "=== All checks completed ==="
