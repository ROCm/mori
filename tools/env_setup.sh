#!/bin/bash
#
# TODO: adapt for MLX (Mellanox/NVIDIA) and BRCM (Broadcom) NICs —
#       currently ionic-specific (PFC/DSCP setup, DCQCN config, device enumeration).
#
# env_setup.sh — setup ionic NIC environment for mori.
#
# Steps (in order):
#   1. setup_pfc         — configure PFC / DSCP / scheduling on all ionic ports
#   2. setup_dcqcn       — configure DCQCN on all ionic ROCE devices
#   3. mori_env_setup    — export MORI_RDMA_SL / MORI_RDMA_TC from QoS config
#
# Usage:  source env_setup.sh
#
# Requires: nicctl, sudo

IONIC_VENDOR_ID="0x1dd8"

GREEN='\033[0;32m' RED='\033[0;31m' YELLOW='\033[0;33m' NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()      { echo -e "${RED}[FAIL]${NC} $*"; return 1; }

query_ionic_devices() {
    local dev vendor
    for dev in $(ibv_devices 2>/dev/null | awk '/^\s+[a-z]/ && $1 != "device" {print $1}'); do
        vendor=$(cat "/sys/class/infiniband/$dev/device/vendor" 2>/dev/null)
        [[ "$vendor" == "$IONIC_VENDOR_ID" ]] && echo "$dev"
    done
}

IONIC_DEVS=$(query_ionic_devices)
if [[ -z "$IONIC_DEVS" ]]; then
    log_warn "no ionic devices found (vendor=$IONIC_VENDOR_ID)"
    return 2>/dev/null || exit 0
fi
command -v nicctl &>/dev/null || { die "ionic devices found but nicctl not available"; return 2>/dev/null || exit 1; }

setup_pfc() {
    sudo nicctl update qos --classification-type dscp                          || { die "set classification-type failed"; return 1; }
    sudo nicctl update port --all --pause-type pfc --rx-pause enable --tx-pause enable || { die "set pause failed"; return 1; }
    sudo nicctl update qos dscp-to-priority --dscp 26 --priority 3             || { die "map DSCP 26 -> priority 3 failed"; return 1; }
    sudo nicctl update qos dscp-to-priority --dscp 48 --priority 6             || { die "map DSCP 48 -> priority 6 failed"; return 1; }
    sudo nicctl update qos pfc --priority 3 --no-drop enable                   || { die "enable PFC no-drop on priority 3 failed"; return 1; }
    sudo nicctl update qos scheduling --priority 0,3,6 --dwrr 10,90,0 --rate-limit 0,0,0 || { die "set scheduling failed"; return 1; }
    sudo nicctl update port --all --admin-state up                             || { die "set admin-state up failed"; return 1; }
    log_ok "PFC / DSCP / scheduling configured"
}

setup_dcqcn() {
    local dev
    for dev in $IONIC_DEVS; do
        sudo nicctl update dcqcn -r "$dev" -i 1 \
            --token-bucket-size 800000 \
            --ai-rate 160 \
            --alpha-update-interval 1 \
            --alpha-update-g 512 \
            --initial-alpha-value 64 \
            --rate-increase-byte-count 431068 \
            --hai-rate 300 \
            --rate-reduce-monitor-period 1 \
            --rate-increase-threshold 1 \
            --rate-increase-interval 1 \
            --cnp-dscp 46 \
            || { die "DCQCN setup failed for $dev"; return 1; }
        log_ok "DCQCN configured on $dev"
    done
}

mori_env_setup() {
    local qos
    qos=$(sudo nicctl show qos) || die "nicctl show qos failed"

    local class_type
    class_type=$(echo "$qos" | grep "Classification type" | head -1 | awk '{print $NF}')
    [[ "$class_type" == "DSCP" ]] || die "classification type is '$class_type', expected 'DSCP'"

    local nd_prio
    nd_prio=$(echo "$qos" | grep "PFC no-drop priorities" | head -1 | awk '{print $NF}')
    [[ -n "$nd_prio" ]] || die "cannot find PFC no-drop priority"

    local pfc_bitmap
    pfc_bitmap=$(echo "$qos" | grep "PFC priority bitmap" | head -1 | awk '{print $NF}')
    if [[ -z "$pfc_bitmap" || "$pfc_bitmap" == "0x0" ]]; then
        log_warn "PFC not enabled (bitmap=$pfc_bitmap)"
    elif ! (( pfc_bitmap & (1 << nd_prio) )); then
        log_warn "PFC bitmap $pfc_bitmap does not cover priority $nd_prio"
    fi

    local dscp_line nd_dscp
    dscp_line=$(echo "$qos" | grep "DSCP" | grep "==>" | grep -v "bitmap" | grep ": ${nd_prio}$" | head -1)
    nd_dscp=$(echo "$dscp_line" | awk -F': ' '{print $2}' | grep -o '[0-9]*' | head -1)
    [[ -n "$nd_dscp" ]] || die "cannot find DSCP mapped to no-drop priority $nd_prio"

    local tc=$(( nd_dscp * 4 ))

    export MORI_RDMA_SL="$nd_prio"
    export MORI_RDMA_TC="$tc"

    log_ok "export MORI_RDMA_SL=$MORI_RDMA_SL"
    log_ok "export MORI_RDMA_TC=$MORI_RDMA_TC"
}

setup_pfc && setup_dcqcn && mori_env_setup

unset -f setup_pfc setup_dcqcn mori_env_setup query_ionic_devices log_ok log_warn die
unset IONIC_VENDOR_ID IONIC_DEVS GREEN RED YELLOW NC
