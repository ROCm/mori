#!/bin/bash
#
# TODO: adapt for MLX (Mellanox/NVIDIA) NICs.
#
# env_setup.sh — setup the RDMA NIC environment for mori.
#
# Supports two NIC families with parallel, independent function sets selected by
# vendor at the bottom (if vendor == bnxt ... elif vendor == ionic ...):
#
#   ionic (AMD Pollara), via nicctl:
#     ionic_setup_pfc / ionic_setup_dcqcn / ionic_mori_env_setup
#   bnxt (Broadcom NetXtreme-E), via dcb + configfs:
#     bnxt_setup_pfc  / bnxt_setup_dcqcn  / bnxt_mori_env_setup
#
# Both paths converge on the same RoCE QoS constants below, so MORI_RDMA_SL / TC
# come out identical regardless of vendor.
#
# Usage:  source env_setup.sh
#
# Requires: ionic -> nicctl, sudo; bnxt -> dcb (iproute2), ethtool, configfs, sudo

IONIC_VENDOR_ID="0x1dd8"
BNXT_VENDOR_ID="0x14e4"   # Broadcom

# RoCE QoS constants — shared by the ionic and bnxt paths so MORI_RDMA_SL / TC
# come out identical regardless of NIC vendor.
ROCE_PRIO=3        # RoCE packet priority (lossless / PFC no-drop TC)
ROCE_DSCP=26       # RoCE DSCP
CNP_PRIO=6         # CNP packet priority
CNP_DSCP=48        # CNP DSCP
ROCE_BW=50         # % link bandwidth for the RoCE TC
L2_BW=$((100 - ROCE_BW))

GREEN='\033[0;32m' RED='\033[0;31m' YELLOW='\033[0;33m' NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()      { echo -e "${RED}[FAIL]${NC} $*"; return 1; }

query_by_vendor() {
    local vid="$1" dev vendor
    for dev in $(ibv_devices 2>/dev/null | awk 'NR>2 && NF {print $1}'); do
        vendor=$(cat "/sys/class/infiniband/$dev/device/vendor" 2>/dev/null)
        [[ "$vendor" == "$vid" ]] && echo "$dev"
    done
}
query_ionic_devices() { query_by_vendor "$IONIC_VENDOR_ID"; }
query_bnxt_devices()  { query_by_vendor "$BNXT_VENDOR_ID"; }

IONIC_DEVS=$(query_ionic_devices)
BNXT_DEVS=$(query_bnxt_devices)

# Wipe all QoS configuration back to a clean "best-effort only" state.
# Idiom adapted from the AMD Pollara 400 ops guide.
# Self-contained — safe to call after `source env_setup.sh` has unset its helpers.
ionic_reset_qos() {
    local p
    # 1) disable PFC no-drop on all 8 priorities (silently — some are already off)
    for p in 0 1 2 3 4 5 6 7; do
        sudo nicctl update qos pfc --priority "$p" --no-drop disable &>/dev/null
    done
    # 2) toggle classification type pcp <-> dscp to flush stale DSCP state
    sudo nicctl update qos --classification-type pcp  &>/dev/null
    sudo nicctl update qos --classification-type dscp &>/dev/null
    # 3) collapse every DSCP back to priority 0 in a single call (range syntax)
    if ! sudo nicctl update qos dscp-to-priority --dscp 0-63 --priority 0; then
        echo -e "\033[0;31m[FAIL]\033[0m reset DSCP 0-63 -> priority 0 failed" >&2
        return 1
    fi
    # 4) collapse scheduling so priority 0 owns the link
    sudo nicctl update qos scheduling --priority 0,1,2,3,4,5,6,7 \
        --dwrr 100,0,0,0,0,0,0,0 --rate-limit 0,0,0,0,0,0,0,0 &>/dev/null \
        || echo -e "\033[0;33m[WARN]\033[0m reset scheduling failed (continuing)" >&2
    echo -e "\033[0;32m[OK]\033[0m   QoS reset: all DSCPs -> priority 0, PFC no-drop disabled, scheduling collapsed"
}

ionic_setup_pfc() {
    command -v nicctl &>/dev/null || { die "ionic devices found but nicctl not available"; return 1; }
    sudo nicctl update qos --classification-type dscp                          || { die "set classification-type failed"; return 1; }
    sudo nicctl update port --all --pause-type pfc --rx-pause enable --tx-pause enable || { die "set pause failed"; return 1; }
    # DSCP-to-priority mapping MUST be done before scheduling — the firmware
    # rejects scheduling updates for priorities that have no DSCP entries yet
    # (nicctl returns "Invalid input" for priority N if N has no DSCP mapping).
    sudo nicctl update qos dscp-to-priority --dscp 26 --priority 3             || { die "map DSCP 26 -> priority 3 failed"; return 1; }
    sudo nicctl update qos dscp-to-priority --dscp 48 --priority 6             || { die "map DSCP 48 -> priority 6 failed"; return 1; }
    # Priority 6 = control/CNP lane: DWRR=0 + 10Gbps strict rate-limit
    # (matches the AMD Pollara reference recipe; bare dwrr=0/rate-limit=0
    # is rejected as "Invalid input").
    sudo nicctl update qos scheduling --priority 0,3,6 --dwrr 10,90,0 --rate-limit 0,0,10 || { die "set scheduling failed"; return 1; }
    sudo nicctl update qos pfc --priority 3 --no-drop enable                   || { die "enable PFC no-drop on priority 3 failed"; return 1; }
    sudo nicctl update port --all --admin-state up                             || { die "set admin-state up failed"; return 1; }
    log_ok "PFC / DSCP / scheduling configured"
}

ionic_setup_dcqcn() {
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

ionic_mori_env_setup() {
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

# ============================================================================
# bnxt (Broadcom NetXtreme-E) path — independent function set, parallel to the
# ionic_* functions above. Selected by vendor in the run section below.
# ============================================================================

_bnxt_cc_write() {  # <configfs_file> <value>
    local f="$1" v="$2"
    [[ -e "$f" ]] || return 0
    echo -n "$v" | sudo tee "$f" >/dev/null 2>&1 || log_warn "bnxt: write $v -> ${f##*/} failed"
}

# Detect the firmware CNP service-profile support (-> 0 or 1). When 1, CNP gets
# its own strict TC2 and the configfs prio/dscp fields are owned by firmware.
_bnxt_cnp_service_type() {  # <ib_dev>
    local info="/sys/kernel/debug/bnxt_re/$1/info" st=""
    [[ -f "$info" ]] && st=$(sudo awk '/fw_service_prof_type_sup/{print $3}' "$info" 2>/dev/null)
    [[ "$st" == "1" ]] && echo 1 || echo 0
}

# Remove any stale dcb app TLVs so a reconfigure doesn't accumulate entries.
_bnxt_dcb_clear_app() {  # <netdev>
    local nd="$1" line sel entry
    while IFS= read -r line; do
        [[ "$line" == *:* ]] || continue
        sel=$(awk '{print $1}' <<<"$line")
        for entry in $(awk '{$1=""; print}' <<<"$line"); do
            sudo dcb app del dev "$nd" "$sel" "$entry" 2>/dev/null
        done
    done < <(sudo dcb app show dev "$nd" 2>/dev/null)
    sudo dcb pfc set dev "$nd" prio-pfc all:off 2>/dev/null
}

# Configure PFC / ETS / DSCP on every bnxt_re port via `dcb`.
bnxt_setup_pfc() {
    command -v dcb &>/dev/null || { die "bnxt devices found but dcb not available"; return 1; }

    local dev ndev cnp_st prio_tc p
    for dev in $BNXT_DEVS; do
        ndev=$(cat "/sys/class/infiniband/$dev/ports/1/gid_attrs/ndevs/0" 2>/dev/null)
        [[ -n "$ndev" ]] || { log_warn "bnxt: no netdev for $dev, skipping"; continue; }
        cnp_st=$(_bnxt_cnp_service_type "$dev")

        # priority -> TC map: ROCE_PRIO -> lossless TC1; with CNP service type,
        # CNP_PRIO -> strict TC2; everything else -> best-effort TC0.
        prio_tc=""
        for p in 0 1 2 3 4 5 6 7; do
            if   [[ "$p" -eq "$ROCE_PRIO" ]];                          then prio_tc+=" $p:1"
            elif [[ "$cnp_st" == "1" && "$p" -eq "$CNP_PRIO" ]];       then prio_tc+=" $p:2"
            else                                                            prio_tc+=" $p:0"
            fi
        done

        # PFC replaces global link pause; clear stale app TLVs before reconfig
        sudo ethtool -A "$ndev" rx off tx off 2>/dev/null
        _bnxt_dcb_clear_app "$ndev"

        if [[ "$cnp_st" == "1" ]]; then
            # 3 TCs preferred; some firmware only accepts the full 8-TC form
            sudo dcb ets set dev "$ndev" tc-tsa 0:ets 1:ets 2:strict tc-bw 0:$L2_BW 1:$ROCE_BW prio-tc$prio_tc 2>/dev/null \
                || sudo dcb ets set dev "$ndev" tc-tsa 0:ets 1:ets 2:strict 3:strict 4:strict 5:strict 6:strict 7:strict tc-bw 0:$L2_BW 1:$ROCE_BW prio-tc$prio_tc \
                || { die "bnxt: dcb ets failed on $ndev"; return 1; }
        else
            sudo dcb ets set dev "$ndev" tc-tsa 0:ets 1:ets tc-bw 0:$L2_BW 1:$ROCE_BW prio-tc$prio_tc \
                || { die "bnxt: dcb ets failed on $ndev"; return 1; }
        fi

        sudo dcb pfc set dev "$ndev" prio-pfc all:off "$ROCE_PRIO":on \
            || { die "bnxt: dcb pfc failed on $ndev"; return 1; }
        # RoCEv2 UDP 4791 and the RoCE DSCP both map to the RoCE priority
        sudo dcb app add dev "$ndev" dgram-port-prio 4791:"$ROCE_PRIO" 2>/dev/null
        sudo dcb app add dev "$ndev" dscp-prio "$ROCE_DSCP":"$ROCE_PRIO" 2>/dev/null
        [[ "$cnp_st" == "1" ]] && sudo dcb app add dev "$ndev" dscp-prio "$CNP_DSCP":"$CNP_PRIO" 2>/dev/null
        log_ok "bnxt PFC/ETS on $dev ($ndev): prio $ROCE_PRIO lossless, DSCP $ROCE_DSCP, bw L2/$L2_BW RoCE/$ROCE_BW, cnp_service_type=$cnp_st"
    done
}

# Enable DCQCN (ECN + congestion control) on every bnxt_re device via configfs,
# and set the default RoCEv2 mode + ToS.
bnxt_setup_dcqcn() {
    local dev cc cnp_st tos=$(( ROCE_DSCP << 2 ))
    for dev in $BNXT_DEVS; do
        cnp_st=$(_bnxt_cnp_service_type "$dev")
        sudo mkdir -p "/sys/kernel/config/bnxt_re/$dev" 2>/dev/null
        cc="/sys/kernel/config/bnxt_re/$dev/ports/1/cc"
        if [[ ! -d "$cc" ]]; then
            log_warn "bnxt: configfs cc dir missing for $dev, skipping"
            sudo rmdir "/sys/kernel/config/bnxt_re/$dev" 2>/dev/null
            continue
        fi
        _bnxt_cc_write "$cc/disable_prio_vlan_tx" 0x1     # use DSCP-based PFC
        _bnxt_cc_write "$cc/ecn_marking"          0x1
        _bnxt_cc_write "$cc/ecn_enable"           0x1
        _bnxt_cc_write "$cc/cc_mode"              1       # DCQCN
        # With CNP service type the firmware owns the prio/dscp fields, so only
        # program them ourselves when service type is 0 (matches bnxt_setupcc.sh).
        if [[ "$cnp_st" != "1" ]]; then
            _bnxt_cc_write "$cc/roce_prio"        "$ROCE_PRIO"
            _bnxt_cc_write "$cc/cnp_prio"         "$CNP_PRIO"
            _bnxt_cc_write "$cc/roce_dscp"        "$ROCE_DSCP"
            _bnxt_cc_write "$cc/cnp_dscp"         "$CNP_DSCP"
        fi
        _bnxt_cc_write "$cc/apply"                0x1
        sudo rmdir "/sys/kernel/config/bnxt_re/$dev" 2>/dev/null

        # default RoCEv2 + ToS (bnxt needs this explicitly; ionic does not)
        sudo mkdir -p "/sys/kernel/config/rdma_cm/$dev" 2>/dev/null
        echo "RoCE v2" | sudo tee "/sys/kernel/config/rdma_cm/$dev/ports/1/default_roce_mode" >/dev/null 2>&1
        echo -n "$tos" | sudo tee "/sys/kernel/config/rdma_cm/$dev/ports/1/default_roce_tos"  >/dev/null 2>&1
        sudo rmdir "/sys/kernel/config/rdma_cm/$dev" 2>/dev/null

        log_ok "bnxt DCQCN on $dev: cc_mode=1, ecn on, roce_dscp=$ROCE_DSCP cnp_dscp=$CNP_DSCP, RoCEv2 tos=$tos"
    done
}

# Export MORI_RDMA_SL / MORI_RDMA_TC for the bnxt path (constants — bnxt has no
# nicctl to read back from, so we use the values we just programmed).
bnxt_mori_env_setup() {
    export MORI_RDMA_SL="$ROCE_PRIO"
    export MORI_RDMA_TC="$(( ROCE_DSCP << 2 ))"
    log_ok "export MORI_RDMA_SL=$MORI_RDMA_SL"
    log_ok "export MORI_RDMA_TC=$MORI_RDMA_TC"
}

# Dispatch by vendor.
if [[ -n "$BNXT_DEVS" ]]; then
    bnxt_setup_pfc  && bnxt_setup_dcqcn  && bnxt_mori_env_setup
elif [[ -n "$IONIC_DEVS" ]]; then
    ionic_setup_pfc && ionic_setup_dcqcn && ionic_mori_env_setup
else
    log_warn "no ionic or bnxt RDMA devices found"
fi

unset -f ionic_setup_pfc ionic_setup_dcqcn ionic_mori_env_setup \
         bnxt_setup_pfc bnxt_setup_dcqcn bnxt_mori_env_setup \
         _bnxt_cc_write _bnxt_cnp_service_type _bnxt_dcb_clear_app \
         query_by_vendor query_ionic_devices query_bnxt_devices \
         log_ok log_warn die
unset IONIC_VENDOR_ID BNXT_VENDOR_ID IONIC_DEVS BNXT_DEVS GREEN RED YELLOW NC
