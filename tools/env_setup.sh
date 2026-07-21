#!/bin/bash
#
# env_setup.sh — setup the RDMA NIC environment for mori.
#
# Supports three NIC families with parallel, independent function sets selected
# by vendor at the bottom (if vendor == bnxt ... elif vendor == ionic ...):
#
#   ionic (AMD Pollara), via nicctl:
#     ionic_setup_pfc / ionic_setup_dcqcn / ionic_mori_env_setup
#   bnxt (Broadcom NetXtreme-E), via dcb + configfs:
#     bnxt_setup_pfc  / bnxt_setup_dcqcn  / bnxt_mori_env_setup
#   mlx5 (Mellanox/NVIDIA ConnectX), via mlnx_qos + mlxconfig:
#     mlx5_setup_pfc  / mlx5_setup_dcqcn  / mlx5_mori_env_setup
#
# All three paths converge on the same RoCE QoS constants below, so
# MORI_RDMA_SL / TC come out identical regardless of vendor.
#
# Usage:  source env_setup.sh
#
# Requires: ionic -> nicctl, sudo; bnxt -> dcb (iproute2), ethtool, configfs,
#           sudo; mlx5 -> mlnx_qos (mlnx-tools), mlxconfig/mst (NVIDIA MFT), sudo

IONIC_VENDOR_ID="0x1dd8"
BNXT_VENDOR_ID="0x14e4"   # Broadcom
MLX_VENDOR_ID="0x15b3"    # Mellanox/NVIDIA

# RoCE QoS constants — shared by the ionic/bnxt/mlx5 paths so MORI_RDMA_SL / TC
# come out identical regardless of NIC vendor.
#
# NOTE: reference defaults only. PFC is hop-by-hop — the switch must use the same
# priority / DSCP / trust mode, or align these constants with the switch.
ROCE_PRIO=3        # RoCE packet priority (lossless / PFC no-drop)
ROCE_DSCP=26       # RoCE DSCP
CNP_PRIO=6         # CNP packet priority
CNP_DSCP=48        # CNP DSCP
ROCE_BW=90         # % link bandwidth for the RoCE TC (lossless)
NON_RDMA_BW=$((100 - ROCE_BW))   # % for non-RDMA traffic on TC0

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

# Only Ethernet/RoCE mlx5 ports use mlnx_qos/mlxconfig (native IB has its own
# flow control), so filter those in.
query_mlx5_devices() {
    local dev
    for dev in $(query_by_vendor "$MLX_VENDOR_ID"); do
        [[ "$(cat "/sys/class/infiniband/$dev/ports/1/link_layer" 2>/dev/null)" == "Ethernet" ]] && echo "$dev"
    done
}

IONIC_DEVS=$(query_ionic_devices)
BNXT_DEVS=$(query_bnxt_devices)
MLX5_ROCE_DEVS=$(query_mlx5_devices)

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
            sudo dcb ets set dev "$ndev" tc-tsa 0:ets 1:ets 2:strict tc-bw 0:$NON_RDMA_BW 1:$ROCE_BW prio-tc$prio_tc 2>/dev/null \
                || sudo dcb ets set dev "$ndev" tc-tsa 0:ets 1:ets 2:strict 3:strict 4:strict 5:strict 6:strict 7:strict tc-bw 0:$NON_RDMA_BW 1:$ROCE_BW prio-tc$prio_tc \
                || { die "bnxt: dcb ets failed on $ndev"; return 1; }
        else
            sudo dcb ets set dev "$ndev" tc-tsa 0:ets 1:ets tc-bw 0:$NON_RDMA_BW 1:$ROCE_BW prio-tc$prio_tc \
                || { die "bnxt: dcb ets failed on $ndev"; return 1; }
        fi

        sudo dcb pfc set dev "$ndev" prio-pfc all:off "$ROCE_PRIO":on \
            || { die "bnxt: dcb pfc failed on $ndev"; return 1; }
        # RoCEv2 UDP 4791 and the RoCE DSCP both map to the RoCE priority
        sudo dcb app add dev "$ndev" dgram-port-prio 4791:"$ROCE_PRIO" 2>/dev/null
        sudo dcb app add dev "$ndev" dscp-prio "$ROCE_DSCP":"$ROCE_PRIO" 2>/dev/null
        [[ "$cnp_st" == "1" ]] && sudo dcb app add dev "$ndev" dscp-prio "$CNP_DSCP":"$CNP_PRIO" 2>/dev/null
        log_ok "bnxt PFC/ETS on $dev ($ndev): prio $ROCE_PRIO lossless, DSCP $ROCE_DSCP, bw RoCE/$ROCE_BW non-RDMA/$NON_RDMA_BW, cnp_service_type=$cnp_st"
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

# ============================================================================
# mlx5 (Mellanox/NVIDIA ConnectX) path — parallel to the ionic_*/bnxt_*
# functions above. Only Ethernet/RoCE ports are touched (query_mlx5_devices).
# ============================================================================

# Configure trust/DSCP/PFC on every mlx5 RoCE port via `mlnx_qos`. TC
# arbitration is left to the NIC's "vendor" algorithm (no bandwidth split);
# only PFC no-drop on the RoCE priority is set up.
mlx5_setup_pfc() {
    command -v mlnx_qos &>/dev/null || { die "mlx5 devices found but mlnx_qos not available"; return 1; }

    local p pfc_bits=""
    for p in 0 1 2 3 4 5 6 7; do
        [[ "$p" -eq "$ROCE_PRIO" ]] && pfc_bits+="1," || pfc_bits+="0,"
    done
    pfc_bits="${pfc_bits%,}"

    local dev eth
    for dev in $MLX5_ROCE_DEVS; do
        eth=$(basename "$(readlink -f "/sys/class/infiniband/$dev/device/net/"* 2>/dev/null)" 2>/dev/null)
        [[ -n "$eth" ]] || { log_warn "mlx5: no netdev for $dev, skipping"; continue; }

        sudo mlnx_qos -i "$eth" --trust=dscp                              &>/dev/null \
            || { die "mlx5: set trust dscp failed on $eth"; return 1; }
        sudo mlnx_qos -i "$eth" --dscp2prio=set,"$ROCE_DSCP","$ROCE_PRIO"  &>/dev/null \
            || { die "mlx5: map DSCP $ROCE_DSCP -> priority $ROCE_PRIO failed on $eth"; return 1; }
        sudo mlnx_qos -i "$eth" --pfc="$pfc_bits"                          &>/dev/null \
            || { die "mlx5: set PFC bitmap failed on $eth"; return 1; }
        sudo mlnx_qos -i "$eth" --tsa=vendor,vendor,vendor,vendor,vendor,vendor,vendor,vendor &>/dev/null \
            || log_warn "mlx5: set vendor TSA failed on $eth (continuing)"

        log_ok "mlx5 PFC/DSCP on $dev ($eth): trust=dscp, DSCP $ROCE_DSCP -> priority $ROCE_PRIO (no-drop), TSA=vendor"
    done
}

# Enable DCQCN on every mlx5 RoCE port via `mlxconfig` (NV config
# ROCE_CC_PRIO_MASK_P1); CNP DSCP and the CC algorithm stay at firmware
# default. NV config needs a firmware reset (mlxfwreset) or reboot to take
# effect — we only warn, since a reset drops the link and can disrupt traffic.
mlx5_setup_dcqcn() {
    command -v mlxconfig &>/dev/null || { die "mlx5 devices found but mlxconfig (NVIDIA MFT) not available"; return 1; }
    command -v mst &>/dev/null && sudo mst start &>/dev/null

    local dev pci mask reset_needed=0
    for dev in $MLX5_ROCE_DEVS; do
        pci=$(basename "$(readlink -f "/sys/class/infiniband/$dev/device")" 2>/dev/null)
        [[ -n "$pci" ]] || { log_warn "mlx5: cannot resolve PCI address for $dev, skipping"; continue; }

        mask=$(sudo mlxconfig -d "$pci" q 2>/dev/null | grep -i "ROCE_CC_PRIO_MASK_P1" | awk '{print $NF}')
        if [[ -n "$mask" && "$mask" != "0" ]]; then
            log_ok "mlx5 DCQCN already enabled on $dev (pci=$pci, ROCE_CC_PRIO_MASK_P1=$mask)"
            continue
        fi

        sudo mlxconfig -d "$pci" -y set ROCE_CC_PRIO_MASK_P1=0xff &>/dev/null \
            || { die "mlx5: mlxconfig set ROCE_CC_PRIO_MASK_P1 failed on $dev (pci=$pci)"; return 1; }
        log_ok "mlx5 DCQCN enabled on $dev (pci=$pci, ROCE_CC_PRIO_MASK_P1=0xff)"
        reset_needed=1
    done

    [[ "$reset_needed" == "1" ]] && \
        log_warn "mlx5: DCQCN NV config changed — run 'mlxfwreset -d <pci> -y reset' (or reboot) to apply"
}

# Export MORI_RDMA_SL / MORI_RDMA_TC for the mlx5 path (constants — same
# rationale as bnxt_mori_env_setup: we just programmed these values ourselves).
mlx5_mori_env_setup() {
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
elif [[ -n "$MLX5_ROCE_DEVS" ]]; then
    mlx5_setup_pfc  && mlx5_setup_dcqcn  && mlx5_mori_env_setup
else
    log_warn "no ionic, bnxt, or mlx5 RoCE devices found"
fi

unset -f ionic_setup_pfc ionic_setup_dcqcn ionic_mori_env_setup \
         bnxt_setup_pfc bnxt_setup_dcqcn bnxt_mori_env_setup \
         mlx5_setup_pfc mlx5_setup_dcqcn mlx5_mori_env_setup \
         _bnxt_cc_write _bnxt_cnp_service_type _bnxt_dcb_clear_app \
         query_by_vendor query_ionic_devices query_bnxt_devices query_mlx5_devices \
         log_ok log_warn die
unset IONIC_VENDOR_ID BNXT_VENDOR_ID MLX_VENDOR_ID IONIC_DEVS BNXT_DEVS MLX5_ROCE_DEVS GREEN RED YELLOW NC
