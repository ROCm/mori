#!/bin/bash
# umbp_spdk_preflight.sh — Pre-flight check for UMBP SPDK backend
#
# Usage:
#   umbp_spdk_preflight.sh                          # auto-detect PCI devices
#   umbp_spdk_preflight.sh --pci 0000:89:00.0       # check specific device
#   umbp_spdk_preflight.sh --help
#
# Checks all prerequisites for running UMBP with SPDK:
#   1. Hugepages allocated
#   2. Hugetlbfs mounted
#   3. VFIO kernel support
#   4. NVMe device bound to vfio-pci
#   5. VFIO group device accessible
#   6. UMBP proxy discovery (informational only; pip install can package it)
#   7. SPDK shared libraries installed

set -o pipefail

PASS=0
FAIL=0
WARN=0
HOST_FAIL=0
HOST_WARN=0
RUNTIME_FAIL=0
RUNTIME_WARN=0
TARGET_PCI=""
TARGET_IOMMU_GROUP=""
NEEDS_HOST_SETUP=false
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MORI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPDK_SETUP_REL="./3rdparty/spdk/scripts/setup.sh"
IS_CONTAINER=false

first_sched=""
read -r first_sched _ < /proc/1/sched 2>/dev/null || true
if [[ -f /.dockerenv ]] || grep -q docker /proc/1/cgroup 2>/dev/null || \
   [[ -n "$first_sched" && "$first_sched" != systemd* ]]; then
    IS_CONTAINER=true
fi

usage() {
    echo "Usage: $0 [--pci PCI_ADDR] [--help]"
    echo ""
    echo "Options:"
    echo "  --pci PCI_ADDR   Check a specific NVMe device (e.g. 0000:89:00.0)"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  UMBP_SPDK_NVME_PCI   Fallback PCI address if --pci is not given"
    echo ""
    echo "This preflight is intended to run on the host (bare metal)."
    echo "Running it inside a container can hide host mounts, hugepage mounts,"
    echo "and VFIO visibility, so container results are not authoritative."
    echo ""
    echo "If NVMe/VFIO host setup is incomplete, run ${SPDK_SETUP_REL} on the host"
    echo "and restart the container after it succeeds."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pci)
            TARGET_PCI="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Fallback to env var
if [[ -z "$TARGET_PCI" ]]; then
    TARGET_PCI="${UMBP_SPDK_NVME_PCI:-}"
fi

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
warn() { echo "  [WARN] $1"; ((WARN++)); }
info() { echo "  [INFO] $1"; }
need_host_setup() { NEEDS_HOST_SETUP=true; }

host_scope_tag() {
    if $IS_CONTAINER; then
        printf "HOST?"
    else
        printf "HOST"
    fi
}

scoped_status() {
    local level="$1"
    local scope="$2"
    local message="$3"

    echo "  [${level}][${scope}] ${message}"
}

scoped_pass() {
    local scope="$1"
    local message="$2"

    scoped_status "PASS" "$scope" "$message"
    ((PASS++))
}

scoped_fail() {
    local scope="$1"
    local message="$2"

    scoped_status "FAIL" "$scope" "$message"
    ((FAIL++))
    if [[ "$scope" == "RUNTIME" ]]; then
        ((RUNTIME_FAIL++))
    else
        ((HOST_FAIL++))
    fi
}

scoped_warn() {
    local scope="$1"
    local message="$2"

    scoped_status "WARN" "$scope" "$message"
    ((WARN++))
    if [[ "$scope" == "RUNTIME" ]]; then
        ((RUNTIME_WARN++))
    else
        ((HOST_WARN++))
    fi
}

scoped_info() {
    local scope="$1"
    local message="$2"

    scoped_status "INFO" "$scope" "$message"
}

scoped_note() {
    local scope="$1"
    local message="$2"

    scoped_status "----" "$scope" "$message"
}

host_pass() { scoped_pass "$(host_scope_tag)" "$1"; }
host_fail() { scoped_fail "$(host_scope_tag)" "$1"; }
host_warn() { scoped_warn "$(host_scope_tag)" "$1"; }
host_info() { scoped_info "$(host_scope_tag)" "$1"; }
host_note() { scoped_note "$(host_scope_tag)" "$1"; }

runtime_pass() { scoped_pass "RUNTIME" "$1"; }
runtime_fail() { scoped_fail "RUNTIME" "$1"; }
runtime_warn() { scoped_warn "RUNTIME" "$1"; }
runtime_info() { scoped_info "RUNTIME" "$1"; }
runtime_note() { scoped_note "RUNTIME" "$1"; }

readiness_word() {
    local fail_count="$1"
    local warn_count="$2"

    if (( fail_count > 0 )); then
        printf "FAIL"
    elif (( warn_count > 0 )); then
        printf "WARN"
    else
        printf "PASS"
    fi
}

setup_sh_command() {
    local pci="${1:-}"

    if [[ -n "$pci" ]]; then
        printf 'cd "%s" && sudo PCI_ALLOWED="%s" %s' "$MORI_ROOT" "$pci" "$SPDK_SETUP_REL"
    else
        printf 'cd "%s" && sudo %s' "$MORI_ROOT" "$SPDK_SETUP_REL"
    fi
}

preflight_command() {
    local pci="${1:-}"

    if [[ -n "$pci" ]]; then
        printf 'cd "%s" && ./tools/umbp_spdk_preflight.sh --pci "%s"' "$MORI_ROOT" "$pci"
    else
        printf 'cd "%s" && ./tools/umbp_spdk_preflight.sh' "$MORI_ROOT"
    fi
}

print_host_setup_hint() {
    local pci="${1:-}"
    local cmd=""

    cmd="$(setup_sh_command "$pci")"
    if $IS_CONTAINER; then
        info "Run on the host (outside the container): ${cmd}"
        info "After setup.sh succeeds on the host, restart the container."
    else
        info "Run: ${cmd}"
    fi
}

print_host_preflight_hint() {
    local pci="${1:-}"
    local cmd=""

    cmd="$(preflight_command "$pci")"
    info "Run on the host (outside the container): ${cmd}"
}

print_hugetlbfs_hint() {
    if $IS_CONTAINER; then
        info "In this container namespace, run: sudo mkdir -p /dev/hugepages && sudo mount -t hugetlbfs nodev /dev/hugepages"
        info "Host results remain authoritative; if needed, re-run this preflight on the host."
    else
        info "Fix: sudo mkdir -p /dev/hugepages && sudo mount -t hugetlbfs nodev /dev/hugepages"
    fi
}

append_unique() {
    local -n arr_ref=$1
    local value="$2"
    local existing=""

    [[ -n "$value" ]] || return 0
    for existing in "${arr_ref[@]}"; do
        [[ "$existing" == "$value" ]] && return 0
    done
    arr_ref+=("$value")
}

join_csv() {
    local IFS=", "
    printf '%s' "$*"
}

inspect_nvme_block_tree() {
    local blk_name="$1"
    local swap_devices="$2"
    local -n blk_names_ref=$3
    local -n mount_points_ref=$4
    local -n system_mounts_ref=$5
    local -n signatures_ref=$6
    local node=""
    local target=""
    local fstype=""

    while IFS= read -r node; do
        [[ -n "$node" ]] || continue

        append_unique blk_names_ref "${node##*/}"

        while IFS= read -r target; do
            [[ -n "$target" ]] || continue
            append_unique mount_points_ref "$target"
            case "$target" in
                /|/boot|/boot/efi)
                    append_unique system_mounts_ref "$target"
                    ;;
            esac
        done < <(findmnt -rn -S "$node" -o TARGET 2>/dev/null || true)

        if [[ -n "$swap_devices" ]] && grep -Fxq "$node" <<< "$swap_devices"; then
            append_unique mount_points_ref "swap"
        fi

        fstype="$(lsblk -dn -o FSTYPE "$node" 2>/dev/null | awk 'NF {print $1; exit}')"
        if [[ -n "$fstype" ]]; then
            append_unique signatures_ref "${node##*/}:$fstype"
        fi
    done < <(lsblk -nrpo NAME "/dev/$blk_name" 2>/dev/null || printf '/dev/%s\n' "$blk_name")
}

pci_nvme_root_blocks() {
    local pci_addr="$1"
    local path=""
    local resolved=""

    for path in /sys/block/nvme*; do
        [[ -e "$path" ]] || continue
        resolved="$(readlink -f "$path" 2>/dev/null || true)"
        [[ -n "$resolved" ]] || continue
        if [[ "$resolved" == *"/${pci_addr}/"* ]]; then
            basename "$path"
        fi
    done
}

echo "=========================================="
echo " UMBP SPDK Preflight Check"
echo "=========================================="
echo ""

echo "0. Execution context"
if $IS_CONTAINER; then
    warn "Running inside a container"
    info "Host-facing checks below are best-effort only; run this script on the host for authoritative results."
    print_host_preflight_hint "$TARGET_PCI"
else
    pass "Running on host (authoritative)"
fi

echo ""
echo "1. Host-facing checks"
hp_total=$(grep HugePages_Total /proc/meminfo 2>/dev/null | awk '{print $2}')
hp_free=$(grep HugePages_Free /proc/meminfo 2>/dev/null | awk '{print $2}')
hp_size=$(grep Hugepagesize /proc/meminfo 2>/dev/null | awk '{print $2}')

if [[ -n "$hp_total" && "$hp_total" -gt 0 ]]; then
    host_pass "HugePages configured: ${hp_total} x ${hp_size} kB (${hp_free} free)"
else
    host_fail "HugePages not configured"
    need_host_setup
fi

noiommu=$(cat /sys/module/vfio/parameters/enable_unsafe_noiommu_mode 2>/dev/null)
iommu_mode=""
if [[ -d /sys/kernel/iommu_groups/0 ]]; then
    iommu_mode=$(cat /sys/kernel/iommu_groups/0/type 2>/dev/null)
fi
if [[ "$iommu_mode" == "identity" || "$noiommu" == "Y" ]]; then
    host_info "IOMMU mode: ${iommu_mode:-noiommu} (noiommu_mode=${noiommu})"
fi

echo ""
echo "  NVMe inventory"

SWAP_DEVICES="$(swapon --noheadings --show=NAME 2>/dev/null || true)"
nvme_devs=()
spdk_ready=()
spdk_available=()
for dev in /sys/bus/pci/devices/*; do
    class=$(cat "$dev/class" 2>/dev/null)
    if [[ "$class" != "0x010802" ]]; then
        continue
    fi

    pci_addr=$(basename "$dev")
    driver=$(basename "$(readlink "$dev/driver" 2>/dev/null)" 2>/dev/null)
    desc=$(lspci -s "$pci_addr" 2>/dev/null | cut -d: -f3- | xargs)
    nvme_devs+=("$pci_addr")

    blk_devs=()
    mount_points=()
    system_mounts=()
    signatures=()
    while IFS= read -r blk_name; do
        [[ -n "$blk_name" ]] || continue
        inspect_nvme_block_tree "$blk_name" "$SWAP_DEVICES" blk_devs mount_points system_mounts signatures
    done < <(pci_nvme_root_blocks "$pci_addr")

    size_str=""
    while IFS= read -r blk_name; do
        [[ -n "$blk_name" ]] || continue
        sz_sectors=$(cat "/sys/block/${blk_name}/size" 2>/dev/null)
        if [[ -n "$sz_sectors" && "$sz_sectors" -gt 0 ]]; then
            sz_gb=$(( sz_sectors * 512 / 1073741824 ))
            size_str="${sz_gb} GB"
            break
        fi
    done < <(pci_nvme_root_blocks "$pci_addr")

    detail="${pci_addr}: ${desc}"
    [[ -n "$size_str" ]] && detail="${detail} (${size_str})"
    if [[ ${#blk_devs[@]} -gt 0 ]]; then
        detail="${detail} [$(join_csv "${blk_devs[@]}")]"
    fi
    if [[ ${#mount_points[@]} -gt 0 ]]; then
        detail="${detail} mounted: $(join_csv "${mount_points[@]}")"
    elif [[ ${#signatures[@]} -gt 0 ]]; then
        detail="${detail} signatures: $(join_csv "${signatures[@]}")"
    fi

    if [[ "$driver" == "vfio-pci" ]]; then
        spdk_ready+=("$pci_addr")
        host_pass "${detail} | Status: READY | Driver: ${driver:-none}"
    elif [[ ${#system_mounts[@]} -gt 0 ]]; then
        host_note "${detail} | Status: SYSTEM DISK | Driver: ${driver:-none}"
    elif [[ ${#mount_points[@]} -gt 0 ]]; then
        host_warn "${detail} | Status: IN USE | Driver: ${driver:-none}"
    elif [[ ${#signatures[@]} -gt 0 ]]; then
        host_warn "${detail} | Status: HAS DATA | Driver: ${driver:-none}"
    elif [[ "$driver" == "nvme" ]]; then
        spdk_available+=("$pci_addr")
        host_info "${detail} | Status: AVAILABLE | Driver: ${driver:-none}"
    elif [[ -z "$driver" ]]; then
        spdk_available+=("$pci_addr")
        host_info "${detail} | Status: NO DRIVER | Driver: none"
    else
        host_info "${detail} | Status: ${driver} | Driver: ${driver}"
    fi
done

if [[ ${#nvme_devs[@]} -eq 0 ]]; then
    host_fail "No NVMe devices found"
fi

echo ""
echo "   Legend: READY     = bound to vfio-pci, usable by SPDK now"
echo "           AVAILABLE = can be rebound for SPDK (not a system disk)"
echo "           IN USE    = mounted or active (unmount/swapoff before rebinding)"
echo "           HAS DATA  = unmounted, but filesystem signatures still exist"
echo "           SYSTEM DISK = backs /, /boot, or /boot/efi; do NOT use"

if [[ ${#spdk_ready[@]} -gt 0 ]]; then
    host_info "SPDK-ready devices: ${spdk_ready[*]}"
fi
if [[ ${#spdk_available[@]} -gt 0 ]]; then
    host_info "Can be prepared for SPDK on the host: ${spdk_available[*]}"
fi
if [[ ${#spdk_ready[@]} -eq 0 && ${#spdk_available[@]} -eq 0 ]]; then
    host_fail "No NVMe devices are currently ready or safely available for SPDK"
fi

if [[ -n "$TARGET_PCI" ]]; then
    echo ""
    echo "  Target device"
    dev_path="/sys/bus/pci/devices/${TARGET_PCI}"
    if [[ ! -d "$dev_path" ]]; then
        host_fail "${TARGET_PCI} not found in sysfs"
    else
        driver=$(basename "$(readlink "$dev_path/driver" 2>/dev/null)" 2>/dev/null)
        if [[ "$driver" == "vfio-pci" ]]; then
            host_pass "${TARGET_PCI} bound to vfio-pci"
        elif [[ "$driver" == "nvme" ]]; then
            host_fail "${TARGET_PCI} still bound to kernel 'nvme' driver"
            need_host_setup
        elif [[ -z "$driver" ]]; then
            host_fail "${TARGET_PCI} has no driver bound (expected vfio-pci)"
            need_host_setup
        else
            host_fail "${TARGET_PCI} bound to '${driver}' (expected vfio-pci)"
            need_host_setup
        fi

        iommu_group=$(basename "$(readlink "$dev_path/iommu_group" 2>/dev/null)" 2>/dev/null)
        if [[ -n "$iommu_group" ]]; then
            TARGET_IOMMU_GROUP="$iommu_group"
            host_info "Target IOMMU group: ${TARGET_IOMMU_GROUP}"
        else
            host_fail "No IOMMU group found for ${TARGET_PCI}"
            need_host_setup
        fi
    fi
fi

echo ""
echo "2. Runtime checks"

if mount | grep -q hugetlbfs; then
    hp_mount=$(mount | grep hugetlbfs | head -1 | awk '{print $3}')
    runtime_pass "hugetlbfs mounted at ${hp_mount}"
else
    runtime_fail "hugetlbfs not mounted in current namespace"
    print_hugetlbfs_hint
fi

if [[ -c /dev/vfio/vfio ]]; then
    runtime_pass "/dev/vfio/vfio visible in current runtime"
else
    runtime_fail "/dev/vfio/vfio not visible in current runtime"
    need_host_setup
fi

if [[ -n "$TARGET_PCI" && -n "$TARGET_IOMMU_GROUP" ]]; then
    if [[ -c "/dev/vfio/${TARGET_IOMMU_GROUP}" ]]; then
        runtime_pass "VFIO group device /dev/vfio/${TARGET_IOMMU_GROUP} visible"
    elif [[ -c "/dev/vfio/noiommu-${TARGET_IOMMU_GROUP}" ]]; then
        runtime_pass "VFIO no-IOMMU group device /dev/vfio/noiommu-${TARGET_IOMMU_GROUP} visible"
    else
        runtime_fail "VFIO group device for ${TARGET_PCI} not visible in current runtime"
        need_host_setup
    fi
fi

if [[ -n "${UMBP_SPDK_PROXY_BIN:-}" && -x "${UMBP_SPDK_PROXY_BIN}" ]]; then
    runtime_pass "UMBP_SPDK_PROXY_BIN points to ${UMBP_SPDK_PROXY_BIN}"
elif command -v spdk_proxy >/dev/null 2>&1; then
    runtime_pass "spdk_proxy found in PATH: $(command -v spdk_proxy)"
elif [[ -x "$MORI_ROOT/python/mori/spdk_proxy" ]]; then
    runtime_pass "Packaged spdk_proxy available: $MORI_ROOT/python/mori/spdk_proxy"
    runtime_info "Runtime auto-discovery will use the packaged proxy after BUILD_UMBP=1 pip install ."
elif [[ -x "$MORI_ROOT/build_umbp/src/umbp/spdk_proxy" ]]; then
    runtime_pass "Build-tree spdk_proxy available: $MORI_ROOT/build_umbp/src/umbp/spdk_proxy"
else
    runtime_info "spdk_proxy not found in common shell-visible locations"
    runtime_info "BUILD_UMBP=1 pip install . packages spdk_proxy and configures discovery."
fi

if ldconfig -p 2>/dev/null | grep -q libspdk_env_dpdk; then
    runtime_pass "SPDK libraries installed (libspdk_env_dpdk found)"
elif [[ -f /usr/local/lib/libspdk_env_dpdk.so ]]; then
    runtime_pass "SPDK libraries found in /usr/local/lib/"
else
    runtime_fail "SPDK shared libraries not found"
    runtime_info "Fix: cd \"$MORI_ROOT\" && ./tools/setup_spdk.sh"
fi

echo ""
echo "=========================================="
echo " Summary: ${PASS} passed, ${FAIL} failed, ${WARN} warnings"
echo "=========================================="
if $IS_CONTAINER; then
    echo " Host readiness: $(readiness_word "$HOST_FAIL" "$HOST_WARN") (best-effort only inside container)"
else
    echo " Host readiness: $(readiness_word "$HOST_FAIL" "$HOST_WARN")"
fi
echo " Runtime readiness: $(readiness_word "$RUNTIME_FAIL" "$RUNTIME_WARN")"

if [[ $FAIL -eq 0 ]]; then
    echo ""
    echo " SPDK backend is ready."
    if [[ -n "$TARGET_PCI" ]]; then
        echo " Run: UMBP_SPDK_NVME_PCI=${TARGET_PCI} ./bench_umbp_micro --ssd-backend spdk"
    fi
    exit 0
else
    echo ""
    echo " Fix failures above before using the SPDK backend."
    if $IS_CONTAINER; then
        echo ""
        print_host_preflight_hint "$TARGET_PCI"
    fi
    if $NEEDS_HOST_SETUP; then
        echo ""
        if [[ -n "$TARGET_PCI" ]]; then
            print_host_setup_hint "$TARGET_PCI"
        else
            print_host_setup_hint "<addr>"
        fi
    fi
    exit 1
fi
