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
#   6. spdk_proxy binary available
#   7. SPDK shared libraries installed

set -o pipefail

PASS=0
FAIL=0
WARN=0
TARGET_PCI=""

usage() {
    echo "Usage: $0 [--pci PCI_ADDR] [--help]"
    echo ""
    echo "Options:"
    echo "  --pci PCI_ADDR   Check a specific NVMe device (e.g. 0000:89:00.0)"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  UMBP_SPDK_NVME_PCI   Fallback PCI address if --pci is not given"
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

echo "=========================================="
echo " UMBP SPDK Preflight Check"
echo "=========================================="
echo ""

# --- 1. Hugepages ---
echo "1. Hugepages"
hp_total=$(grep HugePages_Total /proc/meminfo 2>/dev/null | awk '{print $2}')
hp_free=$(grep HugePages_Free /proc/meminfo 2>/dev/null | awk '{print $2}')
hp_size=$(grep Hugepagesize /proc/meminfo 2>/dev/null | awk '{print $2}')

if [[ -n "$hp_total" && "$hp_total" -gt 0 ]]; then
    pass "Hugepages allocated: ${hp_total} x ${hp_size} kB (${hp_free} free)"
else
    fail "No hugepages allocated"
    info "Fix: echo 2048 > /proc/sys/vm/nr_hugepages"
fi

# --- 2. Hugetlbfs ---
echo ""
echo "2. Hugetlbfs mount"
if mount | grep -q hugetlbfs; then
    hp_mount=$(mount | grep hugetlbfs | head -1 | awk '{print $3}')
    pass "Hugetlbfs mounted at ${hp_mount}"
else
    fail "Hugetlbfs not mounted"
    info "Fix: mkdir -p /dev/hugepages && mount -t hugetlbfs nodev /dev/hugepages"
fi

# --- 3. VFIO support ---
echo ""
echo "3. VFIO kernel support"
if [[ -c /dev/vfio/vfio ]]; then
    pass "/dev/vfio/vfio exists"
else
    fail "/dev/vfio/vfio not found"
    info "Fix: modprobe vfio-pci (or check kernel config)"
fi

# Check noiommu mode
noiommu=$(cat /sys/module/vfio/parameters/enable_unsafe_noiommu_mode 2>/dev/null)
iommu_mode=""
if [[ -d /sys/kernel/iommu_groups/0 ]]; then
    iommu_mode=$(cat /sys/kernel/iommu_groups/0/type 2>/dev/null)
fi
if [[ "$iommu_mode" == "identity" || "$noiommu" == "Y" ]]; then
    info "IOMMU mode: ${iommu_mode:-noiommu} (noiommu_mode=${noiommu})"
fi

# --- 4. NVMe device discovery ---
echo ""
echo "4. NVMe device discovery"

# Gather mount info once
mount_info=$(mount 2>/dev/null)

# Collect all NVMe devices with status
nvme_devs=()
spdk_ready=()
spdk_available=()
for dev in /sys/bus/pci/devices/*; do
    class=$(cat "$dev/class" 2>/dev/null)
    # NVMe controller class: 0x010802
    if [[ "$class" == "0x010802" ]]; then
        pci_addr=$(basename "$dev")
        driver=$(basename "$(readlink "$dev/driver" 2>/dev/null)" 2>/dev/null)
        desc=$(lspci -s "$pci_addr" 2>/dev/null | cut -d: -f3- | xargs)
        nvme_devs+=("$pci_addr")

        # Detect block devices and mount points
        blk_devs=""
        mount_points=""
        is_system=false
        for blk in "$dev"/block/*/; do
            [[ -d "$blk" ]] || continue
            blk_name=$(basename "$blk")
            blk_devs="${blk_devs:+$blk_devs, }${blk_name}"
            while IFS= read -r mnt; do
                mp=$(echo "$mnt" | awk '{print $3}')
                mount_points="${mount_points:+$mount_points, }${mp}"
                is_system=true
            done < <(echo "$mount_info" | grep "/dev/${blk_name}")
            # Check partitions
            for part in "$blk"/"${blk_name}"p*/; do
                [[ -d "$part" ]] || continue
                part_name=$(basename "$part")
                blk_devs="${blk_devs:+$blk_devs, }${part_name}"
                while IFS= read -r mnt; do
                    mp=$(echo "$mnt" | awk '{print $3}')
                    mount_points="${mount_points:+$mount_points, }${mp}"
                    is_system=true
                done < <(echo "$mount_info" | grep "/dev/${part_name}")
            done
        done

        # Determine device size (from block device or sysfs)
        size_str=""
        for blk in "$dev"/block/*/; do
            [[ -d "$blk" ]] || continue
            blk_name=$(basename "$blk")
            sz_sectors=$(cat "/sys/block/${blk_name}/size" 2>/dev/null)
            if [[ -n "$sz_sectors" && "$sz_sectors" -gt 0 ]]; then
                sz_gb=$(( sz_sectors * 512 / 1073741824 ))
                size_str="${sz_gb} GB"
            fi
            break
        done

        # Categorize
        if [[ "$driver" == "vfio-pci" ]]; then
            status="READY"
            tag="[PASS]"
            spdk_ready+=("$pci_addr")
        elif $is_system; then
            status="SYSTEM DISK"
            tag="[----]"
        elif [[ "$driver" == "nvme" ]]; then
            status="AVAILABLE"
            tag="[INFO]"
            spdk_available+=("$pci_addr")
        elif [[ -z "$driver" ]]; then
            status="NO DRIVER"
            tag="[INFO]"
            spdk_available+=("$pci_addr")
        else
            status="$driver"
            tag="[INFO]"
        fi

        # Format output line
        detail="${pci_addr}: ${desc}"
        [[ -n "$size_str" ]] && detail="${detail} (${size_str})"
        [[ -n "$blk_devs" ]] && detail="${detail} [${blk_devs}]"
        [[ -n "$mount_points" ]] && detail="${detail} mounted: ${mount_points}"
        echo "  ${tag} ${detail}"
        echo "         Status: ${status} | Driver: ${driver:-none}"
    fi
done

if [[ ${#nvme_devs[@]} -eq 0 ]]; then
    fail "No NVMe devices found"
fi

# Print summary table
echo ""
echo "   Legend: READY     = bound to vfio-pci, usable by SPDK now"
echo "           AVAILABLE = can be rebound for SPDK (not a system disk)"
echo "           SYSTEM DISK = has mounted filesystems, do NOT use"

if [[ ${#spdk_ready[@]} -gt 0 ]]; then
    echo ""
    echo "   SPDK-ready devices: ${spdk_ready[*]}"
fi
if [[ ${#spdk_available[@]} -gt 0 ]]; then
    echo "   Can be bound for SPDK: ${spdk_available[*]}"
    info "To bind: PCI_ALLOWED=\"<addr>\" sudo ./3rdparty/spdk/scripts/setup.sh"
fi
if [[ ${#spdk_ready[@]} -eq 0 && ${#spdk_available[@]} -eq 0 ]]; then
    fail "No NVMe devices available for SPDK (all are system disks)"
fi

# Check target device if specified
if [[ -n "$TARGET_PCI" ]]; then
    echo ""
    echo "   Target device: ${TARGET_PCI}"
    dev_path="/sys/bus/pci/devices/${TARGET_PCI}"
    if [[ ! -d "$dev_path" ]]; then
        fail "Device ${TARGET_PCI} not found in sysfs"
    else
        driver=$(basename "$(readlink "$dev_path/driver" 2>/dev/null)" 2>/dev/null)
        if [[ "$driver" == "vfio-pci" ]]; then
            pass "${TARGET_PCI} is bound to vfio-pci"
        elif [[ "$driver" == "nvme" ]]; then
            fail "${TARGET_PCI} is bound to kernel 'nvme' driver"
            info "Fix: Use SPDK setup.sh to rebind:"
            info "  PCI_ALLOWED=\"${TARGET_PCI}\" sudo ./3rdparty/spdk/scripts/setup.sh"
        elif [[ -z "$driver" ]]; then
            warn "${TARGET_PCI} has no driver bound"
            info "Fix: echo 'vfio-pci' > /sys/bus/pci/devices/${TARGET_PCI}/driver_override"
            info "     echo '${TARGET_PCI}' > /sys/bus/pci/drivers_probe"
        else
            fail "${TARGET_PCI} is bound to '${driver}' (expected vfio-pci)"
        fi

        # Check VFIO group
        iommu_group=$(basename "$(readlink "$dev_path/iommu_group" 2>/dev/null)" 2>/dev/null)
        if [[ -n "$iommu_group" ]]; then
            if [[ -c "/dev/vfio/${iommu_group}" ]]; then
                pass "VFIO group device /dev/vfio/${iommu_group} accessible"
            elif [[ -c "/dev/vfio/noiommu-${iommu_group}" ]]; then
                pass "VFIO no-IOMMU group device /dev/vfio/noiommu-${iommu_group} accessible"
            else
                fail "VFIO group ${iommu_group} exists but /dev/vfio/${iommu_group} not found"
                info "If in Docker, the host must pass --device /dev/vfio/${iommu_group}"
            fi
        else
            fail "No IOMMU group for ${TARGET_PCI}"
        fi
    fi
fi

# --- 5. SPDK proxy binary ---
echo ""
echo "5. spdk_proxy binary"
proxy_bin=$(which spdk_proxy 2>/dev/null)
if [[ -n "$proxy_bin" ]]; then
    pass "spdk_proxy found: ${proxy_bin}"
else
    # Check common locations
    found=""
    for dir in /usr/local/bin /usr/bin ./build_umbp/src/umbp; do
        if [[ -x "${dir}/spdk_proxy" ]]; then
            found="${dir}/spdk_proxy"
            break
        fi
    done
    # Check next to bench_umbp_micro
    for f in $(find ./build_umbp -name spdk_proxy -type f 2>/dev/null | head -1); do
        found="$f"
    done
    if [[ -n "$found" ]]; then
        warn "spdk_proxy not in PATH, but found at: ${found}"
        info "Fix: export PATH=\"$(dirname "$found"):\$PATH\""
    else
        fail "spdk_proxy binary not found"
        info "Fix: Build with SPDK support: cmake --build build_umbp -j\$(nproc)"
    fi
fi

# --- 6. SPDK shared libraries ---
echo ""
echo "6. SPDK shared libraries"
if ldconfig -p 2>/dev/null | grep -q libspdk_env_dpdk; then
    pass "SPDK libraries installed (libspdk_env_dpdk found)"
elif [[ -f /usr/local/lib/libspdk_env_dpdk.so ]]; then
    pass "SPDK libraries found in /usr/local/lib/"
else
    fail "SPDK shared libraries not found"
    info "Fix: Build and install SPDK from 3rdparty/spdk/"
fi

# --- 7. Docker detection ---
echo ""
echo "7. Container environment"
if [[ -f /.dockerenv ]] || grep -q docker /proc/1/cgroup 2>/dev/null || [[ "$(cat /proc/1/sched 2>/dev/null | head -1)" != "systemd"* ]]; then
    info "Running inside a container"
    cap_eff=$(grep CapEff /proc/self/status 2>/dev/null | awk '{print $2}')
    if [[ "$cap_eff" == "000001ffffffffff" || "$cap_eff" == "0000003fffffffff" ]]; then
        pass "Container has full capabilities (privileged)"
    else
        warn "Container may not have sufficient capabilities"
        info "Fix: Run with --privileged or add --cap-add SYS_RAWIO --cap-add SYS_ADMIN"
    fi

    # Check if VFIO group devices are visible
    vfio_groups=$(ls /dev/vfio/ 2>/dev/null | grep -E '^[0-9]+$|^noiommu-')
    if [[ -n "$vfio_groups" ]]; then
        pass "VFIO group devices visible: ${vfio_groups}"
    else
        fail "No VFIO group devices in /dev/vfio/"
        info "Fix on host: Bind device to vfio-pci, then start container with:"
        if [[ -n "$TARGET_PCI" ]]; then
            iommu_group=$(basename "$(readlink "/sys/bus/pci/devices/${TARGET_PCI}/iommu_group" 2>/dev/null)" 2>/dev/null)
            if [[ -n "$iommu_group" ]]; then
                info "  docker run --device /dev/vfio/${iommu_group} --device /dev/vfio/vfio ..."
            fi
        fi
        info "  docker run --device /dev/vfio/<group> --device /dev/vfio/vfio ..."
    fi
else
    info "Running on bare metal (not in a container)"
fi

# --- Summary ---
echo ""
echo "=========================================="
echo " Summary: ${PASS} passed, ${FAIL} failed, ${WARN} warnings"
echo "=========================================="
if [[ $FAIL -eq 0 ]]; then
    echo " SPDK backend is ready."
    if [[ -n "$TARGET_PCI" ]]; then
        echo " Run: UMBP_SPDK_NVME_PCI=${TARGET_PCI} ./bench_umbp_micro --ssd-backend spdk"
    fi
    exit 0
else
    echo " Fix the failures above before using SPDK backend."
    exit 1
fi
