#!/usr/bin/env bash
# rail_detect.sh — shared RoCE rail discovery for the cluster-network-topology
# skill. Source it and call rail_detect, or run it with --dump to see what it
# found on this node.
#
# Auto-detects each device's global GID index (IPv4-mapped OR IPv6 ULA/GUA,
# skipping fe80:: link-local) and its address, so no device or netdev name is
# ever hardcoded and the same code works on ionic/bnxt/mlx5.
#
# Populates:
#   RAIL_ALL      every RDMA device, sort -V order, after the regex filters
#   RAIL_DEVS     rail NICs: not the mgmt NIC, and has a usable global GID
#   RAIL_ACTIVE   RAIL_DEVS whose link is ACTIVE — the ones traffic can use
#   RAIL_MGMT_NDEV  netdev on the default route, or empty
#   RAIL_NDEV RAIL_ADDR RAIL_FAM RAIL_GIDIDX RAIL_GIDTYPE RAIL_PCI RAIL_NUMA
#   RAIL_STATE      all keyed by device name
#
# RAIL_DEVS deliberately does not filter on link state: probe_topology.sh pairs
# GPUs to NICs by PCI ordinal, so dropping a present-but-down NIC would shift
# that ordinal and mis-pair every GPU after it. Callers that put traffic on the
# wire use RAIL_ACTIVE.
#
# Optional env overrides:
#   RAIL_DEV_REGEX     only consider RDMA devices matching this ERE   (default: .*)
#   EXCLUDE_DEV_REGEX  drop RDMA devices matching this ERE            (default: none)
#   INCLUDE_MGMT=1     keep the device on the default-route netdev    (default: drop it)
#   GID_INDEX          force this GID index for every device          (default: auto)
#   RAIL_IB_ROOT       sysfs root, for testing against a fake tree    (default: /sys/class/infiniband)

RAIL_IB_ROOT="${RAIL_IB_ROOT:-/sys/class/infiniband}"

RAIL_ALL=(); RAIL_DEVS=(); RAIL_ACTIVE=(); RAIL_MGMT_NDEV=""
declare -A RAIL_NDEV RAIL_ADDR RAIL_FAM RAIL_GIDIDX RAIL_GIDTYPE RAIL_PCI RAIL_NUMA RAIL_STATE

# $1 = .../ports/1 path; echoes a global RoCEv2 GID index, or empty if the port
# has none. Empty is meaningful — guessing an index produces a device that looks
# detected and then fails to carry traffic.
rail_pick_gid() {
  local P=$1 i g t gi=""
  if [ -n "${GID_INDEX:-}" ]; then echo "$GID_INDEX"; return; fi
  for i in $(seq 0 15); do
    g=$(cat "$P/gids/$i" 2>/dev/null); t=$(cat "$P/gid_attrs/types/$i" 2>/dev/null)
    [ "$t" = "RoCE v2" ] || continue
    case "$g" in
      0000:0000:0000:0000:0000:0000:0000:0000) : ;;              # empty
      fe80:*) : ;;                                               # link-local, skip
      0000:0000:0000:0000:0000:ffff:*) gi=$i; break ;;           # IPv4-mapped (best)
      *) [ -z "$gi" ] && gi=$i ;;                                # first global IPv6
    esac
  done
  echo "$gi"
}

rail_detect() {
  RAIL_ALL=(); RAIL_DEVS=(); RAIL_ACTIVE=()
  local d P gi g nd t4 a fam

  [ -d "$RAIL_IB_ROOT" ] || return 0
  RAIL_MGMT_NDEV=$(ip route show default 2>/dev/null | awk '/default/{print $5; exit}')

  # No process substitution: it needs /dev/fd, which is absent in some scheduler
  # step namespaces (e.g. Spur srun). Device names have no spaces.
  for d in $(ls "$RAIL_IB_ROOT" 2>/dev/null | sort -V); do
    echo "$d" | grep -qE "${RAIL_DEV_REGEX:-.*}" || continue
    [ -n "${EXCLUDE_DEV_REGEX:-}" ] && echo "$d" | grep -qE "$EXCLUDE_DEV_REGEX" && continue
    RAIL_ALL+=("$d")
  done
  [ ${#RAIL_ALL[@]} -gt 0 ] || return 0

  for d in "${RAIL_ALL[@]}"; do
    P="$RAIL_IB_ROOT/$d/ports/1"
    gi=$(rail_pick_gid "$P")
    g=$(cat "$P/gids/$gi" 2>/dev/null)
    nd=$(cat "$P/gid_attrs/ndevs/$gi" 2>/dev/null)
    a=""; fam=""
    case "$g" in
      0000:0000:0000:0000:0000:ffff:*)
        t4=$(echo "$g" | awk -F: '{print $7$8}')
        a=$(printf "%d.%d.%d.%d" "0x${t4:0:2}" "0x${t4:2:2}" "0x${t4:4:2}" "0x${t4:6:2}" 2>/dev/null)
        fam=4 ;;
      ""|0000:0000:0000:0000:0000:0000:0000:0000) : ;;
      *)
        [ -n "$nd" ] && a=$(ip -o -6 addr show "$nd" scope global 2>/dev/null \
                              | awk '{print $4}' | cut -d/ -f1 | head -1)
        fam=6 ;;
    esac

    RAIL_GIDIDX[$d]=$gi
    RAIL_GIDTYPE[$d]=$(cat "$P/gid_attrs/types/$gi" 2>/dev/null)
    RAIL_NDEV[$d]=$nd
    RAIL_ADDR[$d]=$a
    RAIL_FAM[$d]=$fam
    RAIL_STATE[$d]=$(awk '{print $2}' "$P/state" 2>/dev/null)
    RAIL_PCI[$d]=$(basename "$(readlink -f "$RAIL_IB_ROOT/$d/device" 2>/dev/null)" 2>/dev/null)
    RAIL_NUMA[$d]=$(cat "$RAIL_IB_ROOT/$d/device/numa_node" 2>/dev/null)

    [ -n "$gi" ] || continue
    if [ -z "${INCLUDE_MGMT:-}" ] && [ -n "$RAIL_MGMT_NDEV" ] && [ "$nd" = "$RAIL_MGMT_NDEV" ]; then
      continue
    fi
    [ -n "$a" ] || continue
    RAIL_DEVS+=("$d")
  done

  # No default route, or no address we could derive: report every device rather
  # than claiming a machine with RDMA hardware has no rails at all.
  [ ${#RAIL_DEVS[@]} -gt 0 ] || RAIL_DEVS=("${RAIL_ALL[@]}")

  for d in "${RAIL_DEVS[@]}"; do
    [ "${RAIL_STATE[$d]}" = ACTIVE ] || continue
    RAIL_ACTIVE+=("$d")
  done
}

rail_dump() {
  local d
  rail_detect
  printf 'ib_root=%s  gid_index=%s  mgmt_ndev=%s\n' \
    "$RAIL_IB_ROOT" "${GID_INDEX:-auto}" "${RAIL_MGMT_NDEV:-none}"
  printf 'devices=%d  rails=%d  active=%d\n\n' \
    "${#RAIL_ALL[@]}" "${#RAIL_DEVS[@]}" "${#RAIL_ACTIVE[@]}"
  for d in ${RAIL_ALL[@]+"${RAIL_ALL[@]}"}; do
    printf '%-10s state=%-8s gid[%-2s]=%-8s ndev=%-12s v%s addr=%-24s pci=%-13s numa=%s\n' \
      "$d" "${RAIL_STATE[$d]:-?}" "${RAIL_GIDIDX[$d]:-?}" "${RAIL_GIDTYPE[$d]:-none}" \
      "${RAIL_NDEV[$d]:-?}" "${RAIL_FAM[$d]:-?}" "${RAIL_ADDR[$d]:-none}" \
      "${RAIL_PCI[$d]:-?}" "${RAIL_NUMA[$d]:-?}"
  done
  printf '\nrails:  %s\nactive: %s\n' "${RAIL_DEVS[*]:-none}" "${RAIL_ACTIVE[*]:-none}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:---dump}" in
    --dump) rail_dump ;;
    *) echo "usage: $0 [--dump]   (otherwise: source it and call rail_detect)" >&2; exit 2 ;;
  esac
fi
