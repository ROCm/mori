#!/usr/bin/env bash
# probe_topology.sh — discover RDMA/GPU rail topology on a node and emit a report
# + Mermaid (.mmd) + Graphviz (.dot). Vendor-generic: AMD (ionic/bnxt/mlx5) & NVIDIA (mlx5).
#
# Usage:
#   ./probe_topology.sh                       # local node only
#   ./probe_topology.sh --peer <PEER_IP_OR_HOST>   # also run cross-rail reachability
#   GID_INDEX=1 ./probe_topology.sh           # override RoCE GID index (default: auto)
#
# Outputs: ./topo_report.txt  ./topo.mmd  ./topo.dot
set -uo pipefail

OUT_DIR="${OUT_DIR:-.}"
REPORT="$OUT_DIR/topo_report.txt"
MMD="$OUT_DIR/topo.mmd"
DOT="$OUT_DIR/topo.dot"
PEER=""
[ "${1:-}" = "--peer" ] && PEER="${2:-}"
: > "$REPORT"

log() { echo "$@" | tee -a "$REPORT"; }

hostn=$(hostname)
log "=== node: $hostn ==="

# ---- 1. RDMA devices ------------------------------------------------------
declare -A NIC_NDEV NIC_IP NIC_GIDTYPE NIC_PCI NIC_NUMA NIC_STATE
IB_ROOT=/sys/class/infiniband
GID_INDEX="${GID_INDEX:-}"

nics=()
[ -d "$IB_ROOT" ] && nics=($(ls "$IB_ROOT" 2>/dev/null | sort -V))
log ""; log "--- RDMA devices (${#nics[@]}) ---"
for d in "${nics[@]}"; do
  P="$IB_ROOT/$d/ports/1"
  state=$(cat "$P/state" 2>/dev/null | awk '{print $2}')
  # pick GID index: explicit ($GID_INDEX), else first RoCEv2 entry with a GLOBAL
  # (routable) address. Rails may be addressed as IPv4-mapped (::ffff:AABBCCDD) OR
  # global IPv6 — commonly a ULA (fc00::/7), each rail its own /64. Skip fe80::
  # link-local and the all-zero entry. IPv4-mapped is preferred when present.
  gi="$GID_INDEX"
  if [ -z "$gi" ]; then
    for i in $(seq 0 7); do
      g=$(cat "$P/gids/$i" 2>/dev/null)
      t=$(cat "$P/gid_attrs/types/$i" 2>/dev/null)
      [ "$t" = "RoCE v2" ] || continue
      case "$g" in
        0000:0000:0000:0000:0000:0000:0000:0000) : ;;                 # empty
        fe80:*) : ;;                                                   # link-local, skip
        0000:0000:0000:0000:0000:ffff:*) gi=$i; break ;;              # IPv4-mapped (best)
        *) [ -z "${gi:-}" ] && gi=$i ;;                                # first global IPv6 (ULA/GUA)
      esac
    done
    gi="${gi:-1}"
  fi
  gid=$(cat "$P/gids/$gi" 2>/dev/null)
  gtype=$(cat "$P/gid_attrs/types/$gi" 2>/dev/null)
  ndev=$(cat "$P/gid_attrs/ndevs/$gi" 2>/dev/null)
  # derive the rail address from the chosen GID
  ip=""
  case "$gid" in
    0000:0000:0000:0000:0000:ffff:*)   # IPv4-mapped -> dotted quad
      tail4=$(echo "$gid" | awk -F: '{print $7$8}')
      ip=$(printf "%d.%d.%d.%d" 0x${tail4:0:2} 0x${tail4:2:2} 0x${tail4:4:2} 0x${tail4:6:2} 2>/dev/null) ;;
    ""|0000:0000:0000:0000:0000:0000:0000:0000) : ;;
    *)                                 # global IPv6 (ULA/GUA) -> read netdev's global v6
      [ -n "$ndev" ] && ip=$(ip -o -6 addr show "$ndev" scope global 2>/dev/null | awk '{print $4}' | head -1) ;;
  esac
  pci=$(basename "$(readlink -f "$IB_ROOT/$d/device" 2>/dev/null)" 2>/dev/null)
  numa=$(cat "$IB_ROOT/$d/device/numa_node" 2>/dev/null)
  NIC_NDEV[$d]=$ndev; NIC_IP[$d]=$ip; NIC_GIDTYPE[$d]=$gtype
  NIC_PCI[$d]=$pci; NIC_NUMA[$d]=$numa; NIC_STATE[$d]=$state
  log "$d  state=$state  ndev=$ndev  gid[$gi]=$gtype  ip=$ip  pci=$pci  numa=$numa"
done

# ---- 2. GPUs --------------------------------------------------------------
declare -A GPU_PCI
log ""; log "--- GPUs ---"
GPU_TMP=$(mktemp)
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showbus 2>/dev/null | sed -nE 's/^GPU\[([0-9]+)\].*PCI Bus: ([0-9A-Fa-f:.]+)/\1 \2/p' > "$GPU_TMP"
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | tr -d ' ' | awk -F, '{print $1" "$2}' > "$GPU_TMP"
fi
while read -r idx bus; do [ -n "$idx" ] && GPU_PCI[$idx]=$bus; done < "$GPU_TMP"
rm -f "$GPU_TMP"
for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do log "GPU$k  pci=${GPU_PCI[$k]}"; done

# ---- 3. GPU <-> NIC affinity (rail-local: same PCI domain, ordinal) --------
# PCI addresses are domain:bus:dev.func (e.g. 0002:00:01.0). Rail-optimized
# boxes place each GPU on the same PCIe domain as its rail NIC; pair the k-th
# GPU with the k-th NIC within that domain (sorted by full PCI address).
#
# First isolate the RAIL NICs: exclude the management/front-end NIC(s) — the one
# on the default route, and any with no global address — otherwise an interspersed
# mgmt NIC shifts the ordinal pairing (single-domain boxes list mgmt + rail NICs
# together, e.g. mlx5 eth0/eth1 among rdma0..7).
MGMT_NDEV=$(ip route show default 2>/dev/null | awk '/default/{print $5; exit}')
railnics=()
for d in "${nics[@]}"; do
  [ -n "$MGMT_NDEV" ] && [ "${NIC_NDEV[$d]}" = "$MGMT_NDEV" ] && continue  # default route = mgmt
  [ -z "${NIC_IP[$d]}" ] && continue                                       # no global addr = not a rail
  railnics+=("$d")
done
[ ${#railnics[@]} -gt 0 ] || railnics=("${nics[@]}")   # fallback: use all if filter emptied

log ""; log "--- GPU <-> NIC PCIe affinity (same-domain ordinal; rail NICs only) ---"
log "rail NICs: ${railnics[*]}   (mgmt/default-route NIC excluded: ${MGMT_NDEV:-none})"
declare -A GPU_NIC
domains=$(for k in "${!GPU_PCI[@]}"; do echo "${GPU_PCI[$k]%%:*}"; done | sort -u)
for dom in $domains; do
  gpus_d=$(for k in "${!GPU_PCI[@]}"; do echo "${GPU_PCI[$k]} $k"; done | grep -i "^${dom}:" | sort | awk '{print $2}')
  nics_d=$(for d in "${railnics[@]}"; do echo "${NIC_PCI[$d]} $d"; done | grep -i "^${dom}:" | sort | awk '{print $2}')
  set -- $nics_d
  for k in $gpus_d; do
    GPU_NIC[$k]="${1:-}"; [ -n "${1:-}" ] && shift
  done
done
for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do
  d=${GPU_NIC[$k]}
  log "GPU$k (${GPU_PCI[$k]}) -> ${d:-?} (${NIC_PCI[$d]:-none}) rail-local"
done

# ---- 4. Rail reachability (optional, needs --peer) ------------------------
if [ -n "$PEER" ]; then
  log ""; log "--- Rail reachability to peer $PEER ---"
  log "(rail-aligned should pass; cross-rail failing => rail-only fabric)"
  for d in "${nics[@]}"; do
    nd=${NIC_NDEV[$d]}; ip=${NIC_IP[$d]}
    [ -z "$nd" ] && continue
    # derive peer same-rail IP by swapping the last octet is site-specific;
    # here we just ping the peer's per-rail IP if provided via PEER_IPS map.
    :
  done
  log "NOTE: supply peer per-rail IPs to fully script cross-rail tests, e.g.:"
  log "  ping -c2 -I <local_rail_dev> <peer_same_rail_ip>   # expect OK"
  log "  ping -c2 -I <local_rail_dev> <peer_other_rail_ip>  # expect FAIL if rail-only"
fi

# ---- 5. Emit Mermaid ------------------------------------------------------
{
  echo "graph LR"
  echo "  subgraph NODE[$hostn]"
  for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do
    d=${GPU_NIC[$k]}
    echo "    G$k[\"GPU$k<br/>${GPU_PCI[$k]}\"] --- N_$d[\"$d / ${NIC_NDEV[$d]}<br/>${NIC_IP[$d]}\"]"
  done
  echo "  end"
} > "$MMD"

# ---- 6. Emit Graphviz -----------------------------------------------------
{
  echo "digraph topo {"
  echo "  rankdir=LR; node [shape=box, style=\"rounded,filled\", fillcolor=\"#eef2f7\"];"
  echo "  label=\"$hostn RDMA/GPU topology\";"
  for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do
    d=${GPU_NIC[$k]}
    echo "  \"GPU$k\" -> \"$d\\n${NIC_IP[$d]}\" [dir=none, style=dotted, label=\"PCIe\"];"
  done
  echo "}"
} > "$DOT"

log ""
log "Wrote: $REPORT  $MMD  $DOT"
log "Render DOT:     dot -Tpng $DOT -o topo.png"
log "Render Mermaid: paste $MMD into https://mermaid.live"
