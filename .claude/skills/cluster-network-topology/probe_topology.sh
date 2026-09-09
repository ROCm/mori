#!/usr/bin/env bash
# probe_topology.sh — discover RDMA/GPU rail topology on a node and emit a report
# + Mermaid (.mmd) + Graphviz (.dot). Vendor-generic: AMD (ionic/bnxt/mlx5) & NVIDIA (mlx5).
#
# Usage:
#   ./probe_topology.sh                       # local node only
#   PEER_IPS='<rail0_ip> <rail1_ip> ...' ./probe_topology.sh --peer <PEER>  # IP reachability matrix
#   GID_INDEX=1 ./probe_topology.sh           # override RoCE GID index (default: auto)
#
# Outputs (in $OUT_DIR, default .):
#   topo_report.<host>.txt             machine-readable probe report
#   topology_<host>_node.auto.{dot,mmd}  raw GPU<->NIC graph
#
# Then, on a machine with graphviz:  python3 make_report.py <OUT_DIR>
set -uo pipefail

OUT_DIR="${OUT_DIR:-.}"
PEER=""
[ "${1:-}" = "--peer" ] && PEER="${2:-}"

hostn=$(hostname)
# Host-suffixed: a 2-node run drops both nodes' probes into one folder, and on
# schedulers with no srun the batch body runs on every node at once. Still matches
# the topo_report*.txt glob make_report.py looks for.
REPORT="$OUT_DIR/topo_report.$hostn.txt"
# ".auto" keeps these raw emissions distinct from make_diagrams.py's richer
# topology_<host>_node.dot, which is what make_report.py embeds.
MMD="$OUT_DIR/topology_${hostn}_node.auto.mmd"
DOT="$OUT_DIR/topology_${hostn}_node.auto.dot"
# Build in a private temp and rename at the end. A scheduler can dispatch this body
# more than once on the same node, and two line-buffered `tee -a` writers interleave
# into a single file that parses cleanly as a node with twice as many GPUs — a wrong
# answer that looks like a right one. rename(2) is atomic, so a duplicate run replaces
# the file wholesale instead of corrupting it.
REPORT_TMP="$REPORT.$$.tmp"
: > "$REPORT_TMP"
trap 'mv -f "$REPORT_TMP" "$REPORT" 2>/dev/null' EXIT

log() { echo "$@" | tee -a "$REPORT_TMP"; }

log "=== node: $hostn ==="

# ---- 1. RDMA devices ------------------------------------------------------
# Detection is shared with xrail_worker.sh / xrail_matrix.sh so the tools never
# disagree about the same machine.
RAIL_LIB="${RAIL_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rail_detect.sh}"
[ -r "$RAIL_LIB" ] || { echo "cannot read $RAIL_LIB (set RAIL_LIB)" >&2; exit 1; }
# shellcheck source=rail_detect.sh
. "$RAIL_LIB"
rail_detect

nics=(${RAIL_ALL[@]+"${RAIL_ALL[@]}"})
log ""; log "--- RDMA devices (${#nics[@]}) ---"
for d in ${nics[@]+"${nics[@]}"}; do
  log "$d  state=${RAIL_STATE[$d]}  ndev=${RAIL_NDEV[$d]}  gid[${RAIL_GIDIDX[$d]:-none}]=${RAIL_GIDTYPE[$d]}  ip=${RAIL_ADDR[$d]}  pci=${RAIL_PCI[$d]}  numa=${RAIL_NUMA[$d]}"
done

# ---- 2. GPUs --------------------------------------------------------------
declare -A GPU_PCI
log ""; log "--- GPUs ---"
GPU_TMP=$(mktemp)
GPU_MODEL=""
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showbus 2>/dev/null | sed -nE 's/^GPU\[([0-9]+)\].*PCI Bus: ([0-9A-Fa-f:.]+)/\1 \2/p' > "$GPU_TMP"
  GPU_MODEL=$(rocm-smi --showproductname 2>/dev/null | sed -nE 's/.*Card Series:[[:space:]]+(.+[^[:space:]])[[:space:]]*$/\1/p' | head -1)
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | tr -d ' ' | awk -F, '{print $1" "$2}' > "$GPU_TMP"
  GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
[ -n "$GPU_MODEL" ] && log "GPU model: $GPU_MODEL"
while read -r idx bus; do [ -n "$idx" ] && GPU_PCI[$idx]=$bus; done < "$GPU_TMP"
rm -f "$GPU_TMP"
for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do log "GPU$k  pci=${GPU_PCI[$k]}"; done

# ---- 3. GPU <-> NIC affinity (rail-local: same PCI domain, ordinal) --------
# PCI addresses are domain:bus:dev.func (e.g. 0002:00:01.0). Rail-optimized
# boxes place each GPU on the same PCIe domain as its rail NIC; pair the k-th
# GPU with the k-th NIC within that domain (sorted by full PCI address).
#
# RAIL_DEVS has already excluded the management/front-end NIC(s) — the one on the
# default route, and any with no global address — otherwise an interspersed mgmt
# NIC shifts the ordinal pairing (single-domain boxes list mgmt + rail NICs
# together, e.g. mlx5 eth0/eth1 among rdma0..7). It deliberately keeps NICs whose
# link is down, since dropping one would shift the ordinal just as badly.
railnics=(${RAIL_DEVS[@]+"${RAIL_DEVS[@]}"})

log ""; log "--- GPU <-> NIC PCIe affinity (same-domain ordinal; rail NICs only) ---"
log "rail NICs: ${railnics[*]:-none}   (mgmt/default-route NIC excluded: ${RAIL_MGMT_NDEV:-none})"
declare -A GPU_NIC
domains=$(for k in "${!GPU_PCI[@]}"; do echo "${GPU_PCI[$k]%%:*}"; done | sort -u)
for dom in $domains; do
  gpus_d=$(for k in "${!GPU_PCI[@]}"; do echo "${GPU_PCI[$k]} $k"; done | grep -i "^${dom}:" | sort | awk '{print $2}')
  nics_d=$(for d in ${railnics[@]+"${railnics[@]}"}; do echo "${RAIL_PCI[$d]} $d"; done | grep -i "^${dom}:" | sort | awk '{print $2}')
  set -- $nics_d
  for k in $gpus_d; do
    GPU_NIC[$k]="${1:-}"; [ -n "${1:-}" ] && shift
  done
done
for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do
  d=${GPU_NIC[$k]}
  log "GPU$k (${GPU_PCI[$k]}) -> ${d:-?} (${RAIL_PCI[$d]:-none}) rail-local"
done

# ---- 4. Rail reachability (optional, needs --peer) ------------------------
if [ -n "$PEER" ]; then
  log ""; log "--- Rail reachability to peer $PEER ---"
  log "(rail-aligned should pass; cross-rail failing => rail-only fabric)"
  # Deriving the peer's rail addresses from ours would mean guessing the site's
  # addressing plan, so require them: PEER_IPS is a space-separated list in rail
  # order. Without it there is nothing honest to measure from one node, and saying
  # so beats printing an empty section that looks like a passing test.
  if [ -z "${PEER_IPS:-}" ]; then
    log "SKIPPED: set PEER_IPS to the peer's per-rail addresses, in rail order, e.g."
    log "  PEER_IPS='192.168.50.2 192.168.51.2 ...' ./probe_topology.sh --peer $PEER"
    log "  (read them off the peer's own topo_report, or just use xrail_worker.sbatch,"
    log "   which discovers both sides itself and tests RDMA as well as IP)"
  else
    read -r -a peer_ips <<< "$PEER_IPS"
    printf -v _hdr "%-10s %-12s %-8s %s" src_rail src_ndev stat kind; log "$_hdr"
    si=0
    for d in ${nics[@]+"${nics[@]}"}; do
      nd=${RAIL_NDEV[$d]}
      [ -z "$nd" ] && continue
      ti=0
      for pip in "${peer_ips[@]}"; do
        pf=-4; case $pip in *:*) pf=-6 ;; esac
        if ping "$pf" -c 2 -W 2 -I "$nd" "$pip" >/dev/null 2>&1; then st=OK; else st=FAIL; fi
        kind=$([ "$si" = "$ti" ] && echo same-rail || echo cross-rail)
        printf -v _row "%-10s %-12s %-8s %s -> %s" "rail$si" "$nd" "$st" "$kind" "$pip"
        log "$_row"
        ti=$((ti + 1))
      done
      si=$((si + 1))
    done
    log "NOTE: IP only. Cross-rail ICMP can pass on a fabric whose cross-rail RDMA"
    log "      still fails — confirm at the RDMA layer before classifying."
  fi
fi

# ---- 5. Emit Mermaid ------------------------------------------------------
{
  echo "graph LR"
  echo "  subgraph NODE[$hostn]"
  for k in $(echo "${!GPU_PCI[@]}" | tr ' ' '\n' | sort -n); do
    d=${GPU_NIC[$k]}
    echo "    G$k[\"GPU$k<br/>${GPU_PCI[$k]}\"] --- N_$d[\"$d / ${RAIL_NDEV[$d]}<br/>${RAIL_ADDR[$d]}\"]"
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
    echo "  \"GPU$k\" -> \"$d\\n${RAIL_ADDR[$d]}\" [dir=none, style=dotted, label=\"PCIe\"];"
  done
  echo "}"
} > "$DOT"

log ""
log "Wrote: $REPORT  $MMD  $DOT"
log "Next:  python3 make_report.py $OUT_DIR   # builds the richer diagrams + HTML report"
log "Render this raw DOT directly: dot -Tpng $DOT -o ${DOT%.dot}.png"
log "Render Mermaid: paste $MMD into https://mermaid.live"
