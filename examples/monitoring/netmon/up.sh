#!/usr/bin/env bash
# Start VictoriaMetrics + Grafana for ainic-netmon recordings.
#
#   ./up.sh [DATA_DIR]          default DATA_DIR: ./data
#   NETMON_VM_PORT=18428 NETMON_GRAFANA_PORT=13000 ./up.sh /scratch/netmon
#   NETMON_VM_LISTEN=10.0.0.3 ./up.sh    # VictoriaMetrics also on this internal address, for
#                                        # pushes from other nodes (firewall it: see README)
#
# Writes .env next to docker-compose.yml so both containers run as you (files written to a
# shared or NFS directory stay yours, not root's).
set -euo pipefail
cd "$(dirname "$0")"
data="$(realpath -m "${1:-${NETMON_DATA:-./data}}")"
case "$data" in
  *"'"* | *$'\n'*)
    echo "error: the data directory path must not contain a single quote or a newline" >&2
    exit 1
    ;;
esac
mkdir -p "$data/vm" "$data/grafana"
for d in "$data/vm" "$data/grafana"; do
  if [ ! -w "$d" ]; then
    echo "error: $d is not writable by $(id -un)" >&2
    exit 1
  fi
done
vm_port="${NETMON_VM_PORT:-8428}"
grafana_port="${NETMON_GRAFANA_PORT:-3000}"
retention="${NETMON_VM_RETENTION:-100y}"
if ! [[ $vm_port =~ ^[0-9]+$ && $grafana_port =~ ^[0-9]+$ && $retention =~ ^[0-9]+[a-z]*$ ]]; then
  echo "error: NETMON_VM_PORT / NETMON_GRAFANA_PORT must be numbers, NETMON_VM_RETENTION like 100y" >&2
  exit 1
fi
extra_listen=""
if [ -n "${NETMON_VM_LISTEN:-}" ]; then
  if ! [[ $NETMON_VM_LISTEN =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || [ "$NETMON_VM_LISTEN" = 0.0.0.0 ]; then
    echo "error: NETMON_VM_LISTEN must be one IPv4 address of this host (not 0.0.0.0)" >&2
    exit 1
  fi
  if ! ip -4 -o addr show 2>/dev/null | grep -q " $NETMON_VM_LISTEN/"; then
    echo "error: $NETMON_VM_LISTEN is not an address of this host" >&2
    exit 1
  fi
  extra_listen=",$NETMON_VM_LISTEN:$vm_port"
fi
# docker compose reads .env as data (single-quoted values are literal); it is never sourced.
cat > .env <<ENV
NETMON_UID='$(id -u)'
NETMON_GID='$(id -g)'
NETMON_DATA='$data'
NETMON_VM_PORT='$vm_port'
NETMON_GRAFANA_PORT='$grafana_port'
NETMON_VM_RETENTION='$retention'
NETMON_VM_EXTRA_LISTEN='$extra_listen'
ENV
docker compose up -d
echo "VictoriaMetrics: http://127.0.0.1:${vm_port}   Grafana: http://127.0.0.1:${grafana_port}"
if [ -n "$extra_listen" ]; then
  echo "VictoriaMetrics also on http://${NETMON_VM_LISTEN}:${vm_port} (no authentication: allow only the pushing nodes)"
fi
echo "import:  ainic-netmon-import RUN_DIR... --url http://127.0.0.1:${vm_port} --grafana http://127.0.0.1:${grafana_port}"
