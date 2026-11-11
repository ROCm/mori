#!/usr/bin/env bash
# Start VictoriaMetrics + Grafana for ainic-netmon recordings.
#
#   ./up.sh [DATA_DIR]          default DATA_DIR: ./data
#   NETMON_VM_PORT=18428 NETMON_GRAFANA_PORT=13000 ./up.sh /scratch/netmon
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
# docker compose reads .env as data (single-quoted values are literal); it is never sourced.
cat > .env <<ENV
NETMON_UID='$(id -u)'
NETMON_GID='$(id -g)'
NETMON_DATA='$data'
NETMON_VM_PORT='$vm_port'
NETMON_GRAFANA_PORT='$grafana_port'
NETMON_VM_RETENTION='$retention'
ENV
docker compose up -d
echo "VictoriaMetrics: http://127.0.0.1:${vm_port}   Grafana: http://127.0.0.1:${grafana_port}"
echo "import:  ainic-netmon-import RUN_DIR... --url http://127.0.0.1:${vm_port} --grafana http://127.0.0.1:${grafana_port}"
