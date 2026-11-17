#!/usr/bin/env bash
# Start VictoriaMetrics + Grafana for ainic-netmon and xgmi-netmon (one stack for both tools).
#
#   ./up.sh [DATA_DIR]          default DATA_DIR: ./data (local disk: VictoriaMetrics on NFS
#                               is not supported)
#   NETMON_VM_PORT=18428 NETMON_GRAFANA_PORT=13000 ./up.sh /scratch/netmon
#   NETMON_VM_LISTEN=10.0.0.3 ./up.sh    # VictoriaMetrics also on this internal address, for
#                                        # pushes from other nodes (firewall it: see README)
#
# Both containers run as you (files in a shared or NFS directory stay yours) on the host
# network, listening on 127.0.0.1. Every dashboard in grafana/dashboards/ is loaded, with one
# data source (uid netmon-vm). docker run, not compose: docker-compose 1.29 is broken on some
# login hosts.
set -euo pipefail
cd "$(dirname "$0")"
data="$(realpath -m "${1:-${NETMON_DATA:-./data}}")"
vm_port="${NETMON_VM_PORT:-8428}"
grafana_port="${NETMON_GRAFANA_PORT:-3000}"
retention="${NETMON_VM_RETENTION:-100y}"
if ! [[ $vm_port =~ ^[0-9]+$ && $grafana_port =~ ^[0-9]+$ && $retention =~ ^[0-9]+[a-z]*$ ]]; then
  echo "error: NETMON_VM_PORT / NETMON_GRAFANA_PORT must be numbers, NETMON_VM_RETENTION like 100y" >&2
  exit 1
fi
listen="127.0.0.1:$vm_port"
if [ -n "${NETMON_VM_LISTEN:-}" ]; then
  if ! [[ $NETMON_VM_LISTEN =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || [ "$NETMON_VM_LISTEN" = 0.0.0.0 ]; then
    echo "error: NETMON_VM_LISTEN must be one IPv4 address of this host (not 0.0.0.0)" >&2
    exit 1
  fi
  if ! ip -4 -o addr show 2>/dev/null | grep -q " $NETMON_VM_LISTEN/"; then
    echo "error: $NETMON_VM_LISTEN is not an address of this host" >&2
    exit 1
  fi
  listen="$listen,$NETMON_VM_LISTEN:$vm_port"
fi
name="netmon-$(id -un)"
for c in "$name-vm" "$name-grafana"; do
  if [ -n "$(docker ps -aq --filter "name=^${c}\$")" ]; then
    echo "error: container $c exists already: ./down.sh first" >&2
    exit 1
  fi
done
# The earlier docker-compose version of this stack holds the same ports: stop it first.
for c in netmon-victoriametrics-1 netmon-grafana-1; do
  if [ -n "$(docker ps -q --filter "name=^${c}\$")" ] &&
    [ "$(docker inspect -f '{{.Config.User}}' "$c")" = "$(id -u):$(id -g)" ]; then
    echo "error: your docker-compose stack of this example ($c) is running: ./down.sh first" >&2
    exit 1
  fi
done
for p in "$vm_port" "$grafana_port"; do
  if ss -Hltn "sport = :$p" 2>/dev/null | grep -q LISTEN; then
    echo "error: port $p is in use (another stack?): choose NETMON_VM_PORT / NETMON_GRAFANA_PORT" >&2
    exit 1
  fi
done
mkdir -p "$data/vm" "$data/grafana"
for d in "$data/vm" "$data/grafana"; do
  if [ ! -w "$d" ]; then
    echo "error: $d is not writable by $(id -un)" >&2
    exit 1
  fi
done
# The docker daemon may not see this checkout (an NFS home with root squash): Grafana gets a
# copy of its configuration and dashboards in the data directory, refreshed on every start.
rm -rf "$data/conf"
cp -r grafana "$data/conf"
docker run -d --name "$name-vm" --network host --restart unless-stopped \
  --user "$(id -u):$(id -g)" -v "$data/vm:/storage" \
  victoriametrics/victoria-metrics:v1.156.0 \
  -storageDataPath=/storage -httpListenAddr="$listen" -retentionPeriod="$retention" \
  -search.latencyOffset=0s >/dev/null
# (-search.latencyOffset=0s: imported data is historical, no need to hide the newest 30 s;
#  no -dedup.minScrapeInterval: it would merge millisecond samples.)
docker run -d --name "$name-grafana" --network host --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -e GF_SERVER_HTTP_ADDR=127.0.0.1 -e GF_SERVER_HTTP_PORT="$grafana_port" \
  -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
  -e GF_AUTH_DISABLE_LOGIN_FORM=true \
  -e GF_DASHBOARDS_MIN_REFRESH_INTERVAL=1s \
  -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/netmon/netmon_overview.json \
  -e NETMON_VM_PORT="$vm_port" \
  -v "$data/grafana:/var/lib/grafana" \
  -v "$data/conf/provisioning/datasources:/etc/grafana/provisioning/datasources:ro" \
  -v "$data/conf/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro" \
  -v "$data/conf/dashboards:/var/lib/grafana/dashboards/netmon:ro" \
  grafana/grafana:13.3.1 >/dev/null
echo "VictoriaMetrics: http://127.0.0.1:${vm_port}   Grafana: http://127.0.0.1:${grafana_port}"
if [ -n "${NETMON_VM_LISTEN:-}" ]; then
  echo "VictoriaMetrics also on http://${NETMON_VM_LISTEN}:${vm_port} (no authentication: allow only the pushing nodes)"
fi
echo "push:    ainic-netmon record ... --push http://127.0.0.1:${vm_port}   (xgmi-netmon: the same URL)"
echo "import:  ainic-netmon-import RUN_DIR... --url http://127.0.0.1:${vm_port} --grafana http://127.0.0.1:${grafana_port}"
