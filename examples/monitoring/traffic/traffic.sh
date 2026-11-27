#!/usr/bin/env bash
# The monitoring stack of the MORI traffic metrics (mori_io_*, mori_ep_*, mori_gpu_rdma_*).
#   ./traffic.sh up [DATA_DIR]      start Prometheus, VictoriaMetrics, Grafana and the sampling
#                                   gateway; data in DATA_DIR (default ./data)
#   ./traffic.sh down               stop them; the data directory is kept
#   ./traffic.sh sampling           show the sampling of every MORI process Prometheus scrapes
#   ./traffic.sh sampling ep=8 io=64    count one EP dispatch + combine in 8, one IO transfer in 64
#                                   (counted N-fold; 1 counts everything, 0 nothing)
set -euo pipefail
cd "$(dirname "$0")"
name="mori-traffic-$(id -un)"

up() {
  data="$(realpath -m "${1:-${TRAFFIC_DATA:-./data}}")"
  prom_port="${TRAFFIC_PROM_PORT:-9090}"
  grafana_port="${TRAFFIC_GRAFANA_PORT:-3000}"
  vm_port="${TRAFFIC_VM_PORT:-8429}" # not 8428: the netmon stack's default, on the same host
  vm_url="${TRAFFIC_VM_URL:-}"
  gateway_port="${TRAFFIC_GATEWAY_PORT:-9390}"
  gateway_image="${TRAFFIC_GATEWAY_IMAGE:-python:3-alpine}" # any image with python3
  grafana_url="${TRAFFIC_GRAFANA_URL:-}"
  retention="${TRAFFIC_RETENTION:-15d}"
  listen="${TRAFFIC_LISTEN:-127.0.0.1}"
  targets_dir="$(realpath -m "${TRAFFIC_TARGETS_DIR:-/tmp/mori-metrics}")"
  if ! [[ $prom_port =~ ^[0-9]+$ && $grafana_port =~ ^[0-9]+$ && $vm_port =~ ^[0-9]+$ && $gateway_port =~ ^[0-9]+$ && $retention =~ ^[0-9]+[a-z]+$ ]]; then
    echo "error: TRAFFIC_PROM_PORT / TRAFFIC_GRAFANA_PORT / TRAFFIC_VM_PORT / TRAFFIC_GATEWAY_PORT must be numbers, TRAFFIC_RETENTION like 15d" >&2
    exit 1
  fi
  if [ -n "$grafana_url" ] && ! [[ $grafana_url =~ ^https?://[A-Za-z0-9_.:/-]+/$ ]]; then
    echo "error: TRAFFIC_GRAFANA_URL must be http(s)://host[:port]/ (with the slash at the end)" >&2
    exit 1
  fi
  if ! [[ $gateway_image =~ ^[A-Za-z0-9_.:/-]+$ ]]; then
    echo "error: TRAFFIC_GATEWAY_IMAGE must be an image name" >&2
    exit 1
  fi
  if [ -n "$vm_url" ] && ! [[ $vm_url =~ ^http://[A-Za-z0-9_.:-]+$ ]]; then
    echo "error: TRAFFIC_VM_URL must be http://host:port" >&2
    exit 1
  fi
  vm_listen="127.0.0.1:$vm_port"
  if [ -n "${TRAFFIC_VM_LISTEN:-}" ]; then
    if ! [[ $TRAFFIC_VM_LISTEN =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || [ "$TRAFFIC_VM_LISTEN" = 0.0.0.0 ]; then
      echo "error: TRAFFIC_VM_LISTEN must be one IPv4 address of this host (not 0.0.0.0)" >&2
      exit 1
    fi
    vm_listen="$vm_listen,$TRAFFIC_VM_LISTEN:$vm_port"
  fi
  if ! [[ $listen =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
    echo "error: TRAFFIC_LISTEN must be an IPv4 address of this host" >&2
    exit 1
  fi
  if [ -n "${TRAFFIC_TARGETS:-}" ] && ! [[ $TRAFFIC_TARGETS =~ ^[A-Za-z0-9_.:,-]+$ ]]; then
    echo "error: TRAFFIC_TARGETS must be a comma separated list of host:port" >&2
    exit 1
  fi
  containers=("$name-prometheus" "$name-grafana" "$name-sampling")
  ports=("$prom_port" "$grafana_port" "$gateway_port")
  if [ -z "$vm_url" ]; then
    containers+=("$name-vm")
    ports+=("$vm_port")
  fi
  for c in "${containers[@]}"; do
    if [ -n "$(docker ps -aq --filter "name=^${c}\$")" ]; then
      echo "error: container $c exists already: ./traffic.sh down first" >&2
      exit 1
    fi
  done
  for p in "${ports[@]}"; do
    if ss -Hltn "sport = :$p" 2>/dev/null | grep -q LISTEN; then
      echo "error: port $p is in use: choose TRAFFIC_PROM_PORT / TRAFFIC_GRAFANA_PORT / TRAFFIC_VM_PORT" >&2
      exit 1
    fi
  done
  mkdir -p "$data/prometheus" "$data/grafana" "$data/vm" "$targets_dir"
  # The docker daemon may not see this checkout (an NFS home with root squash): mount a copy.
  rm -rf "$data/conf"
  mkdir -p "$data/conf/targets-static"
  cp -r grafana prometheus.yml "$data/conf/"
  if [ -n "${TRAFFIC_TARGETS:-}" ]; then
    printf '[{"targets":["%s"]}]\n' "${TRAFFIC_TARGETS//,/\",\"}" >"$data/conf/targets-static/static.json"
  fi
  docker run -d --name "$name-prometheus" --network host --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    -v "$data/prometheus:/prometheus" \
    -v "$data/conf/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
    -v "$data/conf/targets-static:/targets-static:ro" \
    -v "$targets_dir:/targets:ro" \
    prom/prometheus:v3.16.0 \
    --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time="$retention" --web.listen-address="$listen:$prom_port" \
    --web.enable-lifecycle >/dev/null
  if [ -z "$vm_url" ]; then
    # No -dedup.minScrapeInterval: nothing may merge samples that are a millisecond apart.
    docker run -d --name "$name-vm" --network host --restart unless-stopped \
      --user "$(id -u):$(id -g)" -v "$data/vm:/storage" \
      victoriametrics/victoria-metrics:v1.156.0 \
      -storageDataPath=/storage -httpListenAddr="$vm_listen" -retentionPeriod="$retention" \
      -search.latencyOffset=0s >/dev/null
    vm_url="http://127.0.0.1:$vm_port"
  fi
  # Only Grafana talks to the sampling gateway, on 127.0.0.1. It also sets the scrape interval
  # (the overview's second switch): it rewrites the same prometheus.yml in place and has
  # Prometheus reload it (--web.enable-lifecycle above).
  cp sampling_gateway.py "$data/conf/"
  docker run -d --name "$name-sampling" --network host --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    -v "$data/conf/sampling_gateway.py:/sampling_gateway.py:ro" \
    -v "$data/conf/prometheus.yml:/prometheus.yml" \
    "$gateway_image" python3 -u /sampling_gateway.py --listen "127.0.0.1:$gateway_port" \
    --prometheus "http://$listen:$prom_port" --prometheus-config /prometheus.yml >/dev/null
  # The buttons call Grafana at the address it knows itself by: it has to be the browser's.
  if [ -z "$grafana_url" ] && [ "$listen" != 127.0.0.1 ]; then
    grafana_url="http://$listen:$grafana_port/"
  fi
  docker run -d --name "$name-grafana" --network host --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    ${grafana_url:+-e GF_SERVER_ROOT_URL="$grafana_url"} \
    -e GF_SECURITY_ACTIONS_ALLOW_POST_URL="/api/datasources/proxy/uid/mori-sampling/*" \
    -e TRAFFIC_GATEWAY_URL="http://127.0.0.1:$gateway_port" \
    -e GF_SERVER_HTTP_ADDR="$listen" -e GF_SERVER_HTTP_PORT="$grafana_port" \
    -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
    -e GF_AUTH_DISABLE_LOGIN_FORM=true \
    -e GF_DASHBOARDS_MIN_REFRESH_INTERVAL=1s \
    -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/mori-traffic/mori_traffic_overview.json \
    -e TRAFFIC_PROM_PORT="$prom_port" -e TRAFFIC_VM_URL="$vm_url" \
    -v "$data/grafana:/var/lib/grafana" \
    -v "$data/conf/grafana/provisioning/datasources:/etc/grafana/provisioning/datasources:ro" \
    -v "$data/conf/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro" \
    -v "$data/conf/grafana/dashboards:/var/lib/grafana/dashboards/mori-traffic:ro" \
    grafana/grafana:13.3.1 >/dev/null
  echo "Prometheus: http://$listen:${prom_port}/targets   Grafana: http://$listen:${grafana_port}   VictoriaMetrics: $vm_url"
  echo "sampling:   the buttons at the bottom of the overview dashboard (open Grafana at ${grafana_url:-http://localhost:$grafana_port/}), or ./traffic.sh sampling"
  echo "1 ms data:  run the jobs with MORI_METRICS_PUSH_URL=http://${TRAFFIC_VM_LISTEN:-127.0.0.1}:${vm_url##*:} and pick the data source mori-traffic-ms"
  echo "targets:    $targets_dir/*.json (run the jobs with MORI_METRICS_PORT=9400 MORI_METRICS_PORT_DIR=$targets_dir)${TRAFFIC_TARGETS:+ and $TRAFFIC_TARGETS}"
}

down() {
  for c in "$name-prometheus" "$name-grafana" "$name-vm" "$name-sampling"; do
    if [ -n "$(docker ps -aq --filter "name=^${c}\$")" ]; then
      docker rm -f "$c" >/dev/null && echo "removed $c"
    fi
  done
}

sampling() {
  prom="http://${TRAFFIC_LISTEN:-127.0.0.1}:${TRAFFIC_PROM_PORT:-9090}"
  query=""
  for arg in "$@"; do
    if ! [[ $arg =~ ^(ep|io|interval_ms)=[0-9]+$ ]]; then
      echo "error: '$arg' is not ep=<n>, io=<n> or interval_ms=<n>" >&2
      exit 1
    fi
    query="${query:+$query&}$arg"
  done
  if [ -n "${TRAFFIC_TARGETS:-}" ]; then
    if ! [[ $TRAFFIC_TARGETS =~ ^[A-Za-z0-9_.:,-]+$ ]]; then
      echo "error: TRAFFIC_TARGETS must be host:port[,host:port...]" >&2
      exit 1
    fi
    targets="${TRAFFIC_TARGETS//,/ }"
  else
    if ! body="$(curl -fsS -m 10 "$prom/api/v1/targets?state=active")"; then
      echo "error: no Prometheus at $prom (set TRAFFIC_PROM_PORT, or give TRAFFIC_TARGETS)" >&2
      exit 1
    fi
    targets="$(printf '%s' "$body" | python3 -c '
import json, sys
for t in json.load(sys.stdin)["data"]["activeTargets"]:
    if t.get("health") == "up" and t["labels"].get("job") == "mori_traffic":
        print(t["labels"]["instance"])')"
  fi
  if [ -z "$targets" ]; then
    echo "no running MORI process found" >&2
    exit 1
  fi
  failed=0
  for target in $targets; do
    if [ -n "$query" ]; then
      answer="$(curl -sS -m 10 -X POST "http://$target/sampling?$query" 2>&1)" || failed=$((failed + 1))
    else
      answer="$(curl -sS -m 10 "http://$target/sampling" 2>&1)" || failed=$((failed + 1))
    fi
    printf '%-28s %s\n' "$target" "$answer"
  done
  if [ "$failed" -ne 0 ]; then
    echo "$failed process(es) did not answer" >&2
    exit 1
  fi
}

cmd="${1:-}"
[ $# -eq 0 ] || shift
case "$cmd" in
  up) up "$@" ;;
  down) down ;;
  sampling) sampling "$@" ;;
  *)
    echo "usage: $0 up [DATA_DIR] | down | sampling [ep=N] [io=N] [interval_ms=N]" >&2
    exit 1
    ;;
esac
