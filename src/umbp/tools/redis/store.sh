#!/usr/bin/env bash
# Unified entrypoint to bring RESP stores up/down for the UMBP master Redis
# metadata backend. Run this ON THE HOST, next to the host-networked app
# container; every store uses host networking, so the app container reaches it
# at 127.0.0.1:<port>.
#
# Usage:
#   store.sh up   [single|dragonfly|cluster]   # default: single
#   store.sh down [single|dragonfly|cluster]
#   store.sh status [single|dragonfly|cluster]
#   store.sh seeds  [single|dragonfly|cluster] # print the URI(s) to export
#
# Topologies:
#   single    -> UMBP_REDIS_URI=tcp://127.0.0.1:6379
#   dragonfly -> UMBP_REDIS_URI=tcp://127.0.0.1:6380
#   cluster   -> UMBP_REDIS_URI=tcp://127.0.0.1:7000,...,7005  (+ UMBP_REDIS_CLUSTER=1)
#
# Docker (compose) is the productized path. When docker is unavailable (e.g.
# working purely inside an app container without host docker access), single and
# dragonfly fall back to run_local_backends.sh (source-built local processes);
# cluster has no process fallback here — use a host with docker.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${HERE}/docker-compose.yml"
ANNOUNCE_IP="${CLUSTER_ANNOUNCE_IP:-127.0.0.1}"

ACTION="${1:-}"
TOPO="${2:-single}"

have_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }

compose() { CLUSTER_ANNOUNCE_IP="${ANNOUNCE_IP}" docker compose -f "${COMPOSE_FILE}" "$@"; }

# Throwaway redis-cli on the host network (no host redis-cli needed).
rcli() { docker run --rm --network host redis:7-alpine redis-cli "$@"; }

cluster_ports() { echo 7000 7001 7002 7003 7004 7005; }

cluster_seeds() {
  local s="" p
  for p in $(cluster_ports); do s+="${s:+,}tcp://${ANNOUNCE_IP}:${p}"; done
  echo "$s"
}

print_seeds() {
  case "$TOPO" in
    single)    echo "export UMBP_METADATA_BACKEND=redis UMBP_REDIS_URI=tcp://127.0.0.1:6379" ;;
    dragonfly) echo "export UMBP_METADATA_BACKEND=redis UMBP_REDIS_URI=tcp://127.0.0.1:6380" ;;
    cluster)   echo "export UMBP_METADATA_BACKEND=redis UMBP_REDIS_CLUSTER=1 UMBP_REDIS_URI=$(cluster_seeds)" ;;
    *) echo "unknown topology: $TOPO" >&2; exit 2 ;;
  esac
}

wait_cluster_ok() {
  echo "[store] waiting for cluster_state:ok ..."
  for _ in $(seq 1 60); do
    if rcli -h "${ANNOUNCE_IP}" -p 7000 cluster info 2>/dev/null | grep -q "cluster_state:ok"; then
      echo "[store] cluster is ready"; return 0
    fi
    sleep 1
  done
  echo "[store] WARN: cluster not ready after 60s; check 'store.sh status cluster'" >&2
  return 1
}

up() {
  if have_docker; then
    compose --profile "$TOPO" up -d
    [ "$TOPO" = "cluster" ] && wait_cluster_ok || true
  else
    case "$TOPO" in
      single|dragonfly)
        echo "[store] docker unavailable; falling back to local processes (run_local_backends.sh)"
        "${HERE}/run_local_backends.sh" up ;;
      *) echo "[store] docker unavailable and no process fallback for '$TOPO'" >&2; exit 1 ;;
    esac
  fi
  echo "[store] ready. To point the master/tests at it:"
  print_seeds
}

down() {
  if have_docker; then
    compose --profile "$TOPO" down -v --remove-orphans
  else
    "${HERE}/run_local_backends.sh" down || true
  fi
}

status() {
  if have_docker; then
    compose --profile "$TOPO" ps
    [ "$TOPO" = "cluster" ] && rcli -h "${ANNOUNCE_IP}" -p 7000 cluster info 2>/dev/null | grep -E "cluster_state|cluster_known_nodes|cluster_size" || true
  else
    "${HERE}/run_local_backends.sh" status || true
  fi
}

case "$ACTION" in
  up) up ;;
  down) down ;;
  status) status ;;
  seeds) print_seeds ;;
  *) echo "usage: $0 {up|down|status|seeds} [single|dragonfly|cluster]" >&2; exit 2 ;;
esac
