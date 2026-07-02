#!/usr/bin/env bash
# Launch single-node Redis and Dragonfly as local processes (no docker needed),
# for the UMBP master Redis-metadata-store Phase 1 benchmark. Both speak RESP,
# so the master connects to either via UMBP_REDIS_URI only.
#
#   Redis     -> tcp://127.0.0.1:6379
#   Dragonfly -> tcp://127.0.0.1:6380
#
# Usage:
#   src/umbp/tools/redis/run_local_backends.sh up      # install (if needed) + start
#   src/umbp/tools/redis/run_local_backends.sh down     # stop
#   src/umbp/tools/redis/run_local_backends.sh status
set -euo pipefail

RUN_DIR="${UMBP_REDIS_RUN_DIR:-/tmp/umbp_redis_bench}"
REDIS_PORT="${UMBP_REDIS_PORT:-6379}"
DF_PORT="${UMBP_DRAGONFLY_PORT:-6380}"
DF_VERSION="${UMBP_DRAGONFLY_VERSION:-v1.23.2}"
DF_BIN="${RUN_DIR}/dragonfly"

mkdir -p "${RUN_DIR}"

ensure_redis() {
  if command -v redis-server >/dev/null 2>&1; then return 0; fi
  echo "[run_local_backends] installing redis-server via apt..."
  apt-get update -qq && apt-get install -y -qq redis-server >/dev/null
}

ensure_dragonfly() {
  if [[ -x "${DF_BIN}" ]]; then return 0; fi
  echo "[run_local_backends] downloading dragonfly ${DF_VERSION}..."
  local arch tarball url
  arch="$(uname -m)"
  case "${arch}" in
    x86_64) tarball="dragonfly-x86_64.tar.gz" ;;
    aarch64) tarball="dragonfly-aarch64.tar.gz" ;;
    *) echo "[run_local_backends] unsupported arch ${arch}"; return 1 ;;
  esac
  url="https://github.com/dragonflydb/dragonfly/releases/download/${DF_VERSION}/${tarball}"
  curl -fsSL "${url}" -o "${RUN_DIR}/${tarball}"
  tar -xzf "${RUN_DIR}/${tarball}" -C "${RUN_DIR}"
  # The tarball unpacks to dragonfly-<arch>; normalize to ${DF_BIN}.
  local extracted
  extracted="$(find "${RUN_DIR}" -maxdepth 1 -name 'dragonfly-*' -type f | head -1)"
  [[ -n "${extracted}" ]] && cp "${extracted}" "${DF_BIN}"
  chmod +x "${DF_BIN}"
}

up() {
  ensure_redis
  ensure_dragonfly || echo "[run_local_backends] WARN: dragonfly unavailable; redis only"

  if ! redis-cli -p "${REDIS_PORT}" ping >/dev/null 2>&1; then
    echo "[run_local_backends] starting redis-server on ${REDIS_PORT}"
    redis-server --port "${REDIS_PORT}" --daemonize yes \
      --appendonly yes --appendfsync everysec \
      --dir "${RUN_DIR}" --logfile "${RUN_DIR}/redis.log" \
      --save ""
  fi

  if [[ -x "${DF_BIN}" ]]; then
    if ! redis-cli -p "${DF_PORT}" ping >/dev/null 2>&1; then
      echo "[run_local_backends] starting dragonfly on ${DF_PORT}"
      # allow-undeclared-keys: our Lua scripts derive auxiliary same-slot keys
      # (nodes:alive, block:*, ...) from the shared hash tag rather than passing
      # every one via KEYS[]. Redis single-node permits this; Dragonfly enforces
      # declared keys unless told otherwise. One deployment flag keeps a single
      # script implementation portable across both.
      "${DF_BIN}" --port "${DF_PORT}" --logtostderr --alsologtostderr=false \
        --default_lua_flags=allow-undeclared-keys \
        --dir "${RUN_DIR}" >"${RUN_DIR}/dragonfly.log" 2>&1 &
      echo $! >"${RUN_DIR}/dragonfly.pid"
      sleep 1
    fi
  fi
  status
}

down() {
  echo "[run_local_backends] stopping backends"
  redis-cli -p "${REDIS_PORT}" shutdown nosave >/dev/null 2>&1 || true
  if [[ -f "${RUN_DIR}/dragonfly.pid" ]]; then
    kill "$(cat "${RUN_DIR}/dragonfly.pid")" >/dev/null 2>&1 || true
    rm -f "${RUN_DIR}/dragonfly.pid"
  fi
}

status() {
  echo -n "[run_local_backends] redis(${REDIS_PORT}): "
  redis-cli -p "${REDIS_PORT}" ping 2>/dev/null || echo "DOWN"
  echo -n "[run_local_backends] dragonfly(${DF_PORT}): "
  redis-cli -p "${DF_PORT}" ping 2>/dev/null || echo "DOWN"
}

case "${1:-up}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 {up|down|status}"; exit 2 ;;
esac
