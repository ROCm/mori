#!/usr/bin/env bash
# FALLBACK launcher — prefer store.sh (docker compose) for the productized path.
#
# This launches single-node Redis and Dragonfly as local PROCESSES (source-built
# / downloaded binaries, no docker), for environments without host docker access
# (e.g. working purely inside an app container). The productized way to run the
# stores as independent containers is `store.sh {up|down|status} <topology>`
# (docker-compose.yml), which also covers the Redis Cluster topology this script
# does not. store.sh auto-falls-back to this script for single/dragonfly when
# docker is unavailable.
#
# Both speak RESP, so the master connects to either via UMBP_REDIS_URI only.
#
#   Redis     -> tcp://127.0.0.1:6379
#   Dragonfly -> tcp://127.0.0.1:6380
#
# Getting the binaries (in order of preference, auto-selected):
#   Redis:     redis-server on PATH -> prebuilt in RUN_DIR -> apt-get ->
#              build the official source with make (works when the apt mirror
#              is blocked but github over HTTPS is reachable, e.g. this CI image).
#   Dragonfly: prebuilt in RUN_DIR -> download the official release tarball.
# Both are the stock upstream binaries; only how they are obtained varies.
#
# Usage:
#   src/umbp/tools/redis/run_local_backends.sh up       # install (if needed) + start
#   src/umbp/tools/redis/run_local_backends.sh down     # stop
#   src/umbp/tools/redis/run_local_backends.sh status
set -euo pipefail

RUN_DIR="${UMBP_REDIS_RUN_DIR:-/tmp/umbp_redis_bench}"
REDIS_PORT="${UMBP_REDIS_PORT:-6379}"
DF_PORT="${UMBP_DRAGONFLY_PORT:-6380}"
DF_VERSION="${UMBP_DRAGONFLY_VERSION:-v1.23.2}"
DF_BIN="${RUN_DIR}/dragonfly"
REDIS_VERSION="${UMBP_REDIS_VERSION:-7.2.5}"
REDIS_SRC="${RUN_DIR}/redis-src"

# Resolved by ensure_redis() / find_redis_cli().
REDIS_SERVER=""
REDIS_CLI=""

mkdir -p "${RUN_DIR}"

# Locate a redis-cli without triggering an install (used by status/down).
find_redis_cli() {
  if command -v redis-cli >/dev/null 2>&1; then
    command -v redis-cli
  elif [[ -x "${RUN_DIR}/redis-cli" ]]; then
    echo "${RUN_DIR}/redis-cli"
  fi
}

build_redis_from_source() {
  if ! command -v git >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    echo "[run_local_backends] cannot build redis: need git + make" >&2
    return 1
  fi
  echo "[run_local_backends] building official redis ${REDIS_VERSION} from source..."
  if [[ ! -d "${REDIS_SRC}" ]]; then
    git clone --depth 1 --branch "${REDIS_VERSION}" https://github.com/redis/redis.git "${REDIS_SRC}"
  fi
  make -C "${REDIS_SRC}" -j"$(nproc)" BUILD_TLS=no USE_SYSTEMD=no MALLOC=libc
  cp "${REDIS_SRC}/src/redis-server" "${RUN_DIR}/redis-server"
  cp "${REDIS_SRC}/src/redis-cli" "${RUN_DIR}/redis-cli"
}

ensure_redis() {
  # 1) Already installed on PATH.
  if command -v redis-server >/dev/null 2>&1 && command -v redis-cli >/dev/null 2>&1; then
    REDIS_SERVER="$(command -v redis-server)"
    REDIS_CLI="$(command -v redis-cli)"
    return 0
  fi
  # 2) Previously built/downloaded into RUN_DIR.
  if [[ -x "${RUN_DIR}/redis-server" && -x "${RUN_DIR}/redis-cli" ]]; then
    REDIS_SERVER="${RUN_DIR}/redis-server"
    REDIS_CLI="${RUN_DIR}/redis-cli"
    return 0
  fi
  # 3) apt fast path (skipped automatically when the mirror is unreachable).
  if command -v apt-get >/dev/null 2>&1; then
    echo "[run_local_backends] trying apt-get install redis-server..."
    if apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq redis-server >/dev/null 2>&1; then
      REDIS_SERVER="$(command -v redis-server)"
      REDIS_CLI="$(command -v redis-cli)"
      return 0
    fi
    echo "[run_local_backends] apt unavailable; falling back to source build"
  fi
  # 4) Build the official source (apt mirror blocked but github reachable).
  build_redis_from_source
  REDIS_SERVER="${RUN_DIR}/redis-server"
  REDIS_CLI="${RUN_DIR}/redis-cli"
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

  if ! "${REDIS_CLI}" -p "${REDIS_PORT}" ping >/dev/null 2>&1; then
    echo "[run_local_backends] starting redis-server on ${REDIS_PORT}"
    "${REDIS_SERVER}" --port "${REDIS_PORT}" --daemonize yes \
      --appendonly yes --appendfsync everysec \
      --dir "${RUN_DIR}" --logfile "${RUN_DIR}/redis.log" \
      --save ""
  fi

  if [[ -x "${DF_BIN}" ]]; then
    if ! "${REDIS_CLI}" -p "${DF_PORT}" ping >/dev/null 2>&1; then
      echo "[run_local_backends] starting dragonfly on ${DF_PORT}"
      # NO global --default_lua_flags: it forces every Lua script (incl. the read
      # hot path) into Dragonfly's global-transaction mode (store-wide lock),
      # serializing all proactor threads and erasing the block-sharding win. The
      # write/control scripts carry a per-script "--!df flags=allow-undeclared-keys"
      # directive (a no-op Lua comment on Redis) while the read scripts declare
      # all keys via KEYS[], so they run per-shard in parallel. See lua_scripts.h.
      "${DF_BIN}" --port "${DF_PORT}" --logtostderr --alsologtostderr=false \
        --dir "${RUN_DIR}" >"${RUN_DIR}/dragonfly.log" 2>&1 &
      echo $! >"${RUN_DIR}/dragonfly.pid"
      sleep 1
    fi
  fi
  status
}

down() {
  echo "[run_local_backends] stopping backends"
  local cli
  cli="$(find_redis_cli)"
  if [[ -n "${cli}" ]]; then
    "${cli}" -p "${REDIS_PORT}" shutdown nosave >/dev/null 2>&1 || true
  fi
  if [[ -f "${RUN_DIR}/dragonfly.pid" ]]; then
    kill "$(cat "${RUN_DIR}/dragonfly.pid")" >/dev/null 2>&1 || true
    rm -f "${RUN_DIR}/dragonfly.pid"
  fi
}

status() {
  local cli
  cli="$(find_redis_cli)"
  echo -n "[run_local_backends] redis(${REDIS_PORT}): "
  { [[ -n "${cli}" ]] && "${cli}" -p "${REDIS_PORT}" ping 2>/dev/null; } || echo "DOWN"
  echo -n "[run_local_backends] dragonfly(${DF_PORT}): "
  { [[ -n "${cli}" ]] && "${cli}" -p "${DF_PORT}" ping 2>/dev/null; } || echo "DOWN"
}

case "${1:-up}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 {up|down|status}"; exit 2 ;;
esac
