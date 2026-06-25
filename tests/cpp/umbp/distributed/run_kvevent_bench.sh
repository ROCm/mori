#!/usr/bin/env bash
# Driver for bench_umbp_agent_kvevent_pressure.
#
# Runs the master-pressure benchmark across the three kv-event propagation
# schemes (baseline / compressed / flush) in up to three run classes (select via
# the CLASSES env; default runs all three):
#
#   pressure : free-running (no round barrier), read-lag 1, steady-state load
#              across a (clients,gap) grid and both patterns.  At least one tier
#              per pattern uses --get-mode both so BatchRouteGet is exercised.
#              Headline = master per-RPC latency vs QPS.
#   miss     : --round-barrier --read-lag-rounds 0, small client counts.  The
#              gap maps cleanly onto key age, so the read-after-write miss_rate
#              is accurate (bursty load is acceptable here).
#   scale    : high-pressure client-count sweep (16..128) at a small/zero gap,
#              --key-space bounds memory while issuing full-rate RPCs.  Use this
#              to see whether master RPC latency degrades as concurrency grows.
#              Defaults to get-mode=exists: cross-client RDMA fetch (both/fetch)
#              trips a mori RDMA control-plane assertion past ~32 in-process
#              clients, so high-client runs stay master-only.
#
# Each scenario is a FRESH process (the heartbeat interval is read once into a
# function-local static, so a process can only exercise one interval) with a
# unique metrics port.  Two CSV segments per run are split into aggregate files.
#
# Env overrides:
#   BIN                 path to bench_umbp_agent_kvevent_pressure
#   OUTDIR              results directory (default ./kvevent_results)
#   CLASSES             space-separated subset of "pressure miss scale" (default all)
#   METRICS_PORT_BASE   first Prometheus port (default 19200)
#   ROUNDS / WARMUP / BATCH / PAGE_BYTES        pressure+miss workload size
#   CLIENTS_GRID        pressure tiers, "clients:gap_ms" (default "2:250 8:100 32:50")
#   SCALE_CLIENTS / SCALE_GAPS                  scale-class grid (default "16 32 64 96 128" / "0")
#   SCALE_ROUNDS / SCALE_WARMUP / SCALE_KEYSPACE / SCALE_GETMODE
#   COMPRESSED_TTL_SEC / COMPRESSED_DIVISOR     compressed heartbeat interval
#   METRICS_REPORT_INTERVAL_MS                  client metrics flush cadence
#   TASKSET_CPUS        optional cpu list to pin every run (e.g. "0-15")
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_bin() {
  if [[ -n "${BIN:-}" && -x "${BIN}" ]]; then echo "${BIN}"; return; fi
  local c
  for c in \
    "${SCRIPT_DIR}"/../../../../build*/tests/cpp/umbp/distributed/bench_umbp_agent_kvevent_pressure \
    "${SCRIPT_DIR}"/../../../../build/tests/cpp/umbp/distributed/bench_umbp_agent_kvevent_pressure; do
    if [[ -x "${c}" ]]; then echo "${c}"; return; fi
  done
  # Last resort: search the repo build trees.
  find "${SCRIPT_DIR}/../../../.." -type f -name bench_umbp_agent_kvevent_pressure -perm -u+x 2>/dev/null | head -n1
}

BIN="$(find_bin)"
if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  echo "ERROR: bench binary not found. Build it first or set BIN=..." >&2
  exit 1
fi

OUTDIR="${OUTDIR:-${PWD}/kvevent_results}"
mkdir -p "${OUTDIR}"
PORT="${METRICS_PORT_BASE:-19200}"
CLASSES="${CLASSES:-pressure miss scale}"

ROUNDS="${ROUNDS:-200}"
WARMUP="${WARMUP:-10}"
BATCH="${BATCH:-16}"
PAGE_BYTES="${PAGE_BYTES:-4096}"
COMPRESSED_TTL_SEC="${COMPRESSED_TTL_SEC:-1}"
COMPRESSED_DIVISOR="${COMPRESSED_DIVISOR:-10}"        # 1s/10 = 100ms
METRICS_REPORT_INTERVAL_MS="${METRICS_REPORT_INTERVAL_MS:-250}"
CLIENTS_GRID="${CLIENTS_GRID:-2:250 8:100 32:50}"

SCALE_CLIENTS="${SCALE_CLIENTS:-16 32 64 96 128}"
SCALE_GAPS="${SCALE_GAPS:-0}"
SCALE_ROUNDS="${SCALE_ROUNDS:-300}"
SCALE_WARMUP="${SCALE_WARMUP:-30}"
SCALE_KEYSPACE="${SCALE_KEYSPACE:-2048}"
SCALE_GETMODE="${SCALE_GETMODE:-exists}"

WORKLOAD_CSV="${OUTDIR}/workload.csv"
RPC_CSV="${OUTDIR}/rpc_latency.csv"
: > "${WORKLOAD_CSV}"
: > "${RPC_CSV}"

has_class() { [[ " ${CLASSES} " == *" $1 "* ]]; }

TASKSET_PREFIX=()
if [[ -n "${TASKSET_CPUS:-}" ]] && command -v taskset >/dev/null 2>&1; then
  TASKSET_PREFIX=(taskset -c "${TASKSET_CPUS}")
fi

# GNU time (-v gives "Percent of CPU" + maxRSS) is a useful side-observation per
# review S9-G, but it is not always installed; fall back to running bare.
TIMER_PREFIX=()
if [[ -x /usr/bin/time ]]; then
  TIMER_PREFIX=(/usr/bin/time -v)
elif command -v gtime >/dev/null 2>&1; then
  TIMER_PREFIX=(gtime -v)
fi

# heartbeat env for a given mode
hb_env() {
  case "$1" in
    compressed) echo "UMBP_HEARTBEAT_TTL_SEC=${COMPRESSED_TTL_SEC} UMBP_HEARTBEAT_INTERVAL_DIVISOR=${COMPRESSED_DIVISOR}" ;;
    *)          echo "" ;;  # baseline / flush use the default interval
  esac
}

# run_one <tag> <mode> -- remaining args are passed verbatim to the bench, so
# each caller controls --rounds / --batch / --key-space / --pattern / etc.
run_one() {
  local tag="$1"; shift
  local mode="$1"; shift
  local port=$((PORT++))
  local raw="${OUTDIR}/${tag}.out"
  local cpulog="${OUTDIR}/${tag}.cpu"
  local env_extra; env_extra="$(hb_env "${mode}")"

  echo ">> ${tag}  (mode=${mode} metrics_port=${port})"
  # shellcheck disable=SC2086
  env ${env_extra} UMBP_METRICS_REPORT_INTERVAL_MS="${METRICS_REPORT_INTERVAL_MS}" \
    "${TIMER_PREFIX[@]}" "${TASKSET_PREFIX[@]}" "${BIN}" \
      --mode "${mode}" --metrics-port "${port}" "$@" \
      >"${raw}" 2>"${cpulog}"
  if [[ ! -s "${raw}" ]]; then
    echo "   WARNING: empty output (see ${cpulog})" >&2
  fi

  # Split the two CSV segments (a blank line separates them) and tag each row.
  # Segment 1 = workload CSV (header starts "mode,"); segment 2 = rpc CSV
  # (header starts "rpc,").
  local in_seg2=0 wrote_w_hdr wrote_r_hdr
  wrote_w_hdr=$( [[ -s "${WORKLOAD_CSV}" ]] && echo 1 || echo 0 )
  wrote_r_hdr=$( [[ -s "${RPC_CSV}" ]] && echo 1 || echo 0 )
  while IFS= read -r line; do
    if [[ -z "${line}" ]]; then in_seg2=1; continue; fi
    if [[ ${in_seg2} -eq 0 ]]; then
      if [[ "${line}" == mode,* ]]; then
        [[ ${wrote_w_hdr} -eq 0 ]] && { echo "tag,${line}" >> "${WORKLOAD_CSV}"; wrote_w_hdr=1; }
      else
        echo "${tag},${line}" >> "${WORKLOAD_CSV}"
      fi
    else
      if [[ "${line}" == rpc,* ]]; then
        [[ ${wrote_r_hdr} -eq 0 ]] && { echo "tag,${line}" >> "${RPC_CSV}"; wrote_r_hdr=1; }
      else
        echo "${tag},${line}" >> "${RPC_CSV}"
      fi
    fi
  done < "${raw}"
}

echo "bench binary: ${BIN}"
echo "results dir : ${OUTDIR}"
echo "classes     : ${CLASSES}"

# ---- pressure class (free-running, standard matrix) ----
if has_class pressure; then
  for mode in baseline compressed flush; do
    for pat in broadcast rotate; do
      for tier in ${CLIENTS_GRID}; do
        clients="${tier%%:*}"; gap="${tier##*:}"
        run_one "pressure_${mode}_${pat}_c${clients}_g${gap}_exists" "${mode}" \
          --rounds "${ROUNDS}" --warmup-rounds "${WARMUP}" --batch "${BATCH}" --page-bytes "${PAGE_BYTES}" \
          --pattern "${pat}" --get-mode exists --clients "${clients}" --gap-ms "${gap}" --read-lag-rounds 1
      done
      # one get-mode=both tier per (mode,pattern) so BatchRouteGet is exercised
      both_tier="$(echo "${CLIENTS_GRID}" | awk '{print $2}')"
      [[ -z "${both_tier}" ]] && both_tier="8:100"
      bc="${both_tier%%:*}"; bg="${both_tier##*:}"
      run_one "pressure_${mode}_${pat}_c${bc}_g${bg}_both" "${mode}" \
        --rounds "${ROUNDS}" --warmup-rounds "${WARMUP}" --batch "${BATCH}" --page-bytes "${PAGE_BYTES}" \
        --pattern "${pat}" --get-mode both --clients "${bc}" --gap-ms "${bg}" --read-lag-rounds 1
    done
  done
fi

# ---- miss class (barrier, accurate gap->age) ----
if has_class miss; then
  for mode in baseline compressed flush; do
    for gap in 250 100 50; do
      run_one "miss_${mode}_g${gap}" "${mode}" \
        --rounds "${ROUNDS}" --warmup-rounds "${WARMUP}" --batch "${BATCH}" --page-bytes "${PAGE_BYTES}" \
        --pattern rotate --get-mode exists --clients 4 --gap-ms "${gap}" --round-barrier --read-lag-rounds 0
    done
  done
fi

# ---- scale class (high-pressure client-count sweep, key-space bounded) ----
if has_class scale; then
  for mode in baseline compressed flush; do
    for c in ${SCALE_CLIENTS}; do
      for g in ${SCALE_GAPS}; do
        run_one "scale_${mode}_c${c}_g${g}" "${mode}" \
          --rounds "${SCALE_ROUNDS}" --warmup-rounds "${SCALE_WARMUP}" --batch "${BATCH}" --page-bytes "${PAGE_BYTES}" \
          --pattern rotate --get-mode "${SCALE_GETMODE}" --clients "${c}" --gap-ms "${g}" \
          --key-space "${SCALE_KEYSPACE}" --read-lag-rounds 1
      done
    done
  done
fi

echo
echo "== workload summary (${WORKLOAD_CSV}) =="
column -s, -t < "${WORKLOAD_CSV}" 2>/dev/null || cat "${WORKLOAD_CSV}"
echo
echo "== master RPC latency (${RPC_CSV}) =="
column -s, -t < "${RPC_CSV}" 2>/dev/null || cat "${RPC_CSV}"
