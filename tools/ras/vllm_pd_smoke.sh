#!/usr/bin/env bash
# MORI RAS end-to-end smoke test: vLLM prefill/decode (1P1D) on one node, with the MoRIIO
# connector moving KV over xGMI and dummy weights, so it needs two GPUs and no checkpoint.
#
#   tools/ras/vllm_pd_smoke.sh [on] [kill] [off] [blocked]      (default: on)
#
# Each mode serves a few requests through vLLM's MoRIIO toy proxy and checks the RAS log lines:
#   on       MORI_RAS_ENABLE=1: every engine process starts one RAS runtime, and prefill and
#            decode link up. Prefill then stops on SIGTERM, and decode sees it leave.
#   kill     As on, but decode is SIGKILLed. Prefill, which opened the link, sees it close
#            and its reconnect refused (RAS_ENDPOINT_UNAVAILABLE: exit suspected).
#   off      MORI_RAS_ENABLE=0: no RAS runtime starts.
#   blocked  MORI_RAS_ENABLE=1 with MORI_RAS_PORT held by another process: RAS turns itself
#            off, and serving is unaffected.
# In every mode the requests must succeed, and the servers not killed must stop cleanly on
# SIGTERM.
#
# It starts a container from VLLM_IMAGE, copies this checkout into it, and pip-installs MORI
# there, so the checkout gets no root-owned build files. Settings:
#   VLLM_IMAGE      a ROCm vLLM image with the MoRIIO connector  (rocm/vllm:latest)
#   MODEL           a model whose config and tokenizer vLLM can fetch  (Qwen/Qwen3-0.6B)
#   PREFILL_GPU / DECODE_GPU                                       (0 / 1)
#   RESULTS_DIR     logs and the summary  (./ras_vllm_pd_results)
#   HF_HOME         Hugging Face cache, mounted into the container  (~/.cache/huggingface)
#   NUM_REQUESTS    requests per mode  (8)
#   RAS_LOG_LEVEL   MORI_RAS_LOG_LEVEL for the servers  (info)
#   KEEP_CONTAINER  1 keeps the container, and its MORI install, for another run
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODES=("$@")
[[ ${#MODES[@]} -eq 0 ]] && MODES=(on)
VLLM_IMAGE=${VLLM_IMAGE:-rocm/vllm:latest}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}
RESULTS_DIR=${RESULTS_DIR:-$PWD/ras_vllm_pd_results}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
NUM_REQUESTS=${NUM_REQUESTS:-8}
CONTAINER=${CONTAINER:-mori-ras-vllm-pd-$(id -un)}

# ------------------------------------------------------------------------------------------
# Inside the container.

free_port_block() {  # first of $1 consecutive free ports, below the ephemeral range
  python3 - "$1" <<'EOF'
import random, socket, sys
size = int(sys.argv[1])
low = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
def free(p):
    with socket.socket() as s:
        try:
            s.bind(("", p))
            return True
        except OSError:
            return False
while True:
    base = random.randrange(10000, low - size)
    if all(free(p) for p in range(base, base + size)):
        print(base)
        break
EOF
}

wait_healthy() {  # port pid log
  local deadline=$((SECONDS + 900))
  until curl -sf "http://127.0.0.1:$1/health" >/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "server on port $1 exited; see $3"; return 1; }
    ((SECONDS < deadline)) || { echo "server on port $1 not healthy after 900 s"; return 1; }
    sleep 2
  done
}

start_server() {  # role gpu port log; prints the pid
  local cfg
  cfg=$(printf '{"kv_connector":"MoRIIOConnector","kv_role":"%s","kv_connector_extra_config":{"backend":"xgmi","host_ip":"127.0.0.1","proxy_ip":"127.0.0.1","proxy_ping_port":"%s","http_port":"%s","handshake_port":"%s","notify_port":"%s"}}' \
    "$1" "$PING_PORT" "$3" "$(free_port_block 8)" "$(free_port_block 8)")
  HIP_VISIBLE_DEVICES=$2 CUDA_VISIBLE_DEVICES=$2 VLLM_HOST_IP=127.0.0.1 \
    VLLM_PORT=$(free_port_block 32) VLLM_WORKER_MULTIPROC_METHOD=spawn \
    setsid vllm serve "$MODEL" --port "$3" --load-format dummy --enforce-eager \
      --max-model-len 2048 --gpu-memory-utilization 0.2 --kv-transfer-config "$cfg" \
      >"$4" 2>&1 &
  echo $!
}

stop_server() {  # pid; true if it exits within 60 s of SIGTERM
  kill -TERM -- "-$1" 2>/dev/null || return 0
  local deadline=$((SECONDS + 60))
  while kill -0 "$1" 2>/dev/null; do
    ((SECONDS < deadline)) || { kill -KILL -- "-$1" 2>/dev/null; return 1; }
    sleep 1
  done
}

check() {  # description, then a command that must succeed
  local what=$1
  shift
  if "$@"; then
    echo "  PASS  $what"
  else
    echo "  FAIL  $what"
    FAILED=1
  fi
}

count() { grep -c -- "$1" "$2" || true; }

wait_log() {  # pattern log seconds: true once the log matches
  local deadline=$((SECONDS + $3))
  until grep -qE -- "$1" "$2"; do
    ((SECONDS < deadline)) || return 1
    sleep 0.5
  done
}

run_mode() {
  local mode=$1 dir="$RESULTS_DIR/$1"
  mkdir -p "$dir"
  : >"$dir/responses.jsonl"
  echo "== mode: $mode"
  export MORI_RAS_LOG_LEVEL=${RAS_LOG_LEVEL:-info}
  unset MORI_RAS_PORT
  local holder=""
  case $mode in
    on | kill) export MORI_RAS_ENABLE=1 ;;
    off) export MORI_RAS_ENABLE=0 ;;
    blocked)
      export MORI_RAS_ENABLE=1
      MORI_RAS_PORT=$(free_port_block 1)
      export MORI_RAS_PORT
      python3 -c "import socket,time; s=socket.socket(); s.bind(('',$MORI_RAS_PORT)); s.listen(); time.sleep(86400)" &
      holder=$!
      ;;
    *) echo "unknown mode $mode"; return 2 ;;
  esac

  # Older proxies fix service discovery at 36367; newer ones take a port.
  local ping_args=()
  if python3 "$PROXY" --help 2>/dev/null | grep -q -- --service-discovery-port; then
    PING_PORT=$(free_port_block 1)
    ping_args=(--service-discovery-port "$PING_PORT")
  else
    PING_PORT=36367
  fi
  local proxy_port prefill_port decode_port
  proxy_port=$(free_port_block 1)
  prefill_port=$(free_port_block 1)
  decode_port=$(free_port_block 1)
  # Quart's reloader reruns the proxy in a child, which then serves HTTP without the instances
  # that registered with its parent, so the proxy runs with the reloader off.
  setsid python3 - "$PROXY" --port "$proxy_port" "${ping_args[@]}" >"$dir/proxy.log" 2>&1 <<'EOF' &
import runpy, sys, quart
run = quart.Quart.run
quart.Quart.run = lambda self, *a, **k: run(self, *a, **{**k, "use_reloader": False})
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
EOF
  local proxy=$!
  local deadline=$((SECONDS + 60))
  until curl -s -o /dev/null "http://127.0.0.1:$proxy_port/"; do
    kill -0 "$proxy" 2>/dev/null && ((SECONDS < deadline)) || {
      echo "  FAIL  the proxy didn't start; see $dir/proxy.log"
      FAILED=1
      stop_server "$proxy" || true
      return
    }
    sleep 0.5
  done
  PREFILL=$(start_server kv_producer "$PREFILL_GPU" "$prefill_port" "$dir/prefill.log")
  DECODE=$(start_server kv_consumer "$DECODE_GPU" "$decode_port" "$dir/decode.log")

  local served=0
  if wait_healthy "$prefill_port" "$PREFILL" "$dir/prefill.log" &&
     wait_healthy "$decode_port" "$DECODE" "$dir/decode.log"; then
    sleep 5  # both register with the proxy after they are healthy
    for i in $(seq "$NUM_REQUESTS"); do
      curl -sf "http://127.0.0.1:$proxy_port/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"RAS smoke request $i: the quick brown fox\",\"max_tokens\":8,\"temperature\":0}" \
        >>"$dir/responses.jsonl" && served=$((served + 1))
      echo >>"$dir/responses.jsonl"
    done
  fi
  check "$served of $NUM_REQUESTS requests served" test "$served" -eq "$NUM_REQUESTS"

  for role in prefill decode; do
    local log="$dir/$role.log" started off
    started=$(count "RAS runtime .* listening on" "$log")
    off=$(count "RAS is off for this process" "$log")
    case $mode in
      on | kill)
        check "$role: RAS runtime started ($started)" test "$started" -ge 1
        check "$role: RAS link to its peer is up" grep -q "RAS link to runtime .* is up" "$log"
        check "$role: link round trip and clock offset measured" \
          grep -q "RAS link to runtime .*: round trip .* us, clock offset" "$log"
        ;;
      off) check "$role: no RAS runtime" test "$((started + off))" -eq 0 ;;
      blocked) check "$role: RAS off, port in use ($off)" test "$started" -eq 0 -a "$off" -ge 1 ;;
    esac
  done

  case $mode in
    on)
      # A clean exit sends LEAVE, unless vLLM kills its engine process before atexit runs;
      # then decode sees the link close without one.
      check "prefill stops on SIGTERM" stop_server "$PREFILL"
      check "decode: prefill gone (LEAVE, or LINK_CLOSED without one)" \
        wait_log "RAS peer .* left|LINK_CLOSED" "$dir/decode.log" 15
      check "decode stops on SIGTERM" stop_server "$DECODE"
      ;;
    kill)
      kill -KILL -- "-$DECODE"
      check "prefill: killed decode seen (RAS_ENDPOINT_UNAVAILABLE)" \
        wait_log "RAS_ENDPOINT_UNAVAILABLE" "$dir/prefill.log" 15
      check "prefill stops on SIGTERM" stop_server "$PREFILL"
      ;;
    *)
      check "prefill stops on SIGTERM" stop_server "$PREFILL"
      check "decode stops on SIGTERM" stop_server "$DECODE"
      ;;
  esac
  stop_server "$proxy" || true
  [[ -n $holder ]] && kill "$holder" 2>/dev/null || true
  wait 2>/dev/null || true
}

inside() {
  PROXY=$(find / -name moriio_toy_proxy_server.py -path '*disaggregated*' -not -path '/proc/*' \
    2>/dev/null | head -1)
  [[ -n $PROXY ]] || { echo "moriio_toy_proxy_server.py not found in the image"; exit 2; }
  python3 -c 'import quart' 2>/dev/null || pip install -q quart >"$RESULTS_DIR/pip.log" 2>&1 ||
    { echo "the proxy needs quart, and pip couldn't install it; see $RESULTS_DIR/pip.log"; exit 2; }
  if ! python3 -c 'import mori.io' 2>/dev/null; then
    echo "== installing MORI from $REPO"
    rm -rf /tmp/mori-src && mkdir -p /tmp/mori-src
    tar -C "$REPO" --exclude='./build*' --exclude='./ras_vllm_pd_results' -cf - . |
      tar -C /tmp/mori-src -xf -
    (cd /tmp/mori-src && BUILD_RAS=ON BUILD_UMBP=OFF \
      GPU_ARCHS=${GPU_ARCHS:-$(rocminfo | grep -m1 -o 'gfx[0-9a-f]*')} \
      pip install . --no-build-isolation >"$RESULTS_DIR/mori_install.log" 2>&1) ||
      { echo "MORI install failed; see $RESULTS_DIR/mori_install.log"; exit 2; }
  fi
  FAILED=0
  for mode in "${MODES[@]}"; do run_mode "$mode"; done
  exit "$FAILED"
}

if [[ ${MODES[0]} == --inside ]]; then
  MODES=("${MODES[@]:1}")
  inside
fi

# ------------------------------------------------------------------------------------------
# On the host.

mkdir -p "$RESULTS_DIR" "$HF_HOME"
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  # A kept container can't take a new mount, so one made for another RESULTS_DIR is replaced.
  mounted=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/results"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")
  [[ $mounted == "$RESULTS_DIR" ]] || docker rm -f "$CONTAINER" >/dev/null
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  # --init reaps exited servers, so a stopped one doesn't linger as a zombie that looks alive.
  docker run -d --init --name "$CONTAINER" --network host --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --shm-size 16G \
    -v "$REPO:$REPO:ro" -v "$RESULTS_DIR:/results" -v "$HF_HOME:/root/.cache/huggingface" \
    "$VLLM_IMAGE" sleep infinity >/dev/null
fi
rc=0
docker exec -e MODEL="$MODEL" -e PREFILL_GPU="$PREFILL_GPU" -e DECODE_GPU="$DECODE_GPU" \
  -e RESULTS_DIR=/results -e NUM_REQUESTS="$NUM_REQUESTS" -e RAS_LOG_LEVEL="${RAS_LOG_LEVEL:-info}" \
  "$CONTAINER" \
  bash "$REPO/tools/ras/vllm_pd_smoke.sh" --inside "${MODES[@]}" | tee "$RESULTS_DIR/summary.txt" ||
  rc=$?
docker exec "$CONTAINER" chown -R "$(id -u):$(id -g)" /results /root/.cache/huggingface \
  >/dev/null 2>&1 || true
[[ ${KEEP_CONTAINER:-0} == 1 ]] || docker rm -f "$CONTAINER" >/dev/null
exit "$rc"
