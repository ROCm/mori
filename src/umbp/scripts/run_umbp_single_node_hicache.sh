#!/usr/bin/env bash
set -euo pipefail

# Single-node UMBP + SGLang correctness smoke test.
# Reuses the environment setup logic from the bench_pd_disagg skill but
# launches the non-PD hicache command from test_umbp_integration.sh.

# ==============================
# === Shared configuration  ====
# ==============================
USER_HOME="/home/ditian12"
NFS_BASE="/nfs/users/ditian12"
DOCKER_IMAGE="rocm/sgl-dev:v0.5.9-rocm700-mi30x-20260316"
EXTRA_MOUNTS="-v /home/thshan@amd.com:/home/thshan@amd.com -v /apps/data/models/:/models -v /apps:/apps -v /it-share:/it-share -v /usr/sbin/nicctl:/usr/sbin/nicctl"
CONTAINER="umbp-single-node"
MODEL_PATH="${MODEL_PATH:-/nfs/data/Deepseek-R1}"
RESULTS_DIR="${USER_HOME}/umbp_single_node_results"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ENABLE_DP="${ENABLE_DP:-false}"
START_UMBP_MASTER="${START_UMBP_MASTER:-true}"
UMBP_MASTER_ADDRESS="${UMBP_MASTER_ADDRESS:-127.0.0.1:15558}"
UMBP_NODE_ADDRESS="${UMBP_NODE_ADDRESS:-127.0.0.1}"
UMBP_IO_ENGINE_PORT="${UMBP_IO_ENGINE_PORT:-16000}"
UMBP_PEER_SERVICE_PORT="${UMBP_PEER_SERVICE_PORT:-17000}"
USE_DUMMY_WEIGHTS="${USE_DUMMY_WEIGHTS:-false}"

SGLANG_BRANCH="${SGLANG_BRANCH:-main}"
MORI_BRANCH="${MORI_BRANCH:-main}"
SGLANG_REPO="${SGLANG_REPO:-${NFS_BASE}/sglang}"
MORI_REPO="${MORI_REPO:-${NFS_BASE}/mori}"

declare -a SGLANG_REMOTE_CANDIDATES=()
if [[ -n "${SGLANG_REMOTE:-}" ]]; then
  SGLANG_REMOTE_CANDIDATES+=("$SGLANG_REMOTE")
fi
SGLANG_REMOTE_CANDIDATES+=(hicache umb umb_ssh origin)

declare -a MORI_REMOTE_CANDIDATES=()
if [[ -n "${MORI_REMOTE:-}" ]]; then
  MORI_REMOTE_CANDIDATES+=("$MORI_REMOTE")
fi
MORI_REMOTE_CANDIDATES+=(ssh origin)

usage() {
  local exit_code="${1:-0}"
  cat <<EOF
Usage: $(basename "$0") [--use-dummy-weights|--real-weights]

  --use-dummy-weights   Skip checkpoint shard validation and launch SGLang with dummy weights.
  --real-weights        Require real checkpoints (default).
  -h, --help            Show this help message.
EOF
  exit "$exit_code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-dummy-weights)
      USE_DUMMY_WEIGHTS=true
      shift
      ;;
    --real-weights|--no-dummy-weights)
      USE_DUMMY_WEIGHTS=false
      shift
      ;;
    --help|-h)
      usage 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
  esac
done

# ==============================
# === Derived configuration ====
# ==============================
LOCAL_NODE="$(hostname)"
LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "$LOCAL_IP" ]]; then
  LOCAL_IP="127.0.0.1"
fi

# ==============================
# === Helper functions      ====
# ==============================
cleanup() {
  set +e
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker exec "$CONTAINER" pkill -f sglang.launch_server 2>/dev/null || true
    docker exec "$CONTAINER" pkill -f umbp_master 2>/dev/null || true
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  set -e
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

resolve_git_remote() {
  local repo_path="$1"
  shift
  for candidate in "$@"; do
    [[ -z "$candidate" ]] && continue
    if git -C "$repo_path" remote get-url "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

update_repo_branch() {
  local repo_path="$1"
  local branch="$2"
  local label="$3"
  shift 3
  if [[ ! -d "${repo_path}/.git" ]]; then
    fail "${label} repo '${repo_path}' is not a git repository. Override ${label^^}_REPO if needed."
  fi

  local remote
  if ! remote=$(resolve_git_remote "$repo_path" "$@"); then
    fail "No valid git remote found for ${label} repo '${repo_path}'. Set ${label^^}_REMOTE to an existing remote."
  fi

  echo "  - Updating ${label} repo (${repo_path}) to ${remote}/${branch}"
  pushd "$repo_path" >/dev/null

  git fetch "$remote" "$branch"

  if git rev-parse --verify --quiet "$branch" >/dev/null 2>&1; then
    git checkout "$branch" >/dev/null 2>&1 || git checkout "$branch"
  elif git show-ref --verify --quiet "refs/remotes/${remote}/${branch}" >/dev/null 2>&1; then
    git checkout -B "$branch" "${remote}/${branch}"
  else
    popd >/dev/null
    fail "${label} branch '$branch' not found on remote '$remote'."
  fi

  git pull --rebase "$remote" "$branch"
  popd >/dev/null
}

trap cleanup EXIT

mkdir -p "$RESULTS_DIR"

if [[ "$USE_DUMMY_WEIGHTS" != "true" ]]; then
  if [[ ! -d "$MODEL_PATH" ]]; then
    fail "MODEL_PATH '$MODEL_PATH' not found. Override MODEL_PATH or create the directory."
  fi
  shopt -s nullglob
  safer_glob=("$MODEL_PATH"/model-*.safetensors)
  shopt -u nullglob
  if [[ ${#safer_glob[@]} -eq 0 ]]; then
    fail "MODEL_PATH '$MODEL_PATH' does not contain any safetensors shards."
  fi
else
  echo "Dummy weights enabled; skipping checkpoint validation for MODEL_PATH='${MODEL_PATH}'."
  if ! mkdir -p "$MODEL_PATH"; then
    fail "Failed to create MODEL_PATH directory '${MODEL_PATH}' for dummy weights."
  fi
fi

echo "[1/5] Updating git repositories (sglang: ${SGLANG_BRANCH}, mori: ${MORI_BRANCH})"
update_repo_branch "$SGLANG_REPO" "$SGLANG_BRANCH" "sglang" "${SGLANG_REMOTE_CANDIDATES[@]}"
update_repo_branch "$MORI_REPO" "$MORI_BRANCH" "mori" "${MORI_REMOTE_CANDIDATES[@]}"

echo "[2/5] Preparing Docker container (${CONTAINER}) on node ${LOCAL_NODE}"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --device /dev/dri --device /dev/kfd \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  -w "$NFS_BASE" \
  --env HUGGINGFACE_HUB_CACHE=/models --env MODELSCOPE_CACHE=/models \
  -v /nfs:/nfs \
  -v "$USER_HOME:$USER_HOME" \
  $EXTRA_MOUNTS \
  --shm-size 32G \
  "$DOCKER_IMAGE" sleep infinity >/dev/null

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  fail "container failed to start"
fi

if [[ "$USE_DUMMY_WEIGHTS" == "true" ]]; then
  echo "[3/5] Launching SGLang server (dummy weights) inside container"
else
  echo "[3/5] Launching SGLang server (real weights) inside container"
fi

docker exec "$CONTAINER" bash -c '
set -euo pipefail

_IFNAME=$(ip -o addr show 2>/dev/null | awk -v ip="'"${LOCAL_IP}"'" '"'"'$4 ~ ip"/" {print $2; exit}'"'"')
[[ -z "$_IFNAME" ]] && _IFNAME=$(ip route get 127.0.0.1 2>/dev/null | awk '"'"'NR==1{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1);exit}}'"'"')
NET_IFNAME=${_IFNAME:-ens14np0}

_IB_HCA_LIST="" _RDMA_EXCLUDE=""
for _dev in $(ls /sys/class/infiniband/ 2>/dev/null); do
  _is_mgmt=false
  for _nd in $(ls /sys/class/infiniband/$_dev/device/net/ 2>/dev/null); do
    [[ "$_nd" == "$NET_IFNAME" ]] && _is_mgmt=true && break
  done
  if $_is_mgmt; then
    _RDMA_EXCLUDE="${_RDMA_EXCLUDE:+${_RDMA_EXCLUDE},}$_dev"
  else
    _IB_HCA_LIST="${_IB_HCA_LIST:+${_IB_HCA_LIST},}$_dev"
  fi
done
NCCL_IB_HCA=${_IB_HCA_LIST:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}
MORI_RDMA_DEVICES=${_RDMA_EXCLUDE:+^$_RDMA_EXCLUDE}

export PYTHONPATH=/home/ditian12/python_patches:'"${NFS_BASE}"'/sglang/python:'"${NFS_BASE}"'/mori/python:/sgl-workspace/aiter:${PYTHONPATH:-}
export MC_IB_TC=96
export MORI_ENABLE_SDMA=0
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_MORI_FP4_DISP=false
export SGLANG_MORI_FP8_DISP=false
export SGLANG_MORI_FP8_COMB=true
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
export NCCL_IB_HCA=$NCCL_IB_HCA
export GLOO_SOCKET_IFNAME=$NET_IFNAME
export NCCL_SOCKET_IFNAME=$NET_IFNAME
export MORI_SOCKET_IFNAME=$NET_IFNAME
export MORI_RDMA_DEVICES=$MORI_RDMA_DEVICES
export SGLANG_USE_AITER=1
export KV_CACHE_DTYPE=fp8_e4m3
export USE_DUMMY_WEIGHTS='"${USE_DUMMY_WEIGHTS}"'
export MODEL_PATH='"${MODEL_PATH}"'
export MORI_GLOBAL_LOG_LEVEL=INFO
export MORI_UMBP_LOG_LEVEL=INFO
export MORI_IO_LOG_LEVEL=ERROR
export MORI_RDMA_SL=3
export MORI_RDMA_TC=96
export RESULTS_DIR='"${RESULTS_DIR}"'
export UMBP_MASTER_ADDRESS='"${UMBP_MASTER_ADDRESS}"'
export UMBP_NODE_ADDRESS='"${UMBP_NODE_ADDRESS}"'
export UMBP_IO_ENGINE_HOST=127.0.0.1
export UMBP_IO_ENGINE_PORT='"${UMBP_IO_ENGINE_PORT}"'
export UMBP_PEER_SERVICE_PORT='"${UMBP_PEER_SERVICE_PORT}"'
export UMBP_CACHE_REMOTE_FETCHES=false

if [[ "${USE_DUMMY_WEIGHTS}" == "true" ]]; then
  echo "Dummy weights enabled inside container; will launch SGLang with --load-format dummy."
else
  echo "Real weights enabled inside container; expecting checkpoints at ${MODEL_PATH}."
fi

LOAD_FORMAT_FLAG=""
if [[ "${USE_DUMMY_WEIGHTS}" == "true" ]]; then
  LOAD_FORMAT_FLAG="--load-format dummy"
fi

SERVER_LOG=${RESULTS_DIR}/server_$(date +%Y%m%d_%H%M%S).log
mkdir -p ${RESULTS_DIR}

if [[ '"${START_UMBP_MASTER}"' == "true" ]]; then
  MASTER_LOG=${RESULTS_DIR}/umbp_master_$(date +%Y%m%d_%H%M%S).log
  if [[ -x '"${NFS_BASE}"'/mori/build_$(hostname)/src/umbp/umbp_master ]]; then
    '"${NFS_BASE}"'/mori/build_$(hostname)/src/umbp/umbp_master ${UMBP_MASTER_ADDRESS} > "${MASTER_LOG}" 2>&1 &
    MASTER_PID=$!
    echo "Started UMBP master (PID: ${MASTER_PID}, log: ${MASTER_LOG})"
  else
    echo "WARNING: umbp_master binary not found; skipping master launch" >&2
    MASTER_PID=""
  fi
else
  MASTER_PID=""
fi

SGLANG_MORI_FP8_DISP=false \
MORI_SHMEM_MODE=ISOLATION \
SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384 \
python -m sglang.launch_server \
  --enable-cache-report \
  --enable-metrics \
  --model-path "${MODEL_PATH}" \
  --tp-size '"${TP_SIZE}"' \
  --decode-log-interval 1 \
  --trust-remote-code \
  --watchdog-timeout 1000000 \
  --chunked-prefill-size 131072 \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --max-total-tokens 1024 \
  --mem-fraction-static 0.5 \
  --enable-hierarchical-cache \
  --hicache-write-policy write_through \
  --hicache-mem-layout page_first \
  --hicache-ratio 5.0 \
  --hicache-storage-backend umbp \
  --hicache-storage-backend-extra-config '"'"'{
    "dram_capacity_bytes": 1073741824,
    "ssd_enabled": false,
    "auto_promote_on_read": true,
    "prefetch_threshold": 0
  }'"'"' \
  ${LOAD_FORMAT_FLAG:+$LOAD_FORMAT_FLAG} \
  '"$(if [[ "${ENABLE_DP}" == "true" ]]; then printf -- '--dp-size %s --ep-size %s --moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1' "${DP_SIZE}" "${EP_SIZE}"; fi)"' \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID (log: $SERVER_LOG)"

MAX_WAIT=${MAX_WAIT_OVERRIDE:-3600}
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -sf http://127.0.0.1:30000/health >/dev/null; then
    echo "Server ready after $ELAPSED seconds"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server exited unexpectedly; log tail:" >&2
    tail -40 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo "Server failed to start within timeout; log tail:" >&2
  tail -40 "$SERVER_LOG" >&2
  exit 1
fi

echo "[4/5] Running correctness probe"
curl -sf -X POST http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"deepseek-v3\",\"prompt\":\"Say hello in one word.\",\"max_tokens\":8}" \
  | tee ${RESULTS_DIR}/probe_response.json

if ! grep -q "choices" ${RESULTS_DIR}/probe_response.json; then
  echo "Inference response missing choices field" >&2
  exit 1
fi

echo "Probe succeeded"

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
if [[ -n "${MASTER_PID:-}" ]]; then
  kill $MASTER_PID 2>/dev/null || true
  wait $MASTER_PID 2>/dev/null || true
fi
'

echo "[5/5] Completed single-node smoke test; artifacts saved under ${RESULTS_DIR}"
