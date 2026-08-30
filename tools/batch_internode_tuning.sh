#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Batch Internode EP Dispatch+Combine Tuning Script
#
# Runs on the DRIVER node (rank 0). Launches the peer node (rank 1) via SSH.
# Sweeps over (max_tokens x hidden_dim) combinations, saving tuning results
# into JSON config files.
#
# Prerequisites:
#   - Passwordless SSH from driver to peer node
#   - Same repo path on both nodes (or set --remote-repo-root)
#   - Network interface accessible on both nodes (--ifname)
#
# Usage examples:
#
#   # Single shape
#   bash tools/batch_internode_tuning.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --kernel-type v1 --num-qp 2 --tokens-list "128"
#
#   # With docker on both nodes, driver staying on the host. --local-docker
#   # reaches rank 0's container with `docker exec` the same way --docker
#   # reaches the peer's, so SSH runs from the host and its keys never have to
#   # be copied into a container.
#   bash tools/batch_internode_tuning.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --local-docker <CONTAINER0> --docker <CONTAINER1> \
#       --kernel-type v1 --num-qp 2 --dtype fp4 --combine-dtype bf16 \
#       --quant-type fp8_direct_cast
#
#   # v1_ll, fp8→bf16
#   bash tools/batch_internode_tuning.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --kernel-type v1_ll --num-qp 2 --dtype fp8_e4m3_fnuz --combine-dtype bf16 \
#       --quant-type none
#
#   # Exploratory quick sweep -- reduced candidate grid, cannot be saved, so
#   # saving has to be turned off explicitly
#   bash tools/batch_internode_tuning.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --kernel-type v1 --num-qp 2 --tuning-scope quick --config-output ''
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INTERNODE_SCRIPT="examples/ops/dispatch_combine/test_dispatch_combine_internode.py"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

# ---- Defaults ----
MASTER_ADDR=""
MASTER_PORT=1234
PEER_HOST=""
IFNAME=""
REMOTE_REPO_ROOT=""
DOCKER=""
LOCAL_DOCKER=""
SSH_KEY=""
RDMA_SL=""
RDMA_TC=""
TUNING_SCOPE="full"
KERNEL_TYPE="v1"
NUM_QP=1
TOKENS_LIST="64,128,256,512,1024,2048,4096"
HIDDEN_DIMS="7168"
DTYPE="fp4"
COMBINE_DTYPE="bf16"
QUANT_TYPE="fp8_direct_cast"
CONFIG_OUTPUT="auto"
GPU_PER_NODE=8
TIMEOUT_SEC=600

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master-addr)       MASTER_ADDR="$2";       shift 2 ;;
        --master-port)       MASTER_PORT="$2";       shift 2 ;;
        --peer-host)         PEER_HOST="$2";         shift 2 ;;
        --ifname)            IFNAME="$2";            shift 2 ;;
        --remote-repo-root)  REMOTE_REPO_ROOT="$2";  shift 2 ;;
        --docker)            DOCKER="$2";            shift 2 ;;
        --local-docker)      LOCAL_DOCKER="$2";      shift 2 ;;
        --ssh-key)           SSH_KEY="$2";           shift 2 ;;
        --rdma-sl)           RDMA_SL="$2";           shift 2 ;;
        --rdma-tc)           RDMA_TC="$2";           shift 2 ;;
        --tuning-scope)      TUNING_SCOPE="$2";      shift 2 ;;
        --kernel-type)       KERNEL_TYPE="$2";       shift 2 ;;
        --num-qp)            NUM_QP="$2";            shift 2 ;;
        --tokens-list)       TOKENS_LIST="$2";       shift 2 ;;
        --hidden-dims)       HIDDEN_DIMS="$2";       shift 2 ;;
        --dtype)             DTYPE="$2";             shift 2 ;;
        --combine-dtype)     COMBINE_DTYPE="$2";     shift 2 ;;
        --quant-type)        QUANT_TYPE="$2";        shift 2 ;;
        --config-output)     CONFIG_OUTPUT="$2";     shift 2 ;;
        --gpu-per-node)      GPU_PER_NODE="$2";      shift 2 ;;
        --timeout)           TIMEOUT_SEC="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validate required args ----
for var in MASTER_ADDR PEER_HOST IFNAME; do
    if [[ -z "${!var}" ]]; then
        # ${var,,} lowercases but leaves the underscores, which would name a
        # flag that does not exist (--master_addr for --master-addr).
        flag="--${var,,}"
        echo "Error: ${flag//_/-} is required"
        exit 1
    fi
done

[[ -z "$REMOTE_REPO_ROOT" ]] && REMOTE_REPO_ROOT="$REPO_ROOT"

# Saving requires the full sweep. The tuning script itself refuses this
# combination, but it does so per combo after both ranks are up; catching it
# here fails in a second instead of after a two-node launch, and says which of
# the two knobs to change.
if [[ "$TUNING_SCOPE" == "quick" && -n "$CONFIG_OUTPUT" ]]; then
    echo "Error: --tuning-scope quick cannot write a tuning config."
    echo "       quick sweeps a reduced candidate grid, so its winner is not"
    echo "       a result worth committing. Either run the full sweep"
    echo "       (--tuning-scope full) or explore without saving"
    echo "       (--config-output '')."
    exit 1
fi

# ---- SSH options ----
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
[[ -n "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")

# ---- Convert comma-separated lists to arrays ----
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS_LIST"
IFS=',' read -ra HIDDEN_DIM_ARRAY <<< "$HIDDEN_DIMS"

# ---- Build log filename ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COMB_TAG="${COMBINE_DTYPE:-$DTYPE}"
QUANT_TAG=""
[[ "$QUANT_TYPE" != "none" ]] && QUANT_TAG="_${QUANT_TYPE}"
LOG_FILE="${LOG_DIR}/batch_internode_${KERNEL_TYPE}_ep$((GPU_PER_NODE*2))_${DTYPE}_${COMB_TAG}${QUANT_TAG}_${TUNING_SCOPE}_${TIMESTAMP}.log"

TOTAL_COMBOS=$(( ${#HIDDEN_DIM_ARRAY[@]} * ${#TOKEN_ARRAY[@]} ))

# ---- Print summary ----
echo "============================================================"
echo "Batch Internode Tuning"
echo "============================================================"
echo "  master_addr:         $MASTER_ADDR"
echo "  peer_host:           $PEER_HOST"
echo "  ifname:              $IFNAME"
echo "  docker (peer):       ${DOCKER:-<none, direct execution>}"
echo "  docker (local):      ${LOCAL_DOCKER:-<none, direct execution>}"
echo "  ssh_key:             ${SSH_KEY:-<default>}"
echo "  rdma_sl:             ${RDMA_SL:-<not set>}"
echo "  rdma_tc:             ${RDMA_TC:-<not set>}"
echo "  tuning_scope:        $TUNING_SCOPE"
echo "  kernel_type:         $KERNEL_TYPE"
echo "  num_qp:              $NUM_QP"
echo "  gpu_per_node:        $GPU_PER_NODE"
echo "  ep_size:             $((GPU_PER_NODE * 2))"
echo "  tokens_list:         ${TOKEN_ARRAY[*]}"
echo "  hidden_dims:         ${HIDDEN_DIM_ARRAY[*]}"
echo "  dtype:               $DTYPE"
echo "  combine_dtype:       ${COMBINE_DTYPE:-same as dtype}"
echo "  quant_type:          $QUANT_TYPE"
echo "  config_output:       $CONFIG_OUTPUT"
echo "  timeout:             ${TIMEOUT_SEC}s per combo"
echo "  total combos:        $TOTAL_COMBOS"
echo "  log:                 $LOG_FILE"
echo "  local repo:          $REPO_ROOT"
echo "  remote repo:         $REMOTE_REPO_ROOT"
echo "============================================================"
echo ""

# ---- Verify SSH connectivity ----
#
# DOCKER_EXEC is probed rather than fixed: `docker exec` needs access to the
# docker daemon socket, which comes from membership in the `docker` group, not
# from being root. Where the operator is in that group, plain `docker exec`
# works and `sudo` fails outright on hosts that grant no sudo rights at all.
# Try unprivileged first, fall back to sudo, and report both failures together
# rather than blaming SSH for a permissions problem.
DOCKER_EXEC=""
echo "Verifying SSH to $PEER_HOST ..."
if [[ -n "$DOCKER" ]]; then
    for candidate in "docker exec" "sudo -n docker exec"; do
        if ssh "${SSH_OPTS[@]}" "$PEER_HOST" "$candidate $DOCKER bash -c 'echo ok'" &>/dev/null; then
            DOCKER_EXEC="$candidate"
            break
        fi
    done
    if [[ -z "$DOCKER_EXEC" ]]; then
        echo "Error: cannot reach container '$DOCKER' on $PEER_HOST."
        echo "       Tried 'docker exec' and 'sudo -n docker exec'. Check that"
        echo "       passwordless SSH works, the container is running, and the"
        echo "       remote user is in the docker group (or has sudo rights)."
        exit 1
    fi
    echo "  docker exec prefix:  $DOCKER_EXEC"
else
    if ! ssh "${SSH_OPTS[@]}" "$PEER_HOST" "echo ok" &>/dev/null; then
        echo "Error: Cannot SSH to $PEER_HOST (passwordless SSH required)"
        exit 1
    fi
fi
echo "SSH OK"
echo ""

# ---- Resolve how rank 0 is launched ----
#
# --local-docker exists so the driver itself can stay on the host while mori
# lives in a container: rank 0 is then reached with `docker exec` locally, the
# same way the peer is reached remotely. Without it the driver has to run
# inside the container, which means the container needs SSH access to the peer
# -- i.e. a private key has to be placed inside a container purely so the
# script can reach the other node.
LOCAL_DOCKER_EXEC=""
if [[ -n "$LOCAL_DOCKER" ]]; then
    for candidate in "docker exec" "sudo -n docker exec"; do
        if $candidate "$LOCAL_DOCKER" bash -c 'echo ok' &>/dev/null; then
            LOCAL_DOCKER_EXEC="$candidate"
            break
        fi
    done
    if [[ -z "$LOCAL_DOCKER_EXEC" ]]; then
        echo "Error: cannot exec into local container '$LOCAL_DOCKER'."
        echo "       Tried 'docker exec' and 'sudo -n docker exec'. Check that"
        echo "       the container is running and this user is in the docker"
        echo "       group (or has sudo rights)."
        exit 1
    fi
    echo "Local rank runs in container '$LOCAL_DOCKER' via: $LOCAL_DOCKER_EXEC"
    echo ""
fi

# Run a cleanup command where rank 0 lives, so a timed-out `docker exec` does
# not leave the torchrun processes running inside the container.
kill_local() {
    if [[ -n "$LOCAL_DOCKER" ]]; then
        $LOCAL_DOCKER_EXEC "$LOCAL_DOCKER" bash -c "$1" 2>/dev/null || true
    else
        eval "$1" 2>/dev/null || true
    fi
}

# ---- Build common python args (shared by both ranks) ----
build_py_args() {
    local MAX_TOKENS="$1"
    local HIDDEN_DIM="$2"
    local SAVE_ARG="$3"

    local ARGS=(
        --cmd tuning
        --kernel-type "$KERNEL_TYPE"
        --num-qp "$NUM_QP"
        --max-tokens "$MAX_TOKENS"
        --hidden-dim "$HIDDEN_DIM"
        --dtype "$DTYPE"
        --quant-type "$QUANT_TYPE"
    )
    [[ -n "$COMBINE_DTYPE" ]] && ARGS+=(--combine-dtype "$COMBINE_DTYPE")
    [[ -n "$SAVE_ARG" ]] && ARGS+=(--save-tuning-config "$SAVE_ARG")
    echo "${ARGS[@]}"
}

# ---- Build torchrun command for one rank ----
build_torchrun_cmd() {
    local NODE_RANK="$1"
    local THE_REPO_ROOT="$2"
    local PY_ARGS="$3"

    local ENV_VARS="GPU_PER_NODE=$GPU_PER_NODE"
    ENV_VARS+=" GLOO_SOCKET_IFNAME=$IFNAME"
    ENV_VARS+=" MORI_SOCKET_IFNAME=$IFNAME"
    ENV_VARS+=" PYTHONPATH=${THE_REPO_ROOT}/python:${THE_REPO_ROOT}"
    ENV_VARS+=" MORI_TUNING_SCOPE=$TUNING_SCOPE"
    ENV_VARS+=" OMP_NUM_THREADS=4"
    [[ -n "$RDMA_SL" ]] && ENV_VARS+=" MORI_RDMA_SL=$RDMA_SL"
    [[ -n "$RDMA_TC" ]] && ENV_VARS+=" MORI_RDMA_TC=$RDMA_TC"

    echo "cd $THE_REPO_ROOT && $ENV_VARS" \
         "torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=1" \
         "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT" \
         "$INTERNODE_SCRIPT $PY_ARGS"
}

# ---- Launch remote (rank 1) ----
launch_peer() {
    local CMD="$1"
    if [[ -n "$DOCKER" ]]; then
        ssh "${SSH_OPTS[@]}" "$PEER_HOST" \
            "$DOCKER_EXEC -w $REMOTE_REPO_ROOT $DOCKER bash -c \"$CMD\"" &
    else
        ssh "${SSH_OPTS[@]}" "$PEER_HOST" "bash -lc '$CMD'" &
    fi
    PEER_PID=$!
}

# ---- Kill remote processes ----
cleanup_peer() {
    kill "$PEER_PID" 2>/dev/null || true
    wait "$PEER_PID" 2>/dev/null || true
    local KILL_CMD='pkill -9 -f "[t]orchrun.*test_dispatch_combine_internode"; pkill -9 -f "[t]est_dispatch_combine_internode"'
    if [[ -n "$DOCKER" ]]; then
        ssh "${SSH_OPTS[@]}" "$PEER_HOST" \
            "$DOCKER_EXEC $DOCKER bash -c '$KILL_CMD'" 2>/dev/null || true
    else
        ssh "${SSH_OPTS[@]}" "$PEER_HOST" "$KILL_CMD" 2>/dev/null || true
    fi
}

# ---- Pre-run: kill residual processes ----
echo "Cleaning up residual processes..."
KILL_ALL='pkill -9 -f "[t]orchrun.*test_dispatch_combine_internode"; pkill -9 -f "[t]est_dispatch_combine_internode"'
kill_local "$KILL_ALL"
if [[ -n "$DOCKER" ]]; then
    ssh "${SSH_OPTS[@]}" "$PEER_HOST" \
        "$DOCKER_EXEC $DOCKER bash -c '$KILL_ALL'" 2>/dev/null || true
else
    ssh "${SSH_OPTS[@]}" "$PEER_HOST" "$KILL_ALL" 2>/dev/null || true
fi
sleep 2
echo "Cleanup done"
echo ""

COMBO_IDX=0
FAILED=0
HUNG_COMBOS=""

for HIDDEN_DIM in "${HIDDEN_DIM_ARRAY[@]}"; do
    for TOKENS in "${TOKEN_ARRAY[@]}"; do
        COMBO_IDX=$((COMBO_IDX + 1))
        echo ""
        echo "############################################################"
        echo "[$(date)] [$COMBO_IDX/$TOTAL_COMBOS] kernel=$KERNEL_TYPE, hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
        echo "############################################################"
        echo ""

        PY_ARGS_RANK0=$(build_py_args "$TOKENS" "$HIDDEN_DIM" "$CONFIG_OUTPUT")
        PY_ARGS_RANK1=$(build_py_args "$TOKENS" "$HIDDEN_DIM" "")

        CMD_RANK0=$(build_torchrun_cmd 0 "$REPO_ROOT" "$PY_ARGS_RANK0")
        CMD_RANK1=$(build_torchrun_cmd 1 "$REMOTE_REPO_ROOT" "$PY_ARGS_RANK1")

        # Launch peer (rank 1) via SSH in background
        launch_peer "$CMD_RANK1"
        sleep 3

        # Launch local (rank 0) with timeout
        set +e
        # Unquoted $LOCAL_DOCKER_EXEC on purpose: it is "docker exec" or
        # "sudo -n docker exec" and has to split into separate words.
        if [[ -n "$LOCAL_DOCKER" ]]; then
            LOCAL_RUN=($LOCAL_DOCKER_EXEC -w "$REPO_ROOT" "$LOCAL_DOCKER" bash -c "$CMD_RANK0")
        else
            LOCAL_RUN=(bash -c "$CMD_RANK0")
        fi
        timeout "$TIMEOUT_SEC" "${LOCAL_RUN[@]}" 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [[ $EXIT_CODE -eq 124 ]]; then
            FAILED=$((FAILED + 1))
            HUNG_COMBOS="${HUNG_COMBOS}\n  hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
            echo ""
            echo "!!! TIMEOUT (${TIMEOUT_SEC}s): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS — possible kernel hang !!!"
            echo ""
        elif [[ $EXIT_CODE -ne 0 ]]; then
            FAILED=$((FAILED + 1))
            echo ""
            echo "!!! FAILED (exit $EXIT_CODE): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS !!!"
            echo ""
        else
            echo "[$(date)] Completed [$COMBO_IDX/$TOTAL_COMBOS]"
        fi

        # Always cleanup peer
        kill_local 'pkill -9 -f "[t]orchrun.*test_dispatch_combine_internode"; pkill -9 -f "[t]est_dispatch_combine_internode"'
        cleanup_peer
        sleep 2
    done
done

echo ""
echo "============================================================"
echo "Batch internode tuning complete"
echo "  Total:    $TOTAL_COMBOS"
echo "  Failed:   $FAILED"
if [[ -n "$HUNG_COMBOS" ]]; then
    echo -e "  Hung:$HUNG_COMBOS"
fi
echo "  Config:   $CONFIG_OUTPUT"
echo "  Log:      $LOG_FILE"
echo "============================================================"
