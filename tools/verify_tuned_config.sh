#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Verify a saved EP tuning config against the actual correctness check
#
# Runs on the DRIVER node (rank 0). Launches the peer node (rank 1) via SSH.
# Sweeps the same (max_tokens x hidden_dim) combinations batch_internode_tuning.sh
# does, but with --cmd test under MORI_EP_LAUNCH_CONFIG_MODE=AUTO instead of
# --cmd tuning: each combo's dispatch/combine op is built with whatever
# block_num/warp_per_block/rdma_block_num AUTO mode looks up from
# python/mori/ops/tuning_configs/*.json for that shape, and --cmd test then
# runs 500 rounds actually comparing dispatch/combine output against the
# expected values.
#
# Why this exists: the tuning sweep in test_dispatch_combine_internode.py only
# times each (block, warp, rdma) candidate -- run_bench_once never compares
# output, so nothing in the sweep itself confirms the winning candidate it
# writes to disk produces correct results, only that it was fast. This script
# is the correctness check that closes that gap, run once against the winner
# instead of 75 times against every candidate (which would make the sweep
# itself far slower for a question the sweep doesn't need answered while
# picking a fastest config).
#
# Prerequisites: same as batch_internode_tuning.sh (passwordless SSH to the
# peer node, matching repo path or --remote-repo-root, --ifname reachable on
# both nodes). Intended to run right after batch_internode_tuning.sh writes a
# JSON, before that JSON is committed.
#
# Usage examples:
#
#   # Verify what a full tuning run just wrote, same shapes as tuning
#   bash tools/verify_tuned_config.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --kernel-type v1_ll --num-qp 2 --dtype fp8_e4m3_fnuz \
#       --combine-dtype bf16 --quant-type none \
#       --tokens-list "4,8,16" --hidden-dims "6144"
#
#   # With docker (remote node runs inside a container)
#   bash tools/verify_tuned_config.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --docker <CONTAINER> --ssh-key <SSH_KEY_PATH> \
#       --kernel-type v1_ll --num-qp 2 --tokens-list "4,8,16" --hidden-dims "6144"
#
# Exit status: non-zero if any shape reports a nonzero "error times" on any
# rank, or times out, or crashes -- the same signal batch_internode_tuning.sh
# uses for its own per-combo failures.
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
SSH_KEY=""
RDMA_SL=""
KERNEL_TYPE="v1"
NUM_QP=1
TOKENS_LIST="64,128,256,512,1024,2048,4096"
HIDDEN_DIMS="7168"
DTYPE="fp4"
COMBINE_DTYPE="bf16"
QUANT_TYPE="fp8_direct_cast"
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
        --ssh-key)            SSH_KEY="$2";          shift 2 ;;
        --rdma-sl)            RDMA_SL="$2";           shift 2 ;;
        --kernel-type)        KERNEL_TYPE="$2";       shift 2 ;;
        --num-qp)             NUM_QP="$2";            shift 2 ;;
        --tokens-list)        TOKENS_LIST="$2";       shift 2 ;;
        --hidden-dims)        HIDDEN_DIMS="$2";       shift 2 ;;
        --dtype)              DTYPE="$2";             shift 2 ;;
        --combine-dtype)      COMBINE_DTYPE="$2";     shift 2 ;;
        --quant-type)         QUANT_TYPE="$2";        shift 2 ;;
        --gpu-per-node)       GPU_PER_NODE="$2";      shift 2 ;;
        --timeout)            TIMEOUT_SEC="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validate required args ----
for var in MASTER_ADDR PEER_HOST IFNAME; do
    [[ -z "${!var}" ]] && { echo "Error: --${var,,} is required"; exit 1; }
done

[[ -z "$REMOTE_REPO_ROOT" ]] && REMOTE_REPO_ROOT="$REPO_ROOT"

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
LOG_FILE="${LOG_DIR}/verify_tuned_${KERNEL_TYPE}_ep$((GPU_PER_NODE*2))_${DTYPE}_${COMB_TAG}${QUANT_TAG}_${TIMESTAMP}.log"

TOTAL_COMBOS=$(( ${#HIDDEN_DIM_ARRAY[@]} * ${#TOKEN_ARRAY[@]} ))

# ---- Print summary ----
echo "============================================================"
echo "Verify Tuned Config (correctness, not speed)"
echo "============================================================"
echo "  master_addr:         $MASTER_ADDR"
echo "  peer_host:           $PEER_HOST"
echo "  ifname:              $IFNAME"
echo "  docker:              ${DOCKER:-<none, direct execution>}"
echo "  ssh_key:             ${SSH_KEY:-<default>}"
echo "  rdma_sl:             ${RDMA_SL:-<not set>}"
echo "  kernel_type:         $KERNEL_TYPE"
echo "  num_qp:              $NUM_QP"
echo "  gpu_per_node:        $GPU_PER_NODE"
echo "  ep_size:             $((GPU_PER_NODE * 2))"
echo "  tokens_list:         ${TOKEN_ARRAY[*]}"
echo "  hidden_dims:         ${HIDDEN_DIM_ARRAY[*]}"
echo "  dtype:               $DTYPE"
echo "  combine_dtype:       ${COMBINE_DTYPE:-same as dtype}"
echo "  quant_type:          $QUANT_TYPE"
echo "  timeout:             ${TIMEOUT_SEC}s per combo"
echo "  total combos:        $TOTAL_COMBOS"
echo "  log:                 $LOG_FILE"
echo "  local repo:          $REPO_ROOT"
echo "  remote repo:         $REMOTE_REPO_ROOT"
echo "============================================================"
echo ""
echo "Each combo reads block_num/warp_per_block/rdma_block_num from"
echo "python/mori/ops/tuning_configs/*.json via MORI_EP_LAUNCH_CONFIG_MODE=AUTO"
echo "-- this verifies whatever is currently on disk, not a fixed geometry."
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

# ---- Build common python args (shared by both ranks) ----
build_py_args() {
    local MAX_TOKENS="$1"
    local HIDDEN_DIM="$2"

    local ARGS=(
        --cmd test
        --kernel-type "$KERNEL_TYPE"
        --num-qp "$NUM_QP"
        --max-tokens "$MAX_TOKENS"
        --hidden-dim "$HIDDEN_DIM"
        --dtype "$DTYPE"
        --quant-type "$QUANT_TYPE"
    )
    [[ -n "$COMBINE_DTYPE" ]] && ARGS+=(--combine-dtype "$COMBINE_DTYPE")
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
    ENV_VARS+=" MORI_EP_LAUNCH_CONFIG_MODE=AUTO"
    ENV_VARS+=" OMP_NUM_THREADS=4"
    [[ -n "$RDMA_SL" ]] && ENV_VARS+=" MORI_RDMA_SL=$RDMA_SL"

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
    local KILL_CMD='pkill -9 -f torchrun; pkill -9 -f test_dispatch_combine_internode; pkill -9 -f "multiprocessing.spawn"'
    if [[ -n "$DOCKER" ]]; then
        ssh "${SSH_OPTS[@]}" "$PEER_HOST" \
            "$DOCKER_EXEC $DOCKER bash -c '$KILL_CMD'" 2>/dev/null || true
    else
        ssh "${SSH_OPTS[@]}" "$PEER_HOST" "$KILL_CMD" 2>/dev/null || true
    fi
}

# ---- Pre-run: kill residual processes ----
echo "Cleaning up residual processes..."
KILL_ALL='pkill -9 -f torchrun; pkill -9 -f test_dispatch_combine_internode; pkill -9 -f "multiprocessing.spawn"'
eval "$KILL_ALL" 2>/dev/null || true
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
FAILED_COMBOS=""
HUNG_COMBOS=""

for HIDDEN_DIM in "${HIDDEN_DIM_ARRAY[@]}"; do
    for TOKENS in "${TOKEN_ARRAY[@]}"; do
        COMBO_IDX=$((COMBO_IDX + 1))
        echo ""
        echo "############################################################"
        echo "[$(date)] [$COMBO_IDX/$TOTAL_COMBOS] kernel=$KERNEL_TYPE, hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
        echo "############################################################"
        echo ""

        PY_ARGS=$(build_py_args "$TOKENS" "$HIDDEN_DIM")

        CMD_RANK0=$(build_torchrun_cmd 0 "$REPO_ROOT" "$PY_ARGS")
        CMD_RANK1=$(build_torchrun_cmd 1 "$REMOTE_REPO_ROOT" "$PY_ARGS")

        # Launch peer (rank 1) via SSH in background
        launch_peer "$CMD_RANK1"
        sleep 3

        # Launch local (rank 0) with timeout, capture output to grep for errors
        COMBO_LOG="$(mktemp)"
        set +e
        timeout "$TIMEOUT_SEC" bash -c "$CMD_RANK0" 2>&1 | tee "$COMBO_LOG" | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        # test_dispatch_combine() prints one "error times:  N" line per rank;
        # any nonzero N is a real dispatch/combine output mismatch, not noise.
        BAD_RANKS=$(grep -cE "error times: +[1-9]" "$COMBO_LOG" || true)
        rm -f "$COMBO_LOG"

        if [[ $EXIT_CODE -eq 124 ]]; then
            FAILED=$((FAILED + 1))
            HUNG_COMBOS="${HUNG_COMBOS}\n  hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
            echo ""
            echo "!!! TIMEOUT (${TIMEOUT_SEC}s): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS — possible kernel hang !!!"
            echo ""
        elif [[ $EXIT_CODE -ne 0 ]]; then
            FAILED=$((FAILED + 1))
            FAILED_COMBOS="${FAILED_COMBOS}\n  hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS (exit $EXIT_CODE)"
            echo ""
            echo "!!! FAILED (exit $EXIT_CODE): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS !!!"
            echo ""
        elif [[ "$BAD_RANKS" -gt 0 ]]; then
            FAILED=$((FAILED + 1))
            FAILED_COMBOS="${FAILED_COMBOS}\n  hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS ($BAD_RANKS rank(s) reported errors)"
            echo ""
            echo "!!! INCORRECT: hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS — $BAD_RANKS rank(s) reported nonzero error times !!!"
            echo "!!! The saved (block_num, warp_per_block, rdma_block_num) for this shape does not produce correct output. !!!"
            echo ""
        else
            echo "[$(date)] Verified correct [$COMBO_IDX/$TOTAL_COMBOS]"
        fi

        # Always cleanup peer
        cleanup_peer
        sleep 2
    done
done

echo ""
echo "============================================================"
echo "Verify tuned config complete"
echo "  Total:    $TOTAL_COMBOS"
echo "  Failed:   $FAILED"
if [[ -n "$FAILED_COMBOS" ]]; then
    echo -e "  Incorrect/crashed:$FAILED_COMBOS"
fi
if [[ -n "$HUNG_COMBOS" ]]; then
    echo -e "  Hung:$HUNG_COMBOS"
fi
echo "  Log:      $LOG_FILE"
echo "============================================================"

[[ $FAILED -gt 0 ]] && exit 1
exit 0
