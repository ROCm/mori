#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Batch Internode EP Dispatch+Combine Tuning Script
#
# Runs on the DRIVER node (rank 0). Launches the peer node (rank 1) via SSH.
# Sweeps over (max_tokens x hidden_dim) combinations, saving tuning results
# into a single JSON config file.
#
# Prerequisites:
#   - Passwordless SSH from driver to peer node
#   - Same repo path on both nodes (or set --remote-repo-root)
#   - Network interface accessible on both nodes (--ifname)
#
# Usage:
#   bash tools/batch_internode_tuning.sh \
#       --master-addr skyriver07 --peer-host skyriver08 --ifname bond0 \
#       --kernel-type v1
#
#   bash tools/batch_internode_tuning.sh \
#       --master-addr skyriver07 --peer-host skyriver08 --ifname bond0 \
#       --kernel-type v1_ll --dtype bf16 \
#       --tokens-list "128,512,4096" --hidden-dims "2048,7168"
#
#   bash tools/batch_internode_tuning.sh \
#       --master-addr skyriver07 --peer-host skyriver08 --ifname bond0 \
#       --kernel-type async_ll --num-qp 2 \
#       --config-output /path/to/out.json
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

# Typical tuning matrix for EP16 (2 nodes x 8 GPUs, run each separately):
#
#   fp4 dispatch + bf16 combine + fp8 quant:
#     bash tools/batch_internode_tuning.sh --master-addr <host0> --peer-host <host1> --ifname bond0 \
#         --kernel-type v1 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast
#
#   fp8 dispatch + bf16 combine:
#     bash tools/batch_internode_tuning.sh --master-addr <host0> --peer-host <host1> --ifname bond0 \
#         --kernel-type v1 --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master-addr)       MASTER_ADDR="$2";       shift 2 ;;
        --master-port)       MASTER_PORT="$2";       shift 2 ;;
        --peer-host)         PEER_HOST="$2";         shift 2 ;;
        --ifname)            IFNAME="$2";            shift 2 ;;
        --remote-repo-root)  REMOTE_REPO_ROOT="$2";  shift 2 ;;
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
    [[ -z "${!var}" ]] && { echo "Error: --${var,,} is required"; exit 1; }
done

# ---- Remote repo root defaults to local repo root ----
if [[ -z "$REMOTE_REPO_ROOT" ]]; then
    REMOTE_REPO_ROOT="$REPO_ROOT"
fi

# ---- Convert comma-separated lists to arrays ----
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS_LIST"
IFS=',' read -ra HIDDEN_DIM_ARRAY <<< "$HIDDEN_DIMS"

# ---- Build log filename ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COMB_TAG="${COMBINE_DTYPE:-$DTYPE}"
QUANT_TAG=""
if [[ "$QUANT_TYPE" != "none" ]]; then
    QUANT_TAG="_${QUANT_TYPE}"
fi
LOG_FILE="${LOG_DIR}/batch_internode_${KERNEL_TYPE}_ep$((GPU_PER_NODE*2))_${DTYPE}_${COMB_TAG}${QUANT_TAG}_${TIMESTAMP}.log"

TOTAL_COMBOS=$(( ${#HIDDEN_DIM_ARRAY[@]} * ${#TOKEN_ARRAY[@]} ))

# ---- Print summary ----
echo "============================================================"
echo "Batch Internode Tuning"
echo "============================================================"
echo "  master_addr:         $MASTER_ADDR"
echo "  peer_host:           $PEER_HOST"
echo "  ifname:              $IFNAME"
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
echo "Verifying SSH to $PEER_HOST ..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER_HOST" "echo ok" &>/dev/null; then
    echo "Error: Cannot SSH to $PEER_HOST (passwordless SSH required)"
    exit 1
fi
echo "SSH OK"
echo ""

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
    if [[ -n "$COMBINE_DTYPE" ]]; then
        ARGS+=(--combine-dtype "$COMBINE_DTYPE")
    fi
    if [[ -n "$SAVE_ARG" ]]; then
        ARGS+=(--save-tuning-config "$SAVE_ARG")
    fi
    echo "${ARGS[@]}"
}

# ---- Build torchrun command for one rank ----
build_torchrun_cmd() {
    local NODE_RANK="$1"
    local THE_REPO_ROOT="$2"
    local PY_ARGS="$3"

    echo "cd $THE_REPO_ROOT && " \
         "GPU_PER_NODE=$GPU_PER_NODE" \
         "GLOO_SOCKET_IFNAME=$IFNAME" \
         "MORI_SOCKET_IFNAME=$IFNAME" \
         "PYTHONPATH=${THE_REPO_ROOT}/python:\${PYTHONPATH:-}" \
         "torchrun" \
         "--nnodes=2" \
         "--node_rank=$NODE_RANK" \
         "--nproc_per_node=1" \
         "--master_addr=$MASTER_ADDR" \
         "--master_port=$MASTER_PORT" \
         "$INTERNODE_SCRIPT" \
         "$PY_ARGS"
}

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

        REPRO_RANK0="HSA_NO_SCRATCH_RECLAIM=1 $CMD_RANK0"
        REPRO_RANK1="HSA_NO_SCRATCH_RECLAIM=1 $CMD_RANK1"

        # Launch peer (rank 1) via SSH in background
        ssh "$PEER_HOST" "bash -lc '$CMD_RANK1'" &
        PEER_PID=$!

        # Launch local (rank 0) with timeout
        timeout "$TIMEOUT_SEC" bash -c "$CMD_RANK0"
        EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 124 ]]; then
            FAILED=$((FAILED + 1))
            HUNG_COMBOS="${HUNG_COMBOS}\n  hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
            echo ""
            echo "!!! TIMEOUT (${TIMEOUT_SEC}s): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS — possible kernel hang !!!"
            echo "!!! Reproduce rank 0: $REPRO_RANK0 !!!"
            echo "!!! Reproduce rank 1 (on $PEER_HOST): $REPRO_RANK1 !!!"
            echo ""
        elif [[ $EXIT_CODE -ne 0 ]]; then
            FAILED=$((FAILED + 1))
            echo ""
            echo "!!! FAILED (exit $EXIT_CODE): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS !!!"
            echo "!!! Reproduce rank 0: $REPRO_RANK0 !!!"
            echo "!!! Reproduce rank 1 (on $PEER_HOST): $REPRO_RANK1 !!!"
            echo ""
        else
            echo "[$(date)] Completed [$COMBO_IDX/$TOTAL_COMBOS]"
        fi

        # Clean up peer: kill SSH process, then remote torchrun if still alive
        kill "$PEER_PID" 2>/dev/null || true
        wait "$PEER_PID" 2>/dev/null || true
        if [[ $EXIT_CODE -ne 0 ]]; then
            ssh "$PEER_HOST" "pkill -f 'torchrun.*test_dispatch_combine_internode'" 2>/dev/null || true
        fi

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
