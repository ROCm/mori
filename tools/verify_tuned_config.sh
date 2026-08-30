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
# --timeout is PER COMBO and defaults to 600s, which suits small shapes only:
# each combo runs 500 rounds and run_test_once compares output with per-token
# Python loops, so cost grows with max_tokens (measured: ~40s for 500 rounds at
# 4 tokens). The default --tokens-list reaches 4096 and needs far more; a run
# that is merely slow is reported as "possible kernel hang", so raise --timeout
# rather than trusting that message.
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
#   # With docker on both nodes, driver staying on the host. --local-docker
#   # reaches rank 0's container with `docker exec` the same way --docker
#   # reaches the peer's, so SSH runs from the host and its keys never have to
#   # be copied into a container.
#   bash tools/verify_tuned_config.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --local-docker <CONTAINER0> --docker <CONTAINER1> \
#       --kernel-type v1_ll --num-qp 2 --tokens-list "4,8,16" --hidden-dims "6144"
#
#   # Driver already running inside a container (omit --local-docker); the
#   # container then needs its own SSH access to the peer.
#   bash tools/verify_tuned_config.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       --docker <CONTAINER> --ssh-key <SSH_KEY_PATH> \
#       --kernel-type v1_ll --num-qp 2 --tokens-list "4,8,16" --hidden-dims "6144"
#
# Exit status: non-zero if any shape crashes, times out, or reports a rank
# error -- the same signal batch_internode_tuning.sh uses for its own per-combo
# failures. A geometry that produces wrong output shows up as a crash: the
# comparisons in run_test_once are bare `assert`s, so a mismatch raises rather
# than being counted.
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
        --local-docker)      LOCAL_DOCKER="$2";      shift 2 ;;
        --ssh-key)            SSH_KEY="$2";          shift 2 ;;
        --rdma-sl)            RDMA_SL="$2";           shift 2 ;;
        --rdma-tc)            RDMA_TC="$2";           shift 2 ;;
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
    if [[ -z "${!var}" ]]; then
        # ${var,,} lowercases but leaves the underscores, which would name a
        # flag that does not exist (--master_addr for --master-addr).
        flag="--${var,,}"
        echo "Error: ${flag//_/-} is required"
        exit 1
    fi
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
echo "  docker (peer):       ${DOCKER:-<none, direct execution>}"
echo "  docker (local):      ${LOCAL_DOCKER:-<none, direct execution>}"
echo "  ssh_key:             ${SSH_KEY:-<default>}"
echo "  rdma_sl:             ${RDMA_SL:-<not set>}"
echo "  rdma_tc:             ${RDMA_TC:-<not set>}"
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
echo "-- the preflight below reports which phases actually have a matching"
echo "rule; a phase without one runs on the built-in geometry and is NOT"
echo "verified by this run."
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
# script can reach the other node. Running on the host instead uses the SSH
# credentials that are already there and keeps them out of the container.
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

# ---- Preflight: is there anything on disk for these shapes to verify? ----
#
# AUTO mode fails open. If the JSON for this gpu/kernel/ep is missing,
# unparseable, or simply has no rule matching the dtype the op queries with,
# dispatch_combine.py falls back to its built-in geometry, logs at DEBUG (this
# library's logger sits at WARNING), and the run passes -- so without this
# check the script would report "Verified correct" having verified nothing.
#
# Only a necessary condition is checked -- that rules exist for the queried
# dtype -- rather than replaying lookup()'s full ceiling/topk matching, so this
# cannot drift out of step with it and silently start passing.
PREFLIGHT_PY="$LOG_DIR/.verify_preflight.py"
cat > "$PREFLIGHT_PY" <<'PYEOF'
import sys
from mori.ops.tuning_config import (
    TuningConfigManager, kernel_type_to_config_str, detect_gpu_model,
)
from mori.jit.config import detect_gpu_arch
import mori

kt_arg, ep_size = sys.argv[1], int(sys.argv[2])
# Each phase is looked up with the dtype of ITS OWN input tensor: the call
# sites pass dtype=input.dtype to _resolve_launch_params, and combine's input
# has already been converted to the combine dtype. Checking combine against the
# dispatch dtype would report a mismatch that the runtime does not have.
queried = {"dispatch": sys.argv[3], "combine": sys.argv[4] or sys.argv[3]}
kt = {
    "v0": mori.ops.EpDispatchCombineKernelType.IntraNode,
    "v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
    "v1_ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
    "async_ll": mori.ops.EpDispatchCombineKernelType.AsyncLL,
}[kt_arg]
kt_str = kernel_type_to_config_str(kt)
arch, model = detect_gpu_arch(), detect_gpu_model()
mgr = TuningConfigManager.get_instance(arch, kt_str, ep_size, model)
print("  config file:     %s_%s_%s_ep%d" % (arch, model, kt_str, ep_size))
ok = True
for phase, rules in (("dispatch", mgr.dispatch_rules), ("combine", mgr.combine_rules)):
    rules = rules or []
    dtypes = sorted({r["dtype"] for r in rules})
    dt = queried[phase]
    hit = dt in dtypes
    print("  %-8s rules:  %d (dtypes: %s) queried with %s -> %s"
          % (phase, len(rules), ",".join(dtypes) or "-", dt,
             "MATCH" if hit else "NO MATCH"))
    if not hit:
        ok = False
        print("  !! %s has no rule for the dtype it is looked up with, so it "
              "will run" % phase)
        print("     on the built-in geometry and this run does NOT verify its "
              "saved config.")
sys.exit(0 if ok else 3)
PYEOF

echo "Resolving which saved rules this run will actually use ..."
set +e
if [[ -n "$LOCAL_DOCKER" ]]; then
    $LOCAL_DOCKER_EXEC -w "$REPO_ROOT" "$LOCAL_DOCKER" \
        python3 "$PREFLIGHT_PY" "$KERNEL_TYPE" "$((GPU_PER_NODE * 2))" "$DTYPE" "$COMBINE_DTYPE"
else
    python3 "$PREFLIGHT_PY" "$KERNEL_TYPE" "$((GPU_PER_NODE * 2))" "$DTYPE" "$COMBINE_DTYPE"
fi
PREFLIGHT_RC=$?
set -e
rm -f "$PREFLIGHT_PY"

if [[ $PREFLIGHT_RC -eq 3 ]]; then
    PREFLIGHT_INCOMPLETE=1
    echo ""
    echo "WARNING: at least one phase will not be verified (see above)."
elif [[ $PREFLIGHT_RC -ne 0 ]]; then
    echo "Error: could not resolve the tuning config (exit $PREFLIGHT_RC)."
    echo "       Refusing to report a verification that would prove nothing."
    exit 1
else
    PREFLIGHT_INCOMPLETE=0
fi
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
        # Unquoted $LOCAL_DOCKER_EXEC on purpose: it is "docker exec" or
        # "sudo -n docker exec" and has to split into separate words.
        if [[ -n "$LOCAL_DOCKER" ]]; then
            LOCAL_RUN=($LOCAL_DOCKER_EXEC -w "$REPO_ROOT" "$LOCAL_DOCKER" bash -c "$CMD_RANK0")
        else
            LOCAL_RUN=(bash -c "$CMD_RANK0")
        fi
        timeout "$TIMEOUT_SEC" "${LOCAL_RUN[@]}" 2>&1 | tee "$COMBO_LOG" | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        # Patterns are anchored on this script's name, not a bare "torchrun".
        # Without --local-docker the driver runs on the host, where pkill sees
        # every process the user owns on that node, so a bare pattern would
        # take out an unrelated concurrent job. The bracket form additionally
        # stops the kill chain matching its own command line -- the first
        # pkill would otherwise kill the shell running it, leaving the rest
        # unexecuted and the python workers alive.
        #
        # A wrong result surfaces as a NON-ZERO EXIT, not as a count: every
        # comparison in run_test_once is a bare `assert`, so the first mismatch
        # raises and torchrun propagates the failure. The branch below is the
        # one that catches an incorrect config.
        #
        # The "error times: N" line each rank prints is NOT that signal and
        # cannot currently fire: its counter, error_round, is passed into
        # run_test_once but never added to anywhere in the file, so N is always
        # 0. Grepping it anyway costs nothing and starts working if that
        # counter is ever populated upstream -- but it must not be mistaken for
        # the correctness check, which is the exit code.
        BAD_RANKS=$(grep -cE "error times: +[1-9]" "$COMBO_LOG" || true)
        rm -f "$COMBO_LOG"

        if [[ $EXIT_CODE -eq 124 ]]; then
            FAILED=$((FAILED + 1))
            HUNG_COMBOS="${HUNG_COMBOS}\n  hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
            echo ""
            echo "!!! TIMEOUT (${TIMEOUT_SEC}s): hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS !!!"
            echo "!!! Either a kernel hang, or just too little time: 500 rounds at this"
            echo "!!! token count may legitimately need more than ${TIMEOUT_SEC}s. Re-run"
            echo "!!! this shape with a larger --timeout before calling it a hang."
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

        # Always cleanup both ranks. `timeout` kills the local `docker exec`
        # client, not the torchrun it started inside the container, so rank 0
        # needs an explicit kill too or a hung combo poisons the next one.
        kill_local 'pkill -9 -f "[t]orchrun.*test_dispatch_combine_internode"; pkill -9 -f "[t]est_dispatch_combine_internode"'
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
if [[ "$PREFLIGHT_INCOMPLETE" -eq 1 ]]; then
    echo ""
    echo "  NOTE: not every phase was covered -- see the preflight warning at"
    echo "        the top of this run. A phase with no matching rule ran on the"
    echo "        built-in geometry, so a pass says nothing about its saved"
    echo "        config."
fi
echo "  Scope:    ranks 0-$((GPU_PER_NODE - 1)) only. The peer node's output is"
echo "            not captured and its exit status is not collected, so a"
echo "            peer-side failure surfaces here only as a hang or as rank 0"
echo "            failing in a collective."
echo "============================================================"

[[ $FAILED -gt 0 ]] && exit 1
exit 0
