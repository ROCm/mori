#!/usr/bin/env bash
# ==========================================================================
# EP InterNode Dispatch/Combine Performance Benchmark
#
# Sweeps over (token_count x dtype) combinations using bench or tuning mode,
# records raw output and a best-performance summary.
#
# Usage:
#   bash tools/bench_ep_internode_performance.sh \
#       --node0 <host0> --node1 <host1> --container <name>
#
#   bash tools/bench_ep_internode_performance.sh \
#       --node0 <host0> --node1 <host1> --container <name> \
#       --tokens "128,4096" --dtypes "bf16" --cmd tuning --tuning-scope full
#
# Output:
#   <output-dir>/
#     raw/           -- full output per (token, dtype) combo
#     summary.txt    -- best-performance summary table
#     bench.log      -- run log
# ==========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$REPO_ROOT/examples/ops/dispatch_combine/test_dispatch_combine_internode.py"

# ---- Defaults ----
NODE0=""
NODE1=""
CONTAINER=""
USER_NAME="${USER:-$(whoami)}"
SMALL_TOKENS="1,2,4,8,16,32,64,128,256,512,768"
LARGE_TOKENS="4096,8192,16384,32768,65536,131072,262144"
ALL_TOKENS="$SMALL_TOKENS,$LARGE_TOKENS"
TOKENS="$ALL_TOKENS"
DTYPES="fp8_e4m3,bf16"
CMD="tuning"
NUM_QP=2
OUTPUT_DIR=""
TIMEOUT=7200
BASE_PORT=29000
IFACE=""
GPU_PER_NODE=8
TUNING_SCOPE="quick"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)        NODE0="$2";         shift 2 ;;
        --node1)        NODE1="$2";         shift 2 ;;
        --container)    CONTAINER="$2";     shift 2 ;;
        --user)         USER_NAME="$2";     shift 2 ;;
        --tokens)       TOKENS="$2";        shift 2 ;;
        --dtypes)       DTYPES="$2";        shift 2 ;;
        --cmd)          CMD="$2";           shift 2 ;;
        --num-qp)       NUM_QP="$2";        shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --timeout)      TIMEOUT="$2";       shift 2 ;;
        --iface)        IFACE="$2";         shift 2 ;;
        --gpu-per-node) GPU_PER_NODE="$2";  shift 2 ;;
        --tuning-scope) TUNING_SCOPE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# ====/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$NODE0" || -z "$NODE1" || -z "$CONTAINER" ]]; then
    echo "ERROR: --node0, --node1, and --container are required."
    echo "  e.g.: bash $0 --node0 host0 --node1 host1 --container my-dev"
    exit 1
fi

# ---- SSH shorthand ----
_SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# ---- Resolve master address ----
MASTER_ADDR=$($_SSH "${USER_NAME}@${NODE1}" "getent hosts $NODE0 | awk '{print \$1}'" 2>/dev/null)
if [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR=$($_SSH "${USER_NAME}@${NODE0}" "hostname -I | awk '{print \$1}'" 2>/dev/null)
fi
[[ -z "$MASTER_ADDR" ]] && { echo "ERROR: Cannot resolve IP for $NODE0"; exit 1; }

# ---- Auto-detect network interface ----
if [[ -z "$IFACE" ]]; then
    IFACE=$($_SSH "${USER_NAME}@${NODE0}" "ip -o -4 addr show | grep '$MASTER_ADDR' | awk '{print \$2}'" 2>/dev/null)
    [[ -z "$IFACE" ]] && IFACE=$($_SSH "${USER_NAME}@${NODE0}" "ip -o -4 route show default | awk '{print \$5}'" 2>/dev/null)
fi

# ---- Output directory ----
WORLD_SIZE=$(( GPU_PER_NODE * 2 ))
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/bench_results/ep${WORLD_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR/raw"
LOG="$OUTPUT_DIR/bench.log"
SUMMARY="$OUTPUT_DIR/summary.txt"

# ---- SHMEM size mapping ----
get_shmem_size() {
    local t=$1
    if   (( t >= 262144 )); then echo "192G"
    elif (( t >= 131072 )); then echo "96G"
    elif (( t >= 65536  )); then echo "48G"
    elif (( t >= 16384  )); then echo "24G"
    else echo "6G"
    fi
}

get_kernel_type() {
    local t=$1
    if (( t >= 4096 )); then echo "v1"; else echo "v1_ll"; fi
}

# ---- GPU info ----
GPU_INFO=$($_SSH "${USER_NAME}@${NODE0}" \
    "docker exec $CONTAINER python3 -c 'import torch; p=torch.cuda.get_device_properties(0); print(p.name)'" 2>/dev/null || echo "unknown")

{
    echo "============================================================"
    echo "EP${WORLD_SIZE} InterNode Benchmark"
    echo "============================================================"
    echo "  GPU:         $GPU_INFO"
    echo "  world_size:  $WORLD_SIZE"
    echo "  cmd:         $CMD"
    echo "  tuning_scope:$TUNING_SCOPE"
    echo "  tokens:      $TOKENS"
    echo "  dtypes:      $DTYPES"
    echo "  num_qp:      $NUM_QP"
    echo "  output_dir:  $OUTPUT_DIR"
    echo "  started:     $(date)"
    echo "============================================================"
    echo ""
} | tee "$LOG"

# ---- Summary header ----
printf "%-10s %-12s %-10s %-10s %-12s %-12s %-12s %-12s %-20s\n" \
    "tokens" "dtype" "phase" "rdma_bw" "xgmi_bw" "ll_bw" "latency_us" "metric" "config" \
    > "$SUMMARY"

IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS"
IFS=',' read -ra DTYPE_ARRAY <<< "$DTYPES"
TOTAL=$(( ${#TOKEN_ARRAY[@]} * ${#DTYPE_ARRAY[@]} ))
IDX=0

# ---- run_one ----
run_one() {
    local ntokens=$1 kernel_type=$2 dtype=$3 shmem=$4 combine_dtype=$5 port=$6 tag=$7 tmo=$8 run_cmd=$9
    local raw0="$OUTPUT_DIR/raw/${tag}.txt"
    local raw1="$OUTPUT_DIR/raw/${tag}_node1.txt"

    local ENV="MORI_RDMA_SL=3 MORI_RDMA_TC=96 GPU_PER_NODE=$GPU_PER_NODE"
    ENV+=" GLOO_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE"
    ENV+=" HSA_NO_SCRATCH_RECLAIM=1 MORI_SHMEM_HEAP_SIZE=$shmem"
    ENV+=" MORI_TUNING_SCOPE=$TUNING_SCOPE"
    ENV+=" PYTHONPATH=$REPO_ROOT:$REPO_ROOT/python:\$PYTHONPATH"

    local EXTRA=""
    [[ -n "$combine_dtype" ]] && EXTRA+=" --combine-dtype $combine_dtype"
    if (( ntokens >= 65536 )); then
        EXTRA+=" --skip-verify"
    fi
    if [[ "$run_cmd" == "tuning" ]]; then
        EXTRA+=" --save-tuning-config auto"
    fi
    local TCMD="cd $REPO_ROOT && $ENV torchrun \
        --nnodes=2 --node_rank=RANK --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR --master_port=$port \
        $BENCH_SCRIPT \
        --kernel-type $kernel_type --max-tokens $ntokens \
        --cmd $run_cmd --num-qp $NUM_QP --dtype $dtype $EXTRA"

    local SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
    $SSH "${USER_NAME}@${NODE1}" "docker exec $CONTAINER bash -c '${TCMD//RANK/1}'" > "$raw1" 2>&1 &
    local pid1=$!
    sleep 2
    timeout "$tmo" $SSH "${USER_NAME}@${NODE0}" "docker exec $CONTAINER bash -c '${TCMD//RANK/0}'" > "$raw0" 2>&1
    local rc=$?

    if [[ $rc -eq 124 ]]; then
        $SSH "${USER_NAME}@${NODE0}" "docker exec $CONTAINER pkill -9 -f torchrun" 2>/dev/null || true
        $SSH "${USER_NAME}@${NODE1}" "docker exec $CONTAINER pkill -9 -f torchrun" 2>/dev/null || true
    fi
    wait $pid1 2>/dev/null || true

    sleep 3
    $SSH "${USER_NAME}@${NODE0}" "docker exec $CONTAINER bash -c 'pkill -9 -f torchrun; pkill -9 -f test_dispatch_combine'" 2>/dev/null || true
    $SSH "${USER_NAME}@${NODE1}" "docker exec $CONTAINER bash -c 'pkill -9 -f torchrun; pkill -9 -f test_dispatch_combine'" 2>/dev/null || true
    sleep 5
    return $rc
}

# ---- Main loop ----
for NTOKENS in "${TOKEN_ARRAY[@]}"; do
    KERNEL=$(get_kernel_type "$NTOKENS")
    SHMEM=$(get_shmem_size "$NTOKENS")

    for DTYPE in "${DTYPE_ARRAY[@]}"; do
        IDX=$((IDX + 1))
        TAG="${DTYPE}_${KERNEL}_${NTOKENS}"
        COMBINE_DTYPE=""
        BASE_PORT=$((BASE_PORT + 1))

        # Wait until GPUs are idle
        while true; do
            _clear=true
            for _n in "$NODE0" "$NODE1"; do
                _u=$(ssh -o StrictHostKeyChecking=no "${USER_NAME}@${_n}" \
                    'rocm-smi --showuse 2>/dev/null | grep "GPU use" | sed "s/.*: //" | sort -rn | head -1' 2>/dev/null)
                if [[ -n "$_u" && "$_u" != "0" ]]; then
                    _clear=false
                    echo "  [wait] GPU ${_u}% on $_n, retrying in 120s...  $(date)" | tee -a "$LOG"
                    break
                fi
            done
            $_clear && break
            sleep 120
        done

        echo "[$IDX/$TOTAL] tokens=$NTOKENS dtype=$DTYPE kernel=$KERNEL cmd=$CMD shmem=$SHMEM  $(date)" | tee -a "$LOG"

        set +e
        run_one "$NTOKENS" "$KERNEL" "$DTYPE" "$SHMEM" "$COMBINE_DTYPE" "$BASE_PORT" "$TAG" "$TIMEOUT" "$CMD"
        EXIT_CODE=$?
        set -e

        RAW_FILE="$OUTPUT_DIR/raw/${TAG}.txt"

        if [[ $EXIT_CODE -eq 124 ]]; then
            echo "  !! TIMEOUT (${TIMEOUT}s) !!" | tee -a "$LOG"
        elif [[ $EXIT_CODE -ne 0 ]]; then
            echo "  !! FAILED (exit $EXIT_CODE) !!" | tee -a "$LOG"
        else
            echo "  OK" | tee -a "$LOG"
        fi

        # Extract results
        grep -E "Performance Summary|Best |Dispatch Performance|Combine Performance|Best|Worst|Average" "$RAW_FILE" 2>/dev/null | tee -a "$LOG" || true

        python3 -c "
import re, sys
raw = open(sys.argv[1]).read()
tokens, dtype = sys.argv[2], sys.argv[3]
lines = raw.splitlines()

phase = None
cfg = ''
for line in lines:
    # Detect phase from PrettyTable title (handles both bench and tuning output)
    # Bench:  'Dispatch Performance (float8_e4m3fn)'
    # Tuning: 'Dispatch (float8_e4m3fn) block=96 warp=8 rdma=64'
    tl = re.search(r'(Dispatch|Combine)\b', line)
    if tl and ('Performance' in line or 'block=' in line):
        phase = tl.group(1).lower()
        cm = re.search(r'block=(\d+)\s+warp=(\d+)\s+rdma=(\d+)', line)
        cfg = f'bn={cm.group(1)}/w={cm.group(2)}/r={cm.group(3)}' if cm else ''
        continue
    # Parse PrettyTable data rows
    m = re.match(r'\|\s*(Best|Worst|Average)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', line)
    if m and phase:
        metric = m.group(1).lower()
        rdma, xgmi, ll, lat = m.group(2), m.group(3), m.group(4), m.group(5)
        print(f'{tokens:<10} {dtype:<12} {phase:<10} {rdma:<10} {xgmi:<12} {ll:<12} {lat:<12} {metric:<12} {cfg}')
" "$RAW_FILE" "$NTOKENS" "$DTYPE" >> "$SUMMARY" 2>/dev/null || true

        echo "" | tee -a "$LOG"
    done
done

echo "============================================================" | tee -a "$LOG"
echo "All $TOTAL benchmarks complete.  $(date)" | tee -a "$LOG"
echo "  Summary: $SUMMARY" | tee -a "$LOG"
echo "  Raw:     $OUTPUT_DIR/raw/" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

echo ""
echo "=== Performance Summary ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
