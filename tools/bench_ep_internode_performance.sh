#!/usr/bin/env bash
# ==========================================================================
# EP InterNode Dispatch/Combine Performance Benchmark (dual-node, 16 GPUs)
#
# Sweeps over (token_count x dtype) combinations and records raw output
# plus a best-performance summary.
#
# Prerequisites:
#   - Two nodes allocated via salloc, each with 8 GPUs
#   - Docker container "yutong-dev" running on both nodes (see EP16 setup)
#   - mori compiled inside both containers
#
# Usage (run from login node, NOT inside docker):
#   bash tools/bench_ep_internode_performance.sh \
#       --node0 smci355-ccs-aus-n08-33 --node1 smci355-ccs-aus-n09-33
#
#   # Custom options:
#   bash tools/bench_ep_internode_performance.sh \
#       --node0 <host0> --node1 <host1> \
#       --tokens "128,4096"   \
#       --dtypes "bf16"       \
#       --container yutong-dev \
#       --output-dir /tmp/ep16_bench
#
# Output directory layout:
#   <output-dir>/
#     raw/                              -- full output per combo
#       bf16_v1_ll_128.txt              -- node0 (rank0) raw output
#       bf16_v1_ll_128_node1.txt        -- node1 raw output
#       ...
#     summary.txt                       -- tabular best-perf summary
#     bench.log                         -- run log with progress
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$REPO_ROOT/examples/ops/dispatch_combine/test_dispatch_combine_internode.py"

# ---- Defaults ----
NODE0=""
NODE1=""
DOMAIN=".prov.aus.ccs.cpe.ice.amd.com"
CONTAINER="yutong-dev"
SMALL_TOKENS="1,2,4,8,16,32,64,128,256,512,768"
LARGE_TOKENS="4096,8192,16384,32768,65536,131072,262144,524288"
ALL_TOKENS="$SMALL_TOKENS,$LARGE_TOKENS"
TOKENS="$ALL_TOKENS"
DTYPES="fp8_e4m3,bf16"
NUM_QP=2
OUTPUT_DIR=""
TIMEOUT=1800
BASE_PORT=29000
IFACE="enp81s0f1"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)        NODE0="$2";         shift 2 ;;
        --node1)        NODE1="$2";         shift 2 ;;
        --domain)       DOMAIN="$2";        shift 2 ;;
        --container)    CONTAINER="$2";     shift 2 ;;
        --tokens)       TOKENS="$2";        shift 2 ;;
        --dtypes)       DTYPES="$2";        shift 2 ;;
        --num-qp)       NUM_QP="$2";        shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --timeout)      TIMEOUT="$2";       shift 2 ;;
        --iface)        IFACE="$2";         shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$NODE0" || -z "$NODE1" ]]; then
    echo "ERROR: --node0 and --node1 are required."
    echo "  e.g.: bash $0 --node0 smci355-ccs-aus-n08-33 --node1 smci355-ccs-aus-n09-33"
    exit 1
fi

NODE0_FQDN="${NODE0}${DOMAIN}"
NODE1_FQDN="${NODE1}${DOMAIN}"

# Resolve master address (node0 IP, resolvable from node1)
MASTER_ADDR=$(ssh -o StrictHostKeyChecking=no "yutongwu@${NODE1_FQDN}" \
    "getent hosts ${NODE0_FQDN} | awk '{print \$1}'" 2>/dev/null)
if [[ -z "$MASTER_ADDR" ]]; then
    echo "ERROR: Cannot resolve $NODE0_FQDN from $NODE1_FQDN"
    exit 1
fi

# ---- Output directory ----
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/bench_results/ep16_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR/raw"

LOG="$OUTPUT_DIR/bench.log"
SUMMARY="$OUTPUT_DIR/summary.txt"

# ---- SHMEM size mapping (matches bench_ep_performance.sh) ----
get_shmem_size() {
    local t=$1
    # shmem heap ≈ 3.4 × max_tokens × hidden(7168) × dtype(2) × world_size(16)
    # EP16 limit: 262144 tokens → 192G heap (MI355X 288G VRAM)
    # 524288 tokens would need ~460G → exceeds 288G, physically impossible
    if   (( t >= 262144 )); then echo "192G"
    elif (( t >= 131072 )); then echo "96G"
    elif (( t >= 65536  )); then echo "48G"
    elif (( t >= 16384  )); then echo "24G"
    else echo "6G"
    fi
}

# ---- Kernel type: v1_ll for small tokens, v1 for large ----
get_kernel_type() {
    local t=$1
    if (( t >= 4096 )); then echo "v1"
    else echo "v1_ll"
    fi
}

# ---- GPU info ----
GPU_INFO=$(ssh -o StrictHostKeyChecking=no "yutongwu@${NODE0_FQDN}" \
    "docker exec $CONTAINER python3 -c \"
import torch
p = torch.cuda.get_device_properties(0)
print(f'{p.name} (CU={p.multi_processor_count})')
\"" 2>/dev/null)

{
    echo "============================================================"
    echo "EP16 InterNode Benchmark (dual-node, 16 GPUs)"
    echo "============================================================"
    echo "  GPU:           $GPU_INFO"
    echo "  node0:         $NODE0_FQDN"
    echo "  node1:         $NODE1_FQDN"
    echo "  master_addr:   $MASTER_ADDR"
    echo "  container:     $CONTAINER"
    echo "  tokens:        $TOKENS"
    echo "  dtypes:        $DTYPES"
    echo "  num_qp:        $NUM_QP"
    echo "  iface:         $IFACE"
    echo "  output_dir:    $OUTPUT_DIR"
    echo "  started:       $(date)"
    echo "============================================================"
    echo ""
} | tee "$LOG"

# ---- Summary header ----
printf "%-10s %-12s %-8s %-10s %-12s %-12s %-12s %-12s\n" \
    "tokens" "dtype" "phase" "rdma_bw" "xgmi_bw" "ll_bw" "latency_us" "metric" \
    > "$SUMMARY"

# ---- Build arrays ----
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS"
IFS=',' read -ra DTYPE_ARRAY <<< "$DTYPES"

TOTAL=$(( ${#TOKEN_ARRAY[@]} * ${#DTYPE_ARRAY[@]} ))
IDX=0

# ---- run_one: launch torchrun on both nodes, capture output ----
run_one() {
    local ntokens=$1 kernel_type=$2 dtype=$3 shmem=$4 combine_dtype=$5 port=$6 tag=$7 tmo=$8
    local raw_node0="$OUTPUT_DIR/raw/${tag}.txt"
    local raw_node1="$OUTPUT_DIR/raw/${tag}_node1.txt"

    local ENV_PREFIX="MORI_RDMA_SL=3 MORI_RDMA_TC=96 GPU_PER_NODE=8"
    ENV_PREFIX+=" GLOO_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE"
    ENV_PREFIX+=" HSA_NO_SCRATCH_RECLAIM=1 MORI_SHMEM_HEAP_SIZE=$shmem"
    ENV_PREFIX+=" PYTHONPATH=$REPO_ROOT:$REPO_ROOT/python:\$PYTHONPATH"

    local COMBINE_ARG=""
    [[ -n "$combine_dtype" ]] && COMBINE_ARG="--combine-dtype $combine_dtype"

    local TORCHRUN_CMD="cd $REPO_ROOT && $ENV_PREFIX torchrun \
        --nnodes=2 --node_rank=RANK --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR --master_port=$port \
        $BENCH_SCRIPT \
        --kernel-type $kernel_type --max-tokens $ntokens \
        --cmd bench --num-qp $NUM_QP --dtype $dtype $COMBINE_ARG"

    # node1 in background
    ssh -o StrictHostKeyChecking=no "yutongwu@${NODE1_FQDN}" \
        "docker exec $CONTAINER bash -c '${TORCHRUN_CMD//RANK/1}'" \
        > "$raw_node1" 2>&1 &
    local pid1=$!

    sleep 2

    # node0 in foreground with timeout
    timeout "$tmo" ssh -o StrictHostKeyChecking=no "yutongwu@${NODE0_FQDN}" \
        "docker exec $CONTAINER bash -c '${TORCHRUN_CMD//RANK/0}'" \
        > "$raw_node0" 2>&1
    local rc=$?

    if [[ $rc -eq 124 ]]; then
        ssh -o StrictHostKeyChecking=no "yutongwu@${NODE0_FQDN}" \
            "docker exec $CONTAINER pkill -f torchrun" 2>/dev/null || true
        ssh -o StrictHostKeyChecking=no "yutongwu@${NODE1_FQDN}" \
            "docker exec $CONTAINER pkill -f torchrun" 2>/dev/null || true
    fi
    wait $pid1 2>/dev/null || true

    # Ensure all GPU processes are cleaned up before next test
    sleep 3
    ssh -o StrictHostKeyChecking=no "yutongwu@${NODE0_FQDN}" \
        "docker exec $CONTAINER pkill -9 -f torchrun 2>/dev/null; \
         docker exec $CONTAINER pkill -9 -f 'test_dispatch_combine\|bench_internode' 2>/dev/null" 2>/dev/null || true
    ssh -o StrictHostKeyChecking=no "yutongwu@${NODE1_FQDN}" \
        "docker exec $CONTAINER pkill -9 -f torchrun 2>/dev/null; \
         docker exec $CONTAINER pkill -9 -f 'test_dispatch_combine\|bench_internode' 2>/dev/null" 2>/dev/null || true
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

        # FP8 dispatch + BF16 combine for cross-type
        COMBINE_DTYPE=""
        [[ "$DTYPE" == "fp8_e4m3" ]] && COMBINE_DTYPE="bf16"

        BASE_PORT=$((BASE_PORT + 1))

        # Wait until GPUs are idle on both nodes before each test
        while true; do
            _all_clear=true
            for _node_fqdn in "$NODE0_FQDN" "$NODE1_FQDN"; do
                _max_use=$(ssh -o StrictHostKeyChecking=no "yutongwu@${_node_fqdn}" \
                    'rocm-smi --showuse 2>/dev/null | grep "GPU use" | sed "s/.*: //" | sort -rn | head -1' 2>/dev/null)
                if [[ -n "$_max_use" && "$_max_use" != "0" ]]; then
                    _all_clear=false
                    echo "  [wait] GPU max ${_max_use}% on $_node_fqdn, retrying in 120s...  $(date)" | tee -a "$LOG"
                    break
                fi
            done
            $_all_clear && break
            sleep 120
        done

        echo "[$IDX/$TOTAL] tokens=$NTOKENS dtype=$DTYPE kernel=$KERNEL shmem=$SHMEM  $(date)" | tee -a "$LOG"

        set +e
        run_one "$NTOKENS" "$KERNEL" "$DTYPE" "$SHMEM" "$COMBINE_DTYPE" "$BASE_PORT" "$TAG" "$TIMEOUT"
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

        # Extract PrettyTable results and append to log + summary
        grep -E "Dispatch Performance|Combine Performance|Best|Worst|Average" "$RAW_FILE" 2>/dev/null | tee -a "$LOG" || true

        # Parse PrettyTable into structured summary
        python3 -c "
import re, sys
raw = open(sys.argv[1]).read()
tokens, dtype = sys.argv[2], sys.argv[3]
phase = None
for line in raw.splitlines():
    if 'Dispatch Performance' in line:
        phase = 'dispatch'
    elif 'Combine Performance' in line:
        phase = 'combine'
    m = re.match(r'\|\s*(Best|Worst|Average)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', line)
    if m and phase:
        metric = m.group(1).lower()
        rdma, xgmi, ll, lat = m.group(2), m.group(3), m.group(4), m.group(5)
        print(f'{tokens:<10} {dtype:<12} {phase:<8} {rdma:<10} {xgmi:<12} {ll:<12} {lat:<12} {metric}')
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
