#!/usr/bin/env bash
set -uo pipefail

# Quantization / dtype coverage sweep for EP dispatch/combine.
#
# Sweeps the NIC device backend (MLX / BRCM / AINIC) across the quantized and
# cross-dtype dispatch/combine correctness tests, i.e. FP8 direct-cast and FP4
# dispatch including the cross-dtype pairings such as FP4 dispatch + BF16
# combine.
#
# The NIC backend is a compile-time axis: MORI_DEVICE_NIC selects the
# device-side IBGDA provider (-DMORI_DEVICE_NIC_BNXT / -DMORI_DEVICE_NIC_IONIC)
# and is part of the JIT cache key, so each value produces a distinct kernel
# build. Sweeping it therefore verifies that the quantized dispatch/combine
# kernels codegen and run correctly against every NIC's device primitives, even
# on a host that has only one NIC vendor installed (or none).
#
# What this does NOT cover: moving quantized payloads over a real RDMA link.
# That needs two nodes with the NIC actually present and is driven by
# tools/run_internode_test.sh / tools/run_all_internode_tuning.sh. Cells that
# require RDMA hardware are reported as such rather than silently passing.
#
# Usage:
#   bash tools/run_quant_dtype_coverage.sh                 # all available NICs
#   bash tools/run_quant_dtype_coverage.sh --nics mlx5,bnxt
#   bash tools/run_quant_dtype_coverage.sh --all-nics      # incl. NICs with no
#                                                          # host driver library
#   bash tools/run_quant_dtype_coverage.sh --pytest-args "-x -v"
#
# Options:
#   --nics LIST         Comma-separated subset of {mlx5,bnxt,ionic}
#   --all-nics          Sweep every backend even when its provider library is
#                       absent (compile-only validation; default skips those)
#   --pytest-args STR   Extra args forwarded to pytest
#   --keep-going        Continue after a failing group (default: continue)
#   --fail-fast         Stop at the first failing group

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-16G}"

ALL_NICS="mlx5 bnxt ionic"
NICS=""
FORCE_ALL_NICS=0
PYTEST_ARGS=""
FAIL_FAST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nics)        NICS="${2//,/ }"; shift 2 ;;
        --all-nics)    FORCE_ALL_NICS=1; shift ;;
        --pytest-args) PYTEST_ARGS="$2"; shift 2 ;;
        --keep-going)  FAIL_FAST=0; shift ;;
        --fail-fast)   FAIL_FAST=1; shift ;;
        -h|--help)     sed -n '3,37p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# A NIC's user-space provider library is required to actually open a device, but
# not to compile the device-side kernels (bnxt/ionic headers ship in-tree). So a
# missing library downgrades a backend to compile+intranode validation rather
# than excluding it.
nic_lib_present() {
    local lib
    case "$1" in
        mlx5)  lib="libmlx5.so" ;;
        bnxt)  lib="libbnxt_re.so" ;;
        ionic) lib="libionic.so" ;;
        *) return 1 ;;
    esac
    for d in /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
        [[ -e "$d/$lib" ]] && return 0
    done
    return 1
}

if [[ -z "$NICS" ]]; then
    NICS=""
    for nic in $ALL_NICS; do
        if [[ "$FORCE_ALL_NICS" == "1" ]] || nic_lib_present "$nic"; then
            NICS="$NICS $nic"
        else
            echo "note: skipping NIC '$nic' (no provider library on host); use --all-nics to include it"
        fi
    done
fi
NICS="$(echo "$NICS" | xargs)"

if [[ -z "$NICS" ]]; then
    echo "error: no NIC backends selected" >&2
    exit 2
fi

# RDMA devices present => the internode kernels can move data over a real link.
# Without them the internode kernel types still run, but single-node only.
RDMA_PRESENT=0
if [[ -d /sys/class/infiniband ]] && [[ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
    RDMA_PRESENT=1
fi

# Quantized + cross-dtype selections, one per kernel-type family.
declare -a GROUP_NAMES=(
    "IntraNode/IntraNodeLL"
    "InterNodeV1/InterNodeV1LL"
    "AsyncLL"
)
declare -a GROUP_FILES=(
    "tests/python/ops/test_dispatch_combine_intranode.py"
    "tests/python/ops/test_dispatch_combine_internode_v1.py"
    "tests/python/ops/test_dispatch_combine_async_ll.py"
)

mkdir -p "${REPO_ROOT}/logs"
LOGFILE="${REPO_ROOT}/logs/quant_dtype_coverage_$(date +%Y%m%d_%H%M%S).log"

ARCH=$(python3 -c "
import torch
try: print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])
except Exception: print('unknown')
" 2>/dev/null || echo unknown)

{
    echo "=== EP quantization / dtype coverage sweep ==="
    echo "Started at : $(date)"
    echo "Arch       : $ARCH"
    echo "NIC sweep  : $NICS"
    echo "RDMA links : $([[ $RDMA_PRESENT == 1 ]] && echo present || echo "absent (single-node only)")"
    echo "Log        : $LOGFILE"
    echo ""
} | tee "$LOGFILE"

declare -a RESULTS=()
OVERALL=0

for nic in $NICS; do
    for i in "${!GROUP_NAMES[@]}"; do
        name="${GROUP_NAMES[$i]}"
        file="${GROUP_FILES[$i]}"
        desc="nic=${nic} ${name}"
        {
            echo ""
            echo "################################################################"
            echo "# $desc"
            echo "# Started at: $(date)"
            echo "################################################################"
        } | tee -a "$LOGFILE"

        # shellcheck disable=SC2086
        MORI_DEVICE_NIC="$nic" python3 -m pytest "$file" \
            -k "cross_dtype" -q $PYTEST_ARGS 2>&1 | tee -a "$LOGFILE"
        rc=${PIPESTATUS[0]}

        summary=$(grep -Eo '[0-9]+ (passed|failed)[^$]*' "$LOGFILE" | tail -1)
        if [[ $rc -eq 0 ]]; then
            RESULTS+=("PASS  ${desc}  (${summary})")
        else
            RESULTS+=("FAIL  ${desc}  (rc=${rc})")
            OVERALL=1
            [[ "$FAIL_FAST" == "1" ]] && break 2
        fi
    done
done

{
    echo ""
    echo "================================================================"
    echo "Coverage summary"
    echo "================================================================"
    for r in "${RESULTS[@]}"; do echo "  $r"; done
    echo ""
    if [[ $RDMA_PRESENT == 0 ]]; then
        cat <<'EOF'
Not covered by this run (requires hardware):
  * Quantized payloads over a real RDMA link. No RDMA devices are present on
    this host, so the InterNode kernel types ran single-node and the NIC axis
    was validated at codegen + kernel-execution level only. For on-the-wire
    coverage run tools/run_all_internode_tuning.sh (or run_internode_test.sh)
    on a 2-node pair per NIC vendor.
EOF
    fi
    echo "Finished at: $(date)"
} | tee -a "$LOGFILE"

exit $OVERALL
