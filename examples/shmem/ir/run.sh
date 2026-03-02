#!/bin/bash
# Run the shmem kernel POC (LLVM IR + mori, no Triton).
#
# Prerequisites:
#   1. mori installed: cd 3rdparty/mori && BUILD_SHMEM_DEVICE_WRAPPER=ON pip install . --no-build-isolation
#   2. libmori_shmem_device.bc built: bash scripts/build_mori_shmem.sh
#
# Usage:
#   bash run.sh [nproc] [chip]
#
# Examples:
#   bash run.sh 2 gfx942
#   bash run.sh 8

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NPROC=${1:-2}
CHIP=${2:-"gfx942"}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

echo "=============================================="
echo "  Shmem Kernel POC (LLVM IR + mori, no Triton)"
echo "=============================================="
echo "  PEs:    ${NPROC}"
echo "  Chip:   ${CHIP}"
echo "=============================================="

export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-"256M"}

echo ""
echo "[Step 1] Compile-only test ..."
python3 "${SCRIPT_DIR}/mlir_shmem_kernel.py" "${CHIP}"

echo ""
echo "[Step 2] Multi-GPU test (${NPROC} PEs) ..."
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29501 \
    "${SCRIPT_DIR}/test_mlir_shmem.py" "${CHIP}"

echo ""
echo "=============================================="
echo "  All tests passed!"
echo "=============================================="
