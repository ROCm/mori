#!/bin/bash
# Build libmori_shmem_device.bc — the device bitcode library for mori shmem.
#
# This bitcode contains all mori shmem device functions (extern "C" wrappers)
# and the globalGpuStates symbol. It can be linked into any GPU kernel
# (Triton, MLIR, or raw LLVM IR) to enable device-side RDMA operations.
#
# Prerequisites:
#   mori must be built with BUILD_SHMEM_DEVICE_WRAPPER=ON and SAVE_TEMPS=ON:
#     BUILD_SHMEM_DEVICE_WRAPPER=ON pip install . --no-build-isolation
#
# Usage:
#   bash tools/build_shmem_bitcode.sh [output_dir]
#
# Output:
#   <output_dir>/libmori_shmem_device.bc   (default: lib/)

set -e

MORI_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MORI_BUILD_DIR=${MORI_BUILD_DIR:-${MORI_DIR}/build}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
OUTPUT_DIR=${1:-${MORI_DIR}/lib}

# ---------------------------------------------------------------------------
# Detect GPU architecture
# ---------------------------------------------------------------------------
detect_gpu_arch() {
    local arch=""
    if [ -x "${ROCM_PATH}/bin/rocm_agent_enumerator" ]; then
        arch=$(${ROCM_PATH}/bin/rocm_agent_enumerator | grep -v "gfx000" | grep "gfx" | head -1)
    fi
    if [ -z "$arch" ] && command -v rocminfo &> /dev/null; then
        arch=$(rocminfo | grep -oP 'gfx\w+' | head -1)
    fi
    if [ -z "$arch" ] && [ -n "$AMDGPU_TARGETS" ]; then
        arch=$(echo "$AMDGPU_TARGETS" | tr ',' '\n' | grep "gfx" | head -1)
    fi
    if [ -z "$arch" ]; then
        echo "Warning: Could not detect GPU architecture, defaulting to gfx942" >&2
        arch="gfx942"
    fi
    echo "$arch"
}

GPU_ARCH=$(detect_gpu_arch)
echo "[mori] GPU architecture: $GPU_ARCH"

# ---------------------------------------------------------------------------
# Locate BC files from mori build
# ---------------------------------------------------------------------------
BC_DIR="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir"
WRAPPER_BC="${BC_DIR}/shmem_device_api_wrapper-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.bc"
INIT_BC="${BC_DIR}/init-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.bc"
MEMORY_BC="${BC_DIR}/memory-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.bc"

for f in "$WRAPPER_BC" "$INIT_BC" "$MEMORY_BC"; do
    if [ ! -f "$f" ]; then
        echo "Error: BC file not found: $f"
        echo ""
        echo "Make sure mori was built with:"
        echo "  BUILD_SHMEM_DEVICE_WRAPPER=ON pip install . --no-build-isolation"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Link into a single bitcode
# ---------------------------------------------------------------------------
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "[mori] Linking BC files (wrapper + init + memory) ..."
${ROCM_PATH}/lib/llvm/bin/llvm-link \
    "$WRAPPER_BC" "$INIT_BC" "$MEMORY_BC" \
    -o "$TEMP_DIR/libmori_shmem_device.bc"

# ---------------------------------------------------------------------------
# Verify globalGpuStates symbol
# ---------------------------------------------------------------------------
echo "[mori] Verifying ..."
${ROCM_PATH}/lib/llvm/bin/opt -S "$TEMP_DIR/libmori_shmem_device.bc" \
    -o "$TEMP_DIR/libmori_shmem_device.ll"

if grep -q "@_ZN4mori5shmem15globalGpuStatesE" "$TEMP_DIR/libmori_shmem_device.ll"; then
    echo "[mori] globalGpuStates found in linked BC"
else
    echo "Error: globalGpuStates not found in linked BC"
    exit 1
fi

if grep -q "weak.*globalGpuStates" "$TEMP_DIR/libmori_shmem_device.ll"; then
    echo "Warning: globalGpuStates has weak linkage"
fi

# ---------------------------------------------------------------------------
# Copy to output
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
cp -f "$TEMP_DIR/libmori_shmem_device.bc" "$OUTPUT_DIR/"
echo "[mori] Output: $OUTPUT_DIR/libmori_shmem_device.bc"
echo "[mori] Done."
