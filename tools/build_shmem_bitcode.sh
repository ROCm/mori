#!/bin/bash
# Build libmori_shmem_device.bc — the device bitcode library for mori shmem.
#
# This is the shell equivalent of Python's ensure_bitcode() in mori/jit/core.py.
# The bitcode contains all mori shmem device functions (extern "C" wrappers)
# and the globalGpuStates symbol. It can be linked into any GPU kernel
# (Triton, MLIR, or raw LLVM IR) to enable device-side RDMA operations.
#
# Usage:
#   bash tools/build_shmem_bitcode.sh [output_dir] [gpu_arch]
#
# Examples:
#   bash tools/build_shmem_bitcode.sh                    # auto-detect arch+NIC, output to lib/
#   bash tools/build_shmem_bitcode.sh lib/ gfx942        # explicit arch
#
# Output:
#   <output_dir>/libmori_shmem_device.bc   (default: lib/)

set -e

MORI_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
OUTPUT_DIR=${1:-${MORI_DIR}/lib}
GPU_ARCH=${2:-}

# ---------------------------------------------------------------------------
# Detect GPU architecture
# Priority: arg > MORI_GPU_ARCHS > local GPU > PYTORCH_ROCM_ARCH > default
# (matches setup.py _get_gpu_archs)
# ---------------------------------------------------------------------------
if [ -z "$GPU_ARCH" ]; then
    # 1. MORI_GPU_ARCHS env
    if [ -n "$MORI_GPU_ARCHS" ]; then
        GPU_ARCH="$MORI_GPU_ARCHS"
    fi

    # 2. Local GPU detection
    if [ -z "$GPU_ARCH" ] && [ -x "${ROCM_PATH}/bin/rocm_agent_enumerator" ]; then
        GPU_ARCH=$(${ROCM_PATH}/bin/rocm_agent_enumerator | grep -v "gfx000" | grep "gfx" | head -1)
    fi

    # 3. PYTORCH_ROCM_ARCH / GPU_ARCHS env
    if [ -z "$GPU_ARCH" ]; then
        _env_arch="${GPU_ARCHS:-$PYTORCH_ROCM_ARCH}"
        if [ -n "$_env_arch" ]; then
            GPU_ARCH=$(echo "$_env_arch" | tr ';' '\n' | grep "gfx" | head -1)
        fi
    fi

    # 4. Default
    if [ -z "$GPU_ARCH" ]; then
        echo "Warning: Could not detect GPU architecture, defaulting to gfx942" >&2
        GPU_ARCH="gfx942"
    fi
fi
echo "[mori] GPU architecture: $GPU_ARCH"

# ---------------------------------------------------------------------------
# Detect NIC type for device macros
# Priority: env > /sys/class/infiniband (prefix + driver symlink) > lspci > library > mlx5
# (matches setup.py _detect_nic_type and CMake detect_device_nic)
# ---------------------------------------------------------------------------
detect_nic_type() {
    # 1. Environment variable override
    if [ "${USE_BNXT:-}" = "ON" ]; then echo "bnxt"; return; fi
    if [ "${USE_IONIC:-}" = "ON" ]; then echo "ionic"; return; fi
    if [ "${MORI_DEVICE_NIC:-}" != "" ]; then echo "${MORI_DEVICE_NIC}"; return; fi

    # 2. /sys/class/infiniband/ — device name prefix + driver symlink
    local ib_dir="/sys/class/infiniband"
    if [ -d "$ib_dir" ]; then
        local bnxt=0 ionic=0 mlx5=0
        for dev in "$ib_dir"/*; do
            [ -e "$dev" ] || continue
            local name=$(basename "$dev")
            case "$name" in
                bnxt_re*) bnxt=$((bnxt + 1)) ;;
                ionic*)   ionic=$((ionic + 1)) ;;
                mlx5*)    mlx5=$((mlx5 + 1)) ;;
                *)
                    # Driver symlink fallback for generic names (rdma0, etc.)
                    local drv=$(readlink -f "$dev/device/driver" 2>/dev/null | xargs basename 2>/dev/null)
                    case "$drv" in
                        bnxt_re|bnxt_en) bnxt=$((bnxt + 1)) ;;
                        ionic_rdma|ionic) ionic=$((ionic + 1)) ;;
                        mlx5_core|mlx5_ib) mlx5=$((mlx5 + 1)) ;;
                    esac
                    ;;
            esac
        done
        if [ $bnxt -gt 0 ] && [ $bnxt -ge $mlx5 ]; then echo "bnxt"; return; fi
        if [ $ionic -gt 0 ] && [ $ionic -ge $mlx5 ]; then echo "ionic"; return; fi
        if [ $mlx5 -gt 0 ]; then echo "mlx5"; return; fi
    fi

    # 3. lspci PCI vendor ID
    if command -v lspci &>/dev/null; then
        local lspci_out=$(lspci -nn -d ::0200 2>/dev/null || true)
        if [ -n "$lspci_out" ]; then
            local pci_bnxt=$(echo "$lspci_out" | grep -c "14e4" || true)
            local pci_ionic=$(echo "$lspci_out" | grep -c "1dd8" || true)
            local pci_mlx5=$(echo "$lspci_out" | grep -c "15b3" || true)
            if [ $pci_bnxt -gt 0 ] && [ $pci_bnxt -ge $pci_mlx5 ]; then echo "bnxt"; return; fi
            if [ $pci_ionic -gt 0 ] && [ $pci_ionic -ge $pci_mlx5 ]; then echo "ionic"; return; fi
            if [ $pci_mlx5 -gt 0 ]; then echo "mlx5"; return; fi
        fi
    fi

    # 4. Library file fallback
    for dir in /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
        [ -f "$dir/libbnxt_re.so" ] && { echo "bnxt"; return; }
        [ -f "$dir/libionic.so" ] && { echo "ionic"; return; }
    done

    # 5. Default
    echo "mlx5"
}

NIC_TYPE=$(detect_nic_type)
NIC_DEFINES=""
case "$NIC_TYPE" in
    bnxt)  NIC_DEFINES="-DMORI_DEVICE_NIC_BNXT" ;;
    ionic) NIC_DEFINES="-DMORI_DEVICE_NIC_IONIC" ;;
esac
echo "[mori] NIC: ${NIC_TYPE^^}"

# ---------------------------------------------------------------------------
# Compile wrapper + shim to device bitcode
# ---------------------------------------------------------------------------
HIPCC="${ROCM_PATH}/bin/hipcc"
LLVM_LINK="${ROCM_PATH}/lib/llvm/bin/llvm-link"
OPT="${ROCM_PATH}/lib/llvm/bin/opt"

WRAPPER_SRC="${MORI_DIR}/src/shmem/shmem_device_api_wrapper.cpp"
if [ ! -f "$WRAPPER_SRC" ]; then
    echo "Error: Source file not found: $WRAPPER_SRC"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Generate globalGpuStates shim
cat > "$TEMP_DIR/globalGpuStates.hip" << 'EOF'
#include "mori/shmem/internal.hpp"

namespace mori {
namespace shmem {
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;
}
}
EOF

# Include directories (same as JIT's _collect_include_dirs)
INCLUDES="-I${MORI_DIR} -I${MORI_DIR}/include -I${MORI_DIR}/src"
[ -d "${MORI_DIR}/3rdparty/spdlog/include" ] && INCLUDES="$INCLUDES -I${MORI_DIR}/3rdparty/spdlog/include"
[ -d "${MORI_DIR}/3rdparty/msgpack-c/include" ] && INCLUDES="$INCLUDES -I${MORI_DIR}/3rdparty/msgpack-c/include"

# Find MPI include
MPI_INC=$(mpicc --showme:compile 2>/dev/null | grep -oP '(?<=-I)\S+' | head -1 || true)
[ -n "$MPI_INC" ] && INCLUDES="$INCLUDES -I${MPI_INC}"

COMMON_FLAGS="--cuda-device-only -emit-llvm --offload-arch=${GPU_ARCH} -fgpu-rdc -mcode-object-version=5 -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS ${NIC_DEFINES}"

echo "[mori] Compiling wrapper ..."
$HIPCC -c $COMMON_FLAGS $INCLUDES "$WRAPPER_SRC" -o "$TEMP_DIR/wrapper.bc"

echo "[mori] Compiling globalGpuStates shim ..."
$HIPCC -c $COMMON_FLAGS $INCLUDES "$TEMP_DIR/globalGpuStates.hip" -o "$TEMP_DIR/shim.bc"

echo "[mori] Linking ..."
$LLVM_LINK "$TEMP_DIR/wrapper.bc" "$TEMP_DIR/shim.bc" -o "$TEMP_DIR/linked.bc"

echo "[mori] Stripping llvm.lifetime intrinsics ..."
$OPT -S "$TEMP_DIR/linked.bc" -o "$TEMP_DIR/linked.ll"
sed -i '/llvm\.lifetime\./d' "$TEMP_DIR/linked.ll"
$OPT "$TEMP_DIR/linked.ll" -o "$TEMP_DIR/libmori_shmem_device.bc"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo "[mori] Verifying ..."
$OPT -S "$TEMP_DIR/libmori_shmem_device.bc" -o "$TEMP_DIR/verify.ll"
if grep -q "@_ZN4mori5shmem15globalGpuStatesE" "$TEMP_DIR/verify.ll"; then
    echo "[mori] ✓ globalGpuStates symbol found"
else
    echo "Error: globalGpuStates not found in bitcode"
    exit 1
fi

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
cp -f "$TEMP_DIR/libmori_shmem_device.bc" "$OUTPUT_DIR/"
echo "[mori] Output: $OUTPUT_DIR/libmori_shmem_device.bc"
echo "[mori] Done."
