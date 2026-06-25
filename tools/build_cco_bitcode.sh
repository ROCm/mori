#!/bin/bash
# Build libmori_cco_device.bc — the device bitcode library for cco GDA.
#
# Contains the extern "C" device wrappers in src/cco/device/cco_device_wrapper.cpp
# (cco_gda_put / cco_gda_signal / cco_devcomm_rank / ...). Linked into any GPU
# kernel (FlyDSL / raw LLVM IR) to call the cco GDA device API.
#
# Unlike shmem, cco has no global singleton state — no shim, no module_init.
#
# Usage:
#   bash tools/build_cco_bitcode.sh [output_dir] [gpu_arch] [cov]
#
# Examples:
#   bash tools/build_cco_bitcode.sh                       # auto arch+NIC, cov=6, output to lib/
#   bash tools/build_cco_bitcode.sh lib/ gfx942 6
#
# Output:
#   <output_dir>/libmori_cco_device.bc   (default: lib/)

set -e

MORI_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
OUTPUT_DIR=${1:-${MORI_DIR}/lib}
GPU_ARCH=${2:-}
COV=${3:-6}  # FlyDSL ROCm backend uses ABI 600 -> code object version 6

# ---------------------------------------------------------------------------
# Detect GPU architecture (mirrors build_shmem_bitcode.sh)
# ---------------------------------------------------------------------------
if [ -z "$GPU_ARCH" ]; then
    if [ -n "$MORI_GPU_ARCHS" ]; then GPU_ARCH="$MORI_GPU_ARCHS"; fi
    if [ -z "$GPU_ARCH" ] && [ -x "${ROCM_PATH}/bin/rocm_agent_enumerator" ]; then
        GPU_ARCH=$(${ROCM_PATH}/bin/rocm_agent_enumerator | grep -v "gfx000" | grep "gfx" | head -1)
    fi
    if [ -z "$GPU_ARCH" ]; then
        _env_arch="${GPU_ARCHS:-$PYTORCH_ROCM_ARCH}"
        [ -n "$_env_arch" ] && GPU_ARCH=$(echo "$_env_arch" | tr ';' '\n' | grep "gfx" | head -1)
    fi
    [ -z "$GPU_ARCH" ] && { echo "Warning: GPU arch not detected, defaulting gfx942" >&2; GPU_ARCH="gfx942"; }
fi
echo "[cco] GPU architecture: $GPU_ARCH"

# ---------------------------------------------------------------------------
# Detect NIC type for device macros (mirrors build_shmem_bitcode.sh)
# ---------------------------------------------------------------------------
detect_nic_type() {
    if [ "${USE_BNXT:-}" = "ON" ]; then echo "bnxt"; return; fi
    if [ "${USE_IONIC:-}" = "ON" ]; then echo "ionic"; return; fi
    if [ "${MORI_DEVICE_NIC:-}" != "" ]; then echo "${MORI_DEVICE_NIC}"; return; fi
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
                    local drv=$(readlink -f "$dev/device/driver" 2>/dev/null | xargs basename 2>/dev/null)
                    case "$drv" in
                        bnxt_re|bnxt_en) bnxt=$((bnxt + 1)) ;;
                        ionic_rdma|ionic) ionic=$((ionic + 1)) ;;
                        mlx5_core|mlx5_ib) mlx5=$((mlx5 + 1)) ;;
                    esac ;;
            esac
        done
        if [ $bnxt -gt 0 ] && [ $bnxt -ge $mlx5 ]; then echo "bnxt"; return; fi
        if [ $ionic -gt 0 ] && [ $ionic -ge $mlx5 ]; then echo "ionic"; return; fi
        if [ $mlx5 -gt 0 ]; then echo "mlx5"; return; fi
    fi
    echo "mlx5"
}

NIC_TYPE=$(detect_nic_type)
NIC_DEFINES=""
case "$NIC_TYPE" in
    bnxt)  NIC_DEFINES="-DMORI_DEVICE_NIC_BNXT" ;;
    ionic) NIC_DEFINES="-DMORI_DEVICE_NIC_IONIC" ;;
esac
echo "[cco] NIC: ${NIC_TYPE^^}  (cov=${COV})"

# ---------------------------------------------------------------------------
# Compile wrapper to device bitcode
# ---------------------------------------------------------------------------
HIPCC="${ROCM_PATH}/bin/hipcc"
LLVM_LINK="${ROCM_PATH}/lib/llvm/bin/llvm-link"
OPT="${ROCM_PATH}/lib/llvm/bin/opt"

WRAPPER_SRC="${MORI_DIR}/src/cco/device/cco_device_wrapper.cpp"
[ -f "$WRAPPER_SRC" ] || { echo "Error: not found: $WRAPPER_SRC"; exit 1; }

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

INCLUDES="-I${MORI_DIR} -I${MORI_DIR}/include -I${MORI_DIR}/src"
[ -d "${MORI_DIR}/3rdparty/spdlog/include" ] && INCLUDES="$INCLUDES -I${MORI_DIR}/3rdparty/spdlog/include"
[ -d "${MORI_DIR}/3rdparty/msgpack-c/include" ] && INCLUDES="$INCLUDES -I${MORI_DIR}/3rdparty/msgpack-c/include"
MPI_INC=$(mpicc --showme:compile 2>/dev/null | grep -oP '(?<=-I)\S+' | head -1 || true)
[ -n "$MPI_INC" ] && INCLUDES="$INCLUDES -I${MPI_INC}"

COMMON_FLAGS="--cuda-device-only -emit-llvm --offload-arch=${GPU_ARCH} -fgpu-rdc -mcode-object-version=${COV} -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS ${NIC_DEFINES}"

echo "[cco] Compiling wrapper ..."
$HIPCC -c $COMMON_FLAGS $INCLUDES "$WRAPPER_SRC" -o "$TEMP_DIR/wrapper.bc"

echo "[cco] Stripping llvm.lifetime intrinsics ..."
$OPT -S "$TEMP_DIR/wrapper.bc" -o "$TEMP_DIR/wrapper.ll"
sed -i '/llvm\.lifetime\./d' "$TEMP_DIR/wrapper.ll"
$OPT "$TEMP_DIR/wrapper.ll" -o "$TEMP_DIR/libmori_cco_device.bc"

echo "[cco] Verifying symbols ..."
$OPT -S "$TEMP_DIR/libmori_cco_device.bc" -o "$TEMP_DIR/verify.ll"
for sym in cco_gda_put cco_gda_signal cco_gda_wait_signal cco_devcomm_rank; do
    if grep -q "@${sym}\b" "$TEMP_DIR/verify.ll"; then
        echo "[cco] ✓ ${sym}"
    else
        echo "Error: ${sym} not found in bitcode"; exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cp -f "$TEMP_DIR/libmori_cco_device.bc" "$OUTPUT_DIR/"
echo "[cco] Output: $OUTPUT_DIR/libmori_cco_device.bc"
echo "[cco] Done."
