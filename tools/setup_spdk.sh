#!/bin/bash
# setup_spdk.sh — Build and install SPDK from 3rdparty/spdk submodule
#
# Usage:
#   tools/setup_spdk.sh              # build + install to /usr/local
#   tools/setup_spdk.sh --prefix /opt/spdk   # custom install prefix
#   tools/setup_spdk.sh --check      # only check system dependencies
#   tools/setup_spdk.sh --help
#
# This script:
#   1. Checks required system packages
#   2. Initializes the 3rdparty/spdk git submodule (if needed)
#   3. Configures SPDK with shared libraries
#   4. Builds SPDK (parallel make)
#   5. Installs to prefix and runs ldconfig
#
# After running this script, rebuild UMBP to pick up SPDK:
#   cmake --build build_umbp -j$(nproc)
#   # or: BUILD_UMBP=1 pip install .

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MORI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPDK_DIR="$MORI_ROOT/3rdparty/spdk"
PREFIX="/usr/local"
CHECK_ONLY=false
JOBS="$(nproc)"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prefix PATH   Install prefix (default: /usr/local)"
    echo "  --check         Only check system dependencies, don't build"
    echo "  --jobs N         Parallel make jobs (default: $(nproc))"
    echo "  --help           Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)  PREFIX="$2"; shift 2 ;;
        --check)   CHECK_ONLY=true; shift ;;
        --jobs)    JOBS="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *)         echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ===========================================================================
# Step 1: Check system dependencies
# ===========================================================================
echo "=== Step 1: Checking system dependencies ==="

REQUIRED_PKGS=(
    gcc g++ make nasm
    libaio-dev libssl-dev libnuma-dev uuid-dev
    libcunit1-dev libjson-c-dev libcmocka-dev
    libfuse3-dev libelf-dev
    pkg-config python3
)

MISSING=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "  Missing packages: ${MISSING[*]}"
    echo ""
    echo "  Install with:"
    echo "    sudo apt-get update && sudo apt-get install -y ${MISSING[*]}"
    echo ""
    if $CHECK_ONLY; then
        exit 1
    fi
    echo "  Attempting to install..."
    apt-get update -qq && apt-get install -y "${MISSING[@]}" || {
        echo "ERROR: Failed to install dependencies. Install them manually."
        exit 1
    }
else
    echo "  All required packages are installed."
fi

if $CHECK_ONLY; then
    echo ""
    echo "Dependency check passed. Run without --check to build."
    exit 0
fi

# ===========================================================================
# Step 2: Initialize submodule
# ===========================================================================
echo ""
echo "=== Step 2: Initializing SPDK submodule ==="

if [[ ! -f "$SPDK_DIR/configure" ]]; then
    echo "  Initializing 3rdparty/spdk submodule..."
    cd "$MORI_ROOT"
    git submodule update --init 3rdparty/spdk
    echo "  Checking out v26.01 tag..."
    git -C "$SPDK_DIR" fetch origin tag v26.01 --depth 1 2>/dev/null || true
    git -C "$SPDK_DIR" checkout v26.01
else
    echo "  SPDK source already present."
    echo "  Version: $(git -C "$SPDK_DIR" describe --tags 2>/dev/null || git -C "$SPDK_DIR" log --oneline -1)"
fi

# Initialize SPDK's own submodules (dpdk, isa-l, etc.)
echo "  Initializing SPDK submodules (dpdk, isa-l, ...)..."
cd "$SPDK_DIR"
git submodule update --init 2>&1 | tail -5 || true

# ===========================================================================
# Step 3: Configure
# ===========================================================================
echo ""
echo "=== Step 3: Configuring SPDK ==="
echo "  Prefix: $PREFIX"

cd "$SPDK_DIR"
if [[ -f build/config.h ]]; then
    echo "  Existing SPDK build detected. Re-running configure to apply current options."
else
    echo "  Fresh configure."
fi
./configure --with-shared --prefix="$PREFIX" 2>&1 | tail -20

# ===========================================================================
# Step 4: Build
# ===========================================================================
echo ""
echo "=== Step 4: Building SPDK (make -j$JOBS) ==="

cd "$SPDK_DIR"
make -j"$JOBS" 2>&1 | tail -5
echo "  Build complete."

# ===========================================================================
# Step 5: Install
# ===========================================================================
echo ""
echo "=== Step 5: Installing to $PREFIX ==="

cd "$SPDK_DIR"
make install prefix="$PREFIX" 2>&1 | tail -5

# Update shared library cache
ldconfig 2>/dev/null || true

# Verify pkg-config can find SPDK
echo ""
echo "=== Verification ==="
if pkg-config --exists spdk_event 2>/dev/null; then
    echo "  pkg-config: SPDK found"
    echo "    CFLAGS: $(pkg-config --cflags spdk_event)"
    echo "    LIBS:   $(pkg-config --libs spdk_event | cut -c1-80)..."
elif [[ -f "$PREFIX/lib/pkgconfig/spdk_event.pc" ]]; then
    echo "  pkg-config files installed at $PREFIX/lib/pkgconfig/"
    echo "  If not auto-detected, set:"
    echo "    export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
else
    echo "  WARNING: SPDK pkg-config files not found after install."
fi

echo ""
echo "=== Done ==="
echo "SPDK v26.01 installed to $PREFIX"
echo ""
echo "Next steps:"
echo "  1. Rebuild UMBP to pick up SPDK support:"
echo "     cd $MORI_ROOT && BUILD_UMBP=1 pip install ."
echo ""
echo "  2. Verify SPDK environment:"
echo "     tools/umbp_spdk_preflight.sh --pci <your-nvme-pci-addr>"
