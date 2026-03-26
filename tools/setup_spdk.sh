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
BUILD_PYTHON=""
SPDK_REQUIRED_SUBMODULES=(dpdk isa-l isa-l-crypto)
REQUIRED_BUILD_TOOLS=(
    meson
    ninja
    autoreconf
    libtoolize
    help2man
    pkg-config
    python3
)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prefix PATH   Install prefix (default: /usr/local)"
    echo "  --check         Only check system dependencies, don't build"
    echo "  --jobs N        Parallel build/submodule jobs (default: $(nproc))"
    echo "  --help          Show this help"
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

SPDK_CONFIGURE_ARGS=(
    --with-shared
    --prefix="$PREFIX"
    --disable-tests
    --disable-unit-tests
    --disable-examples
)

print_spdk_version() {
    git -C "$SPDK_DIR" describe --tags 2>/dev/null || \
    git -C "$SPDK_DIR" rev-parse --short HEAD 2>/dev/null || \
    echo "unknown"
}

is_git_checkout() {
    local dir="$1"
    local top=""

    [[ -d "$dir" ]] || return 1

    top="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null || true)"
    [[ "$top" == "$dir" ]]
}

directory_has_entries() {
    local dir="$1"
    local entries=()

    [[ -d "$dir" ]] || return 1

    shopt -s nullglob dotglob
    entries=("$dir"/*)
    shopt -u nullglob dotglob

    (( ${#entries[@]} > 0 ))
}

have_submodule_source() {
    local name="$1"

    case "$name" in
        dpdk)
            [[ -f "$SPDK_DIR/dpdk/meson.build" && -f "$SPDK_DIR/dpdk/config/meson.build" ]]
            ;;
        isa-l|isa-l-crypto)
            [[ -f "$SPDK_DIR/$name/autogen.sh" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

python_has_module() {
    local python_bin="$1"
    local module_name="$2"

    [[ -x "$python_bin" ]] || return 1
    "$python_bin" -c "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec('$module_name') else 1)" \
        >/dev/null 2>&1
}

package_is_satisfied() {
    local pkg="$1"
    local active_python=""

    case "$pkg" in
        meson)
            command -v meson >/dev/null 2>&1
            ;;
        ninja-build)
            command -v ninja >/dev/null 2>&1
            ;;
        python3-pyelftools)
            active_python="$(command -v python3 2>/dev/null || true)"
            if [[ -n "$active_python" ]] && python_has_module "$active_python" elftools; then
                return 0
            fi
            python_has_module /usr/bin/python3 elftools
            ;;
        *)
            dpkg -s "$pkg" &>/dev/null
            ;;
    esac
}

check_required_build_tools() {
    local missing=()
    local tool=""

    for tool in "${REQUIRED_BUILD_TOOLS[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Required build tools not found in PATH: ${missing[*]}"
        echo "       Try: sudo apt-get install -y meson ninja-build autoconf automake libtool help2man"
        return 1
    fi
}

select_build_python() {
    local active_python=""

    active_python="$(command -v python3 2>/dev/null || true)"
    if [[ -z "$active_python" ]]; then
        echo "ERROR: python3 not found in PATH."
        return 1
    fi

    if python_has_module "$active_python" elftools; then
        BUILD_PYTHON="$active_python"
        echo "  Python for build: $BUILD_PYTHON"
        return 0
    fi

    if python_has_module /usr/bin/python3 elftools; then
        BUILD_PYTHON="/usr/bin/python3"
        export PATH="/usr/bin:/bin:$PATH"
        echo "  Active python3 ($active_python) lacks 'elftools'; using $BUILD_PYTHON for build."
        return 0
    fi

    echo "ERROR: python3 cannot import required module 'elftools'."
    echo "       Active python3: $active_python"
    echo "       Fix one of these and re-run:"
    echo "         1. python3 -m pip install pyelftools"
    echo "         2. /usr/bin/python3 -m pip install pyelftools"
    echo "         3. sudo apt-get install -y python3-pyelftools"
    return 1
}

spdk_config_value() {
    local key="$1"

    awk -F'\\?=' -v key="$key" '$1 == key { print $2; exit }' "$SPDK_DIR/mk/config.mk"
}

spdk_config_enabled() {
    [[ "$(spdk_config_value "$1")" == "y" ]]
}

spdk_install_dirs() {
    local dirs=(lib module include)

    if spdk_config_enabled CONFIG_SHARED; then
        dirs+=(shared_lib)
    fi
    if spdk_config_enabled CONFIG_APPS; then
        dirs+=(app)
    fi
    if spdk_config_enabled CONFIG_IPSEC_MB; then
        dirs+=(ipsecbuild)
    fi
    if spdk_config_enabled CONFIG_ISAL; then
        dirs+=(isalbuild)
    fi
    if spdk_config_enabled CONFIG_ISAL_CRYPTO; then
        dirs+=(isalcryptobuild)
    fi
    if spdk_config_enabled CONFIG_VFIO_USER; then
        dirs+=(vfiouserbuild)
    fi
    if spdk_config_enabled CONFIG_XNVME; then
        dirs+=(xnvmebuild)
    fi
    if spdk_config_enabled CONFIG_SMA; then
        dirs+=(proto)
    fi
    if spdk_config_enabled CONFIG_GOLANG; then
        dirs+=("go/rpc")
    fi

    if [[ "$(spdk_config_value CONFIG_ENV)" == "$SPDK_DIR/lib/env_dpdk" ]] && \
       [[ "$(spdk_config_value CONFIG_DPDK_DIR)" == "$SPDK_DIR/dpdk/build" ]] && \
       [[ "$(spdk_config_value CONFIG_DPDK_PKG_CONFIG)" != "y" ]]; then
        dirs+=(dpdkbuild)
    fi

    printf '%s ' "${dirs[@]}"
}

install_spdk_core() {
    local install_dirs=()
    local dir=""

    read -r -a install_dirs <<< "$(spdk_install_dirs)"

    for dir in "${install_dirs[@]}"; do
        echo "  Installing $dir..."
        make -C "$SPDK_DIR/$dir" prefix="$PREFIX" install 2>&1 | tail -5
    done
}

ensure_spdk_submodules() {
    local to_init=()
    local name=""
    local dir=""

    echo "  Ensuring required SPDK submodules: ${SPDK_REQUIRED_SUBMODULES[*]}"

    for name in "${SPDK_REQUIRED_SUBMODULES[@]}"; do
        dir="$SPDK_DIR/$name"

        if is_git_checkout "$dir"; then
            echo "    $name: git checkout already present."
            continue
        fi

        if have_submodule_source "$name"; then
            echo "    $name: source tree already present; reusing existing directory."
            continue
        fi

        if directory_has_entries "$dir"; then
            echo "ERROR: $dir exists but is not a usable source tree or git submodule."
            echo "       Remove or repair this directory, then re-run this script."
            exit 1
        fi

        to_init+=("$name")
    done

    if [[ ${#to_init[@]} -gt 0 ]]; then
        echo "  Fetching missing SPDK submodules: ${to_init[*]}"
        git -C "$SPDK_DIR" submodule sync -- "${to_init[@]}"
        git -C "$SPDK_DIR" submodule update --init --depth 1 --jobs "$JOBS" "${to_init[@]}"
    else
        echo "  Required SPDK submodules are already available."
    fi

    for name in "${SPDK_REQUIRED_SUBMODULES[@]}"; do
        if ! have_submodule_source "$name"; then
            echo "ERROR: Required SPDK source tree '$name' is missing after initialization."
            exit 1
        fi
    done
}

# ===========================================================================
# Step 1: Check system dependencies
# ===========================================================================
echo "=== Step 1: Checking system dependencies ==="

REQUIRED_PKGS=(
    gcc g++ make nasm
    meson ninja-build
    autoconf automake libtool help2man
    libaio-dev libssl-dev libnuma-dev uuid-dev
    libcunit1-dev libjson-c-dev libcmocka-dev
    libfuse3-dev libelf-dev
    pkg-config python3 python3-pyelftools
)

MISSING=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! package_is_satisfied "$pkg"; then
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

check_required_build_tools || exit 1
select_build_python || exit 1

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
    if directory_has_entries "$SPDK_DIR"; then
        echo "ERROR: $SPDK_DIR exists but does not look like a valid SPDK source tree."
        echo "       Remove or repair this directory, then re-run this script."
        exit 1
    fi

    echo "  Initializing 3rdparty/spdk submodule..."
    git -C "$MORI_ROOT" submodule sync -- 3rdparty/spdk
    git -C "$MORI_ROOT" submodule update --init --depth 1 --jobs 1 3rdparty/spdk
else
    echo "  SPDK source already present."
fi
echo "  Version: $(print_spdk_version)"

# Initialize only the nested submodules required by our default SPDK build.
ensure_spdk_submodules

# ===========================================================================
# Step 3: Configure
# ===========================================================================
echo ""
echo "=== Step 3: Configuring SPDK ==="
echo "  Prefix: $PREFIX"
echo "  Reduced build scope: tests, unit tests, and examples disabled."

cd "$SPDK_DIR"
if [[ -f build/config.h ]]; then
    echo "  Existing SPDK build detected. Re-running configure to apply current options."
else
    echo "  Fresh configure."
fi
./configure "${SPDK_CONFIGURE_ARGS[@]}" 2>&1 | tail -20

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
echo "  Installing core SPDK artifacts only; skipping optional Python package."

cd "$SPDK_DIR"
install_spdk_core

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
echo "SPDK $(print_spdk_version) installed to $PREFIX"
echo ""
echo "Next steps:"
echo "  1. Rebuild UMBP to pick up SPDK support:"
echo "     cd $MORI_ROOT && BUILD_UMBP=1 pip install ."
echo ""
echo "  2. Optional SPDK Python bindings:"
echo "     python3 -m pip install \"$SPDK_DIR/python\""
echo ""
echo "  3. Verify SPDK environment:"
echo "     tools/umbp_spdk_preflight.sh --pci <your-nvme-pci-addr>"
