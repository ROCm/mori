#!/usr/bin/env bash
# The one place that knows how to compile ualoe_bw.cpp.
#
# There are two reasons this TU gets built outside CMake as well as inside it. Every cell of a sweep
# needs different -D geometry (tile, pipe depth, issuer count are all compile time), and a sweep
# recompiles between configurations; and the sweeps have to run on a node with two idle GPUs, which is
# not where the package is normally installed. So the scripts compile it themselves.
#
# What must not differ between the two paths is the flag set, because the recorded tables in results/
# were measured with it and a silent change there invalidates them. This file is that flag set, and
# `benchmark/CMakeLists.txt` (target `ualoe_bw`) has to be kept in step with it -- both are annotated.
#
# Usage: FLAGS-carrying variables in, path to binary out.
#   ARCH    offload arch, must be gfx125x for the TDM intrinsics to exist  (default gfx1250)
#   BASEX   which sweep the binary runs (-DSWEEP_MATRIX, -DSWEEP_16, ...)
#   GRID    geometry -D flags
#   EXTRAX  anything else the caller needs appended
#   BIN     output path
# Set BIN to an already-built binary and SKIP_BUILD=1 to reuse a CMake artifact instead.
set -uo pipefail

ARCH="${ARCH:-gfx1250}"
BASEX="${BASEX:-}"
GRID="${GRID:-}"
EXTRAX="${EXTRAX:-}"
BIN="${BIN:?BIN is required}"
SRC="${SRC:-ualoe_bw.cpp}"

if [ -n "${SKIP_BUILD:-}" ]; then
  [ -x "$BIN" ] || { echo "SKIP_BUILD set but $BIN is not executable"; exit 1; }
  echo "REUSING $BIN"
  exit 0
fi

# -O3 rather than the -O2 the library's device code uses: every recorded table in results/ was measured
# at -O3, and the TDM issue loop is tight enough that the two are not interchangeable.
# __HIP_PLATFORM_AMD__ and HIP_ENABLE_WARP_SYNC_BUILTINS come from the library build (the latter is a
# global add_definitions in the root CMakeLists) and are carried here so both paths see one dialect.
hipcc -std=c++17 -O3 --offload-arch="$ARCH" \
      -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS \
      $BASEX $GRID $EXTRAX \
      "$SRC" -o "$BIN" || { echo "COMPILE FAILED"; exit 1; }
echo "COMPILE OK $BIN"
