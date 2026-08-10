#!/bin/bash
# CI guard for the JIT v2 host/device split (docs/MORI_JIT_V2_DESIGN.md §3.5).
#
# The property: every header the HOST includes must compile with a plain C++
# compiler. Violating it does not break the build immediately -- it quietly drags
# the host into hipcc, and mori's "pip install needs no hipcc" promise dies with
# no error message. Hence a test rather than a convention.
#
# Checks, per host-shared header:
#   1. compiles standalone under g++ -std=c++17 and -std=c++20
#   2. contains no HIP include
#   3. contains no __host__/__device__/__global__ attribute
set -u

ROOT=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
CXX=${CXX:-g++}
ROCM=${ROCM_PATH:-/opt/rocm}
fail=0

# Headers that both the host and the JIT-generated device TU include.
SHARED_HEADERS=(
  "include/mori/jit/v2/render.hpp"
  "include/mori/ops/dispatch_combine_v2/ep_cfg.hpp"
)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for h in "${SHARED_HEADERS[@]}"; do
  path="$ROOT/$h"
  if [ ! -f "$path" ]; then
    echo "MISSING  $h"
    fail=1
    continue
  fi

  # (2) no HIP dependency
  if grep -qE '^\s*#\s*include\s*[<"]hip/' "$path"; then
    echo "HIP-DEP  $h includes a hip/ header -- host TUs would need hipcc"
    fail=1
  fi

  # (3) no device attributes -- in CODE. The rule itself is stated in comments in
  # these headers, so skip lines that are comments.
  if grep -vE '^\s*(//|\*|/\*)' "$path" | grep -qE '__(host|device|global)__'; then
    echo "ATTR     $h uses a __host__/__device__/__global__ attribute"
    echo "         (shared arithmetic must be plain constexpr; see §3.5.1)"
    fail=1
  fi

  # (1) compiles standalone with a plain C++ compiler, both standards
  echo "#include \"$h\"" > "$tmp/probe.cpp"
  for std in 17 20; do
    if ! "$CXX" -std=c++$std -fsyntax-only -I "$ROOT" -I "$ROOT/include" \
         -I "$ROCM/include" -D__HIP_PLATFORM_AMD__ "$tmp/probe.cpp" 2> "$tmp/err"; then
      echo "CXX$std    $h does not compile with $CXX -std=c++$std:"
      sed 's/^/         /' "$tmp/err" | head -8
      fail=1
    fi
  done
done

# The device kernel headers are the mirror image: no host source may #include
# one, or the host build starts needing hipcc. A path appearing inside a string
# literal is fine -- that is the renderer emitting the include for the DEVICE TU,
# which is exactly how it is supposed to get there.
DEVICE_ONLY_HEADERS=("ep_intranode_kernel.hpp")
pattern=$(printf '|%s' "${DEVICE_ONLY_HEADERS[@]}")
pattern=${pattern:1}
leaks=$(grep -rnE "^[[:space:]]*#[[:space:]]*include[[:space:]]*\"[^\"]*($pattern)\"" \
          "$ROOT/src" "$ROOT/include" --include=*.cpp --include=*.hpp 2>/dev/null || true)
if [ -n "$leaks" ]; then
  echo "LEAK     a host source #includes a device kernel header:"
  echo "$leaks" | sed 's/^/         /'
  fail=1
fi

if [ $fail -eq 0 ]; then
  echo "host/device split OK (${#SHARED_HEADERS[@]} shared headers)"
fi
exit $fail
