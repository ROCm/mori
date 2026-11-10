#!/usr/bin/env bash
# Build and run the complete UMBP real-device validation inventory.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage: tools/umbp_engine_gpu_check.sh [BUILD_ROOT]

BUILD_ROOT defaults to /perf_apps/nima/build/umbp-engine-hbm-gpu.
Optional environment:
  UMBP_GPU_BUILD_JOBS   parallel build jobs (default 8)
  UMBP_GPU_CMAKE_ARGS   extra CMake arguments
  UMBP_GPU_GDS_DIR      local ext4/xfs directory for GDS tests (default TMPDIR or /tmp)
  HIP_VISIBLE_DEVICES   GPU ordinals exposed to the test process
  MORI_RDMA_DEVICES     RDMA device filter used by the real VMM registration test
EOF
  exit 0
fi
build_root=${1:-/perf_apps/nima/build/umbp-engine-hbm-gpu}
cmake_dir="$build_root/cmake"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_dir="$build_root/logs/$stamp"
mkdir -p "$log_dir"

summary() {
  local passed=$1 skipped=$2 failed=$3
  printf 'UMBP_ENGINE_GPU_SUMMARY passed=%s skipped=%s failed=%s\n' \
    "$passed" "$skipped" "$failed"
  printf 'logs=%s\n' "$log_dir"
}

extra_cmake=()
if [[ -n ${UMBP_GPU_CMAKE_ARGS:-} ]]; then
  # Intentional shell-style list supplied by the operator.
  read -r -a extra_cmake <<<"$UMBP_GPU_CMAKE_ARGS"
fi

configure=(cmake -S "$repo_root" -B "$cmake_dir" -G Ninja
  -DCMAKE_BUILD_TYPE=Release -DBUILD_UMBP=ON -DMORI_UMBP_TESTING=ON)
configure+=("${extra_cmake[@]}")
if ! "${configure[@]}" 2>&1 | tee "$log_dir/configure.log"; then
  summary 0 0 1
  exit 1
fi

targets=(
  test_engine_gpu
  test_umbp_ssd_tier_ranges
  test_umbp_pool_client_ranges
  test_umbp_pool_client_batch_put
  test_transfer_engine
  test_hbm_backend
  test_ssd_backend
  test_umbp_sim
  mori_pybinds
  mori_umbp_pybinds
  umbp_standalone_server
)
if cmake --build "$cmake_dir" --target help 2>/dev/null | grep -q 'test_gds_engine'; then
  targets+=(test_gds_engine)
fi
if ! cmake --build "$cmake_dir" -j "${UMBP_GPU_BUILD_JOBS:-8}" --target "${targets[@]}" \
  2>&1 | tee "$log_dir/build.log"; then
  summary 0 0 1
  exit 1
fi

export UMBP_STANDALONE_BIN="$cmake_dir/src/umbp/umbp_standalone_server"
export TMPDIR=${UMBP_GPU_GDS_DIR:-${TMPDIR:-/tmp}}
export MORI_IO_QP_MAX_SEND_WR=${MORI_IO_QP_MAX_SEND_WR:-1024}
export MORI_IO_QP_MAX_CQE=${MORI_IO_QP_MAX_CQE:-4096}
mkdir -p "$TMPDIR"

# CTest binaries carry build-tree rpaths, but the standalone server is exec'd
# as a child of test_engine_gpu and must resolve every Mori DSO independently.
mapfile -t mori_build_lib_dirs < <(
  find "$cmake_dir/src" -type f -name 'libmori*.so' -printf '%h\n' | sort -u
)
if ((${#mori_build_lib_dirs[@]} > 0)); then
  mori_build_library_path=$(IFS=:; echo "${mori_build_lib_dirs[*]}")
  export LD_LIBRARY_PATH="$mori_build_library_path:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

ctest_rc=0
ctest --test-dir "$cmake_dir" -L '^gpu$' --output-on-failure --timeout 600 \
  --output-junit "$log_dir/ctest.xml" 2>&1 | tee "$log_dir/ctest.log" || ctest_rc=${PIPESTATUS[0]}

runtime="$build_root/runtime"
export UMBP_GPU_REPO_ROOT="$repo_root"
export UMBP_GPU_CMAKE_DIR="$cmake_dir"
export UMBP_GPU_RUNTIME="$runtime"
python3 - <<'PY'
import os
import pathlib
import shutil

repo = pathlib.Path(os.environ["UMBP_GPU_REPO_ROOT"])
cmake = pathlib.Path(os.environ["UMBP_GPU_CMAKE_DIR"])
out = pathlib.Path(os.environ["UMBP_GPU_RUNTIME"]) / "mori"
shutil.copytree(
    repo / "python/mori",
    out,
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"),
)
for library in (cmake / "src").rglob("libmori*.so"):
    shutil.copy2(library, out / library.name)
PY

export PYTHONPATH="$runtime${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime/mori:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
runtime_rc=0
python3 -c 'from mori.cpp import _load_umbp; _load_umbp()' \
  2>&1 | tee "$log_dir/runtime_smoke.log" || runtime_rc=${PIPESTATUS[0]}
pytest_rc=0
python3 -m pytest -q -m gpu "$repo_root/tests/python/umbp/test_umbp_engine_gpu.py" \
  --junitxml="$log_dir/pytest.xml" -o cache_dir="$build_root/cache/pytest" \
  2>&1 | tee "$log_dir/pytest.log" || pytest_rc=${PIPESTATUS[0]}

export UMBP_GPU_CTEST_XML="$log_dir/ctest.xml"
export UMBP_GPU_PYTEST_XML="$log_dir/pytest.xml"
read -r passed skipped failed < <(python3 - <<'PY'
import os
import pathlib
import xml.etree.ElementTree as ET

def counts(path):
    path = pathlib.Path(path)
    if not path.exists():
        return 0, 0, 0
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    return tests - failures - errors - skipped, skipped, failures + errors

c = counts(os.environ["UMBP_GPU_CTEST_XML"])
p = counts(os.environ["UMBP_GPU_PYTEST_XML"])
print(c[0] + p[0], c[1] + p[1], c[2] + p[2])
PY
)
if (( ctest_rc != 0 || runtime_rc != 0 || pytest_rc != 0 )); then
  (( failed == 0 )) && failed=1
fi
summary "$passed" "$skipped" "$failed"
(( failed == 0 ))
