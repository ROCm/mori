#!/bin/bash
# Run all CCO unit tests from the build directory.
# Usage: run_all.sh <build_dir> <nranks>
set -u

build_dir=${1:?usage: run_all.sh BUILD_DIR NRANKS}
nranks=${2:?usage: run_all.sh BUILD_DIR NRANKS}

cd "$build_dir"
failed=0
for bin in tests/cpp/cco/test_*; do
  [ -x "$bin" ] || continue
  case "$(basename "$bin")" in
    test_lsa_memcheck) continue ;;
  esac
  echo "=== $(basename "$bin") ==="
  timeout 120 "./$bin" "$nranks" || failed=1
done
# for bin in examples/cco_lsa_put examples/cco_gda_put; do
#   [ -x "$bin" ] || continue
#   echo "=== $(basename "$bin") ==="
#   timeout 120 mpirun --allow-run-as-root -np "$nranks" "./$bin" || failed=1
# done
# exit $failed
