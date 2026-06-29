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
  timeout -k 10 120 "./$bin" "$nranks" || failed=1
done
exit $failed
