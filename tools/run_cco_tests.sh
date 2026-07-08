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
  name="$(basename "$bin")"
  case "$name" in
    test_lsa_memcheck|test_gda_barrier|test_gda_counter|test_gda_multi_context|test_gda_signal_ut|test_gda_thread_aggregate|test_gda_put|test_gda_get) continue ;;
    # SDMA transport tests only exercise real work when SDMA queues exist; run
    # them with MORI_ENABLE_SDMA=1 so they don't silently SKIP.
    test_sdma_*) continue ;;
  esac
  echo "=== $name ==="
  timeout -k 10 120 "./$bin" "$nranks" || failed=1
done

for bin in tests/cpp/cco/test_sdma_*; do
  [ -x "$bin" ] || continue
  echo "=== $(basename "$bin") (MORI_ENABLE_SDMA=1) ==="
  MORI_ENABLE_SDMA=1 timeout -k 10 120 "./$bin" "$nranks" || failed=1
done
exit $failed
