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
    test_lsa_memcheck|test_gda_barrier|test_gda_counter|test_gda_multi_context|test_gda_signal_ut|test_gda_thread_aggregate|test_gda_put|test_gda_get) continue ;;
  esac
  # GDA-FULL tests need intranode cross-rail RDMA (FULL connections). On runners
  # where cross-rail is unavailable they can't pass; skip when
  # MORI_CCO_SKIP_GDA_FULL=1 (set per-runner in CI).
  if [ "${MORI_CCO_SKIP_GDA_FULL:-0}" = "1" ]; then
    case "$(basename "$bin")" in
      test_gda_flush_async|test_gda_modes|test_multiprocess)
        echo "=== $(basename "$bin") — SKIPPED (MORI_CCO_SKIP_GDA_FULL=1) ==="
        continue ;;
    esac
  fi
  echo "=== $(basename "$bin") ==="
  timeout -k 10 120 "./$bin" "$nranks" || failed=1
done
exit $failed
