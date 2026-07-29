#!/bin/bash
# Run mori build / role-switch tests inside the GPU container.
#
# Exists because the remote login shell on the build nodes is tcsh, which
# mangles ${PIPESTATUS[0]} and $(...) in an `ssh host "docker exec ... bash -lc
# ..."` one-liner. Every earlier turn hand-escaped that chain, and one of those
# escapes is why every published PYTEST_RC was measuring `tail` instead of
# pytest (RESULTS_M T5a). A file on NFS is read by exactly one shell, so the
# quoting question does not arise.
#
# Usage (from the repo root, inside the container):
#   tests/python/ops/run_role_switch_suite.sh build
#   tests/python/ops/run_role_switch_suite.sh test '<pytest -k expr>'
#   tests/python/ops/run_role_switch_suite.sh regress
set -u
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 99
git config --global --add safe.directory "$REPO" 2>/dev/null
echo "=== HEAD: $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "=== date: $(date -u +%FT%TZ) ==="

mode="${1:-test}"
shift || true

case "$mode" in
  build)
    python setup.py build_ext --inplace 2>&1 | tail -30
    rc=${PIPESTATUS[0]}
    echo "BUILD_RC=$rc"
    ;;
  test|regress)
    # Free VRAM up front: a run that dies in shmem init because the node is
    # busy is not evidence about the code under test, and three turns of this
    # campaign were lost to not recording it (RESULTS_M T3).
    python -c 'import torch;f,t=torch.cuda.mem_get_info(0);print(f"VRAM free {f/2**30:.1f} GiB of {t/2**30:.1f} GiB")' 2>&1
    if [ "$mode" = "regress" ]; then
      target="tests/python/ops/test_dispatch_combine_intranode.py"
    else
      target="tests/python/ops/test_dispatch_combine_role_switch.py"
    fi
    # tee to a full log, THEN tail. Piping straight into `tail` buffers the
    # whole run, so a suite that hangs shows nothing at all until it exits --
    # which is precisely when a live progress read is worth most. The T6
    # regression run sat 18 min with two defunct workers and a 91-byte log for
    # exactly this reason.
    full="${MORI_TEST_FULL_LOG:-/tmp/mori_suite_full.log}"
    PYTHONPATH="$REPO/python" python -m pytest -q -s -p no:cacheprovider \
      "$target" "$@" 2>&1 | tee "$full" | tail -60
    rc=${PIPESTATUS[0]}
    echo "FULL_LOG=$full ($(wc -l < "$full") lines)"
    echo "PYTEST_RC=$rc"   # real pytest rc: set -o pipefail + PIPESTATUS[0]
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 98
    ;;
esac
