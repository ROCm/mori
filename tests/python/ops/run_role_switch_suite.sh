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
#   tests/python/ops/run_role_switch_suite.sh asyncll '<pytest -k expr>'
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
  test|regress|asyncll)
    # ABORT, do not merely report, when the node cannot hold the run.
    #
    # Printing was not enough: T10a and T10b both bannered "VRAM free 6.3 GiB"
    # and ran anyway, and T9c sampled an idle node 28 s before a vLLM job
    # filled it. A run that dies in shmem init because the node is busy is not
    # evidence about the code under test in EITHER direction, but it costs a
    # full timeout to find that out and it reads like a product failure. Four
    # turns of this campaign went to uninterpretable REDs of exactly this shape.
    #
    # Sample EVERY visible device, not just device 0: the ranks are spread over
    # all of them and one full card is enough to hang the group.
    echo "=== containers on this node ==="
    docker ps --format '{{.Names}}' 2>/dev/null || echo "(docker not visible from inside)"
    heap_need_gib="${MORI_TEST_MIN_FREE_GIB:-}"
    if [ -z "$heap_need_gib" ]; then
      # MORI_SHMEM_HEAP_SIZE is like "32G"/"512M"/bytes; default matches conftest.
      hs="${MORI_SHMEM_HEAP_SIZE:-32G}"
      heap_need_gib=$(python -c "
import re,sys
s=str('''$hs''').strip()
m=re.match(r'^(\d+)\s*([GgMmKk]?)[Bb]?$', s)
if not m: print(36); sys.exit()
n=int(m.group(1)); u=m.group(2).upper()
b=n*(1024**3 if u=='G' else 1024**2 if u=='M' else 1024 if u=='K' else 1)
print(f'{b/2**30+4:.1f}')" 2>/dev/null || echo 36)
    fi
    python - "$heap_need_gib" <<'PYVRAM'
import sys, torch
need = float(sys.argv[1])
worst = None
for i in range(torch.cuda.device_count()):
    f, t = torch.cuda.mem_get_info(i)
    g = f / 2**30
    print(f"VRAM dev{i}: free {g:.1f} GiB of {t/2**30:.1f} GiB")
    worst = g if worst is None else min(worst, g)
print(f"VRAM worst-case free {worst:.1f} GiB, need >= {need:.1f} GiB")
sys.exit(0 if worst is None or worst >= need else 97)
PYVRAM
    vrc=$?
    if [ "$vrc" -eq 97 ] && [ "${MORI_TEST_IGNORE_VRAM:-0}" != "1" ]; then
      echo "ABORT: not enough free VRAM on at least one visible device."
      echo "  This run was NOT started, so it is not a result in either direction."
      echo "  Override with MORI_TEST_IGNORE_VRAM=1 or lower MORI_SHMEM_HEAP_SIZE."
      exit 97
    fi
    if [ "$mode" = "regress" ]; then
      target="tests/python/ops/test_dispatch_combine_intranode.py"
    elif [ "$mode" = "asyncll" ]; then
      # Its own module because MORI_ENABLE_SDMA has to be set before the
      # session-scoped worker pool spawns -- it is a Context-construction
      # snapshot (context.cpp:51), not a per-op read, and a late set
      # desynchronizes SymmMemManager::Malloc from the transport choice.
      # The module sets it at import; this mode exists so the file can be
      # reached at all, since every other mode hardcodes its target.
      target="tests/python/ops/test_dispatch_combine_role_switch_asyncll.py"
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
