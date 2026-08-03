#!/usr/bin/env bash
# How is mori installed in the candidate image, and can the node reach github?
#
# The f01-2 arrangement runs the repo tree, not the wheel: edits to python/mori/... and to the
# kernel .hpp both take effect, which only happens with an editable install (or a JIT source path
# that points at the checkout). Whatever that arrangement is has to be reproduced here, otherwise
# a run on f01-1 measures the image's mori and not this branch -- the same class of mistake as the
# build-cache key, and just as invisible in the output.
set -uo pipefail
IMG="${IMG:-rocm/fw-bringup:gfx1250-atom-dev-20260729-update-compiler}"
docker run --rm --entrypoint bash "$IMG" -lc '
python3 - <<EOF
import mori, os
print("mori file:", mori.__file__)
print("mori version:", getattr(mori, "__version__", "?"))
EOF
echo "--- package dir ---"
d=$(python3 -c "import mori,os;print(os.path.dirname(mori.__file__))")
ls "$d" | head -20
echo "--- jit sources ---"
ls "$d/_jit-sources" 2>/dev/null | head -10 || echo "(none)"
ls "$d/_jit-sources/src/ops/dispatch_combine" 2>/dev/null | head -10 || echo "(no dispatch_combine)"
echo "--- mori_cpp ---"
python3 -c "import mori_cpp; print(mori_cpp.__file__)" 2>&1 | tail -2
python3 -c "import mori_cpp; print([n for n in dir(mori_cpp) if \"build_args\" in n or \"prepare\" in n])" 2>&1 | tail -2
echo "--- dist-info ---"
ls -d /usr/local/lib/python3.12/dist-packages/mori* 2>/dev/null | head
' 2>&1 | head -50
echo "== github reachable from host =="
timeout 40 git ls-remote https://github.com/ROCm/mori.git HEAD 2>&1 | head -2
echo "F1PROBE3_DONE"
