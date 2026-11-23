# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Each ``mori.cco.device.flydsl.buffer`` cache policy reaches the ISA as intended.

A wrong ``aux`` value is silent: the kernel still runs, it just caches where it
must not. So this compiles a load and a store per ``CM_*`` value and reads the
cache bits off the emitted gfx94x/gfx95x instructions. It covers both paths
(``buffer_ops`` on FlyDSL 0.2.x, the raw rocdl ops on 0.3+); nightly runs it on
the older versions.
"""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flydsl")

from mori.cco.device.flydsl import buffer  # noqa: E402

# FlyDSL reads its dump settings at import, so the compile runs in a child. It
# must be a file: the kernel decorator reads the function's source back.
_WORKER = r"""
import glob, json, os, sys
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int64, T
from mori.cco.device.flydsl import buffer

def make(cm):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def cm_probe(addr: Int64):
        rsrc = buffer.create_buffer_resource_from_addr(fx.Int64(addr))
        v = buffer.buffer_load(rsrc, 0, vec_width=1, dtype=T.i32, cache_modifier=cm)
        buffer.buffer_store(v, rsrc, 1, cache_modifier=cm)

    @flyc.jit
    def run(addr: Int64, stream=fx.Stream(None)):
        cm_probe(addr).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)

    return run

def bits(isa, op):
    return [
        sorted(t for t in line.replace(",", " ").split() if t in ("sc0", "sc1", "nt"))
        for line in isa.splitlines()
        if line.strip().startswith(op)
    ]

dump = os.environ["FLYDSL_DUMP_DIR"]
x = torch.zeros(64, dtype=torch.int32, device="cuda")
out = {}
for cm in map(int, sys.argv[1:]):
    pattern = os.path.join(dump, "**", "*_final_isa.s")
    for f in glob.glob(pattern, recursive=True):
        os.remove(f)
    make(cm)(x.data_ptr())
    torch.cuda.synchronize()
    isa = "".join(open(f).read() for f in glob.glob(pattern, recursive=True))
    out[cm] = {"load": bits(isa, "buffer_load_dword"), "store": bits(isa, "buffer_store_dword")}
print("RESULT " + json.dumps(out))
"""

_EXPECTED = {
    buffer.CM_CACHED: [],
    buffer.CM_SC0: ["sc0"],
    buffer.CM_NT: ["nt"],
    buffer.CM_SC1: ["sc1"],
    buffer.CM_SC0_SC1: ["sc0", "sc1"],
}


def test_cache_policy_reaches_the_isa(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to compile for")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if not arch.startswith(("gfx94", "gfx95")):
        pytest.skip(f"sc0/sc1/nt are the gfx94x/gfx95x spellings, not {arch}'s")

    env = dict(
        os.environ,
        FLYDSL_DUMP_IR="1",
        FLYDSL_DEBUG_DUMP_ASM="1",
        FLYDSL_DUMP_DIR=str(tmp_path / "dump"),
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),  # force a compile
    )
    worker = tmp_path / "cm_probe.py"
    worker.write_text(_WORKER)
    proc = subprocess.run(
        [sys.executable, str(worker), *map(str, _EXPECTED)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and result, proc.stdout[-3000:] + proc.stderr[-3000:]

    got = json.loads(result[-1][len("RESULT ") :])
    for cm, want in _EXPECTED.items():
        seen = got[str(cm)]
        assert seen["load"] and seen["store"], f"CM={cm}: no buffer load/store in ISA"
        for op in ("load", "store"):
            assert all(
                b == want for b in seen[op]
            ), f"CM={cm} {op}: {seen[op]} != {want}"
