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
"""Check that an SDMA completion publication covers every submitted queue.

Execute the real kernel body against a queue-state model. This checks the
publication contract; it does not simulate GPU scheduling or DMA latency.
"""

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

KERNEL = Path(__file__).resolve().parents[3] / "python/mori/ops/gemm_ag/kernels_sdma.py"


@lru_cache(maxsize=1)
def drain_body():
    tree = ast.parse(KERNEL.read_text())
    factory = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_sdma_phases"
    )
    builder = next(
        n
        for n in factory.body
        if isinstance(n, ast.FunctionDef) and n.name == "_push_kernel"
    )
    kernel = next(
        n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "push"
    )
    return compile(ast.Module(kernel.body, []), str(KERNEL), "exec")


@pytest.mark.parametrize("queues", [1, 2, 4, 8])
@pytest.mark.parametrize("pushes", [False, True])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_arrival_covers_all_submitted_queues(queues, pushes, rank):
    world = 8
    # The split kernel submits one queue per peer. A chunked fused producer
    # can leave work in every queue, with no ordering between those queues.
    pending = {
        peer: set() if pushes or peer == rank else set(range(queues))
        for peer in range(world)
    }
    published = set()

    class Sdma:
        def put(self, peer, src_win, src_off, dst_win, dst_off, size, qid, **kwargs):
            pending[peer].add(qid)

        def quiet_queue(self, peer, qid):
            pending[peer].discard(qid)

        def quiet(self, peer, **kwargs):
            pending[peer].clear()

    sdma = Sdma()

    def publish(window, flag, peer):
        assert not pending[
            peer
        ], f"arrival for peer {peer} preceded queues {sorted(pending[peer])}"
        published.add(peer)

    for tid in range(world):
        env = dict(
            dev_comm=0,
            win=0,
            rank=rank,
            ws=world,
            queues=queues,
            pushes=pushes,
            my_recv_slot=0,
            slab_bytes=16 << 20,
            signal=False,
            fx=SimpleNamespace(thread_idx=SimpleNamespace(x=tid), Int32=int, Int64=int),
            cco=SimpleNamespace(
                CachedWindow=lambda _: None,
                DevComm=lambda _: SimpleNamespace(sdma=lambda: sdma),
                CoopScope=SimpleNamespace(THREAD=0),
            ),
            const_expr=bool,
            _next_flag=lambda _: (1, None),
            raw_cco=SimpleNamespace(cco_system_fence=lambda _: None),
            _signal_and_wait=publish,
            fgpu=SimpleNamespace(barrier=lambda: None),
            local_store_u32=lambda *args: None,
        )
        exec(drain_body(), env)  # noqa: S102 - execute trusted local kernel source
    assert published == set(range(world))
