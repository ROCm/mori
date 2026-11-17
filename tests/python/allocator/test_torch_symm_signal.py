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
"""barrier / put_signal / wait_signal of the MORI torch SymmetricMemory backend."""
import os
import resource
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
import mori.allocator  # noqa: F401  (importing registers "MORI")

from tests.python.utils import TorchDistContext, get_free_port

TORCH_VERSION = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])


def _setup(rank, n):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    symm_mem.set_backend("MORI")
    t = symm_mem.empty(n, dtype=torch.float32, device=device)
    hdl = symm_mem.rendezvous(t, dist.group.WORLD.group_name)
    return device, t, hdl


def _assert_pad_zero(hdl, rank):
    torch.cuda.synchronize()
    dist.barrier()
    pad = hdl.get_signal_pad(rank)
    assert (
        torch.count_nonzero(pad).item() == 0
    ), "a balanced exchange left the pad dirty"


def _run_signal(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        n = 1 << 16
        device, t, hdl = _setup(rank, n)
        group_name = dist.group.WORLD.group_name
        nxt, prv = (rank + 1) % world_size, (rank - 1) % world_size
        max_channel = hdl.signal_pad_size // 4 // world_size
        peer = hdl.get_buffer(nxt, (n,), torch.float32)

        # A barrier orders this rank's fill before the peer's read, on any channel.
        for it, ch in enumerate([0, 1, max_channel - 1]):
            t.fill_(rank * 100.0 + it)
            hdl.barrier(channel=ch)
            got = peer.clone()
            hdl.barrier(channel=ch)  # the peer is done reading before the next fill
            assert torch.all(got == nxt * 100.0 + it), f"channel {ch}"

        # put_signal publishes a remote write; wait_signal makes it visible.
        src = torch.full((n,), float(rank + 1), device=device)
        for it in range(3):
            peer.copy_(src + it)
            hdl.put_signal(nxt, channel=2)
            hdl.wait_signal(prv, channel=2)
            assert torch.all(t == prv + 1.0 + it)
            hdl.barrier()

        # Shares the pad with torch's own collectives on the same stream.
        t.fill_(rank + 1.0)
        expect = float(sum(r + 1 for r in range(world_size)))
        for _ in range(10):
            hdl.barrier()
            out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        torch.cuda.synchronize()
        assert torch.all(out == expect)

        # Back-to-back, no host sync in between.
        for i in range(500):
            hdl.barrier(channel=i % 8)
        for i in range(500):
            hdl.put_signal(nxt, channel=i % 8)
            hdl.wait_signal(prv, channel=i % 8)
        _assert_pad_zero(hdl, rank)

        # Rejected on the host, with torch's wording.
        with pytest.raises(RuntimeError, match="maximum supported channel"):
            hdl.barrier(channel=max_channel)
        with pytest.raises(RuntimeError, match="maximum supported channel"):
            hdl.put_signal(nxt, channel=max_channel)
        with pytest.raises(RuntimeError, match="maximum supported channel"):
            hdl.wait_signal(prv, channel=max_channel)
        with pytest.raises(RuntimeError, match="non-negative"):
            hdl.barrier(channel=-1)
        for bad in (-1, world_size):
            with pytest.raises(RuntimeError, match=r"must be in \[0"):
                hdl.put_signal(bad)
            with pytest.raises(RuntimeError, match=r"must be in \[0"):
                hdl.wait_signal(bad)


def _run_capture(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        n = 1024
        device, t, hdl = _setup(rank, n)
        nxt, prv = (rank + 1) % world_size, (rank - 1) % world_size
        peer = hdl.get_buffer(nxt, (n,), torch.float32)
        got = torch.zeros(n, device=device)
        counter = torch.zeros(1, device=device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            hdl.barrier(channel=1)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            counter.add_(1.0)
            t.copy_((counter * 10.0 + rank).expand(n))
            hdl.barrier(channel=1)
            got.copy_(peer)
            hdl.put_signal(nxt, channel=2)  # done reading nxt
            hdl.wait_signal(prv, channel=2)  # prv is done reading this rank

        replays = 5
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
        assert torch.all(got == replays * 10.0 + nxt)
        _assert_pad_zero(hdl, rank)


def _run_two_streams(rank, world_size, port):
    """Same-channel barriers on two streams must not consume each other's signals."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device, t, hdl = _setup(rank, 64)
        peer = hdl.get_buffer((rank + 1) % world_size, (64,), torch.float32)
        a, b = torch.cuda.Stream(), torch.cuda.Stream()
        for it in range(3):
            with torch.cuda.stream(a):
                if rank == 0:
                    torch.cuda._sleep(100_000_000)
                t.fill_(rank + 1.0)
                hdl.barrier(channel=0)
            with torch.cuda.stream(b):
                hdl.barrier(channel=0)
                t.fill_(rank + 10.0 + it)
            torch.cuda.synchronize()
            dist.barrier()
            want = (rank + 1) % world_size + 10.0 + it
            assert torch.all(peer == want), f"round {it}"
            dist.barrier()


def _run_timeout(rank, world_size, port, op):
    """Rank 0 times out on a peer that never answers; rank 1 stays alive meanwhile."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    _, _, hdl = _setup(rank, 16)
    store = dist.distributed_c10d._get_default_store()
    dist.barrier()
    if rank != 0:
        store.wait(["done"], timedelta(seconds=120))
        os._exit(0)
    start = time.monotonic()
    try:
        if op == "barrier":
            hdl.barrier(timeout_ms=500)
        elif op == "put_signal":
            hdl.put_signal(1)
            hdl.put_signal(1, timeout_ms=500)  # the first one is never consumed
        else:
            hdl.wait_signal(1, timeout_ms=500)
        torch.cuda.synchronize()
        code = 3  # returned: the timeout never fired
    except RuntimeError:
        code = 0 if time.monotonic() - start < 30 else 4
    store.set("done", "1")
    # The trapped context cannot be torn down cleanly.
    os._exit(code)


def _spawn(fn, world_size, *args):
    mp.spawn(
        fn, args=(world_size, get_free_port(), *args), nprocs=world_size, join=True
    )


def _worlds():
    n = torch.cuda.device_count()
    return [w for w in (2, 4) if w <= n] or [2]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
@pytest.mark.parametrize("world_size", _worlds())
def test_signal_ops(world_size):
    _spawn(_run_signal, world_size)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_signal_graph_capture():
    _spawn(_run_capture, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
@pytest.mark.skipif(
    TORCH_VERSION < (2, 15),
    reason="torch orders signal-pad ops across streams from 2.15",
)
def test_signal_two_streams():
    _spawn(_run_two_streams, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
@pytest.mark.parametrize("op", ["barrier", "put_signal", "wait_signal"])
def test_signal_timeout(op, capfd):
    try:
        _spawn(_run_timeout, 2, op)
    except mp.ProcessExitedException as e:
        # Some ROCm releases abort the process on a trap instead of failing the stream.
        assert e.error_index == 0 and e.signal_name == "SIGABRT", str(e)
        return
    assert f"[FATAL] SymmetricMemory::{op}" in capfd.readouterr().out


def test_sources_are_hipify_neutral(tmp_path):
    """torch's build hipifies these and writes a *_hip copy of any it would change."""
    hipify_python = pytest.importorskip("torch.utils.hipify.hipify_python")
    from mori.jit.config import get_mori_source_root

    root = get_mori_source_root()
    if root is None:
        pytest.skip("mori sources not found")
    src = Path(root) / "src" / "allocator"
    copies = []
    for p in sorted(src.iterdir()):
        if p.suffix in (".cpp", ".hip", ".hpp"):
            (tmp_path / p.name).write_bytes(p.read_bytes())
            copies.append(str(tmp_path / p.name))
    sources = [c for c in copies if not c.endswith(".hpp")]
    out = tmp_path / "out"
    out.mkdir()
    result = hipify_python.hipify(
        project_directory=str(out),
        output_directory=str(out),
        extra_files=sources,
        is_pytorch_extension=True,
        show_detailed=False,
        show_progress=False,
    )
    for s in sources:
        assert (
            result[s].hipified_path == s
        ), f"hipify rewrites {Path(s).name}: {result[s].status}"


def test_jit_build_leaves_sources_alone():
    """Loading the extension must not hipify mori's or ROCm's headers in place."""
    from mori.allocator import _ext
    from mori.jit.config import get_mori_source_root

    _ext()
    root = get_mori_source_root()
    if root is None:
        pytest.skip("mori sources not found")
    stray = [
        str(p)
        for d in ("src/allocator", "include")
        for p in (Path(root) / d).rglob("*_hip.*")
    ]
    assert not stray, f"hipify wrote copies into the source tree: {stray[:5]}"
