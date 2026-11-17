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
import gc
import importlib.machinery
import logging
import os
import sys
import time
import types

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
import mori.allocator as mori_allocator
from mori.allocator import (  # importing registers "MORI"
    handle_type,
    signal_pad_supported,
    window_handle,
)

from tests.python.utils import TorchDistContext, get_free_port

GiB = 1 << 30


def _round_up(v, m):
    return (v + m - 1) // m * m


def _run(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group_name = dist.group.WORLD.group_name

        symm_mem.set_backend("MORI")
        assert symm_mem.get_backend(device) == "MORI"
        assert handle_type(rank) in ("fabric", "posix_fd")

        n = 1024
        t = symm_mem.empty(n, dtype=torch.float32, device=device)
        t.fill_(float(rank + 1))
        torch.cuda.synchronize()

        cpus = os.sched_getaffinity(0)
        hdl = symm_mem.rendezvous(t, group_name)
        assert hdl.world_size == world_size and hdl.rank == rank
        # The backend's comm must leave the caller's CPU affinity alone.
        assert os.sched_getaffinity(0) == cpus, "rendezvous rebound the calling thread"

        # Peers come back the torch way, as one base address per rank; this rank's own
        # entry is the tensor itself rather than cco's alias of it.
        ptrs = hdl.buffer_ptrs
        assert len(ptrs) == world_size and all(p for p in ptrs)
        assert ptrs[rank] == t.data_ptr()

        # Read every peer and check its sentinel.
        for pe in range(world_size):
            peer = hdl.get_buffer(pe, (4,), torch.float32)
            assert abs(peer[0].item() - (pe + 1)) < 1e-6, f"peer {pe} mismatch"

        # The signal pad is its own cco window now, so it is always there and torch's
        # own collectives -- which synchronise through it -- work unconditionally.
        assert signal_pad_supported()
        pads = hdl.signal_pad_ptrs
        assert len(pads) == world_size and all(p for p in pads)

        expect = float(sum(r + 1 for r in range(world_size)))
        out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        torch.cuda.synchronize()
        assert abs(out[0].item() - expect) < 1e-3

        # The backend's own barrier, a kernel on the current stream.
        hdl.barrier(0)

        # Backend-specific escape hatch: the cco window a kernel can address through.
        # torch's handle has no slot for it, same as NCCL's get_window().
        assert window_handle(t) != 0
        assert window_handle(t, signal_pad=True) != 0
        # The wrapper resolves a view to its storage, so both name the same window.
        # (This pins the Python side; the C++ lookup only ever sees the storage base.)
        assert window_handle(t[n // 2 :]) == window_handle(t)

        # A DevComm is parameterised by the algorithm that will run, so it is cached per
        # (group, key) rather than per group -- two collectives want two of them.
        from mori.allocator import dev_comm

        a = dev_comm(group_name, key="all2all")
        assert a != 0
        assert dev_comm(group_name, key="all2all") == a, "same key must reuse"
        assert dev_comm(group_name, key="reduce") != a, "different key must not share"
        # The key is supposed to encode the parameterisation, so asking for a bigger
        # barrier array under a key that already exists must complain rather than hand
        # back the smaller one.
        with pytest.raises(RuntimeError, match="already exists"):
            dev_comm(group_name, key="all2all", lsa_barrier_count=8)
        with pytest.raises(RuntimeError, match="rendezvous"):
            dev_comm("no-such-group")

        dist.barrier()


def _run_release(rank, world_size, port):
    """Releasing a rendezvous'd window must both survive and give the memory back.

    Teardown used to segfault, so it was disabled and every window leaked; this pins
    down both halves of that.
    """
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group_name = dist.group.WORLD.group_name
        symm_mem.set_backend("MORI")

        mib = 1 << 20
        rounds = 4
        # Same ranks, second communicator: each group imports the allocation separately.
        groups = [group_name, dist.new_group(list(range(world_size))).group_name]

        def cycle(ngroups):
            t = symm_mem.empty(16 * mib // 4, dtype=torch.float32, device=device)
            hdls = [symm_mem.rendezvous(t, g) for g in groups[:ngroups]]
            assert all(h.world_size == world_size for h in hdls)
            dist.barrier()
            del hdls, t
            gc.collect()
            torch.cuda.synchronize()
            dist.barrier()

        def measure(ngroups):
            cycle(ngroups)  # first cycle also pays any one-off context growth
            settled = torch.cuda.mem_get_info(device)[0]
            for _ in range(rounds):
                cycle(ngroups)
            return settled - torch.cuda.mem_get_info(device)[0]

        # The control isolates who is at fault. Plain alloc/free exercises only
        # hipMemCreate/hipMemRelease; adding rendezvous brings in the shareable-fd
        # export and import, whose fd ownership differs between ROCm releases. If both
        # leak, the pairing itself is broken rather than anything about sharing.
        plain = measure(0)
        shared = measure(1)
        two = measure(2)

        # mem_get_info is device-wide, so anything else sharing the GPU moves it too.
        # Leaking would cost rounds * 16 MiB; half of one window is a wide enough margin
        # to stay clear of that noise while still failing loudly on a real leak.
        report = (
            f"[rank {rank}] hip={torch.version.hip} world={world_size} "
            f"lost over {rounds} x 16 MiB cycles: alloc/free only={plain / mib:.1f} MiB, "
            f"with rendezvous={shared / mib:.1f} MiB, with two groups={two / mib:.1f} MiB"
        )
        print(report, flush=True)
        assert plain < 8 * mib, f"plain alloc/free leaks, before any sharing. {report}"
        assert shared < 8 * mib, f"the rendezvous'd window leaks. {report}"
        assert two < 8 * mib, f"a second group's import leaks. {report}"


def _run_contract(rank, world_size, port):
    """The parts of torch's backend contract that torch's own suite leans on."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        world = dist.group.WORLD
        symm_mem.set_backend("MORI")
        ext = mori_allocator._ext()
        n = 1024

        # "cuda" without an index is the current device, as with torch's CUDA backend.
        t = symm_mem.empty(n, dtype=torch.float32, device="cuda")
        assert t.device == device
        t.fill_(float(rank + 1))
        torch.cuda.synchronize()

        # Membership: is_symm_mem_tensor follows the allocator, views included, and a
        # foreign tensor gives a null handle instead of an error.
        plain = torch.zeros(n, device=device)
        assert symm_mem.is_symm_mem_tensor(t)
        assert symm_mem.is_symm_mem_tensor(t[n // 2 :])
        assert not symm_mem.is_symm_mem_tensor(plain)
        assert symm_mem.rendezvous(plain, world.group_name) is None

        hdl = symm_mem.rendezvous(t, world.group_name)
        assert hdl.buffer_ptrs[rank] == t.data_ptr()
        assert hdl.get_buffer(rank, (4,), torch.float32).data_ptr() == t.data_ptr()
        assert hdl.offset == 0
        if hasattr(hdl, "group_name"):  # torch >= 2.16
            assert hdl.group_name == world.group_name
        assert ext._rank_to_global_rank(t.data_ptr(), world.group_name) == list(
            range(world_size)
        )
        dist.barrier()
        for pe in range(world_size):
            assert hdl.get_buffer(pe, (4,), torch.float32)[0].item() == pe + 1

        # A small team maps device memory per rank, which shows as the peer stride.
        peers = [r for r in range(world_size) if r != rank]
        stride = (hdl.buffer_ptrs[peers[1]] - hdl.buffer_ptrs[peers[0]]) // (
            peers[1] - peers[0]
        )
        total = torch.cuda.get_device_properties(device).total_memory
        assert stride == _round_up(total, 4 * GiB), (stride, total)

        # Another group over the same ranks: its own handle, pad and window, cached per
        # group rather than per allocation.
        dup = dist.new_group(list(range(world_size)))
        hdl2 = symm_mem.rendezvous(t, dup)
        assert hdl2.signal_pad_ptrs[rank] != hdl.signal_pad_ptrs[rank]
        assert hdl2.buffer_ptrs[rank] == t.data_ptr()
        for pe in range(world_size):
            assert hdl2.get_buffer(pe, (4,), torch.float32)[0].item() == pe + 1
        if hasattr(hdl2, "group_name"):
            assert hdl2.group_name == dup.group_name
        again = symm_mem.rendezvous(t, world.group_name)
        assert again.signal_pad_ptrs == hdl.signal_pad_ptrs
        with pytest.raises(RuntimeError, match="say which one"):
            window_handle(t)
        assert window_handle(t, group=world) != window_handle(t, group=dup)
        assert window_handle(t, group=dup.group_name) == window_handle(t, group=dup)

        # A pad write through one group stays out of the other's.
        hdl.get_signal_pad(rank).fill_(7)
        torch.cuda.synchronize()
        assert hdl2.get_signal_pad(rank).eq(0).all()
        hdl.get_signal_pad(rank).zero_()
        torch.cuda.synchronize()

        # Halves: a subgroup sees its own ranks, and knows their global ranks.
        ranks = list(range(world_size))
        lo = dist.new_group(ranks[: world_size // 2])
        hi = dist.new_group(ranks[world_size // 2 :])
        mine = lo if rank < world_size // 2 else hi
        members = (
            ranks[: world_size // 2]
            if rank < world_size // 2
            else ranks[world_size // 2 :]
        )
        sub = symm_mem.rendezvous(t, mine)
        assert sub.world_size == len(members) and sub.rank == members.index(rank)
        assert ext._rank_to_global_rank(t.data_ptr(), mine.group_name) == members
        for i, pe in enumerate(members):
            assert sub.get_buffer(i, (4,), torch.float32)[0].item() == pe + 1

        # MemPool tensors start inside a block: the handle reports where.
        if hasattr(symm_mem, "get_mem_pool") and hasattr(torch.cuda, "use_mem_pool"):
            pool = symm_mem.get_mem_pool(device)
            with torch.cuda.use_mem_pool(pool):
                a = torch.empty(n, dtype=torch.float32, device=device)
                b = torch.empty(n, dtype=torch.float32, device=device)
            assert symm_mem.is_symm_mem_tensor(a) and symm_mem.is_symm_mem_tensor(b)
            b.fill_(float(rank + 10))
            torch.cuda.synchronize()
            hb = symm_mem.rendezvous(b, world.group_name)
            assert hb.offset == b.data_ptr() - hb.buffer_ptrs[rank]
            dist.barrier()
            for pe in range(world_size):
                remote = hb.get_remote_tensor(pe, (n,), torch.float32)
                assert remote[0].item() == pe + 10
            dist.barrier()
            del hb, a, b

        # alloc and a first rendezvous refuse graph capture; a cached rendezvous is fine.
        fresh = symm_mem.empty(n, dtype=torch.float32, device=device)
        x = torch.zeros(1, device=device)
        errors = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for fn in (
                lambda: symm_mem.empty(n, dtype=torch.float32, device=device),
                lambda: symm_mem.rendezvous(fresh, world.group_name),
            ):
                try:
                    fn()
                except RuntimeError as exc:
                    errors.append(str(exc))
            symm_mem.rendezvous(t, world.group_name)
            x.add_(1)
        assert len(errors) == 2 and all("capturing" in e for e in errors), errors
        graph.replay()
        torch.cuda.synchronize()
        assert x.item() == 1
        symm_mem.rendezvous(fresh, world.group_name)  # still collective-consistent

        # alloc works on the caller's stream: it must not wait for the default stream.
        side = torch.cuda.Stream()
        torch.cuda._sleep(1 << 20)
        torch.cuda.synchronize()
        start = time.time()
        torch.cuda._sleep(1 << 26)
        torch.cuda.synchronize()
        cycles_per_s = (1 << 26) / max(time.time() - start, 1e-6)
        torch.cuda._sleep(int(2 * cycles_per_s))  # ~2 s on the default stream
        with torch.cuda.stream(side):
            kept = symm_mem.empty(n, dtype=torch.float32, device=device)
        assert (
            not torch.cuda.default_stream().query()
        ), "alloc waited on the default stream"
        torch.cuda.synchronize()
        del kept

        dist.barrier()


def _run_overlapping_devices(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        torch.cuda.set_device(rank)
        world = dist.group.WORLD
        symm_mem.set_backend("MORI")
        t = symm_mem.empty(64, dtype=torch.float32, device="cuda:0")
        os.environ.pop("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", None)
        with pytest.raises(RuntimeError, match="overlapping devices"):
            symm_mem.rendezvous(t, world.group_name)

        os.environ["TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"] = "1"
        hdl = symm_mem.rendezvous(t, world.group_name)
        assert hdl.buffer_ptrs[rank] == t.data_ptr()
        t.fill_(float(rank + 1))
        torch.cuda.synchronize()
        dist.barrier()
        for pe in range(world_size):
            assert hdl.get_buffer(pe, (4,), torch.float32)[0].item() == pe + 1
        dist.barrier()


class _Array:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {
            "shape": (n,),
            "typestr": "<f4",
            "data": (ptr, False),
            "version": 3,
        }


def _run_two_comms(rank, world_size, port):
    """One allocation imported into two cco communicators at once, the premise of
    per-group handles. X={0,1} and Y={0,2} share their rank 0, and rank 1 registers
    with X while ranks 0 and 2 are registering with Y."""
    from mori.cco.communicator import Communicator, UniqueId, get_unique_id

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        symm_mem.set_backend("MORI")
        per_rank = 4 * GiB
        n = 1 << 16
        t = symm_mem.empty(n, dtype=torch.float32, device=device)
        t.fill_(float(rank + 1))
        torch.cuda.synchronize()

        def make(members):
            uid = [bytes(get_unique_id()) if rank == members[0] else None]
            dist.broadcast_object_list(uid, src=members[0])
            if rank not in members:
                return None
            uid = UniqueId.from_bytes(uid[0])
            return Communicator.init(len(members), members.index(rank), uid, per_rank)

        comms = {"x": make([0, 1]), "y": make([0, 2])}
        dist.barrier()
        wins = {}
        if rank == 1:
            wins["x"] = comms["x"].register_external_window(t.data_ptr(), n * 4)
        if rank in (0, 2):
            if rank == 0:
                time.sleep(2)
            wins["y"] = comms["y"].register_external_window(t.data_ptr(), n * 4)
        if rank == 0:
            wins["x"] = comms["x"].register_external_window(t.data_ptr(), n * 4)
        torch.cuda.synchronize()
        dist.barrier()
        for name, win in wins.items():
            members = [0, 1] if name == "x" else [0, 2]
            me = members.index(rank)
            for i, pe in enumerate(members):
                ptr = win.local_ptr + (i - me) * per_rank
                got = torch.as_tensor(_Array(ptr, n), device=device)[7].item()
                assert got == pe + 1, (name, pe, got)
        dist.barrier()
        # Dropping one comm leaves the other's mapping intact.
        if rank == 0:
            wins.pop("x").close()
            comms.pop("x").destroy()
            got = torch.as_tensor(
                _Array(wins["y"].local_ptr + per_rank, n), device=device
            )
            assert got[3].item() == 3
        dist.barrier()
        for win in wins.values():
            win.close()
        for comm in comms.values():
            if comm is not None:
                comm.destroy()
        dist.barrier()


def _run_restart(rank, world_size, port, tmpdir):
    """A communicator goes with its process group, and a successor reusing the
    group's name gets a working one (torch's NCCL Restart test, for this backend)."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    ext = mori_allocator._ext()

    def init(tag):
        store = dist.FileStore(os.path.join(tmpdir, tag), world_size)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
            store=store,
            device_id=device,
        )
        dist.all_reduce(torch.ones(1, device=device))

    def roundtrip():
        t = symm_mem.empty(4096, dtype=torch.float32, device=device)
        hdl = symm_mem.rendezvous(t, dist.group.WORLD.group_name)
        t.fill_(float(rank))
        torch.cuda.synchronize()
        hdl.barrier()
        peer = (rank + 1) % world_size
        assert hdl.get_buffer(peer, (4096,), torch.float32).eq(peer).all()
        hdl.barrier()
        return t, hdl

    init("a")
    symm_mem.set_backend("MORI")
    base = ext._live_comms()
    t, hdl = roundtrip()
    assert ext._live_comms() == base + 1
    # Handle dropped first, then the group, then the tensor.
    del hdl
    gc.collect()
    dist.destroy_process_group()
    assert ext._live_comms() == base + 1
    del t
    gc.collect()
    assert ext._live_comms() == base

    init("b")
    t, hdl = roundtrip()
    assert ext._live_comms() == base + 1
    # Tensor first, then the group; the handle is what still holds the comm.
    del t
    gc.collect()
    dist.destroy_process_group()
    assert ext._live_comms() == base + 1
    del hdl
    gc.collect()
    assert ext._live_comms() == base

    init("c")
    t, hdl = roundtrip()
    del t, hdl
    gc.collect()
    dist.destroy_process_group()


def _spawn(fn, world_size, *args):
    port = get_free_port()
    mp.spawn(fn, args=(world_size, port, *args), nprocs=world_size, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_backend():
    _spawn(_run, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs at least 4 GPUs")
def test_symm_release():
    # 4 ranks on purpose: the teardown crash did not show at 2 on every torch.
    _spawn(_run_release, 4)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs at least 4 GPUs")
def test_symm_contract():
    _spawn(_run_contract, 4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_overlapping_devices():
    _spawn(_run_overlapping_devices, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="needs at least 3 GPUs")
def test_one_allocation_two_comms():
    _spawn(_run_two_comms, 3)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_restart(tmp_path):
    port = get_free_port()
    mp.spawn(_run_restart, args=(2, port, str(tmp_path)), nprocs=2, join=True)


def test_torch_fingerprint_tracks_build():
    from mori.allocator._build import torch_fingerprint

    def fake(version, git, hip="7.14", abi=True):
        return types.SimpleNamespace(
            __version__=version,
            version=types.SimpleNamespace(git_version=git, hip=hip),
            _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=abi),
        )

    base = torch_fingerprint(fake("2.16.0", "a"))
    assert base == torch_fingerprint(fake("2.16.0", "a"))
    assert base != torch_fingerprint(fake("2.16.0", "b"))
    assert base != torch_fingerprint(fake("2.15.0", "a"))
    assert base != torch_fingerprint(fake("2.16.0", "a", hip="10.0"))
    assert base.isalnum()


def test_build_links_loaded_hip_runtime(tmp_path, monkeypatch):
    from mori.allocator._build import hip_link_dir, hip_runtime_library, rocm_home

    lib = hip_runtime_library("/nonexistent")
    assert lib is not None and os.path.basename(lib).startswith("libamdhip64.so")
    # torch's own -lamdhip64 must resolve to that runtime, also where only .so.N ships.
    for _ in range(2):  # rebuilds reuse the directory
        link_dir = hip_link_dir("/nonexistent", str(tmp_path / "hip"))
        link = os.path.join(link_dir, "libamdhip64.so")
        assert os.path.realpath(link) == os.path.realpath(lib)

    # A ROCM_HOME without HIP headers (a venv's hipcc wrapper) falls through.
    good = tmp_path / "rocm"
    (good / "include" / "hip").mkdir(parents=True)
    (good / "include" / "hip" / "hip_runtime_api.h").touch()
    monkeypatch.setenv("ROCM_PATH", str(good))
    assert rocm_home(types.SimpleNamespace(ROCM_HOME=str(tmp_path))) == str(good)


def test_prebuilt_for_other_torch_is_refused(monkeypatch, caplog):
    import mori

    fake = types.ModuleType("mori.mori_torch_symm")
    fake.__spec__ = importlib.machinery.ModuleSpec("mori.mori_torch_symm", None)
    fake.torch_fingerprint = "0123456789abcdef"
    monkeypatch.setitem(sys.modules, "mori.mori_torch_symm", fake)
    monkeypatch.setattr(mori, "mori_torch_symm", fake, raising=False)
    with caplog.at_level(logging.INFO, logger="mori.allocator"):
        ext, reason = mori_allocator._prebuilt_ext("fedcba9876543210")
    assert ext is None and "another torch build" in str(reason)
    assert any("another torch build" in r.getMessage() for r in caplog.records)
    ext, reason = mori_allocator._prebuilt_ext("0123456789abcdef")
    assert ext is fake and reason is None


def test_registration_failure_warns(monkeypatch, caplog):
    def broken():
        raise ImportError("no compiler")

    monkeypatch.setattr(mori_allocator, "_ext", broken)
    with caplog.at_level(logging.WARNING, logger="mori.allocator"):
        mori_allocator._register_on_import()
    assert any(
        r.levelno == logging.WARNING and "no compiler" in r.getMessage()
        for r in caplog.records
    )
