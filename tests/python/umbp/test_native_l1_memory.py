# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU lifecycle checks for the real native L1 allocation helper.

Torch and allocation RPCs are controlled boundaries, never alternate helper
implementations. The last tests exercise the real native client/server fence.
"""

import contextlib
import gc
import importlib.util
import sys
import uuid
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[3] / "python/mori/umbp_l1/__init__.py"
    spec = importlib.util.spec_from_file_location("_tested_native_l1_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    # Tests use fake byte buffers only. No exported GPU memory is discarded.
    module._LIVE_NATIVE_L1_MEMORY.clear()


@pytest.fixture
def runtime(monkeypatch):
    calls = []

    class Device:
        def __init__(self, value, index=None):
            if isinstance(value, Device):
                self.type, self.index = value.type, value.index
            else:
                parts = str(value).split(":")
                self.type = parts[0]
                self.index = (
                    index
                    if index is not None
                    else int(parts[1]) if len(parts) == 2 else None
                )

    class Tensor:
        def __init__(self, count, device):
            self.data = bytearray([255]) * count
            self.device = device

        def numel(self):
            return len(self.data)

        def data_ptr(self):
            return 0x100000

    class MemPool:
        pass

    @contextlib.contextmanager
    def scope(kind, value):
        calls.append((kind, id(value) if kind == "allocator" else value))
        yield

    def empty(nbytes, *, dtype, device):
        calls.append(("empty", nbytes))
        assert dtype == "uint8"
        return Tensor(nbytes, device)

    def zero(arena, value, *, out):
        assert arena is out and value == 0
        calls.append(("in_place_zero", arena.data_ptr()))
        arena.data[:] = bytes(arena.numel())

    def synchronize(device):
        calls.append(("synchronize", device.index))

    def export(ptr, nbytes, device):
        calls.append(("export", ptr, nbytes, device))
        return b"fake-export-descriptor"

    torch = ModuleType("torch")
    torch.device, torch.uint8, torch.empty, torch.bitwise_and = (
        Device,
        "uint8",
        empty,
        zero,
    )
    torch.cuda = SimpleNamespace(
        device=lambda d: scope("device", d.index),
        use_mem_pool=lambda p: scope("allocator", p),
        current_device=lambda: 1,
        device_count=lambda: 4,
        synchronize=synchronize,
    )
    io = ModuleType("mori.io")
    io.make_fabric_mem_pool = MemPool
    native = ModuleType("mori_umbp_l1")
    native.NATIVE_L1_FABRIC_VERSION = 1
    native.export_fabric_memory = export
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "mori.io", io)
    monkeypatch.setitem(sys.modules, "mori_umbp_l1", native)
    return SimpleNamespace(calls=calls, torch=torch, native=native, Tensor=Tensor)


class Client:
    def __init__(self, calls):
        self.calls = calls
        self.fail_attach = self.fail_bind = self.fail_detach = False

    def attach(self, *args, **kwargs):
        self.calls.append(("attach", args, kwargs))
        if self.fail_attach:
            raise RuntimeError("Attach response lost after server accepted")

    def bind_layout(self, *args):
        self.calls.append(("bind", args))
        if self.fail_bind:
            raise RuntimeError("BindLayout response lost")

    def detach(self, pool_id):
        self.calls.append(("detach", pool_id))
        if self.fail_detach:
            raise RuntimeError("detach: owner incarnation mismatch")


def allocate(helper, runtime):
    client = Client(runtime.calls)
    owner = helper.NativeL1Memory(client, "pool", "model-format")
    arena = owner.allocate(1024, "cuda")
    return client, owner, arena


def test_final_arena_zeroed_in_place_and_registered_after_sync(helper, runtime):
    _, owner, arena = allocate(helper, runtime)
    assert arena.data == bytes(1024)
    assert arena.device.index == 1
    owner.bind_layout(arena, 4, [(0, 32), (160, 17)])
    names = [call[0] for call in runtime.calls]
    assert (
        names.index("in_place_zero")
        < names.index("synchronize")
        < names.index("export")
        < names.index("attach")
    )
    exported = next(c for c in runtime.calls if c[0] == "export")
    assert exported[1:] == (arena.data_ptr(), 1024, 1)
    attached = next(c for c in runtime.calls if c[0] == "attach")
    assert attached[2] == {
        "external_handle": b"fake-export-descriptor",
        "external_offset": 0,
        "external_kind": "hip_fabric",
    }
    assert helper._LIVE_NATIVE_L1_MEMORY[id(owner)] is owner
    allocator = weakref.ref(arena._mori_native_l1_mem_pool)
    owner.close()
    owner.close()
    assert [c[0] for c in runtime.calls].count("detach") == 1
    assert id(owner) not in helper._LIVE_NATIVE_L1_MEMORY
    del owner
    gc.collect()
    assert allocator() is not None  # The returned arena still owns its allocator.


@pytest.mark.parametrize("failure", ["fail_attach", "fail_bind"])
def test_uncertain_registration_retains_owner_without_caller_refs(
    helper, runtime, failure
):
    client, owner, arena = allocate(helper, runtime)
    setattr(client, failure, True)
    with pytest.raises(
        helper.L1MemoryInDoubtError, match="registration was not confirmed"
    ):
        owner.bind_layout(arena, 4, [(0, 32)])
    owner_ref, arena_ref = weakref.ref(owner), weakref.ref(arena)
    allocator_ref = weakref.ref(arena._mori_native_l1_mem_pool)
    del owner, arena
    gc.collect()
    assert (
        owner_ref() is not None
        and arena_ref() is not None
        and allocator_ref() is not None
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        owner_ref().bind_layout(arena_ref(), 4, [(0, 32)])
    owner_ref().close()
    gc.collect()
    assert owner_ref() is None and arena_ref() is None and allocator_ref() is None


def test_failed_incarnation_fence_retains_until_confirmed_detach(helper, runtime):
    client, owner, arena = allocate(helper, runtime)
    owner.bind_layout(arena, 4, [(0, 32)])
    client.fail_detach = True
    with pytest.raises(
        helper.L1MemoryInDoubtError, match="safe close was not confirmed"
    ):
        owner.close()
    assert helper._LIVE_NATIVE_L1_MEMORY[id(owner)] is owner
    with pytest.raises(RuntimeError, match="closing or closed"):
        owner.allocate(32, 0)
    with pytest.raises(RuntimeError, match="closing or closed"):
        owner.bind_layout(arena, 4, [(0, 32)])
    client.fail_detach = False
    owner.close()
    assert id(owner) not in helper._LIVE_NATIVE_L1_MEMORY


def test_failed_owner_gpu_drain_does_not_send_detach(helper, runtime, monkeypatch):
    _, owner, arena = allocate(helper, runtime)
    owner.bind_layout(arena, 4, [(0, 32)])

    def fail(_):
        raise RuntimeError("GPU work completion unknown")

    monkeypatch.setattr(runtime.torch.cuda, "synchronize", fail)
    with pytest.raises(helper.L1MemoryInDoubtError):
        owner.close()
    assert not any(c[0] == "detach" for c in runtime.calls)
    assert helper._LIVE_NATIVE_L1_MEMORY[id(owner)] is owner


@pytest.mark.parametrize(
    "regions", [[], [(0, 0)], [(0, 300)], [(0, 32), (64, 32)], [(-1, 32)]]
)
def test_invalid_layout_cannot_publish_descriptor(helper, runtime, regions):
    _, owner, arena = allocate(helper, runtime)
    with pytest.raises(ValueError):
        owner.bind_layout(arena, 4, regions)
    assert not any(c[0] in ("export", "attach") for c in runtime.calls)
    assert id(owner) not in helper._LIVE_NATIVE_L1_MEMORY
    owner.close()


def test_native_failed_fabric_attach_fences_detach_without_incarnation(tmp_path):
    native = pytest.importorskip("mori_umbp_l1")
    address = f"unix:///tmp/l1_attach_{uuid.uuid4().hex}.sock"
    server = native.HostTestServer(address)
    assert server.ok()
    client = native.L1Client(address)
    with pytest.raises(RuntimeError):
        client.attach(
            "owner", "g", 0, 128, external_handle=b"invalid", external_kind="hip_fabric"
        )
    with pytest.raises(RuntimeError, match="incarnation is unknown"):
        client.detach("owner")


def test_native_old_owner_cannot_detach_replacement_server(tmp_path):
    native = pytest.importorskip("mori_umbp_l1")
    address = f"unix:///tmp/l1_epoch_{uuid.uuid4().hex}.sock"
    server = native.HostTestServer(address)
    assert server.ok()
    old_client = native.L1Client(address)
    old_client.attach("owner", "g", 0, 128)
    del server
    replacement = native.HostTestServer(address)
    assert replacement.ok()
    new_client = native.L1Client(address)
    new_client.attach("owner", "g", 0, 128)
    with pytest.raises(RuntimeError, match="incarnation"):
        old_client.detach("owner")
    assert [row["pool_id"] for row in new_client.stats()] == ["owner"]
    new_client.detach("owner")
