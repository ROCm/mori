# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native allocation lifecycle with fake Torch/RPC boundaries and real RPC fences."""

import contextlib
import gc
import importlib.util
import sys
import uuid
import weakref
from pathlib import Path
from types import SimpleNamespace

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

    def device(value, index=None):
        if isinstance(value, SimpleNamespace):
            return value
        kind, _, ordinal = str(value).partition(":")
        if index is None and ordinal:
            index = int(ordinal)
        return SimpleNamespace(type=kind, index=index)

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

    torch = SimpleNamespace(
        device=device,
        uint8="uint8",
        empty=empty,
        bitwise_and=zero,
        cuda=SimpleNamespace(
            device=contextlib.nullcontext,
            use_mem_pool=contextlib.nullcontext,
            current_device=lambda: 1,
            device_count=lambda: 4,
            synchronize=synchronize,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "mori.io", SimpleNamespace(make_fabric_mem_pool=MemPool)
    )
    monkeypatch.setitem(
        sys.modules,
        "mori_umbp_l1",
        SimpleNamespace(NATIVE_L1_FABRIC_VERSION=1, export_fabric_memory=export),
    )
    return SimpleNamespace(calls=calls, torch=torch)


def unconfirmed(*args, **kwargs):
    raise RuntimeError("operation was not confirmed")


def allocate(helper, runtime):
    client = SimpleNamespace(
        attach=lambda *args, **kwargs: runtime.calls.append(("attach", args, kwargs)),
        bind_layout=lambda *args: runtime.calls.append(("bind", args)),
        detach=lambda pool_id: runtime.calls.append(("detach", pool_id)),
    )
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


@pytest.mark.parametrize("operation", ["attach", "bind_layout"])
def test_uncertain_registration_retains_owner_without_caller_refs(
    helper, runtime, operation
):
    client, owner, arena = allocate(helper, runtime)
    setattr(client, operation, unconfirmed)
    with pytest.raises(
        helper.L1MemoryInDoubtError, match="registration was not confirmed"
    ):
        owner.bind_layout(arena, 4, [(0, 32)])
    owner_ref, arena_ref = weakref.ref(owner), weakref.ref(arena)
    allocator_ref = weakref.ref(arena._mori_native_l1_mem_pool)
    del owner, arena
    gc.collect()
    assert all(ref() is not None for ref in (owner_ref, arena_ref, allocator_ref))
    with pytest.raises(RuntimeError, match="already attempted"):
        owner_ref().bind_layout(arena_ref(), 4, [(0, 32)])
    owner_ref().close()
    gc.collect()
    assert owner_ref() is None and arena_ref() is None and allocator_ref() is None


@pytest.mark.parametrize("operation", ["detach", "synchronize"])
def test_failed_close_retains_owner_until_confirmed_retry(
    helper, runtime, monkeypatch, operation
):
    client, owner, arena = allocate(helper, runtime)
    owner.bind_layout(arena, 4, [(0, 32)])

    target = client if operation == "detach" else runtime.torch.cuda
    with monkeypatch.context() as fault:
        fault.setattr(target, operation, unconfirmed)
        with pytest.raises(helper.L1MemoryInDoubtError, match="safe close"):
            owner.close()
        assert helper._LIVE_NATIVE_L1_MEMORY[id(owner)] is owner
        with pytest.raises(RuntimeError, match="closing or closed"):
            owner.allocate(32, 0)
        with pytest.raises(RuntimeError, match="closing or closed"):
            owner.bind_layout(arena, 4, [(0, 32)])
        if operation == "synchronize":
            assert not any(c[0] == "detach" for c in runtime.calls)
    owner.close()
    assert id(owner) not in helper._LIVE_NATIVE_L1_MEMORY


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


def test_native_failed_fabric_attach_fences_detach_without_incarnation():
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


def test_native_old_owner_cannot_detach_replacement_server():
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
