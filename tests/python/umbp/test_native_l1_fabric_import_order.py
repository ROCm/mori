# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real FABRIC export alongside installed MORI-IO, in both import orders.

Each order runs in a fresh process: an ABI regression may abort native code and
must not kill the pytest coordinator. No model, RPC service, or peer is needed.
The model's installed MORI-IO can be older than the tested L1 extension.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def fabric_gpu():
    if importlib.util.find_spec("mori_umbp_l1") is None:
        pytest.skip("native L1 extension is not installed")
    if (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("mori") is None
    ):
        pytest.skip("Torch and the installed MORI allocator are required")
    # Keep even capability probing out of the pytest process's GPU context.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; from pathlib import Path; "
                "print(json.dumps(dict(gpu=torch.cuda.is_available(), "
                "active=any(p.read_text().strip() in ('active','ready') "
                "for p in Path('/sys/bus/pci/devices').glob('*/ualink/accel_state')))))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    capabilities = json.loads(probe.stdout.strip().splitlines()[-1])
    if not capabilities["gpu"]:
        pytest.skip("no visible GPU")
    if not capabilities["active"]:
        pytest.skip("no active UALink domain reported by sysfs")


@pytest.mark.parametrize("order", ["native_first", "io_first"])
def test_fabric_export_survives_installed_io_import_order(fabric_gpu, order):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", order],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1"),
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"FABRIC export child {order} exited {result.returncode}; "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    rows = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    assert rows[-1]["phase"] == "export_verified"
    assert rows[-1]["order"] == order
    assert rows[-1]["arena_bytes"] == 4 << 20
    assert rows[-1]["descriptor_bytes"] >= 112
    assert rows[-1]["contents_unchanged"] is True


def _child(order):
    import struct

    # The order is the regression trigger. In particular, importing the native
    # extension earlier in a fixture would mask io_first's symbol interposition.
    if order == "native_first":
        native = importlib.import_module("mori_umbp_l1")
        make_fabric_mem_pool = importlib.import_module("mori.io").make_fabric_mem_pool
    elif order == "io_first":
        make_fabric_mem_pool = importlib.import_module("mori.io").make_fabric_mem_pool
        native = importlib.import_module("mori_umbp_l1")
    else:
        raise ValueError(order)

    import torch
    from mori.io.fabric_allocator import _mori_io_lib_path

    assert native.NATIVE_L1_FABRIC_VERSION == 1
    torch.cuda.set_device(0)
    print(
        json.dumps(
            {
                "phase": "imports_ready",
                "order": order,
                "native_extension": native.__file__,
                "installed_io_library": _mori_io_lib_path(),
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )
    pool = make_fabric_mem_pool()
    nbytes = 4 << 20
    with torch.cuda.use_mem_pool(pool):
        arena = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
        torch.bitwise_and(arena, 0, out=arena)
        torch.bitwise_or(arena, 37, out=arena)
    torch.cuda.synchronize()
    before = arena.data_ptr()
    print(json.dumps({"phase": "exporting", "order": order}), flush=True)
    descriptor = native.export_fabric_memory(before, nbytes, 0)
    assert descriptor[:8] == b"L1FBR001"
    allocation_bytes, offset, view_bytes = struct.unpack_from("<QQQ", descriptor, 72)
    assert view_bytes == nbytes and offset + view_bytes <= allocation_bytes
    assert arena.data_ptr() == before
    assert int(torch.count_nonzero(arena != 37).item()) == 0
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "phase": "export_verified",
                "order": order,
                "arena_bytes": nbytes,
                "descriptor_bytes": len(descriptor),
                "contents_unchanged": True,
            }
        ),
        flush=True,
    )
    # No Attach was sent, so there can be no peer access to this allocation.
    del arena
    torch.cuda.synchronize()
    del pool


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("use pytest, or --child native_first|io_first")
    _child(sys.argv[2])
