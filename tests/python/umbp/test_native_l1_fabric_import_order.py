# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Exercise both native L1/MORI-IO import orders in isolated GPU processes."""

import importlib
import importlib.util
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("order", ["native_first", "io_first"])
def test_fabric_export_survives_installed_io_import_order(order):
    # An ABI regression may abort; even capability probing stays in the child.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", order],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1"),
        timeout=90,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, (
        f"FABRIC export child {order} exited {result.returncode}; "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert f"export verified: {order}" in result.stdout


def _child(order):
    if order not in ("native_first", "io_first"):
        raise ValueError(order)
    for name in ("torch", "mori", "mori_umbp_l1"):
        if importlib.util.find_spec(name) is None:
            print(f"{name} is not installed")
            return 77

    if not any(
        p.read_text().strip() in ("active", "ready")
        for p in Path("/sys/bus/pci/devices").glob("*/ualink/accel_state")
    ):
        print("an active UALink domain is required")
        return 77

    # Do not import either module before this point: ordering triggers the bug.
    names = ("mori_umbp_l1", "mori.io")
    if order == "io_first":
        names = tuple(reversed(names))
    modules = {name: importlib.import_module(name) for name in names}
    native = modules["mori_umbp_l1"]
    import torch

    if not torch.cuda.is_available():
        print("no visible GPU")
        return 77
    assert native.NATIVE_L1_FABRIC_VERSION == 1
    torch.cuda.set_device(0)
    pool = modules["mori.io"].make_fabric_mem_pool()
    nbytes = 4 << 20
    with torch.cuda.use_mem_pool(pool):
        arena = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
        torch.bitwise_and(arena, 0, out=arena)
        torch.bitwise_or(arena, 37, out=arena)
    torch.cuda.synchronize()
    before = arena.data_ptr()
    descriptor = native.export_fabric_memory(before, nbytes, 0)
    assert len(descriptor) >= 112 and descriptor[:8] == b"L1FBR001"
    allocation_bytes, offset, view_bytes = struct.unpack_from("<QQQ", descriptor, 72)
    assert view_bytes == nbytes and offset + view_bytes <= allocation_bytes
    assert arena.data_ptr() == before
    assert int(torch.count_nonzero(arena != 37).item()) == 0
    torch.cuda.synchronize()
    # No Attach was sent, so no peer can still access this allocation.
    del arena
    torch.cuda.synchronize()
    del pool
    print(f"export verified: {order}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("use pytest, or --child native_first|io_first")
    raise SystemExit(_child(sys.argv[2]))
