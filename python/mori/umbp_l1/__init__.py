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
"""Client-owned final KV HBM; Torch and native libraries load only when needed."""

from __future__ import annotations

from threading import RLock

__all__ = ["L1MemoryInDoubtError", "NativeL1Memory", "require_native_l1_support"]


class L1MemoryInDoubtError(RuntimeError):
    """Memory could still be accessed; its owner must not free or reuse it."""


# Retain before Attach until confirmed Detach. This cannot protect HBM after
# process exit: drain peers before terminating an owner whose close failed.
_LIVE_NATIVE_L1_MEMORY: dict[int, NativeL1Memory] = {}


def require_native_l1_support(module=None):
    """Return the compatible native extension without initializing a GPU."""
    if module is None:
        try:
            import mori_umbp_l1 as module
        except ImportError as exc:
            raise RuntimeError(
                "Native L1 FABRIC needs mori_umbp_l1 on PYTHONPATH, built with "
                "-DBUILD_UMBP_L1_PYTHON=ON"
            ) from exc
    if getattr(module, "NATIVE_L1_FABRIC_VERSION", None) != 2 or not callable(
        getattr(module, "export_fabric_memory", None)
    ):
        raise RuntimeError(
            "Native L1 FABRIC needs the matching mori_umbp_l1 extension "
            "(NATIVE_L1_FABRIC_VERSION=2 with export_fabric_memory)"
        )
    return module


class NativeL1Memory:
    """One zeroed FABRIC arena; allocation and Attach are each attempted once.

    Retain this helper while any tensor/view is in use. The caller owns layout,
    slot scheduling and write fencing, and must stop GPU work before close.
    Any Attach attempt requires confirmed close before releasing memory;
    neither timeout nor destruction permits freeing an uncertain allocation.
    """

    def __init__(self, client, pool_id: str, group: str):
        if not isinstance(pool_id, str) or not pool_id:
            raise ValueError("pool_id must be a nonempty string")
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a nonempty string")
        self._module = require_native_l1_support()
        self._client = client
        self._pool_id = pool_id
        self._group = group
        self._lock = RLock()
        self._arena = None
        self._mem_pool = None
        self._device = None
        self._allocated = False
        self._attach_attempted = False
        self._closing = False
        self._closed = False

    def allocate(self, nbytes: int, device):
        """Allocate final zeroed HBM, scoped to the FABRIC Torch allocator."""
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("Native L1 memory is closing or closed")
            if self._allocated:
                raise RuntimeError("Native L1 memory allows only one allocation")
            if type(nbytes) is not int or not 0 < nbytes < (1 << 64):
                raise ValueError("nbytes must be a positive uint64 integer")

            import torch

            try:
                from mori.io import make_fabric_mem_pool
            except ImportError as exc:
                raise RuntimeError(
                    "Native L1 FABRIC needs MORI's fabric Torch allocator"
                ) from exc

            owner_device = (
                torch.device("cuda", device)
                if isinstance(device, int)
                else torch.device(device)
            )
            if owner_device.type != "cuda":
                raise ValueError("Native L1 FABRIC requires a GPU device")
            if owner_device.index is None:
                owner_device = torch.device("cuda", torch.cuda.current_device())
            if not 0 <= owner_device.index < torch.cuda.device_count():
                raise ValueError("Native L1 FABRIC GPU device is out of range")

            self._allocated = True
            self._device = owner_device
            self._mem_pool = make_fabric_mem_pool()
            with (
                torch.cuda.device(owner_device),
                torch.cuda.use_mem_pool(self._mem_pool),
            ):
                self._arena = torch.empty(
                    nbytes, dtype=torch.uint8, device=owner_device
                )
                # Keep the allocator if the original tensor outlives this owner.
                self._arena._mori_native_l1_mem_pool = self._mem_pool
                # Some FABRIC runtimes lack hipMemset; zero in place with ATen.
                torch.bitwise_and(self._arena, 0, out=self._arena)
            return self._arena

    def bind_layout(self, buf, num_slots: int, regions) -> None:
        """Bind (offset, unit_bytes) regions, each containing num_slots units."""
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("Native L1 memory is closing or closed")
            if self._attach_attempted:
                raise RuntimeError(
                    "Native L1 Attach was already attempted; close this owner "
                    "before creating another allocation"
                )
            if self._arena is None or buf is not self._arena:
                raise ValueError("bind_layout must use the original native L1 arena")
            if type(num_slots) is not int or not 0 < num_slots < (1 << 32):
                raise ValueError("num_slots must be a positive uint32 integer")
            layout = list(regions)
            if not layout:
                raise ValueError("Native L1 layout must have at least one region")
            for offset, unit_bytes in layout:
                if (
                    type(offset) is not int
                    or type(unit_bytes) is not int
                    or offset < 0
                    or unit_bytes <= 0
                    or offset + num_slots * unit_bytes > buf.numel()
                ):
                    raise ValueError("Native L1 region lies outside its arena")
            ordered = sorted(layout)
            for current, following in zip(ordered, ordered[1:]):
                if current[0] + num_slots * current[1] > following[0]:
                    raise ValueError("Native L1 layout regions overlap")

            import torch

            # Initialize before Attach; the caller fences later writes via Commit.
            torch.cuda.synchronize(self._device)
            handle = self._module.export_fabric_memory(
                buf.data_ptr(), buf.numel(), self._device.index
            )
            _LIVE_NATIVE_L1_MEMORY[id(self)] = self
            self._attach_attempted = True
            try:
                self._client.attach(
                    self._pool_id,
                    self._group,
                    self._device.index,
                    buf.numel(),
                    external_handle=handle,
                    # The descriptor already includes the suballocation offset.
                    external_offset=0,
                    external_kind="hip_fabric",
                )
                self._client.bind_layout(self._pool_id, num_slots, layout)
            except Exception as exc:
                raise L1MemoryInDoubtError(
                    f"Native L1 pool {self._pool_id}: registration was not "
                    "confirmed; retaining exported memory until safe close"
                ) from exc

    def close(self) -> None:
        """Require incarnation-fenced Detach after stopping writers/Commit reporter.

        Failure retains memory and permits retry. Success is idempotent and
        preserves tensor/allocator references, but never permits reallocation.
        """
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._arena is not None:
                _LIVE_NATIVE_L1_MEMORY[id(self)] = self
            try:
                if self._arena is not None:
                    import torch

                    torch.cuda.synchronize(self._device)
                if self._attach_attempted:
                    # L1Client uses the incarnation recorded by this owner's
                    # Attach. A replacement server is not proof of safe release.
                    self._client.detach(self._pool_id)
            except Exception as exc:
                raise L1MemoryInDoubtError(
                    f"Native L1 pool {self._pool_id}: safe close was not "
                    "confirmed; retaining exported memory. Drain/stop peer "
                    "users before terminating this owner."
                ) from exc
            self._closed = True
            _LIVE_NATIVE_L1_MEMORY.pop(id(self), None)
            # Keep arena and allocator for as long as this owner survives.
