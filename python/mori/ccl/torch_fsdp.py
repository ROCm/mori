# mypy: allow-untyped-defs
"""MORI SDMA all-gather backend for PyTorch FSDP2 on ROCm.

This is an opt-in :class:`AllGather` backend backed by the ROCm `MORI
<https://github.com/ROCm/mori>`_ SDMA collectives. Enable it on an FSDP module
with::

    from mori.ccl.torch_fsdp import MoriSdmaAllGather

    model.set_custom_all_gather(MoriSdmaAllGather(zero_copy_output=True))

When ``zero_copy_output`` is set the backend produces a parameter-contiguous
output that FSDP can use in place, avoiding the rank-major copy-out. The
``mori`` package is imported lazily so importing this module does not require
ROCm/MORI to be installed. Each instance must be installed on only one FSDP
parameter group. Instances may share registered storage through an explicitly
sized ``MoriSdmaAllGatherPool`` with fixed slot assignments.

Without a pool, every instance retains a full output after reshard, including
when zero-copy is disabled. A pool retains only its configured slot capacities;
reshard returns a slot lease without unregistering or reallocating its storage.
Both modes follow FSDP's logical reshard and backward all-gather policy.
"""

import importlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from torch.distributed.fsdp._fully_shard._all_gather_layout import (
    AllGatherInputMetadata,
    AllGatherLayout,
    AllGatherOutputs,
    AllGatherParamMetadata,
    DefaultAllGatherLayout,
)
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather


_POOLED_RANK_MAJOR = object()


@dataclass
class _OutputSlot:
    buffer: torch.Tensor
    owner: object | None = None
    last_use: torch.Event | None = None


class MoriSdmaAllGatherPool:
    """Fixed registered output slots shared by non-overlapping FSDP groups.

    Construct collectively, with identical byte capacities on every rank.
    Each backend has a fixed slot so saved parameter views keep their storage.
    Assign concurrently unsharded groups, including a non-resharding root, to
    different slots. Slot conflicts raise instead of overwriting live outputs.
    FSDP must call ``AllGather.release_output`` after reshard. This pool requires
    the matching FSDP layout/lifetime contract and supports eager execution;
    CUDA graph capture is not supported.

    A small SDMA all-gather orders slot reuse across ranks without a host
    completion wait. The arena stays registered until collective ``close()``.
    """

    def __init__(
        self,
        buffer_sizes: Sequence[int],
        *,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        if not buffer_sizes or any(size <= 0 or size % 4 for size in buffer_sizes):
            raise ValueError("pool sizes must be positive multiples of four bytes")
        if not hasattr(AllGather, "release_output"):
            raise RuntimeError("shared outputs require FSDP AllGather.release_output")
        self._group = group
        self._device = torch.device(device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._capacities = tuple(buffer_sizes)
        self._bindings: list[tuple[str, int, bool]] = []
        self._mode: bool | None = None
        self._initialized = False
        self._failed = False
        self._members: tuple[int, ...] = ()
        self._subgroups: set[dist.ProcessGroup] = set()
        # MORI imports IPC allocation bases, not PyTorch suballocation offsets.
        # A fresh private pool makes this arena its allocation's first address.
        capacities = [(size + 15) // 16 * 16 for size in buffer_sizes]
        ready_bytes = (group.size() * 4 + 15) // 16 * 16
        with torch.cuda.device(self._device):
            self._mem_pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(self._mem_pool, device=self._device):
                self._buffer = torch.empty(
                    sum(capacities) + ready_bytes, dtype=torch.uint8, device=self._device
                )
        self._slots = []
        offset = 0
        for size, capacity in zip(buffer_sizes, capacities):
            self._slots.append(_OutputSlot(self._buffer.narrow(0, offset, size)))
            offset += capacity
        self._collective: Any | None = None
        self._ready_input = torch.zeros(1, dtype=torch.int32, device=self._device)
        self._ready_output = self._buffer.narrow(0, offset, group.size() * 4).view(torch.int32)
        self._last_write_event: torch.Event | None = (
            torch.cuda.current_stream(self._device).record_event()
            if self._device.type == "cuda" else None
        )
        self._closed = False

    @property
    def allocated_bytes(self) -> int:
        return self._buffer.numel()

    def _bind(self, comm, key: str | None) -> None:
        if self._initialized or self._closed:
            raise RuntimeError("bind all adapters before pool.initialize()")
        if self._mode is not None and self._mode != comm._zero_copy_output:
            raise ValueError("all adapters in a MORI pool must use the same zero_copy_output")
        key = str(len(self._bindings)) if key is None else key
        if not isinstance(key, str) or any(binding[0] == key for binding in self._bindings):
            raise ValueError("pool group_key values must be unique strings")
        self._mode = comm._zero_copy_output
        self._bindings.append((key, comm._buffer_index, comm._zero_copy_output))

    def initialize(self) -> None:
        """Collectively validate the complete static configuration before use."""
        if self._closed or self._failed:
            raise RuntimeError("MORI output pool is closed or failed")
        if self._initialized:
            return
        members = tuple(dist.get_process_group_ranks(self._group))
        config = (members, self._capacities, tuple(self._bindings))
        configs = [None] * self._group.size()
        dist.all_gather_object(configs, config, group=self._group)
        if not self._bindings or any(peer != config for peer in configs):
            raise ValueError("MORI pool capacities, group mappings, or adapter assignments differ across ranks")
        self._members = members
        self._initialized = True

    def _check_ready(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("MORI output pool is closed or failed; failed communication cannot be reused")
        if not self._initialized:
            raise RuntimeError("call pool.initialize() collectively after binding all adapters")

    def _validate_group(self, group) -> bool:
        self._check_ready()
        if group is self._group:
            return False
        if group not in self._subgroups:
            members = tuple(dist.get_process_group_ranks(group))
            if (
                not 0 < len(members) < len(self._members)
                or not set(members).issubset(self._members)
                or members[group.rank()] != self._members[self._group.rank()]
            ):
                raise ValueError("MORI pool requires its bound process group or a valid strict subgroup")
            self._subgroups.add(group)
        return True

    def _acquire(self, comm, size, dtype, device) -> torch.Tensor:
        self._check_ready()
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self._device:
            raise ValueError("MORI output pool and parameter devices must match")
        slot = self._slots[comm._buffer_index]
        if slot.owner is not None:
            raise RuntimeError(
                "MORI output slot is still leased by an unsharded group; "
                "assign overlapping groups/prefetches to different slots"
            )
        nbytes = _numel(size) * torch.empty((), dtype=dtype).element_size()
        if nbytes > slot.buffer.numel():
            raise ValueError(
                f"MORI output requires {nbytes} bytes, slot has {slot.buffer.numel()}; "
                "size slots before training so registered addresses remain stable"
            )
        output = slot.buffer.narrow(0, 0, nbytes).view(dtype).view(size)
        if slot.last_use is not None:
            torch.cuda.current_stream(self._device).wait_event(slot.last_use)
        slot.owner = comm
        return output

    def _before_write(self, comm, stream) -> None:
        self._check_ready()
        slot = self._slots[comm._buffer_index]
        if self._closed or slot.owner is not comm:
            raise RuntimeError("MORI output writes require an active slot lease")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MORI shared output slots do not support graph capture")
        if slot.last_use is not None:
            stream.wait_event(slot.last_use)
        if self._last_write_event is not None:
            stream.wait_event(self._last_write_event)
        # Only the disjoint control region is written until every peer is ready.
        collective = self._collective_for(comm, self._group)
        collective.enqueue(self._ready_input, self._ready_output, 1, stream=stream)

    def _after_write(self, stream) -> None:
        # One collective/arena is shared, including when callers change streams.
        self._last_write_event = stream.record_event()

    def _release(self, comm) -> None:
        slot = self._slots[comm._buffer_index]
        if slot.owner is not comm:
            raise RuntimeError("MORI output slot released by a different group")
        if not self._failed:
            slot.last_use = torch.cuda.current_stream(self._device).record_event()
        slot.owner = None

    def _collective_for(self, comm, group):
        self._check_ready()
        if group is not self._group:
            raise ValueError("only the bound process group may use the MORI pool collective")
        if self._collective is None:
            self._collective = comm._make_collective(group)
            self._collective.register_output_buffer(self._buffer)
        return self._collective

    def close(self) -> None:
        """Collectively unregister slots after all groups have resharded."""
        if self._closed:
            return
        if self._failed:
            raise RuntimeError("cannot close a failed MORI pool safely; terminate the distributed workers")
        if any(slot.owner is not None for slot in self._slots):
            raise RuntimeError("reshard all groups before closing the MORI output pool")
        torch.cuda.synchronize(self._device)
        dist.barrier(group=self._group)
        if self._collective is not None:
            self._collective.deregister_output_buffer(self._buffer)
            self._collective = None
        self._closed = True


class _MoriSdmaAllGatherWork(dist.Work):
    def __init__(self, collective: Any, stream: torch.Stream) -> None:
        super().__init__()
        self._collective = collective
        self._device = stream.device
        self._event = stream.record_event()

    def wait(self, timeout: object | None = None) -> bool:
        torch.cuda.current_stream(self._device).wait_event(self._event)
        return True


class _MoriSdmaAllGatherLayout(AllGatherLayout):
    def __init__(self, comm: "MoriSdmaAllGather") -> None:
        self._comm = comm
        self._metadata_key: tuple | None = None
        self._cached_metadata: tuple[torch.Tensor, torch.Tensor, int] | None = None

    def prepare_output(
        self,
        input_metadata: AllGatherInputMetadata,
    ) -> object | None:
        input_split_sizes = input_metadata.input_split_sizes
        input_numel = input_metadata.input_numel
        world_size = input_metadata.world_size
        dtype, device = input_metadata.dtype, input_metadata.device
        self._comm._clear_prepared_output()
        pool = self._comm._output_pool
        self._comm._native_fallback = pool is not None and world_size != pool._group.size()
        fallback = _POOLED_RANK_MAJOR if pool is not None else None
        if pool is not None and world_size != pool._group.size():
            return fallback
        if not self._comm._zero_copy_output or not input_metadata.can_use_param_contiguous_output:
            return fallback
        if not input_split_sizes:
            raise RuntimeError("MORI zero-copy allgather requires non-empty splits")
        if sum(input_split_sizes) != input_numel:
            raise RuntimeError(
                "MORI zero-copy allgather split sizes do not match input numel"
            )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (tuple(input_split_sizes), input_numel, world_size, dtype, device)
        metadata = self._cached_metadata
        if key != self._metadata_key or metadata is None:
            element_size = torch.empty((), dtype=dtype).element_size()
            split_sizes_u32: list[int] = []
            split_offsets_u32: list[int] = []
            offset = 0
            for split_size in input_split_sizes:
                split_nbytes = int(split_size) * element_size
                if split_nbytes % 4 != 0:
                    self._comm._warn_unaligned_fallback(
                        "MORI zero-copy allgather requires every split to be "
                        "4-byte aligned; falling back to rank-major output"
                    )
                    return fallback
                split_u32 = split_nbytes // 4
                split_offsets_u32.append(offset)
                split_sizes_u32.append(split_u32)
                offset += split_u32
            if offset * 4 != input_numel * element_size:
                raise RuntimeError("MORI zero-copy allgather byte size mismatch")
            # Keep one immutable entry; pending calls may still use an older one.
            metadata = (
                torch.tensor(split_sizes_u32, dtype=torch.int64, device=device),
                torch.tensor(split_offsets_u32, dtype=torch.int64, device=device),
                input_numel * element_size,
            )
            self._cached_metadata = metadata
            self._metadata_key = key
        (
            self._comm._param_contiguous_split_sizes,
            self._comm._param_contiguous_split_offsets,
            self._comm._param_contiguous_input_nbytes,
        ) = metadata
        return (
            self._comm._param_contiguous_split_sizes,
            self._comm._param_contiguous_split_offsets,
        )

    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_metadata: list[AllGatherParamMetadata],
        world_size: int,
        output_metadata: object | None,
    ) -> AllGatherOutputs:
        if output_metadata is _POOLED_RANK_MAJOR:
            outputs = DefaultAllGatherLayout().finalize_outputs(
                all_gather_output, param_metadata, world_size, None
            )
            # Existing zero-copy parameters can still alias this slot on fallback.
            if not any(param.backend_owned for param in param_metadata):
                self._comm.release_output()
            return outputs
        if self._comm._output_pool is not None:
            self._comm._aliases_pool = any(
                not param.outputs or param.backend_owned for param in param_metadata
            )
        return AllGatherOutputs(
            self.param_contiguous_output_views(
                all_gather_output, [param.input_numels for param in param_metadata], world_size
            ),
            backend_owned=True,
        )


class MoriSdmaAllGather(AllGather):
    """All-gather backend using MORI SDMA collectives (ROCm).

    Args:
        zero_copy_output (bool): produce a parameter-contiguous output that FSDP
            uses in place, skipping the rank-major copy-out wherever the
            parameter group is eligible. Defaults to ``True``.
        output_pool: Optional shared registered storage. Each backend retains a
            separate layout/owner; only its output memory is shared.
        buffer_index: Fixed slot in ``output_pool``. Assign groups that may be
            unsharded concurrently to distinct slots.
        group_key: Optional unique name for cross-rank binding validation.
            Without a name, adapters are identified by their construction order.
    """

    reuses_output_storage = True

    def __init__(
        self, zero_copy_output: bool = True, *,
        output_pool: MoriSdmaAllGatherPool | None = None,
        buffer_index: int = 0,
        group_key: str | None = None,
    ) -> None:
        if output_pool is not None and not 0 <= buffer_index < len(output_pool._slots):
            raise ValueError("buffer_index is outside the MORI output pool")
        self._output_pool = output_pool
        self._buffer_index = buffer_index
        self._pool_active = False
        self._native_fallback = False
        self._aliases_pool = False
        self._zero_copy_output = zero_copy_output
        self.layout = _MoriSdmaAllGatherLayout(self)
        self._collective: Any | None = None
        self._process_group: dist.ProcessGroup | None = None
        self._output_buffer: torch.Tensor | None = None
        self._output_active = False
        self._last_use_event: torch.Event | None = None
        self._output_buffer_nbytes = 0
        self._registered_output_ptr: int | None = None
        self._param_contiguous_split_sizes: torch.Tensor | None = None
        self._param_contiguous_split_offsets: torch.Tensor | None = None
        self._param_contiguous_input_nbytes = 0
        self._warned_unaligned = False
        if output_pool is not None:
            output_pool._bind(self, group_key)
        elif not zero_copy_output:
            warnings.warn(
                "MORI without output_pool retains a full output per group even with "
                "zero_copy_output=False; use a shared pool for bounded output storage",
                stacklevel=2,
            )

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self._output_pool is not None:
            self._output_pool._check_ready()
            if self._native_fallback:
                output = torch.empty(size, dtype=dtype, device=device)
                if self._aliases_pool:
                    self._output_pool._acquire(self, size, dtype, device)
                    self._pool_active = True
                return output
            output = self._output_pool._acquire(self, size, dtype, device)
            self._pool_active = True
            return output
        if self._last_use_event is not None:
            torch.cuda.current_stream(device).wait_event(self._last_use_event)
        numel = _numel(size)
        if (
            self._output_buffer is not None
            and self._output_buffer.dtype == dtype
            and self._output_buffer.device == device
            and self._output_buffer.numel() >= numel
        ):
            output = self._output_buffer.narrow(0, 0, numel)
            self._output_active = True
            return output
        self._deregister_output_buffer_if_needed()
        self._output_buffer = torch.empty(*size, dtype=dtype, device=device)
        self._output_buffer_nbytes = _tensor_nbytes(self._output_buffer)
        self._registered_output_ptr = None
        self._output_active = True
        return self._output_buffer

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Any | None:
        pool = self._output_pool
        may_have_launched = False
        try:
            subgroup = pool._validate_group(group) if pool is not None else False
            self._validate_tensors(output_tensor, input_tensor, group)
            if subgroup and output_tensor.untyped_storage().data_ptr() == pool._buffer.data_ptr():
                raise ValueError("subgroup fallback requires independent staging from layout preparation")
            may_have_launched = True
            if subgroup:
                return dist.all_gather_into_tensor(
                    output_tensor, input_tensor, group=group, async_op=async_op
                )
            return self._call_full_group(output_tensor, input_tensor, group, async_op)
        except BaseException:
            if may_have_launched and pool is not None:
                pool._failed = True
            self.release_output()
            raise
        finally:
            self._clear_prepared_output()

    def _call_full_group(self, output_tensor, input_tensor, group, async_op):
        if self._output_pool is not None:
            stream = torch.cuda.current_stream(input_tensor.device)
            self._output_pool._before_write(self, stream)
        elif self._last_use_event is not None:
            torch.cuda.current_stream(input_tensor.device).wait_event(self._last_use_event)
        if _tensor_nbytes(input_tensor) % 4 != 0:
            self._warn_unaligned_fallback(
                "MORI SDMA allgather requires 4-byte-aligned input; falling "
                "back to the process-group all-gather"
            )
            work = dist.all_gather_into_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=async_op,
            )
            if self._output_pool is not None:
                self._output_pool._after_write(stream)
            return work
        collective = self._get_collective(group)
        stream = torch.cuda.current_stream(input_tensor.device)
        count = input_tensor.numel()
        self._ensure_output_registered(collective, output_tensor)
        if self._can_call_param_contiguous(input_tensor):
            split_sizes = self._param_contiguous_split_sizes
            split_offsets = self._param_contiguous_split_offsets
            if split_sizes is None or split_offsets is None:
                raise RuntimeError(
                    "MORI param-contiguous allgather metadata is not initialized"
                )
            # This kernel waits for remote SDMA writes before completing.
            collective.enqueue_param_contiguous(
                input_tensor,
                output_tensor,
                count,
                split_sizes,
                split_offsets,
                stream=stream,
            )
            split_sizes.record_stream(stream)
            split_offsets.record_stream(stream)
        else:
            collective.enqueue(input_tensor, output_tensor, count, stream=stream)
        # MORI uses raw pointers, so the allocator cannot track these uses.
        input_tensor.record_stream(stream)
        output_tensor.record_stream(stream)
        if self._output_pool is not None:
            self._output_pool._after_write(stream)
        if async_op:
            return _MoriSdmaAllGatherWork(collective, stream)
        return None

    def release_output(self) -> None:
        self._clear_prepared_output()
        if self._output_pool is not None:
            if self._pool_active:
                self._output_pool._release(self)
                self._pool_active = False
        elif self._output_active and self._output_buffer is not None:
            self._last_use_event = torch.cuda.current_stream(self._output_buffer.device).record_event()
            self._output_active = False

    def _clear_prepared_output(self) -> None:
        """Discard this call's layout selection without changing storage or leases."""
        self._native_fallback = False
        self._param_contiguous_split_sizes = None
        self._param_contiguous_split_offsets = None
        self._param_contiguous_input_nbytes = 0

    def _warn_unaligned_fallback(self, message: str) -> None:
        if not self._warned_unaligned:
            warnings.warn(message, stacklevel=3)
            self._warned_unaligned = True

    def _can_call_param_contiguous(self, input_tensor: torch.Tensor) -> bool:
        if (
            self._param_contiguous_split_sizes is None
            or self._param_contiguous_split_offsets is None
        ):
            return False
        if self._param_contiguous_input_nbytes != _tensor_nbytes(input_tensor):
            self._clear_prepared_output()
            return False
        return True

    def _validate_tensors(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> None:
        if not input_tensor.is_cuda or not output_tensor.is_cuda:
            raise RuntimeError("MORI FSDP SDMA allgather requires CUDA tensors")
        if input_tensor.device != output_tensor.device:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires input and output on the same device"
            )
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires input and output dtypes to match"
            )
        expected_numel = input_tensor.numel() * group.size()
        if output_tensor.numel() != expected_numel:
            raise RuntimeError(
                "MORI FSDP SDMA allgather expected output numel "
                f"{expected_numel}, got {output_tensor.numel()}"
            )

    def _get_collective(self, group: dist.ProcessGroup) -> Any:
        if self._output_pool is not None:
            return self._output_pool._collective_for(self, group)
        if self._collective is not None:
            if group is not self._process_group:
                raise ValueError("a MORI adapter cannot reuse its collective with a different process group")
            return self._collective

        self._collective = self._make_collective(group)
        self._process_group = group
        self._registered_output_ptr = None
        return self._collective

    def _make_collective(self, group: dist.ProcessGroup) -> Any:
        rank, world_size = group.rank(), group.size()
        try:
            shmem = importlib.import_module("mori.shmem")
            AllgatherSdma = importlib.import_module("mori.ccl").AllgatherSdma
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "mori":
                raise RuntimeError(
                    "MoriSdmaAllGather requires the optional ROCm MORI Python "
                    "package providing `mori.shmem` and `mori.ccl`. Install or "
                    "load MORI before using this backend."
                ) from exc
            raise

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        if my_pe != rank or npes != world_size:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires the FSDP process group to "
                f"match SHMEM PEs, got rank/world_size={rank}/{world_size} and "
                f"my_pe/npes={my_pe}/{npes}"
            )

        return AllgatherSdma(
            my_pe,
            npes,
            input_buffer_size=4,
            output_buffer_size=4,
            copy_output_to_user=not self._zero_copy_output,
        )

    def _ensure_output_registered(
        self, collective: Any, output_tensor: torch.Tensor
    ) -> None:
        if self._output_pool is not None:
            if not collective.is_output_registered(output_tensor):
                raise RuntimeError("MORI output does not belong to a registered pool slot")
            return
        ptr = output_tensor.data_ptr()
        nbytes = _tensor_nbytes(output_tensor)
        if self._registered_output_ptr == ptr and self._output_buffer_nbytes >= nbytes:
            return
        if collective.is_output_registered(output_tensor):
            self._registered_output_ptr = ptr
            return
        collective.register_output_buffer(output_tensor)
        if self._zero_copy_output and not collective.is_output_registered(
            output_tensor
        ):
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires registered output buffers "
                "when zero-copy output is enabled"
            )
        self._registered_output_ptr = ptr

    def _deregister_output_buffer_if_needed(self) -> None:
        if self._output_pool is not None:
            return  # The pool owns the registration across all its users.
        if self._collective is None or self._output_buffer is None:
            return
        if self._registered_output_ptr != self._output_buffer.data_ptr():
            return
        self._collective.deregister_output_buffer(self._output_buffer)
        self._registered_output_ptr = None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _numel(size: Sequence[int | torch.SymInt]) -> int:
    numel = 1
    for dim in size:
        numel *= int(dim)
    return numel
