"""Cross-node FSDP2 all-gather backend backed by mori.ccl.HierAllGather.

Intra-node traffic rides the SDMA copy engines (XGMI); inter-node traffic goes
over RDMA (NIC). Wired into FSDP2 via ``FSDPModule.set_custom_all_gather``.

Unlike the intra-node ``AllgatherSdma`` backend, ``HierAllGather`` exposes a
flat ``__call__(input, output, count, stream)`` interface (no registered-output
/ param-contiguous zero-copy path), so this backend always produces a rank-major
output that FSDP copies out (``supports_param_contiguous_output = False``).
"""

import importlib
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather


class _HierWork:
    def __init__(self, event: "torch.cuda.Event", device: torch.device) -> None:
        self._event = event
        self._device = device
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            torch.cuda.current_stream(self._device).wait_event(self._event)
            self._waited = True
        return True


class MoriAllGather(AllGather):
    """Unified MORI all-gather backend for FSDP2 — the SAME class for single-node
    and cross-node. Internally routes intra-node traffic over SDMA (XGMI) and, when
    the process group spans multiple nodes, inter-node traffic over RDMA. User code
    is identical in both cases: ``model.set_custom_all_gather(MoriAllGather())``.
    """

    supports_param_contiguous_output = False

    def __init__(self, ranks_per_node: int | None = None) -> None:
        self._ranks_per_node = ranks_per_node
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._cap_bytes = 0
        self._output_buffer: torch.Tensor | None = None

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        numel = 1
        for dim in size:
            numel *= int(dim)
        if (
            self._output_buffer is not None
            and self._output_buffer.dtype == dtype
            and self._output_buffer.device == device
            and self._output_buffer.numel() >= numel
        ):
            return self._output_buffer.narrow(0, 0, numel)
        self._output_buffer = torch.empty(numel, dtype=dtype, device=device)
        return self._output_buffer

    def _ranks_per_node_value(self, world_size: int) -> int:
        if self._ranks_per_node is not None:
            return self._ranks_per_node
        env = os.environ.get("LOCAL_WORLD_SIZE")
        if env:
            return int(env)
        return min(torch.cuda.device_count(), world_size)

    def _get_collective(self, group: dist.ProcessGroup, per_rank_bytes: int) -> Any:
        rank, world_size = group.rank(), group.size()
        if (
            self._collective is not None
            and self._rank == rank
            and self._world_size == world_size
            and self._cap_bytes >= per_rank_bytes
        ):
            return self._collective

        shmem = importlib.import_module("mori.shmem")
        HierAllGather = importlib.import_module("mori.ccl").HierAllGather
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        if my_pe != rank or npes != world_size:
            raise RuntimeError(
                "MORI FSDP Hier allgather requires the FSDP process group to "
                f"match SHMEM PEs, got rank/world_size={rank}/{world_size} and "
                f"my_pe/npes={my_pe}/{npes}"
            )
        cap = max(per_rank_bytes, self._cap_bytes)
        ranks_per_node = self._ranks_per_node_value(world_size)
        self._collective = HierAllGather(
            my_pe,
            npes,
            input_buffer_size=cap,
            output_buffer_size=cap * world_size,
            copy_output_to_user=True,
            ranks_per_node=ranks_per_node,
        )
        self._rank = rank
        self._world_size = world_size
        self._cap_bytes = cap
        return self._collective

    def _validate(self, output_tensor, input_tensor, group) -> None:
        if not input_tensor.is_cuda or not output_tensor.is_cuda:
            raise RuntimeError("MORI FSDP Hier allgather requires CUDA tensors")
        if input_tensor.device != output_tensor.device:
            raise RuntimeError("MORI FSDP Hier allgather requires same device")
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError("MORI FSDP Hier allgather requires matching dtypes")
        expected = input_tensor.numel() * group.size()
        if output_tensor.numel() != expected:
            raise RuntimeError(
                f"MORI FSDP Hier allgather expected output numel {expected}, "
                f"got {output_tensor.numel()}"
            )
        if (input_tensor.numel() * input_tensor.element_size()) % 4 != 0:
            raise RuntimeError(
                "MORI FSDP Hier allgather requires 4-byte-aligned input bytes"
            )

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Any | None:
        self._validate(output_tensor, input_tensor, group)
        count = input_tensor.numel()
        per_rank_bytes = count * input_tensor.element_size()
        collective = self._get_collective(group, per_rank_bytes)
        device = input_tensor.device
        stream = torch.cuda.current_stream(device)
        ok = collective(input_tensor, output_tensor, count, stream=stream)
        if not ok:
            raise RuntimeError("MORI HierAllGather call failed")
        if async_op:
            event = torch.cuda.Event()
            event.record(stream)
            return _HierWork(event, device)
        return None


# Backward-compatible alias (old name).
MoriHierAllGather = MoriAllGather
