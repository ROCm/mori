import os
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
import torch.distributed as dist

from ._fsdp_api import AllGather


def is_mori_fsdp_sdma_enabled() -> bool:
    raw = os.environ.get("MORI_FSDP_ENABLE_SDMA", "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


class _MoriSdmaAllGatherWork:
    def __init__(self, collective: Any, stream: torch.cuda.Stream):
        self._collective = collective
        self._stream = stream
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            self._collective.wait_async(stream=self._stream)
            self._waited = True
        return True


class MoriSdmaAllGather(AllGather):
    def __init__(self) -> None:
        self._collective: Optional[Any] = None
        self._rank: Optional[int] = None
        self._world_size: Optional[int] = None
        self._input_buffer_size = 0
        self._output_buffer_size = 0

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[Any]:
        self._validate_tensors(output_tensor, input_tensor, group)
        collective = self._get_collective(output_tensor, input_tensor, group)
        stream = torch.cuda.current_stream(input_tensor.device)
        count = input_tensor.numel()
        if async_op:
            collective.start_async(input_tensor, output_tensor, count, stream=stream)
            return _MoriSdmaAllGatherWork(collective, stream)
        collective(input_tensor, output_tensor, count, stream=stream)
        return None

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
        world_size = group.size()
        expected_numel = input_tensor.numel() * world_size
        if output_tensor.numel() != expected_numel:
            raise RuntimeError(
                "MORI FSDP SDMA allgather expected output numel "
                f"{expected_numel}, got {output_tensor.numel()}"
            )
        input_nbytes = _tensor_nbytes(input_tensor)
        output_nbytes = _tensor_nbytes(output_tensor)
        if input_nbytes % 4 != 0 or output_nbytes % 4 != 0:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires input/output byte sizes "
                "to be 4-byte aligned"
            )

    def _get_collective(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> Any:
        rank, world_size = group.rank(), group.size()
        input_nbytes = _tensor_nbytes(input_tensor)
        output_nbytes = _tensor_nbytes(output_tensor)
        if (
            self._collective is not None
            and self._rank == rank
            and self._world_size == world_size
            and self._input_buffer_size >= input_nbytes
            and self._output_buffer_size >= output_nbytes
        ):
            return self._collective

        from mori import shmem
        from mori.ccl import AllgatherSdma

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        if my_pe != rank or npes != world_size:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires the FSDP process group to "
                f"match SHMEM PEs, got rank/world_size={rank}/{world_size} and "
                f"my_pe/npes={my_pe}/{npes}"
            )

        self._collective = AllgatherSdma(
            my_pe,
            npes,
            input_buffer_size=max(input_nbytes, 4),
            output_buffer_size=max(output_nbytes, 4),
            copy_output_to_user=True,
        )
        self._rank = rank
        self._world_size = world_size
        self._input_buffer_size = input_nbytes
        self._output_buffer_size = output_nbytes
        return self._collective


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
