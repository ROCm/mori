# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Adaptive MORI/Kiwi expert-parallel dispatch/combine wrapper."""

from __future__ import annotations

import logging
import os
import socket
import threading
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.distributed as dist

from .dispatch_combine import EpDispatchCombineOp

logger = logging.getLogger(__name__)

_KIWI_CONTEXT = None
_KIWI_CONTEXT_KEY = None
_KIWI_CONTEXT_LOCK = threading.Lock()


def _load_kiwi_extension():
    """Load Kiwi EP, accepting its pre-rename GLCI module name."""

    try:
        import kiwi_ep_ext

        return kiwi_ep_ext, kiwi_ep_ext.KiwiEpOp
    except ImportError as kiwi_error:
        try:
            import glci_ep_ext

            return glci_ep_ext, glci_ep_ext.GlciEpOp
        except ImportError:
            raise RuntimeError(
                "adaptive MORI/Kiwi EP requires an importable kiwi_ep_ext "
                "(or legacy glci_ep_ext)"
            ) from kiwi_error


def _resolve_process_group(group_name: str):
    if not dist.is_initialized():
        raise RuntimeError(
            "adaptive MORI/Kiwi EP requires torch.distributed to be initialized"
        )
    return dist.distributed_c10d._resolve_process_group(group_name)


def _group_root_global_rank(group) -> int:
    get_global_rank = getattr(dist, "get_global_rank", None)
    if get_global_rank is not None:
        return int(get_global_rank(group, 0))
    return int(dist.distributed_c10d._get_global_rank(group, 0))


def _root_rendezvous(group, master_addr: str | None, master_port: int | None):
    rank = dist.get_rank(group)
    if rank == 0:
        addr = master_addr or socket.gethostbyname(socket.gethostname())
        if master_port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((addr, 0))
                port = int(probe.getsockname()[1])
        else:
            port = int(master_port)
        rendezvous = [addr, port]
    else:
        rendezvous = [None, None]
    dist.broadcast_object_list(
        rendezvous, src=_group_root_global_rank(group), group=group
    )
    return str(rendezvous[0]), int(rendezvous[1])


@contextmanager
def _temporary_environ(values: dict[str, str]):
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def initialize_kiwi_lci_from_torch_process_group(
    group_name: str,
    *,
    master_addr: str | None = None,
    master_port: int | None = None,
    device_name: str = "",
):
    """Initialize Kiwi's LCI runtime with a named PyTorch process group.

    LCI's TCP PMI backend consumes rank and rendezvous information from the
    environment during construction. The environment is restored immediately
    afterwards; the returned process-global context retains the initialized
    runtime. All ranks in ``group_name`` must call this collectively.
    """

    global _KIWI_CONTEXT, _KIWI_CONTEXT_KEY

    group = _resolve_process_group(group_name)
    rank = int(dist.get_rank(group))
    world_size = int(dist.get_world_size(group))
    get_group_ranks = getattr(dist, "get_process_group_ranks", None)
    group_ranks = (
        tuple(get_group_ranks(group))
        if get_group_ranks is not None
        else (group_name, rank, world_size)
    )
    key = (group_ranks, device_name)

    with _KIWI_CONTEXT_LOCK:
        if _KIWI_CONTEXT is not None:
            if _KIWI_CONTEXT_KEY != key:
                raise RuntimeError(
                    "Kiwi LCI is already initialized for a different process group: "
                    f"existing={_KIWI_CONTEXT_KEY}, requested={key}"
                )
            return _KIWI_CONTEXT

        addr, port = _root_rendezvous(group, master_addr, master_port)

        kiwi_ep_ext, _ = _load_kiwi_extension()

        if not device_name:
            get_hca = getattr(kiwi_ep_ext, "get_hca_name_for_current_gpu", None)
            if get_hca is not None:
                device_name = str(get_hca())

        env = {
            "LCT_PMI_BACKEND": "tcp",
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LCI_MASTER_ADDR": addr,
            "LCI_MASTER_PORT": str(port),
        }
        with _temporary_environ(env):
            context = kiwi_ep_ext.LciContext(device_name)

        if int(kiwi_ep_ext.get_rank()) != rank:
            raise RuntimeError(
                f"Kiwi LCI rank {kiwi_ep_ext.get_rank()} does not match "
                f"PyTorch group rank {rank}"
            )
        if int(kiwi_ep_ext.get_nranks()) != world_size:
            raise RuntimeError(
                f"Kiwi LCI world size {kiwi_ep_ext.get_nranks()} does not match "
                f"PyTorch group size {world_size}"
            )

        _KIWI_CONTEXT = context
        _KIWI_CONTEXT_KEY = key
        return context


class AdaptiveEpDispatchCombineOp:
    """Select MORI or Kiwi for each complete dispatch/combine pair.

    ``input.size(0)`` is the selection key. Under vLLM CUDA graph capture this
    is the final padded token-bucket size, so each captured graph permanently
    records one backend. Callers must provide the same bucket on every rank.
    """

    def __init__(
        self,
        config,
        *,
        kiwi_max_num_tokens: int,
        kiwi_op=None,
        mori_op=None,
        dispatch_dtype: torch.dtype | None = None,
        combine_dtype: torch.dtype | None = None,
        group_name: str = "mori",
        num_blocks: int = 4,
        num_proxy_threads: int = 4,
        queue_capacity: int = 2048,
        device_name: str = "",
        lci_master_addr: str | None = None,
        lci_master_port: int | None = None,
        selection_num_tokens_fn: Callable[[int], int] | None = None,
    ):
        if kiwi_max_num_tokens < 0:
            raise ValueError("kiwi_max_num_tokens must be non-negative")

        self.config = config
        self.kiwi_max_num_tokens = int(kiwi_max_num_tokens)
        self._selection_num_tokens_fn = selection_num_tokens_fn
        self.mori_op = mori_op if mori_op is not None else EpDispatchCombineOp(config)

        if kiwi_op is None:
            if dispatch_dtype is None or combine_dtype is None:
                raise ValueError(
                    "dispatch_dtype and combine_dtype are required when kiwi_op "
                    "is not supplied"
                )
            if not device_name:
                kiwi_ep_ext, _ = _load_kiwi_extension()
                get_hca = getattr(
                    kiwi_ep_ext, "get_hca_name_for_current_gpu", None
                )
                if get_hca is not None:
                    device_name = str(get_hca())
            initialize_kiwi_lci_from_torch_process_group(
                group_name,
                master_addr=lci_master_addr,
                master_port=lci_master_port,
                device_name=device_name,
            )
            kiwi_op = self._create_kiwi_op(
                dispatch_dtype,
                combine_dtype,
                num_blocks,
                num_proxy_threads,
                queue_capacity,
                device_name,
            )
        self.kiwi_op = kiwi_op
        self._active_backend: str | None = None
        self._last_backend: str | None = None
        self._logged_backends: set[str] = set()
        self._validate_capacities()

    def _create_kiwi_op(
        self,
        dispatch_dtype,
        combine_dtype,
        num_blocks,
        num_proxy_threads,
        queue_capacity,
        device_name,
    ):
        _, kiwi_op_type = _load_kiwi_extension()

        if self.config.max_total_recv_tokens:
            raise NotImplementedError(
                "Kiwi adaptive backend does not support max_total_recv_tokens"
            )
        if not self.config.use_external_inp_buf:
            raise NotImplementedError(
                "Kiwi adaptive backend requires use_external_inp_buf=True"
            )
        if str(self.config.quant_type).lower() != "none":
            raise NotImplementedError(
                "Kiwi adaptive backend does not support MORI combine quantization"
            )

        dispatch_quant_format = "none"
        dispatch_name = str(dispatch_dtype)
        if "float8" in dispatch_name:
            dispatch_quant_format = "fp8"
        elif "float4" in dispatch_name:
            dispatch_quant_format = "fp4"

        scale_dtype = torch.float32
        if self.config.scale_dim and self.config.scale_type_size == 1:
            scale_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)

        return kiwi_op_type(
            device_id=torch.cuda.current_device(),
            world_size=self.config.world_size,
            hidden_dim=self.config.hidden_dim,
            topk=self.config.num_experts_per_token,
            num_experts_per_rank=self.config.num_experts_per_rank,
            max_num_inp_token_per_rank=self.config.max_num_inp_token_per_rank,
            dispatch_dtype=dispatch_dtype,
            combine_dtype=combine_dtype,
            num_blocks=num_blocks,
            num_proxy_threads=num_proxy_threads,
            dispatch_scale_dim=self.config.scale_dim,
            dispatch_scale_type_size=self.config.scale_type_size,
            dispatch_scale_dtype=scale_dtype,
            device_name=device_name,
            dispatch_quant_format=dispatch_quant_format,
            queue_capacity=queue_capacity,
        )

    def _validate_capacities(self):
        accessors = (
            "max_num_tokens_to_recv",
            "max_num_tokens_to_recv_per_rank",
            "max_num_tokens_to_send",
            "max_num_tokens_to_send_per_rank",
        )
        for name in accessors:
            mori_value = int(getattr(self.mori_op, name)())
            kiwi_value = int(getattr(self.kiwi_op, name)())
            if mori_value != kiwi_value:
                raise ValueError(
                    f"MORI/Kiwi capacity mismatch for {name}: "
                    f"mori={mori_value}, kiwi={kiwi_value}"
                )

    def backend_for_num_tokens(self, num_tokens: int) -> str:
        return "kiwi" if int(num_tokens) <= self.kiwi_max_num_tokens else "mori"

    def dispatch(
        self,
        input,
        weights,
        scales,
        indices,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
        call_local_expert_count=False,
        *,
        routing=None,
        return_routing=False,
    ):
        if self._active_backend is not None:
            raise RuntimeError("dispatch called before the previous combine completed")

        local_num_tokens = int(input.size(0))
        selection_num_tokens = (
            self._selection_num_tokens_fn(local_num_tokens)
            if self._selection_num_tokens_fn is not None
            else local_num_tokens
        )
        backend = self.backend_for_num_tokens(selection_num_tokens)
        if backend not in self._logged_backends:
            message = (
                f"Adaptive MORI/Kiwi EP selected backend={backend} "
                f"local_num_tokens={local_num_tokens} "
                f"selection_num_tokens={selection_num_tokens} "
                f"kiwi_max_num_tokens={self.kiwi_max_num_tokens}"
            )
            logger.info(message)
            if os.environ.get("MORI_EP_KIWI_TRACE") == "1":
                print(message, flush=True)
            self._logged_backends.add(backend)
        self._active_backend = backend
        try:
            if backend == "kiwi":
                if call_local_expert_count:
                    raise NotImplementedError(
                        "Kiwi adaptive dispatch does not support local expert counts"
                    )
                if routing is not None or return_routing:
                    raise NotImplementedError(
                        "Kiwi adaptive dispatch does not support MORI routing handles"
                    )
                return self.kiwi_op.dispatch(
                    input,
                    weights,
                    scales,
                    indices,
                    block_num=block_num,
                    rdma_block_num=rdma_block_num,
                    warp_per_block=warp_per_block,
                )
            return self.mori_op.dispatch(
                input,
                weights,
                scales,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
                call_local_expert_count=call_local_expert_count,
                routing=routing,
                return_routing=return_routing,
            )
        except Exception:
            self._active_backend = None
            raise

    def combine(
        self,
        input,
        weights,
        indices,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
        use_external_inp_buf=-1,
        call_reset=False,
        *,
        routing=None,
    ):
        backend = self._active_backend
        if backend is None:
            raise RuntimeError("combine requires a preceding adaptive dispatch")
        try:
            if backend == "kiwi":
                if routing is not None:
                    raise NotImplementedError(
                        "Kiwi adaptive combine does not support MORI routing handles"
                    )
                return self.kiwi_op.combine(
                    input,
                    weights,
                    indices,
                    block_num=block_num,
                    rdma_block_num=rdma_block_num,
                    warp_per_block=warp_per_block,
                    use_external_inp_buf=use_external_inp_buf,
                    call_reset=call_reset,
                )
            return self.mori_op.combine(
                input,
                weights,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
                use_external_inp_buf=use_external_inp_buf,
                call_reset=call_reset,
                routing=routing,
            )
        finally:
            self._last_backend = backend
            self._active_backend = None

    def dispatch_send(self, *args, **kwargs):
        raise NotImplementedError(
            "adaptive MORI/Kiwi EP supports complete dispatch/combine pairs only"
        )

    def dispatch_recv(self, *args, **kwargs):
        raise NotImplementedError(
            "adaptive MORI/Kiwi EP supports complete dispatch/combine pairs only"
        )

    def combine_send(self, *args, **kwargs):
        raise NotImplementedError(
            "adaptive MORI/Kiwi EP supports complete dispatch/combine pairs only"
        )

    def combine_recv(self, *args, **kwargs):
        raise NotImplementedError(
            "adaptive MORI/Kiwi EP supports complete dispatch/combine pairs only"
        )

    def reset(self):
        self.mori_op.reset()
        self.kiwi_op.reset()
        self._active_backend = None
        self._last_backend = None

    @property
    def active_backend(self):
        return self._active_backend

    @property
    def last_backend(self):
        return self._last_backend

    def __getattr__(self, name: str) -> Any:
        backend = self.__dict__.get("_active_backend")
        if backend is None:
            backend = self.__dict__.get("_last_backend")
        if backend == "kiwi":
            kiwi_op = self.__dict__.get("kiwi_op")
            if kiwi_op is not None and hasattr(kiwi_op, name):
                return getattr(kiwi_op, name)
        return getattr(self.__dict__["mori_op"], name)
