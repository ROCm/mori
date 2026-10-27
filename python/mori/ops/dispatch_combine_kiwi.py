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
"""KIWI EP dispatch/combine with the same Python interface as EpDispatchCombineOp.

The kernels push RDMA work to CPU proxy threads through KIWI invoke queues; the
proxies post it with ibverbs. Setup runs over a torch.distributed process
group, which must span the same ranks as ``config.world_size``.

Example::

    from mori.ops import EpDispatchCombineConfig
    from mori.ops.dispatch_combine_kiwi import EpDispatchCombineKiwiOp

    op = EpDispatchCombineKiwiOp(EpDispatchCombineConfig(...))
    out, out_w, out_s, out_idx, total_recv = op.dispatch(x, w, s, idx)
    y, y_w = op.combine(expert_out, w, idx)
"""

import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

_MODULE_NAME = "libmori_kiwi_ep_pybinds"
_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(pkg_dir, f"{_MODULE_NAME}.so")
    if not os.path.exists(path):
        raise ImportError(
            f"{path} not found; build mori with BUILD_OPS_KIWI=ON to use the KIWI EP kernels"
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ext = module
    return _ext


def is_available():
    """Whether this mori build includes the KIWI EP kernels (BUILD_OPS_KIWI=ON)."""
    try:
        _load_ext()
    except ImportError:
        return False
    return True


@dataclass
class EpDispatchCombineKiwiConfig:
    """Same fields as ``mori.ops.EpDispatchCombineConfig``.

    Either this class or MORI's config may be passed to EpDispatchCombineKiwiOp.
    Launch-shape fields (``warp_num_per_block``, ``block_num``, ``kernel_type``,
    ``gpu_per_node``, ``rdma_block_num``, ``num_qp_per_pe``) are accepted but
    unused: KIWI chooses its own launch shape.
    """

    data_type: torch.dtype
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    max_total_recv_tokens: int = 0
    use_external_inp_buf: bool = True
    kernel_type: object = None
    gpu_per_node: int = 8
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1
    quant_type: str = "none"


class _GpuBytes:
    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _view(ptr, shape, dtype, device):
    """Zero-copy tensor over operator-owned device memory."""
    if not ptr:
        return None
    itemsize = torch.empty((), dtype=dtype).element_size()
    nbytes = math.prod(shape) * itemsize
    raw = torch.as_tensor(_GpuBytes(ptr, nbytes), device=device)
    return raw.view(dtype).view(shape)


def _quant_format(dtype):
    if getattr(torch, "float4_e2m1fn_x2", None) == dtype:
        return "fp4"
    name = str(dtype)
    if name.startswith("torch.float8_e4m3") or name.startswith("torch.float8_e5m2"):
        return "fp8"
    return "none"


def _element_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _combine_dtype(config):
    dispatch_dtype = config.data_type
    dispatch_size = _element_size(dispatch_dtype)
    max_size = (
        config.max_token_type_size if config.max_token_type_size > 0 else dispatch_size
    )
    if max_size == dispatch_size:
        return dispatch_dtype
    if max_size == 2:
        return torch.bfloat16
    if max_size == 4:
        return torch.float32
    raise ValueError(
        f"cannot derive a combine dtype from max_token_type_size={config.max_token_type_size}"
    )


def _torch_allgather(group, world_size):
    def allgather(payload: bytes) -> bytes:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload, group=group)
        return b"".join(gathered)

    return allgather


def _current_stream():
    return torch.cuda.current_stream().cuda_stream


class EpDispatchCombineKiwiOp:
    """KIWI EP operator with the EpDispatchCombineOp calling convention.

    Args:
        config: ``EpDispatchCombineConfig`` or ``EpDispatchCombineKiwiConfig``.
        group: torch.distributed process group used for setup. Defaults to the
            world group; a single-rank op needs no process group.
        combine_dtype: combine input/output dtype. Defaults to MORI's rule:
            ``data_type`` if ``max_token_type_size`` equals its element size,
            else bfloat16 (2 bytes) or float32 (4 bytes).
        num_blocks: dispatch/combine grid size; 0 picks it per call from the
            token count.
        num_proxy_threads: CPU threads posting RDMA work (one QP set each).
        queue_capacity: invoke queue lines per queue (power of two).
        rcomp_backend: "auto", "rdma-only", or "hybrid" (HIP IPC for peers on
            the same host).
        device_name: RDMA device, e.g. "mlx5_0". Defaults to the NIC nearest
            the current GPU; ``MORI_KIWI_DEVICE_NAME`` overrides it.
    """

    def __init__(
        self,
        config,
        group=None,
        *,
        combine_dtype: Optional[torch.dtype] = None,
        num_blocks: int = 0,
        num_proxy_threads: int = 4,
        queue_capacity: int = 2048,
        rcomp_backend: str = "auto",
        device_name: str = "",
    ):
        quant_type = getattr(config, "quant_type", "none") or "none"
        if not str(getattr(quant_type, "name", quant_type)).lower().endswith("none"):
            raise NotImplementedError(
                f"EpDispatchCombineKiwiOp does not support quant_type={config.quant_type!r}"
            )
        ext = _load_ext()
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._dispatch_dtype = config.data_type
        self._combine_dtype = (
            combine_dtype if combine_dtype is not None else _combine_dtype(config)
        )

        cfg = ext.KiwiEpOpConfig()
        cfg.rank = config.rank
        cfg.world_size = config.world_size
        cfg.hidden_dim = config.hidden_dim
        cfg.topk = config.num_experts_per_token
        cfg.num_experts_per_rank = config.num_experts_per_rank
        cfg.max_num_inp_token_per_rank = config.max_num_inp_token_per_rank
        cfg.dispatch_elem_size = _element_size(self._dispatch_dtype)
        cfg.combine_elem_size = _element_size(self._combine_dtype)
        cfg.dispatch_quant_format = _quant_format(self._dispatch_dtype)
        cfg.combine_quant_format = _quant_format(self._combine_dtype)
        cfg.dispatch_scale_dim = config.scale_dim
        cfg.dispatch_scale_type_size = (
            config.scale_type_size if config.scale_dim > 0 else 0
        )
        cfg.num_blocks = num_blocks
        cfg.num_proxy_threads = num_proxy_threads
        cfg.queue_capacity = queue_capacity
        cfg.rcomp_backend = rcomp_backend
        cfg.device_name = device_name

        if config.world_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "EpDispatchCombineKiwiOp needs torch.distributed for multi-rank setup"
                )
            allgather = _torch_allgather(group, config.world_size)
        else:
            allgather = lambda payload: payload  # noqa: E731
        self._handle = ext.KiwiEpOpHandle(cfg, allgather)

        self._max_slots = self._handle.max_slots()
        self._dispatch_hidden = self._handle.dispatch_hidden_elems()
        self._combine_hidden = self._handle.combine_hidden_elems()
        topk = config.num_experts_per_token
        h = self._handle
        self._out = _view(
            h.recv_hidden_ptr(),
            (self._max_slots, self._dispatch_hidden),
            self._dispatch_dtype,
            self.device,
        )
        self._out_weights = _view(
            h.recv_weights_ptr(), (self._max_slots, topk), torch.float32, self.device
        )
        self._out_indices = _view(
            h.recv_indices_ptr(), (self._max_slots, topk), torch.int32, self.device
        )
        self._total_recv = _view(h.recv_count_ptr(), (1,), torch.int32, self.device)
        self._out_scales = {}
        self._combine_out = torch.empty(
            (config.max_num_inp_token_per_rank, self._combine_hidden),
            dtype=self._combine_dtype,
            device=self.device,
        )
        self._combine_out_weights = torch.empty(
            (config.max_num_inp_token_per_rank, topk),
            dtype=torch.float32,
            device=self.device,
        )
        self._registered_combine_input = None
        self._last_combine = None

    # ----- capacity -----

    def max_num_tokens_to_recv(self):
        return self._handle.max_num_tokens_to_recv()

    def max_num_tokens_to_recv_per_rank(self):
        return self._handle.max_num_tokens_to_recv_per_rank()

    def max_num_tokens_to_send(self):
        return self._handle.max_num_tokens_to_send()

    def max_num_tokens_to_send_per_rank(self):
        return self._handle.max_num_tokens_to_send_per_rank()

    def decode_send_flat_idx(self, flat_idx):
        """Decode a flat send index into (rank, local_token_id)."""
        stride = self.max_num_tokens_to_send()
        return int(flat_idx) // stride, int(flat_idx) % stride

    def get_registered_combine_input_buffer(
        self, dtype: torch.dtype, hidden_dim: int = -1
    ):
        hidden = hidden_dim if hidden_dim > 0 else self._combine_hidden
        if hidden != self._combine_hidden:
            raise ValueError(
                f"hidden_dim={hidden_dim} must match the configured combine hidden "
                f"size {self._combine_hidden}"
            )
        buf = self._registered_combine_input
        if buf is None or buf.dtype != dtype:
            buf = torch.empty(
                (self._max_slots, hidden), dtype=dtype, device=self.device
            )
            self._registered_combine_input = buf
        return buf

    # ----- dispatch -----

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: Optional[torch.Tensor],
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
        *,
        routing=None,
        return_routing: bool = False,
    ):
        """Returns ``(out, out_weights, out_scales, out_indices, total_recv)``.

        Outputs alias operator-owned buffers with ``max_num_tokens_to_recv()``
        rows; the first ``total_recv[0]`` rows are valid.
        """
        if routing is not None or return_routing:
            raise NotImplementedError(
                "EpDispatchCombineKiwiOp has no routing-handle path"
            )
        if call_local_expert_count:
            raise NotImplementedError(
                "EpDispatchCombineKiwiOp does not count local experts"
            )
        num_tokens = int(input.size(0))
        self._check_tensor(
            "input", input, self._dispatch_dtype, (num_tokens, self._dispatch_hidden)
        )
        self._check_tensor(
            "weights",
            weights,
            torch.float32,
            (num_tokens, self.config.num_experts_per_token),
        )
        self._check_tensor(
            "indices",
            indices,
            torch.int32,
            (num_tokens, self.config.num_experts_per_token),
        )
        scales_ptr = 0
        out_scales = None
        if scales is not None and scales.numel() > 0 and self.config.scale_dim > 0:
            self._check_tensor(
                "scales", scales, scales.dtype, (num_tokens, self.config.scale_dim)
            )
            if scales.element_size() != self.config.scale_type_size:
                raise ValueError(
                    f"scales element size {scales.element_size()} != "
                    f"config.scale_type_size {self.config.scale_type_size}"
                )
            scales_ptr = scales.data_ptr()
            out_scales = self._scales_view(scales.dtype)

        self._handle.dispatch(
            input.data_ptr(),
            weights.data_ptr(),
            scales_ptr,
            indices.data_ptr(),
            num_tokens,
            _current_stream(),
        )
        return (
            self._out,
            self._out_weights,
            out_scales,
            self._out_indices,
            self._total_recv,
        )

    def dispatch_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: Optional[torch.Tensor],
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        return self.dispatch(
            input,
            weights,
            scales,
            indices,
            call_local_expert_count=call_local_expert_count,
        )

    def dispatch_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        """No-op: dispatch_send already completes the receive on the stream."""

    # ----- combine -----

    def combine(
        self,
        input: torch.Tensor,
        weights: Optional[torch.Tensor],
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        use_external_inp_buf: int = -1,
        call_reset: bool = False,
        *,
        routing=None,
    ):
        """Reduce post-expert tokens back onto this rank's tokens.

        ``input`` holds the expert outputs in dispatch-output row order and
        ``indices`` is this rank's own [num_token, topk] routing passed to
        ``dispatch()``. Returns ``(out, out_weights)``; ``out_weights`` is None
        when ``weights`` is None.
        """
        if routing is not None:
            raise NotImplementedError(
                "EpDispatchCombineKiwiOp has no routing-handle path"
            )
        num_tokens = self._handle.num_input_tokens()
        if input.dtype != self._combine_dtype:
            raise ValueError(
                f"combine input dtype {input.dtype} != configured combine dtype "
                f"{self._combine_dtype}"
            )
        if (
            input.dim() != 2
            or input.size(1) != self._combine_hidden
            or not input.is_contiguous()
        ):
            raise ValueError(
                f"combine input must be contiguous [rows, {self._combine_hidden}], "
                f"got {tuple(input.shape)}"
            )
        topk = self.config.num_experts_per_token
        self._check_tensor("indices", indices, torch.int32, (num_tokens, topk))
        weights_ptr = 0
        if weights is not None:
            if weights.dtype != torch.float32 or not weights.is_contiguous():
                raise ValueError("combine weights must be contiguous float32")
            weights_ptr = weights.data_ptr()

        self._handle.combine(
            input.data_ptr(),
            weights_ptr,
            indices.data_ptr(),
            self._combine_out.data_ptr(),
            self._combine_out_weights.data_ptr(),
            num_tokens,
            _current_stream(),
        )
        out_weights = self._combine_out_weights if weights is not None else None
        self._last_combine = (self._combine_out, out_weights)
        if call_reset:
            self.reset()
        return self._last_combine

    def combine_send(
        self,
        input: torch.Tensor,
        weights: Optional[torch.Tensor],
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self.combine(input, weights, indices)

    def combine_recv(self, block_num: int = -1, warp_per_block: int = -1):
        """Returns the outputs of the preceding combine_send."""
        if self._last_combine is None:
            raise RuntimeError(
                "combine_recv requires a prior combine/combine_send call"
            )
        return self._last_combine

    # ----- state -----

    def reset(self):
        self._handle.reset(_current_stream())
        self._last_combine = None

    def get_dispatch_src_token_pos(self):
        """Source position of each received row, encoded as
        ``src_rank * max_num_tokens_to_send() + src_token_id``."""
        count = self._handle.copy_recv_count_to_host(_current_stream())
        return _view(
            self._handle.recv_src_token_pos_ptr(),
            (self._max_slots,),
            torch.int32,
            self.device,
        )[:count]

    def set_debug_timing_enabled(self, enabled: bool):
        self._handle.set_debug_timing_enabled(enabled)

    def get_debug_timing_snapshot(self):
        torch.cuda.synchronize()
        return {
            "enabled": self._handle.debug_timing_enabled(),
            "time_ticks": self._handle.debug_time_words(),
            "meta": self._handle.debug_meta_words(),
        }

    # ----- helpers -----

    def _scales_view(self, dtype):
        view = self._out_scales.get(dtype)
        if view is None:
            view = _view(
                self._handle.recv_scales_ptr(),
                (self._max_slots, self.config.scale_dim),
                dtype,
                self.device,
            )
            self._out_scales[dtype] = view
        return view

    def _check_tensor(self, name, tensor, dtype, shape):
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"{name} shape must be {tuple(shape)}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


__all__ = ["EpDispatchCombineKiwiConfig", "EpDispatchCombineKiwiOp", "is_available"]
