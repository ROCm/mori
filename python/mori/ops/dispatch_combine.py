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
from mori import cpp as mori_cpp
from mori.tensor_utils import from_gpu_ptr, dtype_to_int
import os
from dataclasses import dataclass
import torch
import torch.distributed as dist

TOPK_IDX_DTYPE = torch.int32


class EpDispatchCombineKernelType(mori_cpp.EpDispatchCombineKernelType):
    def __str__(self):
        return self.name


class EpDispatchCombineQuantType(mori_cpp.EpDispatchCombineQuantType):
    def __str__(self):
        return self.name


_QUANT_TYPE_MAP = {
    "none": EpDispatchCombineQuantType.None_,
    "fp8_direct_cast": EpDispatchCombineQuantType.Fp8DirectCast,
}


def _normalize_quant_type(quant_type):
    if isinstance(quant_type, EpDispatchCombineQuantType):
        return quant_type
    if isinstance(quant_type, str):
        key = quant_type.strip().lower()
        if key in _QUANT_TYPE_MAP:
            return _QUANT_TYPE_MAP[key]
    raise ValueError(
        f"invalid quant_type '{quant_type}', expected one of {list(_QUANT_TYPE_MAP.keys())}"
    )


def _current_stream():
    return torch.cuda.current_stream().cuda_stream


def _opt_ptr(t):
    """Extract data_ptr from an optional tensor, returning 0 for None/empty."""
    if t is None:
        return 0
    if isinstance(t, torch.Tensor) and t.numel() == 0:
        return 0
    return t.data_ptr()


@dataclass
class EpDispatchCombineConfig:
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
    use_external_inp_buf: bool = True
    kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
    gpu_per_node: int = 8
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1
    quant_type: str = "none"

    @property
    def max_num_tokens_to_recv(self):
        return self.world_size * self.max_num_inp_token_per_rank


def _cpp_dispatch_combine_factory(entity_name, allow_missing=False):
    """Get a C++ binding by name from the mori_cpp module.

    Args:
        entity_name: Name of the C++ binding (function or class) to retrieve.
        allow_missing: If True, return None when binding doesn't exist
            (e.g., when compiled without ENABLE_STANDARD_MOE_ADAPT).
            If False, raise AttributeError when binding is missing.

    Returns:
        The C++ binding if found, or None if allow_missing=True and not found.
    """
    if allow_missing:
        return getattr(mori_cpp, entity_name, None)
    return getattr(mori_cpp, entity_name)


class EpDispatchCombineOp:
    def __init__(self, config):
        self.config = config

        handle_class = _cpp_dispatch_combine_factory("EpDispatchCombineHandle")
        cpp_config = mori_cpp.EpDispatchCombineConfig(
            rank=config.rank,
            world_size=config.world_size,
            hidden_dim=config.hidden_dim,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            max_token_type_size=config.max_token_type_size,
            max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
            num_experts_per_rank=config.num_experts_per_rank,
            num_experts_per_token=config.num_experts_per_token,
            warp_num_per_block=config.warp_num_per_block,
            block_num=config.block_num,
            use_external_inp_buf=config.use_external_inp_buf,
            kernel_type=config.kernel_type,
            gpu_per_node=config.gpu_per_node,
            rdma_block_num=config.rdma_block_num,
            num_qp_per_pe=config.num_qp_per_pe,
            quant_type=_normalize_quant_type(config.quant_type),
        )

        self._handle = handle_class(cpp_config)

        self._dispatch_func = _cpp_dispatch_combine_factory("launch_dispatch")
        self._dispatch_recv_func = _cpp_dispatch_combine_factory("launch_dispatch_recv")
        self._combine_func = _cpp_dispatch_combine_factory("launch_combine")
        self._combine_recv_func = _cpp_dispatch_combine_factory("launch_combine_recv")
        self._reset_func = _cpp_dispatch_combine_factory("launch_reset")
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_factory(
            "get_dispatch_src_token_pos"
        )
        self._get_cur_rank_num_token = _cpp_dispatch_combine_factory(
            "get_cur_rank_num_token"
        )
        self._get_dispatch_sender_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_sender_token_idx_map"
        )
        self._get_dispatch_receiver_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_receiver_token_idx_map"
        )
        self._get_registered_combine_input_buffer = _cpp_dispatch_combine_factory(
            "get_registered_combine_input_buffer"
        )

        self._dispatch_standard_moe_func = _cpp_dispatch_combine_factory(
            "launch_dispatch_standard_moe", allow_missing=True
        )
        self._combine_standard_moe_func = _cpp_dispatch_combine_factory(
            "launch_combine_standard_moe", allow_missing=True
        )
        self._reset_func = _cpp_dispatch_combine_factory("launch_reset")
        self._convert_dispatch_output_func = _cpp_dispatch_combine_factory(
            "convert_dispatch_output", allow_missing=True
        )
        self._convert_combine_input_func = _cpp_dispatch_combine_factory(
            "convert_combine_input", allow_missing=True
        )

        self.launch_config_mode = os.environ.get("MORI_EP_LAUNCH_CONFIG_MODE", "MANUAL")
        if self.launch_config_mode == "AUTO":
            if self.config.kernel_type.value == EpDispatchCombineKernelType.InterNodeV1.value:
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (96, 64, 8)
            elif self.config.kernel_type.value == EpDispatchCombineKernelType.InterNodeV1LL.value:
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (256, 128, 8)
            else:
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (128, 0, 16)
        elif self.launch_config_mode == "MANUAL":
            self.auto_block_num, self.auto_rdma_block_num, self.auto_warp_per_block = (
                None,
                None,
                None,
            )
        else:
            raise ValueError(
                f"invalid MORI_EP_LAUNCH_CONFIG_MODE, must be ['MANUAL', 'AUTO'], got '{self.launch_config_mode}'"
            )

    def get_launch_config(self, is_dispatch=True, block_num=-1, rdma_block_num=-1, warp_per_block=-1):
        return (
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def get_registered_combine_input_buffer(
        self, dtype: torch.dtype, hidden_dim: int = -1
    ):
        ptr, shape0, shape1 = self._get_registered_combine_input_buffer(
            self._handle, hidden_dim
        )
        return from_gpu_ptr(ptr, (shape0, shape1), dtype)

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Dispatch tokens to experts based on top-k indices.

        Args:
            input: Input token tensor.
            weights: Token weights for each expert.
            scales: Quantization scales (optional).
            indices: Top-k expert indices.
            block_num: Override config.block_num if > 0.
            warp_per_block: Override config.warp_num_per_block if > 0.
        """
        hidden_dim = input.size(1)
        scale_ptr = _opt_ptr(scales)
        has_scales = scale_ptr != 0 and self.config.scale_dim > 0

        out_ptr, outW_ptr, outS_ptr, outI_ptr, total_ptr = self._dispatch_func(
            self._handle,
            self.config.kernel_type.value,
            input.data_ptr(),
            dtype_to_int(input.dtype),
            input.size(0),
            hidden_dim,
            _opt_ptr(weights),
            scale_ptr,
            indices.data_ptr(),
            _current_stream(),
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

        max_recv = self.config.max_num_tokens_to_recv
        out = from_gpu_ptr(out_ptr, (max_recv, hidden_dim), input.dtype)
        out_weights = from_gpu_ptr(
            outW_ptr, (max_recv, self.config.num_experts_per_token), torch.float32
        )
        out_scales = None
        if has_scales and outS_ptr:
            out_scales = from_gpu_ptr(
                outS_ptr, (max_recv, self.config.scale_dim), scales.dtype
            )
        out_indices = from_gpu_ptr(
            outI_ptr, (max_recv, self.config.num_experts_per_token), TOPK_IDX_DTYPE
        )
        total_recv = from_gpu_ptr(total_ptr, (1,), TOPK_IDX_DTYPE)

        return (out, out_weights, out_scales, out_indices, total_recv)

    def dispatch_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self.dispatch(
            input,
            weights,
            scales,
            indices,
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def dispatch_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self._dispatch_recv_func(
            self._handle,
            self.config.kernel_type.value,
            _current_stream(),
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        use_external_inp_buf: int = -1,
        call_reset: bool = False,
    ):
        """Combine tokens from experts back to original positions.

        Args:
            input: Expert output tensor.
            weights: Token weights for weighted combination.
            indices: Top-k expert indices.
            block_num: Override config.block_num if > 0.
            warp_per_block: Override config.warp_num_per_block if > 0.
            use_external_inp_buf: Override config.use_external_inp_buf if >= 0.
                0 = use zero-copy (registered combine input buffer),
                1 = use external input buffer (non-zero-copy).
            call_reset: Whether to call reset after combine.
        """
        hidden_dim = input.size(1)
        weight_ptr = _opt_ptr(weights)

        out_ptr, outW_ptr = self._combine_func(
            self._handle,
            self.config.kernel_type.value,
            input.data_ptr(),
            dtype_to_int(input.dtype),
            hidden_dim,
            weight_ptr,
            indices.data_ptr(),
            _current_stream(),
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
            use_external_inp_buf,
        )

        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None
        if weight_ptr and outW_ptr:
            out_weights = from_gpu_ptr(
                outW_ptr,
                (self.config.max_num_inp_token_per_rank, self.config.num_experts_per_token),
                weights.dtype,
            )

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def combine_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self.combine(
            input,
            weights,
            indices,
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def combine_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self._combine_recv_func(
            self._handle,
            self.config.kernel_type.value,
            _current_stream(),
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def dispatch_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """DeepEP compatibility: dispatch + convert in one launch.

        Args:
            input: Input token tensor.
            weights: Token weights for each expert.
            scales: Quantization scales (optional).
            indices: Top-k expert indices.
            block_num: Override config.block_num if > 0.
            rdma_block_num: Override config.rdma_block_num if > 0 (unused in current impl).
            warp_per_block: Override config.warp_num_per_block if > 0.
        """
        if self._dispatch_standard_moe_func is None:
            raise RuntimeError(
                "dispatch_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        block_num, rdma_block_num, warp_per_block = self.get_launch_config(
            is_dispatch=True,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        hidden_dim = input.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = self.config.world_size * self.config.max_num_inp_token_per_rank

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=input.dtype, device=input.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32, device=input.device,
        )
        packed_recv_layout_range = torch.empty(0, dtype=torch.int64, device=input.device)

        packed_recv_count_ptr = self._dispatch_standard_moe_func(
            self._handle,
            self.config.kernel_type.value,
            input.data_ptr(),
            dtype_to_int(input.dtype),
            input.size(0),
            hidden_dim,
            _opt_ptr(weights),
            _opt_ptr(scales),
            indices.data_ptr(),
            _current_stream(),
            block_num,
            rdma_block_num,
            warp_per_block,
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
        )

        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range)

    def combine_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ):
        """DeepEP compatibility: combine with standard MoE inputs (no extra convert).

        Args:
            input: Expert output tensor.
            weights: Token weights for weighted combination.
            indices: Top-k expert indices.
            block_num: Override config.block_num if > 0.
            rdma_block_num: Override config.rdma_block_num if > 0 (unused in current impl).
            warp_per_block: Override config.warp_num_per_block if > 0.
            call_reset: Whether to call reset after combine.
        """
        if self._combine_standard_moe_func is None:
            raise RuntimeError(
                "combine_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        block_num, rdma_block_num, warp_per_block = self.get_launch_config(
            is_dispatch=False,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        hidden_dim = input.size(2)
        out_ptr, outW_ptr = self._combine_standard_moe_func(
            self._handle,
            self.config.kernel_type.value,
            input.data_ptr(),
            dtype_to_int(input.dtype),
            hidden_dim,
            _opt_ptr(weights),
            indices.data_ptr(),
            _current_stream(),
            block_num,
            rdma_block_num,
            warp_per_block,
        )

        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def convert_dispatch_output(
        self,
        dispatch_out_x: torch.Tensor,
        dispatch_out_topk_idx: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Convert dispatch outputs to standard MoE 3D layout (DeepEP-compatible).

        Args:
            dispatch_out_x: 2D dispatch output tokens (from dispatch).
            dispatch_out_topk_idx: 2D top-k indices aligned with dispatch_out_x.
            block_num: Override config.block_num if > 0.
            warp_per_block: Override config.warp_num_per_block if > 0.
        """
        if self._convert_dispatch_output_func is None:
            raise RuntimeError(
                "convert_dispatch_output is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = dispatch_out_x.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = self.config.world_size * self.config.max_num_inp_token_per_rank

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=dispatch_out_x.dtype, device=dispatch_out_x.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32, device=dispatch_out_x.device,
        )
        packed_recv_layout_range = torch.empty(0, dtype=torch.int64, device=dispatch_out_x.device)

        packed_recv_count_ptr = self._convert_dispatch_output_func(
            self._handle,
            dispatch_out_x.data_ptr(),
            dispatch_out_topk_idx.data_ptr(),
            hidden_dim,
            _current_stream(),
            block_num,
            warp_per_block,
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
        )

        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range)

    def convert_combine_input(
        self,
        packed_recv_x: torch.Tensor,
        packed_recv_src_info: torch.Tensor,
        packed_recv_layout_range: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Prepare standard MoE combine inputs (DeepEP-compatible).

        Args:
            packed_recv_x: 3D packed receive tensor from MoE.
            packed_recv_src_info: Source token info aligned with packed_recv_x.
            packed_recv_layout_range: Layout ranges aligned with packed_recv_x (unused in kernel).
            block_num: Override config.block_num if > 0.
            warp_per_block: Override config.warp_num_per_block if > 0.
        """
        if self._convert_combine_input_func is None:
            raise RuntimeError(
                "convert_combine_input is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = packed_recv_x.size(2)
        combine_input_ptr = self._convert_combine_input_func(
            self._handle,
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
            _current_stream(),
            block_num,
            warp_per_block,
        )

        max_recv = self.config.max_num_tokens_to_recv
        return from_gpu_ptr(combine_input_ptr, (max_recv, hidden_dim), packed_recv_x.dtype)

    def reset(self):
        self._reset_func(self._handle, _current_stream())

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.config.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()

        if self.config.kernel_type.value in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
            EpDispatchCombineKernelType.InterNodeV1LL.value,
            EpDispatchCombineKernelType.AsyncLL.value,
        ):
            ptr, size = self._get_dispatch_src_token_pos_func(self._handle)
            return from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_sender_token_idx_map_func(self._handle)
        dispatch_sender_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_receiver_token_idx_map_func(self._handle)
        dispatch_receiver_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        cur_rank_num_token = self._get_cur_rank_num_token(self._handle)
        all_rank_num_token = [torch.empty(1) for i in range(self.config.world_size)]
        dist.all_gather(all_rank_num_token, torch.Tensor([cur_rank_num_token]))

        reverse_sender_token_id_map = {}
        for r in range(self.config.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : int(all_rank_num_token[r][0].item())
                    * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.config.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )
        src_token_pos = []
        for i, recv_mapped_id in enumerate(dispatch_receiver_token_id_map.tolist()):
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            if recv_mapped_id not in reverse_sender_token_id_map:
                print(
                    f"Warning: rank {self.config.rank} src_pe {src_pe} max_num_token_to_send_per_rank {max_num_token_to_send_per_rank} recv_mapped_id {recv_mapped_id} not in reverse_sender_token_id_map"
                )
                raise
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            src_token_pos.append(src_pe * max_num_token_to_send_per_rank + src_tok_id)

        return torch.tensor(src_token_pos, dtype=torch.int)

    def get_debug_time_buf(self):
        """Get the debug time buffer as a torch.Tensor (int64)."""
        ptr, size = mori_cpp.get_debug_time_buf(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int64)

    def get_debug_time_offset(self):
        """Get the debug time offset buffer as a torch.Tensor (int32)."""
        ptr, size = mori_cpp.get_debug_time_offset(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int32)
