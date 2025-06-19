from mori import cpp as mori_cpp

from dataclasses import dataclass

import torch


@dataclass
class EpDispatchCombineConfig:
    data_type: torch.dtype
    rank: int
    world_size: int
    hidden_dim: int
    num_scales: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int


def _cpp_dispatch_combine_factory(data_type: torch.dtype, entity_name):
    _mori_ops_dtype_map = {
        torch.float32: "Fp32",
        torch.bfloat16: "Bf16",
        torch.float8_e4m3fnuz: "Fp8E4m3Fnuz",
    }
    return getattr(mori_cpp, entity_name + _mori_ops_dtype_map[data_type])


class EpDispatchCombineOp:
    def __init__(self, config):
        self.config = config

        handle_class = _cpp_dispatch_combine_factory(
            config.data_type, "EpDispatchCombineHandle"
        )
        self._handle = handle_class(
            mori_cpp.EpDispatchCombineConfig(
                rank=config.rank,
                world_size=config.world_size,
                hidden_dim=config.hidden_dim,
                num_scales=config.num_scales,
                max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
                num_experts_per_rank=config.num_experts_per_rank,
                num_experts_per_token=config.num_experts_per_token,
                warp_num_per_block=4,  # config.warp_num_per_block,
                block_num=256,  # config.block_num,
            )
        )

        self._intra_node_dispatch_func = _cpp_dispatch_combine_factory(
            config.data_type, "launch_intra_node_dispatch_"
        )
        self._intra_node_combine_func = _cpp_dispatch_combine_factory(
            config.data_type, "launch_intra_node_combine_"
        )
        self._reset_func = _cpp_dispatch_combine_factory(
            config.data_type, "launch_reset_"
        )
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_factory(
            config.data_type, "get_dispatch_src_token_pos_"
        )

    def dispatch(
        self, input: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, indicies: torch.Tensor
    ):
        return self._intra_node_dispatch_func(
            self._handle,
            input,
            weights,
            scales,
            indicies,
        )

    def combine(
        self, input: torch.Tensor, weights: torch.Tensor, indicies: torch.Tensor
    ):
        output = self._intra_node_combine_func(
            self._handle,
            input,
            weights,
            indicies,
        )
        self._reset_func(self._handle)
        return output

    def get_dispatch_src_token_pos(self):
        return self._get_dispatch_src_token_pos_func(self._handle)
