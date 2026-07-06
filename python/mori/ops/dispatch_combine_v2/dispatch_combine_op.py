"""mori-parity host op-layer for the cco-LSA intranode dispatch/combine kernels.

One SymmArena window holds the symmetric staging; per-rank metadata are plain
device tensors surfaced to the caller via from_gpu_ptr.
"""
from dataclasses import dataclass

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from mori.tensor_utils import from_gpu_ptr

from symm_arena import SymmArena
from dispatch_kernel import make_dispatch
from combine_kernel import make_combine, make_combine_scatter
from stdmoe_kernel import make_convert_dispatch_output, make_convert_combine_input
from local_expert_count_kernel import make_local_expert_count

_QUANT_TYPES = ("none", "fp8_direct_cast", "fp8_blockwise")

_DT = {torch.bfloat16: 2, torch.float32: 4}


@dataclass
class EpDispatchCombineConfig:
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    data_type: torch.dtype = torch.bfloat16
    # Per-token quant scales forwarded verbatim to dest out_scales (0 disables).
    scale_dim: int = 0
    scale_type_size: int = 0
    # "gather" (UseP2PRead) or "scatter" (mori _nop2p, fp8 compression home).
    combine_mode: str = "gather"
    quant_type: str = "none"          # none | fp8_direct_cast | fp8_blockwise
    dispatch_block_num: int = 64
    combine_block_num: int = 128
    warp_num_per_block: int = 16
    enable_std_moe: bool = False
    max_total_recv_tokens: int = 0    # mori maxTotalRecvTokens; 0 = worst-case ws*M

    def __post_init__(self):
        if self.quant_type not in _QUANT_TYPES:
            raise ValueError(f"quant_type must be one of {_QUANT_TYPES}, got {self.quant_type!r}")
        if self.combine_mode not in ("gather", "scatter"):
            raise ValueError(f"combine_mode must be gather|scatter, got {self.combine_mode!r}")
        if self.quant_type != "none":
            self.combine_mode = "scatter"

    @property
    def is_scatter(self):
        return self.combine_mode == "scatter"

    @property
    def fp8_direct_cast(self):
        return self.quant_type == "fp8_direct_cast"

    @property
    def fp8_blockwise(self):
        return self.quant_type == "fp8_blockwise"

    @property
    def combine_scale_dim(self):
        """Per-token block count for fp8_blockwise combine (block_elems=128)."""
        return self.hidden_dim // 128 if self.fp8_blockwise else 0

    @property
    def wire_elem_size(self):
        """comb_inp transport element size: 1 byte for fp8 paths, else elem_size."""
        return 1 if self.quant_type in ("fp8_direct_cast", "fp8_blockwise") else self.elem_size

    @classmethod
    def tuned(cls, **kwargs):
        """Build a config with block/warp geometry pulled from tuning_configs
        (unless explicitly overridden in kwargs)."""
        from tuning_configs import lookup
        t = lookup(kwargs["world_size"], kwargs["hidden_dim"],
                   kwargs["num_experts_per_token"])
        for k, v in t.items():
            kwargs.setdefault(k, v)
        return cls(**kwargs)

    @property
    def elem_size(self):
        return _DT[self.data_type]

    @property
    def max_recv(self):
        """Sentinel / sender-side atomic-add allocation bound (always ws*M)."""
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def effective_max_recv_per_rank(self):
        if self.max_total_recv_tokens <= 0:
            return self.max_num_inp_token_per_rank
        per = (self.max_total_recv_tokens + self.world_size - 1) // self.world_size
        return min(per, self.max_num_inp_token_per_rank)

    @property
    def effective_max_recv(self):
        """Recv-slot cap passed to the kernels as max_recv (mori MaxNumTokensToRecv)."""
        return self.world_size * self.effective_max_recv_per_rank


@dataclass
class EpDispatchRoutingHandle:
    """Per-call routing snapshot (mori EpDispatchRoutingHandle parity).

    disp_dest_tok_id_map: forward (src_tok,k)->dest flat slot (v2 tok_map).
    disp_tok_id_to_src_tok_id_local: reverse recv-slot->src token (v2 tis).
    inter_node_*: empty placeholders (v2 is intranode-only; kept for 5-tensor
    shape parity so downstream unpacking works).
    """
    disp_dest_tok_id_map: torch.Tensor
    inter_node_disp_dest_tok_id_map: torch.Tensor
    inter_node_disp_send_map: torch.Tensor
    total_recv_token_num: torch.Tensor
    disp_tok_id_to_src_tok_id_local: torch.Tensor
    cur_rank_num_token: int = 0

    def tensors(self):
        return (
            self.disp_dest_tok_id_map,
            self.inter_node_disp_dest_tok_id_map,
            self.inter_node_disp_send_map,
            self.total_recv_token_num,
            self.disp_tok_id_to_src_tok_id_local,
        )

    @classmethod
    def from_tensors(cls, tensors, cur_rank_num_token=0):
        return cls(*tensors, cur_rank_num_token=cur_rank_num_token)


class EpDispatchCombineOp:
    def __init__(self, cfg: EpDispatchCombineConfig, comm):
        self.cfg = cfg
        self.comm = comm
        c = cfg
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        esz, K, hid = c.elem_size, c.num_experts_per_token, c.hidden_dim
        M = c.max_num_inp_token_per_rank
        mr = c.effective_max_recv          # recv-slot cap (== ws*M unless capped)
        self._mr = mr

        self._scale_bytes = c.scale_dim * c.scale_type_size
        self._scale_n_i32 = (self._scale_bytes + 3) // 4
        self._enable_scales = self._scale_bytes > 0

        regions = [("tok_off", 4), ("recv_num", c.world_size * 4), ("tis", mr * 4),
                   ("out_idx", mr * K * 4), ("out_wts", mr * K * 4),
                   ("out_tok", mr * hid * esz), ("xdb_mem", c.world_size * 8)]
        if self._enable_scales:
            regions.append(("out_scales", mr * self._scale_n_i32 * 4))
        # scatter combine needs its own staging regions
        if c.is_scatter:
            wesz = c.wire_elem_size
            regions.append(("comb_inp", c.world_size * M * hid * wesz))
            regions.append(("comb_wts", c.world_size * M * K * 4))
            if c.fp8_blockwise:
                regions.append(("comb_scales", c.world_size * M * c.combine_scale_dim * 4))
        self.arena = SymmArena(comm, regions)
        from cco_example_common import zero
        zero(self.arena.local_ptr("tok_off"), self.arena.total_bytes)

        self.tok_map = torch.full((M * K,), -1, dtype=torch.int32, device=dev)
        # snapshot of the arena tis region, so a routing handle outlives the next dispatch
        self.tok_src_snapshot = torch.zeros(mr, dtype=torch.int32, device=dev)
        self._empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)  # inter-node placeholders
        self.dest_pe_ctr = torch.zeros(c.world_size, dtype=torch.int32, device=dev)
        self.disp_bar = torch.zeros(1, dtype=torch.int32, device=dev)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
        self.comb_bar = torch.zeros(1, dtype=torch.int32, device=dev)
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=dev)
        self.comb_out = torch.zeros(M * hid, dtype=torch.int16 if esz == 2 else torch.int32,
                                    device=dev)
        self.comb_out_wts = torch.zeros(M * K, dtype=torch.float32, device=dev)

        a = self.arena
        self._disp_kwargs = dict(
            rank=c.rank, npes=c.world_size, experts_per_rank=c.num_experts_per_rank,
            experts_per_token=K, hidden_dim=hid, hidden_elem_size=esz, max_tok_per_rank=M,
            max_recv=mr, block_num=c.dispatch_block_num, warp_num_per_block=c.warp_num_per_block,
            off_tok_off=a.offset("tok_off"), off_recv_num=a.offset("recv_num"),
            off_tis=a.offset("tis"), off_out_idx=a.offset("out_idx"),
            off_out_wts=a.offset("out_wts"), off_out_tok=a.offset("out_tok"),
            off_out_scales=a.offset("out_scales") if self._enable_scales else 0,
            scale_dim=c.scale_dim, scale_type_size=c.scale_type_size)
        self._dispatch = make_dispatch(**self._disp_kwargs)
        self._dispatch_replay = None   # lazily compiled
        if c.is_scatter:
            self._combine = make_combine_scatter(
                rank=c.rank, npes=c.world_size, experts_per_token=K, hidden_dim=hid,
                hidden_elem_size=esz, max_tok_per_rank=M, max_recv=mr,
                block_num=c.combine_block_num, warp_num_per_block=c.warp_num_per_block,
                off_out_tok=a.offset("out_tok"), off_comb_inp=a.offset("comb_inp"),
                off_tis=a.offset("tis"), off_xdb_mem=a.offset("xdb_mem"),
                off_out_wts=a.offset("out_wts"), off_comb_wts=a.offset("comb_wts"),
                off_comb_scales=a.offset("comb_scales") if c.fp8_blockwise else 0,
                fp8_direct_cast=c.fp8_direct_cast, fp8_blockwise=c.fp8_blockwise,
                scale_dim=c.combine_scale_dim, reset_total_recv=False)
        else:
            self._combine = make_combine(
                rank=c.rank, npes=c.world_size, experts_per_token=K, hidden_dim=hid,
                hidden_elem_size=esz, max_tok_per_rank=M, max_recv=mr,
                block_num=c.combine_block_num, warp_num_per_block=c.warp_num_per_block,
                off_out_tok=a.offset("out_tok"), off_xdb_mem=a.offset("xdb_mem"),
                off_out_wts=a.offset("out_wts"), reset_total_recv=True)

        self._lec_buf = torch.zeros(c.num_experts_per_rank, dtype=torch.int32, device=dev)
        self._local_expert_count = make_local_expert_count(
            rank=c.rank, experts_per_rank=c.num_experts_per_rank, experts_per_token=K,
            block_num=c.dispatch_block_num, warp_num_per_block=c.warp_num_per_block)

        if c.enable_std_moe:
            assert esz == 2, "StdMoE convert path is bf16-only"
            EPR = c.num_experts_per_rank
            mtpe = c.world_size * M
            self._mtpe = mtpe
            self.packed_x = torch.zeros(EPR * mtpe * hid, dtype=torch.int16, device=dev)
            self.packed_cnt = torch.zeros(EPR, dtype=torch.int32, device=dev)
            self.packed_src = torch.zeros(EPR * mtpe, dtype=torch.int32, device=dev)
            self.slot_map = torch.full((mr * K,), -1, dtype=torch.int64, device=dev)
            self._cvt_disp = make_convert_dispatch_output(
                rank=c.rank, experts_per_rank=EPR, experts_per_token=K, hidden_dim=hid,
                hidden_elem_size=esz, max_tok_per_expert=mtpe,
                block_num=c.dispatch_block_num, warp_num_per_block=c.warp_num_per_block)
            self._cvt_comb = make_convert_combine_input(
                rank=c.rank, experts_per_rank=EPR, experts_per_token=K, hidden_dim=hid,
                hidden_elem_size=esz, max_tok_per_expert=mtpe,
                block_num=c.combine_block_num, warp_num_per_block=c.warp_num_per_block)

    def recv_tokens(self):
        """Arena out_tok [max_recv, hidden] (dispatch dest / expert-GEMM input)."""
        return from_gpu_ptr(self.arena.local_ptr("out_tok"),
                            (self._mr, self.cfg.hidden_dim), self.cfg.data_type)

    def convert_dispatch_output(self):
        """mori ConvertDispatchOutput: repack recv tokens into per-local-expert
        buckets. Returns (packed_x, packed_count, packed_src); GEMM overwrites
        packed_x in place."""
        assert self.cfg.enable_std_moe, "op built without enable_std_moe"
        self.packed_cnt.zero_()
        self.slot_map.fill_(-1)
        a = self.arena
        stream = fx.Stream(torch.cuda.current_stream())
        self._cvt_disp(a.local_ptr("out_tok"), a.local_ptr("out_idx"), a.local_ptr("tis"),
                       self.total_recv.data_ptr(), self.packed_x.data_ptr(),
                       self.packed_cnt.data_ptr(), self.packed_src.data_ptr(),
                       self.slot_map.data_ptr(), stream)
        EPR, mtpe, hid = self.cfg.num_experts_per_rank, self._mtpe, self.cfg.hidden_dim
        px = from_gpu_ptr(self.packed_x.data_ptr(), (EPR, mtpe, hid), self.cfg.data_type)
        return px, self.packed_cnt, self.packed_src

    def convert_combine_input(self, routing):
        """mori ConvertCombineInput: weighted-reduce each recv token's local-expert
        outputs from packed_x back into out_tok. Run after GEMM, before combine."""
        assert self.cfg.enable_std_moe, "op built without enable_std_moe"
        a = self.arena
        stream = fx.Stream(torch.cuda.current_stream())
        self._cvt_comb(a.local_ptr("out_tok"), a.local_ptr("out_wts"),
                       routing.total_recv_token_num.data_ptr(), self.packed_x.data_ptr(),
                       self.slot_map.data_ptr(), stream)

    def recv_weights(self):
        """Arena out_wts as [max_recv, topk] f32 (forwarded per-token weights)."""
        return from_gpu_ptr(self.arena.local_ptr("out_wts"),
                            (self._mr, self.cfg.num_experts_per_token), torch.float32)

    def recv_indices(self):
        """Arena out_idx as [max_recv, topk] i32 (forwarded expert indices)."""
        return from_gpu_ptr(self.arena.local_ptr("out_idx"),
                            (self._mr, self.cfg.num_experts_per_token), torch.int32)

    def recv_scales(self):
        """Forwarded per-token scales as opaque i32 dwords [max_recv, scale_n_i32],
        or None if built without scales."""
        if not self._enable_scales:
            return None
        return from_gpu_ptr(self.arena.local_ptr("out_scales"),
                            (self._mr, self._scale_n_i32), torch.int32)

    def dispatch(self, input, weights, scales, indices, *,
                 routing=None, return_routing=False):
        """mori-parity dispatch. input [n_tok,hidden], weights [n_tok,topk] f32,
        scales [n_tok,scale_dim] (or None), indices [n_tok,topk] i32.

        routing=: replay a prior handle (reuse cached dest-slot layout, skip
        atomic routing). return_routing=: also return the handle. Mutually
        exclusive. Returns (out, out_weights, out_scales, out_indices,
        total_recv[, routing]); out == arena out_tok.
        """
        if routing is not None and return_routing:
            raise ValueError("pass either routing= (replay) or return_routing=True, not both")
        cur = input.shape[0]
        self.total_recv.zero_()
        scale_ptr = scales.data_ptr() if (scales is not None and self._enable_scales) else 0
        stream = fx.Stream(torch.cuda.current_stream())
        weight_ptr = weights.data_ptr() if weights is not None else 0
        if routing is not None:
            if self._dispatch_replay is None:
                self._dispatch_replay = make_dispatch(replay=True, **self._disp_kwargs)
            tok_map_ptr = routing.disp_dest_tok_id_map.data_ptr()
            self._dispatch_replay(self.arena.handle, input.data_ptr(), indices.data_ptr(),
                                  weight_ptr, tok_map_ptr, self.dest_pe_ctr.data_ptr(),
                                  self.disp_bar.data_ptr(), self.total_recv.data_ptr(), scale_ptr,
                                  self.cfg.rank, cur, stream)
        else:
            self._dispatch(self.arena.handle, input.data_ptr(), indices.data_ptr(),
                           weight_ptr, self.tok_map.data_ptr(), self.dest_pe_ctr.data_ptr(),
                           self.disp_bar.data_ptr(), self.total_recv.data_ptr(), scale_ptr,
                           self.cfg.rank, cur, stream)

        out = self.recv_tokens()
        out_weights = self.recv_weights()
        out_scales = self.recv_scales()
        out_indices = self.recv_indices()
        base = (out, out_weights, out_scales, out_indices, self.total_recv)
        if not return_routing:
            return base

        tis_src = from_gpu_ptr(self.arena.local_ptr("tis"), (self._mr,), torch.int32)
        self.tok_src_snapshot.copy_(tis_src)
        routing = EpDispatchRoutingHandle(
            disp_dest_tok_id_map=self.tok_map.clone(),
            inter_node_disp_dest_tok_id_map=self._empty_i32,
            inter_node_disp_send_map=self._empty_i32,
            total_recv_token_num=self.total_recv,
            disp_tok_id_to_src_tok_id_local=self.tok_src_snapshot.clone(),
            cur_rank_num_token=cur,
        )
        return base + (routing,)

    def combine(self, input, weights=None, indices=None, *, routing):
        """mori-parity combine. input [<=max_recv,hidden] post-expert tokens
        (copied into arena out_tok if not already there). weights/indices are
        accepted for API parity but unused (weights come from forwarded out_wts,
        routing carries the mapping). Returns (out [ct,hidden], out_weights [ct,topk])."""
        out_tok_ptr = self.arena.local_ptr("out_tok")
        if input.data_ptr() != out_tok_ptr:
            dst = self.recv_tokens().view(-1)[: input.numel()]
            dst.copy_(input.reshape(-1))
        self.comb_out.zero_()
        stream = fx.Stream(torch.cuda.current_stream())
        self._combine(self.arena.handle, routing.disp_dest_tok_id_map.data_ptr(),
                      self.comb_bar.data_ptr(),
                      self.xdb_flag.data_ptr(), routing.total_recv_token_num.data_ptr(),
                      self.comb_out.data_ptr(), self.comb_out_wts.data_ptr(),
                      self.cfg.rank, routing.cur_rank_num_token, stream)
        ct, hid, K = routing.cur_rank_num_token, self.cfg.hidden_dim, self.cfg.num_experts_per_token
        out = self.comb_out[: ct * hid].view(self.cfg.data_type).view(ct, hid)
        return out, self.comb_out_wts[: ct * K].view(ct, K)

    def local_expert_count(self):
        """[num_experts_per_rank] i32: recv tokens per local expert. Call after
        dispatch, before combine (gather resets total_recv)."""
        self._lec_buf.zero_()
        stream = fx.Stream(torch.cuda.current_stream())
        self._local_expert_count(self.arena.local_ptr("out_idx"), self.total_recv.data_ptr(),
                                 self._lec_buf.data_ptr(), stream)
        return self._lec_buf

    def reset(self):
        """Zero arena staging + per-rank counters/barriers (mori LaunchReset).
        Kernels self-reset counters already; this forces a clean slate."""
        from cco_example_common import zero
        zero(self.arena.local_ptr("tok_off"), self.arena.total_bytes)
        self.tok_map.fill_(-1)
        self.dest_pe_ctr.zero_()
        self.disp_bar.zero_()
        self.total_recv.zero_()
        self.comb_bar.zero_()
        self.xdb_flag.fill_(1)
