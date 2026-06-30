"""Host op-layer for the cco-LSA intranode dispatch/combine kernels.

mori-parity surface (cf. mori EpDispatchCombineOp / EpDispatchCombineConfig and
the FlyDSL FlyDSLDispatchCombineIntraNodeOp), adapted to cco-LSA: one SymmArena
window holds the symmetric staging; per-rank metadata are plain device tensors;
arena regions are surfaced to the caller as torch tensors via from_gpu_ptr.

Flow (gather / UseP2PRead combine):
  out, total_recv, routing = op.dispatch(input, weights, indices)
  # expert GEMM writes its output back into `out` (in place; out == arena out_tok)
  combined, combined_wts = op.combine(out, routing)
"""
from dataclasses import dataclass, field

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from mori.tensor_utils import from_gpu_ptr

from symm_arena import SymmArena
from dispatch_kernel import make_dispatch
from combine_kernel import make_combine

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
    dispatch_block_num: int = 64
    combine_block_num: int = 128
    warp_num_per_block: int = 16

    @property
    def elem_size(self):
        return _DT[self.data_type]

    @property
    def max_recv(self):
        return self.world_size * self.max_num_inp_token_per_rank


@dataclass
class EpDispatchRoutingHandle:
    """Per-rank routing metadata: dispatch produces it, combine consumes it
    (mori EpDispatchRoutingHandle analogue). Caches the (token,k)->dest flat
    map + recv count so combine can run without re-dispatching."""
    cur_rank_num_token: int
    total_recv: torch.Tensor          # device int32[1]
    tok_map: torch.Tensor             # device int32[cur_tok * topk]
    meta: dict = field(default_factory=dict)


class EpDispatchCombineOp:
    def __init__(self, cfg: EpDispatchCombineConfig, comm):
        self.cfg = cfg
        self.comm = comm
        c = cfg
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        esz, K, hid = c.elem_size, c.num_experts_per_token, c.hidden_dim
        mr, M = c.max_recv, c.max_num_inp_token_per_rank

        regions = [("tok_off", 4), ("recv_num", c.world_size * 4), ("tis", mr * 4),
                   ("out_idx", mr * K * 4), ("out_wts", mr * K * 4),
                   ("out_tok", mr * hid * esz), ("xdb_mem", c.world_size * 8)]
        self.arena = SymmArena(comm, regions)
        # zero the whole window once (counters/flags live here).
        from cco_example_common import zero  # local import: example helper
        zero(self.arena.local_ptr("tok_off"), self.arena.total_bytes)

        self.tok_map = torch.full((M * K,), -1, dtype=torch.int32, device=dev)
        self.dest_pe_ctr = torch.zeros(c.world_size, dtype=torch.int32, device=dev)
        self.disp_bar = torch.zeros(1, dtype=torch.int32, device=dev)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
        self.comb_bar = torch.zeros(1, dtype=torch.int32, device=dev)
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=dev)
        self.comb_out = torch.zeros(M * hid, dtype=torch.int16 if esz == 2 else torch.int32,
                                    device=dev)
        self.comb_out_wts = torch.zeros(M * K, dtype=torch.float32, device=dev)

        a = self.arena
        self._dispatch = make_dispatch(
            rank=c.rank, npes=c.world_size, experts_per_rank=c.num_experts_per_rank,
            experts_per_token=K, hidden_dim=hid, hidden_elem_size=esz, max_tok_per_rank=M,
            max_recv=mr, block_num=c.dispatch_block_num, warp_num_per_block=c.warp_num_per_block,
            off_tok_off=a.offset("tok_off"), off_recv_num=a.offset("recv_num"),
            off_tis=a.offset("tis"), off_out_idx=a.offset("out_idx"),
            off_out_wts=a.offset("out_wts"), off_out_tok=a.offset("out_tok"))
        self._combine = make_combine(
            rank=c.rank, npes=c.world_size, experts_per_token=K, hidden_dim=hid,
            hidden_elem_size=esz, max_tok_per_rank=M, max_recv=mr,
            block_num=c.combine_block_num, warp_num_per_block=c.warp_num_per_block,
            off_out_tok=a.offset("out_tok"), off_xdb_mem=a.offset("xdb_mem"),
            off_out_wts=a.offset("out_wts"), reset_total_recv=True)

    def recv_tokens(self):
        """Arena out_tok as a torch tensor [max_recv, hidden] (dispatch dest /
        expert-GEMM input; written in place, read remotely by combine)."""
        return from_gpu_ptr(self.arena.local_ptr("out_tok"),
                            (self.cfg.max_recv, self.cfg.hidden_dim), self.cfg.data_type)

    def dispatch(self, input, weights, indices):
        """input [n_tok, hidden], weights [n_tok, topk] f32, indices [n_tok, topk]
        i32. Returns (recv_x [max_recv, hidden], total_recv int, routing)."""
        cur = input.shape[0]
        self.total_recv.zero_()
        stream = fx.Stream(torch.cuda.current_stream())
        self._dispatch(self.arena.handle, input.data_ptr(), indices.data_ptr(),
                       weights.data_ptr(), self.tok_map.data_ptr(), self.dest_pe_ctr.data_ptr(),
                       self.disp_bar.data_ptr(), self.total_recv.data_ptr(),
                       self.cfg.rank, cur, stream)
        routing = EpDispatchRoutingHandle(cur, self.total_recv, self.tok_map.clone())
        return self.recv_tokens(), int(self.total_recv.cpu().item()), routing

    def combine(self, input, routing):
        """input [<=max_recv, hidden] post-expert tokens. If not already the
        arena out_tok buffer, it is copied in. Returns (out [cur_tok, hidden],
        out_weights [cur_tok, topk])."""
        out_tok_ptr = self.arena.local_ptr("out_tok")
        if input.data_ptr() != out_tok_ptr:
            dst = self.recv_tokens().view(-1)[: input.numel()]
            dst.copy_(input.reshape(-1))
        self.comb_out.zero_()
        stream = fx.Stream(torch.cuda.current_stream())
        self._combine(self.arena.handle, routing.tok_map.data_ptr(), self.comb_bar.data_ptr(),
                      self.xdb_flag.data_ptr(), routing.total_recv.data_ptr(),
                      self.comb_out.data_ptr(), self.comb_out_wts.data_ptr(),
                      self.cfg.rank, routing.cur_rank_num_token, stream)
        ct, hid, K = routing.cur_rank_num_token, self.cfg.hidden_dim, self.cfg.num_experts_per_token
        out = self.comb_out[: ct * hid].view(self.cfg.data_type).view(ct, hid)
        return out, self.comb_out_wts[: ct * K].view(ct, K)
