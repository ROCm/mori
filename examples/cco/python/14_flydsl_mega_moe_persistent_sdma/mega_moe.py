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
"""Host side: ``MegaMoE``, one EP rank of the MoE layer (quant, the persistent kernel, combine).

Per call, on one stream::

    quant     bf16 tokens -> fp8 rows + one e8m0 per 32 values      (quant.per_1x32_mx_quant)
    kernel    dispatch, GEMM1, GEMM2, SDMA sends                     (kernel.compile_mega_moe)
    combine   wait for the peers' rows, top-k weighted sum -> bf16   (sdma_parts.build_combine)

Memory:

* the mori shmem symmetric heap: what peers write with P2P stores -- the dispatch payload
  (fp8 rows, e8m0, routing weight, source word per sorted row), the dispatch plan's
  exchange (count matrix, per-source base rows, ready flags), the counters that take
  system-scope atomics, and ``comb_inp`` (Stage2 rows that do not fit the SDMA reservation);
* one cco window: what the copy engines move -- the Stage2 send buffer ``S``, the landing
  buffer ``R``, the per-slot row index ``inv`` and the flags (``sdma_parts.ArenaLayout``).

Only the v4_pro geometry at 8192 tokens/rank was tuned and measured (``compile_mega_moe``'s
defaults: sort tiles of 128 rows, 96 dispatch workgroups).
"""

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

import flydsl.expr as fx

from dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from flyutil import _run_compiled
from kernel import CHUNK_TILES, PTR_TABLE, compile_mega_moe, sdma_layout
from quant import per_1x32_mx_quant
from sdma_parts import CDONE_STRIDE, build_combine

SORT_BLOCK_M = 128
META_BLOCK_M = 32  # sorted-row metadata granularity of the dispatch plan


def _sym(shape, dtype):
    t = mori_shmem_create_tensor(shape, dtype)
    t.zero_()
    return t


def _p2p_table(t, rank, npes, dev):
    """i64 table of every peer's address of the symmetric tensor ``t``."""
    table = torch.zeros(npes, dtype=torch.int64, device=dev)
    for peer in range(npes):
        ptr = ms.shmem_ptr_p2p(t.data_ptr(), rank, peer)
        if not ptr:
            raise RuntimeError(f"no P2P path to peer {peer}")
        table[peer] = ptr
    return table


def _view(ptr, nbytes, dtype):
    """A torch tensor over ``nbytes`` of device memory at ``ptr`` (not owned)."""

    class V:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    return torch.as_tensor(V(), device=torch.cuda.current_device()).view(dtype)


class MegaMoE:
    """One rank of an EP MoE layer, A8W4 (fp8 activations, mxfp4 weights), SwiGLU.

    ``w1``: [experts_per_rank, 2 * inter, model_dim / 2] mxfp4 in the a16w4 GEMM layout
    (``weights.shuffle_weight_a16w4(.., gate_up=True)``), gate rows then up rows;
    ``w2``: [experts_per_rank, model_dim, inter / 2]; scales from
    ``weights.shuffle_scale_a16w4``. Needs torch.distributed and mori shmem initialised,
    ``MORI_ENABLE_SDMA=1`` and mori built with ``BUILD_CCO_SDMA=ON``.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        max_tok_per_rank,
        swiglu_limit,
        w1,
        w1_scale,
        w2,
        w2_scale,
    ):
        import torch.distributed as dist
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

        assert experts % world_size == 0 and world_size <= 8
        assert max_tok_per_rank & (max_tok_per_rank - 1) == 0
        self.rank, self.npes = int(rank), int(world_size)
        self.model_dim, self.inter_dim = int(model_dim), int(inter_dim)
        self.epr, self.topk = experts // world_size, int(topk)
        self.mtpr, self.swiglu_limit = int(max_tok_per_rank), float(swiglu_limit)
        self.dev = dev = torch.device("cuda", torch.cuda.current_device())
        npes, epr, K = self.npes, self.epr, self.model_dim
        i32 = {"dtype": torch.int32, "device": dev}

        # --- dispatch payload and plan (symmetric heap) ------------------------------------
        # Sorted rows: every route of every source, plus padding of each expert (and of the
        # two fanout sections) to whole sort tiles.
        self.nvm = npes * self.mtpr * self.topk + (epr + 2) * SORT_BLOCK_M
        nvm = self.nvm
        self.scale_n_i32 = (K // 32 + 3) // 4
        meta_blocks = (nvm + META_BLOCK_M - 1) // META_BLOCK_M
        self.rx_em = _sym((nvm * K,), torch.int8)  # fp8 rows
        # As [rows, K]: a kernel tensor argument's extents are 32-bit.
        self.rx_rows = self.rx_em.view(torch.float8_e4m3fn).view(nvm, K)
        self.scale_em = _sym((nvm * self.scale_n_i32,), torch.int32)  # their e8m0
        self.wts_em = _sym((nvm,), torch.float32)  # routing weight per sorted row
        self.srcmap_em = _sym((nvm,), torch.int32)  # source rank / token / top-k slot
        self.sorted_expert_ids = torch.zeros(meta_blocks, **i32)
        self.tile_row_base = torch.zeros(meta_blocks, **i32)
        self.num_valid = torch.zeros(2, **i32)
        te = npes * epr
        ws = {
            "local_hist": torch.zeros(te, **i32),
            "local_cursor": torch.zeros(te, **i32),
            "pair_order": torch.empty(self.mtpr * self.topk, **i32),
            "pair_base": torch.empty(te, **i32),
            "pair_ready": torch.zeros(2, **i32),
            "entry_count": torch.zeros(10, dtype=torch.int64, device=dev),
            "epoch_gate": torch.zeros(10, **i32),
            "pair_order_ready": torch.zeros(2, **i32),
            "work_head": torch.zeros(8 * 16, **i32),
            "work_tail": torch.zeros(1, **i32),
            "expert_tile_end": torch.empty(epr, **i32),
            "max_expert_tiles": torch.zeros(1, **i32),
            "payload_chunk_done": torch.zeros(te, **i32),
            "tile_expected": torch.zeros(meta_blocks, **i32),
            "active_payload_blocks": torch.zeros(1, **i32),
            "payload_blocks_per_destination": torch.zeros(npes, **i32),
            "payload_chunks_per_destination": torch.zeros(npes, **i32),
            "group_done": torch.zeros(npes, **i32),
        }
        sym = {
            "bigcnt": (te,),
            "count_done": (2 * npes,),
            "my_base": (te,),
            "plan_ready": (2 * npes,),
            "payload_ready": (2 * epr,),
            "launch_ready": (npes,),
            "tile_ready": (meta_blocks,),
            "tile_expected_shared": (meta_blocks,),
            "ready_tile_queue": (meta_blocks,),
            "ready_tile_epoch": (meta_blocks,),
            "ready_tile_tail": (2,),
            "payload_ready_rows": (1,),
        }
        for name, shape in sym.items():
            ws[name] = _sym(shape, torch.int32)
        # Counters: GEMM1 completions per sort tile, then the GEMM2 claim heads.
        max_sort_tiles = (nvm + SORT_BLOCK_M - 1) // SORT_BLOCK_M + 1
        self.mtile_ctr = _sym(
            ((max_sort_tiles * 4 + 63) // 64 * 16 + 8 * 16,), torch.int32
        )
        # Stage2 rows past the SDMA reservation land here, at the source's [token, slot] row.
        self.row_bytes = K + K // 32
        self.comb_inp = _sym((self.mtpr * self.topk * self.row_bytes,), torch.uint8)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        def p2p(t):
            return _p2p_table(t, self.rank, npes, dev)

        S = DispatchSlot
        table = [0] * DISPATCH_TABLE_SIZE
        local = {
            S.PAIR_BASE: ws["pair_base"],
            S.SORTED_EXPERT: self.sorted_expert_ids,
            S.TILE_ROW_BASE: self.tile_row_base,
            S.NUM_VALID: self.num_valid,
            S.SRCMAP: self.srcmap_em,
            S.LOCAL_HIST: ws["local_hist"],
            S.COUNT_MATRIX: ws["bigcnt"],
            S.COUNT_DONE: ws["count_done"],
            S.TASK_ROW_BASE: ws["my_base"],
            S.LOCAL_CURSOR: ws["local_cursor"],
            S.PAIR_ORDER: ws["pair_order"],
            S.PLAN_READY: ws["plan_ready"],
            S.PAIR_READY: ws["pair_ready"],
            S.ENTRY_COUNT: ws["entry_count"],
            S.EPOCH_GATE: ws["epoch_gate"],
            S.PAIR_ORDER_READY: ws["pair_order_ready"],
            S.WORK_HEAD: ws["work_head"],
            S.WORK_TAIL: ws["work_tail"],
            S.EXPERT_TILE_END: ws["expert_tile_end"],
            S.GROUP_DONE: ws["group_done"],
            S.LAUNCH_READY: ws["launch_ready"],
            S.MAX_EXPERT_TILES: ws["max_expert_tiles"],
            S.PAYLOAD_CHUNK_DONE: ws["payload_chunk_done"],
            S.TILE_READY: ws["tile_ready"],
            S.TILE_EXPECTED: ws["tile_expected"],
            S.TILE_EXPECTED_SHARED: ws["tile_expected_shared"],
            S.READY_TILE_QUEUE: ws["ready_tile_queue"],
            S.READY_TILE_EPOCH: ws["ready_tile_epoch"],
            S.READY_TILE_TAIL: ws["ready_tile_tail"],
            S.ACTIVE_PAYLOAD_BLOCKS: ws["active_payload_blocks"],
            S.PAYLOAD_READY_ROWS: ws["payload_ready_rows"],
            S.PAYLOAD_BLOCKS_PER_DESTINATION: ws["payload_blocks_per_destination"],
            S.PAYLOAD_CHUNKS_PER_DESTINATION: ws["payload_chunks_per_destination"],
        }
        peers = {
            S.P2P_TOKEN: self.rx_em,
            S.P2P_SCALE: self.scale_em,
            S.P2P_WEIGHT: self.wts_em,
            S.P2P_SRCMAP: self.srcmap_em,
            S.P2P_COUNT_MATRIX: ws["bigcnt"],
            S.P2P_COUNT_DONE: ws["count_done"],
            S.P2P_TASK_ROW_BASE: ws["my_base"],
            S.P2P_PLAN_READY: ws["plan_ready"],
            S.P2P_PAYLOAD_READY: ws["payload_ready"],
            S.P2P_LAUNCH_READY: ws["launch_ready"],
            S.P2P_TILE_READY: ws["tile_ready"],
            S.P2P_TILE_EXPECTED: ws["tile_expected_shared"],
            S.P2P_READY_TILE_QUEUE: ws["ready_tile_queue"],
            S.P2P_READY_TILE_EPOCH: ws["ready_tile_epoch"],
            S.P2P_READY_TILE_TAIL: ws["ready_tile_tail"],
            S.P2P_PAYLOAD_READY_ROWS: ws["payload_ready_rows"],
        }
        self._keep = [ws]
        for slot, t in local.items():
            table[slot] = t.data_ptr()
        for slot, t in peers.items():
            tab = p2p(t)
            self._keep.append(tab)
            table[slot] = tab.data_ptr()
        self.disp = torch.tensor(table, dtype=torch.int64, device=dev)
        self.p2p_comb_inp = p2p(self.comb_inp)
        self.parity = torch.zeros(
            1, **i32
        )  # launch parity and expected values, per call
        self.expected = torch.zeros(2, **i32)

        # --- GEMM1 output (GEMM2 input) -----------------------------------------------------
        self.s1_out = torch.zeros(
            (nvm, self.inter_dim), dtype=torch.float8_e4m3fn, device=dev
        )
        rows256 = (nvm + 255) // 256 * 256
        cols = (self.inter_dim // 32 + 7) // 8 * 8
        self.s1_scale = torch.zeros(
            rows256 * cols + self.inter_dim, dtype=torch.uint8, device=dev
        )

        # --- weights ------------------------------------------------------------------------
        self.w1 = w1.contiguous().view(torch.uint8)
        self.w1_scale = w1_scale.contiguous().view(torch.uint8)
        self.w2 = w2.contiguous().view(torch.uint8).view(-1)
        self.w2_scale = w2_scale.contiguous().view(torch.uint8).view(-1)

        # --- Stage2 SDMA: one cco window ------------------------------------------------------
        self.layout = lay = sdma_layout(
            npes=npes, mtpr=self.mtpr, topk=self.topk, model_dim=K
        )
        uid = [Communicator.get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(uid, src=0)
        self.comm = Communicator.init(
            npes,
            self.rank,
            uid[0],
            per_rank_vmm=max(
                Communicator.DEFAULT_PER_RANK_VMM, lay.total_bytes + (256 << 20)
            ),
        )
        self.mem = self.comm.alloc_mem(lay.total_bytes)
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
        _view(self.mem.ptr, lay.total_bytes, torch.uint8).zero_()
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1  # one channel per peer
        self.dev_comm = self.comm.create_dev_comm(reqs)
        if getattr(self.dev_comm, "sdma_num_queue", 0) < 1:
            self.comm.close_all_resources()
            raise RuntimeError(
                "no SDMA queue: build mori with BUILD_CCO_SDMA=ON, set MORI_ENABLE_SDMA=1"
            )
        self.inv = _view(self.mem.ptr + lay.off_inv, lay.slot_cap * 4, torch.int32)
        self.inv.fill_(-1)
        self.cdone = torch.zeros(
            (max_sort_tiles // CHUNK_TILES + 2) * CDONE_STRIDE, **i32
        )
        # Stage2 plan: SEG / CNT / D per (expert, origin), then the plan-ready flag.
        self.plan = torch.zeros(3 * epr * npes + 64, **i32)
        ptrs = {
            "win": int(self.win.handle),
            "dev_comm": int(self.dev_comm.ptr),
            "send": self.mem.ptr + lay.off_s,
            "cdone": self.cdone.data_ptr(),
            "plan": self.plan.data_ptr(),
        }
        self.ptr_table = torch.tensor(
            [ptrs[k] for k in PTR_TABLE], dtype=torch.int64, device=dev
        )
        self.out = torch.empty(self.mtpr, K, dtype=torch.bfloat16, device=dev)
        self.combine_fin = torch.zeros(1, **i32)
        n_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        self.comb_blocks = 2 * n_cu
        self.combine = build_combine(
            layout=lay, rank=self.rank, hidden=K, topk=self.topk
        )
        self.kernel = compile_mega_moe(
            model_dim=K,
            inter_dim=self.inter_dim,
            rank=self.rank,
            experts_per_rank=epr,
            fuse_npes=npes,
            fuse_topk=self.topk,
            fuse_mtpr=self.mtpr,
            num_valid_max=nvm,
            num_cu=n_cu,
            swiglu_limit=self.swiglu_limit,
        )
        torch.cuda.synchronize()
        self.comm.barrier()
        ms.shmem_barrier_all()

    def __call__(self, x, wts, topk_ids):
        """``x`` [tokens, model_dim] bf16, ``wts`` [tokens, topk] f32, ``topk_ids`` [tokens, topk]
        i32 global expert ids -> [tokens, model_dim] bf16 (a view of an internal buffer, valid
        until the next call)."""
        tokens = int(x.shape[0])
        assert (
            0 < tokens <= self.mtpr and x.dtype == torch.bfloat16 and x.is_contiguous()
        )
        assert wts.dtype == torch.float32 and topk_ids.dtype == torch.int32
        assert tuple(wts.shape) == tuple(topk_ids.shape) == (tokens, self.topk)
        stream = torch.cuda.current_stream()
        s = fx.Stream(stream.cuda_stream)
        x_q, x_scale = per_1x32_mx_quant(x, quant_mode="fp8")
        _run_compiled(
            self.kernel,
            self.rx_rows,  # GEMM1 A: the sorted payload rows
            self.w1,
            self.w1_scale,
            self.s1_out,  # GEMM1 output = GEMM2 A
            self.s1_scale,
            self.w2,
            self.w2_scale,
            fx.Int32(self.nvm),
            fx.Int64(self.disp.data_ptr()),
            fx.Int32(tokens),
            fx.Int64(x_q.data_ptr()),
            fx.Int64(topk_ids.data_ptr()),
            fx.Int64(wts.data_ptr()),
            fx.Int64(x_scale.data_ptr()),
            fx.Int64(self.parity.data_ptr()),
            fx.Int64(self.expected.data_ptr()),
            fx.Int64(self.mtile_ctr.data_ptr()),
            fx.Int64(self.s1_out.data_ptr()),
            fx.Int64(self.s1_scale.data_ptr()),
            fx.Int64(self.scale_em.data_ptr()),  # raw e8m0 of the payload rows
            fx.Int64(self.srcmap_em.data_ptr()),
            fx.Int64(self.wts_em.data_ptr()),
            fx.Int64(self.tile_row_base.data_ptr()),
            fx.Int64(self.p2p_comb_inp.data_ptr()),
            fx.Int64(self.ptr_table.data_ptr()),
            s,
        )
        _run_compiled(
            self.combine,
            fx.Int64(int(self.win.handle)),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.out.data_ptr()),
            fx.Int64(self.comb_inp.data_ptr()),
            fx.Int64(self.combine_fin.data_ptr()),
            fx.Int64(int(self.dev_comm.ptr)),
            fx.Int32(self.comb_blocks),
            fx.Int32(tokens),
            s,
        )
        return self.out[:tokens]

    def close(self):
        """Wait until every peer has read what this rank last sent, then free the window."""
        torch.cuda.synchronize()
        self.comm.barrier()
        self.comm.close_all_resources()
