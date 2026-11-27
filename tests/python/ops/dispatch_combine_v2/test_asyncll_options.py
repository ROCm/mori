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
"""AsyncLL policy checks: imports Torch/native bindings, but uses no GPU."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp
from mori.ops.dispatch_combine_v2.hip_backend import EpDispatchCombineOpHip
from mori.ops.dispatch_combine_v2.internode_regions import (
    asyncll_regions,
    asyncll_reserve_words,
    asyncll_seg_layout,
    asyncll_seg_words,
)


class AsyncLLOptionsTest(unittest.TestCase):
    def resolve(self, env, transport="sdma", nodes=1, arch="gfx1250"):
        cfg = SimpleNamespace(async_ll_transport=transport, nodes=nodes)
        with patch.dict(os.environ, env, clear=True):
            mode, skip_reset = EpDispatchCombineOpHip._asyncll_options(cfg, arch)
            seq = EpDispatchCombineOpHip._asyncll_seq(cfg, mode)
        return mode, skip_reset, seq

    def test_default_and_legacy_keep_receive_completion_passes(self):
        for env, expected in [
            ({}, (1, True)),
            ({"MORI_ASYNCLL_SKIP_RESET": "0"}, (1, False)),
            ({"MORI_ASYNCLL_EARLY_NOTIFY": "0"}, (0, False)),
        ]:
            with self.subTest(env=env):
                mode, skip, seq = self.resolve(env)
                self.assertEqual((mode, skip), expected)
                self.assertEqual(
                    seq["dispatch_recv"],
                    ("dispatch_recvtransfer", "dispatch_recvcopy"),
                )
                self.assertEqual(
                    seq["combine_recv"],
                    ("combine_recvtransfer", "combine_recvcopy"),
                )

    def test_fused_receive_uses_one_pass_per_receive_phase(self):
        mode, skip, seq = self.resolve({"MORI_ASYNCLL_FUSED_RECV": "1"})
        self.assertEqual((mode, skip), (2, True))
        self.assertEqual(seq["dispatch_recv"], ("dispatch_recvcopy",))
        self.assertEqual(seq["combine_recv"], ("combine_recvcopy",))

    def test_sdma_options_do_not_change_lsa_or_multinode(self):
        env = {
            "MORI_ASYNCLL_EARLY_NOTIFY": "1",
            "MORI_ASYNCLL_SKIP_RESET": "1",
            "MORI_ASYNCLL_FUSED_RECV": "1",
        }
        for lsa_fused in ("0", "1"):
            with self.subTest(lsa_fused=lsa_fused):
                mode, skip, seq = self.resolve(
                    dict(env, MORI_ASYNCLL_LSA_FUSED=lsa_fused), transport="lsa"
                )
                self.assertEqual((mode, skip), (0, False))
                self.assertEqual(
                    seq["combine_recv"],
                    ("combine_recvtransfer_lsa", "combine_recvcopy"),
                )
                self.assertEqual(
                    len(seq["dispatch_send"]), 2 if lsa_fused == "1" else 3
                )
        mode, skip, seq = self.resolve(env, nodes=2)
        self.assertEqual((mode, skip), (0, False))
        self.assertIn("dispatch_recvtransfer", seq["dispatch_recv"])

    def test_gfx1250_strict_matches_gfx1250(self):
        for env in ({}, {"MORI_ASYNCLL_FUSED_RECV": "1"}):
            with self.subTest(env=env):
                self.assertEqual(
                    self.resolve(env, arch="gfx1250-strict"), self.resolve(env)
                )

    def test_gfx950_defaults_to_early_notification(self):
        self.assertEqual(self.resolve({}, arch="gfx950")[:2], (1, True))
        self.assertEqual(
            self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "0"}, arch="gfx950")[:2],
            (0, False),
        )

    def test_other_architectures_require_explicit_opt_in(self):
        self.assertEqual(self.resolve({}, arch="gfx942")[:2], (0, False))
        self.assertEqual(
            self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "1"}, arch="gfx942")[:2],
            (1, True),
        )

    def test_direct_sources_ride_on_early_notification(self):
        resolve = EpDispatchCombineOpHip._asyncll_direct
        for env, mode, expected in [
            ({}, 1, False),
            ({"MORI_ASYNCLL_DIRECT": "1"}, 1, True),
            ({"MORI_ASYNCLL_DIRECT": "1"}, 2, True),
            ({"MORI_ASYNCLL_DIRECT": "1"}, 0, False),
            ({"MORI_ASYNCLL_DIRECT": "0"}, 1, False),
        ]:
            with (
                self.subTest(env=env, mode=mode),
                patch.dict(os.environ, env, clear=True),
            ):
                self.assertEqual(resolve(mode), expected)
        with patch.dict(os.environ, {"MORI_ASYNCLL_DIRECT": "yes"}, clear=True):
            with self.assertRaises(ValueError):
                resolve(1)

    def test_direct_sources_default_on_for_segmented_gfx950(self):
        resolve = EpDispatchCombineOpHip._asyncll_direct
        seg = SimpleNamespace(async_ll_segmented=True, quant_type="none")
        compact = SimpleNamespace(async_ll_segmented=False, quant_type="none")
        quant = SimpleNamespace(
            async_ll_segmented=False, quant_type="fp8_blockwise_1x128"
        )
        for env, mode, cfg, arch, expected in [
            ({}, 1, seg, "gfx950", True),
            ({}, 2, seg, "gfx950", True),
            ({}, 0, seg, "gfx950", False),  # no early notification
            ({}, 1, compact, "gfx950", False),
            ({}, 1, quant, "gfx950", False),
            ({}, 1, seg, "gfx942", False),
            ({}, 1, seg, "gfx1250", False),
            ({"MORI_ASYNCLL_DIRECT": ""}, 1, seg, "gfx950", True),
            ({"MORI_ASYNCLL_DIRECT": "0"}, 1, seg, "gfx950", False),
            ({"MORI_ASYNCLL_DIRECT": "1"}, 1, compact, "gfx942", True),
        ]:
            with (
                self.subTest(env=env, mode=mode, cfg=cfg, arch=arch),
                patch.dict(os.environ, env, clear=True),
            ):
                self.assertEqual(resolve(mode, cfg, arch), expected)
        cfg = SimpleNamespace(async_ll_transport="sdma", nodes=1)
        seq = EpDispatchCombineOpHip._asyncll_seq(cfg, 1, direct=True)
        self.assertEqual(
            seq["dispatch_send"],
            ("dispatch_slotassign", "dispatch_sendcopy", "dispatch_sendtransfer"),
        )
        self.assertEqual(seq["combine_send"], ("combine_sendtransfer",))

    def test_fused_receive_grid_ceiling_reaches_only_the_fused_mode(self):
        resolve = EpDispatchCombineOpHip._asyncll_fused_recv_blocks
        for env, mode, expected in [
            ({}, 2, 0),
            ({"MORI_ASYNCLL_FUSED_RECV_BLOCKS": "64"}, 2, 64),
            ({"MORI_ASYNCLL_FUSED_RECV_BLOCKS": "64"}, 1, 0),
            ({"MORI_ASYNCLL_FUSED_RECV_BLOCKS": "64"}, 0, 0),
            ({"MORI_ASYNCLL_FUSED_RECV_CTAS": "32"}, 2, 32),
            (
                {
                    "MORI_ASYNCLL_FUSED_RECV_CTAS": "32",
                    "MORI_ASYNCLL_FUSED_RECV_BLOCKS": "64",
                },
                2,
                64,
            ),
        ]:
            with (
                self.subTest(env=env, mode=mode),
                patch.dict(os.environ, env, clear=True),
            ):
                self.assertEqual(resolve(mode), expected)
        for value in ("-1", "half"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with patch.dict(
                    os.environ, {"MORI_ASYNCLL_FUSED_RECV_BLOCKS": value}, clear=True
                ):
                    resolve(2)

    def test_fused_receive_ceiling_follows_the_cu_mask(self):
        resolve = EpDispatchCombineOpHip._asyncll_fused_recv_blocks
        for env, dev, expected in [
            ({}, 0, 0),  # unrestricted: the native default, half the device
            ({"HSA_CU_MASK": "0-7:0-63"}, 3, 32),
            ({"HSA_CU_MASK": "1:0-15,32-47;0:0-127"}, 1, 16),
            ({"HSA_CU_MASK": "1:0-15,32-47;0:0-127"}, 0, 64),
            ({"HSA_CU_MASK": "1:0-63"}, 0, 0),  # mask names other GPUs only
            ({"HSA_CU_MASK": "0:0-511"}, 0, 0),  # not a restriction
            ({"HSA_CU_MASK": "0:0x0f"}, 0, 0),  # unreadable: whole device
            (
                {"HSA_CU_MASK": "0-7:0-63", "MORI_ASYNCLL_FUSED_RECV_BLOCKS": "48"},
                0,
                48,
            ),
        ]:
            with (
                self.subTest(env=env, dev=dev),
                patch.dict(os.environ, env, clear=True),
            ):
                self.assertEqual(resolve(2, dev, 256), expected)
        with patch.dict(os.environ, {"HSA_CU_MASK": "0-7:0-63"}, clear=True):
            self.assertEqual(resolve(1, 0, 256), 0)

    def test_reject_inconsistent_or_invalid_options(self):
        for option in ("MORI_ASYNCLL_FUSED_RECV", "MORI_ASYNCLL_SKIP_RESET"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "0", option: "1"})
        with self.assertRaises(ValueError):
            self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "yes"})


class AsyncLLSegmentedTest(unittest.TestCase):
    _SEND = (
        "dispatch_slotassign",
        "dispatch_sendcopy_seg",
        "dispatch_sendtransfer_seg",
    )
    _SEND_FUSED_META = ("dispatch_slotassign", "dispatch_sendtransfer_seg")

    def seg_seq(self, mode, fuse_meta, env=None):
        # The sequence depends only on its arguments, never on the live env.
        with patch.dict(os.environ, env or {}, clear=True):
            return EpDispatchCombineOpHip._asyncll_seg_seq(mode, fuse_meta)

    def test_sequences(self):
        for mode in (0, 1):
            with self.subTest(mode=mode):
                seq = self.seg_seq(mode, False)
                self.assertEqual(seq["dispatch_send"], self._SEND)
                self.assertEqual(
                    seq["dispatch_recv"],
                    ("dispatch_recvtransfer", "dispatch_recvfinalize_seg"),
                )
                self.assertEqual(seq["combine_send"], ("combine_sendtransfer_seg",))
                self.assertEqual(
                    seq["combine_recv"],
                    ("combine_recvtransfer", "combine_recvcopy_seg"),
                )

    def test_fused_receive_sequence(self):
        seq = self.seg_seq(2, False)
        self.assertEqual(seq["dispatch_send"], self._SEND)
        self.assertEqual(seq["dispatch_recv"], ("dispatch_recvfinalize_seg",))
        self.assertEqual(seq["combine_send"], ("combine_sendtransfer_seg",))
        self.assertEqual(seq["combine_recv"], ("combine_recvcopy_seg",))

    def test_fuse_meta_drops_only_sendcopy(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                fused = self.seg_seq(mode, True)
                plain = self.seg_seq(mode, False)
                self.assertEqual(fused["dispatch_send"], self._SEND_FUSED_META)
                for group in ("dispatch_recv", "combine_send", "combine_recv"):
                    self.assertEqual(fused[group], plain[group])
        # The env alone does not drop the pass; only the resolved value does.
        self.assertEqual(
            self.seg_seq(1, False, {"MORI_ASYNCLL_FUSE_META": "1"})["dispatch_send"],
            self._SEND,
        )

    def test_fuse_meta_needs_direct_and_segmented(self):
        resolve = EpDispatchCombineOpHip._asyncll_fuse_meta
        for env, direct, segmented, expected in [
            ({}, True, True, True),
            ({}, False, True, False),
            ({"MORI_ASYNCLL_FUSE_META": "0"}, True, True, False),
            ({"MORI_ASYNCLL_FUSE_META": "1"}, True, True, True),
            ({"MORI_ASYNCLL_FUSE_META": "1"}, False, True, False),
            ({"MORI_ASYNCLL_FUSE_META": "1"}, True, False, False),
            ({"MORI_ASYNCLL_FUSE_META": "1"}, False, False, False),
        ]:
            with (
                self.subTest(env=env, direct=direct, segmented=segmented),
                patch.dict(os.environ, env, clear=True),
            ):
                self.assertIs(resolve(direct, segmented), expected)
        for value in ("yes", "2", ""):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"MORI_ASYNCLL_FUSE_META": value}, clear=True),
                self.assertRaises(ValueError),
            ):
                resolve(False, False)

    def test_fuse_meta_follows_direct_resolution(self):
        # Direct off (mode 0, or MORI_ASYNCLL_DIRECT unset) keeps SendCopySeg.
        for env, mode, want in [
            ({"MORI_ASYNCLL_FUSE_META": "1"}, 1, self._SEND),
            (
                {"MORI_ASYNCLL_FUSE_META": "1", "MORI_ASYNCLL_DIRECT": "1"},
                0,
                self._SEND,
            ),
            (
                {"MORI_ASYNCLL_FUSE_META": "1", "MORI_ASYNCLL_DIRECT": "1"},
                1,
                self._SEND_FUSED_META,
            ),
            ({"MORI_ASYNCLL_DIRECT": "1"}, 1, self._SEND_FUSED_META),
            (
                {"MORI_ASYNCLL_DIRECT": "1", "MORI_ASYNCLL_FUSE_META": "0"},
                1,
                self._SEND,
            ),
        ]:
            with (
                self.subTest(env=env, mode=mode),
                patch.dict(os.environ, env, clear=True),
            ):
                direct = EpDispatchCombineOpHip._asyncll_direct(mode)
                fuse = EpDispatchCombineOpHip._asyncll_fuse_meta(direct, True)
                seq = EpDispatchCombineOpHip._asyncll_seg_seq(mode, fuse)
                self.assertEqual(seq["dispatch_send"], want)

    def test_config_values(self):
        def cfg(layout, async_ll=True):
            return EpDispatchCombineConfig(
                rank=0,
                world_size=8,
                hidden_dim=7168,
                max_num_inp_token_per_rank=64,
                num_experts_per_rank=48,
                num_experts_per_token=6,
                kernel_backend="hip",
                async_ll=async_ll,
                async_ll_transport="sdma",
                async_ll_recv_layout=layout,
            )

        for value, want in (("compact", False), ("segmented", True)):
            with self.subTest(value=value):
                self.assertEqual(cfg(value).async_ll_recv_layout, value)
                self.assertIs(cfg(value).async_ll_segmented, want)
        # "auto" stays for the backend to resolve; it is not segmented yet.
        self.assertEqual(cfg("auto").async_ll_recv_layout, "auto")
        self.assertIs(cfg("auto").async_ll_segmented, False)
        # "auto" without async_ll resolves to compact at once; "segmented" there
        # is an error.
        self.assertEqual(cfg("auto", async_ll=False).async_ll_recv_layout, "compact")
        for bad in (True, 1, "on", "zero_copy"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cfg(bad)
        with self.assertRaises(ValueError):
            cfg("segmented", async_ll=False)

    def test_layout_fits_the_regions(self):
        for scale_dim, token_nbytes, segmented in ((0, 14336, 0), (224, 7168, 1)):
            cfg = SimpleNamespace(
                world_size=8,
                gpu_per_node=8,
                hidden_dim=7168,
                max_num_inp_token_per_rank=100,
                max_total_recv_tokens=0,
                num_experts_per_rank=32,
                num_experts_per_token=6,
                max_token_type_size=2,
                token_nbytes=token_nbytes,
                scale_dim=scale_dim,
                scale_type_size=1,
                num_qp_per_pe=2,
                async_ll_segmented=segmented,
            )
            layout = asyncll_seg_layout(cfg)
            offsets = [layout[k] for k in ("hidden", "index", "weight", "scale")]
            self.assertTrue(
                all(off % 256 == 0 for off in offsets + [layout["src_tok"]])
            )
            regions = dict(asyncll_regions(cfg))
            # The segmented expert output (world_size * S rows, S <= the receive
            # cap) fits the compact-sized inter_dispatch_staging.
            self.assertEqual(
                regions["inter_dispatch_staging"],
                cfg.world_size * cfg.max_num_inp_token_per_rank * cfg.hidden_dim * 2,
            )
            if segmented:
                self.assertGreaterEqual(regions["inter_dispatch_inp"], layout["bytes"])
                self.assertGreaterEqual(regions["inter_staging"], layout["bytes"])
                # count/epoch pairs, then mode words, mode source, 2 error
                # counters and per-source counts (asyncll_seg_words).
                self.assertEqual(regions["send_count"], (8 * 2 + 8 + 1 + 2 + 8) * 8)
            else:
                self.assertEqual(regions["send_count"], 8 * 2 * 8)

        # With a receive cap the segmented op keeps the capped staging too, so
        # the layout costs no extra arena there (and no KV-cache pool).
        cfg.max_total_recv_tokens = 8 * 50
        regions = dict(asyncll_regions(cfg))
        self.assertEqual(regions["inter_dispatch_staging"], 8 * 50 * 7168 * 2)

    @staticmethod
    def regions_cfg(scale_dim, token_nbytes, segmented, recv_cap=0):
        return SimpleNamespace(
            world_size=8,
            gpu_per_node=8,
            hidden_dim=7168,
            max_num_inp_token_per_rank=100,
            max_total_recv_tokens=recv_cap,
            num_experts_per_rank=32,
            num_experts_per_token=6,
            max_token_type_size=2,
            token_nbytes=token_nbytes,
            scale_dim=scale_dim,
            scale_type_size=1,
            num_qp_per_pe=2,
            async_ll_segmented=segmented,
        )

    def test_direct_off_keeps_the_regions(self):
        # What every non-direct kernel addresses; direct=False must not move a byte.
        common = [
            ("inter_combine_inp", 11468800),
            ("inter_dispatch_out", 11468800),
            ("inter_combine_out", 1433600),
            ("inter_dispatch_staging", 11468800),
            ("inp_weights", 19200),
            ("dispatch_out_weights", 19200),
            ("combine_out_weights", 2400),
            ("out_indices", 19200),
            ("recv_token_num", 128),
            ("node_recv_token_num", 8),
            ("disp_tok_offset", 4),
            ("disp_tok_id_to_src_tok_id", 102400),
            ("cross_device_barrier", 512),
            ("inter_node_chunk_flag", 800),
        ]
        golden = {
            (0, 14336, 0): [("inter_dispatch_inp", 11510400)]
            + common[:1]
            + [("inter_staging", 11510400)]
            + common[1:]
            + [("send_count", 128), ("inter_combine_staging", 11468800)],
            (224, 7168, 1): [("inter_dispatch_inp", 11689600)]
            + common[:1]
            + [("inter_staging", 11689600)]
            + common[1:]
            + [
                ("out_scales", 179200),
                ("send_count", 280),
                ("inter_combine_staging", 11468800),
            ],
        }
        for key, want in golden.items():
            with self.subTest(key=key):
                cfg = self.regions_cfg(*key)
                self.assertEqual(asyncll_regions(cfg), want)
                self.assertEqual(asyncll_regions(cfg, direct=False), want)

    def test_direct_regions_cover_the_kernel_extents(self):
        w, m, topk = 8, 100, 6
        dropped = {
            "inp_weights",
            "combine_out_weights",
            "node_recv_token_num",
            "disp_tok_offset",
            "inter_node_chunk_flag",
            "inter_combine_staging",
        }
        for scale_dim, token_nbytes, segmented, cap in (
            (0, 14336, 0, 0),
            (0, 14336, 1, 0),
            (224, 7168, 0, 0),
            (224, 7168, 1, 0),
            (224, 7168, 1, 8 * 50),
        ):
            with self.subTest(scales=scale_dim, segmented=segmented, cap=cap):
                cfg = self.regions_cfg(scale_dim, token_nbytes, segmented, cap)
                staged = asyncll_regions(cfg)
                direct = asyncll_regions(cfg, direct=True)
                full, regions = dict(staged), dict(direct)
                # Same order, minus the regions no direct kernel reads; none grows.
                self.assertEqual(
                    [n for n, _ in direct], [n for n, _ in staged if n not in dropped]
                )
                for name, nbytes in direct:
                    self.assertLessEqual(nbytes, full[name], name)
                # Regions peers write keep the extent the non-direct kernels address.
                meta = topk * 4 + topk * 4 + 4 + scale_dim
                xd = token_nbytes + meta
                seg_land = asyncll_seg_layout(cfg)["bytes"] if segmented else 0
                self.assertEqual(
                    regions["inter_dispatch_inp"], max(w * m * xd, seg_land)
                )
                for name in (
                    "inter_combine_inp",
                    "recv_token_num",
                    "cross_device_barrier",
                    "send_count",
                ):
                    self.assertEqual(regions[name], full[name], name)
                # Compact: DirectStagingMetaOffset(W-1, M-1) + meta, and the landing's
                # DirectMetaOffset(W-1, M-1) + meta (its hidden rows end before that).
                self.assertGreaterEqual(regions["inter_staging"], w * m * meta)
                self.assertGreaterEqual(
                    regions["inter_dispatch_inp"],
                    (w - 1) * m * xd + m * token_nbytes + m * meta,
                )
                if segmented:
                    # Peers' field rows d < W*M in the hidden-less staging layout
                    # (SegStagingLayoutOf); own rows in the landing layout.
                    lay = asyncll_seg_layout(cfg, hidden_nbytes=0)
                    self.assertEqual(lay["index"], 0)
                    self.assertGreaterEqual(
                        regions["inter_staging"], lay["src_tok"] + w * m * 4
                    )
                    self.assertGreaterEqual(regions["inter_dispatch_inp"], seg_land)
                # RecvCopy rows < W*R at the dispatch width; reverse map rows < W*M.
                per_rank = min(-(-cap // w), m) if cap else m
                self.assertEqual(
                    regions["inter_dispatch_out"], w * per_rank * token_nbytes
                )
                self.assertEqual(regions["disp_tok_id_to_src_tok_id"], w * m * 4)

    def test_per_rank_vmm_follows_the_direct_flag(self):
        cfg = EpDispatchCombineConfig(
            rank=0,
            world_size=8,
            hidden_dim=7168,
            max_num_inp_token_per_rank=16384,
            num_experts_per_rank=32,
            num_experts_per_token=8,
            kernel_backend="hip",
            async_ll=True,
            async_ll_transport="sdma",
            async_ll_recv_layout="auto",
        )

        def vmm(env, n_ops=1):
            with patch.dict(os.environ, env, clear=True):
                return EpDispatchCombineOp.per_rank_vmm(cfg, n_ops=n_ops, arch="gfx950")

        gib = 1 << 30
        staged, direct = vmm({"MORI_ASYNCLL_DIRECT": "0"}), vmm(
            {"MORI_ASYNCLL_DIRECT": "1"}
        )
        # "auto" resolves to segmented here, where direct sources are the default on gfx950.
        self.assertEqual(vmm({}), direct)
        # cco rounds the slot to 4 GiB: 12 GiB per op staged, 8 GiB direct at 16k.
        self.assertTrue(8 * gib < staged <= 12 * gib, staged)
        self.assertTrue(7 * gib < direct <= 8 * gib, direct)
        self.assertEqual(vmm({"MORI_ASYNCLL_DIRECT": "1"}, n_ops=2), 2 * direct)
        # Direct rides on early notification; without it the arena is the staged one.
        off = {"MORI_ASYNCLL_DIRECT": "1", "MORI_ASYNCLL_EARLY_NOTIFY": "0"}
        self.assertEqual(vmm(off), staged)


class AsyncLLCompactReserveTest(unittest.TestCase):
    compact = SimpleNamespace(
        async_ll_segmented=False, quant_type="none", async_ll_transport="sdma", nodes=1
    )

    def reserve(self, env, mode=1, cfg=None, arch="gfx950"):
        with patch.dict(os.environ, env, clear=True):
            return EpDispatchCombineOpHip._asyncll_reserve(
                mode, cfg or self.compact, arch
            )

    def test_off_unless_asked(self):
        self.assertFalse(self.reserve({}))
        self.assertFalse(self.reserve({"MORI_ASYNCLL_COMPACT_RESERVE": "0"}))
        self.assertTrue(self.reserve({"MORI_ASYNCLL_COMPACT_RESERVE": "1"}))
        self.assertTrue(self.reserve({"MORI_ASYNCLL_COMPACT_RESERVE": "1"}, mode=2))
        with self.assertRaises(ValueError):
            self.reserve({"MORI_ASYNCLL_COMPACT_RESERVE": "yes"})

    def test_refused_where_it_cannot_run(self):
        on = {"MORI_ASYNCLL_COMPACT_RESERVE": "1"}
        seg = SimpleNamespace(async_ll_segmented=True, quant_type="none")
        quant = SimpleNamespace(
            async_ll_segmented=False, quant_type="fp8_blockwise_1x128"
        )
        for env, mode, cfg, arch in [
            (
                on,
                0,
                None,
                "gfx950",
            ),  # no early notification (LSA, several nodes, opted out)
            (on, 1, seg, "gfx950"),
            (on, 1, quant, "gfx950"),
            (on, 1, None, "gfx942"),
            (on, 1, None, "gfx1250"),
            ({**on, "MORI_ASYNCLL_DIRECT": "0"}, 1, None, "gfx950"),
            ({**on, "MORI_ASYNCLL_FUSE_META": "0"}, 1, None, "gfx950"),
        ]:
            with self.subTest(env=env, mode=mode, cfg=cfg, arch=arch):
                with self.assertRaises(ValueError):
                    self.reserve(env, mode, cfg, arch)

    def test_sequences(self):
        seq = EpDispatchCombineOpHip._asyncll_seq(self.compact, 1, True, reserve=True)
        self.assertEqual(
            seq["dispatch_send"], ("dispatch_slotassign", "dispatch_sendtransfer_seg")
        )
        self.assertEqual(
            seq["dispatch_recv"], ("dispatch_recvtransfer", "dispatch_recvfinalize_seg")
        )
        self.assertEqual(seq["combine_send"], ("combine_sendtransfer",))
        self.assertEqual(
            seq["combine_recv"], ("combine_recvtransfer", "combine_recvcopy")
        )
        fused = EpDispatchCombineOpHip._asyncll_seq(self.compact, 2, True, reserve=True)
        self.assertEqual(fused["dispatch_recv"], ("dispatch_recvfinalize_seg",))
        self.assertEqual(fused["combine_recv"], ("combine_recvcopy",))
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                EpDispatchCombineOpHip._asyncll_fuse_meta(True, False, True)
            )
            self.assertFalse(EpDispatchCombineOpHip._asyncll_fuse_meta(True, False))

    def test_words_follow_the_segmented_ones(self):
        for w in (2, 8, 16):
            with self.subTest(world_size=w):
                seg, res = asyncll_seg_words(w), asyncll_reserve_words(w)
                self.assertEqual(res["base"], seg["counts"] + w)
                self.assertEqual(res["place"], res["base"] + w)
                # The counter sits alone on a 256 B line past the place words.
                self.assertGreaterEqual(res["counter"], res["place"] + w)
                self.assertEqual(res["counter"] % 32, 0)
                self.assertEqual(res["words"], res["counter"] + 32)

    def test_regions_drop_the_landing(self):
        for scale_dim, token_nbytes, cap in (
            (0, 14336, 0),
            (224, 7168, 0),
            (224, 7168, 400),
        ):
            with self.subTest(scales=scale_dim, cap=cap):
                cfg = AsyncLLSegmentedTest.regions_cfg(scale_dim, token_nbytes, 0, cap)
                direct = dict(asyncll_regions(cfg, direct=True))
                reserve = dict(asyncll_regions(cfg, direct=True, reserve=True))
                self.assertNotIn("inter_dispatch_inp", reserve)
                self.assertEqual(
                    reserve["send_count"], asyncll_reserve_words(8)["words"] * 8
                )
                # Field rows for every destination, own rank included, in the seg staging layout.
                self.assertGreaterEqual(
                    reserve["inter_staging"],
                    asyncll_seg_layout(cfg, hidden_nbytes=0)["bytes"],
                )
                # What peers place into and combine sends from keep the compact extents.
                for name in (
                    "inter_dispatch_out",
                    "out_indices",
                    "dispatch_out_weights",
                    "disp_tok_id_to_src_tok_id",
                    "inter_combine_inp",
                    "inter_dispatch_staging",
                    "recv_token_num",
                ):
                    self.assertEqual(reserve[name], direct[name], name)
                if scale_dim:
                    self.assertEqual(reserve["out_scales"], direct["out_scales"])

    def test_per_rank_vmm_follows_the_reserve_flag(self):
        cfg = EpDispatchCombineConfig(
            rank=0,
            world_size=8,
            hidden_dim=7168,
            max_num_inp_token_per_rank=16384,
            num_experts_per_rank=32,
            num_experts_per_token=8,
            kernel_backend="hip",
            async_ll=True,
            async_ll_transport="sdma",
        )

        def vmm(env):
            with patch.dict(os.environ, env, clear=True):
                return EpDispatchCombineOp.per_rank_vmm(cfg, arch="gfx950")

        direct = vmm({"MORI_ASYNCLL_DIRECT": "1"})
        reserve = vmm({"MORI_ASYNCLL_COMPACT_RESERVE": "1"})
        self.assertLess(reserve, direct)


def _fake_plan_type(name):
    # Grid as EpAsyncLLGeometry derives it: capped copy passes, a per-peer transfer.
    field = {
        "dispatch_slotassign": "asyncLlSendBlocks",
        "combine_recvcopy_seg": "asyncLlRecvBlocks",
    }.get(name)

    class Plan:
        def __init__(self, **request):
            self.request = request
            self.bound = {}

        @classmethod
        def describe(cls, **request):
            grid = 8
            if field:
                ceiling = 128 if field == "asyncLlRecvBlocks" else 256
                grid = min(request[field] or ceiling, ceiling)
            return {"gridX": grid, "blockX": 256, "sharedBytes": 0, "cacheDir": ""}

        @property
        def info(self):
            return dict(self.describe(**self.request), cacheDir="/cache/" + name)

        def bind(self, **args):
            self.bound.update(args)

        def close(self):
            pass

    return Plan


class AsyncLLBlocksScheduleTest(unittest.TestCase):
    def parse(self, side, value):
        env = {f"MORI_ASYNCLL_{side}_BLOCKS_SCHEDULE": value}
        with patch.dict(os.environ, env, clear=True):
            return EpDispatchCombineOpHip._asyncll_blocks_schedule(side)

    def test_parse(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(EpDispatchCombineOpHip._asyncll_blocks_schedule("RECV"))
        self.assertIsNone(self.parse("RECV", " , "))
        self.assertEqual(
            self.parse("RECV", "128:64,1024:128,0:0"), ((128, 64), (1024, 128), (0, 0))
        )
        self.assertEqual(self.parse("SEND", " 0:32, 1024:8 "), ((1024, 8), (0, 32)))
        for bad in ("128", "a:8", "-1:8", "8:-1", "1:2:3", "8:16,8:32", "128:"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.parse("SEND", bad)

    def test_pick(self):
        pick = EpDispatchCombineOpHip._asyncll_pick_bucket
        buckets = ((128, "a"), (1024, "b"), (None, "c"))
        for t, want in [(0, "a"), (128, "a"), (129, "b"), (1024, "b"), (1025, "c")]:
            with self.subTest(t=t):
                self.assertEqual(pick(buckets, t), want)

    def build(self, env):
        from mori.jit.v2 import plan_api
        from mori.ops.dispatch_combine_v2 import hip_backend

        types = {}

        def plan_type(name):
            return types.setdefault(name, _fake_plan_type(name))

        seq = {
            "dispatch_send": ("dispatch_slotassign", "dispatch_sendtransfer_seg"),
            "combine_send": ("combine_sendtransfer_seg",),
            "combine_recv": ("combine_recvcopy_seg",),
        }
        base = dict(asyncLlSendBlocks=0, asyncLlRecvBlocks=64, asyncLlFusedRecvBlocks=0)
        plans_by_pass = {
            n: plan_type(n)(**base) for names in seq.values() for n in names
        }
        groups = {g: tuple(plans_by_pass[n] for n in names) for g, names in seq.items()}
        op = object.__new__(EpDispatchCombineOpHip)
        op._plans = list(plans_by_pass.values())
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(hip_backend.cb, "asyncll_plan", plan_type),
            patch.object(plan_api, "make_launch_group", tuple),
        ):
            (buckets,) = op._build_asyncll_buckets(
                SimpleNamespace(rank=1),
                lambda g: dict(base),
                plans_by_pass,
                ((seq, groups),),
            )
        return op, plans_by_pass, groups, buckets

    def test_unset_builds_nothing(self):
        op, plans_by_pass, _, buckets = self.build({})
        self.assertEqual(buckets, {})
        self.assertEqual(len(op._plans), len(plans_by_pass))

    def test_buckets_reuse_plans_the_cap_does_not_change(self):
        op, plans_by_pass, groups, buckets = self.build(
            {
                "MORI_ASYNCLL_SEND_BLOCKS_SCHEDULE": "1024:8,0:32",
                "MORI_ASYNCLL_RECV_BLOCKS_SCHEDULE": "128:64,1024:128",
            }
        )
        send = buckets["dispatch_send"]
        # The schedule's own catch-all ends it; no fallback to the single cap.
        self.assertEqual([b for b, _ in send], [1024, None])
        (slot8, xfer8), (slot32, xfer32) = (g for _, g in send)
        self.assertEqual((slot8.info["gridX"], slot32.info["gridX"]), (8, 32))
        self.assertIs(xfer8, plans_by_pass["dispatch_sendtransfer_seg"])
        self.assertIs(xfer32, xfer8)
        recv = buckets["combine_recv"]
        self.assertEqual([b for b, _ in recv], [128, 1024, None])
        # 64 is the single cap's grid, so the op's group is reused; past 1024 too.
        self.assertIs(recv[0][1], groups["combine_recv"])
        self.assertEqual(recv[1][1][0].info["gridX"], 128)
        self.assertIs(recv[2][1], groups["combine_recv"])
        new = op._plans[len(plans_by_pass) :]
        self.assertEqual(len(new), 3)
        for plan in new:
            self.assertEqual(plan.bound, {"rdmaBlockNum": 0, "replayMode": 0})
        self.assertNotIn("dispatch_recv", buckets)
        # No pass of combine_send takes the cap, so it launches as without a schedule.
        self.assertNotIn("combine_send", buckets)


class AsyncLLSdmaQueuesTest(unittest.TestCase):
    """The count AsyncLL asks cco for. cco creates queues only for a count > 0,
    so this must never return 0."""

    def queues(self, env, model=None, gpu_per_node=8):
        from mori.ops.dispatch_combine_v2.hip_backend import _asyncll_sdma_queues

        with patch.dict(os.environ, env, clear=True):
            return _asyncll_sdma_queues(model, gpu_per_node)

    def test_default_is_one_per_peer_capped_by_model(self):
        self.assertEqual(self.queues({}), 8)
        self.assertEqual(self.queues({}, gpu_per_node=4), 4)
        self.assertEqual(self.queues({}, model="mi308x"), 3)

    def test_env_wins_and_is_clamped(self):
        env = "MORI_SDMA_NUM_CHANNELS"
        self.assertEqual(self.queues({env: "3"}), 3)
        self.assertEqual(self.queues({env: "1"}, model="mi308x"), 1)
        self.assertEqual(self.queues({env: "12"}), 8)

    def test_invalid_env_falls_back_to_the_default(self):
        env = "MORI_SDMA_NUM_CHANNELS"
        for raw in ("0", "-2", "abc", ""):
            self.assertEqual(self.queues({env: raw}), 8, raw)


if __name__ == "__main__":
    unittest.main()
