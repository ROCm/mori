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

from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig
from mori.ops.dispatch_combine_v2.hip_backend import EpDispatchCombineOpHip
from mori.ops.dispatch_combine_v2.internode_regions import (
    asyncll_regions,
    asyncll_seg_layout,
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

    def test_fused_receive_grid_ceiling_reaches_only_the_fused_mode(self):
        resolve = EpDispatchCombineOpHip._asyncll_fused_recv_ctas
        for env, mode, expected in [
            ({}, 2, 0),
            ({"MORI_ASYNCLL_FUSED_RECV_CTAS": "64"}, 2, 64),
            ({"MORI_ASYNCLL_FUSED_RECV_CTAS": "64"}, 1, 0),
            ({"MORI_ASYNCLL_FUSED_RECV_CTAS": "64"}, 0, 0),
        ]:
            with (
                self.subTest(env=env, mode=mode),
                patch.dict(os.environ, env, clear=True),
            ):
                self.assertEqual(resolve(mode), expected)
        for value in ("-1", "half"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with patch.dict(
                    os.environ, {"MORI_ASYNCLL_FUSED_RECV_CTAS": value}, clear=True
                ):
                    resolve(2)

    def test_fused_receive_ceiling_follows_the_cu_mask(self):
        resolve = EpDispatchCombineOpHip._asyncll_fused_recv_ctas
        for env, dev, expected in [
            ({}, 0, 0),  # unrestricted: the native default, half the device
            ({"HSA_CU_MASK": "0-7:0-63"}, 3, 32),
            ({"HSA_CU_MASK": "1:0-15,32-47;0:0-127"}, 1, 16),
            ({"HSA_CU_MASK": "1:0-15,32-47;0:0-127"}, 0, 64),
            ({"HSA_CU_MASK": "1:0-63"}, 0, 0),  # mask names other GPUs only
            ({"HSA_CU_MASK": "0:0-511"}, 0, 0),  # not a restriction
            ({"HSA_CU_MASK": "0:0x0f"}, 0, 0),  # unreadable: whole device
            ({"HSA_CU_MASK": "0-7:0-63", "MORI_ASYNCLL_FUSED_RECV_CTAS": "48"}, 0, 48),
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
    def test_sequences(self):
        seq = EpDispatchCombineOpHip._asyncll_seg_seq()
        self.assertEqual(
            seq["dispatch_send"],
            (
                "dispatch_slotassign",
                "dispatch_sendcopy_seg",
                "dispatch_sendtransfer_seg",
            ),
        )
        self.assertEqual(
            seq["dispatch_recv"], ("dispatch_recvtransfer", "dispatch_recvfinalize_seg")
        )
        self.assertEqual(seq["combine_send"], ("combine_sendtransfer_seg",))
        self.assertEqual(
            seq["combine_recv"], ("combine_recvtransfer", "combine_recvcopy_seg")
        )

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


if __name__ == "__main__":
    unittest.main()
