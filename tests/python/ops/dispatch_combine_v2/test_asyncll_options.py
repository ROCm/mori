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

from mori.ops.dispatch_combine_v2.hip_backend import EpDispatchCombineOpHip


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

    def test_other_architectures_require_explicit_opt_in(self):
        self.assertEqual(self.resolve({}, arch="gfx950")[:2], (0, False))
        self.assertEqual(
            self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "1"}, arch="gfx950")[:2],
            (1, True),
        )

    def test_reject_inconsistent_or_invalid_options(self):
        for option in ("MORI_ASYNCLL_FUSED_RECV", "MORI_ASYNCLL_SKIP_RESET"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "0", option: "1"})
        with self.assertRaises(ValueError):
            self.resolve({"MORI_ASYNCLL_EARLY_NOTIFY": "yes"})


if __name__ == "__main__":
    unittest.main()
