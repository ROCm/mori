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
"""gfx1250-strict arch detection and the ASIC-revision tuning-table keys. No GPU."""

import os
import re
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mori.jit import config
from mori.ops import utils as gpu
from mori.ops.dispatch_combine_v2 import hip_tuning_configs as tc
from mori.ops.dispatch_combine_v2 import tuning_configs as flydsl_tc

REPO = Path(__file__).resolve().parents[4]


class DetectArchTest(unittest.TestCase):
    def test_whole_target_names(self):
        for text, want in [
            ("gfx1250-strict", "gfx1250-strict"),
            ("amdgcn-amd-amdhsa--gfx1250-strict", "gfx1250-strict"),
            ("gfx1250", "gfx1250"),
            ("gfx1250;gfx1250-strict", "gfx1250"),
            ("gfx942;gfx950;gfx1250", "gfx942"),
            ("gfx90a", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(config._match_arch(text), want)

    def test_env_override(self):
        with patch.dict(os.environ, {"MORI_GPU_ARCHS": "gfx1250-strict"}):
            self.assertEqual(config.detect_gpu_arch(), "gfx1250-strict")

    def test_a_build_list_defers_to_the_gpu(self):
        for listed, device, want in [
            ("gfx1250;gfx1250-strict", "gfx1250-strict", "gfx1250-strict"),
            ("gfx1250;gfx1250-strict", "gfx1250", "gfx1250"),
            ("gfx942;gfx950", "gfx950", "gfx950"),
            ("gfx942;gfx950;gfx1250", "gfx1250-strict", "gfx1250-strict"),
            ("gfx942;gfx950", None, "gfx942"),  # no GPU: the list's first
            ("gfx1250", "gfx1250-strict", "gfx1250"),  # one arch overrides
        ]:
            with (
                self.subTest(listed=listed, device=device),
                patch.dict(os.environ, {"MORI_GPU_ARCHS": listed}),
                patch.object(config, "enumerated_gpu_arch", return_value=device),
                patch.object(config, "_rocminfo_gpu_arch", return_value=None),
            ):
                self.assertEqual(config.detect_gpu_arch(), want)


class AsicRevisionTest(unittest.TestCase):
    def test_kfd_capability_bits(self):
        for cap, want in [(0x80000040, 0), ((1 << 22) | 0x80000040, 1), (5 << 22, 5)]:
            with patch.object(gpu, "_first_gpu_node", return_value={"capability": cap}):
                self.assertEqual(gpu.asic_revision(), want)
        with patch.object(gpu, "_first_gpu_node", return_value={}):
            self.assertIsNone(gpu.asic_revision())


class TuningKeyTest(unittest.TestCase):
    SHAPE = (4, 7168, 8)
    REV0 = dict(rev=0, arch="gfx1250")
    REV0_STRICT = dict(rev=None, arch="gfx1250-strict")  # strict alone implies rev 0
    REV1 = dict(rev=1, arch="gfx1250")

    def patched(self, dev, cus=0):
        return (
            patch.object(tc, "_device_key", return_value="gfx1250"),
            patch.object(tc._gpu, "asic_revision", return_value=dev["rev"]),
            patch.object(tc, "_detected_arch", return_value=dev["arch"]),
            patch.object(tc._gpu, "cu_count", return_value=cus),  # 0: no clamp
        )

    def keys(self, dev):
        a, b, c, d = self.patched(dev)
        with a, b, c, d:
            return tc._device_keys()

    def test_rev0_tries_its_own_key_first(self):
        self.assertEqual(self.keys(self.REV0), ("gfx1250-strict", "gfx1250"))
        self.assertEqual(self.keys(self.REV0_STRICT), ("gfx1250-strict", "gfx1250"))
        self.assertEqual(self.keys(self.REV1), ("gfx1250",))

    def lookup(self, dev, tables, shape=SHAPE, cus=0, **kw):
        a, b, c, d = self.patched(dev, cus)
        with a, b, c, d, patch.dict(tc._DISPATCH_TABLE, tables):
            return tc.lookup(*shape, **kw)

    def test_rev0_falls_back_to_gfx1250_rows(self):
        shape = (8, 7168, 6)  # no gfx1250-strict row
        rev0 = self.lookup(self.REV0, {}, shape)
        self.assertEqual(rev0, self.lookup(self.REV1, {}, shape))
        self.assertIsNotNone(rev0["schedule"])

    def test_ep4_hidden5120_rows(self):
        # (bound, dispatch block, warp, combine block, warp), both DeepSeek-V4.1-Flash MoEs.
        topk6_quant = (
            (128, 64, 4, 64, 8),
            (256, 64, 4, 128, 8),
            (512, 64, 8, 128, 8),
            (1024, 128, 8, 128, 8),
            (None, 256, 8, 128, 8),
        )
        topk3_quant = (
            (128, 64, 4, 64, 8),
            (256, 64, 4, 128, 8),
            (512, 64, 8, 128, 8),
            (1024, 128, 8, 256, 4),
            (None, 256, 8, 256, 4),
        )
        want = {
            (6, 96, "bf16"): (
                (128, 64, 4, 64, 8),
                (256, 64, 4, 128, 8),
                (512, 128, 4, 128, 8),
                (2048, 256, 4, 128, 8),
                (None, 128, 16, 128, 8),
            ),
            (6, 96, "fp8"): topk6_quant,
            (6, 96, "fp4"): topk6_quant,
            (3, 32, "bf16"): (
                (128, 64, 4, 64, 8),
                (256, 64, 4, 128, 8),
                (512, 128, 4, 128, 8),
                (2048, 256, 4, 256, 4),
                (None, 128, 16, 256, 4),
            ),
            (3, 32, "fp8"): topk3_quant,
            (3, 32, "fp4"): topk3_quant,
        }
        for (topk, epr, dtype), schedule in want.items():
            t = self.lookup(
                self.REV1, {}, (4, 5120, topk), dtype=dtype, experts_per_rank=epr
            )
            with self.subTest(topk=topk, dtype=dtype):
                self.assertEqual(tuple(map(tuple, t["schedule"])), schedule)

    def test_rev0_row_wins(self):
        row = {"gfx1250-strict": {self.SHAPE + (None,): {None: ((None, 32, 4),)}}}
        schedule = self.lookup(self.REV0, row)["schedule"]
        self.assertEqual({(b[1], b[2]) for b in schedule}, {(32, 4)})

    def test_rev0_takes_no_ll128_plans(self):
        args = (4, 7168, 6, "fp4", 64)
        for dev, want in (
            (self.REV0, False),
            (self.REV0_STRICT, False),
            (self.REV1, True),
        ):
            a, b, c, d = self.patched(dev)
            with a, b, c, d:
                got = tc.ll128(*args, quant_type="fp4_blockwise")
            self.assertEqual(got is not None, want, dev)

    @staticmethod
    def rev0_widest_grid():
        rows = [tc._COMBINE_TABLE["gfx1250-strict"]]
        rows += list(tc._DISPATCH_TABLE["gfx1250-strict"].values())
        rows += [t["gfx1250-strict"] for t in tc._COMBINE_PUSH_TABLE.values()]
        return max(b for row in rows for bs in row.values() for _, b, _ in bs)

    def test_rev0_rows_stay_below_the_gfx1250_grids(self):
        self.assertLess(self.rev0_widest_grid(), 256)

    def test_rev0_ep4_takes_its_own_rows(self):
        cap = self.rev0_widest_grid()
        kw = dict(dtype="fp4", quant_type="fp4_blockwise")
        for dev in (self.REV0, self.REV0_STRICT):
            rev0 = self.lookup(dev, {}, (4, 7168, 9), **kw)["schedule"]
            self.assertLessEqual(max(max(b[1], b[3]) for b in rev0), cap)
        rev1 = self.lookup(self.REV1, {}, (4, 7168, 9), **kw)["schedule"]
        self.assertGreater(max(max(b[1], b[3]) for b in rev1), cap)

    def test_every_rev0_lookup_fits_once_capped(self):
        # A block holds a CU and waits for the whole grid. Shapes without a rev-0 row
        # fall back to gfx1250 rows, some wider than these parts; every lookup path
        # must cap them at the CU count.
        cap = self.rev0_widest_grid()
        shapes = set()
        for table in (tc._DISPATCH_TABLE, tc._COMBINE_TABLE):
            for dev in ("gfx1250", "gfx1250-strict"):
                shapes |= set(table.get(dev, {}))
        for quant in tc._COMBINE_PUSH_TABLE.values():
            for dev in ("gfx1250", "gfx1250-strict"):
                shapes |= set(quant.get(dev, {}))
        quants = ["none", *tc._COMBINE_PUSH_TABLE]
        for world, hidden, topk, epr in sorted(shapes, key=str):
            for dtype in ("bf16", "fp8", "fp4"):
                for quant in quants:
                    t = self.lookup(
                        self.REV0,
                        {},
                        (world, hidden, topk),
                        cus=cap,
                        dtype=dtype,
                        experts_per_rank=epr,
                        quant_type=quant,
                    )
                    blocks = [t["dispatch_block_num"], t["combine_block_num"]]
                    blocks += [max(b[1], b[3]) for b in t["schedule"] or ()]
                    with self.subTest(shape=(world, topk, epr), dtype=dtype, q=quant):
                        self.assertLessEqual(max(blocks), cap)

    def test_lookups_clamp_blocks_to_the_cu_count(self):
        kw = dict(dtype="fp4", quant_type="fp4_blockwise")
        t = self.lookup(self.REV1, {}, (4, 7168, 9), cus=160, **kw)
        self.assertLessEqual(max(max(b[1], b[3]) for b in t["schedule"]), 160)
        with (
            patch.object(flydsl_tc, "_device_key", return_value="gfx1250"),
            patch.object(flydsl_tc.gpu_utils, "cu_count", return_value=160),
        ):
            t = flydsl_tc.lookup(4, 7168, 6, dtype="bf16")
        self.assertIsNotNone(t["schedule"])
        self.assertLessEqual(max(max(b[1], b[3]) for b in t["schedule"]), 160)


class DeviceMacroTest(unittest.TestCase):
    # gfx1250-strict defines __gfx1250_strict__, not __gfx1250__, so device code tests
    # MORI_GFX125X (mori/core/arch.hpp) or CCO_GFX125X (cco.hpp), defined only here.
    FAMILY = "#if defined(__gfx1250__) || defined(__gfx1251__) || defined(__gfx1250_strict__)"
    DEFINERS = {"include/mori/core/arch.hpp", "include/mori/cco/cco.hpp"}
    SUFFIXES = {".h", ".hpp", ".hip", ".cpp", ".cu", ".cuh", ".py"}

    def test_both_definers_test_the_same_family(self):
        for rel in sorted(self.DEFINERS):
            lines = [s.strip() for s in (REPO / rel).read_text().splitlines()]
            with self.subTest(rel=rel):
                self.assertIn(self.FAMILY, lines)

    def test_v1_jit_passes_the_hosts_family_decision(self):
        from mori.jit import core

        for arch, want in [
            ("gfx1250", ["-DMORI_HOST_GFX125X=1"]),
            ("gfx1250-strict", ["-DMORI_HOST_GFX125X=1"]),
            ("gfx950", []),
        ]:
            with self.subTest(arch=arch):
                self.assertEqual(core._arch_family_defines(arch), want)

    def test_no_per_target_gfx125x_macros(self):
        pat = re.compile(r"__gfx125[0-9]__|__gfx1250_strict__")
        hits = []
        for top in ("include", "src", "python/mori"):
            for f in (REPO / top).rglob("*"):
                if f.suffix not in self.SUFFIXES or not f.is_file():
                    continue
                rel = f.relative_to(REPO).as_posix()
                if "/_jit-sources/" in rel:  # the build's copy of include/ and src/
                    continue
                for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    s = line.strip()
                    comment = ("#",) if f.suffix == ".py" else ("//", "/*", "*")
                    if not pat.search(s) or s.startswith(comment):
                        continue
                    if rel in self.DEFINERS and s == self.FAMILY:
                        continue
                    hits.append(f"{rel}:{n}: {s}")
        self.assertEqual(hits, [], "use MORI_GFX125X / CCO_GFX125X")


class DeviceMacroCompileTest(unittest.TestCase):
    """arch.hpp as the device compiler sees it, for each target it accepts."""

    CASES = {"gfx950": "0 0", "gfx1250": "1 1", "gfx1250-strict": "1 1"}
    PROBE = (
        '#include "mori/core/arch.hpp"\n#if defined(MORI_GFX125X)\n1\n#else\n0\n#endif\n'
        + "#if defined(MORI_WAVE32)\n1\n#else\n0\n#endif\n"
    )

    @staticmethod
    def clang():
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        for rel in ("lib/llvm/bin/clang++", "llvm/bin/clang++"):
            path = Path(rocm) / rel
            if path.is_file():
                return str(path)
        return None

    def test_family_macros(self):
        clang = self.clang()
        if clang is None:
            self.skipTest("no ROCm clang")
        for arch, want in self.CASES.items():
            cmd = [
                clang,
                "-x",
                "hip",
                f"--offload-arch={arch}",
                "--cuda-device-only",
                "-nogpulib",
                "-nogpuinc",
                "-E",
                "-P",
                "-I",
                str(REPO / "include"),
                "-",
            ]
            r = subprocess.run(cmd, input=self.PROBE, capture_output=True, text=True)
            with self.subTest(arch=arch):
                if r.returncode != 0:
                    self.skipTest(f"{arch}: {r.stderr.strip().splitlines()[-1:]}")
                self.assertEqual(" ".join(r.stdout.split()), want)


def _gfx1250_gpus():
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return torch.cuda.device_count() if arch == "gfx1250" else 0


@unittest.skipUnless(_gfx1250_gpus() >= 4, "needs 4 gfx1250 GPUs")
class Ep4LdsLimitTest(unittest.TestCase):
    """EP4 hip dispatch/combine at the 64x16 dispatch geometry, eager and graph.

    At hidden 5120 bf16 this launch used to need more LDS than a workgroup has, so it
    was never scheduled and every rank waited forever; a hang shows up as a timeout.
    """

    TIMEOUT = 300

    def run_bench(self, hidden, topk, epr):
        env = dict(
            os.environ,
            BACKENDS="hip",
            DISP="bf16",
            COMB_MODE="pull",
            HIDDEN=str(hidden),
            TOPK=str(topk),
            EPR=str(epr),
            DBN="64",
            DWPB="16",
            CBN="64",
            CWPB="8",
            MODES="eager,graph",
            SWEEP="64,512",
            ITERS="5",
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(Path(__file__).with_name("bench_ep.py")),
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=self.TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            self.fail(
                f"hidden {hidden} 64x16 did not finish in {self.TIMEOUT} s:\n{out}"
            )
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn("# PASS: 0 failed", out)

    def test_64x16_dispatch_completes(self):
        for hidden, topk, epr in (
            (5120, 6, 96),
            (5120, 3, 32),
            (4096, 6, 96),
            (7168, 6, 96),
        ):
            with self.subTest(hidden=hidden, topk=topk):
                self.run_bench(hidden, topk, epr)


if __name__ == "__main__":
    unittest.main()
