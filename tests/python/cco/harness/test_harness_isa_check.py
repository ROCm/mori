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
"""benchmark/cco/harness/isa_check.py on the CPU, against small hand-written dumps.

The dumps follow the layout of a FlyDSL ``*_final_isa.s`` on gfx950: an
``.amdgcn_target`` line, ``.type ...,@function`` and a label per function,
``.LBB`` labels, ``.Lfunc_end``, ``.set .L<kernel>.num_vgpr`` symbols and an
``amdhsa.kernels`` metadata block.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[4] / "benchmark" / "cco" / "harness"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"cco_harness_{name}", HARNESS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


isa_check = _load("isa_check")

MFMA = "\tv_mfma_f32_16x16x32_bf16 a[0:3], v[40:43], v[44:47], a[0:3]"

# One release before the counter atomic and one acquire after it, outside every loop:
# what an acq_rel atomic emits once per block (G950-SYNC-01).
EPILOGUE = [
    "\tglobal_store_dwordx4 v[10:11], v[0:3], off sc0 sc1",
    "\ts_waitcnt vmcnt(0)",
    "\ts_barrier",
    "\tbuffer_wbl2 sc0 sc1",
    "\ts_waitcnt vmcnt(0)",
    "\tglobal_atomic_add_u32 v5, v[12:13], v6, off sc0 sc1",
    "\ts_waitcnt vmcnt(0)",
    "\tbuffer_inv sc0 sc1",
]


def k_loop(label, early_wait=False, extra=(), mfma=8):
    """Four loads, then MFMAs. Without ``early_wait`` half of the MFMAs run before the wait."""
    lines = [f"{label}:"]
    lines += [
        f"\tbuffer_load_dwordx4 v[{4 * i}:{4 * i + 3}], v20, s[8:11], 0 offen"
        for i in range(4)
    ]
    if early_wait:
        lines += ["\ts_waitcnt vmcnt(0)"] + [MFMA] * mfma
    else:
        lines += (
            [MFMA] * (mfma // 2)
            + ["\ts_waitcnt vmcnt(0)"]
            + [MFMA] * (mfma - mfma // 2)
        )
    lines += list(extra)
    lines += [
        "\ts_add_u32 s20, s20, 1",
        "\ts_cmp_lg_u32 s20, s21",
        f"\ts_cbranch_scc1 {label}",
    ]
    return lines


def poll_loop(label, extra=()):
    lines = [
        f"{label}:",
        "\ts_sleep 1",
        "\tglobal_load_dword v1, v[2:3], off sc0 sc1",
        "\ts_waitcnt vmcnt(0)",
    ]
    lines += list(extra)
    lines += ["\tv_cmp_lt_u32_e32 vcc, v1, v3", f"\ts_cbranch_vccnz {label}"]
    return lines


HELPER = """\t.p2align\t2
\t.type\thelper,@function
helper:
\ts_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
\tbuffer_inv sc0 sc1
\ts_setpc_b64 s[30:31]
.Lfunc_end0:
\t.size\thelper, .Lfunc_end0-helper
"""

CALL_HELPER = [
    "\ts_getpc_b64 s[16:17]",
    "\ts_add_u32 s16, s16, helper@rel32@lo+4",
    "\ts_addc_u32 s17, s17, helper@rel32@hi+12",
    "\ts_swappc_b64 s[30:31], s[16:17]",
]


def dump(
    tmp_path,
    body,
    name="fused",
    *,
    vgpr=128,
    vgpr_spill=0,
    sgpr_spill=0,
    scratch=0,
    lds=65536,
    helper=False,
):
    """Write a one-kernel dump and return its path."""
    index = 1 if helper else 0
    text = f"""\t.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
\t.amdhsa_code_object_version 6
\t.text
{HELPER if helper else ""}\t.globl\t{name}
\t.p2align\t8
\t.type\t{name},@function
{name}:
; %bb.0:
{chr(10).join(body)}
\ts_endpgm
.Lfunc_end{index}:
\t.size\t{name}, .Lfunc_end{index}-{name}
\t.set .L{name}.num_vgpr, {vgpr}
\t.set .L{name}.num_agpr, 0
\t.set .L{name}.numbered_sgpr, 40
\t.set .L{name}.private_seg_size, {scratch}
\t.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: {lds}
    .max_flat_workgroup_size: 256
    .name:           {name}
    .private_segment_fixed_size: {scratch}
    .sgpr_count:     46
    .sgpr_spill_count: {sgpr_spill}
    .vgpr_count:     {vgpr}
    .vgpr_spill_count: {vgpr_spill}
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...
\t.end_amdgpu_metadata
"""
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "22_final_isa.s"
    path.write_text(text)
    return path


def check(path, *extra):
    out = path.parent / "report.json"
    status = isa_check.main([str(path), "--json", str(out), *extra])
    return status, json.loads(out.read_text()) if out.exists() else None


def codes(report, level):
    return [f["code"] for f in report["findings"] if f["level"] == level]


def test_clean_fused_kernel_passes(tmp_path):
    body = k_loop(".LBB0_1") + EPILOGUE + poll_loop(".LBB0_3")
    status, report = check(dump(tmp_path, body))
    assert status == 0 and report["result"] == "PASS"
    kernel = report["kernels"][0]
    assert sorted(lp["kind"] for lp in kernel["loops"]) == ["k-loop", "poll"]
    assert kernel["k_loop"]["mfma"] == 8 and kernel["k_loop"]["loads"] == 4
    assert kernel["k_loop"]["mean_load_wait"] == pytest.approx((8 + 7 + 6 + 5) / 4)
    assert [op["loop"] for op in kernel["cache_ops"]] == [None, None]
    assert kernel["stores"]["write_through"] == 1
    assert (
        kernel["resources"]["vgpr"] == 128 and kernel["resources"]["arch_vgpr"] == 128
    )


def test_acquire_in_poll_loop_fails(tmp_path):
    body = k_loop(".LBB0_1") + poll_loop(".LBB0_3", extra=["\tbuffer_inv sc0 sc1"])
    status, report = check(dump(tmp_path, body))
    assert status == 1
    assert "cache-op-in-loop" in codes(report, "FAIL")
    assert "poll loop .LBB0_3" in report["findings"][0]["message"]


def test_release_in_k_loop_fails(tmp_path):
    body = k_loop(".LBB0_1", extra=["\tbuffer_wbl2 sc1"])
    status, report = check(dump(tmp_path, body))
    assert status == 1
    assert codes(report, "FAIL") == ["cache-op-in-loop"]


def test_release_once_per_tile_in_an_outer_loop_warns(tmp_path):
    # a persistent tile loop around the K loop, releasing once per tile
    body = (
        [".LBB0_0:"]
        + k_loop(".LBB0_1")
        + EPILOGUE[:4]
        + ["\ts_cmp_lg_u32 s22, s23", "\ts_cbranch_scc1 .LBB0_0"]
    )
    path = dump(tmp_path, body)
    status, report = check(path)
    assert status == 0
    assert codes(report, "WARN") == ["cache-op-in-loop"]
    assert check(path, "--strict")[0] == 1


def test_cache_op_reached_through_a_call_in_a_loop_fails(tmp_path):
    body = (
        [".LBB1_2:", "\ts_sleep 1"]
        + CALL_HELPER
        + ["\tv_cmp_lt_u32_e32 vcc, v1, v3", "\ts_cbranch_vccnz .LBB1_2"]
    )
    status, report = check(dump(tmp_path, body, helper=True))
    assert status == 1
    assert codes(report, "FAIL") == ["cache-op-via-call-in-loop"]
    kernel = report["kernels"][0]
    assert kernel["calls"] == [
        {
            "line": kernel["calls"][0]["line"],
            "function": "fused",
            "callee": "helper",
            "loop": ".LBB1_2",
        }
    ]
    assert [op["function"] for op in kernel["cache_ops"]] == ["helper"]


def test_scratch_fails_and_sgpr_spills_warn(tmp_path):
    status, report = check(
        dump(tmp_path, k_loop(".LBB0_1"), vgpr_spill=3, sgpr_spill=2, scratch=16)
    )
    assert status == 1
    assert codes(report, "FAIL") == ["scratch"]
    assert codes(report, "WARN") == ["sgpr-spill"]


def test_lost_latency_hiding_against_the_baseline_fails(tmp_path):
    base = dump(tmp_path, k_loop(".LBB0_1"), name="comm_off")
    fused = dump(tmp_path, k_loop(".LBB0_1", early_wait=True) + EPILOGUE, name="fused")
    status, report = check(fused, "--baseline", str(base))
    assert status == 1
    assert codes(report, "FAIL") == ["k-loop-latency-hiding"]
    assert report["delta"]["mean_load_wait"] == pytest.approx(
        (4 + 3 + 2 + 1) / 4 - (8 + 7 + 6 + 5) / 4
    )
    # the same schedule passes
    same = dump(tmp_path, k_loop(".LBB0_1") + EPILOGUE, name="fused_same")
    assert check(same, "--baseline", str(base))[0] == 0


def test_mfma_count_change_and_more_lds_against_the_baseline(tmp_path):
    base = dump(tmp_path, k_loop(".LBB0_1"), name="comm_off", lds=65536)
    fused = dump(tmp_path, k_loop(".LBB0_1", mfma=10), name="fused", lds=98304)
    status, report = check(fused, "--baseline", str(base))
    assert status == 1
    assert "k-loop-mfma-count" in codes(report, "FAIL")
    assert "more-lds_bytes" in codes(report, "WARN")


def test_stores_that_are_not_write_through(tmp_path):
    body = (
        k_loop(".LBB0_1") + EPILOGUE + ["\tglobal_store_dwordx4 v[10:11], v[0:3], off"]
    )
    path = dump(tmp_path, body)
    status, report = check(path)
    assert status == 0 and codes(report, "WARN") == ["store-policy"]
    assert check(path, "--require-write-through")[0] == 1


@pytest.mark.parametrize("args", [["nowhere"], ["DIR", "--kernel", "^nomatch$"]])
def test_unreadable_input_exits_2(tmp_path, args):
    path = dump(tmp_path, k_loop(".LBB0_1"))
    args = [str(path.parent) if a == "DIR" else a for a in args]
    assert isa_check.main(args) == 2


def test_directory_with_two_dumps_exits_2(tmp_path):
    path = dump(tmp_path, k_loop(".LBB0_1"))
    (path.parent / "23_final_isa.s").write_text(path.read_text())
    assert isa_check.main([str(path.parent)]) == 2


def test_non_gfx950_target_warns(tmp_path):
    path = dump(tmp_path, k_loop(".LBB0_1"))
    path.write_text(path.read_text().replace("gfx950", "gfx1250"))
    status, report = check(path)
    assert status == 0 and "target" in codes(report, "WARN")
