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
"""expected.tsv rules of the torch symm_mem conformance runner, without a GPU."""
import argparse
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent / "torch_conformance"
VER_GATE = (
    "skip_if_rocm_ver_lessthan_multiprocess: ROCm (7, 14, 1) is available but "
    "(10, 1) required"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_conformance", HERE / "run_conformance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(runner, tmp_path, view, cause, expected=HERE / "expected.tsv"):
    """Exit status of the report for one skipped test with this reason."""
    row = ["SymmetricMemoryTest::test_x", "SKIPPED", "0", "1", "0", cause]
    (tmp_path / "summary.tsv").write_text(
        "\t".join(runner.COLUMNS)
        + "\ntest_symmetric_memory.py::"
        + "\t".join(row)
        + "\n"
    )
    args = argparse.Namespace(
        report_only=False,
        expected=expected,
        view=view,
        backend="MORI",
        as_cuda=False,
        verbose=False,
        baseline=None,
    )
    info = {"version": "2.15.0+rocm7.14", "git": "x", "hip": "7.14"}
    return runner.report(args, tmp_path, info)


@pytest.mark.parametrize(
    "view,cause,status",
    [
        ("A", "Test skipped for ROCm", 0),
        ("B", "Test skipped for ROCm", 0),
        ("C", "Test skipped for ROCm", 1),
        ("A", VER_GATE, 0),
        ("B", VER_GATE, 1),
        ("C", VER_GATE, 1),
        ("A", "skip_if_rocm_arch_multiprocess: test skipped on ('gfx950',)", 0),
        ("C", "skip_if_rocm_arch_multiprocess: test skipped on ('gfx950',)", 1),
        ("A", "symmetric memory hangs on MI350 CI runners before ROCm 10.1", 0),
        ("A", "a new gate before ROCm 10.1", 1),
        ("C", "multicast support is not available", 0),
    ],
)
def test_skip_excused_only_in_its_views(runner, tmp_path, view, cause, status):
    assert _report(runner, tmp_path, view, cause) == status


def test_view_field_is_checked(runner, tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("SKIP\t*\tAD\t*\t.\tcat\tnote\n")
    with pytest.raises(SystemExit, match="view must be"):
        runner.load_expected(bad, "2.15.0")
