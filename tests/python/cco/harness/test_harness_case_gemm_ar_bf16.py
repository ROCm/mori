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
"""The gemm_ar_bf16 case on the CPU: interface, op_harness helpers, the clean-room files.

What needs GPUs (op_harness.run, existing.py) is validated on a node, not here.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
CASE = REPO / "benchmark" / "cco" / "harness" / "cases" / "gemm_ar_bf16"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


interface = _load(CASE / "interface.py", "case_gemm_ar_bf16_interface")
harness = _load(CASE / "op_harness.py", "case_gemm_ar_bf16_op_harness")
audit = _load(CASE / "audit.py", "case_gemm_ar_bf16_audit")
run = _load(REPO / "benchmark" / "cco" / "harness" / "run.py", "case_gemm_ar_bf16_run")

GOOD = """
class Impl:
    fused_path_ran = None
    @staticmethod
    def window_bytes_for(world_size, *, m_max, n, k):
        return 0
    def __init__(self, comm, *, n, k, m_max):
        pass
    def set_weight(self, b):
        pass
    def __call__(self, a):
        return a
    def close(self):
        pass

class Partial:
    def __call__(self, a):
        return a
"""


def test_interface_lists_what_is_missing(tmp_path):
    module = tmp_path / "impl.py"
    module.write_text(GOOD)
    cls = harness.load_impl(f"{module}:Impl")
    assert interface.missing(cls) == []
    with pytest.raises(TypeError, match="window_bytes_for"):
        harness.load_impl(f"{module}:Partial")
    with pytest.raises(ValueError):
        harness.load_impl("no-colon")


def test_held_out_ms_are_seeded_and_cover_the_awkward_cases():
    ms = harness.held_out_ms(16384, 16384, 3, seed=0)
    assert ms[0] == 1 and any(m % 256 for m in ms[1:]) and 16384 not in ms
    assert ms == harness.held_out_ms(16384, 16384, 3, seed=0)
    assert ms != harness.held_out_ms(16384, 16384, 3, seed=1)
    assert all(1 <= m <= 16384 for m in harness.held_out_ms(16384, 16384, 6, seed=3))


def test_merge_takes_the_worst_rank():
    a = [
        {
            "kind": "eager",
            "m": 1,
            "salt": 0,
            "rel_l2": 1e-3,
            "pass": True,
            "fused_path_ran": True,
        }
    ]
    b = [
        {
            "kind": "eager",
            "m": 1,
            "salt": 0,
            "rel_l2": 9e-3,
            "pass": False,
            "fused_path_ran": None,
        }
    ]
    (case,) = harness.merge([a, b])
    assert (
        case["rel_l2"] == 9e-3
        and case["pass"] is False
        and case["fused_path_ran"] is None
    )


def test_operands_and_reference_are_deterministic_per_rank():
    torch = pytest.importorskip("torch")
    a0 = harness.activation(0, 0, 0, 8, 16, "cpu")
    assert a0.dtype == torch.bfloat16 and a0.shape == (8, 16)
    assert torch.equal(a0, harness.activation(0, 0, 0, 8, 16, "cpu"))
    assert not torch.equal(a0, harness.activation(0, 1, 0, 8, 16, "cpu"))
    ref = harness.reference(0, 2, 0, 8, 4, 16, "cpu")
    manual = sum(
        harness.activation(0, r, 0, 8, 16, "cpu").float()
        @ harness.weight(0, r, 4, 16, "cpu").float().t()
        for r in range(2)
    )
    assert torch.allclose(ref, manual)
    assert harness.rel_l2(ref.to(torch.bfloat16), ref) < 1e-2
    assert harness.rel_l2(ref, ref) == 0.0


def _line(kind, payload):
    if kind == "use":
        block = {
            "type": "tool_use",
            "id": "t1",
            "name": "Bash",
            "input": {"command": payload},
        }
    else:
        block = {"type": "tool_result", "tool_use_id": "t1", "content": payload}
    return json.dumps(
        {
            "type": "assistant" if kind == "use" else "user",
            "message": {"content": [block]},
        }
    )


def test_audit_separates_access_from_exposure(tmp_path):
    session = tmp_path / "session"
    (session / "subagents").mkdir(parents=True)
    (session / "tool-results").mkdir()
    (session / "subagents" / "agent-1.jsonl").write_text(
        _line("use", "cat python/mori/ops/gemm_ar/kernels_fused.py") + "\n"
    )
    (tmp_path / "main.jsonl").write_text(
        _line("use", "ls python/mori/ops")
        + "\n"
        + _line("result", "x/mori/ops/gemm_ar/op.py:12: def f")
        + "\n"
    )
    (session / "tool-results" / "r.txt").write_text("nothing here")
    findings = audit.audit(
        [tmp_path / "main.jsonl", session],
        audit.read_patterns(CASE / "hidden_paths.txt"),
    )
    kinds = sorted((f["kind"], f["pattern"]) for f in findings)
    assert kinds == [
        ("access", "mori/ops/gemm_ar/kernels_fused.py"),
        ("exposure", "mori/ops/gemm_ar/op.py"),
    ]
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        _line("use", "cat python/mori/ops/gemm_ar/_gemm_a16w16_8wave.py") + "\n"
    )
    assert audit.main([str(clean)]) == 0
    assert audit.main([str(tmp_path / "main.jsonl")]) == 1
    assert audit.main([str(tmp_path / "absent.jsonl")]) == 2


def test_clean_room_files_agree():
    patterns = audit.read_patterns(CASE / "hidden_paths.txt")
    deny = json.loads((CASE / "settings.implementer.json").read_text())["permissions"][
        "deny"
    ]
    for p in patterns:
        assert any(p in rule for rule in deny), p
    assert "Bash(git push:*)" in deny
    for p in ("_gemm_a8w8_8wave.py", "_gemm_a16w16_8wave.py", "_compat.py"):
        assert not any(p in pattern for pattern in patterns), p


def test_arms_and_contract_point_at_real_files():
    for name in ("arms.dev.yaml", "arms.verify.yaml"):
        arms = run.load_arms(CASE / name)
        assert arms[0]["name"] == "existing" and all(
            (REPO / a["bench"]).is_file() for a in arms
        )
    yaml = pytest.importorskip("yaml")
    contract = yaml.safe_load((CASE / "contract.yaml").read_text())
    assert (REPO / contract["interface"]).is_file()
    for key in ("hidden", "settings", "audit"):
        assert (REPO / contract["clean_room"][key]).is_file(), key
    assert contract["targets"]["within_existing_pct"] == 5
