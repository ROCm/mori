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
guard = _load(CASE / "guard.py", "case_gemm_ar_bf16_guard")

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


HIDDEN_OP = """import torch
SECRET_A = "the chunk schedule rotates the tile order per destination"
SECRET_B = "every destination gets its own submit lock and counter"
SECRET_C = "the drain kernel waits for the last tile of each chunk"
SECRET_D = "the gather uses lsa pulls once all reductions finished"
SHARED = "this line also lives in a visible file somewhere else"
p.add_argument("--sdma-queues", type=int, default=8)
"""
SECRETS = HIDDEN_OP.splitlines()[1:5]


def _git(repo, *args):
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )


@pytest.fixture
def mini_repo(tmp_path):
    """A checkout with one hidden file, one visible sibling and the hidden list."""
    repo = tmp_path / "repo"
    files = {
        "python/mori/ops/gemm_ar/op.py": HIDDEN_OP,
        "python/mori/ops/gemm_ag/op.py": 'SHARED = "this line also lives in a visible '
        'file somewhere else"\n',
        "hidden_paths.txt": "# hidden\nmori/ops/gemm_ar/op.py\n",
    }
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    patterns = audit.read_patterns(repo / "hidden_paths.txt")
    fp = audit.fingerprints(audit.tree_files(repo, "HEAD"), patterns)
    return repo, patterns, fp


def _use(cid, tool, inp):
    block = {"type": "tool_use", "id": cid, "name": tool, "input": inp}
    return {"type": "assistant", "message": {"content": [block]}}


def _result(cid, text, error=False):
    block = {
        "type": "tool_result",
        "tool_use_id": cid,
        "content": text,
        "is_error": error,
    }
    return {"type": "user", "message": {"content": [block]}}


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _kinds(findings):
    return sorted(
        ((f["kind"], f["call_id"]) for f in findings), key=lambda k: (k[0], k[1] or "")
    )


def test_fingerprints_keep_only_text_no_visible_file_has(mini_repo):
    _, _, fp = mini_repo
    assert sorted(fp) == sorted(audit.norm(s) for s in SECRETS)
    assert fp[audit.norm(SECRETS[0])] == ["python/mori/ops/gemm_ar/op.py"]


def test_hidden_text_is_a_leak_however_it_arrives(mini_repo, tmp_path):
    _, patterns, fp = mini_repo
    numbered = "\n".join(f"{i}\t{s}" for i, s in enumerate(HIDDEN_OP.splitlines(), 1))
    diff = "\n".join(
        [
            "diff --git a/python/mori/ops/gemm_ar/op.py b/python/mori/ops/gemm_ar/op.py",
            "+++ b/python/mori/ops/gemm_ar/op.py",
            "@@ -1,2 +1,2 @@",
            "+" + SECRETS[0],
        ]
    )
    records = [
        _use("cat", "Bash", {"command": "cat python/mori/ops/gemm_ar/op.py"}),
        _result("cat", HIDDEN_OP),
        _use(
            "script", "Bash", {"command": "python - <<EOF\nprint(open(p).read())\nEOF"}
        ),
        _result("script", "\n".join(SECRETS[:3])),
        _use("grep", "Bash", {"command": "grep -rn chunk python/"}),
        _result("grep", "python/mori/ops/gemm_ar/op.py:2:" + SECRETS[0]),
        _use("read", "Read", {"file_path": "/v/site-packages/mori/ops/gemm_ar/op.py"}),
        _result("read", numbered),
        _use("show", "Bash", {"command": "git show HEAD~3 -- python"}),
        _result("show", diff),
        {
            "type": "attachment",
            "attachment": {
                "type": "file",
                "filename": "/w/python/mori/ops/gemm_ar/op.py",
                "content": {"file": {"content": SECRETS[1]}},
            },
        },
    ]
    findings, _ = audit.audit([_write(tmp_path / "s.jsonl", records)], patterns, fp)
    assert _kinds(findings) == [
        ("leak", None),
        ("leak", "cat"),
        ("leak", "grep"),
        ("leak", "read"),
        ("leak", "script"),
        ("leak", "show"),
    ]
    by_call = {f["call_id"]: f for f in findings}
    assert by_call["cat"]["lines"] == 4 and by_call["cat"]["tool"] == "Bash"


def test_names_tracebacks_and_coincidences_are_not_leaks(mini_repo, tmp_path):
    _, patterns, fp = mini_repo
    tb = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "/v/site-packages/mori/ops/gemm_ar/op.py", line 3, in f',
            "    " + SECRETS[1],
            "RuntimeError: boom",
        ]
    )
    records = [
        _use("denied", "Read", {"file_path": "/w/python/mori/ops/gemm_ar/op.py"}),
        _result("denied", "Permission to read this file has been denied.", True),
        _use("ls", "Bash", {"command": "ls -l python/mori/ops/gemm_ar/op.py"}),
        _result("ls", "-rw-r--r-- 1 u u 300 python/mori/ops/gemm_ar/op.py"),
        _use("mod", "Bash", {"command": "python -c 'import mori.ops.gemm_ar.op'"}),
        _result("mod", ""),
        _use("base", "Bash", {"command": "torchrun op_harness.py --impl existing"}),
        _result("base", tb),
        _use("mine", "Bash", {"command": "cat mine.py"}),
        _result("mine", "\n".join(SECRETS[:2])),
        _use("facts", "Read", {"file_path": "/w/.claude/skills/f/gfx950.md"}),
        _result("facts", "1\tSource: `python/mori/ops/gemm_ar/op.py:12`"),
        _use("sib", "Read", {"file_path": "/w/python/mori/ops/gemm_ag/op.py"}),
        _result("sib", "1\t" + HIDDEN_OP.splitlines()[5]),
        _use("v2", "Bash", {"command": "cat tests/python/cco/test_gemm_ar_v2.py"}),
        _result("v2", "ok"),
    ]
    findings, _ = audit.audit([_write(tmp_path / "s.jsonl", records)], patterns, fp)
    assert _kinds(findings) == [
        ("attempt", "ls"),
        ("attempt", "mod"),
        ("blocked", "denied"),
        ("review", "base"),
        ("review", "mine"),
    ]


def test_long_results_and_subagents_are_read(mini_repo, tmp_path):
    _, patterns, fp = mini_repo
    session = tmp_path / "session"
    _write(
        session / "subagents" / "agent-1.jsonl",
        [
            _use(
                "big", "Bash", {"command": "sed -n 1,9p python/mori/ops/gemm_ar/op.py"}
            ),
            _result("big", "Output too large. Full output saved to tool-results"),
        ],
    )
    (session / "tool-results").mkdir()
    (session / "tool-results" / "big.txt").write_text(SECRETS[2] + "\n")
    findings, files = audit.audit([session], patterns, fp)
    assert _kinds(findings) == [("leak", "big")] and len(files) == 2


def test_audit_main_reports_and_keeps_the_base_before_the_session(mini_repo, tmp_path):
    repo, _, _ = mini_repo
    mine = repo / "python" / "mori" / "ops" / "gemm_ar_v2" / "op.py"
    mine.parent.mkdir(parents=True)
    mine.write_text("\n".join(SECRETS[:3]) + "\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "the session's candidate, copied")
    copy = _write(
        tmp_path / "copy.jsonl",
        [_use("c", "Bash", {"command": "cat mine.py"}), _result("c", mine.read_text())],
    )
    clean = _write(
        tmp_path / "clean.jsonl",
        [
            _use("d", "Read", {"file_path": "/w/python/mori/ops/gemm_ar/op.py"}),
            _result("d", "File is covered by a Read deny rule.", True),
        ],
    )
    common = ["--repo", str(repo), "--hidden", str(repo / "hidden_paths.txt")]
    report = tmp_path / "report.json"
    # the default base is the commit that last changed the hidden list, before the copy
    assert audit.main([str(copy), *common, "--json", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["verdict"] == "void" and data["counts"]["leak"] == 1
    assert data["hidden_files"] == 1 and data["fingerprint_lines"] == 4
    assert audit.main([str(copy), *common, "--base", "HEAD"]) == 0
    assert audit.main([str(clean), *common, "--json", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["verdict"] == "clean" and data["counts"]["blocked"] == 1
    assert audit.main([str(tmp_path / "absent.jsonl"), *common]) == 2


def test_clean_room_files_agree():
    patterns = audit.read_patterns(CASE / "hidden_paths.txt")
    deny = json.loads((CASE / "settings.implementer.json").read_text())["permissions"][
        "deny"
    ]
    reads = [r for r in deny if r.startswith("Read(")]
    assert reads == [
        f"Read(//**/{p}**)" if p.endswith("/") else f"Read(//**/{p})" for p in patterns
    ]
    assert "Bash(git push:*)" in deny
    hidden = [
        "python/mori/ops/gemm_ar/op.py",
        "/opt/venv/lib/python3.10/site-packages/mori/ops/gemm_ar/kernels_fused.py",
        "benchmark/cco/flydsl/gemm_ar/timing.py",
        "tests/python/cco/test_gemm_ar_bench.py",
        "benchmark/cco/harness/cases/gemm_ar_bf16/existing.py",
    ]
    visible = [
        "python/mori/ops/gemm_ar/_gemm_a16w16_8wave.py",
        "python/mori/ops/gemm_ar/__init__.py",
        "python/mori/ops/gemm_ar_v2/op.py",
        "tests/python/cco/test_gemm_ar_v2.py",
        "tests/python/cco/gemm_ar_v2_worker.py",
        "benchmark/cco/flydsl/gemm_ar_v2/bench.py",
        "docs/fused-gemm-ar-v2.md",
        "benchmark/cco/harness/cases/gemm_ar_bf16/op_harness.py",
    ]
    assert all(audit.hidden_pattern(p, patterns) for p in hidden)
    assert not any(audit.hidden_pattern(p, patterns) for p in visible)


def test_hidden_list_matches_files_in_the_repository():
    import subprocess

    try:
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for pattern in audit.read_patterns(CASE / "hidden_paths.txt"):
        assert any(audit.hidden_pattern(p, [pattern]) for p in tracked), pattern


def test_baselines_run_by_alias():
    assert set(harness.ALIASES) == {"existing", "existing-q8"}
    text = (CASE / "existing.py").read_text()
    for target in harness.ALIASES.values():
        path, _, cls = target.partition(":")
        assert (CASE / path).is_file() and f"class {cls}" in text
    try:
        cls = harness.load_impl("existing")
    except ImportError:  # no mori here: the alias got as far as importing the op
        return
    assert cls.__name__ == "ExistingGemmAllReduce"


def test_arms_and_contract_point_at_real_files():
    patterns = audit.read_patterns(CASE / "hidden_paths.txt")
    for name in ("arms.dev.yaml", "arms.verify.yaml"):
        arms = run.load_arms(CASE / name)
        args = arms[0]["args"]
        assert (
            arms[0]["name"] == "existing"
            and args[args.index("--impl") + 1] == "existing"
        )
        assert all((REPO / a["bench"]).is_file() for a in arms)
    bounds = run.load_arms(CASE / "arms.bounds.yaml")
    assert [a["name"] for a in bounds] == ["split-bench", "gemm-only-bench"]
    assert all((REPO / a["bench"]).is_file() for a in bounds)
    # ISA dumps come from a launch that runs no hidden code
    for a in run.load_arms(CASE / "arms.isa.yaml"):
        assert not any("existing" in x for x in a["args"]) and not audit.hidden_pattern(
            a["bench"], patterns
        )
    yaml = pytest.importorskip("yaml")
    contract = yaml.safe_load((CASE / "contract.yaml").read_text())
    assert (REPO / contract["interface"]).is_file()
    for key in ("hidden", "settings", "audit"):
        assert (REPO / contract["clean_room"][key]).is_file(), key
    assert contract["targets"]["within_existing_pct"] == 5


def _hook(command, cwd=REPO):
    return guard.decision(
        {"tool_name": "Bash", "cwd": str(cwd), "tool_input": {"command": command}},
        audit.read_patterns(CASE / "hidden_paths.txt"),
    )


_needs_git = pytest.mark.skipif(
    not (REPO / ".git").exists(),
    reason="the guard reads the hidden files' places from git",
)


@_needs_git
@pytest.mark.parametrize(
    "command",
    [
        "python3 -c \"print(open('python/mori/ops/gemm_ar/op.py').read())\"",
        "git show HEAD:benchmark/cco/flydsl/gemm_ar/timing.py | head",
        "python3 -c 'import mori.ops.gemm_ar.kernels_fused as k'",
        "python - <<'EOF'\nprint(open('docs/fused-gemm-ar.md').read())\nEOF",
        "grep -n chunk python/mori/ops/gemm_ar/layout.py",
        "grep -rn sdma_queues python/mori/ops/",
        "grep -e x -r python/mori/ops/gemm_ar",
        "timeout 30 grep -Rn chunk benchmark/cco/harness",
        "rg chunk",
        "git grep -n chunk",
        "find python -name '*.py' | xargs grep -n chunk",
        "cd python && grep -R chunk .",
    ],
)
def test_guard_denies_commands_that_reach_hidden_code(command):
    out = _hook(command)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"].startswith("Denied by the gemm_ar_bf16")


@_needs_git
@pytest.mark.parametrize(
    "command",
    [
        "torchrun --nproc_per_node=8 benchmark/cco/harness/cases/gemm_ar_bf16/"
        "op_harness.py --mode check --impl existing",
        "python benchmark/cco/harness/run.py --arms "
        "benchmark/cco/harness/cases/gemm_ar_bf16/arms.bounds.yaml --rounds 3",
        "cat tests/python/cco/test_gemm_ar_v2.py python/mori/ops/gemm_ar_v2/op.py",
        "grep -rn sdma_queues python/mori/ops/gemm_ag",
        "grep -n chunk python/mori/ops/gemm_ar/_gemm_a16w16_8wave.py",
        "rg chunk python/mori/ops/gemm_a2a",
        "git grep -n chunk -- python/mori/ops/gemm_ag",
        "cd python/mori/ops/gemm_ag && grep -rn x .",
        "find python/mori/ops/gemm_ag -name '*.py' -exec grep -n chunk {} +",
        "grep -rn chunk /nonexistent/dir",
        "git log --oneline -5",
    ],
)
def test_guard_lets_black_box_runs_and_narrow_searches_through(command):
    assert _hook(command) is None


def test_guard_finds_installed_copies_and_ignores_other_tools(tmp_path):
    site = tmp_path / "site-packages"
    (site / "mori" / "ops" / "gemm_ar").mkdir(parents=True)
    (site / "mori" / "ops" / "gemm_ar" / "op.py").write_text("x\n")
    (site / "flydsl").mkdir()
    assert _hook(f"grep -rn chunk {site}") is not None
    assert _hook(f"grep -rn chunk {site / 'flydsl'}") is None
    patterns = audit.read_patterns(CASE / "hidden_paths.txt")
    read = {
        "tool_name": "Read",
        "tool_input": {"file_path": "python/mori/ops/gemm_ar/op.py"},
    }
    assert guard.decision(read, patterns) is None


def test_guard_speaks_the_hook_protocol():
    import subprocess

    def call(command):
        payload = json.dumps(
            {"tool_name": "Bash", "cwd": str(REPO), "tool_input": {"command": command}}
        )
        return subprocess.run(
            [sys.executable, str(CASE / "guard.py")],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    out = json.loads(call("cat python/mori/ops/gemm_ar/op.py"))
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert call("ls") == ""
    settings = json.loads((CASE / "settings.implementer.json").read_text())
    (entry,) = settings["hooks"]["PreToolUse"]
    assert entry["matcher"] == "Bash"
    assert entry["hooks"][0]["command"].endswith('/gemm_ar_bf16/guard.py"')
