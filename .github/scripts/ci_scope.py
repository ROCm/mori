#!/usr/bin/env python3
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
"""Decide which GPU CI lanes a pull request needs, and say so on the PR.

Run once per CI run by .github/workflows/ci-scope.yml, which ci.yml calls first;
ci.yml hands the cco and fused lanes to ci_cco.yml. Writes these step outputs:

  mori    MoRI CI on MI355X: EP v1, shmem, IO, CCL
  mi300x  MoRI CI on MI300X: intranode, internode
  umbp    UMBP unit tests
  cco     CCO unit test, CCO EPv2 single-node test, CCO internode (ci_cco.yml)
  fused   CCO fused GEMM + collective job (ci_cco.yml)
  build   the MI355X build job (compile check)
  groups  JSON list of test groups (GROUPS); a step runs only if its group is listed
  full    full internode bench/stress sweeps: workflow_dispatch, or ci:full on a PR
  reason  why

Non-PR events (push to main, workflow_dispatch) run everything. On a PR:

  - a draft runs nothing, unless it has the ci:full label;
  - ci:full runs everything, with the full internode sweeps; ci:mi300x adds the
    MI300X lanes to the lanes the changed files pick. Adding either label re-runs
    the PR's CI (ci-rerun.yml); labels are read live, so the re-run sees it;
  - otherwise the test groups follow the changed files (rules below) and the lanes
    follow the groups (LANE_GROUPS), after the pre-commit check of the head commit
    has passed. A rename counts under both its old and its new path. A tools/ file
    counts only if something PR CI runs, builds or imports names it (ci_tools); a
    module that importing mori.ops/io/shmem runs also counts for umbp; a conftest.py,
    or a test package's __init__.py, counts for the groups of every file below it.

With COMMENT=true it keeps one comment on the PR up to date with the lanes.

`--files FILE` classifies a list of paths offline, for checking the rules.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LANES = ("mori", "mi300x", "umbp", "cco", "fused", "build")
# Test groups: the CI steps that test one part of mori. A job runs if one of its
# groups does; inside it, a step runs if its group does. "build" is the compile check.
GROUPS = (
    "ep",
    "kiwi",
    "ccl",
    "io",
    "ras",
    "shmem",
    "umbp",
    "cco",
    "epv2",
    "fused",
    "build",
)
ALL = frozenset(GROUPS)
# The groups each lane's jobs test. KIWI runs only on MI300X (build_ops_kiwi).
LANE_GROUPS = {
    "mori": {"ep", "ccl", "io", "ras", "shmem"},
    "mi300x": {"ep", "kiwi", "ccl", "io", "ras", "shmem"},
    "umbp": {"umbp"},
    "cco": {"cco", "epv2"},
    "fused": {"fused"},
}
# Who uses each shared library, from a scan of the build graph, the Python imports
# and the JIT sources behind every group's tests.
CCO_USERS = {"cco", "epv2", "fused"}
IO_USERS = {"io", "ccl", "ras", "umbp"}
SHMEM_USERS = {"shmem", "ep", "kiwi", "ccl", "cco", "epv2", "io"}
OPS_USERS = {"ep", "kiwi", "epv2", "fused"}
FUSED_OPS = r"(gemm_ar|gemm_ag|gemm_a2a|ag_gemm|a2a_gemm)"

# (area, groups, pattern): first match wins; a path that matches nothing is core and
# runs every group. Rules stay at directory level: mori.ops imports subpackages by
# name at run time.
AREAS = [
    (
        "docs",
        (),
        r"(\.md|\.rst|\.png|\.jpg|\.svg)$|^docs/|^LICENSE|^\.gitignore$|^\.claude/"
        r"|^\.github/(ISSUE_TEMPLATE|PULL_REQUEST_TEMPLATE|CODEOWNERS)|^src/umbp/doc/",
    ),
    # Lint and style configs: only the pre-commit check reads them.
    (
        "lint",
        (),
        r"^(\.clang-format|\.typos\.toml|CPPLINT\.cfg|\.pre-commit-config\.yaml)$"
        r"|^tools/codestyle/",
    ),
    ("ci_cco", CCO_USERS, r"^\.github/workflows/ci_cco\.yml$"),
    # ci.yml runs the scope and calls ci_cco.yml, so it can affect every group.
    (
        "ci",
        ALL,
        r"^\.github/workflows/(ci|ci-scope)\.yml$|^\.github/scripts/ci_scope\.py$",
    ),
    # Other workflows (nightly, docs, pre-commit, ...) do not run in PR CI.
    ("other_ci", (), r"^\.github/"),
    # Examples PR CI neither builds nor runs.
    (
        "unused_examples",
        (),
        r"^examples/(fsdp_sdma|torch_symm_all2all|monitoring|io)/"
        r"|^examples/cco/python/(03|06|09|11|12)_",
    ),
    (
        "epv2",
        {"epv2"},
        r"^tools/jit_v2/|^tests/cpp/jit/|(^|/)dispatch_combine_v2(/|$)|(^|/)jit/v2/"
        r"|^(src|include/mori)/jit/",
    ),
    # tools/ that CI runs, by the groups of the jobs running them.
    ("ep", {"ep", "kiwi", "epv2"}, r"^tools/run_internode_test\.sh$"),
    ("ccl", {"ccl"}, r"^tools/run_internode_ccl_ag\.sh$"),
    ("io", IO_USERS, r"^tools/run_internode_io_benchmark\.sh$"),
    ("cco", CCO_USERS, r"^tools/build_cco_bitcode\.sh$"),
    ("shmem", SHMEM_USERS, r"^tools/build_shmem_bitcode\.sh$"),
    # Any other tools/ file counts only if CI builds or runs it (ci_tools); then the
    # rules below classify it.
    ("tools", "tools", r"^tools/"),
    # The EPv2 tests import it.
    ("cco_tests", {"cco", "epv2"}, r"^examples/cco/python/cco_example_common\.py$"),
    (
        "fused",
        {"fused"},
        rf"^python/mori/ops/{FUSED_OPS}/|^tests/python/cco/(test_)?({FUSED_OPS}|flydsl_ar)"
        r"|^tests/python/cco/kfd_settle\.py$|^benchmark/cco/flydsl/|^python/mori/ir/flydsl/",
    ),
    (
        "cco_tests",
        {"cco"},
        r"^tests/(cpp|python)/cco/|^examples/cco|^benchmark/cco/|^tools/run_cco_tests\.sh$"
        r"|^(src|tests/python)/allocator/",
    ),
    (
        "cco",
        CCO_USERS,
        r"^(src|include/mori|python/mori)/cco/|^python/mori/ir/triton/cco/",
    ),
    ("umbp", {"umbp"}, r"(^|/|_)umbp(/|_|\.|$)"),
    (
        "kiwi",
        {"kiwi"},
        r"(^|/)(dispatch_combine_)?kiwi(/|_|\.|$)|^tests/python/ops/test_kiwi_",
    ),
    ("ops_shared", OPS_USERS | {"ccl"}, r"^python/mori/ops/_jit_loader\.py$"),
    (
        "ops_shared",
        OPS_USERS,
        r"^python/mori/ops/(__init__|utils|dispatch_combine)\.py$",
    ),
    (
        "ep",
        {"ep", "kiwi"},
        r"^(src|include/mori)/ops/|^src/pybind/pybind_(ops|xla_ffi_ops)\.cpp$"
        r"|^python/mori/(ops|jax)/|^tests/python/ops/|^tests/python/test_tuning_config_quant_types\.py$"
        r"|^examples/ops/",
    ),
    (
        "ccl",
        {"ccl"},
        r"^(src|include/mori)/collective/|^python/mori/ccl/|^src/pybind/pybind_ccl\.cpp$"
        r"|^tests/python/ccl/",
    ),
    ("ras_tests", {"ras"}, r"^tests/cpp/ras/"),
    ("io_tests", {"io"}, r"^tests/(python|cpp)/io/"),
    # RAS is built into mori_io.
    (
        "io",
        IO_USERS,
        r"^(src|include/mori)/(io|ras)/|^python/mori/io/|^src/pybind/pybind_io\.cpp$",
    ),
    ("metrics", {"umbp"}, r"^(src|include/mori)/metrics/|^tests/cpp/metrics/"),
    (
        "shmem_tests",
        {"shmem"},
        r"^tests/python/shmem/|^examples/shmem/|^benchmark/shmem/",
    ),
    # The application layer includes it.
    ("core", ALL, r"^include/mori/shmem/internal\.hpp$"),
    (
        "shmem",
        SHMEM_USERS,
        r"^(src|include/mori|python/mori)/shmem/|^src/pybind/pybind_shmem\.cpp$|^python/mori/ir/",
    ),
    # Built by the build job, run nowhere.
    (
        "build_only",
        {"build"},
        r"^examples/(sdma|collective|local_rdma_ops|dist_rdma_ops|utils|application|benchmarks)/",
    ),
    (
        "build",
        ALL,
        r"^(setup\.py|pyproject\.toml|CMakeLists\.txt|requirements[^/]*\.txt)$|^cmake/|^docker/|^3rdparty/",
    ),
]

# On a PR the MI300X lanes run only when one of these changed (or with ci:mi300x):
# EP v1, shmem, KIWI, mlx5, IO, the shared RDMA/SDMA/bootstrap/topology code, CCL,
# NIC detection, and what builds or runs them there.
MI300X_PATHS = re.compile(
    r"^(src|include/mori|python/mori)/ops/(?!dispatch_combine_v2/|gemm_|ag_gemm/|a2a_gemm/|__init__\.py$)"
    r"|^tests/python/ops/(?!dispatch_combine_v2/)|^examples/ops/"
    r"|(^|/)(shmem|kiwi)(/|_|\.|$)|mlx5|(^|/|_)io(/|_|\.|$)|(^|/)ras(/|_|\.|$)"
    r"|^(src|include/mori)/application/(transport/(rdma|sdma)|bootstrap|topology|context|memory)/"
    r"|^include/mori/core/|^(src|include/mori)/collective/|^python/mori/(jit/(?!v2/)|ccl/)"
    r"|^tests/python/ccl/|^tools/run_internode_ccl_ag\.sh$|^src/pybind/pybind_(ops|shmem|ccl)\.cpp$"
    r"|^(setup\.py|pyproject\.toml|CMakeLists\.txt)$|^cmake/|^docker/|^3rdparty/"
    r"|^\.github/workflows/ci\.yml$|^\.github/workflows/ci-scope\.yml$"
    r"|^\.github/scripts/ci_scope\.py$|^tools/run_internode_test\.sh$"
)


# What PR CI runs, builds or imports: a tools/ file named in one of these, in a
# docker/ file the workflows use, or in a tools/ file that counts, counts.
CI_WORKFLOWS = (".github/workflows/ci.yml", ".github/workflows/ci_cco.yml")
CI_ENTRY_FILES = (
    *CI_WORKFLOWS,
    "setup.py",
    "pyproject.toml",
    "**/CMakeLists.txt",
    "cmake/**/*",
    "python/mori/**/*.py",
)
# test_umbp_lazy_load imports these and checks that UMBP stays unloaded, so every
# module those imports run counts for umbp too (umbp_lazy_files).
UMBP_LAZY_IMPORTS = ("mori.ops", "mori.io", "mori.shmem")
# pytest runs a conftest.py, and the __init__.py of a test package, for every test
# below it, so they count for the groups of every file there.
TEST_SETUP = re.compile(r"(^|/)conftest\.py$|^tests/(.+/)?__init__\.py$")
ROOT = Path(__file__).resolve().parents[2]
_ci_tools = None
_umbp_lazy = None
_tracked = None


def ci_tools():
    """tools/ files that CI uses; None (every one counts) without a checkout."""
    global _ci_tools
    if _ci_tools is None:
        if not (ROOT / ".github/workflows/ci.yml").is_file():
            return None
        texts = [
            f.read_text(errors="ignore")
            for pattern in CI_ENTRY_FILES
            for f in ROOT.glob(pattern)
            if f.is_file() and "3rdparty" not in f.parts
        ]
        workflows = "\n".join(
            (ROOT / w).read_text(errors="ignore") for w in CI_WORKFLOWS
        )
        texts += [
            f.read_text(errors="ignore")
            for f in (ROOT / "docker").rglob("*")
            if f.is_file() and str(f.relative_to(ROOT)) in workflows
        ]
        tools = {
            str(f.relative_to(ROOT)) for f in (ROOT / "tools").rglob("*") if f.is_file()
        }
        used, todo = set(), "\n".join(texts)
        while todo:
            new = {
                t
                for t in tools - used
                if t in todo
                or re.search(rf"(?<![\w.-]){re.escape(Path(t).name)}(?![\w-])", todo)
            }
            used |= new
            todo = "\n".join((ROOT / t).read_text(errors="ignore") for t in new)
        _ci_tools = used
    return _ci_tools


def import_time_imports(f, pkg):
    """Modules the file f of package pkg imports when it is imported."""
    try:
        nodes, mods = list(ast.parse(f.read_text()).body), []
    except (SyntaxError, ValueError):
        return []
    while nodes:
        n = nodes.pop()
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = [n.module] if n.module else []
            if n.level:
                parts = pkg.split(".")
                base = parts[: len(parts) + 1 - n.level] + base
            mods += [".".join(base)] + [".".join(base + [a.name]) for a in n.names]
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes += ast.iter_child_nodes(n)
    return mods


def umbp_lazy_files():
    """Files that importing UMBP_LAZY_IMPORTS runs; None without a checkout."""
    global _umbp_lazy
    if _umbp_lazy is None:
        py = ROOT / "python"
        if not (py / "mori/__init__.py").is_file():
            return None
        found, todo = {}, list(UMBP_LAZY_IMPORTS)
        while todo:
            parts = todo.pop().split(".")
            for i in range(1, len(parts) + 1):  # parent packages run first
                mod, base = ".".join(parts[:i]), py.joinpath(*parts[:i])
                f = base / "__init__.py"
                f = f if f.is_file() else base.with_suffix(".py")
                if mod in found or not f.is_file():
                    continue
                found[mod] = f
                pkg = mod if f.name == "__init__.py" else mod.rpartition(".")[0]
                todo += import_time_imports(f, pkg)
        _umbp_lazy = {str(f.relative_to(ROOT)) for f in found.values()}
    return _umbp_lazy


def tracked_files():
    """Every tracked path (a sparse checkout lists them too); None without git."""
    global _tracked
    if _tracked is None:
        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "-z"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        _tracked = [f for f in out.split("\0") if f]
    return _tracked


def area_of(path):
    """(area, groups) of one path."""
    if TEST_SETUP.search(path):
        tracked = tracked_files()
        if tracked is None:
            return "core", set(ALL)
        below = os.path.dirname(path) + "/" if "/" in path else ""
        files = [f for f in tracked if f.startswith(below) and not TEST_SETUP.search(f)]
        return "test_setup", set().union(*(area_of(f)[1] for f in files))
    for area, groups, pattern in AREAS:
        if not re.search(pattern, path):
            continue
        if groups == "tools":
            used = ci_tools()
            if used is None or path in used:
                continue
            return "unused_tools", set()
        lazy = umbp_lazy_files()
        imported = path.startswith("python/mori/") if lazy is None else path in lazy
        return area, set(groups) | ({"umbp"} if imported else set())
    return "core", set(ALL)


def lanes_for_files(files, labels):
    """(lanes, groups, reason) for a PR's files and labels."""
    found = [area_of(f) for f in files]
    groups = set().union(*(g for _, g in found)) if found else set()
    lanes = {lane: bool(groups & want) for lane, want in LANE_GROUPS.items()}
    lanes["build"] = bool(groups - {"fused"})
    mi300x_hit = any(MI300X_PATHS.search(f) for f in files)
    lanes["mi300x"] = "ci:mi300x" in labels or (lanes["mi300x"] and mi300x_hit)
    if lanes["mi300x"] and not groups & LANE_GROUPS["mi300x"]:
        # ci:mi300x alone: the MI300X jobs run all their groups.
        groups |= LANE_GROUPS["mi300x"]
    areas = sorted({a for a, g in found if g})
    reason = "changed areas: " + (", ".join(areas) or "none that need GPU CI")
    return lanes, groups, reason


def gh_lines(*args):
    out = subprocess.run(
        ["gh", "api", *args], check=True, capture_output=True, text=True
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def changed_files(repo, pr):
    return gh_lines(
        "--paginate",
        f"repos/{repo}/pulls/{pr}/files",
        "--jq",
        ".[] | .filename, (.previous_filename // empty)",
    )


def wait_for_precommit(repo, sha, appear_s=300, finish_s=1200):
    """Conclusion of the head commit's pre-commit check; None if it never starts."""
    start = time.time()
    while True:
        runs = [
            json.loads(x)
            for x in gh_lines(
                f"repos/{repo}/commits/{sha}/check-runs?check_name=pre-commit",
                "--jq",
                ".check_runs[] | {status, conclusion}",
            )
        ]
        done = [r["conclusion"] for r in runs if r["status"] == "completed"]
        if done:
            return "success" if "success" in done else done[0]
        waited = time.time() - start
        if (not runs and waited > appear_s) or waited > finish_s:
            return None if not runs else "unfinished"
        time.sleep(20)


def live_labels(env):
    """The PR's labels now; a re-run replays the original event's payload."""
    try:
        return set(
            gh_lines(
                f"repos/{env['REPO']}/issues/{env['PR_NUMBER']}/labels",
                "--jq",
                ".[].name",
            )
        )
    except subprocess.CalledProcessError:
        return set(json.loads(env.get("PR_LABELS") or "[]"))


def decide(env):
    """(lanes, groups, reason, full)."""
    event = env.get("EVENT_NAME", "")
    if event != "pull_request":
        full = event == "workflow_dispatch"
        return (
            dict.fromkeys(LANES, True),
            set(ALL),
            f"{event} event runs everything",
            full,
        )
    labels = live_labels(env)
    full = "ci:full" in labels
    if env.get("PR_DRAFT") == "true" and not full:
        return (
            dict.fromkeys(LANES, False),
            set(),
            "draft PR (add ci:full or mark it ready to run CI)",
            full,
        )
    precommit = wait_for_precommit(env["REPO"], env["PR_HEAD_SHA"])
    if precommit == "unfinished":
        raise SystemExit(
            "pre-commit has not finished after 20 min; /ci-rerun once it has"
        )
    if precommit not in (None, "success"):
        raise SystemExit(f"pre-commit is '{precommit}'; fix it to run GPU CI")
    if full:
        return dict.fromkeys(LANES, True), set(ALL), "ci:full label", full
    lanes, groups, reason = lanes_for_files(
        changed_files(env["REPO"], env["PR_NUMBER"]), labels
    )
    return lanes, groups, reason, full


MARKER = "<!-- ci-scope -->"
# Each lane's jobs and the groups that start them (their if: in ci.yml, ci_cco.yml).
JOBS = {
    "mori": (
        ("MI355X intranode", LANE_GROUPS["mori"]),
        ("MI355X ROCm 10", {"ep"}),
        ("MI355X JAX", {"ep"}),
        ("MI355X internode", {"ep", "ccl", "io"}),
    ),
    "mi300x": (
        ("MI300X intranode", LANE_GROUPS["mi300x"]),
        ("MI300X internode", {"ep", "kiwi", "ccl", "io"}),
    ),
    "umbp": (("UMBP unit test", {"umbp"}),),
    "cco": (
        ("CCO unit test", {"cco", "epv2"}),
        ("CCO EPv2 single-node test", {"epv2"}),
        ("CCO internode", {"epv2"}),
    ),
    "fused": (("CCO fused GEMM test", {"fused"}),),
    "build": (("MoRI build (compile check)", ALL),),
}
HINT = (
    "Lanes follow the changed files. Adding `ci:full` re-runs CI with every lane and "
    "the full internode sweeps; adding `ci:mi300x` re-runs it with the MI300X lanes "
    "added. `/ci-rerun` re-runs CI against the current main."
)


def comment_body(env, lanes, reason, full, groups=frozenset()):
    """The PR comment; lanes=None means GPU CI did not start."""
    run = "{}/{}/actions/runs/{}".format(
        env.get("GITHUB_SERVER_URL", "https://github.com"),
        env.get("REPO", ""),
        env.get("GITHUB_RUN_ID", ""),
    )
    head = (
        f"{MARKER}\n**GPU CI for `{env.get('PR_HEAD_SHA', '')[:8]}`** ([run]({run})): "
    )
    if lanes is None:
        return head + f"not started, {reason}."
    if not any(lanes.values()):
        return head + f"none, {reason}.\n\n{HINT}"
    if full:
        reason += "; full internode sweeps"
    rows = []
    for lane in LANES:
        # A lane that runs lists the jobs its groups start; one that does not, all.
        jobs = [j for j, want in JOBS[lane] if not lanes[lane] or groups & want]
        rows.append(
            f"| `{lane}` | {'yes' if lanes[lane] else 'no'} | {', '.join(jobs)} |"
        )
    tests = ", ".join(g for g in GROUPS if g in groups) or "none"
    return "\n".join(
        [
            head + reason,
            "",
            "| lane | runs | jobs |",
            "|---|---|---|",
            *rows,
            "",
            f"Test groups (the steps that run in those jobs): {tests}.",
            "",
            HINT,
        ]
    )


def post_comment(env, body):
    """Create or update the PR's scope comment. A failure only warns: a fork PR's
    token cannot write."""
    repo, pr = env["REPO"], env["PR_NUMBER"]
    try:
        ids = gh_lines(
            "--paginate",
            f"repos/{repo}/issues/{pr}/comments",
            "--jq",
            '.[] | select(.user.login == "github-actions[bot]") '
            f'| select(.body | startswith("{MARKER}")) | .id',
        )
        if ids:
            gh_lines(
                "-X",
                "PATCH",
                f"repos/{repo}/issues/comments/{ids[0]}",
                "-f",
                f"body={body}",
            )
        else:
            gh_lines(
                "-X", "POST", f"repos/{repo}/issues/{pr}/comments", "-f", f"body={body}"
            )
    except subprocess.CalledProcessError as e:
        print(
            f"::warning::could not update the CI scope comment: {(e.stderr or '').strip()[:300]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files", help="classify the paths in this file (one per line) and exit"
    )
    ap.add_argument(
        "--labels", default="", help="comma-separated PR labels for --files"
    )
    args = ap.parse_args()
    if args.files:
        files = [line.strip() for line in open(args.files) if line.strip()]
        lanes, groups, reason = lanes_for_files(
            files, set(filter(None, args.labels.split(",")))
        )
        print(
            json.dumps(
                {
                    **lanes,
                    "groups": [g for g in GROUPS if g in groups],
                    "reason": reason,
                }
            )
        )
        return

    env = os.environ
    want = (
        env.get("COMMENT") == "true"
        and env.get("EVENT_NAME") == "pull_request"
        and bool(env.get("PR_NUMBER"))
    )
    try:
        lanes, groups, reason, full = decide(env)
    except SystemExit as e:
        if want:
            post_comment(env, comment_body(env, None, str(e.code), False))
        print(f"::error::{e.code}")
        return 1
    names = [g for g in GROUPS if g in groups]
    with open(env["GITHUB_OUTPUT"], "a") as out:
        for lane in LANES:
            out.write(f"{lane}={str(lanes[lane]).lower()}\n")
        out.write(f"groups={json.dumps(names)}\n")
        out.write(f"full={str(full).lower()}\n")
        out.write(f"reason={reason}\n")
    summary = [
        "## CI scope",
        "",
        f"**Reason:** {reason}",
        "",
        "| lane | runs |",
        "|---|---|",
    ]
    summary += [f"| {lane} | {'yes' if lanes[lane] else 'no'} |" for lane in LANES]
    summary += [
        "",
        "**Test groups:** " + (", ".join(names) or "none"),
    ]
    with open(env.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as out:
        out.write("\n".join(summary) + "\n")
    if want:
        post_comment(env, comment_body(env, lanes, reason, full, groups))
    print(reason, lanes, names)


if __name__ == "__main__":
    sys.exit(main())
