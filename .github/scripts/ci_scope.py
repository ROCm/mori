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
  full    full internode bench/stress sweeps: workflow_dispatch, or ci:full on a PR
  reason  why

Non-PR events (push to main, workflow_dispatch) run everything. On a PR:

  - a draft runs nothing, unless it has the ci:full label;
  - ci:full runs everything, with the full internode sweeps; ci:mi300x adds the
    MI300X lanes to the lanes the changed files pick. Adding either label re-runs
    the PR's CI (ci-rerun.yml); labels are read live, so the re-run sees it;
  - otherwise the lanes follow the changed files (rules below), after the
    pre-commit check of the head commit has passed. A rename counts under both
    its old and its new path.

With COMMENT=true it keeps one comment on the PR up to date with the lanes.

`--files FILE` classifies a list of paths offline, for checking the rules.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

LANES = ("mori", "mi300x", "umbp", "cco", "fused")

# First match wins; a path that matches nothing is "core" and runs every lane
# except mi300x, which also needs MI300X_PATHS.
AREAS = [
    ("ignored", r"(\.md|\.rst|\.png|\.jpg|\.svg)$|^docs/|^LICENSE|^\.gitignore$"),
    (
        "ignored",
        r"^\.github/(ISSUE_TEMPLATE|PULL_REQUEST_TEMPLATE|CODEOWNERS)|^\.claude/",
    ),
    ("ci_cco", r"^\.github/workflows/ci_cco\.yml$"),
    # ci.yml runs the scope and calls ci_cco.yml, so it can affect every lane.
    (
        "ci_shared",
        r"^\.github/workflows/(ci|ci-scope)\.yml$|^\.github/scripts/ci_scope\.py$",
    ),
    # Other workflows (nightly, docs, pre-commit, ...) do not run in PR CI.
    ("ignored", r"^\.github/"),
    ("umbp", r"(^|/|_)umbp(/|_|\.|$)|^examples/monitoring/"),
    ("epv2", r"(^|/)dispatch_combine_v2(/|$)|(^|/)jit/v2/"),
    (
        "fused",
        r"^python/mori/ops/(gemm_ar|gemm_ag|gemm_a2a|ag_gemm|a2a_gemm)/"
        r"|^tests/python/cco/(test_)?(gemm_ar|gemm_ag|gemm_a2a|ag_gemm|a2a_gemm|flydsl_ar)"
        r"|^tests/python/cco/kfd_settle\.py$"
        r"|^benchmark/cco/flydsl/(gemm_ar|gemm_ag|gemm_a2a|ag_gemm|a2a_gemm|ar)/"
        r"|^examples/cco/python/11_flydsl_mega_moe_sdma/",
    ),
    ("cco", r"(^|/|_)cco(/|_|\.|$)"),
    ("io", r"(^|/|_)io(/|_|\.|$)"),
    (
        "build",
        r"^(setup\.py|pyproject\.toml|CMakeLists\.txt|requirements[^/]*\.txt)$|^cmake/|^docker/|^3rdparty/",
    ),
]

# On a PR the MI300X lanes run only when one of these changed (or with ci:mi300x):
# EP v1, shmem, KIWI, mlx5, IO, the shared RDMA/SDMA/bootstrap/topology code, CCL,
# NIC detection, and what builds or runs them there.
MI300X_PATHS = re.compile(
    r"^(src|include/mori|python/mori)/ops/(?!dispatch_combine_v2/|gemm_|ag_gemm/|a2a_gemm/|__init__\.py$)"
    r"|^tests/python/ops/(?!dispatch_combine_v2/)|^examples/ops/"
    r"|(^|/)(shmem|kiwi)(/|_|\.|$)|mlx5|(^|/|_)io(/|_|\.|$)"
    r"|^(src|include/mori)/application/(transport/(rdma|sdma)|bootstrap|topology|context|memory)/"
    r"|^include/mori/core/|^(src|include/mori)/collective/|^python/mori/(jit/(?!v2/)|ccl/)"
    r"|^tests/python/ccl/|^tools/run_internode_ccl_ag\.sh$|^src/pybind/pybind_(ops|shmem|ccl)\.cpp$"
    r"|^(setup\.py|pyproject\.toml|CMakeLists\.txt)$|^cmake/|^docker/|^3rdparty/"
    r"|^\.github/workflows/ci\.yml$|^\.github/workflows/ci-scope\.yml$"
    r"|^\.github/scripts/ci_scope\.py$|^tools/run_internode_test\.sh$"
)

# Which areas switch on each lane; "core" (anything unmatched) switches on all.
LANE_AREAS = {
    "mori": {"core", "io", "build", "ci_shared"},
    "umbp": {"core", "io", "build", "umbp", "ci_shared"},
    "cco": {"core", "build", "cco", "epv2", "ci_cco", "ci_shared"},
    "fused": {"core", "build", "cco", "fused", "ci_cco", "ci_shared"},
}


def area_of(path):
    for name, pattern in AREAS:
        if re.search(pattern, path):
            return name
    return "core"


def lanes_for_files(files, labels):
    areas = {area_of(f) for f in files} - {"ignored"}
    lanes = {lane: bool(areas & want) for lane, want in LANE_AREAS.items()}
    mi300x_hit = any(MI300X_PATHS.search(f) for f in files)
    lanes["mi300x"] = "ci:mi300x" in labels or (lanes["mori"] and mi300x_hit)
    reason = "changed areas: " + (", ".join(sorted(areas)) or "none that need GPU CI")
    return lanes, reason


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
    """(lanes, reason, full)."""
    event = env.get("EVENT_NAME", "")
    if event != "pull_request":
        full = event == "workflow_dispatch"
        return dict.fromkeys(LANES, True), f"{event} event runs everything", full
    labels = live_labels(env)
    full = "ci:full" in labels
    if env.get("PR_DRAFT") == "true" and not full:
        return (
            dict.fromkeys(LANES, False),
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
        return dict.fromkeys(LANES, True), "ci:full label", full
    lanes, reason = lanes_for_files(
        changed_files(env["REPO"], env["PR_NUMBER"]), labels
    )
    return lanes, reason, full


MARKER = "<!-- ci-scope -->"
JOBS = {
    "mori": "MoRI CI on MI355X: build, intranode, ROCm 10, JAX, internode",
    "mi300x": "MoRI CI on MI300X: intranode, internode",
    "umbp": "UMBP unit test",
    "cco": "CCO unit test, CCO EPv2 single-node test, CCO internode",
    "fused": "CCO fused GEMM test",
}
HINT = (
    "Lanes follow the changed files. Adding `ci:full` re-runs CI with every lane and "
    "the full internode sweeps; adding `ci:mi300x` re-runs it with the MI300X lanes "
    "added. `/ci-rerun` re-runs CI against the current main."
)


def comment_body(env, lanes, reason, full):
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
    rows = [
        f"| `{lane}` | {'yes' if lanes[lane] else 'no'} | {JOBS[lane]} |"
        for lane in LANES
    ]
    return "\n".join(
        [head + reason, "", "| lane | runs | jobs |", "|---|---|---|", *rows, "", HINT]
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
        lanes, reason = lanes_for_files(
            files, set(filter(None, args.labels.split(",")))
        )
        print(json.dumps({**lanes, "reason": reason}))
        return

    env = os.environ
    want = (
        env.get("COMMENT") == "true"
        and env.get("EVENT_NAME") == "pull_request"
        and bool(env.get("PR_NUMBER"))
    )
    try:
        lanes, reason, full = decide(env)
    except SystemExit as e:
        if want:
            post_comment(env, comment_body(env, None, str(e.code), False))
        print(f"::error::{e.code}")
        return 1
    with open(env["GITHUB_OUTPUT"], "a") as out:
        for lane in LANES:
            out.write(f"{lane}={str(lanes[lane]).lower()}\n")
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
    with open(env.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as out:
        out.write("\n".join(summary) + "\n")
    if want:
        post_comment(env, comment_body(env, lanes, reason, full))
    print(reason, lanes)


if __name__ == "__main__":
    sys.exit(main())
