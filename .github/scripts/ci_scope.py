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
"""Decide which GPU CI lanes a pull request needs.

Run by .github/workflows/ci-scope.yml. Writes these step outputs:

  mori    MoRI CI on MI355X: EP v1, shmem, IO, CCL (ci.yml)
  mi300x  MoRI CI on MI300X: build, intranode, internode (ci.yml)
  umbp    UMBP unit tests (ci.yml)
  cco     CCO unit test (cco and EPv2) and CCO internode (ci_cco.yml)
  fused   CCO fused GEMM + collective job (ci_cco.yml)
  reason  why

Non-PR events (push to main, workflow_dispatch) run everything. On a PR:

  - a draft runs nothing, unless it has the ci:full label;
  - ci:full runs everything; ci:mi300x adds the MI300X lanes to the lanes the
    changed files pick; labels are read live, so a re-run sees the current ones;
  - adding any other label runs nothing (a re-run of that run is a normal run);
  - otherwise the lanes follow the changed files (rules below), after the
    pre-commit check of the head commit has passed. A rename counts under both
    its old and its new path.

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
    ("ci_mori", r"^\.github/workflows/ci\.yml$"),
    ("ci_cco", r"^\.github/workflows/ci_cco\.yml$"),
    (
        "ci_shared",
        r"^\.github/workflows/ci-scope\.yml$|^\.github/scripts/ci_scope\.py$",
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
    "mori": {"core", "io", "build", "ci_mori", "ci_shared"},
    "umbp": {"core", "io", "build", "umbp", "ci_mori", "ci_shared"},
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
            return None if not runs else "timed_out"
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
    event, action = env.get("EVENT_NAME", ""), env.get("EVENT_ACTION", "")
    if event != "pull_request":
        return dict.fromkeys(LANES, True), f"{event} event runs everything"
    labels = live_labels(env)
    label = env.get("LABEL_NAME", "")
    first_attempt = env.get("GITHUB_RUN_ATTEMPT", "1") == "1"
    # Labels whose first run replaces the PR's run; ci:mi300x only for a caller
    # that has MI300X lanes.
    run_labels = (
        ("ci:full", "ci:mi300x")
        if env.get("HAS_MI300X", "true") == "true"
        else ("ci:full",)
    )
    if action == "labeled" and first_attempt and label not in run_labels:
        return dict.fromkeys(LANES, False), f"label '{label}' needs no CI"
    if env.get("PR_DRAFT") == "true" and "ci:full" not in labels:
        return (
            dict.fromkeys(LANES, False),
            "draft PR (add ci:full or mark it ready to run CI)",
        )
    precommit = wait_for_precommit(env["REPO"], env["PR_HEAD_SHA"])
    if precommit not in (None, "success"):
        raise SystemExit(
            f"::error::pre-commit is '{precommit}' on {env['PR_HEAD_SHA']}; fix it to run GPU CI"
        )
    if "ci:full" in labels:
        return dict.fromkeys(LANES, True), "ci:full label"
    return lanes_for_files(changed_files(env["REPO"], env["PR_NUMBER"]), labels)


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

    lanes, reason = decide(os.environ)
    # Full internode bench/stress sweeps: workflow_dispatch, or ci:full on a PR.
    event = os.environ.get("EVENT_NAME", "")
    full = event == "workflow_dispatch" or (
        event == "pull_request" and "ci:full" in live_labels(os.environ)
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
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
    with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as out:
        out.write("\n".join(summary) + "\n")
    print(reason, lanes)


if __name__ == "__main__":
    sys.exit(main())
