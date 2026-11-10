# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Verify the base, apply the recovered model patch and record a new snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path("/sgl-workspace")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch: {path}: {actual}, expected {expected}")


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, timeout=30
    ).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_reference(check_base=False):
    reference = json.loads((ROOT / "reference/source.json").read_text())
    for name, metadata in reference["files"].items():
        require_hash(ROOT / "reference" / name, metadata["sha256"])
        if check_base:
            require_hash(
                WORKSPACE / "sglang/python" / metadata["original_path"],
                metadata["sha256"],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "runtime-lock.json").read_text())
    status = {}
    for name, expected in lock["git"].items():
        path = WORKSPACE / name
        actual = git(path, "rev-parse", "HEAD")
        if actual != expected:
            raise RuntimeError(f"{name} revision {actual}, expected {expected}")
        # The pinned base contains vendor changes. Preserve and record them.
        status[name] = git(path, "status", "--porcelain", "--untracked-files=no")
    require_hash(Path(lock["standalone"]["path"]), lock["standalone"]["sha256"])
    source = WORKSPACE / "sglang/python"
    require_hash(source / lock["linker"]["path"], lock["linker"]["sha256"])
    verify_reference(check_base=True)
    patch = lock["model_patch"]
    model = source / patch["path"]
    require_hash(model, patch["before_sha256"])
    before = (
        "                self.input_layernorm.variance_epsilon,\n"
        "                fn_transpose=False,\n"
    )
    after = before.replace("fn_transpose=False", "fn_transpose=True")
    content = model.read_text()
    if content.count(before) != 1:
        raise RuntimeError("Expected exactly one mHC input-normalization patch site")
    model.write_text(content.replace(before, after))
    require_hash(model, patch["after_sha256"])

    integrated = ROOT / "integrated"
    shutil.copytree(
        source / "sglang",
        integrated / "sglang",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    hashes = {
        str(path.relative_to(integrated)): sha256(path)
        for path in sorted(integrated.rglob("*"))
        if path.is_file()
    }
    save(
        ROOT / "source-manifest.json",
        {"kind": "new-demo-snapshot", "integrated": hashes},
    )
    recipe = json.loads((ROOT / "recipe/source.json").read_text())
    for name, expected in recipe["sha256"].items():
        require_hash(ROOT / "recipe" / name, expected)
    save(
        ROOT / "build-provenance.json",
        {
            "kind": "new-demo-snapshot",
            "base_image_requested": args.base_image,
            "runtime_lock": lock,
            "base_git_status": status,
            "recipe": recipe,
            "reference": json.loads((ROOT / "reference/source.json").read_text()),
            "model_patch": patch,
            "source_manifest_sha256": sha256(ROOT / "source-manifest.json"),
            "historical_integrated_manifest_recovered": False,
        },
    )
    print(f"Prepared {len(hashes)} source files; {patch['path']} patched and verified")


if __name__ == "__main__":
    main()
