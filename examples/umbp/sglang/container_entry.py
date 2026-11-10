# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run the pinned supervisor using the source baked into this demo image."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("check", "check-hardware", "preview", "serve", "smoke", "fill-replay"),
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--idle-seconds", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    model = args.model or Path("/model/DeepSeek-V4-Pro")
    check = [sys.executable, str(ROOT / "preflight.py")]
    if args.action in ("check", "check-hardware"):
        if args.model:
            check += ["--model", str(model)]
        if args.action == "check-hardware":
            check += ["--hardware"]
        os.execv(sys.executable, check)
    if args.action != "preview":
        if not args.out:
            parser.error("--out is required for a model run")
        if args.out.exists():
            parser.error("--out must be a new directory")
        subprocess.run(check + ["--model", str(model), "--hardware"], check=True)
    command = [
        sys.executable,
        str(ROOT / "recipe/run.py"),
        "umbp",
        "--source",
        str(ROOT / "integrated"),
        "--manifest",
        str(ROOT / "source-manifest.json"),
        "--out",
        str(args.out or Path("/runs/preview")),
        "--execution",
        "kernel_sync",
        "--idle-seconds",
        str(args.idle_seconds),
        "--timeout",
        str(args.timeout),
    ]
    if args.action == "preview":
        command += ["--dry-run"]
    elif args.action in ("smoke", "fill-replay"):
        script = (
            "sglang_linker_smoke.py" if args.action == "smoke" else "fill_replay.py"
        )
        command += [
            "--client",
            sys.executable,
            str(ROOT / script),
            "--out",
            str(args.out / "client"),
        ]
        if args.action == "fill-replay":
            command += ["--model", str(model)]
    os.environ["MODEL_PATH"] = str(model)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
