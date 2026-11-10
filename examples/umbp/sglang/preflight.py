# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check the built runtime, optionally model files and MI455 host visibility."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_image import ROOT, WORKSPACE, git, require_hash, sha256, verify_reference


def kv_check():
    import mori.umbp as umbp
    import torch

    lock = json.loads((ROOT / "runtime-lock.json").read_text())
    if torch.__version__ != lock["torch"] or torch.version.hip != lock["hip"]:
        raise RuntimeError("PyTorch/HIP differs from runtime-lock.json")
    config = umbp.UMBPConfig()
    config.dram.capacity_bytes = 4 << 20
    config.dram.use_hugepages = False
    config.ssd.enabled = False
    client = umbp.UMBPClient(config)
    payload = b"UMBP CPU preflight\x00\xff"
    source = ctypes.create_string_buffer(payload, len(payload))
    target = ctypes.create_string_buffer(len(payload))
    if not client.put_from_ptr("preflight", ctypes.addressof(source), len(payload)):
        raise RuntimeError("UMBP put failed")
    if not client.get_into_ptr("preflight", ctypes.addressof(target), len(payload)):
        raise RuntimeError("UMBP get failed")
    if target.raw != payload:
        raise RuntimeError("UMBP byte comparison failed")
    del client
    print("PASS: CPU-only UMBP 4 MiB DRAM put/get and byte comparison")


def model_check(model):
    from transformers import AutoTokenizer

    model = model.resolve(strict=True)
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "deepseek_v4":
        raise RuntimeError("This TP4 recipe requires DeepSeek V4 Pro (deepseek_v4)")
    # V4 Pro and Flash share model_type; reject a known Flash model explicitly.
    if "flash" in str(config.get("_name_or_path", "")).lower():
        raise RuntimeError("DeepSeek V4 Flash is not the model for this recipe")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        weights = set(json.loads(index.read_text())["weight_map"].values())
    elif (model / "model.safetensors").is_file():
        weights = {"model.safetensors"}
    else:
        raise RuntimeError("Missing model.safetensors or model.safetensors.index.json")
    if not weights:
        raise RuntimeError("Empty model weight index")
    for name in weights:
        path = (model / name).resolve(strict=True)
        if not path.is_relative_to(model) or path.stat().st_size == 0:
            raise RuntimeError(f"Invalid model shard: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=True
    )
    if not tokenizer.encode("UMBP demo", add_special_tokens=False):
        raise RuntimeError("Tokenizer returned no tokens")
    return {
        "path": str(model),
        "config_sha256": sha256(model / "config.json"),
        "weight_shards": len(weights),
        "weight_bytes": sum((model / name).stat().st_size for name in weights),
        "note": "Presence/tokenizer check; not a weight checksum or GPU load test",
    }


def hardware_check():
    for path in ("/dev/kfd", "/dev/dri", "/proc/1/status"):
        if not Path(path).exists():
            raise RuntimeError(f"Missing runtime visibility: {path}")
    if not shutil.which("amd-smi"):
        raise RuntimeError("amd-smi is not on PATH")
    from recipe.run import gpu_pids, kernel_log

    devices = sorted(
        path
        for path in Path("/sys/bus/pci/devices").glob("*/mem_info_vram_used")
        if (path.parent / "vendor").read_text().strip() == "0x1002"
    )
    if len(devices) != 4:
        raise RuntimeError("Expected exactly four AMD GPU memory counters")
    kernel_log()
    return {
        "gpu_processes": sorted(gpu_pids()),
        "vram_bytes": [int(path.read_text()) for path in devices],
        "note": "Visibility/schema check only; supervisor runs the idle gate and probes",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--kv-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.kv_only:
        kv_check()
        return
    lock = json.loads((ROOT / "runtime-lock.json").read_text())
    for name, expected in lock["git"].items():
        if git(WORKSPACE / name, "rev-parse", "HEAD") != expected:
            raise RuntimeError(f"Unexpected {name} revision")
    require_hash(Path(lock["standalone"]["path"]), lock["standalone"]["sha256"])
    source = ROOT / "integrated"
    require_hash(source / lock["linker"]["path"], lock["linker"]["sha256"])
    verify_reference()
    require_hash(
        source / lock["model_patch"]["path"], lock["model_patch"]["after_sha256"]
    )
    provenance = json.loads((ROOT / "build-provenance.json").read_text())
    manifest_path = ROOT / "source-manifest.json"
    require_hash(manifest_path, provenance["source_manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())["integrated"]
    actual_names = {
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if set(manifest) != actual_names or not manifest:
        raise RuntimeError("Source snapshot file set changed")
    for name, expected in manifest.items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source):
            raise RuntimeError(f"Source path escapes snapshot: {name}")
        require_hash(path, expected)
    recipe = json.loads((ROOT / "recipe/source.json").read_text())
    for name, expected in recipe["sha256"].items():
        require_hash(ROOT / "recipe" / name, expected)
    child_env = {
        key: value for key, value in os.environ.items() if not key.startswith("UMBP_")
    }
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--kv-only"],
        env=child_env,
        check=True,
        timeout=45,
    )
    result = {"ok": True, "source_files": len(manifest), "runtime_lock": lock}
    if args.model:
        result["model"] = model_check(args.model)
    if args.hardware:
        result["hardware"] = hardware_check()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
