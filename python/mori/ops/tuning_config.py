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
"""EP Dispatch/Combine tuning configuration management.

Tuning configs are stored as two separate JSON files per (gpu_arch,
gpu_model, kernel_type, ep_size) combination:

  {arch}_{model}_{kernel}_ep{n}_dispatch.json
  {arch}_{model}_{kernel}_ep{n}_combine.json

Dispatch files contain rules keyed by (dtype, num_tokens, hidden_dim).
Combine files contain rules keyed by (dtype, num_tokens, hidden_dim,
zero_copy, quant_type).  This separation allows dispatch rules to be
shared across different quant_type / zero_copy configurations.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified dtype registry
# ---------------------------------------------------------------------------

_DTYPE_REGISTRY: list[tuple] = [
    (torch.bfloat16, "bf16", "bf16", "bf16"),
]

try:
    _DTYPE_REGISTRY.append(
        (torch.float8_e4m3fnuz, "fp8_e4m3_fnuz", "fp8fnuz", "fp8_fnuz")
    )
except AttributeError:
    pass

try:
    _DTYPE_REGISTRY.append((torch.float8_e4m3fn, "fp8_e4m3", "fp8ocp", "fp8_ocp"))
except AttributeError:
    pass

try:
    _DTYPE_REGISTRY.append((torch.float4_e2m1fn_x2, "fp4", "fp4", "fp4"))
except AttributeError:
    pass

DTYPE_TO_CONFIG_STR: dict[torch.dtype, str] = {r[0]: r[1] for r in _DTYPE_REGISTRY}
CONFIG_STR_TO_DTYPE: dict[str, torch.dtype] = {r[1]: r[0] for r in _DTYPE_REGISTRY}
CONFIG_STR_TO_SHORT_NAME: dict[str, str] = {r[1]: r[2] for r in _DTYPE_REGISTRY}

_KERNEL_TYPE_NAMES = frozenset(
    {"IntraNode", "InterNode", "InterNodeV1", "InterNodeV1LL", "AsyncLL"}
)

_QUANT_TYPE_CONFIG_STRS = {"none", "fp8_direct_cast"}


def kernel_type_to_config_str(kernel_type) -> str:
    """Normalize kernel_type (str or enum) to a config string like 'IntraNode'."""
    name = getattr(kernel_type, "name", None) or str(kernel_type)
    if name in _KERNEL_TYPE_NAMES:
        return name
    raise ValueError(
        f"Unknown kernel_type: {kernel_type!r}, "
        f"expected one of {sorted(_KERNEL_TYPE_NAMES)}"
    )


def quant_type_to_config_str(quant_type) -> str:
    """Normalize quant_type (str or enum) to 'none' or 'fp8_direct_cast'."""
    if isinstance(quant_type, str):
        s = quant_type.strip().lower()
        if s in _QUANT_TYPE_CONFIG_STRS:
            return s
        if s == "none_":
            return "none"
    else:
        name = getattr(quant_type, "name", str(quant_type))
        if name == "None_":
            return "none"
        if name == "Fp8DirectCast":
            return "fp8_direct_cast"
    raise ValueError(f"Unknown quant_type: {quant_type!r}")


def dtype_to_config_str(dtype: torch.dtype) -> str:
    """Convert a torch dtype to its config string representation."""
    return DTYPE_TO_CONFIG_STR[dtype]


# ---------------------------------------------------------------------------
# GPU model detection
# ---------------------------------------------------------------------------

_SUPPORTED_VERSION = "1.0"
_TUNING_CONFIGS_DIR = Path(__file__).parent / "tuning_configs"

_gpu_model_cache: str | None = None
_gpu_model_detected: bool = False


def detect_gpu_model() -> str | None:
    """Detect GPU model from device name, e.g. 'mi300x', 'mi308x'."""
    global _gpu_model_cache, _gpu_model_detected
    if _gpu_model_detected:
        return _gpu_model_cache
    _gpu_model_detected = True
    try:
        name = torch.cuda.get_device_properties(0).name.lower()
    except Exception:
        return None
    import re

    m = re.search(r"\bmi\d+\w*", name)
    if m:
        _gpu_model_cache = m.group(0)
    return _gpu_model_cache


# ---------------------------------------------------------------------------
# Rule validation — dispatch and combine have different required fields
# ---------------------------------------------------------------------------

_DISPATCH_RULE_REQUIRED = frozenset(
    {
        "dtype",
        "num_tokens",
        "hidden_dim",
        "block_num",
        "rdma_block_num",
        "warp_per_block",
    }
)

_COMBINE_RULE_REQUIRED = frozenset(
    {
        "dtype",
        "num_tokens",
        "hidden_dim",
        "zero_copy",
        "quant_type",
        "block_num",
        "rdma_block_num",
        "warp_per_block",
    }
)

# ---------------------------------------------------------------------------
# File naming — files split by phase (dispatch / combine), no quant in name
# ---------------------------------------------------------------------------


def build_config_filename(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    gpu_model: str | None = None,
    phase: str = "dispatch",
) -> str:
    """Build config filename: ``{arch}_{model}_{kernel}_ep{n}_{phase}.json``."""
    parts = [gpu_arch]
    if gpu_model:
        parts.append(gpu_model)
    parts.append(f"{kernel_type}_ep{ep_size}_{phase}")
    return "_".join(parts) + ".json"


def _find_fallback_config(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    phase: str,
) -> Path | None:
    """Find any config file matching the same (arch, kernel, ep, phase)
    but different gpu_model."""
    suffix = f"_{kernel_type}_ep{ep_size}_{phase}.json"
    prefix = f"{gpu_arch}_"
    if not _TUNING_CONFIGS_DIR.is_dir():
        return None
    for f in sorted(_TUNING_CONFIGS_DIR.iterdir()):
        if f.name.startswith(prefix) and f.name.endswith(suffix) and f.is_file():
            return f
    return None


def config_path_for(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    gpu_model: str | None = None,
    phase: str = "dispatch",
) -> Path:
    """Resolve config file path for a given phase.

    Priority:
    1. MORI_EP_TUNING_CONFIG env var (exact path override, ignores phase)
    2. Exact match with gpu_model
    3. Fallback: any file with same (arch, kernel, ep, phase) but different model
    """
    env_override = os.environ.get("MORI_EP_TUNING_CONFIG")
    if env_override:
        return Path(env_override)
    exact = _TUNING_CONFIGS_DIR / build_config_filename(
        gpu_arch,
        kernel_type,
        ep_size,
        gpu_model,
        phase,
    )
    if exact.is_file():
        return exact
    fallback = _find_fallback_config(gpu_arch, kernel_type, ep_size, phase)
    if fallback is not None:
        logger.info(
            "No tuning config for gpu_model=%s phase=%s, using fallback: %s",
            gpu_model,
            phase,
            fallback.name,
        )
        return fallback
    return exact


# ---------------------------------------------------------------------------
# LaunchParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchParams:
    block_num: int
    rdma_block_num: int
    warp_per_block: int


# ---------------------------------------------------------------------------
# TuningConfigManager
# ---------------------------------------------------------------------------


class TuningConfigManager:
    """Loads, caches, and queries tuning configurations.

    Each instance holds dispatch_rules and combine_rules loaded from
    two separate JSON files (one per phase).
    """

    _cache: ClassVar[dict[str, "TuningConfigManager"]] = {}

    def __init__(self, dispatch_rules: list[dict], combine_rules: list[dict]):
        self.dispatch_rules: list[dict] = dispatch_rules
        self.combine_rules: list[dict] = combine_rules

    @classmethod
    def get_instance(
        cls,
        gpu_arch: str,
        kernel_type: str,
        ep_size: int,
        gpu_model: str | None = None,
    ) -> "TuningConfigManager":
        cache_key = f"{gpu_arch}_{gpu_model or ''}_{kernel_type}_ep{ep_size}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        dispatch_path = config_path_for(
            gpu_arch,
            kernel_type,
            ep_size,
            gpu_model,
            "dispatch",
        )
        combine_path = config_path_for(
            gpu_arch,
            kernel_type,
            ep_size,
            gpu_model,
            "combine",
        )
        dispatch_rules = cls._load_phase_rules(
            dispatch_path,
            gpu_arch,
            "dispatch",
            _DISPATCH_RULE_REQUIRED,
        )
        combine_rules = cls._load_phase_rules(
            combine_path,
            gpu_arch,
            "combine",
            _COMBINE_RULE_REQUIRED,
        )
        instance = cls(dispatch_rules, combine_rules)
        cls._cache[cache_key] = instance
        return instance

    @classmethod
    def _load_phase_rules(
        cls,
        path: Path,
        expected_gpu_arch: str,
        phase: str,
        required_fields: frozenset[str],
    ) -> list[dict]:
        if not path.is_file():
            logger.debug("Tuning config not found: %s", path)
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load tuning config %s: %s", path, exc)
            return []

        version = data.get("version")
        if version != _SUPPORTED_VERSION:
            logger.warning(
                "Unsupported tuning config version %r in %s (expected %s)",
                version,
                path,
                _SUPPORTED_VERSION,
            )
            return []

        file_gpu_arch = data.get("gpu_arch")
        if file_gpu_arch and file_gpu_arch != expected_gpu_arch:
            logger.warning(
                "GPU arch mismatch in %s: file says %s, detected %s",
                path,
                file_gpu_arch,
                expected_gpu_arch,
            )

        raw_rules = data.get("rules", [])
        valid: list[dict] = []
        for i, rule in enumerate(raw_rules):
            missing = required_fields - rule.keys()
            if missing:
                logger.warning(
                    "Skipping %s rule %d in %s: missing fields %s",
                    phase,
                    i,
                    path,
                    missing,
                )
                continue
            valid.append(rule)

        if phase == "dispatch":
            valid.sort(key=lambda r: (r["dtype"], r["hidden_dim"], r["num_tokens"]))
        else:
            valid.sort(
                key=lambda r: (
                    r["dtype"],
                    r["hidden_dim"],
                    r["zero_copy"],
                    r["quant_type"],
                    r["num_tokens"],
                )
            )
        logger.debug("Loaded %d %s rules from %s", len(valid), phase, path)
        return valid

    # ------------------------------------------------------------------
    # Runtime lookup
    # ------------------------------------------------------------------

    @staticmethod
    def lookup(
        sorted_rules: list[dict],
        dtype: torch.dtype,
        num_tokens: int,
        hidden_dim: int,
        zero_copy: bool | None = None,
        quant_type: str | None = None,
    ) -> LaunchParams | None:
        """Find the best matching launch params.

        Exact-matches dtype, hidden_dim, and (for combine) zero_copy and
        quant_type.  Then picks the tightest (smallest) rule num_tokens
        that is >= actual num_tokens (ceiling match).
        """
        dtype_str = DTYPE_TO_CONFIG_STR.get(dtype)
        if dtype_str is None:
            return None
        for rule in sorted_rules:
            if rule["dtype"] != dtype_str:
                continue
            if rule["hidden_dim"] != hidden_dim:
                continue
            if zero_copy is not None and rule.get("zero_copy") != zero_copy:
                continue
            if quant_type is not None and rule.get("quant_type") != quant_type:
                continue
            if num_tokens <= rule["num_tokens"]:
                return LaunchParams(
                    block_num=rule["block_num"],
                    rdma_block_num=rule["rdma_block_num"],
                    warp_per_block=rule["warp_per_block"],
                )
        return None

    # ------------------------------------------------------------------
    # Tuning result persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_tuning_result(
        path: str | Path,
        metadata: dict,
        entry: dict,
        phase: str = "dispatch",
    ) -> None:
        """Save or merge a single tuning rule into a phase JSON file.

        entry format for dispatch:
            {"dtype": "fp4", "num_tokens": 128, "hidden_dim": 3584,
             "block_num": 64, "rdma_block_num": 0, "warp_per_block": 16,
             "bandwidth_gbps": 41.5}

        entry format for combine (adds zero_copy + quant_type):
            {"dtype": "bf16", "num_tokens": 128, "hidden_dim": 7168,
             "zero_copy": true, "quant_type": "none",
             "block_num": 64, "rdma_block_num": 0, "warp_per_block": 16,
             "bandwidth_gbps": 97.75}
        """
        path = Path(path)

        if path.is_file():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = None
        else:
            data = None

        if data is None:
            data = {"version": _SUPPORTED_VERSION, **metadata, "rules": []}

        rules: list[dict] = data.setdefault("rules", [])

        def _dispatch_key(r):
            return (r.get("dtype"), r.get("num_tokens"), r.get("hidden_dim"))

        def _combine_key(r):
            return (
                r.get("dtype"),
                r.get("num_tokens"),
                r.get("hidden_dim"),
                r.get("zero_copy"),
                r.get("quant_type"),
            )

        def _dispatch_sort(r):
            return (r["dtype"], r["hidden_dim"], r["num_tokens"])

        def _combine_sort(r):
            return (
                r["dtype"],
                r["hidden_dim"],
                r.get("zero_copy", False),
                r.get("quant_type", ""),
                r["num_tokens"],
            )

        if phase == "dispatch":
            merge_key = _dispatch_key(entry)
            match_fn = _dispatch_key
            sort_key = _dispatch_sort
        else:
            merge_key = _combine_key(entry)
            match_fn = _combine_key
            sort_key = _combine_sort

        matched_idx = None
        for i, rule in enumerate(rules):
            if match_fn(rule) == merge_key:
                matched_idx = i
                break

        if matched_idx is not None:
            old_bw = rules[matched_idx].get("bandwidth_gbps", 0)
            new_bw = entry.get("bandwidth_gbps", 0)
            if new_bw > old_bw:
                rules[matched_idx] = entry
                logger.info(
                    "Updated %s rule for %s (%.2f -> %.2f GB/s)",
                    phase,
                    merge_key,
                    old_bw,
                    new_bw,
                )
            else:
                logger.info(
                    "Kept existing %s rule for %s (existing %.2f >= new %.2f GB/s)",
                    phase,
                    merge_key,
                    old_bw,
                    new_bw,
                )
        else:
            rules.append(entry)
            logger.info("Added %s rule: %s", phase, merge_key)

        rules.sort(key=sort_key)
        data["rules"] = rules

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=path.stem
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
